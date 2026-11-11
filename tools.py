"""The three read-only tools Gemini 4 Carbon may request. They read the validation window only.

Each tool returns plain numbers. None of them says whether a requirement passes;
comparing the numbers with the policy is the agent's job.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import config as cfg

VALIDATION_FILE = Path("data/validation_scored.csv")
SUMMARY_FILE = Path("data/model_summary.json")

DECLARATIONS = [
    {
        "type": "function",
        "name": "get_audit_context",
        "description": ("Return the classifier summary, the validation window, PayGuard's current "
                        "configuration, the fixed deployment policy, and the permitted changes. Read-only."),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "evaluate_global_threshold",
        "description": ("Apply one threshold to both channels on the validation window and return "
                        "fraud-value recall overall and by channel, daily flag counts, and diagnostics. "
                        "A transaction is flagged when risk_score >= threshold. Read-only."),
        "parameters": {
            "type": "object",
            "properties": {"threshold": {"type": "number", "description": "Between 0 and 1."}},
            "required": ["threshold"],
        },
    },
    {
        "type": "function",
        "name": "evaluate_policy_change",
        "description": (
            "Evaluate one permitted policy change on the validation window. "
            "change_type 'capacity_increase' needs threshold and daily_capacity (201 to "
            f"{cfg.MAX_PERMITTED_DAILY_REVIEWS}). change_type 'channel_thresholds' needs "
            "website_threshold and mobile_app_threshold and keeps the original daily capacity. "
            "The two changes cannot be combined. Returns the same measurements as "
            "evaluate_global_threshold. Read-only; it approves nothing."),
        "parameters": {
            "type": "object",
            "properties": {
                "change_type": {"type": "string", "enum": ["capacity_increase", "channel_thresholds"]},
                "threshold": {"type": "number"},
                "daily_capacity": {"type": "integer"},
                "website_threshold": {"type": "number"},
                "mobile_app_threshold": {"type": "number"},
            },
            "required": ["change_type"],
        },
    },
]


class ToolError(ValueError):
    """Raised for an invalid argument. The message goes back to the model."""


def _threshold(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ToolError(f"{name} must be a number")
    if not 0.0 <= value <= 1.0:
        raise ToolError(f"{name} must be between 0 and 1, got {value}")
    return float(value)


def _dollars(cents) -> float:
    return round(int(cents) / 100, 2)


class AuditTools:
    def __init__(self):
        df = pd.read_csv(VALIDATION_FILE)
        df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%d")
        self.df = df
        self.dates = sorted(df["date"].unique())
        self.summary = json.loads(SUMMARY_FILE.read_text())

    # Tool 1 ---------------------------------------------------------------
    def get_audit_context(self) -> dict:
        df, by_channel = self.df, {}
        for ch in cfg.CHANNELS:
            part = df[df["channel"] == ch]
            fraud = part[part["is_fraud"] == 1]
            by_channel[ch] = {"transactions": len(part), "labeled_fraud": len(fraud),
                              "labeled_fraud_value_usd": _dollars(fraud["amount_cents"].sum())}
        daily = df.groupby("date").size()
        return {
            "classifier": self.summary,
            "validation_window": {
                "dates": f"{self.dates[0]} to {self.dates[-1]} (UTC calendar days)",
                "days": len(self.dates),
                "transactions": len(df),
                "transactions_per_day": {"min": int(daily.min()), "mean": round(float(daily.mean()), 1),
                                         "max": int(daily.max())},
                "by_channel": by_channel,
                "average_precision": round(float(average_precision_score(df["is_fraud"], df["risk_score"])), 3),
            },
            "risk_score": "Rounded to 0.001. A threshold between grid points flags the same rows as the next grid point up.",
            "current_configuration": {"threshold": cfg.CURRENT_THRESHOLD, "rule": "flag when risk_score >= threshold",
                                      "daily_review_capacity": cfg.MAX_DAILY_REVIEWS},
            "policy": cfg.POLICY_TEXT,
            "holdout": "A later holdout window exists. It is sealed, and no tool can read it.",
        }

    # Shared measurement ----------------------------------------------------
    def _measure(self, web_t: float, mobile_t: float) -> dict:
        df = self.df
        cut = np.where(df["channel"] == "Website", web_t, mobile_t)
        flagged = df["risk_score"].to_numpy() >= cut
        fraud = df["is_fraud"].to_numpy() == 1
        amount = df["amount_cents"].to_numpy()

        recall = {}
        for ch, key in (("Website", "website"), ("Mobile App", "mobile_app")):
            m = (df["channel"] == ch).to_numpy() & fraud
            recall[key] = {"flagged_fraud_value_usd": _dollars(amount[m & flagged].sum()),
                           "labeled_fraud_value_usd": _dollars(amount[m].sum()),
                           "fraud_value_recall": round(amount[m & flagged].sum() / amount[m].sum(), 4)}
        overall = round(amount[fraud & flagged].sum() / amount[fraud].sum(), 4)

        counts = pd.Series(flagged, index=df["date"]).groupby(level=0).sum().reindex(self.dates, fill_value=0)
        tp, fp = int((flagged & fraud).sum()), int((flagged & ~fraud).sum())
        return {
            "overall_fraud_value_recall": overall,
            "by_channel": recall,
            "flagged_transactions": int(flagged.sum()),
            "daily_flags": {"max": int(counts.max()), "max_date": str(counts.idxmax()),
                            "mean": round(float(counts.mean()), 1), "min": int(counts.min()),
                            "by_date": {d: int(c) for d, c in counts.items()}},
            "diagnostics": {"precision": round(tp / max(tp + fp, 1), 4),
                            "transaction_recall": round(tp / int(fraud.sum()), 4),
                            "true_positives": tp, "false_positives": fp},
        }

    # Tool 2 ---------------------------------------------------------------
    def evaluate_global_threshold(self, threshold=None) -> dict:
        t = _threshold(threshold, "threshold")
        return {"threshold": t, **self._measure(t, t)}

    # Tool 3 ---------------------------------------------------------------
    def evaluate_policy_change(self, change_type=None, threshold=None, daily_capacity=None,
                               website_threshold=None, mobile_app_threshold=None) -> dict:
        if change_type == "capacity_increase":
            if website_threshold is not None or mobile_app_threshold is not None:
                raise ToolError("capacity_increase uses one global threshold; the two changes cannot be combined")
            t = _threshold(threshold, "threshold")
            if isinstance(daily_capacity, float) and daily_capacity.is_integer():
                daily_capacity = int(daily_capacity)
            if isinstance(daily_capacity, bool) or not isinstance(daily_capacity, int):
                raise ToolError("daily_capacity must be an integer")
            if not cfg.MAX_DAILY_REVIEWS < daily_capacity <= cfg.MAX_PERMITTED_DAILY_REVIEWS:
                raise ToolError(f"daily_capacity must be between {cfg.MAX_DAILY_REVIEWS + 1} and "
                                f"{cfg.MAX_PERMITTED_DAILY_REVIEWS}, got {daily_capacity}")
            return {"change_type": change_type, "threshold": t, "daily_capacity": daily_capacity,
                    **self._measure(t, t)}
        if change_type == "channel_thresholds":
            if threshold is not None or daily_capacity is not None:
                raise ToolError("channel_thresholds keeps the original daily capacity; the two changes cannot be combined")
            w = _threshold(website_threshold, "website_threshold")
            m = _threshold(mobile_app_threshold, "mobile_app_threshold")
            return {"change_type": change_type, "website_threshold": w, "mobile_app_threshold": m,
                    "daily_capacity": cfg.MAX_DAILY_REVIEWS, **self._measure(w, m)}
        raise ToolError("change_type must be 'capacity_increase' or 'channel_thresholds'")

    def call(self, name: str, arguments: dict) -> tuple[dict, bool]:
        """Run one allowlisted tool. Returns (result, is_error)."""
        fn = {"get_audit_context": self.get_audit_context,
              "evaluate_global_threshold": self.evaluate_global_threshold,
              "evaluate_policy_change": self.evaluate_policy_change}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}, True
        try:
            return fn(**(arguments or {})), False
        except TypeError as exc:  # unexpected argument names
            return {"error": f"invalid arguments: {exc}"}, True
        except ToolError as exc:
            return {"error": str(exc)}, True
