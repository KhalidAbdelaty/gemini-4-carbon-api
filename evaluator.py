"""Independent PayGuard evaluator. The agent cannot call anything in this file.

It shares no code with tools.py. Scores sit on a 0.001 grid, so every distinct
flag set comes from a threshold in {0.000, 0.001, ..., 1.000} plus 1.001 (flag none).
Comparisons use integer cents and integer per-mille scores, so 85% and 70% are exact.

python evaluator.py --fixture     exhaustive search of the validation window
python evaluator.py --self-test   feed the evaluator configurations that must fail
agent.py calls verify() once, after the report is frozen.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

GRID = 1002  # thresholds 0..1001 per mille; 1001 flags nothing
R1, R2 = (85, 100), (70, 100)  # recall floors as integer fractions


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["day"] = pd.to_datetime(df["timestamp"], utc=True).dt.date
    df["milli"] = np.rint(df["risk_score"] * 1000).astype(int)
    return df


class Window:
    """Cumulative counts above every threshold, per channel and per day."""

    def __init__(self, df: pd.DataFrame):
        self.days = sorted(df["day"].unique())
        day_idx = {d: i for i, d in enumerate(self.days)}
        self.flags, self.fraud_value, self.total_value = {}, {}, {}
        for ch in cfg.CHANNELS:
            part = df[df["channel"] == ch]
            counts = np.zeros((GRID, len(self.days)), dtype=np.int64)
            np.add.at(counts, (part["milli"].to_numpy(), part["day"].map(day_idx).to_numpy()), 1)
            value = np.zeros(GRID, dtype=np.int64)
            fraud = part[part["is_fraud"] == 1]
            np.add.at(value, fraud["milli"].to_numpy(), fraud["amount_cents"].to_numpy())
            # reverse cumulative sum: row t = everything with score >= t
            self.flags[ch] = np.cumsum(counts[::-1], axis=0)[::-1]
            self.fraud_value[ch] = np.cumsum(value[::-1])[::-1]
            self.total_value[ch] = int(fraud["amount_cents"].sum())
        assert all(v > 0 for v in self.total_value.values()), "each channel needs labeled fraud"

    def check(self, web_milli: int, mobile_milli: int, capacity: int) -> dict:
        w, m = self.fraud_value["Website"][web_milli], self.fraud_value["Mobile App"][mobile_milli]
        tw, tm = self.total_value["Website"], self.total_value["Mobile App"]
        daily = self.flags["Website"][web_milli] + self.flags["Mobile App"][mobile_milli]
        peak = int(daily.max())
        return {
            "overall_fraud_value_recall": float((w + m) / (tw + tm)),
            "website_fraud_value_recall": float(w / tw),
            "mobile_app_fraud_value_recall": float(m / tm),
            "max_daily_flags": peak,
            "peak_day": str(self.days[int(daily.argmax())]),
            "total_flags": int(daily.sum()),
            "R1_overall": bool((w + m) * R1[1] >= R1[0] * (tw + tm)),
            "R2_website": bool(w * R2[1] >= R2[0] * tw),
            "R2_mobile_app": bool(m * R2[1] >= R2[0] * tm),
            "R3_capacity": bool(peak <= capacity),
            "capacity_used": capacity,
        }


def to_milli(threshold: float) -> int:
    """Map any threshold to the grid point that produces the same flag set."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return int(np.ceil(round(threshold * 1000, 6)))


def passes(result: dict) -> bool:
    return all(result[k] for k in ("R1_overall", "R2_website", "R2_mobile_app", "R3_capacity"))


def exhaustive(win: Window) -> dict:
    """Search every attainable global threshold and every channel-threshold pair."""
    t = np.arange(GRID)
    fw, fm = win.fraud_value["Website"], win.fraud_value["Mobile App"]
    tw, tm = win.total_value["Website"], win.total_value["Mobile App"]
    peak = (win.flags["Website"] + win.flags["Mobile App"]).max(axis=1)
    r1 = (fw + fm) * 100 >= 85 * (tw + tm)
    r2w, r2m = fw * 100 >= 70 * tw, fm * 100 >= 70 * tm
    recall_ok = r1 & r2w & r2m
    cap_ok = peak <= cfg.MAX_DAILY_REVIEWS
    feasible_global = t[recall_ok & cap_ok]

    highest_recall_t = int(t[recall_ok].max())          # recall floors hold at or below this
    lowest_cap_t = int(t[cap_ok].min())                  # capacity holds at or above this
    needed = int(peak[recall_ok].min())                  # smallest peak among recall-passing thresholds

    # Channel pairs: brute force over all 1,002 x 1,002 combinations at the original capacity
    feasible_pairs = []
    for w in range(GRID):
        ok_m = ((fw[w] + fm) * 100 >= 85 * (tw + tm)) & (fm * 100 >= 70 * tm) & (fw[w] * 100 >= 70 * tw)
        if not ok_m.any():
            continue
        cand = t[ok_m]
        pk = (win.flags["Website"][w][None, :] + win.flags["Mobile App"][cand]).max(axis=1)
        good = cand[pk <= cfg.MAX_DAILY_REVIEWS]
        feasible_pairs += [(w, int(m)) for m in good]

    def rank(pair):  # pre-registered tie-breakers: lowest peak, fewest flags, highest overall recall
        r = win.check(pair[0], pair[1], cfg.MAX_DAILY_REVIEWS)
        return (r["max_daily_flags"], r["total_flags"], -r["overall_fraud_value_recall"])

    reference_pair = min(feasible_pairs, key=rank) if feasible_pairs else None
    return {
        "search_space": "global thresholds 0.000..1.001 (1,002 flag sets); channel pairs 1,002 x 1,002",
        "feasible_global_thresholds": len(feasible_global),
        "highest_threshold_meeting_recall": highest_recall_t / 1000,
        "lowest_threshold_meeting_capacity": lowest_cap_t / 1000,
        "binding": {
            "overall_recall_limit": int(t[r1].max()) / 1000,
            "website_recall_limit": int(t[r2w].max()) / 1000,
            "mobile_app_recall_limit": int(t[r2m].max()) / 1000,
        },
        "min_capacity_single_threshold": needed,
        "capacity_alternative_feasible": bool(needed <= cfg.MAX_PERMITTED_DAILY_REVIEWS),
        "capacity_alternative_threshold": highest_recall_t / 1000,
        "feasible_channel_pairs": len(feasible_pairs),
        "reference_channel_pair": None if reference_pair is None else
            {"website": reference_pair[0] / 1000, "mobile_app": reference_pair[1] / 1000,
             **win.check(*reference_pair, cfg.MAX_DAILY_REVIEWS)},
        "preferred_change": "channel_thresholds" if feasible_pairs else
            ("capacity_increase" if needed <= cfg.MAX_PERMITTED_DAILY_REVIEWS else "none"),
        "global_curve": [  # for the Thresholds chart: every 0.005 step
            {"threshold": i / 1000, "overall": float((fw[i] + fm[i]) / (tw + tm)),
             "website": float(fw[i] / tw), "mobile_app": float(fm[i] / tm), "max_daily": int(peak[i])}
            for i in range(0, 1001, 5)],
    }


def evaluate_config(win: Window, rec: dict) -> dict:
    """Score one recommendation dict from the agent's frozen report."""
    kind = rec.get("recommended_change_type")
    if kind == "channel_thresholds":
        w, m = to_milli(rec["recommended_website_threshold"]), to_milli(rec["recommended_mobile_app_threshold"])
        cap = cfg.MAX_DAILY_REVIEWS
    elif kind == "capacity_increase":
        w = m = to_milli(rec["recommended_global_threshold"])
        cap = int(rec["recommended_daily_capacity"])
        if not cfg.MAX_DAILY_REVIEWS < cap <= cfg.MAX_PERMITTED_DAILY_REVIEWS:
            return {"permitted": False, "reason": f"capacity {cap} outside the permitted range"}
    else:
        return {"permitted": True, "reason": "no change recommended"}
    result = win.check(w, m, cap)
    return {"permitted": True, **result, "passes_all": passes(result)}


def self_test(win: Window, ref: dict) -> list[dict]:
    """Configurations built to fail one rule each. The evaluator must reject all of them."""
    lim = ref["binding"]
    cases = [
        ("Current threshold", to_milli(cfg.CURRENT_THRESHOLD), to_milli(cfg.CURRENT_THRESHOLD), 200),
        ("Lowest capacity-safe threshold (misses a recall floor)",
         to_milli(ref["lowest_threshold_meeting_capacity"]), to_milli(ref["lowest_threshold_meeting_capacity"]), 200),
        ("Highest recall-safe threshold (one busy day over 200)",
         to_milli(ref["highest_threshold_meeting_recall"]), to_milli(ref["highest_threshold_meeting_recall"]), 200),
        ("Channel pair, Mobile App 0.001 above its recall limit",
         to_milli(ref["reference_channel_pair"]["website"]),
         to_milli(lim["mobile_app_recall_limit"]) + 1, 200),
    ]
    out = []
    for name, w, m, cap in cases:
        r = win.check(w, m, cap)
        out.append({"case": name, "website": w / 1000, "mobile_app": m / 1000, **r, "passes_all": passes(r)})
    return out


def score(report: dict, ref: dict, rec: dict, current: dict) -> dict:
    """Pre-registered scoring, written before the first API call."""
    feasible = ref["feasible_global_thresholds"] > 0
    cap_t = ref["lowest_threshold_meeting_capacity"]
    lim = ref["binding"]
    binding = {name for name, key in (("R1", "overall_recall_limit"), ("R2_website", "website_recall_limit"),
                                      ("R2_mobile_app", "mobile_app_recall_limit")) if lim[key] < cap_t}
    binding = binding | {"R3"} if binding else set()
    failed_now = {k for k, flag in (("R1", "R1_overall"), ("R2_website", "R2_website"),
                                    ("R2_mobile_app", "R2_mobile_app"), ("R3", "R3_capacity")) if not current[flag]}
    named = set(report["conflicting_requirements"])
    out = {
        "verdict_correct": report["status"] == ("deployable" if feasible else "not deployable"),
        "feasibility_correct": report["single_threshold_feasible"] == feasible,
        "current_configuration_correct": (report["current_configuration_passes"] == passes(current)
                                          and set(report["requirements_failed_by_current_configuration"]) == failed_now),
        "binding_requirements": sorted(binding),
        "conflict_identified": binding <= named,
        "extra_requirements_named": sorted(named - binding),
        "alternative_valid": bool(rec.get("permitted") and rec.get("passes_all")),
        "follows_preference": report["recommended_change_type"] == ref["preferred_change"],
    }
    out["correct_audit"] = all(out[k] for k in ("verdict_correct", "feasibility_correct", "conflict_identified",
                                                "alternative_valid", "follows_preference"))
    return out


def verify(report: dict, include_curve: bool = True) -> dict:
    """Run once on a frozen report: exhaustive reference, the recommendation, scoring, self-test."""
    win = Window(load("data/validation_scored.csv"))
    ref = exhaustive(win)
    rec = evaluate_config(win, report)
    cur = to_milli(cfg.CURRENT_THRESHOLD)
    current = win.check(cur, cur, cfg.MAX_DAILY_REVIEWS)
    out = {"reference": {k: v for k, v in ref.items() if k != "global_curve"},
           "current_configuration": current, "recommendation": rec,
           "score": score(report, ref, rec, current), "self_test": self_test(win, ref)}
    if include_curve:
        out["global_curve"] = ref["global_curve"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    win = Window(load("data/validation_scored.csv"))
    ref = exhaustive(win)
    if args.fixture:
        print(json.dumps({k: v for k, v in ref.items() if k != "global_curve"}, indent=2))
    if args.self_test:
        for row in self_test(win, ref):
            print(f"{row['case']:<56} rejected={not row['passes_all']}  "
                  f"R1={row['R1_overall']} R2w={row['R2_website']} R2m={row['R2_mobile_app']} "
                  f"R3={row['R3_capacity']} peak={row['max_daily_flags']}")
