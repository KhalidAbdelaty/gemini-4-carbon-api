"""PayGuard audit agent: Gemini 4 Carbon on the Interactions API with three read-only tools.

python agent.py                    full experiment: medium audit, evaluator, holdout, clean high rerun
python agent.py --level high --attempt-only   one extra attempt, saved under runs/attempts/
"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from dotenv import dotenv_values
from pydantic import BaseModel, Field

import config as cfg
from tools import DECLARATIONS, AuditTools

RUNS = Path("runs")

SYSTEM_INSTRUCTION = """You are auditing whether PayGuard's fraud classifier can go live under a fixed
deployment policy. Python tools do every calculation. You decide which checks to run and what
the results show.

How to work:
- Read the audit context first, then evaluate the current configuration before anything else.
- Check every requirement yourself by comparing tool numbers with the policy text: R1, R2 for
  Website, R2 for Mobile App, and R3. R3 is a limit on the busiest single day, not on the average.
- Use only numbers that a tool returned. Do not estimate a metric you have not measured.
- You may investigate only the permitted changes listed in the policy. You cannot approve a
  change, rewrite a requirement, retrain the model, or read the holdout window.
- If you recommend a change, follow the policy's preference rule and test the exact
  configuration you recommend.
- If the evidence does not settle a question, say so. Inconclusive is an acceptable answer.
- When you have finished investigating, reply with a short plain-text summary and no tool call."""

TASK = ("Audit PayGuard's fraud classifier for deployment. Decide whether the model can be deployed "
        "under the original policy, explain which requirements conflict if it cannot, and investigate "
        "whether a permitted policy change would work.")

REPORT_PROMPT = """Write the final audit report as JSON that matches the schema.
- status: "deployable" if at least one single global threshold meets R1, R2 (both channels),
  and R3 under the original policy; "not deployable" if none does; "inconclusive" if your
  evidence cannot decide.
- Cite tool calls by the call number shown in each tool result.
- Every number in the report must come from a tool result.
- Leave recommended_* fields null when they do not apply."""

Requirement = Literal["R1", "R2_website", "R2_mobile_app", "R3"]


class Evidence(BaseModel):
    tool_call: int = Field(description="Call number from the tool result.")
    finding: str


class AuditReport(BaseModel):
    status: Literal["deployable", "not deployable", "inconclusive"]
    current_configuration_passes: bool
    requirements_failed_by_current_configuration: list[Requirement]
    single_threshold_feasible: bool
    conflicting_requirements: list[Requirement]
    evidence: list[Evidence]
    recommended_change_type: Literal["none", "capacity_increase", "channel_thresholds"]
    recommended_global_threshold: float | None
    recommended_daily_capacity: int | None
    recommended_website_threshold: float | None
    recommended_mobile_app_threshold: float | None
    required_approvals: list[str]
    unresolved_issues: list[str]
    summary: str


def make_client():
    """Load only GEMINI_API_KEY from this project's .env. GOOGLE_API_KEY would take precedence."""
    key = dotenv_values(".env").get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ["GEMINI_API_KEY"] = key
    from google import genai
    return genai.Client()


def rates_for(day) -> dict:
    intro = cfg.PRICES["introductory"]
    return intro if day <= intro["until"] else cfg.PRICES["standard"]


def turn_cost(u: dict, rates: dict) -> float:
    """total_input_tokens includes cached tokens; thinking is billed at the output rate."""
    uncached = u["input"] - u["cached"]
    return (uncached * rates["input"] + u["cached"] * rates["cached"]
            + (u["output"] + u["thought"]) * rates["output"]) / 1_000_000


def _usage(interaction) -> dict:
    u = interaction.usage
    return {"input": u.total_input_tokens or 0, "cached": u.total_cached_tokens or 0,
            "output": u.total_output_tokens or 0, "thought": u.total_thought_tokens or 0,
            "tool_use": u.total_tool_use_tokens or 0}


def _summaries(interaction) -> list[str]:
    out = []
    for step in interaction.steps or []:
        if step.type == "thought" and getattr(step, "summary", None):
            out += [c.text for c in step.summary if getattr(c, "text", None)]
    return out


def run_audit(level: str, log: Callable[[str], None] = print) -> dict:
    """One attempt from a clean state. Returns the attempt record with a frozen report."""
    client, tools = make_client(), AuditTools()
    request = dict(
        model=cfg.MODEL_ID,
        system_instruction=SYSTEM_INSTRUCTION,
        tools=DECLARATIONS,
        generation_config={"thinking_level": level, "max_output_tokens": cfg.MAX_OUTPUT_TOKENS,
                           "thinking_summaries": "auto"},
    )
    started = datetime.now(timezone.utc)
    rates = rates_for(started.date())
    t0 = time.perf_counter()
    turns, calls, stop_reason, spend = [], [], None, 0.0

    def record_turn(interaction, seconds, kind):
        nonlocal spend
        u = _usage(interaction)
        spend += turn_cost(u, rates)
        turns.append({"kind": kind, "status": interaction.status, "interaction_id": interaction.id,
                      "seconds": round(seconds, 2), "usage": u, "thought_summaries": _summaries(interaction)})
        log(f"turn {len(turns)}: {interaction.status}, {seconds:.1f}s, "
            f"thought={u['thought']} output={u['output']} input={u['input']} cached={u['cached']}")

    def create(**kwargs):
        t = time.perf_counter()
        interaction = client.interactions.create(timeout=cfg.HTTP_TIMEOUT_SECONDS, **kwargs)
        return interaction, time.perf_counter() - t

    try:
        interaction, secs = create(input=TASK, **request)
        record_turn(interaction, secs, "investigate")
        while True:
            fcs = [s for s in interaction.steps or [] if s.type == "function_call"]
            if interaction.status in ("failed", "cancelled", "incomplete"):
                stop_reason = f"interaction {interaction.status}"
                break
            if not fcs:
                stop_reason = "agent finished"
                break
            results = []
            for fc in fcs:
                if len(calls) >= cfg.MAX_TOOL_CALLS:
                    out, is_error = {"error": "tool-call limit reached; finish with what you have"}, True
                else:
                    out, is_error = tools.call(fc.name, dict(fc.arguments or {}))
                calls.append({"n": len(calls) + 1, "turn": len(turns), "name": fc.name,
                              "arguments": dict(fc.arguments or {}), "is_error": is_error, "result": out})
                out = {"call_number": len(calls), **out}
                log(f"  call {len(calls)}: {fc.name}({json.dumps(dict(fc.arguments or {}))})"
                    + (f" -> error: {out['error']}" if is_error else ""))
                results.append({"type": "function_result", "name": fc.name, "call_id": fc.id,
                                "is_error": is_error, "result": [{"type": "text", "text": json.dumps(out)}]})
            # Application limits: checked before every new request
            elapsed = time.perf_counter() - t0
            if len(turns) >= cfg.MAX_TURNS:
                stop_reason = "turn limit"
            elif len(calls) > cfg.MAX_TOOL_CALLS:
                stop_reason = "tool-call limit"
            elif elapsed >= cfg.MAX_ELAPSED_SECONDS:
                stop_reason = "time limit"
            elif spend >= cfg.MAX_SPEND_USD:
                stop_reason = "spend limit"
            if stop_reason:
                break
            # tools, system_instruction, and generation_config are re-sent: they don't carry over
            interaction, secs = create(input=results, previous_interaction_id=interaction.id, **request)
            record_turn(interaction, secs, "investigate")
    except Exception as exc:  # API failure: record it and stop this attempt
        stop_reason = f"API error: {type(exc).__name__}: {exc}"
        log(stop_reason)

    record = {"thinking_level": level, "model": cfg.MODEL_ID, "started_utc": started.isoformat(),
              "rates": {k: v for k, v in rates.items() if k != "until" and k != "from"},
              "rate_period": "introductory" if rates is cfg.PRICES["introductory"] else "standard",
              "limits": {"turns": cfg.MAX_TURNS, "tool_calls": cfg.MAX_TOOL_CALLS,
                         "seconds": cfg.MAX_ELAPSED_SECONDS, "spend_usd": cfg.MAX_SPEND_USD},
              "stop_reason": stop_reason, "tool_calls": calls, "turns": turns, "report": None}

    # Final report: same history, same instruction and config, no tools, JSON schema
    if stop_reason == "agent finished":
        record["investigation_summary"] = interaction.output_text
        try:
            final, secs = create(input=REPORT_PROMPT, previous_interaction_id=interaction.id,
                                 model=cfg.MODEL_ID, system_instruction=SYSTEM_INSTRUCTION,
                                 generation_config=request["generation_config"],
                                 response_format={"type": "text", "mime_type": "application/json",
                                                  "schema": AuditReport.model_json_schema()})
            record_turn(final, secs, "report")
            record["report"] = AuditReport.model_validate_json(final.output_text).model_dump()
        except Exception as exc:
            record["report_error"] = f"{type(exc).__name__}: {exc}"
            log(record["report_error"])

    record["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    record["checks"] = harness_checks(record)
    record["cost"] = cost_summary(turns)
    if record["report"]:  # freeze before any evaluator or holdout code runs
        blob = json.dumps(record["report"], sort_keys=True).encode()
        record["frozen_utc"] = datetime.now(timezone.utc).isoformat()
        record["report_sha256"] = hashlib.sha256(blob).hexdigest()
    return record


def harness_checks(record: dict) -> dict:
    """Process and grounding checks in Python. They describe the attempt; they do not repair it."""
    calls, rep = record["tool_calls"], record["report"]
    names = [c["name"] for c in calls]
    checked_current = any(c["name"] == "evaluate_global_threshold" and not c["is_error"]
                          and c["arguments"].get("threshold") == cfg.CURRENT_THRESHOLD for c in calls)
    out = {"read_context_first": bool(names) and names[0] == "get_audit_context",
           "evaluated_current_threshold": checked_current,
           "invalid_tool_calls": sum(c["is_error"] for c in calls)}
    if not rep:
        return out
    out["cited_calls_exist"] = all(1 <= e["tool_call"] <= len(calls) for e in rep["evidence"])
    kind, tested = rep["recommended_change_type"], False
    for c in calls:
        a = c["arguments"]
        if c["is_error"] or c["name"] != "evaluate_policy_change" or a.get("change_type") != kind:
            continue
        if kind == "channel_thresholds":
            tested |= (a.get("website_threshold") == rep["recommended_website_threshold"]
                       and a.get("mobile_app_threshold") == rep["recommended_mobile_app_threshold"])
        elif kind == "capacity_increase":
            tested |= (a.get("threshold") == rep["recommended_global_threshold"]
                       and a.get("daily_capacity") == rep["recommended_daily_capacity"])
    out["recommendation_was_tested"] = tested if kind != "none" else None
    return out


def cost_summary(turns: list[dict]) -> dict:
    tot = {k: sum(t["usage"][k] for t in turns) for k in ("input", "cached", "output", "thought", "tool_use")}
    return {"tokens": tot, "api_turns": len(turns),
            "introductory_usd": round(sum(turn_cost(t["usage"], cfg.PRICES["introductory"]) for t in turns), 4),
            "standard_usd": round(sum(turn_cost(t["usage"], cfg.PRICES["standard"]) for t in turns), 4)}


def fixture_fingerprint() -> dict:
    """Hash the validation file so every run records exactly which data it saw."""
    data = Path("data/validation_scored.csv").read_bytes()
    return {"validation_sha256": hashlib.sha256(data).hexdigest(), "google_genai": _sdk_version()}


def _sdk_version() -> str:
    from importlib.metadata import version
    return version("google-genai")


def save(record: dict, folder: Path = RUNS) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"{stamp}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def run_experiment(log: Callable[[str], None] = print) -> Path:
    """Medium audit -> freeze -> evaluator -> holdout once -> clean high rerun -> evaluator."""
    import evaluator
    import holdout_check

    log(f"Medium attempt on {cfg.MODEL_ID}")
    medium = run_audit(cfg.PRIMARY_THINKING_LEVEL, log)
    if medium["report"]:
        medium["evaluator"] = evaluator.verify(medium["report"])
        log(f"Evaluator: correct_audit={medium['evaluator']['score']['correct_audit']}")
        holdout_check.check_once(medium)
        log("Holdout checked once on the frozen recommendation")
    log(f"Clean {cfg.RERUN_THINKING_LEVEL} rerun (no medium history)")
    high = run_audit(cfg.RERUN_THINKING_LEVEL, log)
    if high["report"]:
        high["evaluator"] = evaluator.verify(high["report"], include_curve=False)
        log(f"Evaluator: correct_audit={high['evaluator']['score']['correct_audit']}")
    path = save({"fixture": fixture_fingerprint(), "medium": medium, "high": high})
    log(f"Saved {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default=cfg.PRIMARY_THINKING_LEVEL, choices=["low", "medium", "high"])
    ap.add_argument("--attempt-only", action="store_true")
    args = ap.parse_args()
    if args.attempt_only:
        import evaluator
        rec = run_audit(args.level)
        rec["fixture"] = fixture_fingerprint()
        if rec["report"]:
            rec["evaluator"] = evaluator.verify(rec["report"], include_curve=False)
        print("saved", save(rec, RUNS / "attempts"))
    else:
        run_experiment()
