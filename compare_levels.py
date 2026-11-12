"""Summarize every saved attempt by thinking level. No API call.

python compare_levels.py                  the tutorial's six attempts in recorded_runs/
python compare_levels.py --folder runs    your own attempts from agent.py
"""

import argparse
import json
from pathlib import Path


def attempts(folder: Path) -> list[dict]:
    """agent.py saves medium and high together; --attempt-only saves one attempt per file."""
    out = []
    for path in sorted(folder.rglob("*.json")):
        record = json.loads(path.read_text())
        out += [record[k] for k in ("medium", "high") if k in record] if "medium" in record else [record]
    return out


def summarize(rows: list[dict]) -> dict:
    completed = [a for a in rows if a.get("report")]
    correct = [a for a in completed if a.get("evaluator", {}).get("score", {}).get("correct_audit")]
    n = len(rows)
    return {
        "attempts": n,
        "completed": len(completed),
        "correct": len(correct),
        "stopped_by": sorted({a["stop_reason"] for a in rows if not a.get("report")}),
        "mean_thinking_tokens": round(sum(a["cost"]["tokens"]["thought"] for a in rows) / n),
        "mean_seconds": round(sum(a["elapsed_seconds"] for a in rows) / n, 1),
        "tool_calls_requested": [len(a["tool_calls"]) for a in rows],
        # every attempt's spend counts, including attempts that stopped without a report
        "usd_per_correct_introductory": round(sum(a["cost"]["introductory_usd"] for a in rows) / len(correct), 4)
        if correct else None,
        "usd_per_correct_standard": round(sum(a["cost"]["standard_usd"] for a in rows) / len(correct), 4)
        if correct else None,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="recorded_runs")
    folder = Path(ap.parse_args().folder)
    rows = attempts(folder)
    if not rows:
        raise SystemExit(f"No saved attempts in {folder}/")
    for level in ("low", "medium", "high"):
        group = [a for a in rows if a["thinking_level"] == level]
        if group:
            print(level, json.dumps(summarize(group), indent=2))
