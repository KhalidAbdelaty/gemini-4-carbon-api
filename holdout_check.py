"""Apply a frozen recommendation to the sealed holdout window, once. Nothing goes back to the agent.

agent.run_experiment() calls check_once() after the evaluator. The function refuses a report
that is not frozen, a report whose hash changed, or a record that already has a holdout result.
"""

import hashlib
import json

import config as cfg
import evaluator as ev
from generate_data import SEALED


def check_once(record: dict) -> dict:
    report = record.get("report")
    if not report or "report_sha256" not in record:
        raise RuntimeError("No frozen recommendation, so the holdout stays sealed.")
    if hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest() != record["report_sha256"]:
        raise RuntimeError("The report changed after it was frozen.")
    if record.get("holdout"):
        raise RuntimeError("This recommendation already has a holdout result.")

    if report["recommended_change_type"] == "none":
        config = {"recommended_change_type": "none"}
    else:
        config = {k: report[k] for k in report if k.startswith("recommended_")}
    windows = {"validation": ev.Window(ev.load("data/validation_scored.csv")),
               "holdout": ev.Window(ev.load(SEALED / "holdout_scored.csv"))}
    result = {"frozen_sha256": record["report_sha256"], "configuration": config}
    for name, win in windows.items():
        result[name] = ev.evaluate_config(win, config) if config["recommended_change_type"] != "none" else \
            win.check(ev.to_milli(cfg.CURRENT_THRESHOLD), ev.to_milli(cfg.CURRENT_THRESHOLD), cfg.MAX_DAILY_REVIEWS)
        result[name]["days"] = len(win.days)
    record["holdout"] = result
    return result
