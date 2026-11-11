"""Frozen settings for the PayGuard audit. Nothing here changes after the first API call."""

from datetime import date

# Fixed seeds
GENERATOR_SEED = 20261111
MODEL_SEED = 7

# Chronological windows (UTC calendar days, end dates inclusive)
TRAIN_START, TRAIN_END = date(2026, 6, 1), date(2026, 8, 2)
VALIDATION_START, VALIDATION_END = date(2026, 8, 3), date(2026, 8, 30)
HOLDOUT_START, HOLDOUT_END = date(2026, 8, 31), date(2026, 9, 27)

CHANNELS = ("Website", "Mobile App")
SCORE_DECIMALS = 3  # PayGuard stores risk scores rounded to 0.001

# Deployment policy (fictional targets, written before any model call)
CURRENT_THRESHOLD = 0.5
MIN_OVERALL_FRAUD_VALUE_RECALL = 0.85
MIN_CHANNEL_FRAUD_VALUE_RECALL = 0.70
MAX_DAILY_REVIEWS = 200
MAX_PERMITTED_DAILY_REVIEWS = 260  # ceiling for a capacity exception

POLICY_TEXT = f"""PayGuard deployment policy (fixed; the audit may not change it)
- One global decision threshold applies to both channels. A transaction is flagged
  for manual review when risk_score >= threshold.
- R1: Flagged transactions must cover at least {MIN_OVERALL_FRAUD_VALUE_RECALL:.0%} of the
  labeled fraud value across all channels.
- R2: Flagged transactions must cover at least {MIN_CHANNEL_FRAUD_VALUE_RECALL:.0%} of the
  labeled fraud value within each channel (Website and Mobile App separately).
- R3: No more than {MAX_DAILY_REVIEWS} transactions may be flagged on any single UTC
  calendar day. This is a maximum per day, not an average.
- Fraud-value recall = value of labeled fraud that is flagged / total value of labeled fraud.
  Flagging a transaction does not mean the fraud was stopped or the money recovered.
- Precision, transaction-level recall, false positives, and average precision are
  diagnostics only. They are not requirements.
Permitted changes the audit may recommend (each needs explicit human approval):
- A higher daily review capacity, up to {MAX_PERMITTED_DAILY_REVIEWS} transactions per day,
  with one global threshold.
- Separate thresholds for Website and Mobile App with the original daily capacity.
- The two changes may not be combined.
Preference: a compliant option that needs no extra review capacity is preferred. If none
exists, the smallest verified capacity increase within the permitted range is preferred."""

# Gemini settings
MODEL_ID = "gemini-4-carbon"
PRIMARY_THINKING_LEVEL = "medium"
RERUN_THINKING_LEVEL = "high"
MAX_OUTPUT_TOKENS = 65_536  # default synchronous cap for thinking plus output
HTTP_TIMEOUT_SECONDS = 900  # Google recommends at least 15 minutes

# Application limits, enforced in agent.py
MAX_TURNS = 25
MAX_TOOL_CALLS = 60
MAX_ELAPSED_SECONDS = 1_800
MAX_SPEND_USD = 3.00

# Standard paid-tier prices per 1M tokens (Gemini 4 Carbon pricing page, Nov 11, 2026)
PRICES = {
    "introductory": {"until": date(2026, 11, 30), "input": 2.00, "cached": 0.10, "output": 10.00},
    "standard": {"from": date(2026, 12, 1), "input": 4.00, "cached": 0.20, "output": 20.00},
}
