"""Generate PayGuard's synthetic transactions: train, validation, and a sealed holdout.

Run once: python generate_data.py
Every value comes from GENERATOR_SEED, so each reader gets identical files.
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

DATA = Path("data")
SEALED = DATA / "sealed"  # holdout lives here; the agent's tools never open this folder

BASE_DAILY_VOLUME = 4_600
WEEKDAY_FACTOR = [0.92, 0.95, 0.97, 1.0, 1.12, 1.24, 1.05]  # Monday first
FRAUD_RATE = {"Website": 0.018, "Mobile App": 0.030}
MOBILE_SHARE = 0.28
WEB_MISMATCH = 0.98        # Website fraud: country mismatch is almost always present
WEB_FRAUD_AGE = 8          # mean account age in days for Website fraud
WEB_LEGIT_MISMATCH = 0.30  # travelers and VPN users make many honest Website orders look risky


def _day_frame(rng: np.random.Generator, day) -> pd.DataFrame:
    n = rng.poisson(BASE_DAILY_VOLUME * WEEKDAY_FACTOR[day.weekday()] * rng.normal(1.0, 0.04))
    channel = np.where(rng.random(n) < MOBILE_SHARE, "Mobile App", "Website")
    is_web = channel == "Website"
    fraud = rng.random(n) < np.where(is_web, FRAUD_RATE["Website"], FRAUD_RATE["Mobile App"])

    # Amounts in cents; fraud skews toward larger baskets
    log_amount = np.where(fraud, rng.normal(4.7, 0.85, n), rng.normal(3.7, 0.85, n))
    amount_cents = np.clip(np.round(np.exp(log_amount) * 100), 150, 500_000).astype(int)

    # Behavioral signals. Website fraud leaves clear traces; Mobile App fraud mostly does not.
    hour = np.where(fraud & (rng.random(n) < 0.45), rng.integers(0, 6, n), rng.integers(0, 24, n))
    account_age_days = np.round(np.where(
        fraud, rng.exponential(np.where(is_web, WEB_FRAUD_AGE, 260), n), rng.exponential(420, n))).astype(int)
    tx_last_24h = rng.poisson(np.where(fraud, np.where(is_web, 4.0, 2.0), 1.4))
    ip_country_mismatch = (rng.random(n) < np.where(
        is_web, np.where(fraud, WEB_MISMATCH, WEB_LEGIT_MISMATCH), 0.02)).astype(int)
    new_device = (rng.random(n) < np.where(fraud & ~is_web, 0.30, 0.06)).astype(int)

    seconds = rng.integers(0, 3600, n) + hour * 3600
    timestamp = pd.Timestamp(day, tz="UTC") + pd.to_timedelta(seconds, unit="s")
    return pd.DataFrame({
        "timestamp": timestamp,
        "amount_cents": amount_cents,
        "channel": channel,
        "hour": hour,
        "account_age_days": account_age_days,
        "tx_last_24h": tx_last_24h,
        "ip_country_mismatch": ip_country_mismatch,
        "new_device": new_device,
        "is_fraud": fraud.astype(int),
    })


def generate() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.GENERATOR_SEED)
    frames, day = [], cfg.TRAIN_START
    while day <= cfg.HOLDOUT_END:
        frames.append(_day_frame(rng, day))
        day += timedelta(days=1)
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable")
    df.insert(0, "transaction_id", [f"TX{i:07d}" for i in range(len(df))])
    d = df["timestamp"].dt.date
    return {
        "train": df[d <= cfg.TRAIN_END],
        "validation": df[(d >= cfg.VALIDATION_START) & (d <= cfg.VALIDATION_END)],
        "holdout": df[d >= cfg.HOLDOUT_START],
    }


if __name__ == "__main__":
    SEALED.mkdir(parents=True, exist_ok=True)
    splits = generate()
    splits["train"].to_csv(DATA / "train.csv", index=False)
    splits["validation"].to_csv(DATA / "validation.csv", index=False)
    splits["holdout"].to_csv(SEALED / "holdout.csv", index=False)
    for name, frame in splits.items():
        print(f"{name:<10} {len(frame):>7,} rows  fraud={frame['is_fraud'].sum():>5,}")
