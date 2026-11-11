"""Fit PayGuard's fraud classifier on the training window and score the later windows.

Run after generate_data.py: python prepare_model.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

import config as cfg
from generate_data import DATA, SEALED

FEATURES = ["log_amount", "is_mobile", "night", "log_account_age",
            "tx_last_24h", "ip_country_mismatch", "new_device"]


def features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "log_amount": np.log(df["amount_cents"] / 100),
        "is_mobile": (df["channel"] == "Mobile App").astype(int),
        "night": (df["hour"] < 6).astype(int),
        "log_account_age": np.log1p(df["account_age_days"]),
        "tx_last_24h": df["tx_last_24h"],
        "ip_country_mismatch": df["ip_country_mismatch"],
        "new_device": df["new_device"],
    })


def scored(df: pd.DataFrame, model) -> pd.DataFrame:
    """Keep only what the audit needs: time, amount, channel, score, label."""
    risk = np.round(model.predict_proba(features(df))[:, 1], cfg.SCORE_DECIMALS)
    return df[["transaction_id", "timestamp", "amount_cents", "channel", "is_fraud"]].assign(risk_score=risk)


if __name__ == "__main__":
    train = pd.read_csv(DATA / "train.csv")
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                           random_state=cfg.MODEL_SEED)
    model.fit(features(train), train["is_fraud"])

    validation = scored(pd.read_csv(DATA / "validation.csv"), model)
    validation.to_csv(DATA / "validation_scored.csv", index=False)
    scored(pd.read_csv(SEALED / "holdout.csv"), model).to_csv(SEALED / "holdout_scored.csv", index=False)

    summary = {
        "classifier": "scikit-learn HistGradientBoostingClassifier (max_iter=200, learning_rate=0.05)",
        "features": FEATURES,
        "training_window": f"{cfg.TRAIN_START} to {cfg.TRAIN_END}",
        "training_rows": int(len(train)),
        "training_fraud_rate": round(float(train["is_fraud"].mean()), 4),
        "train_average_precision": round(float(average_precision_score(
            train["is_fraud"], model.predict_proba(features(train))[:, 1])), 3),
    }
    (DATA / "model_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
