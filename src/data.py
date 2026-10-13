import numpy as np
import pandas as pd


def generate_events(n_users=6000, months=12, seed=7):
    """Users sign up over time and return with a probability that decays with age.

    Later cohorts retain slightly better, to give the analysis something to find.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")
    signup_month = rng.integers(0, months, n_users)
    signup_day = rng.integers(0, 28, n_users)
    signups = pd.DataFrame(
        {
            "user_id": np.arange(n_users),
            "signup_date": [
                start + pd.DateOffset(months=int(m), days=int(d))
                for m, d in zip(signup_month, signup_day)
            ],
        }
    )

    rows = []
    for uid, m in zip(signups["user_id"], signup_month):
        rows.append((uid, start + pd.DateOffset(months=int(m))))
        quality = 0.35 + 0.015 * m
        for age in range(1, months - m):
            p = quality * np.exp(-0.25 * (age - 1)) + 0.04
            if rng.random() < p:
                rows.append((uid, start + pd.DateOffset(months=int(m + age))))
    events = pd.DataFrame(rows, columns=["user_id", "event_date"])
    return signups, events
