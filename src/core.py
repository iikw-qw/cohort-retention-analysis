import pandas as pd


def to_month(s):
    return s.dt.to_period("M")


def activity_table(signups, events):
    """One row per user and active month, with cohort and age in months."""
    df = events.merge(signups, on="user_id")
    df["cohort"] = to_month(df["signup_date"])
    df["active_month"] = to_month(df["event_date"])
    df["age"] = (df["active_month"] - df["cohort"]).apply(lambda x: x.n)
    return df[df["age"] >= 0].drop_duplicates(["user_id", "active_month"])


def retention_matrix(activity, signups, as_share=True):
    sizes = to_month(signups["signup_date"]).value_counts().sort_index()
    counts = (
        activity.groupby(["cohort", "age"])["user_id"].nunique().unstack(fill_value=0)
    )
    counts = counts.reindex(sizes.index)
    if not as_share:
        return counts
    return counts.div(sizes, axis=0)


def mask_future(matrix, last_month):
    """Blank out cells that fall after the end of the observation window."""
    out = matrix.copy()
    for cohort in out.index:
        max_age = (last_month - cohort).n
        out.loc[cohort, out.columns > max_age] = float("nan")
    return out


def average_curve(matrix, min_cohorts=3):
    curve = matrix.mean(axis=0, skipna=True)
    support = matrix.notna().sum(axis=0)
    return curve[support >= min_cohorts]


def cohort_quality(matrix, age=1):
    """Retention at a fixed age for every cohort."""
    return matrix[age].dropna()
