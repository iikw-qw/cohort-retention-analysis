import numpy as np
import pytest

from src import core, data


@pytest.fixture(scope="module")
def prepared():
    signups, events = data.generate_events(n_users=800, months=8, seed=3)
    activity = core.activity_table(signups, events)
    return signups, events, activity


def test_age_zero_is_full_cohort(prepared):
    signups, _, activity = prepared
    matrix = core.retention_matrix(activity, signups)
    assert np.allclose(matrix[0], 1.0)


def test_retention_between_zero_and_one(prepared):
    signups, _, activity = prepared
    matrix = core.retention_matrix(activity, signups)
    assert matrix.min().min() >= 0
    assert matrix.max().max() <= 1.0


def test_future_cells_are_masked(prepared):
    signups, events, activity = prepared
    matrix = core.retention_matrix(activity, signups)
    last = core.to_month(events["event_date"]).max()
    masked = core.mask_future(matrix, last)
    newest = masked.index[-1]
    assert masked.loc[newest, 1:].isna().all()


def test_average_curve_decreases(prepared):
    signups, events, activity = prepared
    matrix = core.mask_future(
        core.retention_matrix(activity, signups), core.to_month(events["event_date"]).max()
    )
    curve = core.average_curve(matrix)
    assert curve.iloc[0] > curve.iloc[-1]
