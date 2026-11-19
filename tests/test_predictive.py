from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from openmobility.processing.predictive import _metrics, _make_future_features


def test_metrics_are_calculated():
    result = _metrics(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    assert result["mae"] == 2.0
    assert result["rmse"] == 2.0
    assert result["observations"] == 2


def test_future_features_have_expected_lags():
    history = pd.Series(np.arange(1.0, 200.0))
    ts = pd.Timestamp("2026-05-01 08:00:00", tz="Europe/Paris")
    features = _make_future_features(history, ts, 12.0)
    assert features["lag_1"] == 199.0
    assert features["lag_24"] == 176.0
    assert features["lag_168"] == 32.0
    assert features["hour"] == 8
