from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validation import common


def test_haversine_nyc_to_la():
    nyc_lat, nyc_lon = 40.7128, -74.0060
    la_lat, la_lon = 34.0522, -118.2437
    km = common.haversine(nyc_lat, nyc_lon, la_lat, la_lon)
    assert 3930 < km < 3960  # canonical great-circle distance ~3935 km


def test_haversine_zero_distance():
    assert common.haversine(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-6)


def test_haversine_vectorized_pair():
    out = common.haversine(
        np.array([0.0, 40.7128]),
        np.array([0.0, -74.0060]),
        np.array([0.0, 34.0522]),
        np.array([0.0, -118.2437]),
    )
    assert out.shape == (2,)
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert 3930 < out[1] < 3960


def test_centroid_baseline_simple():
    df = pd.DataFrame({"Lat": [0.0, 10.0, 20.0], "Lon": [0.0, 10.0, 20.0]})
    lat, lon = common.centroid_baseline(df)
    assert lat == pytest.approx(10.0)
    assert lon == pytest.approx(10.0)


def test_parse_predlocs_csv(tmp_path: Path):
    p = tmp_path / "run_predlocs.txt"
    p.write_text("sampleID,x,y\nA,1.5,2.5\nB,-3.0,4.0\n")
    df = common.parse_predlocs(p)
    assert list(df.columns) == ["sampleID", "x", "y"]
    assert len(df) == 2
    assert df.loc[df["sampleID"] == "A", "x"].iloc[0] == 1.5


def test_gpu_round_robin():
    assert common.gpu_round_robin(3, 0) == 0
    assert common.gpu_round_robin(3, 1) == 1
    assert common.gpu_round_robin(3, 2) == 2
    assert common.gpu_round_robin(3, 3) == 0
    assert common.gpu_round_robin(3, 7) == 1
