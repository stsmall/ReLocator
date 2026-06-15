from __future__ import annotations

import json
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
    df = pd.DataFrame(
        {"Lat": [0.0, 10.0, 20.0], "Lon": [0.0, 10.0, 20.0]}
    )
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


import gzip
import subprocess
import sys
from textwrap import dedent


def _write_synthetic_inputs(tmp_path: Path, n_samples: int = 4):
    """Write minimal sample_table.tsv, samples_locations.tsv, and a fake beagle.gz."""
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    # Touch fake BAM files so they exist on disk.
    bam_paths = []
    for i in range(n_samples):
        bam = samples_dir / f"S{i}.bam"
        bam.write_bytes(b"")
        bam_paths.append(bam)

    # 2 sites, alternating assignment.
    sample_table = tmp_path / "sample_table.tsv"
    sample_table.write_text(
        "sample_name\tbam\tsite\n"
        + "\n".join(
            f"S{i}\t{bam_paths[i]}\t{'01-AAA-XX' if i % 2 == 0 else '02-BBB-YY'}"
            for i in range(n_samples)
        )
        + "\n"
    )

    samples_locations = tmp_path / "samples_locations.tsv"
    samples_locations.write_text(
        dedent(
            """\
            State\tSite\tLat\tLon\tSamples\tSiteCode
            XX\tAAA\t10.0\t20.0\t2\t01-AAA-XX
            YY\tBBB\t30.0\t40.0\t2\t02-BBB-YY
            """
        )
    )

    # Beagle header with marker, allele1, allele2, then 3 cols per sample.
    beagle = tmp_path / "fake.beagle.gz"
    header = ["marker", "allele1", "allele2"]
    for i in range(n_samples):
        header += [f"Ind{i}", f"Ind{i}", f"Ind{i}"]
    with gzip.open(beagle, "wt") as fh:
        fh.write("\t".join(header) + "\n")
        # One data row so the file isn't header-only.
        row = ["chr1_1", "A", "C"] + ["0.33"] * (3 * n_samples)
        fh.write("\t".join(row) + "\n")

    return sample_table, samples_locations, beagle, bam_paths


def _run_build_inputs(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "validation.build_inputs", *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )


def test_build_inputs_writes_filelist_and_sample_data(tmp_path: Path):
    sample_table, samples_locations, beagle, bam_paths = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "out"
    proc = _run_build_inputs(
        "--sample-table", str(sample_table),
        "--samples-locations", str(samples_locations),
        "--beagle", str(beagle),
        "--out-dir", str(out_dir),
    )
    assert proc.returncode == 0, proc.stderr

    bam_filelist = (out_dir / "bam.filelist").read_text().splitlines()
    assert len(bam_filelist) == 4
    assert bam_filelist == [str(p) for p in bam_paths]

    sd = pd.read_csv(out_dir / "sample_data.txt", sep="\t")
    assert list(sd.columns) == ["x", "y", "sampleID"]
    assert sd["sampleID"].tolist() == ["S0", "S1", "S2", "S3"]
    # S0 is at site AAA (Lat=10, Lon=20); x=Lon, y=Lat.
    assert sd.loc[sd["sampleID"] == "S0", "x"].iloc[0] == 20.0
    assert sd.loc[sd["sampleID"] == "S0", "y"].iloc[0] == 10.0


def test_build_inputs_rejects_missing_bam(tmp_path: Path):
    sample_table, samples_locations, beagle, bam_paths = _write_synthetic_inputs(tmp_path)
    bam_paths[0].unlink()  # delete one BAM
    out_dir = tmp_path / "out"
    proc = _run_build_inputs(
        "--sample-table", str(sample_table),
        "--samples-locations", str(samples_locations),
        "--beagle", str(beagle),
        "--out-dir", str(out_dir),
    )
    assert proc.returncode != 0
    assert "missing bam" in proc.stderr.lower() or "does not exist" in proc.stderr.lower()


def test_build_inputs_rejects_column_mismatch(tmp_path: Path):
    sample_table, samples_locations, _beagle, _bam_paths = _write_synthetic_inputs(tmp_path)
    # Build a beagle with the WRONG column count (5 samples worth of cols, not 4).
    bad_beagle = tmp_path / "bad.beagle.gz"
    header = ["marker", "allele1", "allele2"] + [f"Ind{i}" for i in range(5) for _ in range(3)]
    with gzip.open(bad_beagle, "wt") as fh:
        fh.write("\t".join(header) + "\n")
    out_dir = tmp_path / "out"
    proc = _run_build_inputs(
        "--sample-table", str(sample_table),
        "--samples-locations", str(samples_locations),
        "--beagle", str(bad_beagle),
        "--out-dir", str(out_dir),
    )
    assert proc.returncode != 0
    assert "column" in proc.stderr.lower() or "mismatch" in proc.stderr.lower()


def test_build_inputs_rejects_unjoinable_site(tmp_path: Path):
    sample_table, samples_locations, beagle, _bam_paths = _write_synthetic_inputs(tmp_path)
    # Replace one site code with a code that doesn't exist in samples_locations.
    text = sample_table.read_text().replace("01-AAA-XX", "99-ZZZ-XX", 1)
    sample_table.write_text(text)
    out_dir = tmp_path / "out"
    proc = _run_build_inputs(
        "--sample-table", str(sample_table),
        "--samples-locations", str(samples_locations),
        "--beagle", str(beagle),
        "--out-dir", str(out_dir),
    )
    assert proc.returncode != 0
    assert "join" in proc.stderr.lower() or "unmatched" in proc.stderr.lower()


def test_blank_holdout_locations_only_for_target_site(tmp_path: Path):
    from validation import run_loso

    sd = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [10.0, 20.0, 30.0, 40.0],
            "sampleID": ["s0", "s1", "s2", "s3"],
        }
    )
    sample_table = pd.DataFrame(
        {
            "sample_name": ["s0", "s1", "s2", "s3"],
            "site": ["A", "A", "B", "B"],
        }
    )
    out = run_loso.blank_holdout_locations(sd, sample_table, target_site="A")
    # site A (s0, s1) blanked; site B (s2, s3) untouched.
    assert (out.loc[out["sampleID"].isin(["s0", "s1"]), "x"] == "NA").all()
    assert (out.loc[out["sampleID"].isin(["s0", "s1"]), "y"] == "NA").all()
    assert out.loc[out["sampleID"] == "s2", "x"].iloc[0] == 3.0
    assert out.loc[out["sampleID"] == "s3", "y"].iloc[0] == 40.0


def test_aggregate_folds(tmp_path: Path):
    from validation import summarize

    for site in ["A", "B"]:
        d = tmp_path / "loso_dosage" / site
        d.mkdir(parents=True)
        (d / "fold_result.json").write_text(
            json.dumps(
                {
                    "site": site,
                    "true_lat": 10.0 if site == "A" else 20.0,
                    "true_lon": 100.0 if site == "A" else 200.0,
                    "gl_mode": "dosage",
                    "gpu": 0,
                    "status": "OK",
                    "n_held_out": 2,
                    "mean_pred_lat": 11.0 if site == "A" else 21.0,
                    "mean_pred_lon": 101.0 if site == "A" else 201.0,
                    "mean_error_km": 50.0 if site == "A" else 75.0,
                    "median_error_km": 50.0 if site == "A" else 75.0,
                    "per_sample": [],
                }
            )
        )
    df = summarize.aggregate_folds(tmp_path / "loso_dosage", "dosage")
    assert len(df) == 2
    assert set(df["site"]) == {"A", "B"}
    assert df["mean_error_km"].sum() == 125.0


# ---------------------------------------------------------------------------
# Coastline projection
# ---------------------------------------------------------------------------


def test_coast_index_zero_at_first_site():
    from validation import coastline

    cum_km, points = coastline.build_coast_index()
    # First site = origin of polyline; cum km is 0.
    assert cum_km[0] == pytest.approx(0.0)
    # Polyline length is the sum of 11 great-circle segments through the
    # 12 sample sites — empirically ~3887 km. Bracket loosely so a small
    # change in site coordinates doesn't break the test, but tight enough
    # to catch a units regression.
    assert 3500 <= cum_km[-1] <= 4500


def test_project_at_known_site():
    from validation import coastline

    cum_km, points = coastline.build_coast_index()
    # Site index 5 is "06-MEA-OR" at lat 45.486, lon -123.975.
    site_lat, site_lon = float(points[5, 0]), float(points[5, 1])
    coast_pos, offshore = coastline.project_to_coast(
        site_lat, site_lon, cum_km, points
    )
    assert coast_pos == pytest.approx(cum_km[5], abs=1.0)
    assert offshore < 1.0


def test_project_offshore():
    from validation import coastline

    cum_km, points = coastline.build_coast_index()
    # Take site 6 (CPE-OR at ~44.28, -124.11) and shift it ~50 km west by
    # adding 0.65 degrees of longitude (≈50 km at lat 44).
    site_lat, site_lon = float(points[6, 0]), float(points[6, 1])
    coast_pos, offshore = coastline.project_to_coast(
        site_lat, site_lon - 0.65, cum_km, points
    )
    # Projection should still land near site 6's polyline position.
    assert abs(coast_pos - cum_km[6]) < 30.0
    # Offshore distance should be roughly 50 km.
    assert 30.0 < offshore < 75.0


def test_along_coast_distance_ordering():
    from validation import coastline

    cum_km, points = coastline.build_coast_index()
    # Predict site 1's lat/lon, with truth = site 4. The distance should be
    # cum_km[4] - cum_km[1] (positive, large).
    pred_lat, pred_lon = float(points[1, 0]), float(points[1, 1])
    err = coastline.along_coast_distance(pred_lat, pred_lon, 4, cum_km, points)
    expected = cum_km[4] - cum_km[1]
    assert err == pytest.approx(expected, abs=1.0)
