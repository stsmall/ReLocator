"""Tests for scripts/microsat_to_locator.py.

Helper functions are tested by direct import; full-script behavior is tested
through subprocess invocations against synthetic fixtures (matches
test_input_extensions.py pattern for the GL converter).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "microsat_to_locator.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("microsat_to_locator", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def msl():
    return _load_module()


def run_script(*args):
    cmd = [sys.executable, str(SCRIPT_PATH), *map(str, args)]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


# ---------------------------------------------------------------------------
# Task 1: parse_genotype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cell, expected",
    [
        ("12,14", (12, 14)),
        ("12/14", (12, 14)),
        ("12 14", (12, 14)),
        ("12|14", (12, 14)),
        (" 12 , 14 ", (12, 14)),
        ("14", (14, 14)),
        ("NA", (None, None)),
        ("nan", (None, None)),
        (".", (None, None)),
        ("", (None, None)),
        ("0,0", (None, None)),
        ("not_a_number", (None, None)),
        ("12,not_a_number", (None, None)),
    ],
)
def test_parse_genotype_variants(msl, cell, expected):
    assert msl.parse_genotype(cell) == expected


# ---------------------------------------------------------------------------
# Task 2: build_allele_catalog
# ---------------------------------------------------------------------------

def _df_pairs(rows: list[dict]) -> pd.DataFrame:
    """Helper: build a microsat DataFrame from a list of {sampleID, locus: 'a,b'} dicts."""
    df = pd.DataFrame(rows).set_index("sampleID")
    return df


def test_catalog_keeps_all_alleles_above_maf(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,11"},
        {"sampleID": "s2", "L1": "10,12"},
        {"sampleID": "s3", "L1": "11,12"},
        {"sampleID": "s4", "L1": "10,11"},
    ])
    catalog = msl.build_allele_catalog(df, ["L1"], min_allele_freq=0.0, max_locus_missing=1.0)
    assert catalog["L1"] == [10, 11, 12]


def test_catalog_drops_rare_alleles(msl):
    rows = [{"sampleID": f"s{i}", "L1": "10,11"} for i in range(99)]
    rows.append({"sampleID": "s99", "L1": "10,99"})
    df = _df_pairs(rows)
    catalog = msl.build_allele_catalog(df, ["L1"], min_allele_freq=0.05, max_locus_missing=1.0)
    assert 99 not in catalog["L1"]
    assert 10 in catalog["L1"] and 11 in catalog["L1"]


def test_catalog_handles_all_missing_locus(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "NA"},
        {"sampleID": "s2", "L1": "NA"},
    ])
    catalog = msl.build_allele_catalog(df, ["L1"], min_allele_freq=0.0, max_locus_missing=1.0)
    assert catalog["L1"] == []


def test_catalog_warns_above_missing_threshold(msl, capsys):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,11"},
        {"sampleID": "s2", "L1": "NA"},
        {"sampleID": "s3", "L1": "NA"},
    ])
    catalog = msl.build_allele_catalog(
        df, ["L1"], min_allele_freq=0.0, max_locus_missing=0.5
    )
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "L1" in captured.err
    assert catalog["L1"] == [10, 11]  # alleles still returned despite warning


def test_catalog_does_not_warn_at_default_threshold(msl, capsys):
    """max_locus_missing=1.0 (Task 7 default) should never warn — '1.0 = never warn'."""
    df = _df_pairs([
        {"sampleID": "s1", "L1": "NA"},
        {"sampleID": "s2", "L1": "NA"},
    ])
    msl.build_allele_catalog(df, ["L1"], min_allele_freq=0.0, max_locus_missing=1.0)
    captured = capsys.readouterr()
    assert "WARNING" not in captured.err


# ---------------------------------------------------------------------------
# Task 3: encode_dosage_block
# ---------------------------------------------------------------------------

def test_dosage_block_basic(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,11"},
        {"sampleID": "s2", "L1": "10,10"},
        {"sampleID": "s3", "L1": "11,11"},
    ])
    catalog = {"L1": [10, 11]}
    matrix, col_names = msl.encode_dosage_block(df, ["L1"], catalog)
    assert col_names == ["dosage_L1_10", "dosage_L1_11"]
    np.testing.assert_array_equal(matrix, np.array([[1, 1], [2, 0], [0, 2]], dtype=np.float32))


def test_dosage_block_multi_locus(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,11", "L2": "20,21"},
        {"sampleID": "s2", "L1": "11,11", "L2": "20,20"},
    ])
    catalog = {"L1": [10, 11], "L2": [20, 21]}
    matrix, col_names = msl.encode_dosage_block(df, ["L1", "L2"], catalog)
    assert col_names == ["dosage_L1_10", "dosage_L1_11", "dosage_L2_20", "dosage_L2_21"]
    expected = np.array([[1, 1, 1, 1], [0, 2, 2, 0]], dtype=np.float32)
    np.testing.assert_array_equal(matrix, expected)


def test_dosage_block_imputes_missing_with_site_mean(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,10"},
        {"sampleID": "s2", "L1": "10,11"},
        {"sampleID": "s3", "L1": "NA"},
    ])
    catalog = {"L1": [10, 11]}
    matrix, _ = msl.encode_dosage_block(df, ["L1"], catalog)
    # site mean across non-missing: col 0 = (2+1)/2 = 1.5; col 1 = (0+1)/2 = 0.5
    np.testing.assert_allclose(matrix[2], np.array([1.5, 0.5], dtype=np.float32))


def test_dosage_block_drops_alleles_outside_catalog(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,99"},  # 99 is not in catalog
        {"sampleID": "s2", "L1": "10,10"},
    ])
    catalog = {"L1": [10]}
    matrix, col_names = msl.encode_dosage_block(df, ["L1"], catalog)
    assert col_names == ["dosage_L1_10"]
    # s1 has only one in-catalog allele (10), so dosage = 1
    np.testing.assert_allclose(matrix, np.array([[1.0], [2.0]], dtype=np.float32))


# ---------------------------------------------------------------------------
# Task 4: encode_geometry_block
# ---------------------------------------------------------------------------

def test_geometry_block_hand_computation(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,12"},  # mean=11, diff=2, het=1
        {"sampleID": "s2", "L1": "14,14"},  # mean=14, diff=0, het=0
    ])
    matrix, col_names = msl.encode_geometry_block(df, ["L1"])
    assert col_names == ["geom_L1_mean_repeat", "geom_L1_allele_diff", "geom_L1_het"]
    expected = np.array([[11.0, 2.0, 1.0], [14.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(matrix, expected)


def test_geometry_block_imputes_missing(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,10"},   # mean=10, diff=0, het=0
        {"sampleID": "s2", "L1": "12,14"},   # mean=13, diff=2, het=1
        {"sampleID": "s3", "L1": "NA"},
    ])
    matrix, _ = msl.encode_geometry_block(df, ["L1"])
    # imputed: mean_repeat = (10+13)/2 = 11.5, allele_diff = (0+2)/2 = 1.0, het rate = 0.5
    np.testing.assert_allclose(matrix[2], np.array([11.5, 1.0, 0.5], dtype=np.float32))


def test_geometry_block_multi_locus_column_order(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,12", "L2": "20,22"},
    ])
    _, col_names = msl.encode_geometry_block(df, ["L1", "L2"])
    assert col_names == [
        "geom_L1_mean_repeat", "geom_L1_allele_diff", "geom_L1_het",
        "geom_L2_mean_repeat", "geom_L2_allele_diff", "geom_L2_het",
    ]


def test_geometry_block_all_missing_locus_yields_nan(msl):
    """All-missing locus → NaN-filled columns (loud signal, not silent zeros)."""
    df = _df_pairs([
        {"sampleID": "s1", "L1": "NA"},
        {"sampleID": "s2", "L1": "NA"},
    ])
    matrix, _ = msl.encode_geometry_block(df, ["L1"])
    assert matrix.shape == (2, 3)
    assert np.all(np.isnan(matrix))


# ---------------------------------------------------------------------------
# Task 5: encode_repeat_norm_block
# ---------------------------------------------------------------------------

def test_repeat_norm_zscore_basic(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,10"},  # mean = 10
        {"sampleID": "s2", "L1": "12,12"},  # mean = 12
        {"sampleID": "s3", "L1": "14,14"},  # mean = 14
    ])
    matrix, col_names = msl.encode_repeat_norm_block(df, ["L1"])
    assert col_names == ["rnorm_L1"]
    # mean=12, std=sqrt(((10-12)^2 + 0 + (14-12)^2)/3) = sqrt(8/3) ≈ 1.633
    expected_std = np.sqrt(8 / 3)
    expected = np.array(
        [[(10 - 12) / expected_std], [0.0], [(14 - 12) / expected_std]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(matrix, expected, atol=1e-5)


def test_repeat_norm_zero_variance_locus(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,10"},
        {"sampleID": "s2", "L1": "10,10"},
    ])
    matrix, _ = msl.encode_repeat_norm_block(df, ["L1"])
    # std = 0; we set z-score to 0 for all samples (no information)
    np.testing.assert_array_equal(matrix, np.zeros((2, 1), dtype=np.float32))


def test_repeat_norm_imputes_missing_to_zero(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,10"},
        {"sampleID": "s2", "L1": "14,14"},
        {"sampleID": "s3", "L1": "NA"},
    ])
    matrix, _ = msl.encode_repeat_norm_block(df, ["L1"])
    # Missing → per-locus mean → 0 after z-scoring
    assert matrix[2, 0] == pytest.approx(0.0, abs=1e-5)


def test_repeat_norm_multi_locus_column_order(msl):
    df = _df_pairs([
        {"sampleID": "s1", "L1": "10,12", "L2": "20,22"},
        {"sampleID": "s2", "L1": "14,16", "L2": "30,32"},
    ])
    _, col_names = msl.encode_repeat_norm_block(df, ["L1", "L2"])
    assert col_names == ["rnorm_L1", "rnorm_L2"]


def test_repeat_norm_all_missing_locus_yields_zeros(msl):
    """All-missing locus → all-zeros column.

    Different from geometry's NaN-fill: repeat_norm columns are z-scores,
    not biological magnitudes. Zero z-score = at the population mean = the
    correct 'no information' prior for an unobserved standardized variable.
    """
    df = _df_pairs([
        {"sampleID": "s1", "L1": "NA"},
        {"sampleID": "s2", "L1": "NA"},
    ])
    matrix, _ = msl.encode_repeat_norm_block(df, ["L1"])
    assert matrix.shape == (2, 1)
    np.testing.assert_array_equal(matrix, np.zeros((2, 1), dtype=np.float32))


# ---------------------------------------------------------------------------
# Task 6: format detection
# ---------------------------------------------------------------------------


def test_detect_format_pair(msl):
    df = pd.DataFrame({
        "sampleID": ["s1", "s2"],
        "L1": ["10,11", "12,13"],
    })
    assert msl.detect_format(df) == "pair"


def test_detect_format_two_column(msl):
    df = pd.DataFrame({
        "sampleID": ["s1", "s2"],
        "variant_0": ["10", "12"],
        "variant_1": ["11", "13"],
    })
    assert msl.detect_format(df) == "two_column"


def test_detect_format_two_column_odd_columns_raises(msl):
    df = pd.DataFrame({
        "sampleID": ["s1"],
        "variant_0": ["10"],
        "variant_1": ["11"],
        "variant_2": ["12"],  # odd
    })
    with pytest.raises(ValueError, match="odd"):
        msl.detect_format(df)


def test_convert_two_column_pairs_alleles(msl):
    df = pd.DataFrame({
        "sampleID": ["s1", "s2"],
        "variant_0": ["10", "14"],
        "variant_1": ["11", "16"],
        "variant_2": ["20", "22"],
        "variant_3": ["21", "24"],
    })
    out = msl.convert_two_column_to_pair(df)
    assert list(out.columns) == ["sampleID", "locus_0", "locus_1"]
    assert list(out["locus_0"]) == ["10,11", "14,16"]
    assert list(out["locus_1"]) == ["20,21", "22,24"]


# ---------------------------------------------------------------------------
# Task 7: end-to-end CLI
# ---------------------------------------------------------------------------

def _write_pair_tsv(path: Path) -> int:
    """Write a small pair-format input. Returns n_samples."""
    rows = [
        ["sampleID", "L1", "L2"],
        ["s1", "10,11", "20,22"],
        ["s2", "10,10", "20,20"],
        ["s3", "11,12", "22,24"],
        ["s4", "10,12", "NA"],
    ]
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    return len(rows) - 1


def test_cli_default_features_combines_all_three_modes(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out)
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(out, sep="\t")
    cols = list(df.columns)
    assert cols[0] == "sampleID"
    assert any(c.startswith("dosage_") for c in cols)
    assert any(c.startswith("geom_") for c in cols)
    assert any(c.startswith("rnorm_") for c in cols)


def test_cli_features_subset_dosage_only(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out, "--features", "dosage")
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(out, sep="\t")
    cols = list(df.columns)
    assert cols[0] == "sampleID"
    for c in cols[1:]:
        assert c.startswith("dosage_"), c


def test_cli_unknown_feature_errors(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out, "--features", "bogus")
    assert proc.returncode != 0
    assert "bogus" in (proc.stderr + proc.stdout).lower()


def test_cli_two_column_format_works(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    rows = [
        ["sampleID", "variant_0", "variant_1", "variant_2", "variant_3"],
        ["s1", "10", "11", "20", "22"],
        ["s2", "10", "10", "20", "20"],
        ["s3", "11", "12", "22", "24"],
    ]
    inp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out, "--features", "geometry")
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(out, sep="\t")
    assert len(df) == 3
    # Two reconstructed loci × 3 geometry columns each = 6 feature cols + sampleID
    assert df.shape[1] == 7


def test_cli_combined_mode_column_order_is_dosage_then_geom_then_rnorm(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out)
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(out, sep="\t")
    cols = [c for c in df.columns if c != "sampleID"]
    prefixes = [c.split("_")[0] for c in cols]
    last_dosage = max(i for i, p in enumerate(prefixes) if p == "dosage")
    first_geom = min(i for i, p in enumerate(prefixes) if p == "geom")
    last_geom = max(i for i, p in enumerate(prefixes) if p == "geom")
    first_rnorm = min(i for i, p in enumerate(prefixes) if p == "rnorm")
    assert last_dosage < first_geom < last_geom < first_rnorm


def test_cli_writes_encoding_report(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    out = tmp_path / "feat.tsv"
    enc = tmp_path / "encoding.tsv"
    proc = run_script(
        "--microsat", inp, "--out", out,
        "--features", "dosage,geometry",
        "--report_encoding", enc,
    )
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(enc, sep="\t")
    assert {"feature_type", "locus", "column_name"}.issubset(df.columns)


def test_cli_missing_sampleID_column_returns_2(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    inp.write_text("badheader\tL1\nfoo\t10,11\n")
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out)
    assert proc.returncode == 2
    assert "sampleID" in (proc.stderr + proc.stdout)


def test_cli_duplicate_sampleIDs_returns_4(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    rows = [
        ["sampleID", "L1"],
        ["s1", "10,11"],
        ["s1", "10,12"],  # duplicate
    ]
    inp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out)
    assert proc.returncode == 4
    assert "duplicate" in (proc.stderr + proc.stdout).lower() or "Duplicate" in proc.stderr


def test_cli_no_active_loci_returns_3(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    # Use pair format (comma-separated) so detect_format returns "pair" cleanly,
    # and all genotypes are NA so no loci survive → exit 3.
    rows = [
        ["sampleID", "L1"],
        ["s1", "NA,NA"],
        ["s2", "NA,NA"],
    ]
    inp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out)
    assert proc.returncode == 3
    assert "no loci" in (proc.stderr + proc.stdout).lower()


def test_cli_two_column_odd_columns_returns_5_clean_error(tmp_path: Path):
    """Odd column count in two-column format should produce a clean error, not a traceback."""
    inp = tmp_path / "ms.tsv"
    rows = [
        ["sampleID", "v0", "v1", "v2"],  # 3 locus cols, odd
        ["s1", "10", "11", "12"],
    ]
    inp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out)
    assert proc.returncode == 5
    assert "odd" in (proc.stderr + proc.stdout).lower()
    # No raw Python traceback in stderr
    assert "Traceback" not in proc.stderr


def test_cli_encoding_report_row_count_matches_features(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    out = tmp_path / "feat.tsv"
    enc = tmp_path / "encoding.tsv"
    proc = run_script(
        "--microsat", inp, "--out", out,
        "--features", "dosage,geometry,repeat_norm",
        "--report_encoding", enc,
    )
    assert proc.returncode == 0, proc.stderr
    out_df = pd.read_csv(out, sep="\t")
    enc_df = pd.read_csv(enc, sep="\t")
    feature_cols = [c for c in out_df.columns if c != "sampleID"]
    assert len(enc_df) == len(feature_cols), (
        f"encoding report has {len(enc_df)} rows but output has {len(feature_cols)} features"
    )
