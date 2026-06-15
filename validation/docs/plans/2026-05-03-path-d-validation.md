# Path D Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the validation tooling and execute the full Path A → B → C arc against the 54-sample balanus test set, producing a `summary.md` checkpoint that the human uses to decide whether to push the `genotype_likelihoods` branch.

**Architecture:** Five new files under `scripts/validation/` (no production-script changes). TDD where logic permits (joins, math, file I/O); the LOSO orchestrator is exercised as an integration test by running the real sweep. Subprocess calls to the existing `locator` CLI keep this decoupled from ReLocator internals.

**Tech Stack:** Python 3.12, pandas, numpy, matplotlib, the existing `pixi` env (TF 2.19.1, 3× A100 80GB), `pytest` via `pixi run pytest`, the existing `locator` CLI from the editable install.

**Spec:** `docs/superpowers/specs/2026-05-03-path-d-validation-design.md`.

---

## File map

Created in this plan:

| Path | Responsibility |
|---|---|
| `scripts/validation/__init__.py` | empty marker |
| `scripts/validation/common.py` | `haversine`, `centroid_baseline`, `parse_predlocs`, `gpu_round_robin` |
| `scripts/validation/build_inputs.py` | TSV join → `bam.filelist` + `sample_data.txt`, with validation |
| `scripts/validation/run_smoke.py` | Path A executor |
| `scripts/validation/run_loso.py` | Paths B+C orchestrator (12 folds × 1 mode per invocation, 3-way GPU parallelism) |
| `scripts/validation/summarize.py` | aggregate fold JSONs → summary table, plot, markdown report |
| `tests/test_validation.py` | unit tests for `common.py` and `build_inputs.py` |

Output tree (created at runtime by the scripts):

```
out/path_d_validation/
    inputs/{bam.filelist, sample_data.txt, build_log.txt}
    smoke/{gl_dosage.txt, locator_run/, result.json}
    loso_dosage/<SITE>/{sample_data.txt, locator_run/, fold_result.json}
    loso_full_gl/<SITE>/{sample_data.txt, locator_run/, fold_result.json}
    summary/{summary.tsv, summary.md, dosage_vs_full_gl.png}
```

---

## Conventions (apply to every task)

- All Python files start with `from __future__ import annotations` and follow the existing `scripts/` style (argparse-based CLI with a `main()` entrypoint guarded by `if __name__ == "__main__":`).
- All commands run inside the pixi env: `pixi run <cmd>`. Bare `pytest` will fail (`conftest.py` imports `allel`, `zarr`).
- The `.claude/hooks/ruff-on-edit.sh` hook auto-runs ruff after every Edit/Write — code must be ruff-clean.
- The `.pre-commit-config.yaml` runs ruff and other hooks on commit; if a commit fails on a hook, fix the underlying issue and create a NEW commit (do not `--amend` or `--no-verify`).
- `pixi.lock` is untracked and the `.claude/hooks/block-pixi-lock.sh` hook prevents direct edits — leave it alone.
- The active branch is `genotype_likelihoods`. All commits go there. Do not push.

---

## Task 1: Scaffold validation module + `common.py` utilities

**Files:**
- Create: `scripts/validation/__init__.py`
- Create: `scripts/validation/common.py`
- Create: `tests/test_validation.py`

- [ ] **Step 1: Create the empty package marker**

```bash
mkdir -p <repo-root>/scripts/validation
touch <repo-root>/scripts/validation/__init__.py
```

- [ ] **Step 2: Write the failing tests for `common.py`**

Create `tests/test_validation.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.validation import common


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
```

- [ ] **Step 3: Run tests — verify they fail with ImportError**

```bash
cd <repo-root> && pixi run pytest tests/test_validation.py -v
```

Expected: all 6 tests ERROR with `ModuleNotFoundError: No module named 'scripts.validation.common'` (or fail on missing attribute).

- [ ] **Step 4: Implement `common.py`**

Create `scripts/validation/common.py`:

```python
"""Shared utilities for path D validation scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Great-circle distance in km. Accepts scalars or arrays of equal shape."""
    lat1 = np.deg2rad(np.asarray(lat1, dtype=np.float64))
    lon1 = np.deg2rad(np.asarray(lon1, dtype=np.float64))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=np.float64))
    lon2 = np.deg2rad(np.asarray(lon2, dtype=np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    out = EARTH_RADIUS_KM * c
    if out.ndim == 0:
        return float(out)
    return out


def centroid_baseline(samples_locations: pd.DataFrame) -> tuple[float, float]:
    """Unweighted mean of (Lat, Lon) across all rows. Returns (lat, lon)."""
    return float(samples_locations["Lat"].mean()), float(samples_locations["Lon"].mean())


def parse_predlocs(path: str | Path) -> pd.DataFrame:
    """Read ReLocator's *_predlocs.txt (CSV-format despite the .txt extension)."""
    df = pd.read_csv(path)
    expected = {"sampleID", "x", "y"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"Predlocs file {path} missing required columns; got {list(df.columns)}"
        )
    return df[["sampleID", "x", "y"]].copy()


def gpu_round_robin(n_gpus: int, fold_idx: int) -> int:
    """Map a fold index to a GPU id via round-robin. Returns 0..n_gpus-1."""
    if n_gpus < 1:
        raise ValueError("n_gpus must be >= 1")
    return fold_idx % n_gpus
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
cd <repo-root> && pixi run pytest tests/test_validation.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Confirm existing test suite still green**

```bash
cd <repo-root> && pixi run pytest tests/test_input_extensions.py -v
```

Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
cd <repo-root>
git add scripts/validation/__init__.py scripts/validation/common.py tests/test_validation.py
git commit -m "$(cat <<'EOF'
Add validation module scaffold and shared utilities

Adds scripts/validation/{__init__.py,common.py} with haversine, centroid_baseline,
parse_predlocs, and gpu_round_robin helpers, plus 6 unit tests. First file under
the new validation package that will host the path D end-to-end test arc.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `build_inputs.py` — TSV join + validation

**Files:**
- Create: `scripts/validation/build_inputs.py`
- Modify: `tests/test_validation.py` (add tests for build_inputs)

- [ ] **Step 1: Add failing tests for `build_inputs`**

Append to `tests/test_validation.py`:

```python
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
        [sys.executable, "-m", "scripts.validation.build_inputs", *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd <repo-root> && pixi run pytest tests/test_validation.py -v -k build_inputs
```

Expected: all 4 build_inputs tests FAIL with `ModuleNotFoundError` for `scripts.validation.build_inputs`.

- [ ] **Step 3: Implement `build_inputs.py`**

Create `scripts/validation/build_inputs.py`:

```python
#!/usr/bin/env python3
"""Build bam.filelist and sample_data.txt for the path D validation pipeline.

Joins sample_table.tsv (sample_name, bam, site) with samples_locations.tsv
(SiteCode, Lat, Lon) and emits two files consumed by gl_to_locator.py and
locator. Hard-fails on missing BAMs, NaN coordinates, beagle column mismatch,
or unmatched site codes.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import pandas as pd


def _die(msg: str) -> None:
    print(f"build_inputs: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _read_beagle_header(beagle_path: Path) -> list[str]:
    with gzip.open(beagle_path, "rt") as fh:
        first = fh.readline().rstrip("\n")
    return first.split("\t")


def _validate_join(sample_table: pd.DataFrame, samples_locations: pd.DataFrame) -> pd.DataFrame:
    merged = sample_table.merge(
        samples_locations,
        left_on="site",
        right_on="SiteCode",
        how="left",
        validate="many_to_one",
    )
    unmatched = merged[merged["SiteCode"].isna()]["site"].tolist()
    if unmatched:
        _die(f"unmatched/unjoinable site codes: {sorted(set(unmatched))}")
    if merged["Lat"].isna().any() or merged["Lon"].isna().any():
        _die("NaN Lat/Lon after join — check samples_locations.tsv")
    return merged


def _validate_bams(merged: pd.DataFrame) -> None:
    missing = [b for b in merged["bam"] if not Path(b).exists()]
    if missing:
        _die(f"missing bam files (first 3 of {len(missing)}): {missing[:3]}")


def _validate_beagle_columns(beagle_path: Path, n_samples: int) -> None:
    header = _read_beagle_header(beagle_path)
    expected = 3 + 3 * n_samples
    if len(header) != expected:
        _die(
            f"beagle column count mismatch: header has {len(header)} cols, "
            f"expected {expected} ({n_samples} samples * 3 + 3 leading cols)"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-table", required=True, type=Path)
    p.add_argument("--samples-locations", required=True, type=Path)
    p.add_argument("--beagle", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    sample_table = pd.read_csv(args.sample_table, sep="\t")
    samples_locations = pd.read_csv(args.samples_locations, sep="\t")

    required_st = {"sample_name", "bam", "site"}
    if not required_st.issubset(sample_table.columns):
        _die(f"sample_table missing columns; need {required_st}")
    required_sl = {"SiteCode", "Lat", "Lon"}
    if not required_sl.issubset(samples_locations.columns):
        _die(f"samples_locations missing columns; need {required_sl}")

    merged = _validate_join(sample_table, samples_locations)
    _validate_bams(merged)
    _validate_beagle_columns(args.beagle, n_samples=len(merged))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    bam_filelist = args.out_dir / "bam.filelist"
    bam_filelist.write_text("\n".join(merged["bam"].tolist()) + "\n")

    sample_data = pd.DataFrame(
        {
            "x": merged["Lon"].astype(float),
            "y": merged["Lat"].astype(float),
            "sampleID": merged["sample_name"],
        }
    )
    sample_data.to_csv(args.out_dir / "sample_data.txt", sep="\t", index=False)

    log = args.out_dir / "build_log.txt"
    log.write_text(
        f"n_samples = {len(merged)}\n"
        f"n_sites = {merged['SiteCode'].nunique()}\n"
        f"sample_id_order = {merged['sample_name'].tolist()}\n"
        "ASSUMPTION: ANGSD was invoked with BAMs in sample_table row order. "
        "Beagle column count was verified but column-to-sample identity was not; "
        "see spec section 'Open assumptions / load-bearing risks'.\n"
    )

    print(f"build_inputs: OK — wrote {bam_filelist}, sample_data.txt, build_log.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd <repo-root> && pixi run pytest tests/test_validation.py -v
```

Expected: 10 passed (6 from Task 1 + 4 new).

- [ ] **Step 5: Commit**

```bash
cd <repo-root>
git add scripts/validation/build_inputs.py tests/test_validation.py
git commit -m "$(cat <<'EOF'
Add build_inputs.py for path D validation

Joins sample_table.tsv and samples_locations.tsv, validates BAM existence and
beagle column count, and writes bam.filelist + sample_data.txt + build_log.txt
into the validation output tree. 4 unit tests cover happy path and three
hard-fail conditions (missing BAM, column mismatch, unmatched site code).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Run `build_inputs.py` against the real test set (manual integration check)

**Files:**
- No new files. Generates `out/path_d_validation/inputs/{bam.filelist, sample_data.txt, build_log.txt}`.

- [ ] **Step 1: Execute against real data**

```bash
cd <repo-root>
pixi run python -m scripts.validation.build_inputs \
    --sample-table <test-data>/sample_table.tsv \
    --samples-locations <test-data>/samples_locations.tsv \
    --beagle <test-data>/snp_calling_global/combined.beagle.gz \
    --out-dir out/path_d_validation/inputs
```

Expected: prints `build_inputs: OK — wrote ...`. Exit code 0.

- [ ] **Step 2: Verify output shape**

```bash
wc -l out/path_d_validation/inputs/bam.filelist
head -3 out/path_d_validation/inputs/sample_data.txt
wc -l out/path_d_validation/inputs/sample_data.txt
```

Expected:
- `bam.filelist`: 54 lines.
- `sample_data.txt`: 55 lines (1 header + 54 rows). Header is `x\ty\tsampleID`. First data row is `Bg_AFB_03` at site AFB (Lat=60.1210, Lon=−149.3701), so `-149.3701\t60.1210\tBg_AFB_03`.

- [ ] **Step 3: Spot-check sample order vs `sample_table.tsv`**

```bash
diff <(awk -F'\t' 'NR>1 {print $1}' <test-data>/sample_table.tsv) \
     <(awk -F'\t' 'NR>1 {print $3}' out/path_d_validation/inputs/sample_data.txt)
```

Expected: empty diff. If not, abort and investigate before continuing — the LOSO sweep relies on this ordering.

- [ ] **Step 4: No commit** (these are runtime artifacts under `out/`, which is `.gitignore`-d).

---

## Task 4: `run_smoke.py` — Path A executor

**Files:**
- Create: `scripts/validation/run_smoke.py`

- [ ] **Step 1: Implement `run_smoke.py`**

This script is itself the integration test for the input pipeline; no separate unit test (would just shadow the real run). Keep the logic linear and observable.

```python
#!/usr/bin/env python3
"""Path A: end-to-end smoke test.

Runs gl_to_locator.py once in dosage mode, validates output, then trains
ReLocator on all 54 samples (no held-out) for a small number of epochs.
PASS if the pipeline completes cleanly and the network produces a
*_predlocs.txt with 54 rows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True)


def _result(out_dir: Path, status: str, **fields: object) -> int:
    payload = {"status": status, **fields}
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str))
    return 0 if status == "PASS" else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True, type=Path)
    p.add_argument("--beagle", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--gpu-number", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_data = args.inputs_dir / "sample_data.txt"
    bam_list = args.inputs_dir / "bam.filelist"

    gl_dosage = args.out_dir / "gl_dosage.txt"

    # --- Step 1: convert beagle -> dosage ---
    proc = _run(
        [
            "pixi", "run", "python", "scripts/gl_to_locator.py",
            "--beagle", str(args.beagle),
            "--bam_list", str(bam_list),
            "--out", str(gl_dosage),
            "--max_missing_frac", "0.50",
            "--gl_mode", "dosage",
        ]
    )
    if proc.returncode != 0:
        return _result(args.out_dir, "FAIL", stage="gl_to_locator", stderr=proc.stderr[-2000:])

    # If no sites survived the 0.50 threshold, retry once at 0.95.
    df = pd.read_csv(gl_dosage, sep="\t")
    if df.shape[1] - 1 < 1000:
        print(f"smoke: only {df.shape[1] - 1} sites at max_missing_frac=0.50; retrying at 0.95")
        proc = _run(
            [
                "pixi", "run", "python", "scripts/gl_to_locator.py",
                "--beagle", str(args.beagle),
                "--bam_list", str(bam_list),
                "--out", str(gl_dosage),
                "--max_missing_frac", "0.95",
                "--gl_mode", "dosage",
            ]
        )
        if proc.returncode != 0:
            return _result(args.out_dir, "FAIL", stage="gl_to_locator_retry", stderr=proc.stderr[-2000:])
        df = pd.read_csv(gl_dosage, sep="\t")

    n_sites = df.shape[1] - 1
    n_samples = df.shape[0]
    if n_samples != 54:
        return _result(args.out_dir, "FAIL", stage="dim_check", n_samples=n_samples)
    if n_sites < 1000:
        return _result(args.out_dir, "FAIL", stage="dim_check", n_sites=n_sites)

    # --- Step 2: cross-check sample IDs ---
    sd = pd.read_csv(sample_data, sep="\t")
    if df["sampleID"].tolist() != sd["sampleID"].tolist():
        return _result(args.out_dir, "FAIL", stage="sample_id_check",
                       reason="sampleID order in dosage file does not match sample_data.txt")

    # --- Step 3: train ReLocator ---
    run_dir = args.out_dir / "locator_run"
    run_dir.mkdir(exist_ok=True)
    out_prefix = run_dir / "smoke"
    proc = _run(
        [
            "pixi", "run", "locator",
            "--matrix", str(gl_dosage),
            "--sample_data", str(sample_data),
            "--out", str(out_prefix),
            "--max_epochs", str(args.max_epochs),
            "--patience", "10",
            "--seed", "42",
            "--gpu_number", str(args.gpu_number),
            "--no_verbose",
        ]
    )
    (run_dir / "stdout.log").write_text(proc.stdout)
    (run_dir / "stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        return _result(args.out_dir, "FAIL", stage="locator", stderr=proc.stderr[-2000:])

    predlocs = Path(f"{out_prefix}_predlocs.txt")
    if not predlocs.exists():
        return _result(args.out_dir, "FAIL", stage="predlocs_missing", expected=str(predlocs))

    pred = pd.read_csv(predlocs)
    if len(pred) != 54:
        return _result(args.out_dir, "FAIL", stage="predlocs_rowcount", got=len(pred))

    return _result(
        args.out_dir,
        "PASS",
        n_samples=n_samples,
        n_sites=n_sites,
        predlocs_rows=len(pred),
    )


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-import the new module to catch syntax errors before the long run**

```bash
cd <repo-root> && pixi run python -c "import scripts.validation.run_smoke"
```

Expected: silent. No traceback.

- [ ] **Step 3: Commit (before executing the run, so the script is captured even if Path A reveals a bug)**

```bash
cd <repo-root>
git add scripts/validation/run_smoke.py
git commit -m "$(cat <<'EOF'
Add run_smoke.py for path D validation (path A)

Runs gl_to_locator.py and a stock-defaults ReLocator training over all 54
samples, validates dimensions and sample-ID alignment, and emits result.json
with PASS/FAIL plus diagnostics. The ReLocator call uses a small max_epochs
(50) and patience (10) to keep the smoke run quick; a longer run is the LOSO
sweep in the next tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Execute Path A (smoke run on real data)

**Files:**
- No new files. Produces `out/path_d_validation/smoke/`.

- [ ] **Step 1: Run path A**

```bash
cd <repo-root>
pixi run python -m scripts.validation.run_smoke \
    --inputs-dir out/path_d_validation/inputs \
    --beagle <test-data>/snp_calling_global/combined.beagle.gz \
    --out-dir out/path_d_validation/smoke \
    --gpu-number 0 2>&1 | tee out/path_d_validation/smoke/stdout.log
```

Expected: ends with a JSON block whose top-level `"status"` is `"PASS"`. Wall-clock ≤ 30 min.

- [ ] **Step 2: Inspect result**

```bash
cat out/path_d_validation/smoke/result.json
ls out/path_d_validation/smoke/locator_run/
```

Expected: `result.json` shows PASS, `n_samples=54`, `n_sites>=1000`, `predlocs_rows=54`. `locator_run/` contains the training artifacts and `smoke_predlocs.txt`.

- [ ] **Step 3: If FAIL, stop**

If `status` is `FAIL`, do not proceed to Task 6. Read `stage` and the captured stderr; debug. Common causes:
- All sites dropped at both missingness thresholds → flat-GL-dominated dataset; spec calls for `--gl_mode full_gl` here as a fallback (escalate to user).
- Sample-ID order mismatch → ANGSD BAM order ≠ `sample_table` order; need to recover the actual order (may require re-running ANGSD or contacting whoever produced the beagle).
- locator failure → read `out/path_d_validation/smoke/locator_run/stderr.log`.

- [ ] **Step 4: No commit** (runtime artifacts).

---

## Task 6: `run_loso.py` — paths B & C orchestrator

**Files:**
- Create: `scripts/validation/run_loso.py`

This script is parameterized by `--gl-mode`; one invocation produces 12 folds for one mode. We test the fold-data-prep helpers as units; the orchestrator itself is exercised by the real sweep.

- [ ] **Step 1: Add fold-prep tests**

Append to `tests/test_validation.py`:

```python
def test_blank_holdout_locations_only_for_target_site(tmp_path: Path):
    from scripts.validation import run_loso

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
```

- [ ] **Step 2: Run test — verify it fails**

```bash
cd <repo-root> && pixi run pytest tests/test_validation.py::test_blank_holdout_locations_only_for_target_site -v
```

Expected: FAIL with `ModuleNotFoundError` or `AttributeError`.

- [ ] **Step 3: Implement `run_loso.py`**

Create `scripts/validation/run_loso.py`:

```python
#!/usr/bin/env python3
"""Paths B & C: leave-one-site-out (LOSO) sweep.

Generates the feature matrix once with gl_to_locator.py, then iterates over
the 12 sites, blanking each site's coordinates in turn and training ReLocator.
3 folds run concurrently, one per GPU (round-robin via --gpu_number).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.validation import common


def blank_holdout_locations(
    sample_data: pd.DataFrame, sample_table: pd.DataFrame, target_site: str
) -> pd.DataFrame:
    """Return a copy of sample_data with x,y set to 'NA' for samples at target_site."""
    held = sample_table.loc[sample_table["site"] == target_site, "sample_name"].tolist()
    out = sample_data.copy()
    mask = out["sampleID"].isin(held)
    out.loc[mask, "x"] = "NA"
    out.loc[mask, "y"] = "NA"
    return out


def _run(cmd: list[str], log_path: Path) -> int:
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}")
    return proc.returncode


def _run_one_fold(
    *,
    fold_idx: int,
    site_code: str,
    site_lat: float,
    site_lon: float,
    feature_matrix: Path,
    sample_data_full: pd.DataFrame,
    sample_table: pd.DataFrame,
    out_dir: Path,
    gl_mode: str,
    n_gpus: int,
    max_epochs: int,
    timeout_s: int,
) -> dict:
    fold_dir = out_dir / site_code
    fold_dir.mkdir(parents=True, exist_ok=True)

    held_sd = blank_holdout_locations(sample_data_full, sample_table, target_site=site_code)
    sd_path = fold_dir / "sample_data.txt"
    held_sd.to_csv(sd_path, sep="\t", index=False)

    out_prefix = fold_dir / "locator"
    gpu = common.gpu_round_robin(n_gpus, fold_idx)
    cmd = [
        "pixi", "run", "locator",
        "--matrix", str(feature_matrix),
        "--sample_data", str(sd_path),
        "--out", str(out_prefix),
        "--max_epochs", str(max_epochs),
        "--patience", "30",
        "--seed", "42",
        "--gpu_number", str(gpu),
        "--no_verbose",
    ]
    log_path = fold_dir / "locator.log"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        log_path.write_text(f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}")
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        log_path.write_text(f"TIMEOUT after {timeout_s}s\nSTDOUT:\n{e.stdout or ''}\n\nSTDERR:\n{e.stderr or ''}")
        rc = -1

    fold_result: dict = {
        "site": site_code,
        "true_lat": site_lat,
        "true_lon": site_lon,
        "gl_mode": gl_mode,
        "gpu": gpu,
    }
    predlocs = Path(f"{out_prefix}_predlocs.txt")
    if rc != 0 or not predlocs.exists():
        fold_result["status"] = "FAILED"
        fold_result["reason"] = f"locator returncode={rc}, predlocs_exists={predlocs.exists()}"
    else:
        try:
            pred = common.parse_predlocs(predlocs)
            held_ids = sample_table.loc[sample_table["site"] == site_code, "sample_name"].tolist()
            held_pred = pred[pred["sampleID"].isin(held_ids)].copy()
            held_pred["error_km"] = common.haversine(
                held_pred["y"].values, held_pred["x"].values, site_lat, site_lon
            )
            fold_result["status"] = "OK"
            fold_result["n_held_out"] = int(len(held_pred))
            fold_result["mean_pred_lat"] = float(held_pred["y"].mean())
            fold_result["mean_pred_lon"] = float(held_pred["x"].mean())
            fold_result["mean_error_km"] = float(held_pred["error_km"].mean())
            fold_result["median_error_km"] = float(held_pred["error_km"].median())
            fold_result["per_sample"] = held_pred.to_dict(orient="records")
        except (ValueError, pd.errors.EmptyDataError) as exc:
            fold_result["status"] = "FAILED"
            fold_result["reason"] = f"predlocs parse error: {exc}"

    (fold_dir / "fold_result.json").write_text(json.dumps(fold_result, indent=2, default=str))
    return fold_result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True, type=Path)
    p.add_argument("--samples-locations", required=True, type=Path)
    p.add_argument("--sample-table", required=True, type=Path)
    p.add_argument("--beagle", required=True, type=Path)
    p.add_argument("--gl-mode", required=True, choices=["dosage", "full_gl"])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-parallel", type=int, default=3)
    p.add_argument("--n-gpus", type=int, default=3)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--timeout-s", type=int, default=1800)
    p.add_argument("--max-missing-frac", type=float, default=0.50)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    feature_matrix = args.out_dir / f"gl_{args.gl_mode}.txt"
    if not feature_matrix.exists():
        rc = _run(
            [
                "pixi", "run", "python", "scripts/gl_to_locator.py",
                "--beagle", str(args.beagle),
                "--bam_list", str(args.inputs_dir / "bam.filelist"),
                "--out", str(feature_matrix),
                "--max_missing_frac", str(args.max_missing_frac),
                "--gl_mode", args.gl_mode,
            ],
            log_path=args.out_dir / "gl_to_locator.log",
        )
        if rc != 0:
            print(f"run_loso: gl_to_locator failed with rc={rc}", file=sys.stderr)
            return 1

    sample_data_full = pd.read_csv(args.inputs_dir / "sample_data.txt", sep="\t")
    sample_table = pd.read_csv(args.sample_table, sep="\t")
    samples_locations = pd.read_csv(args.samples_locations, sep="\t")

    site_rows = list(samples_locations.itertuples(index=False))

    print(f"run_loso: starting {len(site_rows)} folds for gl_mode={args.gl_mode} "
          f"with n_parallel={args.n_parallel} on {args.n_gpus} GPUs")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_parallel) as pool:
        futures = []
        for idx, row in enumerate(site_rows):
            futures.append(
                pool.submit(
                    _run_one_fold,
                    fold_idx=idx,
                    site_code=row.SiteCode,
                    site_lat=float(row.Lat),
                    site_lon=float(row.Lon),
                    feature_matrix=feature_matrix,
                    sample_data_full=sample_data_full,
                    sample_table=sample_table,
                    out_dir=args.out_dir,
                    gl_mode=args.gl_mode,
                    n_gpus=args.n_gpus,
                    max_epochs=args.max_epochs,
                    timeout_s=args.timeout_s,
                )
            )
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            print(f"run_loso: fold {res['site']} -> {res['status']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests — verify the fold-prep test passes and the rest still pass**

```bash
cd <repo-root> && pixi run pytest tests/test_validation.py -v
```

Expected: 11 passed (10 from earlier tasks + the fold-prep test).

- [ ] **Step 5: Commit**

```bash
cd <repo-root>
git add scripts/validation/run_loso.py tests/test_validation.py
git commit -m "$(cat <<'EOF'
Add run_loso.py for path D validation (paths B/C)

Leave-one-site-out orchestrator parameterized by --gl-mode. Builds the feature
matrix once via gl_to_locator.py, then runs 12 folds via ProcessPoolExecutor
with up to 3 in flight (one per A100). Each fold blanks the held-out site's
coordinates, calls locator, parses predlocs, and writes fold_result.json with
haversine error against the true site centroid. Failed folds are recorded but
do not abort the sweep.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Execute Path B (LOSO sweep, dosage mode)

**Files:**
- No new files. Produces `out/path_d_validation/loso_dosage/`.

- [ ] **Step 1: Run the dosage sweep**

```bash
cd <repo-root>
pixi run python -m scripts.validation.run_loso \
    --inputs-dir out/path_d_validation/inputs \
    --samples-locations <test-data>/samples_locations.tsv \
    --sample-table <test-data>/sample_table.tsv \
    --beagle <test-data>/snp_calling_global/combined.beagle.gz \
    --gl-mode dosage \
    --out-dir out/path_d_validation/loso_dosage \
    --max-epochs 500 2>&1 | tee out/path_d_validation/loso_dosage/sweep.log
```

Expected: 12 fold lines printed (`run_loso: fold <SITE> -> OK`); wall-clock ≈ (12 / 3) × per-fold time. Expect 30 min – 2 hr depending on convergence.

- [ ] **Step 2: Verify all 12 folds produced JSON**

```bash
ls out/path_d_validation/loso_dosage/*/fold_result.json | wc -l
grep -l '"status": "FAILED"' out/path_d_validation/loso_dosage/*/fold_result.json || echo "no failed folds"
```

Expected: `12`. If any folds failed, read their `locator.log` and `fold_result.json` reason; decide case-by-case whether to re-run that fold or accept the failure into `summary.md`.

- [ ] **Step 3: No commit** (runtime artifacts).

---

## Task 8: Execute Path C runs (LOSO sweep, full_gl mode)

**Files:**
- No new files. Produces `out/path_d_validation/loso_full_gl/`.

- [ ] **Step 1: Run the full_gl sweep**

```bash
cd <repo-root>
pixi run python -m scripts.validation.run_loso \
    --inputs-dir out/path_d_validation/inputs \
    --samples-locations <test-data>/samples_locations.tsv \
    --sample-table <test-data>/sample_table.tsv \
    --beagle <test-data>/snp_calling_global/combined.beagle.gz \
    --gl-mode full_gl \
    --out-dir out/path_d_validation/loso_full_gl \
    --max-epochs 500 2>&1 | tee out/path_d_validation/loso_full_gl/sweep.log
```

Expected: 12 fold lines, all `OK` ideally. The full_gl feature matrix is 3× wider than dosage; expect ~3× the per-fold wall-clock.

- [ ] **Step 2: Verify all 12 folds**

```bash
ls out/path_d_validation/loso_full_gl/*/fold_result.json | wc -l
grep -l '"status": "FAILED"' out/path_d_validation/loso_full_gl/*/fold_result.json || echo "no failed folds"
```

Expected: `12`. Same failure-handling rule as Task 7.

- [ ] **Step 3: No commit** (runtime artifacts).

---

## Task 9: `summarize.py` — aggregate, plot, and write the report

**Files:**
- Create: `scripts/validation/summarize.py`
- Modify: `tests/test_validation.py` (add tests for the summary helpers)

- [ ] **Step 1: Add tests for the summary aggregation function**

Add `import json` to the imports at the top of `tests/test_validation.py` (it was not imported in earlier tasks). Then append the test:

```python
def test_aggregate_folds(tmp_path: Path):
    from scripts.validation import summarize

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
```

- [ ] **Step 2: Run test — verify it fails**

```bash
cd <repo-root> && pixi run pytest tests/test_validation.py::test_aggregate_folds -v
```

Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement `summarize.py`**

Create `scripts/validation/summarize.py`:

```python
#!/usr/bin/env python3
"""Aggregate path D fold outputs into summary.tsv, summary.md, and a plot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.validation import common


def aggregate_folds(loso_dir: Path, gl_mode: str) -> pd.DataFrame:
    rows = []
    for fold_json in sorted(loso_dir.glob("*/fold_result.json")):
        rows.append(json.loads(fold_json.read_text()))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["gl_mode"] = gl_mode
    return df


def _format_smoke(smoke_dir: Path) -> str:
    p = smoke_dir / "result.json"
    if not p.exists():
        return "**Path A:** result.json missing — smoke run not executed."
    r = json.loads(p.read_text())
    return f"**Path A:** {r.get('status')} — {json.dumps({k: v for k, v in r.items() if k != 'status'})}"


def _format_loso(df: pd.DataFrame, centroid_err_km: float, label: str) -> tuple[str, float]:
    if df.empty:
        return f"**{label}:** no fold results.", float("nan")
    ok = df[df["status"] == "OK"]
    if ok.empty:
        return f"**{label}:** all {len(df)} folds FAILED.", float("nan")
    median_err = float(ok["median_error_km"].median())
    ratio = median_err / centroid_err_km if centroid_err_km else float("nan")

    cols = ["site", "n_held_out", "true_lat", "true_lon",
            "mean_pred_lat", "mean_pred_lon", "mean_error_km", "median_error_km", "status"]
    table = ok[cols].to_markdown(index=False, floatfmt=".3f")
    failed = df[df["status"] != "OK"]
    failed_section = ""
    if not failed.empty:
        failed_section = "\n\nFailed folds:\n" + failed[["site", "reason"]].to_markdown(index=False)

    body = (
        f"**{label}:** median per-fold error = {median_err:.1f} km; "
        f"centroid baseline = {centroid_err_km:.1f} km; "
        f"ratio = {ratio:.3f} ({'PASS' if ratio < 0.5 else 'FAIL — > 0.5'}).\n\n"
        f"{table}{failed_section}"
    )
    return body, ratio


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--samples-locations", required=True, type=Path)
    p.add_argument("--sample-table", required=True, type=Path)
    args = p.parse_args()

    summary_dir = args.out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    samples_locations = pd.read_csv(args.samples_locations, sep="\t")
    centroid_lat, centroid_lon = common.centroid_baseline(samples_locations)
    # Centroid baseline error: mean haversine from each site centroid to the global centroid.
    centroid_err_km = float(
        common.haversine(
            samples_locations["Lat"].values,
            samples_locations["Lon"].values,
            centroid_lat,
            centroid_lon,
        ).mean()
    )

    dosage_df = aggregate_folds(args.out_dir / "loso_dosage", "dosage")
    full_gl_df = aggregate_folds(args.out_dir / "loso_full_gl", "full_gl")

    combined = pd.concat([dosage_df, full_gl_df], ignore_index=True)
    if not combined.empty:
        combined.drop(columns=["per_sample"], errors="ignore").to_csv(
            summary_dir / "summary.tsv", sep="\t", index=False
        )

    # Plot: dosage vs full_gl per-site mean error.
    if not dosage_df.empty and not full_gl_df.empty:
        merged = dosage_df.merge(
            full_gl_df, on="site", suffixes=("_dosage", "_full_gl")
        )
        ok = merged[(merged["status_dosage"] == "OK") & (merged["status_full_gl"] == "OK")]
        if not ok.empty:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(ok["mean_error_km_dosage"], ok["mean_error_km_full_gl"], s=40)
            for _, row in ok.iterrows():
                ax.annotate(row["site"],
                            (row["mean_error_km_dosage"], row["mean_error_km_full_gl"]),
                            fontsize=8)
            lim = max(ok["mean_error_km_dosage"].max(), ok["mean_error_km_full_gl"].max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=0.5)
            ax.set_xlabel("dosage mean error (km)")
            ax.set_ylabel("full_gl mean error (km)")
            ax.set_title("Per-site mean prediction error: dosage vs full_gl")
            fig.tight_layout()
            fig.savefig(summary_dir / "dosage_vs_full_gl.png", dpi=120)
            plt.close(fig)

    smoke_block = _format_smoke(args.out_dir / "smoke")
    dosage_block, _dr = _format_loso(dosage_df, centroid_err_km, "Path B (dosage LOSO)")
    full_gl_block, _fr = _format_loso(full_gl_df, centroid_err_km, "Path C (full_gl LOSO)")

    md = (
        f"# Path D validation summary\n\n"
        f"Centroid baseline (mean across-site haversine to global centroid) = {centroid_err_km:.1f} km.\n\n"
        f"{smoke_block}\n\n"
        f"---\n\n"
        f"{dosage_block}\n\n"
        f"---\n\n"
        f"{full_gl_block}\n\n"
        f"---\n\n"
        f"## Recommendation checkpoint\n\n"
        f"This report is a checkpoint, not an automated decision. Read Path A "
        f"status, Path B PASS/FAIL ratio, and the dosage vs full_gl comparison "
        f"in Path C, then decide whether the genotype_likelihoods branch is "
        f"ready to push.\n"
    )
    (summary_dir / "summary.md").write_text(md)
    print(f"summarize: wrote {summary_dir}/summary.{{tsv,md}} and dosage_vs_full_gl.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run all tests — verify everything passes**

```bash
cd <repo-root> && pixi run pytest tests/test_validation.py tests/test_input_extensions.py -v
```

Expected: 18 passed (12 validation + 6 input_extensions).

- [ ] **Step 5: Generate the actual summary against the real run outputs**

```bash
cd <repo-root>
pixi run python -m scripts.validation.summarize \
    --out-dir out/path_d_validation \
    --samples-locations <test-data>/samples_locations.tsv \
    --sample-table <test-data>/sample_table.tsv
```

Expected: prints the summarize line; emits `summary.tsv`, `summary.md`, `dosage_vs_full_gl.png`.

- [ ] **Step 6: Commit `summarize.py` and the test additions** (NOT the runtime outputs)

```bash
cd <repo-root>
git add scripts/validation/summarize.py tests/test_validation.py
git commit -m "$(cat <<'EOF'
Add summarize.py for path D validation

Aggregates smoke/result.json plus per-fold fold_result.json files for both GL
modes into summary.tsv, summary.md, and a dosage-vs-full_gl scatter plot.
Computes the centroid-baseline error for Path B's PASS/FAIL ratio. The
summary.md is a human checkpoint, not an automated decision.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final review checkpoint

**Files:**
- No code changes. Read-only review.

- [ ] **Step 1: Read the summary**

```bash
cat <repo-root>/out/path_d_validation/summary/summary.md
```

Walk through:
- Path A status — PASS expected.
- Path B ratio — PASS if median per-fold error / centroid baseline < 0.5.
- Path C — read the table and the scatter (`dosage_vs_full_gl.png`); decide which mode to recommend as the default in CLAUDE.md.
- Failed folds — interpret per-fold reasons; decide if any need a re-run.

- [ ] **Step 2: Inspect the plot**

```bash
ls -la <repo-root>/out/path_d_validation/summary/dosage_vs_full_gl.png
```

Open the PNG (or scp to a viewing host) and check whether one mode is systematically below the y=x line.

- [ ] **Step 3: Decide on push readiness**

Hand the summary to the user. The user makes the call:
- All three paths clean → consider opening a feature branch on the remote.
- Path A or B failed → debug; do not push.
- Path C inconclusive → fine; push, document the indeterminacy as a TODO.

- [ ] **Step 4: No commit** — this is a read-only checkpoint.

---

## Self-review notes

- Spec section "Architecture" enumerates 5 files; Tasks 1, 2, 4, 6, 9 each create one. ✓
- Spec section "Acceptance criteria" — A enforced in Task 4 (`run_smoke.py`) and Task 5 (executes); B's ratio < 0.5 enforced in Task 9's `_format_loso` PASS/FAIL string; C's descriptive output covered by `_format_loso` table + plot. ✓
- Spec section "Testing" — 6 listed unit tests are split across Tasks 1, 2, 6, 9; all 6 covered (haversine, centroid, parse_predlocs, build_inputs basic, build_inputs missing-bam, build_inputs column-mismatch) plus added build_inputs unjoinable-site, fold-prep blanking, and aggregate-folds tests. ✓
- Spec section "Build sequence" — matches Tasks 3 → 5 → 7 → 8 → 9 → 10. ✓
- Spec section "Open assumptions" (BAM order) — surfaced in `build_log.txt` per Task 2 and noted in Task 5 failure debugging. ✓
- All tasks use exact file paths and produce committable code with no TBDs.
