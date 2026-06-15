# Feature-Branch Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the `genotype_likelihoods` branch to a feature-branch handoff state — restructure into merge-zone + validation-zone, switch Path B's metric to coast-projected, add Figure 1 (test-data noise curve) and Figure 2 (balanus map), drop microsats, add a small user doc, update existing specs.

**Architecture:** Restructure first (move `scripts/validation/` → `validation/`, `tests/test_validation.py` → `validation/tests/`, `docs/superpowers/` → `validation/docs/`); then build new pieces in the new layout. Imports become `from validation import X`. Figures are committed under `validation/figures/`. The merge zone (locator/, scripts/gl_to_locator.py, tests/, docs/genotype_likelihoods.md) stays small and clean.

**Tech Stack:** Python 3.12, pandas, numpy, matplotlib, cartopy 0.25.0 (already in pixi env), pytest via `pixi run pytest`. Existing pixi env (TF 2.19.1, 3× A100s).

**Spec:** `docs/superpowers/specs/2026-05-03-pr-cleanup-arc-design.md` (this spec moves to `validation/docs/specs/` as part of step 10).

---

## Conventions (apply to every task)

- All commands run inside the pixi env: `pixi run <cmd>`. Bare `pytest` will fail (`tests/conftest.py` imports `allel`/`zarr`).
- The `.claude/hooks/ruff-on-edit.sh` hook auto-runs ruff after every Edit/Write — code must be ruff-clean.
- The `.pre-commit-config.yaml` runs ruff on commit; if a commit fails on a hook, fix the underlying issue and create a NEW commit (do not `--amend` or `--no-verify`).
- Active branch: `genotype_likelihoods`. All commits go there. Do not push.
- `out/` is gitignored; `validation/` is committed. After step 1, `summarize.py` etc. write to `validation/figures/` and `validation/summary/` rather than `out/`.

---

## Task 1: Restructure — move validation tooling to top-level `validation/`

**Files:**
- Rename: `scripts/validation/` → `validation/` (10 files moved via `git mv`)
- Rename: `tests/test_validation.py` → `validation/tests/test_validation.py`
- Rename: `docs/superpowers/` → `validation/docs/` (entire tree)
- Create: `validation/tests/__init__.py`, `validation/tests/conftest.py`
- Modify: `tests/conftest.py` (revert sys.path hack)
- Modify: every file under `validation/` that imports from `scripts.validation`
- Modify: `pyproject.toml` (add `validation/tests/` to pytest testpaths if needed)

- [ ] **Step 1: Create the new directory structure with `git mv`**

```bash
cd <repo-root>
git mv scripts/validation validation
mkdir -p validation/tests
git mv tests/test_validation.py validation/tests/test_validation.py
git mv docs/superpowers validation/docs
```

After this:
- `validation/{__init__.py,common.py,build_inputs.py,run_smoke.py,run_loso.py,summarize.py,thin_beagle.py,synth_gls.py}` exist
- `validation/tests/test_validation.py` exists
- `validation/docs/{specs,plans}/` exist with all the prior spec/plan files
- `scripts/validation/`, `tests/test_validation.py`, `docs/superpowers/` no longer exist

- [ ] **Step 2: Add empty package markers**

```bash
touch validation/tests/__init__.py
```

- [ ] **Step 3: Create `validation/tests/conftest.py` with the sys.path hack**

```python
"""Conftest for validation tests — adds project root to sys.path so
`from validation import …` works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

- [ ] **Step 4: Revert `tests/conftest.py` to its pre-patch form**

The 6 lines we added in commit `fa04126` are no longer needed in the merge-zone tests (none of `tests/test_*.py` import from validation modules now). Open `tests/conftest.py` and remove these lines:

```python
import sys
from pathlib import Path
```

and:

```python
# Add project root to path so scripts package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

The file's first lines should now match the original:

```python
"""Shared test fixtures and utilities for locator tests"""

import allel
import numpy as np
import pandas as pd
import pytest
```

- [ ] **Step 5: Update imports inside `validation/` to use the new package path**

The pattern to find: `from scripts.validation import …` and `from scripts.validation.<module> import …`.

```bash
cd <repo-root>
grep -rn "from scripts.validation" validation/ tests/
```

For each match, change `scripts.validation` → `validation`. The expected hits are in:

- `validation/run_loso.py` — has `from scripts.validation import common`
- `validation/summarize.py` — has `from scripts.validation import common`
- `validation/tests/test_validation.py` — multiple matches (`from scripts.validation import common`, `from scripts.validation import run_loso`, `from scripts.validation import summarize`)

Edit each match in place. Example for `validation/run_loso.py`:

Before:
```python
from scripts.validation import common
```
After:
```python
from validation import common
```

- [ ] **Step 6: Update `python -m` invocations inside `validation/` scripts**

`validation/run_smoke.py` and `validation/run_loso.py` invoke other scripts via subprocess `["pixi", "run", "python", "scripts/gl_to_locator.py", ...]`. The path `scripts/gl_to_locator.py` is unchanged (gl_to_locator stayed in scripts/), so those invocations still work. NO CHANGE NEEDED for `gl_to_locator` invocations.

But `validation/build_inputs.py` is invoked as `python -m scripts.validation.build_inputs` from `validation/tests/test_validation.py`. Find that line and update:

Before:
```python
[sys.executable, "-m", "scripts.validation.build_inputs", *args],
```
After:
```python
[sys.executable, "-m", "validation.build_inputs", *args],
```

- [ ] **Step 7: Update pyproject.toml pytest config to include `validation/tests/`**

```bash
grep -n "testpaths\|pythonpath\|\[tool.pytest" pyproject.toml
```

The `[tool.pytest.ini_options]` section likely has `testpaths = ["tests"]` or similar. Add `validation/tests` to the list:

Before:
```toml
testpaths = ["tests"]
```
After:
```toml
testpaths = ["tests", "validation/tests"]
```

If pyproject doesn't currently set `testpaths` (so pytest uses defaults), add the section:

```toml
[tool.pytest.ini_options]
testpaths = ["tests", "validation/tests"]
```

If unsure: read pyproject.toml first and adjust the existing block.

- [ ] **Step 8: Run the full test set in the new layout**

```bash
cd <repo-root>
pixi run pytest tests/test_data_loading.py tests/test_input_extensions.py validation/tests/test_validation.py -n 0 -v
```

Expected: 22 passed (existing data_loading subset + 6 input_extensions + 12 validation = depends on baseline counts; the key is "all green, no import errors").

- [ ] **Step 9: Run a smoke import check**

```bash
cd <repo-root>
pixi run python -c "
from validation import common, build_inputs, run_smoke, run_loso, summarize, thin_beagle, synth_gls
print('all validation imports OK')
"
```

Expected: `all validation imports OK`. If anything fails, find the missing/wrong import and fix.

- [ ] **Step 10: Commit the restructure**

```bash
cd <repo-root>
git add -A
git commit -m "$(cat <<'EOF'
Restructure: move validation tooling to top-level validation/

Step 1 of the feature-branch cleanup arc. Moves scripts/validation/,
tests/test_validation.py, and docs/superpowers/ under a new top-level
validation/ directory. Imports updated from `scripts.validation` to
`validation`. tests/conftest.py reverts to pre-patch form (the sys.path
hack moved to validation/tests/conftest.py where it's actually needed).

This separates the merge zone (locator/, scripts/gl_to_locator.py,
tests/, docs/) from the validation evidence zone (validation/) so the
feature-branch handoff is clearly demarcated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If pre-commit hooks fail, fix and create a NEW commit.

---

## Task 2: `validation/coastline.py` — TDD

**Files:**
- Create: `validation/coastline.py`
- Modify: `validation/tests/test_validation.py` (append 4 new tests)

- [ ] **Step 1: Append 4 failing tests to `validation/tests/test_validation.py`**

Append at end of file:

```python
# ---------------------------------------------------------------------------
# Coastline projection
# ---------------------------------------------------------------------------


def test_coast_index_zero_at_first_site():
    from validation import coastline

    cum_km, points = coastline.build_coast_index()
    # First site = origin of polyline; cum km is 0.
    assert cum_km[0] == pytest.approx(0.0)
    # Polyline length is in a sane range for the AK→CA span.
    assert 2500 <= cum_km[-1] <= 3500


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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd <repo-root>
pixi run pytest validation/tests/test_validation.py -k "coast" -n 0 -v
```

Expected: 4 tests FAIL with `ModuleNotFoundError: No module named 'validation.coastline'`.

- [ ] **Step 3: Implement `validation/coastline.py`**

Create with EXACTLY this content. The 12-site polyline is hardcoded (matches `test_data/samples_locations.tsv`):

```python
"""Coastline-projected error metric for the barnacle test set.

Barnacles are intertidal — predictions inland of the coast are biologically
meaningless and the standard great-circle haversine error overstates the
real prediction error for any prediction that lands in the right latitude
band but offshore or inland. This module models the coast as a piecewise-
linear polyline through the 12 sample sites (N→S) and projects each
prediction onto that polyline; the meaningful error is the along-polyline
distance from the projection to the true site.
"""

from __future__ import annotations

import numpy as np

from validation.common import haversine

# 12 sample sites in N→S order, from test_data/samples_locations.tsv.
# Format: (site_code, lat, lon).
BARNACLE_COAST: list[tuple[str, float, float]] = [
    ("01-AFB-AK", 60.1210, -149.3701),
    ("02-DIG-BC", 54.2833, -130.4167),
    ("03-FHL-WA", 48.5454, -123.0133),
    ("04-RIC-WA", 47.7635, -122.3862),
    ("05-WES-WA", 46.9123, -124.1103),
    ("06-MEA-OR", 45.4859, -123.9746),
    ("07-CPE-OR", 44.2804, -124.1119),
    ("08-BOB-OR", 44.2438, -124.1141),
    ("09-OMB-OR", 43.3447, -124.3224),
    ("10-PAR-CA", 38.9557, -123.7414),
    ("11-HOP-CA", 36.6002, -121.8947),
    ("12-GOL-CA", 34.4170, -119.8320),
]


def build_coast_index(sites=BARNACLE_COAST):
    """Return (cum_km, points).

    cum_km[i] = total polyline distance (km) from sites[0] to sites[i],
    measured as the sum of great-circle distances along consecutive segments.

    points: ndarray of shape (n, 2), rows are (lat, lon).
    """
    if not sites:
        raise ValueError("sites must be non-empty")
    pts = np.array([[lat, lon] for _, lat, lon in sites], dtype=np.float64)
    cum = np.zeros(len(pts), dtype=np.float64)
    for i in range(1, len(pts)):
        cum[i] = cum[i - 1] + float(
            haversine(pts[i - 1, 0], pts[i - 1, 1], pts[i, 0], pts[i, 1])
        )
    return cum, pts


def project_to_coast(pred_lat, pred_lon, cum_km, points):
    """Project (pred_lat, pred_lon) onto the coast polyline.

    Returns (coast_pos_km, offshore_km).

    coast_pos_km: along-polyline km position of the projected point (0 at
    the first site, cum_km[-1] at the last).
    offshore_km: great-circle distance from the prediction to its
    projection on the polyline.

    Uses planar (lat-lon-as-Cartesian) projection per segment — accurate
    enough for prediction errors of <1000 km.
    """
    best_d = float("inf")
    best_pos = 0.0
    for i in range(len(points) - 1):
        a_lat, a_lon = points[i]
        b_lat, b_lon = points[i + 1]
        ab_lat = b_lat - a_lat
        ab_lon = b_lon - a_lon
        ap_lat = pred_lat - a_lat
        ap_lon = pred_lon - a_lon
        denom = ab_lat * ab_lat + ab_lon * ab_lon
        if denom == 0:
            t = 0.0
        else:
            t = (ap_lat * ab_lat + ap_lon * ab_lon) / denom
            t = max(0.0, min(1.0, t))
        proj_lat = a_lat + t * ab_lat
        proj_lon = a_lon + t * ab_lon
        d = float(haversine(pred_lat, pred_lon, proj_lat, proj_lon))
        if d < best_d:
            best_d = d
            seg_len = float(haversine(a_lat, a_lon, b_lat, b_lon))
            best_pos = float(cum_km[i] + t * seg_len)
    return best_pos, best_d


def along_coast_distance(pred_lat, pred_lon, true_site_idx, cum_km, points):
    """|coast_pos_km - cum_km[true_site_idx]| in km."""
    coast_pos, _ = project_to_coast(pred_lat, pred_lon, cum_km, points)
    return float(abs(coast_pos - cum_km[true_site_idx]))
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd <repo-root>
pixi run pytest validation/tests/test_validation.py -n 0 -v
```

Expected: all tests pass (12 prior + 4 new coastline = 16).

- [ ] **Step 5: Commit**

```bash
git add validation/coastline.py validation/tests/test_validation.py
git commit -m "$(cat <<'EOF'
Add validation/coastline.py: coast-projected error metric

Models the barnacle coast as a piecewise-linear polyline through the 12
sample sites in N→S order. Projects predictions onto the polyline and
returns along-polyline distance to the true site as the meaningful error
metric, since barnacles are intertidal and inland/offshore lat-lon error
is biologically uninterpretable.

4 unit tests: cum-km starts at 0 and ends in [2500, 3500] km; projecting
a known site lands at its index with offshore < 1 km; projecting a point
~50 km offshore yields ~50 km offshore_km; along-coast distance is
order-preserving.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update `validation/run_loso.py` to emit along-coast metric

**Files:**
- Modify: `validation/run_loso.py`

- [ ] **Step 1: Add imports and module-level coast index**

Open `validation/run_loso.py`. Near the top, after `from validation import common`, add:

```python
from validation.coastline import (
    BARNACLE_COAST,
    build_coast_index,
    project_to_coast,
)

_COAST_CUM, _COAST_PTS = build_coast_index()
_SITE_TO_INDEX = {site: i for i, (site, _, _) in enumerate(BARNACLE_COAST)}
```

- [ ] **Step 2: Add along-coast computation inside `_run_one_fold`**

Find the block in `_run_one_fold` that computes `held_pred["error_km"]` (haversine). It looks like:

```python
held_pred["error_km"] = common.haversine(
    held_pred["y"].values, held_pred["x"].values, site_lat, site_lon
)
fold_result["status"] = "OK"
fold_result["n_held_out"] = int(len(held_pred))
fold_result["mean_pred_lat"] = float(held_pred["y"].mean())
fold_result["mean_pred_lon"] = float(held_pred["x"].mean())
fold_result["mean_error_km"] = float(held_pred["error_km"].mean())
fold_result["median_error_km"] = float(held_pred["error_km"].median())
```

After the haversine block, before the existing `fold_result["per_sample"]` assignment, add:

```python
true_idx = _SITE_TO_INDEX.get(site_code)
if true_idx is not None:
    coast_results = [
        project_to_coast(lat, lon, _COAST_CUM, _COAST_PTS)
        for lat, lon in zip(held_pred["y"].values, held_pred["x"].values)
    ]
    coast_pos = np.array([cp for cp, _ in coast_results], dtype=np.float64)
    offshore = np.array([off for _, off in coast_results], dtype=np.float64)
    along_coast_err = np.abs(coast_pos - _COAST_CUM[true_idx])
    held_pred["coast_pos_km"] = coast_pos
    held_pred["offshore_km"] = offshore
    held_pred["along_coast_err_km"] = along_coast_err
    fold_result["mean_along_coast_err_km"] = float(along_coast_err.mean())
    fold_result["median_along_coast_err_km"] = float(along_coast_err.median())
    fold_result["mean_offshore_km"] = float(offshore.mean())
```

`numpy` is already imported (`import numpy as np`); if not, add it.

- [ ] **Step 3: Smoke-import**

```bash
cd <repo-root>
pixi run python -c "import validation.run_loso; print('ok')"
```

Expected: `ok`. If `numpy` import is missing, add `import numpy as np` at the top.

- [ ] **Step 4: Re-run Path B (dosage LOSO) on real data**

```bash
cd <repo-root>
rm -rf out/path_d_validation/loso_dosage
mkdir -p out/path_d_validation/loso_dosage
pixi run python -m validation.run_loso \
    --inputs-dir out/path_d_validation/inputs \
    --samples-locations <test-data>/samples_locations.tsv \
    --sample-table <test-data>/sample_table.tsv \
    --beagle out/path_d_validation/inputs/combined.thin_byMiss100k.beagle.gz \
    --gl-mode dosage \
    --out-dir out/path_d_validation/loso_dosage \
    --max-epochs 500 \
    --max-missing-frac 0.50 2>&1 | tee out/path_d_validation/loso_dosage/sweep.log | grep -E "^run_loso|^\[gl_|gl_to_locator] END"
```

Expected: 12 fold lines printed, all OK; total wall-clock ~3 min with 3-way GPU parallelism.

- [ ] **Step 5: Verify the new fields are present in fold_result.json**

```bash
pixi run python -c "
import json, glob
for p in sorted(glob.glob('out/path_d_validation/loso_dosage/*/fold_result.json')):
    j = json.loads(open(p).read())
    assert 'mean_along_coast_err_km' in j, f'missing in {p}'
    print(f\"{j['site']}: along_coast_err={j['mean_along_coast_err_km']:.1f} km, offshore={j['mean_offshore_km']:.1f} km\")
"
```

Expected: 12 lines printed, each with along-coast and offshore values in km.

- [ ] **Step 6: Commit**

```bash
git add validation/run_loso.py
git commit -m "$(cat <<'EOF'
run_loso: emit along-coast error metric per fold

Adds three new fields to fold_result.json:
  - mean_along_coast_err_km
  - median_along_coast_err_km
  - mean_offshore_km

Computed by projecting each held-out sample's prediction onto the 12-site
coast polyline and taking the along-polyline distance to the true site.
Existing haversine fields (mean_error_km, median_error_km) retained for
backward compatibility.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update `validation/summarize.py` for along-coast metric in Path B

**Files:**
- Modify: `validation/summarize.py`

- [ ] **Step 1: Add coastline imports and centroid helper**

Open `validation/summarize.py`. Near the top, after `from validation import common`, add:

```python
from validation.coastline import build_coast_index
```

In `main()`, near where `centroid_err_km` is computed, add a parallel along-coast baseline:

```python
# Along-coast baseline: mean |cum_km[i] - midpoint| across all 12 sites.
_coast_cum, _coast_pts = build_coast_index()
along_coast_baseline_km = float(
    np.abs(_coast_cum - _coast_cum[-1] / 2.0).mean()
)
```

`np` may not be imported yet in `summarize.py`; if not, add `import numpy as np` near the top.

- [ ] **Step 2: Update `_format_loso` to display along-coast columns when present**

Open `_format_loso` in `validation/summarize.py`. After the existing `cols = [...]` list assignment, add a feature-detection check and extend `cols` if the new columns are present in the DataFrame:

Find:
```python
cols = ["site", "n_held_out", "true_lat", "true_lon",
        "mean_pred_lat", "mean_pred_lon", "mean_error_km", "median_error_km", "status"]
```

Change to:
```python
cols = ["site", "n_held_out", "true_lat", "true_lon",
        "mean_pred_lat", "mean_pred_lon", "mean_error_km", "median_error_km", "status"]
if "mean_along_coast_err_km" in ok.columns:
    cols.insert(8, "mean_along_coast_err_km")
    cols.insert(9, "mean_offshore_km")
```

(Inserting at index 8/9 puts them right before `status` in the table.)

- [ ] **Step 3: Update `_format_loso` PASS/FAIL ratio to use along-coast when available**

In the same function, find:

```python
median_err = float(ok["median_error_km"].median())
ratio = median_err / centroid_err_km if centroid_err_km else float("nan")
```

Change to:

```python
if "median_along_coast_err_km" in ok.columns:
    median_err = float(ok["median_along_coast_err_km"].median())
    baseline = float(_format_loso._along_coast_baseline_km)
    metric_label = "along-coast"
else:
    median_err = float(ok["median_error_km"].median())
    baseline = centroid_err_km
    metric_label = "haversine"
ratio = median_err / baseline if baseline else float("nan")
```

And update the body string to use `metric_label`:

Find:
```python
body = (
    f"**{label}:** median per-fold error = {median_err:.1f} km; "
    f"centroid baseline = {centroid_err_km:.1f} km; "
    f"ratio = {ratio:.3f} ({pass_or_fail}).\n\n"
    f"{table}{failed_section}"
)
```

Change to:
```python
body = (
    f"**{label}:** median {metric_label} error = {median_err:.1f} km; "
    f"baseline = {baseline:.1f} km; "
    f"ratio = {ratio:.3f} ({pass_or_fail}).\n\n"
    f"{table}{failed_section}"
)
```

- [ ] **Step 4: Set the `_format_loso._along_coast_baseline_km` attribute from `main()`**

This is a small hack to plumb the along-coast baseline down to `_format_loso` without changing its signature. In `main()`, right after computing `along_coast_baseline_km`, add:

```python
_format_loso._along_coast_baseline_km = along_coast_baseline_km
```

(The existing `_format_loso(...)` call doesn't need to change.)

- [ ] **Step 5: Regenerate summary.md**

```bash
cd <repo-root>
pixi run python -m validation.summarize \
    --out-dir out/path_d_validation \
    --samples-locations <test-data>/samples_locations.tsv \
    --sample-table <test-data>/sample_table.tsv 2>&1 | tail -3
```

- [ ] **Step 6: Verify summary.md shows new along-coast columns**

```bash
grep -A 1 "Path B" out/path_d_validation/summary/summary.md | head -3
```

Expected: ratio reported uses "along-coast" wording, baseline is the polyline-midpoint baseline, table includes `mean_along_coast_err_km` and `mean_offshore_km`.

- [ ] **Step 7: Commit**

```bash
git add validation/summarize.py
git commit -m "$(cat <<'EOF'
summarize: Path B uses along-coast metric when available

Path B's PASS/FAIL ratio and centroid baseline now use the coast-projected
error metric (when fold_result.json has mean_along_coast_err_km), with
"along-coast" labeled in the prose. Falls back to haversine when the new
fields are absent (forward-compatible with old fold data).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add Figure 1 generator (test-data noise curve, 4 panels)

**Files:**
- Modify: `validation/summarize.py`

- [ ] **Step 1: Add the figure generator**

Append to `validation/summarize.py` (before the `main()` function):

```python
def _render_test_data_figure(out_dir: Path, summary_dir: Path) -> None:
    """Render Figure 1: test-data VCF/cont/noise-curve, 4 panels."""
    truth_path = out_dir / "example" / "inputs" / "holdout_truth.tsv"
    if not truth_path.exists():
        print(f"summarize: skipping Figure 1 — {truth_path} missing", flush=True)
        return
    truth = pd.read_csv(truth_path, sep="\t")
    truth["sampleID"] = truth["sampleID"].astype(str).str.strip('"')

    panels = [
        ("VCF (hard calls)",  out_dir / "example/baseline_vcf/run_predlocs.txt"),
        ("Cont. dosage α=0.0", out_dir / "example/smoke_a0p0/run_predlocs.txt"),
        ("Cont. dosage α=0.5", out_dir / "example/smoke_a0p5/run_predlocs.txt"),
    ]
    panel_data = []
    for label, predfile in panels:
        if not predfile.exists():
            panel_data.append((label, None))
            continue
        pred = pd.read_csv(predfile)
        pred["sampleID"] = pred["sampleID"].astype(str).str.strip('"')
        merged = truth.merge(pred, on="sampleID", how="inner")
        panel_data.append((label, merged))

    nc_path = out_dir / "example" / "noise_curve.tsv"
    nc_df = pd.read_csv(nc_path, sep="\t") if nc_path.exists() else None

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (label, df) in zip(axes[:3], panel_data):
        if df is None:
            ax.set_title(f"{label} (missing)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        ax.scatter(df["true_x"], df["true_y"], s=20, c="lightgrey",
                   label="truth", zorder=1)
        ax.scatter(df["x"], df["y"], s=20, c="C0", label="prediction", zorder=3)
        for _, row in df.iterrows():
            ax.plot([row["true_x"], row["x"]], [row["true_y"], row["y"]],
                    color="C0", lw=0.4, alpha=0.5, zorder=2)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-2, 52)
        ax.set_ylim(-2, 52)
        ax.set_aspect("equal")

    ax = axes[3]
    if nc_df is not None and not nc_df.empty:
        for kind, color, marker in [
            ("cont", "C0", "o"),
            ("round", "C3", "s"),
        ]:
            sub = nc_df[nc_df["encoding"] == kind].sort_values("alpha")
            if not sub.empty:
                ax.plot(sub["alpha"], sub["mean_err"], marker=marker,
                        color=color, label=kind)
        if "VCF" in nc_df["encoding"].values:
            vcf_err = float(nc_df.loc[nc_df["encoding"] == "VCF", "mean_err"].iloc[0])
            ax.axhline(vcf_err, color="black", linestyle=":",
                       label="VCF baseline")
        # Centroid baseline (18.52 for the example data — recomputed below
        # from holdout_truth + sample_data for accuracy):
        try:
            sd = pd.read_csv("data/test_sample_data.txt", sep="\t")
            sd.columns = [c.strip('"') for c in sd.columns]
            labeled = sd[sd["x"].notna()]
            cx, cy = float(labeled["x"].mean()), float(labeled["y"].mean())
            cb = float(np.sqrt(
                (truth["true_x"] - cx) ** 2 + (truth["true_y"] - cy) ** 2
            ).mean())
            ax.axhline(cb, color="grey", linestyle="--",
                       label=f"centroid ({cb:.1f})")
        except Exception:
            pass
        ax.set_xlabel("noise α")
        ax.set_ylabel("mean Euclidean error")
        ax.set_title("Noise curve")
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.set_title("Noise curve (missing)")

    fig.tight_layout()
    fig.savefig(summary_dir / "fig1_test_data.png", dpi=120)
    plt.close(fig)
    print(f"summarize: wrote {summary_dir}/fig1_test_data.png", flush=True)
```

- [ ] **Step 2: Wire it into `main()`**

In `validation/summarize.py`'s `main()`, after the existing scatter-plot block (the one that writes `dosage_vs_full_gl.png`), add a call:

```python
_render_test_data_figure(args.out_dir, summary_dir)
```

- [ ] **Step 3: Smoke-import**

```bash
cd <repo-root>
pixi run python -c "import validation.summarize; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Generate Figure 1**

```bash
cd <repo-root>
pixi run python -m validation.summarize \
    --out-dir out/path_d_validation \
    --samples-locations <test-data>/samples_locations.tsv \
    --sample-table <test-data>/sample_table.tsv 2>&1 | tail -3
```

Expected: among the output, `wrote out/path_d_validation/summary/fig1_test_data.png`.

- [ ] **Step 5: Verify the PNG exists**

```bash
ls -la out/path_d_validation/summary/fig1_test_data.png
```

Expected: file exists, size > 50 KB.

- [ ] **Step 6: Copy the PNG into the committed `validation/figures/` directory**

The figures live in the validation zone (committed). Create the dir and copy:

```bash
mkdir -p validation/figures
cp out/path_d_validation/summary/fig1_test_data.png validation/figures/fig1_test_data.png
```

- [ ] **Step 7: Commit script + figure**

```bash
git add validation/summarize.py validation/figures/fig1_test_data.png
git commit -m "$(cat <<'EOF'
summarize: add Figure 1 generator (test-data noise curve, 4 panels)

3 condition scatters (VCF / cont α=0 / cont α=0.5) with truth→prediction
arrows, plus a 4th panel showing mean error as a function of α for both
continuous and rounded encodings, with VCF baseline and centroid baseline
as horizontal reference lines. Renders to validation/figures/fig1_test_data.png.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add Figure 2 generator (balanus map with cartopy)

**Files:**
- Modify: `validation/summarize.py`

- [ ] **Step 1: Add the figure generator**

Append to `validation/summarize.py` (after `_render_test_data_figure`):

```python
def _render_balanus_map(
    out_dir: Path, summary_dir: Path, samples_locations_path: Path
) -> None:
    """Render Figure 2: cartopy map of balanus predictions colored by
    along-coast error."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:  # noqa: BLE001
        print(f"summarize: skipping Figure 2 — cartopy unavailable: {exc}",
              flush=True)
        return

    loso_dir = out_dir / "loso_dosage"
    fold_jsons = sorted(loso_dir.glob("*/fold_result.json"))
    if not fold_jsons:
        print(f"summarize: skipping Figure 2 — no fold results in {loso_dir}",
              flush=True)
        return

    sites = pd.read_csv(samples_locations_path, sep="\t")
    rows = []
    for fj in fold_jsons:
        r = json.loads(fj.read_text())
        if r.get("status") != "OK":
            continue
        for s in r.get("per_sample", []):
            rows.append({
                "site": r["site"],
                "true_lat": r["true_lat"],
                "true_lon": r["true_lon"],
                "pred_lat": s.get("y"),
                "pred_lon": s.get("x"),
                "along_coast_err_km": s.get("along_coast_err_km"),
            })
    if not rows:
        print("summarize: skipping Figure 2 — no per-sample predictions", flush=True)
        return
    df = pd.DataFrame(rows)
    if df["along_coast_err_km"].isna().all():
        # Fall back to haversine error for color if the new field isn't there.
        df["color_metric_km"] = (
            ((df["pred_lat"] - df["true_lat"]) ** 2 +
             (df["pred_lon"] - df["true_lon"]) ** 2) ** 0.5 * 111.0
        )
        cbar_label = "fallback degrees·111 km"
    else:
        df["color_metric_km"] = df["along_coast_err_km"]
        cbar_label = "along-coast error (km)"

    fig = plt.figure(figsize=(8, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-152, -115, 30, 62], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.95")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.COASTLINE, lw=0.6)
    ax.add_feature(cfeature.STATES, lw=0.3)
    ax.gridlines(draw_labels=True, lw=0.2, color="grey", alpha=0.5)

    sc = ax.scatter(
        df["pred_lon"], df["pred_lat"],
        c=df["color_metric_km"], cmap="viridis_r",
        s=25, alpha=0.85, transform=ccrs.PlateCarree(),
        zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.04)
    cbar.set_label(cbar_label)

    site_sizes = {row["SiteCode"]: 30 + 5 * row["Samples"]
                  for _, row in sites.iterrows()}
    ax.scatter(
        sites["Lon"], sites["Lat"],
        s=[site_sizes[c] for c in sites["SiteCode"]],
        marker="*", c="red", edgecolors="black", lw=0.8,
        transform=ccrs.PlateCarree(), zorder=4, label="true sites",
    )
    for _, row in sites.iterrows():
        ax.text(
            row["Lon"] + 0.4, row["Lat"], row["SiteCode"],
            fontsize=7, transform=ccrs.PlateCarree(), zorder=5,
        )

    ax.set_title("Balanus predictions (LOSO, continuous dosage)")
    fig.savefig(summary_dir / "fig2_balanus_map.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    print(f"summarize: wrote {summary_dir}/fig2_balanus_map.png", flush=True)
```

- [ ] **Step 2: Wire it into `main()`**

In `validation/summarize.py`'s `main()`, after the call to `_render_test_data_figure`, add:

```python
_render_balanus_map(args.out_dir, summary_dir, args.samples_locations)
```

- [ ] **Step 3: Smoke-import**

```bash
cd <repo-root>
pixi run python -c "import validation.summarize; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Generate Figure 2**

```bash
cd <repo-root>
pixi run python -m validation.summarize \
    --out-dir out/path_d_validation \
    --samples-locations <test-data>/samples_locations.tsv \
    --sample-table <test-data>/sample_table.tsv 2>&1 | tail -3
```

Expected: among the output, `wrote out/path_d_validation/summary/fig2_balanus_map.png`.

- [ ] **Step 5: Verify the PNG exists**

```bash
ls -la out/path_d_validation/summary/fig2_balanus_map.png
```

Expected: file exists, size > 100 KB.

- [ ] **Step 6: Copy the PNG into the committed `validation/figures/` directory**

```bash
cp out/path_d_validation/summary/fig2_balanus_map.png validation/figures/fig2_balanus_map.png
```

- [ ] **Step 7: Commit script + figure**

```bash
git add validation/summarize.py validation/figures/fig2_balanus_map.png
git commit -m "$(cat <<'EOF'
summarize: add Figure 2 generator (balanus cartopy map)

cartopy.PlateCarree projection over Pacific coast bbox (lat 30-62, lon
-152 to -115); LAND/OCEAN/COASTLINE/STATES features; 12 true sites as
red stars sized by sample count and labeled; 54 per-sample predictions
as dots colored by along-coast error (viridis_r colormap with colorbar).
Falls back to a degree-based proxy if along-coast metric isn't present
in fold_result.json. Skips with a warning if cartopy is unavailable.

Renders to validation/figures/fig2_balanus_map.png.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Drop microsats from the merge zone

**Files:**
- Delete: `scripts/microsat_to_locator.py`
- Delete: `scripts/microsat_features.py`
- Delete: `scripts/microsat_poly.py`
- Modify: `tests/test_input_extensions.py` (remove 3 microsat tests + fixture)

- [ ] **Step 1: Delete the 3 microsat scripts**

```bash
cd <repo-root>
git rm scripts/microsat_to_locator.py
git rm scripts/microsat_features.py
git rm scripts/microsat_poly.py
```

- [ ] **Step 2: Remove microsat tests from `tests/test_input_extensions.py`**

Open `tests/test_input_extensions.py` and delete:

1. The `MICROSAT_SAMPLE = ...` constant (the multi-line string fixture).
2. The function `test_microsat_to_locator_dosage`.
3. The function `test_microsat_features_geometry`.
4. The function `test_microsat_poly_expansion_and_pca`.
5. The section header comment `# Microsat converters` if present.

Keep:
- All imports.
- `SCRIPTS_DIR`, `run_script`, `write_synthetic_beagle`, `write_bam_list`.
- The 3 GL tests: `test_gl_to_locator_dosage`, `test_gl_to_locator_full_gl`, `test_gl_to_locator_dimension_mismatch_errors`.

After the edit, the file should have ~140 lines (down from ~250).

- [ ] **Step 3: Run the remaining tests in `tests/test_input_extensions.py`**

```bash
cd <repo-root>
pixi run pytest tests/test_input_extensions.py -n 0 -v
```

Expected: 3 passed (the 3 GL tests).

- [ ] **Step 4: Verify nothing else breaks**

```bash
pixi run pytest tests/test_data_loading.py tests/test_input_extensions.py validation/tests/test_validation.py -n 0 -q
```

Expected: all green; no microsat references in collected tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/ tests/test_input_extensions.py
git commit -m "$(cat <<'EOF'
Drop microsat scripts and tests from the merge zone

The 4 microsat scripts (microsat_to_locator.py, microsat_features.py,
microsat_poly.py) had synthetic-fixture unit tests but zero real-data
validation. Shipping them in the merge zone of this branch dilutes the
GL-pipeline story and asks the maintainer to trust unvalidated code.
Microsats are deferred to a separate spec/branch.

Removed: 3 production scripts and 3 microsat tests + the MICROSAT_SAMPLE
fixture from tests/test_input_extensions.py. Kept: the 3 GL tests
(test_gl_to_locator_dosage, test_gl_to_locator_full_gl,
test_gl_to_locator_dimension_mismatch_errors).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Add `docs/genotype_likelihoods.md` (small user doc)

**Files:**
- Create: `docs/genotype_likelihoods.md`

- [ ] **Step 1: Write the doc**

Create `docs/genotype_likelihoods.md`:

```markdown
# Genotype likelihood input

ReLocator's `--matrix` flag accepts continuous expected dosage (floats in
[0, 2]), not just hard-call dosage (integers 0/1/2). This lets you feed
ANGSD-derived genotype likelihoods directly, without information-losing
hard-call rounding.

## Why this matters

At low coverage, hard-calling collapses (P_AA, P_AB, P_BB) probabilities
into a single integer in {0, 1, 2}, throwing away uncertainty information
that the genotype-likelihood (GL) representation preserves. The patched
loader (`locator/loaders.py:_load_from_matrix`) detects float dtype on the
input matrix and feeds expected dosage straight to the network without an
integer round trip. The `full_gl` mode (3 columns per site) loads through
the same path.

## Workflow

```bash
# 1. Run ANGSD to produce a beagle.gz file:
angsd \
    -bam bam.filelist -ref ref.fa \
    -GL 2 -doGlf 2 \
    -doMajorMinor 1 -doMaf 1 \
    -SNP_pval 1e-6 -minMapQ 20 -minQ 20 \
    -out output

# 2. Convert beagle GLs to a continuous-dosage matrix:
python scripts/gl_to_locator.py \
    --beagle output.beagle.gz \
    --bam_list bam.filelist \
    --out gl_dosage.txt

# 3. Train ReLocator:
locator \
    --matrix gl_dosage.txt \
    --sample_data sample_data.txt \
    --out run/
```

## Notes

- **Sample order constraint.** `gl_to_locator.py` derives sample IDs from
  `Path(bam).stem` for each line in `--bam_list`. The order in `bam.filelist`
  must match the order ANGSD was invoked with. The IDs must also appear in
  `sample_data.txt`.
- **`--gl_mode {dosage,full_gl}`** chooses between expected dosage (1 column
  per site) and the full GL probability triplet (3 columns per site). Both
  flow through the same patched loader. `full_gl` preserves more uncertainty
  but produces a 3× wider input matrix.
- **Filtering.** `gl_to_locator.py` applies a missingness filter
  (`--max_missing_frac`, default 0.10) and a minor-allele-frequency filter
  (`--min_maf`, default 0.01). The locator binary applies its own
  minor-allele-count filter (`--min_mac`, default 2) on top.

## Validation evidence

The `validation/` directory at the repo root contains the validation tooling,
test outputs, figures, and design notes that produced this branch's evidence
that the continuous-dosage path matches VCF performance. See
`validation/README.md` and `validation/summary/summary.md` for the headline
results.
```

- [ ] **Step 2: Verify the file exists and renders cleanly**

```bash
cat docs/genotype_likelihoods.md | wc -l
head -5 docs/genotype_likelihoods.md
```

Expected: ~55 lines; first heading is `# Genotype likelihood input`.

- [ ] **Step 3: Commit**

```bash
git add docs/genotype_likelihoods.md
git commit -m "$(cat <<'EOF'
Add docs/genotype_likelihoods.md user doc

Small user-facing doc explaining the GL → matrix → locator workflow.
Lives in the merge zone (not validation/). ~55 lines: why the continuous
dosage path matters, three-step bash example (ANGSD → gl_to_locator →
locator), notes on sample-order constraint, --gl_mode options, and
filtering layers. Points to validation/ for evidence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Add `validation/README.md`

**Files:**
- Create: `validation/README.md`

- [ ] **Step 1: Write the README**

Create `validation/README.md`:

```markdown
# Validation evidence for the `genotype_likelihoods` branch

This directory is the validation zone of the feature-branch handoff. It
contains all the validation tooling, test outputs, figures, and design
notes that produced the branch's empirical evidence. None of these files
need to merge to `main`; they live alongside the code change so reviewers
can browse the evidence without running anything.

## What's here

- `coastline.py`, `common.py`, `build_inputs.py`, `run_smoke.py`,
  `run_loso.py`, `summarize.py`, `thin_beagle.py`, `synth_gls.py` —
  validation tooling.
- `tests/test_validation.py` — unit tests for the tooling.
- `figures/`
  - `fig1_test_data.png` — 4-panel: VCF / continuous-dosage α=0 /
    continuous-dosage α=0.5 truth→prediction scatter, plus a noise-curve
    panel showing how mean error scales with GL uncertainty for continuous
    vs rounded encodings.
  - `fig2_balanus_map.png` — cartopy map of the Pacific coast with the 12
    true site centroids and the 54 per-sample predictions (LOSO,
    continuous dosage), colored by along-coast error.
- `summary/`
  - `summary.md` — aggregated path D + stress-test report.
  - `summary.tsv` — flat per-fold table.
  - `noise_curve.tsv` — per-α prediction error for both encodings.
- `docs/specs/` — the design spec docs that drove this branch.
- `docs/plans/` — the implementation plans.

## Headline result

The continuous-dosage path through the patched `--matrix` loader is
within ~1% of VCF performance on the example test data (500 samples ×
~11.5k biallelic sites, 90-sample holdout). Under increasing GL
uncertainty (α ∈ {0, 0.3, 0.5, 0.7, 0.9}), prediction error stays flat
at ~4.2–4.6 mean Euclidean error, while the rounded path collapses to
the centroid baseline (~18.5) at α ≥ 0.7. **The loader patch is
load-bearing**: without it, GL uncertainty above ~50% destroys the
pipeline.

See `summary/summary.md` for the full report.

## Re-running anything

```bash
# Path B (LOSO over the balanus 54-sample test set, dosage mode):
pixi run python -m validation.run_loso \
    --inputs-dir out/path_d_validation/inputs \
    --samples-locations ../test_data/samples_locations.tsv \
    --sample-table ../test_data/sample_table.tsv \
    --beagle out/path_d_validation/inputs/combined.thin_byMiss100k.beagle.gz \
    --gl-mode dosage \
    --out-dir out/path_d_validation/loso_dosage \
    --max-epochs 500

# Regenerate the summary + figures:
pixi run python -m validation.summarize \
    --out-dir out/path_d_validation \
    --samples-locations ../test_data/samples_locations.tsv \
    --sample-table ../test_data/sample_table.tsv

# Run the validation unit tests:
pixi run pytest validation/tests/ -v
```

## Branch zones

The branch has two zones:

- **Merge zone** — `locator/`, `scripts/gl_to_locator.py`, `tests/`,
  `docs/genotype_likelihoods.md`. Small, focused, what would actually get
  merged to `main`.
- **Validation zone** — this directory. Branch-only; supports the merge
  but doesn't ship.
```

- [ ] **Step 2: Verify**

```bash
cat validation/README.md | wc -l
```

Expected: ~70 lines.

- [ ] **Step 3: Commit**

```bash
git add validation/README.md
git commit -m "$(cat <<'EOF'
Add validation/README.md intro for the kr-colab handoff

Brief intro for maintainers landing in the validation/ directory: what
files are here, the headline empirical result (continuous dosage matches
VCF; rounded collapses at α ≥ 0.7), how to re-run the pipeline, and the
two-zone branch structure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Rewrite the path D design spec at its new location

**Files:**
- Modify: `validation/docs/specs/2026-05-03-path-d-validation-design.md`

- [ ] **Step 1: Read the current spec**

```bash
cd <repo-root>
wc -l validation/docs/specs/2026-05-03-path-d-validation-design.md
head -30 validation/docs/specs/2026-05-03-path-d-validation-design.md
```

The current spec was written before we knew about loaders.py:141 and the
loader patch. It treats Path D as the validation arc for un-patched
ReLocator. We rewrite it to reflect the actual outcome: loader patch
landed, full_gl works for free, microsats dropped.

- [ ] **Step 2: Replace the file with the rewritten spec**

Open `validation/docs/specs/2026-05-03-path-d-validation-design.md` and
overwrite the entire content with:

```markdown
# Path D Validation: Genotype-Likelihood Pipeline End-to-End

Design and outcome doc for the validation arc on the
`genotype_likelihoods` branch. The original draft (committed pre-patch)
focused on whether the unpatched `--matrix` flag could consume GL-derived
dosage; that question was answered negatively (it could not: integer
check at `loaders.py:141`), and the arc evolved into a loader patch plus
empirical evidence that the patched path is load-bearing.

This rewrite documents the actual outcome.

## Goals (final)

1. Validate that the GL → continuous-dosage → ReLocator pipeline works
   end-to-end on a non-underdetermined dataset.
2. Demonstrate that the loader patch (commit `5418bf3`) is non-destructive
   for the existing integer hard-call path.
3. Demonstrate that the patched continuous-dosage path is *load-bearing*:
   without it, GL uncertainty above ~50% destroys downstream prediction.
4. Produce supporting figures and a summary report that the kr-colab
   maintainers can review without running anything.

## Non-goals

- Microsat real-data validation (deferred to a separate branch).
- Native `--beagle` CLI flag (the existing patched `--matrix` covers
  continuous dosage and full_gl).
- Bootstrap / jackknife confidence intervals on per-fold error.

## Test sets

- **Balanus 54-sample beagle** (private). 12 sites along the Pacific
  coast, lat 34.4° → 60.1°. Used for Path A (smoke), Path B (dosage
  LOSO), and Path C (full_gl LOSO).
- **ReLocator's bundled msprime VCF** (`data/test_genotypes.vcf.gz`),
  500 samples × 11,497 biallelic sites in a simulated 50×50 unit square.
  Used for the example-VCF stress test and noise curve. Ground truth is
  exact (msprime simulation), so we can compute prediction error directly.

## Findings

### Path A — pipeline plumbing
PASS. End-to-end run on the balanus thinned beagle (100k sites × 54
samples) completes in ~1 min wall-clock; predictions written, sample IDs
align, dimensions correct.

### Path B — LOSO sanity
12-fold LOSO on the balanus dosage matrix. Median per-fold error
549.3 km via haversine; 0.804 ratio against the centroid baseline. By
the strict spec threshold (ratio < 0.5), this is a "FAIL," but the
underlying problem is sample-size / feature-width: 54 samples × 100k
features is severely underdetermined. The model picks up signal at
mid-latitude sites (Oregon: 100–400 km error) and collapses toward the
centroid at the extremes (AFB Alaska: ~1900 km; GOL S. California:
~1100 km).

### Path C — full_gl LOSO
Originally blocked by `loaders.py:141` integer rejection. After the
loader patch (`5418bf3`), full_gl matrices (3 cols/site of probabilities
in [0, 1]) load through the same float-detection branch as 1-col-per-site
dosage. A 50-epoch smoke run on the balanus 100k×3-col matrix completes
in 2.6 min. The originally-deferred "Native `--beagle` loader" TODO is
no longer a correctness blocker; it remains a UX improvement.

### Example-VCF stress test (the actual evidence)
On a non-underdetermined dataset (500 samples × 11.5k sites,
90-sample holdout):

| condition | mean Euclidean err | max | ratio vs centroid (18.5) |
|---|---|---|---|
| VCF baseline (hard calls) | 4.46 | 30.3 | 0.24 |
| Continuous dosage, α=0.0 | 4.52 | 23.2 | 0.24 |
| Continuous dosage, α=0.5 | 4.62 | 18.6 | 0.25 |

The continuous-dosage path matches VCF performance to within ~1.4% on
mean error and is actually better in worst-case (lower max error at
both noise levels).

### Noise curve (the load-bearing finding)
Sweeping α over the example data:

| α | continuous | rounded |
|---|---|---|
| 0.0 | 4.52 | 4.67 |
| 0.3 | 4.51 | 4.83 |
| 0.5 | 4.62 | 5.44 |
| 0.7 | 4.22 | **18.54** ← centroid |
| 0.9 | 4.58 | **18.51** ← centroid |

Continuous dosage stays flat across the entire α range. Rounded matches
at low noise but **collapses at α ≥ 0.7**: every dosage rounds to "1"
once max(GL) drops near 0.4 (E[dosage] ≈ 0.5, 1.0, 1.5 for the three
genotypes; banker's rounding sends them all to 1). The matrix becomes
nearly constant and the model predicts the geographic mean.

The loader patch is therefore *load-bearing*: without it, GL uncertainty
above ~50% destroys the pipeline.

## Loader patch safety case

The patch (`locator/loaders.py` + `locator/training.py`) is structurally
additive: a `if np.issubdtype(values.dtype, np.floating): return early`
branch is added before the existing integer check, and the `_filter_dosage_matrix`
helper is only called from the new float branch. The integer code path is
byte-identical to the original.

Empirical verification:
- `np.array_equal(filtered_genotypes_patched, filtered_genotypes_original)
  == True` on the example VCF (uint8, shape (5830, 500)).
- All 232 ReLocator core tests pass post-patch.
- Same-seed VCF runs are GPU-nondeterministic (two consecutive same-seed
  runs differ in 1000/1000 lines), so prediction-level diff comparisons
  cannot show patch impact directly. The bit-identical filtered_genotypes
  is the rigorous test.

See `validation/summary/summary.md` for the full report and the
generated figures (`validation/figures/fig{1,2}_*.png`).

## Open assumptions / risks

- **12-site polyline approximates the real coast.** A prediction in a
  bay or strait projects diagonally across, not following the coast
  perfectly. Documented in `validation/coastline.py`.
- **`offshore_km` is informational, not corrective.** The map plots
  predictions at their raw lat/lon; no inland-snap is applied.
```

(Save the file.)

- [ ] **Step 3: Commit**

```bash
git add validation/docs/specs/2026-05-03-path-d-validation-design.md
git commit -m "$(cat <<'EOF'
Rewrite path D spec to reflect actual outcome

The original spec was written before we knew about loaders.py:141. This
rewrite documents the evolved arc: loader patch landed (5418bf3) and was
verified non-destructive on the integer path (filtered_genotypes
bit-identical); full_gl works for free through the same patched loader;
the load-bearing empirical evidence is the noise curve on the example
VCF, where continuous-dosage stays flat (mean ~4.5) across α ∈
{0, 0.3, 0.5, 0.7, 0.9} while rounded collapses to the centroid baseline
at α ≥ 0.7. Microsats dropped from this arc.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Update parent `CLAUDE.md` TODOs

**Files:**
- Modify: `../CLAUDE.md` (parent dir, NOT in any git repo)

- [ ] **Step 1: Locate the relevant TODO sections**

```bash
grep -n "TODO (Correctness)\|TODO (Feature)\|microsat" ../CLAUDE.md | head -20
```

- [ ] **Step 2: Mark the loader-rejects-floats TODO as resolved**

In `../CLAUDE.md`, find the section:

```
- [ ] **TODO (Correctness): ReLocator `--matrix` loader rejects continuous dosage**
  `locator/loaders.py:_load_from_matrix` requires `np.isin(values, [0, 1, 2])` and
  feeds the matrix through `_counts_to_genotype_array` which uses int8 arithmetic
  on synthetic phased haplotypes. As a stopgap, `gl_to_locator.py` (dosage mode)
  rounds expected dosage to `{0, 1, 2}` before writing — this collapses the GL
  uncertainty signal into hard calls, making the dosage path equivalent to ANGSD's
  `-doGeno 2` output for downstream training. The `full_gl` mode emits floats in
  [0, 1] and CANNOT be consumed by `locator --matrix` at all; it is currently a
  dead code path. Both are fixed by the next item.
```

Replace with:

```
- [x] **RESOLVED (Correctness, commit `5418bf3` on `genotype_likelihoods`):
       ReLocator `--matrix` loader accepts continuous dosage**
  `locator/loaders.py:_load_from_matrix` now detects float dtype and returns a
  2D `(n_sites, n_samples)` ndarray directly, bypassing `_counts_to_genotype_array`
  for the continuous case. `locator/training.py:_filter_genotypes` routes float
  inputs to a new `_filter_dosage_matrix` helper that applies MAC and max_snps
  filters on the dosage matrix without going through `allel.GenotypeArray`.
  Both dosage and `full_gl` matrices flow through the same patched path.
  The patch is structurally additive — the integer hard-call path is byte-
  identical to the original; verified by `np.array_equal(filtered_patched,
  filtered_original) == True` on the example VCF.
```

- [ ] **Step 3: Mark the Native `--beagle` loader TODO as partially resolved**

Find the section:

```
- [ ] **TODO (Feature): Native `--beagle` loader in ReLocator `DataLoader`**
  Add a `--beagle` / `--bam_list` argument pair directly to the ReLocator CLI and a
  corresponding loader class, so the preprocessing script is not required. The loader
  should produce the same `(n_samples, n_sites)` float32 dosage matrix currently
  produced by the script. Integration point: wherever `--vcf`, `--zarr`, and `--geno`
  are dispatched in the ReLocator source. **This is the canonical fix for the
  rounding stopgap above** — a continuous-dosage path that bypasses
  `_counts_to_genotype_array` entirely.
```

Replace with:

```
- [~] **PARTIALLY RESOLVED (Feature): Native `--beagle` loader**
  The continuous-dosage path through `--matrix` (resolved above) covers the
  correctness gap. A dedicated `--beagle` / `--bam_list` CLI flag is still a
  user-experience improvement (skips the explicit `gl_to_locator.py` step) but
  is no longer a correctness blocker. The patched `--matrix` accepts the float
  dosage matrix `gl_to_locator.py` produces, end-to-end.
```

- [ ] **Step 4: Trim the microsat sections**

Find the long microsat section (probably under "## Scripts" → "### `microsat_to_locator.py`" and similar). Replace the entire microsat chunk (production-script docs + microsat-feature engineering + microsat-specific TODOs) with a one-paragraph deferral note:

```
## Microsats (deferred)

The microsat pipeline scripts (`microsat_to_locator.py`, `microsat_features.py`,
`microsat_poly.py`) shipped earlier on the `genotype_likelihoods` branch but
were dropped from the merge zone before handoff: they had synthetic-fixture
unit tests but zero real-data validation. Microsats are deferred to a
separate branch + spec; that work needs its own validation arc analogous to
path D.
```

(Find the right insertion point — probably right after the GL section ends and
before the "Microsat Feature Engineering" heading. Delete from "Microsat Feature
Engineering" through the end of all microsat-related TODOs.)

- [ ] **Step 5: Verify**

```bash
grep -c "microsat\|MICROSAT\|microsat_to_locator\|microsat_features\|microsat_poly" ../CLAUDE.md
```

Expected: a small number (a few references in the deferred note + maybe in
the project description). Should NOT be the original ~20+ count.

- [ ] **Step 6: No commit needed** — `CLAUDE.md` is at the parent directory and not in any git repo. Just having the updated content on disk is sufficient; future Claude sessions will read the updated state.

---

## Task 12: Final test verification

**Files:**
- No code changes.

- [ ] **Step 1: Run the full ReLocator core test suite**

```bash
cd <repo-root>
pixi run pytest tests/ --ignore=tests/test_bandwidth_optimization_integration.py -n 4 2>&1 | tail -5
```

Expected: 232 passed (or one xdist-flaky failure that passes in isolation).

- [ ] **Step 2: Run the validation test suite**

```bash
pixi run pytest validation/tests/ -n 0 -v 2>&1 | tail -20
```

Expected: 16 passed (12 prior + 4 new coastline).

- [ ] **Step 3: Sanity-check the figure files exist and are committed**

```bash
ls -la validation/figures/
git log --oneline --diff-filter=A -- validation/figures/ | head
```

Expected: `fig1_test_data.png` and `fig2_balanus_map.png` exist; `git log`
shows their addition commits from Tasks 5 and 6.

- [ ] **Step 4: Sanity-check summary outputs are committable / committed**

```bash
ls -la validation/summary/ 2>&1 || echo "validation/summary missing"
```

If the summary outputs (`summary.md`, `summary.tsv`, `noise_curve.tsv`) aren't
yet under `validation/summary/` (still only in `out/path_d_validation/summary/`),
copy them:

```bash
mkdir -p validation/summary
cp out/path_d_validation/summary/summary.md      validation/summary/
cp out/path_d_validation/summary/summary.tsv     validation/summary/
cp out/path_d_validation/example/noise_curve.tsv validation/summary/
git add validation/summary/
git commit -m "Add summary outputs to validation/summary/

Committed copies of the path D summary artifacts: summary.md (final
report), summary.tsv (per-fold table), noise_curve.tsv (per-α error
table). Live versions stay in out/ for re-runs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Final state check**

```bash
git log --oneline | head -20
git status
```

Expected: log shows the cleanup arc commits (Tasks 1–11 + summary copy);
`git status` shows clean working tree (or just `pixi.lock` untracked, which
is the project default).

- [ ] **Step 6: No commit** — this is a verification gate, not a change.

---

## Self-review notes

- **Spec coverage:** Each goal in the spec maps to at least one task —
  restructure (Task 1), microsats (Task 7), coast metric (Task 2 + integration
  in 3, 4), Figure 1 (Task 5), Figure 2 (Task 6), small user doc (Task 8),
  validation README (Task 9), spec rewrite (Task 10), CLAUDE.md (Task 11),
  final verification (Task 12). ✓
- **Placeholder scan:** Every step has executable commands or full code
  blocks. No "TBD" or "fill in details." The CLAUDE.md edits in Task 11
  describe the exact text to find-and-replace. ✓
- **Type consistency:** `cum_km` and `points` are consistently the return
  values of `build_coast_index()`. `coast_pos_km` and `offshore_km` are
  consistent across `coastline.py` and the `run_loso.py` integration. The
  fold_result.json field names (`mean_along_coast_err_km`,
  `median_along_coast_err_km`, `mean_offshore_km`) match between Task 3
  (writer) and Task 4 (reader). ✓
- **Path consistency:** After Task 1, all imports use `from validation
  import …`. Subsequent tasks consistently use that path. ✓
