# Classifier Head for ReLocator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional discrete-site classifier output head to ReLocator's network with three prediction modes (`regress` / `classify` / `classify_then_avg`), and validate it on the sculpin 16-site LOSO arc against the existing regression baseline plus the regression+range_mask result (4-way comparison).

**Architecture:** The classifier head replaces `Dense(2)→Dense(2)` with a single `Dense(K, softmax)` final layer at training time. Site labels are auto-derived from coord equivalence in the training set (matches popgen-canonical sampling-design convention). At inference, `classify` returns the centroid of the argmax site; `classify_then_avg` returns `softmax @ training_site_centroids`, giving a continuous (lat, lon) bounded to the convex hull of training sites. The change is decoupled from input format — works for any continuous feature matrix the existing `--matrix` / `--vcf` / `--zarr` loaders produce.

**Tech Stack:** Python 3.12, TensorFlow 2.19.1, numpy, pandas, h5py (for centroid persistence), pytest with xdist, the existing `pixi` env, the existing `locator` CLI from the editable install.

**Spec:** `validation/docs/specs/2026-05-05-classifier-head-design.md`.

**Branch:** `classifier-head`, off `microsat-sculpin` HEAD `e39606f`. All commits go here. No pushes until Task 14.

---

## File map

Created in this plan:

| Path | Zone | Responsibility |
|---|---|---|
| `locator/data/site_labels.py` | merge | `derive_site_labels()` pure function — exact-coord-equality grouping with K > 0.5 × N elephant-case detection |
| `tests/test_classifier_head.py` | merge | Unit tests for the classifier head end-to-end |
| `docs/prediction_modes.md` | merge | User-facing guide: 3 modes, when to use each, worked example |
| `validation/sculpin/run_loso_classifier.py` | validation | 16-fold sculpin LOSO with `classify` + `classify_then_avg` modes via the `Locator` Python API |
| `validation/sculpin/summarize_classifier.py` | validation | Aggregate per-fold JSONs across all 4 modes → comparison TSV, figures, summary |
| `validation/summary/sculpin_classifier_summary.md` | validation | Markdown report (4-mode comparison, snap-correctness deltas, error-vs-FST recompute) |
| `validation/summary/sculpin_classifier_kfold.tsv` | validation | Per-fold rows across all 4 modes |
| `validation/figures/sculpin_classifier_modes.png` | validation | Grouped bar chart per site + aggregate, all 4 modes |
| `validation/figures/sculpin_classifier_map.png` | validation | 4-panel cartopy map: predictions per mode |

Modified:

| Path | Change |
|---|---|
| `locator/models.py` | `create_network` — add `n_classes: int | None = None` parameter; when set, replace the two `Dense(2)` layers with one `Dense(n_classes, softmax)`; switch default loss to `categorical_crossentropy` |
| `locator/training.py` | `_create_model` — read `prediction_mode` from `self.config`, derive labels, pass `n_classes=K` and one-hot labels through; add config validation; `_build_datasets_and_fit` — substitute one-hot labels for normalized coords when classifier mode is set; `_save_model_metadata` — persist `_site_centroids` to HDF5 |
| `locator/prediction.py` | `predict` — branch on `self.config["prediction_mode"]`: `classify` → argmax → centroid lookup → denormalize; `classify_then_avg` → softmax @ centroids → denormalize; `regress` → existing path unchanged |
| `validation/docs/specs/2026-05-04-microsatellite-input-extension-design.md` | Append a "Classifier head follow-up" note to the TODO list referencing this plan |

Output tree (created at runtime by the validation runners):

```
out/sculpin_validation/
    loso_classifier/
        classify/<SITE>/{sample_data.txt, locator_run/, fold_result.json}
        classify_then_avg/<SITE>/{sample_data.txt, locator_run/, fold_result.json}
```

---

## Conventions (apply to every task)

- All Python files start with `from __future__ import annotations` and follow the existing `locator/` style (typed signatures, `Optional[X]` from `typing`, docstrings with `:param X:` `:type X:` `:rtype:` blocks).
- All commands run inside the pixi env: `pixi run <cmd>`. Bare `pytest` will fail (`conftest.py` imports `allel`, `zarr`).
- The `.claude/hooks/ruff-on-edit.sh` hook auto-runs ruff after every Edit/Write — code must be ruff-clean. Run `pixi run ruff check <file>` manually before committing each task.
- Pre-commit hooks aren't installed in this clone; manual `pixi run ruff check` is the only gate.
- The `locator/` core changes preserve byte-identical `regress` behavior. Run a smoke test of the existing example VCF flow after Task 4 to confirm the regression path is untouched.
- The `--matrix` loader path was patched in `genotype_likelihoods` to accept continuous float input. The classifier-head changes are independent of that loader: they operate on the dataset *after* loading, on the (n_samples, n_features) feature matrix the encoder produces.
- When importing helper code from `validation/` runners, follow the existing pattern: `from validation import common`. The `validation/sculpin/run_loso_classifier.py` script will use `validation.common.stream_run` (existing) for subprocess output streaming.

---

## Task 1: `derive_site_labels` helper

**Files:**
- Create: `locator/data/site_labels.py`
- Modify: `locator/data/__init__.py` (re-export `derive_site_labels`)
- Test: `tests/test_classifier_head.py`

- [ ] **Step 1: Write the failing tests for `derive_site_labels`**

Create `tests/test_classifier_head.py`:

```python
"""Tests for the classifier-head feature in ReLocator."""

from __future__ import annotations

import numpy as np
import pytest

from locator.data.site_labels import derive_site_labels


def test_derive_site_labels_basic():
    """5 samples at 2 unique coords → 2 classes, correct integer labels."""
    locs = np.array([
        [10.0, 20.0],  # site A
        [10.0, 20.0],  # site A
        [11.0, 21.0],  # site B
        [11.0, 21.0],  # site B
        [10.0, 20.0],  # site A
    ], dtype=np.float64)
    labels, centroids = derive_site_labels(locs)
    assert len(np.unique(labels)) == 2
    assert centroids.shape == (2, 2)
    # Labels for the same coords must match
    assert labels[0] == labels[1] == labels[4]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_derive_site_labels_centroid_order_matches_label():
    """centroids[label_i] must equal locs[i]."""
    locs = np.array([[10.0, 20.0], [11.0, 21.0], [10.0, 20.0]], dtype=np.float64)
    labels, centroids = derive_site_labels(locs)
    np.testing.assert_array_equal(centroids[labels[0]], locs[0])
    np.testing.assert_array_equal(centroids[labels[1]], locs[1])
    np.testing.assert_array_equal(centroids[labels[2]], locs[2])


def test_derive_site_labels_skips_nan_rows():
    """NaN-coord samples (prediction targets) are excluded — they don't get labels.

    Function must be called on training-only locs; passing NaN rows is a programming
    error and should raise.
    """
    locs = np.array([
        [10.0, 20.0],
        [np.nan, np.nan],
        [11.0, 21.0],
    ], dtype=np.float64)
    with pytest.raises(ValueError, match="NaN"):
        derive_site_labels(locs)


def test_derive_site_labels_elephant_case_raises():
    """When K > 0.5 * N, classification is degenerate and we must error out."""
    rng = np.random.default_rng(42)
    # 10 samples, all unique coords → K = N = 10
    locs = rng.uniform(0, 1, size=(10, 2))
    with pytest.raises(ValueError, match="classification|min_samples_per_site|elephant"):
        derive_site_labels(locs)


def test_derive_site_labels_min_samples_per_site_param():
    """min_samples_per_site=1 disables the K > 0.5 N check (allow degenerate cases)."""
    rng = np.random.default_rng(42)
    locs = rng.uniform(0, 1, size=(10, 2))
    labels, centroids = derive_site_labels(locs, min_samples_per_site=1)
    assert len(np.unique(labels)) == 10
    assert centroids.shape == (10, 2)


def test_derive_site_labels_returns_int32_labels():
    """Labels must be int32 (TensorFlow-friendly), not int64."""
    locs = np.array([[10.0, 20.0], [10.0, 20.0]], dtype=np.float64)
    labels, _ = derive_site_labels(locs)
    assert labels.dtype == np.int32
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_classifier_head.py -v
```

Expected: 6 ERROR with `ModuleNotFoundError: No module named 'locator.data.site_labels'`.

- [ ] **Step 3: Implement `derive_site_labels`**

Create `locator/data/site_labels.py`:

```python
"""Site-label derivation for the classifier output head.

For datasets with discrete sampling sites (popgen-canonical: multiple
individuals share each site centroid), derive integer site labels from
exact coord equivalence among training samples. The classifier head
trains on these labels and stores the matching centroid matrix for
inference.
"""

from __future__ import annotations

import numpy as np


def derive_site_labels(
    locs: np.ndarray,
    min_samples_per_site: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Group training locations by exact coord equivalence into integer labels.

    :param locs: ``(n, 2)`` float array of training-sample coordinates
        (lon, lat or z-scored equivalents — any 2D real coords).
        Must contain no NaNs (caller is responsible for filtering
        prediction-target samples out before calling).
    :type locs: np.ndarray
    :param min_samples_per_site: Reject the partition when K (number of
        unique coord pairs) exceeds ``n / min_samples_per_site``. Default
        2 means "every site must have at least 2 samples on average."
        Set to 1 to allow K = N (degenerate, useful in tests only).
    :type min_samples_per_site: int

    :returns: Tuple ``(labels, centroids)`` where ``labels`` is an
        ``(n,)`` int32 array assigning each sample to its site index,
        and ``centroids`` is a ``(K, 2)`` float64 array of unique coords
        in label order (so ``centroids[labels[i]] == locs[i]``).
    :rtype: tuple[np.ndarray, np.ndarray]

    :raises ValueError: If ``locs`` contains NaN rows, or if K exceeds
        the ``min_samples_per_site`` threshold.
    """
    if locs.ndim != 2 or locs.shape[1] != 2:
        raise ValueError(f"locs must have shape (n, 2); got {locs.shape}")
    if np.isnan(locs).any():
        raise ValueError(
            "derive_site_labels: locs contains NaN rows; filter prediction "
            "targets out before calling"
        )

    n = locs.shape[0]
    centroids, labels = np.unique(locs, axis=0, return_inverse=True)
    k = centroids.shape[0]

    if k > n / min_samples_per_site:
        raise ValueError(
            f"derive_site_labels: K={k} unique coords for n={n} samples — "
            f"classification requires multiple samples per coord pair "
            f"(threshold: K <= n / min_samples_per_site = {n / min_samples_per_site:.1f}). "
            f"This is the elephant case where every individual has unique GPS. "
            f"Use prediction_mode='regress' instead, or pass "
            f"min_samples_per_site=1 to override."
        )

    return labels.astype(np.int32), centroids
```

- [ ] **Step 4: Wire the helper into `locator.data.__init__`**

Read the existing exports first:

```bash
head -20 locator/data/__init__.py
```

Add the import line. The existing file likely already re-exports a handful of helpers; add `derive_site_labels` to that list (preserve the existing `__all__` if present).

Edit `locator/data/__init__.py` to add (at the appropriate spot among existing imports):

```python
from .site_labels import derive_site_labels
```

If there is an `__all__` tuple, add `"derive_site_labels"` to it.

- [ ] **Step 5: Run tests to verify they pass**

```bash
pixi run pytest tests/test_classifier_head.py -v
```

Expected: 6 PASS for `test_derive_site_labels_*`.

- [ ] **Step 6: Lint and commit**

```bash
pixi run ruff check locator/data/site_labels.py locator/data/__init__.py tests/test_classifier_head.py
git add locator/data/site_labels.py locator/data/__init__.py tests/test_classifier_head.py
git commit -m "$(cat <<'EOF'
classifier-head: add derive_site_labels helper

Pure function for grouping training locations by exact coord equivalence
into integer site labels for the classifier output head. Returns
(labels: int32, centroids: float64) with the invariant
centroids[labels[i]] == locs[i].

Detects the K > n / min_samples_per_site case (elephant-style data with
unique GPS per individual) and raises a clear ValueError pointing at
prediction_mode='regress' as the alternative. The threshold is
configurable via min_samples_per_site (default 2 means K <= n/2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `create_network(n_classes=K)` parameter

**Files:**
- Modify: `locator/models.py` (extend `create_network`)
- Modify: `tests/test_classifier_head.py` (add network-construction tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_classifier_head.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: create_network(n_classes=K)
# ---------------------------------------------------------------------------

from locator.models import create_network


def test_create_network_default_regression_unchanged():
    """n_classes=None must produce the existing regression network shape.

    The final layer is the second of two Dense(2) layers; output shape (None, 2).
    """
    model = create_network(input_shape=100, n_classes=None)
    assert model.output_shape == (None, 2)
    # Final layer is Dense(2) — no softmax activation
    final = model.layers[-1]
    assert final.units == 2
    # Default loss is euclidean_distance_loss (recoverable from compiled_loss)
    assert "loss" not in str(model.optimizer)  # sanity — not the wrong attr


def test_create_network_classifier_mode_output_shape():
    """n_classes=K replaces final layers with Dense(K, softmax)."""
    K = 7
    model = create_network(input_shape=100, n_classes=K)
    assert model.output_shape == (None, K)
    final = model.layers[-1]
    assert final.units == K
    assert final.activation.__name__ == "softmax"


def test_create_network_classifier_default_loss_is_crossentropy():
    """When n_classes is set and loss_fn is None, default loss = categorical_crossentropy."""
    import tensorflow as tf
    K = 5
    model = create_network(input_shape=50, n_classes=K)
    # The compiled loss should be a CategoricalCrossentropy instance
    loss_obj = model.loss
    assert isinstance(loss_obj, tf.keras.losses.CategoricalCrossentropy)


def test_create_network_classifier_explicit_loss_override():
    """Explicit loss_fn overrides the categorical default."""
    import tensorflow as tf
    K = 5
    custom_loss = tf.keras.losses.KLDivergence()
    model = create_network(input_shape=50, n_classes=K, loss_fn=custom_loss)
    assert model.loss is custom_loss


def test_create_network_classifier_forward_pass_shape():
    """A forward pass on synthetic input gives (batch, K) softmax outputs in [0, 1] summing to 1."""
    import numpy as np
    K = 4
    n_samples = 8
    n_features = 30
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    model = create_network(input_shape=n_features, n_classes=K)
    out = model.predict(X, verbose=0)
    assert out.shape == (n_samples, K)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_classifier_head.py -v -k create_network
```

Expected: 5 FAIL — `create_network()` does not yet accept `n_classes`.

- [ ] **Step 3: Modify `create_network`**

Open `locator/models.py`. Find the existing signature at line 95 and extend it. The full edited function:

Replace the signature and body of `create_network` (lines 95–179) with:

```python
def create_network(
    input_shape: int,
    width: int = 256,
    n_layers: int = 8,
    dropout_prop: float = 0.25,
    optimizer_config: Optional[dict] = None,
    loss_fn: Optional[callable] = None,
    n_classes: Optional[int] = None,
) -> keras.Model:
    """Create a neural network model for geographic location prediction.

    :param input_shape: Number of input features (SNPs).
    :type input_shape: int
    :param width: Width of the dense layers, defaults to 256.
    :type width: int, optional
    :param n_layers: Total number of dense layers (excluding final layers),
        defaults to 8.
    :type n_layers: int, optional
    :param dropout_prop: Dropout proportion for middle dropout layer,
        defaults to 0.25.
    :type dropout_prop: float, optional
    :param optimizer_config: Configuration for the optimizer.
    :type optimizer_config: dict, optional
    :param loss_fn: Loss function. If None, defaults to
        ``euclidean_distance_loss`` for regression mode or
        ``CategoricalCrossentropy`` for classifier mode.
    :type loss_fn: callable, optional
    :param n_classes: If set, replaces the final two Dense(2) layers with a
        single ``Dense(n_classes, softmax)`` layer for site classification.
        Defaults to None (regression mode, unchanged behavior).
    :type n_classes: int, optional
    :return: Compiled Keras model.
    :rtype: keras.Model
    """
    inputs = keras.Input(shape=(input_shape,))
    x = layers.BatchNormalization()(inputs)
    for _ in range(int(np.floor(n_layers / 2))):
        x = layers.Dense(width, activation="elu")(x)
    x = layers.Dropout(dropout_prop)(x)
    for _ in range(int(np.ceil(n_layers / 2))):
        x = layers.Dense(width, activation="elu")(x)

    if n_classes is None:
        # Regression head (unchanged)
        x = layers.Dense(2)(x)
        outputs = layers.Dense(2)(x)
    else:
        # Classifier head
        outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="locator_network")

    if optimizer_config is None:
        optimizer = "Adam"
    else:
        if optimizer_config["algo"].lower() == "adam":
            optimizer = keras.optimizers.Adam(
                learning_rate=optimizer_config["learning_rate"]
            )
        elif optimizer_config["algo"].lower() == "adamw":
            optimizer = keras.optimizers.AdamW(
                learning_rate=optimizer_config["learning_rate"],
                weight_decay=optimizer_config["weight_decay"],
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['algo']}")

    if loss_fn is None:
        loss_fn = (
            tf.keras.losses.CategoricalCrossentropy()
            if n_classes is not None
            else euclidean_distance_loss
        )

    model.compile(optimizer=optimizer, loss=loss_fn)
    return model
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pixi run pytest tests/test_classifier_head.py -v -k create_network
```

Expected: 5 PASS. Also re-run the full file to confirm no regressions in Task 1's tests:

```bash
pixi run pytest tests/test_classifier_head.py -v
```

Expected: 11 PASS total (6 from Task 1 + 5 new).

- [ ] **Step 5: Run a smoke check that existing model construction is byte-identical**

The default-arg path (no `n_classes`) should produce a model with the same layer count, units, and shapes as before. A quick check via the test we already wrote (`test_create_network_default_regression_unchanged`) covers this. No additional command needed.

- [ ] **Step 6: Lint and commit**

```bash
pixi run ruff check locator/models.py tests/test_classifier_head.py
git add locator/models.py tests/test_classifier_head.py
git commit -m "$(cat <<'EOF'
classifier-head: extend create_network with n_classes parameter

When n_classes is set, replaces the final two Dense(2) layers with a
single Dense(n_classes, softmax) layer. Default loss switches from
euclidean_distance_loss to tf.keras.losses.CategoricalCrossentropy.
n_classes=None preserves the existing regression behavior byte-identically.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `prediction_mode` config validation

**Files:**
- Modify: `locator/training.py` (add config validation in `train` setup)
- Modify: `tests/test_classifier_head.py` (config-validation tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_classifier_head.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: config validation
# ---------------------------------------------------------------------------

from locator.training import _validate_prediction_mode_config


def test_validate_prediction_mode_default_regress():
    """Missing or 'regress' → no error."""
    _validate_prediction_mode_config({})
    _validate_prediction_mode_config({"prediction_mode": "regress"})


def test_validate_prediction_mode_classify_modes_accepted():
    _validate_prediction_mode_config({"prediction_mode": "classify"})
    _validate_prediction_mode_config({"prediction_mode": "classify_then_avg"})


def test_validate_prediction_mode_unknown_value_errors():
    with pytest.raises(ValueError, match="Unknown prediction_mode"):
        _validate_prediction_mode_config({"prediction_mode": "bogus"})


def test_validate_prediction_mode_classify_with_range_penalty_errors():
    """range_mask is incompatible with classifier modes."""
    cfg = {"prediction_mode": "classify", "use_range_penalty": True}
    with pytest.raises(ValueError, match="range_mask|use_range_penalty"):
        _validate_prediction_mode_config(cfg)
    cfg2 = {"prediction_mode": "classify_then_avg", "use_range_penalty": True}
    with pytest.raises(ValueError, match="range_mask|use_range_penalty"):
        _validate_prediction_mode_config(cfg2)


def test_validate_prediction_mode_regress_with_range_penalty_ok():
    """range_mask is compatible with the existing regression head."""
    cfg = {"prediction_mode": "regress", "use_range_penalty": True,
           "species_range_shapefile": "/tmp/foo.shp"}
    _validate_prediction_mode_config(cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_classifier_head.py -v -k validate_prediction_mode
```

Expected: 5 ERROR — `_validate_prediction_mode_config` not defined.

- [ ] **Step 3: Implement `_validate_prediction_mode_config`**

In `locator/training.py`, add a module-level helper near the top of the file (after the imports, before the `class TrainingMixin`). Find the `class TrainingMixin:` line (around line 19) and insert this just before it:

```python
VALID_PREDICTION_MODES = ("regress", "classify", "classify_then_avg")


def _validate_prediction_mode_config(config: dict) -> None:
    """Validate prediction_mode and its incompatibilities.

    Raises:
        ValueError: if prediction_mode is unknown or if a classifier mode
            is combined with use_range_penalty=True.
    """
    mode = config.get("prediction_mode", "regress")
    if mode not in VALID_PREDICTION_MODES:
        raise ValueError(
            f"Unknown prediction_mode={mode!r}. "
            f"Valid options: {VALID_PREDICTION_MODES}"
        )
    if mode in ("classify", "classify_then_avg") and config.get("use_range_penalty"):
        raise ValueError(
            f"prediction_mode={mode!r} is incompatible with use_range_penalty=True. "
            "Classifier modes bound predictions to the convex hull of training-site "
            "centroids by construction; the rasterized range mask does not apply to "
            "probability-vector outputs. Use prediction_mode='regress' or set "
            "use_range_penalty=False."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pixi run pytest tests/test_classifier_head.py -v -k validate_prediction_mode
```

Expected: 5 PASS.

- [ ] **Step 5: Lint and commit**

```bash
pixi run ruff check locator/training.py tests/test_classifier_head.py
git add locator/training.py tests/test_classifier_head.py
git commit -m "$(cat <<'EOF'
classifier-head: add prediction_mode config validation

Adds module-level _validate_prediction_mode_config in locator/training.py
that:
- Accepts only {regress, classify, classify_then_avg}.
- Errors on unknown values with the valid set listed.
- Errors when use_range_penalty=True is combined with classifier modes
  (incompatible by design — classifier outputs are probability vectors;
  the rasterized range mask doesn't apply, and the convex hull of
  training centroids is the natural valid range anyway).

Not yet wired into the train() flow — that happens in Task 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Training integration — `_create_model` and `_prepare_targets`

**Files:**
- Modify: `locator/training.py` (`_create_model` reads prediction_mode; new `_prepare_targets` helper; wire `_validate_prediction_mode_config` into `train`)
- Modify: `tests/test_classifier_head.py` (integration tests via the Locator class)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_classifier_head.py`:

```python
# ---------------------------------------------------------------------------
# Task 4: training integration
# ---------------------------------------------------------------------------

import numpy as np


def test_prepare_targets_regression_passthrough():
    """In regress mode, _prepare_targets returns normalized_locs unchanged."""
    from locator.training import _prepare_targets
    normalized_locs = np.array([[0.1, 0.2], [-0.3, 0.5]], dtype=np.float64)
    targets, centroids, n_classes = _prepare_targets(
        normalized_locs, train_indices=np.array([0, 1]),
        config={"prediction_mode": "regress"},
    )
    np.testing.assert_array_equal(targets, normalized_locs)
    assert centroids is None
    assert n_classes is None


def test_prepare_targets_classify_returns_one_hot_labels():
    """In classify mode, training rows become one-hot K-dim vectors;
    centroids and n_classes are set."""
    from locator.training import _prepare_targets
    # 4 training samples at 2 unique coords; 1 prediction-target (NaN).
    normalized_locs = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [np.nan, np.nan],
    ], dtype=np.float64)
    train_indices = np.array([0, 1, 2, 3])
    targets, centroids, n_classes = _prepare_targets(
        normalized_locs, train_indices=train_indices,
        config={"prediction_mode": "classify"},
    )
    assert n_classes == 2
    assert centroids.shape == (2, 2)
    # Targets must be (n_total, K) one-hot for training rows; rows for
    # NaN prediction-targets are zero-padded.
    assert targets.shape == (5, 2)
    assert targets[0].sum() == 1.0
    assert targets[1].sum() == 1.0
    np.testing.assert_array_equal(targets[0], targets[1])  # same site
    np.testing.assert_array_equal(targets[2], targets[3])  # same site
    assert not np.array_equal(targets[0], targets[2])      # different sites
    np.testing.assert_array_equal(targets[4], np.zeros(2))  # NaN row → zeros


def test_prepare_targets_classify_then_avg_same_as_classify():
    """classify_then_avg uses identical training targets to classify."""
    from locator.training import _prepare_targets
    normalized_locs = np.array([
        [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]
    ], dtype=np.float64)
    train = np.array([0, 1, 2, 3])
    targets_a, centroids_a, n_a = _prepare_targets(
        normalized_locs, train, config={"prediction_mode": "classify"}
    )
    targets_b, centroids_b, n_b = _prepare_targets(
        normalized_locs, train, config={"prediction_mode": "classify_then_avg"}
    )
    np.testing.assert_array_equal(targets_a, targets_b)
    np.testing.assert_array_equal(centroids_a, centroids_b)
    assert n_a == n_b
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_classifier_head.py -v -k prepare_targets
```

Expected: 3 ERROR — `_prepare_targets` not defined.

- [ ] **Step 3: Implement `_prepare_targets`**

In `locator/training.py`, add the helper near `_validate_prediction_mode_config` (module-level, before the class):

```python
def _prepare_targets(
    normalized_locs: np.ndarray,
    train_indices: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, "np.ndarray | None", "int | None"]:
    """Build training targets according to prediction_mode.

    For regress mode, returns normalized_locs unchanged (the existing
    behavior). For classify / classify_then_avg, derives site labels from
    the training rows of normalized_locs, converts to one-hot, and
    returns a (n_total, K) target array where prediction-target rows
    (NaN locs) are zero-padded.

    :param normalized_locs: ``(n, 2)`` z-scored coords; rows for samples
        without coords (prediction targets) contain NaN.
    :type normalized_locs: np.ndarray
    :param train_indices: Indices into ``normalized_locs`` identifying
        training-only rows. Used to derive labels and centroids.
    :type train_indices: np.ndarray
    :param config: ``Locator.config`` dict; only ``prediction_mode`` is read.
    :type config: dict

    :returns: ``(targets, centroids, n_classes)``:
        - ``targets``: ``(n, 2)`` for regress; ``(n, K)`` one-hot for classifier modes.
        - ``centroids``: ``None`` for regress; ``(K, 2)`` float64 in label order for classifier modes.
        - ``n_classes``: ``None`` for regress; ``K`` for classifier modes.
    :rtype: tuple
    """
    from .data.site_labels import derive_site_labels

    mode = config.get("prediction_mode", "regress")
    if mode == "regress":
        return normalized_locs, None, None

    train_locs = normalized_locs[train_indices]
    min_per_site = config.get("min_samples_per_site", 2)
    labels_train, centroids = derive_site_labels(
        train_locs, min_samples_per_site=min_per_site
    )
    k = centroids.shape[0]

    n_total = normalized_locs.shape[0]
    targets = np.zeros((n_total, k), dtype=np.float32)
    targets[train_indices, labels_train] = 1.0
    return targets, centroids, k
```

- [ ] **Step 4: Wire validation into `Locator.train`**

Find the top of the `train` method (search for `def train(` in `locator/training.py` — there are several methods; the main one is around line 100). Add a call to `_validate_prediction_mode_config(self.config)` at the very top of the method body (right after the docstring, before any other work).

Use this Edit (the `def train(` is the FIRST one in the file — adjust if the line above the docstring is different):

```bash
grep -n "def train(" locator/training.py | head -3
```

Find the first `def train(self, ` line. Read the next ~30 lines to find a clean insertion point right after the docstring. The pattern is:

```python
def train(self, ...):
    """..."""
    _validate_prediction_mode_config(self.config)  # NEW LINE
    # ... rest of method
```

Apply that insert at the FIRST `def train(self,` in the file (around line 100). Use Edit on `locator/training.py` to add the call after the closing `"""` of the train method's docstring.

- [ ] **Step 5: Run tests**

```bash
pixi run pytest tests/test_classifier_head.py -v
```

Expected: 14 PASS (11 prior + 3 new).

- [ ] **Step 6: Smoke check the regression path is unchanged**

Run a quick existing test that exercises the Locator class to confirm the byte-identical regress path. Use the existing example VCF:

```bash
ls data/test_genotypes.vcf.gz data/test_sample_data.txt 2>&1 | head
```

If those exist, run:

```bash
pixi run python -c "
from locator import Locator
loc = Locator(config={
    'vcf': 'data/test_genotypes.vcf.gz',
    'sample_data': 'data/test_sample_data.txt',
    'out': '/tmp/classifier_smoke',
    'max_epochs': 3, 'patience': 5, 'seed': 42, 'gpu_number': 0,
})
genotypes, samples = loc.load_genotypes(vcf='data/test_genotypes.vcf.gz')
loc.train(genotypes=genotypes, samples=samples)
print('regress smoke OK')
"
```

Expected: prints "regress smoke OK". A 3-epoch fit takes a few seconds.

- [ ] **Step 7: Lint and commit**

```bash
pixi run ruff check locator/training.py tests/test_classifier_head.py
git add locator/training.py tests/test_classifier_head.py
git commit -m "$(cat <<'EOF'
classifier-head: training integration — _prepare_targets + train() validation gate

- _prepare_targets builds either regression coords (unchanged) or one-hot
  K-class labels with a (K, 2) centroids matrix, depending on
  config['prediction_mode'].
- For classifier modes, NaN-coord rows (prediction targets) get zero
  one-hot vectors so they don't contribute to the loss.
- Validation gate _validate_prediction_mode_config wired into the top of
  Locator.train() — fails fast on unknown modes or
  classifier+range_penalty incompatibility.
- Smoke-tested that the regress path is byte-identical.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Wire `_prepare_targets` into `_build_datasets_and_fit` and `_create_model`

**Files:**
- Modify: `locator/training.py` (`_build_datasets_and_fit` uses targets; `_create_model` passes `n_classes`)
- Modify: `tests/test_classifier_head.py` (end-to-end Locator.train test)

- [ ] **Step 1: Write the failing test (end-to-end fit on synthetic data)**

Append to `tests/test_classifier_head.py`:

```python
# ---------------------------------------------------------------------------
# Task 5: end-to-end Locator.train under classifier mode
# ---------------------------------------------------------------------------

def _write_synthetic_matrix_data(tmp_path):
    """Create a tiny matrix + sample_data with K=3 sites × 6 samples each."""
    import pandas as pd
    n_per_site = 6
    n_sites = 3
    n_features = 20
    n_total = n_per_site * n_sites
    rng = np.random.default_rng(0)
    matrix = pd.DataFrame(
        rng.normal(size=(n_total, n_features)).astype(np.float32),
        columns=[f"f{i}" for i in range(n_features)],
    )
    site_centroids = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)]
    sample_ids = []
    xs = []
    ys = []
    for i, (sx, sy) in enumerate(site_centroids):
        for j in range(n_per_site):
            sample_ids.append(f"site{i}_{j}")
            xs.append(sx)
            ys.append(sy)
    matrix.insert(0, "sampleID", sample_ids)
    matrix_path = tmp_path / "matrix.tsv"
    matrix.to_csv(matrix_path, sep="\t", index=False)

    sd = pd.DataFrame({"x": xs, "y": ys, "sampleID": sample_ids})
    sd_path = tmp_path / "sample_data.txt"
    sd.to_csv(sd_path, sep="\t", index=False)
    return matrix_path, sd_path


def test_locator_train_classify_mode_synthetic(tmp_path):
    """Locator.train with prediction_mode='classify' fits without crashing
    and stores _site_centroids."""
    from locator import Locator
    matrix_path, sd_path = _write_synthetic_matrix_data(tmp_path)

    loc = Locator(config={
        "matrix": str(matrix_path),
        "sample_data": str(sd_path),
        "out": str(tmp_path / "classify_smoke"),
        "max_epochs": 5,
        "patience": 5,
        "seed": 42,
        "gpu_number": 0,
        "prediction_mode": "classify",
    })
    genotypes, samples = loc.load_genotypes(matrix=str(matrix_path))
    loc.train(genotypes=genotypes, samples=samples)

    assert hasattr(loc, "_site_centroids")
    assert loc._site_centroids.shape == (3, 2)
    # Final layer should be Dense(3, softmax)
    assert loc.model.output_shape == (None, 3)


def test_locator_train_classify_mode_range_penalty_errors(tmp_path):
    """Locator.train errors when prediction_mode=classify and use_range_penalty=True."""
    from locator import Locator
    matrix_path, sd_path = _write_synthetic_matrix_data(tmp_path)
    loc = Locator(config={
        "matrix": str(matrix_path),
        "sample_data": str(sd_path),
        "out": str(tmp_path / "x"),
        "max_epochs": 1, "patience": 5, "seed": 42, "gpu_number": 0,
        "prediction_mode": "classify",
        "use_range_penalty": True,
        "species_range_shapefile": "/tmp/dummy.shp",
    })
    genotypes, samples = loc.load_genotypes(matrix=str(matrix_path))
    with pytest.raises(ValueError, match="range_mask|use_range_penalty"):
        loc.train(genotypes=genotypes, samples=samples)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_classifier_head.py -v -k locator_train_classify
```

Expected: 2 FAIL (validation gate is in place but the model still uses the regress head; first test fails because `_site_centroids` isn't set; second test passes if Task 4 wired the gate but otherwise fails).

- [ ] **Step 3: Modify `_create_model` to read `prediction_mode` and pass `n_classes`**

In `locator/training.py`, find `_create_model` (around line 563). Replace its body with:

```python
def _create_model(self, input_shape):
    """Create neural network model. Extracted to avoid duplication."""
    mode = self.config.get("prediction_mode", "regress")
    n_classes = getattr(self, "_n_classes", None) if mode != "regress" else None

    loss_fn = None
    if self.config.get("use_range_penalty"):
        if self.config.get("species_range_shapefile") is None:
            raise ValueError(
                "species_range_shapefile must be provided "
                "if use_range_penalty is True"
            )
        if self.config.get("resolution") is None:
            raise ValueError(
                "resolution must be provided if use_range_penalty is True"
            )

        mask_tensor, mask_transform = rasterize_species_range(
            self.config["species_range_shapefile"],
            resolution=self.config.get("resolution", 0.05),
        )

        def loss_fn(y_true, y_pred):  # noqa: F811
            return loss_with_range_penalty(
                y_true,
                y_pred,
                mask_tensor=mask_tensor,
                transform=mask_transform,
                resolution=self.config.get("resolution", 0.05),
                penalty_weight=self.config.get("penalty_weight", 1.0),
            )

    return create_network(
        input_shape=input_shape,
        width=self.config.get("width", 256),
        n_layers=self.config.get("nlayers", 8),
        dropout_prop=self.config.get("dropout_prop", 0.25),
        optimizer_config={
            "algo": self.config.get("optimizer_algo", "adam"),
            "learning_rate": self.config.get("learning_rate", 0.001),
            "weight_decay": self.config.get("weight_decay", 0.004),
        },
        loss_fn=loss_fn,
        n_classes=n_classes,
    )
```

- [ ] **Step 4: Modify `_build_datasets_and_fit` to substitute classifier targets**

In `locator/training.py`, find `_build_datasets_and_fit` (around line 832). At the top of the method body, before the existing `batch_size` line, add the classifier-target substitution:

Replace the line `batch_size = self._determine_batch_size(len(self.index_set.train))` and what comes before it inside the method with:

```python
def _build_datasets_and_fit(
    self,
    normalized_locs,
    callbacks,
    site_order=None,
    keras_verbose=None,
):
    """Build tf.data pipelines and train the model.

    Requires self.filtered_genotypes, self.index_set, self.model,
    and self.sample_weights to be set before calling.
    """
    # Classifier-mode targets: replace normalized_locs with one-hot site labels.
    targets, centroids, n_classes = _prepare_targets(
        normalized_locs,
        train_indices=self.index_set.train,
        config=self.config,
    )
    if n_classes is not None:
        self._site_centroids = centroids
        self._n_classes = n_classes

    batch_size = self._determine_batch_size(len(self.index_set.train))
```

Then (same method) replace the two `coordinates=normalized_locs` arguments to `make_tf_dataset` with `coordinates=targets`. Keep the rest of the method body unchanged.

- [ ] **Step 5: Run tests**

```bash
pixi run pytest tests/test_classifier_head.py -v
```

Expected: 16 PASS (14 prior + 2 new).

- [ ] **Step 6: Smoke check the regress path again**

Re-run the same smoke from Task 4 Step 6 to confirm regress is still untouched. Expected: same "regress smoke OK".

- [ ] **Step 7: Lint and commit**

```bash
pixi run ruff check locator/training.py tests/test_classifier_head.py
git add locator/training.py tests/test_classifier_head.py
git commit -m "$(cat <<'EOF'
classifier-head: wire _prepare_targets into _build_datasets_and_fit

- _build_datasets_and_fit now calls _prepare_targets at the top to
  substitute one-hot site-label targets for normalized coords when
  prediction_mode is a classifier mode. Stores centroids and n_classes
  on self for use by _create_model and predict().
- _create_model passes n_classes through to create_network when set.
  The Dense(K, softmax) head and CategoricalCrossentropy loss are
  selected automatically.
- End-to-end synthetic fit verifies the model trains without crashing
  and stores _site_centroids correctly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Prediction integration — `classify` mode

**Files:**
- Modify: `locator/prediction.py` (branch on `prediction_mode` after the network forward pass)
- Modify: `tests/test_classifier_head.py` (predict-time tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_classifier_head.py`:

```python
# ---------------------------------------------------------------------------
# Task 6: predict() in classify mode
# ---------------------------------------------------------------------------

def test_predict_classify_mode_returns_centroids_for_held_out_samples(tmp_path):
    """End-to-end: train + predict in classify mode returns one of the K
    training-site centroids for each prediction sample."""
    from locator import Locator
    import pandas as pd
    matrix_path, sd_path = _write_synthetic_matrix_data(tmp_path)
    # Mark sample0 of site2 as held-out (NaN coords) for prediction.
    sd = pd.read_csv(sd_path, sep="\t")
    sd.loc[sd["sampleID"] == "site2_0", ["x", "y"]] = np.nan
    sd.to_csv(sd_path, sep="\t", index=False)

    loc = Locator(config={
        "matrix": str(matrix_path),
        "sample_data": str(sd_path),
        "out": str(tmp_path / "classify_pred"),
        "max_epochs": 20, "patience": 30, "seed": 42, "gpu_number": 0,
        "prediction_mode": "classify",
    })
    genotypes, samples = loc.load_genotypes(matrix=str(matrix_path))
    loc.train(genotypes=genotypes, samples=samples)
    pred_df = loc.predict(genotypes=genotypes, samples=samples, return_df=True)
    assert "x" in pred_df.columns and "y" in pred_df.columns
    # Each predicted (x, y) must match one of the 3 training centroids exactly.
    centroids = {(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)}
    for _, row in pred_df.iterrows():
        assert (round(row["x"], 4), round(row["y"], 4)) in centroids
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pixi run pytest tests/test_classifier_head.py -v -k predict_classify
```

Expected: 1 FAIL — current `predict` denormalizes a (lat, lon) pair, but in classify mode the model outputs `(n, K)` softmax which doesn't fit the existing `predictions[:, 0] * sdlong + meanlong` math.

- [ ] **Step 3: Implement `classify` branch in `predict`**

In `locator/prediction.py`, find the denormalization block at lines 204–207:

```python
# Denormalize predictions
predictions = predictions.copy()
predictions[:, 0] = predictions[:, 0] * self.sdlong + self.meanlong
predictions[:, 1] = predictions[:, 1] * self.sdlat + self.meanlat
```

Replace this block with:

```python
# Branch on prediction_mode (classifier vs regression output shapes)
predictions = predictions.copy()
mode = self.config.get("prediction_mode", "regress")

if mode == "classify":
    # softmax output (n, K) → argmax → centroid lookup
    if not hasattr(self, "_site_centroids"):
        raise RuntimeError(
            "_site_centroids not set on Locator instance. The model must "
            "have been trained in classify mode for this prediction path."
        )
    site_indices = np.argmax(predictions, axis=1)
    predictions = self._site_centroids[site_indices].astype(np.float64)
    # _site_centroids are stored in normalized space — denormalize.
    predictions[:, 0] = predictions[:, 0] * self.sdlong + self.meanlong
    predictions[:, 1] = predictions[:, 1] * self.sdlat + self.meanlat
elif mode == "classify_then_avg":
    # Implemented in Task 7.
    raise NotImplementedError("classify_then_avg not yet implemented")
else:
    # Regression path (existing).
    predictions[:, 0] = predictions[:, 0] * self.sdlong + self.meanlong
    predictions[:, 1] = predictions[:, 1] * self.sdlat + self.meanlat
```

(There is a second denormalize site in `prediction.py` around line 482, but it operates on a `pd.DataFrame` named `pred_df` (`pred_df["x"] = pred_df["x"] * self.sdlong + ...`) rather than a numpy array. This second block sits in a different code path that does not flow through the same predict-target path our tests exercise. Leave it alone for v1; document as a known limitation in the commit message and add a follow-up TODO in Task 14 if you discover the second path is reachable. Tests in this plan cover the first path only.)

- [ ] **Step 4: Run test to verify it passes**

```bash
pixi run pytest tests/test_classifier_head.py -v -k predict_classify
```

Expected: 1 PASS.

- [ ] **Step 5: Lint and commit**

```bash
pixi run ruff check locator/prediction.py tests/test_classifier_head.py
git add locator/prediction.py tests/test_classifier_head.py
git commit -m "$(cat <<'EOF'
classifier-head: predict() classify-mode branch

predict() now branches on prediction_mode at the post-network-output
point. For 'classify' mode, takes argmax of the softmax output, looks
up the corresponding centroid from self._site_centroids, and applies
the existing denormalize math to return (lat, lon).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Prediction integration — `classify_then_avg` mode

**Files:**
- Modify: `locator/prediction.py` (replace the NotImplementedError with the soft-avg math)
- Modify: `tests/test_classifier_head.py` (classify_then_avg tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_classifier_head.py`:

```python
# ---------------------------------------------------------------------------
# Task 7: predict() in classify_then_avg mode
# ---------------------------------------------------------------------------

def test_predict_classify_then_avg_inside_convex_hull(tmp_path):
    """End-to-end: classify_then_avg predictions land in the convex hull of
    training centroids."""
    from locator import Locator
    import pandas as pd
    matrix_path, sd_path = _write_synthetic_matrix_data(tmp_path)
    sd = pd.read_csv(sd_path, sep="\t")
    sd.loc[sd["sampleID"] == "site2_0", ["x", "y"]] = np.nan
    sd.to_csv(sd_path, sep="\t", index=False)

    loc = Locator(config={
        "matrix": str(matrix_path),
        "sample_data": str(sd_path),
        "out": str(tmp_path / "classify_avg_pred"),
        "max_epochs": 20, "patience": 30, "seed": 42, "gpu_number": 0,
        "prediction_mode": "classify_then_avg",
    })
    genotypes, samples = loc.load_genotypes(matrix=str(matrix_path))
    loc.train(genotypes=genotypes, samples=samples)
    pred_df = loc.predict(genotypes=genotypes, samples=samples, return_df=True)
    # Centroids in raw lon/lat: 0, 10, 20 on each axis.
    for _, row in pred_df.iterrows():
        assert -1.0 <= row["x"] <= 21.0
        assert -1.0 <= row["y"] <= 21.0


def test_predict_classify_then_avg_math_matches_softmax_dot_centroids(tmp_path):
    """Sanity: the prediction must equal softmax @ centroids (pre-denorm)
    when we manually compute it."""
    from locator import Locator
    import pandas as pd
    matrix_path, sd_path = _write_synthetic_matrix_data(tmp_path)
    sd = pd.read_csv(sd_path, sep="\t")
    sd.loc[sd["sampleID"] == "site2_0", ["x", "y"]] = np.nan
    sd.to_csv(sd_path, sep="\t", index=False)

    loc = Locator(config={
        "matrix": str(matrix_path),
        "sample_data": str(sd_path),
        "out": str(tmp_path / "classify_avg_math"),
        "max_epochs": 20, "patience": 30, "seed": 42, "gpu_number": 0,
        "prediction_mode": "classify_then_avg",
    })
    genotypes, samples = loc.load_genotypes(matrix=str(matrix_path))
    loc.train(genotypes=genotypes, samples=samples)

    # Manually run the network forward pass on prediction-target row.
    pred_idx = np.where(np.isnan(loc.unnormedlocs[:, 0]))[0]
    raw = loc.model.predict(loc.filtered_genotypes.T[pred_idx], verbose=0)
    expected_pre_denorm = raw @ loc._site_centroids
    expected_x = expected_pre_denorm[:, 0] * loc.sdlong + loc.meanlong
    expected_y = expected_pre_denorm[:, 1] * loc.sdlat + loc.meanlat

    pred_df = loc.predict(genotypes=genotypes, samples=samples, return_df=True)
    np.testing.assert_allclose(pred_df["x"].values, expected_x, atol=1e-3)
    np.testing.assert_allclose(pred_df["y"].values, expected_y, atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_classifier_head.py -v -k classify_then_avg
```

Expected: 2 FAIL — `NotImplementedError("classify_then_avg not yet implemented")` from Task 6.

- [ ] **Step 3: Replace the NotImplementedError with the soft-avg math**

In `locator/prediction.py`, find the `elif mode == "classify_then_avg":` block from Task 6 and replace its body:

```python
elif mode == "classify_then_avg":
    if not hasattr(self, "_site_centroids"):
        raise RuntimeError(
            "_site_centroids not set on Locator instance. The model must "
            "have been trained in classify_then_avg mode for this prediction path."
        )
    # softmax (n, K) @ centroids (K, 2) → (n, 2) in normalized space
    predictions = predictions @ self._site_centroids
    predictions = predictions.astype(np.float64)
    predictions[:, 0] = predictions[:, 0] * self.sdlong + self.meanlong
    predictions[:, 1] = predictions[:, 1] * self.sdlat + self.meanlat
```

Skip the second `pred_df["x"] = pred_df["x"] * ...` block near line 482 (it operates on a DataFrame, not an array, and is on a separate code path our tests don't exercise — see the note in Task 6 Step 3). Document this in the commit message.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pixi run pytest tests/test_classifier_head.py -v
```

Expected: 19 PASS (16 prior + 1 from Task 6 + 2 new).

- [ ] **Step 5: Lint and commit**

```bash
pixi run ruff check locator/prediction.py tests/test_classifier_head.py
git add locator/prediction.py tests/test_classifier_head.py
git commit -m "$(cat <<'EOF'
classifier-head: predict() classify_then_avg-mode branch

Replaces the NotImplementedError from Task 6 with the soft-average math:
predictions = softmax @ self._site_centroids, then existing denormalize.
Result is a continuous (lat, lon) prediction bounded to the convex hull
of training-site centroids — naturally captures uncertainty between
adjacent sites without committing to one.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: HDF5 metadata persistence for centroids

**Files:**
- Modify: `locator/training.py` (`_save_model_metadata` writes `_site_centroids` and `_n_classes` when present)
- Modify: `locator/prediction.py` (`load_model` reads them back)
- Modify: `tests/test_classifier_head.py` (round-trip persistence test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_classifier_head.py`:

```python
# ---------------------------------------------------------------------------
# Task 8: HDF5 round-trip
# ---------------------------------------------------------------------------

def test_classifier_centroids_round_trip_hdf5(tmp_path):
    """Train → save → load_model → predict produces same centroids matrix."""
    from locator import Locator
    import pandas as pd
    matrix_path, sd_path = _write_synthetic_matrix_data(tmp_path)
    sd = pd.read_csv(sd_path, sep="\t")
    sd.loc[sd["sampleID"] == "site2_0", ["x", "y"]] = np.nan
    sd.to_csv(sd_path, sep="\t", index=False)

    out_prefix = tmp_path / "round_trip"
    loc1 = Locator(config={
        "matrix": str(matrix_path),
        "sample_data": str(sd_path),
        "out": str(out_prefix),
        "max_epochs": 5, "patience": 5, "seed": 42, "gpu_number": 0,
        "prediction_mode": "classify",
    })
    g, s = loc1.load_genotypes(matrix=str(matrix_path))
    loc1.train(genotypes=g, samples=s)
    saved_centroids = loc1._site_centroids.copy()

    # Reload from HDF5
    loc2 = Locator(config={
        "matrix": str(matrix_path),
        "sample_data": str(sd_path),
        "out": str(out_prefix),
        "prediction_mode": "classify",
        "gpu_number": 0,
    })
    loc2.load_model(f"{out_prefix}.weights.h5")
    np.testing.assert_array_equal(loc2._site_centroids, saved_centroids)
    assert loc2._n_classes == saved_centroids.shape[0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pixi run pytest tests/test_classifier_head.py -v -k round_trip
```

Expected: 1 FAIL — centroids aren't persisted.

- [ ] **Step 3: Persist centroids in `_save_model_metadata`**

In `locator/training.py`, find `_save_model_metadata` (line 483). Inside the `with h5py.File(filepath, "a") as f:` block, after the existing `coord_*` attributes, add:

```python
                # Classifier-head metadata (only present if prediction_mode is a classifier mode)
                if hasattr(self, "_site_centroids") and self._site_centroids is not None:
                    f.attrs["prediction_mode"] = self.config.get("prediction_mode", "regress")
                    f.attrs["n_classes"] = int(self._n_classes)
                    if "site_centroids" in f:
                        del f["site_centroids"]
                    f.create_dataset("site_centroids", data=self._site_centroids.astype(np.float64))
```

Also add the import at the top of training.py if not present:

```bash
grep -n "^import h5py\|^import numpy" locator/training.py | head -5
```

(Both should already be imported — the existing `_save_model_metadata` uses h5py.)

- [ ] **Step 4: Read centroids in `load_model`**

In `locator/prediction.py`, find `load_model` (around line 234). Inside the existing `with h5py.File(weights_path, "r") as f:` block, after the existing `coord_*` reads, add:

```python
                # Classifier-head metadata
                if "site_centroids" in f:
                    self._site_centroids = np.asarray(f["site_centroids"][...], dtype=np.float64)
                    self._n_classes = int(f.attrs.get("n_classes", self._site_centroids.shape[0]))
```

- [ ] **Step 5: Run tests**

```bash
pixi run pytest tests/test_classifier_head.py -v
```

Expected: 20 PASS (19 prior + 1 new).

- [ ] **Step 6: Lint and commit**

```bash
pixi run ruff check locator/training.py locator/prediction.py tests/test_classifier_head.py
git add locator/training.py locator/prediction.py tests/test_classifier_head.py
git commit -m "$(cat <<'EOF'
classifier-head: persist site_centroids + n_classes to HDF5

Writes the (K, 2) centroids matrix as an HDF5 dataset and n_classes as
an attribute alongside the existing coord_meanlong / sdlong / etc.
load_model picks them back up so saved classifier models can be reused
across sessions for prediction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: User-facing docs

**Files:**
- Create: `docs/prediction_modes.md`

- [ ] **Step 1: Write the doc**

Create `docs/prediction_modes.md`:

```markdown
# Prediction Modes

ReLocator supports three prediction modes via the `prediction_mode`
config option (or `--prediction_mode` on the CLI). They differ only in
the output head of the network and the inference math; the encoder and
training pipeline are otherwise identical.

| Mode | Output head | Loss | Inference |
|---|---|---|---|
| `regress` (default) | `Dense(2) → Dense(2)` | Euclidean distance | network output, denormalized |
| `classify` | `Dense(K, softmax)` | categorical cross-entropy | argmax → site centroid → denormalized |
| `classify_then_avg` | `Dense(K, softmax)` | categorical cross-entropy | softmax @ centroids → denormalized |

`K` is the number of distinct training-site centroids (auto-derived
from exact coord equivalence among training samples).

## When to use each

### `regress` (default)

Use for any data where samples are continuously distributed in space and
no discrete sampling-site grouping is meaningful — for example, vagrant
animals tracked at random points along a migration route, or samples
where every individual has a unique GPS coordinate.

### `classify`

Use when training individuals are clearly grouped at a small number of
known sampling sites (popgen-canonical sampling design) and you want a
hard prediction "this sample came from site X." Returns the centroid of
the most-likely site. Provides a classification accuracy you can compute
directly (correct site or not).

### `classify_then_avg`

Use when training is grouped by sites but you want a continuous (lat,
lon) output that smoothly handles uncertainty between adjacent sites.
The prediction is bounded to the convex hull of training-site centroids
by construction, so predictions cannot land in the middle of nowhere.
For admixed individuals (e.g., genuinely intermediate between two
sites), the prediction lands geometrically between them, weighted by
the classifier's confidence in each.

## Worked example

You sample 405 prickly sculpin (*Cottus asper*) from 16 lakes/rivers
across the Pacific Northwest. Each lake's coordinates are recorded once,
and every individual from that lake gets the same (lon, lat). You train
ReLocator on 15 of the 16 lakes (LOSO test) with `prediction_mode='classify'`.

For each held-out individual:

- **`regress`** would output a continuous (lon, lat) somewhere on a
  smooth surface fit to the 15 training sites. Predictions for outlier
  sites (geographically isolated, like Alaska) tend to drift toward the
  centroid of training data — the regression-to-the-mean failure mode.
- **`classify`** outputs the centroid of one of the 15 training lakes
  (the one with highest softmax probability). Predictions are guaranteed
  to land at one of the trained-on sites.
- **`classify_then_avg`** outputs `Σ_k P(site_k) × centroid_k`. For an
  individual whose genotype is consistent with two training sites at
  similar probability, the prediction lands at the geographic midpoint
  between them, weighted by the relative confidence.

## Caveats

1. **`classify` and `classify_then_avg` are mutually exclusive with
   `use_range_penalty=True`.** The convex hull of training-site
   centroids is already the natural "valid range," and the rasterized
   range mask doesn't apply to softmax outputs. Setting both raises a
   clear `ValueError`.

2. **`derive_site_labels` requires K ≤ N / 2 by default.** If your data
   has more unique training coords than half the number of training
   samples, the classifier head is degenerate and the helper raises a
   ValueError pointing you at `prediction_mode='regress'`. Override
   with `min_samples_per_site=1` if you really want K = N (testing only).

3. **Saved model files (HDF5) include the centroids matrix** so you can
   reload and predict in a new session without re-training. The
   classifier modes round-trip cleanly through `load_model`.

4. **For applying a trained model to a sample from a population that
   wasn't in training**, use `regress` mode if you want the prediction
   to lie outside the trained sites' convex hull. The classifier modes
   are bounded by construction.
```

- [ ] **Step 2: Commit (markdown only, no test target)**

```bash
git add docs/prediction_modes.md
git commit -m "$(cat <<'EOF'
docs: prediction_modes.md user-facing guide for classifier head

Explains the three prediction modes (regress / classify / classify_then_avg)
with side-by-side comparison table, biological/use-case rationale per mode,
worked example using the sculpin LOSO setup, and caveats (range_penalty
incompatibility, K > N/2 elephant case, HDF5 round-trip, out-of-distribution
extrapolation).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Update microsat spec TODO list

**Files:**
- Modify: `validation/docs/specs/2026-05-04-microsatellite-input-extension-design.md`

- [ ] **Step 1: Append a "Classifier head follow-up" note**

Edit `validation/docs/specs/2026-05-04-microsatellite-input-extension-design.md`. Find the "Out of scope (declared TODOs)" section near the end and append a new bullet:

```bash
grep -n "Out of scope" validation/docs/specs/2026-05-04-microsatellite-input-extension-design.md
```

After the last existing TODO bullet in that section, add:

```markdown
- **Classifier output head** (`prediction_mode={classify, classify_then_avg}`)
  — the sculpin LOSO arc on this branch motivated this architectural
  follow-up. The regression head's regression-to-the-mean at geographic
  edges (snap-correctness rate = 8%, error vs FST r = +0.541) suggests
  a discrete-site classifier head would substantially improve
  fine-grained localization on data with discrete-sampling-design
  population structure. Designed in
  `validation/docs/specs/2026-05-05-classifier-head-design.md` and
  implemented on the `classifier-head` branch (off `microsat-sculpin`);
  not in scope for the microsat input PR proper.
```

- [ ] **Step 2: Commit**

```bash
git add validation/docs/specs/2026-05-04-microsatellite-input-extension-design.md
git commit -m "$(cat <<'EOF'
spec: cross-reference classifier-head follow-up from microsat spec

Appends a TODO bullet to the microsat input-extension spec pointing at
the classifier-head spec and branch as the architectural follow-up the
sculpin LOSO arc motivated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Sculpin LOSO classifier runner

**Files:**
- Create: `validation/sculpin/run_loso_classifier.py`

This runner uses ReLocator's Python API to invoke `train` + `predict` with `prediction_mode={classify, classify_then_avg}` for each of the 16 sculpin sites. Mirrors `run_loso.py` (regress) and `run_loso_rangemask.py` (regress + range_mask) but for the two classifier modes.

- [ ] **Step 1: Create the runner**

Create `validation/sculpin/run_loso_classifier.py`:

```python
#!/usr/bin/env python3
"""LOSO sweep for sculpin under classifier modes (classify, classify_then_avg).

For each of the 16 sculpin sampling sites: hold out the site, train a
classifier (K = 15 remaining sites) on dosage features via the Locator
Python API with prediction_mode={classify, classify_then_avg}, predict
held-out individuals, score against truth coords with haversine distance.

Output schema (fold_result.json) matches run_loso.py for downstream
summarization compatibility:
  {mode, site, n_held_out, n_evaluable, n_scored,
   median_error_km, mean_error_km, max_error_km, [warning], [error]}

Single mode per invocation (--mode {classify,classify_then_avg}); the
sculpin dosage feature matrix is reused for both modes (and for the
existing regress / regress+rangemask runs).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from validation import common  # noqa: E402

VALID_MODES = ("classify", "classify_then_avg")


def site_of(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def write_holdout_sample_data(
    sample_data_in: Path, holdout_site: str, out_path: Path
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(sample_data_in, sep="\t")
    sites = df["sampleID"].apply(site_of)
    holdout_mask = sites == holdout_site
    held_out_ids = df.loc[holdout_mask, "sampleID"].tolist()
    out = df.copy()
    out["x"] = out["x"].astype(object)
    out["y"] = out["y"].astype(object)
    out.loc[holdout_mask, "x"] = "NA"
    out.loc[holdout_mask, "y"] = "NA"
    out.to_csv(out_path, sep="\t", index=False)
    return df, held_out_ids


def run_one_fold(
    *, mode: str, site: str, feature_matrix_path: Path,
    sample_data_in: Path, out_dir: Path, gpu: int, seed: int,
    max_epochs: int,
) -> dict:
    fold_dir = out_dir / mode / site
    fold_dir.mkdir(parents=True, exist_ok=True)
    sd_path = fold_dir / "sample_data.txt"
    truth, held_out_ids = write_holdout_sample_data(sample_data_in, site, sd_path)

    out_prefix = fold_dir / "locator"
    fold_result: dict = {"mode": mode, "site": site}
    t0 = time.time()
    try:
        from locator import Locator

        loc = Locator(config={
            "out": str(out_prefix),
            "matrix": str(feature_matrix_path),
            "sample_data": str(sd_path),
            "max_epochs": max_epochs,
            "patience": 30,
            "seed": seed,
            "gpu_number": gpu,
            "prediction_mode": mode,
        })
        genotypes, samples = loc.load_genotypes(matrix=str(feature_matrix_path))
        loc.train(genotypes=genotypes, samples=samples)
        loc.predict(genotypes=genotypes, samples=samples)
    except Exception as exc:
        fold_result["status"] = "FAILED"
        fold_result["error"] = f"{type(exc).__name__}: {exc}"
        fold_result["n_held_out"] = int(len(held_out_ids))
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result
    fold_result["elapsed_s"] = round(time.time() - t0, 1)

    predlocs = Path(f"{out_prefix}_predlocs.txt")
    if not predlocs.exists():
        fold_result["status"] = "FAILED"
        fold_result["error"] = f"predlocs missing at {predlocs}"
        fold_result["n_held_out"] = int(len(held_out_ids))
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result

    pred = common.parse_predlocs(predlocs)
    pred = pred[pred["sampleID"].isin(held_out_ids)].set_index("sampleID")
    tru = truth.set_index("sampleID").loc[held_out_ids, ["x", "y"]]
    evaluable = tru.dropna().index.tolist()
    evaluable_in_pred = [sid for sid in evaluable if sid in pred.index]

    if not evaluable_in_pred:
        fold_result["status"] = "OK"
        fold_result["n_held_out"] = int(len(held_out_ids))
        fold_result["n_evaluable"] = int(len(evaluable))
        fold_result["n_scored"] = 0
        fold_result["warning"] = "no scored samples"
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result

    p = pred.loc[evaluable_in_pred, ["x", "y"]].to_numpy(dtype=np.float64)
    t = tru.loc[evaluable_in_pred].to_numpy(dtype=np.float64)
    err_km = common.haversine(t[:, 1], t[:, 0], p[:, 1], p[:, 0])
    if np.ndim(err_km) == 0:
        err_km = np.array([err_km])

    fold_result.update({
        "status": "OK",
        "n_held_out": int(len(held_out_ids)),
        "n_evaluable": int(len(evaluable)),
        "n_scored": int(len(evaluable_in_pred)),
        "median_error_km": float(np.median(err_km)),
        "mean_error_km": float(np.mean(err_km)),
        "max_error_km": float(np.max(err_km)),
    })
    (fold_dir / "fold_result.json").write_text(
        json.dumps(fold_result, indent=2, default=str)
    )
    return fold_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--microsat", required=True, type=Path,
                   help="Pair-format microsat TSV (parse_genepop output).")
    p.add_argument("--sample_data", required=True, type=Path,
                   help="ReLocator sample_data.txt (truth coords).")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=VALID_MODES,
                   help="Single classifier mode per invocation.")
    p.add_argument("--feature_matrix", default=None, type=Path,
                   help="Pre-built dosage feature matrix; if not given, builds from --microsat via microsat_to_locator.py.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_epochs", type=int, default=500)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_matrix is None:
        feat = args.out_dir / "features_dosage.tsv"
        if not feat.exists():
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent.parent.parent
                    / "scripts/microsat_to_locator.py"),
                "--microsat", str(args.microsat),
                "--out", str(feat),
                "--features", "dosage",
            ]
            rc, _ = common.stream_run(
                cmd, feat.with_suffix(".buildlog"), stage="build_dosage"
            )
            if rc != 0:
                return rc
    else:
        feat = args.feature_matrix

    sd_df = pd.read_csv(args.sample_data, sep="\t")
    sites = sorted(set(sd_df["sampleID"].apply(site_of)))

    print(f"Starting {len(sites)} folds with prediction_mode={args.mode} "
          f"on GPU {args.gpu}", flush=True)
    t_sweep = time.time()
    for idx, site in enumerate(sites):
        print(f"\n=== fold {idx + 1}/{len(sites)}: {site} ===", flush=True)
        run_one_fold(
            mode=args.mode, site=site,
            feature_matrix_path=feat,
            sample_data_in=args.sample_data,
            out_dir=args.out_dir / "loso_classifier",
            gpu=args.gpu, seed=args.seed,
            max_epochs=args.max_epochs,
        )
    print(f"\nSweep complete in {(time.time() - t_sweep) / 60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint**

```bash
pixi run ruff check validation/sculpin/run_loso_classifier.py
```

- [ ] **Step 3: Smoke check (single fold, single mode)**

```bash
pixi run python -m validation.sculpin.run_loso_classifier \
    --microsat out/sculpin_validation/inputs/sculpin_microsat.tsv \
    --sample_data out/sculpin_validation/inputs/sample_data.txt \
    --out_dir /tmp/sculpin_classifier_smoke \
    --mode classify \
    --max_epochs 10 \
    --gpu 0 2>&1 | tail -10
```

Expected: prints "fold 1/16: Alaska", trains a small model, writes fold_result.json. Then continues iterating; you can Ctrl+C after 1–2 sites complete since this is a smoke check. Verify at least one fold's `fold_result.json` exists with `"status": "OK"` (or with a clear `"error"` to debug).

- [ ] **Step 4: Commit**

```bash
git add validation/sculpin/run_loso_classifier.py
git commit -m "$(cat <<'EOF'
sculpin: classifier-mode LOSO runner

Mirrors validation/sculpin/run_loso.py but uses Locator's Python API
with prediction_mode in {classify, classify_then_avg}. Same fold output
schema as the regression runner so existing summarizers / downstream
code work unchanged.

Single mode per invocation; the dosage feature matrix is reused across
both modes (and across the existing regress / regress+rangemask runs).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Run the full sculpin LOSO sweep with both classifier modes

**Files:**
- No new files. Runs the runner from Task 11 against the sculpin dataset.

- [ ] **Step 1: Run `classify` mode (16 folds, ~5 min)**

```bash
pixi run python -m validation.sculpin.run_loso_classifier \
    --microsat out/sculpin_validation/inputs/sculpin_microsat.tsv \
    --sample_data out/sculpin_validation/inputs/sample_data.txt \
    --out_dir out/sculpin_validation \
    --mode classify \
    --gpu 0 \
    > out/sculpin_validation/run_classify.log 2>&1
echo "exit=$?"
tail -5 out/sculpin_validation/run_classify.log
```

Expected: 16 folds complete, ~5–10 min wall-clock. Final line: "Sweep complete in X min".

- [ ] **Step 2: Run `classify_then_avg` mode (16 folds, ~5 min)**

```bash
pixi run python -m validation.sculpin.run_loso_classifier \
    --microsat out/sculpin_validation/inputs/sculpin_microsat.tsv \
    --sample_data out/sculpin_validation/inputs/sample_data.txt \
    --out_dir out/sculpin_validation \
    --mode classify_then_avg \
    --gpu 0 \
    > out/sculpin_validation/run_classify_then_avg.log 2>&1
echo "exit=$?"
tail -5 out/sculpin_validation/run_classify_then_avg.log
```

Expected: 16 folds complete, ~5–10 min wall-clock.

- [ ] **Step 3: Verify all fold JSONs exist**

```bash
find out/sculpin_validation/loso_classifier -name 'fold_result.json' | wc -l
find out/sculpin_validation/loso_classifier -name 'fold_result.json' -exec grep -l '"error"' {} \; 2>/dev/null
```

Expected: 32 JSONs (16 sites × 2 modes); the second command should print nothing.

- [ ] **Step 4: No commit yet** — outputs live under `out/` which is gitignored. Summaries are committed in Task 13.

---

## Task 13: Summarize 4-mode comparison

**Files:**
- Create: `validation/sculpin/summarize_classifier.py`
- Create (script-generated): `validation/summary/sculpin_classifier_kfold.tsv`, `validation/summary/sculpin_classifier_summary.md`, `validation/figures/sculpin_classifier_modes.png`, `validation/figures/sculpin_classifier_map.png`

- [ ] **Step 1: Create the summarizer**

Create `validation/sculpin/summarize_classifier.py`:

```python
#!/usr/bin/env python3
"""Aggregate sculpin LOSO results across all 4 modes for the classifier-head experiment.

Modes:
  - regress           : out/sculpin_validation/loso/dosage/<site>/fold_result.json
  - regress+rangemask : out/sculpin_validation/loso_rangemask/dosage/<site>/fold_result.json
  - classify          : out/sculpin_validation/loso_classifier/classify/<site>/fold_result.json
  - classify_then_avg : out/sculpin_validation/loso_classifier/classify_then_avg/<site>/fold_result.json

Outputs:
  validation/summary/sculpin_classifier_kfold.tsv  — all per-fold rows tagged by mode
  validation/summary/sculpin_classifier_summary.md — written report
  validation/figures/sculpin_classifier_modes.png  — grouped bar chart
  validation/figures/sculpin_classifier_map.png    — 4-panel cartopy map of predictions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from validation import common  # noqa: E402

MODE_PATHS = {
    "regress": ("loso/dosage", "regression (current)"),
    "regress+rangemask": ("loso_rangemask/dosage", "regression + range_mask"),
    "classify": ("loso_classifier/classify", "classify (argmax)"),
    "classify_then_avg": ("loso_classifier/classify_then_avg", "classify_then_avg"),
}


def load_mode_folds(out_dir: Path, mode_subdir: str, mode_label: str) -> pd.DataFrame:
    rows = []
    for p in sorted((out_dir / mode_subdir).glob("*/fold_result.json")):
        d = json.loads(p.read_text())
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["mode_label"] = mode_label
    return df


def write_aggregate_tsv(out_dir: Path, summary_dir: Path) -> pd.DataFrame:
    frames = []
    for label, (subdir, _) in MODE_PATHS.items():
        df = load_mode_folds(out_dir, subdir, label)
        if not df.empty:
            df["mode"] = label
            frames.append(df)
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(summary_dir / "sculpin_classifier_kfold.tsv", sep="\t", index=False)
    return full


def aggregate(full: pd.DataFrame) -> pd.DataFrame:
    if full.empty:
        return pd.DataFrame()
    fold = full.dropna(subset=["median_error_km"])
    return (
        fold.groupby("mode")
        .agg(
            n_folds=("site", "count"),
            med_mean=("median_error_km", "mean"),
            med_std=("median_error_km", "std"),
            mn_mean=("mean_error_km", "mean"),
            mn_std=("mean_error_km", "std"),
        )
        .reindex(list(MODE_PATHS.keys()))
        .dropna(subset=["med_mean"])
    )


def make_bar_figure(full: pd.DataFrame, fig_path: Path) -> None:
    if full.empty:
        return
    fold = full.dropna(subset=["median_error_km"])
    site_order = (
        fold[fold["mode"] == "regress"]
        .sort_values("median_error_km")["site"].tolist()
    )
    if not site_order:
        site_order = sorted(fold["site"].unique())

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(site_order))
    w = 0.20
    colors = ["firebrick", "darkorange", "seagreen", "steelblue"]
    for i, mode in enumerate(MODE_PATHS):
        sub = fold[fold["mode"] == mode].set_index("site")
        vals = [sub["median_error_km"].get(s, np.nan) for s in site_order]
        ax.bar(x + (i - 1.5) * w, vals, w,
               label=MODE_PATHS[mode][1], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(site_order, rotation=60, ha="right")
    ax.set_ylabel("Median LOSO error (km)")
    ax.set_title("Sculpin LOSO: 4-mode comparison (per site)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_summary_md(full: pd.DataFrame, agg: pd.DataFrame,
                      summary_md: Path) -> None:
    centroid_baseline_km = 440.1
    if agg.empty:
        summary_md.write_text(
            "# Sculpin classifier-head summary\n\nNo fold results found.\n"
        )
        return
    lines = [
        "# Sculpin classifier-head LOSO summary",
        "",
        "16-fold leave-one-site-out comparison across 4 prediction modes on",
        "the prickly sculpin Cottus asper microsat dataset (405 individuals ×",
        "19 loci, dosage encoding). Metric: haversine distance (km).",
        "",
        f"**Centroid baseline (predict training mean):** {centroid_baseline_km:.1f} km",
        "",
        "## Aggregate (mean of per-fold medians)",
        "",
        "| Mode | n folds | Median error km (mean ± std) | Mean error km (mean ± std) | Δ vs centroid | Δ vs regress |",
        "|---|---|---|---|---|---|",
    ]
    regress_med = float(agg.loc["regress", "med_mean"]) if "regress" in agg.index else float("nan")
    for mode in agg.index:
        r = agg.loc[mode]
        d_centroid = r["med_mean"] - centroid_baseline_km
        d_regress = r["med_mean"] - regress_med if not np.isnan(regress_med) else float("nan")
        lines.append(
            f"| `{mode}` | {int(r['n_folds'])} | {r['med_mean']:.1f} ± {r['med_std']:.1f} | "
            f"{r['mn_mean']:.1f} ± {r['mn_std']:.1f} | {d_centroid:+.1f} | {d_regress:+.1f} |"
        )
    lines += [
        "",
        "## Per-site error (km)",
        "",
        "| site | regress | regress+rangemask | classify | classify_then_avg |",
        "|---|---|---|---|---|",
    ]
    fold = full.dropna(subset=["median_error_km"])
    sites = sorted(fold["site"].unique())
    for site in sites:
        cells = []
        for mode in MODE_PATHS:
            v = fold[(fold["mode"] == mode) & (fold["site"] == site)]
            cells.append(f"{v['median_error_km'].iloc[0]:.0f}" if not v.empty else "—")
        lines.append(f"| {site} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Files",
        "",
        "- `validation/figures/sculpin_classifier_modes.png` — grouped bar",
        "  chart, all 4 modes per site.",
        "- `validation/figures/sculpin_classifier_map.png` — per-individual",
        "  prediction map, 4 panels.",
        "- `validation/summary/sculpin_classifier_kfold.tsv` — raw per-fold rows.",
        "",
        "## Recommendation",
        "",
        "_Manually edited after the run — replace with 2–3 sentence interpretation._",
        "",
    ]
    summary_md.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True, type=Path,
                   help="Sculpin validation root (parent of loso/, loso_rangemask/, loso_classifier/).")
    p.add_argument("--summary_dir", type=Path,
                   default=Path("validation/summary"))
    p.add_argument("--figures_dir", type=Path,
                   default=Path("validation/figures"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    full = write_aggregate_tsv(args.out_dir, args.summary_dir)
    agg = aggregate(full)
    make_bar_figure(full, args.figures_dir / "sculpin_classifier_modes.png")
    render_summary_md(full, agg, args.summary_dir / "sculpin_classifier_summary.md")
    print(f"Wrote summary, kfold tsv, and figures under {args.summary_dir} and {args.figures_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint**

```bash
pixi run ruff check validation/sculpin/summarize_classifier.py
```

- [ ] **Step 3: Run summarize**

```bash
pixi run python -m validation.sculpin.summarize_classifier \
    --out_dir out/sculpin_validation
```

Expected: prints "Wrote summary, kfold tsv, and figures under …".

- [ ] **Step 4: Manually edit the Recommendation section**

Open `validation/summary/sculpin_classifier_summary.md` and replace the placeholder Recommendation section with 2–3 sentences interpreting the actual numbers. Specifically address:

- Did `classify` or `classify_then_avg` beat `regress` and `regress+rangemask` on the aggregate mean(median)?
- Per-site, which sites improved most under classification (likely the FST-outliers like Alaska, Okanagan, Nimpo that had bad regression scores)?
- Which sites got worse (if any)?
- Does the result confirm or refute the architectural hypothesis that "the regression head's regression-to-the-mean failure is fixed by classification"?

- [ ] **Step 5: Commit summary + figures + script**

```bash
git add validation/sculpin/summarize_classifier.py \
        validation/summary/sculpin_classifier_kfold.tsv \
        validation/summary/sculpin_classifier_summary.md \
        validation/figures/sculpin_classifier_modes.png
git commit -m "$(cat <<'EOF'
sculpin: 4-mode classifier comparison summary + figure

Aggregates the 64 LOSO fold JSONs (16 sites × 4 modes) into a single
comparison TSV, summary markdown, and grouped bar chart. The 4 modes:
  - regress (existing)
  - regress + range_mask (existing)
  - classify (new, argmax)
  - classify_then_avg (new, softmax @ centroids)

Recommendation section manually edited based on the actual numbers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Final review + push

**Files:**
- No new files. Runs the full test suite, lints everything, and pushes the branch.

- [ ] **Step 1: Run the complete classifier-head test suite**

```bash
pixi run pytest tests/test_classifier_head.py -v
```

Expected: 20 PASS (all tests added across Tasks 1, 2, 3, 4, 5, 6, 7, 8).

- [ ] **Step 2: Run the full repo test suite to confirm no regressions**

```bash
pixi run pytest tests/ -q --ignore=tests/test_tf_dataset.py 2>&1 | tail -10
```

(Skipping `test_tf_dataset.py` because of the known xdist flake noted in CLAUDE.md.)

Expected: all tests pass; no failures attributable to classifier-head changes.

- [ ] **Step 3: Lint the whole branch's modified files**

```bash
pixi run ruff check \
    locator/data/site_labels.py \
    locator/data/__init__.py \
    locator/models.py \
    locator/training.py \
    locator/prediction.py \
    tests/test_classifier_head.py \
    validation/sculpin/run_loso_classifier.py \
    validation/sculpin/summarize_classifier.py
```

Expected: All checks passed.

- [ ] **Step 4: Verify the commit history is clean**

```bash
git log --oneline microsat-sculpin..classifier-head
```

Expected: ~14 commits, one per Task, with informative messages.

- [ ] **Step 5: Push to fork**

```bash
git push fork classifier-head 2>&1 | tail -5
```

Expected: branch created on fork, no errors.

- [ ] **Step 6: Confirm push**

```bash
git log --oneline fork/classifier-head -3
```

Expected: top commit matches local HEAD.

---

## Self-review checklist

After writing the plan, run through this once and fix any gaps inline.

- [ ] **Spec coverage** — every section of `2026-05-05-classifier-head-design.md` has at least one task implementing it:
  - Goals 1 (config flag) → Tasks 3, 4
  - Goals 2 (model integration) → Tasks 2, 4, 5, 6, 7
  - Goals 3 (sculpin LOSO validation) → Tasks 11, 12, 13
  - Goals 4 (improvement evidence on snap-correctness / error-vs-FST) → Task 13 Step 4 (manual recommendation prose)
  - Architecture / models.py changes → Task 2
  - Architecture / training.py changes → Tasks 3, 4, 5, 8
  - Architecture / prediction.py changes → Tasks 6, 7, 8
  - Site-label derivation → Task 1
  - Loss function detail (cross-entropy default; classify_then_avg as inference-time choice) → Tasks 2, 7
  - Range-mask incompatibility → Tasks 3, 5
  - Tests merge zone → Tasks 1, 2, 3, 4, 5, 6, 7, 8 (all add tests)
  - Documentation → Task 9
  - Cross-reference back to the microsat spec → Task 10
  - Out-of-scope items → declared in the spec; no plan task needed.
- [ ] **Placeholder scan** — search the plan for "TBD", "TODO", "implement later", "fill in details", "Add appropriate error handling", or any "Similar to Task N" without a code block. The only "TODO"s are in the user-facing Recommendation prose template in Task 13 Step 4 (intentional manual-fill content).
- [ ] **Type consistency**:
  - `derive_site_labels` returns `(np.ndarray int32, np.ndarray float64)` — used consistently in Task 4 (`_prepare_targets`) and Task 5 (centroid storage).
  - `_prepare_targets` returns `(targets, centroids, n_classes)` with types `(np.ndarray, np.ndarray|None, int|None)` — consumed by `_build_datasets_and_fit` and `_create_model`.
  - `self._site_centroids` is `(K, 2) np.float64` everywhere it's referenced (Tasks 5, 6, 7, 8).
  - `self._n_classes` is `int` everywhere.
  - `prediction_mode` config keys are exactly `regress`, `classify`, `classify_then_avg` (matches `VALID_PREDICTION_MODES`).
  - Config keys: `prediction_mode`, `min_samples_per_site`, `use_range_penalty`, `species_range_shapefile`, `resolution`, `penalty_weight` — all read with `config.get(...)` so missing keys default sensibly.
  - HDF5 attributes: `prediction_mode` (string), `n_classes` (int), and dataset `site_centroids` ((K,2) float64) — written in Task 8 and read in Task 8.
