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
    # Default loss is euclidean_distance_loss (the function object itself)
    from locator.models import euclidean_distance_loss
    assert model.loss is euclidean_distance_loss


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


def test_create_network_classifier_invalid_n_classes_errors():
    """n_classes < 2 raises ValueError with a clear message."""
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        create_network(input_shape=10, n_classes=0)
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        create_network(input_shape=10, n_classes=1)


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


# ---------------------------------------------------------------------------
# Task 4: training integration
# ---------------------------------------------------------------------------


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
        "optimize_tf_parallelism": False,
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
        "optimize_tf_parallelism": False,
    })
    genotypes, samples = loc.load_genotypes(matrix=str(matrix_path))
    with pytest.raises(ValueError, match="range_mask|use_range_penalty"):
        loc.train(genotypes=genotypes, samples=samples)


# ---------------------------------------------------------------------------
# Task 6: predict() in classify mode
# ---------------------------------------------------------------------------

def test_predict_classify_mode_returns_centroids_for_held_out_samples(tmp_path):
    """End-to-end: train + predict in classify mode returns one of the K
    training-site centroids for each prediction sample."""
    import pandas as pd

    from locator import Locator
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
        "optimize_tf_parallelism": False,
    })
    genotypes, samples = loc.load_genotypes(matrix=str(matrix_path))
    loc.train(genotypes=genotypes, samples=samples)
    pred_df = loc.predict(genotypes=genotypes, samples=samples, return_df=True)
    assert "x" in pred_df.columns and "y" in pred_df.columns
    # Each predicted (x, y) must match one of the 3 training centroids exactly.
    centroids = {(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)}
    for _, row in pred_df.iterrows():
        assert (round(row["x"], 4), round(row["y"], 4)) in centroids


# ---------------------------------------------------------------------------
# Task 7: predict() in classify_then_avg mode
# ---------------------------------------------------------------------------

def test_predict_classify_then_avg_inside_convex_hull(tmp_path):
    """End-to-end: classify_then_avg predictions land in the convex hull of
    training centroids."""
    import pandas as pd

    from locator import Locator
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
        "optimize_tf_parallelism": False,
    })
    genotypes, samples = loc.load_genotypes(matrix=str(matrix_path))
    loc.train(genotypes=genotypes, samples=samples)
    pred_df = loc.predict(genotypes=genotypes, samples=samples, return_df=True)
    # Centroids in raw lon/lat: 0, 10, 20 on each axis. Predictions must
    # fall inside the convex hull (with small slack for float noise).
    for _, row in pred_df.iterrows():
        assert -1.0 <= row["x"] <= 21.0
        assert -1.0 <= row["y"] <= 21.0


def test_predict_classify_then_avg_math_matches_softmax_dot_centroids(tmp_path):
    """Sanity: the prediction must equal softmax @ centroids (pre-denorm)
    when we manually compute it."""
    import pandas as pd

    from locator import Locator
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
        "optimize_tf_parallelism": False,
    })
    genotypes, samples = loc.load_genotypes(matrix=str(matrix_path))
    loc.train(genotypes=genotypes, samples=samples)

    # Manually run the network forward pass on prediction-target row.
    _, locs = loc.sort_samples(samples)
    pred_idx = np.where(np.isnan(locs[:, 0]))[0]
    raw = loc.model.predict(loc.filtered_genotypes.T[pred_idx], verbose=0)
    expected_pre_denorm = raw @ loc._site_centroids
    expected_x = expected_pre_denorm[:, 0] * loc.sdlong + loc.meanlong
    expected_y = expected_pre_denorm[:, 1] * loc.sdlat + loc.meanlat

    pred_df = loc.predict(genotypes=genotypes, samples=samples, return_df=True)
    np.testing.assert_allclose(pred_df["x"].values, expected_x, atol=1e-3)
    np.testing.assert_allclose(pred_df["y"].values, expected_y, atol=1e-3)


# ---------------------------------------------------------------------------
# Task 8: HDF5 round-trip
# ---------------------------------------------------------------------------

def test_classifier_centroids_round_trip_hdf5(tmp_path):
    """Train → save → load_model → predict produces same centroids matrix."""
    import pandas as pd

    from locator import Locator

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
        "optimize_tf_parallelism": False,
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
        "optimize_tf_parallelism": False,
    })
    loc2.load_model(f"{out_prefix}.weights.h5")
    np.testing.assert_array_equal(loc2._site_centroids, saved_centroids)
    assert loc2._n_classes == saved_centroids.shape[0]
