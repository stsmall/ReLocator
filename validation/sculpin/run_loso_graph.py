#!/usr/bin/env python3
"""Train + predict each sculpin LOSO fold with graph topology smoothing.

Trains classify_then_avg models from scratch (deterministic with seed=42,
matching the classifier-head sweep) and applies heat-kernel smoothing
at predict time. We retrain rather than reusing classifier-head saved
weights because Locator.load_model() + train(setup_only=True) does not
round-trip the class-order mapping cleanly — the reloaded architecture
ends up with weights/centroid mismatch and garbage predictions.

For each of the 16 sites:
  1. Hold out the site by setting its samples' coords to NaN in
     sample_data.txt (read from the classifier-head fold dir).
  2. Train a fresh Locator with prediction_mode='classify_then_avg' AND
     graph_topology=<GeoJSON> AND graph_kernel_resolution=t.
  3. predict() applies the heat-kernel smoothing.
  4. Write fold_result.json under
     out/sculpin_validation/loso_graph/<SITE>/.

Single mode per invocation. Default graph_kernel_resolution=1.0; sweep
via --t to test alternatives.
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

DEFAULT_T = 1.0


def site_of(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def run_one_fold(
    *, site: str,
    classifier_fold_dir: Path,
    sample_data_in: Path,
    feature_matrix_path: Path,
    topology_path: Path,
    out_dir: Path,
    gpu: int,
    t: float,
) -> dict:
    fold_dir = out_dir / site
    fold_dir.mkdir(parents=True, exist_ok=True)

    cls_sd_path = classifier_fold_dir / "sample_data.txt"
    if not cls_sd_path.exists():
        return {
            "mode": "classify_then_avg_graph", "site": site,
            "status": "FAILED",
            "error": f"classifier-head fold sample_data missing at {cls_sd_path}",
        }

    sd_path = fold_dir / "sample_data.txt"
    sd_path.write_text(cls_sd_path.read_text())

    out_prefix = fold_dir / "locator"
    fold_result: dict = {
        "mode": "classify_then_avg_graph", "site": site,
        "graph_kernel_resolution": float(t),
    }

    t_start = time.time()
    try:
        from locator import Locator

        loc = Locator(config={
            "out": str(out_prefix),
            "matrix": str(feature_matrix_path),
            "sample_data": str(sd_path),
            "seed": 42,
            "gpu_number": gpu,
            "prediction_mode": "classify_then_avg",
        })
        genotypes, samples = loc.load_genotypes(matrix=str(feature_matrix_path))
        loc.train(genotypes=genotypes, samples=samples)

        # First predict pass: no graph (paired baseline from this exact model).
        loc.predict(genotypes=genotypes, samples=samples)
        baseline_predlocs = fold_dir / "locator_predlocs_paired_baseline.txt"
        Path(f"{out_prefix}_predlocs.txt").rename(baseline_predlocs)

        # Second predict pass: enable graph topology + heat-kernel smoothing
        # on the SAME trained model. Any difference in error is solely from
        # smoothing — not training noise.
        loc.config["graph_topology"] = str(topology_path)
        loc.config["graph_kernel_resolution"] = float(t)
        if hasattr(loc, "_heat_kernel"):
            del loc._heat_kernel  # force rebuild against the new topology
        loc.predict(genotypes=genotypes, samples=samples)
    except Exception as exc:
        fold_result["status"] = "FAILED"
        fold_result["error"] = f"{type(exc).__name__}: {exc}"
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result
    fold_result["elapsed_s"] = round(time.time() - t_start, 1)

    predlocs = Path(f"{out_prefix}_predlocs.txt")
    if not predlocs.exists() or not baseline_predlocs.exists():
        fold_result["status"] = "FAILED"
        fold_result["error"] = "missing predlocs"
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result

    sd_df = pd.read_csv(sd_path, sep="\t")
    held_out_ids = sd_df.loc[pd.isna(sd_df["x"]), "sampleID"].tolist()
    if not held_out_ids:
        fold_result["status"] = "FAILED"
        fold_result["error"] = f"no NA-coord samples in {sd_path}"
        (fold_dir / "fold_result.json").write_text(
            json.dumps(fold_result, indent=2, default=str)
        )
        return fold_result

    truth = pd.read_csv(sample_data_in, sep="\t")
    tru = truth.set_index("sampleID").loc[held_out_ids, ["x", "y"]]
    evaluable = tru.dropna().index.tolist()

    def _score(predlocs_path: Path) -> dict:
        pred = common.parse_predlocs(predlocs_path)
        pred = pred[pred["sampleID"].isin(held_out_ids)].set_index("sampleID")
        evaluable_in_pred = [sid for sid in evaluable if sid in pred.index]
        if not evaluable_in_pred:
            return {"n_scored": 0}
        p = pred.loc[evaluable_in_pred, ["x", "y"]].to_numpy(dtype=np.float64)
        t_arr = tru.loc[evaluable_in_pred].to_numpy(dtype=np.float64)
        err_km = common.haversine(t_arr[:, 1], t_arr[:, 0], p[:, 1], p[:, 0])
        if np.ndim(err_km) == 0:
            err_km = np.array([err_km])
        return {
            "n_scored": int(len(evaluable_in_pred)),
            "median_error_km": float(np.median(err_km)),
            "mean_error_km": float(np.mean(err_km)),
            "max_error_km": float(np.max(err_km)),
        }

    graph_metrics = _score(predlocs)
    paired_baseline = _score(baseline_predlocs)

    fold_result["status"] = "OK"
    fold_result["n_held_out"] = int(len(held_out_ids))
    fold_result["n_evaluable"] = int(len(evaluable))
    fold_result.update(graph_metrics)
    fold_result["paired_baseline"] = paired_baseline

    (fold_dir / "fold_result.json").write_text(
        json.dumps(fold_result, indent=2, default=str)
    )
    return fold_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--classifier_fold_root", required=True, type=Path,
                   help="Root containing classify_then_avg/<SITE>/sample_data.txt "
                        "(reused for the held-out site definition per fold); "
                        "typically out/sculpin_validation/loso_classifier/classify_then_avg")
    p.add_argument("--sample_data", required=True, type=Path,
                   help="Original sample_data.txt with truth coords.")
    p.add_argument("--feature_matrix", required=True, type=Path,
                   help="Pre-built dosage feature matrix from earlier sweeps.")
    p.add_argument("--topology", required=True, type=Path,
                   help="GeoJSON topology (sculpin_topology.geojson).")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--t", type=float, default=DEFAULT_T,
                   help="graph_kernel_resolution / heat-kernel diffusion time.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sd_df = pd.read_csv(args.sample_data, sep="\t")
    sites = sorted(set(sd_df["sampleID"].apply(site_of)))

    print(f"Re-predicting {len(sites)} folds with graph topology, "
          f"t={args.t} on GPU {args.gpu}", flush=True)
    t_sweep = time.time()
    for idx, site in enumerate(sites):
        print(f"\n=== fold {idx + 1}/{len(sites)}: {site} ===", flush=True)
        run_one_fold(
            site=site,
            classifier_fold_dir=args.classifier_fold_root / site,
            sample_data_in=args.sample_data,
            feature_matrix_path=args.feature_matrix,
            topology_path=args.topology,
            out_dir=args.out_dir / "loso_graph",
            gpu=args.gpu,
            t=args.t,
        )
    print(f"\nSweep complete in {(time.time() - t_sweep) / 60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
