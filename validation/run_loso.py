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
import time
from pathlib import Path

import numpy as np
import pandas as pd

from validation import common
from validation.coastline import (
    BARNACLE_COAST,
    build_coast_index,
    project_to_coast,
)
from validation.common import stream_run

_COAST_CUM, _COAST_PTS = build_coast_index()
_SITE_TO_INDEX = {site: i for i, (site, _, _) in enumerate(BARNACLE_COAST)}


def blank_holdout_locations(
    sample_data: pd.DataFrame, sample_table: pd.DataFrame, target_site: str
) -> pd.DataFrame:
    """Return a copy of sample_data with x,y set to 'NA' for samples at target_site."""
    held = sample_table.loc[sample_table["site"] == target_site, "sample_name"].tolist()
    out = sample_data.copy()
    out["x"] = out["x"].astype(object)
    out["y"] = out["y"].astype(object)
    mask = out["sampleID"].isin(held)
    out.loc[mask, "x"] = "NA"
    out.loc[mask, "y"] = "NA"
    return out


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
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        log_path.write_text(f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}")
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        log_path.write_text(
            f"TIMEOUT after {timeout_s}s\n"
            f"STDOUT:\n{e.stdout or ''}\n\nSTDERR:\n{e.stderr or ''}"
        )
        rc = -1
    elapsed = time.time() - t0

    fold_result: dict = {
        "site": site_code,
        "true_lat": site_lat,
        "true_lon": site_lon,
        "gl_mode": gl_mode,
        "gpu": gpu,
        "elapsed_s": round(elapsed, 1),
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
            true_idx = _SITE_TO_INDEX.get(site_code)
            if true_idx is not None:
                coast_results = [
                    project_to_coast(lat, lon, _COAST_CUM, _COAST_PTS)
                    for lat, lon in zip(held_pred["y"].values, held_pred["x"].values, strict=True)
                ]
                coast_pos = np.array([cp for cp, _ in coast_results], dtype=np.float64)
                offshore = np.array([off for _, off in coast_results], dtype=np.float64)
                along_coast_err = np.abs(coast_pos - _COAST_CUM[true_idx])
                held_pred["coast_pos_km"] = coast_pos
                held_pred["offshore_km"] = offshore
                held_pred["along_coast_err_km"] = along_coast_err
                fold_result["mean_along_coast_err_km"] = float(along_coast_err.mean())
                fold_result["median_along_coast_err_km"] = float(np.median(along_coast_err))
                fold_result["mean_offshore_km"] = float(offshore.mean())
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
        rc, _ = stream_run(
            [
                "pixi", "run", "python", "scripts/gl_to_locator.py",
                "--beagle", str(args.beagle),
                "--bam_list", str(args.inputs_dir / "bam.filelist"),
                "--out", str(feature_matrix),
                "--max_missing_frac", str(args.max_missing_frac),
                "--gl_mode", args.gl_mode,
            ],
            log_path=args.out_dir / "gl_to_locator.log",
            stage="gl_to_locator",
        )
        if rc != 0:
            print(f"run_loso: gl_to_locator failed with rc={rc}", flush=True)
            return 1

    sample_data_full = pd.read_csv(args.inputs_dir / "sample_data.txt", sep="\t")
    sample_table = pd.read_csv(args.sample_table, sep="\t")
    samples_locations = pd.read_csv(args.samples_locations, sep="\t")

    site_rows = list(samples_locations.itertuples(index=False))

    print(
        f"run_loso: starting {len(site_rows)} folds for gl_mode={args.gl_mode} "
        f"with n_parallel={args.n_parallel} on {args.n_gpus} GPUs",
        flush=True,
    )

    t_sweep = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_parallel) as pool:
        futures = {}
        for idx, row in enumerate(site_rows):
            fut = pool.submit(
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
            futures[fut] = row.SiteCode
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            site = res.get("site", futures[fut])
            elapsed = res.get("elapsed_s", "?")
            extra = ""
            if res["status"] == "OK":
                extra = (
                    f" mean_err={res['mean_error_km']:.1f}km "
                    f"median_err={res['median_error_km']:.1f}km"
                )
            else:
                extra = f" reason={res.get('reason', '?')}"
            print(
                f"run_loso: fold {site} -> {res['status']} ({elapsed}s){extra}",
                flush=True,
            )

    sweep_elapsed = time.time() - t_sweep
    print(f"run_loso: sweep done in {sweep_elapsed:.1f}s ({sweep_elapsed/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
