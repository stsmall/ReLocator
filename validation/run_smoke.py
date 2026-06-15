#!/usr/bin/env python3
"""Path A: end-to-end smoke test.

Runs gl_to_locator.py once in dosage mode, validates output, then trains
ReLocator on all 54 samples (no held-out) for a small number of epochs.
PASS if the pipeline completes cleanly and the network produces a
*_predlocs.txt with 54 rows.

Subprocess output is streamed to both this script's stdout and a per-stage
log file so progress is visible while the job runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from validation.common import stream_run


def _result(out_dir: Path, status: str, **fields: object) -> int:
    payload = {"status": status, **fields}
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
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
    gl_log = args.out_dir / "gl_to_locator.log"

    # --- Stage 1: convert beagle -> dosage ---
    rc, tail = stream_run(
        [
            "pixi", "run", "python", "scripts/gl_to_locator.py",
            "--beagle", str(args.beagle),
            "--bam_list", str(bam_list),
            "--out", str(gl_dosage),
            "--max_missing_frac", "0.50",
            "--gl_mode", "dosage",
        ],
        log_path=gl_log,
        stage="gl_to_locator",
    )
    if rc != 0:
        return _result(args.out_dir, "FAIL", stage="gl_to_locator", stderr_tail=tail)

    df = pd.read_csv(gl_dosage, sep="\t")
    n_sites_first = df.shape[1] - 1
    print(f"[smoke] dosage TSV after first pass: {df.shape[0]} samples x {n_sites_first} sites", flush=True)

    # If too few sites survived, retry once at a permissive threshold.
    if n_sites_first < 1000:
        print(
            f"[smoke] only {n_sites_first} sites at max_missing_frac=0.50; "
            f"retrying at 0.95",
            flush=True,
        )
        rc, tail = stream_run(
            [
                "pixi", "run", "python", "scripts/gl_to_locator.py",
                "--beagle", str(args.beagle),
                "--bam_list", str(bam_list),
                "--out", str(gl_dosage),
                "--max_missing_frac", "0.95",
                "--gl_mode", "dosage",
            ],
            log_path=args.out_dir / "gl_to_locator_retry.log",
            stage="gl_to_locator_retry",
        )
        if rc != 0:
            return _result(args.out_dir, "FAIL", stage="gl_to_locator_retry", stderr_tail=tail)
        df = pd.read_csv(gl_dosage, sep="\t")

    n_sites = df.shape[1] - 1
    n_samples = df.shape[0]
    if n_samples != 54:
        return _result(args.out_dir, "FAIL", stage="dim_check", n_samples=n_samples)
    if n_sites < 1000:
        return _result(args.out_dir, "FAIL", stage="dim_check", n_sites=n_sites)

    # --- Stage 2: cross-check sample IDs ---
    sd = pd.read_csv(sample_data, sep="\t")
    if df["sampleID"].tolist() != sd["sampleID"].tolist():
        return _result(
            args.out_dir,
            "FAIL",
            stage="sample_id_check",
            reason="sampleID order in dosage file does not match sample_data.txt",
        )
    print("[smoke] sample-ID order matches sample_data.txt", flush=True)

    # --- Stage 3: train ReLocator ---
    run_dir = args.out_dir / "locator_run"
    run_dir.mkdir(exist_ok=True)
    out_prefix = run_dir / "smoke"
    rc, tail = stream_run(
        [
            "pixi", "run", "locator",
            "--matrix", str(gl_dosage),
            "--sample_data", str(sample_data),
            "--out", str(out_prefix),
            "--max_epochs", str(args.max_epochs),
            "--patience", "10",
            "--seed", "42",
            "--gpu_number", str(args.gpu_number),
        ],
        log_path=run_dir / "locator.log",
        stage="locator",
    )
    if rc != 0:
        return _result(args.out_dir, "FAIL", stage="locator", stderr_tail=tail)

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
