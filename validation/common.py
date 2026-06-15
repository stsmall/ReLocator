"""Shared utilities for path D validation scripts."""

from __future__ import annotations

import subprocess
import time
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


def strip_quoted(series: pd.Series) -> pd.Series:
    """Remove surrounding double-quotes from a string-ish pandas Series.

    R-style CSVs (used by some `data/test_sample_data.txt` and predlocs files
    written through R) wrap string values in literal double-quotes. Strip
    them before merging on string columns.
    """
    return series.astype(str).str.strip('"')


def stream_run(cmd: list[str], log_path: Path, stage: str) -> tuple[int, str]:
    """Run cmd, streaming combined stdout+stderr to screen and log_path.

    Returns ``(returncode, log_tail)`` where log_tail is the last 2000 chars
    of the captured output (useful for compact error reporting). Callers
    that don't need the tail can ignore it.
    """
    print(f"[{stage}] START $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with log_path.open("w") as log, subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = proc.wait()
    elapsed = time.time() - t0
    print(f"[{stage}] END elapsed={elapsed:.1f}s rc={rc}", flush=True)
    tail = log_path.read_text()[-2000:] if log_path.exists() else ""
    return rc, tail
