#!/usr/bin/env python3
"""Population-genetics sanity check for the sculpin LOSO results.

Three classical analyses of the same genotype matrix ReLocator was trained
on, plus their relationship to ReLocator's per-site prediction error:

1. **PCA** of the dosage matrix, points colored by sampling site.
2. **Pairwise Wright's FST** between the 16 sites (heatmap).
3. **Mantel correlation** between the FST matrix and the haversine
   geographic-distance matrix between site centroids.
4. **LOSO error vs genetic distinctiveness**: for each site, mean FST to
   all other sites is a one-number proxy for "how genetically distinct."
   Plot ReLocator's LOSO median error against that — sites that should
   be easy to locate from genetics alone (high FST to neighbors) should
   have lower error.

Outputs:
  --out_pca_png       PCA scatter (PC1 vs PC2, colored by site)
  --out_fst_png       FST heatmap + Mantel scatter
  --out_err_png       LOSO error vs mean-FST scatter
  --out_summary_md    Markdown report with the numbers
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
from scipy.stats import pearsonr  # noqa: E402

from validation import common  # noqa: E402

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_features(features_tsv: Path) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load the dosage feature matrix and pull out site labels from sampleIDs."""
    df = pd.read_csv(features_tsv, sep="\t")
    sids = df["sampleID"]
    sites = sids.str.rsplit("_", n=1).str[0]
    X = df.drop(columns=["sampleID"]).to_numpy(dtype=np.float64)
    return df, sites, X


def load_pair_matrix(pair_tsv: Path) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Load the pair-format microsat TSV. Returns df, locus column names, and site labels."""
    df = pd.read_csv(pair_tsv, sep="\t", dtype=str)
    sids = df["sampleID"]
    sites = sids.str.rsplit("_", n=1).str[0]
    loci = [c for c in df.columns if c != "sampleID"]
    return df, loci, sites


def load_loso_errors(loso_dir: Path, mode: str = "dosage") -> pd.DataFrame:
    rows = []
    for p in sorted((loso_dir / mode).glob("*/fold_result.json")):
        rows.append(json.loads(p.read_text()))
    df = pd.DataFrame(rows)
    return df[["site", "n_evaluable", "median_error_km", "mean_error_km"]].set_index("site")


# ---------------------------------------------------------------------------
# Population-genetics primitives
# ---------------------------------------------------------------------------

def parse_pair(cell: str) -> tuple[int | None, int | None]:
    s = str(cell).strip().upper()
    if s in ("NA", "NAN", ".", "", "0,0", "0/0"):
        return (None, None)
    parts = s.split(",")
    if len(parts) != 2:
        return (None, None)
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return (None, None)


def site_allele_freqs(pair_df: pd.DataFrame, loci: list[str], sites: pd.Series) -> dict:
    """Return dict[site][locus] -> dict[allele -> frequency].

    Allele frequencies are normalized within each (site, locus); missing
    genotypes are skipped, not imputed.
    """
    out: dict[str, dict[str, dict[int, float]]] = {}
    for site in sites.unique():
        out[site] = {}
        sub = pair_df[sites.values == site]
        for locus in loci:
            counts: dict[int, int] = {}
            total = 0
            for cell in sub[locus]:
                a1, a2 = parse_pair(cell)
                if a1 is None:
                    continue
                counts[a1] = counts.get(a1, 0) + 1
                counts[a2] = counts.get(a2, 0) + 1
                total += 2
            if total == 0:
                out[site][locus] = {}
            else:
                out[site][locus] = {a: c / total for a, c in counts.items()}
    return out


def pairwise_fst(freqs: dict, loci: list[str]) -> pd.DataFrame:
    """Pairwise Wright (1951) FST between sites, averaged over loci.

    For each pair (i, j) and each locus l:
        HT_l = 1 - sum((p_i + p_j)/2)^2
        HS_l = (1 - sum p_i^2 + 1 - sum p_j^2) / 2
        FST_l = (HT_l - HS_l) / HT_l           (0 if HT_l == 0)
    Then FST(i, j) = mean over loci.
    """
    sites = list(freqs.keys())
    n = len(sites)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            site_i, site_j = sites[i], sites[j]
            fst_locus_vals = []
            for locus in loci:
                pi = freqs[site_i].get(locus, {})
                pj = freqs[site_j].get(locus, {})
                if not pi or not pj:
                    continue
                alleles = set(pi) | set(pj)
                pmean = {a: (pi.get(a, 0) + pj.get(a, 0)) / 2.0 for a in alleles}
                hs_i = 1 - sum(p * p for p in pi.values())
                hs_j = 1 - sum(p * p for p in pj.values())
                ht = 1 - sum(p * p for p in pmean.values())
                hs = (hs_i + hs_j) / 2.0
                if ht <= 0:
                    continue
                fst_locus_vals.append((ht - hs) / ht)
            mat[i, j] = mat[j, i] = (
                float(np.mean(fst_locus_vals)) if fst_locus_vals else np.nan
            )
    return pd.DataFrame(mat, index=sites, columns=sites)


# ---------------------------------------------------------------------------
# Mantel test (Pearson on flattened upper triangles)
# ---------------------------------------------------------------------------

def upper_triangle(m: np.ndarray) -> np.ndarray:
    n = m.shape[0]
    iu = np.triu_indices(n, k=1)
    return m[iu]


def mantel_correlation(
    matrix_a: pd.DataFrame, matrix_b: pd.DataFrame, n_permutations: int = 999
) -> tuple[float, float]:
    """Pearson r and permutation p-value on the off-diagonal upper triangle.

    Both matrices must be aligned on the same row/column index.
    """
    sites = list(matrix_a.index)
    a = matrix_a.loc[sites, sites].to_numpy(dtype=np.float64)
    b = matrix_b.loc[sites, sites].to_numpy(dtype=np.float64)
    a_vec = upper_triangle(a)
    b_vec = upper_triangle(b)
    finite = np.isfinite(a_vec) & np.isfinite(b_vec)
    a_vec = a_vec[finite]
    b_vec = b_vec[finite]
    if len(a_vec) < 3:
        return (float("nan"), float("nan"))
    r_obs, _ = pearsonr(a_vec, b_vec)

    rng = np.random.default_rng(42)
    n = len(sites)
    above = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        a_perm = a[perm][:, perm]
        a_perm_vec = upper_triangle(a_perm)[finite]
        r_perm, _ = pearsonr(a_perm_vec, b_vec)
        if abs(r_perm) >= abs(r_obs):
            above += 1
    p_val = (above + 1) / (n_permutations + 1)
    return (float(r_obs), float(p_val))


# ---------------------------------------------------------------------------
# Geographic-distance matrix
# ---------------------------------------------------------------------------

def site_distance_matrix(sites_tsv: Path, sites: list[str]) -> pd.DataFrame:
    sd = pd.read_csv(sites_tsv, sep="\t").set_index("site")
    n = len(sites)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = mat[j, i] = float(common.haversine(
                sd.loc[sites[i], "lat"], sd.loc[sites[i], "lon"],
                sd.loc[sites[j], "lat"], sd.loc[sites[j], "lon"],
            ))
    return pd.DataFrame(mat, index=sites, columns=sites)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pca(X: np.ndarray, sites: pd.Series, out_png: Path) -> tuple[float, float]:
    """Standardize then PCA. Returns (PC1 var explained, PC2 var explained)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Xz = Xc / sd_safe

    # Truncated SVD on the standardized matrix → PCs.
    U, s, Vt = np.linalg.svd(Xz, full_matrices=False)
    pcs = U * s  # (n_samples, n_components)
    var_explained = (s ** 2) / np.sum(s ** 2)

    fig, ax = plt.subplots(figsize=(9, 7))
    sites_unique = sorted(sites.unique())
    palette = plt.cm.tab20(np.linspace(0, 1, len(sites_unique)))
    color_map = dict(zip(sites_unique, palette, strict=False))
    for site in sites_unique:
        m = sites.values == site
        ax.scatter(pcs[m, 0], pcs[m, 1], color=color_map[site],
                   label=site, s=40, edgecolor="black", lw=0.3, alpha=0.85)
    ax.set_xlabel(f"PC1 ({var_explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1] * 100:.1f}%)")
    ax.set_title("Sculpin microsat dosage matrix — PCA, colored by site")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return float(var_explained[0]), float(var_explained[1])


def plot_fst_heatmap_and_mantel(
    fst_df: pd.DataFrame, dist_df: pd.DataFrame, mantel_r: float, mantel_p: float,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Heatmap of FST.
    sites_ordered = list(fst_df.index)
    im = axes[0].imshow(fst_df.values, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(sites_ordered)))
    axes[0].set_yticks(range(len(sites_ordered)))
    axes[0].set_xticklabels(sites_ordered, rotation=80, fontsize=8)
    axes[0].set_yticklabels(sites_ordered, fontsize=8)
    axes[0].set_title("Pairwise Wright FST (averaged over 19 loci)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="FST")

    # Mantel scatter: FST vs geographic distance.
    fst_vec = upper_triangle(fst_df.to_numpy(dtype=np.float64))
    dist_vec = upper_triangle(dist_df.to_numpy(dtype=np.float64))
    finite = np.isfinite(fst_vec) & np.isfinite(dist_vec)
    axes[1].scatter(dist_vec[finite], fst_vec[finite], s=18, alpha=0.7,
                    edgecolor="black", lw=0.3)
    axes[1].set_xlabel("Pairwise geographic distance (km, haversine)")
    axes[1].set_ylabel("Pairwise FST")
    axes[1].set_title(f"Isolation-by-distance: Mantel r = {mantel_r:.3f}, "
                      f"p ≈ {mantel_p:.3f} (n = {finite.sum()} pairs)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_err_vs_fst(
    err_per_site: pd.Series, mean_fst_per_site: pd.Series, out_png: Path,
) -> tuple[float, float]:
    df = pd.concat([err_per_site, mean_fst_per_site], axis=1).dropna()
    df.columns = ["err_km", "mean_fst"]
    if len(df) < 3:
        return (float("nan"), float("nan"))
    r, p = pearsonr(df["mean_fst"], df["err_km"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["mean_fst"], df["err_km"], s=70, alpha=0.85,
               edgecolor="black", lw=0.4)
    for site, row in df.iterrows():
        ax.annotate(site, (row["mean_fst"], row["err_km"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("mean pairwise FST to all other sites (genetic distinctiveness)")
    ax.set_ylabel("ReLocator LOSO median error (km, dosage mode)")
    ax.set_title(f"Per-site ReLocator error vs genetic distinctiveness\n"
                 f"Pearson r = {r:.3f}, p = {p:.3f} (n = {len(df)} sites)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return float(r), float(p)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features_tsv", required=True, type=Path,
                   help="Combined dosage+geometry+repeat_norm feature matrix "
                        "(used for PCA only; PCA is on the full feature space).")
    p.add_argument("--pair_tsv", required=True, type=Path,
                   help="Pair-format microsat TSV (used for FST computation).")
    p.add_argument("--sites_tsv", required=True, type=Path,
                   help="sites.tsv with site, lat, lon columns.")
    p.add_argument("--loso_dir", required=True, type=Path,
                   help="LOSO outputs directory containing <mode>/<site>/fold_result.json.")
    p.add_argument("--mode", default="dosage", help="Which LOSO mode to take errors from.")
    p.add_argument("--out_pca_png", required=True, type=Path)
    p.add_argument("--out_fst_png", required=True, type=Path)
    p.add_argument("--out_err_png", required=True, type=Path)
    p.add_argument("--out_summary_md", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("Loading inputs...", flush=True)
    _, sites_feat, X = load_features(args.features_tsv)
    pair_df, loci, sites_pair = load_pair_matrix(args.pair_tsv)
    err_df = load_loso_errors(args.loso_dir, args.mode)

    print(f"  features: {X.shape}", flush=True)
    print(f"  pair_df: {pair_df.shape} ({len(loci)} loci)", flush=True)
    print(f"  loso errors: {len(err_df)} sites", flush=True)

    # --- 1. PCA ---
    print("Running PCA...", flush=True)
    args.out_pca_png.parent.mkdir(parents=True, exist_ok=True)
    pc1_var, pc2_var = plot_pca(X, sites_feat, args.out_pca_png)
    print(f"  wrote {args.out_pca_png} (PC1={pc1_var * 100:.1f}%, PC2={pc2_var * 100:.1f}%)",
          flush=True)

    # --- 2. FST + 3. Mantel ---
    print("Computing pairwise FST...", flush=True)
    freqs = site_allele_freqs(pair_df, loci, sites_pair)
    fst_df = pairwise_fst(freqs, loci)
    print(f"  FST matrix: {fst_df.shape}, mean off-diagonal {upper_triangle(fst_df.to_numpy()).mean():.4f}",
          flush=True)
    dist_df = site_distance_matrix(args.sites_tsv, list(fst_df.index))
    mantel_r, mantel_p = mantel_correlation(fst_df, dist_df, n_permutations=999)
    print(f"  Mantel test: r = {mantel_r:.3f}, p = {mantel_p:.3f}", flush=True)
    plot_fst_heatmap_and_mantel(fst_df, dist_df, mantel_r, mantel_p, args.out_fst_png)
    print(f"  wrote {args.out_fst_png}", flush=True)

    # --- 4. LOSO error vs genetic distinctiveness ---
    print("LOSO error vs mean FST...", flush=True)
    mean_fst = (fst_df.sum(axis=1) / (len(fst_df) - 1)).rename("mean_fst")
    err_per_site = err_df["median_error_km"].rename("median_error_km")
    err_r, err_p = plot_err_vs_fst(err_per_site, mean_fst, args.out_err_png)
    print(f"  Pearson r = {err_r:.3f}, p = {err_p:.3f}", flush=True)
    print(f"  wrote {args.out_err_png}", flush=True)

    # --- Summary ---
    args.out_summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        f"# Sculpin: ReLocator results vs classical population-genetic structure\n"
        f"\n"
        f"Three sanity checks comparing ReLocator's per-site LOSO error against\n"
        f"two classical descriptors of population structure (PCA of the\n"
        f"genotype dosage matrix, pairwise Wright FST between sites).\n"
        f"\n"
        f"## PCA\n"
        f"\n"
        f"Standardized feature matrix → SVD → first two principal components\n"
        f"explain **{pc1_var * 100:.1f}% (PC1)** and **{pc2_var * 100:.1f}% (PC2)** of total variance.\n"
        f"See `validation/figures/sculpin_pca.png` — points colored by site;\n"
        f"if the model is using real population structure, sites should cluster\n"
        f"in PC space.\n"
        f"\n"
        f"## Pairwise FST (Wright 1951)\n"
        f"\n"
        f"Mean off-diagonal pairwise FST across the 16 sites: "
        f"**{upper_triangle(fst_df.to_numpy()).mean():.4f}**.\n"
        f"\n"
        f"Isolation-by-distance Mantel test (FST matrix vs haversine geographic\n"
        f"distance matrix between site centroids, 999 permutations):\n"
        f"\n"
        f"  **Pearson r = {mantel_r:.3f}**\n"
        f"  **p ≈ {mantel_p:.3f}**\n"
        f"\n"
        f"See `validation/figures/sculpin_fst.png` for the heatmap and the\n"
        f"FST-vs-distance scatter.\n"
        f"\n"
        f"## ReLocator error vs genetic distinctiveness\n"
        f"\n"
        f"For each site, the mean pairwise FST to all other sites is a\n"
        f"one-number proxy for \"how genetically distinct.\" Sites that should\n"
        f"be easy to locate from genetics alone (high mean FST) should have\n"
        f"low ReLocator LOSO error.\n"
        f"\n"
        f"  **Pearson r = {err_r:.3f}**\n"
        f"  **p = {err_p:.3f}**\n"
        f"\n"
        f"  - Negative r ⇒ ReLocator does what FST predicts: distinct sites\n"
        f"    → low error.\n"
        f"  - Positive r or r ≈ 0 ⇒ ReLocator's errors are uncorrelated with\n"
        f"    classical structure, which would suggest the network is using\n"
        f"    something other than population structure (or, more likely,\n"
        f"    that microsat-derived structure isn't enough at this n / loci).\n"
        f"\n"
        f"See `validation/figures/sculpin_err_vs_fst.png` for the per-site\n"
        f"scatter (each point labeled with its site name).\n"
    )
    args.out_summary_md.write_text(summary)
    print(f"Wrote {args.out_summary_md}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
