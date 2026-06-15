# Sculpin: ReLocator results vs classical population-genetic structure

Three sanity checks comparing ReLocator's per-site LOSO error against
two classical descriptors of population structure (PCA of the
genotype dosage matrix, pairwise Wright FST between sites).

## PCA

Standardized feature matrix → SVD → first two principal components
explain **8.0% (PC1)** and **3.3% (PC2)** of total variance.
See `validation/figures/sculpin_pca.png` — points colored by site;
if the model is using real population structure, sites should cluster
in PC space.

## Pairwise FST (Wright 1951)

Mean off-diagonal pairwise FST across the 16 sites: **0.1583**.

Isolation-by-distance Mantel test (FST matrix vs haversine geographic
distance matrix between site centroids, 999 permutations):

  **Pearson r = 0.201**
  **p ≈ 0.077**

See `validation/figures/sculpin_fst.png` for the heatmap and the
FST-vs-distance scatter.

## ReLocator error vs genetic distinctiveness

For each site, the mean pairwise FST to all other sites is a
one-number proxy for "how genetically distinct." Sites that should
be easy to locate from genetics alone (high mean FST) should have
low ReLocator LOSO error.

  **Pearson r = 0.541**
  **p = 0.030**

  - Negative r ⇒ ReLocator does what FST predicts: distinct sites
    → low error.
  - Positive r or r ≈ 0 ⇒ ReLocator's errors are uncorrelated with
    classical structure, which would suggest the network is using
    something other than population structure (or, more likely,
    that microsat-derived structure isn't enough at this n / loci).

See `validation/figures/sculpin_err_vs_fst.png` for the per-site
scatter (each point labeled with its site name).
