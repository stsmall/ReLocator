# Sculpin validation: microsat loader demonstration + publication figures

This directory holds the real-data validation of ReLocator's native microsatellite loader (PR #46 → `locator --microsat ...` / `loc.load_genotypes(microsat=...)`), run on a published prickly-sculpin (*Cottus asper*) microsat dataset.

## Data source

| Item | Path | Notes |
|---|---|---|
| Genepop microsat genotypes | `/sietch_colab/ssmall/projects/relocator_dir/test_data/cottus_asper/DRYAD_comb19MSA.gen` | 405 individuals × 19 loci across 16 populations; Dennenmoser et al. 2014 (Dryad doi:10.5061/dryad.8ht04) |
| Site coordinates | `validation/sculpin/sites.tsv` | 16 sites with lat/lon |
| Range polygon | `validation/sculpin/freshwater_range.{shp,dbf,prj,shx,cpg}` | Pacific NW freshwater range used by the LOSO range-mask penalty |

## Quick demo — proves the native `--microsat` loader works on sculpin

One command, end to end, no GPU required:

```bash
TEST_DATA=/sietch_colab/ssmall/projects/relocator_dir/test_data
mkdir -p /tmp/sculpin-demo

# Convert the Dryad Genepop file → ReLocator pair-format TSV + sample_data.txt
pixi run python -m validation.sculpin.parse_genepop \
  --genepop          $TEST_DATA/cottus_asper/DRYAD_comb19MSA.gen \
  --sites            validation/sculpin/sites.tsv \
  --out_microsat     /tmp/sculpin-demo/sculpin_microsat.tsv \
  --out_sample_data  /tmp/sculpin-demo/sample_data.txt

# Load through the native --microsat path (no preprocessing script needed)
pixi run python -c "
from locator.core import Locator
g, s = Locator({}).load_genotypes(microsat='/tmp/sculpin-demo/sculpin_microsat.tsv')
print(f'sculpin: {g.shape[1]} samples × {g.shape[0]} multi-allelic dosage features')
print(f'first 3 samples: {list(s[:3])}')
"
```

Expected output:

```
sculpin: 405 samples × 207 multi-allelic dosage features
first 3 samples: ['Alaska_001', 'Alaska_002', 'Alaska_003']
```

The 207 multi-allelic dosage features encode the 19 microsat loci (one column per unique allele per locus, 0/1/2 counts; rare alleles below `--microsat_maf` dropped at load time).

## Full reproduction — LOSO sweep + publication figures

Run all commands from the repo root (`/sietch_colab/ssmall/projects/relocator_dir/ReLocator/`).

```bash
GENEPOP=$TEST_DATA/cottus_asper/DRYAD_comb19MSA.gen
SCULPIN=validation/sculpin
SHP=$SCULPIN/freshwater_range.shp

# 1. Parse the Dryad Genepop file (one-time prep)
mkdir -p out/sculpin/inputs
pixi run python -m $SCULPIN/parse_genepop \
  --genepop          $GENEPOP \
  --sites            $SCULPIN/sites.tsv \
  --out_microsat     out/sculpin/inputs/sculpin_microsat.tsv \
  --out_sample_data  out/sculpin/inputs/sample_data.txt

# 2. LOSO sweep with the location-masking penalty (dosage mode)
pixi run python -m validation.sculpin.run_loso_rangemask \
  --microsat        out/sculpin/inputs/sculpin_microsat.tsv \
  --sample_data     out/sculpin/inputs/sample_data.txt \
  --range_shp       $SHP \
  --out_dir         out/sculpin/loso_rangemask \
  --mode            dosage \
  --max_epochs      500

# 3. Summary table + sculpin_summary.md + sculpin_loso.tsv
pixi run python -m validation.sculpin.summarize \
  --out_dir         out/sculpin \
  --sample_data     out/sculpin/inputs/sample_data.txt

# 4. Cartopy map (sculpin_map_dosage_rangemask.png)
pixi run python -m validation.sculpin.make_map \
  --out_dir         out/sculpin/loso_rangemask \
  --sample_data     out/sculpin/inputs/sample_data.txt \
  --out_png         validation/figures/sculpin_map_dosage_rangemask.png \
  --range_shp       $SHP

# 5. Genetic-structure sanity check (sculpin_pca.png, sculpin_fst.png, sculpin_err_vs_fst.png)
pixi run python -m validation.sculpin.genetic_structure \
  --features_tsv    out/sculpin/loso_rangemask/features_dosage.tsv \
  --pair_tsv        out/sculpin/inputs/sculpin_microsat.tsv \
  --sites_tsv       $SCULPIN/sites.tsv \
  --loso_dir        out/sculpin/loso_rangemask \
  --out_pca_png     validation/figures/sculpin_pca.png \
  --out_fst_png     validation/figures/sculpin_fst.png \
  --out_err_png     validation/figures/sculpin_err_vs_fst.png

# 6. Snap-to-nearest-site experiment (sculpin_snap.png)
pixi run python -m validation.sculpin.snap_to_site \
  --loso_dir        out/sculpin/loso_rangemask \
  --sample_data     out/sculpin/inputs/sample_data.txt \
  --sites_tsv       $SCULPIN/sites.tsv \
  --out_csv         validation/summary/sculpin_snap_per_site.tsv \
  --out_png         validation/figures/sculpin_snap.png \
  --out_summary_md  validation/summary/sculpin_snap.md
```

LOSO step ≈ hours on a single GPU (16 folds × 500 epochs); the four post-LOSO steps each run in seconds–minutes.

## Figure index

Existing figures on this branch (already produced, ready for publication use):

| Figure | What it shows | Reproducer |
|---|---|---|
| `figures/sculpin_map_dosage.png` | Cartopy map: LOSO predictions on plain dosage mode | `make_map.py --modes dosage` |
| `figures/sculpin_map_dosage_rangemask.png` | Same as above with the range-mask training penalty applied | `make_map.py` with rangemask LOSO outputs |
| `figures/sculpin_map_compare.png` | Side-by-side comparison of dosage vs range-mask predictions | `make_map.py` |
| `figures/sculpin_pca.png` | PCA of multi-allelic dosage features; points colored by site | `genetic_structure.py --out_pca_png` |
| `figures/sculpin_fst.png` | Pairwise F<sub>ST</sub> heatmap across the 16 sites | `genetic_structure.py --out_fst_png` |
| `figures/sculpin_err_vs_fst.png` | Per-site LOSO mean error vs population isolation (F<sub>ST</sub>) | `genetic_structure.py --out_err_png` |
| `figures/sculpin_snap.png` | Effect of snapping LOSO predictions to the nearest known site | `snap_to_site.py --out_png` |
| `figures/sculpin_modes.png` | Bar chart: per-mode (dosage / geometry / repeat_norm) median LOSO error | **branch-only experimental; needs deleted `scripts/microsat_to_locator.py`** — keep as historical |

Quantitative summaries:

- `summary/sculpin_summary.md` — narrative summary with the headline LOSO numbers
- `summary/sculpin_loso.tsv` — per-fold table (mode × holdout site × mean km error)
- `summary/sculpin_genetic_structure.md` — F<sub>ST</sub> + PCA + Mantel narrative
- `summary/sculpin_snap.md` + `sculpin_snap_per_site.tsv` — snap experiment results

## Known limitations

- **`run_loso.py` is currently broken** (calls the deleted `scripts/microsat_to_locator.py` to compute geometry / repeat_norm features). The dosage mode of the model comparison is still reproducible via `run_loso_rangemask.py --mode dosage`, but the full 3-mode comparison that produced `sculpin_modes.png` requires resurrecting `microsat_to_locator.py` (e.g., `git show archive/microsat-sculpin-pre-rebuild:scripts/microsat_to_locator.py > scripts/microsat_to_locator.py`) or porting the geometry / repeat_norm encoders out of the legacy script. Geometry and repeat_norm were ruled out for merge in the sculpin LOSO comparison anyway (dosage 237 km median << geometry 404 km << repeat_norm 395 km), so this gap mostly matters for historical reproducibility, not new scientific output.
- **`run_loso_rangemask.py`** still expects a pre-computed `--feature-matrix` TSV in the legacy path. A small rewire to call `loc.load_genotypes(microsat=...)` directly would close that gap. Pending follow-up — say the word and I'll do it.
