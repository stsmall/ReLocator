# KNP elephants: ReLocator validation summary

Real-data validation of the microsat input pipeline against an elephant
microsat dataset from Kibale National Park, Uganda
(*Loxodonta africana × L. cyclotis* hybrid zone). 124 individuals, 14
microsat loci, **per-individual GPS coords** spanning a tight ~30 × 50 km
area inside the park (lon 30.26°-30.54°E, lat 0.22°-0.68°N, equatorial).
Random 80/20 split (smoke) + 5-fold CV × 3 seeds.

Source data: `KNP_ConsensusGenotypes.xlsx` (forest×savanna hybrids per
STRUCTURE Q≥0.9 classification: 101 hybrid, 22 savanna, 1 forest; 100
S-clade mtDNA, 24 F-clade). Note that the source file's GPS column
labels are swapped — `GPS_long_DD` actually contains latitudes,
`GPS_lat_DD` contains longitudes — this is corrected by
`validation/elephants/parse_xlsx.py`.

Metric: Euclidean distance in decimal degrees on (lon, lat). Over a
~50 km box at the equator this is ≈ haversine in km × 111.

## Centroid baseline

Predict the training-fold mean (lon, lat) for every held-out individual
— i.e. learn nothing from the genotypes:

  k-fold mean(median): **0.136° ≈ 15.1 km**

This is the floor any feature-based mode must beat to demonstrate signal.

## Results

K-fold (k=5, 3 seeds, mean ± std of per-fold median):

| Mode | Median error (°) | ≈ km | Δ vs centroid |
|---|---|---|---|
| `centroid` baseline | 0.136 ± 0.010 | 15.1 | — |
| `dosage` | 0.129 ± 0.021 | 14.4 | -0.007° (-0.7 km, **better**) |
| `geometry` | 0.147 ± 0.022 | 16.3 | +0.011° (+1.2 km, **worse**) |
| `repeat_norm` | 0.135 ± 0.011 | 15.0 | -0.001° (-0.1 km, tied) |

**No mode meaningfully beats the centroid baseline at this scale.**
`dosage` is marginally below baseline (~0.8 km, error bars overlap
heavily); `geometry` is slightly worse than baseline; `repeat_norm` is
tied. The 14 microsat loci × 50 km park is in the regime the user
predicted upfront would be too sparse to extract within-population
structure: there's not enough genetic differentiation between
individuals at fine spatial scales for ReLocator's regression head to
exploit.

## Per-individual map

`validation/figures/elephants/elephant_map_dosage.png` shows true GPS
coords (★ blue) and median k-fold prediction (● red) for each of the
124 individuals, with lines connecting prediction to truth. Predictions
are scattered through the park region with no clear localization
pattern — consistent with the bar-chart finding that the model is
essentially predicting toward the centroid for everyone.

## What this confirms

1. **The pipeline works on real elephant microsat data** — the converter
   handles the `<locus>.1`/`<locus>.2` paired-column input format, the
   GPS-label-swap is corrected, and 48/48 LOSO folds completed without
   errors. The integration is sound.

2. **Fine-scale (≤ 50 km) localization from 14 microsats is below the
   resolution limit** of this architecture-and-data combination. This
   is consistent with the user's hypothesis going in: at this geographic
   scale, microsat-based population structure can't drive sub-km
   localization without either (a) more loci, (b) more individuals, or
   (c) a model architecture that captures relatedness/kinship rather
   than treating individuals i.i.d.

3. **In contrast, real microsat data at 2500 km (sculpin, 19 loci × 405
   indiv × 16 sites) DOES show real signal** — see
   `validation/summary/sculpin_summary.md`: dosage gets 267 km median
   LOSO error vs a 440 km centroid baseline. The relevant ratio appears
   to be (geographic scale) ÷ (number of independent loci): sculpin's
   ratio is ~130 km/locus vs Kibale's ~3.5 km/locus.

## Files

- `validation/elephants/parse_xlsx.py` — xlsx → pair-format TSV +
  sample_data.txt (with GPS-label unswap).
- `validation/elephants/make_map.py` — Kibale-area cartopy map.
- `validation/summary/elephants/elephant_kfold.tsv` — raw per-fold rows.
- `validation/figures/elephants/elephant_modes.png` — bar chart with
  centroid overlay.
- `validation/figures/elephants/elephant_scatter.png` — predicted-vs-true
  smoke scatter, one panel per mode.
- `validation/figures/elephants/elephant_map_dosage.png` — Kibale map.
