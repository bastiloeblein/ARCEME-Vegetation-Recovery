# Paper figures

This directory contains small, reproducible utilities for thesis figures that
are not part of model training or model selection.

## Cross-validation split map

`plot_cv_split_maps.py` reads `cv_splits.json` from a completed CV run and
plots the exact training and validation cube locations for every fold. It also
writes a CSV table with the number of cubes and the withheld Köppen-Geiger
classes per fold.

Run from the repository root on Phaestos:

```bash
python -m evaluation.paper_figures.plot_cv_split_maps \
  --run-dir /net/home/sloeblein/ARCEME-Vegetation-Recovery/model/wand_db_logs/CV_Training_SGEDConvLSTM_RGBI_2026-08-03_20-20-39 \
  --metadata-csv data_processing/data/train_test_split.csv
```

The script requires Cartopy for the world basemap. If it is not installed in
`my_env`, install it once with:

```bash
conda install -c conda-forge cartopy
```

Outputs are written to `results/paper_figures/cv_splits/` within the selected
run directory by default.
