# Rainfall RQA Pipeline

Pipeline for identifying spatially coherent rainfall communities and
analyzing their recurrence dynamics via Recurrence Quantification Analysis
(RQA), for two study regions:

- **SP** – South Peninsular India (IMD gridded rainfall, CSV, already in mm)
- **GHA** – Greater Horn of Africa (ERA5 `tp`, NetCDF, natively in metres)

Each region is analyzed for two seasons via presets: `JJAS_full` (Kiremt /
boreal summer) and `OND_full` (short rains).

## Data

The input rainfall data is **not** included in this repository and must be
downloaded separately from the HU-Box (UP.box) share. The link and password
are given in the thesis document (MA) itself.

After downloading, place the files so the paths below (from `submit.sh` /
`submit_EA.sh`) resolve correctly, e.g.:

```
data/
├── Rainfall_SP_1951_2023.csv     # SP / India, IMD
├── ea_bimodal_fin.nc             # GHA, ERA5
├── South_Peninsular.shp          # SP clipping shapefile (+ .dbf/.shx/.prj)
└── full_east_africa.geojson      # GHA clipping shapefile
```

## Environment

```bash
conda env create -f environment.yml
conda activate ghasp
```

> **Note:** `submit.sh` and `submit_EA.sh` currently still call
> `source activate rain`. Update that line to `source activate ghasp` in
> both scripts (or rename your local environment to `rain`) before
> submitting them, so the environment name matches.

## Repository layout

| File | Purpose |
|---|---|
| `analysis.py` | Command-line entry point. Runs the full pipeline (prepare → cluster → RQA → plots) for one or several presets. |
| `summarized_func.py` | Data preparation (`prepare`, `heatmap_periods`), clustering (`run_k_analysis`, `smooth_cluster`), and RQA orchestration (`rqa_full`) for one preset. |
| `community.py` | Community-detection pipeline: PCA/spectral filtering, sliding-window heatmap, agglomerative clustering (k via Kneedle on the q-statistic), and mask post-processing (smoothing, splitting, merging) into final communities. |
| `rqa.py` | Recurrence Quantification Analysis on rainfall time series: recurrence plots, RQA metrics (DET, LAM, ENTR, L_mean), Theiler-window estimation, rolling-window RQA with IAAFT surrogate significance testing, and Mann-Kendall trend export to CSV. |
| `plots_and_stats.py` | Plotting and export utilities: styling helpers, community export to shapefile, cluster/heatmap plots, and RQA/rain diagnostic plots. |
| `create_plots.ipynb` | Notebook that builds the final thesis figures (Mann-Kendall dot-plot grids, combined cluster overview grids, rank-significance-period plots) from the CSV/plot outputs that `analysis.py` already produced. Run this **after** the pipeline has completed for the presets you want to compare. |
| `submit.sh` | SLURM job script running the pipeline for **SP / India**. |
| `submit_EA.sh` | SLURM job script running the pipeline for **GHA**. |
| `environment.yml` | Conda environment definition (`ghasp`). |

## Running the pipeline

`analysis.py` is a plain Python CLI script; `submit.sh` / `submit_EA.sh`
are just SLURM wrappers around the same calls (with `#SBATCH` headers and
logging). To run it directly, activate the environment and call `python`:

### India / SP

Loops over both presets and only passes `--min_community_size 34` for the
`OND_*` preset (JJAS uses the automatic Kneedle-based size):

```bash
conda activate ghasp

for PRESET in OND_full JJAS_full ; do

  MCS_ARGS=()
  if [[ "$PRESET" == OND_* ]]; then
    MCS_ARGS=(--min_community_size 34)
  fi

  python -u analysis.py \
    --path "data/Rainfall_SP_1951_2023.csv" \
    --shapefile "data/South_Peninsular.shp" \
    --mode preset \
    --preset "$PRESET" \
    --out_dir "results/SP" \
    --cluster_figsize "8,6" \
    --cluster_scatter "50" \
    --panel_figsize "16,6" \
    --panel_scatter "40" \
    --sort_by "south" \
    "${MCS_ARGS[@]}"

done
```

### GHA

Loops over both presets with the same arguments (no `--min_community_size`
override — always automatic):

```bash
conda activate ghasp

for PRESET in JJAS_full OND_full ; do
  python -u analysis.py \
    --path "data/ea_bimodal_fin.nc" \
    --shapefile "data/full_east_africa.geojson" \
    --mode preset \
    --preset "$PRESET" \
    --out_dir "results/GHA" \
    --cluster_figsize "6,6" \
    --cluster_scatter "540" \
    --panel_figsize "16,6" \
    --panel_scatter "40" \
    --sort_by "north"
done
```

To submit either region as a SLURM job on the cluster instead, run
`sbatch submit.sh` / `sbatch submit_EA.sh`.

Note the two regions use different `--sort_by`: cluster IDs are ordered by
latitude centroid (`south`) for SP, and by northernmost pixel (`north`) for
GHA — see `smooth_cluster` in `summarized_func.py`.

Other useful CLI options (see `analysis.py --help` for the full list):

- `--mode all` — run every preset in `PRESETS`, optionally restricted with `--only "JJAS_full,OND_full"`
- `--mode single --window <N> --months "6,7,8,9" --year "none"` — run an arbitrary window/month/year combination outside the defined presets
- `--list` — print the available presets and exit

## Outputs

All outputs for a given run land under `<out_dir>/<preset>/`, e.g.
`results/SP/JJAS_full/`:

| Path | Produced by | Contents |
|---|---|---|
| `heatmap.png`, `Heatmap.csv` | `prepare` | Recurrence-fraction heatmap and its per-pixel CSV export. |
| `*_sc_curve.svg` | `prepare` | Spectral-concentration threshold curves for sampled windows. |
| `community_size_distribution_final.svg` | `prepare` | Community-size histogram/CCDF with the chosen `min_community_size`. |
| `aggl_clu.png` | `plot_cluster` | Raw agglomerative clustering result. |
| `smoothing.png` | `plot_smoothing` | Heatmap / raw clustering / smoothed clustering, side by side. |
| `two_panel.png` | `plot_two_panel` | Final interpolated heatmap + smoothed community map. |
| `communities.shp` (+ sidecars) | `export_communities` | Final community masks as a shapefile. |
| `periods/` | `heatmap_periods` | Recurrence-fraction heatmap per sub-period (grid PNG, per-period SVGs and shapefiles), for checking community stability over time. |
| `{DET,LAM,ENTR,RAIN,LMEAN,DRY_FRACTION,DRY_NODE_FRACTION}.png` | `rqa_full` | Combined plot per metric, overlaying all communities. |
| `C{NN}_rqa_rain.png` | `plot_rqa_with_rain` | Per-community RQA + rain diagnostic panels. |
| `rqa_rW/cluster_{NN}_{METRIC}_{preset}.csv` | `plot_metric_rolling` (via `rqa_full`) | Rolling metric series plus Mann-Kendall results, per community and metric. |
| `rqa_rW/mk_results_{preset}.csv` | `rqa_full` | Combined Mann-Kendall trend table (all communities, all metrics, all periods) — this is the file `create_plots.ipynb` reads. |

`create_plots.ipynb` reads from these `results/<region>/<preset>/` folders
(via `base_dir` / `results_dir` arguments) and writes the final thesis
figures to whatever `out_dir` / `plot_path` you pass in its example calls
(e.g. a separate `Plotting/` folder).

## Generating the final figures (`create_plots.ipynb`)

Once `analysis.py` has finished for the presets you want to compare, run
`create_plots.ipynb` (`conda activate ghasp && jupyter lab`) to build the
final thesis figures from the `results/<region>/<preset>/...` outputs.
