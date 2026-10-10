# Photon-noise floors of methane retrieval: reproducing the paper

Code behind *On the limits of methane detection for Sentinel-2 and Landsat* (in preparation).

The paper propagates photon noise through the multi-band multi-pass (MBMP) retrieval to a per-pixel
floor on the retrieved ΔXCH₄, at three levels of what a background estimate may replace (L1, L2,
L3). It compares those floors with the noise the retrieval actually reads on plume-free pixels over
the MARS-S2L test split and a worldwide CloudSEN12 control corpus, and converts both into the
smallest source rate detected half the time.

## Data

Everything below reads a local copy of the
[MARS-S2L dataset on Hugging Face](https://huggingface.co/datasets/UNEP-IMEO/MARS-S2L), with the
layout of the repository:

```bash
hf download --local-dir <MARS-S2L root> --repo-type dataset UNEP-IMEO/MARS-S2L
```

The files the analysis uses (columns are described in the dataset card):

| File | What it holds |
| --- | --- |
| `validated_images_all.csv` | Image metadata, including the reference (background) pass: `tile_date_bg`, `sza_bg`, `vza_bg`, `satellite_bg`, and the `sza_source` / `sza_bg_source` flags |
| `validated_images_plumes.csv` | Validated plumes, with their source rate recomputed on the native grid (`*_native` columns) |
| `data/stats_dataset.csv` | Per-image statistics of the train, validation and test images on the 10 m chips, with the photon-noise columns |
| `data/stats_dataset_native.csv` | The paper's statistics: test split, Sentinel-2 at 20 m and Landsat at 30 m |
| `cloudsen12_clear_images.csv` | Metadata of the CloudSEN12 control scenes, with their reference pass |
| `cloudsen12_data/cloudsen12_stats_dataset.csv` | CloudSEN12 statistics on the 10 m chips |
| `cloudsen12_data/cloudsen12_stats_dataset_native.csv` | CloudSEN12 statistics at 20 m, used by the paper |

The Permian basin label needs the EIA's `PermianBasin_Extent` layer, from
[PermianBasin_Boundary_Structural_Tectonic.zip](https://www.eia.gov/maps/map_data/PermianBasin_Boundary_Structural_Tectonic.zip).

## Code

| Module | Role |
| --- | --- |
| `marss2l/shot_noise.py` | Reflectance to radiance, SNR scaled to the radiance of each pixel, the L1/L2/L3 floors, σ(ΔXCH₄) and the minimum significant enhancement ε, and the Monte-Carlo check of the closed forms |
| `marss2l/stats_dataset.py` | The per-image sweep: measured noise on plume-free pixels and the floors, one row per image |
| `marss2l/resampling.py` | Sentinel-2 back on its native 20 m grid, by inverting the bilinear 20 → 10 m step of the chips |
| `marss2l/solar_geometry.py` | Solar zenith angles: parsing the acquisition time from a product name, and repairing implausible stored angles |
| `scripts/shot_noise/figure_regional.py` | Every data figure (floors, gap, drivers, breaches, CloudSEN12 supplement, detectable source rate) and the per-region table |
| `scripts/shot_noise/figure_example_scenes.py` | The raster figures: example scenes and plumes drawn against their per-pixel floor |
| `scripts/shot_noise/figure_monte_carlo.py` | The agreement of the first-order expansions with Monte Carlo |
| `scripts/shot_noise/make_paper_figures.sh` | All the figures in one command, every scene id pinned |

## Reproducing each product

Run from the root of this repository.

| Product | How to regenerate it | Data it needs |
| --- | --- | --- |
| All figures and the per-region table | `bash scripts/shot_noise/make_paper_figures.sh <MARS-S2L root> <PermianBasin_Extent shapefile> <output dir>` | Public. The two Landsat-9 rows of `example_plumes.png` were drawn from 30 m chips: *private only* (see below); from the public 10 m chips they look smoother, with the same labels |
| Monte-Carlo figure alone | `python -m scripts.shot_noise.figure_monte_carlo figures --output-dir <dir>` | None |
| `data/stats_dataset.csv` | `python -m marss2l.stats_dataset --csv-path <root>/validated_images_all.csv --path-prepend-data <root> --split <split> --output-file <file>` for `train_2023`, `val_2023` and `test_2023`, then concatenated | Public |
| Sentinel-2 rows of `data/stats_dataset_native.csv` | the same command with `--split test_2023 --native-grid --window-size 100` | Public |
| Landsat rows of `data/stats_dataset_native.csv` | the same sweep with `--window-size 67` over Landsat chips exported at 30 m | *Private only*: the 30 m chips are not published |
| `ch4_valid_noplume_std_10m` of the native files | `ch4_valid_noplume_std` of the same image in the 10 m statistics | Public for MARS-S2L, *private only* for CloudSEN12 |
| CloudSEN12 statistics, 10 m and 20 m | `python -m marss2l.stats_dataset --dataset-name CloudSEN12` over the CloudSEN12 scenes, with `--native-grid --window-size 100` for 20 m | *Private only*: the CloudSEN12 imagery is not published |
| Reference-pass columns of the images tables | Read from the operational database of the monitoring system, then `solar_geometry.repair_sza_dataframe` | *Private only*; the repair itself is public code |

The sweep takes `--num-workers` data-loader processes (default 4). A sweep over all splits runs for
a few hours on a workstation; `--max-iter` stops it early for a quick look.

## Tests

`tests/test_shot_noise.py`, `tests/test_stats_dataset.py`, `tests/test_resampling.py` and
`tests/test_solar_geometry.py` cover the modules above, including the Monte-Carlo agreement of the
closed forms.
