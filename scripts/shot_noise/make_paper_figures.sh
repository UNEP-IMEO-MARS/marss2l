#!/usr/bin/env bash
# The figures of "On the limits of methane detection for Sentinel-2 and Landsat", from a local
# copy of the MARS-S2L dataset (Hugging Face: UNEP-IMEO/MARS-S2L).
#
# Usage, from the root of this repository:
#
#     scripts/shot_noise/make_paper_figures.sh <MARS-S2L root> <Permian basin shapefile> <output dir>
#
# The Permian basin polygon is the EIA's PermianBasin_Extent layer
# (PermianBasin_Boundary_Structural_Tectonic.zip).
#
# The raster figures draw Sentinel-2 on its native 20 m grid, recovered from the 10 m chips.
# Landsat is drawn from the chips the images CSV points to: the two Landsat-9 rows of
# example_plumes.png were drawn from 30 m chips, which are not part of the dataset, so from the
# published 10 m chips those two rows look smoother than in the paper. Their labels, which come
# from the statistics, are the same. Set IMAGES to an images CSV pointing at 30 m Landsat chips
# to reproduce them exactly.
set -euo pipefail

DATA=$1
PERMIAN=$2
OUT=$3
STATS=$DATA/data/stats_dataset_native.csv
IMAGES=${IMAGES:-$DATA/validated_images_all.csv}
mkdir -p "$OUT"

# Floors, gap, breaches, noise drivers, CloudSEN12 supplement and detectable flux (11 figures),
# and the per-region table the text quotes.
python -m scripts.shot_noise.figure_regional figures "$STATS" "$IMAGES" \
    --extra-stats-csv "$DATA/cloudsen12_data/cloudsen12_stats_dataset_native.csv" \
    --extra-images-csv "$DATA/cloudsen12_clear_images.csv" \
    --permian-shapefile "$PERMIAN" --output-dir "$OUT" --summary-csv "$OUT/region_summary.csv"

# Appendix B: the first-order expansions against Monte Carlo. No data needed.
python -m scripts.shot_noise.figure_monte_carlo figures --output-dir "$OUT"

scenes() {
    python -m scripts.shot_noise.figure_example_scenes figure "$STATS" "$IMAGES" \
        --permian-shapefile "$PERMIAN" --path-prepend-data "$DATA" --native-grid "$@"
}

# Three plume-free Sentinel-2A scenes: Permian basin, Uzbekistan & Kazakhstan, China.
scenes --output-path "$OUT/example_scenes.png" \
    --only 98e49b1f-4838-4c31-a684-efe3e23c48b8 \
    --only 2da11c8c-b70c-427c-bcb9-7e2d179e6cf2 \
    --only 303719b4-fcde-48e5-81d0-c4a63af07748

# Five validated plumes: Egypt, Turkmenistan, Syria, Permian basin, Iran.
scenes --plumes --output-path "$OUT/example_plumes.png" \
    --only eb382a5d-d768-4661-9aab-0c07a20456ff \
    --only 57485b16-9c17-46fc-9a3a-9d74a44aa6b8 \
    --only 9a7e5701-163a-47cb-aded-5613d12e7543 \
    --only e57557e3-ec88-4bd3-9189-b9af1648454d \
    --only b16dd610-f395-436b-b144-4e364fe01434

# Weak plumes near the L1 floor: Turkmenistan, Algeria, Syria, Saudi Arabia.
scenes --rung L1 --label-by country --ch4-vmax 600 --plumes --show-flux \
    --output-path "$OUT/weak_plumes_at_limit.png" \
    --only eb53420b-b504-44db-9e3a-37992a056e3c \
    --only d16faad1-3b99-44d8-a85c-4667a58b9a4d \
    --only a6032970-a58f-410d-94d8-1cb5815c121b \
    --only b2d60cfe-00c5-4b9c-b103-7bf08f1ba3e1

# Plume-free scenes whose noise reads below L1: Libya, Algeria, Egypt, Turkmenistan.
scenes --rung L1 --label-by country --ch4-vmax 600 --detection \
    --output-path "$OUT/below_L1_scenes.png" \
    --only 06e0e66f-d375-4584-adb2-f0e34f2cabfd \
    --only f122a63a-eff6-4962-82ce-758bddf43db6 \
    --only cf408830-1b42-460b-a28b-41ec87480617 \
    --only d3b77218-3ab8-4654-bf0b-80f0ca85b7bf
