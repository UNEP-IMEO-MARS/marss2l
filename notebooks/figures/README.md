# Reproduce the figures of the paper

Code to reproduce the figures of the paper [Artificial intelligence for methane detection: from continuous monitoring to verified mitigation](https://arxiv.org/abs/2511.21777)

## Install `marss2l` package

```bash
pip install marss2l
```

## Re evaluate the models in the MARS-S2L dataset

### Step 1: Download the dataset

```bash
hf download --local-dir /path/to/localdir/MARS-S2L \
            --repo-type dataset UNEP-IMEO/MARS-S2L
```

### Step 2: Download the model weights

```bash
# MARS S2L model
hf download --local-dir . --repo-type dataset UNEP-IMEO/MARS-S2L \
                         trained_models/MARSS2L_20250326/best_epoch

hf download --local-dir . --repo-type dataset UNEP-IMEO/MARS-S2L \
                         trained_models/MARSS2L_20250326/config_experiment.json

# MARS S2L offshore
hf download --local-dir . --repo-type dataset UNEP-IMEO/MARS-S2L \
                         trained_models/MARSS2L_off_20250523/best_epoch

hf download --local-dir . --repo-type dataset UNEP-IMEO/MARS-S2L \
                         trained_models/MARSS2L_off_20250523/config_experiment.json

# MARS S2L nosim
hf download --local-dir . --repo-type dataset UNEP-IMEO/MARS-S2L \
                         trained_models/MARSS2Lnosim_20250605/best_epoch

hf download --local-dir . --repo-type dataset UNEP-IMEO/MARS-S2L \
                         trained_models/MARSS2Lnosim_20250605/config_experiment.json

# Ch4Net
hf download --local-dir . --repo-type dataset UNEP-IMEO/MARS-S2L \
                         trained_models/CH4Net_20250329/best_epoch

hf download --local-dir . --repo-type dataset UNEP-IMEO/MARS-S2L \
                         trained_models/CH4Net_20250329/config_experiment.json

# Ch4Net sim 
hf download --local-dir . --repo-type dataset UNEP-IMEO/MARS-S2L \
                         trained_models/CH4Netsim_20250605/best_epoch

hf download --local-dir . --repo-type dataset UNEP-IMEO/MARS-S2L \
                         trained_models/CH4Netsim_20250605/config_experiment.json
```

### Step 3: Run eval

```bash

# Run eval in the test split MARS S2L model
python -m marss2l.eval_final \
           --split test_2023 \
           --output_dir trained_models/MARSS2L_20250326 \
           --device cpu \
           --path_prepend_data /path/to/localdir/ \
           --num_workers 16\
           --csv_path /path/to/localdir/MARS-S2L/validated_images_all.csv

# Run eval in the test split MARS S2L offshore
python -m marss2l.eval_final \
           --split test_2023 \
           --output_dir trained_models/MARSS2L_off_20250523 \
           --device cpu \
           --path_prepend_data /path/to/localdir/ \
           --num_workers 16\
           --csv_path /path/to/localdir/MARS-S2L/validated_images_all.csv

# Run eval in the test split MARS S2L nosim
python -m marss2l.eval_final \
           --split test_2023 \
           --output_dir trained_models/MARSS2Lnosim_20250605 \
           --device cpu \
           --path_prepend_data /path/to/localdir/ \
           --num_workers 16\
           --csv_path /path/to/localdir/MARS-S2L/validated_images_all.csv

# Run eval in the test split Ch4Net
python -m marss2l.eval_final \
           --split test_2023 \
           --output_dir trained_models/CH4Net_20250329 \
           --device cpu \
           --path_prepend_data /path/to/localdir/ \
           --num_workers 16\
           --csv_path /path/to/localdir/MARS-S2L/validated_images_all.csv

# Run eval in the test split Ch4Net sin
python -m marss2l.eval_final \
           --split test_2023 \
           --output_dir trained_models/CH4Netsim_20250605 \
           --device cpu \
           --path_prepend_data /path/to/localdir/ \
           --num_workers 16\
           --csv_path /path/to/localdir/MARS-S2L/validated_images_all.csv

# Run eval in the controlled releases MARS S2L model
python -m marss2l.eval_final \
           --split control_releases_test \
           --output_dir trained_models/MARSS2L_20250326 \
           --device cpu \
           --path_prepend_data /path/to/localdir/ \
           --num_workers 16\
           --suffix_output controlled_releases\
           --csv_path /path/to/localdir/MARS-S2L/validated_images_all.csv

# Run eval in the CloudSEN12 dataset MARS S2L model
python -m marss2l.eval_final \
           --split 'no split' \
           --output_dir trained_models/MARSS2L_20250326 \
           --device cpu \
           --path_prepend_data /path/to/localdir/ \
           --num_workers 16\
           --suffix_output cloudsen12\
           --csv_path /path/to/localdir/MARS-S2L/cloudsen12_clear_images.csv

# Run eval in the CloudSEN12 dataset CH4Net model
python -m marss2l.eval_final \
           --split 'no split' \
           --output_dir trained_models/CH4Net_20250329 \
           --device cpu \
           --path_prepend_data /path/to/localdir/ \
           --num_workers 16\
           --suffix_output cloudsen12\
           --csv_path /path/to/localdir/MARS-S2L/cloudsen12_clear_images.csv

# MBMP baselines
## MARS S2L test
python -m marss2l.eval_baseline \
           --split test_2023 \
           --output_dir trained_models/MBMP \
           --path_prepend_data /path/to/localdir/ \
           --num_workers 16\
           --csv_path /path/to/localdir/MARS-S2L/validated_images_all.csv

## CloudSEN12
python -m marss2l.eval_baseline \
           --split 'no split' \
           --output_dir trained_models/MBMP \
           --path_prepend_data /path/to/localdir/ \
           --num_workers 16\
           --suffix_output cloudsen12\
           --csv_path /path/to/localdir/MARS-S2L/cloudsen12_clear_images.csv
```

## Figures and statistics about the dataset:

1. Number of images, plumes and sites in each of the training splits: [dataset_stats_by_split_and_geopackage_locations.ipynb](./dataset_stats_by_split_and_geopackage_locations.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marss2l/blob/main/notebooks/figures/dataset_stats_by_split_and_geopackage_locations.ipynb)
2. Time series of images, plumes and sites on each of the regions. Histograms of flux rate of plumes on each of the regions: [figure_number_of_images_per_country.ipynb](./figure_number_of_images_per_country.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marss2l/blob/main/notebooks/figures/figure_number_of_images_per_country.ipynb)
3. Minimum detection limit by case study and global: [mdl_exploration_by_case_study.ipynb](./mdl_exploration_by_case_study.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marss2l/blob/main/notebooks/figures/mdl_exploration_by_case_study.ipynb), [mdl_exploration_adapted.ipynb](./mdl_exploration_adapted.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marss2l/blob/main/notebooks/figures/mdl_exploration_adapted.ipynb)
4. Wind speed distribution and distribution of size of the plumes: [figure_wind_speed.ipynb](./figure_wind_speed.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marss2l/blob/main/notebooks/figures/figure_wind_speed.ipynb)
5. Distribution of ToA reflectances and MBMP noise: [stats_dataset_toareflectances.ipynb](./stats_dataset_toareflectances.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marss2l/blob/main/notebooks/figures/stats_dataset_toareflectances.ipynb)

## Figures and statistics of the model:

1. Statistics and figures on the 2024 test set: [eval_model_and_figure_prob_vs_emission_rate.ipynb](./eval_model_and_figure_prob_vs_emission_rate.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marss2l/blob/main/notebooks/figures/eval_model_and_figure_prob_vs_emission_rate.ipynb)
2. Statistics and figures on the controlled releases [figure_controlled_releases](./figure_controlled_releases.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marss2l/blob/main/notebooks/figures/figure_controlled_releases.ipynb)
3. Statistics and figures on the CloudSEN12 experiment [cloudsen12_experiment.ipynb](./cloudsen12_experiment.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marss2l/blob/main/notebooks/figures/cloudsen12_experiment.ipynb)
4. Ablation in the number of pixels to provide the scene-level score [ablation_threshold_pixels.ipynb](./ablation_threshold_pixels.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marss2l/blob/main/notebooks/figures/ablation_threshold_pixels.ipynb)
5. Examples of false positive detections, one figure per example: [false_positives_examples.ipynb](./false_positives_examples.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marss2l/blob/main/notebooks/figures/false_positives_examples.ipynb)

## Other
1. IEA estimates of emissions: [iea_estimates_of_emissions.ipynb](./iea_estimates_of_emissions.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UNEP-IMEO-MARS/marss2l/blob/main/notebooks/figures/iea_estimates_of_emissions.ipynb)
