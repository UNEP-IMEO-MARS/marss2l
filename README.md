# MARS-S2L

Code to reproduce [*AI for methane emission detection and mitigation*](https://arxiv.org/abs/2408.04745v1)

### Licence

The MARS-S2L database and all pre-trained models are released under a [Creative Commons Non-Commercial Share-Alike licence](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt). The code in this package is published under a [GNU Lesser GPL v3 licence](https://www.gnu.org/licenses/lgpl-3.0.en.html).

## Install

Create the `marss2lpy312` virtual environment with conda:

```
conda env create -f environment/conda_vm.yaml
```

Install `marss2l` package:

```
pip install -e .
```

## Retrain the MARS-S2L model

We trained the model on a A100 gpu with 80Gb RAM. In order to re-train the proposed model and the baseline run:

```
wandb init # select the wandb project to display the training metrics
python -m marss2l.reproduce_training --csv_path az://public/MARS-S2L/dataset_20250609/validated_images_all.csv\
                                     --batch_size 128 --num_workers 16 --num_workers_val 8\
                                     --batch_size_val 256 --n_samples_per_epoch_train 65536\
                                     --basename_dir train_logs # Where to store the training outputs
```

This code will train 4 models the proposed MARS-S2L model and the CH4Net model both trained on the train split of the dataset. In addition it will evaluate these models in the test split of the dataset. After the training the content of the `basename_dir` folder should be:

```
train_logs/
├── marss2l
│   ├── best_epoch
│   ├── config_experiment.json
│   └── preds_test_2023.csv
├── ch4net
│   ├── best_epoch
│   ├── config_experiment.json
│   └── preds_test_2023.csv
├── ch4net_sim
│   ├── best_epoch
│   ├── config_experiment.json
│   └── preds_test_2023.csv
└── marss2l_nosim
    ├── best_epoch
    ├── config_experiment.json
    └── preds_test_2023.csv

```

Where:
* `best_epoch` are the weights of the trained model
* `config_experiment.json` is the metadata with the configuration of the experiment
* `preds_test_2023.csv` are the model predictions for all entries in the test set.


## Reproduce the figures

In the folder `notebooks/figures` there is the code to reproduce most of the figures of the paper. These notebooks read the data directly from the public Azure container. See [README in this folder](https://github.com/gonzmg88/marss2l/blob/main/notebooks/figures/README.md).

## Load trained models

The MARS-S2L model of the paper is stored in a public Azure container. The files are stored in the following folders following the same naming convention as described above.
* MARS-S2L model: [`best_epoch`](https://unepazeconomyadlsstorage.blob.core.windows.net/public/MARS-S2L/train_logs_revision/MARSS2L_20250326/best_epoch)  [`config_experiment.json`](https://unepazeconomyadlsstorage.blob.core.windows.net/public/MARS-S2L/train_logs_revision/MARSS2L_20250326/config_experiment.json).
* MARS-S2L model offshore: [`best_epoch`](https://unepazeconomyadlsstorage.blob.core.windows.net/public/MARS-S2L/train_logs_revision/MARSS2L_off_20250523/best_epoch)  [`config_experiment.json`](https://unepazeconomyadlsstorage.blob.core.windows.net/public/MARS-S2L/train_logs_revision/MARSS2L_off_20250523/config_experiment.json)
* The baseline CH4Net model [`best_epoch`](https://unepazeconomyadlsstorage.blob.core.windows.net/public/MARS-S2L/train_logs_revision/CH4Net_20250329/best_epoch) [`config_experiment.json`](https://unepazeconomyadlsstorage.blob.core.windows.net/public/MARS-S2L/train_logs_revision/CH4Net_20250329/config_experiment.json)

In order to run inference with the MARS-S2L trained model see notebook [`examples/run_inference.ipynb`](https://github.com/gonzmg88/marss2l/blob/main/notebooks/examples/run_inference.ipynb).

In order to re-run the evaluation in the test set with the trained model proposed in the paper do:

```bash
# Download model weights and config
mkdir -p train_logs_revision/MARSS2L_20250326
wget https://unepazeconomyadlsstorage.blob.core.windows.net/public/MARS-S2L/train_logs_revision/MARSS2L_20250326/best_epoch -P train_logs_revision/MARSS2L_20250326/
wget https://unepazeconomyadlsstorage.blob.core.windows.net/public/MARS-S2L/train_logs_revision/MARSS2L_20250326/config_experiment.json -P train_logs_revision/MARSS2L_20250326/

# Run eval in the test split
python -m marss2l.eval_final --split test_2023 --output_dir train_logs_revision/MARSS2L_20250326 --device cpu --log_images --num_workers 0
```

## Dataset

The MARS-S2L database and all pre-trained models are released under a [Creative Commons Non-Commercial Share-Alike licence](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt). 

Currently the dataset is publicly available in an Azure container and in [Hugging Face](https://huggingface.co/datasets/UNEP-IMEO/MARS-S2L).

The following notebooks show how to load and plot images using the `torch.Dataset` directly from the Azure container: [`notebooks/examples/plot_images_dataset_train.ipynb`](https://github.com/gonzmg88/marss2l/blob/main/notebooks/examples/plot_images_dataset_train.ipynb) and [`notebooks/examples/plot_images_dataset_test.ipynb`](https://github.com/gonzmg88/marss2l/blob/main/notebooks/examples/plot_images_dataset_test.ipynb).

### Dataset structure

The file [`validated_images_all.csv`](https://unepazeconomyadlsstorage.blob.core.windows.net/public/MARS-S2L/dataset_20250609/validated_images_all.csv) contains the metadata of all the items in the dataset. Each item comprises a Sentinel-2 or Landsat image of 200x200 pixels
together with its concatenated background image, the plume mask, the cloud mask and the methane enhancement image in ppb. In addition, each item
includes several metadata which is described in the *Images metadata description* section.

The file `validated_images_plumes.csv` contains all validated plumes in the dataset. Notice that there are few images that contain more than one plume. 
the content of this file is described in the *Plumes metadata description* section.

### Images metadata description

Column description of file [`validated_images_all.csv`](https://unepazeconomyadlsstorage.blob.core.windows.net/public/MARS-S2L/dataset_20250609/validated_images_all.csv):

| Name      | Description      | Example | 
| ------------- | ------------- | -------- |
|s2path| Path to S2 or Landsat image concatenated with the background image. S2 L1C image with bands: B02, B03, B04, B08, B11, B12. Landsat TOA reflectance image with bands  B02, B03, B04, B05, B06, B07 | model_data_all/b82be820-fe9c-411a-af20-82cf162760a7_ea782bfb-21b6-46c5-a6ed-d3b8ef8815a0_s2.tif |
|plumepath| Path to binary representation of the mask. | model_data_all/b82be820-fe9c-411a-af20-82cf162760a7_ea782bfb-21b6-46c5-a6ed-d3b8ef8815a0_label.tif |
|cloudmaskpath| Path to cloud mask generated with CloudSEN12 with interpretation `{0 : "clear", 1: "Thick cloud", 2: "Thin cloud", 3: "Cloud shadow"}` | model_data_all/b82be820-fe9c-411a-af20-82cf162760a7_ea782bfb-21b6-46c5-a6ed-d3b8ef8815a0_cloudmask.tif |
|ch4path| Path to CH4 enhancement image in ppb. | model_data_all/b82be820-fe9c-411a-af20-82cf162760a7_ea782bfb-21b6-46c5-a6ed-d3b8ef8815a0_ch4.tif |
|wind_u| U component of the wind at 10m | -1.87 |
|wind_v| V component of the wind at 10m | 3.76 |
|wind_source| Source where the wind data is downloaded from. | ECMWF/ERA5_LAND/HOURLY |
|vza| Viewing zenith angle. | 5.40 |
|sza| Solar zenith angle. | 16.32 |
|percentage_clear| Percentage of pixels clear according to CloudSEN12 mask. | 100.0 |
|tile| Name of the S2 or Landsat product | S2A_MSIL1C_20240613T100031_N0510_R122_T32RNS_20240613T134114 or LC08_L1TP_192040_20190601_20200828_02_T1 |
|isplume| Whether there is a methane plume on the image or not | True |
|ch4_fluxrate| Flux rate in kg/h if there is a plume in the tile. | 7756.50 |
|ch4_fluxrate_std| Standard deviation of the fluxrate. | 2941.19 |
|satellite| Name of the satellite | S2A, S2B, LC08 or LC09 |
|tile_date| Date of acquisition | 2024-06-13 10:00:31+00:00 |
|notified| If the image contains a plume, whether the plume was notified or not | False |
|id_location| UUID of the site | b82be820-fe9c-411a-af20-82cf162760a7 |
|last_update| Last time this registry was modified | 2025-05-20 10:00:31+00:00 |
|location_name| Code of the location. | A_10 |
|country| Country of the location | Algeria |
|lon| Longitude of the center of the location | 9.783133 |
|lat| Latitude of the center of the location | 28.087894 |
|offshore| Whether the location is offshore or onshore. | False |
|sector| Sector of the observation | Oil and Gas, Coal, Waste or Unknown |
|observability| Given observability to the image. Clear and cloudy is computed automatically with the cloudmask (50% threshold). Additionally the analyst can change this category (e.g. if the cloudmask is not correct or if there're artifacts in the scene such as snow, smoke, shadows...). | clear, cloudy, bad_retrieval, smoke, shadow, snow, out_of_swath |
|background_image_tile| Name of the S2 or Landsat product used as background. | S2A_MSIL1C_20240603T100031_N0510_R122_T32RNS_20240603T134002 |
|crs| Coordinate reference system of the image | EPSG:32632 |
|transform_a| Geotransform of the image | 10.0 |
|transform_b| Geotransform of the image | 0.0 |
|transform_c| Geotransform of the image | 575940.0 |
|transform_d| Geotransform of the image | 0.0 |
|transform_e| Geotransform of the image | -10.0 |
|transform_f| Geotransform of the image | 3108180.0 |
|width| Width of the image | 200 |
|height| Height of the image | 200 |
|window_row_off| Top left row of the location of the plume within the image (used for the simulation). | 0.0 |
|window_col_off| Top left column of the location of the plume within the image (used for the simulation). | 62.0 |
|window_height| Height in pixels of the plume within the image (used for the simulation). | 101.0 |
|window_width| Width in pixels of the plume within the image (used for the simulation). | 46.0 |
|id_loc_image| Id of the image | ea782bfb-21b6-46c5-a6ed-d3b8ef8815a0 |
|plume| Vector WKT representation of the plume. | MULTIPOLYGON (((9.781389 28.090856, 9.781646 28.090288, 9.781861 28.089796, 9.781818 28.089134, 9.781989 28.088622, 9.782268 28.088187, 9.782633 28.087903, 9.783105 28.087865, 9.783362 28.088244, 9.783083 28.088774, 9.78304 28.089436, 9.783148 28.090061, 9.783384 28.090705, 9.783663 28.091424, 9.783856 28.092522, 9.783642 28.093299, 9.783234 28.093905, 9.783106 28.094794, 9.783041 28.095571, 9.782312 28.096195, 9.78154 28.096422, 9.780896 28.096934, 9.779888 28.097104, 9.779352 28.096915, 9.779759 28.096063, 9.780596 28.09523, 9.781132 28.094472, 9.780896 28.093867, 9.780896 28.09309, 9.781154 28.092314, 9.781239 28.091784, 9.781154 28.091292, 9.781389 28.090856))) |
|footprint| Vector WKT representation of the footprint of the image | POLYGON ((9.772934575210368 28.09701036650939, 9.7728039936767 28.07877712456095, 9.793361656935707 28.07865982254285, 9.7934957112434 28.09689297500473, 9.772934575210368 28.09701036650939)) |
|split_name| Name of the split the item belongs to. One of 'train_2023', 'test_2023',  'val_2023' or 'Not Used'  | 'train_2023' |
|case_study | Case study area of the same. | Turkmenistan |

### Plumes metadata description

Column description of file [`validated_images_plumes.csv`](https://unepazeconomyadlsstorage.blob.core.windows.net/public/MARS-S2L/dataset_20250609/validated_images_plumes.csv).

| Name      | Description      | Example | 
| ------------- | ------------- | -------- |
|id_plume | UUID of the plume | 652fd72a-4a24-4938-8646-84309f899914|
|id_loc_image | UUID of its corresponding image | 37ecaefe-619e-4dc7-90f1-d57bd48dbf0f|
|tile| Name of the S2 or Landsat product | S2A_MSIL1C_20240613T100031_N0510_R122_T32RNS_20240613T134114 or LC08_L1TP_192040_20190601_20200828_02_T1 |
|tile_date| Date of acquisition | 2024-06-13 10:00:31+00:00 |
|satellite| Name of the satellite | S2A, S2B, LC08 or LC09 |
|lat | Latitude origin of the plume | 28.088329877970512|
|lon | Longitude origin of the plume | 9.78350881586494|
|ch4_fluxrate | Estimated flux rate of the plume in kg/h | 9639.58562343052|
|ch4_fluxrate_std | Estimated std of the flux rate of the plume in kg/h | 3964.424919968837|
|geometry | Plume shape | MULTIPOLYGON (((9.782459 28.090423, 9.782716 28.089211, 9.782802 28.088378, 9.783746 28.088151, 9.78396 28.089249, 9.784217 28.090953, 9.784475 28.092316, 9.784432 28.093528, 9.783788 28.095156, 9.782802 28.095383, 9.78293 28.09402, 9.782716 28.093339, 9.782201 28.092203, 9.782416 28.09118, 9.782459 28.090423)))|
|last_update | Last time this registry was updated | 2025-05-19 05:30:16.151729+00:00|
|window_row_off | Pixel upper left row coordinate within the image where the plume is located | 15|
|window_col_off | Pixel upper left column coordinate within the image where the plume is located | 88|
|window_height | Number of columns from the upper left coordinate | 82|
|window_width | Number of rows from the upper left coordinate | 25|

### Licence

The MARS-S2L database and all pre-trained models are released under a [Creative Commons Non-Commercial Share-Alike licence](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt). 

The code in this package is published under a [GNU Lesser GPL v3 licence](https://www.gnu.org/licenses/lgpl-3.0.en.html) <img src="https://www.gnu.org/graphics/lgplv3-88x31.png" alt="licence" width="80">.

### Cite
If you find this work useful, please cite:

```bibtex
@article{allen_ai_2024,
    title = {{AI} for methane emission detection and mitigation},
    author = {Allen, Anna and Mateo-Garcia, Gonzalo and Irakulis-Loitxate, Itziar and Montesinos San Martin, Manuel and Watine, Marc and Fernandez-Poblaciones, Pablo and Requeima, James and Gorroño, Javier and Randles, Cynthia and Turner, Richard E. and Caltagirone, Manfredi and Cifarelli, Claudio},
    year={2024},
    month = aug,
    eprint={2408.04745},
    archivePrefix={arXiv}}
}
```

