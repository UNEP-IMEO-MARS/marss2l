import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import math
from datetime import datetime, timezone
import uuid

from georeader.readers import S2_SAFE_reader
from georeader import plot

from typing import Tuple, Dict, Optional, List, Union, Any
from . import mbmp_torch


from marss2l.utils import get_remote_filesystem

from marss2l.mars_sentinel2 import plumesimulation
from marss2l.mars_sentinel2 import wind
from marss2l.mars_sentinel2 import mixing_ratio_methane
from marss2l.mars_sentinel2 import transmittance_to_ch4

import matplotlib.pyplot as plt
import torch
import logging
import fsspec

from shapely.geometry import Polygon, MultiPolygon
from shapely import wkt
from shapely import make_valid

from georeader.geotensor import GeoTensor
from georeader import rasterize
from rasterio import Affine

from threading import Lock

RELATION_CHANNELS_S2_L89 = {
    "B01": "B01",
    "B02": "B02",
    "B03": "B03",
    "B04": "B04",
    "B08": "B05", # OR B08A
    "B10": "B09",
    "B11": "B06",
    "B12": "B07"
}

def bands_in_l89(channels_query_s2:List[str]) -> List[str]:
    """ This is basically all the channels in the RELATION_CHANNELS_S2_L89 but to make sure they're consistently ordered """
    return [RELATION_CHANNELS_S2_L89[c] for c in S2_SAFE_reader.normalize_band_names(channels_query_s2) if c in RELATION_CHANNELS_S2_L89]

RELATION_CHANNELS_L89_S2 = {v: k for k, v in RELATION_CHANNELS_S2_L89.items()}


SPLITS ={
    "ablation": ["train", "val", "test"],
    "all": ["train_2023", "val_2023", "test_2023"],
    "control_releases": ["control_releases_train", "control_releases_val", "control_releases_test"]
}

ALL_DATE_CUT = '2023-12' # test_2023 will be >= ALL_DATE_CUT

LOCATIONS_CONTROL_RELEASES = ["Standford_controlled_releases", 
                              "Standford_controlled_releases_2021", 
                              "CNT_REL_P0_2024"]

LOCS_OFFSHORE_ABLATION = ["MEX_005", "MEX_003", "MYS_1"]

LOCS_TRAINING_ABLATION = ['L_11', 'T_26', 'A_30', 'GA_054', 'Iq_14', 
                          'K_S_1', 'T_12', 'A_6', 'L_19', 'K_S_2', 'A_38', 
                          'A_7', 'A_13', 'A_20', 'A_14', 'T_23', 'L_3', 
                          'T_7', 'L_2', 'A_15', 'A_33', 'A_41', 'T_25',
                          'A_24', 'A_5', 'T_27', 'A_22', 'GA_001', 
                          'K_2', 'A_3', 'A_29', 'A_36', 'L_16', 'T_4', 
                          'A_17', 'T_6', 'T_11', 'A_18', 'K_3', 'Y_1', 
                          'A_11', 'T_13', 'A_12', 'T_17', 'L_14', 'A_28', 
                          'A_35', 'K_6', 'T_9', 'A_32', 'T_22', 'T_20', 'A_27', 
                          'A_40', 'T_10', 'A_34', 'T_1', 'K_8', 'T_0', 'T_14', 
                          'A_25', 'A_8', 'L_18', 'A_26', 'A_10', 'Iq_4', 'A_9', 
                          'K_9', 'T_19', 'T_5', 'L_15', 'L_4', 'L_17', 'T_24', 
                          'Iq_12', 'Iq_1', 'A_31', 'I_S_1', 'L_1', 'A_37', 'A_43', 
                          'K_7', 'Iq_5', 'T_28', 'A_16', 'A_4']

WINDOW_SIZE_TRAINING = 192
WINDOW_SIZE_DATA = 200
MIN_FLUXRATE_SIM = 3_500
MIN_FLUXRATE_SIM_OFFSHORE = 20_000

# Define constants for defaults
DEFAULT_SPLIT = "all"
DEFAULT_WINDOW_SIZE_TRAINING = 192
DEFAULT_DO_SIMULATION = True
DEFAULT_FILM_TRAIN_ZERO_ID = True
DEFAULT_CAT_MBMP = True
DEFAULT_BANDS_L8 = True
DEFAULT_WIND = True
DEFAULT_NORM_WIND = True
DEFAULT_CLOUD_MASK = True
DEFAULT_LOAD_CH4 = False
DEFAULT_MULTIPASS = True
DEFAULT_STRATIFY_BY_LOCATION = True
DEFAULT_ONLY_ONSHORE = False
DEFAULT_ONLY_OFFSHORE = False


BANDS_S2_IN_L8 = ["B02", "B03", "B04", "B08", "B11", "B12"]
N_POS_SIMULATE = 5
MIN_SAMPLES_LOCATION_TRAIN = 30
MIN_SAMPLES_NEGATIVE_TRAIN = 15


CSV_PATH_DEFAULT = "az://public/MARS-S2L/dataset_20250609/validated_images_all.csv"
CSV_PLUME_PATH_DEFAULT = "az://public/MARS-S2L/dataset_20250609/validated_images_plumes.csv"

NSAMPLES_PER_EPOCH_DEFAULT = 2048*32
INTERVALS_FLUXRATE = np.array([-0.001, 0, 500, 1000, 1500, 2000,2500, 3000, 4000,5000,
                               6000, 7000,8000,9000,10_000, 15_000, 20_000,  999_000])/1_000


def get_n_channels_out(multipass:bool, wind:bool, bands_l8:bool, 
                       cloud_mask:bool, cat_mbmp:bool) -> int:
    if bands_l8:
        nbands = len(BANDS_S2_IN_L8)
    else:
        nbands = len(S2_SAFE_reader.BANDS_S2_L1C)
    if multipass:
        in_channels = 2 * nbands
    else:
        in_channels = nbands

    if wind:
        in_channels+=2

    if cloud_mask:
        in_channels+=1
    
    if cat_mbmp:
        in_channels+=1
    
    return in_channels

def randint_safe(low:int, high:int) -> int:
    if low >= high:
        return low
    else:
        return np.random.randint(low, high)
    
def sample_window(window_plume_row_off:int, window_plume_col_off:int, 
                  window_plume_width:int, window_plume_height:int,
                  window_size_training:int=WINDOW_SIZE_TRAINING,
                  window_size_data:int=WINDOW_SIZE_DATA,
                  add_jitter:bool=True) -> Tuple[int, int]:
    """
    Sample a window of size `window_size_training` surrounding the plume center. If add_jitter is True, the center
    of the window will be jittered by a random value between -window_size_training//4 and window_size_training//4

    Args:
        window_plume_row_off (int): row offset of the plume (between 0 and window_size_data - window_plume_height)
        window_plume_col_off (int): col offset of the plume (between 0 and window_size_data - window_plume_width)
        window_plume_width (int): width of the plume
        window_plume_height (int): height of the plume
        add_jitter (bool, optional): If True, the center of the window will be jittered. Defaults to True.

    Returns:
        Tuple[int, int]: row_off, col_off to sample the window
    """
    window_col_center_jitter = window_plume_col_off + window_plume_width//2
    window_row_center_jitter = window_plume_row_off + window_plume_height//2

    if add_jitter:
        window_col_center_jitter += np.random.randint(-window_plume_width//4, window_plume_width//4)
        window_row_center_jitter += np.random.randint(-window_plume_height//4, window_plume_height//4)

    window_row_off_jitter = window_row_center_jitter - window_size_training//2
    window_col_off_jitter = window_col_center_jitter - window_size_training//2

    # Make sure the window is inside the image
    window_row_off_jitter = min(max(0, window_row_off_jitter), window_size_data-window_size_training)
    window_col_off_jitter = min(max(0, window_col_off_jitter), window_size_data-window_size_training)


    return window_row_off_jitter, window_col_off_jitter



PolygonorMultiPolygonOrStr = Union[Polygon, MultiPolygon, str]
def percentage_overlap(plume:PolygonorMultiPolygonOrStr, 
                       footprint:PolygonorMultiPolygonOrStr) -> float:
    """
    This function computes the percentage of overlap between the plume and the footprint.

    It is used to discard samples with low overlap between the plume and the footprint.

    Args:
        plume (PolygonorMultiPolygonOrStr): Plume geometry
        footprint (PolygonorMultiPolygonOrStr): Footprint geometry

    Returns:
        float: percentage of overlap between the plume and the footprint
            if the plume is empty, it returns -1
    """
    
    if isinstance(plume, str):
        plume = make_valid(wkt.loads(plume))
    if isinstance(footprint, str):
        footprint =make_valid(wkt.loads(footprint))

    if plume.is_empty:
        return -1

    if not plume.intersects(footprint):
        return 0

    if footprint.contains(plume):
        return 1
        
    intersection = plume.intersection(footprint)

    return intersection.area / plume.area

FILE_SYSTEM = fsspec.filesystem("file")

def _get_fs(fs:Optional[fsspec.AbstractFileSystem], path:str) -> fsspec.AbstractFileSystem:
    if fs is None and path.startswith("az://"):
        fs = get_remote_filesystem()
    elif not path.startswith("az://"):
        fs = FILE_SYSTEM
    return fs


COUNTRIES_CASE_STUDIES = ['United States of America', 
                          'Turkmenistan',
                          'Algeria',  
                          'Libya', 
                          'Iran (Islamic Republic of)', 
                          'Syrian Arab Republic',
                          "Egypt", 
                          "Iraq",
                          'Venezuela',
                          'Offshore']
COUNTRIES_ARABIAN_PENINSULA = ["Saudi Arabia", 'Kuwait', 'Bahrain', 'Oman', "Qatar", "Yemen", "United Arab Emirates"]
UZB_AND_KAZAKH = ['Kazakhstan', 'Uzbekistan']

def _set_case_study(country:str) -> str:
    if country in COUNTRIES_CASE_STUDIES:
        return country
    if country == "Syria":
        return "Syrian Arab Republic"
    if country == "United States":
        return "United States of America"
    if country == "Iran":
        return "Iran (Islamic Republic of)"
    if country in COUNTRIES_ARABIAN_PENINSULA:
        return "Arabian peninsula"
    if country in UZB_AND_KAZAKH:
        return "Uzbekistan & Kazakhstan"
    return "Rest"

ORDER_CASE_STUDIES = ['United States of America', 
                      'Turkmenistan',
                      'Algeria',  
                      'Libya', 
                      "Arabian peninsula",
                      "Uzbekistan & Kazakhstan",
                      'Iran (Islamic Republic of)', 
                      'Syrian Arab Republic',
                      "Egypt", 
                      "Iraq",
                      'Venezuela',
                      'Offshore',
                      "Rest"]


def read_csv(csv_path:str=CSV_PATH_DEFAULT, fs:Optional[fsspec.AbstractFileSystem]=None,
             add_columns_for_analysis:bool=False, 
             split:Optional[str]=None,
             add_case_study:bool=False,
             add_loc_type:bool=False) -> pd.DataFrame:
    """
    Read the CSV file, process and add columns to the dataframe.

    Args:
        csv_path (str): Path to the CSV file
        split (str): Adds a column with split_name with the split that belongs that record. 
            If None, it does not add the column. One of SPLITS.keys() (e.g. "all")
        fs (fsspec.AbstractFileSystem, optional): Filesystem to use. Defaults to None.
        add_columns_for_analysis (bool, optional): Add columns for analysis. Defaults to False.
        add_case_study (bool, optional): Add column with case study. Defaults to False.
        add_loc_type (bool, optional): Add column with location type. Defaults to False.

    Returns:
        pd.DataFrame: 
    """
    fs = _get_fs(fs, csv_path)
    
    # Load CSV file
    assert fs.exists(csv_path), f"Path {csv_path} does not exist. Should contain the csv with the data."
    with fs.open(csv_path) as f:
        dataframe = pd.read_csv(f)
    
    # Derive columns
    # self.dataframe["tile_date"] = pd.to_datetime(self.dataframe["tile_date"])
    dataframe["percent_overlap"] = dataframe.apply(lambda row: percentage_overlap(row["plume"], row["footprint"]), 
                                                   axis=1)
    # Remove rows with percent_overlap > -1 and < 0.5
    high_overlap = (dataframe.percent_overlap == -1) | (dataframe.percent_overlap > 0.5)
    if not high_overlap.all():
        n_high_overlap = high_overlap.sum()
        n_total = dataframe.shape[0]
        dataframe = dataframe[high_overlap].copy()
        print(f"Removed {n_total - n_high_overlap} rows with percent_overlap < 0.5")

    dataframe["year"] = dataframe["tile_date"].apply(lambda x: int(x[:4]))
    dataframe["year_month"] = dataframe["tile_date"].apply(lambda x: x[:7])
    dataframe["tile_date"] = dataframe["tile_date"].apply(lambda x: datetime.fromisoformat(x))
    dataframe["year_month_day"] = dataframe["tile_date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    dataframe["wind_speed"] = dataframe.apply(lambda row: math.sqrt(row.wind_u**2 + row.wind_v**2),axis=1)
    dataframe["isplumeneg"] = ~dataframe.isplume

    # Set country to Offshore if offshore is True
    dataframe["country"] = dataframe.apply(lambda row: "Offshore" if row.offshore else row.country, axis=1)

    if add_columns_for_analysis:
        dataframe["id_loc_image"] = dataframe["id_loc_image"].map(lambda x: uuid.UUID(x))
        dataframe["last_update"] = dataframe["last_update"].apply(lambda x: datetime.fromisoformat(x))
        dataframe["date"] = dataframe["tile_date"].apply(lambda x: x.strftime("%Y-%m-%d"))
        dataframe["year_month"] = dataframe["tile_date"].apply(lambda x: datetime.strptime(x.strftime("%Y-%m-01"),"%Y-%m-%d"))
        dataframe["satellite_constellation"] = dataframe.satellite.apply(lambda x: "Sentinel-2" if x in ["S2A", "S2B"] else "Landsat")
        dataframe["year_quarter"] = dataframe["tile_date"].apply(lambda x: f"{x.year}-{(x.month-1)//3 + 1}Q")

        dataframe["ch4_fluxrate_th"] = dataframe["ch4_fluxrate"] / 1000
        dataframe["interval_ch4_fluxrate"] = pd.cut(dataframe.ch4_fluxrate_th, INTERVALS_FLUXRATE, # 30_000, 50_000,
                                                    include_lowest=True)

        dataframe["interval_ch4_fluxrate_str"] =  dataframe["interval_ch4_fluxrate"].apply(lambda x: str(x).replace("(-0.001001", "[0").replace(".0",""))
        dataframe["interval_ch4_fluxrate_str"] =  dataframe["interval_ch4_fluxrate_str"].apply(lambda x: x.replace("(20, 999]", ">20"))

    if split is not None:
        train_split, val_split, test_split = SPLITS[split]
        dataframe_data_traintest_indexed = dataframe.set_index("id_loc_image").copy()
        dataframe_data_traintest_indexed["split_name"] = "Not Used"
        for split_name, traintestval_split in zip([train_split, val_split, test_split],["train", "val", "test"]):
            df_splitted, _ = load_dataframe_split(dataframe_or_csv_path=dataframe, split=split_name, fs=fs,
                                                  load_plumes=False)
            ids_split = df_splitted.id_loc_image
            if not (dataframe_data_traintest_indexed.loc[ids_split, "split_name"] == "Not Used").all():
                splits_overlap = dataframe_data_traintest_indexed.loc[ids_split, "split_name"].unique()
                splits_overlap = splits_overlap[splits_overlap != "Not Used"].tolist()
                raise ValueError(f"BAD SPLITING!!! {split} {split_name} there is overlap with {splits_overlap}")
            
            dataframe_data_traintest_indexed.loc[ids_split, "split_name"] = split_name

            dataframe = dataframe_data_traintest_indexed.reset_index()
        
        if add_loc_type:
            # Add column with location type
            summaries_by_loc = dataframe.groupby(["split_name","location_name"])["isplume"].agg(["count", "sum"]).rename(columns={"count":"nimages", "sum":"nplumes"}).reset_index()
            summaries_by_loc["loc_type"] = (summaries_by_loc["nimages"] >= MIN_SAMPLES_LOCATION_TRAIN) & (summaries_by_loc["nplumes"] >= N_POS_SIMULATE) & ((summaries_by_loc["nimages"] - summaries_by_loc["nplumes"]) >= MIN_SAMPLES_NEGATIVE_TRAIN)
            summaries_by_loc_train = summaries_by_loc[summaries_by_loc.split_name == train_split]
            locs_film = set(summaries_by_loc_train.loc[summaries_by_loc_train["loc_type"], "location_name"].values)
            locs_train = set(summaries_by_loc_train["location_name"].values)
            dataframe["loc_type"] = dataframe.location_name.apply(lambda x: "FiLM" if x in locs_film else "few samples" if x in locs_train else "no samples")
        
    if add_case_study:
        # from marss2l import locations_case_studies
        # dataframe["case_study"] = dataframe["location_name"].apply(lambda x: locations_case_studies.REV_CASE_STUDIES.get(x, "None"))
        dataframe["case_study"] = dataframe["country"].apply(_set_case_study)
        
    return dataframe


COLUMNS_MERGE_PLUMES = ["id_loc_image","location_name",
                        "ch4path","plumepath", "crs", 
                        "width", "height", "transform_a", 
                        "transform_b", "transform_c", 
                        "transform_d", "transform_e", 
                        "transform_f"]

def read_csv_plumes(csv_path:str, dataframe_images:Optional[pd.DataFrame]=None,
                    fs:Optional[fsspec.AbstractFileSystem]=None) -> pd.DataFrame:
    """
    Read the CSV file with the plumes.

    Args:
        csv_path (str): Path to the CSV
        dataframe_images (pd.DataFrame): Dataframe with the images
        fs (fsspec.AbstractFileSystem, optional): Filesystem to use. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with the plumes
    """
    fs = _get_fs(fs, csv_path)
    with fs.open(csv_path) as f:
        dataframe_plumes = pd.read_csv(f)
    
    # Add aux columns
    dataframe_plumes["year"] = dataframe_plumes["tile_date"].apply(lambda x: int(x[:4]))
    dataframe_plumes["year_month"] = dataframe_plumes["tile_date"].apply(lambda x: x[:7])
    dataframe_plumes["tile_date"] = dataframe_plumes["tile_date"].apply(lambda x: datetime.fromisoformat(x))
    dataframe_plumes["wind_speed"] = dataframe_plumes.apply(lambda row: math.sqrt(row.wind_u**2 + row.wind_v**2),axis=1)
    
    if dataframe_images is not None:
        dataframe_plumes = pd.merge(dataframe_plumes, 
                                    dataframe_images[COLUMNS_MERGE_PLUMES], 
                                    on="id_loc_image")
    
    return dataframe_plumes 


def split_control_releases(dataframe:pd.DataFrame, split:str, logger:Optional[logging.Logger]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into control releases and non-control releases.

    For location 'Standford_controlled_releases' the split is:
    - train: 2020-01-01 to 2022-01-01
    - val: 2022-01-01 to 2022-10-01
    - test: 2022-10-10 to 2022-11-28
    
    For location 'Standford_controlled_releases_2021' the split is:
    - train: 2020-01-01 to 2021-10-01
    - val: 2022-01-01 to 2022-10-01
    - test: 2021-10-19 to 2021-11-03


    """
    control_releases_images = dataframe.location_name.isin(LOCATIONS_CONTROL_RELEASES)
    if split=="control_releases_train":
        split_data = control_releases_images &\
                     ((dataframe.location_name == 'Standford_controlled_releases') & (dataframe.tile_date < datetime(2022, 1, 1, tzinfo=timezone.utc)) & (dataframe.tile_date >= datetime(2020, 1, 1, tzinfo=timezone.utc)) |\
                       (dataframe.location_name == 'Standford_controlled_releases_2021') & (dataframe.tile_date < datetime(2021, 10, 1, tzinfo=timezone.utc)) & (dataframe.tile_date >= datetime(2020, 1, 1, tzinfo=timezone.utc)))
        plume_dataframe = dataframe[dataframe["isplume"] & ~control_releases_images].copy()
    elif split=="control_releases_val":
        split_data = control_releases_images &\
                     (dataframe.tile_date < datetime(2022, 10, 1, tzinfo=timezone.utc)) & (dataframe.tile_date >= datetime(2022, 1, 1, tzinfo=timezone.utc))
        plume_dataframe = None
    elif split=="control_releases_test":
        split_data = control_releases_images &\
                     ((dataframe.location_name == 'Standford_controlled_releases') & (dataframe.tile_date < datetime(2022, 11, 28, 23, 59, 59, tzinfo=timezone.utc)) & (dataframe.tile_date >= datetime(2022, 10, 10, tzinfo=timezone.utc)) |\
                       (dataframe.location_name == 'Standford_controlled_releases_2021') & (dataframe.tile_date < datetime(2021, 11, 3, 23, 59, 59, tzinfo=timezone.utc)) & (dataframe.tile_date >= datetime(2021, 10, 19, tzinfo=timezone.utc)))
        plume_dataframe = None
        
    else:
        raise ValueError(f"Unknown split {split}. Expected 'control_releases_train', 'control_releases_val', 'control_releases_test'")
    
    dataframe = dataframe.loc[split_data].copy()

    return dataframe, plume_dataframe   


def load_dataframe_split(split:str, 
                         dataframe_or_csv_path:Union[str,pd.DataFrame]=CSV_PATH_DEFAULT,
                         dataframe_or_csv_path_plumes:Optional[Union[str,pd.DataFrame]]=None,
                         fs:Optional[fsspec.AbstractFileSystem]=None, 
                         logger:Optional[logging.Logger]=None,
                         load_plumes:bool=True,
                         all_locs:Optional[List[str]]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the dataframe from the csv file and split it according to the split.

    """
    if isinstance(dataframe_or_csv_path, str):
        csv_path = dataframe_or_csv_path
        dataframe_image = read_csv(csv_path, fs)
        if dataframe_or_csv_path_plumes is None:
            dataframe_or_csv_path_plumes = dataframe_or_csv_path.replace("validated_images_all.csv", "validated_images_plumes.csv")
            fs = _get_fs(fs, dataframe_or_csv_path_plumes)
            assert fs.exists(dataframe_or_csv_path_plumes), f"Path {dataframe_or_csv_path_plumes} does not exist. Should contain the csv with the plumes if not provided."
    else:
        dataframe_image = dataframe_or_csv_path.copy()
        assert not load_plumes or dataframe_or_csv_path_plumes is not None, "csv_path_plumes should be provided if dataframe_or_csv_path is a DataFrame"
    
    # Load plumes dataframe
    if load_plumes:
        assert dataframe_or_csv_path_plumes is not None, "csv_path_plumes should be provided"
        if dataframe_or_csv_path_plumes is not None:
            if isinstance(dataframe_or_csv_path_plumes, str):
                dataframe_plumes = read_csv_plumes(dataframe_or_csv_path_plumes, dataframe_image, fs)         
            else:
                dataframe_plumes = dataframe_or_csv_path_plumes.copy()
                dataframe_plumes = pd.merge(dataframe_plumes, 
                                            dataframe_image[COLUMNS_MERGE_PLUMES], 
                                            on="id_loc_image")

    # Make sure to exclude Control release locations
    locs_control_releases_serie = dataframe_image.location_name.isin(LOCATIONS_CONTROL_RELEASES)
    if split.startswith("control_releases"):
        if not locs_control_releases_serie.any():
            raise ValueError(f"Locations {LOCATIONS_CONTROL_RELEASES} not found in the dataset")
        return split_control_releases(dataframe_image, split, logger)
    else:
        if locs_control_releases_serie.any():
            dataframe_image = dataframe_image.loc[~locs_control_releases_serie].copy()
            # logger.info(f"Excluded locations {LOCATIONS_CONTROL_RELEASES}")

    # Keep only data from 2018 on
    data_pre_2018 = dataframe_image["year"] < 2018
    if data_pre_2018.any():
        # logger.info(f"Discarding data from years before 2018. There are {data_pre_2018.sum()} samples before 2018")
        dataframe_image = dataframe_image[~data_pre_2018].copy()
    
    # Split the data
    if split=="train":
        split_data = (dataframe_image["year"] < 2020)  & dataframe_image.location_name.isin(LOCS_TRAINING_ABLATION + LOCS_OFFSHORE_ABLATION)
    elif split=="test":
        split_data = (dataframe_image["year"] == 2020) & dataframe_image.location_name.isin(LOCS_TRAINING_ABLATION + LOCS_OFFSHORE_ABLATION)
    elif (split=="val") or (split=="val_2023") or (split=="val_pre_2023"):
        split_data = (dataframe_image["year"] == 2021) & dataframe_image.location_name.isin(LOCS_TRAINING_ABLATION + LOCS_OFFSHORE_ABLATION)
    elif split=="train_2023":
        split_data = (dataframe_image["year_month"] < ALL_DATE_CUT) & (dataframe_image["year"] != 2021)
    elif split=="test_2023":
        split_data = (dataframe_image["year_month"] > ALL_DATE_CUT)
    elif split=="train_pre_2023":
        split_data = (dataframe_image["year"] < 2023) & (dataframe_image["year"] != 2021)
    elif split in ["test_pre_2023"]:  # "post_2022", "post_2022_tune", "post_2022_test",  "post_2022_val"]:
        split_data = dataframe_image["year"] > 2022
    elif split == "no split":
        # Split data is all true
        split_data = dataframe_image["year"] > 0
    else:
        raise ValueError(f"Unknown split {split}. Expected 'train', 'test', 'val', 'train_2023', 'test_2023',  'val_2023'")
        #  'post_2022', 'post_2022_tune', 'post_2022_test', 'post_2022_val'

    # Set self.plume_dataframe to simulate plumes
    # if split=="post_2022_tune":
    #     # plume_dataframe = dataframe_image[(dataframe_image["year"] < 2023) & dataframe_image["isplume"]].copy()
    #     dataframe_image = dataframe_image.loc[split_data].copy()
    #     train_data_pre_2022 = dataframe_image.loc[~split_data].copy()
    
    dataframe_image = dataframe_image.loc[split_data].copy()
    
    # Set self.all_locs and keep only data from all_locs
    if all_locs is not None:
        images_from_loc = dataframe_image.location_name.isin(all_locs)
        if not images_from_loc.any():
            raise ValueError(f"None of the locations in 'all_locs' where found in the dataset in split {split}")
        
        dataframe_image = dataframe_image.loc[images_from_loc].copy()
    
    # Split for tunning in post 2022 data
    # if split=="post_2022_tune":
    #     dfs = []
    #     for loc in dataframe_image["location_name"].unique().tolist():
    #         ldf = dataframe_image[dataframe_image.location_name == loc].sort_values("tile_date")
    #         dfs.append(ldf.reset_index().iloc[:2*len(ldf)//4])
    #     dfs.append(train_data_pre_2022)
    #     dataframe_image = pd.concat(dfs).reset_index()
    # elif split=="post_2022_val":
    #     dfs = []
    #     for loc in dataframe_image["location_name"].unique().tolist():
    #         ldf = dataframe_image[dataframe_image.location_name == loc].sort_values("tile_date")
    #         dfs.append(ldf.reset_index().iloc[len(ldf)//4:2*len(ldf)//4])
    #     dataframe_image = pd.concat(dfs).reset_index()
    # elif split=="post_2022_test":
    #     dfs = []
    #     for loc in dataframe_image["location_name"].unique().tolist():
    #         ldf = dataframe_image[dataframe_image.location_name == loc].sort_values("tile_date")
    #         dfs.append(ldf.reset_index().iloc[2*len(ldf)//4:])
    #     dataframe_image = pd.concat(dfs).reset_index()
    
    if not load_plumes:
        return dataframe_image, None
        
    split_data_plumes = dataframe_plumes.id_loc_image.isin(dataframe_image.id_loc_image)
    dataframe_plumes = dataframe_plumes.loc[split_data_plumes].copy()

    return dataframe_image, dataframe_plumes


class DatasetPlumes(Dataset):
    def __init__(self, 
                 device:torch.device=torch.device("cpu"),
                 mode:str="train",
                 multipass:bool=DEFAULT_MULTIPASS,
                 wind:bool=DEFAULT_WIND,
                 bands_l8:bool=DEFAULT_BANDS_L8,
                 cloud_mask:bool=DEFAULT_CLOUD_MASK,
                 cat_mbmp:bool=DEFAULT_CAT_MBMP,
                 norm_wind:bool=DEFAULT_NORM_WIND,
                 stratify_by_location:bool=DEFAULT_STRATIFY_BY_LOCATION,
                 dataframe_or_csv_path:Union[pd.DataFrame,str]=CSV_PATH_DEFAULT,
                 dataframe_or_csv_path_plumes:Optional[Union[pd.DataFrame,str]]=None,
                 split:str=SPLITS[DEFAULT_SPLIT][0],
                 do_simulation:bool=DEFAULT_DO_SIMULATION,
                 logger:Optional[logging.Logger]=None,
                 film_dict_mapping:Optional[Dict[str,int]]=None,
                 film_train_zero_id:bool=DEFAULT_FILM_TRAIN_ZERO_ID,
                 window_size_training:int=WINDOW_SIZE_TRAINING,
                 all_locs:Optional[List[str]]=None,
                 n_samples_per_epoch_train:int=NSAMPLES_PER_EPOCH_DEFAULT,
                 load_ch4:bool=DEFAULT_LOAD_CH4,
                 only_film_locs:bool=False,
                 only_onshore:bool=DEFAULT_ONLY_ONSHORE,
                 only_offshore:bool=DEFAULT_ONLY_OFFSHORE,
                 cache:bool=False,
                 mask_input_data:bool=False,
                 fs:Optional[fsspec.AbstractFileSystem]=None):
        """
        Initialize the DatasetPlumes class.

        Args:
            device (torch.device, optional): Device to use. Defaults to torch.device("cpu").
            mode (str, optional): Mode of the dataset, one of "train", "test" or "val". Defaults to "train".
            multipass (bool, optional): Use 2 images as input. Defaults to True.
            wind (bool, optional): Add 2 layers with the U and V wind components. Defaults to True.
            bands_l8 (bool, optional): Load only the bands of Sentinel-2 that are also in Landsat-8. Defaults to True.
                These are bands B02, B03, B04, B08, B11 and B12.
            cloud_mask (bool, optional): Add the cloud mask as an extra layer to the input. Defaults to True.
            cat_mbmp (bool, optional): Add the MBMP as an extra layer. Defaults to True.
            norm_wind (bool, optional): Normalize the wind component to be on a similar range as TOA reflectance,
                that is, divide by 8. Defaults to True.
            stratify_by_location (bool, optional): The get_item method will stratify the samples by location. Defaults to True.
            dataframe_or_csv_path (Union[pd.DataFrame, str], optional): Path to the CSV file or DataFrame containing the data. Defaults to CSV_PATH_DEFAULT.
                The dataframe or the CSV should have the following columns:
                - id_loc_image (str): Unique identifier for each image
                - location_name (str): Name of the location where the image was captured
                - tile (str): Tile identifier for the image
                - tile_date (str/datetime): Date of the image in ISO format
                - s2path (str): Path to the Sentinel-2/Landsat image data (numpy array or GeoTIFF)
                - plumepath (str): Path to the plume mask (for positive samples)
                - cloudmaskpath (str): Path to the cloud mask data
                - isplume (bool): Flag indicating if the image contains a plume
                - wind_u (float): U-component of wind velocity in m/s
                - wind_v (float): V-component of wind velocity in m/s
                - observability (str): Image observability conditions (e.g., "clear")
                - percentage_clear (float): Percentage of clear (non-cloudy) pixels in the image
                - offshore (bool): Flag indicating if the location is offshore
                - country (str): Country where the location is situated
                - plume (str): Plume geometry in WKT format
                - footprint (str): Footprint geometry in WKT format
                - window_row_off (int): Row offset of the plume window (for positive samples)
                - window_col_off (int): Column offset of the plume window (for positive samples)
                - window_width (int): Width of the plume window (for positive samples)
                - window_height (int): Height of the plume window (for positive samples)
                - ch4_fluxrate (float): Methane flux rate in kg/h
                - satellite (str): Satellite ID (e.g., "S2A", "S2B", "L8", "L9")
                - sza (float): Solar zenith angle in degrees
                - vza (float): Viewing zenith angle in degrees
                - transform_a (float): First element of georeferencing transformation matrix
                - transform_b (float): Second element of georeferencing transformation matrix
                - transform_c (float): Third element of georeferencing transformation matrix
                - transform_d (float): Fourth element of georeferencing transformation matrix
                - transform_e (float): Fifth element of georeferencing transformation matrix
                - transform_f (float): Sixth element of georeferencing transformation matrix
                - crs (str): Coordinate reference system
                - width (int): Width of the original image in pixels
                - height (int): Height of the original image in pixels
                - last_update (str/datetime): Timestamp of the last update to the record
                
                During processing, additional columns are derived:
                - year (int): Extracted from tile_date
                - year_month (str): Year and month (YYYY-MM) extracted from tile_date
                - year_month_day (str): Formatted date (YYYY-MM-DD)
                - wind_speed (float): Calculated from wind_u and wind_v components
                - isplumeneg (bool): Inverse of isplume
                - percent_overlap (float): Calculated overlap between plume and footprint geometries
            dataframe_or_csv_path_plumes (Optional[Union[pd.DataFrame, str]], optional): Path to the CSV file or DataFrame containing the plumes. Defaults to None.
                If None and dataframe_or_csv_path is a string, it will use the same path as dataframe_or_csv_path but replacing "validated_images_all.csv" with "validated_images_plumes.csv".
            split (str, optional): Split of the dataset, one of "train_2023", "test_2023", "val_2023", "train", "test", "val". Defaults to SPLITS[DEFAULT_SPLIT][0].
            do_simulation (bool, optional): Simulate plumes. Defaults to True.
            logger (Optional[logging.Logger], optional): Logger to use. Defaults to None.
            film_dict_mapping (Optional[Dict[str, int]], optional): Dictionary mapping location names to site_ids. Defaults to None.
            film_train_zero_id (bool, optional): If True, set site_ids to zero 50% of the time in train mode. Defaults to True.
            window_size_training (int, optional): Size of the training window. Defaults to WINDOW_SIZE_TRAINING.
            all_locs (Optional[List[str]], optional): List of locations to use. If None, it uses the locations in `LOCS_TRAINING_ABLATION` if split is "train", "test" or "val".
                Otherwise, it uses all the locations in the dataset. Defaults to None.
            n_samples_per_epoch_train (int, optional): Number of samples per epoch during training. Defaults to NSAMPLES_PER_EPOCH_DEFAULT.
            load_ch4 (bool, optional): Load the CH4 band. Defaults to False.
            only_film_locs (bool, optional): Load only locations that satisfy the FiLM condition. This is used to 
                fine-tune the FiLM parameters. Defaults to False.
            only_onshore (bool, optional): Load only onshore locations. Defaults to False.
            only_offshore (bool, optional): Load only offshore locations. Defaults to False.
            cache (bool, optional): Cache the images. Defaults to False.
            mask_input_data (bool, optional): Mask the input data with the cloud mask. That is, it will set the input data to zero where the cloud mask is not clear. Defaults to False.
            fs (Optional[fsspec.AbstractFileSystem], optional): Filesystem to use. Defaults to None.

        Raises:
            ValueError: If mode is not one of "train", "test" or "val".
            ValueError: If the path to the CSV file does not exist.
        
        Attributes:
            locs_few_samples (set): Set of locations with few samples.
            locs_few_neg (set): Set of locations with few negative samples.
            locs_few_pos (set): Set of locations with few positive samples.
            dataframe_few_samples_or_few_neg (pd.DataFrame): DataFrame with locations with few samples or few negative samples.
            ch42tr (TransmittanceCH4InterpolationFromDict): Object to map transmittance from the MBMP ratio to Delta CH4 concentrations.
            cache (dict): Cache for the images.
            dataframe (pd.DataFrame): DataFrame with the data.
            plume_dataframe (pd.DataFrame): DataFrame with the plumes for simulation. If do_simulation is False, it is None.
            all_locs (list): List of all locations in the dataset.
            simulator (PlumeSimulator): Plume simulator.
            bands (list): S2 bands loaded from the image.
            bands_out (list): List of bands output by the dataset.
            bands_expected_sentinel_2 (list): List of expected bands in Sentinel-2.
            bands_expected_landsat_8_s2naming (list): List of expected bands in Landsat-8 with Sentinel-2 naming.
            dataframe_id_loc_image_indexed (pd.DataFrame): DataFrame indexed by id_loc_image.
            total_pos (int): Total number of positive samples.
            total_neg (int): Total number of negative samples.
        """

        super().__init__()

        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        if fs is None:
            self.fs = fsspec.filesystem("file")
        else:
            self.fs = fs
        
        self.device = device
        self.mode = mode
        self.multipass = multipass
        self.wind = wind
        self.norm_wind = norm_wind
        self.stratify_by_location = stratify_by_location
        self.cloud_mask = cloud_mask
        self.mask_input_data = mask_input_data        
        self.cat_mbmp = cat_mbmp
        self.do_simulation = do_simulation
        self.film_train_zero_id = film_train_zero_id
        self.locs_few_samples = set()
        self.locs_few_neg = set()
        self.locs_few_pos = set()
        self.dataframe_few_samples_or_few_neg = None
        self.split = split
        self.window_size_training = window_size_training
        self.n_samples_per_epoch_train = n_samples_per_epoch_train
        self.load_ch4 = load_ch4
        self.load_common_bands_landsat_and_s2 = bands_l8
        self.only_film_locs = only_film_locs
        self.only_onshore = only_onshore
        self.only_offshore = only_offshore

        # assert only_onshore and only_offshore are not both True
        assert not (only_onshore and only_offshore), "only_onshore and only_offshore cannot be both True"

        assert self.window_size_training <= WINDOW_SIZE_DATA, f"window_size_training should be less than or equal to {WINDOW_SIZE_DATA} given {self.window_size_training}"

        # assert only_film_locs only in train mode
        assert not only_film_locs or mode == "train", "only_film_locs is only supported in train mode"

        # assert only_film_locs film_dict_mapping is provided
        assert not only_film_locs or film_dict_mapping is not None, "only_film_locs is True, but film_dict_mapping is not provided"

        # set simulation to false if only_film_locs (and raise warning)
        if only_film_locs:
            if self.do_simulation:
                self.logger.warning("Setting do_simulation to False because only_film_locs is True")
                self.do_simulation = False

        # Object to map transmittance from the MBMP ratio to \Delta CH4 concentrations
        self.ch42tr = transmittance_to_ch4.TransmittanceCH4InterpolationFromDict()

        # TODO move dataframe splitting out of the loader!
        self.dataframe, self.plume_dataframe = load_dataframe_split(dataframe_or_csv_path=dataframe_or_csv_path, 
                                                                    dataframe_or_csv_path_plumes=dataframe_or_csv_path_plumes,
                                                                    split=self.split, 
                                                                    fs=self.fs, logger=self.logger, 
                                                                    all_locs=all_locs,
                                                                    load_plumes=self.do_simulation)
        if self.plume_dataframe is not None:
            self.plume_dataframe = self.plume_dataframe.reset_index(drop=True) 
            self.plume_dataframe["int_index"] = self.plume_dataframe.index.values
        self.all_locs = self.dataframe["location_name"].unique().tolist()

        if self.do_simulation:
            assert self.plume_dataframe is not None and (self.plume_dataframe.shape[0] > 0), "No plumes found in the plume_dataframe and simulation is set to True"
            assert self.mode == "train", "Simulation is only supported in train mode"
            self.simulator = plumesimulation.PlumeSimulator()

        if self.load_common_bands_landsat_and_s2:
            self.bands = BANDS_S2_IN_L8
        else:
            raise NotImplementedError("Not supported training with all bands of S2")
            # self.bands = S2_SAFE_reader.BANDS_S2_L1C
            # logger.info("Using all bands DISCARDING images from Landsat-8/9!!")
            # self.dataframe = self.dataframe[self.dataframe.satellite.str.startswith("S2")].copy()
        
        # Subset if only_onshore or only_offshore
        if self.only_onshore:
            self.dataframe = self.dataframe[~self.dataframe.offshore].copy()
            self.all_locs = self.dataframe["location_name"].unique().tolist()
            self.logger.info(f"Keep only onshore locations in the dataset. There are {len(self.all_locs)} locations")
        elif self.only_offshore:
            self.dataframe = self.dataframe[self.dataframe.offshore].copy()
            self.all_locs = self.dataframe["location_name"].unique().tolist()
            self.logger.info(f"Keep only offshore locations in the dataset. There are {len(self.all_locs)} locations")
            
        # Set up if mode is train: set locs with few samples, few negatives and few positives
        if self.mode == "train":
            # Compute location with few samples
            self._compute_locations_few_samples()
        
        # Set up FiLM dict
        self.film_dict_mapping  = film_dict_mapping.copy() if film_dict_mapping is not None else None
        if self.film_dict_mapping is not None:
            missing_keys = [k for k in self.all_locs if k not in self.film_dict_mapping]
            if len(missing_keys) > 0:
                msg = f"Locations in all_locs not found in the film_dict_mapping. Keys not found: {missing_keys}"
                if mode != "train":
                    self.logger.error(msg)
                else:
                    raise ValueError(msg)

            # Set to zero_id sites in self.locs_few_samples, self.locs_few_neg, self.locs_few_pos
            if self.mode == "train":
                n_samples_zero_id = 0
                n_locs_film = 0
                for k in self.film_dict_mapping:
                    n_locs_film += 1
                    if (k in self.locs_few_samples) or (k in self.locs_few_neg) or (k in self.locs_few_pos) or (k not in self.all_locs):
                        self.film_dict_mapping[k] = 0            
                        n_samples_zero_id += 1
                self.logger.info(f"Set {n_samples_zero_id} locations to zero_id in the film_dict_mapping out of {n_locs_film} locations")

                # keep only locs with film_dict_mapping > 0 if only_film_locs
                if self.only_film_locs:
                    locs_film = set([k for k, v in self.film_dict_mapping.items() if v > 0])
                    self.dataframe = self.dataframe[self.dataframe.location_name.isin(locs_film)].copy()
                    self.all_locs = self.dataframe["location_name"].unique().tolist()
                    self.logger.info(f"Keep only locations with FiLM mapping in the dataset. There are {len(locs_film)} locations")
        

        # Reset the index and store a copy of the dataframe indexed by id_loc_image
        self.dataframe = self.dataframe.reset_index(drop=True)
        self.dataframe["int_index"] = self.dataframe.index.values
        self.dataframe_id_loc_image_indexed = self.dataframe.set_index("id_loc_image")
        
        # Figure out expected output bands (for sanity checks)
        bands_out = self.bands.copy()
        if self.multipass:
            bands_out.extend([f"{b}_bg" for b in self.bands])
        
        if self.wind:
            bands_out.extend(["U", "V"])
        
        if self.cloud_mask:
            bands_out.append("cloudmask")
        
        if self.cat_mbmp:
            bands_out.insert(0, "MBMP")
        
        self.bands_out = bands_out

        # Bands expected in the raw data
        # New data is always BANDS_S2_IN_L8!
        self.bands_expected_sentinel_2 = BANDS_S2_IN_L8 # S2_SAFE_reader.BANDS_S2_L1C
        self.bands_expected_landsat_8_s2naming = BANDS_S2_IN_L8 # [RELATION_CHANNELS_L89_S2[b] for b in bands_in_l89(self.bands_expected_sentinel_2)]

        self.log_info_data()

        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # Initialise caching
        self.cache = cache
        if self.cache:
            self.initialize_cache()

    @property
    def total_pos(self):
        return self.dataframe.isplume.sum()

    @property
    def total_neg(self):
        return self.dataframe.shape[0] - self.total_pos

    def log_info_data(self):
        # Compute total number of positive and negative samples and log stats of the data
        total_pos = self.total_pos
        total_neg = self.total_neg

        self.logger.info(f"{self.split} {self.mode} data from {len(self.all_locs)} locations")
        self.logger.info(f"{self.split} {self.mode} data between {min(self.dataframe['tile_date'])} to {max(self.dataframe['tile_date'])}")
        self.logger.info(f"{self.split} {self.mode} data size {self.dataframe.shape[0]} with {total_pos} plumes and {total_neg} images without plumes")

        # log different satellites
        self.logger.info(f"{self.split} {self.mode} Satellites in the dataset: {self.dataframe.satellite.unique()}")

        if self.do_simulation:
            self.logger.info(f"{self.split} {self.mode} Plumes dataset to simulate: {self.plume_dataframe.shape[0]}")
            # log dates and number of unique locations
            self.logger.info(f"{self.split} {self.mode} Plumes to simulate between {min(self.plume_dataframe['tile_date'])} to {max(self.plume_dataframe['tile_date'])}")
            self.logger.info(f"{self.split} {self.mode} Plumes to simulate from {len(self.plume_dataframe['location_name'].unique())} locations")
        
        # Log bands
        self.logger.info(f"{self.split} {self.mode} Bands output by the dataset: {self.bands_out}")

    def _compute_locations_few_samples(self):
        samples_per_location = self.dataframe.groupby("location_name").size()
        locs_few_samples_serie = samples_per_location[samples_per_location < MIN_SAMPLES_LOCATION_TRAIN].index
        if locs_few_samples_serie.shape[0] > 0:
            self.logger.info(f"{self.split} {self.mode}. There are {len(locs_few_samples_serie)} locations that have less than {MIN_SAMPLES_LOCATION_TRAIN} samples")
        
        # Compute location with no few samples
        negativesamples_in_location = self.dataframe.groupby("location_name").isplumeneg.sum()
        locs_few_neg_serie = negativesamples_in_location[negativesamples_in_location < MIN_SAMPLES_NEGATIVE_TRAIN].index
        if locs_few_neg_serie.shape[0] > 0:
            self.logger.info(f"{self.split} {self.mode}. There are {len(locs_few_neg_serie)} locations that have less than {MIN_SAMPLES_NEGATIVE_TRAIN} negative samples")
        
        # Compute locations with few positive samples
        positivesamples_in_location = self.dataframe.groupby("location_name").isplume.sum()
        locs_few_pos_serie = negativesamples_in_location[positivesamples_in_location < N_POS_SIMULATE].index
        if locs_few_pos_serie.shape[0] > 0:
            self.logger.info(f"{self.split} {self.mode}. There are {len(locs_few_pos_serie)} locations that have less than {N_POS_SIMULATE} positive samples")
        
        # Use dataframe_few_samples_or_few_neg to sample when the location has few samples or few negative samples
        self.dataframe["locs_few_samples"] = self.dataframe.location_name.isin(locs_few_samples_serie)
        self.dataframe["locs_few_neg"] = self.dataframe.location_name.isin(locs_few_neg_serie)
        self.dataframe["locs_few_pos"] = self.dataframe.location_name.isin(locs_few_pos_serie)
        self.dataframe_few_samples_or_few_neg = self.dataframe[self.dataframe.locs_few_samples | self.dataframe.locs_few_neg]
        
        # If there're still few samples or few negative samples in a single location, drop them from the dataframe
        if (self.dataframe_few_samples_or_few_neg.shape[0] < MIN_SAMPLES_LOCATION_TRAIN) or (self.dataframe_few_samples_or_few_neg.isplumeneg.sum() < MIN_SAMPLES_NEGATIVE_TRAIN):
            self.logger.info(f"Drop locations with less than {MIN_SAMPLES_LOCATION_TRAIN} samples or less than {MIN_SAMPLES_NEGATIVE_TRAIN} negative samples")
            self.logger.info(f"There was only {self.dataframe_few_samples_or_few_neg.shape[0]} samples and {self.dataframe_few_samples_or_few_neg.isplumeneg.sum()} negative samples in total across all these locations.")
            self.dataframe = self.dataframe.loc[~self.dataframe.locs_few_samples & ~self.dataframe.locs_few_neg].copy().reset_index(drop=True)
            self.dataframe["int_index"] = self.dataframe.index.values
            self.dataframe_id_loc_image_indexed = self.dataframe.set_index("id_loc_image")
            self.dataframe_few_samples_or_few_neg = None
            self.all_locs = self.dataframe["location_name"].unique().tolist()
        else:
            self.logger.info(f"There are {self.dataframe_few_samples_or_few_neg.shape[0]} samples and {self.dataframe_few_samples_or_few_neg.isplumeneg.sum()} negative samples in total in locations with few samples or few negative samples.")
        
        self.locs_few_pos = set(self.dataframe.location_name[self.dataframe.locs_few_pos].unique())
        self.locs_few_neg = set(self.dataframe.location_name[self.dataframe.locs_few_neg].unique())
        self.locs_few_samples = set(self.dataframe.location_name[self.dataframe.locs_few_samples].unique())

    def __len__(self):
        if self.mode=="train":
            return self.n_samples_per_epoch_train
        else:   
            return len(self.dataframe)
    
    def to_tensor(self, arr):
        return torch.from_numpy(arr.astype(np.float32)).to(self.device)
    
    def __getitem__(self, idx:int)-> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the processed data:
                - "y_context_ls0_0": torch.Tensor with the input data (C, H, W) values ToA reflectance * 2.
                    This tensor has concatenated the input image, the reference image, the mbmp, the cloud mask, 
                    and the wind components. Names of the bands in this tensor are given by self.bands_out.
                    Shape: (C, H, W), dtype: torch.float32
                - "y_target": torch.Tensor with the plume mask.
                    Shape: (H, W), dtype: torch.float32
                - "mbmp": torch.Tensor with the MBMP.
                    Shape: (H, W), dtype: torch.float32
                - "ch4": torch.Tensor with the CH4 values.
                    Shape: (1, H, W), dtype: torch.float32
                - "isplume": torch.Tensor with 1/0 if the image has a plume or not.
                    Shape: (), dtype: torch.int64
                - "simulated": torch.Tensor with 1/0 if the plume was simulated or not.
                    Shape: (), dtype: torch.int64
                - "location_name": str with the location name.
                - "tile": str with the tile name.
                - "id_loc_image": str with the id_loc_image.
                - "wind": torch.Tensor with the wind components U and V.
                    Shape: (2,), dtype: torch.float32
                - "site_ids": torch.Tensor with the site IDs for FiLM.
                    Shape: (), dtype: torch.int64

        Raises:
            ValueError: If the number of bands in the data is not the expected number of bands.
        """
        if self.mode == "train": # , "post_2022_tune"]
            return self.stratified_sample()

        item = self.dataframe.iloc[idx]

        # s2_data = self.to_tensor(self.load_image(item, "s2path")).permute(2, 0, 1)
        s2_data = self.load_image(item, "s2path")
        if item["isplume"]:
            # label = self.to_tensor(self.load_image(item, "plumepath"))
            label = self.load_image(item, "plumepath").astype(bool)
            window_row_off = int(round(item["window_row_off"]))
            window_col_off = int(round(item["window_col_off"]))
            window_width = int(round(item["window_width"]))
            window_height = int(round(item["window_height"]))
        else:
            # label = torch.zeros_like(s2_data[0,...], device=self.device)
            label = np.zeros(s2_data.shape[1:], dtype=bool)
            window_row_off, window_col_off, window_width, window_height = None, None, None, None
        
        isplume = int(item["isplume"]) 
        
        return self.postprocess_item(isplume=isplume, 
                                     window_row_off=window_row_off, window_col_off=window_col_off,
                                     window_width=window_width, window_height=window_height, 
                                     s2_data=s2_data, label=label,
                                     simulated=0, item=item)
    
    def initialize_cache(self):
        if not self.cache:
            self.logger.warning("Cache is not enabled. Call the Dataset with cache=True")
            return
        
        # Add a thread lock for cache writes
        self.cache_lock = Lock()  # <--- NEW
        
        self.cache_s2 = np.empty((len(self.dataframe), len(self.bands_expected_sentinel_2)*2, 
                                  WINDOW_SIZE_DATA, WINDOW_SIZE_DATA), dtype=np.uint16)
        self.cache_plumepath = np.empty((len(self.dataframe), 
                                            WINDOW_SIZE_DATA, WINDOW_SIZE_DATA), dtype=np.uint8)
        self.cache_cloudmask = np.empty((len(self.dataframe), 
                                            WINDOW_SIZE_DATA, WINDOW_SIZE_DATA), dtype=np.uint8)
        self.dict_cache_name = {"s2path": "cache_s2", 
                                "plumepath": "cache_plumepath", 
                                "cloudmaskpath": "cache_cloudmask"}
        # Set column in dataframe name and bool saying if the image is in cache
        for key in self.dict_cache_name:
            self.dataframe[f"{key}_in_cache"] = False

        if self.do_simulation:
            self.cache_lock_plumes = Lock()  # <--- NEW
            self.cache_ch4path_plumes = np.empty((len(self.plume_dataframe), 
                                                    WINDOW_SIZE_DATA, WINDOW_SIZE_DATA), dtype=np.float32)
            self.cache_plumepath_plumes = np.empty((len(self.plume_dataframe), 
                                                    WINDOW_SIZE_DATA, WINDOW_SIZE_DATA), dtype=np.uint8)
            self.dict_cache_name_plumes = {"ch4path": "cache_ch4path_plumes",
                                            "plumepath": "cache_plumepath_plumes"}
            self.plume_dataframe["in_cache"] = False
        
        if self.dataframe_few_samples_or_few_neg is not None:
            self.dataframe_few_samples_or_few_neg = self.dataframe[self.dataframe.locs_few_samples | self.dataframe.locs_few_neg]
        
    def load_image_method(self, item, key:str) -> NDArray:
        path:str = item[key]
        if path.endswith(".npy"):
            with _get_fs(self.fs, path).open(path, "rb") as f:
                values = np.load(f)
            if len(values.shape) == 3:
                values = np.transpose(values, (2, 0, 1))
                if values.shape[0] == 1:
                    return values[0]
            return values
        values = GeoTensor.load_file(path, 
                                     fs=self.fs if path.startswith("az://") else None).values
        if len(values.shape) == 3 and values.shape[0] == 1:
            return values[0]
        return values        
    
    def load_plume_simulation_method(self, plume_item) -> Tuple[NDArray, NDArray]:
        geo = load_image(plume_item, "ch4path", fs=self.fs)        
        geometry = wkt.loads(plume_item["geometry"])
        plume_mask = rasterize.rasterize_geometry_like(geometry, geo, 
                                                       crs_geometry="EPSG:4326", 
                                                       all_touched=True,
                                                       return_only_data=True)

        return geo.values, plume_mask

    def load_image(self, item, key:str) -> NDArray:
        if self.cache:
            key_array = self.dict_cache_name[key]
            int_index = item["int_index"]
            if not item[f"{key}_in_cache"]:
                value = self.load_image_method(item, key)
                
                # Acquire lock before modifying shared data
                with self.cache_lock:  # <--- THREAD-SAFE WRITES
                    array = getattr(self, key_array)
                    array[int_index] = value
                    self.dataframe.loc[int_index, f"{key}_in_cache"] = True
                return value
            else:
                return getattr(self, key_array)[int_index]
        else:
            return self.load_image_method(item, key)
    
    def cache_image(self, item:Dict[str, Any], keys:List[str]):
        for key in keys:
            self.load_image(item, key)
    
    def load_plume_simulation(self, plume_item:Dict[str, Any]) -> Tuple[NDArray, NDArray]:
        if self.cache:
            int_index = plume_item["int_index"]
            if not plume_item["in_cache"]:
                ch4, plume_mask = self.load_plume_simulation_method(plume_item)
                with self.cache_lock_plumes:  # <--- THREAD-SAFE WRITES
                    self.cache_ch4path_plumes[int_index] = ch4
                    self.cache_plumepath_plumes[int_index] = plume_mask
                    self.plume_dataframe.loc[int_index, "in_cache"] = True
            
            return self.cache_ch4path_plumes[int_index], self.cache_plumepath_plumes[int_index]
        else:
            return self.load_plume_simulation_method(plume_item)
    
    def subset_data_for_debugging(self, nimages:int, nplumes:Optional[int]=None, 
                                  cache:bool=False):
        self.dataframe = self.dataframe.sample(n=nimages).reset_index(drop=True)
        self.dataframe["int_index"] = self.dataframe.index.values
        self.all_locs = self.dataframe["location_name"].unique().tolist()
        self._compute_locations_few_samples()
        self.plume_dataframe = self.plume_dataframe.sample(n=nplumes).reset_index(drop=True)
        self.plume_dataframe["int_index"] = self.plume_dataframe.index.values

        if cache:
            self.cache = cache
            self.initialize_cache()
            self.cache_all(nworkers=4)
        self.log_info_data()

    def cache_all(self, nworkers:int=0):
        self.cache_all_images(nworkers)
        if self.do_simulation:
            self.cache_plumes_simulation(nworkers)

    def cache_all_images(self, nworkers:int=0):
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        assert self.cache is not None, "Cache is not enabled call the Dataset with cache=True"
        self.logger.info(f"Caching {self.dataframe.shape[0]} images")
        items = self.dataframe.to_dict(orient="records")
        keys_load = ["s2path", "plumepath", "cloudmaskpath"]

        if nworkers == 0:
            for item in tqdm(items, total=len(items)):
                self.cache_image(item, keys_load)
        else:
            with ThreadPoolExecutor(max_workers=nworkers) as executor:
                list(tqdm(executor.map(lambda x: self.cache_image(x, keys=keys_load), items), 
                          total=len(items),
                          desc="Caching images"))
        
        # Assert all images are in cache
        assert self.dataframe["s2path_in_cache"].all(), "Not all images are in cache!"
        assert self.dataframe["plumepath_in_cache"].all(), "Not all images are in cache!"
        assert self.dataframe["cloudmaskpath_in_cache"].all(), "Not all images are in cache!"
        
        # Resubset dataframe_few_samples_or_few_neg to have caching fields
        if self.dataframe_few_samples_or_few_neg is not None:
            self.dataframe_few_samples_or_few_neg = self.dataframe[self.dataframe.locs_few_samples | self.dataframe.locs_few_neg]
        
        # Remove the lock
        self.cache_lock = None
    
    def cache_plumes_simulation(self, nworkers:int=0):
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        assert self.cache is not None, "Cache is not enabled call the Dataset with cache=True"
        assert self.do_simulation, "Simulation is not enabled"
        self.logger.info(f"Caching {self.plume_dataframe.shape[0]} plumes to simulate")
        items_plumes = self.plume_dataframe.to_dict(orient="records")

        if nworkers == 0:
            for item in tqdm(items_plumes, total=len(items_plumes)):
                self.load_plume_simulation(item)
        else:
            with ThreadPoolExecutor(max_workers=nworkers) as executor:
                list(tqdm(executor.map(lambda x: self.load_plume_simulation(x), 
                                       items_plumes),
                            total=len(items_plumes),
                            desc="Caching plumes to simulate"))
        self.logger.info("Plumes to simulate cached")

        # Assert all plumes are in cache
        assert self.plume_dataframe["in_cache"].all(), "Not all plumes are in cache!"

        # Remove the lock to avoid concurrency issues
        self.cache_lock_plumes = None
        
    def stratified_sample(self) -> Dict[str, torch.Tensor]:        
        # Sample location
        if self.stratify_by_location:
            _location_name = np.random.choice(self.all_locs)
            if (_location_name in self.locs_few_samples) or (_location_name in self.locs_few_neg):
                # Sample from locations with few samples or no negative samples
                self.logger.debug(f"Sampled location {_location_name} with few samples or no negative samples")
                data_loc = self.dataframe_few_samples_or_few_neg
                loc_few_samples = True
            else:
                data_loc = self.dataframe[(self.dataframe.location_name==_location_name)]
                self.logger.debug(f"Sampled location {_location_name}")
                loc_few_samples = False
        else:
            data_loc = self.dataframe
            loc_few_samples = False
        
        # Sample plume y/n
        sample_plume = np.random.choice([True, False])
        n_pos = data_loc.isplume.sum()
        cm = None
        if not self.do_simulation and sample_plume and (n_pos == 0):
            sample_plume = False

        if not sample_plume:
            self.logger.debug(f"\tSampling no plume")
            neg_data_loc = data_loc.loc[(~data_loc.isplume)]
            if neg_data_loc.shape[0] == 0:
                raise ValueError(f"No negative samples in location {_location_name}. Samples {data_loc.shape[0]} n positives {n_pos}")

            index = np.random.choice(neg_data_loc.shape[0])
            item = neg_data_loc.iloc[index]
            # s2_data = self.to_tensor(self.load_image(item, "s2path")).permute(2,0,1)
            s2_data = self.load_image(item, "s2path")
            label = np.zeros(s2_data.shape[1:], dtype=bool)
            simulated = 0
            isplume = 0
        else:
            self.logger.debug(f"\tSampling plume")
            if self.do_simulation:
                if loc_few_samples and (n_pos > 0):
                    # 50%
                    simulate = np.random.choice([True, False])
                elif (n_pos > N_POS_SIMULATE):
                    # simulate 10% of the times
                    simulate = np.random.choice([True, False], p=[.1, .9])
                elif (n_pos > 0):
                    # simulate 90% of the times
                    simulate = np.random.choice([True, False], p=[.9, .1])
                else:
                    simulate = True
                self.logger.debug(f"\t loc with {n_pos} plumes. Simulate: {simulate}")
            else:
                self.logger.debug(f"\t loc with {n_pos} plumes. SIMULATION DISABLED")
                simulate = False
            
            # If there are enough positive images sample from them
            if not simulate:
                pos_data_loc = data_loc.loc[(data_loc.isplume)]
                index = np.random.choice(pos_data_loc.shape[0])
                item = pos_data_loc.iloc[index]
                self.logger.debug(f"\t\tSampled plume image: {item['year_month_day']} wind speed: {item['wind_speed']:.2f}m/s observability: {item['observability']} flux: {item['ch4_fluxrate']/1000:.1f}t/h")
                # s2_data = self.to_tensor(self.load_image(item, "s2path")).permute(2,0,1)
                s2_data = self.load_image(item, "s2path")
                # label = self.to_tensor(self.load_image(item, "plumepath"))
                label = self.load_image(item, "plumepath").astype(bool)
                window_row_off = int(round(item["window_row_off"]))
                window_col_off = int(round(item["window_col_off"]))
                window_width = int(round(item["window_width"]))
                window_height = int(round(item["window_height"]))
                simulated = 0
                isplume = 1
            else:
                # Otherwise construct a fake plume image
                # Sample a negative image
                neg_data_loc = data_loc.loc[(~data_loc.isplume)]
                index = np.random.choice(neg_data_loc.shape[0])
                item = neg_data_loc.iloc[index]
                s2_data = self.load_image(item, "s2path")
                self.logger.debug(f"\t\tSampled no plume image: {item['year_month_day']} wind speed: {item['wind_speed']:.2f}m/s observability: {item['observability']}")
                if (item["wind_speed"] > 9) or (item["observability"] != "clear") or item['offshore']:
                    # Do not sample plume if wind speed is high or observability is not clear
                    self.logger.debug(f"\t\tSampling no plume. Wind speed: {item['wind_speed']:.2f}m/s observability: {item['observability']} offshore: {item['offshore']}")
                    # s2_data = self.to_tensor(s2).permute(2,0,1)                    
                    # label = torch.zeros_like(s2_data[0,...], device=self.device)
                    label = np.zeros(s2_data.shape[1:], dtype=bool)
                    simulated = 0
                    isplume = 0
                else:
                    min_flux_rate_sim = MIN_FLUXRATE_SIM  
                                     
                    self.logger.debug(f"\t\tSampling no plume and simulating a plume with fluxrate > {min_flux_rate_sim/1000:.1f}t/h")
                    # sample a plume with similar wind speed
                    wind_distance = np.abs(self.plume_dataframe.wind_speed - item["wind_speed"])
                    min_distance = wind_distance.min()
                    distance_search = max(1.5, min_distance)
                    plumes_samples = self.plume_dataframe[(self.plume_dataframe.ch4_fluxrate > min_flux_rate_sim) & (wind_distance <= distance_search)]
                    plume_sample_size = plumes_samples.shape[0]
                    if plume_sample_size == 0:
                        self.logger.debug(f"\t\tSampling no plume. No plumes found with fluxrate > {min_flux_rate_sim/1000:.1f}t/h and wind speed similar to {item['wind_speed']:.2f}m/s")
                        # label = torch.zeros_like(s2_data[0,...], device=self.device)
                        label = np.zeros(s2_data.shape[1:], dtype=bool)
                        simulated = 0
                        isplume = 0
                    else:
                        plume_item = plumes_samples.iloc[np.random.choice(plume_sample_size)]

                        ch4_plume, plume_mask = self.load_plume_simulation(plume_item)

                        window_row_off_plume = int(round(plume_item["window_row_off"]))
                        window_col_off_plume = int(round(plume_item["window_col_off"]))
                        window_width_plume = int(round(plume_item["window_width"]))
                        window_height_plume = int(round(plume_item["window_height"]))
                        
                        ch4_plume = ch4_plume[window_row_off_plume:(window_row_off_plume+window_height_plume),
                                            window_col_off_plume:(window_col_off_plume+window_width_plume)]
                        plume_mask = plume_mask[window_row_off_plume:(window_row_off_plume+window_height_plume),
                                                window_col_off_plume:(window_col_off_plume+window_width_plume)]

                        # augment by scaling the ch4_plume by uniform sampling scale in [0.5, 1.5]
                        scale = np.random.uniform(0.5, 1.5)
                        ch4_plume = ch4_plume * scale
                        self.logger.debug(f"\t\t Simulating plume from image {plume_item['location_name']} {plume_item['tile']} scale: {scale:.2f} fluxrate: {scale * plume_item['ch4_fluxrate']/1000:.1f}t/h")
                        
                        try:
                            simout = self.simulator.simulate_plume(ch4=ch4_plume, plume_mask=plume_mask,
                                                                    wind_vector_ch4=[plume_item["wind_u"], plume_item["wind_v"]],
                                                                    image=s2_data, 
                                                                    b11_index=self.b11_index_original_input_image(item["satellite"]),
                                                                    b12_index=self.b12_index_original_input_image(item["satellite"]),
                                                                    satellite=item["satellite"],
                                                                    wind_vector_image=[item["wind_u"], item["wind_v"]], 
                                                                    vza=item["vza"], sza=item["sza"])
                            
                            isplume = 1

                            # Set as no plume if the plume mask intersects with the cloud mask in more than 50% of the pixels
                            if item["percentage_clear"] < 90:
                                cm = self.load_image(item, "cloudmaskpath")
                                clear_mask = cm == 0
                                percentage_pixels_plume_clear = np.sum(simout["label"] & clear_mask) / np.sum(simout["label"]) * 100
                                if percentage_pixels_plume_clear < 50:
                                    self.logger.debug(f"\t\tSampling no plume. Percentage of plume pixels clear {percentage_pixels_plume_clear} < 50%")
                                    simout["label"] = np.zeros_like(simout["label"])
                                    simout["image"] = s2_data
                                    isplume = 0
                            
                            # TODO create a flare mask and set the label to zero if there is overlap with the flare mask

                            # TODO simulate only in images with observability clear?

                            # s2_data = self.to_tensor(simout["image"])
                            # label = self.to_tensor(simout["label"])
                            s2_data = simout["image"]
                            label = simout["label"].astype(bool)
                            window_row_off = int(round(simout["window_row_off"]))
                            window_col_off = int(round(simout["window_col_off"]))
                            window_width = int(round(simout["window_width"]))
                            window_height = int(round(simout["window_height"]))
                            simulated = 1
                        except Exception:
                            self.logger.error(f"""Simulation failed. 
                                            Plume {plume_item['tile']} location name {plume_item['location_name']} with wind {plume_item['wind_u']}, {plume_item['wind_v']}. 
                                            Image {item['tile']} location name {item['location_name']} with wind {item['wind_u']}, {item['wind_v']}
                                            Window plume: Window(row_off={window_row_off_plume}, col_off={window_col_off_plume}, width={window_width_plume}, height={window_height_plume})
                                            """, 
                                            exc_info=True)
                            # label = torch.zeros_like(s2_data[0,...], device=self.device)
                            label = np.zeros(s2_data.shape[1:], dtype=bool)
                            simulated = 0
                            isplume = 0
        
        if isplume == 0:
            window_row_off, window_col_off, window_width, window_height = None, None, None, None
        
        return self.postprocess_item(isplume=isplume, window_row_off=window_row_off, 
                                     window_col_off=window_col_off, 
                                     window_width=window_width, window_height=window_height, 
                                     s2_data=s2_data, label=label, 
                                     simulated=simulated, item=item,
                                     cm=cm)
    
    def band_names_original_input_image(self, satellite:str) -> List[str]:
        if satellite.startswith("S2"):
            all_bands_satellite_s2_naming = self.bands_expected_sentinel_2
        else:
            all_bands_satellite_s2_naming = self.bands_expected_landsat_8_s2naming
        return all_bands_satellite_s2_naming
    
    def b11_index_original_input_image(self, satellite:str) -> int:
        return self.band_names_original_input_image(satellite).index("B11")

    def b12_index_original_input_image(self, satellite:str) -> int:
        return self.band_names_original_input_image(satellite).index("B12")


    def postprocess_item(self, isplume:int, window_row_off:Optional[int],
                         window_col_off:Optional[int], window_width:Optional[int],
                         window_height:Optional[int], s2_data:NDArray,
                         label:NDArray, simulated:int, 
                         item:pd.Series,
                         cm:Optional[NDArray]=None) -> Dict[str, torch.Tensor]:
        """
        Postprocess the item sampled from the dataset. This includes cropping the image, computing the ch4,
        computing the mbmp and normalizing the input data.

        Args:
            isplume (int): 1/0 if the image has a plume or not
            window_row_off (Optional[int]): row offset of the window containing the plume. None if isplume is 0
            window_col_off (Optional[int]): column offset of the window containing the plume. None if isplume is 0
            window_width (Optional[int]): width of the window containing the plume. None if isplume is 0
            window_height (Optional[int]): height of the window containing the plume. None if isplume is 0
            s2_data (NDArray): loaded image from the dataset in format (C, H, W) with 
                input values in ToA reflectance multiplied by 10_000 and dtype: uint16.
            label (NDArray): Plume mask in format (H, W). True if plume, False if not plume
            simulated (int): 1/0 if the plume was simulated or not
            item (pd.Series): metadata of the image.
            cm (Optional[NDArray], optional): cloud mask. Defaults to None.
                If not None, it should be in format (H, W) with 0 if clear >=1 if contaminated (cloud or cloud shadow)

        Raises:
            ValueError: If the number of bands in the data is not the expected number of bands

        Returns:
            Dict[str, torch.Tensor]: dictionary with the processed data:
                - "y_context_ls0_0": torch.Tensor with the input data (C, H, W) values ToA reflectance / 2.
                    This tensor has concatenated the input image, the reference image, the mbmp, the cloud mask and the wind components.
                    Names of the bands in this tensor are given by self.bands_out
                - "label": torch.Tensor with the plume mask.
                - "mbmp": torch.Tensor with the MBMP
                - "ch4": torch.Tensor with the ch4 values
                - "isplume": torch.Tensor with 1/0 if the image has a plume or not
                - "simulated": torch.Tensor with 1/0 if the plume was simulated or not
                - "location_name": str with the location name
                - "tile": str with the tile name
                - "id_loc_image": str with the id_loc_image
                - "wind": vector with the wind components U and V
        """
        location_name = item["location_name"]

        # Select the window to crop the image
        if self.mode == "train":
            if self.window_size_training < WINDOW_SIZE_DATA:
                if (isplume == 0):
                # isplume is false, sample a random window of size self.window_size_training x self.window_size_training
                    start_col = np.random.choice(range(0, WINDOW_SIZE_DATA - self.window_size_training))
                    start_row = np.random.choice(range(0, WINDOW_SIZE_DATA - self.window_size_training))
                else:
                    start_row, start_col = sample_window(window_row_off, window_col_off, window_width, window_height,
                                                        window_size_training=self.window_size_training,
                                                        add_jitter=self.mode=="train")
            else:
                start_row = 0
                start_col = 0
            end_row = start_row + self.window_size_training
            end_col = start_col + self.window_size_training
        else:
            start_row = 0
            start_col = 0
            end_row = WINDOW_SIZE_DATA
            end_col = WINDOW_SIZE_DATA
        
        # Expected bands in the data
        all_bands_satellite_s2_naming = self.band_names_original_input_image(item["satellite"])
        
        if s2_data.shape[0] != (2*len(all_bands_satellite_s2_naming)):
            error_msg = f"Item: {item['location_name']} {item['tile']} {item['id_loc_image']} Expected {len(all_bands_satellite_s2_naming)} bands, got {s2_data.shape[0]} bands"
            self.logger.error(error_msg)
            raise ValueError(error_msg)            

        b11_index = all_bands_satellite_s2_naming.index("B11")
        b12_index = all_bands_satellite_s2_naming.index("B12")

        # compute ch4
        if  cm is None:
            cm = self.load_image(item, "cloudmaskpath")
        
        validmask = cm == 0

        # Estimate the Delta transmittance of the B12/B11 ratio
        if item["offshore"]:
            self.logger.debug(f"Using SBMP for offshore location {location_name}")
            dtrest = mixing_ratio_methane.ratio_bands(s2_data, numerator_index=b12_index,
                                                      denominator_index=b11_index, validmask=validmask,
                                                      fill_value_default=1,
                                                      plumemaskbool=label)
        else:
            self.logger.debug(f"Using MBMP for onshore location {location_name}")
            dtrest = mixing_ratio_methane.ratio_IL(s2_data[:len(all_bands_satellite_s2_naming),...],
                                                   background_s2=s2_data[len(all_bands_satellite_s2_naming):,...],
                                                   b11_index=b11_index, b12_index=b12_index,
                                                   fill_value_ratio_il=1,
                                                   validmask=validmask, plumemaskbool=label, corregister=False)
        ch4 = self.ch42tr.deltach4_from_ratio_transmittance(satellite=item["satellite"],
                                                            sza=item["sza"], vza=item["vza"],
                                                            ratio_il=dtrest)
        
        # Crop the images
        s2_data = s2_data[:, start_row:end_row, start_col:end_col]
        label = label[start_row:end_row, start_col:end_col]
        ch4 = ch4[start_row:end_row, start_col:end_col]
        cm = cm[start_row:end_row, start_col:end_col]
        wind_vector = [_wind_value(item["wind_u"]), _wind_value(item["wind_v"])]
        wind_vector = np.array(wind_vector, dtype=np.float32)

        # If mode == train rotate the images (90,180, 270, 0) degrees
        if self.mode == "train" and not self.only_film_locs: # , "post_2022_tune"
            angle = np.random.choice([0, 90, 180, 270])
            if angle != 0:
                self.logger.debug(f"\t\t Rotating image {angle} degrees")
                s2_data = np.rot90(s2_data, k=angle//90, axes=(1,2))
                label = np.rot90(label, k=angle//90, axes=(0,1))
                ch4 = np.rot90(ch4, k=angle//90, axes=(0,1))
                cm = np.rot90(cm, k=angle//90, axes=(0,1))
                wind_vector = plumesimulation.rotate_wind_vector(wind_vector, angle)

        # Convert to torch tensors
        s2_data = self.to_tensor(s2_data)
        label = self.to_tensor(label)
        ch4 = self.to_tensor(ch4)

        # Normalize s2_data. Input is given in ToA reflectance multiplied by 10_000 
        s2_data[torch.isnan(s2_data)] = 0
        s2_data = s2_data/5000
        s2_data = torch.clamp(s2_data, 0, 2) # 0 to 1 ToA reflectance
        s2_data[torch.isinf(s2_data)] = 2

        with torch.no_grad():
            # MBMP is computed with the original s2/landsat bands. 
            # Hence s2_data could have 13*2 bands if S2 and 8*2 if L8
            mbmp =  mbmp_torch.to_mbmp(s2_data, b11_index=b11_index,
                                       b12_index=b12_index, 
                                       b11_index_prev=b11_index + len(all_bands_satellite_s2_naming),
                                       b12_index_prev=b12_index + len(all_bands_satellite_s2_naming))
            # Note we can't use dtrest here because it's normalized using the plumemask! 
            
            # TODO: do MBSP if offshore?

        if len(self.bands) != len(all_bands_satellite_s2_naming):
            band_indexes = [all_bands_satellite_s2_naming.index(b) for b in self.bands]
            bands_indexes_all = band_indexes + [b+len(all_bands_satellite_s2_naming) for b in band_indexes]
            s2_data = s2_data[tuple(bands_indexes_all),...]

        if not self.multipass:
            s2_data = s2_data[:len(self.bands),...]
        
        # Concatenate wind
        if self.wind:
            wind_u = torch.ones_like(label, device=self.device).unsqueeze(0)*wind_vector[0]
            wind_v = torch.ones_like(label, device=self.device).unsqueeze(0)*wind_vector[1]
            if self.norm_wind:
                wind_u = wind_u / 8
                wind_v = wind_v / 8
            s2_data = torch.cat([s2_data, wind_u, wind_v])

        # Concatenate cloud mask
        if self.cloud_mask:            
            cm = self.to_tensor(cm).unsqueeze(0)

            # Set to 1 if cloud, 0 otherwise
            cm[cm>0] = 1
            cm[torch.isnan(cm)] = 0
            cm[torch.isinf(cm)] = 1

            s2_data = torch.cat([s2_data, cm])
            
            if self.mask_input_data:
                s2_data = s2_data * (1-cm)
        
        # Concatenate MBMP
        if self.cat_mbmp:
            s2_data = torch.cat([mbmp.unsqueeze(0), s2_data])         

        task = {"y_context_ls0_0": s2_data, # (C, H, W)
                "y_target": label, # (H, W)
                "mbmp": mbmp,
                "ch4": ch4.unsqueeze(0),
                "simulated": torch.tensor(simulated, device=self.device, dtype=torch.long),
                "isplume": torch.tensor(isplume, device=self.device, dtype=torch.long),
                "wind": torch.tensor(wind_vector, 
                                     device=self.device),
                "tile": item["tile"],
                "location_name": location_name,
                "id_loc_image": str(item["id_loc_image"])}
        
        if self.film_dict_mapping is not None:
            # set site_ids to zero 50% of the time in train mode
            if self.mode == "train" and self.film_train_zero_id and np.random.choice([True, False]):
                task["site_ids"] = 0
            elif location_name in self.film_dict_mapping:
                task["site_ids"] = self.film_dict_mapping[location_name]
            else:
                task["site_ids"] = 0
        else:
            task["site_ids"] = 0
        
        task["site_ids"] = torch.tensor(task["site_ids"], device=self.device, dtype=torch.long)

        return task
    
    def plot_item(self, item, sizeimg:int=4, text_prepend:str="", norm_rgb:float=1.) -> tuple[plt.Figure, plt.Axes]:
        nrows,ncols = 2, 5
        fig, ax = plt.subplots(nrows,ncols, figsize=(ncols*sizeimg, nrows*sizeimg),sharex=True, sharey=True)
        
        ax = ax.flatten()
        
        item_mli = self.dataframe_id_loc_image_indexed.loc[item['id_loc_image']]

        tiledate = item_mli["tile_date"]
        if isinstance(tiledate, pd.Timestamp):
            tiledate = tiledate.to_pydatetime().replace(tzinfo=timezone.utc)
            tiledate = tiledate.strftime("%Y-%m-%d")
        elif isinstance(tiledate, str):
            tiledate = tiledate[:10]
        
        text_prepend = text_prepend.rstrip()
        text = f"{text_prepend} {item_mli['country']} {item_mli['location_name']} {item_mli['satellite']} {tiledate} isplume: {item['isplume']} simulated: {item['simulated']} id_film: {item['site_ids']} {item['id_loc_image']}"
        input_data = item["y_context_ls0_0"]
        
        nbands = 6
        
        i = 0
        rgb = input_data[(3,2,1),...] / norm_rgb
        rgb = torch.permute(rgb, (1, 2, 0))
        ax[i].imshow(rgb.clip(0,1))
        ax[i].set_title(r"RGB")
        
        rgb_bg = input_data[(3+nbands,2+nbands,1+nbands),...]
        rgb_bg = torch.permute(rgb_bg, (1, 2, 0))
        i = 1
        ax[i].imshow(rgb_bg.clip(0,1))
        ax[i].set_title(r"RGB (Bg)")
        
        i = 2
        mbmp = input_data[0]
        vmin_mbmp = max(0.92, mbmp.min())
        im = ax[i].imshow(mbmp,cmap="plasma_r",vmax=1,vmin=vmin_mbmp, interpolation="nearest")
        plot.colorbar_next_to(im, ax[i])
        ax[i].set_title(r"MBMP")
        wind.add_wind_to_plot(item["wind"], ax=ax[i])
        
        i = 3
        ch4 = item["ch4"]
        im = ax[i].imshow(ch4[0],cmap="plasma",vmax=2_000,vmin=0)
        plot.colorbar_next_to(im, ax[i])
        ax[i].set_title(r"CH$_4$ (ppb)")
        wind.add_wind_to_plot(item["wind"], ax=ax[i])
        
        i = 4
        y_target = item["y_target"]
        im = ax[i].imshow(y_target,cmap="magma",vmax=1,vmin=0, interpolation="nearest")
        ax[i].set_title(f"Label {item['isplume']}")
        
        i = 5
        
        idx_cloudmask = self.bands_out.index("cloudmask")
        
        cm = input_data[idx_cloudmask]
        im = ax[i].imshow(cm, cmap="magma",vmax=1,vmin=0, interpolation="nearest")
        ax[i].set_title(f"Cloudmask")
        
        i = 6
        idx_b12 = self.bands_out.index("B12")
        b12 = input_data[idx_b12]
        im = ax[i].imshow(b12 / 2, cmap="magma", interpolation="nearest")
        plot.colorbar_next_to(im, ax[i])
        ax[i].set_title(f"B12")
                            
        i = 7
        idx_b12_bg = self.bands_out.index("B12_bg")
        b12_bg = input_data[idx_b12_bg]
        im = ax[i].imshow(b12_bg/ 2, cmap="magma", interpolation="nearest")
        plot.colorbar_next_to(im, ax[i])
        ax[i].set_title(f"B12 Bg") 
        
        i = 8
        # MBMP cropped to plume
        mbmp_cropped = mbmp * y_target
        mbmp_cropped[mbmp_cropped == 0] = 1
        im = ax[i].imshow(mbmp_cropped,cmap="plasma_r",vmax=1,vmin=vmin_mbmp, interpolation="nearest")
        plot.colorbar_next_to(im, ax[i])
        ax[i].set_title(r"MBMP cropped")
        wind.add_wind_to_plot(item["wind"], ax=ax[i])
        
        i = 9 
        # CH4 cropped
        ch4_cropped = ch4[0] * y_target
        im = ax[i].imshow(ch4_cropped,cmap="plasma",vmax=2_000,vmin=0)
        plot.colorbar_next_to(im, ax[i])
        ax[i].set_title(r"CH$_4$ (ppb) cropped")
        wind.add_wind_to_plot(item["wind"], ax=ax[i])
        
        
        
        fig.suptitle(text)   
        fig.tight_layout()
        
        for axs in ax:
            axs.axis("off")
        
        return fig, ax


def _wind_value(wind_val) -> float:
    if pd.isna(wind_val) or wind_val is None or not np.isfinite(wind_val):
        return 0

    # clamp value to -20, 20
    wind_val = np.clip(wind_val, -20, 20)
    
    return wind_val


def load_image(item:Union[pd.Series, dict[str, Any]], 
               key:str, 
               fs:Optional[fsspec.AbstractFileSystem]=None) -> GeoTensor:
    """
    Load the image from the item. The item can be a pandas Series or a dictionary.
    The key is the name of the column in the dataframe or the key in the dictionary.
    The image is loaded using `fsspec` and returned as a `GeoTensor`.

    Args:
        item (Union[pd.Series, dict[str, Any]]): Item to load the image from
        key (str): Key to load the image from
        fs (Optional[fs.AbstractFileSystem], optional): Filesystem to use. Defaults to None.

    Returns:
        GeoTensor: Loaded image
    """
    path:str = item[key]
    if path.endswith(".npy"):
        with _get_fs(fs, path).open(path, "rb") as f:
            values = np.load(f)
        if len(values.shape) == 3:
            values = np.transpose(values, (2, 0, 1))
            if values.shape[0] == 1:
                values = values[0]
        transform = Affine(*[item["transform_a"], item["transform_b"], item["transform_c"],
                             item["transform_d"], item["transform_e"], item["transform_f"]])
        crs = item["crs"]
        gt = GeoTensor(values=values, 
                       transform=transform, 
                       crs=crs)
        return gt
        
    gt = GeoTensor.load_file(path, 
                             fs=fs if path.startswith("az://") else None)
    if len(gt.shape) == 3 and gt.shape[0] == 1:
        gt.values = gt.values[0]
    return gt
