from concurrent.futures import ThreadPoolExecutor
from datetime import timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import fsspec
import loguru
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from georeader import plot, rasterize
from loguru._logger import Logger
from georeader.geotensor import GeoTensor
from georeader.readers import S2_SAFE_reader
from numpy.typing import NDArray
from rasterio.windows import Window
from torch.utils.data import Dataset
from tqdm import tqdm
from datetime import datetime

from marss2l.dataframe_image_plumes import (
    ALL_DATE_CUT,
    COLUMNS_MERGE_PLUMES,
    COUNTRIES_ARABIAN_PENINSULA,
    COUNTRIES_CASE_STUDIES,
    CSV_PATH_DEFAULT,
    CSV_PLUME_PATH_DEFAULT,
    INTERVALS_FLUXRATE,
    LOCATIONS_CONTROL_RELEASES,
    LOCS_OFFSHORE_ABLATION,
    LOCS_TRAINING_ABLATION,
    MIN_SAMPLES_LOCATION_TRAIN,
    MIN_SAMPLES_NEGATIVE_TRAIN,
    N_POS_SIMULATE,
    ORDER_CASE_STUDIES,
    SPLITS,
    UZB_AND_KAZAKH,
    PolygonorMultiPolygonOrStr,
    _set_case_study,
    fs_from_path,
    load_dataframe_split,
    load_image,
    plumes_good_overlap,
    read_csv_images,
    fake_reader,
    read_csv_plumes,
    set_interval_fluxrate,
    split_control_releases,
)
from marss2l.utils import isremotepath

read_csv = read_csv_images

from marss2l.mars_sentinel2 import mixing_ratio_methane, plumesimulation, transmittance_to_ch4, wind
from marss2l.mars_sentinel2.transmittance_to_ch4 import compute_xch4_retrieval
from marss2l.sampling import (
    WINDOW_SIZE_DATA,
    WINDOW_SIZE_TRAINING,
    get_window_from_item,
    sample_window,
)

from . import mbmp_torch

RELATION_CHANNELS_S2_L89 = {
    "B01": "B01",
    "B02": "B02",
    "B03": "B03",
    "B04": "B04",
    "B08": "B05",  # OR B08A
    "B10": "B09",
    "B11": "B06",
    "B12": "B07",
}


def bands_in_l89(channels_query_s2: List[str]) -> List[str]:
    """This is basically all the channels in the RELATION_CHANNELS_S2_L89 but to make sure they're consistently ordered"""
    return [
        RELATION_CHANNELS_S2_L89[c]
        for c in S2_SAFE_reader.normalize_band_names(channels_query_s2)
        if c in RELATION_CHANNELS_S2_L89
    ]


RELATION_CHANNELS_L89_S2 = {v: k for k, v in RELATION_CHANNELS_S2_L89.items()}


MIN_FLUXRATE_SIM = 3_500
MAX_FLUXRATE_SIM = 70_000

MIN_FLUXRATE_SIM_OFFSHORE = 20_000

# When we simulate over sources we will shift the ch4 values by this factor (i.e. ch4 = ch4 / DIV_FACTOR_SIMULATE_SOURCES)
MIN_DESIRED_FLUX_SIMULATE_SOURCES = 800
DIV_FACTOR_SIMULATE_SOURCES = MIN_FLUXRATE_SIM / MIN_DESIRED_FLUX_SIMULATE_SOURCES


# Define constants for defaults
DEFAULT_SPLIT = "all"
DEFAULT_WINDOW_SIZE_TRAINING = 200
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
DEFAULT_SIMULATE_ON_SOURCE_FRACTION = 0


BANDS_S2_IN_L8 = ["B02", "B03", "B04", "B08", "B11", "B12"]
N_POS_SIMULATE = 5
MIN_SAMPLES_LOCATION_TRAIN = 30
MIN_SAMPLES_NEGATIVE_TRAIN = 15

NSAMPLES_PER_EPOCH_DEFAULT = 2048 * 32


class DatasetPlumes(Dataset):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        mode: str = "train",
        multipass: bool = DEFAULT_MULTIPASS,
        wind: bool = DEFAULT_WIND,
        bands_l8: bool = DEFAULT_BANDS_L8,
        cloud_mask: bool = DEFAULT_CLOUD_MASK,
        cat_mbmp: bool = DEFAULT_CAT_MBMP,
        norm_wind: bool = DEFAULT_NORM_WIND,
        stratify_by_location: bool = DEFAULT_STRATIFY_BY_LOCATION,
        image_dataframe: pd.DataFrame = None,
        plume_dataframe: Optional[pd.DataFrame] = None,
        sources_dataframe: Optional[pd.DataFrame] = None,
        strprependlogs: str = SPLITS[DEFAULT_SPLIT][0],
        do_simulation: bool = DEFAULT_DO_SIMULATION,
        logger: Optional[Logger] = None,
        film_dict_mapping: Optional[Dict[str, int]] = None,
        film_train_zero_id: bool = DEFAULT_FILM_TRAIN_ZERO_ID,
        window_size_training: int = WINDOW_SIZE_TRAINING,
        window_size_data: int = WINDOW_SIZE_DATA,
        n_samples_per_epoch_train: int = NSAMPLES_PER_EPOCH_DEFAULT,
        simulate_on_source_fraction: float = DEFAULT_SIMULATE_ON_SOURCE_FRACTION,
        only_film_locs: bool = False,
        analysis_mode: bool = False,
        rotate_data_augmentation: bool = True,
        cache: bool = False,
        mask_input_data: bool = False,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        min_fluxrate_sim: float = MIN_FLUXRATE_SIM,
        max_fluxrate_sim: float = MAX_FLUXRATE_SIM,
        div_factor_simulate_sources: float = DIV_FACTOR_SIMULATE_SOURCES,
    ):
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
            image_dataframe (pd.DataFrame, optional): DataFrame containing the data. Defaults to None.
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
            plume_dataframe (Optional[pd.DataFrame], optional): DataFrame containing the plumes to simulate. Defaults to None.
            sources_dataframe (Optional[pd.DataFrame], optional): DataFrame containing the location of the sources for the simulation. Defaults to None.
            split (str, optional): Split of the dataset, one of "train_2023", "test_2023", "val_2023", "train", "test", "val". Defaults to SPLITS[DEFAULT_SPLIT][0].
            do_simulation (bool, optional): Simulate plumes. Defaults to True.
            logger (Optional[Logger], optional): Logger to use. Defaults to None.
            film_dict_mapping (Optional[Dict[str, int]], optional): Dictionary mapping location names to site_ids. Defaults to None.
            film_train_zero_id (bool, optional): If True, set site_ids to zero 50% of the time in train mode. Defaults to True.
            window_size_training (int, optional): Size of the training window. Defaults to WINDOW_SIZE_TRAINING.
                This is used during training to sample random windows of this size.
            window_size_data (int, optional): Size of the data window. Defaults to WINDOW_SIZE_DATA.
                This is used to define the size of the cache and it will be the size of the images returned in test/val mode.
            n_samples_per_epoch_train (int, optional): Number of samples per epoch during training. Defaults to NSAMPLES_PER_EPOCH_DEFAULT.
            simulate_on_source_fraction (float, optional): Fraction of source images to simulate plumes on. Defaults to 0.5.
            only_film_locs (bool, optional): Load only locations that satisfy the FiLM condition. This is used to
                fine-tune the FiLM parameters. Defaults to False.
            analysis_mode (bool, optional): Enable analysis mode, which alters the output format and content for analysis purposes. Defaults to False.
            rotate_data_augmentation (bool, optional): Apply data augmentation by rotating the images. Defaults to True.
            cache (bool, optional): Cache the images. Defaults to False.
            mask_input_data (bool, optional): Mask the input data with the cloud mask. That is, it will set the input data to zero where the cloud mask is not clear. Defaults to False.
            fs (Optional[fsspec.AbstractFileSystem], optional): Filesystem to use. Defaults to None.
            min_fluxrate_sim (float, optional): Minimum flux rate for plume simulation in kg/h. Defaults to MIN_FLUXRATE_SIM (3500).
            max_fluxrate_sim (float, optional): Maximum flux rate for plume simulation in kg/h. Defaults to MAX_FLUXRATE_SIM (70000).
            div_factor_simulate_sources (float, optional): Division factor to scale CH4 values when simulating plumes on sources. Defaults to DIV_FACTOR_SIMULATE_SOURCES.

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

        if image_dataframe is None:
            raise ValueError("image_dataframe should be provided")

        if logger is None:
            self.logger = loguru.logger
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
        self.strprependlogs = strprependlogs
        self.window_size_training = window_size_training
        self.n_samples_per_epoch_train = n_samples_per_epoch_train
        self.load_common_bands_landsat_and_s2 = bands_l8
        self.only_film_locs = only_film_locs
        self.simulate_on_source_fraction = (
            simulate_on_source_fraction if simulate_on_source_fraction is not None else 0.0
        )
        if not self.do_simulation:
            self.simulate_on_source_fraction = 0

        self.min_fluxrate_sim = min_fluxrate_sim
        self.max_fluxrate_sim = max_fluxrate_sim
        self.div_factor_simulate_sources = div_factor_simulate_sources

        self.rotate_data_augmentation = rotate_data_augmentation
        if not self.mode == "train":
            self.rotate_data_augmentation = False

        self.analysis_mode = analysis_mode

        self.window_size_data = window_size_data

        assert (
            self.window_size_training <= self.window_size_data
        ), f"window_size_training should be less than or equal to {self.window_size_data} given {self.window_size_training}"

        # assert only_film_locs only in train mode
        assert (
            not self.only_film_locs or self.mode == "train"
        ), "only_film_locs is only supported in train mode"

        # assert only_film_locs film_dict_mapping is provided
        assert (
            not self.only_film_locs or self.film_dict_mapping is not None
        ), "only_film_locs is True, but film_dict_mapping is not provided"

        # set simulation to false if only_film_locs (and raise warning)
        if self.only_film_locs:
            if self.do_simulation:
                self.logger.warning("Setting do_simulation to False because only_film_locs is True")
                self.do_simulation = False

        # Object to map transmittance from the MBMP ratio to \Delta CH4 concentrations
        self.ch42tr = transmittance_to_ch4.TransmittanceCH4InterpolationFromDict()

        self.image_dataframe = image_dataframe
        self.plume_dataframe = plume_dataframe
        self.sources_dataframe = sources_dataframe

        if self.simulate_on_source_fraction > 0:
            assert (
                self.sources_dataframe is not None
            ), "sources_dataframe must be provided if simulate_on_source_fraction > 0"
            self.sources_dataframe = self.sources_dataframe.set_index("id_loc_image")

        if self.plume_dataframe is not None:
            # Filter detached plumes and log number of plumes discarded
            num_detached = self.plume_dataframe["is_detached"].sum()
            if num_detached > 0:
                self.logger.info(f"Filtering out {num_detached} detached plumes for simulation from {self.plume_dataframe.shape[0]} total plumes.")
                self.plume_dataframe = self.plume_dataframe[~self.plume_dataframe["is_detached"]].copy()
            
            # Drop plumes with fluxrate NA
            self.plume_dataframe = self.plume_dataframe[
                self.plume_dataframe["ch4_fluxrate"].notna()
            ]
            self.plume_dataframe = self.plume_dataframe.reset_index(drop=True)
            self.plume_dataframe["int_index"] = self.plume_dataframe.index.values
        self.all_locs = self.image_dataframe["location_name"].unique().tolist()

        if self.do_simulation:
            assert self.plume_dataframe is not None and (
                self.plume_dataframe.shape[0] > 0
            ), "No plumes found in the plume_dataframe and simulation is set to True"

            # TODO support simulation in validation?
            assert self.mode == "train", "Simulation is only supported in train mode"
            self.simulator = plumesimulation.PlumeSimulator()

        if not self.load_common_bands_landsat_and_s2:
            raise NotImplementedError("Not supported training with all bands of S2")

        self.bands = BANDS_S2_IN_L8

        # Set up if mode is train: set locs with few samples, few negatives and few positives
        if self.mode == "train":
            # Compute location with few samples
            self._compute_locations_few_samples()

        # Set up FiLM dict
        self.film_dict_mapping = film_dict_mapping.copy() if film_dict_mapping is not None else None
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
                    if (
                        (k in self.locs_few_samples)
                        or (k in self.locs_few_neg)
                        or (k in self.locs_few_pos)
                        or (k not in self.all_locs)
                    ):
                        self.film_dict_mapping[k] = 0
                        n_samples_zero_id += 1
                self.logger.info(
                    f"Set {n_samples_zero_id} locations to zero_id in the film_dict_mapping out of {n_locs_film} locations"
                )

                # keep only locs with film_dict_mapping > 0 if only_film_locs
                if self.only_film_locs:
                    locs_film = set([k for k, v in self.film_dict_mapping.items() if v > 0])
                    self.image_dataframe = self.image_dataframe[
                        self.image_dataframe.location_name.isin(locs_film)
                    ].copy()
                    self.all_locs = self.image_dataframe["location_name"].unique().tolist()
                    self.logger.info(
                        f"Keep only locations with FiLM mapping in the dataset. There are {len(locs_film)} locations"
                    )

        # Reset the index and store a copy of the dataframe indexed by id_loc_image
        self.image_dataframe = self.image_dataframe.reset_index(drop=True)
        self.image_dataframe["int_index"] = self.image_dataframe.index.values
        self.dataframe_id_loc_image_indexed = self.image_dataframe.set_index("id_loc_image")

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
        self.bands_expected_sentinel_2 = BANDS_S2_IN_L8  # S2_SAFE_reader.BANDS_S2_L1C
        self.bands_expected_landsat_8_s2naming = BANDS_S2_IN_L8  # [RELATION_CHANNELS_L89_S2[b] for b in bands_in_l89(self.bands_expected_sentinel_2)]

        self.log_info_data()

        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # Initialise caching
        self.cache = cache
        if self.cache:
            self.initialize_cache()

    @property
    def total_pos(self):
        return self.image_dataframe.isplume.sum()

    @property
    def total_neg(self):
        return self.image_dataframe.shape[0] - self.total_pos

    def log_info_data(self):
        # Compute total number of positive and negative samples and log stats of the data
        total_pos = self.total_pos
        total_neg = self.total_neg

        self.logger.info(
            f"{self.strprependlogs} {self.mode} data from {len(self.all_locs)} locations"
        )
        self.logger.info(
            f"{self.strprependlogs} {self.mode} data between {min(self.image_dataframe['tile_date'])} to {max(self.image_dataframe['tile_date'])}"
        )
        self.logger.info(
            f"{self.strprependlogs} {self.mode} data size {self.image_dataframe.shape[0]} with {total_pos} plumes and {total_neg} images without plumes"
        )

        # log different satellites
        self.logger.info(
            f"{self.strprependlogs} {self.mode} Satellites in the dataset: {self.image_dataframe.satellite.unique()}"
        )

        if self.do_simulation:
            self.logger.info(
                f"{self.strprependlogs} {self.mode} Plumes dataset to simulate: {self.plume_dataframe.shape[0]}"
            )
            # log dates and number of unique locations
            self.logger.info(
                f"{self.strprependlogs} {self.mode} Plumes to simulate between {min(self.plume_dataframe['tile_date'])} to {max(self.plume_dataframe['tile_date'])}"
            )
            self.logger.info(
                f"{self.strprependlogs} {self.mode} Plumes to simulate from {len(self.plume_dataframe['location_name'].unique())} locations"
            )

        # Log bands
        self.logger.info(
            f"{self.strprependlogs} {self.mode} Bands output by the dataset: {self.bands_out}"
        )
    
    def find_image(self, location_name:str, tile:Optional[str]=None, tile_date:Optional[str | datetime]=None) -> Optional[pd.Series]:
        """
        Find an image in the dataframe by location name and tile.

        Args:
            location_name (str): Name of the location.
            tile (str): Tile identifier.
            tile_date (str | datetime): Date of the tile.
        
        Returns:
            Optional[pd.Series]: The row of the dataframe corresponding to the image, or None if not found.
        """
        if tile is not None and tile_date is not None:
            raise ValueError("Provide either tile or tile_date, not both")

        if tile_date is None:
            df_loc = self.image_dataframe[
                (self.image_dataframe.location_name == location_name)
                & (self.image_dataframe.tile == tile)
            ]
        else:
            if isinstance(tile_date, str):
                tile_date = datetime.fromisoformat(tile_date)
            df_loc = self.image_dataframe[
                (self.image_dataframe.location_name == location_name)
                & (self.image_dataframe.tile_date == tile_date)
            ]

        if df_loc.shape[0] == 0:
             return None
        else:
            return df_loc.iloc[0]
    
    def find_plume(self, location_name:str, tile:Optional[str]=None, tile_date:Optional[str | datetime]=None) -> Optional[pd.Series]:
        """
        Find a plume in the plume dataframe by location name and tile.

        Args:
            location_name (str): Name of the location.
            tile (str): Tile identifier.
            tile_date (str | datetime): Date of the tile.
        Returns:
            Optional[pd.Series]: The row of the plume dataframe corresponding to the plume, or None if not found.
        """
        if tile is not None and tile_date is not None:
            raise ValueError("Provide either tile or tile_date, not both")

        if tile_date is None:
            df_loc = self.plume_dataframe[
                (self.plume_dataframe.location_name == location_name)
                & (self.plume_dataframe.tile == tile)
            ]
        else:
            if isinstance(tile_date, str):
                tile_date = datetime.fromisoformat(tile_date)
            df_loc = self.plume_dataframe[
                (self.plume_dataframe.location_name == location_name)
                & (self.plume_dataframe.tile_date == tile_date)
            ]

        if df_loc.shape[0] == 0:
             return None
        else:
            return df_loc.iloc[0]

    def _compute_locations_few_samples(self):
        """
        Compute locations with few samples, few negative samples, or few positive samples.

        This method identifies locations that don't have enough training data and adds flags
        to the dataframe. It also creates a subset dataframe for locations that need special
        sampling treatment during training.

        The method performs the following operations:
        1. Identifies locations with fewer than MIN_SAMPLES_LOCATION_TRAIN total samples
        2. Identifies locations with fewer than MIN_SAMPLES_NEGATIVE_TRAIN negative samples
        3. Identifies locations with fewer than N_POS_SIMULATE positive samples
        4. If the combined subset of these locations still has too few samples, drops them entirely

        Adds the following columns to self.dataframe:
            - locs_few_samples (bool): True if the location has fewer than MIN_SAMPLES_LOCATION_TRAIN
              total samples (default: 30 samples)
            - locs_few_neg (bool): True if the location has fewer than MIN_SAMPLES_NEGATIVE_TRAIN
              negative samples (default: 15 negative samples)
            - locs_few_pos (bool): True if the location has fewer than N_POS_SIMULATE positive
              samples (default: 5 positive samples)

        Sets the following instance attributes:
            - self.locs_few_samples (set): Set of location names with few total samples
            - self.locs_few_neg (set): Set of location names with few negative samples
            - self.locs_few_pos (set): Set of location names with few positive samples
            - self.dataframe_few_samples_or_few_neg (pd.DataFrame or None): Subset of self.dataframe
              containing only locations with few samples or few negatives. Set to None if this
              subset is dropped due to insufficient data.

        Notes:
            - Locations with insufficient data are either:
              a) Kept in a special subset (dataframe_few_samples_or_few_neg) for special sampling, or
              b) Dropped entirely if the combined subset is too small
            - This method is only called when mode == "train"
            - Logs information about the number of affected locations
        """

        samples_per_location = self.image_dataframe.groupby("location_name").size()
        locs_few_samples_serie = samples_per_location[
            samples_per_location < MIN_SAMPLES_LOCATION_TRAIN
        ].index
        if locs_few_samples_serie.shape[0] > 0:
            self.logger.info(
                f"{self.strprependlogs} {self.mode}. There are {len(locs_few_samples_serie)} locations that have less than {MIN_SAMPLES_LOCATION_TRAIN} samples"
            )

        # Compute location with no few samples
        negativesamples_in_location = self.image_dataframe.groupby("location_name").isplumeneg.sum()
        locs_few_neg_serie = negativesamples_in_location[
            negativesamples_in_location < MIN_SAMPLES_NEGATIVE_TRAIN
        ].index
        if locs_few_neg_serie.shape[0] > 0:
            self.logger.info(
                f"{self.strprependlogs} {self.mode}. There are {len(locs_few_neg_serie)} locations that have less than {MIN_SAMPLES_NEGATIVE_TRAIN} negative samples"
            )

        # Compute locations with few positive samples
        positivesamples_in_location = self.image_dataframe.groupby("location_name").isplume.sum()
        locs_few_pos_serie = negativesamples_in_location[
            positivesamples_in_location < N_POS_SIMULATE
        ].index
        if locs_few_pos_serie.shape[0] > 0:
            self.logger.info(
                f"{self.strprependlogs} {self.mode}. There are {len(locs_few_pos_serie)} locations that have less than {N_POS_SIMULATE} positive samples"
            )

        # Use dataframe_few_samples_or_few_neg to sample when the location has few samples or few negative samples
        self.image_dataframe["locs_few_samples"] = self.image_dataframe.location_name.isin(
            locs_few_samples_serie
        )
        self.image_dataframe["locs_few_neg"] = self.image_dataframe.location_name.isin(
            locs_few_neg_serie
        )
        self.image_dataframe["locs_few_pos"] = self.image_dataframe.location_name.isin(
            locs_few_pos_serie
        )
        self.dataframe_few_samples_or_few_neg = self.image_dataframe[
            self.image_dataframe.locs_few_samples | self.image_dataframe.locs_few_neg
        ]

        # If there're still few samples or few negative samples in a single location, drop them from the dataframe
        if (self.dataframe_few_samples_or_few_neg.shape[0] < MIN_SAMPLES_LOCATION_TRAIN) or (
            self.dataframe_few_samples_or_few_neg.isplumeneg.sum() < MIN_SAMPLES_NEGATIVE_TRAIN
        ):
            self.logger.info(
                f"Drop locations with less than {MIN_SAMPLES_LOCATION_TRAIN} samples or less than {MIN_SAMPLES_NEGATIVE_TRAIN} negative samples"
            )
            self.logger.info(
                f"There was only {self.dataframe_few_samples_or_few_neg.shape[0]} samples and {self.dataframe_few_samples_or_few_neg.isplumeneg.sum()} negative samples in total across all these locations."
            )
            self.image_dataframe = (
                self.image_dataframe.loc[
                    ~self.image_dataframe.locs_few_samples & ~self.image_dataframe.locs_few_neg
                ]
                .copy()
                .reset_index(drop=True)
            )
            self.image_dataframe["int_index"] = self.image_dataframe.index.values
            self.dataframe_id_loc_image_indexed = self.image_dataframe.set_index("id_loc_image")
            self.dataframe_few_samples_or_few_neg = None
            self.all_locs = self.image_dataframe["location_name"].unique().tolist()
        else:
            self.logger.info(
                f"There are {self.dataframe_few_samples_or_few_neg.shape[0]} samples and {self.dataframe_few_samples_or_few_neg.isplumeneg.sum()} negative samples in total in locations with few samples or few negative samples."
            )

        self.locs_few_pos = set(
            self.image_dataframe.location_name[self.image_dataframe.locs_few_pos].unique()
        )
        self.locs_few_neg = set(
            self.image_dataframe.location_name[self.image_dataframe.locs_few_neg].unique()
        )
        self.locs_few_samples = set(
            self.image_dataframe.location_name[self.image_dataframe.locs_few_samples].unique()
        )

    def __len__(self):
        if self.mode == "train":
            return self.n_samples_per_epoch_train
        else:
            return len(self.image_dataframe)

    def to_tensor(self, arr):
        return torch.from_numpy(arr.astype(np.float32)).to(self.device)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
        if self.mode == "train":
            return self.stratified_sampling()

        item = self.image_dataframe.iloc[idx]

        s2_data = self.load_image(item, "s2path")
        if item["isplume"]:
            label = self.load_image(item, "plumepath").astype(bool)
            plume_window = get_window_from_item(item)
        else:
            label = np.zeros(s2_data.shape[1:], dtype=bool)
            plume_window = None

        return self.postprocess_item(
            isplume=int(item["isplume"]),
            plume_window=plume_window,
            s2_data=s2_data,
            label=label,
            simulated=0,
            item=item,
            ch4_fluxrate=item.get("ch4_fluxrate", 0),
        )

    def get_sample(self, location_name: str, tile: Optional[str] = None, tile_date: Optional[str | datetime] = None) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset by location name and tile.

        Args:
            location_name (str): Name of the location.
            tile (str): Tile identifier.
            tile_date (str | datetime): Date of the tile.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the processed data.
        """
        item = self.find_image(location_name, tile, tile_date)
        if item is None:
            raise ValueError("Image not found")

        s2_data = self.load_image(item, "s2path")
        if item["isplume"]:
            label = self.load_image(item, "plumepath").astype(bool)
            plume_window = get_window_from_item(item)
        else:
            label = np.zeros(s2_data.shape[1:], dtype=bool)
            plume_window = None

        return self.postprocess_item(
            isplume=int(item["isplume"]),
            plume_window=plume_window,
            s2_data=s2_data,
            label=label,
            simulated=0,
            item=item,
            ch4_fluxrate=item.get("ch4_fluxrate", 0),
        )

    def initialize_cache(self):
        if not self.cache:
            self.logger.warning("Cache is not enabled. Call the Dataset with cache=True")
            return

        # Add a thread lock for cache writes
        self.cache_lock = Lock()

        self.cache_s2 = np.empty(
            (
                len(self.image_dataframe),
                len(self.bands_expected_sentinel_2) * 2,
                self.window_size_data,
                self.window_size_data,
            ),
            dtype=np.uint16,
        )
        self.cache_plumepath = np.empty(
            (len(self.image_dataframe), self.window_size_data, self.window_size_data),
            dtype=np.uint8,
        )
        self.cache_cloudmask = np.empty(
            (len(self.image_dataframe), self.window_size_data, self.window_size_data),
            dtype=np.uint8,
        )
        self.dict_cache_name = {
            "s2path": "cache_s2",
            "cloudmaskpath": "cache_cloudmask",
        }
        # Set column in dataframe name and bool saying if the image is in cache
        for key in self.dict_cache_name:
            self.image_dataframe[f"{key}_in_cache"] = False

        if self.do_simulation:
            self.cache_lock_plumes = Lock()
            self.cache_ch4path_plumes = np.empty(
                (len(self.plume_dataframe), self.window_size_data, self.window_size_data),
                dtype=np.float32,
            )
            self.cache_plumepath_plumes = np.empty(
                (len(self.plume_dataframe), self.window_size_data, self.window_size_data),
                dtype=np.uint8,
            )
            self.dict_cache_name_plumes = {
                "ch4path": "cache_ch4path_plumes",
                "plumepath": "cache_plumepath_plumes",
            }
            self.plume_dataframe["in_cache"] = False

        if self.dataframe_few_samples_or_few_neg is not None:
            self.dataframe_few_samples_or_few_neg = self.image_dataframe[
                self.image_dataframe.locs_few_samples | self.image_dataframe.locs_few_neg
            ]

    def load_image_method(self, item, key: str) -> NDArray:
        """
        Load an image from the dataset.

        Args:
            item (dict): dictionary with the item information.
            key (str): key for the image to load. One of "s2path", "plumepath", "cloudmaskpath".

        Returns:
            NDArray: image data as a NumPy array.
        """
        path: str = item[key]
        fs = self.fs
        if fs is None:
            fs = fs_from_path(path)
        if path.endswith(".npy"):                          
            with fs.open(path, "rb") as f:
                values = np.load(f)
            if len(values.shape) == 3:
                values = np.transpose(values, (2, 0, 1))
                if values.shape[0] == 1:
                    return values[0]
            return values
        values = GeoTensor.load_file(path, fs=fs).values
        if len(values.shape) == 3 and values.shape[0] == 1:
            return values[0]
        return values

    def load_plume_method(self, item) -> NDArray:
        """
        Load plume mask for a given item.

        Args:
            item (dict): dictionary with the item information.
        Returns:
            NDArray: plume mask as a NumPy array.
        """
        geo = fake_reader(item)
        geometry = item["plume"]
        plume_mask = rasterize.rasterize_geometry_like(
            geometry, geo, crs_geometry="EPSG:4326", all_touched=True, return_only_data=True
        )
        return plume_mask


    def load_plume_simulation_method(self, plume_item) -> Tuple[NDArray, NDArray]:
        """
        Load ch4 and plume mask for a plume to simulate.

        Args:
            plume_item (dict): dictionary with the plume item information.

        Returns:
            Tuple[NDArray, NDArray]: CH4 image and plume mask as NumPy arrays.
        """

        geo = load_image(plume_item, "ch4path", fs=self.fs)
        geometry = plume_item["geometry"]
        plume_mask = rasterize.rasterize_geometry_like(
            geometry, geo, crs_geometry="EPSG:4326", all_touched=True, return_only_data=True
        )

        return geo.values, plume_mask

    def load_image(self, item, key: str) -> NDArray:
        """
        Load an image from the dataset, using cache if enabled.

        Args:
            item (dict): dictionary with the item information.
            key (str): key for the image to load. One of "s2path", "plumepath", "cloudmaskpath".

        Returns:
            NDArray: image data as a NumPy array.
        """
        if key == "plumepath":
            return self.load_plume_method(item)

        if self.cache:
            key_array = self.dict_cache_name[key]
            int_index = item["int_index"]
            if not item[f"{key}_in_cache"]:
                value = self.load_image_method(item, key)

                # TODO pad if needed to size self.window_size_data

                # Acquire lock before modifying shared data
                with self.cache_lock:  # <--- THREAD-SAFE WRITES
                    array = getattr(self, key_array)
                    array[int_index] = value
                    self.image_dataframe.loc[int_index, f"{key}_in_cache"] = True
                return value
            else:
                return getattr(self, key_array)[int_index]
        else:
            return self.load_image_method(item, key)

    def cache_image(self, item: Dict[str, Any], keys: List[str]):
        for key in keys:
            self.load_image(item, key)

    def load_plume_simulation(self, plume_item: Dict[str, Any]) -> Tuple[NDArray, NDArray]:
        if self.cache:
            int_index = plume_item["int_index"]
            if not plume_item["in_cache"]:
                ch4, plume_mask = self.load_plume_simulation_method(plume_item)
                # TODO pad if needed to size self.window_size_data
                with self.cache_lock_plumes:  # <--- THREAD-SAFE WRITES
                    self.cache_ch4path_plumes[int_index] = ch4
                    self.cache_plumepath_plumes[int_index] = plume_mask
                    self.plume_dataframe.loc[int_index, "in_cache"] = True

            return self.cache_ch4path_plumes[int_index], self.cache_plumepath_plumes[int_index]
        else:
            return self.load_plume_simulation_method(plume_item)

    def subset_data_for_debugging(
        self, nimages: int, nplumes: Optional[int] = None, cache: bool = False
    ):
        self.image_dataframe = self.image_dataframe.sample(n=nimages).reset_index(drop=True)
        self.image_dataframe["int_index"] = self.image_dataframe.index.values
        self.all_locs = self.image_dataframe["location_name"].unique().tolist()
        self._compute_locations_few_samples()
        self.plume_dataframe = self.plume_dataframe.sample(n=nplumes).reset_index(drop=True)
        self.plume_dataframe["int_index"] = self.plume_dataframe.index.values

        if cache:
            self.cache = cache
            self.initialize_cache()
            self.cache_all(nworkers=4)
        self.log_info_data()

    def cache_all(self, nworkers: int = 0):
        self.cache_all_images(nworkers)
        if self.do_simulation:
            self.cache_plumes_simulation(nworkers)

    def cache_all_images(self, nworkers: int = 0):
        assert self.cache is not None, "Cache is not enabled call the Dataset with cache=True"
        self.logger.info(f"Caching {self.image_dataframe.shape[0]} images")
        items = self.image_dataframe.to_dict(orient="records")
        keys_load = ["s2path", "cloudmaskpath"]

        if nworkers == 0:
            for item in tqdm(items, total=len(items)):
                self.cache_image(item, keys_load)
        else:
            with ThreadPoolExecutor(max_workers=nworkers) as executor:
                list(
                    tqdm(
                        executor.map(lambda x: self.cache_image(x, keys=keys_load), items),
                        total=len(items),
                        desc="Caching images",
                    )
                )

        # Assert all images are in cache
        assert self.image_dataframe["s2path_in_cache"].all(), "Not all images are in cache!"
        assert self.image_dataframe["cloudmaskpath_in_cache"].all(), "Not all images are in cache!"

        # Resubset dataframe_few_samples_or_few_neg to have caching fields
        if self.dataframe_few_samples_or_few_neg is not None:
            self.dataframe_few_samples_or_few_neg = self.image_dataframe[
                self.image_dataframe.locs_few_samples | self.image_dataframe.locs_few_neg
            ]

        # Remove the lock
        self.cache_lock = None

    def cache_plumes_simulation(self, nworkers: int = 0):
        assert self.cache is not None, "Cache is not enabled call the Dataset with cache=True"
        assert self.do_simulation, "Simulation is not enabled"
        self.logger.info(f"Caching {self.plume_dataframe.shape[0]} plumes to simulate")
        items_plumes = self.plume_dataframe.to_dict(orient="records")

        if nworkers == 0:
            for item in tqdm(items_plumes, total=len(items_plumes)):
                self.load_plume_simulation(item)
        else:
            with ThreadPoolExecutor(max_workers=nworkers) as executor:
                list(
                    tqdm(
                        executor.map(lambda x: self.load_plume_simulation(x), items_plumes),
                        total=len(items_plumes),
                        desc="Caching plumes to simulate",
                    )
                )
        self.logger.info("Plumes to simulate cached")

        # Assert all plumes are in cache
        assert self.plume_dataframe["in_cache"].all(), "Not all plumes are in cache!"

        # Remove the lock to avoid concurrency issues
        self.cache_lock_plumes = None

    def stratified_sampling(self) -> Dict[str, torch.Tensor]:
        """
        Perform stratified sampling to select and process a training sample.

        This method implements a sophisticated sampling strategy for training that:
        1. Samples a location (optionally with stratification)
        2. Decides whether to sample a plume or non-plume image
        3. For plume samples, decides whether to use real plumes or simulate synthetic ones
        4. Returns the processed sample ready for training

        The sampling strategy ensures balanced training by:
        - Treating locations with few samples specially (via dataframe_few_samples_or_few_neg)
        - Deciding simulation probability based on the number of positive samples available:
          * If location has few samples and some positives: 50% simulation
          * If location has >N_POS_SIMULATE positives: 10% simulation
          * If location has few positives (but >0): 90% simulation
          * If location has no positives: 100% simulation

        Simulation is only performed if:
        - do_simulation is True
        - The sampled negative image has good conditions (wind_speed ≤ 9 m/s,
          observability == "clear", not offshore)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the processed sample with keys:
                - "y_context_ls0_0": Input tensor (C, H, W) with stacked bands
                - "y_target": Binary plume mask (H, W)
                - "mbmp": MBMP ratio (H, W)
                - "ch4": CH4 concentration (1, H, W)
                - "isplume": Binary flag (0 or 1) indicating presence of plume
                - "simulated": Binary flag (0 or 1) indicating if plume was simulated
                - "location_name": Name of the location
                - "tile": Tile identifier
                - "id_loc_image": Unique image identifier
                - "wind": Wind vector (U, V) components (2,)
                - "site_ids": Site ID for FiLM conditioning

        Raises:
            ValueError: If no negative samples are available in the selected location

        Notes:
            - This method is only called when mode == "train"
            - Logging at DEBUG level provides detailed information about sampling decisions
            - The method handles edge cases like locations with no positive or negative samples
        """
        # Sample location
        if self.stratify_by_location:
            _location_name = np.random.choice(self.all_locs)
            if (_location_name in self.locs_few_samples) or (_location_name in self.locs_few_neg):
                # Sample from locations with few samples or no negative samples
                self.logger.debug(
                    f"Sampled location {_location_name} with few samples or no negative samples"
                )
                data_loc = self.dataframe_few_samples_or_few_neg
                loc_few_samples = True
            else:
                data_loc = self.image_dataframe[
                    (self.image_dataframe.location_name == _location_name)
                ]
                self.logger.debug(f"Sampled location {_location_name}")
                loc_few_samples = False
        else:
            data_loc = self.image_dataframe
            loc_few_samples = False

        # Sample plume y/n
        sample_plume = np.random.choice([True, False])
        n_pos = data_loc.isplume.sum()
        if not self.do_simulation and sample_plume and (n_pos == 0):
            sample_plume = False

        if not sample_plume:
            item = self.sample_no_plume_image(dataframe=data_loc)
            self.logger.debug(
                f"\tSampled no plume: {item['location_name']} {item['satellite']} {item['year_month_day']} {item['id_loc_image']}"
            )
            # s2_data = self.to_tensor(self.load_image(item, "s2path")).permute(2,0,1)
            s2_data = self.load_image(item, "s2path")
            return self.postprocess_item(
                isplume=0,
                plume_window=None,
                s2_data=s2_data,
                label=np.zeros(s2_data.shape[1:], dtype=bool),
                simulated=0,
                item=item,
                ch4_fluxrate=0.0,
            )

        self.logger.debug(f"\tSampling plume")
        if self.do_simulation:
            if loc_few_samples and (n_pos > 0):
                # 50%
                simulate = np.random.choice([True, False])
            elif n_pos > N_POS_SIMULATE:
                # simulate 10% of the times
                simulate = np.random.choice([True, False], p=[0.1, 0.9])
            elif n_pos > 0:
                # simulate 90% of the times
                simulate = np.random.choice([True, False], p=[0.9, 0.1])
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
            self.logger.debug(
                f"\t\tSampled plume image: {item['location_name']} {item['satellite']} {item['year_month_day']} wind speed: {item['wind_speed']:.2f}m/s observability: {item['observability']} flux: {item['ch4_fluxrate']/1000:.1f}t/h {item['id_loc_image']}"
            )
            s2_data = self.load_image(item, "s2path")
            label = self.load_image(item, "plumepath").astype(bool)
            plume_window = get_window_from_item(item)
            return self.postprocess_item(
                isplume=1,
                plume_window=plume_window,
                s2_data=s2_data,
                label=label,
                simulated=0,
                item=item,
                ch4_fluxrate=item.get("ch4_fluxrate", 0),
            )

        # Otherwise construct a fake plume image
        # Sample a negative image
        item = self.sample_no_plume_image(dataframe=data_loc)
        return self.simulate_plume(item)

    def sample_no_plume_image(self, dataframe: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Samples a no plume image from the dataframe

        Args:
            dataframe (Optional[pd.DataFrame]): DataFrame to sample from. If None, use self.image_dataframe.

        Raises:
            ValueError: If there are no negative samples in the dataframe.

        Returns:
            pd.Series: Item sampled with no plume.
        """
        if dataframe is None:
            dataframe = self.image_dataframe

        neg_data_loc = dataframe.loc[(~dataframe.isplume)]
        if neg_data_loc.shape[0] == 0:
            raise ValueError(f"No negative samples in dataframe {dataframe.shape[0]}")
        index = np.random.choice(neg_data_loc.shape[0])
        return neg_data_loc.iloc[index]

    def get_plumes_samples(self, wind_speed: float, offshore: bool) -> Optional[pd.DataFrame]:
        """
        Get plumes suitable for simulation given wind speed and offshore status.

        These are plumes with fluxrate between MIN_FLUXRATE_SIM(3500) and MAX_FLUXRATE_SIM(70000) and
        with wind speed similar to the given wind speed.

        Args:
            wind_speed (float): Wind speed in m/s to find similar plumes
            offshore (bool): Whether the location is offshore

        Returns:
            Optional[pd.DataFrame]: DataFrame with suitable plumes for simulation,
                or None if no suitable plumes are found
        """
        min_fluxrate = MIN_FLUXRATE_SIM_OFFSHORE if offshore else self.min_fluxrate_sim

        self.logger.debug(
            f"\t\tSearching for plumes to simulate with fluxrate between {min_fluxrate/1000:.1f} and {self.max_fluxrate_sim/1000:.1f} t/h"
        )

        # Find plumes with similar wind speed
        wind_distance = np.abs(self.plume_dataframe.wind_speed - wind_speed)
        min_distance = wind_distance.min()
        distance_search = max(1.5, min_distance)

        plumes_samples = self.plume_dataframe[
            (self.plume_dataframe.ch4_fluxrate > min_fluxrate)
            & (self.plume_dataframe.ch4_fluxrate < self.max_fluxrate_sim)
            & (wind_distance <= distance_search)
        ]

        if plumes_samples.shape[0] == 0:
            self.logger.debug(
                f"\t\tNo plumes found with fluxrate between {min_fluxrate/1000:.1f}t/h "
                f"and {self.max_fluxrate_sim/1000:.1f}t/h and wind speed similar to {wind_speed:.2f}m/s (search distance: {distance_search:.2f}m/s)"
            )
            return None

        self.logger.debug(
            f"\t\tFound {plumes_samples.shape[0]} plumes with fluxrate in "
            f"[{min_fluxrate/1000:.1f}, {self.max_fluxrate_sim/1000:.1f}]t/h "
            f"and wind speed within {distance_search:.2f}m/s of {wind_speed:.2f}m/s"
        )

        return plumes_samples

    def simulate_plume(
        self,
        item: Dict[str, Any],
        simulate_on_source_fraction: Optional[float] = None,
        loc_injection: Optional[tuple[int, int]] = None,
        plume_item: Optional[dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        if simulate_on_source_fraction is None:
            simulate_on_source_fraction = self.simulate_on_source_fraction

        s2_data = self.load_image(item, "s2path")

        wind_speed = _wind_value(item["wind_speed"])
        self.logger.debug(
            f"\t\tSampled no plume image: {item['location_name']} {item['year_month_day']} {item['satellite']} wind speed: {wind_speed:.2f}m/s observability: {item['observability']} {item['id_loc_image']}"
        )
        if (wind_speed > 9) or (item["observability"] != "clear") or item["offshore"]:
            # Do not sample plume if wind speed is high or observability is not clear
            self.logger.debug(
                f"\t\tSampling no plume. Wind speed: {wind_speed:.2f}m/s observability: {item['observability']} offshore: {item['offshore']}"
            )
            return self.postprocess_item(
                isplume=0,
                plume_window=None,
                s2_data=s2_data,
                label=np.zeros(s2_data.shape[1:], dtype=bool),
                simulated=0,
                ch4_fluxrate=0.0,
                item=item,
            )

        if plume_item is None:
            # Get suitable plumes for simulation based on wind speed
            plumes_samples = self.get_plumes_samples(wind_speed=wind_speed, offshore=item["offshore"])

            if plumes_samples is None or plumes_samples.shape[0] == 0:
                self.logger.debug("\t\tSampling no plume. No suitable plumes found for simulation")
                return self.postprocess_item(
                    isplume=0,
                    plume_window=None,
                    s2_data=s2_data,
                    label=np.zeros(s2_data.shape[1:], dtype=bool),
                    simulated=0,
                    ch4_fluxrate=0.0,
                    item=item,
                )

            plume_item = plumes_samples.iloc[np.random.choice(plumes_samples.shape[0])]

        ch4_plume, plume_mask = self.load_plume_simulation(plume_item)

        window_plume = get_window_from_item(plume_item, include_source=True)

        ch4_plume = ch4_plume[
            window_plume.row_off : (window_plume.row_off + window_plume.height),
            window_plume.col_off : (window_plume.col_off + window_plume.width),
        ]
        plume_mask = plume_mask[
            window_plume.row_off : (window_plume.row_off + window_plume.height),
            window_plume.col_off : (window_plume.col_off + window_plume.width),
        ]

        # Simulate the plume over a source
        if (loc_injection is None) and (simulate_on_source_fraction > 0):
            simulate_on_source = np.random.choice(
                [True, False],
                p=[simulate_on_source_fraction, 1 - simulate_on_source_fraction],
            )
            if simulate_on_source and (item["id_loc_image"] in self.sources_dataframe.index):
                locs_in_source = self.sources_dataframe.loc[item["id_loc_image"]]

                # Handle both Series (single source) and DataFrame (multiple sources) cases
                if isinstance(locs_in_source, pd.Series):
                    sampled_source = locs_in_source
                else:
                    # Multiple sources - sample randomly one of them
                    sampled_source = locs_in_source.iloc[np.random.choice(locs_in_source.shape[0])]

                loc_injection = sampled_source["pixel_row"], sampled_source["pixel_col"]
                self.logger.debug(
                    f"\t\t Simulating plume on source {sampled_source['id_mars_source']} at pixel {loc_injection}"
                )
            elif simulate_on_source:
                self.logger.debug(
                    f"\t\t Could not simulate plume on source as no source information available for image {item['id_loc_image']}"
                )
        else:
            self.logger.debug(f"\t\t Not simulating plume on source")

        plume_source = None
        fluxrate = plume_item["ch4_fluxrate"]
        if loc_injection is not None:
            # Load plume source location
            plume_source = (
                plume_item["pixel_row"] - window_plume.row_off,
                plume_item["pixel_col"] - window_plume.col_off,
            )
            # Make enhancement lower when we are simulating in sources
            ch4_plume = ch4_plume / self.div_factor_simulate_sources
            fluxrate = fluxrate / self.div_factor_simulate_sources

        # augment by scaling the ch4_plume by uniform sampling scale in [0.5, 1.5]
        scale = np.random.uniform(0.5, 1.5)
        ch4_plume = ch4_plume * scale
        fluxrate = fluxrate * scale
        self.logger.debug(
            f"\t\t Simulating plume {plume_item['location_name']} {plume_item['satellite']} {plume_item['tile_date'].strftime('%Y-%m-%d')} fluxrate: {fluxrate/1000:.1f}t/h id_plume: {plume_item['id_plume']}"
        )

        try:
            simout = self.simulator.simulate_plume(
                ch4=ch4_plume,
                plume_mask=plume_mask,
                wind_vector_ch4=[
                    _wind_value(plume_item["wind_u"]),
                    _wind_value(plume_item["wind_v"]),
                ],
                image=s2_data,
                loc_injection=loc_injection,
                plume_source=plume_source,
                b11_index=self.b11_index_original_input_image(item["satellite"]),
                b12_index=self.b12_index_original_input_image(item["satellite"]),
                satellite=item["satellite"],
                wind_vector_image=[_wind_value(item["wind_u"]), _wind_value(item["wind_v"])],
                vza=item["vza"],
                sza=item["sza"],
                return_transmittance_and_ch4=self.analysis_mode,
            )

            # Set as no plume if the plume mask intersects with the cloud mask in more than 50% of the pixels
            if item["percentage_clear"] < 90:
                cm = self.load_image(item, "cloudmaskpath")
                clear_mask = cm == 0
                percentage_pixels_plume_clear = (
                    np.sum(simout["label"] & clear_mask) / np.sum(simout["label"]) * 100
                )
                if percentage_pixels_plume_clear < 50:
                    self.logger.debug(
                        f"\t\tSampling no plume. Percentage of plume pixels clear {percentage_pixels_plume_clear} < 50%"
                    )
                    return self.postprocess_item(
                        isplume=0,
                        plume_window=None,
                        s2_data=s2_data,
                        label=np.zeros_like(simout["label"]),
                        simulated=0,
                        item=item,
                        cm=cm,
                    )

            # TODO create a flare mask and set the label to zero if there is overlap with the flare mask
            s2_data_sim = simout["image"]
            label = simout["label"].astype(bool)
            plume_window = get_window_from_item(simout)
            return self.postprocess_item(
                isplume=1,
                plume_window=plume_window,
                s2_data=s2_data_sim,
                s2_data_before_sim=s2_data,
                label=label,
                simulated=1,
                ch4sim=simout.get("ch4", None),
                ch4_fluxrate=fluxrate,
                item=item,
            )

        except Exception:
            self.logger.opt(exception=True).error(
                f"""Simulation failed.
                            Plume {plume_item['tile']} location name {plume_item['location_name']} with wind {plume_item['wind_u']}, {plume_item['wind_v']}.
                            Image {item['tile']} location name {item['location_name']} with wind {item['wind_u']}, {item['wind_v']}
                            Window plume: {window_plume}
                            """
            )
            # label = torch.zeros_like(s2_data[0,...], device=self.device)
            label = np.zeros(s2_data.shape[1:], dtype=bool)

            return self.postprocess_item(
                isplume=0,
                plume_window=None,
                s2_data=s2_data,
                label=label,
                simulated=0,
                ch4_fluxrate=0.0,
                item=item,
            )

    def band_names_original_input_image(self, satellite: str) -> List[str]:
        if satellite.startswith("S2"):
            all_bands_satellite_s2_naming = self.bands_expected_sentinel_2
        else:
            all_bands_satellite_s2_naming = self.bands_expected_landsat_8_s2naming
        return all_bands_satellite_s2_naming

    def b11_index_original_input_image(self, satellite: str) -> int:
        return self.band_names_original_input_image(satellite).index("B11")

    def b12_index_original_input_image(self, satellite: str) -> int:
        return self.band_names_original_input_image(satellite).index("B12")

    def postprocess_item(
        self,
        isplume: int,
        plume_window: Optional[Window],
        s2_data: NDArray,
        label: NDArray,
        simulated: int,
        item: pd.Series,
        s2_data_before_sim: Optional[NDArray] = None,
        cm: Optional[NDArray] = None,
        ch4sim: Optional[NDArray] = None,
        ch4_fluxrate: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Postprocess the item sampled from the dataset. This includes cropping the image, computing the ch4,
        computing the mbmp and normalizing the input data.

        Args:
            isplume (int): 1/0 if the image has a plume or not
            plume_window (Optional[Window]): Window containing the plume. None if isplume is 0
            s2_data (NDArray): loaded image from the dataset in format (C, H, W) with
                input values in ToA reflectance multiplied by 10_000 and dtype: uint16.
            label (NDArray): Plume mask in format (H, W). True if plume, False if not plume
            simulated (int): 1/0 if the plume was simulated or not
            item (pd.Series): metadata of the image.
            s2_data_before_sim (Optional[NDArray], optional): loaded image before simulation. Defaults to None.
                If simulated == 1, this should be the image before simulation.
            cm (Optional[NDArray], optional): cloud mask. Defaults to None.
                If not None, it should be in format (H, W) with 0 if clear >=1 if contaminated (cloud or cloud shadow)
            ch4sim (Optional[NDArray], optional): ch4 values for simulated plumes. Defaults to None.
            ch4_fluxrate (Optional[float], optional): fluxrate of the plume. Defaults to None.


        Raises:
            ValueError: If the number of bands in the data is not the expected number of bands

        Returns:
            Dict[str, torch.Tensor]: dictionary with the processed data:
                - "y_context_ls0_0": torch.Tensor with the input data (C, H, W) values ToA reflectance / 2.
                    This tensor has concatenated the input image, the reference image, the mbmp, the cloud mask and the wind components.
                    Names of the bands in this tensor are given by self.bands_out
                - "y_target": torch.Tensor with the plume mask (H, W)
                - "ch4forweighting": torch.Tensor with the ch4 values used for loss weighting (1, H, W).
                    For simulated plumes (simulated==1), this is ch4sim. For real plumes, this is the retrieval.
                - "isplume": torch.Tensor with 1/0 if the image has a plume or not
                - "tile": str with the tile name
                - "id_loc_image": str with the id_loc_image
                - "location_name": str with the location name
                - "site_ids": torch.Tensor with the site ID for FiLM conditioning

            if analysis_mode is True, also returns:
                - "mbmp": torch.Tensor with the MBMP (H, W)
                - "simulated": torch.Tensor with 1/0 if the plume was simulated or not
                - "wind": torch.Tensor with the wind components U and V (2,)
                - "tile_date": str with the tile date in ISO format
                - "satellite": str with the satellite name
                - "ch4": torch.Tensor with the ch4 retrieval (1, H, W)
                - "ch4sim": torch.Tensor with the ch4 values for simulated plumes (1, H, W)
                - "ch4_retrieval_before_sim": torch.Tensor with the ch4 retrieval before simulation (1, H, W)
                - "ch4_fluxrate": torch.Tensor with the fluxrate of the plume in kg/h
                - "angle_rotation": int with the rotation angle applied (0, 90, 180, or 270)
        """
        location_name = item["location_name"]

        # Select the window to crop the image
        if self.mode == "train":
            nrows = s2_data.shape[1]
            ncols = s2_data.shape[2]

            # # Figure out start_row and start_col
            start_row = 0
            start_col = 0
            if isplume == 0:
                if self.window_size_training < nrows:
                    start_row = np.random.choice(range(0, nrows - self.window_size_training))
                if self.window_size_training < ncols:
                    start_col = np.random.choice(range(0, ncols - self.window_size_training))
            else:
                if (self.window_size_training < ncols) or (self.window_size_training < nrows):
                    start_row, start_col = sample_window(
                        plume_window,
                        window_size_training=self.window_size_training,
                        add_jitter=True,
                        window_size_data=(nrows, ncols),
                    )

            # Define the end row and column
            end_row = start_row + self.window_size_training
            end_col = start_col + self.window_size_training
        else:
            start_row = 0
            start_col = 0
            end_row = self.window_size_data
            end_col = self.window_size_data

        # Expected bands in the data
        all_bands_satellite_s2_naming = self.band_names_original_input_image(item["satellite"])

        if s2_data.shape[0] != (2 * len(all_bands_satellite_s2_naming)):
            error_msg = f"Item: {item['location_name']} {item['tile']} {item['id_loc_image']} Expected {len(all_bands_satellite_s2_naming)} bands, got {s2_data.shape[0]} bands"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        b11_index = all_bands_satellite_s2_naming.index("B11")
        b12_index = all_bands_satellite_s2_naming.index("B12")

        # compute ch4
        if cm is None:
            cm = self.load_image(item, "cloudmaskpath")

        validmask = cm == 0

        # Estimate the Delta transmittance of the B12/B11 ratio and compute XCH4 retrieval
        ch4 = compute_xch4_retrieval(
            s2l_data=s2_data[: len(all_bands_satellite_s2_naming), ...],
            background_s2l=s2_data[len(all_bands_satellite_s2_naming) :, ...],
            offshore=item["offshore"],
            satellite=item["satellite"],
            sza=item["sza"],
            vza=item["vza"],
            b11_index=b11_index,
            b12_index=b12_index,
            validmask=validmask,
            label=label,
            transmittance_interpolator=self.ch42tr)
        
        if simulated == 1:
            # For simulated plumes, combine the simulated ch4 with the retrieval outside the plume
            ch4_retrieval_before_sim = compute_xch4_retrieval(
                s2l_data=s2_data_before_sim[: len(all_bands_satellite_s2_naming), ...],
                background_s2l=s2_data_before_sim[len(all_bands_satellite_s2_naming) :, ...],
                offshore=item["offshore"],
                satellite=item["satellite"],
                sza=item["sza"],
                vza=item["vza"],
                b11_index=b11_index,
                b12_index=b12_index,
                validmask=validmask,
                label=label,
                transmittance_interpolator=self.ch42tr
            )
        else:
            ch4_retrieval_before_sim = np.copy(ch4)
        
        # Crop the images
        # TODO or pad if needed (!)
        s2_data = s2_data[:, start_row:end_row, start_col:end_col]
        label = label[start_row:end_row, start_col:end_col]
        ch4 = ch4[start_row:end_row, start_col:end_col]
        cm = cm[start_row:end_row, start_col:end_col]
        ch4_retrieval_before_sim = ch4_retrieval_before_sim[
            start_row:end_row, start_col:end_col
        ]
        if ch4sim is not None:
            ch4sim = ch4sim[start_row:end_row, start_col:end_col]

        wind_vector = [_wind_value(item["wind_u"]), _wind_value(item["wind_v"])]
        wind_vector = np.array(wind_vector, dtype=np.float32)

        # If mode == train rotate the images (90,180, 270, 0) degrees
        angle = 0
        if self.rotate_data_augmentation and (not self.only_film_locs):  # , "post_2022_tune"
            angle = np.random.choice([0, 90, 180, 270])
            if angle != 0:
                self.logger.debug(f"\t\t Rotating image {angle} degrees")
                s2_data = np.rot90(s2_data, k=angle // 90, axes=(1, 2))
                label = np.rot90(label, k=angle // 90, axes=(0, 1))
                ch4 = np.rot90(ch4, k=angle // 90, axes=(0, 1))
                cm = np.rot90(cm, k=angle // 90, axes=(0, 1))
                if ch4sim is not None:
                    ch4sim = np.rot90(ch4sim, k=angle // 90, axes=(0, 1))
                ch4_retrieval_before_sim = np.rot90(
                    ch4_retrieval_before_sim, k=angle // 90, axes=(0, 1)
                )

                wind_vector = plumesimulation.rotate_wind_vector(wind_vector, angle)

        # Convert to torch tensors
        s2_data = self.to_tensor(s2_data)
        label = self.to_tensor(label)
        ch4 = self.to_tensor(ch4)
        ch4_retrieval_before_sim = self.to_tensor(ch4_retrieval_before_sim)

        # Normalize s2_data. Input is given in ToA reflectance multiplied by 10_000
        s2_data[torch.isnan(s2_data)] = 0
        s2_data = s2_data / 5000
        s2_data = torch.clamp(s2_data, 0, 2)  # 0 to 1 ToA reflectance
        s2_data[torch.isinf(s2_data)] = 2

        with torch.no_grad():
            # MBMP is computed with the original s2/landsat bands.
            # Hence s2_data could have 13*2 bands if S2 and 8*2 if L8
            mbmp = mbmp_torch.to_mbmp(
                s2_data,
                b11_index=b11_index,
                b12_index=b12_index,
                b11_index_prev=b11_index + len(all_bands_satellite_s2_naming),
                b12_index_prev=b12_index + len(all_bands_satellite_s2_naming),
            )
            # Note we can't use dtrest here because it's normalized using the plumemask!

            # TODO: do MBSP if offshore?

        if len(self.bands) != len(all_bands_satellite_s2_naming):
            band_indexes = [all_bands_satellite_s2_naming.index(b) for b in self.bands]
            bands_indexes_all = band_indexes + [
                b + len(all_bands_satellite_s2_naming) for b in band_indexes
            ]
            s2_data = s2_data[tuple(bands_indexes_all), ...]

        if not self.multipass:
            s2_data = s2_data[: len(self.bands), ...]

        # Concatenate wind
        if self.wind:
            wind_u = torch.ones_like(label, device=self.device).unsqueeze(0) * wind_vector[0]
            wind_v = torch.ones_like(label, device=self.device).unsqueeze(0) * wind_vector[1]
            if self.norm_wind:
                wind_u = wind_u / 8
                wind_v = wind_v / 8
            s2_data = torch.cat([s2_data, wind_u, wind_v])

        # Concatenate cloud mask
        if self.cloud_mask:
            cm = self.to_tensor(cm).unsqueeze(0)

            # Set to 1 if cloud, 0 otherwise
            cm[cm > 0] = 1
            cm[torch.isnan(cm)] = 0
            cm[torch.isinf(cm)] = 1

            s2_data = torch.cat([s2_data, cm])

            if self.mask_input_data:
                s2_data = s2_data * (1 - cm)

        # Concatenate MBMP
        if self.cat_mbmp:
            s2_data = torch.cat([mbmp.unsqueeze(0), s2_data])

        if ch4sim is None:
            ch4sim = torch.zeros_like(ch4, device=self.device)
        else:
            ch4sim = self.to_tensor(ch4sim)
        
        if simulated == 1:
            # For simulated plumes, combine the simulated ch4 with the retrieval outside the plume
            if ch4sim is None:
                raise ValueError("ch4sim must be provided for simulated plumes")
            ch4forweighting = ch4sim
            ch4noise_img = ch4_retrieval_before_sim
        else:
            if isplume == 1:
                ch4noise_img = (1-label).float() * ch4
            else:
                ch4noise_img = ch4

            ch4forweighting = ch4.clone()
        
        ch4noise = ch4noise_img.mean(axis=(-2, -1))

        task = {
            "y_context_ls0_0": s2_data,  # (C, H, W)
            "y_target": label,  # (H, W)
            "ch4forweighting": ch4forweighting.unsqueeze(0),  # (1, H, W)  
            "ch4noise": ch4noise, # (,) mean ch4 noise value
            "isplume": torch.tensor(isplume, device=self.device, dtype=torch.long),
            "location_name": location_name,
            "tile": item["tile"],
            "id_loc_image": str(item["id_loc_image"]),
        }

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

        if self.analysis_mode:
            task.update(
                {
                    "mbmp": mbmp,  # (H, W)
                    "simulated": torch.tensor(simulated, device=self.device, dtype=torch.long),
                    "wind": torch.tensor(wind_vector, device=self.device),
                    "tile_date": item["tile_date"].isoformat(),
                    "satellite": item["satellite"],
                    # Solar/view geometry of both passes, for the shot-noise
                    # propagation: converting each pass's reflectances to radiances
                    # needs its own angle and date. The _bg fields are empty for an
                    # offshore scene, which has no reference pass, and absent from
                    # CSVs exported before those columns existed.
                    "sza": _as_float(item["sza"]),
                    "vza": _as_float(item["vza"]),
                    "satellite_bg": _as_str(item.get("satellite_bg")),
                    "sza_bg": _as_float(item.get("sza_bg")),
                    "tile_date_bg": _as_str(item.get("tile_date_bg")),
                    "ch4": ch4.unsqueeze(0),  # (1, H, W)
                    "ch4sim": ch4sim.unsqueeze(0),  # (1, H, W)
                    "ch4_retrieval_before_sim": ch4_retrieval_before_sim.unsqueeze(0),  # (1, H, W)
                    # "ch4retrieval": ch4.unsqueeze(0),  # (1, H, W)
                    "ch4_fluxrate": torch.tensor(
                        ch4_fluxrate if ch4_fluxrate is not None else 0.0, device=self.device
                    ),
                    "angle_rotation": angle,
                }
            )

        return task

    def plot_item(
        self,
        item,
        sizeimg: int = 4,
        text_prepend: str = "",
        norm_rgb: float = 1.0,
        vmax_ppb: float = 2_000,
        add_sources: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        
        if not self.analysis_mode:
            raise ValueError("plot_item is only available in analysis_mode=True")
        
        nrows = 2
        ncols = 6

        fig, ax = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * sizeimg, nrows * sizeimg),
            sharex=True,
            sharey=True,
            tight_layout=True,
        )

        ax = ax.flatten()

        item_mli = self.dataframe_id_loc_image_indexed.loc[item["id_loc_image"]]

        tiledate = item_mli["tile_date"]
        if isinstance(tiledate, pd.Timestamp):
            tiledate = tiledate.to_pydatetime().replace(tzinfo=timezone.utc)
            tiledate = tiledate.strftime("%Y-%m-%d")
        elif isinstance(tiledate, str):
            tiledate = tiledate[:10]

        text_prepend = text_prepend.rstrip()
        fluxrate = item["ch4_fluxrate"].item() / 1000
        text = f"{text_prepend} {item_mli['country']} {item_mli['location_name']} {item_mli['satellite']} {tiledate} isplume: {item['isplume']} simulated: {item['simulated']} fluxrate: {fluxrate:.1f} t/h {item['id_loc_image']}"

        input_data = item["y_context_ls0_0"]

        nbands = 6

        i = 0
        rgb = input_data[(3, 2, 1), ...] / norm_rgb
        rgb = torch.permute(rgb, (1, 2, 0))
        ax[i].imshow(rgb.clip(0, 1))
        ax[i].set_title(r"RGB")

        rgb_bg = input_data[(3 + nbands, 2 + nbands, 1 + nbands), ...]
        rgb_bg = torch.permute(rgb_bg, (1, 2, 0))
        i = 1
        ax[i].imshow(rgb_bg.clip(0, 1))
        ax[i].set_title(r"RGB (Bg)")

        i = 2
        mbmp = input_data[0]
        vmin_mbmp = max(0.92, mbmp.min())
        im = ax[i].imshow(mbmp, cmap="plasma_r", vmax=1, vmin=vmin_mbmp, interpolation="nearest")
        plot.colorbar_next_to(im, ax[i])
        ax[i].set_title(r"MBMP")
        wind.add_wind_to_plot(item["wind"], ax=ax[i])

        i = 3
        ch4 = item["ch4"]
        im = ax[i].imshow(ch4[0], cmap="plasma", vmax=vmax_ppb, vmin=0)
        plot.colorbar_next_to(im, ax[i])
        ax[i].set_title(r"$\Delta$XCH$_4$ (ppb)")
        wind.add_wind_to_plot(item["wind"], ax=ax[i])

        i = 4
        y_target = item["y_target"]
        im = ax[i].imshow(y_target, cmap="magma", vmax=1, vmin=0, interpolation="nearest")
        ax[i].set_title(f"Label {item['isplume']}")

        i = 5

        idx_cloudmask = self.bands_out.index("cloudmask")

        cm = input_data[idx_cloudmask]
        im = ax[i].imshow(cm, cmap="magma", vmax=1, vmin=0, interpolation="nearest")
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
        im = ax[i].imshow(b12_bg / 2, cmap="magma", interpolation="nearest")
        plot.colorbar_next_to(im, ax[i])
        ax[i].set_title(f"B12 Bg")

        i = 8
        # CH4 for weighting
        ch4forweighting = item["ch4forweighting"]
        im = ax[i].imshow(ch4forweighting[0], cmap="plasma", vmax=vmax_ppb, vmin=0)
        plot.colorbar_next_to(im, ax[i])
        ax[i].set_title(r"$\Delta$XCH$_4$ for weighting (ppb)")
        wind.add_wind_to_plot(item["wind"], ax=ax[i])

        i = 9
        # CH4 cropped
        ch4_cropped = ch4[0] * y_target
        im = ax[i].imshow(ch4_cropped, cmap="plasma", vmax=vmax_ppb, vmin=0)
        plot.colorbar_next_to(im, ax[i])
        ax[i].set_title(r"$\Delta$XCH$_4$ (ppb) cropped")
        wind.add_wind_to_plot(item["wind"], ax=ax[i])

        i = 10
        # Add plot if ch4sim
        ch4sim = item["ch4sim"]
        im = ax[i].imshow(ch4sim[0], cmap="plasma", vmax=vmax_ppb, vmin=0)
        plot.colorbar_next_to(im, ax[i])
        ax[i].set_title(r"$\Delta$XCH$_4$ Simulated (ppb)")
        wind.add_wind_to_plot(item["wind"], ax=ax[i])

        i = 11
        # Plot ch4_retrieval_before_sim
        ch4_retrieval_before_sim = item["ch4_retrieval_before_sim"]
        im = ax[i].imshow(ch4_retrieval_before_sim[0], cmap="plasma", vmax=vmax_ppb, vmin=0)
        plot.colorbar_next_to(im, ax=ax[i])
        ax[i].set_title(r"$\Delta$XCH$_4$ Retrieval before sim (ppb)")
        wind.add_wind_to_plot(item["wind"], ax=ax[i])
            

        if (
            add_sources
            and (self.sources_dataframe is not None)
            and (item["id_loc_image"] in self.sources_dataframe.index)
        ):
            locs_in_source = self.sources_dataframe.loc[item["id_loc_image"]]
            pixel_row = locs_in_source["pixel_row"]
            pixel_col = locs_in_source["pixel_col"]
            if self.rotate_data_augmentation:
                angle = item.get("angle_rotation", 0)
                if angle != 0:
                    # np.rot90 with positive k rotates array counter-clockwise but moves pixels clockwise
                    # rotate_pixel_coordinates expects counter-clockwise, so negate the angle
                    pixel_row, pixel_col = plumesimulation.rotate_pixel_coordinates(
                        pixel_row,
                        pixel_col,
                        center=(item["y_target"].shape[0] // 2, item["y_target"].shape[1] // 2),
                        angle=-angle,
                    )
            for axs in ax:
                axs.scatter(pixel_col, pixel_row, marker="x", c="red", s=100)

        fig.suptitle(text)
        fig.tight_layout()

        for axs in ax:
            axs.axis("off")

        return fig, ax


def _as_float(value) -> float:
    """Coerce a dataframe field to a plain float, missing values becoming NaN.

    The default collate turns floats into tensors but chokes on ``None``, so a
    missing angle has to arrive as NaN rather than as nothing.
    """
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _as_str(value) -> str:
    """Coerce a dataframe field to a plain string, missing values becoming ``""``.

    Same reason as :func:`_as_float`: the collate handles strings but not ``None``,
    and an empty string is a value the consumer can test.
    """
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _wind_value(wind_val) -> float:
    if pd.isna(wind_val) or wind_val is None or not np.isfinite(wind_val):
        return 4.0

    # clamp value to -20, 20
    wind_val = np.clip(wind_val, -20, 20)

    return wind_val
