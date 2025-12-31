import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple, Union

import fsspec
import numpy as np
import pandas as pd
import rasterio.warp
import rasterio.windows
from georeader import get_utm_epsg, window_utils, read
from georeader.geotensor import GeoTensor
from georeader.abstract_reader import FakeGeoData
from huggingface_hub import hf_hub_url
from rasterio import Affine, warp
from shapely import make_valid, wkt
from shapely.geometry import MultiPolygon, Polygon, shape, Point

from georeader.window_utils import polygon_to_crs
from georeader import get_utm_epsg

from marss2l.huggingface import CSV_PATH_DEFAULT_HF, CSV_PLUME_PATH_DEFAULT_HF, REPO_ID
from marss2l.locations_case_studies import (
    LOCATIONS_CONTROL_RELEASES,
    LOCS_OFFSHORE_ABLATION,
    LOCS_TRAINING_ABLATION,
)
from marss2l.utils import fs_from_path, pathjoin

# Minimal constants required
SPLITS = {
    "all": ["train_2023", "val_2023", "test_2023"],
    "control_releases": [
        "control_releases_train",
        "control_releases_val",
        "control_releases_test",
    ],
    "all_train_test": ["all_train_test_train", "val_2023", "val_2023"],
}

CSV_PATH_DEFAULT = CSV_PATH_DEFAULT_HF
CSV_PLUME_PATH_DEFAULT = CSV_PLUME_PATH_DEFAULT_HF
CSV_LOCSOURCES_PATH_DEFAULT = None 


ALL_DATE_CUT = "2023-12"
MIN_SAMPLES_LOCATION_TRAIN = 30
MIN_SAMPLES_NEGATIVE_TRAIN = 15
N_POS_SIMULATE = 5
COLUMNS_MERGE_PLUMES = [
    "id_loc_image",
    "location_name",
    "ch4path",
    "plumepath",
    "crs",
    "width",
    "height",
    "geotransform",
    "transform_a",
    "transform_b",
    "transform_c",
    "transform_d",
    "transform_e",
    "transform_f",
]

INTERVALS_FLUXRATE = (
    np.array(
        [
            -0.001,
            0,
            500,
            1000,
            1500,
            2000,
            2500,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10_000,
            15_000,
            20_000,
            999_000,
        ]
    )
    / 1_000
)

COUNTRIES_CASE_STUDIES = [
    "United States of America",
    "Turkmenistan",
    "Algeria",
    "Libya",
    "Iran (Islamic Republic of)",
    "Syrian Arab Republic",
    "Egypt",
    "Iraq",
    "Venezuela",
    "Offshore",
]

COUNTRIES_ARABIAN_PENINSULA = [
    "Saudi Arabia",
    "Kuwait",
    "Bahrain",
    "Oman",
    "Qatar",
    "Yemen",
    "United Arab Emirates",
]
UZB_AND_KAZAKH = ["Kazakhstan", "Uzbekistan"]

ORDER_CASE_STUDIES = [
    "United States of America",
    "Turkmenistan",
    "Algeria",
    "Libya",
    "Arabian peninsula",
    "Uzbekistan & Kazakhstan",
    "Iran (Islamic Republic of)",
    "Syrian Arab Republic",
    "Egypt",
    "Iraq",
    "Venezuela",
    "Offshore",
    "Rest",
]


PolygonorMultiPolygonOrStr = Union[Polygon, MultiPolygon, str]


def compute_area(geometry, crs_geometry: str = "EPSG:4326") -> float:
    """
    Compute the area of a geometry in square meters by projecting it to UTM.

    Args:
        geometry (Geometry): geometry to compute the area
        crs_geometry (_type_, optional): CRS of the geometry. Defaults to "EPSG:4326".

    Returns:
        float: area of the geometry in square meters
    """
    center = geometry.centroid
    utm_crs = get_utm_epsg(center)
    geometry_utm = shape(rasterio.warp.transform_geom(crs_geometry, utm_crs, geometry))
    return geometry_utm.area

def get_window(
    geometry: MultiPolygon, img: GeoTensor | FakeGeoData
) -> rasterio.windows.Window:
    window_plume = read.window_from_polygon(
        img, geometry, crs_polygon="EPSG:4326", window_surrounding=True
    )
    window_plume = window_utils.round_outer_window(window_plume)
    window_plume = rasterio.windows.intersection(
        window_plume,
        rasterio.windows.Window(col_off=0, row_off=0, width=img.width, height=img.height),
    )

    return window_plume


WINDOWS_KEYS = ["window_row_off", "window_col_off", "window_height", "window_width"]
def window_to_dict(window_plume: rasterio.windows.Window) -> dict[str, int]:
    item_wd = {}
    item_wd["window_row_off"] = window_plume.row_off
    item_wd["window_col_off"] = window_plume.col_off
    item_wd["window_height"] = window_plume.height
    item_wd["window_width"] = window_plume.width
    return item_wd


def plumes_good_overlap(
    plume: PolygonorMultiPolygonOrStr,
    footprint: PolygonorMultiPolygonOrStr,
    min_intersection_ratio: float = 0.5,
) -> bool:
    """
    Calculate the percentage of the plume area that overlaps with the footprint.
    This is the area of the intersection divided by the area of the plume.

    Args:
        plume (PolygonorMultiPolygonOrStr): The plume geometry, which can be a Polygon, MultiPolygon, or WKT string.
        footprint (PolygonorMultiPolygonOrStr): The footprint geometry, which can be a Polygon, MultiPolygon, or WKT string.

    Returns:
        float: The percentage of the plume area that overlaps with the footprint.
             It returns -1 if the plume is empty, 0 if there is no intersection,
             and 1 if the footprint completely contains the plume.
    """
    if isinstance(plume, str):
        plume = make_valid(wkt.loads(plume))
    if isinstance(footprint, str):
        footprint = make_valid(wkt.loads(footprint))
    if plume.is_empty:
        return True
    if not plume.intersects(footprint):
        return False
    if footprint.contains(plume):
        return True
    intersection = plume.intersection(footprint)

    if (intersection.area / plume.area) >= min_intersection_ratio:
        return True

    area = compute_area(intersection)
    if area > (200 * 10 * 10):
        return True
    # TODO if the origin of the plume is in the image and
    # there are 200 pixels in the footprint it might not need to be filtered

    return False


def set_interval_fluxrate(
    dataframe: pd.DataFrame,
    interval_fluxrate: np.array = INTERVALS_FLUXRATE,
) -> pd.DataFrame:
    dataframe["ch4_fluxrate_th"] = dataframe["ch4_fluxrate"] / 1000
    dataframe["interval_ch4_fluxrate"] = pd.cut(
        dataframe.ch4_fluxrate_th, interval_fluxrate, include_lowest=True
    )
    dataframe["interval_ch4_fluxrate_str"] = dataframe["interval_ch4_fluxrate"].apply(
        lambda x: str(x).replace("(-0.001001", "[0").replace(".0", "")
    )
    dataframe["interval_ch4_fluxrate_str"] = dataframe["interval_ch4_fluxrate_str"].apply(
        lambda x: x.replace("(20, 999]", ">20")
    )
    return dataframe


def _set_case_study(country: str) -> str:
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


def make_valid_load(geom: str) -> Union[Polygon, MultiPolygon]:
    """Make valid a geometry given as WKT string and return as Polygon or MultiPolygon."""
    geometry = wkt.loads(geom)
    return make_valid(geometry)


def compute_footprint(row: pd.Series, crs="EPSG:4326") -> MultiPolygon:
    pol = window_utils.window_polygon(
        rasterio.windows.Window(row_off=0, col_off=0, height=row.height, width=row.width),
        row.geotransform,
    )
    if (crs is None) or window_utils.compare_crs(row.crs, crs):
        return pol

    return window_utils.polygon_to_crs(pol, row.crs, crs)

def distance_source_to_plume(row: pd.Series | dict[str, Any]) -> float:
    """
    Compute the distance in meters from the source point to the nearest point of the plume.    

    Args:
        row (pd.Series | dict[str, Any]): Row with the data. It must contain the following fields:
            - lon: Longitude of the source point
            - lat: Latitude of the source point
            - geometry: Geometry of the plume (Polygon or MultiPolygon)


    Returns:
        float: Distance in meters from the source point to the nearest point of the plume.
    """
    point_source = Point(row["lon"], row["lat"])
    if row["geometry"].contains(point_source):
        return 0.0
    
    crs_utm = get_utm_epsg((row["lon"], row["lat"]), "EPSG:4326")
    polygon_utm = polygon_to_crs(row["geometry"], crs_polygon="EPSG:4326", dst_crs=crs_utm)
    point_source_utm = polygon_to_crs(point_source, crs_polygon="EPSG:4326", dst_crs=crs_utm)
    distance_m = polygon_utm.distance(point_source_utm)
    return distance_m


def read_csv_images(
    csv_path: str = CSV_PATH_DEFAULT,
    fs: Optional[fsspec.AbstractFileSystem] = None,
    add_columns_for_analysis: bool = False,
    split: Optional[str] = None,
    add_case_study: bool = False,
    add_loc_type: bool = False,
    path_prepend_data: Optional[str] = None,
    recompute_footprint: bool = False,
    recompute_windows: bool = False,
) -> pd.DataFrame:
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
        path_prepend_data (str, optional): Path to prepend to the s2path, plumepath, cloudmaskpath and ch4path columns. Defaults to None.
            If None, it does not prepend anything. This field is required for reading the data downloaded from HuggingFace.
        recompute_footprint (bool, optional): Whether to recompute the footprint. Defaults to False.
        recompute_windows (bool, optional): Whether to recompute the window where the plume lies in the image. Defaults to False.

    Returns:
        pd.DataFrame:
    """
    if fs is None:
        fs = fs_from_path(csv_path)

    # Load CSV file
    assert csv_path.startswith("https://") or fs.exists(
        csv_path
    ), f"Path {csv_path} does not exist in filesystem {fs}"

    with fs.open(csv_path) as f:
        dataframe = pd.read_csv(f)

    # Derive columns
    # self.dataframe["tile_date"] = pd.to_datetime(self.dataframe["tile_date"])
    dataframe["plume"] = dataframe["plume"].apply(make_valid_load)
    # Add geotransform
    dataframe["geotransform"] = dataframe.apply(
        lambda row: Affine(
            row.transform_a,
            row.transform_b,
            row.transform_c,
            row.transform_d,
            row.transform_e,
            row.transform_f,
        ),
        axis=1,
    )
    if recompute_footprint or "footprint" not in dataframe.columns:
        dataframe["footprint"] = dataframe.apply(compute_footprint, axis=1)
    else:
        dataframe["footprint"] = dataframe["footprint"].apply(make_valid_load)
    
    if recompute_windows or not all(k in dataframe.columns for k in WINDOWS_KEYS):
        windows_series = dataframe.apply(
            lambda row: window_to_dict(get_window(row["plume"], fake_reader(row))) if row.isplume else None,
            axis=1,
        )
        for k in WINDOWS_KEYS:
            dataframe[k] = windows_series.apply(lambda w: w[k] if w is not None else None)

    dataframe["plumes_good_overlap"] = dataframe.apply(
        lambda row: plumes_good_overlap(row["plume"], row["footprint"]), axis=1
    )

    # Remove rows with percent_overlap > -1 and < 0.5
    if not dataframe["plumes_good_overlap"].all():
        n_high_overlap = dataframe["plumes_good_overlap"].sum()
        n_total = dataframe.shape[0]
        dataframe = dataframe[dataframe["plumes_good_overlap"]].copy()
        print(f"Removed {n_total - n_high_overlap} rows with percent_overlap < 0.5")

    dataframe["year"] = dataframe["tile_date"].apply(lambda x: int(x[:4]))
    dataframe["year_month"] = dataframe["tile_date"].apply(lambda x: x[:7])
    dataframe["tile_date"] = dataframe["tile_date"].apply(lambda x: datetime.fromisoformat(x))
    dataframe["year_month_day"] = dataframe["tile_date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    dataframe["wind_speed"] = dataframe.apply(
        lambda row: math.sqrt(row.wind_u**2 + row.wind_v**2), axis=1
    )
    dataframe["isplumeneg"] = ~dataframe.isplume
    if CSV_PATH_DEFAULT_HF == csv_path:
        # Convert the paths (s2path, plumepath, cloudmaskpath, ch4path) to HuggingFace URLs
        dataframe["s2path"] = dataframe["s2path"].apply(
            lambda x: hf_hub_url(repo_id=REPO_ID, filename=x, repo_type="dataset")
        )
        dataframe["plumepath"] = dataframe["plumepath"].apply(
            lambda x: (
                hf_hub_url(repo_id=REPO_ID, filename=x, repo_type="dataset")
                if ((x is not None) and (not pd.isna(x)))
                else None
            )
        )
        dataframe["cloudmaskpath"] = dataframe["cloudmaskpath"].apply(
            lambda x: hf_hub_url(repo_id=REPO_ID, filename=x, repo_type="dataset")
        )
        dataframe["ch4path"] = dataframe["ch4path"].apply(
            lambda x: (
                hf_hub_url(repo_id=REPO_ID, filename=x, repo_type="dataset")
                if ((x is not None) and (not pd.isna(x)))
                else None
            )
        )

    elif path_prepend_data is not None:
        dataframe["s2path"] = dataframe["s2path"].apply(lambda x: pathjoin(path_prepend_data, x))
        dataframe["plumepath"] = dataframe["plumepath"].apply(
            lambda x: pathjoin(path_prepend_data, x) if ((x is not None) and (not pd.isna(x))) else None
        )
        dataframe["cloudmaskpath"] = dataframe["cloudmaskpath"].apply(
            lambda x: pathjoin(path_prepend_data, x)
        )
        dataframe["ch4path"] = dataframe["ch4path"].apply(
            lambda x: pathjoin(path_prepend_data, x) if ((x is not None) and (not pd.isna(x))) else None
        )

    # Set country to Offshore if offshore is True
    dataframe["country"] = dataframe.apply(
        lambda row: "Offshore" if row.offshore else row.country, axis=1
    )

    if add_columns_for_analysis:
        dataframe["id_loc_image"] = dataframe["id_loc_image"].map(lambda x: uuid.UUID(x))
        dataframe["last_update"] = dataframe["last_update"].apply(
            lambda x: datetime.fromisoformat(x)
        )
        dataframe["date"] = dataframe["tile_date"].apply(lambda x: x.strftime("%Y-%m-%d"))
        dataframe["year_month"] = dataframe["tile_date"].apply(
            lambda x: datetime.strptime(x.strftime("%Y-%m-01"), "%Y-%m-%d")
        )
        dataframe["satellite_constellation"] = dataframe.satellite.apply(
            lambda x: "Sentinel-2" if x in ["S2A", "S2B", "S2C"] else "Landsat"
        )
        dataframe["year_quarter"] = dataframe["tile_date"].apply(
            lambda x: f"{x.year}-{(x.month-1)//3 + 1}Q"
        )

        dataframe["ch4_fluxrate_th"] = dataframe["ch4_fluxrate"] / 1000
        dataframe["interval_ch4_fluxrate"] = pd.cut(
            dataframe.ch4_fluxrate_th,
            INTERVALS_FLUXRATE,  # 30_000, 50_000,
            include_lowest=True,
        )

        dataframe["interval_ch4_fluxrate_str"] = dataframe["interval_ch4_fluxrate"].apply(
            lambda x: str(x).replace("(-0.001001", "[0").replace(".0", "")
        )
        dataframe["interval_ch4_fluxrate_str"] = dataframe["interval_ch4_fluxrate_str"].apply(
            lambda x: x.replace("(20, 999]", ">20")
        )

    if split is not None:
        train_split, val_split, test_split = SPLITS[split]
        dataframe_data_traintest_indexed = dataframe.set_index("id_loc_image").copy()
        dataframe_data_traintest_indexed["split_name"] = "Not Used"
        for split_name, traintestval_split in zip(
            [train_split, val_split, test_split], ["train", "val", "test"]
        ):
            df_splitted, _, _ = load_dataframe_split(
                dataframe_or_csv_path=dataframe,
                split=split_name,
                fs=fs,
                load_plumes=False,
            )
            ids_split = df_splitted.id_loc_image
            if not (
                dataframe_data_traintest_indexed.loc[ids_split, "split_name"] == "Not Used"
            ).all():
                splits_overlap = dataframe_data_traintest_indexed.loc[
                    ids_split, "split_name"
                ].unique()
                splits_overlap = splits_overlap[splits_overlap != "Not Used"].tolist()
                raise ValueError(
                    f"BAD SPLITING!!! {split} {split_name} there is overlap with {splits_overlap}"
                )

            dataframe_data_traintest_indexed.loc[ids_split, "split_name"] = split_name

            dataframe = dataframe_data_traintest_indexed.reset_index()

        if add_loc_type:
            # Add column with location type
            summaries_by_loc = (
                dataframe.groupby(["split_name", "location_name"])["isplume"]
                .agg(["count", "sum"])
                .rename(columns={"count": "nimages", "sum": "nplumes"})
                .reset_index()
            )
            summaries_by_loc["loc_type"] = (
                (summaries_by_loc["nimages"] >= MIN_SAMPLES_LOCATION_TRAIN)
                & (summaries_by_loc["nplumes"] >= N_POS_SIMULATE)
                & (
                    (summaries_by_loc["nimages"] - summaries_by_loc["nplumes"])
                    >= MIN_SAMPLES_NEGATIVE_TRAIN
                )
            )
            summaries_by_loc_train = summaries_by_loc[summaries_by_loc.split_name == train_split]
            locs_film = set(
                summaries_by_loc_train.loc[
                    summaries_by_loc_train["loc_type"], "location_name"
                ].values
            )
            locs_train = set(summaries_by_loc_train["location_name"].values)
            dataframe["loc_type"] = dataframe.location_name.apply(
                lambda x: (
                    "FiLM" if x in locs_film else "few samples" if x in locs_train else "no samples"
                )
            )

    if add_case_study:
        # from marss2l import locations_case_studies
        # dataframe["case_study"] = dataframe["location_name"].apply(lambda x: locations_case_studies.REV_CASE_STUDIES.get(x, "None"))
        dataframe["case_study"] = dataframe["country"].apply(_set_case_study)

    return dataframe


read_csv = read_csv_images  # for backward compatibility


def read_csv_plumes(
    csv_path: str = CSV_PLUME_PATH_DEFAULT,
    fs: Optional[fsspec.AbstractFileSystem] = None,
    recompute_is_detached: bool = False,
) -> pd.DataFrame:
    """
    Read the CSV file with the plumes.

    Args:
        csv_path (str): Path to the CSV
        fs (fsspec.AbstractFileSystem, optional): Filesystem to use. Defaults to None.
        recompute_is_detached (bool, optional): Whether to recompute the is_detached column. A plume is detached if the 
            distance from the origin to the nearest point of the plume is greater than 200m.
            Defaults to False.

    Returns:
        pd.DataFrame: Dataframe with the plumes
    """
    if fs is None:
        fs = fs_from_path(csv_path)
    with fs.open(csv_path) as f:
        dataframe_plumes = pd.read_csv(f)

    # Add aux columns
    dataframe_plumes["year"] = dataframe_plumes["tile_date"].apply(lambda x: int(x[:4]))
    dataframe_plumes["year_month"] = dataframe_plumes["tile_date"].apply(lambda x: x[:7])
    dataframe_plumes["tile_date"] = dataframe_plumes["tile_date"].apply(
        lambda x: datetime.fromisoformat(x)
    )
    dataframe_plumes["wind_speed"] = dataframe_plumes.apply(
        lambda row: math.sqrt(row.wind_u**2 + row.wind_v**2), axis=1
    )
    dataframe_plumes["geometry"] = dataframe_plumes["geometry"].apply(make_valid_load)

    if recompute_is_detached or "is_detached" not in dataframe_plumes.columns:
        dataframe_plumes["is_detached"] = dataframe_plumes.apply(
            lambda row: distance_source_to_plume(row) > 200, axis=1
        )    
    
    # from shapely.ops import nearest_points
    # nearest_point_on_polygon, _ = nearest_points(polygon, point)

    return dataframe_plumes


PIXEL_WARNING_THRESHOLD = 5  # Only warn if pixel index is off by more than 5


class LonLatOutOfImageException(Exception):
    """Exception raised when lon/lat coordinates fall outside the image bounds."""

    pass


def get_pixel_coordinates_from_lonlat(
    lon: float,
    lat: float,
    transform: Affine,
    crs_out: str,
    width: int,
    height: int,
    string_id_for_logs: str,
    logger: logging.Logger,
    raise_if_out_of_image: bool = False,
) -> Tuple[int, int]:
    """
    Compute pixel coordinates from lon/lat using the provided geotransform and CRS.

    Parameters
    ----------
    lon : float
        Longitude of the point.
    lat : float
        Latitude of the point.
    transform : Affine
        Geotransform of the image.
    crs_out : str
        Coordinate reference system of the image.
    width : int
        Width of the image in pixels.
    height : int
        Height of the image in pixels.
    string_id_for_logs : str
        Identifier string for logging purposes.
    logger : logging.Logger
        Logger for warnings about out-of-bounds pixel indices.
    raise_if_out_of_image : bool, default False
        If True, raise LonLatOutOfImageException when coordinates fall outside image bounds
        instead of clipping to valid range.

    Returns:
    -------
    Tuple[int, int]
        Pixel row and column indices.

    Raises:
    ------
    LonLatOutOfImageException
        If raise_if_out_of_image is True and the coordinates fall outside the image bounds.
    """

    coords_transformed = warp.transform("EPSG:4326", crs_out, [lon], [lat])
    x, y = coords_transformed[0][0], coords_transformed[1][0]
    col, row = ~transform * (x, y)
    col = int(round(col))
    row = int(round(row))
    # Only warn if discrepancy is greater than threshold
    if col < 0:
        if raise_if_out_of_image:
            raise LonLatOutOfImageException(
                f"Column index {col} is out of bounds for {string_id_for_logs}"
            )
        if col < -PIXEL_WARNING_THRESHOLD:
            logger.warning(
                f"Column index {col} is out of bounds for {string_id_for_logs}. Setting to 0."
            )
        col = 0

    if row < 0:
        if raise_if_out_of_image:
            raise LonLatOutOfImageException(
                f"Row index {row} is out of bounds for {string_id_for_logs}"
            )
        if row < -PIXEL_WARNING_THRESHOLD:
            logger.warning(
                f"Row index {row} is out of bounds for {string_id_for_logs}. Setting to 0."
            )
        row = 0

    if col >= width:
        if raise_if_out_of_image:
            raise LonLatOutOfImageException(
                f"Column index {col} is out of bounds for {string_id_for_logs}"
            )
        if col >= width + PIXEL_WARNING_THRESHOLD:
            logger.warning(
                f"Column index {col} is out of bounds for {string_id_for_logs}. Setting to {width - 1}."
            )
        col = width - 1

    if row >= height:
        if raise_if_out_of_image:
            raise LonLatOutOfImageException(
                f"Row index {row} is out of bounds for {string_id_for_logs}"
            )
        if row >= height + PIXEL_WARNING_THRESHOLD:
            logger.warning(
                f"Row index {row} is out of bounds for {string_id_for_logs}. Setting to {height - 1}."
            )
        row = height - 1

    return row, col


def process_images_and_plumes(
    dataframe_plumes: pd.DataFrame,
    dataframe_images: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
    recompute_windows: bool = False,
) -> pd.DataFrame:
    """
    Merge plume and image dataframes and compute pixel coordinates for plume origins.

    This function merges the plumes dataframe with selected columns from the images dataframe
    using the 'id_loc_image' key. It then computes the pixel row and column indices for each plume
    origin (lon, lat) using the image geotransform and CRS, and adds these as 'pixel_row' and 'pixel_col'
    columns to the plumes dataframe.

    Parameters
    ----------
    dataframe_plumes : pd.DataFrame
        DataFrame containing plume records, must include 'lon', 'lat', 'id_loc_image', and geospatial columns.
    dataframe_images : pd.DataFrame
        DataFrame containing image records, must include columns listed in COLUMNS_MERGE_PLUMES.
    logger : logging.Logger, optional
        Logger for warnings about out-of-bounds pixel indices.
    recompute_windows : bool, optional
        Whether to recompute the window where the plume lies in the image. Defaults to False.

    Returns:
    -------
    pd.DataFrame
        Plumes dataframe with columns `COLUMNS_MERGE_PLUMES` from the images dataframe and
        added 'pixel_row' and 'pixel_col' columns.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    dataframe_plumes = pd.merge(
        dataframe_plumes, dataframe_images[COLUMNS_MERGE_PLUMES], on="id_loc_image"
    )

    pixel_rows = []
    pixel_cols = []
    for _, row_df in dataframe_plumes.iterrows():
        transform = row_df["geotransform"]
        crs_out = row_df["crs"]
        width = row_df["width"]
        height = row_df["height"]
        lon = row_df["lon"]
        lat = row_df["lat"]
        string_id_for_logs = f"id_plume: {row_df['id_plume']} id_image: {row_df['id_loc_image']}"
        row, col = get_pixel_coordinates_from_lonlat(
            lon=lon,
            lat=lat,
            transform=transform,
            crs_out=crs_out,
            width=width,
            height=height,
            string_id_for_logs=string_id_for_logs,
            logger=logger,
        )

        pixel_rows.append(row)
        pixel_cols.append(col)

    # TODO filter out/control detached plumes?
    # TODO maybe pass a flag to load only suitable plumes for simulation?

    dataframe_plumes["pixel_row"] = pixel_rows
    dataframe_plumes["pixel_col"] = pixel_cols

    if recompute_windows or not all(
            k in dataframe_plumes.columns for k in WINDOWS_KEYS
        ):
        windows_series = dataframe_plumes.apply(
            lambda row: window_to_dict(get_window(row["geometry"], fake_reader(row))),
            axis=1,
        )
        for k in WINDOWS_KEYS:
            dataframe_plumes[k] = windows_series.apply(lambda w: w[k])

    return dataframe_plumes


def read_csv_locs_sources(
    csv_path: str = CSV_LOCSOURCES_PATH_DEFAULT, fs: Optional[fsspec.AbstractFileSystem] = None
) -> pd.DataFrame:
    """
    This function reads the dataframe with the list of sources for each location.

    This dataframe has the following columns:
    * source_name: Name of the source
    * lat: Latitude of the source
    * lon: Longitude of the source
    * id_mars_source: UUID of the source in the MARS database
    * location_name: Name of the location
    * sector: Sector of the source
    * source_type: Type of the source
    * geometry: Geometry of the location (shapely Polygon or MultiPolygon)
    * country: Country of the source
    * id_location: UUID of the location in the MARS database.

    Args:
        csv_path (str, optional): Path to the CSV file. Defaults to CSV_LOCSOURCES_PATH_DEFAULT.
        fs (Optional[fsspec.AbstractFileSystem], optional): Filesystem to use. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with the list of sources for each location.
    """

    if fs is None:
        fs = fs_from_path(csv_path)
    with fs.open(csv_path) as f:
        dataframe_sources = pd.read_csv(f)
    return dataframe_sources


def process_images_and_sources(
    dataframe_sources: pd.DataFrame,
    dataframe_images: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    This function joins the images and source dataframe by id_location and compute the pixel coordinates
    of each of the sources in the images.

    The output dataset has the following columns:
    * id_loc_image: UUID of the image location
    * id_location: UUID of the source location
    * id_mars_source: UUID of the source in the MARS database
    * pixel_row: Pixel row of the source in the image
    * pixel_col: Pixel column of the source in the image
    * lon: Longitude of the source
    * lat: Latitude of the source
    * crs: CRS of the image
    * geotransform: Geotransform of the image
    * width: Width of the image
    * height: Height of the image

    Sources that fall outside image bounds are filtered out.

    Args:
        dataframe_sources (pd.DataFrame): Dataframe with the list of sources for each location.
        dataframe_images (pd.DataFrame): Dataframe with the list of images.
        logger (Optional[logging.Logger], optional): Logger to use. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with the joined images and sources (excluding out-of-bounds sources).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    cols_image_df = ["id_loc_image", "id_location", "crs", "geotransform", "width", "height"]
    cols_sources_df = ["id_location", "id_mars_source", "lon", "lat"]

    dataframe_merged = pd.merge(
        dataframe_sources[cols_sources_df], dataframe_images[cols_image_df], on="id_location"
    )

    pixel_rows = []
    pixel_cols = []
    valid_indices = []

    for idx, row_df in dataframe_merged.iterrows():
        transform = row_df["geotransform"]
        crs_out = row_df["crs"]
        width = row_df["width"]
        height = row_df["height"]
        lon = row_df["lon"]
        lat = row_df["lat"]
        string_id_for_logs = f"id_location: {row_df['id_location']} id_image: {row_df['id_loc_image']} id_mars_source: {row_df['id_mars_source']}"

        try:
            row, col = get_pixel_coordinates_from_lonlat(
                lon=lon,
                lat=lat,
                transform=transform,
                crs_out=crs_out,
                width=width,
                height=height,
                string_id_for_logs=string_id_for_logs,
                logger=logger,
                raise_if_out_of_image=True,
            )
            pixel_rows.append(row)
            pixel_cols.append(col)
            valid_indices.append(idx)
        except LonLatOutOfImageException as e:
            logger.debug(f"Skipping source outside image bounds: {e}")
            continue

    # Filter to only valid indices
    dataframe_merged = dataframe_merged.loc[valid_indices].copy()
    dataframe_merged["pixel_row"] = -1
    dataframe_merged["pixel_col"] = -1
    dataframe_merged.loc[valid_indices, "pixel_row"] = pixel_rows
    dataframe_merged.loc[valid_indices, "pixel_col"] = pixel_cols

    return dataframe_merged


def split_control_releases(
    dataframe: pd.DataFrame, split: str, logger: Optional[logging.Logger] = None
) -> pd.Series:
    """
    Get boolean mask for control releases split.

    For location 'Standford_controlled_releases' the split is:
    - train: 2020-01-01 to 2022-01-01
    - val: 2022-01-01 to 2022-10-01
    - test: 2022-10-10 to 2022-11-28

    For location 'Standford_controlled_releases_2021' the split is:
    - train: 2020-01-01 to 2021-10-01
    - val: 2022-01-01 to 2022-10-01
    - test: 2021-10-19 to 2021-11-03
    
    Returns:
        pd.Series: Boolean mask for the split
    """
    control_releases_images = dataframe.location_name.isin(LOCATIONS_CONTROL_RELEASES)

    if split == "control_releases_train":
        stanford_2020 = (
            dataframe.location_name == "Standford_controlled_releases"
        ) & dataframe.tile_date.between(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2022, 1, 1, tzinfo=timezone.utc),
            inclusive="left",
        )
        stanford_2021 = (
            dataframe.location_name == "Standford_controlled_releases_2021"
        ) & dataframe.tile_date.between(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2021, 10, 1, tzinfo=timezone.utc),
            inclusive="left",
        )
        split_data = control_releases_images & (stanford_2020 | stanford_2021)

    elif split == "control_releases_val":
        split_data = control_releases_images & dataframe.tile_date.between(
            datetime(2022, 1, 1, tzinfo=timezone.utc),
            datetime(2022, 10, 1, tzinfo=timezone.utc),
            inclusive="left",
        )
    elif split == "control_releases_test":
        stanford_2020_test = (
            dataframe.location_name == "Standford_controlled_releases"
        ) & dataframe.tile_date.between(
            datetime(2022, 10, 10, tzinfo=timezone.utc),
            datetime(2022, 11, 28, 23, 59, 59, tzinfo=timezone.utc),
            inclusive="both",
        )
        stanford_2021_test = (
            dataframe.location_name == "Standford_controlled_releases_2021"
        ) & dataframe.tile_date.between(
            datetime(2021, 10, 19, tzinfo=timezone.utc),
            datetime(2021, 11, 3, 23, 59, 59, tzinfo=timezone.utc),
            inclusive="both",
        )
        split_data = control_releases_images & (stanford_2020_test | stanford_2021_test)

    else:
        raise ValueError(
            f"Unknown split {split}. Expected 'control_releases_train', 'control_releases_val', 'control_releases_test'"
        )

    return split_data


def load_dataframe_split(
    split: str,
    dataframe_or_csv_path: Union[str, pd.DataFrame] = CSV_PATH_DEFAULT,
    dataframe_or_csv_path_plumes: Optional[Union[str, pd.DataFrame]] = None,
    dataframe_or_csv_path_sources: Optional[Union[str, pd.DataFrame]] = None,
    fs: Optional[fsspec.AbstractFileSystem] = None,
    logger: Optional[logging.Logger] = None,
    load_plumes: bool = True,
    all_locs: Optional[List[str]] = None,
    only_onshore: bool = False,
    only_offshore: bool = False,
    smoke_test: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the image dataframe and (optionally) the plume dataframe, then apply the requested split.

    This function supports loading from CSV file paths (local or remote, e.g. "az://...") or from
    pre-loaded pandas DataFrames. It returns the split image dataframe and, if requested, the
    corresponding plume dataframe.

    Parameters
    ----------
    split : str
        Name of the split to apply. Supported values include:
        - "train_2023", "val_2023", "test_2023"
        - "all_train_test_train", "all_train_test_val", "all_train_test_test"
        - "no split" (returns all data)
        - "control_releases_train", "control_releases_val", "control_releases_test"
    dataframe_or_csv_path : str or pd.DataFrame, default CSV_PATH_DEFAULT
        Path to the images CSV file or a pre-loaded DataFrame of images.
    dataframe_or_csv_path_plumes : str or pd.DataFrame or None, optional
        Path to the plumes CSV file or a pre-loaded DataFrame of plumes.
        - If `load_plumes=True` and `dataframe_or_csv_path` is a DataFrame,
          you MUST provide this argument (as a DataFrame or path).
        - If `load_plumes=False`, this argument is ignored.
    dataframe_or_csv_path_sources : str or pd.DataFrame or None, optional
        Path to the sources CSV file or a pre-loaded DataFrame of sources.
    fs : fsspec.AbstractFileSystem, optional
        Filesystem for loading remote CSVs. If None, a suitable filesystem is inferred.
    logger : logging.Logger, optional
        Logger for messages.
    load_plumes : bool, default True
        If True, load and return the plumes dataframe (second tuple element).
        If False, plumes are not loaded and the function returns (images_df, None).
    all_locs : list[str], optional
        If provided, only keep records whose location_name is in this list.
    only_onshore : bool, default False
        If True, only keep onshore locations.
    only_offshore : bool, default False
        If True, only keep offshore locations.
    smoke_test : bool, default False
        If True, only load a small subset of the data for quick testing.

    Returns
    -------
    tuple (pd.DataFrame, pd.DataFrame or None)
        (images_dataframe_after_split, plumes_dataframe_after_split_or_None, sources_dataframe_after_split_or_None)

    Usage Notes
    -----------
    - If you pass pre-loaded DataFrames for images and plumes, set `load_plumes=True` and provide both.
    - If you want only images, set `load_plumes=False` (second return value will be None).
    - If `split` starts with "control_releases", the function returns the control release split and
      a plume dataframe (or None) as appropriate.
    - If `all_locs` is provided and none of the requested locations exist in the filtered dataset,
      a ValueError is raised.

    """
    # only_onshore and only_offshore are mutually exclusive
    assert not (
        only_onshore and only_offshore
    ), "only_onshore and only_offshore cannot be both True"

    if logger is None:
        logger = logging.getLogger(__name__)

    if isinstance(dataframe_or_csv_path, str):
        csv_path = dataframe_or_csv_path
        dataframe_images = read_csv_images(csv_path, fs)
        if dataframe_or_csv_path_plumes is None:
            # dataframe_or_csv_path_plumes = dataframe_or_csv_path.replace("validated_images_all.csv", "validated_images_plumes.csv")
            dataframe_or_csv_path_plumes = CSV_PLUME_PATH_DEFAULT
            if fs is None:
                fs = fs_from_path(dataframe_or_csv_path_plumes)
            assert dataframe_or_csv_path_plumes.startswith("https://") or fs.exists(
                dataframe_or_csv_path_plumes
            ), f"Path {dataframe_or_csv_path_plumes} does not exist. Should contain the csv with the plumes if not provided."
    else:
        dataframe_images = dataframe_or_csv_path.copy()
        assert (not load_plumes) or (
            dataframe_or_csv_path_plumes is not None
        ), "csv_path_plumes should be provided if dataframe_or_csv_path is a DataFrame and load_plumes is True"

    # Subset if only_onshore or only_offshore
    if only_onshore:
        dataframe = dataframe[~dataframe.offshore].copy()
        logger.info(
            f"Keep only onshore locations in the dataset. There are {len(dataframe['location_name'].unique())} locations"
        )
    elif only_offshore:
        dataframe = dataframe[dataframe.offshore].copy()
        logger.info(
            f"Keep only offshore locations in the dataset. There are {len(dataframe['location_name'].unique())} locations"
        )

    # Load plumes dataframe
    dataframe_plumes = None
    if load_plumes:
        assert dataframe_or_csv_path_plumes is not None, "csv_path_plumes should be provided"

        if isinstance(dataframe_or_csv_path_plumes, str):
            dataframe_plumes = read_csv_plumes(dataframe_or_csv_path_plumes, fs)
        else:
            dataframe_plumes = dataframe_or_csv_path_plumes.copy()

    dataframe_sources = None
    if dataframe_or_csv_path_sources is not None:
        if isinstance(dataframe_or_csv_path_sources, str):
            dataframe_sources = read_csv_locs_sources(dataframe_or_csv_path_sources, fs)
        else:
            dataframe_sources = dataframe_or_csv_path_sources.copy()
    
    # Keep only data from 2018 on
    data_pre_2018 = dataframe_images["year"] < 2018
    if data_pre_2018.any():
        # logger.info(f"Discarding data from years before 2018. There are {data_pre_2018.sum()} samples before 2018")
        dataframe_images = dataframe_images[~data_pre_2018].copy()

    # Make sure to exclude Control release locations
    iscontrolreleasessplit = split.startswith("control_releases")
    locs_control_releases_serie = dataframe_images.location_name.isin(LOCATIONS_CONTROL_RELEASES)
    if not iscontrolreleasessplit and locs_control_releases_serie.any():
        dataframe_images = dataframe_images.loc[~locs_control_releases_serie].copy()

    # Split the data
    if iscontrolreleasessplit:
        if not locs_control_releases_serie.any():
            raise ValueError(f"Locations {LOCATIONS_CONTROL_RELEASES} not found in the dataset")
        split_data =  split_control_releases(dataframe_images, split, logger)
    elif split == "val_2023":
        split_data = (dataframe_images["year"] == 2021) & dataframe_images.location_name.isin(
            LOCS_TRAINING_ABLATION + LOCS_OFFSHORE_ABLATION
        )
    elif split == "train_2023":
        split_data = (dataframe_images["year_month"] < ALL_DATE_CUT) & (
            dataframe_images["year"] != 2021
        )
    elif split == "test_2023":
        split_data = dataframe_images["year_month"] > ALL_DATE_CUT
    elif split == "no split":
        # Split data is all true
        split_data = dataframe_images["year"] > 0
    elif split == "all_train_test_train":
        # train: all train_2023 + test_2023 (i.e. everything except val_2023)
        split_data = dataframe_images["year"] != 2021
    else:
        raise ValueError(
            f"Unknown split {split}. Expected 'train_2023', 'test_2023', 'val_2023', 'all_train_test_train', 'all_train_test_val', 'all_train_test_test'"
        )

    # Subset dataframe_images, dataframe_plumes and dataframe_sources
    dataframe_images_splitted = dataframe_images.loc[split_data].copy()
    if iscontrolreleasessplit:
        # This assumes that for the control releases we will simulate plumes only taken from non-control release images
        dataframe_images_for_plumes = dataframe_images[~locs_control_releases_serie].copy()
    else:
        dataframe_images_for_plumes = dataframe_images_splitted

    # Set self.all_locs and keep only data from all_locs
    if all_locs is not None:
        images_from_loc = dataframe_images_splitted.location_name.isin(all_locs)
        if not images_from_loc.any():
            raise ValueError(
                f"None of the locations in 'all_locs' where found in the dataset in split {split}"
            )

        dataframe_images_splitted = dataframe_images_splitted.loc[images_from_loc].copy()

    if dataframe_plumes is not None:
        dataframe_plumes = process_images_and_plumes(dataframe_plumes, 
                                                     dataframe_images_for_plumes,
                                                     logger=logger)
    if dataframe_sources is not None:
        dataframe_sources = process_images_and_sources(
            dataframe_sources, dataframe_images_splitted, logger=logger
        )
    
    # If smoke_test, keep only 200 samples 100 with plumes and 100 without plumes
    if smoke_test:
        dataframe_images_with_plume = dataframe_images_splitted[
            dataframe_images_splitted.isplume
        ].head(100)
        dataframe_images_without_plume = dataframe_images_splitted[
            ~dataframe_images_splitted.isplume
        ].head(100)
        dataframe_images_splitted = pd.concat(
            [dataframe_images_with_plume, dataframe_images_without_plume], ignore_index=True
        )
        if dataframe_plumes is not None:
            dataframe_plumes = dataframe_plumes[
                dataframe_plumes.id_loc_image.isin(dataframe_images_with_plume.id_loc_image)
            ].copy()
        if dataframe_sources is not None:
            dataframe_sources = dataframe_sources[
                dataframe_sources.id_loc_image.isin(dataframe_images_splitted.id_loc_image)
            ].copy()
        logger.info(
            f"Smoke test: keeping only {len(dataframe_images_splitted)} images, {0 if dataframe_plumes is None else len(dataframe_plumes)} plumes, {0 if dataframe_sources is None else len(dataframe_sources)} sources"
        )

    return dataframe_images_splitted, dataframe_plumes, dataframe_sources


def load_image(
    item: Union[pd.Series, dict[str, Any]],
    key: str,
    fs: Optional[fsspec.AbstractFileSystem] = None,
) -> GeoTensor:
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
    path: str = item[key]
    if fs is None:
        fs = fs_from_path(path)
    if path.endswith(".npy"):
        with fs.open(path, "rb") as f:
            values = np.load(f)
        if len(values.shape) == 3:
            values = np.transpose(values, (2, 0, 1))
            if values.shape[0] == 1:
                values = values[0]
        if "geotransform" in item:
            transform = item["geotransform"]
        else:
            transform = Affine(
                item["transform_a"],
                item["transform_b"],
                item["transform_c"],
                item["transform_d"],
                item["transform_e"],
                item["transform_f"],
            )
        crs = item["crs"]
        gt = GeoTensor(values=values, transform=transform, crs=crs)
        return gt

    gt = GeoTensor.load_file(path, fs=fs)
    if len(gt.shape) == 3 and gt.shape[0] == 1:
        gt.values = gt.values[0]
    return gt


def fake_reader(item) -> FakeGeoData:
    transform = rasterio.Affine(
        item["transform_a"],
        item["transform_b"],
        item["transform_c"],
        item["transform_d"],
        item["transform_e"],
        item["transform_f"],
    )
    return FakeGeoData(
        transform=transform, crs=item["crs"], # width=item["width"], height=item["height"]
        shape=(item["height"], item["width"])
    )
