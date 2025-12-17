import json
import logging
import os
import warnings
from datetime import datetime, timezone
from typing import Optional, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from shapely import Geometry
from shapely.geometry import Point

MAX_DATE_ERA5 = None
FIRST_DATE_NASA_GEOS_FP = datetime(2014, 2, 20, tzinfo=timezone.utc)


def datetime_with_timezone(date_of_acquisition: datetime) -> datetime:
    """
    Add timezone to a datetime object if it doesn't have it.

    Args:
        date_of_acquisition: Date of acquisition.

    Returns:
        Date of acquisition with timezone.
    """
    if date_of_acquisition.tzinfo is None:
        return date_of_acquisition.replace(tzinfo=timezone.utc)
    else:
        return date_of_acquisition.astimezone(timezone.utc)


def download_wind_nasa_geos_fp(
    location: Union[Tuple[float, float], Point],
    date_of_acquisition: datetime,
) -> Tuple[float, float]:
    """
    Download NASA GEOS-FP wind data and interpolate to location.

    Downloads the NASA GEOS-FP product for the given date and interpolates
    the wind components (U10M, V10M) to the specified location.

    Args:
        location: Location as (lon, lat) tuple or shapely Point geometry.
        date_of_acquisition: Date and time for wind data acquisition.

    Returns:
        Tuple of (u10, v10) wind components in m/s at 10m above ground.

    Raises:
        ValueError: If date is before 2014-02-20 or wind values are NaN.
        FileNotFoundError: If download fails.

    Example:
        >>> from datetime import datetime
        >>> location = (-95.5, 29.8)  # lon, lat
        >>> date = datetime(2021, 1, 1, 12, 0)
        >>> u10, v10 = download_wind_nasa_geos_fp(location, date)
    """
    from georeader.readers import download_utils

    # Extract coordinates from Point if needed
    if isinstance(location, Point):
        lon, lat = location.x, location.y
    else:
        lon, lat = location

    # Add timezone if not present
    date_of_acquisition = datetime_with_timezone(date_of_acquisition)

    # Check date validity
    if date_of_acquisition <= FIRST_DATE_NASA_GEOS_FP:
        raise ValueError(
            f"NASA GEOS-FP data not available before 2014-02-20. Got {date_of_acquisition}."
        )

    # Build URL
    url = (
        f"https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/"
        f"Y{date_of_acquisition.year}/M{date_of_acquisition.month:02d}/"
        f"D{date_of_acquisition.day:02d}/"
        f"GEOS.fp.asm.tavg1_2d_slv_Nx.{date_of_acquisition.strftime('%Y%m%d_%H')}30.V01.nc4"
    )

    # Download file
    filename = os.path.basename(url)
    warnings.filterwarnings(
        "ignore", message="Adding certificate verification is strongly advised."
    )
    filelocal = download_utils.download_product(url, filename=filename)

    if not os.path.exists(filelocal):
        raise FileNotFoundError(f"Download failed: {filelocal} not found.")

    # Load and process dataset
    ds = xr.open_dataset(filelocal, cache=False)
    ds = ds[["U10M", "V10M"]].load()
    ds = ds.rename({"U10M": "u10", "V10M": "v10"})
    ds = ds.squeeze("time")

    # Interpolate to location
    u10 = float(ds.u10.interp(lon=lon, lat=lat).values)
    v10 = float(ds.v10.interp(lon=lon, lat=lat).values)

    # Check for NaN values
    if np.isnan(u10) or np.isnan(v10):
        raise ValueError(f"Wind data is NaN for location ({lon}, {lat}) at {date_of_acquisition}.")

    return u10, v10


def max_date_era5() -> datetime:
    """
    Get the maximum date available for ERA5 data.
    """
    global MAX_DATE_ERA5
    if MAX_DATE_ERA5 is not None:
        return MAX_DATE_ERA5

    home_dir = os.path.join(os.path.expanduser("~"), ".georeader")
    json_file = os.path.join(home_dir, "marss2l.json")

    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            json_dict = json.load(f)
        last_update_time_era5 = json_dict["last_update_time_era5"]
        last_update_time_era5 = datetime.fromisoformat(last_update_time_era5).replace(
            tzinfo=timezone.utc
        )

        # if there is less than 1 day with now return last_update_time_era5
        if (datetime.now(tz=timezone.utc) - last_update_time_era5).days < 1:
            MAX_DATE_ERA5 = json_dict["max_date_era5"]
            MAX_DATE_ERA5 = datetime.fromisoformat(MAX_DATE_ERA5).replace(tzinfo=timezone.utc)
            return MAX_DATE_ERA5

    import ee

    image_id = "ECMWF/ERA5_LAND/HOURLY"
    max_date_ms = ee.ImageCollection(image_id).aggregate_max("system:time_start")
    MAX_DATE_ERA5 = pd.to_datetime(max_date_ms.getInfo(), unit="ms", utc=True)

    # cache file in json_file
    os.makedirs(home_dir, exist_ok=True)
    json_dict = {
        "last_update_time_era5": datetime.now(tz=timezone.utc).isoformat(),
        "max_date_era5": MAX_DATE_ERA5.isoformat(),
    }
    with open(json_file, "w") as f:
        json.dump(json_dict, f)

    return MAX_DATE_ERA5


GEE_CHUNK_SIZE = 4500


def download_from_gee(
    locations_dates: gpd.GeoDataFrame,
    datetime_column: str = "date_of_acquisition",
    collection_name: str = "ECMWF/ERA5_LAND/HOURLY",
    logger: Optional[logging.Logger] = None,
) -> gpd.GeoDataFrame:
    """
    Download wind data from GEE. From either of these collections:

    * [NASA/GEOS-CF/v1/rpl/htf](https://developers.google.com/earth-engine/datasets/catalog/NASA_GEOS-CF_v1_rpl_htf#bands)
    * [ECMWF/ERA5_LAND/HOURLY](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY]).
        ERA5-Land data is available from 1981 to **three months from real-time**. The fuction will
        warn if the date of acquisition is less than 3 months from real-time. For ERA5 we query the wind variables
        `u_component_of_wind_10m` and `v_component_of_wind_10m`.

    Args:
        locations_dates (gpd.GeoDataFrame): Locations to download wind data for. Must have a geometry column.
        datetime_column (str, optional): name of the datetime column in locations_dates. Defaults to "date_of_acquisition".
        collection_name (str, optional): Name of the collection to download from GEE. One of:
            "ECMWF/ERA5_LAND/HOURLY", 'NASA/GEOS-CF/v1/rpl/htf'. Defaults to  "ECMWF/ERA5_LAND/HOURLY".

    Returns:
        gpd.GeoDataFrame: Wind data for the locations with columns
            ["geometry", "U", "V", "wind_index", "collection_name_wind"].
            It will have the same index and crs as locations_dates GeoDataFrame.
    """

    assert collection_name in [
        "ECMWF/ERA5_LAND/HOURLY",
        "NASA/GEOS-CF/v1/rpl/htf",
    ], f"collection_name must be one of ['ECMWF/ERA5_LAND/HOURLY', 'NASA/GEOS-CF/v1/rpl/htf'], got {collection_name}"

    assert (
        datetime_column in locations_dates.columns
    ), f"datetime_column must be in locations_dates.columns, got {datetime_column}"

    locations_dates_copy = locations_dates.copy()
    dates_of_acquisition = locations_dates_copy[datetime_column]

    # set timezone to UTC if not already
    if dates_of_acquisition.dt.tz is None:
        dates_of_acquisition = dates_of_acquisition.dt.tz_localize("UTC")
    else:
        dates_of_acquisition = dates_of_acquisition.dt.tz_convert("UTC")

    if collection_name == "ECMWF/ERA5_LAND/HOURLY":
        wind_variables = ["u_component_of_wind_10m", "v_component_of_wind_10m"]
        locations_dates_copy["wind_index"] = dates_of_acquisition.round("60min").dt.strftime(
            "%Y%m%dT%H"
        )

        # warn if date_of_acquisition is in three months from real-time
        if dates_of_acquisition.max() > max_date_era5():
            filter_dates = dates_of_acquisition <= max_date_era5()
            if not filter_dates.any():
                msg = f"All dates queried are posterior to latest available ERA55_LAND data: {max_date_era5()} no wind information will be returned."
                if logger is None:
                    warnings.warn(msg)
                else:
                    logger.warning(msg)

                return locations_dates_copy[filter_dates]
            else:
                n_images_with_wind = filter_dates.sum()
                msg = (
                    f"Warning: max date_of_acquisition is posterior to {max_date_era5()}.  Only {n_images_with_wind} out of {len(filter_dates)} have wind available."
                    + " ERA5-Land data is available 5 days from real time (see https://climate.copernicus.eu/climate-reanalysis)"
                )
                if logger is None:
                    warnings.warn(msg)
                else:
                    logger.warning(msg)

                locations_dates_copy = locations_dates_copy[filter_dates].copy()
    else:
        wind_variables = ["U", "V"]
        locations_dates_copy["wind_index"] = dates_of_acquisition.round("15min").dt.strftime(
            "%Y%m%d_%H%Mz"
        )

    locations_dates_index_name = locations_dates_copy.index.name
    if locations_dates_index_name is None:
        locations_dates_index_name = "location_dates_index"
        locations_dates_copy.index.name = locations_dates_index_name

    if locations_dates_copy.crs is None:
        locations_dates_copy = locations_dates_copy.set_crs("EPSG:4326")

    locations_dates_query = (
        locations_dates_copy[["geometry", "wind_index"]].reset_index().to_crs("EPSG:4326")
    )

    def query_gee(locations_dates_query_iter):
        import ee

        locations_dates_json = eval(locations_dates_query_iter.to_json(drop_id=True))
        locations_dates_ee = ee.FeatureCollection(locations_dates_json)

        image_collection = ee.ImageCollection(collection_name).select(wind_variables)

        def map_fun(feature: ee.Feature) -> ee.Feature:
            image_col = image_collection.filter(
                ee.Filter.eq("system:index", feature.get("wind_index"))
            )

            return ee.Algorithms.If(
                image_col.size().eq(0),
                feature,
                feature.set(image_col.first().reduceRegion(ee.Reducer.mean(), feature.geometry())),
            )

        wind_data_iter = locations_dates_ee.map(map_fun)
        wind_data_iter = gpd.GeoDataFrame.from_features(wind_data_iter.getInfo())
        return wind_data_iter

    if locations_dates_query.shape[0] > GEE_CHUNK_SIZE:
        # Query GEE in chunks of 4500
        wind_data = pd.concat(
            [
                query_gee(locations_dates_query.iloc[i : i + GEE_CHUNK_SIZE])
                for i in range(0, locations_dates_query.shape[0], GEE_CHUNK_SIZE)
            ],
            ignore_index=True,
        )
    else:
        wind_data = query_gee(locations_dates_query)

    if wind_data.shape[0] == 0:
        return wind_data

    wind_data = wind_data.set_index(locations_dates_index_name).set_crs("EPSG:4326")
    wind_data.to_crs(locations_dates_copy.crs, inplace=True)

    wind_data["collection_name_wind"] = collection_name
    wind_data.loc[locations_dates_copy.index, datetime_column] = dates_of_acquisition

    if collection_name == "ECMWF/ERA5_LAND/HOURLY":
        # rename wind columns to "U", "V"
        wind_data = wind_data.rename(
            dict(zip(["u_component_of_wind_10m", "v_component_of_wind_10m"], ["U", "V"])), axis=1
        )

    return wind_data


def add_wind_to_plot(
    wind_vector: ArrayLike,
    ax: Optional[plt.Axes] = None,
    color: str = "white",
    fontsize: int = 10,
    size_factor: float = 0.08,
    units: str = "m/s",
    loc: Union[Tuple[float, float], str] = "bottom left",
    head_width: Optional[float] = None,
    width: float = 0.001,
) -> plt.Axes:
    """
    Add wind vector to plot.

    Args:
        wind_vector (Tuple[float, float]): Wind vector, U, V components.
        ax (Optional[plt.Axes], optional): Axes to add wind vector to. Defaults to None.
        color (str, optional): Color of wind vector. Defaults to "white".
        fontsize (int, optional): Fontsize of wind speed. Defaults to 10.
        size_factor (float, optional): Size of wind vector. Defaults to .01.
        units (str, optional): Units of wind speed. Defaults to "m/s".

    Returns:
        plt.Axes: Axes with wind vector added.

    Example:
        >>> from georeader.plot import show
        >>> from marss2l.wind import add_wind_to_plot
        >>> from georeader.rasterio_reader import RasterioReader
        >>> mbmp_IL = RasterioReader('database/Turkmenistan/A3/mbmpIL/2019-07-19_S2A.tif').load().squeeze()
        >>> ax = show(mbmp_IL, add_colorbar_next_to=True,cmap="plasma",vmin=0,vmax=7_000, title=r"$\\Delta$CH$_4$ (ppb)")
        >>> add_wind_to_plot((-1,1), ax=ax)

    """
    if ax is None:
        ax = plt.gca()

    # bottom, top = ax.get_ylim()
    # left, right = ax.get_xlim()

    # xmin = left
    # xmax = right
    # ymin = bottom
    # ymax = top

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    if isinstance(loc, str):
        if (loc == "bottom left") or (loc == "lower left"):
            wind_vector_loc = (xmin * 0.8 + xmax * 0.2), (ymin * 0.8 + ymax * 0.2)
            wind_speed_loc = (xmin * 0.85 + xmax * 0.15), (ymin * 0.95 + ymax * 0.05)
        elif (loc == "bottom right") or (loc == "lower right"):
            wind_vector_loc = (xmin * 0.2 + xmax * 0.8), (ymin * 0.8 + ymax * 0.2)
            wind_speed_loc = (xmin * 0.15 + xmax * 0.85), (ymin * 0.95 + ymax * 0.05)
        elif (loc == "top left") or (loc == "upper left"):
            wind_vector_loc = (xmin * 0.8 + xmax * 0.2), (ymin * 0.2 + ymax * 0.8)
            wind_speed_loc = (xmin * 0.85 + xmax * 0.15), (ymin * 0.05 + ymax * 0.95)
        elif (loc == "top right") or (loc == "upper right"):
            wind_vector_loc = (xmin * 0.2 + xmax * 0.8), (ymin * 0.2 + ymax * 0.8)
            wind_speed_loc = (xmin * 0.15 + xmax * 0.85), (ymin * 0.05 + ymax * 0.95)
        else:
            raise ValueError(
                f"loc must be one of ['bottom left', 'bottom right', 'top left', 'top right'], got {loc}"
            )
    else:
        wind_vector_loc = loc
        # Shift wind speed location to the right
        wind_speed_loc = (wind_vector_loc[0] + (xmax - xmin) * 0.1, wind_vector_loc[1])

    size_x = (xmax - xmin) * size_factor
    size_y = (ymax - ymin) * size_factor

    wind_vector = np.array(wind_vector)
    wind_speed = np.linalg.norm(wind_vector)
    wind_vector_dir = wind_vector / wind_speed
    head_width = head_width or size_x * 0.2
    ax.arrow(
        wind_vector_loc[0],
        wind_vector_loc[1],
        wind_vector_dir[0] * size_x,
        wind_vector_dir[1] * size_y,
        head_width=head_width,
        width=width,
        color=color,
    )

    ax.annotate(f"{wind_speed:.2f} {units}", wind_speed_loc, color=color, fontsize=fontsize)

    return ax
