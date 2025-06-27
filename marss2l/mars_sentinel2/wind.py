
from typing import Union, Tuple, Optional
from datetime import datetime, timezone
from shapely import Geometry
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import warnings
import logging
import os
import json


MAX_DATE_ERA5 = None
def max_date_era5() -> datetime:
    """
    Get the maximum date available for ERA5 data. 
    """
    global MAX_DATE_ERA5
    if MAX_DATE_ERA5 is not None:
        return MAX_DATE_ERA5
    
    home_dir = os.path.join(os.path.expanduser('~'),".georeader")
    json_file = os.path.join(home_dir, "marss2l.json")

    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            json_dict = json.load(f)
        last_update_time_era5 = json_dict["last_update_time_era5"]
        last_update_time_era5 = datetime.fromisoformat(last_update_time_era5).replace(tzinfo=timezone.utc)

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
    json_dict = {"last_update_time_era5": datetime.now(tz=timezone.utc).isoformat(),
                 "max_date_era5": MAX_DATE_ERA5.isoformat()}
    with open(json_file, "w") as f:
        json.dump(json_dict, f)
    
    return MAX_DATE_ERA5


GEE_CHUNK_SIZE = 4500

def download_from_gee(locations_dates:gpd.GeoDataFrame, 
                      datetime_column:str="date_of_acquisition", 
                      collection_name:str= "ECMWF/ERA5_LAND/HOURLY",
                      logger:Optional[logging.Logger]=None) -> gpd.GeoDataFrame:
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

    assert collection_name in ["ECMWF/ERA5_LAND/HOURLY", 'NASA/GEOS-CF/v1/rpl/htf'], \
        f"collection_name must be one of ['ECMWF/ERA5_LAND/HOURLY', 'NASA/GEOS-CF/v1/rpl/htf'], got {collection_name}"
    
    assert datetime_column in locations_dates.columns, \
        f"datetime_column must be in locations_dates.columns, got {datetime_column}"
    
    locations_dates_copy = locations_dates.copy()
    dates_of_acquisition = locations_dates_copy[datetime_column]

    # set timezone to UTC if not already
    if dates_of_acquisition.dt.tz is None:
        dates_of_acquisition = dates_of_acquisition.dt.tz_localize("UTC")
    else:
        dates_of_acquisition = dates_of_acquisition.dt.tz_convert("UTC")

    if collection_name == "ECMWF/ERA5_LAND/HOURLY":
        wind_variables = ["u_component_of_wind_10m", "v_component_of_wind_10m"]
        locations_dates_copy["wind_index"] = dates_of_acquisition.round("60min").dt.strftime('%Y%m%dT%H')

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
                msg = f"Warning: max date_of_acquisition is posterior to {max_date_era5()}.  Only {n_images_with_wind} out of {len(filter_dates)} have wind available."+\
                       " ERA5-Land data is available 5 days from real time (see https://climate.copernicus.eu/climate-reanalysis)"
                if logger is None:
                    warnings.warn(msg)
                else:
                    logger.warning(msg)
                
                locations_dates_copy = locations_dates_copy[filter_dates].copy()
    else:
        wind_variables = ["U", "V"]
        locations_dates_copy["wind_index"] = dates_of_acquisition.round("15min").dt.strftime('%Y%m%d_%H%Mz')
    
    locations_dates_index_name = locations_dates_copy.index.name
    if locations_dates_index_name is None:
        locations_dates_index_name = "location_dates_index"
        locations_dates_copy.index.name = locations_dates_index_name
    
    if locations_dates_copy.crs is None:
        locations_dates_copy = locations_dates_copy.set_crs("EPSG:4326")
    
    locations_dates_query = locations_dates_copy[["geometry","wind_index"]].reset_index().to_crs("EPSG:4326")
    
    
    def query_gee(locations_dates_query_iter):
        import ee
        locations_dates_json = eval(locations_dates_query_iter.to_json(drop_id=True))
        locations_dates_ee = ee.FeatureCollection(locations_dates_json)

        image_collection = ee.ImageCollection(collection_name).select(wind_variables)

        def map_fun(feature:ee.Feature) -> ee.Feature:
            image_col = image_collection.filter(ee.Filter.eq("system:index", feature.get("wind_index")))

            return ee.Algorithms.If(image_col.size().eq(0),
                                    feature,
                                    feature.set(image_col.first().reduceRegion(ee.Reducer.mean(), feature.geometry())))
        
        wind_data_iter = locations_dates_ee.map(map_fun)
        wind_data_iter = gpd.GeoDataFrame.from_features(wind_data_iter.getInfo())
        return wind_data_iter

    if locations_dates_query.shape[0] > GEE_CHUNK_SIZE:
        # Query GEE in chunks of 4500
        wind_data = pd.concat([query_gee(locations_dates_query.iloc[i:i+GEE_CHUNK_SIZE]) for i in range(0, locations_dates_query.shape[0], GEE_CHUNK_SIZE)],
                              ignore_index=True)
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
        wind_data = wind_data.rename(dict(zip(["u_component_of_wind_10m", "v_component_of_wind_10m"], ["U", "V"])),
                                     axis=1)

    return wind_data


def add_wind_to_plot(wind_vector: ArrayLike, ax: Optional[plt.Axes]=None,
                     color:str="white",fontsize:int=10, size_factor:float=.08,
                     units:str="m/s", loc:Union[Tuple[float,float], str]="bottom left",
                     head_width:Optional[float]=None, width:float=0.001) -> plt.Axes:
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
            wind_vector_loc = (xmin*.8 + xmax*.2), (ymin*.8 + ymax*.2)
            wind_speed_loc = (xmin*.85 + xmax*.15), (ymin*.95 + ymax*.05)
        elif (loc == "bottom right") or (loc == "lower right"):
            wind_vector_loc = (xmin*.2 + xmax*.8), (ymin*.8 + ymax*.2)
            wind_speed_loc = (xmin*.15 + xmax*.85), (ymin*.95 + ymax*.05)
        elif (loc == "top left") or (loc == "upper left"):
            wind_vector_loc = (xmin*.8 + xmax*.2), (ymin*.2 + ymax*.8)
            wind_speed_loc = (xmin*.85 + xmax*.15), (ymin*.05 + ymax*.95)
        elif (loc == "top right") or (loc == "upper right"):
            wind_vector_loc = (xmin*.2 + xmax*.8), (ymin*.2 + ymax*.8)
            wind_speed_loc = (xmin*.15 + xmax*.85), (ymin*.05 + ymax*.95)
        else:
            raise ValueError(f"loc must be one of ['bottom left', 'bottom right', 'top left', 'top right'], got {loc}")
    else:
        wind_vector_loc = loc
        # Shift wind speed location to the right
        wind_speed_loc = (wind_vector_loc[0] + (xmax - xmin)*.1, wind_vector_loc[1])

    size_x = (xmax - xmin) * size_factor
    size_y = (ymax - ymin) * size_factor

    wind_vector = np.array(wind_vector)
    wind_speed = np.linalg.norm(wind_vector)
    wind_vector_dir = wind_vector / wind_speed
    head_width = head_width or size_x*.2
    ax.arrow(wind_vector_loc[0], wind_vector_loc[1], 
             wind_vector_dir[0]*size_x, wind_vector_dir[1]*size_y,
             head_width=head_width, width=width,
             color=color)
    
    ax.annotate(f"{wind_speed:.2f} {units}",
                wind_speed_loc, color=color, 
                fontsize=fontsize)
    
    return ax


