import matplotlib.pyplot as plt
from typing import Optional
import numpy as np

# define colors for water and land
OCEAN_COLOR = (plt.get_cmap('ocean'))(210)
LAND_COLOR = plt.get_cmap('gist_earth')(200)


def plot_geographical_location(lon:float, 
                               lat:float, ax:Optional[plt.Axes]=None,
                               text_annotation:Optional[str]=None) -> plt.Axes:
    """
    Plot the globe with the given location

    Args:
        lon (float): longitude of the location
        lat (float): _latitude of the location
        ax (plt.Axes): axis to plot on. If None, it uses `ptl.gca()`
        text_annotation (str, optional): text to annotate the location. Defaults to None.


    Returns:
        ax: plt.Axes: axis with the plot
    """
    if ax is None:
        ax = plt.gca()
    
    from mpl_toolkits.basemap import Basemap


    # set perspective angle
    lat_viewing_angle = lat # 50
    lon_viewing_angle = lon # -73

    # call the basemap and use orthographic projection at viewing angle
    m = Basemap(projection='ortho',
                lat_0=lat_viewing_angle, lon_0=lon_viewing_angle,
                ax=ax)

    # coastlines, map boundary, fill continents/water, fill ocean, draw countries
    m.drawcoastlines()
    m.drawmapboundary(fill_color=OCEAN_COLOR)
    m.fillcontinents(color=LAND_COLOR,lake_color=OCEAN_COLOR)
    m.drawcountries()

    # latitude/longitude line vectors
    lat_line_range = [-90,90]
    lat_lines = 8
    lat_line_count = (lat_line_range[1]-lat_line_range[0])/lat_lines

    merid_range = [-180,180]
    merid_lines = 8
    merid_count = (merid_range[1]-merid_range[0])/merid_lines

    m.drawparallels(np.arange(lat_line_range[0],lat_line_range[1],lat_line_count))
    m.drawmeridians(np.arange(merid_range[0],merid_range[1],merid_count))

    # scatter to indicate lat/lon point
    x,y = m(lon_viewing_angle,lat_viewing_angle)
    m.scatter(x,y,marker='o',color='#DDDDDD',s=3000,zorder=10,alpha=0.7,\
            edgecolor='#000000')
    m.scatter(x,y,marker='o',color='#000000',s=100,zorder=10,alpha=0.7,\
            edgecolor='#000000')

    if text_annotation is not None:
        plt.annotate(text_annotation, xy=(x, y),  xycoords='data',
                     xytext=(-110, -10), textcoords='offset points',
                     color='k',fontsize=12,bbox=dict(facecolor='w', alpha=0.5),
                     arrowprops=dict(arrowstyle="fancy", color='k'),
                     zorder=20)
    
    return m
