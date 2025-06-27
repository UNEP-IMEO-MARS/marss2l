import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
from .colors import PALETTE_ALL, C0, C1, C2, C4
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd


def calculate_images_by_yearmonth_and_summary(dataframe_data_traintest:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    images_by_yearmonth_total = dataframe_data_traintest.groupby(["year_month"])["isplume"].agg(["count","sum"]).rename({"count": "# images", "sum": "# plumes"}, axis=1)
    images_by_yearmonth_total["# locs"] = dataframe_data_traintest.groupby(["year_month"])["location_name"].nunique()
    images_by_yearmonth_total = images_by_yearmonth_total.reset_index()

    summaries = dataframe_data_traintest.groupby("split_name")["isplume"].agg(["sum", "count"]).rename(columns={"sum": "nplumes", "count": "nimages"})
    summaries["nlocs"] = dataframe_data_traintest.groupby("split_name")["location_name"].nunique()
    
    return images_by_yearmonth_total, summaries

def _plot_images_by_month(images_country_by_yearmonth, ax:plt.Axes, 
                         datetime_start:Optional[datetime]=None,
                         datetime_end:Optional[datetime]=None, step_in_months:int=3, 
                         loc_legend:str="center left", plot_legend:bool=True):
    """
    Plots the number of images, plumes, and locations by month.

    Args:
        images_country_by_yearmonth (pd.DataFrame): DataFrame containing:
            - 'year_month': datetime (or date-like) for each row.
            - '# images': int representing the number of images in that month.
            - '# plumes': int representing the number of plumes in that month.
            - '# locs': int representing the number of locations in that month.
        ax (plt.Axes): Matplotlib axes to draw on.
        datetime_start (Optional[datetime], optional): Start date for plotting. Defaults to None.
        datetime_end (Optional[datetime], optional): End date for plotting. Defaults to None.
        step_in_months (int, optional): Interval step (in months) for x-axis ticks. Defaults to 3.
        loc_legend (str, optional): Location of the legend. Defaults to "center left".
        plot_legend (bool, optional): If True, adds legend. Defaults to True.

    Returns:
        None: The function adds plots to the provided axes.
    """
    
    lty = ax.plot(images_country_by_yearmonth["year_month"], 
                  images_country_by_yearmonth["# images"], 
                  c=C0, label="# images")
    # legend1 = ax.legend(loc="center left")

    if datetime_end is None:
        datetime_end = images_country_by_yearmonth.year_month.max() + timedelta(days=step_in_months*30)
    
    if datetime_start is None:
        datetime_start = images_country_by_yearmonth.year_month.min()
    else:
        ax.set_xlim(datetime_start, datetime_end)
    
    axtwin =ax.twinx()
    lty2 = axtwin.plot(images_country_by_yearmonth["year_month"],images_country_by_yearmonth["# plumes"],c=C2, label="# plumes")
    lty3 = axtwin.plot(images_country_by_yearmonth["year_month"], images_country_by_yearmonth["# locs"], c=C4, label="# locs")
    
    # t = np.arange(datetime(2018,1,1), datetime(2024,3,1), timedelta(days=120)).astype(datetime)
    t = np.arange(datetime_start, 
                  datetime_end, np.timedelta64(step_in_months, 'M'),  dtype='datetime64[M]') 
    # t = np.arange(np.datetime64("2018-01-01"), np.datetime64("2024-03-01"), np.timedelta64(3, 'M'),  dtype='datetime64[M]') 
    # labels = [p.strftime("%Y-%m") for p in t]
    ax.set_xticks(t,t, rotation=30)
    if plot_legend:
        legend1 = plt.legend(lty + lty3 + lty2,
                             ["# images","# locs", "# plumes"], 
                             loc=loc_legend)
    axtwin.set_ylabel("# plumes & # locs")
    ax.set_ylabel("# images")
    ax.grid(axis="x")


def add_date_region(ax, start_date, end_date, label, stats=None, 
                    y_pos=0.98,
                    box_y=0.95-0.38,
                    box_height = 0.33,
                    box_widh_data_perc=0.8,
                    alpha=0.15, color="gray", linestyle="--", linewidth=1,
                   plot_start_line=True, plot_end_line=True):
    """
    Add a shaded region with boundary lines, text label, and statistics box.
    
    Args:
        ax: Matplotlib axis
        start_date: Start date for the region
        end_date: End date for the region
        label: Text to display in the region
        split_name: Name of the split to look up in summaries_df
        summaries_df: DataFrame containing the statistics
        y_pos: Vertical position of the label (in axis coordinates)
        alpha: Transparency of shaded region
        color: Color of shading and lines
        linestyle: Style for boundary lines
        linewidth: Width of boundary lines
        plot_start_line: Whether to draw the start boundary line
        plot_end_line: Whether to draw the end boundary line
    """
    # Add shaded region
    ax.axvspan(start_date, end_date, alpha=alpha, color=color)
    
    # Add boundary lines
    if plot_start_line:
        ax.axvline(start_date, color=color, linestyle=linestyle, linewidth=linewidth)
    if plot_end_line:
        ax.axvline(end_date, color=color, linestyle=linestyle, linewidth=linewidth)
    
    # Calculate midpoint for text placement
    mid_point = start_date + (end_date - start_date) / 2
    
    # Add label
    ax.text(mid_point, y_pos, label, ha='center', va='top',
            transform=ax.get_xaxis_transform(), fontweight='bold')
    
    # Add stats box if provided
    if stats is not None:
        # Create stats text
        stats_text = f"Images: {stats['nimages']:,}\nPlumes: {stats['nplumes']:,}\nLocs: {stats['nlocs']:,}"
        
        # Calculate box position and dimensions
             # Height in axes coordinates
        
        # Convert datetime to matplotlib's numerical date format
        mid_point_num = mdates.date2num(mid_point)
        # Calculate x-span in data coordinates
        x_range = mdates.date2num(end_date) - mdates.date2num(start_date)
        box_width_data = x_range * box_widh_data_perc  # 60% of segment width
        
        # Add background box
        rect = Rectangle((mid_point_num - box_width_data/2, box_y), 
                        width=box_width_data, height=box_height,
                        transform=ax.get_xaxis_transform(), 
                        facecolor='white', edgecolor='gray', alpha=0.9,
                        zorder=5)
        ax.add_patch(rect)
        
        # Add text on top of box
        ax.text(mid_point, box_y + box_height/2, stats_text,
                ha='center', va='center', transform=ax.get_xaxis_transform(),
                fontsize=12, zorder=6)
        
    
def plot_images_by_month(dataframe_data_traintest:pd.DataFrame, 
                         ax:Optional[plt.Axes]=None, 
                         datetime_start:Optional[datetime]=None,
                         datetime_end:Optional[datetime]=None, 
                         step_in_months:int=3, 
                         loc_legend:str="center left", 
                         plot_legend:bool=True):
    """
    Plots the number of images, plumes, and locations by month.

    Args:
        dataframe_data_traintest (pd.DataFrame): DataFrame containing:
        ax (Optional[plt.Axes], optional): Matplotlib axes to draw on. Defaults to None.
        datetime_start (Optional[datetime], optional): Start date for plotting. Defaults to None.
        datetime_end (Optional[datetime], optional): End date for plotting. Defaults to None.
        step_in_months (int, optional): Interval step (in months) for x-axis ticks. Defaults to 3.
        loc_legend (str, optional): Location of the legend. Defaults to "center left".
        plot_legend (bool, optional): If True, adds legend. Defaults to True.

    Returns:
        None: The function adds plots to the provided axes.
    """
    
    images_country_by_yearmonth, summaries = calculate_images_by_yearmonth_and_summary(dataframe_data_traintest)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 8), tight_layout=True)
    
    _plot_images_by_month(images_country_by_yearmonth, ax=ax,
                          datetime_start=datetime_start,
                          datetime_end=datetime_end,
                          step_in_months=step_in_months,
                          loc_legend=loc_legend,
                          plot_legend=plot_legend)
    
    ylims = ax.get_ylim()
    
    # Add date regions with statistics
    stats = summaries.loc["train"].to_dict() if "train" in summaries.index else None
    add_date_region(ax, datetime(2018, 1, 1), datetime(2020, 12, 31), "Train", 
                                          box_widh_data_perc=0.5,
                stats=stats, plot_start_line=False)

    stats = summaries.loc["val"].to_dict() if "val" in summaries.index else None
    add_date_region(ax, datetime(2021, 1, 1), datetime(2021, 12, 31), "Val",
                stats=stats)

    add_date_region(ax, datetime(2022, 1, 1), datetime(2023, 11, 30), "Train",
                stats=None)

    stats = summaries.loc["test"].to_dict() if "test" in summaries.index else None
    add_date_region(ax, datetime(2024, 1, 1), datetime(2024, 12, 31), "Test", 
                stats=stats, plot_end_line=True, 
                box_y=0.95-0.45)
    ax.set_ylim(ylims)
    