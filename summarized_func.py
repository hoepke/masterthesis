"""
summarized_func.py

Data preparation, clustering, and RQA orchestration for one preset
(season/window combination).

1. Styling helpers        - style_axes, set_aspect_latlon, cluster_norm,
                             get_cluster_cmap, get_cluster_norm, _rain_scale
2. Data preparation        - prepare (normalization, PCA input, heatmap),
                             heatmap_periods
3. Clustering              - optimize_cluster, run_k_analysis
                             (agglomerative clustering, k via Kneedle),
                             smooth_cluster (mask post-processing)
4. RQA orchestration       - compute_rain_rolling, rqa_full
                             (runs RQA per community, saves Mann-Kendall
                             trend results)
"""




import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import netCDF4 # noqa: F401

from rqa import rolling_window_rqa,plot_metric_rolling, apply_grid, extract_X
from community import (analyze_rainfall_data, plot_heatmap, raw_cluster_aggl,
                       preprocess_ds, q_statistic,summarize_k, find_knee,
                       plot_k_analysis,majority_filter_voronoi,split_connected_components,
                       merge_remaining_small,merge_small_with_small,
                       export_heatmap, clip_to_shapefile)
from plots_and_stats import plot_ts  


import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
from cmcrameri import cm
from matplotlib.colors import ListedColormap,BoundaryNorm
import matplotlib as mpl
from pathlib import Path
import geopandas as gpd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from shapely.ops import unary_union
from sklearn.preprocessing import StandardScaler   
from joblib import Parallel, delayed              
import itertools


from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import box
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex

##################################### DESIGN & HELPER FUNCTIONS #####################################

mpl.rcParams.update({
    
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,

    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.linewidth": 0.7,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})
mpl.rcParams['text.usetex'] = False


CLUSTER_COLORS = [
'#9437FF','#FF1150','#FFA300',"#009193", "#011993","#7A81FF"
]

CLUSTER_COLORS_2 = [
    '#0378A6',  # blue
    '#F27329',  # orange
    '#BFB73F',  # yellow-green
    '#6B2FA0',  # purple
    '#CC3366',  # magenta
    '#f76906',  # orange2
    '#088267',  # green
]

cluster_cmap = ListedColormap(CLUSTER_COLORS)

def cluster_norm(k):
    """Discrete color norm so each of the k cluster IDs gets its own solid color."""
    bounds = np.arange(-0.5, k + 0.5, 1)
    return BoundaryNorm(bounds, k)

def get_cluster_cmap(k):
    """Return appropriate colormap for k clusters.
    If k <= len(CLUSTER_COLORS), use discrete colors.
    Otherwise, use continuous colormap (tab20 or hsv)."""
    if k <= len(CLUSTER_COLORS):
        return cluster_cmap
    if k <= 20:
        return plt.get_cmap('tab20')
    return plt.get_cmap('hsv')

def get_cluster_norm(k):
    """
    Return appropriate norm for k clusters.
    If k <= len(CLUSTER_COLORS), use BoundaryNorm.
    Otherwise, use continuous norm (0 to k-1).
    """
    if k <= len(CLUSTER_COLORS):
        return cluster_norm(k)
    return mpl.colors.Normalize(vmin=-0.5, vmax=k - 0.5)


boot_colors = [
    {"before": "#9437FF", "after": "#9437FF"},   # cluster 1
    {"before": "#FF1150", "after": "#FF1150"},   # cluster 2   
    {"before": "#FFA300", "after": "#FFA300"},   # cluster 3
    {"before": "#009193", "after": "#009193"},   # cluster 4
    {"before": "#011993", "after": "#011993"},   # cluster 5
    {"before": "#7A81FF", "after": "#7A81FF"},   # cluster 6
]

def style_axes(ax):
    """Apply the standard white background, major/minor grid, and tick style to an axes."""
    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=9)
    
def set_aspect_latlon(ax, lats, lons):
    """Set a geographically correct aspect ratio for lat/lon axes, based on the mean latitude."""
    lat_center = (np.nanmax(lats) + np.nanmin(lats)) / 2
    aspect = 1 / np.cos(np.radians(lat_center))
    ax.set_aspect(aspect)
    
def _rain_scale(ds):
    """Return 1000.0 for ERA5 data in metres, 1.0 for data already in mm."""
    units = ds["rainfall"].attrs.get("units", "").strip().lower()
    return 1000.0 if units in ("m", "meters", "metres") else 1.0
    

    
########## PCA ##########

def prepare(path,
            window_size,
            overlap_p,
            plot_path,
            shapefile_path=None,
            show=False,
            months=None,
            year=None,
            SC_threshold=None,
            plot_curves=False,
            plot_stride=25,
            plot_first_n=3,
            title="Heatmap",
            figsize=(8,6),
            scatter=40,
            export_file=False,
            min_community_size="auto",        
):
    """
    Load rainfall data (CSV in mm, or NetCDF/ERA5 in metres), apply seasonal
    filtering and normalization, compute anomalies/detrending, and run the
    sliding-window heatmap analysis used as PCA input for clustering.

    ds_norm_season is the standardized time series for the selected season only,
    ds_norm_all is the full time series standardized the same way.

    Returns
    -------
    tuple
        (ds, B_norm, ds_season, ds_norm_season, ds_norm_all, heatmap_fraction,
         heatmap, coords_lat, coords_lon, ds_anom, ds_anom_std, ds_detrended,
         ds_detrend_norm)
    """
    path = Path(path)

    if path.suffix.lower() == ".csv":
        rainfall_data = (
            pd.read_csv(path)
            .drop(columns=["Unnamed: 0"], errors="ignore")
            .drop(columns=["geometry"], errors="ignore")
            .melt(id_vars=["lat", "lon"], var_name="time", value_name="rainfall")
            .assign(time=lambda df: pd.to_datetime(df["time"], format="mixed"))
        )

        ds = rainfall_data.set_index(['time', 'lat', 'lon']).to_xarray()
    else:
        try:
            ds = xr.open_dataset(path, chunks={"time": 365}).rename({"tp": "rainfall", "valid_time": "time"})
        except ImportError as exc:
            if "chunk manager 'dask'" not in str(exc):
                raise
            ds = xr.open_dataset(path).rename({"tp": "rainfall", "valid_time": "time"})

    # note: ERA5 "rainfall" stays in its native units (metres) here; the
    # metres -> mm conversion is applied via _rain_scale() only where a
    # rainfall metric is actually computed (see below and compute_rain_rolling).

    if shapefile_path is not None:
        n_before = int(ds["rainfall"].notnull().any("time").sum())
        ds = clip_to_shapefile(ds, shapefile_path)
        n_after = int(ds["rainfall"].notnull().any("time").sum())
        print(f"shapefile clip: {n_before} -> {n_after} valid cells")

    if months is not None:
        ds_month = ds.sel(time=ds.time.dt.month.isin(months))
    else:
        ds_month = ds
    
    # hydrological year starts in June
    hydro_year = ds_month.time.dt.year.where(
        ds_month.time.dt.month >= 6,
        ds_month.time.dt.year - 1
        )
    ds_month = ds_month.assign_coords(hydro_year=("time", hydro_year.values))
        
    if year == "before_1981":
        ds_year = ds_month.where(ds_month["time"].dt.year < 1981, drop=True)
    elif year == "after_1981":
        ds_year = ds_month.where(ds_month["time"].dt.year >= 1981, drop=True)
    elif year == "before_1991":
        ds_year = ds_month.where(ds_month["time"].dt.year < 1991, drop=True)
    elif year == "after_1991":
        ds_year = ds_month.where(ds_month["time"].dt.year >= 1991, drop=True)
    elif year in [None, "all"]:
        ds_year = ds_month
    
    ds_season = ds_year

    mean = ds_year['rainfall'].mean(dim='time', skipna=True)
    std = ds_year['rainfall'].std(dim='time', skipna=True)
    std = std.where(std > 0, other=1)
    
    # mean total rainfall per node, per year, averaged over the whole region
    total_per_node = ds_year['rainfall'].sum(dim='time', skipna=True) * _rain_scale(ds_year)
    total_per_node = total_per_node.where(ds_year['rainfall'].notnull().any(dim='time'))
    n_years = np.unique(ds_year['hydro_year'].values).size
    mean_total_rain = float(total_per_node.mean(skipna=True)) / n_years
    print(f"Mean total rainfall per node per year ({Path(plot_path).name}) {mean_total_rain:.2f} mm")

    ds_norm_season = ds_year.copy()
    ds_norm_season['rainfall'] = (ds_year['rainfall'] - mean) / std
        
    if (months is not None) or (year is not None):
        ds_norm_all = ds.copy()
        mean = ds_norm_all['rainfall'].mean(dim='time', skipna=True)
        std = ds_norm_all['rainfall'].std(dim='time', skipna=True)
        std = std.where(std > 0, other=1)

        ds_norm_all['rainfall'] = (ds_norm_all['rainfall'] - mean) / std
            
    else:
        ds_norm_all = ds_norm_season
    
    # anomalies
    climatology = (
        ds_year['rainfall']
        .groupby('time.month')
        .mean(dim='time', skipna=True)
    )

    ds_anom = ds_year.copy()
    ds_anom['rainfall'] = (
        ds_year['rainfall']
        .groupby('time.month')
        - climatology
    )
    
    # standardized anomalies
    std = (
    ds_anom['rainfall']
    .groupby('time.month')
    .std(dim='time', skipna=True)
    .where(lambda x: x > 0, 1)
    )

    ds_anom_std = ds_anom.copy()
    ds_anom_std['rainfall'] = (
        ds_anom['rainfall']
        .groupby('time.month') / std
    )
    
    #linear detrend
    trend = ds_anom['rainfall'].polyfit(
        dim='time',
        deg=1,
        skipna=True
    )

    trend_line = xr.polyval(
        ds_anom['time'],
        trend.polyfit_coefficients
    )

    # Detrend
    ds_detrended = ds_anom.copy()
    ds_detrended['rainfall'] = ds_anom['rainfall'] - trend_line
    
    # normalize
    std = ds_detrended['rainfall'].std(dim='time', skipna=True).where(lambda x: x > 0, 1)
    ds_detrend_norm = ds_detrended.copy()
    ds_detrend_norm['rainfall'] = ds_detrended['rainfall'] / std

    S = ds_norm_season["rainfall"].stack(spatial=("lat", "lon"))  # dims: (time, spatial)

    coords_lat = S["lat"].values
    coords_lon = S["lon"].values

    B_norm = np.nan_to_num(S.transpose("spatial", "time").values)

    # Domain summary
    valid    = ~np.all(np.isnan(S.values), axis=0)
    n_total  = S.sizes["spatial"]
    n_valid  = int(valid.sum())

    lat_u = np.unique(coords_lat)
    lon_u = np.unique(coords_lon)
    lat_res = float(np.median(np.diff(lat_u)))
    lon_res = float(np.median(np.diff(lon_u)))

    lat_c = np.deg2rad(coords_lat[valid])
    cell_km2 = (lat_res * 111.32) * (lon_res * 111.32 * np.cos(lat_c))
    area_km2 = float(cell_km2.sum())

    print(f"grid            {len(lat_u)} x {len(lon_u)} at {lat_res:.3f} deg")
    print(f"cells           {n_valid} valid of {n_total} total")
    print(f"area            {area_km2:,.0f} km2")
    print(f"lat range       {lat_u.min():.2f} to {lat_u.max():.2f}")
    print(f"lon range       {lon_u.min():.2f} to {lon_u.max():.2f}")
    

    # heatmap analysis
    print("calculate heatmap ...")
    
    overlap = int(round(window_size * overlap_p))
    
    print(f"overlap of {overlap} days")
    
    time_labels,_, heatmap_fraction, heatmap, coords_lat, coords_lon = analyze_rainfall_data(
        B_norm, ds_norm_season, window_size=window_size, overlap=overlap, SC_threshold=SC_threshold,
        plot_curves=plot_curves, plot_stride=plot_stride, plot_first_n=plot_first_n, plot_path=plot_path,
        min_community_size=min_community_size)
        

    plot_heatmap(
        heatmap_fraction,
        coords_lat,
        coords_lon,
        Title=title,
        figsize=figsize,
        scatter=scatter,
        plot_path =plot_path,
        show=show
    )
        
    if export_file:
        export_heatmap(
            heatmap_values=heatmap_fraction,
            coords_lat=coords_lat,
            coords_lon=coords_lon,
            output_path=Path(plot_path) / title.replace(" ", "_"),
            value_name="rec_frac",
        )

    return (ds, B_norm, ds_season, ds_norm_season, ds_norm_all, heatmap_fraction, heatmap,
            coords_lat, coords_lon,
            ds_anom, ds_anom_std, ds_detrended, ds_detrend_norm)



def heatmap_periods(
    ds,
    window_size,
    overlap_p,
    plot_path,
    SC_threshold    = None,
    period_length   = 15,
    custom_periods  = None,
    start_year      = 1951,
    end_year        = 2023,
    figsize         = None,
    scatter         = 50,
    vmin            = 0,
    vmax            = 1,
    show            = False,
    n_cols          = 3,
    min_years       = None,
    shapefile_path  = None,
):
    """
    Plot the recurrence-fraction heatmap separately for each sub-period
    (custom_periods, or evenly split into period_length-year chunks between
    start_year and end_year). Produces one combined grid figure plus one SVG
    and one shapefile per period, for comparing community stability over time.
    """
    def shapefile_clip_patch(ax, shapefile_path):
        """Clip an axes' pcolormesh to the shapefile boundary and return the clip patch."""
        gdf = gpd.read_file(shapefile_path)
        geom = unary_union(gdf.geometry)

        def ring_to_path(coords):
            verts = list(coords)
            codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
            return verts, codes

        all_verts, all_codes = [], []
        geoms = geom.geoms if hasattr(geom, 'geoms') else [geom]
        for g in geoms:
            v, c = ring_to_path(g.exterior.coords)
            all_verts += v; all_codes += c
            for interior in g.interiors:
                v, c = ring_to_path(interior.coords)
                all_verts += v; all_codes += c

        path = MplPath(all_verts, all_codes)
        patch = PathPatch(path, transform=ax.transData, facecolor='none', edgecolor='none')
        ax.add_patch(patch)
        return patch

    def make_smooth_grid(lons_v, lats_v, values, smooth_factor=1, upsample=2):
        """Interpolate scattered (lon, lat, value) points onto a regular, upsampled grid, masking cells too far from any original point."""
        res_lon = np.min(np.diff(np.unique(np.sort(lons_v)))) / upsample
        res_lat = np.min(np.diff(np.unique(np.sort(lats_v)))) / upsample
        print(f"res_lon={res_lon:.4f}, res_lat={res_lat:.4f}, upsample={upsample}")
        orig_res_lon = res_lon * upsample
        orig_res_lat = res_lat * upsample

        grid_lon, grid_lat = np.meshgrid(
            np.arange(lons_v.min(), lons_v.max() + res_lon, res_lon),
            np.arange(lats_v.min(), lats_v.max() + res_lat, res_lat),
        )
        grid_z = griddata(
            points=np.column_stack([lons_v, lats_v]),
            values=values,
            xi=(grid_lon, grid_lat),
            method="linear",
        )
        tree = cKDTree(np.column_stack([lons_v, lats_v]))
        dist, _ = tree.query(np.column_stack([grid_lon.ravel(), grid_lat.ravel()]))
        mask = dist.reshape(grid_lon.shape) > max(orig_res_lon, orig_res_lat) * smooth_factor
        grid_z = np.ma.array(grid_z, mask=mask)
        return grid_lon, grid_lat, grid_z

    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    if min_years is None:
        min_years = period_length // 2

    if custom_periods is not None:
        periods = custom_periods
    else:
        periods = []
        y = start_year
        while y <= end_year:
            y_end = min(y + period_length - 1, end_year)
            if (y_end - y + 1) >= min_years:
                periods.append((y, y_end))
            else:
                print(f"  skipped period {y}–{y_end} ")
            y += period_length

    print(f"Periods: {periods}")
    n_periods = len(periods)
    n_cols    = min(n_cols, n_periods)
    n_rows    = int(np.ceil(n_periods / n_cols))

    if figsize is None:
        lon_range = ds["lon"].max().item() - ds["lon"].min().item()
        lat_range = ds["lat"].max().item() - ds["lat"].min().item()
        lat_center   = (ds["lat"].min().item() + ds["lat"].max().item()) / 2
        aspect       = 1 / np.cos(np.radians(lat_center))
        inch_per_deg = 0.55
        subplot_w    = lon_range * inch_per_deg
        subplot_h    = lat_range * inch_per_deg * aspect
        figsize = (subplot_w * n_cols + 1.5, subplot_h * n_rows + 2.5) 
        print(f"  figsize: {figsize} (lon_range={lon_range:.1f}°, lat_range={lat_range:.1f}°)")

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        squeeze=False,
        gridspec_kw={"hspace": 0.45, "wspace": 0.25},
    )
    axes_flat = axes.flatten()

    for j in range(n_periods, len(axes_flat)):
        axes_flat[j].set_visible(False)

    mesh_last = None
    period_data = []

    for idx, (y_start, y_end) in enumerate(periods):
        ax = axes_flat[idx]
        print(f"\n── Period {y_start}–{y_end} ──")

        ds_p = ds.sel(time=ds.time.dt.year.isin(range(y_start, y_end + 1)))

        if ds_p.time.size == 0:
            print(f"  No data")
            ax.set_title(f"{y_start}–{y_end}\n(no data)")
            continue

        mean    = ds_p["rainfall"].mean(dim="time", skipna=True)
        std     = ds_p["rainfall"].std(dim="time", skipna=True)
        std     = std.where(std > 0, other=1)
        ds_norm = ds_p.copy()
        ds_norm["rainfall"] = (ds_p["rainfall"] - mean) / std

        S = ds_norm["rainfall"].stack(spatial=("lat", "lon"))
        B = np.nan_to_num(S.transpose("spatial", "time").values)
        overlap = int(round(window_size * overlap_p))

        time_labels, _, heatmap_fraction, heatmap, coords_lat, coords_lon = analyze_rainfall_data(
            B            = B,
            ds           = ds_norm,
            window_size  = window_size,
            overlap      = overlap,
            SC_threshold = SC_threshold,
            plot_curves  = False,
            plot_stride  = None,
            plot_first_n = 0,
            plot_path    = plot_path / f"{y_start}_{y_end}",
            show         = False,
        )

        heatmap_fraction = np.where(np.isnan(heatmap_fraction), 0.0, heatmap_fraction)

        valid = np.isfinite(heatmap_fraction)
        grid_lon, grid_lat, grid_z = make_smooth_grid(
            coords_lon[valid], coords_lat[valid], heatmap_fraction[valid]
        )
        
        period_data.append({
            "y_start":      y_start,
            "y_end":        y_end,
            "grid_lon":     grid_lon,
            "grid_lat":     grid_lat,
            "grid_z":       grid_z,
            "coords_lat":   coords_lat,
            "coords_lon":   coords_lon,
        })

        mesh = ax.pcolormesh(
            grid_lon, grid_lat, grid_z,
            cmap=cm.lipari,
            vmin=vmin, vmax=vmax,
            shading="nearest",
        )
        ax.set_xlim(coords_lon[valid].min() - 0.5, coords_lon[valid].max() + 0.5)
        ax.set_ylim(coords_lat[valid].min() - 0.5, coords_lat[valid].max() + 0.5)

        ax.contour(
            grid_lon, grid_lat, grid_z,
            levels=5,
            colors='white',
            linewidths=0.4,
            alpha=0.8,
                )
        

        if shapefile_path:
            clip = shapefile_clip_patch(ax, shapefile_path)
            mesh.set_clip_path(clip)

        mesh_last = mesh

        ax.set_title(f"{y_start}–{y_end}", fontsize=11, pad=6)
        ax.set_xlabel("Longitude", fontsize=8)
        if idx % n_cols == 0:
            ax.set_ylabel("Latitude", fontsize=8)
        set_aspect_latlon(ax, coords_lat[valid], coords_lon[valid])
        style_axes(ax)
        
        ax.set_gid(f"subplot_{y_start}_{y_end}")
        ax.title.set_gid(f"title_{y_start}_{y_end}")
        ax.xaxis.label.set_gid(f"xlabel_{y_start}_{y_end}")
        ax.yaxis.label.set_gid(f"ylabel_{y_start}_{y_end}")
        ax.xaxis.set_gid(f"xaxis_{y_start}_{y_end}")
        ax.yaxis.set_gid(f"yaxis_{y_start}_{y_end}")

        for line in ax.get_xgridlines():
            line.set_gid(f"xgrid_{y_start}_{y_end}")
        for line in ax.get_ygridlines():
            line.set_gid(f"ygrid_{y_start}_{y_end}")

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_gid(f"ticklabel_{y_start}_{y_end}")

    fig.suptitle(
        f"Heatmap per {period_length}-year period",
        fontsize=13,
        y=0.98,
    )

    fig.subplots_adjust(top=0.93, bottom=0.12, left=0.08, right=0.97)
    if mesh_last is not None:
        cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.025])
        cbar = fig.colorbar(mesh_last, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Recurrence Fraction", fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        cbar_ax.set_gid("colorbar_ax")
        cbar.ax.set_gid("colorbar")

    if fig.texts:
        fig.texts[0].set_gid("suptitle")

    for ax in axes_flat[:n_periods]:
        ax.set_rasterized(True)
        

# save
    for ax in axes_flat[:n_periods]:
        for artist in ax.get_children():
            if artist.__class__.__name__ == 'QuadMesh':
                artist.set_rasterized(True)

    png_path = plot_path / "heatmap_periods.png"
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    print(f"  Saved: {png_path}")

    # svg for each subplot

    for pd_item in period_data:
        y_start    = pd_item["y_start"]
        y_end      = pd_item["y_end"]
        grid_lon   = pd_item["grid_lon"]
        grid_lat   = pd_item["grid_lat"]
        grid_z     = pd_item["grid_z"]
        coords_lat = pd_item["coords_lat"]
        coords_lon = pd_item["coords_lon"]
        valid      = np.isfinite(grid_z.data if hasattr(grid_z, "data") else grid_z)

        fig_single, ax_single = plt.subplots(figsize=(subplot_w, subplot_h))

        mesh = ax_single.pcolormesh(
            grid_lon, grid_lat, grid_z,
            cmap     = cm.lipari,
            vmin     = vmin,
            vmax     = vmax,
            shading  = "nearest",
        )
        mesh.set_rasterized(True)

        ax_single.contour(
            grid_lon, grid_lat, grid_z,
            levels     = 5,
            colors     = "white",
            linewidths = 0.4,
            alpha      = 0.8,
        )

        if shapefile_path:
            clip = shapefile_clip_patch(ax_single, shapefile_path)
            mesh.set_clip_path(clip)

        ax_single.set_xlim(coords_lon.min() - 0.5, coords_lon.max() + 0.5)
        ax_single.set_ylim(coords_lat.min() - 0.5, coords_lat.max() + 0.5)
        ax_single.set_title(f"{y_start}–{y_end}", fontsize=11)
        ax_single.set_xlabel("Longitude", fontsize=8)
        ax_single.set_ylabel("Latitude", fontsize=8)
        set_aspect_latlon(ax_single, coords_lat, coords_lon)
        style_axes(ax_single)

        svg_single = plot_path / f"subplot_{y_start}_{y_end}.svg"
        fig_single.savefig(svg_single, format="svg", bbox_inches="tight", dpi=300)
        plt.close(fig_single)
        print(f"  Saved: {svg_single}")

        # ── shapefile per period ─────────────────────────────────────────────

        norm   = Normalize(vmin=vmin, vmax=vmax)
        mapper = ScalarMappable(norm=norm, cmap=cm.lipari)

        for pd_item in period_data:
            y_start  = pd_item["y_start"]
            y_end    = pd_item["y_end"]
            grid_lon = pd_item["grid_lon"]
            grid_lat = pd_item["grid_lat"]
            grid_z   = pd_item["grid_z"]

            data     = np.ma.filled(grid_z, np.nan)
            res_lon  = grid_lon[0, 1] - grid_lon[0, 0]
            res_lat  = abs(grid_lat[1, 0] - grid_lat[0, 0])

            rows = []
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    if not np.isfinite(val):
                        continue
                    lon = grid_lon[i, j]
                    lat = grid_lat[i, j]
                    geom  = box(
                        lon - res_lon / 2, lat - res_lat / 2,
                        lon + res_lon / 2, lat + res_lat / 2,
                    )
                    color = to_hex(mapper.to_rgba(val))
                    rows.append({
                        "geometry": geom,
                        "rf":       float(val),
                        "period":   f"{y_start}-{y_end}",
                        "color":    color,
                    })

            if len(rows) == 0:
                continue

            gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
            shp_path = plot_path / f"heatmap_{y_start}_{y_end}.shp"
            gdf.to_file(shp_path)
            print(f"  Saved: {shp_path}")
        
    
###### CLUSTERING ###################s

def optimize_cluster(
    heatmap,
    ds,
    min_cluster_fraction: float = 0.0,
    linkage_list: list[str] = ["ward", "average", "complete"],
    metric_list: list[str] = ["euclidean", "manhattan"],
    n_clusters_list: list[int] | None = None,
    use_auto_k: bool = False,
    connectivity: str = "queen",
    min_cluster_number: int = 2,
    alpha: float = 0.05,
    verbose: bool = True,
    n_jobs: int = -1,
):
    """
    Find the best spatial clustering configuration using only the q-statistic.
    The candidate with the highest q-value is selected as the best solution.

    Parameters
    ----------
    heatmap              : 2-D array of attribute values (same spatial shape as ds)
    ds                   : xarray Dataset with a "rainfall" variable (lat × lon × time)
    min_cluster_fraction : fraction of N used to derive the minimum cluster size;
                           clusters smaller than max(5, fraction * N) are merged
                           into their largest neighbour
    linkage_list         : Ward / average / complete linkage methods to search
    metric_list          : distance metrics to search (ward ignores non-euclidean)
    n_clusters_list      : explicit k values to try; required if use_auto_k=False
    use_auto_k           : also search for k via automatic threshold detection
    connectivity         : "queen" (8-connected) or "rook" (4-connected)
    min_cluster_number   : discard solutions with fewer clusters than this
    alpha                : significance level for q-statistic p-value
    verbose              : print progress and results
    n_jobs               : joblib parallelism (-1 = all cores)

    Returns
    -------
    best_labels : np.ndarray  – cluster label for every spatial unit (-1 = invalid)
    best_meta   : dict        – metadata for the winning configuration
    """
    z_v, coords_geo, coords_idx, scaler_z, valid_idx, sp, lats, lons = preprocess_ds(
        heatmap, ds
    )

    scaler_coords = StandardScaler().fit(coords_idx)
    coords_scaled = scaler_coords.transform(coords_idx)
    z_scaled      = scaler_z.transform(z_v[:, None])
    X             = np.hstack([z_scaled, coords_scaled])
    z_raw         = z_v.ravel().astype(float)

    #  Minimum cluster size (single value, derived from fraction)
    N_valid          = len(z_raw)
    min_cluster_size = max(5, int(N_valid * min_cluster_fraction))
    _mcs_list        = [min_cluster_size]

    if verbose:
        print(f"min_cluster_size : {min_cluster_size} "
              f"({min_cluster_fraction*100:.0f}% of N={N_valid})")
        print(f"Connectivity     : {connectivity}")

    # Build parameter grid 
    _explicit_k = n_clusters_list if n_clusters_list is not None else []
    k_values    = ([None] if use_auto_k else []) + _explicit_k
    if not k_values:
        raise ValueError(
            "No k values to search. Set n_clusters_list or use_auto_k=True."
        )

    param_space = [
        (mcs, lnk, met, k)
        for mcs, lnk, met, k in itertools.product(
            _mcs_list, linkage_list, metric_list, k_values
        )
        if not (lnk == "ward" and met != "euclidean")
    ]

    if verbose:
        n_auto     = sum(1 for p in param_space if p[3] is None)
        n_explicit = len(param_space) - n_auto
        print(f"Grid search      : {len(param_space)} configurations "
              f"({n_auto} auto-k, {n_explicit} explicit-k) …\n")

    #  Worker 
    def _run(min_cluster_size, linkage, metric, n_clusters):
        try:
            labels, lab_v, k_found, thr = raw_cluster_aggl(
                linkage, metric, n_clusters, valid_idx, sp, X,
                min_cluster_size, coords_idx, connectivity=connectivity,
            )
            if k_found < min_cluster_number:
                return None
            qs = q_statistic(z_raw, lab_v, alpha=alpha)
            return {
                "min_cluster_size": min_cluster_size,
                "linkage":          linkage,
                "metric":           metric,
                "n_clusters_req":   n_clusters,
                "connectivity":     connectivity,
                "k_found":          k_found,
                "q":                qs["q"],
                "q_p_value":        qs["p_value"],
                "q_significant":    qs["significant"],
                "q_F":              qs["F"],
                "q_lambda":         qs["lambda_"],
                "thr_cut":          thr,
                "labels":           labels,
                "lab_v":            lab_v,
                "lats":             lats,
                "lons":             lons,
            }
        except Exception as e:
            if verbose:
                print(f"  [skip] {linkage}/{metric}/mcs={min_cluster_size}"
                      f"/k={n_clusters}: {e}")
            return None

    #  Run grid search in parallel
    raw_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_run)(mcs, lnk, met, k) for mcs, lnk, met, k in param_space
    )
    results = [r for r in raw_results if r is not None]

    if not results:
        raise ValueError("No valid clustering results found.")

    #  Deduplicate identical partitions 
    ranked = sorted(results, key=lambda r: r["q"], reverse=True)
    seen, unique_results = set(), []
    for r in ranked:
        fp = tuple(r["lab_v"].tolist())
        if fp not in seen:
            seen.add(fp)
            unique_results.append(r)

    n_dropped = len(ranked) - len(unique_results)
    if verbose and n_dropped > 0:
        print(f"Deduplicated     : removed {n_dropped} identical partition(s).\n")

    #  Select best: highest q 
    best = unique_results[0]   # sorted by q DESC

    # output 
    if verbose:
        print("All configurations ranked by q-statistic (top 20):")
        print(f"  {'q':>7}  {'p-value':>10}  {'sig':>5}  {'k':>3}  "
              f"{'k_req':>6}  {'linkage':>8}  {'metric':>9}  {'mcs':>5}")
        print("  " + "-" * 70)
        for r in unique_results[:20]:
            k_req = r["n_clusters_req"] if r["n_clusters_req"] is not None else "auto"
            print(
                f"  {r['q']:7.4f}  {r['q_p_value']:10.2e}  "
                f"{str(r['q_significant']):>5s}  {r['k_found']:3d}  "
                f"{str(k_req):>6}  {r['linkage']:>8}  {r['metric']:>9}  "
                f"{r['min_cluster_size']:5d}"
            )

        print(
            f"\n── Best configuration ──────────────────────────────────\n"
            f"  linkage        : {best['linkage']}\n"
            f"  metric         : {best['metric']}\n"
            f"  connectivity   : {best['connectivity']}\n"
            f"  min_cluster_sz : {best['min_cluster_size']}\n"
            f"  k (clusters)   : {best['k_found']}\n"
            f"  q-statistic    : {best['q']:.4f}  "
            f"(p={best['q_p_value']:.2e}, "
            f"significant={best['q_significant']})\n"
            f"────────────────────────────────────────────────────────"
        )

    best_labels = best["labels"]
    best_meta   = {k: v for k, v in best.items() if k not in ("labels", "lab_v")}
    best_meta["selection_method"] = "highest_q"

    return best_labels, best_meta



def run_k_analysis(
    heatmap,
    ds,
    path: Path | None = None,
    use_knee_as_final: bool = False,
    knee_S: float = 0.7
):
    """
    Run find_best_clustering_q once per k in K_RANGE.

    Parameters
    ----------
    heatmap           : attribute heatmap passed through to find_best_clustering_q
    ds                : xarray Dataset passed through to find_best_clustering_q
    use_knee_as_final : if True, detect the q knee point after the sweep and
                        return the already-computed result for that k directly
                        (no re-run). Returns (df, best_labels, best_meta).
                        If False (default), return only df.
    knee_S            : sensitivity parameter for the Kneedle algorithm (default 1.0)

    Returns
    -------
    df                          : pd.DataFrame with one row per k tested
    best_labels, best_meta      : only returned when use_knee_as_final=True;
                                  taken directly from the sweep, no second run
    """
    K_RANGE = range(2, 15)
    
    FIXED = dict(
    min_cluster_fraction = 0.00,
    linkage_list         = ["complete","ward", "average"], #
    metric_list          = ["euclidean", "manhattan"],
    use_auto_k           = False,
    connectivity         = "queen",
    min_cluster_number   = 2,
    verbose              = False,
    n_jobs               = 1,
    )
    
    records = []   
    stored  = {}   
    n       = len(K_RANGE)

    print(f"K analysis: k_range={list(K_RANGE)[0]}–{list(K_RANGE)[-1]}")

    for i, k in enumerate(K_RANGE):
        print(f"[{i+1}/{n}]  k_req={k} ...", end=" ", flush=True)
        try:
            labels, meta = optimize_cluster(
                heatmap         = heatmap,
                ds              = ds,
                n_clusters_list = [k],
                **FIXED,
            )
            records.append({
                "k_req":            k,
                "k_found":          meta["k_found"],
                "q":                meta["q"],
                "q_significant":    meta["q_significant"],
                "q_p_value":        meta["q_p_value"],
                "selection_method": meta.get("selection_method", ""),
            })
            stored[k] = (labels, meta)
            print(f"k_found={meta['k_found']}  "
                  f"q={meta['q']:.4f}  "
                  f"sig={meta['q_significant']}")
        except Exception as e:
            print(f"ERROR: {e}")
            records.append({
                "k_req":            k,
                "k_found":          np.nan,
                "q":                np.nan,
                "q_significant":    False,
                "q_p_value":        np.nan,
                "selection_method": "error",
            })

    df    = pd.DataFrame(records)
    valid = df[df["k_found"].notna()]

    # print summary
    best_per_k_found = valid.groupby("k_found")["q"].max()
    print("\nBest q per k_found:")
    print(best_per_k_found)

    if not use_knee_as_final:
        return df

    # Kneedle on the q-curve
    knee_k = find_knee(
        valid["k_req"].values,
        valid["q"].values,
        S=knee_S,
        curve="concave",
    )

    if knee_k is None or knee_k not in stored:
        print("\nuse_knee_as_final=True but no knee point detected – "
              "falling back to highest-q row.")
        knee_k = int(valid.loc[valid["q"].idxmax(), "k_req"])

    print(f"\nKnee point: k={knee_k} – extracting result from sweep (no re-run).")
    best_labels, best_meta = stored[knee_k]
    best_meta["knee_k"]    = knee_k
    print(f"Final result: k_found={best_meta['k_found']}  "
          f"q={best_meta['q']:.4f}  sig={best_meta['q_significant']}")
    
    
    print(f"Knee-k : {best_meta['knee_k']}")
    print(f"k_found: {best_meta['k_found']}")
    print(f"q      : {best_meta['q']:.4f}")

    summarize_k(df, S = knee_S)
    if path is not None:
        plot_k_analysis(df,S=knee_S,save_path=path/"k_analysis.svg")

    return df, best_labels, best_meta


def smooth_cluster(
    ds,
    labels,
    stage_size: int = 3,
    min_cluster_size: float = 0.10,
    split: bool = True,
    sort_by: str = "south",
):
    """
    Smooth and refine cluster labels in a single stage.

    1. Majority filter (Voronoi extension)
    2. Split disconnected components (optional)
    3. Merge small clusters into neighbors
    4. Relabel to consecutive IDs

    Parameters
    ----------
    ds               : xarray Dataset with 'lat' and 'lon' coords
    labels           : array-like, shape (lat, lon)
    stage_size       : kernel size for majority filter
    min_cluster_size : clusters smaller than this are merged away
    split            : if True, split disconnected components first

    Returns
    -------
    labels   : 2D np.ndarray, consecutive integer IDs (NaN for invalid)
    mask     : bool array, True where labels are finite and >= 0
    all_mask : list of xr.DataArray, one binary mask per cluster
    """
    lat_n = len(ds["lat"])
    lon_n = len(ds["lon"])

    labels = np.array(labels, float).reshape(lat_n, lon_n)
    labels = np.where(np.isfinite(labels), np.round(labels).astype(float), np.nan)
    mask   = np.isfinite(labels) & (labels >= 0)

    # Majority filter 
    labels = majority_filter_voronoi(labels, mask, size=stage_size)

    # Split disconnected components
    if split:
        labels = split_connected_components(labels)
    
    lat_vals = np.asarray(ds["lat"].values)
    lon_vals = np.asarray(ds["lon"].values)


    def cluster_top_left(cid):
        """
        Geometric sorting key for renumbering cluster IDs, since the labels from
        AgglomerativeClustering carry no spatial meaning. "south" sorts ascending
        by latitude centroid and is robust, "north" uses the northernmost pixel.
        """
        rows, cols = np.where(labels == cid)

        if sort_by == "north":
            return (-float(lat_vals[rows].max()), float(lon_vals[cols].min()))
        if sort_by == "south":
            return (float(lat_vals[rows].mean()), float(lon_vals[cols].mean()))

        raise ValueError(f"unknown sort_by {sort_by}")
    
    finite = np.isfinite(labels)
    unique_ids = np.unique(labels[finite])
    sorted_ids = sorted(unique_ids, key=cluster_top_left)
    new_ids = {old: float(new) for new, old in enumerate(sorted_ids)}
    labels_pre = np.full(labels.shape, np.nan, dtype=float)
    labels_pre[finite] = np.vectorize(new_ids.get)(labels[finite])
    labels = labels_pre

    # Merge small clusters 
    n_valid = int(np.count_nonzero(mask))

    if min_cluster_size < 1:
        min_size = int(np.ceil(min_cluster_size * n_valid))
        print(f"min_cluster_size = {min_size} ({min_cluster_size:.0%} of {n_valid} cells)")
    else:
        min_size = int(min_cluster_size)
        print(f"min_cluster_size = {min_size} cells (fixed, {min_size / n_valid:.0%} of {n_valid})")

    labels = merge_small_with_small(labels, min_size)
    labels = merge_remaining_small(labels, min_size)

    #  Relabeling 
    finite = np.isfinite(labels)
    unique_ids = np.unique(labels[finite])
    sorted_ids = sorted(unique_ids, key=cluster_top_left)
    new_ids = {old: new for new, old in enumerate(sorted_ids)}
    labels_out = np.full(labels.shape, np.nan, dtype=float)
    labels_out[finite] = np.vectorize(new_ids.get)(labels[finite]).astype(float)
    labels = labels_out

    labels_da = xr.DataArray(
        labels,
        coords={"lat": ds["lat"], "lon": ds["lon"]},
        dims=("lat", "lon"),
        name="cluster_labels",
    )

    cluster_ids = np.unique(labels[np.isfinite(labels)]).astype(int)
    all_mask    = [(labels_da == cid).rename(f"cluster_{cid}") for cid in cluster_ids]

    return labels, mask, all_mask



################# RQA ################

def compute_rain_rolling(
    rain_og,
    window_size = 365,
    step        = 30,
    median_w    = "365D",
    alpha       = 0.05,
    dry_threshold = 0.0, 
):
    """
    Compute rolling window rain and dryness series for one community.

    Slides a window of window_size days (step days apart) over the
    community's per pixel rainfall, and for each window returns the
    median rainfall, the fraction of wholly dry days (all pixels dry),
    and the fraction of dry pixels averaged over the window (dry node
    fraction, same definition as in plot_ts). Each series also gets a
    rolling median/low/high band over median_w for later plotting.

    Returns a dataframe with one row per window.
    """
    units = rain_og.attrs.get("units", "").strip().lower()
    if units in ("m", "meters", "metres"):
        rain_og = rain_og * 1000
    
    rain_mean   = rain_og.mean(dim="features").values
    time_values = pd.to_datetime(rain_og.time.values)
    wholly_dry  = (rain_og <= dry_threshold).all(dim="features").values

    dry_node_ts = (rain_og <= dry_threshold).mean(dim="features").values

    windows = [
        (start, start + window_size)
        for start in range(0, len(time_values) - window_size + 1, step)
    ]
    window_times = [time_values[s + window_size // 2] for s, e in windows]
    rain_per_window = [
        np.median(rain_mean[s:e]) for s, e in windows
    ]
    dry_fraction_per_window = [
        wholly_dry[s:e].mean() for s, e in windows
    ]
    dry_node_fraction_per_window = [
        dry_node_ts[s:e].mean() for s, e in windows
    ]

    df = pd.DataFrame({
        "time":              window_times,
        "RAIN":              rain_per_window,
        "DRY_FRACTION":      dry_fraction_per_window,
        "DRY_NODE_FRACTION": dry_node_fraction_per_window,
    }).set_index("time")


    df["RAIN_med"]  = df["RAIN"].rolling(median_w, center=True, min_periods=1).median()
    df["RAIN_low"]  = df["RAIN"].rolling(median_w, center=True, min_periods=1).quantile(alpha / 2)
    df["RAIN_high"] = df["RAIN"].rolling(median_w, center=True, min_periods=1).quantile(1 - alpha / 2)
    df["DRY_FRACTION_med"]  = df["DRY_FRACTION"].rolling(median_w, center=True, min_periods=1).median()
    df["DRY_FRACTION_low"]  = df["DRY_FRACTION"].rolling(median_w, center=True, min_periods=1).quantile(alpha / 2)
    df["DRY_FRACTION_high"] = df["DRY_FRACTION"].rolling(median_w, center=True, min_periods=1).quantile(1 - alpha / 2)
    df["DRY_NODE_FRACTION_med"]  = df["DRY_NODE_FRACTION"].rolling(median_w, center=True, min_periods=1).median()
    df["DRY_NODE_FRACTION_low"]  = df["DRY_NODE_FRACTION"].rolling(median_w, center=True, min_periods=1).quantile(alpha / 2)
    df["DRY_NODE_FRACTION_high"] = df["DRY_NODE_FRACTION"].rolling(median_w, center=True, min_periods=1).quantile(1 - alpha / 2)


    return df.reset_index()

def rqa_full(
    all_mask,
    ds,
    ds_og,
    plot_path,
    cent_mask        = None,
    t_w_dict         = None,       
    embeddings       = None,       
    use_cosine       = False,
    boot_colors      = boot_colors,
    q                = 0.5,
    alpha            = 0.05,
    window_size      = 365,
    step             = 30,
    median_w         = "365D",
    mark_dates       = None,
    save_csv         = False,
    csv_name         = "default",
    out_dir          = "rqa_rW",
    show             = False,
    rp_path          = None,
    rp_select        = None,
    verbose          = False,
):
    """
    Run rolling-window multivariate RQA per community (all_mask), merge in
    the rolling rain/dryness metrics, produce the combined (all-community)
    diagnostic plots, and export per-community Mann-Kendall trend results
    to CSV (no per-community plots).

    Returns a list with one result dict per community, each holding the
    rolling RQA/rain metrics (including DRY_FRACTION and DRY_NODE_FRACTION)
    under "rolling_anual".
    """
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    n_clusters = len(all_mask)
 
    # ── colour palette ────────────────────────────────────────────────────────
    base_boot_colors = list(boot_colors) if boot_colors is not None else []
    if len(base_boot_colors) >= n_clusters:
        boot_colors = base_boot_colors[:n_clusters]
    else:
        fallback = []
        cmap = plt.get_cmap("tab20", max(n_clusters, 1))
        for i in range(n_clusters):
            if i < len(base_boot_colors):
                fallback.append(base_boot_colors[i])
                continue
            rgb_before = np.array(cmap(i % cmap.N)[:3])
            rgb_after  = rgb_before + (1.0 - rgb_before) * 0.35
            fallback.append({
                "before": mpl.colors.to_hex(rgb_before),
                "after":  mpl.colors.to_hex(np.clip(rgb_after, 0.0, 1.0)),
            })
        boot_colors = fallback
        print(
            f"[rqa_full] boot_colors expanded from {len(base_boot_colors)} "
            f"to {n_clusters} using tab20 fallback."
        )
 
    # ── cent_mask → point masks ───────────────────────────────────────────────
    if cent_mask is not None:
        point_masks = []
        for item in cent_mask:
            if isinstance(item, tuple):
                cluster_id, lat, lon, val = item
                mask = xr.full_like(ds["rainfall"].isel(time=0), False, dtype=bool)
                mask.loc[dict(lat=lat, lon=lon)] = True
                point_masks.append(mask)
            else:
                point_masks.append(item)
        all_mask = point_masks
 
    # ── pre-compute vmax and rain series ─────────────────────────────────────
    global_vmax = 0
    all_rain    = []

    for cluster_mask in all_mask:
        mask_computed = cluster_mask.reset_coords(drop=True).compute()
        rain = (
            ds_og.where(mask_computed, drop=True)
            .rainfall
            .stack(features=("lat", "lon"))
            .dropna(dim="features", how="any")
        )
        
        rain = rain * _rain_scale(ds_og)
        
        #rain threshold
        print("Rain stats:")
        print(rain.max().values)   
        print(rain.mean().values) 
        rain = rain.where(rain >= 0.0001, 0.0)
        
        all_rain.append(rain)
        community_max = float(rain.median(dim="features").max())
        global_vmax   = max(global_vmax, community_max)

    # ── main loop ─────────────────────────────────────────────────────────────
    print("\nRolling window RQA...")
    all_results = []
 
    for i, cluster_mask in enumerate(all_mask):
        cluster_id   = i + 1
        cluster_name = f"Community {cluster_id}"
        print(f"  → {cluster_name}")

        rain = all_rain[i]
        dry_stats = plot_ts(rain, cluster_name, plot_path, v_max=global_vmax)
        print(f"  [{cluster_name}] dry days: {dry_stats['dry_count']} "
              f"({dry_stats['dry_fraction']*100:.1f}%) | "
              f"dry nodes avg: {dry_stats['dry_node_fraction']*100:.1f}%")

        X, time_values = extract_X(ds, cluster_mask)
        t_w = int(t_w_dict.get(cluster_name, 0) or 0) if t_w_dict else 0
 
        res = rolling_window_rqa(
            X                = X,
            time_values      = time_values,
            t_w              = t_w,     
            verbose          = verbose,
            eps              = None,
            q                = q,
            use_cosine       = use_cosine,
            window_size      = window_size,
            step             = step,
            alpha            = alpha,
            n_jobs           = 2,
            cluster_name     = cluster_name,
            median_w         = median_w,
            rp_path          = (
                rp_path / f"rp_community_{cluster_id}.svg"
                if rp_path is not None else None
            ),
            rp_select        = rp_select,
        )
        
        df_rain = compute_rain_rolling(
            rain_og     = all_rain[i],
            window_size = window_size,  
            step        = step,         
            median_w    = median_w,      
            alpha       = alpha,       
        )
        res["rolling_anual"] = res["rolling_anual"].merge(
            df_rain[["time", "RAIN", "RAIN_med", "RAIN_low", "RAIN_high",
                     "DRY_FRACTION", "DRY_FRACTION_med", "DRY_FRACTION_low", "DRY_FRACTION_high",
                     "DRY_NODE_FRACTION", "DRY_NODE_FRACTION_med", "DRY_NODE_FRACTION_low", "DRY_NODE_FRACTION_high"]],
            on="time",
            how="left",
        )
        
        # debug: confirm surrogate columns made it through
        surr_cols = [c for c in res["rolling_anual"].columns if "surr" in c]
        print(f"Surrogate columns in rolling_anual: {surr_cols}")

        res["DRY_FRACTION"]      = res["rolling_anual"]["DRY_FRACTION"].values
        res["DRY_NODE_FRACTION"] = res["rolling_anual"]["DRY_NODE_FRACTION"].values
        
        all_results.append(res)
    
    
   
    split_times_dt = (
        [pd.to_datetime(f"{y_end}-01-01") for y_start, y_end in mark_dates]
        if mark_dates is not None else []
    )
    
    y_limits   = {}

    for metric in ["DET", "LAM", "ENTR","LMEAN","DRY_FRACTION","DRY_NODE_FRACTION"]: 
        all_vals = np.concatenate([
            res[metric] for res in all_results
        ] + [
            res["rolling_anual"][f"{metric}_low"].dropna().values
            for res in all_results
        ] + [
            res["rolling_anual"][f"{metric}_high"].dropna().values
            for res in all_results
        ])
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals) == 0:
            print(f"WARNING: no finite values for metric {metric} – using fallback limits")
            y_limits[metric] = (0, 1)
            continue
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        pad = (vmax - vmin) * 0.10
        y_limits[metric] = (vmin - pad, vmax + pad)

        
    rain_vals = np.concatenate([
        res["rolling_anual"]["RAIN"].dropna().values
        for res in all_results
    ])
    rain_vals = rain_vals[np.isfinite(rain_vals)]
    if len(rain_vals) == 0:
        y_limits["RAIN"] = (0, 1)
    else:
        vmin, vmax = np.nanmin(rain_vals), np.nanmax(rain_vals)
        pad = (vmax - vmin) * 0.10
        y_limits["RAIN"] = (vmin - pad, vmax + pad)

    # ── combined plots ────────────────────────────────────────────────────────
    for metric in ["DET", "LAM", "ENTR", "RAIN","LMEAN","DRY_FRACTION", "DRY_NODE_FRACTION"]:
        fig, ax = plt.subplots(figsize=(12, 5))

        event_handles = []

        for i, cluster_dict in enumerate(all_results):
            df_roll = cluster_dict["rolling_anual"]
            time_i  = pd.to_datetime(df_roll["time"].values)
            y       = df_roll[metric].values
            t_roll  = df_roll["time"]

            ax.plot(time_i, y,
                        color=boot_colors[i]["before"], lw=0.9,
                        label=f"Cluster {i+1}")
            ax.plot(t_roll, df_roll[f"{metric}_med"],
                        color=boot_colors[i]["before"], lw=2)
            ax.fill_between(
                    t_roll,
                    df_roll[f"{metric}_low"],
                    df_roll[f"{metric}_high"],
                    color=boot_colors[i]["before"], alpha=0.15,
                )

        ax.set_ylim(y_limits[metric])
        for st in split_times_dt:
            ax.axvline(st, color="grey", lw=1, linestyle="-")
        ax.set_title(metric)
        ax.set_xlabel("Time")
        ax.set_ylabel(metric)
        apply_grid(ax)
        community_handles, community_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles = event_handles + community_handles,
            labels  = [h.get_label() for h in event_handles] + community_labels,
            frameon = False, ncol = 4,
            loc     = "upper center", bbox_to_anchor=(0.5, -0.15),
        )
        plt.tight_layout()
        fig.savefig(plot_path / f"{metric}.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # per-cluster Mann-Kendall trend export (CSV only, no plots)
    print("\nIndividual cluster trend export")

    out_dir_path = Path(plot_path) / out_dir
    out_dir_path.mkdir(parents=True, exist_ok=True)

    all_mk = []
    for i, res in enumerate(all_results):
        print(f"\nCluster {i+1}")
        df = res["rolling_anual"]
        for metric in ["DET", "LAM", "ENTR","RAIN","LMEAN","DRY_FRACTION","DRY_NODE_FRACTION"]:
            result = plot_metric_rolling(
                        df         = df,
                        metric     = metric,
                        cluster_id = i + 1,
                        mark_dates = mark_dates,
                        save_csv   = save_csv,
                        out_dir    = out_dir_path,
                        csv_name   = csv_name,
                    )
            all_mk.append(result)
    
    rows = []
    for r in all_mk:
        for p in r["periods"]:
            rows.append({"cluster_id": r["cluster_id"], "metric": r["metric"], **p})

    mk_table = pd.DataFrame(rows)
    mk_table.to_csv(out_dir_path / f"mk_results_{csv_name}.csv", index=False)
    print(f"saved MK table: {out_dir_path / f'mk_results_{csv_name}.csv'}")                        
    
    return all_results
    
    

    
    
    
