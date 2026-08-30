"""
plots_and_stats.py

Plotting and export utilities used across the pipeline.

1. Styling helpers      - style_axes, set_aspect_latlon, cluster_norm,
                           get_cluster_cmap, get_cluster_norm
2. Community export     - export_communities (masks -> shapefile)
3. Cluster/heatmap plots- plot_cluster, plot_smoothing, plot_two_panel, plot_ts
4. RQA/trend plots      - plot_rqa_with_rain, plot_mk_dotplot
                           (Mann-Kendall dot plots with significance)
"""



import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
from cmcrameri import cm
from matplotlib.colors import ListedColormap,BoundaryNorm
import matplotlib as mpl
from pathlib import Path
from shapely.geometry import shape
import geopandas as gpd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.ops import unary_union
import rasterio


from rqa import apply_grid


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
    """Return appropriate norm for k clusters.
    If k <= len(CLUSTER_COLORS), use BoundaryNorm.
    Otherwise, use continuous norm (0 to k-1)."""
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
    

def set_aspect_latlon(ax, lats, lons):
    """Set a geographically correct aspect ratio for lat/lon axes, based on the mean latitude."""
    lat_center = (np.nanmax(lats) + np.nanmin(lats)) / 2
    aspect = 1 / np.cos(np.radians(lat_center))
    ax.set_aspect(aspect)

def style_axes(ax):
    """Apply the standard white background, major/minor grid, and tick style to an axes."""
    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=9)
    

    
    


def export_communities(all_mask, ds, output_path):
    """
    Save community masks as a single shapefile with a 'community' attribute.

    Parameters
    ----------
    all_mask    : list of xr.DataArray, one binary mask per cluster
    ds          : xarray Dataset with 'lat' and 'lon' coords
    output_path : str or Path, path to output .shp file
    """


    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lats = ds["lat"].values
    lons = ds["lon"].values
    res_lat = abs(lats[1] - lats[0])
    res_lon = abs(lons[1] - lons[0])

    transform = rasterio.transform.from_bounds(
        west  = lons.min() - res_lon / 2,
        south = lats.min() - res_lat / 2,
        east  = lons.max() + res_lon / 2,
        north = lats.max() + res_lat / 2,
        width  = len(lons),
        height = len(lats),
    )

    rows = []
    for i, mask_da in enumerate(all_mask):
        arr = mask_da.values.astype(np.uint8)
        # rasterio expects (north -> south), so flip if needed
        if lats[0] < lats[-1]:
            arr = np.flipud(arr)

        shapes = rasterio.features.shapes(arr, transform=transform)
        for geom, val in shapes:
            if val == 1:
                rows.append({
                    "geometry":  shape(geom),
                    "community": i + 1,
                })

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_file(output_path)
    print(f"Saved {len(gdf)} polygons to {output_path}")
    
    
def plot_cluster(
    k_found,
    labels,
    lons,
    lats,
    plot_path,
    title="Communities Agglomerative Clustering",
    figsize = (8,6),
    scatter = 50,
    show = False
):
    """
    Scatter-plot geographic clusters, colored by cluster ID (negative labels
    treated as noise). Saves to 'aggl_clu.png' in plot_path.
    """
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    cvals = labels.astype(float)
    cvals[cvals < 0] = np.nan


    fig, ax = plt.subplots(figsize=figsize)

    cmap = get_cluster_cmap(k_found)
    norm = get_cluster_norm(k_found)

    sc = ax.scatter(
        lons,
        lats,
        c=cvals,
        cmap=cmap,
        norm=norm,
        s=scatter,
        edgecolors="none"
    )

    style_axes(ax)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


    if k_found > 0:
        cbar = fig.colorbar(
            sc,
            ax=ax,
            orientation="vertical",   
            ticks=np.arange(k_found),
            pad=0.02
        )
        cbar.set_label("Community")
        cbar.set_ticklabels(np.arange(1, k_found + 1))

    plt.tight_layout()
    save_file = plot_path / "aggl_clu.png"
    fig.savefig(save_file, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)



def plot_smoothing(

    heatmap_values, labels1, labels2,coords_lat,coords_lons, lats, lons,plot_path,
    title1="Heatmap",
    title2="Agglomerative Clustering",
    title3="Agglomerative Smoothed Clustering",
    filename="smoothing.png",
    figsize = (16,6),
    scatter = 40,
    show = False
):
    
    """
    Three-panel scatter plot: heatmap, raw clustering, and smoothed clustering
    side by side (labels1/labels2 use -1 for noise points). Saves to filename
    in plot_path.
    """
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
 
    cvals1 = labels1.astype(float)
    cvals1[cvals1 < 0] = np.nan

    cvals2 = labels2.astype(float)
    cvals2[cvals2 < 0] = np.nan


    k1 = int(np.nanmax(cvals1)) + 1 if np.any(np.isfinite(cvals1)) else 0
    k2 = int(np.nanmax(cvals2)) + 1 if np.any(np.isfinite(cvals2)) else 0


    sc0 = axes[0].scatter(
        coords_lat, coords_lons,
        c=heatmap_values,
        cmap=cm.batlow,
        s=scatter,
        edgecolors="none"
        #vmin=0.1,
        #vmax=0.9
    )
    style_axes(axes[0])
    axes[0].set_title(title1)

    cbar0 = fig.colorbar(
        sc0, ax=axes[0],
        orientation="horizontal",
        pad=0.1, fraction=0.05
    )
    cbar0.set_label("Fraction of recurrent points")


    if k1 > 0:
        norm1 = get_cluster_norm(k1)
        cmap1 = get_cluster_cmap(k1)
    else:
        norm1 = None
        cmap1 = cluster_cmap

    sc1 = axes[1].scatter(
        lons, lats,
        c=cvals1,
        cmap=cmap1,
        norm=norm1,
        s=scatter,
        edgecolors="none"
    )
    style_axes(axes[1])
    axes[1].set_title(title2)

    if k1 > 0:
        cbar1 = fig.colorbar(
            sc1, ax=axes[1],
            orientation="horizontal",
            pad=0.1, fraction=0.05,
            ticks=np.arange(k1)
        )
        cbar1.set_label("Community")
        cbar1.set_ticklabels(np.arange(1, k1 + 1))


    if k2 > 0:
        norm2 = get_cluster_norm(k2)
        cmap2 = get_cluster_cmap(k2)
    else:
        norm2 = None
        cmap2 = cluster_cmap

    sc2 = axes[2].scatter(
        lons, lats,
        c=cvals2,
        cmap=cmap2,
        norm=norm2,
        s=scatter,
        edgecolors="none"
    )
    style_axes(axes[2])
    axes[2].set_title(title3)

    if k2 > 0:
        cbar2 = fig.colorbar(
            sc2, ax=axes[2],
            orientation="horizontal",
            pad=0.1, fraction=0.05,
            ticks=np.arange(k2)
        )
        cbar2.set_label("Community")
        cbar2.set_ticklabels(np.arange(1, k2 + 1))

    plt.tight_layout()
    save_file = plot_path / filename
    fig.savefig(save_file, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    


    

def plot_two_panel(
    heatmap_values, labels1, labels2, coords_lat, coords_lons, lats, lons,
    plot_path,
    title1="Recurence Fraction",
    title2="Clustering",
    filename="two_panel.png",
    figsize=(12, 6),
    scatter=40,
    show=False,
    mask_radius=0.4,
    shapefile_path=None,
):
    """
    Two-panel figure: interpolated recurrence-fraction heatmap next to the
    smoothed cluster map, both optionally clipped to a shapefile boundary.
    Saves to filename in plot_path.
    """

    def shapefile_clip_patch(ax, shapefile_path):
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

    def make_grid(lons_v, lats_v, values, method="nearest", smooth_factor=1, fill_nan=False):
        res_lon = np.median(np.diff(np.unique(np.sort(lons_v))))
        res_lat = np.median(np.diff(np.unique(np.sort(lats_v))))

        grid_lon, grid_lat = np.meshgrid(
            np.arange(lons_v.min(), lons_v.max() + res_lon, res_lon),
            np.arange(lats_v.min(), lats_v.max() + res_lat, res_lat),
        )

        grid_z = griddata(
            points=np.column_stack([lons_v, lats_v]),
            values=values,
            xi=(grid_lon, grid_lat),
            method=method,
        )

        tree = cKDTree(np.column_stack([lons_v, lats_v]))
        dist, _ = tree.query(np.column_stack([grid_lon.ravel(), grid_lat.ravel()]))
        mask = dist.reshape(grid_lon.shape) > max(res_lon, res_lat) * smooth_factor

        # set NaN inside the valid region to 0
        if fill_nan:
            grid_z = np.where(~mask & np.isnan(grid_z), 0.0, grid_z)

        grid_z = np.ma.array(grid_z, mask=mask)
        return grid_lon, grid_lat, grid_z

    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0.01)

    cvals2 = labels2.astype(float)
    cvals2[cvals2 < 0] = np.nan
    k2 = int(np.nanmax(cvals2)) + 1 if np.any(np.isfinite(cvals2)) else 0

    # --- Panel 1: heatmap as pcolormesh with linear interpolation ---
    valid0 = np.isfinite(heatmap_values)
    grid_lon0, grid_lat0, grid_z0 = make_grid(
        coords_lons[valid0], coords_lat[valid0],
        heatmap_values[valid0],
        method="linear",
        smooth_factor=1,
        fill_nan=True,
    )

    mesh0 = axes[0].pcolormesh(
        grid_lon0, grid_lat0, grid_z0,
        cmap=cm.lipari,
        shading="nearest",
    )
    axes[0].set_xlim(coords_lons[valid0].min() - 0.5, coords_lons[valid0].max() + 0.5)
    axes[0].set_ylim(coords_lat[valid0].min()  - 0.5, coords_lat[valid0].max()  + 0.5)


    axes[0].contour(
        grid_lon0, grid_lat0, grid_z0,
        levels=5,
        colors='white',
        linewidths=0.4,
        alpha=0.8,
    )

    if shapefile_path:
        clip0 = shapefile_clip_patch(axes[0], shapefile_path)
        mesh0.set_clip_path(clip0)

    style_axes(axes[0])
    axes[0].set_title(title1)
    set_aspect_latlon(axes[0], coords_lat, coords_lons)
    cbar0 = fig.colorbar(mesh0, ax=axes[0], orientation="horizontal", pad=0.1, fraction=0.05)
    cbar0.set_label("Fraction of recurrent points")

   #--- Panel 2: cluster areas at native resolution ---
    if k2 > 0:
        cmap2 = get_cluster_cmap(k2)
        norm2 = get_cluster_norm(k2)

        valid2 = np.isfinite(cvals2)
        grid_lon2, grid_lat2, grid_z2 = make_grid(
            lons[valid2], lats[valid2],
            cvals2[valid2],
            method="nearest",  # nearest for sharp cluster boundaries
            smooth_factor=1,
        )

        mesh2 = axes[1].pcolormesh(
            grid_lon2, grid_lat2, grid_z2,
            cmap=cmap2, norm=norm2,
            shading="nearest",
        )
        
        axes[1].set_xlim(lons[valid2].min() - 0.5, lons[valid2].max() + 0.5)
        axes[1].set_ylim(lats[valid2].min() - 0.5, lats[valid2].max() + 0.5)

        if shapefile_path:
            clip2 = shapefile_clip_patch(axes[1], shapefile_path)
            mesh2.set_clip_path(clip2)

        style_axes(axes[1])
        axes[1].set_title(title2)
        set_aspect_latlon(axes[1], lats, lons)

        sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
        sm.set_array([])
        cbar2 = fig.colorbar(sm, ax=axes[1], orientation="horizontal",
                             pad=0.1, fraction=0.05, ticks=np.arange(k2))
        cbar2.set_label("Community")
        cbar2.set_ticklabels(np.arange(1, k2 + 1))
    

    plt.tight_layout(w_pad=0.1)
    fig.savefig(plot_path / filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_ts(ds, cluster_name, plot_path, v_max=None):
    """
    Four-panel diagnostic for one community's rainfall series: time series,
    distribution histogram, annual dry-node fraction, and annual count of
    wholly dry days. Returns dry-day/dry-node statistics for the community.
    """
    if "features" in ds.dims:
        mean_rain = ds.mean(dim="features")
        raw_values_flat = ds.values.flatten()
    else:
        mean_rain = ds
        raw_values_flat = ds.values.flatten()

    times  = mean_rain.time.values
    values = mean_rain.values
    x      = np.arange(len(times))

    all_days     = len(values)
    dry_count    = int(np.sum(values == 0))
    dry_fraction = dry_count / all_days

    if "features" in ds.dims:
        dry_node_ts       = (ds == 0).mean(dim="features").values
        dry_node_fraction = float(dry_node_ts.mean())
    else:
        dry_node_ts       = (values == 0).astype(float)
        dry_node_fraction = dry_fraction

    # ── figure: 4 panels ──────────────────────────────────────────────────────
    fig, (ax_ts, ax_hist, ax_dry, ax_dry_days) = plt.subplots(
        4, 1,
        figsize=(12, 13),
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
    )

    # ── timeseries ────────────────────────────────────────────────────────────
    ax_ts.bar(x, values, color="steelblue", alpha=0.4, width=1.0)
    ax_ts.plot(x, values, lw=1.5, color="steelblue")
    
    mean_total = values.mean()
    median_total = np.median(raw_values_flat)

    n_ticks  = min(8, len(times))
    tick_idx = np.linspace(0, len(times) - 1, n_ticks, dtype=int)
    ax_ts.set_xticks(tick_idx)
    ax_ts.set_xticklabels(
        [str(times[i])[:10] for i in tick_idx],
        rotation=45, ha="right",
    )
    ax_ts.set_title(f"Rainfall – {cluster_name}")
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("Mean Rainfall")
    ax_ts.grid(True, linestyle="--", alpha=0.5, axis="y")
    
    ax_ts.text(
        0.98, 0.95,
        f"Average: {mean_total:.5f}\nMedian: {median_total:.5f}",
        transform=ax_ts.transAxes,
        ha="right", va="top",
        fontsize=9,
        color="steelblue",
        
    )
    if v_max is not None:
        ax_ts.set_ylim(0, v_max)

    # ── distribution ──────────────────────────────────────────────────────────
    ax_hist.hist(values, bins=30, color="steelblue", alpha=0.6,
                 edgecolor="white", linewidth=0.4)
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("Mean Rainfall")
    ax_hist.set_ylabel("Count (log)")
    ax_hist.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax_hist.text(
        0.98, 0.95,
        f"Dry days: {dry_count} of {all_days} ({dry_fraction*100:.1f}%)",
        transform=ax_hist.transAxes,
        ha="right", va="top",
        fontsize=9,
        color="steelblue",
    )
    if v_max is not None:
        ax_hist.set_xlim(0, v_max)

    # ── dry node fraction over time ───────────────────────────────────────────
    dry_node_da     = xr.DataArray(dry_node_ts, coords={"time": times}, dims=["time"])
    dry_node_annual = dry_node_da.resample(time="1YE").mean()

    x_annual = np.arange(len(dry_node_annual))
    t_annual = dry_node_annual.time.values

    ax_dry.bar(x_annual, dry_node_annual.values * 100,
               color="darkorange", alpha=0.6, width=0.8)
    ax_dry.plot(x_annual, dry_node_annual.values * 100,
                color="darkorange", lw=1.5)

    n_ticks_dry  = min(8, len(t_annual))
    tick_idx_dry = np.linspace(0, len(t_annual) - 1, n_ticks_dry, dtype=int)
    ax_dry.set_xticks(tick_idx_dry)
    ax_dry.set_xticklabels(
        [str(t_annual[i])[:4] for i in tick_idx_dry],
        rotation=45, ha="right",
    )
    ax_dry.set_ylabel("Dry nodes (%)")
    ax_dry.set_xlabel("Year")
    ax_dry.set_ylim(0, 100)
    ax_dry.axhline(
        dry_node_fraction * 100,
        color="darkorange", lw=1.5, ls="--",
        label=f"Mean: {dry_node_fraction*100:.1f}%",
    )
    ax_dry.legend(frameon=False, fontsize=9)
    ax_dry.grid(True, linestyle="--", alpha=0.5, axis="y")

    # ── wholly dry days per year ──────────────────────────────────────────────
    wholly_dry_da     = xr.DataArray(
        (dry_node_ts == 1).astype(float),
        coords={"time": times},
        dims=["time"],
    )
    wholly_dry_annual = wholly_dry_da.resample(time="1YE").sum()

    x_wd = np.arange(len(wholly_dry_annual))
    t_wd = wholly_dry_annual.time.values

    ax_dry_days.bar(x_wd, wholly_dry_annual.values,
                    color="firebrick", alpha=0.6, width=0.8)
    ax_dry_days.plot(x_wd, wholly_dry_annual.values,
                     color="firebrick", lw=1.5)

    n_ticks_wd  = min(8, len(t_wd))
    tick_idx_wd = np.linspace(0, len(t_wd) - 1, n_ticks_wd, dtype=int)
    ax_dry_days.set_xticks(tick_idx_wd)
    ax_dry_days.set_xticklabels(
        [str(t_wd[i])[:4] for i in tick_idx_wd],
        rotation=45, ha="right",
    )
    ax_dry_days.set_ylabel("Wholly dry days per year")
    ax_dry_days.set_xlabel("Year")
    ax_dry_days.axhline(
        wholly_dry_annual.values.mean(),
        color="firebrick", lw=1.5, ls="--",
        label=f"Mean: {wholly_dry_annual.values.mean():.1f}",
    )
    ax_dry_days.legend(frameon=False, fontsize=9)
    ax_dry_days.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    fig.savefig(
        plot_path / f"rainfall_{cluster_name.replace(' ', '_')}.svg",
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "dry_count":         dry_count,
        "dry_fraction":      round(float(dry_fraction), 3),
        "dry_node_fraction": round(float(dry_node_fraction), 3),
        "dry_node_ts":       dry_node_ts,
    }
    
    

def plot_rqa_with_rain(
    all_results,
    all_mask,
    plot_path,
    metrics      = ["DET", "LAM", "ENTR", "LMEAN", "RAIN", "DRY_FRACTION", "DRY_NODE_FRACTION"],
    split_times  = None,
    boot_colors  = boot_colors,
    show         = False,
):
    """
    Per-community figure with one row per metric, showing the rolling
    series, its smoothed median, and confidence band over time. Optional
    vertical lines mark split_times. Saves one PNG per community.
    """
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)
    
    splits = [pd.to_datetime(st) for st in split_times] if split_times is not None else []

    for i, (res, mask) in enumerate(zip(all_results, all_mask)):
        cluster_id   = i + 1
        cluster_name = f"Community {cluster_id}"
        color        = boot_colors[i] if i < len(boot_colors) else {"before": "steelblue", "after": "grey"}

        df = res["rolling_anual"].copy()
        df["time"] = pd.to_datetime(df["time"])

        n_rows = len(metrics)
        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(12, 3 * n_rows),
            sharex=True,
        )
        if n_rows == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            t = df["time"]
            y = df[metric].values

            # rolling metric time series
            ax.plot(t, y, color=color["before"], lw=0.8, alpha=0.6, zorder=2)
            ax.plot(t, df[f"{metric}_med"], color=color["before"], lw=2, zorder=3)
            ax.fill_between(
                t,
                df[f"{metric}_low"],
                df[f"{metric}_high"],
                color=color["before"], alpha=0.15, zorder=1,
            )

            # period split markers
            for st in splits:
                ax.axvline(st, color="grey", lw=1, ls="--", zorder=4)

            ax.set_ylabel(metric)
            apply_grid(ax)


        axes[-1].set_xlabel("Time")
        fig.suptitle(cluster_name, fontsize=12)
        plt.tight_layout()
        fig.savefig(
            plot_path / f"C{cluster_id:02d}_rqa_rain.png",
            dpi=300, bbox_inches="tight"
        )
        if show:
            plt.show()
        plt.close(fig)
        
        
        
        
         
def plot_mk_dotplot(
    csv_path,
    plot_path,
    filename="mk_dotplot.svg",
    figsize=(14, 10),
    show=False,
    periods=("full",),
    metrics=("DET", "LAM", "ENTR", "LMEAN","DRY_FRACTION"),
    colors=None,
):
    """
    Grid of per-community dot plots (one subplot per cluster) showing the
    Mann-Kendall slope of each metric as a colored dot, sized by
    -log10(p) and outlined when significant.
    """
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df["period"].isin(periods)]
    df = df[df["metric"].isin(metrics)]
    df = df.dropna(subset=["slope", "p"])

    df["neg_log_p"] = -np.log10(df["p"].clip(lower=1e-10))
    df["sig"]       = df["p"] < 0.05

    if colors is None:
        colors = {"DET": "#6094e8", "LAM": "#b91109", "ENTR": "#dbaf4d", "LMEAN": "#8c36ff", "DRY_FRACTION":"#f76906" }

    clusters    = sorted(df["cluster_id"].unique())
    n_clusters  = len(clusters)
    period_list = list(periods)
    n_periods   = len(period_list)
    metric_list = list(metrics)

    n_metrics = len(metric_list)
    offsets   = np.linspace(-0.25, 0.25, n_metrics)

    n_cols = 2
    n_rows = int(np.ceil(n_clusters / n_cols))
    x_max  = df["slope"].abs().max() * 1.2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=False, sharex=False)
    axes_flat = axes.flatten()

    for j in range(n_clusters, len(axes_flat)):
        axes_flat[j].set_visible(False)

    for ci, cluster_id in enumerate(clusters):
        ax   = axes_flat[ci]
        df_c = df[df["cluster_id"] == cluster_id]

        for pi, period in enumerate(period_list):
            df_p = df_c[df_c["period"] == period]

            for mi, metric in enumerate(metric_list):
                row = df_p[df_p["metric"] == metric]
                if row.empty:
                    continue

                slope  = row["slope"].values[0]
                neg_lp = row["neg_log_p"].values[0]
                sig    = row["sig"].values[0]
                color  = colors.get(metric, "grey")

                y     = pi + offsets[mi]
                size  = max(30, neg_lp * 40)
                alpha = 1 if sig else 0.5 

                ax.scatter(
                    slope, y,
                    s          = size,
                    color      = color,
                    alpha      = alpha,
                    edgecolors = "black" if sig else "none",
                    linewidths = 0.8,
                    zorder     = 3,
                    label      = metric if pi == 0 else None,
                )

        ax.axvline(0, color="grey", lw=1, ls="--", zorder=1)
        ax.set_yticks(range(n_periods))
        ax.set_yticklabels([str(p) for p in period_list], fontsize=9)
        ax.set_title(f"Community {cluster_id}", fontsize=11)
        ax.set_xlabel("MK Slope", fontsize=9)
        ax.set_xlim(-x_max, x_max)
        apply_grid(ax)

    handles = [
        plt.scatter([], [], s=60, color=colors.get(m, "grey"), label=m)
        for m in metric_list
    ] + [
        plt.scatter([], [], s=40, color="grey", alpha=0.25,
                    label="not significant"),
        plt.scatter([], [], s=40, color="grey", alpha=0.85,
                    label="p < 0.05"),
    ]
    fig.legend(
        handles          = handles,
        frameon          = False,
        ncol             = 6,
        loc              = "lower center",
        bbox_to_anchor   = (0.5, 0.0),
        fontsize         = 9,
    )

    for ax in axes_flat[:n_clusters]:
        ax.set_xlim(-x_max, x_max)

    plt.suptitle("Mann-Kendall Trends per Community", fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(plot_path / filename, format="svg", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)