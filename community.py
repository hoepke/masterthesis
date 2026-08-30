"""
community.py

Pipeline for identifying quasi-periodic regions and defining spatially
coherent rainfall communities.

1. PCA / spectral filtering   - perform_svd, compute_power_spectrum,
                                 compute_spectral_concentration
2. Sliding-window heatmap      - process_window, analyze_rainfall_data
                                 (Kneedle threshold per window -> recurrence
                                 fraction heatmap)
3. Agglomerative clustering    - raw_cluster_aggl, q_statistic, find_knee,
                                 summarize_k (k chosen via Kneedle on
                                 q-statistic curve)
4. Mask post-processing        - pad_nearest_valid, majority_filter_voronoi,
                                 split_connected_components,
                                 merge_small_with_small,
                                 merge_remaining_small
                                 (spatial smoothing -> final communities)
5. Export / plotting helpers   - export_heatmap, plot_heatmap,
                                 clip_to_shapefile, style_axes
"""


from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.ticker as mticker
import pandas as pd
import random

import xarray as xr
import rioxarray


from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from cmcrameri import cm
from scipy.sparse import coo_matrix
from scipy import ndimage
from scipy.stats import f as f_dist
from kneed import KneeLocator
import warnings
import os
from pathlib import Path

import networkx as nx
import geopandas as gpd



##################################### DESIGN & HELPER FUNCTIONS #####################################

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

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


from rqa import apply_grid 
CLUSTER_COLORS = [
'#6094e8','#c9485b', '#dbaf4d','#849b0a',"#8c36ff", "#f76906","#088267"
]

cluster_cmap = ListedColormap(CLUSTER_COLORS)

def style_axes(ax):
    """Apply the standard white background, major/minor grid, and tick style to an axes."""
    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=9)

##################################### PCA #####################################


def perform_svd(Bm):
        """SVD of the (space x time) matrix, rescaled so Z holds standardized PC time series and A the loadings."""
        U, S, VT = np.linalg.svd(Bm, full_matrices=False)
        N = Bm.shape[1]
        Z = np.sqrt(N) * VT
        A = (1 / np.sqrt(N)) * U @ np.diag(S)
        return Z, A
    

def compute_power_spectrum(Z):
        """Peak-normalized power spectrum of each PC time series in Z."""
        Z_hat = np.fft.rfft(Z, axis=1)
        Pj = (np.abs(Z_hat) ** 2) / Z.shape[1]
        if Pj.shape[1] > 2:
            Pj[:, 1:-1] *= 2
        mx = np.max(Pj, axis=1, keepdims=True)
        mx = np.where(mx == 0, 1.0, mx)
        Pj = Pj / mx
        return Pj

def _half_height_window_indices(spectrum, peak_idx):
        """Index range around peak_idx where the spectrum stays above half its peak value."""
        half = spectrum[peak_idx] / 2.0
        l = peak_idx
        while l > 0 and spectrum[l] >= half:
            l -= 1
        r = peak_idx
        while r < len(spectrum) - 1 and spectrum[r] >= half:
            r += 1
        left = min(peak_idx, max(0, l + 1))
        right = max(peak_idx, min(len(spectrum) - 1, r - 1))
        return left, right, half

def compute_spectral_concentration(Pj):
        """Spectral concentration (SC) of each PC: the fraction of power inside the half-height window around its dominant frequency."""
        dominant_idx = np.argmax(Pj, axis=1)
        SC = np.zeros(Pj.shape[0])
        left_idx = np.zeros(Pj.shape[0], dtype=int)
        right_idx = np.zeros(Pj.shape[0], dtype=int)
        half_vals = np.zeros(Pj.shape[0])
        totals = np.sum(Pj, axis=1)

        for j in range(Pj.shape[0]):
            l, r, h = _half_height_window_indices(Pj[j], dominant_idx[j])
            left_idx[j], right_idx[j], half_vals[j] = l, r, h
            SC[j] = np.sum(Pj[j, l:r + 1]) / totals[j] if totals[j] > 0 else 0.0
        
        return SC, dominant_idx, left_idx, right_idx, half_vals

def identify_relevant_points(A, relevant_pcs):
    """Points whose loading on any relevant PC exceeds 3 standard deviations."""
    if len(relevant_pcs) == 0:
        return np.array([], dtype=int)
    
    relevant_points = []
    for j in relevant_pcs:
        a_j = np.abs(A[:, j])
        thr = 3 * np.std(a_j)
        relevant_points.append(np.where(a_j > thr)[0])
    
    all_points = np.unique(np.concatenate(relevant_points)) if relevant_points else np.array([], dtype=int)
    return all_points

 
def build_sc_threshold_curve(SC, base=0):
    """Number/fraction of PCs remaining as the SC threshold is swept upward, for Kneedle detection."""
    SC = np.asarray(SC)
    SC = SC[np.isfinite(SC)]
    
    if SC.size == 0:
        return np.array([base]), np.array([0]), np.array([0.0])

    thresholds = np.sort(np.unique(SC))
    thresholds = thresholds[thresholds >= base]
    thresholds = np.append(thresholds, 1.0)  
    
    if len(thresholds) == 0:
        return np.array([base]), np.array([0]), np.array([0.0])

    n_selected = np.array([np.sum(SC > t) for t in thresholds], dtype=int)
    frac_selected = n_selected / SC.size

    return thresholds, n_selected, frac_selected
    
def kneedle_threshold_from_curve(thresholds, n_selected):
    """Kneedle knee point of the (threshold, n_selected) curve, used as the automatic SC cutoff."""
    x = np.asarray(thresholds, dtype=float)
    y = np.asarray(n_selected, dtype=float)

    if x.size == 0:
        return 0.1
    if x.size == 1:
        return float(x[0])
    
    y = y / y[0]

    kneedle = KneeLocator(
        x, y,
        curve="convex",
        direction="decreasing",
        S=1.0,
    )

    t_knee = kneedle.knee
    if t_knee is None:
        return float(x[0])

    return float(t_knee)


def plot_sc_curve(thresholds, n_selected, t_knee, window_label, plot_path, show=False):
    """Plot the normalized SC threshold curve for one window, marking the detected knee point."""
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    x = np.asarray(thresholds, dtype=float)
    y = np.asarray(n_selected, dtype=float)
    y = y / y[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, linewidth=2, label="PCs(SC > t) normalized")

    if t_knee is not None:
        ax.axvline(t_knee, linestyle="--", color="red",
                   label=f"t*={t_knee:.3f}")

    ax.set_xlabel("SC threshold t")
    ax.set_ylabel("Fraction of PCs remaining")
    ax.set_title(f"SC threshold ({window_label})")
    ax.legend()
    style_axes(ax)
    plt.tight_layout()

    label_str = str(window_label)
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label_str).strip("_")
    if not safe_label:
        safe_label = "window"

    plt.savefig(plot_path / f"{safe_label}_sc_curve.svg", format="svg", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    
    
def plot_community_size_distribution(community_size_distribution, min_community_size,
                                      plot_path, show=False, filename="c_distribution",
                                      cutoff = None,sizes_sorted=None , ccdf_vals=None):
    """Plot the histogram and CCDF of community sizes, with the chosen min_community_size marked."""
    sizes = np.array(community_size_distribution)
    if len(sizes) == 0:
        print("No communities found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    max_size = min(sizes.max(), 50)
    bins = np.arange(1, max_size + 2) - 0.5
    axes[0].hist(sizes, bins=bins, edgecolor='black')
    axes[0].axvline(min_community_size, color='red', linestyle='--', linewidth=1.5,
                    label=f"min_community_size = {min_community_size}")
    axes[0].legend()
    axes[0].set_xlabel("Community size (number of nodes)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of community sizes")
    axes[0].set_yscale('log')
    axes[0].set_ylim(bottom=0.9)

    if sizes.max() > 50:
        axes[0].text(0.98, 0.98, f"max size = {sizes.max()} (truncated at 50)",
                    transform=axes[0].transAxes,
                    ha='right', va='top', fontsize=9, color='gray')

    # CCDF
    sorted_sizes = np.sort(sizes)
    ccdf_full = 1 - np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
    axes[1].plot(sorted_sizes, ccdf_full, color='steelblue', label='CCDF')
    axes[1].axvline(min_community_size, color='red', linestyle='--', linewidth=1.5,
                    label=f"min_community_size = {min_community_size}")
    
    if cutoff is not None:
        axes[1].axvspan(0, cutoff, alpha=0.08, color='gray',
                        label=f'convex region (0–{cutoff:.0f})')
        axes[1].axvline(cutoff, color='orange', linestyle=':', linewidth=1.5,
                        label=f"convex cutoff = {cutoff:.0f}")
    
        
    if sizes_sorted is not None and ccdf_vals is not None:
        axes[1].plot(sizes_sorted, ccdf_vals,
                    color='green', linewidth=1.5, linestyle='--',
                    label='CCDF (arange >2, for knee)')
    
    axes[1].legend()
    axes[1].set_xlabel("Community size")
    axes[1].set_ylabel("Fraction of communities $\\geq$ size")
    axes[1].set_title("Complementary CDF")
    axes[1].set_yscale('log')

    axes[1].set_facecolor("white")
    axes[1].figure.patch.set_facecolor("white")
    axes[1].grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    axes[1].grid(False, which="minor")
    axes[1].tick_params(labelsize=9)

    plt.tight_layout()

    if plot_path:
        Path(plot_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(plot_path) / f"{filename}.svg", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    print(f"Total communities:  {len(sizes)}")
    print(f"Size == 2:          {np.sum(sizes == 2)} ({100*np.mean(sizes == 2):.1f}%)")
    print(f"Size >= 10:         {np.sum(sizes >= 10)} ({100*np.mean(sizes >= 10):.1f}%)")
    print(f"Median size:        {np.median(sizes):.1f}")
    print(f"Max size:           {sizes.max()}")
    


def compute_min_community_size_knee(community_size_distribution):
    """Kneedle knee point of the community-size CCDF, used as the automatic min_community_size."""
    sizes_arr = np.array(community_size_distribution)

    if len(sizes_arr) < 4:
        return 2, None, None

    x = np.arange(3, sizes_arr.max() + 1).astype(float)
    ccdf_vals = np.array([np.mean(sizes_arr >= s) for s in x])

    # drop trailing zeros
    zeros = np.where(ccdf_vals == 0)[0]
    if zeros.size > 0 and zeros[0] > 1:
        x         = x[:zeros[0]]
        ccdf_vals = ccdf_vals[:zeros[0]]

    if len(x) < 4:
        return 3, x, ccdf_vals

    kneedle = KneeLocator(
        x, ccdf_vals,
        curve="convex",
        direction="decreasing",
        S=1.0,
    )

    knee = kneedle.knee
    if knee is None:
        return max(2, int(x[0])), x, ccdf_vals

    return max(2, int(knee)), x, ccdf_vals


def process_window(start, window_size, B, coords_all_lat, coords_all_lon, SC_threshold):
    """Run PCA + spectral concentration on one time window and return the resulting spatial communities."""
    end = start + window_size
    B_window = B[:, start:end]

    Z, A = perform_svd(B_window)
    Pj = compute_power_spectrum(Z)
    SC, dom_idx, L, R, H = compute_spectral_concentration(Pj)
    thresholds, n_sel, frac_sel = build_sc_threshold_curve(SC)
    SC_thr_local = kneedle_threshold_from_curve(thresholds, n_sel)
    if SC_threshold is not None:
        relevant_pcs = np.where(SC > SC_threshold)[0]
    else:
        relevant_pcs = np.where(SC > SC_thr_local)[0]

    relevant_points = identify_relevant_points(A, relevant_pcs)

    communities = []
    all_components = []
        
    if len(relevant_points) >= 2:
        lat_res = np.min(np.diff(np.unique(coords_all_lat)))
        lon_res = np.min(np.diff(np.unique(coords_all_lon)))

        active_lookup = {
            (round(coords_all_lat[p], 6), round(coords_all_lon[p], 6)): local_idx
            for local_idx, p in enumerate(relevant_points)
        }

        G = nx.Graph()
        G.add_nodes_from(range(len(relevant_points)))

        for local_idx, p in enumerate(relevant_points):
            lat = coords_all_lat[p]
            lon = coords_all_lon[p]
            for dlat in [-lat_res, 0, lat_res]:
                for dlon in [-lon_res, 0, lon_res]:
                    if dlat == 0 and dlon == 0:
                        continue
                    neighbor_key = (round(lat + dlat, 6), round(lon + dlon, 6))
                    if neighbor_key in active_lookup:
                        j = active_lookup[neighbor_key]
                        G.add_edge(local_idx, j)

        all_components = list(nx.connected_components(G))
            

    communities = [
        [relevant_points[local_idx] for local_idx in c]
        for c in all_components
        if len(c) >= 2  
    ]
    return communities, SC_thr_local

def analyze_rainfall_data(
    B, ds, window_size, overlap,
    SC_threshold,
    plot_curves,
    plot_stride,
    plot_first_n,
    plot_path,
    min_community_size="auto",  # or a fixed number instead of Kneedle-based selection
    show=False,
):
    """
    Two-pass sliding-window heatmap analysis: pass 1 runs process_window over
    the whole time series to collect the community-size distribution and pick
    min_community_size (if "auto"); pass 2 rebuilds the same windows and
    accumulates a per-pixel recurrence heatmap counting how often each pixel
    was part of a community at least that large.
    """
    if isinstance(overlap, float) and overlap < 1.0:
        overlap = int(overlap * window_size)

    num_time_steps = B.shape[1]
    step = window_size - overlap
    time_values = ds.coords["time"].values
    time_labels = []

    coords_all = ds["rainfall"].stack(spatial=("lat", "lon")).spatial
    coords_all_lat = coords_all["lat"].values
    coords_all_lon = coords_all["lon"].values

    local_thres = []

    window_starts_list = list(range(0, num_time_steps - window_size + 1, step))
    plot_window_indices = set(random.sample(range(len(window_starts_list)), min(5, len(window_starts_list))))

    # First run to sample community distribution
    print("Pass 1: collecting community size distribution...")
    all_window_communities = []
    community_size_distribution = []
    plotted = 0

    for w_i, start in enumerate(range(0, min(num_time_steps - window_size + 1, len(time_values)), step)):
        communities, SC_thr_local = process_window(
            start, window_size, B,
            coords_all_lat, coords_all_lon,
            SC_threshold
        )
        all_window_communities.append(communities)
        time_labels.append(time_values[start])
        
        # DEBUG
        if w_i < 20:
            print(f"  w_i={w_i}: {len(communities)} communities, sizes={[len(c) for c in communities]}")
            print(f"  SC_thr_local={SC_thr_local:.4f}")

        for c in communities:
            community_size_distribution.append(len(c))

        if SC_threshold is None:
            local_thres.append(SC_thr_local)

        if plot_curves:
            do_plot = (
                (plotted < plot_first_n)
                or (plot_stride is not None and plot_stride > 0 and (w_i % plot_stride == 0))
            )
            if do_plot:
                end = start + window_size
                B_window = B[:, start:end]
                Z, A = perform_svd(B_window)
                Pj = compute_power_spectrum(Z)
                SC, _, _, _, _ = compute_spectral_concentration(Pj)
                thresholds, n_sel, _ = build_sc_threshold_curve(SC)

                plot_sc_curve(
                    thresholds=thresholds,
                    n_selected=n_sel,
                    t_knee=SC_thr_local,
                    window_label=f"start={time_values[start]}",
                    plot_path=plot_path,
                    show=show
                )
                plotted += 1

    
    
    # changepoint algorithm
    if min_community_size == "auto" and len(community_size_distribution) > 0:
        min_size_knee, sizes_sorted, ccdf_vals = compute_min_community_size_knee(
            community_size_distribution
        )
    else:
        min_size_knee = min_community_size
        sizes_sorted  = None
        ccdf_vals     = None

    plot_community_size_distribution(
        community_size_distribution = community_size_distribution,
        min_community_size          = min_size_knee,
        plot_path                   = plot_path,
        filename                    = "community_size_distribution_final",
        show                        = show,
        sizes_sorted                = sizes_sorted,
        ccdf_vals                   = ccdf_vals,
    )
    
    print(f"min_community_size: {min_size_knee}")

    # Second run
    print(f"Pass 2: building heatmap (min_community_size={min_size_knee})...")
    heatmap = np.zeros(B.shape[0])
 
    for communities in all_window_communities:
        for c in communities:
            if len(c) >= min_size_knee:
                for point in c:
                    heatmap[point] += 1
 
    if SC_threshold is None and len(local_thres) > 0:
        print(f"median local threshold: {np.median(local_thres):.3f}")
        print(f"max local threshold:    {max(local_thres):.3f}")
        print(f"min local threshold:    {min(local_thres):.3f}")
 
    r = ds["rainfall"]
    roi = (r.notnull().any("time") & (r.std("time") > 0)).stack(spatial=("lat", "lon")).to_numpy()
 
    coords = ds["rainfall"].stack(spatial=("lat", "lon")).spatial
    coords_lat = coords["lat"].values[roi]
    coords_lon = coords["lon"].values[roi]
 
    heatmap_values = np.where(heatmap == 0, np.nan, heatmap.astype(float))
    heatmap_values = heatmap_values[roi]
    heatmap_fraction = heatmap_values / len(time_labels) if len(time_labels) > 0 else heatmap_values
 
    return (
        time_labels,
        heatmap_values,
        heatmap_fraction,
        heatmap,
        coords_lat,
        coords_lon
    )
    

def clip_to_shapefile(ds, shapefile_path, all_touched=True, drop=True):
    """Clip a lat/lon-indexed Dataset to a shapefile geometry, reprojecting the shapefile to EPSG:4326 if needed."""
    gdf = gpd.read_file(shapefile_path)

    if gdf.crs is None:
        raise ValueError("no crs set")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # normalise dim names to what rioxarray expects internally
    dims = set(ds.dims)
    rename_map = {}
    if "longitude" in dims:
        rename_map["longitude"] = "x"
    elif "lon" in dims:
        rename_map["lon"] = "x"
    if "latitude" in dims:
        rename_map["latitude"] = "y"
    elif "lat" in dims:
        rename_map["lat"] = "y"

    if rename_map:
        ds = ds.rename(rename_map)

    ds = ds.rio.write_crs("EPSG:4326", inplace=False)

    if ds["y"].values[0] < ds["y"].values[-1]:
        ds = ds.sortby("y", ascending=False)

    clipped = ds.rio.clip(
        gdf.geometry.values,
        gdf.crs,
        all_touched=all_touched,
        drop=drop,
    )
    back_map = {v: k for k, v in rename_map.items()}
    if back_map:
        clipped = clipped.rename(back_map)

    return clipped



def export_heatmap(heatmap_values, coords_lat, coords_lon, output_path, crs="EPSG:4326", value_name="rec_frac"):
    """Save the per-pixel heatmap values (lat, lon, value) to CSV, dropping non-finite entries."""
    values = np.asarray(heatmap_values).ravel()
    lats   = np.asarray(coords_lat).ravel()
    lons   = np.asarray(coords_lon).ravel()

    valid  = np.isfinite(values)
    values, lats, lons = values[valid], lats[valid], lons[valid]

    csv_path = output_path.with_suffix(".csv")
    pd.DataFrame({
        "lat":      lats,
        "lon":      lons,
        value_name: values,
    }).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
        
        
        

def plot_heatmap(fraction, coords_lat, coords_lon, plot_path,
                 vmax=0.9,
                 Title="SVD Analysis",
                 figsize=None,
                 scatter=60,
                 show=False):
    """Scatter-plot the recurrence-fraction heatmap over the domain's lat/lon points."""

    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    values = np.asarray(fraction)
    values = np.where(values == 0, np.nan, values)

    # figsize proportional to lat/lon extent
    if figsize is None:
        lon_range  = np.nanmax(coords_lon) - np.nanmin(coords_lon)
        lat_range  = np.nanmax(coords_lat) - np.nanmin(coords_lat)
        inch_per_deg = 0.55
        figsize = (lon_range * inch_per_deg + 1.5, 
                   lat_range * inch_per_deg + 1.5)

    fig, ax = plt.subplots(figsize=figsize)

    if vmax is not None:
        sc = ax.scatter(coords_lon, coords_lat,
                        c=values,
                        cmap=cm.batlow,
                        s=scatter,
                        vmax=vmax,
                        vmin=0.1)
    else:
        sc = ax.scatter(coords_lon, coords_lat,
                        c=values,
                        cmap=cm.batlow,
                        s=scatter,
                        vmin=0.1)

    ax.set_aspect("equal", adjustable="box")

    plt.colorbar(sc, ax=ax, label="Fraction of recurrent PCs")

    ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(Title)

    plt.tight_layout()
    save_file = plot_path / "heatmap.png"
    plt.savefig(save_file, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()
    
    
    
    
################### CLUSTERING ###########################

# Q-STATISTIC  (Wang et al. 2016)
def q_statistic(
    z: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Compute q-statistic and p-value for a spatial partition.
    Wang et al. (2016), Equations 1, 7, 10, 11, 12.
    """
    z = np.asarray(z, dtype=float)
    N = len(z)
    strata = np.unique(labels)
    L = len(strata)

    y_bar = z.mean()
    SST   = float(np.sum((z - y_bar) ** 2))

    if SST == 0:
        return dict(q=0.0, F=0.0, lambda_=0.0, df1=L-1, df2=N-L,
                    p_value=1.0, significant=False,
                    SSW=0.0, SST=0.0, SSB=0.0)

    SSW = sum(
        float(np.sum((z[labels == h] - z[labels == h].mean()) ** 2))
        for h in strata
    )

    q   = float(np.clip(1.0 - SSW / SST, 0.0, 1.0))
    SSB = SST - SSW
    df1 = L - 1
    df2 = N - L

    if df1 <= 0 or df2 <= 0:
        return dict(q=q, F=np.nan, lambda_=np.nan, df1=df1, df2=df2,
                    p_value=np.nan, significant=False,
                    SSW=SSW, SST=SST, SSB=SSB)

    F_obs   = np.inf if q >= 1.0 else (df2 / df1) * (q / (1.0 - q))
    sigma2  = SSW / df2
    lambda_ = (SSB / (2.0 * sigma2)) if sigma2 > 0 else 0.0
    p_value = 0.0 if np.isinf(F_obs) else float(f_dist.sf(F_obs, df1, df2))

    return dict(q=q, F=float(F_obs), lambda_=float(lambda_),
                df1=df1, df2=df2, p_value=p_value,
                significant=p_value < alpha,
                SSW=SSW, SST=SST, SSB=SSB)



# AGGLOMERATIVE CLUSTERING  (spatial-constrained)
def raw_cluster_aggl(
    linkage,
    metric,
    n_clusters,
    valid,
    sp,
    X,
    min_cluster_size,
    coords,
    connectivity: str = "queen",
):
    """
    Spatially-constrained agglomerative clustering: builds a grid-adjacency
    connectivity matrix (queen or rook), runs AgglomerativeClustering (with
    explicit n_clusters, or auto via a distance-threshold knee if n_clusters
    is None), then merges any cluster smaller than min_cluster_size into its
    largest neighbouring cluster.

    Returns
    -------
    labels    : full-length label array (-1 outside valid)
    lab_v     : label array over valid points only
    k_found   : number of clusters found
    thr_cut   : distance threshold used (None if n_clusters was given explicitly)
    """
    labels = np.full(sp.size, -1, dtype=int)
    n      = X.shape[0]

    if n == 0:
        return labels, np.array([], dtype=int), 0, None

    index = {tuple(c): i for i, c in enumerate(coords)}

    if connectivity == "queen":
        deltas = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    else:
        deltas = [(-1,0),(1,0),(0,-1),(0,1)]

    rows, cols = [], []
    for i, (r, c) in enumerate(coords):
        for dr, dc in deltas:
            j = index.get((r + dr, c + dc))
            if j is not None:
                rows.append(i)
                cols.append(j)

    if len(rows) == 0:
        lab_v   = np.arange(n, dtype=int)
        thr_cut = None

    else:
        conn = coo_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(n, n)
        ).tocsr()

        eff_metric = "euclidean" if linkage == "ward" else metric

        if n_clusters is not None:
            lab_v = AgglomerativeClustering(
                n_clusters   = n_clusters,
                linkage      = linkage,
                metric       = eff_metric,
                connectivity = conn,
            ).fit_predict(X)
            thr_cut = None

        else:
            full = AgglomerativeClustering(
                n_clusters         = None,
                distance_threshold = 0.0,
                linkage            = linkage,
                metric             = eff_metric,
                connectivity       = conn,
                compute_distances  = True,
            ).fit(X)

            d = np.sort(full.distances_)

            if d.size == 0:
                lab_v   = np.zeros(n, dtype=int)
                thr_cut = 0.0
            else:
                lo  = int(0.33 * len(d))
                rel = np.diff(d) / (d[:-1] + 1e-12)
                rel[:max(lo - 1, 0)] = -np.inf
                j       = int(np.argmax(rel))
                thr_cut = 0.5 * (d[j] + d[j + 1])

                lab_v = AgglomerativeClustering(
                    n_clusters         = None,
                    distance_threshold = thr_cut,
                    linkage            = linkage,
                    metric             = eff_metric,
                    connectivity       = conn,
                ).fit_predict(X)

    sizes = np.bincount(lab_v)
    large = np.where(sizes >= min_cluster_size)[0]

    if len(large) > 0:
        small = np.where(sizes < min_cluster_size)[0]
        for s in small:
            idx = np.where(lab_v == s)[0]
            if len(idx) == 0:
                continue

            neigh = []
            for i in idx:
                r, c = coords[i]
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    j = index.get((r + dr, c + dc))
                    if j is not None and lab_v[j] in large:
                        neigh.append(lab_v[j])

            target = (
                int(np.bincount(neigh).argmax()) if len(neigh) > 0
                else int(large[np.argmax(sizes[large])])
            )
            lab_v[idx] = target

    labels[valid] = lab_v
    k_found = int(len(np.unique(lab_v)))

    return labels, lab_v, k_found, thr_cut



# PREPROCESS
def preprocess_ds(heatmap_full, ds_ref):
    """
    Prepare data for spatial clustering.

    Returns
    -------
    z_v        : (N,)   raw attribute values for valid pixels
    coords_geo : (N, 2) geographic coordinates [lat, lon]
    coords_idx : (N, 2) integer grid indices   [row, col]
    scaler_z   : fitted StandardScaler for z_v
    valid_idx  : (N,)   indices into the flat spatial array
    sp         : xarray spatial coordinate
    lats       : full lat array
    lons       : full lon array
    """
    min_valid = 2
    r = ds_ref["rainfall"]

    valid_count = r.notnull().sum("time")
    valid_mask  = valid_count >= min_valid

    roi = (
        valid_mask &
        r.notnull().any("time") &
        (r.std("time") > 0)
    ).stack(spatial=("lat", "lon")).to_numpy()

    sp   = r.stack(spatial=("lat", "lon"))["spatial"]
    lats = sp["lat"].values
    lons = sp["lon"].values

    z = np.asarray(heatmap_full, float).ravel()
    if z.size != sp.size:
        raise ValueError("heatmap length != spatial length")

    valid = roi
    if not np.any(valid):
        raise ValueError("No valid data points.")
    z_v        = np.where(np.isfinite(z[valid]), z[valid], 0.0)
    coords_geo = np.c_[lats[valid], lons[valid]]

    all_lats_unique = np.unique(lats)
    all_lons_unique = np.unique(lons)
    lat_to_row = {lat: i for i, lat in enumerate(all_lats_unique)}
    lon_to_col = {lon: i for i, lon in enumerate(all_lons_unique)}

    rows = np.array([lat_to_row[la] for la in lats[valid]], dtype=int)
    cols = np.array([lon_to_col[lo] for lo in lons[valid]], dtype=int)
    coords_idx = np.stack([rows, cols], axis=1)

    order      = np.lexsort((coords_geo[:, 1], coords_geo[:, 0]))
    z_v        = z_v[order].astype(np.float64)
    coords_geo = coords_geo[order].astype(np.float64)
    coords_idx = coords_idx[order]

    valid_idx  = np.where(valid)[0][order]
    scaler_z   = StandardScaler().fit(z_v[:, None])

    return z_v, coords_geo, coords_idx, scaler_z, valid_idx, sp, lats, lons


def find_knee(k_vals, metric_vals, S=1.0, curve="concave"):
    k_vals      = np.array(k_vals,      dtype=float)
    metric_vals = np.array(metric_vals, dtype=float)

    mask        = ~np.isnan(metric_vals)
    k_vals      = k_vals[mask]
    metric_vals = metric_vals[mask]

    if len(k_vals) < 3:
        return None

    direction = "increasing" if curve == "concave" else "decreasing"
    kneedle = KneeLocator(
        k_vals, metric_vals,
        curve     = curve,
        direction = direction,
        S         = S,
    )

    knee = kneedle.knee
    if knee is None:
        return None

    return int(knee)




def plot_k_analysis(df, S=1.0, save_path=None):
    """Plot q-statistic vs. requested k, marking significance and the detected knee point."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle("K Selection Analysis  (q-statistic)",
                 fontsize=14, fontweight="bold")

    valid   = df[df["k_found"].notna()]
    valid_q = valid["q"].values
    valid_k = valid["k_req"].values

    plateau_q = find_knee(valid_k, valid_q, S=S, curve="concave")

    #  q-statistic + knee 
    sig   = valid[valid["q_significant"]]
    insig = valid[~valid["q_significant"]]
    ax.plot(valid_k, valid_q, color="gray", linewidth=1, zorder=1)
    ax.scatter(sig["k_req"],   sig["q"],   color="#4CAF50", s=60,
               zorder=3, label="significant")
    ax.scatter(insig["k_req"], insig["q"], color="#F44336", s=60,
               marker="x", zorder=3, label="not significant")
    if plateau_q is not None:
        ax.axvline(plateau_q, color="#2196F3", linestyle="--", linewidth=1.5,
                   label=f"knee: k={plateau_q}")
    ax.set_xlabel("k requested")
    ax.set_ylabel("q-statistic")
    ax.set_title("q-statistic over k_req")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.set_facecolor("white")
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.grid(False, which="minor")

    fig.patch.set_facecolor("white")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig, {"plateau_q": plateau_q}



def summarize_k(df, S=0.2):
    """Print concise summary of k-analysis results."""
    valid    = df[df["k_found"].notna()]
    k_counts = valid["k_found"].value_counts().sort_values(ascending=False)

    print("\n" + "=" * 50)
    print("K SELECTION ANALYSIS SUMMARY  (q only)")
    print("=" * 50)
    print(f"  k_req range tested : {int(df['k_req'].min())} - {int(df['k_req'].max())}")
    print(f"  Successful runs    : {len(valid)} / {len(df)}")

    if valid.empty:
        print("\n  No successful runs.")
        print("=" * 50)
        return

    print(f"\n  k_found distribution:")
    for k, cnt in k_counts.items():
        bar = "█" * int(cnt / len(valid) * 30)
        print(f"    k={int(k):2d}  {bar}  {cnt}/{len(valid)}")

    plateau_q = find_knee(valid["k_req"], valid["q"], S=S) \
        if valid["q"].notna().any() else None

    print(f"\n  Knee point (Kneedle S={S}):")
    print(f"    q : k={plateau_q}")

    best_q_row = valid.loc[valid["q"].idxmax()]
    print(f"\n  Highest q : k_req={int(best_q_row['k_req'])}  "
          f"k_found={int(best_q_row['k_found'])}  "
          f"q={best_q_row['q']:.4f}  "
          f"sig={best_q_row['q_significant']}")

    print("=" * 50)
    

    

############### SMOOTHING #################################


def pad_nearest_valid(labels, mask):
    labels = np.asarray(labels)
    mask = np.asarray(mask).astype(bool)

    _, inds = ndimage.distance_transform_edt(~mask, return_indices=True)

    padded = labels.copy()
    padded[~mask] = labels[tuple(inds[:, ~mask])]
    return padded

def majority_filter_voronoi(labels, mask, size=3):
    """
    Majority filter with Voronoi extension outside the mask, so there's no
    "missing neighborhood" at the edges, giving more stable border labels.
    """
    padded = pad_nearest_valid(labels, mask).astype(int)

    def mode(values):
        v = values.astype(int)
        return np.bincount(v).argmax()

    out = ndimage.generic_filter(
        padded,
        mode,
        size=size,
        mode="nearest"
    ).astype(float)

    out[~mask] = np.nan
    return out



def split_connected_components(labels, connectivity=2):
    """
    Splits a labeled array into connected components based on 4 or 8 neigbourhood.
    This function takes a labeled array and identifies distinct connected components within it,
    assigning a unique identifier to each component.
    
    :param labels: A 2D array of labels where each unique label represents a different region.
    :type labels: numpy.ndarray
    :param connectivity: The connectivity criterion for defining connected components.
                        It can be 1 (4-connectivity) or 2 (8-connectivity) in a 2D array.
                        Default is 2.
    :type connectivity: int
    :return: A 2D array of the same shape as `labels`, where each connected component is assigned
             a unique identifier. Unlabeled regions are filled with NaN.
    :rtype: numpy.ndarray
    """
    structure = ndimage.generate_binary_structure(2, connectivity)
    out = np.full_like(labels, np.nan)

    next_id = 0
    for cid in np.unique(labels[np.isfinite(labels)]):
        mask = labels == cid
        labeled, n = ndimage.label(mask, structure=structure)

        for i in range(1, n + 1):
            out[labeled == i] = next_id
            next_id += 1

    return out


def merge_small_with_small(labels, min_size, connectivity=2):
    """
    Merge small connected clusters in a labeled image into larger clusters.
    This function identifies clusters with sizes smaller than min_size and merges
    connected small clusters together. Connected small clusters are merged into a
    single cluster with the label of the smallest cluster ID in the group.
    
    :param labels: A 2D numpy array of labeled clusters where each unique value
                   represents a distinct cluster. Can contain NaN values for invalid pixels.
    :param min_size: Minimum size threshold for clusters. Clusters with fewer pixels
                     than this value are considered small and subject to merging.
    :param connectivity: Connectivity structure for determining neighboring clusters.
                        2 (default) uses 8-connectivity, 1 uses 4-connectivity.
    :return: A 2D numpy array with the same shape as labels, where small connected
             clusters have been merged into larger clusters with the minimum label ID.
    """
    structure = ndimage.generate_binary_structure(2, connectivity)
    out = labels.copy()

    #count cluster size
    ids, counts = np.unique(out[np.isfinite(out)], return_counts=True)
    sizes = dict(zip(ids.astype(int), counts))
    small_ids = {cid for cid, s in sizes.items() if s < min_size}  # clusters below min_size

    visited = set()  # clusters already processed

    for cid in small_ids:
        if cid in visited:
            continue

        stack = [cid]  # small clusters still to check
        group = set()  # connected group of small clusters found so far

        while stack:
            cur = stack.pop()
            if cur in group:
                continue
            group.add(cur)

            mask = out == cur
            dil = ndimage.binary_dilation(mask, structure=structure)
            neigh = np.unique(out[dil & ~mask & np.isfinite(out)]).astype(int)

            for n in neigh:
                if n in small_ids and n not in group:
                    stack.append(n)

        visited |= group

        if len(group) <= 1:
            continue

        target = min(group)  # all small clusters in the group get the same label
        for g in group:
            if g != target:
                out[out == g] = target

    return out

def merge_remaining_small(labels, min_size, connectivity=2):
    """
    Merge small connected clusters in a labeled image to larger clusters.
    This function identifies clusters with sizes smaller than min_size and merges
    them to a neighbouring larger cluster.
    
    :param labels: A 2D numpy array of labeled clusters where each unique value
                   represents a distinct cluster. Can contain NaN values for invalid pixels.
    :param min_size: Minimum size threshold for clusters. Clusters with fewer pixels
                     than this value are considered small and subject to merging.
    :param connectivity: Connectivity structure for determining neighboring clusters.
                        2 (default) uses 8-connectivity, 1 uses 4-connectivity.
    :return: A 2D numpy array with the same shape as labels, where small connected
             clusters have been merged into larger clusters with the minimum label ID.
    """    
    structure = ndimage.generate_binary_structure(2, connectivity)
    out = labels.copy()

    def get_sizes(arr):
        ids, counts = np.unique(arr[np.isfinite(arr)], return_counts=True)
        return dict(zip(ids.astype(int), counts))

    sizes = get_sizes(out)

    for cid in list(sizes.keys()):
        if sizes.get(cid, 0) >= min_size:
            continue

        mask = out == cid
        if not np.any(mask):  # already merged away
            continue

        dil = ndimage.binary_dilation(mask, structure=structure)
        neigh = np.unique(out[dil & ~mask & np.isfinite(out)]).astype(int)

        candidates = [n for n in neigh if sizes.get(n, 0) >= min_size]
        if not candidates:
            continue

        target = max(candidates, key=lambda x: sizes[x])
        out[mask] = target

        # update sizes
        sizes[target] = sizes.get(target, 0) + sizes.pop(cid, 0)

    return out


