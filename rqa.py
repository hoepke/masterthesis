"""
rqa.py

Recurrence Quantification Analysis (RQA) on rainfall time series.

1. Recurrence plots      - extract_X, multivariate_recurrence_plot
2. RQA metrics           - compute_DET, compute_LAM, compute_ENTR,
                            compute_L_mean (determinism, laminarity,
                            entropy, mean diagonal line length)
3. Theiler window        - _compute_theiler, plot_theiler_parallel
                            (removes short-range autocorrelation from
                            the recurrence plot)
4. Rolling-window RQA    - rolling_window_rqa, iaaft_surrogate_vectorized,
                            compute_surrogate_metrics
                            (metrics over time + IAAFT surrogate
                            significance testing)
5. Trend export helper   - plot_metric_rolling
                            (Mann-Kendall trend tests + CSV export, no plotting)
"""




import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import pymannkendall as mk
from pathlib import Path
from cmcrameri import cm
import tempfile, os
from joblib import Parallel, delayed

from kneed import KneeLocator


def apply_grid(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)


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

CLUSTER_COLORS = [
'#9437FF','#FF1150','#FFA300',"#009193", "#011993","#7A81FF" # '#dbaf4d'
]

cluster_cmap = ListedColormap(CLUSTER_COLORS)

mpl.rcParams['text.usetex'] = False


##### HELPER #####

def extract_X(ds, mask):
    mask = mask.squeeze().drop_vars("time", errors="ignore")
    
    # Ensure mask is computed (not dask array)
    if hasattr(mask.data, 'compute'):
        mask = mask.compute()
    
    ds_filtered = ds.where(mask, drop=True)
    rain_multi = (
        ds_filtered.rainfall
        .stack(features=("lat", "lon"))
        .dropna(dim="features", how="any")
    )
    X = np.nan_to_num(rain_multi.transpose("features", "time").values.T)
    time_values = ds.time.values
    return X, time_values


def plot_diag_histogram_grid(
    all_diag_hists,
    window_times,
    window_labels,
    entr_series,
    cluster_name,
    plot_path,
    every_n = 10,
    l_min   = 2,
):
    """
    Plottet alle n-ten Diagonallinien-Histogramme in einem Grid.
    """
    # Indizes der zu plottenden Fenster
    indices = [i for i in range(len(all_diag_hists)) if i % every_n == 0]
    n       = len(indices)

    if n == 0:
        return

    ncols = min(6, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.5 * ncols, 3.5 * nrows),
        sharey=False,
    )
    axes = np.array(axes).flatten()

    for plot_idx, win_idx in enumerate(indices):
        ax        = axes[plot_idx]
        diag_runs = np.array(all_diag_hists[win_idx], dtype=int)
        diag_runs = diag_runs[diag_runs >= l_min]
        label     = window_labels[win_idx]
        entr_val  = entr_series[win_idx]

        if len(diag_runs) == 0:
            ax.set_title(f"{label}\n(no data)", fontsize=7)
            ax.axis("off")
            continue

        lengths, counts = np.unique(diag_runs, return_counts=True)
        p = counts / counts.sum()

        ax.bar(lengths, p,
               color="#185FA5", alpha=0.75,
               width=0.8, edgecolor="white", linewidth=0.3)

        entr_str = f"{entr_val:.3f}" if np.isfinite(entr_val) else "nan"
        ax.set_title(f"{label}\nENT={entr_str}", fontsize=7)
        ax.set_xlabel("l", fontsize=7)
        ax.set_ylabel("p(l)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3, lw=0.4, axis="y")
        ax.spines[["top", "right"]].set_visible(False)

    # leere Subplots ausblenden
    for k in range(plot_idx + 1, len(axes)):
        axes[k].axis("off")

    fig.suptitle(
        f"{cluster_name} – p(l) every {every_n} windows",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out = Path(plot_path) / f"{cluster_name.replace(' ', '_')}_diag_hist_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


######## RQA METRICS
def count_runs_np(arr, min_length=2):
    """
    Count the lengths of consecutive runs of 1s in a binary array.
    Parameters:
    arr (array-like): Input binary array (0s and 1s).
    min_length (int): Minimum length of runs to be counted (default is 2).
    Returns:
    numpy.ndarray: Lengths of runs that are greater than or equal to min_length.
    """
    
    arr = np.asarray(arr).astype(int).ravel()
    padded = np.concatenate(([0], arr, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts
    return lengths[lengths >= min_length]

# def compute_DET(R, l_min=2):
#     """
#     Calculate the Deterministic Entropy (DET) from a recurrence matrix.
#     Parameters:
#     R (array-like or xr.DataArray): Recurrence matrix.
#     l_min (int, optional): Minimum length of diagonal runs to consider. Default is 2.
#     Returns:
#     float: The DET value.
#     """
    
#     R = R.values if isinstance(R, xr.DataArray) else R
#     total_recurrences = np.sum(R)
#     det_points = sum(
#         np.sum(count_runs_np(np.diag(R, k), l_min))
#         for k in range(-R.shape[0] + 1, R.shape[0])
#         if k != 0
#     )
#     return det_points / (total_recurrences + 1e-8)

# def compute_LAM(R, v_min=2):
#     """
#     Calculate the average length of diagonal lines in a recurrence plot.
#     Parameters:
#     R (array-like): Recurrence matrix or DataArray.
#     v_min (int, optional): Minimum length of diagonal lines to consider. Default is 2.
#     Returns:
#     float: Average length of diagonal lines normalized by total recurrences.
#     """
    
#     R = R.values if isinstance(R, xr.DataArray) else R
#     total_recurrences = np.sum(R)
#     lam_points = sum(
#         np.sum(count_runs_np(R[:, i], v_min))
#         for i in range(R.shape[1])
#     )
#     return lam_points / (total_recurrences + 1e-8)

def compute_DET(R, l_min=2, return_histogram=False):
    R = R.values if isinstance(R, xr.DataArray) else R
    
    diag_runs = []
    for k in range(-R.shape[0] + 1, R.shape[0]):
        if k == 0:
            continue
        runs = count_runs_np(np.diag(R, k), 1)  # alle Runs >= 1
        diag_runs.extend(runs.tolist())
    
    diag_runs       = np.array(diag_runs, dtype=int)
    all_diag_points = diag_runs.sum()
    det_points      = diag_runs[diag_runs >= l_min].sum()
    det             = det_points / (all_diag_points + 1e-8)
    
    if return_histogram:
        return det, diag_runs
    return det


def compute_LAM(R, v_min=2, return_histogram=False):
    R = R.values if isinstance(R, xr.DataArray) else R
    
    vert_runs = []
    for i in range(R.shape[1]):
        runs = count_runs_np(R[:, i], 1)  # alle Runs >= 1
        vert_runs.extend(runs.tolist())
    
    vert_runs       = np.array(vert_runs, dtype=int)
    all_vert_points = vert_runs.sum()
    lam_points      = vert_runs[vert_runs >= v_min].sum()
    lam             = lam_points / (all_vert_points + 1e-8)
    
    if return_histogram:
        return lam, vert_runs
    return lam

def compute_DIV(R):
    """
    Compute the diversity index from a given matrix R.
    Parameters:
    R (xr.DataArray or np.ndarray): Input matrix for which the diversity index is calculated.
    Returns:
    float: The computed diversity index.
    """
    
    R = R.values if isinstance(R, xr.DataArray) else R
    max_diag_len = max(
        (np.max(count_runs_np(np.diag(R, k))) if len(count_runs_np(np.diag(R, k))) > 0 else 0)
        for k in range(-R.shape[0] + 1, R.shape[0])
        if k != 0 
    )
    return 1 / (max_diag_len + 1e-8)

def compute_ENTR(R, l_min=2, normalize=True):
    """
    Compute Shannon entropy of diagonal line length distribution.
    Higher ENTR = more complex/unpredictable dynamics.
    If normalize=True, divides by log(l_max) to yield values in [0, 1].
    """
    R = R.values if isinstance(R, xr.DataArray) else R
    
    #collecting all diagonals
    diag_runs = []
    for k in range(-R.shape[0] + 1, R.shape[0]):
        if k == 0:
            continue
        runs = count_runs_np(np.diag(R, k), l_min)
        diag_runs.extend(runs.tolist())
    
    #compute distribution
    diag_runs = np.array(diag_runs, dtype=int)
    if len(diag_runs) == 0:
        return 0.0
    
    lengths, counts = np.unique(diag_runs, return_counts=True)
    p = counts / counts.sum()
    entr = -np.sum(p * np.log(p + 1e-8)) #shannon entropie
    
    if normalize:
        l_max = R.shape[0] - 1 # max diagonal length
        entr_max = np.log(l_max) if l_max > 1 else 1.0
        entr = entr / entr_max
    
    return entr

def compute_L_mean(R, l_min=2):
    """
    Compute mean diagonal line length.
    Higher L_mean = slower divergence = lower effective Lyapunov exponent.
    Approximation: lambda ≈ 1 / L_mean
    
    Parameters:
    R (xr.DataArray or np.ndarray): Recurrence matrix
    l_min (int): Minimum line length to consider
    
    Returns:
    float: Mean diagonal line length (lines >= l_min only)
    """
    R = R.values if isinstance(R, xr.DataArray) else R
    
    diag_runs = []
    for k in range(-R.shape[0] + 1, R.shape[0]):
        if k == 0:
            continue
        runs = count_runs_np(np.diag(R, k), l_min)
        diag_runs.extend(runs.tolist())
    
    diag_runs = np.array(diag_runs, dtype=int)
    
    if len(diag_runs) == 0:
        return 0.0
    
    return float(np.mean(diag_runs))

def multivariate_recurrence_plot(x: xr.DataArray, eps=None, q=0.1, use_cosine=True, t_w_dict=None, plot=False, rp_path=None, title="Recurrence Plot"):
    """
    Generate a multivariate recurrence plot.
    Parameters:
    x (xr.DataArray): Input data array with a 'time' coordinate.
    eps (float, optional): Threshold for recurrence. If None, calculated from quantile.
    q (float, optional): Quantile to determine eps. Default is 0.1.
    use_cosine (bool, optional): If True, use cosine distance; otherwise, use Euclidean distance.
    Returns:
    tuple: A tuple containing:
        - rec (xr.DataArray): Recurrence matrix.
        - eps (float): Threshold used for recurrence.
    """

    time = x.coords['time'].values
    X = x.values
    N = len(time)

    if use_cosine:
        X = np.nan_to_num(X).astype(np.float32)
        X = X + 1e-6 #constant offset to treat null vector
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_norm = X / norms
        dist = 1 - np.dot(X_norm, X_norm.T)
        np.clip(dist, 0.0, 2.0, out=dist)
    else:
        X = X.astype(np.float32)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)

    tw = t_w_dict if t_w_dict is not None else 0

    mask_tw = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask_tw, False)
    if tw > 0:
        for k in range(1, tw + 1):
            np.fill_diagonal(mask_tw[k:], False)
            np.fill_diagonal(mask_tw[:, k:], False)

    if eps is None:
        valid_dists = dist[mask_tw]
        eps = np.quantile(valid_dists, q)

    R = (dist <= eps) & mask_tw

    rec = xr.DataArray(R, dims=["time", "t2"], coords={"time": time, "t2": time})

    if plot:
        n_ticks = 6
        tick_idx = np.linspace(0, N - 1, n_ticks, dtype=int)
        tick_labels = [pd.Timestamp(time[i]).strftime("%Y-%m") for i in tick_idx]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(R, origin="lower", aspect="auto", cmap="binary")
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=14)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels(tick_labels,fontsize=14)
        ax.set_title(f"{title} (eps={eps:.4f})", fontsize=18)
        plt.tight_layout()
        plt.savefig(rp_path)
        plt.close()

    return rec, eps

######### EMBEDDING #############

# @dataclass
# class EmbedConfig:
#     cluster_id: int
#     tau_local:  int
#     M_local:    int # possibility to save another tau
#     tau_used:   int
#     M_used:     int 
#     embed_mode: str


##### THEILER WINDOW ##############

def _compute_theiler(X, name, use_cosine, max_delta, percentiles):

    N = X.shape[0]

    if use_cosine:
        X = X + 1e-6
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X = X / norms

    delta_bins = np.arange(1, min(max_delta, N) + 1)
    contours = {p: [] for p in percentiles}

    for dt in delta_bins:
        if dt < N:
            left  = X[:-dt]
            right = X[dt:]
            if use_cosine:
                d_at_dt = 1.0 - np.einsum("ij,ij->i", left, right)
                np.clip(d_at_dt, 0.0, 2.0, out=d_at_dt)
                d_at_dt = d_at_dt[d_at_dt > 1e-5]   # only treat days with a distance value < 0
            else:
                d_at_dt = np.linalg.norm(left - right, axis=1)

            for p in percentiles:
                contours[p].append(
                    np.percentile(d_at_dt, p) if d_at_dt.size > 0 else np.nan
                )

    # Median curve normalisieren
    curve = np.array(contours[90]) #90th percentil as it is more robust than median
    curve_norm = (curve - np.nanmin(curve)) / (np.nanmax(curve) - np.nanmin(curve) + 1e-10)

    valid = ~np.isnan(curve_norm)
    T_w_knee = None
    if valid.sum() >= 3:
        x = delta_bins[valid].astype(float)
        y = curve_norm[valid]

        max_knee_delta = 30
        mask_short = x <= max_knee_delta
        x_short = x[mask_short]
        y_short = y[mask_short]

        if mask_short.sum() >= 3:
            kneedle = KneeLocator(
                x_short, y_short,
                curve="concave",
                direction="increasing",
                S=1,
            )
            T_w_knee = int(kneedle.knee) if kneedle.knee is not None else int(x_short[-1])

    T_w = T_w_knee if T_w_knee is not None else int(delta_bins[-1])
    print(f"{name}: T_w_knee={T_w_knee}, T_w={T_w}")
    
    return T_w, contours, delta_bins



def plot_theiler_parallel(
    all_cluster_masks,
    ds,
    plot_path,
    cluster_names=None,
    use_cosine=True,
    max_delta=30,
    percentiles=[1, 10, 50, 90, 99],
    title="Space-Time Separation Plot",
    n_jobs=1,
    show=False,
    embeddings=None, 
    plot = True,
):
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    if embeddings is not None:
        items = [
            (f"Community {cid}", emb["X"])           
            for cid, emb in embeddings.items()
        ]
    else:
        items = [
            (cluster_names[i] if cluster_names else f"Community {i+1}", extract_X(ds, mask)[0])
            for i, mask in enumerate(all_cluster_masks)
        ]

    # Parallel 
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_compute_theiler)(
            name=name, X=X, use_cosine=use_cosine,
            max_delta=max_delta, percentiles=percentiles,
        )
        for name, X in items
    )
    
    t_w_dict = {}
    for (name, X), (T_w, contours, delta_bins) in zip(items, results):
        t_w_dict[name] = T_w

    # Plot
    if plot:
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, (name, X), (T_w, contours, delta_bins) in zip(axes, items, results):
            percentiles_sorted = sorted(percentiles)
            color_positions = np.linspace(0.15, 0.85, len(percentiles_sorted))
            colors = cm.roma(color_positions)
            for p, color in zip(percentiles_sorted, colors):
                ax.plot(delta_bins, contours[p],
                        label=f"p={p}%", color=color, linewidth=1.8)
            ax.axvline(x=T_w, color="0.35", linestyle="-", linewidth=1,
                    label=f"T_w = {T_w}")
            ax.set_title(name)
            ax.set_xlabel(r"$\Delta$t (days)")
            ax.set_facecolor("white")
            ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
            ax.grid(False, which="minor")
            if ax == axes[0]:
                ax.set_ylabel("Cosine distance" if use_cosine else "Euclidean distance")
            ax.legend(fontsize=7)

        fig.suptitle(title)
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        plt.savefig(plot_path / "theiler_window.svg", format="svg", bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    return t_w_dict



########## ROLLING RQA #############

#cross-correlation
def iaaft_surrogate_vectorized(window: np.ndarray, n_iter: int = 20, rng=None) -> np.ndarray:
    """
    Multivariate IAAFT surrogate (nach Prichard & Theiler, 1994),
    die Kreuzkorrelationen zwischen Features erhält, indem die
    Phasenrandomisierung gemeinsam über alle Kanäle läuft, während
    jeder Kanal sein eigenes Amplitudenspektrum und seine eigene
    Randverteilung (Amplitude) behält.

    Parameters
    ----------
    window : np.ndarray, shape (n_features, n_time)
    n_iter : int
    rng : np.random.Generator, optional

    Returns
    -------
    surrogate : np.ndarray, shape (n_features, n_time)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_features, n_time = window.shape

    x_sorted = np.sort(window, axis=1)
    orig_fft = np.fft.rfft(window, axis=1)
    target_amplitudes = np.abs(orig_fft)
    orig_phases = np.angle(orig_fft)  # (n_features, n_freq)

    # feste Phasen-OFFSETS relativ zu Kanal 0, einmalig aus den
    # Originaldaten geschätzt -> kodiert die Kreuz-Phasenbeziehung,
    # die über alle Iterationen erhalten bleiben soll
    phase_offsets = orig_phases - orig_phases[0:1, :]  # (n_features, n_freq)
    n_freq = orig_fft.shape[1]

    # gemeinsame zufällige Startphase, geteilt über alle Kanäle
    common_phase = rng.uniform(-np.pi, np.pi, size=n_freq)

    s_fft = target_amplitudes * np.exp(1j * (common_phase[None, :] + phase_offsets))
    s = np.fft.irfft(s_fft, n=n_time, axis=1)

    rank = np.argsort(np.argsort(s, axis=1), axis=1)
    s = np.take_along_axis(x_sorted, rank, axis=1)

    for _ in range(n_iter):
        s_fft_now = np.fft.rfft(s, axis=1)
        # gemeinsame Phase wird aus Kanal 0 als Referenz "getrieben",
        # die festen relativen Offsets zu den anderen Kanälen bleiben
        common_phase = np.angle(s_fft_now[0, :])

        s_fft = target_amplitudes * np.exp(1j * (common_phase[None, :] + phase_offsets))
        s = np.fft.irfft(s_fft, n=n_time, axis=1)

        rank = np.argsort(np.argsort(s, axis=1), axis=1)
        s = np.take_along_axis(x_sorted, rank, axis=1)

    return s


def compute_surrogate_metrics(window, q, use_cosine, t_w_window, time_values, start, end, n_iter=20):
    """Berechnet RQA-Metriken für ein einzelnes Surrogate — für Parallelisierung."""
    window_surr = iaaft_surrogate_vectorized(window, n_iter=n_iter)
    window_surr_da = xr.DataArray(
        window_surr.T,
        dims=["time", "features"],
        coords={"time": time_values[start:end]},
    )
    rec_surr, _ = multivariate_recurrence_plot(
        window_surr_da,
        eps=None,
        q=q,
        use_cosine=use_cosine,
        t_w_dict=t_w_window,
        plot=False,
    )
    return {
        "DET":   compute_DET(rec_surr),
        "LAM":   compute_LAM(rec_surr),
        "ENTR":  compute_ENTR(rec_surr),
        #"DIV":   compute_DIV(rec_surr),
        "LMEAN": compute_L_mean(rec_surr),
    }


def rolling_window_rqa(
    X,
    time_values,
    # embed_cfg,
    t_w             = 0,
    eps             = None,
    q               = 0.1,
    use_cosine      = False,
    window_size     = 365,
    step            = 30,
    alpha           = 0.05,
    n_jobs          = 2,
    cluster_name    = None,
    median_w        = "365D",
    rp_path         = None,
    rp_select       = None,
    verbose         = False,
):
    data     = X.T
    time_dim = data.shape[1]

    # write arrays to temp files for mmap in workers 
    fd, tmp_path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(tmp_path, data)
    del data

    # window indices
    windows = [
        (start, start + window_size)
        for start in range(0, time_dim - window_size + 1, step)
    ]
    window_starts    = np.array([s for s, _ in windows])
    window_midpoints = window_starts + window_size // 2
    window_times     = pd.to_datetime(time_values[window_midpoints])

    # fallback Theiler when no center_series (raw cluster mode) 
    theiler_fallback = int(t_w)

    # per-window worker 
    def process_window(start, end, win_idx):
        try:
            #load data
            data_mm = np.load(tmp_path, mmap_mode="r")
            window  = np.array(data_mm[:, start:end])
            del data_mm

            t_start      = pd.Timestamp(time_values[start]).strftime("%Y-%m")
            t_end        = pd.Timestamp(time_values[end - 1]).strftime("%Y-%m")
            window_label = f"{t_start} – {t_end}"

            # plot RP 
            do_plot = False
            if rp_select is not None:
                if isinstance(rp_select, int):
                    do_plot = (win_idx % rp_select == 0)
                elif isinstance(rp_select, list):
                    do_plot = win_idx in rp_select

            window_rp_path = (
                rp_path.with_stem(f"{rp_path.stem}_w{win_idx}")
                if do_plot and rp_path is not None else None
            )

            X_embedded = window.T
            t_w_window = theiler_fallback

            if verbose:
                print(
                    f"  [{cluster_name}] window {win_idx:03d} | {window_label} "
                    f"→ T_w={t_w_window}"
                )

            window_da = xr.DataArray(
                window.T,
                dims=["time", "features"],
                coords={"time": time_values[start:end]},
            )

            # Recurrence Plot 
            rec, used_eps = multivariate_recurrence_plot(
                window_da,
                eps=eps,
                q=q,
                use_cosine=use_cosine,
                t_w_dict=t_w_window,
                plot=do_plot,
                rp_path=window_rp_path,
                title=f"{cluster_name} | {window_label}",
            )
            
            #DEBUG
            rr_orig = rec.values.sum() / rec.values.size
            

            det, diag_hist = compute_DET(rec, return_histogram=True)  
            lam, vert_hist = compute_LAM(rec, return_histogram=True)
            #div = compute_DIV(rec)
            entr           = compute_ENTR(rec)
            lmean          = compute_L_mean(rec)
            ratio_DL       = det / (lam + 1e-8)

            if win_idx == 0:
                print(f"  [{cluster_name}] rec shape:         {rec.shape}")
                print(f"  [{cluster_name}] rec dtype:         {rec.dtype}")
                print(f"  [{cluster_name}] rec sum:           {rec.sum()}")
                print(f"  [{cluster_name}] rec unique values: {np.unique(rec)}")
                print(f"  [{cluster_name}] DET={det:.4f}, LAM={lam:.4f}, ENTR = {entr:.4f}, LMEAN={lmean:.4f}, RATIO_DL = {ratio_DL:.4f}") #
                
            # ── Red Noise Surrogate Test ──────────────────────────────────
            print(f"[{cluster_name}] window {win_idx}: starting surrogates")
            N_surr = 99

            results = Parallel(n_jobs=-1, backend="loky")(
                delayed(compute_surrogate_metrics)(
                    window, q, use_cosine, t_w_window, time_values, start, end, n_iter=20
                )
                for _ in range(N_surr)
            )

            surr_metrics = {k: [r[k] for r in results] for k in results[0]}
            
            # Rank order test
            rank_sig = {}
            rank_p   = {}
            obs_vals = {"DET": det, "LAM": lam, "ENTR": entr, "LMEAN": lmean}
            for m in ["DET", "LAM", "ENTR", "LMEAN"]:
                obs   = obs_vals[m]
                surrs = np.array(surr_metrics[m])
                p     = (np.sum(surrs >= obs) + 1) / (len(surrs) + 1)
                sig   = p <= 0.05
                rank_sig[m] = sig
                rank_p[m]   = p
                print(f"  [{cluster_name}] window {win_idx} | {m}: "
                      f"obs={obs:.4f}, p={p:.3f}, sig={sig}") 



            surr_low  = {m: np.percentile(v, 2.5)  for m, v in surr_metrics.items()}
            surr_high = {m: np.percentile(v, 97.5) for m, v in surr_metrics.items()}

            return (
                det, lam, entr, lmean,
                window_label,
                diag_hist,
                vert_hist,
                surr_low,
                surr_high,
                rank_sig
            )

        except Exception as e:
            import traceback
            print(f"[{cluster_name}] window {win_idx} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            return (np.nan, np.nan, np.nan, np.nan, np.nan,
                    f"window_{win_idx}",
                    np.array([]), np.array([]),
                    {m: np.nan for m in ["DET", "LAM", "ENTR", "LMEAN"]},
                    {m: np.nan for m in ["DET", "LAM", "ENTR", "LMEAN"]},
                    {m: False  for m in ["DET", "LAM", "ENTR", "LMEAN"]},
                    )

    # parallelism 
    safe_jobs = max(1, min(int(os.cpu_count() if n_jobs == -1 else n_jobs), 2))

    print(f"\n{cluster_name}: {len(windows)} windows, "
          f"n_jobs={safe_jobs} [tau={t_w}]")

    try:
        try:
            results = Parallel(
                n_jobs=safe_jobs,
                batch_size=1,
                max_nbytes="50M",
                backend="loky",
            )(
                delayed(process_window)(start, end, win_idx)
                for win_idx, (start, end) in enumerate(windows)
            )
        except Exception as exc:
            exc_text = f"{type(exc).__name__}: {exc}"
            if "TerminatedWorkerError" in exc_text or "SIGKILL" in exc_text:
                print(f"{cluster_name}: worker failure – retrying serial...")
                results = [
                    process_window(start, end, win_idx)
                    for win_idx, (start, end) in enumerate(windows)
                ]
            else:
                raise
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        # 
    # assemble results 
    det_series, lam_series, entr_series, lmean_series, \
    window_labels, all_diag_hists, all_vert_hists, \
    all_surr_low, all_surr_high, all_rank_sig = map(list, zip(*results))
    
    #det_p_series, lam_p_series, div_p_series, \
    if rp_path is not None:
        plot_diag_histogram_grid(
            all_diag_hists = all_diag_hists,
            window_times   = window_times,
            window_labels  = window_labels,
            entr_series    = entr_series,
            cluster_name   = cluster_name,
            plot_path      = rp_path.parent,
            every_n        = 10,
            l_min          = 2,
        )
        
    det_series    = np.asarray(det_series)
    lam_series    = np.asarray(lam_series)
    entr_series   = np.asarray(entr_series)
    lmean_series  = np.asarray(lmean_series)
    window_labels = np.asarray(window_labels)


    # check again?
    sentinel = 1e8
    print(f"Zero values before NaN replacement:")
    print(f"  DET  : {(det_series == 0).sum()}")
    print(f"  LAM  : {(lam_series == 0).sum()}")
    print(f"  ENTR : {(entr_series == 0).sum()}")
    print(f"  L_mean: {(lmean_series == 0).sum()}")

    
    det_series[det_series == 0]        = np.nan
    lam_series[lam_series == 0]        = np.nan
    entr_series[entr_series == 0]      = np.nan
    lmean_series[lmean_series == 0]    = np.nan


    df = (
        pd.DataFrame({
            "time":  window_times,
            "DET":   det_series,
            "LAM":   lam_series,
            "ENTR":   entr_series,
            "LMEAN":  lmean_series,

        })
        .dropna(subset=["DET", "LAM", "ENTR","LMEAN"])
        .sort_values("time")
        .set_index("time")
    )

    for metric in ["DET", "LAM", "ENTR", "LMEAN"]:
        df[f"{metric}_med"]  = (
            df[metric].rolling(median_w, center=True, min_periods=1).median()
        )
        df[f"{metric}_low"]  = (
            df[metric].rolling(median_w, center=True, min_periods=1)
            .quantile(alpha / 2)
        )
        df[f"{metric}_high"] = (
            df[metric].rolling(median_w, center=True, min_periods=1)
            .quantile(1 - alpha / 2)
        )

    # Surrogate-Join einmal nach der Schleife – gleiche Einrückungsebene wie for
    surr_low_dict  = {m: [d[m] for d in all_surr_low]  for m in ["DET", "LAM", "ENTR", "LMEAN"]}
    surr_high_dict = {m: [d[m] for d in all_surr_high] for m in ["DET", "LAM", "ENTR", "LMEAN"]}

    df_surr = pd.DataFrame({
        **{f"{m}_surr_low":  surr_low_dict[m]  for m in surr_low_dict},
        **{f"{m}_surr_high": surr_high_dict[m] for m in surr_high_dict},
    })

    # direkt per Position zuweisen, kein Index-Join
    for col in df_surr.columns:
        df[col] = df_surr[col].values
    
    for m in ["DET", "LAM", "ENTR", "LMEAN"]:
        df[f"{m}_rank_sig"] = [r[m] for r in all_rank_sig]
    
    return {
        "window_time":   window_times,
        "window_labels": window_labels,
        "DET":           det_series,
        "LAM":           lam_series,
        "ENTR":          entr_series,
        "LMEAN":         lmean_series,
        "all_diag_hists": all_diag_hists,
        "rolling_anual": df.reset_index(),
        "t_w":           t_w,
    }


def plot_metric_rolling(
    df,
    metric,
    cluster_id,
    mark_dates      = None,
    save_csv        = False,
    out_dir         = None,
    csv_name        = "default",
) -> dict:
    """
    Compute Mann-Kendall trend tests for one rolling RQA metric of one
    community, and optionally export the series plus MK results to CSV.

    Mann-Kendall trend tests are computed for the full series and for any
    period given in mark_dates. Results are returned as a dict and,
    if save_csv is True, also written to a per cluster per metric CSV
    file, which includes the surrogate band columns and the rank order
    column when present in df.

    Parameters
    ----------
    df : pd.DataFrame
        Rolling window results for one community. Must contain "time",
        the metric column, and "{metric}_med", "{metric}_low",
        "{metric}_high". May optionally contain "{metric}_surr_low",
        "{metric}_surr_high", "{metric}_p", and "{metric}_rank_sig".
    metric : str
        Name of the metric column.
    cluster_id : int
        Community index, used in the output CSV filename.
    mark_dates : list of (int, int), optional
        Year pairs marking period boundaries, used to define separate
        Mann Kendall test periods.
    save_csv : bool
        Whether to export the series and MK results to CSV.
    out_dir : str or Path, optional
        Directory the CSV is saved to, required if save_csv is True.
    csv_name : str
        Suffix used in the CSV filename.

    Returns
    -------
    dict
        mk_results with keys "cluster_id", "metric", "median", and
        "periods" (a list of Mann Kendall result dicts, one per period).
    """

    surr_low_col  = f"{metric}_surr_low"
    surr_high_col = f"{metric}_surr_high"
    rank_col      = f"{metric}_rank_sig"

    t = df["time"]

    # ── MK tests ──────────────────────────────────────────────────────────
    mean_full = df[metric].mean()
    std_full  = df[metric].std()

    def _safe_mk(series):
        s = pd.Series(series).dropna()
        if len(s) < 2:
            return None
        return mk.hamed_rao_modification_test(s)

    def _mk_row(test, period):
        if test is None:
            return {"period": period, "slope": None, "slope_rel_pct": None,
                    "slope_z": None, "intercept": None, "p": None,
                    "trend": None, "significant": None}

        # relative slope in percent per year, for RAIN and the dryness metrics
        if mean_full and not np.isnan(mean_full) and mean_full != 0:
            slope_rel_pct = 100.0 * test.slope / mean_full
        else:
            slope_rel_pct = None

        # z-normalized slope in standard deviations per year, for the RQA metrics
        if std_full and not np.isnan(std_full) and std_full != 0:
            slope_z = test.slope / std_full
        else:
            slope_z = None

        return {
            "period":        period,
            "slope":         test.slope,
            "slope_rel_pct": slope_rel_pct,
            "slope_z":       slope_z,
            "intercept":     test.intercept,
            "p":             test.p,
            "trend":         test.trend,
            "significant":   test.p < 0.05,
        }

    test_full = _safe_mk(df[metric])
    print(f"\n--- MK Test: {metric}")
    print(f"Mann-Kendall full: {test_full}")

    period_tests = []
    if mark_dates is not None:
        for y_start, y_end in mark_dates:
            t_start = pd.Timestamp(f"{y_start}-01-01")
            t_end   = pd.Timestamp(f"{y_end}-01-01")
            seg     = (t >= t_start) & (t < t_end)
            label   = f"{y_start}-{y_end}"
            test    = _safe_mk(df.loc[seg, metric])
            print(f"Mann-Kendall {label}: {test}")
            period_tests.append((label, test))

    mk_results = {
        "cluster_id": cluster_id,
        "metric":     metric,
        "median":     float(df[metric].median()),
        "mean_full":  float(mean_full) if not np.isnan(mean_full) else None,
        "std_full":   float(std_full)  if not np.isnan(std_full)  else None,
        "periods":    [_mk_row(test_full, "full")] + [
            _mk_row(test, label) for label, test in period_tests
        ],
    }

    # ── CSV export ────────────────────────────────────────────────────────
    if save_csv and out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        cols = ["time", metric, f"{metric}_med", f"{metric}_low", f"{metric}_high"]
        if surr_low_col in df.columns:
            cols += [surr_low_col, surr_high_col]
        if rank_col in df.columns:
            cols += [rank_col]
        df_out = df[cols].copy()
        if test_full is not None:
            mk_rows = [{"time": "MK_full", metric: None,
                        f"{metric}_med": None, f"{metric}_low": None,
                        f"{metric}_high": None,
                        "mk_slope":           test_full.slope,
                        "mk_intercept":       test_full.intercept,
                        "mk_p":               test_full.p,
                        "mk_trend_direction": test_full.trend}]
            df_out = pd.concat([df_out, pd.DataFrame(mk_rows)], ignore_index=True)
        df_out.to_csv(
            Path(out_dir) / f"cluster_{cluster_id:02d}_{metric}_{csv_name}.csv",
            index=False,
        )

    return mk_results


