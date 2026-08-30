"""
analysis.py

Command-line entry point. Runs the full pipeline (prepare -> cluster ->
RQA -> plots) for one or several presets (season/window combinations).

Usage: --mode preset|single|all, see PRESETS for available presets.
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
from pathlib import Path


from summarized_func import (
    rqa_full,
    prepare,
    run_k_analysis,
    smooth_cluster,
    heatmap_periods
)


from rqa import plot_theiler_parallel
from plots_and_stats import (plot_rqa_with_rain,
                             export_communities,
                             plot_mk_dotplot,
                             plot_smoothing,
                             plot_cluster,
                             plot_two_panel)


# preset name -> (window_size, months, year filter)
PRESETS = [
    ("JJAS_full",    122, [6, 7, 8, 9],  None),
    ("OND_full",     91, [10, 11, 12],  None ),
]



#### HELPER ##########
def _parse_months(s: str):
    """Parse a comma-separated month list, e.g. "6,7,8,9" -> [6, 7, 8, 9]."""
    s = s.strip().lower()
    if s in ("none", ""):
        return None
    return [int(x) for x in s.split(",")]


def _parse_year(s: str):
    """Pass through a year filter string, e.g. "before_1991", or None if unset."""
    s = s.strip()
    if s.lower() in ("none", ""):
        return None
    return s


def _parse_figsize(s: str):
    """Parse a "width,height" string into a (float, float) tuple, or None if unset."""
    s = s.strip().lower()
    if s in ("none", ""):
        return None
    vals = [x.strip() for x in s.split(",")]
    if len(vals) != 2:
        raise ValueError("--figsize must be in format 'width,height', e.g. '10,6'")
    return (float(vals[0]), float(vals[1]))


def _parse_bool(s: str):
    """Parse common truthy/falsy strings into a bool."""
    s = s.strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError("Boolean value expected (true/false)")



def _resolve_plot_params(global_figsize, global_scatter, cluster_figsize, cluster_scatter, panel_figsize, panel_scatter):
    """Fall back to the global figsize/scatter size wherever a plot-specific value wasn't given."""
    c_fig = cluster_figsize if cluster_figsize is not None else global_figsize
    p_fig = panel_figsize if panel_figsize is not None else global_figsize
    c_scat = cluster_scatter if cluster_scatter is not None else global_scatter
    p_scat = panel_scatter if panel_scatter is not None else global_scatter
    return c_fig, c_scat, p_fig, p_scat


def _preset_dict():
    """Build a lookup {preset_name: (window_size, months, year)} from PRESETS."""
    return {name: (ws, months, year) for name, ws, months, year in PRESETS}



def run_pipeline(
    path: str,
    window_size: int,
    months,
    year,
    tag: str,
    out_dir="results",
    figsize=None,
    scatter=None,
    cluster_figsize=None,
    cluster_scatter=None,
    panel_figsize=None,
    panel_scatter=None,
    use_theiler_window=True,
    shapefile_path=None,
    min_community_size="auto",
    sort_by: str = "south",
):
    """
    Run the full pipeline for one preset/parameter set and save all
    outputs under out_dir/tag.

    Steps: PCA-based spectral filtering and heatmap prep, community
    detection (agglomerative clustering + smoothing), rolling-window
    multivariate RQA with Theiler-window and Mann-Kendall trend analysis,
    and the corresponding diagnostic/result plots.
    """

    base = Path(out_dir) / tag
    base.mkdir(parents=True, exist_ok=True)

    print(f"\n=== RUN {tag} | ws={window_size}, months={months}, year={year} ===")


############# 1. PCA-based spectral filtering + heatmap prep ##############

    (
        ds,
        B_norm,
        ds_season,
        ds_norm_season,
        ds_norm_all,
        heatmap_fraction,
        heatmap,
        coords_lat,
        coords_lon,
        ds_anom,
        ds_anom_std,
        ds_detrended,
        ds_detrend_norm,
    ) = prepare(
        path=path,
        window_size=window_size,
        plot_path=base,
        shapefile_path=shapefile_path,
        overlap_p=0,
        SC_threshold=None,
        months=months,
        year=year,
        plot_curves=True,
        plot_stride=25,
        plot_first_n=2,
        title="Heatmap",
        export_file=True,
        min_community_size=min_community_size,
    )


############# 2. Community detection (clustering) ##############

    # sweep k, pick the final clustering via the q-knee point
    df, best_labels, best_meta = run_k_analysis(
        heatmap=heatmap,
        ds=ds_norm_season,
        path=base,
        use_knee_as_final=True,
        knee_S=1,
    )

    # majority-filter, split disconnected pieces, merge small clusters,
    # relabel to consecutive community IDs
    labels_final, mask, all_mask = smooth_cluster(
        ds_norm_all,
        best_labels,
        stage_size=3,
        min_cluster_size=0.10,
        split=True,
        sort_by=sort_by,
    )

    export_communities(
        all_mask=all_mask,
        ds=ds_norm_season,
        output_path=base / "communities.shp",
    )

    c_fig, c_scat, p_fig, p_scat = _resolve_plot_params(
        figsize,
        scatter,
        cluster_figsize,
        cluster_scatter,
        panel_figsize,
        panel_scatter,
    )

    cluster_kwargs = {"plot_path": base, "show": False}
    panel_kwargs = {"plot_path": base, "show": False}
    if c_fig is not None:
        cluster_kwargs["figsize"] = c_fig
    if c_scat is not None:
        cluster_kwargs["scatter"] = c_scat
    if p_fig is not None:
        panel_kwargs["figsize"] = p_fig
    if p_scat is not None:
        panel_kwargs["scatter"] = p_scat

    # raw clustering result before smoothing
    plot_cluster(
        best_meta["k_found"],
        best_labels,
        best_meta["lons"],
        best_meta["lats"],
        **cluster_kwargs
    )
    # side-by-side comparison of raw vs. smoothed labels
    plot_smoothing(
        heatmap_fraction,
        best_labels,
        labels_final.ravel()[:len(best_meta["lats"])],
        coords_lon,
        coords_lat,
        best_meta["lats"],
        best_meta["lons"],
        **panel_kwargs
    )

    ############# community stability across sub-periods ###############
    heatmap_periods(
        ds=ds_norm_season,
        window_size=window_size,
        overlap_p=0.0,
        plot_path=base / "periods",
        SC_threshold=None,
        period_length=30,
        custom_periods=[(1951, 2023), (1951, 1981), (1961, 1991), (1971, 2001), (1981, 2011), (1991, 2021)],
        figsize=None,
        start_year=1951,
        end_year=2023,
        show=False,
        shapefile_path=shapefile_path,
    )

    # final two-panel figure: raw clustering + smoothed communities
    plot_two_panel(
        heatmap_values=heatmap_fraction,
        labels1=best_labels,
        labels2=labels_final.ravel()[:len(best_meta["lats"])],
        coords_lat=coords_lat,
        coords_lons=coords_lon,
        lats=best_meta["lats"],
        lons=best_meta["lons"],
        shapefile_path=shapefile_path,
        **panel_kwargs
    )

   ############# 3. RQA, Mann-Kendall trend test, surrogate testing ##############

    # per-community Theiler window, used to exclude short-range
    # autocorrelation from the recurrence quantification
    t_w_dict = None
    if use_theiler_window:
        t_w_dict = plot_theiler_parallel(
            all_cluster_masks=all_mask,
            ds=ds_norm_season,
            plot_path=base,
            use_cosine=True,
            max_delta=90,
            n_jobs=1,
            show=False,
        )
    if t_w_dict is None:
        t_w_dict = {f"Community {i+1}": 1 for i in range(len(all_mask))}


    res = rqa_full(
        all_mask,
        ds_norm_season,
        ds_og=ds_season,
        plot_path=base,
        t_w_dict=t_w_dict,
        use_cosine=True,
        q=0.1,
        alpha=0.05,
        window_size=window_size * 6,
        step=window_size,
        median_w="2190D",
        mark_dates=[(1951, 1981), (1961, 1991), (1971, 2001), (1981, 2011), (1991, 2021)],
        save_csv=True,
        csv_name=tag,
        out_dir=base,
        show=False,
        rp_path=base,
        rp_select=3,
    )

    plot_rqa_with_rain(
        all_results=res,
        all_mask=all_mask,
        plot_path=base,
        split_times=["1961-01-06", "1991-01-06"],
        show=True,
    )


def main():
    p = argparse.ArgumentParser(description="Run full pipeline with selectable presets or single params")

    p.add_argument("--path", default="data/Rainfall_SP_1951_2023.csv")
    p.add_argument("--shapefile", default=None, help="Path to shapefile for clipping")
    p.add_argument("--out_dir", default="results", help="Base output directory (results/<tag>/...)")
    p.add_argument("--mode", choices=["preset", "single", "all"], default="preset")

    # preset / all
    p.add_argument("--preset", default="summer_full", help="Name from PRESETS (mode=preset)")
    p.add_argument("--only", default=None, help="Comma-separated preset names for mode=all")
    p.add_argument("--list", action="store_true", help="List presets and exit")

    # single
    p.add_argument("--window", type=int, help="Window size (mode=single)")
    p.add_argument("--months", default="none", help='e.g. "6,7,8,9" or "none" (mode=single)')
    p.add_argument("--year", default="none", help='e.g. "before_1991", "after_1991", or "none" (mode=single)')
    p.add_argument("--figsize", default="none", help='Figure size for plotting functions, e.g. "10,6" or "none"')
    p.add_argument("--scatter", type=float, default=None, help="Scatter point size for plotting functions")
    p.add_argument("--cluster_figsize", default="none", help='Figure size for plot_cluster, e.g. "8,6" or "none"')
    p.add_argument("--cluster_scatter", type=float, default=None, help="Scatter point size for plot_cluster")
    p.add_argument("--panel_figsize", default="none", help='Figure size for plot_three_panel, e.g. "16,6" or "none"')
    p.add_argument("--panel_scatter", type=float, default=None, help="Scatter point size for plot_three_panel")
    p.add_argument("--min_community_size", default="auto",
                   help='Fixed minimum community size, e.g. "34", or "auto" for Kneedle detection')
    p.add_argument("--sort_by", default="south", choices=["north", "south"],
                   help="Ordering rule for cluster IDs")

    p.add_argument(
        "--use_theiler_window",
        type=_parse_bool,
        default=True,
        help='Use Theiler window in RQA/correlation calculations (true/false).',
    )

    args = p.parse_args()

    mcs = args.min_community_size
    mcs = mcs if str(mcs).lower() == "auto" else int(mcs)


    figsize = _parse_figsize(args.figsize)
    cluster_figsize = _parse_figsize(args.cluster_figsize)
    panel_figsize = _parse_figsize(args.panel_figsize)

    presets = _preset_dict()

    if args.list:
        for name, ws, m, y in PRESETS:
            print(f"{name}: window={ws}, months={m}, year={y}")
        return

    if args.mode == "all":
        # run every preset, optionally restricted to --only
        only_set = None
        if args.only:
            only_set = {x.strip() for x in args.only.split(",") if x.strip()}

        for name, ws, m, y in PRESETS:
            if only_set is not None and name not in only_set:
                continue
            run_pipeline(
                args.path, ws, m, y,
                tag=name,
                out_dir=args.out_dir,
                figsize=figsize,
                scatter=args.scatter,
                cluster_figsize=cluster_figsize,
                cluster_scatter=args.cluster_scatter,
                panel_figsize=panel_figsize,
                panel_scatter=args.panel_scatter,
                use_theiler_window=args.use_theiler_window,
                shapefile_path=args.shapefile,
                min_community_size=mcs,
                sort_by=args.sort_by,
            )

        return

    if args.mode == "preset":
        # run a single named preset from PRESETS
        if args.preset not in presets:
            raise ValueError(f"Unknown preset '{args.preset}'. Use --list to see options.")
        ws, m, y = presets[args.preset]
        run_pipeline(
            args.path,
            ws,
            m,
            y,
            tag=args.preset,
            out_dir=args.out_dir,
            figsize=figsize,
            scatter=args.scatter,
            cluster_figsize=cluster_figsize,
            cluster_scatter=args.cluster_scatter,
            panel_figsize=panel_figsize,
            panel_scatter=args.panel_scatter,
            use_theiler_window=args.use_theiler_window,
            shapefile_path=args.shapefile,
            min_community_size=mcs,
            sort_by=args.sort_by,
        )
        return

    # single: run one arbitrary window/months/year combination, not tied to a preset
    if args.window is None:
        raise ValueError("mode=single requires --window")
    m = _parse_months(args.months)
    y = _parse_year(args.year)
    tag = f"custom_ws{args.window}_m{args.months}_y{args.year}".replace(",", "-")
    run_pipeline(
        args.path,
        args.window,
        m,
        y,
        tag=tag,
        out_dir=args.out_dir,
        figsize=figsize,
        scatter=args.scatter,
        cluster_figsize=cluster_figsize,
        cluster_scatter=args.cluster_scatter,
        panel_figsize=panel_figsize,
        panel_scatter=args.panel_scatter,
        use_theiler_window=args.use_theiler_window,
        shapefile_path=args.shapefile,
        min_community_size=mcs,
        sort_by=args.sort_by,
    )


if __name__ == "__main__":
    main()
