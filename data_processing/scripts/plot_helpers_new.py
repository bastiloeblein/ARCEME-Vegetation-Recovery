import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from typing import Tuple, Optional, Union, List
import pandas as pd
import numpy as np


def find_cloud_free_indices(ds, threshold=0.99):
    """
    Finds indices of Sentinel 2 timesteps, that are almost cloud free.
    Based on the "mask_phys_strict" mask
    """
    # 1. Calculates mean over x and y per timestep
    quality_means = ds.mask_phys_strict.mean(dim=("x", "y"))

    # 2. Find indices which meet criteria
    indices = np.where(quality_means > threshold)[0]

    # 3. Calculate amount of indices
    count = len(indices)

    print(f"Found cloudfree timesteps (> {threshold*100}%): {count}")

    return indices.tolist()


def plot_acquisition_timelines(ds):
    # 1. Zeitstempel extrahieren und in Pandas Datetime umwandeln
    t1 = pd.to_datetime(ds.time_sentinel_1_rtc.values)
    t2 = pd.to_datetime(ds.time_sentinel_2_l2a.values)

    fig, ax = plt.subplots(figsize=(15, 4))

    # 2. Sentinel-1 (Radar) plotten - Blaue Punkte/Linien
    ax.vlines(
        t1,
        0.8,
        1.2,
        color="royalblue",
        alpha=0.6,
        label="Sentinel-1 (Radar)",
        linewidth=1,
    )
    ax.scatter(t1, [1.0] * len(t1), color="royalblue", s=10, alpha=0.5)

    # 3. Sentinel-2 (Optisch) plotten - Grüne Punkte/Linien
    ax.vlines(
        t2,
        1.8,
        2.2,
        color="forestgreen",
        alpha=0.6,
        label="Sentinel-2 (Optisch)",
        linewidth=1,
    )
    ax.scatter(t2, [2.0] * len(t2), color="forestgreen", s=10, alpha=0.5)

    # 4. Optik & Achsen-Styling
    ax.set_ylim(0.5, 2.5)
    ax.set_yticks([1.0, 2.0])
    ax.set_yticklabels(["Sentinel-1", "Sentinel-2"], fontsize=12, fontweight="bold")

    # X-Achse formatieren
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Strich pro Monat
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))  # 'Jan 2024'
    plt.xticks(rotation=45)

    ax.set_title(
        "Vergleich der Aufnahme-Zeitpunkte (Acquisition Timelines)", fontsize=14, pad=15
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()


def plot_acquisition_timelines_filtered(ds, precip_end_date, offset=6, show=True):
    # 1. Setup Dates
    ref_date = pd.to_datetime(precip_end_date)
    start_obs = ref_date - pd.DateOffset(months=offset)
    end_obs = ref_date + pd.DateOffset(months=offset)

    # 2. Extract and Filter Timestamps
    t1_all = pd.to_datetime(ds.time_sentinel_1_rtc.values)
    t2_all = pd.to_datetime(ds.time_sentinel_2_l2a.values)

    t1 = t1_all[(t1_all >= start_obs) & (t1_all <= end_obs)]
    t2 = t2_all[(t2_all >= start_obs) & (t2_all <= end_obs)]

    # 3. Define 5-day intervals (Bins) relative to precip_end_date
    # Generate bins from start to end in 5-day steps
    bins = pd.date_range(start=start_obs, end=end_obs + pd.Timedelta(days=5), freq="5D")
    bin_centers = bins[:-1] + pd.Timedelta(days=2.5)

    fig, ax = plt.subplots(figsize=(15, 7))

    # --- ROW 1: Sentinel-1 (Radar) ---
    ax.vlines(t1, 0.8, 1.2, color="royalblue", alpha=0.4, linewidth=1)
    ax.scatter(
        t1, [1.0] * len(t1), color="royalblue", s=20, alpha=0.6, label="S1 (Radar)"
    )

    # --- ROW 2: Sentinel-2 (Optical) ---
    ax.vlines(t2, 1.8, 2.2, color="forestgreen", alpha=0.4, linewidth=1)
    ax.scatter(
        t2, [2.0] * len(t2), color="forestgreen", s=20, alpha=0.6, label="S2 (Optical)"
    )

    # --- ROW 3: 5-Day Bins ---
    ax.vlines(bins, 2.7, 3.3, color="red", alpha=0.7, linewidth=1.2)

    # Calculate counts per bin and plot markers
    for i in range(len(bins) - 1):
        mask1 = (t1 >= bins[i]) & (t1 < bins[i + 1])
        mask2 = (t2 >= bins[i]) & (t2 < bins[i + 1])

        c1, c2 = mask1.sum(), mask2.sum()
        center = bin_centers[i]

        if c2 > 0:
            ax.scatter(center, 3.1, marker="o", color="forestgreen", s=50, zorder=3)
            ax.text(
                center,
                2.50,
                str(c2),
                color="forestgreen",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        if c1 > 0:
            ax.scatter(center, 2.9, marker="x", color="royalblue", s=50, zorder=3)
            ax.text(
                center,
                3.35,
                str(c1),
                color="royalblue",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

    # --- Styling ---
    ax.set_xlim(start_obs, end_obs)
    ax.set_ylim(0.5, 4.0)
    ax.set_yticks([1.0, 2.0, 3.0])
    ax.set_yticklabels(
        [
            f"Sentinel-1\n(n={len(t1)})",
            f"Sentinel-2\n(n={len(t2)})",
            "5-Day Intervals\n(Counts)",
        ],
        fontsize=11,
        fontweight="bold",
    )

    # X-Axis: Major ticks every month, minor ticks every 5 days
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=0)

    ax.set_title(
        f"Acquisition Density: ±6 Months from {ref_date.strftime('%Y-%m-%d')}",
        fontsize=15,
        pad=25,
    )
    ax.grid(axis="x", which="both", linestyle="--", alpha=0.3)

    # Event Highlight
    ax.axvline(ref_date, color="black", linestyle="-", linewidth=2.5, zorder=4)
    ax.text(
        ref_date,
        3.8,
        "EVENT",
        color="black",
        ha="center",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )

    plt.tight_layout()
    if show:
        plt.show()

    return fig


def plot_rgb(
    ds: xr.Dataset,
    timestep: int,
    time_dim: str = "time_sentinel_2_l2a",
    bands: Tuple[str, str, str] = ("B04", "B03", "B02"),
    ax: Optional[plt.Axes] = None,
    stretch_pct: Tuple[int, int] = (2, 98),
    figsize: Tuple[int, int] = (10, 6),
    cloud_comp: bool = False,
    title_prefix: Optional[str] = "Timestep",
) -> plt.Figure:

    def error_fig(msg):
        fig, ax_err = plt.subplots(figsize=(6, 2))
        ax_err.text(0.5, 0.5, f"ERROR: {msg}", color="red", ha="center", va="center")
        ax_err.axis("off")
        return fig

    # 1. Validation
    if time_dim not in ds.coords or not 0 <= timestep < len(ds.coords[time_dim]):
        return error_fig(f"Timestep {timestep} out of range")

    def stretch_arr(band):
        arr = band.compute().astype("float32")
        p2, p98 = np.percentile(arr, stretch_pct)
        diff = p98 - p2
        if diff == 0:
            diff = 1.0
        return np.clip((arr - p2) / diff, 0, 1)

    # 2. Subset & RGB Creation
    subset = ds.isel({time_dim: timestep})
    try:
        time_val = pd.to_datetime(subset[time_dim].item()).strftime("%Y-%m-%d")
        r = stretch_arr(subset[bands[0]])
        g = stretch_arr(subset[bands[1]])
        b = stretch_arr(subset[bands[2]])
        rgb = np.dstack([r, g, b])
    except Exception as e:
        return error_fig(str(e))

    # 3. Setup Axes & Figure
    if ax is not None:
        ax1 = ax
        fig = ax.get_figure()
        cloud_comp = False
    else:
        if cloud_comp:
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=figsize, constrained_layout=True
            )
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    # 4. Plotting
    try:
        if cloud_comp:
            # SCL/Cloud Mask Logik
            ax1.imshow(rgb)
            ax1.axis("off")

            is_cloud_scl = subset["SCL"].isin([3, 8, 9, 10])
            is_ext_cloud = subset["cloud_mask"].isin([1, 2, 3])
            overlay = np.dstack(
                [
                    (is_cloud_scl & ~is_ext_cloud).astype(float),
                    (is_ext_cloud & ~is_cloud_scl).astype(float),
                    (is_ext_cloud & is_cloud_scl).astype(float),
                ]
            )
            ax2.imshow(rgb)
            ax2.imshow(overlay, alpha=0.5)
            ax2.axis("off")
            fig.suptitle(f"{title_prefix}: {time_val}", fontsize=14)
        else:
            ax1.imshow(rgb)
            ax1.set_title(f"RGB: {time_val}" if ax is None else "")
            ax1.axis("off")

        return fig
    except Exception as e:
        return error_fig(f"Plotfehler: {str(e)}")


def plot_statistical_outliers(ds, time_index, show=True):
    """
    Plots RGB and highlights only the pixels that are permanently
    masked due to frequent outliers.
    """

    # Identify the "Bad Seeds" (Static mask)
    static_bad_pixels = ds["static_veg_filter"] == 0

    # 2. Prepare RGB for a sample timestep (e.g., middle of the series)
    subset = ds.isel(time_sentinel_2_l2a=time_index)

    def stretch(band):
        # Nutze dropna() oder fülle NaNs für das Perzentil, damit es nicht schluckt
        arr = band.values
        valid_vals = arr[~np.isnan(arr)]
        if valid_vals.size == 0:
            return np.zeros_like(arr)
        p2, p98 = np.percentile(valid_vals, (2, 98))
        return np.clip((arr - p2) / (p98 - p2), 0, 1)

    # Stelle sicher, dass die Bänder existieren (nach Normalisierung heißen sie meist _normalized)
    r = stretch(subset.B04_normalized)
    g = stretch(subset.B03_normalized)
    b = stretch(subset.B02_normalized)
    rgb = np.dstack([r, g, b])

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Left: RGB
    ax1.imshow(rgb)
    ax1.set_title(
        f"Reference RGB\n({pd.to_datetime(subset.time_sentinel_2_l2a.values).date()})"
    )
    ax1.axis("off")

    # Right: The Outliers
    # Show the mask in bright red on a black background for maximum contrast
    outlier_map = static_bad_pixels.astype(float).values.squeeze()
    ax2.imshow(outlier_map, cmap="Reds")
    ax2.set_title(f"Permanently Masked Pixels\n(Freq > {0.75*100}%)")
    ax2.axis("off")

    if show:
        plt.show()

    print(f"Total static outliers found: {int(static_bad_pixels.sum())} pixels")

    return fig


def plot_nan_distribution(ds_old, ds_new, var_name, cutoff_date):
    """
    Plottet den Prozentsatz der NaNs pro Zeitschritt für eine Variable.
    """
    time_dim = "time_sentinel_2_l2a"

    # 1. Berechne NaN-Prozent pro Timestep (Mittelwert über x und y)
    nan_perc_old = (ds_old[var_name].isnull().mean(dim=["x", "y"]) * 100).compute()
    nan_perc_new = (ds_new[var_name].isnull().mean(dim=["x", "y"]) * 100).compute()

    time_axis = ds_old[time_dim].values

    fig = plt.figure(figsize=(15, 5))

    # Flächen füllen für bessere Sichtbarkeit
    plt.fill_between(
        time_axis, nan_perc_old, label="Original NaNs", color="red", alpha=0.3
    )
    plt.fill_between(
        time_axis,
        nan_perc_new,
        label="Post-Interpolation NaNs",
        color="green",
        alpha=0.5,
    )

    # Vertikale Linie für das Event
    plt.axvline(
        x=cutoff_date,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Precip End Date (Event)",
    )

    # Styling
    plt.title(f"NaN Distribution over Time for {var_name}", fontsize=14)
    plt.ylabel("NaNs per Timestep (%)")
    plt.xlabel("Date")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return fig


def plot_spatial_nan_frequency(ds, var_name, cutoff_date):
    """
    Visualisiert, wie oft jeder Pixel im Context-Zeitraum NaN ist.
    """
    time_dim = "time_sentinel_2_l2a"

    # 1. Context isolieren
    context = ds.sel({time_dim: slice(None, cutoff_date)})

    # 2. Berechne die relative Häufigkeit von NaNs pro Pixel (0-100%)
    # isnull() gibt 1 für NaN, 0 für Daten. mean über die Zeit gibt die Rate.
    nan_freq = (context[var_name].isnull().mean(dim=time_dim) * 100).compute()

    # 3. Plotten
    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(
        nan_freq, cmap="RdYlGn_r", vmin=0, vmax=100
    )  # Rot = Häufig NaN, Grün = Fast immer Daten

    cbar = plt.colorbar(im)
    cbar.set_label("Frequency of NaNs (%)", fontsize=12)

    plt.title(f"Spatial NaN Distribution in Context: {var_name}", fontsize=14)
    plt.xlabel("x-coordinate (pixel)")
    plt.ylabel("y-coordinate (pixel)")

    # Zeige Statistik im Plot
    avg_nan = float(nan_freq.mean())
    plt.annotate(
        f"Avg. NaN rate: {avg_nan:.1f}%",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        color="white",
        bbox=dict(boxstyle="round", fc="black", alpha=0.6),
    )

    return fig


def plot_variable_analysis(
    ds: xr.Dataset,
    variables_to_plot: Union[str, List[str]],
    time_dim: str = "time_sentinel_2_l2a",
    spatial_dims: Tuple[str, ...] = ("x", "y"),
    ax: Optional[plt.Axes] = None,  # Neu: Optionaler Parameter für Subplots
    plot_ts_mean: Union[bool, List[bool]] = True,
    plot_ts_median: Union[bool, List[bool]] = False,
    plot_ts_std: Union[bool, List[bool]] = False,
    title: str = "Multi-Variable Time Series Analysis",
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Axes:
    """
    Adaptive plotting function for 3D (S1/S2) and 1D (ERA5) data.
    Can be used as a standalone plot or as a subfigure in a larger grid.
    """
    if isinstance(variables_to_plot, str):
        variables_to_plot = [variables_to_plot]

    n_vars = len(variables_to_plot)
    colors = plt.colormaps.get_cmap("tab10").colors

    # --- Setup: Check if we are plotting in a subfigure ---
    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True

    # Helper for booleans
    def expand_bools(param, length):
        if isinstance(param, bool):
            return [param] * length
        return param

    p_mean = expand_bools(plot_ts_mean, n_vars)
    p_median = expand_bools(plot_ts_median, n_vars)
    p_std = expand_bools(plot_ts_std, n_vars)

    periods = {
        "Drought": (
            pd.to_datetime(ds.attrs["drought_start_date"]),
            pd.to_datetime(ds.attrs["drought_end_date"]),
            "red",
        ),
        "Precipitation": (
            pd.to_datetime(ds.attrs["precip_start_date"]),
            pd.to_datetime(ds.attrs["precip_end_date"]),
            "blue",
        ),
    }

    # --- 4. Period Highlighting (Adaptive for Single Days) ---
    for name, (start, end, col) in periods.items():
        if start == end:
            # Event lasts only one day: Draw a dashed vertical line
            ax.axvline(
                start,
                color=col,
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                zorder=2,
                label=f"{name} (Single Day)",
            )
        else:
            # Event lasts multiple days: Draw the shaded area
            ax.axvspan(
                start, end, facecolor=col, alpha=0.1, zorder=0, label=f"{name} Period"
            )

    for i, var_name in enumerate(variables_to_plot):
        data = ds[var_name]
        color = colors[i % len(colors)]
        has_spatial = all(dim in data.dims for dim in spatial_dims)

        if has_spatial:
            if "is_veg" in ds:
                data = data.where(ds.is_veg == 1)

            display_val = data.mean(dim=spatial_dims).compute() if p_mean[i] else None
            if p_std[i] and display_val is not None:
                std_val = data.std(dim=spatial_dims).compute()
                ax.fill_between(
                    display_val[time_dim].values,
                    display_val - std_val,
                    display_val + std_val,
                    color=color,
                    alpha=0.15,
                    zorder=1,
                )
            if p_median[i]:
                data.median(dim=spatial_dims).compute().plot.line(
                    ax=ax,
                    color=color,
                    linestyle="--",
                    alpha=0.6,
                    label=f"{var_name} (Med)",
                )
        else:
            # for 1D data (ERA5)
            display_val = data.squeeze().compute() if p_mean[i] else None

        if display_val is not None:
            label = f"{var_name}" + (" (Mean)" if has_spatial else "")
            display_val.plot.line(
                ax=ax,
                label=label,
                color=color,
                linewidth=1.5,
                marker="o",  # Adds dots to each data point
                markersize=4,  # Size of the dots
                alpha=0.8,
                zorder=5,
            )

    for name, (start, end, col) in periods.items():
        ax.axvspan(start, end, facecolor=col, alpha=0.1, zorder=0, label=f"{name}")

    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    ax.grid(True, alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax
