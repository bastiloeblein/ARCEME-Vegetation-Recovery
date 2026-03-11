import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
import pandas as pd
import numpy as np


def align_all_to_5d(ds, masking_type, show=True):

    # Enforce same chunking for all vars
    ds = ds.unify_chunks()

    # 1. Define which mask to use
    mask = ds[f"mask_phys_{masking_type}"]

    # 2. Define common date as base for resampling
    start_date = pd.to_datetime(ds.precip_end_date)

    # 2. Identify all variables with S1 and S2 temporal dimension plus static variables
    s1_vars = [v for v in ds.data_vars if "time_sentinel_1_rtc" in ds[v].dims]
    s2_vars = [
        v
        for v in ds.data_vars
        if "time_sentinel_2_l2a" in ds[v].dims and "mask_phys" not in v
    ]  # Exclude masks
    static_features = ds[["ESA_LC", "COP_DEM", "is_veg"]]

    # 2. Resampling S2 data (optical):
    s2_resampler = (
        ds[s2_vars]
        .astype("float32")
        .where(mask == 1)
        .resample(time_sentinel_2_l2a="5D", label="right", origin=start_date)
    )

    # 2.1 Calculate the main statistics
    s2_res = s2_resampler.median()

    # 2.2 Add additional statistics as new variables
    # We use .count() which automatically excludes NaNs
    s2_res_count = (
        s2_resampler.count().rename({v: f"{v}_count" for v in s2_vars}).astype("uint8")
    )
    s2_res_min = s2_resampler.min().rename({v: f"{v}_min" for v in s2_vars})
    s2_res_max = s2_resampler.max().rename({v: f"{v}_max" for v in s2_vars})
    s2_res_std = s2_resampler.std().rename({v: f"{v}_std" for v in s2_vars})

    assess_data_availability(ds[s2_vars[0]], s2_res[s2_vars[0]], "s2")

    # 3. Resampling S1 data (radar):
    s1_resampler = (
        ds[s1_vars]
        .astype("float32")
        .resample(time_sentinel_1_rtc="5D", label="right", origin=start_date)
    )

    # 3.1 Calculate the main statistics
    s1_res = s1_resampler.median()
    s1_res = s1_res.rename({"time_sentinel_1_rtc": "time_sentinel_2_l2a"})

    # 3.2 Add additional statistics as new variables
    # We use .count() which automatically excludes NaNs
    s1_res_count = (
        s1_resampler.count()
        .rename({v: f"{v}_count" for v in s1_vars})
        .rename({"time_sentinel_1_rtc": "time_sentinel_2_l2a"})
        .astype("uint8")
    )
    s1_res_min = (
        s1_resampler.min()
        .rename({v: f"{v}_min" for v in s1_vars})
        .rename({"time_sentinel_1_rtc": "time_sentinel_2_l2a"})
    )
    s1_res_max = (
        s1_resampler.max()
        .rename({v: f"{v}_max" for v in s1_vars})
        .rename({"time_sentinel_1_rtc": "time_sentinel_2_l2a"})
    )
    s1_res_std = (
        s1_resampler.std()
        .rename({v: f"{v}_std" for v in s1_vars})
        .rename({"time_sentinel_1_rtc": "time_sentinel_2_l2a"})
    )

    # Merge S1 stats and THEN reindex to match S2 time exactly
    s1_combined_stats = xr.merge(
        [s1_res, s1_res_count, s1_res_min, s1_res_max, s1_res_std]
    )
    # Optional: Think about it: This forces S1 to have the exact same time points as S2 (if there are S1 observations before or after S2 period, they get lost)
    s1_aligned = s1_combined_stats.reindex(
        time_sentinel_2_l2a=s2_res.time_sentinel_2_l2a, method=None
    )

    assess_data_availability(ds[s1_vars[0]], s1_res[s1_vars[0]], "s1")

    # Check consistency of resampled timesteps for S1 and S2
    assert len(s2_res.time_sentinel_2_l2a) == len(
        s1_aligned.time_sentinel_2_l2a
    ), f"Missmatch in number of timesteos! S2: {len(s2_res.time_sentinel_2_l2a)}, S1: {len(s1_aligned.time_sentinel_2_l2a)}"
    assert s2_res.time_sentinel_2_l2a.equals(
        s1_aligned.time_sentinel_2_l2a
    ), "Timesteps of S1 and S2 do not match after resampling!"

    print("✅ Timesteps perfectly aligned.")

    # 5. Get information on aggregation
    fig = plot_full_acquisition_analysis(ds, show)

    # 6. Merge datasets
    combined = xr.merge(
        [
            s2_res,
            s2_res_count,
            s2_res_min,
            s2_res_max,
            s2_res_std,
            s1_aligned,
            static_features,
        ]
    )  # Add static features (later on maybe think about broadcasting them to time_sentinel_2 as well)

    # Create mask for S2 and S1
    valid_binary = combined["kNDVI"].notnull()
    combined[f"s2_final_mask_{masking_type}"] = valid_binary.astype("uint8")
    combined["s1_final_mask"] = combined["vh"].notnull().astype("uint8")

    # Some assertions
    assert (
        combined[f"s2_final_mask_{masking_type}"].max() <= 1
        and combined[f"s2_final_mask_{masking_type}"].min() >= 0
    )
    assert "time_sentinel_2_l2a" in combined.dims
    assert (
        combined[f"s2_final_mask_{masking_type}"].shape[0]
        == combined.s1_final_mask.shape[0]
    ), "Time dimension mismatch after merge!"

    return combined, fig


def assess_data_availability(ds_before, ds_after, sentinel):
    # 1. Erstelle ein statistisches Dataset (Lazy)
    # Wir bündeln alle Berechnungen in ein Objekt
    stats_ds = xr.Dataset(
        {
            "invalid_pre": ds_before.isnull().sum(),
            "gaps_post": ds_after.isnull().sum(),
            "permanent_gaps": (
                ds_after.notnull().sum(dim="time_sentinel_2_l2a") == 0
            ).sum(),
        }
    )

    # 2. Ein einziger .compute() Aufruf triggert die gesamte Engine nur EINMAL
    # Das spart massiv Zeit, da die Daten nur einmal gestreamt werden
    computed_stats = stats_ds.compute()

    # Werte extrahieren
    invalid_pre = computed_stats.invalid_pre.item()
    gaps_post = computed_stats.gaps_post.item()
    permanent_gaps = computed_stats.permanent_gaps.item()

    # Konstanten für die Berechnung
    total_pixels_before = ds_before.size
    total_pixels_after = ds_after.size
    if sentinel == "s1":
        timesteps_before = len(ds_before.time_sentinel_1_rtc)
    else:
        timesteps_before = len(ds_before.time_sentinel_2_l2a)
    timesteps_after = len(ds_after.time_sentinel_2_l2a)

    # Prozente berechnen
    pct_invalid_pre = (invalid_pre / total_pixels_before) * 100
    pct_gaps_post = (gaps_post / total_pixels_after) * 100
    pct_permanent = (permanent_gaps / (ds_after.shape[1] * ds_after.shape[2])) * 100

    print(f"--- Data Availability Report: {sentinel.upper()}  ---")
    print(f"Timesteps: {timesteps_before} -> {timesteps_after}")
    print(f"Pre-Resampling Invalidity: {pct_invalid_pre:.2f}%")
    print(f"Post-Resampling Gaps:      {pct_gaps_post:.2f}%")
    print(f"Permanently missing px:    {pct_permanent:.2f}%")

    return pct_gaps_post


def plot_full_acquisition_analysis(ds, show=True):

    # 1. Timestamps & Validity
    t1_raw = pd.to_datetime(ds.time_sentinel_1_rtc.values)
    t2_raw = pd.to_datetime(ds.time_sentinel_2_l2a.values)
    ref_date = pd.to_datetime(ds.precip_end_date)

    # Check validity (pixel counts)
    s1_valid_counts = ds["vh"].count(dim=["x", "y"]).values
    s2_valid_counts = ds["kNDVI"].count(dim=["x", "y"]).values

    # 2. Dynamic Binning
    all_times = np.concatenate([t1_raw, t2_raw])
    start_obs, end_obs = all_times.min(), all_times.max()

    bins_before = pd.date_range(
        end=ref_date, start=start_obs - pd.Timedelta(days=5), freq="5D"
    )
    bins_after = pd.date_range(
        start=ref_date, end=end_obs + pd.Timedelta(days=5), freq="5D"
    )
    bins = sorted(list(set(bins_before.union(bins_after))))
    bin_centers = [bins[i] + (bins[i + 1] - bins[i]) / 2 for i in range(len(bins) - 1)]
    num_bins = len(bins) - 1

    # Stats counters
    s1_stats = {"no_obs": 0, "only_nan": 0}
    s2_stats = {"no_obs": 0, "only_nan": 0}

    fig, ax = plt.subplots(figsize=(22, 10))

    # --- Plotting Original Data ---
    ax.scatter(
        t2_raw,
        [2.0] * len(t2_raw),
        color=["forestgreen" if c > 0 else "red" for c in s2_valid_counts],
        s=45,
        zorder=4,
    )
    ax.scatter(
        t1_raw,
        [1.0] * len(t1_raw),
        color=["royalblue" if c > 0 else "red" for c in s1_valid_counts],
        s=45,
        zorder=4,
    )

    # Bin Dividers
    ax.vlines(bins, 2.7, 3.6, color="red", alpha=0.3, linewidth=1.2)

    s1_is_valid_any = s1_valid_counts > 0
    s2_is_valid_any = s2_valid_counts > 0

    for i in range(num_bins):
        m1_idx = np.where((t1_raw >= bins[i]) & (t1_raw < bins[i + 1]))[0]
        m2_idx = np.where((t2_raw >= bins[i]) & (t2_raw < bins[i + 1]))[0]

        c1, c2 = len(m1_idx), len(m2_idx)

        # Superschneller Check auf dem vorbereiteten Boolean-Array
        valid1 = s1_is_valid_any[m1_idx].any() if c1 > 0 else False
        valid2 = s2_is_valid_any[m2_idx].any() if c2 > 0 else False

        # Statistics
        if c1 == 0:
            s1_stats["no_obs"] += 1
        elif not valid1:
            s1_stats["only_nan"] += 1

        if c2 == 0:
            s2_stats["no_obs"] += 1
        elif not valid2:
            s2_stats["only_nan"] += 1

        center = bin_centers[i]
        if c2 > 0:
            color = "forestgreen" if valid2 else "red"
            ax.scatter(center, 3.3, marker="o", color=color, s=70, alpha=0.7)
            ax.text(
                center,
                3.45,
                str(c2),
                color=color,
                ha="center",
                fontweight="bold",
                fontsize=11,
            )

        if c1 > 0:
            color = "royalblue" if valid1 else "red"
            ax.scatter(center, 3.0, marker="x", color=color, s=70, alpha=0.7)
            ax.text(
                center,
                2.75,
                str(c1),
                color=color,
                ha="center",
                fontweight="bold",
                fontsize=11,
            )

    # --- Printing detailed Report ---
    total_empty_s1 = s1_stats["no_obs"] + s1_stats["only_nan"]
    total_empty_s2 = s2_stats["no_obs"] + s2_stats["only_nan"]

    print("\n📊 DETAILED BINNING REPORT (5-Day Intervals)")
    print(f"Total Intervals: {num_bins}")
    print(
        f"{'Platform':<15} | {'Missing Obs':<15} | {'Only NaN':<15} | {'Total Unusable'}"
    )
    print("-" * 70)
    print(
        f"{'Sentinel-1':<15} | {s1_stats['no_obs']:<15} | {s1_stats['only_nan']:<15} | {total_empty_s1} ({total_empty_s1/num_bins:.1%})"
    )
    print(
        f"{'Sentinel-2':<15} | {s2_stats['no_obs']:<15} | {s2_stats['only_nan']:<15} | {total_empty_s2} ({total_empty_s2/num_bins:.1%})"
    )
    print("-" * 70 + "\n")

    # Final Axis Styling
    ax.set_ylim(0.5, 4.2)
    ax.set_yticks([1.0, 2.0, 3.15])
    ax.set_yticklabels(
        [
            f"S1 Original\n(n={len(t1_raw)})",
            f"S2 Original\n(n={len(t2_raw)})",
            "5-Day Bins\n(Counts)",
        ],
        fontweight="bold",
        fontsize=12,
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.axvline(ref_date, color="black", linewidth=3, zorder=5)
    ax.text(
        ref_date,
        4.0,
        "EVENT",
        color="white",
        ha="center",
        fontweight="bold",
        bbox=dict(facecolor="black", boxstyle="round"),
    )

    plt.tight_layout()

    if show:
        plt.show()

    return fig
