import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.model_selection import GroupKFold
import random
import os


def create_spacetime_folds(
    valid_zarrs_paths,
    df_path,
    spacevar=None,
    timevar=None,
    k=10,
    classvar=None,
    seed=42,
    show=True,
    save_path=None,
):
    """
    Python translation of CAST::CreateSpacetimeFolds (Meyer et al., 2018).
    Code adapted from https://github.com/HannaMeyer/CAST/blob/master/R/CreateSpacetimeFolds.R
    Creates spatial, temporal, or spatio-temporal folds for cross validation.

    Args:
        valid_zarrs_paths (list): List of paths to valid zarr files.
        df_path (str): Path to the dataframe file.
        spacevar (str): Column name for spatial units (e.g., 'koppen_geiger' or 'cube_id').
        timevar (str): Column name for temporal units (e.g., 'pheno_season_name').
        k (int): Number of folds.
        classvar (str, optional): Column name for stratified splitting by class.
        seed (int): Random seed for reproducibility.
        show (bool): Whether to display visualization.
        save_path (str, optional): Path to save the visualization.


    Returns:
        dict: A dictionary containing:
            - 'index': list of lists (train row indices for each fold)
            - 'indexOut': list of lists (test/val row indices for each fold)
            - 'cluster': numpy array matching len(df) indicating the val fold affiliation.
    """
    # 1. Load metadata dataframe
    df = pd.read_csv(df_path)

    # 2. Extract cube IDs from valid paths and filter the dataframe to only include these cubes
    valid_cube_ids = []
    for path in valid_zarrs_paths:
        filename = os.path.basename(path)
        # Extrahiert die ID (z.B. 2020-0218-GTM)
        cube_id = filename.split("_postprocessed")[0].split(".zarr")[0]
        valid_cube_ids.append(cube_id)

    # 3. Filter Dataframe
    initial_len = len(df)
    df = df[df["DisNo."].isin(valid_cube_ids)].copy()
    df = df.reset_index(drop=True)

    id_to_path = {cid: path for cid, path in zip(valid_cube_ids, valid_zarrs_paths)}
    df["full_path"] = df["DisNo."].map(id_to_path)

    print(
        f"📊 Filtering: {len(df)}/{initial_len} rows kept based on valid_zarrs_paths."
    )

    # 2. Define group ids for plotting
    if spacevar and timevar:
        df["group"] = df[spacevar].astype(str) + "_" + df[timevar].astype(str)
    elif spacevar:
        df["group"] = df[spacevar].astype(str)
    elif timevar:
        df["group"] = df[timevar].astype(str)
    else:
        df["group"] = "all"

    unique_groups = sorted(df["group"].unique())
    group_counts = df["group"].value_counts().to_dict()

    cv_results = {"folds": [], "metadata": {"total_cubes": len(df), "k": k}}
    vis_matrix = np.zeros((k, len(unique_groups)))
    annot_matrix = np.full((k, len(unique_groups)), "", dtype=object)

    # 1. Handle Class Stratification (if clusters should be distributed evenly across a class)
    if classvar is not None:
        if pd.api.types.is_numeric_dtype(df[classvar]):
            raise ValueError("Argument 'classvar' only works for categorical data")

        unit = df[[spacevar, classvar]].drop_duplicates().reset_index(drop=True)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

        unit["CAST_fold"] = -1
        for fold_idx, (_, test_idx) in enumerate(skf.split(unit, unit[classvar])):
            unit.loc[test_idx, "CAST_fold"] = fold_idx

        # Merge back to assign the folds
        df = df.merge(unit, on=[spacevar, classvar], how="left")
        spacevar = "CAST_fold"

    # 2. Adjust 'k' if it's larger than available unique units
    if spacevar is not None:
        unique_space = df[spacevar].unique()
        if k > len(unique_space):
            k = len(unique_space)
            print(
                f"Warning: k is higher than number of unique locations. k is set to {k}"
            )

    if timevar is not None:
        unique_time = df[timevar].unique()
        if k > len(unique_time):
            k = len(unique_time)
            print(
                f"Warning: k is higher than number of unique points in time. k is set to {k}"
            )

    # 3. Split unique space and time units into K folds
    spacefolds = []
    if spacevar is not None:
        kf_space = KFold(n_splits=k, shuffle=True, random_state=seed)
        unique_space = df[spacevar].unique()
        for _, test_idx in kf_space.split(unique_space):
            spacefolds.append(unique_space[test_idx])

    timefolds = []
    if timevar is not None:
        kf_time = KFold(n_splits=k, shuffle=True, random_state=seed)
        unique_time = df[timevar].unique()
        for _, test_idx in kf_time.split(unique_time):
            timefolds.append(unique_time[test_idx])

    # 4. Combine Space and Time Folds
    for i in range(k):
        if timevar is not None and spacevar is not None:
            # LLTO (Leave-Location-and-Time-Out)
            test_mask = df[spacevar].isin(spacefolds[i]) & df[timevar].isin(
                timefolds[i]
            )
            train_mask = (~df[spacevar].isin(spacefolds[i])) & (
                ~df[timevar].isin(timefolds[i])
            )

        elif spacevar is not None and timevar is None:
            # LLO (Leave-Location-Out)
            test_mask = df[spacevar].isin(spacefolds[i])
            train_mask = ~df[spacevar].isin(spacefolds[i])

        elif timevar is not None and spacevar is None:
            # LTO (Leave-Time-Out)
            test_mask = df[timevar].isin(timefolds[i])
            train_mask = ~df[timevar].isin(timefolds[i])

        for g_idx, grp in enumerate(unique_groups):
            # Zähle Cubes dieser Gruppe in den jeweiligen Masken
            group_mask = df["group"] == grp
            train_count = len(df[train_mask & group_mask])
            val_count = len(df[test_mask & group_mask])

            if val_count > 0:
                vis_matrix[i, g_idx] = 2  # Color: Red
                annot_matrix[i, g_idx] = str(val_count)
            elif train_count > 0:
                vis_matrix[i, g_idx] = 1  # Color: Blue
                annot_matrix[i, g_idx] = str(train_count)
            else:
                vis_matrix[i, g_idx] = 0  # Color: Grey
                total_grp_count = len(df[df["group"] == grp])
                annot_matrix[i, g_idx] = f"({total_grp_count})"

        train_paths = df[train_mask]["full_path"].tolist()
        val_paths = df[test_mask]["full_path"].tolist()

        cv_results["folds"].append(
            {
                "fold": i,
                "num_train": len(train_paths),
                "num_val": len(val_paths),
                "train_files": train_paths,
                "val_files": val_paths,
            }
        )

    if show:
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            _plot_meyer_strategy(
                vis_matrix,
                annot_matrix,
                unique_groups,
                group_counts,
                k,
                cv_results,
                save_path=save_path,
            )
        else:
            _plot_meyer_strategy(
                vis_matrix, annot_matrix, unique_groups, group_counts, k, cv_results
            )

    return cv_results


def _plot_meyer_strategy(
    vis_matrix, annot_matrix, groups, counts, k, cv_results, save_path=None
):
    """
    Generates the Meyer et al. CV plot using pre-calculated stats from cv_results.
    """
    plt.figure(figsize=(22, 10))

    cmap = ["#ecf0f1", "#3498db", "#e74c3c"]

    # Extract the pre-calculated numbers from the cv_results dictionary
    # We build the labels for the right-hand side here
    fold_labels = []
    total_cubes = cv_results["metadata"]["total_cubes"]

    for i in range(k):
        f_data = cv_results["folds"][i]
        n_train = f_data["num_train"]
        n_val = f_data["num_val"]
        # Excluded is whatever is left from the total
        n_excl = total_cubes - n_train - n_val

        fold_labels.append(
            f"Fold {i}\n" f"Train: {n_train}\n" f"Val: {n_val}\n" f"Excl: {n_excl}"
        )

    # Create the heatmap
    ax = sns.heatmap(
        vis_matrix,
        cmap=cmap,
        annot=annot_matrix,
        fmt="",
        vmin=0,
        vmax=2,
        cbar=False,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 10, "weight": "bold"},
    )

    # Title
    plt.title(
        "Spatio-Temporal CV Strategy (Meyer et al. 2018)\n"
        "Blue = Training | Red = Validation | Grey = Excluded (Leakage Protection)",
        fontsize=16,
        pad=25,
    )

    # X-axis: Spatio-temporal groups
    plt.xticks(
        np.arange(len(groups)) + 0.5,
        [f"{g}\n(n={counts[g]})" for g in groups],
        rotation=90,
        fontsize=9,
    )

    # Y-axis (Right side): Stats from cv_results
    ax.yaxis.tick_right()
    plt.yticks(
        np.arange(k) + 0.5, fold_labels, rotation=0, fontsize=10, fontweight="bold"
    )
    ax.yaxis.set_label_position("right")

    plt.xlabel("Spatio-Temporal Groups (Climate_Season)", fontsize=12, labelpad=15)

    # Legend
    legend_elements = [
        Patch(facecolor="#3498db", label="Used for Training"),
        Patch(facecolor="#e74c3c", label="Used for Validation"),
        Patch(facecolor="#ecf0f1", label="Excluded (Leakage Protection)"),
    ]
    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(-0.18, 1))

    plt.tight_layout()

    if save_path:
        out_file = os.path.join(save_path, "cv_meyer_final_plot.png")
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        print(f"✅ Final CV Strategy Plot saved to: {out_file}")
    else:
        plt.show()

    plt.close()


### OLD LOGIC


def assert_disjoint_val_sets(cv_splits):
    val_sets = [set(val) for _, val in cv_splits]

    # Pairwise overlap check
    for i in range(len(val_sets)):
        for j in range(i + 1, len(val_sets)):
            overlap = val_sets[i] & val_sets[j]
            if overlap:
                raise ValueError(
                    f"Val overlap between fold {i} and {j}: {len(overlap)} files"
                )

    print("OK: All validation sets are pairwise disjoint.")


# --- Implementation of Leave-Time-and-Region-Out CV ---
def get_llto_splits(
    valid_paths, csv_path="train_test_split.csv", k=3, show=False, exclude_list=None
):
    """
    Implements k-fold Leave-Location-and-Time-Out (LLTO) splitting according to
    https://doi.org/10.1016/j.envsoft.2017.12.001.
    Groups data by Koppen-Geiger region (Location) and Phenological Season (Time)
    and ensures that Region+Season group are never split between train and val.

    Args:
        root_dir: Directory containing the .zarr cubes.
        csv_path: Path to the metadata CSV.
        k: Number of folds (e.g., 3, 4, or 5).

    Returns:
        List of tuples: [(train_paths, val_paths), ...] for k folds.
    """
    # 1. Load and filter only the training split
    df = pd.read_csv(csv_path)
    df = df[df["split"] == "train"].copy()

    # 2. Map file paths from the provided list of valid paths
    processed_files = {}
    for path in valid_paths:
        filename = os.path.basename(path)
        cube_id = filename.replace("_postprocessed.zarr", "").replace(".zarr", "")
        processed_files[cube_id] = path

    df["full_path"] = df["DisNo."].map(processed_files)

    # Drop entries not in valid_paths
    initial_count = len(df)
    df = df.dropna(subset=["full_path"])
    print(
        f"Matched {len(df)}/{initial_count} cubes from CSV to the provided valid paths."
    )

    # --- Filter excluded cubes before splitting ---
    # --- Just as backup ---
    if exclude_list is not None:
        if isinstance(exclude_list, str):
            df_ex = pd.read_csv(exclude_list)
            excluded_ids = set(df_ex["cube_id"].astype(str).tolist())
        else:
            excluded_ids = set(exclude_list)

        before_count = len(df)
        df = df[~df["DisNo."].isin(excluded_ids)]
        print(
            f"Filter (Pre-Split): {before_count} -> {len(df)} Cubes (Removed {before_count - len(df)} bad cubes)."
        )

    # 3. Create the unique 'Location-Time' Group ID
    df["llto_group"] = (
        df["koppen_geiger"].astype(str) + "_" + df["pheno_season_name"].astype(str)
    )
    unique_groups = sorted(df["llto_group"].unique())
    group_to_idx = {grp: i for i, grp in enumerate(unique_groups)}

    # 4. Initialize GroupKFold
    gkf = GroupKFold(n_splits=k)
    cv_splits = []
    vis_matrix = np.zeros((k, len(unique_groups)))

    indices = np.arange(len(df))
    groups = df["llto_group"].values

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        cv_splits.append((train_df["full_path"].tolist(), val_df["full_path"].tolist()))

        # Fill visualization matrix: 1 for Train (Blue), 2 for Val (Red)
        for grp in train_df["llto_group"].unique():
            vis_matrix[fold_idx, group_to_idx[grp]] = 1
        for grp in val_df["llto_group"].unique():
            vis_matrix[fold_idx, group_to_idx[grp]] = 2

        print(f"Fold {fold_idx}:")
        print(
            f"  > Training:   {len(train_df):3d} cubes ({(len(train_df)/len(df)*100):5.1f}%)"
        )
        print(
            f"  > Validation: {len(val_df):3d} cubes ({(len(val_df)/len(df)*100):5.1f}%)"
        )
        print(
            f"  > Excluded:   {(len(df) - len(train_df) - len(val_df)):3d} cubes ({(len(df) - len(train_df) - len(val_df))/len(df)*100:5.1f}%) [Blocked due to Loc/Time overlap]"
        )

        overlap = len(
            set(train_df["llto_group"]).intersection(set(val_df["llto_group"]))
        )
        print(f"  > Group Overlap between Train/Val: {overlap} (Must be 0!)")

    print(f"Successfully created {k} folds using LLTO GroupKFold.")

    if show:
        _plot_llto_strategy(vis_matrix, unique_groups, k)

    return cv_splits


def get_llto_splits_strict(
    valid_paths,
    csv_path="train_test_split.csv",
    k=3,
    min_val_ratio=0.15,
    show=False,
    save_path=None,
    exclude_list=None,
):
    """
    Implements LLTO-CV ensuring a minimum validation size while reporting data usage.

    This strategy strictly follows the principle of leaving out both locations
    and time units to test model generalization.

    Args:
        root_dir (str): Directory containing the 1000x1000 .zarr cubes.
        csv_path (str): Metadata file path.
        k (int): Number of folds.
        min_val_ratio (float): Minimum percentage of total cubes for validation (e.g., 0.15).
        show (bool): If True, generates the strategy visualization plot.
        exclude_list (set or list): List of excluded cube IDs.
    Returns:
        list: [(train_paths, val_paths), ...] for k folds.
    """
    # 1. Loading and Path Mapping
    df = pd.read_csv(csv_path)
    df = df[df["split"] == "train"].copy()

    # 2. Map file paths from the provided list of valid paths
    processed_files = {}
    for path in valid_paths:
        filename = os.path.basename(path)
        cube_id = filename.replace("_postprocessed.zarr", "").replace(".zarr", "")
        processed_files[cube_id] = path

    df["full_path"] = df["DisNo."].map(processed_files)

    # Cubes that are not in vaid_paths are dropped (as they cannot be used for training/validation)
    initial_count = len(df)
    df = df.dropna(subset=["full_path"])
    print(
        f"Matched {len(df)}/{initial_count} cubes from CSV to the provided valid paths."
    )

    # --- Filter bad cubes before splitting (optionally - should be done beforehand) ---
    if exclude_list is not None:
        if isinstance(exclude_list, str):
            df_ex = pd.read_csv(exclude_list)
            excluded_ids = set(df_ex["cube_id"].astype(str).tolist())
        else:
            excluded_ids = set(exclude_list)

        before_count = len(df)
        df = df[~df["DisNo."].isin(excluded_ids)]
        print(f"Filter (Pre-Split Strict): {before_count} -> {len(df)} Cubes.")

    total_cubes = len(df)
    min_val_cubes = int(total_cubes * min_val_ratio)

    # 3. Create the unique 'Location-Time' Group ID
    df["llto_group"] = (
        df["koppen_geiger"].astype(str) + "_" + df["pheno_season_name"].astype(str)
    )
    unique_groups = sorted(df["llto_group"].unique())
    group_to_idx = {grp: i for i, grp in enumerate(unique_groups)}

    all_locations = df["koppen_geiger"].unique()
    all_seasons = df["pheno_season_name"].unique()

    cv_splits = []

    # vis_matrix (0=Excluded, 1=Train, 2=Val)
    vis_matrix = np.zeros((k, len(unique_groups)))

    print(f"--- LLTO-CV Initialization (Total Cubes: {total_cubes}) ---")

    for fold_idx in range(k):
        found_valid_split = False
        attempts = 0

        while not found_valid_split and attempts < 500:
            attempts += 1
            # Randomly sample Location and Season combinations
            sel_locs = random.sample(
                list(all_locations), max(1, len(all_locations) // 4)
            )
            sel_seasons = random.sample(
                list(all_seasons), max(1, len(all_seasons) // 3)
            )

            # Validation: Intersection of selected Location AND Season [cite: 209]
            val_mask = df["koppen_geiger"].isin(sel_locs) & df[
                "pheno_season_name"
            ].isin(sel_seasons)
            val_df = df[val_mask]

            if len(val_df) >= min_val_cubes:
                # Training: Must share NEITHER Location NOR Season with Validation [cite: 130, 209]
                train_mask = (~df["koppen_geiger"].isin(sel_locs)) & (
                    ~df["pheno_season_name"].isin(sel_seasons)
                )
                train_df = df[train_mask]

                if (
                    len(train_df) > total_cubes * 0.1
                ):  # Ensure at least some training data remains
                    found_valid_split = True

        if not found_valid_split:
            raise RuntimeError(
                f"Could not find a valid LLTO split for fold {fold_idx} with current constraints."
            )

        # Calculate Percentages
        train_pct = (len(train_df) / total_cubes) * 100
        val_pct = (len(val_df) / total_cubes) * 100
        excluded_pct = 100 - train_pct - val_pct

        print(f"Fold {fold_idx}:")
        print(f"  > Training:   {len(train_df):3d} cubes ({train_pct:5.1f}%)")
        print(f"  > Validation: {len(val_df):3d} cubes ({val_pct:5.1f}%)")
        print(
            f"  > Excluded:   {(total_cubes - len(train_df) - len(val_df)):3d} cubes ({excluded_pct:5.1f}%) [Blocked due to Loc/Time overlap]"
        )

        cv_splits.append((train_df["full_path"].tolist(), val_df["full_path"].tolist()))

        # 3. Visualization Matrix Update
        for grp in train_df["llto_group"].unique():
            vis_matrix[fold_idx, group_to_idx[grp]] = 1
        for grp in val_df["llto_group"].unique():
            vis_matrix[fold_idx, group_to_idx[grp]] = 2

    if show:
        _plot_llto_strategy(vis_matrix, unique_groups, k, save_path=save_path)

    assert_disjoint_val_sets(cv_splits)

    return cv_splits


def _plot_llto_strategy(vis_matrix, groups, k, save_path=None):
    """Generates a target-oriented cross-validation matrix plot."""
    plt.figure(figsize=(16, 7))
    cmap = ["#bdc3c7", "#3498db", "#e74c3c"]  # Grey, Blue, Red
    sns.heatmap(
        vis_matrix,
        annot=False,
        cbar=False,
        cmap=cmap,
        vmin=0,
        vmax=2,
        linewidths=1,
        linecolor="white",
    )

    plt.title(
        f"Target-Oriented LLTO-CV Strategy ({k} Folds)\n"
        f"Testing Generalization on Unknown Locations and Time Steps"
    )
    plt.xlabel("Spatio-Temporal Groups (KoppenGeiger_Season)")
    plt.ylabel("Fold Index")

    legend_elements = [
        Patch(facecolor="#3498db", label="Used for Training"),
        Patch(facecolor="#e74c3c", label="Used for Validation"),
        Patch(facecolor="#bdc3c7", label="Excluded (Spatial/Temporal Overlap)"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.25, 1))

    if len(groups) < 40:
        plt.xticks(np.arange(len(groups)) + 0.5, groups, rotation=90, fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        out_path = os.path.join(save_path, "cv_splits_plot.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"✅ CV Split Plot gespeichert: {out_path}")
    else:
        plt.show()

    plt.close()
