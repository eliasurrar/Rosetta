#%%
# Load the necessary libraries
import os
import pandas as pd
import numpy as np


DATA_PATH_REACTORS = "/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/reactors/dataset_per_sample_reactor_complete.csv"
DATA_PATH_COLUMNS = '/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/columns/df_model_catcontrol_projects_with_reactors_fit.csv'
PROJECT_ROOT = "/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Rosetta"
DB_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "database_ready")
os.makedirs(DB_OUTPUT_DIR, exist_ok=True)


def _append_filter_snapshot(df, stage, snapshots):
    if snapshots is None:
        return
    snapshot = df.copy()
    snapshot["filter_stage"] = stage
    snapshots.append(snapshot)

def _ensure_columns(df, columns, default_value=np.nan):
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = default_value
    return df


def apply_column_filters(df, snapshots=None):
    df = df.copy()
    df = _ensure_columns(df, ["feed_mass_kg", "column_inner_diameter_m"])
    _append_filter_snapshot(df, "columns_raw", snapshots)

    # Separate UGM2 samples for ROM and Crushed
    df.loc[
        df["project_col_id"].str.startswith("jetti_file_elephant_ii_ver_2_ugm_ur"),
        "project_sample_id",
    ] = "jetti_file_elephant_ii_ugm2_coarse"
    _append_filter_snapshot(df, "columns_ugm2_sample_id_adjusted", snapshots)

    # REMOVE PVO1 TO AVOID WEIRD BEHAVIOURS
    df = df[df["project_col_id"] != "006_jetti_project_file_pvo1"]
    _append_filter_snapshot(df, "columns_remove_pvo1", snapshots)

    # REMOVE WHOLE PVLS BECAUSE IT IS RESIDUES LEACHED???
    df = df[df["project_sample_id"] != "006_jetti_project_file_pvls"]
    _append_filter_snapshot(df, "columns_remove_pvls", snapshots)

    # Remove Project 022 < 7.0 recovery
    df = df[
        ~(
            (df["project_sample_id"].str.startswith("022_jetti_project_file_"))
            & (df["cu_recovery_%"] < 7.0)
        )
    ]
    _append_filter_snapshot(df, "columns_remove_022_low_recovery", snapshots)

    # Remove supergene sample
    df = df[df["project_sample_id"] != "020_jetti_project_file_hypogene_supergene_super"]
    _append_filter_snapshot(df, "columns_remove_supergene", snapshots)

    # Calculate Fe/Cu ratios and other new columns
    df["fe:cu"] = df["feed_head_fe_%"] / df["feed_head_cu_%"]
    df["cu:fe"] = df["feed_head_cu_%"] / df["feed_head_fe_%"]
    df["copper_primary_sulfides_equivalent"] = df["cu_%"] * df["residual_cpy_%"] / 100
    df["copper_secondary_sulfides_equivalent"] = df["cu_%"] * df["cyanide_soluble_%"] / 100
    df["copper_sulfides_equivalent"] = df["cu_%"] * (df["residual_cpy_%"] + df["cyanide_soluble_%"]) / 100
    df["copper_oxides_equivalent"] = df["cu_%"] * df["acid_soluble_%"] / 100

    return df


def apply_reactor_filters(df):
    df = df.copy()

    # Duplicate and append rows for '007ajettiprojectfile_elephant_ugm2_rthead'
    duplicated_rows = df[df["project_sample_id"] == "007ajettiprojectfile_elephant_ugm2_rthead"].copy()
    duplicated_rows["project_sample_id"] = "007ajettiprojectfile_elephant_ugm2_rthead_coarse"
    df = pd.concat([df, duplicated_rows], ignore_index=True)

    # Duplicate and append rows for '007jettiprojectfile_elephant'
    duplicated_rows = df[df["project_sample_id"] == "007jettiprojectfile_elephant"].copy()
    duplicated_rows["project_sample_id"] = "007jettiprojectfile_elephant_site"
    df = pd.concat([df, duplicated_rows], ignore_index=True)

    # Duplicate and append rows for '015jettiprojectfile_pv'
    duplicated_rows = df[df["project_sample_id"] == "015jettiprojectfile_pv"].copy()
    duplicated_rows["project_sample_id"] = "015jettiprojectfile_pv_6in"
    df = pd.concat([df, duplicated_rows], ignore_index=True)
    duplicated_rows["project_sample_id"] = "015jettiprojectfile_pv_8in"
    df = pd.concat([df, duplicated_rows], ignore_index=True)

    # Fill NaN temperatures with 25.0
    df["temp_(c)_mean"] = df["temp_(c)_mean"].fillna(25.0)
    # Filter out temperature above 40 degrees on reactors
    df = df[df["temp_(c)_mean"] <= 40.0]

    return df


DYNAMIC_ARRAY_COLUMNS = [
    "leach_duration_days",
    "cu_recovery_%",
    "cumulative_catalyst_addition_kg_t",
    "feed_orp_mv_ag_agcl",
    "feed_flowrate_ml_min",
    "irrigation_rate_l_h_m2",
    "raff_assay_fe_ii_mg_l",
    "raff_assay_fe_iii_mg_l",
    "pls_fe_ii_mg_l",
    "pls_fe_iii_mg_l",
    "cumulative_lixiviant_m3_t",
]


def _present_dynamic_array_cols(df, extra_cols=None):
    cols = [col for col in DYNAMIC_ARRAY_COLUMNS if col in df.columns]
    for col in extra_cols or []:
        if col in df.columns and col not in cols:
            cols.append(col)
    return cols


def _fill_irrigation_rate_array_fallback(df):
    df = df.copy()
    if "irrigation_rate_l_m2_h" not in df.columns:
        return df

    static_irrigation = pd.to_numeric(df["irrigation_rate_l_m2_h"], errors="coerce")
    if "irrigation_rate_l_h_m2" not in df.columns:
        df["irrigation_rate_l_h_m2"] = static_irrigation
    else:
        array_like_irrigation = pd.to_numeric(df["irrigation_rate_l_h_m2"], errors="coerce")
        df["irrigation_rate_l_h_m2"] = array_like_irrigation.fillna(static_irrigation)

    return df

#%%

# ---------------------------
# Load Data
# ---------------------------
# df_model_recCu_catalyzed_projects = pd.read_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/columns/df_model_catalyzed_projects_with_reactors_fit.csv', sep=',')
# df_model_recCu_control_projects = pd.read_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/columns/df_model_control_projects_with_reactors_fit.csv', sep=',')
df_model_recCu_catcontrol_projects_raw = pd.read_csv(DATA_PATH_COLUMNS, sep=',')
df_reactors_raw = pd.read_csv(DATA_PATH_REACTORS, sep=',')
df_model_recCu_catcontrol_projects = apply_column_filters(df_model_recCu_catcontrol_projects_raw)
df_reactors = apply_reactor_filters(df_reactors_raw)

cols_to_check = ['project_sample_id', 'start_cell', 'temp_(c)_mean', 'catalyst_type']
df_reactors[df_reactors['project_sample_id'] == '017jettiprojectfile_ea'][cols_to_check]


# Filter to target lixiviant/ph/catalyst subset before analysis
lixiviant_filter = ["Inoculum"]
ph_target_filter = ["1.5", "1.7", "2.0", "2.3"]
catalyst_filter = ["Control", "100-CA"]
catalyst_dose_filter = [0, 18] #, 36, 54] # have to run analysis to make sure more catalyst does not have more recovery
real_ph_filter = 2.1

required_cols = [
    "project_sample_id",
    "start_cell",
    "lixiviant",
    "ph_target",
    "catalyst_type",
    "final_cu_extraction",
    "a1_param",
    "b1_param",
    "a2_param",
    "b2_param",
]
missing_cols = [col for col in required_cols if col not in df_reactors.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df_reactors_filtered = df_reactors[
    df_reactors["lixiviant"].isin(lixiviant_filter)
    & df_reactors["ph_target"].astype(str).isin(ph_target_filter)
    & df_reactors["catalyst_type"].isin(catalyst_filter)
    & df_reactors["catalyst_dose_(mg_l)"].isin(catalyst_dose_filter)
    & (df_reactors["ph_mean"] <= real_ph_filter)
    & (df_reactors["final_cu_extraction"] <= 105) # assume 5% error
].copy()

df_reactors_filtered["project_col_id"] = np.where(
    df_reactors_filtered["catalyst_type"] == "100-CA",
    "Catalyzed",
    "Control",
)

# Force numeric columns
numeric_cols = ["temp_(c)_mean", "ph_mean", "final_cu_extraction", "a1_param", "b1_param", "a2_param", "b2_param",
                "cu_%", "acid_soluble_%", "cyanide_soluble_%", "residue_cpy_%", "bornite", "enargite", "pyrite", "chlorite", "fe_oxides",
                "feed_head_cu_%", "feed_head_fe_%", "material_size_p80_in", "transition_time", "catalyst_saturation_inside_column_day",
                "column_height_m", "column_inner_diameter_m", "feed_mass_kg", "catalyst_dose_(mg_l)", "irrigation_rate_l_m2_h"]

for col in [col for col in numeric_cols if col in df_reactors_filtered.columns]:
    df_reactors_filtered[col] = pd.to_numeric(df_reactors_filtered[col], errors='coerce')

#%%


def process_arrays_by_weekly_intervals(
    arr,
    leach_days,
    col_name,
    test_ids=None,
    singleton_drop_threshold=0.8,
    min_tests_for_singleton_drop=2,
    interpolate_nans=True,
    weekly_days=None,
    return_full_weekly=False,
):
    """
    Process a single array of time-varying data by averaging values within strict 7-day batches
    (0-7, 7-14, 14-21, ...), taking the maximum of averaged values, skipping empty batches,
    and interpolating to estimate the maximum for each batch using subsequent batches if needed.
    
    Parameters:
    - arr: Array of values for the time-varying column (e.g., cu_recovery_%)
    - leach_days: Array of corresponding leach_duration_days
    - col_name: Name of the column (to determine if it's leach_duration_days)
    - test_ids: Optional array of test identifiers aligned to arr/leach_days
    - singleton_drop_threshold: If fraction of weekly bins with >= min_tests_for_singleton_drop exceeds this,
                                weekly bins with only 1 test are set to NaN (non-leach columns only)
    - min_tests_for_singleton_drop: Minimum unique tests in a weekly bin to be considered "multi-test"
    - interpolate_nans: If True, interpolate NaNs after processing
    
    Returns:
    - Array of maximum values for each non-empty weekly batch
    """
    arr = np.asarray(arr, dtype=float)
    leach_days = np.asarray(leach_days, dtype=float)

    if arr.size == 0 or leach_days.size == 0:
        return np.array([])

    # Drop non-finite leach_days; keep arr (NaNs handled per-bin)
    finite_mask = np.isfinite(leach_days)
    arr = arr[finite_mask]
    leach_days = leach_days[finite_mask]
    if test_ids is not None:
        test_ids = np.asarray(test_ids)
        if test_ids.shape[0] == finite_mask.shape[0]:
            test_ids = test_ids[finite_mask]
        else:
            test_ids = None
    if arr.size == 0 or leach_days.size == 0:
        return np.array([])

    # Sort by leach_days for stable batching
    order = np.argsort(leach_days)
    arr = arr[order]
    leach_days = leach_days[order]
    
    # Define weekly batch endpoints (7, 14, 21, ...)
    if weekly_days is None:
        max_days = leach_days.max()
        weekly_days = np.arange(7, max_days + 7, 7)  # [7, 14, 21, ...]
    else:
        weekly_days = np.asarray(weekly_days, dtype=float)
    
    processed = []
    weekly_counts = []
    for end_day in weekly_days:
        # Define batch range: [start_day, end_day)
        start_day = end_day - 7
        # Find values within the batch [start_day, end_day)
        idx = np.where((leach_days >= start_day) & (leach_days < end_day))[0]
        if len(idx) > 0:  # Only process non-empty batches
            if col_name == 'leach_duration_days':
                processed.append(end_day)  # Use batch endpoint for leach_duration_days
            else:
                # Average values within the batch (ignore NaNs)
                batch_vals = arr[idx]
                avg_value = np.nanmean(batch_vals) if batch_vals.size > 0 else np.nan
                processed.append(avg_value)
            if test_ids is not None:
                if col_name == 'leach_duration_days':
                    weekly_counts.append(len(np.unique(test_ids[idx])))
                else:
                    valid_idx = idx[np.isfinite(arr[idx])]
                    weekly_counts.append(len(np.unique(test_ids[valid_idx])) if valid_idx.size > 0 else 0)
            else:
                weekly_counts.append(len(idx))
        elif return_full_weekly:
            if col_name == 'leach_duration_days':
                processed.append(end_day)
            else:
                processed.append(np.nan)
            weekly_counts.append(0)

    if not processed:  # If no non-empty batches, return empty array
        return np.array([])
    
    processed = np.array(processed, dtype=float)

    if (
        test_ids is not None
        and col_name != 'leach_duration_days'
        and len(weekly_counts) == len(processed)
        and len(weekly_counts) > 0
    ):
        weekly_counts = np.array(weekly_counts, dtype=float)
        frac_multi = np.mean(weekly_counts >= min_tests_for_singleton_drop)
        if frac_multi > singleton_drop_threshold:
            processed[weekly_counts < min_tests_for_singleton_drop] = np.nan
    
    # Interpolate NaNs and estimate maximum for each batch
    if interpolate_nans and np.any(np.isnan(processed)):
        x = np.arange(len(processed))
        valid = ~np.isnan(processed)
        if valid.sum() > 1:  # Need at least 2 points for interpolation
            # Interpolate NaN values
            interpolated = np.interp(x, x[valid], processed[valid])
            # Estimate maximum by considering interpolated values and subsequent batches
            processed = np.maximum.accumulate(interpolated)  # Forward-fill max values
        else:
            # Fallback: use mean of non-NaN values or 0 if all NaN
            processed = np.where(np.isnan(processed), 
                               np.mean(processed[valid]) if valid.any() else np.nan, 
                               processed)
    
    return processed

def prepare_column_train_data(
    df,
    config=None,
    output_type='original',
    fill_noncat_averages=False,
    return_df_only=False,
    dropna_subset=True,
    drop_nan_rows=True,
    filter_snapshots=None,
):
    """
    Prepare column train data with option to select original or split grouping.
    
    Parameters:
    - df: Input DataFrame
    - config: Configuration dictionary
    - output_type: 'original' for project_col_id grouping or 'averaged' for project_sample_id with catalyst split
    - return_df_only: If True, return only the processed DataFrame (no tensors/scalers)
    - dropna_subset: If True, drop rows with NaNs in required time-varying columns before grouping
    - drop_nan_rows: If True, drop rows with NaNs in numeric features after grouping
    - filter_snapshots: Optional list; if provided, appends intermediate DataFrames with filter_stage labels
    
    Returns:
    - If return_df_only is True: processed DataFrame (out_df or out_df_averaged)
    - Otherwise: Tuple containing (X_tensor, Y_tensor, time_tensor, catalyst_tensor, transition_time_tensor,
                                  sample_ids, sample_col_ids, feature_weights, numeric_cols, scaler_X, out_df)
    """
    if output_type not in ['original', 'averaged']:
        raise ValueError("output_type must be 'original' or 'averaged'")

    if config is None:
        config = {}

    df = df.copy()
    df = _fill_irrigation_rate_array_fallback(df)
    id_col = 'project_sample_id'
    col_idx = 'project_col_id'

    # Define time-varying columns
    special = config.get('special_feats', {})
    target_feats = list(special.get('target_feat', []))
    target_col = target_feats[0] if target_feats else 'cu_recovery_%'
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' in input DataFrame.")

    _append_filter_snapshot(df, "prepare_pre_target_filter", filter_snapshots)
    df = df[df[target_col] > 1.0].copy()
    _append_filter_snapshot(df, "prepare_post_target_filter", filter_snapshots)

    # Active time feature for this run
    time_feat_list = list(special.get('time_feat', []))
    time_feat_col = time_feat_list[0] if time_feat_list else 'leach_duration_days'
    leach_days = 'leach_duration_days'
    include_lixiviant = time_feat_col == 'cumulative_lixiviant_m3_t'
    if leach_days not in df.columns:
        raise ValueError(f"Missing required time column '{leach_days}' in input DataFrame.")

    # Keep requested leaching signals as arrays through the whole batching/export pipeline.
    time_varying_cols = _present_dynamic_array_cols(
        df,
        extra_cols=[target_col, leach_days, 'cumulative_catalyst_addition_kg_t', time_feat_col],
    )

    catalyst_feats = list(special.get('catalyst_feat', []))
    catalyst_col = catalyst_feats[0] if catalyst_feats else 'cumulative_catalyst_addition_kg_t'
    if catalyst_col not in df.columns:
        raise ValueError(f"Missing catalyst column '{catalyst_col}' in input DataFrame.")

    # Filter rows with NaNs in key columns (only those relevant for this run)
    subset_cols = [target_col, leach_days, 'cumulative_catalyst_addition_kg_t']
    if include_lixiviant:
        subset_cols.append('cumulative_lixiviant_m3_t')
    if dropna_subset:
        df_filtered = df.dropna(subset=subset_cols).copy()
        _append_filter_snapshot(df_filtered, "prepare_post_subset_dropna", filter_snapshots)
    else:
        df_filtered = df.copy()

    # Original grouping by project_col_id (unchanged)
    grouped_data = []
    for col_id, group in df_filtered.groupby(col_idx):
        row = {}
        for col in df.columns:
            if col in time_varying_cols:
                row[col] = group[col].values.astype(float)
            else:
                row[col] = group[col].iloc[0]
        grouped_data.append(row)
    out_df = pd.DataFrame(grouped_data, columns=df.columns)

    # Split grouping by project_sample_id with catalyst split
    grouped_data_averaged = []
    for sample_id, group in df_filtered.groupby(id_col):
        # Split into catalyst = 0 and catalyst > 0
        no_catalyst = group[group[catalyst_col] == 0]
        with_catalyst = group[group[catalyst_col] > 0]

        # Process no-catalyst data
        if not no_catalyst.empty:
            row_no = {'catalyst_status': 'no_catalyst'}
            for col in df.columns:
                if col in time_varying_cols:
                    row_no[col] = no_catalyst[col].values.astype(float)
                else:
                    if col.endswith('_true'):
                        row_no[col] = np.median(no_catalyst[col].values.astype(float))
                    else:
                        row_no[col] = no_catalyst[col].iloc[0]
            row_no[id_col] = sample_id
            grouped_data_averaged.append(row_no)
        
        # Process with-catalyst data
        if not with_catalyst.empty:
            row_with = {'catalyst_status': 'with_catalyst'}
            for col in df.columns:
                if col in time_varying_cols:
                    row_with[col] = with_catalyst[col].values.astype(float)
                else:
                    if col.endswith('_true'):
                        row_with[col] = np.median(with_catalyst[col].values.astype(float))
                    else:
                        row_with[col] = with_catalyst[col].iloc[0]
            row_with[id_col] = sample_id
            grouped_data_averaged.append(row_with)
    
    out_df_averaged = pd.DataFrame(grouped_data_averaged, columns=[*df.columns, 'catalyst_status'])
    
    # Rename project_col_id in split DataFrame
    out_df_averaged[col_idx] = np.where(out_df_averaged['catalyst_status'] == 'no_catalyst', 'Control', 'Catalyzed')
    
    def _aggregate_weekly_by_test(df_subset, col, leach_col, test_col):
        if df_subset.empty:
            return np.array([])
        max_days = df_subset[leach_col].max()
        weekly_grid = np.arange(7, max_days + 7, 7) if np.isfinite(max_days) and max_days > 0 else np.array([])
        if col == leach_col:
            return weekly_grid

        per_test = []
        for _, g in df_subset.groupby(test_col):
            days = g[leach_col].values.astype(float)
            vals = g[col].values.astype(float)
            mask = np.isfinite(days) & np.isfinite(vals)
            if mask.sum() == 0:
                continue
            days = days[mask]
            vals = vals[mask]
            order = np.argsort(days)
            days = days[order]
            vals = vals[order]

            weekly_vals = np.full(len(weekly_grid), np.nan, dtype=float)
            for i, end_day in enumerate(weekly_grid):
                start_day = end_day - 7
                idx = (days >= start_day) & (days < end_day)
                if np.any(idx):
                    weekly_vals[i] = np.nanmean(vals[idx])

            # Interpolate missing internal weeks (no extrapolation beyond observed range)
            valid = np.isfinite(weekly_vals)
            if valid.sum() >= 2:
                weekly_vals = np.interp(
                    weekly_grid, weekly_grid[valid], weekly_vals[valid], left=np.nan, right=np.nan
                )
            per_test.append(weekly_vals)

        if not per_test:
            return np.array([])
        stacked = np.vstack(per_test)
        with np.errstate(all="ignore"):
            avg = np.nanmean(stacked, axis=0)
        return avg

    # Process time-varying columns for split grouping by weekly batches
    final_grouped_data_averaged = []
    for sample_id, group in out_df_averaged.groupby(id_col):
        no_cat_rows = group[group['catalyst_status'] == 'no_catalyst']
        with_cat_rows = group[group['catalyst_status'] == 'with_catalyst']
        
        # Define weekly intervals based on all data for this sample
        max_days = 0
        if not no_cat_rows.empty:
            max_days = max([row[leach_days].max() for _, row in no_cat_rows.iterrows() if len(row['leach_duration_days']) > 0])
        if not with_cat_rows.empty:
            max_days = max(max_days, max([row[leach_days].max() for _, row in with_cat_rows.iterrows() if len(row['leach_duration_days']) > 0]))
        
        # Process no-catalyst data
        no_cat_processed = {}
        sample_raw = df_filtered[df_filtered[id_col] == sample_id]
        no_cat_raw = sample_raw[sample_raw[catalyst_col] == 0]
        with_cat_raw = sample_raw[sample_raw[catalyst_col] > 0]
        if not no_cat_rows.empty and not no_cat_raw.empty:
            row_no = {'catalyst_status': 'no_catalyst', id_col: sample_id}
            for col in time_varying_cols:
                no_cat_processed[col] = _aggregate_weekly_by_test(
                    no_cat_raw, col, leach_days, col_idx
                )
            # Skip if no non-empty batches
            if len(no_cat_processed[leach_days]) > 0:
                row_no.update(no_cat_processed)
                for col in df.columns:
                    if col not in time_varying_cols:
                        if col.endswith('_true'):
                            row_no[col] = np.median(no_cat_rows[col].values.astype(float))
                        else:
                            row_no[col] = no_cat_rows[col].iloc[0]
                final_grouped_data_averaged.append(row_no)
        
        # Process with-catalyst data, prepending no-catalyst averages
        if not with_cat_rows.empty and not with_cat_raw.empty:
            row_with = {'catalyst_status': 'with_catalyst', id_col: sample_id}
            processed_cols = {}
            with_cat_days_processed = _aggregate_weekly_by_test(
                with_cat_raw, leach_days, leach_days, col_idx
            )
            for col in time_varying_cols:
                # Process catalyzed data
                if col == leach_days:
                    processed_with = with_cat_days_processed
                else:
                    processed_with = _aggregate_weekly_by_test(
                        with_cat_raw, col, leach_days, col_idx
                    )
                # Prepend no-catalyst data if available
                if not no_cat_rows.empty and len(no_cat_processed[leach_days]) > 0 and fill_noncat_averages:
                    no_cat_data = no_cat_processed[col]
                    no_cat_days = no_cat_processed[leach_days]
                    # Ensure no overlap in leach_duration_days
                    with_cat_start = with_cat_days_processed.min() if len(with_cat_days_processed) > 0 else np.inf
                    no_cat_idx = np.where(no_cat_days < with_cat_start)[0] if len(no_cat_days) > 0 else []
                    if len(no_cat_idx) > 0:
                        # Concatenate no-catalyst and with-catalyst data
                        if col == leach_days:
                            processed_cols[col] = np.concatenate([no_cat_days[no_cat_idx], processed_with])
                        elif col == 'cumulative_catalyst_addition_kg_t':
                            # For catalyst, use zeros for no-catalyst portion
                            zeros = np.zeros(len(no_cat_idx))
                            processed_cols[col] = np.concatenate([zeros, processed_with])
                        else:
                            processed_cols[col] = np.concatenate([no_cat_data[no_cat_idx], processed_with])
                    else:
                        processed_cols[col] = processed_with
                else:
                    processed_cols[col] = processed_with
            # Skip if no non-empty batches
            if len(processed_cols[leach_days]) > 0:
                row_with.update(processed_cols)
                for col in df.columns:
                    if col not in time_varying_cols:
                        if col.endswith('_true'):
                            row_with[col] = np.median(with_cat_rows[col].values.astype(float))
                        else:
                            row_with[col] = with_cat_rows[col].iloc[0] if not with_cat_rows[col].empty else no_cat_rows[col].iloc[0]
                final_grouped_data_averaged.append(row_with)

    # Skip if no valid rows
    if not final_grouped_data_averaged:
        print("Warning: No non-empty batches found for any samples in split grouping.")
        return None

    # Create out_df_averaged with processed values
    out_df_averaged = pd.DataFrame(final_grouped_data_averaged, columns=[*df.columns, 'catalyst_status'])

    # Compute transition_time for original grouping (unchanged)
    out_df['transition_time'] = 0.0
    for idx, row in out_df.iterrows():
        catalyst_days = row[time_feat_col][row['cumulative_catalyst_addition_kg_t'] > 0]
        out_df.at[idx, 'transition_time'] = catalyst_days.min() if catalyst_days.size > 0 else row[time_feat_col].max()
    out_df['transition_time'] = out_df['transition_time'].round(0)

    # Compute transition_time for split grouping
    out_df_averaged['transition_time'] = 0.0
    for idx, row in out_df_averaged.iterrows():
        sample_id = row[id_col]
        sample_data = df_filtered[df_filtered[id_col] == sample_id]
        catalyst_days = sample_data[time_feat_col][sample_data['cumulative_catalyst_addition_kg_t'] > 0]
        if row['catalyst_status'] == 'with_catalyst':
            # First day where catalyst > 0, or last day if no catalyst
            out_df_averaged.at[idx, 'transition_time'] = (
                catalyst_days.min() if catalyst_days.size > 0 else 
                sample_data[time_feat_col].max() if sample_data[time_feat_col].size > 0 else 0.0
            )
        else:
            # For no_catalyst, use the last day of the processed no_catalyst data
            out_df_averaged.at[idx, 'transition_time'] = (
                row[time_feat_col][-1] if len(row[time_feat_col]) > 0 else 0.0
            )
    out_df_averaged['transition_time'] = out_df_averaged['transition_time'].round(0)

    # Prepare features for training (exclude non-feature columns)
    drop_cols = ['project_sample_id_reactormatch', 'project_sample_id', 'project_col_id', 
                 time_feat_col, target_col, catalyst_col, 'catalyst_status']
    # Always drop leach_days; if active time is leach_days it's already in drop_cols via time_feat_col
    if leach_days not in drop_cols:
        drop_cols.append(leach_days)
    feature_cols = [c for c in df.columns if c not in drop_cols]
    numeric_cols = [c for c in feature_cols if np.issubdtype(out_df[c].dtype, np.number)]
    
    # Get feature weights from config
    raw_weights = config.get('column_tests_feature_weighting', {}).get('weights', {})
    feature_weights = {k: v[1] for k, v in raw_weights.items() if k in numeric_cols}
    
    # Check for NaNs in original DataFrame
    if drop_nan_rows and out_df[numeric_cols].isnull().any().any():
        nan_positions = out_df[numeric_cols].isnull().any(axis=1)
        print("Warning: NaN values found in original DataFrame rows:")
        print(out_df[nan_positions][[id_col, col_idx, *numeric_cols]])
        out_df = out_df.dropna(subset=numeric_cols)
        _append_filter_snapshot(out_df, "prepare_post_numeric_nan_drop_original", filter_snapshots)

    # Check for NaNs in split DataFrame numeric columns
    if drop_nan_rows and out_df_averaged[numeric_cols].isnull().any().any():
        nan_positions = out_df_averaged[numeric_cols].isnull().any(axis=1)
        print("Warning: NaN values found in split DataFrame rows:")
        print(out_df_averaged[nan_positions][[id_col, col_idx, *numeric_cols]])
        out_df_averaged = out_df_averaged.dropna(subset=numeric_cols)
        _append_filter_snapshot(out_df_averaged, "prepare_post_numeric_nan_drop_averaged", filter_snapshots)

    # Select output based on output_type
    if return_df_only:
        return out_df if output_type == 'original' else out_df_averaged

    # Lazy imports for training-only outputs
    from sklearn.preprocessing import StandardScaler
    import torch
    device = torch.device('cpu')

    if output_type == 'original':
        # If time_feat is different from leach_duration_days, remove leach_duration_days from numeric_cols
        if time_feat_col != leach_days and leach_days in numeric_cols:
            numeric_cols.remove(leach_days)
            
        # Scale numerical features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(out_df[numeric_cols])
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        # Convert time-varying columns and other outputs to tensors
        Y_tensor = [torch.tensor(row[target_col], dtype=torch.float32).to(device) for _, row in out_df.iterrows()]
        time_tensor = [torch.tensor(row[time_feat_col], dtype=torch.float32).to(device) for _, row in out_df.iterrows()]
        catalyst_tensor = [torch.tensor(row['cumulative_catalyst_addition_kg_t'], dtype=torch.float32).to(device) for _, row in out_df.iterrows()]
        transition_time_tensor = torch.tensor(out_df['transition_time'].values, dtype=torch.float32).to(device)
        sample_ids = out_df[id_col].values
        sample_col_ids = out_df[col_idx].values

        # Debug information
        print("Original grouping - Transition time min/max:", out_df['transition_time'].min(), out_df['transition_time'].max())
        print("Original grouping - X_tensor shape:", X_tensor.shape)
        print("Original grouping - Y_tensor length:", len(Y_tensor))
        print("Original grouping - Recovery values min and max:", min([t.min().item() for t in Y_tensor]), max([t.max().item() for t in Y_tensor]))

        return (
            X_tensor,
            Y_tensor,
            time_tensor,
            catalyst_tensor,
            transition_time_tensor,
            sample_ids,
            sample_col_ids,
            feature_weights,
            numeric_cols,
            scaler_X,
            out_df
        )
    else:  # output_type == 'averaged'
        # Scale numerical features
        # If time_feat is different from leach_duration_days, remove leach_duration_days from numeric_cols
        if time_feat_col != leach_days and leach_days in numeric_cols:
            numeric_cols.remove(leach_days)

        scaler_X_averaged = StandardScaler()
        X_scaled_averaged = scaler_X_averaged.fit_transform(out_df_averaged[numeric_cols])
        X_tensor_averaged = torch.tensor(X_scaled_averaged, dtype=torch.float32).to(device)

        # Convert time-varying columns and other outputs to tensors
        Y_tensor_averaged = [torch.tensor(row['cu_recovery_%'], dtype=torch.float32).to(device) for _, row in out_df_averaged.iterrows()]
        time_tensor_averaged = [torch.tensor(row[time_feat_col], dtype=torch.float32).to(device) for _, row in out_df_averaged.iterrows()]
        catalyst_tensor_averaged = [torch.tensor(row['cumulative_catalyst_addition_kg_t'], dtype=torch.float32).to(device) for _, row in out_df_averaged.iterrows()]
        transition_time_tensor_averaged = torch.tensor(out_df_averaged['transition_time'].values, dtype=torch.float32).to(device)
        sample_ids_averaged = out_df_averaged[id_col].values
        sample_col_ids_averaged = out_df_averaged[col_idx].values

        # Debug information
        print("Split grouping - Transition time min/max:", out_df_averaged['transition_time'].min(), out_df_averaged['transition_time'].max())
        print("Split grouping - X_tensor_averaged shape:", X_tensor_averaged.shape)
        print("Split grouping - Y_tensor_averaged length:", len(Y_tensor_averaged))
        print("Split grouping - Recovery values min and max:", min([t.min().item() for t in Y_tensor_averaged]), max([t.max().item() for t in Y_tensor_averaged]))

        return (
            X_tensor_averaged,
            Y_tensor_averaged,
            time_tensor_averaged,
            catalyst_tensor_averaged,
            transition_time_tensor_averaged,
            sample_ids_averaged,
            sample_col_ids_averaged,
            feature_weights,
            numeric_cols,
            scaler_X_averaged,
            out_df_averaged
        )


def _prepare_array_cols_for_csv(df, cols, decimals=3):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: (np.round(v, decimals).tolist() if isinstance(v, np.ndarray) else v)
            )
    return df


def _round_numeric_df(df, decimals=3):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].round(decimals)
    return df


COLUMN_CONFIG = {
    "special_feats": {
        "target_feat": ["cu_recovery_%"],
        "time_feat": ["leach_duration_days"],
        "catalyst_feat": ["cumulative_catalyst_addition_kg_t"],
    },
    "column_tests_feature_weighting": {"weights": {}},
}


#%%
# Create the dictionary of connections for samples
df_model_recCu_catcontrol_projects['project_sample_id'].unique()
df_reactors['project_sample_id'].unique()

match_dict = {
    '003_jetti_project_file_amcf_head': ['003jettiprojectfile_amcf'],
    '006_jetti_project_file_pvo': [''], 
    '007_jetti_project_file_leopard_lep': ['007jettiprojectfile_leopard'],
    '007b_jetti_project_file_tiger_tgr': ['007bjettiprojectfile_tiger_m1'],
    '011_jetti_project_file_rm': ['011jettiprojectfile_rm'],
    '011_jetti_project_file_rm_crushed': ['011jettiprojectfile_rm_crushed'],
    '014_jetti_project_file_bag': ['014jettiprojectfile_bag'],
    '014_jetti_project_file_kmb': ['014jettiprojectfile_kmb'],
    '015_jetti_project_file_amcf_6in': ['015jettiprojectfile_pv', '015jettiprojectfile_pv_6in'],
    '015_jetti_project_file_amcf_8in': ['015jettiprojectfile_pv', '015jettiprojectfile_pv_8in'],
    '017_jetti_project_file_ea_mill_feed_combined': ['017jettiprojectfile_ea'],
    '020_jetti_project_file_hypogene_supergene_hypogene_master_composite': ['020jettiprojectfile_hyp'],
    '022_jetti_project_file_stingray_1': ['022jettiprojectfile_stingray'],
    '024_jetti_project_file_024cv_cpy': ['024jettiprojectfile_cpy'],
    '026_jetti_project_file_sample_1_primary_sulfide': ['026jettiprojectfile_primarysulfide'],
    '026_jetti_project_file_sample_2_carrizalillo': ['026jettiprojectfile_carrizalillo'],
    '026_jetti_project_file_sample_3_secondary_sulfide': ['026jettiprojectfile_secondarysulfide'],
    'jetti_file_elephant_ii_pq': ['jettifile_elephant_pq'],
    'jetti_file_elephant_ii_ugm2': ['jettifile_elephant_ugm'],
    'jetti_file_elephant_ii_ugm2_coarse': ['007ajettiprojectfile_elephant_ugm2_rthead_coarse'],
    'jetti_project_file_elephant_site': [''],
    'jetti_project_file_elephant_scl_sample_escondida': [''],
    'jetti_project_file_leopard_scl_sample_los_bronces': ['jettiprojectfile_leopard'],
    'jetti_project_file_toquepala_scl_sample_fresca': ['007jettiprojectfile_toquepala_fresca', 'jettiprojectfile_toquepala_fresca'],
    'jetti_project_file_zaldivar_scl_sample_zaldivar': ['007jettiprojectfile_zaldivar', 'jettiprojectfile_zaldivar'],
}

def add_match_keys(df_columns, df_reactors, match_map, key_col="project_sample_id_reactormatch"):
    df_columns = df_columns.copy()
    df_reactors = df_reactors.copy()

    # Columns: use their project_sample_id as the canonical key when present in match_map
    df_columns[key_col] = np.nan
    in_map = df_columns["project_sample_id"].isin(match_map.keys())
    df_columns.loc[in_map, key_col] = df_columns.loc[in_map, "project_sample_id"]

    # Reactors: map reactor sample ids back to column sample ids
    inv_map = {}
    for col_id, reactor_ids in match_map.items():
        for rid in reactor_ids:
            if rid and rid not in inv_map:
                inv_map[rid] = col_id

    df_reactors[key_col] = df_reactors["project_sample_id"].map(inv_map)

    return df_columns, df_reactors
 

#%%
# Rename the Cu sequential columns
df_reactors_filtered = df_reactors_filtered.rename(columns={
    "cu_seq_h2so4_norm%": "acid_soluble_%",
    "cu_seq_nacn_norm%": "cyanide_soluble_%",
    "cu_seq_a_r_norm%": "residual_cpy_%",
    # "cu_%": "feed_head_cu_%"
})

df_reactors_filtered['acid_soluble_%'] = df_reactors_filtered['acid_soluble_%'] * 100.0
df_reactors_filtered['cyanide_soluble_%'] = df_reactors_filtered['cyanide_soluble_%'] * 100.0
df_reactors_filtered['residual_cpy_%'] = df_reactors_filtered['residual_cpy_%'] * 100

#%%
# Create the curves of recovery for reactors using a1 to b2 params
def construct_reactor_recovery_curve(a1, b1, a2, b2, t_days):
    """Double-exponential recovery curve used for reactors (same functional form as columns)."""
    t = np.asarray(t_days, dtype=float)
    t = np.clip(t, 0.0, None)
    return np.round((float(a1) * (1.0 - np.exp(-float(b1) * t))
            + float(a2) * (1.0 - np.exp(-float(b2) * t))), 1)

def build_reactor_recovery_arrays(
    df,
    max_days=125,
    step_days=1,
    time_col="leach_duration_days_const",
    target_col="cu_recovery_%_calc",
    param_cols=("a1_param", "b1_param", "a2_param", "b2_param"),
):
    df = df.copy()
    time_grid = np.arange(0, max_days + step_days, step_days, dtype=float)

    def _build_curve(row):
        try:
            a1 = float(row[param_cols[0]])
            b1 = float(row[param_cols[1]])
            a2 = float(row[param_cols[2]])
            b2 = float(row[param_cols[3]])
        except Exception:
            return np.full_like(time_grid, np.nan, dtype=float)
        if not np.isfinite([a1, b1, a2, b2]).all():
            return np.full_like(time_grid, np.nan, dtype=float)
        return construct_reactor_recovery_curve(a1, b1, a2, b2, time_grid)

    df[time_col] = [time_grid.copy() for _ in range(len(df))]
    df[target_col] = df.apply(_build_curve, axis=1)
    return df

df_reactors_filtered = build_reactor_recovery_arrays(
    df_reactors_filtered,
    max_days=125,
    step_days=1,
)


#%%
if __name__ == "__main__":
    filter_snapshots = []
    df_columns_raw = pd.read_csv(DATA_PATH_COLUMNS, sep=",")
    df_columns_filtered = apply_column_filters(df_columns_raw, snapshots=filter_snapshots)
    df_columns_filtered, df_reactors_filtered = add_match_keys(
        df_columns_filtered, df_reactors_filtered, match_dict
    )

    df_model_recCu_catcontrol_projects_processed = prepare_column_train_data(
        df_columns_filtered,
        COLUMN_CONFIG,
        output_type="averaged",
        fill_noncat_averages=False,
        return_df_only=True,
        dropna_subset=False,
        drop_nan_rows=False,
        filter_snapshots=filter_snapshots,
    )

    def _pad_or_truncate(arr, length, fill_value=np.nan):
        if arr is None:
            return np.full(length, fill_value, dtype=float)
        arr = np.asarray(arr, dtype=float)
        if arr.size == length:
            return arr
        if arr.size < length:
            out = np.full(length, fill_value, dtype=float)
            out[:arr.size] = arr
            return out
        return arr[:length]

    def _align_row_arrays(row, cols, sort_by="leach_duration_days"):
        base = row.get(sort_by)
        if not isinstance(base, (list, np.ndarray)):
            return row
        base_arr = np.asarray(base, dtype=float)
        if base_arr.size == 0:
            return row
        order = np.argsort(base_arr)
        base_arr = base_arr[order]
        row[sort_by] = base_arr

        for col in cols:
            if col == sort_by:
                continue
            val = row.get(col)
            arr = np.asarray(val, dtype=float) if isinstance(val, (list, np.ndarray)) else None
            aligned = _pad_or_truncate(arr, base_arr.size, fill_value=np.nan)
            if aligned.size == base_arr.size:
                aligned = aligned[order]
            row[col] = aligned
        return row

    def _safe_filename(text):
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(text))

    def _as_array(val):
        if isinstance(val, (list, np.ndarray)):
            arr = np.asarray(val, dtype=float)
            return arr
        return None

    def _round_array_to_intlike(val):
        if isinstance(val, (list, np.ndarray)):
            arr = np.asarray(val, dtype=float)
            rounded = np.rint(arr)
            return np.where(np.isfinite(arr), rounded, np.nan)
        return val

    base_array_cols = _present_dynamic_array_cols(df_model_recCu_catcontrol_projects_processed)

    df_model_recCu_catcontrol_projects_processed = (
        df_model_recCu_catcontrol_projects_processed.apply(
            _align_row_arrays,
            axis=1,
            cols=base_array_cols,
        )
    )
    df_model_recCu_catcontrol_projects_processed["leach_duration_days"] = (
        df_model_recCu_catcontrol_projects_processed["leach_duration_days"]
        .apply(_round_array_to_intlike)
    )

    denom = (
        ((df_model_recCu_catcontrol_projects_processed["column_inner_diameter_m"] / 2) ** 2)
        * np.pi
        * df_model_recCu_catcontrol_projects_processed["irrigation_rate_l_m2_h"]
        * 24
    )
    df_model_recCu_catcontrol_projects_processed["catalyst_saturation_inside_column_day"] = np.nan
    with_cat_mask = (
        df_model_recCu_catcontrol_projects_processed["catalyst_status"] == "with_catalyst"
    )
    df_model_recCu_catcontrol_projects_processed.loc[with_cat_mask, "catalyst_saturation_inside_column_day"] = (
        (df_model_recCu_catcontrol_projects_processed.loc[with_cat_mask, "feed_mass_kg"] * 0.08)
        / denom.loc[with_cat_mask]
    )
    df_model_recCu_catcontrol_projects_processed["catalyst_saturation_inside_column_day"] = (
        df_model_recCu_catcontrol_projects_processed["catalyst_saturation_inside_column_day"]
        .replace([np.inf, -np.inf], np.nan)
        .round(0)
    )

    df_model_recCu_catcontrol_projects_processed["separation_adjusted_days"] = pd.Series(
        [np.nan] * len(df_model_recCu_catcontrol_projects_processed),
        index=df_model_recCu_catcontrol_projects_processed.index,
        dtype=object,
    )
    df_model_recCu_catcontrol_projects_processed["delta_cu_rec"] = pd.Series(
        [np.nan] * len(df_model_recCu_catcontrol_projects_processed),
        index=df_model_recCu_catcontrol_projects_processed.index,
        dtype=object,
    )

    for sample_id, group in df_model_recCu_catcontrol_projects_processed.groupby("project_sample_id"):
        catalyzed_row = group[group["project_col_id"] == "Catalyzed"]
        control_row = group[group["project_col_id"] == "Control"]
        if catalyzed_row.empty:
            continue
        cat_idx = catalyzed_row.index[0]
        cat_leach = catalyzed_row["leach_duration_days"].iloc[0]
        cat_cu = catalyzed_row["cu_recovery_%"].iloc[0]
        cat_cat = catalyzed_row["cumulative_catalyst_addition_kg_t"].iloc[0]
        if not all(isinstance(v, (list, np.ndarray)) for v in [cat_leach, cat_cu, cat_cat]):
            continue
        cat_leach = np.asarray(cat_leach, dtype=float)
        cat_cu = np.asarray(cat_cu, dtype=float)
        cat_cat = np.asarray(cat_cat, dtype=float)
        if cat_leach.size == 0:
            continue
        cat_cu = _pad_or_truncate(cat_cu, cat_leach.size, fill_value=np.nan)
        cat_cat = _pad_or_truncate(cat_cat, cat_leach.size, fill_value=np.nan)

        ctrl_cu = None
        ctrl_leach = None
        if not control_row.empty:
            ctrl_vals = control_row["cu_recovery_%"].iloc[0]
            ctrl_days = control_row["leach_duration_days"].iloc[0]
            if isinstance(ctrl_vals, (list, np.ndarray)) and isinstance(ctrl_days, (list, np.ndarray)):
                ctrl_cu = np.asarray(ctrl_vals, dtype=float)
                ctrl_leach = np.asarray(ctrl_days, dtype=float)
                n_ctrl = min(ctrl_cu.size, ctrl_leach.size)
                ctrl_cu = ctrl_cu[:n_ctrl]
                ctrl_leach = ctrl_leach[:n_ctrl]

        transition_time = catalyzed_row["transition_time"].iloc[0] if "transition_time" in catalyzed_row.columns else np.nan
        saturation_time = (
            catalyzed_row["catalyst_saturation_inside_column_day"].iloc[0]
            if "catalyst_saturation_inside_column_day" in catalyzed_row.columns
            else np.nan
        )
        sep_days = cat_leach - float(transition_time) - float(saturation_time)
        df_model_recCu_catcontrol_projects_processed.at[cat_idx, "separation_adjusted_days"] = sep_days

        delta = np.full(cat_leach.size, np.nan, dtype=float)
        if ctrl_cu is not None and ctrl_leach is not None:
            common_days, idx_cat, idx_ctrl = np.intersect1d(
                cat_leach, ctrl_leach, return_indices=True
            )
            if common_days.size > 0:
                delta[idx_cat] = cat_cu[idx_cat] - ctrl_cu[idx_ctrl]
        df_model_recCu_catcontrol_projects_processed.at[cat_idx, "delta_cu_rec"] = delta

    df_model_recCu_catcontrol_projects_processed["separation_adjusted_days"] = (
        df_model_recCu_catcontrol_projects_processed["separation_adjusted_days"]
        .apply(_round_array_to_intlike)
    )

    df_model_recCu_catcontrol_projects_processed = (
        df_model_recCu_catcontrol_projects_processed.apply(
            _align_row_arrays,
            axis=1,
            cols=_present_dynamic_array_cols(
                df_model_recCu_catcontrol_projects_processed,
                extra_cols=["separation_adjusted_days", "delta_cu_rec"],
            ),
        )
    )

    # Plotting
    plots_dir = os.path.join(DB_OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for sample_id, group in df_model_recCu_catcontrol_projects_processed.groupby("project_sample_id"):
        safe_id = _safe_filename(sample_id)
        catalyzed_row = group[group["project_col_id"] == "Catalyzed"]
        control_row = group[group["project_col_id"] == "Control"]

        # Control vs Catalyzed curves
        plt.figure(figsize=(8, 5))
        plotted = False
        if not control_row.empty:
            ctrl_days = _as_array(control_row["leach_duration_days"].iloc[0])
            ctrl_cu = _as_array(control_row["cu_recovery_%"].iloc[0])
            if ctrl_days is not None and ctrl_cu is not None and len(ctrl_days) > 0:
                plt.plot(ctrl_days, ctrl_cu, color="blue", label="Control")
                plotted = True
        if not catalyzed_row.empty:
            cat_days = _as_array(catalyzed_row["leach_duration_days"].iloc[0])
            cat_cu = _as_array(catalyzed_row["cu_recovery_%"].iloc[0])
            if cat_days is not None and cat_cu is not None and len(cat_days) > 0:
                plt.plot(cat_days, cat_cu, color="orange", label="Catalyzed")
                plotted = True
        if plotted:
            plt.title(f"Control vs Catalyzed - {sample_id}")
            plt.xlabel("leach_duration_days")
            plt.ylabel("cu_recovery_%")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"control_vs_catalyzed_{safe_id}.png"), dpi=200)
        plt.close()

        # Delta Cu Recovery plot (Catalyzed - Control)
        if not catalyzed_row.empty:
            delta = _as_array(catalyzed_row["delta_cu_rec"].iloc[0])
            sep_days = _as_array(catalyzed_row["separation_adjusted_days"].iloc[0])
            if delta is not None and sep_days is not None and len(delta) > 0 and len(sep_days) > 0:
                n = min(len(delta), len(sep_days))
                plt.figure(figsize=(8, 5))
                plt.plot(sep_days[:n], delta[:n], color="orange", label="Delta Cu Rec")
                plt.title(f"Delta Cu Recovery - {sample_id}")
                plt.xlabel("separation_adjusted_days")
                plt.ylabel("delta_cu_rec")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"delta_cu_rec_{safe_id}.png"), dpi=200)
                plt.close()

    df_model_recCu_catcontrol_projects_processed = _prepare_array_cols_for_csv(
        df_model_recCu_catcontrol_projects_processed,
        _present_dynamic_array_cols(
            df_model_recCu_catcontrol_projects_processed,
            extra_cols=["separation_adjusted_days", "delta_cu_rec"],
        ),
    )
    df_model_recCu_catcontrol_projects_processed = _round_numeric_df(
        df_model_recCu_catcontrol_projects_processed, decimals=3
    )

    if filter_snapshots:
        filter_snapshots_df = pd.concat(filter_snapshots, ignore_index=True, sort=False)
        filter_snapshots_df = _prepare_array_cols_for_csv(
            filter_snapshots_df,
            _present_dynamic_array_cols(filter_snapshots_df),
        )
        filter_snapshots_df = _round_numeric_df(filter_snapshots_df, decimals=3)
        filter_snapshots_df.to_csv(
            os.path.join(DB_OUTPUT_DIR, "df_model_recCu_catcontrol_projects_filter_stages.csv"),
            index=False,
        )

    df_reactors_out = _round_numeric_df(df_reactors_filtered, decimals=3)
    df_reactors_out.to_csv(os.path.join(DB_OUTPUT_DIR, "df_reactors_filtered.csv"), index=False)
    df_reactors_out.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/power_bi/df_reactors_filtered.csv', index=False,)
    df_reactors_out.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/reactors/df_reactors_filtered.csv', index=False,)

    df_model_recCu_catcontrol_projects_processed.to_csv(
        os.path.join(DB_OUTPUT_DIR, "df_recCu_catcontrol_projects_averaged.csv"),
        index=False,
    )
    df_model_recCu_catcontrol_projects_processed.to_csv(
        '/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/power_bi/df_recCu_catcontrol_projects_averaged.csv',
        index=False,
    )
    df_model_recCu_catcontrol_projects_processed.to_csv(
        '/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/columns/df_recCu_catcontrol_projects_averaged.csv',
        index=False,
    )
    df_model_recCu_catcontrol_projects_processed.to_csv(
        '/Users/administration/Library/CloudStorage/OneDrive-JettiResources/PythonProjects/Rosetta/build_assets/df_recCu_catcontrol_projects_averaged.csv',
        index=False,
    )
    print(
        "Saved df_reactors and df_recCu_catcontrol_projects_averaged to",
        DB_OUTPUT_DIR,
    )


#%%
# 
