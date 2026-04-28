"""
step_06_columns_reactors.py — Column/reactor averaging and final Power BI outputs.

Corresponds to: data_prep_columns_reactors.py

This step:
  1. Loads the model catcontrol CSV from step_05
  2. Applies column-level filters (manual exclusions)
  3. Adds reactor match keys
  4. Runs prepare_column_train_data (averaged mode) to produce
     one row per project_sample_id × catalyst_status
  5. Computes delta_cu_rec (catalyzed minus control recovery)
  6. Saves final outputs for Power BI

Final outputs (the ones that don't change after this step):
  leaching_performance_weekly_averaged.csv
  leaching_performance_weekly.csv
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PATHS,
    COLS_TO_MATCH_CHEMCHAR,
    COLS_TO_MATCH_COLUMN_SUMMARY,
    COL_TO_MATCH_MINERALOGY,
    COLS_TO_MATCH_QEMSCAN,
)
from utils import save_intermediate, save_to_paths

INTERMEDIATE_DIR = Path(__file__).parent / "intermediate"


# ── Constants ──────────────────────────────────────────────────────────────────

MATCH_DICT = {
    '003_jetti_project_file_amcf_head':     ['003jettiprojectfile_amcf'],
    '006_jetti_project_file_pvo':           [''],
    '007_jetti_project_file_leopard_lep':   ['007jettiprojectfile_leopard'],
    '007b_jetti_project_file_tiger_tgr':    ['007bjettiprojectfile_tiger_m1'],
    '011_jetti_project_file_rm':            ['011jettiprojectfile_rm'],
    '011_jetti_project_file_rm_crushed':    ['011jettiprojectfile_rm_crushed'],
    '014_jetti_project_file_bag':           ['014jettiprojectfile_bag'],
    '014_jetti_project_file_kmb':           ['014jettiprojectfile_kmb'],
    '015_jetti_project_file_amcf_6in':      ['015jettiprojectfile_pv','015jettiprojectfile_pv_6in'],
    '015_jetti_project_file_amcf_8in':      ['015jettiprojectfile_pv','015jettiprojectfile_pv_8in'],
    '017_jetti_project_file_ea_mill_feed_combined': ['017jettiprojectfile_ea'],
    '020_jetti_project_file_hypogene_supergene_hypogene_master_composite': ['020jettiprojectfile_hyp'],
    '022_jetti_project_file_stingray_1':    ['022jettiprojectfile_stingray'],
    '024_jetti_project_file_024cv_cpy':     ['024jettiprojectfile_cpy'],
    '026_jetti_project_file_sample_1_primary_sulfide':  ['026jettiprojectfile_primarysulfide'],
    '026_jetti_project_file_sample_2_carrizalillo':     ['026jettiprojectfile_carrizalillo'],
    '026_jetti_project_file_sample_3_secondary_sulfide':['026jettiprojectfile_secondarysulfide'],
    'jetti_file_elephant_ii_pq':            ['jettifile_elephant_pq'],
    'jetti_file_elephant_ii_ugm2':          ['jettifile_elephant_ugm'],
    'jetti_file_elephant_ii_ugm2_coarse':   ['007ajettiprojectfile_elephant_ugm2_rthead_coarse'],
    'jetti_project_file_elephant_site':     [''],
    'jetti_project_file_elephant_scl_sample_escondida': [''],
    'jetti_project_file_leopard_scl_sample_los_bronces':['jettiprojectfile_leopard'],
    'jetti_project_file_toquepala_scl_sample_fresca':   ['007jettiprojectfile_toquepala_fresca','jettiprojectfile_toquepala_fresca'],
    'jetti_project_file_zaldivar_scl_sample_zaldivar':  ['007jettiprojectfile_zaldivar','jettiprojectfile_zaldivar'],
}

DYNAMIC_ARRAY_COLUMNS = [
    'leach_duration_days', 'cu_recovery_%', 'cumulative_catalyst_addition_kg_t',
    'cumulative_lixiviant_flowthrough_l', 'cumulative_lixiviant_m3_t',
    'feed_orp_mv_ag_agcl', 'pls_orp_mv_ag_agcl',       # feed ORP + PLS ORP
    'feed_flowrate_ml_min', 'irrigation_rate_l_h_m2',
    'catalyst_addition_mg_l',  # time-series catalyst concentration from leaching performance
    # catalyst_dosage_mg_l and catalyst_dosage_mg_day are SCALAR design values
    # from the column summary — they are NOT array-like
    'raff_assay_fe_mg_l',                               # total raffinate Fe
    'raff_assay_fe_ii_mg_l', 'raff_assay_fe_iii_mg_l', # raffinate Fe speciation
    'pls_fe_ii_mg_l',        'pls_fe_iii_mg_l',         # PLS Fe speciation
]

MONOTONIC_ARRAY_COLUMNS = {
    'cu_recovery_%',
    'cumulative_catalyst_addition_kg_t',
    'cumulative_lixiviant_flowthrough_l',
    'delta_cu_rec',
}


# ── Helper functions ───────────────────────────────────────────────────────────

def apply_column_filters(df, snapshots=None):
    """Apply non-destructive column-level adjustments for the averaged export."""
    df = df.copy()
    for col in ["feed_mass_kg","column_inner_diameter_m"]:
        if col not in df.columns:
            df[col] = np.nan

    # MANUAL FILTER: Separate UGM2 ROM vs Crushed
    if 'project_col_id' in df.columns:
        mask = df["project_col_id"].str.startswith("jetti_file_elephant_ii_ver_2_ugm_ur", na=False)
        df.loc[mask, "project_sample_id"] = "jetti_file_elephant_ii_ugm2_coarse"

    # Computed ratio columns
    for col_a, col_b, new_col in [
        ('feed_head_fe_%', 'feed_head_cu_%', 'fe:cu'),
        ('feed_head_cu_%', 'feed_head_fe_%', 'cu:fe'),
    ]:
        if col_a in df.columns and col_b in df.columns:
            df[new_col] = df[col_a] / df[col_b]

    for num_col, denom_col, new_col in [
        ('cu_%', 'residual_cpy_%',   'copper_primary_sulfides_equivalent'),
        ('cu_%', 'cyanide_soluble_%','copper_secondary_sulfides_equivalent'),
        ('cu_%', 'acid_soluble_%',   'copper_oxides_equivalent'),
    ]:
        if num_col in df.columns and denom_col in df.columns:
            df[new_col] = df[num_col] * df[denom_col] / 100

    if 'residual_cpy_%' in df.columns and 'cyanide_soluble_%' in df.columns and 'cu_%' in df.columns:
        df['copper_sulfides_equivalent'] = df['cu_%'] * (df['residual_cpy_%'] + df['cyanide_soluble_%']) / 100

    return df


def add_match_keys(df_columns, df_reactors, match_map, key_col="project_sample_id_reactormatch"):
    """
    Add reactor match keys to columns and reactors DataFrames.
    BUG FIX: initialise with None (object dtype) not np.nan (float64) to
    accept Arrow string values.
    """
    df_columns  = df_columns.copy()
    df_reactors = df_reactors.copy()

    df_columns[key_col] = None
    df_columns[key_col] = df_columns[key_col].astype(object)
    in_map = df_columns["project_sample_id"].isin(match_map.keys())
    df_columns.loc[in_map, key_col] = df_columns.loc[in_map, "project_sample_id"]

    inv_map = {}
    for col_id, reactor_ids in match_map.items():
        for rid in reactor_ids:
            if rid and rid not in inv_map:
                inv_map[rid] = col_id
    df_reactors[key_col] = df_reactors["project_sample_id"].map(inv_map)

    return df_columns, df_reactors


def process_arrays_by_weekly_intervals(
    arr,
    leach_days,
    col_name,
    test_ids=None,
    singleton_drop_threshold=0.8,
    min_tests_for_singleton_drop=2,
    interpolate_nans=True,
    weekly_days=None,
    enforce_monotonic=False,
):
    """
    Average time-series values within strict 7-day bins (0-7, 7-14, 14-21, …)
    while preserving the full weekly grid.
    """
    weekly_days = np.asarray(weekly_days, dtype=float) if weekly_days is not None else None
    if weekly_days is not None and col_name == 'leach_duration_days':
        return weekly_days.copy()

    arr = np.asarray(arr, dtype=float)
    leach_days = np.asarray(leach_days, dtype=float)
    if arr.size == 0 or leach_days.size == 0:
        if weekly_days is not None:
            return np.full(len(weekly_days), np.nan, dtype=float)
        return np.array([])

    finite_mask = np.isfinite(leach_days)
    arr = arr[finite_mask]
    leach_days = leach_days[finite_mask]
    if test_ids is not None:
        test_ids = np.asarray(test_ids)
        if test_ids.shape[0] == finite_mask.shape[0]:
            test_ids = test_ids[finite_mask]
        else:
            test_ids = None
    if arr.size == 0:
        if weekly_days is not None:
            return np.full(len(weekly_days), np.nan, dtype=float)
        return np.array([])

    order = np.argsort(leach_days)
    arr = arr[order]
    leach_days = leach_days[order]
    if test_ids is not None:
        test_ids = test_ids[order]

    if weekly_days is None:
        max_days = leach_days.max()
        weekly_days = np.arange(7, max_days + 7, 7)
    if weekly_days.size == 0:
        return np.array([])

    processed = np.full(len(weekly_days), np.nan, dtype=float)
    weekly_counts = np.zeros(len(weekly_days), dtype=float)
    for i, end_day in enumerate(weekly_days):
        start_day = end_day - 7
        idx = np.where((leach_days >= start_day) & (leach_days < end_day))[0]
        if len(idx) == 0:
            continue
        batch_vals = arr[idx]
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            processed[i] = np.nanmean(batch_vals) if batch_vals.size > 0 else np.nan
        if test_ids is not None:
            valid_idx = idx[np.isfinite(arr[idx])]
            weekly_counts[i] = len(np.unique(test_ids[valid_idx])) if valid_idx.size > 0 else 0
        else:
            weekly_counts[i] = len(idx)

    # Singleton drop: if most bins have multi-test coverage, zero-out solo bins
    if (
        test_ids is not None
        and col_name != 'leach_duration_days'
        and len(weekly_counts) > 0
    ):
        frac_multi = np.mean(weekly_counts >= min_tests_for_singleton_drop)
        if frac_multi > singleton_drop_threshold:
            processed[weekly_counts < min_tests_for_singleton_drop] = np.nan

    if interpolate_nans:
        processed = _interpolate_internal(
            processed,
            enforce_monotonic=enforce_monotonic,
        )

    return processed


def _pad_or_truncate_array(arr, length, fill_value=np.nan):
    if length <= 0:
        return np.array([], dtype=float)
    if arr is None or not isinstance(arr, (list, np.ndarray)):
        return np.full(length, fill_value, dtype=float)

    arr = np.asarray(arr, dtype=float)
    if arr.size == length:
        return arr.copy()
    out = np.full(length, fill_value, dtype=float)
    copy_len = min(arr.size, length)
    out[:copy_len] = arr[:copy_len]
    return out


def _normalise_gate_mask(mask, length):
    if mask is None:
        return np.ones(length, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.size == length:
        return mask.copy()
    out = np.zeros(length, dtype=bool)
    copy_len = min(mask.size, length)
    out[:copy_len] = mask[:copy_len]
    return out


def _interpolate_internal(arr, gate_mask=None, enforce_monotonic=False):
    arr = np.asarray(arr, dtype=float).copy()
    if arr.size == 0:
        return arr

    gate_mask = _normalise_gate_mask(gate_mask, arr.size)
    valid = np.isfinite(arr)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size >= 2:
        x = np.arange(arr.size, dtype=float)
        interpolated = np.interp(x, x[valid], arr[valid])
        internal_mask = (
            ~valid
            & gate_mask
            & (x > valid_idx[0])
            & (x < valid_idx[-1])
        )
        arr[internal_mask] = interpolated[internal_mask]

    if enforce_monotonic:
        finite_idx = np.flatnonzero(np.isfinite(arr))
        if finite_idx.size > 0:
            arr[finite_idx] = np.maximum.accumulate(arr[finite_idx])

    return arr


def _repair_monotonic_increase(arr, gate_mask=None):
    """
    Repair decreasing segments in arrays that should be cumulative / monotonic.

    Rule:
    - keep the existing weekly averages where they already respect monotonicity
    - when a value drops below the last valid value, bridge that segment to the
      next future value that is at least as large as the last valid value
    - if no such future value exists, fall back to a cumulative max on the tail
    """
    arr = np.asarray(arr, dtype=float).copy()
    if arr.size == 0:
        return arr

    gate_mask = _normalise_gate_mask(gate_mask, arr.size)
    eligible_idx = np.flatnonzero(gate_mask & np.isfinite(arr))
    if eligible_idx.size < 2:
        return arr

    pos = 1
    prev_idx = eligible_idx[0]
    prev_val = arr[prev_idx]

    while pos < eligible_idx.size:
        cur_idx = eligible_idx[pos]
        cur_val = arr[cur_idx]

        if cur_val >= prev_val:
            prev_idx = cur_idx
            prev_val = cur_val
            pos += 1
            continue

        future_pos = None
        for probe in range(pos + 1, eligible_idx.size):
            probe_idx = eligible_idx[probe]
            probe_val = arr[probe_idx]
            if probe_val >= prev_val:
                future_pos = probe
                break

        if future_pos is None:
            tail_idx = eligible_idx[pos:]
            arr[tail_idx] = np.maximum.accumulate(np.maximum(arr[tail_idx], prev_val))
            break

        future_idx = eligible_idx[future_pos]
        future_val = arr[future_idx]
        span_idx = np.arange(prev_idx, future_idx + 1, dtype=int)
        gated_span = span_idx[gate_mask[span_idx]]
        if gated_span.size >= 2:
            arr[gated_span] = np.interp(
                gated_span.astype(float),
                np.array([prev_idx, future_idx], dtype=float),
                np.array([prev_val, future_val], dtype=float),
            )

        prev_idx = future_idx
        prev_val = arr[future_idx]
        pos = future_pos + 1

    eligible_idx = np.flatnonzero(gate_mask & np.isfinite(arr))
    if eligible_idx.size > 0:
        arr[eligible_idx] = np.maximum.accumulate(arr[eligible_idx])

    return arr


def _extract_numeric_scalar(value):
    if isinstance(value, (list, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        finite = arr[np.isfinite(arr)]
        return float(finite[0]) if finite.size > 0 else np.nan
    try:
        coerced = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    except Exception:
        coerced = np.nan
    return float(coerced) if pd.notna(coerced) else np.nan


def _infer_catalyst_status(subset, cat_col='cumulative_catalyst_addition_kg_t'):
    if cat_col not in subset.columns:
        return 'no_catalyst'
    cat_vals = pd.to_numeric(subset[cat_col], errors='coerce')
    return 'with_catalyst' if (cat_vals > 0).any() else 'no_catalyst'


def _build_weekly_grid(group, leach_col='leach_duration_days'):
    if leach_col not in group.columns:
        return np.array([], dtype=float)
    all_days = pd.to_numeric(group[leach_col], errors='coerce').to_numpy(dtype=float)
    all_days = all_days[np.isfinite(all_days)]
    if all_days.size == 0:
        return np.array([], dtype=float)
    shared_max = float(np.nanmax(all_days))
    if not np.isfinite(shared_max) or shared_max <= 0:
        return np.array([], dtype=float)
    return np.arange(7, shared_max + 7, 7, dtype=float)


def _build_weekly_arrays_for_subset(
    subset,
    weekly_grid,
    time_varying,
    leach_col='leach_duration_days',
    target_col='cu_recovery_%',
    test_id_col=None,
):
    n_bins = len(weekly_grid)
    if n_bins == 0:
        return {}

    test_ids = subset[test_id_col].values if test_id_col and test_id_col in subset.columns else None
    arrays = {
        leach_col: weekly_grid.copy(),
    }

    target_arr = process_arrays_by_weekly_intervals(
        pd.to_numeric(subset[target_col], errors='coerce').to_numpy(dtype=float),
        pd.to_numeric(subset[leach_col], errors='coerce').to_numpy(dtype=float),
        target_col,
        test_ids=test_ids,
        weekly_days=weekly_grid,
        interpolate_nans=False,
    )
    target_arr = _interpolate_internal(target_arr)
    target_arr = _repair_monotonic_increase(target_arr)
    arrays[target_col] = _pad_or_truncate_array(target_arr, n_bins)
    target_mask = np.isfinite(arrays[target_col])

    for col in time_varying:
        if col in (leach_col, target_col):
            continue
        if col not in subset.columns:
            arrays[col] = np.full(n_bins, np.nan, dtype=float)
            continue
        col_arr = process_arrays_by_weekly_intervals(
            pd.to_numeric(subset[col], errors='coerce').to_numpy(dtype=float),
            pd.to_numeric(subset[leach_col], errors='coerce').to_numpy(dtype=float),
            col,
            test_ids=test_ids,
            weekly_days=weekly_grid,
            interpolate_nans=False,
        )
        col_arr = _pad_or_truncate_array(col_arr, n_bins)
        col_arr[~target_mask] = np.nan
        col_arr = _interpolate_internal(col_arr, gate_mask=target_mask)
        col_arr[~target_mask] = np.nan
        if col in MONOTONIC_ARRAY_COLUMNS:
            col_arr = _repair_monotonic_increase(col_arr, gate_mask=target_mask)
            col_arr[~target_mask] = np.nan
        arrays[col] = col_arr

    return arrays


def _present_array_columns(df, extra_cols=None):
    cols = [col for col in DYNAMIC_ARRAY_COLUMNS if col in df.columns]
    for col in extra_cols or []:
        if col in df.columns and col not in cols:
            cols.append(col)
    return cols


def _align_row_array_lengths(row, array_cols, base_col='leach_duration_days'):
    base = row.get(base_col)
    if not isinstance(base, (list, np.ndarray)):
        return row

    base_arr = np.asarray(base, dtype=float)
    if base_arr.size == 0:
        return row

    order = np.argsort(np.where(np.isfinite(base_arr), base_arr, np.inf))
    base_arr = base_arr[order]
    row[base_col] = base_arr

    for col in array_cols:
        if col == base_col or col not in row:
            continue
        value = row.get(col)
        if not isinstance(value, (list, np.ndarray)):
            continue
        aligned = _pad_or_truncate_array(value, base_arr.size)
        row[col] = aligned[order]

    return row


def _validate_array_contract(
    df,
    array_cols,
    id_col='project_sample_id',
    col_id='project_col_id',
    base_col='leach_duration_days',
    pair_control_id='Control',
    pair_catalyzed_id='Catalyzed',
):
    issues = []

    for idx, row in df.iterrows():
        base = row.get(base_col)
        if not isinstance(base, (list, np.ndarray)):
            continue
        base_len = len(np.asarray(base, dtype=float))
        for col in array_cols:
            value = row.get(col)
            if isinstance(value, (list, np.ndarray)):
                arr_len = len(np.asarray(value, dtype=float))
                if arr_len != base_len:
                    issues.append(
                        f"row {idx} sample={row.get(id_col)} col={col} len={arr_len} base_len={base_len}"
                    )

    if (
        pair_control_id is not None
        and pair_catalyzed_id is not None
        and id_col in df.columns
        and col_id in df.columns
        and base_col in df.columns
    ):
        for sample_id, group in df.groupby(id_col):
            control_row = group[group[col_id] == pair_control_id]
            catalyzed_row = group[group[col_id] == pair_catalyzed_id]
            if control_row.empty or catalyzed_row.empty:
                continue
            control_base = control_row[base_col].iloc[0]
            catalyzed_base = catalyzed_row[base_col].iloc[0]
            if not isinstance(control_base, (list, np.ndarray)) or not isinstance(catalyzed_base, (list, np.ndarray)):
                continue
            control_len = len(np.asarray(control_base, dtype=float))
            catalyzed_len = len(np.asarray(catalyzed_base, dtype=float))
            if control_len != catalyzed_len:
                issues.append(
                    f"sample={sample_id} pair mismatch control_len={control_len} catalyzed_len={catalyzed_len}"
                )

    return issues


def _validate_interpolation_gate(df, gated_cols, target_col='cu_recovery_%'):
    issues = []
    for idx, row in df.iterrows():
        target = row.get(target_col)
        if not isinstance(target, (list, np.ndarray)):
            continue
        target_arr = np.asarray(target, dtype=float)
        gate_mask = np.isfinite(target_arr)
        for col in gated_cols:
            value = row.get(col)
            if not isinstance(value, (list, np.ndarray)):
                continue
            arr = _pad_or_truncate_array(value, target_arr.size)
            invalid_idx = np.flatnonzero(~gate_mask & np.isfinite(arr))
            if invalid_idx.size > 0:
                issues.append(
                    f"row {idx} sample={row.get('project_sample_id')} col={col} has values outside cu_recovery gate at indices {invalid_idx[:5].tolist()}"
                )
    return issues


def prepare_averaged(df, fill_noncat_averages=False, dropna_subset=False):
    """
    Produce one row per (project_sample_id, catalyst_status).

    Time-varying columns (DYNAMIC_ARRAY_COLUMNS) are stored as numpy arrays
    produced by process_arrays_by_weekly_intervals — the same weekly-binning
    function used in the original prepare_column_train_data.

    Scalar columns take the first value of the group.
    """
    df = df.copy()
    id_col = 'project_sample_id'
    col_idx = 'project_col_id'
    target_col = 'cu_recovery_%'
    leach_col = 'leach_duration_days'
    cat_col = 'cumulative_catalyst_addition_kg_t'

    if target_col not in df.columns or leach_col not in df.columns:
        raise ValueError(f"Missing required columns: {target_col}, {leach_col}")

    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df[leach_col] = pd.to_numeric(df[leach_col], errors='coerce')
    if cat_col in df.columns:
        df[cat_col] = pd.to_numeric(df[cat_col], errors='coerce')

    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    time_varying = [c for c in DYNAMIC_ARRAY_COLUMNS if c in df.columns]
    df_filtered  = df.dropna(
        subset=[c for c in [target_col, leach_col, cat_col] if c in df.columns]
    ).copy() if dropna_subset else df.copy()

    start_col = 'catalyst_start_days_of_leaching'

    def _col_is_catalyzed(col_group: pd.DataFrame) -> bool:
        """
        Determine whether a project_col_id represents a catalyzed column.

        Primary check  : catalyst_start_days_of_leaching is a finite positive number.
        Secondary check: cumulative_catalyst_addition_kg_t has at least one value > 0.

        A column is catalyzed if EITHER condition is satisfied — this catches:
        - Columns where catalyst was added but cumulative tracking started late (start_days > 0)
        - Columns where the start field is missing but catalyst IS recorded (cumulative > 0)
        The double-check guards against columns that were zero-filled in both fields.
        """
        has_start = False
        if start_col in col_group.columns:
            vals = pd.to_numeric(col_group[start_col], errors='coerce').dropna()
            has_start = bool((vals > 0).any())

        has_cumulative = False
        if cat_col in col_group.columns:
            vals = pd.to_numeric(col_group[cat_col], errors='coerce').dropna()
            has_cumulative = bool((vals > 0).any())

        return has_start or has_cumulative

    final_rows = []
    for sample_id, group in df_filtered.groupby(id_col):
        # ── Classify each project_col_id as catalyzed or control ───────────────
        # Column-level classification: ALL rows of a catalyzed column belong to
        # with_cat_raw regardless of their individual cumulative value.
        # This correctly handles the pre-catalyst phase (cumulative still 0 but
        # catalyst_start_days_of_leaching > 0 tells us it IS a catalyzed column).
        cat_col_ids = set()
        ctl_col_ids = set()
        if col_idx in group.columns:
            for c_id, c_grp in group.groupby(col_idx, sort=False):
                (cat_col_ids if _col_is_catalyzed(c_grp) else ctl_col_ids).add(c_id)
        else:
            # No project_col_id column — fall back to row-level cumulative check
            if cat_col in group.columns:
                cat_col_ids = {'_all_cat'} if (group[cat_col] > 0).any() else set()
                ctl_col_ids = {'_all_ctl'} if (group[cat_col] == 0).any() else set()

        no_cat_raw   = group[group[col_idx].isin(ctl_col_ids)] if ctl_col_ids and col_idx in group.columns else pd.DataFrame(columns=group.columns)
        with_cat_raw = group[group[col_idx].isin(cat_col_ids)] if cat_col_ids and col_idx in group.columns else pd.DataFrame(columns=group.columns)

        shared_grid = _build_weekly_grid(group, leach_col=leach_col)
        if shared_grid.size == 0:
            continue

        # ── No-catalyst row ────────────────────────────────────────────────────
        no_cat_arrays = {}
        if not no_cat_raw.empty:
            row_no = {'catalyst_status': 'no_catalyst', id_col: sample_id, col_idx: 'Control'}
            no_cat_arrays = _build_weekly_arrays_for_subset(
                no_cat_raw,
                shared_grid,
                time_varying,
                leach_col=leach_col,
                target_col=target_col,
                test_id_col=col_idx,
            )
            if np.any(np.isfinite(no_cat_arrays.get(leach_col, np.array([])))):
                row_no.update(no_cat_arrays)
                for col in df.columns:
                    if col not in time_varying and col not in row_no:
                        row_no[col] = no_cat_raw[col].iloc[0] if col in no_cat_raw.columns else np.nan
                final_rows.append(row_no)

        # ── With-catalyst row ──────────────────────────────────────────────────
        if not with_cat_raw.empty:
            row_with = {'catalyst_status': 'with_catalyst', id_col: sample_id, col_idx: 'Catalyzed'}
            with_cat_arrays = _build_weekly_arrays_for_subset(
                with_cat_raw,
                shared_grid,
                time_varying,
                leach_col=leach_col,
                target_col=target_col,
                test_id_col=col_idx,
            )
            if fill_noncat_averages and no_cat_arrays:
                n_bins = len(shared_grid)
                with_start_idx = np.flatnonzero(np.isfinite(with_cat_arrays[target_col]))
                if with_start_idx.size > 0:
                    pre_idx = np.flatnonzero(
                        np.isfinite(no_cat_arrays[target_col]) & (np.arange(n_bins) < with_start_idx[0])
                    )
                    if pre_idx.size > 0:
                        with_cat_arrays[target_col][pre_idx] = no_cat_arrays[target_col][pre_idx]
                        for col in time_varying:
                            if col == leach_col:
                                continue
                            if col == cat_col:
                                with_cat_arrays[col][pre_idx] = 0.0
                            else:
                                with_cat_arrays[col][pre_idx] = no_cat_arrays.get(
                                    col,
                                    np.full(n_bins, np.nan),
                                )[pre_idx]
                        gate_mask = np.isfinite(with_cat_arrays[target_col])
                        for col in time_varying:
                            if col in (leach_col, target_col):
                                continue
                            with_cat_arrays[col][~gate_mask] = np.nan

            row_with.update(with_cat_arrays)
            if np.any(np.isfinite(with_cat_arrays.get(leach_col, np.array([])))):
                for col in df.columns:
                    if col not in time_varying and col not in row_with:
                        row_with[col] = with_cat_raw[col].iloc[0] if col in with_cat_raw.columns else np.nan
                final_rows.append(row_with)

    if not final_rows:
        return pd.DataFrame()

    out_cols = list(df.columns) + ['catalyst_status']
    out_df = pd.DataFrame(final_rows, columns=[c for c in out_cols if any(c in r for r in final_rows)])

    # ── Align arrays by leach_duration_days and round to integer-like ─────────
    array_cols = [c for c in time_varying if c in out_df.columns]
    out_df = out_df.apply(
        _align_row_array_lengths,
        axis=1,
        array_cols=array_cols,
        base_col=leach_col,
    )
    # Round leach_duration_days to nearest integer
    if leach_col in out_df.columns:
        out_df[leach_col] = out_df[leach_col].apply(
            lambda v: np.rint(np.asarray(v, dtype=float)) if isinstance(v, np.ndarray) else v
        )

    # ── Transition time ────────────────────────────────────────────────────────
    # For catalyzed rows: prefer catalyst_start_days_of_leaching (most accurate),
    # fall back to the first day cumulative catalyst > 0, then max of leach days.
    out_df['transition_time'] = 0.0
    for idx, row in out_df.iterrows():
        raw_group = df_filtered[df_filtered[id_col] == row.get(id_col)]
        if row.get('catalyst_status') == 'with_catalyst':
            # Primary: catalyst_start_days_of_leaching (from column summary)
            t_start = np.nan
            if start_col in raw_group.columns:
                start_vals = pd.to_numeric(raw_group[start_col], errors='coerce').dropna()
                valid_starts = start_vals[start_vals > 0]
                if not valid_starts.empty:
                    t_start = float(valid_starts.min())
            # Secondary: first leach day where cumulative catalyst > 0
            if not np.isfinite(t_start) and cat_col in raw_group.columns:
                cat_days_raw = raw_group[leach_col][raw_group[cat_col] > 0] if leach_col in raw_group.columns else pd.Series(dtype=float)
                if not cat_days_raw.empty:
                    t_start = float(cat_days_raw.min())
            # Fallback: last day of leaching
            if not np.isfinite(t_start) and leach_col in raw_group.columns:
                t_start = float(raw_group[leach_col].max())
            out_df.at[idx, 'transition_time'] = t_start if np.isfinite(t_start) else 0.0
        else:
            ld = row.get(leach_col)
            cu = row.get(target_col)
            if isinstance(ld, np.ndarray) and isinstance(cu, np.ndarray):
                valid_idx = np.flatnonzero(np.isfinite(cu))
                out_df.at[idx, 'transition_time'] = (
                    float(ld[valid_idx[-1]]) if valid_idx.size > 0 else 0.0
                )
                continue
            out_df.at[idx, 'transition_time'] = (
                float(ld[-1]) if isinstance(ld, np.ndarray) and ld.size > 0 else 0.0
            )
    out_df['transition_time'] = out_df['transition_time'].round(0)

    # ── Catalyst saturation (days to fully saturate column with catalyst) ──────
    if all(c in out_df.columns for c in ['feed_mass_kg', 'column_inner_diameter_m']):
        irrigation_rate = pd.Series(np.nan, index=out_df.index, dtype=float)
        if 'irrigation_rate_l_h_m2' in out_df.columns:
            irrigation_rate = out_df['irrigation_rate_l_h_m2'].apply(_extract_numeric_scalar)
        if 'irrigation_rate_l_m2_h' in out_df.columns:
            fallback_irrigation = out_df['irrigation_rate_l_m2_h'].apply(_extract_numeric_scalar)
            irrigation_rate = irrigation_rate.fillna(fallback_irrigation)

        denom = (
            ((pd.to_numeric(out_df['column_inner_diameter_m'], errors='coerce') / 2) ** 2)
            * np.pi
            * irrigation_rate
            * 24
        )
        out_df['catalyst_saturation_inside_column_day'] = np.nan
        wc_mask = out_df['catalyst_status'] == 'with_catalyst'
        out_df.loc[wc_mask, 'catalyst_saturation_inside_column_day'] = (
            (pd.to_numeric(out_df.loc[wc_mask, 'feed_mass_kg'], errors='coerce') * 0.08)
            / denom.loc[wc_mask]
        ).replace([np.inf, -np.inf], np.nan).round(0)

    return out_df


def prepare_weekly_by_column(df, dropna_subset=False):
    """
    Produce one weekly row per original project_col_id while preserving the
    shared weekly grid per sample.
    """
    df = df.copy()
    id_col = 'project_sample_id'
    col_idx = 'project_col_id'
    target_col = 'cu_recovery_%'
    leach_col = 'leach_duration_days'
    cat_col = 'cumulative_catalyst_addition_kg_t'

    if target_col not in df.columns or leach_col not in df.columns or col_idx not in df.columns:
        raise ValueError(f"Missing required columns: {target_col}, {leach_col}, {col_idx}")

    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df[leach_col] = pd.to_numeric(df[leach_col], errors='coerce')
    if cat_col in df.columns:
        df[cat_col] = pd.to_numeric(df[cat_col], errors='coerce')

    time_varying = [c for c in DYNAMIC_ARRAY_COLUMNS if c in df.columns]
    df_filtered = df.dropna(
        subset=[c for c in [target_col, leach_col, cat_col] if c in df.columns]
    ).copy() if dropna_subset else df.copy()

    final_rows = []
    for sample_id, group in df_filtered.groupby(id_col):
        shared_grid = _build_weekly_grid(group, leach_col=leach_col)
        if shared_grid.size == 0:
            continue

        for project_col_id, subset in group.groupby(col_idx):
            arrays = _build_weekly_arrays_for_subset(
                subset,
                shared_grid,
                time_varying,
                leach_col=leach_col,
                target_col=target_col,
                test_id_col=None,
            )
            if not np.any(np.isfinite(arrays.get(leach_col, np.array([])))):
                continue

            row = {
                id_col: sample_id,
                col_idx: project_col_id,
                'catalyst_status': _infer_catalyst_status(subset, cat_col=cat_col),
            }
            row.update(arrays)
            for col in df.columns:
                if col not in time_varying and col not in row:
                    row[col] = subset[col].iloc[0] if col in subset.columns else np.nan
            final_rows.append(row)

    if not final_rows:
        return pd.DataFrame()

    out_cols = list(df.columns) + ['catalyst_status']
    out_df = pd.DataFrame(final_rows, columns=[c for c in out_cols if any(c in r for r in final_rows)])
    array_cols = [c for c in time_varying if c in out_df.columns]
    out_df = out_df.apply(
        _align_row_array_lengths,
        axis=1,
        array_cols=array_cols,
        base_col=leach_col,
    )
    if leach_col in out_df.columns:
        out_df[leach_col] = out_df[leach_col].apply(
            lambda v: np.rint(np.asarray(v, dtype=float)) if isinstance(v, np.ndarray) else v
        )

    return out_df


def _add_transition_and_saturation(out_df, raw_df, group_cols):
    if out_df.empty or 'leach_duration_days' not in out_df.columns:
        return out_df

    out_df = out_df.copy()
    leach_col = 'leach_duration_days'
    target_col = 'cu_recovery_%'
    cat_col = 'cumulative_catalyst_addition_kg_t'

    out_df['transition_time'] = 0.0
    for idx, row in out_df.iterrows():
        raw_group = raw_df.copy()
        for col in group_cols:
            raw_group = raw_group[raw_group[col] == row.get(col)]

        cat_days_raw = raw_group[leach_col][pd.to_numeric(raw_group.get(cat_col), errors='coerce') > 0] if cat_col in raw_group.columns else pd.Series(dtype=float)
        if row.get('catalyst_status') == 'with_catalyst':
            out_df.at[idx, 'transition_time'] = (
                float(cat_days_raw.min()) if not cat_days_raw.empty
                else float(pd.to_numeric(raw_group.get(leach_col), errors='coerce').max()) if leach_col in raw_group.columns and not raw_group.empty
                else 0.0
            )
        else:
            ld = row.get(leach_col)
            cu = row.get(target_col)
            if isinstance(ld, np.ndarray) and isinstance(cu, np.ndarray):
                valid_idx = np.flatnonzero(np.isfinite(cu))
                out_df.at[idx, 'transition_time'] = (
                    float(ld[valid_idx[-1]]) if valid_idx.size > 0 else 0.0
                )
            else:
                out_df.at[idx, 'transition_time'] = (
                    float(ld[-1]) if isinstance(ld, np.ndarray) and ld.size > 0 else 0.0
                )
    out_df['transition_time'] = out_df['transition_time'].round(0)

    if all(c in out_df.columns for c in ['feed_mass_kg', 'column_inner_diameter_m']):
        irrigation_rate = pd.Series(np.nan, index=out_df.index, dtype=float)
        if 'irrigation_rate_l_h_m2' in out_df.columns:
            irrigation_rate = out_df['irrigation_rate_l_h_m2'].apply(_extract_numeric_scalar)
        if 'irrigation_rate_l_m2_h' in out_df.columns:
            fallback_irrigation = out_df['irrigation_rate_l_m2_h'].apply(_extract_numeric_scalar)
            irrigation_rate = irrigation_rate.fillna(fallback_irrigation)

        denom = (
            ((pd.to_numeric(out_df['column_inner_diameter_m'], errors='coerce') / 2) ** 2)
            * np.pi
            * irrigation_rate
            * 24
        )
        out_df['catalyst_saturation_inside_column_day'] = np.nan
        wc_mask = out_df['catalyst_status'] == 'with_catalyst'
        out_df.loc[wc_mask, 'catalyst_saturation_inside_column_day'] = (
            (pd.to_numeric(out_df.loc[wc_mask, 'feed_mass_kg'], errors='coerce') * 0.08)
            / denom.loc[wc_mask]
        ).replace([np.inf, -np.inf], np.nan).round(0)

    return out_df


def _add_derived_arrays(df, raw_df, pair_mode='averaged'):
    if df.empty or 'leach_duration_days' not in df.columns:
        return df

    df = df.copy()
    df['separation_adjusted_days'] = pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    df['delta_cu_rec'] = pd.Series([np.nan] * len(df), index=df.index, dtype=object)

    control_baselines = {}
    if pair_mode == 'column' and not raw_df.empty:
        for sample_id, sample_raw in raw_df.groupby('project_sample_id'):
            control_raw = sample_raw[pd.to_numeric(sample_raw.get('cumulative_catalyst_addition_kg_t'), errors='coerce') == 0]
            if control_raw.empty:
                continue
            shared_grid = _build_weekly_grid(sample_raw, leach_col='leach_duration_days')
            if shared_grid.size == 0:
                continue
            baseline_arrays = _build_weekly_arrays_for_subset(
                control_raw,
                shared_grid,
                [c for c in DYNAMIC_ARRAY_COLUMNS if c in raw_df.columns],
                leach_col='leach_duration_days',
                target_col='cu_recovery_%',
                test_id_col='project_col_id',
            )
            control_baselines[sample_id] = baseline_arrays.get('cu_recovery_%')

    for sample_id, grp in df.groupby('project_sample_id'):
        averaged_ctl_row = grp[grp['project_col_id'] == 'Control']

        for idx, row in grp.iterrows():
            leach = row.get('leach_duration_days')
            cu = row.get('cu_recovery_%')
            if not isinstance(leach, (list, np.ndarray)) or not isinstance(cu, (list, np.ndarray)):
                continue

            leach = np.asarray(leach, dtype=float)
            cu = _pad_or_truncate_array(cu, leach.size)
            if leach.size == 0:
                continue

            if row.get('catalyst_status') != 'with_catalyst':
                df.at[idx, 'separation_adjusted_days'] = np.full(leach.size, np.nan, dtype=float)
                df.at[idx, 'delta_cu_rec'] = np.full(leach.size, np.nan, dtype=float)
                continue

            cu_mask = np.isfinite(cu)
            transition_time = pd.to_numeric(pd.Series([row.get('transition_time')]), errors='coerce').iloc[0]
            saturation_time = pd.to_numeric(pd.Series([row.get('catalyst_saturation_inside_column_day')]), errors='coerce').iloc[0]
            sep_days = leach - float(transition_time) - float(saturation_time)
            sep_days[~cu_mask] = np.nan
            df.at[idx, 'separation_adjusted_days'] = sep_days

            delta = np.full(leach.size, np.nan, dtype=float)
            if pair_mode == 'averaged' and not averaged_ctl_row.empty:
                ctl_cu = averaged_ctl_row['cu_recovery_%'].iloc[0]
            else:
                ctl_cu = control_baselines.get(sample_id)

            if isinstance(ctl_cu, (list, np.ndarray)):
                ctl_cu = _pad_or_truncate_array(ctl_cu, leach.size)
                valid = np.isfinite(cu) & np.isfinite(ctl_cu)
                delta[valid] = cu[valid] - ctl_cu[valid]
                delta = _repair_monotonic_increase(delta, gate_mask=valid)
                delta[~valid] = np.nan

            df.at[idx, 'delta_cu_rec'] = delta

    return df


def _serialize_arrays_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Serialize numpy-array cells to JSON strings for CSV export.
    Scalar cells are left unchanged.
    Arrays are stored as "[v1, v2, v3, ...]" so they can be loaded back with
    json.loads() or ast.literal_eval().
    """
    import json
    df = df.copy()
    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        if isinstance(sample.iloc[0], np.ndarray):
            df[col] = df[col].apply(
                lambda v: json.dumps(
                    [round(float(x), 4) if np.isfinite(x) else None
                     for x in np.asarray(v, dtype=float)]
                ) if isinstance(v, np.ndarray) else v
            )
    return df


def _reorder_averaged_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order columns by source dataset, alphabetically within each source group.
    """
    if df.empty:
        return df

    identifier_cols = [
        'project_name',
        'project_sample_id',
        'project_col_id',
        'catalyst_status',
        'project_sample_id_reactormatch',
    ]

    time_series_cols = sorted([
        col for col in [
            'cu_recovery_%',
            'cumulative_catalyst_addition_kg_t',
            'cumulative_lixiviant_flowthrough_l',
            'cumulative_lixiviant_m3_t',
            'delta_cu_rec',
            'feed_flowrate_ml_min',
            'feed_orp_mv_ag_agcl',
            'irrigation_rate_l_h_m2',
            'leach_duration_days',
            'pls_fe_ii_mg_l',
            'pls_fe_iii_mg_l',
            'raff_assay_fe_ii_mg_l',
            'raff_assay_fe_iii_mg_l',
            'separation_adjusted_days',
        ]
        if col in df.columns
    ])

    chemchar_cols = sorted([col for col in COLS_TO_MATCH_CHEMCHAR if col in df.columns])

    column_summary_cols = sorted([col for col in COLS_TO_MATCH_COLUMN_SUMMARY if col in df.columns])

    modal_qemscan_overlap = set(COLS_TO_MATCH_QEMSCAN)
    modal_mineralogy_cols = sorted([
        col for col in COL_TO_MATCH_MINERALOGY
        if col in df.columns and col not in modal_qemscan_overlap
    ])

    qemscan_cols = sorted([col for col in COLS_TO_MATCH_QEMSCAN if col in df.columns])

    grouped_mineralogy_cols = sorted([col for col in df.columns if col.startswith('grouped_')])

    derived_cols = sorted([
        col for col in [
            'catalyst_saturation_inside_column_day',
            'copper_oxides_equivalent',
            'copper_primary_sulfides_equivalent',
            'copper_secondary_sulfides_equivalent',
            'copper_sulfides_equivalent',
            'cu:fe',
            'fe:cu',
            'transition_time',
        ]
        if col in df.columns
    ])

    ordered = []
    for group in [
        identifier_cols,
        time_series_cols,
        chemchar_cols,
        column_summary_cols,
        modal_mineralogy_cols,
        qemscan_cols,
        grouped_mineralogy_cols,
        derived_cols,
    ]:
        for col in group:
            if col in df.columns and col not in ordered:
                ordered.append(col)

    remaining = sorted([col for col in df.columns if col not in ordered])
    return df[ordered + remaining]


def run(step5: dict = None, intermediate_dir: Path = INTERMEDIATE_DIR) -> dict:
    """
    Produce averaged per-sample outputs for Power BI.
    If step5 is provided, uses its catcontrol CSV; otherwise reads from disk.
    """
    intermediate_dir.mkdir(exist_ok=True)

    # ── 1. Load inputs ─────────────────────────────────────────────────────────
    if step5 is not None and 'df_model_catcontrol_projects' in step5:
        df_columns_raw = step5["df_model_catcontrol_projects"].copy()
    else:
        # Fallback: read from pipeline/outputs/ (never reads from external folders)
        outputs_dir = intermediate_dir.parent / "outputs"
        data_path = str(outputs_dir / "df_model_catcontrol_projects_with_reactors_fit.csv")
        print(f"[step_06] Loading columns data from {data_path}...")
        df_columns_raw = pd.read_csv(data_path, sep=',', low_memory=False)

    # dataset_per_sample_reactor_complete.csv is generated by step_03.
    # Read from step_03's return dict if available; otherwise from pipeline/outputs/.
    if step5 is not None and 'df_reactor_complete' in (step5.get('_step3_passthrough') or {}):
        df_reactors_raw = step5['_step3_passthrough']['df_reactor_complete'].copy()
        print(f"[step_06] Using reactor data from step_03: {len(df_reactors_raw):,} rows")
    else:
        outputs_dir   = intermediate_dir.parent / "outputs"
        reactors_path = str(outputs_dir / "dataset_per_sample_reactor_complete.csv")
        try:
            df_reactors_raw = pd.read_csv(reactors_path, sep=',', low_memory=False)
            print(f"[step_06] Loaded reactor data from outputs/: {len(df_reactors_raw):,} rows")
        except FileNotFoundError:
            print(f"[step_06] ⚠ dataset_per_sample_reactor_complete.csv not found — skipping reactor matching")
            df_reactors_raw = pd.DataFrame(columns=['project_sample_id'])

    # ── 2. Apply column filters ────────────────────────────────────────────────
    print("[step_06] Applying column filters...")
    df_columns_filtered = apply_column_filters(df_columns_raw)

    # ── 3. Prepare reactors ────────────────────────────────────────────────────
    df_reactors_filtered = df_reactors_raw.copy()
    if not df_reactors_filtered.empty:
        if 'project_sample_id' in df_reactors_filtered.columns:
            # Duplicate UGM2 for coarse
            dup = df_reactors_filtered[df_reactors_filtered['project_sample_id'] == '007ajettiprojectfile_elephant_ugm2_rthead'].copy()
            dup['project_sample_id'] = '007ajettiprojectfile_elephant_ugm2_rthead_coarse'
            df_reactors_filtered = pd.concat([df_reactors_filtered, dup], ignore_index=True)
        # Fill and filter temperature
        if 'temp_(c)_mean' in df_reactors_filtered.columns:
            df_reactors_filtered['temp_(c)_mean'] = df_reactors_filtered['temp_(c)_mean'].fillna(25.0)
            df_reactors_filtered = df_reactors_filtered[df_reactors_filtered['temp_(c)_mean'] <= 40.0]

    # ── 4. Add match keys ──────────────────────────────────────────────────────
    print("[step_06] Adding match keys...")
    if 'project_sample_id' in df_columns_filtered.columns:
        df_columns_filtered, df_reactors_filtered = add_match_keys(
            df_columns_filtered, df_reactors_filtered, MATCH_DICT
        )

    # ── 5. Produce weekly outputs ──────────────────────────────────────────────
    print("[step_06] Producing weekly averaged output (one row per sample × catalyst status)...")
    df_averaged = prepare_averaged(
        df_columns_filtered,
        fill_noncat_averages=False,
        dropna_subset=False,
    )
    print("[step_06] Producing weekly per-column output (one row per original project_col_id)...")
    df_weekly = prepare_weekly_by_column(
        df_columns_filtered,
        dropna_subset=False,
    )

    if df_averaged is None or df_averaged.empty:
        print("[step_06] ⚠ No averaged data produced.")
        df_averaged = pd.DataFrame()
    if df_weekly is None or df_weekly.empty:
        print("[step_06] ⚠ No per-column weekly data produced.")
        df_weekly = pd.DataFrame()

    # ── 6. Build derived arrays and validate both exports ─────────────────────
    if not df_averaged.empty:
        df_averaged = _add_transition_and_saturation(
            df_averaged,
            df_columns_filtered,
            group_cols=['project_sample_id'],
        )
        df_averaged = _add_derived_arrays(
            df_averaged,
            df_columns_filtered,
            pair_mode='averaged',
        )
        array_cols_avg = _present_array_columns(
            df_averaged,
            extra_cols=['separation_adjusted_days', 'delta_cu_rec'],
        )
        df_averaged = df_averaged.apply(
            _align_row_array_lengths,
            axis=1,
            array_cols=array_cols_avg,
            base_col='leach_duration_days',
        )
        contract_issues = _validate_array_contract(df_averaged, array_cols_avg)
        gate_issues = _validate_interpolation_gate(
            df_averaged,
            [col for col in array_cols_avg if col != 'leach_duration_days'],
        )
        if contract_issues or gate_issues:
            issue_preview = (contract_issues + gate_issues)[:10]
            raise ValueError(
                "[step_06] Averaged array alignment validation failed:\n" + "\n".join(issue_preview)
            )
        df_averaged = _reorder_averaged_columns(df_averaged)

    if not df_weekly.empty:
        df_weekly = _add_transition_and_saturation(
            df_weekly,
            df_columns_filtered,
            group_cols=['project_sample_id', 'project_col_id'],
        )
        df_weekly = _add_derived_arrays(
            df_weekly,
            df_columns_filtered,
            pair_mode='column',
        )
        array_cols_weekly = _present_array_columns(
            df_weekly,
            extra_cols=['separation_adjusted_days', 'delta_cu_rec'],
        )
        df_weekly = df_weekly.apply(
            _align_row_array_lengths,
            axis=1,
            array_cols=array_cols_weekly,
            base_col='leach_duration_days',
        )
        contract_issues = _validate_array_contract(
            df_weekly,
            array_cols_weekly,
            pair_control_id=None,
            pair_catalyzed_id=None,
        )
        gate_issues = _validate_interpolation_gate(
            df_weekly,
            [col for col in array_cols_weekly if col != 'leach_duration_days'],
        )
        if contract_issues or gate_issues:
            issue_preview = (contract_issues + gate_issues)[:10]
            raise ValueError(
                "[step_06] Weekly array alignment validation failed:\n" + "\n".join(issue_preview)
            )
        df_weekly = _reorder_averaged_columns(df_weekly)

    # ── 7. Serialize arrays to JSON strings for CSV export ────────────────────
    # Arrays are stored as "[v1, v2, ...]" strings — loadable via json.loads().
    # This preserves the full weekly time-series in every cell.
    if not df_averaged.empty:
        df_averaged_csv = _serialize_arrays_for_csv(df_averaged)
        print(f"[step_06] Averaged rows: {len(df_averaged_csv)} | "
              f"Array cols: {sum(1 for c in DYNAMIC_ARRAY_COLUMNS if c in df_averaged_csv.columns)}")
    else:
        df_averaged_csv = df_averaged
    if not df_weekly.empty:
        df_weekly_csv = _serialize_arrays_for_csv(df_weekly)
        print(f"[step_06] Weekly rows: {len(df_weekly_csv)} | "
              f"Array cols: {sum(1 for c in DYNAMIC_ARRAY_COLUMNS if c in df_weekly_csv.columns)}")
    else:
        df_weekly_csv = df_weekly

    # ── 8. Save — outputs stay inside pipeline/ only ─────────────────────────
    outputs_dir = intermediate_dir.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    save_intermediate(
        df_averaged_csv if not df_averaged_csv.empty else pd.DataFrame(),
        str(intermediate_dir / "step_06_leaching_performance_weekly_averaged.csv"),
        "leaching_performance_weekly_averaged"
    )
    save_intermediate(
        df_weekly_csv if not df_weekly_csv.empty else pd.DataFrame(),
        str(intermediate_dir / "step_06_leaching_performance_weekly.csv"),
        "leaching_performance_weekly"
    )
    if not df_averaged_csv.empty:
        out_path_averaged = str(outputs_dir / "leaching_performance_weekly_averaged.csv")
        df_averaged_csv.to_csv(out_path_averaged, index=False)
        print(f"  → {out_path_averaged}")
    if not df_weekly_csv.empty:
        out_path_weekly = str(outputs_dir / "leaching_performance_weekly.csv")
        df_weekly_csv.to_csv(out_path_weekly, index=False)
        print(f"  → {out_path_weekly}")

    print("[step_06] Done.")
    return {
        "df_averaged_csv": df_averaged_csv,
        "df_weekly_csv": df_weekly_csv,
    }
