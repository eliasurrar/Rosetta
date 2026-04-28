"""
step_05_model_datasets.py — Build ML training datasets.

Corresponds to: MLmodel_input_datasets.py

Steps:
  1. Filter leaching_performance to LEACHING_COLS_TO_KEEP
  2. Apply manual filters (stopped columns, catalyst zeroing, etc.)
  3. Merge mineralogy grouped data
  4. Build catalyzed/control/catcontrol splits
  5. preprocess_data (with all bug fixes)
  6. Merge with reactor fit stats (a, b, c parameters)
  7. Reorder columns and save final outputs
"""

import warnings, re
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PATHS, LEACHING_COLS_TO_KEEP, PREDICTORS_DICT, ID_COLS,
    MIN_THRESH, Y_RESULT, CATEGORICAL_FEATS, COLS_TO_FFILL, MANUAL_FILTERS,
    COL_TO_MATCH_MINERALOGY,
)
from utils import (
    convert_cols_to_numeric, is_numeric_col,
    protect_id_cols_dropna, filter_increasing_within_group,
    save_intermediate, save_to_paths,
)

INTERMEDIATE_DIR = Path(__file__).parent / "intermediate"

X_PREDICTORS        = list(PREDICTORS_DICT.keys())
RESTRICTIONS_CONTROL   = [v[0] for v in PREDICTORS_DICT.values()]
RESTRICTIONS_CATALYZED = [v[1] for v in PREDICTORS_DICT.values()]
RESTRICTIONS_1MODEL    = [v[2] for v in PREDICTORS_DICT.values()]


def _apply_manual_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all MANUAL_FILTER data exclusions (documented in config.MANUAL_FILTERS)."""
    df = df.copy()

    # MANUAL FILTER: remove_015_holdup_gre — keep only standard recovery
    if 'condition' in df.columns:
        df = df[df['condition'].isnull()]

    # MANUAL FILTER: stopped_columns_zaldivar_69_70
    mask = df['project_col_id'].isin(['jetti_project_file_zaldivar_scl_col69','jetti_project_file_zaldivar_scl_col70'])
    df.loc[mask] = df.loc[mask & (df['leach_duration_days'] < 1361)]

    # MANUAL FILTER: stopped_columns_leopard_rom
    mask = df['project_col_id'].isin(['jetti_project_file_leopard_scl_rom1','jetti_project_file_leopard_scl_rom2'])
    df.loc[mask] = df.loc[mask & ((df['leach_duration_days'] < 438) | (df['leach_duration_days'] > 444))]

    # MANUAL FILTER: stopped_columns_elephant_42_43
    mask = df['project_col_id'].isin(['jetti_project_file_elephant_scl_col42','jetti_project_file_elephant_scl_col43'])
    df.loc[mask] = df.loc[mask & (df['leach_duration_days'] < 1115)]

    # MANUAL FILTER: stopped_columns_toquepala
    mask = df['project_col_id'].isin(['jetti_project_file_toquepala_scl_col63','jetti_project_file_toquepala_scl_col64'])
    df.loc[mask] = df.loc[mask & (df['leach_duration_days'] < 952)]

    # MANUAL FILTER: stopped_columns_rm_1
    mask = df['project_col_id'] == 'jetti_project_file_rm_1'
    df.loc[mask] = df.loc[mask & ((df['leach_duration_days'] < 632) | (df['leach_duration_days'] > 850))]

    # MANUAL FILTER: zero_catalyst_control
    control_projects = [k for k, v in LEACHING_COLS_TO_KEEP.items() if v[3] == 'control']
    df.loc[df['project_col_id'].isin(control_projects), 'cumulative_catalyst_addition_kg_t'] = 0

    return df


def preprocess_data(
    df: pd.DataFrame,
    numerical_features: list,
    cols_can_have_zeros: list,
    id_cols: list,
    min_thresh: float,
    y_col: str,
    time_feature: str,
    cols_to_ffill: list,
    keep_all_rows: bool = False,
) -> pd.DataFrame:
    """
    Preprocess ML training dataframe with all bug fixes applied.

    Bug fixes:
    - ID cols protected from dropna(axis=1)
    - filter_increasing uses diff-mask not groupby.apply returning DataFrame
    - is_numeric_col uses pd.api.types
    """
    df = df.copy()

    # Convert numerical features to numeric
    for c in numerical_features:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Replace zeros with NaN (except protected cols)
    for c in df.columns:
        if c not in cols_can_have_zeros and c not in id_cols:
            df[c] = df[c].replace(0, np.nan)

    # BUG FIX: protect id_cols from dropna(axis=1)
    df = protect_id_cols_dropna(
        df, id_cols,
        thresh=int(len(df.columns) * min_thresh), axis=1
    )

    # Forward-fill within groups (only for increasing leach_duration_days)
    if time_feature in df.columns:
        for cf in [x for x in cols_to_ffill if x in df.columns]:
            try:
                result = df.groupby(
                    [c for c in id_cols if c in df.columns],
                    group_keys=False, sort=False
                ).apply(
                    lambda g, _cf=cf: g[_cf].where(g[time_feature].diff().ge(0))
                    .interpolate(method='linear', limit_direction='forward',
                                 limit_area='inside', limit=30)
                )
                df[cf] = result
            except Exception:
                pass

    # MANUAL FILTER: preprocess_drop_zero_recovery
    if not keep_all_rows and y_col in df.columns:
        df = df[df[y_col] > 0.0]

    # MANUAL FILTER: preprocess_require_increasing
    # BUG FIX: use diff-mask instead of groupby.apply returning DataFrame
    if not keep_all_rows and y_col in df.columns:
        df = filter_increasing_within_group(
            df, [c for c in id_cols if c in df.columns], y_col
        )

    return df


def run(step2: dict, step4: dict,
        intermediate_dir: Path = INTERMEDIATE_DIR) -> dict:
    intermediate_dir.mkdir(exist_ok=True)
    df_leaching   = step2["df_leaching_performance"].copy()
    df_min_grouped= step4["df_mineralogy_grouped"].copy()

    print("[step_05] Preparing project-facing dataset with maximal sample coverage...")

    # ── 1. Apply manual filters without restricting sample coverage ────────────
    df_leaching = _apply_manual_filters(df_leaching)

    # ── 3. Merge mineralogy grouped ────────────────────────────────────────────
    grouped_cols = [c for c in df_min_grouped.columns if c.startswith('grouped_')]
    result_df = df_leaching.merge(
        df_min_grouped[['project_sample_id'] + grouped_cols],
        on='project_sample_id', how='left'
    )
    print(f"  After mineralogy merge: {len(result_df):,} rows")

    # ── 5. MANUAL FILTER: mineralogy low-value zeroing ─────────────────────────
    low_threshold = 0.0005
    for col in ['pyrite','chlorite','bornite','enargite','fe_oxides']:
        if col in result_df.columns:
            result_df[col] = result_df[col].apply(lambda x: 0 if pd.notna(x) and x < low_threshold else x)
            result_df[col] = result_df[col].fillna(0)

    # ── 6. Build cols_can_have_zeros (from config + mineralogy) ────────────────
    modal_mineral_cols = [c for c in COL_TO_MATCH_MINERALOGY if c in result_df.columns]
    cols_can_have_zeros = ['cumulative_catalyst_addition_kg_t'] + grouped_cols + modal_mineral_cols

    # ── 7. Define numerical_features ──────────────────────────────────────────
    all_columns = list(set(result_df.columns))
    numerical_features = list(set(all_columns) - set(CATEGORICAL_FEATS) - set(ID_COLS))

    COLS_TO_IDENTIFY = ['project_name','project_col_id','project_sample_id']
    # ML training subset — only the model predictors (used for _prepare / model fitting)
    ml_cols = [c for c in (COLS_TO_IDENTIFY + X_PREDICTORS + Y_RESULT) if c in result_df.columns]
    # ALL columns — used for the _projects versions that go to step_06 and Power BI.
    # This preserves pls_orp_mv_ag_agcl, raff_assay_fe_mg_l, and any other
    # measurement columns that aren't ML predictors but are valuable for analysis.
    all_result_cols = list(result_df.columns)

    catalyst_status_lookup = (
        result_df
        .assign(_cat_value=pd.to_numeric(result_df.get('cumulative_catalyst_addition_kg_t', 0), errors='coerce'))
        .groupby('project_col_id', sort=False)['_cat_value']
        .max()
        .fillna(0)
    )
    catalyzed_col_ids = set(catalyst_status_lookup[catalyst_status_lookup > 0].index)
    control_col_ids   = set(catalyst_status_lookup[catalyst_status_lookup <= 0].index)

    # ML training versions use the narrow predictor-only selection
    df_cat = result_df[result_df['project_col_id'].isin(catalyzed_col_ids)][ml_cols].copy()
    df_ctl = result_df[result_df['project_col_id'].isin(control_col_ids)][ml_cols].copy()
    df_cat.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_ctl.replace([np.inf, -np.inf], np.nan, inplace=True)

    # _projects versions keep ALL columns for downstream visualization/averaging
    df_cat_projects = result_df[result_df['project_col_id'].isin(catalyzed_col_ids)].copy()
    df_ctl_projects = result_df[result_df['project_col_id'].isin(control_col_ids)].copy()
    df_catcontrol_projects = result_df.copy()
    for df_p in [df_cat_projects, df_ctl_projects, df_catcontrol_projects]:
        df_p.replace([np.inf, -np.inf], np.nan, inplace=True)

    save_intermediate(
        df_catcontrol_projects,
        str(intermediate_dir / "step_05_catcontrol_projects_prepreprocess.csv"),
        "catcontrol_projects_prepreprocess"
    )

    # ── 9. prepare_dataframe_to_train_model (simplified inline version) ────────
    def _prepare(df_in):
        try:
            df_tmp = df_in.copy()
            for c in list(set(df_tmp.columns) - set(CATEGORICAL_FEATS)):
                df_tmp[c] = pd.to_numeric(df_tmp[c], errors='coerce')
            # Drop sparse columns
            df_tmp = protect_id_cols_dropna(
                df_tmp, ID_COLS,
                thresh=int(len(X_PREDICTORS) * MIN_THRESH), axis=1
            )
            # Replace zeros
            for c in df_tmp.columns:
                if c not in cols_can_have_zeros and c not in ID_COLS:
                    df_tmp[c] = df_tmp[c].replace(0, np.nan)
            # Forward fill
            if 'leach_duration_days' in df_tmp.columns:
                for cf in [x for x in COLS_TO_FFILL if x in df_tmp.columns]:
                    id_present = [c for c in ID_COLS if c in df_tmp.columns]
                    if id_present:
                        result = df_tmp.groupby(id_present, group_keys=False, sort=False).apply(
                            lambda g, _cf=cf: g[_cf].where(g['leach_duration_days'].diff().ge(0))
                            .interpolate(method='linear', limit_direction='forward',
                                         limit_area='inside', limit=30)
                        )
                        df_tmp[cf] = result
            # Filter
            y_col = Y_RESULT[0]
            if y_col in df_tmp.columns:
                df_tmp = df_tmp[df_tmp[y_col] > 1.0]
            df_tmp = df_tmp.dropna(axis=1, how='all')
            df_tmp = df_tmp.dropna(axis=0, how='any')
            return df_tmp
        except Exception as e:
            print(f"  ⚠ prepare failed: {e}")
            return pd.DataFrame(columns=['no_columns'])

    df_cat_model = _prepare(df_cat.copy())
    df_ctl_model = _prepare(df_ctl.copy())
    df_catcontrol_model = pd.concat([df_cat_model, df_ctl_model], axis=0, ignore_index=True)

    # ── 10. preprocess_data on _projects ──────────────────────────────────────
    print("[step_05] Running preprocess_data on _projects versions...")
    def _preprocess_projects(df_in):
        return preprocess_data(
            df_in, numerical_features, cols_can_have_zeros,
            ID_COLS, MIN_THRESH, Y_RESULT[0],
            'leach_duration_days', COLS_TO_FFILL,
            keep_all_rows=True,
        )

    df_cat_projects = _preprocess_projects(df_cat_projects)
    df_ctl_projects = _preprocess_projects(df_ctl_projects)
    df_catcontrol_projects = _preprocess_projects(
        pd.concat([
            step2["df_leaching_performance"][step2["df_leaching_performance"]['project_col_id'].isin(catalyzed_col_ids | control_col_ids)],
        ], ignore_index=True)
    ) if False else _preprocess_projects(pd.concat([df_cat_projects, df_ctl_projects], axis=0, ignore_index=True))

    # ── 11. Merge mineralogy into _projects (fill any grouped gaps row-wise) ───
    mineral_lookup = (
        df_min_grouped[['project_sample_id'] + grouped_cols]
        .drop_duplicates(subset='project_sample_id', keep='first')
    ) if grouped_cols else pd.DataFrame(columns=['project_sample_id'])

    def _fill_grouped_mineralogy(df_proj):
        if df_proj.empty or 'project_sample_id' not in df_proj.columns or not grouped_cols:
            return df_proj
        merged = df_proj.merge(
            mineral_lookup,
            on='project_sample_id',
            how='left',
            suffixes=('', '_lookup'),
        )
        for col in grouped_cols:
            lookup_col = f'{col}_lookup'
            if col not in merged.columns and lookup_col in merged.columns:
                merged[col] = merged[lookup_col]
            elif lookup_col in merged.columns:
                merged[col] = merged[col].fillna(merged[lookup_col])
            merged.drop(columns=[lookup_col], inplace=True, errors='ignore')
        return merged

    df_cat_projects = _fill_grouped_mineralogy(df_cat_projects)
    df_ctl_projects = _fill_grouped_mineralogy(df_ctl_projects)
    df_catcontrol_projects = _fill_grouped_mineralogy(df_catcontrol_projects)

    # ── 13. Reorder columns ────────────────────────────────────────────────────
    first_cols = ['project_name','project_sample_id','project_col_id',
                  'leach_duration_days','cu_recovery_%','acid_soluble_%',
                  'cyanide_soluble_%','residual_cpy_%']
    all_catcontrol_cols = set(df_catcontrol_projects.columns)

    def _reorder(df_in, all_cols):
        fc = [c for c in first_cols if c in df_in.columns]
        other = sorted([c for c in df_in.columns if c not in fc])
        avail_order = fc + other
        return df_in[[c for c in avail_order if c in df_in.columns]]

    df_cat_projects        = _reorder(df_cat_projects, all_catcontrol_cols)
    df_ctl_projects        = _reorder(df_ctl_projects, all_catcontrol_cols)
    df_catcontrol_projects = _reorder(df_catcontrol_projects, all_catcontrol_cols)

    # ── 14. Save — outputs stay inside pipeline/ only ─────────────────────────
    # Catalyzed/control split CSVs saved to outputs/ for inspection.
    # Only df_model_catcontrol_projects flows to step_06 (the combined preprocessed version).
    outputs_dir = intermediate_dir.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    save_intermediate(df_catcontrol_projects, str(intermediate_dir / "step_05_catcontrol_projects.csv"), "catcontrol_projects")

    print("[step_05] Saving final model CSVs to pipeline/outputs/ ...")
    df_catcontrol_projects.to_csv(str(outputs_dir / "df_model_catcontrol_projects_with_reactors_fit.csv"), index=False)
    df_cat_projects.to_csv(       str(outputs_dir / "df_model_catalyzed_projects_with_reactors_fit.csv"),  index=False)
    df_ctl_projects.to_csv(       str(outputs_dir / "df_model_control_projects_with_reactors_fit.csv"),    index=False)
    print(f"  → {outputs_dir / 'df_model_catcontrol_projects_with_reactors_fit.csv'}")

    print(f"[step_05] Done. catcontrol_projects rows: {len(df_catcontrol_projects):,}")
    return {
        "df_model_catcontrol_projects": df_catcontrol_projects,
    }
