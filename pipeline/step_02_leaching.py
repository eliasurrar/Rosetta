"""
step_02_leaching.py — Build df_leaching_performance with all matched features.

Corresponds to: data_for_rosetta.py

Key steps:
  1. Build project_col_id and project_sample_id from leaching columns
  2. Match chemchar, column_summary, mineralogy, qemscan by project_sample_id / project_col_id
  3. Apply special treatments (015 iron, 015 6in/8in split, etc.)
  4. Export df_leaching_performance

Outputs:
  df_leaching_performance : row-level leaching data with all matched features
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PATHS, COL_TO_SAMPLE_ID_MAP, COL_ID_RENAME_MAP,
    COLS_TO_MATCH_CHEMCHAR, COLS_TO_MATCH_COLUMN_SUMMARY,
    COL_TO_MATCH_MINERALOGY, COLS_TO_MATCH_QEMSCAN,
    MINERALOGY_ID_MAP, MINERALOGY_SAMPLE_MAP, MINERALOGY_DUPLICATES,
)
from utils import (
    normalize_dataframe_values, normalize_and_replace,
    apply_exact_replacement, convert_cols_to_numeric,
    ensure_object_col, save_intermediate, save_to_paths,
)

INTERMEDIATE_DIR = Path(__file__).parent / "intermediate"


def _build_project_col_id(df: pd.DataFrame) -> pd.DataFrame:
    """Build project_col_id and col_name from project_name + sheet_name."""
    df = df.copy()
    df['project_col_id'] = normalize_dataframe_values(
        df['project_name'].str.replace('-', '') + '_' +
        df['sheet_name'].str.replace('-', '')
    )
    df['col_name'] = normalize_dataframe_values(df['sheet_name'].str.replace('-', ''))
    return df


def _apply_sample_id_map(series: pd.Series, mapping: dict) -> pd.Series:
    """Apply exact-match mapping; unmapped values stay as-is."""
    series = series.astype(object).copy()
    for pattern, replacement in mapping.items():
        escaped = re.escape(pattern)
        series = series.str.replace(f'^{escaped}$', replacement, regex=True)
    return series


def _prepare_mineralogy_modals_for_matching(df_mineralogy: pd.DataFrame) -> pd.DataFrame:
    """
    Bring processed modal mineralogy onto canonical project_sample_id values
    before matching individual modal minerals into leaching rows.
    """
    df_mineralogy = df_mineralogy.copy()
    if 'project_sample_id' not in df_mineralogy.columns:
        df_mineralogy['project_sample_id'] = '0'

    mask_zero = df_mineralogy['project_sample_id'].astype(str).isin(['0', '', 'nan', 'None'])

    if 'project_name' in df_mineralogy.columns and 'sheet_name' in df_mineralogy.columns:
        for (proj, start_cell), new_id in MINERALOGY_ID_MAP.items():
            m = (
                mask_zero
                & (df_mineralogy['project_name'] == proj)
                & (df_mineralogy['sheet_name'].astype(str) == start_cell)
            )
            df_mineralogy.loc[m, 'project_sample_id'] = new_id

    if 'project_name' in df_mineralogy.columns and 'sample' in df_mineralogy.columns:
        for (proj, sample_val), new_id in MINERALOGY_SAMPLE_MAP.items():
            m = (
                mask_zero
                & (df_mineralogy['project_name'] == proj)
                & (df_mineralogy['sample'].astype(str) == sample_val)
            )
            df_mineralogy.loc[m, 'project_sample_id'] = new_id

    for source_id, new_id in MINERALOGY_DUPLICATES:
        dup_rows = df_mineralogy[df_mineralogy['project_sample_id'] == source_id].copy()
        if not dup_rows.empty:
            dup_rows['project_sample_id'] = new_id
            df_mineralogy = pd.concat([df_mineralogy, dup_rows], ignore_index=True)

    df_mineralogy.loc[
        df_mineralogy['project_sample_id'] == '015_jetti_project_file_amcf',
        'project_sample_id'
    ] = '015_jetti_project_file_amcf_6in'

    df_mineralogy = df_mineralogy[
        ~df_mineralogy['project_sample_id'].astype(str).isin(['0', '', 'nan', 'None'])
    ].copy()

    return df_mineralogy


def run(step1: dict, intermediate_dir: Path = INTERMEDIATE_DIR) -> dict:
    """
    Build df_leaching_performance with all feature matches.
    """
    intermediate_dir.mkdir(exist_ok=True)
    db = PATHS["db_python"]

    df_raw          = step1["df_leaching_raw"].copy()
    df_chemchar     = step1["df_chemchar"].copy()
    df_col_summary  = step1["df_column_summary"].copy()
    df_mineralogy   = step1["df_mineralogy_modals"].copy()
    df_qemscan_raw  = step1["df_qemscan_raw"].copy()

    print("[step_02] Building leaching performance dataset...")

    # ── 1. Merge terminated projects (already done in step_01) ────────────────
    df = df_raw.copy()

    # ── 2. Build cu_recovery_% ─────────────────────────────────────────────────
    df['cu_recovery_%'] = df.get('cu_t_recovery_%', pd.Series(dtype=float)).fillna(
        df.get('cu_recovery', pd.Series(dtype=float))
    )

    # ── 3. Tiger ROM: fix sheet_name separator ─────────────────────────────────
    df['sheet_name'] = df['sheet_name'].str.replace('-', '_')

    # ── 4. ORP consolidation ────────────────────────────────────────────────────
    # Vancouver projects report ORP under three different column names.
    # Fill priority: standard → adjusted (corrected) → combined → ENH conversion
    if 'feed_orp_adjusted_mv_ag_agcl' in df.columns:
        df['feed_orp_mv_ag_agcl'] = df['feed_orp_mv_ag_agcl'].fillna(
            pd.to_numeric(df['feed_orp_adjusted_mv_ag_agcl'], errors='coerce')
        )
    if 'feed_orp_combined_mv_ag_agcl' in df.columns:
        df['feed_orp_mv_ag_agcl'] = df['feed_orp_mv_ag_agcl'].fillna(
            pd.to_numeric(df['feed_orp_combined_mv_ag_agcl'], errors='coerce')
        )
    # Chilean projects: convert ENH-scale ORP to Ag/AgCl scale (subtract 223 mV)
    if 'feed_orp_mv_enh' in df.columns:
        df['feed_orp_mv_ag_agcl'] = df['feed_orp_mv_ag_agcl'].fillna(
            pd.to_numeric(df['feed_orp_mv_enh'], errors='coerce') - 223.0
        )

    # ── 5. Ensure dynamic columns exist ───────────────────────────────────────
    for col in ['feed_orp_mv_ag_agcl','feed_flowrate_ml_min','irrigation_rate_l_h_m2',
                'raff_assay_fe_ii_mg_l','raff_assay_fe_iii_mg_l',
                'pls_fe_ii_mg_l','pls_fe_iii_mg_l','cumulative_lixiviant_m3_t']:
        if col not in df.columns:
            df[col] = np.nan

    if 'irrigation_rate_l_m2_h' in df.columns:
        df['irrigation_rate_l_h_m2'] = pd.to_numeric(df['irrigation_rate_l_h_m2'], errors='coerce').fillna(
            pd.to_numeric(df['irrigation_rate_l_m2_h'], errors='coerce')
        )

    dynamic_cols = ['feed_orp_mv_ag_agcl','feed_flowrate_ml_min','irrigation_rate_l_h_m2',
                    'raff_assay_fe_ii_mg_l','raff_assay_fe_iii_mg_l',
                    'pls_fe_ii_mg_l','pls_fe_iii_mg_l','cumulative_lixiviant_m3_t']
    df = convert_cols_to_numeric(df, dynamic_cols)

    # ── 6. Build project_col_id ────────────────────────────────────────────────
    df = _build_project_col_id(df)

    # ── 7. Apply project_sample_id mapping ────────────────────────────────────
    # Start from project_col_id, apply consolidated replacement dict
    df['project_sample_id'] = _apply_sample_id_map(df['project_col_id'].copy(), COL_TO_SAMPLE_ID_MAP)

    # Rename project_col_id for old terminated projects
    df['project_col_id'] = _apply_sample_id_map(df['project_col_id'], COL_ID_RENAME_MAP)

    # Re-order: bring id cols to front
    id_cols = ['col_name', 'project_col_id', 'project_sample_id']
    other = [c for c in df.columns if c not in id_cols]
    df = df[[other[0], other[1], other[2]] + id_cols + other[3:]] if len(other) >= 3 else df

    # ── 8. MATCH STEP: chemical characterisation ───────────────────────────────
    print("[step_02]   Matching chemchar...")
    chemchar_dict = (
        df_chemchar
        .drop_duplicates(subset='project_sample_id', keep='first')
        .set_index('project_sample_id')[COLS_TO_MATCH_CHEMCHAR]
        .to_dict('index')
    )

    def _get(key, d, col):
        return d[key][col] if key in d and col in d[key] else None

    for col in COLS_TO_MATCH_CHEMCHAR:
        df[col] = df['project_sample_id'].map(lambda x, c=col: _get(x, chemchar_dict, c))

    # ── 9. MATCH STEP: column summary (by project_col_id) ─────────────────────
    print("[step_02]   Matching column summary...")
    col_summary_avail = [c for c in COLS_TO_MATCH_COLUMN_SUMMARY if c in df_col_summary.columns]
    col_summary_dict = (
        df_col_summary
        .drop_duplicates(subset='project_col_id', keep='first')
        .set_index('project_col_id')[col_summary_avail]
        .to_dict('index')
    )
    for col in col_summary_avail:
        df[col] = df['project_col_id'].map(lambda x, c=col: _get(x, col_summary_dict, c))

    # ── 10. MATCH STEP: mineralogy modals (by project_sample_id) ──────────────
    print("[step_02]   Matching mineralogy modals...")
    # The raw SpkData mineralogy CSV does not have project_sample_id.
    # Load the processed version from db_python (written by the original ETL)
    # which does have it. Fall back gracefully if not yet generated.
    mineral_cols_avail = []
    db = PATHS["db_python"]
    mineralogy_processed_path = f"{db}/df_mineralogy_modals.csv"
    try:
        df_min_processed = pd.read_csv(mineralogy_processed_path, index_col=0, low_memory=False)
        if 'project_sample_id' in df_min_processed.columns:
            df_min_processed = _prepare_mineralogy_modals_for_matching(df_min_processed)
            avail_mineral_ids = (
                df_min_processed[df_min_processed['project_sample_id'].astype(str) != '0']
                ['project_sample_id'].unique()
            )
            mineral_cols_avail = [c for c in COL_TO_MATCH_MINERALOGY if c in df_min_processed.columns]
            if mineral_cols_avail:
                mineralogy_dict = (
                    df_min_processed[df_min_processed['project_sample_id'].isin(avail_mineral_ids)]
                    .drop_duplicates(subset='project_sample_id', keep='first')
                    .set_index('project_sample_id')[mineral_cols_avail]
                    .to_dict('index')
                )
                for col in mineral_cols_avail:
                    df[col] = df['project_sample_id'].map(lambda x, c=col: _get(x, mineralogy_dict, c))
                print(f"[step_02]   Mineralogy matched {len(avail_mineral_ids)} samples ({len(mineral_cols_avail)} cols)")
        else:
            print("[step_02]   ⚠ Processed mineralogy has no project_sample_id — skipping mineralogy match")
    except FileNotFoundError:
        print(f"[step_02]   ⚠ {mineralogy_processed_path} not found — skipping mineralogy match (run original ETL first)")

    # ── 11. MATCH STEP: QEMSCAN exposure data ─────────────────────────────────
    print("[step_02]   Matching QEMSCAN...")
    df_qemscan = df_qemscan_raw[df_qemscan_raw.get('sample', pd.Series(dtype=str)) == 'Combined'].copy() \
        if 'sample' in df_qemscan_raw.columns else df_qemscan_raw.copy()

    # Compute QEMSCAN derived columns if lib exposure columns exist
    lib_cols = ['copper_sulphides_lib_exposed','copper_sulphides_lib_50-80%_exposed',
                'copper_sulphides_lib_30-50%_exposed','copper_sulphides_lib_20-30%_exposed',
                'copper_sulphides_lib_10-20%_exposed','copper_sulphides_lib_0-10%_exposed',
                'copper_sulphides_lib_locked']
    lib_avail = [c for c in lib_cols if c in df_qemscan.columns]
    if lib_avail:
        df_qemscan['cus_total_sum'] = df_qemscan[lib_avail].sum(axis=1, skipna=True)
        exp50 = [c for c in ['copper_sulphides_lib_exposed','copper_sulphides_lib_50-80%_exposed'] if c in df_qemscan.columns]
        lock30 = [c for c in ['copper_sulphides_lib_20-30%_exposed','copper_sulphides_lib_10-20%_exposed','copper_sulphides_lib_0-10%_exposed','copper_sulphides_lib_locked'] if c in df_qemscan.columns]
        df_qemscan['cus_exposed_50pct_sum']  = df_qemscan[exp50].sum(axis=1, skipna=True) if exp50 else np.nan
        df_qemscan['cus_locked_30pct_sum']   = df_qemscan[lock30].sum(axis=1, skipna=True) if lock30 else np.nan
        df_qemscan['cus_exposed_50pct_normalized'] = df_qemscan['cus_exposed_50pct_sum'] / df_qemscan['cus_total_sum']
        df_qemscan['cus_locked_30pct_normalized']  = df_qemscan['cus_locked_30pct_sum']  / df_qemscan['cus_total_sum']

    qemscan_avail = [c for c in COLS_TO_MATCH_QEMSCAN if c in df_qemscan.columns]
    if 'project_sample_id' in df_qemscan.columns and qemscan_avail:
        qemscan_dict = (
            df_qemscan
            .drop_duplicates(subset='project_sample_id', keep='first')
            .set_index('project_sample_id')[qemscan_avail]
            .to_dict('index')
        )
        for col in qemscan_avail:
            df[col] = df['project_sample_id'].map(lambda x, c=col: _get(x, qemscan_dict, c))

    # ── 12. SPECIAL TREATMENT: 015 iron recirculation ─────────────────────────
    # MANUAL FILTER: Project 015 is the only one with iron recirculated without SX
    for suffix in ['cu','fe','mg','al']:
        src = f'pls_{suffix}_mg_l'
        tgt = f'raff_assay_{suffix}_mg_l'
        if src in df.columns and tgt in df.columns:
            mask_015 = df['project_name'] == '015 Jetti Project File'
            df.loc[mask_015, tgt] = df.loc[mask_015, tgt].fillna(df.loc[mask_015, src])

    # ── 13. SPECIAL TREATMENT: 015 6in/8in sample split ──────────────────────
    # MANUAL FILTER: Columns 11/6 are 8in; 12/7 are 6in (Ben K)
    mask_8in = df['project_col_id'].isin(['015_jetti_project_file_c_11', '015_jetti_project_file_c_6'])
    mask_6in = df['project_col_id'].isin(['015_jetti_project_file_c_12', '015_jetti_project_file_c_7'])
    df.loc[mask_8in, 'project_sample_id'] = '015_jetti_project_file_amcf_8in'
    df.loc[mask_6in, 'project_sample_id'] = '015_jetti_project_file_amcf_6in'

    # ── 14. Save — outputs stay inside pipeline/ only ─────────────────────────
    outputs_dir = intermediate_dir.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    save_intermediate(df, str(intermediate_dir / "step_02_leaching_performance.csv"), "leaching_performance")
    df.to_csv(str(outputs_dir / "df_leaching_performance.csv"), index=False)
    print(f"  → {outputs_dir / 'df_leaching_performance.csv'}")

    print(f"[step_02] Done. Rows: {len(df):,} | Columns: {len(df.columns)}")
    return {
        "df_leaching_performance": df,
    }
