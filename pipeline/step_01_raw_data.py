"""
step_01_raw_data.py — Load and prepare all raw source tables.

Corresponds to: data_for_tableau.py (loading + filtering sections)

Outputs:
  df_leaching_raw      : raw leaching columns from SpkData
  df_column_summary    : column initial conditions (filtered)
  df_chemchar          : chemical characterisation (filtered, grouped by sample)
  df_mineralogy_modals : mineralogy modal percentages
  df_ac_summary        : acid consumption summary (filtered)
  df_separation        : Nelson separation data
  df_comments          : monthly comments
  df_maker_index       : project index with cat/control reactor IDs
  df_master_reactors   : catalyzed vs control reactor comparison table
"""

import os, re
import pandas as pd
import numpy as np
from pathlib import Path

# Add Rosetta to path for functions_general (fallback to pipeline utils)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PATHS, ACSUMMARY_SAMPLE_ID_FIXES, CHEMCHAR_SAMPLE_ID_FIXES,
    COL_TO_SAMPLE_ID_MAP, COL_ID_RENAME_MAP,
)
from utils import (
    normalize_and_replace, normalize_dataframe_values, normalize_series,
    apply_exact_replacement, apply_partial_replacement,
    convert_cols_to_numeric, ensure_object_col, save_intermediate,
)

# COLUMN_SUMMARY_SAMPLE_ID_FIXES may not be defined in config yet — default empty
try:
    from config import COLUMN_SUMMARY_SAMPLE_ID_FIXES
except ImportError:
    COLUMN_SUMMARY_SAMPLE_ID_FIXES = {}


INTERMEDIATE_DIR = Path(__file__).parent / "intermediate"


def run(intermediate_dir: Path = INTERMEDIATE_DIR) -> dict:
    """
    Load all source tables and apply normalisations.
    Returns a dict of dataframes.
    """
    intermediate_dir.mkdir(exist_ok=True)
    spk = PATHS["spkdata_csvs"]
    uef = PATHS["user_editable_files"]   # terminated + qemscan files live here

    print("[step_01] Loading raw source CSVs...")

    # ── 1. Primary source tables ───────────────────────────────────────────────
    df_merged        = pd.read_csv(f"{spk}/leaching_columns.csv",                              sep=',', low_memory=False)
    df_col_summary   = pd.read_csv(f"{spk}/dataset_column_summary.csv",                        sep=',', low_memory=False)
    df_bottles       = pd.read_csv(f"{spk}/dataset_rolling_bottles_detailed.csv",              sep=',', low_memory=False)
    df_reactors_raw  = pd.read_csv(f"{spk}/dataset_reactor_summary_detailed.csv",              sep=',', low_memory=False)
    df_chemchar_raw  = pd.read_csv(f"{spk}/dataset_characterization_summary.csv",              sep=',', low_memory=False)
    df_ac_summary_raw= pd.read_csv(f"{spk}/dataset_acid_consumption_summary_summaries.csv",    sep=',', low_memory=False)
    df_mineralogy_raw= pd.read_csv(f"{spk}/dataset_mineralogy_summary_modals.csv",             sep=',', low_memory=False)
    df_separation    = pd.read_csv(f"{spk}/dataset_nelson_separation.csv",                     sep=',', low_memory=False)
    df_comments      = pd.read_csv(f"{spk}/dataset_comments_monthly.csv",                      sep=',', low_memory=False)
    df_maker_index   = pd.read_excel(f"{PATHS['spkdata_inputs']}/input_dataframe_maker.xlsx",  sheet_name='project_index')

    # Drop unnamed index columns from maker_index
    df_maker_index = df_maker_index.loc[:, ~df_maker_index.columns.str.contains('^Unnamed')]

    # ── 2. User-editable files: terminated projects + QEMSCAN ─────────────────
    # All files in pipeline/user_editable_files/ — copy updated versions there.
    print("[step_01] Loading user-editable files (terminated projects + QEMSCAN)...")
    def _load_uef(filename, **kwargs):
        path = f"{uef}/{filename}"
        try:
            return pd.read_excel(path, **kwargs)
        except FileNotFoundError:
            print(f"  ⚠ {filename} not found in user_editable_files/ — returning empty DataFrame")
            return pd.DataFrame()

    df_term_leaching   = _load_uef("merged_terminated_projects_testDetails_MetallurgicalBalance.xlsx")
    df_term_chemchar   = _load_uef("dataset_characterization_summary_terminated_projects.xlsx")
    df_term_mineralogy = _load_uef("dataset_mineralogy_summary_modals_terminated_projects.xlsx")
    df_term_col_summary= _load_uef("df_column_summary_terminated_projects.xlsx")
    df_qemscan_raw     = _load_uef("df_qemscan_compilation.xlsx")

    # ── 3. Merge current + terminated ─────────────────────────────────────────
    df_merged        = pd.concat([df_term_leaching,   df_merged],          axis=0, ignore_index=True)
    df_col_summary   = pd.concat([df_term_col_summary, df_col_summary],    axis=0, ignore_index=True)
    df_chemchar_raw  = pd.concat([df_term_chemchar,    df_chemchar_raw],    axis=0, ignore_index=True)
    df_mineralogy_raw= pd.concat([df_term_mineralogy,  df_mineralogy_raw],  axis=0, ignore_index=True)

    # ── 4. Force numeric on known columns ─────────────────────────────────────
    pls_cols = ['pls_cu_mg_l','pls_fe_mg_l','pls_mg_mg_l','pls_al_mg_l',
                'pls_si_mg_l','pls_co_mg_l','pls_li_mg_l','pls_u_mg_l']
    df_merged = convert_cols_to_numeric(df_merged, pls_cols)

    sep_non_numeric = ['origin','project_name','sheet_name','start_cell']
    df_separation = convert_cols_to_numeric(
        df_separation,
        [c for c in df_separation.columns if c not in sep_non_numeric]
    )

    # ── 5. Column summary: fix col_name from index ─────────────────────────────
    # BUG FIX: col_name is float64 (all-NaN), assigning strings fails → cast first
    df_col_summary = ensure_object_col(df_col_summary, 'col_name')
    mask_underscore = df_col_summary['index'].str.startswith('_')
    df_col_summary.loc[mask_underscore, 'col_name'] = (
        df_col_summary.loc[mask_underscore, 'index'].str.lstrip('_')
    )
    df_col_summary['col_name'] = df_col_summary['col_name'].fillna(df_col_summary['index'])

    # Special: Tiger ROM uses '-' in sheet_name
    df_merged['sheet_name'] = df_merged['sheet_name'].str.replace('-', '_')

    # Special: 026 SecS → SS in column_summary index
    mask_026 = df_col_summary['project_name'] == '026 Jetti Project File'
    df_col_summary.loc[mask_026, 'index'] = df_col_summary.loc[mask_026, 'index'].str.replace('SecS', 'SS', regex=False)

    # ── 6. Column summary: build project_col_id and project_sample_id ─────────
    cols_col_summary = [
        'origin','project_name','index','sample_id','material_size_p80_in',
        'feed_head_cu_%','feed_head_fe_%','feed_head_mg_%','feed_head_al_%',
        'feed_head_co_%','feed_head_si_%','lixiviant_initial_ph',
        'lixiviant_initial_orp_mv','lixiviant_initial_cu_mg_l','lixiviant_initial_fe_mg_l',
        'catalyst_start_days_of_leaching','column_height_m','column_inner_diameter_m',
        'feed_mass_kg','irrigation_rate_l_m2_h','agglomeration_y_n','agglomeration_medium',
        'acid_in_agglomeration_kg_t','lixiviant_inoc_site_raff_syn_raff','inoculum_%',
        'aeration_y_n','aeration_dosage_l_min','catalyst_y_n','catalyst_dosage_mg_day',
        'catalyst_dosage_mg_l',
    ]
    avail = [c for c in cols_col_summary if c in df_col_summary.columns]
    df_col_summary_filt = df_col_summary[avail].copy()

    df_col_summary_filt['project_col_id'] = normalize_dataframe_values(
        df_col_summary_filt['project_name'].astype(str).str.replace('-','') + '_' +
        df_col_summary_filt['index'].astype(str).str.replace('-','')
    )
    df_col_summary_filt['col_name'] = normalize_dataframe_values(
        df_col_summary_filt['index'].astype(str).str.replace('-','')
    )

    # Numeric conversion for column summary
    non_num_cs = ['origin','project_name','index','sample_id','aeration_y_n',
                  'agglomeration_y_n','agglomeration_medium','catalyst_y_n',
                  'lixiviant_inoc_site_raff_syn_raff','project_col_id','col_name']
    df_col_summary_filt = convert_cols_to_numeric(
        df_col_summary_filt,
        [c for c in avail if c not in non_num_cs]
    )

    # Fix 011 crushed col IDs
    for old, new in [
        ('011_jetti_project_filecrushed_rm_5', '011_jetti_project_filecrushed_011rm_5'),
        ('011_jetti_project_filecrushed_rm_6', '011_jetti_project_filecrushed_011rm_6'),
        ('011_jetti_project_filecrushed_rm_7', '011_jetti_project_filecrushed_011rm_7'),
        ('011_jetti_project_filecrushed_rm_8', '011_jetti_project_filecrushed_011rm_8'),
    ]:
        df_col_summary_filt.loc[df_col_summary_filt['project_col_id'] == old, 'project_col_id'] = new

    # Keep column-summary IDs on the same naming scheme as leaching step_02.
    df_col_summary_filt['project_col_id'] = apply_exact_replacement(
        df_col_summary_filt['project_col_id'],
        COL_ID_RENAME_MAP,
    )

    # Apply sample_id mapping to column_summary using the same canonical IDs.
    df_col_summary_filt['project_sample_id'] = apply_exact_replacement(
        df_col_summary_filt['project_col_id'].copy(),
        COL_TO_SAMPLE_ID_MAP,
    )
    if COLUMN_SUMMARY_SAMPLE_ID_FIXES:
        df_col_summary_filt['project_sample_id'] = apply_exact_replacement(
            df_col_summary_filt['project_sample_id'],
            COLUMN_SUMMARY_SAMPLE_ID_FIXES,
        )

    # Duplicate Copperhead 015 rows for holdup conditions
    if 'project_sample_condition_id' in df_col_summary_filt.columns:
        for cond in ['015_jetti_project_file_amcf_pv4_6in', '015_jetti_project_file_amcf_pv4_8in']:
            dup = df_col_summary_filt[df_col_summary_filt['project_sample_condition_id'] == cond].copy()
            if not dup.empty:
                dup['project_sample_condition_id'] = cond + '_holdup'
                df_col_summary_filt = pd.concat([df_col_summary_filt, dup], ignore_index=True)

    # ── 7. Chemical characterisation ──────────────────────────────────────────
    df_chemchar_raw = df_chemchar_raw[df_chemchar_raw['analyte_units'].notnull()]
    df_chemchar_filt = df_chemchar_raw[[
        'origin','project_name','analyte_units',
        'cu_%','fe_%','cu_seq_h2so4_%','cu_seq_nacn_%','cu_seq_a_r_%'
    ]].copy()

    df_chemchar_filt['project_sample_id'] = normalize_dataframe_values(
        df_chemchar_filt['project_name'].str.replace('-','') + '_' +
        df_chemchar_filt['analyte_units'].astype(str).str.replace('-','')
    )
    df_chemchar_filt['project_sample_id'] = df_chemchar_filt['project_sample_id'].str.replace(
        r'_dup$|_\((a|b)\)$', '', regex=True
    )

    # Aggregate duplicates
    agg = {}
    for col in df_chemchar_filt.columns:
        agg[col] = 'mean' if pd.api.types.is_numeric_dtype(df_chemchar_filt[col]) else 'first'
    df_chemchar_filt = df_chemchar_filt.groupby('project_sample_id', as_index=False).agg(agg)

    # Normalised solubility fractions
    tot = df_chemchar_filt[['cu_seq_h2so4_%','cu_seq_nacn_%','cu_seq_a_r_%']].sum(skipna=True, axis=1)
    df_chemchar_filt['acid_soluble_%']   = df_chemchar_filt['cu_seq_h2so4_%'] / tot * 100
    df_chemchar_filt['cyanide_soluble_%']= df_chemchar_filt['cu_seq_nacn_%']  / tot * 100
    df_chemchar_filt['residual_cpy_%']   = df_chemchar_filt['cu_seq_a_r_%']   / tot * 100

    # Apply exact-match sample_id fixes
    df_chemchar_filt['project_sample_id_original'] = df_chemchar_filt['project_sample_id'].copy()
    df_chemchar_filt['project_sample_id'] = apply_exact_replacement(
        df_chemchar_filt['project_sample_id'], CHEMCHAR_SAMPLE_ID_FIXES
    )

    # ── 8. AC summary ─────────────────────────────────────────────────────────
    df_ac = df_ac_summary_raw[[
        'origin','project_name','start_cell','test_id','ore_type','target_ph','h2so4_kg_t'
    ]].copy()
    df_ac['project_sample_id'] = normalize_dataframe_values(
        df_ac['project_name'].astype(str).str.replace('-','') + '_' +
        df_ac['ore_type'].astype(str).str.replace('-','')
    )
    # MANUAL FILTER: drop rows with null test_id
    df_ac = df_ac[~pd.isnull(df_ac['test_id'])]
    # MANUAL FILTER: drop 24h AC tests
    df_ac = df_ac[~df_ac['start_cell'].str.contains('24h', na=False)]
    # MANUAL FILTER: drop Leopard Bronces pH=2 duplicate (tbl-Table4)
    df_ac = df_ac[~(
        (df_ac['project_sample_id'] == 'jetti_project_file_leopard_scl_sample_los_bronces') &
        (df_ac['start_cell'] == 'tbl-Table4')
    )]
    # Apply partial-string sample_id fixes
    df_ac['project_sample_id'] = apply_partial_replacement(
        df_ac['project_sample_id'], ACSUMMARY_SAMPLE_ID_FIXES
    )
    # Duplicate 011 RM data for 011 Crushed
    rm_crushed_label = '011_jetti_project_file_rm_crushed'
    if not (df_ac['project_sample_id'] == rm_crushed_label).any():
        df_ac_rm = df_ac[df_ac['project_sample_id'] == '011_jetti_project_file_rm'].copy()
        if not df_ac_rm.empty:
            df_ac_rm['project_sample_id'] = rm_crushed_label
            df_ac_rm['project_name'] = '011 Jetti Project File-Crushed'
            df_ac = pd.concat([df_ac, df_ac_rm], ignore_index=True)

    # ── 9. Mineralogy modals (keep all, project_sample_id built in step_04) ───
    df_mineralogy = df_mineralogy_raw.copy()

    # ── 10. Maker index — build catalyzed/control col IDs ─────────────────────
    df_maker_index['project_col_id'] = normalize_dataframe_values(
        df_maker_index['file_name'].astype(str).str.replace('-','') + '_' +
        df_maker_index['sample_id'].astype(str).str.replace('-','')
    ) if 'sample_id' in df_maker_index.columns else None

    for attr, col in [
        ('project_cat1_col_id',      'catalyzed_col_1'),
        ('project_cat2_col_id',      'catalyzed_col_2'),
        ('project_control1_col_id',  'control_col_1'),
        ('project_control2_col_id',  'control_col_2'),
        ('project_cat_reactor_id',   'catalyzed_reactor_1'),
        ('project_control_reactor_id','control_reactor_1'),
    ]:
        if col in df_maker_index.columns:
            nan_mask = df_maker_index[['file_name', col]].isnull().any(axis=1)
            vals = normalize_dataframe_values(
                df_maker_index['file_name'].astype(str).str.replace('-','') + '_' +
                df_maker_index[col].astype(str).str.replace('-','')
            )
            df_maker_index[attr] = np.where(nan_mask, np.nan, vals)

    # Fix 011 crushed col IDs in maker_index
    replacements = {
        'project_cat1_col_id':      ('011_jetti_project_file_crushed_rm_7', '011_jetti_project_filecrushed_rm_7'),
        'project_cat2_col_id':      ('011_jetti_project_file_crushed_rm_8', '011_jetti_project_filecrushed_rm_8'),
        'project_control1_col_id':  ('011_jetti_project_file_crushed_rm_5', '011_jetti_project_filecrushed_rm_5'),
        'project_control2_col_id':  ('011_jetti_project_file_crushed_rm_6', '011_jetti_project_filecrushed_rm_6'),
        'project_cat_reactor_id':   ('011_jetti_project_file_crushed_rt_24','011_jetti_project_filecrushed_rt_24'),
        'project_control_reactor_id':('011_jetti_project_file_crushed_rt_21','011_jetti_project_filecrushed_rt_21'),
    }
    for col, (old, new) in replacements.items():
        if col in df_maker_index.columns:
            df_maker_index.loc[df_maker_index[col] == old, col] = new

    # ── 11. Maker index — build project_sample_id ──────────────────────────────
    if 'project_sample_id' not in df_maker_index.columns and 'catalyzed_col_1' in df_maker_index.columns:
        df_maker_index['project_sample_id'] = normalize_dataframe_values(
            df_maker_index['file_name'].astype(str).str.replace('-','') + '_' +
            df_maker_index.get('sample_id', '').astype(str).str.replace('-','')
        )

    # ── 12. Master reactors table ──────────────────────────────────────────────
    df_bottles_filt = df_bottles[['origin','project_name','start_cell','time_(day)','cu_extraction_actual_(%)']].copy()
    df_bottles_filt['project_test_id'] = normalize_dataframe_values(
        df_bottles_filt['project_name'].str.replace('-','') + '_' +
        df_bottles_filt['start_cell'].astype(str).str.split('-').str[1].fillna('')
    )
    df_bottles_filt['reactor_name'] = df_bottles_filt['start_cell'].astype(str).str.split('-').str[1].fillna('')

    df_reactors_raw['time_(day)'] = df_reactors_raw['time_(day)'].fillna(
        df_reactors_raw.get('time_(days)', pd.Series(dtype=float))
    )
    df_reactors_filt = df_reactors_raw[['origin','project_name','start_cell','time_(day)','cu_extraction_actual_(%)']].copy()
    df_reactors_filt['project_test_id'] = normalize_dataframe_values(
        df_reactors_filt['project_name'].str.replace('-','') + '_' +
        df_reactors_filt['start_cell'].astype(str).str.split('-').str[1].fillna('')
    )
    df_reactors_filt['reactor_name'] = df_reactors_filt['start_cell'].astype(str).str.split('-').str[1].fillna('')

    df_reactors_bottles = pd.concat([df_bottles_filt, df_reactors_filt], axis=0, join='outer')

    # Build catalyzed/control column arrays from maker_index
    cat_nan = df_maker_index[['file_name','catalyzed_reactor_1']].isnull().any(axis=1) if 'catalyzed_reactor_1' in df_maker_index.columns else pd.Series([True]*len(df_maker_index))
    ctl_nan = df_maker_index[['file_name','control_reactor_1']].isnull().any(axis=1) if 'control_reactor_1' in df_maker_index.columns else pd.Series([True]*len(df_maker_index))

    if 'catalyzed_reactor_1' in df_maker_index.columns:
        cat_vals = normalize_dataframe_values(
            df_maker_index['file_name'].astype(str).str.replace('-','') + '_' +
            df_maker_index['catalyzed_reactor_1'].astype(str).str.replace('-','')
        )
        catalyzed_cols = np.where(cat_nan, np.nan, cat_vals)
    else:
        catalyzed_cols = np.array([np.nan] * len(df_maker_index))

    if 'control_reactor_1' in df_maker_index.columns:
        ctl_vals = normalize_dataframe_values(
            df_maker_index['file_name'].astype(str).str.replace('-','') + '_' +
            df_maker_index['control_reactor_1'].astype(str).str.replace('-','')
        )
        control_cols = np.where(ctl_nan, np.nan, ctl_vals)
    else:
        control_cols = np.array([np.nan] * len(df_maker_index))

    df_m1 = pd.DataFrame()
    df_m2 = pd.DataFrame()
    cat_mask = df_reactors_bottles['project_test_id'].isin(catalyzed_cols)
    ctl_mask = df_reactors_bottles['project_test_id'].isin(control_cols)

    if cat_mask.any():
        df_m1[['origin','project_name','reactor_name','project_test_id','leach_duration_days','catalyzed_cu_extraction_actual_(%)']] = \
            df_reactors_bottles[cat_mask][['origin','project_name','reactor_name','project_test_id','time_(day)','cu_extraction_actual_(%)']]
        df_m1['leach_duration_days'] = df_m1['leach_duration_days'].round(0)

    if ctl_mask.any():
        df_m2[['origin','project_name','reactor_name','project_test_id','leach_duration_days','control_cu_extraction_actual_(%)']] = \
            df_reactors_bottles[ctl_mask][['origin','project_name','reactor_name','project_test_id','time_(day)','cu_extraction_actual_(%)']]
        df_m2['leach_duration_days'] = df_m2['leach_duration_days'].round(0)

    df_master_reactors = pd.DataFrame()
    if not df_m1.empty and not df_m2.empty:
        df_m1['project_name'] = df_m1['project_name'].astype(str)
        df_m2['project_name'] = df_m2['project_name'].astype(str)
        df_master_reactors = df_m2.merge(df_m1, on=['origin','project_name','leach_duration_days'], how='outer')

        # Match project_sample_id from maker_index
        if 'project_sample_id' in df_maker_index.columns:
            df_master_reactors['project_sample_id'] = None
            df_master_reactors['project_sample_id'] = df_master_reactors['project_sample_id'].astype(object)
            for idx, row in df_master_reactors.iterrows():
                mask = (df_maker_index['file_name'] == row.get('project_name')) & \
                       (df_maker_index.get('project_cat_reactor_id', '') == row.get('project_test_id_y','')) & \
                       (df_maker_index.get('project_control_reactor_id', '') == row.get('project_test_id_x',''))
                matches = df_maker_index[mask]['project_sample_id'].values
                df_master_reactors.at[idx, 'project_sample_id'] = matches[0] if len(matches) > 0 else None

    # ── 13. Save intermediates — all SpkData tables preserved here ────────────
    save_intermediate(df_merged,          str(intermediate_dir / "step_01_leaching_raw.csv"),       "leaching_raw")
    save_intermediate(df_col_summary_filt,str(intermediate_dir / "step_01_column_summary.csv"),     "column_summary")
    save_intermediate(df_chemchar_filt,   str(intermediate_dir / "step_01_chemchar.csv"),           "chemchar")
    save_intermediate(df_mineralogy,      str(intermediate_dir / "step_01_mineralogy_modals.csv"),  "mineralogy_modals")
    save_intermediate(df_ac,              str(intermediate_dir / "step_01_ac_summary.csv"),         "ac_summary")
    save_intermediate(df_separation,      str(intermediate_dir / "step_01_separation.csv"),         "separation")
    save_intermediate(df_comments,        str(intermediate_dir / "step_01_comments.csv"),           "comments")
    save_intermediate(df_master_reactors, str(intermediate_dir / "step_01_master_reactors.csv"),    "master_reactors")
    save_intermediate(df_maker_index,     str(intermediate_dir / "step_01_maker_index.csv"),        "maker_index")

    print(f"[step_01] Done. Leaching rows: {len(df_merged):,} | ChemChar samples: {len(df_chemchar_filt):,}")

    # Return all SpkData pipeline tables so downstream steps can use any of them.
    # Every table here comes from the original data_for_tableau.py export list.
    return {
        "df_leaching_raw":      df_merged,
        "df_column_summary":    df_col_summary_filt,
        "df_chemchar":          df_chemchar_filt,
        "df_mineralogy_modals": df_mineralogy,
        "df_ac_summary":        df_ac,
        "df_separation":        df_separation,
        "df_comments":          df_comments,
        "df_maker_index":       df_maker_index,
        "df_master_reactors":   df_master_reactors,
        "df_qemscan_raw":       df_qemscan_raw,
    }
