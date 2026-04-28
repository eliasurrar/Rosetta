"""
step_03_reactors.py — Reactor bi-exponential curve fitting.

Corresponds to: rosetta_reactors_pca.py (fitting portion only; PCA removed)

Standardised fitting methodology (bi-exponential):
    recovery(t) = a1 * (1 - exp(-b1 * t)) + a2 * (1 - exp(-b2 * t))

  where t = leach_duration_days and recovery = cu_extraction_actual_(%).

  - Weighted least-squares via scipy curve_fit; later data points weighted higher
    (weights increase linearly from 1 to 10 across the observed time series).
  - Predictions are clipped to [0, 99.5].
  - Fallback when optimisation fails: a1 = 0.7*max_y, a2 = 0.3*max_y, b1=b2=b_min.
  - Applied uniformly to every (project_name, start_cell) combination.

Outputs:
  df_exponential_model          : full bi-exponential predictions (all reactors)
  df_exponential_model_filtered : filtered to LEACHING_COLS_TO_KEEP reactors only
  fit_stats                     : a1, b1, a2, b2 + R² and RMSE per reactor
  leaching_cols_to_keep         : from config (for downstream use)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import PATHS, LEACHING_COLS_TO_KEEP
from utils import convert_cols_from_index, save_intermediate

INTERMEDIATE_DIR = Path(__file__).parent / "intermediate"


def biexponential_model(t, a1, b1, a2, b2):
    """
    Bi-exponential copper recovery model.
    recovery(t) = a1 * (1 - exp(-b1 * t)) + a2 * (1 - exp(-b2 * t))
    t  : leach_duration_days (scalar or array)
    a1 : amplitude of fast component  (%)
    b1 : rate constant of fast component (1/day)
    a2 : amplitude of slow component  (%)
    b2 : rate constant of slow component (1/day)
    """
    t = np.asarray(t, dtype=float)
    t = np.clip(t, 0.0, None)
    return a1 * (1.0 - np.exp(-b1 * t)) + a2 * (1.0 - np.exp(-b2 * t))


def duplicate_reactor_label_rows(df, alias_map):
    """Duplicate reactor rows under alternate labels expected downstream."""
    aliased_rows = []
    for target_key, source_key in alias_map.items():
        target_project, target_sc = target_key
        source_project, source_sc = source_key
        if ((df['project_name'] == target_project) & (df['start_cell'] == target_sc)).any():
            continue
        src_mask = (df['project_name'] == source_project) & (df['start_cell'] == source_sc)
        if src_mask.any():
            aliased_rows.append(
                df.loc[src_mask].assign(project_name=target_project, start_cell=target_sc)
            )
    if aliased_rows:
        df = pd.concat([df] + aliased_rows, ignore_index=True)
    return df


def run(intermediate_dir: Path = INTERMEDIATE_DIR) -> dict:
    intermediate_dir.mkdir(exist_ok=True)
    spk = PATHS["spkdata_csvs"]
    uef = PATHS["user_editable_files"]

    print("[step_03] Loading reactor and bottle data...")

    # ── 1. Load data ───────────────────────────────────────────────────────────
    df_reactors = pd.read_csv(f"{spk}/dataset_reactor_summary_detailed.csv", low_memory=False)
    df_bottles  = pd.read_csv(f"{spk}/dataset_rolling_bottles_detailed.csv", low_memory=False)
    term_path   = f"{uef}/dataset_reactor_summary_detailed_terminated_projects.xlsx"
    try:
        df_reactors_term = pd.read_excel(term_path)
    except FileNotFoundError:
        print(f"  ⚠ Terminated reactor file not found in user_editable_files/ — skipping")
        df_reactors_term = pd.DataFrame()

    # BUG FIX: use column-by-column conversion (slice assignment fails on Arrow dtype)
    df_reactors['time_(day)'] = df_reactors['time_(day)'].fillna(df_reactors.get('time_(days)', pd.Series(dtype=float)))
    df_reactors = convert_cols_from_index(df_reactors, 5)
    df_bottles  = convert_cols_from_index(df_bottles, 5)
    df_reactors_term = convert_cols_from_index(df_reactors_term, 5)

    # ── 2. Merge terminated ────────────────────────────────────────────────────
    df_reactors = pd.concat([df_reactors, df_reactors_term], axis=0, ignore_index=True)

    # ── 3. 011 Crushed — copy 011 rows under crushed label ─────────────────────
    df_reactors = pd.concat([
        df_reactors,
        df_reactors[df_reactors['project_name'] == '011 Jetti Project File']
        .assign(project_name='011 Jetti Project File-Crushed')
    ], ignore_index=True)

    # ── 4. SCL projects — copy reactor curves from 007 variants ───────────────
    for src, tgt in [
        ('007 Jetti Project File - Zaldivar', 'Jetti Project File - Zaldivar SCL'),
        ('007 Jetti Project File - Toquepala', 'Jetti Project File - Toquepala SCL'),
    ]:
        df_reactors = pd.concat([
            df_reactors,
            df_reactors[df_reactors['project_name'] == src].assign(project_name=tgt)
        ], ignore_index=True)

    # Elephant SCL: remove existing tbl-RT_8 then add from 007
    df_reactors = df_reactors.drop(
        df_reactors[(df_reactors['project_name'] == 'Jetti Project File - Elephant SCL') &
                    (df_reactors['start_cell'] == 'tbl-RT_8')].index
    )
    eleph_mask = (
        (df_reactors['project_name'] == '007 Jetti Project File - Elephant, Leopard, Tiger') &
        (df_reactors.get('sheet_name', '') == 'RT Summary Elephant_Leopard') &
        (df_reactors['start_cell'].isin(['tbl-RT_5R','tbl-RT_6R','tbl-RT_8']))
    )
    df_reactors = pd.concat([df_reactors, df_reactors[eleph_mask].assign(project_name='Jetti Project File - Elephant SCL')], ignore_index=True)

    leopard_mask = (
        (df_reactors['project_name'] == '007 Jetti Project File - Elephant, Leopard, Tiger') &
        (df_reactors.get('sheet_name', '') == 'RT Summary Elephant_Leopard') &
        (df_reactors['start_cell'].isin(['tbl-RT_2R','tbl-RT_4R']))
    )
    df_reactors = pd.concat([df_reactors, df_reactors[leopard_mask].assign(project_name='Jetti Project File - Leopard SCL')], ignore_index=True)

    # ── 5. Merge bottles ───────────────────────────────────────────────────────
    df_reactors = pd.concat([df_reactors, df_bottles], axis=0, ignore_index=True)

    # MANUAL FILTER: keep only positive extraction values
    df_reactors = df_reactors[df_reactors['cu_extraction_actual_(%)'] > 0]

    # ── 6. Alias reactor labels (011 RT_21/RT_24 stored under Crushed label) ──
    reactor_aliases = {
        ('011 Jetti Project File', 'tbl-RT_21'): ('011 Jetti Project File-Crushed', 'tbl-RT_21'),
        ('011 Jetti Project File', 'tbl-RT_24'): ('011 Jetti Project File-Crushed', 'tbl-RT_24'),
    }
    df_reactors = duplicate_reactor_label_rows(df_reactors, reactor_aliases)

    # ── 7. Round time, deduplicate ─────────────────────────────────────────────
    df_reactors['time_(day)'] = np.round(df_reactors['time_(day)'] * 4) / 4
    df_reactors = df_reactors.groupby(
        ['project_name','sheet_name','catalyzed','start_cell','time_(day)'],
        dropna=False
    ).last().reset_index()

    # ── 8. Pivot to time-series matrix ────────────────────────────────────────
    df_pivot = df_reactors.pivot_table(
        index=['project_name','start_cell'],
        columns='time_(day)',
        values='cu_extraction_actual_(%)'
    )

    # MANUAL FILTER: specific data point removals
    for idx_key, day, val in [
        (('014 Jetti Project File','tbl-RTB_7'), 42.25, np.nan),
        (('017 Jetti Project File','tbl-RTEA_1'), 48.0, np.nan),
        (('017 Jetti Project File','tbl-RTEA_1'), 42.0, np.nan),
    ]:
        if idx_key in df_pivot.index and day in df_pivot.columns:
            df_pivot.loc[idx_key, day] = val

    # MANUAL FILTER: drop early noisy points for 020 hypogene RT_19/RT_20
    for rt in ['tbl-RT_19','tbl-RT_20']:
        key = ('020 Jetti Project File Hypogene_Supergene', rt)
        if key in df_pivot.index:
            early_days = [d for d in [0.25, 0.5, 1.0, 2.0, 3.0] if d in df_pivot.columns]
            df_pivot.loc[key, early_days] = np.nan

    print(f"[step_03] Pivot shape: {df_pivot.shape}")

    # ── 9. Bi-exponential model fitting ───────────────────────────────────────
    # Model: recovery(t) = a1*(1-exp(-b1*t)) + a2*(1-exp(-b2*t))
    # b_min ensures 99% of plateau is reached by day 200 at most.
    # Weights increase linearly so later data points are trusted more.
    print("[step_03] Fitting bi-exponential models...")
    plateau_forced = 200
    b_min = -np.log(0.01) / plateau_forced          # ≈ 0.023
    b_fast_guess  = max(b_min, 0.10)                # fast component initial guess
    b_slow_guess  = max(b_min, 0.01)                # slow component initial guess
    original_cols = df_pivot.columns.astype(float)
    new_df     = pd.DataFrame(index=df_pivot.index, columns=original_cols)
    fit_stats  = pd.DataFrame(index=df_pivot.index,
                              columns=["a1","b1","a2","b2","R_squared","RMSE"])

    for idx in df_pivot.index:
        row_data = df_pivot.loc[idx]
        mask     = row_data.notna()
        t_data   = row_data.index[mask].astype(float)
        y_data   = row_data[mask].values.astype(float)

        if len(y_data) == 0:
            # Fallback: split plateau 70/30 between components
            a1_fit, b1_fit = 0.7 * 99.5, b_min
            a2_fit, b2_fit = 0.3 * 99.5, b_min
            r2, rmse = np.nan, np.nan
        else:
            max_y = float(np.max(y_data))
            cap   = min(max_y, 99.5)
            n     = len(y_data)
            # Later points weighted up to 10× more than first point
            sigma = 1.0 / np.sqrt(np.linspace(1, 10, n))
            try:
                params, _ = curve_fit(
                    biexponential_model,
                    t_data, y_data,
                    p0=(0.7 * cap, b_fast_guess,
                        0.3 * cap, b_slow_guess),
                    bounds=(
                        [0.0,   b_min, 0.0,   b_min],
                        [99.5, np.inf, 99.5, np.inf],
                    ),
                    sigma=sigma,
                    maxfev=20000,
                )
                a1_fit, b1_fit, a2_fit, b2_fit = params
                y_pred = biexponential_model(t_data, a1_fit, b1_fit, a2_fit, b2_fit)
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r2   = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
                rmse = float(np.sqrt(np.mean((y_data - y_pred) ** 2)))
            except (RuntimeError, ValueError):
                # Fallback: proportional split of last observed maximum
                a1_fit, b1_fit = 0.7 * cap, b_min
                a2_fit, b2_fit = 0.3 * cap, b_min
                r2, rmse = np.nan, np.nan

        # Generate predictions for all time points; clip to [0, 99.5]
        preds = np.clip(
            biexponential_model(original_cols, a1_fit, b1_fit, a2_fit, b2_fit),
            0.0, 99.5,
        )
        new_df.loc[idx]    = preds
        fit_stats.loc[idx] = [a1_fit, b1_fit, a2_fit, b2_fit, r2, rmse]

    new_df    = new_df.astype(float)
    fit_stats = fit_stats.astype({
        "a1": float, "b1": float,
        "a2": float, "b2": float,
        "R_squared": float, "RMSE": float,
    })
    # (fit_stats saved to outputs/ in the save block below)

    # ── 10. Build final model dataframe (exponential model only) ─────────────
    filtered_columns = [
        col for col in new_df.columns
        if (float(col) >= 5.0 and str(col).endswith(".0")) or float(col) < 5.0 or float(col) >= 125.0
    ]
    df_exponential_model = new_df[filtered_columns].copy()
    df_exponential_model.columns = df_exponential_model.columns.astype(str)

    # ── 11. Build leaching index and filter ───────────────────────────────────
    leaching_df = pd.DataFrame.from_dict(
        LEACHING_COLS_TO_KEEP, orient='index',
        columns=['project_name','start_cell','project_sample_id','catalyzed_y_n','ongoing_y_n']
    )
    leaching_index = pd.MultiIndex.from_frame(leaching_df[['project_name','start_cell']])

    df_exponential_model_filtered = df_exponential_model.loc[
        df_exponential_model.index.intersection(leaching_index)
    ].copy()

    # ── 12. Build dataset_per_sample_reactor_complete.csv ─────────────────────
    # Replaces the external user-editable file — generated entirely from SpkData
    # CSVs + bi-exponential fit_stats. step_06 reads this from pipeline/outputs/.
    print("[step_03] Building dataset_per_sample_reactor_complete.csv...")
    try:
        spk = PATHS["spkdata_csvs"]
        df_summaries = pd.read_csv(f"{spk}/dataset_reactor_summary_summaries.csv", low_memory=False)
        df_detailed  = pd.read_csv(f"{spk}/dataset_reactor_summary_detailed.csv",  low_memory=False)

        # Compute per-reactor mean temperature and pH from detailed data
        temp_col = next((c for c in df_detailed.columns if 'temp' in c.lower() and '(' in c), None)
        ph_col   = next((c for c in df_detailed.columns if c.lower() in ('ph', 'adjusted_ph')), None)

        agg = {'project_name': 'first', 'start_cell': 'first'}
        if temp_col:
            df_detailed[temp_col] = pd.to_numeric(df_detailed[temp_col], errors='coerce')
            agg[temp_col] = 'mean'
        if ph_col:
            df_detailed[ph_col] = pd.to_numeric(df_detailed[ph_col], errors='coerce')
            agg[ph_col] = 'mean'

        df_means = (
            df_detailed.groupby(['project_name', 'start_cell'], as_index=False)
            .agg({k: v for k, v in agg.items() if k in df_detailed.columns})
        )
        rename_map = {}
        if temp_col: rename_map[temp_col] = 'temp_(c)_mean'
        if ph_col:   rename_map[ph_col]   = 'ph_mean'
        df_means = df_means.rename(columns=rename_map)

        # Final cu_extraction per reactor (last non-NaN value)
        cu_col = 'cu_extraction_actual_(%)'
        if cu_col in df_detailed.columns:
            df_final_cu = (
                df_detailed[df_detailed[cu_col].notna()]
                .sort_values('time_(day)' if 'time_(day)' in df_detailed.columns else df_detailed.columns[0])
                .groupby(['project_name', 'start_cell'])[cu_col]
                .last()
                .reset_index()
                .rename(columns={cu_col: 'final_cu_extraction'})
            )
        else:
            df_final_cu = pd.DataFrame(columns=['project_name', 'start_cell', 'final_cu_extraction'])

        # Fit parameters (a1, b1, a2, b2) from our bi-exponential fitting
        df_fit = fit_stats.reset_index()
        df_fit.columns = ['project_name', 'start_cell'] + list(df_fit.columns[2:])
        df_fit = df_fit.rename(columns={'a1': 'a1_param', 'b1': 'b1_param',
                                        'a2': 'a2_param', 'b2': 'b2_param'})

        # Map (project_name, start_cell) → project_sample_id + catalyzed_y_n
        sample_map = {
            (v[0], v[1]): {'project_sample_id': v[2], 'catalyzed_y_n': v[3]}
            for v in LEACHING_COLS_TO_KEEP.values() if v[1]
        }

        # Build the complete dataset: summaries + means + final_cu + fit params
        df_complete = df_summaries.copy()
        df_complete = df_complete.merge(df_means,    on=['project_name','start_cell'], how='left')
        df_complete = df_complete.merge(df_final_cu, on=['project_name','start_cell'], how='left')
        df_complete = df_complete.merge(
            df_fit[['project_name','start_cell','a1_param','b1_param','a2_param','b2_param','R_squared','RMSE']],
            on=['project_name','start_cell'], how='left'
        )

        # Add project_sample_id and catalyst label
        def _map_sample(row):
            info = sample_map.get((row['project_name'], row['start_cell']), {})
            return pd.Series({
                'project_sample_id': info.get('project_sample_id', ''),
                'catalyst_type':     'Control' if info.get('catalyzed_y_n') == 'control' else
                                     '100-CA'  if info.get('catalyzed_y_n') == 'catalyzed' else
                                     row.get('catalyst_type', ''),
            })

        mapped = df_complete.apply(_map_sample, axis=1)
        df_complete['project_sample_id'] = mapped['project_sample_id']
        # Only override catalyst_type if we have a mapping; otherwise keep original
        df_complete['catalyst_type'] = np.where(
            mapped['catalyst_type'] != '',
            mapped['catalyst_type'],
            df_complete.get('catalyst_type', '')
        )

        # Also use cu_extraction_actual_(%) from summaries if final_cu is missing
        if 'cu_extraction_actual_(%)' in df_complete.columns and 'final_cu_extraction' in df_complete.columns:
            df_complete['final_cu_extraction'] = df_complete['final_cu_extraction'].fillna(
                pd.to_numeric(df_complete['cu_extraction_actual_(%)'], errors='coerce')
            )

        print(f"  Built {len(df_complete):,} rows × {len(df_complete.columns)} cols")
    except Exception as e:
        print(f"  ⚠ Could not build dataset_per_sample_reactor_complete: {e}")
        df_complete = pd.DataFrame()

    # ── 13. Save — outputs stay inside pipeline/ only ─────────────────────────
    outputs_dir = intermediate_dir.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    save_intermediate(fit_stats.reset_index(), str(intermediate_dir / "step_03_fit_stats.csv"), "fit_stats")
    fit_stats.to_csv(str(outputs_dir / "reactors_expmodel_fit_stats.csv"))
    df_exponential_model.to_csv(str(outputs_dir / "df_reactors_exponential_model_all.csv"))
    df_exponential_model_filtered.to_csv(str(outputs_dir / "df_reactors_exponential_model_filtered.csv"))
    if not df_complete.empty:
        reactor_complete_path = str(outputs_dir / "dataset_per_sample_reactor_complete.csv")
        df_complete.to_csv(reactor_complete_path, index=False)
        print(f"  → {reactor_complete_path}")
    print(f"  → {outputs_dir / 'df_reactors_exponential_model_filtered.csv'}")

    print(f"[step_03] Done. Fitted: {len(df_exponential_model)} reactors | In-scope: {len(df_exponential_model_filtered)}")
    return {
        "df_exponential_model_filtered": df_exponential_model_filtered,
        "fit_stats":                     fit_stats,
        "df_reactor_complete":           df_complete,
    }
