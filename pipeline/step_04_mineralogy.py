"""
step_04_mineralogy.py — Mineralogy grouping.

Corresponds to: rosetta_mineralogy_clustering.py (grouping portion; PCA removed)

Outputs:
  df_mineralogy_grouped : summed mineral groups per project_sample_id
"""

import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import PATHS, MINERALOGY_ID_MAP, MINERALOGY_SAMPLE_MAP, MINERALOGY_DUPLICATES, MINERAL_GROUPS
from utils import save_intermediate, save_to_paths

INTERMEDIATE_DIR = Path(__file__).parent / "intermediate"


def sum_available_columns(df: pd.DataFrame, cols: list) -> pd.Series:
    """Sum available columns, replacing NaN and whitespace with 0."""
    df = df.copy()
    df.replace(' ', 0, inplace=True)
    df.replace(np.nan, 0, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    avail = list(set(cols) & set(df.columns))
    return df[avail].sum(axis=1) if avail else pd.Series(0, index=df.index)


def run(intermediate_dir: Path = INTERMEDIATE_DIR) -> dict:
    intermediate_dir.mkdir(exist_ok=True)
    db = PATHS["db_python"]

    print("[step_04] Loading mineralogy modals CSV...")
    df_mineralogy = pd.read_csv(f"{db}/df_mineralogy_modals.csv", index_col=0, low_memory=False)
    df_mineralogy.dropna(subset=['project_sample_id'], inplace=True)
    df_mineralogy.reset_index(drop=True, inplace=True)

    # ── 1. Assign project_sample_id for unmapped rows (project_sample_id == '0') ──
    mask_zero = df_mineralogy['project_sample_id'].astype(str) == '0'

    # By (project_name, start_cell)
    for (proj, sc), new_id in MINERALOGY_ID_MAP.items():
        m = mask_zero & (df_mineralogy['project_name'] == proj) & (df_mineralogy['start_cell'].astype(str) == sc)
        df_mineralogy.loc[m, 'project_sample_id'] = new_id

    # By (project_name, sample column value)
    for (proj, sample_val), new_id in MINERALOGY_SAMPLE_MAP.items():
        m = mask_zero & (df_mineralogy['project_name'] == proj) & (df_mineralogy['sample'].astype(str) == sample_val)
        df_mineralogy.loc[m, 'project_sample_id'] = new_id

    # ── 2. Duplicate rows for shared mineralogy ────────────────────────────────
    for (source_id, new_id) in MINERALOGY_DUPLICATES:
        dup_rows = df_mineralogy[df_mineralogy['project_sample_id'] == source_id].copy()
        if not dup_rows.empty:
            dup_rows['project_sample_id'] = new_id
            df_mineralogy = pd.concat([df_mineralogy, dup_rows], ignore_index=True)

    # ── 3. Drop identifier columns ─────────────────────────────────────────────
    cols_to_drop = ['origin','project_name','sample','sheet_name','start_cell','index']
    df_mineralogy.drop(columns=[c for c in cols_to_drop if c in df_mineralogy.columns], inplace=True)

    # ── 4. Special: rename 015_amcf → 015_amcf_6in (8in was duplicated above) ─
    df_mineralogy.loc[
        df_mineralogy['project_sample_id'] == '015_jetti_project_file_amcf',
        'project_sample_id'
    ] = '015_jetti_project_file_amcf_6in'

    print(f"[step_04] Mineralogy rows after ID assignment: {len(df_mineralogy)} | unique IDs: {df_mineralogy['project_sample_id'].nunique()}")

    # ── 5. Compute grouped mineral sums ───────────────────────────────────────
    df_mineralogy_grouped = df_mineralogy[['project_sample_id']].copy()
    for group_name, mineral_list in MINERAL_GROUPS.items():
        df_mineralogy_grouped[f'grouped_{group_name}'] = sum_available_columns(df_mineralogy, mineral_list)

    df_mineralogy_grouped = df_mineralogy_grouped.round(2)

    # Drop rows still with project_sample_id == '0'
    df_mineralogy_grouped = df_mineralogy_grouped[df_mineralogy_grouped['project_sample_id'].astype(str) != '0']
    df_mineralogy_grouped = df_mineralogy_grouped.drop_duplicates(subset='project_sample_id')

    print(f"[step_04] Grouped mineralogy: {len(df_mineralogy_grouped)} samples")

    # ── 6. Save — outputs stay inside pipeline/ only ──────────────────────────
    outputs_dir = intermediate_dir.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    save_intermediate(df_mineralogy_grouped, str(intermediate_dir / "step_04_mineralogy_grouped.csv"), "mineralogy_grouped")
    df_mineralogy_grouped.to_csv(str(outputs_dir / "df_mineralogy_grouped.csv"), index=False)
    print(f"  → {outputs_dir / 'df_mineralogy_grouped.csv'}")

    print(f"[step_04] Done.")
    return {
        "df_mineralogy_grouped": df_mineralogy_grouped,
    }
