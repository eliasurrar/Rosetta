"""
utils.py — Shared utilities for the Rosetta ETL pipeline.

All functions here are Arrow/pandas-2.x safe and include every bug fix
that was identified across the original codebase.
"""

import re
import unicodedata
import warnings

import numpy as np
import pandas as pd


# ==============================================================================
# STRING NORMALISATION
# (fixed version of normalize_and_replace from functions_general.py)
# ==============================================================================

def normalize_and_replace(
    string,
    chars_to_keep=r'[^a-z0-9_+\-%#\(\)]',
    replace_nan_unnamed=True,
    remove_numbers=False,
):
    """Normalize a single string value. NaN / non-string values pass through unchanged."""
    # BUG FIX: original crashed with TypeError when string was float (NaN)
    if not isinstance(string, str):
        return string
    normalized = unicodedata.normalize('NFKD', string)
    lower = normalized.lower()
    if replace_nan_unnamed:
        lower = lower.replace('nan', '').replace('unnamed', '').replace('units', '').replace('unit', '')
    if remove_numbers:
        lower = re.sub(r'\d', '', lower)
    replaced = re.sub(chars_to_keep, '_', lower)
    cleaned = re.sub(r'_+', '_', replaced)
    return cleaned.strip('_')


def normalize_series(series: pd.Series) -> pd.Series:
    """Apply normalize_and_replace element-wise to a Series."""
    return series.apply(normalize_and_replace)


def normalize_dataframe_values(s: pd.Series) -> pd.Series:
    """Alias kept for compatibility with original codebase."""
    return normalize_series(s)


# ==============================================================================
# NUMERIC CONVERSION  (Arrow-safe)
# ==============================================================================

def safe_to_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric, coercing errors. Arrow-safe."""
    return pd.to_numeric(series, errors='coerce')


def is_numeric_col(series: pd.Series) -> bool:
    """
    Check if a column is numeric dtype.
    BUG FIX: np.issubdtype crashes on Arrow/StringDtype; use pd.api.types instead.
    """
    return pd.api.types.is_numeric_dtype(series)


def convert_cols_to_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Convert a list of columns to numeric, column by column (Arrow-safe).
    BUG FIX: slice assignment df.iloc[:, n:] = ... fails on Arrow dtypes.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def convert_all_numeric_except(df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
    """
    Convert all columns to numeric except those in exclude_cols, column by column.
    """
    df = df.copy()
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def convert_cols_from_index(df: pd.DataFrame, start_col_index: int) -> pd.DataFrame:
    """
    Convert all columns from start_col_index onward to numeric (column by column).
    BUG FIX: replaces df.iloc[:, n:] = df.iloc[:, n:].apply(pd.to_numeric, ...)
    which fails on Arrow-backed string columns.
    """
    df = df.copy()
    for col in df.columns[start_col_index:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ==============================================================================
# STRING ASSIGNMENT  (Arrow-safe)
# ==============================================================================

def safe_string_assign(df: pd.DataFrame, col: str, values) -> pd.DataFrame:
    """
    Assign string values to df[col] safely.
    BUG FIX: initialising a column with np.nan creates float64; assigning
    ArrowStringArray into it raises TypeError. Use None (object dtype) instead.
    """
    df = df.copy()
    if col not in df.columns:
        df[col] = None
    df[col] = df[col].astype(object)
    df[col] = values
    return df


def ensure_object_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Cast an existing column to object dtype before string assignment."""
    df = df.copy()
    if col in df.columns:
        df[col] = df[col].astype(object)
    else:
        df[col] = None
    return df


# ==============================================================================
# ID COLUMN PROTECTION
# ==============================================================================

def protect_id_cols_dropna(df: pd.DataFrame, id_cols: list, **dropna_kwargs) -> pd.DataFrame:
    """
    Run dropna(axis=1) while guaranteeing that id_cols are never dropped.
    BUG FIX: dropna(axis=1, thresh=...) was silently dropping project_name,
    project_col_id, project_sample_id when they were sparse.
    """
    # Save id_col data before dropping
    present = [c for c in id_cols if c in df.columns]
    saved = df[present].copy()

    df = df.dropna(**dropna_kwargs)

    # Restore any that were dropped
    for col in present:
        if col not in df.columns:
            df[col] = saved[col]

    return df


# ==============================================================================
# GROUP FILTERING  (replaces groupby.apply returning DataFrame)
# ==============================================================================

def filter_increasing_within_group(
    df: pd.DataFrame,
    group_cols: list,
    value_col: str,
) -> pd.DataFrame:
    """
    Keep only rows where value_col is strictly increasing within each group.
    BUG FIX: the original used groupby(id_cols, group_keys=False).apply(lambda g:
    g[g[col].diff().gt(0)]) which in pandas 2.x can silently absorb the groupby
    keys into the MultiIndex, removing them from .columns.
    This vectorised version is equivalent and never loses columns.
    """
    present = [c for c in group_cols if c in df.columns]
    if present:
        diff_mask = df.groupby(present, sort=False)[value_col].diff().gt(0)
    else:
        diff_mask = df[value_col].diff().gt(0)
    return df[diff_mask].reset_index(drop=True)


# fill_with_bins removed — pipeline standardised on exponential model fitting only.


# ==============================================================================
# SAFE NANMEAN
# ==============================================================================

def safe_nanmean(arr, axis=0):
    """
    np.nanmean with RuntimeWarning suppressed for all-NaN slices.
    BUG FIX: np.errstate does not suppress Python-level RuntimeWarning from
    nanmean; we need warnings.catch_warnings as well.
    """
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(arr, axis=axis)


# ==============================================================================
# REPLACEMENT DICT APPLICATION
# ==============================================================================

def apply_exact_replacement(series: pd.Series, mapping: dict) -> pd.Series:
    """Apply an exact-match replacement dict to a string Series."""
    series = series.copy().astype(object)
    for pattern, replacement in mapping.items():
        escaped = re.escape(pattern)
        series = series.str.replace(f'^{escaped}$', replacement, regex=True)
    return series


def apply_partial_replacement(series: pd.Series, mapping: dict) -> pd.Series:
    """Apply a partial (contains) replacement dict to a string Series."""
    series = series.copy().astype(object)
    for pattern, replacement in mapping.items():
        series = series.str.replace(pattern, replacement, regex=False)
    return series


# ==============================================================================
# DATAFRAME SAVE HELPERS
# ==============================================================================

def save_intermediate(df: pd.DataFrame, path: str, label: str = '') -> None:
    """Save a dataframe to CSV, printing a summary line."""
    df.to_csv(path, index=False)
    tag = f'[{label}] ' if label else ''
    print(f'  {tag}Saved {len(df):,} rows × {len(df.columns)} cols → {path}')


def save_to_paths(df: pd.DataFrame, paths: list, index: bool = False) -> None:
    """Save a dataframe to multiple output paths."""
    import os
    for p in paths:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            df.to_csv(p, index=index)
            print(f'  → {p}')
        except Exception as e:
            print(f'  ⚠ Could not save to {p}: {e}')
