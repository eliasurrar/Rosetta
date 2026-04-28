"""
pipeline.py — Rosetta ETL Pipeline Orchestrator

Run all steps in sequence. To disable a step, comment out its call below.
Each step saves intermediate CSVs to pipeline/intermediate/ for debugging,
and all final outputs go to pipeline/outputs/ only — existing folders are untouched.

Usage:
    cd /path/to/Rosetta/pipeline
    python pipeline.py

Order:
    step_01 → raw data loading and normalisation
    step_02 → leaching performance with all feature matches (needs step_01)
    step_03 → reactor exponential curve fitting (independent; no PCA)
    step_04 → mineralogy grouping (independent; no PCA)
    step_05 → ML model datasets — merges step_02, step_03, step_04
    step_06 → column averages for Power BI (needs step_05)

Fitting methodology (standardised across all steps):
    Bi-exponential:  recovery(t) = a1*(1-exp(-b1*t)) + a2*(1-exp(-b2*t))
    t  = leach_duration_days,  recovery = cu_extraction / cu_recovery_%
    Fit via scipy curve_fit with linearly increasing weights (later data weighted higher).
    Predictions clipped to [0, 99.5].
    Fallback (optimisation failure): a1=0.7*max_y, a2=0.3*max_y, b1=b2=b_min.
"""


import time
from pathlib import Path

# ── Directory setup ────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent
INTERMEDIATE = PIPELINE_DIR / "intermediate"
OUTPUTS      = PIPELINE_DIR / "outputs"
INTERMEDIATE.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

# ── Step imports ───────────────────────────────────────────────────────────────
from step_00_sharepoint       import run as run_step_00
from step_01_raw_data         import run as run_step_01
from step_02_leaching         import run as run_step_02
from step_03_reactors         import run as run_step_03
from step_04_mineralogy       import run as run_step_04
from step_05_model_datasets   import run as run_step_05
from step_06_columns_reactors import run as run_step_06


def _time_step(name, fn, *args, **kwargs):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Running {name}...")
    print(f"{'='*60}")
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"  ✓ {name} completed in {elapsed:.1f}s")
    return result


if __name__ == "__main__":
    t_total = time.time()
    print("Rosetta ETL Pipeline — starting")
    print(f"Intermediate outputs → {INTERMEDIATE}")
    print(f"Final outputs        → {OUTPUTS}")

    # ── Step 0: SharePoint download (optional) ─────────────────────────────────
    # Prompts whether to pull the latest Excel files from SharePoint.
    # Uses browser-based device-code auth; token cached in pipeline/token_cache.bin.
    # Raw copies saved to pipeline/outputs/raw_excel/. Comment to skip entirely.
    s0 = _time_step("step_00_sharepoint", run_step_00, INTERMEDIATE)

    # ── Step 1: Raw data loading ───────────────────────────────────────────────
    # Reads from SpkData CSVs and db_python manual Excels. Comment to skip.
    s1 = _time_step("step_01_raw_data", run_step_01, INTERMEDIATE)

    # ── Step 2: Leaching performance ───────────────────────────────────────────
    # Builds project_col_id / project_sample_id and matches all feature tables.
    # Requires: s1. Comment to skip.
    s2 = _time_step("step_02_leaching", run_step_02, s1, INTERMEDIATE)

    # ── Step 3: Reactor exponential fitting ────────────────────────────────────
    # Fits a*(1-exp(-b*x))+c to each reactor curve. Independent of s1/s2.
    # Comment to skip.
    s3 = _time_step("step_03_reactors", run_step_03, INTERMEDIATE)

    # ── Step 4: Mineralogy grouping ────────────────────────────────────────────
    # Assigns project_sample_id to unmapped rows, sums mineral groups.
    # Independent. Comment to skip.
    s4 = _time_step("step_04_mineralogy", run_step_04, INTERMEDIATE)

    # ── Step 5: ML model datasets ──────────────────────────────────────────────
    # Merges leaching + mineralogy; builds catalyzed/control splits.
    # Requires: s2, s4. (s3 fit stats saved to outputs/ for reference only.)
    # Comment to skip.
    s5 = _time_step("step_05_model_datasets", run_step_05, s2, s4, INTERMEDIATE)

    # ── Step 6: Column averages for Power BI ───────────────────────────────────
    # Produces one averaged row per (project_sample_id, catalyst_status).
    # Requires: s5 (or reads from pipeline/outputs/ if s5 is commented out).
    # Comment to skip.
    s6 = _time_step("step_06_columns_reactors", run_step_06, s5, INTERMEDIATE)

    # ── Summary ────────────────────────────────────────────────────────────────
    total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total:.1f}s ({total/60:.1f} min)")
    print(f"{'='*60}")

    if s6 and 'df_averaged_csv' in s6:
        df_avg = s6['df_averaged_csv']
        print(f"\nFinal output: {len(df_avg)} rows in averaged catcontrol dataset")
        if not df_avg.empty:
            grouped_cols = [c for c in df_avg.columns if c.startswith('grouped_')]
            filled = (~df_avg[grouped_cols].isna().all(axis=1)).sum() if grouped_cols else 0
            print(f"  Rows with mineralogy data: {filled}/{len(df_avg)}")
