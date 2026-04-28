# Rosetta ETL Pipeline - Refactored Structure

## Overview
Complete refactoring of Rosetta ETL pipeline consolidating all configuration and utilities into modular, maintainable components.

## File Structure

```
pipeline/
├── config.py                 # All hardcoded data consolidated
├── utils.py                  # Shared utility functions with bug fixes
├── step_01_raw_data.py      # Load and merge raw data
├── step_02_leaching.py      # Process leaching performance data
├── step_03_reactors.py      # Process reactor and bottle data with PCA
├── step_04_mineralogy.py    # Process mineralogy data with grouping and PCA
├── step_05_model_datasets.py # Merge all data for ML models
├── step_06_columns_reactors.py # Prepare final column reactor datasets
├── pipeline.py              # Main orchestrator
├── intermediate/            # Intermediate processing outputs
└── outputs/                 # Final model-ready datasets
```

## Critical Bug Fixes Included

### 1. normalize_and_replace (config.py)
- **Bug**: Crashes on NaN values
- **Fix**: Guard with `if not isinstance(string, str): return string`

### 2. Column assignment (step_01_raw_data.py)
- **Bug**: Direct string assignment fails on Arrow arrays
- **Fix**: Use `astype(object)` before string assignment
```python
df['col_name'] = df['col_name'].astype(object)
df.loc[mask, 'col_name'] = new_value
```

### 3. fill_with_bins (step_03_reactors.py)
- **Bug**: Arrow arrays are read-only
- **Fix**: Convert to writable numpy array
```python
row = np.array(row, dtype=float)  # ensure writable
```

### 4. filter_increasing (utils.py)
- **Bug**: groupby.apply removes rows not matching filter
- **Fix**: Use diff mask instead
```python
df['_diff'] = df.groupby(group_cols)[value_col].diff()
increasing_mask = (df['_diff'] > 0) | (df['_diff'].isna())
df = df[increasing_mask].drop(columns=['_diff'])
```

### 5. preprocess_data (step_05_model_datasets.py)
- **Bug**: dropna removes id columns
- **Fix**: Protect id columns before/after dropna
```python
id_data = df[id_cols].copy()
df = df.dropna(axis=1)
# restore id columns
```

### 6. add_match_keys (step_06_columns_reactors.py)
- **Bug**: Uses np.nan instead of None for missing values
- **Fix**: Use None for better compatibility
```python
df[col] = None  # not np.nan
```

### 7. is_numeric_col (utils.py)
- **Bug**: Uses np.issubdtype which is unreliable
- **Fix**: Use pd.api.types.is_numeric_dtype
```python
from pandas.api.types import is_numeric_dtype
is_numeric_dtype(series)
```

### 8. warnings handling (step_06_columns_reactors.py)
- **Bug**: nanmean warnings flood logs
- **Fix**: Use warnings.catch_warnings context manager
```python
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    result = np.nanmean(values)
```

## Configuration (config.py)

### PATHS
All file paths centralized:
- SpkData source locations
- Reporting database path
- Pipeline intermediate/output directories

### COL_TO_SAMPLE_ID_MAP
Complete mapping of project_col_id → project_sample_id (200+ entries)
Consolidated from all replacement_dict entries across original scripts

### MINERALOGY_ID_MAP
Mapping: (project_name, start_cell) → project_sample_id
Identifies unmapped mineralogy rows via Excel table references

### MINERALOGY_SAMPLE_MAP
Mapping: (project_name, sample_value) → project_sample_id
Identifies rows by sample column when start_cell not available

### MINERALOGY_DUPLICATES
Rows requiring duplication:
- UGM2 coarse variant
- Project 015 6in/8in condition split

### LEACHING_COLS_TO_KEEP
Dictionary of columns with effect codes [control, catalyzed, model, description]
150+ properties for reactor data analysis

### PREDICTORS_DICT
ML model features with effect codes and human-readable names
Includes all mineralogy groupings and PCA components

### MANUAL_FILTERS
Documentation of intentional data exclusions with rationale:
- PVO1: causes model instability
- PVLS: residue data, incomparable to column data
- Project 022 low recovery: statistical outliers
- Supergene sample: excluded from training

## Step Functions

### step_01_raw_data.py
**Function**: `run() → dict`
- Loads all SpkData CSVs (leaching, column_summary, chemchar, mineralogy, etc.)
- Merges terminated projects with current data
- Normalizes column names
- Fixes col_name assignment (astype object fix)
- Builds df_master via data_for_tableau.py logic
- **Returns**: dict with 9 dataframes + df_master
- **Saves**: pipeline/intermediate/raw_data.csv

### step_02_leaching.py
**Function**: `run(step1_outputs) → dict`
- Applies COL_TO_SAMPLE_ID_MAP to build project_sample_id
- Matches chemchar features via project_sample_id # MATCH STEP
- Matches mineralogy features via project_sample_id # MATCH STEP
- Matches qemscan features via project_sample_id # MATCH STEP
- Builds df_leaching_performance with all matched features
- **Returns**: df_leaching_performance, col_to_match_mineralogy
- **Saves**: pipeline/intermediate/leaching_performance.csv

### step_03_reactors.py
**Function**: `run(step1_outputs) → dict`
- Loads reactor/bottles data from SpkData
- Column-by-column numeric conversion (not slice assignment)
- Duplicates reactor rows for project aliases
- fill_with_bins for Cu extraction curves (writable array fix)
- Performs PCA on reactor curve characteristics
- Fits exponential extraction models
- **Returns**: df_pca_for_rosetta, exponential models, combined power, leaching_cols_to_keep
- **Saves**: pipeline/intermediate/reactors_pca.csv, exponential_models.csv

### step_04_mineralogy.py
**Function**: `run() → dict`
- Loads df_mineralogy_modals.csv
- Applies MINERALOGY_ID_MAP for unmapped rows
- Applies MINERALOGY_SAMPLE_MAP for sample-based identification
- Applies MINERALOGY_DUPLICATES to create variants
- Computes grouped_* columns (primary sulfides, secondary, oxides, gangue, etc.)
- Performs PCA on grouped mineralogy
- **Returns**: df_mineralogy_grouped, pca_mineralogy_grouped
- **Saves**: pipeline/intermediate/mineralogy_grouped.csv

### step_05_model_datasets.py
**Function**: `run(step2, step3, step4) → dict`
- Merges leaching + reactors PCA + mineralogy
- Catalyzed/control project split
- preprocess_data with id_col protection + diff-mask filter
- Builds *_projects versions (catalyzed, control)
- Xfactor merges for model-ready features
- Column reordering for consistency
- **Returns**: 3 model datasets (catalyzed, control, combined)
- **Saves**: pipeline/outputs/df_model_*.csv

### step_06_columns_reactors.py
**Function**: `run(step5, step3) → dict`
- Loads model CSV
- apply_column_filters (MANUAL FILTER comments)
- add_match_keys with None (not np.nan) fix
- prepare_column_train_data with is_numeric_dtype fix + warnings catch
- Grouping/averaging by project_sample_id
- **Returns**: df_averaged, df_catalyzed_averaged, df_control_averaged
- **Saves**: pipeline/outputs/df_averaged_*.csv
- **Also saves**: OneDrive power_bi and columns folders

## pipeline.py - Main Orchestrator

```python
#!/usr/bin/env python3
import time
import os
from step_01_raw_data import run as step1
from step_02_leaching import run as step2
from step_03_reactors import run as step3
from step_04_mineralogy import run as step4
from step_05_model_datasets import run as step5
from step_06_columns_reactors import run as step6

# Create directories
os.makedirs('pipeline/intermediate', exist_ok=True)
os.makedirs('pipeline/outputs', exist_ok=True)

print("Rosetta ETL Pipeline Starting...")

# Each step on its own line (easy to comment out to disable)
start_time = time.time()

step1_time = time.time()
step1_outputs = step1()
print(f"Step 1 (Raw Data) completed in {time.time() - step1_time:.1f}s")

step2_time = time.time()
step2_outputs = step2(step1_outputs)
print(f"Step 2 (Leaching) completed in {time.time() - step2_time:.1f}s")

step3_time = time.time()
step3_outputs = step3(step1_outputs)
print(f"Step 3 (Reactors) completed in {time.time() - step3_time:.1f}s")

step4_time = time.time()
step4_outputs = step4()
print(f"Step 4 (Mineralogy) completed in {time.time() - step4_time:.1f}s")

step5_time = time.time()
step5_outputs = step5(step2_outputs, step3_outputs, step4_outputs)
print(f"Step 5 (Model Datasets) completed in {time.time() - step5_time:.1f}s")

step6_time = time.time()
step6_outputs = step6(step5_outputs, step3_outputs)
print(f"Step 6 (Columns Reactors) completed in {time.time() - step6_time:.1f}s")

print(f"\nPipeline completed in {time.time() - start_time:.1f}s")
```

## Design Patterns

### Error Handling
- All functions use try/except for CSV operations
- Functions handle missing columns gracefully
- NaN/None values treated appropriately per context

### Data Types
- Column-by-column conversion (not slice assignment)
- Always guard string operations with isinstance checks
- Use astype(object) before string assignment

### Performance
- All data loading is lazy where possible
- PCA performed once per dataset
- Intermediate results cached to CSV

### Testing
- Each step validates output shape and key columns
- Manual filter rationale documented
- Termiated projects integration verified

## Key Identifiers

### project_col_id
Unique column identifier: normalized(project_name + sheet_name)

### project_sample_id
Unique sample identifier: remapped via COL_TO_SAMPLE_ID_MAP for grouping similar columns

### project_sample_condition_id
Condition variant: identifies specific experimental conditions (6in/8in, catalyzed/control)

## Data Flow

```
Raw SpkData CSVs → Step 1 → Normalized + Master
                  ↓
            Leaching features
            + Matched chemchar/mineralogy/qemscan → Step 2
                  ↓
            df_leaching_performance
                  ↓
                  ├→ Reactor data + PCA + Exponential models → Step 3
                  │
                  ├→ Mineralogy grouping + PCA → Step 4
                  │
                  └→ All sources → Step 5 → Model datasets
                           ↓
                      Step 6 → Final averaged datasets
                           ↓
                      CSV outputs + OneDrive sync
```

## Files Generated

**Intermediate** (for debugging):
- raw_data.csv
- leaching_performance.csv
- reactors_pca.csv, exponential_models.csv
- mineralogy_grouped.csv

**Outputs** (ready for analysis):
- df_model_catalyzed_projects.csv
- df_model_control_projects.csv
- df_model_catcontrol_projects.csv
- df_averaged_catalyzed.csv
- df_averaged_control.csv
- df_averaged_catcontrol.csv

## Migration from Legacy

To migrate from old scripts:
1. Update PATHS in config.py to match your environment
2. Verify COL_TO_SAMPLE_ID_MAP has all projects (compare vs old replacement_dicts)
3. Run pipeline.py and compare outputs with legacy versions
4. Archive old notebooks once validated
