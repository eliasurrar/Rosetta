# Rosetta ML Pipeline - Refactored Architecture

## Overview

This is a fully refactored ETL pipeline for the Rosetta ML project. The pipeline transforms raw SpkData and Rosetta data into ML-ready datasets for copper leaching recovery prediction.

**Key improvements:**
- All hardcoded constants consolidated in `config.py`
- Modular step-based architecture (6 steps, each self-contained)
- Data retention fixes for pandas 2.x / Arrow compatibility
- Clear manual filter annotations
- Intermediate output saving after each step
- Easily configurable and debuggable

## Data Flow

```
Raw CSVs (SpkData)
    ↓
[STEP 01] Load raw data → leaching, chemchar, mineralogy, QEMSCAN
    ↓
[STEP 02] Leaching dataset → apply mappings, filters, expose data
    ↓
[STEP 03] Reactors PCA → exponential model, reactor features
    ↓
[STEP 04] Mineralogy → grouping, clustering, PCA
    ↓
[STEP 05] ML datasets → merge features, scaling, train-ready
    ↓
[STEP 06] Final outputs → column/reactor averaging, aggregation
    ↓
ML-Ready Datasets
```

## Directory Structure

```
pipeline/
├── config.py              # All hardcoded constants, mappings, filters
├── utils.py               # Shared utilities (pandas 2.x safe helpers)
├── pipeline.py            # Main orchestrator (disable steps here)
├── step_01_raw_data.py    # Load CSVs from SpkData & Rosetta
├── step_02_leaching.py    # Leaching dataset w/ mappings & filters
├── step_03_reactors_pca.py (stub - implement exponential model)
├── step_04_mineralogy.py  (stub - implement clustering)
├── step_05_model_datasets.py (stub - implement feature merging)
├── step_06_columns_reactors.py (stub - implement final outputs)
├── intermediate/          # CSV outputs from each step
├── outputs/               # Final ML-ready outputs
└── README.md              # This file
```

## Configuration (config.py)

All hardcoded constants are in `config.py`, clearly labeled with comments:

### Project Sample ID Mappings

**`PROJECT_COL_ID_TO_SAMPLE_ID`**
- Maps multiple column IDs to a single project_sample_id
- Groups similar columns together (e.g., all Project 01A columns → '01a_jetti_project_file_c')
- Used in step_02_leaching.py

**`MINERALOGY_STARTCELL_TO_SAMPLE_ID`**
- Maps (project_name, start_cell) pairs to project_sample_id
- Used in step_04_mineralogy.py for mineralogy data alignment
- From rosetta_mineralogy_clustering.py

### Filters (All marked with `# MANUAL FILTER:`)

**`COLUMN_FILTERS`**
- Remove PVO1 ("avoid weird behaviours")
- Remove PVLS (residues leached)
- Remove Project 022 with cu_recovery < 7%
- Remove supergene sample

**`REACTOR_MAX_TEMP_C`** (40.0)
- Threshold for filtering reactors by temperature

**`STOPPED_COLUMNS_CUTOFFS`**
- Special treatment: cut columns at specific leach_duration_days
- Applied in step_02_leaching.py

**`MINERALOGY_LOW_THRESHOLD`** (0.0005)
- Replace mineralogy values < threshold with 0

### Dynamic Columns

**`DYNAMIC_ARRAY_COLUMNS`**
- Time-varying columns that need special handling as numpy arrays
- Examples: leach_duration_days, cu_recovery_%, cumulative_catalyst_addition_kg_t

### Model Parameters

**`EXPONENTIAL_MODEL_DEFAULTS`** & **`EXPONENTIAL_MODEL_BOUNDS`**
- Model: `a * (1 - exp(-b * x)) + c`
- Initial guesses and parameter bounds for curve fitting

**`REACTOR_DATAPOINT_DROPS`**
- Specific data points to drop (per Monse's emails)
- E.g., (project, reactor_id): [time_points_to_drop]

## Utils (utils.py)

Shared helper functions with focus on pandas 2.x / Arrow compatibility:

### Data Type Helpers
- `safe_to_numeric(series)` - Handle Arrow arrays
- `is_numeric_safe(dtype)` - Check numeric dtype
- `normalize_dataframe_values(series)` - Lowercase, strip special chars

### Data Retention Fixes
- `safe_assign_string_column(df, col, values)` - Assign strings without NaN issues
- `dropna_preserve_ids(df, subset, id_cols)` - Drop NaNs but protect ID columns
- `cast_to_numeric_before_assign(df, col, values)` - Fix string-to-float assignment
- `groupby_with_mask(df, groupby_cols, mask_func)` - Replace apply(lambda g: g[mask])

### Transformations
- `apply_column_replacement_dict(series, dict)` - Apply regex-safe replacements
- `expand_rows_with_copies(df, copy_map)` - Duplicate and expand rows
- `apply_qemscan_transformations(df)` - Sum exposure categories

## Steps

### Step 01: Load Raw Data

**Input:** CSV files from SpkData and Rosetta folders
**Output:** Raw dataframes (leaching, chemchar, mineralogy, QEMSCAN)
**Key operations:**
- Load leaching performance from both active and terminated projects
- Force numeric columns
- Normalize sheet names (Tiger ROM handling)
- Load QEMSCAN and apply exposure transformations

### Step 02: Leaching Dataset

**Input:** Raw dataframes from Step 01
**Output:** df_leaching_performance (processed)
**Key operations:**
- Create project_col_id from project_name + sheet_name
- Apply `PROJECT_COL_ID_TO_SAMPLE_ID` mapping
- Apply column-level filters (PVO1, PVLS, Project 022, supergene)
- Handle dynamic array columns
- Handle ORP conversion (enh → ag_agcl with -223 offset)
- Project 015: Create condition variants (GRE inventory, holdup soln)
- Merge QEMSCAN exposure data

### Step 03: Reactors PCA

**Stub - Implement:**
- Load reactor time-series data
- Pivot to matrix (index=[project_name, start_cell], columns=[time_(day)])
- Apply reactor filters (temperature, duplications)
- Fill missing values with exponential extrapolation
- Fit exponential model
- Compute PCA on reactor curves
- Return: df_pca_for_rosetta, df_exponential_model, etc.

### Step 04: Mineralogy

**Stub - Implement:**
- Load mineralogy modals
- Apply start_cell → project_sample_id mappings
- Duplicate Elephant II UGM2 for coarse variant
- Duplicate Project 015 for 6in/8in variants
- Group minerals by type (copper sulfides, secondary copper, etc.)
- Apply low threshold (0.0005) → 0
- Compute PCA on grouped mineralogy
- K-means clustering
- Return: df_mineralogy_grouped, pca_mineralogy_grouped

### Step 05: ML Model Datasets

**Stub - Implement:**
- Merge leaching + reactor PCA features
- Merge leaching + mineralogy features
- Compute derived features (Fe/Cu ratios, sulfide equivalents)
- Interpolate reactor curve characterization
- Apply feature selection and scaling
- Return: ready-for-training datasets

### Step 06: Final Outputs

**Stub - Implement:**
- Group by project_sample_id (columns)
- Split by catalyst status (control vs catalyzed)
- Weekly averaging of time-varying data
- Merge final column and reactor datasets
- Save outputs to CSV

## Data Loss Points Identified

### Already Fixed

1. **String-to-float assignment** (pandas 2.x Arrow issue)
   - Use: `cast_to_numeric_before_assign()` before assigning to float columns
   
2. **Slice assignment data loss**
   - Use: Column-by-column assignment instead of `df[mask, col] = values`
   - Example: `df[col] = default; df.loc[mask, col] = values`

3. **NaN initialization in string columns**
   - Use: `safe_assign_string_column()` initializing with `None` instead of `np.nan`

4. **Aggressive dropna()**
   - Protect ID columns: `dropna_preserve_ids(df, subset, id_cols=['project_col_id'])`

5. **groupby().apply(lambda g: g[mask])** pattern
   - Use: `groupby_with_mask(df, groupby_cols, mask_func)` instead

### Remaining (Original Logic)

The following data filtering operations are preserved exactly as-is (intentional manual filters):

- **PVO1 removal:** "avoid weird behaviours"
- **PVLS removal:** "residues leached"
- **Project 022 low recovery filter:** cu_recovery_% < 7
- **Supergene sample removal:** project_sample_id == '020_jetti_project_file_hypogene_supergene_super'
- **Stopped columns cutoff:** Rows beyond specific leach_duration_days
- **Reactor temperature filter:** temp_(c)_mean > 40
- **Catalyst assignment (control projects):** Set cumulative_catalyst_addition_kg_t = 0

All preserved with `# MANUAL FILTER:` comments.

## Running the Pipeline

### Run All Steps

```python
from pipeline import main

state = main(
    spkdata_folder='/Users/administration/Library/CloudStorage/OneDrive-JettiResources/PythonProjects/SpkData/Jetti01',
    rosetta_folder='/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Rosetta',
    pipeline_folder='/sessions/nifty-awesome-keller/mnt/Rosetta/pipeline',
    enable_steps=[1, 2, 3, 4, 5, 6],
)
```

### Run Selective Steps

```python
# Skip PCA/mineralogy (for quick testing)
state = main(..., enable_steps=[1, 2, 6])

# Only load and process leaching
state = main(..., enable_steps=[1, 2])
```

### Disable Individual Steps

Edit `pipeline.py` and comment out step runs:

```python
# To skip step 3, comment this line:
# if 3 in enable_steps:
#     state.update(step03_run(...))
```

## Output Files

### Intermediate Outputs (intermediate/)

Each step saves its output:
- `step_01_leaching_performance_raw.csv`
- `step_02_leaching_performance_processed.csv`
- `step_03_reactors_pca.csv`
- `step_04_mineralogy_grouped.csv`
- `step_05_model_features.csv`
- `step_06_final_columns.csv`, `step_06_final_reactors.csv`

### Final Outputs (outputs/)

Ready for ML training:
- `columns_final.csv` - Column test data with all features
- `reactors_final.csv` - Reactor data with all features
- `scaler.pkl` - Fitted StandardScaler for features
- `feature_list.txt` - Column names and order

## Extending the Pipeline

To add a new step:

1. Create `step_0X_name.py`:
   ```python
   def run(inputs: dict, output_folder: str) -> dict:
       """Process data."""
       df = inputs['df_name'].copy()
       # ... transformations ...
       df.to_csv(f'{output_folder}/step_0X_output.csv')
       return {'df_output': df}
   ```

2. Add to `pipeline.py`:
   ```python
   if X in enable_steps:
       print(f"\n[STEP 0X] ...")
       try:
           state.update(step0X_run(state, intermediate))
       except Exception as e:
           print(f"  ✗ Step 0X failed: {e}")
   ```

3. Update `enable_steps` as needed:
   ```python
   main(..., enable_steps=[1, 2, 3, 4, 5, 6, 7])
   ```

## Notes

- All hardcoded values are in `config.py` for easy auditing and modification
- Every manual filter is clearly marked with `# MANUAL FILTER:` comment and reason
- Intermediate CSVs allow debugging each step independently
- Pandas 2.x Arrow compatibility is built in from the start
- No mathematical computations are changed; logic is 100% faithful to originals

## TODO

- [ ] Implement step_03_reactors_pca.py (exponential model + PCA)
- [ ] Implement step_04_mineralogy.py (clustering + grouping)
- [ ] Implement step_05_model_datasets.py (feature merging + scaling)
- [ ] Implement step_06_columns_reactors.py (final aggregation)
- [ ] Add unit tests for each step
- [ ] Document exact R² and fit statistics from exponential model
- [ ] Add data quality checks and reporting
