#%%

from base64 import b16decode
import os
from matplotlib import axes
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns
import sys
import datetime as dt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from sympy import N

from functions_general import normalize_dataframe_values, normalize_and_replace, dataframe_to_python_code
from plot_helpers import show_or_autoclose_plot

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

# Configuración matplotlib y seaborn
# ==============================================================================

plt.rcParams['image.cmap'] = "bwr"
plt.rcParams['figure.dpi'] = "300"
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['font.size'] = '15'
plt.style.use('ggplot')
sns.set_context("paper", font_scale=1.0)  # paper, notebook, poster, talk
# sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})

# ==============================================================================
folder_path_load = '/Users/administration/OneDrive - Jetti Resources/PythonProjects/SpkData/Jetti01/'
folder_path_save = '/Users/administration/OneDrive - Jetti Resources/Reporting/db_python/csv_inputs'

min_thresh = 0.8 # 0.9

#%%

def duplicate_reactor_label_rows(df, alias_map):
    """Duplicate reactor rows under alternate labels expected downstream."""
    aliased_rows = []
    missing_sources = []

    for target_key, source_key in alias_map.items():
        target_project, target_start_cell = target_key
        source_project, source_start_cell = source_key

        target_mask = (
            (df['project_name'] == target_project) &
            (df['start_cell'] == target_start_cell)
        )
        if target_mask.any():
            continue

        source_mask = (
            (df['project_name'] == source_project) &
            (df['start_cell'] == source_start_cell)
        )
        if not source_mask.any():
            missing_sources.append({
                'target_project_name': target_project,
                'target_start_cell': target_start_cell,
                'source_project_name': source_project,
                'source_start_cell': source_start_cell,
            })
            continue

        aliased_rows.append(
            df.loc[source_mask].assign(
                project_name=target_project,
                start_cell=target_start_cell,
            )
        )

    if aliased_rows:
        df = pd.concat([df] + aliased_rows, ignore_index=True)

    return df, missing_sources


def build_reactor_label_audit(leaching_map, pivot_index, model_index, filtered_index):
    """Track which project/reactor labels survive the reactor processing pipeline."""
    model_frame = model_index.to_frame(index=False)
    rows = []

    for project_col_id, (project_name, start_cell, sample_id, catalyzed_y_n, ongoing_y_n) in leaching_map.items():
        key = (project_name, start_cell)
        same_start_cell_projects = sorted(
            model_frame.loc[model_frame['start_cell'] == start_cell, 'project_name']
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        same_project_reactors = sorted(
            model_frame.loc[model_frame['project_name'] == project_name, 'start_cell']
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        rows.append({
            'project_col_id': project_col_id,
            'project_name': project_name,
            'start_cell': start_cell,
            'project_sample_id': sample_id,
            'catalyzed_y_n': catalyzed_y_n,
            'ongoing_y_n': ongoing_y_n,
            'available_in_pivot': key in pivot_index,
            'available_in_exponential_model': key in model_index,
            'available_in_filtered_exponential_model': key in filtered_index,
            'same_start_cell_projects': ', '.join(same_start_cell_projects),
            'same_project_reactors': ', '.join(same_project_reactors),
        })

    return pd.DataFrame(rows)

df_reactors = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/dataset_reactor_summary_detailed.csv', low_memory=False)
# df_reactors['count_unique'] = df_reactors['project_name'] + '_' + df_reactors['start_cell']
# df_reactors['count_unique'].unique().size
# line added because they change column names
df_reactors['time_(day)'] = df_reactors['time_(day)'].fillna(df_reactors['time_(days)'])
for col in df_reactors.columns[5:]:
    df_reactors[col] = pd.to_numeric(df_reactors[col], errors='coerce')

df_bottles = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/dataset_rolling_bottles_detailed.csv')
for col in df_bottles.columns[5:]:
    df_bottles[col] = pd.to_numeric(df_bottles[col], errors='coerce')

df_reactors_terminated = pd.read_excel('/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/dataset_reactor_summary_detailed_terminated_projects.xlsx')
for col in df_reactors_terminated.columns[5:]:
    df_reactors_terminated[col] = pd.to_numeric(df_reactors_terminated[col], errors='coerce')

# merge with terminated
df_reactors = pd.concat([df_reactors, df_reactors_terminated], axis=0)

# REPLACE PROJECT 011 AND 020 HAR REACTORS FOR BOTTLES: # as from March 19th 2025 Project 011 has its own reactors (Crushed File)
# df_reactors = df_reactors.drop(index=df_reactors[df_reactors['project_name'] == '011 Jetti Project File'].index)
# df_reactors = df_reactors.drop(index=df_reactors[df_reactors['project_name'] == '020 Jetti Project File Hardy and Waste'].index)

'''
# COPY PROJECT 003 AND PLACE ON PROJECT 003 AS WELL
df_reactors.loc[df_reactors['project_name'] == '015 Jetti Project File'][['project_name', 'start_cell', 'time_(day)', 'cu_extraction_actual_(%)']]
df_reactors.loc[df_reactors['project_name'] == '003 Jetti Project File'][['project_name', 'time_(day)', 'cu_extraction_actual_(%)']]
'''

# USE SAME REACTORS FOR PROJECT 011 AND 011 CRUSHED (copy the data from 011 and paste on 011 crushed)
# df_reactors.loc[df_reactors['project_name'] == '011 Jetti Project File-Crushed', 'project_name'] = '011 Jetti Project File'
# df_reactors.loc[df_reactors['project_name'] == '011 Jetti Project File'][['project_name', 'start_cell', 'time_(day)', 'cu_extraction_actual_(%)']]
# Copy 011 reactor rows and duplicate them as 011 Crushed
df_reactors = pd.concat(
    [
        df_reactors,
        df_reactors[df_reactors['project_name'] == '011 Jetti Project File']
        .assign(project_name='011 Jetti Project File-Crushed')
    ],
    ignore_index=True
)


# EXCHANGE / ADD REACTORS FROM ZALDIVAR, TOQUEPALA AND ELEPHANT TO THE ONES IN PROJECT 007
# df_reactors.loc[df_reactors['project_name'] == '007 Jetti Project File - Zaldivar', 'project_name'] = 'Jetti Project File - Zaldivar SCL'
# df_reactors.loc[df_reactors['project_name'] == '007 Jetti Project File - Toquepala', 'project_name'] = 'Jetti Project File - Toquepala SCL'
# df_reactors.loc[df_reactors['project_name'] == '007 Jetti Project File - Elephant, Leopard, Tiger', 'project_name'] = 'Jetti Project File - Elephant SCL' # careful if duplicated nams of reactors, drop originals if so

# Zaldivar and Toquepala
df_reactors = pd.concat([df_reactors, df_reactors[df_reactors['project_name'] == '007 Jetti Project File - Zaldivar'].
                         assign(project_name='Jetti Project File - Zaldivar SCL')], ignore_index=True)
df_reactors = pd.concat([df_reactors, df_reactors[df_reactors['project_name'] == '007 Jetti Project File - Toquepala'].
                         assign(project_name='Jetti Project File - Toquepala SCL')], ignore_index=True)

# Elephant (first remove data point already existing)
# df_reactors = df_reactors.loc[(df_reactors['project_name'] == 'Jetti Project File - Elephant SCL') & (df_reactors['start_cell'] == 'tbl-RT_8')]
df_reactors = df_reactors.drop(
    df_reactors[(df_reactors['project_name'] == 'Jetti Project File - Elephant SCL') & 
                (df_reactors['start_cell'] == 'tbl-RT_8')].index
)
df_reactors = pd.concat([df_reactors, 
                         df_reactors[(df_reactors['project_name'] == '007 Jetti Project File - Elephant, Leopard, Tiger') &
                                     (df_reactors['sheet_name'] == 'RT Summary Elephant_Leopard') &
                                     (df_reactors['start_cell'].isin(['tbl-RT_5R', 'tbl-RT_6R', 'tbl-RT_8']))].
                         assign(project_name='Jetti Project File - Elephant SCL')], ignore_index=True)

# Leopard
df_reactors = pd.concat([df_reactors, 
                         df_reactors[(df_reactors['project_name'] == '007 Jetti Project File - Elephant, Leopard, Tiger') &
                                     (df_reactors['sheet_name'] == 'RT Summary Elephant_Leopard') &
                                     (df_reactors['start_cell'].isin(['tbl-RT_2R', 'tbl-RT_4R']))].
                         assign(project_name='Jetti Project File - Leopard SCL')], ignore_index=True)

                
df_reactors = pd.concat([df_reactors, df_bottles], axis=0, ignore_index=True)
df_reactors = df_reactors[df_reactors['cu_extraction_actual_(%)'] > 0]

# The maker index now points the original 011 sample to RT_21/RT_24, but those
# curves are stored under the crushed project label in the raw reactor export.
reactor_label_aliases = {
    ('011 Jetti Project File', 'tbl-RT_21'): ('011 Jetti Project File-Crushed', 'tbl-RT_21'),
    ('011 Jetti Project File', 'tbl-RT_24'): ('011 Jetti Project File-Crushed', 'tbl-RT_24'),
}
df_reactors, missing_reactor_aliases = duplicate_reactor_label_rows(df_reactors, reactor_label_aliases)
if missing_reactor_aliases:
    print("⚠️ Missing reactor label aliases:")
    print(pd.DataFrame(missing_reactor_aliases).to_string(index=False))

# Round `time_(day)` to the nearest 0.5
df_reactors['time_(day)'] = np.round(df_reactors['time_(day)'] * 4) / 4
# Keeping only the last entry per group
df_reactors = df_reactors.groupby(['project_name', 'sheet_name', 'catalyzed', 'start_cell', 'time_(day)']).last().reset_index()


# Group by project_name and start_cell, pivoting to create a time-series matrix for cu_extraction_actual_%
df_pivot = df_reactors.pivot_table(
    index=['project_name', 'start_cell'],
    columns='time_(day)',
    values='cu_extraction_actual_(%)'
)

# Calculate representative values for ph, orp_(mv), and cumulative_catalyst_(kg_t) for each (project_name, start_cell)
df_factors = df_reactors.groupby(['project_name', 'start_cell']).agg({
    'ph': 'mean',  # Choose appropriate statistic (e.g., 'mean', 'median', or 'last')
    'orp_(mv)': 'mean',
    'cumulative_catalyst_(kg_t)': 'last'
})

# drop some data points based on Monse's email on March 27th 2025
df_pivot.loc[df_pivot.index == ('014 Jetti Project File', 'tbl-RTB_7'), 42.25] = np.nan
df_pivot.loc[df_pivot.index == ('017 Jetti Project File', 'tbl-RTEA_1'), 48.0] = np.nan
df_pivot.loc[df_pivot.index == ('017 Jetti Project File', 'tbl-RTEA_1'), 42.0] = np.nan


# drop some of 020 to fit better
df_pivot.loc[df_pivot.index == ('020 Jetti Project File Hypogene_Supergene', 'tbl-RT_19'), [0.25, 0.5, 1.0, 2.0, 3.0]] = np.nan
df_pivot.loc[df_pivot.index == ('020 Jetti Project File Hypogene_Supergene', 'tbl-RT_20'), [0.25, 0.5, 1.0, 2.0, 3.0]] = np.nan


# visualize project 026 reactors
df_pivot[df_pivot.index == ('026 Jetti Project File', 'tbl-RT_1')]

#%%
# Function to fill missing values using bins with linear interpolation and extrapolation
def fill_with_bins(row, times, bin_size=5, cap_value=99.5, slope_decay=0.99):
    row = np.array(row, dtype=float)  # ensure a writable float array (Arrow arrays are read-only)
    # Ensure the first value is 0 if the first column name is 0
    if times[0] == 0:
        row[0] = 0

    # Drop NaN values for initial interpolation
    valid_indices = ~np.isnan(row)
    x_valid = times[valid_indices]
    y_valid = row[valid_indices]

    if len(x_valid) < 2:  # Insufficient data for interpolation
        return row  # Return unchanged

    # Step 0: Ensure valid `y` values are monotonically non-decreasing
    for i in range(len(y_valid) - 1, 0, -1):  # Iterate backwards
        if y_valid[i - 1] > y_valid[i]:
            y_valid[i - 1] = np.nan  # Adjust to make it non-decreasing

    # Recalculate valid indices after modifications
    valid_indices = ~np.isnan(y_valid)
    x_valid = x_valid[valid_indices]
    y_valid = y_valid[valid_indices]

    if len(x_valid) < 2:  # Check again after adjustments
        return row  # Return unchanged

    # Step 1: Linear Interpolation for missing values within the range
    y_interpolated = np.interp(times, x_valid, y_valid)

    # Step 2: Linear Extrapolation beyond the last valid value
    extrapolated = np.copy(y_interpolated)
    last_valid_idx = np.max(np.where(~np.isnan(row)))  # Use original row for this

    # Initialize the previous slope
    prev_slope = None

    for i in range(last_valid_idx + 1, len(times)):
        start_idx = max(0, i - bin_size)
        end_idx = i
        x_bin = times[start_idx:end_idx]
        y_bin = extrapolated[start_idx:end_idx]

        if len(x_bin) < 2:  # Insufficient data for linear regression
            continue

        # Perform linear regression to get slope and intercept
        slope, intercept = np.polyfit(x_bin, y_bin, 1)
        
        # Apply slope decay 
        if prev_slope is not None and prev_slope > 0 and prev_slope < 1.0:
            slope = prev_slope * slope_decay
        elif prev_slope is not None and prev_slope >= 1.0:
            slope = prev_slope * (slope_decay * 0.33)
            
        # Calculate the new extrapolated value
        new_value = slope * times[i] + intercept

        # Ensure the new value does not exceed the cap value
        extrapolated[i] = min(new_value, cap_value)

        # Ensure the new value does not drop below the last valid value
        if extrapolated[i] < extrapolated[i - 1]:
            extrapolated[i] = extrapolated[i - 1]

        # Update the previous slope
        prev_slope = slope

    return extrapolated


def fill_with_bins_smoother(row, times, bin_size=5, cap_value=99.5, slope_decay=0.99):
    # Ensure the first value is 0 if the first column name is 0
    if times[0] == 0:
        row[0] = 0

    # Drop NaN values for initial interpolation
    valid_indices = ~np.isnan(row)
    x_valid = times[valid_indices]
    y_valid = row[valid_indices]

    if len(x_valid) < 2:  # Insufficient data for interpolation
        return np.full_like(times, np.nan, dtype=np.float64)  # Ensure output shape is maintained

    def smooth_trend(y_values, times, window_size=3, tolerance=0.75):
        """
        Ensures that values follow a smooth increasing trend within a moving window.
        """
        y_smoothed = y_values.copy()
        
        for i in range(len(y_values) - window_size + 1):
            start_idx = i
            mid_idx = i + window_size // 2
            end_idx = i + window_size - 1

            y_start, y_mid, y_end = y_smoothed[start_idx], y_smoothed[mid_idx], y_smoothed[end_idx]
            t_start, t_mid, t_end = times[start_idx], times[mid_idx], times[end_idx]

            if np.isnan([y_start, y_mid, y_end]).any():
                continue  # Skip if there's NaN in the window

            # Expected value at mid_idx using linear interpolation
            expected_y_mid = y_start + (y_end - y_start) * ((t_mid - t_start) / (t_end - t_start))

            # Check if mid value deviates too much
            if abs(y_mid - expected_y_mid) > tolerance * abs(y_end - y_start):
                y_smoothed[mid_idx] = expected_y_mid
        
        return y_smoothed

    # Apply smoothing before interpolation
    y_valid = smooth_trend(y_valid, x_valid, window_size=3, tolerance=0.15)

    # Recalculate valid indices after modifications
    valid_indices = ~np.isnan(y_valid)
    x_valid = x_valid[valid_indices]
    y_valid = y_valid[valid_indices]

    if len(x_valid) < 2:  # Ensure enough points remain after smoothing
        return np.full_like(times, np.nan, dtype=np.float64)

    # Step 1: Linear Interpolation
    y_interpolated = np.interp(times, x_valid, y_valid)

    # Step 2: Linear Extrapolation Beyond the Last Valid Value
    extrapolated = np.copy(y_interpolated)
    last_valid_idx = np.max(np.where(~np.isnan(row)))  # Use original row for this

    prev_slope = None

    for i in range(last_valid_idx + 1, len(times)):
        start_idx = max(0, i - bin_size)
        end_idx = i
        x_bin = times[start_idx:end_idx]
        y_bin = extrapolated[start_idx:end_idx]

        if len(x_bin) < 2:
            continue

        # Perform linear regression
        slope, intercept = np.polyfit(x_bin, y_bin, 1)
        
        # Apply slope decay
        if prev_slope is not None:
            if prev_slope > 0 and prev_slope < 1.0:
                slope = prev_slope * slope_decay
            elif prev_slope >= 1.0:
                slope = prev_slope * (slope_decay * 0.33)

        # Compute new value and enforce constraints
        new_value = slope * times[i] + intercept
        new_value = min(new_value, cap_value)  # Cap the value

        # Ensure monotonicity
        # extrapolated[i] = max(new_value, extrapolated[i - 1])
        # Ensure the new value does not drop below the last valid value
        extrapolated[i] = min(new_value, cap_value)
        if extrapolated[i] < extrapolated[i - 1]:
            extrapolated[i] = extrapolated[i - 1]
            
        prev_slope = slope

    # Smooth the extrapolated values
    valid_indices = ~np.isnan(extrapolated)
    times_valid = times[valid_indices]  # Ensure times are correctly aligned
    extrapolated = extrapolated[valid_indices]
    
    if len(extrapolated) > 0:  # Avoid empty array errors
        extrapolated = smooth_trend(extrapolated, times_valid, window_size=3, tolerance=0.05)
    
    # Smooth the extrapolated values
    valid_indices = ~np.isnan(extrapolated)
    times_valid = times[valid_indices]  # Ensure times are correctly aligned
    extrapolated = extrapolated[valid_indices]
    
    # Ensure the returned array is the same length as `times`
    final_output = np.full_like(times, np.nan, dtype=np.float64)
    
    # Assign extrapolated values back
    final_output[valid_indices] = extrapolated  

    return final_output

#%%

df = df_pivot.copy()

def exponential_model(x, a, b, c):
    return a * (1 - np.exp(-b * x)) + c # c term added after monse's email March 27th 2025, based on high secondaries for some samples.

# Original columns from the DataFrame
original_columns = df.columns.astype(float)

# Create a new DataFrame with the same index and original columns
new_df = pd.DataFrame(index=df.index, columns=original_columns)

# Create a DataFrame to store fit statistics
fit_stats = pd.DataFrame(index=df.index, columns=["a", "b", "c", "R_squared", "RMSE"])

# Calculate the minimum 'b' to ensure 99% of asymptote 'a' is reached by day 125
plateau_forced = 200
b_min = -np.log(0.01) / plateau_forced  # Ensures 1 - exp(-b*125) >= 0.99

for idx in df.index:
    row_data = df.loc[idx]
    mask = row_data.notna()
    x_data = row_data.index[mask].astype(float)
    y_data = row_data[mask].values.astype(float)
    
    if len(y_data) == 0:
        # If no data, default to asymptote 99.5 and minimal b
        a_fit = 99.5
        b_fit = b_min
        c_fit = 0 # Default value for c
        r_squared = np.nan
        rmse = np.nan
    else:
        max_y = np.max(y_data)
        # Ensure lower bound for 'a' is strictly less than upper bound
        a_lower = max_y if max_y < 99.5 else 99.4  # Adjust if max_y >= 99.5
        a_upper = 99.5
        # Initial guesses
        a_guess = max_y if max_y < 99.5 else 99.5
        b_guess = max(b_min, 1.0)  # Start with higher b if possible
        c_lower = 0.0
        c_upper = 20.0
        c_guess = max(c_lower, c_upper)
        
        # Assign higher weights to later data points (lower sigma = higher weight)
        n = len(y_data)
        if n > 0:
            # Weight later points more (e.g., linearly increasing weights)
            weights = np.linspace(1, 10, n)  # Weights increase from 1 to 10
            sigma = 1 / np.sqrt(weights)  # Convert weights to sigma
            # Ensure the last point has the smallest sigma (highest weight)
            # sigma[-1] *= 0.1  # Further emphasize the last point
            # sigma[-2] *= 0.2  # Emphasize the second-to-last point
            # sigma[-3] *= 0.3  # Emphasize the third-to-last point
            # sigma[-4] *= 0.3
            # sigma[-5] *= 0.3
        else:
            sigma = None
        
        try:
            params, _ = curve_fit(exponential_model, x_data, y_data,
                                  p0=(a_guess, b_guess, c_guess),
                                  bounds=([a_lower, b_min, c_lower], [a_upper, np.inf, c_upper]),
                                  sigma=sigma,
                                  maxfev=10000)
            a_fit, b_fit, c_fit = params
            
            # Calculate R-squared
            y_pred_fit = exponential_model(x_data, a_fit, b_fit, c_fit)
            ss_res = np.sum((y_data - y_pred_fit) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((y_data - y_pred_fit) ** 2))
        except RuntimeError:
            # If fitting fails, use max_y as a and enforce b_min
            a_fit = min(max_y, 99.5)
            b_fit = b_min
            c_fit = 0.0
            r_squared = np.nan
            rmse = np.nan
    
    # Predict for all original columns
    y_pred = exponential_model(original_columns, a_fit, b_fit, c_fit)
    new_df.loc[idx] = y_pred
    
    # Save fit statistics
    fit_stats.loc[idx] = [a_fit, b_fit, c_fit, r_squared, rmse]


# Ensure all values are filled and convert to float
new_df = new_df.astype(float)
fit_stats = fit_stats.astype({"a": float, "b": float, "c": float, "R_squared": float, "RMSE": float})
fit_stats.to_csv(folder_path_save + '/reactors_expmodel_fit_stats.csv')

# Display the fit statistics DataFrame
print("Fit Statistics:")
print(fit_stats)


#%%
# Ensure times are numeric
times = df_pivot.columns.to_numpy(dtype=float)
df_pivot_increasing = df_pivot.copy()

# Creating a new DataFrame with interpolated and extrapolated values
df_filled = new_df.copy()
# df_filled_4params = new_df_4params.copy()

# Interpolating and extrapolating for each row
filled_data = {}
for row_idx in df_pivot.index:
    filled_data[row_idx] = fill_with_bins(pd.to_numeric(df_pivot_increasing.loc[row_idx]).to_numpy(), times, bin_size=30, cap_value=99.0, slope_decay=0.97)

# Creating a new DataFrame with interpolated and extrapolated values
df_pivot_interpolated_log = pd.DataFrame(filled_data, index=times).T



#%%
# Apply the conditional interpolation function row-wise
# df_pivot_interpolated_log = df_pivot.apply(fill_nans_linear_with_bins_power, axis=1)


# Exponentiate to revert the log transformation
df_pivot_extrapolated = df_pivot_interpolated_log.copy()
df_pivot_extrapolated.iloc[:, 0] = 0
# df_filled.iloc[:, 0] = 0 # commented after adding 'c' parameter on exponential model
# df_filled_4params.iloc[:, 0] = 0


# Define the filtering logic for columns
filtered_columns = [
    col for col in df_pivot_extrapolated.columns
    if (float(col) >= 5.0 and str(col).endswith(".0")) or float(col) < 5.0 or float(col) >= 125.0
]

# Subset the DataFrame based on filtered columns
df_combined_power = df_pivot_extrapolated[filtered_columns]
df_exponential_model = df_filled[filtered_columns]
# df_exponential_model_4params = df_filled_4params[filtered_columns]


# drop unnecessary data
df_combined_power = df_combined_power.drop(index=('003 Jetti Project File', 'tbl-RT_10R')) # weird values
df_combined_power = df_combined_power.drop(index=('024 Jetti Project File', 'tbl-RT_2')) # weird values

# convert to string the column names
df_combined_power.columns = df_combined_power.columns.astype(str)
df_exponential_model.columns = df_exponential_model.columns.astype(str)
# df_exponential_model_4params.columns = df_exponential_model_4params.columns.astype(str)


#%%
# plot to see (first control, then catalyzed for color match)
to_plot = [
    #('017 Jetti Project File', 'tbl-RTEA_1'),
    #('017 Jetti Project File', 'tbl-RTEA_2'),
    #('022 Jetti Project File', 'tbl-RT_1'),
    #('022 Jetti Project File', 'tbl-RT_2'),
    #('014 Jetti Project File', 'tbl-RTB_8'),
    #('014 Jetti Project File', 'tbl-RTB_7'),
    ('15289-004 - Column Leach_v1 2020-11-30', 'tbl-RT_BAG_Control'),
    ('15289-004 - Column Leach_v1 2020-11-30', 'tbl-RT_BAG_Catalyzed'),
    #('020 Jetti Project File Hypogene_Supergene', 'tbl-RT_1'), #control
    #('020 Jetti Project File Hypogene_Supergene', 'tbl-RT_2'), #catalyzed
    ('024 Jetti Project File', 'tbl-RT_1'), #control
    ('024 Jetti Project File', 'tbl-RT_3'), #catalyzed
    ('Jetti Project File - Zaldivar SCL', 'tbl-RT_1'), #control
    ('Jetti Project File - Zaldivar SCL', 'tbl-RT_2'), #catalyzed
    ('15289-01A Column Leach_2020-02-19', 'tbl-R_3'), # control
    ('15289-01A Column Leach_2020-02-19', 'tbl-R_4'), # catalyzed
]



for z in [df_pivot, df_combined_power, df_exponential_model]:
    df_to_plot = z.loc[to_plot]
    df_to_plot = df_to_plot.T
    df_to_plot.index = pd.to_numeric(df_to_plot.index, errors='coerce')
    df_to_plot = df_to_plot.apply(pd.to_numeric)

    # Set the color palette
    palette = sns.color_palette('Paired', n_colors=len(df_to_plot.columns))
    
    # Iterate over columns and create scatter plots
    for idx, column in enumerate(df_to_plot.columns):
        sns.lineplot(
            x=df_to_plot.index, 
            y=df_to_plot[column], 
            label=column, 
            color=palette[idx]
        )
    
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.ylim(0, 100)
    plt.xlim(0, 200)
    plt.legend(loc='lower right', fontsize='xx-small')
    show_or_autoclose_plot(plt)


# ======= DEFINE WHICH ONE TO USE
df_combined = df_exponential_model.copy()
df_combined.columns
list(df_combined.index.unique())

# ==================================
# Filter for PCA only for project that are being used only  (use because all reactors were carried out in different conditions)


leaching_cols_to_keep = { # project_col_id: [project_name, reactor, sample_id, catalyzed_y_n, ongoing_y_n]
    # '011_jetti_project_file_rm_1': ['011 Jetti Project File', 'tbl-BRM_4', '011_jetti_project_file_rm', 'catalyzed', 'ongoing'],
    # '011_jetti_project_file_rm_2': ['011 Jetti Project File', 'tbl-BRM_5', '011_jetti_project_file_rm', 'control', 'ongoing'],
    '011_jetti_project_file_rm_1': ['011 Jetti Project File', 'tbl-RT_24', '011_jetti_project_file_rm', 'catalyzed', 'ongoing'],
    '011_jetti_project_file_rm_2': ['011 Jetti Project File', 'tbl-RT_21', '011_jetti_project_file_rm', 'control', 'ongoing'],
    '011_jetti_project_filecrushed_011rm_5': ['011 Jetti Project File-Crushed', 'tbl-RT_21', '011_jetti_project_file_rm_crushed', 'control', 'ongoing'],
    '011_jetti_project_filecrushed_011rm_6': ['011 Jetti Project File-Crushed', 'tbl-RT_21', '011_jetti_project_file_rm_crushed', 'control', 'ongoing'],
    '011_jetti_project_filecrushed_011rm_7': ['011 Jetti Project File-Crushed', 'tbl-RT_24', '011_jetti_project_file_rm_crushed', 'catalyzed', 'ongoing'],
    '011_jetti_project_filecrushed_011rm_8': ['011 Jetti Project File-Crushed', 'tbl-RT_24', '011_jetti_project_file_rm_crushed', 'catalyzed', 'ongoing'],
    '014_jetti_project_file_b_4': ['014 Jetti Project File', 'tbl-RTB_8', '014_jetti_project_file_bag', 'catalyzed', 'terminated'],
    '014_jetti_project_file_b_2': ['014 Jetti Project File', 'tbl-RTB_7', '014_jetti_project_file_bag', 'control', 'terminated'],
    '014_jetti_project_file_k_4': ['014 Jetti Project File', 'tbl-RTK_8', '014_jetti_project_file_kmb', 'catalyzed', 'terminated'],
    '014_jetti_project_file_k_1': ['014 Jetti Project File', 'tbl-RTK_7', '014_jetti_project_file_kmb', 'control', 'terminated'],
    '015_jetti_project_file_c_12': ['015 Jetti Project File', 'tbl-RT_3', '015_jetti_project_file_amcf_6in', 'catalyzed', 'terminated'],
    '015_jetti_project_file_c_7': ['015 Jetti Project File', 'tbl-RT_2', '015_jetti_project_file_amcf_6in', 'control', 'terminated'], # NO HEAD SAMPLE QEMSCAN DONE AT SGS (Ben K added head sample for 003 and 015 on January 22nd 2025)
    '015_jetti_project_file_c_6': ['015 Jetti Project File', 'tbl-RT_3', '015_jetti_project_file_amcf_8in', 'catalyzed', 'terminated'],
    '015_jetti_project_file_c_11': ['015 Jetti Project File', 'tbl-RT_2', '015_jetti_project_file_amcf_8in', 'control', 'terminated'], # NO HEAD SAMPLE QEMSCAN DONE AT SGS (Ben K added head sample for 003 and 015 on January 22nd 2025)
    '017_jetti_project_file_ea_4': ['017 Jetti Project File', 'tbl-RTEA_2', '017_jetti_project_file_ea_mill_feed_combined', 'catalyzed', 'ongoing'],
    '017_jetti_project_file_ea_1': ['017 Jetti Project File', 'tbl-RTEA_1', '017_jetti_project_file_ea_mill_feed_combined', 'control', 'ongoing'],
    # Monse '020_jetti_project_file_hardy_and_waste_har_3': ['020 Jetti Project File Hardy and Waste', 'tbl-BR_3', '020_jetti_project_file_hardy_and_waste_h21_master_comp', 'catalyzed', 'terminated'],
    # Monse '020_jetti_project_file_hardy_and_waste_har_1': ['020 Jetti Project File Hardy and Waste', 'tbl-BR_1', '020_jetti_project_file_hardy_and_waste_h21_master_comp', 'control', 'terminated'],
    # Monse '020_jetti_project_file_hardy_and_waste_har_2': ['020 Jetti Project File Hardy and Waste', 'tbl-BR_1', '020_jetti_project_file_hardy_and_waste_h21_master_comp', 'control', 'terminated'], # NO HEAD SAMPLE QEMSCAN DONE AT SGS
    '020_jetti_project_file_hypogene_supergene_hyp_2': ['020 Jetti Project File Hypogene_Supergene', 'tbl-RT_2', '020_jetti_project_file_hypogene_supergene_hypogene_master_composite', 'catalyzed', 'ongoing'],
    '020_jetti_project_file_hypogene_supergene_hyp_1': ['020 Jetti Project File Hypogene_Supergene', 'tbl-RT_1', '020_jetti_project_file_hypogene_supergene_hypogene_master_composite', 'control', 'ongoing'],
    '020_jetti_project_file_hypogene_supergene_hyp_3': ['020 Jetti Project File Hypogene_Supergene', 'tbl-RT_1', '020_jetti_project_file_hypogene_supergene_hypogene_master_composite', 'control', 'ongoing'],
    '020_jetti_project_file_hypogene_supergene_sup_2': ['020 Jetti Project File Hypogene_Supergene', 'tbl-RT_20', '020_jetti_project_file_hypogene_supergene_super', 'catalyzed', 'terminated'],
    '020_jetti_project_file_hypogene_supergene_sup_1': ['020 Jetti Project File Hypogene_Supergene', 'tbl-RT_19', '020_jetti_project_file_hypogene_supergene_super', 'control', 'terminated'],
    '020_jetti_project_file_hypogene_supergene_sup_3': ['020 Jetti Project File Hypogene_Supergene', 'tbl-RT_19', '020_jetti_project_file_hypogene_supergene_super', 'control', 'terminated'],
    '022_jetti_project_file_s_2': ['022 Jetti Project File', 'tbl-RT_2', '022_jetti_project_file_stingray_1', 'catalyzed', 'terminated'],
    '022_jetti_project_file_s_1': ['022 Jetti Project File', 'tbl-RT_1', '022_jetti_project_file_stingray_1', 'control', 'terminated'],
    # '023_jetti_project_file_ot_10':['', '', '', ''],
    # '023_jetti_project_file_ot_9':['', '', '', ''],
    '024_jetti_project_file_cv_4': ['024 Jetti Project File', 'tbl-RT_3', '024_jetti_project_file_024cv_cpy', 'catalyzed', 'ongoing'],
    '024_jetti_project_file_cv_1': ['024 Jetti Project File', 'tbl-RT_1', '024_jetti_project_file_024cv_cpy', 'control', 'ongoing'],
    'jetti_project_file_leopard_scl_col1': ['Jetti Project File - Leopard SCL', 'tbl-RT_2', 'jetti_project_file_leopard_scl_sample_los_bronces', 'catalyzed', 'ongoing'],
    'jetti_project_file_leopard_scl_col2': ['Jetti Project File - Leopard SCL', 'tbl-RT_1', 'jetti_project_file_leopard_scl_sample_los_bronces', 'control', 'ongoing'], # (august 16th JU and MR leave out)
    'jetti_project_file_leopard_scl_rom1': ['Jetti Project File - Leopard SCL', 'tbl-RT_2', 'jetti_project_file_leopard_scl_sample_los_bronces', 'catalyzed', 'ongoing'], # previously tbl-RT_2
    'jetti_project_file_leopard_scl_rom2': ['Jetti Project File - Leopard SCL', 'tbl-RT_1', 'jetti_project_file_leopard_scl_sample_los_bronces', 'control', 'ongoing'], # previously tbl-RT_1
    # 'jetti_project_file_tiger_rom_c_4': ['', '', '', ''],
    # 'jetti_project_file_tiger_rom_c_5': ['', '', '', ''], (august 16th JU and MR leave out)
    # 'jetti_project_file_tiger_rom_rom2': ['', '', '', ''],
    # 'jetti_project_file_tiger_rom_rom3': ['', '', '', ''], (august 16th JU and MR leave out)
    # 'jetti_project_file_tiger_rom_c_6': ['', '', '', ''],
    # 'jetti_project_file_tiger_rom_c_7': ['', '', '', ''], (august 16th JU and MR leave out)
    # 'jetti_project_file_tiger_rom_c_8': ['', '', '', ''],
    # 'jetti_project_file_tiger_rom_c_9': ['', '', '', ''], (august 16th JU and MR leave out)
    'jetti_project_file_elephant_scl_col42': ['Jetti Project File - Elephant SCL', 'tbl-RT_8', 'jetti_project_file_elephant_scl_sample_escondida', 'catalyzed', 'terminated'], # previously tbl-RT_2
    'jetti_project_file_elephant_scl_col43': ['Jetti Project File - Elephant SCL', 'tbl-RT_5R', 'jetti_project_file_elephant_scl_sample_escondida', 'control', 'terminated'], # previously tbl-RT_1
    'jetti_project_file_toquepala_scl_col63': ['Jetti Project File - Toquepala SCL', 'tbl-RT_24', 'jetti_project_file_toquepala_scl_sample_fresca', 'catalyzed', 'terminated'], # previously tbl-RT_8
    'jetti_project_file_toquepala_scl_col64': ['Jetti Project File - Toquepala SCL', 'tbl-RT_21', 'jetti_project_file_toquepala_scl_sample_fresca', 'control', 'terminated'], # previously tbl-RT_11
    'jetti_project_file_zaldivar_scl_col69': ['Jetti Project File - Zaldivar SCL', 'tbl-RT_32', 'jetti_project_file_zaldivar_scl_sample_zaldivar', 'catalyzed', 'terminated'], # previously tbl-RT_2
    'jetti_project_file_zaldivar_scl_col70': ['Jetti Project File - Zaldivar SCL', 'tbl-RT_29', 'jetti_project_file_zaldivar_scl_sample_zaldivar', 'control', 'terminated'], # previously tbl-RT_1
    'jetti_project_file_elephant_(site)_fat4': ['Jetti Project File - Elephant SCL', 'tbl-RT_8', 'jetti_project_file_elephant_scl_sample_escondida', 'catalyzed', 'ongoing'], # fijarse en titulo proyecto # previously tbl-RT_2
    'jetti_project_file_elephant_(site)_fat6': ['Jetti Project File - Elephant SCL', 'tbl-RT_5R', 'jetti_project_file_elephant_scl_sample_escondida', 'control', 'ongoing'], # previously tbl-RT_1
    '003_jetti_project_file_be_2': ['003 Jetti Project File', 'tbl-RT_3', '003_jetti_project_file_amcf_head', 'catalyzed', 'terminated'],
    '003_jetti_project_file_be_1': ['003 Jetti Project File', 'tbl-RT_2', '003_jetti_project_file_amcf_head', 'control', 'terminated'], # NO HEAD SAMPLE QEMSCAN DONE AT SGS
    # Monse '012_jetti_project_file_cs_q_3': ['012 Jetti Project File CS', 'tbl-RT_C', '012_jetti_project_file_quebalix', 'control', 'terminated'],# ===== CONFIRMAR REACTORES =======
    # Monse '012_jetti_project_file_cs_q_2': ['012 Jetti Project File CS', 'tbl-RT_D', '012_jetti_project_file_quebalix', 'catalyzed', 'terminated'], # # ===== CONFIRMAR REACTORES ======= DISCARDED FOR CONTAMINATION ON SX (CANDICE OCT 23RD 2024)
    # Monse '012_jetti_project_file_cs_i_3': ['012 Jetti Project File CS', 'tbl-RT_E', '012_jetti_project_file_incremento', 'control', 'terminated'],  # ===== CONFIRMAR REACTORES =======
    # Monse '012_jetti_project_file_cs_i_2': ['012 Jetti Project File CS', 'tbl-RT_F', '012_jetti_project_file_incremento', 'catalyzed', 'terminated'], # # ===== CONFIRMAR REACTORES ======= DISCARDED FOR CONTAMINATION ON SX (CANDICE OCT 23RD 2024)
    '013_jetti_project_file_o_4': ['013 Jetti Project File', 'tbl-RTO_4', '013_jetti_project_file_combined', 'catalyzed', 'terminated'],
    '013_jetti_project_file_o_2': ['013 Jetti Project File', 'tbl-RTO_2', '013_jetti_project_file_combined', 'control', 'terminated'],
    '013_jetti_project_file_o_3': ['013 Jetti Project File', 'tbl-RTO_2', '013_jetti_project_file_combined', 'control', 'terminated'],
    '026_jetti_project_file_ps_1': ['026 Jetti Project File', 'tbl-RT_1', '026_jetti_project_file_sample_1_primary_sulfide', 'control', 'ongoing'],  # can be RT1 or RT2
    '026_jetti_project_file_ps_2': ['026 Jetti Project File', 'tbl-RT_1', '026_jetti_project_file_sample_1_primary_sulfide', 'control', 'ongoing'], # can be RT1 or RT2
    '026_jetti_project_file_ps_3': ['026 Jetti Project File', 'tbl-RT_4', '026_jetti_project_file_sample_1_primary_sulfide', 'catalyzed', 'ongoing'], 
    '026_jetti_project_file_ps_4': ['026 Jetti Project File', 'tbl-RT_4', '026_jetti_project_file_sample_1_primary_sulfide', 'catalyzed', 'ongoing'],
    '026_jetti_project_file_cr_2': ['026 Jetti Project File', 'tbl-RT_10', '026_jetti_project_file_sample_1_primary_sulfide', 'control', 'ongoing'],
    '026_jetti_project_file_cr_4': ['026 Jetti Project File', 'tbl-RT_12', '026_jetti_project_file_sample_1_primary_sulfide', 'catalyzed', 'ongoing'],
    '026_jetti_project_file_ss_2': ['026 Jetti Project File', 'tbl-RT_18', '026_jetti_project_file_sample_1_primary_sulfide', 'control', 'ongoing'],
    '026_jetti_project_file_ss_4': ['026 Jetti Project File', 'tbl-RT_20', '026_jetti_project_file_sample_1_primary_sulfide', 'catalyzed', 'ongoing'],
    # no columns for 026 oxides (mixed)
    'jetti_file_elephant_ii_ver_2_pq_pr_2': ['Jetti File - Elephant II Ver 2 PQ', '', 'jetti_file_elephant_ii_pq_rom', 'catalyzed', 'ongoing'],
    'jetti_file_elephant_ii_ver_2_pq_pr_1': ['Jetti File - Elephant II Ver 2 PQ', '', 'jetti_file_elephant_ii_pq_rom', 'control', 'ongoing'],
    # Monse 'jetti_file_elephant_ii_ver_2_pq_pc_4': ['Jetti File - Elephant II Ver 2 P Q', '', 'jetti_file_elephant_ii_pq_crushed', 'catalyzed', 'ongoing'],
    # Monse 'jetti_file_elephant_ii_ver_2_pq_pc_2': ['Jetti File - Elephant II Ver 2 PQ', '', 'jetti_file_elephant_ii_pq_crushed', 'control', 'ongoing'],
    'jetti_file_elephant_ii_ver_2_ugm_ur_2': ['Jetti File - Elephant II Ver 2 UGM', '', 'jetti_file_elephant_ii_ugm2_rom', 'catalyzed', 'ongoing'],
    'jetti_file_elephant_ii_ver_2_ugm_ur_1': ['Jetti File - Elephant II Ver 2 UGM', '', 'jetti_file_elephant_ii_ugm2_rom', 'control', 'ongoing'],
    'jetti_file_elephant_ii_ver_2_ugm_uc_4': ['Jetti File - Elephant II Ver 2 UGM', '', 'jetti_file_elephant_ii_ugm2_crushed', 'catalyzed', 'ongoing'],
    'jetti_file_elephant_ii_ver_2_ugm_uc_1': ['Jetti File - Elephant II Ver 2 UGM', '', 'jetti_file_elephant_ii_ugm2_crushed', 'control', 'ongoing'],
    'jetti_file_elephant_ii_ver_2_ugm_uc_3': ['Jetti File - Elephant II Ver 2 UGM', '', 'jetti_file_elephant_ii_ugm2_crushed', 'control', 'ongoing'],
    # Monse '01a_jetti_project_file_c7': ['15289-01A Column Leach_2020-02-19', 'tbl-R_3', '01a_jetti_project_file_c', 'control', 'terminated'], # zorro
    # Monse '01a_jetti_project_file_c9': ['15289-01A Column Leach_2020-02-19', 'tbl-R_4', '01a_jetti_project_file_c', 'catalyzed', 'terminated'],
    # '002_jetti_project_file_qb1': ['15289-002 - Column Leach_v1 2019-12-1', 'tbl-RT_1', '002_jetti_project_file_qb', 'control'], # dejar poryecto 002 fuera por ley de cabeza... MR nov 26th, Boulder
    # '002_jetti_project_file_qb2': ['15289-002 - Column Leach_v1 2019-12-1', 'tbl-RT_3', '002_jetti_project_file_qb', 'catalyzed'], # dejar poryecto 002 fuera por ley de cabeza... MR nov 26th, Boulder
    # '002_jetti_project_file_qb3': ['15289-002 - Column Leach_v1 2019-12-1', 'tbl-RT_3', '002_jetti_project_file_qb', 'catalyzed'], # dejar poryecto 002 fuera por ley de cabeza... MR nov 26th, Boulder
    # Monse '006_jetti_project_file_pvls1': ['15289-006 - Column Leach_v1.2020-08-28', '', '006_jetti_project_file_pvls', 'catalyzed', 'terminated'],
    '006_jetti_project_file_pvls2': ['15289-006 - Column Leach_v1.2020-08-28', '', '006_jetti_project_file_pvls', 'catalyzed', 'terminated'],
    '006_jetti_project_file_pvls3': ['15289-006 - Column Leach_v1.2020-08-28', '', '006_jetti_project_file_pvls', 'control', 'terminated'],
    # Monse '006_jetti_project_file_pvls4': ['15289-006 - Column Leach_v1.2020-08-28', '', '006_jetti_project_file_pvls', 'control', 'terminated'],
    '006_jetti_project_file_pvo1': ['15289-006 - Column Leach_v1.2020-08-28', 'tbl-RT_PVO_Catalyzed', '006_jetti_project_file_pvo', 'catalyzed', 'terminated'],
    '006_jetti_project_file_pvo2': ['15289-006 - Column Leach_v1.2020-08-28', 'tbl-RT_PVO_Catalyzed', '006_jetti_project_file_pvo', 'catalyzed', 'terminated'],
    '006_jetti_project_file_pvo3': ['15289-006 - Column Leach_v1.2020-08-28', 'tbl-RT_PVO_Control', '006_jetti_project_file_pvo', 'control', 'terminated'],
    # Monse '006_jetti_project_file_pvo4': ['15289-006 - Column Leach_v1.2020-08-28', 'tbl-RT_PVO_Control', '006_jetti_project_file_pvo', 'control', 'terminated'],
    # '004_jetti_project_file_mo1': ['15289-004 - Column Leach_v1 2020-11-30', 'tbl-RT_BAG_Catalyzed', '004_jetti_project_file_mo', 'catalyzed', 'terminated'],
    # '004_jetti_project_file_mo2': ['15289-004 - Column Leach_v1 2020-11-30', 'tbl-RT_BAG_Catalyzed', '004_jetti_project_file_mo', 'catalyzed', 'terminated'],
    # '004_jetti_project_file_mo3': ['15289-004 - Column Leach_v1 2020-11-30', 'tbl-RT_BAG_Control', '004_jetti_project_file_mo', 'control', 'terminated'],
    # '004_jetti_project_file_mo4': ['15289-004 - Column Leach_v1 2020-11-30', 'tbl-RT_BAG_Control', '004_jetti_project_file_mo', 'control', 'terminated'],
    # '004_jetti_project_file_mols1': ['15289-004 - Column Leach_v1 2020-11-30', '', '004_jetti_project_file_mols', 'catalyzed', 'terminated'],
    # '004_jetti_project_file_mols3': ['15289-004 - Column Leach_v1 2020-11-30', '', '004_jetti_project_file_mols', 'catalyzed', 'terminated'],
    # '004_jetti_project_file_mols4': ['15289-004 - Column Leach_v1 2020-11-30', '', '004_jetti_project_file_mols', 'control', 'terminated'],
    # '004_jetti_project_file_mols2': ['15289-004 - Column Leach_v1 2020-11-30', '', '004_jetti_project_file_mols', 'control', 'terminated'], revisar Cu Recovery y Catalyst addition, MOLS2, 3, y 4 fuera no tiene recCu ni catalyst
    '007b_jetti_project_file_tiger_tgr_1': ['007B Jetti Project File - Tiger', 'tbl-RT_1', '007b_jetti_project_file_tiger_tgr', 'control', 'ongoing'],
    '007b_jetti_project_file_tiger_tgr_2': ['007B Jetti Project File - Tiger', 'tbl-RT_1', '007b_jetti_project_file_tiger_tgr', 'control', 'ongoing'],
    # '007b_jetti_project_file_tiger_tgr_3': ['007B Jetti Project File - Tiger', 'tbl-RT_1', '007b_jetti_project_file_tiger_tgr', 'catalyzed', 'ongoing'],
    '007b_jetti_project_file_tiger_tgr_4': ['007B Jetti Project File - Tiger', 'tbl-RT_4', '007b_jetti_project_file_tiger_tgr', 'catalyzed', 'ongoing'],
    '007_jetti_project_file_leopard_lep_1': ['007 Jetti Project File - Leopard', 'tbl-RT_1', '007_jetti_project_file_leopard_lep', 'control', 'ongoing'],
    '007_jetti_project_file_leopard_lep_2': ['007 Jetti Project File - Leopard', 'tbl-RT_1', '007_jetti_project_file_leopard_lep', 'control', 'ongoing'],
    '007_jetti_project_file_leopard_lep_4': ['007 Jetti Project File - Leopard', 'tbl-RT_4', '007_jetti_project_file_leopard_lep', 'catalyzed', 'ongoing'],

}

# Convert the dictionary to a DataFrame for easier filtering
leaching_df = pd.DataFrame.from_dict(leaching_cols_to_keep, orient='index', columns=['title', 'reactor', 'sample_id', 'catalyzed_y_n', 'ongoing_y_n'])
leaching_df.to_excel(folder_path_save + '/leaching_cols_reactors.xlsx')


# Create a MultiIndex from the dictionary values
leaching_index = pd.MultiIndex.from_frame(leaching_df[['title', 'reactor']])

df_combined[df_combined.index.get_level_values(0) == '020 Jetti Project File Hypogene_Supergene']
# Filter df_combined to keep only the rows that match the MultiIndex
df_combined_filtered = df_combined.loc[df_combined.index.intersection(leaching_index)]

# df_combined_filtered = df_combined.copy()

# %% export csv

df_exponential_model.to_excel(folder_path_save + '/df_reactors_exponential_model.xlsx')
df_exponential_model.to_csv(folder_path_save + '/df_reactors_exponential_model.csv')

df_exponential_model_filtered = df_exponential_model.loc[df_exponential_model.index.intersection(leaching_index)].copy()
df_exponential_model_filtered.to_excel(folder_path_save + '/df_reactors_exponential_model_filtered.xlsx')

reactor_label_audit = build_reactor_label_audit(
    leaching_cols_to_keep,
    df_pivot.index,
    df_exponential_model.index,
    df_exponential_model_filtered.index,
)
reactor_label_audit.to_csv(folder_path_save + '/reactor_label_mapping_audit.csv', index=False)

missing_reactor_labels = reactor_label_audit[
    ~reactor_label_audit['available_in_filtered_exponential_model']
]
if not missing_reactor_labels.empty:
    print("⚠️ Missing leaching/reactor mappings after filtering:")
    print(
        missing_reactor_labels[
            [
                'project_col_id',
                'project_name',
                'start_cell',
                'same_start_cell_projects',
                'same_project_reactors',
            ]
        ].to_string(index=False)
    )

# df_exponential_model_4params.to_excel(folder_path_save + '/df_reactors_exponential_model_4params.xlsx')
# df_exponential_model_filtered_4params = df_exponential_model_4params.loc[df_exponential_model_4params.index.intersection(leaching_index)].copy()
# df_exponential_model_filtered_4params.to_excel(folder_path_save + '/df_reactors_exponential_model_filtered_4params.xlsx')

df_combined_power.to_excel(folder_path_save + '/df_reactors_slope_approach.xlsx')
df_combined_power_filtered = df_combined_power.loc[df_combined_power.index.intersection(leaching_index)].copy()
df_combined_power_filtered.to_excel(folder_path_save + '/df_reactors_slope_approach_model_filtered.xlsx')


#%% ==================================
# CREATE PLOTS FOR EVERY SAMPLE
# Define colors for catalyzed/control

colors = {
    "catalyzed": {"scatter": "blue", "line": "blue"},
    "control": {"scatter": "red", "line": "red"}
}

colors = {
    'catalyzed': {'scatter': '#8B0000', 'line': '#FF6347'},  # Dark red, Light red
    'control': {'scatter': '#006400', 'line': '#32CD32'}    # Dark green, Light green
}


# Step 1: Create a mapping from project_sample_id → [(project, reactor, treatment)]
sample_id_mapping = {}

for details in leaching_cols_to_keep.values():
    project, reactor, sample_id, treatment, _ = details
    if sample_id not in sample_id_mapping:
        sample_id_mapping[sample_id] = []
    sample_id_mapping[sample_id].append((project, reactor, treatment))

# Step 1.5: Ensure each sample_id has exactly one "control" and one "catalyzed"
for sample_id, reactor_list in sample_id_mapping.items():
    # Separate reactors by treatment type
    controls = [reactor for reactor in reactor_list if reactor[2] == "control"]
    catalyzed = [reactor for reactor in reactor_list if reactor[2] == "catalyzed"]

    # Select one reactor from each group (if multiple exist)
    selected_control = controls[0] if controls else None
    selected_catalyzed = catalyzed[0] if catalyzed else None

    # Update the mapping to include only one control and one catalyzed
    sample_id_mapping[sample_id] = []
    if selected_control:
        sample_id_mapping[sample_id].append(selected_control)
    if selected_catalyzed:
        sample_id_mapping[sample_id].append(selected_catalyzed)

# Step 2: Iterate over project_sample_id and find catalyzed-control pairs
for sample_id, reactor_list in sample_id_mapping.items():
    if len(reactor_list) != 2:
        print(f"⚠️ Skipping {sample_id}, does not have both catalyzed & control")
        continue

    # Ensure we have exactly one catalyzed and one control reactor
    (proj1, react1, treat1), (proj2, react2, treat2) = reactor_list
    if treat1 == treat2:
        print(f"⚠️ Skipping {sample_id}, does not have mixed treatment types")
        continue

    # Assign catalyzed and control
    catalyzed = (proj1, react1) if treat1 == "catalyzed" else (proj2, react2)
    control = (proj2, react2) if treat1 == "catalyzed" else (proj1, react1)
    paired_project, paired_reactor = control  # control reactor as reference

    print(f"✅ Matching {sample_id}: {catalyzed} (catalyzed) vs {control} (control)")

    # Step 3: Check existence in both DataFrames
    if catalyzed not in df_pivot.index or control not in df_pivot.index:
        print(f"⚠️ Skipping {sample_id}, missing in df_pivot")
        continue
    if catalyzed not in df_exponential_model_filtered.index or control not in df_exponential_model_filtered.index:
        print(f"⚠️ Skipping {sample_id}, missing in df_exponential_model_filtered")
        continue

    # Step 4: Convert column names to numeric for plotting
    x_scatter = pd.to_numeric(df_pivot.columns, errors='coerce')
    x_line = pd.to_numeric(df_exponential_model_filtered.columns, errors='coerce')

    # Step 5: Plot using Seaborn
    plt.figure(figsize=(8, 5))

    # Plot catalyzed sample
    sns.scatterplot(
        x=x_scatter, 
        y=df_pivot.loc[catalyzed] if catalyzed in df_pivot.index else None,
        color=colors["catalyzed"]['scatter'], 
        label=f'{catalyzed[1]} (catalyzed)'
    )
    sns.lineplot(
        x=x_line, 
        y=df_exponential_model_filtered.loc[catalyzed], 
        color=colors["catalyzed"]['line'], 
        label=f'{catalyzed[1]} Model'
    )

    # Plot control sample
    sns.scatterplot(
        x=x_scatter, 
        y=df_pivot.loc[control] if control in df_pivot.index else None, 
        color=colors["control"]['scatter'], 
        label=f'{control[1]} (control)'
    )
    sns.lineplot(
        x=x_line, 
        y=df_exponential_model_filtered.loc[control], 
        color=colors["control"]['line'], 
        label=f'{control[1]} Model'
    )

    # Labels and formatting
    plt.ylabel('Cu Recovery (%)')
    plt.xlabel('Time (days)')
    plt.axvline(plateau_forced, color='black', linestyle='--', label='Plateau forced')
    plt.ylim(0, 100)
    plt.xlim(0, 210)
    plt.legend()
    plt.title(f'Comparison: {catalyzed[0]} (Catalyzed vs Control)')
    plt.savefig(folder_path_save + f'/plots/reactors_curvefit_{sample_id}''.png')
    show_or_autoclose_plot(plt)



# %% Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_combined_filtered)


# Apply PCA
pca = PCA(n_components=2)  # Choose the number of components to retain most variance
pca_results = pca.fit_transform(df_scaled)

# Create a DataFrame for the PCA results, labeling by project_name and start_cell
df_pca = pd.DataFrame(
    data=pca_results,
    index=df_combined_filtered.index,
    columns=['reactors_PCA1', 'reactors_PCA2']  # More components can be added if needed
)

# Display results
print(df_pca.head())
print('REACTORS PCA ===========\nVariance explained per component (Eigen values) [PCA1  PCA2]:', np.round(pca.explained_variance_, 2))
print('Proportion of variance explained per component [PCA1  PCA2]:', np.round(pca.explained_variance_ratio_*100, 1), '%')
print('Proportion of total variance explained:', np.round(np.sum(pca.explained_variance_ratio_*100), 1), '%')


df_pca_for_rosetta = df_pca.reset_index(drop=False)
df_pca_for_rosetta.rename(columns={'level_0': 'project_name', 'level_1': 'start_cell'}, inplace=True)

df_pca_for_rosetta.to_excel(folder_path_save + '/pca_reactors.xlsx', index=False)

#%%

# CREATE A MODEL TO FIT THE CURVES LEAVING ONE SAMPLE OUT TO TEST THE MODEL AN GET ALL THE STATS FOR EACH CURVE
'''
# Create a DataFrame to store fit statistics
fit_stats = pd.DataFrame(index=df_combined_filtered.index, 
'''


#%% comparison 2 vs 4 parameters

# Define time vector
t = np.linspace(0, 10, 200)

# Two-parameter model: y = a*(1-exp(-b*t))
a = 1.5
b = 1.0
y_two = a * (1 - np.exp(-b * t))

# Four-parameter model: y = a*(1-exp(-b*t)) + c*(1-exp(-d*t))
# Example parameters: first term has fast dynamics, second term is slower
a1 = 1.0 # 0.7
b1 = 0.8 # 1.5
a2 = 0.5
b2 = 0.3
y_four = a1 * (1 - np.exp(-b1 * t)) + a2 * (1 - np.exp(-b2 * t))

# Plotting both curves for comparison
plt.figure(figsize=(10, 5))

# Plot the two-parameter model
plt.plot(t, y_two, label=r'Two-parameter: $y=a(1-e^{-bt})$', linewidth=2)

# Plot the four-parameter model
plt.plot(t, y_four, label=r'Four-parameter: $y=a_1(1-e^{-b_1t})+a_2(1-e^{-b_2t})$', linewidth=2)

plt.xlabel('Time (t)')
plt.ylabel('y')
plt.title('Comparison of Two- and Four-Parameter Exponential Models')
plt.legend()
plt.grid(True)
show_or_autoclose_plot(plt)

# %%

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import lightgbm as lgb

folder_path_load_originals = '/Users/administration/Library/CloudStorage/OneDrive-JettiResources/PythonProjects/SpkData/Jetti01/3_dataframes_from_csv'
# folder_path_save_reactors = '/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Rosetta/ML_reactors'
folder_path_save_reactors = '/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/csv_outputs'


df_reactors_conditions = pd.read_csv(folder_path_load_originals + '/dataset_reactors_conditions.csv')
df_reactors_summary = pd.read_csv(folder_path_load_originals + '/dataset_reactor_summary_summaries.csv')
df_reactors_detailed = pd.read_csv(folder_path_load_originals + '/dataset_reactor_summary_detailed.csv')

df_ac_summary = pd.read_csv(folder_path_load_originals + '/dataset_acid_consumption_summary_summaries.csv')
df_chemchar = pd.read_csv(folder_path_load_originals + '/dataset_characterization_summary.csv')
df_mineralogy_modals = pd.read_csv(folder_path_load_originals + '/dataset_mineralogy_summary_modals.csv')

df_qemscan_compilation = pd.read_excel('/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/df_qemscan_compilation.xlsx')
df_qemscan_compilation = df_qemscan_compilation[[
    'origin', 'file_name', 'project_name', 'sample_id', 'sample', 
    'project_sample_id_raw', 'project_sample_id', 'project_sample_condition_id', 
    'sheet_name', 'cpy_+50%_exposed_norm', 'cpy_locked_norm', 'cpy_associated_norm', 
    'cpy_+50%_exposed', 'cpy_locked', 'cpy_associated', 'copper_sulphides_lib_exposed', 
    'copper_sulphides_lib_50-80%_exposed', 'copper_sulphides_lib_30-50%_exposed', 
    'copper_sulphides_lib_20-30%_exposed', 'copper_sulphides_lib_10-20%_exposed', 
    'copper_sulphides_lib_0-10%_exposed', 'copper_sulphides_lib_locked'
    ]]


for i, x in enumerate(leaching_cols_to_keep.values()):
    print(x[2])

# Special treatment fot df_reactors_conditions because of inconsistent column names
list(sorted(df_reactors_conditions.columns))
df_reactors_conditions['2nd_wash_(ml)_al_(mg_l_%)'] = df_reactors_conditions['2nd_wash_(ml)_al(mg_l_%)'].fillna(df_reactors_conditions['2nd_wash_(ml)_al'])
df_reactors_conditions['2nd_wash_(ml)_cu_(mg_l_%)'] = df_reactors_conditions['2nd_wash_(ml)_cu(mg_l_%)'].fillna(df_reactors_conditions['2nd_wash_(ml)_cu'])
df_reactors_conditions['2nd_wash_(ml)_fe_(mg_l_%)'] = df_reactors_conditions['2nd_wash_(ml)_fe_(mg_l_%)'].fillna(df_reactors_conditions['2nd_wash_(ml)_fe'])
df_reactors_conditions['2nd_wash_(ml)_mg_(mg_l_%)'] = df_reactors_conditions['2nd_wash_(ml)_mg(mg_l_%)'].fillna(df_reactors_conditions['2nd_wash_(ml)_mg'])
df_reactors_conditions['2nd_wash_(ml)_si_(mg_l_%)'] = df_reactors_conditions['2nd_wash_(ml)_si(mg_l_%)'].fillna(df_reactors_conditions['2nd_wash_(ml)_si'])
df_reactors_conditions['assayed_head_(g)_al_(mg_l_%)'] = df_reactors_conditions['assayed_head_(g)_al_(mg_l_%)'].fillna(df_reactors_conditions['assayed_head_(g)_al(mg_l_%)']).fillna(df_reactors_conditions['assayed_head_(g)_al'])
df_reactors_conditions['assayed_head_(g)_cu_(mg_l_%)'] = df_reactors_conditions['assayed_head_(g)_cu_(mg_l_%)'].fillna(df_reactors_conditions['assayed_head_(g)_cu(mg_l_%)']).fillna(df_reactors_conditions['assayed_head_(g)_cu'])
df_reactors_conditions['assayed_head_(g)_fe_(mg_l_%)'] = df_reactors_conditions['assayed_head_(g)_fe_(mg_l_%)'].fillna(df_reactors_conditions['assayed_head_(g)_fe'])
# finish this with all the columns...



#%%
df_reactors_conditions.to_csv(folder_path_save_reactors + '/dataset_reactors_conditions.csv', index=False)
df_reactors_summary.to_csv(folder_path_save_reactors + '/dataset_reactor_summary_summaries.csv', index=False)
df_reactors_detailed.to_csv(folder_path_save_reactors + '/dataset_reactor_summary_detailed.csv', index=False)
df_ac_summary.to_csv(folder_path_save_reactors + '/dataset_acid_consumption_summary_summaries.csv', index=False)
df_chemchar.to_csv(folder_path_save_reactors + '/dataset_characterization_summary.csv', index=False)
df_mineralogy_modals.to_csv(folder_path_save_reactors + '/dataset_mineralogy_summary_modals.csv', index=False)
df_qemscan_compilation.to_csv(folder_path_load_originals + '/df_qemscan_compilation.csv', index=False)

df_ac_summary[df_ac_summary['project_name'] == '030 Jetti Project File']


#%%


import pandas as pd
import os
import re
from collections import defaultdict
import unicodedata

# ============================================================================
# CONFIGURATION - Modify these variables as needed
# ============================================================================

# Input and output directories
INPUT_DIR = "/Users/administration/Library/CloudStorage/OneDrive-JettiResources/PythonProjects/SpkData/Jetti01/3_dataframes_from_csv"  # Directory containing the original CSV files
OUTPUT_DIR = "/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/csv_outputs"  # Directory where processed files will be saved

# Input filenames - list of CSV files to process
INPUT_FILES = [
    "dataset_acid_consumption_summary_summaries.csv",
    "dataset_characterization_summary.csv",
    "dataset_mineralogy_summary_modals.csv",
    "dataset_reactor_summary_detailed.csv",
    "dataset_reactor_summary_summaries.csv",
    "dataset_reactors_conditions.csv",
    # "df_qemscan_compilation.csv"
]

# Output filename for the merged dataset
MERGED_DATASET_FILENAME = "merged_dataset_reactors.csv"

# Output filename for the duplicate IDs report
DUPLICATE_IDS_REPORT_FILENAME = "duplicate_ids_report.txt"

# Output filename for the summary report
SUMMARY_REPORT_FILENAME = "summary_report_reactors.md"

# Suffix to add to output filenames
OUTPUT_SUFFIX = "_with_id"

# Debug / audit controls (can be memory heavy on large datasets)
CAPTURE_INTERMEDIATES = True
CAPTURE_DROPPED_ROWS = False
CAPTURE_DUPLICATE_ROWS = False
CAPTURE_MERGE_COLLISIONS = False
SAVE_INTERMEDIATE_FRAMES = True
INTERMEDIATE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "intermediate_frames")
CAPTURE_INTERMEDIATES_IN_MEMORY = CAPTURE_INTERMEDIATES
# In-memory copies for line-by-line inspection; can be large.
INTERMEDIATE_STEPS = {}

# ============================================================================
# FUNCTIONS
# ============================================================================

def _save_intermediate_dataframe(df, filename, output_dir=INTERMEDIATE_OUTPUT_DIR):
    if not SAVE_INTERMEDIATE_FRAMES:
        return
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)

def _save_intermediate_dataframes(all_dataframes, stage, output_dir=INTERMEDIATE_OUTPUT_DIR):
    if not SAVE_INTERMEDIATE_FRAMES:
        return
    os.makedirs(output_dir, exist_ok=True)
    for filename, df in all_dataframes.items():
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}_{stage}.csv")
        df.to_csv(output_path, index=False)

def create_project_sample_id(row, columns_to_use):
    """
    Create a project_sample_id from specified columns in a row.
    
    Args:
        row: DataFrame row
        columns_to_use: List of column names to use for ID creation
        
    Returns:
        String containing the project_sample_id
    """
    components = []
    
    # Extract base sample name from analyte_units if it contains "Sample X" pattern
    sample_name = None
    if 'analyte_units' in columns_to_use and pd.notna(row.get('analyte_units')):
        match = re.match(r'(Sample\s+\w+)', str(row['analyte_units']))
        if match:
            sample_name = match.group(1).strip()
    
    # Add components in order of importance
    for col in columns_to_use:
        if col in row and pd.notna(row[col]):
            # Skip if we already extracted this sample name from analyte_units
            if col == 'analyte_units' and sample_name:
                continue
            """    
            # Clean the value: replace spaces with underscores, remove special chars except +-%
            value = str(row[col])
            value = re.sub(r'[^\w+\-%]', '_', value)  # Replace special chars with underscore
            value = re.sub(r'_+', '_', value)  # Replace multiple underscores with single
            value = value.strip('_')  # Remove leading/trailing underscores
            """

            value = str(row[col])

            value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')  # Remove accents
            value = value.lower()  # Convert to lowercase
            value = re.sub(r'[^a-z0-9%+]', '', value)  # Remove unwanted special characters

            if value and value not in components:
                components.append(value)
    
    # If we extracted a sample name from analyte_units, use it instead of the full value
    if sample_name:
        """
        sample_name = re.sub(r'[^\w+\-%]', '_', sample_name)
        sample_name = re.sub(r'_+', '_', sample_name)
        sample_name = sample_name.strip('_')
        """
        sample_name = unicodedata.normalize('NFKD', sample_name).encode('ascii', 'ignore').decode('ascii')
        sample_name = sample_name.lower()
        sample_name = re.sub(r'[^a-z0-9%+]', '', sample_name)
        if sample_name and sample_name not in components:
            components.append(sample_name)
    
    # Join components with underscores
    project_sample_id = '_'.join(components)
    
    # Ensure the ID is not empty
    if not project_sample_id:
        project_sample_id = "unknown_sample"
        
    return project_sample_id



def process_datasets(
    capture_intermediates=False,
    capture_dropped_rows=False,
    capture_duplicate_rows=False,
    capture_intermediates_in_memory=CAPTURE_INTERMEDIATES_IN_MEMORY,
):
    """
    Process all datasets to add project_sample_id column.
    
    Returns:
        Tuple of (processed dataframes, duplicate_ids, audit_log)
    """
    # Define which columns to use for each dataset
    id_creation_logic = {
        'dataset_acid_consumption_summary_summaries.csv': ['project_name', 'sheet_name', 'ore_type'], # 'test_id'
        'dataset_characterization_summary.csv': ['project_name', 'analyte_units'], # 'sheet_name'
        'dataset_mineralogy_summary_modals.csv': ['project_name', 'sample', 'start_cell', 'index'], # 'sheet_name'
        'dataset_reactor_summary_detailed.csv': ['project_name', 'start_cell'], # 'sheet_name'
        'dataset_reactor_summary_summaries.csv': ['project_name', 'ore_type', 'test_id'], # 'sheet_name'
        'dataset_reactors_conditions.csv': ['project_name', 'catalyzed'], # 'sheet_name', 'test_id', 'ore_type', 'start_cell'
        # 'df_qemscan_compilation.csv': ['project_name', 'sample_id', 'sheet_name']  # This file is not processed in this function
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each dataset
    print("Processing datasets and creating project_sample_id column...")
    all_dataframes = {}
    id_counts = defaultdict(int)  # Track ID occurrences across all datasets
    duplicate_ids = {}  # Track which IDs needed suffixes
    audit_log = {"datasets": {}, "cross_dataset_duplicates": {}}

    if capture_intermediates_in_memory:
        global INTERMEDIATE_STEPS
        INTERMEDIATE_STEPS = {}
    
    for filename in INPUT_FILES:
        if filename not in id_creation_logic:
            print(f"  Warning: No ID creation logic defined for {filename}, skipping.")
            continue
            
        columns_to_use = id_creation_logic[filename]
        print(f"\nProcessing {filename}...")
        
        # Read the dataset
        file_path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(file_path):
            print(f"  Warning: File {file_path} not found, skipping.")
            continue
            
        base_name = os.path.splitext(filename)[0]
        df = pd.read_csv(file_path)
        if capture_intermediates_in_memory:
            dataset_steps = INTERMEDIATE_STEPS.setdefault(filename, {})
            dataset_steps["raw"] = df.copy()
        if capture_intermediates:
            _save_intermediate_dataframe(df, f"{base_name}_raw.csv")
        dataset_audit = None
        if capture_intermediates:
            dataset_audit = {
                "columns_used_for_id": columns_to_use,
                "rows_before": len(df),
            }
        
        # Create project_sample_id column
        if filename != 'dataset_mineralogy_summary_modals.csv':
            missing_mask = df[columns_to_use].isna().any(axis=1)
            if capture_intermediates_in_memory:
                dataset_steps["missing_mask"] = missing_mask.copy()
            if capture_intermediates:
                dropped_count = int(missing_mask.sum())
                dataset_audit["dropped_rows_count"] = dropped_count
                dataset_audit["rows_after_dropna"] = len(df) - dropped_count
                if capture_dropped_rows and dropped_count > 0:
                    dataset_audit["dropped_rows"] = df.loc[missing_mask].copy()
            df = df.loc[~missing_mask].copy()
            if capture_intermediates_in_memory:
                dataset_steps["after_dropna"] = df.copy()
            if capture_intermediates:
                _save_intermediate_dataframe(df, f"{base_name}_after_dropna.csv")
        else:
            if capture_intermediates:
                dataset_audit["dropped_rows_count"] = 0
                dataset_audit["rows_after_dropna"] = len(df)
                _save_intermediate_dataframe(df, f"{base_name}_after_dropna.csv")
            if capture_intermediates_in_memory:
                dataset_steps["after_dropna"] = df.copy()
        df['project_sample_id_raw'] = df.apply(lambda row: create_project_sample_id(row, columns_to_use), axis=1)
        if capture_intermediates_in_memory:
            dataset_steps["after_id_creation"] = df.copy()
        if capture_intermediates:
            _save_intermediate_dataframe(df, f"{base_name}_after_id_creation.csv")
        
        # Check for and handle duplicate IDs within this dataset
        id_counts_this_df = df['project_sample_id_raw'].value_counts()
        duplicates_in_df = id_counts_this_df[id_counts_this_df > 1]
        '''
        if duplicates_in_df:
            print(f"  Found {len(duplicates_in_df)} duplicate IDs in this dataset.")
            for dup_id in duplicates_in_df:
                duplicate_ids[dup_id] = True
                # Add suffix to duplicates
                mask = df['project_sample_id_raw'] == dup_id
                dup_indices = df.index[mask].tolist()
                for i, idx in enumerate(dup_indices):
                    if i > 0:  # Keep the first occurrence as is
                        df.at[idx, 'project_sample_id_raw'] = f"{dup_id}_dup{i}"
        '''
        if capture_intermediates:
            dataset_audit["rows_after_id_creation"] = len(df)
            dataset_audit["duplicate_id_counts"] = duplicates_in_df.to_dict()
            if capture_duplicate_rows and not duplicates_in_df.empty:
                dataset_audit["duplicate_rows"] = df[df['project_sample_id_raw'].isin(duplicates_in_df.index)].copy()
            audit_log["datasets"][filename] = dataset_audit
        if capture_intermediates_in_memory:
            dataset_steps["duplicate_id_counts"] = duplicates_in_df.copy()

        # Store the processed dataframe
        all_dataframes[filename] = df
        
        # Update global ID counts for cross-dataset uniqueness check
        for id_val in df['project_sample_id_raw'].unique():
            id_counts[id_val] += 1
        
        print(f"  Added project_sample_id column to {filename}")
        print(f"  Sample IDs: {df['project_sample_id_raw'].head(3).tolist()}")
    
    # Check for duplicate IDs across datasets
    cross_dataset_duplicates = {id_val: count for id_val, count in id_counts.items() if count > 1}
    if cross_dataset_duplicates:
        print(f"\nFound {len(cross_dataset_duplicates)} IDs that appear in multiple datasets:")
        for dup_id, count in list(cross_dataset_duplicates.items())[:5]:  # Show first 5 examples
            print(f"  {dup_id}: appears in {count} datasets")
        
        # Handle cross-dataset duplicates by adding dataset prefix
        for filename, df in all_dataframes.items():
            dataset_prefix = filename.split('_')[1][:3]  # Use first 3 chars of the second part of filename
            for dup_id in cross_dataset_duplicates:
                mask = df['project_sample_id_raw'] == dup_id
                if mask.any():
                    # df.loc[mask, 'project_sample_id_raw'] = f"{dataset_prefix}_{df.loc[mask, 'project_sample_id_raw']}"
                    df.loc[mask, 'project_sample_id_raw'] = dataset_prefix + '_' + df.loc[mask, 'project_sample_id_raw'].astype(str)
        if capture_intermediates_in_memory:
            for filename, df in all_dataframes.items():
                dataset_steps = INTERMEDIATE_STEPS.setdefault(filename, {})
                dataset_steps["after_cross_dataset_prefix"] = df.copy()
        if capture_intermediates:
            _save_intermediate_dataframes(all_dataframes, "after_cross_dataset_prefix")
    
    if capture_intermediates:
        audit_log["cross_dataset_duplicates"] = cross_dataset_duplicates
        _save_intermediate_dataframes(all_dataframes, "after_process_datasets")
    return all_dataframes, duplicate_ids, audit_log

def save_processed_datasets(all_dataframes, duplicate_ids, capture_merge_collisions=False):
    """
    Save processed datasets and create reports.
    
    Args:
        all_dataframes: Dictionary of processed dataframes
        duplicate_ids: Dictionary of duplicate IDs
    """
    # Save the processed datasets
    print("\nSaving processed datasets...")
    for filename, df in all_dataframes.items():
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}{OUTPUT_SUFFIX}.csv")
        df['project_sample_id_raw'] = df['project_sample_id_raw'].astype(str, errors='ignore').str.replace(' ', '_')  # Replace spaces with underscores
        df.to_csv(output_path, index=False)
        print(f"  Saved {output_path}")
    
    # Create a report of duplicate IDs
    if duplicate_ids:
        duplicate_report_path = os.path.join(OUTPUT_DIR, DUPLICATE_IDS_REPORT_FILENAME)
        with open(duplicate_report_path, "w") as f:
            f.write("Duplicate IDs found and handled with suffixes:\n")
            for dup_id in duplicate_ids:
                f.write(f"- {dup_id}\n")
        print(f"  Saved duplicate IDs report to {duplicate_report_path}")
    
    # Create a merged dataframe
    print("\nCreating merged dataframe...")
    # First, identify common columns across all datasets
    all_columns = set()
    for df in all_dataframes.values():
        all_columns.update(df.columns)
    
    # Create a dictionary to store the merged data
    merged_data = {}
    merge_collisions = defaultdict(list)
    for filename, df in all_dataframes.items():
        dataset_name = os.path.splitext(filename)[0].replace('dataset_', '')
        for row_idx, row in df.iterrows():
            sample_id = row['project_sample_id_raw']
            if sample_id not in merged_data:
                merged_data[sample_id] = {'project_sample_id_raw': sample_id}
            elif capture_merge_collisions:
                merge_collisions[sample_id].append({"filename": filename, "row_index": row_idx})
            
            # Add dataset-specific prefix to columns to avoid conflicts
            for col in df.columns:
                if col != 'project_sample_id_raw':  # Don't prefix the ID column
                    merged_data[sample_id][f"{dataset_name}_{col}"] = row[col]
    
    # Convert to dataframe
    merged_df = pd.DataFrame.from_dict(merged_data, orient='index')
    merged_output_path = os.path.join(OUTPUT_DIR, MERGED_DATASET_FILENAME)
    merged_df.to_csv(merged_output_path, index=False)
    print(f"  Saved merged dataset to {merged_output_path}")
    
    return merged_df, merge_collisions

def create_summary_report(all_dataframes, duplicate_ids):
    """
    Create a summary report of the processed datasets.
    
    Args:
        all_dataframes: Dictionary of processed dataframes
        duplicate_ids: Dictionary of duplicate IDs
    """
    print("\nCreating summary report...")
    
    # Get file info for each processed dataset
    file_info = []
    for filename, df in all_dataframes.items():
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}{OUTPUT_SUFFIX}.csv")
        
        if os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            rows = len(df)
            cols = len(df.columns)
            
            # Get sample of project_sample_id values
            sample_ids = []
            if 'project_sample_id_raw' in df.columns:
                sample_ids = df['project_sample_id_raw'].head(3).tolist()
            
            file_info.append({
                'filename': f"{base_name}{OUTPUT_SUFFIX}.csv",
                'size_kb': size_kb,
                'rows': rows,
                'columns': cols,
                'sample_ids': sample_ids
            })
    
    # Add merged dataset info
    merged_output_path = os.path.join(OUTPUT_DIR, MERGED_DATASET_FILENAME)
    if os.path.exists(merged_output_path):
        merged_df = pd.read_csv(merged_output_path)
        size_kb = os.path.getsize(merged_output_path) / 1024
        file_info.append({
            'filename': MERGED_DATASET_FILENAME,
            'size_kb': size_kb,
            'rows': len(merged_df),
            'columns': len(merged_df.columns),
            'sample_ids': merged_df['project_sample_id_raw'].head(3).tolist() if 'project_sample_id_raw' in merged_df.columns else []
        })
    
    # Create the summary report
    report_path = os.path.join(OUTPUT_DIR, SUMMARY_REPORT_FILENAME)
    with open(report_path, "w") as f:
        f.write("# Dataset Processing Summary Report\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report summarizes the processing of multiple datasets to add a 'project_sample_id_raw' column for relationship mapping.\n\n")
        
        f.write("## Implementation Details\n\n")
        f.write("The 'project_sample_id_raw' column was created using the following approach:\n\n")
        f.write("1. For each dataset, relevant columns were selected based on data availability:\n")
        f.write("   - dataset_acid_consumption_summary_summaries: project_name, sheet_name, test_id, ore_type\n")
        f.write("   - dataset_characterization_summary: project_name, sheet_name, analyte_units\n")
        f.write("   - dataset_mineralogy_summary_modals: project_name, sheet_name, sample\n")
        f.write("   - dataset_reactor_summary_detailed: project_name, sheet_name, catalyzed, test_id, ore_type\n")
        f.write("   - dataset_reactor_summary_summaries: project_name, sheet_name, catalyzed, test_id, ore_type\n")
        f.write("   - dataset_reactors_conditions: project_name, sheet_name, catalyzed, start_cell, test_id, ore_type\n\n")
        
        f.write("2. Special handling for 'Sample Fresca' type entries:\n")
        f.write("   - When analyte_units contained values like 'Sample Fresca', 'Sample Fresca +3/4\"', etc.\n")
        f.write("   - These were treated as a single sample with the base name extracted\n\n")
        
        f.write("3. ID formatting:\n")
        f.write("   - Underscores used as separators\n")
        f.write("   - Special characters removed (except +, -, % for numbers)\n")
        f.write("   - Multiple underscores consolidated\n\n")
        
        f.write("4. Duplicate handling:\n")
        f.write("   - Within-dataset duplicates: suffix added (e.g., '_dup1', '_dup2')\n")
        f.write("   - Cross-dataset uniqueness verified\n\n")
        
        f.write("## Processed Files\n\n")
        f.write("| Filename | Size (KB) | Rows | Columns | Sample IDs |\n")
        f.write("|----------|-----------|------|---------|------------|\n")
        
        for info in file_info:
            sample_id_str = ", ".join([f"`{id}`" for id in info['sample_ids'][:2]])
            if len(info['sample_ids']) > 2:
                sample_id_str += ", ..."
            
            f.write(f"| {info['filename']} | {info['size_kb']:.1f} | {info['rows']} | {info['columns']} | {sample_id_str} |\n")
        
        f.write("\n## Duplicate IDs\n\n")
        duplicate_report_path = os.path.join(OUTPUT_DIR, DUPLICATE_IDS_REPORT_FILENAME)
        if os.path.exists(duplicate_report_path):
            with open(duplicate_report_path, "r") as dup_file:
                f.write("```\n")
                f.write(dup_file.read())
                f.write("```\n\n")
        else:
            f.write("No duplicate IDs report found.\n\n")
        
        f.write("## Merged Dataset\n\n")
        f.write("A merged dataset was created combining data from all individual datasets, using the project_sample_id as the key.\n")
        f.write("This allows for easy access to all related data for a given sample across different datasets.\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("The processed datasets can now be used for relationship mapping and analysis. The project_sample_id column\n")
        f.write("provides a consistent way to identify and link samples across different datasets.\n")
    
    print(f"  Summary report created: {report_path}")

def validate_uniqueness(all_dataframes):
    """
    Validate the uniqueness of project_sample_id values.
    
    Args:
        all_dataframes: Dictionary of processed dataframes
    """
    print("\nValidating uniqueness of project_sample_id values...")
    
    # Check uniqueness within each dataset
    print("\nChecking uniqueness within each dataset:")
    for filename, df in all_dataframes.items():
        duplicate_count = df.duplicated(subset=['project_sample_id_raw']).sum()
        if duplicate_count > 0:
            print(f"  WARNING: {filename} still has {duplicate_count} duplicate project_sample_id values!")
        else:
            print(f"  {filename}: All project_sample_id values are unique within the dataset ✓")
    
    # Check for cross-dataset duplicates
    print("\nChecking for cross-dataset duplicates:")
    all_ids = defaultdict(list)
    for filename, df in all_dataframes.items():
        for id_val in df['project_sample_id_raw'].unique():
            all_ids[id_val].append(filename)
    
    cross_duplicates = {id_val: files for id_val, files in all_ids.items() if len(files) > 1}
    if cross_duplicates:
        print(f"  Found {len(cross_duplicates)} IDs that appear in multiple datasets:")
        for id_val, files in list(cross_duplicates.items())[:10]:  # Show first 10 examples
            print(f"  - '{id_val}' appears in: {', '.join(files)}")
    else:
        print("  All project_sample_id values are unique across all datasets ✓")
    
    # Verify merged dataset
    print("\nVerifying merged dataset:")
    merged_output_path = os.path.join(OUTPUT_DIR, MERGED_DATASET_FILENAME)
    if os.path.exists(merged_output_path):
        merged_df = pd.read_csv(merged_output_path)
        print(f"  Merged dataset has {len(merged_df)} rows")
        print(f"  Unique project_sample_id count: {merged_df['project_sample_id_raw'].nunique()}")
        if len(merged_df) != merged_df['project_sample_id_raw'].nunique():
            print("  WARNING: Merged dataset has duplicate project_sample_id values!")
        else:
            print("  All project_sample_id values in merged dataset are unique ✓")
    else:
        print(f"  Merged dataset file not found: {merged_output_path}")

# ============================================================================
# RUNNER (call manually from your IDE)
# ============================================================================

def run_project_sample_id_processing(
    capture_intermediates=CAPTURE_INTERMEDIATES,
    capture_dropped_rows=CAPTURE_DROPPED_ROWS,
    capture_duplicate_rows=CAPTURE_DUPLICATE_ROWS,
    capture_merge_collisions=CAPTURE_MERGE_COLLISIONS,
):
    """
    Run the project_sample_id processing workflow and return debug artifacts.
    """
    print("Starting process to add project_sample_id to datasets")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files to process: {', '.join(INPUT_FILES)}")

    # Process datasets
    all_dataframes, duplicate_ids, audit_log = process_datasets(
        capture_intermediates=capture_intermediates,
        capture_dropped_rows=capture_dropped_rows,
        capture_duplicate_rows=capture_duplicate_rows,
    )

    # Save processed datasets and create merged dataset
    merged_df, merge_collisions = save_processed_datasets(
        all_dataframes,
        duplicate_ids,
        capture_merge_collisions=capture_merge_collisions,
    )
    if capture_intermediates:
        _save_intermediate_dataframes(all_dataframes, "after_save_processed_datasets")

    # Validate uniqueness
    validate_uniqueness(all_dataframes)

    # Create summary report
    create_summary_report(all_dataframes, duplicate_ids)

    print("\nProcessing complete!")

    return {
        "all_dataframes": all_dataframes,
        "duplicate_ids": duplicate_ids,
        "audit_log": audit_log,
        "merged_df": merged_df,
        "merge_collisions": merge_collisions,
        "intermediate_steps": INTERMEDIATE_STEPS,
    }


# Example:
# processing_results = run_project_sample_id_processing()
# audit_log = processing_results["audit_log"]


#%%
# STEP-BY-STEP (line-by-line): project_sample_id processing
# Use this block to run manually without calling the functions above.
step_capture_intermediates = CAPTURE_INTERMEDIATES
step_capture_dropped_rows = CAPTURE_DROPPED_ROWS
step_capture_duplicate_rows = CAPTURE_DUPLICATE_ROWS
step_capture_merge_collisions = CAPTURE_MERGE_COLLISIONS

print("Starting process to add project_sample_id to datasets")
print(f"Input directory: {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Files to process: {', '.join(INPUT_FILES)}")

# Define which columns to use for each dataset
id_creation_logic = {
    'dataset_acid_consumption_summary_summaries.csv': ['project_name', 'sheet_name', 'ore_type'],  # 'test_id'
    'dataset_characterization_summary.csv': ['project_name', 'analyte_units'],  # 'sheet_name'
    'dataset_mineralogy_summary_modals.csv': ['project_name', 'sample', 'start_cell', 'index'],  # 'sheet_name'
    'dataset_reactor_summary_detailed.csv': ['project_name', 'start_cell'],  # 'sheet_name'
    'dataset_reactor_summary_summaries.csv': ['project_name', 'ore_type', 'test_id'],  # 'sheet_name'
    'dataset_reactors_conditions.csv': ['project_name', 'catalyzed'],  # 'sheet_name', 'test_id', 'ore_type', 'start_cell'
    # 'df_qemscan_compilation.csv': ['project_name', 'sample_id', 'sheet_name']
}

# Process each dataset
print("Processing datasets and creating project_sample_id column...")
all_dataframes = {}
id_counts = defaultdict(int)
duplicate_ids = {}
audit_log = {"datasets": {}, "cross_dataset_duplicates": {}}

for filename in INPUT_FILES:
    if filename not in id_creation_logic:
        print(f"  Warning: No ID creation logic defined for {filename}, skipping.")
        continue

    columns_to_use = id_creation_logic[filename]
    print(f"\nProcessing {filename}...")

    file_path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(file_path):
        print(f"  Warning: File {file_path} not found, skipping.")
        continue

    base_name = os.path.splitext(filename)[0]
    df = pd.read_csv(file_path)
    if step_capture_intermediates:
        _save_intermediate_dataframe(df, f"{base_name}_raw.csv")

    dataset_audit = None
    if step_capture_intermediates:
        dataset_audit = {
            "columns_used_for_id": columns_to_use,
            "rows_before": len(df),
        }

    if filename != 'dataset_mineralogy_summary_modals.csv':
        missing_mask = df[columns_to_use].isna().any(axis=1)
        if step_capture_intermediates:
            dropped_count = int(missing_mask.sum())
            dataset_audit["dropped_rows_count"] = dropped_count
            dataset_audit["rows_after_dropna"] = len(df) - dropped_count
            if step_capture_dropped_rows and dropped_count > 0:
                dataset_audit["dropped_rows"] = df.loc[missing_mask].copy()
        df = df.loc[~missing_mask].copy()
        if step_capture_intermediates:
            _save_intermediate_dataframe(df, f"{base_name}_after_dropna.csv")
    else:
        if step_capture_intermediates:
            dataset_audit["dropped_rows_count"] = 0
            dataset_audit["rows_after_dropna"] = len(df)
            _save_intermediate_dataframe(df, f"{base_name}_after_dropna.csv")

    df['project_sample_id_raw'] = df.apply(lambda row: create_project_sample_id(row, columns_to_use), axis=1)
    if step_capture_intermediates:
        _save_intermediate_dataframe(df, f"{base_name}_after_id_creation.csv")

    id_counts_this_df = df['project_sample_id_raw'].value_counts()
    duplicates_in_df = id_counts_this_df[id_counts_this_df > 1]
    if step_capture_intermediates:
        dataset_audit["rows_after_id_creation"] = len(df)
        dataset_audit["duplicate_id_counts"] = duplicates_in_df.to_dict()
        if step_capture_duplicate_rows and not duplicates_in_df.empty:
            dataset_audit["duplicate_rows"] = df[df['project_sample_id_raw'].isin(duplicates_in_df.index)].copy()
        audit_log["datasets"][filename] = dataset_audit

    all_dataframes[filename] = df
    for id_val in df['project_sample_id_raw'].unique():
        id_counts[id_val] += 1

    print(f"  Added project_sample_id column to {filename}")
    print(f"  Sample IDs: {df['project_sample_id_raw'].head(3).tolist()}")

# Check for duplicate IDs across datasets
cross_dataset_duplicates = {id_val: count for id_val, count in id_counts.items() if count > 1}
if cross_dataset_duplicates:
    print(f"\nFound {len(cross_dataset_duplicates)} IDs that appear in multiple datasets:")
    for dup_id, count in list(cross_dataset_duplicates.items())[:5]:
        print(f"  {dup_id}: appears in {count} datasets")

    for filename, df in all_dataframes.items():
        dataset_prefix = filename.split('_')[1][:3]
        for dup_id in cross_dataset_duplicates:
            mask = df['project_sample_id_raw'] == dup_id
            if mask.any():
                # df.loc[mask, 'project_sample_id_raw'] = f"{dataset_prefix}_{df.loc[mask, 'project_sample_id_raw']}"
                df.loc[mask, 'project_sample_id_raw'] = dataset_prefix + '_' + df.loc[mask, 'project_sample_id_raw'].astype(str)
    if step_capture_intermediates:
        _save_intermediate_dataframes(all_dataframes, "after_cross_dataset_prefix")

if step_capture_intermediates:
    audit_log["cross_dataset_duplicates"] = cross_dataset_duplicates
    _save_intermediate_dataframes(all_dataframes, "after_process_datasets")

# Save processed datasets
print("\nSaving processed datasets...")
for filename, df in all_dataframes.items():
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}{OUTPUT_SUFFIX}.csv")
    df['project_sample_id_raw'] = df['project_sample_id_raw'].astype(str, errors='ignore').str.replace(' ', '_')
    df.to_csv(output_path, index=False)
    print(f"  Saved {output_path}")

if step_capture_intermediates:
    _save_intermediate_dataframes(all_dataframes, "after_save_processed_datasets")

# Create a report of duplicate IDs
if duplicate_ids:
    duplicate_report_path = os.path.join(OUTPUT_DIR, DUPLICATE_IDS_REPORT_FILENAME)
    with open(duplicate_report_path, "w") as f:
        f.write("Duplicate IDs found and handled with suffixes:\n")
        for dup_id in duplicate_ids:
            f.write(f"- {dup_id}\n")
    print(f"  Saved duplicate IDs report to {duplicate_report_path}")

# Create a merged dataframe
print("\nCreating merged dataframe...")
all_columns = set()
for df in all_dataframes.values():
    all_columns.update(df.columns)

merged_data = {}
merge_collisions = defaultdict(list)
for filename, df in all_dataframes.items():
    dataset_name = os.path.splitext(filename)[0].replace('dataset_', '')
    for row_idx, row in df.iterrows():
        sample_id = row['project_sample_id_raw']
        if sample_id not in merged_data:
            merged_data[sample_id] = {'project_sample_id_raw': sample_id}
        elif step_capture_merge_collisions:
            merge_collisions[sample_id].append({"filename": filename, "row_index": row_idx})

        for col in df.columns:
            if col != 'project_sample_id_raw':
                merged_data[sample_id][f"{dataset_name}_{col}"] = row[col]

merged_df = pd.DataFrame.from_dict(merged_data, orient='index')
merged_output_path = os.path.join(OUTPUT_DIR, MERGED_DATASET_FILENAME)
merged_df.to_csv(merged_output_path, index=False)
print(f"  Saved merged dataset to {merged_output_path}")

# Validate uniqueness
print("\nValidating uniqueness of project_sample_id values...")

print("\nChecking uniqueness within each dataset:")
for filename, df in all_dataframes.items():
    duplicate_count = df.duplicated(subset=['project_sample_id_raw']).sum()
    if duplicate_count > 0:
        print(f"  WARNING: {filename} still has {duplicate_count} duplicate project_sample_id values!")
    else:
        print(f"  {filename}: All project_sample_id values are unique within the dataset ✓")

print("\nChecking for cross-dataset duplicates:")
all_ids = defaultdict(list)
for filename, df in all_dataframes.items():
    for id_val in df['project_sample_id_raw'].unique():
        all_ids[id_val].append(filename)

cross_duplicates = {id_val: files for id_val, files in all_ids.items() if len(files) > 1}
if cross_duplicates:
    print(f"  Found {len(cross_duplicates)} IDs that appear in multiple datasets:")
    for id_val, files in list(cross_duplicates.items())[:10]:
        print(f"  - '{id_val}' appears in: {', '.join(files)}")
else:
    print("  All project_sample_id values are unique across all datasets ✓")

print("\nVerifying merged dataset:")
if os.path.exists(merged_output_path):
    merged_df_check = pd.read_csv(merged_output_path)
    print(f"  Merged dataset has {len(merged_df_check)} rows")
    print(f"  Unique project_sample_id count: {merged_df_check['project_sample_id_raw'].nunique()}")
    if len(merged_df_check) != merged_df_check['project_sample_id_raw'].nunique():
        print("  WARNING: Merged dataset has duplicate project_sample_id values!")
    else:
        print("  All project_sample_id values in merged dataset are unique ✓")
else:
    print(f"  Merged dataset file not found: {merged_output_path}")

# Create summary report
print("\nCreating summary report...")
file_info = []
for filename, df in all_dataframes.items():
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}{OUTPUT_SUFFIX}.csv")

    if os.path.exists(output_path):
        size_kb = os.path.getsize(output_path) / 1024
        rows = len(df)
        cols = len(df.columns)

        sample_ids = []
        if 'project_sample_id_raw' in df.columns:
            sample_ids = df['project_sample_id_raw'].head(3).tolist()

        file_info.append({
            'filename': f"{base_name}{OUTPUT_SUFFIX}.csv",
            'size_kb': size_kb,
            'rows': rows,
            'columns': cols,
            'sample_ids': sample_ids
        })

if os.path.exists(merged_output_path):
    merged_df_info = pd.read_csv(merged_output_path)
    size_kb = os.path.getsize(merged_output_path) / 1024
    file_info.append({
        'filename': MERGED_DATASET_FILENAME,
        'size_kb': size_kb,
        'rows': len(merged_df_info),
        'columns': len(merged_df_info.columns),
        'sample_ids': merged_df_info['project_sample_id_raw'].head(3).tolist() if 'project_sample_id_raw' in merged_df_info.columns else []
    })

report_path = os.path.join(OUTPUT_DIR, SUMMARY_REPORT_FILENAME)
with open(report_path, "w") as f:
    f.write("# Dataset Processing Summary Report\n\n")

    f.write("## Overview\n\n")
    f.write("This report summarizes the processing of multiple datasets to add a 'project_sample_id_raw' column for relationship mapping.\n\n")

    f.write("## Implementation Details\n\n")
    f.write("The 'project_sample_id_raw' column was created using the following approach:\n\n")
    f.write("1. For each dataset, relevant columns were selected based on data availability:\n")
    f.write("   - dataset_acid_consumption_summary_summaries: project_name, sheet_name, test_id, ore_type\n")
    f.write("   - dataset_characterization_summary: project_name, sheet_name, analyte_units\n")
    f.write("   - dataset_mineralogy_summary_modals: project_name, sheet_name, sample\n")
    f.write("   - dataset_reactor_summary_detailed: project_name, sheet_name, catalyzed, test_id, ore_type\n")
    f.write("   - dataset_reactor_summary_summaries: project_name, sheet_name, catalyzed, test_id, ore_type\n")
    f.write("   - dataset_reactors_conditions: project_name, sheet_name, catalyzed, start_cell, test_id, ore_type\n\n")

    f.write("2. Special handling for 'Sample Fresca' type entries:\n")
    f.write("   - When analyte_units contained values like 'Sample Fresca', 'Sample Fresca +3/4\"', etc.\n")
    f.write("   - These were treated as a single sample with the base name extracted\n\n")

    f.write("3. ID formatting:\n")
    f.write("   - Underscores used as separators\n")
    f.write("   - Special characters removed (except +, -, % for numbers)\n")
    f.write("   - Multiple underscores consolidated\n\n")

    f.write("4. Duplicate handling:\n")
    f.write("   - Within-dataset duplicates: suffix added (e.g., '_dup1', '_dup2')\n")
    f.write("   - Cross-dataset uniqueness verified\n\n")

    f.write("## Processed Files\n\n")
    f.write("| Filename | Size (KB) | Rows | Columns | Sample IDs |\n")
    f.write("|----------|-----------|------|---------|------------|\n")

    for info in file_info:
        sample_id_str = ", ".join([f"`{id}`" for id in info['sample_ids'][:2]])
        if len(info['sample_ids']) > 2:
            sample_id_str += ", ..."

        f.write(f"| {info['filename']} | {info['size_kb']:.1f} | {info['rows']} | {info['columns']} | {sample_id_str} |\n")

    f.write("\n## Duplicate IDs\n\n")
    duplicate_report_path = os.path.join(OUTPUT_DIR, DUPLICATE_IDS_REPORT_FILENAME)
    if os.path.exists(duplicate_report_path):
        with open(duplicate_report_path, "r") as dup_file:
            f.write("```\n")
            f.write(dup_file.read())
            f.write("```\n\n")
    else:
        f.write("No duplicate IDs report found.\n\n")

    f.write("## Merged Dataset\n\n")
    f.write("A merged dataset was created combining data from all individual datasets, using the project_sample_id as the key.\n")
    f.write("This allows for easy access to all related data for a given sample across different datasets.\n\n")

    f.write("## Next Steps\n\n")
    f.write("The processed datasets can now be used for relationship mapping and analysis. The project_sample_id column\n")
    f.write("provides a consistent way to identify and link samples across different datasets.\n")

print(f"  Summary report created: {report_path}")
print("\nProcessing complete!")


  # %%
FOLDER_PATH_LOAD_IDS = '/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/csv_outputs'

# import all ids in datasets
df_ac_summary_ids = pd.read_csv(FOLDER_PATH_LOAD_IDS + '/dataset_acid_consumption_summary_summaries_with_id.csv')
df_chemchar_ids = pd.read_csv(FOLDER_PATH_LOAD_IDS + '/dataset_characterization_summary_with_id.csv')
df_mineralogy_modals_ids = pd.read_csv(FOLDER_PATH_LOAD_IDS + '/dataset_mineralogy_summary_modals_with_id.csv')
df_reactors_conditions_ids = pd.read_csv(FOLDER_PATH_LOAD_IDS + '/dataset_reactors_conditions_with_id.csv')
df_reactors_summary_ids = pd.read_csv(FOLDER_PATH_LOAD_IDS + '/dataset_reactor_summary_summaries_with_id.csv')
df_reactors_detailed_ids = pd.read_csv(FOLDER_PATH_LOAD_IDS + '/dataset_reactor_summary_detailed_with_id.csv')


df_ac_summary_ids[df_ac_summary_ids['project_name'] == '030 Jetti Project File']
df_reactors_summary_ids[df_reactors_summary_ids['project_name'] == '030 Jetti Project File']
df_reactors_summary_ids[df_reactors_summary_ids['project_name'] == '032J - UGM-3 Jetti File']

df_reactors_summary_ids[df_reactors_summary_ids['project_name'] == '026 Jetti Project File']


#%% =============== SPECIAL TREATENT FOR ELEPHANT

# add an average row of data for project_sample_id_raw = 'jettiprojectfileelephantscl_tblmodalsmel_sampleescondida' and 'jettiprojectfileelephantsite_tblmineralogymodalsm1_elephanthead'
# IDs to average
ids_to_average = [
    'jettiprojectfileelephantsite_tblmineralogymodalsm1_elephanthead',
    'jettiprojectfileelephantscl_tblmodalsmel_sampleescondida'
]

# Select the rows to average
rows_to_average = df_mineralogy_modals_ids[df_mineralogy_modals_ids['project_sample_id_raw'].isin(ids_to_average)]

# Treat NaNs as 0 for averaging
rows_to_average_filled = rows_to_average.iloc[:, 6:-1].fillna(0)

# Compute the mean for numeric columns
mean_row = rows_to_average_filled.select_dtypes(include='number').mean()

# For non-numeric columns, you can set them to None or a representative value
new_row = {}
for col in df_mineralogy_modals_ids.columns:
    if col == 'project_sample_id_raw':
        new_row[col] = '007jettiprojectfile_elephant'
    elif col in mean_row.index:
        new_row[col] = mean_row[col]
    else:
        new_row[col] = None  # or set to a representative value if needed

# Append the new row to the DataFrame
df_mineralogy_modals_ids = pd.concat(
    [df_mineralogy_modals_ids, pd.DataFrame([new_row])],
    ignore_index=True
)

print(list(df_mineralogy_modals_ids.loc[df_mineralogy_modals_ids['project_name'] == '026 Jetti Project File']['project_sample_id_raw']))


#%%

# Create a list with all the unique ids for column 'project_sample_id_raw'
all_ids = pd.Series(df_ac_summary_ids['project_sample_id_raw'].tolist() + 
                    df_chemchar_ids['project_sample_id_raw'].tolist() +
                    df_mineralogy_modals_ids['project_sample_id_raw'].tolist() + 
                    df_reactors_conditions_ids['project_sample_id_raw'].tolist() +
                    df_reactors_summary_ids['project_sample_id_raw'].tolist() + 
                    df_reactors_detailed_ids['project_sample_id_raw'].tolist()).unique().tolist()
all_ids = sorted(all_ids)
print(all_ids)

grouped_ids = defaultdict(list)


for id_ in all_ids:
    match = re.match(r'^(.*?file)', id_)
    prefix = match.group(1) if match else 'no_prefix'
    grouped_ids[prefix].append(id_)

group_vars = {}
# Create variables dynamically for each group
for prefix, ids in grouped_ids.items():
    # Ensure valid Python variable name by replacing invalid characters
    print(prefix)
    var_name = prefix.rstrip("_").replace("-", "_").replace(" ", "_")
    var_name = re.sub(r'\W|^(?=\d)', '_', var_name)  # valid identifier
    globals()[var_name] = ids  # This creates the variable in the global scope
    group_vars[var_name] = ids  # Store in the group_vars dictionary

# Example: print one of the created variables to confirm
for var in list(globals().keys()):  # make a static list copy of keys
    if var.endswith("_file"):
        print(f"{var} = {globals()[var]}")

print("\nStored in group_vars:")
for key, value in group_vars.items():
    print(f"{key}: {value}")

# %%

original_printed = {
    '_003jettiprojectfile': ['003jettiprojectfile_003rt1', '003jettiprojectfile_003rt10', '003jettiprojectfile_003rt11', '003jettiprojectfile_003rt2', '003jettiprojectfile_003rt3', '003jettiprojectfile_003rt4', '003jettiprojectfile_003rt5', '003jettiprojectfile_003rt6', '003jettiprojectfile_003rt7', '003jettiprojectfile_003rt8', '003jettiprojectfile_003rt9', '003jettiprojectfile_amcf+1', '003jettiprojectfile_amcf+12', '003jettiprojectfile_amcf+14', '003jettiprojectfile_amcf+34', '003jettiprojectfile_amcf+6mesh', '003jettiprojectfile_amcf6mesh', '003jettiprojectfile_amcfhead', '003jettiprojectfile_amcfhead_tblmineralogymodals_%', '003jettiprojectfile_catalyzedbe2residue_tblmineralogymodals_%', '003jettiprojectfile_controlbe1residue_tblmineralogymodals_%', '003jettiprojectfile_pv_rt1', '003jettiprojectfile_pv_rt10', '003jettiprojectfile_pv_rt10r', '003jettiprojectfile_pv_rt11', '003jettiprojectfile_pv_rt2', '003jettiprojectfile_pv_rt3', '003jettiprojectfile_pv_rt4', '003jettiprojectfile_pv_rt5', '003jettiprojectfile_pv_rt6', '003jettiprojectfile_pv_rt7', '003jettiprojectfile_pv_rt8', '003jettiprojectfile_pv_rt9', '003jettiprojectfile_repamcf6mesh', '003jettiprojectfile_tblrt1', '003jettiprojectfile_tblrt10', '003jettiprojectfile_tblrt10r', '003jettiprojectfile_tblrt11', '003jettiprojectfile_tblrt2', '003jettiprojectfile_tblrt3', '003jettiprojectfile_tblrt4', '003jettiprojectfile_tblrt5', '003jettiprojectfile_tblrt6', '003jettiprojectfile_tblrt7', '003jettiprojectfile_tblrt8', '003jettiprojectfile_tblrt9'],
    '_007ajettiprojectfile': ['007ajettiprojectfile_acpq_pq', '007ajettiprojectfile_acugm2_ugm2', '007ajettiprojectfile_elephant2raffshipment26jul2024avg', '007ajettiprojectfile_hrt1', '007ajettiprojectfile_hrt10', '007ajettiprojectfile_hrt11', '007ajettiprojectfile_hrt12', '007ajettiprojectfile_hrt1r', '007ajettiprojectfile_hrt2', '007ajettiprojectfile_hrt3', '007ajettiprojectfile_hrt4', '007ajettiprojectfile_hrt5', '007ajettiprojectfile_hrt6', '007ajettiprojectfile_hrt7', '007ajettiprojectfile_hrt8', '007ajettiprojectfile_hrt9', '007ajettiprojectfile_pq_rt47', '007ajettiprojectfile_pq_rt48', '007ajettiprojectfile_pq_rt49', '007ajettiprojectfile_pq_rt50', '007ajettiprojectfile_pq_rt51', '007ajettiprojectfile_pq_rt52', '007ajettiprojectfile_pq_rt59', '007ajettiprojectfile_pq_rt60', '007ajettiprojectfile_pq_rt61', '007ajettiprojectfile_pq_rt62', '007ajettiprojectfile_pq_rt62r', '007ajettiprojectfile_pq_rt63', '007ajettiprojectfile_pq_rt64', '007ajettiprojectfile_pq_rt73', '007ajettiprojectfile_pq_rt74', '007ajettiprojectfile_pqcolumnheadavg', '007ajettiprojectfile_pqreactorsrt47to50heads', '007ajettiprojectfile_rt41', '007ajettiprojectfile_rt41r', '007ajettiprojectfile_rt42', '007ajettiprojectfile_rt43', '007ajettiprojectfile_rt44', '007ajettiprojectfile_rt44r', '007ajettiprojectfile_rt45', '007ajettiprojectfile_rt46', '007ajettiprojectfile_rt47', '007ajettiprojectfile_rt48', '007ajettiprojectfile_rt49', '007ajettiprojectfile_rt50', '007ajettiprojectfile_rt51', '007ajettiprojectfile_rt52', '007ajettiprojectfile_rt53', '007ajettiprojectfile_rt54', '007ajettiprojectfile_rt55', '007ajettiprojectfile_rt56', '007ajettiprojectfile_rt57', '007ajettiprojectfile_rt58', '007ajettiprojectfile_rt59', '007ajettiprojectfile_rt60', '007ajettiprojectfile_rt61', '007ajettiprojectfile_rt62', '007ajettiprojectfile_rt62r', '007ajettiprojectfile_rt63', '007ajettiprojectfile_rt64', '007ajettiprojectfile_rt65', '007ajettiprojectfile_rt66', '007ajettiprojectfile_rt67', '007ajettiprojectfile_rt68', '007ajettiprojectfile_rt69', '007ajettiprojectfile_rt70', '007ajettiprojectfile_rt73', '007ajettiprojectfile_rt74', '007ajettiprojectfile_tblhrt1', '007ajettiprojectfile_tblhrt10', '007ajettiprojectfile_tblhrt11', '007ajettiprojectfile_tblhrt12', '007ajettiprojectfile_tblhrt1r', '007ajettiprojectfile_tblhrt2', '007ajettiprojectfile_tblhrt3', '007ajettiprojectfile_tblhrt4', '007ajettiprojectfile_tblhrt5', '007ajettiprojectfile_tblhrt6', '007ajettiprojectfile_tblhrt7', '007ajettiprojectfile_tblhrt8', '007ajettiprojectfile_tblhrt9', '007ajettiprojectfile_tblmineralogymodalspq_%', '007ajettiprojectfile_tblmineralogymodalsugm2_%', '007ajettiprojectfile_tblrt41', '007ajettiprojectfile_tblrt41r', '007ajettiprojectfile_tblrt42', '007ajettiprojectfile_tblrt43', '007ajettiprojectfile_tblrt44', '007ajettiprojectfile_tblrt44r', '007ajettiprojectfile_tblrt45', '007ajettiprojectfile_tblrt46', '007ajettiprojectfile_tblrt47', '007ajettiprojectfile_tblrt48', '007ajettiprojectfile_tblrt49', '007ajettiprojectfile_tblrt50', '007ajettiprojectfile_tblrt51', '007ajettiprojectfile_tblrt52', '007ajettiprojectfile_tblrt53', '007ajettiprojectfile_tblrt54', '007ajettiprojectfile_tblrt55', '007ajettiprojectfile_tblrt56', '007ajettiprojectfile_tblrt57', '007ajettiprojectfile_tblrt58', '007ajettiprojectfile_tblrt59', '007ajettiprojectfile_tblrt60', '007ajettiprojectfile_tblrt61', '007ajettiprojectfile_tblrt62', '007ajettiprojectfile_tblrt62r', '007ajettiprojectfile_tblrt63', '007ajettiprojectfile_tblrt64', '007ajettiprojectfile_tblrt65', '007ajettiprojectfile_tblrt66', '007ajettiprojectfile_tblrt67', '007ajettiprojectfile_tblrt68', '007ajettiprojectfile_tblrt69', '007ajettiprojectfile_tblrt70', '007ajettiprojectfile_tblrt73', '007ajettiprojectfile_tblrt74', '007ajettiprojectfile_ugm2_hrt1', '007ajettiprojectfile_ugm2_hrt10', '007ajettiprojectfile_ugm2_hrt11', '007ajettiprojectfile_ugm2_hrt12', '007ajettiprojectfile_ugm2_hrt1r', '007ajettiprojectfile_ugm2_hrt2', '007ajettiprojectfile_ugm2_hrt3', '007ajettiprojectfile_ugm2_hrt4', '007ajettiprojectfile_ugm2_hrt5', '007ajettiprojectfile_ugm2_hrt6', '007ajettiprojectfile_ugm2_hrt7', '007ajettiprojectfile_ugm2_hrt8', '007ajettiprojectfile_ugm2_hrt9', '007ajettiprojectfile_ugm2_rt41', '007ajettiprojectfile_ugm2_rt41r', '007ajettiprojectfile_ugm2_rt42', '007ajettiprojectfile_ugm2_rt43', '007ajettiprojectfile_ugm2_rt44', '007ajettiprojectfile_ugm2_rt44r', '007ajettiprojectfile_ugm2_rt45', '007ajettiprojectfile_ugm2_rt46', '007ajettiprojectfile_ugm2_rt53', '007ajettiprojectfile_ugm2_rt54', '007ajettiprojectfile_ugm2_rt55', '007ajettiprojectfile_ugm2_rt56', '007ajettiprojectfile_ugm2_rt57', '007ajettiprojectfile_ugm2_rt58', '007ajettiprojectfile_ugm2_rt65', '007ajettiprojectfile_ugm2_rt66', '007ajettiprojectfile_ugm2_rt67', '007ajettiprojectfile_ugm2_rt68', '007ajettiprojectfile_ugm2_rt69', '007ajettiprojectfile_ugm2_rt70', '007ajettiprojectfile_ugm2columnheadavg', '007ajettiprojectfile_ugm2reactorsrt41to44heads'],
    '_007bjettiprojectfile': ['007bjettiprojectfiletiger_6925domain7', '007bjettiprojectfiletiger_6925domain7dup', '007bjettiprojectfiletiger_6925doman7avg', '007bjettiprojectfiletiger_acsummary_tiger', '007bjettiprojectfiletiger_domain7_rt10', '007bjettiprojectfiletiger_domain7_rt11', '007bjettiprojectfiletiger_domain7_rt12', '007bjettiprojectfiletiger_domain7_rt9', '007bjettiprojectfiletiger_rt1', '007bjettiprojectfiletiger_rt10', '007bjettiprojectfiletiger_rt11', '007bjettiprojectfiletiger_rt12', '007bjettiprojectfiletiger_rt2', '007bjettiprojectfiletiger_rt3', '007bjettiprojectfiletiger_rt4', '007bjettiprojectfiletiger_rt5', '007bjettiprojectfiletiger_rt6', '007bjettiprojectfiletiger_rt7', '007bjettiprojectfiletiger_rt8', '007bjettiprojectfiletiger_rt9', '007bjettiprojectfiletiger_tblmineralcomposition_%', '007bjettiprojectfiletiger_tblrt1', '007bjettiprojectfiletiger_tblrt10', '007bjettiprojectfiletiger_tblrt11', '007bjettiprojectfiletiger_tblrt12', '007bjettiprojectfiletiger_tblrt2', '007bjettiprojectfiletiger_tblrt3', '007bjettiprojectfiletiger_tblrt4', '007bjettiprojectfiletiger_tblrt5', '007bjettiprojectfiletiger_tblrt6', '007bjettiprojectfiletiger_tblrt7', '007bjettiprojectfiletiger_tblrt8', '007bjettiprojectfiletiger_tblrt9', '007bjettiprojectfiletiger_tiger_rt1', '007bjettiprojectfiletiger_tiger_rt2', '007bjettiprojectfiletiger_tiger_rt3', '007bjettiprojectfiletiger_tiger_rt4', '007bjettiprojectfiletiger_tiger_rt5', '007bjettiprojectfiletiger_tiger_rt6', '007bjettiprojectfiletiger_tiger_rt7', '007bjettiprojectfiletiger_tiger_rt8', '007bjettiprojectfiletiger_tigerhead', '007bjettiprojectfiletiger_tigersample+12inch', '007bjettiprojectfiletiger_tigersample+14inch', '007bjettiprojectfiletiger_tigersample+34inch', '007bjettiprojectfiletiger_tigersample+6mesh', '007bjettiprojectfiletiger_tigersample6mesh'],
    '_007jettiprojectfile': ['007jettiprojectfile_elephant', '007jettiprojectfileleopard_007rt1', '007jettiprojectfileleopard_007rt2', '007jettiprojectfileleopard_007rt2r', '007jettiprojectfileleopard_007rt3', '007jettiprojectfileleopard_007rt37', '007jettiprojectfileleopard_007rt38', '007jettiprojectfileleopard_007rt39', '007jettiprojectfileleopard_007rt3r', '007jettiprojectfileleopard_007rt4', '007jettiprojectfileleopard_007rt40', '007jettiprojectfileleopard_007rt4r', '007jettiprojectfileleopard_007rt5', '007jettiprojectfileleopard_007rt5r', '007jettiprojectfileleopard_007rt6', '007jettiprojectfileleopard_007rt6r', '007jettiprojectfileleopard_007rt7', '007jettiprojectfileleopard_007rt7r', '007jettiprojectfileleopard_007rt8', '007jettiprojectfileleopard_acleopard_leopard', '007jettiprojectfileleopard_elephant_rt5', '007jettiprojectfileleopard_elephant_rt5r', '007jettiprojectfileleopard_elephant_rt6', '007jettiprojectfileleopard_elephant_rt6r', '007jettiprojectfileleopard_elephant_rt7', '007jettiprojectfileleopard_elephant_rt7r', '007jettiprojectfileleopard_elephant_rt8', '007jettiprojectfileleopard_leopard', '007jettiprojectfileleopard_leopard_rt1', '007jettiprojectfileleopard_leopard_rt2', '007jettiprojectfileleopard_leopard_rt2r', '007jettiprojectfileleopard_leopard_rt3', '007jettiprojectfileleopard_leopard_rt37', '007jettiprojectfileleopard_leopard_rt38', '007jettiprojectfileleopard_leopard_rt39', '007jettiprojectfileleopard_leopard_rt3r', '007jettiprojectfileleopard_leopard_rt4', '007jettiprojectfileleopard_leopard_rt40', '007jettiprojectfileleopard_leopard_rt4r', '007jettiprojectfileleopard_leoparddup', '007jettiprojectfileleopard_tblmineralogymodalslep_%', '007jettiprojectfileleopard_tblrt1', '007jettiprojectfileleopard_tblrt2', '007jettiprojectfileleopard_tblrt2r', '007jettiprojectfileleopard_tblrt3', '007jettiprojectfileleopard_tblrt37', '007jettiprojectfileleopard_tblrt38', '007jettiprojectfileleopard_tblrt39', '007jettiprojectfileleopard_tblrt3r', '007jettiprojectfileleopard_tblrt4', '007jettiprojectfileleopard_tblrt40', '007jettiprojectfileleopard_tblrt4r', '007jettiprojectfileleopard_tblrt5', '007jettiprojectfileleopard_tblrt5r', '007jettiprojectfileleopard_tblrt6', '007jettiprojectfileleopard_tblrt6r', '007jettiprojectfileleopard_tblrt7', '007jettiprojectfileleopard_tblrt7r', '007jettiprojectfileleopard_tblrt8', '007jettiprojectfilertm2_acsummary_rtm2', '007jettiprojectfilertm2_rt1forcheckvalues', '007jettiprojectfilertm2_rt33', '007jettiprojectfilertm2_rt34', '007jettiprojectfilertm2_rt35', '007jettiprojectfilertm2_rt36', '007jettiprojectfilertm2_rtm2_rt33', '007jettiprojectfilertm2_rtm2_rt34', '007jettiprojectfilertm2_rtm2_rt35', '007jettiprojectfilertm2_rtm2_rt36', '007jettiprojectfilertm2_rtm2a', '007jettiprojectfilertm2_rtm2b', '007jettiprojectfilertm2_tblmineralogymodalsrtm2_%', '007jettiprojectfilertm2_tblrt1d', '007jettiprojectfilertm2_tblrt33', '007jettiprojectfilertm2_tblrt34', '007jettiprojectfilertm2_tblrt35', '007jettiprojectfilertm2_tblrt36', '007jettiprojectfiletoquepala_acsummaryantigua_antigua', '007jettiprojectfiletoquepala_acsummaryfresca_fresca', '007jettiprojectfiletoquepala_rt21', '007jettiprojectfiletoquepala_rt22', '007jettiprojectfiletoquepala_rt23', '007jettiprojectfiletoquepala_rt24', '007jettiprojectfiletoquepala_rt25', '007jettiprojectfiletoquepala_rt26', '007jettiprojectfiletoquepala_rt27', '007jettiprojectfiletoquepala_rt28', '007jettiprojectfiletoquepala_tblmineralogymodalstoquepalaantigua_%', '007jettiprojectfiletoquepala_tblmineralogymodalstoquepalafrecsa_%', '007jettiprojectfiletoquepala_tblrt21', '007jettiprojectfiletoquepala_tblrt22', '007jettiprojectfiletoquepala_tblrt23', '007jettiprojectfiletoquepala_tblrt24', '007jettiprojectfiletoquepala_tblrt25', '007jettiprojectfiletoquepala_tblrt26', '007jettiprojectfiletoquepala_tblrt27', '007jettiprojectfiletoquepala_tblrt28', '007jettiprojectfiletoquepala_toquepalaantigua_rt25', '007jettiprojectfiletoquepala_toquepalaantigua_rt26', '007jettiprojectfiletoquepala_toquepalaantigua_rt27', '007jettiprojectfiletoquepala_toquepalaantigua_rt28', '007jettiprojectfiletoquepala_toquepalaantiguaa', '007jettiprojectfiletoquepala_toquepalaantiguab', '007jettiprojectfiletoquepala_toquepalaantiguasxs+10mesh', '007jettiprojectfiletoquepala_toquepalaantiguasxs+12inch', '007jettiprojectfiletoquepala_toquepalaantiguasxs+14inch', '007jettiprojectfiletoquepala_toquepalaantiguasxs+34inch', '007jettiprojectfiletoquepala_toquepalaantiguasxs10mesh', '007jettiprojectfiletoquepala_toquepalafresca_rt21', '007jettiprojectfiletoquepala_toquepalafresca_rt22', '007jettiprojectfiletoquepala_toquepalafresca_rt23', '007jettiprojectfiletoquepala_toquepalafresca_rt24', '007jettiprojectfiletoquepala_toquepalafrescaa', '007jettiprojectfiletoquepala_toquepalafrescab', '007jettiprojectfilezaldivar_acsummary_zaldivar', '007jettiprojectfilezaldivar_rt29', '007jettiprojectfilezaldivar_rt30', '007jettiprojectfilezaldivar_rt31', '007jettiprojectfilezaldivar_rt32', '007jettiprojectfilezaldivar_tblmineralogymodalszaldivar_%', '007jettiprojectfilezaldivar_tblrt29', '007jettiprojectfilezaldivar_tblrt30', '007jettiprojectfilezaldivar_tblrt31', '007jettiprojectfilezaldivar_tblrt32', '007jettiprojectfilezaldivar_zaldivar_rt29', '007jettiprojectfilezaldivar_zaldivar_rt30', '007jettiprojectfilezaldivar_zaldivar_rt31', '007jettiprojectfilezaldivar_zaldivar_rt32', '007jettiprojectfilezaldivar_zaldivarcpya', '007jettiprojectfilezaldivar_zaldivarcpyb', '007jettiprojectfilezaldivar_zaldivarcpysxs+10mesh', '007jettiprojectfilezaldivar_zaldivarcpysxs+12inch', '007jettiprojectfilezaldivar_zaldivarcpysxs+14inch', '007jettiprojectfilezaldivar_zaldivarcpysxs10mesh'],
    '_011jettiprojectfile': ['011jettiprojectfile_%_tblmineralogymodals', '011jettiprojectfile_011rt10', '011jettiprojectfile_011rt14', '011jettiprojectfile_011rt15', '011jettiprojectfile_011rt7', '011jettiprojectfile_011rt8', '011jettiprojectfile_011rt9', '011jettiprojectfile_021rt1', '011jettiprojectfile_021rt2', '011jettiprojectfile_acsummary_rm2020', '011jettiprojectfile_catresiduekg_tblmineralogymodals_%', '011jettiprojectfile_controlresiduekg_tblmineralogymodals_%', '011jettiprojectfile_headkg_tblmineralogymodals_%', '011jettiprojectfile_rm1catresidue_tblmineralogymodals_%', '011jettiprojectfile_rm2020_rt1', '011jettiprojectfile_rm2020_rt10', '011jettiprojectfile_rm2020_rt14', '011jettiprojectfile_rm2020_rt15', '011jettiprojectfile_rm2020_rt2', '011jettiprojectfile_rm2020_rt7', '011jettiprojectfile_rm2020_rt8', '011jettiprojectfile_rm2020_rt9', '011jettiprojectfile_rm2controlresidue_tblmineralogymodals_%', '011jettiprojectfile_rmheadsample', '011jettiprojectfile_tblrt1', '011jettiprojectfile_tblrt10', '011jettiprojectfile_tblrt14', '011jettiprojectfile_tblrt15', '011jettiprojectfile_tblrt2', '011jettiprojectfile_tblrt7', '011jettiprojectfile_tblrt8', '011jettiprojectfile_tblrt9', '011jettiprojectfilecrushed_rm2024_rt17', '011jettiprojectfilecrushed_rm2024_rt18', '011jettiprojectfilecrushed_rm2024_rt19', '011jettiprojectfilecrushed_rm2024_rt20', '011jettiprojectfilecrushed_rm2024_rt21', '011jettiprojectfilecrushed_rm2024_rt22', '011jettiprojectfilecrushed_rm2024_rt23', '011jettiprojectfilecrushed_rm2024_rt24', '011jettiprojectfilecrushed_tblrt17', '011jettiprojectfilecrushed_tblrt18', '011jettiprojectfilecrushed_tblrt19', '011jettiprojectfilecrushed_tblrt20', '011jettiprojectfilecrushed_tblrt21', '011jettiprojectfilecrushed_tblrt22', '011jettiprojectfilecrushed_tblrt23', '011jettiprojectfilecrushed_tblrt24'],
    '_012jettiprojectfile': ['012jettiprojectfilecs_012rt1', '012jettiprojectfilecs_012rt2', '012jettiprojectfilecs_012rt3', '012jettiprojectfilecs_012rt4', '012jettiprojectfilecs_012rt5', '012jettiprojectfilecs_012rt6', '012jettiprojectfilecs_012rtb', '012jettiprojectfilecs_012rte', '012jettiprojectfilecs_012rtf', '012jettiprojectfilecs_012rtg', '012jettiprojectfilecs_acsummaryincremento_incremento', '012jettiprojectfilecs_acsummarykino_kino', '012jettiprojectfilecs_acsummaryquebalix_quebalix', '012jettiprojectfilecs_incremento', '012jettiprojectfilecs_incremento_rt1', '012jettiprojectfilecs_incremento_rt2', '012jettiprojectfilecs_incremento_rt3', '012jettiprojectfilecs_incremento_rt4', '012jettiprojectfilecs_incremento_rt5', '012jettiprojectfilecs_incremento_rt6', '012jettiprojectfilecs_incremento_rtb', '012jettiprojectfilecs_incremento_rte', '012jettiprojectfilecs_incremento_rtf', '012jettiprojectfilecs_incremento_rtg', '012jettiprojectfilecs_incremento_tblmineralogymodals_%', '012jettiprojectfilecs_kino', '012jettiprojectfilecs_kino_tblmineralogymodals_%', '012jettiprojectfilecs_quebalix', '012jettiprojectfilecs_quebalix_rt10', '012jettiprojectfilecs_quebalix_rt7', '012jettiprojectfilecs_quebalix_rt8', '012jettiprojectfilecs_quebalix_rt9', '012jettiprojectfilecs_quebalix_rta', '012jettiprojectfilecs_quebalix_rtc', '012jettiprojectfilecs_quebalix_rtd', '012jettiprojectfilecs_quebalixiv_tblmineralogymodals_%', '012jettiprojectfilecs_rt10', '012jettiprojectfilecs_rt7', '012jettiprojectfilecs_rt8', '012jettiprojectfilecs_rt9', '012jettiprojectfilecs_rta', '012jettiprojectfilecs_rtc', '012jettiprojectfilecs_rtd', '012jettiprojectfilecs_tblrt1', '012jettiprojectfilecs_tblrt10', '012jettiprojectfilecs_tblrt2', '012jettiprojectfilecs_tblrt3', '012jettiprojectfilecs_tblrt4', '012jettiprojectfilecs_tblrt5', '012jettiprojectfilecs_tblrt6', '012jettiprojectfilecs_tblrt7', '012jettiprojectfilecs_tblrt8', '012jettiprojectfilecs_tblrt9', '012jettiprojectfilecs_tblrta', '012jettiprojectfilecs_tblrtb', '012jettiprojectfilecs_tblrtc', '012jettiprojectfilecs_tblrtd', '012jettiprojectfilecs_tblrte', '012jettiprojectfilecs_tblrtf', '012jettiprojectfilecs_tblrtg'],
    '_014jettiprojectfile': ['014jettiprojectfile_acsummarybag_bag', '014jettiprojectfile_acsummarykmb_kmb', '014jettiprojectfile_bag_rtb1', '014jettiprojectfile_bag_rtb2', '014jettiprojectfile_bag_rtb3', '014jettiprojectfile_bag_rtb4', '014jettiprojectfile_bag_rtb5', '014jettiprojectfile_bag_rtb6', '014jettiprojectfile_bag_rtb7', '014jettiprojectfile_bag_rtb8', '014jettiprojectfile_bag_rtb9', '014jettiprojectfile_baghead', '014jettiprojectfile_kmb_rtk1r', '014jettiprojectfile_kmb_rtk2', '014jettiprojectfile_kmb_rtk3', '014jettiprojectfile_kmb_rtk4', '014jettiprojectfile_kmb_rtk5', '014jettiprojectfile_kmb_rtk6', '014jettiprojectfile_kmb_rtk7', '014jettiprojectfile_kmb_rtk8', '014jettiprojectfile_kmb_rtk9', '014jettiprojectfile_kmbhead', '014jettiprojectfile_tblbagmineralogymodals_bag%', '014jettiprojectfile_tblkmbmineralogymodals_mineralmasskmb%', '014jettiprojectfile_tblrtb1', '014jettiprojectfile_tblrtb2', '014jettiprojectfile_tblrtb3', '014jettiprojectfile_tblrtb4', '014jettiprojectfile_tblrtb5', '014jettiprojectfile_tblrtb6', '014jettiprojectfile_tblrtb7', '014jettiprojectfile_tblrtb8', '014jettiprojectfile_tblrtk1r', '014jettiprojectfile_tblrtk2', '014jettiprojectfile_tblrtk3', '014jettiprojectfile_tblrtk4', '014jettiprojectfile_tblrtk5', '014jettiprojectfile_tblrtk6', '014jettiprojectfile_tblrtk7', '014jettiprojectfile_tblrtk8'],
    '_015jettiprojectfile': ['015jettiprojectfile_003rt1', '015jettiprojectfile_003rt2', '015jettiprojectfile_003rt3', '015jettiprojectfile_acsummary_acmr', '015jettiprojectfile_amcf+1', '015jettiprojectfile_amcf+12', '015jettiprojectfile_amcf+14', '015jettiprojectfile_amcf+34', '015jettiprojectfile_amcf+6mesh', '015jettiprojectfile_amcf6mesh', '015jettiprojectfile_amcf_hrt1', '015jettiprojectfile_amcf_hrt2', '015jettiprojectfile_amcf_hrt3', '015jettiprojectfile_amcf_hrt4', '015jettiprojectfile_amcf_hrt5', '015jettiprojectfile_amcfhead', '015jettiprojectfile_pv_rt1', '015jettiprojectfile_pv_rt2', '015jettiprojectfile_pv_rt3', '015jettiprojectfile_repamcf6mesh', '015jettiprojectfile_tblhrt1', '015jettiprojectfile_tblhrt2', '015jettiprojectfile_tblhrt3', '015jettiprojectfile_tblmineralogymodalssgs_%', '015jettiprojectfile_tblrt1', '015jettiprojectfile_tblrt2', '015jettiprojectfile_tblrt3'],
    '_017jettiprojectfile': ['017jettiprojectfile_017rtea1', '017jettiprojectfile_017rtea2', '017jettiprojectfile_017rtea3', '017jettiprojectfile_017rtea4', '017jettiprojectfile_017rtea5', '017jettiprojectfile_017rtea6', '017jettiprojectfile_72hracsummary_eamillfeed', '017jettiprojectfile_catea4residue%_tblelamineralogymodals_%', '017jettiprojectfile_catea4residuekg_tblelamineralogymodals_%', '017jettiprojectfile_controlea1residue%_tblelamineralogymodals_%', '017jettiprojectfile_controlea1residuekg_tblelamineralogymodals_%', '017jettiprojectfile_eamillfeed+1', '017jettiprojectfile_eamillfeed+10m', '017jettiprojectfile_eamillfeed+12', '017jettiprojectfile_eamillfeed+14', '017jettiprojectfile_eamillfeed+150m', '017jettiprojectfile_eamillfeed+34', '017jettiprojectfile_eamillfeed+6m', '017jettiprojectfile_eamillfeed150m', '017jettiprojectfile_eamillfeed_rtea1', '017jettiprojectfile_eamillfeed_rtea2', '017jettiprojectfile_eamillfeed_rtea3', '017jettiprojectfile_eamillfeed_rtea4', '017jettiprojectfile_eamillfeed_rtea5', '017jettiprojectfile_eamillfeed_rtea6', '017jettiprojectfile_eamillfeed_rtea7', '017jettiprojectfile_eamillfeedcombined', '017jettiprojectfile_head%_tblelamineralogymodals_%', '017jettiprojectfile_headkg_tblelamineralogymodals_%', '017jettiprojectfile_tblrtea1', '017jettiprojectfile_tblrtea2', '017jettiprojectfile_tblrtea3', '017jettiprojectfile_tblrtea4', '017jettiprojectfile_tblrtea5', '017jettiprojectfile_tblrtea6'],
    '_020jettiprojectfile': ['020jettiprojectfilehardyandwaste_020rt25', '020jettiprojectfilehardyandwaste_020rt26', '020jettiprojectfilehardyandwaste_020rt27', '020jettiprojectfilehardyandwaste_020rt28', '020jettiprojectfilehardyandwaste_acsummary_hardy', '020jettiprojectfilehardyandwaste_h21c1', '020jettiprojectfilehardyandwaste_h21c2', '020jettiprojectfilehardyandwaste_h21e', '020jettiprojectfilehardyandwaste_h21mastercomp', '020jettiprojectfilehardyandwaste_h21n', '020jettiprojectfilehardyandwaste_h21nw', '020jettiprojectfilehardyandwaste_h21sw', '020jettiprojectfilehardyandwaste_har%_tblmineralogymodals_%', '020jettiprojectfilehardyandwaste_harcatalyzed%_tblmineralogymodals_%', '020jettiprojectfilehardyandwaste_harcatkg_tblmineralogymodals_%', '020jettiprojectfilehardyandwaste_harcontrol%_tblmineralogymodals_%', '020jettiprojectfilehardyandwaste_harcontrolkg_tblmineralogymodals_%', '020jettiprojectfilehardyandwaste_hardy_rt25', '020jettiprojectfilehardyandwaste_hardy_rt26', '020jettiprojectfilehardyandwaste_hardy_rt27', '020jettiprojectfilehardyandwaste_hardy_rt28', '020jettiprojectfilehardyandwaste_harheadkg_tblmineralogymodals_%', '020jettiprojectfilehardyandwaste_tblrt25', '020jettiprojectfilehardyandwaste_tblrt26', '020jettiprojectfilehardyandwaste_tblrt27', '020jettiprojectfilehardyandwaste_tblrt28', '020jettiprojectfilehardyandwaste_wda', '020jettiprojectfilehardyandwaste_wdb', '020jettiprojectfilehardyandwaste_wdc', '020jettiprojectfilehardyandwaste_wdd', '020jettiprojectfilehardyandwaste_wde', '020jettiprojectfilehardyandwaste_wdf', '020jettiprojectfilehypogenesupergene_020rt1', '020jettiprojectfilehypogenesupergene_020rt19', '020jettiprojectfilehypogenesupergene_020rt2', '020jettiprojectfilehypogenesupergene_020rt20', '020jettiprojectfilehypogenesupergene_020rt3', '020jettiprojectfilehypogenesupergene_020rt4', '020jettiprojectfilehypogenesupergene_020rt5', '020jettiprojectfilehypogenesupergene_020rt6', '020jettiprojectfilehypogenesupergene_achyp_hypogenecomp', '020jettiprojectfilehypogenesupergene_acsup_supergene', '020jettiprojectfilehypogenesupergene_hyp%_tblmineralogymodalshyp_%', '020jettiprojectfilehypogenesupergene_hypcatresidue%_tblmineralogymodalshyp_%', '020jettiprojectfilehypogenesupergene_hypcatresiduekg_tblmineralogymodalshyp_%', '020jettiprojectfilehypogenesupergene_hypcontrolresidue%_tblmineralogymodalshyp_%', '020jettiprojectfilehypogenesupergene_hypcontrolresiduekg_tblmineralogymodalshyp_%', '020jettiprojectfilehypogenesupergene_hypheadkg_tblmineralogymodalshyp_%', '020jettiprojectfilehypogenesupergene_hypoamp', '020jettiprojectfilehypogenesupergene_hypobporp', '020jettiprojectfilehypogenesupergene_hypodio', '020jettiprojectfilehypogenesupergene_hypogene_rt1', '020jettiprojectfilehypogenesupergene_hypogene_rt2', '020jettiprojectfilehypogenesupergene_hypogene_rt3', '020jettiprojectfilehypogenesupergene_hypogene_rt4', '020jettiprojectfilehypogenesupergene_hypogene_rt5', '020jettiprojectfilehypogenesupergene_hypogene_rt6', '020jettiprojectfilehypogenesupergene_hypogenemastercompositea', '020jettiprojectfilehypogenesupergene_hypogenemastercompositeb', '020jettiprojectfilehypogenesupergene_hypoporp', '020jettiprojectfilehypogenesupergene_hypoqfg', '020jettiprojectfilehypogenesupergene_super', '020jettiprojectfilehypogenesupergene_supergene_rt19', '020jettiprojectfilehypogenesupergene_supergene_rt20', '020jettiprojectfilehypogenesupergene_tblmineralogymodalshardy_har%', '020jettiprojectfilehypogenesupergene_tblmineralogymodalssup2_sup%', '020jettiprojectfilehypogenesupergene_tblrt1', '020jettiprojectfilehypogenesupergene_tblrt19', '020jettiprojectfilehypogenesupergene_tblrt2', '020jettiprojectfilehypogenesupergene_tblrt20', '020jettiprojectfilehypogenesupergene_tblrt3', '020jettiprojectfilehypogenesupergene_tblrt4', '020jettiprojectfilehypogenesupergene_tblrt5', '020jettiprojectfilehypogenesupergene_tblrt6'],
    '_021jettiprojectfile': ['021jettiprojectfile_021rt1', '021jettiprojectfile_021rt2', '021jettiprojectfile_021rt3', '021jettiprojectfile_021rt4', '021jettiprojectfile_021rt5', '021jettiprojectfile_acsummaryenriched_enriched', '021jettiprojectfile_acsummaryhypogene24hr_hypogene', '021jettiprojectfile_acsummaryhypogene72hr_hypogene', '021jettiprojectfile_enriched', '021jettiprojectfile_hypogene', '021jettiprojectfile_hypogene_rt1', '021jettiprojectfile_hypogene_rt2', '021jettiprojectfile_hypogene_rt3', '021jettiprojectfile_hypogene_rt4', '021jettiprojectfile_hypogene_rt5', '021jettiprojectfile_tblmineralogymodals_%', '021jettiprojectfile_tblrt1', '021jettiprojectfile_tblrt2', '021jettiprojectfile_tblrt3', '021jettiprojectfile_tblrt4', '021jettiprojectfile_tblrt5'],
    '_022jettiprojectfile': ['022jettiprojectfile_022rt1', '022jettiprojectfile_022rt10', '022jettiprojectfile_022rt11', '022jettiprojectfile_022rt12', '022jettiprojectfile_022rt13', '022jettiprojectfile_022rt2', '022jettiprojectfile_022rt3', '022jettiprojectfile_022rt4', '022jettiprojectfile_022rt5', '022jettiprojectfile_022rt6', '022jettiprojectfile_022rt7', '022jettiprojectfile_022rt8', '022jettiprojectfile_022rt9', '022jettiprojectfile_acsummary_belowcutoffgrade', '022jettiprojectfile_belowcutoffgrade_catalyzed', '022jettiprojectfile_belowcutoffgrade_control', '022jettiprojectfile_belowcutoffgrade_rt1', '022jettiprojectfile_belowcutoffgrade_rt10', '022jettiprojectfile_belowcutoffgrade_rt11', '022jettiprojectfile_belowcutoffgrade_rt12', '022jettiprojectfile_belowcutoffgrade_rt13', '022jettiprojectfile_belowcutoffgrade_rt2', '022jettiprojectfile_belowcutoffgrade_rt3', '022jettiprojectfile_belowcutoffgrade_rt4', '022jettiprojectfile_belowcutoffgrade_rt5', '022jettiprojectfile_belowcutoffgrade_rt6', '022jettiprojectfile_belowcutoffgrade_rt9', '022jettiprojectfile_stingray1head', '022jettiprojectfile_stingray1sxs+12', '022jettiprojectfile_stingray1sxs+14', '022jettiprojectfile_stingray1sxs+34', '022jettiprojectfile_stingray1sxs+6mesh', '022jettiprojectfile_stingray1sxs6mesh', '022jettiprojectfile_tblmineralcomposition_%', '022jettiprojectfile_tblrt1', '022jettiprojectfile_tblrt10', '022jettiprojectfile_tblrt11', '022jettiprojectfile_tblrt12', '022jettiprojectfile_tblrt13', '022jettiprojectfile_tblrt2', '022jettiprojectfile_tblrt3', '022jettiprojectfile_tblrt4', '022jettiprojectfile_tblrt5', '022jettiprojectfile_tblrt6', '022jettiprojectfile_tblrt7', '022jettiprojectfile_tblrt8', '022jettiprojectfile_tblrt9'],
    '_023jettiprojectfile': ['023jettiprojectfile_023rt1', '023jettiprojectfile_023rt10', '023jettiprojectfile_023rt11', '023jettiprojectfile_023rt12r', '023jettiprojectfile_023rt13r', '023jettiprojectfile_023rt14r', '023jettiprojectfile_023rt15r', '023jettiprojectfile_023rt16r', '023jettiprojectfile_023rt2', '023jettiprojectfile_023rt3', '023jettiprojectfile_023rt4', '023jettiprojectfile_023rt5r', '023jettiprojectfile_023rt6r', '023jettiprojectfile_023rt7r', '023jettiprojectfile_023rt8', '023jettiprojectfile_023rt8r', '023jettiprojectfile_023rt9', '023jettiprojectfile_acsummaryot1024hrs_ot10', '023jettiprojectfile_acsummaryot1072hrs_ot10', '023jettiprojectfile_acsummaryot924hrs_ot9', '023jettiprojectfile_acsummaryot972hrs_ot9', '023jettiprojectfile_ot10', '023jettiprojectfile_ot10_rt10', '023jettiprojectfile_ot10_rt11', '023jettiprojectfile_ot10_rt12r', '023jettiprojectfile_ot10_rt13r', '023jettiprojectfile_ot10_rt14r', '023jettiprojectfile_ot10_rt15r', '023jettiprojectfile_ot10_rt8', '023jettiprojectfile_ot10_rt8r', '023jettiprojectfile_ot10_rt9', '023jettiprojectfile_ot10avg', '023jettiprojectfile_ot10dup', '023jettiprojectfile_ot9', '023jettiprojectfile_ot9_rt1', '023jettiprojectfile_ot9_rt16r', '023jettiprojectfile_ot9_rt2', '023jettiprojectfile_ot9_rt3', '023jettiprojectfile_ot9_rt4', '023jettiprojectfile_ot9_rt5r', '023jettiprojectfile_ot9_rt6r', '023jettiprojectfile_ot9_rt7r', '023jettiprojectfile_ot9avg', '023jettiprojectfile_ot9dup', '023jettiprojectfile_siteraffinate', '023jettiprojectfile_tblmineralogymodalsot10_ot10%', '023jettiprojectfile_tblmineralogymodalsot9_ot09%', '023jettiprojectfile_tblrt1', '023jettiprojectfile_tblrt10', '023jettiprojectfile_tblrt11', '023jettiprojectfile_tblrt12r', '023jettiprojectfile_tblrt13r', '023jettiprojectfile_tblrt14r', '023jettiprojectfile_tblrt15r', '023jettiprojectfile_tblrt16r', '023jettiprojectfile_tblrt2', '023jettiprojectfile_tblrt3', '023jettiprojectfile_tblrt4', '023jettiprojectfile_tblrt5r', '023jettiprojectfile_tblrt6r', '023jettiprojectfile_tblrt7r', '023jettiprojectfile_tblrt8', '023jettiprojectfile_tblrt8r', '023jettiprojectfile_tblrt9'],
    '_024jettiprojectfile': ['024jettiprojectfile_024cvcpy', '024jettiprojectfile_024cvcpy+12', '024jettiprojectfile_024cvcpy+14', '024jettiprojectfile_024cvcpy+34', '024jettiprojectfile_024cvcpy+6mesh', '024jettiprojectfile_024cvcpy6mesh', '024jettiprojectfile_024cvcpyavg', '024jettiprojectfile_024cvcpydup', '024jettiprojectfile_24hracsummary_belowcutoff', '024jettiprojectfile_belowcutoff_rt1', '024jettiprojectfile_belowcutoff_rt2', '024jettiprojectfile_belowcutoff_rt3', '024jettiprojectfile_belowcutoff_rt4', '024jettiprojectfile_belowcutoff_rt5', '024jettiprojectfile_belowcutoff_rt6', '024jettiprojectfile_belowcutoff_rt7', '024jettiprojectfile_belowcutoff_rt8', '024jettiprojectfile_belowcutoff_rt9', '024jettiprojectfile_catalyzedkg_tblmineralogymodals_%', '024jettiprojectfile_catresiduecv4%_tblmineralogymodals_%', '024jettiprojectfile_controlkg_tblmineralogymodals_%', '024jettiprojectfile_controlresiduecv1%_tblmineralogymodals_%', '024jettiprojectfile_fecontrolcatcv3%_tblmineralogymodals_%', '024jettiprojectfile_fecontrolcatcv3kg_tblmineralogymodals_%', '024jettiprojectfile_fectrlcontrolcv2%_tblmineralogymodals_%', '024jettiprojectfile_fectrlcontrolcv2kg_tblmineralogymodals_%', '024jettiprojectfile_head%_tblmineralogymodals_%', '024jettiprojectfile_headkg_tblmineralogymodals_%', '024jettiprojectfile_syntheticraffinate17112022', '024jettiprojectfile_syntheticraffinatedup17112022', '024jettiprojectfile_tblrt1', '024jettiprojectfile_tblrt2', '024jettiprojectfile_tblrt3', '024jettiprojectfile_tblrt4', '024jettiprojectfile_tblrt5', '024jettiprojectfile_tblrt6', '024jettiprojectfile_tblrt7'],
    '_025jettiprojectfile': ['025jettiprojectfile_023rt14', '025jettiprojectfile_025rt1', '025jettiprojectfile_025rt10', '025jettiprojectfile_025rt11', '025jettiprojectfile_025rt12', '025jettiprojectfile_025rt13', '025jettiprojectfile_025rt15', '025jettiprojectfile_025rt16', '025jettiprojectfile_025rt17', '025jettiprojectfile_025rt18', '025jettiprojectfile_025rt19', '025jettiprojectfile_025rt2', '025jettiprojectfile_025rt20', '025jettiprojectfile_025rt21', '025jettiprojectfile_025rt22', '025jettiprojectfile_025rt23', '025jettiprojectfile_025rt24', '025jettiprojectfile_025rt25', '025jettiprojectfile_025rt26', '025jettiprojectfile_025rt27', '025jettiprojectfile_025rt28', '025jettiprojectfile_025rt29', '025jettiprojectfile_025rt3', '025jettiprojectfile_025rt30', '025jettiprojectfile_025rt31', '025jettiprojectfile_025rt32', '025jettiprojectfile_025rt33', '025jettiprojectfile_025rt34', '025jettiprojectfile_025rt39', '025jettiprojectfile_025rt4', '025jettiprojectfile_025rt40', '025jettiprojectfile_025rt41', '025jettiprojectfile_025rt42', '025jettiprojectfile_025rt43', '025jettiprojectfile_025rt5', '025jettiprojectfile_025rt6', '025jettiprojectfile_025rt7', '025jettiprojectfile_025rt8', '025jettiprojectfile_025rt9', '025jettiprojectfile_acsummary_chalcopyrite', '025jettiprojectfile_acsummary_highsecondary', '025jettiprojectfile_acsummary_oxide', '025jettiprojectfile_chalcopyrite', '025jettiprojectfile_chalcopyrite_rt17', '025jettiprojectfile_chalcopyrite_rt18', '025jettiprojectfile_chalcopyrite_rt19', '025jettiprojectfile_chalcopyrite_rt20', '025jettiprojectfile_chalcopyrite_rt21', '025jettiprojectfile_chalcopyrite_rt22', '025jettiprojectfile_chalcopyrite_rt23', '025jettiprojectfile_chalcopyrite_rt24', '025jettiprojectfile_chalcopyrite_rt25', '025jettiprojectfile_chalcopyrite_rt26', '025jettiprojectfile_chalcopyrite_rt27', '025jettiprojectfile_chalcopyrite_rt28', '025jettiprojectfile_chalcopyrite_rt29', '025jettiprojectfile_chalcopyrite_rt30', '025jettiprojectfile_chalcopyrite_rt39', '025jettiprojectfile_chalcopyrite_rt40', '025jettiprojectfile_chalcopyrite_rt41', '025jettiprojectfile_chalcopyrite_rt42', '025jettiprojectfile_chalcopyrite_rt43', '025jettiprojectfile_chalcopyrite_rt44', '025jettiprojectfile_chalcopyrite_rt45', '025jettiprojectfile_chalcopyritedup', '025jettiprojectfile_highsecondary', '025jettiprojectfile_highsecondary_rt1', '025jettiprojectfile_highsecondary_rt2', '025jettiprojectfile_highsecondary_rt3', '025jettiprojectfile_highsecondary_rt31', '025jettiprojectfile_highsecondary_rt32', '025jettiprojectfile_highsecondary_rt33', '025jettiprojectfile_highsecondary_rt34', '025jettiprojectfile_highsecondary_rt4', '025jettiprojectfile_highsecondary_rt5', '025jettiprojectfile_highsecondary_rt6', '025jettiprojectfile_highsecondary_rt7', '025jettiprojectfile_highsecondary_rt8', '025jettiprojectfile_highsecondarydup', '025jettiprojectfile_oxide', '025jettiprojectfile_oxide_rt10', '025jettiprojectfile_oxide_rt11', '025jettiprojectfile_oxide_rt12', '025jettiprojectfile_oxide_rt13', '025jettiprojectfile_oxide_rt14', '025jettiprojectfile_oxide_rt15', '025jettiprojectfile_oxide_rt16', '025jettiprojectfile_oxide_rt35', '025jettiprojectfile_oxide_rt36', '025jettiprojectfile_oxide_rt37', '025jettiprojectfile_oxide_rt38', '025jettiprojectfile_oxide_rt9', '025jettiprojectfile_oxidedup', '025jettiprojectfile_tblmineralogymodalschalcopyrite_%', '025jettiprojectfile_tblmineralogymodalsoxide_%', '025jettiprojectfile_tblmineralogymodalssecondary_%', '025jettiprojectfile_tblrt1', '025jettiprojectfile_tblrt10', '025jettiprojectfile_tblrt11', '025jettiprojectfile_tblrt12', '025jettiprojectfile_tblrt13', '025jettiprojectfile_tblrt14', '025jettiprojectfile_tblrt15', '025jettiprojectfile_tblrt16', '025jettiprojectfile_tblrt17', '025jettiprojectfile_tblrt18', '025jettiprojectfile_tblrt19', '025jettiprojectfile_tblrt2', '025jettiprojectfile_tblrt20', '025jettiprojectfile_tblrt21', '025jettiprojectfile_tblrt22', '025jettiprojectfile_tblrt23', '025jettiprojectfile_tblrt24', '025jettiprojectfile_tblrt25', '025jettiprojectfile_tblrt26', '025jettiprojectfile_tblrt27', '025jettiprojectfile_tblrt28', '025jettiprojectfile_tblrt29', '025jettiprojectfile_tblrt3', '025jettiprojectfile_tblrt30', '025jettiprojectfile_tblrt31', '025jettiprojectfile_tblrt32', '025jettiprojectfile_tblrt33', '025jettiprojectfile_tblrt34', '025jettiprojectfile_tblrt35', '025jettiprojectfile_tblrt36', '025jettiprojectfile_tblrt37', '025jettiprojectfile_tblrt38', '025jettiprojectfile_tblrt39', '025jettiprojectfile_tblrt4', '025jettiprojectfile_tblrt40', '025jettiprojectfile_tblrt41', '025jettiprojectfile_tblrt42', '025jettiprojectfile_tblrt43', '025jettiprojectfile_tblrt44', '025jettiprojectfile_tblrt45', '025jettiprojectfile_tblrt5', '025jettiprojectfile_tblrt6', '025jettiprojectfile_tblrt7', '025jettiprojectfile_tblrt8', '025jettiprojectfile_tblrt9'],
    '_026jettiprojectfile': ['026jettiprojectfile_acsample1_sample1', '026jettiprojectfile_acsample2_sample2', '026jettiprojectfile_acsample3_sample3', '026jettiprojectfile_acsample4_sample4', '026jettiprojectfile_raffinatereceivednov42024', '026jettiprojectfile_raffinateshipment1', '026jettiprojectfile_raffinateshipment2', '026jettiprojectfile_rt1', '026jettiprojectfile_rt10', '026jettiprojectfile_rt11', '026jettiprojectfile_rt12', '026jettiprojectfile_rt13', '026jettiprojectfile_rt14', '026jettiprojectfile_rt15', '026jettiprojectfile_rt16', '026jettiprojectfile_rt17', '026jettiprojectfile_rt18', '026jettiprojectfile_rt19', '026jettiprojectfile_rt2', '026jettiprojectfile_rt20', '026jettiprojectfile_rt21', '026jettiprojectfile_rt22', '026jettiprojectfile_rt23', '026jettiprojectfile_rt24', '026jettiprojectfile_rt25', '026jettiprojectfile_rt26', '026jettiprojectfile_rt27', '026jettiprojectfile_rt28', '026jettiprojectfile_rt29', '026jettiprojectfile_rt3', '026jettiprojectfile_rt30', '026jettiprojectfile_rt31', '026jettiprojectfile_rt32', '026jettiprojectfile_rt4', '026jettiprojectfile_rt5', '026jettiprojectfile_rt6', '026jettiprojectfile_rt7', '026jettiprojectfile_rt8', '026jettiprojectfile_rt9', '026jettiprojectfile_sample1primarysulfide', '026jettiprojectfile_sample1primarysulfide_rt1', '026jettiprojectfile_sample1primarysulfide_rt2', '026jettiprojectfile_sample1primarysulfide_rt3', '026jettiprojectfile_sample1primarysulfide_rt33', '026jettiprojectfile_sample1primarysulfide_rt34', '026jettiprojectfile_sample1primarysulfide_rt35', '026jettiprojectfile_sample1primarysulfide_rt36', '026jettiprojectfile_sample1primarysulfide_rt4', '026jettiprojectfile_sample1primarysulfide_rt5', '026jettiprojectfile_sample1primarysulfide_rt6', '026jettiprojectfile_sample1primarysulfide_rt7', '026jettiprojectfile_sample1primarysulfide_rt8', '026jettiprojectfile_sample2carrizalillo', '026jettiprojectfile_sample2carrizalillo_rt10', '026jettiprojectfile_sample2carrizalillo_rt11', '026jettiprojectfile_sample2carrizalillo_rt12', '026jettiprojectfile_sample2carrizalillo_rt13', '026jettiprojectfile_sample2carrizalillo_rt14', '026jettiprojectfile_sample2carrizalillo_rt15', '026jettiprojectfile_sample2carrizalillo_rt16', '026jettiprojectfile_sample2carrizalillo_rt37', '026jettiprojectfile_sample2carrizalillo_rt38', '026jettiprojectfile_sample2carrizalillo_rt39', '026jettiprojectfile_sample2carrizalillo_rt40', '026jettiprojectfile_sample2carrizalillo_rt9', '026jettiprojectfile_sample3secondarysulfide_rt17', '026jettiprojectfile_sample3secondarysulfide_rt18', '026jettiprojectfile_sample3secondarysulfide_rt20', '026jettiprojectfile_sample3secondarysulfide_rt21', '026jettiprojectfile_sample3secondarysulfide_rt22', '026jettiprojectfile_sample3secondarysulfide_rt23', '026jettiprojectfile_sample3secondarysulfide_rt24', '026jettiprojectfile_sample3secondarysulfide_rt41', '026jettiprojectfile_sample3secondarysulfide_rt42', '026jettiprojectfile_sample3secondarysulfide_rt43', '026jettiprojectfile_sample3secondarysulfide_rt44', '026jettiprojectfile_sample3secondarysulfide_rt49', '026jettiprojectfile_sample3secondarysulfide_rt50', '026jettiprojectfile_sample3secondarysulfide_rt51', '026jettiprojectfile_sample3secondarysulfide_rt52', '026jettiprojectfile_sample3secondarysulfide_rt53', '026jettiprojectfile_sample4mixedmaterial', '026jettiprojectfile_sample4mixedmaterial_rt25', '026jettiprojectfile_sample4mixedmaterial_rt26', '026jettiprojectfile_sample4mixedmaterial_rt27', '026jettiprojectfile_sample4mixedmaterial_rt28', '026jettiprojectfile_sample4mixedmaterial_rt29', '026jettiprojectfile_sample4mixedmaterial_rt30', '026jettiprojectfile_sample4mixedmaterial_rt31', '026jettiprojectfile_sample4mixedmaterial_rt32', '026jettiprojectfile_sample4mixedmaterial_rt45', '026jettiprojectfile_sample4mixedmaterial_rt46', '026jettiprojectfile_sample4mixedmaterial_rt47', '026jettiprojectfile_sample4mixedmaterial_rt48', '026jettiprojectfile_tblmineralogymodalscarrizalillo_%', '026jettiprojectfile_tblmineralogymodalsmixed_%', '026jettiprojectfile_tblmineralogymodalsprimarysulfide_%', '026jettiprojectfile_tblmineralogymodalssecondarysulfide_%', '026jettiprojectfile_tblrt1', '026jettiprojectfile_tblrt10', '026jettiprojectfile_tblrt11', '026jettiprojectfile_tblrt12', '026jettiprojectfile_tblrt13', '026jettiprojectfile_tblrt14', '026jettiprojectfile_tblrt15', '026jettiprojectfile_tblrt16', '026jettiprojectfile_tblrt17', '026jettiprojectfile_tblrt18', '026jettiprojectfile_tblrt19', '026jettiprojectfile_tblrt2', '026jettiprojectfile_tblrt20', '026jettiprojectfile_tblrt21', '026jettiprojectfile_tblrt22', '026jettiprojectfile_tblrt23', '026jettiprojectfile_tblrt24', '026jettiprojectfile_tblrt25', '026jettiprojectfile_tblrt26', '026jettiprojectfile_tblrt27', '026jettiprojectfile_tblrt28', '026jettiprojectfile_tblrt29', '026jettiprojectfile_tblrt3', '026jettiprojectfile_tblrt30', '026jettiprojectfile_tblrt31', '026jettiprojectfile_tblrt32', '026jettiprojectfile_tblrt33', '026jettiprojectfile_tblrt34', '026jettiprojectfile_tblrt35', '026jettiprojectfile_tblrt36', '026jettiprojectfile_tblrt37', '026jettiprojectfile_tblrt38', '026jettiprojectfile_tblrt39', '026jettiprojectfile_tblrt4', '026jettiprojectfile_tblrt40', '026jettiprojectfile_tblrt41', '026jettiprojectfile_tblrt42', '026jettiprojectfile_tblrt43', '026jettiprojectfile_tblrt44', '026jettiprojectfile_tblrt45', '026jettiprojectfile_tblrt46', '026jettiprojectfile_tblrt47', '026jettiprojectfile_tblrt48', '026jettiprojectfile_tblrt49', '026jettiprojectfile_tblrt5', '026jettiprojectfile_tblrt50', '026jettiprojectfile_tblrt51', '026jettiprojectfile_tblrt52', '026jettiprojectfile_tblrt53', '026jettiprojectfile_tblrt6', '026jettiprojectfile_tblrt7', '026jettiprojectfile_tblrt8', '026jettiprojectfile_tblrt9'],
    '_028jettiprojectfile': ['028jettiprojectfile_028rt11', '028jettiprojectfile_028rt12', '028jettiprojectfile_028rt13', '028jettiprojectfile_028rt14', '028jettiprojectfile_acandesitesummary_andesite', '028jettiprojectfile_accompsummary_composite', '028jettiprojectfile_acmonzonitesummary_monzonite', '028jettiprojectfile_andesite', '028jettiprojectfile_andesite_rt1', '028jettiprojectfile_andesite_rt2', '028jettiprojectfile_andesite_rt3', '028jettiprojectfile_andesite_rt4', '028jettiprojectfile_andesitedup', '028jettiprojectfile_composite', '028jettiprojectfile_composite_rt11', '028jettiprojectfile_composite_rt12', '028jettiprojectfile_composite_rt13', '028jettiprojectfile_composite_rt14', '028jettiprojectfile_monzonite', '028jettiprojectfile_monzonite_rt10', '028jettiprojectfile_monzonite_rt7', '028jettiprojectfile_monzonite_rt8', '028jettiprojectfile_monzonite_rt9', '028jettiprojectfile_monzonitedup', '028jettiprojectfile_tblmineralogymodalsandesite_%', '028jettiprojectfile_tblmineralogymodalscomp_%', '028jettiprojectfile_tblmineralogymodalsmonzonite_%', '028jettiprojectfile_tblrt1', '028jettiprojectfile_tblrt10', '028jettiprojectfile_tblrt11', '028jettiprojectfile_tblrt12', '028jettiprojectfile_tblrt13', '028jettiprojectfile_tblrt14', '028jettiprojectfile_tblrt2', '028jettiprojectfile_tblrt3', '028jettiprojectfile_tblrt4', '028jettiprojectfile_tblrt7', '028jettiprojectfile_tblrt8', '028jettiprojectfile_tblrt9'],
    '_030jettiprojectfile': ['030jettiprojectfile_acsummarycpy_cpy', '030jettiprojectfile_acsummaryss_ss', '030jettiprojectfile_cpy', '030jettiprojectfile_cpy_rt1', '030jettiprojectfile_cpy_rt1r', '030jettiprojectfile_cpy_rt2', '030jettiprojectfile_cpy_rt3', '030jettiprojectfile_cpy_rt4', '030jettiprojectfile_cpy_rt4r', '030jettiprojectfile_rt1', '030jettiprojectfile_rt1r', '030jettiprojectfile_rt2', '030jettiprojectfile_rt3', '030jettiprojectfile_rt4', '030jettiprojectfile_rt4r', '030jettiprojectfile_ss', '030jettiprojectfile_tblcpymodals_%', '030jettiprojectfile_tblrt1', '030jettiprojectfile_tblrt1r', '030jettiprojectfile_tblrt2', '030jettiprojectfile_tblrt3', '030jettiprojectfile_tblrt4', '030jettiprojectfile_tblrt4r', '030jettiprojectfile_tblssmodals_%'],
    '_031jettiprojectfile': ['031jettiprojectfile_031rt1', '031jettiprojectfile_031rt2', '031jettiprojectfile_031rt3', '031jettiprojectfile_031rt4', '031jettiprojectfile_acsummary_antcomp', '031jettiprojectfile_antcomp', '031jettiprojectfile_antcomp_rt1', '031jettiprojectfile_antcomp_rt2', '031jettiprojectfile_antcomp_rt3', '031jettiprojectfile_antcomp_rt4', '031jettiprojectfile_tblcumodals031_%', '031jettiprojectfile_tblrt1', '031jettiprojectfile_tblrt2', '031jettiprojectfile_tblrt3', '031jettiprojectfile_tblrt4'],
    '_032augm2jettifile': ['032augm2jettifile_acsummary_ugm2', '032augm2jettifile_elephant2ugma', '032augm2jettifile_elephant2ugmb', '032augm2jettifile_elephant2ugmc', '032augm2jettifile_repelephant2ugma', '032augm2jettifile_rt1', '032augm2jettifile_rt2', '032augm2jettifile_rt3', '032augm2jettifile_rt4', '032augm2jettifile_rt5', '032augm2jettifile_rt6', '032augm2jettifile_rt7', '032augm2jettifile_rt8', '032augm2jettifile_tblmineralogymodalsugm2_%', '032augm2jettifile_tblrt1', '032augm2jettifile_tblrt2', '032augm2jettifile_tblrt3', '032augm2jettifile_tblrt4', '032augm2jettifile_tblrt5', '032augm2jettifile_tblrt6', '032augm2jettifile_tblrt7', '032augm2jettifile_tblrt8', '032augm2jettifile_ugm2_rt1', '032augm2jettifile_ugm2_rt2', '032augm2jettifile_ugm2_rt3', '032augm2jettifile_ugm2_rt4', '032augm2jettifile_ugm2_rt5', '032augm2jettifile_ugm2_rt6', '032augm2jettifile_ugm2_rt7', '032augm2jettifile_ugm2_rt8', '032augm2jettifile_ugm2average', '032augm2jettifile_acsummary_ugm2', '032augm2jettifile_elephant2ugma', '032augm2jettifile_elephant2ugmb', '032augm2jettifile_elephant2ugmc', '032augm2jettifile_repelephant2ugma', '032augm2jettifile_rt1', '032augm2jettifile_rt2', '032augm2jettifile_rt3', '032augm2jettifile_rt4', '032augm2jettifile_rt5', '032augm2jettifile_rt6', '032augm2jettifile_rt7', '032augm2jettifile_rt8', '032augm2jettifile_tblmineralogymodalsugm2_%', '032augm2jettifile_tblrt1', '032augm2jettifile_tblrt2', '032augm2jettifile_tblrt3', '032augm2jettifile_tblrt4', '032augm2jettifile_tblrt5', '032augm2jettifile_tblrt6', '032augm2jettifile_tblrt7', '032augm2jettifile_tblrt8', '032augm2jettifile_ugm2_rt1', '032augm2jettifile_ugm2_rt2', '032augm2jettifile_ugm2_rt3', '032augm2jettifile_ugm2_rt4', '032augm2jettifile_ugm2_rt5', '032augm2jettifile_ugm2_rt6', '032augm2jettifile_ugm2_rt7', '032augm2jettifile_ugm2_rt8', '032augm2jettifile_ugm2average'],
    '_032bpqjettifile': ['032bpqjettifile_acsummary_pq', '032bpqjettifile_elephant2pqa', '032bpqjettifile_elephant2pqb', '032bpqjettifile_elephant2pqc', '032bpqjettifile_pq_rt10', '032bpqjettifile_pq_rt11', '032bpqjettifile_pq_rt12', '032bpqjettifile_pq_rt13', '032bpqjettifile_pq_rt14', '032bpqjettifile_pq_rt15', '032bpqjettifile_pq_rt16', '032bpqjettifile_pq_rt9', '032bpqjettifile_pqaverage', '032bpqjettifile_rt10', '032bpqjettifile_rt11', '032bpqjettifile_rt12', '032bpqjettifile_rt13', '032bpqjettifile_rt14', '032bpqjettifile_rt15', '032bpqjettifile_rt16', '032bpqjettifile_rt9', '032bpqjettifile_tblmineralogymodalspq_%', '032bpqjettifile_tblrt10', '032bpqjettifile_tblrt11', '032bpqjettifile_tblrt12', '032bpqjettifile_tblrt13', '032bpqjettifile_tblrt14', '032bpqjettifile_tblrt15', '032bpqjettifile_tblrt16', '032bpqjettifile_tblrt9', '032bpqjettifile_acsummary_pq', '032bpqjettifile_elephant2pqa', '032bpqjettifile_elephant2pqb', '032bpqjettifile_elephant2pqc', '032bpqjettifile_pq_rt10', '032bpqjettifile_pq_rt11', '032bpqjettifile_pq_rt12', '032bpqjettifile_pq_rt13', '032bpqjettifile_pq_rt14', '032bpqjettifile_pq_rt15', '032bpqjettifile_pq_rt16', '032bpqjettifile_pq_rt9', '032bpqjettifile_pqaverage', '032bpqjettifile_rt10', '032bpqjettifile_rt11', '032bpqjettifile_rt12', '032bpqjettifile_rt13', '032bpqjettifile_rt14', '032bpqjettifile_rt15', '032bpqjettifile_rt16', '032bpqjettifile_rt9', '032bpqjettifile_tblmineralogymodalspq_%', '032bpqjettifile_tblrt10', '032bpqjettifile_tblrt11', '032bpqjettifile_tblrt12', '032bpqjettifile_tblrt13', '032bpqjettifile_tblrt14', '032bpqjettifile_tblrt15', '032bpqjettifile_tblrt16', '032bpqjettifile_tblrt9'],
    '_032cugm2hjettifile': ['032cugm2hjettifile_acsummary_ugm2h', '032cugm2hjettifile_rt17', '032cugm2hjettifile_rt18', '032cugm2hjettifile_rt19', '032cugm2hjettifile_rt20', '032cugm2hjettifile_rt21', '032cugm2hjettifile_rt22', '032cugm2hjettifile_rt23', '032cugm2hjettifile_rt24', '032cugm2hjettifile_rt25', '032cugm2hjettifile_rt26', '032cugm2hjettifile_rt27', '032cugm2hjettifile_rt28', '032cugm2hjettifile_tblmineralogymodalsugm2h_%', '032cugm2hjettifile_tblrt17', '032cugm2hjettifile_tblrt18', '032cugm2hjettifile_tblrt19', '032cugm2hjettifile_tblrt20', '032cugm2hjettifile_tblrt21', '032cugm2hjettifile_tblrt22', '032cugm2hjettifile_tblrt23', '032cugm2hjettifile_tblrt24', '032cugm2hjettifile_tblrt25', '032cugm2hjettifile_tblrt26', '032cugm2hjettifile_tblrt27', '032cugm2hjettifile_tblrt28', '032cugm2hjettifile_ugm2h_rt17', '032cugm2hjettifile_ugm2h_rt18', '032cugm2hjettifile_ugm2h_rt19', '032cugm2hjettifile_ugm2h_rt20', '032cugm2hjettifile_ugm2h_rt21', '032cugm2hjettifile_ugm2h_rt22', '032cugm2hjettifile_ugm2h_rt23', '032cugm2hjettifile_ugm2h_rt24', '032cugm2hjettifile_ugm2h_rt25', '032cugm2hjettifile_ugm2h_rt26', '032cugm2hjettifile_ugm2h_rt27', '032cugm2hjettifile_ugm2h_rt28', '032cugm2hjettifile_ugm2high', '032cugm2hjettifile_ugm2high+12inches', '032cugm2hjettifile_ugm2high+14inches', '032cugm2hjettifile_ugm2high+15inches', '032cugm2hjettifile_ugm2high+1inches', '032cugm2hjettifile_ugm2high+34inches', '032cugm2hjettifile_ugm2high+6mesh', '032cugm2hjettifile_ugm2high6mesh', '032cugm2hjettifile_ugm2highaverage', '032cugm2hjettifile_ugm2highdup', '032cugm2hjettifile_acsummary_ugm2h', '032cugm2hjettifile_rt17', '032cugm2hjettifile_rt18', '032cugm2hjettifile_rt19', '032cugm2hjettifile_rt20', '032cugm2hjettifile_rt21', '032cugm2hjettifile_rt22', '032cugm2hjettifile_rt23', '032cugm2hjettifile_rt24', '032cugm2hjettifile_rt25', '032cugm2hjettifile_rt26', '032cugm2hjettifile_rt27', '032cugm2hjettifile_rt28', '032cugm2hjettifile_tblmineralogymodalsugm2h_%', '032cugm2hjettifile_tblrt17', '032cugm2hjettifile_tblrt18', '032cugm2hjettifile_tblrt19', '032cugm2hjettifile_tblrt20', '032cugm2hjettifile_tblrt21', '032cugm2hjettifile_tblrt22', '032cugm2hjettifile_tblrt23', '032cugm2hjettifile_tblrt24', '032cugm2hjettifile_tblrt25', '032cugm2hjettifile_tblrt26', '032cugm2hjettifile_tblrt27', '032cugm2hjettifile_tblrt28', '032cugm2hjettifile_ugm2h_rt17', '032cugm2hjettifile_ugm2h_rt18', '032cugm2hjettifile_ugm2h_rt19', '032cugm2hjettifile_ugm2h_rt20', '032cugm2hjettifile_ugm2h_rt21', '032cugm2hjettifile_ugm2h_rt22', '032cugm2hjettifile_ugm2h_rt23', '032cugm2hjettifile_ugm2h_rt24', '032cugm2hjettifile_ugm2h_rt25', '032cugm2hjettifile_ugm2h_rt26', '032cugm2hjettifile_ugm2h_rt27', '032cugm2hjettifile_ugm2h_rt28', '032cugm2hjettifile_ugm2high', '032cugm2hjettifile_ugm2high+12inches', '032cugm2hjettifile_ugm2high+14inches', '032cugm2hjettifile_ugm2high+15inches', '032cugm2hjettifile_ugm2high+1inches', '032cugm2hjettifile_ugm2high+34inches', '032cugm2hjettifile_ugm2high+6mesh', '032cugm2hjettifile_ugm2high6mesh', '032cugm2hjettifile_ugm2highaverage', '032cugm2hjettifile_ugm2highdup'],
    '_032dandqscfat46jettifile': ['032dandqscfat46jettifile_acsummary_andqsc', '032dandqscfat46jettifile_andqsc', '032dandqscfat46jettifile_andqsc+12inches', '032dandqscfat46jettifile_andqsc+14inches', '032dandqscfat46jettifile_andqsc+15inches', '032dandqscfat46jettifile_andqsc+1inches', '032dandqscfat46jettifile_andqsc+34inches', '032dandqscfat46jettifile_andqsc+6mesh', '032dandqscfat46jettifile_andqsc6mesh', '032dandqscfat46jettifile_andqscdup', '032dandqscfat46jettifile_andqscfat46_rt29', '032dandqscfat46jettifile_andqscfat46_rt30', '032dandqscfat46jettifile_andqscfat46_rt31', '032dandqscfat46jettifile_andqscfat46_rt32', '032dandqscfat46jettifile_andqscfat46_rt33', '032dandqscfat46jettifile_andqscfat46_rt34', '032dandqscfat46jettifile_andqscfat46_rt35', '032dandqscfat46jettifile_andqscfat46_rt36', '032dandqscfat46jettifile_andqscfat46_rt37', '032dandqscfat46jettifile_andqscfat46_rt38', '032dandqscfat46jettifile_andqscfat46_rt39', '032dandqscfat46jettifile_andqscfat46_rt40', '032dandqscfat46jettifile_andqscfat46average', '032dandqscfat46jettifile_rt29', '032dandqscfat46jettifile_rt30', '032dandqscfat46jettifile_rt31', '032dandqscfat46jettifile_rt32', '032dandqscfat46jettifile_rt33', '032dandqscfat46jettifile_rt34', '032dandqscfat46jettifile_rt35', '032dandqscfat46jettifile_rt36', '032dandqscfat46jettifile_rt37', '032dandqscfat46jettifile_rt38', '032dandqscfat46jettifile_rt39', '032dandqscfat46jettifile_rt40', '032dandqscfat46jettifile_tblmineralogymodalsandqsc_%', '032dandqscfat46jettifile_tblrt29', '032dandqscfat46jettifile_tblrt30', '032dandqscfat46jettifile_tblrt31', '032dandqscfat46jettifile_tblrt32', '032dandqscfat46jettifile_tblrt33', '032dandqscfat46jettifile_tblrt34', '032dandqscfat46jettifile_tblrt35', '032dandqscfat46jettifile_tblrt36', '032dandqscfat46jettifile_tblrt37', '032dandqscfat46jettifile_tblrt38', '032dandqscfat46jettifile_tblrt39', '032dandqscfat46jettifile_tblrt40', '032dandqscfat46jettifile_acsummary_andqsc', '032dandqscfat46jettifile_andqsc', '032dandqscfat46jettifile_andqsc+12inches', '032dandqscfat46jettifile_andqsc+14inches', '032dandqscfat46jettifile_andqsc+15inches', '032dandqscfat46jettifile_andqsc+1inches', '032dandqscfat46jettifile_andqsc+34inches', '032dandqscfat46jettifile_andqsc+6mesh', '032dandqscfat46jettifile_andqsc6mesh', '032dandqscfat46jettifile_andqscdup', '032dandqscfat46jettifile_andqscfat46_rt29', '032dandqscfat46jettifile_andqscfat46_rt30', '032dandqscfat46jettifile_andqscfat46_rt31', '032dandqscfat46jettifile_andqscfat46_rt32', '032dandqscfat46jettifile_andqscfat46_rt33', '032dandqscfat46jettifile_andqscfat46_rt34', '032dandqscfat46jettifile_andqscfat46_rt35', '032dandqscfat46jettifile_andqscfat46_rt36', '032dandqscfat46jettifile_andqscfat46_rt37', '032dandqscfat46jettifile_andqscfat46_rt38', '032dandqscfat46jettifile_andqscfat46_rt39', '032dandqscfat46jettifile_andqscfat46_rt40', '032dandqscfat46jettifile_andqscfat46average', '032dandqscfat46jettifile_rt29', '032dandqscfat46jettifile_rt30', '032dandqscfat46jettifile_rt31', '032dandqscfat46jettifile_rt32', '032dandqscfat46jettifile_rt33', '032dandqscfat46jettifile_rt34', '032dandqscfat46jettifile_rt35', '032dandqscfat46jettifile_rt36', '032dandqscfat46jettifile_rt37', '032dandqscfat46jettifile_rt38', '032dandqscfat46jettifile_rt39', '032dandqscfat46jettifile_rt40', '032dandqscfat46jettifile_tblmineralogymodalsandqsc_%', '032dandqscfat46jettifile_tblrt29', '032dandqscfat46jettifile_tblrt30', '032dandqscfat46jettifile_tblrt31', '032dandqscfat46jettifile_tblrt32', '032dandqscfat46jettifile_tblrt33', '032dandqscfat46jettifile_tblrt34', '032dandqscfat46jettifile_tblrt35', '032dandqscfat46jettifile_tblrt36', '032dandqscfat46jettifile_tblrt37', '032dandqscfat46jettifile_tblrt38', '032dandqscfat46jettifile_tblrt39', '032dandqscfat46jettifile_tblrt40'],
    '_032eactive604jettifile': ['032eactive604jettifile_acsummary_active1604', '032eactive604jettifile_active1604_rt41', '032eactive604jettifile_active1604_rt42', '032eactive604jettifile_active1604_rt43', '032eactive604jettifile_active1604_rt44', '032eactive604jettifile_active1604_rt45', '032eactive604jettifile_active1604_rt46', '032eactive604jettifile_active1604_rt47', '032eactive604jettifile_active1604_rt48', '032eactive604jettifile_active1604_rt48r', '032eactive604jettifile_active1604_rt49', '032eactive604jettifile_active1604_rt50', '032eactive604jettifile_active1604_rt51', '032eactive604jettifile_active1604_rt52', '032eactive604jettifile_active1604average', '032eactive604jettifile_active1604r604', '032eactive604jettifile_active1604r604dup', '032eactive604jettifile_r604+12inch', '032eactive604jettifile_r604+14inch', '032eactive604jettifile_r604+15inch', '032eactive604jettifile_r604+1inch', '032eactive604jettifile_r604+34inch', '032eactive604jettifile_r604+6mesh', '032eactive604jettifile_r6046mesh', '032eactive604jettifile_rt41', '032eactive604jettifile_rt42', '032eactive604jettifile_rt43', '032eactive604jettifile_rt44', '032eactive604jettifile_rt45', '032eactive604jettifile_rt46', '032eactive604jettifile_rt47', '032eactive604jettifile_rt48r', '032eactive604jettifile_rt49', '032eactive604jettifile_rt50', '032eactive604jettifile_rt51', '032eactive604jettifile_rt52', '032eactive604jettifile_tblmineralogymodalsactive1_%', '032eactive604jettifile_tblrt41', '032eactive604jettifile_tblrt42', '032eactive604jettifile_tblrt43', '032eactive604jettifile_tblrt44', '032eactive604jettifile_tblrt45', '032eactive604jettifile_tblrt46', '032eactive604jettifile_tblrt47', '032eactive604jettifile_tblrt48r', '032eactive604jettifile_tblrt49', '032eactive604jettifile_tblrt50', '032eactive604jettifile_tblrt51', '032eactive604jettifile_tblrt52', '032eactive604jettifile_acsummary_active1604', '032eactive604jettifile_active1604_rt41', '032eactive604jettifile_active1604_rt42', '032eactive604jettifile_active1604_rt43', '032eactive604jettifile_active1604_rt44', '032eactive604jettifile_active1604_rt45', '032eactive604jettifile_active1604_rt46', '032eactive604jettifile_active1604_rt47', '032eactive604jettifile_active1604_rt48', '032eactive604jettifile_active1604_rt48r', '032eactive604jettifile_active1604_rt49', '032eactive604jettifile_active1604_rt50', '032eactive604jettifile_active1604_rt51', '032eactive604jettifile_active1604_rt52', '032eactive604jettifile_active1604average', '032eactive604jettifile_active1604r604', '032eactive604jettifile_active1604r604dup', '032eactive604jettifile_r604+12inch', '032eactive604jettifile_r604+14inch', '032eactive604jettifile_r604+15inch', '032eactive604jettifile_r604+1inch', '032eactive604jettifile_r604+34inch', '032eactive604jettifile_r604+6mesh', '032eactive604jettifile_r6046mesh', '032eactive604jettifile_rt41', '032eactive604jettifile_rt42', '032eactive604jettifile_rt43', '032eactive604jettifile_rt44', '032eactive604jettifile_rt45', '032eactive604jettifile_rt46', '032eactive604jettifile_rt47', '032eactive604jettifile_rt48r', '032eactive604jettifile_rt49', '032eactive604jettifile_rt50', '032eactive604jettifile_rt51', '032eactive604jettifile_rt52', '032eactive604jettifile_tblmineralogymodalsactive1_%', '032eactive604jettifile_tblrt41', '032eactive604jettifile_tblrt42', '032eactive604jettifile_tblrt43', '032eactive604jettifile_tblrt44', '032eactive604jettifile_tblrt45', '032eactive604jettifile_tblrt46', '032eactive604jettifile_tblrt47', '032eactive604jettifile_tblrt48r', '032eactive604jettifile_tblrt49', '032eactive604jettifile_tblrt50', '032eactive604jettifile_tblrt51', '032eactive604jettifile_tblrt52'],
    '_032factive605606jettifile': ['032factive605606jettifile_acsummary_active2605606', '032factive605606jettifile_active2605606+12inch', '032factive605606jettifile_active2605606+14inch', '032factive605606jettifile_active2605606+15inch', '032factive605606jettifile_active2605606+1inch', '032factive605606jettifile_active2605606+34inch', '032factive605606jettifile_active2605606+6mesh', '032factive605606jettifile_active26056066mesh', '032factive605606jettifile_active2605606_rt53', '032factive605606jettifile_active2605606_rt54', '032factive605606jettifile_active2605606_rt55', '032factive605606jettifile_active2605606_rt56', '032factive605606jettifile_active2605606_rt57', '032factive605606jettifile_active2605606_rt58', '032factive605606jettifile_active2605606_rt59', '032factive605606jettifile_active2605606_rt60', '032factive605606jettifile_active2605606_rt61', '032factive605606jettifile_active2605606_rt62', '032factive605606jettifile_active2605606_rt63', '032factive605606jettifile_active2605606_rt64', '032factive605606jettifile_active2605606avg', '032factive605606jettifile_active2605and606', '032factive605606jettifile_active2605and606dup', '032factive605606jettifile_rt53', '032factive605606jettifile_rt54', '032factive605606jettifile_rt55', '032factive605606jettifile_rt56', '032factive605606jettifile_rt57', '032factive605606jettifile_rt58', '032factive605606jettifile_rt59', '032factive605606jettifile_rt60', '032factive605606jettifile_rt61', '032factive605606jettifile_rt62', '032factive605606jettifile_rt63', '032factive605606jettifile_rt64', '032factive605606jettifile_tblmineralogymodalsactive2_%', '032factive605606jettifile_tblrt53', '032factive605606jettifile_tblrt54', '032factive605606jettifile_tblrt55', '032factive605606jettifile_tblrt56', '032factive605606jettifile_tblrt57', '032factive605606jettifile_tblrt58', '032factive605606jettifile_tblrt59', '032factive605606jettifile_tblrt60', '032factive605606jettifile_tblrt61', '032factive605606jettifile_tblrt62', '032factive605606jettifile_tblrt63', '032factive605606jettifile_tblrt64', '032factive605606jettifile_acsummary_active2605606', '032factive605606jettifile_active2605606+12inch', '032factive605606jettifile_active2605606+14inch', '032factive605606jettifile_active2605606+15inch', '032factive605606jettifile_active2605606+1inch', '032factive605606jettifile_active2605606+34inch', '032factive605606jettifile_active2605606+6mesh', '032factive605606jettifile_active26056066mesh', '032factive605606jettifile_active2605606_rt53', '032factive605606jettifile_active2605606_rt54', '032factive605606jettifile_active2605606_rt55', '032factive605606jettifile_active2605606_rt56', '032factive605606jettifile_active2605606_rt57', '032factive605606jettifile_active2605606_rt58', '032factive605606jettifile_active2605606_rt59', '032factive605606jettifile_active2605606_rt60', '032factive605606jettifile_active2605606_rt61', '032factive605606jettifile_active2605606_rt62', '032factive605606jettifile_active2605606_rt63', '032factive605606jettifile_active2605606_rt64', '032factive605606jettifile_active2605606avg', '032factive605606jettifile_active2605and606', '032factive605606jettifile_active2605and606dup', '032factive605606jettifile_rt53', '032factive605606jettifile_rt54', '032factive605606jettifile_rt55', '032factive605606jettifile_rt56', '032factive605606jettifile_rt57', '032factive605606jettifile_rt58', '032factive605606jettifile_rt59', '032factive605606jettifile_rt60', '032factive605606jettifile_rt61', '032factive605606jettifile_rt62', '032factive605606jettifile_rt63', '032factive605606jettifile_rt64', '032factive605606jettifile_tblmineralogymodalsactive2_%', '032factive605606jettifile_tblrt53', '032factive605606jettifile_tblrt54', '032factive605606jettifile_tblrt55', '032factive605606jettifile_tblrt56', '032factive605606jettifile_tblrt57', '032factive605606jettifile_tblrt58', '032factive605606jettifile_tblrt59', '032factive605606jettifile_tblrt60', '032factive605606jettifile_tblrt61', '032factive605606jettifile_tblrt62', '032factive605606jettifile_tblrt63', '032factive605606jettifile_tblrt64'],
    '_032gugm1jettifile': ['032gugm1jettifile_acsummary_ugm1', '032gugm1jettifile_rt65', '032gugm1jettifile_rt66', '032gugm1jettifile_rt67', '032gugm1jettifile_rt68', '032gugm1jettifile_rt69', '032gugm1jettifile_rt70', '032gugm1jettifile_rt71', '032gugm1jettifile_rt72', '032gugm1jettifile_tblmineralogymodalsugm1_%', '032gugm1jettifile_tblrt65', '032gugm1jettifile_tblrt66', '032gugm1jettifile_tblrt67', '032gugm1jettifile_tblrt68', '032gugm1jettifile_tblrt69', '032gugm1jettifile_tblrt70', '032gugm1jettifile_tblrt71', '032gugm1jettifile_tblrt72', '032gugm1jettifile_ugm1', '032gugm1jettifile_ugm1_rt65', '032gugm1jettifile_ugm1_rt66', '032gugm1jettifile_ugm1_rt67', '032gugm1jettifile_ugm1_rt68', '032gugm1jettifile_ugm1_rt69', '032gugm1jettifile_ugm1_rt70', '032gugm1jettifile_ugm1_rt71', '032gugm1jettifile_ugm1_rt72', '032gugm1jettifile_ugm1average', '032gugm1jettifile_ugm1dup', '032gugm1jettifile_acsummary_ugm1', '032gugm1jettifile_rt65', '032gugm1jettifile_rt66', '032gugm1jettifile_rt67', '032gugm1jettifile_rt68', '032gugm1jettifile_rt69', '032gugm1jettifile_rt70', '032gugm1jettifile_rt71', '032gugm1jettifile_rt72', '032gugm1jettifile_tblmineralogymodalsugm1_%', '032gugm1jettifile_tblrt65', '032gugm1jettifile_tblrt66', '032gugm1jettifile_tblrt67', '032gugm1jettifile_tblrt68', '032gugm1jettifile_tblrt69', '032gugm1jettifile_tblrt70', '032gugm1jettifile_tblrt71', '032gugm1jettifile_tblrt72', '032gugm1jettifile_ugm1', '032gugm1jettifile_ugm1_rt65', '032gugm1jettifile_ugm1_rt66', '032gugm1jettifile_ugm1_rt67', '032gugm1jettifile_ugm1_rt68', '032gugm1jettifile_ugm1_rt69', '032gugm1jettifile_ugm1_rt70', '032gugm1jettifile_ugm1_rt71', '032gugm1jettifile_ugm1_rt72', '032gugm1jettifile_ugm1average', '032gugm1jettifile_ugm1dup'],
    '_032hugm4jettifile': ['032hugm4jettifile_acsummary_ugm4', '032hugm4jettifile_rt73', '032hugm4jettifile_rt74', '032hugm4jettifile_rt75', '032hugm4jettifile_rt76', '032hugm4jettifile_rt77', '032hugm4jettifile_rt78', '032hugm4jettifile_rt79', '032hugm4jettifile_rt80', '032hugm4jettifile_tblmineralogymodalsugm4_%', '032hugm4jettifile_tblrt73', '032hugm4jettifile_tblrt74', '032hugm4jettifile_tblrt75', '032hugm4jettifile_tblrt76', '032hugm4jettifile_tblrt77', '032hugm4jettifile_tblrt78', '032hugm4jettifile_tblrt79', '032hugm4jettifile_tblrt80', '032hugm4jettifile_ugm4', '032hugm4jettifile_ugm4_rt73', '032hugm4jettifile_ugm4_rt74', '032hugm4jettifile_ugm4_rt75', '032hugm4jettifile_ugm4_rt76', '032hugm4jettifile_ugm4_rt77', '032hugm4jettifile_ugm4_rt78', '032hugm4jettifile_ugm4_rt79', '032hugm4jettifile_ugm4_rt80', '032hugm4jettifile_ugm4average', '032hugm4jettifile_ugm4dup', '032hugm4jettifile_acsummary_ugm4', '032hugm4jettifile_rt73', '032hugm4jettifile_rt74', '032hugm4jettifile_rt75', '032hugm4jettifile_rt76', '032hugm4jettifile_rt77', '032hugm4jettifile_rt78', '032hugm4jettifile_rt79', '032hugm4jettifile_rt80', '032hugm4jettifile_tblmineralogymodalsugm4_%', '032hugm4jettifile_tblrt73', '032hugm4jettifile_tblrt74', '032hugm4jettifile_tblrt75', '032hugm4jettifile_tblrt76', '032hugm4jettifile_tblrt77', '032hugm4jettifile_tblrt78', '032hugm4jettifile_tblrt79', '032hugm4jettifile_tblrt80', '032hugm4jettifile_ugm4', '032hugm4jettifile_ugm4_rt73', '032hugm4jettifile_ugm4_rt74', '032hugm4jettifile_ugm4_rt75', '032hugm4jettifile_ugm4_rt76', '032hugm4jettifile_ugm4_rt77', '032hugm4jettifile_ugm4_rt78', '032hugm4jettifile_ugm4_rt79', '032hugm4jettifile_ugm4_rt80', '032hugm4jettifile_ugm4average', '032hugm4jettifile_ugm4dup'],
    '_032iandbiojettifile': ['032iandbiojettifile_acsummary_andbio', '032iandbiojettifile_andbio', '032iandbiojettifile_andbio_rt81', '032iandbiojettifile_andbio_rt82', '032iandbiojettifile_andbio_rt83', '032iandbiojettifile_andbio_rt84', '032iandbiojettifile_andbio_rt85', '032iandbiojettifile_andbio_rt86', '032iandbiojettifile_andbio_rt87', '032iandbiojettifile_andbio_rt88', '032iandbiojettifile_andbioaverage', '032iandbiojettifile_andbiodup', '032iandbiojettifile_rt81', '032iandbiojettifile_rt82', '032iandbiojettifile_rt83', '032iandbiojettifile_rt84', '032iandbiojettifile_rt85', '032iandbiojettifile_rt86', '032iandbiojettifile_rt87', '032iandbiojettifile_rt88', '032iandbiojettifile_tblmineralogymodalsandbio_%', '032iandbiojettifile_tblrt81', '032iandbiojettifile_tblrt82', '032iandbiojettifile_tblrt83', '032iandbiojettifile_tblrt84', '032iandbiojettifile_tblrt85', '032iandbiojettifile_tblrt86', '032iandbiojettifile_tblrt87', '032iandbiojettifile_tblrt88', '032iandbiojettifile_acsummary_andbio', '032iandbiojettifile_andbio', '032iandbiojettifile_andbio_rt81', '032iandbiojettifile_andbio_rt82', '032iandbiojettifile_andbio_rt83', '032iandbiojettifile_andbio_rt84', '032iandbiojettifile_andbio_rt85', '032iandbiojettifile_andbio_rt86', '032iandbiojettifile_andbio_rt87', '032iandbiojettifile_andbio_rt88', '032iandbiojettifile_andbioaverage', '032iandbiojettifile_andbiodup', '032iandbiojettifile_rt81', '032iandbiojettifile_rt82', '032iandbiojettifile_rt83', '032iandbiojettifile_rt84', '032iandbiojettifile_rt85', '032iandbiojettifile_rt86', '032iandbiojettifile_rt87', '032iandbiojettifile_rt88', '032iandbiojettifile_tblmineralogymodalsandbio_%', '032iandbiojettifile_tblrt81', '032iandbiojettifile_tblrt82', '032iandbiojettifile_tblrt83', '032iandbiojettifile_tblrt84', '032iandbiojettifile_tblrt85', '032iandbiojettifile_tblrt86', '032iandbiojettifile_tblrt87', '032iandbiojettifile_tblrt88'],
    '_032jugm3jettifile': ['032jugm3jettifile_acsummary_ugm3', '032jugm3jettifile_rt89', '032jugm3jettifile_rt90', '032jugm3jettifile_rt91', '032jugm3jettifile_rt92', '032jugm3jettifile_rt93', '032jugm3jettifile_rt94', '032jugm3jettifile_rt95', '032jugm3jettifile_rt96', '032jugm3jettifile_tblmineralogymodalsugm3_%', '032jugm3jettifile_tblrt89', '032jugm3jettifile_tblrt90', '032jugm3jettifile_tblrt91', '032jugm3jettifile_tblrt92', '032jugm3jettifile_tblrt93', '032jugm3jettifile_tblrt94', '032jugm3jettifile_tblrt95', '032jugm3jettifile_tblrt96', '032jugm3jettifile_ugm3', '032jugm3jettifile_ugm3_rt89', '032jugm3jettifile_ugm3_rt90', '032jugm3jettifile_ugm3_rt91', '032jugm3jettifile_ugm3_rt92', '032jugm3jettifile_ugm3_rt93', '032jugm3jettifile_ugm3_rt94', '032jugm3jettifile_ugm3_rt95', '032jugm3jettifile_ugm3_rt96', '032jugm3jettifile_ugm3average', '032jugm3jettifile_ugm3dup', '032jugm3jettifile_acsummary_ugm3', '032jugm3jettifile_rt89', '032jugm3jettifile_rt90', '032jugm3jettifile_rt91', '032jugm3jettifile_rt92', '032jugm3jettifile_rt93', '032jugm3jettifile_rt94', '032jugm3jettifile_rt95', '032jugm3jettifile_rt96', '032jugm3jettifile_tblmineralogymodalsugm3_%', '032jugm3jettifile_tblrt89', '032jugm3jettifile_tblrt90', '032jugm3jettifile_tblrt91', '032jugm3jettifile_tblrt92', '032jugm3jettifile_tblrt93', '032jugm3jettifile_tblrt94', '032jugm3jettifile_tblrt95', '032jugm3jettifile_tblrt96', '032jugm3jettifile_ugm3', '032jugm3jettifile_ugm3_rt89', '032jugm3jettifile_ugm3_rt90', '032jugm3jettifile_ugm3_rt91', '032jugm3jettifile_ugm3_rt92', '032jugm3jettifile_ugm3_rt93', '032jugm3jettifile_ugm3_rt94', '032jugm3jettifile_ugm3_rt95', '032jugm3jettifile_ugm3_rt96', '032jugm3jettifile_ugm3average', '032jugm3jettifile_ugm3dup'],
    'cha_337____026jettiprojectfile': ['cha_337____026jettiprojectfile_sample3secondarysulfide\nName:_project_sample_id_raw,_dtype:_object'],
    'jettifile': ['jettifileelephantiiver2pq_acsummarynew_pq', 'jettifileelephantiiver2pq_crushed_tblelephantmodals_%', 'jettifileelephantiiver2pq_pqaverage', 'jettifileelephantiiver2pq_pqcch', 'jettifileelephantiiver2pq_pqroma', 'jettifileelephantiiver2pq_pqromb', 'jettifileelephantiiver2pq_rom_tblelephantmodals_%', 'jettifileelephantiiver2ugm_acsummarynew_ugm2', 'jettifileelephantiiver2ugm_aqglobalaugm4', 'jettifileelephantiiver2ugm_aqglobalbugm4', 'jettifileelephantiiver2ugm_tblmineralogymodals_%', 'jettifileelephantiiver2ugm_ugmaverage'],
    'jettiprojectfile': ['jettiprojectfileelephantscl_acsummary_sampleescondida', 'jettiprojectfileelephantscl_columna42', 'jettiprojectfileelephantscl_columna43', 'jettiprojectfileelephantscl_columna52', 'jettiprojectfileelephantscl_columna53', 'jettiprojectfileelephantscl_rt1', 'jettiprojectfileelephantscl_rt2', 'jettiprojectfileelephantscl_rt3', 'jettiprojectfileelephantscl_rt4', 'jettiprojectfileelephantscl_rt5', 'jettiprojectfileelephantscl_rt6', 'jettiprojectfileelephantscl_rt7', 'jettiprojectfileelephantscl_rt8', 'jettiprojectfileelephantscl_sampleescondida', 'jettiprojectfileelephantscl_sampleescondida_rt1', 'jettiprojectfileelephantscl_sampleescondida_rt2', 'jettiprojectfileelephantscl_sampleescondida_rt3', 'jettiprojectfileelephantscl_sampleescondida_rt4', 'jettiprojectfileelephantscl_sampleescondida_rt5', 'jettiprojectfileelephantscl_sampleescondida_rt6', 'jettiprojectfileelephantscl_sampleescondida_rt7', 'jettiprojectfileelephantscl_sampleescondida_rt8', 'jettiprojectfileelephantscl_tblmodalsmel_sampleescondida', 'jettiprojectfileelephantscl_tblrt1', 'jettiprojectfileelephantscl_tblrt19', 'jettiprojectfileelephantscl_tblrt2', 'jettiprojectfileelephantscl_tblrt20', 'jettiprojectfileelephantscl_tblrt21', 'jettiprojectfileelephantscl_tblrt22', 'jettiprojectfileelephantscl_tblrt23', 'jettiprojectfileelephantscl_tblrt3', 'jettiprojectfileelephantscl_tblrt4', 'jettiprojectfileelephantscl_tblrt5', 'jettiprojectfileelephantscl_tblrt6', 'jettiprojectfileelephantscl_tblrt7', 'jettiprojectfileelephantscl_tblrt8', 'jettiprojectfileelephantsite_elephant', 'jettiprojectfileelephantsite_tblmineralogymodalsm1_elephanthead', 'jettiprojectfiletigerrom_bt213', 'jettiprojectfiletigerrom_bt214', 'jettiprojectfiletigerrom_bt424', 'jettiprojectfiletigerrom_bt425', 'jettiprojectfiletigerrom_bt426', 'jettiprojectfiletigerrom_bt427', 'jettiprojectfiletigerrom_m1', 'jettiprojectfiletigerrom_m1_bt213', 'jettiprojectfiletigerrom_m1_bt214', 'jettiprojectfiletigerrom_m1acsummary_m1', 'jettiprojectfiletigerrom_m2', 'jettiprojectfiletigerrom_m2_bt424', 'jettiprojectfiletigerrom_m2_bt425', 'jettiprojectfiletigerrom_m2acsummary_m2', 'jettiprojectfiletigerrom_m3', 'jettiprojectfiletigerrom_m3_bt426', 'jettiprojectfiletigerrom_m3_bt427', 'jettiprojectfiletigerrom_m3acsummary_m3', 'jettiprojectfiletigerrom_tblbt213', 'jettiprojectfiletigerrom_tblbt214', 'jettiprojectfiletigerrom_tblbt424', 'jettiprojectfiletigerrom_tblbt425', 'jettiprojectfiletigerrom_tblbt426', 'jettiprojectfiletigerrom_tblbt427', 'jettiprojectfiletigerrom_tblmineralogymodalsm1_m1head', 'jettiprojectfiletigerrom_tblmineralogymodalsm2_m2head', 'jettiprojectfiletigerrom_tblmineralogymodalsm3_m3head', 'jettiprojectfiletoquepalascl_acsummaryantigua_sampleantigua', 'jettiprojectfiletoquepalascl_acsummaryfresca_samplefresca', 'jettiprojectfiletoquepalascl_rt1', 'jettiprojectfiletoquepalascl_rt11', 'jettiprojectfiletoquepalascl_rt12', 'jettiprojectfiletoquepalascl_rt1r', 'jettiprojectfiletoquepalascl_rt2', 'jettiprojectfiletoquepalascl_rt3', 'jettiprojectfiletoquepalascl_rt3r', 'jettiprojectfiletoquepalascl_rt7', 'jettiprojectfiletoquepalascl_rt8', 'jettiprojectfiletoquepalascl_sampleantigua', 'jettiprojectfiletoquepalascl_sampleantigua_rt11', 'jettiprojectfiletoquepalascl_sampleantigua_rt12', 'jettiprojectfiletoquepalascl_samplefresca', 'jettiprojectfiletoquepalascl_samplefresca_rt1', 'jettiprojectfiletoquepalascl_samplefresca_rt11', 'jettiprojectfiletoquepalascl_samplefresca_rt1r', 'jettiprojectfiletoquepalascl_samplefresca_rt2', 'jettiprojectfiletoquepalascl_samplefresca_rt3', 'jettiprojectfiletoquepalascl_samplefresca_rt3r', 'jettiprojectfiletoquepalascl_samplefresca_rt7', 'jettiprojectfiletoquepalascl_samplefresca_rt8', 'jettiprojectfiletoquepalascl_tblmodalstoquepalaantigua_sampleantigua', 'jettiprojectfiletoquepalascl_tblmodalstoquepalafresca_samplefresca', 'jettiprojectfiletoquepalascl_tblrt1', 'jettiprojectfiletoquepalascl_tblrt11', 'jettiprojectfiletoquepalascl_tblrt12', 'jettiprojectfiletoquepalascl_tblrt1r', 'jettiprojectfiletoquepalascl_tblrt2', 'jettiprojectfiletoquepalascl_tblrt3', 'jettiprojectfiletoquepalascl_tblrt3r', 'jettiprojectfiletoquepalascl_tblrt7', 'jettiprojectfiletoquepalascl_tblrt8', 'jettiprojectfilezaldivarscl_acsummary_samplezaldivar', 'jettiprojectfilezaldivarscl_rt1', 'jettiprojectfilezaldivarscl_rt2', 'jettiprojectfilezaldivarscl_samplezaldivar', 'jettiprojectfilezaldivarscl_samplezaldivar_rt1', 'jettiprojectfilezaldivarscl_samplezaldivar_rt2', 'jettiprojectfilezaldivarscl_tblmodalszaldivar_samplezaldivar', 'jettiprojectfilezaldivarscl_tblrt1', 'jettiprojectfilezaldivarscl_tblrt2'],
    'rea_434____026jettiprojectfile': ['rea_434____026jettiprojectfile_sample3secondarysulfide\nName:_project_sample_id_raw,_dtype:_object'],
    '_035jettiprojectfile': ['035jettiprojectfile_rt1', '035jettiprojectfile_rt10', '035jettiprojectfile_rt2', '035jettiprojectfile_rt3', '035jettiprojectfile_rt4', '035jettiprojectfile_rt5', '035jettiprojectfile_rt6', '035jettiprojectfile_rt7', '035jettiprojectfile_rt8', '035jettiprojectfile_rt9', '035jettiprojectfile_sample1alhassar', '035jettiprojectfile_sample1alhassaravg', '035jettiprojectfile_sample1alhassardup', '035jettiprojectfile_sample2', '035jettiprojectfile_sample2_rt1', '035jettiprojectfile_sample2_rt2', '035jettiprojectfile_sample2_rt3', '035jettiprojectfile_sample2_rt4', '035jettiprojectfile_sample2_rt5', '035jettiprojectfile_sample2acsummary_sample2', '035jettiprojectfile_sample2jabalshaybanavg', '035jettiprojectfile_sample3', '035jettiprojectfile_sample3_rt10', '035jettiprojectfile_sample3_rt6', '035jettiprojectfile_sample3_rt7', '035jettiprojectfile_sample3_rt8', '035jettiprojectfile_sample3_rt9', '035jettiprojectfile_sample3acsummary_sample3', '035jettiprojectfile_tblrt1', '035jettiprojectfile_tblrt10', '035jettiprojectfile_tblrt2', '035jettiprojectfile_tblrt3', '035jettiprojectfile_tblrt4', '035jettiprojectfile_tblrt5', '035jettiprojectfile_tblrt6', '035jettiprojectfile_tblrt7', '035jettiprojectfile_tblrt8', '035jettiprojectfile_tblrt9'],
}

# modified dictionary for new project_sample_ids (manual separation looking at jetti project files)

modified_keys_for_project_sample_id = {
    '003jettiprojectfile_amcf': ['003jettiprojectfile_amcf+1', '003jettiprojectfile_amcf+12', '003jettiprojectfile_amcf+14', '003jettiprojectfile_amcf+34', '003jettiprojectfile_amcf+6mesh', '003jettiprojectfile_amcf6mesh', '003jettiprojectfile_amcfhead', '003jettiprojectfile_amcfhead_tblmineralogymodals_%', '003jettiprojectfile_catalyzedbe2residue_tblmineralogymodals_%', '003jettiprojectfile_controlbe1residue_tblmineralogymodals_%', '003jettiprojectfile_pv_rt1', '003jettiprojectfile_pv_rt10', '003jettiprojectfile_pv_rt10r', '003jettiprojectfile_pv_rt11', '003jettiprojectfile_pv_rt2', '003jettiprojectfile_pv_rt3', '003jettiprojectfile_pv_rt4', '003jettiprojectfile_pv_rt5', '003jettiprojectfile_pv_rt6', '003jettiprojectfile_pv_rt7', '003jettiprojectfile_pv_rt8', '003jettiprojectfile_pv_rt9', '003jettiprojectfile_repamcf6mesh', '003jettiprojectfile_rt1', '003jettiprojectfile_rt10', '003jettiprojectfile_rt11', '003jettiprojectfile_rt2', '003jettiprojectfile_rt3', '003jettiprojectfile_rt4', '003jettiprojectfile_rt5', '003jettiprojectfile_rt6', '003jettiprojectfile_rt7', '003jettiprojectfile_rt8', '003jettiprojectfile_rt9', '003jettiprojectfile_tblrt1', '003jettiprojectfile_tblrt10', '003jettiprojectfile_tblrt10r', '003jettiprojectfile_tblrt11', '003jettiprojectfile_tblrt2', '003jettiprojectfile_tblrt3', '003jettiprojectfile_tblrt4', '003jettiprojectfile_tblrt5', '003jettiprojectfile_tblrt6', '003jettiprojectfile_tblrt7', '003jettiprojectfile_tblrt8', '003jettiprojectfile_tblrt9'],
    # review '007ajettiprojectfile_rt73', '007ajettiprojectfile_rt74'
    # '007ajettiprojectfile_elephant_ugm2_colhead': [],
    # '007ajettiprojectfile_elephant_ugm2_rthead': ['007ajettiprojectfile_acugm2_ugm2', '007ajettiprojectfile_hrt1', '007ajettiprojectfile_hrt10', '007ajettiprojectfile_hrt11', '007ajettiprojectfile_hrt12', '007ajettiprojectfile_hrt1r', '007ajettiprojectfile_hrt2', '007ajettiprojectfile_hrt3', '007ajettiprojectfile_hrt4', '007ajettiprojectfile_hrt5', '007ajettiprojectfile_hrt6', '007ajettiprojectfile_hrt7', '007ajettiprojectfile_hrt8', '007ajettiprojectfile_hrt9', '007ajettiprojectfile_rt41', '007ajettiprojectfile_rt41r', '007ajettiprojectfile_rt42', '007ajettiprojectfile_rt43', '007ajettiprojectfile_rt44', '007ajettiprojectfile_rt44r', '007ajettiprojectfile_rt45', '007ajettiprojectfile_rt46', '007ajettiprojectfile_rt53', '007ajettiprojectfile_rt54', '007ajettiprojectfile_rt55', '007ajettiprojectfile_rt56', '007ajettiprojectfile_rt57', '007ajettiprojectfile_rt58', '007ajettiprojectfile_rt65', '007ajettiprojectfile_rt66', '007ajettiprojectfile_rt67', '007ajettiprojectfile_rt68', '007ajettiprojectfile_rt69', '007ajettiprojectfile_rt70', '007ajettiprojectfile_tblhrt1', '007ajettiprojectfile_tblhrt10', '007ajettiprojectfile_tblhrt11', '007ajettiprojectfile_tblhrt12', '007ajettiprojectfile_tblhrt1r', '007ajettiprojectfile_tblhrt2', '007ajettiprojectfile_tblhrt3', '007ajettiprojectfile_tblhrt4', '007ajettiprojectfile_tblhrt5', '007ajettiprojectfile_tblhrt6', '007ajettiprojectfile_tblhrt7', '007ajettiprojectfile_tblhrt8', '007ajettiprojectfile_tblhrt9', '007ajettiprojectfile_tblmineralogymodalsugm2_%', '007ajettiprojectfile_tblrt41', '007ajettiprojectfile_tblrt41r', '007ajettiprojectfile_tblrt42', '007ajettiprojectfile_tblrt43', '007ajettiprojectfile_tblrt44', '007ajettiprojectfile_tblrt44r', '007ajettiprojectfile_tblrt45', '007ajettiprojectfile_tblrt46', '007ajettiprojectfile_tblrt53', '007ajettiprojectfile_tblrt54', '007ajettiprojectfile_tblrt55', '007ajettiprojectfile_tblrt56', '007ajettiprojectfile_tblrt57', '007ajettiprojectfile_tblrt58', '007ajettiprojectfile_tblrt65', '007ajettiprojectfile_tblrt66', '007ajettiprojectfile_tblrt67', '007ajettiprojectfile_tblrt68', '007ajettiprojectfile_tblrt69', '007ajettiprojectfile_tblrt70', '007ajettiprojectfile_tblrt73', '007ajettiprojectfile_tblrt74', '007ajettiprojectfile_ugm2columnheadavg', '007ajettiprojectfile_ugm2reactorsrt41to44heads'],
    # '007ajettiprojectfile_elephant_pq_colhead': [],
    # '007ajettiprojectfile_elephant_pq_rthead': ['007ajettiprojectfile_acpq_pq', '007ajettiprojectfile_pqcolumnheadavg', '007ajettiprojectfile_pqreactorsrt47to50heads', '007ajettiprojectfile_rt47', '007ajettiprojectfile_rt48', '007ajettiprojectfile_rt49', '007ajettiprojectfile_rt50', '007ajettiprojectfile_rt51', '007ajettiprojectfile_rt52', '007ajettiprojectfile_rt59', '007ajettiprojectfile_rt60', '007ajettiprojectfile_rt61', '007ajettiprojectfile_rt63', '007ajettiprojectfile_rt64', '007ajettiprojectfile_rt73', '007ajettiprojectfile_rt74', '007ajettiprojectfile_tblmineralogymodalspq_%', '007ajettiprojectfile_tblrt47', '007ajettiprojectfile_tblrt48', '007ajettiprojectfile_tblrt49', '007ajettiprojectfile_tblrt50', '007ajettiprojectfile_tblrt51', '007ajettiprojectfile_tblrt52', '007ajettiprojectfile_tblrt59', '007ajettiprojectfile_tblrt60', '007ajettiprojectfile_tblrt61', '007ajettiprojectfile_tblrt63', '007ajettiprojectfile_tblrt64', '007ajettiprojectfile_tblrt73', '007ajettiprojectfile_tblrt74'],
    '007ajettiprojectfile_elephant_ugm2_rthead': ['007ajettiprojectfile_ugm2_hrt1', '007ajettiprojectfile_acugm2_ugm2', '007ajettiprojectfile_hrt10', '007ajettiprojectfile_hrt11', '007ajettiprojectfile_hrt12', '007ajettiprojectfile_hrt1r', '007ajettiprojectfile_hrt2', '007ajettiprojectfile_hrt3', '007ajettiprojectfile_hrt4', '007ajettiprojectfile_hrt5', '007ajettiprojectfile_hrt6', '007ajettiprojectfile_hrt7', '007ajettiprojectfile_hrt8', '007ajettiprojectfile_hrt9', '007ajettiprojectfile_rt41', '007ajettiprojectfile_rt41r', '007ajettiprojectfile_rt42', '007ajettiprojectfile_rt43', '007ajettiprojectfile_rt44', '007ajettiprojectfile_rt44r', '007ajettiprojectfile_rt45', '007ajettiprojectfile_rt46', '007ajettiprojectfile_rt53', '007ajettiprojectfile_rt54', '007ajettiprojectfile_rt55', '007ajettiprojectfile_rt56', '007ajettiprojectfile_rt57', '007ajettiprojectfile_rt58', '007ajettiprojectfile_rt65', '007ajettiprojectfile_rt66', '007ajettiprojectfile_rt67', '007ajettiprojectfile_rt68', '007ajettiprojectfile_rt69', '007ajettiprojectfile_rt70', '007ajettiprojectfile_tblhrt10', '007ajettiprojectfile_tblhrt11', '007ajettiprojectfile_tblhrt12', '007ajettiprojectfile_tblhrt1r', '007ajettiprojectfile_tblhrt2', '007ajettiprojectfile_tblhrt3', '007ajettiprojectfile_tblhrt4', '007ajettiprojectfile_tblhrt5', '007ajettiprojectfile_tblhrt6', '007ajettiprojectfile_tblhrt7', '007ajettiprojectfile_tblhrt8', '007ajettiprojectfile_tblhrt9', '007ajettiprojectfile_tblmineralogymodalsugm2_%', '007ajettiprojectfile_tblrt41', '007ajettiprojectfile_tblrt41r', '007ajettiprojectfile_tblrt42', '007ajettiprojectfile_tblrt43', '007ajettiprojectfile_tblrt44', '007ajettiprojectfile_tblrt44r', '007ajettiprojectfile_tblrt45', '007ajettiprojectfile_tblrt46', '007ajettiprojectfile_tblrt53', '007ajettiprojectfile_tblrt54', '007ajettiprojectfile_tblrt55', '007ajettiprojectfile_tblrt56', '007ajettiprojectfile_tblrt57', '007ajettiprojectfile_tblrt58', '007ajettiprojectfile_tblrt65', '007ajettiprojectfile_tblrt66', '007ajettiprojectfile_tblrt67', '007ajettiprojectfile_tblrt68', '007ajettiprojectfile_tblrt69', '007ajettiprojectfile_tblrt70', '007ajettiprojectfile_ugm2_hrt10', '007ajettiprojectfile_ugm2_hrt11', '007ajettiprojectfile_ugm2_hrt12', '007ajettiprojectfile_ugm2_hrt1r', '007ajettiprojectfile_ugm2_hrt2', '007ajettiprojectfile_ugm2_hrt3', '007ajettiprojectfile_ugm2_hrt4', '007ajettiprojectfile_ugm2_hrt5', '007ajettiprojectfile_ugm2_hrt6', '007ajettiprojectfile_ugm2_hrt7', '007ajettiprojectfile_ugm2_hrt8', '007ajettiprojectfile_ugm2_hrt9', '007ajettiprojectfile_ugm2_rt41', '007ajettiprojectfile_ugm2_rt41r', '007ajettiprojectfile_ugm2_rt42', '007ajettiprojectfile_ugm2_rt43', '007ajettiprojectfile_ugm2_rt44', '007ajettiprojectfile_ugm2_rt44r', '007ajettiprojectfile_ugm2_rt45', '007ajettiprojectfile_ugm2_rt46', '007ajettiprojectfile_ugm2_rt53', '007ajettiprojectfile_ugm2_rt54', '007ajettiprojectfile_ugm2_rt55', '007ajettiprojectfile_ugm2_rt56', '007ajettiprojectfile_ugm2_rt57', '007ajettiprojectfile_ugm2_rt58', '007ajettiprojectfile_ugm2_rt65', '007ajettiprojectfile_ugm2_rt66', '007ajettiprojectfile_ugm2_rt67', '007ajettiprojectfile_ugm2_rt68', '007ajettiprojectfile_ugm2_rt69', '007ajettiprojectfile_ugm2_rt70', '007ajettiprojectfile_ugm2columnheadavg', '007ajettiprojectfile_ugm2reactorsrt41to44heads'],
    '007ajettiprojectfile_elephant_pq_rthead': ['007ajettiprojectfile_pq_rt62', '007ajettiprojectfile_pq_rt62r', '007ajettiprojectfile_acpq_pq', '007ajettiprojectfile_pq_rt47', '007ajettiprojectfile_pq_rt48', '007ajettiprojectfile_pq_rt49', '007ajettiprojectfile_pq_rt50', '007ajettiprojectfile_pq_rt51', '007ajettiprojectfile_pq_rt52', '007ajettiprojectfile_pq_rt59', '007ajettiprojectfile_pq_rt60', '007ajettiprojectfile_pq_rt61', '007ajettiprojectfile_pq_rt63', '007ajettiprojectfile_pq_rt64', '007ajettiprojectfile_pq_rt73', '007ajettiprojectfile_pq_rt74', '007ajettiprojectfile_pqcolumnheadavg', '007ajettiprojectfile_pqreactorsrt47to50heads', '007ajettiprojectfile_rt47', '007ajettiprojectfile_rt48', '007ajettiprojectfile_rt49', '007ajettiprojectfile_rt50', '007ajettiprojectfile_rt51', '007ajettiprojectfile_rt52', '007ajettiprojectfile_rt59', '007ajettiprojectfile_rt60', '007ajettiprojectfile_rt61', '007ajettiprojectfile_rt63', '007ajettiprojectfile_rt64', '007ajettiprojectfile_rt73', '007ajettiprojectfile_rt74', '007ajettiprojectfile_tblmineralogymodalspq_%', '007ajettiprojectfile_tblrt47', '007ajettiprojectfile_tblrt48', '007ajettiprojectfile_tblrt49', '007ajettiprojectfile_tblrt50', '007ajettiprojectfile_tblrt51', '007ajettiprojectfile_tblrt52', '007ajettiprojectfile_tblrt59', '007ajettiprojectfile_tblrt60', '007ajettiprojectfile_tblrt61', '007ajettiprojectfile_tblrt63', '007ajettiprojectfile_tblrt64', '007ajettiprojectfile_tblrt73', '007ajettiprojectfile_tblrt74'],
    # excluded from 007a : RT1, RT62 y RT62R
    '007ajettiprojectfile_elephant_raffshipment': ['007ajettiprojectfile_elephant2raffshipment26jul2024avg'],
    
    '007bjettiprojectfile_tiger_m1': ['007bjettiprojectfiletiger_6925domain7', '007bjettiprojectfiletiger_6925domain7dup', '007bjettiprojectfiletiger_6925doman7avg', '007bjettiprojectfiletiger_acsummary_tiger', '007bjettiprojectfiletiger_domain7_rt10', '007bjettiprojectfiletiger_domain7_rt11', '007bjettiprojectfiletiger_domain7_rt12', '007bjettiprojectfiletiger_domain7_rt9', '007bjettiprojectfiletiger_rt1', '007bjettiprojectfiletiger_rt10', '007bjettiprojectfiletiger_rt11', '007bjettiprojectfiletiger_rt12', '007bjettiprojectfiletiger_rt2', '007bjettiprojectfiletiger_rt3', '007bjettiprojectfiletiger_rt4', '007bjettiprojectfiletiger_rt5', '007bjettiprojectfiletiger_rt6', '007bjettiprojectfiletiger_rt7', '007bjettiprojectfiletiger_rt8', '007bjettiprojectfiletiger_rt9', '007bjettiprojectfiletiger_tblmineralcomposition_%', '007bjettiprojectfiletiger_tblrt1', '007bjettiprojectfiletiger_tblrt10', '007bjettiprojectfiletiger_tblrt11', '007bjettiprojectfiletiger_tblrt12', '007bjettiprojectfiletiger_tblrt2', '007bjettiprojectfiletiger_tblrt3', '007bjettiprojectfiletiger_tblrt4', '007bjettiprojectfiletiger_tblrt5', '007bjettiprojectfiletiger_tblrt6', '007bjettiprojectfiletiger_tblrt7', '007bjettiprojectfiletiger_tblrt8', '007bjettiprojectfiletiger_tblrt9', '007bjettiprojectfiletiger_tiger_rt1', '007bjettiprojectfiletiger_tiger_rt2', '007bjettiprojectfiletiger_tiger_rt3', '007bjettiprojectfiletiger_tiger_rt4', '007bjettiprojectfiletiger_tiger_rt5', '007bjettiprojectfiletiger_tiger_rt6', '007bjettiprojectfiletiger_tiger_rt7', '007bjettiprojectfiletiger_tiger_rt8', '007bjettiprojectfiletiger_tigerhead', '007bjettiprojectfiletiger_tigersample+12inch', '007bjettiprojectfiletiger_tigersample+14inch', '007bjettiprojectfiletiger_tigersample+34inch', '007bjettiprojectfiletiger_tigersample+6mesh', '007bjettiprojectfiletiger_tigersample6mesh'],
    '007bjettiprojectfile_tiger_m2': ['007bjettiprojectfiletiger_6925domain7', '007bjettiprojectfiletiger_6925domain7dup', '007bjettiprojectfiletiger_6925doman7avg', '007bjettiprojectfiletiger_acsummary_tiger', '007bjettiprojectfiletiger_domain7_rt10', '007bjettiprojectfiletiger_domain7_rt11', '007bjettiprojectfiletiger_domain7_rt12', '007bjettiprojectfiletiger_domain7_rt9', '007bjettiprojectfiletiger_rt1', '007bjettiprojectfiletiger_rt10', '007bjettiprojectfiletiger_rt11', '007bjettiprojectfiletiger_rt12', '007bjettiprojectfiletiger_rt2', '007bjettiprojectfiletiger_rt3', '007bjettiprojectfiletiger_rt4', '007bjettiprojectfiletiger_rt5', '007bjettiprojectfiletiger_rt6', '007bjettiprojectfiletiger_rt7', '007bjettiprojectfiletiger_rt8', '007bjettiprojectfiletiger_rt9', '007bjettiprojectfiletiger_tblmineralcomposition_%', '007bjettiprojectfiletiger_tblrt1', '007bjettiprojectfiletiger_tblrt10', '007bjettiprojectfiletiger_tblrt11', '007bjettiprojectfiletiger_tblrt12', '007bjettiprojectfiletiger_tblrt2', '007bjettiprojectfiletiger_tblrt3', '007bjettiprojectfiletiger_tblrt4', '007bjettiprojectfiletiger_tblrt5', '007bjettiprojectfiletiger_tblrt6', '007bjettiprojectfiletiger_tblrt7', '007bjettiprojectfiletiger_tblrt8', '007bjettiprojectfiletiger_tblrt9', '007bjettiprojectfiletiger_tiger_rt1', '007bjettiprojectfiletiger_tiger_rt2', '007bjettiprojectfiletiger_tiger_rt3', '007bjettiprojectfiletiger_tiger_rt4', '007bjettiprojectfiletiger_tiger_rt5', '007bjettiprojectfiletiger_tiger_rt6', '007bjettiprojectfiletiger_tiger_rt7', '007bjettiprojectfiletiger_tiger_rt8', '007bjettiprojectfiletiger_tigerhead', '007bjettiprojectfiletiger_tigersample+12inch', '007bjettiprojectfiletiger_tigersample+14inch', '007bjettiprojectfiletiger_tigersample+34inch', '007bjettiprojectfiletiger_tigersample+6mesh', '007bjettiprojectfiletiger_tigersample6mesh'],
    '007bjettiprojectfile_tiger_m3': ['007bjettiprojectfiletiger_6925domain7', '007bjettiprojectfiletiger_6925domain7dup', '007bjettiprojectfiletiger_6925doman7avg', '007bjettiprojectfiletiger_acsummary_tiger', '007bjettiprojectfiletiger_domain7_rt10', '007bjettiprojectfiletiger_domain7_rt11', '007bjettiprojectfiletiger_domain7_rt12', '007bjettiprojectfiletiger_domain7_rt9', '007bjettiprojectfiletiger_rt1', '007bjettiprojectfiletiger_rt10', '007bjettiprojectfiletiger_rt11', '007bjettiprojectfiletiger_rt12', '007bjettiprojectfiletiger_rt2', '007bjettiprojectfiletiger_rt3', '007bjettiprojectfiletiger_rt4', '007bjettiprojectfiletiger_rt5', '007bjettiprojectfiletiger_rt6', '007bjettiprojectfiletiger_rt7', '007bjettiprojectfiletiger_rt8', '007bjettiprojectfiletiger_rt9', '007bjettiprojectfiletiger_tblmineralcomposition_%', '007bjettiprojectfiletiger_tblrt1', '007bjettiprojectfiletiger_tblrt10', '007bjettiprojectfiletiger_tblrt11', '007bjettiprojectfiletiger_tblrt12', '007bjettiprojectfiletiger_tblrt2', '007bjettiprojectfiletiger_tblrt3', '007bjettiprojectfiletiger_tblrt4', '007bjettiprojectfiletiger_tblrt5', '007bjettiprojectfiletiger_tblrt6', '007bjettiprojectfiletiger_tblrt7', '007bjettiprojectfiletiger_tblrt8', '007bjettiprojectfiletiger_tblrt9', '007bjettiprojectfiletiger_tiger_rt1', '007bjettiprojectfiletiger_tiger_rt2', '007bjettiprojectfiletiger_tiger_rt3', '007bjettiprojectfiletiger_tiger_rt4', '007bjettiprojectfiletiger_tiger_rt5', '007bjettiprojectfiletiger_tiger_rt6', '007bjettiprojectfiletiger_tiger_rt7', '007bjettiprojectfiletiger_tiger_rt8', '007bjettiprojectfiletiger_tigerhead', '007bjettiprojectfiletiger_tigersample+12inch', '007bjettiprojectfiletiger_tigersample+14inch', '007bjettiprojectfiletiger_tigersample+34inch', '007bjettiprojectfiletiger_tigersample+6mesh', '007bjettiprojectfiletiger_tigersample6mesh'],
    '007jettiprojectfile_leopard': ['007jettiprojectfileleopard_007rt1', '007jettiprojectfileleopard_007rt2', '007jettiprojectfileleopard_007rt2r', '007jettiprojectfileleopard_007rt3', '007jettiprojectfileleopard_007rt37', '007jettiprojectfileleopard_007rt38', '007jettiprojectfileleopard_007rt39', '007jettiprojectfileleopard_007rt3r', '007jettiprojectfileleopard_007rt4', '007jettiprojectfileleopard_007rt40', '007jettiprojectfileleopard_007rt4r', '007jettiprojectfileleopard_007rt5', '007jettiprojectfileleopard_007rt5r', '007jettiprojectfileleopard_007rt6', '007jettiprojectfileleopard_007rt6r', '007jettiprojectfileleopard_007rt7', '007jettiprojectfileleopard_007rt7r', '007jettiprojectfileleopard_007rt8', '007jettiprojectfileleopard_acleopard_leopard', '007jettiprojectfileleopard_leopard', '007jettiprojectfileleopard_leopard_rt1', '007jettiprojectfileleopard_leopard_rt2', '007jettiprojectfileleopard_leopard_rt2r', '007jettiprojectfileleopard_leopard_rt3', '007jettiprojectfileleopard_leopard_rt37', '007jettiprojectfileleopard_leopard_rt38', '007jettiprojectfileleopard_leopard_rt39', '007jettiprojectfileleopard_leopard_rt3r', '007jettiprojectfileleopard_leopard_rt4', '007jettiprojectfileleopard_leopard_rt40', '007jettiprojectfileleopard_leopard_rt4r', '007jettiprojectfileleopard_leoparddup', '007jettiprojectfileleopard_tblmineralogymodalslep_%', '007jettiprojectfileleopard_tblrt1', '007jettiprojectfileleopard_tblrt2', '007jettiprojectfileleopard_tblrt2r', '007jettiprojectfileleopard_tblrt3', '007jettiprojectfileleopard_tblrt37', '007jettiprojectfileleopard_tblrt38', '007jettiprojectfileleopard_tblrt39', '007jettiprojectfileleopard_tblrt3r', '007jettiprojectfileleopard_tblrt4', '007jettiprojectfileleopard_tblrt40', '007jettiprojectfileleopard_tblrt4r'],
    '007jettiprojectfile_elephant': ['007jettiprojectfile_elephant', '007jettiprojectfileelephantleopardtiger_007rt5', '007jettiprojectfileelephantleopardtiger_007rt5r', '007jettiprojectfileelephantleopardtiger_007rt6', '007jettiprojectfileelephantleopardtiger_007rt6r', '007jettiprojectfileelephantleopardtiger_007rt7', '007jettiprojectfileelephantleopardtiger_007rt7r', '007jettiprojectfileelephantleopardtiger_007rt8', '007jettiprojectfileelephantleopardtiger_acelephant', '007jettiprojectfileelephantleopardtiger_acelephant_elephant', '007jettiprojectfileelephantleopardtiger_elephant', '007jettiprojectfileelephantleopardtiger_elephant_rt5', '007jettiprojectfileelephantleopardtiger_elephant_rt5r', '007jettiprojectfileelephantleopardtiger_elephant_rt6', '007jettiprojectfileelephantleopardtiger_elephant_rt6r', '007jettiprojectfileelephantleopardtiger_elephant_rt7', '007jettiprojectfileelephantleopardtiger_elephant_rt7r', '007jettiprojectfileelephantleopardtiger_elephant_rt8', '007jettiprojectfileelephantleopardtiger_elephantdup', '007jettiprojectfileelephantleopardtiger_tblrt5', '007jettiprojectfileelephantleopardtiger_tblrt5r', '007jettiprojectfileelephantleopardtiger_tblrt6', '007jettiprojectfileelephantleopardtiger_tblrt6r', '007jettiprojectfileelephantleopardtiger_tblrt7', '007jettiprojectfileelephantleopardtiger_tblrt7r', '007jettiprojectfileelephantleopardtiger_tblrt8'],
    # '007jettiprojectfile_tiger_m1': ['007jettiprojectfileelephantleopardtiger_007rt10', '007jettiprojectfileelephantleopardtiger_007rt11', '007jettiprojectfileelephantleopardtiger_007rt12', '007jettiprojectfileelephantleopardtiger_007rt9', '007jettiprojectfileelephantleopardtiger_tblrt10', '007jettiprojectfileelephantleopardtiger_tblrt11', '007jettiprojectfileelephantleopardtiger_tblrt12', '007jettiprojectfileelephantleopardtiger_tblrt9', '007jettiprojectfileelephantleopardtiger_tigerm1007rt9101112', '007jettiprojectfileelephantleopardtiger_tigerm1_rt10', '007jettiprojectfileelephantleopardtiger_tigerm1_rt11', '007jettiprojectfileelephantleopardtiger_tigerm1_rt12', '007jettiprojectfileelephantleopardtiger_tigerm1_rt9', '007jettiprojectfileelephantleopardtiger_tigerm1dup'],
    # '007jettiprojectfile_tiger_m2': ['007jettiprojectfileelephantleopardtiger_007rt13', '007jettiprojectfileelephantleopardtiger_007rt14', '007jettiprojectfileelephantleopardtiger_007rt15', '007jettiprojectfileelephantleopardtiger_007rt16', '007jettiprojectfileelephantleopardtiger_tblrt13', '007jettiprojectfileelephantleopardtiger_tblrt14', '007jettiprojectfileelephantleopardtiger_tblrt15', '007jettiprojectfileelephantleopardtiger_tblrt16', '007jettiprojectfileelephantleopardtiger_tigerm2007rt13141516', '007jettiprojectfileelephantleopardtiger_tigerm2_rt13', '007jettiprojectfileelephantleopardtiger_tigerm2_rt14', '007jettiprojectfileelephantleopardtiger_tigerm2_rt15', '007jettiprojectfileelephantleopardtiger_tigerm2_rt16', '007jettiprojectfileelephantleopardtiger_tigerm2dup'],
    # '007jettiprojectfile_tiger_m3': ['007jettiprojectfileelephantleopardtiger_007rt17', '007jettiprojectfileelephantleopardtiger_007rt18', '007jettiprojectfileelephantleopardtiger_007rt19', '007jettiprojectfileelephantleopardtiger_007rt20', '007jettiprojectfileelephantleopardtiger_tblrt17', '007jettiprojectfileelephantleopardtiger_tblrt18', '007jettiprojectfileelephantleopardtiger_tblrt19', '007jettiprojectfileelephantleopardtiger_tblrt20', '007jettiprojectfileelephantleopardtiger_tigerm3007rt17181920', '007jettiprojectfileelephantleopardtiger_tigerm3_rt17', '007jettiprojectfileelephantleopardtiger_tigerm3_rt18', '007jettiprojectfileelephantleopardtiger_tigerm3_rt19', '007jettiprojectfileelephantleopardtiger_tigerm3_rt20', '007jettiprojectfileelephantleopardtiger_tigerm3dup'],
    '007jettiprojectfile_rtm2': ['007jettiprojectfilertm2', '007jettiprojectfilertm2_tblmineralogymodalsrtm2_%', '007jettiprojectfilertm2_acsummary', '007jettiprojectfilertm2_acsummary_rtm2', '007jettiprojectfilertm2_rt1forcheckvalues', '007jettiprojectfilertm2_rt33', '007jettiprojectfilertm2_rt34', '007jettiprojectfilertm2_rt35', '007jettiprojectfilertm2_rt36', '007jettiprojectfilertm2_rtm2_rt33', '007jettiprojectfilertm2_rtm2_rt34', '007jettiprojectfilertm2_rtm2_rt35', '007jettiprojectfilertm2_rtm2_rt36', '007jettiprojectfilertm2_rtm2a', '007jettiprojectfilertm2_rtm2b', '007jettiprojectfilertm2_tblrt1d', '007jettiprojectfilertm2_tblrt33', '007jettiprojectfilertm2_tblrt34', '007jettiprojectfilertm2_tblrt35', '007jettiprojectfilertm2_tblrt36'],
    '007jettiprojectfile_toquepala_antigua': ['007jettiprojectfiletoquepala_acsummaryantigua', '007jettiprojectfiletoquepala_acsummaryantigua_antigua', '007jettiprojectfiletoquepala_rt25', '007jettiprojectfiletoquepala_rt26', '007jettiprojectfiletoquepala_rt27', '007jettiprojectfiletoquepala_rt28', '007jettiprojectfiletoquepala_tblmineralogymodalstoquepalaantigua_%', '007jettiprojectfiletoquepala_tblrt25', '007jettiprojectfiletoquepala_tblrt26', '007jettiprojectfiletoquepala_tblrt27', '007jettiprojectfiletoquepala_tblrt28', '007jettiprojectfiletoquepala_toquepalaantigua_rt25', '007jettiprojectfiletoquepala_toquepalaantigua_rt26', '007jettiprojectfiletoquepala_toquepalaantigua_rt27', '007jettiprojectfiletoquepala_toquepalaantigua_rt28', '007jettiprojectfiletoquepala_toquepalaantiguaa', '007jettiprojectfiletoquepala_toquepalaantiguab', '007jettiprojectfiletoquepala_toquepalaantiguasxs+10mesh', '007jettiprojectfiletoquepala_toquepalaantiguasxs+12inch', '007jettiprojectfiletoquepala_toquepalaantiguasxs+14inch', '007jettiprojectfiletoquepala_toquepalaantiguasxs+34inch', '007jettiprojectfiletoquepala_toquepalaantiguasxs10mesh'],
    '007jettiprojectfile_toquepala_fresca': ['007jettiprojectfiletoquepala_acsummaryfresca', '007jettiprojectfiletoquepala_acsummaryfresca_fresca', '007jettiprojectfiletoquepala_rt21', '007jettiprojectfiletoquepala_rt22', '007jettiprojectfiletoquepala_rt23', '007jettiprojectfiletoquepala_rt24', '007jettiprojectfiletoquepala_tblmineralogymodalstoquepalafrecsa_%', '007jettiprojectfiletoquepala_tblrt21', '007jettiprojectfiletoquepala_tblrt22', '007jettiprojectfiletoquepala_tblrt23', '007jettiprojectfiletoquepala_tblrt24', '007jettiprojectfiletoquepala_toquepalafresca_rt21', '007jettiprojectfiletoquepala_toquepalafresca_rt22', '007jettiprojectfiletoquepala_toquepalafresca_rt23', '007jettiprojectfiletoquepala_toquepalafresca_rt24', '007jettiprojectfiletoquepala_toquepalafrescaa', '007jettiprojectfiletoquepala_toquepalafrescab'],    
    '007jettiprojectfile_zaldivar': ['007jettiprojectfilezaldivar', '007jettiprojectfilezaldivar_tblmineralogymodalszaldivar_%', '007jettiprojectfilezaldivar_acsummary', '007jettiprojectfilezaldivar_acsummary_zaldivar', '007jettiprojectfilezaldivar_rt29', '007jettiprojectfilezaldivar_rt30', '007jettiprojectfilezaldivar_rt31', '007jettiprojectfilezaldivar_rt32', '007jettiprojectfilezaldivar_tblrt29', '007jettiprojectfilezaldivar_tblrt30', '007jettiprojectfilezaldivar_tblrt31', '007jettiprojectfilezaldivar_tblrt32', '007jettiprojectfilezaldivar_zaldivar_rt29', '007jettiprojectfilezaldivar_zaldivar_rt30', '007jettiprojectfilezaldivar_zaldivar_rt31', '007jettiprojectfilezaldivar_zaldivar_rt32', '007jettiprojectfilezaldivar_zaldivarcpya', '007jettiprojectfilezaldivar_zaldivarcpyb', '007jettiprojectfilezaldivar_zaldivarcpysxs+10mesh', '007jettiprojectfilezaldivar_zaldivarcpysxs+12inch', '007jettiprojectfilezaldivar_zaldivarcpysxs+14inch', '007jettiprojectfilezaldivar_zaldivarcpysxs10mesh'],
    '007bjettiprojectfiletiger_domain7': ['007bjettiprojectfiletiger_domain7_rt9', '007bjettiprojectfiletiger_domain7_rt10', '007bjettiprojectfiletiger_domain7_rt11', '007bjettiprojectfiletiger_domain7_rt12'],
    # 'original_007jettiprojectfile': ['007jettiprojectfileelephantleopardtiger_007rt1', '007jettiprojectfileelephantleopardtiger_007rt10', '007jettiprojectfileelephantleopardtiger_007rt11', '007jettiprojectfileelephantleopardtiger_007rt12', '007jettiprojectfileelephantleopardtiger_007rt13', '007jettiprojectfileelephantleopardtiger_007rt14', '007jettiprojectfileelephantleopardtiger_007rt15', '007jettiprojectfileelephantleopardtiger_007rt16', '007jettiprojectfileelephantleopardtiger_007rt17', '007jettiprojectfileelephantleopardtiger_007rt18', '007jettiprojectfileelephantleopardtiger_007rt19', '007jettiprojectfileelephantleopardtiger_007rt2', '007jettiprojectfileelephantleopardtiger_007rt20', '007jettiprojectfileelephantleopardtiger_007rt2r', '007jettiprojectfileelephantleopardtiger_007rt3', '007jettiprojectfileelephantleopardtiger_007rt3r', '007jettiprojectfileelephantleopardtiger_007rt4', '007jettiprojectfileelephantleopardtiger_007rt4r', '007jettiprojectfileelephantleopardtiger_007rt5', '007jettiprojectfileelephantleopardtiger_007rt5r', '007jettiprojectfileelephantleopardtiger_007rt6', '007jettiprojectfileelephantleopardtiger_007rt6r', '007jettiprojectfileelephantleopardtiger_007rt7', '007jettiprojectfileelephantleopardtiger_007rt7r', '007jettiprojectfileelephantleopardtiger_007rt8', '007jettiprojectfileelephantleopardtiger_007rt9', '007jettiprojectfileelephantleopardtiger_acelephant', '007jettiprojectfileelephantleopardtiger_acelephant_elephant', '007jettiprojectfileelephantleopardtiger_acleopard', '007jettiprojectfileelephantleopardtiger_acleopard_leopard', '007jettiprojectfileelephantleopardtiger_elephant', '007jettiprojectfileelephantleopardtiger_elephant_rt5', '007jettiprojectfileelephantleopardtiger_elephant_rt5r', '007jettiprojectfileelephantleopardtiger_elephant_rt6', '007jettiprojectfileelephantleopardtiger_elephant_rt6r', '007jettiprojectfileelephantleopardtiger_elephant_rt7', '007jettiprojectfileelephantleopardtiger_elephant_rt7r', '007jettiprojectfileelephantleopardtiger_elephant_rt8', '007jettiprojectfileelephantleopardtiger_elephantdup', '007jettiprojectfileelephantleopardtiger_leopard', '007jettiprojectfileelephantleopardtiger_leopard_rt1', '007jettiprojectfileelephantleopardtiger_leopard_rt2', '007jettiprojectfileelephantleopardtiger_leopard_rt2r', '007jettiprojectfileelephantleopardtiger_leopard_rt3', '007jettiprojectfileelephantleopardtiger_leopard_rt37', '007jettiprojectfileelephantleopardtiger_leopard_rt38', '007jettiprojectfileelephantleopardtiger_leopard_rt39', '007jettiprojectfileelephantleopardtiger_leopard_rt3r', '007jettiprojectfileelephantleopardtiger_leopard_rt4', '007jettiprojectfileelephantleopardtiger_leopard_rt40', '007jettiprojectfileelephantleopardtiger_leopard_rt4r', '007jettiprojectfileelephantleopardtiger_leoparddup', '007jettiprojectfileelephantleopardtiger_tblmineralogymodalslep_%', '007jettiprojectfileelephantleopardtiger_tblrt1', '007jettiprojectfileelephantleopardtiger_tblrt10', '007jettiprojectfileelephantleopardtiger_tblrt11', '007jettiprojectfileelephantleopardtiger_tblrt12', '007jettiprojectfileelephantleopardtiger_tblrt13', '007jettiprojectfileelephantleopardtiger_tblrt14', '007jettiprojectfileelephantleopardtiger_tblrt15', '007jettiprojectfileelephantleopardtiger_tblrt16', '007jettiprojectfileelephantleopardtiger_tblrt17', '007jettiprojectfileelephantleopardtiger_tblrt18', '007jettiprojectfileelephantleopardtiger_tblrt19', '007jettiprojectfileelephantleopardtiger_tblrt2', '007jettiprojectfileelephantleopardtiger_tblrt20', '007jettiprojectfileelephantleopardtiger_tblrt2r', '007jettiprojectfileelephantleopardtiger_tblrt3', '007jettiprojectfileelephantleopardtiger_tblrt37', '007jettiprojectfileelephantleopardtiger_tblrt38', '007jettiprojectfileelephantleopardtiger_tblrt39', '007jettiprojectfileelephantleopardtiger_tblrt3r', '007jettiprojectfileelephantleopardtiger_tblrt4', '007jettiprojectfileelephantleopardtiger_tblrt40', '007jettiprojectfileelephantleopardtiger_tblrt4r', '007jettiprojectfileelephantleopardtiger_tblrt5', '007jettiprojectfileelephantleopardtiger_tblrt5r', '007jettiprojectfileelephantleopardtiger_tblrt6', '007jettiprojectfileelephantleopardtiger_tblrt6r', '007jettiprojectfileelephantleopardtiger_tblrt7', '007jettiprojectfileelephantleopardtiger_tblrt7r', '007jettiprojectfileelephantleopardtiger_tblrt8', '007jettiprojectfileelephantleopardtiger_tblrt9', '007jettiprojectfileelephantleopardtiger_tigerm1007rt9101112', '007jettiprojectfileelephantleopardtiger_tigerm1_rt10', '007jettiprojectfileelephantleopardtiger_tigerm1_rt11', '007jettiprojectfileelephantleopardtiger_tigerm1_rt12', '007jettiprojectfileelephantleopardtiger_tigerm1_rt9', '007jettiprojectfileelephantleopardtiger_tigerm1dup', '007jettiprojectfileelephantleopardtiger_tigerm2007rt13141516', '007jettiprojectfileelephantleopardtiger_tigerm2_rt13', '007jettiprojectfileelephantleopardtiger_tigerm2_rt14', '007jettiprojectfileelephantleopardtiger_tigerm2_rt15', '007jettiprojectfileelephantleopardtiger_tigerm2_rt16', '007jettiprojectfileelephantleopardtiger_tigerm2dup', '007jettiprojectfileelephantleopardtiger_tigerm3007rt17181920', '007jettiprojectfileelephantleopardtiger_tigerm3_rt17', '007jettiprojectfileelephantleopardtiger_tigerm3_rt18', '007jettiprojectfileelephantleopardtiger_tigerm3_rt19', '007jettiprojectfileelephantleopardtiger_tigerm3_rt20', '007jettiprojectfileelephantleopardtiger_tigerm3dup', '007jettiprojectfilertm2', '007jettiprojectfilertm2_acsummary', '007jettiprojectfilertm2_acsummary_rtm2', '007jettiprojectfilertm2_rt1forcheckvalues', '007jettiprojectfilertm2_rt33', '007jettiprojectfilertm2_rt34', '007jettiprojectfilertm2_rt35', '007jettiprojectfilertm2_rt36', '007jettiprojectfilertm2_rtm2_rt33', '007jettiprojectfilertm2_rtm2_rt34', '007jettiprojectfilertm2_rtm2_rt35', '007jettiprojectfilertm2_rtm2_rt36', '007jettiprojectfilertm2_rtm2a', '007jettiprojectfilertm2_rtm2b', '007jettiprojectfilertm2_tblrt1d', '007jettiprojectfilertm2_tblrt33', '007jettiprojectfilertm2_tblrt34', '007jettiprojectfilertm2_tblrt35', '007jettiprojectfilertm2_tblrt36', '007jettiprojectfiletoquepala_acsummaryantigua', '007jettiprojectfiletoquepala_acsummaryantigua_antigua', '007jettiprojectfiletoquepala_acsummaryfresca', '007jettiprojectfiletoquepala_acsummaryfresca_fresca', '007jettiprojectfiletoquepala_rt21', '007jettiprojectfiletoquepala_rt22', '007jettiprojectfiletoquepala_rt23', '007jettiprojectfiletoquepala_rt24', '007jettiprojectfiletoquepala_rt25', '007jettiprojectfiletoquepala_rt26', '007jettiprojectfiletoquepala_rt27', '007jettiprojectfiletoquepala_rt28', '007jettiprojectfiletoquepala_tblmineralogymodalstoquepalaantigua_%', '007jettiprojectfiletoquepala_tblmineralogymodalstoquepalafrecsa_%', '007jettiprojectfiletoquepala_tblrt21', '007jettiprojectfiletoquepala_tblrt22', '007jettiprojectfiletoquepala_tblrt23', '007jettiprojectfiletoquepala_tblrt24', '007jettiprojectfiletoquepala_tblrt25', '007jettiprojectfiletoquepala_tblrt26', '007jettiprojectfiletoquepala_tblrt27', '007jettiprojectfiletoquepala_tblrt28', '007jettiprojectfiletoquepala_toquepalaantigua_rt25', '007jettiprojectfiletoquepala_toquepalaantigua_rt26', '007jettiprojectfiletoquepala_toquepalaantigua_rt27', '007jettiprojectfiletoquepala_toquepalaantigua_rt28', '007jettiprojectfiletoquepala_toquepalaantiguaa', '007jettiprojectfiletoquepala_toquepalaantiguab', '007jettiprojectfiletoquepala_toquepalaantiguasxs+10mesh', '007jettiprojectfiletoquepala_toquepalaantiguasxs+12inch', '007jettiprojectfiletoquepala_toquepalaantiguasxs+14inch', '007jettiprojectfiletoquepala_toquepalaantiguasxs+34inch', '007jettiprojectfiletoquepala_toquepalaantiguasxs10mesh', '007jettiprojectfiletoquepala_toquepalafresca_rt21', '007jettiprojectfiletoquepala_toquepalafresca_rt22', '007jettiprojectfiletoquepala_toquepalafresca_rt23', '007jettiprojectfiletoquepala_toquepalafresca_rt24', '007jettiprojectfiletoquepala_toquepalafrescaa', '007jettiprojectfiletoquepala_toquepalafrescab', '007jettiprojectfilezaldivar', '007jettiprojectfilezaldivar_acsummary', '007jettiprojectfilezaldivar_acsummary_zaldivar', '007jettiprojectfilezaldivar_rt29', '007jettiprojectfilezaldivar_rt30', '007jettiprojectfilezaldivar_rt31', '007jettiprojectfilezaldivar_rt32', '007jettiprojectfilezaldivar_tblrt29', '007jettiprojectfilezaldivar_tblrt30', '007jettiprojectfilezaldivar_tblrt31', '007jettiprojectfilezaldivar_tblrt32', '007jettiprojectfilezaldivar_zaldivar_rt29', '007jettiprojectfilezaldivar_zaldivar_rt30', '007jettiprojectfilezaldivar_zaldivar_rt31', '007jettiprojectfilezaldivar_zaldivar_rt32', '007jettiprojectfilezaldivar_zaldivarcpya', '007jettiprojectfilezaldivar_zaldivarcpyb', '007jettiprojectfilezaldivar_zaldivarcpysxs+10mesh', '007jettiprojectfilezaldivar_zaldivarcpysxs+12inch', '007jettiprojectfilezaldivar_zaldivarcpysxs+14inch', '007jettiprojectfilezaldivar_zaldivarcpysxs10mesh'],
    '011jettiprojectfile_rm': ['011jettiprojectfile_%_tblmineralogymodals', '011jettiprojectfile_011rt10', '011jettiprojectfile_011rt14', '011jettiprojectfile_011rt15', '011jettiprojectfile_011rt7', '011jettiprojectfile_011rt8', '011jettiprojectfile_011rt9', '011jettiprojectfile_021rt1', '011jettiprojectfile_021rt2', '011jettiprojectfile_acsummary_rm2020', '011jettiprojectfile_catresiduekg_tblmineralogymodals_%', '011jettiprojectfile_controlresiduekg_tblmineralogymodals_%', '011jettiprojectfile_headkg_tblmineralogymodals_%', '011jettiprojectfile_rm1catresidue_tblmineralogymodals_%', '011jettiprojectfile_rm2020_rt1', '011jettiprojectfile_rm2020_rt10', '011jettiprojectfile_rm2020_rt14', '011jettiprojectfile_rm2020_rt15', '011jettiprojectfile_rm2020_rt2', '011jettiprojectfile_rm2020_rt7', '011jettiprojectfile_rm2020_rt8', '011jettiprojectfile_rm2020_rt9', '011jettiprojectfile_rm2controlresidue_tblmineralogymodals_%', '011jettiprojectfile_rmheadsample', '011jettiprojectfile_tblrt1', '011jettiprojectfile_tblrt10', '011jettiprojectfile_tblrt14', '011jettiprojectfile_tblrt15', '011jettiprojectfile_tblrt2', '011jettiprojectfile_tblrt7', '011jettiprojectfile_tblrt8', '011jettiprojectfile_tblrt9'],
    '011jettiprojectfile_rm_crushed': ['011jettiprojectfile_%_tblmineralogymodals', '011jettiprojectfile_catresiduekg_tblmineralogymodals_%', '011jettiprojectfile_controlresiduekg_tblmineralogymodals_%', '011jettiprojectfile_headkg_tblmineralogymodals_%', '011jettiprojectfilecrushed_rm2024_rt17', '011jettiprojectfilecrushed_rm2024_rt18', '011jettiprojectfilecrushed_rm2024_rt19', '011jettiprojectfilecrushed_rm2024_rt20', '011jettiprojectfilecrushed_rm2024_rt21', '011jettiprojectfilecrushed_rm2024_rt22', '011jettiprojectfilecrushed_rm2024_rt23', '011jettiprojectfilecrushed_rm2024_rt24', '011jettiprojectfilecrushed_tblrt17', '011jettiprojectfilecrushed_tblrt18', '011jettiprojectfilecrushed_tblrt19', '011jettiprojectfilecrushed_tblrt20', '011jettiprojectfilecrushed_tblrt21', '011jettiprojectfilecrushed_tblrt22', '011jettiprojectfilecrushed_tblrt23', '011jettiprojectfilecrushed_tblrt24'],
    '012jettiprojectfile_incremento': ['012jettiprojectfilecs_012rt1', '012jettiprojectfilecs_012rt2', '012jettiprojectfilecs_012rt3', '012jettiprojectfilecs_012rt4', '012jettiprojectfilecs_012rt5', '012jettiprojectfilecs_012rt6', '012jettiprojectfilecs_012rtb', '012jettiprojectfilecs_012rte', '012jettiprojectfilecs_012rtf', '012jettiprojectfilecs_012rtg', '012jettiprojectfilecs_acsummaryincremento', '012jettiprojectfilecs_acsummaryincremento_incremento', '012jettiprojectfilecs_incremento', '012jettiprojectfilecs_incremento_rt1', '012jettiprojectfilecs_incremento_rt2', '012jettiprojectfilecs_incremento_rt3', '012jettiprojectfilecs_incremento_rt4', '012jettiprojectfilecs_incremento_rt5', '012jettiprojectfilecs_incremento_rt6', '012jettiprojectfilecs_incremento_rtb', '012jettiprojectfilecs_incremento_rte', '012jettiprojectfilecs_incremento_rtf', '012jettiprojectfilecs_incremento_rtg', '012jettiprojectfilecs_incremento_tblmineralogymodals_%', '012jettiprojectfilecs_tblrt1', '012jettiprojectfilecs_tblrt2', '012jettiprojectfilecs_tblrt3', '012jettiprojectfilecs_tblrt4', '012jettiprojectfilecs_tblrt5', '012jettiprojectfilecs_tblrt6', '012jettiprojectfilecs_tblrtb', '012jettiprojectfilecs_tblrte', '012jettiprojectfilecs_tblrtf', '012jettiprojectfilecs_tblrtg'],
    '012jettiprojectfile_quebalix': ['012jettiprojectfilecs_acsummaryquebalix', '012jettiprojectfilecs_acsummaryquebalix_quebalix', '012jettiprojectfilecs_quebalix', '012jettiprojectfilecs_quebalix_rt10', '012jettiprojectfilecs_quebalix_rt7', '012jettiprojectfilecs_quebalix_rt8', '012jettiprojectfilecs_quebalix_rt9', '012jettiprojectfilecs_quebalix_rta', '012jettiprojectfilecs_quebalix_rtc', '012jettiprojectfilecs_quebalix_rtd', '012jettiprojectfilecs_quebalixiv_tblmineralogymodals_%', '012jettiprojectfilecs_rt10', '012jettiprojectfilecs_rt7', '012jettiprojectfilecs_rt8', '012jettiprojectfilecs_rt9', '012jettiprojectfilecs_rta', '012jettiprojectfilecs_rtc', '012jettiprojectfilecs_rtd', '012jettiprojectfilecs_tblrt10', '012jettiprojectfilecs_tblrt7', '012jettiprojectfilecs_tblrt8', '012jettiprojectfilecs_tblrt9', '012jettiprojectfilecs_tblrta', '012jettiprojectfilecs_tblrtc', '012jettiprojectfilecs_tblrtd'],
    '012jettiprojectfile_kino': ['012jettiprojectfilecs_acsummarykino', '012jettiprojectfilecs_acsummarykino_kino', '012jettiprojectfilecs_kino', '012jettiprojectfilecs_kino_tblmineralogymodals_%'],
    '012jettiprojectfile_noid': ['012jettiprojectfilecs', ],
    '013jettiprojectfile_oz': ['013jettiprojectfile_021rt4', '013jettiprojectfile_acsummary24h', '013jettiprojectfile_acsummary24h_oz', '013jettiprojectfile_acsummary72h', '013jettiprojectfile_acsummary72h_oz', '013jettiprojectfile_combinedheada', '013jettiprojectfile_combinedheadaverage', '013jettiprojectfile_combinedheadb', '013jettiprojectfile_combinedheadreport', '013jettiprojectfile_koala_rto1', '013jettiprojectfile_koala_rto2', '013jettiprojectfile_koala_rto3', '013jettiprojectfile_koala_rto4', '013jettiprojectfile_rto1', '013jettiprojectfile_rto2', '013jettiprojectfile_rto3', '013jettiprojectfile_tblmineralogymodals_%', '013jettiprojectfile_tblrto1', '013jettiprojectfile_tblrto2', '013jettiprojectfile_tblrto3', '013jettiprojectfile_tblrto4'],
    '014jettiprojectfile_bag': ['014jettiprojectfile_acsummarybag', '014jettiprojectfile_acsummarybag_bag', '014jettiprojectfile_bag_rtb1', '014jettiprojectfile_bag_rtb2', '014jettiprojectfile_bag_rtb3', '014jettiprojectfile_bag_rtb4', '014jettiprojectfile_bag_rtb5', '014jettiprojectfile_bag_rtb6', '014jettiprojectfile_bag_rtb7', '014jettiprojectfile_bag_rtb8', '014jettiprojectfile_bag_rtb9', '014jettiprojectfile_baghead', '014jettiprojectfile_tblbagmineralogymodals_bag%', '014jettiprojectfile_tblrtb1', '014jettiprojectfile_tblrtb2', '014jettiprojectfile_tblrtb3', '014jettiprojectfile_tblrtb4', '014jettiprojectfile_tblrtb5', '014jettiprojectfile_tblrtb6', '014jettiprojectfile_tblrtb7', '014jettiprojectfile_tblrtb8', '014jettiprojectfile_tblrtb9'],
    '014jettiprojectfile_kmb': ['014jettiprojectfile_acsummarykmb', '014jettiprojectfile_acsummarykmb_kmb', '014jettiprojectfile_kmb_rtk1r', '014jettiprojectfile_kmb_rtk2', '014jettiprojectfile_kmb_rtk3', '014jettiprojectfile_kmb_rtk4', '014jettiprojectfile_kmb_rtk5', '014jettiprojectfile_kmb_rtk6', '014jettiprojectfile_kmb_rtk7', '014jettiprojectfile_kmb_rtk8', '014jettiprojectfile_kmb_rtk9', '014jettiprojectfile_kmbhead', '014jettiprojectfile_tblkmbmineralogymodals_mineralmasskmb%', '014jettiprojectfile_tblrtk1r', '014jettiprojectfile_tblrtk2', '014jettiprojectfile_tblrtk3', '014jettiprojectfile_tblrtk4', '014jettiprojectfile_tblrtk5', '014jettiprojectfile_tblrtk6', '014jettiprojectfile_tblrtk7', '014jettiprojectfile_tblrtk8', '014jettiprojectfile_tblrtk9'],
    '014jettiprojectfile_noid': ['014jettiprojectfile'],
    '015jettiprojectfile_pv': ['015jettiprojectfile', '015jettiprojectfile_003rt1', '015jettiprojectfile_003rt2', '015jettiprojectfile_003rt3', '015jettiprojectfile_acsummary', '015jettiprojectfile_acsummary_acmr', '015jettiprojectfile_amcf+1', '015jettiprojectfile_amcf+12', '015jettiprojectfile_amcf+14', '015jettiprojectfile_amcf+34', '015jettiprojectfile_amcf+6mesh', '015jettiprojectfile_amcf6mesh', '015jettiprojectfile_amcf_hrt1', '015jettiprojectfile_amcf_hrt2', '015jettiprojectfile_amcf_hrt3', '015jettiprojectfile_amcf_hrt4', '015jettiprojectfile_amcf_hrt5', '015jettiprojectfile_amcfhead', '015jettiprojectfile_pv_rt1', '015jettiprojectfile_pv_rt2', '015jettiprojectfile_pv_rt3', '015jettiprojectfile_repamcf6mesh', '015jettiprojectfile_tblhrt1', '015jettiprojectfile_tblhrt2', '015jettiprojectfile_tblhrt3', '015jettiprojectfile_tblmineralogymodalssgs_%', '015jettiprojectfile_tblrt1', '015jettiprojectfile_tblrt2', '015jettiprojectfile_tblrt3'],
    '017jettiprojectfile_ea': ['017jettiprojectfile_017rtea1', '017jettiprojectfile_017rtea2', '017jettiprojectfile_017rtea3', '017jettiprojectfile_017rtea4', '017jettiprojectfile_017rtea5', '017jettiprojectfile_017rtea6', '017jettiprojectfile_017rtea7', '017jettiprojectfile_72hracsummary_eamillfeed', '017jettiprojectfile_catea4residue%_tblelamineralogymodals_%', '017jettiprojectfile_catea4residuekg_tblelamineralogymodals_%', '017jettiprojectfile_controlea1residue%_tblelamineralogymodals_%', '017jettiprojectfile_controlea1residuekg_tblelamineralogymodals_%', '017jettiprojectfile_eamillfeed+1', '017jettiprojectfile_eamillfeed+10m', '017jettiprojectfile_eamillfeed+12', '017jettiprojectfile_eamillfeed+14', '017jettiprojectfile_eamillfeed+150m', '017jettiprojectfile_eamillfeed+34', '017jettiprojectfile_eamillfeed+6m', '017jettiprojectfile_eamillfeed150m', '017jettiprojectfile_eamillfeed_rtea1', '017jettiprojectfile_eamillfeed_rtea2', '017jettiprojectfile_eamillfeed_rtea3', '017jettiprojectfile_eamillfeed_rtea4', '017jettiprojectfile_eamillfeed_rtea5', '017jettiprojectfile_eamillfeed_rtea6', '017jettiprojectfile_eamillfeed_rtea7', '017jettiprojectfile_eamillfeedcombined', '017jettiprojectfile_head%_tblelamineralogymodals_%', '017jettiprojectfile_headkg_tblelamineralogymodals_%', '017jettiprojectfile_tblrtea1', '017jettiprojectfile_tblrtea2', '017jettiprojectfile_tblrtea3', '017jettiprojectfile_tblrtea4', '017jettiprojectfile_tblrtea5', '017jettiprojectfile_tblrtea6',  '017jettiprojectfile_tblrtea7'],
    # pending for check '020jettiprojectfilehypogenesupergene_tblmineralogymodalshyp_hyp%'
    '020jettiprojectfile_hyp': ['020jettiprojectfilehypogenesupergene_tblmineralogymodalshyp_hyp', '020jettiprojectfilehypogenesupergene_hyp%_tblmineralogymodalshyp_%', '020jettiprojectfilehypogenesupergene_020rt1', '020jettiprojectfilehypogenesupergene_020rt2', '020jettiprojectfilehypogenesupergene_020rt3', '020jettiprojectfilehypogenesupergene_020rt4', '020jettiprojectfilehypogenesupergene_020rt5', '020jettiprojectfilehypogenesupergene_020rt6', '020jettiprojectfilehypogenesupergene_achyp', '020jettiprojectfilehypogenesupergene_achyp_hypogenecomp', '020jettiprojectfilehypogenesupergene_hypoamp', '020jettiprojectfilehypogenesupergene_hypobporp', '020jettiprojectfilehypogenesupergene_hypodio', '020jettiprojectfilehypogenesupergene_hypogene_rt1', '020jettiprojectfilehypogenesupergene_hypogene_rt2', '020jettiprojectfilehypogenesupergene_hypogene_rt3', '020jettiprojectfilehypogenesupergene_hypogene_rt4', '020jettiprojectfilehypogenesupergene_hypogene_rt5', '020jettiprojectfilehypogenesupergene_hypogene_rt6', '020jettiprojectfilehypogenesupergene_hypogenemastercompositea', '020jettiprojectfilehypogenesupergene_hypogenemastercompositeb', '020jettiprojectfilehypogenesupergene_hypoporp', '020jettiprojectfilehypogenesupergene_hypoqfg', '020jettiprojectfilehypogenesupergene_tblrt1', '020jettiprojectfilehypogenesupergene_tblrt2', '020jettiprojectfilehypogenesupergene_tblrt3', '020jettiprojectfilehypogenesupergene_tblrt4', '020jettiprojectfilehypogenesupergene_tblrt5', '020jettiprojectfilehypogenesupergene_tblrt6'],
    '020jettiprojectfile_sup': ['020jettiprojectfilehypogenesupergene_020rt19', '020jettiprojectfilehypogenesupergene_020rt20', '020jettiprojectfilehypogenesupergene_acsup', '020jettiprojectfilehypogenesupergene_acsup_supergene', '020jettiprojectfilehypogenesupergene_super', '020jettiprojectfilehypogenesupergene_supergene_rt19', '020jettiprojectfilehypogenesupergene_supergene_rt20', '020jettiprojectfilehypogenesupergene_tblmineralogymodalssup_sup%', '020jettiprojectfilehypogenesupergene_tblmineralogymodalssup2_sup%', '020jettiprojectfilehypogenesupergene_tblrt19', '020jettiprojectfilehypogenesupergene_tblrt20'],
    '020jettiprojectfile_har': ['020jettiprojectfilehardyandwaste_har%_tblmineralogymodals_%','020jettiprojectfilehardyandwaste_020rt25', '020jettiprojectfilehardyandwaste_020rt26', '020jettiprojectfilehardyandwaste_020rt27', '020jettiprojectfilehardyandwaste_020rt28', '020jettiprojectfilehardyandwaste_acsummary', '020jettiprojectfilehardyandwaste_acsummary_hardy', '020jettiprojectfilehardyandwaste_h21c1', '020jettiprojectfilehardyandwaste_h21c2', '020jettiprojectfilehardyandwaste_h21e', '020jettiprojectfilehardyandwaste_h21mastercomp', '020jettiprojectfilehardyandwaste_h21n', '020jettiprojectfilehardyandwaste_h21nw', '020jettiprojectfilehardyandwaste_h21sw', '020jettiprojectfilehardyandwaste_hardy_rt25', '020jettiprojectfilehardyandwaste_hardy_rt26', '020jettiprojectfilehardyandwaste_hardy_rt27', '020jettiprojectfilehardyandwaste_hardy_rt28', '020jettiprojectfilehardyandwaste_tblmineralogymodals_har%', '020jettiprojectfilehardyandwaste_tblrt25', '020jettiprojectfilehardyandwaste_tblrt26', '020jettiprojectfilehardyandwaste_tblrt27', '020jettiprojectfilehardyandwaste_tblrt28', '020jettiprojectfilehardyandwaste_wda', '020jettiprojectfilehardyandwaste_wdb', '020jettiprojectfilehardyandwaste_wdc', '020jettiprojectfilehardyandwaste_wdd', '020jettiprojectfilehardyandwaste_wde', '020jettiprojectfilehardyandwaste_wdf'],
    '021jettiprojectfile': ['021jettiprojectfile_hypogene_rt1', '021jettiprojectfile', '021jettiprojectfile_021rt1', '021jettiprojectfile_021rt2', '021jettiprojectfile_021rt3', '021jettiprojectfile_021rt4', '021jettiprojectfile_021rt5', '021jettiprojectfile_acsummaryhypogene24hr', '021jettiprojectfile_acsummaryhypogene24hr_hypogene', '021jettiprojectfile_acsummaryhypogene72hr', '021jettiprojectfile_acsummaryhypogene72hr_hypogene', '021jettiprojectfile_hypogene', '021jettiprojectfile_hypogene_rt1', '021jettiprojectfile_hypogene_rt2', '021jettiprojectfile_hypogene_rt3', '021jettiprojectfile_hypogene_rt4', '021jettiprojectfile_hypogene_rt5', '021jettiprojectfile_tblmineralogymodals_%', '021jettiprojectfile_tblrt1', '021jettiprojectfile_tblrt2', '021jettiprojectfile_tblrt3', '021jettiprojectfile_tblrt4', '021jettiprojectfile_tblrt5'],
    '022jettiprojectfile_stingray': ['022jettiprojectfile', '022jettiprojectfile_022rt1', '022jettiprojectfile_022rt10', '022jettiprojectfile_022rt11', '022jettiprojectfile_022rt12', '022jettiprojectfile_022rt13', '022jettiprojectfile_022rt2', '022jettiprojectfile_022rt3', '022jettiprojectfile_022rt4', '022jettiprojectfile_022rt5', '022jettiprojectfile_022rt6', '022jettiprojectfile_022rt7', '022jettiprojectfile_022rt8', '022jettiprojectfile_022rt9', '022jettiprojectfile_acsummary', '022jettiprojectfile_acsummary_belowcutoffgrade', '022jettiprojectfile_belowcutoffgrade_catalyzed', '022jettiprojectfile_belowcutoffgrade_control', '022jettiprojectfile_belowcutoffgrade_rt1', '022jettiprojectfile_belowcutoffgrade_rt10', '022jettiprojectfile_belowcutoffgrade_rt11', '022jettiprojectfile_belowcutoffgrade_rt12', '022jettiprojectfile_belowcutoffgrade_rt13', '022jettiprojectfile_belowcutoffgrade_rt2', '022jettiprojectfile_belowcutoffgrade_rt3', '022jettiprojectfile_belowcutoffgrade_rt4', '022jettiprojectfile_belowcutoffgrade_rt5', '022jettiprojectfile_belowcutoffgrade_rt6', '022jettiprojectfile_belowcutoffgrade_rt9', '022jettiprojectfile_stingray1head', '022jettiprojectfile_stingray1sxs+12', '022jettiprojectfile_stingray1sxs+14', '022jettiprojectfile_stingray1sxs+34', '022jettiprojectfile_stingray1sxs+6mesh', '022jettiprojectfile_stingray1sxs6mesh', '022jettiprojectfile_tblmineralcomposition_%', '022jettiprojectfile_tblrt1', '022jettiprojectfile_tblrt10', '022jettiprojectfile_tblrt11', '022jettiprojectfile_tblrt12', '022jettiprojectfile_tblrt13', '022jettiprojectfile_tblrt2', '022jettiprojectfile_tblrt3', '022jettiprojectfile_tblrt4', '022jettiprojectfile_tblrt5', '022jettiprojectfile_tblrt6', '022jettiprojectfile_tblrt7', '022jettiprojectfile_tblrt8', '022jettiprojectfile_tblrt9'],
    '023jettiprojectfile_ot9': ['023jettiprojectfile_023rt1', '023jettiprojectfile_023rt16r', '023jettiprojectfile_023rt2', '023jettiprojectfile_023rt3', '023jettiprojectfile_023rt4', '023jettiprojectfile_023rt5r', '023jettiprojectfile_023rt6r', '023jettiprojectfile_023rt7r', '023jettiprojectfile_acsummaryot924hrs', '023jettiprojectfile_acsummaryot924hrs_ot9', '023jettiprojectfile_acsummaryot972hrs', '023jettiprojectfile_acsummaryot972hrs_ot9', '023jettiprojectfile_ot9', '023jettiprojectfile_ot9_rt1', '023jettiprojectfile_ot9_rt16r', '023jettiprojectfile_ot9_rt2', '023jettiprojectfile_ot9_rt3', '023jettiprojectfile_ot9_rt4', '023jettiprojectfile_ot9_rt5r', '023jettiprojectfile_ot9_rt6r', '023jettiprojectfile_ot9_rt7r', '023jettiprojectfile_ot9avg', '023jettiprojectfile_ot9dup', '023jettiprojectfile_tblmineralogymodalsot9_ot09%', '023jettiprojectfile_tblrt1', '023jettiprojectfile_tblrt16r', '023jettiprojectfile_tblrt2', '023jettiprojectfile_tblrt3', '023jettiprojectfile_tblrt4', '023jettiprojectfile_tblrt5r', '023jettiprojectfile_tblrt6r', '023jettiprojectfile_tblrt7r'],
    '023jettiprojectfile_ot10': ['023jettiprojectfile_023rt10', '023jettiprojectfile_023rt11', '023jettiprojectfile_023rt12r', '023jettiprojectfile_023rt13r', '023jettiprojectfile_023rt14r', '023jettiprojectfile_023rt15r', '023jettiprojectfile_023rt8', '023jettiprojectfile_023rt8r', '023jettiprojectfile_023rt9', '023jettiprojectfile_acsummaryot1024hrs', '023jettiprojectfile_acsummaryot1024hrs_ot10', '023jettiprojectfile_acsummaryot1072hrs', '023jettiprojectfile_acsummaryot1072hrs_ot10', '023jettiprojectfile_ot10', '023jettiprojectfile_ot10_rt10', '023jettiprojectfile_ot10_rt11', '023jettiprojectfile_ot10_rt12r', '023jettiprojectfile_ot10_rt13r', '023jettiprojectfile_ot10_rt14r', '023jettiprojectfile_ot10_rt15r', '023jettiprojectfile_ot10_rt8', '023jettiprojectfile_ot10_rt8r', '023jettiprojectfile_ot10_rt9', '023jettiprojectfile_ot10avg', '023jettiprojectfile_ot10dup', '023jettiprojectfile_tblmineralogymodalsot10_ot10%', '023jettiprojectfile_tblrt10', '023jettiprojectfile_tblrt11', '023jettiprojectfile_tblrt12r', '023jettiprojectfile_tblrt13r', '023jettiprojectfile_tblrt14r', '023jettiprojectfile_tblrt15r', '023jettiprojectfile_tblrt8', '023jettiprojectfile_tblrt8r', '023jettiprojectfile_tblrt9'],
    '023jettiprojectfile_noid': ['023jettiprojectfile', '023jettiprojectfile_siteraffinate'],
    '024jettiprojectfile_cpy': ['024jettiprojectfile_024cvcpy', '024jettiprojectfile_024cvcpy+12', '024jettiprojectfile_024cvcpy+14', '024jettiprojectfile_024cvcpy+34', '024jettiprojectfile_024cvcpy+6mesh', '024jettiprojectfile_024cvcpy6mesh', '024jettiprojectfile_024cvcpyavg', '024jettiprojectfile_024cvcpydup', '024jettiprojectfile_24hracsummary_belowcutoff', '024jettiprojectfile_belowcutoff_rt1', '024jettiprojectfile_belowcutoff_rt2', '024jettiprojectfile_belowcutoff_rt3', '024jettiprojectfile_belowcutoff_rt4', '024jettiprojectfile_belowcutoff_rt5', '024jettiprojectfile_belowcutoff_rt6', '024jettiprojectfile_belowcutoff_rt7', '024jettiprojectfile_belowcutoff_rt8', '024jettiprojectfile_belowcutoff_rt9', '024jettiprojectfile_catalyzedkg_tblmineralogymodals_%', '024jettiprojectfile_catresiduecv4%_tblmineralogymodals_%', '024jettiprojectfile_controlkg_tblmineralogymodals_%', '024jettiprojectfile_controlresiduecv1%_tblmineralogymodals_%', '024jettiprojectfile_fecontrolcatcv3%_tblmineralogymodals_%', '024jettiprojectfile_fecontrolcatcv3kg_tblmineralogymodals_%', '024jettiprojectfile_fectrlcontrolcv2%_tblmineralogymodals_%', '024jettiprojectfile_fectrlcontrolcv2kg_tblmineralogymodals_%', '024jettiprojectfile_head%_tblmineralogymodals_%', '024jettiprojectfile_headkg_tblmineralogymodals_%', '024jettiprojectfile_syntheticraffinate17112022', '024jettiprojectfile_syntheticraffinatedup17112022', '024jettiprojectfile_tblrt1', '024jettiprojectfile_tblrt2', '024jettiprojectfile_tblrt3', '024jettiprojectfile_tblrt4', '024jettiprojectfile_tblrt5', '024jettiprojectfile_tblrt6', '024jettiprojectfile_tblrt7'],
    '025jettiprojectfile_oxide': ['025jettiprojectfile_tblmineralogymodalsoxide_%', '025jettiprojectfile_tblmineralogycudeportmentoxide_%', '025jettiprojectfile_023rt14', '025jettiprojectfile_025rt10', '025jettiprojectfile_025rt11', '025jettiprojectfile_025rt12', '025jettiprojectfile_025rt13', '025jettiprojectfile_025rt15', '025jettiprojectfile_025rt16', '025jettiprojectfile_025rt9', '025jettiprojectfile_acsummary_oxide', '025jettiprojectfile_oxide', '025jettiprojectfile_oxide_rt10', '025jettiprojectfile_oxide_rt11', '025jettiprojectfile_oxide_rt12', '025jettiprojectfile_oxide_rt13', '025jettiprojectfile_oxide_rt14', '025jettiprojectfile_oxide_rt15', '025jettiprojectfile_oxide_rt16', '025jettiprojectfile_oxide_rt35', '025jettiprojectfile_oxide_rt36', '025jettiprojectfile_oxide_rt37', '025jettiprojectfile_oxide_rt38', '025jettiprojectfile_oxide_rt9', '025jettiprojectfile_oxidedup', '025jettiprojectfile_tblrt10', '025jettiprojectfile_tblrt11', '025jettiprojectfile_tblrt12', '025jettiprojectfile_tblrt13', '025jettiprojectfile_tblrt14', '025jettiprojectfile_tblrt15', '025jettiprojectfile_tblrt16', '025jettiprojectfile_tblrt35', '025jettiprojectfile_tblrt36', '025jettiprojectfile_tblrt37', '025jettiprojectfile_tblrt38',  '025jettiprojectfile_tblrt9'],
    '025jettiprojectfile_secondary': ['025jettiprojectfile_tblmineralogymodalssecondary_%', '025jettiprojectfile_tblmineralogycudeportmentsecondary_%', '025jettiprojectfile_025rt1','025jettiprojectfile_025rt2', '025jettiprojectfile_025rt3', '025jettiprojectfile_025rt31', '025jettiprojectfile_025rt32', '025jettiprojectfile_025rt33', '025jettiprojectfile_025rt34', '025jettiprojectfile_025rt4', '025jettiprojectfile_025rt5', '025jettiprojectfile_025rt6', '025jettiprojectfile_025rt7', '025jettiprojectfile_025rt8', '025jettiprojectfile_acsummary_highsecondary', '025jettiprojectfile_highsecondary', '025jettiprojectfile_highsecondary_rt1', '025jettiprojectfile_highsecondary_rt2', '025jettiprojectfile_highsecondary_rt3', '025jettiprojectfile_highsecondary_rt31', '025jettiprojectfile_highsecondary_rt32', '025jettiprojectfile_highsecondary_rt33', '025jettiprojectfile_highsecondary_rt34', '025jettiprojectfile_highsecondary_rt4', '025jettiprojectfile_highsecondary_rt5', '025jettiprojectfile_highsecondary_rt6', '025jettiprojectfile_highsecondary_rt7', '025jettiprojectfile_highsecondary_rt8', '025jettiprojectfile_highsecondarydup', '025jettiprojectfile_tblrt1', '025jettiprojectfile_tblrt2', '025jettiprojectfile_tblrt3', '025jettiprojectfile_tblrt31', '025jettiprojectfile_tblrt32', '025jettiprojectfile_tblrt33', '025jettiprojectfile_tblrt34', '025jettiprojectfile_tblrt4', '025jettiprojectfile_tblrt5', '025jettiprojectfile_tblrt6', '025jettiprojectfile_tblrt7', '025jettiprojectfile_tblrt8'],
    '025jettiprojectfile_chalcopyrite': ['025jettiprojectfile_tblmineralogymodalschalcopyrite_%', '025jettiprojectfile_tblmineralogycudeportmentchalcopyrite_%', '025jettiprojectfile_025rt17', '025jettiprojectfile_025rt18', '025jettiprojectfile_025rt19', '025jettiprojectfile_025rt20', '025jettiprojectfile_025rt21', '025jettiprojectfile_025rt22', '025jettiprojectfile_025rt23', '025jettiprojectfile_025rt24', '025jettiprojectfile_025rt25', '025jettiprojectfile_025rt26', '025jettiprojectfile_025rt27', '025jettiprojectfile_025rt28', '025jettiprojectfile_025rt29', '025jettiprojectfile_025rt30', '025jettiprojectfile_025rt39', '025jettiprojectfile_025rt40', '025jettiprojectfile_025rt41', '025jettiprojectfile_025rt42', '025jettiprojectfile_025rt43', '025jettiprojectfile_acsummary_chalcopyrite', '025jettiprojectfile_chalcopyrite', '025jettiprojectfile_chalcopyrite_rt17', '025jettiprojectfile_chalcopyrite_rt18', '025jettiprojectfile_chalcopyrite_rt19', '025jettiprojectfile_chalcopyrite_rt20', '025jettiprojectfile_chalcopyrite_rt21', '025jettiprojectfile_chalcopyrite_rt22', '025jettiprojectfile_chalcopyrite_rt23', '025jettiprojectfile_chalcopyrite_rt24', '025jettiprojectfile_chalcopyrite_rt25', '025jettiprojectfile_chalcopyrite_rt26', '025jettiprojectfile_chalcopyrite_rt27', '025jettiprojectfile_chalcopyrite_rt28', '025jettiprojectfile_chalcopyrite_rt29', '025jettiprojectfile_chalcopyrite_rt30', '025jettiprojectfile_chalcopyrite_rt39', '025jettiprojectfile_chalcopyrite_rt40', '025jettiprojectfile_chalcopyrite_rt41', '025jettiprojectfile_chalcopyrite_rt42', '025jettiprojectfile_chalcopyrite_rt43', '025jettiprojectfile_chalcopyrite_rt44', '025jettiprojectfile_chalcopyrite_rt45', '025jettiprojectfile_chalcopyritedup', '025jettiprojectfile_tblrt17', '025jettiprojectfile_tblrt18', '025jettiprojectfile_tblrt19', '025jettiprojectfile_tblrt20', '025jettiprojectfile_tblrt21', '025jettiprojectfile_tblrt22', '025jettiprojectfile_tblrt23', '025jettiprojectfile_tblrt24', '025jettiprojectfile_tblrt25', '025jettiprojectfile_tblrt26', '025jettiprojectfile_tblrt27', '025jettiprojectfile_tblrt28', '025jettiprojectfile_tblrt29', '025jettiprojectfile_tblrt30', '025jettiprojectfile_tblrt39', '025jettiprojectfile_tblrt4', '025jettiprojectfile_tblrt40', '025jettiprojectfile_tblrt41', '025jettiprojectfile_tblrt42', '025jettiprojectfile_tblrt43', '025jettiprojectfile_tblrt44', '025jettiprojectfile_tblrt45'],
    '025jettiprojectfile_noid': ['025jettiprojectfile', '025jettiprojectfile_acsummary'],
    # '_026jettiprojectfile': ['026jettiprojectfile_raffinatereceivednov42024', '026jettiprojectfile_raffinateshipment1', '026jettiprojectfile_raffinateshipment2'],
    '026jettiprojectfile_primarysulfide': ['026jettiprojectfile_acsample1', '026jettiprojectfile_acsample1_sample1', '026jettiprojectfile_rt1', '026jettiprojectfile_rt2', '026jettiprojectfile_rt3', '026jettiprojectfile_rt33', '026jettiprojectfile_rt34', '026jettiprojectfile_rt35', '026jettiprojectfile_rt36', '026jettiprojectfile_rt4', '026jettiprojectfile_rt5', '026jettiprojectfile_rt6', '026jettiprojectfile_rt7', '026jettiprojectfile_rt8', '026jettiprojectfile_sample1primarysulfide', '026jettiprojectfile_sample1primarysulfide_rt1', '026jettiprojectfile_sample1primarysulfide_rt2', '026jettiprojectfile_sample1primarysulfide_rt3', '026jettiprojectfile_sample1primarysulfide_rt33', '026jettiprojectfile_sample1primarysulfide_rt34', '026jettiprojectfile_sample1primarysulfide_rt35', '026jettiprojectfile_sample1primarysulfide_rt36', '026jettiprojectfile_sample1primarysulfide_rt4', '026jettiprojectfile_sample1primarysulfide_rt5', '026jettiprojectfile_sample1primarysulfide_rt6', '026jettiprojectfile_sample1primarysulfide_rt7', '026jettiprojectfile_sample1primarysulfide_rt8', '026jettiprojectfile_tblmineralogymodalsprimarysulfide_%', '026jettiprojectfile_tblrt1',  '026jettiprojectfile_tblrt2', '026jettiprojectfile_tblrt3', '026jettiprojectfile_tblrt33', '026jettiprojectfile_tblrt34', '026jettiprojectfile_tblrt35', '026jettiprojectfile_tblrt36', '026jettiprojectfile_tblrt4', '026jettiprojectfile_tblrt5', '026jettiprojectfile_tblrt6', '026jettiprojectfile_tblrt7', '026jettiprojectfile_tblrt8'],
    '026jettiprojectfile_carrizalillo': ['026jettiprojectfile_acsample2', '026jettiprojectfile_acsample2_sample2', '026jettiprojectfile_rt10', '026jettiprojectfile_rt11', '026jettiprojectfile_rt12', '026jettiprojectfile_rt13', '026jettiprojectfile_rt14', '026jettiprojectfile_rt15', '026jettiprojectfile_rt16', '026jettiprojectfile_rt37', '026jettiprojectfile_rt38', '026jettiprojectfile_rt39', '026jettiprojectfile_rt40', '026jettiprojectfile_rt9', '026jettiprojectfile_sample2carrizalillo', '026jettiprojectfile_sample2carrizalillo_rt10', '026jettiprojectfile_sample2carrizalillo_rt11', '026jettiprojectfile_sample2carrizalillo_rt12', '026jettiprojectfile_sample2carrizalillo_rt13', '026jettiprojectfile_sample2carrizalillo_rt14', '026jettiprojectfile_sample2carrizalillo_rt15', '026jettiprojectfile_sample2carrizalillo_rt16', '026jettiprojectfile_sample2carrizalillo_rt37', '026jettiprojectfile_sample2carrizalillo_rt38', '026jettiprojectfile_sample2carrizalillo_rt39', '026jettiprojectfile_sample2carrizalillo_rt40', '026jettiprojectfile_sample2carrizalillo_rt9', '026jettiprojectfile_tblmineralogymodalscarrizalillo_%', '026jettiprojectfile_tblrt10', '026jettiprojectfile_tblrt11', '026jettiprojectfile_tblrt12', '026jettiprojectfile_tblrt13', '026jettiprojectfile_tblrt14', '026jettiprojectfile_tblrt15', '026jettiprojectfile_tblrt16', '026jettiprojectfile_tblrt37', '026jettiprojectfile_tblrt38', '026jettiprojectfile_tblrt39', '026jettiprojectfile_tblrt40', '026jettiprojectfile_tblrt9'],
    '026jettiprojectfile_secondarysulfide': ['cha_026jettiprojectfile_sample3secondarysulfide', 'rea_026jettiprojectfile_sample3secondarysulfide', '026jettiprojectfile_acsample3', '026jettiprojectfile_acsample3_sample3', '026jettiprojectfile_rt17', '026jettiprojectfile_rt18', '026jettiprojectfile_rt19', '026jettiprojectfile_rt20', '026jettiprojectfile_rt21', '026jettiprojectfile_rt22', '026jettiprojectfile_rt23', '026jettiprojectfile_rt24',  '026jettiprojectfile_rt41', '026jettiprojectfile_rt42', '026jettiprojectfile_rt43', '026jettiprojectfile_rt44', '026jettiprojectfile_sample3secondarysulfide', '026jettiprojectfile_sample3secondarysulfide_rt17', '026jettiprojectfile_sample3secondarysulfide_rt18', '026jettiprojectfile_sample3secondarysulfide_rt19', '026jettiprojectfile_sample3secondarysulfide_rt20', '026jettiprojectfile_sample3secondarysulfide_rt21', '026jettiprojectfile_sample3secondarysulfide_rt22', '026jettiprojectfile_sample3secondarysulfide_rt23', '026jettiprojectfile_sample3secondarysulfide_rt24', '026jettiprojectfile_sample3secondarysulfide_rt41', '026jettiprojectfile_sample3secondarysulfide_rt42', '026jettiprojectfile_sample3secondarysulfide_rt43', '026jettiprojectfile_sample3secondarysulfide_rt44', '026jettiprojectfile_sample3secondarysulfide_rt49', '026jettiprojectfile_sample3secondarysulfide_rt50', '026jettiprojectfile_sample3secondarysulfide_rt51', '026jettiprojectfile_sample3secondarysulfide_rt52', '026jettiprojectfile_sample3secondarysulfide_rt53', '026jettiprojectfile_tblmineralogymodalssecondarysulfide_%', '026jettiprojectfile_tblrt17', '026jettiprojectfile_tblrt18', '026jettiprojectfile_tblrt19', '026jettiprojectfile_tblrt20', '026jettiprojectfile_tblrt21', '026jettiprojectfile_tblrt22', '026jettiprojectfile_tblrt23', '026jettiprojectfile_tblrt24', '026jettiprojectfile_tblrt41', '026jettiprojectfile_tblrt42', '026jettiprojectfile_tblrt43', '026jettiprojectfile_tblrt44', '026jettiprojectfile_tblrt49', '026jettiprojectfile_tblrt50', '026jettiprojectfile_tblrt51', '026jettiprojectfile_tblrt52', '026jettiprojectfile_tblrt53'],
    # find out why the key is not being corrrectly created cha_239....
    '026jettiprojectfile_oxides': ['026jettiprojectfile_acsample4', '026jettiprojectfile_acsample4_sample4', '026jettiprojectfile_rt25', '026jettiprojectfile_rt26', '026jettiprojectfile_rt27', '026jettiprojectfile_rt28', '026jettiprojectfile_rt29', '026jettiprojectfile_rt30', '026jettiprojectfile_rt31', '026jettiprojectfile_rt32', '026jettiprojectfile_rt45', '026jettiprojectfile_rt46', '026jettiprojectfile_rt47', '026jettiprojectfile_rt48', '026jettiprojectfile_sample4mixedmaterial', '026jettiprojectfile_sample4mixedmaterial_rt25', '026jettiprojectfile_sample4mixedmaterial_rt26', '026jettiprojectfile_sample4mixedmaterial_rt27', '026jettiprojectfile_sample4mixedmaterial_rt28', '026jettiprojectfile_sample4mixedmaterial_rt29', '026jettiprojectfile_sample4mixedmaterial_rt30', '026jettiprojectfile_sample4mixedmaterial_rt31', '026jettiprojectfile_sample4mixedmaterial_rt32', '026jettiprojectfile_sample4mixedmaterial_rt45', '026jettiprojectfile_sample4mixedmaterial_rt46', '026jettiprojectfile_sample4mixedmaterial_rt47', '026jettiprojectfile_sample4mixedmaterial_rt48', '026jettiprojectfile_tblrt25', '026jettiprojectfile_tblrt26', '026jettiprojectfile_tblrt27', '026jettiprojectfile_tblrt28', '026jettiprojectfile_tblrt29', '026jettiprojectfile_tblrt30', '026jettiprojectfile_tblrt31', '026jettiprojectfile_tblrt32', '026jettiprojectfile_tblrt45', '026jettiprojectfile_tblrt46', '026jettiprojectfile_tblrt47', '026jettiprojectfile_tblrt48', '026jettiprojectfile_tblmineralogymodalsmixed_%'],
    '028jettiprojectfile_composite': ['028jettiprojectfile_accompositesummary', '028jettiprojectfile_accompsummary_composite', '028jettiprojectfile_composite', '028jettiprojectfile_composite_rt11', '028jettiprojectfile_composite_rt12', '028jettiprojectfile_composite_rt13', '028jettiprojectfile_composite_rt14', '028jettiprojectfile_tblmineralogymodalscomp_%', '028jettiprojectfile_tblrt11', '028jettiprojectfile_tblrt12', '028jettiprojectfile_tblrt13', '028jettiprojectfile_tblrt14'],
    '028jettiprojectfile_andesite': ['028jettiprojectfile_acandesitesummary', '028jettiprojectfile_acandesitesummary_andesite', '028jettiprojectfile_andesite', '028jettiprojectfile_andesite_rt1', '028jettiprojectfile_andesite_rt2', '028jettiprojectfile_andesite_rt3', '028jettiprojectfile_andesite_rt4', '028jettiprojectfile_andesitedup', '028jettiprojectfile_tblmineralogymodalsandesite_%', '028jettiprojectfile_tblrt1', '028jettiprojectfile_tblrt2', '028jettiprojectfile_tblrt3', '028jettiprojectfile_tblrt4'],
    '028jettiprojectfile_monzonite': ['028jettiprojectfile_acmonzonitesummary', '028jettiprojectfile_acmonzonitesummary_monzonite', '028jettiprojectfile_monzonite', '028jettiprojectfile_monzonite_rt10', '028jettiprojectfile_monzonite_rt7', '028jettiprojectfile_monzonite_rt8', '028jettiprojectfile_monzonite_rt9', '028jettiprojectfile_monzonitedup', '028jettiprojectfile_tblmineralogymodalsmonzonite_%', '028jettiprojectfile_tblrt10', '028jettiprojectfile_tblrt7', '028jettiprojectfile_tblrt8', '028jettiprojectfile_tblrt9'],
    '028jettiprojectfile_noid': ['028jettiprojectfile'],
    '030jettiprojectfile_cpy': ['030jettiprojectfile', '030jettiprojectfile_tblcpymodals_%', '030jettiprojectfile_acsummarycpy_cpy', '030jettiprojectfile_acsummarycpy', '030jettiprojectfile_cpy', '030jettiprojectfile_cpy_rt1', '030jettiprojectfile_cpy_rt1r', '030jettiprojectfile_cpy_rt2', '030jettiprojectfile_cpy_rt3', '030jettiprojectfile_cpy_rt4', '030jettiprojectfile_cpy_rt4r', '030jettiprojectfile_rt1', '030jettiprojectfile_rt1r', '030jettiprojectfile_rt2', '030jettiprojectfile_rt3', '030jettiprojectfile_rt4', '030jettiprojectfile_rt4r', '030jettiprojectfile_tblrt1', '030jettiprojectfile_tblrt1r', '030jettiprojectfile_tblrt2', '030jettiprojectfile_tblrt3', '030jettiprojectfile_tblrt4', '030jettiprojectfile_tblrt4r'],
    '030jettiprojectfile_ss': ['030jettiprojectfile_tblssmodals_%', '030jettiprojectfile_acsummaryss', '030jettiprojectfile_acsummaryss_ss', '030jettiprojectfile_ss'],
    '031jettiprojectfile_sample': ['031jettiprojectfile_tblcumodals031_%', '031jettiprojectfile_031rt1', '031jettiprojectfile_031rt2', '031jettiprojectfile_031rt3', '031jettiprojectfile_031rt4', '031jettiprojectfile_acsummary_antcomp', '031jettiprojectfile_antcomp', '031jettiprojectfile_antcomp_rt1', '031jettiprojectfile_antcomp_rt2', '031jettiprojectfile_antcomp_rt3', '031jettiprojectfile_antcomp_rt4', '031jettiprojectfile_tblrt1', '031jettiprojectfile_tblrt2', '031jettiprojectfile_tblrt3', '031jettiprojectfile_tblrt4'],
    '032a_jettifile_ugm2': ['032augm2jettifile_acsummary_ugm2', '032augm2jettifile_elephant2ugma', '032augm2jettifile_elephant2ugmb', '032augm2jettifile_elephant2ugmc', '032augm2jettifile_repelephant2ugma', '032augm2jettifile_rt1', '032augm2jettifile_rt2', '032augm2jettifile_rt3', '032augm2jettifile_rt4', '032augm2jettifile_rt5', '032augm2jettifile_rt6', '032augm2jettifile_rt7', '032augm2jettifile_rt8', '032augm2jettifile_tblmineralogymodalsugm2_%', '032augm2jettifile_tblrt1', '032augm2jettifile_tblrt2', '032augm2jettifile_tblrt3', '032augm2jettifile_tblrt4', '032augm2jettifile_tblrt5', '032augm2jettifile_tblrt6', '032augm2jettifile_tblrt7', '032augm2jettifile_tblrt8', '032augm2jettifile_ugm2_rt1', '032augm2jettifile_ugm2_rt2', '032augm2jettifile_ugm2_rt3', '032augm2jettifile_ugm2_rt4', '032augm2jettifile_ugm2_rt5', '032augm2jettifile_ugm2_rt6', '032augm2jettifile_ugm2_rt7', '032augm2jettifile_ugm2_rt8', '032augm2jettifile_ugm2average', '032augm2jettifile_acsummary_ugm2', '032augm2jettifile_elephant2ugma', '032augm2jettifile_elephant2ugmb', '032augm2jettifile_elephant2ugmc', '032augm2jettifile_repelephant2ugma', '032augm2jettifile_rt1', '032augm2jettifile_rt2', '032augm2jettifile_rt3', '032augm2jettifile_rt4', '032augm2jettifile_rt5', '032augm2jettifile_rt6', '032augm2jettifile_rt7', '032augm2jettifile_rt8', '032augm2jettifile_tblmineralogymodalsugm2_%', '032augm2jettifile_tblrt1', '032augm2jettifile_tblrt2', '032augm2jettifile_tblrt3', '032augm2jettifile_tblrt4', '032augm2jettifile_tblrt5', '032augm2jettifile_tblrt6', '032augm2jettifile_tblrt7', '032augm2jettifile_tblrt8', '032augm2jettifile_ugm2_rt1', '032augm2jettifile_ugm2_rt2', '032augm2jettifile_ugm2_rt3', '032augm2jettifile_ugm2_rt4', '032augm2jettifile_ugm2_rt5', '032augm2jettifile_ugm2_rt6', '032augm2jettifile_ugm2_rt7', '032augm2jettifile_ugm2_rt8', '032augm2jettifile_ugm2average'],
    '032b_jettifile_pq': ['032bpqjettifile_acsummary_pq', '032bpqjettifile_elephant2pqa', '032bpqjettifile_elephant2pqb', '032bpqjettifile_elephant2pqc', '032bpqjettifile_pq_rt10', '032bpqjettifile_pq_rt11', '032bpqjettifile_pq_rt12', '032bpqjettifile_pq_rt13', '032bpqjettifile_pq_rt14', '032bpqjettifile_pq_rt15', '032bpqjettifile_pq_rt16', '032bpqjettifile_pq_rt9', '032bpqjettifile_pqaverage', '032bpqjettifile_rt10', '032bpqjettifile_rt11', '032bpqjettifile_rt12', '032bpqjettifile_rt13', '032bpqjettifile_rt14', '032bpqjettifile_rt15', '032bpqjettifile_rt16', '032bpqjettifile_rt9', '032bpqjettifile_tblmineralogymodalspq_%', '032bpqjettifile_tblrt10', '032bpqjettifile_tblrt11', '032bpqjettifile_tblrt12', '032bpqjettifile_tblrt13', '032bpqjettifile_tblrt14', '032bpqjettifile_tblrt15', '032bpqjettifile_tblrt16', '032bpqjettifile_tblrt9', '032bpqjettifile_acsummary_pq', '032bpqjettifile_elephant2pqa', '032bpqjettifile_elephant2pqb', '032bpqjettifile_elephant2pqc', '032bpqjettifile_pq_rt10', '032bpqjettifile_pq_rt11', '032bpqjettifile_pq_rt12', '032bpqjettifile_pq_rt13', '032bpqjettifile_pq_rt14', '032bpqjettifile_pq_rt15', '032bpqjettifile_pq_rt16', '032bpqjettifile_pq_rt9', '032bpqjettifile_pqaverage', '032bpqjettifile_rt10', '032bpqjettifile_rt11', '032bpqjettifile_rt12', '032bpqjettifile_rt13', '032bpqjettifile_rt14', '032bpqjettifile_rt15', '032bpqjettifile_rt16', '032bpqjettifile_rt9', '032bpqjettifile_tblmineralogymodalspq_%', '032bpqjettifile_tblrt10', '032bpqjettifile_tblrt11', '032bpqjettifile_tblrt12', '032bpqjettifile_tblrt13', '032bpqjettifile_tblrt14', '032bpqjettifile_tblrt15', '032bpqjettifile_tblrt16', '032bpqjettifile_tblrt9'],
    '032c_jettifile_ugm2h': ['032cugm2hjettifile_acsummary_ugm2h', '032cugm2hjettifile_rt17', '032cugm2hjettifile_rt18', '032cugm2hjettifile_rt19', '032cugm2hjettifile_rt20', '032cugm2hjettifile_rt21', '032cugm2hjettifile_rt22', '032cugm2hjettifile_rt23', '032cugm2hjettifile_rt24', '032cugm2hjettifile_rt25', '032cugm2hjettifile_rt26', '032cugm2hjettifile_rt27', '032cugm2hjettifile_rt28', '032cugm2hjettifile_tblmineralogymodalsugm2h_%', '032cugm2hjettifile_tblrt17', '032cugm2hjettifile_tblrt18', '032cugm2hjettifile_tblrt19', '032cugm2hjettifile_tblrt20', '032cugm2hjettifile_tblrt21', '032cugm2hjettifile_tblrt22', '032cugm2hjettifile_tblrt23', '032cugm2hjettifile_tblrt24', '032cugm2hjettifile_tblrt25', '032cugm2hjettifile_tblrt26', '032cugm2hjettifile_tblrt27', '032cugm2hjettifile_tblrt28', '032cugm2hjettifile_ugm2h_rt17', '032cugm2hjettifile_ugm2h_rt18', '032cugm2hjettifile_ugm2h_rt19', '032cugm2hjettifile_ugm2h_rt20', '032cugm2hjettifile_ugm2h_rt21', '032cugm2hjettifile_ugm2h_rt22', '032cugm2hjettifile_ugm2h_rt23', '032cugm2hjettifile_ugm2h_rt24', '032cugm2hjettifile_ugm2h_rt25', '032cugm2hjettifile_ugm2h_rt26', '032cugm2hjettifile_ugm2h_rt27', '032cugm2hjettifile_ugm2h_rt28', '032cugm2hjettifile_ugm2high', '032cugm2hjettifile_ugm2high+12inches', '032cugm2hjettifile_ugm2high+14inches', '032cugm2hjettifile_ugm2high+15inches', '032cugm2hjettifile_ugm2high+1inches', '032cugm2hjettifile_ugm2high+34inches', '032cugm2hjettifile_ugm2high+6mesh', '032cugm2hjettifile_ugm2high6mesh', '032cugm2hjettifile_ugm2highaverage', '032cugm2hjettifile_ugm2highdup', '032cugm2hjettifile_acsummary_ugm2h', '032cugm2hjettifile_rt17', '032cugm2hjettifile_rt18', '032cugm2hjettifile_rt19', '032cugm2hjettifile_rt20', '032cugm2hjettifile_rt21', '032cugm2hjettifile_rt22', '032cugm2hjettifile_rt23', '032cugm2hjettifile_rt24', '032cugm2hjettifile_rt25', '032cugm2hjettifile_rt26', '032cugm2hjettifile_rt27', '032cugm2hjettifile_rt28', '032cugm2hjettifile_tblmineralogymodalsugm2h_%', '032cugm2hjettifile_tblrt17', '032cugm2hjettifile_tblrt18', '032cugm2hjettifile_tblrt19', '032cugm2hjettifile_tblrt20', '032cugm2hjettifile_tblrt21', '032cugm2hjettifile_tblrt22', '032cugm2hjettifile_tblrt23', '032cugm2hjettifile_tblrt24', '032cugm2hjettifile_tblrt25', '032cugm2hjettifile_tblrt26', '032cugm2hjettifile_tblrt27', '032cugm2hjettifile_tblrt28', '032cugm2hjettifile_ugm2h_rt17', '032cugm2hjettifile_ugm2h_rt18', '032cugm2hjettifile_ugm2h_rt19', '032cugm2hjettifile_ugm2h_rt20', '032cugm2hjettifile_ugm2h_rt21', '032cugm2hjettifile_ugm2h_rt22', '032cugm2hjettifile_ugm2h_rt23', '032cugm2hjettifile_ugm2h_rt24', '032cugm2hjettifile_ugm2h_rt25', '032cugm2hjettifile_ugm2h_rt26', '032cugm2hjettifile_ugm2h_rt27', '032cugm2hjettifile_ugm2h_rt28', '032cugm2hjettifile_ugm2high', '032cugm2hjettifile_ugm2high+12inches', '032cugm2hjettifile_ugm2high+14inches', '032cugm2hjettifile_ugm2high+15inches', '032cugm2hjettifile_ugm2high+1inches', '032cugm2hjettifile_ugm2high+34inches', '032cugm2hjettifile_ugm2high+6mesh', '032cugm2hjettifile_ugm2high6mesh', '032cugm2hjettifile_ugm2highaverage', '032cugm2hjettifile_ugm2highdup'],
    '032d_jettifile_andqscfat46': ['032dandqscfat46jettifile_acsummary_andqsc', '032dandqscfat46jettifile_andqsc', '032dandqscfat46jettifile_andqsc+12inches', '032dandqscfat46jettifile_andqsc+14inches', '032dandqscfat46jettifile_andqsc+15inches', '032dandqscfat46jettifile_andqsc+1inches', '032dandqscfat46jettifile_andqsc+34inches', '032dandqscfat46jettifile_andqsc+6mesh', '032dandqscfat46jettifile_andqsc6mesh', '032dandqscfat46jettifile_andqscdup', '032dandqscfat46jettifile_andqscfat46_rt29', '032dandqscfat46jettifile_andqscfat46_rt30', '032dandqscfat46jettifile_andqscfat46_rt31', '032dandqscfat46jettifile_andqscfat46_rt32', '032dandqscfat46jettifile_andqscfat46_rt33', '032dandqscfat46jettifile_andqscfat46_rt34', '032dandqscfat46jettifile_andqscfat46_rt35', '032dandqscfat46jettifile_andqscfat46_rt36', '032dandqscfat46jettifile_andqscfat46_rt37', '032dandqscfat46jettifile_andqscfat46_rt38', '032dandqscfat46jettifile_andqscfat46_rt39', '032dandqscfat46jettifile_andqscfat46_rt40', '032dandqscfat46jettifile_andqscfat46average', '032dandqscfat46jettifile_rt29', '032dandqscfat46jettifile_rt30', '032dandqscfat46jettifile_rt31', '032dandqscfat46jettifile_rt32', '032dandqscfat46jettifile_rt33', '032dandqscfat46jettifile_rt34', '032dandqscfat46jettifile_rt35', '032dandqscfat46jettifile_rt36', '032dandqscfat46jettifile_rt37', '032dandqscfat46jettifile_rt38', '032dandqscfat46jettifile_rt39', '032dandqscfat46jettifile_rt40', '032dandqscfat46jettifile_tblmineralogymodalsandqsc_%', '032dandqscfat46jettifile_tblrt29', '032dandqscfat46jettifile_tblrt30', '032dandqscfat46jettifile_tblrt31', '032dandqscfat46jettifile_tblrt32', '032dandqscfat46jettifile_tblrt33', '032dandqscfat46jettifile_tblrt34', '032dandqscfat46jettifile_tblrt35', '032dandqscfat46jettifile_tblrt36', '032dandqscfat46jettifile_tblrt37', '032dandqscfat46jettifile_tblrt38', '032dandqscfat46jettifile_tblrt39', '032dandqscfat46jettifile_tblrt40', '032dandqscfat46jettifile_acsummary_andqsc', '032dandqscfat46jettifile_andqsc', '032dandqscfat46jettifile_andqsc+12inches', '032dandqscfat46jettifile_andqsc+14inches', '032dandqscfat46jettifile_andqsc+15inches', '032dandqscfat46jettifile_andqsc+1inches', '032dandqscfat46jettifile_andqsc+34inches', '032dandqscfat46jettifile_andqsc+6mesh', '032dandqscfat46jettifile_andqsc6mesh', '032dandqscfat46jettifile_andqscdup', '032dandqscfat46jettifile_andqscfat46_rt29', '032dandqscfat46jettifile_andqscfat46_rt30', '032dandqscfat46jettifile_andqscfat46_rt31', '032dandqscfat46jettifile_andqscfat46_rt32', '032dandqscfat46jettifile_andqscfat46_rt33', '032dandqscfat46jettifile_andqscfat46_rt34', '032dandqscfat46jettifile_andqscfat46_rt35', '032dandqscfat46jettifile_andqscfat46_rt36', '032dandqscfat46jettifile_andqscfat46_rt37', '032dandqscfat46jettifile_andqscfat46_rt38', '032dandqscfat46jettifile_andqscfat46_rt39', '032dandqscfat46jettifile_andqscfat46_rt40', '032dandqscfat46jettifile_andqscfat46average', '032dandqscfat46jettifile_rt29', '032dandqscfat46jettifile_rt30', '032dandqscfat46jettifile_rt31', '032dandqscfat46jettifile_rt32', '032dandqscfat46jettifile_rt33', '032dandqscfat46jettifile_rt34', '032dandqscfat46jettifile_rt35', '032dandqscfat46jettifile_rt36', '032dandqscfat46jettifile_rt37', '032dandqscfat46jettifile_rt38', '032dandqscfat46jettifile_rt39', '032dandqscfat46jettifile_rt40', '032dandqscfat46jettifile_tblmineralogymodalsandqsc_%', '032dandqscfat46jettifile_tblrt29', '032dandqscfat46jettifile_tblrt30', '032dandqscfat46jettifile_tblrt31', '032dandqscfat46jettifile_tblrt32', '032dandqscfat46jettifile_tblrt33', '032dandqscfat46jettifile_tblrt34', '032dandqscfat46jettifile_tblrt35', '032dandqscfat46jettifile_tblrt36', '032dandqscfat46jettifile_tblrt37', '032dandqscfat46jettifile_tblrt38', '032dandqscfat46jettifile_tblrt39', '032dandqscfat46jettifile_tblrt40'],
    '032e_jettifile_active604': ['032eactive604jettifile_acsummary_active1604', '032eactive604jettifile_active1604_rt41', '032eactive604jettifile_active1604_rt42', '032eactive604jettifile_active1604_rt43', '032eactive604jettifile_active1604_rt44', '032eactive604jettifile_active1604_rt45', '032eactive604jettifile_active1604_rt46', '032eactive604jettifile_active1604_rt47', '032eactive604jettifile_active1604_rt48', '032eactive604jettifile_active1604_rt48r', '032eactive604jettifile_active1604_rt49', '032eactive604jettifile_active1604_rt50', '032eactive604jettifile_active1604_rt51', '032eactive604jettifile_active1604_rt52', '032eactive604jettifile_active1604average', '032eactive604jettifile_active1604r604', '032eactive604jettifile_active1604r604dup', '032eactive604jettifile_r604+12inch', '032eactive604jettifile_r604+14inch', '032eactive604jettifile_r604+15inch', '032eactive604jettifile_r604+1inch', '032eactive604jettifile_r604+34inch', '032eactive604jettifile_r604+6mesh', '032eactive604jettifile_r6046mesh', '032eactive604jettifile_rt41', '032eactive604jettifile_rt42', '032eactive604jettifile_rt43', '032eactive604jettifile_rt44', '032eactive604jettifile_rt45', '032eactive604jettifile_rt46', '032eactive604jettifile_rt47', '032eactive604jettifile_rt48r', '032eactive604jettifile_rt49', '032eactive604jettifile_rt50', '032eactive604jettifile_rt51', '032eactive604jettifile_rt52', '032eactive604jettifile_tblmineralogymodalsactive1_%', '032eactive604jettifile_tblrt41', '032eactive604jettifile_tblrt42', '032eactive604jettifile_tblrt43', '032eactive604jettifile_tblrt44', '032eactive604jettifile_tblrt45', '032eactive604jettifile_tblrt46', '032eactive604jettifile_tblrt47', '032eactive604jettifile_tblrt48r', '032eactive604jettifile_tblrt49', '032eactive604jettifile_tblrt50', '032eactive604jettifile_tblrt51', '032eactive604jettifile_tblrt52', '032eactive604jettifile_acsummary_active1604', '032eactive604jettifile_active1604_rt41', '032eactive604jettifile_active1604_rt42', '032eactive604jettifile_active1604_rt43', '032eactive604jettifile_active1604_rt44', '032eactive604jettifile_active1604_rt45', '032eactive604jettifile_active1604_rt46', '032eactive604jettifile_active1604_rt47', '032eactive604jettifile_active1604_rt48', '032eactive604jettifile_active1604_rt48r', '032eactive604jettifile_active1604_rt49', '032eactive604jettifile_active1604_rt50', '032eactive604jettifile_active1604_rt51', '032eactive604jettifile_active1604_rt52', '032eactive604jettifile_active1604average', '032eactive604jettifile_active1604r604', '032eactive604jettifile_active1604r604dup', '032eactive604jettifile_r604+12inch', '032eactive604jettifile_r604+14inch', '032eactive604jettifile_r604+15inch', '032eactive604jettifile_r604+1inch', '032eactive604jettifile_r604+34inch', '032eactive604jettifile_r604+6mesh', '032eactive604jettifile_r6046mesh', '032eactive604jettifile_rt41', '032eactive604jettifile_rt42', '032eactive604jettifile_rt43', '032eactive604jettifile_rt44', '032eactive604jettifile_rt45', '032eactive604jettifile_rt46', '032eactive604jettifile_rt47', '032eactive604jettifile_rt48r', '032eactive604jettifile_rt49', '032eactive604jettifile_rt50', '032eactive604jettifile_rt51', '032eactive604jettifile_rt52', '032eactive604jettifile_tblmineralogymodalsactive1_%', '032eactive604jettifile_tblrt41', '032eactive604jettifile_tblrt42', '032eactive604jettifile_tblrt43', '032eactive604jettifile_tblrt44', '032eactive604jettifile_tblrt45', '032eactive604jettifile_tblrt46', '032eactive604jettifile_tblrt47', '032eactive604jettifile_tblrt48r', '032eactive604jettifile_tblrt49', '032eactive604jettifile_tblrt50', '032eactive604jettifile_tblrt51', '032eactive604jettifile_tblrt52'],
    '032f_jettifile_active605606': ['032factive605606jettifile_acsummary_active2605606', '032factive605606jettifile_active2605606+12inch', '032factive605606jettifile_active2605606+14inch', '032factive605606jettifile_active2605606+15inch', '032factive605606jettifile_active2605606+1inch', '032factive605606jettifile_active2605606+34inch', '032factive605606jettifile_active2605606+6mesh', '032factive605606jettifile_active26056066mesh', '032factive605606jettifile_active2605606_rt53', '032factive605606jettifile_active2605606_rt54', '032factive605606jettifile_active2605606_rt55', '032factive605606jettifile_active2605606_rt56', '032factive605606jettifile_active2605606_rt57', '032factive605606jettifile_active2605606_rt58', '032factive605606jettifile_active2605606_rt59', '032factive605606jettifile_active2605606_rt60', '032factive605606jettifile_active2605606_rt61', '032factive605606jettifile_active2605606_rt62', '032factive605606jettifile_active2605606_rt63', '032factive605606jettifile_active2605606_rt64', '032factive605606jettifile_active2605606avg', '032factive605606jettifile_active2605and606', '032factive605606jettifile_active2605and606dup', '032factive605606jettifile_rt53', '032factive605606jettifile_rt54', '032factive605606jettifile_rt55', '032factive605606jettifile_rt56', '032factive605606jettifile_rt57', '032factive605606jettifile_rt58', '032factive605606jettifile_rt59', '032factive605606jettifile_rt60', '032factive605606jettifile_rt61', '032factive605606jettifile_rt62', '032factive605606jettifile_rt63', '032factive605606jettifile_rt64', '032factive605606jettifile_tblmineralogymodalsactive2_%', '032factive605606jettifile_tblrt53', '032factive605606jettifile_tblrt54', '032factive605606jettifile_tblrt55', '032factive605606jettifile_tblrt56', '032factive605606jettifile_tblrt57', '032factive605606jettifile_tblrt58', '032factive605606jettifile_tblrt59', '032factive605606jettifile_tblrt60', '032factive605606jettifile_tblrt61', '032factive605606jettifile_tblrt62', '032factive605606jettifile_tblrt63', '032factive605606jettifile_tblrt64', '032factive605606jettifile_acsummary_active2605606', '032factive605606jettifile_active2605606+12inch', '032factive605606jettifile_active2605606+14inch', '032factive605606jettifile_active2605606+15inch', '032factive605606jettifile_active2605606+1inch', '032factive605606jettifile_active2605606+34inch', '032factive605606jettifile_active2605606+6mesh', '032factive605606jettifile_active26056066mesh', '032factive605606jettifile_active2605606_rt53', '032factive605606jettifile_active2605606_rt54', '032factive605606jettifile_active2605606_rt55', '032factive605606jettifile_active2605606_rt56', '032factive605606jettifile_active2605606_rt57', '032factive605606jettifile_active2605606_rt58', '032factive605606jettifile_active2605606_rt59', '032factive605606jettifile_active2605606_rt60', '032factive605606jettifile_active2605606_rt61', '032factive605606jettifile_active2605606_rt62', '032factive605606jettifile_active2605606_rt63', '032factive605606jettifile_active2605606_rt64', '032factive605606jettifile_active2605606avg', '032factive605606jettifile_active2605and606', '032factive605606jettifile_active2605and606dup', '032factive605606jettifile_rt53', '032factive605606jettifile_rt54', '032factive605606jettifile_rt55', '032factive605606jettifile_rt56', '032factive605606jettifile_rt57', '032factive605606jettifile_rt58', '032factive605606jettifile_rt59', '032factive605606jettifile_rt60', '032factive605606jettifile_rt61', '032factive605606jettifile_rt62', '032factive605606jettifile_rt63', '032factive605606jettifile_rt64', '032factive605606jettifile_tblmineralogymodalsactive2_%', '032factive605606jettifile_tblrt53', '032factive605606jettifile_tblrt54', '032factive605606jettifile_tblrt55', '032factive605606jettifile_tblrt56', '032factive605606jettifile_tblrt57', '032factive605606jettifile_tblrt58', '032factive605606jettifile_tblrt59', '032factive605606jettifile_tblrt60', '032factive605606jettifile_tblrt61', '032factive605606jettifile_tblrt62', '032factive605606jettifile_tblrt63', '032factive605606jettifile_tblrt64'],
    '032g_jettifile_ugm1': ['032gugm1jettifile_acsummary_ugm1', '032gugm1jettifile_rt65', '032gugm1jettifile_rt66', '032gugm1jettifile_rt67', '032gugm1jettifile_rt68', '032gugm1jettifile_rt69', '032gugm1jettifile_rt70', '032gugm1jettifile_rt71', '032gugm1jettifile_rt72', '032gugm1jettifile_tblmineralogymodalsugm1_%', '032gugm1jettifile_tblrt65', '032gugm1jettifile_tblrt66', '032gugm1jettifile_tblrt67', '032gugm1jettifile_tblrt68', '032gugm1jettifile_tblrt69', '032gugm1jettifile_tblrt70', '032gugm1jettifile_tblrt71', '032gugm1jettifile_tblrt72', '032gugm1jettifile_ugm1', '032gugm1jettifile_ugm1_rt65', '032gugm1jettifile_ugm1_rt66', '032gugm1jettifile_ugm1_rt67', '032gugm1jettifile_ugm1_rt68', '032gugm1jettifile_ugm1_rt69', '032gugm1jettifile_ugm1_rt70', '032gugm1jettifile_ugm1_rt71', '032gugm1jettifile_ugm1_rt72', '032gugm1jettifile_ugm1average', '032gugm1jettifile_ugm1dup', '032gugm1jettifile_acsummary_ugm1', '032gugm1jettifile_rt65', '032gugm1jettifile_rt66', '032gugm1jettifile_rt67', '032gugm1jettifile_rt68', '032gugm1jettifile_rt69', '032gugm1jettifile_rt70', '032gugm1jettifile_rt71', '032gugm1jettifile_rt72', '032gugm1jettifile_tblmineralogymodalsugm1_%', '032gugm1jettifile_tblrt65', '032gugm1jettifile_tblrt66', '032gugm1jettifile_tblrt67', '032gugm1jettifile_tblrt68', '032gugm1jettifile_tblrt69', '032gugm1jettifile_tblrt70', '032gugm1jettifile_tblrt71', '032gugm1jettifile_tblrt72', '032gugm1jettifile_ugm1', '032gugm1jettifile_ugm1_rt65', '032gugm1jettifile_ugm1_rt66', '032gugm1jettifile_ugm1_rt67', '032gugm1jettifile_ugm1_rt68', '032gugm1jettifile_ugm1_rt69', '032gugm1jettifile_ugm1_rt70', '032gugm1jettifile_ugm1_rt71', '032gugm1jettifile_ugm1_rt72', '032gugm1jettifile_ugm1average', '032gugm1jettifile_ugm1dup'],
    '032h_jettifile_ugm4': ['032hugm4jettifile_acsummary_ugm4', '032hugm4jettifile_rt73', '032hugm4jettifile_rt74', '032hugm4jettifile_rt75', '032hugm4jettifile_rt76', '032hugm4jettifile_rt77', '032hugm4jettifile_rt78', '032hugm4jettifile_rt79', '032hugm4jettifile_rt80', '032hugm4jettifile_tblmineralogymodalsugm4_%', '032hugm4jettifile_tblrt73', '032hugm4jettifile_tblrt74', '032hugm4jettifile_tblrt75', '032hugm4jettifile_tblrt76', '032hugm4jettifile_tblrt77', '032hugm4jettifile_tblrt78', '032hugm4jettifile_tblrt79', '032hugm4jettifile_tblrt80', '032hugm4jettifile_ugm4', '032hugm4jettifile_ugm4_rt73', '032hugm4jettifile_ugm4_rt74', '032hugm4jettifile_ugm4_rt75', '032hugm4jettifile_ugm4_rt76', '032hugm4jettifile_ugm4_rt77', '032hugm4jettifile_ugm4_rt78', '032hugm4jettifile_ugm4_rt79', '032hugm4jettifile_ugm4_rt80', '032hugm4jettifile_ugm4average', '032hugm4jettifile_ugm4dup', '032hugm4jettifile_acsummary_ugm4', '032hugm4jettifile_rt73', '032hugm4jettifile_rt74', '032hugm4jettifile_rt75', '032hugm4jettifile_rt76', '032hugm4jettifile_rt77', '032hugm4jettifile_rt78', '032hugm4jettifile_rt79', '032hugm4jettifile_rt80', '032hugm4jettifile_tblmineralogymodalsugm4_%', '032hugm4jettifile_tblrt73', '032hugm4jettifile_tblrt74', '032hugm4jettifile_tblrt75', '032hugm4jettifile_tblrt76', '032hugm4jettifile_tblrt77', '032hugm4jettifile_tblrt78', '032hugm4jettifile_tblrt79', '032hugm4jettifile_tblrt80', '032hugm4jettifile_ugm4', '032hugm4jettifile_ugm4_rt73', '032hugm4jettifile_ugm4_rt74', '032hugm4jettifile_ugm4_rt75', '032hugm4jettifile_ugm4_rt76', '032hugm4jettifile_ugm4_rt77', '032hugm4jettifile_ugm4_rt78', '032hugm4jettifile_ugm4_rt79', '032hugm4jettifile_ugm4_rt80', '032hugm4jettifile_ugm4average', '032hugm4jettifile_ugm4dup'],
    '032i_jettifile_andbio': ['032iandbiojettifile_acsummary_andbio', '032iandbiojettifile_andbio', '032iandbiojettifile_andbio_rt81', '032iandbiojettifile_andbio_rt82', '032iandbiojettifile_andbio_rt83', '032iandbiojettifile_andbio_rt84', '032iandbiojettifile_andbio_rt85', '032iandbiojettifile_andbio_rt86', '032iandbiojettifile_andbio_rt87', '032iandbiojettifile_andbio_rt88', '032iandbiojettifile_andbioaverage', '032iandbiojettifile_andbiodup', '032iandbiojettifile_rt81', '032iandbiojettifile_rt82', '032iandbiojettifile_rt83', '032iandbiojettifile_rt84', '032iandbiojettifile_rt85', '032iandbiojettifile_rt86', '032iandbiojettifile_rt87', '032iandbiojettifile_rt88', '032iandbiojettifile_tblmineralogymodalsandbio_%', '032iandbiojettifile_tblrt81', '032iandbiojettifile_tblrt82', '032iandbiojettifile_tblrt83', '032iandbiojettifile_tblrt84', '032iandbiojettifile_tblrt85', '032iandbiojettifile_tblrt86', '032iandbiojettifile_tblrt87', '032iandbiojettifile_tblrt88', '032iandbiojettifile_acsummary_andbio', '032iandbiojettifile_andbio', '032iandbiojettifile_andbio_rt81', '032iandbiojettifile_andbio_rt82', '032iandbiojettifile_andbio_rt83', '032iandbiojettifile_andbio_rt84', '032iandbiojettifile_andbio_rt85', '032iandbiojettifile_andbio_rt86', '032iandbiojettifile_andbio_rt87', '032iandbiojettifile_andbio_rt88', '032iandbiojettifile_andbioaverage', '032iandbiojettifile_andbiodup', '032iandbiojettifile_rt81', '032iandbiojettifile_rt82', '032iandbiojettifile_rt83', '032iandbiojettifile_rt84', '032iandbiojettifile_rt85', '032iandbiojettifile_rt86', '032iandbiojettifile_rt87', '032iandbiojettifile_rt88', '032iandbiojettifile_tblmineralogymodalsandbio_%', '032iandbiojettifile_tblrt81', '032iandbiojettifile_tblrt82', '032iandbiojettifile_tblrt83', '032iandbiojettifile_tblrt84', '032iandbiojettifile_tblrt85', '032iandbiojettifile_tblrt86', '032iandbiojettifile_tblrt87', '032iandbiojettifile_tblrt88'],
    '032j_jettifile_ugm3': ['032jugm3jettifile_acsummary_ugm3', '032jugm3jettifile_rt89', '032jugm3jettifile_rt90', '032jugm3jettifile_rt91', '032jugm3jettifile_rt92', '032jugm3jettifile_rt93', '032jugm3jettifile_rt94', '032jugm3jettifile_rt95', '032jugm3jettifile_rt96', '032jugm3jettifile_tblmineralogymodalsugm3_%', '032jugm3jettifile_tblrt89', '032jugm3jettifile_tblrt90', '032jugm3jettifile_tblrt91', '032jugm3jettifile_tblrt92', '032jugm3jettifile_tblrt93', '032jugm3jettifile_tblrt94', '032jugm3jettifile_tblrt95', '032jugm3jettifile_tblrt96', '032jugm3jettifile_ugm3', '032jugm3jettifile_ugm3_rt89', '032jugm3jettifile_ugm3_rt90', '032jugm3jettifile_ugm3_rt91', '032jugm3jettifile_ugm3_rt92', '032jugm3jettifile_ugm3_rt93', '032jugm3jettifile_ugm3_rt94', '032jugm3jettifile_ugm3_rt95', '032jugm3jettifile_ugm3_rt96', '032jugm3jettifile_ugm3average', '032jugm3jettifile_ugm3dup', '032jugm3jettifile_acsummary_ugm3', '032jugm3jettifile_rt89', '032jugm3jettifile_rt90', '032jugm3jettifile_rt91', '032jugm3jettifile_rt92', '032jugm3jettifile_rt93', '032jugm3jettifile_rt94', '032jugm3jettifile_rt95', '032jugm3jettifile_rt96', '032jugm3jettifile_tblmineralogymodalsugm3_%', '032jugm3jettifile_tblrt89', '032jugm3jettifile_tblrt90', '032jugm3jettifile_tblrt91', '032jugm3jettifile_tblrt92', '032jugm3jettifile_tblrt93', '032jugm3jettifile_tblrt94', '032jugm3jettifile_tblrt95', '032jugm3jettifile_tblrt96', '032jugm3jettifile_ugm3', '032jugm3jettifile_ugm3_rt89', '032jugm3jettifile_ugm3_rt90', '032jugm3jettifile_ugm3_rt91', '032jugm3jettifile_ugm3_rt92', '032jugm3jettifile_ugm3_rt93', '032jugm3jettifile_ugm3_rt94', '032jugm3jettifile_ugm3_rt95', '032jugm3jettifile_ugm3_rt96', '032jugm3jettifile_ugm3average', '032jugm3jettifile_ugm3dup'],
    '035jettiprojectfile_sample2': ['035jettiprojectfile_rt1', '035jettiprojectfile_rt2', '035jettiprojectfile_rt3', '035jettiprojectfile_rt4', '035jettiprojectfile_rt5', '035jettiprojectfile_sample2', '035jettiprojectfile_sample2_rt1', '035jettiprojectfile_sample2_rt2', '035jettiprojectfile_sample2_rt3', '035jettiprojectfile_sample2_rt4', '035jettiprojectfile_sample2_rt5', '035jettiprojectfile_sample2acsummary_sample2', '035jettiprojectfile_sample2jabalshaybanavg', '035jettiprojectfile_tblrt1', '035jettiprojectfile_tblrt2', '035jettiprojectfile_tblrt3', '035jettiprojectfile_tblrt4', '035jettiprojectfile_tblrt5', ],
    '035jettiprojectfile_sample3': ['035jettiprojectfile_rt6', '035jettiprojectfile_rt7', '035jettiprojectfile_rt8', '035jettiprojectfile_rt9', '035jettiprojectfile_rt10', '035jettiprojectfile_sample3', '035jettiprojectfile_sample3_rt10', '035jettiprojectfile_sample3_rt6', '035jettiprojectfile_sample3_rt7', '035jettiprojectfile_sample3_rt8', '035jettiprojectfile_sample3_rt9', '035jettiprojectfile_sample3acsummary_sample3', '035jettiprojectfile_tblrt6', '035jettiprojectfile_tblrt7', '035jettiprojectfile_tblrt8', '035jettiprojectfile_tblrt9', '035jettiprojectfile_tblrt10'],
    # 'cha_225____013jettiprojectfile': ['cha_225____013jettiprojectfile\n226____013jettiprojectfile\n227____013jettiprojectfile\n228____013jettiprojectfile\nName:_project_sample_id_raw,_dtype:_object'],
    # 'cha_92____007bjettiprojectfile': ['cha_92____007bjettiprojectfiletiger\n93____007bjettiprojectfiletiger\nName:_project_sample_id_raw,_dtype:_object'],
    # ELEPHANT PQ AND UGM replaced by 007A,
    'jettifile_elephant_ugm': ['jettifileelephantiiver2ugm', 'jettifileelephantiiver2ugm_acsummarynew', 'jettifileelephantiiver2ugm_acsummarynew_ugm2', 'jettifileelephantiiver2ugm_aqglobalaugm2', 'jettifileelephantiiver2ugm_aqglobalbugm2', 'jettifileelephantiiver2ugm_tblmineralogymodals_%', 'jettifileelephantiiver2ugm_ugmaverage'],
    'jettifile_elephant_pq': ['jettifileelephantiiver2pq', 'jettifileelephantiiver2pq_acsummarynew', 'jettifileelephantiiver2pq_acsummarynew_pq', 'jettifileelephantiiver2pq_pqaverage', 'jettifileelephantiiver2pq_pqcch', 'jettifileelephantiiver2pq_pqroma', 'jettifileelephantiiver2pq_pqromb', 'jettifileelephantiiver2pq_tblelephantmodals_%'],
    # ELEPHANT SCL replaced by 007,
    'jettiprojectfile_elephantscl': ['jettiprojectfileelephantscl', 'jettiprojectfileelephantscl_acsummary', 'jettiprojectfileelephantscl_acsummary_sampleescondida', 'jettiprojectfileelephantscl_columna42', 'jettiprojectfileelephantscl_columna43', 'jettiprojectfileelephantscl_columna52', 'jettiprojectfileelephantscl_columna53', 'jettiprojectfileelephantscl_rt1', 'jettiprojectfileelephantscl_rt2', 'jettiprojectfileelephantscl_rt3', 'jettiprojectfileelephantscl_rt4', 'jettiprojectfileelephantscl_rt5', 'jettiprojectfileelephantscl_rt6', 'jettiprojectfileelephantscl_rt7', 'jettiprojectfileelephantscl_rt8', 'jettiprojectfileelephantscl_sampleescondida', 'jettiprojectfileelephantscl_sampleescondida_rt1', 'jettiprojectfileelephantscl_sampleescondida_rt2', 'jettiprojectfileelephantscl_sampleescondida_rt3', 'jettiprojectfileelephantscl_sampleescondida_rt4', 'jettiprojectfileelephantscl_sampleescondida_rt5', 'jettiprojectfileelephantscl_sampleescondida_rt6', 'jettiprojectfileelephantscl_sampleescondida_rt7', 'jettiprojectfileelephantscl_sampleescondida_rt8', 'jettiprojectfileelephantscl_tblmodalsmel_sampleescondida', 'jettiprojectfileelephantscl_tblrt1', 'jettiprojectfileelephantscl_tblrt19', 'jettiprojectfileelephantscl_tblrt2', 'jettiprojectfileelephantscl_tblrt20', 'jettiprojectfileelephantscl_tblrt21', 'jettiprojectfileelephantscl_tblrt22', 'jettiprojectfileelephantscl_tblrt23', 'jettiprojectfileelephantscl_tblrt3', 'jettiprojectfileelephantscl_tblrt4', 'jettiprojectfileelephantscl_tblrt5', 'jettiprojectfileelephantscl_tblrt6', 'jettiprojectfileelephantscl_tblrt7', 'jettiprojectfileelephantscl_tblrt8'],
    'jettiprojectfile_elephantsite': ['jettiprojectfileelephantsite', 'jettiprojectfileelephantsite_elephant', 'jettiprojectfileelephantsite_tblmineralogymodalsm1_elephanthead'],
    'jettiprojectfile_leopard': ['jettiprojectfileleopardscl_acsummarysgsburnaby_samplelosbronces', 'jettiprojectfileleopardscl_acsummarysgssantiago_samplelosbronces', 'jettiprojectfileleopardscl_rt1', 'jettiprojectfileleopardscl_rt2', 'jettiprojectfileleopardscl_rt3', 'jettiprojectfileleopardscl_rt4', 'jettiprojectfileleopardscl_rt5', 'jettiprojectfileleopardscl_rt6', 'jettiprojectfileleopardscl_samplelos', 'jettiprojectfileleopardscl_samplelosbronces_rt1', 'jettiprojectfileleopardscl_samplelosbronces_rt2', 'jettiprojectfileleopardscl_samplelosbronces_rt3', 'jettiprojectfileleopardscl_samplelosbronces_rt4', 'jettiprojectfileleopardscl_samplelosbronces_rt5', 'jettiprojectfileleopardscl_samplelosbronces_rt6', 'jettiprojectfileleopardscl_tblmodalslosbronces_samplelosbronces', 'jettiprojectfileleopardscl_tblrt1', 'jettiprojectfileleopardscl_tblrt2', 'jettiprojectfileleopardscl_tblrt3', 'jettiprojectfileleopardscl_tblrt4', 'jettiprojectfileleopardscl_tblrt5', 'jettiprojectfileleopardscl_tblrt6',],
    # TIGER replaced by 007B, TOQUEPALA AND ZALDIVAR ARE REPLACED BY 007 (MONSE TEAMS MESSAGE APRIL 11TH 2025)
    'jettiprojectfile_tiger_m1': ['jettiprojectfiletigerrom', 'jettiprojectfiletigerrom_bt213', 'jettiprojectfiletigerrom_bt214', 'jettiprojectfiletigerrom_m1', 'jettiprojectfiletigerrom_m1_bt213', 'jettiprojectfiletigerrom_m1_bt214', 'jettiprojectfiletigerrom_m1acsummary', 'jettiprojectfiletigerrom_m1acsummary_m1', 'jettiprojectfiletigerrom_tblmineralogymodalsm1_m1head', 'jettiprojectfiletigerrom_tblbt213', 'jettiprojectfiletigerrom_tblbt214'],
    'jettiprojectfile_tiger_m2': ['jettiprojectfiletigerrom_bt424', 'jettiprojectfiletigerrom_bt425', 'jettiprojectfiletigerrom_m2', 'jettiprojectfiletigerrom_m2_bt424', 'jettiprojectfiletigerrom_m2_bt425', 'jettiprojectfiletigerrom_m2acsummary', 'jettiprojectfiletigerrom_m2acsummary_m2', 'jettiprojectfiletigerrom_tblbt424', 'jettiprojectfiletigerrom_tblbt425', 'jettiprojectfiletigerrom_tblmineralogymodalsm2_m2head'],
    'jettiprojectfile_tiger_m3': ['jettiprojectfiletigerrom_bt426', 'jettiprojectfiletigerrom_bt427', 'jettiprojectfiletigerrom_m3', 'jettiprojectfiletigerrom_m3_bt426', 'jettiprojectfiletigerrom_m3_bt427', 'jettiprojectfiletigerrom_m3acsummary', 'jettiprojectfiletigerrom_m3acsummary_m3', 'jettiprojectfiletigerrom_tblbt426', 'jettiprojectfiletigerrom_tblbt427', 'jettiprojectfiletigerrom_tblmineralogymodalsm3_m3head'],
    'jettiprojectfile_toquepala_antigua': ['jettiprojectfiletoquepalascl_acsummaryantigua', 'jettiprojectfiletoquepalascl_acsummaryantigua_sampleantigua', 'jettiprojectfiletoquepalascl_rt11', 'jettiprojectfiletoquepalascl_rt12', 'jettiprojectfiletoquepalascl_sampleantigua', 'jettiprojectfiletoquepalascl_sampleantigua_rt11', 'jettiprojectfiletoquepalascl_sampleantigua_rt12', 'jettiprojectfiletoquepalascl_tblmodalstoquepalaantigua_sampleantigua', 'jettiprojectfiletoquepalascl_tblrt11', 'jettiprojectfiletoquepalascl_tblrt12'],
    'jettiprojectfile_toquepala_fresca': ['jettiprojectfiletoquepalascl_acsummaryfresca', 'jettiprojectfiletoquepalascl_acsummaryfresca_samplefresca', 'jettiprojectfiletoquepalascl_rt1', 'jettiprojectfiletoquepalascl_rt1r', 'jettiprojectfiletoquepalascl_rt2', 'jettiprojectfiletoquepalascl_rt3', 'jettiprojectfiletoquepalascl_rt3r', 'jettiprojectfiletoquepalascl_rt7', 'jettiprojectfiletoquepalascl_rt8', 'jettiprojectfiletoquepalascl_samplefresca', 'jettiprojectfiletoquepalascl_samplefresca_rt1', 'jettiprojectfiletoquepalascl_samplefresca_rt11', 'jettiprojectfiletoquepalascl_samplefresca_rt1r', 'jettiprojectfiletoquepalascl_samplefresca_rt2', 'jettiprojectfiletoquepalascl_samplefresca_rt3', 'jettiprojectfiletoquepalascl_samplefresca_rt3r', 'jettiprojectfiletoquepalascl_samplefresca_rt7', 'jettiprojectfiletoquepalascl_samplefresca_rt8', 'jettiprojectfiletoquepalascl_tblmodalstoquepalafresca_samplefresca', 'jettiprojectfiletoquepalascl_tblrt1', 'jettiprojectfiletoquepalascl_tblrt1r', 'jettiprojectfiletoquepalascl_tblrt2', 'jettiprojectfiletoquepalascl_tblrt3', 'jettiprojectfiletoquepalascl_tblrt3r', 'jettiprojectfiletoquepalascl_tblrt7', 'jettiprojectfiletoquepalascl_tblrt8'],
    'jettiprojectfile_zaldivar': ['jettiprojectfilezaldivarscl_acsummary', 'jettiprojectfilezaldivarscl_acsummary_samplezaldivar', 'jettiprojectfilezaldivarscl_rt1', 'jettiprojectfilezaldivarscl_rt2', 'jettiprojectfilezaldivarscl_samplezaldivar', 'jettiprojectfilezaldivarscl_samplezaldivar_rt1', 'jettiprojectfilezaldivarscl_samplezaldivar_rt2', 'jettiprojectfilezaldivarscl_tblmodalszaldivar_samplezaldivar', 'jettiprojectfilezaldivarscl_tblrt1', 'jettiprojectfilezaldivarscl_tblrt2'],
    #'rea_256____013jettiprojectfile': ['rea_256____013jettiprojectfile\nName:_project_sample_id_raw,_dtype:_object'],
    #'rea_33____007bjettiprojectfile': ['rea_33____007bjettiprojectfiletiger\n34____007bjettiprojectfiletiger\n35____007bjettiprojectfiletiger\n36____007bjettiprojectfiletiger\n37____007bjettiprojectfiletiger\n38____007bjettiprojectfiletiger\nName:_project_sample_id_raw,_dtype:_object'],
}



# Function to map project_col_id_raw to project_sample_id
def map_project_sample_id(project_sample_id_raw):
    for project_sample_id, raw_ids in modified_keys_for_project_sample_id.items():
        if project_sample_id_raw in raw_ids:
            return project_sample_id
    return None  # or np.nan if you prefer

# List of your dataframes
dataframes = [
    df_ac_summary_ids,
    df_chemchar_ids,
    df_mineralogy_modals_ids,
    df_reactors_conditions_ids,
    df_reactors_summary_ids,
    df_reactors_detailed_ids
]
df_ac_summary_ids[df_ac_summary_ids['project_sample_id_raw'] == '030jettiprojectfile_acsummarycpy']
df_reactors_summary_ids[df_reactors_summary_ids['project_sample_id_raw'] == '030jettiprojectfile_cpy_rt1']
df_reactors_summary_ids[df_reactors_summary_ids['project_sample_id_raw'] == '031jettiprojectfile_antcomp_rt4']

# Apply the mapping to each dataframe
for df in dataframes:
    df['project_sample_id'] = df['project_sample_id_raw'].apply(map_project_sample_id)

df_reactors_summary_ids[df_reactors_summary_ids['project_sample_id'] == '030jettiprojectfile_cpy']
df_reactors_summary_ids[df_reactors_summary_ids['project_sample_id'] == '031jettiprojectfile_sample']

df_reactors_summary_ids[df_reactors_summary_ids['project_sample_id'] == '026jettiprojectfile_secondarysulfide']

#====== SPECIAL TREATMENT FOR ELEPHANT (LINKED TO THE ONE BEFORE)
df_mineralogy_modals_ids.loc[df_mineralogy_modals_ids['project_sample_id_raw'] == '007jettiprojectfile_elephant', 'project_sample_id'] = '007jettiprojectfile_elephant'

#====== SPECIAL TREATMENT FOR 011 RM CRUSHED
rm_sample_id = '011jettiprojectfile_rm'
rm_crushed_sample_id = '011jettiprojectfile_rm_crushed'
rm_crushed_chemchar_raw_ids = [
    '011jettiprojectfilecrushed_rm2024',
    '011jettiprojectfilecrushed_rm2024duplicate',
]

# RM2024 has its own chemical characterization rows, but they are not covered by the
# generic project_sample_id mapping dictionary.
df_chemchar_ids.loc[
    df_chemchar_ids['project_sample_id_raw'].isin(rm_crushed_chemchar_raw_ids),
    'project_sample_id'
] = rm_crushed_sample_id

# Mineralogy for RM2024 should inherit the original 011 RM mineralogy. The mapping
# function is one-to-one, so duplicate those rows explicitly for the crushed sample.
if not (df_mineralogy_modals_ids['project_sample_id'] == rm_crushed_sample_id).any():
    df_mineralogy_modals_rm = df_mineralogy_modals_ids[
        df_mineralogy_modals_ids['project_sample_id'] == rm_sample_id
    ].copy()
    if not df_mineralogy_modals_rm.empty:
        df_mineralogy_modals_rm.loc[:, 'project_sample_id'] = rm_crushed_sample_id
        df_mineralogy_modals_rm.loc[:, 'project_name'] = '011 Jetti Project File-Crushed'
        df_mineralogy_modals_ids = pd.concat(
            [df_mineralogy_modals_ids, df_mineralogy_modals_rm],
            ignore_index=True,
        )

# Elephant II PQ mineralogy is stored under the crushed row label and needs the
# downstream sample id used by the reactor/characterization exports.
elephant_pq_sample_id = 'jettifile_elephant_pq'
if not (df_mineralogy_modals_ids['project_sample_id'] == elephant_pq_sample_id).any():
    df_mineralogy_elephant_pq = df_mineralogy_modals_ids[
        (df_mineralogy_modals_ids['project_name'] == 'Jetti File - Elephant II Ver 2 PQ') &
        (df_mineralogy_modals_ids['sample'] == 'Crushed')
    ].copy()
    if not df_mineralogy_elephant_pq.empty:
        df_mineralogy_elephant_pq.loc[:, 'project_sample_id'] = elephant_pq_sample_id
        df_mineralogy_modals_ids = pd.concat(
            [df_mineralogy_modals_ids, df_mineralogy_elephant_pq],
            ignore_index=True,
        )

#%% copied from the load of the datasets
df_ac_summary_ids.to_csv(OUTPUT_DIR + '/dataset_acid_consumption_summary_summaries_with_id.csv', index=False)
df_chemchar_ids.to_csv(OUTPUT_DIR + '/dataset_characterization_summary_with_id.csv', index=False)
df_mineralogy_modals_ids.to_csv(OUTPUT_DIR + '/dataset_mineralogy_summary_modals_with_id.csv', index=False)
df_reactors_conditions_ids.to_csv(OUTPUT_DIR + '/dataset_reactors_conditions_with_id.csv', index=False)
df_reactors_summary_ids.to_csv(OUTPUT_DIR + '/dataset_reactor_summary_summaries_with_id.csv', index=False)
df_reactors_detailed_ids.to_csv(OUTPUT_DIR + '/dataset_reactor_summary_detailed_with_id.csv', index=False)

df_mineralogy_modals_ids.loc[df_mineralogy_modals_ids['project_name'] == '026 Jetti Project File']
 
#%% Check
df_mineralogy_modals_ids[df_mineralogy_modals_ids['project_name'] == '020 Jetti Project File Hypogene_Supergene']
list(df_mineralogy_modals_ids['project_sample_id'])
# %%

# CHOOSING DATA TO MAP AND POPULATE THE DATASET

df_ac_filtered = df_ac_summary_ids[['project_name', 'start_cell', 'project_sample_id', 'test_id', 'target_ph', 'h2so4_kg_t']].copy()
df_ac_filtered = df_ac_filtered[df_ac_filtered['project_sample_id'].notnull()].copy()
# transpose dataframe and get one row per project_sample_id
df_ac_filtered = df_ac_filtered.groupby(['project_sample_id']).agg(lambda x: x.tolist()).reset_index()
# calculate average ig h2so4_kg_t and add it to the dataframe
df_ac_filtered['avg_h2so4_kg_t'] = df_ac_filtered['h2so4_kg_t'].apply(lambda x: np.mean(x))

df_chemchar_filtered = df_chemchar_ids[df_chemchar_ids['project_sample_id_raw'].notnull()].copy()
cols_to_include_chemchar = ['cu_%', 'cu_seq_h2so4_%', 'cu_seq_nacn_%', 'cu_seq_a_r_%', 'fe_%', 'as_%', 'ba_%', 'be_%', 'bi_g_t', 'ca_%', 'co3_%', 'co_g_t', 'cr_%', 'li_g_t', 'mg_g_ton', 'mn_%', 'mo_%', 'na_%', 'ni_g_t', 'sb_g_t', 'p_%', 'pb_g_t', 'sr_%', 'ti_%', 'v_%', 'y_g_t', 'zn_%']
df_chemchar_filtered = df_chemchar_filtered[['project_name', 'analyte_units', 'project_sample_id'] + cols_to_include_chemchar].copy()
# choose and keep only first row of each project_sample_id so have 1 row of data 
df_chemchar_filtered = df_chemchar_filtered[df_chemchar_filtered['project_sample_id'].notnull()]
df_chemchar_filtered.drop_duplicates(subset=['project_sample_id'], keep='first', inplace=True)

df_mineralogy_modals_filtered = df_mineralogy_modals_ids[df_mineralogy_modals_ids['project_sample_id'].notnull()]
df_mineralogy_modals_filtered = df_mineralogy_modals_filtered.drop_duplicates(subset=['project_sample_id'], keep='first')
df_mineralogy_modals_filtered.drop(columns=['sample', 'sheet_name', 'index', 'start_cell', 'project_sample_id_raw'], inplace=True)
df_mineralogy_modals_filtered.fillna(0, inplace=True)
df_mineralogy_modals_ids.loc[df_mineralogy_modals_ids['project_name'] == '026 Jetti Project File']


df_mineralogy_modals_filtered[df_mineralogy_modals_filtered['project_name'] == '020 Jetti Project File Hypogene_Supergene']


print(df_reactors_detailed_ids[df_reactors_detailed_ids['project_sample_id'].isnull()][['project_sample_id_raw', 'project_sample_id', 'project_name']].groupby('project_sample_id_raw').first())

df_reactors_detailed_filtered = df_reactors_detailed_ids[df_reactors_detailed_ids['project_sample_id'].notnull()].copy()
df_reactors_detailed_filtered.drop(columns=['sheet_name', 'catalyzed', 'project_sample_id_raw'], inplace=True)
df_reactors_detailed_filtered['time_(day)'] = df_reactors_detailed_filtered['time_(day)'].fillna(df_reactors_detailed_filtered['time_(days)']).astype(float)
df_reactors_detailed_filtered['ph'] = df_reactors_detailed_filtered['ph'].fillna(df_reactors_detailed_filtered['ph_before_acid_addition_(-logh+)']).fillna(df_reactors_detailed_filtered['ph_before_acid_addition'])
df_reactors_detailed_filtered['adjusted_ph'] = df_reactors_detailed_filtered['adjusted_ph'].fillna(df_reactors_detailed_filtered['adjuted_ph_(-logh+)'])
df_reactors_detailed_filtered['orp_(mv)'] = df_reactors_detailed_filtered['orp_(mv)'].fillna(df_reactors_detailed_filtered['orp(mv)']).fillna(df_reactors_detailed_filtered['orp_(mv_she)'] - 223.0)
df_reactors_detailed_filtered = df_reactors_detailed_filtered[['project_name', 'project_sample_id', 'start_cell', 'time_(day)', 'temp_(c)', 'ph', 'orp_(mv)', 'cumulative_catalyst_(kg_t)', 'cumulative_h2so4_(kg_t)', 'reactor_tare_(g)', 'solids_mass_(g)', 'solution_mass_(g)', 'solution_volume_(l)', 'cu_extraction_actual_(%)']] #, 'fe_extraction_actual_(%)']] fe_extraction_actual_ too many negative values
df_reactors_detailed_filtered = df_reactors_detailed_filtered[df_reactors_detailed_filtered['time_(day)'].notnull()].copy()
df_reactors_detailed_filtered['start_cell'] = df_reactors_detailed_filtered['start_cell'].apply(lambda x: x.split('-')[1])


df_reactors_summary_filtered = df_reactors_summary_ids[df_reactors_summary_ids['project_sample_id'].notnull()].copy()
df_reactors_summary_filtered = df_reactors_summary_filtered[['project_name', 'project_sample_id', 'test_id', 'catalyst_dose_(mg_l)', 'catalyst_type', 'lixiviant', 'catalyst_addition_day', 'ph_target']].copy()

df_reactors_merged = pd.merge(
    df_reactors_detailed_filtered,
    df_reactors_summary_filtered[['project_sample_id', 'test_id', 'catalyst_dose_(mg_l)', 'catalyst_type', 'lixiviant', 'catalyst_addition_day', 'ph_target']],
    left_on=['project_sample_id', 'start_cell'],  # Columns from df_reactors_detailed_filtered
    right_on=['project_sample_id', 'test_id'],   # Columns from df_reactors_summary_filtered
    how='left'  # Use 'left' join to keep all rows from df_reactors_detailed_filtered
)
df_reactors_merged = df_reactors_merged.drop(columns=['test_id'])
df_reactors_merged = df_reactors_merged[df_reactors_merged['time_(day)'] > 0.5].copy() # keep only data after 0.5 days

# Special Treatments for some reactors
df_reactors_merged.loc[
    df_reactors_merged.loc[
        (df_reactors_merged['project_sample_id'] == '012jettiprojectfile_incremento') & 
        (df_reactors_merged['start_cell'] == 'RT_B')
    ].iloc[-3].name, 
    'cu_extraction_actual_(%)'
] = np.nan

df_reactors_merged.loc[
    df_reactors_merged.loc[
        (df_reactors_merged['project_sample_id'] == '012jettiprojectfile_quebalix') & 
        (df_reactors_merged['start_cell'] == 'RT_A')
    ].iloc[-3].name, 
    'cu_extraction_actual_(%)'
] = np.nan

df_reactors_merged.loc[
    df_reactors_merged.loc[
        (df_reactors_merged['project_sample_id'] == '030jettiprojectfile_cpy') & 
        (df_reactors_merged['start_cell'] == 'RT_4')
    ].iloc[-1].name, 
    'cu_extraction_actual_(%)'
] = np.nan

df_reactors_merged.to_csv(OUTPUT_DIR + '/dataset_reactor_summary_detailed_with_id_filtered.csv')

df_qemscan_filtered = df_qemscan_compilation[df_qemscan_compilation['project_sample_id_raw'].notnull()].copy()
# choose only the rows that have the number 106 on 'sample' column
df_qemscan_filtered = df_qemscan_filtered[df_qemscan_filtered['sample'].str.contains('106')].copy() # WARNING: 011 PROJECT HAS NO 106, SO IT IS NOT INCLUDED
df_qemscan_filtered.drop_duplicates(subset=['project_sample_id'], keep='first', inplace=True)
df_qemscan_filtered = df_qemscan_filtered[['project_sample_id_raw', 'project_sample_id', 'project_sample_condition_id', 'cpy_+50%_exposed_norm', 'cpy_locked_norm', 'cpy_associated_norm']]
df_qemscan_filtered.rename(columns={'project_sample_id_raw': 'project_sample_id', 'project_sample_id': 'project_sample_id_original'}, inplace=True)
df_qemscan_filtered.to_csv(OUTPUT_DIR + '/dataset_qemscan_with_id_filtered.csv', index=False)

df_mineralogy_modals_ids[df_mineralogy_modals_ids['project_name'] == '020 Jetti Project File Hypogene_Supergene']

df_mineralogy_modals_ids.loc[df_mineralogy_modals_ids['project_sample_id'] == '026jettiprojectfile_secondarysulfide', ['project_sample_id', 'start_cell', 'chalcopyrite', 'pyrite']]

#%% CHARACTERIZE EACH TESTWORK

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Define the copper recovery curve function: a*(1-exp(-b*t))+c
def copper_recovery_curve(t, a, b, c):
    return a * (1 - np.exp(-b * t)) + c

def copper_recovery_curve_P(t, P, b, c):
    # P is the final plateau (so we can directly bound P ≤ 99)
    return (P - c) * (1 - np.exp(-b * t)) + c  # P is the plateau, b is the rate, c is the initial value

def copper_recovery_curve_4param(t, a1, b1, a2, b2):
    return a1 * (1 - np.exp(-b1 * t)) + a2 * (1 - np.exp(-b2 * t))

def order_kinetic_parameter_pairs(a1, b1, a2, b2):
    # Keep kinetic pairs together and enforce fast component first (higher b).
    if b2 > b1:
        return a2, b2, a1, b1
    return a1, b1, a2, b2


'''
# Function to fit curve parameters with constraints (a and b positive)
def fit_curve_parameters(time_data, extraction_data):
    # Convert data to numeric, handling potential errors
    time_data = pd.to_numeric(time_data, errors='coerce')
    extraction_data = pd.to_numeric(extraction_data, errors='coerce')
    
    # Drop NaN values
    mask = ~(np.isnan(time_data) | np.isnan(extraction_data))
    time_data = time_data[mask]
    extraction_data = extraction_data[mask]
    
    # If not enough data points, return None
    if len(time_data) < 3:
        return None, None, None, None, None
    
    try:
        # Initial parameter guesses
        p0 = [max(extraction_data), 0.1, min(extraction_data)]
        
        # Parameter bounds (a and b positive, c any value)
        bounds = ([0, 1e-4, -60], [99, np.inf, 60])
        
        # Fit the curve
        popt, pcov = curve_fit(copper_recovery_curve, time_data, extraction_data, 
                              p0=p0, bounds=bounds, maxfev=10000)
        

        # Calculate R-squared
        residuals = extraction_data - copper_recovery_curve(time_data, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((extraction_data - np.mean(extraction_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean(residuals**2))
        
        return popt[0], popt[1], popt[2], r_squared, rmse
    
    except Exception as e:
        print(f"Fitting error: {e}")
        return None, None, None, None, None
'''

def fit_curve_parameters(
    time_data,
    extraction_data,
    Tmax=None, # None for not use, usually 200 days
    frac=0.99,
    epsilon=1e-4,
    last_n=5,
    last_points_weight=(2.0, 10.0),
    jitter_starts=0,
    jitter_scale=0.3,
    random_state=None
):
    time_data = np.asarray(pd.to_numeric(time_data, errors='coerce'))
    extraction_data = np.asarray(pd.to_numeric(extraction_data, errors='coerce'))
    mask = ~(np.isnan(time_data) | np.isnan(extraction_data))
    time_data, extraction_data = time_data[mask], extraction_data[mask]
    if len(time_data) < 3:
        return None, None, None, None, None, None

    b_min = 0.001
    b_max = 10.0
    a_min = 2.5
    a_max = min(99, np.max(extraction_data) + 10)

    # Data-driven initial guesses
    plateau = np.percentile(extraction_data, 90)
    a_guess1 = plateau / 2
    a_guess2 = plateau / 2
    b_guess1 = 0.1
    b_guess2 = 0.01
    def clamp_a1(x):
        return float(np.clip(x, a_min * 1.2, a_max))

    def clamp_a2(x):
        return float(np.clip(x, a_min, a_max))

    def clamp_b(x):
        return float(np.clip(x, b_min, b_max))

    p0_list = [
        [clamp_a1(a_guess1), clamp_b(b_guess1), clamp_a2(a_guess2), clamp_b(b_guess2)],
        [clamp_a1(plateau), clamp_b(0.05), clamp_a2(a_min), clamp_b(0.01)],
        [clamp_a1(np.max(extraction_data)), clamp_b(0.05), clamp_a2(a_min), clamp_b(0.01)],
        [clamp_a1(plateau * 0.7), clamp_b(0.2), clamp_a2(plateau * 0.3), clamp_b(0.02)],
        [clamp_a1(plateau * 0.9), clamp_b(0.03), clamp_a2(plateau * 0.1), clamp_b(0.005)],
        [clamp_a1(10.0), clamp_b(0.01), clamp_a2(10.0), clamp_b(0.01)]
    ]

    jitter_starts = int(max(0, jitter_starts))
    if jitter_starts > 0:
        rng = np.random.default_rng(random_state)
        if isinstance(jitter_scale, (tuple, list, np.ndarray)):
            if len(jitter_scale) == 2:
                a_scale, b_scale = float(jitter_scale[0]), float(jitter_scale[1])
                scales = [a_scale, b_scale, a_scale, b_scale]
            elif len(jitter_scale) == 4:
                scales = [float(s) for s in jitter_scale]
            else:
                scales = [float(jitter_scale[0])] * 4
        else:
            scales = [float(jitter_scale)] * 4
        scales = [max(0.0, s) for s in scales]
        jittered = []
        for base in p0_list:
            for _ in range(jitter_starts):
                factors = np.exp(rng.normal(0.0, scales))
                jittered.append([
                    clamp_a1(base[0] * factors[0]),
                    clamp_b(base[1] * factors[1]),
                    clamp_a2(base[2] * factors[2]),
                    clamp_b(base[3] * factors[3])
                ])
        p0_list.extend(jittered)

    bounds = [
        (a_min, a_max),
        (b_min, b_max),
        (a_min, a_max),
        (b_min, b_max)
    ]

    constraints = [
        {'type': 'ineq', 'fun': lambda p: 99 - p[0] - p[2]},        # a1 + a2 <= 99
        # {'type': 'ineq', 'fun': lambda p: p[1] - p[3] * 1.05}              # b1 - b2 >= 0 (b1 greater than b2 by 5%)
    ]

    last_n = int(max(0, last_n))
    weights = np.ones_like(time_data, dtype=np.float64)
    if last_n and last_points_weight is not None:
        if isinstance(last_points_weight, (tuple, list, np.ndarray)) and len(last_points_weight) == 2:
            weight_start, weight_end = last_points_weight
        else:
            weight_start, weight_end = 1.0, last_points_weight
        weight_start = min(max(float(weight_start), 1.0), 10.0)
        weight_end = min(max(float(weight_end), 1.0), 10.0)
        if max(weight_start, weight_end) > 1.0:
            last_indices = np.argsort(time_data)[-min(last_n, len(time_data)):]
            last_indices = last_indices[np.argsort(time_data[last_indices])]
            weights[last_indices] = np.linspace(weight_start, weight_end, num=len(last_indices))
    '''
    def objective(p):
        resid = extraction_data - copper_recovery_curve_4param(time_data, *p)
        resid = resid * weights
        loss = np.sum(resid**2)
        # Penalize solutions that do not reach the asymptote by Tmax
        if Tmax is not None and np.isfinite(Tmax):
            asymptote = p[0] + p[2]
            pred_T = copper_recovery_curve_4param(Tmax, *p)
            # Require pred_T to be very close to asymptote (epsilon) or at least frac*asymptote
            target = max(frac * asymptote, asymptote - epsilon)
            deficit = max(0.0, target - pred_T)
            scale = np.nanstd(extraction_data)
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            penalty_weight = (scale ** 2) * len(time_data) * 100.0
            loss += penalty_weight * (deficit ** 2)
        return loss
    '''
    # before objective, once:
    last_idx = np.argsort(time_data)[-min(last_n, len(time_data)):]
    t_last = time_data[last_idx]
    y_last = extraction_data[last_idx]
    tail_slope = np.polyfit(t_last, y_last, 1)[0]  # % per day

    apply_tmax_penalty = tail_slope < 0.05 # tune threshold
    apply_tmax_penalty_2 = y_last.mean() >= 80.0

    def objective(p): # Make the Tmax penalty relative to asymptote (dimensionless)
        pred = copper_recovery_curve_4param(time_data, *p)
        resid = (extraction_data - pred) * weights
        loss = np.sum(resid**2)

        if apply_tmax_penalty and apply_tmax_penalty_2 and Tmax is not None and np.isfinite(Tmax):
            asym = p[0] + p[2]
            if asym > 0:
                pred_T = copper_recovery_curve_4param(Tmax, *p)
                target = frac * asym  # drop the epsilon logic; it makes it "almost hard"
                deficit = max(0.0, target - pred_T) / asym   # relative deficit in [0,1]

                # much smaller / interpretable lambda
                lam = 10.0 * np.sum(weights)  # start here; tune 1..100
                loss += lam * (deficit ** 2)
        return loss

    best_res = None
    best_loss = np.inf

    for p0 in p0_list:
        try:
            res = minimize(
                objective,
                p0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 10000}
            )
            if res.success and objective(res.x) < best_loss:
                best_res = res
                best_loss = objective(res.x)
        except Exception as e:
            continue

    if best_res is not None and best_res.success:
        popt = best_res.x
        a1_fit, b1_fit, a2_fit, b2_fit = popt
        a1_fit, b1_fit, a2_fit, b2_fit = order_kinetic_parameter_pairs(a1_fit, b1_fit, a2_fit, b2_fit)
        resid = extraction_data - copper_recovery_curve_4param(time_data, *popt)
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((extraction_data - extraction_data.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        rmse = np.sqrt(np.mean(resid**2))
        # If the fit is identical to the initial guess, treat as failure
        if np.allclose(popt, p0_list[0], atol=1e-3):
            print("⚠️ Fit stuck at initial guess, try different initializations or check data.")
            return None, None, None, None, None, None
        return a1_fit, b1_fit, a2_fit, b2_fit, r2, rmse
    else:
        print("Fitting error: No successful fit found.")
        return None, None, None, None, None, None
    

# Main function to process the data
def process_reactor_data(file_path):
    print("Loading and processing data...")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Convert columns to numeric where needed
    numeric_columns = ['time_(day)', 'temp_(c)', 'orp_(mv)', 'ph', 'cumulative_catalyst_(kg_t)', 
                      'catalyst_dose_(mg_l)', 'cumulative_h2so4_(kg_t)', 'cu_extraction_actual_(%)'
                      ]
    
    for col in df.columns:
        if col in numeric_columns or 'extraction' in col.lower():
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create a list to store curve fitting results
    curve_fit_results = []
    
    # Create a list to store time series features
    time_series_features = []
    
    # Process each unique test (project_sample_id + start_cell combination)
    unique_tests = df.groupby(['project_sample_id', 'start_cell'])
    
    print(f"Processing {len(unique_tests)} unique tests...")
    
    for (project_id, reactor_id), test_data in unique_tests:
        # Sort by time
        test_data = test_data.sort_values('time_(day)')
        
        # Extract time and copper extraction data
        time_data = test_data['time_(day)']
        cu_extraction = test_data['cu_extraction_actual_(%)']
        
        # Fit curve parameters
        a1, b1, a2, b2, r_squared, rmse = fit_curve_parameters(time_data, cu_extraction)
        
        # Calculate additional time series features if enough data points
        if len(test_data) >= 3:
            # Calculate statistics for varying columns
            varying_cols = ['temp_(c)', 'ph', 'orp_(mv)', 'cumulative_catalyst_(kg_t)', 
                            'cumulative_h2so4_(kg_t)', 'cu_extraction_actual_(%)', 
                            # 'fe_extraction_actual_(%)', 
                            # 'catalyst_dose_(mg_l)'
                            ]
            
            features = {
                'project_sample_id': project_id,
                'start_cell': reactor_id,
                'num_data_points': len(test_data),
                'max_time_day': time_data.max(),
                'a1_param': a1,
                'b1_param': b1,
                'a2_param': a2,
                'b2_param': b2,
                'r_squared': r_squared,
                'rmse': rmse,
                'final_cu_extraction': test_data['cu_extraction_actual_(%)'].iloc[-1] if not test_data['cu_extraction_actual_(%)'].iloc[-1] is np.nan else None,
                'extraction_rate_30d': None  # Will calculate later if a and b are valid
            }
            
            # Calculate predicted extraction at 30 days if parameters are valid
            if a1 is not None and b1 is not None and a2 is not None and b2 is not None:
                features['extraction_rate_30d'] = copper_recovery_curve_4param(30, a1, b1, a2, b2)
                
                # Add curve fit results
                curve_fit_results.append({
                    'project_sample_id': project_id,
                    'start_cell': reactor_id,
                    'a1': a1,
                    'b1': b1,
                    'a2': a2,
                    'b2': b2,
                    'r_squared': r_squared,
                    'rmse': rmse
                })
            
            # Add statistics for varying columns
            for col in varying_cols:
                col_data = pd.to_numeric(test_data[col], errors='coerce')
                if not col_data.empty and not col_data.isna().all():
                    features[f'{col}_mean'] = col_data.mean()
                    features[f'{col}_std'] = col_data.std()
                    # features[f'{col}_min'] = col_data.min()
                    features[f'{col}_max'] = col_data.max()
                    
                    # Calculate slope (rate of change)
                    if len(col_data) > 1 and not time_data.isna().any():
                        valid_mask = ~col_data.isna()
                        if valid_mask.sum() > 1:
                            x = time_data[valid_mask].values
                            y = col_data[valid_mask].values
                            if len(x) > 1:  # Ensure we have at least 2 points
                                slope = np.polyfit(x, y, 1)[0]
                                features[f'{col}_slope'] = slope
            
            # Add constant features (take the first non-NaN value)
            constant_cols = [col for col in df.columns if col not in varying_cols and 
                            col not in ['project_sample_id', 'start_cell', 'time_(day)']]
            
            for col in constant_cols:
                non_nan_values = test_data[col].dropna()
                if not non_nan_values.empty:
                    features[col] = non_nan_values.iloc[0]
            
            time_series_features.append(features)
    
    # Create DataFrame from time series features
    features_df = pd.DataFrame(time_series_features)
    
    # Create the first dataframe: 1 row per project_sample_id
    df_per_project = features_df.groupby('project_sample_id').agg({
        'a1_param': 'mean',
        'b1_param': 'mean',
        'a2_param': 'mean',
        'b2_param': 'mean',
        'r_squared': 'mean',
        'rmse': 'mean',
        'final_cu_extraction': 'mean',
        'extraction_rate_30d': 'mean',
        'num_data_points': 'sum',
        'max_time_day': 'max'
    }).reset_index()
    
    # Add additional aggregated statistics for each project_sample_id
    for col in features_df.columns:
        if col not in ['project_sample_id', 'start_cell', 'a1_param', 'b1_param', 'a2_param', 'b2_param',
                      'r_squared', 'rmse', 'final_cu_extraction', 'extraction_rate_30d', 
                      'num_data_points', 'max_time_day']:
            try:
                agg_stats = features_df.groupby('project_sample_id')[col].agg(['mean', 'std', 'min', 'max'])
                for stat in ['mean', 'std', 'min', 'max']:
                    if not agg_stats[stat].isna().all():
                        df_per_project[f'{col}_{stat}'] = agg_stats[stat].values
            except:
                pass
    
    # Create the second dataframe: 1 row per project_sample_id and start_cell
    df_per_project_cell = features_df.copy()
    
    # Apply dimensionality reduction and clustering for characterization
    print("Applying characterization techniques...")
    
    # Function to apply PCA and clustering
    def apply_characterization(df, id_cols, prefix=''):
        # Select numeric columns for analysis
        print([col for col in df.columns if 'lixiviant' in col])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in id_cols]
        
        if len(numeric_cols) < 2:
            print("Not enough numeric columns for characterization")
            return df
        
        # Create a copy of the dataframe with only numeric columns
        df_numeric = df[numeric_cols].copy()

        # Fill NaN values with column means
        df_numeric = df_numeric.fillna(df_numeric.mean())
        
        # Skip if we still have NaN values
        if df_numeric.isna().any().any():
            print("Warning: NaN values remain after filling with means")
            df_numeric = df_numeric.fillna(0)
        
        # Handle categorical columns
        # add these cat columns at df_per_project and df_per_project_cell

        categorical_cols = ['catalyst_type', 'lixiviant']
        df_categorical = df[categorical_cols].copy()

        # One-hot encode the specified categorical columns
        encoder = OneHotEncoder(drop='first')  # Drop first to avoid multicollinearity
        encoded_categorical = encoder.fit_transform(df_categorical)
        
        # Create a DataFrame for the encoded categorical data
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df_categorical.index
        )
        
        # Concatenate numeric and encoded categorical data
        df_numeric = pd.concat([df_numeric, encoded_categorical_df], axis=1)
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric)
        
        # Apply PCA
        n_components = min(5, len(numeric_cols))
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Add PCA components to the dataframe
        for i in range(n_components):
            df[f'{prefix}pca_component_{i+1}'] = pca_result[:, i]
        
        # Add explained variance as a feature
        df[f'{prefix}pca_explained_variance'] = np.round(np.sum(pca.explained_variance_ratio_*100), 1)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=min(5, len(df)), random_state=42)
        df[f'{prefix}cluster'] = kmeans.fit_predict(scaled_data)
        
        return df
    
    # Apply characterization to both dataframes
    # df_per_project = apply_characterization(df_per_project, ['project_sample_id'], 'project_')
    # df_per_project_cell = apply_characterization(df_per_project_cell, ['project_sample_id', 'start_cell'], 'reactor_')
    
    # Save results to CSV
    df_per_project.to_csv(OUTPUT_DIR + '/processed_reactors_per_project.csv', index=False)
    df_per_project_cell.to_csv(OUTPUT_DIR + '/processed_reactors_per_test.csv', index=False)
    
    curve_results_df = pd.DataFrame(curve_fit_results)
    curve_results_df.to_csv(OUTPUT_DIR + '/curve_fitting_results.csv', index=False)
    # Create a dataframe with curve fitting results
    
    # Generate some visualizations
    print("Generating visualizations...")
    
    # Plot distribution of curve parameters
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 2, 1)
    plt.hist(curve_results_df['a1'], bins=20, alpha=0.7)
    plt.title('Distribution of Parameter a1')
    plt.xlabel('a1 value')
    plt.ylabel('Frequency')
    
    plt.subplot(3, 2, 2)
    plt.hist(curve_results_df['b1'], bins=20, alpha=0.7)
    plt.title('Distribution of Parameter b1')
    plt.xlabel('b1 value')
    plt.ylabel('Frequency')
    
    plt.subplot(3, 2, 3)
    plt.hist(curve_results_df['a2'], bins=20, alpha=0.7)
    plt.title('Distribution of Parameter a2')
    plt.xlabel('a2 value')
    plt.ylabel('Frequency')

    plt.subplot(3, 2, 4)
    plt.hist(curve_results_df['b2'], bins=20, alpha=0.7)
    plt.title('Distribution of Parameter b2')
    plt.xlabel('b2 value')
    plt.ylabel('Frequency')
    
    plt.subplot(3, 2, 5)
    plt.hist(curve_results_df['r_squared'], bins=20, alpha=0.7)
    plt.title('Distribution of R-squared')
    plt.xlabel('R-squared value')
    plt.ylabel('Frequency')
    
    plt.subplot(3, 2, 6)
    plt.hist(curve_results_df['rmse'], bins=20, alpha=0.7)
    plt.title('Distribution of R-squared')
    plt.xlabel('RMSE')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + '/parameter_distributions.png')
    plt.close('all')
    
    # Plot example curves for a few samples
    
    for j, sample_unique in enumerate(curve_results_df['project_sample_id'].unique()):
        plt.figure(figsize=(15, 20))
        # Filter the dataframe for the current sample
        curve_results_df_plot = curve_results_df[curve_results_df['project_sample_id'] == sample_unique].copy()

        # Select a few samples to plot
        if len(curve_results_df_plot) > 15:
            samples_to_plot = curve_results_df_plot.sample(min(15, len(curve_results_df_plot))).sort_values('start_cell', ascending=True)
        else:
            samples_to_plot = curve_results_df_plot
        
        # x lim for all plot of the sample
        xlim_sample = max(df[(df['project_sample_id'] == sample_unique)]['time_(day)']) + 1
        
        for i, (_, sample) in enumerate(samples_to_plot.iterrows()):
            # Get the original data
            sample_data = df[(df['project_sample_id'] == sample_unique) & 
                            (df['start_cell'] == sample['start_cell'])]
            
            # Sort by time
            sample_data = sample_data.sort_values('time_(day)')

            # fill catalyst_type missing values, control for 0 and nan in catalyst_dose_(mg_l),  and others with 100-CA
            sample_data['catalyst_type'] = sample_data.apply(
                lambda row: 'Control' if (pd.isna(row['catalyst_type']) and (pd.isna(row['catalyst_dose_(mg_l)']) or row['catalyst_dose_(mg_l)'] == 0)) else 
                            ('100-CA' if pd.isna(row['catalyst_type']) else row['catalyst_type']),
                axis=1
            )
            
            # Extract time and copper extraction data
            time_data = pd.to_numeric(sample_data['time_(day)'], errors='coerce')
            cu_extraction = pd.to_numeric(sample_data['cu_extraction_actual_(%)'], errors='coerce')
            catalyst = sample_data['catalyst_type'].iloc[0] # if not sample_data['catalyst_type'].isna().all() else 'Control'
            lixiviant = sample_data['lixiviant'].iloc[0]
            dose = sample_data['catalyst_dose_(mg_l)'].iloc[0]
            mean_temperature = sample_data['temp_(c)'].mean()

            
            # Drop NaN values
            mask = ~(np.isnan(time_data) | np.isnan(cu_extraction))
            time_data = time_data[mask]
            cu_extraction = cu_extraction[mask]
            
            if len(time_data) < 3:
                continue
            
            # Generate curve with fitted parameters
            t_curve = np.linspace(0, max(time_data), 100)
            cu_curve = copper_recovery_curve_4param(t_curve, sample['a1'], sample['b1'], sample['a2'], sample['b2'])
            
            plt.subplot(5, 3, i+1)
            plt.scatter(time_data, cu_extraction, label='Actual data')
            plt.plot(t_curve, cu_curve, 'r-', label='Fitted curve')
            plt.title(f"Sample {sample['project_sample_id']}, Cell {sample['start_cell']}\nCatalyst: {catalyst}, Lixiviant: {lixiviant}\nDose: {dose} mg/L, Mean Temp: {mean_temperature:.1f}°C")
            plt.xlabel('Time (days)')
            plt.ylabel('Cu Extraction (%)')
            plt.ylim(0, 100)
            plt.xlim(0, xlim_sample)
            plt.legend()
            
            # Add parameters to the plot
            plt.annotate(rf"$a_1={sample['a1']:.1f}, b_1={sample['b1']:.3f}$"
                         f"\n"
                         rf"$a_2={sample['a2']:.1f}, b_2={sample['b2']:.3f}$"
                         f"\n"
                         f"R²={sample['r_squared']:.2f}",
                        xy=(0.65, 0.05), xycoords='axes fraction')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + f'/example_fitted_curves_{sample_unique}.png')
        plt.close('all')
        
    print("Processing complete!")
    print(f"Results saved to:")
    print("- dataframe_per_project.csv")
    print("- dataframe_per_project_cell.csv")
    print("- curve_fitting_results.csv")
    print("- parameter_distributions.png")
    print("- example_fitted_curves.png")
    
    return df_per_project, df_per_project_cell, curve_results_df

# Execute the function if this script is run directly
# file_path = 'upload/dataset_reactor_summary_detailed_with_id_filtered.csv'
df_per_project, df_per_project_cell, curve_results_df = process_reactor_data(FOLDER_PATH_LOAD_IDS + '/dataset_reactor_summary_detailed_with_id_filtered.csv')

df_per_project[df_per_project['project_sample_id'] == '030jettiprojectfile_cpy']
curve_results_df[curve_results_df['project_sample_id'] == '030jettiprojectfile_cpy']
df_per_project_cell[df_per_project_cell['project_sample_id'] == '030jettiprojectfile_cpy']

df_per_project_cell[df_per_project_cell['project_sample_id'] == '026jettiprojectfile_secondarysulfide']
df_per_project_cell.columns


# %% Merge with all other characterizations.

# Merge with acid consumption data per reactor (cell)
df_ac_filtered_tomerge = df_ac_filtered[['project_sample_id', 'avg_h2so4_kg_t']].copy()
df_ac_filtered_tomerge = df_ac_filtered_tomerge[df_ac_filtered_tomerge['project_sample_id'].notnull()]

# identify and print duplicates
duplicates = df_ac_filtered_tomerge[df_ac_filtered_tomerge.duplicated(subset=['project_sample_id'], keep=False)]
if not duplicates.empty:
    print("Duplicates found in df_ac_filtered_tomerge:")
    print(duplicates)
df_ac_filtered_tomerge.drop_duplicates(subset=['project_sample_id'], keep='first', inplace=True)

# add some missing data for the project_sample_id (Monse's Teams message Apr 25th)
new_row_003 = {'project_sample_id': '003jettiprojectfile_amcf', 'avg_h2so4_kg_t': 13.5}
# new_row_030 = {'project_sample_id': '030jettiprojectfile_cpy', 'avg_h2so4_kg_t': 10.0} # Check AC with monse
new_row_011crushed = {'project_sample_id': '011jettiprojectfile_rm_crushed', 'avg_h2so4_kg_t': 12.56}

df_ac_filtered_tomerge = pd.concat([df_ac_filtered_tomerge, pd.DataFrame([new_row_003])], ignore_index=True)
# df_ac_filtered_tomerge = pd.concat([df_ac_filtered_tomerge, pd.DataFrame([new_row_030])], ignore_index=True)
df_ac_filtered_tomerge = pd.concat([df_ac_filtered_tomerge, pd.DataFrame([new_row_011crushed])], ignore_index=True)

df_per_sample = df_ac_filtered_tomerge.copy()
df_per_sample[df_per_sample['project_sample_id'] == '030jettiprojectfile_cpy']

# Handle transormation of Copper Sequential test (cu seq):
df_chemchar_filtered['cu_seq_h2so4_norm%'] = df_chemchar_filtered['cu_seq_h2so4_%'] / df_chemchar_filtered[['cu_seq_h2so4_%', 'cu_seq_a_r_%', 'cu_seq_nacn_%']].sum(axis=1)
df_chemchar_filtered['cu_seq_a_r_norm%'] = df_chemchar_filtered['cu_seq_a_r_%'] / df_chemchar_filtered[['cu_seq_h2so4_%', 'cu_seq_a_r_%', 'cu_seq_nacn_%']].sum(axis=1)
df_chemchar_filtered['cu_seq_nacn_norm%'] = df_chemchar_filtered['cu_seq_nacn_%'] / df_chemchar_filtered[['cu_seq_h2so4_%', 'cu_seq_a_r_%', 'cu_seq_nacn_%']].sum(axis=1)


# Merge with characterization data
df_chemchar_filtered_tomerge = df_chemchar_filtered[['project_sample_id', 'cu_seq_h2so4_norm%', 'cu_seq_nacn_norm%', 'cu_seq_a_r_norm%'] + cols_to_include_chemchar].copy()
df_chemchar_filtered_tomerge = df_chemchar_filtered_tomerge[df_chemchar_filtered_tomerge['project_sample_id'].notnull()]

# fill nans with 0
df_chemchar_filtered_tomerge.fillna(0, inplace=True)
# identify and print duplicates
duplicates = df_chemchar_filtered_tomerge[df_chemchar_filtered_tomerge.duplicated(subset=['project_sample_id'], keep=False)]
if not duplicates.empty:
    print("Duplicates found in df_chemchar_filtered_tomerge:")
    print(duplicates)
# ensure match for all project_sample_id and print not matched sample_ids
not_matched = df_chemchar_filtered_tomerge[~df_chemchar_filtered_tomerge['project_sample_id'].isin(df_per_sample['project_sample_id'])]
if not not_matched.empty:
    print("Not matched project_sample_id in df_chemchar_filtered_tomerge:")
    print(list(not_matched['project_sample_id']))
df_chemchar_filtered_tomerge.drop_duplicates(subset=['project_sample_id'], keep='first', inplace=True)
df_per_sample = pd.merge(
    df_per_sample,
    df_chemchar_filtered_tomerge,
    on='project_sample_id',
    how='left'
)


#%% Merge with mineralogy data

df_mineralogy_modals_filtered_tomerge = df_mineralogy_modals_filtered.drop(columns=['origin', 'project_name']).copy()
df_mineralogy_modals_filtered_tomerge = df_mineralogy_modals_filtered_tomerge[df_mineralogy_modals_filtered_tomerge['project_sample_id'].notnull()]

df_mineralogy_modals_filtered_tomerge[df_mineralogy_modals_filtered_tomerge['project_sample_id'] == '020jettiprojectfile_hyp']

# Define mineral groupings

primary_copper_sulfides = [
    'chalcopyrite',
    'bornite',
    'enargite',
    'tennantite',
    'tetrahedrite',
    'luzonite',
    'cubanite',
    'other_cu_sulfides'
]


secondary_copper_sulfides = [
    'chalcocite',
    'digenite',
    'chalcocite_digenite',
    'covellite',
    'yarrowite',
    'anilite',
    'geerite',
    'spionkopite'
]


copper_oxides = [
    'cuprite',
    'tenorite',
    'native_copper',

    # carbonates
    'malachite',
    'azurite',
    'malachite_azurite',

    # silicates
    'chrysocolla',
    'dioptase',
    'plancheite',
    'shattuckite',

    # chlorides / hydroxides
    'atacamite',
    'paratacamite',
    'clinoatacamite',
    'brochantite',

    # phosphates
    'pseudomalachite',
    'turquoise',
    'libethenite'
]


mixed_copper_ores = [
    'cu_oxides_carbonates',
    'cu_bearing_clay',
    'cu_bearing_fe_ox_oh',
    'cu_bearing_silicates',
    'cu_wad',
    'cu_mn_wad',
    'other_copper',
    'other_cu_minerals'
]


# Group 1: Key Copper Sulfides
copper_sulfides = ['chalcopyrite', 'bornite', 'chalcocite', 'covellite', 'enargite', 
                   'chalcocite_digenite', 'enargite_tennantite', 'other_cu_sulfides', 'other_cu_minerals']

# Group 2: Secondary Copper Minerals
secondary_copper = ['other_copper', 'cuprite', 'brochantite', 'atacamite', 'chrysocolla', 
                    'cu_oxides_carbonates', 'native_copper', 'cu_bearing_clay', 'cu_bearing_fe_ox_oh', 'cu_bearing_silicates', 
                    'cu_wad', 'cu_mn_wad', 'malachite_azurite', 'pseudomalachite', 'turquoise', 'cubanite']

# Group 3: Acid-Generating Sulfides
acid_generating_sulfides = ['pyrite', 'molybdenite', 'other_sulfides']

# Group 4: Gangue Sulfides (base metal sulfides)
gangue_sulfides = ['sphalerite', 'galena']

# Group 5: Gangue Silicates (feldspars and related)
gangue_silicates = ['quartz', 'plagioclase', 'feldspar_albite', 'albite', 'orthoclase', 
                    'cana-plagioclases', 'ca-plagioclase', 'na_ca_plagioclase', 'na_plagioclase',  'ca_na-plagioclase', 
                    'plagioclase_feldspar', 'plagioclases', 'k-feldspar', 'other_silicates', 'amphibole_pyroxene', 
                    'amphibole', 'pyroxenes', 'sericite_muscovite', 'muscovite', 'biotite', 'chlorite', 
                    'clays_other_silicates', 'clays__other_silicates', 'tourmaline', 
                    'clays', 'fe_clay', 'pyrophyllite', 'montmorillonite', 'kaolinite', # Group 6
                    'biotite_phlogopite', 'muscovite_sericite', 'chlorites_smectites',
                    'epidote', 'epidote_group', 'actinolite', 'titanite', 'andalusite', 'sphene', # Group 7
                    'gypsum', 'jarosite', 'alunite', 'fe_sulphate_low_al_si_k', 'sulphates', # Group 8
                    'anhydrite_gypsum', 'gypsum_anhydrite', 'sulphur', 'alunite_jarosite', 'other_sulfates']
"""
# Group 6: Clays and Micas
clays_and_micas = ['clays', 'fe_clay', 'pyrophyllite', 'montmorillonite', 'kaolinite', 
                   'biotite_phlogopite', 'muscovite_sericite', 'chlorites_smectites']

# Group 7: Accessory Silicates / Other Silicates
accessory_silicates = ['epidote', 'epidote_group', 'actinolite', 'titanite', 'andalusite', 'sphene']

# Group 8: Sulfates
sulfates = ['gypsum', 'jarosite', 'alunite', 'fe_sulphate_low_al_si_k', 'sulphates',
            'anhydrite_gypsum', 'gypsum_anhydrite', 'sulphur', 'alunite_jarosite', 'other_sulfates']
"""
# Group 9: Iron Oxides (including generic 'oxides')
fe_oxides = ['oxides', 'fe_oxides', 'other_oxides', 'hematite', 'magnetite',
             'magnetite-hematite', 'hematite-magnetite', 'fe_oxides_hydroxides', 'fe_oxides_cu', 'limonite-cu']

# Group 10: Accessory & Miscellaneous

# Group 11: Carbonates / acid consuming
carbonates = ['carbonates', 'calcite', 'dolomite', 'other_carbonates', 'siderite']

# Group 12: Accessory Minerals (& Miscellaneous, merged with 10)
accessory_minerals = ['apatite_monazite', 'zircon', 'barite', 'rutile', 'rutile_anatase', 'ilmenite', 'mg_so4', 'svanbergite', 'fe_al_po4', 'monazite', 'dioptase', 'corundum_gibbsite_boehmite', 'apatite']

# Group 13: Others (ambiguous minerals)
# all_groups = copper_sulfides + secondary_copper + acid_generating_sulfides + gangue_sulfides + gangue_silicates + clays_and_micas + accessory_silicates + sulfates + fe_oxides + accessory_misc + carbonates + accessory_minerals
# all_groups = copper_sulfides + secondary_copper + acid_generating_sulfides + gangue_sulfides + gangue_silicates + fe_oxides + carbonates + accessory_minerals
all_groups = primary_copper_sulfides + secondary_copper_sulfides + mixed_copper_ores + copper_oxides + acid_generating_sulfides + gangue_sulfides + gangue_silicates + fe_oxides + carbonates + accessory_minerals
other_not_grouped = list(set(df_mineralogy_modals_filtered_tomerge.columns) - set(all_groups) - set(['project_sample_id']))

# Function to safely sum available columns
def sum_available_columns(df, columns):
    df = df.copy()
    df.replace(' ', 0, inplace=True)
    df.replace(np.nan, 0, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df[list(set(columns) & set(df.columns))].sum(axis=1)

df_mineralogy_grouped = df_mineralogy_modals_filtered_tomerge[['project_sample_id']].copy()

# Summing the grouped minerals
# df_mineralogy_grouped['grouped_copper_sulfides'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, copper_sulfides)
# df_mineralogy_grouped['grouped_secondary_copper'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, secondary_copper)
df_mineralogy_grouped['grouped_primary_copper_sulfides'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, primary_copper_sulfides)
df_mineralogy_grouped['grouped_secondary_copper_sulfides'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, secondary_copper_sulfides)
df_mineralogy_grouped['grouped_mixed_copper_ores'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, mixed_copper_ores)
df_mineralogy_grouped['grouped_copper_oxides'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, copper_oxides)

df_mineralogy_grouped['grouped_acid_generating_sulfides'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, acid_generating_sulfides)
df_mineralogy_grouped['grouped_gangue_sulfides'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, gangue_sulfides)
df_mineralogy_grouped['grouped_gangue_silicates'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, gangue_silicates)
# df_mineralogy_grouped['grouped_clays_and_micas'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, clays_and_micas)
# df_mineralogy_grouped['grouped_accesory_silicates'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, accessory_silicates)
# df_mineralogy_grouped['grouped_sulfates'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, sulfates)
df_mineralogy_grouped['grouped_fe_oxides'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, fe_oxides)
# df_mineralogy_grouped['grouped_accessory_misc'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, accessory_misc)
df_mineralogy_grouped['grouped_carbonates'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, carbonates)
df_mineralogy_grouped['grouped_accessory_minerals'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, accessory_minerals)
df_mineralogy_grouped['grouped_other_not_grouped'] = sum_available_columns(df_mineralogy_modals_filtered_tomerge, other_not_grouped)

df_mineralogy_grouped.iloc[:, 1:].sum(axis=1)


# identify and print duplicates
duplicates = df_mineralogy_grouped[df_mineralogy_grouped.duplicated(subset=['project_sample_id'], keep=False)]
if not duplicates.empty:
    print("Duplicates found in df_mineralogy_modals_filtered_tomerge:")
    print(duplicates)
# ensure match for all project_sample_id and print not matched sample_ids
not_matched = df_mineralogy_grouped[~df_mineralogy_grouped['project_sample_id'].isin(df_per_sample['project_sample_id'])]
if not not_matched.empty:
    print("Not matched project_sample_id in df_mineralogy_modals_filtered_tomerge:")
    print(list(not_matched['project_sample_id']))

df_per_sample[df_per_sample['project_sample_id'].str.startswith('032')]

df_mineralogy_grouped.drop_duplicates(subset=['project_sample_id'], keep='first', inplace=True)
df_per_sample = pd.merge(
    df_per_sample,
    df_mineralogy_grouped,
    on='project_sample_id',
    how='left'
)


#  Add qemscan data here once it is available
df_qemscan_filtered[df_qemscan_filtered['project_sample_id'] == '030jettiprojectfile_cpy']
df_qemscan_filtered.drop(columns=['project_sample_condition_id', 'project_sample_id_original'], inplace=True)
df_per_sample = pd.merge(
    df_per_sample,
    df_qemscan_filtered,
    on='project_sample_id',
    how='left'
)

df_per_sample[df_per_sample['project_sample_id'] == '026jettiprojectfile_secondarysulfide'][['project_sample_id', 'cu_seq_h2so4_%', 'cu_%', 'grouped_acid_generating_sulfides']]


#%% Merge with processed reactors per test

df_reactors_merged_tomerge = df_per_project_cell[[
    'project_sample_id',
    'start_cell',
    'max_time_day', 
    'a1_param',
    'b1_param',
    'a2_param',
    'b2_param',
    'final_cu_extraction',
    'extraction_rate_30d',
    'temp_(c)_mean',
    'temp_(c)_std',
    'temp_(c)_max',
    'temp_(c)_slope',
    'ph_mean',
    'ph_std',
    'ph_max',
    'ph_slope',
    'orp_(mv)_mean',
    'orp_(mv)_std',
    'orp_(mv)_max',
    'orp_(mv)_slope',
    'cumulative_catalyst_(kg_t)_mean',
    'cumulative_catalyst_(kg_t)_std',
    'cumulative_catalyst_(kg_t)_max',
    'cumulative_catalyst_(kg_t)_slope',
    'cu_extraction_actual_(%)_mean',
    'cu_extraction_actual_(%)_std',
    'cu_extraction_actual_(%)_max',
    'cu_extraction_actual_(%)_slope',
    #'fe_extraction_actual_(%)_mean',
    #'fe_extraction_actual_(%)_std',
    #'fe_extraction_actual_(%)_max',
    #'fe_extraction_actual_(%)_slope',
    'catalyst_type',
    'lixiviant',
    'ph_target',
    'catalyst_dose_(mg_l)',
    #'catalyst_dose_(mg_l)_mean',
    #'catalyst_dose_(mg_l)_std',
    #'catalyst_dose_(mg_l)_max',
    #'catalyst_dose_(mg_l)_slope',
    'catalyst_addition_day',
    ]].copy()
df_reactors_merged_tomerge = df_reactors_merged_tomerge[df_reactors_merged_tomerge['project_sample_id'].notnull()]
# identify and print duplicates
duplicates = df_reactors_merged_tomerge[df_reactors_merged_tomerge.duplicated(subset=['project_sample_id', 'start_cell'], keep=False)]
if not duplicates.empty:
    print("Duplicates found in df_reactors_merged_tomerge:")
    print(duplicates)

df_reactors_merged_tomerge[df_reactors_merged_tomerge['project_sample_id'] == '030jettiprojectfile_cpy']
df_per_sample[df_per_sample['project_sample_id'] == '030jettiprojectfile_cpy']
df_reactors_merged_tomerge.drop_duplicates(subset=['project_sample_id', 'start_cell'], keep='first', inplace=True)
df_per_sample_reactor = pd.merge(
    df_per_sample,
    df_reactors_merged_tomerge,
    on='project_sample_id',
    how='left'
)

# Define the desired order of the first columns
first_columns = ['project_sample_id', 'start_cell', 'a1_param', 'b1_param', 'a2_param', 'b2_param',]
remaining_columns = [col for col in df_per_sample_reactor.columns if col not in first_columns]
new_column_order = first_columns + remaining_columns
df_per_sample_reactor = df_per_sample_reactor[new_column_order]

grouped_cols = [col for col in df_per_sample_reactor.columns if col.startswith('grouped_')]
df_per_sample_reactor.loc[df_per_sample_reactor['project_sample_id'] == '024jettiprojectfile_cpy', ['project_sample_id', 'start_cell'] + grouped_cols]

df_per_sample_reactor[df_per_sample_reactor['project_sample_id'].str.startswith('032')]

# %% POST PROCESSING
# Standardize categorical columns
categorical_columns = ['catalyst_type', 'lixiviant', 'ph_target']
for col in categorical_columns:
    print(df_per_sample_reactor[col].unique())

'''
[nan '100-CA' 'YVR' 'Control' 'Control 1' 'Control 2' ' ---'
 'Champ 1 60ppm' '100-CA + Champ1 60ppm' 'Champ 1' '100-CA + Champ 1'
 'Champ 2' '100-CA + Champ 2' 'BE Reagent' '100-CA + BE Reagent']
['Inoculum' 'Mine Raff w/o Cu' nan 'Synthetic raffinate'
 'Synthetic Raffinate' 'Synthetic' 'Synthetic Raf' 'Modified Raf'
 'Ferric Sulfate Solution' 'Innoculum' 'Raff' 'Inoc' 'Syn. Raff'
 'Syn Raff' 'Mine Water' 'Raff w/o Cu + Cl (5g/L)' 'Raff w/o Cu'
 'Synthetic Raff' 'Mine Water w/o Cu  ' 'Mine Water w/o Cu + Fe'
 'Mine Raff' '25C Inoculum' '25C Inoculum`' '35C Inoculum' '45C Inoculum'
 'Raffinate']
['2.1' 'As-is' nan '1.7' '1.4' 'as is' '1.8' '-' '1.6' '2.3' '1.8-2.1'
 '2.0' '1.5' '1.2' '2']
'''
 
cat_type = {
    '100-CA': ['100-CA'],
    # 'YVR': ['YVR'],
    'Control': ['Control',  'Control 1',  'Control 2', ' ---', np.nan, 'N/A', pd.NA, 'NaN', '', 'nan'],
    'other_catalysts': ['Champ 1', '100-CA + Champ 1', 'Champ 2', '100-CA + Champ 2', 'BE Reagent', '100-CA + BE Reagent', 'YVR', 'Champ 1 60ppm', '100-CA + Champ1 60ppm', ]
}
# Replace values in the 'catalyst_type' column based on the mapping
df_per_sample_reactor['catalyst_type'] = df_per_sample_reactor['catalyst_type'].replace(
    {v: k for k, values in cat_type.items() for v in values}
)
df_per_sample_reactor['catalyst_type'].unique()
df_per_sample_reactor['catalyst_type'].astype('category')


lix_type = {
    'Inoculum': ['Inoculum', 'Innoculum', 'Inoc', '25C Inoculum', '25C Inoculum`', '35C Inoculum', '45C Inoculum'],
    'Mine Raff': ['Mine Raff', 'Mine Raff w/o Cu', 'Raff w/o Cu',  'Raffinate', 'Raff',  'Raff w/o Cu + Cl (5g/L)'],
    'Synthetic Raffinate': ['Synthetic Raffinate', 'Synthetic raffinate', 'Synthetic Raf', 'Synthetic Raff', 'Synthetic', 'Syn. Raff', 'Syn Raff', ],
    'Modified Raffinate': ['Modified Raf', 'Modified Raff'],
    'Ferric Sulfate Solution': ['Ferric Sulfate Solution', 'Ferric Sulfate'],
    'Mine Water': ['Mine Water', 'Mine Water w/o Cu  ', 'Mine Water w/o Cu + Fe'],
    'Undetermined (N/A)': [np.nan, 'N/A', pd.NA, 'NaN', '', 'nan'],
}
df_per_sample_reactor['lixiviant'] = df_per_sample_reactor['lixiviant'].replace(
    {v: k for k, values in lix_type.items() for v in values}
)
df_per_sample_reactor['lixiviant'].unique()
df_per_sample_reactor['lixiviant'].astype('category')

ph_target_type = {
    '1.7': ['1.7', '1.6', '1.8'],
    '2.0': ['2.1', '2.0', '2', '1.9', '1.8-2.1', '2.1'],
    '2.3': ['2.3'],
    'As-is': ['As-is', 'as is', '-', 'NaN', np.nan, 'N/A', pd.NA, 'nan'],
    '1.5': ['1.5', '1.4', '1.2', '1.6'],
}
df_per_sample_reactor['ph_target'] = df_per_sample_reactor['ph_target'].replace(
    {v: k for k, values in ph_target_type.items() for v in values}
)
df_per_sample_reactor['ph_target'].unique()
df_per_sample_reactor['ph_target'].astype('category')


# Fill some NaN values with 0
df_per_sample_reactor['catalyst_dose_(mg_l)'].fillna(0, inplace=True)
df_per_sample_reactor['catalyst_addition_day'].fillna(0, inplace=True)

df_per_sample_reactor[df_per_sample_reactor['project_sample_id'] == '020jettiprojectfile_hyp']


grouped_cols = [col for col in df_per_sample_reactor.columns if col.startswith('grouped_')]
df_per_sample_reactor.loc[df_per_sample_reactor['project_sample_id'] == '017jettiprojectfile_ea', ['project_sample_id', 'start_cell'] + grouped_cols]
df_per_sample_reactor[df_per_sample_reactor['project_sample_id'].str.startswith('032')]
#%% Some special treatments due to inconsistencies on Jetti Project Files
# 1. Change catalyst type on Control tests if no catalyst dose is present (project 026)
'''
df_per_sample_reactor.loc[
    (df_per_sample_reactor['catalyst_dose_(mg_l)'] == 0),
    'catalyst_type'
] = 'Control'

# 2. Change catalyst type on catalyzed tests if catalyst dose is different from 0 (project 028)
df_per_sample_reactor.loc[
    (df_per_sample_reactor['catalyst_dose_(mg_l)'] != 0),
    'catalyst_type'
] = '100-CA'
'''


# Special treatment 031 because not identifying Control reactors properly before
df_per_sample_reactor.loc[
    (df_per_sample_reactor['project_sample_id'] == '031jettiprojectfile_sample') &
    (df_per_sample_reactor['catalyst_dose_(mg_l)'] == 0),
    'catalyst_type'
] = 'Control'

# Special treament on jettiprojectfile_elephantscl and jettiprojectfile_leopard and jettiprojectfile_tiger_m1, jettiprojectfile_tiger_m2, jettiprojectfile_tiger_m3, jettiprojectfile_toquepala_antigua, jettiprojectfile_toquepala_fresca, jettiprojectfile_zaldivar
# because not identifyint Catalyzed reactors properly before
df_per_sample_reactor.loc[
    (df_per_sample_reactor['project_sample_id'] == 'jettiprojectfile_elephantscl') &
    (df_per_sample_reactor['catalyst_dose_(mg_l)'] > 0),
    'catalyst_type'
] = '100-CA'

df_per_sample_reactor.loc[
    (df_per_sample_reactor['project_sample_id'] == 'jettiprojectfile_leopard') &
    (df_per_sample_reactor['catalyst_dose_(mg_l)'] > 0),
    'catalyst_type'
] = '100-CA'

df_per_sample_reactor.loc[
    (df_per_sample_reactor['project_sample_id'] == 'jettiprojectfile_tiger_m1') &
    (df_per_sample_reactor['catalyst_dose_(mg_l)'] > 0),
    'catalyst_type'
] = '100-CA'

df_per_sample_reactor.loc[
    (df_per_sample_reactor['project_sample_id'] == 'jettiprojectfile_tiger_m2') &
    (df_per_sample_reactor['catalyst_dose_(mg_l)'] > 0),
    'catalyst_type'
] = '100-CA'

df_per_sample_reactor.loc[
    (df_per_sample_reactor['project_sample_id'] == 'jettiprojectfile_tiger_m3') &
    (df_per_sample_reactor['catalyst_dose_(mg_l)'] > 0),
    'catalyst_type'
] = '100-CA'

df_per_sample_reactor.loc[
    (df_per_sample_reactor['project_sample_id'] == 'jettiprojectfile_toquepala_antigua') &
    (df_per_sample_reactor['catalyst_dose_(mg_l)'] > 0),
    'catalyst_type'
] = '100-CA'

df_per_sample_reactor.loc[
    (df_per_sample_reactor['project_sample_id'] == 'jettiprojectfile_toquepala_fresca') &
    (df_per_sample_reactor['catalyst_dose_(mg_l)'] > 0),
    'catalyst_type'
] = '100-CA'

df_per_sample_reactor.loc[
    (df_per_sample_reactor['project_sample_id'] == 'jettiprojectfile_zaldivar') &
    (df_per_sample_reactor['catalyst_dose_(mg_l)'] > 0),
    'catalyst_type'
] = '100-CA'

grouped_cols = [col for col in df_per_sample_reactor.columns if col.startswith('grouped_')]
df_per_sample_reactor.loc[df_per_sample_reactor['project_sample_id'] == '017jettiprojectfile_ea', ['project_sample_id', 'start_cell'] + grouped_cols]

# Display the reordered DataFrame
print(df_per_sample_reactor)
df_per_sample_reactor.describe().to_csv(OUTPUT_DIR + '/dataset_per_sample_description.csv')

df_per_sample_reactor.to_csv(OUTPUT_DIR + '/dataset_per_sample_reactor.csv', index=False)
df_per_sample_reactor.to_csv('/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/csv_outputs/dataset_per_sample_reactor.csv', index=False)
df_per_sample_reactor.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/reactors/dataset_per_sample_reactor.csv', index=False)

# %%
# =============
# Add all mineralogies to the grouped ones
# =============
df_per_sample_reactor_complete = pd.merge(
    df_per_sample_reactor,
    df_mineralogy_modals_filtered_tomerge,
    on='project_sample_id',
    how='left'
)

headers_to_arrays = ['time_(day)', 'cu_extraction_actual_(%)', 'orp_(mv)']
# Add the actual data points for every reactor into separate array columns (one per header)
def extract_reactor_arrays(df, project_sample_id, start_cell, headers):
    filtered_df = df[(df['project_sample_id'] == project_sample_id) & (df['start_cell'] == start_cell)]
    filtered_df = filtered_df.sort_values('time_(day)')
    arrays = {}
    for header in headers:
        series = pd.to_numeric(filtered_df[header], errors='coerce')
        arrays[header] = series.round(2).tolist()
    return arrays

reactor_arrays = df_per_sample_reactor_complete.apply(
    lambda row: pd.Series(
        extract_reactor_arrays(df_reactors_merged, row['project_sample_id'], row['start_cell'], headers_to_arrays)
    ),
    axis=1
)


# Append the array columns to the simplified per-reactor dataframe
df_per_sample_reactor_complete = pd.concat([df_per_sample_reactor_complete, reactor_arrays], axis=1)

excluded_reactor_keys = {
    ('032h_jettifile_ugm4', 'RT_75'), # excluded due to catalyst error addition
    ('032h_jettifile_ugm4', 'RT_76'), # check if must be discarded
}
df_per_sample_reactor_complete = df_per_sample_reactor_complete.loc[
    ~df_per_sample_reactor_complete[['project_sample_id', 'start_cell']].apply(tuple, axis=1).isin(excluded_reactor_keys)
]

df_per_sample_reactor_complete.to_csv(OUTPUT_DIR + '/dataset_per_sample_reactor_complete.csv', index=False)
df_per_sample_reactor_complete.to_csv('/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/csv_outputs/dataset_per_sample_reactor_complete.csv', index=False)
df_per_sample_reactor_complete.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/reactors/dataset_per_sample_reactor_complete.csv', index=False)
df_per_sample_reactor_complete.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/power_bi/dataset_per_sample_reactor_complete.csv', index=False)


# %%
