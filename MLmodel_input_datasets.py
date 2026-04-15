# %%
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator
from pyparsing import col
import seaborn as sns
import sys
import shap
import joblib
import datetime as dt
from statsmodels.nonparametric.smoothers_lowess import lowess
import re
from scipy.interpolate import interp1d


from functions_general import normalize_dataframe_values, normalize_and_replace, dataframe_to_python_code
from plot_helpers import show_or_autoclose_plot
from data_for_rosetta import df_leaching_performance, col_to_match_mineralogy
from rosetta_reactors_pca import df_pca_for_rosetta, df_exponential_model, df_exponential_model_filtered, df_combined_power, df_combined_power_filtered, leaching_cols_to_keep
from rosetta_mineralogy_clustering import df_mineralogy_grouped, pca_mineralogy_grouped # df_mineralogy_hierarchical

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

df_leaching_performance[
    (df_leaching_performance['project_name'] == '020 Jetti Project File Hypogene_Supergene') & 
    (df_leaching_performance['sheet_name'] == 'HYP_1')
    ].head(20)


df_leaching_performance[
    (df_leaching_performance['project_name'] == '007B Jetti Project File - Tiger')
    ]['acid_soluble_%']

# %%======================= Definición de variables de modelo y respuestas

y_result = ['cu_recovery_%']

predictors_dict = { # {control_effect, catalyzed_effect, 1model_effect, name}
    'leach_duration_days': [1, 1, 1, 'Leach Duration (days)'], # 1 best iteration so far
    # 'column_status': [0, 'Columns Status'],
    'feed_flowrate_ml_min': [0, 0, 0, 'Air flow (ml/min)'],
    # 'bed_depth_cm': [0, 'Bed Depth (cm)'],
    # 'feed_ph': [0, 0, 0, 'Feed pH'], #try to get rid , check.... Force to +1 (bibliography)// out after Feature Importance analysis on Dec10th2024
    'feed_orp_mv_ag_agcl': [0, 0, 0, 'Feed ORP (mV)'], # 1 best it so far, based on multiple iterations (June26th 2024)
    # 'pls_orp_mv_ag_agcl': [0, 'PLS ORP (mV)'],
    # 'cumulative_h2so4_kg_t': [0, 0, 0, 'Cumulative H2SO4 (kg/t)'],
    'cumulative_catalyst_addition_kg_t': [1, 1, 1, 'Cumulative Catalyst added (kg/t)'], # nan for chileans??
    'cumulative_lixiviant_flowthrough_l': [0, 0, 0, 'Cumulative Lixiviant (L)'],
    'cumulative_lixiviant_m3_t': [1, 1, 1, 'Cumulative Lixiviant (m3/t)'], 
    'cu_%': [1, 1, 1, 'Head Cu (%)'],
    # 'fe_%': [0, 0, 0, 'Head Fe (%)'],
    'acid_soluble_%': [1, 1, 1, 'Acid Soluble Cu (%norm)'], # 1 based on multiple iterations (June26th 2024)
    'cyanide_soluble_%': [0, 0, 0, 'Cyanide Soluble (%norm)'],
    'residual_cpy_%': [-1, -1, -1, 'Residual Chalcopyrite (%norm)'], # -1 best try so far, based on multiple iterations (June26th 2024)
    'cu_seq_h2so4_%': [0, 0, 0, 'Acid Soluble (%)'],
    'cu_seq_nacn_%': [0, 0, 0, 'Cyanide Soluble (%)'],
    'cu_seq_a_r_%': [0, 0, 0, 'Residual Chalcopyrite (%)'],
    'material_size_p80_in': [-1, -1, -1, 'Material Size P80 (in)'],
    'feed_head_cu_%': [1, 1, 1, 'Head Cu (%)'], # 1  best it so far, based on multiple iterations (June26th 2024)
    'feed_head_fe_%': [0, 0, 0, 'Head Fe (%)'],
    'column_height_m': [-1, -1, -1, 'Column Height (m)'], #JU
    'feed_mass_kg': [0, 0, 0, 'Column Feed Mass (Kg)'],
    'irrigation_rate_l_h_m2': [0, 0, 0, 'Irrigation Rate (L/h/m2)'],
    'irrigation_rate_l_m2_h': [0, 0, 0, 'Irrigation Rate (L/h/m2)'], # try to leave to out (JU)
    'column_inner_diameter_m': [0, 0, 0, 'Column Inner Diameter (m)'],
    # 'agglomeration_y_n': [0, 'Agglomeration'], # JU try to leave out
    # 'agglomeration_medium': [0, ''],
    # 'acid_in_agglomeration_kg_t': [0, ''],
    # 'aeration_y_n': [0, 'Aeration'],
    # 'raff_assay_fe_mg_l': [-1, 'Raffinate Fe (mg/L)'],  # forced after Dry Run (April 19th) (out temporarily cause of terminated projects not integrated yet Aug 5th 2024)
    'raff_assay_fe_ii_mg_l': [0, 0, 0, 'Raffinate Fe(II) (mg/L)'],
    'raff_assay_fe_iii_mg_l': [0, 0, 0, 'Raffinate Fe(III) (mg/L)'],
    'pls_fe_ii_mg_l': [0, 0, 0, 'PLS Fe(II) (mg/L)'],
    'pls_fe_iii_mg_l': [0, 0, 0, 'PLS Fe(III) (mg/L)'],
    # 'pyrite': [-1, -1, -1, 'Pyrite (%)'], # -1 best it so far
    # 'chalcopyrite': [-1, 1, 0, 'Chalcopyrite (%)'],
    # 'quartz': [-1, 'Quartz (%)'], # -1 based on iterations of jun 25th 2024
    # 'k-feldspar': [0, 'K-Felds (%)'],
    # 'biotite': [1, 'Biotite (%)'], # 1 based on iterations of jun 25th 2024
    'chlorite': [0, 0, 0, 'Chlorite (%)'],
    # 'clays': [0, 0, 0, 'Clays (%)'],
    'bornite': [0, 0, 0, 'Bornite (%)'],
    # 'covellite': [0, 0, 0, 'Covellite (%)'],
    # 'chalcocite': [1, 1, 1, 'Chalcocite (%)'], # 1 based on iterations of jun 25th 2024// out because of feature_importance details, Dec10th2024
    'enargite': [0, 0, 0, 'Enargite (%)'],
    # 'molybdenite': [-1, -1, -1, 'Molybdenite (%)'], # -1 based on july 1st 2024 iterations (removed Sep 25th by JU)
    # 'plagioclase': [0, 'Plagioclase (%)'],
    # 'sericite_muscovite': [0, 'Seric-Muscovite (%)'],
    'fe_oxides': [0, 0, 0, 'Fe Oxides (%)'],
    # 'other_oxides': [0, 0, 0, 'Other Oxides (%)'],
    # 'sphalerite': [0, 'Sphalerite (%)'],
    # 'epidote': [0, 'Epidote (%)'],
    # 'rutile': [0, 'Rutile (%)'],
    # 'apatite': [0, 'Apatite (%)'],
    # Monse's Call Feb25 'cus_exposed_50pct_normalized': [0, 1, 0, 'Cu-Sulphides Exp (+50%norm)'], # must be free cause it should not impact on control but should on catalyzed (1 for catalyzed, -1 for control)
    # Monse's Call Feb25 'cus_locked_30pct_normalized': [0, 0, 0, 'Cu-Sulphides Locked (-30%norm)'],
    'cus_exposed_50pct_sum': [0, 0, 0, 'Cu-Sulphides Exposed (+50%)'],
    'cus_locked_30pct_sum': [0, 0, 0, 'Cu-Sulphides Locked (-30%)'],
    'reactors_PCA1': [1, 1, 1, 'Reactors PCA1'],
    'reactors_PCA2': [-1, -1, -1, 'Reactors PCA2'],
    # 'reactors_curve_characterization': [1, 1, 1, 'Reactors Curve Characterization'],
    # 'mineralogy_cluster': [0, 0, 0, 'Mineralogy Cluster'],
    # 'grouped_copper_sulfides': [1, 1, 1, 'Copper Sulphides (%)'],
    # 'grouped_secondary_copper': [1, 1, 1, 'Secondary Copper (%)'],
    'grouped_primary_copper_sulfides': [1, 1, 1, 'Primary Copper Sulphides (%)'],
    'grouped_secondary_copper_sulfides': [1, 1, 1, 'Secondary Copper Sulphides (%)'],
    'grouped_copper_oxides': [0, 0, 0, 'Copper Oxides (%)'],
    'grouped_mixed_copper_ores': [0, 0, 0, 'Mixed Copper Ores (%)'],
    'grouped_acid_generating_sulfides': [1, 1, 1, 'Acid Generating Sulphides (%)'],
    'grouped_gangue_sulfides': [0, 0, 0, 'Gangue Sulphides (%)'],
    'grouped_gangue_silicates': [0, 0, 0, 'Gangue Silicates (%)'],
    # 'grouped_clays_and_micas': [0, 0, 0, 'Clays and Micas (%)'],
    # 'grouped_accesory_silicates': [0, 0, 0, 'Accesory Silicates (%)'],
    # 'grouped_sulfates': [0, 0, 0, 'Sulfates (%)'],
    'grouped_fe_oxides': [-1, -1, -1, 'Fe Oxides (%)'],
    # 'grouped_accessory_misc': [0, 0, 0, 'Accessory Misc (%)'],
    'grouped_carbonates': [0, 0, 0, 'Carbonates (%)'],
    'grouped_accessory_minerals': [0, 0, 0, 'Accessory Minerals (%)'],
    'grouped_phosphate_minerals': [0, 0, 0, 'Phosphate Minerals (%)'],
    # 'grouped_others_not_grouped': [0, 0, 0, 'Others not grouped (%)'],
    # 'mineralogy_pca_1': [0, 0, 0, 'Mineralogy PCA1'],
    # 'mineralogy_pca_2': [0, 0, 0, 'Mineralogy PCA2'],
    # 'mineralogy_pca_3': [0, 0, 0, 'Mineralogy PCA3'],
    # 'mineralogy_pca_4': [0, 0, 0, 'Mineralogy PCA4'],
    # 'mineralogy_pca_5': [0, 0, 0, 'Mineralogy PCA5'],
    # 'mineralogy_pca_6': [0, 0, 0, 'Mineralogy PCA6'],
    # 'reactorsfit_r2': [0, 0, 0, 'Reactors Fit R2'],
    # 'reactorsfit_bias': [0, 0, 0, 'Reactors Fit Bias'],
    # 'reactorsfit_over': [-1, -1, -1, 'Reactors Fit Over/Under est'],
}

categorical_feats = ['mineralogy_cluster']

x_predictors = list(predictors_dict.keys())
# restrictions = str(tuple(predictors_dict.values())) # para xgboost
restrictions_control = [item[0] for item in list(predictors_dict.values())]  # para lightgbm debe ser lista  con [ ] ### list(zip(*list(predictors_dict.values())))[0]
# categ_vars = list(df_master.columns[df_master.columns.get_loc('TratA2 (tph)')+1:])
restrictions_catalyzed = [item[1] for item in list(predictors_dict.values())]
restrictions_1model = [item[2] for item in list(predictors_dict.values())]

df_leaching_performance.groupby(['project_name', 'sheet_name'])[['residual_cpy_%', 'cu_recovery_%']].max().to_excel(folder_path_save + '/residual_cpy.xlsx')
df_leaching_performance.to_csv(folder_path_save + '/df_leaching_performance.csv', sep=',')

df_leaching_performance[
    (df_leaching_performance['project_name'] == '026 Jetti Project File') & 
    (df_leaching_performance['sheet_name'] == 'PS_4')
    ].head(20)

df_leaching_performance[
    (df_leaching_performance['project_name'] == '007B Jetti Project File - Tiger')
    ]


# Replace mineralogies low values and NaN with 0
low_threshold = 0.0005
mineralogy_cols = ['pyrite', 'chlorite', 'bornite', 'enargite', 'fe_oxides']
for col in mineralogy_cols:
    df_leaching_performance[col] = df_leaching_performance[col].apply(lambda x: 0 if x < low_threshold else x)
    # replace NaN with 0 (in case there are any left after the previous fillna, just to be sure)
    df_leaching_performance[col] = df_leaching_performance[col].apply(lambda x: 0 if pd.isna(x) else x)



# %%======================= Definicion de variables de modelo y respuestas

cols_to_identify = ['project_name', 'project_col_id', 'project_sample_id']

# Keep only some columns or projects from the training:

leaching_cols_to_keep #imported from PCA reactors (before it was a dictionary in this script)

list(set(df_leaching_performance['project_col_id'].unique()) - set(leaching_cols_to_keep.keys()))
len(leaching_cols_to_keep)
df_leaching_performance = df_leaching_performance[df_leaching_performance['project_col_id'].isin(leaching_cols_to_keep)]


# =========== REMOVE HOLDUP SOLUTION AND GRE INVENTORY FROM PROJECT 015
df_leaching_performance = df_leaching_performance[df_leaching_performance['condition'].isnull()]

#============ SPECIAL TREATMENT FOR STOPPED COLUMNS (eMail NL oct 29th 2024)
df_leaching_performance.loc[(df_leaching_performance['project_col_id'].isin(['jetti_project_file_zaldivar_scl_col69', 'jetti_project_file_zaldivar_scl_col70']))] = df_leaching_performance.loc[
    (df_leaching_performance['project_col_id'].isin(['jetti_project_file_zaldivar_scl_col69', 'jetti_project_file_zaldivar_scl_col70'])) &
    (df_leaching_performance['leach_duration_days'] < 1361)
]

df_leaching_performance.loc[(df_leaching_performance['project_col_id'].isin(['jetti_project_file_leopard_scl_rom1', 'jetti_project_file_leopard_scl_rom2']))] = df_leaching_performance.loc[
    (df_leaching_performance['project_col_id'].isin(['jetti_project_file_leopard_scl_rom1', 'jetti_project_file_leopard_scl_rom2'])) &
    (df_leaching_performance['leach_duration_days'] < 438) & (df_leaching_performance['leach_duration_days'] > 444)
]

df_leaching_performance.loc[(df_leaching_performance['project_col_id'].isin(['jetti_project_file_elephant_scl_col42', 'jetti_project_file_elephant_scl_col43']))] = df_leaching_performance.loc[
    (df_leaching_performance['project_col_id'].isin(['jetti_project_file_elephant_scl_col42', 'jetti_project_file_elephant_scl_col43'])) &
    (df_leaching_performance['leach_duration_days'] < 1115) # previously 1580 (email NL) ,  1115 Monse's call
]

df_leaching_performance.loc[(df_leaching_performance['project_col_id'].isin(['jetti_project_file_toquepala_scl_col63', 'jetti_project_file_toquepala_scl_col64']))] = df_leaching_performance.loc[
    (df_leaching_performance['project_col_id'].isin(['jetti_project_file_toquepala_scl_col63', 'jetti_project_file_toquepala_scl_col64'])) &
    (df_leaching_performance['leach_duration_days'] < 952)
]

df_leaching_performance.loc[(df_leaching_performance['project_col_id'].isin(['jetti_project_file_rm_1']))] = df_leaching_performance.loc[
    (df_leaching_performance['project_col_id'].isin(['jetti_project_file_rm_1'])) &
    (df_leaching_performance['leach_duration_days'] < 632) & (df_leaching_performance['leach_duration_days'] > 850)
] # Monse's call March 18th 2025 (Santiago)

#============ SPECIAL TREATMENT: CUMULATIVE CATALYST ADDED (SOME OLD TERMINATED PROJECTS HAVE DATA IN CATALYST ADDITION WHEN THEY ARE CONTROL)

# Identify 'control' projects from the dictionary
control_projects = [key for key, value in leaching_cols_to_keep.items() if value[3] == 'control']

# Set cumulative_catalyst_addition_kg_t to 0 for 'control' projects
df_leaching_performance.loc[df_leaching_performance['project_col_id'].isin(control_projects), 'cumulative_catalyst_addition_kg_t'] = 0


#========== INCLUSION OF REACTORS DATA (NOV 18th 2024)

df_pca_reactors = df_pca_for_rosetta[~df_pca_for_rosetta['reactors_PCA1'].isnull()]

# Convert leaching_cols_to_keep to a DataFrame for easier manipulation
leaching_df_pca_reactors = pd.DataFrame.from_dict(leaching_cols_to_keep, orient='index', columns=['project_name', 'start_cell', 'project_sample_id', 'catalyzed_y_n', 'ongoing_y_n'])
leaching_df_pca_reactors.reset_index(inplace=True)
leaching_df_pca_reactors.rename(columns={'index': 'project_col_id'}, inplace=True)

# Merge leaching_df with pca_reactors based on project_name and start_cell
merged_df_pca_reactors = leaching_df_pca_reactors.merge(
    df_pca_reactors,
    on=['project_name', 'start_cell'],
    how='left'
)

# Merge with df_leaching_performance to bring reactors_PCA1 and reactors_PCA2
result_df_leaching_performance_pca_reactors = df_leaching_performance.merge(
    merged_df_pca_reactors[['project_col_id', 'reactors_PCA1', 'reactors_PCA2']],
    on='project_col_id',
    how='left'
)

# result:
result_df_leaching_performance_pca_reactors.to_csv(folder_path_save + '/df_leaching_performance_reactors_pca.csv', sep=',')

df_leaching_performance[
    (df_leaching_performance['project_name'] == '026 Jetti Project File') & 
    (df_leaching_performance['sheet_name'] == 'PS_4')
    ].head(20)

df_leaching_performance[
    (df_leaching_performance['project_name'] == '007 Jetti Project File - Leopard')
    ]

[col for col in df_leaching_performance.columns if col.startswith('grouped')]


#%% ==================== PRINT DATAFRAMES AND PROJECTS TO CHECK
"""
dataframe_to_python_code(result_df_leaching_performance_pca_reactors.head(500), folder=folder_path_save + '/', filename='df_leaching_performance_pca_reactors.txt')
dataframe_to_python_code(df_combined_filtered.head(500), folder=folder_path_save + '/', filename='df_combined_filtered.txt')

"""
#%%
# ================================================================
# ADD REACTORS AS AN STANDARDIZE INCREASIGN VALUE AND NOT ONLY AS A PCA VALUE
'''
# Initialize the new column with NaN
result_df_leaching_performance_pca_reactors['reactors_curve_characterization'] = np.nan

# Iterate through each project_col_id in the dictionary
for project_col_id, project_info in leaching_cols_to_keep.items():
    project_name, reactor_id, _, _, ongoing_y_n = project_info
    
    # Extract the original data for the current project and reactor
    try:
        df_original = df_exponential_model_filtered.xs((project_name, reactor_id), level=(0, 1))
    except KeyError:
        print(f"Data not found for Project: {project_name}, Reactor: {reactor_id}")
        continue
    
    # Proceed if data is available
    if not df_original.empty:
        # Original days (x-axis) and values (y-axis)
        days_original = df_original.columns.astype(float).to_numpy()
        original_max_day = days_original.max()
        
        # Sort the original days and corresponding values to ensure correct interpolation
        sorted_indices = np.argsort(days_original)
        days_original_sorted = days_original[sorted_indices]
        values_original_sorted = df_original.iloc[0, sorted_indices].to_numpy().astype(float)
        
        # Filter rows in result_df for the current project_col_id
        project_mask = result_df_leaching_performance_pca_reactors['project_col_id'] == project_col_id
        df_project = result_df_leaching_performance_pca_reactors.loc[project_mask]
        
        if df_project.empty:
            print(f"No rows found for Project ID: {project_col_id}")
            continue
        
        # Determine new_max_day considering ongoing status or short durations
        new_max_day = df_project['leach_duration_days'].max()
        if ongoing_y_n == 'ongoing' or new_max_day < 600:
            new_max_day = 365 * 4  # Extend to 4 years (1460 days)
        
        # Avoid division by zero if new_max_day is zero (unlikely given checks)
        if new_max_day <= 0:
            print(f"Invalid new_max_day ({new_max_day}) for Project ID: {project_col_id}")
            continue
        
        # Calculate interpolated values for each row in the project
        for idx, row in df_project.iterrows():
            t_new = row['leach_duration_days']
            # Compute the corresponding original time point
            t_original = t_new * (original_max_day / new_max_day)
            # Clip to the original days range to prevent extrapolation
            t_original_clipped = np.clip(t_original, days_original_sorted.min(), days_original_sorted.max())
            # Perform linear interpolation
            y_new = np.interp(t_original_clipped, days_original_sorted, values_original_sorted)
            # Assign the interpolated value to the new column
            result_df_leaching_performance_pca_reactors.at[idx, 'reactors_curve_characterization'] = y_new
'''

# Initialize the new column with NaN
result_df_leaching_performance_pca_reactors['reactors_curve_characterization'] = np.nan

# filter the columns to keep (less than 80 days)
df_exponential_model_filtered = df_exponential_model_filtered.iloc[:, (df_exponential_model_filtered.columns.astype(float).to_numpy() < 80)].copy()

# Iterate through each project_col_id in the dictionary
for project_col_id, project_info in leaching_cols_to_keep.items():
    project_name, reactor_id, _, _, ongoing_y_n = project_info
    print(project_info)
    
    # Extract the original data for the current project and reactor
    try:
        df_original = df_exponential_model_filtered.xs((project_name, reactor_id), level=(0, 1)).copy()
    except KeyError:
        print(f"Data not found for Project: {project_name}, Reactor: {reactor_id}")
        continue
    
    if df_original.empty:
        continue
    
    # Original days (x-axis) and values (y-axis)
    days_original = df_original.columns.astype(float).to_numpy()
    sorted_indices = np.argsort(days_original)
    days_original_sorted = days_original[sorted_indices]
    values_original_sorted = df_original.iloc[0, sorted_indices].to_numpy().astype(float)
    original_max_day = days_original_sorted.max()
    
    # Filter rows in result_df for the current project_col_id
    project_mask = result_df_leaching_performance_pca_reactors['project_col_id'] == project_col_id
    df_project = result_df_leaching_performance_pca_reactors.loc[project_mask]
    
    if df_project.empty:
        print(f"No rows found for Project ID: {project_col_id}")
        continue
    
    # Determine new_max_day considering ongoing status or short durations
    new_max_day = df_project['leach_duration_days'].max()
    if ongoing_y_n == 'ongoing' or new_max_day < 900:
        new_max_day = 365 * 4  # Extend to 4 years (1460 days)
    
    # Calculate interpolated values for each row in the project
    for idx, row in df_project.iterrows():
        t_new = row['leach_duration_days']
        material_size = row['material_size_p80_in']
        # column_height = row['column_height_m']
        
        # Calculate scaling based on material_size_p80_in
        y = 20 * np.exp(-1/2 * material_size) # original 13.5 * np.exp(-0.5 * material_size)
        y_max = 20 * np.exp(-1/2 * 0.5)  # y at minimum material_size (0.5) & minimum column_height_m(0.5)
        scaling_ratio = y / y_max
        
        scaling_ratio = 1.0
        
        # Adjust the original_max_day based on scaling
        original_max_day_adjusted = original_max_day * scaling_ratio
        
        # Compute the corresponding original time point with adjusted scaling
        t_original = t_new * (original_max_day_adjusted / new_max_day)
        
        # Clip to the original days range to prevent extrapolation
        t_original_clipped = np.clip(t_original, days_original_sorted.min(), days_original_sorted.max())
        
        # Perform linear interpolation
        y_new = np.interp(t_original_clipped, days_original_sorted, values_original_sorted)
        
        # Assign the interpolated value to the new column
        result_df_leaching_performance_pca_reactors.at[idx, 'reactors_curve_characterization'] = y_new

result_df_leaching_performance_pca_reactors[
    (result_df_leaching_performance_pca_reactors['project_name'] == '020 Jetti Project File Hypogene_Supergene') & 
    (result_df_leaching_performance_pca_reactors['sheet_name'] == 'HYP_1')
    ].head(20)

[col for col in result_df_leaching_performance_pca_reactors.columns if col.startswith('grouped')]


#%%
# plot some projects to compare results:

def require_multiindex_rows(df, keys, df_name):
    missing_keys = [key for key in keys if key not in df.index]
    if not missing_keys:
        return df.loc[keys]

    index_frame = df.index.to_frame(index=False)
    diagnostic_lines = []
    for project_name, start_cell in missing_keys:
        same_project_reactors = sorted(
            index_frame.loc[index_frame['project_name'] == project_name, 'start_cell']
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        same_start_cell_projects = sorted(
            index_frame.loc[index_frame['start_cell'] == start_cell, 'project_name']
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        diagnostic_lines.append(
            f"{(project_name, start_cell)} | "
            f"same_project_reactors={same_project_reactors or ['none']} | "
            f"same_start_cell_projects={same_start_cell_projects or ['none']}"
        )

    raise KeyError(
        f"Missing rows in {df_name}: " + " ; ".join(diagnostic_lines)
    )

project_col_id = 'jetti_project_file_toquepala_scl_col64'
project_name = 'Jetti Project File - Toquepala SCL'
reactor_id = 'tbl-RT_11'

project_col_id = '017_jetti_project_file_ea_4'
project_name = '017 Jetti Project File'
reactor_id = 'tbl-RTEA_2'

project_col_id = '020_jetti_project_file_hardy_and_waste_har_3'
project_name = '020 Jetti Project File Hardy and Waste'
reactor_id = 'tbl-BR_3'

project_col_id = '011_jetti_project_file_rm_1'
project_name = '011 Jetti Project File'
reactor_id = 'tbl-RT_24'

sns.lineplot(data=result_df_leaching_performance_pca_reactors[result_df_leaching_performance_pca_reactors['project_col_id'] == project_col_id], x='leach_duration_days', y='reactors_curve_characterization')
plt.title(f'Reactor Curve Characterization: {project_col_id}')
plt.ylim(0, 100)
show_or_autoclose_plot(plt)


to_plot = [#('017 Jetti Project File', 'tbl-RTEA_2'),
        #('017 Jetti Project File', 'tbl-RTEA_1'),
        #('022 Jetti Project File', 'tbl-RT_1'),
        #('022 Jetti Project File', 'tbl-RT_2'),
        #('014 Jetti Project File', 'tbl-RTB_8'),
        #('014 Jetti Project File', 'tbl-RTB_7'),
        (project_name, reactor_id),
        ]

df_to_plot = require_multiindex_rows(
    df_exponential_model_filtered,
    to_plot,
    'df_exponential_model_filtered',
)
df_to_plot = df_to_plot.T
df_to_plot.index = pd.to_numeric(df_to_plot.index, errors='coerce')
df_to_plot = df_to_plot.apply(pd.to_numeric)

# Iterate over columns and create scatter plots
for column in df_to_plot.columns:
    sns.scatterplot(x=df_to_plot.index, y=df_to_plot[column], label=column)
plt.xlabel('Index')
plt.ylabel('Values')
plt.ylim(0, 100)
# plt.xlim(0, 200)
plt.legend(loc='best')
show_or_autoclose_plot(plt)

result_df_leaching_performance_pca_reactors[result_df_leaching_performance_pca_reactors['project_col_id'] == project_col_id][['leach_duration_days','reactors_curve_characterization']]

result_df_leaching_performance_pca_reactors[
    (result_df_leaching_performance_pca_reactors['project_name'] == '020 Jetti Project File Hypogene_Supergene') & 
    (result_df_leaching_performance_pca_reactors['sheet_name'] == 'HYP_1')
    ].head(20)

[col for col in result_df_leaching_performance_pca_reactors.columns if col.startswith('grouped')]

#%% ==================== ADD CATEGORICAL FEATURE MINERALOGY CLUSTER
'''
df_mineralogy_hierarchical[['mineralogy_cluster']] = df_mineralogy_hierarchical[['mineralogy_cluster']].astype(int)
result_df_leaching_performance_pca_reactors = result_df_leaching_performance_pca_reactors.merge(
    df_mineralogy_hierarchical[['project_sample_id', 'mineralogy_cluster']],
    on='project_sample_id',
    how='left'
)

result_df_leaching_performance_pca_reactors.dropna(subset='mineralogy_cluster', inplace=True)
result_df_leaching_performance_pca_reactors['mineralogy_cluster'] = pd.to_numeric(result_df_leaching_performance_pca_reactors['mineralogy_cluster'], downcast='integer')
result_df_leaching_performance_pca_reactors['mineralogy_cluster'] = result_df_leaching_performance_pca_reactors[['mineralogy_cluster']].astype('category', errors='ignore')


# change dtypes
result_df_leaching_performance_pca_reactors[['project_name', 'project_col_id', 'project_sample_id']] = result_df_leaching_performance_pca_reactors[['project_name', 'project_col_id', 'project_sample_id']].astype('string', errors='ignore')
'''

#======= Add mineralogy groups to the dataframe
df_mineralogy_grouped = df_mineralogy_grouped.round(2).copy()
result_df_leaching_performance_pca_reactors = result_df_leaching_performance_pca_reactors.merge(
    df_mineralogy_grouped,
    on='project_sample_id',
    how='left'
)

list(df_mineralogy_grouped['project_sample_id'])

#======= PCA MINERALOGY GROUPED
pca_mineralogy_grouped = pca_mineralogy_grouped[~pca_mineralogy_grouped['mineralogy_pca_1'].isnull()]

# Merge leaching_df with pca_reactors based on project_name and start_cell
result_df_leaching_performance_pca_reactors = result_df_leaching_performance_pca_reactors.merge(
    pca_mineralogy_grouped,
    on=['project_sample_id'],
    how='left'
)

result_df_leaching_performance_pca_reactors[
    (result_df_leaching_performance_pca_reactors['project_name'] == '020 Jetti Project File Hypogene_Supergene') & 
    (result_df_leaching_performance_pca_reactors['sheet_name'] == 'HYP_1')
    ].head(20)

[col for col in result_df_leaching_performance_pca_reactors.columns if col.startswith('grouped')]



#%%
#====================
control_cols = [key for key, value in leaching_cols_to_keep.items() if value[3] == 'control']
catalyzed_cols = [key for key, value in leaching_cols_to_keep.items() if value[3] == 'catalyzed']

# added line to make the df run correctly (added after)
x_predictors = list(set(predictors_dict.keys()) - 
                    set(['reactorsfit_over', 'reactorsfit_bias', 'reactorsfit_r2']))


df_model_recCu_catalyzed = \
    result_df_leaching_performance_pca_reactors[
        (result_df_leaching_performance_pca_reactors.index >= result_df_leaching_performance_pca_reactors.index.min()) &
        (result_df_leaching_performance_pca_reactors.index <= result_df_leaching_performance_pca_reactors.index.max()) &
        # (result_df_leaching_performance_pca_reactors['cu_recovery_%'] > 1.0) & # added in line 269 inside prepare_dataframe_to_train_model function.
        # (result_df_leaching_performance_pca_reactors['leach_duration_days'] >= (result_df_pca_reactors['catalyst_start_days_of_leaching'] + ((result_df_pca_reactors['feed_mass_kg'] * 0.08) / (((result_df_pca_reactors['column_inner_diameter_m'] / 2) ** 2 * np.pi) * result_df_pca_reactors['irrigation_rate_l_m2_h'] * 24)))) &  # 8% saturation
        #(result_df_leaching_performance_pca_reactors['catalyst_y_n'] == 'Y') &
        (result_df_leaching_performance_pca_reactors['project_col_id'].isin(catalyzed_cols))
    ][cols_to_identify + x_predictors + y_result]
df_model_recCu_catalyzed.replace([np.inf, -np.inf], np.nan, inplace=True)

df_model_recCu_control = \
    result_df_leaching_performance_pca_reactors[
        (result_df_leaching_performance_pca_reactors.index >= result_df_leaching_performance_pca_reactors.index.min()) &
        (result_df_leaching_performance_pca_reactors.index <= result_df_leaching_performance_pca_reactors.index.max()) &
        # (result_df_leaching_performance_pca_reactors['cu_recovery_%'] > 1.0) & # added in line 269 inside prepare_dataframe_to_train_model function.
        #(result_df_leaching_performance_pca_reactors['catalyst_y_n'] == 'N') &
        (result_df_leaching_performance_pca_reactors['project_col_id'].isin(control_cols))
    ][cols_to_identify + x_predictors + y_result]
df_model_recCu_control.replace([np.inf, -np.inf], np.nan, inplace=True)


# ===== convert to boolean yeses and no's

def convert_to_boolean(value):
    if pd.isna(value) or value == None or value == ' ':
        return False
    elif isinstance(value, str) and (any(char.isdigit() for char in value) or value.upper() in ['Y', 'YES']):
        return True
    else:
        return False

if 'agglomeration_y_n' in df_model_recCu_catalyzed.columns:
    df_model_recCu_catalyzed['agglomeration_y_n'] = df_model_recCu_catalyzed['agglomeration_y_n'].apply(convert_to_boolean)
else:
    pass
if 'agglomeration_y_n' in df_model_recCu_control.columns:
    df_model_recCu_control['agglomeration_y_n'] = df_model_recCu_control['agglomeration_y_n'].apply(convert_to_boolean)
else:
    pass

if 'aeration_y_n' in df_model_recCu_catalyzed.columns:
    df_model_recCu_catalyzed['aeration_y_n'] = df_model_recCu_catalyzed['aeration_y_n'].apply(convert_to_boolean)
else:
    pass
if 'aeration_y_n' in df_model_recCu_control.columns:
    df_model_recCu_control['aeration_y_n'] = df_model_recCu_control['aeration_y_n'].apply(convert_to_boolean)
else:
    pass

if 'column_status' in df_model_recCu_catalyzed.columns:
    df_model_recCu_catalyzed['column_status'] = df_model_recCu_catalyzed['column_status'].str.lower()
    df_model_recCu_catalyzed['column_status'] = df_model_recCu_catalyzed['column_status'].map({'open': True}).fillna(False)
    df_model_recCu_catalyzed['column_status'] = df_model_recCu_catalyzed['column_status'].fillna(False)
else:
    pass
if 'column_status' in df_model_recCu_control.columns:
    df_model_recCu_control['column_status'] = df_model_recCu_control['column_status'].str.lower()
    df_model_recCu_control['column_status'] = df_model_recCu_control['column_status'].map({'open': True}).fillna(False)
    df_model_recCu_control['column_status'] = df_model_recCu_control['column_status'].fillna(False)
else:
    pass



#======== Estadisticas before fill

df_model_recCu_catalyzed.to_csv(folder_path_save + '/df_model_catalyzed_before_fill.csv', sep=',')
df_model_recCu_control.to_csv(folder_path_save + '/df_model_control_before_fill.csv', sep=',')

stats_include = ['count', 'mean', 'std']
cols_to_omit = ['project_name', 'project_col_id', 'project_sample_id', 'mineralogy_cluster']

for c in df_model_recCu_catalyzed.columns:
    if c not in cols_to_omit:
        df_model_recCu_catalyzed[c] = pd.to_numeric(df_model_recCu_catalyzed[c], errors='coerce')
    else:
        pass
    
for c in df_model_recCu_control.columns:
    if c not in cols_to_omit:
        df_model_recCu_control[c] = pd.to_numeric(df_model_recCu_control[c], errors='coerce')
    else:
        pass
    
estadisticos_recCu_catalyzed_bfill = df_model_recCu_catalyzed.groupby(['project_name', 'project_col_id']).describe(include='all')
estadisticos_recCu_catalyzed_bfill = estadisticos_recCu_catalyzed_bfill.loc[:, (slice(None), stats_include)]
estadisticos_recCu_catalyzed_bfill.to_excel(folder_path_save + '/statsRecCu_df_model_catalyzed_bfill.xlsx')
estadisticos_recCu_control_bfill = df_model_recCu_control.groupby(['project_name', 'project_col_id']).describe(include='all')
estadisticos_recCu_control_bfill = estadisticos_recCu_control_bfill.loc[:, (slice(None), stats_include)]
estadisticos_recCu_control_bfill.to_excel(folder_path_save + '/statsRecCu_df_model_control_bfill.xlsx')

df_model_recCu_catalyzed_projects = df_model_recCu_catalyzed.copy()
df_model_recCu_control_projects = df_model_recCu_control.copy()

'''
def prepare_dataframe_to_train_model(df:pd.DataFrame, y:list, X:list,
                                     min_thresh:float, cols_to_fill:list, col_contain_zeros:list,
                                     max_rows_interpolation:int, csv_path_save:str, csv_suffix='', splitter=None):
    try:
        for c in list(set(df.columns) - set([splitter])):
            df[c] = pd.to_numeric(df[c], errors='coerce')
        # df = df[df[y[0]].notnull()] # deprecated on Sep 25th 2024 to avoid loosing NaN con Cu Recovery
        if splitter != None:
            df[[splitter] + X] = df[[splitter] + X].dropna(thresh=int(len(X) * min_thresh), axis=1) # Try None instead of ''
        else:
            df[X] = df[X].dropna(thresh=int(len(X) * min_thresh), axis=1) # Try None instead of ''
        try:
            if c != col_contain_zeros:
                df[c] = df[c].replace(0, np.nan)
        except:
            pass
        try:
            for cf in [x for x in cols_to_fill if x in df.columns]:
                df[cf] = df[cf].interpolate(method='linear', axis=0, limit_direction='forward', limit_area='inside', limit=max_rows_interpolation)
        except:
            pass
        df = df[df[y[0]] > 1.0]
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='any')
        df.to_csv(csv_path_save + f'/df_model_{csv_suffix}.csv', sep=',')
        df_describe = df[X + y].copy().describe(include='all')
        df_describe.to_csv(csv_path_save + f'/stats_model_{csv_suffix}.csv', sep=',')
        return df
    except:
        print('Could not process dataframe')
        return pd.DataFrame(columns=['no_columns'])
'''


def prepare_dataframe_to_train_model(df: pd.DataFrame, y: list, X: list,
                                     min_thresh: float, cols_to_fill: list, cat_cols: list, col_contain_zeros: list,
                                     max_rows_interpolation: int, csv_path_save: str, csv_suffix='', splitter=None):
    try:
        for c in list(set(df.columns) - set(cat_cols) - set([splitter])):
            df[c] = pd.to_numeric(df[c], errors='coerce')

        if splitter is not None:
            df[[splitter] + X] = df[[splitter] + X].dropna(thresh=int(len(X) * min_thresh), axis=1)
        else:
            df[X] = df[X].dropna(thresh=int(len(X) * min_thresh), axis=1)

        # Replace zeros with NaN, except for the columns in 'col_contain_zeros'
        try:
            for c in df.columns:
                if c not in col_contain_zeros:
                    df[c] = df[c].replace(0, np.nan)
        except Exception as e:
            print(f"Error replacing zeros with NaN: {e}")

        # Interpolate based on the splitter and leach_duration_days conditions
        try:
            if splitter is not None and 'leach_duration_days' in df.columns:
                for cf in [x for x in cols_to_fill if x in df.columns]:
                    # Group by splitter and apply custom interpolation
                    df[cf] = df.groupby(splitter, group_keys=False).apply(
                        lambda group: group[cf].where(group['leach_duration_days'].diff().ge(0))
                        .interpolate(method='linear', limit_direction='forward', limit_area='inside',
                                     limit=max_rows_interpolation)
                    )
            else:
                # Ensure 'leach_duration_days' is increasing and interpolate without the splitter
                if 'leach_duration_days' in df.columns:
                    for cf in [x for x in cols_to_fill if x in df.columns]:
                        df[cf] = df[cf].where(df['leach_duration_days'].diff().ge(0)) \
                            .interpolate(method='linear', axis=0, limit_direction='forward', limit_area='inside',
                                         limit=max_rows_interpolation)
        except Exception as e:
            print(f"Error during interpolation: {e}")

        # Filter rows and drop any remaining NaN values
        df = df[df[y[0]] > 1.0]
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='any')

        # Save the resulting DataFrame and its description to CSV files
        df.to_csv(f'{csv_path_save}/df_model_{csv_suffix}.csv', sep=',')
        df_describe = df[X + y].copy().describe(include='all')
        df_describe.to_csv(f'{csv_path_save}/stats_model_{csv_suffix}.csv', sep=',')

        return df

    except Exception as e:
        print(f"Could not process dataframe: {e}")
        return pd.DataFrame(columns=['no_columns'])

# Forward fill for some columns
cols_to_ffill = [
    'cu_recovery_%',
    # 'leach_duration_days',
    # 'column_status',
    'feed_flowrate_ml_min',
    # 'bed_depth_cm',
    'feed_ph',
    'feed_orp_mv_ag_agcl',
    #'cumulative_h2so4_kg_t',
    'cumulative_catalyst_addition_kg_t',
    'cumulative_lixiviant_m3_t',
    # 'cumulative_lixiviant_flowthrough_l',
    # 'cu_%',
    # 'fe_%',
    # 'acid_soluble_%',
    # 'cyanide_soluble_%',
    # 'residual_cpy_%',
    # 'material_size_p80_in',
    # 'feed_head_cu_%',
    # 'feed_head_fe_%',
    # 'column_height_m',
    'irrigation_rate_l_h_m2',
    'irrigation_rate_l_m2_h',  #deprecated since June 26th 2024
    # 'agglomeration_y_n',
    # 'aeration_y_n',
    # 'cu_recovery_%',
    'raff_assay_fe_mg_l',
    'raff_assay_fe_ii_mg_l',
    'raff_assay_fe_iii_mg_l',
    'pls_fe_ii_mg_l',
    'pls_fe_iii_mg_l',
]

cols_can_have_zeros = ['cumulative_catalyst_addition_kg_t'] + col_to_match_mineralogy + df_mineralogy_grouped.columns.tolist()


'''
df=df_model_recCu_control
y=y_result
X=x_predictors
col_contain_zeros=cols_can_have_zeros
min_thresh=min_thresh
cols_to_fill=cols_to_ffill
max_rows_interpolation=30
csv_path_save=folder_path_save
csv_suffix='catalyzed'
'''

df_model_recCu_catalyzed = prepare_dataframe_to_train_model(df=df_model_recCu_catalyzed,
                                                            y=y_result,
                                                            X=x_predictors,
                                                            col_contain_zeros=cols_can_have_zeros,
                                                            min_thresh=min_thresh,
                                                            cols_to_fill=cols_to_ffill,
                                                            cat_cols=['mineralogy_cluster'],
                                                            max_rows_interpolation=30,
                                                            csv_path_save=folder_path_save,
                                                            csv_suffix='catalyzed')
df_model_recCu_control = prepare_dataframe_to_train_model(df=df_model_recCu_control, 
                                                          y=y_result,
                                                          X=x_predictors,
                                                          col_contain_zeros=cols_can_have_zeros,
                                                          min_thresh=min_thresh,
                                                          cols_to_fill=cols_to_ffill,
                                                          cat_cols=['mineralogy_cluster'],
                                                          max_rows_interpolation=30,
                                                          csv_path_save=folder_path_save,
                                                          csv_suffix='control')

df_model_recCu_catcontrol = pd.concat([df_model_recCu_control, df_model_recCu_catalyzed], axis=0)
df_model_recCu_catcontrol_projects = pd.concat([df_model_recCu_catalyzed_projects, df_model_recCu_control_projects], axis=0)

df_model_recCu_catalyzed_projects.to_csv(folder_path_save + '/df_model_catalyzed_projects.csv', sep=',')
df_model_recCu_control_projects.to_csv(folder_path_save + '/df_model_control_projects.csv', sep=',')
df_model_recCu_catcontrol_projects.to_csv(folder_path_save + '/df_model_catcontrol_projects.csv', sep=',')

#%%====== X-FACTOR : ADD BIAS (OVER/UNDER ESTIMATED) VALUES FROM REACTORS LINEAR REGRESSION
'''
df_xfactor_reactors_control = pd.read_excel('/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/reactor_error_metrics_x_factor_control.xlsx')
df_xfactor_reactors_catalyzed = pd.read_excel('/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/reactor_error_metrics_x_factor_catalyzed.xlsx')
df_xfactor_reactors_1model = pd.read_excel('/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/reactor_error_metrics_x_factor_1model.xlsx')

df_xfactor_reactors_control.rename({
    'project_col_id': 'project_col_id',
    'r2': 'reactorsfit_r2',
    'bias': 'reactorsfit_bias',
    'Over': 'reactorsfit_over'
}, axis=1, inplace=True)

df_xfactor_reactors_catalyzed.rename({
    'project_col_id': 'project_col_id',
    'r2': 'reactorsfit_r2',
    'bias': 'reactorsfit_bias',
    'Over': 'reactorsfit_over'
}, axis=1, inplace=True)

df_xfactor_reactors_1model.rename({
    'project_col_id': 'project_col_id',
    'r2': 'reactorsfit_r2',
    'bias': 'reactorsfit_bias',
    'Over': 'reactorsfit_over'
}, axis=1, inplace=True)


# Identify unique 'project_col_id' values that are not in the other DataFrame
missing_in_xfactor_vs_catalyzed = df_model_recCu_catalyzed_projects[~df_model_recCu_catalyzed_projects['project_col_id'].isin(df_xfactor_reactors_catalyzed['project_col_id'])]
missing_in_xfactor_vs_control = df_model_recCu_control_projects[~df_model_recCu_control_projects['project_col_id'].isin(df_xfactor_reactors_control['project_col_id'])]
missing_in_xfactor_vs_1model = df_model_recCu_catcontrol_projects[~df_model_recCu_catcontrol_projects['project_col_id'].isin(df_xfactor_reactors_1model['project_col_id'])]

# Get unique 'project_col_id' values
unique_missing_in_catalyzed = missing_in_xfactor_vs_catalyzed['project_col_id'].unique()
unique_missing_in_control = missing_in_xfactor_vs_control['project_col_id'].unique()
unique_missing_in_1model = missing_in_xfactor_vs_1model['project_col_id'].unique()

# Print the unique missing 'project_col_id' values
print("Unique project_col_id missing in df_xfactor_reactors_catalyzed:")
print(unique_missing_in_catalyzed)
print("\nUnique project_col_id missing in df_xfactor_reactors_control:")
print(unique_missing_in_control)
print("\nUnique project_col_id missing in df_xfactor_reactors_1model:")
print(unique_missing_in_1model)


df_model_recCu_catalyzed_projects = df_model_recCu_catalyzed_projects.merge(
    df_xfactor_reactors_catalyzed[['project_col_id', 'reactorsfit_r2', 'reactorsfit_bias', 'reactorsfit_over']],
    on='project_col_id',
    how='left'
)

df_model_recCu_control_projects = df_model_recCu_control_projects.merge(
    df_xfactor_reactors_control[['project_col_id', 'reactorsfit_r2', 'reactorsfit_bias', 'reactorsfit_over']],
    on='project_col_id',
    how='left'
)

df_model_recCu_catcontrol_projects = df_model_recCu_catcontrol_projects.merge(
    df_xfactor_reactors_1model[['project_col_id', 'reactorsfit_r2', 'reactorsfit_bias', 'reactorsfit_over']],
    on='project_col_id',
    how='left'
)
'''
#%%====== ADD QEMSCAN DATA
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
df_qemscan_filtered = df_qemscan_compilation[df_qemscan_compilation['project_sample_id_raw'].notnull()].copy()
# choose only the rows that have the'combined' on 'sample' column
df_qemscan_filtered = df_qemscan_filtered[df_qemscan_filtered['sample'].str.contains('Combined')].copy()
df_qemscan_filtered.drop_duplicates(subset=['project_sample_id'], keep='first', inplace=True)
df_qemscan_filtered = df_qemscan_filtered[['project_sample_id_raw', 'project_sample_id', 'project_sample_condition_id', 'cpy_+50%_exposed_norm', 'cpy_locked_norm', 'cpy_associated_norm']]

df_qemscan_filtered_106 = df_qemscan_compilation[df_qemscan_compilation['project_sample_id_raw'].notnull()].copy()
# choose only the rows that have the number 106 on 'sample' column
df_qemscan_filtered_106 = df_qemscan_filtered_106[df_qemscan_filtered_106['sample'].str.contains('106')].copy()
df_qemscan_filtered_106.drop_duplicates(subset=['project_sample_id'], keep='first', inplace=True)
df_qemscan_filtered_106 = df_qemscan_filtered_106[['project_sample_id_raw', 'project_sample_id', 'project_sample_condition_id', 'cpy_+50%_exposed_norm', 'cpy_locked_norm', 'cpy_associated_norm']]
df_qemscan_filtered_106.rename(columns={'cpy_+50%_exposed_norm': 'cpy_+50%_exposed106_norm', 'cpy_locked_norm': 'cpy_locked106_norm', 'cpy_associated_norm': 'cpy_associated106_norm'}, inplace=True)

df_qemscan_filtered['project_sample_id'].unique()

df_model_recCu_catalyzed_projects = pd.merge(
    df_model_recCu_catalyzed_projects,
    df_qemscan_filtered,
    on='project_sample_id',
    how='left'
)
df_model_recCu_catalyzed_projects = pd.merge(
    df_model_recCu_catalyzed_projects,
    df_qemscan_filtered_106,
    on='project_sample_id',
    how='left'
)

df_model_recCu_control_projects = pd.merge(
    df_model_recCu_control_projects,
    df_qemscan_filtered,
    on='project_sample_id',
    how='left'
)
df_model_recCu_control_projects = pd.merge(
    df_model_recCu_control_projects,
    df_qemscan_filtered_106,
    on='project_sample_id',
    how='left'
)

df_model_recCu_catcontrol_projects = pd.merge(
    df_model_recCu_catcontrol_projects,
    df_qemscan_filtered,
    on='project_sample_id',
    how='left'
)
df_model_recCu_catcontrol_projects = pd.merge(
    df_model_recCu_catcontrol_projects,
    df_qemscan_filtered_106,
    on='project_sample_id',
    how='left'
)

cols_to_ffill = [
    'cu_recovery_%',
    'feed_flowrate_ml_min',
    'feed_ph',
    'feed_orp_mv_ag_agcl',
    'cumulative_catalyst_addition_kg_t',
    'cumulative_lixiviant_m3_t',
    'irrigation_rate_l_h_m2',
    'irrigation_rate_l_m2_h',  #deprecated since June 26th 2024
    'raff_assay_fe_mg_l',
    'raff_assay_fe_ii_mg_l',
    'raff_assay_fe_iii_mg_l',
    'pls_fe_ii_mg_l',
    'pls_fe_iii_mg_l',
]

all_columns = list(set(df_model_recCu_catcontrol_projects.columns))
time_feature = ['leach_duration_days']
cols_can_have_zeros = ['cumulative_catalyst_addition_kg_t'] + [col for col in all_columns if col.startswith('grouped')]
id_cols = ['project_name', 'project_sample_id', 'project_col_id']
numerical_features = list(set(all_columns) - set(categorical_feats) - set(id_cols))

#%% Preprocessing Function
def preprocess_data(df):
    for c in numerical_features:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in df.columns:
        if c not in cols_can_have_zeros and c not in id_cols:
            df[c] = df[c].replace(0, np.nan)
    # Save id_cols before dropna — they must never be dropped even if sparse
    id_cols_present = [c for c in id_cols if c in df.columns]
    saved_id_data = df[id_cols_present].copy()
    df = df.dropna(thresh=int(len(df.columns) * min_thresh), axis=1)
    for col in id_cols_present:
        if col not in df.columns:
            df[col] = saved_id_data[col]
    if time_feature[0] in df.columns:
        for cf in [x for x in cols_to_ffill if x in df.columns]:
            df[cf] = df.groupby(id_cols, group_keys=False).apply(
                lambda group: group[cf].where(group[time_feature[0]].diff().ge(0))
                .interpolate(method='linear', limit_direction='forward', 
                             limit_area='inside', limit=30)
            )
    df = df[df[y_result[0]] > 0.0]
    # Keep only rows where y_result is strictly increasing within each group.
    # Avoid groupby().apply() returning a DataFrame — pandas 2.x may absorb
    # the groupby keys into a MultiIndex, silently removing them from columns.
    id_cols_in_df = [c for c in id_cols if c in df.columns]
    if id_cols_in_df:
        diff_mask = df.groupby(id_cols_in_df, sort=False)[y_result[0]].diff().gt(0)
    else:
        diff_mask = df[y_result[0]].diff().gt(0)
    df = df[diff_mask].reset_index(drop=True)
    # df = df[id_cols + categorical_feats + numerical_features]
    # df = df.dropna(axis=0, how='any')
    return df


def ensure_irrigation_alias(df):
    df = df.copy()
    if 'irrigation_rate_l_m2_h' not in df.columns:
        return df
    if 'irrigation_rate_l_h_m2' not in df.columns:
        df['irrigation_rate_l_h_m2'] = df['irrigation_rate_l_m2_h']
    else:
        df['irrigation_rate_l_h_m2'] = df['irrigation_rate_l_h_m2'].fillna(df['irrigation_rate_l_m2_h'])
    return df

df_model_recCu_catalyzed_projects = preprocess_data(df_model_recCu_catalyzed_projects.copy())
df_model_recCu_control_projects = preprocess_data(df_model_recCu_control_projects.copy())
df_model_recCu_catcontrol_projects = preprocess_data(df_model_recCu_catcontrol_projects.copy())

df_model_recCu_catalyzed_projects = ensure_irrigation_alias(df_model_recCu_catalyzed_projects)
df_model_recCu_control_projects = ensure_irrigation_alias(df_model_recCu_control_projects)
df_model_recCu_catcontrol_projects = ensure_irrigation_alias(df_model_recCu_catcontrol_projects)


first_cols = [
    'project_name',
    'project_sample_id',
    'project_col_id',
    'leach_duration_days',
    'cu_recovery_%',
    'acid_soluble_%',
    'cyanide_soluble_%',
    'residual_cpy_%'
]
other_cols = sorted([col for col in df_model_recCu_catcontrol_projects.columns if col not in first_cols])
new_col_order = first_cols + other_cols

df_model_recCu_catalyzed_projects = df_model_recCu_catalyzed_projects[new_col_order]
df_model_recCu_control_projects = df_model_recCu_control_projects[new_col_order]
df_model_recCu_catcontrol_projects = df_model_recCu_catcontrol_projects[new_col_order]

#%% ==================== SAVE FINAL DATAFRAMES WITH REACTORS FIT METRICS


# Save the final DataFrames with reactors fit metrics
df_model_recCu_catalyzed_projects.to_csv('/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/csv_outputs/df_model_catalyzed_projects_with_reactors_fit.csv', sep=',', index=False)
df_model_recCu_catalyzed_projects.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/columns/df_model_catalyzed_projects_with_reactors_fit.csv', sep=',', index=False)

df_model_recCu_control_projects.to_csv('/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/csv_outputs/df_model_control_projects_with_reactors_fit.csv', sep=',', index=False)
df_model_recCu_control_projects.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/columns/df_model_control_projects_with_reactors_fit.csv', sep=',', index=False)


# Replace mineralogies low values and NaN with 0
low_threshold = 0.001
mineralogy_cols = ['pyrite', 'chlorite', 'bornite', 'enargite', 'fe_oxides']
for col in [col for col in df_model_recCu_catcontrol_projects.columns if col in mineralogy_cols]:
    df_model_recCu_catcontrol_projects[col] = df_model_recCu_catcontrol_projects[col].apply(lambda x: 0 if x < low_threshold else x)
    # replace NaN with 0 (in case there are any left after the previous fillna, just to be sure)
    df_model_recCu_catcontrol_projects[col] = df_model_recCu_catcontrol_projects[col].apply(lambda x: 0 if pd.isna(x) else x)

df_model_recCu_catcontrol_projects.to_csv('/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/csv_outputs/df_model_catcontrol_projects_with_reactors_fit.csv', sep=',', index=False)
df_model_recCu_catcontrol_projects.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/columns/df_model_catcontrol_projects_with_reactors_fit.csv', sep=',', index=False)
df_model_recCu_catcontrol_projects.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/power_bi/df_model_catcontrol_projects_with_reactors_fit.csv', sep=',', index=False)

#%%
# df_qemscan_filtered revisar project sample id poorque es diferente al de los proyectos
