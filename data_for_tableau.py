#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sys
import re
from functions_general import normalize_dataframe_values, normalize_series_values, normalize_and_replace
import os
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
folder_path_load = '/Users/administration/Library/CloudStorage/OneDrive-JettiResources/PythonProjects/SpkData/Jetti01'
folder_path_save = '/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python'

df_merged = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/leaching_columns.csv', sep=',', low_memory=False)
df_column_summary = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/dataset_column_summary.csv', sep=',', low_memory=False)
df_bottles = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/dataset_rolling_bottles_detailed.csv', sep=',', low_memory=False)
df_reactors = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/dataset_reactor_summary_detailed.csv', sep=',', low_memory=False)
df_chemchar = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/dataset_characterization_summary.csv', sep=',', low_memory=False)
df_chemchar = df_chemchar[df_chemchar['analyte_units'].notnull()]
df_ac_summary = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/dataset_acid_consumption_summary_summaries.csv', sep=',', low_memory=False)
df_mineralogy_modals = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/dataset_mineralogy_summary_modals.csv', sep=',', low_memory=False)
df_separation = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/dataset_nelson_separation.csv', sep=',', low_memory=False)
df_comments = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/dataset_comments_monthly.csv', sep=',', low_memory=False)

# new data tables added on june 24th 2024
# df_qemscan_exposure = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/dataset_exposure.csv', sep=',', low_memory=False)
# df_qemscan_meangrain = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/qemscan_dataset_modals_mean_grain.csv', sep=',', low_memory=False)
# df_qemscan_mineralmass = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/qemscan_dataset_modals_mineral_mass.csv', sep=',', low_memory=False)

# maker index
df_maker_index = pd.read_excel(folder_path_load + '/input_sheets/input_dataframe_maker.xlsx', sheet_name='project_index')
df_maker_index = df_maker_index.loc[:, ~df_maker_index.columns.str.contains('^Unnamed')]
#df_maker_index['project_sample_id'] = normalize_dataframe_values(df_maker_index['file_name'].astype(str).str.replace('-', '_') + '_' + df_maker_index['sample_id'].astype(str).replace(np.nan, '').str.replace('-', '_') + '_' + df_maker_index['condition_id'].astype(str).replace(np.nan, '').str.replace('-', '_')).str.replace(r'_avg$|_head$|_dup$|_\((a|b)\)$', '', regex=True)
df_maker_index['project_cat1_col_id'] = normalize_dataframe_values(df_maker_index['file_name'].astype(str).str.replace('-', '_') + '_' + df_maker_index['catalyzed_col_1'].astype(str).replace(np.nan, '').str.replace('-', '_'))
df_maker_index['project_cat2_col_id'] = normalize_dataframe_values(df_maker_index['file_name'].astype(str).str.replace('-', '_') + '_' + df_maker_index['catalyzed_col_2'].astype(str).replace(np.nan, '').str.replace('-', '_'))
df_maker_index['project_control1_col_id'] = normalize_dataframe_values(df_maker_index['file_name'].astype(str).str.replace('-', '_') + '_' + df_maker_index['control_col_1'].astype(str).replace(np.nan, '').str.replace('-', '_'))
df_maker_index['project_control2_col_id'] = normalize_dataframe_values(df_maker_index['file_name'].astype(str).str.replace('-', '_') + '_' + df_maker_index['control_col_2'].astype(str).replace(np.nan, '').str.replace('-', '_'))
df_maker_index['project_cat_reactor_id'] = normalize_dataframe_values(df_maker_index['file_name'].astype(str).str.replace('-', '_') + '_' + df_maker_index['catalyzed_reactor_1'].astype(str).replace(np.nan, '').str.replace('-', '_'))
df_maker_index['project_control_reactor_id'] = normalize_dataframe_values(df_maker_index['file_name'].astype(str).str.replace('-', '_') + '_' + df_maker_index['control_reactor_1'].astype(str).replace(np.nan, '').str.replace('-', '_'))

df_maker_index.to_csv(folder_path_save +'/df_maker_index.csv', sep=',')


#%% Merge with terminated old projects (01A, 002, 004, 006) (Manual merge still)

df_terminated_leaching = pd.read_excel(folder_path_save + '/merged_terminated_projects_testDetails_MetallurgicalBalance.xlsx')
df_terminated_chemchar = pd.read_excel(folder_path_save + '/dataset_characterization_summary_terminated_projects.xlsx')
df_terminated_mineralogy_modals = pd.read_excel(folder_path_save + '/dataset_mineralogy_summary_modals_terminated_projects.xlsx')
df_terminated_mineralogy_deportment = pd.read_excel(folder_path_save + '/dataset_mineralogy_summary_deportments_terminated_projects.xlsx')
df_terminated_column_summary = pd.read_excel(folder_path_save + '/df_column_summary_terminated_projects.xlsx')
df_qemscan_terminated = pd.read_excel(folder_path_save + '/df_qemscan_compilation.xlsx')


df_merged = pd.concat([df_terminated_leaching, df_merged], axis=0, ignore_index=True)
df_column_summary = pd.concat([df_terminated_column_summary, df_column_summary], axis=0, ignore_index=True)
df_chemchar = pd.concat([df_terminated_chemchar, df_chemchar], axis=0, ignore_index=True)
df_mineralogy_modals = pd.concat([df_terminated_mineralogy_modals, df_mineralogy_modals], axis=0, ignore_index=True)


 #%% Force numeric

cols_to_numeric = [
    'pls_cu_mg_l', 'pls_fe_mg_l', 'pls_mg_mg_l', 'pls_al_mg_l', 
    'pls_si_mg_l', 'pls_co_mg_l', 'pls_li_mg_l', 'pls_u_mg_l',
]

for b in cols_to_numeric:
    df_merged[b] = pd.to_numeric(df_merged[b], errors='coerce')

df_column_summary['col_name'] = df_column_summary['col_name'].astype(object)
df_column_summary.loc[df_column_summary['index'].str.startswith('_'), 'col_name'] = df_column_summary.loc[ df_column_summary['index'].str.startswith('_'), 'index'].str.lstrip('_')
df_column_summary.loc[:, 'col_name'] = df_column_summary['col_name'].fillna(df_column_summary['index'])
df_merged.loc[:, 'sheet_name'] = df_merged['sheet_name'].str.replace('-', '_') # special treatment for Tiger ROM file which has - instead of _ on the sheet name

cols_not_to_numeric_separation = [
    'origin', 'project_name', 'sheet_name', 'start_cell'
]
for col in df_separation.columns:
    if col in cols_not_to_numeric_separation:
        continue
    df_separation[col] = pd.to_numeric(df_separation[col], errors='coerce')


# df_column_summary[['project_name', 'col_name']].to_csv('/Users/administration/Downloads/column_match.csv', sep=',')

# Special treatment for 026 seconday sulfide sample 3 # replace secs for ss on every string found
df_column_summary.loc[df_column_summary['project_name'] == '026 Jetti Project File', 'index'] = df_column_summary.loc[df_column_summary['project_name'] == '026 Jetti Project File', 'index'].str.replace('SecS', 'SS', regex=False)

df_column_summary[df_column_summary['project_name'] == '007 Jetti Project File - Leopard']

# %% HEADERS FOR COLUMN SUMMARY TABLE

cols_column_summary = [
    'origin',
    'project_name',
    'index',
    'sample_id',
    'material_size_p80_in',
    'feed_head_cu_%',
    'feed_head_fe_%',
    'feed_head_mg_%',
    'feed_head_al_%',
    'feed_head_co_%',
    'feed_head_si_%',
    'lixiviant_initial_ph',
    'lixiviant_initial_orp_mv',
    'lixiviant_initial_cu_mg_l',
    'lixiviant_initial_fe_mg_l',
    'catalyst_start_days_of_leaching',
    'column_height_m',
    'column_inner_diameter_m',
    'feed_mass_kg',
    'irrigation_rate_l_m2_h',
    'agglomeration_y_n',
    'agglomeration_medium',
    'acid_in_agglomeration_kg_t',
    'lixiviant_inoc_site_raff_syn_raff',
    'inoculum_%',
    'aeration_y_n',
    'aeration_dosage_l_min',
    'catalyst_y_n',
    'catalyst_dosage_mg_day',
    'catalyst_dosage_mg_l',
]
df_column_summary_filtered = df_column_summary[cols_column_summary].copy()

df_column_summary_filtered.loc[:, 'project_col_id'] = normalize_dataframe_values(df_column_summary_filtered['project_name'].astype(str).str.replace('-', '') + '_' + df_column_summary_filtered['index'].astype(str).str.replace('-', ''))
df_column_summary_filtered.loc[:, 'col_name'] = normalize_dataframe_values(df_column_summary_filtered['index'].astype(str).str.replace('-', ''))
#df_column_summary_filtered['project_sample_id'] = normalize_dataframe_values(df_column_summary_filtered['project_name'].astype(str).str.replace('-', '') + '_' + df_column_summary_filtered['sample_id'].astype(str).str.replace('-', ''))

cols_to_numeric = [x for x in cols_column_summary if x not in ['origin', 'project_name', 'index', 'sample_id', 'aeration_y_n', 'agglomeration_y_n', 'agglomeration_medium', 'catalyst_y_n', 'lixiviant_inoc_site_raff_syn_raff']]
for c in cols_to_numeric:
    df_column_summary_filtered.loc[:, c] = pd.to_numeric(df_column_summary_filtered[c], errors='coerce')

df_column_summary_filtered['project_col_id'].unique()
df_column_summary_filtered[df_column_summary_filtered['project_name'] == '026 Jetti Project File']

df_column_summary_filtered[df_column_summary_filtered['project_name'] == '007 Jetti Project File - Leopard']
df_column_summary_filtered[df_column_summary_filtered['project_name'] == '007B Jetti Project File - Tiger']
df_column_summary_filtered[df_column_summary_filtered['project_name'] == '011 Jetti Project File-Crushed']


replacement_dict = {
    '011_jetti_project_file_rm_1': '011_jetti_project_file_rm',
    '011_jetti_project_file_rm_2': '011_jetti_project_file_rm',
    '011_jetti_project_filecrushed_rm_5': '011_jetti_project_file_rm_crushed',
    '011_jetti_project_filecrushed_rm_6': '011_jetti_project_file_rm_crushed',
    '011_jetti_project_filecrushed_rm_7': '011_jetti_project_file_rm_crushed',
    '011_jetti_project_filecrushed_rm_8': '011_jetti_project_file_rm_crushed',
    '012_jetti_project_file_cs_i_1': '012_jetti_project_file_incremento',
    '012_jetti_project_file_cs_i_2': '012_jetti_project_file_incremento',
    '012_jetti_project_file_cs_i_3': '012_jetti_project_file_incremento',
    '012_jetti_project_file_cs_q_1': '012_jetti_project_file_quebalix',
    '012_jetti_project_file_cs_q_2': '012_jetti_project_file_quebalix',
    '012_jetti_project_file_cs_q_3': '012_jetti_project_file_quebalix',
    '014_jetti_project_file_b_1': '014_jetti_project_file_bag',
    '014_jetti_project_file_b_2': '014_jetti_project_file_bag',
    '014_jetti_project_file_b_3': '014_jetti_project_file_bag',
    '014_jetti_project_file_b_4': '014_jetti_project_file_bag',
    '014_jetti_project_file_k_1': '014_jetti_project_file_kmb',
    '014_jetti_project_file_k_2': '014_jetti_project_file_kmb',
    '014_jetti_project_file_k_3': '014_jetti_project_file_kmb',
    '014_jetti_project_file_k_4': '014_jetti_project_file_kmb',
    '015_jetti_project_file_c_1': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_6': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_11': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_2': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_7': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_12': '015_jetti_project_file_amcf',
    '021_jetti_project_file_c_1': '021_jetti_project_file_hypogene',
    '021_jetti_project_file_c_2': '021_jetti_project_file_hypogene',
    '021_jetti_project_file_c_3': '021_jetti_project_file_hypogene',
    '003_jetti_project_file_be_1': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_be_2': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_be_3': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_be_4': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_be_5': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_be_6': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_be_7': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_be_8': '003_jetti_project_file_amcf_head',
    '030_jetti_project_file_cpy_1': '030_jetti_project_file_cpy',
    '030_jetti_project_file_cpy_2': '030_jetti_project_file_cpy',
    '030_jetti_project_file_ss_1': '030_jetti_project_file_ss',
    '030_jetti_project_file_ss_2': '030_jetti_project_file_ss',
    '017_jetti_project_file_ea_1': '017_jetti_project_file_ea_mill_feed_combined',
    '017_jetti_project_file_ea_2': '017_jetti_project_file_ea_mill_feed_combined',
    '017_jetti_project_file_ea_3': '017_jetti_project_file_ea_mill_feed_combined',
    '017_jetti_project_file_ea_4': '017_jetti_project_file_ea_mill_feed_combined',
    '020_jetti_project_file_hardy_and_waste_har_1': '020_jetti_project_file_hardy_and_waste_h21_master_comp',
    '020_jetti_project_file_hardy_and_waste_har_2': '020_jetti_project_file_hardy_and_waste_h21_master_comp',
    '020_jetti_project_file_hardy_and_waste_har_3': '020_jetti_project_file_hardy_and_waste_h21_master_comp',
    '020_jetti_project_file_hypogene_supergene_hyp_1': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite',
    '020_jetti_project_file_hypogene_supergene_hyp_2': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite',
    '020_jetti_project_file_hypogene_supergene_hyp_3': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite',
    '020_jetti_project_file_hypogene_supergene_sup_1': '020_jetti_project_file_hypogene_supergene_super',
    '020_jetti_project_file_hypogene_supergene_sup_2': '020_jetti_project_file_hypogene_supergene_super',
    '020_jetti_project_file_hypogene_supergene_sup_3': '020_jetti_project_file_hypogene_supergene_super',
    '022_jetti_project_file_s_1': '022_jetti_project_file_stingray_1',
    '022_jetti_project_file_s_2': '022_jetti_project_file_stingray_1',
    #'023_jetti_project_file': '023_jetti_project_file_ot_10',
    #'023_jetti_project_file': '023_jetti_project_file_ot_9',
    '024_jetti_project_file_cv_1': '024_jetti_project_file_024cv_cpy',
    '024_jetti_project_file_cv_2': '024_jetti_project_file_024cv_cpy',
    '024_jetti_project_file_cv_3': '024_jetti_project_file_024cv_cpy',
    '024_jetti_project_file_cv_4': '024_jetti_project_file_024cv_cpy',
    '028_jetti_project_file_comp_1': '028_jetti_project_file_andesite', # chequear por compund 70/30
    '028_jetti_project_file_comp_2': '028_jetti_project_file_andesite', # chequear por compund 70/30
    '028_jetti_project_file_comp_3': '028_jetti_project_file_monzonite', # chequear por compund 70/30
    '028_jetti_project_file_comp_4': '028_jetti_project_file_monzonite', # chequear por compund 70/30
    'jetti_project_file_leopard_scl_col1': 'jetti_project_file_leopard_scl_sample_los_bronces',
    'jetti_project_file_leopard_scl_col2': 'jetti_project_file_leopard_scl_sample_los_bronces',
    'jetti_project_file_leopard_scl_rom1': 'jetti_project_file_leopard_scl_sample_los_bronces',
    'jetti_project_file_leopard_scl_rom2': 'jetti_project_file_leopard_scl_sample_los_bronces',
    'jetti_project_file_tiger_rom_rom1': 'jetti_project_file_tiger_rom_m1',
    'jetti_project_file_tiger_rom_rom2': 'jetti_project_file_tiger_rom_m1',
    'jetti_project_file_tiger_rom_rom3': 'jetti_project_file_tiger_rom_m1',
    'jetti_project_file_tiger_rom_c_4': 'jetti_project_file_tiger_rom_m1',
    'jetti_project_file_tiger_rom_c_5': 'jetti_project_file_tiger_rom_m1',
    'jetti_project_file_tiger_rom_c_6': 'jetti_project_file_tiger_rom_m2',
    'jetti_project_file_tiger_rom_c_7': 'jetti_project_file_tiger_rom_m2',
    'jetti_project_file_tiger_rom_c_8': 'jetti_project_file_tiger_rom_m3',
    'jetti_project_file_tiger_rom_c_9': 'jetti_project_file_tiger_rom_m3',
    'jetti_project_file_elephant_scl_col42': 'jetti_project_file_elephant_scl_sample_escondida',
    'jetti_project_file_elephant_scl_col43': 'jetti_project_file_elephant_scl_sample_escondida',
    'jetti_project_file_elephant_scl_col52': 'jetti_project_file_elephant_scl_sample_escondida',
    'jetti_project_file_elephant_scl_col53': 'jetti_project_file_elephant_scl_sample_escondida',
    'jetti_project_file_toquepala_scl_col63': 'jetti_project_file_toquepala_scl_sample_fresca',
    'jetti_project_file_toquepala_scl_col64': 'jetti_project_file_toquepala_scl_sample_fresca',
    'jetti_project_file_toquepala_scl_col65': 'jetti_project_file_toquepala_scl_sample_fresca',
    'jetti_project_file_toquepala_scl_col66': 'jetti_project_file_toquepala_scl_sample_antigua',
    'jetti_project_file_toquepala_scl_col67': 'jetti_project_file_toquepala_scl_sample_antigua',
    'jetti_project_file_zaldivar_scl_col68': 'jetti_project_file_zaldivar_scl_sample_zaldivar',
    'jetti_project_file_zaldivar_scl_col69': 'jetti_project_file_zaldivar_scl_sample_zaldivar',
    'jetti_project_file_zaldivar_scl_col70': 'jetti_project_file_zaldivar_scl_sample_zaldivar',
    'jetti_project_file_elephant_(site)_fat4': 'jetti_project_file_elephant_site',
    'jetti_project_file_elephant_(site)_fat6': 'jetti_project_file_elephant_site',
    'jetti_project_file_elephant_(site)_s3': 'jetti_project_file_elephant_site_s3',
    'jetti_file_elephant_ii_ver_2_ugm_ur_1': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_ur_2': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_1': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_2': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_3': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_4': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_5': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_6': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_7': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_8': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_9': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_pq_pr_1': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pr_2': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_1': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_2': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_3': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_4': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_5': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_6': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_7': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_8': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_9': 'jetti_file_elephant_ii_pq',
    '1528901a_column_leach_20200219_c1': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c2': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c3': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c4': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c5': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c6': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c7': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c8': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c9': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c10': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c11': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c12': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c13': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c14': '01a_jetti_project_file_c',
    '15289002_column_leach_v1_20191217_qb1': '002_jetti_project_file_qb',
    '15289002_column_leach_v1_20191217_qb2': '002_jetti_project_file_qb',
    '15289002_column_leach_v1_20191217_qb3': '002_jetti_project_file_qb',
    '15289002_column_leach_v1_20191217_qb4': '002_jetti_project_file_qb',
    '15289004_column_leach_v1_20201130_mo1': '004_jetti_project_file_mo',
    '15289004_column_leach_v1_20201130_mo2': '004_jetti_project_file_mo',
    '15289004_column_leach_v1_20201130_mo3': '004_jetti_project_file_mo',
    '15289004_column_leach_v1_20201130_mo4': '004_jetti_project_file_mo',
    '15289004_column_leach_v1_20201130_mols1': '004_jetti_project_file_mols',
    '15289004_column_leach_v1_20201130_mols2': '004_jetti_project_file_mols',
    '15289004_column_leach_v1_20201130_mols3': '004_jetti_project_file_mols',
    '15289004_column_leach_v1_20201130_mols4': '004_jetti_project_file_mols',
    '15289006_column_leach_v1_20200828_pvls1': '006_jetti_project_file_pvls',
    '15289006_column_leach_v1_20200828_pvls2': '006_jetti_project_file_pvls',
    '15289006_column_leach_v1_20200828_pvls3': '006_jetti_project_file_pvls',
    '15289006_column_leach_v1_20200828_pvls4': '006_jetti_project_file_pvls',
    '15289006_column_leach_v1_20200828_pvo1': '006_jetti_project_file_pvo',
    '15289006_column_leach_v1_20200828_pvo2': '006_jetti_project_file_pvo',
    '15289006_column_leach_v1_20200828_pvo3': '006_jetti_project_file_pvo',
    '15289006_column_leach_v1_20200828_pvo4': '006_jetti_project_file_pvo',
    '003_jetti_project_file_oxide_columns_beo_1': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_2': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_3': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_4': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_5': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_6': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_7': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_8': '003_jetti_project_file_amcf_head',
    '013_jetti_project_file_o_1': '013_jetti_project_file_combined',
    '013_jetti_project_file_o_2': '013_jetti_project_file_combined',
    '013_jetti_project_file_o_3': '013_jetti_project_file_combined',
    '013_jetti_project_file_o_4': '013_jetti_project_file_combined',
    '026_jetti_project_file_ps_1': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_ps_2': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_ps_3': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_ps_4': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_cr_1': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_cr_2': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_cr_3': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_cr_4': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_ss_1': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_ss_2': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_ss_3': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_ss_4': '026_jetti_project_file_sample_3_secondary_sulfide',
    '007_jetti_project_file_leopard_lep_1': '007_jetti_project_file_leopard_lep',
    '007_jetti_project_file_leopard_lep_2': '007_jetti_project_file_leopard_lep',
    '007_jetti_project_file_leopard_lep_3': '007_jetti_project_file_leopard_lep',
    '007_jetti_project_file_leopard_lep_4': '007_jetti_project_file_leopard_lep',
    '007b_jetti_project_file_tiger_tgr_1': '007b_jetti_project_file_tiger_tgr',
    '007b_jetti_project_file_tiger_tgr_2': '007b_jetti_project_file_tiger_tgr',
    '007b_jetti_project_file_tiger_tgr_3': '007b_jetti_project_file_tiger_tgr',
    '007b_jetti_project_file_tiger_tgr_4': '007b_jetti_project_file_tiger_tgr',
}


df_column_summary_filtered.loc[:, 'project_sample_id'] = df_column_summary_filtered['project_col_id'].copy()
for pattern, replacement in replacement_dict.items():
    escaped_pattern = re.escape(pattern)
    df_column_summary_filtered.loc[:, 'project_sample_id'] = df_column_summary_filtered['project_sample_id'].str.replace(f"^{escaped_pattern}$", replacement, regex=True)

df_column_summary_filtered[df_column_summary_filtered['project_name'] == '026 Jetti Project File']
df_column_summary_filtered[df_column_summary_filtered['project_name'] == '011 Jetti Project File-Crushed']


cols_to_match2 = [
    'project_sample_condition_id'
]

# 'project_cat_col_id' for 011 crushed is not being correctly handled. Replace values to make the correct match:
df_maker_index.loc[
    df_maker_index['project_cat1_col_id'] == '011_jetti_project_file_crushed_rm_7',
    'project_cat1_col_id'
] = '011_jetti_project_filecrushed_rm_7'

df_maker_index.loc[
    df_maker_index['project_cat2_col_id'] == '011_jetti_project_file_crushed_rm_8',
    'project_cat2_col_id'
] = '011_jetti_project_filecrushed_rm_8'

df_maker_index.loc[
    df_maker_index['project_control1_col_id'] == '011_jetti_project_file_crushed_rm_5',
    'project_control1_col_id'
] = '011_jetti_project_filecrushed_rm_5'

df_maker_index.loc[
    df_maker_index['project_control2_col_id'] == '011_jetti_project_file_crushed_rm_6',
    'project_control2_col_id'
] = '011_jetti_project_filecrushed_rm_6'

df_maker_index.loc[
    df_maker_index['project_cat_reactor_id'] == '011_jetti_project_file_crushed_rt_24',
    'project_cat_reactor_id'
] = '011_jetti_project_filecrushed_rt_24'

df_maker_index.loc[
    df_maker_index['project_control_reactor_id'] == '011_jetti_project_file_crushed_rt_21',
    'project_control_reactor_id'
] = '011_jetti_project_filecrushed_rt_21'

for index, row in df_column_summary_filtered.iterrows():
    mask = (normalize_dataframe_values(df_maker_index['project_cat1_col_id'].astype(str).str.replace('-', '_')) == normalize_and_replace(str(row['project_col_id']).replace('-', '_'))) | \
    (normalize_dataframe_values(df_maker_index['project_cat2_col_id'].astype(str).str.replace('-', '_')) == normalize_and_replace(str(row['project_col_id']).replace('-', '_'))) | \
    (normalize_dataframe_values(df_maker_index['project_control1_col_id'].astype(str).str.replace('-', '_')) == normalize_and_replace(str(row['project_col_id']).replace('-', '_'))) | \
    (normalize_dataframe_values(df_maker_index['project_control2_col_id'].astype(str).str.replace('-', '_')) == normalize_and_replace(str(row['project_col_id']).replace('-', '_')))

    # Extract the value(s) from df_column_summary
    matching_values = df_maker_index[mask][cols_to_match2].values

    df_column_summary_filtered.loc[index, 'project_sample_condition_id'] = matching_values[0][0] if len(matching_values) > 0 else None # type: ignore

df_column_summary_filtered[df_column_summary_filtered['project_name'] == '026 Jetti Project File']


replacement_dict_old_terminated = {
    '1528901a_column_leach_20200219_c1': '01a_jetti_project_file_c1',
    '1528901a_column_leach_20200219_c2': '01a_jetti_project_file_c2',
    '1528901a_column_leach_20200219_c3': '01a_jetti_project_file_c3',
    '1528901a_column_leach_20200219_c4': '01a_jetti_project_file_c4',
    '1528901a_column_leach_20200219_c5': '01a_jetti_project_file_c5',
    '1528901a_column_leach_20200219_c6': '01a_jetti_project_file_c6',
    '1528901a_column_leach_20200219_c7': '01a_jetti_project_file_c7',
    '1528901a_column_leach_20200219_c8': '01a_jetti_project_file_c8',
    '1528901a_column_leach_20200219_c9': '01a_jetti_project_file_c9',
    '1528901a_column_leach_20200219_c10': '01a_jetti_project_file_c10',
    '1528901a_column_leach_20200219_c11': '01a_jetti_project_file_c11',
    '1528901a_column_leach_20200219_c12': '01a_jetti_project_file_c12',
    '1528901a_column_leach_20200219_c13': '01a_jetti_project_file_c13',
    '1528901a_column_leach_20200219_c14': '01a_jetti_project_file_c14',
    '15289002_column_leach_v1_20191217_qb1': '002_jetti_project_file_qb1',
    '15289002_column_leach_v1_20191217_qb2': '002_jetti_project_file_qb2',
    '15289002_column_leach_v1_20191217_qb3': '002_jetti_project_file_qb3',
    '15289002_column_leach_v1_20191217_qb4': '002_jetti_project_file_qb4',
    '15289004_column_leach_v1_20201130_mo1': '004_jetti_project_file_mo1',
    '15289004_column_leach_v1_20201130_mo2': '004_jetti_project_file_mo2',
    '15289004_column_leach_v1_20201130_mo3': '004_jetti_project_file_mo3',
    '15289004_column_leach_v1_20201130_mo4': '004_jetti_project_file_mo4',
    '15289004_column_leach_v1_20201130_mols1': '004_jetti_project_file_mols1',
    '15289004_column_leach_v1_20201130_mols2': '004_jetti_project_file_mols2',
    '15289004_column_leach_v1_20201130_mols3': '004_jetti_project_file_mols3',
    '15289004_column_leach_v1_20201130_mols4': '004_jetti_project_file_mols4',
    '15289006_column_leach_v1_20200828_pvls1': '006_jetti_project_file_pvls1',
    '15289006_column_leach_v1_20200828_pvls2': '006_jetti_project_file_pvls2',
    '15289006_column_leach_v1_20200828_pvls3': '006_jetti_project_file_pvls3',
    '15289006_column_leach_v1_20200828_pvls4': '006_jetti_project_file_pvls4',
    '15289006_column_leach_v1_20200828_pvo1': '006_jetti_project_file_pvo1',
    '15289006_column_leach_v1_20200828_pvo2': '006_jetti_project_file_pvo2',
    '15289006_column_leach_v1_20200828_pvo3': '006_jetti_project_file_pvo3',
    '15289006_column_leach_v1_20200828_pvo4': '006_jetti_project_file_pvo4',
}
for pattern, replacement in replacement_dict_old_terminated.items():
    escaped_pattern = re.escape(pattern)
    df_column_summary_filtered.loc[:, 'project_col_id'] = df_column_summary_filtered['project_col_id'].str.replace(f"^{escaped_pattern}$", replacement, regex=True)

#special treatment for copperhead
conditions_to_match = ['015_jetti_project_file_amcf_pv4_6in', '015_jetti_project_file_amcf_pv4_8in']
result_df = pd.DataFrame()

for condition in conditions_to_match:
    duplicated_rows = df_column_summary_filtered[df_column_summary_filtered['project_sample_condition_id'] == condition].copy()
    duplicated_rows.loc[:, 'project_sample_condition_id'] = duplicated_rows['project_sample_condition_id'] + '_holdup'
    result_df = pd.concat([result_df, duplicated_rows], ignore_index=True)

df_column_summary_filtered = pd.concat([df_column_summary_filtered, result_df], ignore_index=True)

df_column_summary_filtered.to_csv(folder_path_save +'/df_column_summary.csv', sep=',')
df_column_summary_filtered.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/power_bi/df_column_summary.csv')

df_column_summary_filtered[df_column_summary_filtered['project_name'] == '026 Jetti Project File']

df_column_summary_filtered[df_column_summary_filtered['project_name'] == '007 Jetti Project File - Leopard']
df_column_summary_filtered[df_column_summary_filtered['project_name'] == '007B Jetti Project File - Tiger']
df_column_summary_filtered[df_column_summary_filtered['project_name'] == '011 Jetti Project File-Crushed']

# %% HEADERS FOR BOTTLES AND REACTORS

df_bottles_filtered = df_bottles[[
    'origin',
    'project_name',
    'start_cell',
    'time_(day)',
    'cu_extraction_actual_(%)'
]].copy()

df_bottles_filtered.loc[:, 'project_test_id'] = normalize_dataframe_values(df_bottles_filtered['project_name'].str.replace('-', '') + '_' + df_bottles_filtered['start_cell'].astype(str).str.split('-').str[1].replace('-', ''))
df_bottles_filtered.loc[:, 'reactor_name'] = df_bottles_filtered['start_cell'].astype(str).str.split('-').str[1].replace('-', '')

# line added because they change column names
df_reactors['time_(day)'] = df_reactors['time_(day)'].fillna(df_reactors['time_(days)'])

df_reactors_filtered = df_reactors[[
    'origin',
    'project_name',
    'start_cell',
    'time_(day)',
    'cu_extraction_actual_(%)'
]].copy()

df_reactors_filtered.loc[:, 'project_test_id'] = normalize_dataframe_values(df_reactors_filtered['project_name'].str.replace('-', '') + '_' + df_reactors_filtered['start_cell'].astype(str).str.split('-').str[1].replace('-', ''))
df_reactors_filtered.loc[:, 'reactor_name'] = df_reactors_filtered['start_cell'].astype(str).str.split('-').str[1].replace('-', '')

df_reactors_bottles = pd.concat([df_bottles_filtered, df_reactors_filtered], axis=0, join='outer')
df_reactors_bottles_export = df_reactors_bottles.copy()


nan_mask_catalyzed = df_maker_index[['file_name', 'catalyzed_reactor_1']].isnull().any(axis=1)
catalyzed_cols = np.where(nan_mask_catalyzed, np.nan, normalize_dataframe_values(df_maker_index['file_name'].astype(str).str.replace('-', '') + '_' + df_maker_index['catalyzed_reactor_1'].astype(str).str.replace('-', '')))

nan_mask_control = df_maker_index[['file_name', 'control_reactor_1']].isnull().any(axis=1)
control_cols = np.where(nan_mask_control, np.nan, normalize_dataframe_values(df_maker_index['file_name'].astype(str).str.replace('-', '') + '_' + df_maker_index['control_reactor_1'].astype(str).str.replace('-', '')))


df_master1 = pd.DataFrame()
df_master2 = pd.DataFrame()

df_master1[['origin', 'project_name', 'reactor_name', 'project_test_id', 'leach_duration_days', 'catalyzed_cu_extraction_actual_(%)']] = df_reactors_bottles_export[df_reactors_bottles_export['project_test_id'].isin(catalyzed_cols)][['origin', 'project_name', 'reactor_name', 'project_test_id', 'time_(day)', 'cu_extraction_actual_(%)']]
df_master2[['origin', 'project_name', 'reactor_name', 'project_test_id', 'leach_duration_days', 'control_cu_extraction_actual_(%)']] = df_reactors_bottles_export[df_reactors_bottles_export['project_test_id'].isin(control_cols)][['origin', 'project_name', 'reactor_name', 'project_test_id', 'time_(day)', 'cu_extraction_actual_(%)']]

df_master1['project_name'] = df_master1['project_name'].astype(str)
df_master2['project_name'] = df_master2['project_name'].astype(str)

df_master1['leach_duration_days'] = round(df_master1['leach_duration_days'], 0)
df_master2['leach_duration_days'] = round(df_master2['leach_duration_days'], 0)


df_master_reactors = pd.DataFrame()
df_master_reactors = df_master2.merge(df_master1, on=['origin', 'project_name', 'leach_duration_days'], how='outer')

cols_to_match = [
    'project_sample_id'
]


for index, row in df_master_reactors.iterrows():
    # Find the corresponding row(s) in df_column_summary based on conditions
    mask = (df_maker_index['file_name'] == row['project_name']) & \
    ((df_maker_index['project_cat_reactor_id'] == row['project_test_id_y']) & 
     (df_maker_index['project_control_reactor_id'] == row['project_test_id_x']))

    matching_values = df_maker_index[mask][cols_to_match].values

    df_master_reactors.loc[index, 'project_sample_id'] = matching_values[0][0] if len(matching_values) > 0 else None # type: ignore


df_master_reactors[['project_name', 'project_test_id_x', 'project_test_id_y']]


df_master_reactors.to_csv(folder_path_save +'/df_reactors.csv', sep=',')
df_master_reactors.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/power_bi/df_reactors.csv', sep=',')


df_master_reactors[df_master_reactors['project_name'] == '035 Jetti Project File']

# %% HEADERS FOR AC SUMMARY

df_acsummary_filtered = df_ac_summary[[
    'origin',
    'project_name',
    'start_cell',
    'test_id',
    'ore_type',
    'target_ph',
    'h2so4_kg_t'
]].copy()

df_acsummary_filtered.loc[:, 'project_sample_id'] = normalize_dataframe_values(df_acsummary_filtered['project_name'].astype(str).str.replace('-', '') +
                                                                        '_' + 
                                                                        df_acsummary_filtered['ore_type'].astype(str).str.replace('-', ''))

df_acsummary_filtered = df_acsummary_filtered[~pd.isnull(df_acsummary_filtered['test_id'])]

# Drop rows where 'start_cell' contains '24h'
df_acsummary_filtered = df_acsummary_filtered[~df_acsummary_filtered['start_cell'].str.contains('24h')]

# drop an aditional value for pH=2 in Leopard Bronces
df_acsummary_filtered = df_acsummary_filtered[~((df_acsummary_filtered['project_sample_id']=='jetti_project_file_leopard_scl_sample_los_bronces') & (df_acsummary_filtered['start_cell']=='tbl-Table4'))]

df_acsummary_filtered[df_acsummary_filtered['project_name']=='035 Jetti Project File']

sorted(df_acsummary_filtered['project_sample_id'].dropna().unique())
df_maker_index['project_sample_id'].unique()

replacement_dict = {
    '011_jetti_project_file_rm2020': '011_jetti_project_file_rm',
    '011_jetti_project_filecrushed_rm2024': '011_jetti_project_file_rm',
    '022_jetti_project_file_below_cutoff_grade': '022_jetti_project_file_stingray_1',
    '017_jetti_project_file_ea_mill_feed': '017_jetti_project_file_ea_mill_feed_combined',
    '020_jetti_project_file_hypogene_supergene_supergene': '020_jetti_project_file_hypogene_supergene_super',
    '020_jetti_project_file_hardy_and_waste_hardy': '020_jetti_project_file_hardy_and_waste_h21_master_comp',
    '020_jetti_project_file_hypogene_supergene_hypogene_comp': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite',
    '015_jetti_project_file_acmr': '015_jetti_project_file_amcf',
    '023_jetti_project_file_ot9': '023_jetti_project_file_ot_9',
    '023_jetti_project_file_ot10': '023_jetti_project_file_ot_10',
    '024_jetti_project_file': '024_jetti_project_file_024cv_cpy',
    '026_jetti_project_file_sample_1': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_sample_2': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_sample_3': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_sample_4': '026_jetti_project_file_sample_4_mixed_material',
    'jetti_file_elephant_ii_ver_2_pq_pq': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_ugm_ugm2': 'jetti_file_elephant_ii_ugm2',
    '007_jetti_project_file_leopard_leopard': '007_jetti_project_file_leopard_lep',
    '007b_jetti_project_file_tiger_tiger': '007b_jetti_project_file_tiger_tgr',
    '035_jetti_project_file_sample_#2': '035_jetti_project_file_sample_2',
    '035_jetti_project_file_sample_#3': '035_jetti_project_file_sample_3',
}

# Replace using the dictionary
for pattern, replacement in replacement_dict.items():
    df_acsummary_filtered['project_sample_id'] = df_acsummary_filtered['project_sample_id'].str.replace(pattern, replacement, regex=True)

# Duplicate AC data for crushed RM (011) and assign new label
rm_crushed_label = '011_jetti_project_file_rm_crushed'
if not (df_acsummary_filtered['project_sample_id'] == rm_crushed_label).any():
    df_acsummary_rm = df_acsummary_filtered[df_acsummary_filtered['project_sample_id'] == '011_jetti_project_file_rm'].copy()
    if not df_acsummary_rm.empty:
        df_acsummary_rm.loc[:, 'project_sample_id'] = rm_crushed_label
        df_acsummary_rm.loc[:, 'project_name'] = '011 Jetti Project File-Crushed'
        df_acsummary_filtered = pd.concat([df_acsummary_filtered, df_acsummary_rm], ignore_index=True)

df_acsummary_filtered.to_csv(folder_path_save + '/df_acsummary.csv', sep=',')
df_acsummary_filtered.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/power_bi/df_acsummary.csv', sep=',')

df_acsummary_filtered[df_acsummary_filtered['project_name'] == '007 Jetti Project File - Leopard']
df_acsummary_filtered[df_acsummary_filtered['project_name'] == '007B Jetti Project File - Tiger']

df_acsummary_filtered[df_acsummary_filtered['project_sample_id'] == '011_jetti_project_file_rm_crushed']
df_acsummary_filtered[df_acsummary_filtered['project_sample_id'] == '011_jetti_project_file_rm']


# %%  HEADERS FOR CHEMICHAL CHARACTERIZATION

df_chemchar_filtered = df_chemchar[[
    'origin',
    'project_name',
    'analyte_units',
    'cu_%',
    'fe_%',
    'cu_seq_h2so4_%',
    'cu_seq_nacn_%',
    'cu_seq_a_r_%'
]].copy()

'''
df_chemchar_filtered = df_chemchar_filtered.dropna(subset=['analyte_units']).loc[
    df_chemchar_filtered['analyte_units'].isin(df_maker_index['sample_id']) |
    df_chemchar_filtered['analyte_units'].isin(df_maker_index['sample_id_2'])
]
'''

df_chemchar_filtered.loc[:, 'project_sample_id'] = normalize_dataframe_values(df_chemchar_filtered['project_name'].str.replace('-', '') + '_' + df_chemchar_filtered['analyte_units'].astype(str).str.replace('-', ''))

# Remove '_dup', '_(a)', or '_(b)' from 'project_sample_id' column
df_chemchar_filtered.loc[:, 'project_sample_id'] = df_chemchar_filtered['project_sample_id'].str.replace(r'_dup$|_\((a|b)\)$', '', regex=True)

# Define custom aggregation functions
agg_funcs = {}
for col in df_chemchar_filtered.columns:
    if pd.api.types.is_numeric_dtype(df_chemchar_filtered[col]):
        agg_funcs[col] = 'mean'
    else:
        agg_funcs[col] = 'first'

# Group by 'project_sample_id' and aggregate using custom functions
df_chemchar_filtered = df_chemchar_filtered.groupby('project_sample_id', as_index=False).agg(agg_funcs)


df_chemchar_filtered[df_chemchar_filtered['project_name'] == '020 Jetti Project File Hypogene Supergene']['project_sample_id']
df_chemchar_filtered[df_chemchar_filtered['project_name'] == '030 Jetti Project File']

df_chemchar_filtered['acid_soluble_%'] = df_chemchar_filtered['cu_seq_h2so4_%'] / df_chemchar_filtered[['cu_seq_h2so4_%', 'cu_seq_nacn_%', 'cu_seq_a_r_%']].sum(skipna=True, axis=1) * 100
df_chemchar_filtered['cyanide_soluble_%'] = df_chemchar_filtered['cu_seq_nacn_%'] / df_chemchar_filtered[['cu_seq_h2so4_%', 'cu_seq_nacn_%', 'cu_seq_a_r_%']].sum(skipna=True, axis=1) * 100
df_chemchar_filtered['residual_cpy_%'] = df_chemchar_filtered['cu_seq_a_r_%'] / df_chemchar_filtered[['cu_seq_h2so4_%', 'cu_seq_nacn_%', 'cu_seq_a_r_%']].sum(skipna=True, axis=1) * 100

df_chemchar_filtered['project_sample_id'].unique()
df_maker_index['project_sample_id'].unique()

replacement_dict = {
    'jetti_project_file_elephant_(site)_elephant': 'jetti_project_file_elephant_site',
    '015_jetti_project_file_amcf_head': '015_jetti_project_file_amcf',
    '014_jetti_project_file_bag_head': '014_jetti_project_file_bag',
    '014_jetti_project_file_kmb_head': '014_jetti_project_file_kmb',
    '024_jetti_project_file_024cvcpy_avg': '024_jetti_project_file_024cv_cpy',
    '023_jetti_project_file_ot10_avg': '023_jetti_project_file_ot_10',
    '023_jetti_project_file_ot9_avg': '023_jetti_project_file_ot_9',
    # '003_jetti_project_file_amcf_head': '003_jetti_project_file_amcf',
    # '003 oxide': '',
    '011_jetti_project_file_rm_head_sample': '011_jetti_project_file_rm',
    '011_jetti_project_filecrushed_rm_2024': '011_jetti_project_file_rm_crushed',
    '012_jetti_project_file_cs_quebalix': '012_jetti_project_file_quebalix',
    '012_jetti_project_file_cs_incremento': '012_jetti_project_file_incremento',
    '013_jetti_project_file_combined_head_average': '013_jetti_project_file_combined',
    # '020_jetti_project_file_hypogene_supergene_hypogene_master_composite': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite',
    # '020_jetti_project_file_hypogene_supergene_super': '020_jetti_project_file_hypogene_supergene_super',
    # '020_jetti_project_file_hardy_and_waste_h21_master_comp': '020_jetti_project_file_hardy_and_waste_h21_master_comp',
    '022_jetti_project_file_stingray_1_head': '022_jetti_project_file_stingray_1',
    '026_jetti_project_file_sample_#1_(primary_sulfide)': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_sample_#2_(carrizalillo)': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_sample_#3_(secondary_sulfide)': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_sample_#4_(mixed_material)': '026_jetti_project_file_sample_4_mixed_material',
    'jetti_file_elephant_ii_ver_2_pq_pq_average': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_ugm_ugm_average': 'jetti_file_elephant_ii_ugm2',
    '1528901a_column_leach_20200219_muestra_1_head': '01a_jetti_project_file_c',
    '15289002_column_leach_v1_20191217_qb': '002_jetti_project_file_qb',
    '15289004_column_leach_v1_20201130_bagdad': '004_jetti_project_file_mols',
    '15289004_column_leach_v1_20201130_morenci': '004_jetti_project_file_mo',
    '15289006_column_leach_v1_20200828_pvo': '006_jetti_project_file_pvo',
    '15289006_column_leach_v1_20200828_pvls': '006_jetti_project_file_pvls',
    '007_jetti_project_file_leopard_leopard': '007_jetti_project_file_leopard_lep',
    '007_jetti_project_file_leopard_leopard_(dup)': '007_jetti_project_file_leopard_lep',
    '007b_jetti_project_file_tiger_tiger_head': '007b_jetti_project_file_tiger_tgr',
    '035_jetti_project_file_sample_#2_jabal_shayban_(avg)': '035_jetti_project_file_sample_2',
    '035_jetti_project_file_sample_3_avg': '035_jetti_project_file_sample_3',

}
set(df_chemchar_filtered['project_sample_id'].unique()) - set(replacement_dict.keys())

df_chemchar_filtered['project_sample_id_original'] = df_chemchar_filtered['project_sample_id']
for pattern, replacement in replacement_dict.items():
    escaped_pattern = re.escape(pattern)
    df_chemchar_filtered['project_sample_id'] = df_chemchar_filtered['project_sample_id'].str.replace(f"^{escaped_pattern}$", replacement, regex=True)

'''
# Special treatment for Elephant Site
    
df_concat_site = df_column_summary_filtered[df_column_summary_filtered['project_sample_condition_id'] == 'jetti_project_file_elephant_site'][['origin', 'project_name', 'project_sample_id', 'feed_head_cu_%', 'feed_head_fe_%']]
df_concat_site.rename(columns = {
    'feed_head_cu_%': 'cu_%',
    'feed_head_fe_%': 'fe_%',
}, inplace=True)
df_chemchar_filtered =  pd.concat([df_chemchar_filtered, df_concat_site], join='outer', axis=0, ignore_index=True)
'''

df_chemchar_filtered.to_csv(folder_path_save +'/df_chemchar.csv', sep=',')
df_chemchar_filtered.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/power_bi/df_chemchar.csv', sep=',')
df_chemchar_filtered[df_chemchar_filtered['project_sample_id'] == '020_jetti_project_file_hypogene_supergene_super']
df_chemchar_filtered[df_chemchar_filtered['project_sample_id'] == '020_jetti_project_file_hardy_and_waste_h21_master_comp']

df_chemchar_filtered[df_chemchar_filtered['project_sample_id'] == '026_jetti_project_file_sample_3_secondary_sulfide']

df_chemchar_filtered[df_chemchar_filtered['project_name'] == '007 Jetti Project File - Leopard']
df_chemchar_filtered[df_chemchar_filtered['project_name'] == '007B Jetti Project File - Tiger']

df_chemchar_filtered[df_chemchar_filtered['project_name'] == '011 Jetti Project File-Crushed']

# %% HEADERS FOR LEACHING PERFORMANCE

df_leaching_performance = df_merged[[
    'origin', 
    'project_name', 
    'sheet_name', 
    'catalyzed',
    'column_status',
    'leach_duration_days',
    'pls_cu_mg_l',
    'cu_recovery', #this is for the df_leaching_performance_terminated_projects which is no longer in use
    'cu_recovery_w_holdup_soln_%',
    'cu_recovery_w_gre_inventory_%',
    'cu_t_recovery_%'
]].copy()

df_leaching_performance = df_leaching_performance[pd.notnull(df_leaching_performance['pls_cu_mg_l'])]
df_leaching_performance = df_leaching_performance[list(set(df_leaching_performance.columns) - {'pls_cu_mg_l'})]

df_leaching_performance[
    (df_leaching_performance['project_name'] == '020 Jetti Project File Hypogene_Supergene') & 
    (df_leaching_performance['sheet_name'] == 'HYP_1')
    ]

# Special treatment for Elephant II (reunion con Nicolas Lorca 19 de julio 2024)
df_leaching_performance = df_leaching_performance[
    ~((df_leaching_performance['sheet_name'].isin(['UC_1', 'UC_3'])) & (df_leaching_performance['leach_duration_days'] < 41))
]

df_leaching_performance.loc[:, 'cu_recovery_%'] = df_leaching_performance['cu_recovery'].fillna(df_leaching_performance['cu_t_recovery_%']).fillna(df_leaching_performance['cu_recovery_w_gre_inventory_%']).fillna(df_leaching_performance['cu_recovery_w_holdup_soln_%'])

#=========== Special treatment for Copperhead project 015 for Cu Recoveries with HoldUp
df_leaching_015_gre_inv = df_leaching_performance[df_leaching_performance['cu_recovery_w_gre_inventory_%'].notnull()][['origin', 'project_name', 'sheet_name', 'catalyzed', 'column_status', 'leach_duration_days', 'cu_recovery_w_gre_inventory_%']]
df_leaching_015_holdup_soln = df_leaching_performance[df_leaching_performance['cu_recovery_w_holdup_soln_%'].notnull()][['origin', 'project_name', 'sheet_name', 'catalyzed', 'column_status', 'leach_duration_days', 'cu_recovery_w_holdup_soln_%']]

df_leaching_015_gre_inv.rename(columns={
    'cu_recovery_w_gre_inventory_%': 'cu_recovery_%',
}, inplace=True)
df_leaching_015_holdup_soln.rename(columns={
    'cu_recovery_w_holdup_soln_%': 'cu_recovery_%',
}, inplace=True)

df_leaching_015_gre_inv.loc[:, 'condition'] = 'w_gre_inventory'
df_leaching_015_holdup_soln.loc[:, 'condition'] = 'w_holdup_sln'
df_leaching_015_gre_inv.reset_index(drop=True, inplace=True)
df_leaching_015_holdup_soln.reset_index(drop=True, inplace=True)
#df_leaching_performance.reset_index(drop=True, inplace=True)

df_leaching_performance = pd.concat([df_leaching_performance, df_leaching_015_gre_inv, df_leaching_015_holdup_soln], axis=0, join='outer', ignore_index=True)

df_leaching_performance.loc[:, 'project_col_id'] = normalize_dataframe_values(df_leaching_performance['project_name'].str.replace('-', '') + '_' + df_leaching_performance['sheet_name'].str.replace('-', ''))

df_leaching_performance['project_col_condition_id'] = df_leaching_performance.apply(
    lambda row: normalize_and_replace(
        row['project_name'].replace('-', '') + '_' +
        row['sheet_name'].replace('-', '') +
        ('_' + row['condition'].replace('-', '') if pd.notnull(row['condition']) else '')
    ),
    axis=1
)

df_leaching_performance.loc[:, 'col_name'] = normalize_dataframe_values(df_leaching_performance['sheet_name'].str.replace('-', ''))

non_nan_elements = df_maker_index[['catalyzed_col_1', 'catalyzed_col_2', 'control_col_1', 'control_col_2']].values.flatten()
non_nan_elements = non_nan_elements[~pd.isnull(non_nan_elements)].tolist()
non_nan_elements = normalize_series_values(pd.Series(non_nan_elements)).str.replace('-', '_').tolist()

terminated_old_columns = [
    'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14',
    'qb1', 'qb2', 'qb3', 'qb4',
    'pvls1', 'pvls2', 'pvls3', 'pvls4',
    'pvo1', 'pvo2', 'pvo3', 'pvo4',
    'mo1', 'mo2', 'mo3', 'mo4',
    'mols1', 'mols2', 'mols3', 'mols4',
]


essential_cols = non_nan_elements + terminated_old_columns

df_leaching_performance = df_leaching_performance[normalize_dataframe_values(df_leaching_performance['sheet_name']).str.replace('-', '_').isin(essential_cols)].copy()

list(df_maker_index['project_sample_condition_id'].unique())
list(df_leaching_performance['project_col_condition_id'].unique())

replacement_dict_leaching = {
    '1528901a_column_leach_20200219_c1': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c2': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c3': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c4': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c5': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c6': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c7': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c8': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c9': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c10': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c11': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c12': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c13': '01a_jetti_project_file_c',
    '1528901a_column_leach_20200219_c14': '01a_jetti_project_file_c',
    '15289002_column_leach_v1_20191217_qb1': '002_jetti_project_file_qb',
    '15289002_column_leach_v1_20191217_qb2': '002_jetti_project_file_qb',
    '15289002_column_leach_v1_20191217_qb3': '002_jetti_project_file_qb',
    '15289002_column_leach_v1_20191217_qb4': '002_jetti_project_file_qb',
    '15289004_column_leach_v1_20201130_mo1': '004_jetti_project_file_mo',
    '15289004_column_leach_v1_20201130_mo2': '004_jetti_project_file_mo',
    '15289004_column_leach_v1_20201130_mo3': '004_jetti_project_file_mo',
    '15289004_column_leach_v1_20201130_mo4': '004_jetti_project_file_mo',
    '15289004_column_leach_v1_20201130_mols_1': '004_jetti_project_file_mols',
    '15289004_column_leach_v1_20201130_mols_2': '004_jetti_project_file_mols',
    '15289004_column_leach_v1_20201130_mols_3': '004_jetti_project_file_mols',
    '15289004_column_leach_v1_20201130_mols_4': '004_jetti_project_file_mols',
    '15289006_column_leach_v1_20200828_pvls1': '006_jetti_project_file_pvls',
    '15289006_column_leach_v1_20200828_pvls2': '006_jetti_project_file_pvls',
    '15289006_column_leach_v1_20200828_pvls3': '006_jetti_project_file_pvls',
    '15289006_column_leach_v1_20200828_pvls4': '006_jetti_project_file_pvls',
    '15289006_column_leach_v1_20200828_pvo1': '006_jetti_project_file_pvo',
    '15289006_column_leach_v1_20200828_pvo2': '006_jetti_project_file_pvo',
    '15289006_column_leach_v1_20200828_pvo3': '006_jetti_project_file_pvo',
    '15289006_column_leach_v1_20200828_pvo4': '006_jetti_project_file_pvo',
    '012_jetti_project_file_cs_q_2': '012_jetti_project_file_quebalix',
    '012_jetti_project_file_cs_q_3': '012_jetti_project_file_quebalix',
    '012_jetti_project_file_cs_i_2': '012_jetti_project_file_incremento',
    '012_jetti_project_file_cs_i_3': '012_jetti_project_file_incremento',
    '011_jetti_project_file_rm_1': '011_jetti_project_file_rm',
    '011_jetti_project_file_rm_2': '011_jetti_project_file_rm',
    '011_jetti_project_filecrushed_011rm_5': '011_jetti_project_file_rm_crushed',
    '011_jetti_project_filecrushed_011rm_6': '011_jetti_project_file_rm_crushed',
    '011_jetti_project_filecrushed_011rm_7': '011_jetti_project_file_rm_crushed',
    '011_jetti_project_filecrushed_011rm_8': '011_jetti_project_file_rm_crushed',
    '024_jetti_project_file_cv_1': '024_jetti_project_file_024cv_cpy',
    '024_jetti_project_file_cv_4': '024_jetti_project_file_024cv_cpy',
    '015_jetti_project_file_c_1': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_6': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_11': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_2': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_7': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_12': '015_jetti_project_file_amcf',
    '023_jetti_project_file_ea_1': '',
    '023_jetti_project_file_ea_4': '',
    '014_jetti_project_file_k_1': '014_jetti_project_file_kmb',
    '014_jetti_project_file_k_4': '014_jetti_project_file_kmb',
    '014_jetti_project_file_b_2': '014_jetti_project_file_bag',
    '014_jetti_project_file_b_4': '014_jetti_project_file_bag',
    '021_jetti_project_file_c_1': '021_jetti_project_file_hypogene',
    '021_jetti_project_file_c_2': '021_jetti_project_file_hypogene',
    '021_jetti_project_file_c_3': '021_jetti_project_file_hypogene',
    '003_jetti_project_file_be_1': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_be_2': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_be_3': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_be_4': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_be_5': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_be_6': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_be_7': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_be_8': '003_jetti_project_file_amcf_head',
    '030_jetti_project_file_cpy_1': '030_jetti_project_file_cpy',
    '030_jetti_project_file_cpy_2': '030_jetti_project_file_cpy',
    '030_jetti_project_file_ss_1': '030_jetti_project_file_ss',
    '030_jetti_project_file_ss_2': '030_jetti_project_file_ss',
    '020_jetti_project_file_hardy_and_waste_har_1': '020_jetti_project_file_hardy_and_waste_h21_master_comp',
    '020_jetti_project_file_hardy_and_waste_har_2': '020_jetti_project_file_hardy_and_waste_h21_master_comp',
    '020_jetti_project_file_hardy_and_waste_har_3': '020_jetti_project_file_hardy_and_waste_h21_master_comp',
    '022_jetti_project_file_s_1': '022_jetti_project_file_stingray_1',
    '022_jetti_project_file_s_2': '022_jetti_project_file_stingray_1',
    '020_jetti_project_file_hypogene_supergene_sup_1': '020_jetti_project_file_hypogene_supergene_super',
    '020_jetti_project_file_hypogene_supergene_sup_2': '020_jetti_project_file_hypogene_supergene_super',
    '020_jetti_project_file_hypogene_supergene_sup_3': '020_jetti_project_file_hypogene_supergene_super',
    '020_jetti_project_file_hypogene_supergene_hyp_1': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite',
    '020_jetti_project_file_hypogene_supergene_hyp_2': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite',
    '020_jetti_project_file_hypogene_supergene_hyp_3': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite',
    '017_jetti_project_file_ea_1': '017_jetti_project_file_ea_mill_feed_combined',
    '017_jetti_project_file_ea_4': '017_jetti_project_file_ea_mill_feed_combined',
    'jetti_project_file_zaldivar_scl_col69': 'jetti_project_file_zaldivar_scl_sample_zaldivar',
    'jetti_project_file_zaldivar_scl_col70': 'jetti_project_file_zaldivar_scl_sample_zaldivar',
    'jetti_project_file_leopard_scl_col1': 'jetti_project_file_leopard_scl_sample_los_bronces_crushed',
    'jetti_project_file_leopard_scl_col2': 'jetti_project_file_leopard_scl_sample_los_bronces_crushed',
    'jetti_project_file_leopard_scl_rom1': 'jetti_project_file_leopard_scl_sample_los_bronces_rom',
    'jetti_project_file_leopard_scl_rom2': 'jetti_project_file_leopard_scl_sample_los_bronces_rom',
    'jetti_project_file_elephant_scl_col42': 'jetti_project_file_elephant_scl_sample_escondida_santiago',
    'jetti_project_file_elephant_scl_col43': 'jetti_project_file_elephant_scl_sample_escondida_santiago',
    'jetti_project_file_tiger_rom_rom1': '',
    'jetti_project_file_tiger_rom_rom2': 'jetti_project_file_tiger_rom_m1_rom',
    'jetti_project_file_tiger_rom_rom3': 'jetti_project_file_tiger_rom_m1_rom',
    'jetti_project_file_tiger_rom_c_4': 'jetti_project_file_tiger_rom_m2_crushed',
    'jetti_project_file_tiger_rom_c_5': 'jetti_project_file_tiger_rom_m2_crushed',
    'jetti_project_file_tiger_rom_c_6': 'jetti_project_file_tiger_rom_m2_crushed',
    'jetti_project_file_tiger_rom_c_7': 'jetti_project_file_tiger_rom_m2_crushed',
    'jetti_project_file_tiger_rom_c_8': 'jetti_project_file_tiger_rom_m2_crushed',
    'jetti_project_file_tiger_rom_c_9': 'jetti_project_file_tiger_rom_m2_crushed',
    'jetti_file_elephant_ii_ver_2_ugm_ur_1': 'jetti_file_elephant_ii_ugm2_rom',
    'jetti_file_elephant_ii_ver_2_ugm_ur_2': 'jetti_file_elephant_ii_ugm2_rom',
    'jetti_file_elephant_ii_ver_2_ugm_uc_1': 'jetti_file_elephant_ii_ugm2_crushed',
    'jetti_file_elephant_ii_ver_2_ugm_uc_3': 'jetti_file_elephant_ii_ugm2_crushed',
    'jetti_file_elephant_ii_ver_2_ugm_uc_4': 'jetti_file_elephant_ii_ugm2_crushed',
    'jetti_file_elephant_ii_ver_2_ugm_uc_7': 'jetti_file_elephant_ii_ugm2_crushed',
    'jetti_file_elephant_ii_ver_2_pq_pr_1': 'jetti_file_elephant_ii_pq_rom',
    'jetti_file_elephant_ii_ver_2_pq_pr_2': 'jetti_file_elephant_ii_pq_rom',
    'jetti_file_elephant_ii_ver_2_pq_pc_2': 'jetti_file_elephant_ii_pq_crushed',
    'jetti_file_elephant_ii_ver_2_pq_pc_4': 'jetti_file_elephant_ii_pq_crushed',
    'jetti_file_elephant_ii_ver_2_pq_pc_7': 'jetti_file_elephant_ii_pq_crushed',
    'jetti_project_file_elephant_(site)_fat4': 'jetti_project_file_elephant_site',
    'jetti_project_file_elephant_(site)_fat6': 'jetti_project_file_elephant_site',
    'jetti_project_file_toquepala_scl_col63': 'jetti_project_file_toquepala_scl_sample_fresca',
    'jetti_project_file_toquepala_scl_col64': 'jetti_project_file_toquepala_scl_sample_fresca',
    '015_jetti_project_file_c_1_w_gre_inventory': '',
    '015_jetti_project_file_c_6_w_gre_inventory': '015_jetti_project_file_amcf_pv4_8in_holdup',
    '015_jetti_project_file_c_11_w_gre_inventory': '015_jetti_project_file_amcf_pv4_8in_holdup',
    '015_jetti_project_file_c_2_w_gre_inventory': '',
    '015_jetti_project_file_c_7_w_gre_inventory': '015_jetti_project_file_amcf_pv4_6in_holdup',
    '015_jetti_project_file_c_12_w_gre_inventory': '015_jetti_project_file_amcf_pv4_6in_holdup',
    '015_jetti_project_file_c_1_w_holdup_sln': '',
    '015_jetti_project_file_c_6_w_holdup_sln': '015_jetti_project_file_amcf_pv4_8in_holdup',
    '015_jetti_project_file_c_11_w_holdup_sln': '015_jetti_project_file_amcf_pv4_8in_holdup',
    '015_jetti_project_file_c_2_w_holdup_sln': '',
    '015_jetti_project_file_c_7_w_holdup_sln': '015_jetti_project_file_amcf_pv4_6in_holdup',
    '015_jetti_project_file_c_12_w_holdup_sln': '015_jetti_project_file_amcf_pv4_6in_holdup',
    '003_jetti_project_file_oxide_columns_beo_1': '003_jetti_project_file_amcf_head', # considerarr
    '003_jetti_project_file_oxide_columns_beo_2': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_3': '003_jetti_project_file_amcf_head', # considerar
    '003_jetti_project_file_oxide_columns_beo_4': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_5': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_6': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_7': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_8': '003_jetti_project_file_amcf_head',
    '013_jetti_project_file_o_1': '013_jetti_project_file_combined', # descartar
    '013_jetti_project_file_o_2': '013_jetti_project_file_combined',
    '013_jetti_project_file_o_3': '013_jetti_project_file_combined',
    '013_jetti_project_file_o_4': '013_jetti_project_file_combined', # descartar
    '026_jetti_project_file_ps_1': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_ps_2': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_ps_3': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_ps_4': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_cr_1': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_cr_2': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_cr_3': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_cr_4': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_ss_1': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_ss_2': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_ss_3': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_ss_4': '026_jetti_project_file_sample_3_secondary_sulfide',
    '007_jetti_project_file_leopard_lep_1': '007_jetti_project_file_leopard_lep',
    '007_jetti_project_file_leopard_lep_2': '007_jetti_project_file_leopard_lep',
    '007_jetti_project_file_leopard_lep_3': '007_jetti_project_file_leopard_lep',
    '007_jetti_project_file_leopard_lep_4': '007_jetti_project_file_leopard_lep',
    '007b_jetti_project_file_tiger_tgr_1': '007b_jetti_project_file_tiger_tgr',
    '007b_jetti_project_file_tiger_tgr_2': '007b_jetti_project_file_tiger_tgr',
    '007b_jetti_project_file_tiger_tgr_3': '007b_jetti_project_file_tiger_tgr',
    '007b_jetti_project_file_tiger_tgr_4': '007b_jetti_project_file_tiger_tgr',
}
set(list(df_leaching_performance['project_col_condition_id'].unique())) - set(replacement_dict_leaching.keys())


df_leaching_performance['project_sample_condition_id'] = df_leaching_performance['project_col_condition_id'].copy()
for pattern, replacement in replacement_dict_leaching.items():
    escaped_pattern = re.escape(pattern)
    df_leaching_performance['project_sample_condition_id'] = df_leaching_performance['project_sample_condition_id'].str.replace(f"^{escaped_pattern}$", replacement, regex=True)

df_leaching_performance['leach_duration_days'] = df_leaching_performance['leach_duration_days'].round(0)


df_leaching_performance[
    (df_leaching_performance['project_name'] == '020 Jetti Project File Hypogene_Supergene') & 
    (df_leaching_performance['sheet_name'] == 'HYP_1')
    ]

# Group by leach_duration_days
grouped_df = df_leaching_performance.groupby('leach_duration_days').agg(
    {
        'cu_t_recovery_%': 'mean',
        'cu_recovery_%': 'mean',
        **{col: 'last' for col in df_leaching_performance.columns if col not in ['leach_duration_days', 'cu_t_recovery_%', 'cu_recovery_%']}
    }
).reset_index()

df_master = df_leaching_performance.copy()

df_master = df_master[df_master['cu_recovery_%'] > 0]

df_master.to_csv(folder_path_save +'/df_master.csv', sep=',')


df_master[df_master['project_name'] == '007 Jetti Project File - Leopard']
df_master[df_master['project_name'] == '007B Jetti Project File - Tiger']

# %% HEADERS FOR SEPARATION

df_separation_merged = pd.DataFrame()
df_separation_merged[['origin', 'project_name', 'start_cell', 'actual', 'adjusted']] = df_separation[['origin', 'project_name', 'start_cell', 'actual', 'adjusted']].copy()
df_separation_merged['project_col_id'] = normalize_dataframe_values(df_separation['project_name'].astype(str).str.replace('-', '') + '_' + df_separation['start_cell'].astype(str).str.replace('-', '_')).copy()
df_separation['project_col_id'] = normalize_dataframe_values(df_separation['project_name'].astype(str).str.replace('-', '') + '_' + df_separation['start_cell'].astype(str).str.replace('-', '_')).copy()

column_names_by_project = {}
for project_id in df_separation['project_col_id'].unique():
    project_df = df_separation[df_separation['project_col_id'] == project_id]
    non_empty_columns = project_df.columns[project_df.notna().any()]
    column_names_by_project[project_id] = non_empty_columns.tolist()

for project_id, column_names in column_names_by_project.items():
    print(f"=====\nFor project_col_id {project_id}: {', '.join(column_names)}")

# Print columns containing ('control' and '1') and ('control' and '2')  or 'catalyzed' in their names
control_1_columns = [col for col in df_separation.columns if 'control' in col and '1' in col]
control_2_columns = [col for col in df_separation.columns if 'control' in col and '2' in col]
control_only_columns = [col for col in df_separation.columns if 'control' in col and '1' not in col and '2' not in col and not 'avg' in col and not 'average' in col and not 'delta' in col and not 'cat' in col]
catalyzed_columns = [col for col in df_separation.columns if 'catalyzed' in col and not 'rate' in col]

print("Columns with 'control' and '1':")
for c in control_1_columns:
    print(c)

print("\nColumns with 'control' and '2':")
for c in control_2_columns:
    print(c)

print("\nColumns with 'control' only:")
for c in control_only_columns:
    print(c)

print("\nColumns with 'catalyzed':")
for c in catalyzed_columns:
    print(c)


df_separation.filter(regex='lep').columns
df_separation.filter(regex='tgr').columns

df_separation_merged['control_col_1'] = df_separation['control_rm_2']\
    .fillna(df_separation['control_1_(cv_1)']) \
    .fillna(df_separation['control_-_c_11']) \
    .fillna(df_separation['control_-_c_7']) \
    .fillna(df_separation['control_excluding_inventory_ore_only_(west_demo_heap)']) \
    .fillna(df_separation['control_1_(kmb_1)']) \
    .fillna(df_separation['control_1_(bag_1)']) \
    .fillna(df_separation['control_1_(har_1)']) \
    .fillna(df_separation['control_-_s_1']) \
    .fillna(df_separation['control_1_(sup_1)']) \
    .fillna(df_separation['control_1_(hyp_1)']) \
    .fillna(df_separation['control_1_(ea_1)']) \
    .fillna(df_separation['control_-_col_70']) \
    .fillna(df_separation['control_-_col_2']) \
    .fillna(df_separation['control_-_col_43']) \
    .fillna(df_separation['control_-_rom1']) \
    .fillna(df_separation['control_-_c5']) \
    .fillna(df_separation['control_-_c7']) \
    .fillna(df_separation['control_-_c9']) \
    .fillna(df_separation['control_1_fat-6']) \
    .fillna(df_separation['control_-_col_64']) \
    .fillna(df_separation['control_1_(ps_1)']) \
    .fillna(df_separation['control_1_(cr_1)']) \
    .fillna(df_separation['control_1_(ss_1)']) \
    .fillna(df_separation['control_1_(lep_1)']) \
    .fillna(df_separation['control_1_(tgr_1)']) \
    .fillna(df_separation['control_1_(be_1)']) \
    .fillna(df_separation['control_i_3']) \
    .fillna(df_separation['control_q_3']) \
    .fillna(df_separation['control_-_col_70'])    
    
df_separation_merged['control_col_2'] = df_separation['control_rm_2']\
    .fillna(df_separation['control_2_(cv_2)']) \
    .fillna(df_separation['control_-_c_11']) \
    .fillna(df_separation['control_-_c_7']) \
    .fillna(df_separation['control_excluding_inventory_ore_only_(west_demo_heap)']) \
    .fillna(df_separation['control_1_(kmb_1)']) \
    .fillna(df_separation['control_1_(bag_1)']) \
    .fillna(df_separation['control_2_(har_2)']) \
    .fillna(df_separation['control_-_s_1']) \
    .fillna(df_separation['control_2_(sup_3)']) \
    .fillna(df_separation['control_2_(hyp_3)']) \
    .fillna(df_separation['control_1_(ea_1)']) \
    .fillna(df_separation['control_-_col_70']) \
    .fillna(df_separation['control_-_col_2']) \
    .fillna(df_separation['control_-_col_43']) \
    .fillna(df_separation['control_-_rom1']) \
    .fillna(df_separation['control_-_c5']) \
    .fillna(df_separation['control_-_c7']) \
    .fillna(df_separation['control_-_c9']) \
    .fillna(df_separation['control_1_fat-6']) \
    .fillna(df_separation['control_-_col_64']) \
    .fillna(df_separation['control_2_(ps_2)']) \
    .fillna(df_separation['control_2_(cr_2)']) \
    .fillna(df_separation['control_2_(ss_2)']) \
    .fillna(df_separation['control_2_(lep_2)_open']) \
    .fillna(df_separation['control_2_(tgr_2)']) \
    .fillna(df_separation['control_2_(be_8)']) \

df_separation_merged['catalyzed_col_max'] = df_separation['catalyzed_rm_1']\
    .fillna(df_separation['max_catalyzed_(cv_4)']) \
    .fillna(df_separation['catalyzed_-_c_6']) \
    .fillna(df_separation['catalyzed_-_c_12']) \
    .fillna(df_separation['catalyzed_excluding_inventory_ore_only_(east_demo_heap)']) \
    .fillna(df_separation['max_catalyzed_(kmb_4)']) \
    .fillna(df_separation['max_catalyzed_(bag_4)']) \
    .fillna(df_separation['max_catalyzed_(har_3)']) \
    .fillna(df_separation['catalyzed_-_s_2']) \
    .fillna(df_separation['max_catalyzed_(sup_2)']) \
    .fillna(df_separation['max_catalyzed_(hyp_2)']) \
    .fillna(df_separation['max_catalyzed_(ea_4)']) \
    .fillna(df_separation['catalyzed_-_col_69']) \
    .fillna(df_separation['catalyzed_-_col_1']) \
    .fillna(df_separation['catalyzed_-_col_42']) \
    .fillna(df_separation['catalyzed_-_rom2']) \
    .fillna(df_separation['catalyzed_-_c4']) \
    .fillna(df_separation['catalyzed_-_c6']) \
    .fillna(df_separation['catalyzed_-_c8']) \
    .fillna(df_separation['max_catalyzed_fat-4']) \
    .fillna(df_separation['catalyzed_-_col_63']) \
    .fillna(df_separation['max_catalyzed_(ps_4)']) \
    .fillna(df_separation['max_catalyzed_(cr_4)']) \
    .fillna(df_separation['max_catalyzed_(ss_4)']) \
    .fillna(df_separation['catalyzed_(lep_4)_closed']) \
    .fillna(df_separation['max_catalyzed_(tgr_4)']) \
    .fillna(df_separation['catalyzed_i_2']) \
    .fillna(df_separation['catalyzed_q_2']) \
    .fillna(df_separation['max_catalyzed_(be_2)'])    
    

df_separation_merged['project_col_id'].unique()
replacement_dict = {
    '003_jetti_project_file_tbl_separationtable': '003_jetti_project_file_amcf_head',
    '011_jetti_project_file_tbl_table42': '011_jetti_project_file_rm',
    '011_jetti_project_filecrushed_tbl_separationtable': '011_jetti_project_file_rm_crushed',
    '012_jetti_project_file_cs_tbl_table42': '012_jetti_project_file_incremento',
    '012_jetti_project_file_cs_tbl_table4254': '012_jetti_project_file_quebalix',
    '013_jetti_project_file_tbl_table42': '013_jetti_project_file_combined',
    '014_jetti_project_file_tbl_table42': '014_jetti_project_file_kmb',
    '014_jetti_project_file_tbl_table43': '014_jetti_project_file_bag',
    '015_jetti_project_file_tbl_separation_pv4_rom8': '015_jetti_project_file_amcf_pv4_8in',
    '015_jetti_project_file_tbl_separation_pv4_romhold': '015_jetti_project_file_amcf_pv4_8in_holdup',
    '015_jetti_project_file_tbl_nelson_pv4_rom6': '015_jetti_project_file_amcf_pv4_6in',
    '015_jetti_project_file_tbl_separation_pv4_rom6hold': '015_jetti_project_file_amcf_pv4_6in_holdup',
    '015_jetti_project_file_tbl_separationtable': '015_jetti_project_file_demoheap',
    '017_jetti_project_file_tbl_table42': '017_jetti_project_file_ea_mill_feed_combined',
    '020_jetti_project_file_hardy_and_waste_tbl_nelson_table': '020_jetti_project_file_hardy_and_waste_h21_master_comp',
    '020_jetti_project_file_hypogene_supergene_tbl_nelson_hyp': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite',
    '020_jetti_project_file_hypogene_supergene_tbl_nelson_sup': '020_jetti_project_file_hypogene_supergene_super',
    '022_jetti_project_file_tbl_nelson_stingray': '022_jetti_project_file_stingray_1',
    #r'023_jetti_project_file': '023_jetti_project_file_ot_10',
    #r'023_jetti_project_file': '023_jetti_project_file_ot_9',
    '024_jetti_project_file_tbl_separationtable': '024_jetti_project_file_024cv_cpy',
    '026_jetti_project_file_tbl_separationtableps': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_tbl_separationtablecr': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_tbl_separationtabless': '026_jetti_project_file_sample_3_secondary_sulfide',
    '028_jetti_project_file_tbl_separationtable': '028_jetti_project_file_andesite', # chequear por compund 70/30
    #'028_jetti_project_file_tbl_separationtable': '028_jetti_project_file_monzonite', # chequear por compund 70/30
    'jetti_project_file_leopard_scl_tbl_nelson_los_bronces_crushed': 'jetti_project_file_leopard_scl_sample_los_bronces_crushed',
    'jetti_project_file_leopard_scl_tbl_nelson_los_bronces_rom': 'jetti_project_file_leopard_scl_sample_los_bronces_rom',
    'jetti_project_file_tiger_rom_tbl_nelson_tiger_m1rom': 'jetti_project_file_tiger_rom_m1_rom',
    'jetti_project_file_tiger_rom_tbl_nelson_tiger_m1': 'jetti_project_file_tiger_rom_m1_crushed',
    'jetti_project_file_tiger_rom_tbl_nelson_tiger_m2': 'jetti_project_file_tiger_rom_m2_crushed',
    'jetti_project_file_tiger_rom_tbl_nelson_tiger_m3': 'jetti_project_file_tiger_rom_m3_crushed',
    'jetti_project_file_elephant_scl_tbl_nelson_mel_scl': 'jetti_project_file_elephant_scl_sample_escondida_santiago',
    'jetti_project_file_toquepala_scl_tbl_nelson_toquepala': 'jetti_project_file_toquepala_scl_sample_fresca',
    #'jetti_project_file_toquepala_scl_tbl_nelson_toquepala': 'jetti_project_file_toquepala_scl_sample_antigua',    'jetti_project_file_zaldivar_scl_col68': 'jetti_project_file_zaldivar_scl_sample_zaldivar',
    'jetti_project_file_zaldivar_scl_tbl_nelson_table': 'jetti_project_file_zaldivar_scl_sample_zaldivar',
    'jetti_project_file_elephant_(site)_tbl_nelson_mel': 'jetti_project_file_elephant_site',
    '007_jetti_project_file_leopard_tbl_separationtable': '007_jetti_project_file_leopard_lep',
    '007b_jetti_project_file_tiger_tbl_separationtable': '007b_jetti_project_file_tiger_tgr'
}
set(df_separation_merged['project_col_id'].unique()) - set(replacement_dict.keys())


df_separation_merged['project_sample_condition_id'] = df_separation_merged['project_col_id'].copy()
for pattern, replacement in replacement_dict.items():
    escaped_pattern = re.escape(pattern)
    df_separation_merged['project_sample_condition_id'] = df_separation_merged['project_sample_condition_id'].str.replace(f"^{escaped_pattern}$", replacement, regex=True)


df_separation_merged.replace(0, np.nan, inplace=True)
df_separation_merged = df_separation_merged[pd.notnull(df_separation_merged['adjusted'])].copy()
df_separation_merged['adjusted_rounded'] = round(df_separation_merged['adjusted'], 0)

# Average control_col_1 and control_col_2, if only one is present, take that one
df_separation_merged['control_col_avg'] = df_separation_merged[['control_col_1', 'control_col_2']].mean(axis=1)
df_separation_merged['delta_cu'] =  df_separation_merged['catalyzed_col_max'] - df_separation_merged['control_col_avg']


df_separation_merged.to_csv(folder_path_save +'/df_separation.csv', sep=',')
df_separation_merged.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/power_bi/df_separation.csv', sep=',')


df_separation_merged[df_separation_merged['project_name'] == '007 Jetti Project File - Leopard']
df_separation_merged[df_separation_merged['project_name'] == '007B Jetti Project File - Tiger']


# %% COMMENTS


replacement_dict_comments = {
    'MEL SCL': 'jetti_project_file_elephant_scl_sample_escondida_santiago',
    'Leopard ROM': 'jetti_project_file_leopard_scl_sample_los_bronces_rom', 
    'Leopard Crushed': 'jetti_project_file_leopard_scl_sample_los_bronces_crushed', 
    'Zaldivar': 'jetti_project_file_zaldivar_scl_sample_zaldivar',
    'Toquepala': 'jetti_project_file_toquepala_scl_sample_fresca', 
    'Tiger S1 ROM': 'jetti_project_file_tiger_rom_m1_rom', 
    'Tiger S1 Crushed': 'jetti_project_file_tiger_rom_m1_crushed',
    'Tiger S2 Crushed': 'jetti_project_file_tiger_rom_m2_crushed', 
    'Tiger S3 Crushed': 'jetti_project_file_tiger_rom_m3_crushed', 
    '011 - Ray Mine': '011_jetti_project_file_rm',
    '014 - Bobcat-KMB': '014_jetti_project_file_kmb', 
    '014 - Bobcat-BAG': '014_jetti_project_file_bag', 
    '017 - Alpaca': '017_jetti_project_file_ea_mill_feed_combined',
    '020 - Wallaby-HYP': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite', 
    '020 - Wallaby-SUP': '020_jetti_project_file_hypogene_supergene_super', 
    '020 - Wallaby-HAR': '020_jetti_project_file_hardy_and_waste_h21_master_comp',
    '022 - Sierrita - Stingray': '022_jetti_project_file_stingray_1', 
    '023 - Cheetah - Chino OT-9': '023_jetti_project_file_ot_9',
    '023 - Cheetah - Chino OT-10': '023_jetti_project_file_ot_10', 
    '024 - CV': '024_jetti_project_file_024cv_cpy',
    '028 - Oroco - Monzonite': '028_jetti_project_file_monzonite', 
    '028 - Oroco - Andesite': '028_jetti_project_file_andesite',
    '028 - Oroco - Composite': '028_jetti_project_file_composite',
    'Copperhead 6"': '015_jetti_project_file_amcf_pv4_6in', #duplicar para holdup
    'Copperhead 8"': '015_jetti_project_file_amcf_pv4_8in', #duplicar para holdup
    'Demo heap': '015_jetti_project_file_demoheap', 
    'MEL Site': 'jetti_project_file_elephant_site',
    'Elephant II - UGM-2 ROM': 'jetti_file_elephant_ii_ugm2_rom',
    'Elephant II - UGM-2 Crushed': 'jetti_file_elephant_ii_ugm2_crushed',
    'Elephant II - PQ ROM': 'jetti_file_elephant_ii_pq_rom',
    'Elephant II - PQ Crushed': 'jetti_file_elephant_ii_pq_crushed',
    '026 - Caserones S#1': '026_jetti_project_file_sample_1_primary_sulfide',
    '026 - Caserones S#2': '026_jetti_project_file_sample_2_carrizalillo',
    '026 - Caserones S#3': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026 - Caserones S#4': '026_jetti_project_file_sample_4_mixed_material',
    '007 - Leopard': '007_jetti_project_file_leopard_lep',
    '007B - Tiger': '007b_jetti_project_file_tiger_tgr',
}

df_comments['project_sample_condition_id'] = df_comments['sheet_name']
for pattern, replacement in replacement_dict_comments.items():
    escaped_pattern = re.escape(pattern)
    df_comments['project_sample_condition_id'] = df_comments['project_sample_condition_id'].str.replace(f"^{escaped_pattern}$", replacement, regex=True)

# Special treatement for PV Copperhead
conditions_to_match = ['015_jetti_project_file_amcf_pv4_6in', '015_jetti_project_file_amcf_pv4_8in']

# Create an empty DataFrame to store the results
result_df = pd.DataFrame()

# Iterate over each condition
for condition in conditions_to_match:
    # Duplicate rows based on the condition
    duplicated_rows = df_comments[df_comments['project_sample_condition_id'] == condition].copy()
    
    # Modify the 'project_sample_condition_id' for duplicated rows
    duplicated_rows['project_sample_condition_id'] = duplicated_rows['project_sample_condition_id'] + '_holdup'
    
    # Concatenate the duplicated rows to the result DataFrame
    result_df = pd.concat([result_df, duplicated_rows], ignore_index=True)

# Concatenate the original DataFrame and the result DataFrame
df_comments = pd.concat([df_comments, result_df], ignore_index=True)

df_comments.replace(',', ';')
df_comments.to_csv(folder_path_save + '/df_comments.csv', sep=',')


# %% HEADERS MINERALOGY MODALS

df_mineralogy_modals_filtered = df_mineralogy_modals.copy()

df_mineralogy_modals_filtered['project_sample_id'] = normalize_dataframe_values(df_mineralogy_modals_filtered['project_name'].astype(str).str.replace('-', '') + 
                                                                        '_' + 
                                                                        df_mineralogy_modals_filtered['start_cell'].astype(str).str.replace('-', '') +
                                                                        '_' +
                                                                        df_mineralogy_modals_filtered['index'].astype(str).str.replace('-', '') +
                                                                        '_' +
                                                                        df_mineralogy_modals_filtered['sample'].astype(str).str.replace('-', '')
                                                                        )


df_mineralogy_modals_filtered['project_sample_id'].unique()
df_maker_index['project_sample_id'].unique()
# replace table names for actual project_sample_id in df_maker_index

replacement_dict_mineralogy_modals = {
    '1528901a_column_leach_20200219_muestra_1_head_muestra_1_head': '01a_jetti_project_file_c',
    '15289002_column_leach_v1_20191217_qb_qb': '002_jetti_project_file_qb',
    '15289004_column_leach_v1_20201130_bagdad_bagdad': '004_jetti_project_file_mols',
    '15289004_column_leach_v1_20201130_morenci_morenci': '004_jetti_project_file_mo',
    '15289006_column_leach_v1_20200828_pvls_pvls': '006_jetti_project_file_pvls',
    '15289006_column_leach_v1_20200828_pvo_pvo': '006_jetti_project_file_pvo',
    '011_jetti_project_file_tblmineralogy_modals_%_(%)': '011_jetti_project_file_rm',
    '024_jetti_project_file_tblmineralogy_modals_%_head_(%)': '024_jetti_project_file_024cv_cpy',
    '015_jetti_project_file_tblmineralogy_modals_amcf': '015_jetti_project_file_amcf',
    '015_jetti_project_file_tblmineralogy_modals_sgs_(%)': '015_jetti_project_file_amcf', #cambiaron nombre de tabla (revisado 22 enero 2025)
    '028_jetti_project_file_tblmineralogymodals_comp_(%)': '028_jetti_project_file_composite',
    '023_jetti_project_file_tblmineralogy_modals_ot9_ot09(%)': '023_jetti_project_file_ot_9',
    '023_jetti_project_file_tblmineralogy_modals_ot10_ot10(%)': '023_jetti_project_file_ot_10',
    '007_jetti_project_file_toquepala_tblmineralogy_modals_toquepala_frecsa_(%)': '',
    '007_jetti_project_file_toquepala_tblmineralogy_modals_toquepala_antigua_(%)': '',
    '014_jetti_project_file_tblbag_mineralogy_modals_bag(%)': '014_jetti_project_file_bag',
    '014_jetti_project_file_tblkmb_mineralogy_modals_mineral_mass_kmb_(%)': '014_jetti_project_file_kmb',
    '021_jetti_project_file_tblmineralogy_modals_(%)': '021_jetti_project_file_hypogene',
    '003_jetti_project_file_tblmineralogy_modals_amcf': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_tblmineralogy_modals_%_amcf_head': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_tblmineralogy_modals_%_control_(be1)_residue': '',
    '003_jetti_project_file_tblmineralogy_modals_%_catalyzed_(be2)_residue': '',
    '022_jetti_project_file_tblmineralcomposition_%': '022_jetti_project_file_stingray_1',
    '020_jetti_project_file_hypogene_supergene_tblmineralogy_modals_sup_sup_(%)': '020_jetti_project_file_hypogene_supergene_super',
    '020_jetti_project_file_hypogene_supergene_tblmineralogy_modals_sup2_sup_(%)': '020_jetti_project_file_hypogene_supergene_super',
    '020_jetti_project_file_hypogene_supergene_tblmineralogy_modals_hyp_hyp_(%)': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite',
    '020_jetti_project_file_hypogene_supergene_tblmineralogy_modals_hyp_%_hyp_(%)': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite',
    '020_jetti_project_file_hypogene_supergene_tblmineralogy_modals_hyp_%_hyp_head_(kg)': '',
    '020_jetti_project_file_hypogene_supergene_tblmineralogy_modals_hardy_har_(%)': '', # HAR tiene muestra en archivo serparado (next)
    '020_jetti_project_file_hardy_and_waste_tblmineralogy_modals_har_(%)': '020_jetti_project_file_hardy_and_waste_h21_master_comp',
    '013_jetti_project_file_tblmineralogy_modals_(%)': '013_jetti_project_file_combined',
    '026_jetti_project_file_tblmineralogy_modals_primary_sulfide_(%)': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_tblmineralogy_modals_carrizalillo_(%)': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_tblmineralogy_modals_secondary_sulfide_(%)': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_tblmineralogy_modals_mixed_(%)': '026_jetti_project_file_sample_4_mixed_material',
    '017_jetti_project_file_tblela_mineralogy_modals_%_head_(%)': '017_jetti_project_file_ea_mill_feed_combined',
    '012_jetti_project_file_cs_tblmineralogy_modals_%_incremento': '012_jetti_project_file_incremento',
    '012_jetti_project_file_cs_tblmineralogy_modals_%_kino': '012_jetti_project_file_kino',
    '012_jetti_project_file_cs_tblmineralogy_modals_%_quebalix_iv': '012_jetti_project_file_quebalix',
    'jetti_project_file_zaldivar_scl_tblmodals_zaldivar_sample_zaldi_var': 'jetti_project_file_zaldivar_scl_sample_zaldivar',
    'jetti_project_file_leopard_scl_tblmodals_los_bronces_sample_los_bronces': 'jetti_project_file_leopard_scl_sample_los_bronces',
    'jetti_project_file_elephant_scl_tblmodals_mel_sample_escondida': 'jetti_project_file_elephant_scl_sample_escondida',
    'jetti_project_file_tiger_rom_tblmineralogy_modals_m1_m1_head': 'jetti_project_file_tiger_rom_m1',
    'jetti_project_file_tiger_rom_tblmineralogy_modals_m2_m2_head': 'jetti_project_file_tiger_rom_m2',
    'jetti_project_file_tiger_rom_tblmineralogy_modals_m3_m3_head': 'jetti_project_file_tiger_rom_m3',
    'jetti_file_elephant_ii_ver_2_ugm_tblmineralogy_modals_(%)': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_pq_tblelephant_modals_%_crushed': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_tblelephant_modals_%_rom': '',
    'jetti_project_file_elephant_(site)_tblmineralogy_modals_m1_elephant_head': 'jetti_project_file_elephant_site',
    'jetti_project_file_toquepala_scl_tblmodals_toquepala_fresca_sample_fresca': 'jetti_project_file_toquepala_scl_sample_fresca',
    'jetti_project_file_toquepala_scl_tblmodals_toquepala_antigua_sample_antigua': 'jetti_project_file_toquepala_scl_sample_antigua',
    '007_jetti_project_file_leopard_tblmineralogy_modals_lep_(%)': '007_jetti_project_file_leopard_lep',
    '007b_jetti_project_file_tiger_tblmineralcomposition_%': '007b_jetti_project_file_tiger_tgr',
    '007_jetti_project_file_rtm2_tblmineralogy_modals_rt_m2_(%)': '007_jetti_project_file_rt_m2',
    '007_jetti_project_file_zaldivar_tblmineralogy_modals_zaldivar_(%)': '007_jetti_project_file_zaldivar',
    '007a_jetti_project_file_tblmineralogy_modals_pq_(%)': '007a_jetti_project_escondida_pq',
    '007a_jetti_project_file_tblmineralogy_modals_ugm2_(%)': '007a_jetti_project_escondida_ugm2',
    '020_jetti_project_file_hardy_and_waste_tblmineralogy_modals_%_har_(%)': '',
    '020_jetti_project_file_hardy_and_waste_tblmineralogy_modals_%_har_cat_(kg)': '',
    '020_jetti_project_file_hardy_and_waste_tblmineralogy_modals_%_har_catalyzed_(%)': '',
    '020_jetti_project_file_hardy_and_waste_tblmineralogy_modals_%_har_control_(%)': '',
    '020_jetti_project_file_hardy_and_waste_tblmineralogy_modals_%_har_control_(kg)': '',
    '020_jetti_project_file_hardy_and_waste_tblmineralogy_modals_%_har_head_(kg)': '',
    '020_jetti_project_file_hypogene_supergene_tblmineralogy_modals_hyp_%_hyp_cat_residue_(%)': '',
    '020_jetti_project_file_hypogene_supergene_tblmineralogy_modals_hyp_%_hyp_cat_residue_(kg)': '',
    '020_jetti_project_file_hypogene_supergene_tblmineralogy_modals_hyp_%_hyp_control_residue_(%)': '',
    '020_jetti_project_file_hypogene_supergene_tblmineralogy_modals_hyp_%_hyp_control_residue_(kg)': '',
    '025_jetti_project_file_tblmineralogy_modals_chalcopyrite_(%)': '025jettiprojectfile_chalcopyrite',
    '025_jetti_project_file_tblmineralogy_modals_oxide_(%)': '025jettiprojectfile_oxide',
    '025_jetti_project_file_tblmineralogy_modals_secondary_(%)': '025jettiprojectfile_secondary',
    '028_jetti_project_file_tblmineralogymodals_andesite_(%)': '028jettiprojectfile_andesite',
    '028_jetti_project_file_tblmineralogymodals_monzonite_(%)': '028jettiprojectfile_monzonite',
    '030_jetti_project_file_tblcpy_modals_(%)': '030_jetti_project_file_cpy',
    '030_jetti_project_file_tblss_modals_(%)': '030_jetti_project_file_ss',
    '031_jetti_project_file_tblcu_modals031_(%)': '031jettiprojectfile',
}
set(df_mineralogy_modals_filtered['project_sample_id'].unique()) - set(replacement_dict_mineralogy_modals.keys())

# Replace using the dictionary
for pattern, replacement in replacement_dict_mineralogy_modals.items():
    escaped_pattern = re.escape(pattern)
    df_mineralogy_modals_filtered['project_sample_id'] = df_mineralogy_modals_filtered['project_sample_id'].str.replace(f"^{escaped_pattern}$", replacement, regex=True)

df_mineralogy_modals_filtered['project_sample_id'] = df_mineralogy_modals_filtered['project_sample_id'].replace({
    '030jettiprojectfile_cpy': '030_jetti_project_file_cpy',
    '030jettiprojectfile_ss': '030_jetti_project_file_ss',
})
df_mineralogy_modals_filtered.loc[
    (df_mineralogy_modals_filtered['project_name'] == '020 Jetti Project File Hardy and Waste') &
    (df_mineralogy_modals_filtered['sample'] == 'HAR (%)'),
    'project_sample_id'
] = '020_jetti_project_file_hardy_and_waste_h21_master_comp'

# Duplicate the data of 011_jetti_project_file_rm and use new id: 011_jetti_project_file_rm_crushed
rm_crushed_label = '011_jetti_project_file_rm_crushed'
if not (df_mineralogy_modals_filtered['project_sample_id'] == rm_crushed_label).any():
    df_mineralogy_modals_rm = df_mineralogy_modals_filtered[df_mineralogy_modals_filtered['project_sample_id'] == '011_jetti_project_file_rm'].copy()
    if not df_mineralogy_modals_rm.empty:
        df_mineralogy_modals_rm.loc[:, 'project_sample_id'] = rm_crushed_label
        df_mineralogy_modals_filtered = pd.concat([df_mineralogy_modals_filtered, df_mineralogy_modals_rm], ignore_index=True)


df_mineralogy_modals_filtered.replace(np.nan, 0, inplace=True)

df_mineralogy_modals_filtered.to_csv(folder_path_save + '/df_mineralogy_modals.csv', sep=',')

df_mineralogy_modals_filtered[df_mineralogy_modals_filtered['project_sample_id'] == '020_jetti_project_file_hardy_and_waste_h21_master_comp']
df_mineralogy_modals_filtered[df_mineralogy_modals_filtered['project_sample_id'] == '020_jetti_project_file_hardy_and_waste_h21_master_comp']


# %%
