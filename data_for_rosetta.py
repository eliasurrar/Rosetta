# %% MERGE DATA TABLES FOR ROSETTA

from matplotlib import axis
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

from data_for_tableau import df_acsummary_filtered, df_chemchar_filtered, \
    df_column_summary_filtered, df_comments, df_maker_index, df_master, df_separation_merged,\
    df_mineralogy_modals_filtered
    
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

df_ac_summary = df_acsummary_filtered
df_chemchar = df_chemchar_filtered
df_column_summary = df_column_summary_filtered
df_comments = df_comments
df_leaching_performance = pd.read_csv(folder_path_load + '/3_dataframes_from_csv/leaching_columns.csv', sep=',', low_memory=False, index_col=0)
df_mineralogy_modals = df_mineralogy_modals_filtered
df_qemscan = pd.read_excel('/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/df_qemscan_compilation.xlsx')

df_maker_index = df_maker_index
df_master_tableau = df_master
#df_reactors = pd.read_csv(folder_path_save + '/df_reactors.csv', sep=',', low_memory=False, index_col=0)
# df_bottles = pd.read_csv(folder_path_save + '/3_dataframes_from_csv/dataset_rolling_bottles_detailed.csv', sep=',', low_memory=False, index_col=0)
df_separation = df_separation_merged

# MERGE OLD (TERMINATED) AND NEW LEACHING PERFORMANCE
# df_terminated_leaching = pd.read_excel(folder_path_save + '/df_leaching_performance_terminated_projects_new.xlsx')
df_terminated_leaching = pd.read_excel(folder_path_save + '/merged_terminated_projects_testDetails_MetallurgicalBalance.xlsx')

# df_terminated_leaching['cumulative_catalyst_addition_kg_t'].fillna(0, inplace=True)


df_leaching_performance = pd.concat([df_terminated_leaching, df_leaching_performance], axis=0, ignore_index=True)

cols_to_numeric = [
    'pls_cu_mg_l', 'pls_fe_mg_l', 'pls_mg_mg_l', 'pls_al_mg_l', 
    'pls_si_mg_l', 'pls_co_mg_l', 'pls_li_mg_l', 'pls_u_mg_l',
]

for b in cols_to_numeric:
    df_leaching_performance[b] = pd.to_numeric(df_leaching_performance[b], errors='coerce')

df_column_summary.loc[df_column_summary['index'].str.startswith('_'), 'col_name'] = df_column_summary.loc[df_column_summary['index'].str.startswith('_'), 'index'].str.lstrip('_').copy()
df_column_summary['col_name'] = df_column_summary['col_name'].fillna(df_column_summary['index'])

# 'project_cat_col_id' for 011 crushed is not being correctly handled. Replace values to make the correct match:
df_column_summary.loc[
    df_column_summary['project_col_id'] == '011_jetti_project_filecrushed_rm_5',
    'project_col_id'
] = '011_jetti_project_filecrushed_011rm_5'

df_column_summary.loc[
    df_column_summary['project_col_id'] == '011_jetti_project_filecrushed_rm_6',
    'project_col_id'
] = '011_jetti_project_filecrushed_011rm_6'

df_column_summary.loc[
    df_column_summary['project_col_id'] == '011_jetti_project_filecrushed_rm_7',
    'project_col_id'
] = '011_jetti_project_filecrushed_011rm_7'

df_column_summary.loc[
    df_column_summary['project_col_id'] == '011_jetti_project_filecrushed_rm_8',
    'project_col_id'
] = '011_jetti_project_filecrushed_011rm_8'



df_leaching_performance['sheet_name'] = df_leaching_performance['sheet_name'].str.replace('-', '_') # special treatment for Tiger ROM file which has - instead of _ on the sheet name

'''
for c in df_leaching_performance.columns:
    print(c)
'''


# Qemscan treatment
df_qemscan_filtered = df_qemscan[df_qemscan['sample'] == 'Combined']
df_qemscan_filtered['cus_total_sum'] = df_qemscan_filtered[['copper_sulphides_lib_exposed',
                                                                    'copper_sulphides_lib_50-80%_exposed',
                                                                    'copper_sulphides_lib_30-50%_exposed',
                                                                    'copper_sulphides_lib_20-30%_exposed',
                                                                    'copper_sulphides_lib_10-20%_exposed',
                                                                    'copper_sulphides_lib_0-10%_exposed',
                                                                    'copper_sulphides_lib_locked']].sum(axis=1, skipna=True)
df_qemscan_filtered['cus_exposed_50pct_sum'] = df_qemscan_filtered[['copper_sulphides_lib_exposed',
                                                                    'copper_sulphides_lib_50-80%_exposed']].sum(axis=1, skipna=True)
df_qemscan_filtered['cus_locked_30pct_sum'] = df_qemscan_filtered[['copper_sulphides_lib_20-30%_exposed',
                                                                  'copper_sulphides_lib_10-20%_exposed',
                                                                  'copper_sulphides_lib_0-10%_exposed',
                                                                  'copper_sulphides_lib_locked']].sum(axis=1, skipna=True)
df_qemscan_filtered['cus_exposed_50pct_normalized'] = df_qemscan_filtered['cus_exposed_50pct_sum'] / df_qemscan_filtered['cus_total_sum']
df_qemscan_filtered['cus_locked_30pct_normalized'] = df_qemscan_filtered['cus_locked_30pct_sum'] / df_qemscan_filtered['cus_total_sum']

df_qemscan_filtered.loc[df_qemscan_filtered['project_sample_id'] == '003_jetti_project_file_amcf_head']
df_qemscan_filtered.loc[df_qemscan_filtered['project_sample_id'] == '015_jetti_project_file_amcf']


#%%


df_leaching_performance[
    (df_leaching_performance['project_name'] == '020 Jetti Project File Hypogene_Supergene') & 
    (df_leaching_performance['sheet_name'] == 'HYP_1')
    ].head(10)


df_leaching_performance['cu_recovery_%'] = df_leaching_performance['cu_t_recovery_%'].fillna(df_leaching_performance['cu_recovery'])#.fillna(df_leaching_performance['cu_recovery_w_gre_inventory_%']).fillna(df_leaching_performance['cu_recovery_w_holdup_soln_%'])

# Special treatment for Copperhead project 015 for Cu Recoveries with HoldUp
df_leaching_015_gre_inv = df_leaching_performance[df_leaching_performance['cu_recovery_w_gre_inventory_%'].notnull()].copy()#[['origin', 'project_name', 'sheet_name', 'catalyzed', 'column_status', 'leach_duration_days', 'cu_recovery_w_gre_inventory_%']]
df_leaching_015_holdup_soln = df_leaching_performance[df_leaching_performance['cu_recovery_w_holdup_soln_%'].notnull()].copy()#[['origin', 'project_name', 'sheet_name', 'catalyzed', 'column_status', 'leach_duration_days', 'cu_recovery_w_holdup_soln_%']]

'''
df_leaching_015_gre_inv.rename(columns={
    'cu_recovery_w_gre_inventory_%': 'cu_recovery_%',
}, inplace=True)
df_leaching_015_holdup_soln.rename(columns={
    'cu_recovery_w_holdup_soln_%': 'cu_recovery_%',
}, inplace=True)
'''

df_leaching_015_gre_inv.loc[:, 'condition'] = 'w_gre_inventory'
df_leaching_015_holdup_soln.loc[:, 'condition'] = 'w_holdup_sln'
# df_leaching_015_gre_inv.reset_index(drop=True, inplace=True)
# df_leaching_015_holdup_soln.reset_index(drop=True, inplace=True)
#df_leaching_performance.reset_index(drop=True, inplace=True)

df_leaching_performance = pd.concat([df_leaching_performance, df_leaching_015_holdup_soln, df_leaching_015_gre_inv], join='outer', axis=0, ignore_index=True)

df_leaching_performance['project_col_id'] = normalize_dataframe_values(df_leaching_performance['project_name'].str.replace('-', '') + '_' + df_leaching_performance['sheet_name'].str.replace('-', '')).copy()
df_leaching_performance['col_name'] = normalize_dataframe_values(df_leaching_performance['sheet_name'].str.replace('-', '')).copy()

# treatment for TIGER and other chilean projects for ORP:
df_leaching_performance['feed_orp_mv_ag_agcl'] = df_leaching_performance['feed_orp_mv_ag_agcl'].fillna(df_leaching_performance['feed_orp_mv_enh'] - 223.0).copy() # Monse (mensaje Teams) 9 de abril 2024

dynamic_leaching_cols = [
    'feed_orp_mv_ag_agcl',
    'feed_flowrate_ml_min',
    'irrigation_rate_l_h_m2',
    'raff_assay_fe_ii_mg_l',
    'raff_assay_fe_iii_mg_l',
    'pls_fe_ii_mg_l',
    'pls_fe_iii_mg_l',
    'cumulative_lixiviant_m3_t',
]
for col in dynamic_leaching_cols:
    if col not in df_leaching_performance.columns:
        df_leaching_performance[col] = np.nan

if 'irrigation_rate_l_m2_h' in df_leaching_performance.columns:
    df_leaching_performance['irrigation_rate_l_h_m2'] = pd.to_numeric(
        df_leaching_performance['irrigation_rate_l_h_m2'], errors='coerce'
    ).fillna(
        pd.to_numeric(df_leaching_performance['irrigation_rate_l_m2_h'], errors='coerce')
    )

for col in dynamic_leaching_cols:
    df_leaching_performance[col] = pd.to_numeric(df_leaching_performance[col], errors='coerce')

df_leaching_performance['project_col_id'].unique()
replacement_dict = {
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
    '15289006_column_leach_v1_20200828_pvls1': '006_jetti_project_file_pvls',
    '15289006_column_leach_v1_20200828_pvls2': '006_jetti_project_file_pvls',
    '15289006_column_leach_v1_20200828_pvls3': '006_jetti_project_file_pvls',
    '15289006_column_leach_v1_20200828_pvls4': '006_jetti_project_file_pvls',
    '15289006_column_leach_v1_20200828_pvo1': '006_jetti_project_file_pvo',
    '15289006_column_leach_v1_20200828_pvo2': '006_jetti_project_file_pvo',
    '15289006_column_leach_v1_20200828_pvo3': '006_jetti_project_file_pvo',
    '15289006_column_leach_v1_20200828_pvo4': '006_jetti_project_file_pvo',
    '15289004_column_leach_v1_20201130_mo1': '004_jetti_project_file_mo',
    '15289004_column_leach_v1_20201130_mo2': '004_jetti_project_file_mo',
    '15289004_column_leach_v1_20201130_mo3': '004_jetti_project_file_mo',
    '15289004_column_leach_v1_20201130_mo4': '004_jetti_project_file_mo',
    '15289004_column_leach_v1_20201130_mols1': '004_jetti_project_file_mols',
    '15289004_column_leach_v1_20201130_mols2': '004_jetti_project_file_mols',
    '15289004_column_leach_v1_20201130_mols3': '004_jetti_project_file_mols',
    '15289004_column_leach_v1_20201130_mols4': '004_jetti_project_file_mols',
    '012_jetti_project_file_cs_i_1': '012_jetti_project_file_incremento',
    '012_jetti_project_file_cs_i_2': '012_jetti_project_file_incremento',
    '012_jetti_project_file_cs_i_3': '012_jetti_project_file_incremento',
    '012_jetti_project_file_cs_q_1': '012_jetti_project_file_quebalix',
    '012_jetti_project_file_cs_q_2': '012_jetti_project_file_quebalix',
    '012_jetti_project_file_cs_q_3': '012_jetti_project_file_quebalix',
    '011_jetti_project_file_rm_1': '011_jetti_project_file_rm',
    '011_jetti_project_file_rm_2': '011_jetti_project_file_rm',
    '011_jetti_project_filecrushed_011rm_5': '011_jetti_project_file_rm_crushed',
    '011_jetti_project_filecrushed_011rm_6': '011_jetti_project_file_rm_crushed',
    '011_jetti_project_filecrushed_011rm_7': '011_jetti_project_file_rm_crushed',
    '011_jetti_project_filecrushed_011rm_8': '011_jetti_project_file_rm_crushed',
    '007b_jetti_project_file_tiger_tgr_1': '007b_jetti_project_file_tiger_tgr',
    '007b_jetti_project_file_tiger_tgr_2': '007b_jetti_project_file_tiger_tgr',
    '007b_jetti_project_file_tiger_tgr_3': '007b_jetti_project_file_tiger_tgr',
    '007b_jetti_project_file_tiger_tgr_4': '007b_jetti_project_file_tiger_tgr',
    '024_jetti_project_file_cv_1': '024_jetti_project_file_024cv_cpy',
    '024_jetti_project_file_cv_2': '024_jetti_project_file_024cv_cpy',
    '024_jetti_project_file_cv_3': '024_jetti_project_file_024cv_cpy',
    '024_jetti_project_file_cv_4': '024_jetti_project_file_024cv_cpy',
    '015_jetti_project_file_c_1': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_6': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_11': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_2': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_7': '015_jetti_project_file_amcf',
    '015_jetti_project_file_c_12': '015_jetti_project_file_amcf',
    '028_jetti_project_file_comp_1': '028_jetti_project_file_composite',
    '028_jetti_project_file_comp_2': '028_jetti_project_file_composite',
    '028_jetti_project_file_comp_3': '028_jetti_project_file_composite',
    '028_jetti_project_file_comp_4': '028_jetti_project_file_composite',
    '023_jetti_project_file_ea_1': '',
    '023_jetti_project_file_ea_2': '',
    '023_jetti_project_file_ea_3': '',
    '023_jetti_project_file_ea_4': '',
    '014_jetti_project_file_k_1': '014_jetti_project_file_kmb',
    '014_jetti_project_file_k_2': '014_jetti_project_file_kmb',
    '014_jetti_project_file_k_3': '014_jetti_project_file_kmb',
    '014_jetti_project_file_k_4': '014_jetti_project_file_kmb',
    '014_jetti_project_file_b_1': '014_jetti_project_file_bag',
    '014_jetti_project_file_b_2': '014_jetti_project_file_bag',
    '014_jetti_project_file_b_3': '014_jetti_project_file_bag',
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
    '020_jetti_project_file_hardy_and_waste_har_1': '020_jetti_project_file_hardy_and_waste_h21_master_comp',
    '020_jetti_project_file_hardy_and_waste_har_2': '020_jetti_project_file_hardy_and_waste_h21_master_comp',
    '020_jetti_project_file_hardy_and_waste_har_3': '020_jetti_project_file_hardy_and_waste_h21_master_comp',
    '022_jetti_project_file_s_1': '022_jetti_project_file_stingray_1',
    '022_jetti_project_file_s_2': '022_jetti_project_file_stingray_1',
    '003_jetti_project_file_oxide_columns_beo_1': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_2': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_3': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_4': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_5': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_6': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_7': '003_jetti_project_file_amcf_head',
    '003_jetti_project_file_oxide_columns_beo_8': '003_jetti_project_file_amcf_head',
    '020_jetti_project_file_hypogene_supergene_sup_1': '020_jetti_project_file_hypogene_supergene_super',
    '020_jetti_project_file_hypogene_supergene_sup_2': '020_jetti_project_file_hypogene_supergene_super',
    '020_jetti_project_file_hypogene_supergene_sup_3': '020_jetti_project_file_hypogene_supergene_super',
    '020_jetti_project_file_hypogene_supergene_hyp_1': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite',
    '020_jetti_project_file_hypogene_supergene_hyp_2': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite',
    '020_jetti_project_file_hypogene_supergene_hyp_3': '020_jetti_project_file_hypogene_supergene_hypogene_master_composite',
    '013_jetti_project_file_o_1': '013_jetti_project_file_combined',
    '013_jetti_project_file_o_2': '013_jetti_project_file_combined',
    '013_jetti_project_file_o_3': '013_jetti_project_file_combined',
    '013_jetti_project_file_o_4': '013_jetti_project_file_combined',
    '030_jetti_project_file_cpy_1': '030_jetti_project_file_cpy',
    '030_jetti_project_file_cpy_2': '030_jetti_project_file_cpy',
    '030_jetti_project_file_ss_1': '030_jetti_project_file_ss',
    '030_jetti_project_file_ss_2': '030_jetti_project_file_ss',
    '026_jetti_project_file_ps_1': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_ps_2': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_ps_3': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_ps_4': '026_jetti_project_file_sample_1_primary_sulfide',
    '026_jetti_project_file_cr_1': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_cr_2': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_cr_3': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_cr_4': '026_jetti_project_file_sample_2_carrizalillo',
    '026_jetti_project_file_secs_1': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_secs_2': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_secs_3': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_secs_4': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_ss_1': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_ss_2': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_ss_3': '026_jetti_project_file_sample_3_secondary_sulfide',
    '026_jetti_project_file_ss_4': '026_jetti_project_file_sample_3_secondary_sulfide',
    '017_jetti_project_file_ea_1': '017_jetti_project_file_ea_mill_feed_combined',
    '017_jetti_project_file_ea_2': '017_jetti_project_file_ea_mill_feed_combined',
    '017_jetti_project_file_ea_3': '017_jetti_project_file_ea_mill_feed_combined',
    '017_jetti_project_file_ea_4': '017_jetti_project_file_ea_mill_feed_combined',
    '007_jetti_project_file_leopard_lep_1': '007_jetti_project_file_leopard_lep',
    '007_jetti_project_file_leopard_lep_2': '007_jetti_project_file_leopard_lep',
    '007_jetti_project_file_leopard_lep_3': '007_jetti_project_file_leopard_lep',
    '007_jetti_project_file_leopard_lep_4': '007_jetti_project_file_leopard_lep',
    'jetti_project_file_zaldivar_scl_col68': 'jetti_project_file_zaldivar_scl_sample_zaldivar',
    'jetti_project_file_zaldivar_scl_col69': 'jetti_project_file_zaldivar_scl_sample_zaldivar',
    'jetti_project_file_zaldivar_scl_col70': 'jetti_project_file_zaldivar_scl_sample_zaldivar',
    'jetti_project_file_leopard_scl_col1': 'jetti_project_file_leopard_scl_sample_los_bronces',
    'jetti_project_file_leopard_scl_col2': 'jetti_project_file_leopard_scl_sample_los_bronces',
    'jetti_project_file_leopard_scl_rom1': 'jetti_project_file_leopard_scl_sample_los_bronces',
    'jetti_project_file_leopard_scl_rom2': 'jetti_project_file_leopard_scl_sample_los_bronces',
    'jetti_project_file_elephant_scl_col42': 'jetti_project_file_elephant_scl_sample_escondida',
    'jetti_project_file_elephant_scl_col43': 'jetti_project_file_elephant_scl_sample_escondida',
    'jetti_project_file_elephant_scl_col52': 'jetti_project_file_elephant_scl_sample_escondida',
    'jetti_project_file_elephant_scl_col53': 'jetti_project_file_elephant_scl_sample_escondida',
    'jetti_project_file_tiger_rom_rom1': 'jetti_project_file_tiger_rom_m1',
    'jetti_project_file_tiger_rom_rom2': 'jetti_project_file_tiger_rom_m1',
    'jetti_project_file_tiger_rom_rom3': 'jetti_project_file_tiger_rom_m1',
    'jetti_project_file_tiger_rom_c_4': 'jetti_project_file_tiger_rom_m1',
    'jetti_project_file_tiger_rom_c_5': 'jetti_project_file_tiger_rom_m1',
    'jetti_project_file_tiger_rom_c_6': 'jetti_project_file_tiger_rom_m2',
    'jetti_project_file_tiger_rom_c_7': 'jetti_project_file_tiger_rom_m2',
    'jetti_project_file_tiger_rom_c_8': 'jetti_project_file_tiger_rom_m3',
    'jetti_project_file_tiger_rom_c_9': 'jetti_project_file_tiger_rom_m3',
    'jetti_file_elephant_ii_ver_2_ugm_ur_1': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_ur_2': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_1': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_2': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_3': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_4': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_5': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_6': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_ugm_uc_7': 'jetti_file_elephant_ii_ugm2',
    'jetti_file_elephant_ii_ver_2_pq_pr_1': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pr_2': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_1': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_2': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_3': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_4': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_5': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_6': 'jetti_file_elephant_ii_pq',
    'jetti_file_elephant_ii_ver_2_pq_pc_7': 'jetti_file_elephant_ii_pq',
    'jetti_project_file_elephant_(site)_fat4': 'jetti_project_file_elephant_site',
    'jetti_project_file_elephant_(site)_fat6': 'jetti_project_file_elephant_site',
    'jetti_project_file_elephant_(site)_s3': '',
    'jetti_project_file_toquepala_scl_col63': 'jetti_project_file_toquepala_scl_sample_fresca',
    'jetti_project_file_toquepala_scl_col64': 'jetti_project_file_toquepala_scl_sample_fresca',
    'jetti_project_file_toquepala_scl_col65': 'jetti_project_file_toquepala_scl_sample_fresca',
    'jetti_project_file_toquepala_scl_col66': 'jetti_project_file_toquepala_scl_sample_fresca',
    'jetti_project_file_toquepala_scl_col67': 'jetti_project_file_toquepala_scl_sample_fresca',
}

df_leaching_performance['project_sample_id'] = df_leaching_performance['project_col_id'].copy()
for pattern, replacement in replacement_dict.items():
    escaped_pattern = re.escape(pattern)
    df_leaching_performance['project_sample_id'] = df_leaching_performance['project_sample_id'].str.replace(f"^{escaped_pattern}$", replacement, regex=True)

df_leaching_performance['project_col_id'].unique()
# replace column names for old terminated projects
replacement_dict = {
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
for pattern, replacement in replacement_dict.items():
    escaped_pattern = re.escape(pattern)
    df_leaching_performance['project_col_id'] = df_leaching_performance['project_col_id'].str.replace(f"^{escaped_pattern}$", replacement, regex=True)

# re-order columns
current_order = df_leaching_performance.columns.tolist()

columns_to_move = ['col_name', 'project_col_id', 'project_sample_id']
for col in columns_to_move:
    current_order.remove(col)

new_order = current_order[:3] + columns_to_move + current_order[3:]
df_leaching_performance = df_leaching_performance[new_order]
df_leaching_performance[
    (df_leaching_performance['project_name'] == '020 Jetti Project File Hypogene_Supergene') & 
    (df_leaching_performance['sheet_name'] == 'HYP_1')
    ].head(10)


#%% match with chemchar and column summary
cols_to_match_chemchar = [
    'cu_%',
    'fe_%',
    'cu_seq_h2so4_%',
    'cu_seq_nacn_%',
    'cu_seq_a_r_%',
    'acid_soluble_%',
    'cyanide_soluble_%',
    'residual_cpy_%'
]

cols_to_match_columnsummary = [
    #'origin',
    #'project_name',
    #'index',
    #'sample_id',
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

col_to_match_mineralogy = [
    'pyrite',
    'chalcopyrite',
    'quartz',
    'k-feldspar',
    'biotite',
    'chlorite',
    'clays',
    'bornite',
    'covellite',
    'chalcocite',
    'enargite',
    'molybdenite',
    'plagioclase',
    'sericite_muscovite',
    'fe_oxides',
    'other_oxides',
    'sphalerite',
    'epidote', # merge epidote and group_epidote from mineralogy data table (not made at june 26th 2024)
    'rutile',
    'apatite',
]

cols_to_match_qemscan_exposure = [
    'copper_sulphides_lib_exposed',
    'copper_sulphides_lib_50-80%_exposed',
    'copper_sulphides_lib_30-50%_exposed',
    'copper_sulphides_lib_20-30%_exposed',
    'copper_sulphides_lib_10-20%_exposed',
    'copper_sulphides_lib_0-10%_exposed',
    'copper_sulphides_lib_locked',
    'cus_total_sum',
    'cus_exposed_50pct_sum',
    'cus_locked_30pct_sum',
    'cus_exposed_50pct_normalized',
    'cus_locked_30pct_normalized',
]


df_leaching_performance = df_leaching_performance.copy()

'''
for index, row in df_leaching_performance.iterrows():
    # Matching for cols_to_match_chemchar
    mask_chemchar = (df_chemchar['project_sample_id'] == row['project_sample_id'])
    matching_values_chemchar = df_chemchar[mask_chemchar][cols_to_match_chemchar].values
    for i in range(len(cols_to_match_chemchar)):
        if len(matching_values_chemchar) > 0:
            df_leaching_performance = df_leaching_performance.copy()
            df_leaching_performance.loc[index, cols_to_match_chemchar[i]] = matching_values_chemchar[0][i]
        else:
            df_leaching_performance = df_leaching_performance.copy()
            df_leaching_performance.loc[index, cols_to_match_chemchar[i]] = None

    # Matching for cols_to_match_columnsummary
    mask_columnsummary = (df_column_summary['project_col_id'] == row['project_col_id'])
    matching_values_columnsummary = df_column_summary[mask_columnsummary][cols_to_match_columnsummary].values
    for j in range(len(cols_to_match_columnsummary)):
        if len(matching_values_columnsummary) > 0:
            df_leaching_performance = df_leaching_performance.copy()
            df_leaching_performance.loc[index, cols_to_match_columnsummary[j]] = matching_values_columnsummary[0][j]
        else:
            df_leaching_performance = df_leaching_performance.copy()
            df_leaching_performance.loc[index, cols_to_match_columnsummary[j]] = None

    # Matching for col_to_match_mineralogy
    mask_mineralogy = (df_mineralogy_modals['project_sample_id'] == row['project_sample_id'])
    matching_values_mineralogy = df_mineralogy_modals[mask_mineralogy][col_to_match_mineralogy].values.copy()
    for k in range(len(col_to_match_mineralogy)):
        if len(matching_values_mineralogy) > 0:
            df_leaching_performance = df_leaching_performance.copy()
            df_leaching_performance.loc[index, col_to_match_mineralogy[k]] = matching_values_mineralogy[0][k]
        else:
            df_leaching_performance = df_leaching_performance.copy()
            df_leaching_performance.loc[index, col_to_match_mineralogy[k]] = None
'''

'''
# Define your matching function for a single row
def match_row(index_row_tuple):
    index, row = index_row_tuple

    # Initialize result dictionary
    result = {index: {}}

    # Matching for cols_to_match_chemchar
    mask_chemchar = (df_chemchar['project_sample_id'] == row['project_sample_id'])
    matching_values_chemchar = df_chemchar[mask_chemchar][cols_to_match_chemchar].values
    for i in range(len(cols_to_match_chemchar)):
        if len(matching_values_chemchar) > 0:
            result[index][cols_to_match_chemchar[i]] = matching_values_chemchar[0][i]
        else:
            result[index][cols_to_match_chemchar[i]] = None

    # Matching for cols_to_match_columnsummary
    mask_columnsummary = (df_column_summary['project_col_id'] == row['project_col_id'])
    matching_values_columnsummary = df_column_summary[mask_columnsummary][cols_to_match_columnsummary].values
    for j in range(len(cols_to_match_columnsummary)):
        if len(matching_values_columnsummary) > 0:
            result[index][cols_to_match_columnsummary[j]] = matching_values_columnsummary[0][j]
        else:
            result[index][cols_to_match_columnsummary[j]] = None

    # Matching for col_to_match_mineralogy
    mask_mineralogy = (df_mineralogy_modals['project_sample_id'] == row['project_sample_id'])
    matching_values_mineralogy = df_mineralogy_modals[mask_mineralogy][col_to_match_mineralogy].values.copy()
    for k in range(len(col_to_match_mineralogy)):
        if len(matching_values_mineralogy) > 0:
            result[index][col_to_match_mineralogy[k]] = matching_values_mineralogy[0][k]
        else:
            result[index][col_to_match_mineralogy[k]] = None

    return result

# Create a list of tuples (index, row) for each row in the DataFrame
rows = list(df_leaching_performance.iterrows())

# Use multiprocessing to parallelize
num_cores = cpu_count()
pool = Pool(num_cores)
results = pool.map(match_row, rows)
pool.close()
pool.join()

# Combine the results into the original DataFrame
for result in results:
    for index, values in result.items():
        for col, val in values.items():
            df_leaching_performance.at[index, col] = val
'''
df_chemchar[df_chemchar['project_name'] == '003 Jetti Project File']
df_mineralogy_modals[df_mineralogy_modals['project_name'] == '015 Jetti Project File']

# Create dictionary mappings
chemchar_dict = df_chemchar.drop_duplicates(subset='project_sample_id', keep='first').set_index('project_sample_id')[cols_to_match_chemchar].to_dict('index')
columnsummary_dict = df_column_summary.drop_duplicates(subset='project_col_id', keep='first').set_index('project_col_id')[cols_to_match_columnsummary].to_dict('index')
mineralogy_dict = df_mineralogy_modals.drop_duplicates(subset='project_sample_id', keep='first').set_index('project_sample_id')[col_to_match_mineralogy].to_dict('index')
qemscan_dict = df_qemscan_filtered.drop_duplicates(subset='project_sample_id', keep='first').set_index('project_sample_id')[cols_to_match_qemscan_exposure].to_dict('index') # drops what is not "Combined" or "Head"

list(columnsummary_dict.keys())

# Function to get values from the dictionary and handle missing values
def get_value_from_dict(key, dictionary, col):
    if key in dictionary and col in dictionary[key]:
        return dictionary[key][col]
    return None

# Apply the matching logic using vectorized operations
for col in cols_to_match_chemchar:
    df_leaching_performance[col] = df_leaching_performance['project_sample_id'].map(lambda x: get_value_from_dict(x, chemchar_dict, col))

for col in cols_to_match_columnsummary:
    df_leaching_performance[col] = df_leaching_performance['project_col_id'].map(lambda x: get_value_from_dict(x, columnsummary_dict, col))

for col in col_to_match_mineralogy:
    df_leaching_performance[col] = df_leaching_performance['project_sample_id'].map(lambda x: get_value_from_dict(x, mineralogy_dict, col))

for col in cols_to_match_qemscan_exposure:
    df_leaching_performance[col] = df_leaching_performance['project_sample_id'].map(lambda x: get_value_from_dict(x, qemscan_dict, col))

stats_by_col = df_leaching_performance.groupby('project_col_id')['catalyst_dosage_mg_l'].describe()

df_leaching_performance[df_leaching_performance['project_sample_id'] == '003_jetti_project_file_amcf_head'][['residual_cpy_%', 'acid_soluble_%', 'cus_exposed_50pct_normalized']]
df_leaching_performance[df_leaching_performance['project_sample_id'] == '015_jetti_project_file_amcf'][['residual_cpy_%', 'acid_soluble_%', 'cus_exposed_50pct_normalized']]

df_leaching_performance[df_leaching_performance['project_name'] == '003 Jetti Project File'][['residual_cpy_%', 'acid_soluble_%']]

df_leaching_performance[
    (df_leaching_performance['project_name'] == '020 Jetti Project File Hypogene_Supergene') & 
    (df_leaching_performance['sheet_name'] == 'HYP_1')
    ].head(10)


df_leaching_performance[df_leaching_performance['project_name'] == '011 Jetti Project File-Crushed'][['project_col_id', 'column_inner_diameter_m', 'residual_cpy_%', 'acid_soluble_%']]


#%% SPECIAL TREATMENTS

# IRON IN PROJECT 015 (the only one that has iron recirculated without SX) Ben on April 18th 2024 (vanc)

df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'raff_assay_cu_mg_l'] = df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'raff_assay_cu_mg_l'].fillna(df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'pls_cu_mg_l']).copy()
df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'raff_assay_fe_mg_l'] = df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'raff_assay_fe_mg_l'].fillna(df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'pls_fe_mg_l']).copy()
df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'raff_assay_mg_mg_l'] = df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'raff_assay_mg_mg_l'].fillna(df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'pls_mg_mg_l']).copy()
df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'raff_assay_al_mg_l'] = df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'raff_assay_al_mg_l'].fillna(df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'pls_al_mg_l']).copy()
# df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'raff_assay_zn_mg_l'] = df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'raff_assay_zn_mg_l'].fillna(df_leaching_performance.loc[df_leaching_performance['project_name'] == '015 Jetti Project File', 'pls_zn_mg_l'])


# Project 015 has 2 columns under same project_sample_id. We divide the samples in 2
# 11 and 6 are 8in, 11 is control and 6 is catalyzed
# 12 and 7 are 6in, 7 is control and 11 is catalyzed
df_leaching_performance.loc[(df_leaching_performance['project_col_id'] == '015_jetti_project_file_c_11') | 
                            (df_leaching_performance['project_col_id'] == '015_jetti_project_file_c_6'), 'project_sample_id'] = '015_jetti_project_file_amcf_8in'

df_leaching_performance.loc[(df_leaching_performance['project_col_id'] == '015_jetti_project_file_c_12') | 
                            (df_leaching_performance['project_col_id'] == '015_jetti_project_file_c_7'), 'project_sample_id'] = '015_jetti_project_file_amcf_6in'

df_leaching_performance[
    (df_leaching_performance['project_name'] == '020 Jetti Project File Hypogene_Supergene') & 
    (df_leaching_performance['sheet_name'] == 'HYP_1')
    ].head(20)

#%% EXPORT FINAL DATASET

df_leaching_performance.to_csv(folder_path_save + '/df_leaching_performance.csv', sep=',')
df_leaching_performance.to_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/power_bi/df_leaching_performance.csv', sep=',')


# %%
