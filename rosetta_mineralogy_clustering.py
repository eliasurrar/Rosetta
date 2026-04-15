#%%
from calendar import c
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


import matplotlib.pyplot as plt
import seaborn as sns
import sys
import datetime as dt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.append('/Users/administration/Library/CloudStorage/OneDrive-Personal/PycharmProjects/Jetti')
from functions_general import normalize_dataframe_values, normalize_and_replace
sys.path.append('/Users/administration/Library/CloudStorage/OneDrive-Personal/PycharmProjects/Jetti/Rosetta/lab')

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
folder_path_load = '/Users/administration/OneDrive - Jetti Resources/PycharmProjects/Jetti/SpkData/Jetti01'
folder_path_save = '/Users/administration/OneDrive - Jetti Resources/Rosetta/ML_Model'

min_thresh = 0.8 # 0.9

#%%
# Load data
# ==============================================================================
df_mineralogy = pd.read_csv('/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Reporting/db_python/df_mineralogy_modals.csv', index_col=0)
df_mineralogy.dropna(subset=['project_sample_id'], inplace=True)
df_mineralogy.reset_index(drop=True, inplace=True)

# ============================================================
# Fix project_sample_id for rows not mapped by ETL
# (project_sample_id = '0' means the ETL did not assign an ID)
# Identified unambiguously via start_cell and sample columns.
# Rows with no match in the leaching CSV are intentionally skipped:
#   - 026 mixed, Toquepala Antigua, 020 Hardy (already mapped separately)
# ============================================================
_start_cell_map = {
    ('007B Jetti Project File - Tiger',          'tbl-Mineralcomposition'):                      '007b_jetti_project_file_tiger_tgr',
    ('007 Jetti Project File - Leopard',         'tbl-Mineralogy_Modals_LEP'):                   '007_jetti_project_file_leopard_lep',
    ('015 Jetti Project File',                   'tbl-Mineralogy_Modals_SGS'):                   '015_jetti_project_file_amcf',  # duplicated below for 6in/8in
    ('014 Jetti Project File',                   'tbl-BAG_Mineralogy_Modals'):                   '014_jetti_project_file_bag',
    ('014 Jetti Project File',                   'tbl-KMB_Mineralogy_Modals'):                   '014_jetti_project_file_kmb',
    ('022 Jetti Project File',                   'tbl-Mineralcomposition'):                      '022_jetti_project_file_stingray_1',
    ('026 Jetti Project File',                   'tbl-Mineralogy_Modals_primary_sulfide'):       '026_jetti_project_file_sample_1_primary_sulfide',
    ('026 Jetti Project File',                   'tbl-Mineralogy_Modals_carrizalillo'):          '026_jetti_project_file_sample_2_carrizalillo',
    ('026 Jetti Project File',                   'tbl-Mineralogy_Modals_secondary_sulfide'):     '026_jetti_project_file_sample_3_secondary_sulfide',
    ('Jetti Project File - Zaldivar SCL',        'tbl-Modals_Zaldivar'):                        'jetti_project_file_zaldivar_scl_sample_zaldivar',
    ('Jetti Project File - Leopard SCL',         'tbl-Modals_Los_Bronces'):                     'jetti_project_file_leopard_scl_sample_los_bronces',
    ('Jetti Project File - Elephant SCL',        'tbl-Modals_MEL'):                             'jetti_project_file_elephant_scl_sample_escondida',
    ('Jetti File - Elephant II Ver 2 UGM',       'tbl-Mineralogy_Modals'):                      'jetti_file_elephant_ii_ugm2',
    ('Jetti Project File - Elephant (Site)',     'tbl-Mineralogy_Modals_M1'):                   'jetti_project_file_elephant_site',
    ('Jetti Project File - Toquepala SCL',       'tbl-Modals_Toquepala_Fresca'):                'jetti_project_file_toquepala_scl_sample_fresca',
    ('020 Jetti Project File Hypogene_Supergene','tbl-Mineralogy_Modals_Sup2'):                 '020_jetti_project_file_hypogene_supergene_super',
}
_mask_zero = df_mineralogy['project_sample_id'].astype(str) == '0'
for (proj, sc), new_id in _start_cell_map.items():
    _mask = _mask_zero & (df_mineralogy['project_name'] == proj) & (df_mineralogy['start_cell'].astype(str) == sc)
    df_mineralogy.loc[_mask, 'project_sample_id'] = new_id
# 006: identified via sample column
_mask_006_pvo  = _mask_zero & (df_mineralogy['project_name'] == '15289-006 - Column Leach_v1.2020-08-28') & (df_mineralogy['sample'].astype(str) == 'PVO')
_mask_006_pvls = _mask_zero & (df_mineralogy['project_name'] == '15289-006 - Column Leach_v1.2020-08-28') & (df_mineralogy['sample'].astype(str) == 'PVLS')
df_mineralogy.loc[_mask_006_pvo,  'project_sample_id'] = '006_jetti_project_file_pvo'
df_mineralogy.loc[_mask_006_pvls, 'project_sample_id'] = '006_jetti_project_file_pvls'
# Elephant II UGM2: same mineralogy applies to both ugm2 and ugm2_coarse
_ugm2_rows = df_mineralogy[df_mineralogy['project_sample_id'] == 'jetti_file_elephant_ii_ugm2'].copy()
_ugm2_rows['project_sample_id'] = 'jetti_file_elephant_ii_ugm2_coarse'
df_mineralogy = pd.concat([df_mineralogy, _ugm2_rows], ignore_index=True)

cols_to_drop = ['origin', 'project_name', 'sample', 'sheet_name', 'start_cell', 'index']
df_mineralogy.drop(columns=cols_to_drop, inplace=True)
list(df_mineralogy.columns)

# DUPLICATE PROJECT 015 AS WE SEPARATED 6 AND 8IN CONDITIONS
rows_to_duplicate = df_mineralogy[df_mineralogy['project_sample_id'] == '015_jetti_project_file_amcf'].copy()
rows_to_duplicate['project_sample_id'] = rows_to_duplicate['project_sample_id'] + '_8in'
df_mineralogy = pd.concat([df_mineralogy, rows_to_duplicate], ignore_index=True)
df_mineralogy.loc[df_mineralogy['project_sample_id'] == '015_jetti_project_file_amcf', 'project_sample_id'] = '015_jetti_project_file_amcf_6in'

# ==================
# Define mineral groupings

primary_copper_sulfides = [
    'chalcopyrite',
    'enargite',
    'tennantite',
    'tetrahedrite',
    'luzonite',
    'cubanite',
    'other_cu_sulfides'
]


secondary_copper_sulfides = [
    'chalcocite',
    'bornite', # 2% dissolution on sulfuric acid, 100% dissolution on cyanide
    'digenite',
    'chalcocite_digenite',
    'covellite',
    'yarrowite',
    'anilite',
    'geerite',
    'spionkopite'
    'native_copper', # only 5% dissolution on sulfuric acid, 100% dissolution on cyanide
]


copper_oxides = [
    'cuprite',
    'tenorite',

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
acid_generating_sulfides = ['pyrite', 'molybdenite']

# Group 4: Gangue Sulfides (base metal sulfides)
gangue_sulfides = ['sphalerite', 'galena', 'other_sulfides']

# Group 5: Gangue Silicates (feldspars and related)
gangue_silicates = ['quartz', 'plagioclase', 'feldspar_albite', 'albite', 'orthoclase', 
                    'cana-plagioclases', 'ca-plagioclase', 'na_ca_plagioclase', 'na_plagioclase',  'ca_na-plagioclase', 
                    'plagioclase_feldspar', 'plagioclases', 'k-feldspar', 'other_silicates', 'amphibole_pyroxene', 
                    'amphibole', 'pyroxenes', 'sericite_muscovite', 'muscovite', 'biotite', 'chlorite', 
                    'clays_other_silicates', 'clays__other_silicates', 'tourmaline', 
                    'clays', 'fe_clay', 'pyrophyllite', 'montmorillonite', 'kaolinite', # Group 6
                    'biotite_phlogopite', 'muscovite_sericite', 'chlorites_smectites', 
                    'chlorites', 'micas_illite', # for 025
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
             'magnetite-hematite', 'hematite-magnetite', 'fe_oxides_hydroxides', 'fe_oxides_cu', 'limonite-cu', 'limonite']

# Group 10: Accessory & Miscellaneous

# Group 11: Carbonates / acid consuming
carbonates = ['carbonates', 'calcite', 'dolomite', 'other_carbonates', 'siderite']

# Group 12: Accessory Minerals (& Miscellaneous, merged with 10)
accessory_minerals = ['zircon', 'barite', 'rutile', 'rutile_anatase', 'ilmenite', 'mg_so4', 'dioptase', 'corundum_gibbsite_boehmite',]

# Group 12.1 Phospate minerals (potentially acid consuming, but not currently weighted as such)
phosphate_minerals = ['apatite_monazite', 'fe_al_po4', 'monazite', 'apatite', 'svanbergite']

# Group 13: Others (ambiguous minerals)
# all_groups = copper_sulfides + secondary_copper + acid_generating_sulfides + gangue_sulfides + gangue_silicates + clays_and_micas + accessory_silicates + sulfates + fe_oxides + accessory_misc + carbonates + accessory_minerals
all_groups = copper_sulfides + secondary_copper + acid_generating_sulfides + gangue_sulfides + gangue_silicates + fe_oxides + carbonates + accessory_minerals

other_not_grouped = list(set(df_mineralogy.columns) - set(all_groups) - set(['project_sample_id']))

# Function to safely sum available columns
def sum_available_columns(df, columns):
    df = df.copy()
    df.replace(' ', 0, inplace=True)
    df.replace(np.nan, 0, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df[list(set(columns) & set(df.columns))].sum(axis=1)

df_mineralogy_grouped = df_mineralogy[['project_sample_id']].copy()

# Summing the grouped minerals
# df_mineralogy_grouped['grouped_copper_sulfides'] = sum_available_columns(df_mineralogy, copper_sulfides)
# df_mineralogy_grouped['grouped_secondary_copper'] = sum_available_columns(df_mineralogy, secondary_copper)
df_mineralogy_grouped['grouped_primary_copper_sulfides'] = sum_available_columns(df_mineralogy, primary_copper_sulfides)
df_mineralogy_grouped['grouped_secondary_copper_sulfides'] = sum_available_columns(df_mineralogy, secondary_copper_sulfides)
df_mineralogy_grouped['grouped_copper_oxides'] = sum_available_columns(df_mineralogy, copper_oxides)
df_mineralogy_grouped['grouped_mixed_copper_ores'] = sum_available_columns(df_mineralogy, mixed_copper_ores)

df_mineralogy_grouped['grouped_acid_generating_sulfides'] = sum_available_columns(df_mineralogy, acid_generating_sulfides)
df_mineralogy_grouped['grouped_gangue_sulfides'] = sum_available_columns(df_mineralogy, gangue_sulfides)
df_mineralogy_grouped['grouped_gangue_silicates'] = sum_available_columns(df_mineralogy, gangue_silicates)
# df_mineralogy_grouped['grouped_clays_and_micas'] = sum_available_columns(df_mineralogy, clays_and_micas)
# df_mineralogy_grouped['grouped_accesory_silicates'] = sum_available_columns(df_mineralogy, accessory_silicates)
# df_mineralogy_grouped['grouped_sulfates'] = sum_available_columns(df_mineralogy, sulfates)
df_mineralogy_grouped['grouped_fe_oxides'] = sum_available_columns(df_mineralogy, fe_oxides)
# df_mineralogy_grouped['grouped_accessory_misc'] = sum_available_columns(df_mineralogy, accessory_misc)
df_mineralogy_grouped['grouped_carbonates'] = sum_available_columns(df_mineralogy, carbonates)
df_mineralogy_grouped['grouped_accessory_minerals'] = sum_available_columns(df_mineralogy, accessory_minerals)
df_mineralogy_grouped['grouped_phosphate_minerals'] = sum_available_columns(df_mineralogy, phosphate_minerals)
df_mineralogy_grouped['grouped_other_not_grouped'] = sum_available_columns(df_mineralogy, other_not_grouped)



df_mineralogy_grouped.iloc[:, 1:].sum(axis=1)


list(df_mineralogy_grouped['project_sample_id'].unique())

#%%

# Standardize the data
features = df_mineralogy.columns.drop(['project_sample_id'])
x = df_mineralogy.loc[:, features].values
x = StandardScaler().fit_transform(x)

# Apply PCA
pca = PCA(n_components=6)  # Reduce to 2 components for visualization
principal_components = pca.fit_transform(x)
pca_df = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2', 'principal_component_3', 'principal_component_4', 'principal_component_5', 'principal_component_6'])
pca_df['project_sample_id'] = df_mineralogy['project_sample_id']

# Display results
print(pca_df.head())
print('MINERALOGY_CLUSTERS=============\nVariance explained per component (Eigen values) [PCA1  PCA2]:', np.round(pca.explained_variance_, 2))
print('Proportion of variance explained per component [PCA1  PCA2]:', np.round(pca.explained_variance_ratio_*100, 1), '%')
print('Proportion of total variance explained:', np.round(np.sum(pca.explained_variance_ratio_*100), 1), '%')



# Same as before but grouping
features = df_mineralogy_grouped.columns.drop(['project_sample_id'])
x = df_mineralogy_grouped.loc[:, features].values
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=6)
principal_components = pca.fit_transform(x)
pca_df = pd.DataFrame(data=principal_components, columns=['mineralogy_pca_1', 'mineralogy_pca_2', 'mineralogy_pca_3', 'mineralogy_pca_4', 'mineralogy_pca_5', 'mineralogy_pca_6'])
pca_df['project_sample_id'] = df_mineralogy_grouped['project_sample_id']
print(pca_df.head())
print('MINERALOGY_CLUSTERS=============\nVariance explained per component (Eigen values) [PCA1  PCA2]:', np.round(pca.explained_variance_, 2))
print('Proportion of variance explained per component [PCA1  PCA2]:', np.round(pca.explained_variance_ratio_*100, 1), '%')
print('Proportion of total variance explained:', np.round(np.sum(pca.explained_variance_ratio_*100), 1), '%')


pca_mineralogy_grouped = pca_df.copy()

#%%
"""
# Determine the optimal number of clusters using the elbow method on the original scaled data
df_mineralogy_kmeans = df_mineralogy.copy()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means clustering on the original scaled data
optimal_clusters = 4  # Choose the optimal number of clusters based on the elbow plot
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_mineralogy_kmeans['cluster'] = kmeans.fit_predict(x)

# Visualize the clusters using PCA components
pca_df['cluster'] = df_mineralogy_kmeans['cluster']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='principal_component_1', y='principal_component_2', hue='cluster', palette='viridis')
plt.title('Clusters of Mineralogy Samples')
plt.show()

# Display the clustered DataFrame
print(df_mineralogy_kmeans.head())

'''
# SAME WITH GROUPED
df_mineralogy_kmeans = df_mineralogy_grouped.copy()
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
optimal_clusters = 4  # Choose the optimal number of clusters based on the elbow plot
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_mineralogy_kmeans['cluster'] = kmeans.fit_predict(x)
pca_df['cluster'] = df_mineralogy_kmeans['cluster']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='principal_component_1', y='principal_component_2', hue='cluster', palette='viridis')
plt.title('Clusters of Mineralogy Samples')
plt.show()
print(df_mineralogy_kmeans.head())
'''
# %%
from sklearn.metrics import silhouette_score

df_mineralogy_silhouette = df_mineralogy.copy()

# Evaluate silhouette score for different numbers of clusters
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(x)
    silhouette_avg = silhouette_score(x, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score For Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Choose the number of clusters with the highest silhouette score
optimal_clusters = np.argmax(silhouette_scores) + 2
print(f'Optimal number of clusters based on silhouette score: {optimal_clusters}')

# Apply K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_mineralogy_silhouette['cluster'] = kmeans.fit_predict(x)




# SAME WITH GROUPED
df_mineralogy_silhouette = df_mineralogy_grouped.copy()
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(x)
    silhouette_avg = silhouette_score(x, cluster_labels)
    silhouette_scores.append(silhouette_avg)
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score For Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()
optimal_clusters = np.argmax(silhouette_scores) + 2
print(f'Optimal number of clusters based on silhouette score: {optimal_clusters}')
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_mineralogy_silhouette['cluster'] = kmeans.fit_predict(x)

#%%
from sklearn.cluster import DBSCAN

df_mineralogy_dbscan = df_mineralogy.copy()

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
df_mineralogy_dbscan['cluster'] = dbscan.fit_predict(x)

# Visualize the clusters using PCA components
pca_df['cluster'] = df_mineralogy_dbscan['cluster']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='principal_component_1', y='principal_component_2', hue='cluster', palette='viridis')
plt.title('DBSCAN Clusters of Mineralogy Samples')
plt.show()

# Display the clustered DataFrame
print(df_mineralogy_dbscan.head())



# SAME WITH GROUPED
'''
df_mineralogy_dbscan = df_mineralogy_grouped.copy()
dbscan = DBSCAN(eps=0.5, min_samples=5)
df_mineralogy_dbscan['cluster'] = dbscan.fit_predict(x)
pca_df['cluster'] = df_mineralogy_dbscan['cluster']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='principal_component_1', y='principal_component_2', hue='cluster', palette='viridis')
plt.title('DBSCAN Clusters of Mineralogy Samples')
plt.show()
print(df_mineralogy_dbscan.head())
'''
#%%
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

df_mineralogy_hierarchical = df_mineralogy.copy()

# Perform hierarchical clustering
Z = linkage(x, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=df_mineralogy['project_sample_id'].values, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample index')
plt.xticks(rotation=60, ha='right')
plt.ylabel('Distance')
plt.hlines(16.5, 0, 1000, colors='r', linestyles='dashed')
plt.show()

# Choose the number of clusters by cutting the dendrogram
max_d = 16.5  # Adjust this value based on the dendrogram
df_mineralogy_hierarchical['mineralogy_cluster'] = fcluster(Z, max_d, criterion='distance')

# Visualize the clusters using PCA components
pca_df['cluster'] = df_mineralogy_hierarchical['mineralogy_cluster']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='principal_component_1', y='principal_component_2', hue='cluster', palette='viridis')
plt.title('Hierarchical Clusters of Mineralogy Samples')
plt.show()

# Visualize the clusters using PCA components (zoom)
pca_df['cluster'] = df_mineralogy_hierarchical['mineralogy_cluster']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='principal_component_1', y='principal_component_2', hue='cluster', palette='viridis')
plt.title('Hierarchical Clusters of Mineralogy Samples')
plt.ylim(-4, 4)
plt.xlim(-2, 2)
plt.show()

# Display the clustered DataFrame
print(df_mineralogy_hierarchical.head())




# SAME WITH GROUPED
'''
df_mineralogy_hierarchical = df_mineralogy_grouped.copy()
Z = linkage(x, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=df_mineralogy_grouped['project_sample_id'].values, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample index')
plt.xticks(rotation=60, ha='right')
plt.ylabel('Distance')
plt.hlines(5, 0, 1000, colors='r', linestyles='dashed')
plt.show()
max_d = 5  # Adjust this value based on the dendrogram
df_mineralogy_hierarchical['mineralogy_cluster'] = fcluster(Z, max_d, criterion='distance')
pca_df['cluster'] = df_mineralogy_hierarchical['mineralogy_cluster']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='principal_component_1', y='principal_component_2', hue='cluster', palette='viridis')
plt.title('Hierarchical Clusters of Mineralogy Samples')
plt.show()
pca_df['cluster'] = df_mineralogy_hierarchical['mineralogy_cluster']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='principal_component_1', y='principal_component_2', hue='cluster', palette='viridis')
plt.title('Hierarchical Clusters of Mineralogy Samples')
plt.ylim(-4, 4)
plt.xlim(-2, 2)
plt.show()
print(df_mineralogy_hierarchical.head())
'''
#%%

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

df_mineralogy_gmm = df_mineralogy.copy()

# Evaluate GMM with different numbers of components
bic_scores = []
silhouette_scores = []
for n_components in range(2, 11):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(x)
    cluster_labels = gmm.predict(x)
    bic_scores.append(gmm.bic(x))
    silhouette_avg = silhouette_score(x, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), bic_scores, marker='o', label='BIC')
plt.plot(range(2, 11), silhouette_scores, marker='o', label='Silhouette Score')
plt.title('BIC and Silhouette Score for GMM')
plt.xlabel('Number of components')
plt.ylabel('Score')
plt.legend()
plt.show()

# Choose the number of components with the lowest BIC or highest silhouette score
optimal_components = np.argmin(bic_scores) + 2
print(f'Optimal number of components based on BIC: {optimal_components}')

# Apply GMM with the optimal number of components
gmm = GaussianMixture(n_components=optimal_components, random_state=42)
df_mineralogy_gmm['cluster'] = gmm.fit_predict(x)

# Visualize the clusters using PCA components
pca_df['cluster'] = df_mineralogy_gmm['cluster']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='principal_component_1', y='principal_component_2', hue='cluster', palette='viridis')
plt.title('GMM Clusters of Mineralogy Samples')
plt.show()

# Display the clustered DataFrame
print(df_mineralogy_gmm.head())

''


# SAME WITH GROUPED
'''
df_mineralogy_gmm = df_mineralogy_grouped.copy()
bic_scores = []
silhouette_scores = []
for n_components in range(2, 11):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(x)
    cluster_labels = gmm.predict(x)
    bic_scores.append(gmm.bic(x))
    silhouette_avg = silhouette_score(x, cluster_labels)
    silhouette_scores.append(silhouette_avg)
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), bic_scores, marker='o', label='BIC')
plt.plot(range(2, 11), silhouette_scores, marker='o', label='Silhouette Score')
plt.title('BIC and Silhouette Score for GMM')
plt.xlabel('Number of components')
plt.ylabel('Score')
plt.legend()
plt.show()
optimal_components = np.argmin(bic_scores) + 2
print(f'Optimal number of components based on BIC: {optimal_components}')
gmm = GaussianMixture(n_components=optimal_components, random_state=42)
df_mineralogy_gmm['cluster'] = gmm.fit_predict(x)
pca_df['cluster'] = df_mineralogy_gmm['cluster']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='principal_component_1', y='principal_component_2', hue='cluster', palette='viridis')
plt.title('GMM Clusters of Mineralogy Samples')
plt.show()
print(df_mineralogy_gmm.head())
'''
"""
# %%
