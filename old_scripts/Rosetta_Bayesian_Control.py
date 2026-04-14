#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import pymc as pm
import torch.multiprocessing as mp
import numpy as np

import nutpie

from sklearn.preprocessing import LabelEncoder

import arviz as az
import matplotlib.pyplot as plt


import jax
print(jax.default_backend())  # Should print 'metal' if successful
print(jax.devices())  # Should show [MetalDevice(id=0)]
jax.config.update('jax_enable_x64', True)


# In[6]:


reactor_df = pd.read_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/reactors/dataset_per_sample_reactor.csv')

reactor_df=reactor_df[reactor_df['catalyst_type']=='Control']


df_reac = reactor_df

df_reac['proj_col_id'] = df_reac['project_sample_id']+df_reac['start_cell']

time_values = list(range(1, 126))

expanded_rows = []
for _, row in df_reac.iterrows():
    for time_val in time_values:
        new_row = row.copy()
        new_row['time'] = time_val
        expanded_rows.append(new_row)

df_expanded = pd.DataFrame(expanded_rows)

time_df = pd.DataFrame({'time': time_values})

# Cross join (Cartesian product) to create all combinations
df_expanded = df_reac.assign(key=1).merge(time_df.assign(key=1), on='key').drop('key', axis=1)

df_control =df_expanded

df_control['cu_rec_%'] = (df_control['a1_param']) * (1 - np.exp(-df_control['b1_param'] * df_control['time'])) + (df_control['a2_param']) * (1 - np.exp(-df_control['b2_param'] * df_control['time']))

df_control['t']=df_control['time']

df = df_control.set_index(['proj_col_id','time'])
df2=df[df['ph_target']=='2.0']
df2=df2[df2['lixiviant']=='Inoculum']
df2=pd.DataFrame(df2)

df2['acid_soluble_%']=df2['cu_seq_h2so4_%']*100
df2['cyanide_soluble_%']=df2['cu_seq_nacn_%']*100
df2['residual_cpy_%']=df2['cu_seq_a_r_%']*100


df2['oxides']=df2['acid_soluble_%']
df2['secondaries']=df2['cyanide_soluble_%']
df2['cpy']=df2['residual_cpy_%']

catalyzed_df_new = pd.read_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/columns/df_model_control_projects_with_reactors_fit.csv')

cats = ['003_jetti_project_file_be_1','024_jetti_project_file_cv_1','014_jetti_project_file_b_2','jetti_project_file_zaldivar_scl_col70',
          '014_jetti_project_file_k_1','jetti_project_file_toquepala_scl_col64','jetti_project_file_elephant_scl_col43','006_jetti_project_file_pvo3','006_jetti_project_file_pvo4','013_jetti_project_file_o_2','013_jetti_project_file_o_3',
        '020_jetti_project_file_hypogene_supergene_hyp_1','020_jetti_project_file_hypogene_supergene_hyp_3','020_jetti_project_file_hypogene_supergene_sup_1','020_jetti_project_file_hypogene_supergene_sup_3','017_jetti_project_file_ea_1','jetti_project_file_elephant_(site)_fat6',
        '015_jetti_project_file_c_11','015_jetti_project_file_c_7','011_jetti_project_file_rm_2','022_jetti_project_file_s_1','026_jetti_project_file_ps_1', '026_jetti_project_file_ps_2',
       '026_jetti_project_file_cr_2', '026_jetti_project_file_ss_2',
       '006_jetti_project_file_pvls3',
       'jetti_file_elephant_ii_ver_2_pq_pr_1',
       'jetti_file_elephant_ii_ver_2_ugm_uc_1',
       'jetti_file_elephant_ii_ver_2_ugm_uc_3',
       'jetti_file_elephant_ii_ver_2_ugm_ur_1']

new_cols = catalyzed_df_new[catalyzed_df_new['project_col_id'].isin(cats)]
new_cols['oxides'] =  new_cols['acid_soluble_%'] * new_cols['feed_head_cu_%']
new_cols['secondaries'] =  new_cols['cyanide_soluble_%'] * new_cols['feed_head_cu_%']
new_cols['cpy'] =  new_cols['residual_cpy_%'] * new_cols['feed_head_cu_%']
cat_cols = new_cols.copy()
cat_cols=cat_cols[['project_col_id','leach_duration_days','acid_soluble_%','cyanide_soluble_%', 'residual_cpy_%','feed_head_cu_%','cu_recovery_%']]
cat_cols=cat_cols.dropna()


# In[28]:


group_1_cids = ['006_jetti_project_file_pvls3','026_jetti_project_file_cr_2','026_jetti_project_file_ps_2','026_jetti_project_file_ps_1','003_jetti_project_file_be_1','024_jetti_project_file_cv_1','014_jetti_project_file_b_2','jetti_project_file_zaldivar_scl_col70',
          '014_jetti_project_file_k_1','jetti_project_file_toquepala_scl_col64','jetti_project_file_elephant_scl_col43','006_jetti_project_file_pvo3','006_jetti_project_file_pvo4','013_jetti_project_file_o_2','013_jetti_project_file_o_3']
group_2_cids = ['020_jetti_project_file_hypogene_supergene_hyp_1','020_jetti_project_file_hypogene_supergene_hyp_3','020_jetti_project_file_hypogene_supergene_sup_1','020_jetti_project_file_hypogene_supergene_sup_3']
group_3_cids = ['017_jetti_project_file_ea_1','jetti_project_file_elephant_(site)_fat6','026_jetti_project_file_ss_2','jetti_file_elephant_ii_ver_2_pq_pr_1','jetti_file_elephant_ii_ver_2_ugm_uc_1','jetti_file_elephant_ii_ver_2_ugm_uc_3','jetti_file_elephant_ii_ver_2_ugm_ur_1']
group_4_cids = ['015_jetti_project_file_c_11','015_jetti_project_file_c_7','011_jetti_project_file_rm_2','022_jetti_project_file_s_1']


# In[29]:


#cat

def assign_group_from_lists(project_name):
    if project_name in group_1_cids:
        return 0
    elif project_name in group_2_cids:
        return 1
    elif project_name in group_3_cids:
        return 2
    elif project_name in group_4_cids:
        return 3
    else:
        return -1  # Flag for unmapped projects

# Apply to your dataframe
cat_cols['group_id'] = cat_cols['project_col_id'].apply(assign_group_from_lists)  # Replace 'project_name' with your actual column name

# Check for any unmapped projects
unmapped = cat_cols[cat_cols['group_id'] == -1]
if len(unmapped) > 0:
    print("Warning: These projects couldn't be mapped to any group:")
    print(unmapped['project_col_id'].unique())
else:
    print("All projects successfully mapped to groups!")

# Verify the grouping
print("\nGroup distribution:")
print(cat_cols['group_id'].value_counts().sort_index())

print("\nSample mapping verification:")
for group_id in range(4):
    sample_projects = cat_cols[cat_cols['group_id'] == group_id]['project_col_id'].unique()[:3]
    print(f"Group {group_id} sample projects: {list(sample_projects)}")


# In[30]:

df_cols = cat_cols

new_df = df_cols.copy()

new_df['time']=new_df['leach_duration_days']
new_df['oxides'] =  new_df['acid_soluble_%'] * new_df['feed_head_cu_%']

new_df['secondaries'] =  new_df['cyanide_soluble_%'] * new_df['feed_head_cu_%']

new_df['cpy'] =  new_df['residual_cpy_%'] * new_df['feed_head_cu_%']

new_df['t'] =  new_df['leach_duration_days']

new_df['t_log'] =  np.log(new_df['leach_duration_days']+1)


control_r = df2.reset_index()
control_r['leach_duration_days'] = control_r['time']
control_r['project_col_id'] = control_r['proj_col_id']
control_r['t_log'] = np.log(control_r['time']+1) 
control_r['oxides_2']=control_r['oxides']**2
control_r['secondaries_2']=control_r['secondaries']**2
control_r['cpy_2']=control_r['cpy']**2
control_r = control_r[['cu_rec_%','proj_col_id','project_col_id','oxides','secondaries','cpy','time','t_log','leach_duration_days']]
control_r = control_r.dropna()
control_r = control_r[control_r['oxides']!=0]
control_r = control_r[control_r['cu_rec_%']>0]
print(f"  DataFrame shape: {control_r.shape}")


experiment_encoder = LabelEncoder()
control_r['experiment_idx'] = experiment_encoder.fit_transform(control_r['project_col_id'])  # Replace with actual column name
print(f"\nExperiment encoding:")
print(f"  Number of experiments: {control_r['experiment_idx'].nunique()}")
print(f"  Experiment range: {control_r['experiment_idx'].min()} to {control_r['experiment_idx'].max()}")


control_r['time_unscaled'] = control_r['time']
control_r['time_log_unscaled'] = control_r['t_log']  

print(f"  Original time range: {control_r['time'].min():.3f} to {control_r['time'].max():.3f}")
print(f"  Original time_log range: {control_r['t_log'].min():.3f} to {control_r['t_log'].max():.3f}")


mineral_columns = ['oxides', 'secondaries', 'cpy']  # Replace with actual names
X_minerals = control_r[mineral_columns].values


print(f"  Mineral columns: {mineral_columns}")
print(f"  Minerals shape: {X_minerals.shape}")
for i, col in enumerate(mineral_columns):
    print(f"    {col}: {X_minerals[:, i].min():.3f} to {X_minerals[:, i].max():.3f}")


y_recovery = control_r['cu_rec_%'].values  # Keep original percentage scale


new_data_dict_con_r = {
    'experiment_idx': control_r['experiment_idx'].values,
    'time': control_r['time_unscaled'].values,
    't_log': control_r['time_log_unscaled'].values,# Original time units  # Original log time units
    'X_minerals': X_minerals,                        # Original mineral units
    'y_recovery': y_recovery,                        # 0-100 percentage scale
    'n_experiments': control_r['experiment_idx'].nunique(),
    'n_minerals': len(mineral_columns),
    'n_obs': len(control_r)
}

new_data_numpy_con_r = {}
for key, value in new_data_dict_con_r.items():
    if isinstance(value, (list, tuple)) or hasattr(value, '__array__'):
        new_data_numpy_con_r[key] = np.asarray(value)
    else:
        new_data_numpy_con_r[key] = value


print("FINAL DATA SUMMARY")

print(f"\nData shapes:")
for key, value in new_data_numpy_con_r.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: {value.shape}")
    else:
        print(f"  {key}: {value}")

print(f"\nData ranges (all in original units):")
print(f"  Time: {new_data_numpy_con_r['time'].min():.3f} to {new_data_numpy_con_r['time'].max():.3f}")
#print(f"  Time Log: {new_data_numpy_con_r['time_sat'].min():.3f} to {new_data_numpy_con_r['time_sat'].max():.3f}")
print(f"  Recovery: {new_data_numpy_con_r['y_recovery'].min():.1f}% to {new_data_numpy_con_r['y_recovery'].max():.1f}%")
print(f"  Experiment IDs: {new_data_numpy_con_r['experiment_idx'].min()} to {new_data_numpy_con_r['experiment_idx'].max()}")
#print(f"  Group IDs: {np.unique(new_data_numpy_con_r['group_id'])}")

print(f"\nMinerals ranges:")
for i, col in enumerate(mineral_columns):
    mineral_data = new_data_numpy_con_r['X_minerals'][:, i]
    print(f"  {col}: {mineral_data.min():.3f} to {mineral_data.max():.3f}")

#%%
print("Input data overview:")
print(f"  DataFrame shape: {new_df.shape}")


experiment_encoder = LabelEncoder()
new_df['experiment_idx'] = experiment_encoder.fit_transform(new_df['project_col_id'])  # Replace with actual column name
print(f"\nExperiment encoding:")
print(f"  Number of experiments: {new_df['experiment_idx'].nunique()}")
print(f"  Experiment range: {new_df['experiment_idx'].min()} to {new_df['experiment_idx'].max()}")



new_df['time_unscaled'] = new_df['time']  
new_df['time_log_unscaled'] = new_df['t_log']  

print(f"  Original time range: {new_df['time'].min():.3f} to {new_df['time'].max():.3f}")
print(f"  Original time_log range: {new_df['t_log'].min():.3f} to {new_df['t_log'].max():.3f}")


print(f"\nMineral data (keeping original units):")

# Adjust these column names to match your actual mineral columns
mineral_columns = ['oxides', 'secondaries', 'cpy']  # Replace with actual names
X_minerals = new_df[mineral_columns].values

# Keep minerals in original units - NO SCALING
print(f"  Mineral columns: {mineral_columns}")
print(f"  Minerals shape: {X_minerals.shape}")
for i, col in enumerate(mineral_columns):
    print(f"    {col}: {X_minerals[:, i].min():.3f} to {X_minerals[:, i].max():.3f}")



# Keep recovery as percentages (0-100 scale) - DON'T divide by 100
y_recovery = new_df['cu_recovery_%'].values  # Keep original percentage scale


new_data_dict_con = {
    'experiment_idx': new_df['experiment_idx'].values,
    'time': new_df['time_unscaled'].values,          # Original time units
    'time_log': new_df['time_log_unscaled'].values,  # Original log time units
    'X_minerals': X_minerals,                        # Original mineral units
    'y_recovery': y_recovery,                        # 0-100 percentage scale
    'group_id': new_df['group_id'].values,
    'n_experiments': new_df['experiment_idx'].nunique(),
    'n_minerals': len(mineral_columns),
    'n_groups': 4,
    'n_obs': len(new_df)
}

# Convert to numpy for PyMC
new_data_numpy_con = {}
for key, value in new_data_dict_con.items():
    if isinstance(value, (list, tuple)) or hasattr(value, '__array__'):
        new_data_numpy_con[key] = np.asarray(value)
    else:
        new_data_numpy_con[key] = value



print(f"\nData shapes:")
for key, value in new_data_numpy_con.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: {value.shape}")
    else:
        print(f"  {key}: {value}")

print(f"\nData ranges (all in original units):")
print(f"  Time: {new_data_numpy_con['time'].min():.3f} to {new_data_numpy_con['time'].max():.3f}")
print(f"  Time Log: {new_data_numpy_con['time_log'].min():.3f} to {new_data_numpy_con['time_log'].max():.3f}")
print(f"  Recovery: {new_data_numpy_con['y_recovery'].min():.1f}% to {new_data_numpy_con['y_recovery'].max():.1f}%")
print(f"  Experiment IDs: {new_data_numpy_con['experiment_idx'].min()} to {new_data_numpy_con['experiment_idx'].max()}")
print(f"  Group IDs: {np.unique(new_data_numpy_con['group_id'])}")

print(f"\nMinerals ranges:")
for i, col in enumerate(mineral_columns):
    mineral_data = new_data_numpy_con['X_minerals'][:, i]
    print(f"  {col}: {mineral_data.min():.3f} to {mineral_data.max():.3f}")


#%%
## Physics-based systematic error approach with exponential time function ##

# =================================================================
# LAYER 1: BASE MODEL WITH EXPONENTIAL TIME FUNCTION
# =================================================================


# Prepare first dataset
X_minerals_1 = new_data_numpy_con_r['X_minerals']  # Shape: (n_obs, 3)
time_1 = new_data_numpy_con_r['time']  # Shape: (n_obs,)
y_recovery_1 = new_data_numpy_con_r['y_recovery']  # Shape: (n_obs,) in % scale
n_obs_1 = len(y_recovery_1)

print(f"    First dataset: {n_obs_1} observations")
print(f"    Recovery range: {y_recovery_1.min():.1f}% to {y_recovery_1.max():.1f}%")
print(f"    Time range: {time_1.min():.1f} to {time_1.max():.1f}")

# =================================================================
# PHYSICS-BASED TIME FEATURES
# =================================================================


# Layer 1 time scales 
time_scales_layer1 = [20, 40, 80, 100]  # Appropriate for 0-150 day range
exp_features_1 = np.column_stack([1 - np.exp(-time_1 / ts) for ts in time_scales_layer1])



with pm.Model() as layer1_model:
    

    
    # Mineral effects (these will be transferred)
    mineral_1_coeff = pm.Normal('mineral_1_coeff', mu=0, sigma=5.0)
    mineral_2_coeff = pm.Normal('mineral_2_coeff', mu=0, sigma=5.0)
    mineral_3_coeff = pm.Normal('mineral_3_coeff', mu=0, sigma=5.0)
    
    # Physics-based time effects (exponential saturation terms)
    exp_time_coeffs = pm.Normal('exp_time_coeffs', mu=0, sigma=5.0, shape=len(time_scales_layer1))
    
    # Base recovery level (ensures positive recovery)
    base_recovery = pm.Normal('base_recovery', mu=20, sigma=10.0)
    
    # =================================================================
    # PREDICTION WITH PHYSICAL CONSTRAINTS
    # =================================================================
    
    # Mineral contribution
    mineral_effect = (mineral_1_coeff * X_minerals_1[:, 0] +
                     mineral_2_coeff * X_minerals_1[:, 1] +
                     mineral_3_coeff * X_minerals_1[:, 2])
    
    # Exponential time contribution (sum of multiple time scales)
    exp_time_effect = pm.math.sum([exp_time_coeffs[i] * exp_features_1[:, i] 
                                  for i in range(len(time_scales_layer1))], axis=0)
    
    # Total prediction (constrained to be positive)
    mu_unconstrained = base_recovery + mineral_effect + exp_time_effect
    
    # Apply softplus to ensure positive recovery: softplus(x) = log(1 + exp(x))
    mu = pm.math.log(1 + pm.math.exp(mu_unconstrained))
    
    # Observation model
    sigma = pm.HalfNormal('sigma', sigma=5.0)
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y_recovery_1)


# Sample Layer 1 model
print("\n Sampling Layer 1 model...")
compiled_layer1 = nutpie.compile_pymc_model(layer1_model)
trace_layer1 = nutpie.sample(compiled_layer1, 
                             draws=1000, 
                             tune=500, 
                             chains=6, 
                             seed=42, 
                             progress_bar=True, 
                             target_accept=0.99, 
                             cores=mp.cpu_count()
                             )

# Extract learned parameters
layer1_params = {
    'base_recovery': float(trace_layer1.posterior['base_recovery'].mean()),
    'mineral_1_coeff': float(trace_layer1.posterior['mineral_1_coeff'].mean()),
    'mineral_2_coeff': float(trace_layer1.posterior['mineral_2_coeff'].mean()),
    'mineral_3_coeff': float(trace_layer1.posterior['mineral_3_coeff'].mean()),
    'exp_time_coeffs': trace_layer1.posterior['exp_time_coeffs'].mean(dim=['chain', 'draw']).values,
    'sigma': float(trace_layer1.posterior['sigma'].mean())
}

print("Layer 1 complete! Learned parameters:")
print(f"   base_recovery: {layer1_params['base_recovery']:.4f}")
print(f"   mineral_1_coeff: {layer1_params['mineral_1_coeff']:.4f}")
print(f"   mineral_2_coeff: {layer1_params['mineral_2_coeff']:.4f}")
print(f"   mineral_3_coeff: {layer1_params['mineral_3_coeff']:.4f}")
for i, coeff in enumerate(layer1_params['exp_time_coeffs']):
    print(f"   exp_time_coeff_{time_scales_layer1[i]:2d}d: {coeff:8.4f}")
print(f"   sigma: {layer1_params['sigma']:.4f}")

# =================================================================
# LAYER 2: TRANSFER + EARLY RECOVERY ADJUSTMENTS
# =================================================================

print(f"\nLAYER 2: TRANSFER + EARLY RECOVERY ADJUSTMENTS (PHYSICS-BASED)")

# Prepare second dataset
X_minerals_2 = new_data_numpy_con['X_minerals']
time_2 = new_data_numpy_con['time']
y_recovery_2 = new_data_numpy_con['y_recovery']
experiment_idx_2 = new_data_numpy_con['experiment_idx']
group_id_2 = new_data_numpy_con['group_id']
n_obs_2 = len(y_recovery_2)
n_experiments_2 = new_data_numpy_con['n_experiments']
n_groups_2 = len(np.unique(group_id_2))

print(f"    Second dataset: {n_obs_2} observations")
print(f"    {n_experiments_2} experiments, {n_groups_2} groups")

# Layer 2 time scales (0-1500 days, avg <1000): faster time scales for 95% recovery by 500d
time_scales_layer2 = [200, 400, 600, 800]  # Faster scales for 95% recovery by day 500

# Create physics-based time features for Layer 2
exp_features_2 = np.column_stack([1 - np.exp(-time_2 / ts) for ts in time_scales_layer2])





# =================================================================
# EARLY RECOVERY ANALYSIS (FIRST 200 DAYS)
# =================================================================


# Create mask for early recovery period
early_mask = time_2 <= 250
n_early_obs = np.sum(early_mask)
print(f"    Early period observations: {n_early_obs} out of {n_obs_2}")

# Extract early recovery data
time_early = time_2[early_mask]
y_recovery_early = y_recovery_2[early_mask]
X_minerals_early = X_minerals_2[early_mask]
experiment_idx_early = experiment_idx_2[early_mask]
group_id_early = group_id_2[early_mask]
exp_features_early = exp_features_2[early_mask]

# Check which experiments have early data
experiments_with_early_data = np.unique(experiment_idx_early)

# =================================================================
# BASE PREDICTIONS FROM LAYER 1 (PHYSICS-BASED)
# =================================================================

def calculate_layer1_prediction(X_minerals, exp_features, layer1_params):
    """Calculate Layer 1 predictions using physics-based model"""
    mineral_effect = (layer1_params['mineral_1_coeff'] * X_minerals[:, 0] +
                     layer1_params['mineral_2_coeff'] * X_minerals[:, 1] +
                     layer1_params['mineral_3_coeff'] * X_minerals[:, 2])
    
    exp_time_effect = np.sum([layer1_params['exp_time_coeffs'][i] * exp_features[:, i] 
                             for i in range(len(time_scales_layer1))], axis=0)
    
    mu_unconstrained = layer1_params['base_recovery'] + mineral_effect + exp_time_effect
    
    # Apply softplus constraint (approximation for sampling): softplus(x) = log(1 + exp(x))
    mu = np.log(1 + np.exp(mu_unconstrained))
    
    return mu, mineral_effect, exp_time_effect

# Calculate base predictions
base_pred_early, mineral_eff_early, time_eff_early = calculate_layer1_prediction(
    X_minerals_early, exp_features_early, layer1_params)

base_pred_all, mineral_eff_all, time_eff_all = calculate_layer1_prediction(
    X_minerals_2, exp_features_2, layer1_params)



# =================================================================
# BUILD LAYER 2 MODEL WITH PHYSICS-BASED ADJUSTMENTS
# =================================================================


with pm.Model() as layer2_model:
    
    # =================================================================
    # GROUP-LEVEL SYSTEMATIC ERROR (Applied to all data)
    # =================================================================
    
    group_systematic_error = pm.TruncatedNormal('group_systematic_error', 
                                               mu=-5.0,  # Expect negative bias
                                               sigma=5.0, 
                                               upper=0.0,  # Force negative or zero
                                               shape=n_groups_2)
    
    # =================================================================
    # EXPERIMENT ADJUSTMENTS: LEARNED FROM EARLY DATA ONLY
    # =================================================================
    
    # Recovery level adjustment (additive bias)
    experiment_recovery_adjustment = pm.Normal('experiment_recovery_adjustment', 
                                              mu=0.0, 
                                              sigma=3.0, 
                                              shape=n_experiments_2)
    
    # Time scale adjustments (modify the exponential behavior)
    experiment_timescale_adjustment = pm.Normal('experiment_timescale_adjustment', 
                                               mu=0.0, 
                                               sigma=0.2,  # Small adjustments to time scales
                                               shape=(n_experiments_2, len(time_scales_layer2)))
    
    # =================================================================
    # EARLY RECOVERY MODEL (First 200 days) - LEARNS ADJUSTMENTS
    # =================================================================
    
    # Apply experiment-specific time scale adjustments to early data
    # Get the adjustments for each observation's experiment
    exp_adjustments_early = experiment_timescale_adjustment[experiment_idx_early]  # Shape: (n_early_obs, n_time_scales)
    
    # Apply adjustments: multiply each time feature by (1 + adjustment)
    adjusted_exp_features_early = exp_features_early * (1 + exp_adjustments_early)
    
    # Calculate adjusted time effect for early period
    adjusted_time_effect_early = pm.math.sum([layer1_params['exp_time_coeffs'][i] * adjusted_exp_features_early[:, i] 
                                             for i in range(len(time_scales_layer1))], axis=0)
    
    # Early recovery prediction with adjustments
    early_prediction_unconstrained = (layer1_params['base_recovery'] + 
                                     mineral_eff_early +  # Fixed mineral effect
                                     adjusted_time_effect_early +  # Adjusted time effect
                                     group_systematic_error[group_id_early] +  # Group systematic error
                                     experiment_recovery_adjustment[experiment_idx_early])  # Recovery adjustment
    
    # Apply physics constraint (positive recovery): softplus(x) = log(1 + exp(x))
    early_prediction = pm.math.log(1 + pm.math.exp(early_prediction_unconstrained))
    
    # Early recovery likelihood - this trains the adjustments
    early_sigma = pm.HalfNormal('early_sigma', sigma=3.0)
    early_likelihood = pm.Normal('early_likelihood', 
                                mu=early_prediction, 
                                sigma=early_sigma, 
                                observed=y_recovery_early)
    
    # =================================================================
    # APPLY LEARNED ADJUSTMENTS TO FULL TIME SERIES
    # =================================================================
    
    # Apply experiment-specific adjustments to all data
    # Get the adjustments for each observation's experiment
    exp_adjustments_all = experiment_timescale_adjustment[experiment_idx_2]  # Shape: (n_obs, n_time_scales)
    
    # Apply adjustments: multiply each time feature by (1 + adjustment)
    adjusted_exp_features_all = exp_features_2 * (1 + exp_adjustments_all)
    
    # Calculate adjusted time effect for all data
    adjusted_time_effect_all = pm.math.sum([layer1_params['exp_time_coeffs'][i] * adjusted_exp_features_all[:, i] 
                                           for i in range(len(time_scales_layer1))], axis=0)
    
    # Full time series prediction with adjustments
    full_prediction_unconstrained = (layer1_params['base_recovery'] + 
                                    mineral_eff_all +  # Fixed mineral effect
                                    adjusted_time_effect_all +  # Adjusted time effect
                                    group_systematic_error[group_id_2] +  # Group systematic error
                                    experiment_recovery_adjustment[experiment_idx_2])  # Recovery adjustment
    
    # Apply physics constraint: softplus(x) = log(1 + exp(x))
    full_prediction = pm.math.log(1 + pm.math.exp(full_prediction_unconstrained))
    
    # Full time series likelihood
    full_sigma = pm.HalfNormal('full_sigma', sigma=3.0)
    full_likelihood = pm.Normal('full_likelihood', 
                               mu=full_prediction, 
                               sigma=full_sigma, 
                               observed=y_recovery_2)



# =================================================================
# SAMPLE LAYER 2 MODEL
# =================================================================
'''
compiled_layer2 = nutpie.compile_pymc_model(layer2_model)
trace_layer2 = nutpie.sample(compiled_layer2, draws=1000, tune=500, chains=6, seed=42, progress_bar=True, target_accept=0.99, cores=mp.cpu_count())

print(" Layer 2 sampling complete!")

# =================================================================
# EXTRACT AND DISPLAY RESULTS
# =================================================================



# Layer 2 results
recovery_adjustments = trace_layer2.posterior['experiment_recovery_adjustment'].mean(dim=['chain', 'draw']).values
timescale_adjustments = trace_layer2.posterior['experiment_timescale_adjustment'].mean(dim=['chain', 'draw']).values
group_errors = trace_layer2.posterior['group_systematic_error'].mean(dim=['chain', 'draw']).values

for group_id in range(n_groups_2):
    print(f"   Group {group_id}: {group_errors[group_id]:+.3f}%")

print(f"{'Exp':>3} {'Recovery Adj':>12} {'TimeScale Adjustments (100/200/300/400d)':>50}")
print("-" * 70)
for exp_id in range(min(10, n_experiments_2)):  # Show first 10 experiments
    ts_adjs = " ".join([f"{timescale_adjustments[exp_id, i]:+.3f}" for i in range(len(time_scales_layer2))])
    print(f"{exp_id:>3} {recovery_adjustments[exp_id]:>+11.3f}% {ts_adjs:>50}")
'''



#%%
# =================================================================
# PREPARE DATA FROM LAYER 2 DATASET
# =================================================================

# Using the Layer 2 dataset (new_data_numpy_con)
X_minerals_2 = new_data_numpy_con['X_minerals']
time_2 = new_data_numpy_con['time']
y_recovery_2 = new_data_numpy_con['y_recovery']
experiment_idx_2 = new_data_numpy_con['experiment_idx']
group_id_2 = new_data_numpy_con['group_id']
n_experiments_2 = new_data_numpy_con['n_experiments']
n_groups_2 = len(np.unique(group_id_2))



# Time scales for Layer 2
time_scales_layer2 = [150,250,400,600]

# Store all predictions and errors
loo_predictions = {}
all_rmse_values = []
all_mae_values = []

# =================================================================
# LEAVE-ONE-OUT CROSS-VALIDATION LOOP WITH IMMEDIATE PLOTTING
# =================================================================

for held_out_exp in range(n_experiments_2):
    print(f"\n{'='*80}")
    print(f"ITERATION {held_out_exp + 1}/{n_experiments_2}: Holding out Experiment {held_out_exp}")
    print(f"{'='*80}")
    
    # =================================================================
    # SPLIT DATA: TRAINING vs HELD-OUT
    # =================================================================
    
    # Training data: all experiments except held_out_exp
    train_mask = experiment_idx_2 != held_out_exp
    X_minerals_train = X_minerals_2[train_mask]
    time_train = time_2[train_mask]
    y_recovery_train = y_recovery_2[train_mask]
    experiment_idx_train = experiment_idx_2[train_mask]
    group_id_train = group_id_2[train_mask]
    
    # Held-out data: only the held-out experiment, first 250 days
    held_out_mask = (experiment_idx_2 == held_out_exp) & (time_2 <= 250)
    X_minerals_held = X_minerals_2[held_out_mask]
    time_held = time_2[held_out_mask]
    y_recovery_held = y_recovery_2[held_out_mask]
    group_id_held = group_id_2[held_out_mask]
    
    # Full held-out data for comparison (all time points)
    held_out_full_mask = experiment_idx_2 == held_out_exp
    time_held_full = time_2[held_out_full_mask]
    y_recovery_held_full = y_recovery_2[held_out_full_mask]

    # =================================================================
    # CREATE TIME FEATURES
    # =================================================================
    
    # Training data time features (using only early period for training)
    train_early_mask = time_train <= 250
    exp_features_train_early = np.column_stack([
        1 - np.exp(-time_train[train_early_mask] / ts) for ts in time_scales_layer2
    ])
    
    # Full training time features
    exp_features_train_full = np.column_stack([
        1 - np.exp(-time_train / ts) for ts in time_scales_layer2
    ])
    
    # Held-out early time features
    exp_features_held = np.column_stack([
        1 - np.exp(-time_held / ts) for ts in time_scales_layer2
    ])
    
    # =================================================================
    # TRAIN LAYER 2 MODEL (WITHOUT HELD-OUT EXPERIMENT)
    # =================================================================
    
    print(" Training Layer 2 model on training set...")
    
    with pm.Model() as layer2_loo_model:
        
        # Group systematic errors
        group_systematic_error = pm.TruncatedNormal(
            'group_systematic_error', 
            mu=-5.0, 
            sigma=5.0, 
            upper=0.0,
            shape=n_groups_2
        )
        
        # Experiment adjustments
        experiment_recovery_adjustment = pm.Normal(
            'experiment_recovery_adjustment', 
            mu=0.0, 
            sigma=3.0, 
            shape=n_experiments_2
        )
        
        experiment_timescale_adjustment = pm.Normal(
            'experiment_timescale_adjustment', 
            mu=0.0, 
            sigma=0.2,
            shape=(n_experiments_2, len(time_scales_layer2))
        )
        
        # Calculate predictions for EARLY training data
        mineral_effect_train = (
            layer1_params['mineral_1_coeff'] * X_minerals_train[train_early_mask, 0] +
            layer1_params['mineral_2_coeff'] * X_minerals_train[train_early_mask, 1] +
            layer1_params['mineral_3_coeff'] * X_minerals_train[train_early_mask, 2]
        )
        
        # Apply adjustments
        exp_idx_early = experiment_idx_train[train_early_mask]
        exp_adjustments_early = experiment_timescale_adjustment[exp_idx_early]
        adjusted_exp_features_early = exp_features_train_early * (1 + exp_adjustments_early)
        
        # Time effect
        time_effect_early = pm.math.sum([
            layer1_params['exp_time_coeffs'][i] * adjusted_exp_features_early[:, i]
            for i in range(min(len(layer1_params['exp_time_coeffs']), len(time_scales_layer2)))
        ], axis=0)
        
        # Early prediction
        mu_early_unconstrained = (
            layer1_params['base_recovery'] +
            mineral_effect_train +
            time_effect_early +
            group_systematic_error[group_id_train[train_early_mask]] +
            experiment_recovery_adjustment[exp_idx_early]
        )
        mu_early = pm.math.log(1 + pm.math.exp(mu_early_unconstrained))
        
        # Early likelihood
        sigma_early = pm.HalfNormal('sigma_early', sigma=3.0)
        early_likelihood = pm.Normal(
            'early_likelihood',
            mu=mu_early,
            sigma=sigma_early,
            observed=y_recovery_train[train_early_mask]
        )
        
        # Full time series predictions
        mineral_effect_full = (
            layer1_params['mineral_1_coeff'] * X_minerals_train[:, 0] +
            layer1_params['mineral_2_coeff'] * X_minerals_train[:, 1] +
            layer1_params['mineral_3_coeff'] * X_minerals_train[:, 2]
        )
        
        exp_adjustments_full = experiment_timescale_adjustment[experiment_idx_train]
        adjusted_exp_features_full = exp_features_train_full * (1 + exp_adjustments_full)
        
        time_effect_full = pm.math.sum([
            layer1_params['exp_time_coeffs'][i] * adjusted_exp_features_full[:, i]
            for i in range(min(len(layer1_params['exp_time_coeffs']), len(time_scales_layer2)))
        ], axis=0)
        
        mu_full_unconstrained = (
            layer1_params['base_recovery'] +
            mineral_effect_full +
            time_effect_full +
            group_systematic_error[group_id_train] +
            experiment_recovery_adjustment[experiment_idx_train]
        )
        mu_full = pm.math.log(1 + pm.math.exp(mu_full_unconstrained))
        
        sigma_full = pm.HalfNormal('sigma_full', sigma=3.0)
        full_likelihood = pm.Normal(
            'full_likelihood',
            mu=mu_full,
            sigma=sigma_full,
            observed=y_recovery_train
        )
    
    # Sample the model
    compiled_loo = nutpie.compile_pymc_model(layer2_loo_model)
    trace_loo = nutpie.sample(
        compiled_loo, 
        draws=500,
        tune=300, 
        chains=6,
        seed=42 + held_out_exp,
        progress_bar=True,
        cores=mp.cpu_count()
    )
    
    # Extract learned parameters
    group_errors_loo = trace_loo.posterior['group_systematic_error'].mean(dim=['chain', 'draw']).values
    
    # =================================================================
    # LEARN HELD-OUT EXPERIMENT ADJUSTMENTS
    # =================================================================
    

    with pm.Model() as held_out_model:
        
        # Only learn adjustments for the held-out experiment
        held_out_recovery_adj = pm.Normal('held_out_recovery_adj', mu=0.0, sigma=3.0)
        held_out_timescale_adj = pm.Normal(
            'held_out_timescale_adj', 
            mu=0.0, 
            sigma=0.2,
            shape=len(time_scales_layer2)
        )
        
        # Fixed parameters
        held_out_group = group_id_held[0]
        group_error = group_errors_loo[held_out_group]
        
        # Calculate prediction for held-out early data
        mineral_effect_held = (
            layer1_params['mineral_1_coeff'] * X_minerals_held[:, 0] +
            layer1_params['mineral_2_coeff'] * X_minerals_held[:, 1] +
            layer1_params['mineral_3_coeff'] * X_minerals_held[:, 2]
        )
        
        adjusted_exp_features_held = exp_features_held * (1 + held_out_timescale_adj)
        
        time_effect_held = pm.math.sum([
            layer1_params['exp_time_coeffs'][i] * adjusted_exp_features_held[:, i]
            for i in range(min(len(layer1_params['exp_time_coeffs']), len(time_scales_layer2)))
        ], axis=0)
        
        mu_held_unconstrained = (
            layer1_params['base_recovery'] +
            mineral_effect_held +
            time_effect_held +
            group_error +
            held_out_recovery_adj
        )
        mu_held = pm.math.log(1 + pm.math.exp(mu_held_unconstrained))
        
        sigma_held = pm.HalfNormal('sigma_held', sigma=3.0)
        held_likelihood = pm.Normal(
            'held_likelihood',
            mu=mu_held,
            sigma=sigma_held,
            observed=y_recovery_held
        )
    
    # Sample to learn held-out adjustments
    compiled_held = nutpie.compile_pymc_model(held_out_model)
    trace_held = nutpie.sample(
        compiled_held,
        draws=500,
        tune=300,
        chains=6,
        seed=42 + held_out_exp + 1000,
        progress_bar=True,
        cores=mp.cpu_count()
    )
    
    # Extract held-out adjustments
    held_recovery_adj = trace_held.posterior['held_out_recovery_adj'].mean(dim=['chain', 'draw']).values
    held_timescale_adj = trace_held.posterior['held_out_timescale_adj'].mean(dim=['chain', 'draw']).values
    
    # =================================================================
    # GENERATE PREDICTIONS (0 to 1500 days) WITH ALIGNMENT
    # =================================================================

    
    # Create future time points
    future_times = np.arange(0, 1501, 10)
    
    # Get mineral composition
    exp_minerals = X_minerals_held[0]
    
    # Create exponential features
    future_exp_features = np.column_stack([
        1 - np.exp(-future_times / ts) for ts in time_scales_layer2
    ])
    
    # Apply adjustments
    adjusted_features = future_exp_features * (1 + held_timescale_adj)
    
    # Calculate prediction
    mineral_contribution = (
        layer1_params['mineral_1_coeff'] * exp_minerals[0] +
        layer1_params['mineral_2_coeff'] * exp_minerals[1] +
        layer1_params['mineral_3_coeff'] * exp_minerals[2]
    )
    
    time_contribution = np.sum([
        layer1_params['exp_time_coeffs'][i] * adjusted_features[:, i]
        for i in range(min(len(layer1_params['exp_time_coeffs']), len(time_scales_layer2)))
    ], axis=0)
    
    prediction_unconstrained = (
        layer1_params['base_recovery'] +
        mineral_contribution +
        time_contribution +
        group_error +
        held_recovery_adj
    )
    
    predicted_recovery = np.log(1 + np.exp(prediction_unconstrained))
    
    # =================================================================
    # ALIGN PREDICTION CURVE WITH LAST TRAINING POINT
    # =================================================================
    
    # Get the actual recovery value at the last training point (closest to 250 days)
    last_training_idx = np.argmax(time_held)  # Get the index of the maximum time <= 250
    last_training_time = time_held[last_training_idx]
    last_training_recovery = y_recovery_held[last_training_idx]
    
    # Get the predicted value at 250 days (or closest time point)
    # Find the closest time point in future_times to the last training time
    closest_idx = np.argmin(np.abs(future_times - last_training_time))
    predicted_at_last_training = predicted_recovery[closest_idx]
    
    # Calculate the shift needed
    shift = last_training_recovery - predicted_at_last_training
    
    # Apply the shift to the entire prediction curve
    predicted_recovery_aligned = predicted_recovery + shift
    
    # =================================================================
    # CALCULATE METRICS (using aligned predictions)
    # =================================================================
    
    # Calculate RMSE for predictions beyond 250 days
    future_mask = time_held_full > 250
    rmse_future = mae_future = 0
    
    if np.any(future_mask):
        actual_future_times = time_held_full[future_mask]
        actual_future_recovery = y_recovery_held_full[future_mask]
        
        predicted_at_actual = np.interp(actual_future_times, 
                                       future_times, 
                                       predicted_recovery_aligned)
        
        errors = predicted_at_actual - actual_future_recovery
        rmse_future = np.sqrt(np.mean(errors**2))
        mae_future = np.mean(np.abs(errors))
    
    # Store results (using aligned predictions)
    loo_predictions[held_out_exp] = {
        'times': future_times,
        'predicted_recovery': predicted_recovery_aligned,
        'actual_times': time_held_full,
        'actual_recovery': y_recovery_held_full,
        'early_times': time_held,
        'early_recovery': y_recovery_held,
        'rmse': rmse_future,
        'mae': mae_future
    }
    
    all_rmse_values.append(rmse_future)
    all_mae_values.append(mae_future)
    
    # =================================================================
    # PLOT IMMEDIATELY AFTER COMPLETION WITH ERROR BARS
    # =================================================================
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot predicted recovery curve (aligned)
    ax.plot(future_times, predicted_recovery_aligned, 'b-', 
            label=f'Predicted (RMSE={rmse_future:.2f}%)', linewidth=2, alpha=0.8)
    
    # Mark early data region
    ax.axvspan(0, 250, alpha=0.2, color='green', 
              label='Used for learning (≤250 days)')
    
    # Plot early observations (no error bars)
    ax.scatter(time_held, y_recovery_held,
              c='red', s=30, alpha=0.8, label='Early data used', zorder=5)
    
    # Plot all actual observations with error bars for points after 250 days
    # First plot points <= 250 days without error bars
    early_mask_full = time_held_full <= 250
    ax.scatter(time_held_full[early_mask_full], y_recovery_held_full[early_mask_full],
              c='orange', s=20, alpha=0.6, marker='^', zorder=4)
    
    # Then plot points > 250 days with error bars
    future_mask_plot = time_held_full > 250
    if np.any(future_mask_plot):
        future_times_plot = time_held_full[future_mask_plot]
        future_recovery_plot = y_recovery_held_full[future_mask_plot]
        
        # Plot with error bars (±6%)
        ax.errorbar(future_times_plot, future_recovery_plot,
                   yerr=6.0,  # ±6% error bars
                   fmt='^', color='orange', markersize=5, alpha=0.6,
                   capsize=3, capthick=1, elinewidth=1,
                   label='Actual (>250d, ±6% error)', zorder=4)
    
    # Add vertical line at 250 days
    ax.axvline(x=250, color='gray', linestyle='--', alpha=0.5)
    
    # Labels and formatting
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Recovery (%)', fontsize=12)
    ax.set_title(f'Leave-One-Out: Experiment {held_out_exp} (Iteration {held_out_exp + 1}/{n_experiments_2})\n' +
                f'Group {group_id_held[0]} | Recovery Adj: {held_recovery_adj:+.2f}% | ' +
                f'RMSE (>250d): {rmse_future:.2f}% | MAE: {mae_future:.2f}%', 
                fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, max(100, predicted_recovery_aligned.max() * 1.1))
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary for this experiment
    print(f" Experiment {held_out_exp} complete:")
    print(f"   Recovery adjustment: {held_recovery_adj:+.3f}%")
    print(f"   Timescale adjustments: {held_timescale_adj}")
    print(f"   RMSE (>250 days): {rmse_future:.2f}%")
    print(f"   MAE (>250 days): {mae_future:.2f}%")
    print(f"   Running average RMSE: {np.mean(all_rmse_values):.2f}%")

#%%
# =================================================================
# FINAL SUMMARY PLOT - ALL EXPERIMENTS TOGETHER WITH ERROR BARS
# =================================================================

print(f"\n{'='*80}")
print("CREATING SUMMARY PLOT OF ALL EXPERIMENTS")
print(f"{'='*80}")

# Create summary figure with all experiments
n_cols = 4
n_rows = int(np.ceil(n_experiments_2 / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))

if n_rows == 1:
    axes = axes.reshape(1, -1)
if n_cols == 1:
    axes = axes.reshape(-1, 1)

for exp_id in range(n_experiments_2):
    row = exp_id // n_cols
    col = exp_id % n_cols
    ax = axes[row, col]
    
    pred = loo_predictions[exp_id]
    
    # Plot predicted recovery curve
    ax.plot(pred['times'], pred['predicted_recovery'], 'b-', 
            linewidth=2, alpha=0.8)
    
    # Mark early data region
    ax.axvspan(0, 250, alpha=0.2, color='green')
    
    # Plot early observations
    ax.scatter(pred['early_times'], pred['early_recovery'],
              c='red', s=15, alpha=0.8, zorder=5)
    
    # Plot all actual observations with error bars for points after 250 days
    # First plot points <= 250 days without error bars
    early_mask = pred['actual_times'] <= 250
    ax.scatter(pred['actual_times'][early_mask], pred['actual_recovery'][early_mask],
              c='orange', s=10, alpha=0.6, marker='^', zorder=4)
    
    # Then plot points > 250 days with error bars
    future_mask = pred['actual_times'] > 250
    if np.any(future_mask):
        ax.errorbar(pred['actual_times'][future_mask], pred['actual_recovery'][future_mask],
                   yerr=6.0,  # ±6% error bars
                   fmt='^', color='orange', markersize=4, alpha=0.6,
                   capsize=2, capthick=0.5, elinewidth=0.5, zorder=4)
    
    ax.set_title(f'Exp {exp_id} | RMSE: {pred["rmse"]:.1f}%', fontsize=9)
    ax.set_xlabel('Time (days)', fontsize=8)
    ax.set_ylabel('Recovery (%)', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 100)

# Remove empty subplots
for i in range(n_experiments_2, n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    fig.delaxes(axes[row, col])

plt.suptitle(f'Leave-One-Out Cross-Validation Summary (All {n_experiments_2} Experiments)\n' +
            f'Mean RMSE: {np.mean(all_rmse_values):.2f}% | Mean MAE: {np.mean(all_mae_values):.2f}%',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#%%

# =================================================================
# FINAL PERFORMANCE SUMMARY
# =================================================================

print(f"\n{'='*80}")
print("FINAL PERFORMANCE SUMMARY")
print(f"{'='*80}")

print(f"\n Overall Metrics (predictions > 250 days):")
print(f"   Mean RMSE: {np.mean(all_rmse_values):.2f}%")
print(f"   Std RMSE:  {np.std(all_rmse_values):.2f}%")
print(f"   Min RMSE:  {np.min(all_rmse_values):.2f}%")
print(f"   Max RMSE:  {np.max(all_rmse_values):.2f}%")
print(f"\n   Mean MAE:  {np.mean(all_mae_values):.2f}%")
print(f"   Std MAE:   {np.std(all_mae_values):.2f}%")

print(f"\n Leave-one-out cross-validation complete!")
print(f"   - Trained {n_experiments_2} models")
print(f"   - Each used only first 250 days to predict to 1500 days")
print(f"   - Results plotted immediately after each iteration")

#%%
# =================================================================
# PREPARE DATA FROM LAYER 2 DATASET
# =================================================================

# Using the Layer 2 dataset (new_data_numpy_con)
X_minerals_2 = new_data_numpy_con['X_minerals']
time_2 = new_data_numpy_con['time']
y_recovery_2 = new_data_numpy_con['y_recovery']
experiment_idx_2 = new_data_numpy_con['experiment_idx']
group_id_2 = new_data_numpy_con['group_id']
n_experiments_2 = new_data_numpy_con['n_experiments']
n_groups_2 = len(np.unique(group_id_2))



# Time scales for Layer 2
time_scales_layer2 = [150,250,400,600]

# Store all predictions and errors
loo_predictions = {}
all_rmse_values = []
all_mae_values = []

# =================================================================
# LEAVE-ONE-OUT CROSS-VALIDATION LOOP WITH IMMEDIATE PLOTTING
# =================================================================

for held_out_exp in range(n_experiments_2):
    print(f"\n{'='*80}")
    print(f"ITERATION {held_out_exp + 1}/{n_experiments_2}: Holding out Experiment {held_out_exp}")
    print(f"{'='*80}")
    
    # =================================================================
    # SPLIT DATA: TRAINING vs HELD-OUT
    # =================================================================
    
    # Training data: all experiments except held_out_exp
    train_mask = experiment_idx_2 != held_out_exp
    X_minerals_train = X_minerals_2[train_mask]
    time_train = time_2[train_mask]
    y_recovery_train = y_recovery_2[train_mask]
    experiment_idx_train = experiment_idx_2[train_mask]
    group_id_train = group_id_2[train_mask]
    
    # Held-out data: only the held-out experiment, first 250 days
    held_out_mask = (experiment_idx_2 == held_out_exp) & (time_2 <= 250)
    X_minerals_held = X_minerals_2[held_out_mask]
    time_held = time_2[held_out_mask]
    y_recovery_held = y_recovery_2[held_out_mask]
    group_id_held = group_id_2[held_out_mask]
    
    # Full held-out data for comparison (all time points)
    held_out_full_mask = experiment_idx_2 == held_out_exp
    time_held_full = time_2[held_out_full_mask]
    y_recovery_held_full = y_recovery_2[held_out_full_mask]

    # =================================================================
    # CREATE TIME FEATURES
    # =================================================================
    
    # Training data time features (using only early period for training)
    train_early_mask = time_train <= 250
    exp_features_train_early = np.column_stack([
        1 - np.exp(-time_train[train_early_mask] / ts) for ts in time_scales_layer2
    ])
    
    # Full training time features
    exp_features_train_full = np.column_stack([
        1 - np.exp(-time_train / ts) for ts in time_scales_layer2
    ])
    
    # Held-out early time features
    exp_features_held = np.column_stack([
        1 - np.exp(-time_held / ts) for ts in time_scales_layer2
    ])
    
    # =================================================================
    # TRAIN LAYER 2 MODEL (WITHOUT HELD-OUT EXPERIMENT)
    # =================================================================
    
    print("🔧 Training Layer 2 model on training set...")
    
    with pm.Model() as layer2_loo_model:
        
        # Group systematic errors
        group_systematic_error = pm.TruncatedNormal(
            'group_systematic_error', 
            mu=-5.0, 
            sigma=5.0, 
            upper=0.0,
            shape=n_groups_2
        )
        
        # Experiment adjustments
        experiment_recovery_adjustment = pm.Normal(
            'experiment_recovery_adjustment', 
            mu=0.0, 
            sigma=3.0, 
            shape=n_experiments_2
        )
        
        experiment_timescale_adjustment = pm.Normal(
            'experiment_timescale_adjustment', 
            mu=0.0, 
            sigma=0.2,
            shape=(n_experiments_2, len(time_scales_layer2))
        )
        
        # Calculate predictions for EARLY training data
        mineral_effect_train = (
            layer1_params['mineral_1_coeff'] * X_minerals_train[train_early_mask, 0] +
            layer1_params['mineral_2_coeff'] * X_minerals_train[train_early_mask, 1] +
            layer1_params['mineral_3_coeff'] * X_minerals_train[train_early_mask, 2]
        )
        
        # Apply adjustments
        exp_idx_early = experiment_idx_train[train_early_mask]
        exp_adjustments_early = experiment_timescale_adjustment[exp_idx_early]
        adjusted_exp_features_early = exp_features_train_early * (1 + exp_adjustments_early)
        
        # Time effect
        time_effect_early = pm.math.sum([
            layer1_params['exp_time_coeffs'][i] * adjusted_exp_features_early[:, i]
            for i in range(min(len(layer1_params['exp_time_coeffs']), len(time_scales_layer2)))
        ], axis=0)
        
        # Early prediction
        mu_early_unconstrained = (
            layer1_params['base_recovery'] +
            mineral_effect_train +
            time_effect_early +
            group_systematic_error[group_id_train[train_early_mask]] +
            experiment_recovery_adjustment[exp_idx_early]
        )
        mu_early = pm.math.log(1 + pm.math.exp(mu_early_unconstrained))
        
        # Early likelihood
        sigma_early = pm.HalfNormal('sigma_early', sigma=3.0)
        early_likelihood = pm.Normal(
            'early_likelihood',
            mu=mu_early,
            sigma=sigma_early,
            observed=y_recovery_train[train_early_mask]
        )
        
        # Full time series predictions
        mineral_effect_full = (
            layer1_params['mineral_1_coeff'] * X_minerals_train[:, 0] +
            layer1_params['mineral_2_coeff'] * X_minerals_train[:, 1] +
            layer1_params['mineral_3_coeff'] * X_minerals_train[:, 2]
        )
        
        exp_adjustments_full = experiment_timescale_adjustment[experiment_idx_train]
        adjusted_exp_features_full = exp_features_train_full * (1 + exp_adjustments_full)
        
        time_effect_full = pm.math.sum([
            layer1_params['exp_time_coeffs'][i] * adjusted_exp_features_full[:, i]
            for i in range(min(len(layer1_params['exp_time_coeffs']), len(time_scales_layer2)))
        ], axis=0)
        
        mu_full_unconstrained = (
            layer1_params['base_recovery'] +
            mineral_effect_full +
            time_effect_full +
            group_systematic_error[group_id_train] +
            experiment_recovery_adjustment[experiment_idx_train]
        )
        mu_full = pm.math.log(1 + pm.math.exp(mu_full_unconstrained))
        
        sigma_full = pm.HalfNormal('sigma_full', sigma=3.0)
        full_likelihood = pm.Normal(
            'full_likelihood',
            mu=mu_full,
            sigma=sigma_full,
            observed=y_recovery_train
        )
    
    # Sample the model
    compiled_loo = nutpie.compile_pymc_model(layer2_loo_model)
    trace_loo = nutpie.sample(
        compiled_loo, 
        draws=500,
        tune=300, 
        chains=6,
        seed=42 + held_out_exp,
        progress_bar=True,
        cores=mp.cpu_count()
    )
    
    # Extract learned parameters
    group_errors_loo = trace_loo.posterior['group_systematic_error'].mean(dim=['chain', 'draw']).values
    
    # =================================================================
    # LEARN HELD-OUT EXPERIMENT ADJUSTMENTS
    # =================================================================
    

    with pm.Model() as held_out_model:
        
        # Only learn adjustments for the held-out experiment
        held_out_recovery_adj = pm.Normal('held_out_recovery_adj', mu=0.0, sigma=3.0)
        held_out_timescale_adj = pm.Normal(
            'held_out_timescale_adj', 
            mu=0.0, 
            sigma=0.2,
            shape=len(time_scales_layer2)
        )
        
        # Fixed parameters
        held_out_group = group_id_held[0]
        group_error = group_errors_loo[held_out_group]
        
        # Calculate prediction for held-out early data
        mineral_effect_held = (
            layer1_params['mineral_1_coeff'] * X_minerals_held[:, 0] +
            layer1_params['mineral_2_coeff'] * X_minerals_held[:, 1] +
            layer1_params['mineral_3_coeff'] * X_minerals_held[:, 2]
        )
        
        adjusted_exp_features_held = exp_features_held * (1 + held_out_timescale_adj)
        
        time_effect_held = pm.math.sum([
            layer1_params['exp_time_coeffs'][i] * adjusted_exp_features_held[:, i]
            for i in range(min(len(layer1_params['exp_time_coeffs']), len(time_scales_layer2)))
        ], axis=0)
        
        mu_held_unconstrained = (
            layer1_params['base_recovery'] +
            mineral_effect_held +
            time_effect_held +
            group_error +
            held_out_recovery_adj
        )
        mu_held = pm.math.log(1 + pm.math.exp(mu_held_unconstrained))
        
        sigma_held = pm.HalfNormal('sigma_held', sigma=3.0)
        held_likelihood = pm.Normal(
            'held_likelihood',
            mu=mu_held,
            sigma=sigma_held,
            observed=y_recovery_held
        )
    
    # Sample to learn held-out adjustments
    compiled_held = nutpie.compile_pymc_model(held_out_model)
    trace_held = nutpie.sample(
        compiled_held,
        draws=500,
        tune=300,
        chains=6,
        seed=42 + held_out_exp + 1000,
        progress_bar=True,
        cores=mp.cpu_count()
    )
    
    # Extract held-out adjustments
    held_recovery_adj = trace_held.posterior['held_out_recovery_adj'].mean(dim=['chain', 'draw']).values
    held_timescale_adj = trace_held.posterior['held_out_timescale_adj'].mean(dim=['chain', 'draw']).values
    
    # =================================================================
    # GENERATE PREDICTIONS (0 to 1500 days) WITH ALIGNMENT
    # =================================================================

    
    # Create future time points
    future_times = np.arange(0, 1501, 10)
    
    # Get mineral composition
    exp_minerals = X_minerals_held[0]
    
    # Create exponential features
    future_exp_features = np.column_stack([
        1 - np.exp(-future_times / ts) for ts in time_scales_layer2
    ])
    
    # Apply adjustments
    adjusted_features = future_exp_features * (1 + held_timescale_adj)
    
    # Calculate prediction
    mineral_contribution = (
        layer1_params['mineral_1_coeff'] * exp_minerals[0] +
        layer1_params['mineral_2_coeff'] * exp_minerals[1] +
        layer1_params['mineral_3_coeff'] * exp_minerals[2]
    )
    
    time_contribution = np.sum([
        layer1_params['exp_time_coeffs'][i] * adjusted_features[:, i]
        for i in range(min(len(layer1_params['exp_time_coeffs']), len(time_scales_layer2)))
    ], axis=0)
    
    prediction_unconstrained = (
        layer1_params['base_recovery'] +
        mineral_contribution +
        time_contribution +
        group_error +
        held_recovery_adj
    )
    
    predicted_recovery = np.log(1 + np.exp(prediction_unconstrained))
    
    # =================================================================
    # ALIGN PREDICTION CURVE WITH LAST TRAINING POINT
    # =================================================================
    
    # Get the actual recovery value at the last training point (closest to 250 days)
    last_training_idx = np.argmax(time_held)  # Get the index of the maximum time <= 250
    last_training_time = time_held[last_training_idx]
    last_training_recovery = y_recovery_held[last_training_idx]
    
    # Get the predicted value at 250 days (or closest time point)
    # Find the closest time point in future_times to the last training time
    closest_idx = np.argmin(np.abs(future_times - last_training_time))
    predicted_at_last_training = predicted_recovery[closest_idx]
    
    # Calculate the shift needed
    shift = last_training_recovery - predicted_at_last_training
    
    # Apply the shift to the entire prediction curve
    predicted_recovery_aligned = predicted_recovery + shift
    
    # =================================================================
    # CALCULATE METRICS (using aligned predictions)
    # =================================================================
    
    # Calculate RMSE for predictions beyond 250 days
    future_mask = time_held_full > 250
    rmse_future = mae_future = 0
    
    if np.any(future_mask):
        actual_future_times = time_held_full[future_mask]
        actual_future_recovery = y_recovery_held_full[future_mask]
        
        predicted_at_actual = np.interp(actual_future_times, 
                                       future_times, 
                                       predicted_recovery_aligned)
        
        errors = predicted_at_actual - actual_future_recovery
        rmse_future = np.sqrt(np.mean(errors**2))
        mae_future = np.mean(np.abs(errors))
    
    # Store results (using aligned predictions)
    loo_predictions[held_out_exp] = {
        'times': future_times,
        'predicted_recovery': predicted_recovery_aligned,
        'actual_times': time_held_full,
        'actual_recovery': y_recovery_held_full,
        'early_times': time_held,
        'early_recovery': y_recovery_held,
        'rmse': rmse_future,
        'mae': mae_future
    }
    
    all_rmse_values.append(rmse_future)
    all_mae_values.append(mae_future)
    
    # =================================================================
    # PLOT IMMEDIATELY AFTER COMPLETION WITH ERROR BARS
    # =================================================================
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot predicted recovery curve (aligned)
    ax.plot(future_times, predicted_recovery_aligned, 'b-', 
            label=f'Predicted (RMSE={rmse_future:.2f}%)', linewidth=2, alpha=0.8)
    
    # Mark early data region
    ax.axvspan(0, 250, alpha=0.2, color='green', 
              label='Used for learning (≤250 days)')
    
    # Plot early observations (no error bars)
    ax.scatter(time_held, y_recovery_held,
              c='red', s=30, alpha=0.8, label='Early data used', zorder=5)
    
    # Plot all actual observations with error bars for points after 250 days
    # First plot points <= 250 days without error bars
    early_mask_full = time_held_full <= 250
    ax.scatter(time_held_full[early_mask_full], y_recovery_held_full[early_mask_full],
              c='orange', s=20, alpha=0.6, marker='^', zorder=4)
    
    # Then plot points > 250 days with error bars
    future_mask_plot = time_held_full > 250
    if np.any(future_mask_plot):
        future_times_plot = time_held_full[future_mask_plot]
        future_recovery_plot = y_recovery_held_full[future_mask_plot]
        
        # Plot with error bars (±6%)
        ax.errorbar(future_times_plot, future_recovery_plot,
                   yerr=6.0,  # ±6% error bars
                   fmt='^', color='orange', markersize=5, alpha=0.6,
                   capsize=3, capthick=1, elinewidth=1,
                   label='Actual (>250d, ±6% error)', zorder=4)
    
    # Add vertical line at 250 days
    ax.axvline(x=250, color='gray', linestyle='--', alpha=0.5)
    
    # Labels and formatting
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Recovery (%)', fontsize=12)
    ax.set_title(f'Leave-One-Out: Experiment {held_out_exp} (Iteration {held_out_exp + 1}/{n_experiments_2})\n' +
                f'Group {group_id_held[0]} | Recovery Adj: {held_recovery_adj:+.2f}% | ' +
                f'RMSE (>250d): {rmse_future:.2f}% | MAE: {mae_future:.2f}%', 
                fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, max(100, predicted_recovery_aligned.max() * 1.1))
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary for this experiment
    print(f"✅ Experiment {held_out_exp} complete:")
    print(f"   Recovery adjustment: {held_recovery_adj:+.3f}%")
    print(f"   Timescale adjustments: {held_timescale_adj}")
    print(f"   RMSE (>250 days): {rmse_future:.2f}%")
    print(f"   MAE (>250 days): {mae_future:.2f}%")
    print(f"   Running average RMSE: {np.mean(all_rmse_values):.2f}%")

#%%
# =================================================================
# FINAL SUMMARY PLOT - ALL EXPERIMENTS TOGETHER WITH ERROR BARS
# =================================================================

print(f"\n{'='*80}")
print("CREATING SUMMARY PLOT OF ALL EXPERIMENTS")
print(f"{'='*80}")

# Create summary figure with all experiments
n_cols = 4
n_rows = int(np.ceil(n_experiments_2 / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))

if n_rows == 1:
    axes = axes.reshape(1, -1)
if n_cols == 1:
    axes = axes.reshape(-1, 1)

for exp_id in range(n_experiments_2):
    row = exp_id // n_cols
    col = exp_id % n_cols
    ax = axes[row, col]
    
    pred = loo_predictions[exp_id]
    
    # Plot predicted recovery curve
    ax.plot(pred['times'], pred['predicted_recovery'], 'b-', 
            linewidth=2, alpha=0.8)
    
    # Mark early data region
    ax.axvspan(0, 250, alpha=0.2, color='green')
    
    # Plot early observations
    ax.scatter(pred['early_times'], pred['early_recovery'],
              c='red', s=15, alpha=0.8, zorder=5)
    
    # Plot all actual observations with error bars for points after 250 days
    # First plot points <= 250 days without error bars
    early_mask = pred['actual_times'] <= 250
    ax.scatter(pred['actual_times'][early_mask], pred['actual_recovery'][early_mask],
              c='orange', s=10, alpha=0.6, marker='^', zorder=4)
    
    # Then plot points > 250 days with error bars
    future_mask = pred['actual_times'] > 250
    if np.any(future_mask):
        ax.errorbar(pred['actual_times'][future_mask], pred['actual_recovery'][future_mask],
                   yerr=6.0,  # ±6% error bars
                   fmt='^', color='orange', markersize=4, alpha=0.6,
                   capsize=2, capthick=0.5, elinewidth=0.5, zorder=4)
    
    ax.set_title(f'Exp {exp_id} | RMSE: {pred["rmse"]:.1f}%', fontsize=9)
    ax.set_xlabel('Time (days)', fontsize=8)
    ax.set_ylabel('Recovery (%)', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 100)

# Remove empty subplots
for i in range(n_experiments_2, n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    fig.delaxes(axes[row, col])

plt.suptitle(f'Leave-One-Out Cross-Validation Summary (All {n_experiments_2} Experiments)\n' +
            f'Mean RMSE: {np.mean(all_rmse_values):.2f}% | Mean MAE: {np.mean(all_mae_values):.2f}%',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#%%
# =================================================================
# FINAL PERFORMANCE SUMMARY
# =================================================================

print(f"\n{'='*80}")
print("FINAL PERFORMANCE SUMMARY")
print(f"{'='*80}")

print(f"\n Overall Metrics (predictions > 250 days):")
print(f"   Mean RMSE: {np.mean(all_rmse_values):.2f}%")
print(f"   Std RMSE:  {np.std(all_rmse_values):.2f}%")
print(f"   Min RMSE:  {np.min(all_rmse_values):.2f}%")
print(f"   Max RMSE:  {np.max(all_rmse_values):.2f}%")
print(f"\n   Mean MAE:  {np.mean(all_mae_values):.2f}%")
print(f"   Std MAE:   {np.std(all_mae_values):.2f}%")

print(f"\n Leave-one-out cross-validation complete!")
print(f"   - Trained {n_experiments_2} models")
print(f"   - Each used only first 250 days to predict to 1500 days")
print(f"   - Results plotted immediately after each iteration")


# In[ ]:
