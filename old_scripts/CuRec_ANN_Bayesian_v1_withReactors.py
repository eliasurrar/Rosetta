#%%
# Load the necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split, KFold
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools
import copy
import random

from multiprocessing import Pool

from functools import partial

# ---------------------------
# PyTorch Imports and Setup
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR

# ---------------------------
# PyTorch Imports and Setup
# ---------------------------
import pymc as pm
import arviz as az
import pytensor.tensor as pt  # For custom tensor ops in model

'''
# Try to use MPS (Mac), CUDA if available, otherwise CPU
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS device for PyTorch")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA device for PyTorch")
else:
    device = torch.device('cpu')
    print("Using CPU device for PyTorch")
'''

device = torch.device('cpu')

# load the data
# Save the final DataFrames with reactors fit metrics
df_model_recCu_catalyzed_projects = pd.read_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/columns/df_model_catalyzed_projects_with_reactors_fit.csv', sep=',')
df_model_recCu_control_projects = pd.read_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/columns/df_model_control_projects_with_reactors_fit.csv', sep=',')
df_model_recCu_catcontrol_projects = pd.read_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/columns/df_model_catcontrol_projects_with_reactors_fit.csv', sep=',')

df_model_recCu_catcontrol_projects[df_model_recCu_catcontrol_projects['project_col_id'] == '026_jetti_project_file_ps_2']

df_reactors = pd.read_csv('/Users/administration/Library/CloudStorage/OneDrive-SharedLibraries-JettiResources/Jetti Vancouver Projects - projects_database - Documents/reactors/dataset_per_sample_reactor.csv', sep=',')

cols_to_check = [
    'avg_h2so4_kg_t',
    'cu_%',
    'cu_seq_h2so4_norm%',
    'cu_seq_nacn_norm%',
    'cu_seq_a_r_norm%',
    'fe_%',
    'grouped_copper_sulfides',
    'grouped_secondary_copper',
    'grouped_acid_generating_sulfides',
    'grouped_gangue_silicates',
    'grouped_carbonates',
    'cpy_+50%_exposed_norm',
    'cpy_locked_norm',
    'cpy_associated_norm',
    'cumulative_catalyst_(kg_t)_max',
    'cumulative_catalyst_(kg_t)_slope',
    'catalyst_dose_(mg_l)',
    'ph_mean',
    'orp_(mv)_mean',
]
df_reactors[df_reactors['project_sample_id'] == '007ajettiprojectfile_elephant_pq_rthead'][cols_to_check]

df_model_recCu_catcontrol_projects.columns

pd.DataFrame(df_reactors.describe()).to_excel('/Users/administration/OneDrive - Jetti Resources/Rosetta/NNPyTorch_PYMC_mergedmodel/describe_reactors.xlsx')

folder_path_save = '/Users/administration/OneDrive - Jetti Resources/Rosetta/NNPyTorch_PYMC_mergedmodel'

# drop columns  'reactors_PCA1', 'reactors_PCA2', 'reactorsfit_bias', 'reactorsfit_over' , 'reactorsfit_r2'

# Special treatment to joint 6 and 8inches for project 015
df_model_recCu_catcontrol_projects['project_sample_id'].unique()
len(df_model_recCu_catcontrol_projects['project_sample_id'].unique())
# df_model_recCu_catcontrol_projects['project_sample_id'].replace('015_jetti_project_file_amcf_6in', '015_jetti_project_file_amcf', inplace=True)
# df_model_recCu_catcontrol_projects['project_sample_id'].replace('015_jetti_project_file_amcf_8in', '015_jetti_project_file_amcf', inplace=True)

# Separate UGM2 samples for ROM and Crushed
df_model_recCu_catcontrol_projects.loc[
    df_model_recCu_catcontrol_projects['project_col_id'].str.startswith('jetti_file_elephant_ii_ver_2_ugm_ur'),
    'project_sample_id'
] = 'jetti_file_elephant_ii_ugm2_coarse'
# Duplicate and append rows for '007ajettiprojectfile_elephant_ugm2_rthead'
duplicated_rows = df_reactors[df_reactors['project_sample_id'] == '007ajettiprojectfile_elephant_ugm2_rthead'].copy()
duplicated_rows['project_sample_id'] = '007ajettiprojectfile_elephant_ugm2_rthead_coarse'
df_reactors = pd.concat([df_reactors, duplicated_rows], ignore_index=True)

# Duplicate and append rows for '007jettiprojectfile_elephant'
duplicated_rows = df_reactors[df_reactors['project_sample_id'] == '007jettiprojectfile_elephant'].copy()
duplicated_rows['project_sample_id'] = '007jettiprojectfile_elephant_site'
df_reactors = pd.concat([df_reactors, duplicated_rows], ignore_index=True)

# Duplicate and append rows for '015jettiprojectfile_pv' and '006jettiprojectfile_pvo'
duplicated_rows = df_reactors[df_reactors['project_sample_id'] == '015jettiprojectfile_pv'].copy()
duplicated_rows['project_sample_id'] = '015jettiprojectfile_pv_6in'
df_reactors = pd.concat([df_reactors, duplicated_rows], ignore_index=True)
duplicated_rows['project_sample_id'] = '015jettiprojectfile_pv_8in'
df_reactors = pd.concat([df_reactors, duplicated_rows], ignore_index=True)
duplicated_rows['project_sample_id'] = '006jettiprojectfile_pvo'
df_reactors = pd.concat([df_reactors, duplicated_rows], ignore_index=True)


# REMOVE PVO1 TO AVOID WEIRD BEHAVIOURS
df_model_recCu_catcontrol_projects = df_model_recCu_catcontrol_projects[
    df_model_recCu_catcontrol_projects['project_col_id'] != '006_jetti_project_file_pvo1'
]

# REMOVE WHOLE PVLS BECAUSE IT IS RESIDUES LEACHED???
df_model_recCu_catcontrol_projects = df_model_recCu_catcontrol_projects[
    df_model_recCu_catcontrol_projects['project_sample_id'] != '006_jetti_project_file_pvls'
]


#%% ---------------------------
# Define the PyTorch Model
# ---------------------------


# Configuration variables for easy path customization
CONFIG = {
    # Confidence level thresholds - easily adjustable
    'rmse_threshold': 6.0,      # Maximum acceptable RMSE for a "correct" prediction
    'bias_threshold': 6.0,      # Maximum acceptable absolute bias for a "correct" prediction
    'confidence_target': 90.0,  # Target percentage of correct predictions (90%)
    
    # PyTorch training parameters - easily adjustable
    'pytorch_batch_size': 256,   # Batch size for PyTorch training (2^8)
    'pytorch_epochs': 1000,      # Maximum number of epochs for PyTorch training
    'pytorch_patience': 200,      # Early stopping patience for PyTorch training
    'pytorch_learning_rate': 1e-4,  # Learning rate for PyTorch optimizer
    'pytorch_weight_decay': 1e-4,  # Weight decay for AdamW optimizer
    'pytorch_hidden_dim': 128,   # Hidden dimension for PyTorch neural network
    'pytorch_dropout_rate': 0.3,
    'pytorch_timeout': 300,     # Timeout in seconds for PyTorch training (5 minutes)
    'min_pre_catalyst_points': 9,  # Minimum number of pre-catalyst points to consider a sample valid

    # *** NEW: Adaptive Learning Rate Configuration ***
    'adaptive_lr': {
        'enabled': True,
        'scheduler_type': 'reduce_on_plateau',  # Options: 'reduce_on_plateau', 'cosine_annealing', 'step', 'exponential'
        'base_lr': 1e-4,  # Base learning rate (higher than original 1e-5)
        'min_lr': 1e-6,   # Minimum learning rate
        'max_lr': 1e-3,   # Maximum learning rate for cosine annealing
        
        # ReduceLROnPlateau parameters
        'plateau_factor': 0.5,      # Factor to reduce LR by
        'plateau_patience': 100,    # Epochs to wait before reducing
        'plateau_threshold': 0.01,  # Minimum change to qualify as improvement
        
        # CosineAnnealingWarmRestarts parameters
        'cosine_T_0': 200,          # Initial restart period
        'cosine_T_mult': 2,         # Factor to increase restart period
        
        # StepLR parameters
        'step_size': 500,           # Period of learning rate decay
        'step_gamma': 0.7,          # Multiplicative factor of learning rate decay
    },

    # SHAP analysis parameters
    'shap_sample_size': 20,     # Number of samples to use for SHAP analysis
    'shap_background_size': 50, # Number of samples to use for SHAP background
    
    # Plot parameters
    'max_plots_per_grid': 15,   # Maximum number of plots per grid
    'grid_columns': 2,          # Number of columns in the grid plot
    # 'grid_figsize': (15, 20),   # Figure size for grid plots (width, height)
    
    # Recovery calculation parameters
    'recovery_time_points': 100,  # Number of time points for recovery calculation
    'recovery_max_time': 125,     # Maximum time for recovery calculation

    'column_tests_feature_weighting': {
        'enabled': True,  # Whether to enable feature selection and weighting for column tests
        'weights': { # title and weight
            'leach_duration_days': ['Leach Duration (days)', 0.1], 
            'cumulative_catalyst_addition_kg_t': ['Cumulative Catalyst added (kg/t)', 0.5], 
            'acid_soluble_%': ['Acid Soluble Cu (%norm)', 0.4], 
            'residual_cpy_%': ['Residual Chalcopyrite (%norm)', -0.3],
            # 'cyanide_soluble_%': ['Cyanide Soluble (%norm)', 0.0],
            'cu_seq_a_r_%': ['Residual Chalcopyrite (%)', 0.0], 
            'material_size_p80_in': ['Material Size P80 (in)', 0.0],
            # 'reactors_PCA1': ['Reactors PCA1', 0.0],
            # 'reactors_PCA2': ['Reactors PCA2', 0.0],
            'grouped_copper_sulfides': ['Copper Sulphides (%)', 0.0],
            'grouped_secondary_copper': ['Secondary Copper (%)', 0.0],
            'grouped_acid_generating_sulfides': ['Acid Generating Sulphides (%)', 0.0],
            #1'grouped_gangue_sulfides': ['Gangue Sulphides (%)', 0.0],
            #1'grouped_gangue_silicates': ['Gangue Silicates (%)', 0.0],
            # 'grouped_clays_and_micas': ['Clays and Micas (%)', 0.0],
            # 'grouped_accesory_silicates': ['Accesory Silicates (%)', 0.0],
            # 'grouped_sulfates': ['Sulfates (%)', 0.0],
            'grouped_fe_oxides': ['Fe Oxides (%)', 0.0],
            # 'grouped_accessory_misc': ['Accessory Misc (%)', 0.0],
            # 'grouped_carbonates': ['Carbonates (%)', 0.0],
            # 'grouped_accessory_minerals': ['Accessory Minerals (%)', 0.0],
        },
    },

    'special_feats': {
        'dynamic': ['leach_duration_days', 'cumulative_catalyst_addition_kg_t'],
        'time_feat': ['leach_duration_days'],
        'catalyst_feat': ['cumulative_catalyst_addition_kg_t'],
        'categorical': [],
    },

    # Feature weighting configuration (can be modified by user)
    'reactor_tests_feature_weighting': {
        'enabled': True,        # Whether to enable feature selection and weighting
        'param_weights': { # Be aware that a_param now is (P - c) based on the equation c + (P - c)*(1 - np.exp(-b*t))
            'avg_h2so4_kg_t': -0.3,
            'cu_%': 0.3,
            # 'cu_seq_h2so4_%': 0.5,
            # 'cu_seq_nacn_%': 0.4,
            # 'cu_seq_a_r_%': 0.4,
            'cu_seq_h2so4_norm%': 0.5,
            # 'cu_seq_nacn_norm%': 0.0,
            'cu_seq_a_r_norm%': -0.3,
            'fe_%': -0.1,
            'grouped_copper_sulfides': -0.3,
            'grouped_secondary_copper': 0.4,
            'grouped_acid_generating_sulfides': 0.3,
            # 'grouped_gangue_sulfides': -0.1,
            #1'grouped_gangue_silicates': -0.2,
            # 'grouped_fe_oxides': -0.1,
            #1'grouped_carbonates': -0.3,
            # 'grouped_accessory_minerals': 0.0,
            # 'grouped_other_not_grouped': 0.0,
            #1'cpy_+50%_exposed_norm': 0.2,
            #1'cpy_locked_norm': -0.2,
            #1'cpy_associated_norm': -0.1,
            #1'cumulative_catalyst_(kg_t)_max': 0.1,
            #1'cumulative_catalyst_(kg_t)_slope': 0.1,
            'catalyst_dose_(mg_l)': 0.3, 
            #1'ph_mean': 0.1,
            #1'orp_(mv)_mean': 0.1,
        },
    },

    'feature_mapping_reactor_cols' : {  # Reactor feature: Column feature (or None if not shared)
        'avg_h2so4_kg_t': None,
        'cu_%': None, 
        'cu_seq_h2so4_norm%': 'acid_soluble_%',
        'cu_seq_a_r_norm%': 'residual_cpy_%',
        'fe_%': None,
        'grouped_copper_sulfides': 'grouped_copper_sulfides',
        'grouped_secondary_copper': 'grouped_secondary_copper',
        'grouped_acid_generating_sulfides': 'grouped_acid_generating_sulfides',
        'grouped_fe_oxides': 'grouped_fe_oxides',
        'catalyst_dose_(mg_l)': None,  # Reactor-only
        },

    # Select projects to run
    'selected_reactors': {
        'enabled': True,
        '003jettiprojectfile_amcf': ['RT_1', 'RT_3'], # matched in 015, 003 and 006
        '006jettiprojectfile_pvo': ['RT_1', 'RT_3'], # matched in 015, 003 and 006
        '007ajettiprojectfile_elephant_pq_rthead': ['RT_47', 'RT_48', 'RT_51'],
        '007ajettiprojectfile_elephant_ugm2_rthead': ['RT_65', 'RT_66', 'RT_67', 'RT_68'],
        '007ajettiprojectfile_elephant_ugm2_rthead_coarse': ['RT_65', 'RT_66', 'RT_67', 'RT_68'],
        '007bjettiprojectfile_tiger': ['RT_1', 'RT_2', 'RT_4'],
        '007jettiprojectfile_elephant': ['RT_5R', 'RT_8'],
        '007jettiprojectfile_elephant_site': ['RT_5R', 'RT_8'],
        '007jettiprojectfile_leopard': ['RT_2R', 'RT_4R'],
        '007jettiprojectfile_rtm2': ['RT_33', 'RT_34', 'RT_36'],
        # '007jettiprojectfile_toquepala_antigua': [],
        '007jettiprojectfile_toquepala_fresca': ['RT_21', 'RT_22', 'RT_24'],
        '007jettiprojectfile_zaldivar': ['RT_29', 'RT_30', 'RT_32'],
        '011jettiprojectfile_rm': ['RT_21', 'RT_22', 'RT_24'],
        '012jettiprojectfile_incremento': ['RT_E', 'RT_F'],
        # '012jettiprojectfile_kino': [],
        '012jettiprojectfile_quebalix': ['RT_C', 'RT_D'],
        '013jettiprojectfile_oz': ['RTO_1', 'RTO_4'],
        '014jettiprojectfile_bag': ['RTB_7', 'RTB_8'],
        '014jettiprojectfile_kmb': ['RTK_7', 'RTK_8'],
        '015jettiprojectfile_pv_6in': ['RT_1', 'RT_3'], # matched in 015, 003 and 006
        '015jettiprojectfile_pv_8in': ['RT_1', 'RT_3'], # matched in 015, 003 and 006
        '017jettiprojectfile_ea': ['RTEA_1', 'RTEA_2'],
        # '020jettiprojectfile_har': [],
        '020jettiprojectfile_hyp': ['RT_1', 'RT_2'],
        '020jettiprojectfile_sup': ['RT_19', 'RT_20'],
        '022jettiprojectfile_stingray': ['RT_1', 'RT_2'],
        '023jettiprojectfile_ot10': ['RT_8R', 'RT_10'],
        '023jettiprojectfile_ot9': ['RT_1', 'RT_3'],
        '024jettiprojectfile_cpy': ['RT_1', 'RT_3'],
        '025jettiprojectfile_chalcopyrite': ['RT_17', 'RT_18', 'RT_19', 'RT_20', 'RT_21', 'RT_22'],
        '025jettiprojectfile_oxide': ['RT_9', 'RT_12'],
        '025jettiprojectfile_secondary': ['RT_1', 'RT_4'],
        '026jettiprojectfile_carrizalillo': ['RT_10', 'RT_12'],
        '026jettiprojectfile_oxides': ['RT_25', 'RT_26', 'RT_28'],
        '026jettiprojectfile_primarysulfide': ['RT_1', 'RT_2', 'RT_4'],
        '026jettiprojectfile_secondarysulfide': ['RT_17', 'RT_18', 'RT_20'],
        '028jettiprojectfile_andesite': ['RT_1', 'RT_2', 'RT_4'],
        '028jettiprojectfile_composite': ['RT_11', 'RT_12', 'RT_14'],
        '028jettiprojectfile_monzonite': ['RT_7', 'RT_8', 'RT_10'],
        # '030jettiprojectfile_ss': [], # no reactors for this sample
        '030jettiprojectfile_cpy': ['RT_1R', 'RT_4R'], # missing project 030
        '031jettiprojectfile_sample': ['RT_1', 'RT_2', 'RT_4'], # missing all reactors
        # 'jettifile_elephant_pq': [],
        # 'jettifile_elephant_ugm': [],
        # 'jettiprojectfile_elephantscl': [],
        # 'jettiprojectfile_leopard': [],
        # 'jettiprojectfile_tiger_m1': [],
        # 'jettiprojectfile_tiger_m2': [],
        # 'jettiprojectfile_tiger_m3': [],
        # 'jettiprojectfile_toquepala_antigua': [],
        # 'jettiprojectfile_toquepala_fresca': [],
        # 'jettiprojectfile_zaldivar': [],
    },

    'bayesian': {
        'prior_sd_a': 10.0, # SD for a1/a2 priors
        'prior_sd_b': 0.1, # SD for b1/b2 priors
        'num_posterior_samples': 100, # for predictions/unvertainties
    },

    # Use validation data from test to append to training
    'use_val_data_days': 0,  # Number of days from test set to append to training set (0 to disable)
    'generate_plots': False,
}

def parse_array_str(s):
    """Parse string like '[0. 0. ...]' to np.array of floats, or return array if already ndarray."""
    if isinstance(s, str):
        s = s.strip('[]').strip()
        return np.array([float(x) for x in s.split() if x.strip()])
    elif isinstance(s, np.ndarray):
        return s
    else:
        raise ValueError(f"Unsupported type for parsing: {type(s)}")

def get_excluded_ids(test_sample_id):
    paired_groups = [
        {'015_jetti_project_file_amcf_6in', '015_jetti_project_file_amcf_8in', '003_jetti_project_file_amcf_head', '006_jetti_project_file_pvo'},
        {'jetti_file_elephant_ii_ugm2', 'jetti_file_elephant_ii_ugm2_coarse'}
    ]
    for group in paired_groups:
        if test_sample_id in group:
            return group
    return {test_sample_id}

def find_best_config_for_fold(X_tensor, y_tensor, time_tensor, catalyst_tensor, tt_tensor,
                              sample_ids, sample_col_ids, num_cols, device, folder_path_save,
                              train_mask, test_sample_id):
    hyperparameter_space = {
        # apparently, after a few tries, best combination: 
        # [256, 0.3, 'kaiminng', 'reduce_on_plateau'] or
        # [128, 0.2, 'kaiming', 'cosine_annealing']
        'pytorch_hidden_dim': [128], # 64 removed because none was optimal
        'pytorch_dropout_rate': [0.33],
        'init_mode': ['kaiming'], #  'xavier' and 'normal' removed as only 2 samples used it
        'scheduler_type': ['reduce_on_plateau'], #, 'cosine_annealing']
    }

    combinations = list(itertools.product(
        hyperparameter_space['pytorch_hidden_dim'],
        hyperparameter_space['pytorch_dropout_rate'],
        hyperparameter_space['init_mode'],
        hyperparameter_space['scheduler_type']
    ))

    # MODIFIED: Get excluded IDs and update train_mask to exclude paired samples
    excluded_ids = get_excluded_ids(test_sample_id)
    train_mask = [sid not in excluded_ids for sid in sample_ids]

    train_indices = [j for j, mask in enumerate(train_mask) if mask]
    unique_train_sids = list(set(sample_ids[j] for j in train_indices))
    num_samples = len(unique_train_sids)
    num_val = min(3, num_samples)

    results = []

    kf = KFold(n_splits=min(3, num_samples), shuffle=True, random_state=42)
    for hidden_dim, dropout_rate, init_mode, scheduler_type in combinations:
        config_mod = copy.deepcopy(CONFIG)
        config_mod['pytorch_hidden_dim'] = hidden_dim
        config_mod['pytorch_dropout_rate'] = dropout_rate
        config_mod['init_mode'] = init_mode
        config_mod['adaptive_lr']['scheduler_type'] = scheduler_type
        config_mod['skip_hypersearch'] = True
        config_mod['generate_plots'] = False  # Ensure no plots during hypersearch

        print(f"Testing config for fold {test_sample_id}: HD={hidden_dim}, DO={dropout_rate}, INIT={init_mode}, SCH={scheduler_type}")
        
        val_losses = []
        for train_idx, val_idx in kf.split(unique_train_sids):
            val_sid = unique_train_sids[val_idx[0]]
            fold_data = (0, val_sid)
            fold_data_with_params = (
                fold_data, num_cols, config_mod, device, folder_path_save,
                X_tensor, y_tensor, time_tensor, catalyst_tensor, tt_tensor,
                sample_ids, sample_col_ids, config_mod.get('use_val_data_days', 0)
            )
            result = train_single_fold_reactor_scaling(fold_data_with_params)
            val_losses.append(result['val_loss'])

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        results.append({
            'pytorch_hidden_dim': hidden_dim,
            'pytorch_dropout_rate': dropout_rate,
            'init_mode': init_mode,
            'scheduler_type': scheduler_type,
            'val_loss': avg_val_loss
        })

    best = min(results, key=lambda x: x['val_loss'])
    print(f"✅ Best configuration for fold {test_sample_id}: {best}")
    
    return {
        'test_sample_id': test_sample_id,
        'pytorch_hidden_dim': best['pytorch_hidden_dim'],
        'pytorch_dropout_rate': best['pytorch_dropout_rate'],
        'init_mode': best['init_mode'],
        'scheduler_type': best['scheduler_type'],
        'val_loss': best['val_loss']
    }


def plot_diagnostics(y_train, y_train_pred, y_test, y_test_pred, config=None):
    if not config.get('generate_plots', False):
        return
    y_train = y_train.copy()
    y_train_pred = y_train_pred.copy()
    y_test = y_test.copy()
    y_test_pred = y_test_pred.copy()
    
    # Ensure inputs are numpy arrays and squeeze them to 1D
    y_train = np.asarray(y_train).squeeze()
    y_train_pred = np.asarray(y_train_pred).squeeze()
    y_test = np.asarray(y_test).squeeze()
    y_test_pred = np.asarray(y_test_pred).squeeze()

    # Calculate residuals
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred

    # Check for NaN or Inf values in residuals
    valid_idx = np.isfinite(residuals_train)
    if not valid_idx.all():
        print("Warning: Non-finite values detected in residuals_train. Removing invalid values.")
        residuals_train = residuals_train[valid_idx]
        y_train = y_train[valid_idx]  # Filter y_train to match residuals_train length
        y_train_pred = y_train_pred[valid_idx]  # Ensure consistency

    # Handle empty arrays after filtering
    if y_train.size == 0 or y_train_pred.size == 0:
        print("Error: y_train or y_train_pred is empty after filtering. Skipping diagnostic plot.")
        return None

    plt.close('all')
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 8))

    # Predicted vs Actual (Train Set)
    axes[0, 0].scatter(y_train, y_train_pred, edgecolors=(0, 0, 0), alpha=0.4)
    if y_train.size > 0:
        axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'b--', lw=2)
    axes[0, 0].set_title('Predicted vs Actual (Train Set)', fontsize=10, fontweight="bold")
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].tick_params(labelsize=7)

    # Predicted vs Actual (Test Set)
    axes[0, 1].scatter(y_test, y_test_pred, edgecolors=(0, 0, 0), alpha=0.4)
    if y_test.size > 0:
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', lw=2)
    axes[0, 1].set_title('Predicted vs Actual (Test Set)', fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].tick_params(labelsize=7)

    # Residuals vs ID
    if residuals_train.size > 0:
        x_values = list(range(len(residuals_train)))  # Ensuring matching lengths
        axes[2, 1].scatter(x_values, residuals_train, edgecolors=(0, 0, 0), alpha=0.4)
        axes[2, 1].axhline(y=0, linestyle='--', color='black', lw=2)
        axes[2, 1].set_title('Residues of the Model (Train Set)', fontsize=10, fontweight="bold")
        axes[2, 1].set_xlabel('ID')
        axes[2, 1].set_ylabel('Residue')
        axes[2, 1].tick_params(labelsize=7)

    # Residuals Distribution
    if residuals_train.size > 0:
        sns.histplot(
            data=residuals_train,
            stat="density",
            kde=True,
            line_kws={'linewidth': 1},
            color="firebrick",
            alpha=0.3,
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Residues Distribution of the Model (Train Set)', fontsize=10, fontweight="bold")
        axes[1, 0].set_xlabel("Residue")
        axes[1, 0].tick_params(labelsize=7)

    # Q-Q Plot
    if residuals_train.size > 0:
        sm.qqplot(residuals_train, fit=True, line='q', ax=axes[1, 1], color='firebrick', alpha=0.4, lw=2)
        axes[1, 1].set_title('Q-Q Plot for Residues of the Model (Train Set)', fontsize=10, fontweight="bold")
        axes[1, 1].tick_params(labelsize=7)
    else:
        print("Warning: No valid residuals to plot in Q-Q Plot.")

    # Residuals vs Predicted Values
    if residuals_train.size > 0:
        axes[2, 0].scatter(y_train_pred, residuals_train, edgecolors=(0, 0, 0), alpha=0.4)
        axes[2, 0].axhline(y=0, linestyle='--', color='black', lw=2)
        axes[2, 0].set_title('Residues of the Model vs Predicted Values (Train Set)', fontsize=10, fontweight="bold")
        axes[2, 0].set_xlabel('Predicted')
        axes[2, 0].set_ylabel('Residue')
        axes[2, 0].tick_params(labelsize=7)

    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

    return fig

def calculate_confidence_level(y_true, y_pred, confidence_threshold=CONFIG.get('confidence_target', 90), max_bias=CONFIG.get('bias_threshold', 6.0)):
    """
    Calculate the confidence level of predictions based on the difference between true and predicted values.
    Confidence is defined as the percentage of predictions that fall within a specified bias threshold.
    It counts the data points predicted within the bias threshold and divides by the total number of predictions.
    
    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        confidence_threshold (int): Target confidence level in percentage.
        max_bias (float): Maximum allowed bias for confidence calculation.
        
    Returns:
        float: Confidence level as a percentage.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0  # Avoid division by zero

    # Calculate absolute differences
    differences = np.abs(y_true - y_pred)

    # Count predictions within the bias threshold
    within_bias = np.sum(differences <= max_bias)

    # Calculate confidence level as a percentage
    confidence_level = (within_bias / len(y_true)) * 100

    # Ensure confidence level does not exceed 100%
    return min(confidence_level, 100.0)


#%% ---------------------------
# Step 1: Dataset preparation
# ---------------------------
def compile_training_datasets(df_reactors, config):
    def extract_features_and_weights(feature_config):
        feats = {}
        weights = {}
        for param in ['param_weights']:
            feats[param] = [k for k, v in feature_config.get(param, {}).items()]
            weights[param] = {k: v for k, v in feature_config.get(param, {}).items()}
        return feats, weights

    def filter_reactors_by_selected(config, df):
        if not config.get('selected_reactors', {}).get('enabled', False):
            return df
        selected = config['selected_reactors']
        valid_rows = []
        for proj_id, reactor_ids in selected.items():
            if proj_id == 'enabled':
                continue
            subset = df[(df['project_sample_id'] == proj_id) & (df['start_cell'].isin(reactor_ids))]
            valid_rows.append(subset)
        return pd.concat(valid_rows, ignore_index=True) if valid_rows else pd.DataFrame(columns=df.columns)

    # Apply data filters
    if config.get('data_filters', {}).get('enabled', False):
        for col, rule in config.get('data_filters', {}).items():
            if isinstance(rule, list):
                df_reactors[col] = df_reactors[col].astype(str).str.strip().fillna('Control')  # Ensure string type for comparison
                df_reactors = df_reactors[df_reactors[col].isin(rule)]
            elif isinstance(rule, dict):
                df_reactors[col] = pd.to_numeric(df_reactors[col], errors='coerce')  # Ensure numeric type for comparison
                if 'max' in rule:
                    df_reactors = df_reactors[df_reactors[col] <= rule['max']]
                if 'min' in rule:
                    df_reactors = df_reactors[df_reactors[col] >= rule['min']]

    feature_sets, weight_sets = extract_features_and_weights(config.get('reactor_tests_feature_weighting', {}))
    all_features = list(feature_sets['param_weights'])
    cols_to_keep = ['project_sample_id', 'start_cell', 'a1_param', 'b1_param', 'a2_param', 'b2_param'] + all_features
    reactor_filtered = df_reactors[cols_to_keep].copy()
    reactor_filtered.dropna(inplace=True)
    reactor_filtered = filter_reactors_by_selected(config, reactor_filtered)

    # Compute min/max bounds for true parameters
    param_bounds = {
        'a1_min': reactor_filtered['a1_param'].min(),
        'a1_max': reactor_filtered['a1_param'].max(),
        'b1_min': reactor_filtered['b1_param'].min(),
        'b1_max': reactor_filtered['b1_param'].max(),
        'a2_min': reactor_filtered['a2_param'].min(),
        'a2_max': reactor_filtered['a2_param'].max(),
        'b2_min': reactor_filtered['b2_param'].min(),
        'b2_max': reactor_filtered['b2_param'].max()
    }

    return reactor_filtered, feature_sets, weight_sets, param_bounds


#%% ---------------------------
# Step 2: Normalize features and create datasets
# ---------------------------

def copper_recovery_curve_4param(t, a1, b1, a2, b2):
    """
    Compute copper recovery curve using four parameters.
    
    Args:
        t (torch.Tensor): Time points (shape: [n_time_points])
        a1, b1, a2, b2 (torch.Tensor): Parameters (shape: [batch_size] or scalar)
    
    Returns:
        torch.Tensor: Recovery values (shape: [batch_size, n_time_points])
    """
    # Ensure inputs are tensors and have correct shapes
    t = t.view(1, -1)  # Shape: [1, n_time_points]
    a1 = a1.view(-1, 1)  # Shape: [batch_size, 1]
    b1 = b1.view(-1, 1)
    a2 = a2.view(-1, 1)
    b2 = b2.view(-1, 1)
    
    # Compute recovery: a1 * (1 - exp(-b1 * t)) + a2 * (1 - exp(-b2 * t))
    term1 = a1 * (1 - torch.exp(-b1 * t))  # Shape: [batch_size, n_time_points]
    term2 = a2 * (1 - torch.exp(-b2 * t))
    recovery = term1 + term2
    
    return recovery


def get_reactor_priors(sample_id, status, reactor_df):
    """
    Get LF prior means from matching reactor row.
    Match control/catalyzed via catalyst_dose_(mg_l).
    Clean sample_id to match reactor format (e.g., remove '_head').
    """
    catalyst_dose_col = 'catalyst_dose_(mg_l)'
    if status == 'no_catalyst':
        dose_filter = reactor_df[catalyst_dose_col] == 0
    else:
        dose_filter = reactor_df[catalyst_dose_col] > 0
    
    # Clean sample_id (adjust based on your naming conventions, e.g., from CSVs)
    cleaned_id = sample_id.replace('_head', '').replace('_rthead', '').replace('_coarse', '').replace('_site', '').replace('_6in', '').replace('_8in', '')
    
    matching = reactor_df[(reactor_df['project_sample_id'] == cleaned_id) & dose_filter]
    
    if not matching.empty:
        row = matching.iloc[0]  # Assume one match; take first if multiples (e.g., RT_1/RT_3)
        return {
            'a1': row['a1_param'],
            'b1': row['b1_param'],
            'a2': row['a2_param'],
            'b2': row['b2_param']
        }
    else:
        print(f"No reactor match for {sample_id} ({status}); using vague priors.")
        return {'a1': 50.0, 'b1': 0.01, 'a2': 50.0, 'b2': 0.01}  # Vague defaults (adjust based on domain)


def bayesian_fit_two_phase(row, reactor_df, config, save_dir):  # reactor_df unused
    sample_id = row['project_sample_id_reactormatch']
    col_id = row['project_col_id']
    status = row['catalyst_status']
    
    times = parse_array_str(row['leach_duration_days'])
    recoveries = parse_array_str(row['cu_recovery_%'])
    transition = row['transition_time']
    
    priors = {
        'a1': row['reactor_a1_true'],
        'b1': row['reactor_b1_true'],
        'a2': row['reactor_a2_true'],
        'b2': row['reactor_b2_true']
    }
    
    with pm.Model() as model:
        a1 = pm.Normal('a1', mu=priors['a1'], sigma=config['bayesian']['prior_sd_a'])
        b1 = pm.Normal('b1', mu=priors['b1'], sigma=config['bayesian']['prior_sd_b'])
        a2 = pm.Normal('a2', mu=priors['a2'], sigma=config['bayesian']['prior_sd_a'])
        b2 = pm.Normal('b2', mu=priors['b2'], sigma=config['bayesian']['prior_sd_b'])
        
        sigma = pm.HalfNormal('sigma', sigma=5.0)
        
        pre_mask = times < transition
        post_mask = times >= transition
        
        pre_rec = a1 * (1 - pt.exp(-b1 * times))
        pre_at_trans = a1 * (1 - pt.exp(-b1 * transition))
        post_rec = pre_at_trans + a2 * (1 - pt.exp(-b2 * (times - transition)))
        
        mu = pt.switch(pre_mask, pre_rec, post_rec)
        
        pm.Normal('obs', mu=mu, sigma=sigma, observed=recoveries)
        
        trace = pm.sample(
            draws=1000, tune=1000, chains=4, cores=4,
            target_accept=0.95,
            return_inferencedata=True
        )
    
    trace_path = os.path.join(save_dir, f'trace_{sample_id}_{col_id}.nc')
    az.to_netcdf(trace, trace_path)
    
    summary = az.summary(trace)
    print(f"Summary for {sample_id}_{col_id}:\n{summary}")
    
    return trace, summary, model  # Now return model too


#%% ---------------------------
# Step 3: Merge the columns and reactors dataset by id
# ---------------------------
project_dict = {
    '003_jetti_project_file_amcf_head': ['003jettiprojectfile_amcf'],
    '006_jetti_project_file_pvls': [],  # No matching key (residue?)
    '006_jetti_project_file_pvo': ['006jettiprojectfile_pvo'],  # duplicated from 015 and 003
    '011_jetti_project_file_rm': ['011jettiprojectfile_rm'],  # No matching key
    # '011_jetti_project_file_rm_crushed': ['011jettiprojectfile_rm'],
    '012_jetti_project_file_incremento': ['012jettiprojectfile_incremento'],
    '012_jetti_project_file_quebalix': ['012jettiprojectfile_quebalix'],    
    '013_jetti_project_file_combined': ['013jettiprojectfile_oz'],
    '014_jetti_project_file_bag': ['014jettiprojectfile_bag'],
    '014_jetti_project_file_kmb': ['014jettiprojectfile_kmb'],
    '015_jetti_project_file_amcf_6in': ['015jettiprojectfile_pv_6in'],
    '015_jetti_project_file_amcf_8in': ['015jettiprojectfile_pv_8in'],
    '017_jetti_project_file_ea_mill_feed_combined': ['017jettiprojectfile_ea'],
    '020_jetti_project_file_hypogene_supergene_hypogene_master_composite': ['020jettiprojectfile_hyp'],  # No matching key
    '020_jetti_project_file_hypogene_supergene_super': ['020jettiprojectfile_sup'],
    '022_jetti_project_file_stingray_1': ['022jettiprojectfile_stingray'],
    '023_jetti_project_file_ot_9': ['023jettiprojectfile_ot9'],
    '023_jetti_project_file_ot_10': ['023jettiprojectfile_ot10'],
    '024_jetti_project_file_024cv_cpy': ['024jettiprojectfile_cpy'],
    '026_jetti_project_file_sample_1_primary_sulfide': ['026jettiprojectfile_primarysulfide'],
    '026_jetti_project_file_sample_2_carrizalillo': ['026jettiprojectfile_carrizalillo'],
    '026_jetti_project_file_sample_3_secondary_sulfide': ['026jettiprojectfile_secondarysulfide'],
    '026_jetti_project_file_sample_4_mixed_material': ['026jettiprojectfile_oxides'],
    '028_jetti_project_file_andesite': ['028jettiprojectfile_andesite'],
    '028_jetti_project_file_composite': ['028jettiprojectfile_composite'],
    '028_jetti_project_file_monzonite': ['028jettiprojectfile_monzonite'],
    '030_jetti_project_file_cpy': ['030jettiprojectfile_cpy'],  # No matching key
    '031_jetti_project_file_sample': ['031jettiprojectfile_sample'],  # No matching key
    'jetti_file_elephant_ii_pq': ['007ajettiprojectfile_elephant_pq_rthead'],
    'jetti_file_elephant_ii_ugm2': ['007ajettiprojectfile_elephant_ugm2_rthead'],
    'jetti_file_elephant_ii_ugm2_coarse': ['007ajettiprojectfile_elephant_ugm2_rthead_coarse'],
    'jetti_project_file_elephant_site': ['007jettiprojectfile_elephant_site'],
    'jetti_project_file_elephant_scl_sample_escondida': ['007jettiprojectfile_elephant'],
    'jetti_project_file_toquepala_scl_sample_fresca': ['007jettiprojectfile_toquepala_fresca'],
    'jetti_project_file_zaldivar_scl_sample_zaldivar': ['007jettiprojectfile_zaldivar'],
    'jetti_project_file_tiger_scl_sample': ['007bjettiprojectfile_tiger'],
    'jetti_project_file_leopard_scl_sample': ['007jettiprojectfile_leopard'],
}

# replace project_sample_id from reactors database to match the column dataset
replacement_dict = {k: v[0] for k, v in project_dict.items() if v}  # Only include entries with non-empty lists
df_model_recCu_catcontrol_projects['project_sample_id_reactormatch'] = df_model_recCu_catcontrol_projects['project_sample_id'].replace(replacement_dict)
df_model_recCu_control_projects['project_sample_id_reactormatch'] = df_model_recCu_control_projects['project_sample_id'].replace(replacement_dict)
df_model_recCu_catalyzed_projects['project_sample_id_reactormatch'] = df_model_recCu_catalyzed_projects['project_sample_id'].replace(replacement_dict)


def filter_column_dataset_by_config(df_columns, config):
    df_columns = df_columns.copy()
    col_config = config.get('column_tests_feature_weighting', {})
    if not col_config.get('enabled', False):
        return df_columns.copy()

    raw_weights = col_config.get('weights', {})
    selected_features = list(raw_weights.keys())  # include all listed features regardless of weight

    # Only check that the features exist in the DataFrame
    missing_cols = [col for col in selected_features if col not in df_columns.columns]
    if missing_cols:
        print(f"Advertencia: Las siguientes columnas definidas en CONFIG no están en el dataset de columnas: {missing_cols}")

    df_filtered = df_columns[['project_sample_id_reactormatch', 'project_col_id'] + selected_features + [col for col in df_columns.columns if col.startswith('cu_recovery')]].copy()
    return df_filtered

def embed_reactor_knowledge_to_columns(df_reactors, df_columns):
    df_columns = df_columns.copy()
    df_reactors = df_reactors.copy()
    df_encoded = []

    for sample_id in df_columns['project_sample_id_reactormatch'].unique():
        # sample_id = '015jettiprojectfile_pv_6in'
        reactor_samples = df_reactors[df_reactors['project_sample_id'] == sample_id]
        for catalyst_flag in ['Y', 'N']:
            matching_reactors = reactor_samples[(reactor_samples['catalyst_dose_(mg_l)'] > 0) if catalyst_flag == 'Y' else (reactor_samples['catalyst_dose_(mg_l)'] == 0)]
            if matching_reactors.empty:
                continue

            a1_true = matching_reactors['a1_param'].values.mean()
            b1_true = matching_reactors['b1_param'].values.mean()
            a2_true = matching_reactors['a2_param'].values.mean()
            b2_true = matching_reactors['b2_param'].values.mean()
            
            # Generate true and predicted curves
            t = np.linspace(0, CONFIG['recovery_max_time'], CONFIG['recovery_time_points'])
            true_curve = a1_true * (1 - np.exp(-b1_true * t)) + a2_true * (1 - np.exp(-b2_true * t))

            df_encoded.append({
                'project_sample_id_reactormatch': sample_id,
                'catalyzed_y_n': catalyst_flag,
                'reactor_a1_true': a1_true,
                'reactor_b1_true': b1_true,
                'reactor_a2_true': a2_true,
                'reactor_b2_true': b2_true,
            })

    df_columns2 = []
    for col in df_columns['project_col_id'].unique():
        df_col = df_columns[df_columns['project_col_id'] == col]
        if df_col.empty:
            continue
        df_col.loc[:, 'catalyzed_y_n'] = 'Y' if df_col['cumulative_catalyst_addition_kg_t'].sum() > 0 else 'N'
        df_columns2.append(df_col)
    df_columns2 = pd.concat(df_columns2)

    df_repr = pd.DataFrame(df_encoded)
    df_columns_aug = df_columns2.merge(df_repr, on=['project_sample_id_reactormatch', 'catalyzed_y_n'], how='left')
    df_columns_aug.to_csv(os.path.join(folder_path_save, 'plots', 'augmented_columns_nofilter_withReactors.csv'), index=False)
    df_columns_aug.drop(columns=['catalyzed_y_n'], inplace=True, errors='ignore')
    missing_ids = df_columns_aug[df_columns_aug.isnull().any(axis=1)]['project_sample_id_reactormatch'].unique()
    print(f"Se omitirán los siguientes project_sample_id por datos faltantes: {missing_ids}")
    df_columns_aug.dropna(inplace=True)
    print("Paso 6 completado: embeddings agregados a dataset de columnas")
    return df_columns_aug


#%% ---------------------------
# Step 4: Prepare column train data
# ---------------------------

def process_arrays_by_weekly_intervals(arr, leach_days, col_name):
    """
    Process a single array of time-varying data by averaging values within strict 7-day batches
    (0-7, 7-14, 14-21, ...), taking the maximum of averaged values, skipping empty batches,
    and interpolating to estimate the maximum for each batch using subsequent batches if needed.
    
    Parameters:
    - arr: Array of values for the time-varying column (e.g., cu_recovery_%)
    - leach_days: Array of corresponding leach_duration_days
    - col_name: Name of the column (to determine if it's leach_duration_days)
    
    Returns:
    - Array of maximum values for each non-empty weekly batch with no NaNs
    """
    if len(arr) == 0 or len(leach_days) == 0:
        return np.array([])
    
    # Define weekly batch endpoints (7, 14, 21, ...)
    max_days = leach_days.max()
    weekly_days = np.arange(7, max_days + 7, 7)  # [7, 14, 21, ...]
    
    processed = []
    valid_weekly_days = []
    
    for end_day in weekly_days:
        # Define batch range: [start_day, end_day)
        start_day = end_day - 7
        # Find values within the batch [start_day, end_day)
        idx = np.where((leach_days >= start_day) & (leach_days < end_day))[0]
        if len(idx) > 0:  # Only process non-empty batches
            if col_name == 'leach_duration_days':
                processed.append(end_day)  # Use batch endpoint for leach_duration_days
            else:
                # Average values within the batch
                avg_value = np.mean(arr[idx])
                processed.append(avg_value)
            valid_weekly_days.append(end_day)
    
    if not processed:  # If no non-empty batches, return empty array
        return np.array([])
    
    processed = np.array(processed)
    
    # Interpolate NaNs and estimate maximum for each batch
    if np.any(np.isnan(processed)):
        x = np.arange(len(processed))
        valid = ~np.isnan(processed)
        if valid.sum() > 1:  # Need at least 2 points for interpolation
            # Interpolate NaN values
            interpolated = np.interp(x, x[valid], processed[valid])
            # Estimate maximum by considering interpolated values and subsequent batches
            processed = np.maximum.accumulate(interpolated)  # Forward-fill max values
        else:
            # Fallback: use mean of non-NaN values or 0 if all NaN
            processed = np.where(np.isnan(processed), 
                               np.mean(processed[valid]) if valid.any() else np.nan, 
                               processed)
    
    return processed


def prepare_column_train_data(df, config, output_type='original', fill_noncat_averages=False):
    """
    Prepare column train data with option to select original or split grouping.
    
    Parameters:
    - df: Input DataFrame
    - config: Configuration dictionary
    - output_type: 'original' for project_col_id grouping or 'averaged' for project_sample_id with catalyst split
    
    Returns:
    - Tuple containing (X_tensor, Y_tensor, time_tensor, catalyst_tensor, transition_time_tensor,
                       sample_ids, sample_col_ids, feature_weights, numeric_cols, scaler_X, out_df)
    """
    if output_type not in ['original', 'averaged']:
        raise ValueError("output_type must be 'original' or 'averaged'")

    df = df.copy()
    target_col = 'cu_recovery_%'
    id_col = 'project_sample_id_reactormatch'
    col_idx = 'project_col_id'
    df = df[df[target_col] > 1.0].copy()

    # Define time-varying columns
    time_varying_cols = ['leach_duration_days', 'cu_recovery_%', 'cumulative_catalyst_addition_kg_t']
    
    # Filter rows with NaNs in key columns
    df_filtered = df.dropna(subset=[target_col, 'leach_duration_days', 'cumulative_catalyst_addition_kg_t'])

    # Original grouping by project_col_id (unchanged)
    grouped_data = []
    for col_id, group in df_filtered.groupby(col_idx):
        row = {}
        for col in df.columns:
            if col in time_varying_cols:
                row[col] = group[col].values.astype(float)
            else:
                row[col] = group[col].iloc[0]
        grouped_data.append(row)
    out_df = pd.DataFrame(grouped_data, columns=df.columns)

    # Split grouping by project_sample_id with catalyst split
    grouped_data_averaged = []
    for sample_id, group in df_filtered.groupby(id_col):
        # Split into catalyst = 0 and catalyst > 0
        no_catalyst = group[group['cumulative_catalyst_addition_kg_t'] == 0]
        with_catalyst = group[group['cumulative_catalyst_addition_kg_t'] > 0]

        # Process no-catalyst data
        if not no_catalyst.empty:
            row_no = {'catalyst_status': 'no_catalyst'}
            for col in df.columns:
                if col in time_varying_cols:
                    row_no[col] = no_catalyst[col].values.astype(float)
                else:
                    if col.endswith('_true'):
                        row_no[col] = np.median(no_catalyst[col].values.astype(float))
                    else:
                        row_no[col] = no_catalyst[col].iloc[0]
            row_no[id_col] = sample_id
            grouped_data_averaged.append(row_no)
        
        # Process with-catalyst data
        if not with_catalyst.empty:
            row_with = {'catalyst_status': 'with_catalyst'}
            for col in df.columns:
                if col in time_varying_cols:
                    row_with[col] = with_catalyst[col].values.astype(float)
                else:
                    if col.endswith('_true'):
                        row_with[col] = np.median(with_catalyst[col].values.astype(float))
                    else:
                        row_with[col] = with_catalyst[col].iloc[0]
            row_with[id_col] = sample_id
            grouped_data_averaged.append(row_with)
    
    out_df_averaged = pd.DataFrame(grouped_data_averaged, columns=[*df.columns, 'catalyst_status'])
    
    # Rename project_col_id in split DataFrame
    out_df_averaged[col_idx] = np.where(out_df_averaged['catalyst_status'] == 'no_catalyst', 'Control', 'Catalyzed')
    
    # Process time-varying columns for split grouping by weekly batches
    final_grouped_data_averaged = []
    for sample_id, group in out_df_averaged.groupby(id_col):
        no_cat_rows = group[group['catalyst_status'] == 'no_catalyst']
        with_cat_rows = group[group['catalyst_status'] == 'with_catalyst']
        
        # Define weekly intervals based on all data for this sample
        max_days = 0
        if not no_cat_rows.empty:
            max_days = max([row['leach_duration_days'].max() for _, row in no_cat_rows.iterrows() if len(row['leach_duration_days']) > 0])
        if not with_cat_rows.empty:
            max_days = max(max_days, max([row['leach_duration_days'].max() for _, row in with_cat_rows.iterrows() if len(row['leach_duration_days']) > 0]))
        
        # Process no-catalyst data
        no_cat_processed = {}
        if not no_cat_rows.empty:
            row_no = {'catalyst_status': 'no_catalyst', id_col: sample_id}
            for col in time_varying_cols:
                no_cat_processed[col] = process_arrays_by_weekly_intervals(
                    no_cat_rows[col].iloc[0], no_cat_rows['leach_duration_days'].iloc[0], col
                )
            # Skip if no non-empty batches
            if len(no_cat_processed['leach_duration_days']) > 0:
                row_no.update(no_cat_processed)
                for col in df.columns:
                    if col not in time_varying_cols:
                        if col.endswith('_true'):
                            row_no[col] = np.median(no_cat_rows[col].values.astype(float))
                        else:
                            row_no[col] = no_cat_rows[col].iloc[0]
                final_grouped_data_averaged.append(row_no)
        
        # Process with-catalyst data, prepending no-catalyst averages
        if not with_cat_rows.empty:
            row_with = {'catalyst_status': 'with_catalyst', id_col: sample_id}
            processed_cols = {}
            for col in time_varying_cols:
                with_cat_data = with_cat_rows[col].iloc[0]
                with_cat_days = with_cat_rows['leach_duration_days'].iloc[0]
                # Process catalyzed data
                processed_with = process_arrays_by_weekly_intervals(with_cat_data, with_cat_days, col)
                # Prepend no-catalyst data if available
                if not no_cat_rows.empty and len(no_cat_processed['leach_duration_days']) > 0 and fill_noncat_averages:
                    no_cat_data = no_cat_processed[col]
                    no_cat_days = no_cat_processed['leach_duration_days']
                    # Ensure no overlap in leach_duration_days
                    with_cat_start = with_cat_days.min() if len(with_cat_days) > 0 else np.inf
                    no_cat_idx = np.where(no_cat_days < with_cat_start)[0] if len(no_cat_days) > 0 else []
                    if len(no_cat_idx) > 0:
                        # Concatenate no-catalyst and with-catalyst data
                        if col == 'leach_duration_days':
                            processed_cols[col] = np.concatenate([no_cat_days[no_cat_idx], processed_with])
                        elif col == 'cumulative_catalyst_addition_kg_t':
                            # For catalyst, use zeros for no-catalyst portion
                            zeros = np.zeros(len(no_cat_idx))
                            processed_cols[col] = np.concatenate([zeros, processed_with])
                        else:
                            processed_cols[col] = np.concatenate([no_cat_data[no_cat_idx], processed_with])
                    else:
                        processed_cols[col] = processed_with
                else:
                    processed_cols[col] = processed_with
            # Skip if no non-empty batches
            if len(processed_cols['leach_duration_days']) > 0:
                row_with.update(processed_cols)
                for col in df.columns:
                    if col not in time_varying_cols:
                        if col.endswith('_true'):
                            row_with[col] = np.median(with_cat_rows[col].values.astype(float))
                        else:
                            row_with[col] = with_cat_rows[col].iloc[0] if not with_cat_rows[col].empty else no_cat_rows[col].iloc[0]
                final_grouped_data_averaged.append(row_with)

    # Skip if no valid rows
    if not final_grouped_data_averaged:
        print("Warning: No non-empty batches found for any samples in split grouping.")
        return None

    # Create out_df_averaged with processed values
    out_df_averaged = pd.DataFrame(final_grouped_data_averaged, columns=[*df.columns, 'catalyst_status'])

    # Compute transition_time for original grouping (unchanged)
    out_df['transition_time'] = 0.0
    for idx, row in out_df.iterrows():
        catalyst_days = row['leach_duration_days'][row['cumulative_catalyst_addition_kg_t'] > 0]
        out_df.at[idx, 'transition_time'] = catalyst_days.min() if catalyst_days.size > 0 else row['leach_duration_days'].max()

    # Compute transition_time for split grouping
    out_df_averaged['transition_time'] = 0.0
    for idx, row in out_df_averaged.iterrows():
        sample_id = row[id_col]
        sample_data = df_filtered[df_filtered[id_col] == sample_id]
        catalyst_days = sample_data['leach_duration_days'][sample_data['cumulative_catalyst_addition_kg_t'] > 0]
        if row['catalyst_status'] == 'with_catalyst':
            # First day where catalyst > 0, or last day if no catalyst
            out_df_averaged.at[idx, 'transition_time'] = (
                catalyst_days.min() if catalyst_days.size > 0 else 
                sample_data['leach_duration_days'].max() if sample_data['leach_duration_days'].size > 0 else 0.0
            )
        else:
            # For no_catalyst, use the last day of the processed no_catalyst data
            out_df_averaged.at[idx, 'transition_time'] = (
                row['leach_duration_days'][-1] if len(row['leach_duration_days']) > 0 else 0.0
            )

    # Prepare features for training (exclude non-feature columns)
    drop_cols = ['project_sample_id_reactormatch', 'project_sample_id', 'project_col_id', 
                 'leach_duration_days', 'cu_recovery_%', 'cumulative_catalyst_addition_kg_t', 'catalyst_status']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    numeric_cols = [c for c in feature_cols if np.issubdtype(out_df[c].dtype, np.number)]
    
    # Get feature weights from config
    raw_weights = config.get('column_tests_feature_weighting', {}).get('weights', {})
    feature_weights = {k: v[1] for k, v in raw_weights.items() if k in numeric_cols}
    
    # Check for NaNs in original DataFrame
    if out_df[numeric_cols].isnull().any().any():
        nan_positions = out_df[numeric_cols].isnull().any(axis=1)
        print("Warning: NaN values found in original DataFrame rows:")
        print(out_df[nan_positions][[id_col, col_idx, *numeric_cols]])
        out_df = out_df.dropna(subset=numeric_cols)

    # Check for NaNs in split DataFrame numeric columns
    if out_df_averaged[numeric_cols].isnull().any().any():
        nan_positions = out_df_averaged[numeric_cols].isnull().any(axis=1)
        print("Warning: NaN values found in split DataFrame rows:")
        print(out_df_averaged[nan_positions][[id_col, col_idx, *numeric_cols]])
        out_df_averaged = out_df_averaged.dropna(subset=numeric_cols)

    # Select output based on output_type
    if output_type == 'original':
        # Scale numerical features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(out_df[numeric_cols])
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        # Convert time-varying columns and other outputs to tensors
        Y_tensor = [torch.tensor(row['cu_recovery_%'], dtype=torch.float32).to(device) for _, row in out_df.iterrows()]
        time_tensor = [torch.tensor(row['leach_duration_days'], dtype=torch.float32).to(device) for _, row in out_df.iterrows()]
        catalyst_tensor = [torch.tensor(row['cumulative_catalyst_addition_kg_t'], dtype=torch.float32).to(device) for _, row in out_df.iterrows()]
        transition_time_tensor = torch.tensor(out_df['transition_time'].values, dtype=torch.float32).to(device)
        sample_ids = out_df[id_col].values
        sample_col_ids = out_df[col_idx].values

        # Debug information
        print("Original grouping - Transition time min/max:", out_df['transition_time'].min(), out_df['transition_time'].max())
        print("Original grouping - X_tensor shape:", X_tensor.shape)
        print("Original grouping - Y_tensor length:", len(Y_tensor))
        print("Original grouping - Recovery values min and max:", min([t.min().item() for t in Y_tensor]), max([t.max().item() for t in Y_tensor]))

        return (
            X_tensor,
            Y_tensor,
            time_tensor,
            catalyst_tensor,
            transition_time_tensor,
            sample_ids,
            sample_col_ids,
            feature_weights,
            numeric_cols,
            scaler_X,
            out_df
        )
    else:  # output_type == 'averaged'
        # Scale numerical features
        scaler_X_averaged = StandardScaler()
        X_scaled_averaged = scaler_X_averaged.fit_transform(out_df_averaged[numeric_cols])
        X_tensor_averaged = torch.tensor(X_scaled_averaged, dtype=torch.float32).to(device)

        # Convert time-varying columns and other outputs to tensors
        Y_tensor_averaged = [torch.tensor(row['cu_recovery_%'], dtype=torch.float32).to(device) for _, row in out_df_averaged.iterrows()]
        time_tensor_averaged = [torch.tensor(row['leach_duration_days'], dtype=torch.float32).to(device) for _, row in out_df_averaged.iterrows()]
        catalyst_tensor_averaged = [torch.tensor(row['cumulative_catalyst_addition_kg_t'], dtype=torch.float32).to(device) for _, row in out_df_averaged.iterrows()]
        transition_time_tensor_averaged = torch.tensor(out_df_averaged['transition_time'].values, dtype=torch.float32).to(device)
        sample_ids_averaged = out_df_averaged[id_col].values
        sample_col_ids_averaged = out_df_averaged[col_idx].values

        # Debug information
        print("Split grouping - Transition time min/max:", out_df_averaged['transition_time'].min(), out_df_averaged['transition_time'].max())
        print("Split grouping - X_tensor_averaged shape:", X_tensor_averaged.shape)
        print("Split grouping - Y_tensor_averaged length:", len(Y_tensor_averaged))
        print("Split grouping - Recovery values min and max:", min([t.min().item() for t in Y_tensor_averaged]), max([t.max().item() for t in Y_tensor_averaged]))

        return (
            X_tensor_averaged,
            Y_tensor_averaged,
            time_tensor_averaged,
            catalyst_tensor_averaged,
            transition_time_tensor_averaged,
            sample_ids_averaged,
            sample_col_ids_averaged,
            feature_weights,
            numeric_cols,
            scaler_X_averaged,
            out_df_averaged
        )
    

#%% ---------------------------
# REACTOR SCALING ENSEMBLE MODEL
# ---------------------------

class EnsembleModels:
    """Ensemble model for reactor scaling with uncertainty quantification"""
    
    def __init__(self, model_states, val_losses, total_features, config, device, best_configs):
        self.device = device
        self.total_features = total_features
        self.config = config
        self.best_configs = best_configs
        self.models, self.weights = self._create_filtered_ensemble(
            model_states, val_losses, config
        )
     
    def _create_filtered_ensemble(self, model_states, val_losses, config):
        """Create filtered ensemble based on validation losses"""
        median_loss = np.median(val_losses)
        # threshold = median_loss * 1.5  # original was 1.5
        threshold = np.percentile(val_losses, 95) # approx 95% of the models are included
        
        models = []
        weights = []
        
        # Use the overall best hidden_dim and dropout_rate from final_config
        hidden_dim = config.get('pytorch_hidden_dim', 128)
        dropout_rate = config.get('pytorch_dropout_rate', 0.3)
        
        for idx, (model_state, val_loss) in enumerate(zip(model_states, val_losses)):
            if val_loss <= threshold:
                model = AdaptiveTwoPhaseRecoveryModel(
                    total_features=self.total_features,
                    hidden_dim=hidden_dim,  # Use overall best from final_config
                    dropout_rate=dropout_rate,  # Use overall best from final_config
                    init_mode=config.get('init_mode', 'kaiming')  # Use overall best init_mode
                ).to(self.device)
                model.load_state_dict(model_state)
                model.eval()
                weights.append(1.0 / (val_loss + 1e-6))
                models.append(model)
        
        weights = np.array(weights)
        if len(weights) > 0:
            weights /= weights.sum()
        else:
            weights = np.array([1.0])  # Fallback if no models are selected
        
        return models, weights
    
    def predict_with_params_and_uncertainty(self, X, catalyst, transition_time, time_points, sample_ids=None):
        """
        Make ensemble prediction with uncertainty quantification
        """
        all_model_predictions = []
        all_model_params = []
        
        for model in self.models:
            with torch.no_grad():
                params = model(X, catalyst, sample_ids)
                all_model_params.append(params.cpu().numpy())
                
                recovery = generate_two_phase_recovery(
                    time_points, catalyst, transition_time, params
                )
                all_model_predictions.append(recovery.cpu().numpy())
        
        all_model_predictions = np.array(all_model_predictions)
        all_model_params = np.array(all_model_params)
        
        # Weighted ensemble prediction
        weighted_pred = np.average(all_model_predictions, axis=0, weights=self.weights)
        weighted_params = np.average(all_model_params, axis=0, weights=self.weights)
        uncertainty = np.std(all_model_predictions, axis=0)
        
        return weighted_pred, uncertainty, weighted_params
    
    def get_ensemble_info(self):
        """Get ensemble information"""
        return {
            'num_models': len(self.models),
            'weights': self.weights,
            'total_features': self.total_features,
        }

#%% ---------------------------
# REACTOR SCALING MODEL
# ---------------------------

class AdaptiveTwoPhaseRecoveryModel(nn.Module):
    """
    Enhanced Two-Phase Recovery Model with adaptive architecture, biphasic curves, and continuity enforcement.
    """
    def __init__(self, total_features, hidden_dim=128, dropout_rate=0.3, init_mode='kaiming'):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.init_mode = init_mode
        
        # Network for non-catalyst parameters (a1, b1, a2, b2)
        self.base_network = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, 4),
        )
        
        # Network for catalyst parameters (a3, b3, a4, b4, transition_rate_cat)
        self.catalyst_network = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, 4),
        )
        
        self._initialize_weights(self.init_mode)

    def _initialize_weights(self, init_mode='kaiming'):
        for module in [self.base_network, self.catalyst_network]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    if init_mode == 'kaiming':
                        nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                    elif init_mode == 'xavier':
                        nn.init.xavier_uniform_(layer.weight)
                    elif init_mode == 'normal':
                        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                    else:
                        nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')  # fallback
                    nn.init.constant_(layer.bias, 0.0)
        
        # Initialize final layer biases to reasonable parameter values
        with torch.no_grad():
            self.base_network[-1].bias[0] = 0.0   # a1
            self.base_network[-1].bias[1] = -2.7  # b1
            self.base_network[-1].bias[2] = 0.0   # a2
            self.base_network[-1].bias[3] = -2.7  # b2
            
            self.catalyst_network[-1].bias[0] = 0.0   # a3
            self.catalyst_network[-1].bias[1] = -2.7  # b3
            self.catalyst_network[-1].bias[2] = 0.0   # a4
            self.catalyst_network[-1].bias[3] = -2.7  # b4

    def forward(self, x, catalyst, sample_ids=None):
        batch_size = x.size(0)
        
        # Initialize parameter tensor
        params = torch.zeros(batch_size, 8, device=x.device)
        
        # Process base parameters (a1, b1, a2, b2) with consistency constraint
        if sample_ids is not None:
            # Ensure physical consistency: same base parameters for same sample_id
            unique_sample_ids = list(set(sample_ids))
            sample_id_to_base_params = {}
            
            for unique_id in unique_sample_ids:
                # Find indices for this sample_id
                sample_indices = [i for i, sid in enumerate(sample_ids) if sid == unique_id]
                
                if sample_indices:
                    # Use the first occurrence to compute base parameters for this sample_id
                    first_idx = sample_indices[0]
                    
                    # Process base parameters using the first sample's features
                    base_raw = self.base_network(x[first_idx:first_idx+1])
                    
                    a1 = 10.0 + 30.0 * torch.sigmoid(base_raw[:, 0])
                    b1 = 0.001 + 0.1 * torch.sigmoid(base_raw[:, 1])
                    a2 = 5.0 + 20.0 * torch.sigmoid(base_raw[:, 2])
                    b2 = 0.0001 + 0.1 * torch.sigmoid(base_raw[:, 3])
                    
                    '''
                    # ensure a1 >= a2
                    mask = a1 < a2
                    a1_new = torch.where(mask, a2, a1)
                    a2_new = torch.where(mask, a1, a2)
                    a1, a2 = a1_new, a2_new   

                    # ensure b1 > b2
                    mask_b = b1 <= b2
                    b1_new = torch.where(mask_b, b2, b1)
                    b2_new = torch.where(mask_b, b1, b2)
                    b1, b2 = b1_new, b2_new
                    '''

                    # Ensure total asymptote doesn't exceed 70
                    total_asymptote = a1 + a2
                    mask_a = total_asymptote > 70.0
                    scale = torch.where(mask_a & (total_asymptote > 0), 
                                       70.0 / total_asymptote.clamp(min=1.0), 
                                       torch.tensor(1.0, device=x.device))
                    a1, a2 = a1 * scale, a2 * scale
                    
                    # Store base parameters for this sample_id
                    sample_id_to_base_params[unique_id] = torch.stack([a1, b1, a2, b2], dim=1).squeeze(0)
                    
                    # Apply the same base parameters to all samples with this sample_id
                    for idx in sample_indices:
                        params[idx, :4] = sample_id_to_base_params[unique_id]
        else:
            # Fallback to original behavior if sample_ids not provided
            base_raw = self.base_network(x)
            
            a1 = 10.0 + 30.0 * torch.sigmoid(base_raw[:, 0])
            b1 = 0.001 + 0.1 * torch.sigmoid(base_raw[:, 1])
            a2 = 5.0 + 20.0 * torch.sigmoid(base_raw[:, 2])
            b2 = 0.0001 + 0.1 * torch.sigmoid(base_raw[:, 3])
            '''
            # ensure a1 >= a2
            mask = a1 < a2
            a1_new = torch.where(mask, a2, a1)
            a2_new = torch.where(mask, a1, a2)
            a1, a2 = a1_new, a2_new            
            
            #ensure b1 > b2
            mask_b = b1 <= b2
            b1_new = torch.where(mask_b, b2, b1)
            b2_new = torch.where(mask_b, b1, b2)
            b1, b2 = b1_new, b2_new
            '''
            # Ensure total asymptote doesn't exceed 70
            total_asymptote = a1 + a2
            mask_a = total_asymptote > 70.0
            scale = torch.where(mask_a & (total_asymptote > 0), 
                               70.0 / total_asymptote.clamp(min=1.0), 
                               torch.tensor(1.0, device=x.device))
            a1, a2 = a1 * scale, a2 * scale
            
            params[:, :4] = torch.stack([a1, b1, a2, b2], dim=1)
        
        # Process catalyst parameters if catalyst is present
        has_catalyst = torch.any(catalyst > 0, dim=1)
        if has_catalyst.any():
            idx = has_catalyst.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                # Get base parameters for catalyzed samples
                a1_r, b1_r, a2_r, b2_r = [p.squeeze() for p in params[idx, :4].split(1, dim=1)]
                
                # Process catalyst parameters
                cat_raw = self.catalyst_network(x[idx])
                
                # Apply catalyst parameter constraints (ensure positive values for additional recovery)
                a3 = 1.0 + 14.0 * torch.sigmoid(cat_raw[:, 0])
                b3 = 0.0001 + 0.003 * torch.sigmoid(cat_raw[:, 1])
                a4 = 1.0 + 14.0 * torch.sigmoid(cat_raw[:, 2])
                b4 = 0.0001 + 0.003 * torch.sigmoid(cat_raw[:, 3])

                # Ensure total asymptote doesn't exceed 70
                total_asymptote_cat = a1_r + a2_r + a3 + a4
                mask_a = total_asymptote_cat > 70.0
                scale = torch.where(mask_a & (total_asymptote_cat > 0), 
                                70.0 / total_asymptote_cat.clamp(min=1.0), 
                                torch.tensor(1.0, device=x.device))
                a3, a4 = a3 * scale, a4 * scale
                
                params[idx, 4:] = torch.stack([a3, b3, a4, b4], dim=1)
        else:
            params[:, 4:] = np.nan
        
        return params
    

def generate_two_phase_recovery(time, catalyst, transition_time, params):
    """
    Improved two-phase recovery generation with proper physical constraints:
    1. Control curve: a1*(1-exp(-b1*t)) + a2*(1-exp(-b2*t))
    2. Catalyzed curve: Control curve + a3*(1-exp(-b3*(t-t_trans))) + a4*(1-exp(-b4*(t-t_trans))) for t >= t_trans
    This ensures catalyzed curve is always >= control curve and they match at transition time.
    The catalyst effect is calculated as catalyst_effect = catalyst / (catalyst + 1) and used as a tensor.
    """
    # Ensure inputs are on the correct device
    time = time.to(params.device)
    catalyst = catalyst.to(params.device)
    
    # Calculate catalyst effect tensor - this is the key enhancement
    catalyst_effect = catalyst / (catalyst + 1)
    
    # Extract base parameters (same for both control and catalyzed)
    a1 = params[:, 0].unsqueeze(1)
    b1 = params[:, 1].unsqueeze(1)
    a2 = params[:, 2].unsqueeze(1)
    b2 = params[:, 3].unsqueeze(1)
    
    # Compute base recovery with numerical stability (this is the control curve)
    exp_term1 = torch.exp(-b1 * time)
    exp_term1 = torch.clamp(exp_term1, min=1e-8, max=1.0)
    exp_term2 = torch.exp(-b2 * time)
    exp_term2 = torch.clamp(exp_term2, min=1e-8, max=1.0)
    recovery_control = a1 * (1 - exp_term1) + a2 * (1 - exp_term2)
    
    # Start with control recovery
    recovery = recovery_control.clone()
    
    has_catalyst = torch.any(catalyst > 0).item()
    transition_i = transition_time.squeeze()
    
    # Apply catalyst enhancement if catalyst exists and params has catalyst parameters
    if has_catalyst and params.shape[1] > 4 and torch.any(~torch.isnan(params[:, 4:])):
        a3 = params[:, 4].unsqueeze(1)
        b3 = params[:, 5].unsqueeze(1)
        a4 = params[:, 6].unsqueeze(1)
        b4 = params[:, 7].unsqueeze(1)
        
        # Only apply catalyst effect after transition time
        has_catalyst_points = (catalyst > 0) & (time >= transition_i)
        
        if has_catalyst_points.any():
            # Time shifted to start from transition point
            time_shifted = torch.clamp(time - transition_i, min=0.0)
            
            # Use catalyst_effect tensor in the exponential terms for enhanced modeling
            exp_term3 = torch.exp(-torch.abs(b3) * time_shifted * (1 + catalyst_effect))
            exp_term3 = torch.clamp(exp_term3, min=1e-8, max=1.0)
            exp_term4 = torch.exp(-torch.abs(b4) * time_shifted * (1 + catalyst_effect))
            exp_term4 = torch.clamp(exp_term4, min=1e-8, max=1.0)
            
            # Additional recovery from catalyst with catalyst_effect scaling
            additional_recovery = (torch.abs(a3) * catalyst_effect * (1 - exp_term3) + 
                                 torch.abs(a4) * catalyst_effect * (1 - exp_term4))
            
            # Apply catalyst enhancement only where catalyst is present and after transition
            catalyst_enhancement = torch.where(has_catalyst_points, additional_recovery, torch.zeros_like(additional_recovery))
            recovery = recovery_control + catalyst_enhancement
    
    # Apply reasonable bounds to recovery
    recovery = torch.clamp(recovery, min=0.0, max=100.0)
    return recovery


def compute_sample_weights(time_steps, device):
    """
    Compute sample weights giving more importance to later time points.
    """
    weights = []
    num_max_days_standard = 1600
    full_days = np.linspace(0, num_max_days_standard - 1, num_max_days_standard)
    full_weight_curve = np.linspace(1.0, 10.0, num_max_days_standard)
    
    for time_array in time_steps:
        if isinstance(time_array, torch.Tensor):
            time_array = time_array.cpu().numpy()
        sample_weights = np.interp(time_array, full_days, full_weight_curve)
        weights.append(torch.tensor(sample_weights, dtype=torch.float32, device=device))
    
    return weights

def train_single_fold_reactor_scaling(fold_data_with_params):
    """Train a single fold for reactor scaling"""
    fold_data, num_cols, config, device, save_dir, X_tensor, y_tensor, time_tensor, catalyst_tensor, tt_tensor, sample_ids, sample_col_ids, use_val_data_days = fold_data_with_params
    
    i, test_sample_id = fold_data
    print(f"Training fold {i+1}: Testing on {test_sample_id}")
    
    # Training parameters
    epochs = config.get('pytorch_epochs', 1000)
    learning_rate = config.get('pytorch_learning_rate', 1e-4)
    patience = config.get('pytorch_patience', 200)

    excluded_ids = get_excluded_ids(test_sample_id)
    train_mask = [sid not in excluded_ids for sid in sample_ids]
    test_mask = [sid == test_sample_id for sid in sample_ids]  # Test set remains only the selected ID

    X_train = X_tensor[train_mask]
    y_train = [y_tensor[j] for j, mask in enumerate(train_mask) if mask]
    time_train = [time_tensor[j] for j, mask in enumerate(train_mask) if mask]
    catalyst_train = [catalyst_tensor[j] for j, mask in enumerate(train_mask) if mask]
    tt_train = [tt_tensor[j] for j, mask in enumerate(train_mask) if mask]
    
    X_test = X_tensor[test_mask]
    y_test = [y_tensor[j] for j, mask in enumerate(test_mask) if mask]
    time_test = [time_tensor[j] for j, mask in enumerate(test_mask) if mask]
    catalyst_test = [catalyst_tensor[j] for j, mask in enumerate(test_mask) if mask]
    tt_test = [tt_tensor[j] for j, mask in enumerate(test_mask) if mask]
    
    train_sample_ids = [sample_ids[k] for k, mask in enumerate(train_mask) if mask]

    # Append test data to training set if use_val_data_days is True
    if use_val_data_days > 0 and any(test_mask):
        test_indices = [j for j, mask in enumerate(test_mask) if mask]
        for j in test_indices:
            t_arr = time_tensor[j].cpu().numpy()
            is_catalyzed = torch.any(catalyst_tensor[j] > 0).item()
            time_threshold = (tt_tensor[j].item() + use_val_data_days) if is_catalyzed else use_val_data_days
            mask_x_days = np.where(t_arr < time_threshold)[0]
            if len(mask_x_days) > 0:
                print(f"Appending test sample {test_sample_id} (Catalyzed: {is_catalyzed}, Time threshold: {time_threshold} days)")
                X_train = torch.cat([X_train, X_tensor[j:j+1]], dim=0)
                tt_train.append(tt_tensor[j])
                y_train.append(y_tensor[j][mask_x_days])
                time_train.append(time_tensor[j][mask_x_days])
                catalyst_train.append(catalyst_tensor[j][mask_x_days])
                train_sample_ids.append(test_sample_id)

    # Initialize model
    fold_model = AdaptiveTwoPhaseRecoveryModel(
        total_features=len(num_cols),
        hidden_dim=config.get('pytorch_hidden_dim', 128),
        dropout_rate=config.get('pytorch_dropout_rate', 0.3),
        init_mode=config.get('init_mode', 'kaiming')
    ).to(device)
    
    optimizer = optim.AdamW(fold_model.parameters(), lr=learning_rate, weight_decay=config.get('pytorch_weight_decay', 1e-4))
    scheduler = ReduceLROnPlateau(optimizer, 
                                  mode='min', 
                                  factor=config['adaptive_lr']['plateau_factor'], 
                                  patience=config['pytorch_patience']//2, 
                                  min_lr=config['adaptive_lr']['min_lr'])

    # Compute sample weights (assuming compute_sample_weights is defined)
    weights_train = compute_sample_weights(time_train, device)
    
    # Training loop
    best_loss = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    
    for epoch in range(epochs):
        fold_model.train()
        total_loss = 0.0
        
        for j in range(len(X_train)):
            optimizer.zero_grad()
            x_sample = X_train[j:j+1]
            catalyst_sample = catalyst_train[j].unsqueeze(0)
            time_sample = time_train[j].unsqueeze(0)
            tt_sample = tt_train[j]
            batch_sample_ids = [train_sample_ids[j]]
            
            params = fold_model(x_sample, catalyst_sample, batch_sample_ids)
            predicted_recovery = generate_two_phase_recovery(
                time_sample, catalyst_sample, tt_sample, params
            )
            
            target_recovery = y_train[j].unsqueeze(0)
            weights_sample = weights_train[j].unsqueeze(0)
            mse_loss = nn.MSELoss(reduction='none')
            per_element_loss = mse_loss(predicted_recovery, target_recovery)
            weighted_loss = weights_sample * per_element_loss
            loss = weighted_loss.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fold_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(X_train)
        train_losses.append(avg_loss)
        
        # Validation
        fold_model.eval()
        val_losses = []
        val_samples = len(X_train) # min(5, len(X_train))
        with torch.no_grad():
            for j in range(val_samples):
                x_sample = X_train[j:j+1]
                catalyst_sample = catalyst_train[j].unsqueeze(0)
                time_sample = time_train[j].unsqueeze(0)
                tt_sample = tt_train[j]
                batch_sample_ids = [train_sample_ids[j]]
                
                params = fold_model(x_sample, catalyst_sample, batch_sample_ids)
                predicted_recovery = generate_two_phase_recovery(
                    time_sample, catalyst_sample, tt_sample, params
                )
                target_recovery = y_train[j].unsqueeze(0)
                weights_sample = weights_train[j].unsqueeze(0)
                per_element_loss_val = mse_loss(predicted_recovery, target_recovery)
                weighted_loss_val = weights_sample * per_element_loss_val
                val_losses.append(weighted_loss_val.mean().item())
        
        if val_losses:
            val_losses.sort(reverse=True)  # highest losses first
            num_top = max(1, len(val_losses) // 2) # select the 50% worst losses
            avg_val_loss = sum(val_losses[:num_top]) / num_top
        else:
            avg_val_loss = float('inf')

        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = fold_model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Plotting and saving only if generate_plots is True
    if config.get('generate_plots', False):
        # Plot training curves with learning rate
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300)
        ax1.plot(train_losses, label='Training Loss', color='blue')
        if len(train_losses) > 0:
            val_loss_curve = [loss * 1.1 for loss in train_losses[::10]]
            val_epochs = list(range(0, len(train_losses), 10))[:len(val_loss_curve)]
            ax1.plot(val_epochs, val_loss_curve, label='Validation Loss', color='orange')
        
        ax1.set_title(f'Training vs. Validation Loss - {test_sample_id}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.grid(True)
        
        learning_rates = [learning_rate * (config['adaptive_lr']['plateau_factor'] ** (epoch // (config['pytorch_patience']//2))) for epoch in range(len(train_losses))]
        ax2.plot(learning_rates, label='Learning Rate', color='green')
        ax2.set_title(f'Learning Rate Schedule - {test_sample_id}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f'adaptive_loss_curve_{test_sample_id}_withoutReactors.png'), dpi=300)
        plt.close()
        print(f"Saved adaptive loss curve for {test_sample_id}")

    # Load best model and evaluate
    fold_model.load_state_dict(best_model_state)
    fold_model.eval()
    
    # Collect training data for diagnostics
    train_y_true = []
    train_y_pred = []
    
    with torch.no_grad():
        for j in range(len(X_train)):
            x_sample = X_train[j:j+1]
            catalyst_sample = catalyst_train[j].unsqueeze(0)
            time_sample = time_train[j].unsqueeze(0)
            tt_sample = tt_train[j]
            target_recovery = y_train[j]
            batch_sample_ids = [train_sample_ids[j]]
            
            params = fold_model(x_sample, catalyst_sample, batch_sample_ids)
            predicted_recovery = generate_two_phase_recovery(
                time_sample, catalyst_sample, tt_sample, params
            )
            
            train_y_true.extend(target_recovery.cpu().numpy())
            train_y_pred.extend(predicted_recovery.squeeze().cpu().numpy())
    
    # Evaluate on test set
    fold_results = []
    all_y_true = []
    all_y_pred = []
    
    test_sample_ids = [sample_ids[k] for k, mask in enumerate(test_mask) if mask]
    unique_test_sample_ids = list(set(test_sample_ids))
    sample_id_params = {}
    
    with torch.no_grad():
        for unique_sample_id in unique_test_sample_ids:
            sample_indices = [j for j, sid in enumerate(test_sample_ids) if sid == unique_sample_id]
            if not sample_indices:
                continue
            
            control_idx = None
            catalyzed_idx = None
            for idx in sample_indices:
                catalyst_sample = catalyst_test[idx]
                if torch.any(catalyst_sample > 0):
                    catalyzed_idx = idx
                else:
                    control_idx = idx
            
            base_idx = control_idx if control_idx is not None else sample_indices[0]
            x_sample = X_test[base_idx:base_idx+1]
            catalyst_sample = catalyst_test[base_idx].unsqueeze(0)
            batch_sample_ids = [unique_sample_id]
            base_params = fold_model(x_sample, catalyst_sample, batch_sample_ids)
            
            if catalyzed_idx is not None:
                x_cat_sample = X_test[catalyzed_idx:catalyzed_idx+1]
                catalyst_cat_sample = catalyst_test[catalyzed_idx].unsqueeze(0)
                batch_cat_sample_ids = [unique_sample_id]
                cat_params = fold_model(x_cat_sample, catalyst_cat_sample, batch_cat_sample_ids)
                combined_params = base_params[0].clone()
                combined_params[4:] = cat_params[0, 4:]
                sample_id_params[unique_sample_id] = combined_params
            else:
                sample_id_params[unique_sample_id] = base_params[0]
        
        for j, unique_sample_id in enumerate(test_sample_ids):
            x_sample = X_test[j:j+1]
            catalyst_sample = catalyst_test[j].unsqueeze(0)
            time_sample = time_test[j].unsqueeze(0)
            tt_sample = tt_test[j]
            target_recovery = y_test[j]
            col_id = sample_col_ids[test_mask][j]
            sample_params = sample_id_params[unique_sample_id].unsqueeze(0)
            
            predicted_recovery = generate_two_phase_recovery(
                time_sample, catalyst_sample, tt_sample, sample_params
            )
            
            all_y_true.extend(target_recovery.cpu().numpy())
            all_y_pred.extend(predicted_recovery.squeeze().cpu().numpy())
            
            rmse = torch.sqrt(torch.mean((predicted_recovery.squeeze() - target_recovery) ** 2)).item()
            r2 = 1 - torch.sum((predicted_recovery.squeeze() - target_recovery) ** 2) / \
                 torch.sum((target_recovery - target_recovery.mean()) ** 2)
            r2 = r2.item()
            bias = torch.mean(predicted_recovery.squeeze() - target_recovery).item()
            uncertainty = torch.std(predicted_recovery.squeeze() - target_recovery).item()
            confidence_level = calculate_confidence_level(
                target_recovery.cpu().numpy(), 
                predicted_recovery.squeeze().cpu().numpy()
            )
            
            fold_results.append({
                'sample_id': unique_sample_id,
                'sample_col_id': col_id,
                'rmse': rmse,
                'r2': r2,
                'bias': bias,
                'confidence_level': confidence_level,
                'uncertainty': uncertainty,
                'a1_pred': sample_params[0, 0].item(),
                'b1_pred': sample_params[0, 1].item(),
                'a2_pred': sample_params[0, 2].item(),
                'b2_pred': sample_params[0, 3].item(),
                'a3_pred': sample_params[0, 4].item() if torch.any(catalyst_sample > 0) else np.nan,
                'b3_pred': sample_params[0, 5].item() if torch.any(catalyst_sample > 0) else np.nan,
                'a4_pred': sample_params[0, 6].item() if torch.any(catalyst_sample > 0) else np.nan,
                'b4_pred': sample_params[0, 7].item() if torch.any(catalyst_sample > 0) else np.nan,
                'transition_time': tt_sample.item() if isinstance(tt_sample, torch.Tensor) else tt_sample,
            })
    
    # Plot validation recovery curves only if generate_plots is True
    if config.get('generate_plots', False):
        plt.figure(figsize=(12, 8))
        colors = {True: 'darkorange', False: 'royalblue'}
        
        with torch.no_grad():
            for j, unique_sample_id in enumerate(test_sample_ids):
                x_sample = X_test[j:j+1]
                catalyst_sample = catalyst_test[j].unsqueeze(0)
                time_sample = time_test[j].unsqueeze(0)
                tt_sample = tt_test[j]
                target_recovery = y_test[j]
                col_id = sample_col_ids[test_mask][j]
                sample_params = sample_id_params[unique_sample_id].unsqueeze(0)
                
                predicted_recovery = generate_two_phase_recovery(
                    time_sample, catalyst_sample, tt_sample, sample_params
                )
                
                t_np = time_sample.squeeze().cpu().numpy()
                y_true_np = target_recovery.cpu().numpy()
                y_pred_np = predicted_recovery.squeeze().cpu().numpy()
                tt_np = tt_sample.cpu().numpy() if isinstance(tt_sample, torch.Tensor) else tt_sample
                
                rmse = np.sqrt(np.mean((y_pred_np - y_true_np) ** 2))
                bias = np.mean(y_pred_np - y_true_np)
                r2 = 1 - np.sum((y_pred_np - y_true_np) ** 2) / np.sum((y_true_np - y_true_np.mean()) ** 2)
                cl = calculate_confidence_level(y_true_np, y_pred_np)
                
                is_catalyzed = torch.any(catalyst_sample > 0).item()
                
                plt.plot(t_np, y_true_np, 'o', label=f'Actual ({col_id})', alpha=0.3, 
                        markeredgecolor='none', color=colors[is_catalyzed])
                plt.plot(t_np, y_pred_np, '-', 
                        label=f'Predicted ({col_id})\nRMSE: {rmse:.2f}, Bias: {bias:.2f}\nR²: {r2:.2f}, CL: {cl:.0f}%', 
                        color=colors[is_catalyzed])
                
                if is_catalyzed:
                    plt.vlines(x=tt_np, ymin=0, ymax=80, color=colors[is_catalyzed], 
                              linestyle='--', alpha=0.7, label=f'Transition Time ({col_id})')
        
        plt.title(f"Validation Recovery Curves - {test_sample_id}\nVal Loss: {best_val_loss:.4f}")
        plt.xlabel("Leach Duration (Days)")
        plt.ylabel("Cu Recovery (%)")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.ylim(0, 80)
        plt.xlim(0, int(np.ceil(max(tt_tensor)/100)*100))
        plt.tight_layout()
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f"val_columns_{test_sample_id}_withoutReactors.png"), dpi=300)
        plt.close()
        print(f"Saved validation plot for {test_sample_id}")
        
        # Diagnostic plots
        if len(train_y_true) > 0 and len(all_y_true) > 0:
            train_y_true = np.array(train_y_true)
            train_y_pred = np.array(train_y_pred)
            test_y_true = np.array(all_y_true)
            test_y_pred = np.array(all_y_pred)

            fig = plot_diagnostics(train_y_true, train_y_pred, test_y_true, test_y_pred, config)
            if fig is not None:
                plt.suptitle(f"Residues diagnosis for Cu Recovery {test_sample_id}", fontsize=12, fontweight="bold")
                fig.savefig(os.path.join(save_dir, 'plots', f'diagnostics_{test_sample_id}_withoutReactors.png'), dpi=300)
                plt.close(fig)
                print(f"Saved diagnostic plot for {test_sample_id}")
    
    return {
        'model_state': best_model_state,
        'results': fold_results,
        'sample_id': test_sample_id,
        'train_losses': train_losses,
        'val_loss': best_val_loss,
        'config': config
    }

# Define process_fold at the top level
def process_fold(fold_data, X_tensor, y_tensor, time_tensor, catalyst_tensor, 
                 tt_tensor, sample_ids, sample_col_ids, num_cols, config, device, save_dir, 
                 use_val_data_days):
    fold_idx, test_sample_id = fold_data
    print(f"Processing fold {fold_idx + 1}/{len(set(sample_ids))}: {test_sample_id}")
    
    if config is None:
        raise ValueError(f"Config is None in process_fold for fold {fold_idx}")

    excluded_ids = get_excluded_ids(test_sample_id)
    train_mask = [sid not in excluded_ids for sid in sample_ids]
    
    # Find best config for this fold
    if not config.get('skip_hypersearch', False):
        best_config = find_best_config_for_fold(
            X_tensor, y_tensor, time_tensor, catalyst_tensor, tt_tensor,
            sample_ids, sample_col_ids, num_cols, device, save_dir,
            train_mask, test_sample_id
        )
    else:
        best_config = config

    # Create config for final training with best hyperparameters
    final_config = copy.deepcopy(config)
    if not config.get('skip_hypersearch', False):
        final_config['pytorch_hidden_dim'] = best_config['pytorch_hidden_dim']
        final_config['pytorch_dropout_rate'] = best_config['pytorch_dropout_rate']
        final_config['init_mode'] = best_config['init_mode']
        final_config['adaptive_lr']['scheduler_type'] = best_config['scheduler_type']
        final_config['generate_plots'] = True  # Enable plots for final training

    fold_data_with_params = (
        (fold_idx, test_sample_id), num_cols, final_config, device, save_dir,
        X_tensor, y_tensor, time_tensor, catalyst_tensor, tt_tensor,
        sample_ids, sample_col_ids, use_val_data_days
    )
    fold_result = train_single_fold_reactor_scaling(fold_data_with_params)
    
    return {
        'model_state': fold_result['model_state'],
        'val_loss': fold_result['val_loss'],
        'results': fold_result['results'],  # Changed from 'summary' to 'results'
        'config': best_config
    }

def train_reactor_scaling_model_parallel(X_tensor, y_tensor, time_tensor, catalyst_tensor, 
                                        tt_tensor, sample_ids, sample_col_ids, num_cols, 
                                        config, device, save_dir, use_val_data_days=0):
    print("Training Reactor Scaling Model with Parallelization...")
    unique_sample_ids = list(set(sample_ids))
    
    print(f"Starting parallel training with {len(unique_sample_ids)} folds...")
    
    all_models = []
    all_val_losses = []
    all_results = []  # Changed from all_summary to all_results
    all_best_configs = []
    
    with Pool(processes=mp.cpu_count()) as pool:
        fold_results = list(tqdm(pool.imap(
            partial(process_fold,
                    X_tensor=X_tensor,
                    y_tensor=y_tensor,
                    time_tensor=time_tensor,
                    catalyst_tensor=catalyst_tensor,
                    tt_tensor=tt_tensor,
                    sample_ids=sample_ids,
                    sample_col_ids=sample_col_ids,
                    num_cols=num_cols,
                    config=config,
                    device=device,
                    save_dir=save_dir,
                    use_val_data_days=use_val_data_days),
            enumerate(unique_sample_ids)
        ), total=len(unique_sample_ids)))

    for fold_result in fold_results:
        all_models.append(fold_result['model_state'])
        all_val_losses.append(fold_result['val_loss'])
        all_results.extend(fold_result['results'])
        all_best_configs.append(fold_result['config'])

    # Save all per-fold best configs
    all_configs_df = pd.DataFrame(all_best_configs)
    all_configs_df.to_csv(os.path.join(save_dir, 'plots', 'all_best_configs.csv'), index=False)
    print(f"Saved all best configs (per fold) to CSV")

    # Compute overall best config
    config_groups = all_configs_df.groupby(['pytorch_hidden_dim', 'pytorch_dropout_rate', 'init_mode', 'scheduler_type'])
    mean_val_losses = config_groups['val_loss'].mean().reset_index()
    best_overall = mean_val_losses.loc[mean_val_losses['val_loss'].idxmin()]
    
    print(f"Overall best configuration (lowest average val_loss across folds):")
    print(best_overall)
    
    best_overall.to_csv(os.path.join(save_dir, 'plots', 'best_overall_config.csv'), index=False)
    print(f"Saved overall best config to CSV")

    # Train final model with best overall config
    final_config = copy.deepcopy(config)
    final_config['pytorch_hidden_dim'] = best_overall['pytorch_hidden_dim']
    final_config['pytorch_dropout_rate'] = best_overall['pytorch_dropout_rate']
    final_config['init_mode'] = best_overall['init_mode']
    final_config['adaptive_lr']['scheduler_type'] = best_overall['scheduler_type']
    final_config['generate_plots'] = True  # Enable plots for final training
    final_config['skip_hypersearch'] = True

    # Train final model for each fold with best config
    final_results = []
    final_models = []
    final_val_losses = []

    with Pool(processes=mp.cpu_count()) as pool:
        final_fold_results = list(tqdm(pool.imap(
            partial(process_fold,
                    X_tensor=X_tensor,
                    y_tensor=y_tensor,
                    time_tensor=time_tensor,
                    catalyst_tensor=catalyst_tensor,
                    tt_tensor=tt_tensor,
                    sample_ids=sample_ids,
                    sample_col_ids=sample_col_ids,
                    num_cols=num_cols,
                    config=final_config,
                    device=device,
                    save_dir=save_dir,
                    use_val_data_days=use_val_data_days),
            enumerate(unique_sample_ids)
        ), total=len(unique_sample_ids)))

    for fold_result in final_fold_results:
        final_models.append(fold_result['model_state'])
        final_val_losses.append(fold_result['val_loss'])
        final_results.extend(fold_result['results'])

    # Save validation results
    summary_df = pd.DataFrame(final_results)
    summary_df.to_csv(os.path.join(save_dir, 'plots', 'column_prediction_summary_validation_withoutReactors.csv'), index=False)
    print(f"Saved column_prediction_summary_validation with {len(summary_df)} validation entries")

    # Create ensemble model
    ensemble = EnsembleModels(
        model_states=final_models,
        val_losses=final_val_losses,
        total_features=len(num_cols),
        config=final_config,
        device=device,
        best_configs=all_best_configs
    )
    
    print("\n" + "="*60)
    print("REACTOR SCALING MODEL RESULTS SUMMARY")
    print("="*60)
    print(f"Average RMSE: {summary_df['rmse'].mean():.4f} ± {summary_df['rmse'].std():.4f}")
    print(f"Average R²: {summary_df['r2'].mean():.4f} ± {summary_df['r2'].std():.4f}")
    print(f"Average Bias: {summary_df['bias'].mean():.4f} ± {summary_df['bias'].std():.4f}")
    print(f"Average Confidence Level: {summary_df['confidence_level'].mean():.1f}% ± {summary_df['confidence_level'].std():.1f}%")
    print(f"Ensemble models: {ensemble.get_ensemble_info()['num_models']}")
    
    return ensemble, summary_df

def create_ensemble_predictions_with_uncertainty(ensemble, X_tensor, y_tensor, time_tensor, catalyst_tensor, 
                                               tt_tensor, sample_ids, sample_col_ids, save_dir, max_days=1600, config=None):
    """
    Create ensemble predictions with uncertainty quantification and extended time predictions
    """
    if not config.get('generate_plots', False):
        return pd.DataFrame()  # Return empty DataFrame if plots disabled

    print("Creating ensemble predictions with uncertainty...")
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    summary = []
    unique_sample_ids = list(set(sample_ids))
    z_score = 1.645  # 90% confidence interval
    # ±1 standard deviation from the mean covers approximately 68% of the data.#
    # ±1.96 standard deviations from the mean covers approximately 95% of the data.
    # ±3 standard deviations from the mean covers approximately 99.7% of the data.
    # 1.645 standard deviations from the mean covers approximately 90% of the data.
    # 1.282 standard deviations from the mean covers approximately 80% of the data.

    colors = {True: 'darkorange', False: 'royalblue'}  # True for catalyzed, False for control
    
    for sample_id in unique_sample_ids:
        plt.figure(figsize=(12, 8))
        
        # Get indices for this sample
        sample_indices = [i for i, sid in enumerate(sample_ids) if sid == sample_id]
        
        for idx in sample_indices:
            x_sample = X_tensor[idx:idx+1]
            catalyst_sample = catalyst_tensor[idx].unsqueeze(0)
            time_sample = time_tensor[idx].unsqueeze(0)
            tt_sample = tt_tensor[idx]
            target_recovery = y_tensor[idx]
            col_id = sample_col_ids[idx]
            
            # Determine if catalyzed
            is_catalyzed = torch.any(catalyst_sample > 0).item()
            
            # Create extended time series for prediction
            original_time = time_sample.squeeze().cpu().numpy()
            original_catalyst = catalyst_sample.squeeze().cpu().numpy()
            max_original_time = original_time.max() if original_time.size > 0 else max_days
            
            # Extend time range from last observed time to max_days
            extended_time = np.linspace(max_original_time, max_days, 100)[1:]  # Start from after last observed time
            full_time = np.concatenate([original_time, extended_time])
            
            # Extend catalyst values using last 21 days slope calculation
            if is_catalyzed and len(original_time) > 0:
                # Use last 21 days (or available points) to calculate slope
                num_points = min(21, len(original_time))
                if num_points > 1:
                    last_times = original_time[-num_points:]
                    last_catalysts = original_catalyst[-num_points:]
                    # Calculate linear regression slope
                    coeffs = np.polyfit(last_times, last_catalysts, deg=1)
                    slope, intercept = coeffs
                    extended_catalyst_values = intercept + slope * extended_time
                    # Ensure catalyst values don't go negative
                    extended_catalyst_values = np.maximum(extended_catalyst_values, 0.0)
                else:
                    # If only one point, use constant extension
                    extended_catalyst_values = np.full_like(extended_time, original_catalyst[-1])
                
                full_catalyst = np.concatenate([original_catalyst, extended_catalyst_values])
            else:
                # For control samples, keep catalyst at zero
                full_catalyst = np.concatenate([original_catalyst, np.zeros_like(extended_time)])
            
            # Convert to tensors
            extended_time_tensor = torch.tensor(full_time, dtype=torch.float32, device=ensemble.device).unsqueeze(0)
            extended_catalyst = torch.tensor(full_catalyst, dtype=torch.float32, device=ensemble.device).unsqueeze(0)
            
            # Use ensemble prediction method for extended time (plotting)
            batch_sample_ids = [sample_id]
            mean_pred, uncertainty, params_mean = ensemble.predict_with_params_and_uncertainty(
                x_sample, extended_catalyst, tt_sample, extended_time_tensor, batch_sample_ids
            )
            
            # Convert to numpy
            mean_pred = mean_pred.squeeze() if hasattr(mean_pred, 'squeeze') else mean_pred
            uncertainty = uncertainty.squeeze() if hasattr(uncertainty, 'squeeze') else uncertainty
            
            # Compute predictions DIRECTLY at original time points for accurate metrics
            original_time_tensor = time_sample
            original_catalyst = catalyst_tensor[idx].unsqueeze(0)  # Use actual catalyst values
            mean_pred_original, _, _ = ensemble.predict_with_params_and_uncertainty(
                x_sample, original_catalyst, tt_sample, original_time_tensor, batch_sample_ids
            )
            y_pred_original = mean_pred_original.squeeze() if hasattr(mean_pred_original, 'squeeze') else mean_pred_original
            y_true_np = target_recovery.cpu().numpy()
            
            # Ensure y_pred_original is 1D and matches y_true_np's shape
            if y_pred_original.ndim > 1:
                y_pred_original = y_pred_original.flatten()
            if y_true_np.ndim > 1:
                y_true_np = y_true_np.flatten()
            
            # Check for sufficient data points
            if len(y_true_np) == 0 or len(y_pred_original) == 0:
                print(f"Skipping metrics for {sample_id} ({col_id}): Empty data arrays")
                rmse = bias = r2 = cl = np.nan
            elif len(y_true_np) == 1:
                print(f"Warning: Single data point for {sample_id} ({col_id}), metrics may be unreliable")
                rmse = np.abs(y_pred_original - y_true_np)[0]
                bias = (y_pred_original - y_true_np)[0]
                r2 = np.nan  # R² undefined for single point
                cl = 100.0 if rmse <= CONFIG.get('bias_threshold', 6.0) else 0.0
            else:
                rmse = np.sqrt(np.mean((y_pred_original - y_true_np) ** 2))
                bias = np.mean(y_pred_original - y_true_np)
                r2 = 1 - np.sum((y_pred_original - y_true_np) ** 2) / np.sum((y_true_np - y_true_np.mean()) ** 2) if np.var(y_true_np) > 0 else np.nan
                cl = calculate_confidence_level(y_true_np, y_pred_original)
            
            # *** MODIFIED PLOTTING SECTION ***
            # Plot actual data points (always show these)
            plt.plot(original_time, y_true_np, 'o', label=f'Actual ({col_id})', 
                    alpha=0.3, markeredgecolor='none', color=colors[is_catalyzed])
            
            # For plotting predictions: 
            # - Control samples: plot the entire curve
            # - Catalyzed samples: plot only the portion after transition time
            if is_catalyzed:
                # Get transition time value
                tt_value = tt_sample.item() if isinstance(tt_sample, torch.Tensor) else tt_sample
                
                # Filter full time and predictions to only include post-transition period
                post_transition_mask = full_time >= tt_value
                full_time_filtered = full_time[post_transition_mask]
                mean_pred_filtered = mean_pred[post_transition_mask]
                uncertainty_filtered = uncertainty[post_transition_mask]
                
                # Plot only the catalyzed portion (after transition time)
                if len(full_time_filtered) > 0:
                    plt.plot(full_time_filtered, mean_pred_filtered, '-', 
                            label=f'Predicted ({col_id})\nRMSE: {rmse:.2f}, Bias: {bias:.2f}\nR²: {r2:.2f}, CL: {cl:.0f}%', 
                            color=colors[is_catalyzed])
                    plt.fill_between(full_time_filtered, 
                                    mean_pred_filtered - z_score * uncertainty_filtered, 
                                    mean_pred_filtered + z_score * uncertainty_filtered, 
                                    alpha=0.2, color=colors[is_catalyzed], 
                                    label=f'±{z_score:.2f}σ Uncertainty ({col_id})')
                
                # Add transition time line
                if tt_value > 0:
                    plt.vlines(x=tt_value, ymin=0, ymax=80, color=colors[is_catalyzed], 
                              linestyle='--', alpha=0.7, label=f'Transition Time')
            else:
                # Control samples: plot the entire curve as before
                plt.plot(full_time, mean_pred, '-', 
                        label=f'Predicted ({col_id})\nRMSE: {rmse:.2f}, Bias: {bias:.2f}\nR²: {r2:.2f}, CL: {cl:.0f}%', 
                        color=colors[is_catalyzed])
                plt.fill_between(full_time, 
                                mean_pred - z_score * uncertainty, 
                                mean_pred + z_score * uncertainty, 
                                alpha=0.2, color=colors[is_catalyzed], 
                                label=f'±{z_score:.2f}σ Uncertainty ({col_id})')
            
            # Store summary
            summary_entry = {
                'project_sample_id_reactormatch': sample_id,
                'project_col_id': col_id,
                'rmse': rmse,
                'bias': bias,
                'r2': r2,
                'confidence_level': cl,
                'uncertainty': uncertainty.mean(),
                'a1': params_mean[0, 0].item(),
                'b1': params_mean[0, 1].item(),
                'a2': params_mean[0, 2].item(),
                'b2': params_mean[0, 3].item(),
                'a3': params_mean[0, 4].item() if is_catalyzed else np.nan,
                'b3': params_mean[0, 5].item() if is_catalyzed else np.nan,
                'a4': params_mean[0, 6].item() if is_catalyzed else np.nan,
                'b4': params_mean[0, 7].item() if is_catalyzed else np.nan,
                'transition_time': tt_sample.item() if isinstance(tt_sample, torch.Tensor) else tt_sample
            }
            summary.append(summary_entry)
        
        # Finalize plot
        plt.title(f"Ensemble Predictions - {sample_id}\nExtended to {max_days} Days")
        plt.xlabel("Leach Duration (Days)")
        plt.ylabel("Cu Recovery (%)")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.ylim(0, 80)
        plt.xlim(0, max_days)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"ensemble_predictions_{sample_id}_withoutReactors.png"), dpi=300)
        plt.close()
        print(f"Saved ensemble prediction plot for {sample_id}")

    
    # Save ensemble_prediction_summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(save_dir, 'plots', 'ensemble_prediction_summary_withoutReactors.csv'), index=False)
    print(f"Saved ensemble_prediction_summary with {len(summary_df)} entries")
    
    return summary_df


def plot_reactor_scaling_diagnostics(results_df, save_dir):
    """
    Create diagnostic plots for reactor scaling model
    """
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Performance metrics distribution
    fig, axes = plt.subplots(2, 2, figsize=(3*4, 3*3), dpi=300)
    
    # RMSE distribution
    axes[0, 0].hist(results_df['rmse'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(results_df['rmse'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results_df["rmse"].mean():.1f}')
    axes[0, 0].set_xlabel('RMSE')
    axes[0, 0].axvline(round(CONFIG['rmse_threshold'], 1), color='black', linestyle='--', alpha=0.5, label='RMSE Threshold')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('RMSE Distribution')
    axes[0, 0].set_xlim(left=0, right=25)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    '''
    # R² distribution
    axes[0, 1].hist(results_df['r2'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(results_df['r2'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results_df["r2"].mean():.1f}')
    axes[0, 1].set_xlabel('R²')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('R² Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    '''
    # Confidence Level distribution
    axes[0, 1].hist(results_df['confidence_level'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(results_df['confidence_level'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results_df["confidence_level"].mean():.1f}')
    axes[0, 1].set_xlabel('Confidence Level (%)')
    axes[0, 1].axvline(round(CONFIG['confidence_target'], 1), color='black', linestyle='--', alpha=0.5, label='Confidence Level Target')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Confidence Level Distribution')
    axes[0, 1].set_xlim(left=0, right=100)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bias distribution
    axes[1, 0].hist(results_df['bias'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(results_df['bias'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results_df["bias"].mean():.1f}')
    axes[1, 0].axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero bias')
    axes[1, 0].set_xlabel('Bias')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Bias Distribution')
    axes[1, 0].set_xlim(left=-25, right=25)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Control vs Catalyzed performance
    control_results = results_df[results_df['sample_col_id'] == 'Control']
    catalyzed_results = results_df[results_df['sample_col_id'] == 'Catalyzed']
    
    x_pos = [1, 2]
    rmse_means = [control_results['rmse'].mean(), catalyzed_results['rmse'].mean()]
    rmse_stds = [control_results['rmse'].std(), catalyzed_results['rmse'].std()]
    
    axes[1, 1].bar(x_pos, rmse_means, yerr=rmse_stds, alpha=0.7, 
                   color=['blue', 'red'], capsize=5)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(['Control', 'Catalyzed'])
    axes[1, 1].set_ylim(bottom=0, top=14)
    axes[1, 1].axhline(round(CONFIG['rmse_threshold'], 1), color='black', linestyle='--', alpha=0.5, label='RMSE Threshold')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_title('RMSE: Control vs Catalyzed')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_stats_diagnostics_withoutReactors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Diagnostic plots saved to {plots_dir}")

def plot_reactor_scaling_diagnostics_uncertainty(results_df, save_dir):
    """
    Create diagnostic plots for reactor scaling model
    """
    print("Creating uncertainty diagnostic plots...")
    
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Performance metrics distribution
    fig, axes = plt.subplots(2, 2, figsize=(3*4, 3*3), dpi=300)
    
    # RMSE distribution
    axes[0, 0].hist(results_df['rmse'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(results_df['rmse'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results_df["rmse"].mean():.1f}')
    axes[0, 0].set_xlabel('RMSE')
    axes[0, 0].axvline(round(CONFIG['rmse_threshold'], 1), color='black', linestyle='--', alpha=0.5, label='RMSE Threshold')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('RMSE Distribution')
    axes[0, 0].set_xlim(left=0, right=25)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    '''
    # R² distribution
    axes[0, 1].hist(results_df['r2'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(results_df['r2'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results_df["r2"].mean():.1f}')
    axes[0, 1].set_xlabel('R²')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('R² Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    '''
    # Confidence Level distribution
    axes[0, 1].hist(results_df['confidence_level'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(results_df['confidence_level'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results_df["confidence_level"].mean():.1f}')
    axes[0, 1].set_xlabel('Confidence Level (%)')
    axes[0, 1].axvline(round(CONFIG['confidence_target'], 1), color='black', linestyle='--', alpha=0.5, label='Confidence Level Target')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Confidence Level Distribution')
    axes[0, 1].set_xlim(left=0, right=100)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    '''
    # Bias distribution
    axes[1, 0].hist(results_df['bias'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(results_df['bias'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results_df["bias"].mean():.1f}')
    axes[1, 0].axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero bias')
    axes[1, 0].set_xlabel('Bias')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Bias Distribution')
    axes[1, 0].set_xlim(left=-25, right=25)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    '''
    # Uncertainty distribution
    axes[1, 0].hist(results_df['uncertainty'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(results_df['uncertainty'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results_df["uncertainty"].mean():.1f}')
    # axes[1, 0].axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero bias')
    axes[1, 0].set_xlabel('Uncertainty')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Uncertainty Distribution')
    axes[1, 0].set_xlim(left=0, right=10)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Control vs Catalyzed performance
    control_results = results_df[results_df['sample_col_id'] == 'Control']
    catalyzed_results = results_df[results_df['sample_col_id'] == 'Catalyzed']
    
    x_pos = [1, 2]
    rmse_means = [control_results['rmse'].mean(), catalyzed_results['rmse'].mean()]
    rmse_stds = [control_results['rmse'].std(), catalyzed_results['rmse'].std()]
    
    axes[1, 1].bar(x_pos, rmse_means, yerr=rmse_stds, alpha=0.7, 
                   color=['blue', 'red'], capsize=5, )
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(['Control', 'Catalyzed'])
    axes[1, 1].set_ylim(bottom=0, top=14)
    axes[1, 1].axhline(round(CONFIG['rmse_threshold'], 1), color='black', linestyle='--', alpha=0.5, label='RMSE Threshold')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_title('RMSE: Control vs Catalyzed')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_stats_diagnostics_withoutReactors_withUncertainty.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Diagnostic plots saved to {plots_dir}")


def plot_diagnostics_control_catalyzed(results_df, save_dir):
    """
    Create diagnostic plots for reactor scaling model
    """
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    for x in ['Control', 'Catalyzed']:
        # Filter for control samples only
        sample_plot = results_df[results_df['sample_col_id'] == x]
        
        # 1. Performance metrics distribution
        fig, axes = plt.subplots(2, 2, figsize=(3*4, 3*3), dpi=300)
        
        # RMSE distribution
        axes[0, 0].hist(sample_plot['rmse'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(sample_plot['rmse'].mean(), color='red', linestyle='--', 
                        label=f'Mean: {sample_plot["rmse"].mean():.1f}')
        axes[0, 0].set_xlabel('RMSE')
        axes[0, 0].axvline(round(CONFIG['rmse_threshold'], 1), color='black', linestyle='--', alpha=0.5, label='RMSE Threshold')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('RMSE Distribution')
        axes[0, 0].set_xlim(left=0, right=25)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # R² distribution
        axes[1, 1].hist(sample_plot['r2'], bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].axvline(sample_plot['r2'].mean(), color='red', linestyle='--', 
                        label=f'Mean: {sample_plot["r2"].mean():.1f}')
        axes[1, 1].set_xlabel('R²')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('R² Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Confidence Level distribution
        axes[1, 0].hist(sample_plot['confidence_level'], bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].axvline(sample_plot['confidence_level'].mean(), color='red', linestyle='--', 
                        label=f'Mean: {sample_plot["confidence_level"].mean():.1f}')
        axes[1, 0].set_xlabel('Confidence Level (%)')
        axes[1, 0].axvline(round(CONFIG['confidence_target'], 1), color='black', linestyle='--', alpha=0.5, label='Confidence Level Target')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Confidence Level Distribution')
        axes[1, 0].set_xlim(left=0, right=100)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Bias distribution
        axes[0, 1].hist(sample_plot['bias'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].axvline(sample_plot['bias'].mean(), color='red', linestyle='--', 
                        label=f'Mean: {sample_plot["bias"].mean():.1f}')
        axes[0, 1].axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero bias')
        axes[0, 1].set_xlabel('Bias')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Bias Distribution')
        axes[0, 1].set_xlim(left=-25, right=25)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'model_stats_diagnostics_{x}_withoutReactors.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Diagnostic plots saved to {plots_dir}")


#%% ---------------------------
# MAIN EXECUTION
# ---------------------------

if __name__ == "__main__":
    print("Starting Copper Recovery PINN without Reactors")
    print("=" * 60)
    print("\nStep 1: Dataset preparation")
    reactor_filtered, feature_sets, weight_sets, param_bounds = compile_training_datasets(df_reactors, CONFIG)
    reactor_filtered.to_csv(os.path.join(folder_path_save, 'plots', 'processed_data_reactors.csv'), index=False)
    
    df_columns_filtered = filter_column_dataset_by_config(df_model_recCu_catcontrol_projects, CONFIG)
    
    df_columns_aug = embed_reactor_knowledge_to_columns(
        df_reactors=reactor_filtered,
        df_columns=df_columns_filtered,
    )
    df_columns_aug.to_csv(os.path.join(folder_path_save, 'plots', 'augmented_columns_withReactors.csv'), index=False)
    

    X_tensor, y_tensor, time_tensor, catalyst_tensor, tt_tensor, sample_ids, sample_col_ids, feature_weights, num_cols, scaler_X, out_df_unscaled = prepare_column_train_data(
        df=df_columns_aug, 
        config=CONFIG, 
        output_type='averaged', 
        fill_noncat_averages=False
    )
    out_df_unscaled.to_csv(os.path.join(folder_path_save, 'plots', 'processed_data_unscaled.csv'), index=False)
    

    print("\nStep 2: Bayesian Fitting with MF Priors from Reactors")
    print("=" * 60)

    # Collect results
    results = []
    traces = {}

    # Optional: Parallel fit (uncomment Pool if needed)
    # def fit_wrapper(row): return bayesian_fit_two_phase(row, reactor_filtered, CONFIG, folder_path_save)
    # with Pool(processes=4) as pool:
    #     fitted = pool.map(fit_wrapper, [row for _, row in out_df_unscaled.iterrows()])

    for _, row in out_df_unscaled.iterrows():
        trace, summary, model = bayesian_fit_two_phase(row, reactor_filtered, CONFIG, folder_path_save)
        sample_id = row['project_sample_id_reactormatch']
        col_id = row['project_col_id']
        traces[f"{sample_id}_{col_id}"] = trace
        
        # Compute metrics from posterior predictive
        ppc = pm.sample_posterior_predictive(trace, model=model, var_names=['obs'], return_inferencedata=False)
        
        # Get observed recoveries for comparison
        recoveries_obs = parse_array_str(row['cu_recovery_%'])

        # Handle ppc_obs shape properly
        ppc_obs = ppc['obs']

        # Debug: Print shapes to confirm
        print(f"Original ppc_obs shape for {sample_id}_{col_id}: {ppc_obs.shape}")
        print(f"recoveries_obs length: {len(recoveries_obs)}")

        # Reshape if needed: ppc_obs should be (num_samples, num_observations)
        if ppc_obs.ndim == 3:  # Shape: (chains, draws, observations)
            ppc_obs = ppc_obs.reshape(-1, ppc_obs.shape[-1])  # Flatten to (chains*draws, observations)

        # Limit to num_posterior_samples
        num_samples = min(CONFIG['bayesian']['num_posterior_samples'], ppc_obs.shape[0])
        ppc_obs = ppc_obs[:num_samples]

        print(f"Processed ppc_obs shape: {ppc_obs.shape}")

        # Verify shape matches observations
        if ppc_obs.shape[1] != len(recoveries_obs):
            print(f"ERROR: Shape mismatch! ppc_obs has {ppc_obs.shape[1]} observations but recoveries_obs has {len(recoveries_obs)}")
            print(f"Attempting to transpose ppc_obs...")
            ppc_obs = ppc_obs.T  # Try transposing
            print(f"After transpose, ppc_obs shape: {ppc_obs.shape}")

        # Average over samples (first dimension) to get mean prediction
        predicted = ppc_obs.mean(axis=0)

        print(f"Final predicted length: {len(predicted)}")

        # Ensure shapes match before computing metrics
        if len(predicted) != len(recoveries_obs):
            print(f"CRITICAL ERROR: Cannot compute metrics - shape mismatch persists!")
            print(f"Skipping sample {sample_id}_{col_id}")
            continue
        
        recoveries_obs = parse_array_str(row['cu_recovery_%'])
        print(f"recoveries_obs length: {len(recoveries_obs)}, predicted length: {len(predicted)}")
        
        rmse = root_mean_squared_error(recoveries_obs, predicted)
        bias = np.mean(predicted - recoveries_obs)
        r2 = r2_score(recoveries_obs, predicted)
        # Confidence: % of points where abs diff < threshold, averaged over samples
        abs_diffs = np.abs(ppc_obs - recoveries_obs[None, :])  # Broadcast obs to (samples, points)
        confidence_level = np.mean(abs_diffs < CONFIG['rmse_threshold']) * 100
        uncertainty = np.std(ppc_obs, axis=0).mean()  # Average std
        
        results.append({
            'sample_id': sample_id, 'sample_col_id': col_id,
            'rmse': rmse, 'bias': bias, 'r2': r2,
            'confidence_level': confidence_level, 'uncertainty': uncertainty
        })

    # Reuse your plotting functions (e.g., plot_reactor_scaling_diagnostics_uncertainty(results_df, folder_path_save))

    # For predictions/ensemble_summary (adapt create_ensemble_predictions_with_uncertainty)
    # For each test, load trace, sample params, compute mean/std curves at new_times = np.linspace(0, 2500, 100)
    def predict_from_trace(trace, transition, new_times, model=None):
        """
        Generate predictions from Bayesian trace using parameter samples.
        
        Args:
            trace: ArviZ InferenceData object containing posterior samples
            transition: Transition time for two-phase model
            new_times: Array of time points for prediction
            model: Not used (kept for compatibility)
            
        Returns:
            mean_curve: Mean predicted recovery curve
            std_curve: Standard deviation of predictions
        """
        # Extract parameter samples from trace posterior
        # trace.posterior has shape (chain, draw) so we need to flatten
        a1_samples = trace.posterior['a1'].values.flatten()
        b1_samples = trace.posterior['b1'].values.flatten()
        a2_samples = trace.posterior['a2'].values.flatten()
        b2_samples = trace.posterior['b2'].values.flatten()
        
        # Limit to num_posterior_samples
        num_samples = min(CONFIG['bayesian']['num_posterior_samples'], len(a1_samples))
        a1_samples = a1_samples[:num_samples]
        b1_samples = b1_samples[:num_samples]
        a2_samples = a2_samples[:num_samples]
        b2_samples = b2_samples[:num_samples]
        
        curves = []
        for i in range(num_samples):
            # Extract scalar values for this iteration
            a1 = a1_samples[i]  # This is now a scalar
            b1 = b1_samples[i]  # This is now a scalar
            a2 = a2_samples[i]  # This is now a scalar
            b2 = b2_samples[i]  # This is now a scalar
            
            # Two-phase model: pre-catalyst and post-catalyst
            pre_mask = new_times < transition
            pre_rec = a1 * (1 - np.exp(-b1 * new_times))  # Now broadcasts correctly: scalar * array
            pre_at_trans = a1 * (1 - np.exp(-b1 * transition))
            post_rec = pre_at_trans + a2 * (1 - np.exp(-b2 * (new_times - transition)))
            curve = np.where(pre_mask, pre_rec, post_rec)
            curves.append(curve)
        
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        return mean_curve, std_curve

    # Predictions loop: Generate extended curves with uncertainty
    predictions = []
    new_times = np.linspace(0, 2500, 100)  # Consistent with your max_days=2500
    for key, trace in traces.items():
        # Extract sample_id and col_id from key (e.g., '003jettiprojectfile_amcf_Control')
        sample_id, col_id = key.rsplit('_', 1)  # Adjust if key format changes
        # Get transition from out_df_unscaled (match on reactormatch and col_id)
        row_match = out_df_unscaled[(out_df_unscaled['project_sample_id_reactormatch'] == sample_id) & 
                                    (out_df_unscaled['project_col_id'] == col_id)]
        if row_match.empty:
            print(f"Skipping prediction for {key}: No matching row found.")
            continue
        transition = row_match['transition_time'].iloc[0]
        
        mean_curve, std_curve = predict_from_trace(trace, transition, new_times, model)
        
        # Save or collect for plotting (example: append to list for DF)
        predictions.append({
            'sample_id': sample_id,
            'col_id': col_id,
            'new_times': new_times.tolist(),
            'mean_recovery': mean_curve.tolist(),
            'std_recovery': std_curve.tolist()
        })
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(os.path.join(folder_path_save, 'bayesian_predictions_extended.csv'), index=False)
    
    # Optional: Plot example (adapt to your plot_reactor_scaling_diagnostics style)
    for pred in predictions:
        plt.figure(figsize=(10, 6))
        plt.plot(pred['new_times'], pred['mean_recovery'], label='Mean Recovery')
        plt.fill_between(np.array(pred['new_times']), 
                         np.array(pred['mean_recovery']) - np.array(pred['std_recovery']), 
                         np.array(pred['mean_recovery']) + np.array(pred['std_recovery']), 
                         alpha=0.3, label='Uncertainty')
        plt.title(f"Predicted Recovery for {pred['sample_id']}_{pred['col_id']}")
        plt.xlabel('Days')
        plt.ylabel('Cu Recovery %')
        plt.legend()
        plt.savefig(os.path.join(folder_path_save, 'plots', f'prediction_{pred["sample_id"]}_{pred["col_id"]}.png'))
        plt.close()

    scaling_models, scaling_results = train_reactor_scaling_model_parallel(
        X_tensor=X_tensor,
        y_tensor=y_tensor,
        time_tensor=time_tensor,
        catalyst_tensor=catalyst_tensor,
        tt_tensor=tt_tensor,
        sample_ids=sample_ids,
        sample_col_ids=sample_col_ids,
        num_cols=num_cols,
        config=CONFIG,
        device=device,
        save_dir=folder_path_save,
        use_val_data_days=CONFIG.get('use_val_data_days', 0)
    )

    plot_reactor_scaling_diagnostics(scaling_results, folder_path_save)
    plot_reactor_scaling_diagnostics_uncertainty(scaling_results, folder_path_save)
    plot_diagnostics_control_catalyzed(scaling_results, folder_path_save)

    final_config = copy.deepcopy(CONFIG)
    final_config['generate_plots'] = True

    ensemble_summary = create_ensemble_predictions_with_uncertainty(
        ensemble=scaling_models,
        X_tensor=X_tensor,
        y_tensor=y_tensor,
        time_tensor=time_tensor,
        catalyst_tensor=catalyst_tensor,
        tt_tensor=tt_tensor,
        sample_ids=sample_ids,
        sample_col_ids=sample_col_ids,
        save_dir=folder_path_save,
        max_days=2500,
        config=final_config
    )
    
    print("\nStep 3: Reactor Scaling Model Training Complete")
    print("=" * 60)
    print(f"Results saved to: {folder_path_save}")
    print(f"Number of models trained: {scaling_models.get_ensemble_info()['num_models']}")
    print(f"Average RMSE: {scaling_results['rmse'].mean():.4f}")
    print(f"Average R²: {scaling_results['r2'].mean():.4f}")
    print(f"Average Bias: {scaling_results['bias'].mean():.4f}")
    
    torch.save({
        'models': scaling_models,
        'results': scaling_results,
        'num_cols': num_cols,
        'scaler_X': scaler_X
    }, os.path.join(folder_path_save, 'AdaptiveTwoPhaseModel_withoutReactors.pt'))
    
    print("\nReactor Scaling Model training completed successfully!")
    print("All results and plots have been saved.")

# %%
