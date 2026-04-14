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
    # 'feed_head_cu_%',
    # 'feed_head_fe_%',
    'grouped_accessory_minerals',
    'grouped_copper_sulfides',
    'grouped_secondary_copper',
    'grouped_acid_generating_sulfides',
    'grouped_fe_oxides',
    'grouped_gangue_silicates',
    'grouped_carbonates',
    'grouped_secondary_copper',
    'cpy_+50%_exposed_norm',
    'cpy_locked_norm',
    'cpy_associated_norm',
    'cumulative_catalyst_(kg_t)_max',
    'cumulative_catalyst_(kg_t)_slope',
    'catalyst_dose_(mg_l)',
    'ph_mean',
    'orp_(mv)_mean',
    # 'irrigation_rate_l_m2_h',
    # 'cumulative_lixiviant_m3_t'
]
df_reactors[df_reactors['project_sample_id'] == '007ajettiprojectfile_elephant_pq_rthead'][cols_to_check]

df_model_recCu_catcontrol_projects.columns

pd.DataFrame(df_reactors.describe()).to_excel('/Users/administration/OneDrive - Jetti Resources/Rosetta/NN_PyTorch/describe_reactors.xlsx')

folder_path_save = '/Users/administration/OneDrive - Jetti Resources/Rosetta/NN_PyTorch'

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

# Duplicate and append rows for '015jettiprojectfile_pv'
duplicated_rows = df_reactors[df_reactors['project_sample_id'] == '015jettiprojectfile_pv'].copy()
duplicated_rows['project_sample_id'] = '015jettiprojectfile_pv_6in'
df_reactors = pd.concat([df_reactors, duplicated_rows], ignore_index=True)
duplicated_rows['project_sample_id'] = '015jettiprojectfile_pv_8in'
df_reactors = pd.concat([df_reactors, duplicated_rows], ignore_index=True)

# REMOVE PVO1 TO AVOID WEIRD BEHAVIOURS
df_model_recCu_catcontrol_projects = df_model_recCu_catcontrol_projects[
    df_model_recCu_catcontrol_projects['project_col_id'] != '006_jetti_project_file_pvo1'
]

# REMOVE WHOLE PVLS BECAUSE IT IS RESIDUES LEACHED???
df_model_recCu_catcontrol_projects = df_model_recCu_catcontrol_projects[
    df_model_recCu_catcontrol_projects['project_sample_id'] != '006_jetti_project_file_pvls'
]

# Remove Project 022 < 7.0 recovery
df_model_recCu_catcontrol_projects = df_model_recCu_catcontrol_projects[
    ~(
        (df_model_recCu_catcontrol_projects['project_sample_id'].str.startswith('022_jetti_project_file_'))
        &
        (df_model_recCu_catcontrol_projects['cu_recovery_%'] < 7.0)
    )
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
    'pytorch_dropout_rate': 0.33,
    'pytorch_timeout': 300,     # Timeout in seconds for PyTorch training (5 minutes)
    'min_pre_catalyst_points': 9,  # Minimum number of pre-catalyst points to consider a sample valid

    # *** NEW: Adaptive Learning Rate Configuration ***
    'adaptive_lr': {
        'enabled': True,
        'scheduler_type': 'reduce_on_plateau',  # Options: 'reduce_on_plateau', 'cosine_annealing', 'step', 'exponential'
        'base_lr': 1e-4,  # Base learning rate (higher than original 1e-5)
        'min_lr': 1e-6,   # Minimum learning rate
        'max_lr': 1e-2,   # Maximum learning rate
        
        # ReduceLROnPlateau parameters
        'plateau_factor': 0.5,      # Factor to reduce LR by
        'plateau_patience': 200,    # Epochs to wait before reducing
        'plateau_threshold': 0.01,  # Minimum change to qualify as improvement
        
        # CosineAnnealingWarmRestarts parameters
        'cosine_T_0': 200,          # Initial restart period
        'cosine_T_mult': 2,         # Factor to increase restart period
        
        # StepLR parameters
        'step_size': 500,           # Period of learning rate decay
        'step_gamma': 0.7,          # Multiplicative factor of learning rate decay
    },

    # SHAP analysis parameters
    'shap_sample_size': 100,     # Number of samples to use for SHAP analysis
    'shap_background_size': 100, # Number of samples to use for SHAP background
    
    # Plot parameters
    'max_plots_per_grid': 15,   # Maximum number of plots per grid
    'grid_columns': 2,          # Number of columns in the grid plot
    # 'grid_figsize': (15, 20),   # Figure size for grid plots (width, height)
    
    # Recovery calculation parameters
    'recovery_time_points': 100,  # Number of time points for recovery calculation
    'recovery_max_time': 125,     # Maximum time for recovery calculation

    'column_tests_feature_weighting': {
        'enabled': True,  # Whether to enable feature selection and weighting for column tests
        'use_monotonic_constraints': True,  # Hard projection
        'use_gradient_masking': True,       # Gradient blocking
        'use_penalty_loss': True,           # Soft guidance
        'weights': { # title and effect per parameter predicted in this order: a1, b1, a2, b2, a3, b3, a4, b4
            'leach_duration_days': ['Leach Duration (days)', 1, 1, 1, 1, 1, 1, 1, 1], 
            'cumulative_catalyst_addition_kg_t': ['Cumulative Catalyst added (kg/t)', 1, 1, 1, 1, 1, 1, 1, 1], 
            'acid_soluble_%': ['Acid Soluble Cu (%norm)',  1.29, 1.13, 1.41, 1.02, 0.58, 0.89, 1.02, 0.89], 
            'residual_cpy_%': ['Residual Chalcopyrite (%norm)', -1.57, -1.56, -1.51, -1.35, 1.34, 1.26, 1.62, 1.28],
            # 'cyanide_soluble_%': ['Cyanide Soluble (%norm)', 0.0], # removed only to check changes on other features and relationships
            # 'cu_seq_a_r_%': ['Residual Chalcopyrite (%)', 0.0], # removed because of low impact on feature importance plots
            # 'feed_head_cu_%': ['Feed Head Cu (%)', 0.93, 0, 0.90, 0, 1.33, 0, 1.37, 0],
            # 'feed_head_fe_%': ['Feed Head Fe (%)', -1, -1, -1, -1, -1, -1, -1, -1],
            'material_size_p80_in': ['Material Size P80 (in)', -2.41, -0.73, -1.46, -1.50, -1.99, -2.04, -2.04, -2.00],
            'grouped_copper_sulfides': ['Copper Sulphides (%)', 1, 0, 1, 0, 1, 0, 1, 0],
            'grouped_secondary_copper': ['Secondary Copper (%)', 1, 0, 1, 0, 1, 0, 1, 0], # 1, 1, 1, 1, 1, 1, 1, 1
            'grouped_acid_generating_sulfides': ['Acid Generating Sulphides (%)', 0, 0.64, 0, 0.48, 0, 0.36, 0, 0.48],
            # 'grouped_gangue_sulfides': ['Gangue Sulphides (%)', 0.0], # removed because of low impact on feature importance plots
            'grouped_gangue_silicates': ['Gangue Silicates (%)', 0, 0, 0, 0, 0, 0, 0, 0],
            # 'grouped_clays_and_micas': ['Clays and Micas (%)', 0.0],
            # 'grouped_accesory_silicates': ['Accesory Silicates (%)', 0.0],
            # 'grouped_sulfates': ['Sulfates (%)', 0.0],
            'grouped_fe_oxides': ['Fe Oxides (%)', -0.41, -1.16, -0.16, -0.39, -0.92, -0.36, -0.98, -0.39],
            # 'grouped_accessory_misc': ['Accessory Misc (%)', 0.0],
            'grouped_carbonates': ['Carbonates (%)', -1.07, 1.41, -1.10, 0.46, -1.08, 1.02, -0.77, 1.08],
            # 'grouped_accessory_minerals': ['Accessory Minerals (%)', 0.0],
            'cumulative_lixiviant_m3_t': ['Cumulative Lixiviant (kg/t)', 0, 0, 0, 0, 0, 0, 0, 0],
        },
    },
    
    # Monotonic constraint penalty weight
    'monotonic_penalty_weight': 0.01,  # Weight for soft constraint penalty (they are scaled afterwards)
    
    # Adaptive penalty settings (optional)
    'adaptive_penalty': {
        'enabled': True,
        'increase_factor': 1.2,      # Multiply by this when violations are high
        'decrease_factor': 0.9,      # Multiply by this when violations are low
        'violation_threshold_high': 0.10,  # Increase penalty if > 10% violated
        'violation_threshold_low': 0.05,   # Decrease penalty if < 5% violated
        'min_penalty_weight': 0.001,       # Minimum penalty weight
        'max_penalty_weight': 1.0,        # Maximum penalty weight
    },

    'z_score': 1.645,              # nominal 90% multiplier
    'uncertainty_scale': 1.0,      # post-hoc calibration factor (set via validation)
    'target_predictive_interval_coverage': 0.85, # target PI coverage for uncertainty calibration
    # ±1 standard deviation from the mean covers approximately 68% of the data.#
    # ±1.96 standard deviations from the mean covers approximately 95% of the data.
    # ±3 standard deviations from the mean covers approximately 99.7% of the data.
    # 1.645 standard deviations from the mean covers approximately 90% of the data.
    # 1.282 standard deviations from the mean covers approximately 80% of the data.

    'special_feats': {
        'dynamic': ['leach_duration_days', 'cumulative_lixiviant_m3_t', 'cumulative_catalyst_addition_kg_t'],
        'time_feat': ['cumulative_lixiviant_m3_t'],
        'catalyst_feat': ['cumulative_catalyst_addition_kg_t'],
        'categorical': [],
        'target_feat': ['cu_recovery_%'],
    },

    # Use validation data from test to append to training
    'use_val_data_days': 0,  # Number of days from test set to append to training set (0 to disable)
    'generate_plots': False,
    'skip_hypersearch': False,

    # Global recovery caps
    'base_asymptote_cap': 80.0, # applies to a1 + a2 (control)
    'total_asymptote_cap': 95.0,  # applies to a1 + a2 + a3 + a4 (control+catalyzed)

    # Global rate caps
    'base_rate_cap': 2.1,       # applies to b1 + b2 (control)
    'ratio_constraints': {
        'enable': False,
        'potentiate_weight': 0.1,   # weight for pushing ratios upward
        'avoid_weight': 0.1,        # weight for discouraging forbidden ratios
        'hard_avoid_b3_over_b1': True,  # enforce b3 <= 0.1*b1 by projection
        'min_a1_over_b1': 10.0,     # hard floor (avoid low)
        'target_a1_over_b1': 15.0,  # soft target (potentiate)
        'target_a1_over_b3': 50.0,
        'target_a2_over_b1': 20.0,
        'max_b3_over_b1': 0.1
    },
}


CONFIG.update({
    'gamma_max': 2.0,
    'gamma_zero_penalty': 0.01,
    'post_tt_monotone_penalty_weight': 0.05,
})


def project_params_to_caps(params: torch.Tensor,
                           base_cap: float,
                           total_cap_unused: float,
                           base_rate_cap: float) -> torch.Tensor:
    """
    Params shape (B,8): a1,b1,a2,b2,gA1,gB1,gA2,gB2
    Caps:
      a1+a2 <= base_cap
      b1+b2 <= base_rate_cap
      gammas clamped to [0, gamma_max]
    """
    if params.ndim == 1:
        params = params.unsqueeze(0)
    a1,b1,a2,b2 = [params[:, i] for i in range(4)]
    gammas = params[:, 4:8]
    # amplitude cap
    base_sum = a1 + a2
    scale_amp = torch.where(base_sum > base_cap,
                            base_cap / base_sum.clamp(min=1e-6),
                            torch.ones_like(base_sum))
    a1 = a1 * scale_amp
    a2 = a2 * scale_amp
    # rate cap
    rate_sum = b1 + b2
    scale_rate = torch.where(rate_sum > base_rate_cap,
                             base_rate_cap / rate_sum.clamp(min=1e-9),
                             torch.ones_like(rate_sum))
    b1 = b1 * scale_rate
    b2 = b2 * scale_rate
    # gammas clamp
    gamma_max = float(CONFIG.get('gamma_max', 2.0))
    gammas = F.softplus(gammas).clamp(max=gamma_max)
    return torch.cat([a1.unsqueeze(1), b1.unsqueeze(1), a2.unsqueeze(1), b2.unsqueeze(1), gammas], dim=1)

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
        # [256, 0.3, 'kaiming', 'reduce_on_plateau'] or
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
    target_col = CONFIG.get('special_feats', {}).get('target_feat', [])[0]
    id_col = 'project_sample_id'
    col_idx = 'project_col_id'
    df = df[df[target_col] > 1.0].copy()

    # Define time-varying columns
    special = CONFIG.get('special_feats', {})
    time_varying_feats = list(special.get('dynamic', []))
    target_feats = list(special.get('target_feat', []))
    time_varying_cols = time_varying_feats + target_feats

    catalyst_feats = list(special.get('catalyst_feat', []))
    time_feat_feats = list(special.get('time_feat', []))

    target_col = target_feats[0] if target_feats else 'cu_recovery_%'
    catalyst_col = catalyst_feats[0] if catalyst_feats else None
    time_feat_col = time_feat_feats[0] if time_feat_feats else None
    leach_days = 'leach_duration_days'

    # Filter rows with NaNs in key columns
    subset_cols = [target_col] + time_varying_feats
    df_filtered = df.dropna(subset=subset_cols).copy()

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
        no_catalyst = group[group[catalyst_col] == 0]
        with_catalyst = group[group[catalyst_col] > 0]

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
            max_days = max([row[leach_days].max() for _, row in no_cat_rows.iterrows() if len(row['leach_duration_days']) > 0])
        if not with_cat_rows.empty:
            max_days = max(max_days, max([row[leach_days].max() for _, row in with_cat_rows.iterrows() if len(row['leach_duration_days']) > 0]))
        
        # Process no-catalyst data
        no_cat_processed = {}
        if not no_cat_rows.empty:
            row_no = {'catalyst_status': 'no_catalyst', id_col: sample_id}
            for col in time_varying_cols:
                no_cat_processed[col] = process_arrays_by_weekly_intervals(
                    no_cat_rows[col].iloc[0], no_cat_rows[leach_days].iloc[0], col
                )
            # Skip if no non-empty batches
            if len(no_cat_processed[leach_days]) > 0:
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
                with_cat_days = with_cat_rows[leach_days].iloc[0]
                # Process catalyzed data
                processed_with = process_arrays_by_weekly_intervals(with_cat_data, with_cat_days, col)
                # Prepend no-catalyst data if available
                if not no_cat_rows.empty and len(no_cat_processed[leach_days]) > 0 and fill_noncat_averages:
                    no_cat_data = no_cat_processed[col]
                    no_cat_days = no_cat_processed[leach_days]
                    # Ensure no overlap in leach_duration_days
                    with_cat_start = with_cat_days.min() if len(with_cat_days) > 0 else np.inf
                    no_cat_idx = np.where(no_cat_days < with_cat_start)[0] if len(no_cat_days) > 0 else []
                    if len(no_cat_idx) > 0:
                        # Concatenate no-catalyst and with-catalyst data
                        if col == leach_days:
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
            if len(processed_cols[leach_days]) > 0:
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
        catalyst_days = row[time_feat_col][row['cumulative_catalyst_addition_kg_t'] > 0]
        out_df.at[idx, 'transition_time'] = catalyst_days.min() if catalyst_days.size > 0 else row[time_feat_col].max()

    # Compute transition_time for split grouping
    out_df_averaged['transition_time'] = 0.0
    for idx, row in out_df_averaged.iterrows():
        sample_id = row[id_col]
        sample_data = df_filtered[df_filtered[id_col] == sample_id]
        catalyst_days = sample_data[time_feat_col][sample_data['cumulative_catalyst_addition_kg_t'] > 0]
        if row['catalyst_status'] == 'with_catalyst':
            # First day where catalyst > 0, or last day if no catalyst
            out_df_averaged.at[idx, 'transition_time'] = (
                catalyst_days.min() if catalyst_days.size > 0 else 
                sample_data[time_feat_col].max() if sample_data[time_feat_col].size > 0 else 0.0
            )
        else:
            # For no_catalyst, use the last day of the processed no_catalyst data
            out_df_averaged.at[idx, 'transition_time'] = (
                row[time_feat_col][-1] if len(row[time_feat_col]) > 0 else 0.0
            )

    # Prepare features for training (exclude non-feature columns)
    drop_cols = ['project_sample_id_reactormatch', 'project_sample_id', 'project_col_id', 
                 time_feat_col, leach_days, target_col, catalyst_col, 'catalyst_status']
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
        # If time_feat is different from leach_duration_days, remove leach_duration_days from numeric_cols
        if time_feat_col != leach_days and leach_days in numeric_cols:
            numeric_cols.remove(leach_days)
            
        # Scale numerical features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(out_df[numeric_cols])
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        # Convert time-varying columns and other outputs to tensors
        Y_tensor = [torch.tensor(row[target_col], dtype=torch.float32).to(device) for _, row in out_df.iterrows()]
        time_tensor = [torch.tensor(row[time_feat_col], dtype=torch.float32).to(device) for _, row in out_df.iterrows()]
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
        # If time_feat is different from leach_duration_days, remove leach_duration_days from numeric_cols
        if time_feat_col != leach_days and leach_days in numeric_cols:
            numeric_cols.remove(leach_days)

        scaler_X_averaged = StandardScaler()
        X_scaled_averaged = scaler_X_averaged.fit_transform(out_df_averaged[numeric_cols])
        X_tensor_averaged = torch.tensor(X_scaled_averaged, dtype=torch.float32).to(device)

        # Convert time-varying columns and other outputs to tensors
        Y_tensor_averaged = [torch.tensor(row['cu_recovery_%'], dtype=torch.float32).to(device) for _, row in out_df_averaged.iterrows()]
        time_tensor_averaged = [torch.tensor(row[time_feat_col], dtype=torch.float32).to(device) for _, row in out_df_averaged.iterrows()]
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
# Step 2: Normalize features and create datasets
# ---------------------------




#%% ---------------------------
# Step 3: Merge the columns and reactors dataset by id
# ---------------------------


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

    df_filtered = df_columns[['project_sample_id', 'project_col_id'] + selected_features + [col for col in df_columns.columns if col.startswith('cu_recovery')]].copy()
    return df_filtered


#%% ---------------------------
# Step 4: Prepare column train data
# ---------------------------
def get_feature_weight_magnitudes(config, feature_names, zero_policy='one'):
    W = config.get('column_tests_feature_weighting', {}).get('weights', {})
    dyn = set(config.get('special_feats', {}).get('dynamic', []))
    rows = []
    eps = 1e-3
    for f in feature_names:
        if f in dyn or f not in W or len(W[f]) < 9:
            rows.append([1.0]*8)
            continue
        vals = [abs(float(v)) for v in W[f][1:9]]
        if zero_policy == 'one':
            vals = [1.0 if v == 0.0 else v for v in vals]
        elif zero_policy == 'epsilon':
            vals = [eps if v == 0.0 else v for v in vals]
        rows.append(vals)
    return torch.tensor(rows, dtype=torch.float32)

def get_feature_weight_signs(config, feature_names):
    col_cfg = config.get('column_tests_feature_weighting', {})
    if not (col_cfg.get('enabled', False) and col_cfg.get('use_monotonic_constraints', False)):
        return torch.zeros(len(feature_names), 8)
    raw = col_cfg.get('weights', {})
    dyn = set(config.get('special_feats', {}).get('dynamic', []))
    out = []
    for f in feature_names:
        if f in dyn or f not in raw or len(raw[f]) < 9:
            out.append([0.0]*8)
            continue
        w = raw[f][1:9]
        signs = [1.0 if v > 0 else -1.0 if v < 0 else 0.0 for v in w]
        out.append(signs)
    return torch.tensor(out, dtype=torch.float32)

def compute_monotonic_penalty(model, lambda_penalty=0.1):
    if not model.use_monotonic_constraints:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    scales = torch.ones(8, device=next(model.parameters()).device)
    for param_idx in range(8):
        W = model.param_networks[param_idx][0].weight
        for feat_idx in range(model.feature_weight_signs.shape[0]):
            sign_c = model.feature_weight_signs[feat_idx, param_idx].item()
            if sign_c > 0:
                penalty += scales[param_idx] * torch.relu(-W[:, feat_idx]).sum()
            elif sign_c < 0:
                penalty += scales[param_idx] * torch.relu(W[:, feat_idx]).sum()
    return lambda_penalty * penalty

def check_constraint_violations(model):
    if not model.use_monotonic_constraints:
        return 0, 0, 0.0
    violations = 0
    total = 0
    for p_idx in range(8):
        W = model.param_networks[p_idx][0].weight
        for f_idx in range(model.feature_weight_signs.shape[0]):
            s = model.feature_weight_signs[f_idx, p_idx].item()
            if s == 0: 
                continue
            total += 1
            m = W[:, f_idx].mean().item()
            if (s > 0 and m < 0) or (s < 0 and m > 0):
                violations += 1
    return violations, total, (violations/total if total>0 else 0.0)


#%% ---------------------------
# REACTOR SCALING ENSEMBLE MODEL
# ---------------------------

class EnsembleModels:
    def __init__(self, model_states, val_losses, total_features, config, device, best_configs, num_cols):
        self.device = device
        self.total_features = total_features
        self.config = config
        self.best_configs = best_configs
        self.num_cols = num_cols  # store for residual_cpy lookup
        self.feature_weight_signs = get_feature_weight_signs(config, num_cols).to(device)
        self.models, self.weights = self._create_filtered_ensemble(model_states, val_losses, config)

    def _create_filtered_ensemble(self, model_states, val_losses, config):
        """Create filtered ensemble based on validation losses"""
        median_loss = np.median(val_losses)
        # threshold = median_loss * 1.5  # original was 1.5
        threshold = np.percentile(val_losses, 95) # approx 95% of the models are included
        
        models = []
        weights = []
        
        # Use the overall best hidden_dim and dropout_rate from final_config
        hidden_dim = config.get('pytorch_hidden_dim', 128)
        dropout_rate = config.get('pytorch_dropout_rate', 0.33)
        
        for idx, (model_state, val_loss) in enumerate(zip(model_states, val_losses)):
            if val_loss <= threshold:
                model = AdaptiveTwoPhaseRecoveryModel(
                    total_features=self.total_features,
                    hidden_dim=hidden_dim,  # Use overall best from final_config
                    dropout_rate=dropout_rate,  # Use overall best from final_config
                    init_mode=config.get('init_mode', 'kaiming'), # Use overall best init_mode
                    feature_weight_signs=self.feature_weight_signs,
                    # feature_weight_magnitudes=self.feature_weight_mags,
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
        all_preds = []
        all_params = []
        for model in self.models:
            with torch.no_grad():
                params = model(X, catalyst, sample_ids)
                params = project_params_to_caps(params,
                    float(self.config.get('base_asymptote_cap', 80.0)),
                    float(self.config.get('total_asymptote_cap', 95.0)),
                    float(self.config.get('base_rate_cap', 2.1)),
                )
                recovery = generate_two_phase_recovery(time_points, catalyst, transition_time, params)
                all_preds.append(recovery.cpu().numpy())      # (B,T)
                all_params.append(params.cpu().numpy())       # (B,5)
        all_preds = np.array(all_preds)   # (M,B,T)
        all_params = np.array(all_params) # (M,B,5)
        w = np.asarray(self.weights, dtype=float)
        w = w / (w.sum() if w.sum() > 0 else 1.0)
        w3 = w[:, None, None]
        mean_pred = (all_preds * w3).sum(axis=0)
        mean_params = (all_params * w[:, None, None]).sum(axis=0)  # (B,5)
        var = (w3 * (all_preds - mean_pred[None])**2).sum(axis=0)
        uncertainty = np.sqrt(np.maximum(var, 0.0))
        return mean_pred, uncertainty, mean_params
    
    def get_ensemble_info(self):
        return {'num_models': len(self.models), 'weights': [float(w) for w in self.weights]}


#%% ---------------------------
# REACTOR SCALING MODEL
# ---------------------------
class AdaptiveTwoPhaseRecoveryModel(nn.Module):
    def __init__(self, total_features, hidden_dim=128, dropout_rate=0.33,
                 init_mode='kaiming', feature_weight_signs=None, feature_weight_magnitudes=None):
        super().__init__()
        self.total_features = total_features
        self.dropout_rate = dropout_rate
        self.init_mode = init_mode
        if feature_weight_signs is not None:
            self.register_buffer('feature_weight_signs', feature_weight_signs[:, :8])
            self.use_monotonic_constraints = True
        else:
            self.register_buffer('feature_weight_signs', torch.zeros(total_features, 8))
            self.use_monotonic_constraints = False
        self.param_networks = nn.ModuleList()
        for _ in range(8):  # a1,b1,a2,b2,gA1,gB1,gA2,gB2
            net = nn.Sequential(
                nn.Linear(total_features, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.param_networks.append(net)
        self._initialize_weights(self.init_mode)

    def _initialize_weights(self, mode='kaiming'):
        for idx, net in enumerate(self.param_networks):
            for layer in net:
                if isinstance(layer, nn.Linear):
                    if mode == 'kaiming':
                        nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                    elif mode == 'xavier':
                        nn.init.xavier_uniform_(layer.weight)
                    else:
                        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                    nn.init.constant_(layer.bias, 0.0)
        with torch.no_grad():
            # rate heads (b1 idx=1, b2 idx=3)
            self.param_networks[1][-1].bias[0] = -2.7
            self.param_networks[3][-1].bias[0] = -2.7
            # gamma heads strong negative start (4..7)
            for g_idx in range(4,8):
                self.param_networks[g_idx][-1].bias[0] = -6.0

    def apply_monotonic_constraints(self):
        if not self.use_monotonic_constraints: return
        with torch.no_grad():
            for p_idx in range(8):
                W = self.param_networks[p_idx][0].weight
                for f_idx in range(self.feature_weight_signs.shape[0]):
                    sign_c = self.feature_weight_signs[f_idx, p_idx].item()
                    if sign_c > 0: W[:, f_idx].data = torch.abs(W[:, f_idx].data)
                    elif sign_c < 0: W[:, f_idx].data = -torch.abs(W[:, f_idx].data)

    def forward(self, x, catalyst, sample_ids=None):
        if self.training and self.use_monotonic_constraints:
            self.apply_monotonic_constraints()
        B = x.size(0)
        params = torch.zeros(B, 8, device=x.device)
        lim = CONFIG.get('param_limits', {})
        a1_min,a1_max = lim.get('a1', (1.5,45.0))
        b1_min,b1_max = lim.get('b1', (0.01,2.1))
        a2_min,a2_max = lim.get('a2', (9.5,79.0))
        b2_min,b2_max = lim.get('b2', (5e-3,0.3))
        gamma_max = float(CONFIG.get('gamma_max',2.0))

        def map_heads(raws):
            a1_raw,b1_raw,a2_raw,b2_raw,gA1_raw,gB1_raw,gA2_raw,gB2_raw = raws
            a1 = a1_min + (a1_max-a1_min)*torch.sigmoid(a1_raw)
            b1 = b1_min + (b1_max-b1_min)*torch.sigmoid(b1_raw)
            a2 = a2_min + (a2_max-a2_min)*torch.sigmoid(a2_raw)
            b2 = b2_min + (b2_max-b2_min)*torch.sigmoid(b2_raw)
            gammas = [F.softplus(g).clamp(max=gamma_max) for g in [gA1_raw,gB1_raw,gA2_raw,gB2_raw]]
            # enforce a1+a2 cap
            base_cap = float(CONFIG.get('base_asymptote_cap',80.0))
            base_sum = a1 + a2
            scale = torch.where(base_sum > base_cap,
                                base_cap / base_sum.clamp(min=1.0),
                                torch.tensor(1.0, device=x.device))
            a1,a2 = a1*scale, a2*scale
            return a1,b1,a2,b2,*gammas

        if sample_ids is not None:
            unique = list(set(sample_ids))
            cache = {}
            for uid in unique:
                idxs = [i for i,s in enumerate(sample_ids) if s==uid]
                f = idxs[0]
                raws = [self.param_networks[k](x[f:f+1]) for k in range(8)]
                mapped = map_heads(raws)
                cache_vec = torch.cat([m.flatten() for m in mapped])
                for i2 in idxs:
                    params[i2,:] = cache_vec
        else:
            raws = [self.param_networks[k](x) for k in range(8)]
            mapped = map_heads([r.squeeze() for r in raws])
            params[:,:] = torch.stack(mapped, dim=0).T

        params = project_params_to_caps(params,
                                        float(CONFIG.get('base_asymptote_cap',80.0)),
                                        float(CONFIG.get('total_asymptote_cap',95.0)),
                                        float(CONFIG.get('base_rate_cap',2.1)))
        return params

def generate_two_phase_recovery(time, catalyst, transition_time, params):
    """
    control(t)=a1(1-e^{-b1 t}) + a2(1-e^{-b2 t})
    Post-TT:
      amp_a1 = 1+gamma_a1 ; rate_b1 = 1+gamma_b1
      amp_a2 = 1+gamma_a2 ; rate_b2 = 1+gamma_b2
      recovery(t>=TT) = control(TT) +
        a1*amp_a1*(1 - exp(-(b1*rate_b1)*(t-TT))) +
        a2*amp_a2*(1 - exp(-(b2*rate_b2)*(t-TT)))
    Ensures monotone increments (gammas >=0).
    """
    time = time.to(params.device)
    a1 = params[:,0].unsqueeze(1)
    b1 = params[:,1].unsqueeze(1)
    a2 = params[:,2].unsqueeze(1)
    b2 = params[:,3].unsqueeze(1)
    gA1 = params[:,4].unsqueeze(1)
    gB1 = params[:,5].unsqueeze(1)
    gA2 = params[:,6].unsqueeze(1)
    gB2 = params[:,7].unsqueeze(1)

    exp1 = torch.exp(-b1 * time).clamp(min=1e-8,max=1.0)
    exp2 = torch.exp(-b2 * time).clamp(min=1e-8,max=1.0)
    control = a1*(1-exp1) + a2*(1-exp2)
    recovery = control.clone()

    tt = transition_time.squeeze()
    has_cat = torch.any(catalyst > 0).item() if catalyst is not None else True
    if has_cat:
        post_mask = (time >= tt)
        t_shift = torch.clamp(time - tt, min=0.0)
        amp_a1 = 1.0 + gA1
        rate_b1 = 1.0 + gB1
        amp_a2 = 1.0 + gA2
        rate_b2 = 1.0 + gB2
        exp1_post = torch.exp(-(b1*rate_b1)*t_shift).clamp(min=1e-8,max=1.0)
        exp2_post = torch.exp(-(b2*rate_b2)*t_shift).clamp(min=1e-8,max=1.0)
        incr = a1*amp_a1*(1-exp1_post) + a2*amp_a2*(1-exp2_post)
        control_tt = a1*(1-torch.exp(-b1*tt)) + a2*(1-torch.exp(-b2*tt))
        amplified = control_tt + incr
        recovery = torch.where(post_mask, amplified, control)

    total_cap = float(CONFIG.get('total_asymptote_cap',95.0))
    return recovery.clamp(min=0.0, max=total_cap)

def compute_sample_weights_per_time(time_steps, device):
    weights = []
    num_max_days_standard = 1600
    full_days = np.linspace(0, num_max_days_standard - 1, num_max_days_standard)
    full_weight_curve = np.linspace(3.0, 1.0, num_max_days_standard) # try to even out the influence of early time points
    
    for time_array in time_steps:
        if isinstance(time_array, torch.Tensor):
            time_array = time_array.cpu().numpy()
        sample_weights = np.interp(time_array, full_days, full_weight_curve)
        weights.append(torch.tensor(sample_weights, dtype=torch.float32, device=device))
    
    return weights

def compute_sample_weights(time_steps, device):
    weights = []
    for time_array in time_steps:
        if isinstance(time_array, torch.Tensor):
            time_array = time_array.cpu().numpy()
        
        full_weight_curve = np.linspace(3.0, 1.0, len(time_array))
        sample_weights = full_weight_curve
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
    patience = config.get('plateau_patience', 200)

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
    feature_weight_signs = get_feature_weight_signs(config, num_cols).to(device)
    feature_weight_mags  = get_feature_weight_magnitudes(config, num_cols, zero_policy='one').to(device)

    fold_model = AdaptiveTwoPhaseRecoveryModel(
        total_features=len(num_cols),
        hidden_dim=config.get('pytorch_hidden_dim', 128),
        dropout_rate=config.get('pytorch_dropout_rate', 0.33),
        init_mode=config.get('init_mode', 'kaiming'),
        feature_weight_signs=feature_weight_signs,
        # feature_weight_magnitudes=feature_weight_mags
    ).to(device)
        
    
    optimizer = optim.AdamW(fold_model.parameters(), lr=learning_rate, weight_decay=config.get('pytorch_weight_decay', 1e-4))
    scheduler = ReduceLROnPlateau(optimizer, 
                                  mode='min', 
                                  factor=config['adaptive_lr']['plateau_factor'], 
                                  patience=config['adaptive_lr']['plateau_patience']//2, 
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
                time_sample, catalyst_sample, tt_sample, params,
            )
            
            target_recovery = y_train[j].unsqueeze(0)
            weights_sample = weights_train[j].unsqueeze(0)
            
            # Compute MSE loss
            mse_criterion = nn.MSELoss(reduction='none')
            per_element_loss = mse_criterion(predicted_recovery, target_recovery)
            weighted_loss = weights_sample * per_element_loss
            mse_loss_value = weighted_loss.mean()

            # Post-TT monotonicity penalty (only penalize decreasing segments after TT)
            mono_w = float(CONFIG.get('post_tt_monotone_penalty_weight', 0.0))
            if mono_w > 0:
                # deltas over time
                deltas = predicted_recovery[:, 1:] - predicted_recovery[:, :-1]
                # mask only points after TT (align to deltas’ indices)
                post_mask = (time_sample[:, 1:] >= tt_sample).to(deltas.dtype)
                neg = torch.relu(-(deltas * post_mask))  # only negative slopes post-TT
                post_tt_mono_pen = mono_w * neg.mean()
            else:
                post_tt_mono_pen = torch.tensor(0.0, device=device)

            # Physics loss (currently not used)
            physics_loss = torch.tensor(0.0, device=device)

            # Monotonic (feature sign) penalty
            if config.get('column_tests_feature_weighting', {}).get('use_penalty_loss', False):
                monotonic_penalty = compute_monotonic_penalty(
                    fold_model, 
                    lambda_penalty=config.get('monotonic_penalty_weight', 0.1)
                )
            else:
                monotonic_penalty = torch.tensor(0.0, device=device)

            # Gamma→0 for control rows
            gamma_zero_w = float(CONFIG.get('gamma_zero_penalty', 0.0))
            if gamma_zero_w > 0 and params.size(1) >= 8:
                is_catalyzed = torch.any(catalyst_sample > 0).item()
                if not is_catalyzed:
                    gamma_pen = gamma_zero_w * params[:,4:8].abs().mean()
                else:
                    gamma_pen = torch.tensor(0.0, device=device)
            else:
                gamma_pen = torch.tensor(0.0, device=device)

            loss = mse_loss_value + physics_loss + monotonic_penalty + gamma_pen + post_tt_mono_pen
            
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
                    time_sample, catalyst_sample, tt_sample, params,
                )
                target_recovery = y_train[j].unsqueeze(0)
                weights_sample = weights_train[j].unsqueeze(0)
                per_element_loss_val = mse_criterion(predicted_recovery, target_recovery)
                weighted_loss_val = weights_sample * per_element_loss_val
                weighted_loss_val = weighted_loss_val
                val_losses.append(weighted_loss_val.mean().item())
        
        if val_losses:
            val_losses.sort(reverse=True)  # highest losses first
            num_top = max(1, len(val_losses) // 2) # select the 50% worst losses
            avg_val_loss = sum(val_losses[:num_top]) / num_top
        else:
            avg_val_loss = float('inf')

        scheduler.step(avg_val_loss)

        # Adaptive penalty weight adjustment (using CONFIG settings)
        if config.get('column_tests_feature_weighting', {}).get('use_penalty_loss', False):
            adaptive_config = config.get('adaptive_penalty', {})
            if adaptive_config.get('enabled', True):
                violations, total_constraints, violation_ratio = check_constraint_violations(fold_model)
                
                current_penalty_weight = config.get('monotonic_penalty_weight', 0.01)
                
                # Get thresholds from config
                high_threshold = adaptive_config.get('violation_threshold_high', 0.10)
                low_threshold = adaptive_config.get('violation_threshold_low', 0.02)
                increase_factor = adaptive_config.get('increase_factor', 1.2)
                decrease_factor = adaptive_config.get('decrease_factor', 0.9)
                min_weight = adaptive_config.get('min_penalty_weight', 0.001)
                max_weight = adaptive_config.get('max_penalty_weight', 1.0)
                
                # Adjust based on violations
                if violation_ratio > high_threshold:
                    new_penalty_weight = current_penalty_weight * increase_factor
                    config['monotonic_penalty_weight'] = min(new_penalty_weight, max_weight)
                    if epoch % 100 == 0:
                        print(f"{test_sample_id} - ⬆️  Increasing penalty: {current_penalty_weight:.4f} → {config['monotonic_penalty_weight']:.4f}")
                
                elif violation_ratio < low_threshold and current_penalty_weight > min_weight:
                    new_penalty_weight = current_penalty_weight * decrease_factor
                    config['monotonic_penalty_weight'] = max(new_penalty_weight, min_weight)
                    if epoch % 100 == 0:
                        print(f"{test_sample_id} - ⬇️  Decreasing penalty: {current_penalty_weight:.4f} → {config['monotonic_penalty_weight']:.4f}")
                
                # Print status
                if epoch % 100 == 0:
                    if violations > 0:
                        print(f"{test_sample_id} - ⚠️  Constraints: {violations}/{total_constraints} violated ({violation_ratio*100:.1f}%)")
                    else:
                        print(f"{test_sample_id} - ✅ Constraints: All {total_constraints} satisfied")

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
                time_sample, catalyst_sample, tt_sample, params,
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
            sample_id_params[unique_sample_id] = base_params[0]
            
        for j, unique_sample_id in enumerate(test_sample_ids):
            x_sample = X_test[j:j+1]
            catalyst_sample = catalyst_test[j].unsqueeze(0)
            time_sample = time_test[j].unsqueeze(0)
            tt_sample = tt_test[j]
            target_recovery = y_test[j]
            col_id = sample_col_ids[test_mask][j]
            if unique_sample_id not in sample_id_params:
                # fallback: infer params directly
                sample_params = fold_model(x_sample, catalyst_sample, [unique_sample_id])
            else:
                sample_params = sample_id_params[unique_sample_id].unsqueeze(0)
            
            predicted_recovery = generate_two_phase_recovery(
                time_sample, catalyst_sample, tt_sample, sample_params,
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
                'a1_pred': sample_params[0,0].item(),
                'b1_pred': sample_params[0,1].item(),
                'a2_pred': sample_params[0,2].item(),
                'b2_pred': sample_params[0,3].item(),
                'gamma_a1_pred': sample_params[0,4].item(),
                'gamma_b1_pred': sample_params[0,5].item(),
                'gamma_a2_pred': sample_params[0,6].item(),
                'gamma_b2_pred': sample_params[0,7].item(),
                'transition_time': tt_sample.item() if isinstance(tt_sample, torch.Tensor) else tt_sample,
                })
    
    # Plot validation recovery curves only if generate_plots is True
    if config.get('generate_plots', False):
        plt.figure(figsize=(7.25, 5), dpi=300)
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
                    time_sample, catalyst_sample, tt_sample, sample_params,
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
                
                total_cap = float(CONFIG.get('total_asymptote_cap', 95.0))
                if is_catalyzed:
                    plt.vlines(x=tt_np, ymin=0, ymax=100, color=colors[is_catalyzed],
                              linestyle='--', alpha=0.7, label=f'Transition Time ({col_id})')
        
        plt.title(f"Validation Recovery Curves - {test_sample_id}\nVal Loss: {best_val_loss:.4f}")
        plt.xlabel("Cumulative Lixiviant (m3/ton ore)")
        plt.ylabel("Cu Recovery (%)")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.ylim(0, 100)
        plt.xlim(left=0) # int(np.ceil(max(tt_tensor)/30)*30)
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
        best_configs=all_best_configs,
        num_cols=num_cols
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
    z_score = float(config.get('z_score', 1.645)) * float(config.get('uncertainty_scale', 1.0))

    colors = {True: 'darkorange', False: 'royalblue'}  # True for catalyzed, False for control
    
    for sample_id in unique_sample_ids:
        plt.figure(figsize=(7.25, 5), dpi=300)
        
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
            total_cap = float(CONFIG.get('total_asymptote_cap', 95.0))
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
                    plt.vlines(x=tt_value, ymin=0, ymax=100, color=colors[is_catalyzed], 
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
                'uncertainty': float(np.mean(uncertainty[:len(original_time)])) if isinstance(uncertainty, np.ndarray) else float(uncertainty),
                'a1': params_mean[0,0],
                'b1': params_mean[0,1],
                'a2': params_mean[0,2],
                'b2': params_mean[0,3],
                'gamma_a1': params_mean[0,4],
                'gamma_b1': params_mean[0,5],
                'gamma_a2': params_mean[0,6],
                'gamma_b2': params_mean[0,7],
                'transition_time': tt_sample.item() if isinstance(tt_sample, torch.Tensor) else tt_sample
                }
            summary.append(summary_entry)
        
        # Finalize plot
        plt.title(f"Ensemble Predictions - {sample_id}\nExtended to {max_days} [m3 Lix / ton ore]")
        plt.xlabel("Cumulative Lixiviant (m3/ton ore)")
        plt.ylabel("Cu Recovery (%)")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.ylim(0, 100)
        plt.xlim(left=0) # int(np.ceil(max(tt_tensor)/30)*30)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"ensemble_predictions_{sample_id}_withoutReactors.png"), dpi=300)
        plt.close()
        print(f"Saved ensemble prediction plot for {sample_id}")

    
    # Save ensemble_prediction_summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(save_dir, 'plots', 'ensemble_prediction_summary_withoutReactors.csv'), index=False)
    print(f"Saved ensemble_prediction_summary with {len(summary_df)} entries")
    
    return summary_df

def calibrate_uncertainty_scale(ensemble, X_list, y_list, time_list, catalyst_list, tt_list, sample_ids_list, target_cov=CONFIG.get('target_predictive_interval_coverage', 0.9)):
    """
    Compute a scalar c so that P(|y - mu| <= 1.645 * c * sigma) ~= target_cov on validation data.
    Returns c (float). Uses weighted mean and weighted std from ensemble.
    """
    ratios = []
    for i in range(len(X_list)):
        x = X_list[i:i+1] if isinstance(X_list, torch.Tensor) else X_list[i]
        t = time_list[i].unsqueeze(0)
        cat = catalyst_list[i].unsqueeze(0)
        tt = tt_list[i]
        sid = [sample_ids_list[i]]
        with torch.no_grad():
            mu, sig, _ = ensemble.predict_with_params_and_uncertainty(x, cat, tt, t, sid)
        mu = np.asarray(mu).squeeze()
        sig = np.asarray(sig).squeeze()
        y = y_list[i].cpu().numpy()
        if mu.ndim > 1: mu = mu.flatten()
        if y.ndim > 1: y = y.flatten()
        if sig.shape != mu.shape:
            continue
        denom = np.maximum(sig, 1e-6)
        r = np.abs(y - mu) / denom  # pointwise ratio
        ratios.extend(r.tolist())
    if not ratios:
        return 1.0
    # Find c so that 1.645 * c is the target coverage threshold
    q = np.quantile(np.asarray(ratios), target_cov)
    return float(q / float(CONFIG.get('z_score', 1.645)))

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
    
    df_columns_filtered = filter_column_dataset_by_config(df_model_recCu_catcontrol_projects, CONFIG)
    df_columns_aug = df_columns_filtered.copy()
    
    X_tensor, y_tensor, time_tensor, catalyst_tensor, tt_tensor, sample_ids, sample_col_ids, feature_weights, num_cols, scaler_X, out_df_unscaled = prepare_column_train_data(
        df=df_columns_aug, 
        config=CONFIG, 
        output_type='averaged', 
        fill_noncat_averages=False
    )
    out_df_unscaled.to_csv(os.path.join(folder_path_save, 'plots', 'processed_data_unscaled.csv'), index=False)
    
    print("\nStep 2: Training Adaptive two phase recovery Model")
    print("=" * 60)

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

    # Calibrate to target coverage on the training/validation samples you just evaluated
    try:
        cal_scale = calibrate_uncertainty_scale(
            scaling_models, X_tensor, y_tensor, time_tensor, catalyst_tensor, tt_tensor, sample_ids,
            target_cov=final_config['target_predictive_interval_coverage']
        )
        print(f"Calibrated uncertainty_scale: {cal_scale:.3f}")
        final_config['uncertainty_scale'] = cal_scale
    except Exception as e:
        print(f"Uncertainty calibration skipped ({e}). Using scale=1.0")

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
        max_days=30,
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
        'scaler_X': scaler_X,
        'uncertainty_scale': final_config.get('uncertainty_scale', 1.0)
    }, os.path.join(folder_path_save, 'AdaptiveTwoPhaseModel_withoutReactors.pt'))
    
    print("\nReactor Scaling Model training completed successfully!")
    print("All results and plots have been saved.")

    print("\nComputing SHAP explanations for feature impacts on predicted parameters...")
    catalyst_present = np.array([torch.any(cat > 0).item() for cat in catalyst_tensor])
    non_catalyst_indices = np.where(~catalyst_present)[0]

    print(f"  Total samples: {len(X_tensor)}")

    # Model wrapper for base parameters (no catalyst)
    class BaseModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            catalyst_dummy = torch.zeros(x.shape[0], 1, device=x.device)
            p = self.model(x, catalyst_dummy)
            return p[:, :8]  # a1,b1,a2,b2,gA1,gB1,gA2,gB2

    # Compute SHAP for base parameters (a1, b1, a2, b2)
    print("\n  Computing SHAP for base parameters (a1, b1, a2, b2)...")
    if len(non_catalyst_indices) > 0:
        background_indices_base = random.sample(
            list(non_catalyst_indices), 
            min(CONFIG['shap_background_size'], len(non_catalyst_indices))
        )
        explain_indices_base = random.sample(
            list(non_catalyst_indices), 
            min(CONFIG['shap_sample_size'], len(non_catalyst_indices))
        )
        X_background_base = X_tensor[background_indices_base]
        X_explain_base = X_tensor[explain_indices_base]
        
        all_shap_values_base = []
        for idx, model in enumerate(scaling_models.models):
            wrapped_model = BaseModelWrapper(model).to(device)
            explainer = shap.DeepExplainer(wrapped_model, X_background_base)
            shap_values_model = explainer.shap_values(X_explain_base, check_additivity=False)
            if isinstance(shap_values_model, np.ndarray) and shap_values_model.ndim == 3:
                shap_values_model = [shap_values_model[..., k] for k in range(shap_values_model.shape[-1])]
            all_shap_values_base.append(shap_values_model)
        
        # Weighted average for base parameters
        weighted_shap_base = []
        for output_dim in range(5):  # a1, b1, a2, b2, gamma
            output_shap = sum(
                scaling_models.weights[m] * all_shap_values_base[m][output_dim] 
                for m in range(len(scaling_models.models))
            )
            weighted_shap_base.append(output_shap)
    else:
        weighted_shap_base = [np.zeros((1, len(num_cols)))] * 4
        X_explain_base = X_tensor[:1]  # Dummy for consistency
        print("  ⚠️  No non-catalyst samples found for base parameters!")

    # Combine SHAP values for all 4 parameters
    weighted_shap = weighted_shap_base

    # Parameter names
    param_names = ['a1','b1','a2','b2','gamma_a1','gamma_b1','gamma_a2','gamma_b2']

    # Aggregate: Mean absolute SHAP per feature per parameter (overall impact)
    importance_dict = {}
    for output_dim, param in enumerate(param_names):
        mean_abs_shap = np.mean(np.abs(weighted_shap[output_dim]), axis=0)  # [num_features]
        importance_dict[param] = mean_abs_shap

    # Create DataFrame: rows=features, columns=parameters, values=mean abs SHAP
    importance_df = pd.DataFrame(importance_dict, index=num_cols)  # num_cols are feature names
    importance_df.to_csv(os.path.join(folder_path_save, 'plots', 'feature_impact_on_parameters.csv'))
    print(f"Saved feature impact CSV (mean abs SHAP) to {folder_path_save}/plots/feature_impact_on_parameters.csv")

    # Visualizations
    plots_dir = os.path.join(folder_path_save, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Bar plots: Importance per parameter
    for output_dim, param in enumerate(param_names):
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(importance_df[param])[::-1]  # Descending order
        plt.barh(np.array(num_cols)[sorted_idx], importance_df[param].values[sorted_idx])
        plt.xlabel('Mean Absolute SHAP Value (Impact)')
        plt.title(f'Feature Impact on {param}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'shap_importance_{param}.png'), dpi=300)
        plt.close()

    # Beeswarm plots for distribution (one per parameter)
    # Use appropriate X_explain for each parameter group
    for output_dim, param in enumerate(param_names):
        shap_for_plot = weighted_shap[output_dim]
        X_for_plot = X_explain_base.cpu().numpy()
        if shap_for_plot.shape[0] > 0 and np.any(np.abs(shap_for_plot) > 1e-10):
            shap.summary_plot(shap_for_plot, features=X_for_plot, feature_names=num_cols, show=False)
            plt.title(f'SHAP Beeswarm for {param}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'shap_beeswarm_{param}.png'), dpi=300)
            plt.close()
        else:
            print(f"Skipping beeswarm plot for {param} (no SHAP values)")

    print(f"\nSHAP analysis complete! Plots saved to {plots_dir}")

    # Generate overall summary importance plots per predictor variable
    overall_importance = importance_df.sum(axis=1) / importance_df.sum(axis=1).sum() * 100    
    overall_plot = pd.DataFrame({'importance': overall_importance}).sort_values('importance', ascending=False)    
    ax = sns.barplot(data=overall_plot, x='importance', y=overall_plot.index, palette='viridis')
    for i, patch in enumerate(ax.patches):
        ax.text(
            patch.get_width() - 0.5,  # Adjust offset to position inside the bar
            patch.get_y() + patch.get_height() / 2,
            f"{patch.get_width():.0f}%",
            ha='right',  # Right-align the text
            va='center',
            color='white'  # Use white for better visibility on dark bars, or adjust
        )
    plt.title(f'SHAP overall % importance for all parameters')
    plt.tight_layout()
    plt.ylabel('Feature')
    plt.xlabel('% importance')
    plt.savefig(os.path.join(plots_dir, f'percentual_feature_importance_overall.png'), dpi=300)
    plt.show()
    plt.close()

    # Generate a-control params summary importance plots per predictor variable
    a_control_cols = [col for col in importance_df.columns if col.startswith('a1') or col.startswith('a2')]
    importance_df_a = importance_df[a_control_cols]
    overall_importance = importance_df_a.sum(axis=1) / importance_df_a.sum(axis=1).sum() * 100    
    overall_plot = pd.DataFrame({'importance': overall_importance}).sort_values('importance', ascending=False)    
    ax = sns.barplot(data=overall_plot, x='importance', y=overall_plot.index, palette='viridis')
    for i, patch in enumerate(ax.patches):
        ax.text(
            patch.get_width() - 0.5,  # Adjust offset to position inside the bar
            patch.get_y() + patch.get_height() / 2,
            f"{patch.get_width():.0f}%",
            ha='right',  # Right-align the text
            va='center',
            color='white'  # Use white for better visibility on dark bars, or adjust
        )
    plt.title(f'SHAP % importance for Control a-params')
    plt.tight_layout()
    plt.ylabel('Feature')
    plt.xlabel('% importance')
    plt.savefig(os.path.join(plots_dir, f'percentual_feature_importance_a-control-params.png'), dpi=300)
    plt.show()
    plt.close()

    # Generate b-control params summary importance plots per predictor variable
    b_control_cols = [col for col in importance_df.columns if col.startswith('b1') or col.startswith('b2')]
    importance_df_b = importance_df[b_control_cols]
    overall_importance = importance_df_b.sum(axis=1) / importance_df_b.sum(axis=1).sum() * 100    
    overall_plot = pd.DataFrame({'importance': overall_importance}).sort_values('importance', ascending=False)    
    ax = sns.barplot(data=overall_plot, x='importance', y=overall_plot.index, palette='viridis')
    for i, patch in enumerate(ax.patches):
        ax.text(
            patch.get_width() - 0.5,  # Adjust offset to position inside the bar
            patch.get_y() + patch.get_height() / 2,
            f"{patch.get_width():.0f}%",
            ha='right',  # Right-align the text
            va='center',
            color='white'  # Use white for better visibility on dark bars, or adjust
        )
    plt.title(f'SHAP % importance for Control for b-params')
    plt.tight_layout()
    plt.ylabel('Feature')
    plt.xlabel('% importance')
    plt.savefig(os.path.join(plots_dir, f'percentual_feature_importance_b-control-params.png'), dpi=300)
    plt.show()
    plt.close()


# %%
