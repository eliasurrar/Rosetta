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
from statistics import NormalDist

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

# ---------------------------
# Reproducibility helpers
# ---------------------------
def enable_torch_determinism(deterministic=True):
    """Toggle deterministic algorithms where supported."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except Exception as exc:
        if deterministic:
            try:
                torch.use_deterministic_algorithms(deterministic)
            except Exception:
                print(f"Warning: deterministic algorithms not fully enforced ({exc})")


def set_all_seeds(seed=None, deterministic=True):
    """Seed Python, NumPy, and PyTorch RNGs; optionally enforce deterministic ops."""
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        enable_torch_determinism(True)

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
    'pytorch_dropout_rate': 0.30,
    'pytorch_timeout': 300,     # Timeout in seconds for PyTorch training (5 minutes)
    'min_pre_catalyst_points': 9,  # Minimum number of pre-catalyst points to consider a sample valid
    'pytorch_seeds': [1, 11, 21],  # Multi-seed runs to smooth randomness
    'use_deterministic_algorithms': True,  # Try to enforce deterministic ops where supported

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
    'plot_font_family': 'Calibri',  # Global font for all Matplotlib/Seaborn plots
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
        'weights': { # title and effect per parameter predicted in this order: a1, b1, a2, b2, a3, b3, a4, b4; NaN for null impact on predictor parameter
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
            'grouped_acid_generating_sulfides': ['Acid Generating Sulphides (%)', np.nan, 0.64, np.nan, 0.48, np.nan, 0.36, np.nan, 0.48], # 0, 0.64, 0, 0.48, 0, 0.36, 0, 0.48
            # 'grouped_gangue_sulfides': ['Gangue Sulphides (%)', 0.0], # removed because of low impact on feature importance plots
            'grouped_gangue_silicates': ['Gangue Silicates (%)', np.nan, 0, np.nan, 0, np.nan, 0, np.nan, 0], # 0, 0, 0, 0, 0, 0, 0, 0
            # 'grouped_clays_and_micas': ['Clays and Micas (%)', 0.0],
            # 'grouped_accesory_silicates': ['Accesory Silicates (%)', 0.0],
            # 'grouped_sulfates': ['Sulfates (%)', 0.0],
            'grouped_fe_oxides': ['Fe Oxides (%)', -0.41, -1.16, -0.16, -0.39, -0.92, -0.36, -0.98, -0.39], #-0.41, -1.16, -0.16, -0.39, -0.92, -0.36, -0.98, -0.39
            # 'grouped_accessory_misc': ['Accessory Misc (%)', 0.0],
            'grouped_carbonates': ['Carbonates (%)', -1.07, -1.41, -1.10, -0.46, -1.08, -1.02, -0.77, -1.08], # -1.07, 1.41, -1.10, 0.46, -1.08, 1.02, -0.77, 1.08 (according to literature, all negative impacts)
            # 'grouped_accessory_minerals': ['Accessory Minerals (%)', 0.0],
            'cumulative_lixiviant_m3_t': ['Cumulative Lixiviant (kg/t)', 0, 0, 0, 0, 0, 0, 0, 0],
        },
    },
    
    # Monotonic constraint penalty weight
    'monotonic_penalty_weight': 0.01,  # Weight for soft constraint penalty (they are scaled afterwards)
    'control_floor_weight': 25.0,  # soft penalty to keep catalyzed >= control
    
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
    'prediction_interval_bounds': {  # bounds for interactive PI slider (%)
        'min_pct': 50.0,
        'max_pct': 99.0,
        'default_pct': 90.0,
    },
    'loss_scale': 1.0,          # scales train/val losses for readability (keeps ordering)
    # ±1 standard deviation from the mean covers approximately 68% of the data.#
    # ±1.96 standard deviations from the mean covers approximately 95% of the data.
    # ±3 standard deviations from the mean covers approximately 99.7% of the data.
    # 1.645 standard deviations from the mean covers approximately 90% of the data.
    # 1.282 standard deviations from the mean covers approximately 80% of the data.

    'special_feats': {
        'dynamic': ['leach_duration_days', 'cumulative_lixiviant_m3_t', 'cumulative_catalyst_addition_kg_t'],
        'time_feat': ['leach_duration_days', 'cumulative_lixiviant_m3_t'],
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
    'total_rate_cap': 7.0,      # applies to b1 + b2 + b3 + b4 (control+catalyzed)
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

    # Catalyst → kinetics modulation (affects b3,b4 during generation/training)
    'cat_effect_power': 0.7,    # ce^power, ce = cat/(cat+1)
    'cat_rate_gain_b3': 0.3,    # rate multiplier = 1 + gain * ce^power
    'cat_rate_gain_b4': 0.1,    # tune per dataset; set 0.0 to disable
    'cat_additional_scale': 0.5,  # scales catalyst add-on; 1.0 = no change, <1 dampens
}

# Apply global plotting defaults from CONFIG so fonts can be changed in one place
_plot_font_family = CONFIG.get('plot_font_family')
if _plot_font_family:
    plt.rcParams['font.family'] = _plot_font_family
    plt.rcParams['font.sans-serif'] = [_plot_font_family]

def normal_coverage_from_z(z_value: float) -> float:
    """Return two-tailed coverage probability for a given z-score under N(0,1)."""
    try:
        return float(2.0 * NormalDist().cdf(float(z_value)) - 1.0)
    except Exception:
        return float('nan')

def project_params_to_caps(params: torch.Tensor,
                           base_cap: float,
                           total_cap: float,
                           base_rate_cap: float,
                           total_rate_cap: float) -> torch.Tensor:
    """
    Projection enforcing amplitude and kinetic constraints.

    Amplitude caps (apply by scaling catalyst amplitudes only for total):
      a1 + a2 <= base_cap
      a1 + a2 + a3 + a4 <= total_cap   (scales a3,a4 only; a1,a2 already projected)

    Rate caps (apply by scaling catalyst rates only for total):
      b1 + b2 <= base_rate_cap
      b1 + b2 + b3 + b4 <= total_rate_cap (scales b3,b4 only; preserves control kinetics)

    Hard ratio constraints (from CONFIG['ratio_constraints']):
      b3 / b1 <= max_b3_over_b1 (default 0.1)
      a1 / b1 >= min_a1_over_b1 (default 10) via boosting a1 (within remaining amplitude room) or reducing b1

    NaN handling:
      a3,a4,b3,b4 may be NaN for control rows; they are left unchanged (only finite catalyst params scaled).

    Input / Output:
      params: [B, >=8] tensor -> (a1,b1,a2,b2,a3,b3,a4,b4[, extra...])
      returns projected tensor with same shape; any extra columns are passed through unchanged.

    Note:
      Soft “potentiate” targets (a1/b1 > 15, a1/b3 > 50, a2/b1 > 20) and soft “avoid” penalties
      (b3/b1 > 0.1, a1/b1 < 10) are applied via compute_ratio_penalty() in the training loop, not here.
    """
    if params.ndim == 1:
        params = params.unsqueeze(0)

    extra_cols = params[:, 8:] if params.shape[1] > 8 else None

    a1, b1, a2, b2, a3, b3, a4, b4 = [params[:, i] for i in range(8)]

    # ----- Amplitude base cap -----
    base_sum = a1 + a2
    scale_base_amp = torch.where(
        base_sum > base_cap,
        base_cap / base_sum.clamp(min=1e-6),
        torch.ones_like(base_sum)
    )
    a1 = a1 * scale_base_amp
    a2 = a2 * scale_base_amp

    # ----- Amplitude total cap (scale catalyst amplitudes only) -----
    a3_nan = torch.isnan(a3)
    a4_nan = torch.isnan(a4)
    a3n = torch.nan_to_num(a3, nan=0.0)
    a4n = torch.nan_to_num(a4, nan=0.0)

    remain_amp = (total_cap - (a1 + a2)).clamp(min=0.0)
    cat_amp_sum = a3n + a4n
    scale_cat_amp = torch.where(
        cat_amp_sum > remain_amp,
        remain_amp / cat_amp_sum.clamp(min=1e-6),
        torch.ones_like(cat_amp_sum)
    )
    a3n = a3n * scale_cat_amp
    a4n = a4n * scale_cat_amp

    a3 = torch.where(a3_nan, a3, a3n)
    a4 = torch.where(a4_nan, a4, a4n)

    # ----- Rate base cap (scale both b1,b2) -----
    rate_base_sum = b1 + b2
    scale_base_rate = torch.where(
        rate_base_sum > base_rate_cap,
        base_rate_cap / rate_base_sum.clamp(min=1e-9),
        torch.ones_like(rate_base_sum)
    )
    b1 = b1 * scale_base_rate
    b2 = b2 * scale_base_rate

    # ----- Rate total cap (scale catalyst rates only) -----
    b3_nan = torch.isnan(b3)
    b4_nan = torch.isnan(b4)
    b3n = torch.nan_to_num(b3, nan=0.0)
    b4n = torch.nan_to_num(b4, nan=0.0)

    remain_rate = (total_rate_cap - (b1 + b2)).clamp(min=0.0)
    cat_rate_sum = b3n + b4n
    scale_cat_rate = torch.where(
        cat_rate_sum > remain_rate,
        remain_rate / cat_rate_sum.clamp(min=1e-9),
        torch.ones_like(cat_rate_sum)
    )
    b3n = b3n * scale_cat_rate
    b4n = b4n * scale_cat_rate

    b3 = torch.where(b3_nan, b3, b3n)
    b4 = torch.where(b4_nan, b4, b4n)

    projected = torch.stack([a1, b1, a2, b2, a3, b3, a4, b4], dim=1)
    if extra_cols is not None:
        projected = torch.cat([projected, extra_cols], dim=1)
    return projected

def compute_ratio_penalty(params: torch.Tensor) -> torch.Tensor:
    """
    Soft penalties to potentiate:
      a1/b1 >= target_a1_over_b1
      a1/b3 >= target_a1_over_b3 (only where b3 valid)
      a2/b1 >= target_a2_over_b1
    Avoid (soft):
      b3/b1 <= max_b3_over_b1
      a1/b1 not below min_a1_over_b1 (already hard—small penalty if it happens)
    """
    rcfg = CONFIG.get('ratio_constraints', {})
    if not rcfg.get('enable', False):
        return torch.tensor(0.0, device=params.device)

    # Split parameters
    a1, b1, a2, b2, a3, b3, a4, b4 = [params[:, i] for i in range(8)]
    eps = 1e-6

    def masked_mean(tensor, mask):
        if mask is None:
            mask = torch.ones_like(tensor, dtype=torch.bool)
        tensor = tensor[mask]
        return tensor.mean() if tensor.numel() > 0 else torch.tensor(0.0, device=params.device)

    penalty = torch.tensor(0.0, device=params.device)
    pot_w = float(rcfg.get('potentiate_weight', 0.0))
    avoid_w = float(rcfg.get('avoid_weight', 0.0))

    # Ratios for reuse
    a1_over_b1 = a1 / (b1.abs() + eps)
    a2_over_b1 = a2 / (b1.abs() + eps)
    b3_over_b1 = b3 / (b1.abs() + eps)

    # Potentiate targets (encourage higher ratios)
    if pot_w > 0.0:
        tgt_a1_b1 = rcfg.get('target_a1_over_b1')
        if tgt_a1_b1 is not None:
            penalty += pot_w * torch.relu(tgt_a1_b1 - a1_over_b1).mean()

        tgt_a1_b3 = rcfg.get('target_a1_over_b3')
        if tgt_a1_b3 is not None:
            mask = torch.isfinite(b3)
            penalty += pot_w * masked_mean(torch.relu(tgt_a1_b3 - (a1 / (b3.abs() + eps))), mask)

        tgt_a2_b1 = rcfg.get('target_a2_over_b1')
        if tgt_a2_b1 is not None:
            penalty += pot_w * torch.relu(tgt_a2_b1 - a2_over_b1).mean()

    # Avoid forbidden ratios (discourage high b3/b1 or very low a1/b1)
    if avoid_w > 0.0:
        max_b3_b1 = rcfg.get('max_b3_over_b1')
        if max_b3_b1 is not None:
            mask = torch.isfinite(b3)
            penalty += avoid_w * masked_mean(torch.relu(b3_over_b1 - max_b3_b1), mask)

        min_a1_b1 = rcfg.get('min_a1_over_b1')
        if min_a1_b1 is not None:
            penalty += avoid_w * torch.relu(min_a1_b1 - a1_over_b1).mean()

    return penalty

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
        # [256, 0.30, 'kaiming', 'reduce_on_plateau'] or
        # [128, 0.2, 'kaiming', 'cosine_annealing']
        'pytorch_hidden_dim': [128], # 64 removed because none was optimal
        'pytorch_dropout_rate': [0.30],
        'init_mode': ['kaiming'], #  'xavier' and 'normal' removed as only 2 samples used it
        'scheduler_type': ['reduce_on_plateau'], #, 'cosine_annealing']
    }

    combinations = list(itertools.product(
        hyperparameter_space['pytorch_hidden_dim'],
        hyperparameter_space['pytorch_dropout_rate'],
        hyperparameter_space['init_mode'],
        hyperparameter_space['scheduler_type']
    ))

    # If only one combination, skip search and return that config directly
    if len(combinations) == 1:
        hidden_dim, dropout_rate, init_mode, scheduler_type = combinations[0]
        config_mod = copy.deepcopy(CONFIG)
        config_mod['pytorch_hidden_dim'] = hidden_dim
        config_mod['pytorch_dropout_rate'] = dropout_rate
        config_mod['init_mode'] = init_mode
        config_mod['adaptive_lr']['scheduler_type'] = scheduler_type
        config_mod['skip_hypersearch'] = True
        config_mod['generate_plots'] = False
        # mimic search outputs
        return config_mod, [{
            'pytorch_hidden_dim': hidden_dim,
            'pytorch_dropout_rate': dropout_rate,
            'init_mode': init_mode,
            'scheduler_type': scheduler_type,
            'val_loss': np.nan,
        }]

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
        axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_title('Predicted vs Actual (Train Set)', fontsize=10, fontweight="bold")
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].tick_params(labelsize=7)

    # Predicted vs Actual (Test Set)
    axes[0, 1].scatter(y_test, y_test_pred, edgecolors=(0, 0, 0), alpha=0.4)
    if y_test.size > 0:
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
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
            # color="firebrick",
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


def get_time_axis_label(config):
    """Return an x-axis label based on the configured time feature."""
    config = config or {}
    time_feat = config.get('special_feats', {}).get('time_feat', ['leach_duration_days'])
    if isinstance(time_feat, (list, tuple)):
        time_feat = time_feat[0] if len(time_feat) > 0 else 'leach_duration_days'
    return "Cumulative Lixiviant (m3/ton ore)" if time_feat == "cumulative_lixiviant_m3_t" else "Leach Duration (days)"


#%% ---------------------------
# Step 1: Dataset preparation
# ---------------------------


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
    """
    Returns |weight| per feature/parameter (F x 8) for training-time scaling.
    zero_policy:
      'one'     -> zeros become 1.0 (use feature, just unconstrained) [DEFAULT]
      'drop'    -> zeros stay 0.0 (drop contribution)
      'epsilon' -> zeros become small epsilon (very small influence)
    """
    W = config.get('column_tests_feature_weighting', {}).get('weights', {})
    dyn = set(config.get('special_feats', {}).get('dynamic', []))
    rows = []
    eps = 1e-3
    for f in feature_names:
        if f in dyn or f not in W or len(W[f]) < 9:
            rows.append([1.0]*8)
            continue
        vals = []
        for v in W[f][1:9]:
            if pd.isna(v):
                vals.append(0.0)  # forced-null predictors → zero magnitude
            else:
                vals.append(abs(float(v)))
        if zero_policy == 'one':
            vals = [1.0 if v == 0.0 else v for v in vals]
        elif zero_policy == 'epsilon':
            vals = [eps if v == 0.0 else v for v in vals]
        # 'drop' keeps zeros
        rows.append(vals)
    return torch.tensor(rows, dtype=torch.float32)

def get_feature_weight_signs(config, feature_names):
    """
    Extract parameter-specific weight signs from CONFIG.
    
    Returns a matrix of shape [num_features, 8] where each row contains the signs
    for a feature's effect on each parameter (a1, b1, a2, b2, a3, b3, a4, b4).
    
    np.nan (or None) in CONFIG['column_tests_feature_weighting']['weights'] forces
    a null impact; it is treated as "no sign" here and handled separately.
    
    Args:
        config: Configuration dictionary
        feature_names: List of feature names
    
    Returns:
        torch.Tensor: Shape [num_features, 8] with values in {-1, 0, 1}
    """
    col_config = config.get('column_tests_feature_weighting', {})
    if not col_config.get('enabled', False) or not col_config.get('use_monotonic_constraints', False):
        return torch.zeros(len(feature_names), 8)
    
    raw_weights = col_config.get('weights', {})
    special_dynamic = config.get('special_feats', {}).get('dynamic', [])
    
    weight_signs_matrix = []
    
    for feat_name in feature_names:
        if feat_name in special_dynamic:
            # Dynamic features: no constraints
            weight_signs_matrix.append([0.0] * 8)
        elif feat_name in raw_weights:
            weight_list = raw_weights[feat_name]
            
            # Check if per-parameter weights are provided (list has 9 elements: title + 8 weights)
            if len(weight_list) >= 9:
                # Extract the 8 parameter-specific weights (skip title at index 0)
                param_weights = weight_list[1:9]
            else:
                # Fallback: use single weight for all parameters (old format)
                single_weight = weight_list[1] if len(weight_list) > 1 else 0.0
                param_weights = [single_weight] * 8
            
            # Convert weights to signs
            signs = []
            for w in param_weights:
                if pd.isna(w):
                    signs.append(0.0)
                elif w > 0:
                    signs.append(1.0)
                elif w < 0:
                    signs.append(-1.0)
                else:
                    signs.append(0.0)
            weight_signs_matrix.append(signs)
        else:
            # Feature not in weights: no constraints
            weight_signs_matrix.append([0.0] * 8)
    
    return torch.tensor(weight_signs_matrix, dtype=torch.float32)

def get_feature_null_mask(config, feature_names):
    """
    Return a boolean matrix [num_features, 8] indicating forced-null impacts.
    Use np.nan (or None) in CONFIG['column_tests_feature_weighting']['weights']
    to force a predictor to have zero effect on a given parameter.
    """
    col_config = config.get('column_tests_feature_weighting', {})
    if not col_config.get('enabled', False):
        return torch.zeros(len(feature_names), 8, dtype=torch.bool)

    raw_weights = col_config.get('weights', {})
    special_dynamic = config.get('special_feats', {}).get('dynamic', [])

    null_mask_matrix = []

    for feat_name in feature_names:
        if feat_name in special_dynamic:
            null_mask_matrix.append([False] * 8)
        elif feat_name in raw_weights:
            weight_list = raw_weights[feat_name]

            if len(weight_list) >= 9:
                param_weights = weight_list[1:9]
            else:
                single_weight = weight_list[1] if len(weight_list) > 1 else None
                param_weights = [single_weight] * 8

            null_row = [bool(pd.isna(w)) for w in param_weights]
            null_mask_matrix.append(null_row)
        else:
            null_mask_matrix.append([False] * 8)

    return torch.tensor(null_mask_matrix, dtype=torch.bool)

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
    target_col = config.get('special_feats', {}).get('target_feat', [])[0]
    id_col = 'project_sample_id'
    col_idx = 'project_col_id'
    df = df[df[target_col] > 1.0].copy()

    # Define time-varying columns
    special = config.get('special_feats', {})
    target_feats = list(special.get('target_feat', []))
    target_col = target_feats[0] if target_feats else 'cu_recovery_%'

    # Active time feature for this run
    time_feat_list = list(special.get('time_feat', []))
    time_feat_col = time_feat_list[0] if time_feat_list else 'leach_duration_days'
    leach_days = 'leach_duration_days'
    include_lixiviant = time_feat_col == 'cumulative_lixiviant_m3_t'

    # Build time-varying columns depending on the active time feature
    time_varying_cols = [leach_days, target_col, 'cumulative_catalyst_addition_kg_t']
    if include_lixiviant:
        time_varying_cols.append('cumulative_lixiviant_m3_t')

    catalyst_feats = list(special.get('catalyst_feat', []))
    catalyst_col = catalyst_feats[0] if catalyst_feats else None

    # Filter rows with NaNs in key columns (only those relevant for this run)
    subset_cols = [target_col, leach_days, 'cumulative_catalyst_addition_kg_t']
    if include_lixiviant:
        subset_cols.append('cumulative_lixiviant_m3_t')
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
                 time_feat_col, target_col, catalyst_col, 'catalyst_status']
    # Always drop leach_days; if active time is leach_days it's already in drop_cols via time_feat_col
    if leach_days not in drop_cols:
        drop_cols.append(leach_days)
    # Drop cumulative_lixiviant when not the active time feature to avoid leakage
    if not include_lixiviant and 'cumulative_lixiviant_m3_t' in df.columns:
        drop_cols.append('cumulative_lixiviant_m3_t')
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

    # If lixiviant not in use, remove the column from outputs to avoid downstream leakage/saving
    if not include_lixiviant and 'cumulative_lixiviant_m3_t' in out_df.columns:
        out_df = out_df.drop(columns=['cumulative_lixiviant_m3_t'])
    if not include_lixiviant and 'cumulative_lixiviant_m3_t' in out_df_averaged.columns:
        out_df_averaged = out_df_averaged.drop(columns=['cumulative_lixiviant_m3_t'])

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
# REACTOR SCALING ENSEMBLE MODEL
# ---------------------------

class EnsembleModels:
    """Ensemble model for reactor scaling with uncertainty quantification"""
    
    def __init__(self, model_states, val_losses, total_features, config, device, best_configs, num_cols):
        self.device = device
        self.total_features = total_features
        self.config = config
        self.best_configs = best_configs
        self.model_states = list(model_states) if model_states is not None else []
        self.val_losses_raw = list(val_losses) if val_losses is not None else []
        
        self.feature_weight_signs = get_feature_weight_signs(config, num_cols).to(device)
        self.feature_null_mask = get_feature_null_mask(config, num_cols).to(device)
        # self.feature_weight_mags  = get_feature_weight_magnitudes(config, num_cols, zero_policy='one').to(device)
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
        dropout_rate = config.get('pytorch_dropout_rate', 0.30)
        
        for idx, (model_state, val_loss) in enumerate(zip(model_states, val_losses)):
            if val_loss <= threshold:
                model = AdaptiveTwoPhaseRecoveryModel(
                    total_features=self.total_features,
                    hidden_dim=hidden_dim,  # Use overall best from final_config
                    dropout_rate=dropout_rate,  # Use overall best from final_config
                    init_mode=config.get('init_mode', 'kaiming'), # Use overall best init_mode
                    feature_weight_signs=self.feature_weight_signs,
                    feature_null_mask=self.feature_null_mask,
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
        """
        Make ensemble prediction with uncertainty quantification
        """
        all_model_predictions = []
        all_model_params = []
        
        for model in self.models:
            with torch.no_grad():
                params = model(X, catalyst, sample_ids)
                params = project_params_to_caps(
                    params,
                    float(self.config.get('base_asymptote_cap', 80.0)),
                    float(self.config.get('total_asymptote_cap', 95.0)),
                    float(self.config.get('base_rate_cap', 2.1)),
                    float(self.config.get('total_rate_cap', 7.0))
                    ) 
                all_model_params.append(params.cpu().numpy())
                
                recovery = generate_two_phase_recovery(
                    time_points, catalyst, transition_time, params
                )
                all_model_predictions.append(recovery.cpu().numpy())
        
        all_model_predictions = np.array(all_model_predictions)  # (M,B,T)
        all_model_params = np.array(all_model_params)            # (M,B,P)

        # Normalize weights once
        w = np.asarray(self.weights, dtype=float)
        w = w / (w.sum() if np.isfinite(w.sum()) and w.sum() > 0 else 1.0)
        w_broadcast = w[:, None, None]  # (M,1,1)

        # Weighted mean
        weighted_pred = (all_model_predictions * w_broadcast).sum(axis=0)  # (B,T)
        weighted_params = (all_model_params * w[:, None, None]).sum(axis=0)  # (B,P)

        # Weighted std (match the weighted mean)
        diff = all_model_predictions - weighted_pred[None, :, :]  # (M,B,T)
        var_w = (w_broadcast * diff**2).sum(axis=0)  # (B,T)
        uncertainty = np.sqrt(np.maximum(var_w, 0.0))  # (B,T)

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
    Enhanced Two-Phase Recovery Model with separate output heads for exact per-parameter control.
    Each parameter (a1, b1, a2, b2, a3, b3, a4, b4) has its own network head plus
    two catalyst-dose sensitivity heads (gain_b3, gain_b4) for rate modulation.
    """
    def __init__(self, total_features, hidden_dim=128, dropout_rate=0.30, init_mode='kaiming', feature_weight_signs=None, feature_weight_magnitudes=None, feature_null_mask=None):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.init_mode = init_mode
        self.total_features = total_features
        
        # Store feature weight signs for monotonic constraints
        if feature_weight_signs is not None:
            self.register_buffer('feature_weight_signs', feature_weight_signs)
            self.use_monotonic_constraints = True
        else:
            self.register_buffer('feature_weight_signs', torch.zeros(total_features, 8))
            self.use_monotonic_constraints = False
        
        # Magnitudes buffer (not used unless you scale losses/inputs later)
        # if feature_weight_magnitudes is None:
        #     feature_weight_magnitudes = torch.ones(total_features, 8)
        # self.register_buffer('feature_weight_magnitudes', feature_weight_magnitudes)
        # Optional per-feature scalar (mean over 8 params), clamp to avoid zeros
        # self.register_buffer('feature_scale_vector',
        #                     torch.clamp(self.feature_weight_magnitudes.mean(dim=1), min=0.5, max=2.0))

        if feature_null_mask is not None:
            self.register_buffer('feature_null_mask', feature_null_mask.to(torch.bool))
            self.use_forced_null = bool(self.feature_null_mask.any().item())
        else:
            self.register_buffer('feature_null_mask', torch.zeros(total_features, 8, dtype=torch.bool))
            self.use_forced_null = False

        # Initialize per-parameter heads container
        self.param_networks = nn.ModuleList()
        # 0-7: a1,b1,a2,b2,a3,b3,a4,b4; 8-9: catalyst dose gains for b3,b4
        for param_idx in range(10):
            network = nn.Sequential(
                nn.Linear(total_features, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim//2, 1),  # Single output per network
            )
            self.param_networks.append(network)
        
        self._initialize_weights(self.init_mode)

    def _initialize_weights(self, init_mode='kaiming'):
        for network in self.param_networks:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    if init_mode == 'kaiming':
                        nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                    elif init_mode == 'xavier':
                        nn.init.xavier_uniform_(layer.weight)
                    elif init_mode == 'normal':
                        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                    else:
                        nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                    nn.init.constant_(layer.bias, 0.0)
        
        # Initialize final layer biases to reasonable parameter values
        with torch.no_grad():
            self.param_networks[0][-1].bias[0] = 0.0   # a1
            self.param_networks[1][-1].bias[0] = -2.0  # b1
            self.param_networks[2][-1].bias[0] = 0.0   # a2
            self.param_networks[3][-1].bias[0] = -2.0  # b2
            self.param_networks[4][-1].bias[0] = 0.0   # a3
            self.param_networks[5][-1].bias[0] = -2.0  # b3
            self.param_networks[6][-1].bias[0] = 0.0   # a4
            self.param_networks[7][-1].bias[0] = -2.0  # b4
            if len(self.param_networks) > 8:
                self.param_networks[8][-1].bias[0] = 0.0   # gain_b3
                self.param_networks[9][-1].bias[0] = 0.0   # gain_b4

    def apply_monotonic_constraints(self):
        """
        Apply exact per-parameter monotonic constraints to each network head.
        Each parameter network's first layer is constrained based on its specific feature signs.
        """
        if not self.use_monotonic_constraints:
            return
        
        with torch.no_grad():
            # For each parameter network
            for param_idx in range(8):
                # Get the first layer of this parameter's network
                first_layer = self.param_networks[param_idx][0]
                first_weight = first_layer.weight  # Shape: [hidden_dim, num_features]
                
                # For each feature, apply the constraint for this specific parameter
                for feat_idx in range(self.feature_weight_signs.shape[0]):
                    sign_constraint = self.feature_weight_signs[feat_idx, param_idx].item()
                    
                    if sign_constraint > 0:
                        # Positive constraint: ensure all weights are positive
                        first_weight[:, feat_idx].data = torch.abs(first_weight[:, feat_idx].data)
                    elif sign_constraint < 0:
                        # Negative constraint: ensure all weights are negative
                        first_weight[:, feat_idx].data = -torch.abs(first_weight[:, feat_idx].data)
                    # If sign_constraint == 0, no constraint (leave as is)

    def apply_forced_nulls(self):
        """Hard-zero the first layer weights for forced-null feature/parameter pairs."""
        if not self.use_forced_null:
            return

        with torch.no_grad():
            for param_idx in range(8):
                first_layer = self.param_networks[param_idx][0]
                null_mask = self.feature_null_mask[:, param_idx]
                if null_mask.any():
                    first_layer.weight[:, null_mask] = 0.0

    def register_gradient_hooks(self):
        """
        Register hooks to mask gradients that would violate monotonic constraints.
        This prevents the optimizer from updating weights in directions that violate constraints.
        """
        if not self.use_monotonic_constraints:
            return
        
        def create_gradient_mask_hook(param_idx):
            """Create a hook that masks gradients for a specific parameter"""
            def hook(grad):
                if grad is None:
                    return None
                
                # Get the first layer of this parameter's network
                first_layer = self.param_networks[param_idx][0]
                
                # Create a mask for the gradient
                mask = torch.ones_like(grad)
                
                # For each feature
                for feat_idx in range(self.feature_weight_signs.shape[0]):
                    if self.use_forced_null and bool(self.feature_null_mask[feat_idx, param_idx].item()):
                        # Fully block gradients for forced-null feature/parameter pairs
                        mask[:, feat_idx] = 0.0
                        continue

                    sign_constraint = self.feature_weight_signs[feat_idx, param_idx].item()
                    
                    if sign_constraint != 0:
                        # Get current weights for this feature
                        current_weights = first_layer.weight[:, feat_idx]
                        
                        # Get gradients for this feature
                        feature_grad = grad[:, feat_idx]
                        
                        if sign_constraint > 0:
                            # Positive constraint: weights should be positive
                            # Mask gradients that would make positive weights negative
                            # or make negative weights more negative
                            should_be_positive = current_weights > 0
                            would_decrease = feature_grad < 0
                            mask[:, feat_idx] = torch.where(
                                should_be_positive & would_decrease & (current_weights.abs() < 0.01),
                                torch.tensor(0.0, device=grad.device),
                                mask[:, feat_idx]
                            )
                        
                        elif sign_constraint < 0:
                            # Negative constraint: weights should be negative
                            # Mask gradients that would make negative weights positive
                            # or make positive weights more positive
                            should_be_negative = current_weights < 0
                            would_increase = feature_grad > 0
                            mask[:, feat_idx] = torch.where(
                                should_be_negative & would_increase & (current_weights.abs() < 0.01),
                                torch.tensor(0.0, device=grad.device),
                                mask[:, feat_idx]
                            )
                
                return grad * mask
            
            return hook
        
        # Register hooks for each parameter network
        self.gradient_hooks = []
        for param_idx in range(8):
            first_layer = self.param_networks[param_idx][0]
            hook = first_layer.weight.register_hook(create_gradient_mask_hook(param_idx))
            self.gradient_hooks.append(hook)
    
    def remove_gradient_hooks(self):
        """Remove gradient masking hooks"""
        if hasattr(self, 'gradient_hooks'):
            for hook in self.gradient_hooks:
                hook.remove()
            self.gradient_hooks = []

    def forward(self, x, catalyst, sample_ids=None):
        # Apply monotonic constraints during training
        if self.training and self.use_monotonic_constraints:
            self.apply_monotonic_constraints()
        
        # Always enforce forced-null masks
        if self.use_forced_null:
            self.apply_forced_nulls()
        
        batch_size = x.size(0)
        
        # Initialize parameter tensor
        params = torch.zeros(batch_size, 10, device=x.device)
        
        # Parameter limits (from CONFIG if available) 
        lim = CONFIG.get('param_limits', {})
        a1_min, a1_max = lim.get('a1', (10.0, 50.0)) # 1.5 to 68.0
        b1_min, b1_max = lim.get('b1', (1e-3, 2.1)) # 3e-4 to 2.1
        a2_min, a2_max = lim.get('a2', (5.0, 40.0)) # 2.2 to 79.0
        b2_min, b2_max = lim.get('b2', (1e-4, 2.1)) # 1e-4 to 2.1

        # Catalyst parameters
        a3_min, a3_max = lim.get('a3', (5.0, 25.0)) # 0.5 to 45.0
        b3_min, b3_max = lim.get('b3', (1e-4, 1.4)) # 4e-4 to 1.4
        a4_min, a4_max = lim.get('a4', (1.0, 15.0)) # 0.5 to 40.0
        b4_min, b4_max = lim.get('b4', (1e-4, 2.3)) # 4e-4 to 2.3
        gain_b3_min, gain_b3_max = lim.get(
            'gain_b3',
            (0.0, float(CONFIG.get('cat_rate_gain_b3', 0.3)) * 2.0)
        )
        gain_b4_min, gain_b4_max = lim.get(
            'gain_b4',
            (0.0, float(CONFIG.get('cat_rate_gain_b4', 0.1)) * 2.0)
        )

        # Process base parameters (a1, b1, a2, b2) with consistency constraint
        if sample_ids is not None:
            # Ensure physical consistency: same base parameters for same sample_id
            unique_sample_ids = list(set(sample_ids))
            sample_id_to_base_params = {}
            
            for unique_id in unique_sample_ids:
                # Find indices for this sample_id
                sample_indices = [i for i, sid in enumerate(sample_ids) if sid == unique_id]
                
                if sample_indices:
                    # Use the first occurrence to compute base parameters
                    first_idx = sample_indices[0]
                    
                    # Process each base parameter through its dedicated network
                    a1_raw = self.param_networks[0](x[first_idx:first_idx+1])
                    b1_raw = self.param_networks[1](x[first_idx:first_idx+1])
                    a2_raw = self.param_networks[2](x[first_idx:first_idx+1])
                    b2_raw = self.param_networks[3](x[first_idx:first_idx+1])
                    
                    # Apply parameter constraints
                    a1 = a1_min + (a1_max - a1_min) * torch.sigmoid(a1_raw)
                    b1 = b1_min + (b1_max - b1_min) * torch.sigmoid(b1_raw)
                    a2 = a2_min + (a2_max - a2_min) * torch.sigmoid(a2_raw)
                    b2 = b2_min + (b2_max - b2_min) * torch.sigmoid(b2_raw)

                    # Enforce base cap: a1 + a2 <= base_cap
                    base_cap = float(CONFIG.get('base_asymptote_cap', 80.0))
                    total_asymptote = a1 + a2
                    mask_a = total_asymptote > base_cap
                    scale = torch.where(mask_a & (total_asymptote > 0),
                                        base_cap / total_asymptote.clamp(min=1.0),
                                        torch.tensor(1.0, device=x.device))
                    a1, a2 = a1 * scale, a2 * scale
                    
                    # Store base parameters for this sample_id
                    sample_id_to_base_params[unique_id] = torch.cat([a1.flatten(), b1.flatten(), a2.flatten(), b2.flatten()])
                    
                    # Apply the same base parameters to all samples with this sample_id
                    for idx in sample_indices:
                        params[idx, :4] = sample_id_to_base_params[unique_id]
        else:
            # Fallback: process all samples independently
            a1_raw = self.param_networks[0](x)
            b1_raw = self.param_networks[1](x)
            a2_raw = self.param_networks[2](x)
            b2_raw = self.param_networks[3](x)
            
            a1 = a1_min + (a1_max - a1_min) * torch.sigmoid(a1_raw.squeeze())
            b1 = b1_min + (b1_max - b1_min) * torch.sigmoid(b1_raw.squeeze())
            a2 = a2_min + (a2_max - a2_min) * torch.sigmoid(a2_raw.squeeze())
            b2 = b2_min + (b2_max - b2_min) * torch.sigmoid(b2_raw.squeeze())

            # Ensure total asymptote doesn't exceed 80
            base_cap = float(CONFIG.get('base_asymptote_cap', 80.0))
            total_asymptote = a1 + a2
            mask_a = total_asymptote > base_cap
            scale = torch.where(mask_a & (total_asymptote > 0),
                               base_cap / total_asymptote.clamp(min=1.0),
                               torch.tensor(1.0, device=x.device))
            a1, a2 = a1 * scale, a2 * scale
            
        params[:, :4] = torch.stack(
            [a1.reshape(-1), b1.reshape(-1), a2.reshape(-1), b2.reshape(-1)],
            dim=1,
        )
        
        # Process catalyst parameters if catalyst is present
        has_catalyst = torch.any(catalyst > 0, dim=1)
        if has_catalyst.any():
            idx = has_catalyst.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                # Get base parameters for catalyzed samples
                a1_r, b1_r, a2_r, b2_r = [p.squeeze() for p in params[idx, :4].split(1, dim=1)]
                
                # Process catalyst parameters through dedicated networks
                a3_raw = self.param_networks[4](x[idx])
                b3_raw = self.param_networks[5](x[idx])
                a4_raw = self.param_networks[6](x[idx])
                b4_raw = self.param_networks[7](x[idx])
                
                # Apply catalyst parameter constraints
                a3 = a3_min + (a3_max - a3_min) * torch.sigmoid(a3_raw.squeeze())
                b3 = b3_min + (b3_max - b3_min) * torch.sigmoid(b3_raw.squeeze())
                a4 = a4_min + (a4_max - a4_min) * torch.sigmoid(a4_raw.squeeze())
                b4 = b4_min + (b4_max - b4_min) * torch.sigmoid(b4_raw.squeeze())
                gain_b3 = gain_b3_min + (gain_b3_max - gain_b3_min) * torch.sigmoid(self.param_networks[8](x[idx]).squeeze())
                gain_b4 = gain_b4_min + (gain_b4_max - gain_b4_min) * torch.sigmoid(self.param_networks[9](x[idx]).squeeze())

                # Enforce total cap: a1 + a2 + a3 + a4 <= 95
                total_cap = float(CONFIG.get('total_asymptote_cap', 95.0))
                total_asymptote_cat = a1_r + a2_r + a3 + a4
                mask_a = total_asymptote_cat > total_cap
                scale = torch.where(mask_a & (total_asymptote_cat > 0),
                                    total_cap / total_asymptote_cat.clamp(min=1.0),
                                    torch.tensor(1.0, device=x.device))
                a3, a4 = a3 * scale, a4 * scale
                
                params[idx, 4:] = torch.stack(
                    [a3.reshape(-1), b3.reshape(-1), a4.reshape(-1), b4.reshape(-1), gain_b3.reshape(-1), gain_b4.reshape(-1)],
                    dim=1,
                )
        else:
            # params[:, 4:] = np.nan
            params[:, 4:] = torch.full((batch_size, 6), float('nan'), device=x.device)
        
        # Final safety projection with rate caps
        base_cap = float(CONFIG.get('base_asymptote_cap', 80.0))
        total_cap = float(CONFIG.get('total_asymptote_cap', 95.0))
        base_rate_cap = float(CONFIG.get('base_rate_cap', 2.1))
        total_rate_cap = float(CONFIG.get('total_rate_cap', 7.0))
        params = project_params_to_caps(params, base_cap, total_cap, base_rate_cap, total_rate_cap)
        return params

def generate_two_phase_recovery_exp(time, catalyst, transition_time, params):
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
    ce_pow = float(CONFIG.get('cat_effect_power', 1.0))
    if ce_pow != 1.0:
        catalyst_effect = catalyst_effect.pow(ce_pow)

    
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

            # Dose-dependent rate multipliers for b3 and b4
            if params.shape[1] >= 10:
                gain_b3 = params[:, 8].unsqueeze(1)
                gain_b4 = params[:, 9].unsqueeze(1)
            else:
                gain_b3 = params.new_full((params.size(0), 1), float(CONFIG.get('cat_rate_gain_b3', 0.0)))
                gain_b4 = params.new_full((params.size(0), 1), float(CONFIG.get('cat_rate_gain_b4', 0.0)))
            gain_b3 = torch.clamp_min(gain_b3, 0.0)
            gain_b4 = torch.clamp_min(gain_b4, 0.0)

            # shape broadcast: [B,1] * [B,T] -> [B,T]
            rate_mult3 = 1.0 + catalyst_effect * gain_b3
            rate_mult4 = 1.0 + catalyst_effect * gain_b4

            # Use catalyst_effect tensor in the exponential terms for enhanced modeling
            exp_term3 = torch.exp(-torch.abs(b3) * time_shifted * rate_mult3).clamp(min=1e-8, max=1.0)
            exp_term4 = torch.exp(-torch.abs(b4) * time_shifted * rate_mult4).clamp(min=1e-8, max=1.0)
            
            # Additional recovery from catalyst with catalyst_effect scaling
            additional_recovery = catalyst_effect * (torch.abs(a3) * (1 - exp_term3) + torch.abs(a4) * (1 - exp_term4))
            
            # Apply catalyst enhancement only where catalyst is present and after transition
            catalyst_enhancement = torch.where(has_catalyst_points, additional_recovery, torch.zeros_like(additional_recovery))
            recovery = recovery_control + catalyst_enhancement
    
    # Apply reasonable bounds to recovery
    total_cap = float(CONFIG.get('total_asymptote_cap', 95.0))
    recovery = torch.clamp(recovery, min=0.0, max=total_cap)
    return recovery


# Default to the exponential version for modelling/inference
def generate_two_phase_recovery(time, catalyst, transition_time, params):
    return generate_two_phase_recovery_exp(time, catalyst, transition_time, params)

def compute_monotonic_penalty(model, lambda_penalty=0.1):
    """
    Compute penalty for violating monotonic constraints with parameter-specific scaling.
    
    The penalty is scaled based on the typical range of each parameter:
    - a parameters (a1, a2, a3, a4): Range [0, 100] → scale = 1.0
    - b parameters (b1, b2, b3, b4): Range [0.0001, 0.01] → scale = 10000.0
    
    This ensures that violations affecting b parameters are appropriately penalized
    relative to their tiny scale.
    
    Args:
        model: The neural network model
        lambda_penalty: Base weight for the penalty term
    
    Returns:
        torch.Tensor: Penalty value (scalar)
    """
    if not model.use_monotonic_constraints:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    
    # Define parameter-specific scaling factors
    # a parameters: indices 0, 2, 4, 6 (a1, a2, a3, a4) - range [0, 100]
    # b parameters: indices 1, 3, 5, 7 (b1, b2, b3, b4) - range [0.0001, 0.01]
    param_scales = torch.tensor([
        1.0,      # a1: scale = 1.0 (baseline)
        1.0,  # b1: scale = 1000.0 (tiny values need huge penalty)
        1.0,      # a2: scale = 1.0
        1.0,  # b2: scale = 10000.0
        1.0,      # a3: scale = 1.0
        1.0,  # b3: scale = 10000.0
        1.0,      # a4: scale = 1.0
        1.0,  # b4: scale = 10000.0
    ], device=next(model.parameters()).device)
    
    # For each parameter network
    for param_idx in range(8):
        # Get the first layer weights
        first_layer = model.param_networks[param_idx][0]
        first_weight = first_layer.weight  # Shape: [hidden_dim, num_features]
        
        # Get the scaling factor for this parameter
        scale = param_scales[param_idx]
        
        # For each feature
        for feat_idx in range(model.feature_weight_signs.shape[0]):
            sign_constraint = model.feature_weight_signs[feat_idx, param_idx].item()
            
            if sign_constraint > 0:
                # Positive constraint: penalize negative weights
                negative_weights = first_weight[:, feat_idx]
                violation = torch.sum(torch.relu(-negative_weights))
                penalty += scale * violation  # Scale by parameter importance
            
            elif sign_constraint < 0:
                # Negative constraint: penalize positive weights
                positive_weights = first_weight[:, feat_idx]
                violation = torch.sum(torch.relu(positive_weights))
                penalty += scale * violation  # Scale by parameter importance
    
    return lambda_penalty * penalty

def check_constraint_violations(model):
    """
    Count how many constraints are violated.
    
    Args:
        model: The neural network model
    
    Returns:
        tuple: (violations, total_constraints, violation_ratio)
    """
    if not model.use_monotonic_constraints:
        return 0, 0, 0.0
    
    violations = 0
    total_constraints = 0
    
    for param_idx in range(8): # 8 because it is one per network defined (in this case, one network per parameter)
        first_layer = model.param_networks[param_idx][0]
        first_weight = first_layer.weight
        
        for feat_idx in range(model.feature_weight_signs.shape[0]):
            sign_constraint = model.feature_weight_signs[feat_idx, param_idx].item()
            
            if sign_constraint != 0:
                total_constraints += 1
                mean_weight = first_weight[:, feat_idx].mean().item()
                
                # Check if constraint is violated
                if (sign_constraint > 0 and mean_weight < 0) or \
                   (sign_constraint < 0 and mean_weight > 0):
                    violations += 1
    
    violation_ratio = violations / total_constraints if total_constraints > 0 else 0.0
    
    return violations, total_constraints, violation_ratio

def compute_sample_weights(y_data, device):
    """
    Return per-element weights to flatten the influence of large targets.
    Current policy: inverse-abs weighting normalized to mean 1 so the
    optimizer step size remains stable.
    """
    temp_weights = []
    total_weight = 0.0
    total_elems = 0

    for y_array in y_data:
        if not isinstance(y_array, torch.Tensor):
            y_array = torch.tensor(y_array, dtype=torch.float32, device=device)
        # Inverse-abs dampens the impact of large-magnitude targets
        w = 1.0 / (y_array.abs() + 1e-6)
        temp_weights.append(w)
        total_weight += w.sum().item()
        total_elems += w.numel()

    mean_w = (total_weight / total_elems) if total_elems > 0 else 1.0
    return [w / mean_w for w in temp_weights]

def train_single_fold_reactor_scaling(fold_data_with_params):
    """Train a single fold for reactor scaling"""
    fold_data, num_cols, config, device, save_dir, X_tensor, y_tensor, time_tensor, catalyst_tensor, tt_tensor, sample_ids, sample_col_ids, use_val_data_days = fold_data_with_params
    
    i, test_sample_id = fold_data
    print(f"Training fold {i+1}: Testing on {test_sample_id}")
    
    # Enforce reproducibility per fold/seed
    set_all_seeds(config.get('pytorch_seed'), config.get('use_deterministic_algorithms', True))
    
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
            t_arr = time_tensor[j]
            is_catalyzed = torch.any(catalyst_tensor[j] > 0).item()
            time_threshold = (tt_tensor[j].item() + use_val_data_days) if is_catalyzed else use_val_data_days
            mask_x_days = t_arr < time_threshold
            if mask_x_days.any():
                print(f"Appending test sample {test_sample_id} (Catalyzed: {is_catalyzed}, Time threshold: {time_threshold} days)")
                X_train = torch.cat([X_train, X_tensor[j:j+1]], dim=0)
                tt_train.append(tt_tensor[j])
                y_train.append(y_tensor[j][mask_x_days])
                time_train.append(time_tensor[j][mask_x_days])
                catalyst_train.append(catalyst_tensor[j][mask_x_days])
                train_sample_ids.append(test_sample_id)

    # Initialize model
    feature_weight_signs = get_feature_weight_signs(config, num_cols).to(device)
    feature_null_mask = get_feature_null_mask(config, num_cols).to(device)
    feature_weight_mags  = get_feature_weight_magnitudes(config, num_cols, zero_policy='one').to(device)

    fold_model = AdaptiveTwoPhaseRecoveryModel(
        total_features=len(num_cols),
        hidden_dim=config.get('pytorch_hidden_dim', 128),
        dropout_rate=config.get('pytorch_dropout_rate', 0.30),
        init_mode=config.get('init_mode', 'kaiming'),
        feature_weight_signs=feature_weight_signs,
        feature_null_mask=feature_null_mask,
        # feature_weight_magnitudes=feature_weight_mags
    ).to(device)
        
    # Apply hard constraints (projection)
    if config.get('column_tests_feature_weighting', {}).get('use_monotonic_constraints', False):
        # Separate heads already provide structure
        
        # Optional: Add gradient masking
        if config.get('column_tests_feature_weighting', {}).get('use_gradient_masking', False):
            fold_model.register_gradient_hooks()
    
    optimizer = optim.AdamW(fold_model.parameters(), lr=learning_rate, weight_decay=config.get('pytorch_weight_decay', 1e-4))
    scheduler = ReduceLROnPlateau(optimizer, 
                                  mode='min', 
                                  factor=config['adaptive_lr']['plateau_factor'], 
                                  patience=config['adaptive_lr']['plateau_patience']//2, 
                                  min_lr=config['adaptive_lr']['min_lr'])

    weights_train = compute_sample_weights(y_train, device)
    total_weighted_points = sum(w.numel() for w in weights_train) or 1
    loss_scale = float(config.get('loss_scale', 1.0))

    # Training loop
    best_loss = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    best_model_state = fold_model.state_dict().copy()
    
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
            
            # Compute MSE loss
            mse_criterion = nn.MSELoss(reduction='none')
            per_element_loss = mse_criterion(predicted_recovery, target_recovery)
            weighted_loss = weights_sample * per_element_loss
            mse_loss_value = weighted_loss.sum() / total_weighted_points
            
            # Physics loss (currently not used)
            physics_loss = torch.tensor(0.0, device=device)
            
            # Add monotonic constraint penalty (if enabled)
            if config.get('column_tests_feature_weighting', {}).get('use_penalty_loss', False):
                monotonic_penalty = compute_monotonic_penalty(
                    fold_model, 
                    lambda_penalty=config.get('monotonic_penalty_weight', 0.1)
                )
            else:
                monotonic_penalty = torch.tensor(0.0, device=device)
            
            floor_penalty = torch.tensor(0.0, device=device)
            if CONFIG.get('control_floor_weight', 0.0) > 0 and torch.any(catalyst_sample > 0):
                base_params = params.clone()
                base_params[:, 4:] = 0.0
                control_curve = generate_two_phase_recovery(time_sample, torch.zeros_like(time_sample), tt_sample, base_params)
                floor_penalty = CONFIG.get('control_floor_weight', 0.0) * torch.relu(control_curve - predicted_recovery).mean()

            loss = (mse_loss_value + physics_loss + monotonic_penalty + floor_penalty) * loss_scale
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fold_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += (weighted_loss.sum().detach().item() / total_weighted_points)
        
        avg_loss = total_loss  # already normalized across all points
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
                per_element_loss_val = mse_criterion(predicted_recovery, target_recovery)
                weighted_loss_val = weights_sample * per_element_loss_val
                val_losses.append((weighted_loss_val.sum() / total_weighted_points) * loss_scale)
        
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
        val_loss_curve = []
        val_epochs = []
        if len(train_losses) > 0:
            val_loss_curve = [loss * 1.1 for loss in train_losses[::10]]
            val_epochs = list(range(0, len(train_losses), 10))[:len(val_loss_curve)]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=300)
        ax1.plot(train_losses, label='Training Loss', color='blue')
        if val_loss_curve:
            ax1.plot(val_epochs, val_loss_curve, label='Validation Loss', color='orange')

        # ax1.set_title(f'Training vs. Validation Loss - {test_sample_id}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.grid(True)

        learning_rates = [learning_rate * (config['adaptive_lr']['plateau_factor'] ** (epoch // (config['pytorch_patience']//2))) for epoch in range(len(train_losses))]
        ax2.plot(learning_rates, label='Learning Rate', color='green')
        # ax2.set_title(f'Learning Rate Schedule - {test_sample_id}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'adaptive_loss_curve_{test_sample_id}_withoutReactors.png'), dpi=300)
        plt.close()
        print(f"Saved adaptive loss curve for {test_sample_id}")

    # Load best model and evaluate
    fold_model.load_state_dict(best_model_state)
    fold_model.eval()
    
    # Collect training data for diagnostics
    do_plots = config.get('generate_plots', False)
    train_y_true = [] if do_plots else None
    train_y_pred = [] if do_plots else None
    
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
            
            if do_plots:
                train_y_true.append(target_recovery.detach())
                train_y_pred.append(predicted_recovery.detach().squeeze())
    
    # Evaluate on test set
    fold_results = []
    all_y_true = [] if do_plots else None
    all_y_pred = [] if do_plots else None
    
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

                # Enforce caps (base and total) deterministically
                base_cap = float(CONFIG.get('base_asymptote_cap', 80.0))
                total_cap = float(CONFIG.get('total_asymptote_cap', 95.0))
                base_rate_cap = float(CONFIG.get('base_rate_cap', 2.1))
                total_rate_cap = float(CONFIG.get('total_rate_cap', 7.0))
                combined_params = project_params_to_caps(combined_params,
                                                         base_cap,
                                                         total_cap,
                                                         base_rate_cap,
                                                         total_rate_cap).squeeze(0)
                # Store after scaling
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
            
            if do_plots:
                all_y_true.append(target_recovery.detach())
                all_y_pred.append(predicted_recovery.detach().squeeze())
            
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
                
                total_cap = float(CONFIG.get('total_asymptote_cap', 95.0))
                if is_catalyzed:
                    plt.vlines(x=tt_np, ymin=0, ymax=100, color=colors[is_catalyzed],
                              linestyle='--', alpha=0.7, label=f'Transition Time ({col_id})')
        
        plt.title(f"Validation Recovery Curves - {test_sample_id}\nVal Loss: {best_val_loss:.4f}")
        plt.xlabel(get_time_axis_label(config))
        plt.ylabel("Cu Recovery (%)")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.ylim(0, 100)
        plt.xlim(left=0) # int(np.ceil(max(tt_tensor)/30)*30)
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"val_columns_{test_sample_id}_withoutReactors.png"), dpi=300)
        plt.close()
        print(f"Saved validation plot for {test_sample_id}")
        
        # Diagnostic plots
        if len(train_y_true) > 0 and len(all_y_true) > 0:
            try:
                train_y_true_arr = np.concatenate([t.detach().cpu().numpy().ravel() for t in train_y_true])
                train_y_pred_arr = np.concatenate([p.detach().cpu().numpy().ravel() for p in train_y_pred])
                test_y_true_arr = np.concatenate([t.detach().cpu().numpy().ravel() for t in all_y_true])
                test_y_pred_arr = np.concatenate([p.detach().cpu().numpy().ravel() for p in all_y_pred])

                fig = plot_diagnostics(train_y_true_arr, train_y_pred_arr, test_y_true_arr, test_y_pred_arr, config)
                if fig is not None:
                    plt.suptitle(f"Residues diagnosis for Cu Recovery {test_sample_id}", fontsize=12, fontweight="bold")
                    os.makedirs(save_dir, exist_ok=True)
                    fig.savefig(os.path.join(save_dir, f'diagnostics_{test_sample_id}_withoutReactors.png'), dpi=300)
                    plt.close(fig)
                    print(f"Saved diagnostic plot for {test_sample_id}")
            except Exception as diag_exc:
                print(f"Skipping diagnostic plot for {test_sample_id} (reason: {diag_exc})")
    
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
    
    # Seed before any hyper-search randomness inside this fold
    set_all_seeds(config.get('pytorch_seed'), config.get('use_deterministic_algorithms', True))

    excluded_ids = get_excluded_ids(test_sample_id)
    train_mask = [sid not in excluded_ids for sid in sample_ids]
    
    # Find best config for this fold
    if not config.get('skip_hypersearch', False):
        best_config, best_config_records = find_best_config_for_fold(
            X_tensor, y_tensor, time_tensor, catalyst_tensor, tt_tensor,
            sample_ids, sample_col_ids, num_cols, device, save_dir,
            train_mask, test_sample_id
        )
    else:
        best_config = config
        best_config_records = [{
            'pytorch_hidden_dim': config.get('pytorch_hidden_dim'),
            'pytorch_dropout_rate': config.get('pytorch_dropout_rate'),
            'init_mode': config.get('init_mode'),
            'scheduler_type': config.get('adaptive_lr', {}).get('scheduler_type', 'reduce_on_plateau'),
            'val_loss': np.nan,
        }]

    # Create config for final training with best hyperparameters
    final_config = copy.deepcopy(config)
    if not config.get('skip_hypersearch', False):
        final_config['pytorch_hidden_dim'] = best_config['pytorch_hidden_dim']
        final_config['pytorch_dropout_rate'] = best_config['pytorch_dropout_rate']
        final_config['init_mode'] = best_config['init_mode']
        final_config['adaptive_lr']['scheduler_type'] = best_config.get(
            'scheduler_type',
            best_config.get('adaptive_lr', {}).get('scheduler_type', config.get('adaptive_lr', {}).get('scheduler_type', 'reduce_on_plateau'))
        )
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
    orig_generate_plots = config.get('generate_plots', False)
    config_no_plots = copy.deepcopy(config)
    config_no_plots['generate_plots'] = False
    
    use_parallel = device.type == 'cpu'
    mode_msg = "parallel (CPU)" if use_parallel else "serial (non-CPU device detected)"
    print(f"Starting {mode_msg} training with {len(unique_sample_ids)} folds...")
    
    all_models = []
    all_val_losses = []
    all_results = []  # Changed from all_summary to all_results
    all_best_configs = []
    
    if use_parallel:
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
        config=config_no_plots,
        device=device,
        save_dir=save_dir,
        use_val_data_days=use_val_data_days),
                enumerate(unique_sample_ids)
            ), total=len(unique_sample_ids)))
    else:
        fold_results = []
        for fd in tqdm(enumerate(unique_sample_ids), total=len(unique_sample_ids)):
            fold_results.append(process_fold(
                fd,
                X_tensor=X_tensor,
                y_tensor=y_tensor,
                time_tensor=time_tensor,
                catalyst_tensor=catalyst_tensor,
                tt_tensor=tt_tensor,
                sample_ids=sample_ids,
                sample_col_ids=sample_col_ids,
                num_cols=num_cols,
                config=config_no_plots,
                device=device,
                save_dir=save_dir,
                use_val_data_days=use_val_data_days,
            ))

    for fold_result in fold_results:
        all_models.append(fold_result['model_state'])
        all_val_losses.append(fold_result['val_loss'])
        all_results.extend(fold_result['results'])
        cfg = fold_result.get('config')
        records = fold_result.get('config_records')
        if records:
            all_best_configs.extend(records)
        elif cfg:
            all_best_configs.append(cfg)

    # Save all per-fold best configs
    all_configs_df = pd.DataFrame(all_best_configs)
    os.makedirs(save_dir, exist_ok=True)
    all_configs_df.to_csv(os.path.join(save_dir, 'all_best_configs.csv'), index=False)
    print(f"Saved all best configs (per fold) to CSV")

    # Compute overall best config
    if all_configs_df.empty:
        # Fallback to provided config when no search results
        best_overall = pd.Series({
            'pytorch_hidden_dim': config_no_plots.get('pytorch_hidden_dim'),
            'pytorch_dropout_rate': config_no_plots.get('pytorch_dropout_rate'),
            'init_mode': config_no_plots.get('init_mode'),
            'scheduler_type': config_no_plots.get('adaptive_lr', {}).get('scheduler_type', 'reduce_on_plateau'),
            'val_loss': np.nan,
        })
    else:
        # Ensure required columns exist
        defaults = {
            'pytorch_hidden_dim': config_no_plots.get('pytorch_hidden_dim'),
            'pytorch_dropout_rate': config_no_plots.get('pytorch_dropout_rate'),
            'init_mode': config_no_plots.get('init_mode'),
            'scheduler_type': config_no_plots.get('adaptive_lr', {}).get('scheduler_type', 'reduce_on_plateau'),
            'val_loss': np.nan,
        }
        for col, default in defaults.items():
            if col not in all_configs_df.columns:
                all_configs_df[col] = default
        if all_configs_df['val_loss'].notna().any():
            config_groups = all_configs_df.groupby(['pytorch_hidden_dim', 'pytorch_dropout_rate', 'init_mode', 'scheduler_type'])
            mean_val_losses = config_groups['val_loss'].mean().reset_index()
            best_overall = mean_val_losses.loc[mean_val_losses['val_loss'].idxmin()]
            best_overall.to_csv(os.path.join(save_dir, 'best_overall_config.csv'), index=False)
            print(f"Saved overall best config to CSV")
        else:
            best_overall = pd.Series(defaults)
    
    print(f"Overall best configuration (lowest average val_loss across folds):")
    print(best_overall)

    # Train final model with best overall config
    final_config = copy.deepcopy(config_no_plots)
    final_config['pytorch_hidden_dim'] = best_overall.get('pytorch_hidden_dim', config_no_plots.get('pytorch_hidden_dim'))
    final_config['pytorch_dropout_rate'] = best_overall.get('pytorch_dropout_rate', config_no_plots.get('pytorch_dropout_rate'))
    final_config['init_mode'] = best_overall.get('init_mode', config_no_plots.get('init_mode'))
    final_config['adaptive_lr']['scheduler_type'] = best_overall.get('scheduler_type', config_no_plots.get('adaptive_lr', {}).get('scheduler_type', 'reduce_on_plateau'))
    final_config['generate_plots'] = True  # Enable plots only for final training pass
    final_config['skip_hypersearch'] = True

    # Train final model for each fold with best config
    final_results = []
    final_models = []
    final_val_losses = []

    if use_parallel:
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
    else:
        final_fold_results = []
        for fd in tqdm(enumerate(unique_sample_ids), total=len(unique_sample_ids)):
            final_fold_results.append(process_fold(
                fd,
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
                use_val_data_days=use_val_data_days,
            ))

    for fold_result in final_fold_results:
        final_models.append(fold_result['model_state'])
        final_val_losses.append(fold_result['val_loss'])
        final_results.extend(fold_result['results'])

    # Save validation results
    summary_df = pd.DataFrame(final_results)
    os.makedirs(save_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(save_dir, 'column_prediction_summary_validation_withoutReactors.csv'), index=False)
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
    Create ensemble predictions with uncertainty quantification and extended time predictions.
    Returns both the summary dataframe and the raw interval payload for later use.
    """
    config = config or {}
    if not config.get('generate_plots', False):
        return pd.DataFrame(), {}  # Return empty containers if plots disabled

    print("Creating ensemble predictions with uncertainty...")
    time_label = get_time_axis_label(config)
    os.makedirs(save_dir, exist_ok=True)
    
    summary = []
    interval_records = []
    unique_sample_ids = list(set(sample_ids))
    uncert_scale = float(config.get('uncertainty_scale', 1.0))
    pi_bounds = config.get('prediction_interval_bounds', {'min_pct': 50.0, 'max_pct': 99.0, 'default_pct': 90.0})
    default_pi_pct = float(pi_bounds.get('default_pct', 90.0))
    try:
        z_base = NormalDist().inv_cdf(0.5 + default_pi_pct / 200.0)
    except Exception:
        z_base = 1.645  # fallback ~90%
    z_effective = z_base * uncert_scale  # calibrated multiplier for target PI (default 90%)
    nominal_cov = default_pi_pct / 100.0
    pi_label = f'{default_pi_pct:.0f}% PI (mean ± {z_effective:.2f}σ)'

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
            
            # Store interval record (full curve, before plotting filters)
            full_time_arr = np.asarray(full_time, dtype=np.float32)
            mean_pred_arr = np.asarray(mean_pred, dtype=np.float32)
            uncert_arr = np.asarray(uncertainty, dtype=np.float32)
            lower_default = mean_pred_arr - z_effective * uncert_arr
            upper_default = mean_pred_arr + z_effective * uncert_arr
            tt_value = tt_sample.item() if isinstance(tt_sample, torch.Tensor) else float(tt_sample)
            interval_records.append({
                'project_sample_id_reactormatch': sample_id,
                'project_col_id': col_id,
                'time': full_time_arr,
                'mean': mean_pred_arr,
                'std': uncert_arr,
                'lower_default': lower_default,
                'upper_default': upper_default,
                'coverage_pct': nominal_cov * 100.0 if np.isfinite(nominal_cov) else np.nan,
                'z_effective': z_effective,
                'z_base': z_base,
                'uncertainty_scale': uncert_scale,
                'is_catalyzed': bool(is_catalyzed),
                'transition_time': tt_value,
                'post_transition_mask': (full_time_arr >= tt_value),
            })

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
                                    mean_pred_filtered - z_effective * uncertainty_filtered, 
                                    mean_pred_filtered + z_effective * uncertainty_filtered, 
                                    alpha=0.2, color=colors[is_catalyzed], 
                                    label=f'{pi_label} ({col_id})')
                
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
                                mean_pred - z_effective * uncertainty, 
                                mean_pred + z_effective * uncertainty, 
                                alpha=0.2, color=colors[is_catalyzed], 
                                label=f'{pi_label} ({col_id})')
            
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
                'transition_time': tt_sample.item() if isinstance(tt_sample, torch.Tensor) else tt_sample,
                'pi_lower_last': float(lower_default[-1]) if len(lower_default) > 0 else np.nan,
                'pi_upper_last': float(upper_default[-1]) if len(upper_default) > 0 else np.nan,
                'pi_coverage_pct': nominal_cov * 100.0 if np.isfinite(nominal_cov) else np.nan
            }
            summary.append(summary_entry)
        
        # Finalize plot
        plt.title(f"Ensemble Predictions - {sample_id}\nExtended to {max_days} ({time_label})")
        plt.xlabel(time_label)
        plt.ylabel("Cu Recovery (%)")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.ylim(0, 100)
        plt.xlim(left=0) # int(np.ceil(max(tt_tensor)/30)*30)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"ensemble_predictions_{sample_id}_withoutReactors.png"), dpi=300)
        plt.close()
        print(f"Saved ensemble prediction plot for {sample_id}")


    # Save ensemble_prediction_summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(save_dir, 'ensemble_prediction_summary_withoutReactors.csv'), index=False)
    print(f"Saved ensemble_prediction_summary with {len(summary_df)} entries")
    
    interval_payload = {
        'records': interval_records,
        'metadata': {
            'z_base': z_base,
            'uncertainty_scale': uncert_scale,
            'effective_z': z_effective,
            'nominal_coverage_pct': nominal_cov * 100.0 if np.isfinite(nominal_cov) else np.nan,
            'prediction_interval_bounds': pi_bounds,
        }
    }
    interval_path = os.path.join(save_dir, 'ensemble_prediction_intervals_withoutReactors.joblib')
    interval_payload['artifact_path'] = interval_path
    joblib.dump(interval_payload, interval_path)
    print(f"Saved interval payload with {len(interval_records)} curves to {interval_path}")
    
    return summary_df, interval_payload

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
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter to finite values to avoid histogram errors
    finite_df = results_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['rmse', 'confidence_level', 'bias'])

    # 1. Performance metrics distribution
    fig, axes = plt.subplots(2, 2, figsize=(2.5*4, 2.5*3), dpi=300)
    
    # RMSE distribution
    rmse_vals = finite_df['rmse'].dropna()
    if len(rmse_vals) > 0:
        axes[0, 0].hist(rmse_vals, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(rmse_vals.mean(), color='red', linestyle='--', 
                           label=f'Mean: {rmse_vals.mean():.1f}')
        axes[0, 0].axvline(round(CONFIG['rmse_threshold'], 1), color='black', linestyle='--', alpha=0.5, label='RMSE Threshold')
        axes[0, 0].legend()
    axes[0, 0].set_xlabel('RMSE')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('RMSE Distribution')
    axes[0, 0].set_xlim(left=0, right=25)
    axes[0, 0].grid(True, alpha=0.3)
    # Confidence Level distribution
    cl_vals = finite_df['confidence_level'].dropna()
    if len(cl_vals) > 0:
        axes[0, 1].hist(cl_vals, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(cl_vals.mean(), color='red', linestyle='--', 
                           label=f'Mean: {cl_vals.mean():.1f}')
        axes[0, 1].legend()
    axes[0, 1].set_xlabel('Confidence Level (%)')
    axes[0, 1].axvline(round(CONFIG['confidence_target'], 1), color='black', linestyle='--', alpha=0.5, label='Confidence Level Target')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Confidence Level Distribution')
    axes[0, 1].set_xlim(left=0, right=100)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bias distribution
    bias_vals = finite_df['bias'].dropna()
    if len(bias_vals) > 0:
        axes[1, 0].hist(bias_vals, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(bias_vals.mean(), color='red', linestyle='--', 
                           label=f'Mean: {bias_vals.mean():.1f}')
        axes[1, 0].axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero bias')
        axes[1, 0].legend()
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
    plt.savefig(os.path.join(save_dir, 'model_stats_diagnostics_withoutReactors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Diagnostic plots saved to {save_dir}")

def plot_reactor_scaling_diagnostics_uncertainty(results_df, save_dir):
    """
    Create diagnostic plots for reactor scaling model
    """
    print("Creating uncertainty diagnostic plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Performance metrics distribution
    fig, axes = plt.subplots(2, 2, figsize=(2.5*4, 2.5*3), dpi=300)
    
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
    plt.savefig(os.path.join(save_dir, 'model_stats_diagnostics_withoutReactors_withUncertainty.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Diagnostic plots saved to {save_dir}")


def plot_diagnostics_control_catalyzed(results_df, save_dir):
    """
    Create diagnostic plots for reactor scaling model
    """
    os.makedirs(save_dir, exist_ok=True)

    for x in ['Control', 'Catalyzed']:
        # Filter for control samples only
        sample_plot = results_df[results_df['sample_col_id'] == x]
        
        # 1. Performance metrics distribution
        fig, axes = plt.subplots(2, 2, figsize=(2.5*4, 2.5*3), dpi=300)
        
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
        plt.savefig(os.path.join(save_dir, f'model_stats_diagnostics_{x}_withoutReactors.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Diagnostic plots saved to {save_dir}")


def build_runs_from_config(base_config):
    """
    Build the list of training runs based on CONFIG['special_feats']['time_feat'].
    Supports one or multiple time features; deduplicates while preserving order.
    """
    time_feats = base_config.get('special_feats', {}).get('time_feat', [])
    if isinstance(time_feats, str):
        time_feats = [time_feats]
    if not time_feats:
        time_feats = ['leach_duration_days']

    # Preserve order, remove duplicates
    seen = set()
    ordered_time_feats = []
    for tf in time_feats:
        if tf not in seen:
            ordered_time_feats.append(tf)
            seen.add(tf)

    defaults = {
        "cumulative_lixiviant_m3_t": {"name": "lixiviant", "max_days": 30},
        "leach_duration_days": {"name": "leach_days", "max_days": 2500},
    }
    fallback_max_days = base_config.get('recovery_max_time', 1600)

    runs = []
    for tf in ordered_time_feats:
        meta = defaults.get(tf, {})
        runs.append({
            "name": meta.get("name", tf),
            "time_feat": tf,
            "max_days": meta.get("max_days", fallback_max_days),
        })
    return runs


def compute_and_save_shap_outputs(
    scaling_models,
    X_tensor,
    catalyst_tensor,
    num_cols,
    shap_save_dir,
    config,
    run_name="",
):
    """
    Compute SHAP values and importance plots for a single run.
    """
    run_label = f" ({run_name})" if run_name else ""
    print(f"\nComputing SHAP explanations for feature impacts on predicted parameters{run_label}...")

    shap_background_size = config.get('shap_background_size', 100)
    shap_sample_size = config.get('shap_sample_size', 100)

    # Separate samples by catalyst presence
    catalyst_present = np.array([torch.any(cat > 0).item() for cat in catalyst_tensor])
    non_catalyst_indices = np.where(~catalyst_present)[0]
    catalyst_indices = np.where(catalyst_present)[0]

    print(f"  Total samples: {len(X_tensor)}")
    print(f"  Non-catalyst samples: {len(non_catalyst_indices)}")
    print(f"  Catalyst samples: {len(catalyst_indices)}")

    # Model wrappers for SHAP (base vs catalyst parameters)
    class BaseModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            catalyst_dummy = torch.zeros(x.shape[0], 1, device=x.device)
            params = self.model(x, catalyst_dummy)
            return params[:, :4]  # Only return base parameters (a1, b1, a2, b2)

    class CatalystModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            catalyst_dummy = torch.ones(x.shape[0], 1, device=x.device)
            params = self.model(x, catalyst_dummy)
            return params[:, 4:8]  # Only return catalyst amplitudes/rates (a3, b3, a4, b4)

    # Compute SHAP for base parameters (a1, b1, a2, b2)
    print("  Computing SHAP for base parameters (a1, b1, a2, b2)...")
    if len(non_catalyst_indices) > 0:
        background_indices_base = random.sample(
            list(non_catalyst_indices),
            min(shap_background_size, len(non_catalyst_indices))
        )
        explain_indices_base = random.sample(
            list(non_catalyst_indices),
            min(shap_sample_size, len(non_catalyst_indices))
        )
        X_background_base = X_tensor[background_indices_base]
        X_explain_base = X_tensor[explain_indices_base]

        all_shap_values_base = []
        for model in scaling_models.models:
            wrapped_model = BaseModelWrapper(model).to(device)
            explainer = shap.DeepExplainer(wrapped_model, X_background_base)
            shap_values_model = explainer.shap_values(X_explain_base, check_additivity=False)
            if isinstance(shap_values_model, np.ndarray) and shap_values_model.ndim == 3:
                shap_values_model = [shap_values_model[..., k] for k in range(shap_values_model.shape[-1])]
            all_shap_values_base.append(shap_values_model)

        weighted_shap_base = []
        for output_dim in range(4):
            output_shap = sum(
                scaling_models.weights[m] * all_shap_values_base[m][output_dim]
                for m in range(len(scaling_models.models))
            )
            weighted_shap_base.append(output_shap)
    else:
        weighted_shap_base = [np.zeros((1, len(num_cols)))] * 4
        X_explain_base = X_tensor[:1]  # Dummy for consistency
        print("  ⚠️  No non-catalyst samples found for base parameters!")

    # Compute SHAP for catalyst parameters (a3, b3, a4, b4)
    print("  Computing SHAP for catalyst parameters (a3, b3, a4, b4)...")
    if len(catalyst_indices) > 0:
        background_indices_cat = random.sample(
            list(catalyst_indices),
            min(shap_background_size, len(catalyst_indices))
        )
        explain_indices_cat = random.sample(
            list(catalyst_indices),
            min(shap_sample_size, len(catalyst_indices))
        )
        X_background_cat = X_tensor[background_indices_cat]
        X_explain_cat = X_tensor[explain_indices_cat]

        all_shap_values_cat = []
        for model in scaling_models.models:
            wrapped_model = CatalystModelWrapper(model).to(device)
            explainer = shap.DeepExplainer(wrapped_model, X_background_cat)
            shap_values_model = explainer.shap_values(X_explain_cat, check_additivity=False)
            if isinstance(shap_values_model, np.ndarray) and shap_values_model.ndim == 3:
                shap_values_model = [shap_values_model[..., k] for k in range(shap_values_model.shape[-1])]
            all_shap_values_cat.append(shap_values_model)

        weighted_shap_cat = []
        for output_dim in range(4):
            output_shap = sum(
                scaling_models.weights[m] * all_shap_values_cat[m][output_dim]
                for m in range(len(scaling_models.models))
            )
            weighted_shap_cat.append(output_shap)
    else:
        weighted_shap_cat = [np.zeros((1, len(num_cols)))] * 4
        X_explain_cat = X_tensor[:1]  # Dummy for consistency
        print("  ⚠️  No catalyst samples found for catalyst parameters!")

    # Combine SHAP values for all 8 parameters
    weighted_shap = weighted_shap_base + weighted_shap_cat
    param_names = ['a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'a4', 'b4']

    # Aggregate: Mean absolute SHAP per feature per parameter (overall impact)
    importance_dict = {}
    for output_dim, param in enumerate(param_names):
        mean_abs_shap = np.mean(np.abs(weighted_shap[output_dim]), axis=0)
        importance_dict[param] = mean_abs_shap

    # Create DataFrame: rows=features, columns=parameters, values=mean abs SHAP
    shap_save_dir = shap_save_dir or os.path.join(folder_path_save, 'plots')
    os.makedirs(shap_save_dir, exist_ok=True)
    importance_df = pd.DataFrame(importance_dict, index=num_cols)
    importance_df.to_csv(os.path.join(shap_save_dir, 'feature_impact_on_parameters.csv'))
    print(f"Saved feature impact CSV (mean abs SHAP) to {shap_save_dir}/feature_impact_on_parameters.csv")

    # Visualizations
    plots_dir = shap_save_dir

    # Bar plots: Importance per parameter
    for output_dim, param in enumerate(param_names):
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(importance_df[param])[::-1]
        plt.barh(np.array(num_cols)[sorted_idx], importance_df[param].values[sorted_idx])
        plt.xlabel('Mean Absolute SHAP Value (Impact)')
        plt.title(f'Feature Impact on {param}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'shap_importance_{param}.png'), dpi=300)
        plt.close()

    # Beeswarm plots for distribution (one per parameter)
    for output_dim, param in enumerate(param_names):
        if output_dim < 4:
            X_for_plot = X_explain_base.cpu().numpy()
            shap_for_plot = weighted_shap[output_dim]
        else:
            X_for_plot = X_explain_cat.cpu().numpy()
            shap_for_plot = weighted_shap[output_dim]

        if shap_for_plot.shape[0] > 0 and np.any(np.abs(shap_for_plot) > 1e-10):
            shap.summary_plot(shap_for_plot, features=X_for_plot, feature_names=num_cols, show=False)
            plt.title(f'SHAP Beeswarm for {param}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'shap_beeswarm_{param}.png'), dpi=300)
            plt.close()
        else:
            print(f"  ⚠️  Skipping beeswarm plot for {param} (no valid SHAP values)")

    print(f"SHAP analysis complete for {run_name or 'current run'}! Plots saved to {plots_dir}")

    # Generate overall summary importance plots per predictor variable
    overall_importance = importance_df.sum(axis=1) / importance_df.sum(axis=1).sum() * 100
    overall_plot = pd.DataFrame({'importance': overall_importance}).sort_values('importance', ascending=False)
    ax = sns.barplot(data=overall_plot, x='importance', y=overall_plot.index, palette='viridis')
    for patch in ax.patches:
        ax.text(
            patch.get_width() - 0.5,
            patch.get_y() + patch.get_height() / 2,
            f"{patch.get_width():.0f}%",
            ha='right',
            va='center',
            color='white'
        )
    plt.title('SHAP overall % importance for all parameters')
    plt.tight_layout()
    plt.ylabel('Feature')
    plt.xlabel('% importance')
    plt.savefig(os.path.join(plots_dir, 'percentual_feature_importance_overall.png'), dpi=300)
    plt.show()
    plt.close()

    # Generate a-control params summary importance plots per predictor variable
    a_control_cols = [col for col in importance_df.columns if col.startswith('a1') or col.startswith('a2')]
    importance_df_a = importance_df[a_control_cols]
    overall_importance = importance_df_a.sum(axis=1) / importance_df_a.sum(axis=1).sum() * 100
    overall_plot = pd.DataFrame({'importance': overall_importance}).sort_values('importance', ascending=False)
    ax = sns.barplot(data=overall_plot, x='importance', y=overall_plot.index, palette='viridis')
    for patch in ax.patches:
        ax.text(
            patch.get_width() - 0.5,
            patch.get_y() + patch.get_height() / 2,
            f"{patch.get_width():.0f}%",
            ha='right',
            va='center',
            color='white'
        )
    plt.title('SHAP % importance for Control a-params')
    plt.tight_layout()
    plt.ylabel('Feature')
    plt.xlabel('% importance')
    plt.savefig(os.path.join(plots_dir, 'percentual_feature_importance_a-control-params.png'), dpi=300)
    plt.show()
    plt.close()

    # Generate a-catalyzed params summary importance plots per predictor variable
    a_catalyzed_cols = [col for col in importance_df.columns if col.startswith('a3') or col.startswith('a4')]
    importance_df_a = importance_df[a_catalyzed_cols]
    overall_importance = importance_df_a.sum(axis=1) / importance_df_a.sum(axis=1).sum() * 100
    overall_plot = pd.DataFrame({'importance': overall_importance}).sort_values('importance', ascending=False)
    ax = sns.barplot(data=overall_plot, x='importance', y=overall_plot.index, palette='viridis')
    for patch in ax.patches:
        ax.text(
            patch.get_width() - 0.5,
            patch.get_y() + patch.get_height() / 2,
            f"{patch.get_width():.0f}%",
            ha='right',
            va='center',
            color='white'
        )
    plt.title('SHAP % importance for Catalyzed a-params')
    plt.tight_layout()
    plt.ylabel('Feature')
    plt.xlabel('% importance')
    plt.savefig(os.path.join(plots_dir, 'percentual_feature_importance_a-catalyzed-params.png'), dpi=300)
    plt.show()
    plt.close()

    # Generate b-control params summary importance plots per predictor variable
    b_control_cols = [col for col in importance_df.columns if col.startswith('b1') or col.startswith('b2')]
    importance_df_b = importance_df[b_control_cols]
    overall_importance = importance_df_b.sum(axis=1) / importance_df_b.sum(axis=1).sum() * 100
    overall_plot = pd.DataFrame({'importance': overall_importance}).sort_values('importance', ascending=False)
    ax = sns.barplot(data=overall_plot, x='importance', y=overall_plot.index, palette='viridis')
    for patch in ax.patches:
        ax.text(
            patch.get_width() - 0.5,
            patch.get_y() + patch.get_height() / 2,
            f"{patch.get_width():.0f}%",
            ha='right',
            va='center',
            color='white'
        )
    plt.title('SHAP % importance for Control for b-params')
    plt.tight_layout()
    plt.ylabel('Feature')
    plt.xlabel('% importance')
    plt.savefig(os.path.join(plots_dir, 'percentual_feature_importance_b-control-params.png'), dpi=300)
    plt.show()
    plt.close()

    # Generate b-catalyzed params summary importance plots per predictor variable
    b_catalyzed_cols = [col for col in importance_df.columns if col.startswith('b3') or col.startswith('b4')]
    importance_df_b = importance_df[b_catalyzed_cols]
    overall_importance = importance_df_b.sum(axis=1) / importance_df_b.sum(axis=1).sum() * 100
    overall_plot = pd.DataFrame({'importance': overall_importance}).sort_values('importance', ascending=False)
    ax = sns.barplot(data=overall_plot, x='importance', y=overall_plot.index, palette='viridis')
    for patch in ax.patches:
        ax.text(
            patch.get_width() - 0.5,
            patch.get_y() + patch.get_height() / 2,
            f"{patch.get_width():.0f}%",
            ha='right',
            va='center',
            color='white'
        )
    plt.title('SHAP % importance for Catalyzed b-params')
    plt.tight_layout()
    plt.ylabel('Feature')
    plt.xlabel('% importance')
    plt.savefig(os.path.join(plots_dir, 'percentual_feature_importance_b-catalyzed-params.png'), dpi=300)
    plt.show()
    plt.close()


#%% ---------------------------
# MAIN EXECUTION
# ---------------------------

if __name__ == "__main__":
    print("Starting Copper Recovery PINN without Reactors")
    print("=" * 60)

    runs = build_runs_from_config(CONFIG)

    combined_artifacts = {}
    individual_model_filenames = {
        "leach_days": "AdaptiveTwoPhaseModel_withoutReactors_leachdays.pt",
        "lixiviant": "AdaptiveTwoPhaseModel_withoutReactors_lix.pt",
    }

    for run in runs:
        run_config = copy.deepcopy(CONFIG)
        run_config['special_feats']['time_feat'] = [run["time_feat"]]
        # Force plot generation for the final (best) training run
        run_config['generate_plots'] = True
        seeds = run_config.get('pytorch_seeds') or [None]
        if not isinstance(seeds, (list, tuple)):
            seeds = [seeds]

        run_save_dir = os.path.join(folder_path_save, 'plots', run["name"])
        os.makedirs(run_save_dir, exist_ok=True)

        print(f"\n=== Running mode: {run['name']} (time_feat={run['time_feat']}) ===")
        # Seed once before data prep to keep preprocessing stable across runs
        set_all_seeds(seeds[0], run_config.get('use_deterministic_algorithms', True))
        df_columns_filtered = filter_column_dataset_by_config(df_model_recCu_catcontrol_projects, run_config)
        df_columns_aug = df_columns_filtered.copy()

        X_tensor, y_tensor, time_tensor, catalyst_tensor, tt_tensor, sample_ids, sample_col_ids, feature_weights, num_cols, scaler_X, out_df_unscaled = prepare_column_train_data(
            df=df_columns_aug,
            config=run_config,
            output_type='averaged',
            fill_noncat_averages=False
        )
        out_df_unscaled.to_csv(os.path.join(run_save_dir, 'processed_data_unscaled.csv'), index=False)
        # Feature metadata (min/max) for predictors
        feature_metadata = []
        for col in num_cols:
            col_min = float(out_df_unscaled[col].min()) if col in out_df_unscaled.columns else 0.0
            col_max = float(out_df_unscaled[col].max()) if col in out_df_unscaled.columns else 0.0
            feature_metadata.append({'name': col, 'min': col_min, 'max': col_max})

        seed_ensembles = []
        seed_results_frames = []
        combined_states = []
        combined_val_losses = []
        combined_best_configs = []

        for seed in seeds:
            seed_config = copy.deepcopy(run_config)
            seed_config['pytorch_seed'] = seed
            seed_config['generate_plots'] = True
            seed_save_dir = run_save_dir if len(seeds) == 1 else os.path.join(run_save_dir, f"seed_{seed}")
            os.makedirs(seed_save_dir, exist_ok=True)

            # Seed before each multi-seed training pass
            set_all_seeds(seed, seed_config.get('use_deterministic_algorithms', True))

            scaling_models_seed, scaling_results_seed = train_reactor_scaling_model_parallel(
                X_tensor=X_tensor,
                y_tensor=y_tensor,
                time_tensor=time_tensor,
                catalyst_tensor=catalyst_tensor,
                tt_tensor=tt_tensor,
                sample_ids=sample_ids,
                sample_col_ids=sample_col_ids,
                num_cols=num_cols,
                config=seed_config,
                device=device,
                save_dir=seed_save_dir,
                use_val_data_days=seed_config.get('use_val_data_days', 0)
            )

            if isinstance(scaling_results_seed, pd.DataFrame):
                scaling_results_seed = scaling_results_seed.copy()
                scaling_results_seed['seed'] = seed

            seed_ensembles.append(scaling_models_seed)
            seed_results_frames.append(scaling_results_seed)
            combined_states.extend(getattr(scaling_models_seed, 'model_states', []))
            combined_val_losses.extend(getattr(scaling_models_seed, 'val_losses_raw', []))
            combined_best_configs.extend(getattr(scaling_models_seed, 'best_configs', []))

        scaling_results = pd.concat(seed_results_frames, ignore_index=True) if len(seed_results_frames) > 1 else seed_results_frames[0]

        if len(seed_ensembles) > 1:
            if len(combined_states) == 0:
                print(f"No models collected across seeds for run {run['name']}; skipping.")
                continue
            if len(combined_val_losses) == 0:
                combined_val_losses = [1.0] * len(combined_states)
            scaling_models = EnsembleModels(
                model_states=combined_states,
                val_losses=combined_val_losses,
                total_features=len(num_cols),
                config=run_config,
                device=device,
                best_configs=combined_best_configs,
                num_cols=num_cols
            )
            scaling_results.to_csv(os.path.join(run_save_dir, 'column_prediction_summary_validation_withoutReactors_all_seeds.csv'), index=False)
        else:
            scaling_models = seed_ensembles[0]

        final_config = copy.deepcopy(run_config)
        final_config['generate_plots'] = True

        try:
            cal_scale = calibrate_uncertainty_scale(
                scaling_models, X_tensor, y_tensor, time_tensor, catalyst_tensor, tt_tensor, sample_ids,
                target_cov=final_config['target_predictive_interval_coverage']
            )
            print(f"Calibrated uncertainty_scale: {cal_scale:.3f}")
            final_config['uncertainty_scale'] = cal_scale
        except Exception as e:
            print(f"Uncertainty calibration skipped ({e}). Using scale=1.0")

        ensemble_summary, interval_payload = create_ensemble_predictions_with_uncertainty(
            ensemble=scaling_models,
            X_tensor=X_tensor,
            y_tensor=y_tensor,
            time_tensor=time_tensor,
            catalyst_tensor=catalyst_tensor,
            tt_tensor=tt_tensor,
            sample_ids=sample_ids,
            sample_col_ids=sample_col_ids,
            save_dir=run_save_dir,
            max_days=run["max_days"],
            config=final_config
        )

        # Diagnostic plots (final/best training)
        plot_reactor_scaling_diagnostics(scaling_results, run_save_dir)
        plot_reactor_scaling_diagnostics_uncertainty(scaling_results, run_save_dir)
        plot_diagnostics_control_catalyzed(scaling_results, run_save_dir)

        # Store run artifacts to combine into a single checkpoint file
        combined_artifacts[run["name"]] = {
            'models': scaling_models,
            'results': scaling_results,
            'num_cols': num_cols,
            'feature_names': list(num_cols),  # explicit copy of predictor names/order
            'scaler_X': scaler_X,
            'uncertainty_scale': final_config.get('uncertainty_scale', 1.0),
            'config': final_config,
            'feature_metadata': feature_metadata,
            'predictive_intervals': interval_payload,
            'predictive_interval_metadata': interval_payload.get('metadata', {}),
            'seeds_used': seeds,
        }

        print(f"\nMode {run['name']} complete. Results saved to: {run_save_dir}")
        print(f"Number of models trained: {scaling_models.get_ensemble_info()['num_models']}")
        print(f"Average RMSE: {scaling_results['rmse'].mean():.4f}")
        print(f"Average R²: {scaling_results['r2'].mean():.4f}")
        print(f"Average Bias: {scaling_results['bias'].mean():.4f}")

        compute_and_save_shap_outputs(
            scaling_models=scaling_models,
            X_tensor=X_tensor,
            catalyst_tensor=catalyst_tensor,
            num_cols=num_cols,
            shap_save_dir=run_save_dir,
            config=run_config,
            run_name=run["name"],
        )

        # Save individual checkpoint per time feature
        file_name = individual_model_filenames.get(
            run["name"],
            f"AdaptiveTwoPhaseModel_withoutReactors_{run['name']}.pt"
        )
        model_path = os.path.join(folder_path_save, file_name)
        torch.save(combined_artifacts[run["name"]], model_path)
        print(f"Saved {run['name']} model checkpoint to: {model_path}")

    combined_model_path = os.path.join(folder_path_save, "AdaptiveTwoPhaseModels_withoutReactors.pt")
    torch.save(combined_artifacts, combined_model_path)
    print(f"\nCombined model checkpoint saved to: {combined_model_path}")
    print(f"Includes runs: {', '.join(combined_artifacts.keys())}")

    print("\nReactor Scaling Model training completed successfully for all modes!")
    print("All results and plots have been saved.")

# %%
