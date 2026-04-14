"""
STANDALONE INTERACTIVE VERSION - Using Plotly Dash

This script creates a web-based interactive dashboard that runs in your browser.
No Jupyter notebook required!

Requirements:
    pip install dash plotly

Usage:
    python Example_usage_NN_v15_interactive_dash.py
    
Then open your browser to: http://127.0.0.1:8050/
"""

import torch
import torch.nn as nn
import os
import torch.serialization
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Device configuration
device = torch.device('cpu')
CONFIG = {
    'cat_effect_power': 0.7,
    'cat_rate_gain_b3': 0.3,
    'cat_rate_gain_b4': 0.1,
    'cat_additional_scale': 0.5,
    'total_asymptote_cap': 95.0,
}

# ===========================================================================
# Load combined models and per-run datasets
COMBINED_MODEL_PATH = '/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Rosetta/NN_PyTorch/AdaptiveTwoPhaseModels_withoutReactors.pt'
RUN_BASE_DIR = '/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Rosetta/NN_PyTorch/plots'

# ============================================================================
# STEP 0: Define the model classes (REQUIRED for unpickling)
# ============================================================================

def get_feature_weight_signs(config, feature_names):
    """
    Extract parameter-specific weight signs from CONFIG.
    
    Returns a matrix of shape [num_features, 8] where each row contains the signs
    for a feature's effect on each parameter (a1, b1, a2, b2, a3, b3, a4, b4).
    
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
                if w > 0:
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

def project_params_to_caps(params: torch.Tensor,
                           base_cap: float,
                           total_cap: float,
                           base_rate_cap: float,
                           total_rate_cap: float) -> torch.Tensor:
    """Projection enforcing amplitude and kinetic constraints (mirrors training)."""
    if params.ndim == 1:
        params = params.unsqueeze(0)

    a1, b1, a2, b2, a3, b3, a4, b4 = [params[:, i] for i in range(8)]

    # Amplitude base cap
    base_sum = a1 + a2
    scale_base_amp = torch.where(
        base_sum > base_cap,
        base_cap / base_sum.clamp(min=1e-6),
        torch.ones_like(base_sum)
    )
    a1 = a1 * scale_base_amp
    a2 = a2 * scale_base_amp

    # Amplitude total cap (scale catalyst amplitudes only)
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

    # Rate base cap
    rate_base_sum = b1 + b2
    scale_base_rate = torch.where(
        rate_base_sum > base_rate_cap,
        base_rate_cap / rate_base_sum.clamp(min=1e-9),
        torch.ones_like(rate_base_sum)
    )
    b1 = b1 * scale_base_rate
    b2 = b2 * scale_base_rate

    # Rate total cap (scale catalyst rates only)
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

    return torch.stack([a1, b1, a2, b2, a3, b3, a4, b4], dim=1)

class EnsembleModels:
    """Ensemble model for reactor scaling with uncertainty quantification"""
    
    def __init__(self, model_states, val_losses, total_features, config, device, best_configs, num_cols):
        self.device = device
        self.total_features = total_features
        self.config = config
        self.best_configs = best_configs
        
        self.feature_weight_signs = get_feature_weight_signs(config, num_cols).to(device)
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
        dropout_rate = config.get('pytorch_dropout_rate', 0.33)
        
        for idx, (model_state, val_loss) in enumerate(zip(model_states, val_losses)):
            if val_loss <= threshold:
                model = AdaptiveTwoPhaseRecoveryModel(
                    total_features=self.total_features,
                    hidden_dim=hidden_dim,  # Use overall best from final_config
                    dropout_rate=dropout_rate,  # Use overall best from final_config
                    init_mode=config.get('init_mode', 'kaiming'), # Use overall best init_mode
                    feature_weight_signs=self.feature_weight_signs,
                    config=self.config,
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
        w_sum = w.sum()
        w = w / (w_sum if np.isfinite(w_sum) and w_sum > 0 else 1.0)
        w_b = w[:, None, None]  # (M,1,1)

        # Weighted mean
        weighted_pred = (all_model_predictions * w_b).sum(axis=0)        # (B,T)
        weighted_params = (all_model_params * w[:, None, None]).sum(axis=0)  # (B,P)

        # Weighted std (epistemic)
        diff = all_model_predictions - weighted_pred[None, :, :]         # (M,B,T)
        var_w = (w_b * diff**2).sum(axis=0)                              # (B,T)
        uncertainty = np.sqrt(np.maximum(var_w, 0.0))                    # (B,T)

        return weighted_pred, uncertainty, weighted_params

    def get_ensemble_info(self):
        """Get ensemble information"""
        return {
            'num_models': len(self.models),
            'weights': self.weights,
            'total_features': self.total_features,
        }
    
class AdaptiveTwoPhaseRecoveryModel(nn.Module):
    """
    Enhanced Two-Phase Recovery Model with separate output heads for exact per-parameter control.
    Each parameter (a1, b1, a2, b2, a3, b3, a4, b4) has its own network head.
    """
    def __init__(self, total_features, hidden_dim=128, dropout_rate=0.33, init_mode='kaiming', feature_weight_signs=None, config=None):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.init_mode = init_mode
        self.total_features = total_features
        self.config = config or CONFIG
        
        # Store feature weight signs for monotonic constraints
        if feature_weight_signs is not None:
            self.register_buffer('feature_weight_signs', feature_weight_signs)
            self.use_monotonic_constraints = True
        else:
            self.register_buffer('feature_weight_signs', torch.zeros(total_features, 8))
            self.use_monotonic_constraints = False
        
        # Create 8 separate network heads (one for each parameter)
        # Parameters: a1, b1, a2, b2, a3, b3, a4, b4
        self.param_networks = nn.ModuleList()
        
        for param_idx in range(8):
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
            self.param_networks[1][-1].bias[0] = -2.7  # b1
            self.param_networks[2][-1].bias[0] = 0.0   # a2
            self.param_networks[3][-1].bias[0] = -2.7  # b2
            self.param_networks[4][-1].bias[0] = 0.0   # a3
            self.param_networks[5][-1].bias[0] = -2.7  # b3
            self.param_networks[6][-1].bias[0] = 0.0   # a4
            self.param_networks[7][-1].bias[0] = -2.7  # b4

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
        
        batch_size = x.size(0)
        
        # Initialize parameter tensor
        params = torch.zeros(batch_size, 8, device=x.device)
        cfg = getattr(self, 'config', CONFIG)
        lim = cfg.get('param_limits', {})
        a1_min, a1_max = lim.get('a1', (1.5, 75.0))
        b1_min, b1_max = lim.get('b1', (0.015, 2.1))
        a2_min, a2_max = lim.get('a2', (1.8, 71.0))
        b2_min, b2_max = lim.get('b2', (0.01, 2.0))
        a3_min, a3_max = lim.get('a3', (10.0, 60.0))
        b3_min, b3_max = lim.get('b3', (0.004, 1.1))
        a4_min, a4_max = lim.get('a4', (4.0, 40.0))
        b4_min, b4_max = lim.get('b4', (0.018, 1.7))
        base_cap = float(cfg.get('base_asymptote_cap', 80.0))
        total_cap = float(cfg.get('total_asymptote_cap', 95.0))
        base_rate_cap = float(cfg.get('base_rate_cap', 2.1))
        total_rate_cap = float(cfg.get('total_rate_cap', 7.0))
        
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
            
            a1 = 10.0 + 30.0 * torch.sigmoid(a1_raw).view(-1, 1)
            b1 = 0.001 + 0.1 * torch.sigmoid(b1_raw).view(-1, 1)
            a2 = a2_min + (a2_max - a2_min) * torch.sigmoid(a2_raw).view(-1, 1)
            b2 = b2_min + (b2_max - b2_min) * torch.sigmoid(b2_raw).view(-1, 1)
            
            params[:, :4] = torch.cat([a1, b1, a2, b2], dim=1)
        
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
                a3 = a3_min + (a3_max - a3_min) * torch.sigmoid(a3_raw).view(-1, 1)
                b3 = b3_min + (b3_max - b3_min) * torch.sigmoid(b3_raw).view(-1, 1)
                a4 = a4_min + (a4_max - a4_min) * torch.sigmoid(a4_raw).view(-1, 1)
                b4 = b4_min + (b4_max - b4_min) * torch.sigmoid(b4_raw).view(-1, 1)
                
                params[idx, 4:] = torch.cat([a3, b3, a4, b4], dim=1)
        else:
            params[:, 4:] = np.nan
        
        return params
    

def _scaled_sigmoid_torch(amplitude: torch.Tensor, rate: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Monotone logistic that is 0 at t=0 and approaches amplitude as t increases."""
    rate = rate.abs()
    z = torch.clamp(rate * t, min=-60.0, max=60.0)  # avoid overflow
    sig = torch.sigmoid(z)
    scaled = torch.clamp(2.0 * sig - 1.0, min=0.0, max=1.0)  # value=0 when t=0
    return amplitude.abs() * scaled


def generate_two_phase_recovery_sigmoid(time, catalyst, transition_time, params, cfg=None):
    """
    Sigmoid-based recovery aligned with training code:
    - Control: logistic terms anchored at 0, capped by config total_asymptote_cap.
    - Catalyzed: smooth additive logistic terms starting at transition, scaled by catalyst_effect.
    """
    cfg = cfg or CONFIG
    time = time.to(params.device)
    catalyst = catalyst.to(params.device)

    catalyst_effect = catalyst / (catalyst + 1)
    ce_pow = float(cfg.get('cat_effect_power', 1.0))
    if ce_pow != 1.0:
        catalyst_effect = catalyst_effect.pow(ce_pow)

    a1 = params[:, 0].unsqueeze(1)
    b1 = params[:, 1].unsqueeze(1)
    a2 = params[:, 2].unsqueeze(1)
    b2 = params[:, 3].unsqueeze(1)

    t_eff = torch.clamp(time, min=0.0)
    recovery_control = _scaled_sigmoid_torch(a1, b1, t_eff) + _scaled_sigmoid_torch(a2, b2, t_eff)
    recovery = recovery_control.clone()

    has_catalyst = torch.any(catalyst > 0, dim=1, keepdim=True)  # [B,1] bool
    if transition_time.dim() == 0:
        transition_i = transition_time
    elif transition_time.dim() == 1:
        transition_i = transition_time.view(-1, 1)
    else:
        transition_i = transition_time

    if params.shape[1] > 4 and torch.any(~torch.isnan(params[:, 4:])):
        a3 = params[:, 4].unsqueeze(1)
        b3 = params[:, 5].unsqueeze(1)
        a4 = params[:, 6].unsqueeze(1)
        b4 = params[:, 7].unsqueeze(1)

        t_shift = torch.clamp(time - transition_i, min=0.0)
        gain_b3 = float(cfg.get('cat_rate_gain_b3', 0.0))
        gain_b4 = float(cfg.get('cat_rate_gain_b4', 0.0))
        rate_mult3 = 1.0 + gain_b3 * catalyst_effect
        rate_mult4 = 1.0 + gain_b4 * catalyst_effect

        term3 = _scaled_sigmoid_torch(a3, torch.abs(b3) * rate_mult3, t_shift)
        term4 = _scaled_sigmoid_torch(a4, torch.abs(b4) * rate_mult4, t_shift)
        additional = catalyst_effect * (term3 + term4)

        add_term = torch.where(has_catalyst, additional, torch.zeros_like(additional))
        recovery = recovery_control + add_term

    total_cap = float(cfg.get('total_asymptote_cap', 95.0))
    recovery = torch.clamp(recovery, min=0.0, max=total_cap)
    return recovery


def generate_two_phase_recovery(time, catalyst, transition_time, params, cfg=None):
    return generate_two_phase_recovery_sigmoid(time, catalyst, transition_time, params, cfg)


# ============================================================================
# Load combined models (after class definitions for safe unpickling)
# ============================================================================

torch.serialization.add_safe_globals([EnsembleModels, AdaptiveTwoPhaseRecoveryModel])
combined_checkpoint = torch.load(COMBINED_MODEL_PATH, map_location=device, weights_only=False)
available_runs = list(combined_checkpoint.keys())

def load_run_artifacts(run_name):
    artifacts = combined_checkpoint[run_name]
    df_path = os.path.join(RUN_BASE_DIR, run_name, 'processed_data_unscaled.csv')
    df = pd.read_csv(df_path)
    return artifacts, df


# ============================================================================
# Create Dash App
# ============================================================================

# Global config for all graphs
graph_config = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'cu_recovery_prediction',
        'height': 600,
        'width': 900,
        'scale': 3  # 3x resolution for high DPI
    }
}

# Run-specific settings
RUN_SETTINGS = {
    "lixiviant": {
        "time_label": "Cumulative Lixiviant (m3/ton ore)",
        "max_extend": 30,
        "time_col": "cumulative_lixiviant_m3_t",
        "start_max": 5,
        "start_step": 0.1,
        "start_default": 1.0,
        "time_default": 30,
        "dose_max": 0.2,
        "dose_step": 0.001,
        "dose_default": 0.02,
    },
    "leach_days": {
        "time_label": "Leach Duration (days)",
        "max_extend": 2500,
        "time_col": "leach_duration_days",
        "start_max": 500,
        "start_step": 10,
        "start_default": 250,
        "time_default": 2500,
        "dose_max": 0.5,
        "dose_step": 0.001,
        "dose_default": 0.05,
    },
}
DEFAULT_RUN = "lixiviant" if "lixiviant" in available_runs else (available_runs[0] if available_runs else "lixiviant")
default_settings = RUN_SETTINGS.get(DEFAULT_RUN, RUN_SETTINGS["lixiviant"])

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define slider configurations
slider_configs = {
    'acid_soluble': {'label': 'Acid Soluble %', 'min': 1.0, 'max': 26.0, 'step': 0.5, 'value': 3.5},
    'residual_cpy': {'label': 'Residual Cpy %', 'min': 23.0, 'max': 96.0, 'step': 0.5, 'value': 88.5},
    'material_size_p80': {'label': 'Material Size P80 (in)', 'min': 0.5, 'max': 8.0, 'step': 0.1, 'value': 1.0},
    'copper_sulfides': {'label': 'Copper Sulfides', 'min': 0.20, 'max': 1.3, 'step': 0.05, 'value': 0.55},
    'secondary_copper': {'label': 'Secondary Copper', 'min': 0.0, 'max': 0.06, 'step': 0.005, 'value': 0.01},
    'acid_gen_sulfides': {'label': 'Acid Gen. Sulfides', 'min': 0.0, 'max': 12.0, 'step': 0.5, 'value': 3.0},
    'gangue_silicates': {'label': 'Gangue Silicates', 'min': 87.0, 'max': 98.0, 'step': 0.1, 'value': 95.8},
    'fe_oxides': {'label': 'Fe Oxides', 'min': 0.0, 'max': 3.2, 'step': 0.1, 'value': 0.5},
    'carbonates': {'label': 'Carbonates', 'min': 0.0, 'max': 2.1, 'step': 0.1, 'value': 0.0},
    'catalyst_start_day': {'label': 'Catalyst Start Day', 'min': 0, 'max': default_settings['start_max'], 'step': default_settings['start_step'], 'value': default_settings.get('start_default', min(default_settings['start_max'], 5))},
    'catalyst_dose': {'label': 'Catalyst Dose (kg/t/day)', 'min': 0.0, 'max': default_settings['dose_max'], 'step': default_settings['dose_step'], 'value': default_settings['dose_default']},
    'max_time': {'label': 'Max Time (days)', 'min': 0, 'max': default_settings['max_extend'], 'step': 1 if default_settings['max_extend'] <= 100 else 100, 'value': default_settings.get('time_default', default_settings['max_extend'])},
}

# Compute initial values for dependent sliders
slider_configs['cyanide_soluble'] = {'label': 'Cyanide Soluble %', 'min': 3.0, 'max': 55.0, 'step': 0.5, 'value': 100 - slider_configs['acid_soluble']['value'] - slider_configs['residual_cpy']['value']}

sum_others = (slider_configs['copper_sulfides']['value'] + slider_configs['secondary_copper']['value'] + 
              slider_configs['acid_gen_sulfides']['value'] + slider_configs['fe_oxides']['value'] + 
              slider_configs['carbonates']['value'])
slider_configs['gangue_silicates']['value'] = 100 - sum_others

# Create layout
def create_slider_row(slider_id, config, disabled=False):
    return dbc.Row([
        dbc.Col(html.Label(f"{config['label']} [{config['min']}-{config['max']}]:"), width=4),
        dbc.Col(dcc.Slider(
            id=f'slider-{slider_id}',
            min=config['min'],
            max=config['max'],
            step=config['step'],
            value=config['value'],
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
            disabled=disabled
        ), width=6),
        dbc.Col(dcc.Input(
            id=f'input-{slider_id}',
            type='number',
            value=config['value'],
            min=config['min'],
            max=config['max'],
            step=config['step'],
            style={'width': '100%'},
            disabled=disabled
        ), width=2),
    ], className='mb-3')

app.layout = dbc.Container([
    html.H1("Cu Recovery Prediction - Interactive Dashboard", className='text-center my-4'),
    
    dbc.Row([
        # Left column - Controls
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Model Selection")),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='run-select',
                        options=[{'label': 'Lixiviant', 'value': 'lixiviant'},
                                 {'label': 'Leach Days', 'value': 'leach_days'}],
                        value=DEFAULT_RUN,
                        clearable=False
                    )
                ])
            ], className='mb-3'),
            
            dbc.Card([
                dbc.CardHeader(html.H5("Sample Features")),
                dbc.CardBody([
                    create_slider_row('acid_soluble', slider_configs['acid_soluble']),
                    create_slider_row('residual_cpy', slider_configs['residual_cpy']),
                    create_slider_row('cyanide_soluble', slider_configs['cyanide_soluble'], disabled=True),
                    create_slider_row('material_size_p80', slider_configs['material_size_p80']),
                    create_slider_row('copper_sulfides', slider_configs['copper_sulfides']),
                    create_slider_row('secondary_copper', slider_configs['secondary_copper']),
                    create_slider_row('acid_gen_sulfides', slider_configs['acid_gen_sulfides']),
                    create_slider_row('gangue_silicates', slider_configs['gangue_silicates'], disabled=True),
                    create_slider_row('fe_oxides', slider_configs['fe_oxides']),
                    create_slider_row('carbonates', slider_configs['carbonates']),
                ])
            ], className='mb-3'),
            
            dbc.Card([
                dbc.CardHeader(html.H5("Catalyst Parameters")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Label(id='label-catalyst-start', children=f"{slider_configs['catalyst_start_day']['label']} [{slider_configs['catalyst_start_day']['min']}-{slider_configs['catalyst_start_day']['max']}]:"), width=4),
                        dbc.Col(dcc.Slider(
                            id='slider-catalyst_start_day',
                            min=slider_configs['catalyst_start_day']['min'],
                            max=slider_configs['catalyst_start_day']['max'],
                            step=slider_configs['catalyst_start_day']['step'],
                            value=slider_configs['catalyst_start_day']['value'],
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ), width=6),
                        dbc.Col(dcc.Input(
                            id='input-catalyst_start_day',
                            type='number',
                            value=slider_configs['catalyst_start_day']['value'],
                            min=slider_configs['catalyst_start_day']['min'],
                            max=slider_configs['catalyst_start_day']['max'],
                            step=slider_configs['catalyst_start_day']['step'],
                            style={'width': '100%'},
                        ), width=2),
                    ], className='mb-3'),
                    create_slider_row('catalyst_dose', slider_configs['catalyst_dose']),
                ])
            ], className='mb-3'),
            
            dbc.Card([
                dbc.CardHeader(html.H5("Simulation Parameters")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Label(id='label-max-time', children=f"{slider_configs['max_time']['label']} [{slider_configs['max_time']['min']}-{slider_configs['max_time']['max']}]:"), width=4),
                        dbc.Col(dcc.Slider(
                            id='slider-max_time',
                            min=slider_configs['max_time']['min'],
                            max=slider_configs['max_time']['max'],
                            step=slider_configs['max_time']['step'],
                            value=slider_configs['max_time']['value'],
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ), width=6),
                        dbc.Col(dcc.Input(
                            id='input-max_time',
                            type='number',
                            value=slider_configs['max_time']['value'],
                            min=slider_configs['max_time']['min'],
                            max=slider_configs['max_time']['max'],
                            step=slider_configs['max_time']['step'],
                            style={'width': '100%'},
                        ), width=2),
                    ], className='mb-3'),
                ])
            ], className='mb-3'),
        ], width=5, style={'maxHeight': 'auto', 'overflowY': 'auto'}),
        
        # Right column - Plot and Results
        dbc.Col([
            dcc.Graph(id='prediction-plot', style={'height': '600px'}, config=graph_config),
            
            dbc.Card([
                dbc.CardHeader(html.H5("Predicted Parameters")),
                dbc.CardBody(html.Div(id='parameters-output'))
            ], className='mt-3'),
        ], width=7),
    ]),
    
], fluid=True)

# Create callbacks to sync sliders and inputs for non-dependent sliders
dependent_sliders = ['cyanide_soluble', 'gangue_silicates']
for slider_id in slider_configs.keys():
    if slider_id in dependent_sliders:
        continue
    
    @app.callback(
        [Output(f'slider-{slider_id}', 'value', allow_duplicate=True),
         Output(f'input-{slider_id}', 'value', allow_duplicate=True)],
        [Input(f'slider-{slider_id}', 'value'),
         Input(f'input-{slider_id}', 'value')],
        prevent_initial_call='initial_duplicate'
    )
    def sync_slider_input(slider_val, input_val, slider_id=slider_id):
        from dash import callback_context, no_update
        ctx = callback_context
        
        if not ctx.triggered:
            return no_update, no_update
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == f'slider-{slider_id}':
            return no_update, slider_val
        
        elif trigger_id == f'input-{slider_id}':
            if input_val is not None:
                config = slider_configs[slider_id]
                clamped_value = max(config['min'], min(config['max'], float(input_val)))
                return clamped_value, no_update
            return no_update, no_update
        
        return no_update, no_update

# Callback to update cyanide_soluble
@app.callback(
    [Output('slider-cyanide_soluble', 'value', allow_duplicate=True),
     Output('input-cyanide_soluble', 'value', allow_duplicate=True)],
    [Input('slider-acid_soluble', 'value'),
     Input('slider-residual_cpy', 'value')],
    prevent_initial_call='initial_duplicate'
)
def update_cyanide(acid, residual):
    cyanide = 100 - acid - residual
    return cyanide, cyanide

@app.callback(
    Output('input-cyanide_soluble', 'style'),
    Input('input-cyanide_soluble', 'value')
)
def update_cyanide_input_style(value):
    style = {'width': '100%'}
    if (value is not None and value < 3.0) or (value is not None and value > 55.0):
        style['color'] = 'red'
    return style

# Callback to update gangue_silicates
@app.callback(
    [Output('slider-gangue_silicates', 'value', allow_duplicate=True),
     Output('input-gangue_silicates', 'value', allow_duplicate=True)],
    [Input('slider-copper_sulfides', 'value'),
     Input('slider-secondary_copper', 'value'),
     Input('slider-acid_gen_sulfides', 'value'),
     Input('slider-fe_oxides', 'value'),
     Input('slider-carbonates', 'value')],
    prevent_initial_call='initial_duplicate'
)
def update_gangue(cu_sulf, sec_cu, acid_sulf, fe_ox, carb):
    sum_others = cu_sulf + sec_cu + acid_sulf + fe_ox + carb
    gangue = 100 - sum_others
    return gangue, gangue

@app.callback(
    Output('input-gangue_silicates', 'style'),
    Input('input-gangue_silicates', 'value')
)
def update_gangue_input_style(value):
    style = {'width': '100%'}
    if (value is not None and value < 87.0) or (value is not None and value > 98.0):
        style['color'] = 'red'
    return style

# Adjust time-related sliders based on selected run
@app.callback(
    [
        Output('slider-max_time', 'max'),
        Output('slider-max_time', 'step'),
        Output('slider-max_time', 'value', allow_duplicate=True),
        Output('slider-catalyst_start_day', 'max'),
        Output('slider-catalyst_start_day', 'value', allow_duplicate=True),
        Output('slider-catalyst_start_day', 'step'),
        Output('slider-catalyst_dose', 'max'),
        Output('slider-catalyst_dose', 'step'),
        Output('slider-catalyst_dose', 'value', allow_duplicate=True),
        Output('label-max-time', 'children'),
        Output('label-catalyst-start', 'children'),
        Output('input-max_time', 'value', allow_duplicate=True),
        Output('input-catalyst_start_day', 'value', allow_duplicate=True),
        Output('input-max_time', 'max', allow_duplicate=True),
        Output('input-catalyst_start_day', 'max', allow_duplicate=True),
    ],
    Input('run-select', 'value'),
    State('slider-max_time', 'value'),
    State('slider-catalyst_start_day', 'value'),
    State('slider-catalyst_dose', 'value'),
    prevent_initial_call='initial_duplicate',
)
def update_time_sliders(run_name, current_max_time, current_catalyst_start, current_dose):
    settings = RUN_SETTINGS.get(run_name, RUN_SETTINGS['leach_days'])
    max_ext = settings['max_extend']
    step = 1 if max_ext <= 100 else 100
    time_default = settings.get('time_default', max_ext)
    new_max_time = max_ext
    # Reset to run default
    new_time_value = time_default
    start_max = settings.get('start_max', max_ext)
    start_step = settings.get('start_step', step)
    new_catalyst_value = settings.get('start_default', start_max)
    dose_max = settings.get('dose_max', default_settings['dose_max'])
    dose_step = settings.get('dose_step', default_settings['dose_step'])
    dose_default = settings.get('dose_default', default_settings['dose_default'])
    new_dose_value = min(current_dose or dose_default, dose_max)
    label_max = f"{settings['time_label']} [{0}-{max_ext}]"
    label_start = f"{'Catalyst Start (m3/t)' if run_name == 'lixiviant' else 'Catalyst Start Day'} [{0}-{start_max}]"
    return (
        new_max_time, step, new_time_value,
        start_max, new_catalyst_value, start_step,
        dose_max, dose_step, new_dose_value,
        label_max, label_start, new_time_value, new_catalyst_value,
        new_max_time, start_max
    )

# Main prediction callback
@app.callback(
    [
        Output('prediction-plot', 'figure'),
        Output('parameters-output', 'children'),
        # slider mins/maxes/values and inputs (feature bounds)
        Output('slider-acid_soluble', 'min'),
        Output('slider-acid_soluble', 'max'),
        Output('slider-acid_soluble', 'value', allow_duplicate=True),
        Output('input-acid_soluble', 'value', allow_duplicate=True),
        Output('slider-residual_cpy', 'min'),
        Output('slider-residual_cpy', 'max'),
        Output('slider-residual_cpy', 'value', allow_duplicate=True),
        Output('input-residual_cpy', 'value', allow_duplicate=True),
        Output('slider-material_size_p80', 'min'),
        Output('slider-material_size_p80', 'max'),
        Output('slider-material_size_p80', 'value', allow_duplicate=True),
        Output('input-material_size_p80', 'value', allow_duplicate=True),
        Output('slider-copper_sulfides', 'min'),
        Output('slider-copper_sulfides', 'max'),
        Output('slider-copper_sulfides', 'value', allow_duplicate=True),
        Output('input-copper_sulfides', 'value', allow_duplicate=True),
        Output('slider-secondary_copper', 'min'),
        Output('slider-secondary_copper', 'max'),
        Output('slider-secondary_copper', 'value', allow_duplicate=True),
        Output('input-secondary_copper', 'value', allow_duplicate=True),
        Output('slider-acid_gen_sulfides', 'min'),
        Output('slider-acid_gen_sulfides', 'max'),
        Output('slider-acid_gen_sulfides', 'value', allow_duplicate=True),
        Output('input-acid_gen_sulfides', 'value', allow_duplicate=True),
        Output('slider-gangue_silicates', 'min'),
        Output('slider-gangue_silicates', 'max'),
        Output('slider-gangue_silicates', 'value', allow_duplicate=True),
        Output('input-gangue_silicates', 'value', allow_duplicate=True),
        Output('slider-fe_oxides', 'min'),
        Output('slider-fe_oxides', 'max'),
        Output('slider-fe_oxides', 'value', allow_duplicate=True),
        Output('input-fe_oxides', 'value', allow_duplicate=True),
        Output('slider-carbonates', 'min'),
        Output('slider-carbonates', 'max'),
        Output('slider-carbonates', 'value', allow_duplicate=True),
        Output('input-carbonates', 'value', allow_duplicate=True),
    ],
    [Input('run-select', 'value')] + [Input(f'slider-{key}', 'value') for key in slider_configs.keys()],
    prevent_initial_call='initial_duplicate'
)
def update_prediction(run_name, acid_soluble, residual_cpy, material_size_p80, copper_sulfides,
                     secondary_copper, acid_gen_sulfides, gangue_silicates, fe_oxides,
                     carbonates, catalyst_start_day, catalyst_dose, max_time, cyanide_soluble):
    # Load artifacts for selected run
    artifacts, _ = load_run_artifacts(run_name)
    ensemble_model = artifacts['models']
    scaler_X = artifacts['scaler_X']
    feature_names = artifacts.get('feature_names', artifacts.get('num_cols', []))
    num_cols = feature_names
    uncert_scale = artifacts.get('uncertainty_scale', 1.0)
    run_config = artifacts.get('config', {})
    CONFIG.update(run_config or {})
    settings = RUN_SETTINGS.get(run_name, RUN_SETTINGS['leach_days'])
    time_label = settings['time_label']
    time_col = settings['time_col']
    max_ext = settings['max_extend']
    start_max = settings.get('start_max', max_ext)
    max_time = min(max_time, max_ext)
    catalyst_start_day = min(catalyst_start_day, start_max)
    
    # Feature metadata for bounds
    meta = {m['name']: m for m in artifacts.get('feature_metadata', [])}

    # Map slider values to feature names; default 0.0 if not present
    slider_values = {
        'acid_soluble': acid_soluble,
        'residual_cpy': residual_cpy,
        'material_size_p80': material_size_p80,
        'copper_sulfides': copper_sulfides,
        'secondary_copper': secondary_copper,
        'acid_gen_sulfides': acid_gen_sulfides,
        'gangue_silicates': gangue_silicates,
        'fe_oxides': fe_oxides,
        'carbonates': carbonates,
    }
    feature_map = {
        'acid_soluble_%': 'acid_soluble',
        'residual_cpy_%': 'residual_cpy',
        'material_size_p80_in': 'material_size_p80',
        'grouped_copper_sulfides': 'copper_sulfides',
        'grouped_secondary_copper': 'secondary_copper',
        'grouped_acid_generating_sulfides': 'acid_gen_sulfides',
        'grouped_gangue_silicates': 'gangue_silicates',
        'grouped_fe_oxides': 'fe_oxides',
        'grouped_carbonates': 'carbonates',
    }
    sample_features = []
    feature_mins = []
    feature_maxs = []
    feature_values = []
    missing_in_sliders = []
    extra_sliders = []
    model_feature_set = set(feature_names)
    slider_feature_set = set(feature_map.values())
    for col in num_cols:
        slider_key = feature_map.get(col)
        if slider_key is None:
            missing_in_sliders.append(col)
            val = 0.0
        else:
            val = slider_values.get(slider_key, 0.0)
        m = meta.get(col, {})
        fmin = m.get('min', val)
        fmax = m.get('max', val)
        clamped = max(fmin, min(fmax, val))
        sample_features.append(clamped)
        feature_mins.append(fmin)
        feature_maxs.append(fmax)
        feature_values.append(clamped)
    # detect slider entries not used by model
    for sk in slider_feature_set:
        target_cols = [c for c, v in feature_map.items() if v == sk]
        if not any(tc in model_feature_set for tc in target_cols):
            extra_sliders.append(sk)
    if missing_in_sliders or extra_sliders:
        print(f"⚠️ Feature mapping mismatch. Missing in sliders: {missing_in_sliders}; Extra sliders not in model: {extra_sliders}")
    
    # Convert to DataFrame with ordered columns
    X_new_df = pd.DataFrame([sample_features], columns=num_cols)
    X_new_scaled = scaler_X.transform(X_new_df)
    X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)
    
    # Define time points
    time_points = np.linspace(0, max_time, int(max_time) + 1)
    time_tensor = torch.tensor(time_points, dtype=torch.float32).to(device).unsqueeze(0)
    
    # Control scenario
    catalyst_values_control = np.zeros_like(time_points)
    
    # Catalyzed scenario
    catalyst_values_catalyzed = np.zeros_like(time_points)
    for i, t in enumerate(time_points):
        if t >= catalyst_start_day:
            catalyst_values_catalyzed[i] = (t - catalyst_start_day) * catalyst_dose
    
    catalyst_tensor_control = torch.tensor(catalyst_values_control, dtype=torch.float32).to(device).unsqueeze(0)
    catalyst_tensor_catalyzed = torch.tensor(catalyst_values_catalyzed, dtype=torch.float32).to(device).unsqueeze(0)
    
    transition_time_control = torch.tensor([max_time], dtype=torch.float32).to(device)
    transition_time_catalyzed = torch.tensor([catalyst_start_day], dtype=torch.float32).to(device)
    
    # Make predictions
    mean_pred_control, uncertainty_control, params_control = ensemble_model.predict_with_params_and_uncertainty(
        X_new_tensor, catalyst_tensor_control, transition_time_control, time_tensor, sample_ids=None
    )
    
    mean_pred_catalyzed, uncertainty_catalyzed, params_catalyzed = ensemble_model.predict_with_params_and_uncertainty(
        X_new_tensor, catalyst_tensor_catalyzed, transition_time_catalyzed, time_tensor, sample_ids=None
    )
    
    # Convert to numpy
    mean_pred_control = mean_pred_control.squeeze()
    uncertainty_control = uncertainty_control.squeeze()
    mean_pred_catalyzed = mean_pred_catalyzed.squeeze()
    uncertainty_catalyzed = uncertainty_catalyzed.squeeze()
    
    # Nominal 90% band; allow optional calibration carried on the ensemble (if present)
    z_nom = 1.645
    z_score = z_nom * float(uncert_scale)
    
    fig = go.Figure()
    
    # Control scenario
    fig.add_trace(go.Scatter(
        x=time_points,
        y=mean_pred_control,
        mode='lines',
        name='Control',
        line=dict(color='royalblue', width=2),
        hovertemplate='''Control: %{y:.1f}%<extra></extra>'''
    ))

    lower_control = mean_pred_control - z_score * uncertainty_control
    upper_control = mean_pred_control + z_score * uncertainty_control
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_points, time_points[::-1]]),
        y=np.concatenate([upper_control, lower_control[::-1]]),
        fill='toself',
        fillcolor='rgba(65, 105, 225, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'90%CI (±{np.round(z_score, 2)}σ)',
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Control CI range (invisible trace for hover info)
    fig.add_trace(go.Scatter(
        x=time_points,
        y=mean_pred_control,  # Same y as main line
        mode='lines',
        line=dict(color='rgba(255,255,255,0)'),
        fillcolor='rgba(65, 105, 225, 0.2)',
        name=f'90%CI (±{np.round(z_score, 2)}σ)',
        customdata=np.column_stack([lower_control, upper_control]),
        hovertemplate='90%CI: [%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>',
        showlegend=False
    ))
    
    # Catalyzed scenario (only after transition)
    post_transition_mask = time_points >= catalyst_start_day
    time_points_filtered = time_points[post_transition_mask]
    mean_pred_catalyzed_filtered = mean_pred_catalyzed[post_transition_mask]
    uncertainty_catalyzed_filtered = uncertainty_catalyzed[post_transition_mask]
    
    if len(time_points_filtered) > 0:
        fig.add_trace(go.Scatter(
            x=time_points_filtered,
            y=mean_pred_catalyzed_filtered,
            mode='lines',
            name='Catalyzed',
            line=dict(color='darkorange', width=2),
            hovertemplate='''Catalyzed: %{y:.1f}%<extra></extra>'''
        ))

        lower_catalyzed = mean_pred_catalyzed_filtered - z_score * uncertainty_catalyzed_filtered
        upper_catalyzed = mean_pred_catalyzed_filtered + z_score * uncertainty_catalyzed_filtered
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([time_points_filtered, time_points_filtered[::-1]]),
            y=np.concatenate([upper_catalyzed, lower_catalyzed[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 140, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'90%CI (±{np.round(z_score, 2)}σ)',
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Catalyzed CI range (invisible trace for hover info)
        fig.add_trace(go.Scatter(
            x=time_points_filtered,
            y=mean_pred_catalyzed_filtered,  # Same y as main line
            mode='lines',
            line=dict(color='rgba(255,255,255,0)'),
            fillcolor='rgba(255, 140, 0, 0.2)',
            name=f'90%CI (±{np.round(z_score, 2)}σ)',
            customdata=np.column_stack([lower_catalyzed, upper_catalyzed]),
            hovertemplate='90%CI: [%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>',
            showlegend=False
        ))
    
    # Add transition line
    if catalyst_start_day > 0:
        fig.add_vline(
            x=catalyst_start_day,
            line_dash="dash",
            line_color="darkorange",
            annotation_text="Catalyst added",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=f"Ensemble Predictions ({run_name}) - Extended to {int(max_time)}",
        xaxis_title=time_label,
        yaxis_title="Cu Recovery (%)",
        yaxis_range=[0, 80],
        xaxis_range=[0, max_ext],
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
            )
    )
    
    # Create parameters output
    params_text = html.Div([
        html.H5(f"Recovery Summary at {time_label} = {int(max_time)}:"),
        html.Div([
            html.Div([
                html.Span(f"Control Recovery:", style={'display':'inline-block','minWidth':'220px'}),
                html.Span(f"{mean_pred_control[-1]:.1f}%", style={'display':'inline-block','minWidth':'80px','textAlign':'right'}),
            ]),
            html.Div([
                html.Span(f"Catalyzed Recovery:", style={'display':'inline-block','minWidth':'220px'}),
                html.Span(f"{mean_pred_catalyzed[-1]:.1f}%", style={'display':'inline-block','minWidth':'80px','textAlign':'right'}),
            ]),
            html.Div([
                html.Span(f"Catalyst Benefit:", style={'display':'inline-block','minWidth':'220px'}),
                html.Span(f"{(mean_pred_catalyzed[-1]-mean_pred_control[-1]):.1f}%", style={'display':'inline-block','minWidth':'80px','textAlign':'right','fontWeight':'bold'}),
            ]),
        ], style={'fontFamily':'Monaco'}),
        
        html.H5("Scenario Parameters:", className="mt-4"),
        dbc.Table([
            html.Thead(
                html.Tr([
                    html.Th("Scenario", style={'text-align': 'center', 'fontFamily': 'Helvetica'}),
                    html.Th("Control", colSpan="4", style={'text-align': 'center', 'background-color': 'rgba(65, 105, 225, 0.1)', 'fontFamily': 'Helvetica'}),
                    html.Th("Catalyzed", colSpan="4", style={'text-align': 'center', 'background-color': 'rgba(255, 140, 0, 0.1)', 'fontFamily': 'Helvetica'}),
                ])
            ),
            html.Thead(
                html.Tr([
                    html.Th("Parameter"),
                    html.Th("a₁", style={'background-color': 'rgba(65, 105, 225, 0.05)', 'text-align': 'center', 'fontFamily': 'Helvetica'}),
                    html.Th("b₁", style={'background-color': 'rgba(65, 105, 225, 0.05)', 'text-align': 'center', 'fontFamily': 'Helvetica'}),
                    html.Th("a₂", style={'background-color': 'rgba(65, 105, 225, 0.05)', 'text-align': 'center', 'fontFamily': 'Helvetica'}),
                    html.Th("b₂", style={'background-color': 'rgba(65, 105, 225, 0.05)', 'text-align': 'center', 'fontFamily': 'Helvetica'}),
                    html.Th("a₃", style={'background-color': 'rgba(255, 140, 0, 0.05)', 'text-align': 'center', 'fontFamily': 'Helvetica'}),
                    html.Th("b₃", style={'background-color': 'rgba(255, 140, 0, 0.05)', 'text-align': 'center', 'fontFamily': 'Helvetica'}),
                    html.Th("a₄", style={'background-color': 'rgba(255, 140, 0, 0.05)', 'text-align': 'center', 'fontFamily': 'Helvetica'}),
                    html.Th("b₄", style={'background-color': 'rgba(255, 140, 0, 0.05)', 'text-align': 'center', 'fontFamily': 'Helvetica'}),
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td("Values", style={'font-weight': 'bold'}),
                    html.Td(f"{params_control[0, 0]:.1f}", style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(f"{params_control[0, 1]:.2e}", style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(f"{params_control[0, 2]:.1f}", style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(f"{params_control[0, 3]:.2e}", style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(f"{params_catalyzed[0, 4]:.1f}", style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(f"{params_catalyzed[0, 5]:.2e}", style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(f"{params_catalyzed[0, 6]:.1f}", style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(f"{params_catalyzed[0, 7]:.2e}", style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                ])
            ])
        ], bordered=True, hover=True, striped=True, className="mt-2"),
        
        # Add recovery equations
        html.H5("Recovery Equations:", className="mt-4"),
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Strong("Control Cu Recovery (expanded logistic):", style={'color': 'royalblue'}),
                    html.Div([
                        "CuRec",
                        html.Sub("control"),
                        " = a",
                        html.Sub("1"),
                        " · ( 2 · ( 1 / (1 + e",
                        html.Sup("−b₁t"),
                        ") ) − 1 )",
                        " + a",
                        html.Sub("2"),
                        " · ( 2 · ( 1 / (1 + e",
                        html.Sup("−b₂t"),
                        ") ) − 1 )"
                    ], style={'fontSize': '1.1em', 'marginTop': '6px', 'marginBottom': '16px', 'fontStyle': 'italic', 'fontFamily': 'Georgia'}),
                    
                    html.Strong("Catalyzed Cu Recovery (logistic add-on):", style={'color': 'darkorange'}),
                    html.Div([
                        "CuRec",
                        html.Sub("catalyzed"),
                        " = CuRec",
                        html.Sub("control"),
                        " + catalyst",
                        html.Sub("effect"),
                        " · [ a",
                        html.Sub("3"),
                        " · ( 2 · ( 1 / (1 + e",
                        html.Sup("−b₃·(t−t_trans)"),
                        ") ) − 1 ) + a",
                        html.Sub("4"),
                        " · ( 2 · ( 1 / (1 + e",
                        html.Sup("−b₄·(t−t_trans)"),
                        ") ) − 1 ) ]"
                    ], style={'fontSize': '1.1em', 'marginTop': '6px', 'marginBottom': '16px', 'fontStyle': 'italic', 'fontFamily': 'Georgia'}),

                    html.Strong("Catalyst Effect:", style={'color': 'darkorange', 'font-size': '0.95em'}),
                    html.Div([
                        "catalyst",
                        html.Sub("effect"),
                        " = ",
                        html.Div([
                            html.Div("cumulative catalyst [kg/t]", 
                                    style={'borderBottom': '1px solid black', 'paddingBottom': '2px', 'textAlign': 'center'}),
                            html.Div("cumulative catalyst [kg/t] + 1", 
                                    style={'paddingTop': '2px', 'textAlign': 'center'})
                        ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'margin': '0 5px'})
                    ], style={'fontSize': '0.95em', 'marginTop': '6px', 'fontStyle': 'italic', 'fontFamily': 'Georgia'}),

                    html.Div("Each logistic term is 2·(1 / (1 + exp(−rate·t))) − 1, rising smoothly from 0 to amplitude.", style={'fontSize': '0.9em', 'marginTop': '10px', 'fontFamily': 'Georgia', 'color': '#444'})
                ]),
            ]),
        ]),
    ])
    
    outputs = [fig, params_text]
    # Append slider min/max/value and input value in the same order as outputs above
    slider_order = [
        ('acid_soluble', 0), ('residual_cpy', 1), ('material_size_p80', 2),
        ('copper_sulfides', 3), ('secondary_copper', 4), ('acid_gen_sulfides', 5),
        ('gangue_silicates', 6), ('fe_oxides', 7), ('carbonates', 8)
    ]
    for idx, (sid, _) in enumerate(slider_order):
        fmin = feature_mins[idx] if idx < len(feature_mins) else 0.0
        fmax = feature_maxs[idx] if idx < len(feature_maxs) else fmin
        fval = feature_values[idx] if idx < len(feature_values) else fmin
        outputs.extend([fmin, fmax, fval, fval])

    return tuple(outputs)


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Starting Dash server...")
    print("Open your browser and go to: http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop the server")
    print("="*80 + "\n")
    app.run(debug=True, use_reloader=False)
