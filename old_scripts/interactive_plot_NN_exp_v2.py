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
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
'''
# Device configuration
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

CONFIG = {}

# ===========================================================================
# Load actual data (use leach_days run to stay aligned with the model)
df_actual_data = pd.read_csv('/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Rosetta/NN_PyTorch/plots/leach_days/processed_data_unscaled.csv')

# ============================================================================
# STEP 0: Define the model classes (REQUIRED for unpickling)
# ============================================================================

def get_feature_weight_signs(config, feature_names):
    """
    Extract parameter-specific weight signs from CONFIG.
    Returns a matrix of shape [num_features, 8] containing {-1, 0, 1}.
    """
    col_config = config.get('column_tests_feature_weighting', {})
    if not col_config.get('enabled', False) or not col_config.get('use_monotonic_constraints', False):
        return torch.zeros(len(feature_names), 8)

    raw_weights = col_config.get('weights', {})
    special_dynamic = config.get('special_feats', {}).get('dynamic', [])
    weight_signs_matrix = []

    for feat_name in feature_names:
        if feat_name in special_dynamic:
            weight_signs_matrix.append([0.0] * 8)
        elif feat_name in raw_weights:
            weight_list = raw_weights[feat_name]
            if len(weight_list) >= 9:
                param_weights = weight_list[1:9]
            else:
                single_weight = weight_list[1] if len(weight_list) > 1 else 0.0
                param_weights = [single_weight] * 8

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
            weight_signs_matrix.append([0.0] * 8)

    return torch.tensor(weight_signs_matrix, dtype=torch.float32)


def get_feature_null_mask(config, feature_names):
    """
    Return a boolean matrix [num_features, 8] marking forced-null impacts.
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
            null_mask_matrix.append([bool(pd.isna(w)) for w in param_weights])
        else:
            null_mask_matrix.append([False] * 8)

    return torch.tensor(null_mask_matrix, dtype=torch.bool)


def project_params_to_caps(params: torch.Tensor,
                           base_cap: float,
                           total_cap: float,
                           base_rate_cap: float,
                           total_rate_cap: float) -> torch.Tensor:
    """
    Projection enforcing amplitude and kinetic constraints.
    a1+a2 <= base_cap; a1+a2+a3+a4 <= total_cap (scale catalyst amplitudes only)
    b1+b2 <= base_rate_cap; b1+b2+b3+b4 <= total_rate_cap (scale catalyst rates only)
    """
    if params.ndim == 1:
        params = params.unsqueeze(0)

    extra_cols = params[:, 8:] if params.shape[1] > 8 else None
    a1, b1, a2, b2, a3, b3, a4, b4 = [params[:, i] for i in range(8)]

    base_sum = a1 + a2
    scale_base_amp = torch.where(
        base_sum > base_cap,
        base_cap / base_sum.clamp(min=1e-6),
        torch.ones_like(base_sum)
    )
    a1 = a1 * scale_base_amp
    a2 = a2 * scale_base_amp

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

    rate_base_sum = b1 + b2
    scale_base_rate = torch.where(
        rate_base_sum > base_rate_cap,
        base_rate_cap / rate_base_sum.clamp(min=1e-9),
        torch.ones_like(rate_base_sum)
    )
    b1 = b1 * scale_base_rate
    b2 = b2 * scale_base_rate

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
        self.models, self.weights = self._create_filtered_ensemble(
            model_states, val_losses, config
        )

    def _create_filtered_ensemble(self, model_states, val_losses, config):
        """Create filtered ensemble based on validation losses"""
        median_loss = np.median(val_losses)
        threshold = np.percentile(val_losses, 95)

        models = []
        weights = []
        hidden_dim = config.get('pytorch_hidden_dim', 128)
        dropout_rate = config.get('pytorch_dropout_rate', 0.30)

        null_mask = self.feature_null_mask
        for idx, (model_state, val_loss) in enumerate(zip(model_states, val_losses)):
            if val_loss <= threshold:
                model = AdaptiveTwoPhaseRecoveryModel(
                    total_features=self.total_features,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    init_mode=config.get('init_mode', 'kaiming'),
                    feature_weight_signs=self.feature_weight_signs,
                    feature_null_mask=null_mask,
                ).to(self.device)
                model.load_state_dict(model_state)
                model.eval()
                weights.append(1.0 / (val_loss + 1e-6))
                models.append(model)

        weights = np.array(weights)
        if len(weights) > 0:
            weights /= weights.sum()
        else:
            weights = np.array([1.0])

        return models, weights

    def predict_with_params_and_uncertainty(self, X, catalyst, transition_time, time_points, sample_ids=None):
        """
        Make ensemble prediction with uncertainty quantification
        """
        all_model_predictions = []
        all_model_params = []

        base_cap = float(self.config.get('base_asymptote_cap', 80.0))
        total_cap = float(self.config.get('total_asymptote_cap', 95.0))
        base_rate_cap = float(self.config.get('base_rate_cap', 2.1))
        total_rate_cap = float(self.config.get('total_rate_cap', 7.0))

        for model in self.models:
            with torch.no_grad():
                params = model(X, catalyst, sample_ids)
                params = project_params_to_caps(params, base_cap, total_cap, base_rate_cap, total_rate_cap)
                all_model_params.append(params.cpu().numpy())

                recovery = generate_two_phase_recovery(
                    time_points, catalyst, transition_time, params
                )
                all_model_predictions.append(recovery.cpu().numpy())

        all_model_predictions = np.array(all_model_predictions)  # (M,B,T)
        all_model_params = np.array(all_model_params)            # (M,B,P)

        w = np.asarray(self.weights, dtype=float)
        w = w / (w.sum() if np.isfinite(w.sum()) and w.sum() > 0 else 1.0)
        w_broadcast = w[:, None, None]  # (M,1,1)

        weighted_pred = (all_model_predictions * w_broadcast).sum(axis=0)  # (B,T)
        weighted_params = (all_model_params * w[:, None, None]).sum(axis=0)  # (B,P)

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


class AdaptiveTwoPhaseRecoveryModel(nn.Module):
    """
    Enhanced Two-Phase Recovery Model with separate output heads for exact per-parameter control.
    Heads: a1, b1, a2, b2, a3, b3, a4, b4, gain_b3, gain_b4
    """
    def __init__(self, total_features, hidden_dim=128, dropout_rate=0.30, init_mode='kaiming', feature_weight_signs=None, feature_weight_magnitudes=None, feature_null_mask=None):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.init_mode = init_mode
        self.total_features = total_features

        if feature_weight_signs is not None:
            self.register_buffer('feature_weight_signs', feature_weight_signs)
            self.use_monotonic_constraints = True
        else:
            self.register_buffer('feature_weight_signs', torch.zeros(total_features, 8))
            self.use_monotonic_constraints = False

        if feature_null_mask is not None:
            self.register_buffer('feature_null_mask', feature_null_mask.to(torch.bool))
            self.use_forced_null = bool(self.feature_null_mask.any().item())
        else:
            self.register_buffer('feature_null_mask', torch.zeros(total_features, 8, dtype=torch.bool))
            self.use_forced_null = False

        self.param_networks = nn.ModuleList()
        for param_idx in range(10):
            network = nn.Sequential(
                nn.Linear(total_features, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, 1),
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
        """Apply hard monotonic constraints on first-layer weights."""
        if not self.use_monotonic_constraints:
            return

        with torch.no_grad():
            for param_idx in range(8):
                first_layer = self.param_networks[param_idx][0]
                first_weight = first_layer.weight
                for feat_idx in range(self.feature_weight_signs.shape[0]):
                    sign_constraint = self.feature_weight_signs[feat_idx, param_idx].item()
                    if sign_constraint > 0:
                        first_weight[:, feat_idx].data = torch.abs(first_weight[:, feat_idx].data)
                    elif sign_constraint < 0:
                        first_weight[:, feat_idx].data = -torch.abs(first_weight[:, feat_idx].data)

    def apply_forced_nulls(self):
        """Hard-zero first-layer weights for forced-null feature/parameter pairs."""
        if not self.use_forced_null:
            return

        with torch.no_grad():
            for param_idx in range(8):
                first_layer = self.param_networks[param_idx][0]
                null_mask = self.feature_null_mask[:, param_idx]
                if null_mask.any():
                    first_layer.weight[:, null_mask] = 0.0

    def forward(self, x, catalyst, sample_ids=None):
        if self.training and self.use_monotonic_constraints:
            self.apply_monotonic_constraints()
        if self.use_forced_null:
            self.apply_forced_nulls()

        batch_size = x.size(0)
        params = torch.zeros(batch_size, 10, device=x.device)

        lim = CONFIG.get('param_limits', {})
        # Base parameters
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

        if sample_ids is not None:
            unique_sample_ids = list(set(sample_ids))
            sample_id_to_base_params = {}
            for unique_id in unique_sample_ids:
                sample_indices = [i for i, sid in enumerate(sample_ids) if sid == unique_id]
                if sample_indices:
                    first_idx = sample_indices[0]
                    a1_raw = self.param_networks[0](x[first_idx:first_idx+1])
                    b1_raw = self.param_networks[1](x[first_idx:first_idx+1])
                    a2_raw = self.param_networks[2](x[first_idx:first_idx+1])
                    b2_raw = self.param_networks[3](x[first_idx:first_idx+1])

                    a1 = a1_min + (a1_max - a1_min) * torch.sigmoid(a1_raw)
                    b1 = b1_min + (b1_max - b1_min) * torch.sigmoid(b1_raw)
                    a2 = a2_min + (a2_max - a2_min) * torch.sigmoid(a2_raw)
                    b2 = b2_min + (b2_max - b2_min) * torch.sigmoid(b2_raw)

                    base_cap = float(CONFIG.get('base_asymptote_cap', 80.0))
                    total_asymptote = a1 + a2
                    mask_a = total_asymptote > base_cap
                    scale = torch.where(
                        mask_a & (total_asymptote > 0),
                        base_cap / total_asymptote.clamp(min=1.0),
                        torch.tensor(1.0, device=x.device)
                    )
                    a1, a2 = a1 * scale, a2 * scale

                    sample_id_to_base_params[unique_id] = torch.cat(
                        [a1.flatten(), b1.flatten(), a2.flatten(), b2.flatten()]
                    )
                    for idx in sample_indices:
                        params[idx, :4] = sample_id_to_base_params[unique_id]
        else:
            a1_raw = self.param_networks[0](x)
            b1_raw = self.param_networks[1](x)
            a2_raw = self.param_networks[2](x)
            b2_raw = self.param_networks[3](x)

            a1 = a1_min + (a1_max - a1_min) * torch.sigmoid(a1_raw.squeeze())
            b1 = b1_min + (b1_max - b1_min) * torch.sigmoid(b1_raw.squeeze())
            a2 = a2_min + (a2_max - a2_min) * torch.sigmoid(a2_raw.squeeze())
            b2 = b2_min + (b2_max - b2_min) * torch.sigmoid(b2_raw.squeeze())

            base_cap = float(CONFIG.get('base_asymptote_cap', 80.0))
            total_asymptote = a1 + a2
            mask_a = total_asymptote > base_cap
            scale = torch.where(
                mask_a & (total_asymptote > 0),
                base_cap / total_asymptote.clamp(min=1.0),
                torch.tensor(1.0, device=x.device)
            )
            a1, a2 = a1 * scale, a2 * scale

        params[:, :4] = torch.stack(
            [a1.reshape(-1), b1.reshape(-1), a2.reshape(-1), b2.reshape(-1)],
            dim=1,
        )

        has_catalyst = torch.any(catalyst > 0, dim=1)
        if has_catalyst.any():
            idx = has_catalyst.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                a1_r, b1_r, a2_r, b2_r = [p.squeeze() for p in params[idx, :4].split(1, dim=1)]
                a3_raw = self.param_networks[4](x[idx])
                b3_raw = self.param_networks[5](x[idx])
                a4_raw = self.param_networks[6](x[idx])
                b4_raw = self.param_networks[7](x[idx])

                a3 = a3_min + (a3_max - a3_min) * torch.sigmoid(a3_raw.squeeze())
                b3 = b3_min + (b3_max - b3_min) * torch.sigmoid(b3_raw.squeeze())
                a4 = a4_min + (a4_max - a4_min) * torch.sigmoid(a4_raw.squeeze())
                b4 = b4_min + (b4_max - b4_min) * torch.sigmoid(b4_raw.squeeze())
                gain_b3 = gain_b3_min + (gain_b3_max - gain_b3_min) * torch.sigmoid(self.param_networks[8](x[idx]).squeeze())
                gain_b4 = gain_b4_min + (gain_b4_max - gain_b4_min) * torch.sigmoid(self.param_networks[9](x[idx]).squeeze())

                total_cap = float(CONFIG.get('total_asymptote_cap', 95.0))
                total_asymptote_cat = a1_r + a2_r + a3 + a4
                mask_a = total_asymptote_cat > total_cap
                scale = torch.where(
                    mask_a & (total_asymptote_cat > 0),
                    total_cap / total_asymptote_cat.clamp(min=1.0),
                    torch.tensor(1.0, device=x.device)
                )
                a3, a4 = a3 * scale, a4 * scale

                params[idx, 4:] = torch.stack(
                    [a3.reshape(-1), b3.reshape(-1), a4.reshape(-1), b4.reshape(-1), gain_b3.reshape(-1), gain_b4.reshape(-1)],
                    dim=1,
                )
        else:
            params[:, 4:] = torch.full((batch_size, 6), float('nan'), device=x.device)

        base_cap = float(CONFIG.get('base_asymptote_cap', 80.0))
        total_cap = float(CONFIG.get('total_asymptote_cap', 95.0))
        base_rate_cap = float(CONFIG.get('base_rate_cap', 2.1))
        total_rate_cap = float(CONFIG.get('total_rate_cap', 7.0))
        params = project_params_to_caps(params, base_cap, total_cap, base_rate_cap, total_rate_cap)
        return params


def generate_two_phase_recovery_exp(time, catalyst, transition_time, params):
    """
    Two-phase recovery generation with catalyst dose-rate modulation.
    """
    time = time.to(params.device)
    catalyst = catalyst.to(params.device)

    catalyst_effect = catalyst / (catalyst + 1)
    ce_pow = float(CONFIG.get('cat_effect_power', 1.0))
    if ce_pow != 1.0:
        catalyst_effect = catalyst_effect.pow(ce_pow)

    a1 = params[:, 0].unsqueeze(1)
    b1 = params[:, 1].unsqueeze(1)
    a2 = params[:, 2].unsqueeze(1)
    b2 = params[:, 3].unsqueeze(1)

    exp_term1 = torch.exp(-b1 * time).clamp(min=1e-8, max=1.0)
    exp_term2 = torch.exp(-b2 * time).clamp(min=1e-8, max=1.0)
    recovery_control = a1 * (1 - exp_term1) + a2 * (1 - exp_term2)

    recovery = recovery_control.clone()
    has_catalyst = torch.any(catalyst > 0).item()
    transition_i = transition_time.squeeze()

    if has_catalyst and params.shape[1] > 4 and torch.any(~torch.isnan(params[:, 4:])):
        a3 = params[:, 4].unsqueeze(1)
        b3 = params[:, 5].unsqueeze(1)
        a4 = params[:, 6].unsqueeze(1)
        b4 = params[:, 7].unsqueeze(1)

        has_catalyst_points = (catalyst > 0) & (time >= transition_i)
        if has_catalyst_points.any():
            time_shifted = torch.clamp(time - transition_i, min=0.0)

            if params.shape[1] >= 10:
                gain_b3 = params[:, 8].unsqueeze(1)
                gain_b4 = params[:, 9].unsqueeze(1)
            else:
                gain_b3 = params.new_full((params.size(0), 1), float(CONFIG.get('cat_rate_gain_b3', 0.0)))
                gain_b4 = params.new_full((params.size(0), 1), float(CONFIG.get('cat_rate_gain_b4', 0.0)))
            gain_b3 = torch.clamp_min(gain_b3, 0.0)
            gain_b4 = torch.clamp_min(gain_b4, 0.0)

            rate_mult3 = 1.0 + gain_b3 * catalyst_effect
            rate_mult4 = 1.0 + gain_b4 * catalyst_effect

            exp_term3 = torch.exp(-torch.abs(b3) * time_shifted * rate_mult3).clamp(min=1e-8, max=1.0)
            exp_term4 = torch.exp(-torch.abs(b4) * time_shifted * rate_mult4).clamp(min=1e-8, max=1.0)

            additional_recovery = catalyst_effect * (torch.abs(a3) * (1 - exp_term3) + torch.abs(a4) * (1 - exp_term4))
            catalyst_enhancement = torch.where(has_catalyst_points, additional_recovery, torch.zeros_like(additional_recovery))
            recovery = recovery_control + catalyst_enhancement

    total_cap = float(CONFIG.get('total_asymptote_cap', 95.0))
    recovery = torch.clamp(recovery, min=0.0, max=total_cap)
    return recovery


def generate_two_phase_recovery(time, catalyst, transition_time, params):
    return generate_two_phase_recovery_exp(time, catalyst, transition_time, params)


# ============================================================================
# Load the model
# ============================================================================

print("Loading model...")
folder_path = '/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Rosetta/NN_PyTorch/'
model_path = os.path.join(folder_path, 'AdaptiveTwoPhaseModel_withoutReactors_leachdays.pt')

checkpoint = torch.load(model_path, map_location=device, weights_only=False)

CONFIG = checkpoint.get('config', {}) or {}
globals()['CONFIG'] = CONFIG

ensemble_model = checkpoint['models']
scaler_X = checkpoint['scaler_X']
num_cols = checkpoint['num_cols']
results = checkpoint.get('results')
uncertainty_scale = float(checkpoint.get('uncertainty_scale', 1.0))
ensemble_model.uncertainty_scale = uncertainty_scale
feature_metadata = {m['name']: m for m in checkpoint.get('feature_metadata', [])} if checkpoint.get('feature_metadata') else {}
print(f"✓ Loaded uncertainty_scale: {ensemble_model.uncertainty_scale:.3f}")

# Ensure forced-null mask is available on loaded ensemble/models (backward compatible)
feature_null_mask_loaded = get_feature_null_mask(CONFIG, num_cols).to(device)
if not hasattr(ensemble_model, 'feature_null_mask'):
    ensemble_model.feature_null_mask = feature_null_mask_loaded
else:
    ensemble_model.feature_null_mask = feature_null_mask_loaded

for m in getattr(ensemble_model, 'models', []):
    if not hasattr(m, 'feature_null_mask'):
        # register as buffer to stay consistent with training code
        m.register_buffer('feature_null_mask', feature_null_mask_loaded.to(next(m.parameters()).device))
    else:
        m.feature_null_mask = feature_null_mask_loaded.to(m.feature_null_mask.device)
    m.use_forced_null = bool(m.feature_null_mask.any().item())

print(f"✓ Model loaded successfully!")
print(f"✓ Number of models in ensemble: {ensemble_model.get_ensemble_info()['num_models']}")
print(f"✓ Number of features: {len(num_cols)}")

# ============================================================================
# Create Dash App
# ============================================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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

def _feature_range(name, fallback_min, fallback_max):
    """Use checkpoint metadata and actual data to set sensible slider bounds."""
    meta = feature_metadata.get(name, {}) if feature_metadata else {}
    fmin = meta.get('min', fallback_min)
    fmax = meta.get('max', fallback_max)
    if name in df_actual_data.columns:
        col_min = float(df_actual_data[name].min())
        col_max = float(df_actual_data[name].max())
        if np.isfinite(col_min):
            fmin = min(fmin, col_min)
        if np.isfinite(col_max):
            fmax = max(fmax, col_max)
    return float(fmin), float(fmax)


def _clamp(val, bounds):
    return max(bounds[0], min(bounds[1], val))


feature_defaults = {
    'acid_soluble_%': 3.5,
    'residual_cpy_%': 88.5,
    'material_size_p80_in': 1.0,
    'grouped_copper_sulfides': 0.55,
    'grouped_secondary_copper': 0.01,
    'grouped_acid_generating_sulfides': 3.0,
    'grouped_gangue_silicates': 95.8,
    'grouped_fe_oxides': 0.5,
    'grouped_carbonates': 0.0,
}

# Build slider configurations using dataset/metadata bounds
slider_configs = {}
slider_configs['acid_soluble'] = {
    'label': 'Acid Soluble %',
    'min': round((_rng := _feature_range('acid_soluble_%', 1.0, 26.0))[0], 1),
    'max': round(_rng[1], 1),
    'step': 0.5,
    'value': _clamp(feature_defaults['acid_soluble_%'], _rng)
}

slider_configs['residual_cpy'] = {
    'label': 'Residual Cpy %',
    'min': round((_rng := _feature_range('residual_cpy_%', 23.0, 96.0))[0], 1),
    'max': round(_rng[1], 1),
    'step': 0.5,
    'value': _clamp(feature_defaults['residual_cpy_%'], _rng)
}

slider_configs['material_size_p80'] = {
    'label': 'Material Size P80 (in)',
    'min': round((_rng := _feature_range('material_size_p80_in', 0.5, 8.0))[0], 1),
    'max': round(_rng[1], 1),
    'step': 0.1,
    'value': _clamp(feature_defaults['material_size_p80_in'], _rng)
}

slider_configs['copper_sulfides'] = {
    'label': 'Copper Sulfides',
    'min': round((_rng := _feature_range('grouped_copper_sulfides', 0.2, 1.3))[0], 2),
    'max': round(_rng[1], 2),
    'step': 0.05,
    'value': _clamp(feature_defaults['grouped_copper_sulfides'], _rng)
}

slider_configs['secondary_copper'] = {
    'label': 'Secondary Copper',
    'min': round((_rng := _feature_range('grouped_secondary_copper', 0.0, 0.06))[0], 3),
    'max': round(_rng[1], 3),
    'step': 0.005,
    'value': _clamp(feature_defaults['grouped_secondary_copper'], _rng)
}

slider_configs['acid_gen_sulfides'] = {
    'label': 'Acid Gen. Sulfides',
    'min': round((_rng := _feature_range('grouped_acid_generating_sulfides', 0.0, 12.0))[0], 1),
    'max': round(_rng[1], 1),
    'step': 0.5,
    'value': _clamp(feature_defaults['grouped_acid_generating_sulfides'], _rng)
}

slider_configs['gangue_silicates'] = {
    'label': 'Gangue Silicates',
    'min': round((_rng := _feature_range('grouped_gangue_silicates', 87.0, 98.0))[0], 1),
    'max': round(_rng[1], 1),
    'step': 0.1,
    'value': _clamp(feature_defaults['grouped_gangue_silicates'], _rng)
}

slider_configs['fe_oxides'] = {
    'label': 'Fe Oxides',
    'min': round((_rng := _feature_range('grouped_fe_oxides', 0.0, 3.2))[0], 1),
    'max': round(_rng[1], 1),
    'step': 0.1,
    'value': _clamp(feature_defaults['grouped_fe_oxides'], _rng)
}

slider_configs['carbonates'] = {
    'label': 'Carbonates',
    'min': round((_rng := _feature_range('grouped_carbonates', 0.0, 2.1))[0], 1),
    'max': round(_rng[1], 1),
    'step': 0.1,
    'value': _clamp(feature_defaults['grouped_carbonates'], _rng)
}

max_transition = float(df_actual_data['transition_time'].max()) if 'transition_time' in df_actual_data.columns else 1600.0
time_upper = max(1600.0, 2500.0, max_transition + 200.0)
slider_configs['catalyst_start_day'] = {'label': 'Catalyst Start Day', 'min': 0.0, 'max': time_upper, 'step': 10.0, 'value': min(250.0, time_upper)}
slider_configs['catalyst_dose'] = {'label': 'Catalyst Dose (kg/t/day)', 'min': 0.0, 'max': 0.01, 'step': 0.0001, 'value': 0.0009}
slider_configs['max_time'] = {'label': 'Max Time (days)', 'min': 100.0, 'max': time_upper, 'step': 100.0, 'value': min(1000.0, time_upper)}

# Compute initial values for dependent sliders
cyanide_bounds = (3.0, 55.0)
cyanide_guess = 100 - slider_configs['acid_soluble']['value'] - slider_configs['residual_cpy']['value']
slider_configs['cyanide_soluble'] = {'label': 'Cyanide Soluble %', 'min': cyanide_bounds[0], 'max': cyanide_bounds[1], 'step': 0.5, 'value': _clamp(cyanide_guess, cyanide_bounds)}

sum_others = (slider_configs['copper_sulfides']['value'] + slider_configs['secondary_copper']['value'] +
              slider_configs['acid_gen_sulfides']['value'] + slider_configs['fe_oxides']['value'] +
              slider_configs['carbonates']['value'])
gangue_guess = 100 - sum_others
gangue_bounds = (slider_configs['gangue_silicates']['min'], slider_configs['gangue_silicates']['max'])
slider_configs['gangue_silicates']['value'] = _clamp(gangue_guess, gangue_bounds)

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
                    create_slider_row('catalyst_start_day', slider_configs['catalyst_start_day']),
                    create_slider_row('catalyst_dose', slider_configs['catalyst_dose']),
                ])
            ], className='mb-3'),
            
            dbc.Card([
                dbc.CardHeader(html.H5("Simulation Parameters")),
                dbc.CardBody([
                    create_slider_row('max_time', slider_configs['max_time']),
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
        [Output(f'slider-{slider_id}', 'value'),
         Output(f'input-{slider_id}', 'value')],
        [Input(f'slider-{slider_id}', 'value'),
         Input(f'input-{slider_id}', 'value')],
        prevent_initial_call=True
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
    [Output('slider-cyanide_soluble', 'value'),
     Output('input-cyanide_soluble', 'value')],
    [Input('slider-acid_soluble', 'value'),
     Input('slider-residual_cpy', 'value')]
)
def update_cyanide(acid, residual):
    bounds = (slider_configs['cyanide_soluble']['min'], slider_configs['cyanide_soluble']['max'])
    cyanide = _clamp(100 - acid - residual, bounds)
    return cyanide, cyanide

@app.callback(
    Output('input-cyanide_soluble', 'style'),
    Input('input-cyanide_soluble', 'value')
)
def update_cyanide_input_style(value):
    style = {'width': '100%'}
    if value is not None:
        lo, hi = slider_configs['cyanide_soluble']['min'], slider_configs['cyanide_soluble']['max']
        if value < lo or value > hi:
            style['color'] = 'red'
    return style

# Callback to update gangue_silicates
@app.callback(
    [Output('slider-gangue_silicates', 'value'),
     Output('input-gangue_silicates', 'value')],
    [Input('slider-copper_sulfides', 'value'),
     Input('slider-secondary_copper', 'value'),
     Input('slider-acid_gen_sulfides', 'value'),
     Input('slider-fe_oxides', 'value'),
     Input('slider-carbonates', 'value')]
)
def update_gangue(cu_sulf, sec_cu, acid_sulf, fe_ox, carb):
    sum_others = cu_sulf + sec_cu + acid_sulf + fe_ox + carb
    lo, hi = slider_configs['gangue_silicates']['min'], slider_configs['gangue_silicates']['max']
    gangue = _clamp(100 - sum_others, (lo, hi))
    return gangue, gangue

@app.callback(
    Output('input-gangue_silicates', 'style'),
    Input('input-gangue_silicates', 'value')
)
def update_gangue_input_style(value):
    style = {'width': '100%'}
    if value is not None:
        lo, hi = slider_configs['gangue_silicates']['min'], slider_configs['gangue_silicates']['max']
        if value < lo or value > hi:
            style['color'] = 'red'
    return style

# Main prediction callback
@app.callback(
    [Output('prediction-plot', 'figure'),
     Output('parameters-output', 'children')],
    [Input(f'slider-{key}', 'value') for key in slider_configs.keys()]
)
def update_prediction(acid_soluble, residual_cpy, material_size_p80, copper_sulfides,
                     secondary_copper, acid_gen_sulfides, gangue_silicates, fe_oxides,
                     carbonates, catalyst_start_day, catalyst_dose, max_time, cyanide_soluble):
    max_time = float(max_time)
    catalyst_start_day = float(min(catalyst_start_day, max_time))
    
    # Prepare sample features
    sample_features = {
        'acid_soluble_%': acid_soluble,
        'residual_cpy_%': residual_cpy,
        'material_size_p80_in': material_size_p80,
        'grouped_copper_sulfides': copper_sulfides,
        'grouped_secondary_copper': secondary_copper,
        'grouped_acid_generating_sulfides': acid_gen_sulfides,
        'grouped_gangue_silicates': gangue_silicates,
        'grouped_fe_oxides': fe_oxides,
        'grouped_carbonates': carbonates,
    }
    
    # Convert to DataFrame
    X_new_df = pd.DataFrame([[sample_features.get(col, 0.0) for col in num_cols]], columns=num_cols)
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

    def fmt_val(val, fmt_str=".2f"):
        try:
            if np.isfinite(val):
                return format(float(val), fmt_str)
        except Exception:
            pass
        return "—"
    
    # Nominal 90% band; allow optional calibration carried on the ensemble (if present)
    z_nom = 1.645
    z_score = z_nom * uncertainty_scale
    
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
        title=f"Ensemble Predictions - Extended to {int(max_time)} Days",
        xaxis_title="Leach Duration (Days)",
        yaxis_title="Cu Recovery (%)",
        yaxis_range=[0, CONFIG.get('total_asymptote_cap', 100.0)],
        xaxis_range=[0, max_time],
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            x=0.02,  # Position from left (0 = left edge, 1 = right edge)
            y=0.98,  # Position from bottom (0 = bottom, 1 = top)
            xanchor='left',  # Anchor point for x position
            yanchor='top',   # Anchor point for y position
            bgcolor='rgba(255, 255, 255, 0.8)',  # White background with 80% opacity
            bordercolor='rgba(0, 0, 0, 0.3)',    # Light border
            borderwidth=1
            )
    )

    ctrl = params_control[0] if params_control.ndim > 1 else params_control
    cat = params_catalyzed[0] if params_catalyzed.ndim > 1 else params_catalyzed
    ctrl_values = {
        'a1': fmt_val(ctrl[0], ".1f"),
        'b1': fmt_val(ctrl[1], ".2e"),
        'a2': fmt_val(ctrl[2], ".1f"),
        'b2': fmt_val(ctrl[3], ".2e"),
    }
    cat_values = {
        'a3': fmt_val(cat[4] if cat.size > 4 else np.nan, ".1f"),
        'b3': fmt_val(cat[5] if cat.size > 5 else np.nan, ".2e"),
        'a4': fmt_val(cat[6] if cat.size > 6 else np.nan, ".1f"),
        'b4': fmt_val(cat[7] if cat.size > 7 else np.nan, ".2e"),
        'gain_b3': fmt_val(cat[8] if cat.size > 8 else np.nan, ".2f"),
        'gain_b4': fmt_val(cat[9] if cat.size > 9 else np.nan, ".2f"),
    }
    
    # Create parameters output
    params_text = html.Div([
        html.H5(f"Recovery Summary at day {int(max_time)}:"),
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
                    html.Th("Catalyzed", colSpan="6", style={'text-align': 'center', 'background-color': 'rgba(255, 140, 0, 0.1)', 'fontFamily': 'Helvetica'}),
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
                    html.Th("gain b₃", style={'background-color': 'rgba(255, 140, 0, 0.05)', 'text-align': 'center', 'fontFamily': 'Helvetica'}),
                    html.Th("gain b₄", style={'background-color': 'rgba(255, 140, 0, 0.05)', 'text-align': 'center', 'fontFamily': 'Helvetica'}),
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td("Values", style={'font-weight': 'bold'}),
                    html.Td(ctrl_values['a1'], style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(ctrl_values['b1'], style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(ctrl_values['a2'], style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(ctrl_values['b2'], style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(cat_values['a3'], style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(cat_values['b3'], style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(cat_values['a4'], style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(cat_values['b4'], style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(cat_values['gain_b3'], style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                    html.Td(cat_values['gain_b4'], style={'text-align': 'center', 'fontFamily': 'Monaco', 'font-size': '0.9em'}),
                ])
            ])
        ], bordered=True, hover=True, striped=True, className="mt-2"),
        
        # Add recovery equations
        html.H5("Recovery Equations:", className="mt-4"),
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Strong("Control Cu Recovery:", style={'color': 'royalblue'}),
                    html.Div([
                        "Cu Rec",
                        html.Sub("control"),
                        " = a",
                        html.Sub("1"),
                        " · (1 - e",
                        html.Sup("-b₁t"),
                        ") + a",
                        html.Sub("2"),
                        " · (1 - e",
                        html.Sup("-b₂t"),
                        ")"
                    ], style={'fontSize': '1.1em', 'marginTop': '5px', 'marginBottom': '20px', 'fontStyle': 'italic', 'fontFamily': 'Georgia'}),
                    
                    html.Strong("Catalyzed Cu Recovery:", style={'color': 'darkorange'}),
                    html.Div([
                        "Cu Rec",
                        html.Sub("catalyzed"),
                        " = ",
                        "CuRec",
                        html.Sub("control"),
                        " + catalyst",
                        html.Sub("effect"),
                        " · (a",
                        html.Sub("3"),
                        " · (1 - e",
                        html.Sup("-b₃t"),
                        ") + a",
                        html.Sub("4"),
                        " · (1 - e",
                        html.Sup("-b₄t"),
                        "))"
                    ], style={'fontSize': '1.1em', 'marginTop': '5px', 'marginBottom': '20px', 'fontStyle': 'italic', 'fontFamily': 'Georgia'}),

                    html.Strong("where Catalyst Effect:", style={'color': 'darkorange', 'font-size': '0.9em'}),
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
                    ], style={'fontSize': '0.9em', 'marginTop': '5px', 'fontStyle': 'italic', 'fontFamily': 'Georgia'}),

                    html.Div(
                        "Rate terms are further scaled by (1 + gain_b3*catalyst_effect) and (1 + gain_b4*catalyst_effect).",
                        style={'fontSize': '0.9em', 'marginTop': '10px', 'fontFamily': 'Helvetica'}
                    ),
                ]),
            ]),
        ]),
    ])
    
    return fig, params_text


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Starting Dash server...")
    print("Open your browser and go to: http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop the server")
    print("="*80 + "\n")
    app.run(debug=True, use_reloader=False)
