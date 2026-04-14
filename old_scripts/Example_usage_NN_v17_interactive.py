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
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Device configuration
device = torch.device('cpu')

# ===========================================================================
# Load actual data
df_actual_data = pd.read_csv('/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Rosetta/NN_PyTorch/plots/processed_data_unscaled.csv')

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
    def __init__(self, total_features, hidden_dim=128, dropout_rate=0.33, init_mode='kaiming', feature_weight_signs=None):
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
                    a1 = 10.0 + 30.0 * torch.sigmoid(a1_raw)
                    b1 = 0.001 + 0.1 * torch.sigmoid(b1_raw)
                    a2 = 5.0 + 20.0 * torch.sigmoid(a2_raw)
                    b2 = 0.0001 + 0.1 * torch.sigmoid(b2_raw)
                    
                    # Ensure total asymptote doesn't exceed 70
                    total_asymptote = a1 + a2
                    mask_a = total_asymptote > 70.0
                    scale = torch.where(mask_a & (total_asymptote > 0), 
                                       70.0 / total_asymptote.clamp(min=1.0), 
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
            
            a1 = 10.0 + 30.0 * torch.sigmoid(a1_raw.squeeze())
            b1 = 0.001 + 0.1 * torch.sigmoid(b1_raw.squeeze())
            a2 = 5.0 + 20.0 * torch.sigmoid(a2_raw.squeeze())
            b2 = 0.0001 + 0.1 * torch.sigmoid(b2_raw.squeeze())
            
            # Ensure total asymptote doesn't exceed 70
            total_asymptote = a1 + a2
            mask_a = total_asymptote > 70.0
            scale = torch.where(mask_a & (total_asymptote > 0), 
                               70.0 / total_asymptote.clamp(min=1.0), 
                               torch.tensor(1.0, device=x.device))
            a1, a2 = a1 * scale, a2 * scale
            
            params[:, :4] = torch.stack([a1, b1, a2, b2], dim=0).T
        
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
                a3 = 1.0 + 14.0 * torch.sigmoid(a3_raw.squeeze())
                b3 = 0.0001 + 0.003 * torch.sigmoid(b3_raw.squeeze())
                a4 = 1.0 + 14.0 * torch.sigmoid(a4_raw.squeeze())
                b4 = 0.0001 + 0.003 * torch.sigmoid(b4_raw.squeeze())

                # Ensure total asymptote doesn't exceed 70
                total_asymptote_cat = a1_r + a2_r + a3 + a4
                mask_a = total_asymptote_cat > 70.0
                scale = torch.where(mask_a & (total_asymptote_cat > 0), 
                                70.0 / total_asymptote_cat.clamp(min=1.0), 
                                torch.tensor(1.0, device=x.device))
                a3, a4 = a3 * scale, a4 * scale
                
                params[idx, 4:] = torch.stack([a3, b3, a4, b4], dim=0).T
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
    # catalyst_effect = torch.sqrt(catalyst_effect)
    # catalyst_effect = catalyst ** 2
    
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
            # exp_term3 = torch.exp(-torch.abs(b3) * time_shifted * (1 + catalyst_effect))
            exp_term3 = torch.exp(-torch.abs(b3) * time_shifted)
            exp_term3 = torch.clamp(exp_term3, min=1e-8, max=1.0)
            # exp_term4 = torch.exp(-torch.abs(b4) * time_shifted * (1 + catalyst_effect))
            exp_term4 = torch.exp(-torch.abs(b4) * time_shifted)
            exp_term4 = torch.clamp(exp_term4, min=1e-8, max=1.0)
            
            # Additional recovery from catalyst with catalyst_effect scaling
            additional_recovery = (torch.abs(a3) * catalyst_effect * (1 - exp_term3) + torch.abs(a4) * catalyst_effect * (1 - exp_term4))
            
            # Apply catalyst enhancement only where catalyst is present and after transition
            catalyst_enhancement = torch.where(has_catalyst_points, additional_recovery, torch.zeros_like(additional_recovery))
            recovery = recovery_control + catalyst_enhancement
    
    # Apply reasonable bounds to recovery
    recovery = torch.clamp(recovery, min=0.0, max=100.0)
    return recovery


# ============================================================================
# Load the model
# ============================================================================

print("Loading model...")
folder_path = '/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Rosetta/NN_PyTorch/'
model_path = folder_path + 'AdaptiveTwoPhaseModel_withoutReactors.pt'

checkpoint = torch.load(model_path, map_location=device, weights_only=False)

ensemble_model = checkpoint['models']
scaler_X = checkpoint['scaler_X']
num_cols = checkpoint['num_cols']
results = checkpoint['results']
# Apply saved calibration if present; fallback to 1.0
if isinstance(checkpoint, dict) and 'uncertainty_scale' in checkpoint:
    ensemble_model.uncertainty_scale = float(checkpoint['uncertainty_scale'])
    print(f"✓ Loaded uncertainty_scale from checkpoint: {ensemble_model.uncertainty_scale:.3f}")
else:
    ensemble_model.uncertainty_scale = 1.0
    print("⚠️ No uncertainty_scale in checkpoint; using 1.0")

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
    'catalyst_start_day': {'label': 'Catalyst Start Day', 'min': 0, 'max': 500, 'step': 10, 'value': 250},
    'catalyst_dose': {'label': 'Catalyst Dose (kg/t/day)', 'min': 0.0, 'max': 0.01, 'step': 0.0001, 'value': 0.0009},
    'max_time': {'label': 'Max Time (days)', 'min': 100, 'max': 2500, 'step': 100, 'value': 1000},
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

# Main prediction callback
@app.callback(
    [Output('prediction-plot', 'figure'),
     Output('parameters-output', 'children')],
    [Input(f'slider-{key}', 'value') for key in slider_configs.keys()]
)
def update_prediction(acid_soluble, residual_cpy, material_size_p80, copper_sulfides,
                     secondary_copper, acid_gen_sulfides, gangue_silicates, fe_oxides,
                     carbonates, catalyst_start_day, catalyst_dose, max_time, cyanide_soluble):
    
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
    
    # Nominal 90% band; allow optional calibration carried on the ensemble (if present)
    z_nom = 1.645
    scale = checkpoint['uncertainty_scale']
    z_score = z_nom * scale
    
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
        yaxis_range=[0, 80],
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