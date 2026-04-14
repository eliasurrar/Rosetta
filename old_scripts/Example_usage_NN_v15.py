#%%
"""
Example script for using the AdaptiveTwoPhaseModel_withoutReactors.pt model
to make predictions with uncertainty quantification.

UPDATED VERSION: Handles PyTorch 2.6+ unpickling requirements

This script demonstrates:
1. Loading the saved model (with proper class definitions)
2. Preparing new input data
3. Making predictions with uncertainty
4. Plotting the results with uncertainty bands
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Device configuration (use CPU for compatibility)
device = torch.device('cpu')

# ============================================================================
# STEP 0: Define the model classes (REQUIRED for unpickling)
# ============================================================================
# You MUST include these class definitions before loading the model
# These should match the classes in your original training script

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
        
        # Network for catalyst parameters (a3, b3, a4, b4)
        self.catalyst_network = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, 4),
        )

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
                
                # Apply catalyst parameter constraints
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
            params[:, 4:] = float('nan')
        
        return params


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
        threshold = np.percentile(val_losses, 95)
        
        models = []
        weights = []
        
        hidden_dim = config.get('pytorch_hidden_dim', 128)
        dropout_rate = config.get('pytorch_dropout_rate', 0.3)
        
        for idx, (model_state, val_loss) in enumerate(zip(model_states, val_losses)):
            if val_loss <= threshold:
                model = AdaptiveTwoPhaseRecoveryModel(
                    total_features=self.total_features,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    init_mode=config.get('init_mode', 'kaiming')
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


def generate_two_phase_recovery(time, catalyst, transition_time, params):
    """
    Two-phase recovery generation with proper physical constraints
    """
    # Ensure inputs are on the correct device
    time = time.to(params.device)
    catalyst = catalyst.to(params.device)
    
    # Calculate catalyst effect tensor
    catalyst_effect = catalyst / (catalyst + 1)
    
    # Extract base parameters
    a1 = params[:, 0].unsqueeze(1)
    b1 = params[:, 1].unsqueeze(1)
    a2 = params[:, 2].unsqueeze(1)
    b2 = params[:, 3].unsqueeze(1)
    
    # Compute base recovery with numerical stability
    exp_term1 = torch.exp(-b1 * time)
    exp_term1 = torch.clamp(exp_term1, min=1e-8, max=1.0)
    exp_term2 = torch.exp(-b2 * time)
    exp_term2 = torch.clamp(exp_term2, min=1e-8, max=1.0)
    recovery_control = a1 * (1 - exp_term1) + a2 * (1 - exp_term2)
    
    # Start with control recovery
    recovery = recovery_control.clone()
    
    has_catalyst = torch.any(catalyst > 0).item()
    transition_i = transition_time.squeeze()
    
    # Apply catalyst enhancement if catalyst exists
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
            
            # Use catalyst_effect tensor in the exponential terms
            exp_term3 = torch.exp(-torch.abs(b3) * time_shifted * (1 + catalyst_effect))
            exp_term3 = torch.clamp(exp_term3, min=1e-8, max=1.0)
            exp_term4 = torch.exp(-torch.abs(b4) * time_shifted * (1 + catalyst_effect))
            exp_term4 = torch.clamp(exp_term4, min=1e-8, max=1.0)
            
            # Additional recovery from catalyst
            additional_recovery = (torch.abs(a3) * catalyst_effect * (1 - exp_term3) + 
                                 torch.abs(a4) * catalyst_effect * (1 - exp_term4))
            
            # Apply catalyst enhancement only where catalyst is present
            catalyst_enhancement = torch.where(has_catalyst_points, additional_recovery, torch.zeros_like(additional_recovery))
            recovery = recovery_control + catalyst_enhancement
    
    # Apply reasonable bounds to recovery
    recovery = torch.clamp(recovery, min=0.0, max=100.0)
    return recovery

#%%
# ============================================================================
# STEP 1: Load the saved model
# ============================================================================

# Path to the saved model file
folder_path = '/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Rosetta/NN_PyTorch/'
model_path = folder_path + 'AdaptiveTwoPhaseModel_withoutReactors.pt'

# Load the model with weights_only=False (since we trust the source)
# This is necessary because the model contains custom classes
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

# Extract components
ensemble_model = checkpoint['models']  # The EnsembleModels object
scaler_X = checkpoint['scaler_X']      # The StandardScaler for input features
num_cols = checkpoint['num_cols']      # List of feature names
results = checkpoint['results']        # Training results (optional)

print(f"Model loaded successfully!")
print(f"Number of models in ensemble: {ensemble_model.get_ensemble_info()['num_models']}")
print(f"Number of features: {len(num_cols)}")
print(f"Feature names: {num_cols}")

#%%
# ============================================================================
# STEP 2: Prepare new input data
# ============================================================================

# IMPORTANT: You need to provide values for all features in the same order as num_cols
# For this example, we'll use dummy data. Replace this with your actual data.

# Example: Create a sample input (replace with your actual values)
sample_features = {
    # Replace these with your actual feature values
    # The keys should match the feature names in num_cols
    'acid_soluble_%': 7.2,                      # training data from-to: [1.0, 26.0]
    'residual_cpy_%': 72.0,                     # training data from-to: [23.0, 96.0]
    'material_size_p80_in': 1.0,                # training data from-to: [0.5, 8.0]
    'grouped_copper_sulfides': 0.7,             # training data from-to: [0.20, 1.3]
    'grouped_secondary_copper': 0.01,           # training data from-to: [0.0, 0.06]
    'grouped_acid_generating_sulfides': 2.6,    # training data from-to: [0.06, 12.0]
    'grouped_gangue_silicates': 95.0,           # training data from-to: [87.0, 98.0]
    'grouped_fe_oxides': 0.7,                   # training data from-to: [0.03], 3.2]
    'grouped_carbonates': 0.5,                  # training data from-to: [0.0, 2.1]
    # ... add all other features from num_cols
}

sample_features = {
    # Replace these with your actual feature values
    # The keys should match the feature names in num_cols
    'acid_soluble_%': 3.37,                      # training data from-to: [1.0, 26.0]
    'residual_cpy_%': 88.66,                     # training data from-to: [23.0, 96.0]
    'material_size_p80_in': 1.0,                # training data from-to: [0.5, 8.0]
    'grouped_copper_sulfides': 0.52,             # training data from-to: [0.20, 1.3]
    'grouped_secondary_copper': 0.0,           # training data from-to: [0.0, 0.06]
    'grouped_acid_generating_sulfides': 1.71,    # training data from-to: [0.06, 12.0]
    'grouped_gangue_silicates': 95.83,           # training data from-to: [87.0, 98.0]
    'grouped_fe_oxides': 0.42,                   # training data from-to: [0.03], 3.2]
    'grouped_carbonates': 0.0,                  # training data from-to: [0.0, 2.1]
    # ... add all other features from num_cols
}

# Convert to numpy array in the correct order
# IMPORTANT: The order must match num_cols exactly
X_new = np.array([[sample_features.get(col, 0.0) for col in num_cols]])

# Scale the input using the same scaler used during training
X_new_scaled = scaler_X.transform(X_new)

# Convert to PyTorch tensor
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)

#%%
# ============================================================================
# STEP 3: Define the time points and catalyst addition
# ============================================================================

# Define the time points (in days) for which you want predictions
time_points = np.linspace(0, 2500, 2501)  # 0, 1, 2, ..., 2500 days every 1 week
time_tensor = torch.tensor(time_points, dtype=torch.float32).to(device).unsqueeze(0)

# Define the catalyst addition schedule
# Example 1: No catalyst (control scenario)
catalyst_values_control = np.zeros_like(time_points)

# Example 2: Catalyst added starting at day 300
catalyst_values_catalyzed = np.zeros_like(time_points)
catalyst_start_day = 85 # as in 026 primary sulfide
catalyst_dose = 0.0009  # kg/t per day # obtained from the slope of the catalyst addition curve in 026 primary sulfide
for i, t in enumerate(time_points):
    if t >= catalyst_start_day:
        catalyst_values_catalyzed[i] = (t - catalyst_start_day) * catalyst_dose

# Convert to tensor
catalyst_tensor_control = torch.tensor(catalyst_values_control, dtype=torch.float32).to(device).unsqueeze(0)
catalyst_tensor_catalyzed = torch.tensor(catalyst_values_catalyzed, dtype=torch.float32).to(device).unsqueeze(0)

# Define transition time
transition_time_control = torch.tensor([time_points.max()], dtype=torch.float32).to(device)
transition_time_catalyzed = torch.tensor([catalyst_start_day], dtype=torch.float32).to(device)

#%%
# ============================================================================
# STEP 4: Make predictions with uncertainty
# ============================================================================

# Prediction for control scenario
mean_pred_control, uncertainty_control, params_control = ensemble_model.predict_with_params_and_uncertainty(
    X_new_tensor,
    catalyst_tensor_control,
    transition_time_control,
    time_tensor,
    sample_ids=None
)

# Prediction for catalyzed scenario
mean_pred_catalyzed, uncertainty_catalyzed, params_catalyzed = ensemble_model.predict_with_params_and_uncertainty(
    X_new_tensor,
    catalyst_tensor_catalyzed,
    transition_time_catalyzed,
    time_tensor,
    sample_ids=None
)

# Convert to numpy for plotting
mean_pred_control = mean_pred_control.squeeze()
uncertainty_control = uncertainty_control.squeeze()
mean_pred_catalyzed = mean_pred_catalyzed.squeeze()
uncertainty_catalyzed = uncertainty_catalyzed.squeeze()

print(f"\nPrediction completed!")
print(f"Control scenario - Mean recovery at day {int(time_points.max())}: {mean_pred_control[-1]:.2f}%")
print(f"Control scenario - Mean uncertainty: {uncertainty_control.mean():.2f}%")
print(f"Catalyzed scenario - Mean recovery at day {int(time_points.max())}: {mean_pred_catalyzed[-1]:.2f}%")
print(f"Catalyzed scenario - Mean uncertainty: {uncertainty_catalyzed.mean():.2f}%")

#%%
# ============================================================================
# STEP 5: Plot with Original Script Style (Single Combined Plot)
# ============================================================================

# Z-score for confidence interval
z_score = 1.645  # 90% confidence interval
# ±1.645 standard deviations = approximately 90% confidence
# ±1.96 standard deviations = approximately 95% confidence
# ±2.576 standard deviations = approximately 99% confidence

# Color scheme matching original script
colors = {True: 'darkorange', False: 'royalblue'}  # True=catalyzed, False=control

# Create single figure
plt.figure(figsize=(7.25, 5), dpi=300)

# ============================================================================
# Plot Control Scenario
# ============================================================================

is_catalyzed_control = False

# Plot the entire control curve
plt.plot(time_points, mean_pred_control, '-', 
        label=f'Predicted (Control)\nMean Uncertainty: {uncertainty_control.mean():.2f}%', 
        color=colors[is_catalyzed_control], linewidth=2)

plt.fill_between(time_points, 
                mean_pred_control - z_score * uncertainty_control, 
                mean_pred_control + z_score * uncertainty_control, 
                alpha=0.2, color=colors[is_catalyzed_control], 
                label=f'±{z_score:.2f}σ Uncertainty (Control)')

# ============================================================================
# Plot Catalyzed Scenario (Only After Transition Time)
# ============================================================================

is_catalyzed_catalyzed = True
tt_value = catalyst_start_day

# Filter to only plot the catalyzed portion (after transition time)
post_transition_mask = time_points >= tt_value
time_points_filtered = time_points[post_transition_mask]
mean_pred_catalyzed_filtered = mean_pred_catalyzed[post_transition_mask]
uncertainty_catalyzed_filtered = uncertainty_catalyzed[post_transition_mask]

# Plot only the catalyzed portion
if len(time_points_filtered) > 0:
    plt.plot(time_points_filtered, mean_pred_catalyzed_filtered, '-', 
            label=f'Predicted (Catalyzed)\nMean Uncertainty: {uncertainty_catalyzed.mean():.2f}%', 
            color=colors[is_catalyzed_catalyzed], linewidth=2)
    
    plt.fill_between(time_points_filtered, 
                    mean_pred_catalyzed_filtered - z_score * uncertainty_catalyzed_filtered, 
                    mean_pred_catalyzed_filtered + z_score * uncertainty_catalyzed_filtered, 
                    alpha=0.2, color=colors[is_catalyzed_catalyzed], 
                    label=f'±{z_score:.2f}σ Uncertainty (Catalyzed)')

# Add transition time line
if tt_value > 0:
    plt.vlines(x=tt_value, ymin=0, ymax=80, 
              color=colors[is_catalyzed_catalyzed], 
              linestyle='--', alpha=0.7, 
              label=f'Transition Time')

# ============================================================================
# Finalize Plot (Matching Original Script Style)
# ============================================================================

plt.title(f"Ensemble Predictions - Extended to {int(time_points.max())} Days", 
         fontsize=14, fontweight='bold')
plt.xlabel("Leach Duration (Days)", fontsize=12)
plt.ylabel("Cu Recovery (%)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9, loc='best')
plt.ylim(0, 80)
plt.xlim(0, int(time_points.max()))
plt.tight_layout()
plt.savefig(folder_path + 'test/prediction_with_uncertainty.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as 'prediction_with_uncertainty.png'")


# ============================================================================
# Additional Information
# ============================================================================

print("\n" + "="*80)
print("PREDICTED PARAMETERS")
print("="*80)

print("\nControl scenario parameters:")
print(f"  a1 (asymptote 1): {params_control[0, 0]:.4f}")
print(f"  b1 (rate 1): {params_control[0, 1]:.4f}")
print(f"  a2 (asymptote 2): {params_control[0, 2]:.4f}")
print(f"  b2 (rate 2): {params_control[0, 3]:.4f}")

print("\nCatalyzed scenario parameters:")
print(f"  a1 (asymptote 1): {params_catalyzed[0, 0]:.4f}")
print(f"  b1 (rate 1): {params_catalyzed[0, 1]:.4f}")
print(f"  a2 (asymptote 2): {params_catalyzed[0, 2]:.4f}")
print(f"  b2 (rate 2): {params_catalyzed[0, 3]:.4f}")
print(f"  a3 (catalyst asymptote 1): {params_catalyzed[0, 4]:.4f}")
print(f"  b3 (catalyst rate 1): {params_catalyzed[0, 5]:.4f}")
print(f"  a4 (catalyst asymptote 2): {params_catalyzed[0, 6]:.4f}")
print(f"  b4 (catalyst rate 2): {params_catalyzed[0, 7]:.4f}")

print("\n" + "="*80)
print("DONE!")
print("="*80)

#%%
