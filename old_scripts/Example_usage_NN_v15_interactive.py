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

# ============================================================================
# STEP 0: Define the model classes (REQUIRED for unpickling)
# ============================================================================

class AdaptiveTwoPhaseRecoveryModel(nn.Module):
    """Enhanced Two-Phase Recovery Model"""
    def __init__(self, total_features, hidden_dim=128, dropout_rate=0.3, init_mode='kaiming'):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.init_mode = init_mode
        
        self.base_network = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, 4),
        )
        
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
        params = torch.zeros(batch_size, 8, device=x.device)
        
        if sample_ids is not None:
            unique_sample_ids = list(set(sample_ids))
            sample_id_to_base_params = {}
            
            for unique_id in unique_sample_ids:
                sample_indices = [i for i, sid in enumerate(sample_ids) if sid == unique_id]
                
                if sample_indices:
                    first_idx = sample_indices[0]
                    base_raw = self.base_network(x[first_idx:first_idx+1])
                    
                    a1 = 10.0 + 30.0 * torch.sigmoid(base_raw[:, 0])
                    b1 = 0.001 + 0.1 * torch.sigmoid(base_raw[:, 1])
                    a2 = 5.0 + 20.0 * torch.sigmoid(base_raw[:, 2])
                    b2 = 0.0001 + 0.1 * torch.sigmoid(base_raw[:, 3])

                    total_asymptote = a1 + a2
                    mask_a = total_asymptote > 70.0
                    scale = torch.where(mask_a & (total_asymptote > 0), 
                                       70.0 / total_asymptote.clamp(min=1.0), 
                                       torch.tensor(1.0, device=x.device))
                    a1, a2 = a1 * scale, a2 * scale
                    
                    sample_id_to_base_params[unique_id] = torch.stack([a1, b1, a2, b2], dim=1).squeeze(0)
                    
                    for idx in sample_indices:
                        params[idx, :4] = sample_id_to_base_params[unique_id]
        else:
            base_raw = self.base_network(x)
            
            a1 = 10.0 + 30.0 * torch.sigmoid(base_raw[:, 0])
            b1 = 0.001 + 0.1 * torch.sigmoid(base_raw[:, 1])
            a2 = 5.0 + 20.0 * torch.sigmoid(base_raw[:, 2])
            b2 = 0.0001 + 0.1 * torch.sigmoid(base_raw[:, 3])

            total_asymptote = a1 + a2
            mask_a = total_asymptote > 70.0
            scale = torch.where(mask_a & (total_asymptote > 0), 
                               70.0 / total_asymptote.clamp(min=1.0), 
                               torch.tensor(1.0, device=x.device))
            a1, a2 = a1 * scale, a2 * scale
            
            params[:, :4] = torch.stack([a1, b1, a2, b2], dim=1)
        
        has_catalyst = torch.any(catalyst > 0, dim=1)
        if has_catalyst.any():
            idx = has_catalyst.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                a1_r, b1_r, a2_r, b2_r = [p.squeeze() for p in params[idx, :4].split(1, dim=1)]
                
                cat_raw = self.catalyst_network(x[idx])
                
                a3 = 1.0 + 14.0 * torch.sigmoid(cat_raw[:, 0])
                b3 = 0.0001 + 0.003 * torch.sigmoid(cat_raw[:, 1])
                a4 = 1.0 + 14.0 * torch.sigmoid(cat_raw[:, 2])
                b4 = 0.0001 + 0.003 * torch.sigmoid(cat_raw[:, 3])

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
    """Ensemble model with uncertainty quantification"""
    
    def __init__(self, model_states, val_losses, total_features, config, device, best_configs):
        self.device = device
        self.total_features = total_features
        self.config = config
        self.best_configs = best_configs
        self.models, self.weights = self._create_filtered_ensemble(
            model_states, val_losses, config
        )
     
    def _create_filtered_ensemble(self, model_states, val_losses, config):
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
        
        weighted_pred = np.average(all_model_predictions, axis=0, weights=self.weights)
        weighted_params = np.average(all_model_params, axis=0, weights=self.weights)
        uncertainty = np.std(all_model_predictions, axis=0)
        
        return weighted_pred, uncertainty, weighted_params
    
    def get_ensemble_info(self):
        return {
            'num_models': len(self.models),
            'weights': self.weights,
            'total_features': self.total_features,
        }


def generate_two_phase_recovery(time, catalyst, transition_time, params):
    """Two-phase recovery generation"""
    time = time.to(params.device)
    catalyst = catalyst.to(params.device)
    
    catalyst_effect = catalyst / (catalyst + 1)
    
    a1 = params[:, 0].unsqueeze(1)
    b1 = params[:, 1].unsqueeze(1)
    a2 = params[:, 2].unsqueeze(1)
    b2 = params[:, 3].unsqueeze(1)
    
    exp_term1 = torch.exp(-b1 * time)
    exp_term1 = torch.clamp(exp_term1, min=1e-8, max=1.0)
    exp_term2 = torch.exp(-b2 * time)
    exp_term2 = torch.clamp(exp_term2, min=1e-8, max=1.0)
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
            
            exp_term3 = torch.exp(-torch.abs(b3) * time_shifted * (1 + catalyst_effect))
            exp_term3 = torch.clamp(exp_term3, min=1e-8, max=1.0)
            exp_term4 = torch.exp(-torch.abs(b4) * time_shifted * (1 + catalyst_effect))
            exp_term4 = torch.clamp(exp_term4, min=1e-8, max=1.0)
            
            additional_recovery = (torch.abs(a3) * catalyst_effect * (1 - exp_term3) + 
                                 torch.abs(a4) * catalyst_effect * (1 - exp_term4))
            
            catalyst_enhancement = torch.where(has_catalyst_points, additional_recovery, torch.zeros_like(additional_recovery))
            recovery = recovery_control + catalyst_enhancement
    
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
    'acid_gen_sulfides': {'label': 'Acid Gen. Sulfides', 'min': 0.0, 'max': 12.0, 'step': 0.5, 'value': 1.5},
    'gangue_silicates': {'label': 'Gangue Silicates', 'min': 75.0, 'max': 98.0, 'step': 0.1, 'value': 95.8},
    'fe_oxides': {'label': 'Fe Oxides', 'min': 0.0, 'max': 3.2, 'step': 0.1, 'value': 0.5},
    'carbonates': {'label': 'Carbonates', 'min': 0.0, 'max': 2.1, 'step': 0.1, 'value': 0.0},
    'catalyst_start_day': {'label': 'Catalyst Start Day', 'min': 0, 'max': 500, 'step': 5, 'value': 250},
    'catalyst_dose': {'label': 'Catalyst Dose (kg/t/day)', 'min': 0.0, 'max': 0.01, 'step': 0.0001, 'value': 0.0009},
    'max_time': {'label': 'Max Time (days)', 'min': 100, 'max': 2500, 'step': 100, 'value': 1000},
}

# Compute initial values for dependent sliders
slider_configs['cyanide_soluble'] = {'label': 'Cyanide Soluble %', 'min': -25.0, 'max': 55.0, 'step': 0.1, 'value': 100 - slider_configs['acid_soluble']['value'] - slider_configs['residual_cpy']['value']}

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
    if (value is not None and value < 0) or (value is not None and value > 55.0):
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
    
    # Create plot
    z_score = 1.645
    
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
        name=f'CI (±{np.round(z_score, 2)}σ)',
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
        name=f'CI (±{np.round(z_score, 2)}σ)',
        customdata=np.column_stack([lower_control, upper_control]),
        hovertemplate='CI (±1.65σ): [%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>',
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
            name=f'CI (±{np.round(z_score, 2)}σ)',
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
            name=f'CI (±{np.round(z_score, 2)}σ)',
            customdata=np.column_stack([lower_catalyzed, upper_catalyzed]),
            hovertemplate='CI (±1.65σ): [%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>',
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
        html.H5("Recovery Summary:"),
        html.P([
            f"Control Recovery at day {int(max_time)}: {mean_pred_control[-1]:.1f}%", html.Br(),
            f"Catalyzed Recovery at day {int(max_time)}: {mean_pred_catalyzed[-1]:.1f}%", html.Br(),
            html.Strong(f"Catalyst Benefit: {mean_pred_catalyzed[-1] - mean_pred_control[-1]:.1f}%"),
        ]),
        html.H5("Control Scenario Parameters:"),
        html.P([
            f"a1 (asymptote 1):     {params_control[0, 0]:.1f}", html.Br(),
            f"b1 (rate 1):          {params_control[0, 1]:.4f}", html.Br(),
            f"a2 (asymptote 2):     {params_control[0, 2]:.1f}", html.Br(),
            f"b2 (rate 2):          {params_control[0, 3]:.4f}",
        ]),
        html.H5("Catalyzed Scenario Parameters:"),
        html.P([
            # f"a1 (asymptote 1): {params_catalyzed[0, 0]:.4f}", html.Br(),
            # f"b1 (rate 1): {params_catalyzed[0, 1]:.4f}", html.Br(),
            # f"a2 (asymptote 2): {params_catalyzed[0, 2]:.4f}", html.Br(),
            # f"b2 (rate 2): {params_catalyzed[0, 3]:.4f}", html.Br(),
            f"a3 (catalyst asymptote 1):    {params_catalyzed[0, 4]:.1f}", html.Br(),
            f"b3 (catalyst rate 1):         {params_catalyzed[0, 5]:.4f}", html.Br(),
            f"a4 (catalyst asymptote 2):    {params_catalyzed[0, 6]:.1f}", html.Br(),
            f"b4 (catalyst rate 2):         {params_catalyzed[0, 7]:.4f}",
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