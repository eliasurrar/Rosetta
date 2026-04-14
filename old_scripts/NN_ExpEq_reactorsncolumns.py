#%%
import os
import json
import ast
import re
import hashlib
import shutil
import warnings
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------
# PyTorch Imports and Setup
# ---------------------------
# Try to use MPS (Mac), CUDA if available, otherwise CPU
'''
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device for PyTorch")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device for PyTorch")
else:
    device = torch.device("cpu")
    print("Using CPU device for PyTorch")
'''
device = torch.device("cpu")

# ---------------------------
# Force full CPU utilization (CPU only)
# ---------------------------
CPU_COUNT = os.cpu_count() or 1
if device.type == "cpu":
    os.environ["OMP_NUM_THREADS"] = str(CPU_COUNT)
    os.environ["MKL_NUM_THREADS"] = str(CPU_COUNT)
    torch.set_num_threads(CPU_COUNT)
    torch.set_num_interop_threads(max(1, CPU_COUNT // 2))
    print(f"CPU cores available: {CPU_COUNT} (threads set)")
else:
    print(f"Non-CPU device in use ({device.type}); skipping CPU thread overrides.")

# device = torch.device("cpu")

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


# ---------------------------
# Load Data
# ---------------------------
df_model_recCu_catcontrol_projects = pd.read_csv(
    "/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Rosetta/database_ready/df_recCu_catcontrol_projects_averaged.csv",
    sep=",",
)
df_reactors = pd.read_csv(
    "/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Rosetta/database_ready/df_reactors_filtered.csv",
    sep=",",
)

TIME_COL_REACTORS = "leach_duration_days_const" if "leach_duration_days_const" in df_reactors.columns else "leach_duration_days"
TIME_COL_COLUMNS = "leach_duration_days"

DEFAULT_PROJECT_ROOT = "/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Rosetta/NN_Pytorch_ExpEq"
LOCAL_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NN_Pytorch_ExpEq")
PROJECT_ROOT = DEFAULT_PROJECT_ROOT

def _can_write_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


for candidate in [DEFAULT_PROJECT_ROOT, LOCAL_PROJECT_ROOT]:
    if _can_write_dir(candidate):
        PROJECT_ROOT = candidate
        break

PLOTS_ROOT = os.path.join(PROJECT_ROOT, "plots")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs")
INPUT_EXAMPLE_ROOT = os.path.join(PROJECT_ROOT, "input_example")

os.makedirs(PLOTS_ROOT, exist_ok=True)
os.makedirs(MODELS_ROOT, exist_ok=True)
os.makedirs(OUTPUTS_ROOT, exist_ok=True)
os.makedirs(INPUT_EXAMPLE_ROOT, exist_ok=True)


# ---------------------------
# Model Definitions
# ---------------------------
HEADERS_DICT_REACTORS = {
    "leach_duration_days_const": ["Leach Duration (days)", "numerical", 1],
    "avg_h2so4_kg_t": ["Acid Consumption (kg/t)", "numerical", 0],
    "cu_%": ["CuT %", "numerical", 1],
    "acid_soluble_%": ["Acid Soluble Cu (%norm)", "numerical", 1],
    "cyanide_soluble_%": ["Cyanide Soluble (%norm)", "numerical", 0],
    "residual_cpy_%": ["Residual Chalcopyrite (%norm)", "numerical", 0],
    "grouped_copper_sulfides": ["Copper Sulphides (%)", "numerical", 0],
    "grouped_secondary_copper": ["Secondary Copper (%)", "numerical", 0],
    "grouped_acid_generating_sulfides": ["Acid Generating Sulphides (%)", "numerical", 0],
    # "grouped_gangue_silicates": ["Gangue Silicates (%)", "numerical", 0],
    "grouped_fe_oxides": ["Fe Oxides (%)", "numerical", 0],
    "grouped_carbonates": ["Carbonates (%)", "numerical", 0],
    "temp_(c)_mean": ["Avg. Temperature", "numerical", 1],
    "ph_mean": ["Avg. pH", "numerical", 0],
    "catalyst_type": ["is Catalyzed", "categorical", 1],
    "pH_target": ["pH", "categorical", 0],
    "catalyst_dose_(mg_l)": ["Catalyst Dose (mg/L)", "categorical", 1],
    "bornite": ["Bornite (%)", "numerical", 0],
    # "chlorite": ["Chlorite (%)", "numerical", 0],
    # "enargite": ["Enargite (%)", "numerical", 0],
}

if TIME_COL_REACTORS not in HEADERS_DICT_REACTORS:
    if "leach_duration_days_const" in HEADERS_DICT_REACTORS and TIME_COL_REACTORS != "leach_duration_days_const":
        HEADERS_DICT_REACTORS[TIME_COL_REACTORS] = HEADERS_DICT_REACTORS.pop("leach_duration_days_const")
    elif "leach_duration_days" in HEADERS_DICT_REACTORS and TIME_COL_REACTORS != "leach_duration_days":
        HEADERS_DICT_REACTORS[TIME_COL_REACTORS] = HEADERS_DICT_REACTORS.pop("leach_duration_days")
    else:
        HEADERS_DICT_REACTORS[TIME_COL_REACTORS] = ["Leach Duration (days)", "numerical", 1]

HEADERS_DICT_COLUMNS = {
    "leach_duration_days": ["Leach Duration (days)", "numerical", 1],
    "cumulative_catalyst_addition_kg_t": ["Cumulative Catalyst added (kg/t)", "numerical", 1],
    "cu_%": ["CuT %", "numerical", 1],
    "acid_soluble_%": ["Acid Soluble Cu (%norm)", "numerical", 1],
    "cyanide_soluble_%": ["Cyanide Soluble (%norm)", "numerical", 0],
    "residual_cpy_%": ["Residual Chalcopyrite (%norm)", "numerical", 0],
    "material_size_p80_in": ["Material Size P80 (in)", "numerical", -1],
    "grouped_copper_sulfides": ["Copper Sulphides (%)", "numerical", 0],
    "grouped_secondary_copper": ["Secondary Copper (%)", "numerical", 0],
    "grouped_acid_generating_sulfides": ["Acid Generating Sulphides (%)", "numerical", 0],
    # "grouped_gangue_silicates": ["Gangue Silicates (%)", "numerical", 0],
    "grouped_fe_oxides": ["Fe Oxides (%)", "numerical", 0],
    "grouped_carbonates": ["Carbonates (%)", "numerical", 0],
    "column_height_m": ["Column Height (m)", "numerical", 0],
    "column_inner_diameter_m": ["Column Inner Diameter (m)", "numerical", 0],
    "reactor_delta_prior": ["Reactors Delta Prior", "numerical", 1],
    "bornite": ["Bornite (%)", "numerical", 0],
    # "chlorite": ["Chlorite (%)", "numerical", 0],
    # "enargite": ["Enargite (%)", "numerical", 0],

}

TARGET_REACTORS = "cu_recovery_%_calc"
TARGET_COLUMNS = "cu_recovery_%"

CATALYZED_REACTORS_ID = "project_col_id"
CATALYZED_COLUMNS_ID = "project_col_id"


CONFIG = {
    "seed": 42,
    "ensemble": {
        "seeds": [1, 11, 21, 31, 41],
        "pi_lower": 5,
        "pi_upper": 95,
    },
    "experiment_tracking": {
        "enabled": True,
        "tag": "columns_curve",
        "root_dir_name": "experiments",
        "append_history_csv": True,
        "append_history_jsonl": True,
    },
    "param_calibration": {
        "enabled": True,
        "use_direct_targets": True,
        "min_rows_per_status": 6,
        "a_scale_min": 0.20,
        "a_scale_max": 1.50,
        "b_scale_min": 0.005,
        "b_scale_max": 1.25,
        "b2_ratio_cap_quantile": 0.50,
        "b2_ratio_cap_min": 0.02,
        "b2_ratio_cap_max": 0.95,
        "total_cap_quantile": 0.90,
        "total_cap_floor": 5.0,
    },
    "reactor_tuple_transfer": {
        "enabled": True,
        "rf_n_estimators": 500,
        "rf_min_samples_leaf": 2,
        "min_rows_per_status": 12,
        "k_neighbors": 7,
        "distance_eps": 1e-6,
    },
    "curve_model": {
        "use_reactor_similarity_params": True,
        "use_reactor_normalized_uplift_prior": True,
        "enforce_reactor_asymptote_cap": True,
        "force_include_shared_ore_features": True,
        "reactor_similarity_k": 7,
        "reactor_similarity_eps": 1e-6,
        "reactor_norm_uplift_clip_low": -80.0,
        "reactor_norm_uplift_clip_high": 400.0,
        "reactor_norm_uplift_control_floor": 0.25,
        "rf_n_estimators": 500,
        "rf_min_samples_leaf": 2,
        "fit_min_points": 6,
        "fit_maxfev": 60000,
        "fit_b_fast_upper": 3.0,
        "fit_b_min": 1e-5,
        "fit_a_min": 0.0,
        # Keep fast/slow semantics via sanitization; optional fit-time constraint can be enabled if needed.
        "fit_enforce_fast_slow_constraint": False,
        "fit_reject_if_stuck_on_first_seed": False,
        "fit_total_recovery_upper": 99.0,
        "columns_fit_total_recovery_upper_control": 80.0,
        "columns_fit_total_recovery_upper_catalyzed": 90.0,
        "fit_tmax_days": None,
        "fit_tmax_frac": 0.99,
        "fit_tmax_epsilon": 1e-4,
        "fit_last_n": 5,
        "fit_last_points_weight": (2.0, 10.0),
        "fit_jitter_starts": 0,
        "fit_jitter_scale": 0.3,
        "fit_random_state": None,
        "fit_tail_slope_threshold": 0.05,
        "fit_tail_mean_min_for_tmax_penalty": 80.0,
        "fit_tmax_penalty_lambda": 10.0,
    },
    "catalyst_curve_coupling": {
        "enabled": True,
        "touch_first_point": True,
        "first_point_max_gap": 0.05,
        "enforce_catalyzed_above_control": True,
        "a_uplift_per_kg_t": 3.87,
        "a_uplift_max": 0.14,
        "a_uplift_fast_fraction": 0.65,
        "b_uplift_per_kg_t": 0.235,
        "b_uplift_max_multiplier": 1.005,
        "min_catalyst_delta_for_uplift": 1e-6,
        "reactor_norm_uplift_weight": 0.002,
        "use_absolute_catalyst_signal": True,
        "absolute_catalyst_weight": 0.65,
        "min_catalyst_signal_for_uplift": 1e-6,
    },
    "inference_shaping": {
        "use_material_size_p80": True,
        "use_catalyst_context_offsets": True,
        "context_ore_features": None,
        "context_ore_weight_scale": 0.75,
        "context_ore_weight_clip": 0.35,
        "context_ore_score_clip": 2.0,
        "context_ore_sensitivity_a": 0.30,
        "context_ore_sensitivity_b": 0.22,
        "context_ore_sensitivity_dose": 0.30,
        "context_ore_sensitivity_tau": -0.18,
        "context_geom_weight_height": 0.25,
        "context_geom_weight_diameter": 0.20,
        "context_geom_weight_slenderness": 0.55,
        "context_geometry_score_clip": 2.0,
        "context_geometry_sensitivity_a": -0.18,
        "context_geometry_sensitivity_b": -0.14,
        "context_geometry_sensitivity_dose": -0.16,
        "context_geometry_sensitivity_tau": 0.24,
        "context_p80_score_clip": 2.0,
        "context_p80_sensitivity_a": -0.35,
        "context_p80_sensitivity_b": -0.24,
        "context_p80_sensitivity_dose": -0.28,
        "context_p80_sensitivity_tau": 0.30,
        "context_a_factor_min": 0.60,
        "context_a_factor_max": 1.60,
        "context_b_factor_min": 0.70,
        "context_b_factor_max": 1.45,
        "context_dose_factor_min": 0.65,
        "context_dose_factor_max": 1.50,
        "context_tau_factor_min": 0.70,
        "context_tau_factor_max": 1.80,
        "column_nonideal_a_factor": 0.50,
        "column_nonideal_b_factor": 0.585,
        "p80_reference_in": 0.48,
        "p80_b_sensitivity": 0.424,
        "p80_a_sensitivity": 0.283,
        "p80_b_factor_min": 0.25,
        "p80_b_factor_max": 1.176,
        "p80_a_factor_min": 0.20,
        "p80_a_factor_max": 1.081,
        "use_column_kinetics_caps": True,
        "b1_caps_from_fitted_params": True,
        "b1_cap_quantile": 0.90,
        "b1_cap_margin": 1.10,
        "b1_cap_min_quantile": 0.05,
        "b1_cap_max_quantile": 0.98,
        "b1_cap_min_margin": 0.85,
        "b1_cap_max_margin": 1.10,
        "b1_cap_abs_floor": 1e-4,
        "b1_cap_control": None,
        "b1_cap_catalyzed": None,
        "b1_cap_catalyzed_to_control_max_multiplier": 2.25,
        "b1_cap_min": None,
        "b1_cap_max": None,
        "b1_cap_reactor_softness": 0.15,
        "b1_cap_reactor_max_multiplier": 1.30,
        "b1_cap_p80_sensitivity": 0.35,
        "b1_cap_p80_factor_min": 0.45,
        "b1_cap_p80_factor_max": 1.25,
        "b1_cap_height_sensitivity": 0.15,
        "b1_cap_diameter_sensitivity": 0.05,
        "b1_cap_slenderness_sensitivity": 0.12,
        "b1_cap_geometry_factor_min": 0.60,
        "b1_cap_geometry_factor_max": 1.20,
        "b2_cap_ratio_to_b1": 0.75,
        "use_cumulative_catalyst": True,
        "catalyst_signal_weight_final": 0.6,
        "catalyst_signal_weight_avg": 0.4,
        "catalyst_reference_kg_t": None,
        "catalyst_b_sensitivity": 1.3, # 3.309
        "catalyst_a_sensitivity": 5.553,
        "catalyst_b_factor_min": 0.948,
        "catalyst_b_factor_max": 1.127,
        "catalyst_a_factor_min": 0.920,
        "catalyst_a_factor_max": 1.107,
        "catalyst_time_sensitivity": 1.626,
        "catalyst_time_factor_min": 0.993,
        "catalyst_time_factor_max": 1.0035,
        "use_dynamic_catalyst_uplift": True,
        "dynamic_uplift_tau_days": 35.0,
        "dynamic_uplift_jump_fraction": 0.35,
        "use_p80_catalyst_lag": True,
        "p80_lag_reference_in": None,
        "p80_lag_sensitivity": 0.20,
        "p80_lag_min_multiplier": 0.70,
        "p80_lag_max_multiplier": 4.00,
        "use_column_geometry_catalyst_lag": True,
        "geometry_lag_reference_height_m": None,
        "geometry_lag_reference_diameter_m": None,
        "height_tau_sensitivity_per_rel": 0.20,
        "diameter_tau_sensitivity_per_rel": 0.15,
        "slenderness_tau_sensitivity_per_rel": 0.10,
        "geometry_tau_factor_min": 0.70,
        "geometry_tau_factor_max": 2.50,
        "height_lag_days_per_rel": 8.0,
        "diameter_lag_days_per_rel": 5.0,
        "slenderness_lag_days_per_rel": 6.0,
        "dynamic_uplift_base_lag_days": 50.0,
        "dynamic_uplift_p80_lag_days_per_in": 20.0,
        "dynamic_uplift_lag_days_min": 0.0,
        "dynamic_uplift_lag_days_max": 300.0,
        "use_residual_cpy_uplift": True,
        "residual_cpy_reference_pct": None,
        "residual_uplift_sensitivity": 1.00,
        "residual_uplift_factor_min": 0.60,
        "residual_uplift_factor_max": 1.60,
        "use_residual_uplift_interactions": True,
        "residual_p80_interaction_sensitivity": -0.35,
        "residual_geometry_interaction_sensitivity": 0.20,
        "residual_interaction_factor_min": 0.65,
        "residual_interaction_factor_max": 1.35,
        "catalyst_gap_global_damp": 0.85,
        "use_catalyst_gap_soft_cap": True,
        "catalyst_gap_soft_cap_abs_pct": 40.0,
        "catalyst_gap_soft_cap_fraction_of_control": 3.0,
        "catalyst_gap_soft_cap_min_pct": 0.75,
        "catalyst_dose_gamma": 2.0,
        "catalyst_dose_factor_min": 0.2,
        "catalyst_dose_factor_max": 0.7,
        "use_transition_time_gating": True,
        "transition_blend_days": 24.65,
        "catalyst_progress_gamma": 1.77,
        "use_catalyst_residual_memory": True,
        "residual_memory_weight": 0.65,
        "residual_memory_tau_multiplier": 2.0,
        "dose_uses_absolute_cumulative": True,
    },
    "inference_uncertainty": {
        "enabled": True,
        "min_ensemble_members": 5,
        "reactor_distance_reference": None,
        "param_noise_a_base": 0.020,
        "param_noise_a_gain": 0.100,
        "param_noise_b_base": 0.030,
        "param_noise_b_gain": 0.180,
        "control_noise_multiplier": 1.00,
        "catalyzed_noise_multiplier": 1.20,
        "uncertainty_distance_weight": 0.35,
        "uncertainty_p80_weight": 0.25,
        "uncertainty_geometry_weight": 0.15,
        "uncertainty_catalyst_weight": 0.10,
        "uncertainty_residual_weight": 0.15,
        "uncertainty_score_max": 1.50,
    },
    "delta_transfer": {
        "k_neighbors": 5,
        "distance_eps": 1e-6,
        "p80_dampening_alpha": 0.5,
        "p80_dampening_min": 0.2,
        "time_norm": "log",
    },
    "reactors": {
        "hidden_dim": 96,
        "num_hidden_layers": 2,
        "dropout": 0.15,
        "epochs": 400,
        "patience": 60,
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "val_split": 0.25,
    },
    "columns": {
        "hidden_dim": 96,
        "num_hidden_layers": 2,
        "dropout": 0.15,
        "epochs": 500,
        "patience": 200,
        "batch_size": 256,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "val_split": 0.25,
        "freeze_pretrained_layers": False,
        # For parameter targets, copying only the first (shared-feature) layer is safer.
        "transfer_copy_hidden_layers": False,
        "transfer_copy_output_layer": False,
    },
}


ORE_SIMILARITY_CANDIDATES = [
    "cu_%",
    "acid_soluble_%",
    "cyanide_soluble_%",
    "residual_cpy_%",
    # "cu_seq_a_r_%",
    # "cu_seq_h2so4_%",
    # "cu_seq_nacn_%",
    "grouped_copper_sulfides",
    "grouped_secondary_copper",
    "grouped_acid_generating_sulfides",
    # "grouped_gangue_silicates",
    "grouped_gangue_sulfides",
    "grouped_fe_oxides",
    "grouped_carbonates",
    # "grouped_accessory_minerals",
    "bornite",
    "enargite",
    "chlorite",
]

COLUMN_EXTRA_FEATURE_CANDIDATES = [
    "material_size_p80_in",
    "transition_time",
    "column_height_m",
    "column_inner_diameter_m",
    # "irrigation_rate_l_m2_h",
    # "feed_head_cu_%",
    # "feed_head_fe_%",
    # "feed_mass_kg",
    # "reactors_PCA1",
    # "reactors_PCA2",
    # "catalyst_saturation_inside_column_day",
]

REACTOR_TUPLE_TRANSFER_FEATURES = [
    "reactor_tuple_rf_a1",
    "reactor_tuple_rf_b1",
    "reactor_tuple_rf_a2",
    "reactor_tuple_rf_b2",
    "reactor_tuple_rf_a_total",
    "reactor_tuple_rf_a_fast_frac",
    "reactor_tuple_rf_b_ratio",
    "reactor_tuple_rf_tau1_days",
    "reactor_tuple_rf_tau2_days",
    "reactor_tuple_rf_b_gap_log",
    "reactor_tuple_rf_mean_distance",
]

REACTOR_SIM_RELATION_FEATURES = [
    "reactor_sim_a_total",
    "reactor_sim_a_fast_frac",
    "reactor_sim_b_ratio",
    "reactor_sim_tau1_days",
    "reactor_sim_tau2_days",
    "reactor_sim_b_gap_log",
]


# ---------------------------
# Helper functions
# ---------------------------
def parse_listlike(value) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if value is None:
        return np.asarray([], dtype=float)
    if isinstance(value, float) and np.isnan(value):
        return np.asarray([], dtype=float)
    s = str(value).strip()
    if not s:
        return np.asarray([], dtype=float)
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return np.asarray([], dtype=float)
        s_clean = re.sub(r"\bnan\b", "None", s, flags=re.IGNORECASE)
        try:
            parsed = ast.literal_eval(s_clean)
            out = []
            for v in parsed:
                if v is None:
                    out.append(np.nan)
                else:
                    try:
                        out.append(float(v))
                    except Exception:
                        out.append(np.nan)
            return np.asarray(out, dtype=float)
        except Exception:
            # Fallback: extract numeric tokens from space/comma separated arrays
            tokens = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?|nan", inner, flags=re.IGNORECASE)
            if not tokens:
                return np.asarray([], dtype=float)
            out = []
            for tok in tokens:
                if tok.lower() == "nan":
                    out.append(np.nan)
                else:
                    try:
                        out.append(float(tok))
                    except Exception:
                        out.append(np.nan)
            return np.asarray(out, dtype=float)
    try:
        return np.asarray([float(s)], dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def normalize_status(value: str) -> str:
    status = str(value).strip().lower()
    if status.startswith("cat"):
        return "Catalyzed"
    return "Control"


def _serialize_array_for_excel(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=float)
    out = [None if not np.isfinite(v) else float(v) for v in arr]
    return json.dumps(out)


def _generate_inference_input_example_excel(
    df_columns_params: pd.DataFrame,
    ore_similarity_features: List[str],
    reactor_status_medians: Dict[str, np.ndarray],
    output_dir: str,
    random_state: int = None,
) -> Tuple[str, str, List[str]]:
    """
    Build an always-updated inference input example workbook from current script features.
    The schema is derived dynamically from ore_similarity_features and COLUMN_EXTRA_FEATURE_CANDIDATES.
    """
    if df_columns_params.empty:
        raise ValueError("Cannot build input example: df_columns_params is empty.")

    df = df_columns_params.copy()
    df["_status_norm"] = df[CATALYZED_COLUMNS_ID].apply(normalize_status)

    valid_ids = []
    for sample_id, group in df.groupby("project_sample_id"):
        statuses = set(group["_status_norm"].tolist())
        if "Control" in statuses and "Catalyzed" in statuses:
            valid_ids.append(sample_id)
    if not valid_ids:
        raise ValueError("No complete Control/Catalyzed sample pairs available for input example.")

    rng = np.random.default_rng(random_state)
    selected_sample_id = str(rng.choice(np.asarray(valid_ids, dtype=object)))
    sample_group = df[df["project_sample_id"].astype(str) == selected_sample_id].copy()
    row_ctrl = sample_group[sample_group["_status_norm"] == "Control"].iloc[0]
    row_cat = sample_group[sample_group["_status_norm"] == "Catalyzed"].iloc[0]

    t_days = np.asarray(row_cat.get("curve_time_days", []), dtype=float)
    if t_days.size == 0:
        t_days = np.asarray(row_ctrl.get("curve_time_days", []), dtype=float)
    catalyst_profile = _align_profile_to_time_length(
        np.asarray(row_cat.get("curve_catalyst_profile", []), dtype=float),
        len(t_days),
    )

    def _to_num(row: pd.Series, col: str, default=np.nan) -> float:
        v = pd.to_numeric(row.get(col, default), errors="coerce")
        return float(v) if np.isfinite(v) else float(default) if np.isfinite(default) else np.nan

    def _get_reactor_param_block(row: pd.Series, status: str) -> np.ndarray:
        defaults = reactor_status_medians.get(status, np.array([np.nan, np.nan, np.nan, np.nan], dtype=float))
        vals = []
        for i, p_name in enumerate(["a1", "b1", "a2", "b2"]):
            v = pd.to_numeric(row.get(f"reactor_sim_{p_name}", np.nan), errors="coerce")
            if not np.isfinite(v):
                v = defaults[i]
            vals.append(float(v) if np.isfinite(v) else np.nan)
        return np.asarray(vals, dtype=float)

    p_ctrl = _get_reactor_param_block(row_ctrl, "Control")
    p_cat = _get_reactor_param_block(row_cat, "Catalyzed")

    compact_input_row = {
        "project_sample_id": selected_sample_id,
        TIME_COL_COLUMNS: _serialize_array_for_excel(t_days),
        "cumulative_catalyst_addition_kg_t": _serialize_array_for_excel(catalyst_profile),
        "transition_time": _to_num(row_cat, "transition_time"),
        "material_size_p80_in": _to_num(row_cat, "material_size_p80_in"),
        "column_height_m": _to_num(row_cat, "column_height_m"),
        "column_inner_diameter_m": _to_num(row_cat, "column_inner_diameter_m"),
        "residual_cpy_%": _to_num(row_cat, "residual_cpy_%"),
        "reactor_control_a1": float(p_ctrl[0]),
        "reactor_control_b1": float(p_ctrl[1]),
        "reactor_control_a2": float(p_ctrl[2]),
        "reactor_control_b2": float(p_ctrl[3]),
        "reactor_catalyzed_a1": float(p_cat[0]),
        "reactor_catalyzed_b1": float(p_cat[1]),
        "reactor_catalyzed_a2": float(p_cat[2]),
        "reactor_catalyzed_b2": float(p_cat[3]),
        "reactor_norm_uplift_prior_pct": _to_num(row_cat, "reactor_norm_uplift_prior_pct", default=0.0),
    }

    dynamic_context_cols = list(
        dict.fromkeys(
            [c for c in ore_similarity_features if c in df.columns]
            + [c for c in COLUMN_EXTRA_FEATURE_CANDIDATES if c in df.columns]
        )
    )
    # Avoid duplicate aliases in the example template.
    alias_exclusions = {"transition_time"}
    dynamic_context_cols = [c for c in dynamic_context_cols if c not in alias_exclusions]
    for col in dynamic_context_cols:
        if col in compact_input_row:
            continue
        if col in ["curve_time_days", "curve_catalyst_profile", "curve_target_recovery"]:
            continue
        v = row_cat.get(col, np.nan)
        if isinstance(v, (list, tuple, np.ndarray)):
            compact_input_row[col] = _serialize_array_for_excel(np.asarray(v, dtype=float))
        else:
            num = pd.to_numeric(v, errors="coerce")
            compact_input_row[col] = float(num) if np.isfinite(num) else np.nan

    compact_col_order = [
        "project_sample_id",
        TIME_COL_COLUMNS,
        "cumulative_catalyst_addition_kg_t",
        "transition_time",
        "material_size_p80_in",
        "column_height_m",
        "column_inner_diameter_m",
        "residual_cpy_%",
        "reactor_control_a1",
        "reactor_control_b1",
        "reactor_control_a2",
        "reactor_control_b2",
        "reactor_catalyzed_a1",
        "reactor_catalyzed_b1",
        "reactor_catalyzed_a2",
        "reactor_catalyzed_b2",
        "reactor_norm_uplift_prior_pct",
    ]
    compact_columns = compact_col_order + [c for c in dynamic_context_cols if c not in compact_col_order]
    example_compact_df = pd.DataFrame([compact_input_row], columns=compact_columns)

    static_cols = [c for c in compact_columns if c not in {TIME_COL_COLUMNS, "cumulative_catalyst_addition_kg_t"}]
    n_points = max(len(t_days), 1)
    exploded_rows = []
    for i in range(n_points):
        rec = {k: compact_input_row.get(k, np.nan) for k in static_cols}
        rec[TIME_COL_COLUMNS] = float(t_days[i]) if i < len(t_days) and np.isfinite(t_days[i]) else np.nan
        cat_val = catalyst_profile[i] if i < len(catalyst_profile) else np.nan
        rec["cumulative_catalyst_addition_kg_t"] = float(cat_val) if np.isfinite(cat_val) else np.nan
        exploded_rows.append(rec)

    exploded_col_order = [
        "project_sample_id",
        TIME_COL_COLUMNS,
        "cumulative_catalyst_addition_kg_t",
    ] + [c for c in static_cols if c != "project_sample_id"]
    example_exploded_df = pd.DataFrame(exploded_rows, columns=exploded_col_order)

    schema_rows = []

    def _schema(var: str, required: bool, dtype: str, description: str, source: str) -> None:
        schema_rows.append(
            {
                "input_variable": var,
                "required": bool(required),
                "dtype": dtype,
                "description": description,
                "example_source": source,
            }
        )

    _schema("project_sample_id", True, "string", "Unique sample identifier.", "Columns sample metadata")
    _schema(TIME_COL_COLUMNS, True, "float", "Time point (days), exploded format one row per point.", "Columns leach_duration_days")
    _schema("cumulative_catalyst_addition_kg_t", True, "float", "Catalyst cumulative value at leach_duration_days, exploded format.", "Columns cumulative catalyst profile")
    _schema("transition_time", True, "float", "Catalyst start/transition day in columns.", "Columns transition_time")
    _schema("material_size_p80_in", True, "float", "Material size p80 (in).", "Columns material_size_p80_in")
    _schema("column_height_m", True, "float", "Column height (m), affects catalyst response kinetics.", "Columns geometry")
    _schema("column_inner_diameter_m", True, "float", "Column inner diameter (m), affects catalyst response kinetics.", "Columns geometry")
    _schema("residual_cpy_%", True, "float", "Residual chalcopyrite (%), major catalyst uplift magnitude driver.", "Ore characterization")
    _schema("reactor_control_a1", True, "float", "Reactor control fast component a1.", "Reactor similarity prior / median fallback")
    _schema("reactor_control_b1", True, "float", "Reactor control fast kinetics b1.", "Reactor similarity prior / median fallback")
    _schema("reactor_control_a2", True, "float", "Reactor control slow component a2.", "Reactor similarity prior / median fallback")
    _schema("reactor_control_b2", True, "float", "Reactor control slow kinetics b2.", "Reactor similarity prior / median fallback")
    _schema("reactor_catalyzed_a1", True, "float", "Reactor catalyzed fast component a1.", "Reactor similarity prior / median fallback")
    _schema("reactor_catalyzed_b1", True, "float", "Reactor catalyzed fast kinetics b1.", "Reactor similarity prior / median fallback")
    _schema("reactor_catalyzed_a2", True, "float", "Reactor catalyzed slow component a2.", "Reactor similarity prior / median fallback")
    _schema("reactor_catalyzed_b2", True, "float", "Reactor catalyzed slow kinetics b2.", "Reactor similarity prior / median fallback")
    _schema("reactor_norm_uplift_prior_pct", False, "float", "Optional normalized uplift prior from reactors.", "Reactor uplift prior")

    transfer_required_cols = set(dynamic_context_cols)
    for col in dynamic_context_cols:
        if col in compact_col_order:
            continue
        desc = HEADERS_DICT_COLUMNS.get(col, [col])[0] if col in HEADERS_DICT_COLUMNS else col
        _schema(
            col,
            col in transfer_required_cols,
            "float",
            f"Dynamic model context feature: {desc}.",
            "Auto-derived from current script feature lists",
        )

    schema_df = pd.DataFrame(schema_rows)
    meta_df = pd.DataFrame(
        [
            {
                "selected_example_project_sample_id": selected_sample_id,
                "n_input_columns_exploded": len(exploded_col_order),
                "n_rows_exploded": len(example_exploded_df),
                "n_input_columns_compact": len(compact_columns),
                "n_dynamic_context_columns": len(dynamic_context_cols),
                "workbook_layout": "input_example is exploded (one row per time point); input_example_compact keeps arrays.",
                "generated_from_ore_similarity_features": json.dumps(ore_similarity_features),
                "generated_from_column_extra_features": json.dumps([c for c in COLUMN_EXTRA_FEATURE_CANDIDATES if c in df.columns]),
            }
        ]
    )

    os.makedirs(output_dir, exist_ok=True)
    xlsx_path = os.path.join(output_dir, "columns_model_inference_input_example.xlsx")
    with pd.ExcelWriter(xlsx_path) as writer:
        example_exploded_df.to_excel(writer, sheet_name="input_example", index=False)
        example_compact_df.to_excel(writer, sheet_name="input_example_compact", index=False)
        schema_df.to_excel(writer, sheet_name="input_schema", index=False)
        meta_df.to_excel(writer, sheet_name="metadata", index=False)
    return xlsx_path, selected_sample_id, exploded_col_order


def construct_reactor_recovery_curve(a1, b1, a2, b2, t_days):
    """Double-exponential recovery curve used for reactors (same functional form as columns)."""
    t = np.asarray(t_days, dtype=float)
    t = np.clip(t, 0.0, None)
    return np.round(
        (
            float(a1) * (1.0 - np.exp(-float(b1) * t))
            + float(a2) * (1.0 - np.exp(-float(b2) * t))
        ),
        1,
    )


def _double_exp_curve(a1: float, b1: float, a2: float, b2: float, t_days: np.ndarray) -> np.ndarray:
    t = np.asarray(t_days, dtype=float)
    t = np.clip(t, 0.0, None)
    return float(a1) * (1.0 - np.exp(-float(b1) * t)) + float(a2) * (1.0 - np.exp(-float(b2) * t))


def _double_exp_reparam_curve(
    t_days: np.ndarray,
    total_recovery: float,
    fast_fraction: float,
    b_fast: float,
    slow_ratio: float,
) -> np.ndarray:
    total_recovery = float(total_recovery)
    fast_fraction = float(np.clip(fast_fraction, 0.0, 1.0))
    b_fast = float(max(b_fast, 1e-6))
    slow_ratio = float(np.clip(slow_ratio, 1e-4, 1.0))
    a1 = total_recovery * fast_fraction
    a2 = total_recovery * (1.0 - fast_fraction)
    b1 = b_fast
    b2 = b_fast * slow_ratio
    return _double_exp_curve(a1, b1, a2, b2, t_days)


def _enforce_fast_slow_pairing(params: np.ndarray) -> np.ndarray:
    p = np.asarray(params, dtype=float).copy()
    if p.size != 4 or not np.all(np.isfinite(p)):
        return p
    if p[3] > p[1]:
        p = np.array([p[2], p[3], p[0], p[1]], dtype=float)
    return p


def _sanitize_curve_params(
    a1: float,
    b1: float,
    a2: float,
    b2: float,
    total_recovery_upper: float = 100.0,
    b_upper: float = 5.0,
) -> np.ndarray:
    vals = np.array([a1, b1, a2, b2], dtype=float)
    if not np.all(np.isfinite(vals)):
        vals = np.array([30.0, 0.10, 20.0, 0.01], dtype=float)
    a1, b1, a2, b2 = vals
    a1 = max(0.0, float(a1))
    a2 = max(0.0, float(a2))
    b1 = float(np.clip(b1, 1e-5, b_upper))
    b2 = float(np.clip(b2, 1e-5, b_upper))
    ordered = _enforce_fast_slow_pairing(np.array([a1, b1, a2, b2], dtype=float))
    a1, b1, a2, b2 = [float(v) for v in ordered]

    total = a1 + a2
    if total > total_recovery_upper and total > 0:
        scale = total_recovery_upper / total
        a1 *= scale
        a2 *= scale
    return np.array([a1, b1, a2, b2], dtype=float)


def _cap_by_reactor_asymptote(params: np.ndarray, reactor_params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=float).copy()
    reactor_params = np.asarray(reactor_params, dtype=float)
    if params.size != 4 or reactor_params.size != 4 or not np.all(np.isfinite(reactor_params)):
        return params
    reactor_cap = float(max(0.0, reactor_params[0] + reactor_params[2]))
    total = float(max(0.0, params[0] + params[2]))
    if reactor_cap > 0 and total > reactor_cap:
        scale = reactor_cap / total
        params[0] *= scale
        params[2] *= scale
    return params


def _fit_double_exponential_params(
    t_days: np.ndarray,
    recovery: np.ndarray,
    fit_min_points: int,
    fit_maxfev: int,
    fit_b_fast_upper: float,
    fit_b_min: float,
    fit_a_min: float,
    fit_enforce_fast_slow_constraint: bool,
    fit_reject_if_stuck_on_first_seed: bool,
    fit_total_recovery_upper: float,
    catalyst_profile: np.ndarray = None,
    control_params_for_catalyzed: np.ndarray = None,
    transition_time: float = np.nan,
    material_size_p80_in: float = np.nan,
    column_height_m: float = np.nan,
    column_inner_diameter_m: float = np.nan,
    residual_cpy_pct: float = np.nan,
    row_context: pd.Series = None,
    shaping_cfg: Dict = None,
    coupling_cfg: Dict = None,
    tmax_days: float = None,
    frac: float = 0.99,
    epsilon: float = 1e-4,
    last_n: int = 10,
    last_points_weight: Tuple[float, float] = (2.0, 10.0),
    jitter_starts: int = 0,
    jitter_scale: float = 0.3,
    random_state: int = None,
    tail_slope_threshold: float = 0.05,
    tail_mean_min_for_tmax_penalty: float = 80.0,
    tmax_penalty_lambda: float = 10.0,
) -> np.ndarray:
    t = np.asarray(t_days, dtype=float)
    y = np.asarray(recovery, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() < fit_min_points:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
    t = t[valid]
    y = y[valid]

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    t_unique, inv = np.unique(t, return_inverse=True)
    if len(t_unique) < fit_min_points:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
    y_unique = np.zeros_like(t_unique, dtype=float)
    counts = np.zeros_like(t_unique, dtype=float)
    for i, j in enumerate(inv):
        y_unique[j] += y[i]
        counts[j] += 1.0
    y = y_unique / np.maximum(counts, 1.0)
    t = t_unique
    y = np.clip(y, 0.0, fit_total_recovery_upper)

    b_min = float(max(1e-6, fit_b_min))
    b_max = float(max(b_min, fit_b_fast_upper))
    a_min = float(max(0.0, fit_a_min))
    a_max = float(max(a_min + 1e-6, min(fit_total_recovery_upper, float(np.nanmax(y)) + 10.0)))

    plateau = float(np.nanpercentile(y, 90))
    a_guess1 = plateau / 2.0
    a_guess2 = plateau / 2.0
    b_guess1 = 0.1
    b_guess2 = 0.01

    def clamp_a1(x: float) -> float:
        return float(np.clip(x, a_min, a_max))

    def clamp_a2(x: float) -> float:
        return float(np.clip(x, a_min, a_max))

    def clamp_b(x: float) -> float:
        return float(np.clip(x, b_min, b_max))

    p0_list = [
        [clamp_a1(a_guess1), clamp_b(b_guess1), clamp_a2(a_guess2), clamp_b(b_guess2)],
        [clamp_a1(plateau), clamp_b(0.05), clamp_a2(a_min), clamp_b(0.01)],
        [clamp_a1(float(np.nanmax(y))), clamp_b(0.05), clamp_a2(a_min), clamp_b(0.01)],
        [clamp_a1(plateau * 0.7), clamp_b(0.2), clamp_a2(plateau * 0.3), clamp_b(0.02)],
        [clamp_a1(plateau * 0.9), clamp_b(0.03), clamp_a2(plateau * 0.1), clamp_b(0.005)],
        [clamp_a1(10.0), clamp_b(0.01), clamp_a2(10.0), clamp_b(0.01)],
    ]

    jitter_starts = int(max(0, jitter_starts))
    if jitter_starts > 0:
        rng = np.random.default_rng(random_state)
        if isinstance(jitter_scale, (tuple, list, np.ndarray)):
            if len(jitter_scale) == 2:
                a_scale, b_scale = float(jitter_scale[0]), float(jitter_scale[1])
                scales = [a_scale, b_scale, a_scale, b_scale]
            elif len(jitter_scale) == 4:
                scales = [float(s) for s in jitter_scale]
            else:
                scales = [float(jitter_scale[0])] * 4
        else:
            scales = [float(jitter_scale)] * 4
        scales = [max(0.0, s) for s in scales]
        jittered = []
        for base in p0_list:
            for _ in range(jitter_starts):
                factors = np.exp(rng.normal(0.0, scales))
                jittered.append(
                    [
                        clamp_a1(base[0] * factors[0]),
                        clamp_b(base[1] * factors[1]),
                        clamp_a2(base[2] * factors[2]),
                        clamp_b(base[3] * factors[3]),
                    ]
                )
        p0_list.extend(jittered)

    bounds = [(a_min, a_max), (b_min, b_max), (a_min, a_max), (b_min, b_max)]
    constraints = [{"type": "ineq", "fun": lambda p: float(fit_total_recovery_upper) - p[0] - p[2]}]
    if bool(fit_enforce_fast_slow_constraint):
        constraints.append({"type": "ineq", "fun": lambda p: p[1] - p[3]})

    last_n = int(max(0, last_n))
    weights = np.ones_like(t, dtype=np.float64)
    if last_n and last_points_weight is not None:
        if isinstance(last_points_weight, (tuple, list, np.ndarray)) and len(last_points_weight) == 2:
            weight_start, weight_end = last_points_weight
        else:
            weight_start, weight_end = 1.0, last_points_weight
        weight_start = float(np.clip(weight_start, 1.0, 10.0))
        weight_end = float(np.clip(weight_end, 1.0, 10.0))
        if max(weight_start, weight_end) > 1.0:
            last_indices = np.argsort(t)[-min(last_n, len(t)) :]
            last_indices = last_indices[np.argsort(t[last_indices])]
            weights[last_indices] = np.linspace(weight_start, weight_end, num=len(last_indices))

    last_idx = np.argsort(t)[-min(last_n if last_n > 0 else 1, len(t)) :]
    t_last = t[last_idx]
    y_last = y[last_idx]
    if len(t_last) >= 2:
        try:
            tail_slope = float(np.polyfit(t_last, y_last, 1)[0])
        except Exception:
            tail_slope = np.inf
    else:
        tail_slope = np.inf
    apply_tmax_penalty = tail_slope < float(tail_slope_threshold)
    apply_tmax_penalty_2 = float(np.nanmean(y_last)) >= float(tail_mean_min_for_tmax_penalty)

    shaping_cfg = dict(shaping_cfg or {})
    coupling_cfg = dict(coupling_cfg or {})
    use_catalyst_fit_logic = (
        control_params_for_catalyzed is not None
        and np.asarray(control_params_for_catalyzed, dtype=float).size == 4
        and np.all(np.isfinite(np.asarray(control_params_for_catalyzed, dtype=float)))
        and catalyst_profile is not None
    )
    if use_catalyst_fit_logic:
        ctrl_p = _sanitize_curve_params(
            control_params_for_catalyzed[0],
            control_params_for_catalyzed[1],
            control_params_for_catalyzed[2],
            control_params_for_catalyzed[3],
            total_recovery_upper=fit_total_recovery_upper,
            b_upper=fit_b_fast_upper,
        )
        ctrl_curve = _double_exp_curve(ctrl_p[0], ctrl_p[1], ctrl_p[2], ctrl_p[3], t)
        c_prof = _align_profile_to_time_length(np.asarray(catalyst_profile, dtype=float), len(t))
        if c_prof.size == 0:
            c_prof = np.zeros(len(t), dtype=float)
        c_prof = np.where(np.isfinite(c_prof) & (c_prof > 0), c_prof, 0.0)
        c_prof = np.maximum.accumulate(c_prof)
    else:
        ctrl_curve = None
        c_prof = None

    _ = epsilon  # kept for compatibility with existing fit signature

    def objective(p: np.ndarray) -> float:
        pred_base = _double_exp_curve(p[0], p[1], p[2], p[3], t)
        if use_catalyst_fit_logic and ctrl_curve is not None and c_prof is not None:
            pred = _apply_time_dependent_catalyst_separation(
                control_curve=ctrl_curve,
                catalyzed_curve=pred_base,
                catalyst_profile=c_prof,
                shaping_cfg=shaping_cfg,
                coupling_cfg=coupling_cfg,
                time_days=t,
                transition_time=transition_time,
                material_size_p80_in=material_size_p80_in,
                column_height_m=column_height_m,
                column_inner_diameter_m=column_inner_diameter_m,
                residual_cpy_pct=residual_cpy_pct,
                row_context=row_context,
            )
        else:
            pred = pred_base
        resid = (y - pred) * weights
        loss = float(np.sum(resid**2))

        if (
            apply_tmax_penalty
            and apply_tmax_penalty_2
            and tmax_days is not None
            and np.isfinite(tmax_days)
        ):
            asym = float(p[0] + p[2])
            if asym > 0:
                pred_t = float(_double_exp_curve(p[0], p[1], p[2], p[3], np.asarray([tmax_days]))[0])
                target = float(frac) * asym
                deficit = max(0.0, target - pred_t) / asym
                lam = float(max(0.0, tmax_penalty_lambda)) * float(np.sum(weights))
                loss += lam * (deficit**2)
        return loss

    best_res = None
    best_loss = np.inf
    for p0 in p0_list:
        try:
            res = minimize(
                objective,
                p0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": int(max(100, fit_maxfev))},
            )
            if res.success:
                curr = objective(res.x)
                if curr < best_loss:
                    best_loss = curr
                    best_res = res
        except Exception:
            continue

    if best_res is None or not best_res.success:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)

    popt = np.asarray(best_res.x, dtype=float)
    if bool(fit_reject_if_stuck_on_first_seed) and np.allclose(popt, np.asarray(p0_list[0], dtype=float), atol=1e-3):
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)

    return _sanitize_curve_params(
        popt[0],
        popt[1],
        popt[2],
        popt[3],
        total_recovery_upper=fit_total_recovery_upper,
        b_upper=fit_b_fast_upper,
    )


def _summarize_catalyst_profile(
    t_days: np.ndarray,
    catalyst_profile: np.ndarray,
    is_catalyzed: bool,
) -> Dict[str, float]:
    t = np.asarray(t_days, dtype=float)
    c = np.asarray(catalyst_profile, dtype=float)
    valid = np.isfinite(t) & np.isfinite(c)
    if valid.any():
        t = t[valid]
        c = c[valid]
        order = np.argsort(t)
        t = t[order]
        c = c[order]
    else:
        t = np.asarray([], dtype=float)
        c = np.asarray([], dtype=float)

    if not is_catalyzed:
        return {
            "catalyst_final_kg_t": 0.0,
            "catalyst_max_kg_t": 0.0,
            "catalyst_auc_kg_t_day": 0.0,
            "catalyst_rate_kg_t_day": 0.0,
        }

    if len(c) == 0:
        return {
            "catalyst_final_kg_t": np.nan,
            "catalyst_max_kg_t": np.nan,
            "catalyst_auc_kg_t_day": np.nan,
            "catalyst_rate_kg_t_day": np.nan,
        }

    t_max = float(np.nanmax(t)) if len(t) else np.nan
    c_final = float(c[-1]) if len(c) else np.nan
    c_max = float(np.nanmax(c)) if len(c) else np.nan
    c_auc = float(np.trapezoid(c, t)) if len(c) >= 2 else 0.0
    c_rate = float(c_final / t_max) if np.isfinite(t_max) and t_max > 0 else np.nan
    return {
        "catalyst_final_kg_t": c_final,
        "catalyst_max_kg_t": c_max,
        "catalyst_auc_kg_t_day": c_auc,
        "catalyst_rate_kg_t_day": c_rate,
    }


def _prepare_catalyst_profile(catalyst_profile: np.ndarray) -> np.ndarray:
    c = np.asarray(catalyst_profile, dtype=float)
    if c.size == 0:
        return np.asarray([], dtype=float)
    c = np.where(np.isfinite(c), c, np.nan)
    if np.isnan(c).all():
        return np.asarray([], dtype=float)
    series = pd.Series(c).interpolate(limit_direction="both")
    c_filled = series.to_numpy(dtype=float)
    c_filled = np.maximum.accumulate(c_filled)
    return np.maximum(c_filled, 0.0)


def _compute_catalyst_delta(catalyst_profile: np.ndarray) -> float:
    c_filled = _prepare_catalyst_profile(catalyst_profile)
    if c_filled.size == 0:
        return 0.0
    delta = float(c_filled[-1] - c_filled[0])
    return max(0.0, delta)


def _compute_catalyst_signal(
    catalyst_profile: np.ndarray,
    use_absolute_signal: bool,
    absolute_weight: float,
) -> float:
    c_filled = _prepare_catalyst_profile(catalyst_profile)
    if c_filled.size == 0:
        return 0.0

    delta = float(max(0.0, c_filled[-1] - c_filled[0]))
    final_level = float(max(0.0, c_filled[-1]))
    if not use_absolute_signal:
        return delta

    w_abs = float(np.clip(absolute_weight, 0.0, 1.0))
    return (1.0 - w_abs) * delta + w_abs * final_level


def _compute_catalyst_context_offsets(
    row_context: pd.Series = None,
    shaping_cfg: Dict = None,
    material_size_p80_in: float = np.nan,
    column_height_m: float = np.nan,
    column_inner_diameter_m: float = np.nan,
) -> Dict[str, float]:
    cfg = dict(shaping_cfg or {})
    defaults = {
        "a_factor": 1.0,
        "b_factor": 1.0,
        "dose_factor": 1.0,
        "tau_factor": 1.0,
        "ore_score": 0.0,
        "p80_score": 0.0,
        "geometry_score": 0.0,
    }
    if not bool(cfg.get("use_catalyst_context_offsets", True)):
        return defaults

    def _row_num(col: str, default=np.nan) -> float:
        if row_context is None:
            return float(default) if np.isfinite(default) else np.nan
        val = pd.to_numeric(row_context.get(col, default), errors="coerce")
        return float(val) if np.isfinite(val) else np.nan

    p80_val = pd.to_numeric(material_size_p80_in, errors="coerce")
    if not np.isfinite(p80_val):
        p80_val = _row_num("material_size_p80_in", default=np.nan)
    else:
        p80_val = float(p80_val)

    h_val = pd.to_numeric(column_height_m, errors="coerce")
    if not np.isfinite(h_val):
        h_val = _row_num("column_height_m", default=np.nan)
    else:
        h_val = float(h_val)

    d_val = pd.to_numeric(column_inner_diameter_m, errors="coerce")
    if not np.isfinite(d_val):
        d_val = _row_num("column_inner_diameter_m", default=np.nan)
    else:
        d_val = float(d_val)

    ore_feats = cfg.get("context_ore_features", []) or []
    ore_refs = cfg.get("context_ore_references", {}) or {}
    ore_scales = cfg.get("context_ore_scales", {}) or {}
    ore_weights = cfg.get("context_ore_weights", {}) or {}
    ore_score = 0.0
    for feat in ore_feats:
        val = _row_num(feat, default=np.nan)
        ref = pd.to_numeric(ore_refs.get(feat, np.nan), errors="coerce")
        scale = pd.to_numeric(ore_scales.get(feat, np.nan), errors="coerce")
        w = pd.to_numeric(ore_weights.get(feat, 0.0), errors="coerce")
        if not (np.isfinite(val) and np.isfinite(ref) and np.isfinite(scale) and np.isfinite(w)):
            continue
        if float(scale) <= 0:
            continue
        ore_score += float(w) * float((float(val) - float(ref)) / float(scale))
    ore_clip = float(max(0.1, cfg.get("context_ore_score_clip", 2.0)))
    ore_score = float(np.clip(ore_score, -ore_clip, ore_clip))

    p80_ref = pd.to_numeric(cfg.get("p80_reference_in", np.nan), errors="coerce")
    if np.isfinite(p80_val) and np.isfinite(p80_ref):
        p80_score = float((float(p80_val) - float(p80_ref)) / max(abs(float(p80_ref)), 1e-6))
    else:
        p80_score = 0.0
    p80_clip = float(max(0.1, cfg.get("context_p80_score_clip", 2.0)))
    p80_score = float(np.clip(p80_score, -p80_clip, p80_clip))

    h_ref = pd.to_numeric(cfg.get("geometry_lag_reference_height_m", np.nan), errors="coerce")
    d_ref = pd.to_numeric(cfg.get("geometry_lag_reference_diameter_m", np.nan), errors="coerce")
    if np.isfinite(h_val) and np.isfinite(d_val) and np.isfinite(h_ref) and np.isfinite(d_ref) and d_val > 0 and d_ref > 0:
        h_rel = float((h_val - float(h_ref)) / max(abs(float(h_ref)), 1e-6))
        d_rel = float((d_val - float(d_ref)) / max(abs(float(d_ref)), 1e-6))
        slender_val = float(h_val / max(d_val, 1e-6))
        slender_ref = float(float(h_ref) / max(float(d_ref), 1e-6))
        slender_rel = float((slender_val - slender_ref) / max(abs(slender_ref), 1e-6))
        geometry_score = (
            float(cfg.get("context_geom_weight_height", 0.25)) * h_rel
            - float(cfg.get("context_geom_weight_diameter", 0.20)) * d_rel
            + float(cfg.get("context_geom_weight_slenderness", 0.55)) * slender_rel
        )
    else:
        geometry_score = 0.0
    geom_clip = float(max(0.1, cfg.get("context_geometry_score_clip", 2.0)))
    geometry_score = float(np.clip(geometry_score, -geom_clip, geom_clip))

    log_a = (
        float(cfg.get("context_ore_sensitivity_a", 0.0)) * ore_score
        + float(cfg.get("context_p80_sensitivity_a", 0.0)) * p80_score
        + float(cfg.get("context_geometry_sensitivity_a", 0.0)) * geometry_score
    )
    log_b = (
        float(cfg.get("context_ore_sensitivity_b", 0.0)) * ore_score
        + float(cfg.get("context_p80_sensitivity_b", 0.0)) * p80_score
        + float(cfg.get("context_geometry_sensitivity_b", 0.0)) * geometry_score
    )
    log_dose = (
        float(cfg.get("context_ore_sensitivity_dose", 0.0)) * ore_score
        + float(cfg.get("context_p80_sensitivity_dose", 0.0)) * p80_score
        + float(cfg.get("context_geometry_sensitivity_dose", 0.0)) * geometry_score
    )
    log_tau = (
        float(cfg.get("context_ore_sensitivity_tau", 0.0)) * ore_score
        + float(cfg.get("context_p80_sensitivity_tau", 0.0)) * p80_score
        + float(cfg.get("context_geometry_sensitivity_tau", 0.0)) * geometry_score
    )

    a_factor = float(np.exp(log_a))
    b_factor = float(np.exp(log_b))
    dose_factor = float(np.exp(log_dose))
    tau_factor = float(np.exp(log_tau))

    a_factor = float(np.clip(a_factor, float(cfg.get("context_a_factor_min", 0.6)), float(cfg.get("context_a_factor_max", 1.6))))
    b_factor = float(np.clip(b_factor, float(cfg.get("context_b_factor_min", 0.7)), float(cfg.get("context_b_factor_max", 1.45))))
    dose_factor = float(
        np.clip(
            dose_factor,
            float(cfg.get("context_dose_factor_min", 0.65)),
            float(cfg.get("context_dose_factor_max", 1.5)),
        )
    )
    tau_factor = float(np.clip(tau_factor, float(cfg.get("context_tau_factor_min", 0.7)), float(cfg.get("context_tau_factor_max", 1.8))))

    return {
        "a_factor": a_factor,
        "b_factor": b_factor,
        "dose_factor": dose_factor,
        "tau_factor": tau_factor,
        "ore_score": ore_score,
        "p80_score": p80_score,
        "geometry_score": geometry_score,
    }


def _constrain_catalyzed_params_against_control(
    control_params: np.ndarray,
    catalyzed_params: np.ndarray,
    t_days: np.ndarray,
    catalyst_profile: np.ndarray,
    reactor_norm_uplift_prior_pct: float,
    coupling_cfg: Dict,
    fit_total_recovery_upper: float,
    fit_b_fast_upper: float,
    row_context: pd.Series = None,
    shaping_cfg: Dict = None,
) -> np.ndarray:
    p_ctrl = _sanitize_curve_params(
        control_params[0],
        control_params[1],
        control_params[2],
        control_params[3],
        total_recovery_upper=fit_total_recovery_upper,
        b_upper=fit_b_fast_upper,
    )
    p_cat = _sanitize_curve_params(
        catalyzed_params[0],
        catalyzed_params[1],
        catalyzed_params[2],
        catalyzed_params[3],
        total_recovery_upper=fit_total_recovery_upper,
        b_upper=fit_b_fast_upper,
    )

    catalyst_signal = _compute_catalyst_signal(
        catalyst_profile=catalyst_profile,
        use_absolute_signal=bool(coupling_cfg.get("use_absolute_catalyst_signal", False)),
        absolute_weight=float(coupling_cfg.get("absolute_catalyst_weight", 0.0)),
    )
    uplift_prior = pd.to_numeric(reactor_norm_uplift_prior_pct, errors="coerce")
    uplift_prior = float(max(0.0, uplift_prior)) if np.isfinite(uplift_prior) else 0.0
    uplift_weight = float(max(0.0, coupling_cfg.get("reactor_norm_uplift_weight", 0.0)))
    prior_boost = 1.0 + uplift_weight * (uplift_prior / 100.0)
    context_offsets = _compute_catalyst_context_offsets(
        row_context=row_context,
        shaping_cfg=shaping_cfg,
    )
    catalyst_signal_eff = catalyst_signal * float(context_offsets.get("dose_factor", 1.0))

    min_signal = float(coupling_cfg.get("min_catalyst_signal_for_uplift", coupling_cfg.get("min_catalyst_delta_for_uplift", 0.0)))
    if catalyst_signal_eff > min_signal:
        a_cap_base = float(coupling_cfg["a_uplift_max"])
        a_cap_ctx = a_cap_base * max(0.75, float(context_offsets.get("a_factor", 1.0)))
        a_uplift = float(
            np.clip(
                coupling_cfg["a_uplift_per_kg_t"] * catalyst_signal_eff * prior_boost * float(context_offsets.get("a_factor", 1.0)),
                0.0,
                a_cap_ctx,
            )
        )
        a_fast_frac = float(np.clip(coupling_cfg["a_uplift_fast_fraction"], 0.0, 1.0))
        b_mult = 1.0 + float(
            np.clip(
                coupling_cfg["b_uplift_per_kg_t"] * catalyst_signal_eff * prior_boost * float(context_offsets.get("b_factor", 1.0)),
                0.0,
                max(0.0, coupling_cfg["b_uplift_max_multiplier"] - 1.0),
            )
        )
        p_cat[0] = max(p_cat[0], p_ctrl[0] + a_uplift * a_fast_frac)
        p_cat[2] = max(p_cat[2], p_ctrl[2] + a_uplift * (1.0 - a_fast_frac))
        p_cat[1] = max(p_cat[1], p_ctrl[1] * b_mult)
        p_cat[3] = max(p_cat[3], p_ctrl[3] * b_mult)
    else:
        p_cat[0] = max(p_cat[0], p_ctrl[0])
        p_cat[1] = max(p_cat[1], p_ctrl[1])
        p_cat[2] = max(p_cat[2], p_ctrl[2])
        p_cat[3] = max(p_cat[3], p_ctrl[3])

    p_cat = _sanitize_curve_params(
        p_cat[0],
        p_cat[1],
        p_cat[2],
        p_cat[3],
        total_recovery_upper=fit_total_recovery_upper,
        b_upper=fit_b_fast_upper,
    )

    return p_cat


def _build_columns_curve_parameter_table(
    df_columns: pd.DataFrame,
    ore_feature_cols: List[str],
    config: Dict,
) -> pd.DataFrame:
    def _columns_total_cap(status: str) -> float:
        st = normalize_status(status)
        if st == "Catalyzed":
            return float(config.get("columns_fit_total_recovery_upper_catalyzed", config["fit_total_recovery_upper"]))
        return float(config.get("columns_fit_total_recovery_upper_control", config["fit_total_recovery_upper"]))

    def _raise_dtype_error(sample_id: str, status: str, col_name: str, raw_value, parsed: np.ndarray) -> None:
        raw_preview = repr(raw_value)
        if len(raw_preview) > 220:
            raw_preview = raw_preview[:220] + "..."
        raise TypeError(
            f"[CurveFit dtype error] sample={sample_id} status={status} column={col_name} "
            f"cannot be parsed as numeric list. raw={raw_preview}; parsed_len={len(parsed)}"
        )

    fit_min_points = int(config["fit_min_points"])
    shaping_cfg_fit = dict(CONFIG.get("inference_shaping", {}))
    coupling_cfg_fit = dict(CONFIG.get("catalyst_curve_coupling", {}))
    rows = []

    def _fit_params_for_row(
        t_days: np.ndarray,
        recovery: np.ndarray,
        status: str,
        row: pd.Series,
        catalyst_profile: np.ndarray = None,
        control_params_for_catalyzed: np.ndarray = None,
    ) -> np.ndarray:
        total_upper = _columns_total_cap(status)
        params = _fit_double_exponential_params(
            t_days=t_days,
            recovery=recovery,
            fit_min_points=fit_min_points,
            fit_maxfev=config["fit_maxfev"],
            fit_b_fast_upper=config["fit_b_fast_upper"],
            fit_b_min=config.get("fit_b_min", 1e-5),
            fit_a_min=config.get("fit_a_min", 0.0),
            fit_enforce_fast_slow_constraint=config.get("fit_enforce_fast_slow_constraint", False),
            fit_reject_if_stuck_on_first_seed=config.get("fit_reject_if_stuck_on_first_seed", False),
            fit_total_recovery_upper=total_upper,
            catalyst_profile=catalyst_profile,
            control_params_for_catalyzed=control_params_for_catalyzed,
            transition_time=pd.to_numeric(row.get("transition_time", np.nan), errors="coerce"),
            material_size_p80_in=pd.to_numeric(row.get("material_size_p80_in", np.nan), errors="coerce"),
            column_height_m=pd.to_numeric(row.get("column_height_m", np.nan), errors="coerce"),
            column_inner_diameter_m=pd.to_numeric(row.get("column_inner_diameter_m", np.nan), errors="coerce"),
            residual_cpy_pct=pd.to_numeric(row.get("residual_cpy_%", np.nan), errors="coerce"),
            row_context=row,
            shaping_cfg=shaping_cfg_fit,
            coupling_cfg=coupling_cfg_fit,
            tmax_days=config.get("fit_tmax_days", None),
            frac=config.get("fit_tmax_frac", 0.99),
            epsilon=config.get("fit_tmax_epsilon", 1e-4),
            last_n=config.get("fit_last_n", 5),
            last_points_weight=config.get("fit_last_points_weight", (2.0, 10.0)),
            jitter_starts=config.get("fit_jitter_starts", 0),
            jitter_scale=config.get("fit_jitter_scale", 0.3),
            random_state=config.get("fit_random_state", None),
            tail_slope_threshold=config.get("fit_tail_slope_threshold", 0.05),
            tail_mean_min_for_tmax_penalty=config.get("fit_tail_mean_min_for_tmax_penalty", 80.0),
            tmax_penalty_lambda=config.get("fit_tmax_penalty_lambda", 10.0),
        )
        if not np.all(np.isfinite(params)):
            mode = "catalyst-aware" if control_params_for_catalyzed is not None and catalyst_profile is not None else "direct bi-exponential"
            raise RuntimeError(
                f"[CurveFit fail] sample={row.get('project_sample_id')} status={status} "
                f"optimizer did not converge ({mode}). Adjust fitting constraints/config."
            )
        return _sanitize_curve_params(
            *params,
            total_recovery_upper=total_upper,
            b_upper=config["fit_b_fast_upper"],
        )

    if "project_sample_id" in df_columns.columns:
        grouped = df_columns.groupby("project_sample_id", sort=False, dropna=False)
    else:
        grouped = [("unknown_sample_id", df_columns)]

    for sample_id, group in grouped:
        grp = group.copy()
        grp["_status_norm"] = grp[CATALYZED_COLUMNS_ID].apply(normalize_status) if CATALYZED_COLUMNS_ID in grp.columns else "Control"
        control_rows = grp[grp["_status_norm"] == "Control"]
        catalyzed_rows = grp[grp["_status_norm"] == "Catalyzed"]

        control_params_ref = None

        for _, row in control_rows.iterrows():
            status = "Control"
            raw_t = row.get(TIME_COL_COLUMNS, None)
            raw_y = row.get(TARGET_COLUMNS, None)
            t = parse_listlike(raw_t)
            y = parse_listlike(raw_y)
            if len(t) == 0:
                _raise_dtype_error(sample_id, status, TIME_COL_COLUMNS, raw_t, t)
            if len(y) == 0:
                _raise_dtype_error(sample_id, status, TARGET_COLUMNS, raw_y, y)
            n = min(len(t), len(y))
            if n < fit_min_points:
                raise RuntimeError(
                    f"[CurveFit fail] sample={sample_id} status={status} has n_points={n} "
                    f"< fit_min_points={fit_min_points}. Adjust constraints/fit settings."
                )
            t = np.asarray(t[:n], dtype=float)
            y = np.asarray(y[:n], dtype=float)
            valid = np.isfinite(t) & np.isfinite(y)
            if np.isfinite(t).sum() == 0:
                _raise_dtype_error(sample_id, status, TIME_COL_COLUMNS, raw_t, t)
            if np.isfinite(y).sum() == 0:
                _raise_dtype_error(sample_id, status, TARGET_COLUMNS, raw_y, y)
            if int(valid.sum()) < fit_min_points:
                raise RuntimeError(
                    f"[CurveFit fail] sample={sample_id} status={status} has only {int(valid.sum())} "
                    f"valid numeric points after coercion (< {fit_min_points})."
                )

            catalyst = np.zeros(n, dtype=float)
            params = _fit_params_for_row(
                t_days=t,
                recovery=y,
                status=status,
                row=row,
                catalyst_profile=None,
                control_params_for_catalyzed=None,
            )
            params_direct = params.copy()
            control_params_ref = params.copy()

            rec = {
                "project_sample_id": row.get("project_sample_id"),
                CATALYZED_COLUMNS_ID: status,
                "is_catalyzed": 0.0,
                "curve_time_days": t.tolist(),
                "curve_catalyst_profile": catalyst.tolist(),
                "curve_target_recovery": y.tolist(),
                "curve_max_time_days": float(np.nanmax(t)) if len(t) else np.nan,
                "target_a1": float(params[0]),
                "target_b1": float(params[1]),
                "target_a2": float(params[2]),
                "target_b2": float(params[3]),
                "target_direct_a1": float(params_direct[0]),
                "target_direct_b1": float(params_direct[1]),
                "target_direct_a2": float(params_direct[2]),
                "target_direct_b2": float(params_direct[3]),
            }
            rec.update(_summarize_catalyst_profile(t, catalyst, is_catalyzed=False))
            for feat in ore_feature_cols:
                rec[feat] = row.get(feat, np.nan)
            rows.append(rec)

        for _, row in catalyzed_rows.iterrows():
            status = "Catalyzed"
            if control_params_ref is None:
                raise RuntimeError(
                    f"[CurveFit fail] sample={sample_id} has Catalyzed row but no valid Control row. "
                    "Catalyst-aware fitting requires paired control curve."
                )

            raw_t = row.get(TIME_COL_COLUMNS, None)
            raw_y = row.get(TARGET_COLUMNS, None)
            raw_c = row.get("cumulative_catalyst_addition_kg_t", None)
            t = parse_listlike(raw_t)
            y = parse_listlike(raw_y)
            c = parse_listlike(raw_c)
            if len(t) == 0:
                _raise_dtype_error(sample_id, status, TIME_COL_COLUMNS, raw_t, t)
            if len(y) == 0:
                _raise_dtype_error(sample_id, status, TARGET_COLUMNS, raw_y, y)
            if len(c) == 0:
                _raise_dtype_error(sample_id, status, "cumulative_catalyst_addition_kg_t", raw_c, c)

            n = min(len(t), len(y))
            if n < fit_min_points:
                raise RuntimeError(
                    f"[CurveFit fail] sample={sample_id} status={status} has n_points={n} "
                    f"< fit_min_points={fit_min_points}. Adjust constraints/fit settings."
                )
            t = np.asarray(t[:n], dtype=float)
            y = np.asarray(y[:n], dtype=float)
            if len(c) >= n:
                catalyst = np.asarray(c[:n], dtype=float)
            else:
                catalyst = np.concatenate([np.asarray(c, dtype=float), np.repeat(float(c[-1]), n - len(c))])
            catalyst = np.where(np.isfinite(catalyst) & (catalyst > 0), catalyst, 0.0)
            catalyst = np.maximum.accumulate(catalyst)

            valid = np.isfinite(t) & np.isfinite(y)
            if np.isfinite(t).sum() == 0:
                _raise_dtype_error(sample_id, status, TIME_COL_COLUMNS, raw_t, t)
            if np.isfinite(y).sum() == 0:
                _raise_dtype_error(sample_id, status, TARGET_COLUMNS, raw_y, y)
            if int(valid.sum()) < fit_min_points:
                raise RuntimeError(
                    f"[CurveFit fail] sample={sample_id} status={status} has only {int(valid.sum())} "
                    f"valid numeric points after coercion (< {fit_min_points})."
                )

            params = _fit_params_for_row(
                t_days=t,
                recovery=y,
                status=status,
                row=row,
                catalyst_profile=catalyst,
                control_params_for_catalyzed=control_params_ref,
            )
            params_direct = _fit_params_for_row(
                t_days=t,
                recovery=y,
                status=status,
                row=row,
                catalyst_profile=None,
                control_params_for_catalyzed=None,
            )

            rec = {
                "project_sample_id": row.get("project_sample_id"),
                CATALYZED_COLUMNS_ID: status,
                "is_catalyzed": 1.0,
                "curve_time_days": t.tolist(),
                "curve_catalyst_profile": catalyst.tolist(),
                "curve_target_recovery": y.tolist(),
                "curve_max_time_days": float(np.nanmax(t)) if len(t) else np.nan,
                "target_a1": float(params[0]),
                "target_b1": float(params[1]),
                "target_a2": float(params[2]),
                "target_b2": float(params[3]),
                "target_direct_a1": float(params_direct[0]),
                "target_direct_b1": float(params_direct[1]),
                "target_direct_a2": float(params_direct[2]),
                "target_direct_b2": float(params_direct[3]),
            }
            rec.update(_summarize_catalyst_profile(t, catalyst, is_catalyzed=True))
            for feat in ore_feature_cols:
                rec[feat] = row.get(feat, np.nan)
            rows.append(rec)

    return pd.DataFrame(rows)


def _select_ore_similarity_features(
    df_columns: pd.DataFrame,
    df_reactors: pd.DataFrame,
    force_include_shared: bool = True,
) -> List[str]:
    selected = []
    shared = []
    for col in ORE_SIMILARITY_CANDIDATES:
        if col not in df_columns.columns or col not in df_reactors.columns:
            continue
        shared.append(col)
        col_vals = pd.to_numeric(df_columns[col], errors="coerce")
        react_vals = pd.to_numeric(df_reactors[col], errors="coerce")
        if col_vals.notna().sum() < 5 or react_vals.notna().sum() < 20:
            continue
        selected.append(col)
    if force_include_shared:
        selected = list(dict.fromkeys(selected + shared))
    return selected


def _validate_and_impute_transfer_inputs(
    df_columns_params: pd.DataFrame,
    required_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    out = df_columns_params.copy()
    missing_cols = [c for c in required_cols if c not in out.columns]
    if missing_cols:
        raise ValueError(f"Missing required transfer input columns: {missing_cols}")

    impute_report: Dict[str, Dict[str, float]] = {}
    for col in required_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        n_missing = int(out[col].isna().sum())
        if n_missing <= 0:
            continue
        med = pd.to_numeric(out[col], errors="coerce").median()
        fill_value = float(med) if np.isfinite(med) else 0.0
        out[col] = out[col].fillna(fill_value)
        impute_report[col] = {"n_filled": n_missing, "fill_value": fill_value}
    return out, impute_report


def _build_reactor_similarity_param_priors(
    df_columns_params: pd.DataFrame,
    df_reactors: pd.DataFrame,
    similarity_features: List[str],
    k_neighbors: int,
    eps: float,
    fit_total_recovery_upper: float,
    fit_b_fast_upper: float,
) -> pd.DataFrame:
    out = df_columns_params.copy()
    prior_cols = [
        "reactor_sim_a1",
        "reactor_sim_b1",
        "reactor_sim_a2",
        "reactor_sim_b2",
        "reactor_sim_mean_distance",
    ]
    for col in prior_cols:
        out[col] = np.nan
    if not similarity_features:
        return out

    reactors = df_reactors.copy()
    reactors["status_norm"] = reactors[CATALYZED_REACTORS_ID].apply(normalize_status)
    for col in similarity_features + ["a1_param", "b1_param", "a2_param", "b2_param"]:
        reactors[col] = pd.to_numeric(reactors[col], errors="coerce")
    reactors = reactors.dropna(subset=["a1_param", "b1_param", "a2_param", "b2_param"]).copy()
    if reactors.empty:
        return out

    react_params = reactors[["a1_param", "b1_param", "a2_param", "b2_param"]].values.astype(float)
    react_params_clean = np.vstack(
        [
            _sanitize_curve_params(
                p[0],
                p[1],
                p[2],
                p[3],
                total_recovery_upper=fit_total_recovery_upper,
                b_upper=fit_b_fast_upper,
            )
            for p in react_params
        ]
    )
    reactors[["a1_param", "b1_param", "a2_param", "b2_param"]] = react_params_clean

    for status in ["Control", "Catalyzed"]:
        react_status = reactors[reactors["status_norm"] == status].copy()
        cols_status = out[out[CATALYZED_COLUMNS_ID].apply(normalize_status) == status].copy()
        if react_status.empty or cols_status.empty:
            continue

        reactor_feat = react_status[similarity_features].apply(pd.to_numeric, errors="coerce")
        sample_feat = cols_status[similarity_features].apply(pd.to_numeric, errors="coerce")

        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_react = scaler.fit_transform(imp.fit_transform(reactor_feat))
        X_cols = scaler.transform(imp.transform(sample_feat))

        Y_react = react_status[["a1_param", "b1_param", "a2_param", "b2_param"]].values.astype(float)
        k = min(k_neighbors, len(react_status))

        for i, idx in enumerate(cols_status.index):
            dists = np.linalg.norm(X_react - X_cols[i], axis=1)
            nn_idx = np.argsort(dists)[:k]
            nn_dist = dists[nn_idx]
            weights = 1.0 / (nn_dist + eps)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            prior = np.sum(Y_react[nn_idx] * weights[:, None], axis=0)
            prior = _sanitize_curve_params(
                prior[0],
                prior[1],
                prior[2],
                prior[3],
                total_recovery_upper=fit_total_recovery_upper,
                b_upper=fit_b_fast_upper,
            )
            out.loc[idx, "reactor_sim_a1"] = float(prior[0])
            out.loc[idx, "reactor_sim_b1"] = float(prior[1])
            out.loc[idx, "reactor_sim_a2"] = float(prior[2])
            out.loc[idx, "reactor_sim_b2"] = float(prior[3])
            out.loc[idx, "reactor_sim_mean_distance"] = float(np.mean(nn_dist))
    return out


def _add_tuple_relation_features(
    df: pd.DataFrame,
    prefix: str,
    a1_col: str,
    b1_col: str,
    a2_col: str,
    b2_col: str,
    eps: float = 1e-6,
) -> pd.DataFrame:
    out = df.copy()
    relation_cols = [
        f"{prefix}_a_total",
        f"{prefix}_a_fast_frac",
        f"{prefix}_b_ratio",
        f"{prefix}_tau1_days",
        f"{prefix}_tau2_days",
        f"{prefix}_b_gap_log",
    ]
    for col in relation_cols:
        if col not in out.columns:
            out[col] = np.nan

    base_cols = [a1_col, b1_col, a2_col, b2_col]
    if any(c not in out.columns for c in base_cols):
        return out

    a1 = pd.to_numeric(out[a1_col], errors="coerce").to_numpy(dtype=float)
    b1 = pd.to_numeric(out[b1_col], errors="coerce").to_numpy(dtype=float)
    a2 = pd.to_numeric(out[a2_col], errors="coerce").to_numpy(dtype=float)
    b2 = pd.to_numeric(out[b2_col], errors="coerce").to_numpy(dtype=float)

    total = a1 + a2
    total_den = np.where(np.isfinite(total), np.maximum(np.abs(total), eps), np.nan)
    b1_pos = np.where(np.isfinite(b1), np.maximum(b1, eps), np.nan)
    b2_pos = np.where(np.isfinite(b2), np.maximum(b2, eps), np.nan)

    out[f"{prefix}_a_total"] = total
    out[f"{prefix}_a_fast_frac"] = np.where(np.isfinite(a1) & np.isfinite(total_den), a1 / total_den, np.nan)
    out[f"{prefix}_b_ratio"] = np.where(np.isfinite(b1_pos) & np.isfinite(b2_pos), b1_pos / b2_pos, np.nan)
    out[f"{prefix}_tau1_days"] = np.where(np.isfinite(b1_pos), 1.0 / b1_pos, np.nan)
    out[f"{prefix}_tau2_days"] = np.where(np.isfinite(b2_pos), 1.0 / b2_pos, np.nan)
    out[f"{prefix}_b_gap_log"] = np.where(
        np.isfinite(b1_pos) & np.isfinite(b2_pos),
        np.log(b1_pos) - np.log(b2_pos),
        np.nan,
    )
    return out


def _build_reactor_tuple_transfer_features(
    df_columns_params: pd.DataFrame,
    df_reactors: pd.DataFrame,
    similarity_features: List[str],
    tuple_cfg: Dict,
    seeds: List[int],
    fit_total_recovery_upper: float,
    fit_b_fast_upper: float,
) -> pd.DataFrame:
    out = df_columns_params.copy()
    for col in REACTOR_TUPLE_TRANSFER_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    if not similarity_features:
        return out

    reactors = df_reactors.copy()
    reactors["status_norm"] = reactors[CATALYZED_REACTORS_ID].apply(normalize_status)
    for col in similarity_features + ["a1_param", "b1_param", "a2_param", "b2_param"]:
        reactors[col] = pd.to_numeric(reactors[col], errors="coerce")
    reactors = reactors.dropna(subset=["a1_param", "b1_param", "a2_param", "b2_param"]).copy()
    if reactors.empty:
        return out

    react_params = reactors[["a1_param", "b1_param", "a2_param", "b2_param"]].values.astype(float)
    react_params_clean = np.vstack(
        [
            _sanitize_curve_params(
                p[0],
                p[1],
                p[2],
                p[3],
                total_recovery_upper=fit_total_recovery_upper,
                b_upper=fit_b_fast_upper,
            )
            for p in react_params
        ]
    )
    reactors[["a1_param", "b1_param", "a2_param", "b2_param"]] = react_params_clean

    cfg = dict(tuple_cfg or {})
    seed_list = [int(s) for s in seeds] if seeds else [1]
    rf_n_estimators = int(max(50, cfg.get("rf_n_estimators", 500)))
    rf_min_samples_leaf = int(max(1, cfg.get("rf_min_samples_leaf", 2)))
    min_rows = int(max(4, cfg.get("min_rows_per_status", 12)))
    k_neighbors = int(max(1, cfg.get("k_neighbors", 7)))
    eps = float(max(1e-9, cfg.get("distance_eps", 1e-6)))

    for status in ["Control", "Catalyzed"]:
        react_status = reactors[reactors["status_norm"] == status].copy()
        col_mask = out[CATALYZED_COLUMNS_ID].apply(normalize_status) == status
        if react_status.empty or not col_mask.any():
            continue

        cols_status = out.loc[col_mask].copy()
        reactor_feat = react_status[similarity_features].apply(pd.to_numeric, errors="coerce")
        sample_feat = cols_status[similarity_features].apply(pd.to_numeric, errors="coerce")

        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_react = scaler.fit_transform(imp.fit_transform(reactor_feat))
        X_cols = scaler.transform(imp.transform(sample_feat))
        Y_react = react_status[["a1_param", "b1_param", "a2_param", "b2_param"]].values.astype(float)

        k = min(k_neighbors, len(react_status))
        knn_preds = np.full((len(cols_status), 4), np.nan, dtype=float)
        mean_dist = np.full(len(cols_status), np.nan, dtype=float)
        for i in range(len(cols_status)):
            dists = np.linalg.norm(X_react - X_cols[i], axis=1)
            nn_idx = np.argsort(dists)[:k]
            nn_dist = dists[nn_idx]
            weights = 1.0 / (nn_dist + eps)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            pred_knn = np.sum(Y_react[nn_idx] * weights[:, None], axis=0)
            pred_knn = _sanitize_curve_params(
                pred_knn[0],
                pred_knn[1],
                pred_knn[2],
                pred_knn[3],
                total_recovery_upper=fit_total_recovery_upper,
                b_upper=fit_b_fast_upper,
            )
            knn_preds[i, :] = pred_knn
            mean_dist[i] = float(np.mean(nn_dist))

        pred_params = knn_preds.copy()
        if len(react_status) >= min_rows:
            rf_pred = np.full_like(pred_params, np.nan, dtype=float)
            for t_idx in range(4):
                models = _fit_rf_ensemble(
                    X_train=X_react,
                    y_train=Y_react[:, t_idx],
                    seeds=seed_list,
                    n_estimators=rf_n_estimators,
                    min_samples_leaf=rf_min_samples_leaf,
                )
                member_pred = _predict_rf_ensemble(models, X_cols)
                rf_pred[:, t_idx] = np.mean(member_pred, axis=0)
            pred_params = np.vstack(
                [
                    _sanitize_curve_params(
                        p[0],
                        p[1],
                        p[2],
                        p[3],
                        total_recovery_upper=fit_total_recovery_upper,
                        b_upper=fit_b_fast_upper,
                    )
                    for p in rf_pred
                ]
            )

        for i, idx in enumerate(cols_status.index):
            out.loc[idx, "reactor_tuple_rf_a1"] = float(pred_params[i, 0])
            out.loc[idx, "reactor_tuple_rf_b1"] = float(pred_params[i, 1])
            out.loc[idx, "reactor_tuple_rf_a2"] = float(pred_params[i, 2])
            out.loc[idx, "reactor_tuple_rf_b2"] = float(pred_params[i, 3])
            out.loc[idx, "reactor_tuple_rf_mean_distance"] = float(mean_dist[i])

    out = _add_tuple_relation_features(
        out,
        prefix="reactor_tuple_rf",
        a1_col="reactor_tuple_rf_a1",
        b1_col="reactor_tuple_rf_b1",
        a2_col="reactor_tuple_rf_a2",
        b2_col="reactor_tuple_rf_b2",
    )
    return out


def _build_reactor_normalized_uplift_library(
    df_reactors: pd.DataFrame,
    similarity_features: List[str],
    fit_total_recovery_upper: float,
    time_max: float,
    time_norm_mode: str,
    control_floor: float,
    uplift_clip_low: float,
    uplift_clip_high: float,
) -> List[Dict]:
    library = []
    if "project_sample_id" not in df_reactors.columns:
        return library

    reactors = df_reactors.copy()
    reactors["status_norm"] = reactors[CATALYZED_REACTORS_ID].apply(normalize_status)

    for sample_id, group in reactors.groupby("project_sample_id"):
        control_rows = group[group["status_norm"] == "Control"]
        catalyzed_rows = group[group["status_norm"] == "Catalyzed"]
        if control_rows.empty or catalyzed_rows.empty:
            continue
        control_row = control_rows.iloc[0]
        catalyzed_row = catalyzed_rows.iloc[0]

        t_ctrl = parse_listlike(control_row.get(TIME_COL_REACTORS, None))
        y_ctrl = parse_listlike(control_row.get(TARGET_REACTORS, None))
        t_cat = parse_listlike(catalyzed_row.get(TIME_COL_REACTORS, None))
        y_cat = parse_listlike(catalyzed_row.get(TARGET_REACTORS, None))
        if t_ctrl.size == 0 or t_cat.size == 0:
            continue

        grid = np.unique(np.concatenate([t_ctrl, t_cat]))
        if grid.size < 2:
            continue
        ctrl_interp = np.array([_interp_at(t_ctrl, y_ctrl, t) for t in grid], dtype=float)
        cat_interp = np.array([_interp_at(t_cat, y_cat, t) for t in grid], dtype=float)

        denom = np.maximum(ctrl_interp, max(1e-6, control_floor))
        uplift_pct = ((cat_interp - ctrl_interp) / denom) * 100.0
        uplift_pct = np.clip(uplift_pct, uplift_clip_low, uplift_clip_high)

        grid_norm = np.array([_normalize_time(float(t), time_max, time_norm_mode) for t in grid], dtype=float)
        valid = np.isfinite(grid_norm) & np.isfinite(uplift_pct)
        if valid.sum() < 2:
            continue
        grid_norm = grid_norm[valid]
        uplift_pct = uplift_pct[valid]
        order = np.argsort(grid_norm)
        grid_norm = grid_norm[order]
        uplift_pct = uplift_pct[order]

        feats = {}
        for feat in similarity_features:
            feats[feat] = pd.to_numeric(control_row.get(feat, np.nan), errors="coerce")

        library.append(
            {
                "project_sample_id": sample_id,
                "time_norm": grid_norm,
                "uplift_pct": uplift_pct,
                "features": feats,
            }
        )
    return library


def _apply_reactor_normalized_uplift_prior(
    df_columns_params: pd.DataFrame,
    uplift_library: List[Dict],
    similarity_features: List[str],
    k_neighbors: int,
    eps: float,
    time_max: float,
    time_norm_mode: str,
) -> pd.Series:
    if not uplift_library or not similarity_features:
        return pd.Series(np.zeros(len(df_columns_params), dtype=float), index=df_columns_params.index)

    lib_feat_df = pd.DataFrame([lib["features"] for lib in uplift_library])[similarity_features]
    lib_feat_df = lib_feat_df.apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_lib = scaler.fit_transform(imp.fit_transform(lib_feat_df))

    sample_feat = df_columns_params[similarity_features].apply(pd.to_numeric, errors="coerce")
    X_samples = scaler.transform(imp.transform(sample_feat))
    k = min(k_neighbors, len(uplift_library))

    priors = []
    for i, (_, row) in enumerate(df_columns_params.iterrows()):
        if normalize_status(row.get(CATALYZED_COLUMNS_ID, "Control")) != "Catalyzed":
            priors.append(0.0)
            continue

        t_ref = pd.to_numeric(row.get("curve_max_time_days", np.nan), errors="coerce")
        if not np.isfinite(t_ref):
            t_arr = np.asarray(row.get("curve_time_days", []), dtype=float)
            t_ref = float(np.nanmax(t_arr)) if t_arr.size else np.nan
        t_norm = _normalize_time(float(t_ref), time_max, time_norm_mode)

        dists = np.linalg.norm(X_lib - X_samples[i], axis=1)
        nn_idx = np.argsort(dists)[:k]
        nn_dist = dists[nn_idx]
        w = 1.0 / (nn_dist + eps)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)

        vals = []
        for j in nn_idx:
            lib = uplift_library[j]
            lib_time = lib["time_norm"]
            if lib_time.size == 0 or not np.isfinite(t_norm):
                vals.append(np.nan)
                continue
            t_query = min(max(t_norm, lib_time[0]), lib_time[-1])
            vals.append(_interp_at(lib_time, lib["uplift_pct"], t_query))
        vals = np.asarray(vals, dtype=float)
        valid = np.isfinite(vals)
        if not valid.any():
            priors.append(0.0)
            continue
        wv = w[valid]
        wv = wv / wv.sum() if wv.sum() > 0 else np.ones_like(wv) / len(wv)
        priors.append(float(np.sum(vals[valid] * wv)))

    return pd.Series(priors, index=df_columns_params.index)


def _select_column_model_features(
    df_columns_params: pd.DataFrame,
    ore_similarity_features: List[str],
    use_reactor_similarity_params: bool,
) -> List[str]:
    feature_cols = []

    for col in ore_similarity_features:
        if col in df_columns_params.columns:
            feature_cols.append(col)
    for col in COLUMN_EXTRA_FEATURE_CANDIDATES:
        if col in df_columns_params.columns:
            feature_cols.append(col)

    catalyst_feature_cols = [
        "is_catalyzed",
        "curve_max_time_days",
        "catalyst_final_kg_t",
        "catalyst_max_kg_t",
        "catalyst_auc_kg_t_day",
        "catalyst_rate_kg_t_day",
    ]
    for col in catalyst_feature_cols:
        if col in df_columns_params.columns:
            feature_cols.append(col)

    if use_reactor_similarity_params:
        for col in [
            "reactor_sim_a1",
            "reactor_sim_b1",
            "reactor_sim_a2",
            "reactor_sim_b2",
            "reactor_sim_mean_distance",
        ]:
            if col in df_columns_params.columns:
                feature_cols.append(col)
        for col in REACTOR_SIM_RELATION_FEATURES:
            if col in df_columns_params.columns:
                feature_cols.append(col)

    for col in REACTOR_TUPLE_TRANSFER_FEATURES:
        if col in df_columns_params.columns:
            feature_cols.append(col)

    if "reactor_norm_uplift_prior_pct" in df_columns_params.columns:
        feature_cols.append("reactor_norm_uplift_prior_pct")

    # Preserve order, remove duplicates.
    return list(dict.fromkeys(feature_cols))


def _fit_rf_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seeds: List[int],
    n_estimators: int,
    min_samples_leaf: int,
) -> List[RandomForestRegressor]:
    models = []
    for seed in seeds:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        models.append(model)
    return models


def _predict_rf_ensemble(models: List[RandomForestRegressor], X_test: np.ndarray) -> np.ndarray:
    member_preds = [m.predict(X_test) for m in models]
    return np.stack(member_preds, axis=0)


def _align_profile_to_time_length(profile: np.ndarray, time_length: int) -> np.ndarray:
    p = np.asarray(profile, dtype=float)
    if time_length <= 0:
        return np.asarray([], dtype=float)
    if p.size == 0:
        return np.zeros(time_length, dtype=float)
    if p.size >= time_length:
        return p[:time_length]
    pad_val = p[-1] if np.isfinite(p[-1]) else 0.0
    return np.concatenate([p, np.repeat(pad_val, time_length - p.size)])


def _prepare_inference_shaping_config(df_columns_params: pd.DataFrame, shaping_cfg: Dict) -> Dict:
    cfg = dict(shaping_cfg or {})
    if cfg.get("p80_reference_in", None) is None:
        if "material_size_p80_in" in df_columns_params.columns:
            p80 = pd.to_numeric(df_columns_params["material_size_p80_in"], errors="coerce")
            if p80.notna().any():
                cfg["p80_reference_in"] = float(p80.median())
            else:
                cfg["p80_reference_in"] = 0.25
        else:
            cfg["p80_reference_in"] = 0.25

    if cfg.get("p80_lag_reference_in", None) is None:
        cfg["p80_lag_reference_in"] = cfg.get("p80_reference_in", 0.25)

    if cfg.get("geometry_lag_reference_height_m", None) is None:
        if "column_height_m" in df_columns_params.columns:
            h = pd.to_numeric(df_columns_params["column_height_m"], errors="coerce")
            cfg["geometry_lag_reference_height_m"] = float(h.median()) if h.notna().any() else 2.5
        else:
            cfg["geometry_lag_reference_height_m"] = 2.5

    if cfg.get("geometry_lag_reference_diameter_m", None) is None:
        if "column_inner_diameter_m" in df_columns_params.columns:
            d = pd.to_numeric(df_columns_params["column_inner_diameter_m"], errors="coerce")
            cfg["geometry_lag_reference_diameter_m"] = float(d.median()) if d.notna().any() else 0.20
        else:
            cfg["geometry_lag_reference_diameter_m"] = 0.20

    if cfg.get("residual_cpy_reference_pct", None) is None:
        if "residual_cpy_%" in df_columns_params.columns:
            rcpy = pd.to_numeric(df_columns_params["residual_cpy_%"], errors="coerce")
            cfg["residual_cpy_reference_pct"] = float(rcpy.median()) if rcpy.notna().any() else 70.0
        else:
            cfg["residual_cpy_reference_pct"] = 70.0

    if cfg.get("catalyst_reference_kg_t", None) is None:
        w_final = float(cfg.get("catalyst_signal_weight_final", 0.6))
        w_avg = float(cfg.get("catalyst_signal_weight_avg", 0.4))
        c_ref = np.nan
        if all(c in df_columns_params.columns for c in ["catalyst_final_kg_t", "catalyst_auc_kg_t_day", "curve_max_time_days"]):
            sub = df_columns_params.copy()
            if CATALYZED_COLUMNS_ID in sub.columns:
                sub = sub[sub[CATALYZED_COLUMNS_ID].apply(normalize_status) == "Catalyzed"]
            c_final = pd.to_numeric(sub["catalyst_final_kg_t"], errors="coerce")
            c_auc = pd.to_numeric(sub["catalyst_auc_kg_t_day"], errors="coerce")
            t_max = pd.to_numeric(sub["curve_max_time_days"], errors="coerce")
            c_avg = np.where(np.isfinite(c_auc) & np.isfinite(t_max) & (t_max > 0), c_auc / t_max, np.nan)
            c_signal = w_final * c_final.to_numpy(dtype=float) + w_avg * np.asarray(c_avg, dtype=float)
            valid = np.isfinite(c_signal)
            if valid.any():
                c_ref = float(np.nanmedian(c_signal[valid]))
        if not np.isfinite(c_ref):
            c_ref = 0.01
        cfg["catalyst_reference_kg_t"] = float(max(0.0, c_ref))

    # Optional kinetics realism caps for transferred column b-parameters.
    if bool(cfg.get("use_column_kinetics_caps", True)):
        q = float(np.clip(cfg.get("b1_cap_quantile", 0.90), 0.5, 0.995))
        margin = float(max(1.0, cfg.get("b1_cap_margin", 1.10)))
        all_b1_vals = np.asarray([], dtype=float)
        if "target_b1" in df_columns_params.columns:
            all_b1 = pd.to_numeric(df_columns_params["target_b1"], errors="coerce")
            all_b1_vals = all_b1.to_numpy(dtype=float)
        all_b1_vals = all_b1_vals[np.isfinite(all_b1_vals) & (all_b1_vals > 0)]

        fit_cap_min = np.nan
        fit_cap_max = np.nan
        if bool(cfg.get("b1_caps_from_fitted_params", True)) and all_b1_vals.size > 0:
            q_min = float(np.clip(cfg.get("b1_cap_min_quantile", 0.05), 0.001, 0.499))
            q_max = float(np.clip(cfg.get("b1_cap_max_quantile", 0.98), 0.501, 0.999))
            if q_max <= q_min:
                q_max = min(0.999, q_min + 0.10)
            m_min = float(np.clip(cfg.get("b1_cap_min_margin", 0.85), 0.1, 1.5))
            m_max = float(max(1.0, cfg.get("b1_cap_max_margin", 1.10)))
            abs_floor = float(max(1e-6, cfg.get("b1_cap_abs_floor", 1e-4)))
            fit_cap_min = float(max(abs_floor, np.nanquantile(all_b1_vals, q_min) * m_min))
            fit_cap_max = float(max(fit_cap_min, np.nanquantile(all_b1_vals, q_max) * m_max))

        raw_cap_min = pd.to_numeric(cfg.get("b1_cap_min", np.nan), errors="coerce")
        raw_cap_max = pd.to_numeric(cfg.get("b1_cap_max", np.nan), errors="coerce")
        cap_min = float(max(1e-5, raw_cap_min)) if np.isfinite(raw_cap_min) else (float(fit_cap_min) if np.isfinite(fit_cap_min) else 0.005)
        cap_max = float(max(cap_min, raw_cap_max)) if np.isfinite(raw_cap_max) else (float(fit_cap_max) if np.isfinite(fit_cap_max) else 0.45)
        if np.isfinite(fit_cap_min):
            cap_min = max(cap_min, float(fit_cap_min))
        if np.isfinite(fit_cap_max):
            cap_min = min(cap_min, float(fit_cap_max))
            cap_max = min(cap_max, float(fit_cap_max))
        cap_max = max(cap_min, cap_max)
        cfg["b1_cap_min"] = float(cap_min)
        cfg["b1_cap_max"] = float(cap_max)

        def _status_cap(status_name: str) -> float:
            vals = np.asarray([], dtype=float)
            if "target_b1" in df_columns_params.columns:
                b = pd.to_numeric(df_columns_params["target_b1"], errors="coerce")
                if CATALYZED_COLUMNS_ID in df_columns_params.columns:
                    s = df_columns_params[CATALYZED_COLUMNS_ID].apply(normalize_status)
                    mask = (s == status_name) & b.notna()
                    vals = b.loc[mask].to_numpy(dtype=float)
                else:
                    vals = b.dropna().to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return np.nan
            return float(np.nanquantile(vals, q) * margin)

        raw_ctrl = pd.to_numeric(cfg.get("b1_cap_control", np.nan), errors="coerce")
        if np.isfinite(raw_ctrl):
            b1_cap_control = float(raw_ctrl)
        else:
            b1_cap_control = _status_cap("Control")
            if not np.isfinite(b1_cap_control):
                b1_cap_control = _status_cap("Catalyzed")
            if not np.isfinite(b1_cap_control):
                b1_cap_control = 0.20
        cfg["b1_cap_control"] = float(np.clip(b1_cap_control, cap_min, cap_max))

        raw_cat = pd.to_numeric(cfg.get("b1_cap_catalyzed", np.nan), errors="coerce")
        if np.isfinite(raw_cat):
            b1_cap_cat = float(raw_cat)
        else:
            b1_cap_cat = _status_cap("Catalyzed")
            if not np.isfinite(b1_cap_cat):
                b1_cap_cat = _status_cap("Control")
            if not np.isfinite(b1_cap_cat):
                b1_cap_cat = cfg.get("b1_cap_control", 0.20)
        cfg["b1_cap_catalyzed"] = float(np.clip(b1_cap_cat, cap_min, cap_max))

        # Keep catalyzed kinetics cap realistic vs control to avoid reactor-fast transfer bleed-through.
        cat_to_ctrl_max = pd.to_numeric(
            cfg.get("b1_cap_catalyzed_to_control_max_multiplier", np.nan),
            errors="coerce",
        )
        if np.isfinite(cat_to_ctrl_max) and cat_to_ctrl_max > 0:
            ctrl_cap = float(cfg["b1_cap_control"])
            cat_cap = float(cfg["b1_cap_catalyzed"])
            capped_cat = min(cat_cap, ctrl_cap * float(cat_to_ctrl_max))
            cfg["b1_cap_catalyzed"] = float(np.clip(capped_cat, cap_min, cap_max))

    # Context offsets: ore + p80 + geometry modifiers for catalyst effect.
    if bool(cfg.get("use_catalyst_context_offsets", True)):
        raw_feats = cfg.get("context_ore_features", None)
        if isinstance(raw_feats, (list, tuple)):
            context_ore_feats = [str(c) for c in raw_feats if str(c) in df_columns_params.columns]
        else:
            context_ore_feats = [c for c in ORE_SIMILARITY_CANDIDATES if c in df_columns_params.columns]
        context_ore_feats = list(dict.fromkeys(context_ore_feats))
        cfg["context_ore_features"] = context_ore_feats

        # Build robust references/scales (prefer Control rows as baseline).
        if CATALYZED_COLUMNS_ID in df_columns_params.columns:
            status_norm = df_columns_params[CATALYZED_COLUMNS_ID].apply(normalize_status)
            ref_df = df_columns_params[status_norm == "Control"].copy()
            if ref_df.empty:
                ref_df = df_columns_params.copy()
        else:
            ref_df = df_columns_params.copy()

        ore_refs: Dict[str, float] = {}
        ore_scales: Dict[str, float] = {}
        for feat in context_ore_feats:
            vals = pd.to_numeric(ref_df[feat], errors="coerce")
            if vals.notna().sum() <= 0:
                vals = pd.to_numeric(df_columns_params[feat], errors="coerce")
            vals = vals.dropna()
            if vals.empty:
                continue
            arr = vals.to_numpy(dtype=float)
            ref = float(np.nanmedian(arr))
            q25 = float(np.nanquantile(arr, 0.25))
            q75 = float(np.nanquantile(arr, 0.75))
            iqr = float(max(0.0, q75 - q25))
            std = float(np.nanstd(arr))
            scale = float(max(iqr, 0.5 * std, abs(ref) * 0.05, 1e-6))
            ore_refs[feat] = ref
            ore_scales[feat] = scale
        cfg["context_ore_references"] = ore_refs
        cfg["context_ore_scales"] = ore_scales

        # Data-driven ore weights from paired catalyzed-control asymptote uplift.
        has_precomputed_weights = (
            isinstance(cfg.get("context_ore_weights", None), dict)
            and len(cfg.get("context_ore_weights", {})) > 0
        )
        ore_weights: Dict[str, float] = {feat: 0.0 for feat in context_ore_feats}
        if has_precomputed_weights:
            for feat in context_ore_feats:
                w = pd.to_numeric(cfg["context_ore_weights"].get(feat, 0.0), errors="coerce")
                ore_weights[feat] = float(w) if np.isfinite(w) else 0.0
        else:
            need_cols = {"project_sample_id", CATALYZED_COLUMNS_ID, "target_a1", "target_a2"}
            if need_cols.issubset(df_columns_params.columns) and context_ore_feats:
                pair_rows = []
                for sample_id, g in df_columns_params.groupby("project_sample_id", dropna=False):
                    status_norm = g[CATALYZED_COLUMNS_ID].apply(normalize_status)
                    ctrl = g[status_norm == "Control"]
                    cat = g[status_norm == "Catalyzed"]
                    if ctrl.empty or cat.empty:
                        continue
                    ctrl_row = ctrl.iloc[0]
                    cat_row = cat.iloc[0]
                    ctrl_asym = pd.to_numeric(ctrl_row.get("target_a1", np.nan), errors="coerce") + pd.to_numeric(
                        ctrl_row.get("target_a2", np.nan), errors="coerce"
                    )
                    cat_asym = pd.to_numeric(cat_row.get("target_a1", np.nan), errors="coerce") + pd.to_numeric(
                        cat_row.get("target_a2", np.nan), errors="coerce"
                    )
                    if not (np.isfinite(ctrl_asym) and np.isfinite(cat_asym)):
                        continue
                    uplift_norm = float((cat_asym - ctrl_asym) / max(abs(ctrl_asym), 1e-6))
                    rec = {"uplift_norm": uplift_norm}
                    for feat in context_ore_feats:
                        rec[feat] = pd.to_numeric(cat_row.get(feat, np.nan), errors="coerce")
                    pair_rows.append(rec)

                if pair_rows:
                    pair_df = pd.DataFrame(pair_rows)
                    weight_scale = float(max(0.0, cfg.get("context_ore_weight_scale", 0.75)))
                    weight_clip = float(max(0.0, cfg.get("context_ore_weight_clip", 0.35)))
                    for feat in context_ore_feats:
                        valid = pair_df[[feat, "uplift_norm"]].dropna()
                        if len(valid) < 6:
                            continue
                        corr = valid[feat].corr(valid["uplift_norm"], method="spearman")
                        if not np.isfinite(corr):
                            continue
                        reliability = float(np.clip((len(valid) - 3) / 15.0, 0.0, 1.0))
                        ore_weights[feat] = float(np.clip(corr * weight_scale * reliability, -weight_clip, weight_clip))
        cfg["context_ore_weights"] = {feat: float(ore_weights.get(feat, 0.0)) for feat in context_ore_feats}
    else:
        cfg["context_ore_features"] = []
        cfg["context_ore_references"] = {}
        cfg["context_ore_scales"] = {}
        cfg["context_ore_weights"] = {}
    return cfg


def _prepare_inference_uncertainty_config(df_columns_params: pd.DataFrame, unc_cfg: Dict) -> Dict:
    cfg = dict(unc_cfg or {})
    if cfg.get("reactor_distance_reference", None) is None:
        if "reactor_sim_mean_distance" in df_columns_params.columns:
            d = pd.to_numeric(df_columns_params["reactor_sim_mean_distance"], errors="coerce")
            cfg["reactor_distance_reference"] = float(d.median()) if d.notna().any() else 1.0
        else:
            cfg["reactor_distance_reference"] = 1.0
    return cfg


def _resolve_ensemble_seeds(seeds: List[int], min_members: int) -> List[int]:
    out = []
    for s in seeds:
        try:
            v = int(s)
        except Exception:
            continue
        if v not in out:
            out.append(v)
    target = max(int(min_members), 1)
    next_seed = out[-1] + 10 if out else 1
    while len(out) < target:
        out.append(next_seed)
        next_seed += 10
    return out


def _compute_row_input_uncertainty_score(
    row: pd.Series,
    shaping_cfg: Dict,
    unc_cfg: Dict,
) -> float:
    dist = pd.to_numeric(row.get("reactor_sim_mean_distance", np.nan), errors="coerce")
    dist_ref = pd.to_numeric(unc_cfg.get("reactor_distance_reference", np.nan), errors="coerce")
    if np.isfinite(dist) and np.isfinite(dist_ref) and dist_ref > 0:
        dist_score = float(dist / (dist + dist_ref))
    elif np.isfinite(dist):
        dist_score = float(dist / (1.0 + dist))
    else:
        dist_score = 1.0

    p80 = pd.to_numeric(row.get("material_size_p80_in", np.nan), errors="coerce")
    p80_ref = pd.to_numeric(shaping_cfg.get("p80_reference_in", np.nan), errors="coerce")
    if np.isfinite(p80) and np.isfinite(p80_ref) and abs(p80_ref) > 0:
        p80_score = float(abs((p80 - p80_ref) / p80_ref))
    else:
        p80_score = 0.0

    h = pd.to_numeric(row.get("column_height_m", np.nan), errors="coerce")
    d = pd.to_numeric(row.get("column_inner_diameter_m", np.nan), errors="coerce")
    h_ref = pd.to_numeric(shaping_cfg.get("geometry_lag_reference_height_m", np.nan), errors="coerce")
    d_ref = pd.to_numeric(shaping_cfg.get("geometry_lag_reference_diameter_m", np.nan), errors="coerce")
    if np.isfinite(h) and np.isfinite(d) and np.isfinite(h_ref) and np.isfinite(d_ref) and abs(h_ref) > 0 and abs(d_ref) > 0 and d > 0 and d_ref > 0:
        h_rel = abs((h - h_ref) / h_ref)
        d_rel = abs((d - d_ref) / d_ref)
        s_rel = abs(((h / d) - (h_ref / d_ref)) / max(abs(h_ref / d_ref), 1e-6))
        geom_score = float((h_rel + d_rel + s_rel) / 3.0)
    else:
        geom_score = 0.0

    status = normalize_status(row.get(CATALYZED_COLUMNS_ID, "Control"))
    if status == "Catalyzed":
        c_final = pd.to_numeric(row.get("catalyst_final_kg_t", np.nan), errors="coerce")
        c_ref = pd.to_numeric(shaping_cfg.get("catalyst_reference_kg_t", np.nan), errors="coerce")
        if np.isfinite(c_final) and np.isfinite(c_ref) and c_ref > 0:
            catalyst_score = float(abs((c_final - c_ref) / c_ref))
        elif np.isfinite(c_final):
            catalyst_score = float(c_final)
        else:
            catalyst_score = 0.0
    else:
        catalyst_score = 0.0

    residual = pd.to_numeric(row.get("residual_cpy_%", np.nan), errors="coerce")
    residual_ref = pd.to_numeric(shaping_cfg.get("residual_cpy_reference_pct", np.nan), errors="coerce")
    if np.isfinite(residual) and np.isfinite(residual_ref) and residual_ref > 0:
        residual_score = float(abs((residual - residual_ref) / residual_ref))
    else:
        residual_score = 0.0

    score = (
        float(unc_cfg.get("uncertainty_distance_weight", 0.0)) * dist_score
        + float(unc_cfg.get("uncertainty_p80_weight", 0.0)) * p80_score
        + float(unc_cfg.get("uncertainty_geometry_weight", 0.0)) * geom_score
        + float(unc_cfg.get("uncertainty_catalyst_weight", 0.0)) * catalyst_score
        + float(unc_cfg.get("uncertainty_residual_weight", 0.0)) * residual_score
    )
    score_max = float(max(0.1, unc_cfg.get("uncertainty_score_max", 1.5)))
    return float(np.clip(score, 0.0, score_max))


def _attach_uncertainty_scores(
    sample_df: pd.DataFrame,
    shaping_cfg: Dict,
    unc_cfg: Dict,
) -> pd.DataFrame:
    out = sample_df.copy()
    scores = []
    for _, row in out.iterrows():
        scores.append(_compute_row_input_uncertainty_score(row, shaping_cfg, unc_cfg))
    out["input_uncertainty_score"] = np.asarray(scores, dtype=float)
    return out


def _generate_member_param_predictions_from_base(
    sample_df: pd.DataFrame,
    base_pred_by_idx: Dict[int, np.ndarray],
    ensemble_seeds: List[int],
    curve_cfg: Dict,
    unc_cfg: Dict,
    shaping_cfg: Dict,
) -> np.ndarray:
    row_indices = list(sample_df.index)
    n_members = len(ensemble_seeds)
    member_preds = np.zeros((n_members, len(row_indices), 4), dtype=float)
    if bool(unc_cfg.get("enabled", True)):
        a_base = float(max(0.0, unc_cfg.get("param_noise_a_base", 0.0)))
        a_gain = float(max(0.0, unc_cfg.get("param_noise_a_gain", 0.0)))
        b_base = float(max(0.0, unc_cfg.get("param_noise_b_base", 0.0)))
        b_gain = float(max(0.0, unc_cfg.get("param_noise_b_gain", 0.0)))
    else:
        a_base = a_gain = b_base = b_gain = 0.0

    for m, seed in enumerate(ensemble_seeds):
        rng = np.random.default_rng(int(seed))
        for j, row_idx in enumerate(row_indices):
            row = sample_df.loc[row_idx]
            status = normalize_status(row.get(CATALYZED_COLUMNS_ID, "Control"))
            base = np.asarray(base_pred_by_idx[row_idx], dtype=float).copy()
            score = pd.to_numeric(row.get("input_uncertainty_score", np.nan), errors="coerce")
            score = float(score) if np.isfinite(score) else 0.0

            status_mult = (
                float(max(0.1, unc_cfg.get("catalyzed_noise_multiplier", 1.0)))
                if status == "Catalyzed"
                else float(max(0.1, unc_cfg.get("control_noise_multiplier", 1.0)))
            )
            a_sigma = status_mult * (a_base + a_gain * score)
            b_sigma = status_mult * (b_base + b_gain * score)

            p = base.copy()
            p[0] = p[0] * max(0.05, 1.0 + float(rng.normal(0.0, a_sigma)))
            p[2] = p[2] * max(0.05, 1.0 + float(rng.normal(0.0, a_sigma)))
            p[1] = p[1] * max(0.05, 1.0 + float(rng.normal(0.0, b_sigma)))
            p[3] = p[3] * max(0.05, 1.0 + float(rng.normal(0.0, b_sigma)))
            p = _apply_column_kinetics_caps(
                params=p,
                status=status,
                row=row,
                shaping_cfg=shaping_cfg,
            )
            p = _sanitize_curve_params(
                p[0],
                p[1],
                p[2],
                p[3],
                total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
                b_upper=curve_cfg["fit_b_fast_upper"],
            )
            member_preds[m, j, :] = p
    return member_preds


def _compute_column_b1_cap(status: str, row: pd.Series, shaping_cfg: Dict) -> float:
    if not bool(shaping_cfg.get("use_column_kinetics_caps", True)):
        return np.nan

    st = normalize_status(status)
    base_cap = pd.to_numeric(
        shaping_cfg.get("b1_cap_catalyzed" if st == "Catalyzed" else "b1_cap_control", np.nan),
        errors="coerce",
    )
    if not np.isfinite(base_cap):
        base_cap = pd.to_numeric(shaping_cfg.get("b1_cap_control", np.nan), errors="coerce")
    if not np.isfinite(base_cap):
        base_cap = 0.20
    base_cap = float(max(1e-5, base_cap))

    # Larger p80 usually implies slower apparent kinetics in columns.
    p80 = pd.to_numeric(row.get("material_size_p80_in", np.nan), errors="coerce")
    p80_ref = pd.to_numeric(shaping_cfg.get("p80_reference_in", np.nan), errors="coerce")
    if np.isfinite(p80) and np.isfinite(p80_ref):
        p80_delta = float(p80 - p80_ref)
        p80_factor = float(np.exp(-float(shaping_cfg.get("b1_cap_p80_sensitivity", 0.0)) * p80_delta))
        p80_factor = float(
            np.clip(
                p80_factor,
                float(shaping_cfg.get("b1_cap_p80_factor_min", 0.45)),
                float(shaping_cfg.get("b1_cap_p80_factor_max", 1.25)),
            )
        )
        base_cap *= p80_factor

    # Geometry impact on kinetics realism cap.
    h_val = pd.to_numeric(row.get("column_height_m", np.nan), errors="coerce")
    d_val = pd.to_numeric(row.get("column_inner_diameter_m", np.nan), errors="coerce")
    h_ref = pd.to_numeric(shaping_cfg.get("geometry_lag_reference_height_m", np.nan), errors="coerce")
    d_ref = pd.to_numeric(shaping_cfg.get("geometry_lag_reference_diameter_m", np.nan), errors="coerce")
    if np.isfinite(h_val) and np.isfinite(d_val) and np.isfinite(h_ref) and np.isfinite(d_ref) and d_val > 0 and d_ref > 0:
        h_rel = float((h_val - h_ref) / max(abs(h_ref), 1e-6))
        d_rel = float((d_val - d_ref) / max(abs(d_ref), 1e-6))
        slender_val = float(h_val / max(d_val, 1e-6))
        slender_ref = float(h_ref / max(d_ref, 1e-6))
        slender_rel = float((slender_val - slender_ref) / max(abs(slender_ref), 1e-6))
        geom_exp = (
            -float(shaping_cfg.get("b1_cap_height_sensitivity", 0.0)) * h_rel
            + float(shaping_cfg.get("b1_cap_diameter_sensitivity", 0.0)) * d_rel
            - float(shaping_cfg.get("b1_cap_slenderness_sensitivity", 0.0)) * slender_rel
        )
        geom_factor = float(np.exp(geom_exp))
        geom_factor = float(
            np.clip(
                geom_factor,
                float(shaping_cfg.get("b1_cap_geometry_factor_min", 0.60)),
                float(shaping_cfg.get("b1_cap_geometry_factor_max", 1.20)),
            )
        )
        base_cap *= geom_factor

    # Let reactor prior push cap only slightly, never dominate.
    reactor_b1 = pd.to_numeric(row.get("reactor_sim_b1", np.nan), errors="coerce")
    if np.isfinite(reactor_b1) and reactor_b1 > 0:
        softness = float(np.clip(shaping_cfg.get("b1_cap_reactor_softness", 0.15), 0.0, 1.0))
        hard_mult = float(max(1.0, shaping_cfg.get("b1_cap_reactor_max_multiplier", 1.30)))
        over = max(0.0, float(reactor_b1 / max(base_cap, 1e-6) - 1.0))
        base_cap = base_cap * (1.0 + softness * over)
        base_cap = min(base_cap, base_cap / max(1.0 + softness * over, 1e-9) * hard_mult)

    cap_min_raw = pd.to_numeric(shaping_cfg.get("b1_cap_min", np.nan), errors="coerce")
    cap_max_raw = pd.to_numeric(shaping_cfg.get("b1_cap_max", np.nan), errors="coerce")
    cap_min = float(max(1e-5, cap_min_raw)) if np.isfinite(cap_min_raw) else 0.005
    cap_max = float(max(cap_min, cap_max_raw)) if np.isfinite(cap_max_raw) else max(cap_min, 0.45)
    return float(np.clip(base_cap, cap_min, cap_max))


def _apply_column_kinetics_caps(
    params: np.ndarray,
    status: str,
    row: pd.Series,
    shaping_cfg: Dict,
) -> np.ndarray:
    p = np.asarray(params, dtype=float).copy()
    if p.size != 4 or not bool(shaping_cfg.get("use_column_kinetics_caps", True)):
        return p

    # Enforce fast/slow pairing before and after capping.
    p = _enforce_fast_slow_pairing(p)
    b1_cap = _compute_column_b1_cap(status=status, row=row, shaping_cfg=shaping_cfg)
    if np.isfinite(b1_cap):
        p[1] = min(float(p[1]), float(b1_cap))

    b2_ratio = float(np.clip(shaping_cfg.get("b2_cap_ratio_to_b1", 0.75), 0.05, 1.0))
    p[3] = min(float(p[3]), float(max(1e-6, p[1] * b2_ratio)))
    return _enforce_fast_slow_pairing(p)


def _robust_positive_ratio(numer: pd.Series, denom: pd.Series, eps: float = 1e-9) -> np.ndarray:
    n = pd.to_numeric(numer, errors="coerce")
    d = pd.to_numeric(denom, errors="coerce")
    ratio = n / np.maximum(d, float(max(1e-12, eps)))
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    vals = ratio.to_numpy(dtype=float)
    return vals[np.isfinite(vals) & (vals > 0.0)]


def _fit_param_calibration_factors_from_frame(calib_df: pd.DataFrame, cfg: Dict) -> Dict[str, float]:
    default = {
        "a_scale": 1.0,
        "b1_scale": 1.0,
        "b2_scale": 1.0,
        "b2_ratio_cap": np.nan,
        "total_cap": np.nan,
        "n_rows": 0,
    }
    if calib_df.empty:
        return default

    required = [
        "pred_a1",
        "pred_b1",
        "pred_a2",
        "pred_b2",
        "true_direct_a1",
        "true_direct_b1",
        "true_direct_a2",
        "true_direct_b2",
    ]
    if any(c not in calib_df.columns for c in required):
        return default

    pred_total = pd.to_numeric(calib_df["pred_a1"], errors="coerce") + pd.to_numeric(calib_df["pred_a2"], errors="coerce")
    true_total = pd.to_numeric(calib_df["true_direct_a1"], errors="coerce") + pd.to_numeric(calib_df["true_direct_a2"], errors="coerce")

    a_ratio = _robust_positive_ratio(true_total, pred_total)
    b1_ratio = _robust_positive_ratio(calib_df["true_direct_b1"], calib_df["pred_b1"])
    b2_ratio = _robust_positive_ratio(calib_df["true_direct_b2"], calib_df["pred_b2"])

    a_scale = float(np.nanmedian(a_ratio)) if a_ratio.size else 1.0
    b1_scale = float(np.nanmedian(b1_ratio)) if b1_ratio.size else 1.0
    b2_scale = float(np.nanmedian(b2_ratio)) if b2_ratio.size else 1.0

    a_scale = float(
        np.clip(
            a_scale,
            float(cfg.get("a_scale_min", 0.20)),
            float(cfg.get("a_scale_max", 1.50)),
        )
    )
    b1_scale = float(
        np.clip(
            b1_scale,
            float(cfg.get("b_scale_min", 0.005)),
            float(cfg.get("b_scale_max", 1.25)),
        )
    )
    b2_scale = float(
        np.clip(
            b2_scale,
            float(cfg.get("b_scale_min", 0.005)),
            float(cfg.get("b_scale_max", 1.25)),
        )
    )

    true_b2_b1 = _robust_positive_ratio(calib_df["true_direct_b2"], calib_df["true_direct_b1"])
    if true_b2_b1.size:
        q = float(np.clip(cfg.get("b2_ratio_cap_quantile", 0.50), 0.0, 1.0))
        b2_cap = float(np.quantile(true_b2_b1, q))
        b2_cap = float(
            np.clip(
                b2_cap,
                float(cfg.get("b2_ratio_cap_min", 0.02)),
                float(cfg.get("b2_ratio_cap_max", 0.95)),
            )
        )
    else:
        b2_cap = np.nan

    true_total_vals = pd.to_numeric(true_total, errors="coerce").to_numpy(dtype=float)
    true_total_vals = true_total_vals[np.isfinite(true_total_vals) & (true_total_vals > 0.0)]
    if true_total_vals.size:
        tq = float(np.clip(cfg.get("total_cap_quantile", 0.90), 0.0, 1.0))
        total_cap = float(np.quantile(true_total_vals, tq))
        total_cap = max(total_cap, float(cfg.get("total_cap_floor", 5.0)))
    else:
        total_cap = np.nan

    return {
        "a_scale": a_scale,
        "b1_scale": b1_scale,
        "b2_scale": b2_scale,
        "b2_ratio_cap": b2_cap,
        "total_cap": total_cap,
        "n_rows": int(len(calib_df)),
    }


def _fit_foldwise_param_calibration(
    validation_param_df: pd.DataFrame,
    direct_targets_df: pd.DataFrame,
    cfg: Dict,
) -> Dict[str, Dict]:
    out = {
        "enabled": bool(cfg.get("enabled", True)),
        "status_global": {},
        "overall": {},
        "fold_status": {},
    }
    if not out["enabled"]:
        return out
    if validation_param_df.empty or direct_targets_df.empty:
        out["enabled"] = False
        return out

    merge_cols = ["project_sample_id", CATALYZED_COLUMNS_ID]
    merged = validation_param_df.merge(direct_targets_df, on=merge_cols, how="left")
    required_direct = ["true_direct_a1", "true_direct_b1", "true_direct_a2", "true_direct_b2"]
    for col in required_direct:
        if col not in merged.columns:
            out["enabled"] = False
            return out

    min_rows = int(max(1, cfg.get("min_rows_per_status", 6)))

    overall = _fit_param_calibration_factors_from_frame(merged, cfg)
    out["overall"] = overall

    statuses = sorted(merged[CATALYZED_COLUMNS_ID].dropna().astype(str).map(normalize_status).unique().tolist())
    for status in statuses:
        sub = merged[merged[CATALYZED_COLUMNS_ID].apply(normalize_status) == status]
        factors = _fit_param_calibration_factors_from_frame(sub, cfg)
        if factors.get("n_rows", 0) < min_rows:
            factors = overall
        out["status_global"][status] = factors

    fold_ids = sorted(merged["fold_sample_id"].dropna().astype(str).unique().tolist())
    for fold_id in fold_ids:
        fold_train = merged[merged["fold_sample_id"].astype(str) != str(fold_id)]
        for status in statuses:
            sub = fold_train[fold_train[CATALYZED_COLUMNS_ID].apply(normalize_status) == status]
            factors = _fit_param_calibration_factors_from_frame(sub, cfg)
            if factors.get("n_rows", 0) < min_rows:
                factors = out["status_global"].get(status, overall)
            out["fold_status"][(str(fold_id), status)] = factors
    return out


def _get_param_calibration_factors(
    calibration_state: Dict,
    status: str,
    fold_id: str = None,
) -> Dict[str, float]:
    if not calibration_state or not bool(calibration_state.get("enabled", False)):
        return {}
    st = normalize_status(status)
    if fold_id is not None:
        key = (str(fold_id), st)
        if key in calibration_state.get("fold_status", {}):
            return calibration_state["fold_status"][key]
    if st in calibration_state.get("status_global", {}):
        return calibration_state["status_global"][st]
    return calibration_state.get("overall", {})


def _apply_param_calibration(
    params: np.ndarray,
    factors: Dict[str, float],
) -> np.ndarray:
    p = np.asarray(params, dtype=float).copy()
    if p.size != 4 or not factors:
        return p

    a_scale = pd.to_numeric(factors.get("a_scale", np.nan), errors="coerce")
    b1_scale = pd.to_numeric(factors.get("b1_scale", np.nan), errors="coerce")
    b2_scale = pd.to_numeric(factors.get("b2_scale", np.nan), errors="coerce")
    b2_cap = pd.to_numeric(factors.get("b2_ratio_cap", np.nan), errors="coerce")
    total_cap = pd.to_numeric(factors.get("total_cap", np.nan), errors="coerce")

    if np.isfinite(a_scale) and a_scale > 0:
        p[0] = float(p[0]) * float(a_scale)
        p[2] = float(p[2]) * float(a_scale)
    if np.isfinite(b1_scale) and b1_scale > 0:
        p[1] = float(p[1]) * float(b1_scale)
    if np.isfinite(b2_scale) and b2_scale > 0:
        p[3] = float(p[3]) * float(b2_scale)

    if np.isfinite(b2_cap) and b2_cap > 0 and np.isfinite(p[1]):
        p[3] = min(float(p[3]), float(max(1e-6, p[1] * b2_cap)))

    total = float(max(0.0, p[0] + p[2]))
    if np.isfinite(total_cap) and total_cap > 0 and total > total_cap:
        scale = float(total_cap) / max(total, 1e-9)
        p[0] = float(p[0]) * scale
        p[2] = float(p[2]) * scale
    return p


def _apply_material_and_catalyst_shaping(
    params: np.ndarray,
    status: str,
    row: pd.Series,
    shaping_cfg: Dict,
    fit_total_recovery_upper: float,
    fit_b_fast_upper: float,
) -> np.ndarray:
    p = np.asarray(params, dtype=float).copy()
    st = normalize_status(status)

    # Columns are less ideal than reactors in recovery and kinetics.
    p[0] *= float(shaping_cfg.get("column_nonideal_a_factor", 1.0))
    p[2] *= float(shaping_cfg.get("column_nonideal_a_factor", 1.0))
    p[1] *= float(shaping_cfg.get("column_nonideal_b_factor", 1.0))
    p[3] *= float(shaping_cfg.get("column_nonideal_b_factor", 1.0))

    if shaping_cfg.get("use_material_size_p80", True):
        p80 = pd.to_numeric(row.get("material_size_p80_in", np.nan), errors="coerce")
        p80_ref = pd.to_numeric(shaping_cfg.get("p80_reference_in", np.nan), errors="coerce")
        if np.isfinite(p80) and np.isfinite(p80_ref):
            delta = float(p80 - p80_ref)
            b_factor = float(np.exp(-float(shaping_cfg["p80_b_sensitivity"]) * delta))
            a_factor = float(np.exp(-float(shaping_cfg["p80_a_sensitivity"]) * delta))
            b_factor = float(np.clip(b_factor, shaping_cfg["p80_b_factor_min"], shaping_cfg["p80_b_factor_max"]))
            a_factor = float(np.clip(a_factor, shaping_cfg["p80_a_factor_min"], shaping_cfg["p80_a_factor_max"]))
            p[0] *= a_factor
            p[2] *= a_factor
            p[1] *= b_factor
            p[3] *= b_factor

    if st == "Catalyzed" and shaping_cfg.get("use_cumulative_catalyst", True):
        c_final = pd.to_numeric(row.get("catalyst_final_kg_t", 0.0), errors="coerce")
        c_auc = pd.to_numeric(row.get("catalyst_auc_kg_t_day", 0.0), errors="coerce")
        t_max = pd.to_numeric(row.get("curve_max_time_days", np.nan), errors="coerce")
        c_final = float(max(0.0, c_final)) if np.isfinite(c_final) else 0.0
        c_avg = float(max(0.0, c_auc / t_max)) if np.isfinite(c_auc) and np.isfinite(t_max) and t_max > 0 else 0.0

        c_signal = (
            float(shaping_cfg["catalyst_signal_weight_final"]) * c_final
            + float(shaping_cfg["catalyst_signal_weight_avg"]) * c_avg
        )
        c_ref = pd.to_numeric(shaping_cfg.get("catalyst_reference_kg_t", np.nan), errors="coerce")
        if np.isfinite(c_ref):
            delta = float(c_signal - c_ref)
            b_factor = float(np.exp(float(shaping_cfg["catalyst_b_sensitivity"]) * delta))
            a_factor = float(np.exp(float(shaping_cfg["catalyst_a_sensitivity"]) * delta))
            b_factor = float(
                np.clip(
                    b_factor,
                    float(shaping_cfg["catalyst_b_factor_min"]),
                    float(shaping_cfg["catalyst_b_factor_max"]),
                )
            )
            a_factor = float(
                np.clip(
                    a_factor,
                    float(shaping_cfg["catalyst_a_factor_min"]),
                    float(shaping_cfg["catalyst_a_factor_max"]),
                )
            )
            p[0] *= a_factor
            p[2] *= a_factor
            p[1] *= b_factor
            p[3] *= b_factor

    p = _apply_column_kinetics_caps(
        params=p,
        status=st,
        row=row,
        shaping_cfg=shaping_cfg,
    )

    p = _sanitize_curve_params(
        p[0],
        p[1],
        p[2],
        p[3],
        total_recovery_upper=fit_total_recovery_upper,
        b_upper=fit_b_fast_upper,
    )
    return p


def _resolve_catalyst_lag_tau(
    shaping_cfg: Dict,
    material_size_p80_in: float = np.nan,
    column_height_m: float = np.nan,
    column_inner_diameter_m: float = np.nan,
) -> Tuple[float, float]:
    tau_days = float(max(1e-6, shaping_cfg.get("dynamic_uplift_tau_days", 35.0)))
    lag_days = float(max(0.0, shaping_cfg.get("dynamic_uplift_base_lag_days", 0.0)))

    if bool(shaping_cfg.get("use_p80_catalyst_lag", True)):
        p80_val = pd.to_numeric(material_size_p80_in, errors="coerce")
        p80_ref = pd.to_numeric(shaping_cfg.get("p80_lag_reference_in", np.nan), errors="coerce")
        if np.isfinite(p80_val) and np.isfinite(p80_ref):
            p80_delta = float(p80_val - p80_ref)
            tau_mult = float(np.exp(float(shaping_cfg.get("p80_lag_sensitivity", 0.0)) * p80_delta))
            tau_mult = float(
                np.clip(
                    tau_mult,
                    float(shaping_cfg.get("p80_lag_min_multiplier", 0.5)),
                    float(shaping_cfg.get("p80_lag_max_multiplier", 4.0)),
                )
            )
            tau_days = tau_days * tau_mult
            lag_days = lag_days + float(shaping_cfg.get("dynamic_uplift_p80_lag_days_per_in", 0.0)) * p80_delta

    if bool(shaping_cfg.get("use_column_geometry_catalyst_lag", True)):
        h_val = pd.to_numeric(column_height_m, errors="coerce")
        d_val = pd.to_numeric(column_inner_diameter_m, errors="coerce")
        h_ref = pd.to_numeric(shaping_cfg.get("geometry_lag_reference_height_m", np.nan), errors="coerce")
        d_ref = pd.to_numeric(shaping_cfg.get("geometry_lag_reference_diameter_m", np.nan), errors="coerce")
        if np.isfinite(h_val) and np.isfinite(d_val) and np.isfinite(h_ref) and np.isfinite(d_ref) and d_val > 0 and d_ref > 0:
            h_rel = float((h_val - h_ref) / max(abs(h_ref), 1e-6))
            d_rel = float((d_val - d_ref) / max(abs(d_ref), 1e-6))
            slender_val = float(h_val / max(d_val, 1e-6))
            slender_ref = float(h_ref / max(d_ref, 1e-6))
            slender_rel = float((slender_val - slender_ref) / max(abs(slender_ref), 1e-6))

            tau_geom_exp = (
                float(shaping_cfg.get("height_tau_sensitivity_per_rel", 0.0)) * h_rel
                + float(shaping_cfg.get("diameter_tau_sensitivity_per_rel", 0.0)) * d_rel
                + float(shaping_cfg.get("slenderness_tau_sensitivity_per_rel", 0.0)) * slender_rel
            )
            tau_geom_mult = float(np.exp(tau_geom_exp))
            tau_geom_mult = float(
                np.clip(
                    tau_geom_mult,
                    float(shaping_cfg.get("geometry_tau_factor_min", 0.5)),
                    float(shaping_cfg.get("geometry_tau_factor_max", 3.0)),
                )
            )
            tau_days = tau_days * tau_geom_mult

            lag_days = lag_days + (
                float(shaping_cfg.get("height_lag_days_per_rel", 0.0)) * h_rel
                + float(shaping_cfg.get("diameter_lag_days_per_rel", 0.0)) * d_rel
                + float(shaping_cfg.get("slenderness_lag_days_per_rel", 0.0)) * slender_rel
            )

    lag_days = float(
        np.clip(
            lag_days,
            float(shaping_cfg.get("dynamic_uplift_lag_days_min", 0.0)),
            float(shaping_cfg.get("dynamic_uplift_lag_days_max", 120.0)),
        )
    )
    tau_days = float(max(1e-6, tau_days))
    return lag_days, tau_days


def _compute_lagged_catalyst_activity(
    cumulative_catalyst: np.ndarray,
    time_days: np.ndarray,
    lag_days: float,
    tau_days: float,
) -> np.ndarray:
    c = np.asarray(cumulative_catalyst, dtype=float)
    t = np.asarray(time_days, dtype=float)
    n = min(len(c), len(t))
    if n == 0:
        return np.asarray([], dtype=float)
    c = c[:n]
    t = t[:n]

    c = np.where(np.isfinite(c), c, np.nan)
    if np.isnan(c).all():
        return np.zeros(n, dtype=float)
    c = pd.Series(c).interpolate(limit_direction="both").to_numpy(dtype=float)
    c = np.maximum.accumulate(np.maximum(c, 0.0))

    t = np.where(np.isfinite(t), t, np.nan)
    if np.isnan(t).all():
        t = np.arange(n, dtype=float)
    else:
        t = pd.Series(t).interpolate(limit_direction="both").to_numpy(dtype=float)
    t = np.maximum.accumulate(t)

    shifted_t = t - float(max(0.0, lag_days))
    target = np.interp(shifted_t, t, c, left=0.0, right=float(c[-1]))

    tau = float(max(1e-6, tau_days))
    activity = np.zeros(n, dtype=float)
    for i in range(1, n):
        dt = float(max(0.0, t[i] - t[i - 1]))
        alpha = 1.0 - np.exp(-dt / tau) if dt > 0 else 0.0
        activity[i] = activity[i - 1] + alpha * (target[i] - activity[i - 1])
    activity = np.maximum.accumulate(np.maximum(activity, 0.0))
    return activity


def _compute_dynamic_catalyst_progress(
    cumulative_catalyst: np.ndarray,
    time_days: np.ndarray,
    shaping_cfg: Dict,
    material_size_p80_in: float = np.nan,
    column_height_m: float = np.nan,
    column_inner_diameter_m: float = np.nan,
    context_tau_multiplier: float = 1.0,
) -> np.ndarray:
    c = np.asarray(cumulative_catalyst, dtype=float)
    t = np.asarray(time_days, dtype=float)
    n = min(len(c), len(t))
    if n == 0:
        return np.asarray([], dtype=float)
    c = c[:n]
    t = t[:n]

    c = np.where(np.isfinite(c), c, np.nan)
    if np.isnan(c).all():
        return np.zeros(n, dtype=float)
    c = pd.Series(c).interpolate(limit_direction="both").to_numpy(dtype=float)
    c = np.maximum.accumulate(np.maximum(c, 0.0))

    t = np.where(np.isfinite(t), t, np.nan)
    if np.isnan(t).all():
        t = np.arange(n, dtype=float)
    else:
        t = pd.Series(t).interpolate(limit_direction="both").to_numpy(dtype=float)
    t = np.maximum.accumulate(t)

    lag_days, tau_days = _resolve_catalyst_lag_tau(
        shaping_cfg=shaping_cfg,
        material_size_p80_in=material_size_p80_in,
        column_height_m=column_height_m,
        column_inner_diameter_m=column_inner_diameter_m,
    )
    tau_mult = pd.to_numeric(context_tau_multiplier, errors="coerce")
    tau_mult = float(tau_mult) if np.isfinite(tau_mult) else 1.0
    tau_mult = float(np.clip(tau_mult, 0.25, 4.0))
    tau_days = float(max(1e-6, tau_days * tau_mult))

    dose_inc = np.maximum(np.diff(c, prepend=c[0]), 0.0)
    source = dose_inc.copy()
    source[0] = source[0] + float(max(0.0, c[0]))
    total_source = float(np.sum(source))

    if total_source > 0:
        jump_fraction = float(np.clip(shaping_cfg.get("dynamic_uplift_jump_fraction", 0.35), 0.0, 1.0))
        dt = t[:, None] - t[None, :]
        dt_effective = dt - lag_days
        kernel = np.where(dt_effective >= 0.0, 1.0 - np.exp(-dt_effective / tau_days), 0.0)
        smooth_progress = (kernel @ source) / total_source
        step_progress = np.cumsum(source) / total_source
        progress_base = jump_fraction * step_progress + (1.0 - jump_fraction) * smooth_progress
    else:
        progress_base = np.zeros(n, dtype=float)

    if bool(shaping_cfg.get("use_catalyst_residual_memory", True)):
        memory_tau_mult = float(max(1e-6, shaping_cfg.get("residual_memory_tau_multiplier", 2.0)))
        memory_tau_days = float(max(1e-6, tau_days * memory_tau_mult))
        activity = _compute_lagged_catalyst_activity(
            cumulative_catalyst=c,
            time_days=t,
            lag_days=lag_days,
            tau_days=memory_tau_days,
        )
        activity_end = float(activity[-1]) if activity.size else 0.0
        if activity_end > 0:
            progress_memory = np.clip(activity / activity_end, 0.0, 1.0)
        else:
            progress_memory = np.zeros(n, dtype=float)

        memory_weight = float(np.clip(shaping_cfg.get("residual_memory_weight", 0.65), 0.0, 1.0))
        progress = (1.0 - memory_weight) * progress_base + memory_weight * progress_memory
    else:
        progress = progress_base

    progress = np.clip(progress, 0.0, 1.0)
    progress = np.maximum.accumulate(progress)
    return progress


def _apply_time_dependent_catalyst_separation(
    control_curve: np.ndarray,
    catalyzed_curve: np.ndarray,
    catalyst_profile: np.ndarray,
    shaping_cfg: Dict,
    coupling_cfg: Dict,
    time_days: np.ndarray = None,
    transition_time: float = np.nan,
    material_size_p80_in: float = np.nan,
    column_height_m: float = np.nan,
    column_inner_diameter_m: float = np.nan,
    residual_cpy_pct: float = np.nan,
    row_context: pd.Series = None,
) -> np.ndarray:
    ctrl = np.asarray(control_curve, dtype=float)
    cat = np.asarray(catalyzed_curve, dtype=float)
    if ctrl.size == 0 or cat.size == 0:
        return cat

    out = cat.copy()
    if not shaping_cfg.get("use_cumulative_catalyst", True):
        return out

    c = _align_profile_to_time_length(np.asarray(catalyst_profile, dtype=float), len(out))
    if c.size == 0:
        return np.maximum(out, ctrl)
    c = np.where(np.isfinite(c), c, np.nan)
    if np.isnan(c).all():
        return np.maximum(out, ctrl)
    c = pd.Series(c).interpolate(limit_direction="both").to_numpy(dtype=float)
    c = np.maximum.accumulate(np.maximum(c, 0.0))

    t = np.asarray(time_days if time_days is not None else np.arange(len(out)), dtype=float)
    t = _align_profile_to_time_length(t, len(out))
    if np.isnan(np.where(np.isfinite(t), t, np.nan)).all():
        t = np.arange(len(out), dtype=float)
    else:
        t = pd.Series(t).interpolate(limit_direction="both").to_numpy(dtype=float)
    t = np.maximum.accumulate(t)

    c0 = float(c[0]) if c.size else 0.0
    c_end = float(c[-1]) if c.size else 0.0
    context_offsets = _compute_catalyst_context_offsets(
        row_context=row_context,
        shaping_cfg=shaping_cfg,
        material_size_p80_in=material_size_p80_in,
        column_height_m=column_height_m,
        column_inner_diameter_m=column_inner_diameter_m,
    )
    use_dynamic = bool(shaping_cfg.get("use_dynamic_catalyst_uplift", False))
    if use_dynamic:
        progress = _compute_dynamic_catalyst_progress(
            cumulative_catalyst=c,
            time_days=t,
            shaping_cfg=shaping_cfg,
            material_size_p80_in=material_size_p80_in,
            column_height_m=column_height_m,
            column_inner_diameter_m=column_inner_diameter_m,
            context_tau_multiplier=float(context_offsets.get("tau_factor", 1.0)),
        )
    else:
        use_abs_dose = bool(shaping_cfg.get("dose_uses_absolute_cumulative", True))
        if use_abs_dose:
            span = float(max(1e-9, c_end))
            progress = np.clip(c / span, 0.0, 1.0) if span > 1e-9 else np.zeros_like(c)
        else:
            c_delta = np.maximum(c - c0, 0.0)
            span = float(max(1e-9, c_end - c0))
            progress = np.clip(c_delta / span, 0.0, 1.0) if span > 1e-9 else np.zeros_like(c_delta)
        progress = np.power(progress, float(max(0.1, context_offsets.get("tau_factor", 1.0))))
    gamma = float(max(0.1, shaping_cfg.get("catalyst_progress_gamma", 1.0)))
    gamma = gamma / float(max(1e-6, context_offsets.get("b_factor", 1.0)))
    progress = np.power(progress, gamma)

    if shaping_cfg.get("use_transition_time_gating", True):
        if np.isfinite(transition_time):
            blend_days = float(max(1e-6, shaping_cfg.get("transition_blend_days", 1.0)))
            gate_t = np.clip((t - float(transition_time)) / blend_days, 0.0, 1.0)
            progress = progress * gate_t

    c_ref = pd.to_numeric(shaping_cfg.get("catalyst_reference_kg_t", np.nan), errors="coerce")
    if np.isfinite(c_ref):
        level_factor = float(np.exp(float(shaping_cfg["catalyst_time_sensitivity"]) * (c_end - float(c_ref))))
        level_factor = float(
            np.clip(
                level_factor,
                float(shaping_cfg["catalyst_time_factor_min"]),
                float(shaping_cfg["catalyst_time_factor_max"]),
            )
        )
    else:
        level_factor = 1.0
    level_factor = level_factor * float(context_offsets.get("a_factor", 1.0))

    dose_factor = 1.0
    if np.isfinite(c_ref) and float(c_ref) > 0:
        use_abs_dose = bool(shaping_cfg.get("dose_uses_absolute_cumulative", True))
        dose_signal = c_end if use_abs_dose else max(0.0, c_end - c0)
        dose_norm = float(max(0.0, dose_signal) / max(1e-9, float(c_ref)))
        dose_gamma = float(max(0.1, shaping_cfg.get("catalyst_dose_gamma", 1.0)))
        dose_factor = float(np.power(dose_norm, dose_gamma))
        dose_factor = float(
            np.clip(
                dose_factor,
                float(shaping_cfg.get("catalyst_dose_factor_min", 0.0)),
                float(shaping_cfg.get("catalyst_dose_factor_max", 1.0)),
            )
        )
    dose_factor = dose_factor * float(context_offsets.get("dose_factor", 1.0))

    residual_factor = 1.0
    residual_rel = 0.0
    if bool(shaping_cfg.get("use_residual_cpy_uplift", True)):
        residual_val = pd.to_numeric(residual_cpy_pct, errors="coerce")
        residual_ref = pd.to_numeric(shaping_cfg.get("residual_cpy_reference_pct", np.nan), errors="coerce")
        if np.isfinite(residual_val) and np.isfinite(residual_ref):
            residual_rel = float((residual_val - residual_ref) / max(abs(residual_ref), 1e-6))
            residual_factor = float(np.exp(float(shaping_cfg.get("residual_uplift_sensitivity", 0.0)) * residual_rel))
            residual_factor = float(
                np.clip(
                    residual_factor,
                    float(shaping_cfg.get("residual_uplift_factor_min", 0.5)),
                    float(shaping_cfg.get("residual_uplift_factor_max", 2.0)),
                )
            )

    interaction_factor = 1.0
    if bool(shaping_cfg.get("use_residual_uplift_interactions", True)):
        p80_val = pd.to_numeric(material_size_p80_in, errors="coerce")
        p80_ref = pd.to_numeric(shaping_cfg.get("p80_reference_in", np.nan), errors="coerce")
        p80_rel = (
            float((p80_val - p80_ref) / max(abs(p80_ref), 1e-6))
            if np.isfinite(p80_val) and np.isfinite(p80_ref)
            else 0.0
        )

        h_val = pd.to_numeric(column_height_m, errors="coerce")
        d_val = pd.to_numeric(column_inner_diameter_m, errors="coerce")
        h_ref = pd.to_numeric(shaping_cfg.get("geometry_lag_reference_height_m", np.nan), errors="coerce")
        d_ref = pd.to_numeric(shaping_cfg.get("geometry_lag_reference_diameter_m", np.nan), errors="coerce")
        if np.isfinite(h_val) and np.isfinite(d_val) and np.isfinite(h_ref) and np.isfinite(d_ref) and d_val > 0 and d_ref > 0:
            slender_val = float(h_val / max(d_val, 1e-6))
            slender_ref = float(h_ref / max(d_ref, 1e-6))
            slender_rel = float((slender_val - slender_ref) / max(abs(slender_ref), 1e-6))
        else:
            slender_rel = 0.0

        interaction_exp = (
            float(shaping_cfg.get("residual_p80_interaction_sensitivity", 0.0)) * residual_rel * p80_rel
            + float(shaping_cfg.get("residual_geometry_interaction_sensitivity", 0.0)) * residual_rel * slender_rel
        )
        interaction_factor = float(np.exp(interaction_exp))
        interaction_factor = float(
            np.clip(
                interaction_factor,
                float(shaping_cfg.get("residual_interaction_factor_min", 0.5)),
                float(shaping_cfg.get("residual_interaction_factor_max", 1.5)),
            )
        )

    gap = np.maximum(out - ctrl, 0.0)
    gap = gap * progress
    gap = gap * (1.0 + (level_factor - 1.0) * progress)
    gap = gap * dose_factor
    gap = gap * residual_factor
    gap = gap * interaction_factor

    # Moderate catalyst-driven uplift to avoid unrealistic separation spikes.
    gap = gap * float(np.clip(shaping_cfg.get("catalyst_gap_global_damp", 1.0), 0.05, 2.0))
    if bool(shaping_cfg.get("use_catalyst_gap_soft_cap", True)):
        cap_abs = pd.to_numeric(shaping_cfg.get("catalyst_gap_soft_cap_abs_pct", np.nan), errors="coerce")
        cap_rel = pd.to_numeric(shaping_cfg.get("catalyst_gap_soft_cap_fraction_of_control", np.nan), errors="coerce")
        cap_min = pd.to_numeric(shaping_cfg.get("catalyst_gap_soft_cap_min_pct", 0.0), errors="coerce")
        cap_rel_curve = np.full_like(gap, np.inf, dtype=float)
        cap_abs_curve = np.full_like(gap, np.inf, dtype=float)
        if np.isfinite(cap_rel) and cap_rel > 0:
            cap_rel_curve = np.maximum(np.abs(ctrl) * float(cap_rel), 0.0)
        if np.isfinite(cap_abs) and cap_abs > 0:
            cap_abs_curve = np.full_like(gap, float(cap_abs), dtype=float)
        cap_curve = np.minimum(cap_rel_curve, cap_abs_curve)
        if np.isfinite(cap_min) and cap_min > 0:
            cap_curve = np.maximum(cap_curve, float(cap_min))
        valid_cap = np.isfinite(cap_curve) & (cap_curve > 0)
        if np.any(valid_cap):
            # Soft saturation: linear for small gaps, saturates near cap.
            gap_soft = gap.copy()
            gap_soft[valid_cap] = cap_curve[valid_cap] * np.tanh(gap[valid_cap] / np.maximum(cap_curve[valid_cap], 1e-9))
            gap = gap_soft

    out = ctrl + gap

    enforce_above = coupling_cfg.get("enforce_catalyzed_above_control", True)
    if enforce_above:
        out = np.maximum(out, ctrl)

    touch_first = coupling_cfg.get("touch_first_point", True)
    max_gap = max(0.0, float(coupling_cfg.get("first_point_max_gap", 0.0)))
    if touch_first and out.size > 0:
        out[0] = min(out[0], ctrl[0] + max_gap)
        if enforce_above:
            out[0] = max(out[0], ctrl[0])
    return out


def _compute_reactor_status_param_medians(
    df_reactors: pd.DataFrame,
    fit_total_recovery_upper: float,
    fit_b_fast_upper: float,
) -> Dict[str, np.ndarray]:
    reactors = df_reactors.copy()
    reactors["status_norm"] = reactors[CATALYZED_REACTORS_ID].apply(normalize_status)
    for col in ["a1_param", "b1_param", "a2_param", "b2_param"]:
        reactors[col] = pd.to_numeric(reactors[col], errors="coerce")

    out = {}
    default = np.array([30.0, 0.10, 20.0, 0.01], dtype=float)
    for status in ["Control", "Catalyzed"]:
        sub = reactors[reactors["status_norm"] == status].dropna(
            subset=["a1_param", "b1_param", "a2_param", "b2_param"]
        )
        if sub.empty:
            out[status] = default.copy()
            continue
        params = sub[["a1_param", "b1_param", "a2_param", "b2_param"]].values.astype(float)
        params = np.vstack(
            [
                _sanitize_curve_params(
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    total_recovery_upper=fit_total_recovery_upper,
                    b_upper=fit_b_fast_upper,
                )
                for p in params
            ]
        )
        med = np.nanmedian(params, axis=0)
        out[status] = _sanitize_curve_params(
            med[0],
            med[1],
            med[2],
            med[3],
            total_recovery_upper=fit_total_recovery_upper,
            b_upper=fit_b_fast_upper,
        )
    return out


def _infer_sample_params_from_reactor_knowledge(
    sample_df: pd.DataFrame,
    reactor_status_medians: Dict[str, np.ndarray],
    curve_cfg: Dict,
    coupling_cfg: Dict,
    shaping_cfg: Dict,
) -> Dict[int, np.ndarray]:
    pred_by_idx: Dict[int, np.ndarray] = {}
    if sample_df.empty:
        return pred_by_idx

    for row_idx, row in sample_df.iterrows():
        status = normalize_status(row.get(CATALYZED_COLUMNS_ID, "Control"))
        prior = row[["reactor_sim_a1", "reactor_sim_b1", "reactor_sim_a2", "reactor_sim_b2"]].to_numpy(dtype=float)
        if np.all(np.isfinite(prior)):
            p = _sanitize_curve_params(
                prior[0],
                prior[1],
                prior[2],
                prior[3],
                total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
                b_upper=curve_cfg["fit_b_fast_upper"],
            )
        else:
            p = reactor_status_medians.get(status, reactor_status_medians.get("Control", np.array([30.0, 0.10, 20.0, 0.01], dtype=float))).copy()
        p = _apply_material_and_catalyst_shaping(
            params=p,
            status=status,
            row=row,
            shaping_cfg=shaping_cfg,
            fit_total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
            fit_b_fast_upper=curve_cfg["fit_b_fast_upper"],
        )
        if curve_cfg.get("enforce_reactor_asymptote_cap", False) and np.all(np.isfinite(prior)):
            p = _cap_by_reactor_asymptote(p, prior)
        pred_by_idx[row_idx] = p

    if not coupling_cfg.get("enabled", False):
        return pred_by_idx

    status_norm = sample_df[CATALYZED_COLUMNS_ID].apply(normalize_status)
    control_rows = sample_df.index[status_norm == "Control"].tolist()
    catalyzed_rows = sample_df.index[status_norm == "Catalyzed"].tolist()
    if not control_rows or not catalyzed_rows:
        return pred_by_idx

    control_idx = control_rows[0]
    for cat_idx in catalyzed_rows:
        row_cat = sample_df.loc[cat_idx]
        t_full = np.asarray(row_cat.get("curve_time_days", []), dtype=float)
        y_full = np.asarray(row_cat.get("curve_target_recovery", []), dtype=float)
        n_ty = min(len(t_full), len(y_full))
        if n_ty == 0:
            continue
        t_valid = t_full[:n_ty]
        y_valid = y_full[:n_ty]
        valid_mask = np.isfinite(t_valid) & np.isfinite(y_valid)
        t_cat = t_valid[valid_mask]
        c_full = _align_profile_to_time_length(
            np.asarray(row_cat.get("curve_catalyst_profile", []), dtype=float),
            n_ty,
        )
        c_prof = c_full[valid_mask] if valid_mask.size == c_full.size else _align_profile_to_time_length(c_full, len(t_cat))
        if t_cat.size == 0:
            continue

        uplift_prior = pd.to_numeric(row_cat.get("reactor_norm_uplift_prior_pct", 0.0), errors="coerce")
        uplift_prior = float(uplift_prior) if np.isfinite(uplift_prior) else 0.0

        pred_by_idx[cat_idx] = _constrain_catalyzed_params_against_control(
            control_params=pred_by_idx[control_idx],
            catalyzed_params=pred_by_idx[cat_idx],
            t_days=t_cat,
            catalyst_profile=c_prof,
            reactor_norm_uplift_prior_pct=uplift_prior,
            coupling_cfg=coupling_cfg,
            fit_total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
            fit_b_fast_upper=curve_cfg["fit_b_fast_upper"],
            row_context=row_cat,
            shaping_cfg=shaping_cfg,
        )
        pred_by_idx[cat_idx] = _apply_column_kinetics_caps(
            params=pred_by_idx[cat_idx],
            status="Catalyzed",
            row=row_cat,
            shaping_cfg=shaping_cfg,
        )
        reactor_cat_prior = row_cat[["reactor_sim_a1", "reactor_sim_b1", "reactor_sim_a2", "reactor_sim_b2"]].to_numpy(dtype=float)
        if curve_cfg.get("enforce_reactor_asymptote_cap", False) and np.all(np.isfinite(reactor_cat_prior)):
            pred_by_idx[cat_idx] = _cap_by_reactor_asymptote(pred_by_idx[cat_idx], reactor_cat_prior)
    return pred_by_idx


def _prepare_member_params_for_sample(
    sample_df: pd.DataFrame,
    member_param_preds: np.ndarray,
    curve_cfg: Dict,
    coupling_cfg: Dict,
    shaping_cfg: Dict,
    param_calibration: Dict = None,
    fold_id: str = None,
) -> Dict[int, np.ndarray]:
    member_params_by_idx: Dict[int, np.ndarray] = {}
    if sample_df.empty:
        return member_params_by_idx

    for j, row_idx in enumerate(sample_df.index):
        row_meta = sample_df.loc[row_idx]
        status_norm = normalize_status(row_meta.get(CATALYZED_COLUMNS_ID, "Control"))
        calib_factors = _get_param_calibration_factors(
            calibration_state=param_calibration,
            status=status_norm,
            fold_id=fold_id,
        )
        reactor_prior = row_meta[
            ["reactor_sim_a1", "reactor_sim_b1", "reactor_sim_a2", "reactor_sim_b2"]
        ].to_numpy(dtype=float)
        cleaned_members = []
        for pred in member_param_preds[:, j, :]:
            p = _apply_param_calibration(
                params=np.asarray(pred, dtype=float),
                factors=calib_factors,
            )
            p = _sanitize_curve_params(
                p[0],
                p[1],
                p[2],
                p[3],
                total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
                b_upper=curve_cfg["fit_b_fast_upper"],
            )
            p = _apply_column_kinetics_caps(
                params=p,
                status=status_norm,
                row=row_meta,
                shaping_cfg=shaping_cfg,
            )
            if curve_cfg["enforce_reactor_asymptote_cap"]:
                p = _cap_by_reactor_asymptote(p, reactor_prior)
            cleaned_members.append(p)
        member_params_by_idx[row_idx] = np.vstack(cleaned_members)

    if not coupling_cfg.get("enabled", False):
        return member_params_by_idx

    status_norm = sample_df[CATALYZED_COLUMNS_ID].apply(normalize_status)
    control_rows = sample_df.index[status_norm == "Control"].tolist()
    catalyzed_rows = sample_df.index[status_norm == "Catalyzed"].tolist()
    if not control_rows or not catalyzed_rows:
        return member_params_by_idx

    control_idx = control_rows[0]
    n_members = member_params_by_idx[control_idx].shape[0]
    for cat_idx in catalyzed_rows:
        row_cat = sample_df.loc[cat_idx]
        t_full = np.asarray(row_cat.get("curve_time_days", []), dtype=float)
        y_full = np.asarray(row_cat.get("curve_target_recovery", []), dtype=float)
        n_ty = min(len(t_full), len(y_full))
        if n_ty == 0:
            continue
        t_valid = t_full[:n_ty]
        y_valid = y_full[:n_ty]
        valid_mask = np.isfinite(t_valid) & np.isfinite(y_valid)
        t_cat = t_valid[valid_mask]
        c_full = _align_profile_to_time_length(
            np.asarray(row_cat.get("curve_catalyst_profile", []), dtype=float),
            n_ty,
        )
        c_prof = c_full[valid_mask] if valid_mask.size == c_full.size else _align_profile_to_time_length(c_full, len(t_cat))
        if t_cat.size == 0:
            continue
        uplift_prior = pd.to_numeric(row_cat.get("reactor_norm_uplift_prior_pct", 0.0), errors="coerce")
        uplift_prior = float(uplift_prior) if np.isfinite(uplift_prior) else 0.0
        for m in range(n_members):
            p_ctrl = member_params_by_idx[control_idx][m]
            p_cat = member_params_by_idx[cat_idx][m]
            p_cat_adj = _constrain_catalyzed_params_against_control(
                control_params=p_ctrl,
                catalyzed_params=p_cat,
                t_days=t_cat,
                catalyst_profile=c_prof,
                reactor_norm_uplift_prior_pct=uplift_prior,
                coupling_cfg=coupling_cfg,
                fit_total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
                fit_b_fast_upper=curve_cfg["fit_b_fast_upper"],
                row_context=row_cat,
                shaping_cfg=shaping_cfg,
            )
            p_cat_adj = _apply_column_kinetics_caps(
                params=p_cat_adj,
                status="Catalyzed",
                row=row_cat,
                shaping_cfg=shaping_cfg,
            )
            member_params_by_idx[cat_idx][m] = p_cat_adj
    return member_params_by_idx


def _expand_curve_predictions_long(
    row_meta: pd.Series,
    pred_param_members: np.ndarray,
    pi_lower: float,
    pi_upper: float,
    fit_total_recovery_upper: float,
    fit_b_fast_upper: float,
) -> pd.DataFrame:
    t = np.asarray(row_meta["curve_time_days"], dtype=float)
    y_true = np.asarray(row_meta["curve_target_recovery"], dtype=float)
    n = min(len(t), len(y_true))
    if n == 0:
        return pd.DataFrame()
    t = t[:n]
    y_true = y_true[:n]
    valid = np.isfinite(t) & np.isfinite(y_true)
    if not valid.any():
        return pd.DataFrame()
    t = t[valid]
    y_true = y_true[valid]
    c_prof = _align_profile_to_time_length(
        np.asarray(row_meta.get("curve_catalyst_profile", []), dtype=float),
        n,
    )
    c_prof = c_prof[valid] if c_prof.size == n else _align_profile_to_time_length(c_prof, len(t))

    cleaned_members = []
    for member in pred_param_members:
        p = _sanitize_curve_params(
            member[0],
            member[1],
            member[2],
            member[3],
            total_recovery_upper=fit_total_recovery_upper,
            b_upper=fit_b_fast_upper,
        )
        cleaned_members.append(p)
    cleaned_members = np.vstack(cleaned_members)

    curve_members = np.vstack(
        [
            construct_reactor_recovery_curve(p[0], p[1], p[2], p[3], t)
            for p in cleaned_members
        ]
    )
    y_pred_mean = np.mean(curve_members, axis=0)
    y_pred_low = np.percentile(curve_members, pi_lower, axis=0)
    y_pred_high = np.percentile(curve_members, pi_upper, axis=0)

    p_mean = np.mean(cleaned_members, axis=0)
    p_low = np.percentile(cleaned_members, pi_lower, axis=0)
    p_high = np.percentile(cleaned_members, pi_upper, axis=0)
    rec = {
        "project_sample_id": row_meta["project_sample_id"],
        CATALYZED_COLUMNS_ID: row_meta[CATALYZED_COLUMNS_ID],
        TIME_COL_COLUMNS: t,
        "input_uncertainty_score": pd.to_numeric(row_meta.get("input_uncertainty_score", np.nan), errors="coerce"),
        "material_size_p80_in": pd.to_numeric(row_meta.get("material_size_p80_in", np.nan), errors="coerce"),
        "column_height_m": pd.to_numeric(row_meta.get("column_height_m", np.nan), errors="coerce"),
        "column_inner_diameter_m": pd.to_numeric(row_meta.get("column_inner_diameter_m", np.nan), errors="coerce"),
        "residual_cpy_%": pd.to_numeric(row_meta.get("residual_cpy_%", np.nan), errors="coerce"),
        "transition_time": pd.to_numeric(row_meta.get("transition_time", np.nan), errors="coerce"),
        "cumulative_catalyst_addition_kg_t": c_prof,
        "y_true": y_true,
        "y_pred_mean": y_pred_mean,
        "y_pred_p05": y_pred_low,
        "y_pred_p95": y_pred_high,
        "pred_a1": p_mean[0],
        "pred_b1": p_mean[1],
        "pred_a2": p_mean[2],
        "pred_b2": p_mean[3],
        "pred_a1_p05": p_low[0],
        "pred_b1_p05": p_low[1],
        "pred_a2_p05": p_low[2],
        "pred_b2_p05": p_low[3],
        "pred_a1_p95": p_high[0],
        "pred_b1_p95": p_high[1],
        "pred_a2_p95": p_high[2],
        "pred_b2_p95": p_high[3],
    }
    for feat in ORE_SIMILARITY_CANDIDATES:
        if feat in row_meta.index and feat not in rec:
            rec[feat] = pd.to_numeric(row_meta.get(feat, np.nan), errors="coerce")
    return pd.DataFrame(rec)


def _enforce_output_curve_pair_constraints(
    control_df: pd.DataFrame,
    catalyzed_df: pd.DataFrame,
    coupling_cfg: Dict,
    shaping_cfg: Dict = None,
) -> pd.DataFrame:
    if control_df.empty or catalyzed_df.empty:
        return catalyzed_df
    shaping_cfg = shaping_cfg or {}

    ctrl = control_df.sort_values(TIME_COL_COLUMNS).copy()
    cat = catalyzed_df.sort_values(TIME_COL_COLUMNS).copy()
    tc = ctrl[TIME_COL_COLUMNS].to_numpy(dtype=float)
    ta = cat[TIME_COL_COLUMNS].to_numpy(dtype=float)
    if tc.size == 0 or ta.size == 0:
        return catalyzed_df

    pred_cols = ["y_pred_mean", "y_pred_p05", "y_pred_p95"]
    enforce_above = coupling_cfg.get("enforce_catalyzed_above_control", True)
    touch_first = coupling_cfg.get("touch_first_point", True)
    max_gap = max(0.0, float(coupling_cfg.get("first_point_max_gap", 0.0)))
    catalyst_profile = (
        cat["cumulative_catalyst_addition_kg_t"].to_numpy(dtype=float)
        if "cumulative_catalyst_addition_kg_t" in cat.columns
        else np.zeros(len(cat), dtype=float)
    )
    transition_time = pd.to_numeric(cat["transition_time"].iloc[0], errors="coerce") if "transition_time" in cat.columns else np.nan
    p80_in = pd.to_numeric(cat["material_size_p80_in"].iloc[0], errors="coerce") if "material_size_p80_in" in cat.columns else np.nan
    h_m = pd.to_numeric(cat["column_height_m"].iloc[0], errors="coerce") if "column_height_m" in cat.columns else np.nan
    d_m = pd.to_numeric(cat["column_inner_diameter_m"].iloc[0], errors="coerce") if "column_inner_diameter_m" in cat.columns else np.nan
    residual_cpy_pct = pd.to_numeric(cat["residual_cpy_%"].iloc[0], errors="coerce") if "residual_cpy_%" in cat.columns else np.nan
    row_context = cat.iloc[0] if len(cat) > 0 else None

    for col in pred_cols:
        vc = ctrl[col].to_numpy(dtype=float)
        va = cat[col].to_numpy(dtype=float)
        vc_i = np.interp(ta, tc, vc, left=vc[0], right=vc[-1])
        va = _apply_time_dependent_catalyst_separation(
            control_curve=vc_i,
            catalyzed_curve=va,
            catalyst_profile=catalyst_profile,
            time_days=ta,
            transition_time=transition_time,
            material_size_p80_in=p80_in,
            column_height_m=h_m,
            column_inner_diameter_m=d_m,
            residual_cpy_pct=residual_cpy_pct,
            row_context=row_context,
            shaping_cfg=shaping_cfg,
            coupling_cfg=coupling_cfg,
        )
        if enforce_above:
            va = np.maximum(va, vc_i)
        if touch_first and va.size > 0:
            va[0] = min(va[0], vc_i[0] + max_gap)
            if enforce_above:
                va[0] = max(va[0], vc_i[0])
        cat[col] = va

    cat["y_pred_p05"] = np.minimum(cat["y_pred_p05"], cat["y_pred_mean"])
    cat["y_pred_p95"] = np.maximum(cat["y_pred_p95"], cat["y_pred_mean"])
    return cat.sort_index()


def infer_column_curves_from_characterization_and_reactors(
    t_days: np.ndarray,
    reactor_control_params: np.ndarray,
    reactor_catalyzed_params: np.ndarray,
    catalyst_profile: np.ndarray = None,
    material_size_p80_in: float = np.nan,
    column_height_m: float = np.nan,
    column_inner_diameter_m: float = np.nan,
    residual_cpy_pct: float = np.nan,
    transition_time_days: float = np.nan,
    reactor_norm_uplift_prior_pct: float = 0.0,
    curve_cfg: Dict = None,
    coupling_cfg: Dict = None,
    shaping_cfg: Dict = None,
    uncertainty_cfg: Dict = None,
    ensemble_seeds: List[int] = None,
    pi_lower: float = None,
    pi_upper: float = None,
) -> Dict[str, np.ndarray]:
    """
    Inference utility for new ores using only reactor-derived information.
    Returns Control and Catalyzed column recovery curves on the provided time grid.
    """
    curve_cfg = curve_cfg or CONFIG["curve_model"]
    coupling_cfg = coupling_cfg or CONFIG["catalyst_curve_coupling"]
    uncertainty_cfg = dict(uncertainty_cfg or CONFIG.get("inference_uncertainty", {}))
    if pi_lower is None:
        pi_lower = float(CONFIG.get("ensemble", {}).get("pi_lower", 5))
    if pi_upper is None:
        pi_upper = float(CONFIG.get("ensemble", {}).get("pi_upper", 95))
    shaping_cfg = _prepare_inference_shaping_config(
        pd.DataFrame(
            {
                "material_size_p80_in": [material_size_p80_in],
                "column_height_m": [column_height_m],
                "column_inner_diameter_m": [column_inner_diameter_m],
                "residual_cpy_%": [residual_cpy_pct],
                "transition_time": [transition_time_days],
            }
        ),
        shaping_cfg or CONFIG.get("inference_shaping", {}),
    )
    t = np.asarray(t_days, dtype=float)
    if t.size == 0:
        return {"time_days": np.asarray([], dtype=float), "control_curve": np.asarray([], dtype=float), "catalyzed_curve": np.asarray([], dtype=float)}

    p_ctrl = _sanitize_curve_params(
        reactor_control_params[0],
        reactor_control_params[1],
        reactor_control_params[2],
        reactor_control_params[3],
        total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
        b_upper=curve_cfg["fit_b_fast_upper"],
    )
    p_cat = _sanitize_curve_params(
        reactor_catalyzed_params[0],
        reactor_catalyzed_params[1],
        reactor_catalyzed_params[2],
        reactor_catalyzed_params[3],
        total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
        b_upper=curve_cfg["fit_b_fast_upper"],
    )

    c_prof = _align_profile_to_time_length(
        np.asarray(catalyst_profile if catalyst_profile is not None else np.zeros_like(t), dtype=float),
        len(t),
    )
    c_final = float(max(0.0, c_prof[-1])) if c_prof.size else 0.0
    c_auc = float(np.trapezoid(c_prof, t)) if c_prof.size >= 2 else 0.0
    c_avg = float(c_auc / np.max(t)) if c_prof.size and np.max(t) > 0 else 0.0

    row_ctrl = pd.Series(
        {
            "material_size_p80_in": material_size_p80_in,
            "column_height_m": column_height_m,
            "column_inner_diameter_m": column_inner_diameter_m,
            "residual_cpy_%": residual_cpy_pct,
            "catalyst_final_kg_t": 0.0,
            "catalyst_auc_kg_t_day": 0.0,
            "curve_max_time_days": float(np.max(t)) if t.size else np.nan,
            "transition_time": transition_time_days,
        }
    )
    row_cat = pd.Series(
        {
            "material_size_p80_in": material_size_p80_in,
            "column_height_m": column_height_m,
            "column_inner_diameter_m": column_inner_diameter_m,
            "residual_cpy_%": residual_cpy_pct,
            "catalyst_final_kg_t": c_final,
            "catalyst_auc_kg_t_day": c_auc if np.isfinite(c_auc) else c_avg * float(np.max(t)),
            "curve_max_time_days": float(np.max(t)) if t.size else np.nan,
            "transition_time": transition_time_days,
        }
    )
    p_ctrl = _apply_material_and_catalyst_shaping(
        params=p_ctrl,
        status="Control",
        row=row_ctrl,
        shaping_cfg=shaping_cfg,
        fit_total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
        fit_b_fast_upper=curve_cfg["fit_b_fast_upper"],
    )
    p_cat = _apply_material_and_catalyst_shaping(
        params=p_cat,
        status="Catalyzed",
        row=row_cat,
        shaping_cfg=shaping_cfg,
        fit_total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
        fit_b_fast_upper=curve_cfg["fit_b_fast_upper"],
    )

    p_cat = _constrain_catalyzed_params_against_control(
        control_params=p_ctrl,
        catalyzed_params=p_cat,
        t_days=t,
        catalyst_profile=c_prof,
        reactor_norm_uplift_prior_pct=float(reactor_norm_uplift_prior_pct),
        coupling_cfg=coupling_cfg,
        fit_total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
        fit_b_fast_upper=curve_cfg["fit_b_fast_upper"],
        row_context=row_cat,
        shaping_cfg=shaping_cfg,
    )

    sample_df = pd.DataFrame(
        [
            {
                "project_sample_id": "inference_sample",
                CATALYZED_COLUMNS_ID: "Control",
                "curve_time_days": t.tolist(),
                "curve_target_recovery": np.zeros_like(t, dtype=float).tolist(),
                "curve_catalyst_profile": np.zeros_like(t, dtype=float).tolist(),
                "curve_max_time_days": float(np.max(t)) if t.size else np.nan,
                "material_size_p80_in": material_size_p80_in,
                "column_height_m": column_height_m,
                "column_inner_diameter_m": column_inner_diameter_m,
                "residual_cpy_%": residual_cpy_pct,
                "transition_time": transition_time_days,
                "catalyst_final_kg_t": 0.0,
                "reactor_sim_a1": float(p_ctrl[0]),
                "reactor_sim_b1": float(p_ctrl[1]),
                "reactor_sim_a2": float(p_ctrl[2]),
                "reactor_sim_b2": float(p_ctrl[3]),
                "reactor_sim_mean_distance": 0.0,
                "reactor_norm_uplift_prior_pct": 0.0,
            },
            {
                "project_sample_id": "inference_sample",
                CATALYZED_COLUMNS_ID: "Catalyzed",
                "curve_time_days": t.tolist(),
                "curve_target_recovery": np.zeros_like(t, dtype=float).tolist(),
                "curve_catalyst_profile": c_prof.tolist(),
                "curve_max_time_days": float(np.max(t)) if t.size else np.nan,
                "material_size_p80_in": material_size_p80_in,
                "column_height_m": column_height_m,
                "column_inner_diameter_m": column_inner_diameter_m,
                "residual_cpy_%": residual_cpy_pct,
                "transition_time": transition_time_days,
                "catalyst_final_kg_t": c_final,
                "reactor_sim_a1": float(p_cat[0]),
                "reactor_sim_b1": float(p_cat[1]),
                "reactor_sim_a2": float(p_cat[2]),
                "reactor_sim_b2": float(p_cat[3]),
                "reactor_sim_mean_distance": 0.0,
                "reactor_norm_uplift_prior_pct": float(reactor_norm_uplift_prior_pct),
            },
        ]
    )
    sample_df = _attach_uncertainty_scores(sample_df, shaping_cfg=shaping_cfg, unc_cfg=uncertainty_cfg)
    uncertainty_cfg = _prepare_inference_uncertainty_config(sample_df, uncertainty_cfg)

    seeds_in = ensemble_seeds if ensemble_seeds is not None else CONFIG.get("ensemble", {}).get("seeds", [1, 11, 21, 31, 41])
    ensemble_seeds = _resolve_ensemble_seeds(seeds_in, int(uncertainty_cfg.get("min_ensemble_members", 5)))
    base_pred_by_idx = {sample_df.index[0]: p_ctrl, sample_df.index[1]: p_cat}
    member_param_preds = _generate_member_param_predictions_from_base(
        sample_df=sample_df,
        base_pred_by_idx=base_pred_by_idx,
        ensemble_seeds=ensemble_seeds,
        curve_cfg=curve_cfg,
        unc_cfg=uncertainty_cfg,
        shaping_cfg=shaping_cfg,
    )
    member_params_by_idx = _prepare_member_params_for_sample(
        sample_df=sample_df,
        member_param_preds=member_param_preds,
        curve_cfg=curve_cfg,
        coupling_cfg=coupling_cfg,
        shaping_cfg=shaping_cfg,
    )

    ctrl_idx = sample_df.index[0]
    cat_idx = sample_df.index[1]
    y_ctrl_members = []
    y_cat_members = []
    for m in range(len(ensemble_seeds)):
        p_ctrl_m = member_params_by_idx[ctrl_idx][m]
        p_cat_m = member_params_by_idx[cat_idx][m]
        y_ctrl_m = construct_reactor_recovery_curve(p_ctrl_m[0], p_ctrl_m[1], p_ctrl_m[2], p_ctrl_m[3], t)
        y_cat_m = construct_reactor_recovery_curve(p_cat_m[0], p_cat_m[1], p_cat_m[2], p_cat_m[3], t)
        y_cat_m = _apply_time_dependent_catalyst_separation(
            control_curve=y_ctrl_m,
            catalyzed_curve=y_cat_m,
            catalyst_profile=c_prof,
            time_days=t,
            transition_time=transition_time_days,
            material_size_p80_in=material_size_p80_in,
            column_height_m=column_height_m,
            column_inner_diameter_m=column_inner_diameter_m,
            residual_cpy_pct=residual_cpy_pct,
            row_context=row_cat,
            shaping_cfg=shaping_cfg,
            coupling_cfg=coupling_cfg,
        )
        y_ctrl_members.append(y_ctrl_m)
        y_cat_members.append(y_cat_m)
    y_ctrl_members = np.vstack(y_ctrl_members)
    y_cat_members = np.vstack(y_cat_members)

    y_ctrl = np.mean(y_ctrl_members, axis=0)
    y_cat = np.mean(y_cat_members, axis=0)
    y_ctrl_p05 = np.percentile(y_ctrl_members, pi_lower, axis=0)
    y_ctrl_p95 = np.percentile(y_ctrl_members, pi_upper, axis=0)
    y_cat_p05 = np.percentile(y_cat_members, pi_lower, axis=0)
    y_cat_p95 = np.percentile(y_cat_members, pi_upper, axis=0)
    p_ctrl_mean = np.mean(member_params_by_idx[ctrl_idx], axis=0)
    p_cat_mean = np.mean(member_params_by_idx[cat_idx], axis=0)

    return {
        "time_days": t,
        "control_curve": y_ctrl,
        "catalyzed_curve": y_cat,
        "control_curve_p05": y_ctrl_p05,
        "control_curve_p95": y_ctrl_p95,
        "catalyzed_curve_p05": y_cat_p05,
        "catalyzed_curve_p95": y_cat_p95,
        "control_params": p_ctrl_mean,
        "catalyzed_params": p_cat_mean,
        "control_param_members": member_params_by_idx[ctrl_idx],
        "catalyzed_param_members": member_params_by_idx[cat_idx],
        "input_uncertainty_score_control": float(sample_df.loc[ctrl_idx, "input_uncertainty_score"]),
        "input_uncertainty_score_catalyzed": float(sample_df.loc[cat_idx, "input_uncertainty_score"]),
        "ensemble_seeds_used": ensemble_seeds,
    }


def resolve_features(df: pd.DataFrame, headers_dict: Dict[str, List]) -> Tuple[List[str], List[str]]:
    missing = [col for col in headers_dict if col not in df.columns]
    if missing:
        print(f"Warning: missing features dropped: {missing}")
    features = [col for col in headers_dict if col in df.columns]
    categorical = [
        col
        for col, meta in headers_dict.items()
        if meta[1] == "categorical" and col in df.columns
    ]
    return features, categorical


def expand_reactors_long(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    time_col: str = "leach_duration_days_const",
    id_col: str = "project_sample_id",
    catalyst_id_col: str = None,
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        leach = parse_listlike(row.get(time_col, None))
        target = parse_listlike(row.get(target_col, None))
        n = min(len(leach), len(target))
        if n == 0:
            continue
        leach = leach[:n]
        target = target[:n]
        mask = np.isfinite(leach) & np.isfinite(target)
        if not mask.any():
            continue
        for i in np.where(mask)[0]:
            rec = {time_col: float(leach[i]), target_col: float(target[i])}
            for feat in feature_cols:
                if feat == time_col:
                    continue
                rec[feat] = row.get(feat, np.nan)
            rec[id_col] = row.get(id_col)
            if catalyst_id_col and catalyst_id_col in df.columns:
                rec[catalyst_id_col] = row.get(catalyst_id_col)
            rows.append(rec)
    return pd.DataFrame(rows)


def expand_columns_long(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    time_col: str = "leach_duration_days",
    catalyst_col: str = "cumulative_catalyst_addition_kg_t",
    id_col: str = "project_sample_id",
    col_id: str = "project_col_id",
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        leach = parse_listlike(row.get(time_col, None))
        target = parse_listlike(row.get(target_col, None))
        catalyst = parse_listlike(row.get(catalyst_col, None)) if catalyst_col in df.columns else None
        n = min(len(leach), len(target))
        if n == 0:
            continue
        leach = leach[:n]
        target = target[:n]
        if catalyst is None or len(catalyst) == 0:
            catalyst = np.zeros(n, dtype=float)
        else:
            n = min(n, len(catalyst))
            leach = leach[:n]
            target = target[:n]
            catalyst = catalyst[:n]

        mask = np.isfinite(leach) & np.isfinite(target)
        if not mask.any():
            continue

        col_id_value = row.get(col_id, "")
        is_catalyzed = str(col_id_value).lower().startswith("catalyzed")
        if not is_catalyzed:
            catalyst = np.zeros_like(catalyst)
        else:
            catalyst = np.where(np.isfinite(catalyst) & (catalyst > 0), catalyst, 0.0)

        for i in np.where(mask)[0]:
            rec = {
                time_col: float(leach[i]),
                target_col: float(target[i]),
                catalyst_col: float(catalyst[i]),
            }
            for feat in feature_cols:
                if feat in (time_col, catalyst_col):
                    continue
                rec[feat] = row.get(feat, np.nan)
            rec[id_col] = row.get(id_col)
            rec[col_id] = col_id_value
            rows.append(rec)
    return pd.DataFrame(rows)


def _interp_at(times: np.ndarray, values: np.ndarray, t: float) -> float:
    mask = np.isfinite(times) & np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    times = times[mask]
    values = values[mask]
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    if t < times[0] or t > times[-1]:
        return np.nan
    return float(np.interp(t, times, values))


def _normalize_time(t: float, t_max: float, mode: str = "log") -> float:
    if not np.isfinite(t) or not np.isfinite(t_max) or t_max <= 0:
        return np.nan
    t_clamped = min(max(t, 0.0), t_max)
    if mode == "log":
        return float(np.log1p(t_clamped) / np.log1p(t_max))
    return float(t_clamped / t_max)


def _compute_time_max(df: pd.DataFrame, time_col: str) -> float:
    t_max = 0.0
    for v in df[time_col].values:
        arr = parse_listlike(v)
        if arr.size:
            arr_max = np.nanmax(arr)
            if np.isfinite(arr_max):
                t_max = max(t_max, float(arr_max))
    return t_max if t_max > 0 else np.nan


def build_reactor_delta_library(
    df_reactors: pd.DataFrame,
    time_col: str,
    target_col: str,
    id_col: str,
    status_col: str,
    shared_static_feats: List[str],
    time_max: float,
    time_norm_mode: str,
) -> List[Dict]:
    library = []
    for sample_id, group in df_reactors.groupby(id_col):
        control_row = group[group[status_col].astype(str).str.lower() == "control"]
        catalyzed_row = group[group[status_col].astype(str).str.lower() == "catalyzed"]
        if control_row.empty or catalyzed_row.empty:
            continue
        control_row = control_row.iloc[0]
        catalyzed_row = catalyzed_row.iloc[0]

        t_ctrl = parse_listlike(control_row.get(time_col, None))
        y_ctrl = parse_listlike(control_row.get(target_col, None))
        t_cat = parse_listlike(catalyzed_row.get(time_col, None))
        y_cat = parse_listlike(catalyzed_row.get(target_col, None))
        if len(t_ctrl) == 0 or len(t_cat) == 0:
            continue

        grid = np.unique(np.concatenate([t_ctrl, t_cat]))
        if len(grid) < 2:
            continue

        ctrl_interp = np.array([_interp_at(t_ctrl, y_ctrl, t) for t in grid], dtype=float)
        cat_interp = np.array([_interp_at(t_cat, y_cat, t) for t in grid], dtype=float)
        delta = cat_interp - ctrl_interp

        grid_norm = np.array([_normalize_time(t, time_max, time_norm_mode) for t in grid], dtype=float)
        valid = np.isfinite(grid_norm) & np.isfinite(delta)
        if valid.sum() < 2:
            continue
        grid_norm = grid_norm[valid]
        delta = delta[valid]
        order = np.argsort(grid_norm)
        grid_norm = grid_norm[order]
        delta = delta[order]

        feat_vals = {}
        for feat in shared_static_feats:
            feat_vals[feat] = control_row.get(feat, np.nan)

        library.append(
            {
                "project_sample_id": sample_id,
                "time_norm": grid_norm,
                "delta": delta,
                "features": feat_vals,
            }
        )
    return library


def compute_reactor_delta_prior(
    df_columns_long: pd.DataFrame,
    reactor_library: List[Dict],
    shared_static_feats: List[str],
    time_col: str,
    id_col: str,
    status_col: str,
    p80_col: str,
    config: Dict,
    reactor_time_max: float,
) -> pd.Series:
    if not reactor_library or not shared_static_feats:
        return pd.Series(np.zeros(len(df_columns_long)), index=df_columns_long.index)

    reactor_feat_df = pd.DataFrame([lib["features"] for lib in reactor_library])
    reactor_feat_df = reactor_feat_df[shared_static_feats].apply(pd.to_numeric, errors="coerce")

    reactor_imputer = SimpleImputer(strategy="median")
    reactor_imputer.fit(reactor_feat_df)
    reactor_feat_imp = reactor_imputer.transform(reactor_feat_df)
    reactor_scaler = StandardScaler()
    reactor_scaled = reactor_scaler.fit_transform(reactor_feat_imp)

    sample_static = (
        df_columns_long.groupby(id_col)[shared_static_feats]
        .first()
        .apply(pd.to_numeric, errors="coerce")
    )
    sample_scaled = reactor_scaler.transform(reactor_imputer.transform(sample_static))

    k = min(config["k_neighbors"], len(reactor_library))
    eps = config["distance_eps"]

    neighbor_map = {}
    for i, sample_id in enumerate(sample_static.index):
        dists = np.linalg.norm(reactor_scaled - sample_scaled[i], axis=1)
        idx = np.argsort(dists)[:k]
        weights = 1.0 / (dists[idx] + eps)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        neighbor_map[sample_id] = (idx, weights)

    p80_alpha = config.get("p80_dampening_alpha", 0.0)
    p80_min = float(df_columns_long[p80_col].min()) if p80_col in df_columns_long.columns else np.nan
    p80_max = float(df_columns_long[p80_col].max()) if p80_col in df_columns_long.columns else np.nan
    time_norm_mode = config.get("time_norm", "log")

    priors = []
    for _, row in df_columns_long.iterrows():
        sample_id = row[id_col]
        if str(row[status_col]).lower() != "catalyzed":
            priors.append(0.0)
            continue
        t = float(row[time_col])
        t_norm = _normalize_time(t, reactor_time_max, time_norm_mode)
        idx, weights = neighbor_map.get(sample_id, ([], []))
        if len(idx) == 0:
            priors.append(0.0)
            continue
        delta_vals = []
        for j in idx:
            lib = reactor_library[j]
            lib_time = lib["time_norm"]
            if lib_time.size == 0 or not np.isfinite(t_norm):
                delta_val = np.nan
            else:
                t_query = min(max(t_norm, lib_time[0]), lib_time[-1])
                delta_val = _interp_at(lib_time, lib["delta"], t_query)
            delta_vals.append(delta_val)
        delta_vals = np.array(delta_vals, dtype=float)
        valid = np.isfinite(delta_vals)
        if not valid.any():
            priors.append(0.0)
            continue
        weights_valid = weights[valid]
        weights_valid = weights_valid / weights_valid.sum() if weights_valid.sum() > 0 else np.ones_like(weights_valid) / len(weights_valid)
        delta_prior = float(np.sum(delta_vals[valid] * weights_valid))

        if p80_alpha > 0 and p80_col in df_columns_long.columns and np.isfinite(p80_min) and np.isfinite(p80_max) and p80_max > p80_min:
            p80_val = row.get(p80_col, np.nan)
            if np.isfinite(p80_val):
                p80_norm = (float(p80_val) - p80_min) / (p80_max - p80_min)
                damp = 1.0 - p80_alpha * p80_norm
                damp = max(config.get("p80_dampening_min", 0.2), min(1.0, damp))
                delta_prior *= damp

        priors.append(delta_prior)

    return pd.Series(priors, index=df_columns_long.index)


def prepare_feature_matrix(
    df_long: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    categorical_cols: List[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = df_long.copy()
    categorical_cols = set(categorical_cols or [])
    for col in feature_cols:
        if col not in df.columns:
            continue
        if col in categorical_cols:
            df[col] = df[col].astype(str).fillna("Unknown")
            df[col] = pd.Categorical(df[col]).codes
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")
    valid_mask = np.isfinite(y.values)
    df = df.loc[valid_mask].copy()
    y = y.loc[valid_mask]
    X = df[feature_cols].copy()
    return X, y, df


def fit_preprocessor(X: pd.DataFrame) -> Tuple[SimpleImputer, StandardScaler, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)
    return imputer, scaler, X_scaled


def apply_preprocessor(X: pd.DataFrame, imputer: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    X_imp = imputer.transform(X)
    return scaler.transform(X_imp)


def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float(),
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_hidden_layers: int, dropout: float):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        layers = []
        for _ in range(max(num_hidden_layers - 1, 0)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc_in(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.out(x)
        return x.squeeze(1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
) -> Tuple[nn.Module, float]:
    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    patience = config["patience"]
    epochs = config["epochs"]
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if val_loader is None:
            continue

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X).float().to(device)).cpu().numpy()
    return preds


def train_columns_member(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    columns_features: List[str],
    reactor_model: nn.Module,
    reactor_features: List[str],
    config: Dict,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, nn.Module, SimpleImputer, StandardScaler]:
    set_all_seeds(seed, deterministic=True)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=config["val_split"], random_state=seed
    )
    imp_c, scaler_c, X_train_scaled = fit_preprocessor(X_train_split)
    X_val_scaled = apply_preprocessor(X_val_split, imp_c, scaler_c)
    X_test_scaled = apply_preprocessor(X_test, imp_c, scaler_c)

    train_loader_c, val_loader_c = make_dataloaders(
        X_train_scaled,
        y_train_split.values,
        X_val_scaled,
        y_val_split.values,
        batch_size=min(config["batch_size"], len(X_train_scaled)),
    )

    columns_model = MLPRegressor(
        in_dim=len(columns_features),
        hidden_dim=config["hidden_dim"],
        num_hidden_layers=config["num_hidden_layers"],
        dropout=config["dropout"],
    )
    transfer_weights(
        reactor_model,
        columns_model,
        reactor_features,
        columns_features,
        copy_hidden_layers=bool(config.get("transfer_copy_hidden_layers", False)),
        copy_output_layer=bool(config.get("transfer_copy_output_layer", False)),
    )

    if config["freeze_pretrained_layers"]:
        for param in columns_model.fc_in.parameters():
            param.requires_grad = False

    columns_model, _ = train_model(
        columns_model, train_loader_c, val_loader_c, config, device
    )

    y_pred = predict(columns_model, X_test_scaled, device)
    return y_pred, columns_model, imp_c, scaler_c


def ensemble_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    columns_features: List[str],
    reactor_model: nn.Module,
    reactor_features: List[str],
    config: Dict,
    ensemble_seeds: List[int],
    pi_lower: float,
    pi_upper: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict], np.ndarray]:
    preds_list = []
    members = []
    for seed in ensemble_seeds:
        y_pred, model, imputer, scaler = train_columns_member(
            X_train,
            y_train,
            X_test,
            columns_features,
            reactor_model,
            reactor_features,
            config,
            seed,
            device,
        )
        preds_list.append(y_pred)
        members.append(
            {
                "seed": seed,
                "model": model,
                "imputer": imputer,
                "scaler": scaler,
            }
        )

    preds_arr = np.vstack(preds_list)
    mean_pred = np.mean(preds_arr, axis=0)
    lower = np.percentile(preds_arr, pi_lower, axis=0)
    upper = np.percentile(preds_arr, pi_upper, axis=0)
    return mean_pred, lower, upper, members, preds_arr


def transfer_weights(
    reactor_model: nn.Module,
    columns_model: nn.Module,
    reactor_features: List[str],
    columns_features: List[str],
    copy_hidden_layers: bool = True,
    copy_output_layer: bool = True,
) -> List[str]:
    if reactor_model is None or not reactor_features:
        return []
    shared = [feat for feat in columns_features if feat in reactor_features]
    with torch.no_grad():
        # Copy input bias
        react_fc_bias = reactor_model.fc_in.bias.detach().cpu()
        if react_fc_bias.shape == columns_model.fc_in.bias.shape:
            columns_model.fc_in.bias.copy_(react_fc_bias)
        # Copy input weights for shared features
        react_fc_weight = reactor_model.fc_in.weight.detach().cpu()
        for col_idx, feat in enumerate(columns_features):
            if feat in reactor_features:
                react_idx = reactor_features.index(feat)
                columns_model.fc_in.weight[:, col_idx].copy_(react_fc_weight[:, react_idx])
        if copy_hidden_layers:
            # Optionally copy hidden layers if same shape.
            for r_layer, c_layer in zip(reactor_model.hidden, columns_model.hidden):
                if (
                    isinstance(r_layer, nn.Linear)
                    and isinstance(c_layer, nn.Linear)
                    and r_layer.weight.shape == c_layer.weight.shape
                ):
                    c_layer.weight.copy_(r_layer.weight.detach().cpu())
                    c_layer.bias.copy_(r_layer.bias.detach().cpu())
        if copy_output_layer and reactor_model.out.weight.shape == columns_model.out.weight.shape:
            # Output transfer is optional because reactor target scale differs from column param targets.
            columns_model.out.weight.copy_(reactor_model.out.weight.detach().cpu())
            columns_model.out.bias.copy_(reactor_model.out.bias.detach().cpu())
    return shared


def train_reactor_transfer_backbone(
    df_reactors_source: pd.DataFrame,
    config: Dict,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, List[str], Dict[str, float]]:
    reactor_features, reactor_categorical = resolve_features(df_reactors_source, HEADERS_DICT_REACTORS)
    if not reactor_features:
        raise ValueError("No reactor features resolved for transfer backbone training.")

    df_reactors_long = expand_reactors_long(
        df_reactors_source,
        feature_cols=reactor_features,
        target_col=TARGET_REACTORS,
        time_col=TIME_COL_REACTORS,
        id_col="project_sample_id",
        catalyst_id_col=CATALYZED_REACTORS_ID,
    )
    if df_reactors_long.empty:
        raise ValueError("No valid reactor long-format rows available for transfer backbone training.")

    X_reactors, y_reactors, _ = prepare_feature_matrix(
        df_reactors_long,
        feature_cols=reactor_features,
        target_col=TARGET_REACTORS,
        categorical_cols=reactor_categorical,
    )
    if X_reactors.empty:
        raise ValueError("Reactor transfer backbone training matrix is empty.")

    set_all_seeds(seed, deterministic=True)
    n_rows = len(X_reactors)
    val_split = float(config.get("val_split", 0.25))
    use_val_split = n_rows >= 8 and val_split > 0.0
    if use_val_split:
        X_train, X_val, y_train, y_val = train_test_split(
            X_reactors,
            y_reactors,
            test_size=val_split,
            random_state=seed,
        )
    else:
        X_train, y_train = X_reactors, y_reactors
        X_val, y_val = None, None

    imputer_r, scaler_r, X_train_scaled = fit_preprocessor(X_train)
    if X_val is not None and y_val is not None and len(X_val) > 0:
        X_val_scaled = apply_preprocessor(X_val, imputer_r, scaler_r)
    else:
        X_val_scaled, y_val = None, None

    train_loader_r, val_loader_r = make_dataloaders(
        X_train_scaled,
        y_train.values,
        X_val_scaled,
        None if y_val is None else y_val.values,
        batch_size=min(int(config["batch_size"]), max(1, len(X_train_scaled))),
    )

    reactor_model = MLPRegressor(
        in_dim=len(reactor_features),
        hidden_dim=int(config["hidden_dim"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        dropout=float(config["dropout"]),
    )
    reactor_model, best_val_loss = train_model(
        reactor_model,
        train_loader_r,
        val_loader_r,
        config,
        device,
    )

    train_pred = predict(reactor_model, X_train_scaled, device)
    train_metrics = compute_metrics(y_train.values.astype(float), train_pred.astype(float))
    val_metrics = {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan}
    if X_val_scaled is not None and y_val is not None and len(y_val) > 0:
        val_pred = predict(reactor_model, X_val_scaled, device)
        val_metrics = compute_metrics(y_val.values.astype(float), val_pred.astype(float))

    summary = {
        "n_long_rows": int(n_rows),
        "n_train_rows": int(len(X_train)),
        "n_val_rows": int(0 if y_val is None else len(y_val)),
        "best_val_loss_mse": float(best_val_loss if np.isfinite(best_val_loss) else np.nan),
        "train_rmse": float(train_metrics["rmse"]),
        "train_mae": float(train_metrics["mae"]),
        "train_r2": float(train_metrics["r2"]),
        "train_bias": float(train_metrics["bias"]),
        "val_rmse": float(val_metrics["rmse"]),
        "val_mae": float(val_metrics["mae"]),
        "val_r2": float(val_metrics["r2"]),
        "val_bias": float(val_metrics["bias"]),
    }
    return reactor_model, reactor_features, summary


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    if len(np.unique(y_true)) > 1:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = float("nan")
    bias = float(np.mean(y_pred - y_true))
    return {"rmse": rmse, "mae": mae, "r2": r2, "bias": bias}


def plot_sample_predictions(
    df_sample: pd.DataFrame,
    sample_id: str,
    plot_path: str,
    time_col: str,
    target_col: str,
    title_prefix: str = "LOOCV Prediction",
    pred_col: str = "y_pred_mean",
    lower_col: str = "y_pred_p05",
    upper_col: str = "y_pred_p95",
    status_col: str = "project_col_id",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    status_metrics: Dict[str, Dict[str, float]] = {}

    def _interp_on_grid(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return np.full_like(x_grid, np.nan, dtype=float)
        tmp = pd.DataFrame({"x": x[mask], "y": y[mask]}).groupby("x", as_index=False)["y"].mean().sort_values("x")
        xv = tmp["x"].to_numpy(dtype=float)
        yv = tmp["y"].to_numpy(dtype=float)
        if xv.size == 1:
            return np.full_like(x_grid, float(yv[0]), dtype=float)
        return np.interp(x_grid, xv, yv, left=float(yv[0]), right=float(yv[-1]))

    def _extract_pred_params(subset: pd.DataFrame, suffix: str = "") -> np.ndarray:
        req = [f"pred_a1{suffix}", f"pred_b1{suffix}", f"pred_a2{suffix}", f"pred_b2{suffix}"]
        if not all(col in subset.columns for col in req):
            return None
        p = subset[req].apply(pd.to_numeric, errors="coerce").dropna()
        if p.empty:
            return None
        return p.iloc[0].to_numpy(dtype=float)

    def _first_finite(subset: pd.DataFrame, col: str, default: float = np.nan) -> float:
        if col not in subset.columns:
            return default
        vals = pd.to_numeric(subset[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(vals[0]) if vals.size else default

    t_all = pd.to_numeric(df_sample[time_col], errors="coerce").to_numpy(dtype=float) if time_col in df_sample.columns else np.asarray([], dtype=float)
    t_all = t_all[np.isfinite(t_all)]
    t_max = float(np.nanmax(t_all)) if t_all.size else np.nan
    if np.isfinite(t_max) and t_max > 0:
        n_grid = int(np.clip(max(200, np.ceil(t_max) + 1), 200, 800))
        t_grid = np.linspace(0.0, t_max, n_grid)
    elif np.isfinite(t_max):
        t_grid = np.array([0.0], dtype=float)
    else:
        t_grid = np.asarray([], dtype=float)

    control_subset = df_sample[df_sample[status_col] == "Control"].sort_values(time_col)
    control_curve_grid = None
    control_curve_grid_low = None
    control_curve_grid_high = None
    if not control_subset.empty and t_grid.size > 0:
        control_params = _extract_pred_params(control_subset, "")
        if control_params is not None:
            control_curve_grid = _double_exp_curve(
                control_params[0],
                control_params[1],
                control_params[2],
                control_params[3],
                t_grid,
            )
        elif pred_col in control_subset.columns:
            control_curve_grid = _interp_on_grid(
                pd.to_numeric(control_subset[time_col], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(control_subset[pred_col], errors="coerce").to_numpy(dtype=float),
                t_grid,
            )

        control_params_low = _extract_pred_params(control_subset, "_p05")
        if control_params_low is not None:
            control_curve_grid_low = _double_exp_curve(
                control_params_low[0],
                control_params_low[1],
                control_params_low[2],
                control_params_low[3],
                t_grid,
            )
        elif lower_col in control_subset.columns:
            control_curve_grid_low = _interp_on_grid(
                pd.to_numeric(control_subset[time_col], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(control_subset[lower_col], errors="coerce").to_numpy(dtype=float),
                t_grid,
            )

        control_params_high = _extract_pred_params(control_subset, "_p95")
        if control_params_high is not None:
            control_curve_grid_high = _double_exp_curve(
                control_params_high[0],
                control_params_high[1],
                control_params_high[2],
                control_params_high[3],
                t_grid,
            )
        elif upper_col in control_subset.columns:
            control_curve_grid_high = _interp_on_grid(
                pd.to_numeric(control_subset[time_col], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(control_subset[upper_col], errors="coerce").to_numpy(dtype=float),
                t_grid,
            )

        if control_curve_grid_low is None and control_curve_grid is not None:
            control_curve_grid_low = control_curve_grid.copy()
        if control_curve_grid_high is None and control_curve_grid is not None:
            control_curve_grid_high = control_curve_grid.copy()

    for status, color in [("Control", "tab:blue"), ("Catalyzed", "tab:orange")]:
        subset = df_sample[df_sample[status_col] == status].sort_values(time_col)
        if subset.empty:
            continue
        row_context = subset.iloc[0]

        t_obs = pd.to_numeric(subset[time_col], errors="coerce").to_numpy(dtype=float)
        y_true_obs = pd.to_numeric(subset[target_col], errors="coerce").to_numpy(dtype=float)
        y_pred_obs = pd.to_numeric(subset[pred_col], errors="coerce").to_numpy(dtype=float) if pred_col in subset.columns else np.asarray([], dtype=float)

        if pred_col in subset.columns:
            valid = np.isfinite(y_true_obs) & np.isfinite(y_pred_obs)
            if valid.any():
                status_metrics[status] = compute_metrics(y_true_obs[valid], y_pred_obs[valid])

        ax.scatter(
            t_obs,
            y_true_obs,
            color=color,
            alpha=0.75,
            marker="o",
            label=f"{status} Actual",
        )

        grid_plot_mask = np.ones_like(t_grid, dtype=bool) if t_grid.size > 0 else np.asarray([], dtype=bool)
        if status == "Catalyzed" and t_grid.size > 0:
            finite_t_obs = t_obs[np.isfinite(t_obs)]
            if finite_t_obs.size > 0:
                cat_start_t = float(np.nanmin(finite_t_obs))
                grid_plot_mask = t_grid >= cat_start_t
            else:
                grid_plot_mask = np.zeros_like(t_grid, dtype=bool)

        c_grid = None
        transition_time_val = np.nan
        p80_in_val = np.nan
        h_m_val = np.nan
        d_m_val = np.nan
        residual_cpy_val = np.nan
        if status == "Catalyzed" and t_grid.size > 0:
            if "cumulative_catalyst_addition_kg_t" in subset.columns:
                c_obs = pd.to_numeric(subset["cumulative_catalyst_addition_kg_t"], errors="coerce").to_numpy(dtype=float)
                c_grid = _interp_on_grid(t_obs, c_obs, t_grid)
            else:
                c_grid = np.zeros_like(t_grid, dtype=float)
            c_grid = np.maximum.accumulate(np.maximum(np.where(np.isfinite(c_grid), c_grid, 0.0), 0.0))
            transition_time_val = _first_finite(subset, "transition_time", default=np.nan)
            p80_in_val = _first_finite(subset, "material_size_p80_in", default=np.nan)
            h_m_val = _first_finite(subset, "column_height_m", default=np.nan)
            d_m_val = _first_finite(subset, "column_inner_diameter_m", default=np.nan)
            residual_cpy_val = _first_finite(subset, "residual_cpy_%", default=np.nan)

        y_line = None
        if t_grid.size > 0 and pred_col in subset.columns:
            params = _extract_pred_params(subset, "")
            if params is not None:
                y_line = _double_exp_curve(params[0], params[1], params[2], params[3], t_grid)
                if status == "Catalyzed" and control_curve_grid is not None and c_grid is not None:
                    y_line = _apply_time_dependent_catalyst_separation(
                        control_curve=control_curve_grid,
                        catalyzed_curve=y_line,
                        catalyst_profile=c_grid,
                        time_days=t_grid,
                        transition_time=transition_time_val,
                        material_size_p80_in=p80_in_val,
                        column_height_m=h_m_val,
                        column_inner_diameter_m=d_m_val,
                        residual_cpy_pct=residual_cpy_val,
                        row_context=row_context,
                        shaping_cfg=CONFIG.get("inference_shaping", {}),
                        coupling_cfg=CONFIG.get("catalyst_curve_coupling", {}),
                    )
            else:
                y_line = _interp_on_grid(t_obs, y_pred_obs, t_grid)

        if y_line is not None:
            ax.plot(
                t_grid[grid_plot_mask],
                y_line[grid_plot_mask],
                color=color,
                linestyle="-",
                linewidth=1.8,
                label=f"{status} Pred",
            )

        if lower_col in subset.columns and upper_col in subset.columns and t_grid.size > 0:
            params_low = _extract_pred_params(subset, "_p05")
            params_high = _extract_pred_params(subset, "_p95")
            if params_low is not None and params_high is not None:
                y_low = _double_exp_curve(params_low[0], params_low[1], params_low[2], params_low[3], t_grid)
                y_high = _double_exp_curve(params_high[0], params_high[1], params_high[2], params_high[3], t_grid)
                if status == "Catalyzed" and c_grid is not None:
                    base_low = control_curve_grid_low if control_curve_grid_low is not None else control_curve_grid
                    base_high = control_curve_grid_high if control_curve_grid_high is not None else control_curve_grid
                    if base_low is not None:
                        y_low = _apply_time_dependent_catalyst_separation(
                            control_curve=base_low,
                            catalyzed_curve=y_low,
                            catalyst_profile=c_grid,
                            time_days=t_grid,
                            transition_time=transition_time_val,
                            material_size_p80_in=p80_in_val,
                            column_height_m=h_m_val,
                            column_inner_diameter_m=d_m_val,
                            residual_cpy_pct=residual_cpy_val,
                            row_context=row_context,
                            shaping_cfg=CONFIG.get("inference_shaping", {}),
                            coupling_cfg=CONFIG.get("catalyst_curve_coupling", {}),
                        )
                    if base_high is not None:
                        y_high = _apply_time_dependent_catalyst_separation(
                            control_curve=base_high,
                            catalyzed_curve=y_high,
                            catalyst_profile=c_grid,
                            time_days=t_grid,
                            transition_time=transition_time_val,
                            material_size_p80_in=p80_in_val,
                            column_height_m=h_m_val,
                            column_inner_diameter_m=d_m_val,
                            residual_cpy_pct=residual_cpy_val,
                            row_context=row_context,
                            shaping_cfg=CONFIG.get("inference_shaping", {}),
                            coupling_cfg=CONFIG.get("catalyst_curve_coupling", {}),
                        )
            else:
                y_low = _interp_on_grid(
                    t_obs,
                    pd.to_numeric(subset[lower_col], errors="coerce").to_numpy(dtype=float),
                    t_grid,
                )
                y_high = _interp_on_grid(
                    t_obs,
                    pd.to_numeric(subset[upper_col], errors="coerce").to_numpy(dtype=float),
                    t_grid,
                )
            y_low, y_high = np.minimum(y_low, y_high), np.maximum(y_low, y_high)
            if y_line is not None:
                y_low = np.minimum(y_low, y_line)
                y_high = np.maximum(y_high, y_line)
            ax.fill_between(
                t_grid[grid_plot_mask],
                y_low[grid_plot_mask],
                y_high[grid_plot_mask],
                color=color,
                alpha=0.25,
                label=f"{status} 90% PI",
            )

    def _fmt(v: float) -> str:
        return f"{v:.3f}" if np.isfinite(v) else "nan"

    subtitle_parts = []
    for status in ["Control", "Catalyzed"]:
        m = status_metrics.get(status)
        if m is None:
            subtitle_parts.append(f"{status} RMSE=nan, Bias=nan, R2=nan")
        else:
            subtitle_parts.append(
                f"{status} RMSE={_fmt(m['rmse'])}, Bias={_fmt(m['bias'])}, R2={_fmt(m['r2'])}"
            )

    fig.suptitle(f"{title_prefix} - {sample_id}")
    ax.set_title(" | ".join(subtitle_parts), fontsize=9)
    ax.set_xlabel("Leach Duration (days)")
    ax.set_ylabel("Cu Recovery (%)")
    ax.set_xlim(0.0, None)
    ax.set_ylim(0.0, None)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


def _compute_loocv_group_statistics(
    df_long: pd.DataFrame,
    status_col: str = CATALYZED_COLUMNS_ID,
    y_true_col: str = "y_true",
    pred_col: str = "y_pred_mean",
    low_col: str = "y_pred_p05",
    high_col: str = "y_pred_p95",
) -> pd.DataFrame:
    if df_long is None or df_long.empty:
        return pd.DataFrame()

    def _get_numeric_array(frame: pd.DataFrame, col: str) -> np.ndarray:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        return np.full(len(frame), np.nan, dtype=float)

    work = df_long.copy()
    if status_col in work.columns:
        work["_status_norm"] = work[status_col].apply(normalize_status)
    else:
        work["_status_norm"] = "Overall"

    group_masks: List[Tuple[str, np.ndarray]] = [
        ("Overall", np.ones(len(work), dtype=bool)),
        ("Control", (work["_status_norm"] == "Control").to_numpy(dtype=bool)),
        ("Catalyzed", (work["_status_norm"] == "Catalyzed").to_numpy(dtype=bool)),
    ]

    rows = []
    for group_name, mask in group_masks:
        if mask.sum() <= 0:
            rows.append(
                {
                    "group": group_name,
                    "n_pred_points": 0,
                    "rmse": np.nan,
                    "bias": np.nan,
                    "r2": np.nan,
                    "ci_inside_count": 0,
                    "ci_total_count": 0,
                    "ci_outside_count": 0,
                    "ci_coverage_pct": np.nan,
                }
            )
            continue

        sub = work.loc[mask].copy()
        y_true = _get_numeric_array(sub, y_true_col)
        y_pred = _get_numeric_array(sub, pred_col)
        y_low = _get_numeric_array(sub, low_col)
        y_high = _get_numeric_array(sub, high_col)

        valid_pred = np.isfinite(y_true) & np.isfinite(y_pred)
        n_pred = int(valid_pred.sum())
        if n_pred > 0:
            met = compute_metrics(y_true[valid_pred], y_pred[valid_pred])
            rmse_val = float(met["rmse"])
            bias_val = float(met["bias"])
            r2_val = float(met["r2"])
        else:
            rmse_val, bias_val, r2_val = np.nan, np.nan, np.nan

        valid_ci = np.isfinite(y_true) & np.isfinite(y_low) & np.isfinite(y_high)
        n_ci = int(valid_ci.sum())
        if n_ci > 0:
            lo = np.minimum(y_low[valid_ci], y_high[valid_ci])
            hi = np.maximum(y_low[valid_ci], y_high[valid_ci])
            inside = (y_true[valid_ci] >= lo) & (y_true[valid_ci] <= hi)
            ci_inside = int(np.sum(inside))
        else:
            ci_inside = 0
        ci_outside = int(max(0, n_ci - ci_inside))
        ci_cov = float(100.0 * ci_inside / n_ci) if n_ci > 0 else np.nan

        rows.append(
            {
                "group": group_name,
                "n_pred_points": n_pred,
                "rmse": rmse_val,
                "bias": bias_val,
                "r2": r2_val,
                "ci_inside_count": ci_inside,
                "ci_total_count": n_ci,
                "ci_outside_count": ci_outside,
                "ci_coverage_pct": ci_cov,
            }
        )

    return pd.DataFrame(rows)


def _compute_loocv_sample_statistics(
    df_long: pd.DataFrame,
    sample_col: str = "project_sample_id",
    status_col: str = CATALYZED_COLUMNS_ID,
    y_true_col: str = "y_true",
    pred_col: str = "y_pred_mean",
    low_col: str = "y_pred_p05",
    high_col: str = "y_pred_p95",
) -> pd.DataFrame:
    if df_long is None or df_long.empty or sample_col not in df_long.columns:
        return pd.DataFrame()

    rows = []
    for sample_id, sample_df in df_long.groupby(sample_col, dropna=False):
        sample_metrics = _compute_loocv_group_statistics(
            sample_df,
            status_col=status_col,
            y_true_col=y_true_col,
            pred_col=pred_col,
            low_col=low_col,
            high_col=high_col,
        )
        if sample_metrics.empty:
            continue
        sample_metrics[sample_col] = sample_id
        rows.append(sample_metrics)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    cols = [sample_col, "group", "n_pred_points", "rmse", "bias", "r2", "ci_inside_count", "ci_total_count", "ci_outside_count", "ci_coverage_pct"]
    return out[cols]


def plot_loocv_overall_statistics(
    df_long: pd.DataFrame,
    plot_path: str,
    plot_title: str = "LOOCV Overall Statistics",
    status_col: str = CATALYZED_COLUMNS_ID,
    y_true_col: str = "y_true",
    pred_col: str = "y_pred_mean",
    low_col: str = "y_pred_p05",
    high_col: str = "y_pred_p95",
) -> pd.DataFrame:
    stats_df = _compute_loocv_group_statistics(
        df_long=df_long,
        status_col=status_col,
        y_true_col=y_true_col,
        pred_col=pred_col,
        low_col=low_col,
        high_col=high_col,
    )
    if stats_df.empty:
        return stats_df

    order = ["Overall", "Control", "Catalyzed"]
    stats_df["group"] = pd.Categorical(stats_df["group"], categories=order, ordered=True)
    stats_df = stats_df.sort_values("group").reset_index(drop=True)

    labels = stats_df["group"].astype(str).tolist()
    x = np.arange(len(labels), dtype=float)
    colors = ["#4c78a8", "#1f77b4", "#ff7f0e"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # RMSE
    ax = axes[0, 0]
    rmse_vals = pd.to_numeric(stats_df["rmse"], errors="coerce").to_numpy(dtype=float)
    ax.bar(x, np.nan_to_num(rmse_vals, nan=0.0), color=colors, alpha=0.85)
    ax.set_xticks(x, labels)
    ax.set_title("RMSE")
    ax.set_ylabel("Cu Recovery (%)")
    ax.grid(True, axis="y", alpha=0.25)
    for i, v in enumerate(rmse_vals):
        txt = "nan" if not np.isfinite(v) else f"{v:.3f}"
        y_pos = 0.02 if not np.isfinite(v) else max(0.02, v) + 0.01 * (np.nanmax(np.nan_to_num(rmse_vals, nan=0.0)) + 1.0)
        ax.text(i, y_pos, txt, ha="center", va="bottom", fontsize=9)

    # Bias
    ax = axes[0, 1]
    bias_vals = pd.to_numeric(stats_df["bias"], errors="coerce").to_numpy(dtype=float)
    ax.bar(x, np.nan_to_num(bias_vals, nan=0.0), color=colors, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x, labels)
    ax.set_title("Bias (Pred - Actual)")
    ax.set_ylabel("Cu Recovery (%)")
    ax.grid(True, axis="y", alpha=0.25)
    span = max(1.0, float(np.nanmax(np.abs(np.nan_to_num(bias_vals, nan=0.0)))))
    for i, v in enumerate(bias_vals):
        txt = "nan" if not np.isfinite(v) else f"{v:.3f}"
        y_pos = 0.02 * span if not np.isfinite(v) else v + np.sign(v if v != 0 else 1.0) * 0.04 * span
        ax.text(i, y_pos, txt, ha="center", va="bottom" if (not np.isfinite(v) or v >= 0) else "top", fontsize=9)

    # R2
    ax = axes[1, 0]
    r2_vals = pd.to_numeric(stats_df["r2"], errors="coerce").to_numpy(dtype=float)
    ax.bar(x, np.nan_to_num(r2_vals, nan=0.0), color=colors, alpha=0.85)
    ax.set_xticks(x, labels)
    ax.set_title("R2")
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(min(-1.0, np.nanmin(np.nan_to_num(r2_vals, nan=0.0)) - 0.1), max(1.0, np.nanmax(np.nan_to_num(r2_vals, nan=0.0)) + 0.1))
    for i, v in enumerate(r2_vals):
        txt = "nan" if not np.isfinite(v) else f"{v:.3f}"
        y_pos = 0.02 if not np.isfinite(v) else v + 0.04 * (1 if v >= 0 else -1)
        ax.text(i, y_pos, txt, ha="center", va="bottom" if (not np.isfinite(v) or v >= 0) else "top", fontsize=9)

    # CI counts (inside 90% PI)
    ax = axes[1, 1]
    inside = pd.to_numeric(stats_df["ci_inside_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
    outside = pd.to_numeric(stats_df["ci_outside_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
    total = pd.to_numeric(stats_df["ci_total_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
    cov_pct = pd.to_numeric(stats_df["ci_coverage_pct"], errors="coerce").to_numpy(dtype=float)
    ax.bar(x, inside, color="#59a14f", alpha=0.85, label="Inside 90% PI")
    ax.bar(x, outside, bottom=inside, color="#e15759", alpha=0.75, label="Outside 90% PI")
    ax.set_xticks(x, labels)
    ax.set_title("CI Count (Inside 90% PI)")
    ax.set_ylabel("Count of Actual Points")
    ax.grid(True, axis="y", alpha=0.25)
    for i in range(len(labels)):
        cov_txt = "nan" if not np.isfinite(cov_pct[i]) else f"{cov_pct[i]:.1f}%"
        ax.text(
            i,
            inside[i] + outside[i] + max(1.0, 0.01 * max(1.0, np.max(inside + outside))),
            f"{int(inside[i])}/{int(total[i])} ({cov_txt})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(plot_title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    return stats_df


def report_catalyst_correlations(df_long: pd.DataFrame, catalyst_col: str) -> None:
    if catalyst_col not in df_long.columns:
        return
    subset = df_long[df_long[CATALYZED_COLUMNS_ID] == "Catalyzed"].copy()
    subset = subset[np.isfinite(subset[catalyst_col])]
    if subset.empty:
        return
    check_cols = [
        "residual_cpy_%",
        "cu_%",
        "grouped_copper_sulfides",
        "grouped_secondary_copper",
    ]
    print("Catalyzed-only correlations with cumulative catalyst (Spearman):")
    for col in check_cols:
        if col in subset.columns:
            valid = subset[[catalyst_col, col]].dropna()
            if len(valid) < 3:
                continue
            corr = valid[catalyst_col].corr(valid[col], method="spearman")
            print(f"  {col}: {corr:.3f}")


def check_control_catalyzed_pairs(df_long: pd.DataFrame, id_col: str, status_col: str) -> None:
    missing = []
    for sample_id, group in df_long.groupby(id_col):
        statuses = set(group[status_col].astype(str).unique())
        if "Control" not in statuses or "Catalyzed" not in statuses:
            missing.append((sample_id, statuses))
    if missing:
        print("Warning: samples missing Control/Catalyzed pairs:")
        for sample_id, statuses in missing:
            print(f"  {sample_id}: {sorted(statuses)}")

# control and catalyzed cv sample 


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _config_digest(config: Dict) -> str:
    payload = json.dumps(
        _json_safe(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _initialize_experiment_tracker(
    tracking_cfg: Dict,
    config: Dict,
    output_root: str,
) -> Dict[str, Any]:
    enabled = bool(tracking_cfg.get("enabled", True))
    tracker = {"enabled": enabled, "config": dict(tracking_cfg)}
    if not enabled:
        return tracker

    started_at = datetime.now(timezone.utc)
    config_hash = _config_digest(config)
    raw_tag = str(tracking_cfg.get("tag", "run"))
    clean_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_tag).strip("_") or "run"
    run_id = f"{clean_tag}_{started_at.strftime('%Y%m%dT%H%M%SZ')}_{config_hash[:8]}"

    root_dir = os.path.join(output_root, str(tracking_cfg.get("root_dir_name", "experiments")))
    run_dir = os.path.join(root_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    tracker.update(
        {
            "run_id": run_id,
            "started_at_utc": started_at.isoformat(),
            "config_hash": config_hash,
            "root_dir": root_dir,
            "run_dir": run_dir,
            "history_csv_path": os.path.join(root_dir, "history.csv"),
            "history_jsonl_path": os.path.join(root_dir, "history.jsonl"),
            "summary_path": os.path.join(run_dir, "run_summary.json"),
            "config_snapshot_path": os.path.join(run_dir, "config_snapshot.json"),
        }
    )

    config_snapshot = {
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "config_hash": config_hash,
        "script_path": os.path.abspath(__file__),
        "project_root": PROJECT_ROOT,
        "device": str(device),
        "config": _json_safe(config),
    }
    with open(tracker["config_snapshot_path"], "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)

    latest_path = os.path.join(root_dir, "latest_run.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "started_at_utc": started_at.isoformat(),
                "run_dir": run_dir,
                "config_hash": config_hash,
            },
            f,
            indent=2,
        )
    return tracker


def _compute_group_curve_metrics(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str,
    y_pred_col: str,
) -> pd.DataFrame:
    cols = ["group", "n_points", "rmse", "mae", "r2", "bias"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for group, sub in df.groupby(group_col):
        y_true = pd.to_numeric(sub[y_true_col], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(sub[y_pred_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if int(valid.sum()) == 0:
            continue
        metrics = compute_metrics(y_true[valid], y_pred[valid])
        rows.append(
            {
                "group": normalize_status(group),
                "n_points": int(valid.sum()),
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "r2": float(metrics["r2"]),
                "bias": float(metrics["bias"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    return out.sort_values(["group"]).reset_index(drop=True)


def _metric_from_group_df(group_df: pd.DataFrame, group: str, metric: str) -> float:
    if group_df is None or group_df.empty:
        return np.nan
    sub = group_df[group_df["group"] == normalize_status(group)]
    if sub.empty:
        return np.nan
    return float(pd.to_numeric(sub.iloc[0].get(metric, np.nan), errors="coerce"))


def _finalize_experiment_tracking(
    tracker: Dict[str, Any],
    validation_metrics: Dict[str, float],
    full_metrics: Dict[str, float],
    validation_group_metrics_df: pd.DataFrame,
    full_group_metrics_df: pd.DataFrame,
    validation_param_metrics_df: pd.DataFrame,
    loocv_overall_stats_df: pd.DataFrame,
    param_calibration_state: Dict,
    curve_cfg: Dict,
    shaping_cfg: Dict,
    coupling_cfg: Dict,
    uncertainty_cfg: Dict,
    param_calib_cfg: Dict,
    ensemble_seeds: List[int],
    feature_cols: List[str],
    n_samples: int,
    output_file_map: Dict[str, str],
    columns_only_results: Dict[str, Any] = None,
) -> None:
    if not tracker.get("enabled", False):
        return

    finished_at = datetime.now(timezone.utc).isoformat()
    run_dir = tracker.get("run_dir", OUTPUTS_ROOT)
    os.makedirs(run_dir, exist_ok=True)
    run_output_snapshots = {}
    for name, src_path in (output_file_map or {}).items():
        if not isinstance(src_path, str):
            continue
        if not os.path.exists(src_path):
            continue
        dst_path = os.path.join(run_dir, os.path.basename(src_path))
        try:
            shutil.copy2(src_path, dst_path)
            run_output_snapshots[name] = dst_path
        except Exception:
            continue

    knobs = {
        "curve_rf_n_estimators": curve_cfg.get("rf_n_estimators"),
        "curve_rf_min_samples_leaf": curve_cfg.get("rf_min_samples_leaf"),
        "column_nonideal_a_factor": shaping_cfg.get("column_nonideal_a_factor"),
        "column_nonideal_b_factor": shaping_cfg.get("column_nonideal_b_factor"),
        "b2_cap_ratio_to_b1": shaping_cfg.get("b2_cap_ratio_to_b1"),
        "param_calibration_enabled": param_calib_cfg.get("enabled"),
        "param_calibration_b2_ratio_cap_quantile": param_calib_cfg.get("b2_ratio_cap_quantile"),
        "param_calibration_total_cap_quantile": param_calib_cfg.get("total_cap_quantile"),
        "catalyst_coupling_enabled": coupling_cfg.get("enabled"),
        "uncertainty_enabled": uncertainty_cfg.get("enabled"),
    }

    summary = {
        "run_id": tracker["run_id"],
        "started_at_utc": tracker["started_at_utc"],
        "finished_at_utc": finished_at,
        "config_hash": tracker["config_hash"],
        "n_samples": int(n_samples),
        "n_features": int(len(feature_cols)),
        "ensemble_members": int(len(ensemble_seeds)),
        "validation_metrics": _json_safe(validation_metrics),
        "full_metrics": _json_safe(full_metrics),
        "validation_metrics_by_status": _json_safe(validation_group_metrics_df),
        "full_metrics_by_status": _json_safe(full_group_metrics_df),
        "validation_param_metrics": _json_safe(validation_param_metrics_df),
        "loocv_overall_stats": _json_safe(loocv_overall_stats_df),
        "param_calibration_state": _json_safe(param_calibration_state),
        "tuning_knobs": _json_safe(knobs),
        "output_files": _json_safe(output_file_map),
        "run_output_snapshots": _json_safe(run_output_snapshots),
    }
    if columns_only_results:
        summary["columns_only_results"] = _json_safe(columns_only_results)

    with open(tracker["summary_path"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if bool(tracker["config"].get("append_history_jsonl", True)):
        os.makedirs(os.path.dirname(tracker["history_jsonl_path"]), exist_ok=True)
        with open(tracker["history_jsonl_path"], "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=True) + "\n")

    if bool(tracker["config"].get("append_history_csv", True)):
        row = {
            "run_id": tracker["run_id"],
            "started_at_utc": tracker["started_at_utc"],
            "finished_at_utc": finished_at,
            "config_hash": tracker["config_hash"],
            "n_samples": int(n_samples),
            "n_features": int(len(feature_cols)),
            "ensemble_members": int(len(ensemble_seeds)),
            "validation_rmse": float(validation_metrics.get("rmse", np.nan)),
            "validation_mae": float(validation_metrics.get("mae", np.nan)),
            "validation_r2": float(validation_metrics.get("r2", np.nan)),
            "validation_bias": float(validation_metrics.get("bias", np.nan)),
            "validation_control_rmse": _metric_from_group_df(validation_group_metrics_df, "Control", "rmse"),
            "validation_catalyzed_rmse": _metric_from_group_df(validation_group_metrics_df, "Catalyzed", "rmse"),
            "full_rmse": float(full_metrics.get("rmse", np.nan)),
            "full_mae": float(full_metrics.get("mae", np.nan)),
            "full_r2": float(full_metrics.get("r2", np.nan)),
            "full_bias": float(full_metrics.get("bias", np.nan)),
            "full_control_rmse": _metric_from_group_df(full_group_metrics_df, "Control", "rmse"),
            "full_catalyzed_rmse": _metric_from_group_df(full_group_metrics_df, "Catalyzed", "rmse"),
            "param_calibration_enabled": bool(param_calib_cfg.get("enabled", False)),
            "column_nonideal_a_factor": pd.to_numeric(knobs.get("column_nonideal_a_factor", np.nan), errors="coerce"),
            "column_nonideal_b_factor": pd.to_numeric(knobs.get("column_nonideal_b_factor", np.nan), errors="coerce"),
            "b2_cap_ratio_to_b1": pd.to_numeric(knobs.get("b2_cap_ratio_to_b1", np.nan), errors="coerce"),
        }
        if columns_only_results:
            col_val = columns_only_results.get("validation_metrics", {}) or {}
            col_full = columns_only_results.get("full_metrics", {}) or {}
            row["columns_only_validation_rmse"] = float(pd.to_numeric(col_val.get("rmse", np.nan), errors="coerce"))
            row["columns_only_validation_r2"] = float(pd.to_numeric(col_val.get("r2", np.nan), errors="coerce"))
            row["columns_only_full_rmse"] = float(pd.to_numeric(col_full.get("rmse", np.nan), errors="coerce"))
            row["columns_only_full_r2"] = float(pd.to_numeric(col_full.get("r2", np.nan), errors="coerce"))
        history_exists = os.path.exists(tracker["history_csv_path"])
        os.makedirs(os.path.dirname(tracker["history_csv_path"]), exist_ok=True)
        pd.DataFrame([row]).to_csv(
            tracker["history_csv_path"],
            mode="a",
            header=not history_exists,
            index=False,
        )


def _run_columns_only_variant(
    df_columns_params: pd.DataFrame,
    ore_similarity_features: List[str],
    target_cols: List[str],
    curve_cfg: Dict,
    coupling_cfg: Dict,
    shaping_cfg: Dict,
    uncertainty_cfg: Dict,
    ensemble_cfg: Dict,
    param_calib_cfg: Dict,
    ensemble_seeds: List[int],
    device: torch.device,
) -> Dict[str, Any]:
    print("Running columns-only variant (no reactor data, no transfer).")

    curve_cfg_variant = dict(curve_cfg)
    curve_cfg_variant["use_reactor_similarity_params"] = False
    curve_cfg_variant["use_reactor_normalized_uplift_prior"] = False
    curve_cfg_variant["enforce_reactor_asymptote_cap"] = False

    columns_cfg_variant = dict(CONFIG["columns"])
    columns_cfg_variant["transfer_copy_hidden_layers"] = False
    columns_cfg_variant["transfer_copy_output_layer"] = False
    columns_cfg_variant["freeze_pretrained_layers"] = False

    df_variant = df_columns_params.copy().reset_index(drop=True)
    reactor_like_cols = [c for c in df_variant.columns if str(c).startswith("reactor_")]
    for col in reactor_like_cols:
        df_variant[col] = np.nan

    ore_features_columns_only = []
    for col in ORE_SIMILARITY_CANDIDATES:
        if col not in df_variant.columns:
            continue
        vals = pd.to_numeric(df_variant[col], errors="coerce")
        if vals.notna().sum() >= 5:
            ore_features_columns_only.append(col)
    if not ore_features_columns_only:
        ore_features_columns_only = [c for c in ore_similarity_features if c in df_variant.columns]

    feature_cols_variant = _select_column_model_features(
        df_columns_params=df_variant,
        ore_similarity_features=ore_features_columns_only,
        use_reactor_similarity_params=False,
    )
    feature_cols_variant = [
        c
        for c in feature_cols_variant
        if not str(c).startswith("reactor_")
    ]
    for col in feature_cols_variant + target_cols:
        if col in df_variant.columns:
            df_variant[col] = pd.to_numeric(df_variant[col], errors="coerce")
    if df_variant.empty:
        raise ValueError("Columns-only variant has no rows to train.")
    if not feature_cols_variant:
        raise ValueError("Columns-only variant resolved no usable features.")

    print(f"Columns-only features ({len(feature_cols_variant)}): {feature_cols_variant}")

    def _predict_member_params_columns_only(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        fold_label: str,
    ) -> np.ndarray:
        n_members = len(ensemble_seeds)
        n_test = len(test_df)
        member_param_preds = np.full((n_members, n_test, len(target_cols)), np.nan, dtype=float)
        if n_test == 0:
            return member_param_preds

        X_test = test_df[feature_cols_variant].copy()
        for target_idx, target_col in enumerate(target_cols):
            y_train_full = pd.to_numeric(train_df[target_col], errors="coerce")
            valid_train_mask = np.isfinite(y_train_full.values)
            X_train = train_df.loc[valid_train_mask, feature_cols_variant].copy()
            y_train = y_train_full.loc[valid_train_mask].copy()
            if X_train.empty:
                raise ValueError(f"{fold_label}: no training rows for target {target_col}.")

            _, _, _, _, member_preds = ensemble_predictions(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                columns_features=feature_cols_variant,
                reactor_model=None,
                reactor_features=[],
                config=columns_cfg_variant,
                ensemble_seeds=ensemble_seeds,
                pi_lower=ensemble_cfg["pi_lower"],
                pi_upper=ensemble_cfg["pi_upper"],
                device=device,
            )
            if member_preds.shape != (n_members, n_test):
                raise RuntimeError(
                    f"{fold_label}: unexpected ensemble shape for {target_col}. "
                    f"Expected {(n_members, n_test)}, got {member_preds.shape}."
                )
            member_param_preds[:, :, target_idx] = member_preds
        return member_param_preds

    validation_curve_preds = []
    validation_param_rows_raw = []
    validation_member_rows = []
    plot_dir = os.path.join(PLOTS_ROOT, "columns_only_curve_loocv")
    os.makedirs(plot_dir, exist_ok=True)
    direct_targets_df = (
        df_variant[
            [
                "project_sample_id",
                CATALYZED_COLUMNS_ID,
                "target_direct_a1",
                "target_direct_b1",
                "target_direct_a2",
                "target_direct_b2",
            ]
        ]
        .drop_duplicates(subset=["project_sample_id", CATALYZED_COLUMNS_ID], keep="first")
        .rename(
            columns={
                "target_direct_a1": "true_direct_a1",
                "target_direct_b1": "true_direct_b1",
                "target_direct_a2": "true_direct_a2",
                "target_direct_b2": "true_direct_b2",
            }
        )
    )

    sample_ids = sorted(df_variant["project_sample_id"].dropna().unique())
    for sample_id in sample_ids:
        test_df = df_variant[df_variant["project_sample_id"] == sample_id].copy()
        train_df = df_variant[df_variant["project_sample_id"] != sample_id].copy()
        if test_df.empty or train_df.empty:
            continue
        test_df = _attach_uncertainty_scores(
            sample_df=test_df,
            shaping_cfg=shaping_cfg,
            unc_cfg=uncertainty_cfg,
        )
        member_param_preds = _predict_member_params_columns_only(
            train_df=train_df,
            test_df=test_df,
            fold_label=f"ColumnsOnly LOOCV sample={sample_id}",
        )
        raw_member_params_by_idx = _prepare_member_params_for_sample(
            sample_df=test_df,
            member_param_preds=member_param_preds,
            curve_cfg=curve_cfg_variant,
            coupling_cfg=coupling_cfg,
            shaping_cfg=shaping_cfg,
            param_calibration=None,
            fold_id=str(sample_id),
        )

        for row_idx in test_df.index:
            row_meta = test_df.loc[row_idx]
            param_members = raw_member_params_by_idx[row_idx]
            p_mean = np.mean(param_members, axis=0)
            p_p05 = np.percentile(param_members, ensemble_cfg["pi_lower"], axis=0)
            p_p95 = np.percentile(param_members, ensemble_cfg["pi_upper"], axis=0)
            y_true_params = row_meta[target_cols].to_numpy(dtype=float)
            uncertainty_score = pd.to_numeric(row_meta.get("input_uncertainty_score", np.nan), errors="coerce")
            validation_param_rows_raw.append(
                {
                    "project_sample_id": row_meta["project_sample_id"],
                    CATALYZED_COLUMNS_ID: row_meta[CATALYZED_COLUMNS_ID],
                    "fold_sample_id": str(sample_id),
                    "input_uncertainty_score": uncertainty_score,
                    "true_a1": y_true_params[0],
                    "true_b1": y_true_params[1],
                    "true_a2": y_true_params[2],
                    "true_b2": y_true_params[3],
                    "pred_a1": p_mean[0],
                    "pred_b1": p_mean[1],
                    "pred_a2": p_mean[2],
                    "pred_b2": p_mean[3],
                    "pred_a1_p05": p_p05[0],
                    "pred_b1_p05": p_p05[1],
                    "pred_a2_p05": p_p05[2],
                    "pred_b2_p05": p_p05[3],
                    "pred_a1_p95": p_p95[0],
                    "pred_b1_p95": p_p95[1],
                    "pred_a2_p95": p_p95[2],
                    "pred_b2_p95": p_p95[3],
                }
            )
            validation_member_rows.append(
                {
                    "fold_sample_id": str(sample_id),
                    "row_meta": row_meta.copy(),
                    "raw_param_members": np.asarray(param_members, dtype=float).copy(),
                }
            )

    validation_param_raw_df = pd.DataFrame(validation_param_rows_raw)
    if validation_param_raw_df.empty:
        raise ValueError("Columns-only variant produced no parameter predictions.")

    param_calibration_state = _fit_foldwise_param_calibration(
        validation_param_df=validation_param_raw_df,
        direct_targets_df=direct_targets_df,
        cfg=param_calib_cfg,
    )
    validation_curve_map: Dict[Tuple[str, str], Dict[str, pd.DataFrame]] = {}
    validation_param_rows = []
    for entry in validation_member_rows:
        fold_id = str(entry["fold_sample_id"])
        row_meta = entry["row_meta"]
        status_norm = normalize_status(row_meta.get(CATALYZED_COLUMNS_ID, "Control"))
        calib_factors = _get_param_calibration_factors(
            calibration_state=param_calibration_state,
            status=status_norm,
            fold_id=fold_id,
        )
        calibrated_members = []
        for pred in np.asarray(entry["raw_param_members"], dtype=float):
            p = _apply_param_calibration(pred, calib_factors)
            p = _sanitize_curve_params(
                p[0],
                p[1],
                p[2],
                p[3],
                total_recovery_upper=curve_cfg_variant["fit_total_recovery_upper"],
                b_upper=curve_cfg_variant["fit_b_fast_upper"],
            )
            p = _apply_column_kinetics_caps(
                params=p,
                status=status_norm,
                row=row_meta,
                shaping_cfg=shaping_cfg,
            )
            calibrated_members.append(p)
        param_members = np.vstack(calibrated_members)

        p_mean = np.mean(param_members, axis=0)
        p_p05 = np.percentile(param_members, ensemble_cfg["pi_lower"], axis=0)
        p_p95 = np.percentile(param_members, ensemble_cfg["pi_upper"], axis=0)
        y_true_params = row_meta[target_cols].to_numpy(dtype=float)
        uncertainty_score = pd.to_numeric(row_meta.get("input_uncertainty_score", np.nan), errors="coerce")
        validation_param_rows.append(
            {
                "project_sample_id": row_meta["project_sample_id"],
                CATALYZED_COLUMNS_ID: row_meta[CATALYZED_COLUMNS_ID],
                "fold_sample_id": fold_id,
                "input_uncertainty_score": uncertainty_score,
                "true_a1": y_true_params[0],
                "true_b1": y_true_params[1],
                "true_a2": y_true_params[2],
                "true_b2": y_true_params[3],
                "pred_a1": p_mean[0],
                "pred_b1": p_mean[1],
                "pred_a2": p_mean[2],
                "pred_b2": p_mean[3],
                "pred_a1_p05": p_p05[0],
                "pred_b1_p05": p_p05[1],
                "pred_a2_p05": p_p05[2],
                "pred_b2_p05": p_p05[3],
                "pred_a1_p95": p_p95[0],
                "pred_b1_p95": p_p95[1],
                "pred_a2_p95": p_p95[2],
                "pred_b2_p95": p_p95[3],
            }
        )

        curve_df = _expand_curve_predictions_long(
            row_meta=row_meta,
            pred_param_members=param_members,
            pi_lower=ensemble_cfg["pi_lower"],
            pi_upper=ensemble_cfg["pi_upper"],
            fit_total_recovery_upper=curve_cfg_variant["fit_total_recovery_upper"],
            fit_b_fast_upper=curve_cfg_variant["fit_b_fast_upper"],
        )
        if not curve_df.empty:
            curve_df["fold_sample_id"] = fold_id
            key = (fold_id, str(row_meta["project_sample_id"]))
            if key not in validation_curve_map:
                validation_curve_map[key] = {}
            validation_curve_map[key][status_norm] = curve_df

    for sample_curve_map in validation_curve_map.values():
        if "Control" in sample_curve_map and "Catalyzed" in sample_curve_map and coupling_cfg.get("enabled", False):
            sample_curve_map["Catalyzed"] = _enforce_output_curve_pair_constraints(
                sample_curve_map["Control"],
                sample_curve_map["Catalyzed"],
                coupling_cfg,
                shaping_cfg,
            )
        for curve_df in sample_curve_map.values():
            validation_curve_preds.append(curve_df)

    if not validation_curve_preds:
        raise ValueError("Columns-only variant produced no curve predictions.")

    validation_curve_df = pd.concat(validation_curve_preds, ignore_index=True)
    validation_param_df = pd.DataFrame(validation_param_rows)
    curve_metrics = compute_metrics(
        validation_curve_df["y_true"].values.astype(float),
        validation_curve_df["y_pred_mean"].values.astype(float),
    )
    validation_group_metrics_df = _compute_group_curve_metrics(
        validation_curve_df,
        group_col=CATALYZED_COLUMNS_ID,
        y_true_col="y_true",
        y_pred_col="y_pred_mean",
    )
    param_metrics_rows = []
    for true_col, pred_col, label in [
        ("true_a1", "pred_a1", "a1"),
        ("true_b1", "pred_b1", "b1"),
        ("true_a2", "pred_a2", "a2"),
        ("true_b2", "pred_b2", "b2"),
    ]:
        valid = np.isfinite(validation_param_df[true_col].values) & np.isfinite(validation_param_df[pred_col].values)
        if valid.sum() == 0:
            continue
        rmse = float(root_mean_squared_error(validation_param_df.loc[valid, true_col], validation_param_df.loc[valid, pred_col]))
        mae = float(mean_absolute_error(validation_param_df.loc[valid, true_col], validation_param_df.loc[valid, pred_col]))
        param_metrics_rows.append({"parameter": label, "rmse": rmse, "mae": mae, "n": int(valid.sum())})
    param_metrics_df = pd.DataFrame(param_metrics_rows)

    preds_path = os.path.join(OUTPUTS_ROOT, "columns_only_curve_validation_predictions.csv")
    params_path = os.path.join(OUTPUTS_ROOT, "columns_only_curve_validation_param_predictions.csv")
    curve_metrics_path = os.path.join(OUTPUTS_ROOT, "columns_only_curve_validation_metrics.csv")
    param_metrics_path = os.path.join(OUTPUTS_ROOT, "columns_only_curve_validation_param_metrics.csv")
    validation_group_metrics_path = os.path.join(OUTPUTS_ROOT, "columns_only_curve_validation_metrics_by_status.csv")
    validation_curve_df.to_csv(preds_path, index=False)
    validation_param_df.to_csv(params_path, index=False)
    pd.DataFrame([curve_metrics]).to_csv(curve_metrics_path, index=False)
    param_metrics_df.to_csv(param_metrics_path, index=False)
    validation_group_metrics_df.to_csv(validation_group_metrics_path, index=False)

    loocv_overall_plot_path = os.path.join(plot_dir, "columns_only_loocv_overall_statistics.png")
    loocv_overall_stats_df = plot_loocv_overall_statistics(
        validation_curve_df,
        plot_path=loocv_overall_plot_path,
        plot_title="Columns-Only LOOCV Overall Statistics",
        status_col=CATALYZED_COLUMNS_ID,
        y_true_col="y_true",
        pred_col="y_pred_mean",
        low_col="__no_interval__",
        high_col="__no_interval__",
    )
    loocv_overall_stats_path = os.path.join(OUTPUTS_ROOT, "columns_only_curve_validation_overall_statistics.csv")
    if not loocv_overall_stats_df.empty:
        loocv_overall_stats_df.to_csv(loocv_overall_stats_path, index=False)

    loocv_sample_stats_df = _compute_loocv_sample_statistics(
        validation_curve_df,
        sample_col="project_sample_id",
        status_col=CATALYZED_COLUMNS_ID,
        y_true_col="y_true",
        pred_col="y_pred_mean",
        low_col="__no_interval__",
        high_col="__no_interval__",
    )
    loocv_sample_stats_path = os.path.join(OUTPUTS_ROOT, "columns_only_curve_validation_sample_statistics.csv")
    if not loocv_sample_stats_df.empty:
        loocv_sample_stats_df.to_csv(loocv_sample_stats_path, index=False)
    for sample_id, sample_df in validation_curve_df.groupby("project_sample_id"):
        plot_path = os.path.join(plot_dir, f"{sample_id}.png")
        plot_sample_predictions(
            sample_df,
            str(sample_id),
            plot_path,
            time_col=TIME_COL_COLUMNS,
            target_col="y_true",
            title_prefix="Columns-Only Validation Prediction",
            lower_col="__no_interval__",
            upper_col="__no_interval__",
            status_col=CATALYZED_COLUMNS_ID,
        )

    full_eval_df = _attach_uncertainty_scores(
        sample_df=df_variant.copy(),
        shaping_cfg=shaping_cfg,
        unc_cfg=uncertainty_cfg,
    )
    full_member_param_preds = _predict_member_params_columns_only(
        train_df=full_eval_df,
        test_df=full_eval_df,
        fold_label="ColumnsOnly Full-data ensemble",
    )
    full_param_calibration_state = {
        "enabled": bool(param_calibration_state.get("enabled", False)),
        "status_global": dict(param_calibration_state.get("status_global", {})),
        "overall": dict(param_calibration_state.get("overall", {})),
        "fold_status": {},
    }

    full_curve_preds = []
    full_param_rows = []
    full_index_to_pos = {idx: i for i, idx in enumerate(full_eval_df.index)}
    for sample_id, sample_df in full_eval_df.groupby("project_sample_id"):
        row_positions = [full_index_to_pos[idx] for idx in sample_df.index]
        sample_member_preds = full_member_param_preds[:, row_positions, :]
        sample_member_by_idx = _prepare_member_params_for_sample(
            sample_df=sample_df,
            member_param_preds=sample_member_preds,
            curve_cfg=curve_cfg_variant,
            coupling_cfg=coupling_cfg,
            shaping_cfg=shaping_cfg,
            param_calibration=full_param_calibration_state,
            fold_id=None,
        )
        sample_curve_map: Dict[str, pd.DataFrame] = {}
        for row_idx in sample_df.index:
            row_meta = sample_df.loc[row_idx]
            param_members = sample_member_by_idx[row_idx]
            p_mean = np.mean(param_members, axis=0)
            p_p05 = np.percentile(param_members, ensemble_cfg["pi_lower"], axis=0)
            p_p95 = np.percentile(param_members, ensemble_cfg["pi_upper"], axis=0)
            y_true_params = row_meta[target_cols].to_numpy(dtype=float)
            uncertainty_score = pd.to_numeric(row_meta.get("input_uncertainty_score", np.nan), errors="coerce")
            full_param_rows.append(
                {
                    "project_sample_id": row_meta["project_sample_id"],
                    CATALYZED_COLUMNS_ID: row_meta[CATALYZED_COLUMNS_ID],
                    "input_uncertainty_score": uncertainty_score,
                    "true_a1": y_true_params[0],
                    "true_b1": y_true_params[1],
                    "true_a2": y_true_params[2],
                    "true_b2": y_true_params[3],
                    "pred_a1": p_mean[0],
                    "pred_b1": p_mean[1],
                    "pred_a2": p_mean[2],
                    "pred_b2": p_mean[3],
                    "pred_a1_p05": p_p05[0],
                    "pred_b1_p05": p_p05[1],
                    "pred_a2_p05": p_p05[2],
                    "pred_b2_p05": p_p05[3],
                    "pred_a1_p95": p_p95[0],
                    "pred_b1_p95": p_p95[1],
                    "pred_a2_p95": p_p95[2],
                    "pred_b2_p95": p_p95[3],
                }
            )

            curve_df = _expand_curve_predictions_long(
                row_meta=row_meta,
                pred_param_members=param_members,
                pi_lower=ensemble_cfg["pi_lower"],
                pi_upper=ensemble_cfg["pi_upper"],
                fit_total_recovery_upper=curve_cfg_variant["fit_total_recovery_upper"],
                fit_b_fast_upper=curve_cfg_variant["fit_b_fast_upper"],
            )
            if not curve_df.empty:
                sample_curve_map[normalize_status(row_meta[CATALYZED_COLUMNS_ID])] = curve_df

        if "Control" in sample_curve_map and "Catalyzed" in sample_curve_map and coupling_cfg.get("enabled", False):
            sample_curve_map["Catalyzed"] = _enforce_output_curve_pair_constraints(
                sample_curve_map["Control"],
                sample_curve_map["Catalyzed"],
                coupling_cfg,
                shaping_cfg,
            )
        for curve_df in sample_curve_map.values():
            full_curve_preds.append(curve_df)

    if not full_curve_preds:
        raise ValueError("Columns-only full-data prediction produced no curves.")

    full_curve_df = pd.concat(full_curve_preds, ignore_index=True)
    full_param_df = pd.DataFrame(full_param_rows)
    full_curve_path = os.path.join(OUTPUTS_ROOT, "columns_only_curve_full_predictions.csv")
    full_param_path = os.path.join(OUTPUTS_ROOT, "columns_only_curve_full_param_predictions.csv")
    full_curve_df.to_csv(full_curve_path, index=False)
    full_param_df.to_csv(full_param_path, index=False)
    full_metrics = compute_metrics(
        full_curve_df["y_true"].values.astype(float),
        full_curve_df["y_pred_mean"].values.astype(float),
    )
    full_metrics_path = os.path.join(OUTPUTS_ROOT, "columns_only_curve_full_metrics.csv")
    pd.DataFrame([full_metrics]).to_csv(full_metrics_path, index=False)
    full_group_metrics_df = _compute_group_curve_metrics(
        full_curve_df,
        group_col=CATALYZED_COLUMNS_ID,
        y_true_col="y_true",
        y_pred_col="y_pred_mean",
    )
    full_group_metrics_path = os.path.join(OUTPUTS_ROOT, "columns_only_curve_full_metrics_by_status.csv")
    full_group_metrics_df.to_csv(full_group_metrics_path, index=False)

    ensemble_plot_dir = os.path.join(PLOTS_ROOT, "columns_only_ensembled")
    os.makedirs(ensemble_plot_dir, exist_ok=True)
    final_overall_plot_path = os.path.join(ensemble_plot_dir, "columns_only_ensembled_overall_statistics.png")
    plot_loocv_overall_statistics(
        full_curve_df,
        plot_path=final_overall_plot_path,
        plot_title="Columns-Only Ensembled Overall Statistics",
        status_col=CATALYZED_COLUMNS_ID,
        y_true_col="y_true",
        pred_col="y_pred_mean",
        low_col="y_pred_p05",
        high_col="y_pred_p95",
    )
    for sample_id, sample_df in full_curve_df.groupby("project_sample_id"):
        final_plot_path = os.path.join(ensemble_plot_dir, f"{sample_id}.png")
        plot_sample_predictions(
            sample_df,
            str(sample_id),
            final_plot_path,
            time_col=TIME_COL_COLUMNS,
            target_col="y_true",
            title_prefix="Columns-Only Ensembled Prediction",
            status_col=CATALYZED_COLUMNS_ID,
        )

    columns_only_knowledge = {
        "ore_features_columns_only": ore_features_columns_only,
        "feature_cols_for_training": feature_cols_variant,
        "target_cols_for_training": target_cols,
        "curve_model_config": curve_cfg_variant,
        "catalyst_curve_coupling_config": coupling_cfg,
        "inference_shaping_config": shaping_cfg,
        "inference_uncertainty_config": uncertainty_cfg,
        "param_calibration_config": param_calib_cfg,
        "param_calibration_state": {
            "enabled": bool(param_calibration_state.get("enabled", False)),
            "status_global": param_calibration_state.get("status_global", {}),
            "overall": param_calibration_state.get("overall", {}),
        },
        "ensemble_seeds_used_for_intervals": ensemble_seeds,
        "uses_column_targets_for_training": True,
        "uses_reactor_transfer": False,
        "uses_reactor_similarity_params": False,
        "uses_reactor_normalized_uplift_prior": False,
    }
    knowledge_path = os.path.join(MODELS_ROOT, "columns_only_curve_inference_knowledge.json")
    with open(knowledge_path, "w", encoding="utf-8") as f:
        json.dump(columns_only_knowledge, f, indent=2)
    features_path = os.path.join(MODELS_ROOT, "columns_only_curve_features.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ore_features_columns_only": ore_features_columns_only,
                "feature_cols": feature_cols_variant,
                "target_cols": target_cols,
                "curve_model_config": curve_cfg_variant,
                "catalyst_curve_coupling_config": coupling_cfg,
                "inference_shaping_config": shaping_cfg,
                "inference_uncertainty_config": uncertainty_cfg,
                "param_calibration_config": param_calib_cfg,
                "param_calibration_state": {
                    "enabled": bool(param_calibration_state.get("enabled", False)),
                    "status_global": param_calibration_state.get("status_global", {}),
                    "overall": param_calibration_state.get("overall", {}),
                },
                "ensemble_seeds_used_for_intervals": ensemble_seeds,
                "uses_column_targets_for_training": True,
                "uses_reactor_transfer": False,
                "uses_reactor_similarity_params": False,
                "uses_reactor_normalized_uplift_prior": False,
            },
            f,
            indent=2,
        )

    result = {
        "validation_metrics": curve_metrics,
        "full_metrics": full_metrics,
        "validation_group_metrics": _json_safe(validation_group_metrics_df),
        "full_group_metrics": _json_safe(full_group_metrics_df),
        "output_file_map": {
            "validation_curve_predictions": preds_path,
            "validation_param_predictions": params_path,
            "validation_curve_metrics": curve_metrics_path,
            "validation_param_metrics": param_metrics_path,
            "validation_group_metrics": validation_group_metrics_path,
            "validation_overall_stats": loocv_overall_stats_path,
            "validation_sample_stats": loocv_sample_stats_path,
            "full_curve_predictions": full_curve_path,
            "full_param_predictions": full_param_path,
            "full_curve_metrics": full_metrics_path,
            "full_group_metrics": full_group_metrics_path,
            "full_overall_stats_plot": final_overall_plot_path,
            "knowledge_json": knowledge_path,
            "features_json": features_path,
        },
        "feature_cols": feature_cols_variant,
        "param_calibration_state": _json_safe(param_calibration_state),
    }
    print(
        "Columns-only metrics "
        f"(validation_rmse={float(curve_metrics.get('rmse', np.nan)):.4f}, "
        f"validation_r2={float(curve_metrics.get('r2', np.nan)):.4f}, "
        f"full_rmse={float(full_metrics.get('rmse', np.nan)):.4f}, "
        f"full_r2={float(full_metrics.get('r2', np.nan)):.4f})"
    )
    return result


# ---------------------------
# Main pipeline
# ---------------------------
def main():
    set_all_seeds(CONFIG["seed"], deterministic=True)

    curve_cfg = CONFIG["curve_model"]
    ensemble_cfg = CONFIG["ensemble"]
    coupling_cfg = CONFIG["catalyst_curve_coupling"]
    shaping_cfg = dict(CONFIG.get("inference_shaping", {}))
    uncertainty_cfg = dict(CONFIG.get("inference_uncertainty", {}))
    param_calib_cfg = dict(CONFIG.get("param_calibration", {}))
    reactor_tuple_cfg = dict(CONFIG.get("reactor_tuple_transfer", {}))
    tracking_cfg = dict(CONFIG.get("experiment_tracking", {}))
    tracker = _initialize_experiment_tracker(
        tracking_cfg=tracking_cfg,
        config=CONFIG,
        output_root=OUTPUTS_ROOT,
    )
    if tracker.get("enabled", False):
        print(
            "Experiment tracking "
            f"(run_id={tracker['run_id']}, run_dir={tracker['run_dir']}, "
            f"history={tracker['history_csv_path']})"
        )
    else:
        print("Experiment tracking disabled.")
    reactor_time_max = _compute_time_max(df_reactors, TIME_COL_REACTORS)

    ore_similarity_features = _select_ore_similarity_features(
        df_model_recCu_catcontrol_projects,
        df_reactors,
        force_include_shared=bool(curve_cfg.get("force_include_shared_ore_features", True)),
    )
    if not ore_similarity_features:
        raise ValueError("No shared ore-characterization features available for reactor similarity transfer.")
    print(f"Ore similarity features ({len(ore_similarity_features)}): {ore_similarity_features}")

    feature_copy_cols = list(dict.fromkeys(ore_similarity_features + COLUMN_EXTRA_FEATURE_CANDIDATES))
    df_columns_params = _build_columns_curve_parameter_table(
        df_model_recCu_catcontrol_projects,
        feature_copy_cols,
        curve_cfg,
    )
    if df_columns_params.empty:
        raise ValueError("No valid column curves after fitting double-exponential parameters.")

    required_transfer_cols = list(
        dict.fromkeys(
            ore_similarity_features
            + [
                "transition_time",
                "material_size_p80_in",
                "column_height_m",
                "column_inner_diameter_m",
                "residual_cpy_%",
            ]
        )
    )
    df_columns_params, transfer_impute_report = _validate_and_impute_transfer_inputs(
        df_columns_params=df_columns_params,
        required_cols=required_transfer_cols,
    )
    print(f"Transfer input columns enforced ({len(required_transfer_cols)}): {required_transfer_cols}")
    if transfer_impute_report:
        print("Transfer input imputation applied:")
        for col, info in transfer_impute_report.items():
            print(f"  {col}: filled={int(info['n_filled'])}, value={float(info['fill_value']):.6f}")

    if curve_cfg["use_reactor_similarity_params"]:
        df_columns_params = _build_reactor_similarity_param_priors(
            df_columns_params=df_columns_params,
            df_reactors=df_reactors,
            similarity_features=ore_similarity_features,
            k_neighbors=curve_cfg["reactor_similarity_k"],
            eps=curve_cfg["reactor_similarity_eps"],
            fit_total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
            fit_b_fast_upper=curve_cfg["fit_b_fast_upper"],
        )
    else:
        for col in [
            "reactor_sim_a1",
            "reactor_sim_b1",
            "reactor_sim_a2",
            "reactor_sim_b2",
            "reactor_sim_mean_distance",
        ]:
            df_columns_params[col] = np.nan

    if curve_cfg.get("use_reactor_normalized_uplift_prior", True):
        uplift_library = _build_reactor_normalized_uplift_library(
            df_reactors=df_reactors,
            similarity_features=ore_similarity_features,
            fit_total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
            time_max=reactor_time_max,
            time_norm_mode=CONFIG["delta_transfer"]["time_norm"],
            control_floor=curve_cfg["reactor_norm_uplift_control_floor"],
            uplift_clip_low=curve_cfg["reactor_norm_uplift_clip_low"],
            uplift_clip_high=curve_cfg["reactor_norm_uplift_clip_high"],
        )
        df_columns_params["reactor_norm_uplift_prior_pct"] = _apply_reactor_normalized_uplift_prior(
            df_columns_params=df_columns_params,
            uplift_library=uplift_library,
            similarity_features=ore_similarity_features,
            k_neighbors=curve_cfg["reactor_similarity_k"],
            eps=curve_cfg["reactor_similarity_eps"],
            time_max=reactor_time_max,
            time_norm_mode=CONFIG["delta_transfer"]["time_norm"],
        )
    else:
        uplift_library = []
        df_columns_params["reactor_norm_uplift_prior_pct"] = 0.0

    df_columns_params = _add_tuple_relation_features(
        df_columns_params,
        prefix="reactor_sim",
        a1_col="reactor_sim_a1",
        b1_col="reactor_sim_b1",
        a2_col="reactor_sim_a2",
        b2_col="reactor_sim_b2",
    )

    if bool(reactor_tuple_cfg.get("enabled", True)):
        df_columns_params = _build_reactor_tuple_transfer_features(
            df_columns_params=df_columns_params,
            df_reactors=df_reactors,
            similarity_features=ore_similarity_features,
            tuple_cfg=reactor_tuple_cfg,
            seeds=ensemble_cfg.get("seeds", [CONFIG["seed"]]),
            fit_total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
            fit_b_fast_upper=curve_cfg["fit_b_fast_upper"],
        )
    else:
        for col in REACTOR_TUPLE_TRANSFER_FEATURES:
            if col not in df_columns_params.columns:
                df_columns_params[col] = np.nan

    feature_cols = _select_column_model_features(
        df_columns_params=df_columns_params,
        ore_similarity_features=ore_similarity_features,
        use_reactor_similarity_params=curve_cfg["use_reactor_similarity_params"],
    )
    target_cols = ["target_a1", "target_b1", "target_a2", "target_b2"]

    for col in feature_cols + target_cols:
        if col in df_columns_params.columns:
            df_columns_params[col] = pd.to_numeric(df_columns_params[col], errors="coerce")
    df_columns_params = df_columns_params.reset_index(drop=True)
    if df_columns_params.empty:
        raise ValueError("No valid column rows available for validation.")
    shaping_cfg = _prepare_inference_shaping_config(df_columns_params, shaping_cfg)
    uncertainty_cfg = _prepare_inference_uncertainty_config(df_columns_params, uncertainty_cfg)
    ensemble_seeds = _resolve_ensemble_seeds(
        ensemble_cfg.get("seeds", []),
        int(uncertainty_cfg.get("min_ensemble_members", 5)),
    )

    print(f"Column parameter model features ({len(feature_cols)}): {feature_cols}")
    print(
        "Asymptote caps "
        f"(reactors={float(curve_cfg.get('fit_total_recovery_upper', np.nan)):.1f}, "
        f"columns_control={float(curve_cfg.get('columns_fit_total_recovery_upper_control', np.nan)):.1f}, "
        f"columns_catalyzed={float(curve_cfg.get('columns_fit_total_recovery_upper_catalyzed', np.nan)):.1f})"
    )
    print(
        "Fit constraints "
        f"(a_min={float(curve_cfg.get('fit_a_min', np.nan)):.4f}, "
        f"b_min={float(curve_cfg.get('fit_b_min', np.nan)):.6f}, "
        f"enforce_b1_ge_b2={bool(curve_cfg.get('fit_enforce_fast_slow_constraint', False))}, "
        f"reject_if_stuck={bool(curve_cfg.get('fit_reject_if_stuck_on_first_seed', False))})"
    )
    print(f"Inference shaping uses material_size_p80_in with reference={shaping_cfg['p80_reference_in']:.4f}")
    print(f"Inference shaping uses cumulative catalyst with reference={shaping_cfg['catalyst_reference_kg_t']:.6f}")
    print(f"Inference shaping dynamic catalyst uplift: {bool(shaping_cfg.get('use_dynamic_catalyst_uplift', False))}")
    print(f"p80-dependent catalyst lag enabled: {bool(shaping_cfg.get('use_p80_catalyst_lag', False))}")
    print(
        "Geometry/residual uplift context "
        f"(H_ref={float(shaping_cfg.get('geometry_lag_reference_height_m', np.nan)):.3f}, "
        f"D_ref={float(shaping_cfg.get('geometry_lag_reference_diameter_m', np.nan)):.3f}, "
        f"residual_cpy_ref={float(shaping_cfg.get('residual_cpy_reference_pct', np.nan)):.3f})"
    )
    print(
        "Catalyst gap moderation "
        f"(damp={float(shaping_cfg.get('catalyst_gap_global_damp', np.nan)):.3f}, "
        f"soft_cap={bool(shaping_cfg.get('use_catalyst_gap_soft_cap', False))}, "
        f"cap_abs={float(shaping_cfg.get('catalyst_gap_soft_cap_abs_pct', np.nan)):.3f}, "
        f"cap_rel={float(shaping_cfg.get('catalyst_gap_soft_cap_fraction_of_control', np.nan)):.3f})"
    )
    print(
        "Column kinetics caps "
        f"(enabled={bool(shaping_cfg.get('use_column_kinetics_caps', False))}, "
        f"b1_cap_min={float(shaping_cfg.get('b1_cap_min', np.nan)):.4f}, "
        f"b1_cap_max={float(shaping_cfg.get('b1_cap_max', np.nan)):.4f}, "
        f"b1_cap_control={float(shaping_cfg.get('b1_cap_control', np.nan)):.4f}, "
        f"b1_cap_catalyzed={float(shaping_cfg.get('b1_cap_catalyzed', np.nan)):.4f}, "
        f"b2_ratio_cap={float(shaping_cfg.get('b2_cap_ratio_to_b1', np.nan)):.3f})"
    )
    context_feats = shaping_cfg.get("context_ore_features", []) or []
    context_weights = shaping_cfg.get("context_ore_weights", {}) or {}
    weight_preview = []
    for feat in context_feats:
        w = pd.to_numeric(context_weights.get(feat, 0.0), errors="coerce")
        if np.isfinite(w) and abs(float(w)) > 1e-6:
            weight_preview.append(f"{feat}={float(w):+.3f}")
    if len(weight_preview) > 6:
        weight_preview = weight_preview[:6] + ["..."]
    print(
        "Catalyst context offsets "
        f"(enabled={bool(shaping_cfg.get('use_catalyst_context_offsets', True))}, "
        f"ore_features={len(context_feats)}, "
        f"active_weights=[{', '.join(weight_preview)}])"
    )
    print(
        "Inference uncertainty ensemble "
        f"(enabled={bool(uncertainty_cfg.get('enabled', True))}, "
        f"members={len(ensemble_seeds)}, seeds={ensemble_seeds})"
    )
    print(
        "Parameter calibration "
        f"(enabled={bool(param_calib_cfg.get('enabled', True))}, "
        f"use_direct_targets={bool(param_calib_cfg.get('use_direct_targets', True))})"
    )
    print(
        "Reactor tuple transfer "
        f"(enabled={bool(reactor_tuple_cfg.get('enabled', True))}, "
        f"rf_n_estimators={int(reactor_tuple_cfg.get('rf_n_estimators', 500))}, "
        f"min_rows_per_status={int(reactor_tuple_cfg.get('min_rows_per_status', 12))})"
    )
    print(
        "Transfer init "
        f"(copy_hidden_layers={bool(CONFIG['columns'].get('transfer_copy_hidden_layers', False))}, "
        f"copy_output_layer={bool(CONFIG['columns'].get('transfer_copy_output_layer', False))})"
    )
    print("Columns LOOCV now uses supervised training on column targets with reactor-weight transfer.")
    check_control_catalyzed_pairs(df_columns_params, "project_sample_id", CATALYZED_COLUMNS_ID)

    reactor_status_medians = _compute_reactor_status_param_medians(
        df_reactors=df_reactors,
        fit_total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
        fit_b_fast_upper=curve_cfg["fit_b_fast_upper"],
    )
    try:
        input_example_path, input_example_sample_id, input_example_cols = _generate_inference_input_example_excel(
            df_columns_params=df_columns_params,
            ore_similarity_features=ore_similarity_features,
            reactor_status_medians=reactor_status_medians,
            output_dir=INPUT_EXAMPLE_ROOT,
            random_state=None,
        )
        print(
            "Inference input example saved: "
            f"{input_example_path} "
            f"(sample={input_example_sample_id}, columns={len(input_example_cols)})"
        )
    except Exception as exc:
        warnings.warn(f"Could not generate inference input example workbook: {exc}")

    reactor_model, reactor_features, reactor_transfer_summary = train_reactor_transfer_backbone(
        df_reactors_source=df_reactors,
        config=CONFIG["reactors"],
        seed=CONFIG["seed"],
        device=device,
    )
    shared_transfer_features = [f for f in feature_cols if f in reactor_features]
    print(
        "Reactor transfer backbone trained "
        f"(rows={reactor_transfer_summary['n_long_rows']}, "
        f"train_rmse={reactor_transfer_summary['train_rmse']:.4f}, "
        f"val_rmse={reactor_transfer_summary['val_rmse']:.4f}, "
        f"shared_features={len(shared_transfer_features)})"
    )

    def _predict_member_params_with_transfer(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        fold_label: str,
    ) -> np.ndarray:
        n_members = len(ensemble_seeds)
        n_test = len(test_df)
        member_param_preds = np.full((n_members, n_test, len(target_cols)), np.nan, dtype=float)
        if n_test == 0:
            return member_param_preds

        X_test = test_df[feature_cols].copy()
        for target_idx, target_col in enumerate(target_cols):
            y_train_full = pd.to_numeric(train_df[target_col], errors="coerce")
            valid_train_mask = np.isfinite(y_train_full.values)
            X_train = train_df.loc[valid_train_mask, feature_cols].copy()
            y_train = y_train_full.loc[valid_train_mask].copy()
            if X_train.empty:
                raise ValueError(f"{fold_label}: no training rows for target {target_col}.")

            _, _, _, _, member_preds = ensemble_predictions(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                columns_features=feature_cols,
                reactor_model=reactor_model,
                reactor_features=reactor_features,
                config=CONFIG["columns"],
                ensemble_seeds=ensemble_seeds,
                pi_lower=ensemble_cfg["pi_lower"],
                pi_upper=ensemble_cfg["pi_upper"],
                device=device,
            )
            if member_preds.shape != (n_members, n_test):
                raise RuntimeError(
                    f"{fold_label}: unexpected ensemble shape for {target_col}. "
                    f"Expected {(n_members, n_test)}, got {member_preds.shape}."
                )
            member_param_preds[:, :, target_idx] = member_preds
        return member_param_preds

    validation_curve_preds = []
    validation_param_rows_raw = []
    validation_member_rows = []
    plot_dir = os.path.join(PLOTS_ROOT, "columns_curve_loocv")
    os.makedirs(plot_dir, exist_ok=True)
    direct_targets_df = (
        df_columns_params[
            [
                "project_sample_id",
                CATALYZED_COLUMNS_ID,
                "target_direct_a1",
                "target_direct_b1",
                "target_direct_a2",
                "target_direct_b2",
            ]
        ]
        .drop_duplicates(subset=["project_sample_id", CATALYZED_COLUMNS_ID], keep="first")
        .rename(
            columns={
                "target_direct_a1": "true_direct_a1",
                "target_direct_b1": "true_direct_b1",
                "target_direct_a2": "true_direct_a2",
                "target_direct_b2": "true_direct_b2",
            }
        )
    )

    sample_ids = sorted(df_columns_params["project_sample_id"].dropna().unique())
    for sample_id in sample_ids:
        test_df = df_columns_params[df_columns_params["project_sample_id"] == sample_id].copy()
        train_df = df_columns_params[df_columns_params["project_sample_id"] != sample_id].copy()
        if test_df.empty or train_df.empty:
            continue
        test_df = _attach_uncertainty_scores(
            sample_df=test_df,
            shaping_cfg=shaping_cfg,
            unc_cfg=uncertainty_cfg,
        )

        member_param_preds = _predict_member_params_with_transfer(
            train_df=train_df,
            test_df=test_df,
            fold_label=f"LOOCV sample={sample_id}",
        )
        raw_member_params_by_idx = _prepare_member_params_for_sample(
            sample_df=test_df,
            member_param_preds=member_param_preds,
            curve_cfg=curve_cfg,
            coupling_cfg=coupling_cfg,
            shaping_cfg=shaping_cfg,
            param_calibration=None,
            fold_id=str(sample_id),
        )

        for row_idx in test_df.index:
            row_meta = test_df.loc[row_idx]
            param_members = raw_member_params_by_idx[row_idx]
            p_mean = np.mean(param_members, axis=0)
            p_p05 = np.percentile(param_members, ensemble_cfg["pi_lower"], axis=0)
            p_p95 = np.percentile(param_members, ensemble_cfg["pi_upper"], axis=0)
            y_true_params = row_meta[target_cols].to_numpy(dtype=float)
            uncertainty_score = pd.to_numeric(row_meta.get("input_uncertainty_score", np.nan), errors="coerce")

            validation_param_rows_raw.append(
                {
                    "project_sample_id": row_meta["project_sample_id"],
                    CATALYZED_COLUMNS_ID: row_meta[CATALYZED_COLUMNS_ID],
                    "fold_sample_id": str(sample_id),
                    "input_uncertainty_score": uncertainty_score,
                    "true_a1": y_true_params[0],
                    "true_b1": y_true_params[1],
                    "true_a2": y_true_params[2],
                    "true_b2": y_true_params[3],
                    "pred_a1": p_mean[0],
                    "pred_b1": p_mean[1],
                    "pred_a2": p_mean[2],
                    "pred_b2": p_mean[3],
                    "pred_a1_p05": p_p05[0],
                    "pred_b1_p05": p_p05[1],
                    "pred_a2_p05": p_p05[2],
                    "pred_b2_p05": p_p05[3],
                    "pred_a1_p95": p_p95[0],
                    "pred_b1_p95": p_p95[1],
                    "pred_a2_p95": p_p95[2],
                    "pred_b2_p95": p_p95[3],
                }
            )
            validation_member_rows.append(
                {
                    "fold_sample_id": str(sample_id),
                    "row_meta": row_meta.copy(),
                    "raw_param_members": np.asarray(param_members, dtype=float).copy(),
                }
            )

    validation_param_raw_df = pd.DataFrame(validation_param_rows_raw)
    if validation_param_raw_df.empty:
        raise ValueError("Validation produced no parameter predictions.")

    param_calibration_state = _fit_foldwise_param_calibration(
        validation_param_df=validation_param_raw_df,
        direct_targets_df=direct_targets_df,
        cfg=param_calib_cfg,
    )
    if param_calibration_state.get("enabled", False):
        for st, fac in param_calibration_state.get("status_global", {}).items():
            print(
                "Param calibration factors "
                f"(status={st}, a_scale={float(fac.get('a_scale', np.nan)):.4f}, "
                f"b1_scale={float(fac.get('b1_scale', np.nan)):.4f}, "
                f"b2_scale={float(fac.get('b2_scale', np.nan)):.4f}, "
                f"b2_ratio_cap={float(fac.get('b2_ratio_cap', np.nan)):.4f}, "
                f"total_cap={float(fac.get('total_cap', np.nan)):.4f}, "
                f"n={int(fac.get('n_rows', 0))})"
            )
    else:
        print("Param calibration disabled (insufficient data or config).")

    validation_curve_map: Dict[Tuple[str, str], Dict[str, pd.DataFrame]] = {}
    validation_param_rows = []
    for entry in validation_member_rows:
        fold_id = str(entry["fold_sample_id"])
        row_meta = entry["row_meta"]
        status_norm = normalize_status(row_meta.get(CATALYZED_COLUMNS_ID, "Control"))
        calib_factors = _get_param_calibration_factors(
            calibration_state=param_calibration_state,
            status=status_norm,
            fold_id=fold_id,
        )
        reactor_prior = row_meta[
            ["reactor_sim_a1", "reactor_sim_b1", "reactor_sim_a2", "reactor_sim_b2"]
        ].to_numpy(dtype=float)

        calibrated_members = []
        for pred in np.asarray(entry["raw_param_members"], dtype=float):
            p = _apply_param_calibration(pred, calib_factors)
            p = _sanitize_curve_params(
                p[0],
                p[1],
                p[2],
                p[3],
                total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
                b_upper=curve_cfg["fit_b_fast_upper"],
            )
            p = _apply_column_kinetics_caps(
                params=p,
                status=status_norm,
                row=row_meta,
                shaping_cfg=shaping_cfg,
            )
            if curve_cfg["enforce_reactor_asymptote_cap"]:
                p = _cap_by_reactor_asymptote(p, reactor_prior)
            calibrated_members.append(p)
        param_members = np.vstack(calibrated_members)

        p_mean = np.mean(param_members, axis=0)
        p_p05 = np.percentile(param_members, ensemble_cfg["pi_lower"], axis=0)
        p_p95 = np.percentile(param_members, ensemble_cfg["pi_upper"], axis=0)
        y_true_params = row_meta[target_cols].to_numpy(dtype=float)
        uncertainty_score = pd.to_numeric(row_meta.get("input_uncertainty_score", np.nan), errors="coerce")
        validation_param_rows.append(
            {
                "project_sample_id": row_meta["project_sample_id"],
                CATALYZED_COLUMNS_ID: row_meta[CATALYZED_COLUMNS_ID],
                "fold_sample_id": fold_id,
                "input_uncertainty_score": uncertainty_score,
                "true_a1": y_true_params[0],
                "true_b1": y_true_params[1],
                "true_a2": y_true_params[2],
                "true_b2": y_true_params[3],
                "pred_a1": p_mean[0],
                "pred_b1": p_mean[1],
                "pred_a2": p_mean[2],
                "pred_b2": p_mean[3],
                "pred_a1_p05": p_p05[0],
                "pred_b1_p05": p_p05[1],
                "pred_a2_p05": p_p05[2],
                "pred_b2_p05": p_p05[3],
                "pred_a1_p95": p_p95[0],
                "pred_b1_p95": p_p95[1],
                "pred_a2_p95": p_p95[2],
                "pred_b2_p95": p_p95[3],
            }
        )

        curve_df = _expand_curve_predictions_long(
            row_meta=row_meta,
            pred_param_members=param_members,
            pi_lower=ensemble_cfg["pi_lower"],
            pi_upper=ensemble_cfg["pi_upper"],
            fit_total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
            fit_b_fast_upper=curve_cfg["fit_b_fast_upper"],
        )
        if not curve_df.empty:
            curve_df["fold_sample_id"] = fold_id
            key = (fold_id, str(row_meta["project_sample_id"]))
            if key not in validation_curve_map:
                validation_curve_map[key] = {}
            validation_curve_map[key][status_norm] = curve_df

    for sample_curve_map in validation_curve_map.values():
        if "Control" in sample_curve_map and "Catalyzed" in sample_curve_map and coupling_cfg.get("enabled", False):
            sample_curve_map["Catalyzed"] = _enforce_output_curve_pair_constraints(
                sample_curve_map["Control"],
                sample_curve_map["Catalyzed"],
                coupling_cfg,
                shaping_cfg,
            )
        for curve_df in sample_curve_map.values():
            validation_curve_preds.append(curve_df)

    if not validation_curve_preds:
        raise ValueError("Validation produced no curve predictions.")

    validation_curve_df = pd.concat(validation_curve_preds, ignore_index=True)
    validation_param_df = pd.DataFrame(validation_param_rows)

    curve_metrics = compute_metrics(
        validation_curve_df["y_true"].values.astype(float),
        validation_curve_df["y_pred_mean"].values.astype(float),
    )
    print("Columns curve validation metrics:", curve_metrics)
    validation_group_metrics_df = _compute_group_curve_metrics(
        validation_curve_df,
        group_col=CATALYZED_COLUMNS_ID,
        y_true_col="y_true",
        y_pred_col="y_pred_mean",
    )

    param_metrics_rows = []
    for true_col, pred_col, label in [
        ("true_a1", "pred_a1", "a1"),
        ("true_b1", "pred_b1", "b1"),
        ("true_a2", "pred_a2", "a2"),
        ("true_b2", "pred_b2", "b2"),
    ]:
        valid = np.isfinite(validation_param_df[true_col].values) & np.isfinite(validation_param_df[pred_col].values)
        if valid.sum() == 0:
            continue
        rmse = float(root_mean_squared_error(validation_param_df.loc[valid, true_col], validation_param_df.loc[valid, pred_col]))
        mae = float(mean_absolute_error(validation_param_df.loc[valid, true_col], validation_param_df.loc[valid, pred_col]))
        param_metrics_rows.append({"parameter": label, "rmse": rmse, "mae": mae, "n": int(valid.sum())})
    param_metrics_df = pd.DataFrame(param_metrics_rows)

    preds_path = os.path.join(OUTPUTS_ROOT, "columns_curve_validation_predictions.csv")
    params_path = os.path.join(OUTPUTS_ROOT, "columns_curve_validation_param_predictions.csv")
    curve_metrics_path = os.path.join(OUTPUTS_ROOT, "columns_curve_validation_metrics.csv")
    param_metrics_path = os.path.join(OUTPUTS_ROOT, "columns_curve_validation_param_metrics.csv")
    validation_group_metrics_path = os.path.join(OUTPUTS_ROOT, "columns_curve_validation_metrics_by_status.csv")
    validation_curve_df.to_csv(preds_path, index=False)
    validation_param_df.to_csv(params_path, index=False)
    pd.DataFrame([curve_metrics]).to_csv(curve_metrics_path, index=False)
    param_metrics_df.to_csv(param_metrics_path, index=False)
    validation_group_metrics_df.to_csv(validation_group_metrics_path, index=False)
    # Backward-compatibility outputs.
    validation_curve_df.to_csv(os.path.join(OUTPUTS_ROOT, "columns_curve_loocv_predictions.csv"), index=False)
    validation_param_df.to_csv(os.path.join(OUTPUTS_ROOT, "columns_curve_loocv_param_predictions.csv"), index=False)
    pd.DataFrame([curve_metrics]).to_csv(os.path.join(OUTPUTS_ROOT, "columns_curve_loocv_metrics.csv"), index=False)
    param_metrics_df.to_csv(os.path.join(OUTPUTS_ROOT, "columns_curve_loocv_param_metrics.csv"), index=False)
    validation_group_metrics_df.to_csv(
        os.path.join(OUTPUTS_ROOT, "columns_curve_loocv_metrics_by_status.csv"),
        index=False,
    )

    loocv_overall_plot_path = os.path.join(plot_dir, "loocv_overall_statistics.png")
    loocv_overall_stats_df = plot_loocv_overall_statistics(
        validation_curve_df,
        plot_path=loocv_overall_plot_path,
        status_col=CATALYZED_COLUMNS_ID,
        y_true_col="y_true",
        pred_col="y_pred_mean",
        low_col="__no_interval__",
        high_col="__no_interval__",
    )
    if not loocv_overall_stats_df.empty:
        loocv_overall_stats_df.to_csv(
            os.path.join(OUTPUTS_ROOT, "columns_curve_loocv_overall_statistics.csv"),
            index=False,
        )
        # Backward-compatible alias with validation prefix.
        loocv_overall_stats_df.to_csv(
            os.path.join(OUTPUTS_ROOT, "columns_curve_validation_overall_statistics.csv"),
            index=False,
        )
        overall_row = loocv_overall_stats_df[loocv_overall_stats_df["group"] == "Overall"]
        if not overall_row.empty:
            r = overall_row.iloc[0]
            print(
                "LOOCV overall stats "
                f"(RMSE={r.get('rmse', np.nan):.4f}, Bias={r.get('bias', np.nan):.4f}, "
                f"R2={r.get('r2', np.nan):.4f}, CI_inside_count={int(r.get('ci_inside_count', 0))}/"
                f"{int(r.get('ci_total_count', 0))}, CI_coverage_pct={r.get('ci_coverage_pct', np.nan):.2f}%)"
            )
        print(f"LOOCV overall statistics plot saved: {loocv_overall_plot_path}")

    loocv_sample_stats_df = _compute_loocv_sample_statistics(
        validation_curve_df,
        sample_col="project_sample_id",
        status_col=CATALYZED_COLUMNS_ID,
        y_true_col="y_true",
        pred_col="y_pred_mean",
        low_col="__no_interval__",
        high_col="__no_interval__",
    )
    if not loocv_sample_stats_df.empty:
        loocv_sample_stats_df.to_csv(
            os.path.join(OUTPUTS_ROOT, "columns_curve_loocv_sample_statistics.csv"),
            index=False,
        )
        # Backward-compatible alias with validation prefix.
        loocv_sample_stats_df.to_csv(
            os.path.join(OUTPUTS_ROOT, "columns_curve_validation_sample_statistics.csv"),
            index=False,
        )

    for sample_id, sample_df in validation_curve_df.groupby("project_sample_id"):
        plot_path = os.path.join(plot_dir, f"{sample_id}.png")
        plot_sample_predictions(
            sample_df,
            str(sample_id),
            plot_path,
            time_col=TIME_COL_COLUMNS,
            target_col="y_true",
            title_prefix="Validation Prediction",
            lower_col="__no_interval__",
            upper_col="__no_interval__",
            status_col=CATALYZED_COLUMNS_ID,
        )

    # -------- Full-data ensembled predictions --------
    full_eval_df = _attach_uncertainty_scores(
        sample_df=df_columns_params.copy(),
        shaping_cfg=shaping_cfg,
        unc_cfg=uncertainty_cfg,
    )
    full_member_param_preds = _predict_member_params_with_transfer(
        train_df=full_eval_df,
        test_df=full_eval_df,
        fold_label="Full-data ensemble",
    )
    full_param_calibration_state = {
        "enabled": bool(param_calibration_state.get("enabled", False)),
        "status_global": dict(param_calibration_state.get("status_global", {})),
        "overall": dict(param_calibration_state.get("overall", {})),
        "fold_status": {},
    }

    full_curve_preds = []
    full_param_rows = []
    full_index_to_pos = {idx: i for i, idx in enumerate(full_eval_df.index)}
    for sample_id, sample_df in full_eval_df.groupby("project_sample_id"):
        row_positions = [full_index_to_pos[idx] for idx in sample_df.index]
        sample_member_preds = full_member_param_preds[:, row_positions, :]
        sample_member_by_idx = _prepare_member_params_for_sample(
            sample_df=sample_df,
            member_param_preds=sample_member_preds,
            curve_cfg=curve_cfg,
            coupling_cfg=coupling_cfg,
            shaping_cfg=shaping_cfg,
            param_calibration=full_param_calibration_state,
            fold_id=None,
        )

        sample_curve_map: Dict[str, pd.DataFrame] = {}
        for row_idx in sample_df.index:
            row_meta = sample_df.loc[row_idx]
            param_members = sample_member_by_idx[row_idx]
            p_mean = np.mean(param_members, axis=0)
            p_p05 = np.percentile(param_members, ensemble_cfg["pi_lower"], axis=0)
            p_p95 = np.percentile(param_members, ensemble_cfg["pi_upper"], axis=0)
            y_true_params = row_meta[target_cols].to_numpy(dtype=float)
            uncertainty_score = pd.to_numeric(row_meta.get("input_uncertainty_score", np.nan), errors="coerce")

            full_param_rows.append(
                {
                    "project_sample_id": row_meta["project_sample_id"],
                    CATALYZED_COLUMNS_ID: row_meta[CATALYZED_COLUMNS_ID],
                    "input_uncertainty_score": uncertainty_score,
                    "true_a1": y_true_params[0],
                    "true_b1": y_true_params[1],
                    "true_a2": y_true_params[2],
                    "true_b2": y_true_params[3],
                    "pred_a1": p_mean[0],
                    "pred_b1": p_mean[1],
                    "pred_a2": p_mean[2],
                    "pred_b2": p_mean[3],
                    "pred_a1_p05": p_p05[0],
                    "pred_b1_p05": p_p05[1],
                    "pred_a2_p05": p_p05[2],
                    "pred_b2_p05": p_p05[3],
                    "pred_a1_p95": p_p95[0],
                    "pred_b1_p95": p_p95[1],
                    "pred_a2_p95": p_p95[2],
                    "pred_b2_p95": p_p95[3],
                }
            )

            curve_df = _expand_curve_predictions_long(
                row_meta=row_meta,
                pred_param_members=param_members,
                pi_lower=ensemble_cfg["pi_lower"],
                pi_upper=ensemble_cfg["pi_upper"],
                fit_total_recovery_upper=curve_cfg["fit_total_recovery_upper"],
                fit_b_fast_upper=curve_cfg["fit_b_fast_upper"],
            )
            if not curve_df.empty:
                sample_curve_map[normalize_status(row_meta[CATALYZED_COLUMNS_ID])] = curve_df

        if "Control" in sample_curve_map and "Catalyzed" in sample_curve_map and coupling_cfg.get("enabled", False):
            sample_curve_map["Catalyzed"] = _enforce_output_curve_pair_constraints(
                sample_curve_map["Control"],
                sample_curve_map["Catalyzed"],
                coupling_cfg,
                shaping_cfg,
            )
        for curve_df in sample_curve_map.values():
            full_curve_preds.append(curve_df)

    if not full_curve_preds:
        raise ValueError("Full-data ensembled prediction step produced no curve outputs.")

    full_curve_df = pd.concat(full_curve_preds, ignore_index=True)
    full_param_df = pd.DataFrame(full_param_rows)
    full_curve_path = os.path.join(OUTPUTS_ROOT, "columns_curve_full_predictions.csv")
    full_param_path = os.path.join(OUTPUTS_ROOT, "columns_curve_full_param_predictions.csv")
    full_curve_df.to_csv(full_curve_path, index=False)
    full_param_df.to_csv(full_param_path, index=False)
    ensemble_plot_dir = os.path.join(PLOTS_ROOT, "ensembled")
    os.makedirs(ensemble_plot_dir, exist_ok=True)

    direct_param_cols = [
        "project_sample_id",
        CATALYZED_COLUMNS_ID,
        "target_direct_a1",
        "target_direct_b1",
        "target_direct_a2",
        "target_direct_b2",
    ]
    missing_direct_cols = [c for c in direct_param_cols if c not in df_columns_params.columns]
    if missing_direct_cols:
        raise KeyError(
            "Missing direct-fit parameter columns required for export: "
            f"{missing_direct_cols}"
        )
    direct_param_df = (
        df_columns_params[direct_param_cols]
        .drop_duplicates(subset=["project_sample_id", CATALYZED_COLUMNS_ID], keep="first")
        .rename(
            columns={
                "target_direct_a1": "true_a1",
                "target_direct_b1": "true_b1",
                "target_direct_a2": "true_a2",
                "target_direct_b2": "true_b2",
            }
        )
    )
    direct_param_df[CATALYZED_COLUMNS_ID] = direct_param_df[CATALYZED_COLUMNS_ID].apply(normalize_status)
    direct_param_df = direct_param_df.sort_values(["project_sample_id", CATALYZED_COLUMNS_ID]).reset_index(drop=True)

    direct_combined_path = os.path.join(
        OUTPUTS_ROOT,
        "columns_curve_full_param_true_direct_biexp.csv",
    )
    direct_param_df.to_csv(direct_combined_path, index=False)

    print(f"Direct bi-exponential true-parameter export (combined): {direct_combined_path}")

    full_metrics = compute_metrics(
        full_curve_df["y_true"].values.astype(float),
        full_curve_df["y_pred_mean"].values.astype(float),
    )
    print("Columns full-data ensembled metrics:", full_metrics)
    full_group_metrics_df = _compute_group_curve_metrics(
        full_curve_df,
        group_col=CATALYZED_COLUMNS_ID,
        y_true_col="y_true",
        y_pred_col="y_pred_mean",
    )
    full_group_metrics_path = os.path.join(OUTPUTS_ROOT, "columns_curve_full_metrics_by_status.csv")
    full_group_metrics_df.to_csv(full_group_metrics_path, index=False)

    final_overall_plot_path = os.path.join(ensemble_plot_dir, "ensembled_overall_statistics.png")
    plot_loocv_overall_statistics(
        full_curve_df,
        plot_path=final_overall_plot_path,
        plot_title="Ensembled Overall Statistics",
        status_col=CATALYZED_COLUMNS_ID,
        y_true_col="y_true",
        pred_col="y_pred_mean",
        low_col="y_pred_p05",
        high_col="y_pred_p95",
    )
    print(f"Ensembled overall statistics plot saved: {final_overall_plot_path}")

    for sample_id, sample_df in full_curve_df.groupby("project_sample_id"):
        final_plot_path = os.path.join(ensemble_plot_dir, f"{sample_id}.png")
        plot_sample_predictions(
            sample_df,
            str(sample_id),
            final_plot_path,
            time_col=TIME_COL_COLUMNS,
            target_col="y_true",
            title_prefix="Ensembled Prediction",
            status_col=CATALYZED_COLUMNS_ID,
        )
    print(f"Ensembled sample prediction plots saved in: {ensemble_plot_dir}")

    columns_only_results = _run_columns_only_variant(
        df_columns_params=df_columns_params,
        ore_similarity_features=ore_similarity_features,
        target_cols=target_cols,
        curve_cfg=curve_cfg,
        coupling_cfg=coupling_cfg,
        shaping_cfg=shaping_cfg,
        uncertainty_cfg=uncertainty_cfg,
        ensemble_cfg=ensemble_cfg,
        param_calib_cfg=param_calib_cfg,
        ensemble_seeds=ensemble_seeds,
        device=device,
    )

    inference_knowledge = {
        "ore_similarity_features": ore_similarity_features,
        "feature_cols_for_training": feature_cols,
        "target_cols_for_training": target_cols,
        "reactor_status_param_medians": {
            k: [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
            for k, v in reactor_status_medians.items()
        },
        "reactor_uplift_library_size": len(uplift_library),
        "reactor_transfer_summary": reactor_transfer_summary,
        "reactor_transfer_shared_features": shared_transfer_features,
        "curve_model_config": curve_cfg,
        "catalyst_curve_coupling_config": coupling_cfg,
        "inference_shaping_config": shaping_cfg,
        "inference_uncertainty_config": uncertainty_cfg,
        "experiment_tracking_config": tracking_cfg,
        "experiment_run_id": tracker.get("run_id"),
        "reactor_tuple_transfer_config": reactor_tuple_cfg,
        "param_calibration_config": param_calib_cfg,
        "param_calibration_state": {
            "enabled": bool(param_calibration_state.get("enabled", False)),
            "status_global": param_calibration_state.get("status_global", {}),
            "overall": param_calibration_state.get("overall", {}),
        },
        "columns_only_results_summary": {
            "validation_metrics": columns_only_results.get("validation_metrics", {}),
            "full_metrics": columns_only_results.get("full_metrics", {}),
            "output_file_map": columns_only_results.get("output_file_map", {}),
        },
        "ensemble_seeds_used_for_intervals": ensemble_seeds,
        "uses_column_targets_for_training": True,
    }
    with open(os.path.join(MODELS_ROOT, "columns_curve_inference_knowledge.json"), "w", encoding="utf-8") as f:
        json.dump(inference_knowledge, f, indent=2)
    pd.to_pickle(uplift_library, os.path.join(MODELS_ROOT, "reactor_norm_uplift_library.pkl"))

    with open(os.path.join(MODELS_ROOT, "columns_curve_features.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "ore_similarity_features": ore_similarity_features,
                "curve_model_config": curve_cfg,
                "catalyst_curve_coupling_config": coupling_cfg,
                "inference_shaping_config": shaping_cfg,
                "inference_uncertainty_config": uncertainty_cfg,
                "experiment_tracking_config": tracking_cfg,
                "experiment_run_id": tracker.get("run_id"),
                "reactor_tuple_transfer_config": reactor_tuple_cfg,
                "param_calibration_config": param_calib_cfg,
                "param_calibration_state": {
                    "enabled": bool(param_calibration_state.get("enabled", False)),
                    "status_global": param_calibration_state.get("status_global", {}),
                    "overall": param_calibration_state.get("overall", {}),
                },
                "columns_only_results_summary": {
                    "validation_metrics": columns_only_results.get("validation_metrics", {}),
                    "full_metrics": columns_only_results.get("full_metrics", {}),
                    "output_file_map": columns_only_results.get("output_file_map", {}),
                },
                "ensemble_seeds_used_for_intervals": ensemble_seeds,
                "uses_column_targets_for_training": True,
                "reactor_transfer_shared_features": shared_transfer_features,
            },
            f,
            indent=2,
        )

    output_file_map = {
        "validation_curve_predictions": preds_path,
        "validation_param_predictions": params_path,
        "validation_curve_metrics": curve_metrics_path,
        "validation_param_metrics": param_metrics_path,
        "validation_group_metrics": validation_group_metrics_path,
        "validation_overall_stats_plot": loocv_overall_plot_path,
        "validation_sample_stats": os.path.join(OUTPUTS_ROOT, "columns_curve_validation_sample_statistics.csv"),
        "full_curve_predictions": full_curve_path,
        "full_param_predictions": full_param_path,
        "full_group_metrics": full_group_metrics_path,
        "full_direct_true_params": direct_combined_path,
        "full_overall_stats_plot": final_overall_plot_path,
    }
    if isinstance(columns_only_results, dict):
        for k, v in (columns_only_results.get("output_file_map", {}) or {}).items():
            output_file_map[f"columns_only__{k}"] = v
    _finalize_experiment_tracking(
        tracker=tracker,
        validation_metrics=curve_metrics,
        full_metrics=full_metrics,
        validation_group_metrics_df=validation_group_metrics_df,
        full_group_metrics_df=full_group_metrics_df,
        validation_param_metrics_df=param_metrics_df,
        loocv_overall_stats_df=loocv_overall_stats_df,
        param_calibration_state=param_calibration_state,
        curve_cfg=curve_cfg,
        shaping_cfg=shaping_cfg,
        coupling_cfg=coupling_cfg,
        uncertainty_cfg=uncertainty_cfg,
        param_calib_cfg=param_calib_cfg,
        ensemble_seeds=ensemble_seeds,
        feature_cols=feature_cols,
        n_samples=len(sample_ids),
        output_file_map=output_file_map,
        columns_only_results=columns_only_results,
    )
    if tracker.get("enabled", False):
        print(
            "Experiment tracking summary saved: "
            f"{tracker.get('summary_path')}"
        )

    print("Done.")


if __name__ == "__main__":
    main()


#%%
