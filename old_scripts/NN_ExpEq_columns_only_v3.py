# %%
import os
import ast
import json
import random
import re
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(THIS_DIR, ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------
# PyTorch Setup
# ---------------------------
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
# Paths and Data
# ---------------------------
DATA_PATH = (
    "/Users/administration/Library/CloudStorage/OneDrive-JettiResources/"
    "Rosetta/database_ready/df_recCu_catcontrol_projects_averaged.csv"
)

TIME_COL_COLUMNS = "leach_duration_days"
TARGET_COLUMNS = "cu_recovery_%"
STATUS_COL_PRIMARY = "project_col_id"
STATUS_COL_FALLBACK = "catalyst_status"
PAIR_ID_COL = "project_sample_id"
CATALYST_CUM_COL = "cumulative_catalyst_addition_kg_t"
EXCLUDED_TRAIN_PAIR_IDS = {
    # "006_jetti_project_file_pvo",
    # "020_jetti_project_file_hypogene_supergene_hypogene_master_composite",
    # "024_jetti_project_file_024cv_cpy",
    # "007b_jetti_project_file_tiger_tgr",
    # "jetti_file_elephant_ii_pq",
}

DEFAULT_PROJECT_ROOT = (
    "/Users/administration/Library/CloudStorage/OneDrive-JettiResources/"
    "Rosetta/NN_Pytorch_ExpEq_columns_only_v3"
)
LOCAL_PROJECT_ROOT = os.path.join(THIS_DIR, "NN_Pytorch_ExpEq_columns_only_v3")
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
INFERENCE_ROOT = os.path.join(OUTPUTS_ROOT, "inference")
INPUT_EXAMPLE_ROOT = os.path.join(THIS_DIR, "input_example")
for p in [PLOTS_ROOT, MODELS_ROOT, OUTPUTS_ROOT]:
    os.makedirs(p, exist_ok=True)
os.makedirs(INFERENCE_ROOT, exist_ok=True)
os.makedirs(INPUT_EXAMPLE_ROOT, exist_ok=True)


# ---------------------------
# Predictors
# ---------------------------
HEADERS_DICT_COLUMNS = {
    "leach_duration_days": ["Leach Duration (days)", "numerical", 1],
    "cumulative_catalyst_addition_kg_t": ["Cumulative Catalyst added (kg/t)", "numerical", 1],
    "cu_%": ["CuT %", "numerical", 1],
    # "acid_soluble_%": ["Acid Soluble Cu (%)", "numerical", 1],
    # "cyanide_soluble_%": ["Cyanide Soluble (%)", "numerical", 0],
    # "residual_cpy_%": ["Residual Chalcopyrite (%)", "numerical", 0],
    "material_size_p80_in": ["Material Size P80 (in)", "numerical", -1],
    # "grouped_copper_sulfides": ["Copper Sulphides (%)", "numerical", 0],
    # "grouped_secondary_copper": ["Secondary Copper (%)", "numerical", 0],
    # "grouped_primary_copper_sulfides": ["Primary Cu Sulphides (%)", "numerical", 0],
    # "grouped_secondary_copper_sulfides": ["Secondary Cu Sulphides (%)", "numerical", 0],
    # "grouped_copper_oxides": ["Copper Oxides (%)", "numerical", 1],
    # "grouped_mixed_copper_ores": ["Mixed Copper Ores (%)", "numerical", 0],
    "grouped_acid_generating_sulfides": ["Acid Generating Sulphides (%)", "numerical", 0],
    "grouped_gangue_silicates": ["Gangue Silicates (%)", "numerical", 0],
    # "grouped_fe_oxides": ["Fe Oxides (%)", "numerical", 0],
    "grouped_carbonates": ["Carbonates (%)", "numerical", 0],
    "column_height_m": ["Column Height (m)", "numerical", 0],
    "column_inner_diameter_m": ["Column Inner Diameter (m)", "numerical", 0],
    # "bornite": ["Bornite (%)", "numerical", 0],
    "fe:cu": ["Fe/Cu ratio", "numerical", 0],
    # "cu:fe": ["Cu/Fe ratio", "numerical", 0],
    "copper_primary_sulfides_equivalent": ["Copper Primary Sulfides Equivalent (%)", "numerical", 0],
    "copper_secondary_sulfides_equivalent": ["Copper Secondary Sulfides Equivalent (%)", "numerical", 0],
    # "copper_sulfides_equivalent": ["Copper Sulfides Equivalent (%)", "numerical", 0],
    "copper_oxides_equivalent": ["Copper Oxides Equivalent (%)", "numerical", 1],
}

PREDICTOR_COLUMNS = list(HEADERS_DICT_COLUMNS.keys())
STATIC_PREDICTOR_COLUMNS = [
    c for c in PREDICTOR_COLUMNS if c not in {TIME_COL_COLUMNS, CATALYST_CUM_COL}
]

GEO_PRIORITY_COLUMNS = ["material_size_p80_in", "column_height_m", "column_inner_diameter_m"]
CHEMISTRY_INTERACTION_COLUMNS = [
    c for c in STATIC_PREDICTOR_COLUMNS if c not in set(GEO_PRIORITY_COLUMNS)
]
MONOTONIC_STATIC_FEATURE_DIRECTIONS = {
    c: int(HEADERS_DICT_COLUMNS[c][2])
    for c in STATIC_PREDICTOR_COLUMNS
    if int(HEADERS_DICT_COLUMNS[c][2]) in {-1, 1}
}
MINERAL_INVENTORY_COLUMNS = [
    "cu_%",
    "acid_soluble_%",
    "cyanide_soluble_%",
    "residual_cpy_%",
    "grouped_copper_sulfides",
    "grouped_secondary_copper",
    "grouped_primary_copper_sulfides",
    "grouped_secondary_copper_sulfides",
    "grouped_copper_oxides",
    "grouped_mixed_copper_ores",
    "bornite",
    "copper_primary_sulfides_equivalent",
    "copper_secondary_sulfides_equivalent",
    "copper_sulfides_equivalent",
    "copper_oxides_equivalent",
]
ACID_REDOX_COLUMNS = [
    "grouped_acid_generating_sulfides",
    "grouped_carbonates",
    "grouped_fe_oxides",
    "fe:cu",
    "cu:fe",
]
TRANSPORT_GANGUE_COLUMNS = [
    "material_size_p80_in",
    "grouped_gangue_silicates",
    "column_height_m",
    "column_inner_diameter_m",
]

CONFIG = {
    "seed": 2026,
    "cv_n_splits": 5,
    # 5 folds x 4 repeats = 20 ensemble members, which gives more stable tail percentiles.
    "cv_n_repeats": 4,
    "cv_random_state": 2026,
    "cv_member_seed_base": 10000,
    "cv_parallel_workers": 10, # max(1, int(os.cpu_count() or 1)),
    # 0 = auto-compute based on worker count.
    "torch_threads_per_worker": 0,
    "torch_interop_threads_per_worker": 1,
    "epochs": 400,
    "patience": 100,
    "learning_rate": 2.5e-3,
    "weight_decay": 2e-5,
    "grad_clip_norm": 5.0,
    "hidden_dim": 96,
    "dropout": 0.08,
    "min_transition_days": 10.0,
    "max_cat_slope_per_day": 0.20,
    "max_catalyst_aging_strength": 5.0,
    "late_tau_impact_decay_strength": 1.15,
    "min_remaining_ore_factor": 0.08,
    "loss_weights": {
        "gap": 1.0,
        "uplift": 0.35,
        "late_uplift": 0.18,
        "monotonic": 0.02,
        "feature_monotonic": 0.05,
        "state_monotonic": 0.04,
        "param": 0.08,
        "smooth_cat": 0.12,
        "slope_cap": 0.18,
        "latent_smooth": 0.02,
        "latent_cat_rate": 0.03,
    },
    "monotonic_step_scaled": 0.25,
    "plot_dpi": 300,
    "ensemble_pi_low": 10,
    "ensemble_pi_high": 90,
    "ensemble_plot_target_day": 2500.0,
    "ensemble_plot_step_days": 1.0,
    "catalyst_extension_window_days": 50.0,
}

# Interaction specs for latent-parameter heuristics.
# Each block uses the intersection between the declared terms here and the
# tensors that are actually available at runtime. Static predictor terms are
# referenced by raw column name, so you can add/remove predictors in
# HEADERS_DICT_COLUMNS and only the currently available ones will participate.
# Control-vs-catalyzed behavior is handled downstream through the cumulative
# catalyst profile: control columns are evaluated with an all-zero catalyst
# history, while catalyzed columns use their observed cumulative additions.
# Because of that, catalyst-sensitive blocks below intentionally encode ore
# receptivity to catalyst/depassivation/transformation rather than the dose
# itself, and they stay availability-aware as predictors are commented in/out.
LATENT_INTERACTION_SPECS: Dict[str, Dict[str, float]] = {
    "chem_raw": {
        "chem_direct": 0.20,
        "cu_%": 0.10,
        "acid_soluble_%": 0.18,
        "cyanide_soluble_%": 0.22,
        "residual_cpy_%": 0.10,
        "grouped_copper_sulfides": 0.08,
        "grouped_secondary_copper": 0.12,
        "grouped_primary_copper_sulfides": 0.10,
        "grouped_secondary_copper_sulfides": 0.18,
        "grouped_copper_oxides": 0.16,
        "grouped_mixed_copper_ores": 0.05,
        "grouped_acid_generating_sulfides": 0.10,
        "grouped_gangue_silicates": -0.10,
        "grouped_fe_oxides": 0.05,
        "grouped_carbonates": -0.12,
        "column_height_m": -0.04,
        "column_inner_diameter_m": -0.04,
        "bornite": 0.10,
        "fe:cu": 0.06,
        "cu:fe": -0.04,
        "copper_primary_sulfides_equivalent": 0.10,
        "copper_secondary_sulfides_equivalent": 0.20,
        "copper_sulfides_equivalent": 0.04,
        "copper_oxides_equivalent": 0.18,
    },
    "primary_passivation_drive": {
        "residual_cpy_%": 0.65,
        "copper_primary_sulfides_equivalent": 0.80,
        "copper_secondary_sulfides_equivalent": -0.30,
        "copper_oxides_equivalent": -0.28,
        "acid_soluble_%": -0.16,
        "cyanide_soluble_%": -0.22,
        "grouped_copper_sulfides": 0.10,
        "grouped_primary_copper_sulfides": 0.40,
        "grouped_secondary_copper": -0.10,
        "grouped_secondary_copper_sulfides": -0.18,
        "grouped_copper_oxides": -0.18,
        "grouped_mixed_copper_ores": 0.06,
        "fe:cu": 0.20,
        "cu:fe": -0.08,
        "material_size_p80_in": 0.18,
        "grouped_acid_generating_sulfides": 0.14,
        "grouped_gangue_silicates": 0.16,
        "grouped_fe_oxides": 0.05,
        "grouped_carbonates": 0.10,
        "column_height_m": 0.12,
        "column_inner_diameter_m": 0.10,
        "bornite": -0.08,
        "copper_sulfides_equivalent": 0.20,
        "cu_%": 0.05,
    },
    "ferric_synergy": {
        "fe:cu": 0.40,
        "cu:fe": -0.12,
        "residual_cpy_%": 0.10,
        "copper_primary_sulfides_equivalent": 0.18,
        "copper_secondary_sulfides_equivalent": 0.12,
        "copper_oxides_equivalent": -0.08,
        "acid_soluble_%": -0.10,
        "cyanide_soluble_%": 0.06,
        "grouped_primary_copper_sulfides": 0.10,
        "grouped_secondary_copper_sulfides": 0.10,
        "grouped_acid_generating_sulfides": 0.35,
        "grouped_gangue_silicates": -0.10,
        "grouped_fe_oxides": 0.08,
        "grouped_carbonates": -0.20,
        "material_size_p80_in": -0.05,
        "column_height_m": -0.08,
        "column_inner_diameter_m": -0.08,
        "bornite": 0.10,
        "chem_raw": 0.08,
        "primary_passivation_drive": 0.12,
    },
    "chem_interaction": {
        "chem_raw": 0.18,
        "primary_passivation_drive": 0.08,
        "copper_primary_sulfides_equivalent": 0.12,
        "acid_soluble_%": 0.08,
        "copper_secondary_sulfides_equivalent": 0.10,
        "copper_oxides_equivalent": 0.06,
        "grouped_secondary_copper": 0.06,
        "grouped_secondary_copper_sulfides": 0.06,
        "grouped_copper_oxides": 0.04,
        "grouped_mixed_copper_ores": -0.04,
        "fe:cu": 0.10,
        "cu:fe": -0.08,
        "grouped_gangue_silicates": -0.08,
        "grouped_carbonates": -0.16,
        "cyanide_soluble_%": 0.10,
        "grouped_acid_generating_sulfides": 0.08,
        "grouped_fe_oxides": 0.03,
        "material_size_p80_in": -0.06,
        "column_height_m": -0.04,
        "column_inner_diameter_m": -0.04,
        "bornite": 0.06,
        "ferric_synergy": 0.18,
    },
    "primary_catalyst_synergy": {
        "residual_cpy_%": 0.40,
        "copper_primary_sulfides_equivalent": 0.42,
        "copper_secondary_sulfides_equivalent": -0.16,
        "copper_oxides_equivalent": -0.16,
        "acid_soluble_%": -0.10,
        "cyanide_soluble_%": -0.08,
        "grouped_primary_copper_sulfides": 0.18,
        "grouped_secondary_copper": -0.08,
        "grouped_secondary_copper_sulfides": -0.08,
        "grouped_copper_oxides": -0.08,
        "fe:cu": 0.10,
        "cu:fe": -0.06,
        "material_size_p80_in": -0.10,
        "grouped_acid_generating_sulfides": 0.12,
        "grouped_gangue_silicates": -0.10,
        "grouped_fe_oxides": 0.04,
        "grouped_carbonates": -0.12,
        "column_height_m": -0.08,
        "column_inner_diameter_m": -0.08,
        "bornite": 0.06,
        "copper_sulfides_equivalent": 0.10,
        "chem_interaction": 0.10,
        "primary_passivation_drive": 0.20,
        "ferric_synergy": 0.15,
    },
    "fast_leach_inventory": {
        "acid_soluble_%": 0.36,
        "cyanide_soluble_%": 0.34,
        "residual_cpy_%": -0.28,
        "copper_primary_sulfides_equivalent": -0.30,
        "copper_secondary_sulfides_equivalent": 0.42,
        "copper_oxides_equivalent": 0.30,
        "grouped_secondary_copper": 0.18,
        "grouped_primary_copper_sulfides": -0.18,
        "grouped_secondary_copper_sulfides": 0.20,
        "grouped_copper_oxides": 0.24,
        "grouped_mixed_copper_ores": 0.12,
        "bornite": 0.14,
        "cu_%": 0.08,
        "copper_sulfides_equivalent": -0.08,
        "grouped_acid_generating_sulfides": -0.15,
        "chem_raw": 0.08,
        "primary_passivation_drive": -0.12,
    },
    "oxide_inventory": {
        "copper_oxides_equivalent": 0.70,
        "acid_soluble_%": 0.28,
        "cyanide_soluble_%": -0.15,
        "residual_cpy_%": -0.08,
        "copper_primary_sulfides_equivalent": -0.10,
        "copper_secondary_sulfides_equivalent": -0.08,
        "grouped_copper_oxides": 0.30,
        "grouped_mixed_copper_ores": 0.10,
        "grouped_secondary_copper_sulfides": -0.08,
    },
    "acid_buffer_strength": {
        "grouped_carbonates": 0.80,
        "grouped_gangue_silicates": 0.28,
        "grouped_fe_oxides": 0.08,
        "acid_soluble_%": -0.12,
        "copper_oxides_equivalent": -0.08,
        "grouped_copper_oxides": -0.06,
        "grouped_acid_generating_sulfides": -0.16,
        "fe:cu": -0.04,
        "cu:fe": 0.04,
    },
    "acid_buffer_decay": {
        "grouped_carbonates": 0.36,
        "acid_soluble_%": 0.18,
        "copper_oxides_equivalent": 0.12,
        "grouped_copper_oxides": 0.10,
        "cu_%": 0.08,
        "grouped_acid_generating_sulfides": 0.16,
        "fe:cu": 0.08,
        "grouped_gangue_silicates": -0.18,
        "grouped_fe_oxides": -0.06,
        "material_size_p80_in": -0.06,
        "column_height_m": -0.04,
        "column_inner_diameter_m": -0.04,
    },
    "diffusion_drag_strength": {
        "grouped_gangue_silicates": 0.38,
        "material_size_p80_in": 0.30,
        "column_height_m": 0.18,
        "column_inner_diameter_m": 0.16,
        "residual_cpy_%": 0.10,
        "copper_primary_sulfides_equivalent": 0.16,
        "copper_secondary_sulfides_equivalent": -0.10,
        "copper_oxides_equivalent": -0.08,
        "acid_soluble_%": -0.08,
        "cyanide_soluble_%": -0.08,
        "grouped_primary_copper_sulfides": 0.08,
        "grouped_secondary_copper_sulfides": -0.06,
        "grouped_acid_generating_sulfides": 0.06,
        "grouped_fe_oxides": 0.06,
        "grouped_carbonates": 0.06,
        "bornite": -0.04,
        "fast_leach_inventory": -0.10,
        "oxide_inventory": -0.06,
    },
    "surface_refresh": {
        "material_size_p80_in": -0.35,
        "acid_soluble_%": 0.10,
        "cyanide_soluble_%": 0.10,
        "copper_oxides_equivalent": 0.20,
        "copper_secondary_sulfides_equivalent": 0.15,
        "residual_cpy_%": -0.08,
        "grouped_secondary_copper": 0.06,
        "grouped_secondary_copper_sulfides": 0.08,
        "grouped_copper_oxides": 0.10,
        "grouped_gangue_silicates": -0.10,
        "grouped_carbonates": -0.06,
        "column_height_m": -0.14,
        "column_inner_diameter_m": -0.12,
        "bornite": 0.10,
        "chem_interaction": 0.08,
        "ferric_synergy": 0.06,
        "primary_passivation_drive": -0.10,
        "diffusion_drag_strength": -0.18,
    },
    "bed_permeability": {
        "material_size_p80_in": 0.35,
        "grouped_gangue_silicates": -0.20,
        "grouped_carbonates": -0.08,
        "grouped_fe_oxides": -0.10,
        "column_height_m": -0.18,
        "column_inner_diameter_m": -0.12,
        "grouped_acid_generating_sulfides": -0.04,
        "acid_buffer_strength": -0.10,
        "diffusion_drag_strength": -0.25,
        "surface_refresh": 0.08,
    },
    "wetting_efficiency": {
        "material_size_p80_in": -0.15,
        "grouped_gangue_silicates": -0.14,
        "grouped_carbonates": -0.08,
        "column_height_m": -0.12,
        "column_inner_diameter_m": -0.10,
        "acid_soluble_%": 0.06,
        "copper_oxides_equivalent": 0.08,
        "fast_leach_inventory": 0.10,
        "surface_refresh": 0.18,
        "bed_permeability": 0.22,
        "diffusion_drag_strength": -0.18,
        "acid_buffer_strength": -0.10,
    },
    "particle_accessibility": {
        "material_size_p80_in": -0.35,
        "residual_cpy_%": -0.16,
        "copper_primary_sulfides_equivalent": -0.18,
        "copper_secondary_sulfides_equivalent": 0.12,
        "copper_oxides_equivalent": 0.12,
        "acid_soluble_%": 0.08,
        "cyanide_soluble_%": 0.08,
        "grouped_secondary_copper_sulfides": 0.08,
        "bornite": 0.08,
        "wetting_efficiency": 0.28,
        "bed_permeability": 0.08,
        "diffusion_drag_strength": -0.20,
        "primary_passivation_drive": -0.12,
        "surface_refresh": 0.12,
    },
    "pore_decay_tendency": {
        "residual_cpy_%": 0.12,
        "grouped_acid_generating_sulfides": 0.16,
        "grouped_gangue_silicates": 0.14,
        "grouped_carbonates": 0.10,
        "grouped_fe_oxides": 0.10,
        "fe:cu": 0.10,
        "cu:fe": -0.06,
        "material_size_p80_in": -0.14,
        "column_height_m": 0.14,
        "column_inner_diameter_m": 0.08,
        "primary_passivation_drive": 0.18,
        "ferric_synergy": 0.10,
        "acid_buffer_strength": 0.12,
        "diffusion_drag_strength": 0.10,
        "surface_refresh": -0.16,
        "bed_permeability": -0.10,
    },
    "ore_decay_strength": {
        "chem_raw": 0.10,
        "residual_cpy_%": 0.10,
        "copper_primary_sulfides_equivalent": 0.12,
        "grouped_gangue_silicates": 0.08,
        "grouped_carbonates": 0.06,
        "grouped_acid_generating_sulfides": -0.04,
        "column_height_m": 0.06,
        "column_inner_diameter_m": 0.05,
        "bornite": -0.04,
        "primary_passivation_drive": 0.24,
        "fast_leach_inventory": -0.18,
        "oxide_inventory": -0.10,
        "acid_buffer_strength": 0.12,
        "diffusion_drag_strength": 0.22,
        "surface_refresh": -0.08,
    },
    "passivation_strength": {
        "residual_cpy_%": 0.18,
        "copper_primary_sulfides_equivalent": 0.18,
        "copper_secondary_sulfides_equivalent": -0.12,
        "copper_oxides_equivalent": -0.10,
        "cyanide_soluble_%": -0.08,
        "grouped_gangue_silicates": 0.08,
        "grouped_fe_oxides": 0.06,
        "grouped_carbonates": 0.12,
        "grouped_acid_generating_sulfides": 0.10,
        "fe:cu": 0.12,
        "bornite": -0.06,
        "primary_passivation_drive": 0.50,
        "diffusion_drag_strength": 0.12,
        "ore_decay_strength": 0.10,
        "surface_refresh": -0.10,
    },
    "depassivation_strength": {
        "residual_cpy_%": 0.16,
        "copper_primary_sulfides_equivalent": 0.18,
        "copper_secondary_sulfides_equivalent": -0.08,
        "copper_oxides_equivalent": -0.10,
        "acid_soluble_%": -0.08,
        "cyanide_soluble_%": -0.06,
        "material_size_p80_in": -0.10,
        "grouped_acid_generating_sulfides": 0.12,
        "grouped_gangue_silicates": -0.12,
        "grouped_carbonates": -0.12,
        "column_height_m": -0.08,
        "column_inner_diameter_m": -0.08,
        "bornite": 0.06,
        "fe:cu": 0.08,
        "primary_passivation_drive": 0.32,
        "ferric_synergy": 0.24,
        "chem_interaction": 0.10,
        "primary_catalyst_synergy": 0.24,
        "acid_buffer_strength": -0.10,
        "diffusion_drag_strength": -0.16,
        "surface_refresh": 0.20,
        "ore_decay_strength": -0.08,
    },
    "transform_strength": {
        "chem_raw": 0.12,
        "residual_cpy_%": 0.12,
        "copper_primary_sulfides_equivalent": 0.14,
        "copper_secondary_sulfides_equivalent": 0.08,
        "copper_oxides_equivalent": -0.06,
        "acid_soluble_%": -0.04,
        "cyanide_soluble_%": 0.08,
        "material_size_p80_in": -0.08,
        "grouped_mixed_copper_ores": 0.06,
        "grouped_acid_generating_sulfides": 0.10,
        "grouped_gangue_silicates": -0.10,
        "grouped_carbonates": -0.14,
        "column_height_m": -0.06,
        "column_inner_diameter_m": -0.06,
        "bornite": 0.08,
        "fe:cu": 0.06,
        "primary_passivation_drive": 0.20,
        "ferric_synergy": 0.20,
        "chem_interaction": 0.10,
        "primary_catalyst_synergy": 0.22,
        "acid_buffer_strength": -0.08,
        "diffusion_drag_strength": -0.12,
        "surface_refresh": 0.08,
        "depassivation_strength": 0.10,
    },
}


# ---------------------------
# Utilities
# ---------------------------
def enable_torch_determinism(deterministic: bool = True) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except Exception:
        pass


def set_all_seeds(seed: Optional[int] = None, deterministic: bool = True) -> None:
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        enable_torch_determinism(True)


def configure_torch_cpu_parallelism(
    num_threads: Optional[int] = None,
    num_interop_threads: Optional[int] = None,
) -> Dict[str, int]:
    total_cores = max(1, int(os.cpu_count() or 1))

    threads = int(num_threads) if num_threads is not None else 0
    if threads <= 0:
        threads = total_cores
    threads = max(1, min(threads, total_cores))
    torch.set_num_threads(int(threads))

    interop_threads = int(num_interop_threads) if num_interop_threads is not None else 0
    if interop_threads <= 0:
        interop_threads = 1
    interop_threads = max(1, min(interop_threads, total_cores))
    try:
        # Can only be configured once per process in PyTorch.
        torch.set_num_interop_threads(int(interop_threads))
    except RuntimeError:
        pass
    except Exception:
        pass

    return {
        "total_cores": int(total_cores),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def resolve_cv_parallelism(n_members: int) -> Dict[str, int]:
    total_cores = max(1, int(os.cpu_count() or 1))

    requested_workers = int(CONFIG.get("cv_parallel_workers", total_cores))
    if requested_workers <= 0:
        requested_workers = total_cores
    workers = max(1, min(requested_workers, max(1, int(n_members))))

    requested_threads = int(CONFIG.get("torch_threads_per_worker", 0))
    if requested_threads <= 0:
        torch_threads = max(1, total_cores // workers)
    else:
        torch_threads = requested_threads
    torch_threads = max(1, min(torch_threads, total_cores))

    requested_interop = int(CONFIG.get("torch_interop_threads_per_worker", 1))
    if requested_interop <= 0:
        requested_interop = 1
    torch_interop_threads = max(1, min(requested_interop, total_cores))

    return {
        "workers": int(workers),
        "total_cores": int(total_cores),
        "torch_threads_per_worker": int(torch_threads),
        "torch_interop_threads_per_worker": int(torch_interop_threads),
    }


def load_torch_checkpoint(path: str, map_location: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def serialize_array(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=float)
    out = [None if not np.isfinite(v) else float(v) for v in arr]
    return json.dumps(out)


def parse_listlike(value: Any) -> np.ndarray:
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
            tokens = re.findall(
                r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?|nan",
                inner,
                flags=re.IGNORECASE,
            )
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


def normalize_status(value: Any) -> str:
    s = str(value).strip().lower()
    if ("cat" in s and "no" not in s) or ("with" in s and "no" not in s):
        return "Catalyzed"
    return "Control"


def scalar_from_maybe_array(value: Any) -> float:
    arr = parse_listlike(value)
    if arr.size > 0:
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            return float(np.nanmedian(finite))
    try:
        return float(value)
    except Exception:
        return np.nan


def align_profile_to_time_length(profile: np.ndarray, time_length: int) -> np.ndarray:
    p = np.asarray(profile, dtype=float)
    if time_length <= 0:
        return np.asarray([], dtype=float)
    if p.size == time_length:
        return p
    if p.size == 0:
        return np.zeros(time_length, dtype=float)
    if p.size > time_length:
        return p[:time_length]
    pad_val = p[-1] if np.isfinite(p[-1]) else 0.0
    pad = np.full(time_length - p.size, pad_val, dtype=float)
    return np.concatenate([p, pad])


def clean_cumulative_profile(profile: np.ndarray, force_zero: bool = False) -> np.ndarray:
    c = np.asarray(profile, dtype=float).copy()
    if c.size == 0:
        return np.asarray([], dtype=float)
    if force_zero:
        return np.zeros_like(c)
    c = np.where(np.isfinite(c), c, np.nan)
    if np.all(np.isnan(c)):
        c[:] = 0.0
    else:
        last = 0.0
        for i in range(c.size):
            if np.isnan(c[i]):
                c[i] = last
            else:
                last = c[i]
    c = np.clip(c, 0.0, None)
    c = np.maximum.accumulate(c)
    return c


def average_weekly_catalyst_from_recent_history(
    time_days: np.ndarray,
    catalyst_cum: np.ndarray,
    window_days: float = 21.0,
    week_days: float = 7.0,
) -> Tuple[float, float]:
    t = np.asarray(time_days, dtype=float)
    c = clean_cumulative_profile(np.asarray(catalyst_cum, dtype=float), force_zero=False)
    valid = np.isfinite(t) & np.isfinite(c)
    if valid.sum() == 0:
        return 0.0, 0.0

    t = t[valid]
    c = c[valid]
    order = np.argsort(t)
    t = t[order]
    c = c[order]

    t_unique, inv = np.unique(t, return_inverse=True)
    c_unique = np.full(t_unique.shape, np.nan, dtype=float)
    for i, j in enumerate(inv):
        c_unique[j] = c[i] if not np.isfinite(c_unique[j]) else max(c_unique[j], c[i])
    t = t_unique
    c = clean_cumulative_profile(c_unique, force_zero=False)

    if t.size <= 1:
        return 0.0, 0.0

    final_day = float(t[-1])
    start_day = max(float(t[0]), final_day - float(window_days))
    effective_window_days = float(max(0.0, final_day - start_day))
    if effective_window_days <= 1e-9:
        return 0.0, effective_window_days

    c_start = float(np.interp(start_day, t, c))
    c_end = float(c[-1])
    avg_daily_addition = max(0.0, (c_end - c_start) / effective_window_days)
    return float(avg_daily_addition * float(week_days)), effective_window_days


def infer_catalyst_addition_start_day(time_days: np.ndarray, catalyst_cum: np.ndarray) -> float:
    t = np.asarray(time_days, dtype=float)
    c = clean_cumulative_profile(np.asarray(catalyst_cum, dtype=float), force_zero=False)
    valid = np.isfinite(t) & np.isfinite(c)
    if valid.sum() == 0:
        return np.nan

    t = t[valid]
    c = c[valid]
    order = np.argsort(t)
    t = t[order]
    c = c[order]

    t_unique, inv = np.unique(t, return_inverse=True)
    c_unique = np.full(t_unique.shape, np.nan, dtype=float)
    for i, j in enumerate(inv):
        c_unique[j] = c[i] if not np.isfinite(c_unique[j]) else max(c_unique[j], c[i])
    t = t_unique.astype(float)
    c = clean_cumulative_profile(c_unique, force_zero=False)

    if t.size == 0:
        return np.nan

    tol = max(1e-9, 1e-6 * float(max(1.0, np.nanmax(np.abs(c)))))
    if c[0] > tol:
        return float(t[0])

    if t.size <= 1:
        return np.nan

    rise_idx = np.flatnonzero(np.diff(c) > tol)
    if rise_idx.size == 0:
        return np.nan
    return float(t[int(rise_idx[0])])


def infer_catalyst_addition_stop_day(time_days: np.ndarray, catalyst_cum: np.ndarray) -> float:
    t = np.asarray(time_days, dtype=float)
    c = clean_cumulative_profile(np.asarray(catalyst_cum, dtype=float), force_zero=False)
    valid = np.isfinite(t) & np.isfinite(c)
    if valid.sum() == 0:
        return np.nan

    t = t[valid]
    c = c[valid]
    order = np.argsort(t)
    t = t[order]
    c = c[order]

    t_unique, inv = np.unique(t, return_inverse=True)
    c_unique = np.full(t_unique.shape, np.nan, dtype=float)
    for i, j in enumerate(inv):
        c_unique[j] = c[i] if not np.isfinite(c_unique[j]) else max(c_unique[j], c[i])
    t = t_unique.astype(float)
    c = clean_cumulative_profile(c_unique, force_zero=False)

    if t.size == 0:
        return np.nan

    tol = max(1e-9, 1e-6 * float(max(1.0, np.nanmax(np.abs(c)))))
    if t.size == 1:
        return float(t[0]) if c[0] > tol else np.nan

    rise_idx = np.flatnonzero(np.diff(c) > tol)
    if rise_idx.size == 0:
        return float(t[0]) if c[0] > tol else np.nan

    stop_idx = int(rise_idx[-1] + 1)
    if stop_idx >= t.size - 1:
        return np.nan
    return float(t[stop_idx])


def summarize_catalyst_addition_behavior(
    time_days: np.ndarray,
    catalyst_cum: np.ndarray,
    history_window_days: float,
) -> Dict[str, Any]:
    t = np.asarray(time_days, dtype=float)
    c = clean_cumulative_profile(np.asarray(catalyst_cum, dtype=float), force_zero=False)
    valid = np.isfinite(t) & np.isfinite(c)
    if valid.sum() == 0:
        return {
            "catalyst_addition_start_day": np.nan,
            "catalyst_addition_stop_day": np.nan,
            "weekly_catalyst_addition_kg_t": 0.0,
            "weekly_reference_days": 0.0,
            "recent_window_start_day": np.nan,
            "recent_window_delta_kg_t": 0.0,
            "recent_window_delta_tol_kg_t": 0.0,
            "recent_window_growth_near_zero": True,
            "last_observed_day": np.nan,
            "stopped_before_test_end": False,
            "catalyst_addition_state": "no_valid_profile",
        }

    t = t[valid]
    c = c[valid]
    order = np.argsort(t)
    t = t[order]
    c = c[order]

    t_unique, inv = np.unique(t, return_inverse=True)
    c_unique = np.full(t_unique.shape, np.nan, dtype=float)
    for i, j in enumerate(inv):
        c_unique[j] = c[i] if not np.isfinite(c_unique[j]) else max(c_unique[j], c[i])
    t = t_unique.astype(float)
    c = clean_cumulative_profile(c_unique, force_zero=False)

    start_day = infer_catalyst_addition_start_day(t, c)
    stop_day = infer_catalyst_addition_stop_day(t, c)
    weekly_value, reference_days = average_weekly_catalyst_from_recent_history(
        time_days=t,
        catalyst_cum=c,
        window_days=history_window_days,
        week_days=7.0,
    )
    last_day = float(t[-1])
    recent_window_start_day = max(float(t[0]), last_day - float(history_window_days))
    if last_day > recent_window_start_day + 1e-9:
        c_start = float(np.interp(recent_window_start_day, t, c))
        c_end = float(c[-1])
        recent_window_delta = max(0.0, c_end - c_start)
    else:
        recent_window_delta = 0.0
    recent_window_tol = max(1e-9, 1e-6 * float(max(1.0, np.nanmax(np.abs(c)))))
    recent_growth_near_zero = bool(recent_window_delta <= recent_window_tol)
    stopped_before_test_end = bool(
        recent_growth_near_zero
        and np.isfinite(stop_day)
        and np.isfinite(last_day)
        and stop_day < last_day - 1e-9
    )

    if not np.isfinite(start_day):
        state = "no_catalyst_added"
    elif stopped_before_test_end:
        state = "stopped_before_test_end"
    else:
        state = "still_adding_at_end"

    return {
        "catalyst_addition_start_day": float(start_day) if np.isfinite(start_day) else np.nan,
        "catalyst_addition_stop_day": float(stop_day) if np.isfinite(stop_day) else np.nan,
        "weekly_catalyst_addition_kg_t": float(weekly_value),
        "weekly_reference_days": float(reference_days),
        "recent_window_start_day": float(recent_window_start_day),
        "recent_window_delta_kg_t": float(recent_window_delta),
        "recent_window_delta_tol_kg_t": float(recent_window_tol),
        "recent_window_growth_near_zero": bool(recent_growth_near_zero),
        "last_observed_day": float(last_day),
        "stopped_before_test_end": bool(stopped_before_test_end),
        "catalyst_addition_state": state,
    }


def build_catalyst_stop_report(
    pairs: List["PairSample"],
    history_window_days: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        row = {
            "sample_id": pair.sample_id,
            **summarize_catalyst_addition_behavior(
                time_days=pair.catalyzed.time,
                catalyst_cum=pair.catalyzed.catalyst_cum,
                history_window_days=history_window_days,
            ),
        }
        rows.append(row)

    report_df = pd.DataFrame(rows)
    if not report_df.empty:
        report_df = report_df.sort_values(
            ["stopped_before_test_end", "sample_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

    stopped_ids = (
        report_df.loc[report_df["stopped_before_test_end"], "sample_id"].astype(str).tolist()
        if not report_df.empty
        else []
    )
    state_counts = (
        report_df["catalyst_addition_state"].value_counts(dropna=False).to_dict()
        if not report_df.empty
        else {}
    )
    summary = {
        "history_window_days": float(history_window_days),
        "n_pairs": int(len(report_df)),
        "n_stopped_before_test_end": int(len(stopped_ids)),
        "stopped_sample_ids": stopped_ids,
        "state_counts": state_counts,
    }
    return report_df, summary


def build_plot_time_grid(
    observed_time_days: np.ndarray,
    start_day: float,
    target_day: float,
    step_days: float = 1.0,
) -> np.ndarray:
    obs = np.asarray(observed_time_days, dtype=float)
    obs = obs[np.isfinite(obs)]
    end_day = float(max(float(target_day), float(np.nanmax(obs)) if obs.size else float(target_day)))
    start_day = float(start_day)
    if end_day <= start_day:
        return np.asarray([start_day], dtype=float)

    grid = np.arange(start_day, end_day + float(step_days), float(step_days), dtype=float)
    grid = grid[grid <= end_day + 1e-9]
    anchors = obs[(obs >= start_day - 1e-9) & (obs <= end_day + 1e-9)]
    out = np.unique(np.concatenate([grid, anchors, np.asarray([start_day, end_day], dtype=float)]))
    return out.astype(float)


def interpolate_cumulative_profile(
    query_time_days: np.ndarray,
    anchor_time_days: np.ndarray,
    anchor_catalyst_cum: np.ndarray,
) -> np.ndarray:
    qt = np.asarray(query_time_days, dtype=float)
    at = np.asarray(anchor_time_days, dtype=float)
    ac = clean_cumulative_profile(np.asarray(anchor_catalyst_cum, dtype=float), force_zero=False)
    valid = np.isfinite(qt)
    if valid.sum() == 0:
        return np.asarray([], dtype=float)
    if at.size == 0 or ac.size == 0:
        return np.zeros(valid.sum(), dtype=float)
    if at.size == 1:
        return np.full(valid.sum(), float(ac[0]), dtype=float)
    if at.size >= 3:
        interp = PchipInterpolator(at, ac, extrapolate=False)
        out = np.asarray(interp(qt[valid]), dtype=float)
    else:
        out = np.interp(qt[valid], at, ac)
    return clean_cumulative_profile(out, force_zero=False)


def extend_catalyst_profile_for_ensemble_plot(
    time_days: np.ndarray,
    catalyst_cum: np.ndarray,
    target_day: float,
    step_days: float = 7.0,
    history_window_days: float = 21.0,
) -> Dict[str, Any]:
    t = np.asarray(time_days, dtype=float)
    c = clean_cumulative_profile(np.asarray(catalyst_cum, dtype=float), force_zero=False)
    valid = np.isfinite(t) & np.isfinite(c)
    if valid.sum() == 0:
        return {
            "time_days": np.asarray([], dtype=float),
            "catalyst_cum": np.asarray([], dtype=float),
            "catalyst_addition_start_day": np.nan,
            "catalyst_addition_stop_day": np.nan,
            "weekly_catalyst_addition_kg_t": 0.0,
            "weekly_reference_days": 0.0,
            "recent_window_start_day": np.nan,
            "recent_window_delta_kg_t": 0.0,
            "recent_window_delta_tol_kg_t": 0.0,
            "recent_window_growth_near_zero": True,
            "last_observed_day": np.nan,
            "stopped_before_test_end": False,
            "catalyst_addition_state": "no_valid_profile",
            "extension_applied": False,
            "target_day": float(target_day),
        }

    t = t[valid]
    c = c[valid]
    order = np.argsort(t)
    t = t[order]
    c = c[order]

    t_unique, inv = np.unique(t, return_inverse=True)
    c_unique = np.full(t_unique.shape, np.nan, dtype=float)
    for i, j in enumerate(inv):
        c_unique[j] = c[i] if not np.isfinite(c_unique[j]) else max(c_unique[j], c[i])
    t = t_unique.astype(float)
    c = clean_cumulative_profile(c_unique, force_zero=False)

    catalyst_behavior = summarize_catalyst_addition_behavior(
        time_days=t,
        catalyst_cum=c,
        history_window_days=history_window_days,
    )
    weekly_value = float(catalyst_behavior["weekly_catalyst_addition_kg_t"])
    last_day = float(catalyst_behavior["last_observed_day"])

    if last_day >= float(target_day) - 1e-9:
        dense_time_days = build_plot_time_grid(
            observed_time_days=t,
            start_day=float(t[0]),
            target_day=last_day,
            step_days=float(step_days),
        )
        return {
            "time_days": dense_time_days,
            "catalyst_cum": interpolate_cumulative_profile(dense_time_days, t, c),
            **catalyst_behavior,
            "extension_applied": False,
            "target_day": float(target_day),
        }

    future_days = []
    next_day = last_day + float(step_days)
    while next_day < float(target_day) - 1e-9:
        future_days.append(next_day)
        next_day += float(step_days)
    future_days.append(float(target_day))

    future_days_arr = np.asarray(future_days, dtype=float)
    avg_daily_addition = weekly_value / 7.0
    future_cum = float(c[-1]) + avg_daily_addition * (future_days_arr - last_day)
    extended_time_days = np.concatenate([t, future_days_arr])
    extended_catalyst_cum = np.concatenate([c, future_cum])
    dense_time_days = build_plot_time_grid(
        observed_time_days=extended_time_days,
        start_day=float(t[0]),
        target_day=float(target_day),
        step_days=float(step_days),
    )
    dense_catalyst_cum = interpolate_cumulative_profile(dense_time_days, extended_time_days, extended_catalyst_cum)

    return {
        "time_days": dense_time_days,
        "catalyst_cum": dense_catalyst_cum,
        **catalyst_behavior,
        "extension_applied": True,
        "target_day": float(target_day),
    }


def build_shared_ensemble_plot_profile(
    time_days: np.ndarray,
    catalyst_cum: np.ndarray,
    target_day: float,
    step_days: float = 1.0,
    history_window_days: float = 21.0,
) -> Dict[str, Any]:
    profile = extend_catalyst_profile_for_ensemble_plot(
        time_days=time_days,
        catalyst_cum=catalyst_cum,
        target_day=target_day,
        step_days=step_days,
        history_window_days=history_window_days,
    )
    plot_time_days = build_plot_time_grid(
        observed_time_days=np.asarray(profile["time_days"], dtype=float),
        start_day=0.0,
        target_day=float(profile["target_day"]),
        step_days=float(step_days),
    )
    plot_catalyst_cum = interpolate_cumulative_profile(
        query_time_days=plot_time_days,
        anchor_time_days=np.asarray(profile["time_days"], dtype=float),
        anchor_catalyst_cum=np.asarray(profile["catalyst_cum"], dtype=float),
    )
    return {
        **profile,
        "plot_time_days": np.asarray(plot_time_days, dtype=float),
        "plot_catalyst_cum": np.asarray(plot_catalyst_cum, dtype=float),
    }


def prepare_curve_arrays(
    time_arr: np.ndarray,
    rec_arr: np.ndarray,
    cum_arr: Optional[np.ndarray],
    status: str,
    min_points: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(time_arr, dtype=float)
    y = np.asarray(rec_arr, dtype=float)
    if cum_arr is None:
        c = np.zeros_like(t, dtype=float)
    else:
        c = np.asarray(cum_arr, dtype=float)
    n = min(len(t), len(y))
    if c.size > 0:
        n = min(n, len(c))
    t = t[:n]
    y = y[:n]
    if c.size == 0:
        c = np.zeros(n, dtype=float)
    else:
        c = c[:n]
    valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() < min_points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    t = t[valid]
    y = y[valid]
    c = c[valid]
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    c = c[order]

    t_unique, inv = np.unique(t, return_inverse=True)
    y_unique = np.zeros_like(t_unique, dtype=float)
    y_counts = np.zeros_like(t_unique, dtype=float)
    c_unique = np.full_like(t_unique, np.nan, dtype=float)
    for i, j in enumerate(inv):
        y_unique[j] += y[i]
        y_counts[j] += 1.0
        if np.isfinite(c[i]):
            if not np.isfinite(c_unique[j]):
                c_unique[j] = c[i]
            else:
                c_unique[j] = max(c_unique[j], c[i])
    y = np.clip(y_unique / np.maximum(y_counts, 1.0), 0.0, 100.0)
    c = clean_cumulative_profile(c_unique, force_zero=(status == "Control"))
    t = t_unique.astype(float)
    if t.size < min_points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    return t, y, c


# ---------------------------
# Biexponential fitting
# ---------------------------
def double_exp_curve_np(t_days: np.ndarray, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
    t = np.clip(np.asarray(t_days, dtype=float), 0.0, None)
    return float(a1) * (1.0 - np.exp(-float(b1) * t)) + float(a2) * (1.0 - np.exp(-float(b2) * t))


def enforce_fast_slow_pairing(params: np.ndarray) -> np.ndarray:
    p = np.asarray(params, dtype=float).copy()
    if p.size != 4 or not np.all(np.isfinite(p)):
        return p
    if p[3] > p[1]:
        p = np.array([p[2], p[3], p[0], p[1]], dtype=float)
    return p


def sanitize_curve_params(params: np.ndarray) -> np.ndarray:
    p = np.asarray(params, dtype=float).copy()
    if p.size != 4 or not np.all(np.isfinite(p)):
        p = np.array([20.0, 0.02, 10.0, 0.003], dtype=float)
    p[0] = max(0.0, p[0])
    p[2] = max(0.0, p[2])
    p[1] = np.clip(p[1], 1e-5, 2.0)
    p[3] = np.clip(p[3], 1e-5, 2.0)
    p = enforce_fast_slow_pairing(p)
    total = p[0] + p[2]
    if total > 100.0 and total > 0:
        scale = 100.0 / total
        p[0] *= scale
        p[2] *= scale
    return p


def fit_biexponential_params(t_days: np.ndarray, recovery: np.ndarray) -> np.ndarray:
    t = np.asarray(t_days, dtype=float)
    y = np.asarray(recovery, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() < 6:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
    t = t[valid]
    y = np.clip(y[valid], 0.0, 100.0)
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    y_max = float(np.nanmax(y))
    plateau = float(np.nanpercentile(y, 90))
    p0_list = [
        [0.6 * plateau, 0.03, 0.4 * plateau, 0.004],
        [0.8 * y_max, 0.06, 0.2 * y_max, 0.008],
        [0.5 * plateau, 0.02, 0.5 * plateau, 0.002],
        [y_max, 0.03, 1.0, 0.002],
    ]
    lower = [0.0, 1e-5, 0.0, 1e-5]
    upper = [100.0, 2.0, 100.0, 2.0]

    best = None
    best_loss = np.inf
    for p0 in p0_list:
        p0 = sanitize_curve_params(np.asarray(p0, dtype=float))
        try:
            popt, _ = curve_fit(
                lambda t_, a1, b1, a2, b2: double_exp_curve_np(t_, a1, b1, a2, b2),
                t,
                y,
                p0=p0,
                bounds=(lower, upper),
                maxfev=30000,
            )
            popt = sanitize_curve_params(np.asarray(popt, dtype=float))
            pred = double_exp_curve_np(t, popt[0], popt[1], popt[2], popt[3])
            loss = float(np.mean((pred - y) ** 2))
            if loss < best_loss:
                best_loss = loss
                best = popt
        except Exception:
            continue
    if best is None:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
    return sanitize_curve_params(best)


def _bounds_from_param_values(values: np.ndarray, is_rate: bool) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        if is_rate:
            return 1e-5, 0.2
        return 0.0, 60.0
    q05 = float(np.nanquantile(vals, 0.05))
    q95 = float(np.nanquantile(vals, 0.95))
    span = max(1e-4, q95 - q05)
    lo = q05 - 0.25 * span
    hi = q95 + 0.25 * span
    if is_rate:
        lo = max(1e-5, lo)
        hi = max(lo + 1e-5, hi)
        hi = min(2.0, hi)
    else:
        lo = max(0.0, lo)
        hi = max(lo + 1e-3, min(100.0, hi))
    return float(lo), float(hi)


def derive_param_bounds(
    params_matrix: np.ndarray,
    fallback_params_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(params_matrix, dtype=float)
    fb = np.asarray(fallback_params_matrix, dtype=float) if fallback_params_matrix is not None else np.asarray([], dtype=float)
    if p.ndim != 2 or p.shape[1] != 4 or p.shape[0] < 2:
        p = fb
    if p.ndim != 2 or p.shape[1] != 4:
        p = np.asarray(
            [
                [20.0, 0.020, 10.0, 0.004],
                [25.0, 0.030, 12.0, 0.006],
            ],
            dtype=float,
        )

    lo_hi = [
        _bounds_from_param_values(p[:, 0], is_rate=False),
        _bounds_from_param_values(p[:, 1], is_rate=True),
        _bounds_from_param_values(p[:, 2], is_rate=False),
        _bounds_from_param_values(p[:, 3], is_rate=True),
    ]
    lb = np.array([x[0] for x in lo_hi], dtype=float)
    ub = np.array([x[1] for x in lo_hi], dtype=float)
    return lb, ub


# ---------------------------
# Data objects
# ---------------------------
@dataclass
class CurveData:
    status: str
    time: np.ndarray
    recovery: np.ndarray
    catalyst_cum: np.ndarray
    fit_params: np.ndarray
    row_index: int


@dataclass
class PairSample:
    sample_id: str
    static_raw: np.ndarray
    control: CurveData
    catalyzed: CurveData
    static_scaled: Optional[np.ndarray] = None
    static_missing_mask: Optional[np.ndarray] = None


def combine_static_vectors(control_vec: np.ndarray, catalyzed_vec: np.ndarray) -> np.ndarray:
    ctrl = np.asarray(control_vec, dtype=float)
    cat = np.asarray(catalyzed_vec, dtype=float)
    out = cat.copy()
    missing = ~np.isfinite(out)
    out[missing] = ctrl[missing]
    if np.any(~np.isfinite(out)):
        stacked = np.vstack([ctrl, cat])
        med = np.nanmedian(stacked, axis=0)
        out[~np.isfinite(out)] = med[~np.isfinite(out)]
    return out


# ---------------------------
# Analysis and parsing
# ---------------------------
def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "predictor_columns": PREDICTOR_COLUMNS,
        "static_predictor_columns": STATIC_PREDICTOR_COLUMNS,
    }
    if PAIR_ID_COL in df.columns:
        summary["n_unique_pairs"] = int(df[PAIR_ID_COL].nunique(dropna=True))

    if STATUS_COL_PRIMARY in df.columns or STATUS_COL_FALLBACK in df.columns:
        status_source = STATUS_COL_PRIMARY if STATUS_COL_PRIMARY in df.columns else STATUS_COL_FALLBACK
        status_norm = df[status_source].apply(normalize_status)
        summary["status_counts"] = status_norm.value_counts(dropna=False).to_dict()

    missing = {}
    for col in PREDICTOR_COLUMNS:
        if col not in df.columns:
            missing[col] = None
        else:
            missing[col] = int(df[col].isna().sum())
    summary["missing_counts_predictors"] = missing

    dynamic_stats = {}
    for col in [TIME_COL_COLUMNS, CATALYST_CUM_COL, TARGET_COLUMNS]:
        if col not in df.columns:
            continue
        lengths = []
        finite_fracs = []
        for v in df[col]:
            arr = parse_listlike(v)
            lengths.append(len(arr))
            finite_fracs.append(float(np.isfinite(arr).mean()) if len(arr) > 0 else np.nan)
        lengths = np.asarray(lengths, dtype=float)
        finite_fracs = np.asarray(finite_fracs, dtype=float)
        dynamic_stats[col] = {
            "min_len": float(np.nanmin(lengths)) if lengths.size else np.nan,
            "median_len": float(np.nanmedian(lengths)) if lengths.size else np.nan,
            "max_len": float(np.nanmax(lengths)) if lengths.size else np.nan,
            "mean_finite_fraction": float(np.nanmean(finite_fracs)) if finite_fracs.size else np.nan,
        }
    summary["dynamic_array_stats"] = dynamic_stats
    return summary


def apply_training_pair_exclusions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    summary: Dict[str, Any] = {
        "pair_id_column": PAIR_ID_COL,
        "excluded_pair_ids": sorted(EXCLUDED_TRAIN_PAIR_IDS),
        "applied": False,
        "excluded_row_count": 0,
        "excluded_pair_count": 0,
        "remaining_row_count": int(len(df)),
    }
    if not EXCLUDED_TRAIN_PAIR_IDS:
        summary["reason"] = "no_excluded_pair_ids_configured"
        return df, summary
    if PAIR_ID_COL not in df.columns:
        summary["reason"] = f"missing_pair_id_column:{PAIR_ID_COL}"
        return df, summary

    pair_ids = df[PAIR_ID_COL].astype(str).str.strip()
    exclude_mask = pair_ids.isin(EXCLUDED_TRAIN_PAIR_IDS)
    filtered_df = df.loc[~exclude_mask].copy()

    summary.update(
        {
            "applied": True,
            "excluded_row_count": int(exclude_mask.sum()),
            "excluded_pair_count": int(pair_ids.loc[exclude_mask].nunique(dropna=True)),
            "remaining_row_count": int(len(filtered_df)),
        }
    )
    return filtered_df, summary


def prefit_biexponential_for_rows(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for idx, row in df.iterrows():
        status = normalize_status(row.get(STATUS_COL_PRIMARY, row.get(STATUS_COL_FALLBACK, "")))
        t_raw = parse_listlike(row.get(TIME_COL_COLUMNS, np.nan))
        y_raw = parse_listlike(row.get(TARGET_COLUMNS, np.nan))
        t, y, _ = prepare_curve_arrays(t_raw, y_raw, None, status=status, min_points=6)
        params = fit_biexponential_params(t, y) if t.size >= 6 else np.array([np.nan] * 4, dtype=float)
        if np.all(np.isfinite(params)) and t.size > 0:
            pred = double_exp_curve_np(t, params[0], params[1], params[2], params[3])
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        else:
            rmse = np.nan
        out_rows.append(
            {
                "row_index": int(idx),
                "status_norm": status,
                "fit_a1": float(params[0]) if np.isfinite(params[0]) else np.nan,
                "fit_b1": float(params[1]) if np.isfinite(params[1]) else np.nan,
                "fit_a2": float(params[2]) if np.isfinite(params[2]) else np.nan,
                "fit_b2": float(params[3]) if np.isfinite(params[3]) else np.nan,
                "fit_rmse": rmse,
            }
        )
    return pd.DataFrame(out_rows)


def build_pair_samples(df: pd.DataFrame) -> List[PairSample]:
    pairs: List[PairSample] = []
    grouped = df.groupby(PAIR_ID_COL, dropna=False)
    for sample_id, group in grouped:
        if pd.isna(sample_id):
            continue
        by_status: Dict[str, Tuple[CurveData, np.ndarray]] = {}
        for idx, row in group.iterrows():
            status = normalize_status(row.get(STATUS_COL_PRIMARY, row.get(STATUS_COL_FALLBACK, "")))
            t_raw = parse_listlike(row.get(TIME_COL_COLUMNS, np.nan))
            y_raw = parse_listlike(row.get(TARGET_COLUMNS, np.nan))
            c_raw = parse_listlike(row.get(CATALYST_CUM_COL, np.nan))
            t, y, c = prepare_curve_arrays(t_raw, y_raw, c_raw, status=status, min_points=6)
            if t.size < 6:
                continue
            fit_params = np.asarray(
                [
                    pd.to_numeric(row.get("fit_a1", np.nan), errors="coerce"),
                    pd.to_numeric(row.get("fit_b1", np.nan), errors="coerce"),
                    pd.to_numeric(row.get("fit_a2", np.nan), errors="coerce"),
                    pd.to_numeric(row.get("fit_b2", np.nan), errors="coerce"),
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(fit_params)):
                fit_params = fit_biexponential_params(t, y)

            static_vec = np.asarray(
                [scalar_from_maybe_array(row.get(col, np.nan)) for col in STATIC_PREDICTOR_COLUMNS],
                dtype=float,
            )

            curve_data = CurveData(
                status=status,
                time=t,
                recovery=y,
                catalyst_cum=c,
                fit_params=sanitize_curve_params(fit_params),
                row_index=int(idx),
            )
            if status not in by_status or curve_data.time.size > by_status[status][0].time.size:
                by_status[status] = (curve_data, static_vec)

        if "Control" not in by_status or "Catalyzed" not in by_status:
            continue
        ctrl_curve, ctrl_static = by_status["Control"]
        cat_curve, cat_static = by_status["Catalyzed"]
        merged_static = combine_static_vectors(ctrl_static, cat_static)
        pairs.append(
            PairSample(
                sample_id=str(sample_id),
                static_raw=merged_static,
                control=ctrl_curve,
                catalyzed=cat_curve,
            )
        )
    pairs = sorted(pairs, key=lambda x: x.sample_id)
    return pairs


def fit_static_transformers(train_pairs: List[PairSample]) -> Tuple[SimpleImputer, StandardScaler]:
    X_train = np.vstack([p.static_raw for p in train_pairs]).astype(float)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X_train)
    scaler.fit(X_imp)
    return imputer, scaler


def build_static_model_input(static_raw: np.ndarray, static_scaled_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(static_raw, dtype=float)
    scaled = np.asarray(static_scaled_values, dtype=float)
    missing_mask = (~np.isfinite(raw)).astype(float)
    model_input = np.concatenate([scaled, missing_mask], axis=0)
    return np.asarray(model_input, dtype=float), np.asarray(missing_mask, dtype=float)


def apply_static_transformers(
    pairs: List[PairSample],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> None:
    X = np.vstack([p.static_raw for p in pairs]).astype(float)
    X_scaled = scaler.transform(imputer.transform(X))
    for p, xs in zip(pairs, X_scaled):
        model_input, missing_mask = build_static_model_input(p.static_raw, xs)
        p.static_scaled = model_input
        p.static_missing_mask = missing_mask


def build_repeated_kfold_member_splits(
    n_samples: int,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    member_seed_base: int,
) -> List[Dict[str, Any]]:
    if n_samples < 3:
        raise ValueError(f"Need at least 3 samples for repeated K-fold, got {n_samples}.")
    n_splits = int(max(2, min(int(n_splits), n_samples)))
    n_repeats = int(max(1, n_repeats))
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=int(random_state))

    splits: List[Dict[str, Any]] = []
    for member_idx, (train_idx, val_idx) in enumerate(rkf.split(np.arange(n_samples))):
        repeat_idx = member_idx // n_splits
        fold_idx = member_idx % n_splits
        member_seed = int(member_seed_base) + int(member_idx)
        splits.append(
            {
                "member_idx": int(member_idx),
                "repeat_idx": int(repeat_idx),
                "fold_idx": int(fold_idx),
                "member_seed": int(member_seed),
                "train_indices": np.asarray(train_idx, dtype=int),
                "val_indices": np.asarray(val_idx, dtype=int),
            }
        )
    return splits


# ---------------------------
# Model
# ---------------------------
class PairCurveNet(nn.Module):
    def __init__(
        self,
        n_static: int,
        hidden_dim: int,
        dropout: float,
        ctrl_lb: np.ndarray,
        ctrl_ub: np.ndarray,
        cat_lb: np.ndarray,
        cat_ub: np.ndarray,
        tmax_days: float,
        geo_idx: List[int],
        min_transition_days: float,
        max_catalyst_aging_strength: float,
        late_tau_impact_decay_strength: float,
        min_remaining_ore_factor: float,
    ) -> None:
        super().__init__()
        self.n_static = int(n_static)
        self.model_input_dim = 2 * self.n_static
        self.domain_hidden_dim = max(12, hidden_dim // 3)

        self.inventory_idx = [
            STATIC_PREDICTOR_COLUMNS.index(c)
            for c in MINERAL_INVENTORY_COLUMNS
            if c in STATIC_PREDICTOR_COLUMNS
        ]
        self.acid_redox_idx = [
            STATIC_PREDICTOR_COLUMNS.index(c)
            for c in ACID_REDOX_COLUMNS
            if c in STATIC_PREDICTOR_COLUMNS
        ]
        self.transport_idx = [
            STATIC_PREDICTOR_COLUMNS.index(c)
            for c in TRANSPORT_GANGUE_COLUMNS
            if c in STATIC_PREDICTOR_COLUMNS
        ]

        self.inventory_tower = self._make_domain_tower(self.inventory_idx, self.domain_hidden_dim)
        self.acid_redox_tower = self._make_domain_tower(self.acid_redox_idx, self.domain_hidden_dim)
        self.transport_tower = self._make_domain_tower(self.transport_idx, self.domain_hidden_dim)
        self.global_tower = nn.Sequential(
            nn.Linear(self.model_input_dim, self.domain_hidden_dim),
            nn.ReLU(),
        )

        fusion_dim = self.domain_hidden_dim
        if self.inventory_tower is not None:
            fusion_dim += self.domain_hidden_dim
        if self.acid_redox_tower is not None:
            fusion_dim += self.domain_hidden_dim
        if self.transport_tower is not None:
            fusion_dim += self.domain_hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.ctrl_head = nn.Linear(hidden_dim, 4)
        self.cat_delta_head = nn.Linear(hidden_dim, 4)
        self.delay_head = nn.Linear(hidden_dim, 1)
        self.temp_head = nn.Linear(hidden_dim, 1)
        self.kappa_head = nn.Linear(hidden_dim, 1)
        self.aging_head = nn.Linear(hidden_dim, 1)
        self.catalyst_receptivity_head = nn.Linear(hidden_dim, 1)

        self.ore_decay_head = nn.Linear(hidden_dim, 1)
        self.passivation_head = nn.Linear(hidden_dim, 1)
        self.passivation_tau_head = nn.Linear(hidden_dim, 1)
        self.passivation_temp_head = nn.Linear(hidden_dim, 1)
        self.depassivation_head = nn.Linear(hidden_dim, 1)
        self.transform_head = nn.Linear(hidden_dim, 1)
        self.transform_tau_head = nn.Linear(hidden_dim, 1)
        self.transform_temp_head = nn.Linear(hidden_dim, 1)
        self.chem_mix_head = nn.Linear(hidden_dim, 1)
        self.primary_drive_head = nn.Linear(hidden_dim, 1)
        self.fast_inventory_head = nn.Linear(hidden_dim, 1)
        self.oxide_inventory_head = nn.Linear(hidden_dim, 1)
        self.acid_buffer_head = nn.Linear(hidden_dim, 1)
        self.acid_buffer_decay_head = nn.Linear(hidden_dim, 1)
        self.diffusion_drag_head = nn.Linear(hidden_dim, 1)
        self.ferric_synergy_head = nn.Linear(hidden_dim, 1)
        self.primary_catalyst_synergy_head = nn.Linear(hidden_dim, 1)
        self.surface_refresh_head = nn.Linear(hidden_dim, 1)
        self.bed_permeability_head = nn.Linear(hidden_dim, 1)
        self.wetting_efficiency_head = nn.Linear(hidden_dim, 1)
        self.particle_accessibility_head = nn.Linear(hidden_dim, 1)
        self.pore_decay_tendency_head = nn.Linear(hidden_dim, 1)

        self.geo_idx = list(geo_idx)
        self.geo_delay_head = nn.Linear(len(self.geo_idx), 1) if len(self.geo_idx) > 0 else None

        self.chem_idx = [
            STATIC_PREDICTOR_COLUMNS.index(c)
            for c in CHEMISTRY_INTERACTION_COLUMNS
            if c in STATIC_PREDICTOR_COLUMNS
        ]
        self.chem_direct_head = nn.Linear(len(self.chem_idx), 1) if len(self.chem_idx) > 0 else None

        self.static_feature_indices = {
            col_name: idx for idx, col_name in enumerate(STATIC_PREDICTOR_COLUMNS)
        }
        self.interaction_weight_params = nn.ParameterDict()
        self.interaction_param_keys: Dict[str, Dict[str, str]] = {}
        for block_idx, (block_name, term_specs) in enumerate(LATENT_INTERACTION_SPECS.items()):
            block_keys: Dict[str, str] = {}
            for term_idx, (term_name, init_weight) in enumerate(term_specs.items()):
                param_key = f"b{block_idx:02d}_t{term_idx:02d}"
                self.interaction_weight_params[param_key] = nn.Parameter(
                    torch.tensor(float(init_weight), dtype=torch.float32)
                )
                block_keys[term_name] = param_key
            self.interaction_param_keys[block_name] = block_keys

        self.register_buffer("ctrl_lb", torch.tensor(ctrl_lb, dtype=torch.float32))
        self.register_buffer("ctrl_ub", torch.tensor(ctrl_ub, dtype=torch.float32))
        self.register_buffer("cat_lb", torch.tensor(cat_lb, dtype=torch.float32))
        self.register_buffer("cat_ub", torch.tensor(cat_ub, dtype=torch.float32))
        self.tmax_days = float(max(1.0, tmax_days))
        self.min_transition_days = float(max(1.0, min_transition_days))
        self.max_catalyst_aging_strength = float(max(0.0, max_catalyst_aging_strength))
        self.late_tau_impact_decay_strength = float(max(0.0, late_tau_impact_decay_strength))
        self.min_remaining_ore_factor = float(np.clip(min_remaining_ore_factor, 0.0, 1.0))

    @staticmethod
    def _make_domain_tower(indices: List[int], out_dim: int) -> Optional[nn.Module]:
        if len(indices) == 0:
            return None
        in_dim = 2 * len(indices)
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def _split_static_input(self, x_static: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        values = x_static[:, : self.n_static]
        missing_mask = x_static[:, self.n_static : self.n_static + self.n_static]
        return values, missing_mask

    def _domain_input(self, x_static: torch.Tensor, indices: List[int]) -> torch.Tensor:
        values, missing_mask = self._split_static_input(x_static)
        if len(indices) == 0:
            return torch.zeros((x_static.shape[0], 0), dtype=x_static.dtype, device=x_static.device)
        idx = torch.tensor(indices, dtype=torch.long, device=x_static.device)
        value_slice = torch.index_select(values, 1, idx)
        mask_slice = torch.index_select(missing_mask, 1, idx)
        return torch.cat([value_slice, mask_slice], dim=1)

    def _encode_static(self, x_static: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_values, _ = self._split_static_input(x_static)
        domain_parts = [self.global_tower(x_static)]
        for indices, tower in [
            (self.inventory_idx, self.inventory_tower),
            (self.acid_redox_idx, self.acid_redox_tower),
            (self.transport_idx, self.transport_tower),
        ]:
            if tower is None:
                continue
            domain_parts.append(tower(self._domain_input(x_static, indices)))
        encoded = self.encoder(torch.cat(domain_parts, dim=1))
        return encoded, x_values

    def _bounded_params(self, raw: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        return lb.unsqueeze(0) + torch.sigmoid(raw) * (ub - lb).unsqueeze(0)

    @staticmethod
    def _feature_column(x_static: torch.Tensor, idx: Optional[int]) -> torch.Tensor:
        if idx is None or idx < 0 or idx >= x_static.shape[1]:
            return torch.zeros((x_static.shape[0], 1), dtype=x_static.dtype, device=x_static.device)
        return x_static[:, idx : idx + 1]

    def _static_feature_map(self, x_static: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            col_name: self._feature_column(x_static, idx)
            for col_name, idx in self.static_feature_indices.items()
        }

    def _apply_learnable_interactions(
        self,
        base: torch.Tensor,
        block_name: str,
        term_tensors: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        out = base
        block_keys = self.interaction_param_keys.get(block_name, {})
        for term_name, param_key in block_keys.items():
            term_tensor = term_tensors.get(term_name)
            if term_tensor is None:
                continue
            out = out + self.interaction_weight_params[param_key] * term_tensor
        return out

    def _sanitize_params(self, p: torch.Tensor) -> torch.Tensor:
        swap_mask = (p[:, 3] > p[:, 1]).unsqueeze(1)
        p_swapped = torch.stack([p[:, 2], p[:, 3], p[:, 0], p[:, 1]], dim=1)
        p_ordered = torch.where(swap_mask, p_swapped, p)

        a1 = torch.clamp(p_ordered[:, 0], min=0.0, max=100.0)
        b1 = torch.clamp(p_ordered[:, 1], min=1e-5, max=2.0)
        a2 = torch.clamp(p_ordered[:, 2], min=0.0, max=100.0)
        b2 = torch.clamp(p_ordered[:, 3], min=1e-5, max=2.0)

        total = a1 + a2
        scale = torch.clamp(100.0 / torch.clamp(total, min=1e-6), max=1.0)
        a1 = a1 * scale
        a2 = a2 * scale
        return torch.stack([a1, b1, a2, b2], dim=1)

    @staticmethod
    def _double_exp_curve_from_grid_torch(params: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t_grid, min=0.0)
        if t.ndim == 1:
            t = t.view(1, -1)
        batch_size = int(params.shape[0])
        if t.shape[0] != batch_size:
            if t.shape[0] == 1:
                t = t.expand(batch_size, -1)
            else:
                t = t[:1].expand(batch_size, -1)
        a1 = params[:, 0:1]
        b1 = params[:, 1:2]
        a2 = params[:, 2:3]
        b2 = params[:, 3:4]
        return a1 * (1.0 - torch.exp(-b1 * t)) + a2 * (1.0 - torch.exp(-b2 * t))

    @staticmethod
    def _double_exp_curve_torch(params: torch.Tensor, t_days: torch.Tensor) -> torch.Tensor:
        return PairCurveNet._double_exp_curve_from_grid_torch(params, t_days)

    @staticmethod
    def _decayed_effective_cumulative_torch(
        cum_norm: torch.Tensor,
        t_days: torch.Tensor,
        aging_strength: torch.Tensor,
        tmax_days: float,
    ) -> torch.Tensor:
        c = torch.clamp(cum_norm.view(1, -1), min=0.0)
        t = torch.clamp(t_days.view(1, -1), min=0.0)
        batch_size = int(aging_strength.shape[0])
        if c.shape[0] != batch_size:
            c = c.expand(batch_size, -1)
            t = t.expand(batch_size, -1)

        if c.shape[1] == 0:
            return torch.zeros_like(c)
        if c.shape[1] == 1:
            return c.clone()

        strength = torch.clamp(aging_strength.view(-1), min=0.0)
        scale_days = max(float(tmax_days), 1e-6)
        eff_steps = [c[:, 0]]
        for i in range(1, c.shape[1]):
            dt = torch.clamp(t[:, i] - t[:, i - 1], min=0.0)
            decay = torch.exp(-strength * dt / scale_days)
            dc = torch.clamp(c[:, i] - c[:, i - 1], min=0.0)
            eff_steps.append(eff_steps[-1] * decay + dc)
        return torch.stack(eff_steps, dim=1)

    @staticmethod
    def _effective_time_from_rate_torch(t_days: torch.Tensor, rate_multiplier: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t_days, min=0.0)
        if t.ndim == 1:
            t = t.view(1, -1)
        r = torch.clamp(rate_multiplier, min=1e-4)
        if r.ndim == 1:
            r = r.view(1, -1)
        if t.shape[0] != r.shape[0]:
            if t.shape[0] == 1:
                t = t.expand(r.shape[0], -1)
            elif r.shape[0] == 1:
                r = r.expand(t.shape[0], -1)

        if t.shape[1] == 0:
            return torch.zeros_like(t)

        eff_steps = [t[:, 0] * r[:, 0]]
        for i in range(1, t.shape[1]):
            dt = torch.clamp(t[:, i] - t[:, i - 1], min=0.0)
            eff_steps.append(eff_steps[-1] + 0.5 * dt * (r[:, i] + r[:, i - 1]))
        return torch.stack(eff_steps, dim=1)

    def predict_params(
        self,
        x_static: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        h, x_values = self._encode_static(x_static)
        p_ctrl = self._bounded_params(self.ctrl_head(h), self.ctrl_lb, self.ctrl_ub)
        p_ctrl = self._sanitize_params(p_ctrl)

        tau_raw = self.delay_head(h)
        if self.geo_delay_head is not None:
            geo = x_values[:, self.geo_idx]
            tau_raw = tau_raw + self.geo_delay_head(geo)
        tau_days = self.tmax_days * torch.sigmoid(tau_raw)
        temp_days = self.min_transition_days + F.softplus(self.temp_head(h))
        kappa = 1e-3 + F.softplus(self.kappa_head(h))
        aging_strength = self.max_catalyst_aging_strength * torch.sigmoid(self.aging_head(h))
        interaction_terms = self._static_feature_map(x_values)

        chem_raw = self.chem_mix_head(h)
        if self.chem_direct_head is not None:
            interaction_terms["chem_direct"] = self.chem_direct_head(x_values[:, self.chem_idx])
        chem_raw = self._apply_learnable_interactions(
            chem_raw,
            "chem_raw",
            interaction_terms,
        )
        interaction_terms["chem_raw"] = chem_raw

        primary_drive_raw = self._apply_learnable_interactions(
            self.primary_drive_head(h),
            "primary_passivation_drive",
            interaction_terms,
        )
        primary_passivation_drive = torch.sigmoid(primary_drive_raw)
        interaction_terms["primary_passivation_drive"] = primary_passivation_drive

        ferric_synergy = 0.50 + torch.sigmoid(
            self._apply_learnable_interactions(
                self.ferric_synergy_head(h),
                "ferric_synergy",
                interaction_terms,
            )
        )
        interaction_terms["ferric_synergy"] = ferric_synergy

        chem_interaction = 0.50 + torch.sigmoid(
            self._apply_learnable_interactions(
                chem_raw,
                "chem_interaction",
                interaction_terms,
            )
        )
        interaction_terms["chem_interaction"] = chem_interaction

        primary_catalyst_synergy = 0.50 + torch.sigmoid(
            self._apply_learnable_interactions(
                self.primary_catalyst_synergy_head(h),
                "primary_catalyst_synergy",
                interaction_terms,
            )
        )
        interaction_terms["primary_catalyst_synergy"] = primary_catalyst_synergy

        catalyst_receptivity = 0.20 + 1.30 * torch.sigmoid(
            self.catalyst_receptivity_head(h)
            + 0.30 * (primary_catalyst_synergy - 1.0)
            + 0.18 * (ferric_synergy - 1.0)
            + 0.10 * (primary_passivation_drive - 0.5)
        )
        interaction_terms["catalyst_receptivity"] = catalyst_receptivity

        fast_leach_inventory = torch.sigmoid(
            self._apply_learnable_interactions(
                self.fast_inventory_head(h),
                "fast_leach_inventory",
                interaction_terms,
            )
        )
        interaction_terms["fast_leach_inventory"] = fast_leach_inventory
        oxide_inventory = torch.sigmoid(
            self._apply_learnable_interactions(
                self.oxide_inventory_head(h),
                "oxide_inventory",
                interaction_terms,
            )
        )
        interaction_terms["oxide_inventory"] = oxide_inventory
        acid_buffer_strength = 0.85 * torch.sigmoid(
            self._apply_learnable_interactions(
                self.acid_buffer_head(h),
                "acid_buffer_strength",
                interaction_terms,
            )
        )
        interaction_terms["acid_buffer_strength"] = acid_buffer_strength
        acid_buffer_decay = 0.20 + 1.40 * torch.sigmoid(
            self._apply_learnable_interactions(
                self.acid_buffer_decay_head(h),
                "acid_buffer_decay",
                interaction_terms,
            )
        )
        interaction_terms["acid_buffer_decay"] = acid_buffer_decay
        diffusion_drag_strength = 0.75 * torch.sigmoid(
            self._apply_learnable_interactions(
                self.diffusion_drag_head(h),
                "diffusion_drag_strength",
                interaction_terms,
            )
        )
        interaction_terms["diffusion_drag_strength"] = diffusion_drag_strength
        surface_refresh = 0.50 + torch.sigmoid(
            self._apply_learnable_interactions(
                self.surface_refresh_head(h),
                "surface_refresh",
                interaction_terms,
            )
        )
        interaction_terms["surface_refresh"] = surface_refresh
        bed_permeability = 0.20 + 1.30 * torch.sigmoid(
            self._apply_learnable_interactions(
                self.bed_permeability_head(h),
                "bed_permeability",
                interaction_terms,
            )
        )
        interaction_terms["bed_permeability"] = bed_permeability
        wetting_efficiency = 0.15 + 1.35 * torch.sigmoid(
            self._apply_learnable_interactions(
                self.wetting_efficiency_head(h),
                "wetting_efficiency",
                interaction_terms,
            )
        )
        interaction_terms["wetting_efficiency"] = wetting_efficiency
        particle_accessibility = 0.15 + 1.35 * torch.sigmoid(
            self._apply_learnable_interactions(
                self.particle_accessibility_head(h),
                "particle_accessibility",
                interaction_terms,
            )
        )
        interaction_terms["particle_accessibility"] = particle_accessibility
        pore_decay_tendency = 0.85 * torch.sigmoid(
            self._apply_learnable_interactions(
                self.pore_decay_tendency_head(h),
                "pore_decay_tendency",
                interaction_terms,
            )
        )
        interaction_terms["pore_decay_tendency"] = pore_decay_tendency

        ore_decay_strength = 0.15 + 1.85 * torch.sigmoid(
            self._apply_learnable_interactions(
                self.ore_decay_head(h),
                "ore_decay_strength",
                interaction_terms,
            )
        )
        interaction_terms["ore_decay_strength"] = ore_decay_strength
        passivation_strength = 0.85 * torch.sigmoid(
            self._apply_learnable_interactions(
                self.passivation_head(h),
                "passivation_strength",
                interaction_terms,
            )
        )
        interaction_terms["passivation_strength"] = passivation_strength
        passivation_tau = self.tmax_days * torch.sigmoid(self.passivation_tau_head(h))
        passivation_temp = self.min_transition_days + F.softplus(self.passivation_temp_head(h))
        depassivation_strength = 1.35 * torch.sigmoid(
            self._apply_learnable_interactions(
                self.depassivation_head(h),
                "depassivation_strength",
                interaction_terms,
            )
        )
        interaction_terms["depassivation_strength"] = depassivation_strength
        transform_strength = 1.00 * torch.sigmoid(
            self._apply_learnable_interactions(
                self.transform_head(h),
                "transform_strength",
                interaction_terms,
            )
        )
        interaction_terms["transform_strength"] = transform_strength
        transform_tau = self.tmax_days * torch.sigmoid(self.transform_tau_head(h))
        transform_temp = self.min_transition_days + F.softplus(self.transform_temp_head(h))

        delta_cap = torch.clamp(self.cat_ub.unsqueeze(0) - p_ctrl, min=1e-5)
        delta_scale = torch.clamp(catalyst_receptivity, min=0.20, max=1.50)
        cat_delta = torch.sigmoid(self.cat_delta_head(h)) * delta_cap * delta_scale
        p_cat = self._sanitize_params(p_ctrl + cat_delta)

        latent_params = {
            "catalyst_receptivity": catalyst_receptivity,
            "chem_interaction": chem_interaction,
            "primary_passivation_drive": primary_passivation_drive,
            "primary_catalyst_synergy": primary_catalyst_synergy,
            "fast_leach_inventory": fast_leach_inventory,
            "oxide_inventory": oxide_inventory,
            "acid_buffer_strength": acid_buffer_strength,
            "acid_buffer_decay": acid_buffer_decay,
            "diffusion_drag_strength": diffusion_drag_strength,
            "ferric_synergy": ferric_synergy,
            "surface_refresh": surface_refresh,
            "bed_permeability": bed_permeability,
            "wetting_efficiency": wetting_efficiency,
            "particle_accessibility": particle_accessibility,
            "pore_decay_tendency": pore_decay_tendency,
            "ore_decay_strength": ore_decay_strength,
            "passivation_strength": passivation_strength,
            "passivation_tau": passivation_tau,
            "passivation_temp": passivation_temp,
            "depassivation_strength": depassivation_strength,
            "transform_strength": transform_strength,
            "transform_tau": transform_tau,
            "transform_temp": transform_temp,
        }
        return p_ctrl, p_cat, tau_days, temp_days, kappa, aging_strength, latent_params

    def curves_given_params(
        self,
        p_ctrl: torch.Tensor,
        p_cat: torch.Tensor,
        t_days: torch.Tensor,
        cum_norm: torch.Tensor,
        tau_days: torch.Tensor,
        temp_days: torch.Tensor,
        kappa: torch.Tensor,
        aging_strength: torch.Tensor,
        latent_params: Optional[Dict[str, torch.Tensor]] = None,
        return_states: bool = False,
    ) -> Any:
        t = t_days.view(1, -1)
        if latent_params is None:
            batch_size = int(p_ctrl.shape[0])
            ones = torch.ones((batch_size, 1), dtype=p_ctrl.dtype, device=p_ctrl.device)
            zeros = torch.zeros((batch_size, 1), dtype=p_ctrl.dtype, device=p_ctrl.device)
            latent_params = {
                "catalyst_receptivity": ones,
                "chem_interaction": ones,
                "primary_passivation_drive": 0.5 * ones,
                "primary_catalyst_synergy": ones,
                "fast_leach_inventory": 0.5 * ones,
                "oxide_inventory": 0.5 * ones,
                "acid_buffer_strength": zeros,
                "acid_buffer_decay": 0.5 * ones,
                "diffusion_drag_strength": zeros,
                "ferric_synergy": ones,
                "surface_refresh": ones,
                "bed_permeability": ones,
                "wetting_efficiency": ones,
                "particle_accessibility": ones,
                "pore_decay_tendency": 0.25 * ones,
                "ore_decay_strength": zeros,
                "passivation_strength": zeros,
                "passivation_tau": zeros,
                "passivation_temp": self.min_transition_days * ones,
                "depassivation_strength": zeros,
                "transform_strength": zeros,
                "transform_tau": zeros,
                "transform_temp": self.min_transition_days * ones,
            }

        base_ctrl = self._double_exp_curve_torch(p_ctrl, t_days)
        delay_factor = torch.sigmoid((t - tau_days) / torch.clamp(temp_days, min=1e-3))
        effective_catalyst = self._decayed_effective_cumulative_torch(
            cum_norm=cum_norm,
            t_days=t_days,
            aging_strength=aging_strength,
            tmax_days=self.tmax_days,
        )
        catalyst_receptivity = torch.clamp(latent_params["catalyst_receptivity"], min=0.25, max=1.75)

        chem_interaction = torch.clamp(latent_params["chem_interaction"], min=0.25, max=1.75)
        primary_drive = torch.clamp(latent_params["primary_passivation_drive"], min=0.05, max=1.0)
        primary_catalyst_synergy = torch.clamp(latent_params["primary_catalyst_synergy"], min=0.35, max=1.75)
        fast_leach_inventory = torch.clamp(latent_params["fast_leach_inventory"], min=0.0, max=1.0)
        oxide_inventory = torch.clamp(latent_params["oxide_inventory"], min=0.0, max=1.0)
        acid_buffer_strength = torch.clamp(latent_params["acid_buffer_strength"], min=0.0, max=0.95)
        acid_buffer_decay = torch.clamp(latent_params["acid_buffer_decay"], min=1e-3)
        diffusion_drag_strength = torch.clamp(latent_params["diffusion_drag_strength"], min=0.0, max=0.95)
        ferric_synergy = torch.clamp(latent_params["ferric_synergy"], min=0.35, max=1.75)
        surface_refresh = torch.clamp(latent_params["surface_refresh"], min=0.35, max=1.75)
        bed_permeability = torch.clamp(latent_params["bed_permeability"], min=0.15, max=1.75)
        wetting_efficiency = torch.clamp(latent_params["wetting_efficiency"], min=0.10, max=1.75)
        particle_accessibility = torch.clamp(latent_params["particle_accessibility"], min=0.10, max=1.75)
        pore_decay_tendency = torch.clamp(latent_params["pore_decay_tendency"], min=0.0, max=0.95)
        ore_decay_strength = torch.clamp(latent_params["ore_decay_strength"], min=0.0)
        passivation_strength = torch.clamp(latent_params["passivation_strength"], min=0.0, max=1.0)
        passivation_tau = torch.clamp(latent_params["passivation_tau"], min=0.0, max=self.tmax_days)
        passivation_temp = torch.clamp(latent_params["passivation_temp"], min=1e-3)
        depassivation_strength = torch.clamp(latent_params["depassivation_strength"], min=0.0)
        transform_strength = torch.clamp(latent_params["transform_strength"], min=0.0)
        transform_tau = torch.clamp(latent_params["transform_tau"], min=0.0, max=self.tmax_days)
        transform_temp = torch.clamp(latent_params["transform_temp"], min=1e-3)

        ore_accessibility = torch.exp(-ore_decay_strength * torch.clamp(base_ctrl, 0.0, 100.0) / 100.0)
        ore_accessibility = ore_accessibility * torch.exp(
            -0.35 * ore_decay_strength * t / max(self.tmax_days, 1e-6)
        )
        ore_accessibility = torch.clamp(
            ore_accessibility,
            min=self.min_remaining_ore_factor,
            max=1.0,
        )
        fast_release_seed = torch.clamp(
            (0.65 * fast_leach_inventory + 0.35 * oxide_inventory)
            * torch.exp(-1.25 * t / max(self.tmax_days, 1e-6))
            * (0.50 + 0.50 * ore_accessibility),
            min=0.0,
            max=1.25,
        )

        acid_buffer_remaining = torch.exp(-acid_buffer_decay * t / max(self.tmax_days, 1e-6))
        acid_buffer_penalty = torch.clamp(
            acid_buffer_strength * acid_buffer_remaining,
            min=0.0,
            max=0.95,
        )

        diffusion_progress = torch.sqrt(torch.clamp(t / max(self.tmax_days, 1e-6), min=0.0, max=1.0))
        diffusion_drag = torch.clamp(
            diffusion_drag_strength * diffusion_progress * (0.60 + 0.40 * primary_drive),
            min=0.0,
            max=0.95,
        )

        passivation_progress = torch.sigmoid((t - passivation_tau) / passivation_temp)
        transformation_progress = torch.sigmoid((t - transform_tau) / transform_temp)
        passivation_seed = torch.clamp(
            passivation_strength
            * primary_drive
            * chem_interaction
            * (0.60 + 0.40 * ferric_synergy)
            * passivation_progress
            * (1.0 - 0.30 * fast_release_seed)
            * (1.0 + 0.20 * acid_buffer_penalty),
            min=0.0,
            max=1.20,
        )

        batch_size = int(t.shape[0])
        t_scale = max(self.tmax_days, 1e-6)
        bed_perm_scalar = bed_permeability.view(-1)
        wetting_scalar = wetting_efficiency.view(-1)
        particle_scalar = particle_accessibility.view(-1)
        pore_decay_scalar = pore_decay_tendency.view(-1)
        surface_scalar = surface_refresh.view(-1)

        front_prev = torch.zeros(batch_size, dtype=t.dtype, device=t.device)
        wetting_prev = torch.zeros(batch_size, dtype=t.dtype, device=t.device)
        particle_prev = torch.zeros(batch_size, dtype=t.dtype, device=t.device)
        closure_prev = torch.zeros(batch_size, dtype=t.dtype, device=t.device)
        front_steps: List[torch.Tensor] = []
        wetting_steps: List[torch.Tensor] = []
        particle_steps: List[torch.Tensor] = []
        closure_steps: List[torch.Tensor] = []
        hydraulic_steps: List[torch.Tensor] = []

        for i in range(t.shape[1]):
            if i == 0:
                dt_norm = torch.clamp(t[:, 0] / t_scale, min=0.0, max=1.0)
            else:
                dt_norm = torch.clamp((t[:, i] - t[:, i - 1]) / t_scale, min=0.0, max=1.0)

            delay_i = delay_factor[:, i]
            acid_i = acid_buffer_penalty[:, i]
            diff_i = diffusion_drag[:, i]
            ore_i = ore_accessibility[:, i]
            pass_seed_i = passivation_seed[:, i]

            hydraulic_i = torch.clamp(
                bed_perm_scalar
                * (0.35 + 0.65 * delay_i)
                * (1.0 - 0.45 * acid_i)
                * (1.0 - 0.35 * diff_i)
                * (1.0 - 0.25 * closure_prev)
                * (0.60 + 0.40 * ore_i),
                min=0.02,
                max=2.0,
            )
            front_prev = torch.clamp(front_prev + dt_norm * hydraulic_i * (1.0 - front_prev), min=0.0, max=1.0)

            wetting_push = torch.clamp(
                wetting_scalar
                * (0.30 + 0.70 * front_prev)
                * (0.40 + 0.60 * bed_perm_scalar)
                * (1.0 - 0.30 * diff_i)
                * (1.0 - 0.25 * acid_i)
                * (1.0 - 0.30 * closure_prev)
                * (0.65 + 0.35 * surface_scalar),
                min=0.0,
                max=2.0,
            )
            wetting_prev = torch.clamp(
                wetting_prev + dt_norm * wetting_push * (1.0 - wetting_prev),
                min=0.0,
                max=1.0,
            )

            particle_push = torch.clamp(
                particle_scalar
                * (0.25 + 0.75 * wetting_prev)
                * (0.40 + 0.60 * front_prev)
                * (1.0 - 0.45 * diff_i)
                * (1.0 - 0.25 * pass_seed_i)
                * (1.0 - 0.20 * closure_prev)
                * (0.55 + 0.45 * ore_i),
                min=0.0,
                max=2.0,
            )
            particle_prev = torch.clamp(
                particle_prev + dt_norm * particle_push * (1.0 - particle_prev),
                min=0.0,
                max=1.0,
            )

            closure_growth = torch.clamp(
                pore_decay_scalar
                * (0.35 + 0.65 * front_prev)
                * (
                    0.35 * pass_seed_i
                    + 0.25 * acid_i
                    + 0.20 * diff_i
                    + 0.20 * (1.0 - ore_i)
                ),
                min=0.0,
                max=1.5,
            )
            closure_relief = torch.clamp(
                0.20
                * surface_scalar
                * wetting_prev
                * (0.40 + 0.60 * bed_perm_scalar),
                min=0.0,
                max=1.0,
            )
            closure_prev = torch.clamp(
                closure_prev + dt_norm * (closure_growth * (1.0 - closure_prev) - closure_relief * closure_prev),
                min=0.0,
                max=0.98,
            )

            front_steps.append(front_prev)
            wetting_steps.append(wetting_prev)
            particle_steps.append(particle_prev)
            closure_steps.append(closure_prev)
            hydraulic_steps.append(hydraulic_i)

        solution_front = torch.stack(front_steps, dim=1)
        wetting_fraction = torch.stack(wetting_steps, dim=1)
        particle_access = torch.stack(particle_steps, dim=1)
        pore_closure = torch.stack(closure_steps, dim=1)
        hydraulic_flow = torch.stack(hydraulic_steps, dim=1)
        pore_open_fraction = torch.clamp(1.0 - pore_closure, min=0.02, max=1.0)
        packed_bed_contact = torch.clamp(
            solution_front * wetting_fraction * particle_access * pore_open_fraction,
            min=0.0,
            max=1.0,
        )

        fast_release = torch.clamp(
            fast_release_seed
            * (0.35 + 0.65 * packed_bed_contact)
            * (0.50 + 0.50 * particle_access),
            min=0.0,
            max=1.25,
        )
        passivation = torch.clamp(
            passivation_seed
            * ore_accessibility
            * (0.35 + 0.65 * particle_access)
            * (0.45 + 0.55 * solution_front)
            * (1.0 + 0.20 * (1.0 - pore_open_fraction)),
            min=0.0,
            max=0.95,
        )

        catalyst_surface_access = torch.clamp(
            catalyst_receptivity
            * solution_front
            * wetting_fraction
            * particle_access
            * pore_open_fraction
            * ore_accessibility
            * (0.55 + 0.45 * surface_refresh),
            min=0.0,
            max=1.75,
        )
        catalyst_factor = 1.0 - torch.exp(
            -torch.clamp(kappa, min=1e-6) * effective_catalyst * catalyst_surface_access
        )

        transformation = torch.clamp(
            transform_strength
            * chem_interaction
            * transformation_progress
            * (0.35 + 0.65 * ferric_synergy)
            * (0.35 + 0.65 * catalyst_factor)
            * (0.45 + 0.55 * catalyst_receptivity)
            * (0.35 + 0.65 * packed_bed_contact)
            * (0.45 + 0.55 * ore_accessibility),
            min=0.0,
            max=1.25,
        )

        depassivation = torch.clamp(
            depassivation_strength
            * primary_drive
            * chem_interaction
            * ferric_synergy
            * surface_refresh
            * delay_factor
            * catalyst_factor
            * (0.45 + 0.55 * catalyst_receptivity)
            * (0.35 + 0.65 * particle_access)
            * (0.35 + 0.65 * solution_front)
            * (0.35 + 0.65 * transformation),
            min=0.0,
            max=1.50,
        )

        ctrl_rate_multiplier = torch.clamp(
            ore_accessibility
            * (0.35 + 0.65 * packed_bed_contact)
            * (0.50 + 0.50 * hydraulic_flow)
            * (1.0 - passivation)
            * (1.0 - 0.65 * acid_buffer_penalty)
            * (1.0 - diffusion_drag)
            * (0.80 + 0.30 * fast_release + 0.12 * wetting_fraction + 0.08 * transformation),
            min=0.08,
            max=1.75,
        )
        cat_rate_multiplier = torch.clamp(
            ore_accessibility
            * (0.40 + 0.60 * packed_bed_contact)
            * (0.55 + 0.45 * hydraulic_flow)
            * (1.0 - 0.35 * passivation)
            * (1.0 - 0.40 * acid_buffer_penalty)
            * (1.0 - 0.60 * diffusion_drag)
            * (
                0.90
                + 0.30 * fast_release
                + 0.10 * wetting_fraction
                + depassivation
                + 0.25 * transformation
                + 0.18 * primary_catalyst_synergy
                + 0.15 * catalyst_receptivity
            ),
            min=0.08,
            max=2.75,
        )

        t_eff_ctrl = self._effective_time_from_rate_torch(t_days, ctrl_rate_multiplier)
        t_eff_cat = self._effective_time_from_rate_torch(t_days, cat_rate_multiplier)

        y_ctrl = self._double_exp_curve_from_grid_torch(p_ctrl, t_eff_ctrl)
        y_cat_cap = self._double_exp_curve_from_grid_torch(p_cat, t_eff_cat)
        delta_limit = F.softplus(y_cat_cap - y_ctrl)

        remaining_ore_factor = torch.clamp(
            (100.0 - torch.clamp(y_ctrl, 0.0, 100.0)) / 100.0,
            min=self.min_remaining_ore_factor,
            max=1.0,
        )
        late_tau_factor = torch.exp(
            -self.late_tau_impact_decay_strength * torch.clamp(tau_days, min=0.0) / max(self.tmax_days, 1e-6)
        )
        latent_gap_factor = torch.clamp(
            (1.0 - 0.55 * passivation)
            * (1.0 - 0.35 * acid_buffer_penalty)
            * (1.0 - 0.35 * diffusion_drag)
            + depassivation
            + 0.35 * transformation
            + 0.20 * fast_release
            + 0.20 * primary_catalyst_synergy,
            min=0.0,
            max=2.75,
        )
        separation_factor = torch.clamp(
            delay_factor * catalyst_factor * remaining_ore_factor * late_tau_factor * latent_gap_factor,
            min=0.0,
            max=1.0,
        )

        y_cat = y_ctrl + separation_factor * delta_limit
        y_ctrl = torch.cummax(torch.clamp(y_ctrl, 0.0, 100.0), dim=1).values
        y_cat = torch.cummax(torch.clamp(y_cat, 0.0, 100.0), dim=1).values
        y_cat = torch.maximum(y_cat, y_ctrl)

        if not return_states:
            return y_ctrl.squeeze(0), y_cat.squeeze(0)

        states = {
            "ore_accessibility": ore_accessibility.squeeze(0),
            "catalyst_receptivity": catalyst_receptivity.squeeze(0),
            "catalyst_surface_access": catalyst_surface_access.squeeze(0),
            "bed_permeability": bed_permeability.squeeze(0),
            "wetting_efficiency": wetting_efficiency.squeeze(0),
            "particle_accessibility": particle_accessibility.squeeze(0),
            "pore_decay_tendency": pore_decay_tendency.squeeze(0),
            "solution_front": solution_front.squeeze(0),
            "wetting_fraction": wetting_fraction.squeeze(0),
            "particle_access": particle_access.squeeze(0),
            "pore_open_fraction": pore_open_fraction.squeeze(0),
            "packed_bed_contact": packed_bed_contact.squeeze(0),
            "hydraulic_flow": hydraulic_flow.squeeze(0),
            "fast_release": fast_release.squeeze(0),
            "acid_buffer_penalty": acid_buffer_penalty.squeeze(0),
            "diffusion_drag": diffusion_drag.squeeze(0),
            "passivation": passivation.squeeze(0),
            "depassivation": depassivation.squeeze(0),
            "transformation": transformation.squeeze(0),
            "primary_catalyst_synergy": primary_catalyst_synergy.squeeze(0),
            "ctrl_rate_multiplier": ctrl_rate_multiplier.squeeze(0),
            "cat_rate_multiplier": cat_rate_multiplier.squeeze(0),
            "effective_catalyst": effective_catalyst.squeeze(0),
            "catalyst_factor": catalyst_factor.squeeze(0),
            "ferric_synergy": ferric_synergy.squeeze(0),
            "surface_refresh": surface_refresh.squeeze(0),
        }
        return y_ctrl.squeeze(0), y_cat.squeeze(0), states


# ---------------------------
# Inference and Extrapolation Warnings
# ---------------------------
def sanitize_filename(value: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    s = s.strip("._")
    return s or "sample"


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"min": np.nan, "max": np.nan, "p05": np.nan, "p95": np.nan, "mean": np.nan, "std": np.nan}
    return {
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "p05": float(np.nanquantile(v, 0.05)),
        "p95": float(np.nanquantile(v, 0.95)),
        "mean": float(np.nanmean(v)),
        "std": float(np.nanstd(v) + 1e-8),
    }


def dose_profile_descriptors(t_days: np.ndarray, cum_profile: np.ndarray) -> Dict[str, float]:
    t = np.asarray(t_days, dtype=float)
    c = clean_cumulative_profile(np.asarray(cum_profile, dtype=float), force_zero=False)
    if t.size == 0 or c.size == 0:
        return {
            "total_dose": np.nan,
            "final_day": np.nan,
            "first_addition_day": np.nan,
            "day_to_50pct_dose": np.nan,
            "day_to_90pct_dose": np.nan,
            "max_addition_rate": np.nan,
            "active_addition_days": np.nan,
        }
    total = float(np.nanmax(c))
    final_day = float(np.nanmax(t))
    positive_idx = np.where(c > 1e-10)[0]
    if positive_idx.size > 0:
        first_day = float(t[positive_idx[0]])
        active_days = float(max(0.0, final_day - first_day))
    else:
        first_day = np.nan
        active_days = 0.0

    day50 = np.nan
    day90 = np.nan
    if total > 0:
        idx50 = np.where(c >= 0.5 * total)[0]
        idx90 = np.where(c >= 0.9 * total)[0]
        if idx50.size > 0:
            day50 = float(t[idx50[0]])
        if idx90.size > 0:
            day90 = float(t[idx90[0]])

    if len(t) >= 2:
        dt = np.diff(t)
        dc = np.diff(c)
        rate = dc / np.clip(dt, 1e-6, None)
        max_rate = float(np.nanmax(rate)) if np.isfinite(rate).any() else np.nan
    else:
        max_rate = np.nan

    return {
        "total_dose": total,
        "final_day": final_day,
        "first_addition_day": first_day,
        "day_to_50pct_dose": day50,
        "day_to_90pct_dose": day90,
        "max_addition_rate": max_rate,
        "active_addition_days": active_days,
    }


def build_extrapolation_reference(
    train_pairs: List[PairSample],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> Dict[str, Any]:
    x_raw = np.vstack([p.static_raw for p in train_pairs]).astype(float)
    x_imp = imputer.transform(x_raw)
    x_scaled = scaler.transform(x_imp)

    static_stats = {
        col: _summary_stats(x_imp[:, i]) for i, col in enumerate(STATIC_PREDICTOR_COLUMNS)
    }

    if x_scaled.shape[0] >= 2:
        mean_scaled = np.nanmean(x_scaled, axis=0)
        cov = np.cov(x_scaled, rowvar=False)
        if cov.ndim == 0:
            cov = np.asarray([[float(cov)]], dtype=float)
        reg = 1e-6 * np.eye(cov.shape[0], dtype=float)
        inv_cov = np.linalg.pinv(cov + reg)

        mahal = []
        for i in range(x_scaled.shape[0]):
            d = x_scaled[i] - mean_scaled
            mahal.append(float(np.sqrt(max(0.0, d @ inv_cov @ d))))
        mahal = np.asarray(mahal, dtype=float)
        mahal_p95 = float(np.nanquantile(mahal, 0.95)) if mahal.size > 0 else np.nan

        dmat = np.linalg.norm(x_scaled[:, None, :] - x_scaled[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        nn_train = np.min(dmat, axis=1)
        nn_train = nn_train[np.isfinite(nn_train)]
        nn_p95 = float(np.nanquantile(nn_train, 0.95)) if nn_train.size > 0 else np.nan
    else:
        mean_scaled = np.zeros(x_scaled.shape[1], dtype=float)
        inv_cov = np.eye(x_scaled.shape[1], dtype=float)
        mahal_p95 = np.nan
        nn_p95 = np.nan

    dose_rows = [dose_profile_descriptors(p.catalyzed.time, p.catalyzed.catalyst_cum) for p in train_pairs]
    dose_df = pd.DataFrame(dose_rows)
    dose_stats = {c: _summary_stats(dose_df[c].to_numpy(dtype=float)) for c in dose_df.columns}

    return {
        "n_train_pairs": int(len(train_pairs)),
        "static_feature_stats": static_stats,
        "dose_descriptor_stats": dose_stats,
        "x_train_scaled": x_scaled.tolist(),
        "mean_scaled": mean_scaled.tolist(),
        "inv_cov_scaled": inv_cov.tolist(),
        "mahalanobis_p95": mahal_p95,
        "nearest_neighbor_distance_p95": nn_p95,
    }


def create_input_example_excel(
    output_path: str,
    extrapolation_ref: Dict[str, Any],
    example_pair: PairSample,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    static_stats = extrapolation_ref["static_feature_stats"]
    t = np.asarray(example_pair.catalyzed.time, dtype=float)
    c = np.asarray(example_pair.catalyzed.catalyst_cum, dtype=float)
    n_rows = max(len(STATIC_PREDICTOR_COLUMNS), len(t))

    rows = []
    for i in range(n_rows):
        if i < len(STATIC_PREDICTOR_COLUMNS):
            col = STATIC_PREDICTOR_COLUMNS[i]
            stats = static_stats.get(col, {})
            pred_name = col
            pred_val = float(example_pair.static_raw[i]) if np.isfinite(example_pair.static_raw[i]) else np.nan
            tr_min = float(stats.get("min", np.nan))
            tr_max = float(stats.get("max", np.nan))
            warning = "If value is outside [train_min, train_max], extrapolation warning triggers."
        else:
            pred_name = ""
            pred_val = np.nan
            tr_min = np.nan
            tr_max = np.nan
            warning = ""

        t_val = float(t[i]) if i < len(t) else np.nan
        c_val = float(c[i]) if i < len(c) else np.nan
        rows.append(
            {
                "predictor_name": pred_name,
                "value": pred_val,
                "train_min": tr_min,
                "train_max": tr_max,
                "warning_rule": warning,
                "leach_duration_days": t_val,
                "cumulative_catalyst_addition_kg_t": c_val,
            }
        )

    input_df = pd.DataFrame(rows)
    instr_df = pd.DataFrame(
        {
            "instructions": [
                "Fill column 'value' for one-value predictors (left side).",
                "Fill 'leach_duration_days' and 'cumulative_catalyst_addition_kg_t' as multiple rows (right side).",
                "Keep leach days increasing and catalyst cumulative non-decreasing.",
                "Predictor names must not be edited.",
                "Values outside [train_min, train_max] will trigger extrapolation warnings.",
            ]
        }
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        input_df.to_excel(writer, index=False, sheet_name="Input_Template")
        instr_df.to_excel(writer, index=False, sheet_name="Instructions")


def load_model_input_excel(path: str) -> Dict[str, Any]:
    df = pd.read_excel(path, sheet_name="Input_Template")
    required = [
        "predictor_name",
        "value",
        "leach_duration_days",
        "cumulative_catalyst_addition_kg_t",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {path}: {missing_cols}")

    static_map: Dict[str, float] = {}
    for _, row in df.iterrows():
        name = str(row.get("predictor_name", "")).strip()
        if name in STATIC_PREDICTOR_COLUMNS and name not in static_map:
            val = pd.to_numeric(pd.Series([row.get("value", np.nan)]), errors="coerce").iloc[0]
            static_map[name] = float(val) if pd.notna(val) else np.nan
    static_raw = np.asarray([static_map.get(c, np.nan) for c in STATIC_PREDICTOR_COLUMNS], dtype=float)

    t = pd.to_numeric(df["leach_duration_days"], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(df["cumulative_catalyst_addition_kg_t"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(c)
    t = t[valid]
    c = c[valid]
    if len(t) < 4:
        raise ValueError(f"{path} needs at least 4 valid rows for leach_duration_days/cumulative_catalyst_addition_kg_t")

    order = np.argsort(t)
    t = t[order]
    c = c[order]
    t_unique, inv = np.unique(t, return_inverse=True)
    c_unique = np.full_like(t_unique, np.nan, dtype=float)
    for i, j in enumerate(inv):
        if np.isfinite(c[i]):
            c_unique[j] = c[i] if not np.isfinite(c_unique[j]) else max(c_unique[j], c[i])
    c_clean = clean_cumulative_profile(c_unique, force_zero=False)

    return {
        "sample_name": os.path.splitext(os.path.basename(path))[0],
        "source_path": path,
        "static_raw": static_raw,
        "time_days": t_unique.astype(float),
        "catalyst_cum": c_clean.astype(float),
    }


def predict_new_sample_member(
    model: PairCurveNet,
    static_scaled: np.ndarray,
    time_days: np.ndarray,
    catalyst_cum: np.ndarray,
    cum_scale: float,
    control_time_days: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    model.eval()
    same_plot_grid = False
    if control_time_days is not None:
        same_plot_grid = (
            np.asarray(control_time_days, dtype=float).shape == np.asarray(time_days, dtype=float).shape
            and np.allclose(np.asarray(control_time_days, dtype=float), np.asarray(time_days, dtype=float))
        )
    with torch.no_grad():
        x = torch.tensor(static_scaled, dtype=torch.float32, device=device).unsqueeze(0)
        t = torch.tensor(np.asarray(time_days, dtype=float), dtype=torch.float32, device=device)
        c = torch.tensor(np.asarray(catalyst_cum, dtype=float) / cum_scale, dtype=torch.float32, device=device)
        p_ctrl, p_cat, tau, temp, kappa, aging_strength, latent = model.predict_params(x)
        pred_ctrl, pred_cat = model.curves_given_params(
            p_ctrl,
            p_cat,
            t,
            c,
            tau,
            temp,
            kappa,
            aging_strength,
            latent_params=latent,
        )

        pred_ctrl_plot = None
        if control_time_days is not None:
            t_ctrl_plot = torch.tensor(np.asarray(control_time_days, dtype=float), dtype=torch.float32, device=device)
            c_ctrl_plot = torch.zeros_like(t_ctrl_plot)
            pred_ctrl_plot, _ = model.curves_given_params(
                p_ctrl,
                p_cat,
                t_ctrl_plot,
                c_ctrl_plot,
                tau,
                temp,
                kappa,
                aging_strength,
                latent_params=latent,
            )
            if same_plot_grid:
                pred_ctrl = pred_ctrl_plot
                pred_cat = torch.maximum(pred_cat, pred_ctrl_plot)

    out = {
        "control_pred": pred_ctrl.detach().cpu().numpy(),
        "catalyzed_pred": pred_cat.detach().cpu().numpy(),
        "tau_days": float(tau.squeeze().detach().cpu().item()),
        "temp_days": float(temp.squeeze().detach().cpu().item()),
        "kappa": float(kappa.squeeze().detach().cpu().item()),
        "aging_strength": float(aging_strength.squeeze().detach().cpu().item()),
        "catalyst_receptivity": float(latent["catalyst_receptivity"].squeeze().detach().cpu().item()),
        "chem_interaction": float(latent["chem_interaction"].squeeze().detach().cpu().item()),
        "primary_passivation_drive": float(latent["primary_passivation_drive"].squeeze().detach().cpu().item()),
        "primary_catalyst_synergy": float(latent["primary_catalyst_synergy"].squeeze().detach().cpu().item()),
        "fast_leach_inventory": float(latent["fast_leach_inventory"].squeeze().detach().cpu().item()),
        "oxide_inventory": float(latent["oxide_inventory"].squeeze().detach().cpu().item()),
        "acid_buffer_strength": float(latent["acid_buffer_strength"].squeeze().detach().cpu().item()),
        "acid_buffer_decay": float(latent["acid_buffer_decay"].squeeze().detach().cpu().item()),
        "diffusion_drag_strength": float(latent["diffusion_drag_strength"].squeeze().detach().cpu().item()),
        "ferric_synergy": float(latent["ferric_synergy"].squeeze().detach().cpu().item()),
        "surface_refresh": float(latent["surface_refresh"].squeeze().detach().cpu().item()),
        "bed_permeability": float(latent["bed_permeability"].squeeze().detach().cpu().item()),
        "wetting_efficiency": float(latent["wetting_efficiency"].squeeze().detach().cpu().item()),
        "particle_accessibility": float(latent["particle_accessibility"].squeeze().detach().cpu().item()),
        "pore_decay_tendency": float(latent["pore_decay_tendency"].squeeze().detach().cpu().item()),
        "ore_decay_strength": float(latent["ore_decay_strength"].squeeze().detach().cpu().item()),
        "passivation_strength": float(latent["passivation_strength"].squeeze().detach().cpu().item()),
        "depassivation_strength": float(latent["depassivation_strength"].squeeze().detach().cpu().item()),
        "transform_strength": float(latent["transform_strength"].squeeze().detach().cpu().item()),
    }
    if pred_ctrl_plot is not None:
        out["control_pred_plot"] = pred_ctrl_plot.detach().cpu().numpy()
    return out


def predict_new_sample_ensemble(
    member_models: List[Dict[str, Any]],
    static_raw: np.ndarray,
    time_days: np.ndarray,
    catalyst_cum: np.ndarray,
    cum_scale: float,
    pi_low: float,
    pi_high: float,
    control_time_days: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    if len(member_models) == 0:
        raise ValueError("predict_new_sample_ensemble received no member models.")

    member_preds = [
        predict_new_sample_member(
            model=m["model"],
            static_scaled=build_static_model_input(
                static_raw,
                m["scaler"].transform(m["imputer"].transform([static_raw]))[0],
            )[0],
            time_days=time_days,
            catalyst_cum=catalyst_cum,
            cum_scale=cum_scale,
            control_time_days=control_time_days,
        )
        for m in member_models
    ]
    ctrl_stack = np.vstack([p["control_pred"] for p in member_preds])
    cat_stack = np.vstack([p["catalyzed_pred"] for p in member_preds])

    record = {
        "time_days": np.asarray(time_days, dtype=float),
        "cumulative_catalyst_addition_kg_t": np.asarray(catalyst_cum, dtype=float),
        "control_pred_mean": np.mean(ctrl_stack, axis=0),
        "control_pred_p10": np.percentile(ctrl_stack, pi_low, axis=0),
        "control_pred_p90": np.percentile(ctrl_stack, pi_high, axis=0),
        "catalyzed_pred_mean": np.mean(cat_stack, axis=0),
        "catalyzed_pred_p10": np.percentile(cat_stack, pi_low, axis=0),
        "catalyzed_pred_p90": np.percentile(cat_stack, pi_high, axis=0),
        "tau_days_mean": float(np.mean([p["tau_days"] for p in member_preds])),
        "temp_days_mean": float(np.mean([p["temp_days"] for p in member_preds])),
        "kappa_mean": float(np.mean([p["kappa"] for p in member_preds])),
        "aging_strength_mean": float(np.mean([p["aging_strength"] for p in member_preds])),
        "n_members": int(len(member_models)),
    }
    if control_time_days is not None:
        ctrl_plot_stack = np.vstack([p["control_pred_plot"] for p in member_preds])
        record["control_plot_time_days"] = np.asarray(control_time_days, dtype=float)
        record["control_pred_plot_mean"] = np.mean(ctrl_plot_stack, axis=0)
        record["control_pred_plot_p10"] = np.percentile(ctrl_plot_stack, pi_low, axis=0)
        record["control_pred_plot_p90"] = np.percentile(ctrl_plot_stack, pi_high, axis=0)
        record["catalyzed_plot_time_days"] = np.asarray(time_days, dtype=float)
    ctrl_width = np.asarray(record["control_pred_p90"]) - np.asarray(record["control_pred_p10"])
    cat_width = np.asarray(record["catalyzed_pred_p90"]) - np.asarray(record["catalyzed_pred_p10"])
    uncertainty = {
        "control_width_median": float(np.nanmedian(ctrl_width)),
        "control_width_max": float(np.nanmax(ctrl_width)),
        "catalyzed_width_median": float(np.nanmedian(cat_width)),
        "catalyzed_width_max": float(np.nanmax(cat_width)),
        "catalyzed_relative_width_median": float(
            np.nanmedian(cat_width / np.clip(np.asarray(record["catalyzed_pred_mean"]), 1.0, None))
        ),
    }
    return record, uncertainty


def evaluate_extrapolation_risk(
    static_raw: np.ndarray,
    time_days: np.ndarray,
    catalyst_cum: np.ndarray,
    extrapolation_ref: Dict[str, Any],
    imputer: SimpleImputer,
    scaler: StandardScaler,
    uncertainty: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    static_raw = np.asarray(static_raw, dtype=float)
    static_stats = extrapolation_ref.get("static_feature_stats", {})
    hard_flags: List[str] = []
    soft_flags: List[str] = []
    missing_flags: List[str] = []

    for i, col in enumerate(STATIC_PREDICTOR_COLUMNS):
        v = float(static_raw[i]) if i < len(static_raw) else np.nan
        st = static_stats.get(col, {})
        lo, hi = float(st.get("min", np.nan)), float(st.get("max", np.nan))
        p05, p95 = float(st.get("p05", np.nan)), float(st.get("p95", np.nan))
        if not np.isfinite(v):
            missing_flags.append(col)
            continue
        if np.isfinite(lo) and np.isfinite(hi) and (v < lo or v > hi):
            hard_flags.append(f"{col}: {v:.4g} outside train [{lo:.4g}, {hi:.4g}]")
        elif np.isfinite(p05) and np.isfinite(p95) and (v < p05 or v > p95):
            soft_flags.append(f"{col}: {v:.4g} outside train P05-P95 [{p05:.4g}, {p95:.4g}]")

    x_imp = imputer.transform([static_raw])[0]
    x_scaled = scaler.transform([x_imp])[0]

    mean_scaled = np.asarray(extrapolation_ref.get("mean_scaled", []), dtype=float)
    inv_cov = np.asarray(extrapolation_ref.get("inv_cov_scaled", []), dtype=float)
    if mean_scaled.size == x_scaled.size and inv_cov.shape == (x_scaled.size, x_scaled.size):
        diff = x_scaled - mean_scaled
        mahal = float(np.sqrt(max(0.0, diff @ inv_cov @ diff)))
    else:
        mahal = np.nan
    mahal_p95 = float(extrapolation_ref.get("mahalanobis_p95", np.nan))
    if np.isfinite(mahal) and np.isfinite(mahal_p95):
        if mahal > 1.30 * mahal_p95:
            hard_flags.append(f"Mahalanobis distance {mahal:.3f} > 1.30 x train P95 ({mahal_p95:.3f})")
        elif mahal > mahal_p95:
            soft_flags.append(f"Mahalanobis distance {mahal:.3f} > train P95 ({mahal_p95:.3f})")

    x_train_scaled = np.asarray(extrapolation_ref.get("x_train_scaled", []), dtype=float)
    if x_train_scaled.ndim == 2 and x_train_scaled.shape[1] == x_scaled.size and x_train_scaled.shape[0] > 0:
        nn_dist = float(np.min(np.linalg.norm(x_train_scaled - x_scaled[None, :], axis=1)))
    else:
        nn_dist = np.nan
    nn_p95 = float(extrapolation_ref.get("nearest_neighbor_distance_p95", np.nan))
    if np.isfinite(nn_dist) and np.isfinite(nn_p95):
        if nn_dist > 1.30 * nn_p95:
            hard_flags.append(f"Nearest-neighbor distance {nn_dist:.3f} > 1.30 x train P95 ({nn_p95:.3f})")
        elif nn_dist > nn_p95:
            soft_flags.append(f"Nearest-neighbor distance {nn_dist:.3f} > train P95 ({nn_p95:.3f})")

    dose_desc = dose_profile_descriptors(time_days, catalyst_cum)
    dose_stats = extrapolation_ref.get("dose_descriptor_stats", {})
    for k, val in dose_desc.items():
        if not np.isfinite(val):
            continue
        st = dose_stats.get(k, {})
        lo, hi = float(st.get("min", np.nan)), float(st.get("max", np.nan))
        p05, p95 = float(st.get("p05", np.nan)), float(st.get("p95", np.nan))
        if np.isfinite(lo) and np.isfinite(hi) and (val < lo or val > hi):
            hard_flags.append(f"{k}: {val:.4g} outside train [{lo:.4g}, {hi:.4g}]")
        elif np.isfinite(p05) and np.isfinite(p95) and (val < p05 or val > p95):
            soft_flags.append(f"{k}: {val:.4g} outside train P05-P95 [{p05:.4g}, {p95:.4g}]")

    if uncertainty is not None:
        cat_w_max = float(uncertainty.get("catalyzed_width_max", np.nan))
        cat_w_med = float(uncertainty.get("catalyzed_width_median", np.nan))
        cat_w_rel = float(uncertainty.get("catalyzed_relative_width_median", np.nan))
        if np.isfinite(cat_w_max) and cat_w_max > 12.0:
            hard_flags.append(f"High ensemble uncertainty: catalyzed P90-P10 max width={cat_w_max:.2f}")
        elif np.isfinite(cat_w_max) and cat_w_max > 8.0:
            soft_flags.append(f"Moderate ensemble uncertainty: catalyzed P90-P10 max width={cat_w_max:.2f}")
        if np.isfinite(cat_w_med) and cat_w_med > 5.0:
            soft_flags.append(f"Ensemble uncertainty median width={cat_w_med:.2f}")
        if np.isfinite(cat_w_rel) and cat_w_rel > 0.25:
            soft_flags.append(f"Relative uncertainty median={cat_w_rel:.2%}")

    score = 2 * len(hard_flags) + len(soft_flags) + 2 * len(missing_flags)
    if len(hard_flags) >= 3 or score >= 10:
        level = "RED"
        message = "Extrapolating too hard: prediction confidence is low."
    elif len(hard_flags) >= 1 or score >= 4:
        level = "YELLOW"
        message = "Moderate extrapolation risk: use prediction with caution."
    else:
        level = "GREEN"
        message = "Input is within training domain for most checks."

    return {
        "risk_level": level,
        "risk_score": int(score),
        "message": message,
        "missing_predictors": missing_flags,
        "hard_flags": hard_flags,
        "soft_flags": soft_flags,
        "distance_checks": {
            "mahalanobis_distance": mahal,
            "mahalanobis_train_p95": mahal_p95,
            "nearest_neighbor_distance": nn_dist,
            "nearest_neighbor_distance_train_p95": nn_p95,
        },
        "dose_descriptors": dose_desc,
        "uncertainty_summary": dict(uncertainty or {}),
    }


def _annotate_ensemble_extension(ax: Any, record: Dict[str, Any]) -> None:
    catalyst_start_day = float(record.get("catalyst_addition_start_day", np.nan))
    catalyst_stop_day = float(record.get("catalyst_addition_stop_day", np.nan))
    weekly_value = float(record.get("weekly_catalyst_addition_kg_t", np.nan))
    reference_days = float(record.get("weekly_reference_days", np.nan))
    stopped_before_test_end = bool(record.get("stopped_before_test_end", False))
    extension_applied = bool(record.get("extension_applied", False))

    if np.isfinite(catalyst_start_day):
        ax.axvline(catalyst_start_day, color="#666666", lw=1.0, ls="--", alpha=0.7)
    if stopped_before_test_end and np.isfinite(catalyst_stop_day):
        ax.axvline(catalyst_stop_day, color="#444444", lw=1.0, ls=":", alpha=0.85)

    if not extension_applied:
        return
    if not np.isfinite(weekly_value):
        return

    ref_text = f"avg over last {reference_days:.0f} days" if np.isfinite(reference_days) and reference_days > 0 else ""
    text = f"Weekly catalyst used: {weekly_value*1000:.2f} g/t/week"
    if ref_text:
        text = f"{text}\n{ref_text}"

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.88},
    )


def plot_inference_ensemble_curve(record: Dict[str, Any], risk: Dict[str, Any], plot_path: str, title: str) -> None:
    ct = np.asarray(record.get("control_plot_time_days", record["time_days"]), dtype=float)
    cmean = np.asarray(record.get("control_pred_plot_mean", record["control_pred_mean"]), dtype=float)
    clow = np.asarray(record.get("control_pred_plot_p10", record["control_pred_p10"]), dtype=float)
    chigh = np.asarray(record.get("control_pred_plot_p90", record["control_pred_p90"]), dtype=float)
    kt = np.asarray(record.get("catalyzed_plot_time_days", record["time_days"]), dtype=float)
    kmean = np.asarray(record["catalyzed_pred_mean"], dtype=float)
    klow = np.asarray(record["catalyzed_pred_p10"], dtype=float)
    khigh = np.asarray(record["catalyzed_pred_p90"], dtype=float)

    plt.figure(figsize=(9, 5))
    plt.fill_between(kt, klow, khigh, color="#ff7f0e", alpha=0.20, label="Catalyzed P10-P90", zorder=1)
    plt.plot(kt, kmean, color="#ff7f0e", lw=2, label="Catalyzed Pred Mean", zorder=2)
    plt.fill_between(ct, clow, chigh, color="#1f77b4", alpha=0.20, label="Control P10-P90", zorder=3)
    plt.plot(ct, cmean, color="#1f77b4", lw=2, label="Control Pred Mean", zorder=4)
    plt.xlabel("leach_duration_days")
    plt.ylabel("cu_recovery_%")
    plt.ylim(0, 100)
    plt.title(f"{title} | Risk={risk.get('risk_level', 'NA')}")
    plt.grid(alpha=0.25)
    plt.legend()
    _annotate_ensemble_extension(plt.gca(), record)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=int(CONFIG["plot_dpi"]))
    plt.close()


# ---------------------------
# Training and evaluation
# ---------------------------
def _monotonic_penalty(y: torch.Tensor) -> torch.Tensor:
    if y.numel() < 2:
        return torch.tensor(0.0, device=y.device)
    return torch.relu(y[:-1] - y[1:]).mean()


def _curve_slopes(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if y.numel() < 2 or t.numel() < 2:
        return torch.empty(0, dtype=y.dtype, device=y.device)
    dt = torch.clamp(t[1:] - t[:-1], min=1e-3)
    return (y[1:] - y[:-1]) / dt


def _curvature_penalty(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    slope = _curve_slopes(y, t)
    if slope.numel() < 2:
        return torch.tensor(0.0, device=y.device)
    return torch.abs(slope[1:] - slope[:-1]).mean()


def _slope_cap_penalty(y: torch.Tensor, t: torch.Tensor, max_slope_per_day: float) -> torch.Tensor:
    slope = _curve_slopes(y, t)
    if slope.numel() < 1:
        return torch.tensor(0.0, device=y.device)
    return torch.relu(torch.abs(slope) - float(max_slope_per_day)).mean()


def _interp_curve_np(query_t: np.ndarray, anchor_t: np.ndarray, anchor_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q = np.asarray(query_t, dtype=float)
    t = np.asarray(anchor_t, dtype=float)
    y = np.asarray(anchor_y, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() < 2:
        return np.full_like(q, np.nan, dtype=float), np.zeros_like(q, dtype=bool)
    t = t[valid]
    y = y[valid]
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    overlap = np.isfinite(q) & (q >= t[0]) & (q <= t[-1])
    out = np.full_like(q, np.nan, dtype=float)
    if np.any(overlap):
        out[overlap] = np.interp(q[overlap], t, y)
    return out, overlap


def _feature_monotonic_penalty(
    model: PairCurveNet,
    pair: PairSample,
    pred_ctrl: torch.Tensor,
    pred_cat: torch.Tensor,
    ctrl_t: torch.Tensor,
    ctrl_c: torch.Tensor,
    cat_t: torch.Tensor,
    cat_c: torch.Tensor,
) -> torch.Tensor:
    if len(MONOTONIC_STATIC_FEATURE_DIRECTIONS) == 0:
        return torch.tensor(0.0, device=device)

    x_base = torch.tensor(pair.static_scaled, dtype=torch.float32, device=device).unsqueeze(0)
    missing_mask = (
        np.asarray(pair.static_missing_mask, dtype=float)
        if pair.static_missing_mask is not None
        else np.zeros(len(STATIC_PREDICTOR_COLUMNS), dtype=float)
    )
    step = float(CONFIG.get("monotonic_step_scaled", 0.25))
    penalties = []

    for col_name, direction in MONOTONIC_STATIC_FEATURE_DIRECTIONS.items():
        if col_name not in STATIC_PREDICTOR_COLUMNS:
            continue
        idx = STATIC_PREDICTOR_COLUMNS.index(col_name)
        if idx >= len(missing_mask) or missing_mask[idx] >= 0.5:
            continue

        x_pert = x_base.clone()
        x_pert[:, idx] = x_pert[:, idx] + step
        p_ctrl_p, p_cat_p, tau_p, temp_p, kappa_p, aging_p, latent_p = model.predict_params(x_pert)
        pred_ctrl_p, _ = model.curves_given_params(
            p_ctrl_p,
            p_cat_p,
            ctrl_t,
            ctrl_c,
            tau_p,
            temp_p,
            kappa_p,
            aging_p,
            latent_params=latent_p,
        )
        _, pred_cat_p = model.curves_given_params(
            p_ctrl_p,
            p_cat_p,
            cat_t,
            cat_c,
            tau_p,
            temp_p,
            kappa_p,
            aging_p,
            latent_params=latent_p,
        )

        ctrl_delta = pred_ctrl_p[-1] - pred_ctrl[-1]
        cat_delta = pred_cat_p[-1] - pred_cat[-1]
        if direction > 0:
            penalties.append(torch.relu(-ctrl_delta) + torch.relu(-cat_delta))
        else:
            penalties.append(torch.relu(ctrl_delta) + torch.relu(cat_delta))

    if len(penalties) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(penalties).mean()


def pair_training_loss(
    model: PairCurveNet,
    pair: PairSample,
    cum_scale: float,
    loss_weights: Dict[str, float],
    max_cat_slope_per_day: float,
) -> torch.Tensor:
    x = torch.tensor(pair.static_scaled, dtype=torch.float32, device=device).unsqueeze(0)
    ctrl_t = torch.tensor(pair.control.time, dtype=torch.float32, device=device)
    ctrl_y = torch.tensor(pair.control.recovery, dtype=torch.float32, device=device)
    ctrl_c = torch.tensor(pair.control.catalyst_cum / cum_scale, dtype=torch.float32, device=device)

    cat_t = torch.tensor(pair.catalyzed.time, dtype=torch.float32, device=device)
    cat_y = torch.tensor(pair.catalyzed.recovery, dtype=torch.float32, device=device)
    cat_c = torch.tensor(pair.catalyzed.catalyst_cum / cum_scale, dtype=torch.float32, device=device)

    p_ctrl, p_cat, tau, temp, kappa, aging_strength, latent = model.predict_params(x)

    pred_ctrl, _, ctrl_states = model.curves_given_params(
        p_ctrl,
        p_cat,
        ctrl_t,
        ctrl_c,
        tau,
        temp,
        kappa,
        aging_strength,
        latent_params=latent,
        return_states=True,
    )
    cat_ctrl_c = torch.zeros_like(cat_c)
    pred_ctrl_cat_time, _ = model.curves_given_params(
        p_ctrl,
        p_cat,
        cat_t,
        cat_ctrl_c,
        tau,
        temp,
        kappa,
        aging_strength,
        latent_params=latent,
    )
    _, pred_cat, cat_states = model.curves_given_params(
        p_ctrl,
        p_cat,
        cat_t,
        cat_c,
        tau,
        temp,
        kappa,
        aging_strength,
        latent_params=latent,
        return_states=True,
    )

    loss_ctrl = F.smooth_l1_loss(pred_ctrl, ctrl_y)
    loss_cat = F.smooth_l1_loss(pred_cat, cat_y)
    gap_pen = torch.relu(pred_ctrl_cat_time - pred_cat).mean()
    mono_pen = _monotonic_penalty(pred_ctrl) + _monotonic_penalty(pred_cat)
    cat_smooth_pen = _curvature_penalty(pred_cat, cat_t)
    cat_slope_cap_pen = _slope_cap_penalty(pred_cat, cat_t, max_cat_slope_per_day)
    latent_smooth_pen = (
        _curvature_penalty(ctrl_states["passivation"], ctrl_t)
        + _curvature_penalty(cat_states["passivation"], cat_t)
        + _curvature_penalty(cat_states["depassivation"], cat_t)
        + _curvature_penalty(cat_states["transformation"], cat_t)
        + _curvature_penalty(cat_states["acid_buffer_penalty"], cat_t)
        + _curvature_penalty(cat_states["diffusion_drag"], cat_t)
        + _curvature_penalty(cat_states["solution_front"], cat_t)
        + _curvature_penalty(cat_states["wetting_fraction"], cat_t)
        + _curvature_penalty(cat_states["particle_access"], cat_t)
    )
    state_mono_pen = (
        _monotonic_penalty(ctrl_states["solution_front"])
        + _monotonic_penalty(cat_states["solution_front"])
        + _monotonic_penalty(ctrl_states["wetting_fraction"])
        + _monotonic_penalty(cat_states["wetting_fraction"])
        + _monotonic_penalty(ctrl_states["particle_access"])
        + _monotonic_penalty(cat_states["particle_access"])
    )
    latent_cat_rate_pen = torch.relu(cat_states["ctrl_rate_multiplier"] - cat_states["cat_rate_multiplier"]).mean()

    ctrl_true_on_cat_np, overlap_mask_np = _interp_curve_np(
        pair.catalyzed.time,
        pair.control.time,
        pair.control.recovery,
    )
    overlap_mask = torch.tensor(overlap_mask_np, dtype=torch.bool, device=device)
    if bool(torch.any(overlap_mask).item()):
        ctrl_true_on_cat = torch.tensor(ctrl_true_on_cat_np, dtype=torch.float32, device=device)
        uplift_true = torch.clamp(cat_y - ctrl_true_on_cat, min=0.0)
        uplift_pred = torch.clamp(pred_cat - pred_ctrl_cat_time, min=0.0)
        uplift_pen = F.smooth_l1_loss(uplift_pred[overlap_mask], uplift_true[overlap_mask])
        late_threshold = 0.60 * torch.max(cat_t)
        late_mask = overlap_mask & (cat_t >= late_threshold)
        if bool(torch.any(late_mask).item()):
            late_uplift_pen = F.smooth_l1_loss(uplift_pred[late_mask], uplift_true[late_mask])
        else:
            late_uplift_pen = torch.tensor(0.0, device=device)
    else:
        uplift_pen = torch.tensor(0.0, device=device)
        late_uplift_pen = torch.tensor(0.0, device=device)

    feature_mono_pen = _feature_monotonic_penalty(
        model=model,
        pair=pair,
        pred_ctrl=pred_ctrl,
        pred_cat=pred_cat,
        ctrl_t=ctrl_t,
        ctrl_c=ctrl_c,
        cat_t=cat_t,
        cat_c=cat_c,
    )

    target_ctrl_p = torch.tensor(pair.control.fit_params, dtype=torch.float32, device=device)
    target_cat_p = torch.tensor(pair.catalyzed.fit_params, dtype=torch.float32, device=device)
    param_pen = F.smooth_l1_loss(p_ctrl.squeeze(0), target_ctrl_p) + F.smooth_l1_loss(
        p_cat.squeeze(0), target_cat_p
    )

    total = (
        loss_ctrl
        + loss_cat
        + float(loss_weights.get("gap", 1.0)) * gap_pen
        + float(loss_weights.get("uplift", 0.0)) * uplift_pen
        + float(loss_weights.get("late_uplift", 0.0)) * late_uplift_pen
        + float(loss_weights.get("monotonic", 0.02)) * mono_pen
        + float(loss_weights.get("feature_monotonic", 0.0)) * feature_mono_pen
        + float(loss_weights.get("state_monotonic", 0.0)) * state_mono_pen
        + float(loss_weights.get("param", 0.08)) * param_pen
        + float(loss_weights.get("smooth_cat", 0.12)) * cat_smooth_pen
        + float(loss_weights.get("slope_cap", 0.18)) * cat_slope_cap_pen
        + float(loss_weights.get("latent_smooth", 0.02)) * latent_smooth_pen
        + float(loss_weights.get("latent_cat_rate", 0.03)) * latent_cat_rate_pen
    )
    return total


def train_one_member(
    seed: int,
    train_pairs: List[PairSample],
    val_pairs: List[PairSample],
    ctrl_lb: np.ndarray,
    ctrl_ub: np.ndarray,
    cat_lb: np.ndarray,
    cat_ub: np.ndarray,
    tmax_days: float,
    cum_scale: float,
    epochs: int,
    patience: int,
    geo_idx: List[int],
) -> Tuple[PairCurveNet, List[Dict[str, float]], float]:
    set_all_seeds(seed, deterministic=True)
    rng = np.random.default_rng(seed)
    max_cat_slope_per_day = float(CONFIG.get("max_cat_slope_per_day", 0.20))

    boot_indices = rng.choice(len(train_pairs), size=len(train_pairs), replace=True)
    boot_pairs = [train_pairs[i] for i in boot_indices]

    model = PairCurveNet(
        n_static=len(STATIC_PREDICTOR_COLUMNS),
        hidden_dim=int(CONFIG["hidden_dim"]),
        dropout=float(CONFIG["dropout"]),
        ctrl_lb=ctrl_lb,
        ctrl_ub=ctrl_ub,
        cat_lb=cat_lb,
        cat_ub=cat_ub,
        tmax_days=tmax_days,
        geo_idx=geo_idx,
        min_transition_days=float(CONFIG.get("min_transition_days", 10.0)),
        max_catalyst_aging_strength=float(CONFIG.get("max_catalyst_aging_strength", 5.0)),
        late_tau_impact_decay_strength=float(CONFIG.get("late_tau_impact_decay_strength", 1.15)),
        min_remaining_ore_factor=float(CONFIG.get("min_remaining_ore_factor", 0.08)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(CONFIG["learning_rate"]),
        weight_decay=float(CONFIG["weight_decay"]),
    )

    history: List[Dict[str, float]] = []
    best_state = None
    best_eval = np.inf
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(boot_pairs)
        train_losses = []
        for pair in boot_pairs:
            optimizer.zero_grad(set_to_none=True)
            loss = pair_training_loss(
                model=model,
                pair=pair,
                cum_scale=cum_scale,
                loss_weights=CONFIG["loss_weights"],
                max_cat_slope_per_day=max_cat_slope_per_day,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(CONFIG["grad_clip_norm"]))
            optimizer.step()
            train_losses.append(float(loss.item()))
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan

        model.eval()
        eval_pool = val_pairs if len(val_pairs) > 0 else train_pairs
        eval_losses = []
        with torch.no_grad():
            for pair in eval_pool:
                loss = pair_training_loss(
                    model=model,
                    pair=pair,
                    cum_scale=cum_scale,
                    loss_weights=CONFIG["loss_weights"],
                    max_cat_slope_per_day=max_cat_slope_per_day,
                )
                eval_losses.append(float(loss.item()))
        eval_loss = float(np.mean(eval_losses)) if eval_losses else np.nan
        history.append({"epoch": float(epoch), "train_loss": train_loss, "eval_loss": eval_loss})

        if eval_loss + 1e-8 < best_eval:
            best_eval = eval_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience and epoch >= 80:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_eval


def format_member_tag(member_idx: int, repeat_idx: int, fold_idx: int, member_seed: int) -> str:
    return f"m{member_idx:03d}_r{repeat_idx + 1:02d}_f{fold_idx + 1:02d}_s{member_seed}"


def build_member_model_from_checkpoint(checkpoint: Dict[str, Any]) -> PairCurveNet:
    cfg = checkpoint.get("config", CONFIG)
    model = PairCurveNet(
        n_static=len(STATIC_PREDICTOR_COLUMNS),
        hidden_dim=int(cfg.get("hidden_dim", CONFIG["hidden_dim"])),
        dropout=float(cfg.get("dropout", CONFIG["dropout"])),
        ctrl_lb=np.asarray(checkpoint["ctrl_lb"], dtype=float),
        ctrl_ub=np.asarray(checkpoint["ctrl_ub"], dtype=float),
        cat_lb=np.asarray(checkpoint["cat_lb"], dtype=float),
        cat_ub=np.asarray(checkpoint["cat_ub"], dtype=float),
        tmax_days=float(checkpoint["tmax_days"]),
        geo_idx=[int(v) for v in checkpoint.get("geo_idx", [])],
        min_transition_days=float(cfg.get("min_transition_days", CONFIG.get("min_transition_days", 10.0))),
        max_catalyst_aging_strength=float(
            cfg.get("max_catalyst_aging_strength", CONFIG.get("max_catalyst_aging_strength", 5.0))
        ),
        late_tau_impact_decay_strength=float(
            cfg.get("late_tau_impact_decay_strength", CONFIG.get("late_tau_impact_decay_strength", 1.15))
        ),
        min_remaining_ore_factor=float(
            cfg.get("min_remaining_ore_factor", CONFIG.get("min_remaining_ore_factor", 0.08))
        ),
    ).to(device)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as exc:
        load_result = model.load_state_dict(checkpoint["state_dict"], strict=False)
        missing_ok = all(k.startswith("interaction_weight_params.") for k in load_result.missing_keys)
        if load_result.unexpected_keys or not missing_ok:
            raise exc
    return model


def train_validation_member_job(job: Dict[str, Any]) -> Dict[str, Any]:
    configure_torch_cpu_parallelism(
        num_threads=int(job.get("torch_threads_per_worker", 0)),
        num_interop_threads=int(job.get("torch_interop_threads_per_worker", 1)),
    )

    member_idx = int(job["member_idx"])
    repeat_idx = int(job["repeat_idx"])
    fold_idx = int(job["fold_idx"])
    member_seed = int(job["member_seed"])
    member_tag = str(job["member_tag"])
    train_pairs_seed: List[PairSample] = job["train_pairs"]
    val_pairs_seed: List[PairSample] = job["val_pairs"]
    if len(val_pairs_seed) == 0:
        raise ValueError(f"CV member {member_idx} produced empty validation split.")

    print(
        f"[CV Ensemble] Training {member_tag} "
        f"(train={len(train_pairs_seed)}, val={len(val_pairs_seed)})"
    )

    imputer_member, scaler_member = fit_static_transformers(train_pairs_seed)
    apply_static_transformers(train_pairs_seed, imputer_member, scaler_member)
    apply_static_transformers(val_pairs_seed, imputer_member, scaler_member)

    ctrl_seed_params = np.vstack([p.control.fit_params for p in train_pairs_seed])
    cat_seed_params = np.vstack([p.catalyzed.fit_params for p in train_pairs_seed])
    ctrl_lb_seed, ctrl_ub_seed = derive_param_bounds(ctrl_seed_params, None)
    cat_lb_seed, cat_ub_seed = derive_param_bounds(cat_seed_params, None)

    model, history, best_eval = train_one_member(
        seed=member_seed,
        train_pairs=train_pairs_seed,
        val_pairs=val_pairs_seed,
        ctrl_lb=ctrl_lb_seed,
        ctrl_ub=ctrl_ub_seed,
        cat_lb=cat_lb_seed,
        cat_ub=cat_ub_seed,
        tmax_days=float(job["tmax_days"]),
        cum_scale=float(job["cum_scale"]),
        epochs=int(job["epochs"]),
        patience=int(job["patience"]),
        geo_idx=[int(v) for v in job["geo_idx"]],
    )

    model_ckpt_path = os.path.join(str(job["val_member_model_root"]), f"{member_tag}.pt")
    torch.save(
        {
            "member_tag": member_tag,
            "member_idx": member_idx,
            "repeat_idx": repeat_idx,
            "fold_idx": fold_idx,
            "seed": member_seed,
            "state_dict": model.state_dict(),
            "static_predictor_columns": STATIC_PREDICTOR_COLUMNS,
            "geo_idx": [int(v) for v in job["geo_idx"]],
            "ctrl_lb": ctrl_lb_seed.tolist(),
            "ctrl_ub": ctrl_ub_seed.tolist(),
            "cat_lb": cat_lb_seed.tolist(),
            "cat_ub": cat_ub_seed.tolist(),
            "tmax_days": float(job["tmax_days"]),
            "cum_scale": float(job["cum_scale"]),
            "imputer_statistics": imputer_member.statistics_.tolist(),
            "scaler_mean": scaler_member.mean_.tolist(),
            "scaler_scale": scaler_member.scale_.tolist(),
            "config": CONFIG,
        },
        model_ckpt_path,
    )

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(str(job["val_member_out_root"]), f"{member_tag}_history.csv"), index=False)
    if len(history) > 0:
        best_epoch = int(pd.DataFrame(history).sort_values("eval_loss").iloc[0]["epoch"])
    else:
        best_epoch = -1

    metrics_val, records_val = evaluate_model(model, val_pairs_seed, float(job["cum_scale"]))
    records_to_df(records_val, ensemble=False).to_csv(
        os.path.join(str(job["val_member_out_root"]), f"{member_tag}_validation_predictions.csv"),
        index=False,
    )
    save_json(
        os.path.join(str(job["val_member_out_root"]), f"{member_tag}_validation_metrics.json"),
        {**metrics_val, "best_eval_loss": float(best_eval)},
    )

    scatter_path = os.path.join(str(job["val_member_plot_root"]), f"{member_tag}_overall_scatter.png")
    plot_overall_scatter(
        records=records_val,
        plot_path=scatter_path,
        title=f"Validation Scatter - {member_tag}",
        ensemble=False,
    )
    member_plot_dir = os.path.join(str(job["val_member_plot_root"]), member_tag)
    os.makedirs(member_plot_dir, exist_ok=True)
    for r in records_val:
        plot_single_record(
            record=r,
            plot_path=os.path.join(member_plot_dir, f"{r['sample_id']}.png"),
            title=f"Validation Prediction ({r['sample_id']}) - {member_tag}",
        )

    result = {
        "member_tag": member_tag,
        "member_idx": member_idx,
        "repeat_idx": repeat_idx,
        "fold_idx": fold_idx,
        "seed": member_seed,
        "metrics_val": metrics_val,
        "best_eval_loss": float(best_eval),
        "best_epoch": int(best_epoch),
        "n_train_pairs": int(len(train_pairs_seed)),
        "n_validation_pairs": int(len(val_pairs_seed)),
        "history": history,
        "train_sample_ids": [p.sample_id for p in train_pairs_seed],
        "validation_sample_ids": [p.sample_id for p in val_pairs_seed],
        "records_val": records_val,
        "model_ckpt_path": model_ckpt_path,
        "imputer": imputer_member,
        "scaler": scaler_member,
    }
    return result


def predict_pair_record(model: PairCurveNet, pair: PairSample, cum_scale: float) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(pair.static_scaled, dtype=torch.float32, device=device).unsqueeze(0)
        ctrl_t = torch.tensor(pair.control.time, dtype=torch.float32, device=device)
        ctrl_c = torch.tensor(pair.control.catalyst_cum / cum_scale, dtype=torch.float32, device=device)
        cat_t = torch.tensor(pair.catalyzed.time, dtype=torch.float32, device=device)
        cat_c = torch.tensor(pair.catalyzed.catalyst_cum / cum_scale, dtype=torch.float32, device=device)

        p_ctrl, p_cat, tau, temp, kappa, aging_strength, latent = model.predict_params(x)
        pred_ctrl, _, ctrl_states = model.curves_given_params(
            p_ctrl,
            p_cat,
            ctrl_t,
            ctrl_c,
            tau,
            temp,
            kappa,
            aging_strength,
            latent_params=latent,
            return_states=True,
        )
        cat_ctrl_c = torch.zeros_like(cat_c)
        pred_ctrl_cat_time, _ = model.curves_given_params(
            p_ctrl,
            p_cat,
            cat_t,
            cat_ctrl_c,
            tau,
            temp,
            kappa,
            aging_strength,
            latent_params=latent,
        )
        _, pred_cat, cat_states = model.curves_given_params(
            p_ctrl,
            p_cat,
            cat_t,
            cat_c,
            tau,
            temp,
            kappa,
            aging_strength,
            latent_params=latent,
            return_states=True,
        )

        plot_profile = build_shared_ensemble_plot_profile(
            time_days=pair.catalyzed.time,
            catalyst_cum=pair.catalyzed.catalyst_cum,
            target_day=float(CONFIG.get("ensemble_plot_target_day", 2500.0)),
            step_days=float(CONFIG.get("ensemble_plot_step_days", 1.0)),
            history_window_days=float(CONFIG.get("catalyst_extension_window_days", 21.0)),
        )
        control_plot_time_days = np.asarray(plot_profile["plot_time_days"], dtype=float)
        ctrl_plot_t = torch.tensor(control_plot_time_days, dtype=torch.float32, device=device)
        ctrl_plot_c = torch.zeros_like(ctrl_plot_t)
        pred_ctrl_plot, _ = model.curves_given_params(
            p_ctrl,
            p_cat,
            ctrl_plot_t,
            ctrl_plot_c,
            tau,
            temp,
            kappa,
            aging_strength,
            latent_params=latent,
        )

        cat_plot_t = torch.tensor(control_plot_time_days, dtype=torch.float32, device=device)
        cat_plot_c = torch.tensor(
            np.asarray(plot_profile["plot_catalyst_cum"], dtype=float) / cum_scale,
            dtype=torch.float32,
            device=device,
        )
        _, pred_cat_plot = model.curves_given_params(
            p_ctrl,
            p_cat,
            cat_plot_t,
            cat_plot_c,
            tau,
            temp,
            kappa,
            aging_strength,
            latent_params=latent,
        )
        pred_cat_plot = torch.maximum(pred_cat_plot, pred_ctrl_plot)

    rec = {
        "sample_id": pair.sample_id,
        "control_t": pair.control.time.copy(),
        "control_true": pair.control.recovery.copy(),
        "control_pred": pred_ctrl.detach().cpu().numpy(),
        "catalyzed_t": pair.catalyzed.time.copy(),
        "catalyzed_true": pair.catalyzed.recovery.copy(),
        "catalyzed_pred": pred_cat.detach().cpu().numpy(),
        "control_pred_on_catalyzed_t": pred_ctrl_cat_time.detach().cpu().numpy(),
        "tau_days": float(tau.squeeze().detach().cpu().item()),
        "temp_days": float(temp.squeeze().detach().cpu().item()),
        "kappa": float(kappa.squeeze().detach().cpu().item()),
        "aging_strength": float(aging_strength.squeeze().detach().cpu().item()),
        "catalyst_receptivity": float(latent["catalyst_receptivity"].squeeze().detach().cpu().item()),
        "chem_interaction": float(latent["chem_interaction"].squeeze().detach().cpu().item()),
        "primary_passivation_drive": float(latent["primary_passivation_drive"].squeeze().detach().cpu().item()),
        "primary_catalyst_synergy": float(latent["primary_catalyst_synergy"].squeeze().detach().cpu().item()),
        "fast_leach_inventory": float(latent["fast_leach_inventory"].squeeze().detach().cpu().item()),
        "oxide_inventory": float(latent["oxide_inventory"].squeeze().detach().cpu().item()),
        "acid_buffer_strength": float(latent["acid_buffer_strength"].squeeze().detach().cpu().item()),
        "acid_buffer_decay": float(latent["acid_buffer_decay"].squeeze().detach().cpu().item()),
        "diffusion_drag_strength": float(latent["diffusion_drag_strength"].squeeze().detach().cpu().item()),
        "ferric_synergy": float(latent["ferric_synergy"].squeeze().detach().cpu().item()),
        "surface_refresh": float(latent["surface_refresh"].squeeze().detach().cpu().item()),
        "bed_permeability": float(latent["bed_permeability"].squeeze().detach().cpu().item()),
        "wetting_efficiency": float(latent["wetting_efficiency"].squeeze().detach().cpu().item()),
        "particle_accessibility": float(latent["particle_accessibility"].squeeze().detach().cpu().item()),
        "pore_decay_tendency": float(latent["pore_decay_tendency"].squeeze().detach().cpu().item()),
        "ore_decay_strength": float(latent["ore_decay_strength"].squeeze().detach().cpu().item()),
        "passivation_strength": float(latent["passivation_strength"].squeeze().detach().cpu().item()),
        "depassivation_strength": float(latent["depassivation_strength"].squeeze().detach().cpu().item()),
        "transform_strength": float(latent["transform_strength"].squeeze().detach().cpu().item()),
        "catalyzed_fast_release_last": float(cat_states["fast_release"][-1].detach().cpu().item()),
        "catalyzed_acid_buffer_last": float(cat_states["acid_buffer_penalty"][-1].detach().cpu().item()),
        "catalyzed_diffusion_drag_last": float(cat_states["diffusion_drag"][-1].detach().cpu().item()),
        "catalyzed_solution_front_last": float(cat_states["solution_front"][-1].detach().cpu().item()),
        "catalyzed_wetting_fraction_last": float(cat_states["wetting_fraction"][-1].detach().cpu().item()),
        "catalyzed_particle_access_last": float(cat_states["particle_access"][-1].detach().cpu().item()),
        "catalyzed_pore_open_fraction_last": float(cat_states["pore_open_fraction"][-1].detach().cpu().item()),
        "catalyzed_packed_bed_contact_last": float(cat_states["packed_bed_contact"][-1].detach().cpu().item()),
        "catalyzed_hydraulic_flow_last": float(cat_states["hydraulic_flow"][-1].detach().cpu().item()),
        "control_passivation_last": float(ctrl_states["passivation"][-1].detach().cpu().item()),
        "catalyzed_passivation_last": float(cat_states["passivation"][-1].detach().cpu().item()),
        "catalyzed_depassivation_last": float(cat_states["depassivation"][-1].detach().cpu().item()),
        "catalyzed_transformation_last": float(cat_states["transformation"][-1].detach().cpu().item()),
        "catalyzed_catalyst_surface_access_last": float(cat_states["catalyst_surface_access"][-1].detach().cpu().item()),
        "catalyzed_ore_accessibility_last": float(cat_states["ore_accessibility"][-1].detach().cpu().item()),
        "pred_control_a1": float(p_ctrl.squeeze(0)[0].detach().cpu().item()),
        "pred_control_b1": float(p_ctrl.squeeze(0)[1].detach().cpu().item()),
        "pred_control_a2": float(p_ctrl.squeeze(0)[2].detach().cpu().item()),
        "pred_control_b2": float(p_ctrl.squeeze(0)[3].detach().cpu().item()),
        "pred_catalyzed_a1": float(p_cat.squeeze(0)[0].detach().cpu().item()),
        "pred_catalyzed_b1": float(p_cat.squeeze(0)[1].detach().cpu().item()),
        "pred_catalyzed_a2": float(p_cat.squeeze(0)[2].detach().cpu().item()),
        "pred_catalyzed_b2": float(p_cat.squeeze(0)[3].detach().cpu().item()),
        "control_plot_time_days": np.asarray(control_plot_time_days, dtype=float),
        "control_pred_plot": pred_ctrl_plot.detach().cpu().numpy(),
        "catalyzed_plot_time_days": np.asarray(control_plot_time_days, dtype=float),
        "plot_catalyst_cum": np.asarray(plot_profile["plot_catalyst_cum"], dtype=float),
        "catalyzed_pred_plot": pred_cat_plot.detach().cpu().numpy(),
        "catalyst_addition_start_day": float(plot_profile["catalyst_addition_start_day"]),
        "catalyst_addition_stop_day": float(plot_profile["catalyst_addition_stop_day"]),
        "weekly_catalyst_addition_kg_t": float(plot_profile["weekly_catalyst_addition_kg_t"]),
        "weekly_reference_days": float(plot_profile["weekly_reference_days"]),
        "recent_window_start_day": float(plot_profile["recent_window_start_day"]),
        "recent_window_delta_kg_t": float(plot_profile["recent_window_delta_kg_t"]),
        "recent_window_delta_tol_kg_t": float(plot_profile["recent_window_delta_tol_kg_t"]),
        "recent_window_growth_near_zero": bool(plot_profile["recent_window_growth_near_zero"]),
        "last_observed_day": float(plot_profile["last_observed_day"]),
        "stopped_before_test_end": bool(plot_profile["stopped_before_test_end"]),
        "catalyst_addition_state": str(plot_profile["catalyst_addition_state"]),
        "extension_applied": bool(plot_profile["extension_applied"]),
        "extension_target_day": float(plot_profile["target_day"]),
    }
    return rec


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.size < 2:
        return np.nan
    if np.nanstd(yt) < 1e-12:
        return np.nan
    return float(r2_score(yt, yp))


def compute_metrics_from_records(records: List[Dict[str, Any]], ensemble: bool = False) -> Dict[str, float]:
    if ensemble:
        ctrl_pred_key = "control_pred_mean"
        cat_pred_key = "catalyzed_pred_mean"
    else:
        ctrl_pred_key = "control_pred"
        cat_pred_key = "catalyzed_pred"

    ctrl_true = np.concatenate([np.asarray(r["control_true"], dtype=float) for r in records]) if records else np.array([])
    ctrl_pred = np.concatenate([np.asarray(r[ctrl_pred_key], dtype=float) for r in records]) if records else np.array([])
    cat_true = np.concatenate([np.asarray(r["catalyzed_true"], dtype=float) for r in records]) if records else np.array([])
    cat_pred = np.concatenate([np.asarray(r[cat_pred_key], dtype=float) for r in records]) if records else np.array([])

    all_true = np.concatenate([ctrl_true, cat_true]) if ctrl_true.size + cat_true.size > 0 else np.array([])
    all_pred = np.concatenate([ctrl_pred, cat_pred]) if ctrl_pred.size + cat_pred.size > 0 else np.array([])

    def _m(y_true_: np.ndarray, y_pred_: np.ndarray, tag: str) -> Dict[str, float]:
        if y_true_.size == 0:
            return {
                f"{tag}_rmse": np.nan,
                f"{tag}_mae": np.nan,
                f"{tag}_r2": np.nan,
            }
        return {
            f"{tag}_rmse": float(np.sqrt(mean_squared_error(y_true_, y_pred_))),
            f"{tag}_mae": float(mean_absolute_error(y_true_, y_pred_)),
            f"{tag}_r2": _safe_r2(y_true_, y_pred_),
        }

    metrics = {}
    metrics.update(_m(ctrl_true, ctrl_pred, "control"))
    metrics.update(_m(cat_true, cat_pred, "catalyzed"))
    metrics.update(_m(all_true, all_pred, "overall"))

    # Check strict ordering at catalyzed timepoints.
    if ensemble:
        ordering_viol = [
            np.max(np.maximum(np.asarray(r.get("control_pred_on_catalyzed_t_mean", [])) - np.asarray(r["catalyzed_pred_mean"]), 0.0))
            for r in records
        ]
    else:
        ordering_viol = [
            np.max(np.maximum(np.asarray(r.get("control_pred_on_catalyzed_t", [])) - np.asarray(r["catalyzed_pred"]), 0.0))
            for r in records
        ]
    metrics["max_ordering_violation_control_minus_cat"] = float(np.nanmax(ordering_viol)) if len(ordering_viol) else np.nan
    return metrics


def evaluate_model(model: PairCurveNet, pairs: List[PairSample], cum_scale: float) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    records = [predict_pair_record(model, pair, cum_scale) for pair in pairs]
    metrics = compute_metrics_from_records(records, ensemble=False)
    return metrics, records


def records_to_df(records: List[Dict[str, Any]], ensemble: bool = False) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"sample_id": r["sample_id"]}
        scalar_keys = [
            "tau_days",
            "temp_days",
            "kappa",
            "aging_strength",
            "catalyst_receptivity",
            "bed_permeability",
            "wetting_efficiency",
            "particle_accessibility",
            "pore_decay_tendency",
            "pred_control_a1",
            "pred_control_b1",
            "pred_control_a2",
            "pred_control_b2",
            "pred_catalyzed_a1",
            "pred_catalyzed_b1",
            "pred_catalyzed_a2",
            "pred_catalyzed_b2",
            "catalyst_addition_start_day",
            "catalyst_addition_stop_day",
            "weekly_catalyst_addition_kg_t",
            "weekly_reference_days",
            "recent_window_start_day",
            "recent_window_delta_kg_t",
            "recent_window_delta_tol_kg_t",
            "recent_window_growth_near_zero",
            "last_observed_day",
            "stopped_before_test_end",
            "catalyst_addition_state",
            "extension_applied",
            "extension_target_day",
            "catalyzed_catalyst_surface_access_last",
            "catalyzed_solution_front_last",
            "catalyzed_wetting_fraction_last",
            "catalyzed_particle_access_last",
            "catalyzed_pore_open_fraction_last",
            "catalyzed_packed_bed_contact_last",
            "catalyzed_hydraulic_flow_last",
        ]
        for k in scalar_keys:
            if k in r:
                row[k] = r[k]
        array_keys = [
            "control_t",
            "control_true",
            "control_pred",
            "control_pred_on_catalyzed_t",
            "catalyzed_t",
            "catalyzed_true",
            "catalyzed_pred",
        ]
        if ensemble:
            array_keys = [
                "control_t",
                "control_true",
                "control_pred_mean",
                "control_pred_p10",
                "control_pred_p90",
                "control_pred_on_catalyzed_t_mean",
                "catalyzed_t",
                "catalyzed_true",
                "catalyzed_pred_mean",
                "catalyzed_pred_p10",
                "catalyzed_pred_p90",
            ]
        for k in array_keys:
            if k in r:
                row[k] = serialize_array(np.asarray(r[k], dtype=float))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_ensemble_predictions(
    member_record_maps: List[Dict[str, Dict[str, Any]]],
    pairs: List[PairSample],
    pi_low: float,
    pi_high: float,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    agg_records: List[Dict[str, Any]] = []
    for pair in pairs:
        sid = pair.sample_id
        member_recs = [m[sid] for m in member_record_maps if sid in m]
        if len(member_recs) == 0:
            continue
        ctrl_stack = np.vstack([np.asarray(r["control_pred"], dtype=float) for r in member_recs])
        cat_stack = np.vstack([np.asarray(r["catalyzed_pred"], dtype=float) for r in member_recs])
        ctrl_on_cat_stack = np.vstack([np.asarray(r["control_pred_on_catalyzed_t"], dtype=float) for r in member_recs])
        plot_ctrl_stack = np.vstack(
            [np.asarray(r.get("control_pred_plot", r["control_pred_on_catalyzed_t"]), dtype=float) for r in member_recs]
        )
        plot_cat_stack = np.vstack(
            [np.asarray(r.get("catalyzed_pred_plot", r["catalyzed_pred"]), dtype=float) for r in member_recs]
        )

        rec = {
            "sample_id": sid,
            "control_t": np.asarray(member_recs[0]["control_t"], dtype=float),
            "control_true": np.asarray(member_recs[0]["control_true"], dtype=float),
            "control_pred_mean": np.mean(ctrl_stack, axis=0),
            "control_pred_p10": np.percentile(ctrl_stack, pi_low, axis=0),
            "control_pred_p90": np.percentile(ctrl_stack, pi_high, axis=0),
            "control_pred_on_catalyzed_t_mean": np.mean(ctrl_on_cat_stack, axis=0),
            "catalyzed_t": np.asarray(member_recs[0]["catalyzed_t"], dtype=float),
            "catalyzed_true": np.asarray(member_recs[0]["catalyzed_true"], dtype=float),
            "catalyzed_pred_mean": np.mean(cat_stack, axis=0),
            "catalyzed_pred_p10": np.percentile(cat_stack, pi_low, axis=0),
            "catalyzed_pred_p90": np.percentile(cat_stack, pi_high, axis=0),
            "tau_days": float(np.mean([float(r["tau_days"]) for r in member_recs])),
            "temp_days": float(np.mean([float(r["temp_days"]) for r in member_recs])),
            "kappa": float(np.mean([float(r["kappa"]) for r in member_recs])),
            "aging_strength": float(np.mean([float(r.get("aging_strength", np.nan)) for r in member_recs])),
            "n_members": int(len(member_recs)),
            "control_plot_time_days": np.asarray(
                member_recs[0].get("control_plot_time_days", member_recs[0]["control_t"]),
                dtype=float,
            ),
            "catalyzed_plot_time_days": np.asarray(
                member_recs[0].get("catalyzed_plot_time_days", member_recs[0]["catalyzed_t"]),
                dtype=float,
            ),
            "plot_catalyst_cum": np.asarray(
                member_recs[0].get("plot_catalyst_cum", np.zeros_like(member_recs[0]["catalyzed_t"])),
                dtype=float,
            ),
            "control_pred_plot_mean": np.mean(plot_ctrl_stack, axis=0),
            "control_pred_plot_p10": np.percentile(plot_ctrl_stack, pi_low, axis=0),
            "control_pred_plot_p90": np.percentile(plot_ctrl_stack, pi_high, axis=0),
            "catalyzed_pred_plot_mean": np.mean(plot_cat_stack, axis=0),
            "catalyzed_pred_plot_p10": np.percentile(plot_cat_stack, pi_low, axis=0),
            "catalyzed_pred_plot_p90": np.percentile(plot_cat_stack, pi_high, axis=0),
            "catalyst_addition_start_day": float(member_recs[0].get("catalyst_addition_start_day", np.nan)),
            "catalyst_addition_stop_day": float(member_recs[0].get("catalyst_addition_stop_day", np.nan)),
            "weekly_catalyst_addition_kg_t": float(
                np.mean([float(r.get("weekly_catalyst_addition_kg_t", 0.0)) for r in member_recs])
            ),
            "weekly_reference_days": float(np.mean([float(r.get("weekly_reference_days", 0.0)) for r in member_recs])),
            "recent_window_start_day": float(member_recs[0].get("recent_window_start_day", np.nan)),
            "recent_window_delta_kg_t": float(member_recs[0].get("recent_window_delta_kg_t", np.nan)),
            "recent_window_delta_tol_kg_t": float(member_recs[0].get("recent_window_delta_tol_kg_t", np.nan)),
            "recent_window_growth_near_zero": bool(member_recs[0].get("recent_window_growth_near_zero", False)),
            "last_observed_day": float(np.mean([float(r.get("last_observed_day", np.nan)) for r in member_recs])),
            "stopped_before_test_end": bool(member_recs[0].get("stopped_before_test_end", False)),
            "catalyst_addition_state": str(member_recs[0].get("catalyst_addition_state", "")),
            "extension_applied": bool(any(bool(r.get("extension_applied", False)) for r in member_recs)),
            "extension_target_day": float(
                np.mean([float(r.get("extension_target_day", np.nan)) for r in member_recs])
            ),
        }
        agg_records.append(rec)
    metrics = compute_metrics_from_records(agg_records, ensemble=True)
    return metrics, agg_records


# ---------------------------
# Plotting
# ---------------------------
def plot_single_record(record: Dict[str, Any], plot_path: str, title: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(
        record["catalyzed_t"],
        record["catalyzed_true"],
        "o",
        ms=3,
        alpha=0.6,
        color="#ff7f0e",
        label="Catalyzed True",
        zorder=1,
    )
    plt.plot(record["catalyzed_t"], record["catalyzed_pred"], "-", lw=2, color="#ff7f0e", label="Catalyzed Pred", zorder=2)
    plt.plot(
        record["control_t"],
        record["control_true"],
        "o",
        ms=3,
        alpha=0.6,
        color="#1f77b4",
        label="Control True",
        zorder=3,
    )
    plt.plot(record["control_t"], record["control_pred"], "-", lw=2, color="#1f77b4", label="Control Pred", zorder=4)

    plt.xlabel("leach_duration_days")
    plt.ylabel("cu_recovery_%")
    plt.ylim(0, 100)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    _annotate_ensemble_extension(plt.gca(), record)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=int(CONFIG["plot_dpi"]))
    plt.close()


def plot_ensemble_record(record: Dict[str, Any], plot_path: str, title: str) -> None:
    plt.figure(figsize=(9, 5))

    ct = np.asarray(record.get("control_plot_time_days", record["control_t"]), dtype=float)
    cp = np.asarray(record.get("control_pred_plot_mean", record["control_pred_mean"]), dtype=float)
    cp_lo = np.asarray(record.get("control_pred_plot_p10", record["control_pred_p10"]), dtype=float)
    cp_hi = np.asarray(record.get("control_pred_plot_p90", record["control_pred_p90"]), dtype=float)
    cy = np.asarray(record["control_true"], dtype=float)

    kt = np.asarray(record.get("catalyzed_plot_time_days", record["catalyzed_t"]), dtype=float)
    kp = np.asarray(record.get("catalyzed_pred_plot_mean", record["catalyzed_pred_mean"]), dtype=float)
    kp_lo = np.asarray(record.get("catalyzed_pred_plot_p10", record["catalyzed_pred_p10"]), dtype=float)
    kp_hi = np.asarray(record.get("catalyzed_pred_plot_p90", record["catalyzed_pred_p90"]), dtype=float)
    ky = np.asarray(record["catalyzed_true"], dtype=float)

    plt.fill_between(kt, kp_lo, kp_hi, color="#ff7f0e", alpha=0.18, label="Catalyzed P10-P90", zorder=1)
    plt.plot(kt, kp, "-", lw=2, color="#ff7f0e", label="Catalyzed Ensemble Mean", zorder=2)
    plt.plot(
        np.asarray(record["catalyzed_t"], dtype=float),
        ky,
        "o",
        ms=3,
        alpha=0.6,
        color="#ff7f0e",
        label="Catalyzed True",
        zorder=3,
    )

    plt.fill_between(ct, cp_lo, cp_hi, color="#1f77b4", alpha=0.18, label="Control P10-P90", zorder=4)
    plt.plot(ct, cp, "-", lw=2, color="#1f77b4", label="Control Ensemble Mean", zorder=5)
    plt.plot(
        np.asarray(record["control_t"], dtype=float),
        cy,
        "o",
        ms=3,
        alpha=0.6,
        color="#1f77b4",
        label="Control True",
        zorder=6,
    )

    plt.xlabel("leach_duration_days")
    plt.ylabel("cu_recovery_%")
    plt.ylim(0, 100)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    _annotate_ensemble_extension(plt.gca(), record)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=int(CONFIG["plot_dpi"]))
    plt.close()


def plot_overall_scatter(records: List[Dict[str, Any]], plot_path: str, title: str, ensemble: bool) -> None:
    if len(records) == 0:
        return
    if ensemble:
        ctrl_pred_key = "control_pred_mean"
        cat_pred_key = "catalyzed_pred_mean"
    else:
        ctrl_pred_key = "control_pred"
        cat_pred_key = "catalyzed_pred"

    ctrl_true = np.concatenate([np.asarray(r["control_true"], dtype=float) for r in records])
    ctrl_pred = np.concatenate([np.asarray(r[ctrl_pred_key], dtype=float) for r in records])
    cat_true = np.concatenate([np.asarray(r["catalyzed_true"], dtype=float) for r in records])
    cat_pred = np.concatenate([np.asarray(r[cat_pred_key], dtype=float) for r in records])
    all_true = np.concatenate([ctrl_true, cat_true])
    all_pred = np.concatenate([ctrl_pred, cat_pred])

    low = float(min(np.nanmin(all_true), np.nanmin(all_pred), 0.0))
    high = float(max(np.nanmax(all_true), np.nanmax(all_pred), 100.0))

    plt.figure(figsize=(6, 6))
    plt.scatter(ctrl_true, ctrl_pred, s=15, alpha=0.6, color="#1f77b4", label="Control")
    plt.scatter(cat_true, cat_pred, s=15, alpha=0.6, color="#ff7f0e", label="Catalyzed")
    plt.plot([low, high], [low, high], "k--", lw=1.2, label="Ideal")
    plt.xlabel("True cu_recovery_%")
    plt.ylabel("Predicted cu_recovery_%")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=int(CONFIG["plot_dpi"]))
    plt.close()


def plot_overall_metric_bars(metrics: Dict[str, float], plot_path: str, title: str) -> None:
    categories = ["control", "catalyzed", "overall"]
    rmse_vals = [float(metrics.get(f"{c}_rmse", np.nan)) for c in categories]
    mae_vals = [float(metrics.get(f"{c}_mae", np.nan)) for c in categories]
    r2_vals = [float(metrics.get(f"{c}_r2", np.nan)) for c in categories]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(categories))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    axes[0].bar(x, rmse_vals, color=colors, alpha=0.8)
    axes[0].set_title("RMSE")
    axes[0].set_xticks(x, categories, rotation=15)
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(x, mae_vals, color=colors, alpha=0.8)
    axes[1].set_title("MAE")
    axes[1].set_xticks(x, categories, rotation=15)
    axes[1].grid(alpha=0.25, axis="y")

    axes[2].bar(x, r2_vals, color=colors, alpha=0.8)
    axes[2].set_title("R2")
    axes[2].set_xticks(x, categories, rotation=15)
    axes[2].grid(alpha=0.25, axis="y")

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=int(CONFIG["plot_dpi"]))
    plt.close(fig)


def plot_validation_learning_curves(
    member_histories: List[Dict[str, Any]],
    plot_path: str,
    title: str,
) -> None:
    if len(member_histories) == 0:
        return

    max_epoch = int(
        max(
            max(int(row["epoch"]) for row in m["history"])
            for m in member_histories
            if len(m["history"]) > 0
        )
    )
    epochs = np.arange(1, max_epoch + 1)
    train_mat = np.full((len(member_histories), max_epoch), np.nan, dtype=float)
    eval_mat = np.full((len(member_histories), max_epoch), np.nan, dtype=float)

    plt.figure(figsize=(9, 5))
    for i, m in enumerate(member_histories):
        hist = m["history"]
        if len(hist) == 0:
            continue
        hdf = pd.DataFrame(hist).sort_values("epoch")
        e = hdf["epoch"].astype(int).to_numpy()
        tr = hdf["train_loss"].astype(float).to_numpy()
        ev = hdf["eval_loss"].astype(float).to_numpy()
        train_mat[i, e - 1] = tr
        eval_mat[i, e - 1] = ev
        plt.plot(e, tr, color="#1f77b4", alpha=0.20, lw=1.0)
        plt.plot(e, ev, color="#ff7f0e", alpha=0.20, lw=1.0)

    mean_train = np.nanmean(train_mat, axis=0)
    mean_eval = np.nanmean(eval_mat, axis=0)
    plt.plot(epochs, mean_train, color="#1f77b4", lw=2.5, label="Mean Train Loss")
    plt.plot(epochs, mean_eval, color="#ff7f0e", lw=2.5, label="Mean Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=int(CONFIG["plot_dpi"]))
    plt.close()


# ---------------------------
# Orchestration
# ---------------------------
def main() -> None:
    set_all_seeds(int(CONFIG["seed"]), deterministic=True)

    # 1) Data analysis
    df = pd.read_csv(DATA_PATH, sep=",")
    df, training_exclusion_summary = apply_training_pair_exclusions(df)
    if training_exclusion_summary["applied"]:
        print(
            "[Data] Excluded training sample ids: "
            f"{', '.join(training_exclusion_summary['excluded_pair_ids'])} | "
            f"rows={training_exclusion_summary['excluded_row_count']} | "
            f"pairs={training_exclusion_summary['excluded_pair_count']}"
        )
    analysis_summary = analyze_dataset(df)
    analysis_summary["training_exclusions"] = training_exclusion_summary
    save_json(os.path.join(OUTPUTS_ROOT, "data_analysis_summary.json"), analysis_summary)

    # 2) Biexponential prefit for bounds/limits
    df_prefit = df.copy()
    prefit_df = prefit_biexponential_for_rows(df_prefit)
    df_prefit["row_index"] = np.arange(len(df_prefit))
    df_prefit = df_prefit.merge(prefit_df, on="row_index", how="left")
    prefit_out_path = os.path.join(OUTPUTS_ROOT, "row_biexponential_prefit.csv")
    df_prefit.to_csv(prefit_out_path, index=False)

    # 3) Build pair-level control/catalyzed samples with time-dependent arrays
    pairs = build_pair_samples(df_prefit)
    if len(pairs) < 6:
        raise ValueError(f"Expected at least 6 paired samples; got {len(pairs)}.")

    catalyst_stop_window_days = float(CONFIG.get("catalyst_extension_window_days", 21.0))
    catalyst_stop_report_df, catalyst_stop_summary = build_catalyst_stop_report(
        pairs=pairs,
        history_window_days=catalyst_stop_window_days,
    )
    catalyst_stop_report_path = os.path.join(OUTPUTS_ROOT, "catalyst_addition_status_by_sample.csv")
    catalyst_stopped_only_path = os.path.join(OUTPUTS_ROOT, "catalyst_stopped_before_test_end.csv")
    catalyst_stop_summary_path = os.path.join(OUTPUTS_ROOT, "catalyst_addition_status_summary.json")
    catalyst_stop_report_df.to_csv(catalyst_stop_report_path, index=False)
    catalyst_stop_report_df.loc[catalyst_stop_report_df["stopped_before_test_end"]].to_csv(
        catalyst_stopped_only_path,
        index=False,
    )
    save_json(catalyst_stop_summary_path, catalyst_stop_summary)
    print(
        "[Data] Catalyst stop summary: "
        f"window_days={catalyst_stop_window_days:.0f} | "
        f"stopped_before_end={catalyst_stop_summary['n_stopped_before_test_end']} / {catalyst_stop_summary['n_pairs']}"
    )

    deploy_imputer, deploy_scaler = fit_static_transformers(pairs)
    apply_static_transformers(pairs, deploy_imputer, deploy_scaler)

    extrapolation_ref = build_extrapolation_reference(pairs, deploy_imputer, deploy_scaler)
    save_json(os.path.join(OUTPUTS_ROOT, "extrapolation_reference.json"), extrapolation_ref)

    rng_template = np.random.default_rng(int(CONFIG["seed"]))
    example_pair = pairs[int(rng_template.integers(0, len(pairs)))]
    example_input_path = os.path.join(INPUT_EXAMPLE_ROOT, "model_input_example.xlsx")
    create_input_example_excel(
        output_path=example_input_path,
        extrapolation_ref=extrapolation_ref,
        example_pair=example_pair,
    )

    # 4) Global parameter bounds from all paired samples
    ctrl_all_params = np.vstack([p.control.fit_params for p in pairs])
    cat_all_params = np.vstack([p.catalyzed.fit_params for p in pairs])
    ctrl_lb_full, ctrl_ub_full = derive_param_bounds(ctrl_all_params, None)
    cat_lb_full, cat_ub_full = derive_param_bounds(cat_all_params, None)

    param_bounds_payload = {
        "control_lb": ctrl_lb_full.tolist(),
        "control_ub": ctrl_ub_full.tolist(),
        "catalyzed_lb": cat_lb_full.tolist(),
        "catalyzed_ub": cat_ub_full.tolist(),
        "source": "full_data_quantile_05_95_with_margin",
    }
    save_json(os.path.join(OUTPUTS_ROOT, "param_bounds.json"), param_bounds_payload)

    tmax_days = float(
        max(
            max(float(np.max(p.control.time)) for p in pairs),
            max(float(np.max(p.catalyzed.time)) for p in pairs),
        )
    )
    cum_scale = float(
        max(
            1e-6,
            max(float(np.nanmax(p.catalyzed.catalyst_cum)) for p in pairs if p.catalyzed.catalyst_cum.size > 0),
        )
    )
    geo_idx = [STATIC_PREDICTOR_COLUMNS.index(c) for c in GEO_PRIORITY_COLUMNS if c in STATIC_PREDICTOR_COLUMNS]

    # 5) Repeated K-fold ensemble runs (validation + deployed members)
    val_member_plot_root = os.path.join(PLOTS_ROOT, "validation_members")
    val_member_out_root = os.path.join(OUTPUTS_ROOT, "validation_members")
    val_member_model_root = os.path.join(MODELS_ROOT, "validation_members")
    for p in [val_member_plot_root, val_member_out_root, val_member_model_root]:
        os.makedirs(p, exist_ok=True)

    member_record_maps_val: List[Dict[str, Dict[str, Any]]] = []
    member_metrics_rows = []
    member_histories = []
    member_split_summaries = []
    member_models: List[Dict[str, Any]] = []

    cv_splits = build_repeated_kfold_member_splits(
        n_samples=len(pairs),
        n_splits=int(CONFIG.get("cv_n_splits", 5)),
        n_repeats=int(CONFIG.get("cv_n_repeats", 2)),
        random_state=int(CONFIG.get("cv_random_state", CONFIG["seed"])),
        member_seed_base=int(CONFIG.get("cv_member_seed_base", 10000)),
    )

    cv_parallel = resolve_cv_parallelism(len(cv_splits))
    print(
        "[CV Ensemble] Parallel config: "
        f"workers={cv_parallel['workers']}, "
        f"torch_threads/worker={cv_parallel['torch_threads_per_worker']}, "
        f"interop_threads/worker={cv_parallel['torch_interop_threads_per_worker']}, "
        f"cpu_cores={cv_parallel['total_cores']}"
    )

    member_jobs: List[Dict[str, Any]] = []
    for split in cv_splits:
        member_idx = int(split["member_idx"])
        repeat_idx = int(split["repeat_idx"])
        fold_idx = int(split["fold_idx"])
        member_seed = int(split["member_seed"])
        train_indices = split["train_indices"]
        val_indices = split["val_indices"]

        train_pairs_seed = [pairs[i] for i in train_indices]
        val_pairs_seed = [pairs[i] for i in val_indices]
        if len(val_pairs_seed) == 0:
            raise ValueError(f"CV member {member_idx} produced empty validation split.")

        member_tag = format_member_tag(
            member_idx=member_idx,
            repeat_idx=repeat_idx,
            fold_idx=fold_idx,
            member_seed=member_seed,
        )

        member_jobs.append(
            {
                "member_tag": member_tag,
                "member_idx": member_idx,
                "repeat_idx": repeat_idx,
                "fold_idx": fold_idx,
                "member_seed": member_seed,
                "train_pairs": train_pairs_seed,
                "val_pairs": val_pairs_seed,
                "tmax_days": tmax_days,
                "cum_scale": cum_scale,
                "epochs": int(CONFIG["epochs"]),
                "patience": int(CONFIG["patience"]),
                "geo_idx": geo_idx,
                "val_member_model_root": val_member_model_root,
                "val_member_out_root": val_member_out_root,
                "val_member_plot_root": val_member_plot_root,
                "torch_threads_per_worker": int(cv_parallel["torch_threads_per_worker"]),
                "torch_interop_threads_per_worker": int(cv_parallel["torch_interop_threads_per_worker"]),
            }
        )

    member_results: List[Dict[str, Any]] = []
    if int(cv_parallel["workers"]) > 1:
        with ProcessPoolExecutor(max_workers=int(cv_parallel["workers"])) as executor:
            future_to_tag = {executor.submit(train_validation_member_job, job): job["member_tag"] for job in member_jobs}
            for future in as_completed(future_to_tag):
                member_tag = future_to_tag[future]
                try:
                    member_results.append(future.result())
                    print(f"[CV Ensemble] Completed {member_tag}")
                except Exception as exc:
                    raise RuntimeError(f"[CV Ensemble] Failed {member_tag}: {exc}") from exc
    else:
        configure_torch_cpu_parallelism(
            num_threads=int(cv_parallel["torch_threads_per_worker"]),
            num_interop_threads=int(cv_parallel["torch_interop_threads_per_worker"]),
        )
        for job in member_jobs:
            member_results.append(train_validation_member_job(job))

    member_results = sorted(member_results, key=lambda x: int(x["member_idx"]))
    for result in member_results:
        member_record_maps_val.append({r["sample_id"]: r for r in result["records_val"]})
        member_metrics_rows.append(
            {
                "member_tag": result["member_tag"],
                "member_idx": int(result["member_idx"]),
                "repeat_idx": int(result["repeat_idx"]),
                "fold_idx": int(result["fold_idx"]),
                "seed": int(result["seed"]),
                **result["metrics_val"],
                "best_eval_loss": float(result["best_eval_loss"]),
                "best_epoch": int(result["best_epoch"]),
                "n_train_pairs": int(result["n_train_pairs"]),
                "n_validation_pairs": int(result["n_validation_pairs"]),
            }
        )
        member_histories.append({"seed": int(result["seed"]), "history": result["history"]})
        member_split_summaries.append(
            {
                "member_tag": result["member_tag"],
                "member_idx": int(result["member_idx"]),
                "repeat_idx": int(result["repeat_idx"]),
                "fold_idx": int(result["fold_idx"]),
                "seed": int(result["seed"]),
                "n_train_pairs": int(result["n_train_pairs"]),
                "n_validation_pairs": int(result["n_validation_pairs"]),
                "train_sample_ids": result["train_sample_ids"],
                "validation_sample_ids": result["validation_sample_ids"],
            }
        )

        checkpoint = load_torch_checkpoint(result["model_ckpt_path"], map_location=device)
        model = build_member_model_from_checkpoint(checkpoint)
        member_models.append(
            {
                "member_tag": result["member_tag"],
                "member_idx": int(result["member_idx"]),
                "repeat_idx": int(result["repeat_idx"]),
                "fold_idx": int(result["fold_idx"]),
                "seed": int(result["seed"]),
                "model": model,
                "imputer": result["imputer"],
                "scaler": result["scaler"],
            }
        )

    pd.DataFrame(member_metrics_rows).to_csv(
        os.path.join(val_member_out_root, "validation_member_metrics_summary.csv"),
        index=False,
    )
    split_summary = {
        "strategy": "repeated_kfold",
        "n_splits": int(CONFIG.get("cv_n_splits", 5)),
        "n_repeats": int(CONFIG.get("cv_n_repeats", 2)),
        "random_state": int(CONFIG.get("cv_random_state", CONFIG["seed"])),
        "n_pairs_total": int(len(pairs)),
        "members": member_split_summaries,
    }
    save_json(os.path.join(OUTPUTS_ROOT, "train_validation_split_summary.json"), split_summary)
    plot_validation_learning_curves(
        member_histories=member_histories,
        plot_path=os.path.join(val_member_plot_root, "validation_training_learning_curves.png"),
        title="Repeated K-Fold Training Curves Across Members",
    )

    # 6) Validation OOF aggregation and plots (strict no-leak evaluation)
    val_union_ids = sorted({sid for m in member_record_maps_val for sid in m.keys()})
    val_union_set = set(val_union_ids)
    val_pairs_for_agg = [p for p in pairs if p.sample_id in val_union_set]
    val_oof_metrics, val_oof_records = aggregate_ensemble_predictions(
        member_record_maps=member_record_maps_val,
        pairs=val_pairs_for_agg,
        pi_low=float(CONFIG["ensemble_pi_low"]),
        pi_high=float(CONFIG["ensemble_pi_high"]),
    )
    val_oof_out_root = os.path.join(OUTPUTS_ROOT, "validation_oof_ensemble")
    val_oof_plot_root = os.path.join(PLOTS_ROOT, "validation_oof_ensemble")
    os.makedirs(val_oof_out_root, exist_ok=True)
    os.makedirs(val_oof_plot_root, exist_ok=True)

    records_to_df(val_oof_records, ensemble=True).to_csv(
        os.path.join(val_oof_out_root, "validation_oof_ensemble_predictions.csv"),
        index=False,
    )
    save_json(
        os.path.join(val_oof_out_root, "validation_oof_ensemble_metrics.json"),
        val_oof_metrics,
    )
    pd.DataFrame([{"stage": "validation_oof_ensemble", **val_oof_metrics}]).to_csv(
        os.path.join(val_oof_out_root, "validation_oof_ensemble_overall_statistics.csv"),
        index=False,
    )
    plot_overall_scatter(
        records=val_oof_records,
        plot_path=os.path.join(val_oof_plot_root, "validation_oof_ensemble_overall_scatter.png"),
        title="Validation OOF Ensemble Scatter",
        ensemble=True,
    )
    plot_overall_metric_bars(
        metrics=val_oof_metrics,
        plot_path=os.path.join(val_oof_plot_root, "validation_oof_ensemble_overall_statistics.png"),
        title="Validation OOF Ensemble Overall Statistics",
    )
    for r in val_oof_records:
        plot_ensemble_record(
            record=r,
            plot_path=os.path.join(val_oof_plot_root, f"{r['sample_id']}.png"),
            title=f"Validation OOF Ensemble Prediction ({r['sample_id']})",
        )

    # 7) Deployed ensemble from CV members (captures split uncertainty)
    full_member_model_root = val_member_model_root
    full_member_out_root = os.path.join(OUTPUTS_ROOT, "deployed_cv_members")
    full_ensemble_out_root = os.path.join(OUTPUTS_ROOT, "deployed_cv_ensemble")
    full_ensemble_plot_root = os.path.join(PLOTS_ROOT, "deployed_cv_ensemble")
    for p in [full_member_out_root, full_ensemble_out_root, full_ensemble_plot_root]:
        os.makedirs(p, exist_ok=True)

    member_record_maps_full: List[Dict[str, Dict[str, Any]]] = []
    full_member_metrics_rows = []
    for m in member_models:
        member_tag = m["member_tag"]
        apply_static_transformers(pairs, m["imputer"], m["scaler"])
        metrics_member, records_member = evaluate_model(m["model"], pairs, cum_scale)
        records_to_df(records_member, ensemble=False).to_csv(
            os.path.join(full_member_out_root, f"{member_tag}_predictions.csv"),
            index=False,
        )
        save_json(
            os.path.join(full_member_out_root, f"{member_tag}_metrics.json"),
            metrics_member,
        )
        member_record_maps_full.append({r["sample_id"]: r for r in records_member})
        full_member_metrics_rows.append(
            {
                "member_tag": member_tag,
                "member_idx": int(m["member_idx"]),
                "repeat_idx": int(m["repeat_idx"]),
                "fold_idx": int(m["fold_idx"]),
                "seed": int(m["seed"]),
                **metrics_member,
            }
        )

    pd.DataFrame(full_member_metrics_rows).to_csv(
        os.path.join(full_member_out_root, "deployed_cv_member_metrics_summary.csv"),
        index=False,
    )

    final_metrics, final_records = aggregate_ensemble_predictions(
        member_record_maps=member_record_maps_full,
        pairs=pairs,
        pi_low=float(CONFIG["ensemble_pi_low"]),
        pi_high=float(CONFIG["ensemble_pi_high"]),
    )
    records_to_df(final_records, ensemble=True).to_csv(
        os.path.join(full_ensemble_out_root, "deployed_cv_ensemble_predictions.csv"),
        index=False,
    )
    save_json(os.path.join(full_ensemble_out_root, "deployed_cv_ensemble_metrics.json"), final_metrics)
    pd.DataFrame([{"stage": "deployed_cv_ensemble", **final_metrics}]).to_csv(
        os.path.join(full_ensemble_out_root, "deployed_cv_ensemble_overall_statistics.csv"),
        index=False,
    )
    plot_overall_scatter(
        records=final_records,
        plot_path=os.path.join(full_ensemble_plot_root, "deployed_cv_ensemble_overall_scatter.png"),
        title="Deployed CV Ensemble Scatter",
        ensemble=True,
    )
    plot_overall_metric_bars(
        metrics=final_metrics,
        plot_path=os.path.join(full_ensemble_plot_root, "deployed_cv_ensemble_overall_statistics.png"),
        title="Deployed CV Ensemble Overall Statistics",
    )
    for r in final_records:
        plot_ensemble_record(
            record=r,
            plot_path=os.path.join(full_ensemble_plot_root, f"{r['sample_id']}.png"),
            title=f"Deployed CV Ensemble Prediction ({r['sample_id']})",
        )

    # 7b) Validation ensemble using all members (visual consistency with deployed ensemble)
    val_ensemble_metrics, val_ensemble_records = aggregate_ensemble_predictions(
        member_record_maps=member_record_maps_full,
        pairs=val_pairs_for_agg,
        pi_low=float(CONFIG["ensemble_pi_low"]),
        pi_high=float(CONFIG["ensemble_pi_high"]),
    )
    val_ensemble_out_root = os.path.join(OUTPUTS_ROOT, "validation_ensemble")
    val_ensemble_plot_root = os.path.join(PLOTS_ROOT, "validation_ensemble")
    os.makedirs(val_ensemble_out_root, exist_ok=True)
    os.makedirs(val_ensemble_plot_root, exist_ok=True)

    records_to_df(val_ensemble_records, ensemble=True).to_csv(
        os.path.join(val_ensemble_out_root, "validation_ensemble_predictions.csv"),
        index=False,
    )
    save_json(
        os.path.join(val_ensemble_out_root, "validation_ensemble_metrics.json"),
        val_ensemble_metrics,
    )
    pd.DataFrame([{"stage": "validation_ensemble", **val_ensemble_metrics}]).to_csv(
        os.path.join(val_ensemble_out_root, "validation_ensemble_overall_statistics.csv"),
        index=False,
    )
    plot_overall_scatter(
        records=val_ensemble_records,
        plot_path=os.path.join(val_ensemble_plot_root, "validation_ensemble_overall_scatter.png"),
        title="Validation Ensemble Scatter (All Members)",
        ensemble=True,
    )
    plot_overall_metric_bars(
        metrics=val_ensemble_metrics,
        plot_path=os.path.join(val_ensemble_plot_root, "validation_ensemble_overall_statistics.png"),
        title="Validation Ensemble Overall Statistics (All Members)",
    )
    for r in val_ensemble_records:
        plot_ensemble_record(
            record=r,
            plot_path=os.path.join(val_ensemble_plot_root, f"{r['sample_id']}.png"),
            title=f"Validation Ensemble Prediction ({r['sample_id']})",
        )

    # 8) Inference from user-provided Excel inputs + automatic extrapolation warnings
    inference_entries = []
    input_excel_files = [
        p
        for p in sorted(glob.glob(os.path.join(INPUT_EXAMPLE_ROOT, "*.xlsx")))
        if not os.path.basename(p).startswith("~$")
    ]
    for input_path in input_excel_files:
        try:
            input_payload = load_model_input_excel(input_path)
            sample_name = sanitize_filename(input_payload["sample_name"])
            sample_dir = os.path.join(INFERENCE_ROOT, sample_name)
            os.makedirs(sample_dir, exist_ok=True)

            plot_profile = build_shared_ensemble_plot_profile(
                time_days=input_payload["time_days"],
                catalyst_cum=input_payload["catalyst_cum"],
                target_day=float(CONFIG.get("ensemble_plot_target_day", 2500.0)),
                step_days=float(CONFIG.get("ensemble_plot_step_days", 1.0)),
                history_window_days=float(CONFIG.get("catalyst_extension_window_days", 21.0)),
            )
            control_plot_time_days = np.asarray(plot_profile["plot_time_days"], dtype=float)

            pred_record, uncertainty = predict_new_sample_ensemble(
                member_models=member_models,
                static_raw=input_payload["static_raw"],
                time_days=plot_profile["plot_time_days"],
                catalyst_cum=plot_profile["plot_catalyst_cum"],
                cum_scale=cum_scale,
                pi_low=float(CONFIG["ensemble_pi_low"]),
                pi_high=float(CONFIG["ensemble_pi_high"]),
                control_time_days=control_plot_time_days,
            )
            pred_record.update(
                {
                    "catalyzed_plot_time_days": np.asarray(plot_profile["plot_time_days"], dtype=float),
                    "plot_catalyst_cum": np.asarray(plot_profile["plot_catalyst_cum"], dtype=float),
                    "catalyst_addition_start_day": float(plot_profile["catalyst_addition_start_day"]),
                    "catalyst_addition_stop_day": float(plot_profile["catalyst_addition_stop_day"]),
                    "weekly_catalyst_addition_kg_t": float(plot_profile["weekly_catalyst_addition_kg_t"]),
                    "weekly_reference_days": float(plot_profile["weekly_reference_days"]),
                    "recent_window_start_day": float(plot_profile["recent_window_start_day"]),
                    "recent_window_delta_kg_t": float(plot_profile["recent_window_delta_kg_t"]),
                    "recent_window_delta_tol_kg_t": float(plot_profile["recent_window_delta_tol_kg_t"]),
                    "recent_window_growth_near_zero": bool(plot_profile["recent_window_growth_near_zero"]),
                    "last_observed_day": float(plot_profile["last_observed_day"]),
                    "stopped_before_test_end": bool(plot_profile["stopped_before_test_end"]),
                    "catalyst_addition_state": str(plot_profile["catalyst_addition_state"]),
                    "extension_applied": bool(plot_profile["extension_applied"]),
                    "extension_target_day": float(plot_profile["target_day"]),
                }
            )
            risk = evaluate_extrapolation_risk(
                static_raw=input_payload["static_raw"],
                time_days=plot_profile["time_days"],
                catalyst_cum=plot_profile["catalyst_cum"],
                extrapolation_ref=extrapolation_ref,
                imputer=deploy_imputer,
                scaler=deploy_scaler,
                uncertainty=uncertainty,
            )

            pred_df = pd.DataFrame(
                {
                    "leach_duration_days": np.asarray(pred_record["time_days"], dtype=float),
                    "cumulative_catalyst_addition_kg_t": np.asarray(
                        pred_record["cumulative_catalyst_addition_kg_t"], dtype=float
                    ),
                    "control_pred_mean": np.asarray(pred_record["control_pred_mean"], dtype=float),
                    "control_pred_p10": np.asarray(pred_record["control_pred_p10"], dtype=float),
                    "control_pred_p90": np.asarray(pred_record["control_pred_p90"], dtype=float),
                    "catalyzed_pred_mean": np.asarray(pred_record["catalyzed_pred_mean"], dtype=float),
                    "catalyzed_pred_p10": np.asarray(pred_record["catalyzed_pred_p10"], dtype=float),
                    "catalyzed_pred_p90": np.asarray(pred_record["catalyzed_pred_p90"], dtype=float),
                }
            )
            pred_csv_path = os.path.join(sample_dir, "predicted_curve.csv")
            pred_df.to_csv(pred_csv_path, index=False)

            pred_json_path = os.path.join(sample_dir, "prediction_summary.json")
            save_json(
                pred_json_path,
                {
                    "input_file": input_path,
                    "sample_name": sample_name,
                    "prediction": pred_record,
                    "uncertainty": uncertainty,
                    "risk": risk,
                },
            )
            risk_json_path = os.path.join(sample_dir, "extrapolation_warning.json")
            save_json(risk_json_path, risk)

            plot_path = os.path.join(sample_dir, "predicted_curve.png")
            plot_inference_ensemble_curve(
                record=pred_record,
                risk=risk,
                plot_path=plot_path,
                title=f"Inference Prediction ({sample_name})",
            )

            msg = f"[Inference] {sample_name}: risk={risk['risk_level']} | {risk['message']}"
            if risk["risk_level"] == "RED":
                print("WARNING:", msg)
            else:
                print(msg)

            inference_entries.append(
                {
                    "input_file": input_path,
                    "sample_name": sample_name,
                    "risk_level": risk["risk_level"],
                    "risk_score": risk["risk_score"],
                    "message": risk["message"],
                    "predicted_curve_csv": pred_csv_path,
                    "prediction_summary_json": pred_json_path,
                    "warning_json": risk_json_path,
                    "plot_path": plot_path,
                }
            )
        except Exception as exc:
            print(f"[Inference] Failed for {input_path}: {exc}")
            inference_entries.append(
                {
                    "input_file": input_path,
                    "error": str(exc),
                }
            )

    pd.DataFrame(inference_entries).to_csv(
        os.path.join(INFERENCE_ROOT, "inference_summary.csv"),
        index=False,
    )

    manifest = {
        "project_root": PROJECT_ROOT,
        "data_analysis_summary": os.path.join(OUTPUTS_ROOT, "data_analysis_summary.json"),
        "prefit_table": prefit_out_path,
        "param_bounds": os.path.join(OUTPUTS_ROOT, "param_bounds.json"),
        "catalyst_addition_status_csv": catalyst_stop_report_path,
        "catalyst_stopped_before_test_end_csv": catalyst_stopped_only_path,
        "catalyst_addition_status_summary": catalyst_stop_summary_path,
        "input_example_excel": example_input_path,
        "input_example_root": INPUT_EXAMPLE_ROOT,
        "extrapolation_reference": os.path.join(OUTPUTS_ROOT, "extrapolation_reference.json"),
        "train_validation_split_summary": os.path.join(OUTPUTS_ROOT, "train_validation_split_summary.json"),
        "validation_member_outputs": val_member_out_root,
        "validation_member_models": val_member_model_root,
        "validation_member_plots": val_member_plot_root,
        "validation_learning_curves_plot": os.path.join(val_member_plot_root, "validation_training_learning_curves.png"),
        "validation_oof_ensemble_outputs": val_oof_out_root,
        "validation_oof_ensemble_plots": val_oof_plot_root,
        "validation_oof_ensemble_overall_stats_csv": os.path.join(
            val_oof_out_root, "validation_oof_ensemble_overall_statistics.csv"
        ),
        "validation_oof_ensemble_overall_stats_plot": os.path.join(
            val_oof_plot_root, "validation_oof_ensemble_overall_statistics.png"
        ),
        "validation_ensemble_outputs": val_ensemble_out_root,
        "validation_ensemble_plots": val_ensemble_plot_root,
        "validation_ensemble_overall_stats_csv": os.path.join(
            val_ensemble_out_root, "validation_ensemble_overall_statistics.csv"
        ),
        "validation_ensemble_overall_stats_plot": os.path.join(
            val_ensemble_plot_root, "validation_ensemble_overall_statistics.png"
        ),
        "deployed_member_outputs": full_member_out_root,
        "deployed_member_models": full_member_model_root,
        "deployed_ensemble_outputs": full_ensemble_out_root,
        "deployed_ensemble_plots": full_ensemble_plot_root,
        "deployed_ensemble_overall_stats_csv": os.path.join(
            full_ensemble_out_root, "deployed_cv_ensemble_overall_statistics.csv"
        ),
        "deployed_ensemble_overall_stats_plot": os.path.join(
            full_ensemble_plot_root, "deployed_cv_ensemble_overall_statistics.png"
        ),
        "validation_oof_ensemble_metrics": val_oof_metrics,
        "validation_ensemble_metrics": val_ensemble_metrics,
        "deployed_ensemble_metrics": final_metrics,
        "inference_root": INFERENCE_ROOT,
        "inference_summary_csv": os.path.join(INFERENCE_ROOT, "inference_summary.csv"),
        "inference_entries": inference_entries,
    }
    save_json(os.path.join(OUTPUTS_ROOT, "run_manifest.json"), manifest)

    print("\nRun complete")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Validation OOF ensemble metrics: {val_oof_metrics}")
    print(f"Validation ensemble metrics: {val_ensemble_metrics}")
    print(f"Deployed CV ensemble metrics: {final_metrics}")
    print(f"Catalyst stopped-before-end samples: {catalyst_stopped_only_path}")
    print(f"Input example template: {example_input_path}")
    print(f"Inference summary: {os.path.join(INFERENCE_ROOT, 'inference_summary.csv')}")
    print(f"Manifest: {os.path.join(OUTPUTS_ROOT, 'run_manifest.json')}")

if __name__ == "__main__":
    main()

#%%
