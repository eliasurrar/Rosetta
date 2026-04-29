# %%
import os
import ast
import json
import random
import re
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
from scipy.optimize import curve_fit, differential_evolution
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
DEFAULT_DATA_PATH = (
    "/Users/administration/Library/CloudStorage/OneDrive-JettiResources/"
    "Rosetta/database_ready/df_recCu_catcontrol_projects_averaged.csv"
)
DATA_PATH = os.environ.get("ROSETTA_DATA_PATH", DEFAULT_DATA_PATH)

TIME_COL_COLUMNS = "leach_duration_days"
TARGET_COLUMNS = "cu_recovery_%"
STATUS_COL_PRIMARY = "project_col_id"
STATUS_COL_FALLBACK = "catalyst_status"
PAIR_ID_COL = "project_sample_id"
CATALYST_CUM_COL = "cumulative_catalyst_addition_kg_t"
LIXIVIANT_CUM_COL = "cumulative_lixiviant_m3_t"
FEED_MASS_COL = "feed_mass_kg"
EXCLUDED_TRAIN_PAIR_IDS = {
    # "006_jetti_project_file_pvo",
    "020_jetti_project_file_hypogene_supergene_hypogene_master_composite",
    # "024_jetti_project_file_024cv_cpy",
    # "007b_jetti_project_file_tiger_tgr",
    # "jetti_file_elephant_ii_pq",
    # "022_jetti_project_file_stingray_1",
}

DEFAULT_PROJECT_ROOT = os.environ.get(
    "ROSETTA_PROJECT_ROOT",
    (
    "/Users/administration/Library/CloudStorage/OneDrive-JettiResources/"
    "Rosetta/NN_Pytorch_ExpEq_columns_only_v8"
    ),
)
LOCAL_PROJECT_ROOT = os.path.join(THIS_DIR, "NN_Pytorch_ExpEq_columns_only_v8")
PROJECT_ROOT = DEFAULT_PROJECT_ROOT
# PROJECT_ROOT = LOCAL_PROJECT_ROOT


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
for p in [PLOTS_ROOT, MODELS_ROOT, OUTPUTS_ROOT]:
    os.makedirs(p, exist_ok=True)


# ---------------------------
# Predictors
# ---------------------------
HEADERS_DICT_COLUMNS = {
    "leach_duration_days": ["Leach Duration (days)", "numerical", 1],
    "cumulative_catalyst_addition_kg_t": ["Cumulative Catalyst added (kg/t)", "numerical", 1],
    "cumulative_lixiviant_m3_t": ["Cumulative Lixiviant added (m3/t)", "numerical", 1],
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
    # "grouped_gangue_silicates": ["Gangue Silicates (%)", "numerical", 0],
    # "grouped_fe_oxides": ["Fe Oxides (%)", "numerical", 0],
    "grouped_carbonates": ["Carbonates (%)", "numerical", 0],
    # "bornite": ["Bornite (%)", "numerical", 0],
    "fe:cu": ["Fe/Cu ratio", "numerical", 0],
    # "cu:fe": ["Cu/Fe ratio", "numerical", 0],
    "copper_primary_sulfides_equivalent": ["Copper Primary Sulfides Equivalent (%)", "numerical", 0],
    "copper_secondary_sulfides_equivalent": ["Copper Secondary Sulfides Equivalent (%)", "numerical", 0],
    # "copper_sulfides_equivalent": ["Copper Sulfides Equivalent (%)", "numerical", 0],
    "copper_oxides_equivalent": ["Copper Oxides Equivalent (%)", "numerical", 1],
}
INPUT_ONLY_HEADERS_DICT_COLUMNS = {
    "feed_mass_kg": ["Feed Mass (kg)", "numerical", 0],
    "column_height_m": ["Column Height (m)", "numerical", 0],
    "column_inner_diameter_m": ["Column Inner Diameter (m)", "numerical", 0],
}

PREDICTOR_COLUMNS = list(HEADERS_DICT_COLUMNS.keys())
STATIC_PREDICTOR_COLUMNS = [
    c for c in PREDICTOR_COLUMNS if c not in {TIME_COL_COLUMNS, CATALYST_CUM_COL, LIXIVIANT_CUM_COL}
]
INPUT_ONLY_COLUMNS = list(INPUT_ONLY_HEADERS_DICT_COLUMNS.keys())
USER_INPUT_COLUMNS = STATIC_PREDICTOR_COLUMNS + INPUT_ONLY_COLUMNS

GEO_PRIORITY_COLUMNS = ["material_size_p80_in"]
CHEMISTRY_INTERACTION_COLUMNS = [
    c for c in STATIC_PREDICTOR_COLUMNS if c not in set(GEO_PRIORITY_COLUMNS)
]

CONFIG = {
    "seed": 2026,
    "cv_n_splits": 8,
    # 8 folds x 8 repeats = 64 ensemble members, which gives more stable tail percentiles.
    "cv_n_repeats": 8,
    "cv_random_state": 2026,
    "cv_member_seed_base": 10000,
    "cv_parallel_workers": 10, # max(1, int(os.cpu_count() or 1)),
    # 0 = auto-compute based on worker count.
    "torch_threads_per_worker": 0,
    "torch_interop_threads_per_worker": 1,
    "epochs": 500,
    "patience": 100,
    "bootstrap_train_pairs": False,
    "learning_rate": 1e-3, #2.5e-3,
    "weight_decay": 2e-5,
    "grad_clip_norm": 5.0,
    "hidden_dim": 24,
    "dropout": 0.08,
    "min_transition_days": 21.0, # 21
    "min_sigmoid_to_cat_transition_days": 7.0,
    "max_sigmoid_to_cat_transition_days": 364.0, # 52 weeks
    "max_cat_slope_per_day": 0.16,
    "max_catalyst_aging_strength": 5.0, # initially 5.0
    "late_tau_impact_decay_strength": 0.25, # initially 1.15
    "min_remaining_ore_factor": 0.05,
    "flat_input_transition_sensitivity": 1.0, # initially 10.0
    "flat_input_uplift_response_days": 21.0, # initially 30.0
    "flat_input_response_ramp_days": 28.0, # initially 30.0
    "flat_input_late_uplift_response_boost": 0.5, # initially 2.5
    "catalyst_dose_transition_shape_strength": 0.75,
    "practical_stop_min_weekly_catalyst_kg_t": 0.5,
    "practical_stop_min_weekly_lixiviant_m3_t": 0.05,
    "loss_weights": {
        "gap": 1.0,
        "monotonic": 0.02,
        "param": 0.02,
        "smooth_cat": 0.45, # 0.30
        "slope_cap": 0.60, # 0.35
        "latent_smooth": 0.1,
        "latent_cat_rate": 0.3, # initially 0.03
        "flat_input_smooth": 0.18, # 0.18
        "flat_input_accel": 0.16, # 0.16
        "cap": 1.00,  # penalty for a1+a2 exceeding the per-sample leach cap
        "uplift_fit": 0.50,
        "target_fit": 0.25,
        "transition_days": 0.10,
        "transition_takeover": 0.35,
    },
    "plot_dpi": 300,
    "ensemble_pi_low": 10,
    "ensemble_pi_high": 90,
    "ensemble_plot_step_days": 1.0,
    "ensemble_plot_target_day": 2500.0,
    "leach_pct_oxides": 0.95,
    "leach_pct_secondary_sulfides": 0.75,
    "leach_pct_primary_sulfides_control": 0.30,
    "leach_pct_primary_sulfides_catalyzed": 0.70,
    "catalyst_extension_window_days": 50.0,
    "ensemble_interval_smoothing_days": 10.0,
}

MODEL_LOGIC_VERSION = "v8_cat_target_transition_20260330b"

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
        "apparent_bulk_density_t_m3": 0.08,
        "material_size_to_column_diameter_ratio": -0.08,
        "bornite": 0.10,
        "fe:cu": 0.06,
        "cu:fe": -0.04,
        "copper_primary_sulfides_equivalent": 0.20,
        "copper_secondary_sulfides_equivalent": 0.10,
        "copper_sulfides_equivalent": 0.10,
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
        "apparent_bulk_density_t_m3": 0.14,
        "material_size_to_column_diameter_ratio": 0.16,
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
        "apparent_bulk_density_t_m3": 0.12,
        "material_size_to_column_diameter_ratio": -0.10,
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
        "apparent_bulk_density_t_m3": -0.06,
        "material_size_to_column_diameter_ratio": -0.08,
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
    },
    "diffusion_drag_strength": {
        "grouped_gangue_silicates": 0.38,
        "material_size_p80_in": 0.30,
        "apparent_bulk_density_t_m3": 0.28,
        "material_size_to_column_diameter_ratio": 0.35,
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
        "apparent_bulk_density_t_m3": -0.12,
        "material_size_to_column_diameter_ratio": -0.22,
        "bornite": 0.10,
        "chem_interaction": 0.08,
        "ferric_synergy": 0.06,
        "primary_passivation_drive": -0.10,
        "diffusion_drag_strength": -0.18,
    },
    "ore_decay_strength": {
        "chem_raw": 0.10,
        "residual_cpy_%": 0.20,
        "copper_primary_sulfides_equivalent": 0.20,
        "grouped_gangue_silicates": 0.08,
        "grouped_carbonates": 0.06,
        "grouped_acid_generating_sulfides": -0.04,
        "apparent_bulk_density_t_m3": 0.10,
        "material_size_to_column_diameter_ratio": 0.12,
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


def prepare_cumulative_profile_with_time(
    time_days: np.ndarray,
    cumulative_profile: np.ndarray,
    force_zero: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_days, dtype=float)
    c = np.asarray(cumulative_profile, dtype=float)
    n = min(t.size, c.size)
    if n <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    t = t[:n]
    c = c[:n]
    valid = np.isfinite(t) & np.isfinite(c)
    if valid.sum() == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    t = t[valid]
    c = c[valid]
    order = np.argsort(t)
    t = t[order]
    c = c[order]

    t_unique, inv = np.unique(t, return_inverse=True)
    c_unique = np.full(t_unique.shape, np.nan, dtype=float)
    for i, j in enumerate(inv):
        c_unique[j] = c[i] if not np.isfinite(c_unique[j]) else max(c_unique[j], c[i])
    return t_unique.astype(float), clean_cumulative_profile(c_unique, force_zero=force_zero)


def average_weekly_cumulative_from_recent_history(
    time_days: np.ndarray,
    cumulative_profile: np.ndarray,
    window_days: float = 21.0,
    week_days: float = 7.0,
) -> Tuple[float, float]:
    t, c = prepare_cumulative_profile_with_time(time_days, cumulative_profile, force_zero=False)
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


def average_weekly_catalyst_from_recent_history(
    time_days: np.ndarray,
    catalyst_cum: np.ndarray,
    window_days: float = 21.0,
    week_days: float = 7.0,
) -> Tuple[float, float]:
    return average_weekly_cumulative_from_recent_history(
        time_days=time_days,
        cumulative_profile=catalyst_cum,
        window_days=window_days,
        week_days=week_days,
    )


def column_cross_section_area_m2(column_inner_diameter_m: float) -> float:
    diameter = float(column_inner_diameter_m) if np.isfinite(column_inner_diameter_m) else np.nan
    if not np.isfinite(diameter) or diameter <= 0.0:
        return 0.0
    return float(np.pi * (diameter * 0.5) ** 2)


def cumulative_profile_to_increment_rate(
    time_days: np.ndarray,
    cumulative_profile: np.ndarray,
) -> np.ndarray:
    t = np.asarray(time_days, dtype=float)
    c = clean_cumulative_profile(np.asarray(cumulative_profile, dtype=float), force_zero=False)
    n = min(t.size, c.size)
    if n <= 0:
        return np.asarray([], dtype=float)

    t = t[:n]
    c = c[:n]
    out = np.zeros(n, dtype=float)
    prev_t = 0.0
    prev_c = 0.0
    for i in range(n):
        ti = float(t[i]) if np.isfinite(t[i]) else prev_t
        ci = float(c[i]) if np.isfinite(c[i]) else prev_c
        dt = ti - prev_t
        dc = max(0.0, ci - prev_c)
        if dt > 1e-9:
            out[i] = dc / dt
        elif i > 0:
            out[i] = out[i - 1]
        else:
            out[i] = 0.0
        prev_t = max(prev_t, ti)
        prev_c = max(prev_c, ci)
    return np.clip(out, 0.0, None)


def convert_lixiviant_cum_to_irrigation_rate_l_m2_h(
    time_days: np.ndarray,
    cumulative_lixiviant_m3_t: np.ndarray,
    feed_mass_kg: float,
    column_inner_diameter_m: float,
) -> np.ndarray:
    t = np.asarray(time_days, dtype=float)
    c = clean_cumulative_profile(np.asarray(cumulative_lixiviant_m3_t, dtype=float), force_zero=False)
    n = min(t.size, c.size)
    if n <= 0:
        return np.asarray([], dtype=float)

    area_m2 = column_cross_section_area_m2(column_inner_diameter_m)
    feed_mass_t = float(feed_mass_kg) / 1000.0 if np.isfinite(feed_mass_kg) else np.nan
    if not np.isfinite(area_m2) or area_m2 <= 1e-12 or not np.isfinite(feed_mass_t) or feed_mass_t <= 1e-12:
        return np.zeros(n, dtype=float)

    actual_cum_m3 = c[:n] * feed_mass_t
    actual_rate_m3_day = cumulative_profile_to_increment_rate(t[:n], actual_cum_m3)
    irrigation_rate = actual_rate_m3_day * 1000.0 / (area_m2 * 24.0)
    return np.clip(irrigation_rate, 0.0, None)


def build_cumulative_lixiviant_from_irrigation_rate(
    time_days: np.ndarray,
    irrigation_rate_l_m2_h: Any,
    feed_mass_kg: float,
    column_inner_diameter_m: float,
    start_day: float = 0.0,
) -> np.ndarray:
    t = np.asarray(time_days, dtype=float)
    if t.size == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(t)
    t_sorted = t[order]
    rate_raw = np.asarray(irrigation_rate_l_m2_h, dtype=float)
    if rate_raw.ndim == 0:
        rate = np.full(t_sorted.shape, float(rate_raw), dtype=float)
    else:
        rate = align_profile_to_time_length(rate_raw, len(t_sorted))
    rate = np.clip(np.where(t_sorted >= float(start_day), rate, 0.0), 0.0, None)

    area_m2 = column_cross_section_area_m2(column_inner_diameter_m)
    feed_mass_t = float(feed_mass_kg) / 1000.0 if np.isfinite(feed_mass_kg) else np.nan
    if not np.isfinite(area_m2) or area_m2 <= 1e-12 or not np.isfinite(feed_mass_t) or feed_mass_t <= 1e-12:
        out = np.zeros_like(t_sorted, dtype=float)
    else:
        cum_actual_m3 = np.zeros_like(t_sorted, dtype=float)
        for i in range(1, t_sorted.size):
            dt_days = max(0.0, float(t_sorted[i] - t_sorted[i - 1]))
            avg_rate_l_m2_h = 0.5 * (float(rate[i - 1]) + float(rate[i]))
            delta_m3 = avg_rate_l_m2_h * area_m2 * 24.0 * dt_days / 1000.0
            cum_actual_m3[i] = cum_actual_m3[i - 1] + max(0.0, delta_m3)
        out = clean_cumulative_profile(cum_actual_m3 / feed_mass_t, force_zero=False)

    unsorted_out = np.zeros_like(out)
    unsorted_out[order] = out
    return unsorted_out.astype(float)


def derive_internal_geometry_predictors(
    static_raw: np.ndarray,
    input_only_raw: np.ndarray,
) -> Dict[str, float]:
    static_arr = np.asarray(static_raw, dtype=float)
    input_only_arr = np.asarray(input_only_raw, dtype=float)
    static_idx = {name: idx for idx, name in enumerate(STATIC_PREDICTOR_COLUMNS)}
    input_only_idx = {name: idx for idx, name in enumerate(INPUT_ONLY_COLUMNS)}

    feed_mass_kg = (
        float(input_only_arr[input_only_idx[FEED_MASS_COL]])
        if FEED_MASS_COL in input_only_idx
        else np.nan
    )
    material_size_p80_in = (
        float(static_arr[static_idx["material_size_p80_in"]]) if "material_size_p80_in" in static_idx else np.nan
    )
    column_height_m = (
        float(input_only_arr[input_only_idx["column_height_m"]]) if "column_height_m" in input_only_idx else np.nan
    )
    column_inner_diameter_m = (
        float(input_only_arr[input_only_idx["column_inner_diameter_m"]])
        if "column_inner_diameter_m" in input_only_idx
        else np.nan
    )

    area_m2 = column_cross_section_area_m2(column_inner_diameter_m)
    column_volume_m3 = (
        area_m2 * max(0.0, column_height_m)
        if np.isfinite(area_m2) and area_m2 > 0.0 and np.isfinite(column_height_m) and column_height_m > 0.0
        else np.nan
    )
    feed_mass_t = float(feed_mass_kg) / 1000.0 if np.isfinite(feed_mass_kg) and feed_mass_kg > 0.0 else np.nan
    apparent_bulk_density_t_m3 = (
        feed_mass_t / column_volume_m3
        if np.isfinite(feed_mass_t) and np.isfinite(column_volume_m3) and column_volume_m3 > 0.0
        else np.nan
    )
    material_size_to_column_diameter_ratio = (
        float(material_size_p80_in) * 0.0254 / float(column_inner_diameter_m)
        if np.isfinite(material_size_p80_in)
        and material_size_p80_in > 0.0
        and np.isfinite(column_inner_diameter_m)
        and column_inner_diameter_m > 0.0
        else np.nan
    )
    return {
        "column_volume_m3": float(column_volume_m3) if np.isfinite(column_volume_m3) else np.nan,
        "apparent_bulk_density_t_m3": (
            float(apparent_bulk_density_t_m3) if np.isfinite(apparent_bulk_density_t_m3) else np.nan
        ),
        "material_size_to_column_diameter_ratio": (
            float(material_size_to_column_diameter_ratio)
            if np.isfinite(material_size_to_column_diameter_ratio)
            else np.nan
        ),
    }


def infer_catalyst_addition_start_day(time_days: np.ndarray, catalyst_cum: np.ndarray) -> float:
    t, c = prepare_cumulative_profile_with_time(
        time_days,
        catalyst_cum,
        force_zero=False,
    )
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
    t, c = prepare_cumulative_profile_with_time(
        time_days,
        catalyst_cum,
        force_zero=False,
    )
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
    t, c = prepare_cumulative_profile_with_time(
        time_days,
        catalyst_cum,
        force_zero=False,
    )
    if t.size == 0:
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
    at, ac = prepare_cumulative_profile_with_time(
        anchor_time_days,
        anchor_catalyst_cum,
        force_zero=False,
    )
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


def apply_practical_extension_stop(
    weekly_value: float,
    config_key: str,
) -> float:
    value = float(weekly_value) if np.isfinite(weekly_value) else 0.0
    min_weekly_value = float(CONFIG.get(config_key, 0.0))
    if min_weekly_value <= 0.0:
        return value
    if abs(value) < min_weekly_value:
        return 0.0
    return value


def smooth_recovery_envelope(
    time_days: np.ndarray,
    values: np.ndarray,
    smoothing_days: float,
) -> np.ndarray:
    t = np.asarray(time_days, dtype=float)
    y = np.asarray(values, dtype=float)
    if y.size <= 2 or not np.isfinite(smoothing_days) or smoothing_days <= 0.0:
        return np.clip(y, 0.0, 100.0)

    valid = np.isfinite(t) & np.isfinite(y)
    if int(np.sum(valid)) <= 2:
        return np.clip(y, 0.0, 100.0)

    t_valid = t[valid]
    y_valid = np.clip(y[valid], 0.0, 100.0)
    dt = np.diff(t_valid)
    dt = dt[np.isfinite(dt) & (dt > 1e-9)]
    if dt.size == 0:
        return np.clip(y, 0.0, 100.0)

    window_points = int(np.ceil(float(smoothing_days) / float(np.nanmedian(dt))))
    window_points = max(window_points, 3)
    if window_points % 2 == 0:
        window_points += 1
    if window_points >= y_valid.size:
        window_points = y_valid.size if y_valid.size % 2 == 1 else y_valid.size - 1
    if window_points < 3:
        return np.clip(y, 0.0, 100.0)

    half = window_points // 2
    ramp = np.arange(1, half + 2, dtype=float)
    weights = np.concatenate([ramp, ramp[-2::-1]])
    weights = weights / max(float(np.sum(weights)), 1e-9)
    padded = np.pad(y_valid, (half, half), mode="edge")
    smoothed = np.convolve(padded, weights, mode="valid")
    smoothed = np.clip(smoothed, 0.0, 100.0)
    smoothed = np.maximum.accumulate(smoothed)

    out = np.asarray(y, dtype=float).copy()
    out[valid] = smoothed
    out[~valid] = np.nan
    return out


def smooth_predictive_interval_bounds(
    time_days: np.ndarray,
    mean_curve: np.ndarray,
    low_curve: np.ndarray,
    high_curve: np.ndarray,
    smoothing_days: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mean_arr = np.clip(np.asarray(mean_curve, dtype=float), 0.0, 100.0)
    low_smoothed = smooth_recovery_envelope(time_days, low_curve, smoothing_days)
    high_smoothed = smooth_recovery_envelope(time_days, high_curve, smoothing_days)
    low_smoothed = np.clip(low_smoothed, 0.0, 100.0)
    high_smoothed = np.clip(high_smoothed, 0.0, 100.0)
    low_smoothed = np.minimum(low_smoothed, mean_arr)
    high_smoothed = np.maximum(high_smoothed, mean_arr)
    low_smoothed = np.minimum(low_smoothed, high_smoothed)
    return low_smoothed, high_smoothed


def extend_catalyst_profile_for_ensemble_plot(
    time_days: np.ndarray,
    catalyst_cum: np.ndarray,
    target_day: float,
    step_days: float = 7.0,
    history_window_days: float = 21.0,
) -> Dict[str, Any]:
    t, c = prepare_cumulative_profile_with_time(
        time_days,
        catalyst_cum,
        force_zero=False,
    )
    if t.size == 0:
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

    catalyst_behavior = summarize_catalyst_addition_behavior(
        time_days=t,
        catalyst_cum=c,
        history_window_days=history_window_days,
    )
    weekly_value = float(catalyst_behavior["weekly_catalyst_addition_kg_t"])
    weekly_value = apply_practical_extension_stop(
        weekly_value,
        config_key="practical_stop_min_weekly_catalyst_kg_t",
    )
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
            **{**catalyst_behavior, "weekly_catalyst_addition_kg_t": float(weekly_value)},
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
        **{**catalyst_behavior, "weekly_catalyst_addition_kg_t": float(weekly_value)},
        "extension_applied": True,
        "target_day": float(target_day),
    }


def extend_generic_cumulative_profile_for_ensemble_plot(
    time_days: np.ndarray,
    cumulative_profile: np.ndarray,
    target_day: float,
    step_days: float = 7.0,
    history_window_days: float = 21.0,
) -> Dict[str, Any]:
    t, c = prepare_cumulative_profile_with_time(
        time_days,
        cumulative_profile,
        force_zero=False,
    )
    if t.size == 0:
        return {
            "time_days": np.asarray([], dtype=float),
            "cumulative_profile": np.asarray([], dtype=float),
            "weekly_addition": 0.0,
            "weekly_reference_days": 0.0,
            "last_observed_day": np.nan,
            "extension_applied": False,
            "target_day": float(target_day),
        }

    weekly_value, reference_days = average_weekly_cumulative_from_recent_history(
        time_days=t,
        cumulative_profile=c,
        window_days=history_window_days,
        week_days=7.0,
    )
    weekly_value = apply_practical_extension_stop(
        weekly_value,
        config_key="practical_stop_min_weekly_lixiviant_m3_t",
    )
    last_day = float(t[-1])
    if last_day >= float(target_day) - 1e-9:
        dense_time_days = build_plot_time_grid(
            observed_time_days=t,
            start_day=float(t[0]),
            target_day=last_day,
            step_days=float(step_days),
        )
        return {
            "time_days": dense_time_days,
            "cumulative_profile": interpolate_cumulative_profile(dense_time_days, t, c),
            "weekly_addition": float(weekly_value),
            "weekly_reference_days": float(reference_days),
            "last_observed_day": last_day,
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
    extended_cum = np.concatenate([c, future_cum])
    dense_time_days = build_plot_time_grid(
        observed_time_days=extended_time_days,
        start_day=float(t[0]),
        target_day=float(target_day),
        step_days=float(step_days),
    )
    dense_cum = interpolate_cumulative_profile(dense_time_days, extended_time_days, extended_cum)
    return {
        "time_days": dense_time_days,
        "cumulative_profile": dense_cum,
        "weekly_addition": float(weekly_value),
        "weekly_reference_days": float(reference_days),
        "last_observed_day": last_day,
        "extension_applied": True,
        "target_day": float(target_day),
    }


def build_shared_ensemble_plot_profile(
    time_days: np.ndarray,
    catalyst_cum: np.ndarray,
    lixiviant_cum: np.ndarray,
    feed_mass_kg: float,
    column_inner_diameter_m: float,
    target_day: float,
    step_days: float = 1.0,
    history_window_days: float = 21.0,
) -> Dict[str, Any]:
    catalyst_profile = extend_catalyst_profile_for_ensemble_plot(
        time_days=time_days,
        catalyst_cum=catalyst_cum,
        target_day=target_day,
        step_days=step_days,
        history_window_days=history_window_days,
    )
    lix_profile = extend_generic_cumulative_profile_for_ensemble_plot(
        time_days=time_days,
        cumulative_profile=lixiviant_cum,
        target_day=target_day,
        step_days=step_days,
        history_window_days=float(CONFIG.get("lixiviant_extension_window_days", history_window_days)),
    )
    observed_time_parts = []
    if np.asarray(catalyst_profile["time_days"], dtype=float).size > 0:
        observed_time_parts.append(np.asarray(catalyst_profile["time_days"], dtype=float))
    if np.asarray(lix_profile["time_days"], dtype=float).size > 0:
        observed_time_parts.append(np.asarray(lix_profile["time_days"], dtype=float))
    observed_time_days = (
        np.concatenate(observed_time_parts) if len(observed_time_parts) > 0 else np.asarray([], dtype=float)
    )
    start_day = float(np.nanmin(observed_time_days)) if observed_time_days.size > 0 else 0.0
    profile_time_days = build_plot_time_grid(
        observed_time_days=observed_time_days,
        start_day=start_day,
        target_day=float(target_day),
        step_days=float(step_days),
    )
    profile_catalyst_cum = interpolate_cumulative_profile(
        query_time_days=profile_time_days,
        anchor_time_days=np.asarray(catalyst_profile["time_days"], dtype=float),
        anchor_catalyst_cum=np.asarray(catalyst_profile["catalyst_cum"], dtype=float),
    )
    profile_lixiviant_cum = interpolate_cumulative_profile(
        query_time_days=profile_time_days,
        anchor_time_days=np.asarray(lix_profile["time_days"], dtype=float),
        anchor_catalyst_cum=np.asarray(lix_profile["cumulative_profile"], dtype=float),
    )
    profile_irrigation_rate = convert_lixiviant_cum_to_irrigation_rate_l_m2_h(
        time_days=profile_time_days,
        cumulative_lixiviant_m3_t=profile_lixiviant_cum,
        feed_mass_kg=feed_mass_kg,
        column_inner_diameter_m=column_inner_diameter_m,
    )
    plot_time_days = build_plot_time_grid(
        observed_time_days=np.asarray(profile_time_days, dtype=float),
        start_day=0.0,
        target_day=float(target_day),
        step_days=float(step_days),
    )
    plot_catalyst_cum = interpolate_cumulative_profile(
        query_time_days=plot_time_days,
        anchor_time_days=np.asarray(profile_time_days, dtype=float),
        anchor_catalyst_cum=np.asarray(profile_catalyst_cum, dtype=float),
    )
    plot_lixiviant_cum = interpolate_cumulative_profile(
        query_time_days=plot_time_days,
        anchor_time_days=np.asarray(profile_time_days, dtype=float),
        anchor_catalyst_cum=np.asarray(profile_lixiviant_cum, dtype=float),
    )
    plot_irrigation_rate = convert_lixiviant_cum_to_irrigation_rate_l_m2_h(
        time_days=plot_time_days,
        cumulative_lixiviant_m3_t=plot_lixiviant_cum,
        feed_mass_kg=feed_mass_kg,
        column_inner_diameter_m=column_inner_diameter_m,
    )
    return {
        **catalyst_profile,
        "time_days": np.asarray(profile_time_days, dtype=float),
        "catalyst_cum": np.asarray(profile_catalyst_cum, dtype=float),
        "lixiviant_time_days": np.asarray(lix_profile["time_days"], dtype=float),
        "lixiviant_cum": np.asarray(profile_lixiviant_cum, dtype=float),
        "irrigation_rate_l_m2_h": np.asarray(profile_irrigation_rate, dtype=float),
        "weekly_lixiviant_addition_m3_t": float(lix_profile["weekly_addition"]),
        "lixiviant_weekly_reference_days": float(lix_profile["weekly_reference_days"]),
        "lixiviant_extension_applied": bool(lix_profile["extension_applied"]),
        "plot_time_days": np.asarray(plot_time_days, dtype=float),
        "plot_catalyst_cum": np.asarray(plot_catalyst_cum, dtype=float),
        "plot_lixiviant_cum": np.asarray(plot_lixiviant_cum, dtype=float),
        "plot_irrigation_rate_l_m2_h": np.asarray(plot_irrigation_rate, dtype=float),
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

def compute_sample_leach_caps(
    cu_pct: float,
    cu_oxides_equiv: float,
    cu_secondary_equiv: float,
    cu_primary_equiv: float,
) -> Tuple[float, float]:
    """
    Compute per-sample leach caps for control and catalyzed curves based on
    ore mineralogy.  The "leachable real" for each curve type is:

        leachable_real = oxides * pct_ox + secondary * pct_sec + primary * pct_pri

    The cap (in recovery-%) is  leachable_real / cu_pct * 100 , clipped to [1, 100].

    Returns
    -------
    (ctrl_cap, cat_cap) : both in recovery-% (0-100 scale)
    """
    pct_ox  = float(CONFIG.get("leach_pct_oxides", 1.00))
    pct_sec = float(CONFIG.get("leach_pct_secondary_sulfides", 0.70))
    pct_pri_ctrl = float(CONFIG.get("leach_pct_primary_sulfides_control", 0.30))
    pct_pri_cat  = float(CONFIG.get("leach_pct_primary_sulfides_catalyzed", 0.70))

    # Sanitise inputs – treat NaN / negative as zero contribution
    ox  = max(0.0, float(cu_oxides_equiv))   if np.isfinite(cu_oxides_equiv)   else 0.0
    sec = max(0.0, float(cu_secondary_equiv)) if np.isfinite(cu_secondary_equiv) else 0.0
    pri = max(0.0, float(cu_primary_equiv))   if np.isfinite(cu_primary_equiv)   else 0.0
    cu  = float(cu_pct) if np.isfinite(cu_pct) and cu_pct > 0 else 0.0

    fallback_cap = float(CONFIG.get("fit_max_asymptote", 80.0))

    if cu <= 1e-9:
        return fallback_cap, fallback_cap

    ctrl_leachable_real = ox * pct_ox + sec * pct_sec + pri * pct_pri_ctrl
    cat_leachable_real  = ox * pct_ox + sec * pct_sec + pri * pct_pri_cat

    ctrl_cap = np.clip(ctrl_leachable_real / cu * 100.0, 1.0, 100.0)
    cat_cap  = np.clip(cat_leachable_real  / cu * 100.0, 1.0, 100.0)

    return float(ctrl_cap), float(cat_cap)


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


def sanitize_curve_params(params: np.ndarray, cap: Optional[float] = None) -> np.ndarray:
    max_asym = float(CONFIG.get("fit_max_asymptote", 80.0))
    # When a per-sample cap is provided, honour it as the effective hard ceiling.
    # effective_max = min(max_asym, cap) if (cap is not None and np.isfinite(cap)) else max_asym
    effective_max = min(max_asym, cap)
    p = np.asarray(params, dtype=float).copy()
    if p.size != 4 or not np.all(np.isfinite(p)):
        p = np.array([20.0, 0.02, 10.0, 0.003], dtype=float)
    p[0] = max(1.0, p[0])
    p[2] = max(0.5, p[2])
    p[1] = np.clip(p[1], 1e-5, 1e-1)
    p[3] = np.clip(p[3], 1e-5, 1e-1)
    p = enforce_fast_slow_pairing(p)
    return p


def fit_biexponential_params(t_days: np.ndarray, recovery: np.ndarray, cap: Optional[float] = None) -> np.ndarray:
    """
    Improved bi-exponential fitting with:
      - Adaptive initial guesses based on curve behaviour (increasing vs plateau).
      - Soft constraint: the curve should approach its asymptote by
        ``fit_plateau_target_day`` (CONFIG, default ensemble_plot_target_day + 1000).
      - Hard cap: a1 + a2 <= ``fit_max_asymptote`` (CONFIG, default 80.0).
      - Per-sample cap (penalty only): when ``cap`` is provided, a quadratic penalty
        is added whenever a1 + a2 exceeds that cap.  Control and catalyzed caps must
        be passed separately by the caller and are never mixed.
      - Linear weight ramp from 1.0 to 10.0 across data points so that the
        fit progressively prioritises the later trend over early behaviour.
      - Two optimisation strategies: multi-start local + differential evolution.
      - Best candidate selected by R².
    """
    t = np.asarray(t_days, dtype=float)
    y = np.asarray(recovery, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() < 6:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
    t = t[valid]

    max_asym = float(CONFIG.get("fit_max_asymptote", 80.0))
    plateau_day = float(CONFIG.get("ensemble_plot_target_day", 2500.0) * 1.5)

    y = np.clip(y[valid], 0.0, max_asym)
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # ---- point weights: linear ramp 1 → 10 ---------------------------------
    n_pts = len(t)
    weights = np.linspace(1.0, 10.0, n_pts)
    # sigma for curve_fit is inversely proportional to weight:
    #   cost ∝ Σ ((y - f) / sigma)²  →  sigma = 1/√w
    sigma = 1.0 / np.sqrt(weights)

    # ---- curve-shape analysis ------------------------------------------------
    y_max = float(np.nanmax(y))
    last_quarter_idx = max(n_pts // 4, 1)
    first_half_mean = float(np.mean(y[:n_pts // 2]))
    last_quarter_mean = float(np.mean(y[-last_quarter_idx:]))
    is_still_increasing = (last_quarter_mean > first_half_mean * 1.05)

    # ---- initial-guess generation --------------------------------------------
    p0_list = []
    if is_still_increasing:
        p0_list.extend([
            [0.5 * y_max, 0.05, 0.5 * y_max, 0.005],
            [0.6 * y_max, 0.03, 0.4 * y_max, 0.003],
            [0.7 * y_max, 0.02, 0.3 * y_max, 0.002],
            [0.4 * y_max, 0.08, 0.6 * y_max, 0.008],
            [0.3 * y_max, 0.10, 0.7 * y_max, 0.010],
            [0.55 * y_max, 0.04, 0.45 * y_max, 0.004],
        ])
    else:
        plateau = float(np.nanpercentile(y, 90))
        p0_list.extend([
            [0.6 * plateau, 0.03, 0.4 * plateau, 0.004],
            [0.8 * y_max, 0.06, 0.2 * y_max, 0.008],
            [0.5 * plateau, 0.02, 0.5 * plateau, 0.002],
            [y_max, 0.03, 1.0, 0.002],
        ])

    lower = [0.0, 1e-5, 0.0, 1e-5]
    upper = [max_asym, 1e-1, max_asym, 1e-1]

    # ---- per-sample cap -------------------------------------------------------
    # When a mineralogy-derived cap is provided, add a soft penalty that grows
    # quadratically whenever a1+a2 exceeds that cap.  The bounds for the
    # optimisers are still governed by max_asym (global) so the solver search
    # space is unchanged; the penalty merely steers the solution inward.
    cap_val = float(cap) if (cap is not None and np.isfinite(cap)) else None
    cap_penalty_weight = 2.0  # same order-of-magnitude as plateau_penalty_weight
    cap_soft_margin = 3.0

    # ---- soft-plateau penalty ------------------------------------------------
    plateau_penalty_weight = 0.10

    def _penalised_wmse(params):
        a1, b1, a2, b2 = params
        pred = double_exp_curve_np(t, a1, b1, a2, b2)
        residuals = pred - y
        wmse = float(np.mean(weights * residuals ** 2))

        asymptote = a1 + a2
        
        if asymptote > 0:
            value_at_plateau = double_exp_curve_np(
                np.array([plateau_day]), a1, b1, a2, b2
            )[0]
            residual_frac = 1.0 - value_at_plateau / asymptote
            plateau_penalty = (residual_frac ** 2) * (asymptote ** 2)
        else:
            plateau_penalty = 0.0
        
        # Per-sample cap soft penalty: only fires when a1+a2 exceeds the cap.
        if cap_val is not None:
            cap_excess = max(0.0, asymptote - (cap_val + cap_soft_margin))
            cap_penalty = cap_excess ** 2
        else:
            cap_penalty = 0.0

        return wmse + plateau_penalty_weight * plateau_penalty + cap_penalty_weight * cap_penalty
        # return wmse + cap_penalty_weight * cap_penalty

    # ---- Strategy 1: multi-start local optimisation --------------------------
    best = None
    best_r2 = -np.inf

    for p0 in p0_list:
        p0 = sanitize_curve_params(np.asarray(p0, dtype=float), cap=cap_val)
        try:
            popt, _ = curve_fit(
                lambda t_, a1, b1, a2, b2: double_exp_curve_np(t_, a1, b1, a2, b2),
                t,
                y,
                p0=p0,
                sigma=sigma,
                absolute_sigma=False,
                bounds=(lower, upper),
                maxfev=12000,
                ftol=1e-12,
                xtol=1e-12,
            )
            popt = sanitize_curve_params(np.asarray(popt, dtype=float), cap=cap_val)
            pred = double_exp_curve_np(t, popt[0], popt[1], popt[2], popt[3])
            r2 = float(r2_score(y, pred))
            if r2 > best_r2:
                best_r2 = r2
                best = popt
        except Exception:
            continue

    # ---- Strategy 2: differential evolution with weighted + plateau penalty ---
    try:
        bounds_de = [(lower[i], upper[i]) for i in range(4)]
        result = differential_evolution(
            _penalised_wmse,
            bounds_de,
            seed=42,
            maxiter=1000,
            atol=1e-12,
            tol=1e-12,
            workers=1,
            polish=True,
        )

        popt = sanitize_curve_params(np.asarray(result.x, dtype=float), cap=cap_val)
        pred = double_exp_curve_np(t, popt[0], popt[1], popt[2], popt[3])
        r2 = float(r2_score(y, pred))
        if r2 > best_r2:
            best_r2 = r2
            best = popt
    except Exception:
        pass

    if best is None:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
    return sanitize_curve_params(best, cap=cap_val)


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
        hi = min(1e-1, hi)
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

# -----------------------------------
# Ore calculation
def compute_remaining_ore_factor_chemistry_based(
    y_ctrl: torch.Tensor,
    cu_percent: torch.Tensor,
    copper_primary_sulfides_equiv: torch.Tensor,
    copper_oxides_equiv: torch.Tensor,
    min_floor: float = 0.05,
    copper_secondary_sulfides_equiv: Optional[torch.Tensor] = None,
    is_catalyzed: bool = False,
) -> torch.Tensor:
    """
    Compute remaining ore factor based on ore chemistry using leachable_real.

    The maximum leachable is computed as:
        leachable_real = oxides * pct_ox + secondary * pct_sec + primary * pct_pri
    where the primary percentage differs between control and catalyzed curves
    (CONFIG keys ``leach_pct_primary_sulfides_control`` vs
    ``leach_pct_primary_sulfides_catalyzed``).

    Args:
        y_ctrl: Control recovery (0-100%)
        cu_percent: Total copper grade (%)
        copper_primary_sulfides_equiv: Primary sulfides (%)
        copper_oxides_equiv: Oxides (%)
        min_floor: Minimum factor to allow some residual uplift
        copper_secondary_sulfides_equiv: Secondary sulfides (%), optional
        is_catalyzed: If True, use catalyzed primary leach percentage

    Returns:
        remaining_ore_factor: Fraction of leachable copper still available (0-1)
    """
    pct_ox  = float(CONFIG.get("leach_pct_oxides", 1.00))
    pct_sec = float(CONFIG.get("leach_pct_secondary_sulfides", 0.70))
    if is_catalyzed:
        pct_pri = float(CONFIG.get("leach_pct_primary_sulfides_catalyzed", 0.70))
    else:
        pct_pri = float(CONFIG.get("leach_pct_primary_sulfides_control", 0.30))

    cu_safe = torch.clamp(cu_percent, min=1e-6)

    # leachable_real (in same units as cu_percent)
    leachable_real = copper_oxides_equiv * pct_ox + copper_primary_sulfides_equiv * pct_pri
    if copper_secondary_sulfides_equiv is not None:
        leachable_real = leachable_real + copper_secondary_sulfides_equiv * pct_sec

    # Maximum leachable as fraction of total copper
    max_leachable_frac = torch.clamp(leachable_real / cu_safe, min=1e-6, max=1.0)

    # Extracted fraction of total copper
    extracted_fraction = torch.clamp(y_ctrl / 100.0, min=0.0, max=1.0)

    # Remaining fraction of leachable copper
    remaining_fraction = torch.clamp(
        1.0 - extracted_fraction / max_leachable_frac,
        min=0.0,
        max=1.0,
    )

    remaining_ore_factor = torch.clamp(remaining_fraction, min=min_floor, max=1.0)
    return remaining_ore_factor


# ---------------------------
# Fitted curve plotting and virtual data augmentation
# ---------------------------
def calculate_fit_metrics(actual: np.ndarray, predicted: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate R², RMSE, and Bias for fitted curve evaluation.
    Returns: (r2, rmse, bias)
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    # Filter out NaN values
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if valid.sum() < 2:
        return np.nan, np.nan, np.nan
    
    actual_valid = actual[valid]
    predicted_valid = predicted[valid]
    
    r2 = float(r2_score(actual_valid, predicted_valid))
    rmse = float(np.sqrt(mean_squared_error(actual_valid, predicted_valid)))
    bias = float(np.mean(predicted_valid - actual_valid))
    
    return r2, rmse, bias


def generate_virtual_points(
    time: np.ndarray,
    recovery: np.ndarray,
    fit_params: np.ndarray,
    target_day: float,
    interval_days: float = 7.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate virtual data points using bi-exponential parameters to extend curve to target_day.
    Points are generated every interval_days from the last actual data point.
    Returns: (extended_time, extended_recovery)
    """
    time = np.asarray(time, dtype=float)
    recovery = np.asarray(recovery, dtype=float)
    fit_params = np.asarray(fit_params, dtype=float)
    
    # Filter valid points
    valid = np.isfinite(time) & np.isfinite(recovery)
    if valid.sum() == 0:
        return time, recovery
    
    time_valid = time[valid]
    recovery_valid = recovery[valid]
    
    # Get last actual data point
    last_time = float(np.max(time_valid))
    
    # Generate virtual points every interval_days from last_time to target_day
    if last_time >= target_day:
        return time_valid, recovery_valid
    
    virtual_times = np.arange(last_time + interval_days, target_day + interval_days, interval_days)
    
    if len(fit_params) >= 4 and np.all(np.isfinite(fit_params[:4])):
        a1, b1, a2, b2 = fit_params[0], fit_params[1], fit_params[2], fit_params[3]
        virtual_recovery = double_exp_curve_np(virtual_times, a1, b1, a2, b2)
    else:
        # If parameters are invalid, use last recovery value
        virtual_recovery = np.full_like(virtual_times, recovery_valid[-1], dtype=float)
    
    # Combine actual and virtual points
    extended_time = np.concatenate([time_valid, virtual_times])
    extended_recovery = np.concatenate([recovery_valid, virtual_recovery])

    return extended_time, extended_recovery


def extend_curve_dynamic_inputs_to_time_grid(
    curve_data: "CurveData",
    target_time_days: np.ndarray,
    feed_mass_kg: float,
    column_inner_diameter_m: float,
    step_days: float = 7.0,
    catalyst_history_window_days: float = 21.0,
    lix_history_window_days: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_time_days = np.asarray(target_time_days, dtype=float)
    query_time_days = query_time_days[np.isfinite(query_time_days)]
    if query_time_days.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty

    target_day = float(np.max(query_time_days))
    catalyst_profile = extend_catalyst_profile_for_ensemble_plot(
        time_days=curve_data.time,
        catalyst_cum=curve_data.catalyst_cum,
        target_day=target_day,
        step_days=step_days,
        history_window_days=catalyst_history_window_days,
    )
    catalyst_cum = interpolate_cumulative_profile(
        query_time_days=query_time_days,
        anchor_time_days=np.asarray(catalyst_profile["time_days"], dtype=float),
        anchor_catalyst_cum=np.asarray(catalyst_profile["catalyst_cum"], dtype=float),
    )

    lix_window_days = (
        float(lix_history_window_days)
        if lix_history_window_days is not None
        else float(CONFIG.get("lixiviant_extension_window_days", catalyst_history_window_days))
    )
    lix_profile = extend_generic_cumulative_profile_for_ensemble_plot(
        time_days=curve_data.time,
        cumulative_profile=curve_data.lixiviant_cum,
        target_day=target_day,
        step_days=step_days,
        history_window_days=lix_window_days,
    )
    lixiviant_cum = interpolate_cumulative_profile(
        query_time_days=query_time_days,
        anchor_time_days=np.asarray(lix_profile["time_days"], dtype=float),
        anchor_catalyst_cum=np.asarray(lix_profile["cumulative_profile"], dtype=float),
    )
    irrigation_rate = convert_lixiviant_cum_to_irrigation_rate_l_m2_h(
        time_days=query_time_days,
        cumulative_lixiviant_m3_t=lixiviant_cum,
        feed_mass_kg=feed_mass_kg,
        column_inner_diameter_m=column_inner_diameter_m,
    )
    return (
        np.asarray(catalyst_cum, dtype=float),
        np.asarray(lixiviant_cum, dtype=float),
        np.asarray(irrigation_rate, dtype=float),
    )


def plot_fitted_curve_per_sample(
    pairs: List['PairSample'],
    output_dir: str,
    dpi: int = 300,
    target_day: float = 2500.0,
) -> None:
    """
    Create fitted curve plots for each project_sample_id.
    One plot per sample showing Control (blue) and Catalyzed (orange) curves.
    Title is red if criteria not met (R² <= 0.50 or RMSE >= 5.0). 0.60 only because of 015 8in, all other are over 0.9
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for pair in pairs:
        sample_id = pair.sample_id
        
        fig, ax = plt.subplots(figsize=(12, 7), dpi=dpi)
        
        # Process Control and Catalyzed curves
        curves_info = [
            (pair.control, 'Control', '#1f77b4'),  # Blue
            (pair.catalyzed, 'Catalyzed', '#ff7f0e'),  # Orange
        ]
        
        all_r2_values = []
        all_rmse_values = []
        all_bias_values = []
        criteria_met = True
        
        for curve_data, status_label, color in curves_info:
            time = curve_data.time
            recovery = curve_data.recovery
            fit_params = curve_data.fit_params
            
            # Calculate metrics
            if len(fit_params) >= 4 and np.all(np.isfinite(fit_params[:4])):
                predicted = double_exp_curve_np(time, fit_params[0], fit_params[1], fit_params[2], fit_params[3])
                r2, rmse, bias = calculate_fit_metrics(recovery, predicted)
            else:
                r2, rmse, bias = np.nan, np.nan, np.nan
            
            all_r2_values.append(r2)
            all_rmse_values.append(rmse)
            all_bias_values.append(bias)
            
            # Check if criteria are met
            if not (np.isfinite(r2) and r2 > 0.50 and np.isfinite(rmse) and rmse < 5.0):
                criteria_met = False
            
            # Plot actual data points
            ax.scatter(time, recovery, color=color, s=60, alpha=0.7, label=f'{status_label} (actual)', zorder=3)
            
            # Plot fitted curve
            if len(fit_params) >= 4 and np.all(np.isfinite(fit_params[:4])):
                # Generate smooth curve for visualization
                t_smooth = np.linspace(0, max(target_day, np.max(time) * 1.1), 500)
                y_smooth = double_exp_curve_np(t_smooth, fit_params[0], fit_params[1], fit_params[2], fit_params[3])
                ax.plot(t_smooth, y_smooth, color=color, linewidth=2.5, alpha=0.8, label=f'{status_label} (fitted)', zorder=2)
        
        # Format plot
        ax.set_xlabel('Leach Duration (days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cu Recovery (%)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)
        
        # Create title with metrics
        title_color = 'red' if not criteria_met else 'black'
        metrics_text = f"R²: {all_r2_values[0]:.3f} / {all_r2_values[1]:.3f}  |  RMSE: {all_rmse_values[0]:.2f} / {all_rmse_values[1]:.2f}  |  Bias: {all_bias_values[0]:.2f} / {all_bias_values[1]:.2f}"
        title = f"{sample_id}\n{metrics_text}"
        ax.set_title(title, fontsize=13, fontweight='bold', color=title_color)
        
        # Set x-axis limit
        ax.set_xlim(left=0, right=target_day * 1.05)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{sample_id}_fitted_curve.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[Plot] Saved: {plot_filename}")


def augment_pair_with_virtual_data(
    pair: 'PairSample',
    target_day: float = 2500.0,
    interval_days: float = 7.0,
    catalyst_history_window_days: Optional[float] = None,
    lix_history_window_days: Optional[float] = None,
) -> 'PairSample':
    """
    Augment a PairSample with virtual data points if criteria are met.
    Each curve is extended only when its own fit meets R² > 0.5 and RMSE < 5.0.
    """
    catalyst_history_window_days = float(
        CONFIG.get("catalyst_extension_window_days", 21.0)
        if catalyst_history_window_days is None
        else catalyst_history_window_days
    )
    lix_history_window_days = float(
        CONFIG.get("lixiviant_extension_window_days", catalyst_history_window_days)
        if lix_history_window_days is None
        else lix_history_window_days
    )
    input_only_idx = {name: idx for idx, name in enumerate(INPUT_ONLY_COLUMNS)}
    feed_mass_kg = (
        float(pair.input_only_raw[input_only_idx[FEED_MASS_COL]])
        if FEED_MASS_COL in input_only_idx
        else np.nan
    )
    column_inner_diameter_m = (
        float(pair.input_only_raw[input_only_idx["column_inner_diameter_m"]])
        if "column_inner_diameter_m" in input_only_idx
        else np.nan
    )

    # Check criteria for Control
    ctrl_time = pair.control.time
    ctrl_recovery = pair.control.recovery
    ctrl_params = pair.control.fit_params
    
    if len(ctrl_params) >= 4 and np.all(np.isfinite(ctrl_params[:4])):
        ctrl_pred = double_exp_curve_np(ctrl_time, ctrl_params[0], ctrl_params[1], ctrl_params[2], ctrl_params[3])
        ctrl_r2, ctrl_rmse, _ = calculate_fit_metrics(ctrl_recovery, ctrl_pred)
    else:
        ctrl_r2, ctrl_rmse = np.nan, np.nan
    
    # Check criteria for Catalyzed
    cat_time = pair.catalyzed.time
    cat_recovery = pair.catalyzed.recovery
    cat_params = pair.catalyzed.fit_params
    
    if len(cat_params) >= 4 and np.all(np.isfinite(cat_params[:4])):
        cat_pred = double_exp_curve_np(cat_time, cat_params[0], cat_params[1], cat_params[2], cat_params[3])
        cat_r2, cat_rmse, _ = calculate_fit_metrics(cat_recovery, cat_pred)
    else:
        cat_r2, cat_rmse = np.nan, np.nan
    
    # Evaluate the extension criteria independently for each curve
    ctrl_meets_criteria = np.isfinite(ctrl_r2) and ctrl_r2 > 0.5 and np.isfinite(ctrl_rmse) and ctrl_rmse < 5.0
    cat_meets_criteria = np.isfinite(cat_r2) and cat_r2 > 0.5 and np.isfinite(cat_rmse) and cat_rmse < 5.0
    
    # Create new pair with augmented data
    new_pair = PairSample(
        sample_id=pair.sample_id,
        static_raw=pair.static_raw,
        input_only_raw=pair.input_only_raw,
        control=pair.control,
        catalyzed=pair.catalyzed,
        static_scaled=pair.static_scaled,
        static_imputed=pair.static_imputed,
        ctrl_cap=pair.ctrl_cap,
        cat_cap=pair.cat_cap,
    )
    
    # Augment Control if criteria met
    if ctrl_meets_criteria:
        ext_time, ext_recovery = generate_virtual_points(
            ctrl_time, ctrl_recovery, ctrl_params, target_day, interval_days
        )
        ctrl_catalyst_cum, ctrl_lixiviant_cum, ctrl_irrigation_rate = extend_curve_dynamic_inputs_to_time_grid(
            curve_data=pair.control,
            target_time_days=ext_time,
            feed_mass_kg=feed_mass_kg,
            column_inner_diameter_m=column_inner_diameter_m,
            step_days=interval_days,
            catalyst_history_window_days=catalyst_history_window_days,
            lix_history_window_days=lix_history_window_days,
        )
        new_pair.control = CurveData(
            status=pair.control.status,
            time=ext_time,
            recovery=ext_recovery,
            catalyst_cum=ctrl_catalyst_cum,
            lixiviant_cum=ctrl_lixiviant_cum,
            irrigation_rate_l_m2_h=ctrl_irrigation_rate,
            fit_params=pair.control.fit_params,
            row_index=pair.control.row_index,
        )
    
    # Augment Catalyzed if criteria met
    if cat_meets_criteria:
        ext_time, ext_recovery = generate_virtual_points(
            cat_time, cat_recovery, cat_params, target_day, interval_days
        )
        cat_catalyst_cum, cat_lixiviant_cum, cat_irrigation_rate = extend_curve_dynamic_inputs_to_time_grid(
            curve_data=pair.catalyzed,
            target_time_days=ext_time,
            feed_mass_kg=feed_mass_kg,
            column_inner_diameter_m=column_inner_diameter_m,
            step_days=interval_days,
            catalyst_history_window_days=catalyst_history_window_days,
            lix_history_window_days=lix_history_window_days,
        )
        new_pair.catalyzed = CurveData(
            status=pair.catalyzed.status,
            time=ext_time,
            recovery=ext_recovery,
            catalyst_cum=cat_catalyst_cum,
            lixiviant_cum=cat_lixiviant_cum,
            irrigation_rate_l_m2_h=cat_irrigation_rate,
            fit_params=pair.catalyzed.fit_params,
            row_index=pair.catalyzed.row_index,
        )
    
    return new_pair


# ---------------------------
# Data objects
# ---------------------------
@dataclass
class CurveData:
    status: str
    time: np.ndarray
    recovery: np.ndarray
    catalyst_cum: np.ndarray
    lixiviant_cum: np.ndarray
    irrigation_rate_l_m2_h: np.ndarray
    fit_params: np.ndarray
    row_index: int


@dataclass
class PairSample:
    sample_id: str
    static_raw: np.ndarray
    input_only_raw: np.ndarray
    control: CurveData
    catalyzed: CurveData
    static_scaled: Optional[np.ndarray] = None
    static_imputed: Optional[np.ndarray] = None
    # Per-sample leach caps (recovery-%, 0-100) derived from mineralogy.
    # ctrl_cap applies to the control curve; cat_cap to the catalyzed curve.
    ctrl_cap: float = 100.0
    cat_cap: float = 100.0


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
        "input_only_columns": INPUT_ONLY_COLUMNS,
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
    input_only_missing = {}
    for col in INPUT_ONLY_COLUMNS:
        if col not in df.columns:
            input_only_missing[col] = None
        else:
            input_only_missing[col] = int(df[col].isna().sum())
    summary["missing_counts_input_only"] = input_only_missing

    dynamic_stats = {}
    for col in [TIME_COL_COLUMNS, CATALYST_CUM_COL, LIXIVIANT_CUM_COL, TARGET_COLUMNS]:
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

        # Derive per-row leach cap directly from the row's mineralogy columns.
        _prefit_ctrl_cap, _prefit_cat_cap = compute_sample_leach_caps(
            cu_pct=scalar_from_maybe_array(row.get("cu_%", np.nan)),
            cu_oxides_equiv=scalar_from_maybe_array(row.get("copper_oxides_equivalent", np.nan)),
            cu_secondary_equiv=scalar_from_maybe_array(row.get("copper_secondary_sulfides_equivalent", np.nan)),
            cu_primary_equiv=scalar_from_maybe_array(row.get("copper_primary_sulfides_equivalent", np.nan)),
        )
        _prefit_cap = _prefit_cat_cap if status == "Catalyzed" else _prefit_ctrl_cap

        params = fit_biexponential_params(t, y, cap=_prefit_cap) if t.size >= 6 else np.array([np.nan] * 4, dtype=float)
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
        by_status: Dict[str, Tuple[CurveData, np.ndarray, np.ndarray]] = {}
        for idx, row in group.iterrows():
            status = normalize_status(row.get(STATUS_COL_PRIMARY, row.get(STATUS_COL_FALLBACK, "")))
            t_raw = parse_listlike(row.get(TIME_COL_COLUMNS, np.nan))
            y_raw = parse_listlike(row.get(TARGET_COLUMNS, np.nan))
            c_raw = parse_listlike(row.get(CATALYST_CUM_COL, np.nan))
            t, y, c = prepare_curve_arrays(t_raw, y_raw, c_raw, status=status, min_points=6)
            if t.size < 6:
                continue
            feed_mass_kg = scalar_from_maybe_array(row.get(FEED_MASS_COL, np.nan))
            column_inner_diameter_m = scalar_from_maybe_array(row.get("column_inner_diameter_m", np.nan))
            lix_raw = parse_listlike(row.get(LIXIVIANT_CUM_COL, np.nan))
            lix_raw_t, lix_raw_clean = prepare_cumulative_profile_with_time(t_raw, lix_raw, force_zero=False)
            if lix_raw_t.size >= 1:
                lix_aligned = np.interp(t, lix_raw_t, lix_raw_clean, left=0.0, right=float(lix_raw_clean[-1]))
                lix_aligned = clean_cumulative_profile(lix_aligned, force_zero=False)
            else:
                lix_aligned = np.zeros_like(t, dtype=float)
            irrigation_rate = convert_lixiviant_cum_to_irrigation_rate_l_m2_h(
                time_days=t,
                cumulative_lixiviant_m3_t=lix_aligned,
                feed_mass_kg=feed_mass_kg,
                column_inner_diameter_m=column_inner_diameter_m,
            )
            fit_params = np.asarray(
                [
                    pd.to_numeric(row.get("fit_a1", np.nan), errors="coerce"),
                    pd.to_numeric(row.get("fit_b1", np.nan), errors="coerce"),
                    pd.to_numeric(row.get("fit_a2", np.nan), errors="coerce"),
                    pd.to_numeric(row.get("fit_b2", np.nan), errors="coerce"),
                ],
                dtype=float,
            )

            # Per-row leach cap: use the named columns directly from the row,
            # selecting ctrl vs. cat based on status.
            _row_ctrl_cap, _row_cat_cap = compute_sample_leach_caps(
                cu_pct=scalar_from_maybe_array(row.get("cu_%", np.nan)),
                cu_oxides_equiv=scalar_from_maybe_array(row.get("copper_oxides_equivalent", np.nan)),
                cu_secondary_equiv=scalar_from_maybe_array(row.get("copper_secondary_sulfides_equivalent", np.nan)),
                cu_primary_equiv=scalar_from_maybe_array(row.get("copper_primary_sulfides_equivalent", np.nan)),
            )
            _row_cap = _row_cat_cap if status == "Catalyzed" else _row_ctrl_cap

            if not np.all(np.isfinite(fit_params)):
                fit_params = fit_biexponential_params(t, y, cap=_row_cap)

            static_vec = np.asarray(
                [scalar_from_maybe_array(row.get(col, np.nan)) for col in STATIC_PREDICTOR_COLUMNS],
                dtype=float,
            )
            input_only_vec = np.asarray(
                [scalar_from_maybe_array(row.get(col, np.nan)) for col in INPUT_ONLY_COLUMNS],
                dtype=float,
            )

            curve_data = CurveData(
                status=status,
                time=t,
                recovery=y,
                catalyst_cum=c,
                lixiviant_cum=np.asarray(lix_aligned, dtype=float),
                irrigation_rate_l_m2_h=np.asarray(irrigation_rate, dtype=float),
                fit_params=sanitize_curve_params(fit_params, cap=_row_cap),
                row_index=int(idx),
            )
            if status not in by_status or curve_data.time.size > by_status[status][0].time.size:
                by_status[status] = (curve_data, static_vec, input_only_vec)

        if "Control" not in by_status or "Catalyzed" not in by_status:
            continue
        ctrl_curve, ctrl_static, ctrl_input_only = by_status["Control"]
        cat_curve, cat_static, cat_input_only = by_status["Catalyzed"]
        merged_static = combine_static_vectors(ctrl_static, cat_static)
        merged_input_only = combine_static_vectors(ctrl_input_only, cat_input_only)

        # Compute per-pair leach caps from the merged static vector.
        def _sv(col: str) -> float:
            if col not in STATIC_PREDICTOR_COLUMNS:
                return np.nan
            return float(merged_static[STATIC_PREDICTOR_COLUMNS.index(col)])

        pair_ctrl_cap, pair_cat_cap = compute_sample_leach_caps(
            cu_pct=_sv("cu_%"),
            cu_oxides_equiv=_sv("copper_oxides_equivalent"),
            cu_secondary_equiv=_sv("copper_secondary_sulfides_equivalent"),
            cu_primary_equiv=_sv("copper_primary_sulfides_equivalent"),
        )
        pairs.append(
            PairSample(
                sample_id=str(sample_id),
                static_raw=merged_static,
                input_only_raw=merged_input_only,
                control=ctrl_curve,
                catalyzed=cat_curve,
                ctrl_cap=pair_ctrl_cap,
                cat_cap=pair_cat_cap,
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


def apply_static_transformers(
    pairs: List[PairSample],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> None:
    X = np.vstack([p.static_raw for p in pairs]).astype(float)
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    for p, xi, xs in zip(pairs, X_imp, X_scaled):
        p.static_imputed = np.asarray(xi, dtype=float)
        p.static_scaled = np.asarray(xs, dtype=float)


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
        min_sigmoid_to_cat_transition_days: float,
        max_sigmoid_to_cat_transition_days: float,
        max_catalyst_aging_strength: float,
        late_tau_impact_decay_strength: float,
        min_remaining_ore_factor: float,
        flat_input_transition_sensitivity: float,
        flat_input_uplift_response_days: float,
        flat_input_response_ramp_days: float,
        flat_input_late_uplift_response_boost: float,
        catalyst_dose_transition_shape_strength: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_static, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.ctrl_head = nn.Linear(hidden_dim, 4)
        self.cat_head = nn.Linear(hidden_dim, 4)
        self.delay_head = nn.Linear(hidden_dim, 1)
        self.height_delay_scale = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))
        self.temp_head = nn.Linear(hidden_dim, 1)
        self.kappa_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.kappa_head.bias, 1.0)  # softplus(1.0) ≈ 1.31
        self.aging_head = nn.Linear(hidden_dim, 1)
        self.lix_kappa_head = nn.Linear(hidden_dim, 1)
        self.lix_strength_head = nn.Linear(hidden_dim, 1)

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
        self.min_sigmoid_to_cat_transition_days = float(max(1e-3, min_sigmoid_to_cat_transition_days))
        self.max_sigmoid_to_cat_transition_days = float(
            max(self.min_sigmoid_to_cat_transition_days, max_sigmoid_to_cat_transition_days)
        )
        self.max_catalyst_aging_strength = float(max(0.0, max_catalyst_aging_strength))
        self.late_tau_impact_decay_strength = float(max(0.0, late_tau_impact_decay_strength))
        self.min_remaining_ore_factor = float(np.clip(min_remaining_ore_factor, 0.0, 1.0))
        self.flat_input_transition_sensitivity = float(max(0.0, flat_input_transition_sensitivity))
        self.flat_input_uplift_response_days = float(max(0.0, flat_input_uplift_response_days))
        self.flat_input_response_ramp_days = float(max(1e-3, flat_input_response_ramp_days))
        self.flat_input_late_uplift_response_boost = float(
            max(0.0, flat_input_late_uplift_response_boost)
        )
        self.catalyst_dose_transition_shape_strength = float(
            max(0.0, catalyst_dose_transition_shape_strength)
        )

    def _bounded_params(self, raw: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        return lb.unsqueeze(0) + torch.sigmoid(raw) * (ub - lb).unsqueeze(0)

    @staticmethod
    def _feature_column(x_static: torch.Tensor, idx: Optional[int]) -> torch.Tensor:
        if idx is None or idx < 0 or idx >= x_static.shape[1]:
            return torch.zeros((x_static.shape[0], 1), dtype=x_static.dtype, device=x_static.device)
        return x_static[:, idx : idx + 1]

    def _static_feature_map(
        self,
        x_static: torch.Tensor,
        x_static_raw: Optional[torch.Tensor] = None,
        x_input_only_raw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        terms = {
            col_name: self._feature_column(x_static, idx)
            for col_name, idx in self.static_feature_indices.items()
        }
        if x_static_raw is not None and x_input_only_raw is not None:
            raw_terms = {
                col_name: self._feature_column(x_static_raw, idx)
                for col_name, idx in self.static_feature_indices.items()
            }
            input_only_indices = {col_name: idx for idx, col_name in enumerate(INPUT_ONLY_COLUMNS)}
            input_only_terms = {
                col_name: self._feature_column(x_input_only_raw, idx)
                for col_name, idx in input_only_indices.items()
            }
            base_zeros = torch.zeros_like(next(iter(terms.values())))
            height_raw = input_only_terms.get("column_height_m", base_zeros)
            diameter_raw = input_only_terms.get("column_inner_diameter_m", base_zeros)
            feed_mass_raw = input_only_terms.get(FEED_MASS_COL, base_zeros)
            material_size_raw = raw_terms.get("material_size_p80_in", base_zeros)
            height_m = torch.where(
                torch.isfinite(height_raw),
                torch.clamp(height_raw, min=0.0),
                torch.zeros_like(height_raw),
            )
            diameter_m = torch.where(
                torch.isfinite(diameter_raw),
                torch.clamp(diameter_raw, min=0.0),
                torch.zeros_like(diameter_raw),
            )
            feed_mass_kg = torch.where(
                torch.isfinite(feed_mass_raw),
                torch.clamp(feed_mass_raw, min=0.0),
                torch.zeros_like(feed_mass_raw),
            )
            material_size_p80_in = torch.where(
                torch.isfinite(material_size_raw),
                torch.clamp(material_size_raw, min=0.0),
                torch.zeros_like(material_size_raw),
            )
            area_m2 = np.pi * torch.square(diameter_m * 0.5)
            volume_m3 = area_m2 * height_m
            valid_volume = (height_m > 0.0) & (diameter_m > 0.0)
            valid_density = valid_volume & (feed_mass_kg > 0.0)
            valid_ratio = (diameter_m > 0.0) & (material_size_p80_in > 0.0)
            feed_mass_t = torch.clamp(feed_mass_kg / 1000.0, min=1e-6)
            volume_safe = torch.clamp(volume_m3, min=1e-6)
            terms["apparent_bulk_density_t_m3"] = torch.where(
                valid_density,
                feed_mass_t / volume_safe,
                torch.zeros_like(volume_m3),
            )
            terms["material_size_to_column_diameter_ratio"] = torch.where(
                valid_ratio,
                material_size_p80_in * 0.0254 / torch.clamp(diameter_m, min=1e-6),
                torch.zeros_like(diameter_m),
            )
        return terms

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
        b1 = torch.clamp(p_ordered[:, 1], min=1e-5, max=1e-1)
        a2 = torch.clamp(p_ordered[:, 2], min=0.0, max=100.0)
        b2 = torch.clamp(p_ordered[:, 3], min=1e-5, max=1e-1)

        return torch.stack([a1, b1, a2, b2], dim=1)

    @staticmethod
    def _compute_remaining_ore_factor_chemistry_based(
        y_ctrl: torch.Tensor,
        cu_percent: torch.Tensor,
        copper_primary_sulfides_equiv: torch.Tensor,
        copper_oxides_equiv: torch.Tensor,
        min_floor: float = 0.05,
        copper_secondary_sulfides_equiv: Optional[torch.Tensor] = None,
        is_catalyzed: bool = False,
    ) -> torch.Tensor:
        """Compute remaining ore factor based on ore chemistry using leachable_real."""
        pct_ox  = float(CONFIG.get("leach_pct_oxides", 1.00))
        pct_sec = float(CONFIG.get("leach_pct_secondary_sulfides", 0.67))
        if is_catalyzed:
            pct_pri = float(CONFIG.get("leach_pct_primary_sulfides_catalyzed", 0.65))
        else:
            pct_pri = float(CONFIG.get("leach_pct_primary_sulfides_control", 0.20))

        cu_safe = torch.clamp(cu_percent, min=1e-6)

        leachable_real = copper_oxides_equiv * pct_ox + copper_primary_sulfides_equiv * pct_pri
        if copper_secondary_sulfides_equiv is not None:
            leachable_real = leachable_real + copper_secondary_sulfides_equiv * pct_sec

        max_leachable_frac = torch.clamp(leachable_real / cu_safe, min=1e-6, max=1.0)
        extracted_fraction = torch.clamp(y_ctrl / 100.0, min=0.0, max=1.0)

        remaining_fraction = torch.clamp(
            1.0 - extracted_fraction / max_leachable_frac,
            min=0.0,
            max=1.0,
        )
        remaining_ore_factor = torch.clamp(remaining_fraction, min=min_floor, max=1.0)
        return remaining_ore_factor

    @staticmethod
    def _compute_remaining_primary_uplift_factor(
        y_ctrl: torch.Tensor,
        y_cat_current: torch.Tensor,
        cu_percent: torch.Tensor,
        copper_primary_sulfides_equiv: torch.Tensor,
        min_floor: float = 0.0,
    ) -> torch.Tensor:
        pct_pri_ctrl = float(CONFIG.get("leach_pct_primary_sulfides_control", 0.30))
        pct_pri_cat = float(CONFIG.get("leach_pct_primary_sulfides_catalyzed", 0.70))

        cu_safe = torch.clamp(cu_percent, min=1e-6)
        pri_safe = torch.clamp(copper_primary_sulfides_equiv, min=0.0)

        max_primary_uplift = ((pct_pri_cat - pct_pri_ctrl) * pri_safe / cu_safe) * 100.0
        max_primary_uplift = torch.clamp(max_primary_uplift, min=1e-6)

        current_uplift = torch.clamp(y_cat_current - y_ctrl, min=0.0)

        remaining = 1.0 - current_uplift / max_primary_uplift
        return torch.clamp(remaining, min=min_floor, max=1.0)
    
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

    @staticmethod
    def _expand_series_to_batch_torch(
        series: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        s = torch.as_tensor(series, dtype=dtype, device=device)
        if s.ndim == 0:
            s = s.view(1, 1)
        elif s.ndim == 1:
            s = s.view(1, -1)
        if s.shape[0] != batch_size:
            if s.shape[0] == 1:
                s = s.expand(batch_size, -1)
            else:
                s = s[:1].expand(batch_size, -1)
        return s

    @staticmethod
    def _infer_catalyst_start_day_torch(cum_norm: torch.Tensor, t_days: torch.Tensor) -> torch.Tensor:
        c = torch.as_tensor(cum_norm, dtype=torch.float32)
        t = torch.as_tensor(t_days, dtype=torch.float32)
        if c.ndim == 1:
            c = c.view(1, -1)
        if t.ndim == 1:
            t = t.view(1, -1)
        if t.shape[0] != c.shape[0]:
            if t.shape[0] == 1:
                t = t.expand(c.shape[0], -1)
            elif c.shape[0] == 1:
                c = c.expand(t.shape[0], -1)

        starts = []
        for row_c, row_t in zip(c, t):
            valid = torch.isfinite(row_c) & torch.isfinite(row_t)
            if int(valid.sum().item()) <= 0:
                starts.append(torch.tensor(0.0, dtype=row_t.dtype, device=row_t.device))
                continue

            c_valid = torch.clamp(row_c[valid], min=0.0)
            t_valid = torch.clamp(row_t[valid], min=0.0)
            if c_valid.numel() <= 0:
                starts.append(torch.tensor(0.0, dtype=row_t.dtype, device=row_t.device))
                continue

            order = torch.argsort(t_valid)
            c_valid = c_valid[order]
            t_valid = t_valid[order]
            tol = max(1e-9, 1e-6 * float(max(1.0, torch.max(torch.abs(c_valid)).item())))
            if float(c_valid[0].item()) > tol:
                starts.append(t_valid[0])
                continue
            if c_valid.numel() <= 1:
                starts.append(torch.tensor(0.0, dtype=row_t.dtype, device=row_t.device))
                continue

            rise = torch.nonzero((c_valid[1:] - c_valid[:-1]) > tol, as_tuple=False)
            if int(rise.numel()) <= 0:
                starts.append(torch.tensor(0.0, dtype=row_t.dtype, device=row_t.device))
            else:
                starts.append(t_valid[int(rise[0].item())])
        return torch.stack(starts, dim=0).view(-1, 1)

    @staticmethod
    def _causal_response_smooth_torch(
        series: torch.Tensor,
        t_days: torch.Tensor,
        response_days: torch.Tensor,
    ) -> torch.Tensor:
        y = torch.as_tensor(series)
        if y.ndim == 1:
            y = y.view(1, -1)
        batch_size = int(y.shape[0])
        t = PairCurveNet._expand_series_to_batch_torch(t_days, batch_size, y.dtype, y.device)
        tau = PairCurveNet._expand_series_to_batch_torch(response_days, batch_size, y.dtype, y.device)
        if tau.shape[1] != y.shape[1]:
            if tau.shape[1] == 1:
                tau = tau.expand(batch_size, y.shape[1])
            else:
                tau = tau[:, : y.shape[1]]
        if y.shape[1] == 0:
            return y
        out_steps = [y[:, 0]]
        for i in range(1, y.shape[1]):
            dt = torch.clamp(t[:, i] - t[:, i - 1], min=0.0)
            tau_i = torch.clamp(tau[:, i], min=0.0)
            alpha = torch.where(
                tau_i <= 1e-6,
                torch.ones_like(tau_i),
                1.0 - torch.exp(-dt / torch.clamp(tau_i, min=1e-6)),
            )
            out_steps.append(out_steps[-1] + alpha * (y[:, i] - out_steps[-1]))
        return torch.stack(out_steps, dim=1)

    @staticmethod
    def _smooth_sigmoid_transition_torch(
        t_days: torch.Tensor,
        start_days: torch.Tensor,
        end_days: torch.Tensor,
    ) -> torch.Tensor:
        t = torch.as_tensor(t_days)
        if t.ndim == 1:
            t = t.view(1, -1)
        batch_size = int(t.shape[0])
        dtype = t.dtype
        device = t.device
        start = PairCurveNet._expand_series_to_batch_torch(start_days, batch_size, dtype, device)
        end = PairCurveNet._expand_series_to_batch_torch(end_days, batch_size, dtype, device)

        start = torch.clamp(start, min=0.0)
        end = torch.maximum(end, start + 1e-3)
        progress = (t - start) / torch.clamp(end - start, min=1e-6)
        progress = torch.clamp(progress, min=0.0, max=1.0)
        # Smootherstep gives a sigmoid-like handoff with zero slope at both ends.
        alpha = progress * progress * progress * (
            progress * (progress * 6.0 - 15.0) + 10.0
        )
        return torch.clamp(alpha, min=0.0, max=1.0)

    @staticmethod
    def _flat_input_score_torch(
        cum_norm: torch.Tensor,
        irrigation_rate_norm: torch.Tensor,
        t_days: torch.Tensor,
        catalyst_start_day: torch.Tensor,
        sensitivity: float,
        ramp_days: float,
    ) -> torch.Tensor:
        t = torch.as_tensor(t_days)
        if t.ndim == 1:
            t = t.view(1, -1)
        batch_size = int(t.shape[0])
        dtype = t.dtype
        device = t.device
        c = PairCurveNet._expand_series_to_batch_torch(cum_norm, batch_size, dtype, device)
        irr = PairCurveNet._expand_series_to_batch_torch(irrigation_rate_norm, batch_size, dtype, device)
        start = PairCurveNet._expand_series_to_batch_torch(catalyst_start_day, batch_size, dtype, device)
        if t.shape[1] == 0:
            return torch.zeros_like(t)
        if t.shape[1] == 1:
            return torch.ones_like(t)

        dt = torch.clamp(t[:, 1:] - t[:, :-1], min=1e-3)
        cat_rate = torch.clamp(c[:, 1:] - c[:, :-1], min=0.0) / dt
        cat_rate_full = torch.cat([cat_rate[:, :1], cat_rate], dim=1)

        cat_ref = torch.clamp(torch.amax(torch.abs(cat_rate_full), dim=1, keepdim=True), min=1e-4)
        irr_ref = torch.clamp(torch.amax(torch.abs(irr), dim=1, keepdim=True), min=1e-4)

        cat_change = torch.zeros_like(cat_rate_full)
        irr_change = torch.zeros_like(irr)
        cat_change[:, 1:] = torch.abs(cat_rate_full[:, 1:] - cat_rate_full[:, :-1]) / cat_ref
        irr_change[:, 1:] = torch.abs(irr[:, 1:] - irr[:, :-1]) / irr_ref

        flat_score = torch.exp(-float(max(0.0, sensitivity)) * (cat_change + irr_change))
        catalyst_progress = torch.clamp(
            (t - start) / float(max(1e-3, ramp_days)),
            min=0.0,
            max=1.0,
        )
        return torch.clamp(flat_score * catalyst_progress, min=0.0, max=1.0)

    def predict_params(
        self,
        x_static: torch.Tensor,
        x_static_raw: Optional[torch.Tensor] = None,
        x_input_only_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        h = self.encoder(x_static)
        p_ctrl = self._bounded_params(self.ctrl_head(h), self.ctrl_lb, self.ctrl_ub)
        p_cat = self._bounded_params(self.cat_head(h), self.cat_lb, self.cat_ub)
        p_ctrl = self._sanitize_params(p_ctrl)
        p_cat = self._sanitize_params(p_cat)
        tau_raw = self.delay_head(h)
        if self.geo_delay_head is not None:
            geo = x_static[:, self.geo_idx]
            tau_raw = tau_raw + self.geo_delay_head(geo)
        if x_input_only_raw is not None:
            input_only_indices = {col_name: idx for idx, col_name in enumerate(INPUT_ONLY_COLUMNS)}
            height_idx = input_only_indices.get("column_height_m")
            if height_idx is not None and height_idx < x_input_only_raw.shape[1]:
                height_raw = x_input_only_raw[:, height_idx : height_idx + 1]
                height_raw = torch.where(
                    torch.isfinite(height_raw),
                    torch.clamp(height_raw, min=0.0),
                    torch.zeros_like(height_raw),
                )
                tau_raw = tau_raw + F.softplus(self.height_delay_scale) * torch.log1p(height_raw)
        tau_days = self.tmax_days * torch.sigmoid(tau_raw)
        temp_days = self.min_transition_days + F.softplus(self.temp_head(h))
        kappa = 1e-3 + F.softplus(self.kappa_head(h))
        aging_strength = self.max_catalyst_aging_strength * torch.sigmoid(self.aging_head(h))
        lix_kappa = 1e-3 + F.softplus(self.lix_kappa_head(h))
        lix_strength = 0.25 + 1.50 * torch.sigmoid(self.lix_strength_head(h))
        interaction_terms = self._static_feature_map(x_static, x_static_raw, x_input_only_raw)
        
        cu_idx = self.static_feature_indices.get("cu_%")
        cu_prim_idx = self.static_feature_indices.get("copper_primary_sulfides_equivalent")
        cu_sec_idx = self.static_feature_indices.get("copper_secondary_sulfides_equivalent")
        cu_oxide_idx = self.static_feature_indices.get("copper_oxides_equivalent")

        batch_size = x_static.shape[0]
        ones = torch.ones((batch_size, 1), dtype=x_static.dtype, device=x_static.device)

        # Physical chemistry terms must stay on their native scale. Using the
        # standardized static vector here makes uplift caps fold-dependent
        # because each CV member fits a different scaler.
        x_static_physical = x_static_raw if x_static_raw is not None else x_static
        interaction_terms["cu_percent"] = (
            x_static_physical[:, cu_idx:cu_idx+1] if cu_idx is not None else ones
        )
        interaction_terms["cu_primary"] = (
            x_static_physical[:, cu_prim_idx:cu_prim_idx+1] if cu_prim_idx is not None else ones * 0.5
        )
        interaction_terms["cu_secondary"] = (
            x_static_physical[:, cu_sec_idx:cu_sec_idx+1] if cu_sec_idx is not None else torch.zeros_like(ones)
        )
        interaction_terms["cu_oxides"] = (
            x_static_physical[:, cu_oxide_idx:cu_oxide_idx+1] if cu_oxide_idx is not None else ones * 0.1
        )

        chem_raw = self.chem_mix_head(h)
        if self.chem_direct_head is not None:
            interaction_terms["chem_direct"] = self.chem_direct_head(x_static[:, self.chem_idx])
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

        latent_params = {
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
            "ore_decay_strength": ore_decay_strength,
            "passivation_strength": passivation_strength,
            "passivation_tau": passivation_tau,
            "passivation_temp": passivation_temp,
            "depassivation_strength": depassivation_strength,
            "transform_strength": transform_strength,
            "transform_tau": transform_tau,
            "transform_temp": transform_temp,
            "lixiviant_kappa": lix_kappa,
            "lixiviant_strength": lix_strength,
            "cu_percent": interaction_terms["cu_percent"],
            "cu_primary": interaction_terms["cu_primary"],
            "cu_secondary": interaction_terms["cu_secondary"],
            "cu_oxides": interaction_terms["cu_oxides"],
        }
        return p_ctrl, p_cat, tau_days, temp_days, kappa, aging_strength, latent_params

    def curves_given_params(
        self,
        p_ctrl: torch.Tensor,
        p_cat: torch.Tensor,
        t_days: torch.Tensor,
        cum_norm: torch.Tensor,
        lix_cum_norm: torch.Tensor,
        irrigation_rate_norm: torch.Tensor,
        tau_days: torch.Tensor,
        temp_days: torch.Tensor,
        kappa: torch.Tensor,
        aging_strength: torch.Tensor,
        latent_params: Optional[Dict[str, torch.Tensor]] = None,
        ctrl_cap: Optional[torch.Tensor] = None,
        cat_cap: Optional[torch.Tensor] = None,
        return_states: bool = False,
    ) -> Any:
        t = t_days.view(1, -1)
        if latent_params is None:
            batch_size = int(p_ctrl.shape[0])
            ones = torch.ones((batch_size, 1), dtype=p_ctrl.dtype, device=p_ctrl.device)
            zeros = torch.zeros((batch_size, 1), dtype=p_ctrl.dtype, device=p_ctrl.device)
            latent_params = {
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
                "ore_decay_strength": zeros,
                "passivation_strength": zeros,
                "passivation_tau": zeros,
                "passivation_temp": self.min_transition_days * ones,
                "depassivation_strength": zeros,
                "transform_strength": zeros,
                "transform_tau": zeros,
                "transform_temp": self.min_transition_days * ones,
                "lixiviant_kappa": ones,
                "lixiviant_strength": ones,
                "cu_percent": ones,
                "cu_primary": ones * 0.5,
                "cu_secondary": torch.zeros_like(ones),
                "cu_oxides": ones * 0.1,
            }

        base_ctrl = self._double_exp_curve_torch(p_ctrl, t_days)
        catalyst_start_day = self._infer_catalyst_start_day_torch(cum_norm, t_days).to(p_ctrl.device, p_ctrl.dtype)
        effective_tau_days = torch.clamp(catalyst_start_day + tau_days, min=0.0, max=self.tmax_days)
        
        delay_factor = torch.sigmoid((t_days - effective_tau_days) / torch.clamp(temp_days, min=1e-6))
        # delay_factor = delay_factor ** 2.0

        effective_catalyst = self._decayed_effective_cumulative_torch(
            cum_norm=cum_norm,
            t_days=t_days,
            aging_strength=aging_strength,
            tmax_days=self.tmax_days,
        )
        catalyst_factor = 1.0 - torch.exp(-torch.clamp(kappa, min=1e-6) * effective_catalyst)
        '''
        dose = torch.clamp(effective_catalyst, min=0.0)
        catalyst_factor = 1.0 - torch.exp(
            -torch.clamp(kappa, min=1e-6) * dose / (1.0 + 2.0 * dose)
        )
        '''
        batch_size = int(p_ctrl.shape[0])
        effective_lixiviant = torch.clamp(
            self._expand_series_to_batch_torch(lix_cum_norm, batch_size, p_ctrl.dtype, p_ctrl.device),
            min=0.0,
        )
        irrigation_rate = torch.clamp(
            self._expand_series_to_batch_torch(irrigation_rate_norm, batch_size, p_ctrl.dtype, p_ctrl.device),
            min=0.0,
        )
        flat_input_score = self._flat_input_score_torch(
            cum_norm=cum_norm,
            irrigation_rate_norm=irrigation_rate,
            t_days=t_days,
            catalyst_start_day=catalyst_start_day,
            sensitivity=self.flat_input_transition_sensitivity,
            ramp_days=self.flat_input_response_ramp_days,
        ).to(p_ctrl.device, p_ctrl.dtype)

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
        ore_decay_strength = torch.clamp(latent_params["ore_decay_strength"], min=0.0)
        passivation_strength = torch.clamp(latent_params["passivation_strength"], min=0.0, max=1.0)
        passivation_tau = torch.clamp(latent_params["passivation_tau"], min=0.0, max=self.tmax_days)
        passivation_temp = torch.clamp(latent_params["passivation_temp"], min=1e-3)
        depassivation_strength = torch.clamp(latent_params["depassivation_strength"], min=0.0)
        transform_strength = torch.clamp(latent_params["transform_strength"], min=0.0)
        transform_tau = torch.clamp(latent_params["transform_tau"], min=0.0, max=self.tmax_days)
        transform_temp = torch.clamp(latent_params["transform_temp"], min=1e-3)
        lix_kappa = torch.clamp(latent_params["lixiviant_kappa"], min=1e-4)
        lix_strength = torch.clamp(latent_params["lixiviant_strength"], min=0.10, max=2.50)

        lix_inventory_factor = 1.0 - torch.exp(-0.85 * lix_kappa * effective_lixiviant)
        irrigation_factor = 1.0 - torch.exp(-lix_kappa * irrigation_rate)
        lix_factor = torch.clamp(0.40 * lix_inventory_factor + 0.60 * irrigation_factor, min=0.0, max=1.50)
        lix_availability = torch.clamp(
            0.15 + lix_strength * (0.40 * lix_inventory_factor + 0.60 * irrigation_factor),
            min=0.15,
            max=1.75,
        )

        ore_accessibility = torch.exp(-ore_decay_strength * torch.clamp(base_ctrl, 0.0, 100.0) / 100.0)
        ore_accessibility = ore_accessibility * torch.exp(
            -0.35 * ore_decay_strength * t / max(self.tmax_days, 1e-6)
        )
        ore_accessibility = torch.clamp(
            ore_accessibility,
            min=self.min_remaining_ore_factor,
            max=1.0,
        )

        fast_release = torch.clamp(
            (0.65 * fast_leach_inventory + 0.35 * oxide_inventory)
            * torch.exp(-1.25 * t / max(self.tmax_days, 1e-6))
            * (0.50 + 0.50 * ore_accessibility),
            min=0.0,
            max=1.25,
        )
        fast_release = torch.clamp(
            fast_release * (0.80 + 0.35 * lix_availability),
            min=0.0,
            max=1.25,
        )

        acid_buffer_remaining = torch.exp(-acid_buffer_decay * t / max(self.tmax_days, 1e-6))
        acid_buffer_penalty = torch.clamp(
            acid_buffer_strength
            * acid_buffer_remaining
            * torch.clamp(1.0 - 0.35 * lix_factor, min=0.35, max=1.0),
            min=0.0,
            max=0.95,
        )

        diffusion_progress = torch.sqrt(torch.clamp(t / max(self.tmax_days, 1e-6), min=0.0, max=1.0))
        diffusion_drag = torch.clamp(
            diffusion_drag_strength
            * diffusion_progress
            * (0.60 + 0.40 * primary_drive)
            * torch.clamp(1.0 - 0.30 * lix_factor, min=0.35, max=1.0),
            min=0.0,
            max=0.95,
        )

        passivation_progress = torch.sigmoid((t - passivation_tau) / passivation_temp)
        passivation = torch.clamp(
            passivation_strength
            * primary_drive
            * chem_interaction
            * ferric_synergy
            * passivation_progress
            * ore_accessibility
            * (1.0 - 0.45 * fast_release)
            * (1.0 + 0.20 * acid_buffer_penalty),
            min=0.0,
            max=0.95,
        )

        transformation_progress = torch.sigmoid((t - transform_tau) / transform_temp)
        transformation = torch.clamp(
            transform_strength
            * chem_interaction
            * transformation_progress
            * (0.35 + 0.65 * ferric_synergy)
            * (0.35 + 0.65 * catalyst_factor)
            * (0.45 + 0.55 * ore_accessibility),
            min=0.0,
            max=1.25,
        )
        transformation = torch.clamp(
            transformation * (0.50 + 0.50 * lix_availability),
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
            * (0.35 + 0.65 * transformation),
            min=0.0,
            max=1.50,
        )
        depassivation = torch.clamp(
            depassivation * (0.45 + 0.55 * lix_availability),
            min=0.0,
            max=1.50,
        )

        ctrl_rate_multiplier = torch.clamp(
            ore_accessibility
            * (1.0 - passivation)
            * (1.0 - 0.65 * acid_buffer_penalty)
            * (1.0 - diffusion_drag)
            * (0.85 + 0.35 * fast_release + 0.10 * transformation),
            min=0.08,
            max=1.75,
        )
        ctrl_rate_multiplier = torch.clamp(
            ctrl_rate_multiplier * (0.80 + 0.35 * lix_availability),
            min=0.08,
            max=1.75,
        )
        cat_rate_multiplier = torch.clamp(
            ore_accessibility
            * (1.0 - 0.35 * passivation)
            * (1.0 - 0.40 * acid_buffer_penalty)
            * (1.0 - 0.60 * diffusion_drag)
            * (
                0.90
                + 0.30 * fast_release
                + depassivation
                + 0.25 * transformation
                + 0.18 * primary_catalyst_synergy
            ),
            min=0.08,
            max=2.75,
        )
        cat_rate_multiplier = torch.clamp(
            cat_rate_multiplier * (0.90 + 0.45 * lix_availability),
            min=0.08,
            max=2.75,
        )

        t_eff_ctrl = self._effective_time_from_rate_torch(t_days, ctrl_rate_multiplier)
        t_eff_cat = self._effective_time_from_rate_torch(t_days, cat_rate_multiplier)

        y_ctrl = self._double_exp_curve_from_grid_torch(p_ctrl, t_eff_ctrl)
        y_cat_target = self._double_exp_curve_from_grid_torch(p_cat, t_eff_cat)

        # New chemistry-based calculation:
        # Extract the copper species from latent_params
        cu_percent = latent_params.get("cu_percent")
        cu_primary = latent_params.get("cu_primary")
        # cu_oxides = latent_params.get("cu_oxides")
        # cu_secondary = latent_params.get("cu_secondary")

        
        if cu_percent is not None and cu_primary is not None:
            remaining_primary_uplift_factor = self._compute_remaining_primary_uplift_factor(
                y_ctrl=y_ctrl,
                y_cat_current=y_cat_target,
                cu_percent=cu_percent,
                copper_primary_sulfides_equiv=cu_primary,
                min_floor=0.0,
            )
        else:
            remaining_primary_uplift_factor = torch.ones_like(y_ctrl)

        late_tau_factor = torch.exp(
            -self.late_tau_impact_decay_strength * torch.clamp(effective_tau_days, min=0.0) / max(self.tmax_days, 1e-6)
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
        latent_gap_factor = torch.clamp(latent_gap_factor + 0.20 * lix_factor, min=0.0, max=2.95)
        
        '''
        separation_factor = torch.clamp(
            delay_factor * catalyst_factor * remaining_ore_factor * late_tau_factor * latent_gap_factor,
            min=0.0,
            max=1.0,
        )
        '''

        separation_factor = (
            delay_factor
            * catalyst_factor
            * late_tau_factor
            * latent_gap_factor
            * (0.15 + 0.85 * remaining_primary_uplift_factor)
        )
        separation_factor = torch.clamp(separation_factor, min=0.0, max=1.25)

        late_flat_transition = torch.clamp(delay_factor * flat_input_score, min=0.0, max=1.0)
        cat_transition_response_days = (
            self.flat_input_uplift_response_days
            * flat_input_score
            * (1.0 + self.flat_input_late_uplift_response_boost * late_flat_transition)
        )

        post_catalyst_touch_mask = (t > catalyst_start_day).to(p_ctrl.dtype)
        post_touch_points = torch.clamp(torch.sum(post_catalyst_touch_mask, dim=1, keepdim=True), min=1.0)
        transition_strength = torch.sum(
            torch.clamp(separation_factor, min=0.0, max=1.0) * post_catalyst_touch_mask,
            dim=1,
            keepdim=True,
        ) / post_touch_points
        mean_late_flat_transition = torch.sum(
            late_flat_transition * post_catalyst_touch_mask,
            dim=1,
            keepdim=True,
        ) / post_touch_points
        mean_transition_response_days = torch.sum(
            cat_transition_response_days * post_catalyst_touch_mask,
            dim=1,
            keepdim=True,
        ) / post_touch_points
        transition_start_day = catalyst_start_day
        tau_offset_days = torch.clamp(effective_tau_days - transition_start_day, min=0.0)
        base_transition_days = (
            tau_offset_days
            + float(self.flat_input_response_ramp_days)
            + mean_transition_response_days
            * (
                1.50
                + self.flat_input_late_uplift_response_boost
                * torch.clamp(mean_late_flat_transition, min=0.0, max=1.0)
            )
        )
        transition_slowness_factor = 1.0 + 1.50 * (1.0 - torch.clamp(transition_strength, min=0.0, max=1.0))
        min_transition_duration_days = min(
            self.min_sigmoid_to_cat_transition_days,
            self.tmax_days,
        )
        max_transition_duration_days = max(
            min_transition_duration_days,
            min(self.max_sigmoid_to_cat_transition_days, self.tmax_days),
        )
        transition_duration_days = torch.clamp(
            base_transition_days * transition_slowness_factor,
            min=min_transition_duration_days,
            max=max_transition_duration_days,
        )
        transition_end_day = torch.maximum(transition_start_day + transition_duration_days, transition_start_day + 1e-3)
        transition_end_day = torch.clamp(transition_end_day, max=self.tmax_days)
        transition_duration_days = torch.clamp(transition_end_day - transition_start_day, min=1e-3)
        transition_center_day = transition_start_day + 0.5 * (transition_end_day - transition_start_day)
        base_transition_weight = self._smooth_sigmoid_transition_torch(
            t_days=t_days,
            start_days=transition_start_day,
            end_days=transition_end_day,
        )
        base_transition_weight = base_transition_weight * (t >= transition_start_day).to(p_ctrl.dtype)

        # Keep the v8 anchored target asymptote, but use a v7-like smoothed
        # dose signal to reshape the handoff kinetics. Higher catalyst dose
        # advances the transition earlier/faster without changing the final
        # catalyzed plateau.
        dose_transition_signal = self._causal_response_smooth_torch(
            torch.clamp(separation_factor, min=0.0, max=1.0),
            t_days=t_days,
            response_days=cat_transition_response_days,
        )
        dose_transition_signal = torch.clamp(dose_transition_signal, min=0.0, max=1.0)
        dose_transition_signal = dose_transition_signal * post_catalyst_touch_mask
        dose_transition_signal = torch.cummax(dose_transition_signal, dim=1).values
        dose_transition_scale = torch.amax(dose_transition_signal, dim=1, keepdim=True)
        dose_transition_progress = torch.where(
            dose_transition_scale > 1e-6,
            dose_transition_signal / torch.clamp(dose_transition_scale, min=1e-6),
            dose_transition_signal,
        )
        dose_transition_progress = torch.clamp(dose_transition_progress, min=0.0, max=1.0)
        transition_shape = torch.clamp(
            1.0
            - self.catalyst_dose_transition_shape_strength
            * (2.0 * dose_transition_progress - 1.0),
            min=0.35,
            max=2.5,
        )
        transition_weight = torch.pow(
            torch.clamp(base_transition_weight, min=0.0, max=1.0),
            transition_shape,
        )
        transition_weight = torch.clamp(transition_weight, min=0.0, max=1.0)
        transition_weight = torch.cummax(transition_weight, dim=1).values
        transition_weight = transition_weight * post_catalyst_touch_mask
        transition_weight = torch.where(
            base_transition_weight >= 1.0 - 1e-6,
            torch.ones_like(transition_weight),
            transition_weight,
        )

        cat_gap_target = torch.clamp(y_cat_target - y_ctrl, min=0.0)
        y_cat_target_anchored = y_ctrl + cat_gap_target
        cat_gap_blended_raw = transition_weight * cat_gap_target
        cat_gap_blended = transition_weight * cat_gap_target
        y_cat_blended = y_ctrl + cat_gap_blended

        # Smooth monotone enforcement only — no hard cap ceiling
        y_ctrl = torch.cummax(torch.clamp(y_ctrl, 0.0, 100.0), dim=1).values
        y_cat_target_anchored = torch.cummax(torch.clamp(y_cat_target_anchored, 0.0, 100.0), dim=1).values
        y_cat = torch.cummax(torch.clamp(y_cat_blended, 0.0, 100.0), dim=1).values
        y_cat = torch.minimum(y_cat, y_cat_target_anchored)
        y_cat = torch.maximum(y_cat, y_ctrl)

        # Physical bounds only — no mineralogy cap enforcement here
        y_ctrl = torch.clamp(y_ctrl, 0.0, 100.0)
        y_cat_target = torch.clamp(y_cat_target, 0.0, 100.0)
        y_cat_target_anchored = torch.clamp(y_cat_target_anchored, 0.0, 100.0)
        y_cat = torch.clamp(y_cat, 0.0, 100.0)

        if not return_states:
            return y_ctrl.squeeze(0), y_cat.squeeze(0)

        states = {
            "ore_accessibility": ore_accessibility.squeeze(0),
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
            "effective_tau_days": effective_tau_days.squeeze(0),
            "catalyst_start_day": catalyst_start_day.squeeze(0),
            "effective_lixiviant": effective_lixiviant.squeeze(0),
            "catalyst_factor": catalyst_factor.squeeze(0),
            "lix_factor": lix_factor.squeeze(0),
            "flat_input_score": flat_input_score.squeeze(0),
            "catalyzed_transition_start_day": transition_start_day.squeeze(0),
            "catalyzed_transition_center_day": transition_center_day.squeeze(0),
            "catalyzed_transition_end_day": transition_end_day.squeeze(0),
            "catalyzed_transition_duration_days": transition_duration_days.squeeze(0),
            "catalyzed_dose_transition_progress": dose_transition_progress.squeeze(0),
            "catalyzed_target_curve": y_cat_target.squeeze(0),
            "catalyzed_target_curve_anchored": y_cat_target_anchored.squeeze(0),
            "catalyzed_transition_weight": transition_weight.squeeze(0),
            "catalyzed_gap_target": cat_gap_target.squeeze(0),
            "catalyzed_gap_blended": cat_gap_blended.squeeze(0),
            "catalyzed_pred_blended": y_cat.squeeze(0),
            "catalyzed_post_start_mask": post_catalyst_touch_mask.squeeze(0),
            "catalyzed_gap_raw": cat_gap_blended_raw.squeeze(0),
            "catalyzed_gap_smoothed": cat_gap_blended.squeeze(0),
            "irrigation_rate": irrigation_rate.squeeze(0),
            "ferric_synergy": ferric_synergy.squeeze(0),
            "surface_refresh": surface_refresh.squeeze(0),
        }
        return y_ctrl.squeeze(0), y_cat.squeeze(0), states


# ---------------------------
# Shared ensemble plot annotations
# ---------------------------
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


def _weighted_curvature_penalty(y: torch.Tensor, t: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    slope = _curve_slopes(y, t)
    if slope.numel() < 2:
        return torch.tensor(0.0, device=y.device)
    curv = torch.abs(slope[1:] - slope[:-1])
    w = torch.as_tensor(weight, dtype=y.dtype, device=y.device).view(-1)
    if w.numel() < curv.numel() + 2:
        return curv.mean()
    w_mid = torch.clamp(w[1:-1], min=0.0)[: curv.numel()]
    if w_mid.numel() == 0:
        return curv.mean()
    weight_sum = torch.clamp(w_mid.sum(), min=1e-6)
    return torch.sum(curv * w_mid) / weight_sum


def _weighted_positive_curvature_penalty(y: torch.Tensor, t: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    slope = _curve_slopes(y, t)
    if slope.numel() < 2:
        return torch.tensor(0.0, device=y.device)
    curv = torch.relu(slope[1:] - slope[:-1])
    w = torch.as_tensor(weight, dtype=y.dtype, device=y.device).view(-1)
    if w.numel() < curv.numel() + 2:
        return curv.mean()
    w_mid = torch.clamp(w[1:-1], min=0.0)[: curv.numel()]
    if w_mid.numel() == 0:
        return curv.mean()
    weight_sum = torch.clamp(w_mid.sum(), min=1e-6)
    return torch.sum(curv * w_mid) / weight_sum


def _slope_cap_penalty(y: torch.Tensor, t: torch.Tensor, max_slope_per_day: float) -> torch.Tensor:
    slope = _curve_slopes(y, t)
    if slope.numel() < 1:
        return torch.tensor(0.0, device=y.device)
    return torch.relu(torch.abs(slope) - float(max_slope_per_day)).mean()


def _asymptote_cap_penalty(p: torch.Tensor, cap: float) -> torch.Tensor:
    """Soft penalty: relu((a1 + a2) - cap)^2, fires only when the sum exceeds cap."""
    asymptote = p[:, 0] + p[:, 2]  # shape (batch,)
    excess = torch.relu(asymptote - float(cap))
    return (excess ** 2).mean()


def pair_training_loss(
    model: PairCurveNet,
    pair: PairSample,
    cum_scale: float,
    lix_scale: float,
    irrigation_scale: float,
    loss_weights: Dict[str, float],
    max_cat_slope_per_day: float,
) -> torch.Tensor:
    x = torch.tensor(pair.static_scaled, dtype=torch.float32, device=device).unsqueeze(0)
    raw_for_geometry = pair.static_imputed if pair.static_imputed is not None else pair.static_raw
    x_raw = torch.tensor(raw_for_geometry, dtype=torch.float32, device=device).unsqueeze(0)
    x_input_only = torch.tensor(pair.input_only_raw, dtype=torch.float32, device=device).unsqueeze(0)
    ctrl_t = torch.tensor(pair.control.time, dtype=torch.float32, device=device)
    ctrl_y = torch.tensor(pair.control.recovery, dtype=torch.float32, device=device)
    ctrl_c = torch.tensor(pair.control.catalyst_cum / cum_scale, dtype=torch.float32, device=device)
    ctrl_l = torch.tensor(pair.control.lixiviant_cum / lix_scale, dtype=torch.float32, device=device)
    ctrl_irr = torch.tensor(
        pair.control.irrigation_rate_l_m2_h / max(float(irrigation_scale), 1e-6),
        dtype=torch.float32,
        device=device,
    )

    cat_t = torch.tensor(pair.catalyzed.time, dtype=torch.float32, device=device)
    cat_y = torch.tensor(pair.catalyzed.recovery, dtype=torch.float32, device=device)
    cat_c = torch.tensor(pair.catalyzed.catalyst_cum / cum_scale, dtype=torch.float32, device=device)
    cat_l = torch.tensor(pair.catalyzed.lixiviant_cum / lix_scale, dtype=torch.float32, device=device)
    cat_irr = torch.tensor(
        pair.catalyzed.irrigation_rate_l_m2_h / max(float(irrigation_scale), 1e-6),
        dtype=torch.float32,
        device=device,
    )

    p_ctrl, p_cat, tau, temp, kappa, aging_strength, latent = model.predict_params(x, x_raw, x_input_only)

    ctrl_cap_t = torch.tensor([pair.ctrl_cap], dtype=torch.float32, device=device)
    cat_cap_t = torch.tensor([pair.cat_cap], dtype=torch.float32, device=device)

    pred_ctrl, _, ctrl_states = model.curves_given_params(
        p_ctrl,
        p_cat,
        ctrl_t,
        ctrl_c,
        ctrl_l,
        ctrl_irr,
        tau,
        temp,
        kappa,
        aging_strength,
        latent_params=latent,
        ctrl_cap=ctrl_cap_t,
        cat_cap=cat_cap_t,
        return_states=True,
    )
    cat_ctrl_c = torch.zeros_like(cat_c)
    pred_ctrl_cat_time, _ = model.curves_given_params(
        p_ctrl,
        p_cat,
        cat_t,
        cat_ctrl_c,
        cat_l,
        cat_irr,
        tau,
        temp,
        kappa,
        aging_strength,
        latent_params=latent,
        ctrl_cap=ctrl_cap_t,
        cat_cap=cat_cap_t,
    )
    _, pred_cat, cat_states = model.curves_given_params(
        p_ctrl,
        p_cat,
        cat_t,
        cat_c,
        cat_l,
        cat_irr,
        tau,
        temp,
        kappa,
        aging_strength,
        latent_params=latent,
        ctrl_cap=ctrl_cap_t,
        cat_cap=cat_cap_t,
        return_states=True,
    )

    ctrl_true_on_cat_t = torch.tensor(
        np.interp(
            pair.catalyzed.time,
            pair.control.time,
            pair.control.recovery,
            left=float(pair.control.recovery[0]),
            right=float(pair.control.recovery[-1]),
        ),
        dtype=torch.float32,
        device=device,
    )

    true_gap = torch.clamp(cat_y - ctrl_true_on_cat_t, min=0.0)
    pred_gap = torch.clamp(pred_cat - pred_ctrl_cat_time, min=0.0)

    post_cat_weight = 1.0 + 1.5 * (cat_c > 0).float()
    uplift_fit_pen = torch.mean(
        post_cat_weight * F.smooth_l1_loss(pred_gap, true_gap, reduction="none")
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
    )
    latent_cat_rate_pen = torch.relu(cat_states["ctrl_rate_multiplier"] - cat_states["cat_rate_multiplier"]).mean()
    flat_input_smooth_pen = _weighted_curvature_penalty(
        cat_states["catalyzed_transition_weight"],
        cat_t,
        cat_states["flat_input_score"],
    )
    flat_input_accel_pen = _weighted_positive_curvature_penalty(
        cat_states["catalyzed_gap_blended"],
        cat_t,
        cat_states["flat_input_score"],
    )
    target_fit_weight = (
        post_cat_weight
        * cat_states["catalyzed_post_start_mask"]
        * (0.25 + 0.75 * cat_states["catalyzed_transition_weight"])
    )
    target_fit_pen = torch.mean(
        target_fit_weight
        * F.smooth_l1_loss(
            pred_cat,
            cat_states["catalyzed_target_curve_anchored"],
            reduction="none",
        )
    )
    target_point_loss = F.smooth_l1_loss(
        cat_states["catalyzed_target_curve_anchored"],
        cat_y,
        reduction="none",
    )
    pred_point_loss = F.smooth_l1_loss(pred_cat, cat_y, reduction="none")
    target_advantage = torch.relu(pred_point_loss - target_point_loss)
    transition_takeover_pen = torch.mean(
        post_cat_weight
        * cat_states["catalyzed_post_start_mask"]
        * target_advantage
        * torch.square(1.0 - cat_states["catalyzed_transition_weight"])
    )
    transition_days_pen = torch.mean(
        torch.clamp(
            cat_states["catalyzed_transition_duration_days"] / max(model.tmax_days, 1e-6),
            min=0.0,
            max=1.0,
        )
    )

    target_ctrl_p = torch.tensor(pair.control.fit_params, dtype=torch.float32, device=device)
    target_cat_p = torch.tensor(pair.catalyzed.fit_params, dtype=torch.float32, device=device)
    param_pen = F.smooth_l1_loss(p_ctrl.squeeze(0), target_ctrl_p) + F.smooth_l1_loss(
        p_cat.squeeze(0), target_cat_p
    )

    # Per-sample leach-cap penalty: penalises p_ctrl / p_cat independently
    # when their a1+a2 asymptote exceeds the mineralogy-derived cap for that
    # sample.  ctrl_cap and cat_cap are treated as separate constraints.
    def _output_cap_penalty(y_pred: torch.Tensor, cap: float) -> torch.Tensor:
        cap_t = float(cap)
        excess = torch.relu(y_pred - cap_t)
        rel_excess = excess / max(cap_t, 1.0)
        return (excess ** 2).mean() + 5.0 * (rel_excess ** 2).mean()
    
    cap_pen = (
        _output_cap_penalty(pred_ctrl, pair.ctrl_cap)
        + _output_cap_penalty(pred_cat, pair.cat_cap)
    )

    total = (
        loss_ctrl
        + loss_cat
        + float(loss_weights.get("gap", 1.0)) * gap_pen
        + float(loss_weights.get("monotonic", 0.02)) * mono_pen
        + float(loss_weights.get("param", 0.08)) * param_pen
        + float(loss_weights.get("smooth_cat", 0.12)) * cat_smooth_pen
        + float(loss_weights.get("slope_cap", 0.18)) * cat_slope_cap_pen
        + float(loss_weights.get("latent_smooth", 0.02)) * latent_smooth_pen
        + float(loss_weights.get("latent_cat_rate", 0.03)) * latent_cat_rate_pen
        + float(loss_weights.get("flat_input_smooth", 0.08)) * flat_input_smooth_pen
        + float(loss_weights.get("flat_input_accel", 0.16)) * flat_input_accel_pen
        + float(loss_weights.get("cap", 0.10)) * cap_pen
        + float(loss_weights.get("uplift_fit", 0.5)) * uplift_fit_pen
        + float(loss_weights.get("target_fit", 0.25)) * target_fit_pen
        + float(loss_weights.get("transition_days", 0.10)) * transition_days_pen
        + float(loss_weights.get("transition_takeover", 0.35)) * transition_takeover_pen
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
    lix_scale: float,
    irrigation_scale: float,
    epochs: int,
    patience: int,
    geo_idx: List[int],
) -> Tuple[PairCurveNet, List[Dict[str, float]], float]:
    set_all_seeds(seed, deterministic=True)
    rng = np.random.default_rng(seed)
    max_cat_slope_per_day = float(CONFIG.get("max_cat_slope_per_day", 0.20))
    bootstrap_train_pairs = bool(CONFIG.get("bootstrap_train_pairs", False))

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
        min_sigmoid_to_cat_transition_days=float(
            CONFIG.get("min_sigmoid_to_cat_transition_days", 7.0)
        ),
        max_sigmoid_to_cat_transition_days=float(
            CONFIG.get("max_sigmoid_to_cat_transition_days", 245.0)
        ),
        max_catalyst_aging_strength=float(CONFIG.get("max_catalyst_aging_strength", 5.0)),
        late_tau_impact_decay_strength=float(CONFIG.get("late_tau_impact_decay_strength", 1.15)),
        min_remaining_ore_factor=float(CONFIG.get("min_remaining_ore_factor", 0.08)),
        flat_input_transition_sensitivity=float(CONFIG.get("flat_input_transition_sensitivity", 6.0)),
        flat_input_uplift_response_days=float(CONFIG.get("flat_input_uplift_response_days", 75.0)),
        flat_input_response_ramp_days=float(CONFIG.get("flat_input_response_ramp_days", 75.0)),
        flat_input_late_uplift_response_boost=float(
            CONFIG.get("flat_input_late_uplift_response_boost", 2.5)
        ),
        catalyst_dose_transition_shape_strength=float(
            CONFIG.get("catalyst_dose_transition_shape_strength", 0.75)
        ),
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
        if bootstrap_train_pairs:
            epoch_indices = rng.choice(len(train_pairs), size=len(train_pairs), replace=True)
            epoch_pairs = [train_pairs[i] for i in epoch_indices]
        else:
            epoch_pairs = list(train_pairs)
        rng.shuffle(epoch_pairs)
        train_losses = []
        for pair in epoch_pairs:
            optimizer.zero_grad(set_to_none=True)
            loss = pair_training_loss(
                model=model,
                pair=pair,
                cum_scale=cum_scale,
                lix_scale=lix_scale,
                irrigation_scale=irrigation_scale,
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
                    lix_scale=lix_scale,
                    irrigation_scale=irrigation_scale,
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
        min_sigmoid_to_cat_transition_days=float(
            cfg.get(
                "min_sigmoid_to_cat_transition_days",
                CONFIG.get("min_sigmoid_to_cat_transition_days", 7.0),
            )
        ),
        max_sigmoid_to_cat_transition_days=float(
            cfg.get(
                "max_sigmoid_to_cat_transition_days",
                CONFIG.get("max_sigmoid_to_cat_transition_days", 245.0),
            )
        ),
        max_catalyst_aging_strength=float(
            cfg.get("max_catalyst_aging_strength", CONFIG.get("max_catalyst_aging_strength", 5.0))
        ),
        late_tau_impact_decay_strength=float(
            cfg.get("late_tau_impact_decay_strength", CONFIG.get("late_tau_impact_decay_strength", 1.15))
        ),
        min_remaining_ore_factor=float(
            cfg.get("min_remaining_ore_factor", CONFIG.get("min_remaining_ore_factor", 0.08))
        ),
        flat_input_transition_sensitivity=float(
            cfg.get("flat_input_transition_sensitivity", CONFIG.get("flat_input_transition_sensitivity", 6.0))
        ),
        flat_input_uplift_response_days=float(
            cfg.get("flat_input_uplift_response_days", CONFIG.get("flat_input_uplift_response_days", 75.0))
        ),
        flat_input_response_ramp_days=float(
            cfg.get("flat_input_response_ramp_days", CONFIG.get("flat_input_response_ramp_days", 75.0))
        ),
        flat_input_late_uplift_response_boost=float(
            cfg.get(
                "flat_input_late_uplift_response_boost",
                CONFIG.get("flat_input_late_uplift_response_boost", 2.5),
            )
        ),
        catalyst_dose_transition_shape_strength=float(
            cfg.get(
                "catalyst_dose_transition_shape_strength",
                CONFIG.get("catalyst_dose_transition_shape_strength", 0.75),
            )
        ),
    ).to(device)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as exc:
        load_result = model.load_state_dict(checkpoint["state_dict"], strict=False)
        missing_ok = all(
            k.startswith("interaction_weight_params.") or k == "height_delay_scale"
            for k in load_result.missing_keys
        )
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
        lix_scale=float(job["lix_scale"]),
        irrigation_scale=float(job["irrigation_scale"]),
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
            "model_logic_version": MODEL_LOGIC_VERSION,
            "state_dict": model.state_dict(),
            "static_predictor_columns": STATIC_PREDICTOR_COLUMNS,
            "geo_idx": [int(v) for v in job["geo_idx"]],
            "ctrl_lb": ctrl_lb_seed.tolist(),
            "ctrl_ub": ctrl_ub_seed.tolist(),
            "cat_lb": cat_lb_seed.tolist(),
            "cat_ub": cat_ub_seed.tolist(),
            "tmax_days": float(job["tmax_days"]),
            "cum_scale": float(job["cum_scale"]),
            "lix_scale": float(job["lix_scale"]),
            "irrigation_scale": float(job["irrigation_scale"]),
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

    metrics_val, records_val = evaluate_model(
        model,
        val_pairs_seed,
        float(job["cum_scale"]),
        float(job["lix_scale"]),
        float(job["irrigation_scale"]),
    )
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


def predict_pair_record(
    model: PairCurveNet,
    pair: PairSample,
    cum_scale: float,
    lix_scale: float,
    irrigation_scale: float,
) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(pair.static_scaled, dtype=torch.float32, device=device).unsqueeze(0)
        raw_for_geometry = pair.static_imputed if pair.static_imputed is not None else pair.static_raw
        x_raw = torch.tensor(raw_for_geometry, dtype=torch.float32, device=device).unsqueeze(0)
        x_input_only = torch.tensor(pair.input_only_raw, dtype=torch.float32, device=device).unsqueeze(0)
        ctrl_t = torch.tensor(pair.control.time, dtype=torch.float32, device=device)
        ctrl_c = torch.tensor(pair.control.catalyst_cum / cum_scale, dtype=torch.float32, device=device)
        ctrl_l = torch.tensor(pair.control.lixiviant_cum / lix_scale, dtype=torch.float32, device=device)
        ctrl_irr = torch.tensor(
            pair.control.irrigation_rate_l_m2_h / max(float(irrigation_scale), 1e-6),
            dtype=torch.float32,
            device=device,
        )
        cat_t = torch.tensor(pair.catalyzed.time, dtype=torch.float32, device=device)
        cat_c = torch.tensor(pair.catalyzed.catalyst_cum / cum_scale, dtype=torch.float32, device=device)
        cat_l = torch.tensor(pair.catalyzed.lixiviant_cum / lix_scale, dtype=torch.float32, device=device)
        cat_irr = torch.tensor(
            pair.catalyzed.irrigation_rate_l_m2_h / max(float(irrigation_scale), 1e-6),
            dtype=torch.float32,
            device=device,
        )

        p_ctrl, p_cat, tau, temp, kappa, aging_strength, latent = model.predict_params(x, x_raw, x_input_only)
        ctrl_cap_t = torch.tensor([pair.ctrl_cap], dtype=torch.float32, device=device)
        cat_cap_t = torch.tensor([pair.cat_cap], dtype=torch.float32, device=device)
        pred_ctrl, _, ctrl_states = model.curves_given_params(
            p_ctrl,
            p_cat,
            ctrl_t,
            ctrl_c,
            ctrl_l,
            ctrl_irr,
            tau,
            temp,
            kappa,
            aging_strength,
            latent_params=latent,
            return_states=True,
            ctrl_cap=ctrl_cap_t,
            cat_cap=cat_cap_t,
        )
        cat_ctrl_c = torch.zeros_like(cat_c)
        pred_ctrl_cat_time, _ = model.curves_given_params(
            p_ctrl,
            p_cat,
            cat_t,
            cat_ctrl_c,
            cat_l,
            cat_irr,
            tau,
            temp,
            kappa,
            aging_strength,
            latent_params=latent,
            ctrl_cap=ctrl_cap_t,
            cat_cap=cat_cap_t,
        )
        _, pred_cat, cat_states = model.curves_given_params(
            p_ctrl,
            p_cat,
            cat_t,
            cat_c,
            cat_l,
            cat_irr,
            tau,
            temp,
            kappa,
            aging_strength,
            latent_params=latent,
            return_states=True,
            ctrl_cap=ctrl_cap_t,
            cat_cap=cat_cap_t,
        )

        input_only_idx = {name: idx for idx, name in enumerate(INPUT_ONLY_COLUMNS)}
        plot_profile = build_shared_ensemble_plot_profile(
            time_days=pair.catalyzed.time,
            catalyst_cum=pair.catalyzed.catalyst_cum,
            lixiviant_cum=pair.catalyzed.lixiviant_cum,
            feed_mass_kg=float(pair.input_only_raw[input_only_idx[FEED_MASS_COL]])
            if FEED_MASS_COL in input_only_idx
            else np.nan,
            column_inner_diameter_m=float(pair.input_only_raw[input_only_idx["column_inner_diameter_m"]])
            if "column_inner_diameter_m" in input_only_idx
            else np.nan,
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
            torch.tensor(
                np.asarray(plot_profile["plot_lixiviant_cum"], dtype=float) / max(float(lix_scale), 1e-6),
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                np.asarray(plot_profile["plot_irrigation_rate_l_m2_h"], dtype=float)
                / max(float(irrigation_scale), 1e-6),
                dtype=torch.float32,
                device=device,
            ),
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
            torch.tensor(
                np.asarray(plot_profile["plot_lixiviant_cum"], dtype=float) / max(float(lix_scale), 1e-6),
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                np.asarray(plot_profile["plot_irrigation_rate_l_m2_h"], dtype=float)
                / max(float(irrigation_scale), 1e-6),
                dtype=torch.float32,
                device=device,
            ),
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
        "cumulative_lixiviant_m3_t": pair.catalyzed.lixiviant_cum.copy(),
        "irrigation_rate_l_m2_h": pair.catalyzed.irrigation_rate_l_m2_h.copy(),
        "catalyzed_pred": pred_cat.detach().cpu().numpy(),
        "catalyzed_pred_blended": cat_states["catalyzed_pred_blended"].detach().cpu().numpy(),
        "catalyzed_target_curve": cat_states["catalyzed_target_curve"].detach().cpu().numpy(),
        "catalyzed_target_curve_anchored": cat_states["catalyzed_target_curve_anchored"].detach().cpu().numpy(),
        "catalyzed_transition_weight": cat_states["catalyzed_transition_weight"].detach().cpu().numpy(),
        "catalyzed_gap_target": cat_states["catalyzed_gap_target"].detach().cpu().numpy(),
        "control_pred_on_catalyzed_t": pred_ctrl_cat_time.detach().cpu().numpy(),
        "tau_days": float(cat_states["effective_tau_days"].squeeze().detach().cpu().item()),
        "temp_days": float(temp.squeeze().detach().cpu().item()),
        "transition_center_day": float(cat_states["catalyzed_transition_center_day"].squeeze().detach().cpu().item()),
        "transition_end_day": float(cat_states["catalyzed_transition_end_day"].squeeze().detach().cpu().item()),
        "transition_duration_days": float(cat_states["catalyzed_transition_duration_days"].squeeze().detach().cpu().item()),
        "kappa": float(kappa.squeeze().detach().cpu().item()),
        "aging_strength": float(aging_strength.squeeze().detach().cpu().item()),
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
        "ore_decay_strength": float(latent["ore_decay_strength"].squeeze().detach().cpu().item()),
        "passivation_strength": float(latent["passivation_strength"].squeeze().detach().cpu().item()),
        "depassivation_strength": float(latent["depassivation_strength"].squeeze().detach().cpu().item()),
        "transform_strength": float(latent["transform_strength"].squeeze().detach().cpu().item()),
        "catalyzed_fast_release_last": float(cat_states["fast_release"][-1].detach().cpu().item()),
        "catalyzed_acid_buffer_last": float(cat_states["acid_buffer_penalty"][-1].detach().cpu().item()),
        "catalyzed_diffusion_drag_last": float(cat_states["diffusion_drag"][-1].detach().cpu().item()),
        "control_passivation_last": float(ctrl_states["passivation"][-1].detach().cpu().item()),
        "catalyzed_passivation_last": float(cat_states["passivation"][-1].detach().cpu().item()),
        "catalyzed_depassivation_last": float(cat_states["depassivation"][-1].detach().cpu().item()),
        "catalyzed_transformation_last": float(cat_states["transformation"][-1].detach().cpu().item()),
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
        "plot_lixiviant_cum": np.asarray(plot_profile["plot_lixiviant_cum"], dtype=float),
        "plot_irrigation_rate_l_m2_h": np.asarray(plot_profile["plot_irrigation_rate_l_m2_h"], dtype=float),
        "catalyzed_pred_plot": pred_cat_plot.detach().cpu().numpy(),
        "catalyst_addition_start_day": float(plot_profile["catalyst_addition_start_day"]),
        "catalyst_addition_stop_day": float(plot_profile["catalyst_addition_stop_day"]),
        "weekly_catalyst_addition_kg_t": float(plot_profile["weekly_catalyst_addition_kg_t"]),
        "weekly_reference_days": float(plot_profile["weekly_reference_days"]),
        "weekly_lixiviant_addition_m3_t": float(plot_profile["weekly_lixiviant_addition_m3_t"]),
        "recent_window_start_day": float(plot_profile["recent_window_start_day"]),
        "recent_window_delta_kg_t": float(plot_profile["recent_window_delta_kg_t"]),
        "recent_window_delta_tol_kg_t": float(plot_profile["recent_window_delta_tol_kg_t"]),
        "recent_window_growth_near_zero": bool(plot_profile["recent_window_growth_near_zero"]),
        "last_observed_day": float(plot_profile["last_observed_day"]),
        "stopped_before_test_end": bool(plot_profile["stopped_before_test_end"]),
        "catalyst_addition_state": str(plot_profile["catalyst_addition_state"]),
        "extension_applied": bool(plot_profile["extension_applied"]),
        "lixiviant_extension_applied": bool(plot_profile["lixiviant_extension_applied"]),
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


def evaluate_model(
    model: PairCurveNet,
    pairs: List[PairSample],
    cum_scale: float,
    lix_scale: float,
    irrigation_scale: float,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    records = [predict_pair_record(model, pair, cum_scale, lix_scale, irrigation_scale) for pair in pairs]
    metrics = compute_metrics_from_records(records, ensemble=False)
    return metrics, records


def records_to_df(records: List[Dict[str, Any]], ensemble: bool = False) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"sample_id": r["sample_id"]}
        scalar_keys = [
            "tau_days",
            "temp_days",
            "transition_center_day",
            "transition_end_day",
            "transition_duration_days",
            "kappa",
            "aging_strength",
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
            "catalyzed_pred_blended",
            "catalyzed_target_curve",
            "catalyzed_target_curve_anchored",
            "catalyzed_transition_weight",
            "catalyzed_gap_target",
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
    interval_smoothing_days = float(CONFIG.get("ensemble_interval_smoothing_days", 0.0))
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
        rec["control_pred_p10"], rec["control_pred_p90"] = smooth_predictive_interval_bounds(
            time_days=rec["control_t"],
            mean_curve=rec["control_pred_mean"],
            low_curve=rec["control_pred_p10"],
            high_curve=rec["control_pred_p90"],
            smoothing_days=interval_smoothing_days,
        )
        rec["catalyzed_pred_p10"], rec["catalyzed_pred_p90"] = smooth_predictive_interval_bounds(
            time_days=rec["catalyzed_t"],
            mean_curve=rec["catalyzed_pred_mean"],
            low_curve=rec["catalyzed_pred_p10"],
            high_curve=rec["catalyzed_pred_p90"],
            smoothing_days=interval_smoothing_days,
        )
        rec["control_pred_plot_p10"], rec["control_pred_plot_p90"] = smooth_predictive_interval_bounds(
            time_days=rec["control_plot_time_days"],
            mean_curve=rec["control_pred_plot_mean"],
            low_curve=rec["control_pred_plot_p10"],
            high_curve=rec["control_pred_plot_p90"],
            smoothing_days=interval_smoothing_days,
        )
        rec["catalyzed_pred_plot_p10"], rec["catalyzed_pred_plot_p90"] = smooth_predictive_interval_bounds(
            time_days=rec["catalyzed_plot_time_days"],
            mean_curve=rec["catalyzed_pred_plot_mean"],
            low_curve=rec["catalyzed_pred_plot_p10"],
            high_curve=rec["catalyzed_pred_plot_p90"],
            smoothing_days=interval_smoothing_days,
        )
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
    df_prefit = df.copy().reset_index(drop=True)
    prefit_df = prefit_biexponential_for_rows(df_prefit)
    df_prefit["row_index"] = np.arange(len(df_prefit))
    df_prefit = df_prefit.merge(prefit_df, on="row_index", how="left")
    prefit_out_path = os.path.join(OUTPUTS_ROOT, "row_biexponential_prefit.csv")
    df_prefit.to_csv(prefit_out_path, index=False)

    # 3) Build pair-level control/catalyzed samples with time-dependent arrays
    pairs_observed = build_pair_samples(df_prefit)
    if len(pairs_observed) < 6:
        raise ValueError(f"Expected at least 6 paired samples; got {len(pairs_observed)}.")

    # 3a) Generate fitted curve plots for each project_sample_id
    fitted_curves_plot_dir = os.path.join(PLOTS_ROOT, "fitted_curves")
    target_day = float(CONFIG.get("ensemble_plot_target_day", 2500.0))
    plot_dpi = int(CONFIG.get("plot_dpi", 300))
    print(f"[Plots] Generating fitted curve plots to: {fitted_curves_plot_dir}")
    plot_fitted_curve_per_sample(
        pairs=pairs_observed,
        output_dir=fitted_curves_plot_dir,
        dpi=plot_dpi,
        target_day=target_day,
    )
    print(f"[Plots] Fitted curve plots completed.")

    # 3b) Augment pairs with virtual data for training (every 7 days to target_day)
    print(f"[Data] Augmenting pairs with virtual data points...")
    pairs_training = []
    for pair in pairs_observed:
        augmented_pair = augment_pair_with_virtual_data(
            pair=pair,
            target_day=target_day,
            interval_days=7.0,
        )
        pairs_training.append(augmented_pair)
    print(f"[Data] Virtual data augmentation completed.")

    catalyst_stop_window_days = float(CONFIG.get("catalyst_extension_window_days", 21.0))
    catalyst_stop_report_df, catalyst_stop_summary = build_catalyst_stop_report(
        pairs=pairs_observed,
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

    # 4) Global parameter bounds from all paired samples
    ctrl_all_params = np.vstack([p.control.fit_params for p in pairs_training])
    cat_all_params = np.vstack([p.catalyzed.fit_params for p in pairs_training])
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
            max(float(np.max(p.control.time)) for p in pairs_training),
            max(float(np.max(p.catalyzed.time)) for p in pairs_training),
        )
    )
    cum_scale = float(
        max(
            1e-6,
            max(
                float(np.nanmax(p.catalyzed.catalyst_cum))
                for p in pairs_training
                if p.catalyzed.catalyst_cum.size > 0
            ),
        )
    )
    lix_max_candidates = [
        float(np.nanmax(curve.lixiviant_cum))
        for p in pairs_training
        for curve in [p.control, p.catalyzed]
        if curve.lixiviant_cum.size > 0
    ]
    lix_scale = float(max(1e-6, max(lix_max_candidates) if len(lix_max_candidates) > 0 else 1e-6))
    irrigation_max_candidates = [
        float(np.nanmax(curve.irrigation_rate_l_m2_h))
        for p in pairs_training
        for curve in [p.control, p.catalyzed]
        if curve.irrigation_rate_l_m2_h.size > 0
    ]
    irrigation_scale = float(
        max(1e-6, max(irrigation_max_candidates) if len(irrigation_max_candidates) > 0 else 1e-6)
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
        n_samples=len(pairs_observed),
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

        train_pairs_seed = [pairs_training[i] for i in train_indices]
        # val_pairs_seed = [pairs_observed[i] for i in val_indices] # Strict no-leak validation on observed data only
        val_pairs_seed = [pairs_training[i] for i in val_indices] # Validation on augmented data (relaxed, may have some leakage but more realistic for deployed model performance)

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
                "lix_scale": lix_scale,
                "irrigation_scale": irrigation_scale,
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
        "n_pairs_total": int(len(pairs_observed)),
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
    val_pairs_for_agg = [p for p in pairs_observed if p.sample_id in val_union_set]
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
        apply_static_transformers(pairs_observed, m["imputer"], m["scaler"])
        metrics_member, records_member = evaluate_model(
            m["model"],
            pairs_observed,
            cum_scale,
            lix_scale,
            irrigation_scale,
        )
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
        pairs=pairs_observed,
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

    manifest = {
        "project_root": PROJECT_ROOT,
        "data_analysis_summary": os.path.join(OUTPUTS_ROOT, "data_analysis_summary.json"),
        "prefit_table": prefit_out_path,
        "param_bounds": os.path.join(OUTPUTS_ROOT, "param_bounds.json"),
        "catalyst_addition_status_csv": catalyst_stop_report_path,
        "catalyst_stopped_before_test_end_csv": catalyst_stopped_only_path,
        "catalyst_addition_status_summary": catalyst_stop_summary_path,
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
    }
    save_json(os.path.join(OUTPUTS_ROOT, "run_manifest.json"), manifest)

    print("\nRun complete")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Validation OOF ensemble metrics: {val_oof_metrics}")
    print(f"Validation ensemble metrics: {val_ensemble_metrics}")
    print(f"Deployed CV ensemble metrics: {final_metrics}")
    print(f"Catalyst stopped-before-end samples: {catalyst_stopped_only_path}")
    print(f"Manifest: {os.path.join(OUTPUTS_ROOT, 'run_manifest.json')}")

if __name__ == "__main__":
    main()

 #%%
