
# %%
import os
import ast
import json
import random
import re
import shutil
import sys
import math
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
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
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit, differential_evolution
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ---------------------------
# PyTorch Setup
# ---------------------------
def torch_mps_is_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def resolve_torch_device() -> torch.device:
    requested = str(os.environ.get("ROSETTA_TORCH_DEVICE", "")).strip().lower()
    if requested:
        alias = {
            "gpu": "cuda",
            "metal": "mps",
        }.get(requested, requested)
        if alias == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("ROSETTA_TORCH_DEVICE=cuda was requested but CUDA is not available.")
            return torch.device("cuda")
        if alias == "mps":
            if not torch_mps_is_available():
                raise RuntimeError("ROSETTA_TORCH_DEVICE=mps was requested but MPS is not available.")
            return torch.device("mps")
        if alias == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unsupported ROSETTA_TORCH_DEVICE value: {requested}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch_mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_torch_dtype(device: torch.device) -> torch.dtype:
    requested = str(os.environ.get("ROSETTA_TORCH_DTYPE", "")).strip().lower()
    dtype_aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if requested:
        dtype = dtype_aliases.get(requested)
        if dtype is None:
            supported = ", ".join(sorted(dtype_aliases))
            raise ValueError(f"Unsupported ROSETTA_TORCH_DTYPE value: {requested}. Supported values: {supported}")
        if device.type == "cpu" and dtype != torch.float32:
            return torch.float32
        return dtype
    return torch.float32


def torch_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype).replace("torch.", "")


def tensor_to_numpy_float32(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(dtype=torch.float32).cpu().numpy()


def model_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def reset_generated_dir(dir_path: str) -> None:
    """Remove a generated artifact directory so stale files never mix with a new run."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


# device = resolve_torch_device()
device = torch.device("cpu")


DEVICE_IS_ACCELERATOR = device.type in {"cuda", "mps"}
MODEL_TORCH_DTYPE = resolve_torch_dtype(device)
CHECKPOINT_MAP_LOCATION = torch.device("cpu")

if DEVICE_IS_ACCELERATOR:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

if device.type == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# ---------------------------
# Paths and Data
# ---------------------------
DEFAULT_DATA_PATH = (
    "/Users/administration/Library/CloudStorage/OneDrive-JettiResources/"
    "PythonProjects/Rosetta/pipeline/outputs/leaching_performance_weekly.csv"
)
DATA_PATH = os.environ.get("ROSETTA_DATA_PATH", DEFAULT_DATA_PATH)

TIME_COL_COLUMNS = "leach_duration_days"
TARGET_COLUMNS = "cu_recovery_%"
# v11: project_col_id is now used only for column-level filtering/exclusion.
# catalyst_status is the authoritative source for Control vs Catalyzed classification.
# "no_catalyst" → Control; "with_catalyst" → Catalyzed.
STATUS_COL_PRIMARY = "catalyst_status"
STATUS_COL_FALLBACK = "project_col_id"   # legacy fallback only
COL_ID_COL = "project_col_id"            # individual column identifier (1 row per col)
PAIR_ID_COL = "project_sample_id"        # ore-sample identifier (many cols per sample)
PROJECT_NAME_COL = "project_name"
# v12: catalyst_addition_mg_l is the preferred feed-dosage source when present.
# Cumulative catalyst/lixiviant profiles remain the fallback for catalyzed rows
# whose explicit dosage vector is unavailable.
CATALYST_START_DAY_COL = "catalyst_start_days_of_leaching"
TRANSITION_TIME_COL = "transition_time"
CATALYST_CUM_COL      = "cumulative_catalyst_addition_kg_t"  # PRIMARY trusted source
LIXIVIANT_CUM_COL     = "cumulative_lixiviant_m3_t"
ALIGNED_TIME_COL = "leach_duration_days_aligned"
CATALYST_ADDITION_COL = "catalyst_addition_mg_l"
CATALYST_ADDITION_RECON_COL = "catalyst_addition_mg_l_reconstructed"
IRRIGATION_RATE_RECON_COL = "irrigation_rate_l_m2_h_reconstructed"
ORP_PROFILE_COL       = "feed_orp_mv_ag_agcl"     # feed (input) ORP
PLS_ORP_PROFILE_COL   = "pls_orp_mv_ag_agcl"      # PLS (output) ORP – learned decay signal
FEED_MASS_COL         = "feed_mass_kg"
# Column-bed porosity fraction used by the CSTR pore-volume model.
# Typical heap/column leach bed porosity for crushed sulphide ore.
COLUMN_POROSITY: float = 0.35
# Excluded project_sample_ids (mineralogy / recovery data issues):
EXCLUDED_TRAIN_PAIR_IDS = {
    # Explicitly excluded by user (data quality / project scope):
    "002_jetti_project_file_qb",
    "004_jetti_project_file_mo",
    "004_jetti_project_file_mols",
    "006_jetti_project_file_pvls",
    "012_jetti_project_file_incremento",
    "012_jetti_project_file_quebalix",
    "015_jetti_project_file_amcf",
    "01a_jetti_project_file_c",
    "020_jetti_project_file_hypogene_supergene_super",
    "020_jetti_project_file_hardy_and_waste_h21_master_comp",
    "020_jetti_project_file_hypogene_supergene_hypogene_master_composite",
    "021_jetti_project_file_hypogene",
    "023_jetti_project_file_ea_1",
    "023_jetti_project_file_ea_2",
    "023_jetti_project_file_ea_3",
    "023_jetti_project_file_ea_4",
    "jetti_project_file_elephant_site_s3",
    "028_jetti_project_file_andesite",
    "028_jetti_project_file_monzonite",
    "030_jetti_project_file_cpy",
    "030_jetti_project_file_ss",
    "jetti_project_file_tiger_rom_m1",
    "jetti_project_file_tiger_rom_m2",
    "jetti_project_file_tiger_rom_m3",
    "jetti_project_file_toquepala_scl_sample_antigua"

}
# K-fold grouping aliases: samples whose group label should be treated as the
# same group during RepeatedGroupKFold. Use this to tie together related pairs
# (e.g., the same ore at different particle sizes) so they always land in the
# same fold together — either both in validation or both in training.
# The keys are the actual project_sample_id values found in the dataset; the
# values are the shared group label used only for CV splitting.
KFOLD_GROUP_ALIAS: Dict[str, str] = {
    "015_jetti_project_file_amcf_6in": "amcf_head",
    "015_jetti_project_file_amcf_8in": "amcf_head",
    "003_jetti_project_file_amcf_head": "amcf_head",
    "006_jetti_project_file_pvo": "amcf_head",
    "jetti_file_elephant_ii_ugm2": "elephant_ugm2",
    "jetti_file_elephant_ii_ugm2_coarse": "elephant_ugm2",

}
# Excluded individual column ids (project_col_id level):
EXCLUDED_TRAIN_COL_IDS: set = {
    # Add specific project_col_id strings here to exclude individual columns
    # independently of their project_sample_id.
    "003_jetti_project_file_oxide_columns_beo_1", 
    "003_jetti_project_file_oxide_columns_beo_2", 
    "003_jetti_project_file_oxide_columns_beo_3", 
    "003_jetti_project_file_oxide_columns_beo_4", 
    "003_jetti_project_file_oxide_columns_beo_5",
    "003_jetti_project_file_oxide_columns_beo_6",
    "003_jetti_project_file_oxide_columns_beo_7",
    "003_jetti_project_file_oxide_columns_beo_8",
    "003_jetti_project_file_be_8",
    "jetti_file_elephant_ii_ver_2_pq_pr_1",
    "jetti_file_elephant_ii_ver_2_pq_pr_2",
    "jetti_project_file_leopard_scl_col1",
    "jetti_project_file_leopard_scl_col2",
    "jetti_project_file_zaldivar_scl_col68",
    "jetti_file_elephant_ii_ver_2_ugm_uc_5",
    "jetti_project_file_elephant_scl_col53", # further check with Monse

}

DEFAULT_PROJECT_ROOT = (
    "/Users/administration/Library/CloudStorage/OneDrive-JettiResources/"
    "PythonProjects/Rosetta/NN_Pytorch_ExpEq_columns_only_v12"
)
LOCAL_PROJECT_ROOT = os.path.join(THIS_DIR, "NN_Pytorch_ExpEq_columns_only_v12")
PROJECT_ROOT_ENV = str(os.environ.get("ROSETTA_PROJECT_ROOT", "")).strip()
PROJECT_ROOT = PROJECT_ROOT_ENV or LOCAL_PROJECT_ROOT


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


for candidate in ([PROJECT_ROOT_ENV] if PROJECT_ROOT_ENV else [LOCAL_PROJECT_ROOT, DEFAULT_PROJECT_ROOT]):
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
    # v11: cumulative_catalyst_addition_kg_t is REMOVED as a predictor.
    # The catalyst effect is driven by dynamic signals reconstructed internally
    # from cumulative_catalyst_addition_kg_t and cumulative_lixiviant_m3_t.
    "leach_duration_days": ["Leach Duration (days)", "numerical", 1],
    "cumulative_lixiviant_m3_t": ["Cumulative Lixiviant added (m3/t)", "numerical", 1],
    "acid_soluble_%": ["Acid Soluble Cu (%)", "numerical", 1],
    "cyanide_soluble_%": ["Cyanide Soluble (%)", "numerical", 0],
    "residual_cpy_%": ["Residual Chalcopyrite (%)", "numerical", 0],
    "material_size_p80_in": ["Material Size P80 (in)", "numerical", -1],
    # Optional lixiviant-conditioning predictors. These are present in the CSV
    # for most projects and materially improve early kinetics when the leach
    # starts with an already-oxidizing ferric solution (e.g. Toquepala fresca,
    # secondary sulfide systems). Missing values are imputed with dataset-level
    # fallbacks before the feature-vector validation step.
    "lixiviant_initial_fe_mg_l": ["Initial Lixiviant Fe (mg/L)", "numerical", 0],
    "lixiviant_initial_ph": ["Initial Lixiviant pH", "numerical", 0],
    "lixiviant_initial_orp_mv": ["Initial Lixiviant ORP (mV)", "numerical", 0],
    # "grouped_copper_sulfides": ["Copper Sulphides (%)", "numerical", 0],
    # "grouped_secondary_copper": ["Secondary Copper (%)", "numerical", 0],
    # "grouped_primary_copper_sulfides": ["Primary Cu Sulphides (%)", "numerical", 0],
    # "grouped_secondary_copper_sulfides": ["Secondary Cu Sulphides (%)", "numerical", 0],
    # "grouped_copper_oxides": ["Copper Oxides (%)", "numerical", 1],
    # "grouped_mixed_copper_ores": ["Mixed Copper Ores (%)", "numerical", 0],
    "grouped_acid_generating_sulfides": ["Acid Generating Sulphides (%)", "numerical", 0],
    "grouped_phosphate_minerals": ["Phosphate Minerals (%)", "numerical", 0],
    # "grouped_gangue_sulfides": ["Gangue Sulphides (%)", "numerical", 0], # excluded only because most of the values are zero.
    # "grouped_gangue_silicates": ["Gangue Silicates (%)", "numerical", 0], # try this, working perfectly without it
    "grouped_fe_oxides": ["Fe Oxides (%)", "numerical", 0], # try this, working perfectly without it
    "grouped_carbonates": ["Carbonates (%)", "numerical", 0],
    # "grouped_accessory_minerals": ["Accessory Minerals (%)", "numerical", 0],
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
# v11: STATIC predictors exclude all dynamic time-varying signals
_DYNAMIC_COLS = {TIME_COL_COLUMNS, CATALYST_CUM_COL, LIXIVIANT_CUM_COL}
STATIC_PREDICTOR_COLUMNS = [
    c for c in PREDICTOR_COLUMNS if c not in _DYNAMIC_COLS
] + ["terminal_slope_rate"]
# v12: terminal_slope_rate is a derived feature computed from the observed
# recovery time-series (mean daily recovery rate over the last 20 % of test
# duration, %/day).  It is NOT a CSV column — it is appended to static_vec
# at build time in build_pair_samples.  A near-zero terminal rate signals that
# the leach cap is already close to the current observed recovery, giving the
# model a direct empirical anchor on the asymptote for refractory ores.
# Features listed here are EXCLUDED from the CSV column-presence check in
# resolve_dataset_usecols / REQUIRED_DATASET_COLUMNS.
_COMPUTED_STATIC_FEATURES: set = {"terminal_slope_rate"}
CSV_STATIC_PREDICTOR_COLUMNS = [
    c for c in STATIC_PREDICTOR_COLUMNS if c not in _COMPUTED_STATIC_FEATURES
]
OPTIONAL_STATIC_PREDICTOR_COLUMNS: set = {
    "lixiviant_initial_fe_mg_l",
    "lixiviant_initial_ph",
    "lixiviant_initial_orp_mv",
}
REQUIRED_STATIC_PREDICTOR_COLUMNS = [
    c for c in CSV_STATIC_PREDICTOR_COLUMNS if c not in OPTIONAL_STATIC_PREDICTOR_COLUMNS
]
INPUT_ONLY_COLUMNS = list(INPUT_ONLY_HEADERS_DICT_COLUMNS.keys())
USER_INPUT_COLUMNS = STATIC_PREDICTOR_COLUMNS + INPUT_ONLY_COLUMNS
STATIC_PREDICTOR_INDEX = {name: idx for idx, name in enumerate(STATIC_PREDICTOR_COLUMNS)}
INPUT_ONLY_INDEX = {name: idx for idx, name in enumerate(INPUT_ONLY_COLUMNS)}
CURVE_SPECIFIC_STATIC_OVERRIDE_COLUMNS = [
    "lixiviant_initial_fe_mg_l",
    "lixiviant_initial_orp_mv",
    "lixiviant_initial_ph",
    "terminal_slope_rate",
]
CURVE_SPECIFIC_STATIC_OVERRIDE_INDEX = {
    name: idx for idx, name in enumerate(CURVE_SPECIFIC_STATIC_OVERRIDE_COLUMNS)
}

RAW_DATASET_COLUMNS_EXCLUDED = {
    "catalyst_dosage_mg_l",
    "irrigation_rate_l_m2_h",
    "irrigation_rate_l_h_m2",
    "cumulative_lixiviant_flowthrough_l",
}
REQUIRED_DATASET_COLUMNS = (
    # v12: exclude computed (non-CSV) features from the required-column check.
    # terminal_slope_rate is derived from the time-series at build time — it is
    # never a column in the input CSV.
    set(REQUIRED_STATIC_PREDICTOR_COLUMNS)
    | set(INPUT_ONLY_COLUMNS)
    | {
        COL_ID_COL,
        PAIR_ID_COL,
        TIME_COL_COLUMNS,
        TARGET_COLUMNS,
        CATALYST_CUM_COL,
        LIXIVIANT_CUM_COL,
    }
)
OPTIONAL_DATASET_COLUMNS = {
    PROJECT_NAME_COL,
    STATUS_COL_PRIMARY,
    STATUS_COL_FALLBACK,
    CATALYST_START_DAY_COL,
    TRANSITION_TIME_COL,
    CATALYST_ADDITION_COL,
    ORP_PROFILE_COL,
    PLS_ORP_PROFILE_COL,
    *OPTIONAL_STATIC_PREDICTOR_COLUMNS,
}


def resolve_dataset_usecols(csv_path: str) -> List[str]:
    available_columns = list(pd.read_csv(csv_path, sep=",", nrows=0).columns)
    desired_columns = REQUIRED_DATASET_COLUMNS | OPTIONAL_DATASET_COLUMNS
    usecols = [
        col for col in available_columns
        if col in desired_columns and col not in RAW_DATASET_COLUMNS_EXCLUDED
    ]
    missing_required = sorted(REQUIRED_DATASET_COLUMNS.difference(usecols))
    if missing_required:
        raise ValueError(
            "Loaded dataset is missing required columns after schema filtering: "
            + ", ".join(missing_required)
        )
    return usecols

GEO_PRIORITY_COLUMNS = ["material_size_p80_in"]
CHEMISTRY_INTERACTION_COLUMNS = [
    c for c in STATIC_PREDICTOR_COLUMNS if c not in set(GEO_PRIORITY_COLUMNS)
]
# Raw predictor columns remain available to the encoder, but they are blocked
# from entering the recovery heads again through the explicit interaction mixer.
DIRECT_INTERACTION_TERM_NAMES = (
    (set(STATIC_PREDICTOR_COLUMNS) - {
        "copper_primary_sulfides_equivalent",
        "copper_secondary_sulfides_equivalent",
        "copper_oxides_equivalent",
        "fe:cu",
    })
    | set(INPUT_ONLY_COLUMNS)
    | {
    "apparent_bulk_density_t_m3",
    "material_size_to_column_diameter_ratio",
    "column_height_m",
    "column_inner_diameter_m",
    "chem_raw",
    }
)

CONFIG = {
    "seed": 2026,
    "cv_n_splits": 5,
    # 10 split-seeds x 10 repeats x 5 folds = 500 ensemble members.
    "cv_n_repeats": 10,
    "cv_n_split_seeds": 10,
    "cv_random_state": 2026,
    "cv_member_seed_base": 10000,
    # 0 = auto-compute based on dataset size and CPU count.
    "prefit_parallel_workers": 0,
    "prefit_min_rows_per_worker": 12,
    "prefit_parallel_chunks_per_worker": 4,
    "cv_parallel_workers": 10, # max(1, int(os.cpu_count() or 1)),
    # 0 = auto-compute based on worker count.
    "torch_threads_per_worker": 0,
    "torch_interop_threads_per_worker": 1,
    "epochs": 600,
    "patience": 80,
    "bootstrap_train_pairs": False,
    "learning_rate": 7.0e-3,
    "weight_decay": 2.0e-4,
    "grad_clip_norm": 5.0,
    "pair_batch_size": 2,
    "eval_every_n_epochs": 1,
    # Cap CPU-side orchestration threads for MPS-backed training.
    # Set to 0 to disable the cap.
    "mps_torch_threads_cap": 6,
    "hidden_dim": 48,
    "dropout": 0.15,
    "min_transition_days": 4.0,
    "max_transition_days": 220.0,
    "max_cat_slope_per_day": 0.16,
    "max_catalyst_aging_strength": 5.0,
    "late_tau_impact_decay_strength": 0.22,
    "min_remaining_ore_factor": 0.05,
    "flat_input_transition_sensitivity": 0.80,
    "flat_input_uplift_response_days": 18.0,
    "flat_input_response_ramp_days": 24.0,
    "flat_input_late_uplift_response_boost": 0.15,
    "control_early_end_day": 180.0,
    "control_tail_start_day": 365.0,
    "control_fit_value_min_relative_scale": 0.20,
    "control_fit_value_exponent": 0.80,
    "control_weight_early_boost": 1.35,
    "control_weight_tail_boost": 0.45,  # keep extra tail focus, but closer to the v10 baseline of 0.35
    "control_process_end_day": 180.0,
    "control_process_passivation_weight": 0.45,
    "control_process_acid_buffer_weight": 0.30,
    "control_process_diffusion_drag_weight": 0.25,
    "loss_weights": {
        "gap":                      1.0,
        # Raised from 0.02 -> 0.10 to enforce strictly non-decreasing recovery
        # curves (recovery should never go down in time).
        "monotonic":                0.10,
        "param":                    0.12,
        "smooth_cat":               0.24,
        "slope_cap":                0.28,
        "latent_smooth":            0.05,
        "latent_cat_rate":          0.50,   # was 0.65 (v3), 1.0 (v2) — back to v1
        "flat_input_smooth":        0.08,
        "flat_input_accel":         0.06,
        "cap":                      1.50,    # was 3.0 — back to v1 original
        "uplift_fit":               1.20,   # was 1.40 (v3), 1.80 (v2) — back to v1
        "uplift_tail":              1.50,   # was 1.70 (v3), 2.20 (v2) — back to v1
        "final_recovery":           0.55,
        "final_uplift":             0.85,
        "catalyst_use_prior":       0.05,
        "control_interp_fit":       0.90,
        "control_early_fit":        1.15,   # was 1.25 (v3), 1.50 (v2) — back to v1
        "control_tail_fit":         0.55,   # was 0.65 (v3), 0.90 (v2) — back to v1
        "pre_catalyst_control_fit": 1.00,
        "control_early_process":    0.25,
        "tau_onset":                0.50,
        "single_uplift_late_accel": 0.02,   # keep (was already improved in v2/v3)
        "ferric_orp_aux":           0.10,
        "primary_fraction_prior":   0.06,
        # (D) Early-window chord-slope + t50 shape penalties.
        # Keep them as very light nudges only. The first v12 pass already
        # softened v11, but these priors still pushed the fit too hard toward
        # globally "clean" early kinetics instead of letting refractory ores
        # stay slow. Mineralogy gating remains below in the loss.
        "slope_early":              0.14,
        "t50_match":                0.04,
    },
    "plot_dpi": 300,
    "ensemble_pi_low": 10,
    "ensemble_pi_high": 90,
    "ensemble_plot_step_days": 50.0,
    "ensemble_plot_target_day": 2500.0,
    "prefit_min_amplitude": 2.5,
    "prefit_cap_target_penalty_weight": 1.0,
    "prefit_cap_target_soft_margin": 2.0,
    "prefit_cap_target_margin_fraction": 0.05,
    "use_data_informed_prefit_caps": True,
    "final_day_oxide_leach_fraction": 0.95,
    "final_day_secondary_leach_fraction_min": 0.45,
    "final_day_secondary_leach_fraction_max": 0.70,
    "prefit_asymptote_target_day": 2500.0,
    "prefit_target_day_min_asymptote_frac": 0.95,
    "prefit_target_day_max_slope_pct_per_day": 0.00002, # 0.002%/day max slope = 0.002*500=1.0% increase in 500 days
    "prefit_target_day_penalty_weight": 1.0,
    # Catalyzed-only slope enforcement at day 2500.
    # The catalyst accelerates chalcopyrite leaching, so by day 2500 most of the
    # leachable primary sulfide should have dissolved.  Apply a stronger penalty
    # on the day-2500 slope for catalyzed columns to force the prefit curve to
    # flatten regardless of where the asymptote lands relative to the chemistry cap.
    # NOTE: this does NOT force the curve to reach the maximum metallurgical cap —
    # it only enforces near-flatness.  The asymptote can settle anywhere below cap.
    "prefit_cat_target_day_penalty_weight": 6.0,
    # Allow catalyzed columns to have faster b1/b2 kinetic rate constants than the
    # P80-derived upper bound alone would permit.  The catalyst directly accelerates
    # sulphide dissolution kinetics (higher effective reactivity at the mineral
    # surface), so the steep early-kinetic slopes seen in high-dose columns (e.g.
    # 003 AMCF head) are physically real.  Boost is clamped to [1.0, 3.0].
    "prefit_cat_p80_rate_upper_boost": 1.5,
    # When True, the chemistry-derived cap acts as a hard UPPER BOUND for catalyzed
    # columns (asymptote cannot exceed cap) rather than a two-sided target that
    # pulls the asymptote UP toward the cap.  This prevents over-prediction for
    # short tests where the data does not yet support the full metallurgical ceiling.
    "prefit_cat_cap_as_upper_bound_only": True,
    # When True, the day-2500 "shortfall" component (curve must be at 95% of its
    # asymptote by day 2500) is disabled for catalyzed columns.  Only the slope
    # constraint (curve must be near-flat at day 2500) remains active.  Removing
    # the shortfall decouples the asymptote height from the flatness requirement:
    # the curve finds its natural level from the data and is then forced flat there,
    # rather than being pulled both upward (by the cap) and outward (by shortfall).
    "prefit_cat_disable_shortfall_penalty": True,
    # When True, skip the catalyzed re-fit in apply_duplicate_operational_prefit_targets.
    # Set to False so catalyzed columns get same-dose convergence and ctrl-floor enforcement
    # without being pulled toward the chemistry ceiling.  The re-fit now uses fit_asymptote
    # (data-driven) as the base target rather than fit_sample_cap.
    "prefit_cat_skip_duplicate_targets": False,
    # When True, use each column's own data-driven fit_asymptote (rather than the
    # chemistry-ceiling fit_sample_cap) as the base target when re-fitting catalyzed
    # duplicate groups.  This prevents short-test over-prediction.
    "prefit_cat_use_asymptote_as_target": True,
    # "minimum" = only pull the asymptote UP if it's below the shared dose target;
    # columns already above are left untouched.  "exact" would force all members
    # to converge on a single number, potentially pulling good fits downward.
    "prefit_cat_duplicate_target_mode": "minimum",
    # Weight applied to the day-2500 recovery target during the catalyzed re-fit.
    # Lower than the original catalyst_weight (3.0) so the data shape is preserved.
    "prefit_cat_duplicate_target_weight": 2.0,
    # Minimum percentage-point uplift the catalyzed curve must have over the
    # median control fit_asymptote at day 2500.  Enforces the physical expectation
    # that catalyst always accelerates chalcopyrite leaching above the baseline.
    "prefit_cat_min_uplift_over_ctrl_pct": 2.0,
    # --- Tail-anchor weight boost (applies to ALL columns, not catalyzed-only) ---
    # Number of days from the END of each test's data window to treat as the
    # "tail anchor" region.  Points in this window receive a boosted fitting weight
    # to anchor the extrapolation start point to the observed late-test trend.
    "prefit_tail_anchor_days": 50.0,
    # Multiplicative boost applied to tail-window point weights.  1.0 = no change.
    "prefit_tail_anchor_boost": 3.0,
    "prefit_virtual_rebase_enabled": False,
    "prefit_virtual_rebase_recovery_threshold_pct": 5.0,
    "prefit_virtual_rebase_min_points": 6,
    "prefit_sigmoid_gate_enabled": True,
    "prefit_sigmoid_gate_min_p80_in": 0.0,
    "prefit_sigmoid_gate_min_column_height_m": 0.0,
    "prefit_sigmoid_gate_trigger_recovery_pct": 5.0,
    "prefit_sigmoid_gate_min_points_after_trigger": 6,
    # Lowered from 3 → 1: allows gate detection even when only 1 data point exists
    # before the recovery crosses the trigger threshold (fixes 015 amcf onset detection).
    "prefit_sigmoid_gate_min_initial_points": 1,
    # Raised from 1.0 → 4.5 %: initial points at 3–4 % recovery (typical for coarse
    # low-grade chalcopyrite columns like 015 amcf) now count as part of the
    # "near-zero prefix" required to activate the sigmoid gate.
    "prefit_sigmoid_gate_low_recovery_pct": 4.5,
    # Configurable minimum fraction of pre-threshold points that must be ≤
    # prefit_sigmoid_gate_low_recovery_pct.  Replaces the previously hardcoded 0.50.
    # Lowered to 0.25 so that the gate can trigger when only a quarter of the early
    # data qualifies as "low", which is sufficient evidence of an onset delay.
    "prefit_sigmoid_gate_min_low_prefix_frac": 0.25,
    # Raised from 0.03 → 0.05: the gated fit must be 5 % better RMSE than the plain
    # biexponential to be accepted.  The tighter threshold prevents the optimizer
    # from latching onto an artificially sharp gate when a smooth biexponential fits
    # the gradual-onset data (e.g. 015 amcf) nearly as well.
    "prefit_sigmoid_gate_min_improvement_fraction": 0.05,
    "prefit_sigmoid_gate_max_rmse_regression_fraction": 0.02,
    "prefit_sigmoid_gate_min_width_days": 7.0,
    # Set to 200 days (previously 220, briefly 130).  220 was too wide and let the
    # optimizer choose slow gradual gates for 022 / 024 (visually sharp onset).
    # 130 was too narrow and forced an artificially steep gate on 015 amcf whose
    # onset is genuinely gradual (~300–400 days).  200 is a balanced upper bound:
    # narrow enough to capture sharp onsets, wide enough for gradual ores.
    "prefit_sigmoid_gate_max_width_days": 200.0,
    # Gate models the INITIAL column startup delay (acid percolation / wetting),
    # NOT mid-curve kinetic phase changes.  Multi-phase kinetics are handled by
    # the biexponential a2/b2 term.  Keep the gate onset window short.
    "model_initial_gate_max_mid_day": 120.0,
    "model_initial_gate_min_width_days": 7.0,
    "model_initial_gate_max_width_days": 220.0,
    "use_duplicate_operational_prefit_targets": True,
    # Increased from 1.0 → 3.0: gives the shared-target penalty three times as much
    # pull relative to the per-point data WMSE.  Needed to force same-condition
    # control curves (ugm2 uc_1/uc_2/uc_3) and same-dose catalyzed curves (pq
    # pc_3–pc_7) to converge to a single day-2500 asymptote even when the individual
    # data anchors have slightly different values.
    "prefit_duplicate_day2500_target_weight": 3.0,
    "prefit_catalyst_monotonic_day2500_target_weight": 3.0,
    # When > 0, catalyzed columns whose cumulative catalyst doses (kg/t) lie within
    # this percentage of each other are treated as the SAME dose tier and constrained
    # to the same day-2500 asymptote target.  This enforces the chemical expectation
    # that similar catalyst additions produce similar final recoveries (e.g. the five
    # ~37–40 mg/L columns in the pq/elephant_ii sample).
    # 0 = disabled (original exact-match behaviour).
    "prefit_catalyst_dose_bin_pct_tolerance": 12.0,
    # Dose-saturation parameters for the catalyzed recovery cap.
    # cat_cap(dose) = ctrl_cap + dose_frac * (cat_cap_full - ctrl_cap)
    # dose_frac     = dose_mg_l / (dose_mg_l + half_sat_dose_mg_l)
    # Encodes diminishing returns: big impact between 50–100 mg/L, little between 150–200 mg/L.
    "prefit_cat_cap_dose_saturation_enabled": True,
    "prefit_cat_cap_dose_half_sat_mg_l": 80.0,
    # Mineralogy-based sigmoid gate onset prior.
    # The sigmoid gate models the induction period before leaching wakes up.
    # Onset is long when fast-leaching Cu (oxides + secondary sulfides) is
    # scarce relative to chalcopyrite, when the column is tall, or when
    # acid-consuming gangue must be neutralised before effective leaching starts.
    #
    # fast_cu_frac  = (0.7 * cu_oxides + 0.3 * cu_secondary) / total_cu
    # gate_mid_base = base_max_onset * (1 - fast_cu_frac) ^ slow_cu_exponent
    # gate_mid_prior = gate_mid_base * height_factor * acid_factor
    "gate_onset_base_max_days": 200.0,     # onset (days) if 100% chalcopyrite — short startup delay only
    "prefit_sigmoid_gate_max_mid_days": 120.0,  # hard cap on prefit gate_mid search space
    "gate_onset_slow_cu_exponent": 1.5,   # steepness; >1 = gentle for mixed ores
    "gate_onset_height_ref_m": 3.0,       # reference column height
    "gate_onset_height_alpha": 0.40,      # height scaling exponent
    "gate_onset_acid_consumer_strength": 0.50,  # max fractional onset boost from acid consumers
    # Weights in the NN learnable mineralogy gate offset (additive to gate_mid_day).
    # Set to 0 to disable the mineralogy prior in the NN.
    "nn_gate_onset_fast_cu_enabled": True,
    "nn_gate_onset_height_enabled": True,
    "nn_gate_onset_acid_enabled": True,
    # When fast-leaching Cu fraction is high, attenuate gate_strength proportionally.
    # 0 = no attenuation; 1 = gate fully suppressed for pure oxide ore.
    "nn_gate_strength_fast_cu_attenuation": 0.80,
    # P80-based kinetic rate upper-bound scaling for the biexponential prefit.
    # Coarser material (higher P80) has slower leach kinetics because the
    # surface-area-to-volume ratio decreases with particle size.
    #
    # The rate upper bound now uses the same compute_material_size_p80_cap_penalty
    # (shifted-Hill) function as the metallurgical recovery cap, so the kinetic
    # rate constraint is consistent with the cap penalty.
    # Formula:  rate_upper = prefit_p80_rate_base * cap_p80_penalty(p80_in)
    #   - cap_p80_penalty is defined by cap_p80_penalty_d0_in / d50_in / p_inf / n
    #   - prefit_p80_rate_base is the maximum allowed b1/b2 for any particle size
    # NOTE: prefit_p80_rate_ref_in and prefit_p80_rate_alpha are retired;
    #       the cap_p80_penalty_* keys govern both the recovery cap and rate bound.
    "prefit_p80_rate_base": 0.10,         # max b1/b2 rate for fine material
    # True: constrain NN curve params to prefit-derived quantile bounds.
    # False: use broad physical bounds so the model is not tied to prefit ranges.
    "use_prefit_param_bounds": True,
    "param_bounds_disabled_amplitude_upper": 100.0,
    "param_bounds_disabled_rate_upper": 1e-1,
    "leach_pct_oxides": 0.95,
    "leach_pct_secondary_sulfides": 0.70,
    "leach_pct_primary_sulfides_control": 0.30,
    "leach_pct_primary_sulfides_catalyzed": 0.70,
    "primary_control_prior_min": 0.10,
    "primary_control_prior_max": 0.35,
    "primary_catalyzed_prior_min": 0.25,
    "primary_catalyzed_prior_max": 0.70,
    "primary_fraction_learned_delta_max": 0.05,
    "cap_prior_violation_tolerance_pct": 1.0,
    # Moderate residual-CPY control-cap suppression.
    # Residual chalcopyrite still lowers the control cap, but the effect starts
    # later and saturates much more gently than the first v12 implementation.
    "control_cpy_cap_reduction_strength": 0.30,
    "control_cpy_cap_reduction_center_pct": 4.0,
    "control_cpy_cap_reduction_width_pct": 1.0,
    "cap_p80_penalty_d0_in": 2.0,
    "cap_p80_penalty_d50_in": 3.0,
    "cap_p80_penalty_p_inf": 0.4,
    "cap_p80_penalty_n": 2.0,
    "catalyst_extension_window_days": 50.0,
    "orp_aux_recent_window_days": 200.0,
    "orp_aux_window_start_day": 150.0,
    "orp_aux_window_end_day": 400.0,
    "orp_aux_trim_quantile": 0.10,
    "orp_aux_summary_step_days": 1.0,
    "orp_aux_min_recent_points": 3,
    "orp_aux_min_target_points": 5,
    "orp_aux_project_min_pairs": 2,
    "orp_aux_norm_floor_mv": 10.0,
    "orp_aux_source": "control",   # "control", "catalyzed", or "average"
    "ensemble_interval_smoothing_days": 140.0,
    "ensemble_interval_cover_true_curve": True,
    "ensemble_interval_cover_margin_pct": 0.0,
    "member_prediction_gap_band": True,
    "member_prediction_gap_margin_pct": 0.0,
    "min_residual_primary_uplift_factor": 0.20,
    "residual_primary_uplift_softness_power": 0.50,
    # terminal_slope_rate is useful, but because it is derived from the observed
    # curve tail it should act as a soft anchor rather than dominate the encoder.
    "terminal_slope_feature_encoder_scale": 0.40,
    "min_catalyst_use_frac": 0.02,
    "max_catalyst_use_frac": 0.35,
    "min_active_catalyst_inventory_frac": 0.03,
    "catalyst_conc_tau_fixed_scale": -0.10,
    "catalyst_conc_kappa_fixed_scale": 0.18,
    "catalyst_conc_temp_fixed_scale": -0.05,
    "active_catalyst_tau_fixed_scale": -0.05,
    "active_catalyst_kappa_fixed_scale": 0.14,
    "active_catalyst_temp_fixed_scale": -0.03,
    "catalyzed_gap_weight_boost": 2.0,
    "catalyzed_dose_weight_boost": 1.0,
    "final_recovery_weight_tail_boost": 1.5,
    "catalyst_use_prior_min": 0.03,
    "catalyst_use_prior_max": 0.12,
}

MODEL_LOGIC_VERSION = "v12_explicit_catalyst_dose_priority_20260428"

PREFIT_OUTPUT_COLUMNS = [
    "row_index",
    "col_id",      # v11: project_col_id for per-column traceability
    "sample_id",   # v11: project_sample_id for ore-sample traceability
    "status_norm",
    "prefit_logic_version",
    "fit_a1",
    "fit_b1",
    "fit_a2",
    "fit_b2",
    "fit_curve_mode",
    "fit_gate_active",
    "fit_gate_mid_day",
    "fit_gate_width_day",
    "fit_asymptote",
    "fit_p80_cap_factor",
    "fit_ctrl_cap_raw",
    "fit_cat_cap_raw",
    "fit_ctrl_cap",
    "fit_cat_cap",
    "fit_sample_cap",
    "fit_raw_sample_cap",
    "fit_metallurgy_sample_cap",
    "fit_data_informed_cap",
    "fit_data_informed_cap_active",
    "fit_final_secondary_fraction",
    "fit_final_primary_fraction",
    "fit_prefit_rebase_active",
    "fit_prefit_rebase_time_offset_day",
    "fit_prefit_rebase_recovery_offset_pct",
    "fit_prefit_ignored_initial_point_count",
    "fit_prefit_fit_point_count",
    "fit_prefit_fit_start_day",
    "fit_prefit_fit_start_day_source",
    "fit_target_day",
    "fit_target_day_asymptote_frac",
    "fit_target_day_slope_pct_per_day",
    "fit_y2500_asymptote_frac",
    "fit_slope2500_pct_per_day",
    "fit_duplicate_target_active",
    "fit_duplicate_target_day",
    "fit_duplicate_target_recovery",
    "fit_duplicate_target_mode",
    "fit_final_catalyst_cum_kg_t",
    "fit_rmse",
]
# Subset of PREFIT_OUTPUT_COLUMNS that contains actual fit results rather than
# metadata that already exists in the main dataframe (col_id, sample_id,
# status_norm).  The cache validation and merge use ONLY these columns so that
# a prefit CSV produced by an older version (which lacks the v11 traceability
# columns) can still be reused without triggering a spurious recompute.
PREFIT_FIT_COLUMNS = [
    "row_index",
    "prefit_logic_version",
    "fit_a1",
    "fit_b1",
    "fit_a2",
    "fit_b2",
    "fit_curve_mode",
    "fit_gate_active",
    "fit_gate_mid_day",
    "fit_gate_width_day",
    "fit_asymptote",
    "fit_p80_cap_factor",
    "fit_ctrl_cap_raw",
    "fit_cat_cap_raw",
    "fit_ctrl_cap",
    "fit_cat_cap",
    "fit_sample_cap",
    "fit_raw_sample_cap",
    "fit_metallurgy_sample_cap",
    "fit_data_informed_cap",
    "fit_data_informed_cap_active",
    "fit_final_secondary_fraction",
    "fit_final_primary_fraction",
    "fit_prefit_rebase_active",
    "fit_prefit_rebase_time_offset_day",
    "fit_prefit_rebase_recovery_offset_pct",
    "fit_prefit_ignored_initial_point_count",
    "fit_prefit_fit_point_count",
    "fit_prefit_fit_start_day",
    "fit_prefit_fit_start_day_source",
    "fit_target_day",
    "fit_target_day_asymptote_frac",
    "fit_target_day_slope_pct_per_day",
    "fit_y2500_asymptote_frac",
    "fit_slope2500_pct_per_day",
    "fit_duplicate_target_active",
    "fit_duplicate_target_day",
    "fit_duplicate_target_recovery",
    "fit_duplicate_target_mode",
    "fit_final_catalyst_cum_kg_t",
    "fit_rmse",
]
PREFIT_CACHE_COMPARE_COLUMNS = [
    TIME_COL_COLUMNS,
    TARGET_COLUMNS,
    "acid_soluble_%",
    "cyanide_soluble_%",
    "material_size_p80_in",
    "lixiviant_initial_fe_mg_l",
    "lixiviant_initial_ph",
    "lixiviant_initial_orp_mv",
    "grouped_acid_generating_sulfides",
    "grouped_carbonates",
    "column_height_m",
    "residual_cpy_%",
    "copper_oxides_equivalent",
    "copper_secondary_sulfides_equivalent",
    "copper_primary_sulfides_equivalent",
    CATALYST_ADDITION_COL,
    CATALYST_CUM_COL,
    LIXIVIANT_CUM_COL,
    FEED_MASS_COL,
    "column_inner_diameter_m",
]

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
    # ------------------------------------------------------------------
    # Amax-oriented latent: broad chemical recoverability / receptivity
    # Intended support: fit_a1 + fit_a2
    # ------------------------------------------------------------------
    "chem_raw": {
        "acid_soluble_%": 0.26,
        "cyanide_soluble_%": 0.18,
        "residual_cpy_%": -0.18,
        "grouped_copper_sulfides": 0.04,
        "grouped_secondary_copper": 0.12,
        "grouped_primary_copper_sulfides": 0.02,
        "grouped_secondary_copper_sulfides": 0.16,
        "grouped_copper_oxides": 0.20,
        "grouped_mixed_copper_ores": 0.04,
        "grouped_acid_generating_sulfides": 0.04,
        "grouped_gangue_sulfides": -0.08,        # non-Cu sulfides dilute overall Cu recoverability
        "grouped_gangue_silicates": -0.14,
        "grouped_fe_oxides": 0.04, # Mixed group — net slightly positive due to Cu-bearing members
        "grouped_carbonates": -0.24,
        "apparent_bulk_density_t_m3": -0.14,
        "material_size_to_column_diameter_ratio": -0.06,
        "bornite": 0.10,
        "fe:cu": -0.10,
        "cu:fe": 0.06,
        "copper_primary_sulfides_equivalent": 0.10,
        "copper_secondary_sulfides_equivalent": 0.18,
        "copper_sulfides_equivalent": 0.06,
        "copper_oxides_equivalent": 0.22,
    },

    # ------------------------------------------------------------------
    # Slow-tail burden / passivating primary sulfide character
    # Intended support: fit_a2 burden, fit_b2 burden
    # ------------------------------------------------------------------
    "primary_passivation_drive": {
        "residual_cpy_%": 0.42,
        "copper_primary_sulfides_equivalent": 0.70,
        "copper_secondary_sulfides_equivalent": -0.18,
        "copper_oxides_equivalent": -0.24,
        "acid_soluble_%": -0.16,
        "cyanide_soluble_%": -0.12,
        "grouped_copper_sulfides": 0.06,
        "grouped_primary_copper_sulfides": 0.42,
        "grouped_secondary_copper": -0.06,
        "grouped_secondary_copper_sulfides": -0.14,
        "grouped_copper_oxides": -0.16,
        "grouped_mixed_copper_ores": 0.06,
        "fe:cu": 0.16,
        "cu:fe": -0.10,
        "material_size_p80_in": 0.22,
        "grouped_acid_generating_sulfides": 0.00, # Pyrite regenerates Fe³⁺, doesn't passivate CuFeS₂
        "grouped_gangue_sulfides": 0.06,         # galena → PbSO₄ adds a physical passivation component
        "grouped_gangue_silicates": 0.14,
        "grouped_fe_oxides": 0.08,
        "grouped_carbonates": 0.12,
        "apparent_bulk_density_t_m3": 0.16,
        "material_size_to_column_diameter_ratio": 0.05,
        "bornite": -0.06,
        "copper_sulfides_equivalent": 0.12,
        "ferric_ready_primary_score": -0.30,
        "tall_column_oxidant_deficit_score": 0.34,
    },

    # ------------------------------------------------------------------
    # Redox / pyrite / ferric assistance
    # Intended support: catalyst responsiveness and some fit_b2 relief
    # ------------------------------------------------------------------
    "ferric_synergy": {
        "fe:cu": 0.30,
        "cu:fe": -0.12,
        "residual_cpy_%": 0.14,
        "copper_primary_sulfides_equivalent": 0.16,
        "copper_secondary_sulfides_equivalent": 0.08,
        "copper_oxides_equivalent": -0.04,
        "acid_soluble_%": -0.06,
        "cyanide_soluble_%": -0.04,
        "grouped_primary_copper_sulfides": 0.10,
        "grouped_secondary_copper_sulfides": 0.08,
        "grouped_acid_generating_sulfides": 0.34,
        "grouped_gangue_sulfides": -0.24,        # STRONGEST effect: ZnS+PbS both oxidised by Fe³⁺,
                                                 # directly depleting the ferric pool for CuFeS₂ attack
        "grouped_gangue_silicates": -0.06,
        "grouped_fe_oxides": 0.08,
        "grouped_carbonates": -0.18, # Siderite releases Fe²⁺ that partially offsets the acid buffer penalty
        "grouped_phosphate_minerals": -0.04, # Apatite and similar tie up Fe²⁺ and release acidity, both bad for ferric synergy
        "material_size_p80_in": -0.04,
        "apparent_bulk_density_t_m3": -0.02,
        "material_size_to_column_diameter_ratio": -0.06,
        "bornite": 0.10,
        "primary_passivation_drive": 0.16,
        "ferric_ready_primary_score": 0.24,
        "tall_column_oxidant_deficit_score": -0.22,
    },

    # ------------------------------------------------------------------
    # Oxidizing lixiviant bootstrap
    # Intended support: early control/catalyzed takeoff in columns that start
    # with already-ferric, low-pH, oxidizing solution chemistry
    # ------------------------------------------------------------------
    "ferric_bootstrap": {
        "lixiviant_initial_fe_mg_l": 0.42,
        "lixiviant_initial_orp_mv": 0.18,
        "lixiviant_initial_ph": -0.24,
        "acid_soluble_%": 0.08,
        "cyanide_soluble_%": 0.04,
        "grouped_acid_generating_sulfides": 0.10,
        "grouped_carbonates": -0.08,
        "grouped_fe_oxides": 0.04,
        "fe:cu": 0.06,
        "copper_primary_sulfides_equivalent": 0.04,
        "copper_secondary_sulfides_equivalent": 0.08,
        "copper_oxides_equivalent": 0.06,
    },

    # ------------------------------------------------------------------
    # Blended solution responsiveness
    # Intended support: fit_b1 and some Amax expression
    # ------------------------------------------------------------------
    "chem_interaction": {
        "primary_passivation_drive": -0.04,
        "copper_primary_sulfides_equivalent": 0.02,
        "acid_soluble_%": 0.14,
        "copper_secondary_sulfides_equivalent": 0.12,
        "copper_oxides_equivalent": 0.12,
        "grouped_secondary_copper": 0.06,
        "grouped_secondary_copper_sulfides": 0.08,
        "grouped_copper_oxides": 0.08,
        "grouped_mixed_copper_ores": 0.04,
        "fe:cu": 0.04,
        "cu:fe": -0.02,
        "grouped_gangue_sulfides": -0.06,        # reduces clean solution chemistry signal
        "grouped_gangue_silicates": -0.08,
        "grouped_carbonates": -0.16,
        "cyanide_soluble_%": 0.10,
        "grouped_acid_generating_sulfides": 0.06,
        "grouped_fe_oxides": -0.02,
        "material_size_p80_in": -0.08,
        "apparent_bulk_density_t_m3": -0.06,
        "material_size_to_column_diameter_ratio": -0.08,
        "bornite": 0.08,
        "ferric_synergy": 0.14,
        "ferric_bootstrap": 0.18,
        "ferric_ready_primary_score": 0.24,
        "tall_column_oxidant_deficit_score": -0.18,
    },

    # ------------------------------------------------------------------
    # Catalyst benefit for passivated primary systems
    # Intended support: catalyst uplift on fit_a2 and fit_b2
    # ------------------------------------------------------------------
    "primary_catalyst_synergy": {
        "residual_cpy_%": 0.28,
        "copper_primary_sulfides_equivalent": 0.46,
        "copper_secondary_sulfides_equivalent": 0.02,
        "copper_oxides_equivalent": -0.20,
        "acid_soluble_%": -0.12,
        "cyanide_soluble_%": -0.04,
        "grouped_primary_copper_sulfides": 0.34,
        "grouped_secondary_copper": 0.00,
        "grouped_secondary_copper_sulfides": 0.04,
        "grouped_copper_oxides": -0.16,
        "fe:cu": 0.10,
        "cu:fe": -0.06,
        "material_size_p80_in": -0.08,
        "grouped_acid_generating_sulfides": 0.18,
        "grouped_gangue_sulfides": -0.16,        # Fe³⁺ competition reduces how much catalyst benefit
                                                 # reaches the chalcopyrite target
        "grouped_gangue_silicates": -0.10,
        "grouped_fe_oxides": 0.02,
        "grouped_carbonates": -0.16,
        "grouped_phosphate_minerals": -0.06, # Phosphates tie up Fe²⁺ and release acidity, both bad for catalyst synergy
        "bornite": 0.04,
        "copper_sulfides_equivalent": 0.12,
        "chem_interaction": 0.08,
        "primary_passivation_drive": 0.32,
        "ferric_synergy": 0.22,
        "ferric_ready_primary_score": 0.28,
        "tall_column_oxidant_deficit_score": -0.24,
    },

    # ------------------------------------------------------------------
    # Fast accessible inventory
    # Intended support: fit_a1 and fit_b1
    # ------------------------------------------------------------------
    "fast_leach_inventory": {
        "acid_soluble_%": 0.40,
        "cyanide_soluble_%": 0.34,
        "residual_cpy_%": -0.18,
        "copper_primary_sulfides_equivalent": -0.28,
        "copper_secondary_sulfides_equivalent": 0.44,
        "copper_oxides_equivalent": 0.40,
        "grouped_secondary_copper": 0.24,
        "grouped_primary_copper_sulfides": -0.18,
        "grouped_secondary_copper_sulfides": 0.28,
        "grouped_copper_oxides": 0.28,
        "grouped_mixed_copper_ores": 0.12,
        "grouped_fe_oxides": 0.06, # Cu-bearing iron oxides are fast-leaching
        "grouped_gangue_sulfides": -0.06,        # small: competes for acid/oxidant in early stage
        "grouped_gangue_silicates": -0.18,
        "bornite": 0.22,
        "copper_sulfides_equivalent": -0.02,
        "grouped_acid_generating_sulfides": 0.00,
        "primary_passivation_drive": -0.18,
        "ferric_bootstrap": 0.18,
        "diffusion_drag_strength": -0.18,
        "ferric_ready_primary_score": 0.32,
        "tall_column_oxidant_deficit_score": -0.34,
    },

    # ------------------------------------------------------------------
    # Oxide-dominant early inventory
    # Intended support: fit_a1 and early Amax
    # ------------------------------------------------------------------
    "oxide_inventory": {
        "copper_oxides_equivalent": 0.76,
        "acid_soluble_%": 0.26,
        "cyanide_soluble_%": 0.04,
        "residual_cpy_%": -0.10,
        "copper_primary_sulfides_equivalent": -0.12,
        "copper_secondary_sulfides_equivalent": 0.06,
        "grouped_copper_oxides": 0.42,
        "grouped_mixed_copper_ores": 0.12,
        "grouped_secondary_copper_sulfides": 0.04,
        "grouped_primary_copper_sulfides": -0.08,
        # grouped_gangue_sulfides: omitted — no mechanistic link to oxide inventory
        "grouped_fe_oxides": 0.10, # fe_oxides_cu and limonite-cu are Cu oxide sources
    },

    # ------------------------------------------------------------------
    # Acid-consuming / tail-suppressing burden
    # Intended support: fit_b2 burden and Amax suppression
    # ------------------------------------------------------------------
    "acid_buffer_strength": {
        "grouped_carbonates": 0.86,
        "grouped_gangue_silicates": 0.28,
        "grouped_fe_oxides": 0.14, # Goethite/limonite acid consumption underweighted
        "grouped_gangue_sulfides": 0.04,         # weak: ZnS + H₂SO₄ consumes some acid,
                                                 # much weaker than carbonates/silicates
        "acid_soluble_%": -0.16,
        "copper_oxides_equivalent": -0.14,
        "grouped_copper_oxides": -0.10,
        "grouped_acid_generating_sulfides": -0.08,
        "fe:cu": 0.00,
        "cu:fe": 0.00,
        "ferric_ready_primary_score": -0.16,
    },

    # ------------------------------------------------------------------
    # Decay of early acid burden
    # Intended support: secondary modifier on tail kinetics
    # ------------------------------------------------------------------
    "acid_buffer_decay": {
        "grouped_carbonates": 0.38,
        "acid_soluble_%": 0.12,
        "copper_oxides_equivalent": 0.06,
        "grouped_copper_oxides": 0.04,
        "grouped_acid_generating_sulfides": -0.08,
        "fe:cu": -0.04,
        "grouped_gangue_silicates": -0.16,
        "grouped_fe_oxides": -0.04,
        "material_size_p80_in": -0.08,
        "apparent_bulk_density_t_m3": 0.06,
        "material_size_to_column_diameter_ratio": 0.04,
        # grouped_gangue_sulfides: omitted — gangue sulfides don't meaningfully decay acid burden
    },

    # ------------------------------------------------------------------
    # Geometry / transport burden
    # Intended support: fit_b1 burden and fit_b2 burden
    # ------------------------------------------------------------------
    "diffusion_drag_strength": {
        "grouped_gangue_silicates": 0.34,
        "material_size_p80_in": 0.62,
        "apparent_bulk_density_t_m3": 0.10,
        "material_size_to_column_diameter_ratio": 0.10,
        "column_height_m": 0.00,
        "residual_cpy_%": 0.10,
        "copper_primary_sulfides_equivalent": 0.12,
        "copper_secondary_sulfides_equivalent": -0.12,
        "copper_oxides_equivalent": -0.12,
        "acid_soluble_%": -0.10,
        "cyanide_soluble_%": -0.10,
        "grouped_primary_copper_sulfides": 0.08,
        "grouped_secondary_copper_sulfides": -0.08,
        "grouped_acid_generating_sulfides": 0.08,
        "grouped_gangue_sulfides": 0.18,         # galena → insoluble PbSO₄ precipitates in pores,
                                                 # physically blocking solution transport
        "grouped_fe_oxides": 0.08,
        "grouped_carbonates": 0.20, # Gypsum (CaSO₄) precipitation is a real pore-plugging mechanism
        "bornite": -0.02,
        "fast_leach_inventory": -0.14,
        "oxide_inventory": -0.08,
        "ferric_ready_primary_score": -0.18,
        "tall_column_oxidant_deficit_score": 0.34,
    },

    # ------------------------------------------------------------------
    # Exposed fresh reactive surface / accessibility
    # Intended support: fit_b1
    # ------------------------------------------------------------------
    "surface_refresh": {
        "material_size_p80_in": -0.62,
        "acid_soluble_%": 0.08,
        "cyanide_soluble_%": 0.08,
        "copper_oxides_equivalent": 0.24,
        "copper_secondary_sulfides_equivalent": 0.20,
        "residual_cpy_%": -0.08,
        "copper_primary_sulfides_equivalent": -0.06,
        "grouped_secondary_copper": 0.08,
        "grouped_secondary_copper_sulfides": 0.14,
        "grouped_copper_oxides": 0.12,
        "grouped_acid_generating_sulfides": 0.08,
        "grouped_gangue_sulfides": -0.12,        # PbSO₄ coatings and S⁰ deposition reduce
                                                 # accessible surface; pore plugging limits refresh
        "grouped_gangue_silicates": -0.16,
        "grouped_carbonates": -0.14, # Gypsum surface coatings more impactful than currently weighted
        "grouped_fe_oxides": -0.08,
        "apparent_bulk_density_t_m3": -0.18,
        "material_size_to_column_diameter_ratio": -0.10,
        "bornite": 0.18,
        "chem_interaction": 0.08,
        "ferric_synergy": 0.10,
        "ferric_bootstrap": 0.12,
        "primary_passivation_drive": -0.16,
        "diffusion_drag_strength": -0.30,
        "ferric_ready_primary_score": 0.26,
        "tall_column_oxidant_deficit_score": -0.28,
    },

    # ------------------------------------------------------------------
    # Late-time decay / difficult tail inventory
    # Intended support: fit_a2 and fit_b2
    # ------------------------------------------------------------------
    "ore_decay_strength": {
        "residual_cpy_%": 0.24,
        "copper_primary_sulfides_equivalent": 0.24,
        "grouped_gangue_sulfides": 0.14,         # sustained Fe³⁺ competition throughout the leach
                                                 # worsens late recovery — the tail gets harder
        "grouped_gangue_silicates": 0.10,
        "grouped_carbonates": 0.10,
        "grouped_acid_generating_sulfides": 0.00,
        "apparent_bulk_density_t_m3": 0.16,
        "material_size_to_column_diameter_ratio": 0.10,
        "bornite": -0.06,
        "primary_passivation_drive": 0.30,
        "fast_leach_inventory": -0.24,
        "oxide_inventory": -0.14,
        "acid_buffer_strength": 0.16,
        "diffusion_drag_strength": 0.30,
        "surface_refresh": -0.10,
        "depassivation_strength": -0.08,
        "ferric_ready_primary_score": -0.14,
        "tall_column_oxidant_deficit_score": 0.28,
    },

    # ------------------------------------------------------------------
    # Slow-tail passivation severity
    # Intended support: fit_b2
    # ------------------------------------------------------------------
    "passivation_strength": {
        "residual_cpy_%": 0.24,
        "copper_primary_sulfides_equivalent": 0.28,
        "copper_secondary_sulfides_equivalent": -0.08,
        "copper_oxides_equivalent": -0.12,
        "cyanide_soluble_%": -0.08,
        "grouped_gangue_sulfides": 0.10,         # PbSO₄ + S⁰ deposition from gangue sulfide
                                                 # oxidation adds a physical passivation burden;
                                                 # distinct from but additive to CuFeS₂ passivation
        "grouped_gangue_silicates": 0.10,
        "grouped_fe_oxides": 0.10,
        "grouped_carbonates": 0.18,
        "grouped_acid_generating_sulfides": -0.06,
        "grouped_phosphate_minerals": -0.06, # phosphate coatings on mineral surfaces act similarly to sulfur-rich layers
        "fe:cu": 0.10,
        "bornite": -0.08,
        "primary_passivation_drive": 0.58,
        "diffusion_drag_strength": 0.18,
        "ore_decay_strength": 0.14,
        "surface_refresh": -0.12,
        "ferric_ready_primary_score": -0.16,
        "tall_column_oxidant_deficit_score": 0.24,
    },

    # ------------------------------------------------------------------
    # Catalyst-enabled relief of tail burden
    # Intended support: catalyst uplift on fit_a2 and fit_b2
    # ------------------------------------------------------------------
    "depassivation_strength": {
        "residual_cpy_%": 0.20,
        "copper_primary_sulfides_equivalent": 0.22,
        "copper_secondary_sulfides_equivalent": 0.04,
        "copper_oxides_equivalent": -0.08,
        "acid_soluble_%": -0.06,
        "cyanide_soluble_%": -0.04,
        "material_size_p80_in": -0.18,
        "grouped_acid_generating_sulfides": 0.18,
        "grouped_gangue_sulfides": -0.16,        # catalyst cannot address PbSO₄ or ZnS Fe³⁺
                                                 # consumption; gangue sulfides reduce the
                                                 # "headroom" available to catalyst depassivation
        "grouped_gangue_silicates": -0.12,
        "grouped_carbonates": -0.18,
        "grouped_phosphate_minerals": -0.10, # Apatite and similar tie up Fe²⁺ and release acidity, both bad for depassivation
        "bornite": 0.06,
        "fe:cu": 0.10,
        "primary_passivation_drive": 0.18,
        "ferric_synergy": 0.24,
        "chem_interaction": 0.10,
        "primary_catalyst_synergy": 0.34,
        "acid_buffer_strength": -0.12,
        "diffusion_drag_strength": -0.18,
        "surface_refresh": 0.20,
        "ore_decay_strength": -0.08,
    },

    # ------------------------------------------------------------------
    # Transitional / mixed-ore conversion latent
    # Intended support: mild fit_a2 / fit_b2 assistance
    # ------------------------------------------------------------------
    "transform_strength": {
        "residual_cpy_%": 0.08,
        "copper_primary_sulfides_equivalent": 0.10,
        "copper_secondary_sulfides_equivalent": 0.10,
        "copper_oxides_equivalent": -0.08,
        "acid_soluble_%": -0.02,
        "cyanide_soluble_%": 0.04,
        "material_size_p80_in": -0.14,
        "grouped_mixed_copper_ores": 0.18,
        "grouped_acid_generating_sulfides": 0.10,
        # grouped_gangue_sulfides: omitted — no clear transformation pathway,
        # gangue sulfides don't convert to accessible copper
        "grouped_gangue_silicates": -0.08,
        "grouped_carbonates": -0.12,
        "bornite": 0.12,
        "fe:cu": 0.04,
        "primary_passivation_drive": 0.10,
        "ferric_synergy": 0.18,
        "chem_interaction": 0.08,
        "primary_catalyst_synergy": 0.16,
        "acid_buffer_strength": -0.08,
        "diffusion_drag_strength": -0.10,
        "surface_refresh": 0.10,
        "depassivation_strength": 0.14,
    },

    # ------------------------------------------------------------------
    # Characteristic delay / accessibility timescale
    # Intended support: mostly fit_b1 and fit_b2 timing
    # ------------------------------------------------------------------
    "tau_days": {
        "column_height_m": 0.56,
        "material_size_p80_in": 0.74,
        "apparent_bulk_density_t_m3": 0.24,
        "material_size_to_column_diameter_ratio": 0.10,
        "fe:cu": 0.10,
        "cu:fe": -0.08,
        "grouped_acid_generating_sulfides": -0.06,
        "grouped_gangue_sulfides": 0.08,         # PbSO₄ buildup gradually reduces permeability,
                                                 # increasing effective residence time
        "grouped_fe_oxides": 0.06,
        "grouped_carbonates": 0.16,
        "copper_primary_sulfides_equivalent": 0.18,
        "copper_secondary_sulfides_equivalent": -0.10,
        "copper_oxides_equivalent": -0.16,
        "chem_interaction": -0.10,
        "ferric_synergy": -0.12,
        "ferric_bootstrap": -0.14,
        "primary_catalyst_synergy": -0.08,
        "acid_buffer_strength": 0.14,
        "diffusion_drag_strength": 0.24,
        "surface_refresh": -0.14,
        "ore_decay_strength": 0.12,
        "passivation_strength": 0.16,
        "depassivation_strength": -0.10,
        "ferric_ready_primary_score": -0.18,
        "tall_column_oxidant_deficit_score": 0.28,
    },

    # ------------------------------------------------------------------
    # Secondary transition timing
    # Intended support: mostly fit_b2 timing
    # ------------------------------------------------------------------
    "temp_days": {
        "column_height_m": 0.50,
        "material_size_p80_in": 0.66,
        "apparent_bulk_density_t_m3": 0.20,
        "material_size_to_column_diameter_ratio": 0.10,
        "fe:cu": 0.10,
        "cu:fe": -0.06,
        "grouped_acid_generating_sulfides": -0.04,
        "grouped_gangue_sulfides": 0.06,         # same PbSO₄ transport delay, slightly weaker
                                                 # because temp_days is a secondary timing term
        "grouped_fe_oxides": 0.06,
        "grouped_carbonates": 0.14,
        "copper_primary_sulfides_equivalent": 0.16,
        "copper_secondary_sulfides_equivalent": -0.08,
        "copper_oxides_equivalent": -0.12,
        "chem_interaction": -0.08,
        "ferric_synergy": -0.10,
        "ferric_bootstrap": -0.10,
        "primary_catalyst_synergy": -0.06,
        "acid_buffer_strength": 0.12,
        "diffusion_drag_strength": 0.20,
        "surface_refresh": -0.12,
        "ore_decay_strength": 0.12,
        "passivation_strength": 0.14,
        "depassivation_strength": -0.10,
        "ferric_ready_primary_score": -0.12,
        "tall_column_oxidant_deficit_score": 0.24,
    },

    # ------------------------------------------------------------------
    # Tail-shape / curvature prior
    # Intended support: fit_a2 / fit_b2 relationship
    # ------------------------------------------------------------------
    "kappa": {
        "column_height_m": 0.20,
        "material_size_p80_in": -0.20,
        "apparent_bulk_density_t_m3": 0.14,
        "material_size_to_column_diameter_ratio": -0.10,
        "fe:cu": 0.10,
        "cu:fe": -0.06,
        "grouped_acid_generating_sulfides": -0.02,
        "grouped_gangue_sulfides": 0.06,         # progressive pore blocking shifts the tail shape
                                                 # toward slower, more diffusion-limited character
        "grouped_fe_oxides": 0.06,
        "grouped_carbonates": 0.12,
        "copper_primary_sulfides_equivalent": 0.16,
        "copper_secondary_sulfides_equivalent": -0.06,
        "copper_oxides_equivalent": -0.10,
        "chem_interaction": -0.06,
        "ferric_synergy": -0.06,
        "primary_catalyst_synergy": -0.04,
        "acid_buffer_strength": 0.12,
        "diffusion_drag_strength": 0.18,
        "surface_refresh": -0.10,
        "ore_decay_strength": 0.18,
        "passivation_strength": 0.18,
        "depassivation_strength": -0.10,
        "ferric_ready_primary_score": -0.08,
        "tall_column_oxidant_deficit_score": 0.16,
    },

    # ------------------------------------------------------------------
    # Catalyst aging / fade prior
    # Intended support: very late catalyst response only
    # ------------------------------------------------------------------
    "aging_strength": {
        "column_height_m": 0.01,
        "material_size_to_column_diameter_ratio": 0.10,
        "apparent_bulk_density_t_m3": 0.18,
        "material_size_p80_in": 0.08,
        "grouped_gangue_sulfides": 0.08,         # sustained Fe³⁺ consumption makes catalyst appear
                                                 # to "age out" faster — less free ferric means the
                                                 # catalyst effect weakens sooner in practical terms
        "ferric_synergy": 0.22,
        "chem_interaction": 0.12,
        "primary_passivation_drive": 0.20,
        "passivation_strength": 0.16,
        "ore_decay_strength": 0.16,
        "primary_catalyst_synergy": 0.16,
        "depassivation_strength": -0.06,
        "surface_refresh": -0.08,
        "diffusion_drag_strength": 0.22,
        "acid_buffer_strength": -0.10,
        "grouped_carbonates": -0.08,
        "grouped_gangue_silicates": -0.06,
        "fast_leach_inventory": -0.18,
        "oxide_inventory": -0.12,
        "tall_column_oxidant_deficit_score": 0.16,
    },
}

LATENT_TARGET_SUPPORT: Dict[str, Tuple[str, ...]] = {
    "chem_raw": ("fit_a1_plus_a2",),
    "primary_passivation_drive": ("fit_a2", "fit_b2"),
    "ferric_synergy": ("fit_b2", "catalyst_tail_unlock"),
    "ferric_bootstrap": ("fit_b1", "fit_b1_timing"),
    "chem_interaction": ("fit_b1", "fit_a1_plus_a2"),
    "primary_catalyst_synergy": ("catalyst_fit_a2", "catalyst_fit_b2"),
    "fast_leach_inventory": ("fit_a1", "fit_b1"),
    "oxide_inventory": ("fit_a1", "fit_a1_plus_a2"),
    "acid_buffer_strength": ("fit_b2", "fit_a1_plus_a2_negative"),
    "acid_buffer_decay": ("fit_b2_timing",),
    "diffusion_drag_strength": ("fit_b1_negative", "fit_b2_negative"),
    "surface_refresh": ("fit_b1",),
    "ore_decay_strength": ("fit_a2", "fit_b2"),
    "passivation_strength": ("fit_b2",),
    "depassivation_strength": ("catalyst_fit_a2", "catalyst_fit_b2"),
    "transform_strength": ("fit_a2_mild", "fit_b2_mild"),
    "tau_days": ("fit_b1_timing", "fit_b2_timing"),
    "temp_days": ("fit_b2_timing",),
    "kappa": ("fit_a2_fit_b2_shape",),
    "aging_strength": ("late_catalyst_fade",),
}


# ---------------------------
# Utilities
# ---------------------------
def compute_total_copper_equivalent(
    cu_oxides_equiv: float,
    cu_secondary_equiv: float,
    cu_primary_equiv: float,
) -> float:
    total = 0.0
    has_finite_value = False
    for value in (cu_oxides_equiv, cu_secondary_equiv, cu_primary_equiv):
        if np.isfinite(value):
            has_finite_value = True
            total += max(0.0, float(value))
    if not has_finite_value:
        return np.nan
    return float(total)


def enable_torch_determinism(deterministic: bool = True) -> None:
    torch.backends.cudnn.benchmark = bool(device.type == "cuda" and not deterministic)
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
    if DEVICE_IS_ACCELERATOR:
        # A single accelerator-backed process keeps training stable and avoids
        # fighting over one GPU/MPS device with many CV workers.
        workers = 1

    requested_threads = int(CONFIG.get("torch_threads_per_worker", 0))
    if requested_threads <= 0:
        torch_threads = max(1, total_cores // workers)
    else:
        torch_threads = requested_threads
    torch_threads = max(1, min(torch_threads, total_cores))
    if device.type == "mps":
        if MPS_TORCH_THREADS_CAP > 0:
            torch_threads = min(torch_threads, max(1, MPS_TORCH_THREADS_CAP))

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


def resolve_pair_batch_size() -> int:
    requested = int(os.environ.get("ROSETTA_PAIR_BATCH_SIZE", CONFIG.get("pair_batch_size", 0)))
    if requested > 0:
        return int(requested)
    if device.type == "mps":
        return 16
    if device.type == "cuda":
        return 32
    return 1


def resolve_mps_torch_threads_cap() -> int:
    requested = int(
        os.environ.get("ROSETTA_MPS_TORCH_THREADS_CAP", CONFIG.get("mps_torch_threads_cap", 0))
    )
    return max(0, requested)


def resolve_eval_every_n_epochs() -> int:
    requested = int(
        os.environ.get("ROSETTA_EVAL_EVERY_N_EPOCHS", CONFIG.get("eval_every_n_epochs", 0))
    )
    if requested > 0:
        return int(requested)
    if device.type == "mps":
        return 5
    return 1


PAIR_BATCH_SIZE = resolve_pair_batch_size()
MPS_TORCH_THREADS_CAP = resolve_mps_torch_threads_cap()
EVAL_EVERY_N_EPOCHS = resolve_eval_every_n_epochs()


def clear_torch_device_cache() -> None:
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


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
    """Map any status string to canonical 'Catalyzed' or 'Control'.

    v11 primary source: catalyst_status column.
      "no_catalyst"   → "Control"
      "with_catalyst" → "Catalyzed"

    Legacy project_col_id heuristics (fallback):
      Contains 'cat' without 'no' prefix → "Catalyzed"
      Contains 'with' without 'no' prefix → "Catalyzed"
      Anything else → "Control"
    """
    s = str(value).strip().lower()
    # v11 explicit catalyst_status values
    if s == "no_catalyst":
        return "Control"
    if s == "with_catalyst":
        return "Catalyzed"
    # legacy heuristic (project_col_id suffix patterns)
    if ("cat" in s and "no_cat" not in s and "no cat" not in s) or (
        "with" in s and "no" not in s
    ):
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


def infer_last_actual_data_day(time_days: np.ndarray, values: np.ndarray) -> float:
    t = np.asarray(time_days, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.size == 0 or y.size == 0:
        return np.nan
    n = min(t.size, y.size)
    mask = np.isfinite(t[:n]) & np.isfinite(y[:n])
    if not np.any(mask):
        return np.nan
    return float(np.max(t[:n][mask]))


def resolve_curve_catalyst_start_day(
    row: pd.Series,
    status: str,
    time_days: np.ndarray,
    recovery: np.ndarray,
) -> Tuple[float, str]:
    explicit_start = scalar_from_maybe_array(row.get(CATALYST_START_DAY_COL, np.nan))
    if np.isfinite(explicit_start):
        return float(explicit_start), CATALYST_START_DAY_COL

    if str(status) == "Control":
        last_actual_day = infer_last_actual_data_day(time_days, recovery)
        if np.isfinite(last_actual_day):
            return float(last_actual_day), "last_actual_data_day"
        return np.nan, "missing"

    transition_time = scalar_from_maybe_array(row.get(TRANSITION_TIME_COL, np.nan))
    if np.isfinite(transition_time):
        return float(transition_time), TRANSITION_TIME_COL

    return np.nan, "missing"


def resolve_curve_internal_catalyst_effect_start_day(
    row: pd.Series,
    status: str,
    time_days: np.ndarray,
    recovery: np.ndarray,
    catalyst_cum_kg_t: np.ndarray,
    dosage_mg_l: np.ndarray,
    dosage_source: str = CATALYST_ADDITION_RECON_COL,
) -> Tuple[float, str]:
    """Resolve the INTERNAL start day for catalyst physics.

    This does not replace the exported/displayed catalyst_start_day used in the
    plots. It only determines when the model is allowed to treat a labeled
    catalyzed column as truly catalyzed.

    Priority order:
      1. first positive catalyst_addition_mg_l dosage, when that explicit vector exists
      2. first real rise in cumulative catalyst addition when explicit dosage is unavailable
      3. first positive reconstructed dosage when explicit dosage is unavailable
      4. explicit catalyst_start_days_of_leaching
      5. transition_time
      6. missing
    """
    if str(status) == "Control":
        return resolve_curve_catalyst_start_day(
            row=row,
            status=status,
            time_days=time_days,
            recovery=recovery,
        )

    dosage_start = infer_catalyst_start_day_from_dosage_array(time_days, dosage_mg_l)
    if str(dosage_source) == CATALYST_ADDITION_COL:
        if np.isfinite(dosage_start):
            return float(dosage_start), str(dosage_source)
    else:
        cumulative_start = infer_catalyst_addition_start_day(time_days, catalyst_cum_kg_t)
        if np.isfinite(cumulative_start):
            return float(cumulative_start), f"{CATALYST_CUM_COL}_rise"

    if str(dosage_source) != CATALYST_ADDITION_COL and np.isfinite(dosage_start):
        return float(dosage_start), str(dosage_source)

    explicit_start = scalar_from_maybe_array(row.get(CATALYST_START_DAY_COL, np.nan))
    if np.isfinite(explicit_start):
        return float(explicit_start), CATALYST_START_DAY_COL

    transition_time = scalar_from_maybe_array(row.get(TRANSITION_TIME_COL, np.nan))
    if np.isfinite(transition_time):
        return float(transition_time), TRANSITION_TIME_COL

    return np.nan, "missing"


def compute_average_catalyst_dose_mg_l(
    status: str,
    time_days: Optional[np.ndarray] = None,
    catalyst_start_day: float = float("nan"),
    catalyst_cum_kg_t: Optional[np.ndarray] = None,
    lixiviant_cum_m3_t: Optional[np.ndarray] = None,
    catalyst_feed_conc_mg_l: Optional[np.ndarray] = None,
) -> float:
    """Return the average catalyst feed concentration (mg/L) during active dosing intervals.

    The explicit ``catalyst_addition_mg_l`` vector is used when available.
    If it is unavailable for a catalyzed row, the average is back-calculated
    from the per-tonne cumulative profiles:

        C_feed[i] = Δ(cum_catalyst_kg_t) × 1000 / Δ(cum_lixiviant_m3_t)

    feed_mass_kg cancels from numerator and denominator because both profiles
    are already expressed per tonne of ore.
    """
    if str(status) == "Control":
        return 0.0

    active_mask = interval_active_mask_from_start_day(
        np.asarray(time_days, dtype=float) if time_days is not None else np.asarray([], dtype=float),
        catalyst_start_day,
    )

    # PRIMARY: explicit feed concentration profile from catalyst_addition_mg_l.
    if catalyst_feed_conc_mg_l is not None:
        d = np.asarray(catalyst_feed_conc_mg_l, dtype=float)
        n = d.size
        if time_days is not None:
            n = min(n, np.asarray(time_days, dtype=float).size)
        if n > 0 and np.any(np.isfinite(d[:n])):
            d = np.clip(np.where(np.isfinite(d[:n]), d[:n], 0.0), 0.0, None)
            mask = active_mask[:n] if active_mask.size >= n else np.ones(n, dtype=bool)
            eps = 1e-9
            active = d[(d > eps) & mask]
            return float(np.mean(active)) if active.size > 0 else 0.0

    # FALLBACK: derive from cumulative profiles.
    # Both arrays must be on the same aligned time grid (same length).
    if catalyst_cum_kg_t is not None and lixiviant_cum_m3_t is not None:
        cc = np.asarray(catalyst_cum_kg_t, dtype=float)
        cl = np.asarray(lixiviant_cum_m3_t, dtype=float)
        n = min(cc.size, cl.size)
        if n >= 2:
            cc = cc[:n]
            cl = cl[:n]
            # Δcum_catalyst [kg/t] × 1000  →  mg-numerator factor (feed_mass cancels)
            d_cat = cumulative_interval_deltas(cc)
            # Δcum_lixiviant [m³/t]         →  L-denominator factor (feed_mass cancels)
            d_lix = cumulative_interval_deltas(cl)
            eps = 1e-9
            c_feed = np.where(d_lix > eps, d_cat * 1000.0 / np.maximum(d_lix, eps), 0.0)
            mask = active_mask[:n] if active_mask.size >= n else np.ones(n, dtype=bool)
            active = c_feed[(c_feed > eps) & mask]
            if active.size > 0:
                return float(np.mean(active))
    return np.nan


def _format_invalid_row_examples(df: pd.DataFrame, invalid_index: pd.Index, max_examples: int = 5) -> str:
    example_parts: List[str] = []
    for idx in invalid_index[:max_examples]:
        if PAIR_ID_COL in df.columns:
            pair_id = df.at[idx, PAIR_ID_COL]
            example_parts.append(f"row={idx}, pair_id={pair_id}")
        else:
            example_parts.append(f"row={idx}")
    return "; ".join(example_parts)


def validate_required_scalar_columns(
    df: pd.DataFrame,
    columns: List[str],
    context: str,
) -> None:
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{context} is missing required columns: {', '.join(missing_columns)}"
        )

    failures: List[str] = []
    for col in columns:
        parsed = df[col].map(scalar_from_maybe_array)
        invalid_mask = ~parsed.map(np.isfinite)
        if invalid_mask.any():
            invalid_index = df.index[invalid_mask]
            failures.append(
                f"{col}: {int(invalid_mask.sum())} invalid rows "
                f"({_format_invalid_row_examples(df, invalid_index)})"
            )
    if failures:
        raise ValueError(
            f"{context} has missing or invalid required scalar values:\n- "
            + "\n- ".join(failures)
        )


def validate_required_feature_vector(
    values: np.ndarray,
    columns: List[str],
    context: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != len(columns):
        raise ValueError(
            f"{context} shape mismatch: expected ({len(columns)},), got {arr.shape}"
        )
    bad_idx = np.flatnonzero(~np.isfinite(arr))
    if bad_idx.size > 0:
        bad_cols = ", ".join(columns[int(i)] for i in bad_idx[:10])
        raise ValueError(f"{context} has missing or invalid values for: {bad_cols}")
    return arr


def compute_optional_static_predictor_defaults(df: pd.DataFrame) -> Dict[str, float]:
    defaults: Dict[str, float] = {}
    for col in sorted(OPTIONAL_STATIC_PREDICTOR_COLUMNS):
        if col not in df.columns:
            continue
        parsed = df[col].map(scalar_from_maybe_array).to_numpy(dtype=float)
        finite = parsed[np.isfinite(parsed)]
        if finite.size > 0:
            defaults[col] = float(np.nanmedian(finite))
    for col in OPTIONAL_STATIC_PREDICTOR_COLUMNS:
        defaults.setdefault(col, 0.0)
    return defaults


def resolve_static_predictor_value(
    row: pd.Series,
    column_name: str,
    optional_defaults: Optional[Dict[str, float]] = None,
) -> float:
    value = scalar_from_maybe_array(row.get(column_name, np.nan))
    if np.isfinite(value):
        return float(value)
    if column_name in OPTIONAL_STATIC_PREDICTOR_COLUMNS:
        defaults = optional_defaults or {}
        return float(defaults.get(column_name, 0.0))
    return np.nan


def build_missing_model_input_report(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Report model-input values that are missing before pair construction.

    The NN does not silently impute required scalar predictors: those missing
    values fail validation. The optional lixiviant initial-condition predictors
    are the exception; they are filled with dataset medians, or 0.0 if an
    optional column is absent/all-missing. This report is written before that
    validation so it is still available when a run stops on missing required
    inputs.
    """
    optional_defaults = compute_optional_static_predictor_defaults(df)
    rows: List[Dict[str, Any]] = []

    def _metadata(row: pd.Series, row_index: Any) -> Dict[str, Any]:
        return {
            "source_row_index": row_index,
            "project_sample_id": str(row.get(PAIR_ID_COL, "")).strip(),
            "project_col_id": str(row.get(COL_ID_COL, "")).strip(),
            "project_name": str(row.get(PROJECT_NAME_COL, "")).strip() if PROJECT_NAME_COL in row.index else "",
            "catalyst_status": str(row.get(STATUS_COL_PRIMARY, row.get(STATUS_COL_FALLBACK, ""))).strip(),
        }

    def _raw_value_for_csv(value: Any) -> str:
        if isinstance(value, float) and np.isnan(value):
            return ""
        if value is None:
            return ""
        return str(value)

    def _append_missing(
        row: pd.Series,
        row_index: Any,
        header: str,
        input_group: str,
        action: str,
        imputed_value: float = np.nan,
        reason: str = "",
        raw_value: Any = np.nan,
    ) -> None:
        rows.append(
            {
                **_metadata(row, row_index),
                "missing_header": header,
                "input_group": input_group,
                "action": action,
                "imputed_value": float(imputed_value) if np.isfinite(imputed_value) else np.nan,
                "reason": reason,
                "raw_value": _raw_value_for_csv(raw_value),
            }
        )

    for row_index, row in df.iterrows():
        for col in CSV_STATIC_PREDICTOR_COLUMNS:
            raw = row.get(col, np.nan) if col in df.columns else np.nan
            value = scalar_from_maybe_array(raw)
            if np.isfinite(value):
                continue
            if col in OPTIONAL_STATIC_PREDICTOR_COLUMNS:
                default_value = float(optional_defaults.get(col, 0.0))
                if col not in df.columns:
                    reason = "optional column absent from loaded CSV"
                else:
                    reason = "optional scalar missing/non-finite in loaded CSV"
                _append_missing(
                    row=row,
                    row_index=row_index,
                    header=col,
                    input_group="optional_static_predictor",
                    action="imputed_with_training_dataset_default",
                    imputed_value=default_value,
                    reason=reason,
                    raw_value=raw,
                )
            else:
                _append_missing(
                    row=row,
                    row_index=row_index,
                    header=col,
                    input_group="required_static_predictor",
                    action="not_imputed_run_will_error",
                    reason="required scalar predictor missing/non-finite",
                    raw_value=raw,
                )

        for col in INPUT_ONLY_COLUMNS:
            raw = row.get(col, np.nan) if col in df.columns else np.nan
            value = scalar_from_maybe_array(raw)
            if np.isfinite(value):
                continue
            _append_missing(
                row=row,
                row_index=row_index,
                header=col,
                input_group="required_input_only_predictor",
                action="not_imputed_run_will_error",
                reason="required input-only scalar missing/non-finite",
                raw_value=raw,
            )

        t_raw = parse_listlike(row.get(TIME_COL_COLUMNS, np.nan))
        y_raw = parse_listlike(row.get(TARGET_COLUMNS, np.nan))
        if int(np.isfinite(t_raw).sum()) < 6:
            _append_missing(
                row=row,
                row_index=row_index,
                header=TIME_COL_COLUMNS,
                input_group="required_dynamic_profile",
                action="not_imputed_row_skipped_or_run_will_error",
                reason="fewer than 6 finite time points",
                raw_value=row.get(TIME_COL_COLUMNS, np.nan),
            )
        if int(np.isfinite(y_raw).sum()) < 6:
            _append_missing(
                row=row,
                row_index=row_index,
                header=TARGET_COLUMNS,
                input_group="required_dynamic_profile",
                action="not_imputed_row_skipped_or_run_will_error",
                reason="fewer than 6 finite recovery points",
                raw_value=row.get(TARGET_COLUMNS, np.nan),
            )

        lix_raw = parse_listlike(row.get(LIXIVIANT_CUM_COL, np.nan))
        if int(np.isfinite(lix_raw).sum()) < 2:
            _append_missing(
                row=row,
                row_index=row_index,
                header=LIXIVIANT_CUM_COL,
                input_group="required_dynamic_profile",
                action="not_imputed_row_skipped",
                reason="lixiviant cumulative profile has fewer than 2 finite points",
                raw_value=row.get(LIXIVIANT_CUM_COL, np.nan),
            )

        status = normalize_status(row.get(STATUS_COL_PRIMARY, row.get(STATUS_COL_FALLBACK, "")))
        if status != "Control":
            cat_raw = parse_listlike(row.get(CATALYST_CUM_COL, np.nan))
            has_catalyst_profile = cat_raw.size >= 2 and np.any(np.isfinite(cat_raw) & (cat_raw > 0.0))
            if not has_catalyst_profile:
                _append_missing(
                    row=row,
                    row_index=row_index,
                    header=CATALYST_CUM_COL,
                    input_group="required_catalyzed_dynamic_profile",
                    action="not_imputed_row_skipped",
                    reason="catalyzed row lacks a finite positive cumulative catalyst profile",
                    raw_value=row.get(CATALYST_CUM_COL, np.nan),
                )

    report_columns = [
        "source_row_index",
        "project_sample_id",
        "project_col_id",
        "project_name",
        "catalyst_status",
        "missing_header",
        "input_group",
        "action",
        "imputed_value",
        "reason",
        "raw_value",
    ]
    report_df = pd.DataFrame(rows, columns=report_columns)
    summary = {
        "n_rows_in_training_dataframe": int(len(df)),
        "n_missing_entries": int(len(report_df)),
        "n_rows_with_missing_entries": int(report_df[["project_sample_id", "project_col_id"]].drop_duplicates().shape[0])
        if len(report_df) > 0
        else 0,
        "optional_static_defaults": {k: float(v) for k, v in optional_defaults.items()},
        "missing_by_header": (
            report_df.groupby("missing_header").size().sort_values(ascending=False).astype(int).to_dict()
            if len(report_df) > 0
            else {}
        ),
        "missing_by_action": (
            report_df.groupby("action").size().sort_values(ascending=False).astype(int).to_dict()
            if len(report_df) > 0
            else {}
        ),
    }
    return report_df, summary


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


def cumulative_interval_deltas(profile: np.ndarray) -> np.ndarray:
    """Return non-negative interval deltas using the frozen first-point convention."""
    c = clean_cumulative_profile(np.asarray(profile, dtype=float), force_zero=False)
    if c.size == 0:
        return np.zeros(0, dtype=float)
    return np.clip(np.diff(np.concatenate([[0.0], c])), 0.0, None)


def interval_active_mask_from_start_day(
    time_days: np.ndarray,
    start_day: float,
) -> np.ndarray:
    t = np.asarray(time_days, dtype=float)
    if t.size == 0 or not np.isfinite(float(start_day)):
        return np.ones(t.shape, dtype=bool)
    return np.asarray(t >= float(start_day) - 1e-9, dtype=bool)


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


def prepare_observed_profile_with_time(
    time_days: np.ndarray,
    observed_profile: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_days, dtype=float)
    y = np.asarray(observed_profile, dtype=float)
    n = min(t.size, y.size)
    if n <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    t = t[:n]
    y = y[:n]
    valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    t = t[valid]
    y = y[valid]
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    t_unique, inv = np.unique(t, return_inverse=True)
    y_sum = np.zeros(t_unique.shape, dtype=float)
    y_count = np.zeros(t_unique.shape, dtype=float)
    for i, j in enumerate(inv):
        y_sum[j] += float(y[i])
        y_count[j] += 1.0
    y_unique = np.divide(y_sum, np.maximum(y_count, 1.0), dtype=float)
    return t_unique.astype(float), y_unique.astype(float)


def resolve_observed_profile_on_time_grid(
    source_time_days: np.ndarray,
    target_time_days: np.ndarray,
    observed_profile: Any,
    clip_min: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Align a non-cumulative vector profile to a requested time grid."""
    target_t = np.asarray(target_time_days, dtype=float)
    if target_t.size == 0:
        return np.asarray([], dtype=float)

    raw = parse_listlike(observed_profile)
    if raw.size == 0 or not np.any(np.isfinite(raw)):
        return None

    if raw.size == target_t.size and np.asarray(source_time_days, dtype=float).size != raw.size:
        out = np.where(np.isfinite(raw), raw, 0.0).astype(float)
    else:
        source_t = np.asarray(source_time_days, dtype=float)
        obs_t, obs_v = prepare_observed_profile_with_time(source_t, raw)
        if obs_t.size == 0 or obs_v.size == 0:
            return None
        if obs_t.size == 1:
            out = np.full(target_t.shape, float(obs_v[0]), dtype=float)
        else:
            out = np.interp(
                target_t,
                obs_t,
                obs_v,
                left=float(obs_v[0]),
                right=float(obs_v[-1]),
            )

    out = np.where(np.isfinite(out), out, 0.0)
    if clip_min is not None and np.isfinite(float(clip_min)):
        out = np.clip(out, float(clip_min), None)
    return out.astype(float)


def trimmed_mean(values: np.ndarray, trim_quantile: float = 0.10) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    if finite.size <= 2:
        return float(np.mean(finite))

    q = float(np.clip(trim_quantile, 0.0, 0.49))
    if q <= 0.0:
        return float(np.mean(finite))

    lo = float(np.nanquantile(finite, q))
    hi = float(np.nanquantile(finite, 1.0 - q))
    trimmed = finite[(finite >= lo) & (finite <= hi)]
    if trimmed.size == 0:
        trimmed = finite
    return float(np.mean(trimmed))


def compute_orp_aux_target_from_profile(
    time_days: np.ndarray,
    orp_profile: np.ndarray,
    window_start_day: float,
    window_end_day: float,
    history_window_days: float,
    trim_quantile: float = 0.10,
    step_days: float = 1.0,
    min_recent_points: int = 3,
    min_target_points: int = 5,
) -> float:
    t_obs, orp_obs = prepare_observed_profile_with_time(time_days, orp_profile)
    if t_obs.size == 0 or orp_obs.size == 0:
        return np.nan

    target_start = float(window_start_day)
    target_end = float(max(window_end_day, target_start))
    last_day = float(t_obs[-1])
    recent_start = max(float(t_obs[0]), last_day - float(history_window_days))
    recent_mask = t_obs >= recent_start - 1e-9
    recent_vals = orp_obs[recent_mask]
    if recent_vals.size < int(max(1, min_recent_points)):
        return np.nan

    plateau_level = trimmed_mean(recent_vals, trim_quantile=trim_quantile)
    if not np.isfinite(plateau_level):
        return np.nan

    ext_t = np.asarray(t_obs, dtype=float)
    ext_y = np.asarray(orp_obs, dtype=float)
    if last_day < target_end - 1e-9:
        future_time_days = build_plot_time_grid(
            observed_time_days=np.asarray([last_day, target_end], dtype=float),
            start_day=last_day,
            target_day=target_end,
            step_days=float(step_days),
        )
        future_time_days = future_time_days[future_time_days > last_day + 1e-9]
        if future_time_days.size > 0:
            ext_t = np.concatenate([ext_t, future_time_days])
            ext_y = np.concatenate([ext_y, np.full(future_time_days.shape, plateau_level, dtype=float)])

    summary_start = max(target_start, float(ext_t[0]))
    if target_end <= summary_start + 1e-9:
        return np.nan

    query_time_days = build_plot_time_grid(
        observed_time_days=ext_t,
        start_day=summary_start,
        target_day=target_end,
        step_days=float(step_days),
    )
    if query_time_days.size < int(max(1, min_target_points)):
        return np.nan

    query_orp = np.interp(query_time_days, ext_t, ext_y)
    return trimmed_mean(query_orp, trim_quantile=trim_quantile)


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


def summarize_recent_positive_rate_catalyst_window(
    time_days: np.ndarray,
    catalyst_cum: np.ndarray,
    window_days: float,
    week_days: float = 7.0,
) -> Dict[str, float]:
    t, c = prepare_cumulative_profile_with_time(time_days, catalyst_cum, force_zero=False)
    if t.size <= 1:
        return {
            "start_day": np.nan,
            "end_day": np.nan,
            "effective_window_days": 0.0,
            "total_addition": 0.0,
            "weekly_addition": 0.0,
        }

    tol = max(1e-9, 1e-6 * float(max(1.0, np.nanmax(np.abs(c)))))
    dt = np.diff(t)
    dc = np.maximum(0.0, np.diff(c))
    positive_mask = (dt > 1e-9) & (dc > tol)
    if positive_mask.sum() == 0:
        return {
            "start_day": np.nan,
            "end_day": np.nan,
            "effective_window_days": 0.0,
            "total_addition": 0.0,
            "weekly_addition": 0.0,
        }

    interval_start = t[:-1][positive_mask]
    interval_end = t[1:][positive_mask]
    interval_dt = dt[positive_mask]
    interval_dc = dc[positive_mask]
    interval_rate = interval_dc / np.maximum(interval_dt, 1e-9)

    remaining_days = float(max(0.0, window_days))
    selected_segments: List[Tuple[float, float, float]] = []
    total_addition = 0.0
    total_days = 0.0

    for idx in range(interval_start.size - 1, -1, -1):
        if remaining_days <= 1e-9:
            break
        seg_start = float(interval_start[idx])
        seg_end = float(interval_end[idx])
        seg_rate = float(interval_rate[idx])
        seg_days = max(0.0, seg_end - seg_start)
        if seg_days <= 1e-9 or seg_rate <= 0.0:
            continue

        take_days = min(seg_days, remaining_days)
        take_start = seg_end - take_days
        selected_segments.append((take_start, seg_end, seg_rate))
        total_addition += seg_rate * take_days
        total_days += take_days
        remaining_days -= take_days

    if total_days <= 1e-9 or len(selected_segments) == 0:
        return {
            "start_day": np.nan,
            "end_day": np.nan,
            "effective_window_days": 0.0,
            "total_addition": 0.0,
            "weekly_addition": 0.0,
        }

    selected_segments.reverse()
    start_day = float(selected_segments[0][0])
    end_day = float(selected_segments[-1][1])
    weekly_addition = total_addition / total_days * float(week_days)
    return {
        "start_day": start_day,
        "end_day": end_day,
        "effective_window_days": float(total_days),
        "total_addition": float(total_addition),
        "weekly_addition": float(max(0.0, weekly_addition)),
    }


def average_weekly_catalyst_from_recent_history(
    time_days: np.ndarray,
    catalyst_cum: np.ndarray,
    window_days: float = 21.0,
    week_days: float = 7.0,
) -> Tuple[float, float]:
    recent_rate_window = summarize_recent_positive_rate_catalyst_window(
        time_days=time_days,
        catalyst_cum=catalyst_cum,
        window_days=window_days,
        week_days=week_days,
    )
    return (
        float(recent_rate_window["weekly_addition"]),
        float(recent_rate_window["effective_window_days"]),
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


def reconstruct_catalyst_feed_conc_mg_l(
    catalyst_cum_kg_t: np.ndarray,
    lixiviant_cum_m3_t: np.ndarray,
) -> np.ndarray:
    """Back-calculate per-step catalyst feed concentration (mg/L) from cumulative differentials.

    Symmetric counterpart to ``convert_lixiviant_cum_to_irrigation_rate_l_m2_h``.
    Both arrays must already be on the same time grid (same length, cleaned, aligned).

    Formula per time step i:

        C_feed[i]  =  Δ(cum_catalyst_kg_t[i]) × 1000
                      ─────────────────────────────────
                         Δ(cum_lixiviant_m3_t[i])

    The feed_mass_kg factor cancels from numerator and denominator because both
    cumulative profiles are expressed per tonne of ore:

        Δcat_mg  = Δcat [kg/t] × feed_t × 1 e6  →  numerator  ∝ Δcat × feed_t
        Δlix_L   = Δlix [m³/t] × feed_t × 1000  →  denominator ∝ Δlix × feed_t
        C_feed   = Δcat × 1000 / Δlix              [mg/L]

    Returns zero on intervals where Δlix = 0 (no lixiviant flow, no concentration
    signal).  This vector is used as the explicit pre-computed C_feed input to the
    CSTR model and is derived entirely from the cumulative source columns.
    """
    cc = clean_cumulative_profile(np.asarray(catalyst_cum_kg_t, dtype=float), force_zero=False)
    cl = clean_cumulative_profile(np.asarray(lixiviant_cum_m3_t, dtype=float), force_zero=False)
    n = min(cc.size, cl.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    cc = cc[:n]
    cl = cl[:n]
    d_cat = cumulative_interval_deltas(cc)
    d_lix = cumulative_interval_deltas(cl)
    eps = 1e-9
    c_feed = np.where(d_lix > eps, d_cat * 1000.0 / np.maximum(d_lix, eps), 0.0)
    return np.where(np.isfinite(c_feed), c_feed, 0.0)


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


# ---------------------------
# v11 Catalyst / Lixiviant Reconstruction
# ---------------------------
def resolve_cumulative_profile_on_time_grid(
    time_days: np.ndarray,
    cumulative_profile: Any,
    force_zero: bool = False,
) -> np.ndarray:
    t = np.asarray(time_days, dtype=float)
    n = t.size
    if n == 0:
        return np.zeros(0, dtype=float)

    raw = parse_listlike(cumulative_profile)
    if raw.size == 0:
        return np.zeros(n, dtype=float)
    if raw.size == n:
        return clean_cumulative_profile(raw, force_zero=force_zero)

    prof_t, prof_c = prepare_cumulative_profile_with_time(t, raw, force_zero=force_zero)
    if prof_t.size == 0:
        return np.zeros(n, dtype=float)
    if prof_t.size == 1:
        fill_value = float(prof_c[0]) if np.isfinite(prof_c[0]) else 0.0
        return clean_cumulative_profile(
            np.full(n, max(0.0, fill_value), dtype=float),
            force_zero=force_zero,
        )
    return clean_cumulative_profile(
        np.interp(t, prof_t, prof_c, left=0.0, right=float(prof_c[-1])),
        force_zero=force_zero,
    )


def resolve_cumulative_catalyst_profile_kg_t(
    time_days: np.ndarray,
    cumulative_catalyst_kg_t_ref: Any,
) -> np.ndarray:
    """Align and clean the authoritative cumulative catalyst profile (kg/t)."""
    return resolve_cumulative_profile_on_time_grid(
        time_days=time_days,
        cumulative_profile=cumulative_catalyst_kg_t_ref,
        force_zero=False,
    )


def derive_catalyst_feed_conc_from_cumulative_profiles(
    time_days: np.ndarray,
    cumulative_catalyst_kg_t_ref: Any,
    cumulative_lixiviant_m3_t: Any,
) -> np.ndarray:
    """Back-calculate catalyst feed concentration (mg/L) from cumulative source columns."""
    catalyst_cum = resolve_cumulative_catalyst_profile_kg_t(time_days, cumulative_catalyst_kg_t_ref)
    lixiviant_cum = resolve_cumulative_profile_on_time_grid(
        time_days=time_days,
        cumulative_profile=cumulative_lixiviant_m3_t,
        force_zero=False,
    )
    return reconstruct_catalyst_feed_conc_mg_l(
        catalyst_cum_kg_t=catalyst_cum,
        lixiviant_cum_m3_t=lixiviant_cum,
    )


def _compute_cstr_column_concentration(
    time_days: np.ndarray,
    column_inner_diameter_m: float,
    column_height_m: float,
    irrigation_rate_l_m2_h: Optional[np.ndarray] = None,
    cumulative_catalyst_kg_t_ref: Any = None,
    cumulative_lixiviant_m3_t: Any = None,
    catalyst_feed_conc_mg_l: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate the average catalyst concentration (mg/L) inside the column pore
    solution at each observation time using a CSTR (Continuously Stirred Tank
    Reactor) residence-time model.

    Discrete update equation (per time step i):
        C_col[i] = C_col[i-1] × exp(-Δt / τ) + C_feed[i] × (1 - exp(-Δt / τ))

    where:
        τ  = pore_vol_L / (flow_rate_L_day)   [days]
        pore_vol_L = π × (d/2)² × height_m × COLUMN_POROSITY × 1000  [L]
        flow_rate_L_day = irrigation_rate_l_m2_h × area_m2 × 24  [L/day]

    The model represents the column pore space as a single well-mixed reactor
    whose average concentration responds to the feed concentration C_feed with
    a first-order lag determined by the hydraulic residence time τ.

    Feed concentration priority:
      0. catalyst_feed_conc_mg_l pre-computed array (EXPLICIT PRIMARY — normally
         the catalyst_addition_mg_l vector aligned to time_days; for rows lacking
         that vector, reconstructed from cumulative differentials). Bypasses all
         internal derivation.
      1. Derived directly from cumulative_catalyst_kg_t_ref and
         cumulative_lixiviant_m3_t.

    Parameters
    ----------
    time_days : weekly observation time points (days).
    column_inner_diameter_m : inner column diameter (m).
    column_height_m : packed bed height (m).
    irrigation_rate_l_m2_h : irrigation rate profile (L/m²/h), already reconstructed
        from cumulative_lixiviant_m3_t via convert_lixiviant_cum_to_irrigation_rate_l_m2_h.
    cumulative_catalyst_kg_t_ref : cumulative catalyst profile (kg/t).
    cumulative_lixiviant_m3_t : cumulative lixiviant in m³/t.
    catalyst_feed_conc_mg_l : pre-computed C_feed vector [mg/L] on the same time grid.
        When provided this is used directly and internal derivation is skipped.

    Returns
    -------
    np.ndarray  average pore-solution catalyst concentration (mg/L),
                shape = (len(time_days),).  Returns zeros if geometry is
                missing or catalyst feed is zero.
    """
    t = np.asarray(time_days, dtype=float)
    n = t.size
    if n == 0:
        return np.zeros(0, dtype=float)

    # ---- Feed concentration profile C_feed[i] (mg/L) ----
    # EXPLICIT PRIMARY: pre-computed/resolved feed concentration.
    # This vector is already on the same time grid (same length) so use directly.
    if catalyst_feed_conc_mg_l is not None and np.asarray(catalyst_feed_conc_mg_l).size == n:
        C_feed = np.where(
            np.isfinite(np.asarray(catalyst_feed_conc_mg_l, dtype=float)),
            np.asarray(catalyst_feed_conc_mg_l, dtype=float),
            0.0,
        )
    else:
        C_feed = derive_catalyst_feed_conc_from_cumulative_profiles(
            time_days=t,
            cumulative_catalyst_kg_t_ref=cumulative_catalyst_kg_t_ref,
            cumulative_lixiviant_m3_t=cumulative_lixiviant_m3_t,
        )
        if C_feed.size != n:
            return np.zeros(n, dtype=float)
        C_feed = np.where(np.isfinite(C_feed), C_feed, 0.0)

    # ---- Column geometry → pore volume ----
    d = float(column_inner_diameter_m) if np.isfinite(float(column_inner_diameter_m)) else np.nan
    h = float(column_height_m) if np.isfinite(float(column_height_m)) else np.nan
    if not (np.isfinite(d) and d > 0.0 and np.isfinite(h) and h > 0.0):
        # Geometry unknown – return steady-state approximation: C_col ≈ C_feed
        # (single mixing-fraction parameter; model can compensate).
        return C_feed.copy()

    area_m2 = np.pi * (d / 2.0) ** 2
    pore_vol_L = area_m2 * h * COLUMN_POROSITY * 1000.0  # m² × m × fraction × 1000 L/m³

    # ---- Flow rate (L/day) ----
    if irrigation_rate_l_m2_h is not None:
        irr = np.asarray(irrigation_rate_l_m2_h, dtype=float)
        if irr.size == n:
            flow_L_day = np.where(np.isfinite(irr), irr, 0.0) * area_m2 * 24.0
        elif irr.size > 1:
            flow_L_day = (
                np.interp(t, np.linspace(float(t[0]), float(t[-1]), irr.size), irr)
                * area_m2 * 24.0
            )
        else:
            flow_L_day = np.zeros(n, dtype=float)
    else:
        flow_L_day = np.zeros(n, dtype=float)

    # ---- CSTR discrete update ----
    C_col = np.zeros(n, dtype=float)
    for i in range(n):
        f = max(float(flow_L_day[i]), 0.0)
        if f > 1e-9:
            tau_days = pore_vol_L / f
        else:
            tau_days = 1e9  # near-stagnant: concentration frozen

        dt = float(t[i] - t[i - 1]) if i > 0 else float(t[0]) if float(t[0]) > 0.0 else 7.0
        dt = max(dt, 0.0)

        alpha = np.exp(-dt / tau_days) if tau_days < 1e6 else 1.0
        prev = C_col[i - 1] if i > 0 else 0.0
        C_col[i] = prev * alpha + float(C_feed[i]) * (1.0 - alpha)

    return C_col


def compute_catalyst_column_signals(
    time_days: np.ndarray,
    cumulative_catalyst_kg_t_ref: Any,
    cumulative_lixiviant_m3_t: Any,
    column_inner_diameter_m: float,
    column_height_m: float,
    irrigation_rate_l_m2_h: Optional[np.ndarray] = None,
    catalyst_feed_conc_mg_l: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute both catalyst dynamic signals for the v11 model.

    Returns
    -------
    catalyst_cum_kg_t : np.ndarray
        Cumulative catalyst added to the system (kg catalyst / t ore).
        Serves as a proxy for "active catalyst mass" and is taken directly from
        the authoritative `cumulative_catalyst_addition_kg_t` source column.

    catalyst_conc_col_mg_l : np.ndarray
        Average catalyst concentration in the column pore solution (mg/L).
        Computed via a CSTR residence-time model from the resolved feed
        concentration and column geometry.  This signal captures the
        *instantaneous probability* of catalyst-chalcopyrite contact.
    """
    catalyst_cum_kg_t = resolve_cumulative_catalyst_profile_kg_t(
        time_days=time_days,
        cumulative_catalyst_kg_t_ref=cumulative_catalyst_kg_t_ref,
    )
    if catalyst_feed_conc_mg_l is None or np.asarray(catalyst_feed_conc_mg_l).size != np.asarray(time_days).size:
        catalyst_feed_conc_mg_l = derive_catalyst_feed_conc_from_cumulative_profiles(
            time_days=time_days,
            cumulative_catalyst_kg_t_ref=cumulative_catalyst_kg_t_ref,
            cumulative_lixiviant_m3_t=cumulative_lixiviant_m3_t,
        )
    catalyst_conc_col_mg_l = _compute_cstr_column_concentration(
        time_days=time_days,
        column_inner_diameter_m=column_inner_diameter_m,
        column_height_m=column_height_m,
        irrigation_rate_l_m2_h=irrigation_rate_l_m2_h,
        cumulative_catalyst_kg_t_ref=cumulative_catalyst_kg_t_ref,
        cumulative_lixiviant_m3_t=cumulative_lixiviant_m3_t,
        catalyst_feed_conc_mg_l=catalyst_feed_conc_mg_l,
    )

    return catalyst_cum_kg_t, catalyst_conc_col_mg_l


def resolve_aligned_lixiviant_cumulative_m3_t(
    source_time_days: np.ndarray,
    target_time_days: np.ndarray,
    cumulative_lixiviant_m3_t: Any,
) -> Optional[np.ndarray]:
    """Resolve the authoritative cumulative lixiviant profile on the aligned grid."""
    t_source = np.asarray(source_time_days, dtype=float)
    t_target = np.asarray(target_time_days, dtype=float)
    if t_target.size == 0:
        return np.zeros(0, dtype=float)

    lix_m3t_raw = parse_listlike(cumulative_lixiviant_m3_t)
    lix_m3t_t, lix_m3t = prepare_cumulative_profile_with_time(t_source, lix_m3t_raw, force_zero=False)
    if lix_m3t_t.size < 2:
        return None
    return clean_cumulative_profile(
        np.interp(t_target, lix_m3t_t, lix_m3t, left=0.0, right=float(lix_m3t[-1])),
        force_zero=False,
    )


def reconstruct_aligned_dynamic_sequences_for_row(
    row: pd.Series,
    min_points: int = 6,
) -> Optional[Dict[str, np.ndarray]]:
    """Build all dynamic catalyst/lixiviant sequences on the filtered model time grid.

    Returns ``None`` when the row lacks the source profiles needed to build the
    aligned dynamic inputs. The explicit catalyst_addition_mg_l vector is used
    for feed concentration when available; cumulative-profile reconstruction is
    the fallback for catalyzed rows without that vector.
    """
    status = normalize_status(
        row.get(STATUS_COL_PRIMARY, row.get(STATUS_COL_FALLBACK, ""))
    )
    t_raw = parse_listlike(row.get(TIME_COL_COLUMNS, np.nan))
    y_raw = parse_listlike(row.get(TARGET_COLUMNS, np.nan))
    feed_mass_kg = scalar_from_maybe_array(row.get(FEED_MASS_COL, np.nan))
    column_inner_diameter_m = scalar_from_maybe_array(row.get("column_inner_diameter_m", np.nan))
    column_height_m = scalar_from_maybe_array(row.get("column_height_m", np.nan))

    if status != "Control":
        ref_cum_raw = parse_listlike(row.get(CATALYST_CUM_COL, np.nan))
        has_ref_cumulative = ref_cum_raw.size >= 2 and np.any(np.isfinite(ref_cum_raw) & (ref_cum_raw > 0.0))
        if not has_ref_cumulative:
            return None

        catalyst_cum_raw = resolve_cumulative_catalyst_profile_kg_t(
            time_days=t_raw,
            cumulative_catalyst_kg_t_ref=row.get(CATALYST_CUM_COL, np.nan),
        )
    else:
        catalyst_cum_raw = np.zeros(len(t_raw), dtype=float)

    t_aligned, y_aligned, catalyst_cum_aligned = prepare_curve_arrays(
        t_raw,
        y_raw,
        catalyst_cum_raw,
        status=status,
        min_points=min_points,
    )
    if t_aligned.size < min_points:
        return None

    lixiviant_cum_aligned = resolve_aligned_lixiviant_cumulative_m3_t(
        source_time_days=t_raw,
        target_time_days=t_aligned,
        cumulative_lixiviant_m3_t=row.get(LIXIVIANT_CUM_COL, np.nan),
    )
    if lixiviant_cum_aligned is None or np.asarray(lixiviant_cum_aligned).size != t_aligned.size:
        return None

    irrigation_rate_aligned = convert_lixiviant_cum_to_irrigation_rate_l_m2_h(
        time_days=t_aligned,
        cumulative_lixiviant_m3_t=lixiviant_cum_aligned,
        feed_mass_kg=feed_mass_kg,
        column_inner_diameter_m=column_inner_diameter_m,
    )

    if status == "Control":
        catalyst_addition_reconstructed = np.zeros_like(t_aligned, dtype=float)
        catalyst_conc_col_aligned = np.zeros_like(t_aligned, dtype=float)
        catalyst_addition_source = "control_zero"
    else:
        if not np.any(np.isfinite(catalyst_cum_aligned) & (catalyst_cum_aligned > 1e-12)):
            return None
        explicit_catalyst_addition = resolve_observed_profile_on_time_grid(
            source_time_days=t_raw,
            target_time_days=t_aligned,
            observed_profile=row.get(CATALYST_ADDITION_COL, np.nan),
            clip_min=0.0,
        )
        if explicit_catalyst_addition is not None:
            catalyst_addition_reconstructed = np.asarray(explicit_catalyst_addition, dtype=float)
            catalyst_addition_source = CATALYST_ADDITION_COL
        else:
            catalyst_addition_reconstructed = reconstruct_catalyst_feed_conc_mg_l(
                catalyst_cum_kg_t=catalyst_cum_aligned,
                lixiviant_cum_m3_t=lixiviant_cum_aligned,
            )
            catalyst_addition_source = CATALYST_ADDITION_RECON_COL
        catalyst_conc_col_aligned = _compute_cstr_column_concentration(
            time_days=t_aligned,
            column_inner_diameter_m=column_inner_diameter_m,
            column_height_m=column_height_m,
            irrigation_rate_l_m2_h=np.asarray(irrigation_rate_aligned, dtype=float),
            catalyst_feed_conc_mg_l=np.asarray(catalyst_addition_reconstructed, dtype=float),
        )
        if catalyst_conc_col_aligned.size != t_aligned.size:
            if catalyst_conc_col_aligned.size >= 2:
                catalyst_conc_col_aligned = np.interp(
                    t_aligned,
                    np.linspace(0.0, float(t_aligned[-1]), catalyst_conc_col_aligned.size),
                    catalyst_conc_col_aligned,
                    left=float(catalyst_conc_col_aligned[0]),
                    right=float(catalyst_conc_col_aligned[-1]),
                )
            else:
                catalyst_conc_col_aligned = np.zeros_like(t_aligned, dtype=float)

    return {
        "time_days_aligned": np.asarray(t_aligned, dtype=float),
        "recovery_aligned": np.asarray(y_aligned, dtype=float),
        "catalyst_cum_aligned_kg_t": np.asarray(catalyst_cum_aligned, dtype=float),
        "lixiviant_cum_aligned_m3_t": np.asarray(lixiviant_cum_aligned, dtype=float),
        CATALYST_ADDITION_RECON_COL: np.asarray(catalyst_addition_reconstructed, dtype=float),
        "catalyst_addition_mg_l_source": catalyst_addition_source,
        IRRIGATION_RATE_RECON_COL: np.asarray(irrigation_rate_aligned, dtype=float),
        "catalyst_conc_col_mg_l_reconstructed": np.asarray(catalyst_conc_col_aligned, dtype=float),
    }


def append_reconstructed_sequence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add aligned reconstructed sequence columns used by training and the app."""
    if df.empty:
        out = df.copy()
        out[ALIGNED_TIME_COL] = [[] for _ in range(len(out))]
        out[CATALYST_ADDITION_RECON_COL] = [[] for _ in range(len(out))]
        out[IRRIGATION_RATE_RECON_COL] = [[] for _ in range(len(out))]
        return out

    out = df.copy()
    aligned_time_values: List[Any] = []
    catalyst_values: List[Any] = []
    irrigation_values: List[Any] = []
    for _, row in out.iterrows():
        dynamic = reconstruct_aligned_dynamic_sequences_for_row(row, min_points=6)
        if dynamic is None:
            aligned_time_values.append(np.nan)
            catalyst_values.append(np.nan)
            irrigation_values.append(np.nan)
            continue
        aligned_time_values.append(np.asarray(dynamic["time_days_aligned"], dtype=float).tolist())
        catalyst_values.append(np.asarray(dynamic[CATALYST_ADDITION_RECON_COL], dtype=float).tolist())
        irrigation_values.append(np.asarray(dynamic[IRRIGATION_RATE_RECON_COL], dtype=float).tolist())

    out[ALIGNED_TIME_COL] = aligned_time_values
    out[CATALYST_ADDITION_RECON_COL] = catalyst_values
    out[IRRIGATION_RATE_RECON_COL] = irrigation_values
    return out


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
    # The cumulative increase between t[i] and t[i+1] means the sample at t[i]
    # is still pre-addition. Treat the first observed post-rise point as the
    # catalyst-onset sample so internal catalyst physics never starts before the
    # addition is actually reflected in the dosing history.
    start_idx = min(int(rise_idx[0] + 1), int(t.size - 1))
    return float(t[start_idx])


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


def infer_catalyst_start_day_from_dosage_array(
    time_days: np.ndarray,
    dosage_mg_l: np.ndarray,
) -> float:
    """Return the first day on which the resolved feed concentration is positive.

    Uses the feed-concentration array rather than the cumulative proxy so
    that the annotation is immune to cumulative reconstruction artefacts.

    Returns np.nan when no valid nonzero dosage entry is found.
    """
    t = np.asarray(time_days, dtype=float)
    d = np.asarray(dosage_mg_l, dtype=float)
    if t.size == 0 or d.size == 0:
        return np.nan
    n = min(t.size, d.size)
    t, d = t[:n], d[:n]
    tol = max(1e-9, 1e-6 * float(np.nanmax(np.abs(d))) if np.any(np.isfinite(d)) else 1e-9)
    active = np.flatnonzero(np.isfinite(d) & (d > tol))
    if active.size == 0:
        return np.nan
    return float(t[int(active[0])])


def infer_catalyst_stop_day_from_dosage_array(
    time_days: np.ndarray,
    dosage_mg_l: np.ndarray,
) -> float:
    """Return the last day on which the resolved feed concentration is positive.

    Returns np.nan if catalyst addition continues through the final observation
    (i.e. addition has not yet stopped within the observed window) — this
    matches the semantics of ``infer_catalyst_addition_stop_day`` which also
    returns nan for still-active columns.

    Uses the feed-concentration array rather than the cumulative proxy.
    """
    t = np.asarray(time_days, dtype=float)
    d = np.asarray(dosage_mg_l, dtype=float)
    if t.size == 0 or d.size == 0:
        return np.nan
    n = min(t.size, d.size)
    t, d = t[:n], d[:n]
    tol = max(1e-9, 1e-6 * float(np.nanmax(np.abs(d))) if np.any(np.isfinite(d)) else 1e-9)
    active = np.flatnonzero(np.isfinite(d) & (d > tol))
    if active.size == 0:
        return np.nan
    last_active = int(active[-1])
    # Catalyst still active at last observation → stop day unknown
    if last_active >= n - 1:
        return np.nan
    return float(t[last_active])


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
        time_days=time_days,
        catalyst_cum=catalyst_cum,
        window_days=history_window_days,
        week_days=7.0,
    )
    recent_rate_window = summarize_recent_positive_rate_catalyst_window(
        time_days=time_days,
        catalyst_cum=catalyst_cum,
        window_days=history_window_days,
        week_days=7.0,
    )
    last_day = float(t[-1])
    recent_window_start_day = float(recent_rate_window["start_day"])
    recent_window_delta = float(recent_rate_window["total_addition"])
    recent_window_tol = max(1e-9, 1e-6 * float(max(1.0, np.nanmax(np.abs(c)))))
    recent_growth_near_zero = bool(recent_window_delta <= recent_window_tol)

    # Stop-state detection should stay tied to the full observed profile, not the
    # trimmed valid-rate window used for extrapolation.
    full_recent_window_start_day = max(float(t[0]), last_day - float(history_window_days))
    if last_day > full_recent_window_start_day + 1e-9:
        c_start_full = float(np.interp(full_recent_window_start_day, t, c))
        c_end_full = float(c[-1])
        full_recent_window_delta = max(0.0, c_end_full - c_start_full)
    else:
        full_recent_window_delta = 0.0
    full_recent_window_tol = max(1e-9, 1e-6 * float(max(1.0, np.nanmax(np.abs(c)))))
    full_recent_growth_near_zero = bool(full_recent_window_delta <= full_recent_window_tol)
    stopped_before_test_end = bool(
        full_recent_growth_near_zero
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
        behavior = summarize_catalyst_addition_behavior(
            time_days=pair.catalyzed.time,
            catalyst_cum=pair.catalyzed.catalyst_cum,
            history_window_days=history_window_days,
        )
        _start = float(pair.catalyzed.catalyst_start_day)
        _stop = float(pair.catalyzed.catalyst_stop_day)
        if np.isfinite(_start):
            behavior["catalyst_addition_start_day"] = _start
        if np.isfinite(_stop):
            behavior["catalyst_addition_stop_day"] = _stop
        row = {
            "pair_id": pair.pair_id,
            "sample_id": pair.sample_id,
            **behavior,
        }
        rows.append(row)

    report_df = pd.DataFrame(rows)
    if not report_df.empty:
        report_df = report_df.sort_values(
            ["stopped_before_test_end", "sample_id", "pair_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    stopped_pair_ids = (
        report_df.loc[report_df["stopped_before_test_end"], "pair_id"].astype(str).tolist()
        if not report_df.empty
        else []
    )
    stopped_sample_ids = (
        sorted(report_df.loc[report_df["stopped_before_test_end"], "sample_id"].astype(str).unique().tolist())
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
        "n_stopped_before_test_end": int(len(stopped_pair_ids)),
        "stopped_pair_ids": stopped_pair_ids,
        "stopped_sample_ids": stopped_sample_ids,
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

    def _smooth_positive_gap(gap_values: np.ndarray) -> np.ndarray:
        t = np.asarray(time_days, dtype=float)
        gaps = np.clip(np.asarray(gap_values, dtype=float), 0.0, None)
        if gaps.size <= 2 or not np.isfinite(smoothing_days) or smoothing_days <= 0.0:
            return gaps

        valid = np.isfinite(t) & np.isfinite(gaps)
        if int(np.sum(valid)) <= 2:
            return gaps

        t_valid = t[valid]
        gaps_valid = gaps[valid]
        dt = np.diff(t_valid)
        dt = dt[np.isfinite(dt) & (dt > 1e-9)]
        if dt.size == 0:
            return gaps

        sigma_points = max(
            float(smoothing_days) / max(float(np.nanmedian(dt)), 1e-9) / 2.355,
            0.75,
        )
        smoothed_valid = np.expm1(
            gaussian_filter1d(
                np.log1p(gaps_valid),
                sigma=sigma_points,
                mode="nearest",
            )
        )
        smoothed_valid = np.clip(smoothed_valid, 0.0, None)

        out = gaps.copy()
        out[valid] = smoothed_valid
        out[~valid] = np.nan
        return out

    low_gap_smoothed = _smooth_positive_gap(mean_arr - np.asarray(low_curve, dtype=float))
    high_gap_smoothed = _smooth_positive_gap(np.asarray(high_curve, dtype=float) - mean_arr)

    low_smoothed = np.clip(mean_arr - low_gap_smoothed, 0.0, 100.0)
    high_smoothed = np.clip(mean_arr + high_gap_smoothed, 0.0, 100.0)
    low_smoothed = np.maximum.accumulate(np.minimum(low_smoothed, mean_arr))
    high_smoothed = np.maximum.accumulate(np.maximum(high_smoothed, mean_arr))
    low_smoothed = np.minimum(low_smoothed, high_smoothed)
    return low_smoothed, high_smoothed


def _finite_sorted_unique_curve(time_days: Any, values: Any) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_days, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    n = min(t.size, y.size)
    if n <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    t = t[:n]
    y = y[:n]
    valid = np.isfinite(t) & np.isfinite(y)
    if not np.any(valid):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    t_valid = t[valid]
    y_valid = y[valid]
    order = np.argsort(t_valid)
    t_sorted = t_valid[order]
    y_sorted = y_valid[order]

    unique_t, inverse = np.unique(t_sorted, return_inverse=True)
    if unique_t.size == t_sorted.size:
        return unique_t.astype(float), y_sorted.astype(float)

    unique_y = np.empty(unique_t.size, dtype=float)
    for idx in range(unique_t.size):
        with np.errstate(all="ignore"):
            unique_y[idx] = float(np.nanmean(y_sorted[inverse == idx]))
    return unique_t.astype(float), unique_y


def widen_predictive_interval_to_cover_reference(
    time_days: np.ndarray,
    low_curve: np.ndarray,
    high_curve: np.ndarray,
    reference_time: np.ndarray,
    reference_curve: np.ndarray,
    margin_pct: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_days, dtype=float).reshape(-1)
    low = np.asarray(low_curve, dtype=float).reshape(-1).copy()
    high = np.asarray(high_curve, dtype=float).reshape(-1).copy()
    if t.size == 0 or low.size != t.size or high.size != t.size:
        return low, high

    ref_t, ref_y = _finite_sorted_unique_curve(reference_time, reference_curve)
    if ref_t.size == 0:
        return low, high

    ref_on_t = np.full(t.shape, np.nan, dtype=float)
    t_valid = np.isfinite(t)
    if ref_t.size == 1:
        exact = t_valid & np.isclose(t, ref_t[0], rtol=0.0, atol=1e-6)
        ref_on_t[exact] = ref_y[0]
    else:
        in_ref_range = t_valid & (t >= ref_t[0] - 1e-9) & (t <= ref_t[-1] + 1e-9)
        ref_on_t[in_ref_range] = np.interp(t[in_ref_range], ref_t, ref_y)

    margin = max(float(margin_pct), 0.0)
    below = np.isfinite(ref_on_t) & np.isfinite(low) & (ref_on_t < low)
    above = np.isfinite(ref_on_t) & np.isfinite(high) & (ref_on_t > high)
    low[below] = ref_on_t[below] - margin
    high[above] = ref_on_t[above] + margin

    low = np.clip(low, 0.0, 100.0)
    high = np.clip(high, 0.0, 100.0)
    low = np.minimum(low, high)
    return low, high


def predictive_interval_plot_band_with_reference(
    time_days: np.ndarray,
    low_curve: np.ndarray,
    high_curve: np.ndarray,
    reference_time: np.ndarray,
    reference_curve: np.ndarray,
    margin_pct: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    band_t, band_low = _finite_sorted_unique_curve(time_days, low_curve)
    high_t, band_high = _finite_sorted_unique_curve(time_days, high_curve)
    if band_t.size == 0 or high_t.size == 0:
        return band_t, band_low, np.asarray([], dtype=float)
    if band_t.size != high_t.size or not np.allclose(band_t, high_t, rtol=0.0, atol=1e-6):
        common_t = np.union1d(band_t, high_t)
        band_low = np.interp(common_t, band_t, band_low)
        band_high = np.interp(common_t, high_t, band_high)
        band_t = common_t

    ref_t, ref_y = _finite_sorted_unique_curve(reference_time, reference_curve)
    if ref_t.size > 0:
        ref_in_band = ref_t[(ref_t >= band_t[0] - 1e-9) & (ref_t <= band_t[-1] + 1e-9)]
        if ref_in_band.size > 0:
            dense_t = np.union1d(band_t, ref_in_band)
            band_low = np.interp(dense_t, band_t, band_low)
            band_high = np.interp(dense_t, band_t, band_high)
            band_t = dense_t

    band_low, band_high = widen_predictive_interval_to_cover_reference(
        time_days=band_t,
        low_curve=band_low,
        high_curve=band_high,
        reference_time=reference_time,
        reference_curve=reference_curve,
        margin_pct=margin_pct,
    )
    return band_t, band_low, band_high


def add_member_prediction_gap_bands(record: Dict[str, Any]) -> Dict[str, Any]:
    margin_pct = float(CONFIG.get("member_prediction_gap_margin_pct", 0.0))
    ctrl_low, ctrl_high = widen_predictive_interval_to_cover_reference(
        time_days=np.asarray(record.get("control_t", []), dtype=float),
        low_curve=np.asarray(record.get("control_pred", []), dtype=float),
        high_curve=np.asarray(record.get("control_pred", []), dtype=float),
        reference_time=np.asarray(record.get("control_t", []), dtype=float),
        reference_curve=np.asarray(record.get("control_true", []), dtype=float),
        margin_pct=margin_pct,
    )
    cat_low, cat_high = widen_predictive_interval_to_cover_reference(
        time_days=np.asarray(record.get("catalyzed_t", []), dtype=float),
        low_curve=np.asarray(record.get("catalyzed_pred", []), dtype=float),
        high_curve=np.asarray(record.get("catalyzed_pred", []), dtype=float),
        reference_time=np.asarray(record.get("catalyzed_t", []), dtype=float),
        reference_curve=np.asarray(record.get("catalyzed_true", []), dtype=float),
        margin_pct=margin_pct,
    )
    record["control_pred_gap_low"] = ctrl_low
    record["control_pred_gap_high"] = ctrl_high
    record["catalyzed_pred_gap_low"] = cat_low
    record["catalyzed_pred_gap_high"] = cat_high

    if "control_pred_plot" in record:
        ctrl_plot_low, ctrl_plot_high = widen_predictive_interval_to_cover_reference(
            time_days=np.asarray(record.get("control_plot_time_days", record.get("control_t", [])), dtype=float),
            low_curve=np.asarray(record.get("control_pred_plot", []), dtype=float),
            high_curve=np.asarray(record.get("control_pred_plot", []), dtype=float),
            reference_time=np.asarray(record.get("control_t", []), dtype=float),
            reference_curve=np.asarray(record.get("control_true", []), dtype=float),
            margin_pct=margin_pct,
        )
        record["control_pred_plot_gap_low"] = ctrl_plot_low
        record["control_pred_plot_gap_high"] = ctrl_plot_high

    if "catalyzed_pred_plot" in record:
        cat_plot_low, cat_plot_high = widen_predictive_interval_to_cover_reference(
            time_days=np.asarray(record.get("catalyzed_plot_time_days", record.get("catalyzed_t", [])), dtype=float),
            low_curve=np.asarray(record.get("catalyzed_pred_plot", []), dtype=float),
            high_curve=np.asarray(record.get("catalyzed_pred_plot", []), dtype=float),
            reference_time=np.asarray(record.get("catalyzed_t", []), dtype=float),
            reference_curve=np.asarray(record.get("catalyzed_true", []), dtype=float),
            margin_pct=margin_pct,
        )
        record["catalyzed_pred_plot_gap_low"] = cat_plot_low
        record["catalyzed_pred_plot_gap_high"] = cat_plot_high
    return record


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
            "weekly_catalyst_extension_kg_t": 0.0,
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
        time_days=time_days,
        catalyst_cum=catalyst_cum,
        history_window_days=history_window_days,
    )
    weekly_value_raw = float(catalyst_behavior["weekly_catalyst_addition_kg_t"])
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
            **{
                **catalyst_behavior,
                "weekly_catalyst_addition_kg_t": float(weekly_value_raw),
                "weekly_catalyst_extension_kg_t": float(weekly_value_raw),
            },
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
    stop_day = float(catalyst_behavior.get("catalyst_addition_stop_day", np.nan))
    stopped_before_test_end = bool(catalyst_behavior.get("stopped_before_test_end", False))
    if np.isfinite(stop_day) and stopped_before_test_end and stop_day < last_day - 1e-9:
        future_cum = np.full_like(future_days_arr, float(c[-1]), dtype=float)
        weekly_extension_value = 0.0
    else:
        avg_daily_addition = weekly_value_raw / 7.0
        future_cum = float(c[-1]) + avg_daily_addition * (future_days_arr - last_day)
        weekly_extension_value = float(weekly_value_raw)
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
        **{
            **catalyst_behavior,
            "weekly_catalyst_addition_kg_t": float(weekly_value_raw),
            "weekly_catalyst_extension_kg_t": float(weekly_extension_value),
        },
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
    catalyst_feed_conc_mg_l: Optional[np.ndarray],
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
    explicit_profile_catalyst_addition = resolve_observed_profile_on_time_grid(
        source_time_days=time_days,
        target_time_days=profile_time_days,
        observed_profile=catalyst_feed_conc_mg_l,
        clip_min=0.0,
    )
    if explicit_profile_catalyst_addition is not None:
        profile_catalyst_addition = explicit_profile_catalyst_addition
    else:
        profile_catalyst_addition = reconstruct_catalyst_feed_conc_mg_l(
            catalyst_cum_kg_t=profile_catalyst_cum,
            lixiviant_cum_m3_t=profile_lixiviant_cum,
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
    explicit_plot_catalyst_addition = resolve_observed_profile_on_time_grid(
        source_time_days=profile_time_days,
        target_time_days=plot_time_days,
        observed_profile=profile_catalyst_addition,
        clip_min=0.0,
    )
    if explicit_plot_catalyst_addition is not None:
        plot_catalyst_addition = explicit_plot_catalyst_addition
    else:
        plot_catalyst_addition = reconstruct_catalyst_feed_conc_mg_l(
            catalyst_cum_kg_t=plot_catalyst_cum,
            lixiviant_cum_m3_t=plot_lixiviant_cum,
        )
    return {
        **catalyst_profile,
        "time_days": np.asarray(profile_time_days, dtype=float),
        "catalyst_cum": np.asarray(profile_catalyst_cum, dtype=float),
        "lixiviant_time_days": np.asarray(lix_profile["time_days"], dtype=float),
        "lixiviant_cum": np.asarray(profile_lixiviant_cum, dtype=float),
        "irrigation_rate_l_m2_h": np.asarray(profile_irrigation_rate, dtype=float),
        CATALYST_ADDITION_RECON_COL: np.asarray(profile_catalyst_addition, dtype=float),
        "weekly_lixiviant_addition_m3_t": float(lix_profile["weekly_addition"]),
        "lixiviant_weekly_reference_days": float(lix_profile["weekly_reference_days"]),
        "lixiviant_extension_applied": bool(lix_profile["extension_applied"]),
        "plot_time_days": np.asarray(plot_time_days, dtype=float),
        "plot_catalyst_cum": np.asarray(plot_catalyst_cum, dtype=float),
        "plot_lixiviant_cum": np.asarray(plot_lixiviant_cum, dtype=float),
        "plot_irrigation_rate_l_m2_h": np.asarray(plot_irrigation_rate, dtype=float),
        f"plot_{CATALYST_ADDITION_RECON_COL}": np.asarray(plot_catalyst_addition, dtype=float),
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

def compute_material_size_p80_cap_penalty(material_size_p80_in: float) -> float:
    """Shifted Hill penalty applied to chemistry-derived caps using P80 in inches."""
    p80_d0_in = float(CONFIG.get("cap_p80_penalty_d0_in", 2.0))
    p80_d50_in = float(CONFIG.get("cap_p80_penalty_d50_in", 3.0))
    p80_p_inf = float(CONFIG.get("cap_p80_penalty_p_inf", 0.40))
    p80_n = float(CONFIG.get("cap_p80_penalty_n", 2.0))

    d_in = float(material_size_p80_in) if np.isfinite(material_size_p80_in) else np.nan
    if not np.isfinite(d_in) or d_in <= 0.0 or d_in <= p80_d0_in:
        return 1.0

    p_inf = float(np.clip(p80_p_inf, 0.0, 1.0))
    d50 = max(1e-6, p80_d50_in)
    n = max(1e-6, p80_n)
    shifted_ratio = max(0.0, (d_in - p80_d0_in) / d50)
    return float(p_inf + (1.0 - p_inf) / (1.0 + shifted_ratio ** n))


def compute_residual_cpy_control_cap_factor(residual_cpy_pct: float) -> float:
    """Return a mild multiplicative reduction for the control primary fraction."""
    if not np.isfinite(residual_cpy_pct) or float(residual_cpy_pct) <= 0.0:
        return 1.0

    reduction_strength = float(CONFIG.get("control_cpy_cap_reduction_strength", 0.30))
    reduction_center_pct = float(CONFIG.get("control_cpy_cap_reduction_center_pct", 4.0))
    reduction_width_pct = float(CONFIG.get("control_cpy_cap_reduction_width_pct", 1.0))
    scaled_residual = (
        float(residual_cpy_pct) - reduction_center_pct
    ) / max(reduction_width_pct, 1e-6)
    reduction = reduction_strength / (1.0 + math.exp(-scaled_residual))
    return float(np.clip(1.0 - reduction, 0.05, 1.0))


def _sigmoid_scalar(value: float) -> float:
    if not np.isfinite(value):
        return 0.5
    clipped = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + math.exp(-clipped)))


def _signal_score_scalar(
    value: float,
    center: float,
    scale: float,
    *,
    invert: bool = False,
    log_input: bool = False,
) -> float:
    if not np.isfinite(value):
        return 0.5
    v = float(value)
    if log_input:
        v = math.log1p(max(v, 0.0))
        center = math.log1p(max(float(center), 0.0))
    z = (v - float(center)) / max(float(scale), 1e-6)
    if invert:
        z = -z
    return _sigmoid_scalar(z)


def compute_primary_sulfide_cap_prior_terms(
    *,
    cu_oxides_equiv: float,
    cu_secondary_equiv: float,
    cu_primary_equiv: float,
    residual_cpy_pct: float = np.nan,
    material_size_p80_in: float = np.nan,
    column_height_m: float = np.nan,
    lixiviant_initial_fe_mg_l: float = np.nan,
    lixiviant_initial_ph: float = np.nan,
    lixiviant_initial_orp_mv: float = np.nan,
    cyanide_soluble_pct: float = np.nan,
    acid_soluble_pct: float = np.nan,
    grouped_acid_generating_sulfides_pct: float = np.nan,
    grouped_carbonates_pct: float = np.nan,
) -> Dict[str, float]:
    base_ctrl = float(CONFIG.get("leach_pct_primary_sulfides_control", 0.33))
    base_cat = float(CONFIG.get("leach_pct_primary_sulfides_catalyzed", 0.75))
    ctrl_min = float(CONFIG.get("primary_control_prior_min", 0.20))
    ctrl_max = float(CONFIG.get("primary_control_prior_max", 0.55))
    cat_min = float(CONFIG.get("primary_catalyzed_prior_min", 0.60))
    cat_max = float(CONFIG.get("primary_catalyzed_prior_max", 0.75))

    fe_signal = _signal_score_scalar(
        lixiviant_initial_fe_mg_l,
        center=400.0,
        scale=0.80,
        log_input=True,
    )
    orp_signal = _signal_score_scalar(lixiviant_initial_orp_mv, center=440.0, scale=45.0)
    ph_signal = _signal_score_scalar(
        lixiviant_initial_ph,
        center=1.8,
        scale=0.25,
        invert=True,
    )
    cyanide_signal = _signal_score_scalar(cyanide_soluble_pct, center=8.0, scale=4.0)
    acid_signal = _signal_score_scalar(acid_soluble_pct, center=10.0, scale=6.0)
    acid_gen_signal = _signal_score_scalar(
        grouped_acid_generating_sulfides_pct,
        center=2.5,
        scale=1.5,
    )
    carbonate_signal = _signal_score_scalar(grouped_carbonates_pct, center=4.0, scale=2.0)
    secondary_signal = _signal_score_scalar(cu_secondary_equiv, center=0.20, scale=0.10)
    oxide_signal = _signal_score_scalar(cu_oxides_equiv, center=0.15, scale=0.08)
    primary_signal = _signal_score_scalar(cu_primary_equiv, center=0.50, scale=0.20)
    residual_cpy_signal = _signal_score_scalar(residual_cpy_pct, center=4.0, scale=1.25)
    height_signal = _signal_score_scalar(column_height_m, center=4.0, scale=1.0)
    p80_signal = _signal_score_scalar(material_size_p80_in, center=1.75, scale=0.50)
    tall_primary_burden = primary_signal * height_signal

    ferric_ready_raw = (
        1.25 * (fe_signal - 0.5)
        + 0.95 * (orp_signal - 0.5)
        + 0.85 * (ph_signal - 0.5)
        + 0.55 * (cyanide_signal - 0.5)
        + 0.45 * (acid_signal - 0.5)
        + 0.30 * (acid_gen_signal - 0.5)
        + 0.20 * (secondary_signal - 0.5)
        - 0.55 * (height_signal - 0.5)
        - 0.50 * (p80_signal - 0.5)
        - 0.30 * (carbonate_signal - 0.5)
        - 0.35 * (tall_primary_burden - 0.10)
    )
    ferric_ready_primary_score = _sigmoid_scalar(1.25 * ferric_ready_raw)

    tall_column_oxidant_deficit_raw = (
        1.20 * (height_signal - 0.5)
        + 1.00 * (p80_signal - 0.5)
        + 0.85 * (primary_signal - 0.5)
        + 0.80 * (residual_cpy_signal - 0.5)
        + 0.70 * ((1.0 - fe_signal) - 0.5)
        + 0.60 * ((1.0 - orp_signal) - 0.5)
        + 0.35 * ((1.0 - secondary_signal) - 0.5)
        + 0.20 * ((1.0 - oxide_signal) - 0.5)
        + 0.15 * (carbonate_signal - 0.5)
        + 0.75 * (tall_primary_burden - 0.10)
    )
    tall_column_oxidant_deficit_score = _sigmoid_scalar(1.20 * tall_column_oxidant_deficit_raw)

    primary_control_prior_frac = float(
        np.clip(
            base_ctrl
            + 0.18 * (ferric_ready_primary_score - 0.5)
            - 0.16 * (tall_column_oxidant_deficit_score - 0.5),
            ctrl_min,
            ctrl_max,
        )
    )
    primary_control_prior_frac = float(
        np.clip(
            primary_control_prior_frac - 0.15 * (tall_primary_burden - 0.10),
            ctrl_min,
            ctrl_max,
        )
    )
    primary_catalyzed_prior_frac = float(
        np.clip(
            base_cat
            + 0.02 * (ferric_ready_primary_score - 0.5)
            - 0.20 * (tall_column_oxidant_deficit_score - 0.5),
            cat_min,
            cat_max,
        )
    )
    primary_catalyzed_prior_frac = float(
        np.clip(
            primary_catalyzed_prior_frac - 0.10 * (tall_primary_burden - 0.10),
            cat_min,
            cat_max,
        )
    )
    primary_catalyzed_prior_frac = float(
        np.clip(
            max(primary_catalyzed_prior_frac, primary_control_prior_frac, cat_min),
            cat_min,
            cat_max,
        )
    )
    return {
        "base_primary_control_frac": float(base_ctrl),
        "base_primary_catalyzed_frac": float(base_cat),
        "primary_control_prior_frac": primary_control_prior_frac,
        "primary_catalyzed_prior_frac": primary_catalyzed_prior_frac,
        "ferric_ready_primary_score": ferric_ready_primary_score,
        "tall_column_oxidant_deficit_score": tall_column_oxidant_deficit_score,
    }


def compute_chemistry_only_leach_caps(
    cu_oxides_equiv: float,
    cu_secondary_equiv: float,
    cu_primary_equiv: float,
    residual_cpy_pct: float = np.nan,
    material_size_p80_in: float = np.nan,
    column_height_m: float = np.nan,
    lixiviant_initial_fe_mg_l: float = np.nan,
    lixiviant_initial_ph: float = np.nan,
    lixiviant_initial_orp_mv: float = np.nan,
    cyanide_soluble_pct: float = np.nan,
    acid_soluble_pct: float = np.nan,
    grouped_acid_generating_sulfides_pct: float = np.nan,
    grouped_carbonates_pct: float = np.nan,
    primary_control_fraction_override: Optional[float] = None,
    primary_catalyzed_fraction_override: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute chemistry-only control/catalyzed caps before the P80 penalty."""
    pct_ox  = float(CONFIG.get("leach_pct_oxides", 1.00))
    pct_sec = float(CONFIG.get("leach_pct_secondary_sulfides", 0.70))
    prior_terms = compute_primary_sulfide_cap_prior_terms(
        cu_oxides_equiv=cu_oxides_equiv,
        cu_secondary_equiv=cu_secondary_equiv,
        cu_primary_equiv=cu_primary_equiv,
        residual_cpy_pct=residual_cpy_pct,
        material_size_p80_in=material_size_p80_in,
        column_height_m=column_height_m,
        lixiviant_initial_fe_mg_l=lixiviant_initial_fe_mg_l,
        lixiviant_initial_ph=lixiviant_initial_ph,
        lixiviant_initial_orp_mv=lixiviant_initial_orp_mv,
        cyanide_soluble_pct=cyanide_soluble_pct,
        acid_soluble_pct=acid_soluble_pct,
        grouped_acid_generating_sulfides_pct=grouped_acid_generating_sulfides_pct,
        grouped_carbonates_pct=grouped_carbonates_pct,
    )
    pct_pri_ctrl = (
        float(primary_control_fraction_override)
        if primary_control_fraction_override is not None and np.isfinite(primary_control_fraction_override)
        else float(prior_terms["primary_control_prior_frac"])
    )
    pct_pri_cat = (
        float(primary_catalyzed_fraction_override)
        if primary_catalyzed_fraction_override is not None and np.isfinite(primary_catalyzed_fraction_override)
        else float(prior_terms["primary_catalyzed_prior_frac"])
    )
    pct_pri_cat = max(pct_pri_ctrl, pct_pri_cat)

    # Sanitise inputs – treat NaN / negative as zero contribution
    ox  = max(0.0, float(cu_oxides_equiv))   if np.isfinite(cu_oxides_equiv)   else 0.0
    sec = max(0.0, float(cu_secondary_equiv)) if np.isfinite(cu_secondary_equiv) else 0.0
    pri = max(0.0, float(cu_primary_equiv))   if np.isfinite(cu_primary_equiv)   else 0.0
    cu  = compute_total_copper_equivalent(
        cu_oxides_equiv=cu_oxides_equiv,
        cu_secondary_equiv=cu_secondary_equiv,
        cu_primary_equiv=cu_primary_equiv,
    )

    if cu <= 1e-9:
        return np.nan, np.nan

    ctrl_leachable_real = ox * pct_ox + sec * pct_sec + pri * pct_pri_ctrl
    cat_leachable_real  = ox * pct_ox + sec * pct_sec + pri * pct_pri_cat

    ctrl_cap = max(0.0, ctrl_leachable_real / cu * 100.0)
    cat_cap  = max(0.0, cat_leachable_real  / cu * 100.0)

    return float(ctrl_cap), float(cat_cap)


def compute_sample_leach_caps(
    cu_oxides_equiv: float,
    cu_secondary_equiv: float,
    cu_primary_equiv: float,
    material_size_p80_in: float = np.nan,
    residual_cpy_pct: float = np.nan,
    column_height_m: float = np.nan,
    lixiviant_initial_fe_mg_l: float = np.nan,
    lixiviant_initial_ph: float = np.nan,
    lixiviant_initial_orp_mv: float = np.nan,
    cyanide_soluble_pct: float = np.nan,
    acid_soluble_pct: float = np.nan,
    grouped_acid_generating_sulfides_pct: float = np.nan,
    grouped_carbonates_pct: float = np.nan,
    primary_control_fraction_override: Optional[float] = None,
    primary_catalyzed_fraction_override: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute per-sample leach caps for control and catalyzed curves based on
    ore mineralogy, then adjust them using a shifted Hill penalty on
    ``material_size_p80_in``.

    v12: primary-sulfide fractions remain metallurgy priors, but are adjusted
    deterministically from initial oxidizing chemistry and geometry instead of
    using a hard global primary ceiling for every sample.

    Returns
    -------
    (ctrl_cap, cat_cap) : both in recovery-% on the native percentage scale
    """
    ctrl_cap_raw, cat_cap_raw = compute_chemistry_only_leach_caps(
        cu_oxides_equiv=cu_oxides_equiv,
        cu_secondary_equiv=cu_secondary_equiv,
        cu_primary_equiv=cu_primary_equiv,
        residual_cpy_pct=residual_cpy_pct,
        material_size_p80_in=material_size_p80_in,
        column_height_m=column_height_m,
        lixiviant_initial_fe_mg_l=lixiviant_initial_fe_mg_l,
        lixiviant_initial_ph=lixiviant_initial_ph,
        lixiviant_initial_orp_mv=lixiviant_initial_orp_mv,
        cyanide_soluble_pct=cyanide_soluble_pct,
        acid_soluble_pct=acid_soluble_pct,
        grouped_acid_generating_sulfides_pct=grouped_acid_generating_sulfides_pct,
        grouped_carbonates_pct=grouped_carbonates_pct,
        primary_control_fraction_override=primary_control_fraction_override,
        primary_catalyzed_fraction_override=primary_catalyzed_fraction_override,
    )
    p80_factor = compute_material_size_p80_cap_penalty(material_size_p80_in)
    ctrl_cap = ctrl_cap_raw * p80_factor
    cat_cap = cat_cap_raw * p80_factor

    return float(ctrl_cap), float(cat_cap)


def double_exp_curve_np(t_days: np.ndarray, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
    t = np.clip(np.asarray(t_days, dtype=float), 0.0, None)
    return float(a1) * (1.0 - np.exp(-float(b1) * t)) + float(a2) * (1.0 - np.exp(-float(b2) * t))


def double_exp_slope_np(t_days: np.ndarray, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
    t = np.clip(np.asarray(t_days, dtype=float), 0.0, None)
    return (
        float(a1) * float(b1) * np.exp(-float(b1) * t)
        + float(a2) * float(b2) * np.exp(-float(b2) * t)
    )


def sigmoid_gate_np(t_days: np.ndarray, mid_day: float, width_day: float) -> np.ndarray:
    t = np.clip(np.asarray(t_days, dtype=float), 0.0, None)
    width = max(float(width_day), 1e-6)
    z = np.clip((t - float(mid_day)) / width, -60.0, 60.0)
    raw = 1.0 / (1.0 + np.exp(-z))
    z0 = np.clip((0.0 - float(mid_day)) / width, -60.0, 60.0)
    raw0 = float(1.0 / (1.0 + np.exp(-z0)))
    denom = max(1.0 - raw0, 1e-9)
    gate = (raw - raw0) / denom
    return np.clip(gate, 0.0, 1.0)


def sigmoid_gate_slope_np(t_days: np.ndarray, mid_day: float, width_day: float) -> np.ndarray:
    t = np.clip(np.asarray(t_days, dtype=float), 0.0, None)
    width = max(float(width_day), 1e-6)
    z = np.clip((t - float(mid_day)) / width, -60.0, 60.0)
    raw = 1.0 / (1.0 + np.exp(-z))
    z0 = np.clip((0.0 - float(mid_day)) / width, -60.0, 60.0)
    raw0 = float(1.0 / (1.0 + np.exp(-z0)))
    denom = max(1.0 - raw0, 1e-9)
    return raw * (1.0 - raw) / width / denom


def gated_double_exp_curve_np(
    t_days: np.ndarray,
    a1: float,
    b1: float,
    a2: float,
    b2: float,
    gate_mid_day: float,
    gate_width_day: float,
) -> np.ndarray:
    """
    Residence-time-delay biexponential.

    The sigmoid gate represents the induction period *before* leaching starts
    (wetting, acid conditioning, ferric build-up).  The kinetic parameters b1/b2
    therefore describe the leach rate AFTER this induction period, not from t=0.

    Implementation: evaluate the biexponential at effective time
        t_eff = max(t - gate_mid_day, 0)
    so that at t = gate_mid_day the kinetics start from zero, and for t >> gate_mid_day
    the curve grows at the same rate as an un-gated biexponential of the same b1/b2.
    The sigmoid multiplier provides the smooth on-ramp over gate_width_day.
    """
    t = np.asarray(t_days, dtype=float)
    t_eff = np.maximum(t - float(gate_mid_day), 0.0)
    base = double_exp_curve_np(t_eff, a1, b1, a2, b2)
    gate = sigmoid_gate_np(t, gate_mid_day, gate_width_day)
    return gate * base


def gated_double_exp_slope_np(
    t_days: np.ndarray,
    a1: float,
    b1: float,
    a2: float,
    b2: float,
    gate_mid_day: float,
    gate_width_day: float,
) -> np.ndarray:
    """
    Slope (d/dt) of the residence-time-delay gated biexponential.

    d/dt[gate(t) * biexp(t - mid)]
        = gate_slope(t) * biexp(t - mid) + gate(t) * biexp_slope(t - mid)

    The biexp slope w.r.t. t equals its slope w.r.t. t_eff when t > gate_mid
    (chain rule: dt_eff/dt = 1 for t > mid, 0 for t < mid).
    """
    t = np.asarray(t_days, dtype=float)
    t_eff = np.maximum(t - float(gate_mid_day), 0.0)
    base = double_exp_curve_np(t_eff, a1, b1, a2, b2)
    # Kinetic slope is non-zero only after the gate onset
    base_slope_raw = double_exp_slope_np(t_eff, a1, b1, a2, b2)
    base_slope = np.where(t > gate_mid_day, base_slope_raw, 0.0)
    gate = sigmoid_gate_np(t, gate_mid_day, gate_width_day)
    gate_slope = sigmoid_gate_slope_np(t, gate_mid_day, gate_width_day)
    return gate_slope * base + gate * base_slope


def prefit_curve_prediction_np(
    t_days: np.ndarray,
    params: np.ndarray,
    curve_mode: str = "biexponential",
    gate_mid_day: float = np.nan,
    gate_width_day: float = np.nan,
) -> np.ndarray:
    p = np.asarray(params, dtype=float)
    if p.size < 4 or not np.all(np.isfinite(p[:4])):
        return np.full(np.asarray(t_days, dtype=float).shape, np.nan, dtype=float)
    mode = str(curve_mode or "biexponential").strip().lower()
    if (
        mode == "sigmoid_gated_biexponential"
        and np.isfinite(gate_mid_day)
        and np.isfinite(gate_width_day)
        and float(gate_width_day) > 0.0
    ):
        return gated_double_exp_curve_np(t_days, p[0], p[1], p[2], p[3], gate_mid_day, gate_width_day)
    return double_exp_curve_np(t_days, p[0], p[1], p[2], p[3])


def prefit_curve_slope_np(
    t_days: np.ndarray,
    params: np.ndarray,
    curve_mode: str = "biexponential",
    gate_mid_day: float = np.nan,
    gate_width_day: float = np.nan,
) -> np.ndarray:
    p = np.asarray(params, dtype=float)
    if p.size < 4 or not np.all(np.isfinite(p[:4])):
        return np.full(np.asarray(t_days, dtype=float).shape, np.nan, dtype=float)
    mode = str(curve_mode or "biexponential").strip().lower()
    if (
        mode == "sigmoid_gated_biexponential"
        and np.isfinite(gate_mid_day)
        and np.isfinite(gate_width_day)
        and float(gate_width_day) > 0.0
    ):
        return gated_double_exp_slope_np(t_days, p[0], p[1], p[2], p[3], gate_mid_day, gate_width_day)
    return double_exp_slope_np(t_days, p[0], p[1], p[2], p[3])


def enforce_fast_slow_pairing(params: np.ndarray) -> np.ndarray:
    p = np.asarray(params, dtype=float).copy()
    if p.size != 4 or not np.all(np.isfinite(p)):
        return p
    if p[3] > p[1]:
        p = np.array([p[2], p[3], p[0], p[1]], dtype=float)
    return p


def resolve_effective_asymptote_cap(cap: Optional[float] = None) -> float:
    if cap is None or not np.isfinite(cap) or float(cap) < 0.0:
        return np.nan
    return float(cap)


def resolve_prefit_min_amplitude() -> float:
    return max(0.0, float(CONFIG.get("prefit_min_amplitude", 1.0)))


def project_amplitudes_to_cap(a1: float, a2: float, max_total: float) -> Tuple[float, float]:
    min_amp = resolve_prefit_min_amplitude()
    a1_val = max(min_amp, float(a1))
    a2_val = max(min_amp, float(a2))
    if not np.isfinite(max_total):
        return a1_val, a2_val
    max_total_val = max(2.0 * min_amp, float(max_total))
    total = a1_val + a2_val
    if total <= max_total_val + 1e-12:
        return a1_val, a2_val
    slack = max_total_val - 2.0 * min_amp
    if slack <= 1e-12:
        return min_amp, min_amp
    extra1 = max(0.0, a1_val - min_amp)
    extra2 = max(0.0, a2_val - min_amp)
    extra_total = extra1 + extra2
    if extra_total <= 1e-12:
        return min_amp, min_amp
    scale = slack / extra_total
    return min_amp + extra1 * scale, min_amp + extra2 * scale


def sanitize_curve_params(
    params: np.ndarray,
    cap: Optional[float] = None,
    enforce_cap: bool = False,
) -> np.ndarray:
    p = np.asarray(params, dtype=float).copy()
    min_amp = resolve_prefit_min_amplitude()
    if p.size != 4 or not np.all(np.isfinite(p)):
        p = np.array([20.0, 0.02, 10.0, 0.003], dtype=float)
    p[0] = max(min_amp, p[0])
    p[2] = max(min_amp, p[2])
    p[1] = np.clip(p[1], 1e-5, 1e-1)
    p[3] = np.clip(p[3], 1e-5, 1e-1)
    p = enforce_fast_slow_pairing(p)
    if enforce_cap:
        effective_cap = resolve_effective_asymptote_cap(cap)
        p[0], p[2] = project_amplitudes_to_cap(p[0], p[2], effective_cap)
    return p


def prepare_virtual_rebased_prefit_arrays(
    time_days: np.ndarray,
    recovery_pct: np.ndarray,
    *,
    cap: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a pre-fit-only view that ignores delayed sub-threshold recovery.

    Some coarse/tall columns have a long near-zero start followed by a normal
    leach rise. The biexponential form has no lag term, so fitting those early
    points directly makes the curve compromise between "flat start" and
    "post-breakthrough kinetics". For the pre-fit only, discard the prefix below
    the configured recovery threshold, rebase the remaining segment to a
    virtual origin, and fit that post-threshold shape.
    """
    t = np.asarray(time_days, dtype=float)
    y = np.asarray(recovery_pct, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]
    if t.size > 0:
        order = np.argsort(t)
        t = t[order]
        y = y[order]

    out: Dict[str, Any] = {
        "time": t,
        "recovery": y,
        "cap": cap,
        "active": False,
        "time_offset_day": 0.0,
        "recovery_offset_pct": 0.0,
        "ignored_initial_point_count": 0,
        "fit_point_count": int(t.size),
        "target_day_override": None,
    }
    if (
        not bool(CONFIG.get("prefit_virtual_rebase_enabled", True))
        or t.size == 0
        or y.size == 0
    ):
        return out

    threshold = float(CONFIG.get("prefit_virtual_rebase_recovery_threshold_pct", 5.0))
    min_points = max(2, int(CONFIG.get("prefit_virtual_rebase_min_points", 6)))
    if not np.isfinite(threshold) or threshold <= 0.0:
        return out

    crossing = np.flatnonzero(y >= threshold)
    if crossing.size == 0:
        return out
    first_idx = int(crossing[0])
    if first_idx <= 0 or (t.size - first_idx) < min_points:
        return out

    t0 = float(t[first_idx])
    if first_idx > 0:
        t_prev = float(t[first_idx - 1])
        y_prev = float(y[first_idx - 1])
        t_curr = float(t[first_idx])
        y_curr = float(y[first_idx])
        if y_curr > y_prev and y_prev < threshold <= y_curr:
            frac = float(np.clip((threshold - y_prev) / max(y_curr - y_prev, 1e-12), 0.0, 1.0))
            t0 = t_prev + frac * (t_curr - t_prev)

    t_fit = np.clip(t[first_idx:] - t0, 0.0, None)
    y_fit = np.clip(y[first_idx:] - threshold, 0.0, 100.0)
    keep = np.isfinite(t_fit) & np.isfinite(y_fit)
    t_fit = t_fit[keep]
    y_fit = y_fit[keep]
    if t_fit.size < min_points:
        return out

    if t_fit[0] > 1e-9 or y_fit[0] > 1e-9:
        t_fit = np.concatenate([[0.0], t_fit])
        y_fit = np.concatenate([[0.0], y_fit])

    cap_fit = cap
    cap_scalar = resolve_effective_asymptote_cap(cap)
    if np.isfinite(cap_scalar):
        cap_fit = max(2.0, cap_scalar - threshold)

    target_day = float(CONFIG.get("prefit_asymptote_target_day", 2500.0))
    target_day_override = None
    if np.isfinite(target_day) and target_day > 0.0:
        target_day_override = max(1.0, target_day - t0)

    out.update(
        {
            "time": t_fit,
            "recovery": y_fit,
            "cap": cap_fit,
            "active": True,
            "time_offset_day": float(t0),
            "recovery_offset_pct": float(threshold),
            "ignored_initial_point_count": int(first_idx),
            "fit_point_count": int(t_fit.size),
            "target_day_override": target_day_override,
        }
    )
    return out


def fit_biexponential_params(
    t_days: np.ndarray,
    recovery: np.ndarray,
    cap: Optional[float] = None,
    target_day_recovery: Optional[float] = None,
    target_day_recovery_weight: Optional[float] = None,
    target_day_recovery_mode: str = "exact",
    target_day_override: Optional[float] = None,
    material_size_p80_in: float = float("nan"),
    target_day_penalty_weight_override: Optional[float] = None,
    rate_upper_boost: float = 1.0,
    cap_as_upper_bound_only: bool = False,
    disable_shortfall_penalty: bool = False,
) -> np.ndarray:
    """
    Improved bi-exponential fitting with:
      - Adaptive initial guesses based on curve behaviour (increasing vs plateau).
      - Soft asymptote target: when ``cap`` is provided, the fitter guides
        ``a1 + a2`` toward the per-sample leach cap from
        ``compute_sample_leach_caps``. Control and catalyzed caps must be passed
        separately by the caller and are never mixed.
      - Soft long-horizon maturity target: by the configured target day, the
        curve should be near its asymptote and have a small terminal slope.
      - Linear weight ramp from 1.0 to 10.0 across data points so that the
        fit progressively prioritises the later trend over early behaviour.
      - Two optimisation strategies: multi-start local + differential evolution.
      - Best candidate selected by the penalised fit score, with R² as a
        tiebreaker.
    """
    t = np.asarray(t_days, dtype=float)
    y = np.asarray(recovery, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() < 6:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
    t = t[valid]

    effective_cap_raw = resolve_effective_asymptote_cap(cap)
    effective_cap_target = max(2.0, float(effective_cap_raw)) if np.isfinite(effective_cap_raw) else np.nan

    y = np.clip(y[valid], 0.0, 100.0)
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    observed_upper = max(1.0, float(np.nanmax(y)))
    min_amp = resolve_prefit_min_amplitude()
    if np.isfinite(effective_cap_target):
        amplitude_upper = max(5.0, 1.5 * max(observed_upper, float(effective_cap_target)))
    else:
        amplitude_upper = max(5.0, 3.0 * observed_upper)
    amplitude_upper = max(amplitude_upper, min_amp + 1e-3)

    # ---- point weights: exponential ramp 1 → 25 ----------------------------
    n_pts = len(t)
    weights = np.geomspace(1.0, 25.0, n_pts)
    # ---- tail-anchor: boost the final N days for all columns ----------------
    # This anchors the extrapolation start to the observed late-test trend
    # without changing the rest of the curve fit.
    _tail_anchor_days = float(CONFIG.get("prefit_tail_anchor_days", 0.0))
    _tail_anchor_boost = float(np.clip(CONFIG.get("prefit_tail_anchor_boost", 1.0), 1.0, 20.0))
    if _tail_anchor_days > 0.0 and _tail_anchor_boost > 1.0 and n_pts > 1:
        _t_max = float(t[-1])
        _tail_mask = t >= (_t_max - _tail_anchor_days)
        weights = weights.copy()
        weights[_tail_mask] *= _tail_anchor_boost
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

    if np.isfinite(effective_cap_target):
        target = float(effective_cap_target)
        p0_list.extend([
            [0.55 * target, 0.03, 0.45 * target, 0.003],
            [0.70 * target, 0.05, 0.30 * target, 0.004],
            [0.40 * target, 0.08, 0.60 * target, 0.002],
        ])

    lower = [min_amp, 1e-5, min_amp, 1e-5]
    upper = [amplitude_upper, 1e-1, amplitude_upper, 1e-1]

    # ---- P80-based kinetic rate upper bound ------------------------------------
    # Use the same compute_material_size_p80_cap_penalty function that limits the
    # metallurgical recovery cap. This ensures the rate constraint is consistent
    # with the cap penalty: coarser ore dissolves both slower AND to a lower ceiling.
    # rate_upper = base_rate * p80_cap_factor * rate_upper_boost (for catalyzed)
    _p80_rate_base = float(CONFIG.get("prefit_p80_rate_base", 0.10))
    if np.isfinite(material_size_p80_in) and material_size_p80_in > 0.0:
        _p80_cap_factor_for_rate = compute_material_size_p80_cap_penalty(material_size_p80_in)
        _p80_rate_upper = float(np.clip(_p80_rate_base * _p80_cap_factor_for_rate, 1e-5, _p80_rate_base))
        _p80_rate_upper *= float(np.clip(rate_upper_boost, 1.0, 3.0))  # catalyzed kinetic boost
        upper[1] = min(upper[1], _p80_rate_upper)
        upper[3] = min(upper[3], _p80_rate_upper)
        lower[1] = min(lower[1], upper[1] - 1e-6)
        lower[3] = min(lower[3], upper[3] - 1e-6)

    # ---- per-sample asymptote target -----------------------------------------
    # When a mineralogy-derived cap is provided, guide a1+a2 toward that cap
    # without projecting the amplitudes onto it. This preserves slow approaches
    # to the target asymptote for coarse-P80 samples while still biasing the
    # fitter toward the chemistry-derived recovery ceiling.
    cap_target_val = float(effective_cap_target) if np.isfinite(effective_cap_target) else None
    cap_target_penalty_weight = float(CONFIG.get("prefit_cap_target_penalty_weight", 1.0))
    if cap_target_val is not None:
        cap_target_soft_margin = max(
            float(CONFIG.get("prefit_cap_target_soft_margin", 2.0)),
            float(CONFIG.get("prefit_cap_target_margin_fraction", 0.05)) * cap_target_val,
        )
    else:
        cap_target_soft_margin = 0.0
    target_day = (
        float(target_day_override)
        if target_day_override is not None and np.isfinite(target_day_override)
        else float(CONFIG.get("prefit_asymptote_target_day", 2500.0))
    )
    target_day_min_frac = float(np.clip(CONFIG.get("prefit_target_day_min_asymptote_frac", 0.95), 0.0, 1.0))
    target_day_max_slope = max(0.0, float(CONFIG.get("prefit_target_day_max_slope_pct_per_day", 0.005)))
    target_day_penalty_weight = max(0.0, float(CONFIG.get("prefit_target_day_penalty_weight", 1.0)))
    if target_day_penalty_weight_override is not None and np.isfinite(float(target_day_penalty_weight_override)):
        target_day_penalty_weight = max(0.0, float(target_day_penalty_weight_override))
    target_recovery = (
        float(target_day_recovery)
        if target_day_recovery is not None and np.isfinite(target_day_recovery)
        else np.nan
    )
    target_recovery_weight = (
        max(0.0, float(target_day_recovery_weight))
        if target_day_recovery_weight is not None and np.isfinite(target_day_recovery_weight)
        else 0.0
    )
    target_recovery_mode_norm = str(target_day_recovery_mode or "exact").strip().lower()
    if target_recovery_mode_norm not in {"exact", "minimum"}:
        target_recovery_mode_norm = "exact"

    def _penalised_wmse(params):
        a1, b1, a2, b2 = sanitize_curve_params(
            np.asarray(params, dtype=float),
            cap=effective_cap_target,
            enforce_cap=False,
        )
        pred = double_exp_curve_np(t, a1, b1, a2, b2)
        residuals = pred - y
        wmse = float(np.mean(weights * residuals ** 2))

        asymptote = a1 + a2

        if cap_target_val is not None:
            if cap_as_upper_bound_only:
                # One-sided: only penalise if asymptote EXCEEDS the cap.
                asymptote_gap = max(0.0, asymptote - cap_target_val - cap_target_soft_margin)
            else:
                # Two-sided: penalise both under- and over-shoot (default).
                asymptote_gap = max(0.0, abs(asymptote - cap_target_val) - cap_target_soft_margin)
            cap_target_penalty = asymptote_gap ** 2
        else:
            cap_target_penalty = 0.0

        if (
            target_day_penalty_weight > 0.0
            and np.isfinite(target_day)
            and target_day > 0.0
            and asymptote > 1e-9
        ):
            pred_target = float(double_exp_curve_np(np.asarray([target_day]), a1, b1, a2, b2)[0])
            slope_target = float(double_exp_slope_np(np.asarray([target_day]), a1, b1, a2, b2)[0])
            # Shortfall: how far the curve is from 95% of its own asymptote at target_day.
            # Disabled for catalyzed — slope constraint alone enforces near-flatness.
            target_shortfall = 0.0 if disable_shortfall_penalty else max(0.0, target_day_min_frac * asymptote - pred_target)
            slope_excess = max(0.0, slope_target - target_day_max_slope)
            target_day_penalty = target_shortfall ** 2 + (365.0 * slope_excess) ** 2
        else:
            pred_target = np.nan
            target_day_penalty = 0.0

        if (
            target_recovery_weight > 0.0
            and np.isfinite(target_recovery)
            and np.isfinite(target_day)
            and target_day > 0.0
        ):
            if not np.isfinite(pred_target):
                pred_target = float(double_exp_curve_np(np.asarray([target_day]), a1, b1, a2, b2)[0])
            if target_recovery_mode_norm == "minimum":
                target_recovery_resid = max(0.0, target_recovery - pred_target)
            else:
                target_recovery_resid = pred_target - target_recovery
            target_recovery_penalty = target_recovery_resid ** 2
        else:
            target_recovery_penalty = 0.0

        return (
            wmse
            + cap_target_penalty_weight * cap_target_penalty
            + target_day_penalty_weight * target_day_penalty
            + target_recovery_weight * target_recovery_penalty
        )

    # ---- Strategy 1: multi-start local optimisation --------------------------
    best = None
    best_score = np.inf
    best_r2 = -np.inf

    for p0 in p0_list:
        p0 = sanitize_curve_params(np.asarray(p0, dtype=float), cap=effective_cap_target, enforce_cap=False)
        try:
            popt, _ = curve_fit(
                lambda t_, a1, b1, a2, b2: double_exp_curve_np(
                    t_,
                    *sanitize_curve_params(
                        np.array([a1, b1, a2, b2], dtype=float),
                        cap=effective_cap_target,
                        enforce_cap=False,
                    ),
                ),
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
            popt = sanitize_curve_params(np.asarray(popt, dtype=float), cap=effective_cap_target, enforce_cap=False)
            pred = double_exp_curve_np(t, popt[0], popt[1], popt[2], popt[3])
            score = _penalised_wmse(popt)
            r2 = float(r2_score(y, pred))
            if score < best_score - 1e-12 or (abs(score - best_score) <= 1e-12 and r2 > best_r2):
                best_score = score
                best_r2 = r2
                best = popt
        except Exception:
            continue

    # ---- Strategy 2: differential evolution with weighted + cap-target penalty ---
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

        popt = sanitize_curve_params(np.asarray(result.x, dtype=float), cap=effective_cap_target, enforce_cap=False)
        pred = double_exp_curve_np(t, popt[0], popt[1], popt[2], popt[3])
        score = _penalised_wmse(popt)
        r2 = float(r2_score(y, pred))
        if score < best_score - 1e-12 or (abs(score - best_score) <= 1e-12 and r2 > best_r2):
            best_score = score
            best_r2 = r2
            best = popt
    except Exception:
        pass

    if best is None:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
    return sanitize_curve_params(best, cap=effective_cap_target, enforce_cap=False)


def compute_fast_leach_gate_onset_prior_days(
    cu_oxides_equiv: float,
    cu_secondary_equiv: float,
    cu_primary_equiv: float,
    column_height_m: float = float("nan"),
    grouped_carbonates_pct: float = float("nan"),
    grouped_acid_generating_sulfides_pct: float = float("nan"),
) -> float:
    """
    Mineralogy-based prior estimate for the sigmoid gate onset day (gate_mid_day).

    Chemistry rationale
    -------------------
    Copper oxides (weight 0.70) and secondary sulfides (weight 0.30) are
    "fast-leaching" species that dissolve readily in dilute acid from the first
    days of leaching.  When these dominate the ore, recovery rises quickly from
    day 1 — little or no sigmoid gate is needed.

    Conversely, when the Cu pool is dominated by chalcopyrite (primary sulfide),
    almost no copper leaches until:
      (a) the leach solution has built up sufficient ferric iron,
      (b) the column is fully wetted and acidified, and
      (c) the oxidative environment (ORP) is high enough to attack chalcopyrite.

    This induction period can span hundreds of days for coarse, high-primary-Cu
    columns.  The onset is further delayed by:
      - Column height: taller columns require longer for the lixiviant to
        percolate, wick into the ore matrix, and condition the solution.
      - Acid-consuming gangue (carbonates): they neutralise the lixiviant,
        delaying when effective leaching conditions are established.
      - Acid-generating sulfides: net acid producers that can actually SHORTEN
        the onset (more acid available for leaching); effect is weighted at half
        the carbonates effect and subtracted.

    Returns
    -------
    float
        Estimated gate onset in days.  Returns NaN if insufficient mineralogy
        data is available.  The caller may use this as an additional initial
        guess in the optimizer — it is not a hard constraint.
    """
    # ---------- fast-Cu fraction -----------------------------------------------
    _ox = max(0.0, float(cu_oxides_equiv) if np.isfinite(cu_oxides_equiv) else 0.0)
    _sec = max(0.0, float(cu_secondary_equiv) if np.isfinite(cu_secondary_equiv) else 0.0)
    _pri = max(0.0, float(cu_primary_equiv) if np.isfinite(cu_primary_equiv) else 0.0)
    _total_cu = _ox + _sec + _pri
    if _total_cu < 1e-6:
        return float("nan")  # no mineralogy data available — skip

    _fast_cu_weighted = 0.70 * _ox + 0.30 * _sec
    _fast_cu_frac = float(np.clip(_fast_cu_weighted / _total_cu, 0.0, 1.0))
    _slow_cu_frac = 1.0 - _fast_cu_frac  # [0, 1]: fraction that needs induction

    # ---------- base onset from slow-Cu fraction --------------------------------
    # Scale: 0 slow-Cu → ~0 day onset; 100% slow-Cu → base_max_onset days
    _base_max_onset = float(CONFIG.get("gate_onset_base_max_days", 600.0))
    _onset_exponent = float(CONFIG.get("gate_onset_slow_cu_exponent", 1.5))
    _gate_mid_base = _base_max_onset * (_slow_cu_frac ** _onset_exponent)

    # ---------- column-height multiplier ----------------------------------------
    _ref_height = float(CONFIG.get("gate_onset_height_ref_m", 3.0))  # reference column height
    _height_alpha = float(CONFIG.get("gate_onset_height_alpha", 0.40))
    if np.isfinite(column_height_m) and column_height_m > 0.0:
        _h = float(column_height_m)
        _height_factor = float((_h / max(_ref_height, 1e-3)) ** _height_alpha)
    else:
        _height_factor = 1.0

    # ---------- acid-consumer penalty -------------------------------------------
    _carbonates = max(0.0, float(grouped_carbonates_pct) if np.isfinite(grouped_carbonates_pct) else 0.0)
    _acid_gen = max(0.0, float(grouped_acid_generating_sulfides_pct) if np.isfinite(grouped_acid_generating_sulfides_pct) else 0.0)
    # Net acid consumer score: carbonates slow onset; acid-gen sulfides speed it (×0.5 weight)
    _acid_consumer_net = float(np.clip((_carbonates - 0.5 * _acid_gen) / 10.0, 0.0, 1.0))
    _acid_strength = float(CONFIG.get("gate_onset_acid_consumer_strength", 0.50))
    _acid_factor = 1.0 + _acid_strength * _acid_consumer_net

    gate_mid_prior = _gate_mid_base * _height_factor * _acid_factor
    return float(np.clip(gate_mid_prior, 0.0, _base_max_onset * 1.5))


def should_use_sigmoid_gated_prefit(
    *,
    time_days: np.ndarray,
    recovery_pct: np.ndarray,
    material_size_p80_in: float,
    column_height_m: float,
) -> bool:
    if not bool(CONFIG.get("prefit_sigmoid_gate_enabled", True)):
        return False
    p80_min = float(CONFIG.get("prefit_sigmoid_gate_min_p80_in", 3.0))
    height_min = float(CONFIG.get("prefit_sigmoid_gate_min_column_height_m", 3.0))
    if p80_min > 0.0 and not (np.isfinite(material_size_p80_in) and float(material_size_p80_in) >= p80_min):
        return False
    if height_min > 0.0 and not (np.isfinite(column_height_m) and float(column_height_m) > height_min):
        return False

    t = np.asarray(time_days, dtype=float)
    y = np.asarray(recovery_pct, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]
    if t.size < 6:
        return False
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    threshold = float(CONFIG.get("prefit_sigmoid_gate_trigger_recovery_pct", 5.0))
    min_after = max(2, int(CONFIG.get("prefit_sigmoid_gate_min_points_after_trigger", 6)))
    min_initial = max(1, int(CONFIG.get("prefit_sigmoid_gate_min_initial_points", 3)))
    low_recovery_pct = float(CONFIG.get("prefit_sigmoid_gate_low_recovery_pct", 1.0))
    crossing = np.flatnonzero(y >= threshold)
    if crossing.size == 0:
        return False
    first_idx = int(crossing[0])
    if first_idx < min_initial or (y.size - first_idx) < min_after:
        return False
    initial = y[:first_idx]
    if initial.size == 0:
        return False
    low_prefix_frac = float(np.mean(initial <= low_recovery_pct))
    low_prefix_frac_min = float(CONFIG.get("prefit_sigmoid_gate_min_low_prefix_frac", 0.50))
    total_rise = float(np.nanmax(y) - np.nanmin(y))
    early_rise = float(y[first_idx] - np.nanmin(initial))
    return bool(
        low_prefix_frac >= low_prefix_frac_min
        and total_rise >= max(threshold, 1.0)
        and early_rise >= max(0.25 * threshold, 0.5)
    )


def fit_sigmoid_gated_biexponential_params(
    t_days: np.ndarray,
    recovery: np.ndarray,
    cap: Optional[float] = None,
    target_day_recovery: Optional[float] = None,
    target_day_recovery_weight: Optional[float] = None,
    target_day_recovery_mode: str = "exact",
    material_size_p80_in: float = float("nan"),
    cu_oxides_equiv: float = float("nan"),
    cu_secondary_equiv: float = float("nan"),
    cu_primary_equiv: float = float("nan"),
    column_height_m: float = float("nan"),
    grouped_carbonates_pct: float = float("nan"),
    grouped_acid_generating_sulfides_pct: float = float("nan"),
    target_day_penalty_weight_override: Optional[float] = None,
    rate_upper_boost: float = 1.0,
    cap_as_upper_bound_only: bool = False,
    disable_shortfall_penalty: bool = False,
) -> Tuple[np.ndarray, float, float]:
    t = np.asarray(t_days, dtype=float)
    y = np.asarray(recovery, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() < 6:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float), np.nan, np.nan
    t = t[valid]
    y = np.clip(y[valid], 0.0, 100.0)
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    effective_cap_raw = resolve_effective_asymptote_cap(cap)
    effective_cap_target = max(2.0, float(effective_cap_raw)) if np.isfinite(effective_cap_raw) else np.nan
    observed_upper = max(1.0, float(np.nanmax(y)))
    min_amp = resolve_prefit_min_amplitude()
    if np.isfinite(effective_cap_target):
        amplitude_upper = max(5.0, 1.5 * max(observed_upper, float(effective_cap_target)))
    else:
        amplitude_upper = max(5.0, 3.0 * observed_upper)
    amplitude_upper = max(amplitude_upper, min_amp + 1e-3)

    n_pts = len(t)
    weights = np.geomspace(1.0, 25.0, n_pts)
    # ---- tail-anchor: boost the final N days for all columns ----------------
    _tail_anchor_days = float(CONFIG.get("prefit_tail_anchor_days", 0.0))
    _tail_anchor_boost = float(np.clip(CONFIG.get("prefit_tail_anchor_boost", 1.0), 1.0, 20.0))
    if _tail_anchor_days > 0.0 and _tail_anchor_boost > 1.0 and n_pts > 1:
        _t_max = float(t[-1])
        _tail_mask = t >= (_t_max - _tail_anchor_days)
        weights = weights.copy()
        weights[_tail_mask] *= _tail_anchor_boost
    sigma = 1.0 / np.sqrt(weights)

    threshold = float(CONFIG.get("prefit_sigmoid_gate_trigger_recovery_pct", 5.0))
    crossing = np.flatnonzero(y >= threshold)
    if crossing.size > 0:
        first_idx = int(crossing[0])
        gate_mid_guess_data = float(t[first_idx])
    else:
        first_idx = max(1, n_pts // 3)
        gate_mid_guess_data = float(t[first_idx])

    # Mineralogy-based prior for gate onset: slow-Cu (primary sulfide) columns
    # have a longer induction period before leaching accelerates.
    gate_mid_guess_mineralogy = compute_fast_leach_gate_onset_prior_days(
        cu_oxides_equiv=cu_oxides_equiv,
        cu_secondary_equiv=cu_secondary_equiv,
        cu_primary_equiv=cu_primary_equiv,
        column_height_m=column_height_m,
        grouped_carbonates_pct=grouped_carbonates_pct,
        grouped_acid_generating_sulfides_pct=grouped_acid_generating_sulfides_pct,
    )

    # Primary gate_mid guess from the data; mineralogy prior used as an
    # additional candidate if it differs meaningfully from the data-driven guess.
    gate_mid_guess = gate_mid_guess_data
    time_span = max(1.0, float(t[-1] - t[0]))
    width_min = max(1e-3, float(CONFIG.get("prefit_sigmoid_gate_min_width_days", 7.0)))
    width_max = max(width_min + 1e-3, float(CONFIG.get("prefit_sigmoid_gate_max_width_days", 220.0)))
    # Hard cap: the gate models initial column startup delay (acid percolation),
    # NOT mid-curve kinetic phase changes — those belong to a2/b2 in the biexponential.
    _gate_mid_abs_max = float(CONFIG.get("prefit_sigmoid_gate_max_mid_days", 120.0))
    gate_mid_upper = min(max(gate_mid_guess + 1.0, gate_mid_guess * 1.5, 30.0), _gate_mid_abs_max)
    # Allow mineralogy prior to guide the upper bound, but never exceed the hard cap
    if np.isfinite(gate_mid_guess_mineralogy) and gate_mid_guess_mineralogy > gate_mid_upper:
        gate_mid_upper = min(gate_mid_guess_mineralogy * 1.1, _gate_mid_abs_max)

    y_max = float(np.nanmax(y))
    p0_list = []
    base_guess = fit_biexponential_params(t, y, cap=cap)
    if np.all(np.isfinite(base_guess)):
        for width in [0.08 * time_span, 0.14 * time_span, 0.22 * time_span]:
            p0_list.append([*base_guess[:4], gate_mid_guess, float(np.clip(width, width_min, width_max))])
    if np.isfinite(effective_cap_target):
        target = float(effective_cap_target)
        p0_list.extend([
            [0.55 * target, 0.030, 0.45 * target, 0.003, gate_mid_guess, min(max(0.12 * time_span, width_min), width_max)],
            [0.70 * target, 0.045, 0.30 * target, 0.004, gate_mid_guess, min(max(0.08 * time_span, width_min), width_max)],
            [0.40 * target, 0.080, 0.60 * target, 0.002, gate_mid_guess, min(max(0.18 * time_span, width_min), width_max)],
        ])
    p0_list.extend([
        [0.55 * y_max, 0.035, 0.75 * y_max, 0.003, gate_mid_guess, min(max(0.12 * time_span, width_min), width_max)],
        [0.80 * y_max, 0.060, 0.45 * y_max, 0.006, gate_mid_guess, min(max(0.08 * time_span, width_min), width_max)],
    ])
    # Mineralogy-informed initial guesses: if the prior is finite and meaningfully
    # different from the data-driven guess, add it as a candidate so the optimizer
    # can explore the chemistry-predicted onset region.
    if np.isfinite(gate_mid_guess_mineralogy) and gate_mid_guess_mineralogy > 1.0:
        _mg = float(np.clip(gate_mid_guess_mineralogy, 0.0, gate_mid_upper))
        if np.all(np.isfinite(base_guess)):
            for width in [0.10 * time_span, 0.20 * time_span]:
                p0_list.append([*base_guess[:4], _mg, float(np.clip(width, width_min, width_max))])
        if np.isfinite(effective_cap_target):
            target = float(effective_cap_target)
            p0_list.extend([
                [0.60 * target, 0.025, 0.40 * target, 0.002, _mg, min(max(0.15 * time_span, width_min), width_max)],
                [0.50 * target, 0.040, 0.50 * target, 0.003, _mg, min(max(0.25 * time_span, width_min), width_max)],
            ])

    lower = [min_amp, 1e-5, min_amp, 1e-5, 0.0, width_min]
    upper = [amplitude_upper, 1e-1, amplitude_upper, 1e-1, gate_mid_upper, width_max]

    # P80-based kinetic rate upper bound — consistent with fit_biexponential_params.
    # Uses compute_material_size_p80_cap_penalty so the rate constraint shares the
    # same shifted-Hill P80 penalty as the metallurgical cap.
    _p80_rate_base = float(CONFIG.get("prefit_p80_rate_base", 0.10))
    if np.isfinite(material_size_p80_in) and material_size_p80_in > 0.0:
        _p80_cap_factor_for_rate = compute_material_size_p80_cap_penalty(material_size_p80_in)
        _p80_rate_upper = float(np.clip(_p80_rate_base * _p80_cap_factor_for_rate, 1e-5, _p80_rate_base))
        _p80_rate_upper *= float(np.clip(rate_upper_boost, 1.0, 3.0))  # catalyzed kinetic boost
        upper[1] = min(upper[1], _p80_rate_upper)
        upper[3] = min(upper[3], _p80_rate_upper)
        lower[1] = min(lower[1], upper[1] - 1e-6)
        lower[3] = min(lower[3], upper[3] - 1e-6)

    cap_target_val = float(effective_cap_target) if np.isfinite(effective_cap_target) else None
    cap_target_penalty_weight = float(CONFIG.get("prefit_cap_target_penalty_weight", 1.0))
    if cap_target_val is not None:
        cap_target_soft_margin = max(
            float(CONFIG.get("prefit_cap_target_soft_margin", 2.0)),
            float(CONFIG.get("prefit_cap_target_margin_fraction", 0.05)) * cap_target_val,
        )
    else:
        cap_target_soft_margin = 0.0
    target_day = float(CONFIG.get("prefit_asymptote_target_day", 2500.0))
    target_day_min_frac = float(np.clip(CONFIG.get("prefit_target_day_min_asymptote_frac", 0.95), 0.0, 1.0))
    target_day_max_slope = max(0.0, float(CONFIG.get("prefit_target_day_max_slope_pct_per_day", 0.005)))
    target_day_penalty_weight = max(0.0, float(CONFIG.get("prefit_target_day_penalty_weight", 1.0)))
    if target_day_penalty_weight_override is not None and np.isfinite(float(target_day_penalty_weight_override)):
        target_day_penalty_weight = max(0.0, float(target_day_penalty_weight_override))
    target_recovery = (
        float(target_day_recovery)
        if target_day_recovery is not None and np.isfinite(target_day_recovery)
        else np.nan
    )
    target_recovery_weight = (
        max(0.0, float(target_day_recovery_weight))
        if target_day_recovery_weight is not None and np.isfinite(target_day_recovery_weight)
        else 0.0
    )
    target_recovery_mode_norm = str(target_day_recovery_mode or "exact").strip().lower()
    if target_recovery_mode_norm not in {"exact", "minimum"}:
        target_recovery_mode_norm = "exact"

    def _clean_params(params: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        q = np.asarray(params, dtype=float).copy()
        p4 = sanitize_curve_params(q[:4], cap=effective_cap_target, enforce_cap=False)
        gate_mid = float(np.clip(q[4], lower[4], upper[4]))
        gate_width = float(np.clip(q[5], lower[5], upper[5]))
        return float(p4[0]), float(p4[1]), float(p4[2]), float(p4[3]), gate_mid, gate_width

    def _penalised_wmse(params):
        a1, b1, a2, b2, gate_mid, gate_width = _clean_params(np.asarray(params, dtype=float))
        pred = gated_double_exp_curve_np(t, a1, b1, a2, b2, gate_mid, gate_width)
        residuals = pred - y
        wmse = float(np.mean(weights * residuals ** 2))
        asymptote = a1 + a2

        if cap_target_val is not None:
            if cap_as_upper_bound_only:
                asymptote_gap = max(0.0, asymptote - cap_target_val - cap_target_soft_margin)
            else:
                asymptote_gap = max(0.0, abs(asymptote - cap_target_val) - cap_target_soft_margin)
            cap_target_penalty = asymptote_gap ** 2
        else:
            cap_target_penalty = 0.0

        if (
            target_day_penalty_weight > 0.0
            and np.isfinite(target_day)
            and target_day > 0.0
            and asymptote > 1e-9
        ):
            pred_target = float(gated_double_exp_curve_np(np.asarray([target_day]), a1, b1, a2, b2, gate_mid, gate_width)[0])
            slope_target = float(gated_double_exp_slope_np(np.asarray([target_day]), a1, b1, a2, b2, gate_mid, gate_width)[0])
            target_shortfall = 0.0 if disable_shortfall_penalty else max(0.0, target_day_min_frac * asymptote - pred_target)
            slope_excess = max(0.0, slope_target - target_day_max_slope)
            target_day_penalty = target_shortfall ** 2 + (365.0 * slope_excess) ** 2
        else:
            pred_target = np.nan
            target_day_penalty = 0.0

        if (
            target_recovery_weight > 0.0
            and np.isfinite(target_recovery)
            and np.isfinite(target_day)
            and target_day > 0.0
        ):
            if not np.isfinite(pred_target):
                pred_target = float(gated_double_exp_curve_np(np.asarray([target_day]), a1, b1, a2, b2, gate_mid, gate_width)[0])
            if target_recovery_mode_norm == "minimum":
                target_recovery_resid = max(0.0, target_recovery - pred_target)
            else:
                target_recovery_resid = pred_target - target_recovery
            target_recovery_penalty = target_recovery_resid ** 2
        else:
            target_recovery_penalty = 0.0

        return (
            wmse
            + cap_target_penalty_weight * cap_target_penalty
            + target_day_penalty_weight * target_day_penalty
            + target_recovery_weight * target_recovery_penalty
        )

    best = None
    best_score = np.inf
    best_r2 = -np.inf
    for p0 in p0_list:
        p0 = np.asarray(_clean_params(np.asarray(p0, dtype=float)), dtype=float)
        try:
            popt, _ = curve_fit(
                lambda t_, a1, b1, a2, b2, gate_mid, gate_width: gated_double_exp_curve_np(
                    t_,
                    *_clean_params(np.array([a1, b1, a2, b2, gate_mid, gate_width], dtype=float)),
                ),
                t,
                y,
                p0=p0,
                sigma=sigma,
                absolute_sigma=False,
                bounds=(lower, upper),
                maxfev=16000,
                ftol=1e-12,
                xtol=1e-12,
            )
            cand = np.asarray(_clean_params(np.asarray(popt, dtype=float)), dtype=float)
            pred = gated_double_exp_curve_np(t, cand[0], cand[1], cand[2], cand[3], cand[4], cand[5])
            score = _penalised_wmse(cand)
            r2 = float(r2_score(y, pred))
            if score < best_score - 1e-12 or (abs(score - best_score) <= 1e-12 and r2 > best_r2):
                best_score = score
                best_r2 = r2
                best = cand
        except Exception:
            continue

    try:
        bounds_de = [(lower[i], upper[i]) for i in range(6)]
        result = differential_evolution(
            _penalised_wmse,
            bounds_de,
            seed=42,
            maxiter=500,
            atol=1e-12,
            tol=1e-12,
            workers=1,
            polish=True,
        )
        cand = np.asarray(_clean_params(np.asarray(result.x, dtype=float)), dtype=float)
        pred = gated_double_exp_curve_np(t, cand[0], cand[1], cand[2], cand[3], cand[4], cand[5])
        score = _penalised_wmse(cand)
        r2 = float(r2_score(y, pred))
        if score < best_score - 1e-12 or (abs(score - best_score) <= 1e-12 and r2 > best_r2):
            best = cand
    except Exception:
        pass

    if best is None:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float), np.nan, np.nan
    params = sanitize_curve_params(np.asarray(best[:4], dtype=float), cap=effective_cap_target, enforce_cap=False)
    return params, float(best[4]), float(best[5])


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
        hi = max(lo + 1e-3, hi)
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


def resolve_model_param_bounds(
    lb: np.ndarray,
    ub: np.ndarray,
    use_prefit_param_bounds: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if use_prefit_param_bounds:
        return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)

    amp_upper = max(
        1.0 + 1e-3,
        float(CONFIG.get("param_bounds_disabled_amplitude_upper", 100.0)),
    )
    rate_upper = float(
        np.clip(
            CONFIG.get("param_bounds_disabled_rate_upper", 1e-1),
            1e-5 + 1e-5,
            1e-1,
        )
    )
    physical_lb = np.array([1.0, 1e-5, 1.0, 1e-5], dtype=float)
    physical_ub = np.array([amp_upper, rate_upper, amp_upper, rate_upper], dtype=float)
    return physical_lb, physical_ub

# -----------------------------------
# Ore calculation
def compute_remaining_ore_factor_chemistry_based(
    y_ctrl: torch.Tensor,
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
        pct_pri = float(CONFIG.get("leach_pct_primary_sulfides_catalyzed", 0.75))
    else:
        pct_pri = float(CONFIG.get("leach_pct_primary_sulfides_control", 0.33))

    total_copper_equivalent = torch.clamp(
        torch.nan_to_num(copper_oxides_equiv, nan=0.0, posinf=0.0, neginf=0.0),
        min=0.0,
    ) + torch.clamp(
        torch.nan_to_num(copper_primary_sulfides_equiv, nan=0.0, posinf=0.0, neginf=0.0),
        min=0.0,
    )
    if copper_secondary_sulfides_equiv is not None:
        total_copper_equivalent = total_copper_equivalent + torch.clamp(
            torch.nan_to_num(copper_secondary_sulfides_equiv, nan=0.0, posinf=0.0, neginf=0.0),
            min=0.0,
        )
    cu_safe = torch.clamp(total_copper_equivalent, min=1e-6)

    # leachable_real (in same units as the total copper equivalent)
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


def prefit_metric_mask(time_days: np.ndarray, fit_start_day: float = np.nan) -> np.ndarray:
    t = np.asarray(time_days, dtype=float)
    mask = np.isfinite(t)
    if np.isfinite(fit_start_day):
        mask = mask & (t >= float(fit_start_day) - 1e-9)
    return mask


def generate_virtual_points(
    time: np.ndarray,
    recovery: np.ndarray,
    fit_params: np.ndarray,
    target_day: float,
    interval_days: float = 7.0,
    cap: Optional[float] = None,
    curve_mode: str = "biexponential",
    gate_mid_day: float = np.nan,
    gate_width_day: float = np.nan,
    fit_start_day: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an augmented recovery curve extending observed data to ``target_day``.

    The observed portion is returned **as-is** (actual measured values).  Virtual
    future points are generated starting from ``last_time + interval_days`` and
    anchored via delta-continuation:

        virtual_recovery[i] = last_actual_recovery + (prefit(t[i]) - prefit(last_time))

    This guarantees a smooth, seamless transition at the boundary: the synthetic
    tail begins exactly at the last observed recovery level and follows the
    slope / curvature of the pre-fit equation from that point onward.  As a
    physical constraint, virtual recovery values are clipped to be
    non-decreasing relative to the last observed value (leaching is
    monotonically non-decreasing in recovery).

    ``cap`` is retained for API compatibility but is not applied here.

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
    order = np.argsort(time_valid)
    time_valid = time_valid[order]
    recovery_valid = recovery_valid[order]

    # Last actual observed data point
    last_time = float(time_valid[-1])
    last_actual_recovery = float(recovery_valid[-1])

    # No future extrapolation needed — return actual observed data unchanged
    if last_time >= target_day:
        return np.asarray(time_valid, dtype=float), np.asarray(recovery_valid, dtype=float)

    virtual_times = np.arange(last_time + interval_days, target_day + interval_days, interval_days)

    if len(fit_params) >= 4 and np.all(np.isfinite(fit_params[:4])):
        # Evaluate prefit at anchor point and at all virtual times
        prefit_at_last = float(
            prefit_curve_prediction_np(
                np.array([last_time]),
                fit_params,
                curve_mode,
                gate_mid_day,
                gate_width_day,
            )[0]
        )
        prefit_at_virtual = np.asarray(
            prefit_curve_prediction_np(
                virtual_times,
                fit_params,
                curve_mode,
                gate_mid_day,
                gate_width_day,
            ),
            dtype=float,
        )
        # Delta-continuation: shift the prefit slope to start from the last actual value
        delta = prefit_at_virtual - prefit_at_last
        virtual_recovery = last_actual_recovery + delta
        # Physical constraint: recovery is non-decreasing
        virtual_recovery = np.maximum(virtual_recovery, last_actual_recovery)
    else:
        # Fallback: flat continuation from last observed value
        virtual_recovery = np.full_like(virtual_times, last_actual_recovery, dtype=float)

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


def build_sample_plot_subtitle(sample_id: str, col_ids: List[str], width: int = 110) -> str:
    unique_ids = [str(col_id).strip() for col_id in col_ids if str(col_id).strip()]
    if not unique_ids:
        return ""
    subtitle = f"project_col_id: {', '.join(sorted(dict.fromkeys(unique_ids)))}"
    return textwrap.fill(subtitle, width=max(40, int(width)))


def apply_sample_plot_titles(fig: Any, ax: Any, sample_id: str, col_ids: List[str]) -> None:
    fig.suptitle(str(sample_id), fontsize=14, fontweight="bold")
    subtitle = build_sample_plot_subtitle(sample_id=sample_id, col_ids=col_ids)
    if subtitle:
        ax.set_title(subtitle, fontsize=9, fontweight="normal")


def format_average_catalyst_dose_label(avg_dose_mg_l: float) -> str:
    if np.isfinite(float(avg_dose_mg_l)):
        return f"Average Catalyst dose: {float(avg_dose_mg_l):.2f} mg/L"
    return "Average Catalyst dose: n/a"


def plot_fitted_curve_per_sample(
    pairs: List['PairSample'],
    output_dir: str,
    dpi: int = 300,
    target_day: float = 2500.0,
) -> None:
    """
    Create fitted curve plots for each project_sample_id.
    All unique control and catalyzed project_col_id curves for the sample are
    shown on the same plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    sample_curves: Dict[str, Dict[Tuple[str, int, str], CurveData]] = {}
    for pair in pairs:
        curve_map = sample_curves.setdefault(pair.sample_id, {})
        for curve in [pair.control, pair.catalyzed]:
            curve_key = (str(curve.status), int(curve.row_index), str(curve.col_id))
            curve_map[curve_key] = curve

    control_colors = plt.cm.Blues(np.linspace(0.45, 0.90, 12))
    catalyzed_colors = plt.cm.Oranges(np.linspace(0.45, 0.90, 12))

    for sample_id, curve_map in sorted(sample_curves.items(), key=lambda item: item[0]):
        fig, ax = plt.subplots(figsize=(12, 7), dpi=dpi)
        control_curves = sorted(
            [curve for curve in curve_map.values() if str(curve.status) == "Control"],
            key=lambda curve: (curve.col_id, int(curve.row_index)),
        )
        catalyzed_curves = sorted(
            [curve for curve in curve_map.values() if str(curve.status) == "Catalyzed"],
            key=lambda curve: (curve.col_id, int(curve.row_index)),
        )

        all_curves = (
            [(curve, "Control", control_colors[i % len(control_colors)]) for i, curve in enumerate(control_curves)]
            + [(curve, "Catalyzed", catalyzed_colors[i % len(catalyzed_colors)]) for i, curve in enumerate(catalyzed_curves)]
        )

        criteria_met = True

        for curve_data, status_label, color in all_curves:
            time = curve_data.time
            recovery = curve_data.recovery
            fit_params = curve_data.fit_params
            col_id_label = curve_data.col_id or f"row_{int(curve_data.row_index)}"

            # Calculate metrics
            if len(fit_params) >= 4 and np.all(np.isfinite(fit_params[:4])):
                predicted = prefit_curve_prediction_np(
                    time,
                    fit_params,
                    curve_data.fit_curve_mode,
                    curve_data.fit_gate_mid_day,
                    curve_data.fit_gate_width_day,
                )
                metric_mask = prefit_metric_mask(time, curve_data.prefit_fit_start_day)
                r2, rmse, bias = calculate_fit_metrics(recovery[metric_mask], predicted[metric_mask])
            else:
                r2, rmse, bias = np.nan, np.nan, np.nan

            # Check if criteria are met
            if not (np.isfinite(r2) and r2 > 0.50 and np.isfinite(rmse) and rmse < 5.0):
                criteria_met = False

            # Plot actual data points
            ax.scatter(time, recovery, color=color, s=45, alpha=0.50, zorder=3)

            start_day = float(curve_data.catalyst_start_day)
            if np.isfinite(start_day):
                ax.axvline(start_day, color=color, lw=1.0, ls="--", alpha=0.50, zorder=1)

            # Plot fitted curve
            if len(fit_params) >= 4 and np.all(np.isfinite(fit_params[:4])):
                # Generate smooth curve for visualization
                t_left = (
                    float(curve_data.prefit_fit_start_day)
                    if np.isfinite(curve_data.prefit_fit_start_day)
                    else 0.0
                )
                t_smooth = np.linspace(t_left, max(target_day, np.max(time) * 1.1), 500)
                y_smooth = prefit_curve_prediction_np(
                    t_smooth,
                    fit_params,
                    curve_data.fit_curve_mode,
                    curve_data.fit_gate_mid_day,
                    curve_data.fit_gate_width_day,
                )
                dose_text = ""
                if status_label == "Catalyzed":
                    dose_text = f" | {format_average_catalyst_dose_label(curve_data.avg_catalyst_dose_mg_l)}"
                label = f"{status_label} | {col_id_label}{dose_text} | R²={r2:.3f}, RMSE={rmse:.2f}, Bias={bias:.2f}"
                ax.plot(t_smooth, y_smooth, color=color, linewidth=2.2, alpha=0.90, label=label, zorder=2)
            else:
                label = f"{status_label} | {col_id_label} | no valid fit"
                ax.plot([], [], color=color, linewidth=2.2, alpha=0.90, label=label)

        # Format plot
        ax.set_xlabel('Leach Duration (days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cu Recovery (%)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)

        title_color = 'red' if not criteria_met else 'black'
        apply_sample_plot_titles(
            fig,
            ax,
            sample_id=sample_id,
            col_ids=[curve.col_id for curve in control_curves + catalyzed_curves],
        )
        if fig._suptitle is not None:
            fig._suptitle.set_color(title_color)

        # Set x-axis limit
        ax.set_xlim(left=0, right=target_day * 1.05)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

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
    Augment a PairSample from the stored per-column pre-fit curves if criteria
    are met. Each curve is extended only when its own fit meets R² > 0.5 and
    RMSE < 5.0.
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
    column_height_m = (
        float(pair.input_only_raw[input_only_idx["column_height_m"]])
        if "column_height_m" in input_only_idx
        else np.nan
    )

    # Check criteria for Control
    ctrl_time = pair.control.time
    ctrl_recovery = pair.control.recovery
    ctrl_params = pair.control.fit_params
    
    if len(ctrl_params) >= 4 and np.all(np.isfinite(ctrl_params[:4])):
        ctrl_pred = prefit_curve_prediction_np(
            ctrl_time,
            ctrl_params,
            pair.control.fit_curve_mode,
            pair.control.fit_gate_mid_day,
            pair.control.fit_gate_width_day,
        )
        ctrl_metric_mask = prefit_metric_mask(ctrl_time, pair.control.prefit_fit_start_day)
        ctrl_r2, ctrl_rmse, _ = calculate_fit_metrics(ctrl_recovery[ctrl_metric_mask], ctrl_pred[ctrl_metric_mask])
    else:
        ctrl_r2, ctrl_rmse = np.nan, np.nan
    
    # Check criteria for Catalyzed
    cat_time = pair.catalyzed.time
    cat_recovery = pair.catalyzed.recovery
    cat_params = pair.catalyzed.fit_params
    
    if len(cat_params) >= 4 and np.all(np.isfinite(cat_params[:4])):
        cat_pred = prefit_curve_prediction_np(
            cat_time,
            cat_params,
            pair.catalyzed.fit_curve_mode,
            pair.catalyzed.fit_gate_mid_day,
            pair.catalyzed.fit_gate_width_day,
        )
        cat_metric_mask = prefit_metric_mask(cat_time, pair.catalyzed.prefit_fit_start_day)
        cat_r2, cat_rmse, _ = calculate_fit_metrics(cat_recovery[cat_metric_mask], cat_pred[cat_metric_mask])
    else:
        cat_r2, cat_rmse = np.nan, np.nan
    
    # Evaluate the extension criteria independently for each curve
    ctrl_meets_criteria = np.isfinite(ctrl_r2) and ctrl_r2 > 0.5 and np.isfinite(ctrl_rmse) and ctrl_rmse < 5.0
    cat_meets_criteria = np.isfinite(cat_r2) and cat_r2 > 0.5 and np.isfinite(cat_rmse) and cat_rmse < 5.0
    
    # Create new pair with augmented data
    new_pair = PairSample(
        pair_id=pair.pair_id,
        sample_id=pair.sample_id,
        project_name=pair.project_name,
        static_raw=pair.static_raw,
        input_only_raw=pair.input_only_raw,
        control=pair.control,
        catalyzed=pair.catalyzed,
        control_static_raw=pair.control_static_raw,
        catalyzed_static_raw=pair.catalyzed_static_raw,
        static_scaled=pair.static_scaled,
        static_imputed=pair.static_imputed,
        ctrl_cap=pair.ctrl_cap,
        cat_cap=pair.cat_cap,
        control_cap_anchor_pct=pair.control_cap_anchor_pct,
        catalyzed_cap_anchor_pct=pair.catalyzed_cap_anchor_pct,
        control_cap_anchor_active=pair.control_cap_anchor_active,
        catalyzed_cap_anchor_active=pair.catalyzed_cap_anchor_active,
        orp_aux_target_raw=pair.orp_aux_target_raw,
        orp_aux_target_norm=pair.orp_aux_target_norm,
        orp_aux_mask=pair.orp_aux_mask,
        pls_orp_aux_target_raw=pair.pls_orp_aux_target_raw,
        pls_orp_aux_target_norm=pair.pls_orp_aux_target_norm,
        pls_orp_aux_mask=pair.pls_orp_aux_mask,
    )
    
    # Augment Control if criteria met
    if ctrl_meets_criteria:
        ext_time, ext_recovery = generate_virtual_points(
            ctrl_time, ctrl_recovery, ctrl_params, target_day, interval_days,
            cap=pair.ctrl_cap,
            curve_mode=pair.control.fit_curve_mode,
            gate_mid_day=pair.control.fit_gate_mid_day,
            gate_width_day=pair.control.fit_gate_width_day,
            fit_start_day=pair.control.prefit_fit_start_day,
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
        _ctrl_last_actual = float(np.nanmax(ctrl_time[np.isfinite(ctrl_time)])) if np.any(np.isfinite(ctrl_time)) else float("nan")
        new_pair.control = CurveData(
            status=pair.control.status,
            time=ext_time,
            recovery=ext_recovery,
            catalyst_cum=ctrl_catalyst_cum,
            lixiviant_cum=ctrl_lixiviant_cum,
            irrigation_rate_l_m2_h=ctrl_irrigation_rate,
            catalyst_addition_mg_l_reconstructed=np.zeros_like(ctrl_catalyst_cum, dtype=float),
            fit_params=pair.control.fit_params,
            row_index=pair.control.row_index,
            fit_curve_mode=pair.control.fit_curve_mode,
            fit_gate_mid_day=pair.control.fit_gate_mid_day,
            fit_gate_width_day=pair.control.fit_gate_width_day,
            prefit_fit_start_day=pair.control.prefit_fit_start_day,
            prefit_fit_start_day_source=pair.control.prefit_fit_start_day_source,
            pls_orp_profile=pair.control.pls_orp_profile,
            col_id=pair.control.col_id,
            catalyst_conc_col_mg_l=pair.control.catalyst_conc_col_mg_l,
            catalyst_start_day=pair.control.catalyst_start_day,
            catalyst_start_day_source=pair.control.catalyst_start_day_source,
            catalyst_effective_start_day=pair.control.catalyst_effective_start_day,
            catalyst_effective_start_day_source=pair.control.catalyst_effective_start_day_source,
            catalyst_stop_day=pair.control.catalyst_stop_day,
            avg_catalyst_dose_mg_l=pair.control.avg_catalyst_dose_mg_l,
            catalyst_dosage_start_day=pair.control.catalyst_dosage_start_day,
            catalyst_dosage_stop_day=pair.control.catalyst_dosage_stop_day,
            last_actual_day=_ctrl_last_actual,
        )

    # Augment Catalyzed if criteria met
    if cat_meets_criteria:
        ext_time, ext_recovery = generate_virtual_points(
            cat_time, cat_recovery, cat_params, target_day, interval_days,
            cap=pair.cat_cap,
            curve_mode=pair.catalyzed.fit_curve_mode,
            gate_mid_day=pair.catalyzed.fit_gate_mid_day,
            gate_width_day=pair.catalyzed.fit_gate_width_day,
            fit_start_day=pair.catalyzed.prefit_fit_start_day,
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
        _cat_last_actual = float(np.nanmax(cat_time[np.isfinite(cat_time)])) if np.any(np.isfinite(cat_time)) else float("nan")
        cat_catalyst_addition = resolve_observed_profile_on_time_grid(
            source_time_days=pair.catalyzed.time,
            target_time_days=ext_time,
            observed_profile=pair.catalyzed.catalyst_addition_mg_l_reconstructed,
            clip_min=0.0,
        )
        if cat_catalyst_addition is None:
            cat_catalyst_addition = reconstruct_catalyst_feed_conc_mg_l(
                catalyst_cum_kg_t=np.asarray(cat_catalyst_cum, dtype=float),
                lixiviant_cum_m3_t=np.asarray(cat_lixiviant_cum, dtype=float),
            )
        cat_catalyst_conc_col = _compute_cstr_column_concentration(
            time_days=ext_time,
            column_inner_diameter_m=column_inner_diameter_m,
            column_height_m=column_height_m,
            irrigation_rate_l_m2_h=np.asarray(cat_irrigation_rate, dtype=float),
            catalyst_feed_conc_mg_l=np.asarray(cat_catalyst_addition, dtype=float),
        )
        new_pair.catalyzed = CurveData(
            status=pair.catalyzed.status,
            time=ext_time,
            recovery=ext_recovery,
            catalyst_cum=cat_catalyst_cum,
            lixiviant_cum=cat_lixiviant_cum,
            irrigation_rate_l_m2_h=cat_irrigation_rate,
            catalyst_addition_mg_l_reconstructed=np.asarray(cat_catalyst_addition, dtype=float),
            fit_params=pair.catalyzed.fit_params,
            row_index=pair.catalyzed.row_index,
            fit_curve_mode=pair.catalyzed.fit_curve_mode,
            fit_gate_mid_day=pair.catalyzed.fit_gate_mid_day,
            fit_gate_width_day=pair.catalyzed.fit_gate_width_day,
            prefit_fit_start_day=pair.catalyzed.prefit_fit_start_day,
            prefit_fit_start_day_source=pair.catalyzed.prefit_fit_start_day_source,
            pls_orp_profile=pair.catalyzed.pls_orp_profile,
            col_id=pair.catalyzed.col_id,
            catalyst_conc_col_mg_l=np.asarray(cat_catalyst_conc_col, dtype=float),
            catalyst_start_day=pair.catalyzed.catalyst_start_day,
            catalyst_start_day_source=pair.catalyzed.catalyst_start_day_source,
            catalyst_effective_start_day=pair.catalyzed.catalyst_effective_start_day,
            catalyst_effective_start_day_source=pair.catalyzed.catalyst_effective_start_day_source,
            catalyst_stop_day=pair.catalyzed.catalyst_stop_day,
            avg_catalyst_dose_mg_l=pair.catalyzed.avg_catalyst_dose_mg_l,
            catalyst_dosage_start_day=pair.catalyzed.catalyst_dosage_start_day,
            catalyst_dosage_stop_day=pair.catalyzed.catalyst_dosage_stop_day,
            last_actual_day=_cat_last_actual,
        )
    
    return new_pair


def curve_augmentation_cache_key(curve: "CurveData") -> Tuple[int, str, str]:
    return (
        int(curve.row_index),
        str(curve.status),
        str(curve.col_id).strip(),
    )


def augment_curve_with_virtual_data(
    curve: "CurveData",
    feed_mass_kg: float,
    column_inner_diameter_m: float,
    column_height_m: float = np.nan,
    target_day: float = 2500.0,
    interval_days: float = 7.0,
    catalyst_history_window_days: Optional[float] = None,
    lix_history_window_days: Optional[float] = None,
    cap: Optional[float] = None,
) -> "CurveData":
    """
    Extend one project_col_id curve at most once.
    This keeps augmentation keyed to the individual column rather than to each
    control/catalyzed pair that references it.

    ``cap`` is retained only for call-site compatibility. The augmented target
    recovery is sampled directly from the stored pre-fit equation for this
    specific ``project_col_id`` across the full augmented time grid.
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
    fit_params = np.asarray(curve.fit_params, dtype=float)
    if len(fit_params) < 4 or not np.all(np.isfinite(fit_params[:4])):
        return curve

    pred = prefit_curve_prediction_np(
        curve.time,
        fit_params,
        curve.fit_curve_mode,
        curve.fit_gate_mid_day,
        curve.fit_gate_width_day,
    )
    metric_mask = prefit_metric_mask(curve.time, curve.prefit_fit_start_day)
    r2, rmse, _ = calculate_fit_metrics(curve.recovery[metric_mask], pred[metric_mask])
    meets_criteria = np.isfinite(r2) and r2 > 0.5 and np.isfinite(rmse) and rmse < 5.0
    if not meets_criteria:
        return curve

    _curve_last_actual = float(np.nanmax(curve.time[np.isfinite(curve.time)])) if np.any(np.isfinite(curve.time)) else float("nan")
    ext_time, ext_recovery = generate_virtual_points(
        curve.time,
        curve.recovery,
        fit_params,
        target_day,
        interval_days,
        cap=cap,
        curve_mode=curve.fit_curve_mode,
        gate_mid_day=curve.fit_gate_mid_day,
        gate_width_day=curve.fit_gate_width_day,
        fit_start_day=curve.prefit_fit_start_day,
    )
    ext_catalyst_cum, ext_lixiviant_cum, ext_irrigation_rate = extend_curve_dynamic_inputs_to_time_grid(
        curve_data=curve,
        target_time_days=ext_time,
        feed_mass_kg=feed_mass_kg,
        column_inner_diameter_m=column_inner_diameter_m,
        step_days=interval_days,
        catalyst_history_window_days=catalyst_history_window_days,
        lix_history_window_days=lix_history_window_days,
    )
    if str(curve.status) == "Control":
        ext_catalyst_addition = np.zeros_like(ext_time, dtype=float)
        ext_catalyst_conc_col = np.zeros_like(ext_time, dtype=float)
    else:
        resolved_ext_addition = resolve_observed_profile_on_time_grid(
            source_time_days=curve.time,
            target_time_days=ext_time,
            observed_profile=curve.catalyst_addition_mg_l_reconstructed,
            clip_min=0.0,
        )
        ext_catalyst_addition = (
            np.asarray(resolved_ext_addition, dtype=float)
            if resolved_ext_addition is not None
            else reconstruct_catalyst_feed_conc_mg_l(
                catalyst_cum_kg_t=np.asarray(ext_catalyst_cum, dtype=float),
                lixiviant_cum_m3_t=np.asarray(ext_lixiviant_cum, dtype=float),
            )
        )
        ext_catalyst_conc_col = _compute_cstr_column_concentration(
            time_days=ext_time,
            column_inner_diameter_m=column_inner_diameter_m,
            column_height_m=column_height_m,
            irrigation_rate_l_m2_h=np.asarray(ext_irrigation_rate, dtype=float),
            catalyst_feed_conc_mg_l=np.asarray(ext_catalyst_addition, dtype=float),
        )
    return CurveData(
        status=curve.status,
        time=np.asarray(ext_time, dtype=float),
        recovery=np.asarray(ext_recovery, dtype=float),
        catalyst_cum=np.asarray(ext_catalyst_cum, dtype=float),
        lixiviant_cum=np.asarray(ext_lixiviant_cum, dtype=float),
        irrigation_rate_l_m2_h=np.asarray(ext_irrigation_rate, dtype=float),
        catalyst_addition_mg_l_reconstructed=np.asarray(ext_catalyst_addition, dtype=float),
        fit_params=np.asarray(curve.fit_params, dtype=float),
        row_index=int(curve.row_index),
        fit_curve_mode=str(curve.fit_curve_mode),
        fit_gate_mid_day=float(curve.fit_gate_mid_day),
        fit_gate_width_day=float(curve.fit_gate_width_day),
        prefit_fit_start_day=float(curve.prefit_fit_start_day),
        prefit_fit_start_day_source=str(curve.prefit_fit_start_day_source),
        pls_orp_profile=np.asarray(curve.pls_orp_profile, dtype=float),
        col_id=str(curve.col_id),
        catalyst_conc_col_mg_l=np.asarray(ext_catalyst_conc_col, dtype=float),
        catalyst_start_day=float(curve.catalyst_start_day),
        catalyst_start_day_source=str(curve.catalyst_start_day_source),
        catalyst_effective_start_day=float(curve.catalyst_effective_start_day),
        catalyst_effective_start_day_source=str(curve.catalyst_effective_start_day_source),
        catalyst_stop_day=float(curve.catalyst_stop_day),
        avg_catalyst_dose_mg_l=float(curve.avg_catalyst_dose_mg_l),
        # Dosage-based start/stop days: unchanged by augmentation since
        # the raw observed dosage array is not extended — only the fitted
        # tail is extrapolated and these annotation days remain anchored to
        # the original observed window.
        catalyst_dosage_start_day=float(curve.catalyst_dosage_start_day),
        catalyst_dosage_stop_day=float(curve.catalyst_dosage_stop_day),
        last_actual_day=_curve_last_actual,
    )


def augment_pairs_with_virtual_data_by_col_id(
    pairs: List['PairSample'],
    target_day: float = 2500.0,
    interval_days: float = 7.0,
    catalyst_history_window_days: Optional[float] = None,
    lix_history_window_days: Optional[float] = None,
) -> List['PairSample']:
    """
    Augment unique curves per project_col_id/row_index once, then rebuild the
    pair list reusing those augmented curves across every pair that references them.
    """
    input_only_idx = {name: idx for idx, name in enumerate(INPUT_ONLY_COLUMNS)}
    curve_cache: Dict[Tuple[int, str, str], CurveData] = {}
    curve_geometry: Dict[Tuple[int, str, str], Tuple[float, float, float]] = {}

    for pair in pairs:
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
        column_height_m = (
            float(pair.input_only_raw[input_only_idx["column_height_m"]])
            if "column_height_m" in input_only_idx
            else np.nan
        )
        for curve in [pair.control, pair.catalyzed]:
            key = curve_augmentation_cache_key(curve)
            if key not in curve_geometry:
                curve_geometry[key] = (feed_mass_kg, column_inner_diameter_m, column_height_m)

    for pair in pairs:
        for curve, curve_cap in [
            (pair.control,   pair.ctrl_cap),
            (pair.catalyzed, pair.cat_cap),
        ]:
            key = curve_augmentation_cache_key(curve)
            if key in curve_cache:
                continue
            feed_mass_kg, column_inner_diameter_m, column_height_m = curve_geometry[key]
            curve_cache[key] = augment_curve_with_virtual_data(
                curve=curve,
                feed_mass_kg=feed_mass_kg,
                column_inner_diameter_m=column_inner_diameter_m,
                column_height_m=column_height_m,
                target_day=target_day,
                interval_days=interval_days,
                catalyst_history_window_days=catalyst_history_window_days,
                lix_history_window_days=lix_history_window_days,
                cap=curve_cap,
            )

    augmented_pairs: List[PairSample] = []
    for pair in pairs:
        augmented_pairs.append(
            PairSample(
                pair_id=pair.pair_id,
                sample_id=pair.sample_id,
                project_name=pair.project_name,
                static_raw=pair.static_raw,
                input_only_raw=pair.input_only_raw,
                control=curve_cache[curve_augmentation_cache_key(pair.control)],
                catalyzed=curve_cache[curve_augmentation_cache_key(pair.catalyzed)],
                control_static_raw=pair.control_static_raw,
                catalyzed_static_raw=pair.catalyzed_static_raw,
                static_scaled=pair.static_scaled,
                static_imputed=pair.static_imputed,
                ctrl_cap=pair.ctrl_cap,
                cat_cap=pair.cat_cap,
                control_cap_anchor_pct=pair.control_cap_anchor_pct,
                catalyzed_cap_anchor_pct=pair.catalyzed_cap_anchor_pct,
                control_cap_anchor_active=pair.control_cap_anchor_active,
                catalyzed_cap_anchor_active=pair.catalyzed_cap_anchor_active,
                orp_aux_target_raw=pair.orp_aux_target_raw,
                orp_aux_target_norm=pair.orp_aux_target_norm,
                orp_aux_mask=pair.orp_aux_mask,
                pls_orp_aux_target_raw=pair.pls_orp_aux_target_raw,
                pls_orp_aux_target_norm=pair.pls_orp_aux_target_norm,
                pls_orp_aux_mask=pair.pls_orp_aux_mask,
            )
        )

    return augmented_pairs


# ---------------------------
# Data objects
# ---------------------------
@dataclass
class CurveData:
    status: str
    time: np.ndarray
    recovery: np.ndarray
    # v11: catalyst_cum is the authoritative cumulative catalyst load (kg/t).
    # Primary source: cumulative_catalyst_addition_kg_t read directly from the
    # dataset (= Σ Catalyst_Added_mg / feed_kg / 1000, computed in Excel).
    # It is aligned and cleaned directly from the cumulative source profile.
    catalyst_cum: np.ndarray
    lixiviant_cum: np.ndarray
    irrigation_rate_l_m2_h: np.ndarray
    # Resolved feed dosage (mg/L): catalyst_addition_mg_l when present, otherwise
    # the cumulative-difference reconstruction for catalyzed rows.
    catalyst_addition_mg_l_reconstructed: np.ndarray
    fit_params: np.ndarray
    row_index: int
    fit_curve_mode: str = "biexponential"
    fit_gate_mid_day: float = field(default=float("nan"))
    fit_gate_width_day: float = field(default=float("nan"))
    prefit_fit_start_day: float = field(default=float("nan"))
    prefit_fit_start_day_source: str = ""
    # v11: PLS ORP profile – output ORP of the column (after ore interaction).
    # Analogous to feed ORP (ORP_PROFILE_COL) but represents the EXIT signal.
    # The catalyst concentration in the PLS decays following ore-characteristic
    # decay patterns heavily influenced by column_height_m.  This profile is
    # stored here for use in the orp_aux auxiliary target computation.
    pls_orp_profile: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    col_id: str = ""  # project_col_id for traceability
    # v11: CSTR-modelled average catalyst concentration in column pore solution (mg/L).
    # Computed from feed dosage (mg/L) and column geometry via a first-order
    # residence-time model.  Captures the *instantaneous* probability of catalyst-
    # chalcopyrite contact.  Always zero for Control columns.
    catalyst_conc_col_mg_l: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    # Explicit catalyst-start day used across model inference and plot
    # annotations. Priority: catalyst_start_days_of_leaching, then
    # transition_time for catalyzed columns, then last actual data day for
    # control columns.
    catalyst_start_day: float = field(default=float("nan"))
    catalyst_start_day_source: str = ""
    # Internal catalyst-onset used only by model logic/training.
    # Keeps the exported plot/output start day unchanged while allowing the
    # model to treat pre-addition segments of labeled catalyzed columns as
    # control-like behavior.
    catalyst_effective_start_day: float = field(default=float("nan"))
    catalyst_effective_start_day_source: str = ""
    # Dosage-array-derived stop day remains a plotting/extrapolation aid.
    catalyst_stop_day: float = field(default=float("nan"))
    # Average applied catalyst dose (mg/L), averaged over positive observed
    # feed-dosage entries for catalyzed columns.
    avg_catalyst_dose_mg_l: float = field(default=float("nan"))
    # Legacy dosage-array-derived start/stop annotation days retained for
    # backwards compatibility with older outputs.
    catalyst_dosage_start_day: float = field(default=float("nan"))
    catalyst_dosage_stop_day: float = field(default=float("nan"))
    # Last actually-observed time point (before any virtual augmentation).
    # Set by the augmentation step so downstream code can distinguish real
    # measured data points (t <= last_actual_day) from synthetic future
    # extension points (t > last_actual_day).  NaN for non-augmented curves.
    last_actual_day: float = field(default=float("nan"))


@dataclass
class PairSample:
    pair_id: str
    sample_id: str
    project_name: str
    static_raw: np.ndarray
    input_only_raw: np.ndarray
    control: CurveData
    catalyzed: CurveData
    control_static_raw: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    catalyzed_static_raw: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    static_scaled: Optional[np.ndarray] = None
    static_imputed: Optional[np.ndarray] = None
    # Per-sample leach caps (recovery-%) derived from mineralogy.
    # ctrl_cap applies to the control curve; cat_cap to the catalyzed curve.
    ctrl_cap: float = np.nan
    cat_cap: float = np.nan
    control_cap_anchor_pct: float = np.nan
    catalyzed_cap_anchor_pct: float = np.nan
    control_cap_anchor_active: bool = False
    catalyzed_cap_anchor_active: bool = False
    # Feed ORP auxiliary target (existing, from feed_orp_mv_ag_agcl)
    orp_aux_target_raw: float = np.nan
    orp_aux_target_norm: float = np.nan
    orp_aux_mask: float = 0.0
    # v11: PLS ORP auxiliary target (from pls_orp_mv_ag_agcl).
    # Represents the OUTPUT-side ORP of the leach system. The model learns
    # a per-sample decay shape for the catalyst in the PLS, guided by
    # ore characteristics (esp. column_height_m, residual_cpy_%, etc.).
    pls_orp_aux_target_raw: float = np.nan
    pls_orp_aux_target_norm: float = np.nan
    pls_orp_aux_mask: float = 0.0
    _tensor_cache: Dict[Any, Any] = field(default_factory=dict, repr=False, compare=False)
    _profile_cache: Dict[Any, Any] = field(default_factory=dict, repr=False, compare=False)

    def clear_tensor_cache(self) -> None:
        stale_keys = [
            key
            for key in self._tensor_cache.keys()
            if not (isinstance(key, tuple) and len(key) > 0 and key[0] == "plot_tensors")
        ]
        for key in stale_keys:
            self._tensor_cache.pop(key, None)

    def clear_runtime_cache(self) -> None:
        self._tensor_cache.clear()
        self._profile_cache.clear()


def make_pair_id(sample_id: Any, control_curve: CurveData, catalyzed_curve: CurveData) -> str:
    def _slug(value: Any, fallback: str) -> str:
        text = str(value).strip()
        if not text:
            text = fallback
        text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
        text = text.strip("-_.")
        return text or fallback

    sample_token = _slug(sample_id, "sample")
    ctrl_token = _slug(control_curve.col_id, f"ctrl-row-{int(control_curve.row_index)}")
    cat_token = _slug(catalyzed_curve.col_id, f"cat-row-{int(catalyzed_curve.row_index)}")
    return f"{sample_token}__ctrl_{ctrl_token}__cat_{cat_token}"


def combine_static_vectors(
    control_vec: np.ndarray,
    catalyzed_vec: np.ndarray,
    columns: List[str],
    context: str,
) -> np.ndarray:
    ctrl = validate_required_feature_vector(control_vec, columns, f"{context} control")
    cat = validate_required_feature_vector(catalyzed_vec, columns, f"{context} catalyzed")
    return 0.5 * (ctrl + cat)


def _device_cache_token(dev: torch.device) -> Tuple[str, int]:
    return dev.type, int(dev.index if dev.index is not None else -1)


def pair_length_signature(pair: PairSample) -> Tuple[int, int]:
    return int(len(pair.control.time)), int(len(pair.catalyzed.time))


def iter_pair_batches(
    pairs: List[PairSample],
    batch_size: int,
    rng: Optional[np.random.Generator] = None,
) -> List[List[PairSample]]:
    if batch_size <= 1 or len(pairs) <= 1:
        return [[pair] for pair in pairs]

    grouped: Dict[Tuple[int, int], List[PairSample]] = {}
    for pair in pairs:
        grouped.setdefault(pair_length_signature(pair), []).append(pair)

    batches: List[List[PairSample]] = []
    for group_pairs in grouped.values():
        for start_idx in range(0, len(group_pairs), batch_size):
            batches.append(group_pairs[start_idx : start_idx + batch_size])

    if rng is not None and len(batches) > 1:
        rng.shuffle(batches)
    return batches


def build_pair_training_batch(
    pairs: List[PairSample],
    cum_scale: float,
    lix_scale: float,
    irrigation_scale: float,
    conc_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    if len(pairs) == 0:
        raise ValueError("build_pair_training_batch requires at least one pair.")

    bundles = [
        get_pair_training_tensors(pair, cum_scale, lix_scale, irrigation_scale, conc_scale)
        for pair in pairs
    ]
    batch: Dict[str, torch.Tensor] = {}
    concat_keys = {
        "x",
        "x_raw",
        "x_input_only",
        "ctrl_curve_override_raw",
        "cat_curve_override_raw",
        "ctrl_cap_anchor_pct",
        "cat_cap_anchor_pct",
        "ctrl_cap_anchor_active",
        "cat_cap_anchor_active",
    }
    for key in bundles[0]:
        values = [bundle[key] for bundle in bundles]
        if key in concat_keys:
            batch[key] = torch.cat(values, dim=0)
        else:
            batch[key] = torch.stack(values, dim=0)

    ref_dtype = batch["ctrl_y"].dtype
    batch["ctrl_cap"] = torch.as_tensor(
        [float(pair.ctrl_cap) for pair in pairs],
        dtype=ref_dtype,
        device=device,
    )
    batch["cat_cap"] = torch.as_tensor(
        [float(pair.cat_cap) for pair in pairs],
        dtype=ref_dtype,
        device=device,
    )
    return batch


def get_pair_training_tensors(
    pair: PairSample,
    cum_scale: float,
    lix_scale: float,
    irrigation_scale: float,
    conc_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    cache_key = (
        "training_tensors",
        *_device_cache_token(device),
        torch_dtype_name(MODEL_TORCH_DTYPE),
        float(cum_scale),
        float(lix_scale),
        float(irrigation_scale),
        float(conc_scale),
    )
    cached = pair._tensor_cache.get(cache_key)
    if cached is not None:
        return cached

    dtype = MODEL_TORCH_DTYPE
    cum_inv = 1.0 / max(float(cum_scale), 1e-6)
    lix_inv = 1.0 / max(float(lix_scale), 1e-6)
    irr_inv = 1.0 / max(float(irrigation_scale), 1e-6)
    conc_inv = 1.0 / max(float(conc_scale), 1e-6)

    static_scaled = pair.static_scaled if pair.static_scaled is not None else pair.static_raw
    raw_for_geometry = pair.static_imputed if pair.static_imputed is not None else pair.static_raw
    ctrl_curve_override_raw = build_curve_specific_static_override(
        pair.control_static_raw if pair.control_static_raw.size > 0 else pair.static_raw
    )
    cat_curve_override_raw = build_curve_specific_static_override(
        pair.catalyzed_static_raw if pair.catalyzed_static_raw.size > 0 else pair.static_raw
    )
    ctrl_cap_anchor_active = bool(pair.control_cap_anchor_active) and np.isfinite(float(pair.control_cap_anchor_pct))
    cat_cap_anchor_active = bool(pair.catalyzed_cap_anchor_active) and np.isfinite(float(pair.catalyzed_cap_anchor_pct))
    ctrl_cap_anchor_pct = (
        float(pair.control_cap_anchor_pct)
        if ctrl_cap_anchor_active
        else (float(pair.ctrl_cap) if np.isfinite(float(pair.ctrl_cap)) else 100.0)
    )
    cat_cap_anchor_pct = (
        float(pair.catalyzed_cap_anchor_pct)
        if cat_cap_anchor_active
        else (float(pair.cat_cap) if np.isfinite(float(pair.cat_cap)) else 100.0)
    )
    ctrl_true_on_cat_t = np.interp(
        np.asarray(pair.catalyzed.time, dtype=float),
        np.asarray(pair.control.time, dtype=float),
        np.asarray(pair.control.recovery, dtype=float),
        left=float(pair.control.recovery[0]),
        right=float(pair.control.recovery[-1]),
    )

    bundle = {
        "x": torch.as_tensor(np.asarray(static_scaled, dtype=np.float32), dtype=dtype, device=device).unsqueeze(0),
        "x_raw": torch.as_tensor(np.asarray(raw_for_geometry, dtype=np.float32), dtype=dtype, device=device).unsqueeze(0),
        "x_input_only": torch.as_tensor(
            np.asarray(pair.input_only_raw, dtype=np.float32),
            dtype=dtype,
            device=device,
        ).unsqueeze(0),
        "ctrl_curve_override_raw": torch.as_tensor(
            np.asarray(ctrl_curve_override_raw, dtype=np.float32),
            dtype=dtype,
            device=device,
        ).unsqueeze(0),
        "cat_curve_override_raw": torch.as_tensor(
            np.asarray(cat_curve_override_raw, dtype=np.float32),
            dtype=dtype,
            device=device,
        ).unsqueeze(0),
        "ctrl_cap_anchor_pct": torch.as_tensor(
            [[ctrl_cap_anchor_pct]],
            dtype=dtype,
            device=device,
        ),
        "cat_cap_anchor_pct": torch.as_tensor(
            [[cat_cap_anchor_pct]],
            dtype=dtype,
            device=device,
        ),
        "ctrl_cap_anchor_active": torch.as_tensor(
            [[1.0 if ctrl_cap_anchor_active else 0.0]],
            dtype=dtype,
            device=device,
        ),
        "cat_cap_anchor_active": torch.as_tensor(
            [[1.0 if cat_cap_anchor_active else 0.0]],
            dtype=dtype,
            device=device,
        ),
        "ctrl_t": torch.as_tensor(np.asarray(pair.control.time, dtype=np.float32), dtype=dtype, device=device),
        "ctrl_y": torch.as_tensor(np.asarray(pair.control.recovery, dtype=np.float32), dtype=dtype, device=device),
        "ctrl_c": torch.as_tensor(
            np.asarray(pair.control.catalyst_cum, dtype=np.float32) * cum_inv,
            dtype=dtype,
            device=device,
        ),
        "ctrl_l": torch.as_tensor(
            np.asarray(pair.control.lixiviant_cum, dtype=np.float32) * lix_inv,
            dtype=dtype,
            device=device,
        ),
        "ctrl_irr": torch.as_tensor(
            np.asarray(pair.control.irrigation_rate_l_m2_h, dtype=np.float32) * irr_inv,
            dtype=dtype,
            device=device,
        ),
        "ctrl_start_day": torch.as_tensor(float(pair.control.catalyst_start_day), dtype=dtype, device=device),
        "ctrl_prefit_params": torch.as_tensor(
            np.asarray(pair.control.fit_params[:4], dtype=np.float32),
            dtype=dtype,
            device=device,
        ),
        "cat_t": torch.as_tensor(np.asarray(pair.catalyzed.time, dtype=np.float32), dtype=dtype, device=device),
        "cat_y": torch.as_tensor(np.asarray(pair.catalyzed.recovery, dtype=np.float32), dtype=dtype, device=device),
        "cat_c": torch.as_tensor(
            np.asarray(pair.catalyzed.catalyst_cum, dtype=np.float32) * cum_inv,
            dtype=dtype,
            device=device,
        ),
        "cat_l": torch.as_tensor(
            np.asarray(pair.catalyzed.lixiviant_cum, dtype=np.float32) * lix_inv,
            dtype=dtype,
            device=device,
        ),
        "cat_irr": torch.as_tensor(
            np.asarray(pair.catalyzed.irrigation_rate_l_m2_h, dtype=np.float32) * irr_inv,
            dtype=dtype,
            device=device,
        ),
        "cat_start_day": torch.as_tensor(float(pair.catalyzed.catalyst_start_day), dtype=dtype, device=device),
        "cat_effective_start_day": torch.as_tensor(
            float(pair.catalyzed.catalyst_effective_start_day),
            dtype=dtype,
            device=device,
        ),
        "cat_prefit_params": torch.as_tensor(
            np.asarray(pair.catalyzed.fit_params[:4], dtype=np.float32),
            dtype=dtype,
            device=device,
        ),
        "ctrl_true_on_cat_t": torch.as_tensor(np.asarray(ctrl_true_on_cat_t, dtype=np.float32), dtype=dtype, device=device),
        "orp_aux_target": torch.as_tensor(float(pair.orp_aux_target_norm), dtype=dtype, device=device),
        "orp_aux_mask": torch.as_tensor(float(pair.orp_aux_mask), dtype=dtype, device=device),
    }
    bundle["cat_ctrl_c"] = torch.zeros_like(bundle["cat_c"])
    bundle["cat_ctrl_start_day"] = bundle["cat_effective_start_day"]
    cat_post_start_mask = interval_active_mask_from_start_day(
        np.asarray(pair.catalyzed.time, dtype=float),
        float(pair.catalyzed.catalyst_effective_start_day),
    ).astype(np.float32, copy=False)
    bundle["cat_post_start_mask"] = torch.as_tensor(cat_post_start_mask, dtype=dtype, device=device)
    bundle["cat_pre_start_mask"] = torch.as_tensor(1.0 - cat_post_start_mask, dtype=dtype, device=device)
    # The resolved catalyst dosage vector is stored on CurveData and exported,
    # but the deployed checkpoints consume the derived cumulative catalyst history
    # plus the CSTR pore-solution concentration signal. Adding direct dosage as a
    # third dynamic tensor would require retraining and checkpoint migration.
    # cat_conc: catalyzed arm – the CSTR-estimated pore-solution concentration.
    # ctrl_conc: control arm – always zero (no catalyst in feed).
    _cat_conc_arr = np.asarray(pair.catalyzed.catalyst_conc_col_mg_l, dtype=np.float32)
    if _cat_conc_arr.size == len(pair.catalyzed.time):
        _cat_conc_norm = _cat_conc_arr * conc_inv
    else:
        _cat_conc_norm = np.zeros(len(pair.catalyzed.time), dtype=np.float32)
    bundle["cat_conc"] = torch.as_tensor(_cat_conc_norm, dtype=dtype, device=device)
    bundle["ctrl_conc"] = torch.zeros_like(bundle["ctrl_c"])
    pair._tensor_cache[cache_key] = bundle
    return bundle


def get_pair_plot_profile(pair: PairSample) -> Dict[str, Any]:
    target_day = float(CONFIG.get("ensemble_plot_target_day", 2500.0))
    step_days = float(CONFIG.get("ensemble_plot_step_days", 1.0))
    history_window_days = float(CONFIG.get("catalyst_extension_window_days", 21.0))
    cache_key = ("plot_profile", target_day, step_days, history_window_days)
    cached = pair._profile_cache.get(cache_key)
    if cached is not None:
        return cached

    feed_mass_kg = (
        float(pair.input_only_raw[INPUT_ONLY_INDEX[FEED_MASS_COL]])
        if FEED_MASS_COL in INPUT_ONLY_INDEX
        else np.nan
    )
    column_inner_diameter_m = (
        float(pair.input_only_raw[INPUT_ONLY_INDEX["column_inner_diameter_m"]])
        if "column_inner_diameter_m" in INPUT_ONLY_INDEX
        else np.nan
    )
    plot_profile = build_shared_ensemble_plot_profile(
        time_days=pair.catalyzed.time,
        catalyst_cum=pair.catalyzed.catalyst_cum,
        lixiviant_cum=pair.catalyzed.lixiviant_cum,
        catalyst_feed_conc_mg_l=pair.catalyzed.catalyst_addition_mg_l_reconstructed,
        feed_mass_kg=feed_mass_kg,
        column_inner_diameter_m=column_inner_diameter_m,
        target_day=target_day,
        step_days=step_days,
        history_window_days=history_window_days,
    )

    # Plot annotations should follow the explicit per-column catalyst-start day
    # from the dataset rather than inferring onset from the reconstructed
    # cumulative profile.
    _start = float(pair.catalyzed.catalyst_start_day)
    if np.isfinite(_start):
        plot_profile = dict(plot_profile)  # shallow copy so we can override keys safely
        plot_profile["catalyst_addition_start_day"] = _start

    pair._profile_cache[cache_key] = plot_profile
    return plot_profile


def get_pair_plot_tensors(
    pair: PairSample,
    cum_scale: float,
    lix_scale: float,
    irrigation_scale: float,
) -> Dict[str, Any]:
    target_day = float(CONFIG.get("ensemble_plot_target_day", 2500.0))
    step_days = float(CONFIG.get("ensemble_plot_step_days", 1.0))
    history_window_days = float(CONFIG.get("catalyst_extension_window_days", 21.0))
    cache_key = (
        "plot_tensors",
        *_device_cache_token(device),
        torch_dtype_name(MODEL_TORCH_DTYPE),
        float(cum_scale),
        float(lix_scale),
        float(irrigation_scale),
        target_day,
        step_days,
        history_window_days,
    )
    cached = pair._tensor_cache.get(cache_key)
    if cached is not None:
        return cached

    dtype = MODEL_TORCH_DTYPE
    cum_inv = 1.0 / max(float(cum_scale), 1e-6)
    lix_inv = 1.0 / max(float(lix_scale), 1e-6)
    irr_inv = 1.0 / max(float(irrigation_scale), 1e-6)
    plot_profile = get_pair_plot_profile(pair)

    plot_time_days = np.asarray(plot_profile["plot_time_days"], dtype=np.float32)
    plot_bundle = {
        "plot_profile": plot_profile,
        "control_plot_time_days": np.asarray(plot_time_days, dtype=float),
        "ctrl_plot_t": torch.as_tensor(plot_time_days, dtype=dtype, device=device),
        "ctrl_plot_start_day": torch.as_tensor(float(pair.control.catalyst_start_day), dtype=dtype, device=device),
        "cat_plot_start_day": torch.as_tensor(float(pair.catalyzed.catalyst_start_day), dtype=dtype, device=device),
        "cat_plot_effective_start_day": torch.as_tensor(
            float(pair.catalyzed.catalyst_effective_start_day),
            dtype=dtype,
            device=device,
        ),
        "plot_l": torch.as_tensor(
            np.asarray(plot_profile["plot_lixiviant_cum"], dtype=np.float32) * lix_inv,
            dtype=dtype,
            device=device,
        ),
        "plot_irr": torch.as_tensor(
            np.asarray(plot_profile["plot_irrigation_rate_l_m2_h"], dtype=np.float32) * irr_inv,
            dtype=dtype,
            device=device,
        ),
        "cat_plot_c": torch.as_tensor(
            np.asarray(plot_profile["plot_catalyst_cum"], dtype=np.float32) * cum_inv,
            dtype=dtype,
            device=device,
        ),
    }
    plot_bundle["ctrl_plot_c"] = torch.zeros_like(plot_bundle["ctrl_plot_t"])
    pair._tensor_cache[cache_key] = plot_bundle
    return plot_bundle


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

    # v11: also report col_id counts and catalyst_status breakdown
    if COL_ID_COL in df.columns:
        summary["n_unique_cols"] = int(df[COL_ID_COL].nunique(dropna=True))

    dynamic_stats = {}
    for col in [
        TIME_COL_COLUMNS, CATALYST_CUM_COL, LIXIVIANT_CUM_COL,
        TARGET_COLUMNS, ORP_PROFILE_COL, PLS_ORP_PROFILE_COL,
    ]:
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


def _filtered_copy_without_configured_excluded_ids(df: pd.DataFrame) -> pd.DataFrame:
    df_filtered = df.copy()
    if PAIR_ID_COL in df_filtered.columns and EXCLUDED_TRAIN_PAIR_IDS:
        pair_ids = df_filtered[PAIR_ID_COL].astype(str).str.strip()
        df_filtered = df_filtered.loc[~pair_ids.isin(EXCLUDED_TRAIN_PAIR_IDS)].copy()
    if COL_ID_COL in df_filtered.columns and EXCLUDED_TRAIN_COL_IDS:
        col_ids = df_filtered[COL_ID_COL].astype(str).str.strip()
        df_filtered = df_filtered.loc[~col_ids.isin(EXCLUDED_TRAIN_COL_IDS)].copy()
    return df_filtered


def assert_dataframe_respects_training_exclusions(df: pd.DataFrame, context: str) -> None:
    bad_pair_ids: List[str] = []
    bad_col_ids: List[str] = []
    if PAIR_ID_COL in df.columns and EXCLUDED_TRAIN_PAIR_IDS:
        pair_ids = df[PAIR_ID_COL].astype(str).str.strip()
        bad_pair_ids = sorted(set(pair_ids.loc[pair_ids.isin(EXCLUDED_TRAIN_PAIR_IDS)].tolist()))
    if COL_ID_COL in df.columns and EXCLUDED_TRAIN_COL_IDS:
        col_ids = df[COL_ID_COL].astype(str).str.strip()
        bad_col_ids = sorted(set(col_ids.loc[col_ids.isin(EXCLUDED_TRAIN_COL_IDS)].tolist()))
    if bad_pair_ids or bad_col_ids:
        raise ValueError(
            f"{context} still contains excluded training ids | "
            f"project_sample_id={bad_pair_ids[:10]} | project_col_id={bad_col_ids[:10]}"
        )


def assert_pairs_respect_training_exclusions(pairs: List["PairSample"], context: str) -> None:
    bad_sample_ids = sorted(
        {
            str(p.sample_id).strip()
            for p in pairs
            if str(p.sample_id).strip() in EXCLUDED_TRAIN_PAIR_IDS
        }
    )
    bad_col_ids = sorted(
        {
            str(col_id).strip()
            for p in pairs
            for col_id in [p.control.col_id, p.catalyzed.col_id]
            if str(col_id).strip() in EXCLUDED_TRAIN_COL_IDS
        }
    )
    if bad_sample_ids or bad_col_ids:
        raise ValueError(
            f"{context} still contains excluded training ids | "
            f"project_sample_id={bad_sample_ids[:10]} | project_col_id={bad_col_ids[:10]}"
        )


def apply_training_pair_exclusions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply all exclusion filters in sequence.

    v11 exclusion pipeline:
      1. Auto-filter: remove rows where cu_recovery_% has too few valid points
         (< MIN_VALID_RECOVERY_POINTS) to be usable for training.
      2. project_sample_id exclusion (EXCLUDED_TRAIN_PAIR_IDS).
      3. project_col_id exclusion (EXCLUDED_TRAIN_COL_IDS).
    """
    MIN_VALID_RECOVERY_POINTS = 6

    summary: Dict[str, Any] = {
        "pair_id_column": PAIR_ID_COL,
        "col_id_column": COL_ID_COL,
        "excluded_pair_ids": sorted(EXCLUDED_TRAIN_PAIR_IDS),
        "excluded_col_ids": sorted(EXCLUDED_TRAIN_COL_IDS),
        "applied": False,
        "auto_excluded_recovery_row_count": 0,
        "excluded_row_count": 0,
        "excluded_pair_count": 0,
        "excluded_col_count": 0,
        "remaining_row_count": int(len(df)),
    }

    # -- Step 1: Auto-filter rows with insufficient valid cu_recovery_% data --
    if TARGET_COLUMNS in df.columns:
        def _count_valid_recovery(val: Any) -> int:
            arr = parse_listlike(val)
            return int(np.isfinite(arr).sum())
        valid_counts = df[TARGET_COLUMNS].map(_count_valid_recovery)
        auto_exclude_mask = valid_counts < MIN_VALID_RECOVERY_POINTS
        n_auto = int(auto_exclude_mask.sum())
        if n_auto > 0:
            excluded_ids = (
                df.loc[auto_exclude_mask, COL_ID_COL].astype(str).tolist()
                if COL_ID_COL in df.columns
                else []
            )
            print(
                f"[Data] Auto-excluded {n_auto} rows with <{MIN_VALID_RECOVERY_POINTS} valid "
                f"cu_recovery_% points: {excluded_ids}"
            )
            df = df.loc[~auto_exclude_mask].copy()
            summary["auto_excluded_recovery_row_count"] = n_auto

    # -- Step 2: project_sample_id exclusion --
    if EXCLUDED_TRAIN_PAIR_IDS and PAIR_ID_COL in df.columns:
        pair_ids = df[PAIR_ID_COL].astype(str).str.strip()
        pair_exclude_mask = pair_ids.isin(EXCLUDED_TRAIN_PAIR_IDS)
        n_pair_rows = int(pair_exclude_mask.sum())
        n_pair_ids  = int(pair_ids.loc[pair_exclude_mask].nunique(dropna=True))
        df = df.loc[~pair_exclude_mask].copy()
        summary.update({
            "applied": True,
            "excluded_row_count": n_pair_rows,
            "excluded_pair_count": n_pair_ids,
        })

    # -- Step 3: project_col_id exclusion --
    if EXCLUDED_TRAIN_COL_IDS and COL_ID_COL in df.columns:
        col_ids = df[COL_ID_COL].astype(str).str.strip()
        col_exclude_mask = col_ids.isin(EXCLUDED_TRAIN_COL_IDS)
        n_col_rows = int(col_exclude_mask.sum())
        n_col_ids  = int(col_ids.loc[col_exclude_mask].nunique(dropna=True))
        df = df.loc[~col_exclude_mask].copy()
        summary.update({
            "applied": True,
            "excluded_row_count": summary.get("excluded_row_count", 0) + n_col_rows,
            "excluded_col_count": n_col_ids,
        })

    summary["remaining_row_count"] = int(len(df))
    assert_dataframe_respects_training_exclusions(df, context="apply_training_pair_exclusions")
    return df, summary


def _prefit_final_profile_value(value: Any) -> float:
    arr = np.asarray(parse_listlike(value), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.nanmax(arr))


def _prefit_group_scalar(value: Any, decimals: int = 4) -> str:
    scalar = scalar_from_maybe_array(value)
    if not np.isfinite(scalar):
        return "nan"
    return f"{float(scalar):.{int(decimals)}f}"


def _prefit_operation_group_key(row: Dict[str, Any]) -> str:
    status = normalize_status(row.get(STATUS_COL_PRIMARY, row.get(STATUS_COL_FALLBACK, "")))
    parts = [
        str(row.get(PAIR_ID_COL, "")).strip(),
        status,
        _prefit_group_scalar(row.get("material_size_p80_in", np.nan), 3),
        _prefit_group_scalar(row.get("column_height_m", np.nan), 3),
        _prefit_group_scalar(row.get("lixiviant_initial_fe_mg_l", np.nan), 2),
        _prefit_group_scalar(row.get("lixiviant_initial_ph", np.nan), 3),
        _prefit_group_scalar(row.get("lixiviant_initial_orp_mv", np.nan), 2),
    ]
    return "|".join(parts)


def _run_prefit_payloads(row_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prefit_parallel = resolve_prefit_parallelism(len(row_payloads))
    if int(prefit_parallel["workers"]) <= 1 or len(row_payloads) <= 1:
        return [prefit_biexponential_for_row_payload(row) for row in row_payloads]

    chunk_size = int(prefit_parallel["chunk_size"])
    chunks = [row_payloads[i:i + chunk_size] for i in range(0, len(row_payloads), chunk_size)]
    print(
        "[Prefit] Running biexponential pre-fit in parallel "
        f"with {prefit_parallel['workers']} workers across {len(chunks)} chunks."
    )

    out_rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(prefit_parallel["workers"])) as executor:
        future_to_chunk = {executor.submit(prefit_biexponential_for_row_chunk, chunk): chunk_idx for chunk_idx, chunk in enumerate(chunks)}
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                out_rows.extend(future.result())
            except Exception as exc:
                raise RuntimeError(f"[Prefit] Failed biexponential chunk {chunk_idx}: {exc}") from exc

    out_rows.sort(key=lambda row: int(row["row_index"]))
    return out_rows


def _finite_median(values: List[float]) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.nanmedian(vals))


def apply_duplicate_operational_prefit_targets(
    row_payloads: List[Dict[str, Any]],
    out_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not bool(CONFIG.get("use_duplicate_operational_prefit_targets", True)):
        return out_rows

    payload_by_idx = {int(row.get("row_index", 0)): row for row in row_payloads}
    result_by_idx = {int(row.get("row_index", 0)): row for row in out_rows}
    groups: Dict[str, List[int]] = {}
    for payload in row_payloads:
        idx = int(payload.get("row_index", 0))
        groups.setdefault(_prefit_operation_group_key(payload), []).append(idx)

    refit_payloads: List[Dict[str, Any]] = []
    target_day = float(CONFIG.get("prefit_asymptote_target_day", 2500.0))
    control_weight = float(CONFIG.get("prefit_duplicate_day2500_target_weight", 1.0))

    # -----------------------------------------------------------------------
    # Pre-compute median control fit_asymptote per project_sample_id.
    # Used to enforce: catalyzed day-2500 recovery ≥ ctrl_median + uplift_pct.
    # -----------------------------------------------------------------------
    _ctrl_asymptotes_by_sample: Dict[str, List[float]] = {}
    for payload in row_payloads:
        idx = int(payload.get("row_index", 0))
        status = normalize_status(payload.get(STATUS_COL_PRIMARY, payload.get(STATUS_COL_FALLBACK, "")))
        if status != "Control":
            continue
        sample_id = str(payload.get(PAIR_ID_COL, "")).strip()
        asym = scalar_from_maybe_array(result_by_idx.get(idx, {}).get("fit_asymptote", np.nan))
        _ctrl_asymptotes_by_sample.setdefault(sample_id, []).append(asym)
    ctrl_median_by_sample_id: Dict[str, float] = {
        k: _finite_median(v) for k, v in _ctrl_asymptotes_by_sample.items()
    }

    for group_key, indices in groups.items():
        statuses = {
            normalize_status(payload_by_idx[idx].get(STATUS_COL_PRIMARY, payload_by_idx[idx].get(STATUS_COL_FALLBACK, "")))
            for idx in indices
        }
        if len(statuses) != 1:
            continue
        status = next(iter(statuses))

        if status == "Control":
            # Control re-fit: skip single-column groups (no duplicates to align).
            if len(indices) < 2:
                continue
            base_targets = {
                idx: scalar_from_maybe_array(result_by_idx.get(idx, {}).get("fit_sample_cap", np.nan))
                for idx in indices
            }
            target = _finite_median(list(base_targets.values()))
            if not np.isfinite(target):
                continue
            for idx in indices:
                payload = dict(payload_by_idx[idx])
                payload["prefit_duplicate_target_active"] = True
                payload["prefit_duplicate_target_recovery"] = float(target)
                payload["prefit_duplicate_target_mode"] = "exact"
                payload["prefit_duplicate_target_weight"] = control_weight
                payload["prefit_duplicate_group_key"] = group_key
                payload["prefit_duplicate_target_day"] = target_day
                refit_payloads.append(payload)

        elif status == "Catalyzed":
            # ---------------------------------------------------------------
            # Catalyzed re-fit goals (Round 3):
            #   1. Similar catalyst doses → similar day-2500 recovery
            #      (shared median target within each dose bin, monotone across tiers)
            #   2. Catalyzed always ≥ control + uplift_pct at day 2500
            #   3. Higher dose → higher recovery at day 2500
            #   4. "minimum" mode: only pull UP — never drag a good fit down
            #
            # Base target: fit_asymptote (data-driven), NOT fit_sample_cap
            # (chemistry ceiling).  Using the data-driven asymptote prevents
            # short-test over-prediction while still aligning same-dose columns.
            # ---------------------------------------------------------------
            if bool(CONFIG.get("prefit_cat_skip_duplicate_targets", False)):
                continue

            cat_weight = float(CONFIG.get(
                "prefit_cat_duplicate_target_weight",
                CONFIG.get("prefit_catalyst_monotonic_day2500_target_weight", 1.0),
            ))
            cat_mode = str(CONFIG.get("prefit_cat_duplicate_target_mode", "minimum")).strip().lower()
            if cat_mode not in {"exact", "minimum"}:
                cat_mode = "minimum"
            min_uplift_pct = float(CONFIG.get("prefit_cat_min_uplift_over_ctrl_pct", 2.0))
            use_asymptote_target = bool(CONFIG.get("prefit_cat_use_asymptote_as_target", True))

            # Control floor for this sample_id.
            sample_id = str(payload_by_idx[indices[0]].get(PAIR_ID_COL, "")).strip()
            ctrl_median = ctrl_median_by_sample_id.get(sample_id, np.nan)
            ctrl_floor = (ctrl_median + min_uplift_pct) if np.isfinite(ctrl_median) else np.nan

            # Base target per index: data-driven asymptote or chemistry cap.
            result_key = "fit_asymptote" if use_asymptote_target else "fit_sample_cap"
            base_targets_cat = {
                idx: scalar_from_maybe_array(result_by_idx.get(idx, {}).get(result_key, np.nan))
                for idx in indices
            }

            # ---------------------------------------------------------------
            # Dose binning (identical logic to original, preserved verbatim).
            # ---------------------------------------------------------------
            dose_groups: Dict[str, List[int]] = {}
            dose_values: Dict[str, float] = {}
            dose_bin_pct_tol = float(CONFIG.get("prefit_catalyst_dose_bin_pct_tolerance", 0.0))
            _raw_doses: Dict[int, float] = {
                idx: _prefit_final_profile_value(payload_by_idx[idx].get(CATALYST_CUM_COL, np.nan))
                for idx in indices
            }
            if dose_bin_pct_tol > 0.0:
                finite_doses_sorted = sorted(
                    set(d for d in _raw_doses.values() if np.isfinite(d) and d > 1e-12)
                )
                bin_reps: List[float] = []
                dose_to_bin_key: Dict[float, str] = {}
                for d in finite_doses_sorted:
                    matched = False
                    for b_idx, rep in enumerate(bin_reps):
                        if rep > 1e-12 and abs(d - rep) / rep <= dose_bin_pct_tol / 100.0:
                            dose_to_bin_key[d] = f"dose_bin_{b_idx}"
                            matched = True
                            break
                    if not matched:
                        b_idx = len(bin_reps)
                        bin_reps.append(d)
                        dose_to_bin_key[d] = f"dose_bin_{b_idx}"
                for idx in indices:
                    d = _raw_doses[idx]
                    if np.isfinite(d) and d > 1e-12 and d in dose_to_bin_key:
                        dose_key = dose_to_bin_key[d]
                        bin_rep_dose = bin_reps[int(dose_key.split("_")[-1])]
                    else:
                        dose_key = "nan"
                        bin_rep_dose = np.nan
                    dose_groups.setdefault(dose_key, []).append(idx)
                    dose_values[dose_key] = bin_rep_dose
            else:
                for idx in indices:
                    dose = _raw_doses[idx]
                    dose_key = "nan" if not np.isfinite(dose) else f"{dose:.6f}"
                    dose_groups.setdefault(dose_key, []).append(idx)
                    dose_values[dose_key] = dose

            # ---------------------------------------------------------------
            # Per-dose-bin target = median fit_asymptote within the bin,
            # then floored at ctrl_median + uplift_pct.
            # ---------------------------------------------------------------
            dose_targets: Dict[str, float] = {}
            for dose_key, dose_indices in dose_groups.items():
                bin_median = _finite_median([base_targets_cat[idx] for idx in dose_indices])
                # Apply control floor.
                if np.isfinite(ctrl_floor):
                    if np.isfinite(bin_median):
                        bin_median = max(bin_median, ctrl_floor)
                    else:
                        bin_median = ctrl_floor
                dose_targets[dose_key] = bin_median

            # ---------------------------------------------------------------
            # Monotone enforcement: higher catalyst dose → higher target.
            # ---------------------------------------------------------------
            ordered_dose_keys = sorted(
                dose_groups,
                key=lambda key: (
                    not np.isfinite(dose_values.get(key, np.nan)),
                    dose_values.get(key, np.nan) if np.isfinite(dose_values.get(key, np.nan)) else 0.0,
                ),
            )
            running_target = -np.inf
            monotone_targets: Dict[str, float] = {}
            for dose_key in ordered_dose_keys:
                dose_target = dose_targets.get(dose_key, np.nan)
                if not np.isfinite(dose_target):
                    continue
                running_target = max(running_target, float(dose_target))
                monotone_targets[dose_key] = running_target

            for dose_key, dose_indices in dose_groups.items():
                target = monotone_targets.get(dose_key, np.nan)
                if not np.isfinite(target):
                    continue
                for idx in dose_indices:
                    payload = dict(payload_by_idx[idx])
                    payload["prefit_duplicate_target_active"] = True
                    payload["prefit_duplicate_target_recovery"] = float(target)
                    payload["prefit_duplicate_target_mode"] = cat_mode
                    payload["prefit_duplicate_target_weight"] = cat_weight
                    payload["prefit_duplicate_group_key"] = group_key
                    payload["prefit_duplicate_target_day"] = target_day
                    refit_payloads.append(payload)

    if not refit_payloads:
        return out_rows

    print(f"[Prefit] Re-fitting {len(refit_payloads)} duplicate-operation rows with day-{target_day:.0f} targets.")
    refit_rows = _run_prefit_payloads(refit_payloads)
    for row in refit_rows:
        result_by_idx[int(row["row_index"])] = row
    return [result_by_idx[int(row["row_index"])] for row in out_rows]


def prefit_biexponential_for_rows(df: pd.DataFrame) -> pd.DataFrame:
    row_payloads = build_prefit_row_payloads(df)
    out_rows = _run_prefit_payloads(row_payloads)
    out_rows = apply_duplicate_operational_prefit_targets(row_payloads, out_rows)
    return pd.DataFrame(out_rows)


def resolve_prefit_parallelism(n_rows: int) -> Dict[str, int]:
    total_cores = max(1, int(os.cpu_count() or 1))
    requested_workers_raw = str(os.environ.get("ROSETTA_PREFIT_WORKERS", "")).strip()
    if requested_workers_raw:
        requested_workers = int(requested_workers_raw)
    else:
        requested_workers = int(CONFIG.get("prefit_parallel_workers", 0))
    if requested_workers <= 0:
        requested_workers = total_cores

    min_rows_per_worker = max(1, int(CONFIG.get("prefit_min_rows_per_worker", 12)))
    max_useful_workers = max(1, int(math.ceil(float(max(1, n_rows)) / float(min_rows_per_worker))))
    workers = max(1, min(requested_workers, total_cores, max(1, n_rows), max_useful_workers))

    chunks_per_worker = max(1, int(CONFIG.get("prefit_parallel_chunks_per_worker", 4)))
    chunk_count = max(1, min(max(1, n_rows), workers * chunks_per_worker))
    chunk_size = max(1, int(math.ceil(float(max(1, n_rows)) / float(chunk_count))))

    return {
        "total_cores": int(total_cores),
        "workers": int(workers),
        "min_rows_per_worker": int(min_rows_per_worker),
        "chunk_count": int(chunk_count),
        "chunk_size": int(chunk_size),
    }


def build_prefit_row_payloads(df: pd.DataFrame) -> List[Dict[str, Any]]:
    input_cols = [
        "row_index",
        TIME_COL_COLUMNS,
        TARGET_COLUMNS,
        "acid_soluble_%",
        "cyanide_soluble_%",
        "material_size_p80_in",
        "lixiviant_initial_fe_mg_l",
        "lixiviant_initial_ph",
        "lixiviant_initial_orp_mv",
        "grouped_acid_generating_sulfides",
        "grouped_carbonates",
        "column_height_m",
        "copper_oxides_equivalent",
        "copper_secondary_sulfides_equivalent",
        "copper_primary_sulfides_equivalent",
        "residual_cpy_%",
        CATALYST_ADDITION_COL,
        CATALYST_CUM_COL,
        LIXIVIANT_CUM_COL,
        CATALYST_START_DAY_COL,
        TRANSITION_TIME_COL,
        FEED_MASS_COL,
        "column_inner_diameter_m",
    ]
    for optional_col in [COL_ID_COL, PAIR_ID_COL, STATUS_COL_PRIMARY, STATUS_COL_FALLBACK]:
        if optional_col in df.columns and optional_col not in input_cols:
            input_cols.append(optional_col)
    # Only keep columns that exist in the dataframe
    input_cols = [c for c in input_cols if c in df.columns]
    payloads = df.loc[:, input_cols].to_dict(orient="records")
    optional_defaults = compute_optional_static_predictor_defaults(df)
    for payload in payloads:
        for col, default_value in optional_defaults.items():
            if col in payload and not np.isfinite(scalar_from_maybe_array(payload.get(col, np.nan))):
                payload[col] = float(default_value)
    return payloads


def prefit_biexponential_for_row_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    status = normalize_status(row.get(STATUS_COL_PRIMARY, row.get(STATUS_COL_FALLBACK, "")))
    t_raw = parse_listlike(row.get(TIME_COL_COLUMNS, np.nan))
    y_raw = parse_listlike(row.get(TARGET_COLUMNS, np.nan))
    catalyst_cum_raw = parse_listlike(row.get(CATALYST_CUM_COL, np.nan))
    lixiviant_cum_raw = parse_listlike(row.get(LIXIVIANT_CUM_COL, np.nan))
    t, y, catalyst_cum = prepare_curve_arrays(t_raw, y_raw, catalyst_cum_raw, status=status, min_points=6)
    catalyst_feed_conc_for_prefit = resolve_observed_profile_on_time_grid(
        source_time_days=t_raw,
        target_time_days=t,
        observed_profile=row.get(CATALYST_ADDITION_COL, np.nan),
        clip_min=0.0,
    )
    fit_start_day = 0.0
    fit_start_day_source = "test_start"
    fit_ignored_initial_point_count = 0
    if status == "Catalyzed" and t.size > 0:
        resolved_start, resolved_source = resolve_curve_internal_catalyst_effect_start_day(
            row=row,
            status=status,
            time_days=t,
            recovery=y,
            catalyst_cum_kg_t=catalyst_cum,
            dosage_mg_l=(
                np.asarray(catalyst_feed_conc_for_prefit, dtype=float)
                if catalyst_feed_conc_for_prefit is not None
                else np.asarray([], dtype=float)
            ),
            dosage_source=(
                CATALYST_ADDITION_COL
                if catalyst_feed_conc_for_prefit is not None
                else CATALYST_ADDITION_RECON_COL
            ),
        )
        if np.isfinite(resolved_start):
            fit_start_day = float(resolved_start)
            fit_start_day_source = str(resolved_source)
            post_start_mask = t >= fit_start_day - 1e-9
            fit_ignored_initial_point_count = int(np.sum(~post_start_mask))
            t = t[post_start_mask]
            y = y[post_start_mask]
            catalyst_cum = catalyst_cum[post_start_mask]
        else:
            fit_start_day = np.nan
            fit_start_day_source = "missing"
    p80_in = scalar_from_maybe_array(row.get("material_size_p80_in", np.nan))

    cu_oxides_equiv = scalar_from_maybe_array(row.get("copper_oxides_equivalent", np.nan))
    cu_secondary_equiv = scalar_from_maybe_array(row.get("copper_secondary_sulfides_equivalent", np.nan))
    cu_primary_equiv = scalar_from_maybe_array(row.get("copper_primary_sulfides_equivalent", np.nan))
    residual_cpy_pct = scalar_from_maybe_array(row.get("residual_cpy_%", np.nan))
    acid_soluble_pct = scalar_from_maybe_array(row.get("acid_soluble_%", np.nan))
    cyanide_soluble_pct = scalar_from_maybe_array(row.get("cyanide_soluble_%", np.nan))
    lixiviant_initial_fe_mg_l = scalar_from_maybe_array(row.get("lixiviant_initial_fe_mg_l", np.nan))
    lixiviant_initial_ph = scalar_from_maybe_array(row.get("lixiviant_initial_ph", np.nan))
    lixiviant_initial_orp_mv = scalar_from_maybe_array(row.get("lixiviant_initial_orp_mv", np.nan))
    grouped_acid_generating_sulfides_pct = scalar_from_maybe_array(
        row.get("grouped_acid_generating_sulfides", np.nan)
    )
    grouped_carbonates_pct = scalar_from_maybe_array(row.get("grouped_carbonates", np.nan))
    column_height_m = scalar_from_maybe_array(row.get("column_height_m", np.nan))
    terminal_slope_rate = compute_terminal_slope_rate(t, y)
    curve_duration_days = float(t[-1] - t[0]) if t.size > 1 else 0.0
    final_observed_recovery = float(y[-1]) if y.size > 0 else np.nan
    duplicate_target_active = bool(row.get("prefit_duplicate_target_active", False))
    duplicate_target_recovery = scalar_from_maybe_array(row.get("prefit_duplicate_target_recovery", np.nan))
    duplicate_target_mode = str(row.get("prefit_duplicate_target_mode", "exact")).strip().lower() or "exact"
    duplicate_target_weight = scalar_from_maybe_array(row.get("prefit_duplicate_target_weight", 0.0))
    if not duplicate_target_active or not np.isfinite(duplicate_target_recovery):
        duplicate_target_recovery = np.nan
        duplicate_target_weight = 0.0
    duplicate_target_day = scalar_from_maybe_array(
        row.get("prefit_duplicate_target_day", CONFIG.get("prefit_asymptote_target_day", 2500.0))
    )
    final_catalyst_cum_kg_t = _prefit_final_profile_value(row.get(CATALYST_CUM_COL, np.nan))
    _prefit_ctrl_cap_raw, _prefit_cat_cap_raw = compute_chemistry_only_leach_caps(
        cu_oxides_equiv=cu_oxides_equiv,
        cu_secondary_equiv=cu_secondary_equiv,
        cu_primary_equiv=cu_primary_equiv,
        residual_cpy_pct=residual_cpy_pct,
        material_size_p80_in=p80_in,
        column_height_m=column_height_m,
        lixiviant_initial_fe_mg_l=lixiviant_initial_fe_mg_l,
        lixiviant_initial_ph=lixiviant_initial_ph,
        lixiviant_initial_orp_mv=lixiviant_initial_orp_mv,
        cyanide_soluble_pct=cyanide_soluble_pct,
        acid_soluble_pct=acid_soluble_pct,
        grouped_acid_generating_sulfides_pct=grouped_acid_generating_sulfides_pct,
        grouped_carbonates_pct=grouped_carbonates_pct,
    )
    _prefit_p80_cap_factor = compute_material_size_p80_cap_penalty(p80_in)
    _prefit_ctrl_cap, _prefit_cat_cap = compute_sample_leach_caps(
        cu_oxides_equiv=cu_oxides_equiv,
        cu_secondary_equiv=cu_secondary_equiv,
        cu_primary_equiv=cu_primary_equiv,
        material_size_p80_in=p80_in,
        residual_cpy_pct=residual_cpy_pct,
        column_height_m=column_height_m,
        lixiviant_initial_fe_mg_l=lixiviant_initial_fe_mg_l,
        lixiviant_initial_ph=lixiviant_initial_ph,
        lixiviant_initial_orp_mv=lixiviant_initial_orp_mv,
        cyanide_soluble_pct=cyanide_soluble_pct,
        acid_soluble_pct=acid_soluble_pct,
        grouped_acid_generating_sulfides_pct=grouped_acid_generating_sulfides_pct,
        grouped_carbonates_pct=grouped_carbonates_pct,
    )
    _prefit_ctrl_metallurgy_cap = float(_prefit_ctrl_cap)
    _prefit_cat_metallurgy_cap = float(_prefit_cat_cap)

    # -----------------------------------------------------------------------
    # Dose-saturation cap scaling (catalyzed only)
    # -----------------------------------------------------------------------
    # Chemistry rationale: the catalyst amplifies primary-sulfide leaching, but
    # the marginal benefit of additional catalyst dose follows a Michaelis-Menten
    # saturation curve — large gains between 50–100 mg/L, diminishing returns
    # above 150 mg/L.  Without dose-aware caps, all catalyzed columns receive the
    # same maximum-enhancement cap regardless of whether they were dosed at
    # 40 mg/L or 200 mg/L, which causes severe over-prediction for low-dose tests
    # (e.g. 022 stingray at 41 mg/L, 024 cpy at 90 mg/L) and insufficient
    # differentiation between dose tiers (e.g. 014 bag 64 vs 138 mg/L).
    #
    # Adjusted cap:
    #   cat_cap(dose) = ctrl_cap + dose_frac * (cat_cap_full - ctrl_cap)
    #   dose_frac     = dose_mg_l / (dose_mg_l + half_sat_dose_mg_l)
    #
    # At dose == half_sat:  dose_frac = 0.50  (halfway to full enhancement)
    # At dose →  ∞:         dose_frac → 1.00  (full chemistry cap)
    # At dose == 0:          dose_frac = 0.00  (reverts to control cap)
    # -----------------------------------------------------------------------
    if status == "Catalyzed" and bool(CONFIG.get("prefit_cat_cap_dose_saturation_enabled", True)):
        _cat_start_for_dose = (
            float(fit_start_day)
            if np.isfinite(fit_start_day)
            else scalar_from_maybe_array(row.get(CATALYST_START_DAY_COL, np.nan))
        )
        _avg_dose_for_cap = compute_average_catalyst_dose_mg_l(
            status=status,
            time_days=t_raw,
            catalyst_start_day=(
                float(_cat_start_for_dose) if np.isfinite(_cat_start_for_dose) else 0.0
            ),
            catalyst_cum_kg_t=catalyst_cum_raw,
            lixiviant_cum_m3_t=lixiviant_cum_raw,
            catalyst_feed_conc_mg_l=resolve_observed_profile_on_time_grid(
                source_time_days=t_raw,
                target_time_days=t_raw,
                observed_profile=row.get(CATALYST_ADDITION_COL, np.nan),
                clip_min=0.0,
            ),
        )
        _half_sat = float(CONFIG.get("prefit_cat_cap_dose_half_sat_mg_l", 80.0))
        if (
            np.isfinite(_avg_dose_for_cap)
            and _avg_dose_for_cap > 0.0
            and _half_sat > 0.0
            and np.isfinite(_prefit_cat_metallurgy_cap)
            and np.isfinite(_prefit_ctrl_metallurgy_cap)
        ):
            _dose_frac = _avg_dose_for_cap / (_avg_dose_for_cap + _half_sat)
            _ctrl_base = max(0.0, _prefit_ctrl_metallurgy_cap)
            _prefit_cat_metallurgy_cap = float(
                np.clip(
                    _ctrl_base + _dose_frac * (_prefit_cat_metallurgy_cap - _ctrl_base),
                    _ctrl_base,
                    _prefit_cat_metallurgy_cap,
                )
            )

    _prefit_ctrl_data_anchor = resolve_data_informed_cap_anchor_pct(
        chemistry_cap=_prefit_ctrl_metallurgy_cap,
        final_observed_recovery=final_observed_recovery,
        terminal_slope_rate=terminal_slope_rate,
        duration_days=curve_duration_days,
        cu_oxides_equiv=cu_oxides_equiv,
        cu_secondary_equiv=cu_secondary_equiv,
        cu_primary_equiv=cu_primary_equiv,
    )
    _prefit_cat_data_anchor = resolve_data_informed_cap_anchor_pct(
        chemistry_cap=_prefit_cat_metallurgy_cap,
        final_observed_recovery=final_observed_recovery,
        terminal_slope_rate=terminal_slope_rate,
        duration_days=curve_duration_days,
        cu_oxides_equiv=cu_oxides_equiv,
        cu_secondary_equiv=cu_secondary_equiv,
        cu_primary_equiv=cu_primary_equiv,
    )
    if bool(_prefit_ctrl_data_anchor.get("active", False)):
        _prefit_ctrl_cap = float(_prefit_ctrl_data_anchor["cap"])
        _prefit_ctrl_anchor_active = True
    else:
        _prefit_ctrl_cap, _prefit_ctrl_anchor_active = resolve_flat_tail_cap_anchor_pct(
            chemistry_cap=_prefit_ctrl_metallurgy_cap,
            final_observed_recovery=final_observed_recovery,
            terminal_slope_rate=terminal_slope_rate,
            duration_days=curve_duration_days,
        )
    if bool(_prefit_cat_data_anchor.get("active", False)):
        _prefit_cat_cap = float(_prefit_cat_data_anchor["cap"])
        _prefit_cat_anchor_active = True
    else:
        _prefit_cat_cap, _prefit_cat_anchor_active = resolve_flat_tail_cap_anchor_pct(
            chemistry_cap=_prefit_cat_metallurgy_cap,
            final_observed_recovery=final_observed_recovery,
            terminal_slope_rate=terminal_slope_rate,
            duration_days=curve_duration_days,
        )
    _prefit_cap_raw = _prefit_cat_cap_raw if status == "Catalyzed" else _prefit_ctrl_cap_raw
    _prefit_cap = _prefit_cat_cap if status == "Catalyzed" else _prefit_ctrl_cap
    _prefit_metallurgy_cap = (
        _prefit_cat_metallurgy_cap if status == "Catalyzed" else _prefit_ctrl_metallurgy_cap
    )
    _prefit_data_anchor = _prefit_cat_data_anchor if status == "Catalyzed" else _prefit_ctrl_data_anchor
    _prefit_applied_cap = max(2.0, float(_prefit_cap)) if np.isfinite(_prefit_cap) else np.nan
    _use_sigmoid_gate = should_use_sigmoid_gated_prefit(
        time_days=t,
        recovery_pct=y,
        material_size_p80_in=p80_in,
        column_height_m=column_height_m,
    )
    _fit_curve_mode = "sigmoid_gated_biexponential" if _use_sigmoid_gate else "biexponential"
    _fit_gate_mid_day = np.nan
    _fit_gate_width_day = np.nan

    # Per-status prefit tuning: catalyzed columns receive a stronger day-2500 slope
    # penalty (the catalyst should drive most leachable chalcopyrite into solution
    # well before that horizon) and a modest kinetic-rate boost (the catalyst
    # accelerates sulphide dissolution, so steeper early kinetics are physical).
    # NOTE: the stronger penalty does NOT force the asymptote to the chemistry cap —
    # it only enforces near-flatness at day 2500 regardless of the recovery level.
    _is_catalyzed_status = (status == "Catalyzed")
    _cat_penalty_weight: Optional[float] = (
        float(CONFIG.get("prefit_cat_target_day_penalty_weight",
                         CONFIG.get("prefit_target_day_penalty_weight", 1.0)))
        if _is_catalyzed_status else None
    )
    _cat_rate_boost: float = (
        float(np.clip(CONFIG.get("prefit_cat_p80_rate_upper_boost", 1.0), 1.0, 3.0))
        if _is_catalyzed_status else 1.0
    )
    _cat_cap_upper_bound: bool = (
        bool(CONFIG.get("prefit_cat_cap_as_upper_bound_only", True))
        if _is_catalyzed_status else False
    )
    _cat_disable_shortfall: bool = (
        bool(CONFIG.get("prefit_cat_disable_shortfall_penalty", True))
        if _is_catalyzed_status else False
    )

    # Match v10 pre-fit logic: use the per-row control/catalyzed cap derived
    # from the configured leach_pct_* mineralogy fractions and P80 penalty.
    if t.size >= 6:
        base_params = fit_biexponential_params(
            t,
            y,
            cap=_prefit_cap,
            target_day_recovery=duplicate_target_recovery,
            target_day_recovery_weight=duplicate_target_weight,
            target_day_recovery_mode=duplicate_target_mode,
            material_size_p80_in=float(p80_in) if np.isfinite(p80_in) else float("nan"),
            target_day_penalty_weight_override=_cat_penalty_weight,
            rate_upper_boost=_cat_rate_boost,
            cap_as_upper_bound_only=_cat_cap_upper_bound,
            disable_shortfall_penalty=_cat_disable_shortfall,
        )
    else:
        base_params = np.array([np.nan] * 4, dtype=float)
    params = np.asarray(base_params, dtype=float)
    if t.size >= 6 and _use_sigmoid_gate:
        gate_params, gate_mid_day, gate_width_day = fit_sigmoid_gated_biexponential_params(
            t,
            y,
            cap=_prefit_cap,
            target_day_recovery=duplicate_target_recovery,
            target_day_recovery_weight=duplicate_target_weight,
            target_day_recovery_mode=duplicate_target_mode,
            material_size_p80_in=float(p80_in) if np.isfinite(p80_in) else float("nan"),
            cu_oxides_equiv=float(cu_oxides_equiv) if np.isfinite(cu_oxides_equiv) else float("nan"),
            cu_secondary_equiv=float(cu_secondary_equiv) if np.isfinite(cu_secondary_equiv) else float("nan"),
            cu_primary_equiv=float(cu_primary_equiv) if np.isfinite(cu_primary_equiv) else float("nan"),
            column_height_m=float(column_height_m) if np.isfinite(column_height_m) else float("nan"),
            grouped_carbonates_pct=float(grouped_carbonates_pct) if np.isfinite(grouped_carbonates_pct) else float("nan"),
            grouped_acid_generating_sulfides_pct=float(grouped_acid_generating_sulfides_pct) if np.isfinite(grouped_acid_generating_sulfides_pct) else float("nan"),
            target_day_penalty_weight_override=_cat_penalty_weight,
            rate_upper_boost=_cat_rate_boost,
            cap_as_upper_bound_only=_cat_cap_upper_bound,
            disable_shortfall_penalty=_cat_disable_shortfall,
        )
        if np.all(np.isfinite(gate_params)):
            base_pred = prefit_curve_prediction_np(t, base_params, "biexponential")
            gate_pred = prefit_curve_prediction_np(
                t,
                gate_params,
                "sigmoid_gated_biexponential",
                gate_mid_day,
                gate_width_day,
            )
            base_rmse = float(np.sqrt(np.mean((base_pred - y) ** 2))) if np.all(np.isfinite(base_pred)) else np.inf
            gate_rmse = float(np.sqrt(np.mean((gate_pred - y) ** 2))) if np.all(np.isfinite(gate_pred)) else np.inf
            improvement_fraction = float(CONFIG.get("prefit_sigmoid_gate_min_improvement_fraction", 0.03))
            max_regression_fraction = max(
                0.0,
                float(CONFIG.get("prefit_sigmoid_gate_max_rmse_regression_fraction", 0.02)),
            )
            if (
                gate_rmse <= base_rmse * (1.0 - improvement_fraction)
                or gate_rmse <= base_rmse * (1.0 + max_regression_fraction)
            ):
                params = np.asarray(gate_params, dtype=float)
                _fit_curve_mode = "sigmoid_gated_biexponential"
                _fit_gate_mid_day = float(gate_mid_day)
                _fit_gate_width_day = float(gate_width_day)
            else:
                _fit_curve_mode = "biexponential"
    if not np.all(np.isfinite(params)):
        _fit_curve_mode = "biexponential"
        _fit_gate_mid_day = np.nan
        _fit_gate_width_day = np.nan
    fit_asymptote = float(params[0] + params[2]) if np.all(np.isfinite(params[[0, 2]])) else np.nan
    target_day = float(CONFIG.get("prefit_asymptote_target_day", 2500.0))
    if np.all(np.isfinite(params)) and np.isfinite(fit_asymptote) and fit_asymptote > 1e-9:
        fit_target_recovery = float(
            prefit_curve_prediction_np(
                np.asarray([target_day]),
                params,
                _fit_curve_mode,
                _fit_gate_mid_day,
                _fit_gate_width_day,
            )[0]
        )
        fit_y2500_asymptote_frac = fit_target_recovery / fit_asymptote
        fit_slope2500_pct_per_day = float(
            prefit_curve_slope_np(
                np.asarray([target_day]),
                params,
                _fit_curve_mode,
                _fit_gate_mid_day,
                _fit_gate_width_day,
            )[0]
        )
    else:
        fit_y2500_asymptote_frac = np.nan
        fit_slope2500_pct_per_day = np.nan
    if np.all(np.isfinite(params)) and t.size > 0:
        pred = prefit_curve_prediction_np(t, params, _fit_curve_mode, _fit_gate_mid_day, _fit_gate_width_day)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    else:
        rmse = np.nan
    return {
        "row_index": int(row.get("row_index", 0)),
        # v11: include project_col_id and project_sample_id for traceability
        "col_id": str(row.get(COL_ID_COL, "")),
        "sample_id": str(row.get(PAIR_ID_COL, "")),
        "status_norm": status,
        "prefit_logic_version": str(MODEL_LOGIC_VERSION),
        "fit_a1": float(params[0]) if np.isfinite(params[0]) else np.nan,
        "fit_b1": float(params[1]) if np.isfinite(params[1]) else np.nan,
        "fit_a2": float(params[2]) if np.isfinite(params[2]) else np.nan,
        "fit_b2": float(params[3]) if np.isfinite(params[3]) else np.nan,
        "fit_curve_mode": str(_fit_curve_mode),
        "fit_gate_active": bool(_fit_curve_mode == "sigmoid_gated_biexponential"),
        "fit_gate_mid_day": float(_fit_gate_mid_day) if np.isfinite(_fit_gate_mid_day) else np.nan,
        "fit_gate_width_day": float(_fit_gate_width_day) if np.isfinite(_fit_gate_width_day) else np.nan,
        "fit_asymptote": fit_asymptote,
        "fit_p80_cap_factor": float(_prefit_p80_cap_factor),
        "fit_ctrl_cap_raw": float(_prefit_ctrl_cap_raw),
        "fit_cat_cap_raw": float(_prefit_cat_cap_raw),
        "fit_ctrl_cap": float(_prefit_ctrl_cap),
        "fit_cat_cap": float(_prefit_cat_cap),
        "fit_sample_cap": float(_prefit_applied_cap),
        "fit_raw_sample_cap": float(_prefit_cap_raw),
        "fit_metallurgy_sample_cap": float(_prefit_metallurgy_cap),
        "fit_data_informed_cap": (
            float(_prefit_data_anchor["cap"])
            if bool(_prefit_data_anchor.get("active", False)) and np.isfinite(_prefit_data_anchor.get("cap", np.nan))
            else np.nan
        ),
        "fit_data_informed_cap_active": bool(_prefit_data_anchor.get("active", False)),
        "fit_final_secondary_fraction": float(_prefit_data_anchor.get("inferred_secondary_fraction", np.nan)),
        "fit_final_primary_fraction": float(_prefit_data_anchor.get("inferred_primary_fraction", np.nan)),
        "fit_prefit_rebase_active": False,
        "fit_prefit_rebase_time_offset_day": 0.0,
        "fit_prefit_rebase_recovery_offset_pct": 0.0,
        "fit_prefit_ignored_initial_point_count": int(fit_ignored_initial_point_count),
        "fit_prefit_fit_point_count": int(t.size),
        "fit_prefit_fit_start_day": float(fit_start_day) if np.isfinite(fit_start_day) else np.nan,
        "fit_prefit_fit_start_day_source": str(fit_start_day_source),
        "fit_target_day": float(target_day),
        "fit_target_day_asymptote_frac": float(fit_y2500_asymptote_frac),
        "fit_target_day_slope_pct_per_day": float(fit_slope2500_pct_per_day),
        "fit_y2500_asymptote_frac": float(fit_y2500_asymptote_frac),
        "fit_slope2500_pct_per_day": float(fit_slope2500_pct_per_day),
        "fit_duplicate_target_active": bool(duplicate_target_active),
        "fit_duplicate_target_day": float(duplicate_target_day),
        "fit_duplicate_target_recovery": float(duplicate_target_recovery),
        "fit_duplicate_target_mode": str(duplicate_target_mode),
        "fit_final_catalyst_cum_kg_t": float(final_catalyst_cum_kg_t),
        "fit_rmse": rmse,
    }


def prefit_biexponential_for_row_chunk(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [prefit_biexponential_for_row_payload(row) for row in rows]


def _normalize_prefit_cache_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, np.ndarray)):
        return json.dumps(np.asarray(value, dtype=object).tolist(), default=str, separators=(",", ":"))
    try:
        if pd.isna(value):
            return "<NA>"
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).strip()


def _prefit_cache_compare_columns(df: pd.DataFrame) -> List[str]:
    compare_cols = [col for col in PREFIT_CACHE_COMPARE_COLUMNS if col in df.columns]
    for status_col in [STATUS_COL_PRIMARY, STATUS_COL_FALLBACK]:
        if status_col in df.columns:
            compare_cols.append(status_col)
    return compare_cols


def inspect_prefit_cache(
    df_current: pd.DataFrame,
    prefit_out_path: str,
) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    if not os.path.exists(prefit_out_path):
        return False, "prefit file not found", None

    try:
        cached_df = pd.read_csv(prefit_out_path, sep=",")
    except Exception as exc:
        return False, f"could not read cached prefit: {exc}", None

    compare_cols = _prefit_cache_compare_columns(df_current)
    # Validate only the actual fit-result columns (PREFIT_FIT_COLUMNS) plus the
    # fingerprint columns used for cache-match checking.  Metadata columns such
    # as col_id, sample_id and status_norm are intentionally excluded from this
    # check because (a) they already exist in the main dataframe and are not
    # needed from the cache, and (b) older prefit CSVs may not contain them,
    # which would otherwise cause a spurious full recompute every run.
    required_cols = PREFIT_FIT_COLUMNS + compare_cols
    missing_cols = [col for col in required_cols if col not in cached_df.columns]
    if missing_cols:
        return False, f"cached prefit is missing columns: {', '.join(missing_cols)}", None
    if "prefit_logic_version" in cached_df.columns:
        cached_versions = cached_df["prefit_logic_version"].astype(str).str.strip().unique().tolist()
        if cached_versions != [str(MODEL_LOGIC_VERSION)]:
            return (
                False,
                f"cached prefit logic version mismatch (cached={cached_versions}, current={MODEL_LOGIC_VERSION})",
                None,
            )

    if len(cached_df) != len(df_current):
        return (
            False,
            f"cached prefit row count mismatch (cached={len(cached_df)}, current={len(df_current)})",
            None,
        )

    cached_row_index = pd.to_numeric(cached_df["row_index"], errors="coerce").to_numpy()
    expected_row_index = np.arange(len(df_current), dtype=float)
    if not np.array_equal(cached_row_index, expected_row_index):
        return False, "cached prefit row_index does not match the current dataset order", None

    for col in compare_cols:
        current_values = df_current[col].map(_normalize_prefit_cache_value).to_numpy(dtype=object)
        cached_values = cached_df[col].map(_normalize_prefit_cache_value).to_numpy(dtype=object)
        if not np.array_equal(current_values, cached_values):
            return False, f"cached prefit does not match current dataset column '{col}'", None

    # Return only the actual fit-result columns.  Metadata columns (col_id,
    # sample_id, status_norm) are already present in df_prefit from the main
    # dataframe and do not need to come from the cache.  Returning them would
    # also cause duplicate-column conflicts in the subsequent left-merge.
    available_fit_cols = [c for c in PREFIT_FIT_COLUMNS if c in cached_df.columns]
    return True, "cached prefit matches the current dataset", cached_df[available_fit_cols].copy()


def prompt_yes_no(question: str, default: bool = False) -> bool:
    prompt_suffix = "[Y/n]" if default else "[y/N]"
    default_label = "yes" if default else "no"

    if not sys.stdin or not sys.stdin.isatty():
        print(f"{question} {prompt_suffix} -> non-interactive session, defaulting to {default_label}.")
        return default

    while True:
        try:
            answer = input(f"{question} {prompt_suffix}: ").strip().lower()
        except EOFError:
            print(f"{question} {prompt_suffix} -> input unavailable, defaulting to {default_label}.")
            return default

        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def prepare_prefit_dataframe(df: pd.DataFrame, prefit_out_path: str) -> pd.DataFrame:
    df_prefit = _filtered_copy_without_configured_excluded_ids(df).reset_index(drop=True)
    df_prefit = append_reconstructed_sequence_columns(df_prefit)
    assert_dataframe_respects_training_exclusions(df_prefit, context="prepare_prefit_dataframe input")
    df_prefit["row_index"] = np.arange(len(df_prefit))

    cache_is_valid, cache_message, cached_prefit = inspect_prefit_cache(df_prefit, prefit_out_path)
    if cache_is_valid and cached_prefit is not None:
        print(f"[Prefit] Found reusable pre-fit data at: {prefit_out_path}")
        rerun_prefit = prompt_yes_no("[Prefit] Recompute the biexponential pre-fit?", default=False)
        if not rerun_prefit:
            print("[Prefit] Reusing cached pre-fit results.")
            df_prefit = df_prefit.merge(cached_prefit, on="row_index", how="left")
            write_prefit_cap_diagnostics(df_prefit, prefit_out_path)
            return df_prefit
        print("[Prefit] Recomputing pre-fit by user request.")
    else:
        print(f"[Prefit] Cached pre-fit unavailable: {cache_message}. Recomputing.")

    prefit_df = prefit_biexponential_for_rows(df_prefit)
    df_prefit = df_prefit.merge(prefit_df, on="row_index", how="left")
    df_prefit.to_csv(prefit_out_path, index=False)
    print(f"[Prefit] Saved pre-fit table to: {prefit_out_path}")
    write_prefit_cap_diagnostics(df_prefit, prefit_out_path)
    return df_prefit


def _prefit_final_observed_recovery_from_row(row: pd.Series) -> float:
    status = normalize_status(row.get(STATUS_COL_PRIMARY, row.get(STATUS_COL_FALLBACK, "")))
    t_raw = parse_listlike(row.get(TIME_COL_COLUMNS, np.nan))
    y_raw = parse_listlike(row.get(TARGET_COLUMNS, np.nan))
    t, y, _ = prepare_curve_arrays(t_raw, y_raw, None, status=status, min_points=1)
    if y.size == 0:
        return np.nan
    return float(y[-1])


def _summary_quantiles(values: pd.Series) -> Dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"p10": np.nan, "p50": np.nan, "p90": np.nan}
    return {
        "p10": float(np.nanquantile(arr, 0.10)),
        "p50": float(np.nanquantile(arr, 0.50)),
        "p90": float(np.nanquantile(arr, 0.90)),
    }


def build_prefit_cap_diagnostics(df_prefit: pd.DataFrame) -> Dict[str, Any]:
    if "fit_data_informed_cap_active" not in df_prefit.columns:
        return {"available": False}

    active = df_prefit["fit_data_informed_cap_active"].fillna(False).astype(bool)
    duplicate_active = (
        df_prefit["fit_duplicate_target_active"].fillna(False).astype(bool)
        if "fit_duplicate_target_active" in df_prefit.columns
        else pd.Series([False] * len(df_prefit), index=df_prefit.index)
    )
    rebase_active = (
        df_prefit["fit_prefit_rebase_active"].fillna(False).astype(bool)
        if "fit_prefit_rebase_active" in df_prefit.columns
        else pd.Series([False] * len(df_prefit), index=df_prefit.index)
    )
    gate_active = (
        df_prefit["fit_gate_active"].fillna(False).astype(bool)
        if "fit_gate_active" in df_prefit.columns
        else pd.Series([False] * len(df_prefit), index=df_prefit.index)
    )
    old_cap = pd.to_numeric(df_prefit.get("fit_metallurgy_sample_cap", np.nan), errors="coerce")
    new_cap = pd.to_numeric(df_prefit.get("fit_sample_cap", np.nan), errors="coerce")
    delta_cap = new_cap - old_cap
    final_recovery = df_prefit.apply(_prefit_final_observed_recovery_from_row, axis=1)
    data_cap = pd.to_numeric(df_prefit.get("fit_data_informed_cap", np.nan), errors="coerce")
    below_final = active & np.isfinite(data_cap) & np.isfinite(final_recovery) & (data_cap < final_recovery - 1e-9)

    by_status: Dict[str, Any] = {}
    status_values = (
        df_prefit["status_norm"].astype(str)
        if "status_norm" in df_prefit.columns
        else df_prefit.get(STATUS_COL_PRIMARY, pd.Series([""] * len(df_prefit))).astype(str)
    )
    for status in sorted(status_values.unique().tolist()):
        mask = status_values == status
        by_status[str(status)] = {
            "rows": int(mask.sum()),
            "data_informed_active": int((active & mask).sum()),
            "duplicate_day2500_target_active": int((duplicate_active & mask).sum()),
            "virtual_rebase_active": int((rebase_active & mask).sum()),
            "sigmoid_gate_active": int((gate_active & mask).sum()),
            "old_cap_quantiles": _summary_quantiles(old_cap.loc[mask]),
            "new_cap_quantiles": _summary_quantiles(new_cap.loc[mask]),
            "active_cap_delta_quantiles": _summary_quantiles(delta_cap.loc[mask & active]),
        }

    flagged_cols = []
    if below_final.any():
        source = COL_ID_COL if COL_ID_COL in df_prefit.columns else "col_id"
        if source in df_prefit.columns:
            flagged_cols = df_prefit.loc[below_final, source].astype(str).head(20).tolist()

    return {
        "available": True,
        "rows": int(len(df_prefit)),
        "data_informed_active": int(active.sum()),
        "duplicate_day2500_target_active": int(duplicate_active.sum()),
        "virtual_rebase_active": int(rebase_active.sum()),
        "sigmoid_gate_active": int(gate_active.sum()),
        "cap_below_final_recovery_count": int(below_final.sum()),
        "cap_below_final_recovery_col_ids": flagged_cols,
        "old_cap_quantiles": _summary_quantiles(old_cap),
        "new_cap_quantiles": _summary_quantiles(new_cap),
        "active_cap_delta_quantiles": _summary_quantiles(delta_cap.loc[active]),
        "by_status": by_status,
    }


def write_prefit_cap_diagnostics(df_prefit: pd.DataFrame, prefit_out_path: str) -> None:
    summary = build_prefit_cap_diagnostics(df_prefit)
    out_path = os.path.join(os.path.dirname(prefit_out_path), "prefit_cap_diagnostics.json")
    save_json(out_path, summary)
    if not summary.get("available", False):
        print("[Prefit] Cap diagnostics unavailable for this prefit table.")
        return
    print(
        "[Prefit] Data-informed cap anchors: "
        f"{summary['data_informed_active']} / {summary['rows']} rows active | "
        f"duplicate_day2500_targets={summary.get('duplicate_day2500_target_active', 0)} | "
        f"virtual_rebase={summary.get('virtual_rebase_active', 0)} | "
        f"sigmoid_gate={summary.get('sigmoid_gate_active', 0)} | "
        f"below_final_flags={summary['cap_below_final_recovery_count']}"
    )
    print(f"[Prefit] Saved cap diagnostics to: {out_path}")


def compute_terminal_slope_rate(
    time_days: np.ndarray,
    recovery_pct: np.ndarray,
    tail_frac: float = 0.20,
) -> float:
    """Mean daily Cu recovery rate (%/day) over the last ``tail_frac`` fraction
    of the test duration.

    Metallurgical rationale (v12): if the curve has essentially flatlined by
    the end of the test, this value will be near zero, giving the NN a direct
    empirical signal that the leach cap is already close to the current
    observed recovery — preventing extrapolation to unrealistically high
    asymptotes for refractory primary-sulfide ores (e.g. elephant_site, pvo).

    Returns a value clipped to [-0.5, 2.0] %/day to avoid outlier influence.
    Returns 0.0 when the time-series is too short or degenerate.
    """
    t = np.asarray(time_days, dtype=float)
    y = np.asarray(recovery_pct, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]
    if len(t) < 4 or t[-1] <= t[0]:
        return 0.0
    duration = float(t[-1] - t[0])
    window_start = t[-1] - max(tail_frac * duration, 7.0)  # at least 7 days
    mask = t >= window_start
    if int(mask.sum()) < 2:
        return 0.0
    t_win, y_win = t[mask], y[mask]
    delta_t = float(t_win[-1] - t_win[0])
    if delta_t <= 0.0:
        return 0.0
    rate = float(y_win[-1] - y_win[0]) / delta_t
    return float(np.clip(rate, -0.5, 2.0))


def resolve_static_feature_value_from_vector(
    feature_vector: np.ndarray,
    columns: List[str],
    column_name: str,
    default: float = np.nan,
) -> float:
    idx = columns.index(column_name) if column_name in columns else -1
    if idx < 0 or idx >= len(feature_vector):
        return float(default)
    value = scalar_from_maybe_array(feature_vector[idx])
    return float(value) if np.isfinite(value) else float(default)


def build_curve_specific_static_override(static_vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(static_vec, dtype=float)
    return np.asarray(
        [
            resolve_static_feature_value_from_vector(
                vec,
                STATIC_PREDICTOR_COLUMNS,
                col,
                default=np.nan,
            )
            for col in CURVE_SPECIFIC_STATIC_OVERRIDE_COLUMNS
        ],
        dtype=float,
    )


def resolve_data_informed_cap_anchor_pct(
    *,
    chemistry_cap: float,
    final_observed_recovery: float,
    terminal_slope_rate: float,
    duration_days: float,
    cu_oxides_equiv: float,
    cu_secondary_equiv: float,
    cu_primary_equiv: float,
    max_terminal_slope_pct_per_day: Optional[float] = None,
    min_duration_days: Optional[float] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "cap": float(chemistry_cap) if np.isfinite(chemistry_cap) else np.nan,
        "active": False,
        "inferred_secondary_fraction": np.nan,
        "inferred_primary_fraction": np.nan,
        "inferred_primary_fraction_raw": np.nan,
        "tail_flatness": np.nan,
    }
    if not bool(CONFIG.get("use_data_informed_prefit_caps", True)):
        return out

    cap = float(chemistry_cap)
    final_recovery = float(final_observed_recovery)
    terminal_slope = float(terminal_slope_rate)
    duration = float(duration_days)
    if not np.isfinite(cap) or not np.isfinite(final_recovery):
        return out
    min_duration = (
        float(min_duration_days)
        if min_duration_days is not None and np.isfinite(min_duration_days)
        else 350.0
    )
    max_terminal_slope = (
        float(max_terminal_slope_pct_per_day)
        if max_terminal_slope_pct_per_day is not None and np.isfinite(max_terminal_slope_pct_per_day)
        else 0.020
    )
    if not np.isfinite(duration) or duration < min_duration:
        return out
    if not np.isfinite(terminal_slope) or terminal_slope > max_terminal_slope:
        return out

    ox = max(0.0, float(cu_oxides_equiv)) if np.isfinite(cu_oxides_equiv) else np.nan
    sec = max(0.0, float(cu_secondary_equiv)) if np.isfinite(cu_secondary_equiv) else np.nan
    pri = max(0.0, float(cu_primary_equiv)) if np.isfinite(cu_primary_equiv) else np.nan
    if not (np.isfinite(ox) and np.isfinite(sec) and np.isfinite(pri)) or pri <= 1e-9:
        return out

    cu = compute_total_copper_equivalent(
        cu_oxides_equiv=ox,
        cu_secondary_equiv=sec,
        cu_primary_equiv=pri,
    )
    if not np.isfinite(cu) or cu <= 1e-9:
        return out

    tail_flatness = float(np.clip((max_terminal_slope - terminal_slope) / 0.015, 0.0, 1.0))
    sec_min = float(CONFIG.get("final_day_secondary_leach_fraction_min", 0.55))
    sec_max = float(CONFIG.get("final_day_secondary_leach_fraction_max", 0.70))
    sec_lo = float(np.clip(min(sec_min, sec_max), 0.0, 1.0))
    sec_hi = float(np.clip(max(sec_min, sec_max), sec_lo, 1.0))
    inferred_secondary_fraction = sec_lo + (sec_hi - sec_lo) * tail_flatness
    oxide_fraction = float(np.clip(CONFIG.get("final_day_oxide_leach_fraction", 1.0), 0.0, 1.0))
    final_leached_cu = np.clip(final_recovery, 0.0, 100.0) / 100.0 * cu
    inferred_primary_raw = (
        final_leached_cu - oxide_fraction * ox - inferred_secondary_fraction * sec
    ) / max(pri, 1e-9)
    inferred_primary_fraction = float(np.clip(inferred_primary_raw, 0.0, 1.0))

    tail_margin_pct = float(np.clip(2.0 + 220.0 * terminal_slope, 2.0, 8.0))
    data_cap = min(cap, final_recovery + tail_margin_pct)
    data_cap = max(data_cap, final_recovery + 1.0)
    out.update(
        {
            "cap": float(np.clip(data_cap, 0.0, 100.0)),
            "active": True,
            "inferred_secondary_fraction": float(inferred_secondary_fraction),
            "inferred_primary_fraction": float(inferred_primary_fraction),
            "inferred_primary_fraction_raw": float(inferred_primary_raw),
            "tail_flatness": float(tail_flatness),
        }
    )
    return out


def resolve_flat_tail_cap_anchor_pct(
    chemistry_cap: float,
    final_observed_recovery: float,
    terminal_slope_rate: float,
    duration_days: float,
) -> Tuple[float, bool]:
    cap = float(chemistry_cap)
    final_recovery = float(final_observed_recovery)
    terminal_slope = float(terminal_slope_rate)
    duration = float(duration_days)
    if not np.isfinite(cap):
        return np.nan, False
    if not np.isfinite(final_recovery):
        return cap, False
    if not np.isfinite(duration) or duration < 350.0:
        return cap, False
    if not np.isfinite(terminal_slope) or terminal_slope > 0.020:
        return cap, False

    tail_margin_pct = float(np.clip(2.0 + 220.0 * terminal_slope, 2.0, 8.0))
    flatness = float(np.clip((0.020 - terminal_slope) / 0.015, 0.0, 1.0))
    anchor_target = min(cap, final_recovery + tail_margin_pct)
    anchored_cap = cap - flatness * (cap - anchor_target)
    anchored_cap = max(anchored_cap, final_recovery + 1.0)
    anchored_cap = float(np.clip(anchored_cap, 0.0, 100.0))
    return anchored_cap, True


def build_pair_samples(df: pd.DataFrame) -> List[PairSample]:
    df = _filtered_copy_without_configured_excluded_ids(df)
    assert_dataframe_respects_training_exclusions(df, context="build_pair_samples input")
    pairs: List[PairSample] = []
    grouped = df.groupby(PAIR_ID_COL, dropna=False)
    optional_static_defaults = compute_optional_static_predictor_defaults(df)
    orp_window_start_day = float(CONFIG.get("orp_aux_window_start_day", 150.0))
    orp_window_end_day = float(CONFIG.get("orp_aux_window_end_day", 400.0))
    orp_trim_quantile = float(CONFIG.get("orp_aux_trim_quantile", 0.10))
    orp_step_days = float(CONFIG.get("orp_aux_summary_step_days", 1.0))
    orp_history_window_days = float(CONFIG.get("orp_aux_recent_window_days", 21.0))
    orp_min_recent_points = int(CONFIG.get("orp_aux_min_recent_points", 3))
    orp_min_target_points = int(CONFIG.get("orp_aux_min_target_points", 5))
    # v11: each row in the new leaching_performance_weekly.csv is one project_col_id.
    # Pairing logic:
    #   - Group by project_sample_id (ore sample).
    #   - Within each group, catalyst_status determines Control vs. Catalyzed.
    #   - Multiple control and multiple catalyzed columns may exist per sample.
    #   - Every valid control curve is paired with every valid catalyzed curve
    #     from the same sample, so each column is treated independently in
    #     training while still grouped within one ore sample.
    #   - Caps (leach limits) are derived from per-sample-id mineralogy and apply
    #     to all pairs within that sample.
    for sample_id, group in grouped:
        if pd.isna(sample_id):
            continue

        # by_status maps "Control" / "Catalyzed" → all valid CurveData candidates + metadata.
        # Tuple: (CurveData, static_vec, input_only_vec, project_name,
        #          feed_orp_aux_target, pls_orp_aux_target)
        by_status: Dict[str, List[Tuple[CurveData, np.ndarray, np.ndarray, str, float, float]]] = {
            "Control": [],
            "Catalyzed": [],
        }

        for idx, row in group.iterrows():
            col_id = str(row.get(COL_ID_COL, "")).strip()
            # v11: use catalyst_status as primary, project_col_id as legacy fallback
            status = normalize_status(
                row.get(STATUS_COL_PRIMARY, row.get(STATUS_COL_FALLBACK, ""))
            )
            t_raw = parse_listlike(row.get(TIME_COL_COLUMNS, np.nan))
            y_raw = parse_listlike(row.get(TARGET_COLUMNS, np.nan))

            # v12: build catalyst signals from cumulative source columns plus
            # the explicit catalyst_addition_mg_l feed vector when available.
            # Signal 1 (catalyst_cum): authoritative cumulative catalyst load in kg/t.
            # Signal 2 (catalyst_conc_col_mg_l): CSTR column concentration in mg/L.
            dynamic_sequences = reconstruct_aligned_dynamic_sequences_for_row(row, min_points=6)
            if dynamic_sequences is None:
                continue

            t = np.asarray(dynamic_sequences["time_days_aligned"], dtype=float)
            y = np.asarray(dynamic_sequences["recovery_aligned"], dtype=float)
            c = np.asarray(dynamic_sequences["catalyst_cum_aligned_kg_t"], dtype=float)
            lix_aligned = np.asarray(dynamic_sequences["lixiviant_cum_aligned_m3_t"], dtype=float)
            irrigation_rate = np.asarray(dynamic_sequences[IRRIGATION_RATE_RECON_COL], dtype=float)
            catalyst_feed_conc_reconstructed = np.asarray(
                dynamic_sequences[CATALYST_ADDITION_RECON_COL],
                dtype=float,
            )
            catalyst_feed_conc_source = str(
                dynamic_sequences.get("catalyst_addition_mg_l_source", CATALYST_ADDITION_RECON_COL)
            )
            conc_aligned = np.asarray(
                dynamic_sequences["catalyst_conc_col_mg_l_reconstructed"],
                dtype=float,
            )
            _terminal_rate = compute_terminal_slope_rate(t, y)
            _curve_duration_days = float(t[-1] - t[0]) if t.size > 1 else 0.0
            _final_observed_recovery = float(y[-1]) if y.size > 0 else np.nan

            # Pre-fit parameters (per project_col_id: already in df via prepare_prefit_dataframe)
            fit_params = np.asarray(
                [
                    pd.to_numeric(row.get("fit_a1", np.nan), errors="coerce"),
                    pd.to_numeric(row.get("fit_b1", np.nan), errors="coerce"),
                    pd.to_numeric(row.get("fit_a2", np.nan), errors="coerce"),
                    pd.to_numeric(row.get("fit_b2", np.nan), errors="coerce"),
                ],
                dtype=float,
            )
            fit_curve_mode = str(row.get("fit_curve_mode", "biexponential")).strip().lower() or "biexponential"
            fit_gate_mid_day = pd.to_numeric(row.get("fit_gate_mid_day", np.nan), errors="coerce")
            fit_gate_width_day = pd.to_numeric(row.get("fit_gate_width_day", np.nan), errors="coerce")
            prefit_fit_start_day = pd.to_numeric(row.get("fit_prefit_fit_start_day", np.nan), errors="coerce")
            prefit_fit_start_day_source = str(row.get("fit_prefit_fit_start_day_source", "")).strip()
            if fit_curve_mode != "sigmoid_gated_biexponential":
                fit_curve_mode = "biexponential"
                fit_gate_mid_day = np.nan
                fit_gate_width_day = np.nan

            # Fallback refit when prefit CSV values are missing: match the
            # v10 pre-fit path by using the per-row control/catalyzed leach
            # cap derived from the leach_pct_* mineralogy fractions.
            if not np.all(np.isfinite(fit_params)):
                _row_ctrl_cap, _row_cat_cap = compute_sample_leach_caps(
                    cu_oxides_equiv=scalar_from_maybe_array(row.get("copper_oxides_equivalent", np.nan)),
                    cu_secondary_equiv=scalar_from_maybe_array(row.get("copper_secondary_sulfides_equivalent", np.nan)),
                    cu_primary_equiv=scalar_from_maybe_array(row.get("copper_primary_sulfides_equivalent", np.nan)),
                    material_size_p80_in=scalar_from_maybe_array(row.get("material_size_p80_in", np.nan)),
                    residual_cpy_pct=scalar_from_maybe_array(row.get("residual_cpy_%", np.nan)),
                    column_height_m=scalar_from_maybe_array(row.get("column_height_m", np.nan)),
                    lixiviant_initial_fe_mg_l=scalar_from_maybe_array(row.get("lixiviant_initial_fe_mg_l", np.nan)),
                    lixiviant_initial_ph=scalar_from_maybe_array(row.get("lixiviant_initial_ph", np.nan)),
                    lixiviant_initial_orp_mv=scalar_from_maybe_array(row.get("lixiviant_initial_orp_mv", np.nan)),
                    cyanide_soluble_pct=scalar_from_maybe_array(row.get("cyanide_soluble_%", np.nan)),
                    acid_soluble_pct=scalar_from_maybe_array(row.get("acid_soluble_%", np.nan)),
                    grouped_acid_generating_sulfides_pct=scalar_from_maybe_array(
                        row.get("grouped_acid_generating_sulfides", np.nan)
                    ),
                    grouped_carbonates_pct=scalar_from_maybe_array(row.get("grouped_carbonates", np.nan)),
                )
                _row_cap, _ = resolve_flat_tail_cap_anchor_pct(
                    chemistry_cap=_row_cat_cap if status == "Catalyzed" else _row_ctrl_cap,
                    final_observed_recovery=_final_observed_recovery,
                    terminal_slope_rate=_terminal_rate,
                    duration_days=_curve_duration_days,
                )
                _row_data_anchor = resolve_data_informed_cap_anchor_pct(
                    chemistry_cap=_row_cat_cap if status == "Catalyzed" else _row_ctrl_cap,
                    final_observed_recovery=_final_observed_recovery,
                    terminal_slope_rate=_terminal_rate,
                    duration_days=_curve_duration_days,
                    cu_oxides_equiv=scalar_from_maybe_array(row.get("copper_oxides_equivalent", np.nan)),
                    cu_secondary_equiv=scalar_from_maybe_array(row.get("copper_secondary_sulfides_equivalent", np.nan)),
                    cu_primary_equiv=scalar_from_maybe_array(row.get("copper_primary_sulfides_equivalent", np.nan)),
                )
                if bool(_row_data_anchor.get("active", False)):
                    _row_cap = float(_row_data_anchor["cap"])
                _use_row_gate = should_use_sigmoid_gated_prefit(
                    time_days=t,
                    recovery_pct=y,
                    material_size_p80_in=scalar_from_maybe_array(row.get("material_size_p80_in", np.nan)),
                    column_height_m=scalar_from_maybe_array(row.get("column_height_m", np.nan)),
                )
                if _use_row_gate:
                    fit_params, fit_gate_mid_day, fit_gate_width_day = fit_sigmoid_gated_biexponential_params(
                        t,
                        y,
                        cap=_row_cap,
                    )
                    fit_curve_mode = "sigmoid_gated_biexponential"
                if not np.all(np.isfinite(fit_params)):
                    fit_params = fit_biexponential_params(t, y, cap=_row_cap)
                    fit_curve_mode = "biexponential"
                    fit_gate_mid_day = np.nan
                    fit_gate_width_day = np.nan

            # v12: build static_vec for CSV-sourced columns first, then append
            # the computed terminal_slope_rate feature (not in the CSV).
            static_vec = np.asarray(
                [
                    resolve_static_predictor_value(
                        row,
                        col,
                        optional_defaults=optional_static_defaults,
                    )
                    for col in CSV_STATIC_PREDICTOR_COLUMNS
                ],
                dtype=float,
            )
            static_vec = np.append(static_vec, _terminal_rate)
            static_vec = validate_required_feature_vector(
                static_vec,
                STATIC_PREDICTOR_COLUMNS,
                f"sample_id={sample_id} col_id={col_id} row_index={idx} static predictors",
            )
            input_only_vec = np.asarray(
                [scalar_from_maybe_array(row.get(col, np.nan)) for col in INPUT_ONLY_COLUMNS],
                dtype=float,
            )
            input_only_vec = validate_required_feature_vector(
                input_only_vec,
                INPUT_ONLY_COLUMNS,
                f"sample_id={sample_id} col_id={col_id} row_index={idx} input-only predictors",
            )
            project_name = str(row.get(PROJECT_NAME_COL, sample_id)).strip()

            # Feed ORP auxiliary target (input-side, existing logic)
            feed_orp_aux_target_raw = compute_orp_aux_target_from_profile(
                time_days=t_raw,
                orp_profile=parse_listlike(row.get(ORP_PROFILE_COL, np.nan)),
                window_start_day=orp_window_start_day,
                window_end_day=orp_window_end_day,
                history_window_days=orp_history_window_days,
                trim_quantile=orp_trim_quantile,
                step_days=orp_step_days,
                min_recent_points=orp_min_recent_points,
                min_target_points=orp_min_target_points,
            )

            # v11: PLS ORP auxiliary target (output-side, new).
            # The PLS ORP marks what exits the column after solution-ore interaction.
            # The catalyst concentration in the PLS follows a decay pattern that
            # the model learns per sample (guided by column_height_m and ore properties).
            # Here we compute the same windowed summary as feed ORP, but from the
            # pls_orp_mv_ag_agcl profile.
            pls_orp_raw = parse_listlike(row.get(PLS_ORP_PROFILE_COL, np.nan))
            pls_orp_aux_target_raw = compute_orp_aux_target_from_profile(
                time_days=t_raw,
                orp_profile=pls_orp_raw,
                window_start_day=orp_window_start_day,
                window_end_day=orp_window_end_day,
                history_window_days=orp_history_window_days,
                trim_quantile=orp_trim_quantile,
                step_days=orp_step_days,
                min_recent_points=orp_min_recent_points,
                min_target_points=orp_min_target_points,
            )

            # Dosage start/stop annotations follow the resolved feed concentration
            # vector on the aligned model grid.
            if status == "Control":
                _dosage_start_day = float("nan")
                _dosage_stop_day = float("nan")
            else:
                _dosage_start_day = infer_catalyst_start_day_from_dosage_array(
                    t,
                    catalyst_feed_conc_reconstructed,
                )
                _dosage_stop_day = infer_catalyst_stop_day_from_dosage_array(
                    t,
                    catalyst_feed_conc_reconstructed,
                )

            _resolved_start_day, _resolved_start_source = resolve_curve_catalyst_start_day(
                row=row,
                status=status,
                time_days=t_raw,
                recovery=y_raw,
            )
            _effective_start_day, _effective_start_source = resolve_curve_internal_catalyst_effect_start_day(
                row=row,
                status=status,
                time_days=t,
                recovery=y,
                catalyst_cum_kg_t=c,
                dosage_mg_l=catalyst_feed_conc_reconstructed,
                dosage_source=catalyst_feed_conc_source,
            )
            # Average dose follows the same resolved feed-concentration vector
            # used for model inputs, falling back to cumulative profiles only
            # when catalyst_addition_mg_l was unavailable.
            _avg_catalyst_dose_mg_l = compute_average_catalyst_dose_mg_l(
                status=status,
                time_days=t,
                catalyst_start_day=_resolved_start_day,
                catalyst_cum_kg_t=c,
                lixiviant_cum_m3_t=lix_aligned,
                catalyst_feed_conc_mg_l=catalyst_feed_conc_reconstructed,
            )

            curve_data = CurveData(
                status=status,
                time=t,
                recovery=y,
                catalyst_cum=c,
                lixiviant_cum=np.asarray(lix_aligned, dtype=float),
                irrigation_rate_l_m2_h=np.asarray(irrigation_rate, dtype=float),
                catalyst_addition_mg_l_reconstructed=np.asarray(catalyst_feed_conc_reconstructed, dtype=float),
                fit_params=sanitize_curve_params(fit_params, enforce_cap=False),
                row_index=int(idx),
                fit_curve_mode=str(fit_curve_mode),
                fit_gate_mid_day=float(fit_gate_mid_day) if np.isfinite(fit_gate_mid_day) else np.nan,
                fit_gate_width_day=float(fit_gate_width_day) if np.isfinite(fit_gate_width_day) else np.nan,
                prefit_fit_start_day=float(prefit_fit_start_day) if np.isfinite(prefit_fit_start_day) else np.nan,
                prefit_fit_start_day_source=str(prefit_fit_start_day_source),
                pls_orp_profile=np.asarray(pls_orp_raw, dtype=float) if pls_orp_raw.size > 0 else np.array([], dtype=float),
                col_id=col_id,
                catalyst_conc_col_mg_l=np.asarray(conc_aligned, dtype=float),
                catalyst_start_day=float(_resolved_start_day),
                catalyst_start_day_source=str(_resolved_start_source),
                catalyst_effective_start_day=float(_effective_start_day),
                catalyst_effective_start_day_source=str(_effective_start_source),
                catalyst_stop_day=float(_dosage_stop_day),
                avg_catalyst_dose_mg_l=float(_avg_catalyst_dose_mg_l),
                catalyst_dosage_start_day=float(_dosage_start_day),
                catalyst_dosage_stop_day=float(_dosage_stop_day),
            )
            by_status[status].append(
                (
                    curve_data, static_vec, input_only_vec, project_name,
                    feed_orp_aux_target_raw, pls_orp_aux_target_raw,
                )
            )

        if len(by_status["Control"]) == 0 or len(by_status["Catalyzed"]) == 0:
            continue

        for ctrl_curve, ctrl_static, ctrl_input_only, ctrl_project_name, ctrl_feed_orp, ctrl_pls_orp in by_status["Control"]:
            for cat_curve, cat_static, cat_input_only, cat_project_name, cat_feed_orp, cat_pls_orp in by_status["Catalyzed"]:
                merged_static = combine_static_vectors(
                    ctrl_static,
                    cat_static,
                    columns=STATIC_PREDICTOR_COLUMNS,
                    context=f"sample_id={sample_id} pair ctrl={ctrl_curve.col_id} cat={cat_curve.col_id} static predictors",
                )
                merged_input_only = combine_static_vectors(
                    ctrl_input_only,
                    cat_input_only,
                    columns=INPUT_ONLY_COLUMNS,
                    context=f"sample_id={sample_id} pair ctrl={ctrl_curve.col_id} cat={cat_curve.col_id} input-only predictors",
                )

                # v11/v12: caps derived from per-sample mineralogy (project_sample_id level).
                def _sv(col: str) -> float:
                    if col not in STATIC_PREDICTOR_COLUMNS:
                        return np.nan
                    return float(merged_static[STATIC_PREDICTOR_COLUMNS.index(col)])

                def _curve_sv(vec: np.ndarray, col: str) -> float:
                    if col not in STATIC_PREDICTOR_COLUMNS:
                        return np.nan
                    return float(vec[STATIC_PREDICTOR_COLUMNS.index(col)])

                def _iv(col: str) -> float:
                    if col not in INPUT_ONLY_COLUMNS:
                        return np.nan
                    return float(merged_input_only[INPUT_ONLY_COLUMNS.index(col)])

                pair_ctrl_cap, _ = compute_sample_leach_caps(
                    cu_oxides_equiv=_sv("copper_oxides_equivalent"),
                    cu_secondary_equiv=_sv("copper_secondary_sulfides_equivalent"),
                    cu_primary_equiv=_sv("copper_primary_sulfides_equivalent"),
                    material_size_p80_in=_sv("material_size_p80_in"),
                    residual_cpy_pct=_sv("residual_cpy_%"),
                    column_height_m=_iv("column_height_m"),
                    lixiviant_initial_fe_mg_l=_curve_sv(ctrl_static, "lixiviant_initial_fe_mg_l"),
                    lixiviant_initial_ph=_curve_sv(ctrl_static, "lixiviant_initial_ph"),
                    lixiviant_initial_orp_mv=_curve_sv(ctrl_static, "lixiviant_initial_orp_mv"),
                    cyanide_soluble_pct=_sv("cyanide_soluble_%"),
                    acid_soluble_pct=_sv("acid_soluble_%"),
                    grouped_acid_generating_sulfides_pct=_sv("grouped_acid_generating_sulfides"),
                    grouped_carbonates_pct=_sv("grouped_carbonates"),
                )
                _, pair_cat_cap = compute_sample_leach_caps(
                    cu_oxides_equiv=_sv("copper_oxides_equivalent"),
                    cu_secondary_equiv=_sv("copper_secondary_sulfides_equivalent"),
                    cu_primary_equiv=_sv("copper_primary_sulfides_equivalent"),
                    material_size_p80_in=_sv("material_size_p80_in"),
                    residual_cpy_pct=_sv("residual_cpy_%"),
                    column_height_m=_iv("column_height_m"),
                    lixiviant_initial_fe_mg_l=_curve_sv(cat_static, "lixiviant_initial_fe_mg_l"),
                    lixiviant_initial_ph=_curve_sv(cat_static, "lixiviant_initial_ph"),
                    lixiviant_initial_orp_mv=_curve_sv(cat_static, "lixiviant_initial_orp_mv"),
                    cyanide_soluble_pct=_sv("cyanide_soluble_%"),
                    acid_soluble_pct=_sv("acid_soluble_%"),
                    grouped_acid_generating_sulfides_pct=_sv("grouped_acid_generating_sulfides"),
                    grouped_carbonates_pct=_sv("grouped_carbonates"),
                )
                ctrl_final_recovery = float(ctrl_curve.recovery[-1]) if ctrl_curve.recovery.size > 0 else np.nan
                ctrl_terminal_slope = _curve_sv(ctrl_static, "terminal_slope_rate")
                ctrl_duration_days = (
                    float(ctrl_curve.time[-1] - ctrl_curve.time[0]) if ctrl_curve.time.size > 1 else 0.0
                )
                pair_ctrl_data_anchor = resolve_data_informed_cap_anchor_pct(
                    chemistry_cap=pair_ctrl_cap,
                    final_observed_recovery=ctrl_final_recovery,
                    terminal_slope_rate=ctrl_terminal_slope,
                    duration_days=ctrl_duration_days,
                    cu_oxides_equiv=_sv("copper_oxides_equivalent"),
                    cu_secondary_equiv=_sv("copper_secondary_sulfides_equivalent"),
                    cu_primary_equiv=_sv("copper_primary_sulfides_equivalent"),
                )
                if bool(pair_ctrl_data_anchor.get("active", False)):
                    pair_ctrl_cap_anchor_pct = float(pair_ctrl_data_anchor["cap"])
                    pair_ctrl_cap_anchor_active = True
                else:
                    pair_ctrl_cap_anchor_pct, pair_ctrl_cap_anchor_active = resolve_flat_tail_cap_anchor_pct(
                        chemistry_cap=pair_ctrl_cap,
                        final_observed_recovery=ctrl_final_recovery,
                        terminal_slope_rate=ctrl_terminal_slope,
                        duration_days=ctrl_duration_days,
                    )

                cat_final_recovery = float(cat_curve.recovery[-1]) if cat_curve.recovery.size > 0 else np.nan
                cat_terminal_slope = _curve_sv(cat_static, "terminal_slope_rate")
                cat_duration_days = (
                    float(cat_curve.time[-1] - cat_curve.time[0]) if cat_curve.time.size > 1 else 0.0
                )
                pair_cat_data_anchor = resolve_data_informed_cap_anchor_pct(
                    chemistry_cap=pair_cat_cap,
                    final_observed_recovery=cat_final_recovery,
                    terminal_slope_rate=cat_terminal_slope,
                    duration_days=cat_duration_days,
                    cu_oxides_equiv=_sv("copper_oxides_equivalent"),
                    cu_secondary_equiv=_sv("copper_secondary_sulfides_equivalent"),
                    cu_primary_equiv=_sv("copper_primary_sulfides_equivalent"),
                )
                if bool(pair_cat_data_anchor.get("active", False)):
                    pair_cat_cap_anchor_pct = float(pair_cat_data_anchor["cap"])
                    pair_cat_cap_anchor_active = True
                else:
                    pair_cat_cap_anchor_pct, pair_cat_cap_anchor_active = resolve_flat_tail_cap_anchor_pct(
                        chemistry_cap=pair_cat_cap,
                        final_observed_recovery=cat_final_recovery,
                        terminal_slope_rate=cat_terminal_slope,
                        duration_days=cat_duration_days,
                    )
                pair_project_name = cat_project_name or ctrl_project_name or str(sample_id)

                # Resolve feed ORP aux target
                _orp_source = str(CONFIG.get("orp_aux_source", "control")).strip().lower()
                _ctrl_feed_ok = np.isfinite(ctrl_feed_orp)
                _cat_feed_ok = np.isfinite(cat_feed_orp)
                if _orp_source == "catalyzed":
                    pair_orp_aux_target_raw = float(cat_feed_orp) if _cat_feed_ok else (float(ctrl_feed_orp) if _ctrl_feed_ok else np.nan)
                elif _orp_source == "average":
                    if _ctrl_feed_ok and _cat_feed_ok:
                        pair_orp_aux_target_raw = 0.5 * (float(ctrl_feed_orp) + float(cat_feed_orp))
                    elif _ctrl_feed_ok:
                        pair_orp_aux_target_raw = float(ctrl_feed_orp)
                    elif _cat_feed_ok:
                        pair_orp_aux_target_raw = float(cat_feed_orp)
                    else:
                        pair_orp_aux_target_raw = np.nan
                else:  # default: "control"
                    pair_orp_aux_target_raw = float(ctrl_feed_orp) if _ctrl_feed_ok else (float(cat_feed_orp) if _cat_feed_ok else np.nan)

                # v11: resolve PLS ORP aux target.
                # For the PLS ORP, the catalyzed column is the primary source because
                # it carries the catalyst decay signal (catalyst exits with PLS).
                # Control columns provide baseline PLS ORP (no catalyst signature).
                _ctrl_pls_ok = np.isfinite(ctrl_pls_orp)
                _cat_pls_ok = np.isfinite(cat_pls_orp)
                if _cat_pls_ok:
                    pair_pls_orp_aux_target_raw = float(cat_pls_orp)
                elif _ctrl_pls_ok:
                    pair_pls_orp_aux_target_raw = float(ctrl_pls_orp)
                else:
                    pair_pls_orp_aux_target_raw = np.nan

                pairs.append(
                    PairSample(
                        pair_id=make_pair_id(sample_id, ctrl_curve, cat_curve),
                        sample_id=str(sample_id),
                        project_name=str(pair_project_name),
                        static_raw=merged_static,
                        input_only_raw=merged_input_only,
                        control=ctrl_curve,
                        catalyzed=cat_curve,
                        control_static_raw=np.asarray(ctrl_static, dtype=float),
                        catalyzed_static_raw=np.asarray(cat_static, dtype=float),
                        ctrl_cap=pair_ctrl_cap_anchor_pct,
                        cat_cap=pair_cat_cap_anchor_pct,
                        control_cap_anchor_pct=pair_ctrl_cap_anchor_pct,
                        catalyzed_cap_anchor_pct=pair_cat_cap_anchor_pct,
                        control_cap_anchor_active=bool(pair_ctrl_cap_anchor_active),
                        catalyzed_cap_anchor_active=bool(pair_cat_cap_anchor_active),
                        orp_aux_target_raw=pair_orp_aux_target_raw,
                        pls_orp_aux_target_raw=pair_pls_orp_aux_target_raw,
                    )
                )
    pairs = sorted(pairs, key=lambda x: (x.sample_id, x.pair_id))
    assert_pairs_respect_training_exclusions(pairs, context="build_pair_samples output")
    return pairs


def fit_static_transformers(train_pairs: List[PairSample]) -> Tuple[SimpleImputer, StandardScaler]:
    X_train = np.vstack([p.static_raw for p in train_pairs]).astype(float)
    for row_idx, row in enumerate(X_train):
        validate_required_feature_vector(
            row,
            STATIC_PREDICTOR_COLUMNS,
            f"train_pairs[{row_idx}] static_raw",
        )
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
    for row_idx, row in enumerate(X):
        validate_required_feature_vector(
            row,
            STATIC_PREDICTOR_COLUMNS,
            f"pairs[{row_idx}] static_raw",
        )
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    for p, xi, xs in zip(pairs, X_imp, X_scaled):
        validate_required_feature_vector(
            xi,
            STATIC_PREDICTOR_COLUMNS,
            f"pair_id={p.pair_id} sample_id={p.sample_id} static_imputed",
        )
        validate_required_feature_vector(
            xs,
            STATIC_PREDICTOR_COLUMNS,
            f"pair_id={p.pair_id} sample_id={p.sample_id} static_scaled",
        )
        p.static_imputed = np.asarray(xi, dtype=float)
        p.static_scaled = np.asarray(xs, dtype=float)
        p.clear_tensor_cache()


def fit_orp_aux_normalization(train_pairs: List[PairSample]) -> Dict[str, Any]:
    """Compute normalization stats for feed ORP and PLS ORP auxiliary targets.

    v11: returns a nested dict with keys "feed_orp" and "pls_orp", each
    containing global_mean, global_std, and project_stats.
    The function also exposes the feed ORP stats at the top level for
    backward compatibility with any code that accesses norm_stats["global_mean"].
    """
    floor_mv = float(CONFIG.get("orp_aux_norm_floor_mv", 10.0))
    min_project_pairs = int(max(1, CONFIG.get("orp_aux_project_min_pairs", 2)))

    feed_stats = _compute_orp_norm_stats(
        train_pairs, "orp_aux_target_raw", floor_mv, min_project_pairs
    )
    pls_stats = _compute_orp_norm_stats(
        train_pairs, "pls_orp_aux_target_raw", floor_mv, min_project_pairs
    )

    return {
        # Top-level feed ORP keys for backward compatibility
        "global_mean": feed_stats["global_mean"],
        "global_std": feed_stats["global_std"],
        "project_stats": feed_stats["project_stats"],
        # v11: nested per-signal stats
        "feed_orp": feed_stats,
        "pls_orp": pls_stats,
    }


def _compute_orp_norm_stats(
    pairs: List[PairSample],
    raw_attr: str,
    floor_mv: float,
    min_project_pairs: int,
) -> Dict[str, Any]:
    """Generic helper: compute mean/std normalization stats for any ORP aux target."""
    values_by_project: Dict[str, List[float]] = {}
    all_values: List[float] = []
    for pair in pairs:
        value = float(getattr(pair, raw_attr, np.nan))
        if not np.isfinite(value):
            continue
        project_name = str(pair.project_name).strip() or str(pair.sample_id)
        values_by_project.setdefault(project_name, []).append(value)
        all_values.append(value)
    if len(all_values) == 0:
        return {"global_mean": 0.0, "global_std": 1.0, "project_stats": {}}
    all_arr = np.asarray(all_values, dtype=float)
    project_stats: Dict[str, Dict[str, float]] = {}
    for project_name, values in values_by_project.items():
        values_arr = np.asarray(values, dtype=float)
        if values_arr.size < min_project_pairs:
            continue
        project_stats[project_name] = {
            "mean": float(np.mean(values_arr)),
            "std": float(max(np.std(values_arr), floor_mv)),
        }
    return {
        "global_mean": float(np.mean(all_arr)),
        "global_std": float(max(np.std(all_arr), floor_mv)),
        "project_stats": project_stats,
    }


def apply_orp_aux_normalization(
    pairs: List[PairSample],
    norm_stats: Dict[str, Any],
) -> None:
    """Apply normalization for BOTH feed ORP and PLS ORP auxiliary targets."""
    floor_mv = float(CONFIG.get("orp_aux_norm_floor_mv", 10.0))
    min_project_pairs = int(max(1, CONFIG.get("orp_aux_project_min_pairs", 2)))

    # Feed ORP normalization (original)
    feed_stats = norm_stats.get("feed_orp", norm_stats)  # backward-compatible
    feed_project_stats = dict(feed_stats.get("project_stats", {}))
    feed_global_mean = float(feed_stats.get("global_mean", 0.0))
    feed_global_std = float(max(feed_stats.get("global_std", 1.0), 1e-6))

    # PLS ORP normalization (v11)
    pls_stats = norm_stats.get("pls_orp", {})
    pls_project_stats = dict(pls_stats.get("project_stats", {}))
    pls_global_mean = float(pls_stats.get("global_mean", 0.0))
    pls_global_std = float(max(pls_stats.get("global_std", 1.0), 1e-6))

    for pair in pairs:
        # Feed ORP
        raw_feed = float(pair.orp_aux_target_raw)
        if np.isfinite(raw_feed):
            project_name = str(pair.project_name).strip() or str(pair.sample_id)
            stats = feed_project_stats.get(project_name)
            mean = float(stats["mean"]) if stats is not None else feed_global_mean
            std = float(max(stats["std"], 1e-6)) if stats is not None else feed_global_std
            pair.orp_aux_target_norm = float((raw_feed - mean) / std)
            pair.orp_aux_mask = 1.0
        else:
            pair.orp_aux_target_norm = np.nan
            pair.orp_aux_mask = 0.0

        # PLS ORP (v11)
        raw_pls = float(pair.pls_orp_aux_target_raw)
        if np.isfinite(raw_pls):
            project_name = str(pair.project_name).strip() or str(pair.sample_id)
            stats = pls_project_stats.get(project_name)
            mean = float(stats["mean"]) if stats is not None else pls_global_mean
            std = float(max(stats["std"], 1e-6)) if stats is not None else pls_global_std
            pair.pls_orp_aux_target_norm = float((raw_pls - mean) / std)
            pair.pls_orp_aux_mask = 1.0
        else:
            pair.pls_orp_aux_target_norm = np.nan
            pair.pls_orp_aux_mask = 0.0

        pair.clear_tensor_cache()


def build_repeated_group_kfold_member_splits(
    sample_ids: List[str],
    n_splits: int,
    n_repeats: int,
    n_split_seeds: int,
    random_state: int,
    member_seed_base: int,
    group_alias: Optional[Dict[str, str]] = None,
    strata_labels: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Repeated (optionally stratified) Group K-Fold over sample_ids.

    group_alias (G): maps raw sample_ids to a shared group label used only for
    fold construction. Samples sharing the same alias end up in the same fold
    together (either both train or both validation). This is used to tie
    related pairs like '015..amcf_6in' and '015..amcf_8in' to a single group.

    strata_labels (F): maps raw sample_ids to a coarse stratum label (e.g.,
    'primary', 'secondary', 'oxide', 'mixed'). When provided, the fold chunks
    are built within each stratum so every fold sees a representative mix of
    ore types. This stabilizes OOF for minority strata (Elephant / Los
    Bronces / SCL Escondida).
    """
    group_alias = group_alias or {}
    strata_labels = strata_labels or {}

    # Resolve the effective group label (post-alias) for every input row.
    raw_labels = [str(v).strip() for v in sample_ids]
    effective_labels = [group_alias.get(label, label) for label in raw_labels]

    # Unique effective groups become the actual CV units.
    unique_group_labels = sorted({label for label in effective_labels if label})
    n_groups = len(unique_group_labels)
    if n_groups < 3:
        raise ValueError(
            f"Need at least 3 unique project_sample_id groups for repeated group K-fold, got {n_groups}."
        )
    n_splits = int(max(2, min(int(n_splits), n_groups)))
    n_repeats = int(max(1, n_repeats))
    n_split_seeds = int(max(1, n_split_seeds))

    # For each effective group, pick a single stratum label (majority vote of
    # its member sample_ids; fall back to '_' if no strata provided).
    def _stratum_for_group(group_label: str) -> str:
        if not strata_labels:
            return "_"
        members = [
            strata_labels.get(raw_labels[i], "_")
            for i, eff in enumerate(effective_labels)
            if eff == group_label
        ]
        if not members:
            return "_"
        # Most common stratum among this group's members.
        counts: Dict[str, int] = {}
        for m in members:
            counts[m] = counts.get(m, 0) + 1
        return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    group_to_stratum = {g: _stratum_for_group(g) for g in unique_group_labels}
    strata_to_groups: Dict[str, List[str]] = {}
    for g, s in group_to_stratum.items():
        strata_to_groups.setdefault(s, []).append(g)
    for s in strata_to_groups:
        strata_to_groups[s].sort()

    all_indices = np.arange(len(raw_labels), dtype=int)
    splits: List[Dict[str, Any]] = []
    member_idx = 0
    for split_seed_idx in range(n_split_seeds):
        split_random_state = int(random_state) + int(split_seed_idx)
        for repeat_idx in range(n_repeats):
            repeat_random_state = int(split_random_state) + int(repeat_idx)
            rng = np.random.default_rng(repeat_random_state)

            # Stratified chunking: shuffle groups within each stratum, build a
            # single stratum-interleaved queue (round-robin ACROSS strata), then
            # assign that queue round-robin across the K folds with a single
            # shared starting offset. This guarantees every fold receives
            # ⌊G/K⌋ or ⌈G/K⌉ groups (no empty folds) while still keeping
            # stratum balance because consecutive slots in the interleaved
            # queue come from distinct strata.
            stratum_queues: Dict[str, List[str]] = {}
            ordered_strata = sorted(strata_to_groups.keys())
            rng.shuffle(ordered_strata)
            for stratum in ordered_strata:
                gs = list(strata_to_groups[stratum])
                rng.shuffle(gs)
                stratum_queues[stratum] = gs

            interleaved: List[str] = []
            while any(stratum_queues[s] for s in ordered_strata):
                for stratum in ordered_strata:
                    if stratum_queues[stratum]:
                        interleaved.append(stratum_queues[stratum].pop(0))

            fold_groups_by_idx: List[List[str]] = [[] for _ in range(n_splits)]
            global_start = int(rng.integers(0, n_splits))
            for i, group_label in enumerate(interleaved):
                fold_groups_by_idx[(global_start + i) % n_splits].append(group_label)

            for fold_idx, fold_groups in enumerate(fold_groups_by_idx):
                val_group_set = {str(v) for v in fold_groups}
                # A row is validation if its *effective* group falls in
                # this fold's group set. That is what implements G: the
                # aliased 015 amcf 6in/8in always share fate.
                val_mask = np.asarray(
                    [label in val_group_set for label in effective_labels], dtype=bool
                )
                train_idx = all_indices[~val_mask]
                val_idx = all_indices[val_mask]
                if train_idx.size == 0 or val_idx.size == 0:
                    raise ValueError(
                        "Invalid repeated group K-fold split: "
                        f"repeat={repeat_idx}, fold={fold_idx}, "
                        f"train={train_idx.size}, val={val_idx.size}."
                    )

                member_seed = int(member_seed_base) + int(member_idx)
                splits.append(
                    {
                        "member_idx": int(member_idx),
                        "split_seed_idx": int(split_seed_idx),
                        "split_random_state": int(split_random_state),
                        "repeat_idx": int(repeat_idx),
                        "fold_idx": int(fold_idx),
                        "member_seed": int(member_seed),
                        "train_indices": np.asarray(train_idx, dtype=int),
                        "val_indices": np.asarray(val_idx, dtype=int),
                        "train_group_labels": sorted({raw_labels[i] for i in train_idx}),
                        "val_group_labels": sorted({raw_labels[i] for i in val_idx}),
                        "train_effective_groups": sorted({effective_labels[i] for i in train_idx}),
                        "val_effective_groups": sorted({effective_labels[i] for i in val_idx}),
                        "val_strata_counts": {
                            s: sum(1 for g in fold_groups if group_to_stratum.get(g, "_") == s)
                            for s in sorted(strata_to_groups.keys())
                        },
                    }
                )
                member_idx += 1
    return splits


def compute_mineralogy_stratum_labels(
    pairs: List["PairSample"],
) -> Dict[str, str]:
    """Assign each sample_id a coarse mineralogy stratum used for stratified
    K-fold splitting (F). Strata are picked from the dominant Cu-equivalent
    inventory: oxide, secondary (chalcocite/covellite), primary (chalcopyrite)
    or mixed when no component clearly dominates.
    """
    idx_map = STATIC_PREDICTOR_INDEX
    pri_idx = idx_map.get("copper_primary_sulfides_equivalent")
    sec_idx = idx_map.get("copper_secondary_sulfides_equivalent")
    ox_idx = idx_map.get("copper_oxides_equivalent")

    labels: Dict[str, str] = {}
    for pair in pairs:
        sid = str(pair.sample_id)
        try:
            pri = float(pair.static_raw[pri_idx]) if pri_idx is not None else 0.0
        except Exception:
            pri = 0.0
        try:
            sec = float(pair.static_raw[sec_idx]) if sec_idx is not None else 0.0
        except Exception:
            sec = 0.0
        try:
            ox = float(pair.static_raw[ox_idx]) if ox_idx is not None else 0.0
        except Exception:
            ox = 0.0
        pri = max(0.0, pri) if np.isfinite(pri) else 0.0
        sec = max(0.0, sec) if np.isfinite(sec) else 0.0
        ox = max(0.0, ox) if np.isfinite(ox) else 0.0
        total = pri + sec + ox
        if total <= 1e-9:
            labels[sid] = "unknown"
            continue
        frac_pri = pri / total
        frac_sec = sec / total
        frac_ox = ox / total
        # Call it "dominant" if a single component has ≥ 60% of the Cu pool,
        # otherwise "mixed". This keeps the number of strata small and each
        # stratum meaningfully populated.
        if frac_ox >= 0.60:
            labels[sid] = "oxide"
        elif frac_sec >= 0.60:
            labels[sid] = "secondary"
        elif frac_pri >= 0.60:
            labels[sid] = "primary"
        else:
            labels[sid] = "mixed"
    return labels


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
        max_transition_days: float,
        max_catalyst_aging_strength: float,
        late_tau_impact_decay_strength: float,
        min_remaining_ore_factor: float,
        flat_input_transition_sensitivity: float,
        flat_input_uplift_response_days: float,
        flat_input_response_ramp_days: float,
        flat_input_late_uplift_response_boost: float,
        use_prefit_param_bounds: bool = True,
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
        self.cat_delta_head = nn.Linear(hidden_dim, 4)
        self.initial_gate_head = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.initial_gate_head.weight)
        nn.init.constant_(self.initial_gate_head.bias, -3.0)
        self.delay_head = nn.Linear(hidden_dim, 1)
        self.height_delay_scale = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))
        self.p80_tau_scale = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.p80_temp_scale = nn.Parameter(torch.tensor(-1.2, dtype=torch.float32))
        self.p80_kappa_penalty_scale = nn.Parameter(torch.tensor(-1.2, dtype=torch.float32))
        # (A) Learnable shrinking-core / diffusion-length exponent on the
        # overall kinetic rate. The effective "age" of the curve is multiplied
        # by (p80_ref / max(p80, p80_ref)) ** alpha, alpha in [0.3, 1.5],
        # so larger particle sizes always slow every curve (ctrl and cat)
        # monotonically. Initialized near 0.7 (physically: half-way between a
        # surface-limited 0.5 exponent and a diffusion-limited 1.0 exponent).
        self.p80_rate_alpha_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.material_transport_scale = nn.Parameter(torch.tensor(-1.5, dtype=torch.float32))
        self.geometry_response_scale = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))
        # Mineralogy-based sigmoid gate onset priors (learnable scalings).
        # gate_mid_day is additively shifted by:
        #   fast_cu_shift:   slow_cu_frac (= 1 - fast_cu_frac) × gate_mid_max × softplus(scale)
        #   height_shift:    log1p(column_height_m) × softplus(scale)
        #   acid_shift:      acid_consumer_net × gate_mid_max × softplus(scale)
        # Initialised small (≈ 0) so the NN learns from data; the encoder
        # already sees mineralogy through the static feature vector.
        # Initialised negative so softplus gives a small starting contribution.
        self.gate_onset_fast_cu_scale_raw = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.gate_onset_height_scale_raw = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))
        self.gate_onset_acid_scale_raw = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))
        # Acid-consumer feature indices (cached at first use)
        self._carbonates_idx: Optional[int] = None
        self._acid_gen_idx: Optional[int] = None
        self.temp_head = nn.Linear(hidden_dim, 1)
        self.kappa_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.kappa_head.bias, 1.0)  # softplus(1.0) ≈ 1.31
        self.aging_head = nn.Linear(hidden_dim, 1)
        self.tau_active_mean_scale = nn.Parameter(torch.tensor(-0.08, dtype=torch.float32))
        self.tau_active_peak_scale = nn.Parameter(torch.tensor(-0.06, dtype=torch.float32))
        self.tau_active_recent_scale = nn.Parameter(torch.tensor(-0.08, dtype=torch.float32))
        self.temp_active_mean_scale = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))
        self.temp_active_peak_scale = nn.Parameter(torch.tensor(0.06, dtype=torch.float32))
        self.temp_active_recent_scale = nn.Parameter(torch.tensor(0.06, dtype=torch.float32))
        self.kappa_active_mean_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.kappa_active_peak_scale = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))
        self.kappa_active_recent_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.aging_active_mean_scale = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))
        self.aging_active_recent_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.mid_life_head = nn.Linear(1, 1)  # input is aging_raw (B, 1)
        nn.init.constant_(self.mid_life_head.bias, 0.5)  # sigmoid(0) = 0.5 → ~450 days mid-life
        self.use_frac_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.use_frac_head.bias, 0.5)  # sigmoid(0) = 0.5 → 50% use fraction
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
        self.ferric_bootstrap_head = nn.Linear(hidden_dim, 1)
        self.primary_drive_head = nn.Linear(hidden_dim, 1)
        self.fast_inventory_head = nn.Linear(hidden_dim, 1)
        self.oxide_inventory_head = nn.Linear(hidden_dim, 1)
        self.acid_buffer_head = nn.Linear(hidden_dim, 1)
        self.acid_buffer_decay_head = nn.Linear(hidden_dim, 1)
        self.diffusion_drag_head = nn.Linear(hidden_dim, 1)
        self.ferric_synergy_head = nn.Linear(hidden_dim, 1)
        self.primary_catalyst_synergy_head = nn.Linear(hidden_dim, 1)
        self.surface_refresh_head = nn.Linear(hidden_dim, 1)
        # (B) Learnable per-sample leachable fractions of PRIMARY sulfides.
        # These do not replace metallurgy with unconstrained fit-driven caps:
        # both control and catalyzed fractions are later centered on a
        # deterministic chemistry/geometry prior and only allowed to deviate
        # by a narrow +/- band around that prior.
        self.cpy_leach_frac_cat_head = nn.Linear(hidden_dim, 1)
        self.cpy_leach_frac_ctrl_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.cpy_leach_frac_cat_head.weight)
        nn.init.zeros_(self.cpy_leach_frac_ctrl_head.weight)
        nn.init.constant_(self.cpy_leach_frac_cat_head.bias, 0.0)
        nn.init.constant_(self.cpy_leach_frac_ctrl_head.bias, 0.0)

        # (E) Acid-consumption early-delay heads. Fresh ores with high
        # carbonates and/or acid-generating sulfides consume H+/Fe3+ before
        # any copper is leached, so in the first weeks the catalyzed curve
        # can actually sit *below* the control (Toquepala fresca). These two
        # heads give the network a signed early-time deficit term:
        #
        #   deficit(t) = amp * exp(-max(t-t0, 0)/tau) * gate
        #
        # where ``amp`` is the percentage-point magnitude (0 to ~4 pp),
        # ``tau`` is the decay time constant in days (15 to ~180 d), and
        # ``gate`` is a learnable non-negative scalar from the same latent
        # vector so the whole term is gated off for samples without
        # acid-generating chemistry.  Weights are initialised to zero so
        # ``amp`` starts near 0 and the model learns to activate it only for
        # samples where the data demands it.
        self.acid_consumption_amp_head = nn.Linear(hidden_dim, 1)
        self.acid_consumption_tau_head = nn.Linear(hidden_dim, 1)
        self.acid_consumption_gate_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.acid_consumption_amp_head.weight)
        nn.init.zeros_(self.acid_consumption_tau_head.weight)
        nn.init.zeros_(self.acid_consumption_gate_head.weight)
        # amp bias < 0 → sigmoid ≈ 0.018 → ~0.07 pp initial amplitude (near-off).
        nn.init.constant_(self.acid_consumption_amp_head.bias, -4.0)
        # tau bias 0 → sigmoid 0.5 → tau ≈ 60 d initial (with 15 + 150 * sig).
        nn.init.constant_(self.acid_consumption_tau_head.bias, 0.0)
        # gate bias 0 → sigmoid 0.5 (gate half-open until learnt otherwise).
        nn.init.constant_(self.acid_consumption_gate_head.bias, 0.0)
        self.orp_aux_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.orp_aux_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # v11: learnable scales for CSTR column-concentration signal (mg/L normalised).
        # These three scalars gate the concentration summary's influence on tau, kappa,
        # and the transition-time parameters independently of the cumulative mass signal.
        self.conc_tau_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.conc_kappa_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.conc_temp_scale = nn.Parameter(torch.tensor(0.03, dtype=torch.float32))

        self.geo_idx = list(geo_idx)
        self.geo_delay_head = nn.Linear(len(self.geo_idx), 1) if len(self.geo_idx) > 0 else None

        self.static_feature_indices = dict(STATIC_PREDICTOR_INDEX)
        self.input_only_indices = dict(INPUT_ONLY_INDEX)
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

        ctrl_lb_eff, ctrl_ub_eff = resolve_model_param_bounds(
            ctrl_lb,
            ctrl_ub,
            use_prefit_param_bounds=use_prefit_param_bounds,
        )
        cat_lb_eff, cat_ub_eff = resolve_model_param_bounds(
            cat_lb,
            cat_ub,
            use_prefit_param_bounds=use_prefit_param_bounds,
        )
        self.use_prefit_param_bounds = bool(use_prefit_param_bounds)
        self.register_buffer("ctrl_lb", torch.tensor(ctrl_lb_eff, dtype=torch.float32))
        self.register_buffer("ctrl_ub", torch.tensor(ctrl_ub_eff, dtype=torch.float32))
        self.register_buffer("cat_lb", torch.tensor(cat_lb_eff, dtype=torch.float32))
        self.register_buffer("cat_ub", torch.tensor(cat_ub_eff, dtype=torch.float32))
        self.tmax_days = float(max(1.0, tmax_days))
        self.min_transition_days = float(max(1.0, min_transition_days))
        self.max_transition_days = float(max(self.min_transition_days, max_transition_days))
        self.max_catalyst_aging_strength = float(max(0.0, max_catalyst_aging_strength))
        self.late_tau_impact_decay_strength = float(max(0.0, late_tau_impact_decay_strength))
        self.min_remaining_ore_factor = float(np.clip(min_remaining_ore_factor, 0.0, 1.0))
        self.flat_input_transition_sensitivity = float(max(0.0, flat_input_transition_sensitivity))
        self.flat_input_uplift_response_days = float(max(0.0, flat_input_uplift_response_days))
        self.flat_input_response_ramp_days = float(max(1e-3, flat_input_response_ramp_days))
        self.flat_input_late_uplift_response_boost = float(
            max(0.0, flat_input_late_uplift_response_boost)
        )

    def _bounded_params(self, raw: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        return lb.unsqueeze(0) + torch.sigmoid(raw) * (ub - lb).unsqueeze(0)

    def _bounded_transition_days(self, raw: torch.Tensor) -> torch.Tensor:
        span = self.max_transition_days - self.min_transition_days
        return self.min_transition_days + span * torch.sigmoid(raw)

    def predict_orp_aux_target(self, ferric_synergy: torch.Tensor) -> torch.Tensor:
        ferric = torch.as_tensor(ferric_synergy)
        return self.orp_aux_scale * (ferric - 1.0) + self.orp_aux_bias

    @staticmethod
    def _feature_column(x_static: torch.Tensor, idx: Optional[int]) -> torch.Tensor:
        if idx is None or idx < 0 or idx >= x_static.shape[1]:
            raise ValueError(
                f"Required feature index is unavailable for tensor with shape {tuple(x_static.shape)}."
            )
        return x_static[:, idx : idx + 1]

    @staticmethod
    def _require_finite_feature_tensor(tensor: torch.Tensor, feature_name: str) -> torch.Tensor:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Required feature '{feature_name}' contains missing or invalid values.")
        return tensor

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
            input_only_terms = {
                col_name: self._feature_column(x_input_only_raw, idx)
                for col_name, idx in self.input_only_indices.items()
            }
            height_raw = self._require_finite_feature_tensor(
                input_only_terms["column_height_m"],
                "column_height_m",
            )
            diameter_raw = self._require_finite_feature_tensor(
                input_only_terms["column_inner_diameter_m"],
                "column_inner_diameter_m",
            )
            feed_mass_raw = self._require_finite_feature_tensor(
                input_only_terms[FEED_MASS_COL],
                FEED_MASS_COL,
            )
            material_size_raw = self._require_finite_feature_tensor(
                raw_terms["material_size_p80_in"],
                "material_size_p80_in",
            )
            height_m = torch.clamp(height_raw, min=0.0)
            diameter_m = torch.clamp(diameter_raw, min=0.0)
            feed_mass_kg = torch.clamp(feed_mass_raw, min=0.0)
            material_size_p80_in = torch.clamp(material_size_raw, min=0.0)
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
            terms["column_height_m"] = height_m
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

        a1 = torch.clamp(p_ordered[:, 0], min=1.0)
        b1 = torch.clamp(p_ordered[:, 1], min=1e-5, max=1e-1)
        a2 = torch.clamp(p_ordered[:, 2], min=1.0)
        b2 = torch.clamp(p_ordered[:, 3], min=1e-5, max=1e-1)

        return torch.stack([a1, b1, a2, b2], dim=1)

    # Catalyst deltas are built from shared ore-physics latents.
    # The latent dictionary remains shared across control and catalyzed behavior.
    # Intended catalyst uplift is mainly:
    #   - extra slow recoverable inventory (fit_a2 side)
    #   - relief of slow-tail kinetics burden (fit_b2 side)
    # and only secondarily:
    #   - early fast-kinetics uplift (fit_b1 side)
    def _apply_catalyst_delta(
        self,
        p_ctrl: torch.Tensor,
        raw_delta: torch.Tensor,
        primary_catalyst_synergy: torch.Tensor,
        depassivation_strength: torch.Tensor,
        fast_leach_inventory: torch.Tensor,
        oxide_inventory: torch.Tensor,
        ferric_synergy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build catalyzed parameters as a constrained modification of control params.

        Parameterization logic:
        - A_total_cat >= A_total_ctrl
        - b1_cat and b2_cat are free to move up or down relative to control
        - fast fraction can move slightly up or down
        - _sanitize_params() still enforces that b1 is the fast branch within each curve
        """

        eps = 1e-6

        a1_ctrl = p_ctrl[:, 0:1]
        b1_ctrl = p_ctrl[:, 1:2]
        a2_ctrl = p_ctrl[:, 2:3]
        b2_ctrl = p_ctrl[:, 3:4]

        a_total_ctrl = torch.clamp(a1_ctrl + a2_ctrl, min=eps)
        fast_frac_ctrl = torch.clamp(a1_ctrl / a_total_ctrl, min=0.05, max=0.95)

        # Tail-unlocking catalyst benefit should be strongest in primary/passivating ores.
        tail_unlock = torch.clamp(
            0.45 * primary_catalyst_synergy
            + 0.35 * depassivation_strength
            + 0.20 * ferric_synergy,
            min=0.05,
            max=2.00,
        )

        # Easy ores should need less catalyst uplift.
        easy_inventory = torch.clamp(
            0.55 * fast_leach_inventory + 0.45 * oxide_inventory,
            min=0.0,
            max=1.0,
        )

        # ---- delta A_total: non-negative, mostly tail-unlocking, reduced for easy ores
        delta_a_frac = torch.sigmoid(raw_delta[:, 0:1]) * (
            0.06 + 0.34 * tail_unlock
        ) * (1.00 - 0.45 * easy_inventory)
        a_total_cat = a_total_ctrl * (1.0 + delta_a_frac)

        # ---- delta fast fraction: can move up or down a bit
        # Keep this modest because catalyst should mostly unlock extra recovery,
        # not arbitrarily reshuffle the whole curve.
        fast_bias = 0.50 * easy_inventory - 0.35 * tail_unlock
        delta_fast_frac = 0.12 * torch.tanh(raw_delta[:, 1:2]) + 0.06 * fast_bias
        fast_frac_cat = torch.clamp(
            fast_frac_ctrl + delta_fast_frac,
            min=0.05,
            max=0.95,
        )

        # ---- delta b1: signed; catalyst may speed up or slow down the fast branch
        delta_log_b1 = 0.45 * torch.tanh(raw_delta[:, 2:3]) * (
            0.35 + 0.65 * easy_inventory
        )
        b1_cat = torch.clamp(
            b1_ctrl * torch.exp(delta_log_b1),
            min=self.cat_lb[1].item(),
            max=self.cat_ub[1].item(),
        )

        # ---- delta b2: signed; catalyst may speed up or slow down the slow branch
        delta_log_b2 = 0.55 * torch.tanh(raw_delta[:, 3:4]) * (
            0.35 + 0.65 * tail_unlock
        ) * (1.00 - 0.10 * easy_inventory)
        b2_cat = torch.clamp(
            b2_ctrl * torch.exp(delta_log_b2),
            min=self.cat_lb[3].item(),
            max=self.cat_ub[3].item(),
        )

        a1_cat = torch.clamp(a_total_cat * fast_frac_cat, min=self.cat_lb[0].item(), max=self.cat_ub[0].item())
        a2_cat = torch.clamp(a_total_cat * (1.0 - fast_frac_cat), min=self.cat_lb[2].item(), max=self.cat_ub[2].item())

        p_cat = torch.cat([a1_cat, b1_cat, a2_cat, b2_cat], dim=1)
        return self._sanitize_params(p_cat)

    @staticmethod
    def _sum_copper_equivalents_torch(
        copper_primary_sulfides_equiv: torch.Tensor,
        copper_oxides_equiv: torch.Tensor,
        copper_secondary_sulfides_equiv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total = torch.zeros_like(copper_primary_sulfides_equiv)
        has_finite_value = torch.zeros_like(copper_primary_sulfides_equiv, dtype=torch.bool)
        for component in (
            copper_primary_sulfides_equiv,
            copper_oxides_equiv,
            copper_secondary_sulfides_equiv,
        ):
            if component is None:
                continue
            finite_mask = torch.isfinite(component)
            has_finite_value = has_finite_value | finite_mask
            total = total + torch.where(
                finite_mask,
                torch.clamp(component, min=0.0),
                torch.zeros_like(component),
            )
        return torch.where(has_finite_value, total, torch.full_like(total, float("nan")))

    @staticmethod
    def _compute_remaining_ore_factor_chemistry_based(
        y_ctrl: torch.Tensor,
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
            pct_pri = float(CONFIG.get("leach_pct_primary_sulfides_catalyzed", 0.75))
        else:
            pct_pri = float(CONFIG.get("leach_pct_primary_sulfides_control", 0.33))

        total_copper_equivalent = PairCurveNet._sum_copper_equivalents_torch(
            copper_primary_sulfides_equiv=copper_primary_sulfides_equiv,
            copper_oxides_equiv=copper_oxides_equiv,
            copper_secondary_sulfides_equiv=copper_secondary_sulfides_equiv,
        )
        cu_safe = torch.clamp(total_copper_equivalent, min=1e-6)

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
    def _compute_residual_primary_uplift_capacity(
        y_ctrl: torch.Tensor,
        copper_primary_sulfides_equiv: torch.Tensor,
        copper_oxides_equiv: torch.Tensor,
        copper_secondary_sulfides_equiv: Optional[torch.Tensor] = None,
        min_floor: float = 0.0,
        pct_pri_ctrl_override: Optional[torch.Tensor] = None,
        pct_pri_cat_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pct_ox = float(CONFIG.get("leach_pct_oxides", 1.00))
        pct_sec = float(CONFIG.get("leach_pct_secondary_sulfides", 0.67))
        pct_pri_ctrl_default = float(CONFIG.get("leach_pct_primary_sulfides_control", 0.33))
        pct_pri_cat_default = float(CONFIG.get("leach_pct_primary_sulfides_catalyzed", 0.75))

        total_copper_equivalent = PairCurveNet._sum_copper_equivalents_torch(
            copper_primary_sulfides_equiv=copper_primary_sulfides_equiv,
            copper_oxides_equiv=copper_oxides_equiv,
            copper_secondary_sulfides_equiv=copper_secondary_sulfides_equiv,
        )
        cu_safe = torch.clamp(total_copper_equivalent, min=1e-6)
        pri_safe = torch.clamp(copper_primary_sulfides_equiv, min=0.0)
        ox_safe = torch.clamp(copper_oxides_equiv, min=0.0)
        if copper_secondary_sulfides_equiv is None:
            sec_safe = torch.zeros_like(pri_safe)
        else:
            sec_safe = torch.clamp(copper_secondary_sulfides_equiv, min=0.0)

        # (B) Use per-sample primary-sulfide fractions if provided, else
        # fall back to the old scalar CONFIG constants.
        if pct_pri_ctrl_override is not None:
            pct_pri_ctrl_t = torch.clamp(pct_pri_ctrl_override, min=0.0, max=1.0)
        else:
            pct_pri_ctrl_t = torch.full_like(pri_safe, pct_pri_ctrl_default)
        if pct_pri_cat_override is not None:
            pct_pri_cat_t = torch.clamp(pct_pri_cat_override, min=0.0, max=1.0)
        else:
            pct_pri_cat_t = torch.full_like(pri_safe, pct_pri_cat_default)
        pct_pri_cat_t = torch.maximum(pct_pri_cat_t, pct_pri_ctrl_t)

        non_primary_capacity = 100.0 * (pct_ox * ox_safe + pct_sec * sec_safe) / cu_safe
        ctrl_primary_capacity = 100.0 * (pct_pri_ctrl_t * pri_safe) / cu_safe
        cat_extra_primary_capacity = 100.0 * torch.clamp(
            (pct_pri_cat_t - pct_pri_ctrl_t) * pri_safe / cu_safe, min=0.0
        )

        ctrl_primary_progress = torch.where(
            ctrl_primary_capacity > 1e-6,
            torch.clamp(
                (torch.clamp(y_ctrl, min=0.0) - non_primary_capacity) / torch.clamp(ctrl_primary_capacity, min=1e-6),
                min=0.0,
                max=1.0,
            ),
            torch.zeros_like(y_ctrl),
        )
        residual_primary_floor = float(
            max(
                min_floor,
                float(CONFIG.get("min_residual_primary_uplift_factor", 0.0)),
            )
        )
        residual_primary_softness_power = float(
            max(
                1e-6,
                float(CONFIG.get("residual_primary_uplift_softness_power", 1.0)),
            )
        )
        # Soften the depletion curve so the catalyst-side gap does not collapse
        # too abruptly once the control curve starts consuming primary capacity.
        residual_primary_factor = torch.pow(
            torch.clamp(1.0 - ctrl_primary_progress, min=0.0, max=1.0),
            residual_primary_softness_power,
        )
        residual_primary_factor = torch.clamp(
            residual_primary_factor,
            min=residual_primary_floor,
            max=1.0,
        )
        residual_primary_uplift_capacity = cat_extra_primary_capacity * residual_primary_factor
        return residual_primary_uplift_capacity, residual_primary_factor, cat_extra_primary_capacity

    def _compute_material_size_rate_factor_torch(
        self, material_size_p80_in: torch.Tensor
    ) -> torch.Tensor:
        """(A) Monotonically-decreasing kinetic scaling vs. P80.

        Returns a (B,1) factor in (0, 1] that is multiplied into the overall
        rate multiplier of both the control and catalyzed curves. The factor
        equals 1.0 at or below p80_ref (fine material) and decreases as
        (p80_ref / p80) ** alpha for coarser material. alpha is learnable but
        bounded in [0.3, 1.5] so it cannot vanish or explode.
        """
        p80_ref_in = float(CONFIG.get("p80_rate_ref_in", 1.0))
        p80 = torch.clamp(material_size_p80_in, min=1e-6)
        # Map raw parameter → alpha in [0.30, 1.50] via sigmoid.
        alpha = 0.30 + 1.20 * torch.sigmoid(self.p80_rate_alpha_raw)
        ratio = p80_ref_in / torch.clamp(p80, min=p80_ref_in)
        factor = torch.pow(ratio, alpha)
        return torch.clamp(factor, min=1e-3, max=1.0)

    @staticmethod
    def _compute_material_size_p80_cap_penalty_torch(material_size_p80_in: torch.Tensor) -> torch.Tensor:
        p80_d0_in = float(CONFIG.get("cap_p80_penalty_d0_in", 2.0))
        p80_d50_in = float(CONFIG.get("cap_p80_penalty_d50_in", 3.0))
        p80_p_inf = float(CONFIG.get("cap_p80_penalty_p_inf", 0.40))
        p80_n = float(CONFIG.get("cap_p80_penalty_n", 2.0))

        p80 = torch.clamp(material_size_p80_in, min=0.0)
        above_d0 = torch.clamp(p80 - p80_d0_in, min=0.0)
        span = max(p80_d50_in - p80_d0_in, 1e-6)
        x = torch.pow(above_d0 / span, p80_n)
        hill = x / (1.0 + x)
        penalty = 1.0 - (1.0 - p80_p_inf) * hill
        return torch.clamp(penalty, min=p80_p_inf, max=1.0)

    @staticmethod
    def _compute_sample_leach_caps_torch(
        copper_primary_sulfides_equiv: torch.Tensor,
        copper_oxides_equiv: torch.Tensor,
        copper_secondary_sulfides_equiv: Optional[torch.Tensor] = None,
        material_size_p80_in: Optional[torch.Tensor] = None,
        pct_pri_ctrl_override: Optional[torch.Tensor] = None,
        pct_pri_cat_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chemistry + P80 leach caps.

        (B) When `pct_pri_ctrl_override` and/or `pct_pri_cat_override` are
        provided, they replace the scalar CONFIG constants for the PRIMARY
        sulfide fraction. They are expected to be (B,1) tensors in [0,1].
        """
        pct_ox = float(CONFIG.get("leach_pct_oxides", 1.00))
        pct_sec = float(CONFIG.get("leach_pct_secondary_sulfides", 0.70))
        pct_pri_ctrl_default = float(CONFIG.get("leach_pct_primary_sulfides_control", 0.33))
        pct_pri_cat_default = float(CONFIG.get("leach_pct_primary_sulfides_catalyzed", 0.75))

        pri = torch.clamp(copper_primary_sulfides_equiv, min=0.0)
        ox = torch.clamp(copper_oxides_equiv, min=0.0)
        if copper_secondary_sulfides_equiv is None:
            sec = torch.zeros_like(pri)
        else:
            sec = torch.clamp(copper_secondary_sulfides_equiv, min=0.0)

        total_copper_equivalent = PairCurveNet._sum_copper_equivalents_torch(
            copper_primary_sulfides_equiv=pri,
            copper_oxides_equiv=ox,
            copper_secondary_sulfides_equiv=sec,
        )
        valid_total_copper = torch.isfinite(total_copper_equivalent) & (total_copper_equivalent > 1e-9)
        cu_safe = torch.clamp(total_copper_equivalent, min=1e-6)

        if pct_pri_ctrl_override is not None:
            pct_pri_ctrl_t = torch.clamp(pct_pri_ctrl_override, min=0.0, max=1.0)
        else:
            pct_pri_ctrl_t = torch.full_like(pri, pct_pri_ctrl_default)
        if pct_pri_cat_override is not None:
            pct_pri_cat_t = torch.clamp(pct_pri_cat_override, min=0.0, max=1.0)
        else:
            pct_pri_cat_t = torch.full_like(pri, pct_pri_cat_default)
        # Enforce catalyzed >= control numerically in case the incoming
        # tensors violate it slightly after broadcasting.
        pct_pri_cat_t = torch.maximum(pct_pri_cat_t, pct_pri_ctrl_t)

        ctrl_cap = 100.0 * (pct_ox * ox + pct_sec * sec + pct_pri_ctrl_t * pri) / cu_safe
        cat_cap = 100.0 * (pct_ox * ox + pct_sec * sec + pct_pri_cat_t * pri) / cu_safe
        ctrl_cap = torch.clamp(ctrl_cap, min=0.0)
        cat_cap = torch.clamp(cat_cap, min=0.0)

        if material_size_p80_in is not None:
            p80_factor = PairCurveNet._compute_material_size_p80_cap_penalty_torch(material_size_p80_in)
            ctrl_cap = ctrl_cap * p80_factor
            cat_cap = cat_cap * p80_factor

        nan_fill = torch.full_like(ctrl_cap, float("nan"))
        ctrl_cap = torch.where(valid_total_copper, ctrl_cap, nan_fill)
        cat_cap = torch.where(valid_total_copper, cat_cap, nan_fill)
        return ctrl_cap, cat_cap

    @staticmethod
    def _replace_with_finite_anchor_torch(
        cap: torch.Tensor,
        anchor: Optional[torch.Tensor],
        active: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if anchor is None:
            return cap
        anchor_t = torch.as_tensor(anchor, dtype=cap.dtype, device=cap.device)
        if anchor_t.ndim == 0:
            anchor_t = anchor_t.view(1, 1)
        elif anchor_t.ndim == 1:
            anchor_t = anchor_t.view(-1, 1)
        if anchor_t.shape[0] != cap.shape[0]:
            if anchor_t.shape[0] == 1:
                anchor_t = anchor_t.expand(cap.shape[0], -1)
            else:
                anchor_t = anchor_t[:1].expand(cap.shape[0], -1)
        if active is None:
            active_t = torch.ones_like(anchor_t)
        else:
            active_t = torch.as_tensor(active, dtype=cap.dtype, device=cap.device)
            if active_t.ndim == 0:
                active_t = active_t.view(1, 1)
            elif active_t.ndim == 1:
                active_t = active_t.view(-1, 1)
            if active_t.shape[0] != cap.shape[0]:
                if active_t.shape[0] == 1:
                    active_t = active_t.expand(cap.shape[0], -1)
                else:
                    active_t = active_t[:1].expand(cap.shape[0], -1)
        valid_anchor = torch.isfinite(anchor_t) & (active_t > 0.5)
        return torch.where(valid_anchor, anchor_t, cap)

    @staticmethod
    def _soft_upper_bound_torch(values: torch.Tensor, upper: torch.Tensor, softness: float) -> torch.Tensor:
        upper_t = torch.as_tensor(upper, dtype=values.dtype, device=values.device)
        if upper_t.ndim < values.ndim:
            upper_t = upper_t.view(*upper_t.shape, *([1] * (values.ndim - upper_t.ndim)))
        if upper_t.shape != values.shape:
            upper_t = torch.broadcast_to(upper_t, values.shape)
        valid_upper = torch.isfinite(upper_t)
        upper_safe = torch.where(valid_upper, upper_t, torch.full_like(upper_t, 100.0))
        softness_t = torch.full_like(values, max(float(softness), 1e-6))
        excess = torch.clamp(values - upper_safe, min=0.0)
        softened = upper_safe + softness_t * (1.0 - torch.exp(-excess / softness_t))
        return torch.where(valid_upper, torch.where(values > upper_safe, softened, values), values)
    
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
    def _sigmoid_gate_torch(
        t_days: torch.Tensor,
        gate_mid_day: torch.Tensor,
        gate_width_day: torch.Tensor,
    ) -> torch.Tensor:
        t = torch.clamp(torch.as_tensor(t_days), min=0.0)
        if t.ndim == 1:
            t = t.view(1, -1)
        batch_size = int(gate_mid_day.shape[0])
        if t.shape[0] != batch_size:
            if t.shape[0] == 1:
                t = t.expand(batch_size, -1)
            else:
                t = t[:1].expand(batch_size, -1)
        mid = torch.as_tensor(gate_mid_day, dtype=t.dtype, device=t.device)
        width = torch.clamp(torch.as_tensor(gate_width_day, dtype=t.dtype, device=t.device), min=1e-6)
        if mid.ndim == 1:
            mid = mid.view(-1, 1)
        if width.ndim == 1:
            width = width.view(-1, 1)
        z = torch.clamp((t - mid) / width, min=-60.0, max=60.0)
        raw = torch.sigmoid(z)
        raw0 = torch.sigmoid(torch.clamp((torch.zeros_like(mid) - mid) / width, min=-60.0, max=60.0))
        gate = (raw - raw0) / torch.clamp(1.0 - raw0, min=1e-9)
        return torch.clamp(gate, min=0.0, max=1.0)

    @staticmethod
    def _shift_curve_by_gate_mid_torch(
        values: torch.Tensor,   # (B, T)
        t_days: torch.Tensor,   # (1, T) or (B, T)
        gate_mid_day: torch.Tensor,  # (B, 1)
    ) -> torch.Tensor:
        """
        Residence-time-delay interpolation.

        For each batch sample b, return ``values[b]`` re-evaluated at the
        effective time ``t_eff[b, i] = max(t[b, i] - gate_mid[b], 0)`` via
        differentiable piecewise-linear interpolation over the pre-computed
        time grid.  This makes b1/b2 describe kinetics *after* the induction
        period rather than from t = 0.
        """
        B = values.shape[0]
        T = values.shape[1]

        # Expand t to (B, T) if it was broadcast as (1, T)
        if t_days.shape[0] != B:
            t = t_days.expand(B, T).contiguous()
        else:
            t = t_days.contiguous()

        # gate_mid: (B, 1)
        mid = gate_mid_day.view(B, 1)

        # t_eff[b, i] = max(t[b, i] - mid[b], 0)
        t_eff = torch.clamp(t - mid, min=0.0)  # (B, T)

        # For each row b, interpolate values[b] at t_eff[b, :] using the
        # sorted grid t[b, :].  torch.searchsorted operates row-wise when
        # both inputs are (B, T).
        idx_hi = torch.searchsorted(t, t_eff.contiguous())   # (B, T), right boundary
        idx_hi = torch.clamp(idx_hi, 1, T - 1)
        idx_lo = idx_hi - 1                                   # (B, T), left boundary

        t_lo = torch.gather(t,      1, idx_lo)  # (B, T)
        t_hi = torch.gather(t,      1, idx_hi)  # (B, T)
        v_lo = torch.gather(values, 1, idx_lo)  # (B, T)
        v_hi = torch.gather(values, 1, idx_hi)  # (B, T)

        dt = torch.clamp(t_hi - t_lo, min=1e-6)
        w  = torch.clamp((t_eff - t_lo) / dt, min=0.0, max=1.0)  # (B, T)

        return v_lo + w * (v_hi - v_lo)  # (B, T)

    @staticmethod
    def _apply_initial_stage_gate_torch(
        values: torch.Tensor,
        t_days: torch.Tensor,
        gate_strength: torch.Tensor,
        gate_mid_day: torch.Tensor,
        gate_width_day: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the residence-time-delay sigmoid gate.

        ``strength = 0``  →  original curve (no gate, kinetics from t = 0).
        ``strength = 1``  →  full residence-time-delay: the kinetics run from
                             ``t_eff = max(t − gate_mid, 0)`` and are weighted
                             by the sigmoid gate that opens at ``gate_mid``.

        The blended formula is::

            output = (1 − strength) × values
                   + strength       × gate(t) × values_shifted(t − gate_mid)

        where ``values_shifted`` is the same curve re-evaluated at the
        shifted time via differentiable linear interpolation.
        """
        gate = PairCurveNet._sigmoid_gate_torch(t_days, gate_mid_day, gate_width_day)
        strength = torch.clamp(
            torch.as_tensor(gate_strength, dtype=values.dtype, device=values.device),
            min=0.0, max=1.0,
        )
        if strength.ndim == 1:
            strength = strength.view(-1, 1)

        # Shift the curve so kinetics are anchored to the gate onset
        values_shifted = PairCurveNet._shift_curve_by_gate_mid_torch(
            values, t_days, gate_mid_day.view(-1, 1)
        )

        # Blend: no-gate path + gated residence-time-delay path
        return (1.0 - strength) * values + strength * gate * values_shifted

    @staticmethod
    def _active_catalyst_inventory_torch(
        cum_norm: torch.Tensor,
        t_days: torch.Tensor,
        aging_strength: torch.Tensor,
        mid_life_days: torch.Tensor,
        catalyst_use_frac: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the active catalyst inventory by performing a causal convolution
        of catalyst increments with an exponential decay kernel whose half-life
        is ``mid_life_days``.

        The total accumulated inventory at each time step is partitioned into:
          * **used catalyst** = total_inventory * catalyst_use_frac
          * **degraded catalyst** driven by aging_strength
        The **active catalyst** is the remainder:
          active = total_inventory - used - degraded

        Parameters
        ----------
        cum_norm : (B, T) cumulative normalised catalyst additions.
        t_days   : (B, T) or (1, T) time grid in days.
        aging_strength    : (B, 1) learned aging/usage magnitude, capped by
                            ``max_catalyst_aging_strength``.
        mid_life_days     : (B, 1) learned half-life of the catalyst in days
                            (clamped to [100, 800]).
        catalyst_use_frac : (B, 1) fraction of inventory consumed by the
                            reaction (0-1).

        Returns
        -------
        active_catalyst : (B, T) effective catalyst available for driving the
                          catalysed uplift.
        """
        c = torch.as_tensor(cum_norm, dtype=aging_strength.dtype, device=aging_strength.device)
        t = torch.as_tensor(t_days, dtype=aging_strength.dtype, device=aging_strength.device)
        c = torch.clamp(c, min=0.0)
        t = torch.clamp(t, min=0.0)
        if c.ndim == 1:
            c = c.view(1, -1)
        if t.ndim == 1:
            t = t.view(1, -1)
        batch_size = int(aging_strength.shape[0])
        if c.shape[0] != batch_size:
            if c.shape[0] == 1:
                c = c.expand(batch_size, -1)
            else:
                c = c[:1].expand(batch_size, -1)
        if t.shape[0] != batch_size:
            if t.shape[0] == 1:
                t = t.expand(batch_size, -1)
            else:
                t = t[:1].expand(batch_size, -1)

        if c.shape[1] == 0:
            return torch.zeros_like(c)
        if c.shape[1] == 1:
            return c.clone()

        # ---- Exponential decay kernel based on mid_life_days ----
        # decay_rate = ln(2) / mid_life_days  (half-life formula)
        half_life = torch.clamp(mid_life_days.view(-1, 1), min=100.0, max=800.0)
        decay_rate = 0.693147 / half_life  # ln(2) ≈ 0.693147

        dt = torch.clamp(t[:, 1:] - t[:, :-1], min=0.0)
        # Per-step survival factor: fraction of inventory surviving each interval
        step_survival = torch.exp(-decay_rate * dt)

        # Catalyst increments (additions between consecutive time steps)
        dc = torch.clamp(c[:, 1:] - c[:, :-1], min=0.0)

        # ---- Causal convolution: accumulate inventory with exponential decay ----
        # total_inventory[0] = c[:, 0]
        # total_inventory[i] = total_inventory[i-1] * step_survival[i-1] + dc[i-1]
        prefix_survival = torch.cumprod(step_survival, dim=1)
        prefix_survival_safe = torch.clamp(prefix_survival, min=1e-12)
        base = c[:, :1]
        tail = prefix_survival * (
            base + torch.cumsum(dc / prefix_survival_safe, dim=1)
        )
        total_inventory = torch.cat([base, tail], dim=1)

        # ---- Partition into used and degraded catalyst ----
        use_frac = torch.clamp(catalyst_use_frac.view(-1, 1), min=0.0, max=1.0)
        strength = torch.clamp(aging_strength.view(-1, 1), min=0.0)

        used_catalyst = total_inventory * use_frac
        # Degraded catalyst increases with aging_strength and elapsed time
        # relative to half_life; this captures chemical degradation beyond
        # the exponential decay already baked into the convolution kernel.
        elapsed_ratio = torch.clamp(t / half_life, min=0.0)
        degraded_catalyst = total_inventory * (
            strength / (1.0 + strength)
        ) * torch.clamp(1.0 - torch.exp(-elapsed_ratio), min=0.0, max=1.0)

        active_catalyst = torch.clamp(
            total_inventory - used_catalyst - degraded_catalyst,
            min=0.0,
        )
        active_floor_frac = float(np.clip(CONFIG.get("min_active_catalyst_inventory_frac", 0.0), 0.0, 1.0))
        if active_floor_frac > 0.0:
            active_catalyst = torch.maximum(active_catalyst, active_floor_frac * total_inventory)
        return active_catalyst

    @staticmethod
    def _summarize_active_catalyst_inventory_torch(active_catalyst: torch.Tensor) -> Dict[str, torch.Tensor]:
        active = torch.as_tensor(active_catalyst)
        if active.ndim == 1:
            active = active.view(1, -1)
        if active.shape[1] == 0:
            zeros = torch.zeros((active.shape[0], 1), dtype=active.dtype, device=active.device)
            return {
                "mean": zeros,
                "peak": zeros,
                "recent": zeros,
                "mean_log": zeros,
                "peak_log": zeros,
                "recent_log": zeros,
            }

        mean_active = torch.mean(torch.clamp(active, min=0.0), dim=1, keepdim=True)
        peak_active = torch.amax(torch.clamp(active, min=0.0), dim=1, keepdim=True)
        recent_active = torch.clamp(active[:, -1:], min=0.0)
        return {
            "mean": mean_active,
            "peak": peak_active,
            "recent": recent_active,
            "mean_log": torch.log1p(mean_active),
            "peak_log": torch.log1p(peak_active),
            "recent_log": torch.log1p(recent_active),
        }

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
        first = t[:, :1] * r[:, :1]
        if t.shape[1] == 1:
            return first
        dt = torch.clamp(t[:, 1:] - t[:, :-1], min=0.0)
        increments = 0.5 * dt * (r[:, 1:] + r[:, :-1])
        tail = first + torch.cumsum(increments, dim=1)
        return torch.cat([first, tail], dim=1)

    @staticmethod
    def _maybe_squeeze_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim > 0 and tensor.shape[0] == 1:
            return tensor.squeeze(0)
        return tensor

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
        if isinstance(t_days, torch.Tensor):
            base_device = t_days.device
            base_dtype = t_days.dtype
        elif isinstance(cum_norm, torch.Tensor):
            base_device = cum_norm.device
            base_dtype = cum_norm.dtype
        else:
            base_device = None
            base_dtype = torch.float32
        c = torch.as_tensor(cum_norm, dtype=base_dtype, device=base_device)
        t = torch.as_tensor(t_days, dtype=base_dtype, device=base_device)
        if c.ndim == 1:
            c = c.view(1, -1)
        if t.ndim == 1:
            t = t.view(1, -1)
        if t.shape[0] != c.shape[0]:
            if t.shape[0] == 1:
                t = t.expand(c.shape[0], -1)
            elif c.shape[0] == 1:
                c = c.expand(t.shape[0], -1)
        valid = torch.isfinite(c) & torch.isfinite(t)
        valid_counts = valid.sum(dim=1)

        c_clamped = torch.clamp(c, min=0.0)
        t_clamped = torch.clamp(t, min=0.0)
        large_time = torch.full_like(t_clamped, torch.finfo(t_clamped.dtype).max)
        t_sortable = torch.where(valid, t_clamped, large_time)
        order = torch.argsort(t_sortable, dim=1)

        c_sorted = torch.gather(torch.where(valid, c_clamped, torch.zeros_like(c_clamped)), 1, order)
        t_sorted = torch.gather(torch.where(valid, t_clamped, torch.zeros_like(t_clamped)), 1, order)
        valid_sorted = torch.gather(valid, 1, order)

        c_abs_max = torch.amax(torch.abs(c_sorted), dim=1)
        tol = torch.clamp(1e-6 * torch.maximum(c_abs_max, torch.ones_like(c_abs_max)), min=1e-9)

        starts = torch.zeros((c.shape[0],), dtype=t.dtype, device=t.device)
        first_positive = (valid_counts > 0) & (c_sorted[:, 0] > tol)
        starts = torch.where(first_positive, t_sorted[:, 0], starts)

        if c.shape[1] > 1:
            rise_mask = (
                (c_sorted[:, 1:] - c_sorted[:, :-1]) > tol.unsqueeze(1)
            ) & valid_sorted[:, 1:] & valid_sorted[:, :-1]
            has_rise = rise_mask.any(dim=1) & (~first_positive) & (valid_counts > 1)
            first_rise_idx = torch.argmax(rise_mask.to(dtype=torch.int64), dim=1)
            rise_times = torch.gather(t_sorted, 1, first_rise_idx.view(-1, 1)).squeeze(1)
            starts = torch.where(has_rise, rise_times, starts)

        return starts.view(-1, 1)

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
        if y.shape[1] == 1:
            return y

        dt = torch.clamp(t[:, 1:] - t[:, :-1], min=0.0)
        tau_i = torch.clamp(tau[:, 1:], min=1e-6)
        decay = torch.exp(-dt / tau_i)
        alpha = 1.0 - decay

        prefix_decay = torch.cumprod(decay, dim=1)
        prefix_decay_safe = torch.clamp(prefix_decay, min=1e-12)
        innovation = alpha * y[:, 1:]
        tail = prefix_decay * (
            y[:, :1] + torch.cumsum(innovation / prefix_decay_safe, dim=1)
        )
        return torch.cat([y[:, :1], tail], dim=1)

    @staticmethod
    def _evolve_catalyst_gap_torch(
        gap_capacity: torch.Tensor,
        activation: torch.Tensor,
        t_days: torch.Tensor,
        response_days: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cap = torch.as_tensor(gap_capacity)
        if cap.ndim == 1:
            cap = cap.view(1, -1)
        batch_size = int(cap.shape[0])
        act = PairCurveNet._expand_series_to_batch_torch(activation, batch_size, cap.dtype, cap.device)
        t = PairCurveNet._expand_series_to_batch_torch(t_days, batch_size, cap.dtype, cap.device)
        tau = PairCurveNet._expand_series_to_batch_torch(response_days, batch_size, cap.dtype, cap.device)
        if tau.shape[1] != cap.shape[1]:
            if tau.shape[1] == 1:
                tau = tau.expand(batch_size, cap.shape[1])
            else:
                tau = tau[:, : cap.shape[1]]

        act = torch.clamp(act, min=0.0, max=1.10)
        cap = torch.clamp(cap, min=0.0)
        target_gap = act * cap
        if cap.shape[1] == 0:
            return target_gap, target_gap
        realized_gap = PairCurveNet._causal_response_smooth_torch(
            target_gap,
            t_days=t,
            response_days=tau,
        )
        return target_gap, realized_gap

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
        ctrl_curve_override_raw: Optional[torch.Tensor] = None,
        cat_curve_override_raw: Optional[torch.Tensor] = None,
        ctrl_cap_anchor_pct: Optional[torch.Tensor] = None,
        cat_cap_anchor_pct: Optional[torch.Tensor] = None,
        ctrl_cap_anchor_active: Optional[torch.Tensor] = None,
        cat_cap_anchor_active: Optional[torch.Tensor] = None,
        catalyst_t_days: Optional[torch.Tensor] = None,
        catalyst_cum_norm: Optional[torch.Tensor] = None,
        catalyst_conc_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        x_static_encoder = x_static
        terminal_slope_idx = self.static_feature_indices.get("terminal_slope_rate")
        terminal_slope_encoder_scale = float(
            np.clip(CONFIG.get("terminal_slope_feature_encoder_scale", 1.0), 0.0, 1.0)
        )
        if (
            terminal_slope_idx is not None
            and terminal_slope_idx < x_static.shape[1]
            and terminal_slope_encoder_scale < 0.999
        ):
            # terminal_slope_rate is informative, but because it is derived from
            # the observed tail it should remain a soft anchor instead of
            # dominating the latent representation.
            x_static_encoder = x_static.clone()
            x_static_encoder[:, terminal_slope_idx : terminal_slope_idx + 1] = (
                x_static_encoder[:, terminal_slope_idx : terminal_slope_idx + 1]
                * terminal_slope_encoder_scale
            )

        h = self.encoder(x_static_encoder)
        p_ctrl = self._bounded_params(self.ctrl_head(h), self.ctrl_lb, self.ctrl_ub)
        p_ctrl = self._sanitize_params(p_ctrl)
        raw_cat_delta = self.cat_delta_head(h)
        initial_gate_raw = self.initial_gate_head(h)
        initial_gate_strength = torch.sigmoid(initial_gate_raw[:, 0:1])
        gate_mid_max = float(max(1.0, CONFIG.get("model_initial_gate_max_mid_day", 500.0)))
        gate_width_min = float(max(1e-3, CONFIG.get("model_initial_gate_min_width_days", 7.0)))
        gate_width_max = float(max(gate_width_min + 1e-3, CONFIG.get("model_initial_gate_max_width_days", 220.0)))
        initial_gate_mid_day = gate_mid_max * torch.sigmoid(initial_gate_raw[:, 1:2])
        initial_gate_width_day = gate_width_min + (gate_width_max - gate_width_min) * torch.sigmoid(initial_gate_raw[:, 2:3])

        # ---- Mineralogy-based gate onset additive shifts ----------------------
        # The encoder already sees mineralogy through the static feature vector,
        # but adding explicit physics-informed terms gives the model interpretable
        # inductive biases it can strengthen or attenuate during training.
        #
        # (1) Slow-Cu fraction: low fast-Cu (oxides + secondary sulfides) →
        #     longer onset delay before chalcopyrite leaching starts.
        # (2) Column height: taller column → lixiviant takes longer to condition
        #     the ore and establish effective leaching chemistry.
        # (3) Acid consumers (carbonates − 0.5×acid-gen sulfides): more buffering
        #     → more acid consumed before leach front advances → longer onset.
        if bool(CONFIG.get("nn_gate_onset_fast_cu_enabled", True)):
            # Resolve physical (unscaled) mineralogy features
            _x_phys_gate = x_static_raw if x_static_raw is not None else x_static
            _cu_prim_g = self._require_finite_feature_tensor(
                self._feature_column(_x_phys_gate, self.static_feature_indices.get("copper_primary_sulfides_equivalent")),
                "copper_primary_sulfides_equivalent",
            )
            _cu_sec_g = self._require_finite_feature_tensor(
                self._feature_column(_x_phys_gate, self.static_feature_indices.get("copper_secondary_sulfides_equivalent")),
                "copper_secondary_sulfides_equivalent",
            )
            _cu_ox_g = self._require_finite_feature_tensor(
                self._feature_column(_x_phys_gate, self.static_feature_indices.get("copper_oxides_equivalent")),
                "copper_oxides_equivalent",
            )
            _cu_prim_g = torch.clamp(_cu_prim_g, min=0.0)
            _cu_sec_g  = torch.clamp(_cu_sec_g,  min=0.0)
            _cu_ox_g   = torch.clamp(_cu_ox_g,   min=0.0)
            _total_cu_g = _cu_prim_g + _cu_sec_g + _cu_ox_g + 1e-6
            _fast_cu_g = (0.70 * _cu_ox_g + 0.30 * _cu_sec_g) / _total_cu_g
            _slow_cu_g = torch.clamp(1.0 - _fast_cu_g, min=0.0, max=1.0)
            # slow_cu_frac ∈ [0, 1] — high when mostly chalcopyrite
            _gate_cu_shift = (
                gate_mid_max
                * _slow_cu_g
                * F.softplus(self.gate_onset_fast_cu_scale_raw)
            )
            initial_gate_mid_day = initial_gate_mid_day + _gate_cu_shift

        if bool(CONFIG.get("nn_gate_onset_height_enabled", True)) and x_input_only_raw is not None:
            _height_idx_g = self.input_only_indices.get("column_height_m")
            if _height_idx_g is not None and _height_idx_g < x_input_only_raw.shape[1]:
                _h_g = torch.clamp(x_input_only_raw[:, _height_idx_g : _height_idx_g + 1], min=0.0)
                _gate_height_shift = F.softplus(self.gate_onset_height_scale_raw) * torch.log1p(_h_g)
                initial_gate_mid_day = initial_gate_mid_day + _gate_height_shift

        if bool(CONFIG.get("nn_gate_onset_acid_enabled", True)):
            _carb_idx_g = self.static_feature_indices.get("grouped_carbonates")
            _ags_idx_g  = self.static_feature_indices.get("grouped_acid_generating_sulfides")
            _x_phys_acid = x_static_raw if x_static_raw is not None else x_static
            if _carb_idx_g is not None:
                _carbonates_g = torch.clamp(
                    self._feature_column(_x_phys_acid, _carb_idx_g), min=0.0
                )
                _acid_gen_g = torch.zeros_like(_carbonates_g)
                if _ags_idx_g is not None:
                    _acid_gen_g = torch.clamp(
                        self._feature_column(_x_phys_acid, _ags_idx_g), min=0.0
                    )
                # Net acid consumer: normalised to [0, 1] over a 10-unit scale
                _acid_consumer_g = torch.clamp((_carbonates_g - 0.5 * _acid_gen_g) / 10.0, min=0.0, max=1.0)
                _gate_acid_shift = (
                    gate_mid_max
                    * _acid_consumer_g
                    * F.softplus(self.gate_onset_acid_scale_raw)
                )
                initial_gate_mid_day = initial_gate_mid_day + _gate_acid_shift

        # Also modulate gate_strength: when fast-Cu fraction is high (lots of oxide/
        # secondary Cu), the gate should be weaker because these phases dissolve from
        # day 1 and produce early recovery without an induction period.
        # gate_strength_mineralogy = gate_strength * (1 - fast_cu_frac * attenuation_factor)
        if bool(CONFIG.get("nn_gate_onset_fast_cu_enabled", True)):
            _fast_cu_attenuation = float(CONFIG.get("nn_gate_strength_fast_cu_attenuation", 0.80))
            # fast_cu_g is already computed above; shape (B,1)
            _strength_scale = torch.clamp(1.0 - _fast_cu_attenuation * _fast_cu_g, min=0.05, max=1.0)
            initial_gate_strength = initial_gate_strength * _strength_scale

        # Clamp final gate_mid_day to a physically meaningful range
        initial_gate_mid_day = torch.clamp(initial_gate_mid_day, min=0.0, max=gate_mid_max * 2.0)
        # -----------------------------------------------------------------------

        tau_raw = self.delay_head(h)
        if self.geo_delay_head is not None:
            geo = x_static[:, self.geo_idx]
            tau_raw = tau_raw + self.geo_delay_head(geo)
        if x_input_only_raw is not None:
            height_idx = self.input_only_indices.get("column_height_m")
            if height_idx is not None and height_idx < x_input_only_raw.shape[1]:
                height_raw = self._require_finite_feature_tensor(
                    x_input_only_raw[:, height_idx : height_idx + 1],
                    "column_height_m",
                )
                height_raw = torch.clamp(height_raw, min=0.0)
                tau_raw = tau_raw + F.softplus(self.height_delay_scale) * torch.log1p(height_raw)
        temp_raw = self.temp_head(h)
        lix_kappa = 1e-3 + F.softplus(self.lix_kappa_head(h))
        lix_strength = 0.25 + 1.50 * torch.sigmoid(self.lix_strength_head(h))
        interaction_terms = self._static_feature_map(x_static, x_static_raw, x_input_only_raw)

        cu_prim_idx = self.static_feature_indices.get("copper_primary_sulfides_equivalent")
        cu_sec_idx = self.static_feature_indices.get("copper_secondary_sulfides_equivalent")
        cu_oxide_idx = self.static_feature_indices.get("copper_oxides_equivalent")
        mat_size_idx = self.static_feature_indices.get("material_size_p80_in")

        batch_size = x_static.shape[0]
        ones = torch.ones((batch_size, 1), dtype=x_static.dtype, device=x_static.device)
        zeros = torch.zeros_like(ones)

        # Physical chemistry terms must stay on their native scale. Using the
        # standardized static vector here makes uplift caps fold-dependent
        # because each CV member fits a different scaler.
        x_static_physical = x_static_raw if x_static_raw is not None else x_static
        if x_input_only_raw is None:
            raise ValueError("Required input-only feature tensor is missing.")
        x_input_only_physical = x_input_only_raw

        cu_primary = self._require_finite_feature_tensor(
            self._feature_column(x_static_physical, cu_prim_idx),
            "copper_primary_sulfides_equivalent",
        )
        cu_secondary = self._require_finite_feature_tensor(
            self._feature_column(x_static_physical, cu_sec_idx),
            "copper_secondary_sulfides_equivalent",
        )
        cu_oxides = self._require_finite_feature_tensor(
            self._feature_column(x_static_physical, cu_oxide_idx),
            "copper_oxides_equivalent",
        )
        material_size_raw = self._require_finite_feature_tensor(
            self._feature_column(x_static_physical, mat_size_idx),
            "material_size_p80_in",
        )
        p80_log = torch.log1p(torch.clamp(material_size_raw, min=0.0))
        tau_raw = tau_raw + F.softplus(self.p80_tau_scale) * p80_log
        temp_raw = temp_raw + F.softplus(self.p80_temp_scale) * p80_log

        def _interaction_tensor_or_default(name: str, default: float = 0.0) -> torch.Tensor:
            tensor = interaction_terms.get(name)
            if tensor is None:
                return torch.full_like(ones, float(default))
            tensor = torch.as_tensor(tensor, dtype=x_static.dtype, device=x_static.device)
            default_tensor = torch.full_like(tensor, float(default))
            return torch.where(torch.isfinite(tensor), tensor, default_tensor)

        def _curve_override_tensor_or_default(
            override_tensor: Optional[torch.Tensor],
            name: str,
            default: float = 0.0,
        ) -> torch.Tensor:
            fallback = _interaction_tensor_or_default(name, default)
            if override_tensor is None:
                return fallback
            idx = CURVE_SPECIFIC_STATIC_OVERRIDE_INDEX.get(name)
            if idx is None or idx >= override_tensor.shape[1]:
                return fallback
            override = torch.as_tensor(
                override_tensor[:, idx : idx + 1],
                dtype=x_static.dtype,
                device=x_static.device,
            )
            return torch.where(torch.isfinite(override), override, fallback)

        def _active_mask_tensor_or_zero(mask_tensor: Optional[torch.Tensor]) -> torch.Tensor:
            if mask_tensor is None:
                return torch.zeros_like(ones)
            mask_t = torch.as_tensor(mask_tensor, dtype=x_static.dtype, device=x_static.device)
            if mask_t.ndim == 0:
                mask_t = mask_t.view(1, 1)
            elif mask_t.ndim == 1:
                mask_t = mask_t.view(-1, 1)
            if mask_t.shape[0] != batch_size:
                if mask_t.shape[0] == 1:
                    mask_t = mask_t.expand(batch_size, -1)
                else:
                    mask_t = mask_t[:1].expand(batch_size, -1)
            return torch.where(torch.isfinite(mask_t), torch.clamp(mask_t, min=0.0, max=1.0), torch.zeros_like(mask_t))

        def _anchor_tensor_or_nan(
            anchor_tensor: Optional[torch.Tensor],
            active_tensor: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if anchor_tensor is None:
                return torch.full_like(ones, float("nan"))
            anchor_t = torch.as_tensor(anchor_tensor, dtype=x_static.dtype, device=x_static.device)
            if anchor_t.ndim == 0:
                anchor_t = anchor_t.view(1, 1)
            elif anchor_t.ndim == 1:
                anchor_t = anchor_t.view(-1, 1)
            if anchor_t.shape[0] != batch_size:
                if anchor_t.shape[0] == 1:
                    anchor_t = anchor_t.expand(batch_size, -1)
                else:
                    anchor_t = anchor_t[:1].expand(batch_size, -1)
            active_t = _active_mask_tensor_or_zero(active_tensor)
            active_anchor = (active_t > 0.5) & torch.isfinite(anchor_t)
            return torch.where(active_anchor, anchor_t, torch.full_like(anchor_t, float("nan")))

        ctrl_anchor_active_t = _active_mask_tensor_or_zero(ctrl_cap_anchor_active)
        cat_anchor_active_t = _active_mask_tensor_or_zero(cat_cap_anchor_active)

        def _signal_score_torch(
            tensor: torch.Tensor,
            center: float,
            scale: float,
            *,
            invert: bool = False,
            log_input: bool = False,
        ) -> torch.Tensor:
            value_t = torch.as_tensor(tensor, dtype=x_static.dtype, device=x_static.device)
            value_t = torch.where(torch.isfinite(value_t), value_t, torch.full_like(value_t, float(center)))
            if log_input:
                value_t = torch.log1p(torch.clamp(value_t, min=0.0))
                center_value = math.log1p(max(float(center), 0.0))
            else:
                center_value = float(center)
            z = (value_t - center_value) / max(float(scale), 1e-6)
            if invert:
                z = -z
            return torch.sigmoid(z)

        ctrl_lixiviant_initial_fe = torch.clamp(
            _curve_override_tensor_or_default(ctrl_curve_override_raw, "lixiviant_initial_fe_mg_l"),
            min=0.0,
        )
        ctrl_lixiviant_initial_ph = torch.clamp(
            _curve_override_tensor_or_default(ctrl_curve_override_raw, "lixiviant_initial_ph", default=1.8),
            min=0.0,
        )
        ctrl_lixiviant_initial_orp = _curve_override_tensor_or_default(
            ctrl_curve_override_raw,
            "lixiviant_initial_orp_mv",
            default=440.0,
        )
        cat_lixiviant_initial_fe = torch.clamp(
            _curve_override_tensor_or_default(cat_curve_override_raw, "lixiviant_initial_fe_mg_l"),
            min=0.0,
        )
        cat_lixiviant_initial_ph = torch.clamp(
            _curve_override_tensor_or_default(cat_curve_override_raw, "lixiviant_initial_ph", default=1.8),
            min=0.0,
        )
        cat_lixiviant_initial_orp = _curve_override_tensor_or_default(
            cat_curve_override_raw,
            "lixiviant_initial_orp_mv",
            default=440.0,
        )
        control_terminal_slope_rate = torch.clamp(
            _curve_override_tensor_or_default(ctrl_curve_override_raw, "terminal_slope_rate"),
            min=-0.5,
            max=2.0,
        )
        catalyzed_terminal_slope_rate = torch.clamp(
            _curve_override_tensor_or_default(cat_curve_override_raw, "terminal_slope_rate"),
            min=-0.5,
            max=2.0,
        )
        acid_soluble_pct = torch.clamp(_interaction_tensor_or_default("acid_soluble_%"), min=0.0)
        cyanide_soluble_pct = torch.clamp(_interaction_tensor_or_default("cyanide_soluble_%"), min=0.0)
        acid_generating_sulfides_pct = torch.clamp(
            _interaction_tensor_or_default("grouped_acid_generating_sulfides"),
            min=0.0,
        )
        carbonates_pct = torch.clamp(_interaction_tensor_or_default("grouped_carbonates"), min=0.0)
        residual_cpy_pct = torch.clamp(_interaction_tensor_or_default("residual_cpy_%"), min=0.0)
        column_height_m = torch.clamp(_interaction_tensor_or_default("column_height_m"), min=0.0)

        ctrl_fe_signal = _signal_score_torch(
            ctrl_lixiviant_initial_fe,
            center=400.0,
            scale=0.80,
            log_input=True,
        )
        ctrl_orp_signal = _signal_score_torch(ctrl_lixiviant_initial_orp, center=440.0, scale=45.0)
        ctrl_ph_signal = _signal_score_torch(
            ctrl_lixiviant_initial_ph,
            center=1.8,
            scale=0.25,
            invert=True,
        )
        cat_fe_signal = _signal_score_torch(
            cat_lixiviant_initial_fe,
            center=400.0,
            scale=0.80,
            log_input=True,
        )
        cat_orp_signal = _signal_score_torch(cat_lixiviant_initial_orp, center=440.0, scale=45.0)
        cat_ph_signal = _signal_score_torch(
            cat_lixiviant_initial_ph,
            center=1.8,
            scale=0.25,
            invert=True,
        )
        cyanide_signal = _signal_score_torch(cyanide_soluble_pct, center=8.0, scale=4.0)
        acid_signal = _signal_score_torch(acid_soluble_pct, center=10.0, scale=6.0)
        acid_generating_signal = _signal_score_torch(
            acid_generating_sulfides_pct,
            center=2.5,
            scale=1.5,
        )
        carbonate_signal = _signal_score_torch(carbonates_pct, center=4.0, scale=2.0)
        secondary_signal = _signal_score_torch(cu_secondary, center=0.20, scale=0.10)
        oxide_signal = _signal_score_torch(cu_oxides, center=0.15, scale=0.08)
        primary_signal = _signal_score_torch(cu_primary, center=0.50, scale=0.20)
        residual_cpy_signal = _signal_score_torch(residual_cpy_pct, center=4.0, scale=1.25)
        height_signal = _signal_score_torch(column_height_m, center=4.0, scale=1.0)
        p80_signal = _signal_score_torch(material_size_raw, center=1.75, scale=0.50)
        tall_primary_burden = primary_signal * height_signal

        def _ferric_ready_primary_raw_from_signals(
            fe_signal: torch.Tensor,
            orp_signal: torch.Tensor,
            ph_signal: torch.Tensor,
        ) -> torch.Tensor:
            return (
                1.25 * (fe_signal - 0.5)
                + 0.95 * (orp_signal - 0.5)
                + 0.85 * (ph_signal - 0.5)
                + 0.55 * (cyanide_signal - 0.5)
                + 0.45 * (acid_signal - 0.5)
                + 0.30 * (acid_generating_signal - 0.5)
                + 0.20 * (secondary_signal - 0.5)
                - 0.55 * (height_signal - 0.5)
                - 0.50 * (p80_signal - 0.5)
                - 0.30 * (carbonate_signal - 0.5)
                - 0.35 * (tall_primary_burden - 0.10)
            )

        def _oxidant_deficit_raw_from_signals(
            fe_signal: torch.Tensor,
            orp_signal: torch.Tensor,
        ) -> torch.Tensor:
            return (
                1.20 * (height_signal - 0.5)
                + 1.00 * (p80_signal - 0.5)
                + 0.85 * (primary_signal - 0.5)
                + 0.80 * (residual_cpy_signal - 0.5)
                + 0.70 * ((1.0 - fe_signal) - 0.5)
                + 0.60 * ((1.0 - orp_signal) - 0.5)
                + 0.35 * ((1.0 - secondary_signal) - 0.5)
                + 0.20 * ((1.0 - oxide_signal) - 0.5)
                + 0.15 * (carbonate_signal - 0.5)
                + 0.75 * (tall_primary_burden - 0.10)
            )

        control_tail_flatness = torch.clamp((0.020 - control_terminal_slope_rate) / 0.015, min=0.0, max=1.0)
        catalyzed_tail_flatness = torch.clamp(
            (0.020 - catalyzed_terminal_slope_rate) / 0.015,
            min=0.0,
            max=1.0,
        )

        control_ferric_ready_primary_raw = _ferric_ready_primary_raw_from_signals(
            ctrl_fe_signal,
            ctrl_orp_signal,
            ctrl_ph_signal,
        )
        catalyzed_ferric_ready_primary_raw = _ferric_ready_primary_raw_from_signals(
            cat_fe_signal,
            cat_orp_signal,
            cat_ph_signal,
        )
        control_ferric_ready_primary_score = torch.sigmoid(1.25 * control_ferric_ready_primary_raw)
        catalyzed_ferric_ready_primary_score = torch.sigmoid(1.25 * catalyzed_ferric_ready_primary_raw)
        ferric_ready_primary_score = 0.5 * (
            control_ferric_ready_primary_score + catalyzed_ferric_ready_primary_score
        )

        control_oxidant_deficit_raw = _oxidant_deficit_raw_from_signals(
            ctrl_fe_signal,
            ctrl_orp_signal,
        )
        catalyzed_oxidant_deficit_raw = _oxidant_deficit_raw_from_signals(
            cat_fe_signal,
            cat_orp_signal,
        )
        control_oxidant_deficit_score = torch.sigmoid(1.20 * control_oxidant_deficit_raw)
        catalyzed_oxidant_deficit_score = torch.sigmoid(1.20 * catalyzed_oxidant_deficit_raw)
        tall_column_oxidant_deficit_score = 0.5 * (
            control_oxidant_deficit_score + catalyzed_oxidant_deficit_score
        )

        base_ctrl_prior_frac = torch.full_like(
            ones,
            float(CONFIG.get("leach_pct_primary_sulfides_control", 0.33)),
        )
        base_cat_prior_frac = torch.full_like(
            ones,
            float(CONFIG.get("leach_pct_primary_sulfides_catalyzed", 0.75)),
        )
        ctrl_prior_min = float(CONFIG.get("primary_control_prior_min", 0.20))
        ctrl_prior_max = float(CONFIG.get("primary_control_prior_max", 0.55))
        cat_prior_min = float(CONFIG.get("primary_catalyzed_prior_min", 0.60))
        cat_prior_max = float(CONFIG.get("primary_catalyzed_prior_max", 0.75))
        primary_control_prior_frac = torch.clamp(
            base_ctrl_prior_frac
            + 0.18 * (control_ferric_ready_primary_score - 0.5)
            - 0.16 * (control_oxidant_deficit_score - 0.5),
            min=ctrl_prior_min,
            max=ctrl_prior_max,
        )
        primary_control_prior_frac = torch.clamp(
            primary_control_prior_frac
            - 0.15 * (tall_primary_burden - 0.10)
            - 0.08 * control_tail_flatness,
            min=ctrl_prior_min,
            max=ctrl_prior_max,
        )
        primary_catalyzed_prior_frac = torch.clamp(
            base_cat_prior_frac
            + 0.02 * (catalyzed_ferric_ready_primary_score - 0.5)
            - 0.20 * (catalyzed_oxidant_deficit_score - 0.5),
            min=cat_prior_min,
            max=cat_prior_max,
        )
        primary_catalyzed_prior_frac = torch.clamp(
            primary_catalyzed_prior_frac
            - 0.10 * (tall_primary_burden - 0.10)
            - 0.10 * catalyzed_tail_flatness,
            min=cat_prior_min,
            max=cat_prior_max,
        )
        primary_catalyzed_prior_frac = torch.clamp(
            torch.maximum(
                primary_catalyzed_prior_frac,
                torch.maximum(primary_control_prior_frac, torch.full_like(primary_catalyzed_prior_frac, cat_prior_min)),
            ),
            min=cat_prior_min,
            max=cat_prior_max,
        )
        interaction_terms["ferric_ready_primary_score"] = ferric_ready_primary_score
        interaction_terms["tall_column_oxidant_deficit_score"] = tall_column_oxidant_deficit_score

        # ============================================================
        # Amax / overall recoverability latents
        # Intended support: fit_a1 + fit_a2
        # ============================================================
        primary_drive_raw = self._apply_learnable_interactions(
            self.primary_drive_head(h),
            "primary_passivation_drive",
            interaction_terms,
        )
        primary_passivation_drive = torch.sigmoid(primary_drive_raw)
        interaction_terms["primary_passivation_drive"] = primary_passivation_drive

        ferric_bootstrap_shared = 0.50 + torch.sigmoid(
            self._apply_learnable_interactions(
                self.ferric_bootstrap_head(h),
                "ferric_bootstrap",
                interaction_terms,
            )
        )
        control_ferric_bootstrap = torch.clamp(
            ferric_bootstrap_shared
            + 0.16 * (control_ferric_ready_primary_score - ferric_ready_primary_score)
            - 0.12 * control_tail_flatness,
            min=0.35,
            max=1.75,
        )
        catalyzed_ferric_bootstrap = torch.clamp(
            ferric_bootstrap_shared
            + 0.16 * (catalyzed_ferric_ready_primary_score - ferric_ready_primary_score)
            - 0.12 * catalyzed_tail_flatness,
            min=0.35,
            max=1.75,
        )
        ferric_bootstrap = 0.5 * (control_ferric_bootstrap + catalyzed_ferric_bootstrap)
        interaction_terms["ferric_bootstrap"] = ferric_bootstrap

        chem_interaction = 0.35 + 0.75 * torch.sigmoid(
            self._apply_learnable_interactions(
                self.chem_mix_head(h),
                "chem_interaction",
                interaction_terms,
            )
        )
        interaction_terms["chem_interaction"] = chem_interaction

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
        
        # ============================================================
        # Kinetics / transport / tail burden latents
        # Intended support: fit_b1, fit_b2, fit_a2
        # ============================================================
        ferric_synergy = 0.50 + torch.sigmoid(
            self._apply_learnable_interactions(
                self.ferric_synergy_head(h),
                "ferric_synergy",
                interaction_terms,
            )
        )
        interaction_terms["ferric_synergy"] = ferric_synergy

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

        # ============================================================
        # Catalyst-response latents
        # Intended support: catalyst-side fit_a2 and fit_b2 uplift
        # ============================================================
        primary_catalyst_synergy = 0.50 + torch.sigmoid(
            self._apply_learnable_interactions(
                self.primary_catalyst_synergy_head(h),
                "primary_catalyst_synergy",
                interaction_terms,
            )
        )
        interaction_terms["primary_catalyst_synergy"] = primary_catalyst_synergy

        depassivation_strength = 1.35 * torch.sigmoid(
            self._apply_learnable_interactions(
                self.depassivation_head(h),
                "depassivation_strength",
                interaction_terms,
            )
        )
        interaction_terms["depassivation_strength"] = depassivation_strength

        transform_strength = 0.70 * torch.sigmoid(
            self._apply_learnable_interactions(
                self.transform_head(h),
                "transform_strength",
                interaction_terms,
            )
        )
        interaction_terms["transform_strength"] = transform_strength

        passivation_tau = self.tmax_days * torch.sigmoid(self.passivation_tau_head(h))
        passivation_temp = self.min_transition_days + F.softplus(self.passivation_temp_head(h))

        p_cat = self._apply_catalyst_delta(
            p_ctrl=p_ctrl,
            raw_delta=raw_cat_delta,
            primary_catalyst_synergy=primary_catalyst_synergy,
            depassivation_strength=depassivation_strength,
            fast_leach_inventory=fast_leach_inventory,
            oxide_inventory=oxide_inventory,
            ferric_synergy=ferric_synergy,
        )

        transform_tau = self.tmax_days * torch.sigmoid(self.transform_tau_head(h))
        transform_temp = self.min_transition_days + F.softplus(self.transform_temp_head(h))

        # ---- Learnable tau_days via latent interactions ----
        # Intended support: catalyst response delay / accessibility timing
        tau_raw = self._apply_learnable_interactions(
            tau_raw,
            "tau_days",
            interaction_terms,
        )
        tau_days = 0.85 * self.tmax_days * torch.sigmoid(tau_raw)

        # ---- Learnable temp_days via latent interactions ----
        # Intended support: catalyst transition / smoothing timescale
        temp_raw = self._apply_learnable_interactions(
            temp_raw,
            "temp_days",
            interaction_terms,
        )

        # ---- Learnable kappa via latent interactions ----
        # Placed here so all chemistry/transport interaction_terms are available.
        kappa_raw = self.kappa_head(h) - F.softplus(self.p80_kappa_penalty_scale) * p80_log

        # ---- Learnable aging_strength via latent interactions ----
        aging_raw = self.aging_head(h)
        aging_raw = self._apply_learnable_interactions(
            aging_raw,
            "aging_strength",
            interaction_terms,
        )
        aging_strength = self.max_catalyst_aging_strength * torch.sigmoid(aging_raw)

        # ---- Learnable mid_life_days (100-800 day half-life) ----
        # Derived from the same aging interaction pathway: the mid_life_head
        # takes the aging_raw signal (which already encodes all the learned
        # chemistry/transport interactions for aging) and maps it to [100, 800].
        mid_life_days = 100.0 + 700.0 * torch.sigmoid(self.mid_life_head(aging_raw))

        # ---- Learnable catalyst_use_frac (0-1) ----
        catalyst_use_min = float(np.clip(CONFIG.get("min_catalyst_use_frac", 0.0), 0.0, 1.0))
        catalyst_use_max = float(np.clip(CONFIG.get("max_catalyst_use_frac", 1.0), catalyst_use_min, 1.0))
        catalyst_use_frac = catalyst_use_min + (catalyst_use_max - catalyst_use_min) * torch.sigmoid(
            self.use_frac_head(h)
        )

        active_catalyst_summary: Dict[str, torch.Tensor] = {}
        if catalyst_t_days is not None and catalyst_cum_norm is not None:
            provisional_active_catalyst = self._active_catalyst_inventory_torch(
                cum_norm=catalyst_cum_norm,
                t_days=catalyst_t_days,
                aging_strength=aging_strength,
                mid_life_days=mid_life_days,
                catalyst_use_frac=catalyst_use_frac,
            )
            active_catalyst_summary = self._summarize_active_catalyst_inventory_torch(
                provisional_active_catalyst
            )
            tau_raw = (
                tau_raw
                + self.tau_active_mean_scale * active_catalyst_summary["mean_log"]
                + self.tau_active_peak_scale * active_catalyst_summary["peak_log"]
                + self.tau_active_recent_scale * active_catalyst_summary["recent_log"]
            )
            temp_raw = (
                temp_raw
                + self.temp_active_mean_scale * active_catalyst_summary["mean_log"]
                + self.temp_active_peak_scale * active_catalyst_summary["peak_log"]
                + self.temp_active_recent_scale * active_catalyst_summary["recent_log"]
            )
            kappa_raw = (
                kappa_raw
                + self.kappa_active_mean_scale * active_catalyst_summary["mean_log"]
                + self.kappa_active_peak_scale * active_catalyst_summary["peak_log"]
                + self.kappa_active_recent_scale * active_catalyst_summary["recent_log"]
            )
            aging_raw = (
                aging_raw
                + self.aging_active_mean_scale * active_catalyst_summary["mean_log"]
                + self.aging_active_recent_scale * active_catalyst_summary["recent_log"]
            )
            active_drive_log = (
                0.35 * active_catalyst_summary["mean_log"]
                + 0.35 * active_catalyst_summary["peak_log"]
                + 0.30 * active_catalyst_summary["recent_log"]
            )
            tau_raw = tau_raw + float(CONFIG.get("active_catalyst_tau_fixed_scale", 0.0)) * active_drive_log
            kappa_raw = kappa_raw + float(CONFIG.get("active_catalyst_kappa_fixed_scale", 0.0)) * active_drive_log
            temp_raw = temp_raw + float(CONFIG.get("active_catalyst_temp_fixed_scale", 0.0)) * active_drive_log
            tau_days = 0.85 * self.tmax_days * torch.sigmoid(tau_raw)
            aging_strength = self.max_catalyst_aging_strength * torch.sigmoid(aging_raw)
            mid_life_days = 100.0 + 700.0 * torch.sigmoid(self.mid_life_head(aging_raw))

        # v11: CSTR concentration signal modulation.
        # catalyst_conc_norm is the normalised average pore-solution concentration
        # (mg/L / conc_scale).  A log-mean summary is computed and used to gate
        # tau/kappa/temp independently of the cumulative mass signal.
        if catalyst_conc_norm is not None and catalyst_conc_norm.numel() > 0:
            conc_mean_log = torch.log1p(
                torch.clamp(catalyst_conc_norm.mean(dim=-1, keepdim=True), min=0.0)
            )  # (B, 1) or (T,) depending on input shape
            if conc_mean_log.ndim == 1:
                conc_mean_log = conc_mean_log.unsqueeze(-1)
            # Reshape to match (B, 1) expected by tau_raw etc.
            if conc_mean_log.shape[0] != tau_raw.shape[0]:
                conc_mean_log = conc_mean_log.view(tau_raw.shape[0], 1)
            tau_raw = tau_raw + self.conc_tau_scale * conc_mean_log
            kappa_raw = kappa_raw + self.conc_kappa_scale * conc_mean_log
            temp_raw = temp_raw + self.conc_temp_scale * conc_mean_log
            tau_raw = tau_raw + float(CONFIG.get("catalyst_conc_tau_fixed_scale", 0.0)) * conc_mean_log
            kappa_raw = kappa_raw + float(CONFIG.get("catalyst_conc_kappa_fixed_scale", 0.0)) * conc_mean_log
            temp_raw = temp_raw + float(CONFIG.get("catalyst_conc_temp_fixed_scale", 0.0)) * conc_mean_log

        temp_days = self.min_transition_days + 0.85 * F.softplus(temp_raw)
        kappa = 1e-3 + F.softplus(kappa_raw)

        # Per-sample primary-sulfide fractions stay tightly anchored around the
        # deterministic metallurgy prior so cap flexibility remains narrow and
        # generalizable across new ores.
        primary_fraction_delta_max = float(CONFIG.get("primary_fraction_learned_delta_max", 0.05))
        cpy_ctrl_frac = torch.clamp(
            primary_control_prior_frac
            + primary_fraction_delta_max * torch.tanh(self.cpy_leach_frac_ctrl_head(h)),
            min=ctrl_prior_min,
            max=ctrl_prior_max,
        )
        cpy_cat_frac = torch.clamp(
            primary_catalyzed_prior_frac
            + primary_fraction_delta_max * torch.tanh(self.cpy_leach_frac_cat_head(h)),
            min=cat_prior_min,
            max=cat_prior_max,
        )
        cpy_cat_frac = torch.clamp(
            torch.maximum(cpy_cat_frac, torch.maximum(cpy_ctrl_frac, torch.full_like(cpy_cat_frac, cat_prior_min))),
            min=cat_prior_min,
            max=cat_prior_max,
        )

        base_ctrl_cap, base_cat_cap = self._compute_sample_leach_caps_torch(
            copper_primary_sulfides_equiv=cu_primary,
            copper_oxides_equiv=cu_oxides,
            copper_secondary_sulfides_equiv=cu_secondary,
            material_size_p80_in=material_size_raw,
            pct_pri_ctrl_override=base_ctrl_prior_frac,
            pct_pri_cat_override=base_cat_prior_frac,
        )
        effective_ctrl_cap_prior, effective_cat_cap_prior = self._compute_sample_leach_caps_torch(
            copper_primary_sulfides_equiv=cu_primary,
            copper_oxides_equiv=cu_oxides,
            copper_secondary_sulfides_equiv=cu_secondary,
            material_size_p80_in=material_size_raw,
            pct_pri_ctrl_override=primary_control_prior_frac,
            pct_pri_cat_override=primary_catalyzed_prior_frac,
        )
        base_ctrl_cap = self._replace_with_finite_anchor_torch(base_ctrl_cap, ctrl_cap_anchor_pct, ctrl_anchor_active_t)
        base_cat_cap = self._replace_with_finite_anchor_torch(base_cat_cap, cat_cap_anchor_pct, cat_anchor_active_t)
        effective_ctrl_cap_prior = self._replace_with_finite_anchor_torch(
            effective_ctrl_cap_prior,
            ctrl_cap_anchor_pct,
            ctrl_anchor_active_t,
        )
        effective_cat_cap_prior = self._replace_with_finite_anchor_torch(
            effective_cat_cap_prior,
            cat_cap_anchor_pct,
            cat_anchor_active_t,
        )

        # (E) Acid-consumption early-delay parameters (see __init__). These
        # are fed into curves_given_params via latent_params to produce a
        # signed deficit on the catalyzed curve early in time. The learnable
        # gate from the network is multiplied with a physical mineralogy
        # gate so samples with no carbonates and no acid-generating sulfides
        # cannot have any negative-uplift early deficit at all (this also
        # protects the global behaviour of normal samples while still
        # allowing fresh-ore samples like Toquepala fresca to learn it).
        ferric_bootstrap_centered = catalyzed_ferric_bootstrap - 1.0
        acid_amp_pp = 4.0 * torch.sigmoid(
            self.acid_consumption_amp_head(h) - 0.40 * ferric_bootstrap_centered
        )
        acid_tau_days = 15.0 + 150.0 * torch.sigmoid(self.acid_consumption_tau_head(h))
        acid_gate_raw = torch.sigmoid(
            self.acid_consumption_gate_head(h) - 0.75 * ferric_bootstrap_centered
        )
        carb_idx = self.static_feature_indices.get("grouped_carbonates")
        ags_idx = self.static_feature_indices.get("grouped_acid_generating_sulfides")
        carb_pct = (
            self._feature_column(x_static_physical, carb_idx)
            if carb_idx is not None
            else zeros
        )
        ags_pct = (
            self._feature_column(x_static_physical, ags_idx)
            if ags_idx is not None
            else zeros
        )
        carb_pct = torch.where(torch.isfinite(carb_pct), carb_pct, torch.zeros_like(carb_pct))
        ags_pct = torch.where(torch.isfinite(ags_pct), ags_pct, torch.zeros_like(ags_pct))
        # Mineralogy gate: 0 when both fractions are 0%, saturates to 1 once
        # the combined fraction exceeds ~5%. A 1% combined fraction gives
        # gate ~0.18; a 3% fraction gives ~0.55; a 5% fraction gives ~0.78.
        mineralogy_gate = 1.0 - torch.exp(-0.30 * (carb_pct + ags_pct))
        mineralogy_gate = torch.clamp(mineralogy_gate, min=0.0, max=1.0)
        acid_gate = acid_gate_raw * mineralogy_gate

        latent_params = {
            "chem_interaction": chem_interaction,
            "primary_passivation_drive": primary_passivation_drive,
            "ferric_bootstrap": ferric_bootstrap,
            "control_ferric_bootstrap": control_ferric_bootstrap,
            "catalyzed_ferric_bootstrap": catalyzed_ferric_bootstrap,
            "ferric_ready_primary_score": ferric_ready_primary_score,
            "tall_column_oxidant_deficit_score": tall_column_oxidant_deficit_score,
            "primary_catalyst_synergy": primary_catalyst_synergy,
            "cpy_leach_frac_cat": cpy_cat_frac,
            "cpy_leach_frac_ctrl": cpy_ctrl_frac,
            "base_ctrl_cap": base_ctrl_cap,
            "base_cat_cap": base_cat_cap,
            "effective_ctrl_cap_prior": effective_ctrl_cap_prior,
            "effective_cat_cap_prior": effective_cat_cap_prior,
            "primary_control_prior_frac": primary_control_prior_frac,
            "primary_catalyzed_prior_frac": primary_catalyzed_prior_frac,
            "control_terminal_slope_rate": control_terminal_slope_rate,
            "catalyzed_terminal_slope_rate": catalyzed_terminal_slope_rate,
            "control_cap_anchor_pct": _anchor_tensor_or_nan(ctrl_cap_anchor_pct, ctrl_anchor_active_t),
            "catalyzed_cap_anchor_pct": _anchor_tensor_or_nan(cat_cap_anchor_pct, cat_anchor_active_t),
            "control_cap_anchor_active": ctrl_anchor_active_t,
            "catalyzed_cap_anchor_active": cat_anchor_active_t,
            "acid_consumption_amp_pp": acid_amp_pp,
            "acid_consumption_tau_days": acid_tau_days,
            "acid_consumption_gate": acid_gate,
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
            "tau_days": tau_days,
            "temp_days": temp_days,
            "lixiviant_kappa": lix_kappa,
            "lixiviant_strength": lix_strength,
            "mid_life_days": mid_life_days,
            "catalyst_use_frac": catalyst_use_frac,
            "cu_primary": cu_primary,
            "cu_secondary": cu_secondary,
            "cu_oxides": cu_oxides,
            "material_size_p80_in": material_size_raw,
            "initial_gate_strength": initial_gate_strength,
            "initial_gate_mid_day": initial_gate_mid_day,
            "initial_gate_width_day": initial_gate_width_day,
        }
        latent_params.update(
            {
                "active_catalyst_mean": active_catalyst_summary.get("mean", zeros),
                "active_catalyst_peak": active_catalyst_summary.get("peak", zeros),
                "active_catalyst_recent": active_catalyst_summary.get("recent", zeros),
            }
        )
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
        catalyst_start_day_override: Optional[torch.Tensor] = None,
        latent_params: Optional[Dict[str, torch.Tensor]] = None,
        return_states: bool = False,
    ) -> Any:
        t_input = torch.as_tensor(t_days, dtype=p_ctrl.dtype, device=p_ctrl.device)
        squeeze_outputs = bool(t_input.ndim == 1 and int(p_ctrl.shape[0]) == 1)
        t = t_input
        if t.ndim == 1:
            t = t.view(1, -1)
        if latent_params is None:
            raise ValueError("latent_params must be provided; required chemistry and geometry terms cannot be inferred.")
        required_latent_keys = [
            # "total_copper_equivalent",
            # "cu_primary",
            # "cu_secondary",
            # "cu_oxides",
            # "material_size_p80_in",
            # "column_height_m",
            # "material_size_to_column_diameter_ratio",
        ]
        missing_latent_keys = [key for key in required_latent_keys if key not in latent_params]
        if missing_latent_keys:
            raise KeyError(
                f"latent_params is missing required keys: {', '.join(missing_latent_keys)}"
            )

        t_abs = t
        if catalyst_start_day_override is None:
            catalyst_start_day = self._infer_catalyst_start_day_torch(cum_norm, t_days)
        else:
            inferred_start_day = self._infer_catalyst_start_day_torch(cum_norm, t_days)
            catalyst_start_day = torch.as_tensor(
                catalyst_start_day_override,
                dtype=p_ctrl.dtype,
                device=p_ctrl.device,
            )
            if catalyst_start_day.ndim == 0:
                catalyst_start_day = catalyst_start_day.view(1, 1)
            elif catalyst_start_day.ndim == 1:
                if catalyst_start_day.shape[0] == int(p_ctrl.shape[0]):
                    catalyst_start_day = catalyst_start_day.view(-1, 1)
                else:
                    catalyst_start_day = catalyst_start_day.view(1, -1)
            if catalyst_start_day.shape[0] != int(p_ctrl.shape[0]):
                if catalyst_start_day.shape[0] == 1:
                    catalyst_start_day = catalyst_start_day.expand(int(p_ctrl.shape[0]), -1)
                else:
                    catalyst_start_day = catalyst_start_day[:1].expand(int(p_ctrl.shape[0]), -1)
            if catalyst_start_day.shape[1] != 1:
                catalyst_start_day = catalyst_start_day[:, :1]
            catalyst_start_day = torch.where(
                torch.isfinite(catalyst_start_day),
                catalyst_start_day,
                inferred_start_day.to(catalyst_start_day.device, catalyst_start_day.dtype),
            )
        catalyst_start_day = catalyst_start_day.to(p_ctrl.device, p_ctrl.dtype)
        catalyst_elapsed_days = torch.clamp(t_abs - catalyst_start_day, min=0.0)
        effective_tau_days = torch.clamp(catalyst_start_day + tau_days, min=0.0, max=self.tmax_days)

        # Start day should set when the catalyst response begins, not how much
        # uplift capacity remains available. Drive catalyst kinetics off
        # elapsed time since catalyst starts rather than absolute leach time.
        delay_factor = torch.sigmoid(
            (catalyst_elapsed_days - tau_days) / torch.clamp(temp_days, min=1e-6)
        )

        effective_catalyst = self._active_catalyst_inventory_torch(
            cum_norm=cum_norm,
            t_days=t_days,
            aging_strength=aging_strength,
            mid_life_days=latent_params["mid_life_days"],
            catalyst_use_frac=latent_params["catalyst_use_frac"],
        )
        catalyst_factor = 1.0 - torch.exp(-torch.clamp(kappa, min=1e-6) * effective_catalyst)
        '''
        dose = torch.clamp(effective_catalyst, min=0.0)
        catalyst_factor = 1.0 - torch.exp(
            -torch.clamp(kappa, min=1e-6) * dose / (1.0 + 2.0 * dose)
        )
        '''
        batch_size = int(p_ctrl.shape[0])
        cumulative_catalyst_signal = torch.clamp(
            self._expand_series_to_batch_torch(cum_norm, batch_size, p_ctrl.dtype, p_ctrl.device),
            min=0.0,
        )
        catalyst_has_reached_time = cumulative_catalyst_signal > 1e-9
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
        ferric_bootstrap = torch.clamp(
            latent_params.get("ferric_bootstrap", torch.ones_like(primary_drive)),
            min=0.35,
            max=1.75,
        )
        control_ferric_bootstrap = torch.clamp(
            latent_params.get("control_ferric_bootstrap", ferric_bootstrap),
            min=0.35,
            max=1.75,
        )
        catalyzed_ferric_bootstrap = torch.clamp(
            latent_params.get("catalyzed_ferric_bootstrap", ferric_bootstrap),
            min=0.35,
            max=1.75,
        )
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

        # v10: every catalyst-side mechanism rides on one shared smooth progress
        # signal instead of several independent turn-on times. That preserves
        # the delay/tau assumption while preventing separate uplift episodes.
        shared_uplift_progress = torch.clamp(delay_factor * catalyst_factor, min=0.0, max=1.0)
        shared_uplift_response_days = torch.clamp(
            0.45 * temp_days + 0.30 * passivation_temp + 0.25 * transform_temp,
            min=self.min_transition_days,
        )
        continuous_catalyst_drive = self._causal_response_smooth_torch(
            shared_uplift_progress,
            catalyst_elapsed_days,
            shared_uplift_response_days,
        )
        continuous_catalyst_drive = torch.cummax(
            torch.clamp(continuous_catalyst_drive, min=0.0, max=1.0),
            dim=1,
        ).values

        time_scale = max(self.tmax_days, 1e-6)
        base_ctrl_abs = self._double_exp_curve_torch(p_ctrl, t_abs)
        base_ctrl_catalyst_clock = self._double_exp_curve_torch(p_ctrl, catalyst_elapsed_days)

        ore_accessibility_abs = torch.exp(
            -ore_decay_strength * torch.clamp(base_ctrl_abs, 0.0, 100.0) / 100.0
        )
        ore_accessibility_abs = ore_accessibility_abs * torch.exp(
            -0.35 * ore_decay_strength * t_abs / time_scale
        )
        ore_accessibility_abs = torch.clamp(
            ore_accessibility_abs,
            min=self.min_remaining_ore_factor,
            max=1.0,
        )

        ore_accessibility = torch.exp(
            -ore_decay_strength * torch.clamp(base_ctrl_catalyst_clock, 0.0, 100.0) / 100.0
        )
        ore_accessibility = ore_accessibility * torch.exp(
            -0.35 * ore_decay_strength * catalyst_elapsed_days / time_scale
        )
        ore_accessibility = torch.clamp(
            ore_accessibility,
            min=self.min_remaining_ore_factor,
            max=1.0,
        )

        fast_release_abs = torch.clamp(
            (0.65 * fast_leach_inventory + 0.35 * oxide_inventory)
            * torch.exp(-1.25 * t_abs / time_scale)
            * (0.50 + 0.50 * ore_accessibility_abs),
            min=0.0,
            max=1.25,
        )
        fast_release_abs = torch.clamp(
            fast_release_abs * (0.80 + 0.35 * lix_availability),
            min=0.0,
            max=1.25,
        )

        fast_release = torch.clamp(
            (0.65 * fast_leach_inventory + 0.35 * oxide_inventory)
            * torch.exp(-1.25 * catalyst_elapsed_days / time_scale)
            * (0.50 + 0.50 * ore_accessibility),
            min=0.0,
            max=1.25,
        )
        fast_release = torch.clamp(
            fast_release * (0.80 + 0.35 * lix_availability),
            min=0.0,
            max=1.25,
        )

        acid_buffer_remaining_abs = torch.exp(-acid_buffer_decay * t_abs / time_scale)
        acid_buffer_penalty_abs = torch.clamp(
            acid_buffer_strength
            * acid_buffer_remaining_abs
            * torch.clamp(1.0 - 0.35 * lix_factor, min=0.35, max=1.0),
            min=0.0,
            max=0.95,
        )

        acid_buffer_remaining = torch.exp(-acid_buffer_decay * catalyst_elapsed_days / time_scale)
        acid_buffer_penalty = torch.clamp(
            acid_buffer_strength
            * acid_buffer_remaining
            * torch.clamp(1.0 - 0.35 * lix_factor, min=0.35, max=1.0),
            min=0.0,
            max=0.95,
        )

        diffusion_progress_abs = torch.sqrt(torch.clamp(t_abs / time_scale, min=0.0, max=1.0))
        diffusion_drag_abs = torch.clamp(
            diffusion_drag_strength
            * diffusion_progress_abs
            * (0.60 + 0.40 * primary_drive)
            * torch.clamp(1.0 - 0.30 * lix_factor, min=0.35, max=1.0),
            min=0.0,
            max=0.95,
        )

        diffusion_progress = torch.sqrt(
            torch.clamp(catalyst_elapsed_days / time_scale, min=0.0, max=1.0)
        )
        diffusion_drag = torch.clamp(
            diffusion_drag_strength
            * diffusion_progress
            * (0.60 + 0.40 * primary_drive)
            * torch.clamp(1.0 - 0.30 * lix_factor, min=0.35, max=1.0),
            min=0.0,
            max=0.95,
        )

        passivation_response_days = torch.clamp(
            0.35 * passivation_tau + 0.65 * passivation_temp,
            min=self.min_transition_days,
        )
        passivation_target_abs = torch.clamp(
            passivation_strength
            * primary_drive
            * chem_interaction
            * ferric_synergy
            * ore_accessibility_abs
            * (1.0 - 0.45 * fast_release_abs)
            * (1.0 + 0.20 * acid_buffer_penalty_abs),
            min=0.0,
            max=0.95,
        )
        passivation_abs = self._causal_response_smooth_torch(
            passivation_target_abs,
            t_abs,
            passivation_response_days,
        )
        passivation_abs = torch.clamp(passivation_abs, min=0.0, max=0.95)

        passivation_target = torch.clamp(
            passivation_strength
            * primary_drive
            * chem_interaction
            * (2.0 - ferric_synergy)
            * ore_accessibility
            * (1.0 - 0.45 * fast_release)
            * (1.0 + 0.20 * acid_buffer_penalty),
            min=0.0,
            max=0.95,
        )
        passivation_target = torch.clamp(
            passivation_target
            * torch.clamp(
                1.0 - 0.70 * continuous_catalyst_drive * surface_refresh * (0.55 + 0.45 * lix_availability),
                min=0.15,
                max=1.0,
            ),
            min=0.0,
            max=0.95,
        )
        passivation_raw = self._causal_response_smooth_torch(
            passivation_target,
            catalyst_elapsed_days,
            passivation_response_days,
        )
        passivation_raw = torch.clamp(passivation_raw, min=0.0, max=0.95)

        transform_response_days = torch.clamp(
            0.35 * transform_tau + 0.65 * transform_temp,
            min=self.min_transition_days,
        )
        transformation_target = torch.clamp(
            transform_strength
            * chem_interaction
            * (0.35 + 0.65 * ferric_synergy)
            * continuous_catalyst_drive
            * (0.45 + 0.55 * ore_accessibility),
            min=0.0,
            max=1.25,
        )
        transformation_target = torch.clamp(
            transformation_target * (0.50 + 0.50 * lix_availability),
            min=0.0,
            max=1.25,
        )
        transformation = self._causal_response_smooth_torch(
            transformation_target,
            catalyst_elapsed_days,
            transform_response_days,
        )
        transformation = torch.clamp(transformation, min=0.0, max=1.25)

        depassivation_response_days = torch.clamp(
            0.50 * passivation_response_days + 0.50 * transform_response_days,
            min=self.min_transition_days,
        )
        depassivation_target = torch.clamp(
            depassivation_strength
            * primary_drive
            * chem_interaction
            * ferric_synergy
            * surface_refresh
            * continuous_catalyst_drive
            * (0.35 + 0.65 * transformation),
            min=0.0,
            max=1.50,
        )
        depassivation_target = torch.clamp(
            depassivation_target * (0.45 + 0.55 * lix_availability),
            min=0.0,
            max=1.50,
        )
        depassivation = self._causal_response_smooth_torch(
            depassivation_target,
            catalyst_elapsed_days,
            depassivation_response_days,
        )
        depassivation = torch.clamp(depassivation, min=0.0, max=1.50)

        # Effective passivation is the continuously accumulated pressure minus
        # the continuously accumulated catalyst relief.
        passivation = torch.clamp(
            passivation_raw - 0.55 * depassivation,
            min=0.0,
            max=0.95,
        )

        ctrl_rate_multiplier_abs = torch.clamp(
            ore_accessibility_abs
            * (1.0 - passivation_abs)
            * (1.0 - 0.65 * acid_buffer_penalty_abs)
            * (1.0 - diffusion_drag_abs)
            * (0.85 + 0.35 * fast_release_abs),
            min=0.08,
            max=1.75,
        )
        ctrl_rate_multiplier_abs = torch.clamp(
            ctrl_rate_multiplier_abs * (0.80 + 0.35 * lix_availability),
            min=0.08,
            max=1.75,
        )
        control_fast_start_decay_days = max(45.0, min(250.0, 0.18 * time_scale))
        control_fast_start_boost = torch.clamp(
            1.0
            + 0.20 * (control_ferric_bootstrap - 1.0) * torch.exp(-torch.clamp(t_abs, min=0.0) / control_fast_start_decay_days),
            min=0.85,
            max=1.25,
        )
        ctrl_rate_multiplier_abs = torch.clamp(
            ctrl_rate_multiplier_abs * control_fast_start_boost,
            min=0.08,
            max=1.75,
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

        # Retrieve chemistry / geometry latents up front so the (A) P80
        # kinetic scaling factor below has access to material_size_p80_in.
        cu_primary = self._require_finite_feature_tensor(latent_params["cu_primary"], "cu_primary")
        cu_oxides = self._require_finite_feature_tensor(latent_params["cu_oxides"], "cu_oxides")
        cu_secondary = self._require_finite_feature_tensor(latent_params["cu_secondary"], "cu_secondary")
        material_size_p80_in = self._require_finite_feature_tensor(
            latent_params["material_size_p80_in"],
            "material_size_p80_in",
        )

        # (A) Apply the learnable P80 kinetic scaling to every rate
        # multiplier. This is the mechanism that makes 6" particles leach
        # faster than 8" particles at identical chemistry. The factor is
        # (B,1) → broadcasts to (B,T). Using the same factor for ctrl and
        # cat preserves their relative uplift but slows both proportionally.
        size_rate_factor = self._compute_material_size_rate_factor_torch(
            material_size_p80_in
        )
        ctrl_rate_multiplier_abs = torch.clamp(
            ctrl_rate_multiplier_abs * size_rate_factor, min=0.01, max=1.75
        )
        ctrl_rate_multiplier = torch.clamp(
            ctrl_rate_multiplier * size_rate_factor, min=0.01, max=1.75
        )
        cat_rate_multiplier = torch.clamp(
            cat_rate_multiplier * size_rate_factor, min=0.01, max=2.75
        )

        t_eff_ctrl_abs = self._effective_time_from_rate_torch(t_abs, ctrl_rate_multiplier_abs)
        t_eff_ctrl = self._effective_time_from_rate_torch(catalyst_elapsed_days, ctrl_rate_multiplier)
        t_eff_cat = self._effective_time_from_rate_torch(catalyst_elapsed_days, cat_rate_multiplier)

        y_ctrl = self._double_exp_curve_from_grid_torch(p_ctrl, t_eff_ctrl_abs)
        y_ctrl_catalyst_clock = self._double_exp_curve_from_grid_torch(p_ctrl, t_eff_ctrl)
        y_cat_base = self._double_exp_curve_from_grid_torch(p_cat, t_eff_cat)
        base_uplift = F.softplus(y_cat_base - y_ctrl_catalyst_clock)
        # Use the learned primary-sulfide fractions only inside the narrow
        # metallurgy-anchored band around the deterministic prior caps.
        pct_pri_cat_override = latent_params.get("cpy_leach_frac_cat")
        pct_pri_ctrl_override = latent_params.get("cpy_leach_frac_ctrl")
        ctrl_cap_curve, cat_cap_curve = self._compute_sample_leach_caps_torch(
            copper_primary_sulfides_equiv=cu_primary,
            copper_oxides_equiv=cu_oxides,
            copper_secondary_sulfides_equiv=cu_secondary,
            material_size_p80_in=material_size_p80_in,
            pct_pri_ctrl_override=pct_pri_ctrl_override,
            pct_pri_cat_override=pct_pri_cat_override,
        )
        ctrl_cap_curve = self._replace_with_finite_anchor_torch(
            ctrl_cap_curve,
            latent_params.get("control_cap_anchor_pct"),
            latent_params.get("control_cap_anchor_active"),
        )
        cat_cap_curve = self._replace_with_finite_anchor_torch(
            cat_cap_curve,
            latent_params.get("catalyzed_cap_anchor_pct"),
            latent_params.get("catalyzed_cap_anchor_active"),
        )
        residual_primary_uplift_capacity, residual_primary_factor, total_primary_uplift_capacity = (
            self._compute_residual_primary_uplift_capacity(
                y_ctrl=y_ctrl_catalyst_clock,
                copper_primary_sulfides_equiv=cu_primary,
                copper_oxides_equiv=cu_oxides,
                copper_secondary_sulfides_equiv=cu_secondary,
                min_floor=0.0,
                pct_pri_ctrl_override=pct_pri_ctrl_override,
                pct_pri_cat_override=pct_pri_cat_override,
            )
        )
        valid_cap_curve = torch.isfinite(cat_cap_curve) & torch.isfinite(ctrl_cap_curve)
        residual_primary_uplift_capacity = torch.where(
            valid_cap_curve,
            residual_primary_uplift_capacity,
            base_uplift,
        )
        residual_primary_factor = torch.where(
            valid_cap_curve,
            residual_primary_factor,
            torch.ones_like(residual_primary_factor),
        )
        total_primary_uplift_capacity = torch.where(
            valid_cap_curve,
            total_primary_uplift_capacity,
            base_uplift,
        )
        anchor_gap_capacity = torch.clamp(cat_cap_curve - torch.minimum(y_ctrl_catalyst_clock, ctrl_cap_curve), min=0.0)
        residual_primary_uplift_capacity = torch.where(
            valid_cap_curve,
            torch.minimum(residual_primary_uplift_capacity, anchor_gap_capacity),
            residual_primary_uplift_capacity,
        )
        total_primary_uplift_capacity = torch.where(
            valid_cap_curve,
            torch.minimum(total_primary_uplift_capacity, anchor_gap_capacity),
            total_primary_uplift_capacity,
        )

        late_tau_factor = torch.exp(
            -self.late_tau_impact_decay_strength * torch.clamp(tau_days, min=0.0) / time_scale
        )
        continuous_uplift_balance_target = torch.clamp(
            (1.0 - 0.55 * passivation)
            * (1.0 - 0.35 * acid_buffer_penalty)
            * (1.0 - 0.35 * diffusion_drag)
            * (0.55 + 0.45 * continuous_catalyst_drive)
            + 0.55 * depassivation
            + 0.22 * transformation
            + 0.18 * fast_release
            + 0.18 * primary_catalyst_synergy,
            min=0.0,
            max=2.50,
        )
        continuous_uplift_balance_target = torch.clamp(
            continuous_uplift_balance_target + 0.15 * lix_factor,
            min=0.0,
            max=2.65,
        )
        continuous_uplift_balance_response_days = torch.clamp(
            (0.45 * shared_uplift_response_days
            + 0.35 * transform_response_days
            + 0.20 * passivation_response_days),
            # * geometry_response_multiplier,
            min=self.min_transition_days,
        )
        latent_gap_factor = self._causal_response_smooth_torch(
            continuous_uplift_balance_target,
            catalyst_elapsed_days,
            continuous_uplift_balance_response_days,
        )
        latent_gap_factor = torch.clamp(latent_gap_factor, min=0.0, max=2.65)

        primary_uplift_efficiency = torch.clamp(latent_gap_factor / 1.35, min=0.0, max=1.75)
        early_unlock_efficiency = torch.clamp(
            0.95
            + 0.18 * (catalyzed_ferric_bootstrap - 1.0)
            - 0.22 * primary_drive
            - 0.20 * diffusion_drag_strength
            - 0.08 * ore_decay_strength,
            min=0.25,
            max=1.25,
        )
        primary_uplift_efficiency = torch.clamp(
            primary_uplift_efficiency * early_unlock_efficiency,
            min=0.0,
            max=1.75,
        )
        primary_effective_gap_capacity = torch.minimum(
            base_uplift,
            residual_primary_uplift_capacity
            * primary_uplift_efficiency,
            # * material_transport_factor
            # * geometry_transport_factor,
        )

        continuous_uplift_target = torch.clamp(
            continuous_catalyst_drive * late_tau_factor * early_unlock_efficiency, # * material_transport_factor,
            min=0.0,
            max=1.10,
        )

        late_flat_transition = torch.clamp(continuous_catalyst_drive * flat_input_score, min=0.0, max=1.0)
        cat_gap_response_days = (
            self.flat_input_uplift_response_days
            * flat_input_score
            # * geometry_response_multiplier
            * (1.0 + self.flat_input_late_uplift_response_boost * late_flat_transition)
        )
        continuous_uplift_progress = torch.cummax(
            torch.clamp(continuous_uplift_target, min=0.0, max=1.10),
            dim=1,
        ).values

        cat_gap_smoothing_days = torch.clamp(
            0.55 * cat_gap_response_days + 0.45 * temp_days,
            min=0.0,
        )
        cat_gap_raw, cat_gap_smoothed = self._evolve_catalyst_gap_torch(
            gap_capacity=primary_effective_gap_capacity,
            activation=continuous_uplift_progress,
            t_days=catalyst_elapsed_days,
            response_days=cat_gap_smoothing_days,
        )
        cat_gap_smoothed = torch.clamp(cat_gap_smoothed, min=0.0)
        cat_gap_smoothed = torch.cummax(cat_gap_smoothed, dim=1).values

        # (E) Acid-consumption early-time deficit on the catalyzed curve.
        # For samples whose mineralogy gate is open (high carbonates and/or
        # high acid-generating sulfides), the network can learn a small
        # negative-uplift term that decays exponentially with time after
        # catalyst start. This is what allows fresh ores like Toquepala
        # fresca to predict catalyzed *below* control in the first ~30-90
        # days while keeping later catalyzed-above-control kinetics intact.
        # The y_cat cummax below preserves the dip in time but enforces
        # global monotonicity of the catalyzed cumulative recovery.
        acid_amp_pp = latent_params.get("acid_consumption_amp_pp")
        acid_tau_days = latent_params.get("acid_consumption_tau_days")
        acid_gate_val = latent_params.get("acid_consumption_gate")
        if (
            acid_amp_pp is not None
            and acid_tau_days is not None
            and acid_gate_val is not None
        ):
            t_post_cat = torch.clamp(catalyst_elapsed_days, min=0.0)
            tau_safe = torch.clamp(acid_tau_days, min=1.0)
            onset_gate = 1.0 - torch.exp(-t_post_cat / 5.0)
            acid_deficit = (
                acid_amp_pp
                * acid_gate_val
                * torch.exp(-t_post_cat / tau_safe)
                * onset_gate
            )
        else:
            acid_deficit = torch.zeros_like(cat_gap_smoothed)

        # Apply the initial column-startup gate to the CONTROL curve BEFORE the
        # catalyzed curve is constructed.  The gate models the residence-time delay
        # before leaching visibly starts (acid percolation / wetting of the ore).
        #
        # Correct order of operations:
        #   1. gate y_ctrl            — onset delay on the base kinetics
        #   2. y_cat = y_ctrl + gap   — catalyzed is built on the already-gated ctrl
        #
        # This prevents double-gating: the catalyst gap (cat_gap_smoothed) is already
        # anchored to catalyst_elapsed_days = 0 at the first catalyst addition, so it
        # naturally produces zero uplift before catalyst arrives and grows smoothly
        # afterwards.  Gating y_cat *after* adding the gap would suppress and then
        # re-release the catalytic effect, creating an artificial jump artifact.
        initial_gate_strength = latent_params.get("initial_gate_strength")
        initial_gate_mid_day = latent_params.get("initial_gate_mid_day")
        initial_gate_width_day = latent_params.get("initial_gate_width_day")
        if (
            initial_gate_strength is not None
            and initial_gate_mid_day is not None
            and initial_gate_width_day is not None
        ):
            y_ctrl = self._apply_initial_stage_gate_torch(
                y_ctrl,
                t_abs,
                initial_gate_strength,
                initial_gate_mid_day,
                initial_gate_width_day,
            )

        y_cat = y_ctrl + cat_gap_smoothed - acid_deficit

        # Smooth monotone enforcement on recovery percentages.
        y_ctrl = torch.cummax(torch.clamp(y_ctrl, 0.0, 100.0), dim=1).values
        y_cat  = torch.cummax(torch.clamp(y_cat,  0.0, 100.0), dim=1).values

        # Softly compress both curves against the same chemistry/P80-based caps
        # used during training so deployed members do not drift into unrealistic
        # long-horizon asymptotes.
        cap_softness = float(
            max(
                CONFIG.get("prefit_cap_target_soft_margin", 2.0),
                CONFIG.get("prefit_cap_target_margin_fraction", 0.05) * 100.0,
            )
        )
        y_ctrl = self._soft_upper_bound_torch(y_ctrl, ctrl_cap_curve, cap_softness)
        y_cat = self._soft_upper_bound_torch(y_cat, cat_cap_curve, cap_softness)
        y_ctrl = torch.cummax(y_ctrl, dim=1).values
        y_cat = torch.cummax(y_cat, dim=1).values

        # Physical recovery bounds only. The asymptote cap is enforced upstream
        # on the parameter vectors themselves.
        y_ctrl = torch.clamp(y_ctrl, 0.0, 100.0)
        y_cat  = torch.clamp(y_cat,  0.0, 100.0)
        # A zero catalyst schedule must be indistinguishable from control. This
        # final mask also prevents the larger catalyzed cap from creating uplift
        # before any cumulative catalyst has reached the column.
        y_cat = torch.where(catalyst_has_reached_time, y_cat, y_ctrl)

        if not return_states:
            if squeeze_outputs:
                return self._maybe_squeeze_batch_dim(y_ctrl), self._maybe_squeeze_batch_dim(y_cat)
            return y_ctrl, y_cat

        states = {
            "ore_accessibility": self._maybe_squeeze_batch_dim(ore_accessibility) if squeeze_outputs else ore_accessibility,
            "fast_release": self._maybe_squeeze_batch_dim(fast_release) if squeeze_outputs else fast_release,
            "acid_buffer_penalty": self._maybe_squeeze_batch_dim(acid_buffer_penalty) if squeeze_outputs else acid_buffer_penalty,
            "diffusion_drag": self._maybe_squeeze_batch_dim(diffusion_drag) if squeeze_outputs else diffusion_drag,
            "passivation": self._maybe_squeeze_batch_dim(passivation) if squeeze_outputs else passivation,
            "passivation_raw": self._maybe_squeeze_batch_dim(passivation_raw) if squeeze_outputs else passivation_raw,
            "depassivation": self._maybe_squeeze_batch_dim(depassivation) if squeeze_outputs else depassivation,
            "transformation": self._maybe_squeeze_batch_dim(transformation) if squeeze_outputs else transformation,
            "primary_catalyst_synergy": self._maybe_squeeze_batch_dim(primary_catalyst_synergy) if squeeze_outputs else primary_catalyst_synergy,
            "ctrl_rate_multiplier": self._maybe_squeeze_batch_dim(ctrl_rate_multiplier) if squeeze_outputs else ctrl_rate_multiplier,
            "cat_rate_multiplier": self._maybe_squeeze_batch_dim(cat_rate_multiplier) if squeeze_outputs else cat_rate_multiplier,
            "effective_catalyst": self._maybe_squeeze_batch_dim(effective_catalyst) if squeeze_outputs else effective_catalyst,
            "effective_tau_days": self._maybe_squeeze_batch_dim(effective_tau_days) if squeeze_outputs else effective_tau_days,
            "catalyst_start_day": self._maybe_squeeze_batch_dim(catalyst_start_day) if squeeze_outputs else catalyst_start_day,
            "effective_lixiviant": self._maybe_squeeze_batch_dim(effective_lixiviant) if squeeze_outputs else effective_lixiviant,
            "catalyst_factor": self._maybe_squeeze_batch_dim(catalyst_factor) if squeeze_outputs else catalyst_factor,
            "lix_factor": self._maybe_squeeze_batch_dim(lix_factor) if squeeze_outputs else lix_factor,
            "flat_input_score": self._maybe_squeeze_batch_dim(flat_input_score) if squeeze_outputs else flat_input_score,
            "continuous_catalyst_drive": self._maybe_squeeze_batch_dim(continuous_catalyst_drive) if squeeze_outputs else continuous_catalyst_drive,
            "continuous_uplift_balance": self._maybe_squeeze_batch_dim(latent_gap_factor) if squeeze_outputs else latent_gap_factor,
            "continuous_uplift_progress": self._maybe_squeeze_batch_dim(continuous_uplift_progress) if squeeze_outputs else continuous_uplift_progress,
            "residual_primary_factor": self._maybe_squeeze_batch_dim(residual_primary_factor) if squeeze_outputs else residual_primary_factor,
            "residual_primary_uplift_capacity": self._maybe_squeeze_batch_dim(residual_primary_uplift_capacity) if squeeze_outputs else residual_primary_uplift_capacity,
            "total_primary_uplift_capacity": self._maybe_squeeze_batch_dim(total_primary_uplift_capacity) if squeeze_outputs else total_primary_uplift_capacity,
            "control_cap_curve": self._maybe_squeeze_batch_dim(ctrl_cap_curve) if squeeze_outputs else ctrl_cap_curve,
            "catalyzed_cap_curve": self._maybe_squeeze_batch_dim(cat_cap_curve) if squeeze_outputs else cat_cap_curve,
            "primary_uplift_efficiency": self._maybe_squeeze_batch_dim(primary_uplift_efficiency) if squeeze_outputs else primary_uplift_efficiency,
            # "material_transport_factor": self._maybe_squeeze_batch_dim(material_transport_factor) if squeeze_outputs else material_transport_factor,
            # "geometry_response_multiplier": self._maybe_squeeze_batch_dim(geometry_response_multiplier) if squeeze_outputs else geometry_response_multiplier,
            # "geometry_transport_factor": self._maybe_squeeze_batch_dim(geometry_transport_factor) if squeeze_outputs else geometry_transport_factor,
            "catalyzed_gap_raw": self._maybe_squeeze_batch_dim(cat_gap_raw) if squeeze_outputs else cat_gap_raw,
            "catalyzed_gap_smoothed": self._maybe_squeeze_batch_dim(cat_gap_smoothed) if squeeze_outputs else cat_gap_smoothed,
            "initial_gate_strength": self._maybe_squeeze_batch_dim(initial_gate_strength) if squeeze_outputs else initial_gate_strength,
            "initial_gate_mid_day": self._maybe_squeeze_batch_dim(initial_gate_mid_day) if squeeze_outputs else initial_gate_mid_day,
            "initial_gate_width_day": self._maybe_squeeze_batch_dim(initial_gate_width_day) if squeeze_outputs else initial_gate_width_day,
            "irrigation_rate": self._maybe_squeeze_batch_dim(irrigation_rate) if squeeze_outputs else irrigation_rate,
            "ferric_synergy": self._maybe_squeeze_batch_dim(ferric_synergy) if squeeze_outputs else ferric_synergy,
            "surface_refresh": self._maybe_squeeze_batch_dim(surface_refresh) if squeeze_outputs else surface_refresh,
        }
        if squeeze_outputs:
            return self._maybe_squeeze_batch_dim(y_ctrl), self._maybe_squeeze_batch_dim(y_cat), states
        return y_ctrl, y_cat, states


# ---------------------------
# Shared ensemble plot annotations
# ---------------------------
def _annotate_ensemble_extension(ax: Any, record: Dict[str, Any]) -> None:
    catalyst_start_day = float(record.get("catalyst_addition_start_day", np.nan))
    catalyst_stop_day = float(record.get("catalyst_addition_stop_day", np.nan))
    avg_dose_mg_l = float(
        record.get(
            "average_catalyst_dose_mg_l",
            record.get("catalyzed_avg_catalyst_dose_mg_l", np.nan),
        )
    )
    stopped_before_test_end = bool(record.get("stopped_before_test_end", False))
    extension_applied = bool(record.get("extension_applied", False))

    if np.isfinite(catalyst_start_day):
        ax.axvline(catalyst_start_day, color="#666666", lw=1.0, ls="--", alpha=0.7)
    if stopped_before_test_end and np.isfinite(catalyst_stop_day):
        ax.axvline(catalyst_stop_day, color="#444444", lw=1.0, ls=":", alpha=0.85)

    if not extension_applied:
        return
    if not np.isfinite(avg_dose_mg_l):
        return

    text = f"Average Catalyst dose: {avg_dose_mg_l:.2f} mg/L"

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
def _expand_series_to_y_shape(series: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 1:
        return torch.as_tensor(series, dtype=y.dtype, device=y.device).view(-1)
    return PairCurveNet._expand_series_to_batch_torch(series, int(y.shape[0]), y.dtype, y.device)


def _monotonic_penalty(y: torch.Tensor) -> torch.Tensor:
    if y.shape[-1] < 2:
        return torch.tensor(0.0, dtype=y.dtype, device=y.device)
    return torch.relu(y[..., :-1] - y[..., 1:]).mean()


def _curve_slopes(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if y.shape[-1] < 2:
        return torch.empty(0, dtype=y.dtype, device=y.device)
    t_expanded = _expand_series_to_y_shape(t, y)
    if t_expanded.shape[-1] < 2:
        return torch.empty(0, dtype=y.dtype, device=y.device)
    dt = torch.clamp(t_expanded[..., 1:] - t_expanded[..., :-1], min=1e-3)
    return (y[..., 1:] - y[..., :-1]) / dt


def _curvature_penalty(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    slope = _curve_slopes(y, t)
    if slope.shape[-1] < 2:
        return torch.tensor(0.0, dtype=y.dtype, device=y.device)
    return torch.abs(slope[..., 1:] - slope[..., :-1]).mean()


def _weighted_curvature_penalty(y: torch.Tensor, t: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    slope = _curve_slopes(y, t)
    if slope.shape[-1] < 2:
        return torch.tensor(0.0, dtype=y.dtype, device=y.device)
    curv = torch.abs(slope[..., 1:] - slope[..., :-1])
    w = _expand_series_to_y_shape(weight, y)
    if w.shape[-1] < curv.shape[-1] + 2:
        return curv.mean()
    w_mid = torch.clamp(w[..., 1:-1], min=0.0)[..., : curv.shape[-1]]
    if w_mid.numel() == 0:
        return curv.mean()
    weight_sum = torch.clamp(w_mid.sum(), min=1e-6)
    return torch.sum(curv * w_mid) / weight_sum


def _weighted_positive_curvature_penalty(y: torch.Tensor, t: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    slope = _curve_slopes(y, t)
    if slope.shape[-1] < 2:
        return torch.tensor(0.0, dtype=y.dtype, device=y.device)
    curv = torch.relu(slope[..., 1:] - slope[..., :-1])
    w = _expand_series_to_y_shape(weight, y)
    if w.shape[-1] < curv.shape[-1] + 2:
        return curv.mean()
    w_mid = torch.clamp(w[..., 1:-1], min=0.0)[..., : curv.shape[-1]]
    if w_mid.numel() == 0:
        return curv.mean()
    weight_sum = torch.clamp(w_mid.sum(), min=1e-6)
    return torch.sum(curv * w_mid) / weight_sum


def _weighted_late_positive_curvature_penalty(
    y: torch.Tensor,
    t: torch.Tensor,
    progress: torch.Tensor,
) -> torch.Tensor:
    slope = _curve_slopes(y, t)
    if slope.shape[-1] < 2:
        return torch.tensor(0.0, dtype=y.dtype, device=y.device)
    curv = torch.relu(slope[..., 1:] - slope[..., :-1])
    p = _expand_series_to_y_shape(progress, y)
    if p.shape[-1] < curv.shape[-1] + 2:
        return curv.mean()
    # Penalize re-acceleration once the catalyst-driven uplift is already
    # materially underway; this is the shoulder / second-uplift shape that v10
    # is meant to suppress.
    w_mid = torch.clamp(p[..., 1:-1] - 0.35, min=0.0)[..., : curv.shape[-1]]
    if w_mid.numel() == 0:
        return curv.mean()
    w_mid = w_mid * w_mid
    weight_sum = torch.clamp(w_mid.sum(), min=1e-6)
    return torch.sum(curv * w_mid) / weight_sum


def _weighted_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    w = torch.clamp(_expand_series_to_y_shape(weight, loss), min=0.0)
    weight_sum = torch.clamp(w.sum(), min=1e-6)
    return torch.sum(w * loss) / weight_sum


def _normalize_rowwise_weights(weight: torch.Tensor) -> torch.Tensor:
    w = torch.as_tensor(weight)
    one_dim = w.ndim == 1
    if one_dim:
        w = w.view(1, -1)

    out = torch.zeros_like(w, dtype=w.dtype, device=w.device)
    eps = torch.finfo(w.dtype).eps
    for row_idx in range(w.shape[0]):
        finite_idx = torch.nonzero(torch.isfinite(w[row_idx]), as_tuple=False).squeeze(1)
        if int(finite_idx.numel()) <= 0:
            continue
        valid = torch.clamp(w[row_idx, finite_idx], min=0.0)
        mean_val = torch.mean(valid)
        if not torch.isfinite(mean_val) or float(mean_val.detach().cpu().item()) <= float(eps):
            out[row_idx, finite_idx] = 1.0
            continue
        out[row_idx, finite_idx] = valid / torch.clamp(mean_val, min=eps)
    return out.view(-1) if one_dim else out


def _early_leach_day_weights(
    values: torch.Tensor,
    min_relative_scale: float = 0.15,
    exponent: float = 1.0,
) -> torch.Tensor:
    x = torch.as_tensor(values)
    if x.ndim == 1:
        x = x.view(1, -1)

    weights = torch.ones_like(x, dtype=x.dtype, device=x.device)
    eps = torch.finfo(x.dtype).eps
    min_relative_scale = float(max(min_relative_scale, eps))
    exponent = float(max(exponent, eps))

    # Equalize contribution across the target-value range
    # Smaller target values receive more weight so later/higher
    # values do not dominate the fit only because they are numerically larger.
    x_abs = torch.abs(x)
    for row_idx in range(x.shape[0]):
        finite_idx = torch.nonzero(torch.isfinite(x_abs[row_idx]), as_tuple=False).squeeze(1)
        count = int(finite_idx.numel())
        if count <= 0:
            continue
        if count == 1:
            weights[row_idx, finite_idx] = 1.0
            continue

        valid = x_abs[row_idx, finite_idx]
        ref = torch.clamp(torch.amax(valid), min=eps)
        floor = torch.clamp(ref * min_relative_scale, min=eps)
        value_weights = torch.pow(ref / torch.clamp(valid, min=floor), exponent)
        norm = torch.clamp(torch.mean(value_weights), min=eps)
        weights[row_idx, finite_idx] = value_weights / norm
    return weights if values.ndim > 1 else weights.view(-1)


def _control_curve_fit_weights(
    t_days: torch.Tensor,
    values: torch.Tensor,
    early_end_day: float = 180.0,
    tail_start_day: float = 365.0,
    early_boost: float = 1.35,
    tail_boost: float = 0.35,
    min_relative_scale: float = 0.20,
    exponent: float = 0.80,
) -> torch.Tensor:
    value_weights = _early_leach_day_weights(
        values,
        min_relative_scale=min_relative_scale,
        exponent=exponent,
    )
    t = torch.as_tensor(t_days, dtype=torch.as_tensor(values).dtype, device=torch.as_tensor(values).device)
    one_dim = t.ndim == 1
    if one_dim:
        t = t.view(1, -1)
    vw = value_weights.view(1, -1) if value_weights.ndim == 1 else value_weights

    early_end_day = float(max(early_end_day, 1e-6))
    tail_start_day = float(max(tail_start_day, 0.0))
    early_progress = torch.clamp(t / early_end_day, min=0.0, max=1.0)
    early_weight = 1.0 + float(max(early_boost, 0.0)) * (1.0 - early_progress)
    tail_progress = torch.clamp(
        (t - tail_start_day) / max(early_end_day, 1.0),
        min=0.0,
        max=1.0,
    )
    tail_weight = 1.0 + float(max(tail_boost, 0.0)) * tail_progress
    weights = vw * early_weight * tail_weight
    weights = _normalize_rowwise_weights(weights)
    return weights.view(-1) if values.ndim == 1 else weights


def _time_window_weights(
    t_days: torch.Tensor,
    start_day: float = 0.0,
    end_day: Optional[float] = None,
) -> torch.Tensor:
    t = torch.as_tensor(t_days)
    mask = torch.isfinite(t) & (t >= float(start_day))
    if end_day is not None and np.isfinite(end_day):
        mask = mask & (t < float(end_day))
    return mask.to(dtype=t.dtype)


def _windowed_weighted_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    t_days: torch.Tensor,
    start_day: float = 0.0,
    end_day: Optional[float] = None,
    base_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    window_weight = _time_window_weights(t_days, start_day=start_day, end_day=end_day)
    window_weight = _expand_series_to_y_shape(window_weight, target)
    if base_weight is not None:
        window_weight = window_weight * _expand_series_to_y_shape(base_weight, target)
    if not torch.any(window_weight > 0):
        return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
    return _weighted_smooth_l1_loss(pred, target, window_weight)


def _control_early_process_penalty(
    pred_ctrl: torch.Tensor,
    ctrl_y: torch.Tensor,
    ctrl_t: torch.Tensor,
    ctrl_states: Dict[str, torch.Tensor],
    end_day: float = 180.0,
    passivation_weight: float = 0.45,
    acid_buffer_weight: float = 0.30,
    diffusion_drag_weight: float = 0.25,
) -> torch.Tensor:
    early_mask = _expand_series_to_y_shape(
        _time_window_weights(ctrl_t, start_day=0.0, end_day=end_day),
        ctrl_y,
    )
    if not torch.any(early_mask > 0):
        return torch.tensor(0.0, dtype=pred_ctrl.dtype, device=pred_ctrl.device)

    underprediction = torch.relu(ctrl_y - pred_ctrl)
    weights = early_mask * underprediction
    if not torch.any(weights > 0):
        return torch.tensor(0.0, dtype=pred_ctrl.dtype, device=pred_ctrl.device)

    burden = (
        float(passivation_weight) * _expand_series_to_y_shape(ctrl_states["passivation"], ctrl_y)
        + float(acid_buffer_weight) * _expand_series_to_y_shape(ctrl_states["acid_buffer_penalty"], ctrl_y)
        + float(diffusion_drag_weight) * _expand_series_to_y_shape(ctrl_states["diffusion_drag"], ctrl_y)
    )
    return torch.sum(weights * burden) / torch.clamp(weights.sum(), min=1e-6)


def _uplift_onset_delay_penalty(
    effective_tau_days: torch.Tensor,
    cat_t: torch.Tensor,
    true_gap: torch.Tensor,
    min_gap_threshold: float = 0.5,
    gap_fraction_threshold: float = 0.35,
) -> torch.Tensor:
    if true_gap.ndim == 1:
        true_gap = true_gap.view(1, -1)
    t = _expand_series_to_y_shape(cat_t, true_gap)
    tau = torch.as_tensor(effective_tau_days, dtype=true_gap.dtype, device=true_gap.device)
    if tau.ndim == 0:
        tau = tau.view(1, 1)
    elif tau.ndim == 1:
        tau = tau.view(-1, 1)

    max_gap = torch.amax(torch.clamp(true_gap, min=0.0), dim=1, keepdim=True)
    threshold = torch.maximum(
        torch.full_like(max_gap, float(min_gap_threshold)),
        float(gap_fraction_threshold) * max_gap,
    )
    has_material_uplift = (max_gap.squeeze(1) >= float(min_gap_threshold))
    onset_mask = true_gap >= threshold

    first_idx = torch.argmax(onset_mask.to(dtype=torch.int64), dim=1)
    onset_time = torch.gather(t, 1, first_idx.view(-1, 1)).squeeze(1)
    fallback_time = t[:, -1]
    onset_time = torch.where(onset_mask.any(dim=1), onset_time, fallback_time)

    time_span = torch.clamp(t[:, -1] - t[:, 0], min=1.0)
    tau_flat = tau.squeeze(1)
    delay = torch.relu(tau_flat - onset_time) / time_span
    delay = torch.where(has_material_uplift, delay, torch.zeros_like(delay))
    return delay.mean()


def _slope_cap_penalty(y: torch.Tensor, t: torch.Tensor, max_slope_per_day: float) -> torch.Tensor:
    slope = _curve_slopes(y, t)
    if slope.numel() < 1:
        return torch.tensor(0.0, dtype=y.dtype, device=y.device)
    return torch.relu(torch.abs(slope) - float(max_slope_per_day)).mean()


def _early_slope_penalty(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    t: torch.Tensor,
    early_end_day: float,
) -> torch.Tensor:
    """(D) Penalise mismatch in the average early-window slope.

    For each sample the slope is the chord slope from the first time-step
    up to the last time-step whose relative time is within
    ``[0, early_end_day]``:

        slope = (y[idx_end] - y[idx_start]) / (t[idx_end] - t[idx_start])

    The penalty is the mean ``|slope_pred - slope_true|`` over samples with
    a non-degenerate window.  This adds pressure against too-slow takeoffs
    (e.g. SCL Escondida, 026 secondary sulfide) and too-fast early takeoffs
    on refractory samples (e.g. Elephant, Los Bronces).
    """
    if y_pred.shape[-1] < 2 or y_true.shape[-1] < 2:
        return torch.tensor(0.0, dtype=y_pred.dtype, device=y_pred.device)
    t_b = _expand_series_to_y_shape(t, y_pred)
    if t_b.shape != y_pred.shape or t_b.shape != y_true.shape:
        return torch.tensor(0.0, dtype=y_pred.dtype, device=y_pred.device)

    t0 = t_b[..., :1]
    rel_t = t_b - t0
    finite_mask = (
        torch.isfinite(t_b)
        & torch.isfinite(y_pred)
        & torch.isfinite(y_true)
    )
    window_mask = finite_mask & (rel_t >= 0.0) & (rel_t <= float(early_end_day))
    if not torch.any(window_mask):
        return torch.tensor(0.0, dtype=y_pred.dtype, device=y_pred.device)

    idx_range = torch.arange(t_b.shape[-1], device=t_b.device).expand_as(t_b)
    masked_idx = torch.where(window_mask, idx_range, torch.full_like(idx_range, -1))
    end_idx = masked_idx.amax(dim=-1)
    start_idx = torch.zeros_like(end_idx)
    valid_row = end_idx > start_idx
    if not torch.any(valid_row):
        return torch.tensor(0.0, dtype=y_pred.dtype, device=y_pred.device)

    rows = torch.arange(y_pred.shape[0], device=y_pred.device)
    end_idx_c = torch.clamp(end_idx, min=0)
    y_pred_end = y_pred[rows, end_idx_c]
    y_true_end = y_true[rows, end_idx_c]
    y_pred_start = y_pred[rows, start_idx]
    y_true_start = y_true[rows, start_idx]
    t_end = t_b[rows, end_idx_c]
    t_start = t_b[rows, start_idx]

    dt = torch.clamp(t_end - t_start, min=1.0)
    slope_pred = (y_pred_end - y_pred_start) / dt
    slope_true = (y_true_end - y_true_start) / dt

    diff = torch.abs(slope_pred - slope_true)
    valid_f = valid_row.to(diff.dtype)
    diff = diff * valid_f
    weight_sum = torch.clamp(valid_f.sum(), min=1.0)
    return diff.sum() / weight_sum


def _t50_penalty(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    t: torch.Tensor,
    time_scale_days: float = 1000.0,
) -> torch.Tensor:
    """(D) Penalise mismatch in time-to-50%-of-max-true-recovery.

    The target is 50% of ``y_true.amax``.  The first time step at which the
    curve reaches the target defines ``t50``; if never reached, the last
    finite time is used.  The penalty is the mean ``|t50_pred - t50_true|``
    normalised by ``time_scale_days`` so the weight in CONFIG is on a
    comparable scale to the other fit losses.
    """
    if y_pred.shape[-1] < 2 or y_true.shape[-1] < 2:
        return torch.tensor(0.0, dtype=y_pred.dtype, device=y_pred.device)
    t_b = _expand_series_to_y_shape(t, y_pred)
    if t_b.shape != y_pred.shape or t_b.shape != y_true.shape:
        return torch.tensor(0.0, dtype=y_pred.dtype, device=y_pred.device)

    valid = (
        torch.isfinite(t_b)
        & torch.isfinite(y_pred)
        & torch.isfinite(y_true)
    )
    if not torch.any(valid):
        return torch.tensor(0.0, dtype=y_pred.dtype, device=y_pred.device)

    y_true_safe = torch.where(valid, y_true, torch.zeros_like(y_true))
    y_pred_safe = torch.where(valid, y_pred, torch.zeros_like(y_pred))
    y_true_max = torch.amax(y_true_safe, dim=-1, keepdim=True)
    target = 0.5 * y_true_max

    valid_row = (target.squeeze(-1) > 1.0e-3)
    if not torch.any(valid_row):
        return torch.tensor(0.0, dtype=y_pred.dtype, device=y_pred.device)

    big = torch.full_like(t_b, float("inf"))
    t_pred_cross = torch.where((y_pred_safe >= target) & valid, t_b, big)
    t_true_cross = torch.where((y_true_safe >= target) & valid, t_b, big)
    t50_pred = torch.amin(t_pred_cross, dim=-1)
    t50_true = torch.amin(t_true_cross, dim=-1)

    neg_inf = torch.full_like(t_b, float("-inf"))
    t_last = torch.amax(torch.where(valid, t_b, neg_inf), dim=-1)
    t_last = torch.where(torch.isfinite(t_last), t_last, torch.zeros_like(t_last))
    t50_pred = torch.where(torch.isfinite(t50_pred), t50_pred, t_last)
    t50_true = torch.where(torch.isfinite(t50_true), t50_true, t_last)

    diff = torch.abs(t50_pred - t50_true)
    valid_f = valid_row.to(diff.dtype)
    diff = diff * valid_f
    weight_sum = torch.clamp(valid_f.sum(), min=1.0)
    return (diff.sum() / weight_sum) / float(time_scale_days)


def _final_value_smooth_l1_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    tail_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pred.shape[-1] < 1 or target.shape[-1] < 1:
        return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
    pred_f = pred[..., -1]
    target_f = target[..., -1]
    valid = torch.isfinite(pred_f) & torch.isfinite(target_f)
    if not torch.any(valid):
        return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
    loss = F.smooth_l1_loss(pred_f, target_f, reduction="none")
    if tail_weight is None:
        weights = valid.to(dtype=pred.dtype)
    else:
        w = torch.as_tensor(tail_weight, dtype=pred.dtype, device=pred.device)
        if w.ndim > loss.ndim:
            w = w[..., -1]
        while w.ndim < loss.ndim:
            w = w.unsqueeze(-1)
        if w.shape != loss.shape:
            if w.numel() == 1:
                w = w.expand_as(loss)
            elif w.numel() == loss.numel():
                w = w.reshape_as(loss)
            else:
                w = torch.ones_like(loss)
        weights = torch.where(valid & torch.isfinite(w), torch.clamp(w, min=0.0), torch.zeros_like(loss))
    return torch.sum(loss * weights) / torch.clamp(weights.sum(), min=1e-6)


def _catalyst_use_prior_penalty(
    catalyst_use_frac: torch.Tensor,
    cat_c: torch.Tensor,
    cat_conc: Optional[torch.Tensor],
    true_gap: torch.Tensor,
) -> torch.Tensor:
    use_frac = torch.as_tensor(catalyst_use_frac, dtype=true_gap.dtype, device=true_gap.device)
    if use_frac.ndim == 1:
        use_frac = use_frac.view(-1, 1)
    c = PairCurveNet._expand_series_to_batch_torch(cat_c, int(use_frac.shape[0]), true_gap.dtype, true_gap.device)
    exposure = torch.log1p(torch.clamp(c[:, -1:], min=0.0))
    if cat_conc is not None:
        conc = PairCurveNet._expand_series_to_batch_torch(cat_conc, int(use_frac.shape[0]), true_gap.dtype, true_gap.device)
        exposure = exposure + 0.5 * torch.log1p(torch.clamp(torch.mean(conc, dim=-1, keepdim=True), min=0.0))
    gap_max = torch.amax(torch.clamp(true_gap, min=0.0), dim=-1, keepdim=True)
    exposure_score = torch.clamp(exposure / 0.20, min=0.0, max=1.0)
    gap_score = torch.clamp(gap_max / 12.0, min=0.0, max=1.0)
    target_min = float(np.clip(CONFIG.get("catalyst_use_prior_min", 0.02), 0.0, 1.0))
    target_max = float(np.clip(CONFIG.get("catalyst_use_prior_max", 0.12), target_min, 1.0))
    target = target_min + (target_max - target_min) * exposure_score * gap_score
    return torch.mean(torch.relu(target - use_frac) ** 2.0)


def _weighted_mean_penalty(
    values: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    v = torch.as_tensor(values)
    w = torch.clamp(_expand_series_to_y_shape(weight, v), min=0.0)
    if not torch.any(w > 0):
        return torch.zeros((), dtype=v.dtype, device=v.device)
    weight_sum = torch.clamp(w.sum(), min=1e-6)
    return torch.sum(v * w) / weight_sum


def _prefit_param_distillation_penalty(
    pred_params: torch.Tensor,
    target_params: torch.Tensor,
) -> torch.Tensor:
    pred = torch.as_tensor(pred_params)
    target = torch.as_tensor(target_params, dtype=pred.dtype, device=pred.device)
    if pred.ndim == 1:
        pred = pred.view(1, -1)
    if target.ndim == 1:
        target = target.view(1, -1)
    if pred.shape[-1] < 4 or target.shape[-1] < 4:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)

    target = target[..., :4]
    finite_mask = torch.isfinite(pred[..., :4]) & torch.isfinite(target)
    if not torch.any(finite_mask):
        return torch.zeros((), dtype=pred.dtype, device=pred.device)

    amp_pred = pred[:, [0, 2]]
    amp_true = target[:, [0, 2]]
    amp_rel_err = (amp_pred - amp_true) / torch.clamp(torch.abs(amp_true), min=1.0)
    amp_loss = F.smooth_l1_loss(
        amp_rel_err,
        torch.zeros_like(amp_rel_err),
        reduction="none",
    )
    amp_weight = torch.tensor([1.35, 0.90], dtype=pred.dtype, device=pred.device).view(1, 2)
    amp_mask = finite_mask[:, [0, 2]].to(dtype=pred.dtype)

    rate_pred = torch.log(torch.clamp(pred[:, [1, 3]], min=1e-6))
    rate_true = torch.log(torch.clamp(target[:, [1, 3]], min=1e-6))
    rate_loss = F.smooth_l1_loss(rate_pred, rate_true, reduction="none")
    rate_weight = torch.tensor([1.45, 0.95], dtype=pred.dtype, device=pred.device).view(1, 2)
    rate_mask = finite_mask[:, [1, 3]].to(dtype=pred.dtype)

    total_loss = torch.sum(amp_loss * amp_weight * amp_mask) + torch.sum(rate_loss * rate_weight * rate_mask)
    total_weight = torch.clamp(
        torch.sum(amp_weight * amp_mask) + torch.sum(rate_weight * rate_mask),
        min=1e-6,
    )
    return total_loss / total_weight


def _primary_fraction_prior_penalty(
    pred_frac: torch.Tensor,
    target_prior_frac: torch.Tensor,
    delta_scale: float,
) -> torch.Tensor:
    pred = torch.as_tensor(pred_frac)
    target = torch.as_tensor(target_prior_frac, dtype=pred.dtype, device=pred.device)
    scale = max(float(delta_scale), 1e-6)
    finite_mask = torch.isfinite(pred) & torch.isfinite(target)
    if not torch.any(finite_mask):
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    diff = (pred - target) / scale
    loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    weight = finite_mask.to(dtype=pred.dtype)
    return torch.sum(loss * weight) / torch.clamp(weight.sum(), min=1e-6)


def _pair_training_loss_from_tensors(
    model: PairCurveNet,
    tensors: Dict[str, torch.Tensor],
    loss_weights: Dict[str, float],
    max_cat_slope_per_day: float,
) -> torch.Tensor:
    x = tensors["x"]
    x_raw = tensors["x_raw"]
    x_input_only = tensors["x_input_only"]
    ctrl_t = tensors["ctrl_t"]
    ctrl_y = tensors["ctrl_y"]
    ctrl_c = tensors["ctrl_c"]
    ctrl_l = tensors["ctrl_l"]
    ctrl_irr = tensors["ctrl_irr"]
    cat_t = tensors["cat_t"]
    cat_y = tensors["cat_y"]
    cat_c = tensors["cat_c"]
    cat_l = tensors["cat_l"]
    cat_irr = tensors["cat_irr"]
    cat_conc = tensors.get("cat_conc")
    cat_effective_start_day = tensors.get("cat_effective_start_day", tensors["cat_start_day"])
    cat_pre_start_mask = tensors.get("cat_pre_start_mask")
    cat_post_start_mask = tensors.get("cat_post_start_mask")
    ctrl_prefit_params = tensors.get("ctrl_prefit_params")
    cat_prefit_params = tensors.get("cat_prefit_params")
    p_ctrl, p_cat, tau, temp, kappa, aging_strength, latent = model.predict_params(
        x,
        x_raw,
        x_input_only,
        ctrl_curve_override_raw=tensors.get("ctrl_curve_override_raw"),
        cat_curve_override_raw=tensors.get("cat_curve_override_raw"),
        ctrl_cap_anchor_pct=tensors.get("ctrl_cap_anchor_pct"),
        cat_cap_anchor_pct=tensors.get("cat_cap_anchor_pct"),
        ctrl_cap_anchor_active=tensors.get("ctrl_cap_anchor_active"),
        cat_cap_anchor_active=tensors.get("cat_cap_anchor_active"),
        catalyst_t_days=cat_t,
        catalyst_cum_norm=cat_c,
        catalyst_conc_norm=cat_conc,
    )

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
        catalyst_start_day_override=tensors["ctrl_start_day"],
        latent_params=latent,
        return_states=True,
    )
    pred_ctrl_cat_time, _ = model.curves_given_params(
        p_ctrl,
        p_cat,
        cat_t,
        tensors["cat_ctrl_c"],
        cat_l,
        cat_irr,
        tau,
        temp,
        kappa,
        aging_strength,
        catalyst_start_day_override=tensors["cat_ctrl_start_day"],
        latent_params=latent,
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
        catalyst_start_day_override=cat_effective_start_day,
        latent_params=latent,
        return_states=True,
    )
    ctrl_true_on_cat_t = tensors["ctrl_true_on_cat_t"]

    if cat_post_start_mask is None:
        cat_post_start_mask = torch.ones_like(cat_y)
    if cat_pre_start_mask is None:
        cat_pre_start_mask = torch.zeros_like(cat_y)
    cat_post_start_mask = _expand_series_to_y_shape(cat_post_start_mask, cat_y)
    cat_pre_start_mask = _expand_series_to_y_shape(cat_pre_start_mask, cat_y)

    true_gap = torch.clamp(cat_y - ctrl_true_on_cat_t, min=0.0) * cat_post_start_mask
    pred_gap = torch.clamp(pred_cat - pred_ctrl_cat_time, min=0.0) * cat_post_start_mask

    dose_weight_boost = float(CONFIG.get("catalyzed_dose_weight_boost", 0.0))
    exposure_score_for_weight = torch.clamp(
        torch.log1p(torch.clamp(cat_c, min=0.0)) / 0.20,
        min=0.0,
        max=1.0,
    )
    if cat_conc is not None:
        conc_weight_score = torch.clamp(torch.log1p(torch.clamp(cat_conc, min=0.0)) / 0.20, min=0.0, max=1.0)
        exposure_score_for_weight = torch.maximum(exposure_score_for_weight, conc_weight_score)
    post_cat_weight = cat_post_start_mask * (
        1.0
        + 1.5 * cat_post_start_mask
        + dose_weight_boost * exposure_score_for_weight
    )
    uplift_fit_pen = _weighted_smooth_l1_loss(
        pred_gap,
        true_gap,
        post_cat_weight,
    )
    max_true_gap = torch.amax(true_gap, dim=-1, keepdim=True)
    true_gap_scale = torch.where(
        max_true_gap > 1e-6,
        true_gap / torch.clamp(max_true_gap, min=1e-6),
        torch.zeros_like(true_gap),
    )
    uplift_tail_weight = post_cat_weight * (
        1.0
        + (3.0 + float(CONFIG.get("catalyzed_gap_weight_boost", 0.0))) * true_gap_scale
    )
    uplift_tail_pen = _weighted_smooth_l1_loss(
        pred_gap,
        true_gap,
        uplift_tail_weight,
    )
    control_interp_weight = 1.0 + 1.5 * cat_post_start_mask
    control_interp_fit_pen = _weighted_smooth_l1_loss(
        pred_ctrl_cat_time,
        ctrl_true_on_cat_t,
        control_interp_weight,
    )
    control_early_end_day = float(CONFIG.get("control_early_end_day", 180.0))
    control_tail_start_day = float(CONFIG.get("control_tail_start_day", 365.0))
    control_fit_value_min_relative_scale = float(
        CONFIG.get("control_fit_value_min_relative_scale", 0.20)
    )
    control_fit_value_exponent = float(CONFIG.get("control_fit_value_exponent", 0.80))
    control_weight_early_boost = float(CONFIG.get("control_weight_early_boost", 1.35))
    control_weight_tail_boost = float(CONFIG.get("control_weight_tail_boost", 0.35))
    control_process_end_day = float(CONFIG.get("control_process_end_day", control_early_end_day))
    control_process_passivation_weight = float(
        CONFIG.get("control_process_passivation_weight", 0.45)
    )
    control_process_acid_buffer_weight = float(
        CONFIG.get("control_process_acid_buffer_weight", 0.30)
    )
    control_process_diffusion_drag_weight = float(
        CONFIG.get("control_process_diffusion_drag_weight", 0.25)
    )
    tau_onset_pen = _uplift_onset_delay_penalty(
        cat_states["effective_tau_days"],
        cat_t,
        true_gap,
    )

    ctrl_fit_weight = _control_curve_fit_weights(
        ctrl_t,
        ctrl_y,
        early_end_day=control_early_end_day,
        tail_start_day=control_tail_start_day,
        early_boost=control_weight_early_boost,
        tail_boost=control_weight_tail_boost,
        min_relative_scale=control_fit_value_min_relative_scale,
        exponent=control_fit_value_exponent,
    )
    cat_fit_weight = _early_leach_day_weights(true_gap)
    pre_catalyst_control_weight = _control_curve_fit_weights(
        cat_t,
        cat_y,
        early_end_day=control_early_end_day,
        tail_start_day=control_tail_start_day,
        early_boost=control_weight_early_boost,
        tail_boost=control_weight_tail_boost,
        min_relative_scale=control_fit_value_min_relative_scale,
        exponent=control_fit_value_exponent,
    ) * cat_pre_start_mask
    pre_catalyst_control_fit_pen = _weighted_smooth_l1_loss(
        pred_ctrl_cat_time,
        cat_y,
        pre_catalyst_control_weight,
    )
    loss_ctrl = _weighted_smooth_l1_loss(pred_ctrl, ctrl_y, ctrl_fit_weight)
    control_early_fit_pen = _windowed_weighted_smooth_l1_loss(
        pred_ctrl,
        ctrl_y,
        ctrl_t,
        start_day=0.0,
        end_day=control_early_end_day,
        base_weight=ctrl_fit_weight,
    )
    control_tail_fit_pen = _windowed_weighted_smooth_l1_loss(
        pred_ctrl,
        ctrl_y,
        ctrl_t,
        start_day=control_tail_start_day,
        end_day=None,
        base_weight=ctrl_fit_weight,
    )
    control_early_process_pen = _control_early_process_penalty(
        pred_ctrl,
        ctrl_y,
        ctrl_t,
        ctrl_states=ctrl_states,
        end_day=control_process_end_day,
        passivation_weight=control_process_passivation_weight,
        acid_buffer_weight=control_process_acid_buffer_weight,
        diffusion_drag_weight=control_process_diffusion_drag_weight,
    )
    loss_cat_uplift = _weighted_smooth_l1_loss(
        pred_gap,
        true_gap,
        cat_fit_weight * cat_post_start_mask,
    )
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
    latent_cat_rate_pen = _weighted_mean_penalty(
        torch.relu(cat_states["ctrl_rate_multiplier"] - cat_states["cat_rate_multiplier"]),
        cat_post_start_mask,
    )
    param_pen = torch.zeros((), dtype=pred_ctrl.dtype, device=pred_ctrl.device)
    if ctrl_prefit_params is not None:
        param_pen = param_pen + _prefit_param_distillation_penalty(p_ctrl, ctrl_prefit_params)
    if cat_prefit_params is not None:
        param_pen = param_pen + _prefit_param_distillation_penalty(p_cat, cat_prefit_params)
    primary_fraction_prior_pen = torch.zeros((), dtype=pred_ctrl.dtype, device=pred_ctrl.device)
    if "cpy_leach_frac_ctrl" in latent and "primary_control_prior_frac" in latent:
        primary_fraction_prior_pen = primary_fraction_prior_pen + _primary_fraction_prior_penalty(
            latent["cpy_leach_frac_ctrl"],
            latent["primary_control_prior_frac"],
            delta_scale=float(CONFIG.get("primary_fraction_learned_delta_max", 0.05)),
        )
    if "cpy_leach_frac_cat" in latent and "primary_catalyzed_prior_frac" in latent:
        primary_fraction_prior_pen = primary_fraction_prior_pen + _primary_fraction_prior_penalty(
            latent["cpy_leach_frac_cat"],
            latent["primary_catalyzed_prior_frac"],
            delta_scale=float(CONFIG.get("primary_fraction_learned_delta_max", 0.05)),
        )
    final_tail_boost = float(CONFIG.get("final_recovery_weight_tail_boost", 1.0))
    final_ctrl_weight = torch.ones((pred_ctrl.shape[0],), dtype=pred_ctrl.dtype, device=pred_ctrl.device)
    final_cat_weight = torch.clamp(
        1.0 + final_tail_boost * torch.squeeze(torch.clamp(max_true_gap / 15.0, min=0.0, max=1.0), dim=-1),
        min=1.0,
    )
    final_recovery_pen = _final_value_smooth_l1_penalty(
        pred_ctrl,
        ctrl_y,
        final_ctrl_weight,
    ) + _final_value_smooth_l1_penalty(
        pred_cat,
        cat_y,
        final_cat_weight,
    )
    final_uplift_pen = _final_value_smooth_l1_penalty(
        pred_gap,
        true_gap,
        final_cat_weight,
    )
    catalyst_use_prior_pen = _catalyst_use_prior_penalty(
        latent["catalyst_use_frac"],
        cat_c,
        cat_conc,
        true_gap,
    )
    flat_input_smooth_pen = _weighted_curvature_penalty(
        cat_states["catalyzed_gap_smoothed"],
        cat_t,
        cat_states["flat_input_score"],
    )
    flat_input_accel_pen = _weighted_positive_curvature_penalty(
        cat_states["catalyzed_gap_smoothed"],
        cat_t,
        cat_states["flat_input_score"],
    )
    single_uplift_late_accel_pen = _weighted_late_positive_curvature_penalty(
        cat_states["catalyzed_gap_smoothed"],
        cat_t,
        cat_states["continuous_uplift_progress"],
    )

    # Per-sample leach-cap penalty: penalises p_ctrl / p_cat independently
    # when their a1+a2 asymptote exceeds the mineralogy-derived cap for that
    # sample.  ctrl_cap and cat_cap are treated as separate constraints.
    def _output_cap_penalty(y_pred: torch.Tensor, cap: torch.Tensor | float) -> torch.Tensor:
        cap_t = torch.as_tensor(cap, dtype=y_pred.dtype, device=y_pred.device)
        cap_t = torch.where(torch.isfinite(cap_t), cap_t, torch.full_like(cap_t, 100.0))
        if cap_t.ndim > 0:
            cap_t = cap_t.view(-1, *([1] * max(0, y_pred.ndim - 1)))
            if y_pred.ndim > 1 and cap_t.shape[0] != y_pred.shape[0]:
                if cap_t.shape[0] == 1:
                    cap_t = cap_t.expand(y_pred.shape[0], *cap_t.shape[1:])
                else:
                    cap_t = cap_t[:1].expand(y_pred.shape[0], *cap_t.shape[1:])
        excess = torch.relu(y_pred - cap_t)
        rel_excess = excess / torch.clamp(cap_t, min=1.0)
        return (excess ** 2.0).mean() + 5.0 * (rel_excess ** 2.0).mean()
    
    cap_pen = (
        _output_cap_penalty(pred_ctrl, tensors["ctrl_cap"])
        + _output_cap_penalty(pred_cat, tensors["cat_cap"])
    )
    orp_aux_target = torch.as_tensor(tensors["orp_aux_target"], dtype=pred_cat.dtype, device=pred_cat.device).view(-1)
    orp_aux_mask = torch.as_tensor(tensors["orp_aux_mask"], dtype=pred_cat.dtype, device=pred_cat.device).view(-1)
    ferric_orp_pred = model.predict_orp_aux_target(latent["ferric_synergy"]).view(-1)
    valid_orp_aux = (
        (orp_aux_mask > 0.5)
        & torch.isfinite(orp_aux_target)
        & torch.isfinite(ferric_orp_pred)
    )
    if torch.any(valid_orp_aux):
        ferric_orp_aux_pen = F.mse_loss(
            ferric_orp_pred[valid_orp_aux],
            orp_aux_target[valid_orp_aux],
            reduction="mean",
        )
    else:
        ferric_orp_aux_pen = torch.zeros((), dtype=pred_cat.dtype, device=pred_cat.device)

    # (D) Early-window chord-slope + t50 shape penalties, applied to both
    # the control and catalyzed curves.  These give the training loss direct
    # leverage on *when* a curve takes off and *how steeply*, which plain
    # point-wise smooth-L1 loss does not always enforce when the curve has
    # many tail points that dominate the mean error.
    slope_early_end_day = float(CONFIG.get("slope_early_end_day", control_early_end_day))
    t50_time_scale_days = float(CONFIG.get("t50_time_scale_days", 1000.0))
    slope_early_ctrl_pen = _early_slope_penalty(
        pred_ctrl, ctrl_y, ctrl_t, slope_early_end_day
    )
    slope_early_cat_pen = _early_slope_penalty(
        pred_cat, cat_y, cat_t, slope_early_end_day
    )
    slope_early_pen = slope_early_ctrl_pen + slope_early_cat_pen
    t50_ctrl_pen = _t50_penalty(pred_ctrl, ctrl_y, ctrl_t, t50_time_scale_days)
    t50_cat_pen = _t50_penalty(pred_cat, cat_y, cat_t, t50_time_scale_days)
    t50_match_pen = t50_ctrl_pen + t50_cat_pen

    # v12: Mineralogy-conditional slope_early gating.
    # Chalcopyrite-dominated (primary sulfide) ores have inherently slow early
    # kinetics due to passivation lag, ferric iron build-up time, and biofilm
    # colonisation — forcing a steep early slope via slope_early mispredicts their
    # asymptote.  We scale the effective slope_early weight DOWN as the fraction
    # of primary copper equivalents increases.
    #   primary_cu_frac = 0  → scale = 1.0 (full weight, e.g. oxide / secondary ore)
    #   primary_cu_frac = 1  → scale = 0.20 (80 % reduction, refractory CPY ore)
    _pri_idx = STATIC_PREDICTOR_INDEX.get("copper_primary_sulfides_equivalent")
    _sec_idx = STATIC_PREDICTOR_INDEX.get("copper_secondary_sulfides_equivalent")
    _ox_idx  = STATIC_PREDICTOR_INDEX.get("copper_oxides_equivalent")
    if _pri_idx is not None and _sec_idx is not None and _ox_idx is not None:
        _cu_pri = float(x_raw[:, _pri_idx].mean().clamp(min=0.0))
        _cu_sec = float(x_raw[:, _sec_idx].mean().clamp(min=0.0))
        _cu_ox  = float(x_raw[:, _ox_idx].mean().clamp(min=0.0))
        _cu_tot = _cu_pri + _cu_sec + _cu_ox + 1e-6
        _primary_cu_frac = min(1.0, max(0.0, _cu_pri / _cu_tot))
    else:
        _primary_cu_frac = 0.0
    # linear interpolation: weight_scale = 1 - 0.80 * primary_cu_frac
    _slope_early_mineralogy_scale = 1.0 - 0.80 * _primary_cu_frac

    total = (
        loss_ctrl
        + loss_cat_uplift
        + float(loss_weights.get("gap", 1.0)) * gap_pen
        + float(loss_weights.get("monotonic", 0.02)) * mono_pen
        + float(loss_weights.get("param", 0.0)) * param_pen
        + float(loss_weights.get("primary_fraction_prior", 0.0)) * primary_fraction_prior_pen
        + float(loss_weights.get("smooth_cat", 0.12)) * cat_smooth_pen
        + float(loss_weights.get("slope_cap", 0.18)) * cat_slope_cap_pen
        + float(loss_weights.get("latent_smooth", 0.02)) * latent_smooth_pen
        + float(loss_weights.get("latent_cat_rate", 0.03)) * latent_cat_rate_pen
        + float(loss_weights.get("flat_input_smooth", 0.08)) * flat_input_smooth_pen
        + float(loss_weights.get("flat_input_accel", 0.16)) * flat_input_accel_pen
        + float(loss_weights.get("single_uplift_late_accel", 0.0)) * single_uplift_late_accel_pen
        + float(loss_weights.get("cap", 0.10)) * cap_pen
        + float(loss_weights.get("uplift_fit", 0.5)) * uplift_fit_pen
        + float(loss_weights.get("uplift_tail", 0.0)) * uplift_tail_pen
        + float(loss_weights.get("final_recovery", 0.0)) * final_recovery_pen
        + float(loss_weights.get("final_uplift", 0.0)) * final_uplift_pen
        + float(loss_weights.get("catalyst_use_prior", 0.0)) * catalyst_use_prior_pen
        + float(loss_weights.get("control_interp_fit", 0.0)) * control_interp_fit_pen
        + float(loss_weights.get("control_early_fit", 0.0)) * control_early_fit_pen
        + float(loss_weights.get("control_tail_fit", 0.0)) * control_tail_fit_pen
        + float(loss_weights.get("pre_catalyst_control_fit", 0.0)) * pre_catalyst_control_fit_pen
        + float(loss_weights.get("control_early_process", 0.0)) * control_early_process_pen
        + float(loss_weights.get("tau_onset", 0.0)) * tau_onset_pen
        + float(loss_weights.get("ferric_orp_aux", 0.0)) * ferric_orp_aux_pen
        + float(loss_weights.get("slope_early", 0.0)) * _slope_early_mineralogy_scale * slope_early_pen
        + float(loss_weights.get("t50_match", 0.0)) * t50_match_pen
    )
    return total


def pair_training_loss(
    model: PairCurveNet,
    pair: PairSample,
    cum_scale: float,
    lix_scale: float,
    irrigation_scale: float,
    loss_weights: Dict[str, float],
    max_cat_slope_per_day: float,
    conc_scale: float = 1.0,
) -> torch.Tensor:
    tensors = build_pair_training_batch([pair], cum_scale, lix_scale, irrigation_scale, conc_scale)
    return _pair_training_loss_from_tensors(
        model=model,
        tensors=tensors,
        loss_weights=loss_weights,
        max_cat_slope_per_day=max_cat_slope_per_day,
    )


def pair_training_loss_batch(
    model: PairCurveNet,
    pairs: List[PairSample],
    cum_scale: float,
    lix_scale: float,
    irrigation_scale: float,
    loss_weights: Dict[str, float],
    max_cat_slope_per_day: float,
    conc_scale: float = 1.0,
) -> torch.Tensor:
    tensors = build_pair_training_batch(pairs, cum_scale, lix_scale, irrigation_scale, conc_scale)
    return _pair_training_loss_from_tensors(
        model=model,
        tensors=tensors,
        loss_weights=loss_weights,
        max_cat_slope_per_day=max_cat_slope_per_day,
    )


def _pair_diag_summary(pair: PairSample) -> Dict[str, Any]:
    return {
        "pair_id": str(pair.pair_id),
        "sample_id": str(pair.sample_id),
        "control_col_id": str(pair.control.col_id),
        "catalyzed_col_id": str(pair.catalyzed.col_id),
    }


def _model_has_non_finite_params(model: nn.Module) -> bool:
    for param in model.parameters():
        if not torch.isfinite(param).all():
            return True
    return False


def _model_non_finite_param_names(model: nn.Module, limit: int = 25) -> List[str]:
    names: List[str] = []
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            names.append(str(name))
            if len(names) >= limit:
                break
    return names


def _grad_non_finite_param_names(model: nn.Module, limit: int = 25) -> List[str]:
    names: List[str] = []
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            names.append(str(name))
            if len(names) >= limit:
                break
    return names


def diagnose_non_finite_pair_batch(
    model: PairCurveNet,
    pair_batch: List[PairSample],
    cum_scale: float,
    lix_scale: float,
    irrigation_scale: float,
    loss_weights: Dict[str, float],
    max_cat_slope_per_day: float,
    conc_scale: float,
    stage: str,
    epoch: int,
    batch_index: int,
    seed: int,
    trigger_value: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pair_reports: List[Dict[str, Any]] = []
    suspect_col_ids: List[str] = []

    for pair in pair_batch:
        report = _pair_diag_summary(pair)
        report["train_curve_lengths"] = {
            "control_n": int(len(pair.control.time)),
            "catalyzed_n": int(len(pair.catalyzed.time)),
        }

        try:
            tensors = get_pair_training_tensors(pair, cum_scale, lix_scale, irrigation_scale, conc_scale)
            bad_tensor_keys = []
            for key, value in tensors.items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    bad_tensor_keys.append(str(key))
            report["non_finite_tensor_keys"] = bad_tensor_keys

            pair_loss = pair_training_loss(
                model=model,
                pair=pair,
                cum_scale=cum_scale,
                lix_scale=lix_scale,
                irrigation_scale=irrigation_scale,
                loss_weights=loss_weights,
                max_cat_slope_per_day=max_cat_slope_per_day,
                conc_scale=conc_scale,
            )
            pair_loss_value = float(pair_loss.detach().to(dtype=torch.float32).cpu().item())
            report["pair_loss"] = pair_loss_value
            report["pair_loss_is_finite"] = bool(np.isfinite(pair_loss_value))
        except Exception as exc:
            report["pair_loss"] = np.nan
            report["pair_loss_is_finite"] = False
            report["exception"] = repr(exc)

        if (not report.get("pair_loss_is_finite", False)) or len(report.get("non_finite_tensor_keys", [])) > 0:
            for col_id in [report["control_col_id"], report["catalyzed_col_id"]]:
                if str(col_id).strip():
                    suspect_col_ids.append(str(col_id).strip())
        pair_reports.append(report)

    if len(suspect_col_ids) == 0:
        for pair in pair_batch:
            for col_id in [pair.control.col_id, pair.catalyzed.col_id]:
                if str(col_id).strip():
                    suspect_col_ids.append(str(col_id).strip())

    diagnostics = {
        "stage": str(stage),
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "seed": int(seed),
        "trigger_value": float(trigger_value) if np.isfinite(trigger_value) else np.nan,
        "suspect_col_ids": sorted(set(str(v).strip() for v in suspect_col_ids if str(v).strip())),
        "pair_reports": pair_reports,
    }
    if extra:
        diagnostics.update(extra)
    return diagnostics


def build_adamw_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    params = list(model.parameters())
    optimizer_kwargs: Dict[str, Any] = {
        "lr": float(CONFIG["learning_rate"]),
        "weight_decay": float(CONFIG["weight_decay"]),
    }
    if device.type != "mps":
        optimizer_kwargs["foreach"] = True
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(params, fused=True, **optimizer_kwargs)
        except (TypeError, RuntimeError):
            pass
    try:
        return torch.optim.AdamW(params, **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("foreach", None)
        return torch.optim.AdamW(params, **optimizer_kwargs)


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
    conc_scale: float = 1.0,
) -> Tuple[Optional[PairCurveNet], List[Dict[str, float]], float, Optional[Dict[str, Any]]]:
    set_all_seeds(seed, deterministic=True)
    rng = np.random.default_rng(seed)
    loss_weights = dict(CONFIG["loss_weights"])
    max_cat_slope_per_day = float(CONFIG.get("max_cat_slope_per_day", 0.20))
    bootstrap_train_pairs = bool(CONFIG.get("bootstrap_train_pairs", False))
    pair_batch_size = int(max(1, PAIR_BATCH_SIZE))
    eval_every_n_epochs = int(max(1, EVAL_EVERY_N_EPOCHS))

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
        max_transition_days=float(CONFIG.get("max_transition_days", 250.0)),
        max_catalyst_aging_strength=float(CONFIG.get("max_catalyst_aging_strength", 5.0)),
        late_tau_impact_decay_strength=float(CONFIG.get("late_tau_impact_decay_strength", 1.15)),
        min_remaining_ore_factor=float(CONFIG.get("min_remaining_ore_factor", 0.08)),
        flat_input_transition_sensitivity=float(CONFIG.get("flat_input_transition_sensitivity", 6.0)),
        flat_input_uplift_response_days=float(CONFIG.get("flat_input_uplift_response_days", 75.0)),
        flat_input_response_ramp_days=float(CONFIG.get("flat_input_response_ramp_days", 75.0)),
        flat_input_late_uplift_response_boost=float(
            CONFIG.get("flat_input_late_uplift_response_boost", 2.5)
        ),
        use_prefit_param_bounds=bool(CONFIG.get("use_prefit_param_bounds", True)),
    ).to(device=device, dtype=MODEL_TORCH_DTYPE)

    optimizer = build_adamw_optimizer(model)

    history: List[Dict[str, float]] = []
    best_state = None
    best_eval = np.inf
    last_improve_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        if bootstrap_train_pairs:
            epoch_indices = rng.choice(len(train_pairs), size=len(train_pairs), replace=True)
            epoch_pairs = [train_pairs[i] for i in epoch_indices]
        else:
            epoch_pairs = list(train_pairs)
        rng.shuffle(epoch_pairs)
        train_loss_sum: Optional[torch.Tensor] = None
        train_loss_count = 0
        epoch_pair_batches = iter_pair_batches(epoch_pairs, pair_batch_size, rng=rng)
        for batch_index, pair_batch in enumerate(epoch_pair_batches):
            optimizer.zero_grad(set_to_none=True)
            loss = pair_training_loss_batch(
                model=model,
                pairs=pair_batch,
                cum_scale=cum_scale,
                lix_scale=lix_scale,
                irrigation_scale=irrigation_scale,
                loss_weights=loss_weights,
                max_cat_slope_per_day=max_cat_slope_per_day,
                conc_scale=conc_scale,
            )
            loss_value = float(loss.detach().to(dtype=torch.float32).cpu().item())
            if not np.isfinite(loss_value):
                diagnostics = diagnose_non_finite_pair_batch(
                    model=model,
                    pair_batch=pair_batch,
                    cum_scale=cum_scale,
                    lix_scale=lix_scale,
                    irrigation_scale=irrigation_scale,
                    loss_weights=loss_weights,
                    max_cat_slope_per_day=max_cat_slope_per_day,
                    conc_scale=conc_scale,
                    stage="train_loss",
                    epoch=epoch,
                    batch_index=batch_index,
                    seed=seed,
                    trigger_value=loss_value,
                )
                return None, history, best_eval, diagnostics
            loss.backward()
            grad_non_finite_names = _grad_non_finite_param_names(model)
            if len(grad_non_finite_names) > 0:
                diagnostics = diagnose_non_finite_pair_batch(
                    model=model,
                    pair_batch=pair_batch,
                    cum_scale=cum_scale,
                    lix_scale=lix_scale,
                    irrigation_scale=irrigation_scale,
                    loss_weights=loss_weights,
                    max_cat_slope_per_day=max_cat_slope_per_day,
                    conc_scale=conc_scale,
                    stage="train_gradients",
                    epoch=epoch,
                    batch_index=batch_index,
                    seed=seed,
                    trigger_value=loss_value,
                    extra={"non_finite_gradient_params": grad_non_finite_names},
                )
                return None, history, best_eval, diagnostics
            nn.utils.clip_grad_norm_(model.parameters(), float(CONFIG["grad_clip_norm"]))
            optimizer.step()
            if _model_has_non_finite_params(model):
                diagnostics = diagnose_non_finite_pair_batch(
                    model=model,
                    pair_batch=pair_batch,
                    cum_scale=cum_scale,
                    lix_scale=lix_scale,
                    irrigation_scale=irrigation_scale,
                    loss_weights=loss_weights,
                    max_cat_slope_per_day=max_cat_slope_per_day,
                    conc_scale=conc_scale,
                    stage="train_parameters_after_step",
                    epoch=epoch,
                    batch_index=batch_index,
                    seed=seed,
                    trigger_value=loss_value,
                    extra={"non_finite_param_names": _model_non_finite_param_names(model)},
                )
                return None, history, best_eval, diagnostics
            detached_loss = loss.detach()
            train_loss_sum = detached_loss if train_loss_sum is None else (train_loss_sum + detached_loss)
            train_loss_count += 1
        train_loss = (
            float((train_loss_sum / train_loss_count).to(dtype=torch.float32).cpu().item())
            if train_loss_sum is not None and train_loss_count > 0
            else np.nan
        )

        should_eval = (epoch == 1) or (epoch % eval_every_n_epochs == 0) or (epoch == epochs)
        eval_loss = np.nan
        if should_eval:
            model.eval()
            eval_pool = val_pairs if len(val_pairs) > 0 else train_pairs
            eval_loss_sum: Optional[torch.Tensor] = None
            eval_loss_count = 0
            with torch.inference_mode():
                eval_pair_batches = iter_pair_batches(list(eval_pool), pair_batch_size)
                for batch_index, pair_batch in enumerate(eval_pair_batches):
                    loss = pair_training_loss_batch(
                        model=model,
                        pairs=pair_batch,
                        cum_scale=cum_scale,
                        lix_scale=lix_scale,
                        irrigation_scale=irrigation_scale,
                        loss_weights=loss_weights,
                        max_cat_slope_per_day=max_cat_slope_per_day,
                        conc_scale=conc_scale,
                    )
                    loss_value = float(loss.detach().to(dtype=torch.float32).cpu().item())
                    if not np.isfinite(loss_value):
                        diagnostics = diagnose_non_finite_pair_batch(
                            model=model,
                            pair_batch=pair_batch,
                            cum_scale=cum_scale,
                            lix_scale=lix_scale,
                            irrigation_scale=irrigation_scale,
                            loss_weights=loss_weights,
                            max_cat_slope_per_day=max_cat_slope_per_day,
                            conc_scale=conc_scale,
                            stage="eval_loss",
                            epoch=epoch,
                            batch_index=batch_index,
                            seed=seed,
                            trigger_value=loss_value,
                        )
                        return None, history, best_eval, diagnostics
                    detached_loss = loss.detach()
                    eval_loss_sum = detached_loss if eval_loss_sum is None else (eval_loss_sum + detached_loss)
                    eval_loss_count += 1
            eval_loss = (
                float((eval_loss_sum / eval_loss_count).to(dtype=torch.float32).cpu().item())
                if eval_loss_sum is not None and eval_loss_count > 0
                else np.nan
            )
        history.append({"epoch": float(epoch), "train_loss": train_loss, "eval_loss": eval_loss})

        if should_eval:
            if eval_loss + 1e-8 < best_eval:
                best_eval = eval_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                last_improve_epoch = epoch
            else:
                if (epoch - last_improve_epoch) >= patience and epoch >= 80:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_state is None or not np.isfinite(best_eval) or _model_has_non_finite_params(model):
        diagnostics = {
            "stage": "checkpoint_selection",
            "epoch": int(len(history)),
            "batch_index": -1,
            "seed": int(seed),
            "trigger_value": float(best_eval) if np.isfinite(best_eval) else np.nan,
            "suspect_col_ids": [],
            "pair_reports": [],
            "reason": "no_finite_checkpoint_or_model_params",
            "non_finite_param_names": _model_non_finite_param_names(model),
        }
        return None, history, best_eval, diagnostics
    return model, history, best_eval, None


def format_member_tag(
    member_idx: int,
    split_seed_idx: int,
    repeat_idx: int,
    fold_idx: int,
    member_seed: int,
) -> str:
    return (
        f"m{member_idx:03d}_z{split_seed_idx + 1:02d}"
        f"_r{repeat_idx + 1:02d}_f{fold_idx + 1:02d}_s{member_seed}"
    )


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
        max_transition_days=float(cfg.get("max_transition_days", CONFIG.get("max_transition_days", 250.0))),
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
        use_prefit_param_bounds=bool(
            cfg.get("use_prefit_param_bounds", CONFIG.get("use_prefit_param_bounds", True))
        ),
    ).to(device=device, dtype=MODEL_TORCH_DTYPE)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as exc:
        load_result = model.load_state_dict(checkpoint["state_dict"], strict=False)
        missing_ok = all(
            k.startswith("interaction_weight_params.")
            or k in {
                "height_delay_scale",
                "p80_tau_scale",
                "p80_temp_scale",
                "p80_kappa_penalty_scale",
                "material_transport_scale",
                "geometry_response_scale",
                "temp_active_mean_scale",
                "temp_active_recent_scale",
                "orp_aux_scale",
                "orp_aux_bias",
            }
            for k in load_result.missing_keys
        )
        unexpected_ok = all(
            k.startswith("interaction_weight_params.")
            for k in load_result.unexpected_keys
        )
        if not unexpected_ok or not missing_ok:
            raise exc
    return model


def train_validation_member_job(job: Dict[str, Any]) -> Dict[str, Any]:
    configure_torch_cpu_parallelism(
        num_threads=int(job.get("torch_threads_per_worker", 0)),
        num_interop_threads=int(job.get("torch_interop_threads_per_worker", 1)),
    )

    member_idx = int(job["member_idx"])
    split_seed_idx = int(job.get("split_seed_idx", 0))
    split_random_state = int(job.get("split_random_state", CONFIG.get("cv_random_state", CONFIG["seed"])))
    repeat_idx = int(job["repeat_idx"])
    fold_idx = int(job["fold_idx"])
    member_seed = int(job["member_seed"])
    member_tag = str(job["member_tag"])
    train_pairs_seed: List[PairSample] = job["train_pairs"]
    val_pairs_seed: List[PairSample] = job["val_pairs"]
    val_pairs_metrics_seed: List[PairSample] = job.get("val_pairs_metrics", val_pairs_seed)
    if len(val_pairs_seed) == 0:
        raise ValueError(f"CV member {member_idx} produced empty validation split.")
    if len(val_pairs_metrics_seed) == 0:
        raise ValueError(f"CV member {member_idx} produced empty validation metrics split.")

    train_sample_ids = sorted({p.sample_id for p in train_pairs_seed})
    validation_sample_ids = sorted({p.sample_id for p in val_pairs_seed})
    validation_metrics_sample_ids = sorted({p.sample_id for p in val_pairs_metrics_seed})
    overlapping_sample_ids = sorted(set(train_sample_ids) & set(validation_sample_ids))
    if overlapping_sample_ids:
        raise ValueError(
            f"CV member {member_idx} mixes project_sample_id across train/validation: "
            f"{overlapping_sample_ids[:10]}"
        )
    if set(validation_sample_ids) != set(validation_metrics_sample_ids):
        raise ValueError(
            f"CV member {member_idx} has mismatched validation sample sets between training/eval folds."
        )

    print(
        f"[CV Ensemble] Training {member_tag} "
        f"(train={len(train_pairs_seed)}, val={len(val_pairs_seed)})"
    )

    imputer_member, scaler_member = fit_static_transformers(train_pairs_seed)
    apply_static_transformers(train_pairs_seed, imputer_member, scaler_member)
    apply_static_transformers(val_pairs_seed, imputer_member, scaler_member)
    apply_static_transformers(val_pairs_metrics_seed, imputer_member, scaler_member)
    orp_aux_norm_stats = fit_orp_aux_normalization(train_pairs_seed)
    apply_orp_aux_normalization(train_pairs_seed, orp_aux_norm_stats)
    apply_orp_aux_normalization(val_pairs_seed, orp_aux_norm_stats)
    apply_orp_aux_normalization(val_pairs_metrics_seed, orp_aux_norm_stats)

    ctrl_seed_params = np.vstack([p.control.fit_params for p in train_pairs_seed])
    cat_seed_params = np.vstack([p.catalyzed.fit_params for p in train_pairs_seed])
    ctrl_lb_seed, ctrl_ub_seed = derive_param_bounds(ctrl_seed_params, None)
    cat_lb_seed, cat_ub_seed = derive_param_bounds(cat_seed_params, None)

    model, history, best_eval, invalid_diagnostics = train_one_member(
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
        conc_scale=float(job.get("conc_scale", 1.0)),
    )

    if model is None:
        return {
            "member_tag": member_tag,
            "member_idx": member_idx,
            "split_seed_idx": split_seed_idx,
            "split_random_state": split_random_state,
            "repeat_idx": repeat_idx,
            "fold_idx": fold_idx,
            "seed": member_seed,
            "invalid_member": True,
            "invalid_diagnostics": invalid_diagnostics or {},
            "best_eval_loss": float(best_eval),
            "best_epoch": -1,
            "n_train_pairs": int(len(train_pairs_seed)),
            "n_validation_pairs": int(len(val_pairs_seed)),
            "history": history,
            "train_pair_ids": [p.pair_id for p in train_pairs_seed],
            "validation_pair_ids": [p.pair_id for p in val_pairs_metrics_seed],
            "train_sample_ids": train_sample_ids,
            "validation_sample_ids": validation_metrics_sample_ids,
            "records_val": [],
            "model_ckpt_path": "",
            "imputer": imputer_member,
            "scaler": scaler_member,
        }

    model_ckpt_path = os.path.join(str(job["val_member_model_root"]), f"{member_tag}.pt")
    torch.save(
        {
            "member_tag": member_tag,
            "member_idx": member_idx,
            "split_seed_idx": split_seed_idx,
            "split_random_state": split_random_state,
            "repeat_idx": repeat_idx,
            "fold_idx": fold_idx,
            "seed": member_seed,
            "model_logic_version": MODEL_LOGIC_VERSION,
            "state_dict": model_state_dict_to_cpu(model),
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
            "conc_scale": float(job.get("conc_scale", 1.0)),
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
        eval_hist_df = hist_df[np.isfinite(hist_df["eval_loss"].astype(float))]
        best_epoch = int(eval_hist_df.sort_values("eval_loss").iloc[0]["epoch"]) if len(eval_hist_df) > 0 else -1
    else:
        best_epoch = -1

    metrics_val, records_val = evaluate_model(
        model,
        val_pairs_seed,
        float(job["cum_scale"]),
        float(job["lix_scale"]),
        float(job["irrigation_scale"]),
        conc_scale=float(job.get("conc_scale", 1.0)),
    )
    _, records_val_ensemble = evaluate_model(
        model,
        val_pairs_metrics_seed,
        float(job["cum_scale"]),
        float(job["lix_scale"]),
        float(job["irrigation_scale"]),
        conc_scale=float(job.get("conc_scale", 1.0)),
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
            plot_path=os.path.join(member_plot_dir, f"{r['pair_id']}.png"),
            title=f"Validation Prediction ({r['sample_id']} | {r['pair_id']}) - {member_tag}",
        )

    result = {
        "member_tag": member_tag,
        "member_idx": member_idx,
        "split_seed_idx": split_seed_idx,
        "split_random_state": split_random_state,
        "repeat_idx": repeat_idx,
        "fold_idx": fold_idx,
        "seed": member_seed,
        "metrics_val": metrics_val,
        "best_eval_loss": float(best_eval),
        "best_epoch": int(best_epoch),
        "n_train_pairs": int(len(train_pairs_seed)),
        "n_validation_pairs": int(len(val_pairs_seed)),
        "history": history,
        "train_pair_ids": [p.pair_id for p in train_pairs_seed],
        "validation_pair_ids": [p.pair_id for p in val_pairs_seed],
        "train_sample_ids": train_sample_ids,
        "validation_sample_ids": validation_sample_ids,
        "records_val": records_val,
        "records_val_ensemble": records_val_ensemble,
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
    conc_scale: float = 1.0,
) -> Dict[str, Any]:
    model.eval()
    tensors = get_pair_training_tensors(pair, cum_scale, lix_scale, irrigation_scale, conc_scale)
    plot_tensors = get_pair_plot_tensors(pair, cum_scale, lix_scale, irrigation_scale)
    with torch.inference_mode():
        x = tensors["x"]
        x_raw = tensors["x_raw"]
        x_input_only = tensors["x_input_only"]
        ctrl_t = tensors["ctrl_t"]
        ctrl_c = tensors["ctrl_c"]
        ctrl_l = tensors["ctrl_l"]
        ctrl_irr = tensors["ctrl_irr"]
        cat_t = tensors["cat_t"]
        cat_c = tensors["cat_c"]
        cat_l = tensors["cat_l"]
        cat_irr = tensors["cat_irr"]
        cat_conc = tensors.get("cat_conc")

        p_ctrl, p_cat, tau, temp, kappa, aging_strength, latent = model.predict_params(
            x,
            x_raw,
            x_input_only,
            ctrl_curve_override_raw=tensors.get("ctrl_curve_override_raw"),
            cat_curve_override_raw=tensors.get("cat_curve_override_raw"),
            ctrl_cap_anchor_pct=tensors.get("ctrl_cap_anchor_pct"),
            cat_cap_anchor_pct=tensors.get("cat_cap_anchor_pct"),
            ctrl_cap_anchor_active=tensors.get("ctrl_cap_anchor_active"),
            cat_cap_anchor_active=tensors.get("cat_cap_anchor_active"),
            catalyst_t_days=cat_t,
            catalyst_cum_norm=cat_c,
            catalyst_conc_norm=cat_conc,
        )
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
            catalyst_start_day_override=tensors["ctrl_start_day"],
            latent_params=latent,
            return_states=True,
        )
        pred_ctrl_cat_time, _ = model.curves_given_params(
            p_ctrl,
            p_cat,
            cat_t,
            tensors["cat_ctrl_c"],
            cat_l,
            cat_irr,
            tau,
            temp,
            kappa,
            aging_strength,
            catalyst_start_day_override=tensors["cat_ctrl_start_day"],
            latent_params=latent,
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
            catalyst_start_day_override=tensors["cat_effective_start_day"],
            latent_params=latent,
            return_states=True,
        )
        plot_profile = plot_tensors["plot_profile"]
        control_plot_time_days = np.asarray(plot_tensors["control_plot_time_days"], dtype=float)
        pred_ctrl_plot, _ = model.curves_given_params(
            p_ctrl,
            p_cat,
            plot_tensors["ctrl_plot_t"],
            plot_tensors["ctrl_plot_c"],
            plot_tensors["plot_l"],
            plot_tensors["plot_irr"],
            tau,
            temp,
            kappa,
            aging_strength,
            catalyst_start_day_override=plot_tensors["ctrl_plot_start_day"],
            latent_params=latent,
        )

        _, pred_cat_plot = model.curves_given_params(
            p_ctrl,
            p_cat,
            plot_tensors["ctrl_plot_t"],
            plot_tensors["cat_plot_c"],
            plot_tensors["plot_l"],
            plot_tensors["plot_irr"],
            tau,
            temp,
            kappa,
            aging_strength,
            catalyst_start_day_override=plot_tensors["cat_plot_effective_start_day"],
            latent_params=latent,
        )
        pred_cat_plot = torch.maximum(pred_cat_plot, pred_ctrl_plot)

    effective_ctrl_cap_prior = float(latent["effective_ctrl_cap_prior"].squeeze().detach().cpu().item())
    effective_cat_cap_prior = float(latent["effective_cat_cap_prior"].squeeze().detach().cpu().item())
    cap_prior_tolerance = float(CONFIG.get("cap_prior_violation_tolerance_pct", 1.0))
    control_cap_violation = bool(
        np.isfinite(effective_ctrl_cap_prior)
        and np.nanmax(np.asarray(pair.control.recovery, dtype=float)) > effective_ctrl_cap_prior + cap_prior_tolerance
    )
    catalyzed_cap_violation = bool(
        np.isfinite(effective_cat_cap_prior)
        and np.nanmax(np.asarray(pair.catalyzed.recovery, dtype=float)) > effective_cat_cap_prior + cap_prior_tolerance
    )

    rec = {
        "pair_id": pair.pair_id,
        "sample_id": pair.sample_id,
        "control_col_id": pair.control.col_id,
        "catalyzed_col_id": pair.catalyzed.col_id,
        "control_start_day": float(pair.control.catalyst_start_day),
        "catalyzed_start_day": float(pair.catalyzed.catalyst_start_day),
        "control_start_day_source": str(pair.control.catalyst_start_day_source),
        "catalyzed_start_day_source": str(pair.catalyzed.catalyst_start_day_source),
        "control_avg_catalyst_dose_mg_l": float(pair.control.avg_catalyst_dose_mg_l),
        "catalyzed_avg_catalyst_dose_mg_l": float(pair.catalyzed.avg_catalyst_dose_mg_l),
        "control_t": pair.control.time.copy(),
        "control_true": pair.control.recovery.copy(),
        "control_last_actual_day": float(pair.control.last_actual_day),
        "control_pred": tensor_to_numpy_float32(pred_ctrl),
        "catalyzed_t": pair.catalyzed.time.copy(),
        "catalyzed_true": pair.catalyzed.recovery.copy(),
        "catalyzed_last_actual_day": float(pair.catalyzed.last_actual_day),
        "cumulative_lixiviant_m3_t": pair.catalyzed.lixiviant_cum.copy(),
        "irrigation_rate_l_m2_h": pair.catalyzed.irrigation_rate_l_m2_h.copy(),
        CATALYST_ADDITION_RECON_COL: pair.catalyzed.catalyst_addition_mg_l_reconstructed.copy(),
        "catalyzed_pred": tensor_to_numpy_float32(pred_cat),
        "control_pred_on_catalyzed_t": tensor_to_numpy_float32(pred_ctrl_cat_time),
        "control_true_on_catalyzed_t": np.asarray(
            np.interp(
                np.asarray(pair.catalyzed.time, dtype=float),
                np.asarray(pair.control.time, dtype=float),
                np.asarray(pair.control.recovery, dtype=float),
                left=float(pair.control.recovery[0]),
                right=float(pair.control.recovery[-1]),
            ),
            dtype=float,
        ),
        "true_uplift": np.asarray(
            np.maximum(
                np.asarray(pair.catalyzed.recovery, dtype=float)
                - np.interp(
                    np.asarray(pair.catalyzed.time, dtype=float),
                    np.asarray(pair.control.time, dtype=float),
                    np.asarray(pair.control.recovery, dtype=float),
                    left=float(pair.control.recovery[0]),
                    right=float(pair.control.recovery[-1]),
                ),
                0.0,
            ),
            dtype=float,
        ),
        "pred_uplift": tensor_to_numpy_float32(torch.clamp(pred_cat - pred_ctrl_cat_time, min=0.0)),
        "tau_days": float(cat_states["effective_tau_days"].squeeze().detach().cpu().item()),
        "temp_days": float(temp.squeeze().detach().cpu().item()),
        "kappa": float(kappa.squeeze().detach().cpu().item()),
        "initial_gate_strength": float(latent["initial_gate_strength"].squeeze().detach().cpu().item()),
        "initial_gate_mid_day": float(latent["initial_gate_mid_day"].squeeze().detach().cpu().item()),
        "initial_gate_width_day": float(latent["initial_gate_width_day"].squeeze().detach().cpu().item()),
        "aging_strength": float(aging_strength.squeeze().detach().cpu().item()),
        "mid_life_days": float(latent["mid_life_days"].squeeze().detach().cpu().item()),
        "catalyst_use_frac": float(latent["catalyst_use_frac"].squeeze().detach().cpu().item()),
        "active_catalyst_mean": float(latent["active_catalyst_mean"].squeeze().detach().cpu().item()),
        "active_catalyst_peak": float(latent["active_catalyst_peak"].squeeze().detach().cpu().item()),
        "active_catalyst_recent": float(latent["active_catalyst_recent"].squeeze().detach().cpu().item()),
        "chem_interaction": float(latent["chem_interaction"].squeeze().detach().cpu().item()),
        "primary_passivation_drive": float(latent["primary_passivation_drive"].squeeze().detach().cpu().item()),
        "ferric_ready_primary_score": float(latent["ferric_ready_primary_score"].squeeze().detach().cpu().item()),
        "tall_column_oxidant_deficit_score": float(
            latent["tall_column_oxidant_deficit_score"].squeeze().detach().cpu().item()
        ),
        "primary_catalyst_synergy": float(latent["primary_catalyst_synergy"].squeeze().detach().cpu().item()),
        "primary_control_prior_frac": float(latent["primary_control_prior_frac"].squeeze().detach().cpu().item()),
        "primary_catalyzed_prior_frac": float(
            latent["primary_catalyzed_prior_frac"].squeeze().detach().cpu().item()
        ),
        "base_ctrl_cap": float(latent["base_ctrl_cap"].squeeze().detach().cpu().item()),
        "base_cat_cap": float(latent["base_cat_cap"].squeeze().detach().cpu().item()),
        "effective_ctrl_cap_prior": effective_ctrl_cap_prior,
        "effective_cat_cap_prior": effective_cat_cap_prior,
        "control_terminal_slope_rate": resolve_static_feature_value_from_vector(
            pair.control_static_raw if pair.control_static_raw.size > 0 else pair.static_raw,
            STATIC_PREDICTOR_COLUMNS,
            "terminal_slope_rate",
            default=np.nan,
        ),
        "catalyzed_terminal_slope_rate": resolve_static_feature_value_from_vector(
            pair.catalyzed_static_raw if pair.catalyzed_static_raw.size > 0 else pair.static_raw,
            STATIC_PREDICTOR_COLUMNS,
            "terminal_slope_rate",
            default=np.nan,
        ),
        "control_cap_anchor_pct": float(pair.control_cap_anchor_pct),
        "catalyzed_cap_anchor_pct": float(pair.catalyzed_cap_anchor_pct),
        "control_cap_anchor_active": bool(pair.control_cap_anchor_active),
        "catalyzed_cap_anchor_active": bool(pair.catalyzed_cap_anchor_active),
        "cap_prior_violation_flag": bool(control_cap_violation or catalyzed_cap_violation),
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
        "control_pred_plot": tensor_to_numpy_float32(pred_ctrl_plot),
        "catalyzed_plot_time_days": np.asarray(control_plot_time_days, dtype=float),
        "plot_catalyst_cum": np.asarray(plot_profile["plot_catalyst_cum"], dtype=float),
        "plot_lixiviant_cum": np.asarray(plot_profile["plot_lixiviant_cum"], dtype=float),
        "plot_irrigation_rate_l_m2_h": np.asarray(plot_profile["plot_irrigation_rate_l_m2_h"], dtype=float),
        f"plot_{CATALYST_ADDITION_RECON_COL}": np.asarray(
            plot_profile[f"plot_{CATALYST_ADDITION_RECON_COL}"],
            dtype=float,
        ),
        "catalyzed_pred_plot": tensor_to_numpy_float32(pred_cat_plot),
        "average_catalyst_dose_mg_l": float(pair.catalyzed.avg_catalyst_dose_mg_l),
        "catalyst_addition_start_day": float(pair.catalyzed.catalyst_start_day),
        "catalyst_addition_stop_day": float(plot_profile["catalyst_addition_stop_day"]),
        "weekly_catalyst_addition_kg_t": float(plot_profile["weekly_catalyst_addition_kg_t"]),
        "weekly_catalyst_extension_kg_t": float(plot_profile.get("weekly_catalyst_extension_kg_t", np.nan)),
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
        "plot_target_day": float(plot_profile["target_day"]),
        "extension_target_day": float(plot_profile["target_day"]),
    }
    if bool(CONFIG.get("member_prediction_gap_band", True)):
        rec = add_member_prediction_gap_bands(rec)
    return rec


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.size < 2:
        return np.nan
    if np.nanstd(yt) < 1e-12:
        return np.nan
    return float(r2_score(yt, yp))


def _nanmean_array(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = np.asarray(stack, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    with np.errstate(all="ignore"):
        return np.nanmean(arr, axis=axis)


def _nanpercentile_array(stack: np.ndarray, q: float, axis: int = 0) -> np.ndarray:
    arr = np.asarray(stack, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    with np.errstate(all="ignore"):
        return np.nanpercentile(arr, q, axis=axis)


def _nanmean_scalar(values: List[Any], default: float = np.nan) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.mean(finite))


def _record_has_finite_predictions(record: Dict[str, Any]) -> bool:
    arrays = [
        np.asarray(record.get("control_pred", []), dtype=float),
        np.asarray(record.get("catalyzed_pred", []), dtype=float),
    ]
    return any(arr.size > 0 and np.any(np.isfinite(arr)) for arr in arrays)


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

    def _m(y_true_, y_pred_, tag):
        y_true_ = np.asarray(y_true_, dtype=float).reshape(-1)
        y_pred_ = np.asarray(y_pred_, dtype=float).reshape(-1)

        mask = np.isfinite(y_true_) & np.isfinite(y_pred_)
        n_total = len(y_true_)
        n_valid = int(mask.sum())
        n_dropped = n_total - n_valid

        if n_valid == 0:
            return {
                f"{tag}_n_total": int(n_total),
                f"{tag}_n_valid": 0,
                f"{tag}_n_dropped_nan": int(n_dropped),
                f"{tag}_rmse": np.nan,
                f"{tag}_mae": np.nan,
                f"{tag}_r2": np.nan,
                f"{tag}_bias": np.nan,
            }

        y_true_f = y_true_[mask]
        y_pred_f = y_pred_[mask]

        return {
            f"{tag}_n_total": int(n_total),
            f"{tag}_n_valid": int(n_valid),
            f"{tag}_n_dropped_nan": int(n_dropped),
            f"{tag}_rmse": float(np.sqrt(mean_squared_error(y_true_f, y_pred_f))),
            f"{tag}_mae": float(mean_absolute_error(y_true_f, y_pred_f)),
            f"{tag}_r2": float(r2_score(y_true_f, y_pred_f)) if n_valid >= 2 else np.nan,
            f"{tag}_bias": float(np.mean(y_pred_f - y_true_f)),
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


def _build_signal_scale_sources(pairs: List[PairSample], conc_percentile: float = 95.0) -> Dict[str, Dict[str, Any]]:
    """Capture which sample/column produced each normalization scale."""

    def _curve_entry(pair: PairSample, curve: CurveData, value: float, status: str) -> Dict[str, Any]:
        return {
            "project_sample_id": str(pair.sample_id).strip(),
            "project_col_id": str(curve.col_id).strip(),
            "status": str(status),
            "value": float(value),
        }

    cum_entries = [
        _curve_entry(
            pair=p,
            curve=p.catalyzed,
            value=float(np.nanmax(p.catalyzed.catalyst_cum)),
            status="Catalyzed",
        )
        for p in pairs
        if p.catalyzed.catalyst_cum.size > 0 and np.any(np.isfinite(p.catalyzed.catalyst_cum))
    ]
    lix_entries = [
        _curve_entry(
            pair=p,
            curve=curve,
            value=float(np.nanmax(curve.lixiviant_cum)),
            status=curve.status,
        )
        for p in pairs
        for curve in [p.control, p.catalyzed]
        if curve.lixiviant_cum.size > 0 and np.any(np.isfinite(curve.lixiviant_cum))
    ]
    irrigation_entries = [
        _curve_entry(
            pair=p,
            curve=curve,
            value=float(np.nanmax(curve.irrigation_rate_l_m2_h)),
            status=curve.status,
        )
        for p in pairs
        for curve in [p.control, p.catalyzed]
        if curve.irrigation_rate_l_m2_h.size > 0 and np.any(np.isfinite(curve.irrigation_rate_l_m2_h))
    ]
    conc_entries = [
        _curve_entry(
            pair=p,
            curve=p.catalyzed,
            value=float(np.nanmax(p.catalyzed.catalyst_conc_col_mg_l)),
            status="Catalyzed",
        )
        for p in pairs
        if p.catalyzed.catalyst_conc_col_mg_l.size > 0
        and np.any(np.isfinite(p.catalyzed.catalyst_conc_col_mg_l))
    ]

    def _max_entry(entries: List[Dict[str, Any]], fallback_value: float) -> Dict[str, Any]:
        if not entries:
            return {
                "value": float(fallback_value),
                "project_sample_id": "",
                "project_col_id": "",
                "status": "",
                "source_kind": "empty",
            }
        best = max(entries, key=lambda item: float(item["value"]))
        return {
            **best,
            "source_kind": "max",
        }

    def _percentile_entry(entries: List[Dict[str, Any]], percentile: float, fallback_value: float) -> Dict[str, Any]:
        if not entries:
            return {
                "value": float(fallback_value),
                "project_sample_id": "",
                "project_col_id": "",
                "status": "",
                "source_kind": f"p{int(percentile)}_empty",
                "reference_value": float(fallback_value),
            }
        values = np.asarray([float(item["value"]) for item in entries], dtype=float)
        percentile_value = float(np.nanpercentile(values, percentile))
        nearest = min(
            entries,
            key=lambda item: (
                abs(float(item["value"]) - percentile_value),
                float(item["value"]) < percentile_value,
            ),
        )
        return {
            **nearest,
            "value": percentile_value,
            "reference_value": float(nearest["value"]),
            "source_kind": f"p{int(percentile)}_nearest_column_max",
        }

    return {
        "cum_scale": _max_entry(cum_entries, fallback_value=1e-6),
        "lix_scale": _max_entry(lix_entries, fallback_value=1e-6),
        "irrigation_scale": _max_entry(irrigation_entries, fallback_value=1e-6),
        "conc_scale": _percentile_entry(conc_entries, percentile=conc_percentile, fallback_value=1.0),
    }


def evaluate_model(
    model: PairCurveNet,
    pairs: List[PairSample],
    cum_scale: float,
    lix_scale: float,
    irrigation_scale: float,
    conc_scale: float = 1.0,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    assert_pairs_respect_training_exclusions(pairs, context="evaluate_model input")
    records = [predict_pair_record(model, pair, cum_scale, lix_scale, irrigation_scale, conc_scale) for pair in pairs]
    metrics = compute_metrics_from_records(records, ensemble=False)
    return metrics, records


def records_to_df(records: List[Dict[str, Any]], ensemble: bool = False) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {
            "pair_id": r.get("pair_id", r["sample_id"]),
            "sample_id": r["sample_id"],
        }
        scalar_keys = [
            "control_col_id",
            "catalyzed_col_id",
            "control_col_id_oof_member_count",
            "catalyzed_col_id_oof_member_count",
            "control_start_day",
            "catalyzed_start_day",
            "control_start_day_source",
            "catalyzed_start_day_source",
            "control_avg_catalyst_dose_mg_l",
            "catalyzed_avg_catalyst_dose_mg_l",
            "average_catalyst_dose_mg_l",
            "tau_days",
            "temp_days",
            "kappa",
            "initial_gate_strength",
            "initial_gate_mid_day",
            "initial_gate_width_day",
            "aging_strength",
            "mid_life_days",
            "catalyst_use_frac",
            "active_catalyst_mean",
            "active_catalyst_peak",
            "active_catalyst_recent",
            "base_ctrl_cap",
            "base_cat_cap",
            "effective_ctrl_cap_prior",
            "effective_cat_cap_prior",
            "control_terminal_slope_rate",
            "catalyzed_terminal_slope_rate",
            "control_cap_anchor_pct",
            "catalyzed_cap_anchor_pct",
            "control_cap_anchor_active",
            "catalyzed_cap_anchor_active",
            "primary_control_prior_frac",
            "primary_catalyzed_prior_frac",
            "ferric_ready_primary_score",
            "tall_column_oxidant_deficit_score",
            "cap_prior_violation_flag",
            "n_members_total",
            "n_members_excluded_all_nan",
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
            "weekly_catalyst_extension_kg_t",
            "weekly_reference_days",
            "recent_window_start_day",
            "recent_window_delta_kg_t",
            "recent_window_delta_tol_kg_t",
            "recent_window_growth_near_zero",
            "last_observed_day",
            "stopped_before_test_end",
            "catalyst_addition_state",
            "extension_applied",
            "plot_target_day",
            "extension_target_day",
        ]
        for k in scalar_keys:
            if k in r:
                row[k] = r[k]
        array_keys = [
            "control_t",
            "control_true",
            "control_pred",
            "control_pred_gap_low",
            "control_pred_gap_high",
            "control_pred_on_catalyzed_t",
            "catalyzed_t",
            "catalyzed_true",
            "catalyzed_pred",
            "catalyzed_pred_gap_low",
            "catalyzed_pred_gap_high",
            "control_plot_time_days",
            "control_pred_plot",
            "control_pred_plot_gap_low",
            "control_pred_plot_gap_high",
            "catalyzed_plot_time_days",
            "catalyzed_pred_plot",
            "catalyzed_pred_plot_gap_low",
            "catalyzed_pred_plot_gap_high",
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
        pair_id = pair.pair_id
        member_recs_all = [m[pair_id] for m in member_record_maps if pair_id in m]
        if len(member_recs_all) == 0:
            continue
        member_recs = [r for r in member_recs_all if _record_has_finite_predictions(r)]
        if len(member_recs) == 0:
            member_recs = member_recs_all
        control_col_id = str(member_recs_all[0].get("control_col_id", pair.control.col_id)).strip()
        catalyzed_col_id = str(member_recs_all[0].get("catalyzed_col_id", pair.catalyzed.col_id)).strip()
        control_col_id_oof_member_count = int(
            sum(
                1
                for r in member_recs_all
                if str(r.get("control_col_id", pair.control.col_id)).strip() == control_col_id
            )
        )
        catalyzed_col_id_oof_member_count = int(
            sum(
                1
                for r in member_recs_all
                if str(r.get("catalyzed_col_id", pair.catalyzed.col_id)).strip() == catalyzed_col_id
            )
        )
        ctrl_stack = np.vstack([np.asarray(r["control_pred"], dtype=float) for r in member_recs])
        cat_stack = np.vstack([np.asarray(r["catalyzed_pred"], dtype=float) for r in member_recs])
        ctrl_on_cat_stack = np.vstack([np.asarray(r["control_pred_on_catalyzed_t"], dtype=float) for r in member_recs])
        ctrl_low_stack = np.vstack(
            [np.asarray(r.get("control_pred_gap_low", r["control_pred"]), dtype=float) for r in member_recs]
        )
        ctrl_high_stack = np.vstack(
            [np.asarray(r.get("control_pred_gap_high", r["control_pred"]), dtype=float) for r in member_recs]
        )
        cat_low_stack = np.vstack(
            [np.asarray(r.get("catalyzed_pred_gap_low", r["catalyzed_pred"]), dtype=float) for r in member_recs]
        )
        cat_high_stack = np.vstack(
            [np.asarray(r.get("catalyzed_pred_gap_high", r["catalyzed_pred"]), dtype=float) for r in member_recs]
        )
        plot_ctrl_stack = np.vstack(
            [np.asarray(r.get("control_pred_plot", r["control_pred_on_catalyzed_t"]), dtype=float) for r in member_recs]
        )
        plot_cat_stack = np.vstack(
            [np.asarray(r.get("catalyzed_pred_plot", r["catalyzed_pred"]), dtype=float) for r in member_recs]
        )
        plot_ctrl_low_stack = np.vstack(
            [
                np.asarray(
                    r.get("control_pred_plot_gap_low", r.get("control_pred_plot", r["control_pred_on_catalyzed_t"])),
                    dtype=float,
                )
                for r in member_recs
            ]
        )
        plot_ctrl_high_stack = np.vstack(
            [
                np.asarray(
                    r.get("control_pred_plot_gap_high", r.get("control_pred_plot", r["control_pred_on_catalyzed_t"])),
                    dtype=float,
                )
                for r in member_recs
            ]
        )
        plot_cat_low_stack = np.vstack(
            [
                np.asarray(
                    r.get("catalyzed_pred_plot_gap_low", r.get("catalyzed_pred_plot", r["catalyzed_pred"])),
                    dtype=float,
                )
                for r in member_recs
            ]
        )
        plot_cat_high_stack = np.vstack(
            [
                np.asarray(
                    r.get("catalyzed_pred_plot_gap_high", r.get("catalyzed_pred_plot", r["catalyzed_pred"])),
                    dtype=float,
                )
                for r in member_recs
            ]
        )

        rec = {
            "pair_id": pair_id,
            "sample_id": pair.sample_id,
            "control_col_id": control_col_id,
            "catalyzed_col_id": catalyzed_col_id,
            "control_col_id_oof_member_count": control_col_id_oof_member_count,
            "catalyzed_col_id_oof_member_count": catalyzed_col_id_oof_member_count,
            "control_start_day": float(member_recs[0].get("control_start_day", pair.control.catalyst_start_day)),
            "catalyzed_start_day": float(member_recs[0].get("catalyzed_start_day", pair.catalyzed.catalyst_start_day)),
            "control_start_day_source": str(member_recs[0].get("control_start_day_source", pair.control.catalyst_start_day_source)),
            "catalyzed_start_day_source": str(member_recs[0].get("catalyzed_start_day_source", pair.catalyzed.catalyst_start_day_source)),
            "control_avg_catalyst_dose_mg_l": _nanmean_scalar(
                [r.get("control_avg_catalyst_dose_mg_l", 0.0) for r in member_recs],
                default=0.0,
            ),
            "catalyzed_avg_catalyst_dose_mg_l": _nanmean_scalar(
                [r.get("catalyzed_avg_catalyst_dose_mg_l", np.nan) for r in member_recs]
            ),
            "control_t": np.asarray(member_recs[0]["control_t"], dtype=float),
            "control_true": np.asarray(member_recs[0]["control_true"], dtype=float),
            "control_last_actual_day": _nanmean_scalar([r.get("control_last_actual_day", np.nan) for r in member_recs]),
            "catalyzed_last_actual_day": _nanmean_scalar([r.get("catalyzed_last_actual_day", np.nan) for r in member_recs]),
            "control_pred_mean": _nanmean_array(ctrl_stack, axis=0),
            "control_pred_p10": _nanpercentile_array(ctrl_low_stack, pi_low, axis=0),
            "control_pred_p90": _nanpercentile_array(ctrl_high_stack, pi_high, axis=0),
            "control_pred_on_catalyzed_t_mean": _nanmean_array(ctrl_on_cat_stack, axis=0),
            "catalyzed_t": np.asarray(member_recs[0]["catalyzed_t"], dtype=float),
            "catalyzed_true": np.asarray(member_recs[0]["catalyzed_true"], dtype=float),
            "catalyzed_pred_mean": _nanmean_array(cat_stack, axis=0),
            "catalyzed_pred_p10": _nanpercentile_array(cat_low_stack, pi_low, axis=0),
            "catalyzed_pred_p90": _nanpercentile_array(cat_high_stack, pi_high, axis=0),
            "tau_days": _nanmean_scalar([r["tau_days"] for r in member_recs]),
            "temp_days": _nanmean_scalar([r["temp_days"] for r in member_recs]),
            "kappa": _nanmean_scalar([r["kappa"] for r in member_recs]),
            "initial_gate_strength": _nanmean_scalar([r.get("initial_gate_strength", np.nan) for r in member_recs]),
            "initial_gate_mid_day": _nanmean_scalar([r.get("initial_gate_mid_day", np.nan) for r in member_recs]),
            "initial_gate_width_day": _nanmean_scalar([r.get("initial_gate_width_day", np.nan) for r in member_recs]),
            "aging_strength": _nanmean_scalar([r.get("aging_strength", np.nan) for r in member_recs]),
            "mid_life_days": _nanmean_scalar([r.get("mid_life_days", np.nan) for r in member_recs]),
            "catalyst_use_frac": _nanmean_scalar([r.get("catalyst_use_frac", np.nan) for r in member_recs]),
            "active_catalyst_mean": _nanmean_scalar([r.get("active_catalyst_mean", np.nan) for r in member_recs]),
            "active_catalyst_peak": _nanmean_scalar([r.get("active_catalyst_peak", np.nan) for r in member_recs]),
            "active_catalyst_recent": _nanmean_scalar([r.get("active_catalyst_recent", np.nan) for r in member_recs]),
            "base_ctrl_cap": _nanmean_scalar([r.get("base_ctrl_cap", np.nan) for r in member_recs]),
            "base_cat_cap": _nanmean_scalar([r.get("base_cat_cap", np.nan) for r in member_recs]),
            "effective_ctrl_cap_prior": _nanmean_scalar(
                [r.get("effective_ctrl_cap_prior", np.nan) for r in member_recs]
            ),
            "effective_cat_cap_prior": _nanmean_scalar(
                [r.get("effective_cat_cap_prior", np.nan) for r in member_recs]
            ),
            "control_terminal_slope_rate": _nanmean_scalar(
                [r.get("control_terminal_slope_rate", np.nan) for r in member_recs]
            ),
            "catalyzed_terminal_slope_rate": _nanmean_scalar(
                [r.get("catalyzed_terminal_slope_rate", np.nan) for r in member_recs]
            ),
            "control_cap_anchor_pct": _nanmean_scalar(
                [r.get("control_cap_anchor_pct", np.nan) for r in member_recs]
            ),
            "catalyzed_cap_anchor_pct": _nanmean_scalar(
                [r.get("catalyzed_cap_anchor_pct", np.nan) for r in member_recs]
            ),
            "control_cap_anchor_active": bool(
                any(bool(r.get("control_cap_anchor_active", False)) for r in member_recs)
            ),
            "catalyzed_cap_anchor_active": bool(
                any(bool(r.get("catalyzed_cap_anchor_active", False)) for r in member_recs)
            ),
            "primary_control_prior_frac": _nanmean_scalar(
                [r.get("primary_control_prior_frac", np.nan) for r in member_recs]
            ),
            "primary_catalyzed_prior_frac": _nanmean_scalar(
                [r.get("primary_catalyzed_prior_frac", np.nan) for r in member_recs]
            ),
            "ferric_ready_primary_score": _nanmean_scalar(
                [r.get("ferric_ready_primary_score", np.nan) for r in member_recs]
            ),
            "tall_column_oxidant_deficit_score": _nanmean_scalar(
                [r.get("tall_column_oxidant_deficit_score", np.nan) for r in member_recs]
            ),
            "cap_prior_violation_flag": bool(
                any(bool(r.get("cap_prior_violation_flag", False)) for r in member_recs)
            ),
            "n_members_total": int(len(member_recs_all)),
            "n_members": int(len(member_recs)),
            "n_members_excluded_all_nan": int(max(0, len(member_recs_all) - len(member_recs))),
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
            "control_pred_plot_mean": _nanmean_array(plot_ctrl_stack, axis=0),
            "control_pred_plot_p10": _nanpercentile_array(plot_ctrl_low_stack, pi_low, axis=0),
            "control_pred_plot_p90": _nanpercentile_array(plot_ctrl_high_stack, pi_high, axis=0),
            "catalyzed_pred_plot_mean": _nanmean_array(plot_cat_stack, axis=0),
            "catalyzed_pred_plot_p10": _nanpercentile_array(plot_cat_low_stack, pi_low, axis=0),
            "catalyzed_pred_plot_p90": _nanpercentile_array(plot_cat_high_stack, pi_high, axis=0),
            "average_catalyst_dose_mg_l": _nanmean_scalar(
                [r.get("average_catalyst_dose_mg_l", np.nan) for r in member_recs]
            ),
            "catalyst_addition_start_day": float(member_recs[0].get("catalyzed_start_day", np.nan)),
            "catalyst_addition_stop_day": float(member_recs[0].get("catalyst_addition_stop_day", np.nan)),
            "weekly_catalyst_addition_kg_t": _nanmean_scalar(
                [r.get("weekly_catalyst_addition_kg_t", 0.0) for r in member_recs],
                default=0.0,
            ),
            "weekly_catalyst_extension_kg_t": _nanmean_scalar(
                [r.get("weekly_catalyst_extension_kg_t", np.nan) for r in member_recs]
            ),
            "weekly_reference_days": _nanmean_scalar(
                [r.get("weekly_reference_days", 0.0) for r in member_recs],
                default=0.0,
            ),
            "recent_window_start_day": float(member_recs[0].get("recent_window_start_day", np.nan)),
            "recent_window_delta_kg_t": float(member_recs[0].get("recent_window_delta_kg_t", np.nan)),
            "recent_window_delta_tol_kg_t": float(member_recs[0].get("recent_window_delta_tol_kg_t", np.nan)),
            "recent_window_growth_near_zero": bool(member_recs[0].get("recent_window_growth_near_zero", False)),
            "last_observed_day": _nanmean_scalar([r.get("last_observed_day", np.nan) for r in member_recs]),
            "stopped_before_test_end": bool(member_recs[0].get("stopped_before_test_end", False)),
            "catalyst_addition_state": str(member_recs[0].get("catalyst_addition_state", "")),
            "extension_applied": bool(any(bool(r.get("extension_applied", False)) for r in member_recs)),
            "plot_target_day": _nanmean_scalar([r.get("plot_target_day", np.nan) for r in member_recs]),
            "extension_target_day": _nanmean_scalar([r.get("extension_target_day", np.nan) for r in member_recs]),
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
        if bool(CONFIG.get("ensemble_interval_cover_true_curve", True)):
            interval_cover_margin_pct = float(CONFIG.get("ensemble_interval_cover_margin_pct", 0.0))
            rec["control_pred_p10"], rec["control_pred_p90"] = widen_predictive_interval_to_cover_reference(
                time_days=rec["control_t"],
                low_curve=rec["control_pred_p10"],
                high_curve=rec["control_pred_p90"],
                reference_time=rec["control_t"],
                reference_curve=rec["control_true"],
                margin_pct=interval_cover_margin_pct,
            )
            rec["catalyzed_pred_p10"], rec["catalyzed_pred_p90"] = widen_predictive_interval_to_cover_reference(
                time_days=rec["catalyzed_t"],
                low_curve=rec["catalyzed_pred_p10"],
                high_curve=rec["catalyzed_pred_p90"],
                reference_time=rec["catalyzed_t"],
                reference_curve=rec["catalyzed_true"],
                margin_pct=interval_cover_margin_pct,
            )
            rec["control_pred_plot_p10"], rec["control_pred_plot_p90"] = widen_predictive_interval_to_cover_reference(
                time_days=rec["control_plot_time_days"],
                low_curve=rec["control_pred_plot_p10"],
                high_curve=rec["control_pred_plot_p90"],
                reference_time=rec["control_t"],
                reference_curve=rec["control_true"],
                margin_pct=interval_cover_margin_pct,
            )
            rec["catalyzed_pred_plot_p10"], rec["catalyzed_pred_plot_p90"] = widen_predictive_interval_to_cover_reference(
                time_days=rec["catalyzed_plot_time_days"],
                low_curve=rec["catalyzed_pred_plot_p10"],
                high_curve=rec["catalyzed_pred_plot_p90"],
                reference_time=rec["catalyzed_t"],
                reference_curve=rec["catalyzed_true"],
                margin_pct=interval_cover_margin_pct,
            )
        agg_records.append(rec)
    metrics = compute_metrics_from_records(agg_records, ensemble=True)
    return metrics, agg_records


def _curve_error_summary(
    y_true: Any,
    y_pred: Any,
    prefix: str,
) -> Dict[str, float]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(true.size, pred.size)
    out = {
        f"{prefix}_n_total": int(n),
        f"{prefix}_n_valid": 0,
        f"{prefix}_rmse": np.nan,
        f"{prefix}_mae": np.nan,
        f"{prefix}_r2": np.nan,
        f"{prefix}_bias": np.nan,
        f"{prefix}_max_abs_error": np.nan,
        f"{prefix}_final_true": np.nan,
        f"{prefix}_final_pred": np.nan,
        f"{prefix}_final_error": np.nan,
        f"{prefix}_final_abs_error": np.nan,
    }
    if n <= 0:
        return out

    true = true[:n]
    pred = pred[:n]
    mask = np.isfinite(true) & np.isfinite(pred)
    out[f"{prefix}_n_valid"] = int(mask.sum())
    if not np.any(mask):
        return out

    true_f = true[mask]
    pred_f = pred[mask]
    err = pred_f - true_f
    out.update(
        {
            f"{prefix}_rmse": float(np.sqrt(np.mean(err ** 2.0))),
            f"{prefix}_mae": float(np.mean(np.abs(err))),
            f"{prefix}_r2": _safe_r2(true_f, pred_f),
            f"{prefix}_bias": float(np.mean(err)),
            f"{prefix}_max_abs_error": float(np.max(np.abs(err))),
            f"{prefix}_final_true": float(true_f[-1]),
            f"{prefix}_final_pred": float(pred_f[-1]),
            f"{prefix}_final_error": float(err[-1]),
            f"{prefix}_final_abs_error": float(abs(err[-1])),
        }
    )
    return out


def _finite_float_or_nan(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return np.nan
    return x if np.isfinite(x) else np.nan


def _record_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if value is None:
        return False
    try:
        if not np.isfinite(float(value)):
            return False
    except Exception:
        pass
    return bool(value)


def per_sample_metrics_from_records(
    records: List[Dict[str, Any]],
    ensemble: bool = False,
) -> pd.DataFrame:
    ctrl_pred_key = "control_pred_mean" if ensemble else "control_pred"
    cat_pred_key = "catalyzed_pred_mean" if ensemble else "catalyzed_pred"
    ctrl_on_cat_key = "control_pred_on_catalyzed_t_mean" if ensemble else "control_pred_on_catalyzed_t"
    rows: List[Dict[str, Any]] = []

    for r in records:
        ctrl_t = np.asarray(r.get("control_t", []), dtype=float)
        ctrl_true = np.asarray(r.get("control_true", []), dtype=float)
        ctrl_pred = np.asarray(r.get(ctrl_pred_key, []), dtype=float)
        cat_t = np.asarray(r.get("catalyzed_t", []), dtype=float)
        cat_true = np.asarray(r.get("catalyzed_true", []), dtype=float)
        cat_pred = np.asarray(r.get(cat_pred_key, []), dtype=float)
        ctrl_pred_on_cat = np.asarray(r.get(ctrl_on_cat_key, []), dtype=float)

        if ctrl_t.size > 0 and ctrl_true.size > 0 and cat_t.size > 0:
            n_ctrl = min(ctrl_t.size, ctrl_true.size)
            ctrl_true_on_cat = np.interp(
                cat_t,
                ctrl_t[:n_ctrl],
                ctrl_true[:n_ctrl],
                left=float(ctrl_true[:n_ctrl][0]),
                right=float(ctrl_true[:n_ctrl][-1]),
            )
        else:
            ctrl_true_on_cat = np.full_like(cat_true, np.nan, dtype=float)
        n_uplift = min(cat_true.size, cat_pred.size, ctrl_true_on_cat.size, ctrl_pred_on_cat.size)
        if n_uplift > 0:
            true_uplift = np.maximum(cat_true[:n_uplift] - ctrl_true_on_cat[:n_uplift], 0.0)
            pred_uplift = np.maximum(cat_pred[:n_uplift] - ctrl_pred_on_cat[:n_uplift], 0.0)
        else:
            true_uplift = np.asarray([], dtype=float)
            pred_uplift = np.asarray([], dtype=float)

        row: Dict[str, Any] = {
            "pair_id": r.get("pair_id", ""),
            "sample_id": r.get("sample_id", ""),
            "control_col_id": r.get("control_col_id", ""),
            "catalyzed_col_id": r.get("catalyzed_col_id", ""),
            "control_cap_anchor_pct": _finite_float_or_nan(r.get("control_cap_anchor_pct", np.nan)),
            "catalyzed_cap_anchor_pct": _finite_float_or_nan(r.get("catalyzed_cap_anchor_pct", np.nan)),
            "control_cap_anchor_active": _record_bool(r.get("control_cap_anchor_active", False)),
            "catalyzed_cap_anchor_active": _record_bool(r.get("catalyzed_cap_anchor_active", False)),
            "cap_prior_violation_flag": _record_bool(r.get("cap_prior_violation_flag", False)),
            "average_catalyst_dose_mg_l": _finite_float_or_nan(r.get("average_catalyst_dose_mg_l", np.nan)),
            "catalyst_use_frac": _finite_float_or_nan(r.get("catalyst_use_frac", np.nan)),
            "active_catalyst_mean": _finite_float_or_nan(r.get("active_catalyst_mean", np.nan)),
            "active_catalyst_peak": _finite_float_or_nan(r.get("active_catalyst_peak", np.nan)),
            "active_catalyst_recent": _finite_float_or_nan(r.get("active_catalyst_recent", np.nan)),
            "tau_days": _finite_float_or_nan(r.get("tau_days", np.nan)),
            "kappa": _finite_float_or_nan(r.get("kappa", np.nan)),
            "temp_days": _finite_float_or_nan(r.get("temp_days", np.nan)),
            "effective_ctrl_cap_prior": _finite_float_or_nan(r.get("effective_ctrl_cap_prior", np.nan)),
            "effective_cat_cap_prior": _finite_float_or_nan(r.get("effective_cat_cap_prior", np.nan)),
            "base_ctrl_cap": _finite_float_or_nan(r.get("base_ctrl_cap", np.nan)),
            "base_cat_cap": _finite_float_or_nan(r.get("base_cat_cap", np.nan)),
            "primary_control_prior_frac": _finite_float_or_nan(r.get("primary_control_prior_frac", np.nan)),
            "primary_catalyzed_prior_frac": _finite_float_or_nan(r.get("primary_catalyzed_prior_frac", np.nan)),
            "stopped_before_test_end": _record_bool(r.get("stopped_before_test_end", False)),
            "catalyst_addition_state": r.get("catalyst_addition_state", ""),
            "n_members_total": int(r.get("n_members_total", r.get("n_members", 1))),
            "n_members_excluded_all_nan": int(r.get("n_members_excluded_all_nan", 0)),
        }
        row.update(_curve_error_summary(ctrl_true, ctrl_pred, "control"))
        row.update(_curve_error_summary(cat_true, cat_pred, "catalyzed"))
        row.update(_curve_error_summary(true_uplift, pred_uplift, "uplift"))

        combined_true = np.concatenate([ctrl_true.reshape(-1), cat_true.reshape(-1)])
        combined_pred = np.concatenate([ctrl_pred.reshape(-1), cat_pred.reshape(-1)])
        row.update(_curve_error_summary(combined_true, combined_pred, "combined"))
        row["max_curve_mae"] = float(np.nanmax([row["control_mae"], row["catalyzed_mae"]]))
        row["max_final_abs_error"] = float(
            np.nanmax([row["control_final_abs_error"], row["catalyzed_final_abs_error"]])
        )
        row["ordering_violation_max"] = float(
            np.nanmax(np.maximum(ctrl_pred_on_cat[:n_uplift] - cat_pred[:n_uplift], 0.0))
            if n_uplift > 0
            else np.nan
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _numeric_summary(values: Any) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p10": np.nan, "p90": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def write_prediction_diagnostics(
    records: List[Dict[str, Any]],
    output_dir: str,
    stage: str,
    ensemble: bool = False,
    top_n: int = 25,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    diag_df = per_sample_metrics_from_records(records, ensemble=ensemble)
    per_sample_path = os.path.join(output_dir, f"{stage}_per_sample_metrics.csv")
    top_error_csv_path = os.path.join(output_dir, f"{stage}_top_error_report.csv")
    top_error_json_path = os.path.join(output_dir, f"{stage}_top_error_report.json")
    catalyst_summary_path = os.path.join(output_dir, f"{stage}_catalyst_response_summary.json")
    cap_anchor_status_path = os.path.join(output_dir, f"{stage}_cap_anchor_status_summary.json")

    diag_df.to_csv(per_sample_path, index=False)
    if diag_df.empty:
        top_df = diag_df.copy()
    else:
        sort_cols = [
            col for col in ["combined_mae", "catalyzed_mae", "catalyzed_final_abs_error", "uplift_mae"]
            if col in diag_df.columns
        ]
        top_df = diag_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(int(top_n))
    top_df.to_csv(top_error_csv_path, index=False)
    save_json(
        top_error_json_path,
        {
            "stage": stage,
            "ensemble": bool(ensemble),
            "top_n": int(top_n),
            "rows": top_df.to_dict(orient="records"),
        },
    )

    catalyst_cols = [
        "average_catalyst_dose_mg_l",
        "catalyst_use_frac",
        "active_catalyst_mean",
        "active_catalyst_peak",
        "active_catalyst_recent",
        "tau_days",
        "kappa",
        "temp_days",
        "uplift_mae",
        "uplift_final_error",
    ]
    catalyst_summary = {
        "stage": stage,
        "ensemble": bool(ensemble),
        "n_records": int(len(diag_df)),
        "columns": {
            col: _numeric_summary(diag_df[col].to_numpy(dtype=float))
            for col in catalyst_cols
            if col in diag_df.columns
        },
    }
    save_json(catalyst_summary_path, catalyst_summary)

    cap_summary = {
        "stage": stage,
        "ensemble": bool(ensemble),
        "n_records": int(len(diag_df)),
        "control_anchor_active_count": int(diag_df["control_cap_anchor_active"].sum()) if not diag_df.empty else 0,
        "catalyzed_anchor_active_count": int(diag_df["catalyzed_cap_anchor_active"].sum()) if not diag_df.empty else 0,
        "control_anchor_nonfinite_count": int((~np.isfinite(diag_df["control_cap_anchor_pct"].to_numpy(dtype=float))).sum())
        if not diag_df.empty else 0,
        "catalyzed_anchor_nonfinite_count": int((~np.isfinite(diag_df["catalyzed_cap_anchor_pct"].to_numpy(dtype=float))).sum())
        if not diag_df.empty else 0,
        "control_active_nonfinite_count": int(
            (
                diag_df["control_cap_anchor_active"].to_numpy(dtype=bool)
                & ~np.isfinite(diag_df["control_cap_anchor_pct"].to_numpy(dtype=float))
            ).sum()
        )
        if not diag_df.empty else 0,
        "catalyzed_active_nonfinite_count": int(
            (
                diag_df["catalyzed_cap_anchor_active"].to_numpy(dtype=bool)
                & ~np.isfinite(diag_df["catalyzed_cap_anchor_pct"].to_numpy(dtype=float))
            ).sum()
        )
        if not diag_df.empty else 0,
        "cap_prior_violation_count": int(diag_df["cap_prior_violation_flag"].sum()) if not diag_df.empty else 0,
    }
    save_json(cap_anchor_status_path, cap_summary)

    return {
        "per_sample_metrics_csv": per_sample_path,
        "top_error_report_csv": top_error_csv_path,
        "top_error_report_json": top_error_json_path,
        "catalyst_response_summary_json": catalyst_summary_path,
        "cap_anchor_status_summary_json": cap_anchor_status_path,
    }


# ---------------------------
# Plotting
# ---------------------------
def plot_single_record(record: Dict[str, Any], plot_path: str, title: str) -> None:
    plt.figure(figsize=(9, 5))
    member_gap_margin_pct = float(CONFIG.get("member_prediction_gap_margin_pct", 0.0))
    show_member_gap_band = bool(CONFIG.get("member_prediction_gap_band", True))
    if show_member_gap_band:
        cat_band_t, cat_band_low, cat_band_high = predictive_interval_plot_band_with_reference(
            time_days=np.asarray(record["catalyzed_t"], dtype=float),
            low_curve=np.asarray(record.get("catalyzed_pred_gap_low", record["catalyzed_pred"]), dtype=float),
            high_curve=np.asarray(record.get("catalyzed_pred_gap_high", record["catalyzed_pred"]), dtype=float),
            reference_time=np.asarray(record["catalyzed_t"], dtype=float),
            reference_curve=np.asarray(record["catalyzed_true"], dtype=float),
            margin_pct=member_gap_margin_pct,
        )
        ctrl_band_t, ctrl_band_low, ctrl_band_high = predictive_interval_plot_band_with_reference(
            time_days=np.asarray(record["control_t"], dtype=float),
            low_curve=np.asarray(record.get("control_pred_gap_low", record["control_pred"]), dtype=float),
            high_curve=np.asarray(record.get("control_pred_gap_high", record["control_pred"]), dtype=float),
            reference_time=np.asarray(record["control_t"], dtype=float),
            reference_curve=np.asarray(record["control_true"], dtype=float),
            margin_pct=member_gap_margin_pct,
        )
        plt.fill_between(
            cat_band_t,
            cat_band_low,
            cat_band_high,
            color="#ff7f0e",
            alpha=0.12,
            label="Catalyzed Member Gap",
            zorder=0,
        )
        plt.fill_between(
            ctrl_band_t,
            ctrl_band_low,
            ctrl_band_high,
            color="#1f77b4",
            alpha=0.12,
            label="Control Member Gap",
            zorder=0,
        )
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
    interval_cover_margin_pct = float(CONFIG.get("ensemble_interval_cover_margin_pct", 0.0))

    if bool(CONFIG.get("ensemble_interval_cover_true_curve", True)):
        kt_band, kp_lo_band, kp_hi_band = predictive_interval_plot_band_with_reference(
            time_days=kt,
            low_curve=kp_lo,
            high_curve=kp_hi,
            reference_time=np.asarray(record["catalyzed_t"], dtype=float),
            reference_curve=ky,
            margin_pct=interval_cover_margin_pct,
        )
        ct_band, cp_lo_band, cp_hi_band = predictive_interval_plot_band_with_reference(
            time_days=ct,
            low_curve=cp_lo,
            high_curve=cp_hi,
            reference_time=np.asarray(record["control_t"], dtype=float),
            reference_curve=cy,
            margin_pct=interval_cover_margin_pct,
        )
    else:
        kt_band, kp_lo_band, kp_hi_band = kt, kp_lo, kp_hi
        ct_band, cp_lo_band, cp_hi_band = ct, cp_lo, cp_hi

    plt.fill_between(kt_band, kp_lo_band, kp_hi_band, color="#ff7f0e", alpha=0.18, label="Catalyzed P10-P90", zorder=1)
    plt.plot(kt_band, kp_lo_band, "-", lw=0.9, color="#ff7f0e", alpha=0.35, zorder=1)
    plt.plot(kt_band, kp_hi_band, "-", lw=0.9, color="#ff7f0e", alpha=0.35, zorder=1)
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

    plt.fill_between(ct_band, cp_lo_band, cp_hi_band, color="#1f77b4", alpha=0.18, label="Control P10-P90", zorder=4)
    plt.plot(ct_band, cp_lo_band, "-", lw=0.9, color="#1f77b4", alpha=0.35, zorder=4)
    plt.plot(ct_band, cp_hi_band, "-", lw=0.9, color="#1f77b4", alpha=0.35, zorder=4)
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


def plot_ensemble_records_per_sample(
    records: List[Dict[str, Any]],
    output_dir: str,
    dpi: int = 300,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sample_groups: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}

    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            continue
        group = sample_groups.setdefault(sample_id, {"Control": {}, "Catalyzed": {}})

        control_col_id = str(record.get("control_col_id", "")).strip()
        if control_col_id and control_col_id not in group["Control"]:
            group["Control"][control_col_id] = {
                "col_id": control_col_id,
                "status": "Control",
                "true_t": np.asarray(record.get("control_t", []), dtype=float),
                "true_y": np.asarray(record.get("control_true", []), dtype=float),
                "last_actual_day": float(record.get("control_last_actual_day", np.nan)),
                "plot_t": np.asarray(record.get("control_plot_time_days", record.get("control_t", [])), dtype=float),
                "pred_mean": np.asarray(record.get("control_pred_plot_mean", record.get("control_pred_mean", [])), dtype=float),
                "pred_p10": np.asarray(record.get("control_pred_plot_p10", record.get("control_pred_p10", [])), dtype=float),
                "pred_p90": np.asarray(record.get("control_pred_plot_p90", record.get("control_pred_p90", [])), dtype=float),
                "start_day": float(record.get("control_start_day", np.nan)),
                "avg_dose_mg_l": float(record.get("control_avg_catalyst_dose_mg_l", 0.0)),
            }

        catalyzed_col_id = str(record.get("catalyzed_col_id", "")).strip()
        if catalyzed_col_id and catalyzed_col_id not in group["Catalyzed"]:
            group["Catalyzed"][catalyzed_col_id] = {
                "col_id": catalyzed_col_id,
                "status": "Catalyzed",
                "true_t": np.asarray(record.get("catalyzed_t", []), dtype=float),
                "true_y": np.asarray(record.get("catalyzed_true", []), dtype=float),
                "last_actual_day": float(record.get("catalyzed_last_actual_day", np.nan)),
                "plot_t": np.asarray(record.get("catalyzed_plot_time_days", record.get("catalyzed_t", [])), dtype=float),
                "pred_mean": np.asarray(record.get("catalyzed_pred_plot_mean", record.get("catalyzed_pred_mean", [])), dtype=float),
                "pred_p10": np.asarray(record.get("catalyzed_pred_plot_p10", record.get("catalyzed_pred_p10", [])), dtype=float),
                "pred_p90": np.asarray(record.get("catalyzed_pred_plot_p90", record.get("catalyzed_pred_p90", [])), dtype=float),
                "start_day": float(record.get("catalyzed_start_day", record.get("catalyst_addition_start_day", np.nan))),
                "avg_dose_mg_l": float(record.get("catalyzed_avg_catalyst_dose_mg_l", record.get("average_catalyst_dose_mg_l", np.nan))),
            }

    control_colors = plt.cm.Blues(np.linspace(0.45, 0.90, 12))
    catalyzed_colors = plt.cm.Oranges(np.linspace(0.45, 0.90, 12))
    target_day = float(CONFIG.get("ensemble_plot_target_day", 2500.0))
    interval_cover_margin_pct = float(CONFIG.get("ensemble_interval_cover_margin_pct", 0.0))
    cover_true_curve = bool(CONFIG.get("ensemble_interval_cover_true_curve", True))

    for sample_id, grouped_curves in sorted(sample_groups.items(), key=lambda item: item[0]):
        fig, ax = plt.subplots(figsize=(12, 7), dpi=dpi)
        control_curves = [grouped_curves["Control"][key] for key in sorted(grouped_curves["Control"])]
        catalyzed_curves = [grouped_curves["Catalyzed"][key] for key in sorted(grouped_curves["Catalyzed"])]

        for idx, curve in enumerate(control_curves):
            color = control_colors[idx % len(control_colors)]
            if cover_true_curve:
                band_t, band_p10, band_p90 = predictive_interval_plot_band_with_reference(
                    time_days=curve["plot_t"],
                    low_curve=curve["pred_p10"],
                    high_curve=curve["pred_p90"],
                    reference_time=curve["true_t"],
                    reference_curve=curve["true_y"],
                    margin_pct=interval_cover_margin_pct,
                )
            else:
                band_t, band_p10, band_p90 = curve["plot_t"], curve["pred_p10"], curve["pred_p90"]
            ax.fill_between(
                band_t,
                band_p10,
                band_p90,
                color=color,
                alpha=0.14,
                zorder=1,
            )
            ax.plot(band_t, band_p10, color=color, lw=0.8, alpha=0.30, zorder=1)
            ax.plot(band_t, band_p90, color=color, lw=0.8, alpha=0.30, zorder=1)
            ax.plot(
                curve["plot_t"],
                curve["pred_mean"],
                color=color,
                lw=2.0,
                alpha=0.95,
                label=f"Control | {curve['col_id']}",
                zorder=2,
            )
            _lad = curve["last_actual_day"]
            _tt = curve["true_t"]
            _ty = curve["true_y"]
            if np.isfinite(_lad):
                _obs_mask = _tt <= _lad + 1e-6
                _virt_mask = _tt > _lad + 1e-6
                if np.any(_obs_mask):
                    ax.scatter(_tt[_obs_mask], _ty[_obs_mask], color=color, s=28, alpha=0.55, zorder=3)
                if np.any(_virt_mask):
                    ax.scatter(_tt[_virt_mask], _ty[_virt_mask], color=color, s=18,
                               alpha=0.35, zorder=3, marker="x", linewidths=0.9)
            else:
                ax.scatter(_tt, _ty, color=color, s=28, alpha=0.55, zorder=3)
            if np.isfinite(curve["start_day"]):
                ax.axvline(curve["start_day"], color=color, lw=1.0, ls="--", alpha=0.50, zorder=0)

        for idx, curve in enumerate(catalyzed_curves):
            color = catalyzed_colors[idx % len(catalyzed_colors)]
            if cover_true_curve:
                band_t, band_p10, band_p90 = predictive_interval_plot_band_with_reference(
                    time_days=curve["plot_t"],
                    low_curve=curve["pred_p10"],
                    high_curve=curve["pred_p90"],
                    reference_time=curve["true_t"],
                    reference_curve=curve["true_y"],
                    margin_pct=interval_cover_margin_pct,
                )
            else:
                band_t, band_p10, band_p90 = curve["plot_t"], curve["pred_p10"], curve["pred_p90"]
            ax.fill_between(
                band_t,
                band_p10,
                band_p90,
                color=color,
                alpha=0.16,
                zorder=4,
            )
            ax.plot(band_t, band_p10, color=color, lw=0.8, alpha=0.32, zorder=4)
            ax.plot(band_t, band_p90, color=color, lw=0.8, alpha=0.32, zorder=4)
            ax.plot(
                curve["plot_t"],
                curve["pred_mean"],
                color=color,
                lw=2.0,
                alpha=0.95,
                label=f"Catalyzed | {curve['col_id']} | {format_average_catalyst_dose_label(curve['avg_dose_mg_l'])}",
                zorder=5,
            )
            _lad = curve["last_actual_day"]
            _tt = curve["true_t"]
            _ty = curve["true_y"]
            if np.isfinite(_lad):
                _obs_mask = _tt <= _lad + 1e-6
                _virt_mask = _tt > _lad + 1e-6
                if np.any(_obs_mask):
                    ax.scatter(_tt[_obs_mask], _ty[_obs_mask], color=color, s=28, alpha=0.55, zorder=6)
                if np.any(_virt_mask):
                    ax.scatter(_tt[_virt_mask], _ty[_virt_mask], color=color, s=18,
                               alpha=0.35, zorder=6, marker="x", linewidths=0.9)
            else:
                ax.scatter(_tt, _ty, color=color, s=28, alpha=0.55, zorder=6)
            if np.isfinite(curve["start_day"]):
                ax.axvline(curve["start_day"], color=color, lw=1.0, ls="--", alpha=0.55, zorder=0)

        # Draw a single vertical grey line at the latest observed-data boundary
        # across all curves in this sample (the transition to virtual augmentation).
        last_actual_days_all = [
            c["last_actual_day"]
            for c in control_curves + catalyzed_curves
            if np.isfinite(c["last_actual_day"])
        ]
        if last_actual_days_all:
            _boundary_day = float(np.max(last_actual_days_all))
            ax.axvline(
                _boundary_day,
                color="grey",
                lw=1.2,
                ls=":",
                alpha=0.70,
                zorder=0,
                label=f"Obs. end (~day {_boundary_day:.0f})",
            )

        ax.set_xlabel("Leach Duration (days)")
        ax.set_ylabel("Cu Recovery (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(loc="best", fontsize=8)
        apply_sample_plot_titles(
            fig,
            ax,
            sample_id=sample_id,
            col_ids=[curve["col_id"] for curve in control_curves + catalyzed_curves],
        )

        x_candidates: List[float] = [target_day]
        for curve in control_curves + catalyzed_curves:
            if curve["plot_t"].size > 0 and np.any(np.isfinite(curve["plot_t"])):
                x_candidates.append(float(np.nanmax(curve["plot_t"])))
            if curve["true_t"].size > 0 and np.any(np.isfinite(curve["true_t"])):
                x_candidates.append(float(np.nanmax(curve["true_t"])))
        x_right = max([value for value in x_candidates if np.isfinite(value)] or [target_day])
        ax.set_xlim(left=0, right=x_right * 1.05)

        plot_path = os.path.join(output_dir, f"{sample_id}.png")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
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

    ctrl_mask = np.isfinite(ctrl_true) & np.isfinite(ctrl_pred)
    cat_mask = np.isfinite(cat_true) & np.isfinite(cat_pred)
    ctrl_true_f = ctrl_true[ctrl_mask]
    ctrl_pred_f = ctrl_pred[ctrl_mask]
    cat_true_f = cat_true[cat_mask]
    cat_pred_f = cat_pred[cat_mask]

    finite_true_parts = [arr for arr in [ctrl_true_f, cat_true_f] if arr.size > 0]
    finite_pred_parts = [arr for arr in [ctrl_pred_f, cat_pred_f] if arr.size > 0]
    all_true = np.concatenate(finite_true_parts) if finite_true_parts else np.asarray([], dtype=float)
    all_pred = np.concatenate(finite_pred_parts) if finite_pred_parts else np.asarray([], dtype=float)

    plt.figure(figsize=(6, 6))
    if all_true.size == 0 or all_pred.size == 0:
        plt.text(
            0.5,
            0.5,
            "No finite true/predicted pairs available",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=11,
        )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        low = float(min(np.nanmin(all_true), np.nanmin(all_pred), 0.0))
        high = float(max(np.nanmax(all_true), np.nanmax(all_pred), 100.0))
        if np.isclose(low, high):
            high = low + 1.0
        plt.scatter(ctrl_true_f, ctrl_pred_f, s=15, alpha=0.6, color="#1f77b4", label=f"Control (n={ctrl_true_f.size})")
        plt.scatter(cat_true_f, cat_pred_f, s=15, alpha=0.6, color="#ff7f0e", label=f"Catalyzed (n={cat_true_f.size})")
        plt.plot([low, high], [low, high], "k--", lw=1.2, label="Ideal")
        plt.xlim(low, high)
        plt.ylim(low, high)
    plt.xlabel("True cu_recovery_%")
    plt.ylabel("Predicted cu_recovery_%")
    plt.title(title)
    plt.grid(alpha=0.25)
    if all_true.size > 0 and all_pred.size > 0:
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

    for ax, vals, metric_name in zip(axes, [rmse_vals, mae_vals, r2_vals], ["RMSE", "MAE", "R2"]):
        vals_arr = np.asarray(vals, dtype=float)
        plot_vals = np.where(np.isfinite(vals_arr), vals_arr, 0.0)
        bar_colors = [colors[i] if np.isfinite(vals_arr[i]) else "#d9d9d9" for i in range(len(categories))]
        ax.bar(x, plot_vals, color=bar_colors, alpha=0.85)
        ax.set_title(metric_name)
        ax.set_xticks(x, categories, rotation=15)
        ax.grid(alpha=0.25, axis="y")

        finite_vals = vals_arr[np.isfinite(vals_arr)]
        if finite_vals.size > 0:
            ymin = min(0.0, float(np.min(finite_vals)) * 1.15)
            ymax = max(0.0, float(np.max(finite_vals)) * 1.15)
            if np.isclose(ymin, ymax):
                ymax = ymin + 1.0
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(0.0, 1.0)
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes, fontsize=10)

        y_range = max(abs(ax.get_ylim()[1] - ax.get_ylim()[0]), 1.0)
        for idx, value in enumerate(vals_arr):
            label = "NaN" if not np.isfinite(value) else f"{value:.3f}"
            y_text = plot_vals[idx] + 0.02 * y_range
            ax.text(idx, y_text, label, ha="center", va="bottom", fontsize=8)

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
    print(
        "[Torch] Runtime: "
        f"device={device}, "
        f"dtype={torch_dtype_name(MODEL_TORCH_DTYPE)}, "
        f"pair_batch_size={PAIR_BATCH_SIZE}, "
        f"eval_every_n_epochs={EVAL_EVERY_N_EPOCHS}, "
        f"mps_torch_threads_cap={MPS_TORCH_THREADS_CAP}, "
        f"checkpoint_map_location={CHECKPOINT_MAP_LOCATION}"
    )

    # 1) Data loading and analysis (v11)
    print(f"[Data] Loading from: {DATA_PATH}")
    usecols = resolve_dataset_usecols(DATA_PATH)
    df = pd.read_csv(DATA_PATH, sep=",", usecols=usecols)
    print(f"[Data] Loaded {len(df)} rows × {len(df.columns)} columns | "
          f"unique project_col_id: {df[COL_ID_COL].nunique() if COL_ID_COL in df.columns else 'N/A'} | "
          f"unique project_sample_id: {df[PAIR_ID_COL].nunique() if PAIR_ID_COL in df.columns else 'N/A'}")

    # Apply all exclusion filters (auto-filter invalid rows + explicit exclusion lists)
    df, training_exclusion_summary = apply_training_pair_exclusions(df)

    missing_input_report_df, missing_input_summary = build_missing_model_input_report(df)
    missing_input_report_path = os.path.join(OUTPUTS_ROOT, "missing_model_inputs_by_sample_col.csv")
    missing_input_summary_path = os.path.join(OUTPUTS_ROOT, "missing_model_inputs_summary.json")
    missing_input_report_df.to_csv(missing_input_report_path, index=False)
    save_json(missing_input_summary_path, missing_input_summary)
    print(
        "[Data] Missing model input report: "
        f"entries={missing_input_summary['n_missing_entries']} | "
        f"sample_col_rows={missing_input_summary['n_rows_with_missing_entries']} | "
        f"path={missing_input_report_path}"
    )

    validate_required_scalar_columns(
        df,
        REQUIRED_STATIC_PREDICTOR_COLUMNS,
        context="Loaded dataset static predictors",
    )
    validate_required_scalar_columns(
        df,
        INPUT_ONLY_COLUMNS,
        context="Loaded dataset input-only predictors",
    )
    if training_exclusion_summary["applied"]:
        print(
            "[Data] Exclusion summary: "
            f"auto_excluded_recovery={training_exclusion_summary.get('auto_excluded_recovery_row_count', 0)} | "
            f"pair_ids_excluded={training_exclusion_summary.get('excluded_pair_count', 0)} | "
            f"col_ids_excluded={training_exclusion_summary.get('excluded_col_count', 0)} | "
            f"remaining_rows={training_exclusion_summary['remaining_row_count']}"
        )
    analysis_summary = analyze_dataset(df)
    analysis_summary["training_exclusions"] = training_exclusion_summary
    analysis_summary["missing_model_inputs"] = missing_input_summary
    analysis_summary["model_logic_version"] = MODEL_LOGIC_VERSION
    analysis_summary["catalyst_model"] = "explicit_catalyst_addition_with_cumulative_fallback_v12"
    analysis_summary["pls_orp_enabled"] = PLS_ORP_PROFILE_COL in df.columns
    save_json(os.path.join(OUTPUTS_ROOT, "data_analysis_summary.json"), analysis_summary)

    # 2) Biexponential prefit for bounds/limits
    # 2) Biexponential prefit per project_col_id (one row per col in new CSV format).
    # Leach caps (ctrl_cap / cat_cap) derived from mineralogy remain at the
    # project_sample_id level and are computed in build_pair_samples below.
    # The prefit parameters (fit_a1, fit_b1, fit_a2, fit_b2) are stored per
    # col_id row and used to build the augmented curve targets for both
    # training and validation.
    prefit_out_path = os.path.join(OUTPUTS_ROOT, "col_biexponential_prefit.csv")
    df_prefit = prepare_prefit_dataframe(df, prefit_out_path)

    invalid_member_reports: List[Dict[str, Any]] = []
    # 3) Build pair-level control/catalyzed samples with time-dependent arrays.
    pairs_observed = build_pair_samples(df_prefit)
    assert_pairs_respect_training_exclusions(pairs_observed, context="main pairs_observed")
    if len(pairs_observed) < 6:
        raise ValueError(f"Expected at least 6 paired samples; got {len(pairs_observed)}.")

    # 3a) Generate fitted curve plots for each project_sample_id
    fitted_curves_plot_dir = os.path.join(PLOTS_ROOT, "fitted_curves")
    target_day = float(CONFIG.get("ensemble_plot_target_day", 2500.0))
    plot_dpi = int(CONFIG.get("plot_dpi", 300))
    reset_generated_dir(fitted_curves_plot_dir)
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
    pairs_training = augment_pairs_with_virtual_data_by_col_id(
        pairs=pairs_observed,
        target_day=target_day,
        interval_days=7.0,
    )
    assert_pairs_respect_training_exclusions(pairs_training, context="main pairs_training")
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
        "use_prefit_param_bounds": bool(CONFIG.get("use_prefit_param_bounds", True)),
    }
    save_json(os.path.join(OUTPUTS_ROOT, "param_bounds.json"), param_bounds_payload)

    tmax_days = float(
        max(
            max(float(np.max(p.control.time)) for p in pairs_training),
            max(float(np.max(p.catalyzed.time)) for p in pairs_training),
        )
    )
    signal_scale_sources = _build_signal_scale_sources(pairs_training, conc_percentile=95.0)
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
    conc_candidates = [
        float(np.nanmax(p.catalyzed.catalyst_conc_col_mg_l))
        for p in pairs_training
        if p.catalyzed.catalyst_conc_col_mg_l.size > 0
        and np.any(np.isfinite(p.catalyzed.catalyst_conc_col_mg_l))
    ]
    conc_scale = float(
        np.nanpercentile(conc_candidates, 95) if len(conc_candidates) > 0 else 1.0
    )
    conc_scale = max(conc_scale, 1e-6)
    print(
        "[v11] Signal scales:\n"
        f"  cum={cum_scale:.4f} kg/t"
        f" | project_sample_id={signal_scale_sources['cum_scale']['project_sample_id']}"
        f" | project_col_id={signal_scale_sources['cum_scale']['project_col_id']}\n"
        f"  lix={lix_scale:.4f} m3/t"
        f" | project_sample_id={signal_scale_sources['lix_scale']['project_sample_id']}"
        f" | project_col_id={signal_scale_sources['lix_scale']['project_col_id']}"
        f" | status={signal_scale_sources['lix_scale']['status']}\n"
        f"  irr={irrigation_scale:.2f} L/m2/h"
        f" | project_sample_id={signal_scale_sources['irrigation_scale']['project_sample_id']}"
        f" | project_col_id={signal_scale_sources['irrigation_scale']['project_col_id']}"
        f" | status={signal_scale_sources['irrigation_scale']['status']}\n"
        f"  conc={conc_scale:.4f} mg/L"
        f" | source=p95 nearest column max={signal_scale_sources['conc_scale'].get('reference_value', np.nan):.4f}"
        f" | project_sample_id={signal_scale_sources['conc_scale']['project_sample_id']}"
        f" | project_col_id={signal_scale_sources['conc_scale']['project_col_id']}"
    )
    geo_idx = [STATIC_PREDICTOR_COLUMNS.index(c) for c in GEO_PRIORITY_COLUMNS if c in STATIC_PREDICTOR_COLUMNS]

    # 5) Repeated K-fold ensemble runs (validation + deployed members)
    val_member_plot_root = os.path.join(PLOTS_ROOT, "validation_members")
    val_member_out_root = os.path.join(OUTPUTS_ROOT, "validation_members")
    val_member_model_root = os.path.join(MODELS_ROOT, "validation_members")
    for p in [val_member_plot_root, val_member_out_root, val_member_model_root]:
        reset_generated_dir(p)

    member_record_maps_val = []
    member_metrics_rows = []
    member_histories = []
    member_split_summaries = []
    member_models = []

    # Index prebuilt pairs by sample_id so the CV loop can slice without
    # calling build_pair_samples / augment_pairs_with_virtual_data_by_col_id
    # for each member. Keep lists here: each project_sample_id can have many
    # project_col_id control/catalyzed pairings, and OOF must evaluate all of
    # them while still splitting at the sample-id level.
    pairs_observed_by_sid: Dict[str, List["PairSample"]] = {}
    pairs_training_by_sid: Dict[str, List["PairSample"]] = {}
    for pair in pairs_observed:
        pairs_observed_by_sid.setdefault(pair.sample_id, []).append(pair)
    for pair in pairs_training:
        pairs_training_by_sid.setdefault(pair.sample_id, []).append(pair)

    cv_sample_ids = sorted({p.sample_id for p in pairs_observed})
    # (F) Stratify K-fold by dominant mineralogy so every fold sees
    # representative primary / secondary / oxide / mixed ore.
    # (G) Alias related sample_ids (e.g., amcf_6in / amcf_8in) into a single
    # grouping key so they always share fold fate.
    cv_strata_labels = compute_mineralogy_stratum_labels(pairs_observed)
    cv_splits = build_repeated_group_kfold_member_splits(
        sample_ids=cv_sample_ids,
        n_splits=int(CONFIG.get("cv_n_splits", 5)),
        n_repeats=int(CONFIG.get("cv_n_repeats", 2)),
        n_split_seeds=int(CONFIG.get("cv_n_split_seeds", 1)),
        random_state=int(CONFIG.get("cv_random_state", CONFIG["seed"])),
        member_seed_base=int(CONFIG.get("cv_member_seed_base", 10000)),
        group_alias=KFOLD_GROUP_ALIAS,
        strata_labels=cv_strata_labels,
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
        split_seed_idx = int(split.get("split_seed_idx", 0))
        split_random_state = int(split.get("split_random_state", CONFIG.get("cv_random_state", CONFIG["seed"])))
        repeat_idx = int(split["repeat_idx"])
        fold_idx = int(split["fold_idx"])
        member_seed = int(split["member_seed"])
        train_indices = split["train_indices"]
        val_indices = split["val_indices"]
        train_sample_ids_seed = [cv_sample_ids[i] for i in train_indices]
        val_sample_ids_seed = [cv_sample_ids[i] for i in val_indices]

        # Select prebuilt pairs by sample_id — no DataFrame slicing or
        # build_pair_samples / augment calls per member (v10-style fast path).
        train_pairs_observed_seed = [
            pair
            for sid in train_sample_ids_seed
            if sid in pairs_observed_by_sid
            for pair in pairs_observed_by_sid[sid]
        ]
        val_pairs_observed_seed = [
            pair
            for sid in val_sample_ids_seed
            if sid in pairs_observed_by_sid
            for pair in pairs_observed_by_sid[sid]
        ]
        train_pairs_seed = [
            pair
            for sid in train_sample_ids_seed
            if sid in pairs_training_by_sid
            for pair in pairs_training_by_sid[sid]
        ]
        # Validation follows the same augmented per-column pre-fit curves as
        # training so the targets are consistent across the full CV workflow.
        val_pairs_seed = [
            pair
            for sid in val_sample_ids_seed
            if sid in pairs_training_by_sid
            for pair in pairs_training_by_sid[sid]
        ]
        val_pairs_metrics_seed = val_pairs_seed

        if len(train_pairs_seed) == 0 or len(val_pairs_seed) == 0:
            raise ValueError(
                f"CV member {member_idx} produced an empty split after sample-level division: "
                f"train_pairs={len(train_pairs_seed)}, val_pairs={len(val_pairs_seed)}."
            )

        member_tag = format_member_tag(
            member_idx=member_idx,
            split_seed_idx=split_seed_idx,
            repeat_idx=repeat_idx,
            fold_idx=fold_idx,
            member_seed=member_seed,
        )

        member_jobs.append(
            {
                "member_tag": member_tag,
                "member_idx": member_idx,
                "split_seed_idx": split_seed_idx,
                "split_random_state": split_random_state,
                "repeat_idx": repeat_idx,
                "fold_idx": fold_idx,
                "member_seed": member_seed,
                "train_pairs": train_pairs_seed,
                "val_pairs": val_pairs_seed,
                "val_pairs_metrics": val_pairs_metrics_seed,
                "tmax_days": tmax_days,
                "cum_scale": cum_scale,
                "lix_scale": lix_scale,
                "irrigation_scale": irrigation_scale,
                "conc_scale": conc_scale,
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
            clear_torch_device_cache()

    invalid_member_results = [r for r in member_results if bool(r.get("invalid_member", False))]
    invalid_member_reports.extend(invalid_member_results)
    if invalid_member_results:
        print(
            "[CV Ensemble] Invalid member diagnostics: "
            f"count={len(invalid_member_results)}"
        )

    member_results = sorted(
        [r for r in member_results if not bool(r.get("invalid_member", False))],
        key=lambda x: int(x["member_idx"]),
    )
    if len(member_results) == 0:
        raise RuntimeError("All CV members were invalid.")
    invalid_member_summary_rows = [
        {
            "member_tag": r.get("member_tag", ""),
            "member_idx": r.get("member_idx", np.nan),
            "split_seed_idx": r.get("split_seed_idx", np.nan),
            "repeat_idx": r.get("repeat_idx", np.nan),
            "fold_idx": r.get("fold_idx", np.nan),
            "seed": r.get("seed", np.nan),
            "best_eval_loss": r.get("best_eval_loss", np.nan),
            "invalid_stage": (r.get("diagnostics") or {}).get("stage", ""),
            "invalid_epoch": (r.get("diagnostics") or {}).get("epoch", np.nan),
            "invalid_batch_index": (r.get("diagnostics") or {}).get("batch_index", np.nan),
            "trigger_value": (r.get("diagnostics") or {}).get("trigger_value", np.nan),
            "suspect_col_ids": json.dumps((r.get("diagnostics") or {}).get("suspect_col_ids", [])),
        }
        for r in invalid_member_reports
    ]
    pd.DataFrame(invalid_member_summary_rows).to_csv(
        os.path.join(OUTPUTS_ROOT, "invalid_member_summary.csv"),
        index=False,
    )
    save_json(
        os.path.join(OUTPUTS_ROOT, "invalid_member_diagnostics.json"),
        {
            "n_invalid_members_detected": int(len(invalid_member_reports)),
            "dynamic_invalid_col_ids": [],
            "auto_quarantine_disabled": True,
            "invalid_members": invalid_member_reports,
        },
    )
    for result in member_results:
        member_record_maps_val.append({r["pair_id"]: r for r in result["records_val_ensemble"]})
        member_metrics_rows.append(
            {
                "member_tag": result["member_tag"],
                "member_idx": int(result["member_idx"]),
                "split_seed_idx": int(result.get("split_seed_idx", 0)),
                "split_random_state": int(result.get("split_random_state", CONFIG.get("cv_random_state", CONFIG["seed"]))),
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
                "split_seed_idx": int(result.get("split_seed_idx", 0)),
                "split_random_state": int(result.get("split_random_state", CONFIG.get("cv_random_state", CONFIG["seed"]))),
                "repeat_idx": int(result["repeat_idx"]),
                "fold_idx": int(result["fold_idx"]),
                "seed": int(result["seed"]),
                "n_train_pairs": int(result["n_train_pairs"]),
                "n_validation_pairs": int(result["n_validation_pairs"]),
                "train_pair_ids": result.get("train_pair_ids", []),
                "validation_pair_ids": result.get("validation_pair_ids", []),
                "train_sample_ids": result["train_sample_ids"],
                "validation_sample_ids": result["validation_sample_ids"],
            }
        )

        member_models.append(
            {
                "member_tag": result["member_tag"],
                "member_idx": int(result["member_idx"]),
                "split_seed_idx": int(result.get("split_seed_idx", 0)),
                "split_random_state": int(result.get("split_random_state", CONFIG.get("cv_random_state", CONFIG["seed"]))),
                "repeat_idx": int(result["repeat_idx"]),
                "fold_idx": int(result["fold_idx"]),
                "seed": int(result["seed"]),
                "model_ckpt_path": result["model_ckpt_path"],
                "imputer": result["imputer"],
                "scaler": result["scaler"],
            }
        )

    pd.DataFrame(member_metrics_rows).to_csv(
        os.path.join(val_member_out_root, "validation_member_metrics_summary.csv"),
        index=False,
    )
    split_summary = {
        "strategy": "repeated_group_kfold_by_project_sample_id",
        "n_splits": int(CONFIG.get("cv_n_splits", 5)),
        "n_repeats": int(CONFIG.get("cv_n_repeats", 2)),
        "n_split_seeds": int(CONFIG.get("cv_n_split_seeds", 1)),
        "random_state": int(CONFIG.get("cv_random_state", CONFIG["seed"])),
        "n_pairs_total": int(len(pairs_observed)),
        "n_unique_project_sample_id": int(len({p.sample_id for p in pairs_observed})),
        "n_members_total": int(len(cv_splits)),
        "n_members_valid": int(len(member_results)),
        "n_members_invalid": int(len(invalid_member_reports)),
        "dynamic_invalid_col_ids": [],
        "auto_quarantine_disabled": True,
        "invalid_members": invalid_member_reports,
        "members": member_split_summaries,
    }
    save_json(os.path.join(OUTPUTS_ROOT, "train_validation_split_summary.json"), split_summary)
    plot_validation_learning_curves(
        member_histories=member_histories,
        plot_path=os.path.join(val_member_plot_root, "validation_training_learning_curves.png"),
        title="Repeated K-Fold Training Curves Across Members",
    )

    # 6) Validation OOF aggregation and plots (strict no-leak evaluation)
    val_union_ids = sorted({pair_id for m in member_record_maps_val for pair_id in m.keys()})
    val_union_set = set(val_union_ids)
    val_pairs_for_agg = [p for p in pairs_training if p.pair_id in val_union_set]
    val_oof_metrics, val_oof_records = aggregate_ensemble_predictions(
        member_record_maps=member_record_maps_val,
        pairs=val_pairs_for_agg,
        pi_low=float(CONFIG["ensemble_pi_low"]),
        pi_high=float(CONFIG["ensemble_pi_high"]),
    )
    val_oof_out_root = os.path.join(OUTPUTS_ROOT, "validation_oof_ensemble")
    val_oof_plot_root = os.path.join(PLOTS_ROOT, "validation_oof_ensemble")
    reset_generated_dir(val_oof_out_root)
    reset_generated_dir(val_oof_plot_root)

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
    val_oof_diagnostics = write_prediction_diagnostics(
        records=val_oof_records,
        output_dir=val_oof_out_root,
        stage="validation_oof_ensemble",
        ensemble=True,
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
    plot_ensemble_records_per_sample(
        records=val_oof_records,
        output_dir=val_oof_plot_root,
        dpi=int(CONFIG["plot_dpi"]),
    )

    # 7) Deployed ensemble from CV members (captures split uncertainty)
    full_member_model_root = val_member_model_root
    full_member_out_root = os.path.join(OUTPUTS_ROOT, "deployed_cv_members")
    full_ensemble_out_root = os.path.join(OUTPUTS_ROOT, "deployed_cv_ensemble")
    full_ensemble_plot_root = os.path.join(PLOTS_ROOT, "deployed_cv_ensemble")
    for p in [full_member_out_root, full_ensemble_out_root, full_ensemble_plot_root]:
        reset_generated_dir(p)

    member_record_maps_full: List[Dict[str, Dict[str, Any]]] = []
    full_member_metrics_rows = []
    for m in member_models:
        member_tag = m["member_tag"]
        apply_static_transformers(pairs_observed, m["imputer"], m["scaler"])
        checkpoint = load_torch_checkpoint(m["model_ckpt_path"], map_location=CHECKPOINT_MAP_LOCATION)
        model = build_member_model_from_checkpoint(checkpoint)
        metrics_member, records_member = evaluate_model(
            model,
            pairs_observed,
            cum_scale,
            lix_scale,
            irrigation_scale,
            conc_scale=conc_scale,
        )
        records_to_df(records_member, ensemble=False).to_csv(
            os.path.join(full_member_out_root, f"{member_tag}_predictions.csv"),
            index=False,
        )
        save_json(
            os.path.join(full_member_out_root, f"{member_tag}_metrics.json"),
            metrics_member,
        )
        member_record_maps_full.append({r["pair_id"]: r for r in records_member})
        full_member_metrics_rows.append(
            {
                "member_tag": member_tag,
                "member_idx": int(m["member_idx"]),
                "split_seed_idx": int(m.get("split_seed_idx", 0)),
                "split_random_state": int(m.get("split_random_state", CONFIG.get("cv_random_state", CONFIG["seed"]))),
                "repeat_idx": int(m["repeat_idx"]),
                "fold_idx": int(m["fold_idx"]),
                "seed": int(m["seed"]),
                **metrics_member,
            }
        )
        del model
        clear_torch_device_cache()

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
    deployed_diagnostics = write_prediction_diagnostics(
        records=final_records,
        output_dir=full_ensemble_out_root,
        stage="deployed_cv_ensemble",
        ensemble=True,
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
    plot_ensemble_records_per_sample(
        records=final_records,
        output_dir=full_ensemble_plot_root,
        dpi=int(CONFIG["plot_dpi"]),
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
    reset_generated_dir(val_ensemble_out_root)
    reset_generated_dir(val_ensemble_plot_root)

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
    val_ensemble_diagnostics = write_prediction_diagnostics(
        records=val_ensemble_records,
        output_dir=val_ensemble_out_root,
        stage="validation_ensemble",
        ensemble=True,
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
    plot_ensemble_records_per_sample(
        records=val_ensemble_records,
        output_dir=val_ensemble_plot_root,
        dpi=int(CONFIG["plot_dpi"]),
    )

    manifest = {
        "project_root": PROJECT_ROOT,
        "data_analysis_summary": os.path.join(OUTPUTS_ROOT, "data_analysis_summary.json"),
        "missing_model_inputs_csv": missing_input_report_path,
        "missing_model_inputs_summary_json": missing_input_summary_path,
        "prefit_table": prefit_out_path,
        "param_bounds": os.path.join(OUTPUTS_ROOT, "param_bounds.json"),
        "catalyst_addition_status_csv": catalyst_stop_report_path,
        "catalyst_stopped_before_test_end_csv": catalyst_stopped_only_path,
        "catalyst_addition_status_summary": catalyst_stop_summary_path,
        "invalid_member_summary_csv": os.path.join(OUTPUTS_ROOT, "invalid_member_summary.csv"),
        "invalid_member_diagnostics_json": os.path.join(OUTPUTS_ROOT, "invalid_member_diagnostics.json"),
        "train_validation_split_summary": os.path.join(OUTPUTS_ROOT, "train_validation_split_summary.json"),
        "validation_member_outputs": val_member_out_root,
        "validation_member_models": val_member_model_root,
        "validation_member_plots": val_member_plot_root,
        "validation_learning_curves_plot": os.path.join(val_member_plot_root, "validation_training_learning_curves.png"),
        "validation_oof_ensemble_outputs": val_oof_out_root,
        "validation_oof_ensemble_plots": val_oof_plot_root,
        "validation_oof_ensemble_diagnostics": val_oof_diagnostics,
        "validation_oof_ensemble_overall_stats_csv": os.path.join(
            val_oof_out_root, "validation_oof_ensemble_overall_statistics.csv"
        ),
        "validation_oof_ensemble_overall_stats_plot": os.path.join(
            val_oof_plot_root, "validation_oof_ensemble_overall_statistics.png"
        ),
        "validation_ensemble_outputs": val_ensemble_out_root,
        "validation_ensemble_plots": val_ensemble_plot_root,
        "validation_ensemble_diagnostics": val_ensemble_diagnostics,
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
        "deployed_ensemble_diagnostics": deployed_diagnostics,
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
    print(f"Missing model input report: {missing_input_report_path}")
    print(f"Manifest: {os.path.join(OUTPUTS_ROOT, 'run_manifest.json')}")

if __name__ == "__main__":
    main()

 #%%
