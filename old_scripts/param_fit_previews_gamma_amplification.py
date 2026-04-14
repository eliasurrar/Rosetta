#%%
import logging
import os
import ast
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, root_mean_squared_error

# ---------------------------------------------------------------------------
# Configuration (mirrors the training script where relevant)
# ---------------------------------------------------------------------------
CONFIG: Dict[str, float] = {
    "base_asymptote_cap": 80.0,   # cap for a1 + a2
    "total_asymptote_cap": 95.0,  # final clamp
    "base_rate_cap": 2.1,         # cap for b1 + b2
    "total_rate_cap": 7.0,        # keep as global upper bound for any rate if needed
    # Gamma amplification settings
    "gamma_min": 1e-3,            # enforce strictly positive gammas
    "gamma_max": 10.0,             # cap for each gamma_* ≥ 0
    "gamma_zero_penalty": 0.01,   # push gammas→0 on control rows during fitting
    "cat_effect_power": 1.0,      # not used in gamma mode, kept for consistency
    "param_min": 1e-3,  # minimum for a1, b1, a2, b2
}

TIME_COL_PREFERENCE: Tuple[str, ...] = ("leach_duration_days", "cumulative_lixiviant_m3_t")
TARGET_COL: str = "cu_recovery_%"
CATALYST_COL: str = "cumulative_catalyst_addition_kg_t"
SAMPLE_ID_COL: str = "project_sample_id"

BASE_DIR = Path("/Users/administration/Library/CloudStorage/OneDrive-JettiResources/Rosetta/NN_PyTorch")
DATA_PATH = BASE_DIR / "plots" / "processed_data_unscaled.csv"
PLOTS_DIR = BASE_DIR / "plots" / "param_fit_previews"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def parse_array_field(value, dtype=np.float64) -> np.ndarray:
    """Convert strings/lists/scalars into 1D numpy arrays."""
    if isinstance(value, np.ndarray):
        return value.astype(dtype)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=dtype)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith('[') and text.endswith(']'):
            inner = text[1:-1].strip()
            if inner:
                try:
                    parsed = ast.literal_eval(text)
                    return np.asarray(parsed, dtype=dtype)
                except (ValueError, SyntaxError):
                    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", inner)
                    if numbers:
                        return np.asarray([float(n) for n in numbers], dtype=dtype)
        try:
            return np.asarray([float(text)], dtype=dtype)
        except ValueError:
            logging.warning("Unable to parse array from value: %s", value)
            return np.asarray([], dtype=dtype)
    try:
        return np.asarray([value], dtype=dtype)
    except Exception:
        logging.warning("Unexpected value type %s; returning empty array.", type(value))
        return np.asarray([], dtype=dtype)

def select_time_columns(df: pd.DataFrame) -> List[str]:
    """Return the subset of TIME_COL_PREFERENCE that really exist in the DataFrame."""
    return [col for col in TIME_COL_PREFERENCE if col in df.columns]

def project_params_to_caps_np(params: np.ndarray) -> np.ndarray:
    """Caps: a1+a2 <= base_cap, b1+b2 <= base_rate_cap, gammas in [gamma_min, gamma_max]."""
    p = params.astype(np.float64).copy()
    base_cap = CONFIG["base_asymptote_cap"]
    base_rate_cap = CONFIG["base_rate_cap"]
    gamma_max = CONFIG["gamma_max"]
    gamma_min = CONFIG["gamma_min"]

    # Unpack 8 params: a1,b1,a2,b2,gA1,gB1,gA2,gB2
    a1, b1, a2, b2 = p[:4]
    gammas = p[4:8] if p.size >= 8 else np.zeros(4, dtype=np.float64)

    # Amplitude cap
    base_sum = a1 + a2
    if base_sum > base_cap > 0:
        scale = base_cap / max(base_sum, 1e-8)
        a1 *= scale
        a2 *= scale

    # Rate cap
    rate_sum = b1 + b2
    if rate_sum > base_rate_cap > 0:
        scale_r = base_rate_cap / max(rate_sum, 1e-8)
        b1 *= scale_r
        b2 *= scale_r

    # Clamp gammas to [gamma_min, gamma_max]
    gammas = np.clip(gammas, gamma_min, gamma_max)

    return np.array([a1, b1, a2, b2, *gammas], dtype=np.float64)


def generate_two_phase_recovery_np(
    time: np.ndarray,
    catalyst: np.ndarray,
    transition_time: float,
    params: np.ndarray,
) -> np.ndarray:
    """
    Two-term control + post-TT amplification using per-parameter gammas.
      control(t) = a1*(1 - exp(-b1*t)) + a2*(1 - exp(-b2*t))
      for t >= TT:
         amp_a1 = 1 + gamma_a1 ; rate_b1 = 1 + gamma_b1
         amp_a2 = 1 + gamma_a2 ; rate_b2 = 1 + gamma_b2
         recovery(t) = control(TT) +
           a1*amp_a1*(1 - exp(-(b1*rate_b1)*(t-TT))) +
           a2*amp_a2*(1 - exp(-(b2*rate_b2)*(t-TT)))
    Ensures monotone increments after TT (gammas >= gamma_min).
    """
    total_cap = CONFIG["total_asymptote_cap"]
    gamma_min = CONFIG["gamma_min"]

    time = np.asarray(time, dtype=np.float64)
    catalyst = np.asarray(catalyst, dtype=np.float64)
    params = params.astype(np.float64)

    a1, b1, a2, b2 = params[:4]
    gA1, gB1, gA2, gB2 = (params[4:8] if params.size >= 8 else np.zeros(4, dtype=np.float64))

    # Control curve
    control = a1 * (1.0 - np.exp(-b1 * time)) + a2 * (1.0 - np.exp(-b2 * time))
    recovery = control.copy()

    has_catalyst = np.any(catalyst > 0.0)
    if has_catalyst:
        mask = time >= transition_time
        if np.any(mask):
            t_shift = np.clip(time - transition_time, 0.0, None)

            amp_a1 = 1.0 + max(gA1, gamma_min)
            rate_b1 = 1.0 + max(gB1, gamma_min)
            amp_a2 = 1.0 + max(gA2, gamma_min)
            rate_b2 = 1.0 + max(gB2, gamma_min)

            exp1_post = np.exp(-(b1 * rate_b1) * t_shift)
            exp2_post = np.exp(-(b2 * rate_b2) * t_shift)
            exp1_post = np.clip(exp1_post, 1e-8, 1.0)
            exp2_post = np.clip(exp2_post, 1e-8, 1.0)

            incr = (a1 * amp_a1) * (1.0 - exp1_post) + (a2 * amp_a2) * (1.0 - exp2_post)
            control_tt = a1 * (1.0 - np.exp(-b1 * transition_time)) + a2 * (1.0 - np.exp(-b2 * transition_time))
            amplified = control_tt + incr

            recovery = np.where(mask, amplified, control)

    np.clip(recovery, 0.0, total_cap, out=recovery)
    return recovery


def determine_transition_time(time: np.ndarray, catalyst: np.ndarray) -> float:
    positive_idx = np.where(catalyst > 0.0)[0]
    if positive_idx.size == 0:
        return float(time.max())
    return float(time[positive_idx[0]])


def initial_guess(y: np.ndarray,
                  has_catalyst: bool,
                  fixed_control_params: np.ndarray | None = None) -> np.ndarray:
    """
    Initial guess:
      - Control: full 8 params (a1,b1,a2,b2,gammas=min)
      - Catalyzed with fixed control params: only 4 gammas
      - Catalyzed without control: full 8 params
    """
    ymax = float(np.max(y))
    a1 = max(ymax * 0.60, CONFIG.get("param_min", 1e-3))
    a2 = max(ymax * 0.25, CONFIG.get("param_min", 1e-3))
    b1 = max(0.02, CONFIG.get("param_min", 1e-3))
    b2 = max(0.002, CONFIG.get("param_min", 1e-3))
    gmin = float(CONFIG.get("gamma_min", 1e-3))

    if not has_catalyst:
        return np.array([a1, b1, a2, b2, gmin, gmin, gmin, gmin], dtype=np.float64)

    if fixed_control_params is not None:
        # Only gammas to optimize
        gA1 = min(0.2, gmin)
        gB1 = min(0.2, gmin)
        gA2 = min(0.1, gmin)
        gB2 = min(0.1, gmin)
        return np.array([gA1, gB1, gA2, gB2], dtype=np.float64)

    # Full catalyzed fit
    gA1 = min(0.2, gmin)
    gB1 = min(0.2, gmin)
    gA2 = min(0.1, gmin)
    gB2 = min(0.1, gmin)
    return np.array([a1, b1, a2, b2, gA1, gB1, gA2, gB2], dtype=np.float64)

def build_bounds(n_params: int, gammas_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bounds:
      - Full 8 params: a*,b* upper caps; gammas in [gamma_min,gamma_max]
      - Gammas only (length 4): gammas bounds only
    """
    gamma_max = CONFIG["gamma_max"]
    gamma_min = CONFIG["gamma_min"]
    if gammas_only:
        lower = np.full(n_params, gamma_min, dtype=np.float64)
        upper = np.full(n_params, gamma_max, dtype=np.float64)
        return lower, upper

    lower = np.full(n_params, CONFIG.get("param_min", 1e-4), dtype=np.float64)
    upper = np.full(n_params, np.inf, dtype=np.float64)
    total_cap = CONFIG["total_asymptote_cap"]
    total_rate_cap = CONFIG["total_rate_cap"]

    amp_idx = [0, 2]
    rate_idx = [1, 3]
    gamma_idx = [4, 5, 6, 7]

    for idx in amp_idx:
        if idx < n_params:
            upper[idx] = total_cap
    for idx in rate_idx:
        if idx < n_params:
            upper[idx] = total_rate_cap
    for idx in gamma_idx:
        if idx < n_params:
            lower[idx] = gamma_min
            upper[idx] = gamma_max
    return lower, upper

def assemble_full_params(control_params: np.ndarray,
                         gamma_vec: np.ndarray) -> np.ndarray:
    """
    Combine fixed control base params (first 4) with optimized gammas (4).
    control_params: length >=4 (a1,b1,a2,b2,...)
    gamma_vec: length 4 (gA1,gB1,gA2,gB2)
    Returns full 8-length param vector.
    """
    base = control_params[:4].astype(np.float64)
    gammas = gamma_vec.astype(np.float64)
    return np.concatenate([base, gammas], dtype=np.float64)


def fit_sample(time: np.ndarray,
               catalyst: np.ndarray,
               target: np.ndarray,
               has_catalyst: bool,
               fixed_control_params: np.ndarray | None = None
               ) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fit routine:
      - Control: fit all 8 params.
      - Catalyzed with fixed_control_params: fit only 4 gammas (a1..b2 fixed).
      - Catalyzed without control: fit all 8 params.
    """
    transition = determine_transition_time(time, catalyst)
    gammas_only = has_catalyst and fixed_control_params is not None

    theta0 = initial_guess(target, has_catalyst, fixed_control_params if has_catalyst else None)
    lower, upper = build_bounds(len(theta0), gammas_only=gammas_only)
    theta0 = np.clip(theta0, lower, upper)

    gamma_zero_w = float(CONFIG.get("gamma_zero_penalty", 0.0))

    def residuals(theta: np.ndarray) -> np.ndarray:
        if gammas_only:
            full_params = assemble_full_params(fixed_control_params, theta)
        else:
            full_params = theta
        params_proj = project_params_to_caps_np(full_params)
        preds = generate_two_phase_recovery_np(time, catalyst, transition, params_proj)
        resid = preds - target
        if (not has_catalyst) and gamma_zero_w > 0.0 and params_proj.size >= 8:
            gamma_pen = gamma_zero_w * np.mean(np.abs(params_proj[4:8]))
            resid = np.concatenate([resid, np.array([np.sqrt(gamma_pen)], dtype=np.float64)])
        return resid

    result = least_squares(
        residuals,
        x0=theta0,
        bounds=(lower, upper),
        max_nfev=5000,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        verbose=0,
    )

    if gammas_only:
        fitted_full = assemble_full_params(fixed_control_params, result.x)
    else:
        fitted_full = result.x

    fitted_params = project_params_to_caps_np(fitted_full)
    preds = generate_two_phase_recovery_np(time, catalyst, transition, fitted_params)
    rmse = root_mean_squared_error(target, preds)
    bias = float(np.mean(preds - target))
    r2 = float(r2_score(target, preds)) if np.std(target) > 1e-6 else float("nan")

    stats = {
        "rmse": rmse,
        "bias": bias,
        "r2": r2,
        "transition_time": transition,
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "gammas_only": bool(gammas_only),
    }
    return fitted_params, stats


def aggregate_arrays(arr_list: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(arr_list) if arr_list else np.array([], dtype=np.float64)

def plot_combined_sample(
    sample_id: str,
    time_control: np.ndarray,
    target_control: np.ndarray,
    catalyst_control: np.ndarray,
    params_control: np.ndarray,
    time_catal: np.ndarray,
    target_catal: np.ndarray,
    catalyst_catal: np.ndarray,
    params_catal: np.ndarray,
    stats_control: Dict[str, float],
    stats_catal: Dict[str, float],
    out_dir: Path,
    time_col: str,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter points
    if time_catal.size:
        ax.scatter(time_catal, target_catal, s=30, color="darkorange", alpha=0.65, label="Catalyzed observed")
    if time_control.size:
        ax.scatter(time_control, target_control, s=30, color="black", alpha=0.6, label="Control observed")


    # Control curve (gammas zeroed)
    if time_control.size:
        ctrl_params = params_control.copy()
        if ctrl_params.size >= 8:
            ctrl_params[4:8] = 0.0
        control_curve = generate_two_phase_recovery_np(
            time_control, np.zeros_like(catalyst_control), stats_control["transition_time"], ctrl_params
        )
        ax.plot(time_control, control_curve, color="navy", linewidth=2, label="Control fit")

    # Catalyzed curve
    if time_catal.size:
        catal_curve = generate_two_phase_recovery_np(
            time_catal, catalyst_catal, stats_catal["transition_time"], params_catal
        )
        ax.plot(time_catal, catal_curve, color="orangered", linewidth=2, label="Catalyzed fit")

    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel("Cu Recovery (%)")
    ax.set_title(
        f"{sample_id} | Ctrl RMSE={stats_control.get('rmse', np.nan):.2f} "
        f"| Cat RMSE={stats_catal.get('rmse', np.nan):.2f}"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"{sample_id}_combined_fit_{time_col}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Loading processed data from %s", DATA_PATH)
    
    df = pd.read_csv(DATA_PATH)
    if SAMPLE_ID_COL not in df.columns:
        fallback_cols = ["project_sample_id_reactormatch", "project_col_id"]
        for alt in fallback_cols:
            if alt in df.columns:
                df = df.rename(columns={alt: SAMPLE_ID_COL})
                break
    if SAMPLE_ID_COL not in df.columns:
        raise KeyError(f"Required column '{SAMPLE_ID_COL}' not found in {DATA_PATH}")

    available_time_cols = select_time_columns(df)
    if not available_time_cols:
        raise KeyError(f"None of the expected time columns {TIME_COL_PREFERENCE} found in data.")
    logging.info("Found %d usable time column(s): %s", len(available_time_cols), available_time_cols)

    for time_col in available_time_cols:
        time_dir = PLOTS_DIR / time_col
        time_dir.mkdir(parents=True, exist_ok=True)
        logging.info("=== Processing time column: %s ===", time_col)

        # Parse array fields for this time column
        for col in [time_col, TARGET_COL, CATALYST_COL]:
            df[col] = df[col].apply(parse_array_field)

        summary_rows: List[Dict[str, float]] = []

        for sample_id, grp in df.groupby(SAMPLE_ID_COL):
            control_times, control_targets, control_cats = [], [], []
            catal_times, catal_targets, catal_cats = [], [], []

            # Collect all rows for this sample
            for _, row_data in grp.iterrows():
                t_arr = parse_array_field(row_data[time_col])
                y_arr = parse_array_field(row_data[TARGET_COL])
                c_arr = parse_array_field(row_data[CATALYST_COL])

                if t_arr.size == 0 or y_arr.size == 0:
                    continue
                if t_arr.size != y_arr.size:
                    continue
                if c_arr.size not in (1, t_arr.size):
                    continue
                if c_arr.size == 1 and t_arr.size > 1:
                    c_arr = np.full_like(t_arr, c_arr.item(), dtype=np.float64)

                is_catalyzed = bool(np.any(c_arr > 1e-9))
                if is_catalyzed:
                    catal_times.append(t_arr); catal_targets.append(y_arr); catal_cats.append(c_arr)
                else:
                    control_times.append(t_arr); control_targets.append(y_arr); control_cats.append(c_arr)

            # Aggregate
            agg_time_ctrl = aggregate_arrays(control_times)
            agg_target_ctrl = aggregate_arrays(control_targets)
            agg_cat_ctrl = aggregate_arrays(control_cats)

            agg_time_cat = aggregate_arrays(catal_times)
            agg_target_cat = aggregate_arrays(catal_targets)
            agg_cat_cat = aggregate_arrays(catal_cats)

            # Skip if no data
            if agg_time_ctrl.size == 0 and agg_time_cat.size == 0:
                continue

            # Fit control (if present)
            if agg_time_ctrl.size >= 4:
                params_ctrl, stats_ctrl = fit_sample(agg_time_ctrl, agg_cat_ctrl, agg_target_ctrl, has_catalyst=False)
            else:
                params_ctrl = np.array([np.nan]*8); stats_ctrl = {"rmse": np.nan, "bias": np.nan, "r2": np.nan,
                                                                "transition_time": np.nan, "success": False,
                                                                "nfev": 0, "gammas_only": False}

            # Fit catalyzed (if present). If control exists, fix a1..b2.
            if agg_time_cat.size >= 4:
                if np.all(np.isfinite(params_ctrl[:4])):  # have valid control base params
                    params_cat, stats_cat = fit_sample(agg_time_cat, agg_cat_cat, agg_target_cat,
                                                    has_catalyst=True,
                                                    fixed_control_params=params_ctrl)
                else:
                    params_cat, stats_cat = fit_sample(agg_time_cat, agg_cat_cat, agg_target_cat,
                                                    has_catalyst=True,
                                                    fixed_control_params=None)
            else:
                params_cat = np.array([np.nan]*8); stats_cat = {"rmse": np.nan, "bias": np.nan, "r2": np.nan,
                                                                "transition_time": np.nan, "success": False,
                                                                "nfev": 0, "gammas_only": False}
            # Plot combined
            plot_combined_sample(
                sample_id,
                agg_time_ctrl, agg_target_ctrl, agg_cat_ctrl, params_ctrl,
                agg_time_cat, agg_target_cat, agg_cat_cat, params_cat,
                stats_ctrl, stats_cat,
                time_dir, time_col
            )

            # Summary row (one per sample)
            row = {
                "project_sample_id": sample_id,
                "time_column": time_col,
                "has_control": bool(agg_time_ctrl.size),
                "has_catalyzed": bool(agg_time_cat.size),
                "n_points_control": int(agg_time_ctrl.size),
                "n_points_catalyzed": int(agg_time_cat.size),
                "rmse_control": stats_ctrl["rmse"],
                "bias_control": stats_ctrl["bias"],
                "r2_control": stats_ctrl["r2"],
                "rmse_catalyzed": stats_cat["rmse"],
                "bias_catalyzed": stats_cat["bias"],
                "r2_catalyzed": stats_cat["r2"],
            }
            labels = ["a1","b1","a2","b2","gamma_a1","gamma_b1","gamma_a2","gamma_b2"]
            for i,lbl in enumerate(labels):
                row[f"{lbl}_control"] = params_ctrl[i] if params_ctrl.size == 8 else np.nan
                row[f"{lbl}_catalyzed"] = params_cat[i] if params_cat.size == 8 else np.nan
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = time_dir / f"param_fit_summary_combined_{time_col}.csv"
        summary_df.to_csv(summary_path, index=False)
        logging.info("Saved combined summary for %s to %s", time_col, summary_path)


if __name__ == "__main__":
    main()
# %%
