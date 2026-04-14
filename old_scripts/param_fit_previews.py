#%%
import logging
import os
import ast
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Configuration (mirrors the training script where relevant)
# ---------------------------------------------------------------------------
CONFIG: Dict[str, float] = {
    "base_asymptote_cap": 80.0,
    "total_asymptote_cap": 95.0,
    "base_rate_cap": 2.1,
    "total_rate_cap": 7.0,
    "cat_effect_power": 0.7,
    "cat_rate_gain_b3": 0.3,
    "cat_rate_gain_b4": 0.1,
    "cat_additional_scale": 0.5,
    "ratio_penalty_weight": 10.0,  # strength for soft penalties in optimisation
    "control_floor_weight": 25.0,  # penalty strength to keep catalyzed curve above control
    "ratio_constraints": {
        "enable": False,
        "potentiate_weight": 0.1,
        "avoid_weight": 0.1,
        "hard_avoid_b3_over_b1": True,
        "min_a1_over_b1": 10.0,
        "target_a1_over_b1": 15.0,
        "target_a1_over_b3": 50.0,
        "target_a2_over_b1": 20.0,
        "max_b3_over_b1": 0.1,
    },
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
    """Apply amplitude/rate caps and hard ratio constraints (numpy version)."""
    p = params.copy()
    base_cap = CONFIG["base_asymptote_cap"]
    total_cap = CONFIG["total_asymptote_cap"]
    base_rate_cap = CONFIG["base_rate_cap"]
    total_rate_cap = CONFIG["total_rate_cap"]
    rcfg = CONFIG["ratio_constraints"]

    # Unpack (missing entries default to zero)
    a1, b1, a2, b2 = p[0], p[1], p[2], p[3]
    a3 = p[4] if len(p) > 4 else 0.0
    b3 = p[5] if len(p) > 5 else 0.0
    a4 = p[6] if len(p) > 6 else 0.0
    b4 = p[7] if len(p) > 7 else 0.0

    # Amplitude caps
    base_sum = a1 + a2
    if base_sum > base_cap > 0:
        scale = base_cap / base_sum
        a1 *= scale
    # keep a2 positive but ensure cap after adjusting a1 so we recompute scale
        a2 *= scale

    if len(p) > 4:
        remain = max(0.0, total_cap - (a1 + a2))
        cat_sum = a3 + a4
        if cat_sum > remain > 0:
            scale = remain / cat_sum
            a3 *= scale
            a4 *= scale

    # Rate caps
    base_rate_sum = b1 + b2
    if base_rate_sum > base_rate_cap > 0:
        scale = base_rate_cap / base_rate_sum
        b1 *= scale
        b2 *= scale

    if len(p) > 5:
        remain_rate = max(0.0, total_rate_cap - (b1 + b2))
        cat_rate_sum = b3 + b4
        if cat_rate_sum > remain_rate > 0:
            scale = remain_rate / cat_rate_sum
            b3 *= scale
            b4 *= scale

    # Hard ratio constraints
    if rcfg.get("enable", True):
        max_b3_over_b1 = rcfg.get("max_b3_over_b1", 0.1)
        min_a1_over_b1 = rcfg.get("min_a1_over_b1", 10.0)
        safe_b1 = max(b1, 1e-8)

        if len(p) > 5 and rcfg.get("hard_avoid_b3_over_b1", True):
            ratio = b3 / safe_b1
            if ratio > max_b3_over_b1:
                b3 = max_b3_over_b1 * safe_b1

        # Enforce a1/b1 >= min threshold
        ratio_a1_b1 = a1 / safe_b1
        if ratio_a1_b1 < min_a1_over_b1:
            desired_a1 = min_a1_over_b1 * safe_b1
            room = max(0.0, base_cap - a2)
            a1 = min(desired_a1, room)
            ratio_a1_b1 = a1 / safe_b1
            if ratio_a1_b1 < min_a1_over_b1:
                b1 = a1 / max(min_a1_over_b1, 1e-8)

    projected = np.array([a1, b1, a2, b2], dtype=np.float64)
    if len(p) > 4:
        projected = np.concatenate([projected, [a3, b3, a4, b4]])
    return projected


def compute_ratio_penalty_np(params: np.ndarray) -> float:
    rcfg = CONFIG["ratio_constraints"]
    if not rcfg.get("enable", True):
        return 0.0

    a1, b1, a2, b2 = params[:4]
    a3 = params[4] if len(params) > 4 else 0.0
    b3 = params[5] if len(params) > 5 else 0.0

    safe_b1 = max(b1, 1e-8)
    safe_b3 = max(abs(b3), 1e-8)

    pot = rcfg["potentiate_weight"]
    avoid = rcfg["avoid_weight"]

    penalty = 0.0
    penalty += pot * max(0.0, rcfg.get("target_a1_over_b1", 15.0) - a1 / safe_b1)
    penalty += pot * max(0.0, rcfg.get("target_a1_over_b3", 50.0) - a1 / safe_b3)
    penalty += pot * max(0.0, rcfg.get("target_a2_over_b1", 20.0) - a2 / safe_b1)
    penalty += avoid * max(0.0, (b3 / safe_b1) - rcfg.get("max_b3_over_b1", 0.1))
    penalty += avoid * max(0.0, rcfg.get("min_a1_over_b1", 10.0) - a1 / safe_b1)
    return penalty


def generate_two_phase_recovery_np(
    time: np.ndarray,
    catalyst: np.ndarray,
    transition_time: float,
    params: np.ndarray,
) -> np.ndarray:
    """
    Exponential two-phase recovery (matches training/inference code):
      Control: a1*(1-exp(-b1*t)) + a2*(1-exp(-b2*t))
      Catalyzed add-on (after transition): cat_effect * [a3*(1-exp(-b3*(t-t_trans)))
                                                         + a4*(1-exp(-b4*(t-t_trans)))]
    """
    total_cap = CONFIG["total_asymptote_cap"]
    cat_effect_power = CONFIG["cat_effect_power"]
    rate_gain_b3 = CONFIG["cat_rate_gain_b3"]
    rate_gain_b4 = CONFIG["cat_rate_gain_b4"]
    cat_additional_scale = CONFIG.get("cat_additional_scale", 1.0)

    time = np.asarray(time, dtype=np.float64)
    catalyst = np.asarray(catalyst, dtype=np.float64)
    params = params.astype(np.float64)

    catalyst_effect = catalyst / (catalyst + 1.0)
    if cat_effect_power != 1.0:
        catalyst_effect = catalyst_effect**cat_effect_power

    a1, b1, a2, b2 = params[:4]
    control = np.abs(a1) * (1.0 - np.exp(-np.abs(b1) * time)) + np.abs(a2) * (1.0 - np.exp(-np.abs(b2) * time))
    recovery = control.copy()

    if len(params) > 4:
        a3, b3, a4, b4 = params[4:]
        # Transition still uses absolute time; if you want it relative to onset, set transition_time relative.
        mask = time >= transition_time
        if np.any(mask):
            t_shift = np.clip(time - transition_time, 0.0, None)
            rate_mult3 = 1.0 + rate_gain_b3 * catalyst_effect
            rate_mult4 = 1.0 + rate_gain_b4 * catalyst_effect
            term3 = np.abs(a3) * (1.0 - np.exp(-np.abs(b3) * t_shift * rate_mult3))
            term4 = np.abs(a4) * (1.0 - np.exp(-np.abs(b4) * t_shift * rate_mult4))
            additional = cat_additional_scale * catalyst_effect * (term3 + term4)
            recovery = control + mask * additional

    # Guard against any numerical leak and cap at asymptote
    recovery = np.where(time < 0.0, 0.0, recovery)
    np.clip(recovery, 0.0, total_cap, out=recovery)
    return recovery


def determine_transition_time(time: np.ndarray, catalyst: np.ndarray) -> float:
    positive_idx = np.where(catalyst > 0.0)[0]
    if positive_idx.size == 0:
        return float(time.max())
    return float(time[positive_idx[0]])


def initial_guess(y: np.ndarray, has_catalyst: bool) -> np.ndarray:
    ymax = float(np.max(y))
    a1 = ymax * 0.6
    a2 = ymax * 0.25
    b1 = 0.02
    b2 = 0.002
    if not has_catalyst:
        return np.array([a1, b1, a2, b2], dtype=np.float64)

    extra = max(ymax - (a1 + a2), ymax * 0.15)
    a3 = extra * 0.7
    a4 = extra * 0.3
    b3 = 0.03
    b4 = 0.006
    return np.array([a1, b1, a2, b2, a3, b3, a4, b4], dtype=np.float64)


def build_bounds(n_params: int) -> Tuple[np.ndarray, np.ndarray]:
    lower = np.full(n_params, 1e-8, dtype=np.float64)
    upper = np.full(n_params, np.inf, dtype=np.float64)
    total_cap = CONFIG["total_asymptote_cap"]
    total_rate_cap = CONFIG["total_rate_cap"]

    amp_idx = [0, 2, 4, 6][: n_params // 2 * 2]
    rate_idx = [1, 3, 5, 7][: n_params // 2 * 2]

    for idx in amp_idx:
        if idx < n_params:
            upper[idx] = total_cap
    for idx in rate_idx:
        if idx < n_params:
            upper[idx] = total_rate_cap
    return lower, upper


def fit_sample(
    time: np.ndarray,
    catalyst: np.ndarray,
    target: np.ndarray,
    has_catalyst: bool,
    control_fit: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    transition = determine_transition_time(time, catalyst)
    theta0 = initial_guess(target, has_catalyst)
    lower, upper = build_bounds(len(theta0))
    ratio_weight = CONFIG["ratio_penalty_weight"]
    control_floor_weight = CONFIG.get("control_floor_weight", 0.0)

    if control_fit:
        ctrl_params = control_fit["params"]
        ctrl_transition = control_fit["transition_time"]
        ctrl_time = control_fit.get("time", time)
    else:
        ctrl_params = None
        ctrl_transition = None
        ctrl_time = None

    def residuals(theta: np.ndarray) -> np.ndarray:
        params = project_params_to_caps_np(theta)
        preds = generate_two_phase_recovery_np(time, catalyst, transition, params)
        resid = preds - target

        # Soft constraint: catalyzed curve should not drop below control curve.
        if has_catalyst and control_floor_weight > 0.0:
            if ctrl_params is not None:
                ctrl_base = generate_two_phase_recovery_np(
                    ctrl_time if ctrl_time is not None else time,
                    np.zeros_like(ctrl_time if ctrl_time is not None else time),
                    ctrl_transition,
                    ctrl_params,
                )
                if ctrl_base.shape[0] != preds.shape[0]:
                    # interpolate control to current time grid
                    ctrl_curve = np.interp(time, ctrl_time, ctrl_base)
                else:
                    ctrl_curve = ctrl_base
            else:
                # fallback: control implied by zero catalyst on current params
                ctrl_curve = generate_two_phase_recovery_np(time, np.zeros_like(time), transition, params)
            penalty_resid = np.sqrt(control_floor_weight) * np.clip(ctrl_curve - preds, 0.0, None)
            resid = np.concatenate([resid, penalty_resid])

        penalty = compute_ratio_penalty_np(params)
        if penalty > 0.0:
            resid = np.concatenate(
                [resid, np.sqrt(ratio_weight * penalty) * np.ones(1, dtype=np.float64)]
            )
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

    fitted_params = project_params_to_caps_np(result.x)
    preds = generate_two_phase_recovery_np(time, catalyst, transition, fitted_params)
    rmse = np.sqrt(mean_squared_error(target, preds))
    bias = float(np.mean(preds - target))
    r2 = float(r2_score(target, preds)) if np.std(target) > 1e-6 else float("nan")

    stats = {
        "rmse": rmse,
        "bias": bias,
        "r2": r2,
        "transition_time": transition,
        "success": bool(result.success),
        "nfev": int(result.nfev),
    }
    return fitted_params, stats


def plot_sample_fit(
    sample_id: str,
    series_label: str,
    time: np.ndarray,
    catalyst: np.ndarray,
    target: np.ndarray,
    params: np.ndarray,
    stats: Dict[str, float],
    has_catalyst: bool,
    control_fit: dict = None,
    out_dir: Path = PLOTS_DIR,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(time, target, s=25, color="black", alpha=0.6, label="Observed")

    # Prefer the separately fitted control params/time if provided, so the overlay matches the control plot exactly.
    if control_fit:
        ctrl_params = control_fit["params"]
        ctrl_transition = control_fit["transition_time"]
        ctrl_time = control_fit.get("time", time)
    else:
        ctrl_params = params.copy()
        if len(ctrl_params) > 4:
            ctrl_params[4:] = 0.0
        ctrl_transition = stats["transition_time"]
        ctrl_time = time

    control_curve = generate_two_phase_recovery_np(
        ctrl_time,
        np.zeros_like(ctrl_time),
        ctrl_transition,
        ctrl_params,
    )
    ax.plot(ctrl_time, control_curve, color="navy", linewidth=2, label="Control fit")

    if has_catalyst and len(params) > 4:
        catalyzed_curve = generate_two_phase_recovery_np(time, catalyst, stats["transition_time"], params)
        ax.plot(time, catalyzed_curve, color="orange", linewidth=2, label="Catalyzed fit", alpha=0.9)

    ax.set_xlabel("Time")
    ax.set_ylabel("Cu Recovery (%)")
    ax.set_title(
        f"Sample {sample_id} ({series_label}) | RMSE={stats['rmse']:.2f}, Bias={stats['bias']:.2f}, R²={stats['r2']:.3f}"
    )
    ax.grid(True, alpha=0.3)
    # set y limitis
    ax.set_ylim(0, CONFIG["total_asymptote_cap"] + 5)
    ax.legend()
    fig.tight_layout()

    safe_series = series_label.replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"{sample_id}_{safe_series}_fit.png"
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
    # -----------------------------------------------------------------------
    # Determine which time columns are actually present
    # -----------------------------------------------------------------------
    available_time_cols = select_time_columns(df)
    if not available_time_cols:
        raise KeyError(f"None of the expected time columns {TIME_COL_PREFERENCE} found in data.")

    logging.info("Found %d usable time column(s): %s", len(available_time_cols), available_time_cols)

    # -----------------------------------------------------------------------
    # Main fitting loop – one pass per time column
    # -----------------------------------------------------------------------
    for time_col in available_time_cols:
        # ---- create a dedicated folder for this time column -----------------
        time_dir = PLOTS_DIR / time_col
        time_dir.mkdir(parents=True, exist_ok=True)
        time_subdirs: Dict[str, Path] = {}
        time_subdirs[time_col] = time_dir

        logging.info("=== Processing time column: %s ===", time_col)

        # ---- parse array-valued columns (once per time column) -------------
        for col in [time_col, TARGET_COL, CATALYST_COL]:
            df[col] = df[col].apply(parse_array_field)

        # ---- per-sample fitting --------------------------------------------
        summary_rows: List[Dict[str, float]] = []
        control_fits: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

        for sample_id, grp in df.groupby(SAMPLE_ID_COL):
            for _, row_data in grp.iterrows():
                series_label = row_data.get("project_col_id", "series")

                time   = parse_array_field(row_data[time_col])
                target = parse_array_field(row_data[TARGET_COL])
                catalyst = parse_array_field(row_data[CATALYST_COL])

                # ---- sanity checks (unchanged) --------------------------------
                if time.size == 0 or target.size == 0:
                    logging.warning("Skipping %s (%s): empty time/target array.", sample_id, series_label)
                    continue
                if time.size != target.size:
                    logging.warning("Skipping %s (%s): time/target length mismatch (%d vs %d).",
                                    sample_id, series_label, time.size, target.size)
                    continue
                if catalyst.size not in (1, time.size):
                    logging.warning("Skipping %s (%s): catalyst length %d incompatible with time length %d.",
                                    sample_id, series_label, catalyst.size, time.size)
                    continue
                if catalyst.size == 1 and time.size > 1:
                    catalyst = np.full_like(time, catalyst.item(), dtype=np.float64)

                has_catalyst = bool(np.any(catalyst > 1e-9))
                if time.size < 4:
                    logging.warning("Skipping %s (%s): insufficient points (%d).", sample_id, series_label, time.size)
                    continue

                key = (sample_id, time_col)
                control_overlay = control_fits.get(key) if has_catalyst else None

                # ---- fit -------------------------------------------------------
                params, stats = fit_sample(time, catalyst, target, has_catalyst, control_fit=control_overlay)

                if not has_catalyst:
                    control_fits[key] = {
                        "params": params,
                        "transition_time": stats["transition_time"],
                        "time": time,
                    }

                # ---- plot (save inside the time-column sub-folder) -------------
                plot_sample_fit(
                    sample_id, series_label,
                    time, catalyst, target,
                    params, stats, has_catalyst,
                    control_fit=control_overlay,
                    out_dir=time_dir,               # <-- NEW argument
                )

                # ---- collect summary -------------------------------------------
                summary_row = {
                    "project_sample_id": sample_id,
                    "project_col_id": series_label,
                    "time_column": time_col,        # <-- remember which column we used
                    "has_catalyst": has_catalyst,
                    "n_points": int(time.size),
                    "rmse": stats["rmse"],
                    "bias": stats["bias"],
                    "r2": stats["r2"],
                    "transition_time": stats["transition_time"],
                    "fit_success": stats["success"],
                    "nfev": stats["nfev"],
                }
                param_labels = ["a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4"]
                for idx, label in enumerate(param_labels):
                    summary_row[label] = params[idx] if idx < len(params) else np.nan
                summary_rows.append(summary_row)

                logging.info(
                    "Sample %s (%s) fitted (time=%s): RMSE=%.3f, R²=%.3f, catalyst=%s",
                    sample_id, series_label, time_col,
                    stats["rmse"], stats["r2"], "Yes" if has_catalyst else "No",
                )

        # ---- save CSV summary for this time column -----------------------------
        summary_df = pd.DataFrame(summary_rows)
        summary_path = time_dir / f"param_fit_summary_{time_col}.csv"
        summary_df.to_csv(summary_path, index=False)
        logging.info("Saved parameter summary for %s to %s", time_col, summary_path)


if __name__ == "__main__":
    main()
# %%
