
import argparse
import glob
import importlib.util
import io
import inspect
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from dash import Dash, Input, Output, State, dcc, html


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_SCRIPT_PATH = os.path.join(THIS_DIR, "NN_ExpEq_columns_only.py")


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("nn_expeq_columns_only_base", BASE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base model script: {BASE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()

ACID_SOLUBLE_COL = "acid_soluble_%"
CYANIDE_SOLUBLE_COL = "cyanide_soluble_%"
RESIDUAL_CPY_COL = "residual_cpy_%"
GROUPED_MODEL_COLUMNS = [c for c in BASE.STATIC_PREDICTOR_COLUMNS if c.startswith("grouped_")]
GROUPED_GANGUE_INSERT_AFTER = GROUPED_MODEL_COLUMNS[-1] if len(GROUPED_MODEL_COLUMNS) > 0 else None
INTERACTIVE_STATIC_COLUMNS = [c for c in BASE.STATIC_PREDICTOR_COLUMNS if c != CYANIDE_SOLUBLE_COL]
TOP_CONTROL_STATIC_COLUMNS = [
    "column_height_m",
    "column_inner_diameter_m",
    "material_size_p80_in",
]

READONLY_BLOCK_STYLE = {
    "marginBottom": "0",
    "padding": "2px 0 0 0",
    "minWidth": "0",
}

READONLY_VALUE_STYLE = {
    "fontSize": "12px",
    "fontWeight": 600,
    "color": "#55606d",
}

READONLY_NOTE_STYLE = {
    "marginTop": "4px",
    "fontSize": "10px",
    "color": "#66707a",
    "lineHeight": "1.25",
}

READONLY_BADGE_STYLE = {
    "fontSize": "10px",
    "fontWeight": 700,
    "color": "#6b7280",
    "background": "#eceff3",
    "border": "1px solid #d6dbe3",
    "borderRadius": "999px",
    "padding": "2px 7px",
    "letterSpacing": "0.02em",
    "textTransform": "uppercase",
}

READONLY_TRACK_COLOR = "#cfd5dd"
READONLY_FILL_COLOR = "#8f98a3"
READONLY_ALERT_COLOR = "#d62828"

PNG_EXPORT_DPI = 300
PNG_EXPORT_WIDTH_IN = 12.0
PNG_EXPORT_HEIGHT_IN = 8.0
PNG_EXPORT_SCALE = 4
PNG_EXPORT_FINAL_WIDTH_PX = int(PNG_EXPORT_WIDTH_IN * PNG_EXPORT_DPI)
PNG_EXPORT_FINAL_HEIGHT_PX = int(PNG_EXPORT_HEIGHT_IN * PNG_EXPORT_DPI)
PNG_EXPORT_WIDTH_PX = int(PNG_EXPORT_FINAL_WIDTH_PX / PNG_EXPORT_SCALE)
PNG_EXPORT_HEIGHT_PX = int(PNG_EXPORT_FINAL_HEIGHT_PX / PNG_EXPORT_SCALE)


def candidate_project_roots() -> List[str]:
    roots = []
    for attr in ["PROJECT_ROOT", "DEFAULT_PROJECT_ROOT", "LOCAL_PROJECT_ROOT"]:
        value = getattr(BASE, attr, None)
        if isinstance(value, str) and value not in roots:
            roots.append(value)
    sibling_root = os.path.join(THIS_DIR, "NN_Pytorch_ExpEq_columns_only")
    if sibling_root not in roots:
        roots.append(sibling_root)
    return roots


def load_run_manifest(project_root: str) -> Dict[str, Any]:
    manifest_path = os.path.join(project_root, "outputs", "run_manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def candidate_model_dirs(project_root: str) -> List[str]:
    manifest = load_run_manifest(project_root)
    candidates: List[str] = []
    for key in ["deployed_member_models", "validation_member_models"]:
        value = manifest.get(key)
        if isinstance(value, str) and value not in candidates:
            candidates.append(value)
    for rel in ["models/validation_members", "models/full_data_members"]:
        value = os.path.join(project_root, rel)
        if value not in candidates:
            candidates.append(value)
    return candidates


def resolve_project_root(requested_root: str | None = None) -> str:
    candidates = [requested_root] if requested_root else candidate_project_roots()
    for root in candidates:
        if not root:
            continue
        for model_dir in candidate_model_dirs(root):
            if glob.glob(os.path.join(model_dir, "*.pt")):
                return root
    searched = "\n".join(candidates)
    raise FileNotFoundError(
        "Could not find saved member checkpoints. "
        "Run NN_ExpEq_columns_only.py first so it writes the checkpoint paths listed in outputs/run_manifest.json.\n"
        f"Searched:\n{searched}"
    )


def resolve_member_model_dir(project_root: str) -> str:
    searched = candidate_model_dirs(project_root)
    for model_dir in searched:
        if glob.glob(os.path.join(model_dir, "*.pt")):
            return model_dir
    searched_text = "\n".join(searched)
    raise FileNotFoundError(
        "Could not find saved member checkpoints for the interactive app. "
        "Run NN_ExpEq_columns_only.py first and then use the checkpoint directory recorded in outputs/run_manifest.json.\n"
        f"Searched:\n{searched_text}"
    )


def nice_step(min_value: float, max_value: float) -> float:
    span = float(max_value - min_value)
    if not np.isfinite(span) or span <= 0:
        return 1.0
    raw = span / 200.0
    exponent = np.floor(np.log10(raw))
    base = raw / (10.0 ** exponent)
    if base <= 1.0:
        nice = 1.0
    elif base <= 2.0:
        nice = 2.0
    elif base <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * (10.0 ** exponent))


def load_training_dataframe() -> pd.DataFrame:
    return pd.read_csv(BASE.DATA_PATH)


def value_or_default(value: float | None, default: float) -> float:
    return float(default if value is None or not np.isfinite(value) else value)


def slider_position_pct(value: float, min_value: float = 0.0, max_value: float = 100.0) -> float:
    span = max(float(max_value) - float(min_value), 1e-9)
    return float(np.clip((float(value) - float(min_value)) / span * 100.0, 0.0, 100.0))


def readonly_fill_style(position_pct: float, color: str) -> Dict[str, Any]:
    return {
        "position": "absolute",
        "top": "50%",
        "left": "0",
        "height": "4px",
        "width": f"{float(np.clip(position_pct, 0.0, 100.0)):.2f}%",
        "transform": "translateY(-50%)",
        "background": color,
        "borderRadius": "999px",
        "transition": "width 0.15s ease, background 0.15s ease",
        "pointerEvents": "none",
    }


def readonly_thumb_style(position_pct: float, color: str) -> Dict[str, Any]:
    return {
        "position": "absolute",
        "top": "50%",
        "left": f"{float(np.clip(position_pct, 0.0, 100.0)):.2f}%",
        "width": "14px",
        "height": "14px",
        "transform": "translate(-50%, -50%)",
        "background": color,
        "border": "2px solid #ffffff",
        "borderRadius": "50%",
        "boxShadow": "0 0 0 1px rgba(0,0,0,0.06)",
        "pointerEvents": "none",
    }


def chemistry_state(
    acid_value: float | None,
    residual_value: float | None,
    static_specs: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    acid_spec = static_specs[ACID_SOLUBLE_COL]
    residual_spec = static_specs[RESIDUAL_CPY_COL]

    acid = float(np.clip(value_or_default(acid_value, acid_spec["default"]), acid_spec["min"], acid_spec["max"]))
    residual = float(
        np.clip(value_or_default(residual_value, residual_spec["default"]), residual_spec["min"], residual_spec["max"])
    )

    if acid + residual > 100.0:
        residual = max(float(residual_spec["min"]), 100.0 - acid)
    if acid + residual > 100.0:
        acid = max(float(acid_spec["min"]), 100.0 - residual)

    cyanide = max(0.0, 100.0 - acid - residual)
    acid_max = max(float(acid_spec["min"]), min(float(acid_spec["max"]), 100.0 - residual))
    residual_max = max(float(residual_spec["min"]), min(float(residual_spec["max"]), 100.0 - acid))
    return {
        "acid": acid,
        "residual": residual,
        "cyanide": cyanide,
        "acid_max": acid_max,
        "residual_max": residual_max,
    }


def cyanide_visual_state(cyanide: float) -> Dict[str, Any]:
    position_pct = slider_position_pct(cyanide)
    return {
        "value_text": f"{cyanide:.2f}%",
        "fill_style": readonly_fill_style(position_pct, READONLY_FILL_COLOR),
        "thumb_style": readonly_thumb_style(position_pct, READONLY_FILL_COLOR),
    }


def gangue_state(static_values: Dict[str, float]) -> Dict[str, Any]:
    grouped_total = float(sum(value_or_default(static_values.get(col), 0.0) for col in GROUPED_MODEL_COLUMNS))
    gangue = 100.0 - grouped_total
    overflow = max(0.0, grouped_total - 100.0)
    if gangue >= 0.0:
        position_pct = slider_position_pct(gangue)
        slider_color = READONLY_FILL_COLOR
        note_text = ""
        note_style = dict(READONLY_NOTE_STYLE)
        note_style["display"] = "none"
    else:
        position_pct = 100.0
        slider_color = READONLY_ALERT_COLOR
        note_text = f"Grouped ores exceed 100% by {overflow:.2f}%"
        note_style = dict(READONLY_NOTE_STYLE)
        note_style["color"] = "#b42318"

    value_style = dict(READONLY_VALUE_STYLE)
    if gangue < 0.0:
        value_style["color"] = "#b42318"
    return {
        "grouped_total": grouped_total,
        "gangue": gangue,
        "value_text": f"{gangue:.2f}%",
        "fill_style": readonly_fill_style(position_pct, slider_color),
        "thumb_style": readonly_thumb_style(position_pct, slider_color),
        "note_text": note_text,
        "note_style": note_style,
        "value_style": value_style,
    }


def build_static_slider_specs(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    specs: Dict[str, Dict[str, float]] = {}
    for col in BASE.STATIC_PREDICTOR_COLUMNS:
        if col not in df.columns:
            specs[col] = {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.1}
            continue
        values = np.asarray([BASE.scalar_from_maybe_array(v) for v in df[col]], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            specs[col] = {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.1}
            continue
        min_value = float(np.nanmin(values))
        max_value = float(np.nanmax(values))
        if max_value <= min_value:
            max_value = min_value + 1.0
        specs[col] = {
            "min": min_value,
            "max": max_value,
            "default": float(np.nanmedian(values)),
            "step": nice_step(min_value, max_value),
        }
    return specs


def build_weekly_catalyst_spec(df: pd.DataFrame) -> Dict[str, float]:
    status_col = BASE.STATUS_COL_PRIMARY if BASE.STATUS_COL_PRIMARY in df.columns else BASE.STATUS_COL_FALLBACK
    values_gt_week: List[float] = []
    for _, row in df.iterrows():
        status = BASE.normalize_status(row.get(status_col, ""))
        if status != "Catalyzed":
            continue
        t_raw = BASE.parse_listlike(row.get(BASE.TIME_COL_COLUMNS, np.nan))
        c_raw = BASE.parse_listlike(row.get(BASE.CATALYST_CUM_COL, np.nan))
        valid = np.isfinite(t_raw) & np.isfinite(c_raw)
        t = np.asarray(t_raw[valid], dtype=float)
        c = np.asarray(c_raw[valid], dtype=float)
        if t.size < 2:
            continue
        order = np.argsort(t)
        t = t[order]
        c = c[order]
        t_unique, inv = np.unique(t, return_inverse=True)
        c_unique = np.full(t_unique.shape, np.nan, dtype=float)
        for i, j in enumerate(inv):
            c_unique[j] = c[i] if not np.isfinite(c_unique[j]) else max(c_unique[j], c[i])
        c_clean = BASE.clean_cumulative_profile(c_unique, force_zero=False)
        weekly_kg_week, _ = BASE.average_weekly_catalyst_from_recent_history(
            time_days=t_unique,
            catalyst_cum=c_clean,
            window_days=float(BASE.CONFIG.get("catalyst_extension_window_days", 21.0)),
            week_days=7.0,
        )
        if np.isfinite(weekly_kg_week):
            values_gt_week.append(float(weekly_kg_week) * 1000.0)

    if len(values_gt_week) == 0:
        return {"min": 0.0, "max": 50.0, "default": 5.0, "step": 0.5}
    arr = np.asarray(values_gt_week, dtype=float)
    min_value = float(np.nanmin(arr))
    max_value = float(np.nanmax(arr))
    if max_value <= min_value:
        max_value = min_value + 1.0
    return {
        "min": max(0.0, min_value),
        "max": max_value,
        "default": float(np.nanmedian(arr)),
        "step": nice_step(max(0.0, min_value), max_value),
    }


def checkpoint_model_kwargs(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(BASE.CONFIG)
    cfg.update(ckpt.get("config", {}))
    params = inspect.signature(BASE.PairCurveNet.__init__).parameters
    kwargs: Dict[str, Any] = {}
    for name in params:
        if name == "self":
            continue
        if name == "n_static":
            kwargs[name] = len(ckpt["static_predictor_columns"])
        elif name == "hidden_dim":
            kwargs[name] = int(cfg["hidden_dim"])
        elif name == "dropout":
            kwargs[name] = float(cfg["dropout"])
        elif name == "ctrl_lb":
            kwargs[name] = np.asarray(ckpt["ctrl_lb"], dtype=float)
        elif name == "ctrl_ub":
            kwargs[name] = np.asarray(ckpt["ctrl_ub"], dtype=float)
        elif name == "cat_lb":
            kwargs[name] = np.asarray(ckpt["cat_lb"], dtype=float)
        elif name == "cat_ub":
            kwargs[name] = np.asarray(ckpt["cat_ub"], dtype=float)
        elif name == "tmax_days":
            kwargs[name] = float(ckpt["tmax_days"])
        elif name == "geo_idx":
            kwargs[name] = list(ckpt.get("geo_idx", []))
        elif name in cfg:
            kwargs[name] = cfg[name]
        else:
            raise KeyError(f"Checkpoint/model constructor mismatch for argument '{name}'")
    return kwargs


def load_saved_member_models(project_root: str) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    model_dir = resolve_member_model_dir(project_root)
    model_paths = sorted(glob.glob(os.path.join(model_dir, "*.pt")))
    if len(model_paths) == 0:
        raise FileNotFoundError(
            "No member checkpoints were found. "
            "Run NN_ExpEq_columns_only.py first to generate them."
        )

    members: List[Dict[str, Any]] = []
    skipped_paths: List[str] = []
    for path in model_paths:
        ckpt = torch.load(path, map_location=BASE.device)
        model = BASE.PairCurveNet(**checkpoint_model_kwargs(ckpt)).to(BASE.device)
        try:
            model.load_state_dict(ckpt["state_dict"])
        except RuntimeError as exc:
            message = str(exc)
            if "Missing key(s) in state_dict" in message or "Unexpected key(s) in state_dict" in message:
                skipped_paths.append(path)
                continue
            raise RuntimeError(
                f"Checkpoint architecture mismatch for {path}. "
                "Re-run NN_ExpEq_columns_only.py to regenerate the saved member models with the current model structure."
            ) from exc
        model.eval()
        members.append(
            {
                "path": path,
                "name": os.path.basename(path),
                "model": model,
                "checkpoint": ckpt,
                "cum_scale": float(ckpt["cum_scale"]),
            }
        )
    if len(members) == 0:
        skipped_text = "\n".join(skipped_paths) if skipped_paths else model_dir
        raise RuntimeError(
            "No compatible saved member checkpoints were found for the current model structure. "
            "Re-run NN_ExpEq_columns_only.py to regenerate them.\n"
            f"Skipped:\n{skipped_text}"
        )
    return members, model_dir, skipped_paths


def scale_static_with_checkpoint(static_raw: np.ndarray, checkpoint: Dict[str, Any]) -> np.ndarray:
    x = np.asarray(static_raw, dtype=float).copy()
    imp = np.asarray(checkpoint["imputer_statistics"], dtype=float)
    mean = np.asarray(checkpoint["scaler_mean"], dtype=float)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=float)
    missing = ~np.isfinite(x)
    x[missing] = imp[missing]
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return (x - mean) / scale


def build_time_grid(max_day: float) -> np.ndarray:
    step_days = float(BASE.CONFIG.get("ensemble_plot_step_days", 1.0))
    max_day = float(np.clip(max_day, 1.0, 2500.0))
    grid = np.arange(0.0, max_day + step_days, step_days, dtype=float)
    grid = grid[grid <= max_day + 1e-9]
    if grid.size == 0 or abs(grid[-1] - max_day) > 1e-9:
        grid = np.append(grid, max_day)
    return np.unique(grid.astype(float))


def build_catalyst_cumulative_schedule(
    time_days: np.ndarray,
    weekly_catalyst_gt_week: float,
    catalyst_addition_start_day: float,
) -> np.ndarray:
    weekly_kg_week = float(max(0.0, weekly_catalyst_gt_week)) / 1000.0
    start_day = float(np.clip(catalyst_addition_start_day, 0.0, 700.0))
    active_days = np.maximum(np.asarray(time_days, dtype=float) - start_day, 0.0)
    return (weekly_kg_week / 7.0) * active_days


def format_percentile_label(value: float) -> str:
    return f"P{int(round(float(value))):02d}"


def confidence_interval_state(interval_high: float) -> Dict[str, Any]:
    interval_high = float(np.clip(interval_high, 60.0, 95.0))
    pi_low = 100.0 - interval_high
    pi_high = interval_high
    return {
        "interval_high": interval_high,
        "pi_low": pi_low,
        "pi_high": pi_high,
        "band_label": f"{format_percentile_label(pi_low)}-{format_percentile_label(pi_high)}",
    }


def predict_ensemble_curves(
    members: List[Dict[str, Any]],
    static_values: Dict[str, float],
    weekly_catalyst_gt_week: float,
    catalyst_addition_start_day: float,
    confidence_interval_high: float,
    max_day: float,
) -> Dict[str, Any]:
    time_days = build_time_grid(max_day)
    weekly_kg_week = float(max(0.0, weekly_catalyst_gt_week)) / 1000.0
    catalyst_start_day = float(np.clip(catalyst_addition_start_day, 0.0, 700.0))
    ci_state = confidence_interval_state(confidence_interval_high)
    catalyst_cum = build_catalyst_cumulative_schedule(
        time_days=time_days,
        weekly_catalyst_gt_week=weekly_catalyst_gt_week,
        catalyst_addition_start_day=catalyst_start_day,
    )
    static_raw = np.asarray([float(static_values[col]) for col in BASE.STATIC_PREDICTOR_COLUMNS], dtype=float)

    member_preds = []
    for member in members:
        static_scaled = scale_static_with_checkpoint(static_raw, member["checkpoint"])
        member_preds.append(
            BASE.predict_new_sample_member(
                model=member["model"],
                static_scaled=static_scaled,
                time_days=time_days,
                catalyst_cum=catalyst_cum,
                cum_scale=float(member["cum_scale"]),
                control_time_days=time_days,
            )
        )

    ctrl_stack = np.vstack([p["control_pred"] for p in member_preds])
    cat_stack = np.vstack([p["catalyzed_pred"] for p in member_preds])
    return {
        "time_days": time_days,
        "weekly_catalyst_gt_week": float(weekly_catalyst_gt_week),
        "weekly_catalyst_kg_week": weekly_kg_week,
        "catalyst_addition_start_day": catalyst_start_day,
        "confidence_interval_high": ci_state["interval_high"],
        "pi_low": ci_state["pi_low"],
        "pi_high": ci_state["pi_high"],
        "band_label": ci_state["band_label"],
        "cumulative_catalyst_addition_kg_t": catalyst_cum,
        "control_pred_mean": np.mean(ctrl_stack, axis=0),
        "control_pred_p10": np.percentile(ctrl_stack, ci_state["pi_low"], axis=0),
        "control_pred_p90": np.percentile(ctrl_stack, ci_state["pi_high"], axis=0),
        "catalyzed_pred_mean": np.mean(cat_stack, axis=0),
        "catalyzed_pred_p10": np.percentile(cat_stack, ci_state["pi_low"], axis=0),
        "catalyzed_pred_p90": np.percentile(cat_stack, ci_state["pi_high"], axis=0),
        "tau_days_mean": float(np.mean([p["tau_days"] for p in member_preds])),
        "temp_days_mean": float(np.mean([p["temp_days"] for p in member_preds])),
        "kappa_mean": float(np.mean([p["kappa"] for p in member_preds])),
        "aging_strength_mean": float(np.mean([p.get("aging_strength", np.nan) for p in member_preds])),
        "n_members": int(len(member_preds)),
    }


def make_prediction_figure(pred: Dict[str, Any], max_day: float) -> go.Figure:
    t = np.asarray(pred["time_days"], dtype=float)
    catalyst_start_day = float(pred["catalyst_addition_start_day"])
    cat_mask = t >= catalyst_start_day - 1e-9
    t_cat = t[cat_mask]
    control_mean = np.asarray(pred["control_pred_mean"], dtype=float)
    catalyzed_mean = np.asarray(pred["catalyzed_pred_mean"], dtype=float)
    delta_mean = catalyzed_mean - control_mean
    nonzero_control = np.abs(control_mean) > 1e-9
    delta_normalized = np.full_like(control_mean, np.nan)
    delta_normalized[nonzero_control] = (delta_mean[nonzero_control] / control_mean[nonzero_control]) * 100.0
    delta_normalized_text = np.asarray(
        [f"{value:.1f}%" if np.isfinite(value) else "N/A" for value in delta_normalized],
        dtype=object,
    )
    hover_customdata = np.empty((len(t), 4), dtype=object)
    hover_customdata[:, 0] = control_mean
    hover_customdata[:, 1] = catalyzed_mean
    hover_customdata[:, 2] = delta_mean
    hover_customdata[:, 3] = delta_normalized_text
    hover_template = (
        "leach_duration_days = %{x}"
        "<br><span style='color:#1f77b4'>Control Cu Rec = %{customdata[0]:.1f}</span>"
        "<br><span style='color:#ff7f0e'>Catalyzed Cu Rec = %{customdata[1]:.1f}</span>"
        "<br>Delta = %{customdata[2]:.1f}%"
        "<br>DeltaNormalized = %{customdata[3]}"
        "<extra></extra>"
    )
    band_label = str(pred.get("band_label", "P10-P90"))
    upper_label = format_percentile_label(float(pred.get("pi_high", 90.0)))
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.asarray(pred["control_pred_p90"], dtype=float),
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            name=f"Control {upper_label}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.asarray(pred["control_pred_p10"], dtype=float),
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.18)",
            name=f"Control {band_label}",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=control_mean,
            mode="lines",
            line={"color": "#1f77b4", "width": 3},
            name="Control Ensemble Mean",
            customdata=hover_customdata,
            hovertemplate=hover_template,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t_cat,
            y=np.asarray(pred["catalyzed_pred_p90"], dtype=float)[cat_mask],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            name=f"Catalyzed {upper_label}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_cat,
            y=np.asarray(pred["catalyzed_pred_p10"], dtype=float)[cat_mask],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.18)",
            name=f"Catalyzed {band_label}",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_cat,
            y=catalyzed_mean[cat_mask],
            mode="lines",
            line={"color": "#ff7f0e", "width": 3},
            name="Catalyzed Ensemble Mean",
            customdata=hover_customdata[cat_mask],
            hovertemplate=hover_template,
        )
    )

    weekly_text = (
        f"Weekly catalyst used: {pred['weekly_catalyst_gt_week']:.2f} g/t/week"
        f"<br>Catalyst addition start day: {pred['catalyst_addition_start_day']:.0f}"
        f"<br>Confidence band: {band_label}"
        f"<br>Assumption: constant catalyst addition from selected start day"
        f"<br>Members loaded: {pred['n_members']}"
    )
    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        text=weekly_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#666666",
        borderwidth=1,
        font={"size": 12},
    )
    fig.update_layout(
        template="plotly_white",
        title="Interactive Ensemble Prediction",
        height=760,
        margin={"l": 70, "r": 30, "t": 70, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
        hoverlabel={
            "bgcolor": "rgba(255, 255, 255, 0.72)",
            "bordercolor": "rgba(60, 60, 60, 0.25)",
            "font": {"color": "#1f2933"},
        },
    )
    fig.update_xaxes(title_text="leach_duration_days", range=[0, float(max_day)])
    fig.update_yaxes(title_text="cu_recovery_%", range=[0, 80])
    if 0.0 < float(pred["catalyst_addition_start_day"]) < float(max_day):
        fig.add_vline(
            x=float(pred["catalyst_addition_start_day"]),
            line_width=1,
            line_dash="dash",
            line_color="#7a7a7a",
        )
    return fig


def slider_marks(min_value: float, max_value: float) -> Dict[float, str]:
    return {
        float(min_value): f"{min_value:.3g}",
        float(max_value): f"{max_value:.3g}",
    }


def slider_block(
    label: str,
    component_id: str,
    spec: Dict[str, float],
    container_style: Dict[str, Any] | None = None,
) -> html.Div:
    style = {"marginBottom": "0", "minWidth": "0"}
    if container_style:
        style.update(container_style)
    return html.Div(
        [
            html.Div(label, style={"fontWeight": 600, "marginBottom": "4px", "fontSize": "13px"}),
            dcc.Slider(
                id=component_id,
                min=float(spec["min"]),
                max=float(spec["max"]),
                step=float(spec["step"]),
                value=float(spec["default"]),
                marks=slider_marks(float(spec["min"]), float(spec["max"])),
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ],
        style=style,
    )


def readonly_slider_block(
    label: str,
    value_id: str,
    default_text: str,
    fill_id: str,
    default_fill_style: Dict[str, Any],
    thumb_id: str,
    default_thumb_style: Dict[str, Any],
    note_id: str | None = None,
    default_note_text: str = "",
    note_style: Dict[str, Any] | None = None,
    container_style: Dict[str, Any] | None = None,
) -> html.Div:
    style = dict(READONLY_BLOCK_STYLE)
    if container_style:
        style.update(container_style)
    children: List[Any] = [
        html.Div(
            [
                html.Div(label, style={"fontWeight": 600, "fontSize": "13px", "minWidth": "0"}),
                html.Div(
                    [
                        html.Div(id=value_id, children=default_text, style=dict(READONLY_VALUE_STYLE)),
                        html.Span("Auto", style=dict(READONLY_BADGE_STYLE)),
                    ],
                    style={"display": "flex", "alignItems": "center", "gap": "8px", "flexWrap": "wrap"},
                ),
            ],
            style={"display": "flex", "justifyContent": "space-between", "gap": "8px", "alignItems": "center", "marginBottom": "6px"},
        ),
        html.Div(
            [
                html.Div(
                    style={
                        "position": "absolute",
                        "top": "50%",
                        "left": "0",
                        "right": "0",
                        "height": "4px",
                        "transform": "translateY(-50%)",
                        "background": READONLY_TRACK_COLOR,
                        "borderRadius": "999px",
                    }
                ),
                html.Div(id=fill_id, style=default_fill_style),
                html.Div(id=thumb_id, style=default_thumb_style),
            ],
            style={"position": "relative", "height": "18px", "padding": "0 7px", "cursor": "not-allowed"},
        ),
        html.Div(
            [
                html.Span("0"),
                html.Span("100"),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "marginTop": "2px",
                "fontSize": "10px",
                "color": "#8a929c",
            },
        ),
    ]
    if note_id is not None:
        children.append(
            html.Div(
                id=note_id,
                children=default_note_text,
                style=dict(READONLY_NOTE_STYLE if note_style is None else note_style),
            )
        )
    return html.Div(children, style=style)


def graph_export_config() -> Dict[str, Any]:
    return {
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "interactive_ensemble_prediction_300dpi",
            "width": PNG_EXPORT_WIDTH_PX,
            "height": PNG_EXPORT_HEIGHT_PX,
            "scale": PNG_EXPORT_SCALE,
        },
    }


def resolve_excel_writer_engine() -> str:
    for engine in ("openpyxl", "xlsxwriter"):
        if importlib.util.find_spec(engine) is not None:
            return engine
    raise RuntimeError("Excel export requires either openpyxl or xlsxwriter to be installed.")


def build_prediction_export_days(max_day: float, export_step_days: float = 7.0) -> np.ndarray:
    export_step_days = float(max(export_step_days, 1.0))
    max_day = float(np.clip(max_day, 1.0, 2500.0))
    export_days = np.arange(0.0, max_day + export_step_days, export_step_days, dtype=float)
    export_days = export_days[export_days <= max_day + 1e-9]
    if export_days.size == 0 or abs(export_days[-1] - max_day) > 1e-9:
        export_days = np.append(export_days, max_day)
    return np.unique(export_days.astype(float))


def resolve_static_state(
    predictor_values: Tuple[float, ...],
    static_specs: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    raw_static_values = {
        col: value_or_default(val, static_specs[col]["default"])
        for col, val in zip(INTERACTIVE_STATIC_COLUMNS, predictor_values)
    }
    chemistry = chemistry_state(
        raw_static_values.get(ACID_SOLUBLE_COL),
        raw_static_values.get(RESIDUAL_CPY_COL),
        static_specs,
    )
    static_values = {
        col: raw_static_values.get(col, float(static_specs[col]["default"]))
        for col in BASE.STATIC_PREDICTOR_COLUMNS
        if col != CYANIDE_SOLUBLE_COL
    }
    static_values[ACID_SOLUBLE_COL] = chemistry["acid"]
    static_values[RESIDUAL_CPY_COL] = chemistry["residual"]
    static_values[CYANIDE_SOLUBLE_COL] = chemistry["cyanide"]
    return static_values, cyanide_visual_state(chemistry["cyanide"]), gangue_state(static_values)


def build_export_dataframes(
    pred: Dict[str, Any],
    static_values: Dict[str, float],
    max_day: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    time_days = np.asarray(pred["time_days"], dtype=float)
    export_days = build_prediction_export_days(max_day=max_day, export_step_days=7.0)
    control_mean = np.interp(export_days, time_days, np.asarray(pred["control_pred_mean"], dtype=float))
    catalyzed_mean = np.interp(export_days, time_days, np.asarray(pred["catalyzed_pred_mean"], dtype=float))
    catalyst_cumulative = np.interp(
        export_days,
        time_days,
        np.asarray(pred["cumulative_catalyst_addition_kg_t"], dtype=float),
    )
    delta = catalyzed_mean - control_mean
    delta_normalized_pct = np.full_like(control_mean, np.nan)
    nonzero_control = np.abs(control_mean) > 1e-9
    delta_normalized_pct[nonzero_control] = (delta[nonzero_control] / control_mean[nonzero_control]) * 100.0

    inputs_rows = [
        {"section": "plot_settings", "parameter": "max_day", "label": "Max day", "value": float(max_day)},
        {
            "section": "plot_settings",
            "parameter": "weekly_catalyst_gt_week",
            "label": "Weekly catalyst used (g/t/week)",
            "value": float(pred["weekly_catalyst_gt_week"]),
        },
        {
            "section": "plot_settings",
            "parameter": "catalyst_addition_start_day",
            "label": "Catalyst addition start day",
            "value": float(pred["catalyst_addition_start_day"]),
        },
        {
            "section": "plot_settings",
            "parameter": "confidence_interval_high_pct",
            "label": "Confidence interval (%)",
            "value": float(pred["confidence_interval_high"]),
        },
        {
            "section": "plot_settings",
            "parameter": "ensemble_band_label",
            "label": "Ensemble band label",
            "value": str(pred["band_label"]),
        },
        {
            "section": "plot_settings",
            "parameter": "n_members",
            "label": "Members loaded",
            "value": int(pred["n_members"]),
        },
    ]
    for col in BASE.STATIC_PREDICTOR_COLUMNS:
        inputs_rows.append(
            {
                "section": "model_inputs",
                "parameter": col,
                "label": BASE.HEADERS_DICT_COLUMNS[col][0],
                "value": float(static_values[col]),
            }
        )

    inputs_df = pd.DataFrame(inputs_rows)
    predictions_df = pd.DataFrame(
        {
            "leach_duration_days": export_days,
            "cumulative_catalyst_addition_kg_t": catalyst_cumulative,
            "control_pred_mean_cu_recovery_pct": control_mean,
            "catalyzed_pred_mean_cu_recovery_pct": catalyzed_mean,
            "delta_cu_recovery_pct": delta,
            "delta_normalized_pct": delta_normalized_pct,
        }
    )
    return inputs_df, predictions_df


def build_prediction_export_bytes(
    pred: Dict[str, Any],
    static_values: Dict[str, float],
    max_day: float,
) -> bytes:
    inputs_df, predictions_df = build_export_dataframes(pred=pred, static_values=static_values, max_day=max_day)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine=resolve_excel_writer_engine()) as writer:
        inputs_df.to_excel(writer, sheet_name="inputs", index=False)
        predictions_df.to_excel(writer, sheet_name="predictions_7d", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def create_app(project_root: str) -> Dash:
    df = load_training_dataframe()
    static_specs = build_static_slider_specs(df)
    weekly_spec = build_weekly_catalyst_spec(df)
    members, model_dir, skipped_paths = load_saved_member_models(project_root)

    app = Dash(__name__)
    default_static_values = {col: static_specs[col]["default"] for col in BASE.STATIC_PREDICTOR_COLUMNS}
    default_chemistry = chemistry_state(
        default_static_values.get(ACID_SOLUBLE_COL),
        default_static_values.get(RESIDUAL_CPY_COL),
        static_specs,
    )
    default_static_values[ACID_SOLUBLE_COL] = default_chemistry["acid"]
    default_static_values[RESIDUAL_CPY_COL] = default_chemistry["residual"]
    default_static_values[CYANIDE_SOLUBLE_COL] = default_chemistry["cyanide"]
    default_cyanide_visual = cyanide_visual_state(default_chemistry["cyanide"])
    default_gangue = gangue_state(default_static_values)

    predictor_controls: List[html.Div] = []
    for col in BASE.STATIC_PREDICTOR_COLUMNS:
        if col in TOP_CONTROL_STATIC_COLUMNS:
            continue
        if col == CYANIDE_SOLUBLE_COL:
            predictor_controls.append(
                readonly_slider_block(
                    label=BASE.HEADERS_DICT_COLUMNS[col][0],
                    value_id="derived-cyanide-value",
                    default_text=f"{default_chemistry['cyanide']:.2f}%",
                    fill_id="derived-cyanide-fill",
                    default_fill_style=default_cyanide_visual["fill_style"],
                    thumb_id="derived-cyanide-thumb",
                    default_thumb_style=default_cyanide_visual["thumb_style"],
                )
            )
            continue

        spec = dict(static_specs[col])
        spec["default"] = float(default_static_values[col])
        if col == ACID_SOLUBLE_COL:
            spec["max"] = float(default_chemistry["acid_max"])
        elif col == RESIDUAL_CPY_COL:
            spec["max"] = float(default_chemistry["residual_max"])

        predictor_controls.append(
            slider_block(
                label=BASE.HEADERS_DICT_COLUMNS[col][0],
                component_id=f"predictor-{col}",
                spec=spec,
            )
        )

        if col == GROUPED_GANGUE_INSERT_AFTER:
            predictor_controls.append(
                readonly_slider_block(
                    label="Gangue (derived)",
                    value_id="derived-gangue-value",
                    default_text=default_gangue["value_text"],
                    fill_id="derived-gangue-fill",
                    default_fill_style=default_gangue["fill_style"],
                    thumb_id="derived-gangue-thumb",
                    default_thumb_style=default_gangue["thumb_style"],
                    note_id="derived-gangue-note",
                    default_note_text=default_gangue["note_text"],
                    note_style=default_gangue["note_style"],
                )
            )

    default_pred = predict_ensemble_curves(
        members=members,
        static_values=default_static_values,
        weekly_catalyst_gt_week=float(weekly_spec["default"]),
        catalyst_addition_start_day=0.0,
        confidence_interval_high=90.0,
        max_day=float(BASE.CONFIG.get("ensemble_plot_target_day", 2500.0)),
    )

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1("NN ExpEq Columns Only Interactive Plot", style={"marginBottom": "6px"}),
                    html.Div(
                        [
                            html.Div(f"Model root: {project_root}"),
                            html.Div(f"Using saved checkpoints from: {model_dir}"),
                            html.Div(f"Loaded {len(members)} member model(s)."),
                            html.Div(f"Skipped {len(skipped_paths)} incompatible checkpoint(s)."),
                        ],
                        style={"color": "#50555c", "fontSize": "14px", "lineHeight": "1.6"},
                    ),
                ],
                style={"padding": "24px 28px 8px 28px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "Controls",
                                style={"fontSize": "20px", "fontWeight": 700, "marginBottom": "18px"},
                            ),
                            html.Div(
                                [
                                    slider_block(
                                        label="Max day",
                                        component_id="max-day",
                                        spec={
                                            "min": 50.0,
                                            "max": float(BASE.CONFIG.get("ensemble_plot_target_day", 2500.0)),
                                            "default": float(BASE.CONFIG.get("ensemble_plot_target_day", 2500.0)),
                                            "step": 10.0,
                                        },
                                    ),
                                    slider_block(
                                        label="Weekly catalyst used (g/t/week)",
                                        component_id="weekly-catalyst-gt-week",
                                        spec=weekly_spec,
                                    ),
                                    slider_block(
                                        label="Confidence interval (%)",
                                        component_id="confidence-interval",
                                        spec={
                                            "min": 60.0,
                                            "max": 95.0,
                                            "default": 90.0,
                                            "step": 1.0,
                                        },
                                    ),
                                    slider_block(
                                        label="Catalyst addition start day",
                                        component_id="catalyst-addition-start-day",
                                        spec={
                                            "min": 0.0,
                                            "max": 700.0,
                                            "default": 200.0,
                                            "step": 1.0,
                                        },
                                    ),
                                    *[
                                        slider_block(
                                            label=BASE.HEADERS_DICT_COLUMNS[col][0],
                                            component_id=f"predictor-{col}",
                                            spec={
                                                **dict(static_specs[col]),
                                                "default": float(default_static_values[col]),
                                            },
                                        )
                                        for col in TOP_CONTROL_STATIC_COLUMNS
                                    ],
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))",
                                    "gap": "12px 14px",
                                    "alignItems": "start",
                                },
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Download Excel (7-day)",
                                        id="download-predictions-excel-button",
                                        n_clicks=0,
                                        style={
                                            "padding": "10px 14px",
                                            "border": "1px solid #b4aa95",
                                            "background": "#efe7d8",
                                            "color": "#1c232b",
                                            "borderRadius": "6px",
                                            "fontWeight": 700,
                                            "cursor": "pointer",
                                        },
                                    ),
                                    html.Div(
                                        "Exports current inputs plus predictions every 7 days to Excel.",
                                        style={"fontSize": "12px", "color": "#5a626c"},
                                    ),
                                    dcc.Download(id="download-predictions-excel"),
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "6px",
                                    "marginTop": "14px",
                                },
                            ),
                            html.Hr(),
                            html.Div(
                                predictor_controls,
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))",
                                    "gap": "12px 14px",
                                    "maxHeight": "70vh",
                                    "overflowY": "auto",
                                    "paddingRight": "8px",
                                    "paddingBottom": "52px",
                                    "alignItems": "start",
                                },
                            ),
                        ],
                        style={
                            "width": "32%",
                            "minWidth": "340px",
                            "padding": "22px 24px",
                            "background": "#f7f4ee",
                            "borderRight": "1px solid #d9d3c5",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="prediction-graph",
                                figure=make_prediction_figure(
                                    default_pred,
                                    float(BASE.CONFIG.get("ensemble_plot_target_day", 2500.0)),
                                ),
                                config=graph_export_config(),
                                style={"height": "78vh"},
                            ),
                            html.Div(
                                id="prediction-summary",
                                style={
                                    "padding": "0 20px 20px 20px",
                                    "color": "#50555c",
                                    "fontSize": "14px",
                                    "lineHeight": "1.7",
                                },
                            ),
                        ],
                        style={"width": "68%", "background": "#fffdf7"},
                    ),
                ],
                style={
                    "display": "flex",
                    "minHeight": "calc(100vh - 88px)",
                    "background": "#fffdf7",
                    "color": "#1c232b",
                },
            ),
        ],
        style={"fontFamily": "Helvetica, Arial, sans-serif", "background": "#fffdf7"},
    )

    callback_inputs = [
        Input("max-day", "value"),
        Input("weekly-catalyst-gt-week", "value"),
        Input("confidence-interval", "value"),
        Input("catalyst-addition-start-day", "value"),
        *[Input(f"predictor-{col}", "value") for col in INTERACTIVE_STATIC_COLUMNS],
    ]
    callback_states = [
        State("max-day", "value"),
        State("weekly-catalyst-gt-week", "value"),
        State("confidence-interval", "value"),
        State("catalyst-addition-start-day", "value"),
        *[State(f"predictor-{col}", "value") for col in INTERACTIVE_STATIC_COLUMNS],
    ]

    @app.callback(
        Output("prediction-graph", "figure"),
        Output("prediction-summary", "children"),
        Output(f"predictor-{ACID_SOLUBLE_COL}", "max"),
        Output(f"predictor-{ACID_SOLUBLE_COL}", "marks"),
        Output(f"predictor-{RESIDUAL_CPY_COL}", "max"),
        Output(f"predictor-{RESIDUAL_CPY_COL}", "marks"),
        Output("derived-cyanide-value", "children"),
        Output("derived-cyanide-fill", "style"),
        Output("derived-cyanide-thumb", "style"),
        Output("derived-gangue-value", "children"),
        Output("derived-gangue-value", "style"),
        Output("derived-gangue-fill", "style"),
        Output("derived-gangue-thumb", "style"),
        Output("derived-gangue-note", "children"),
        Output("derived-gangue-note", "style"),
        callback_inputs,
    )
    def update_plot(
        max_day: float,
        weekly_catalyst_gt_week: float,
        confidence_interval: float,
        catalyst_addition_start_day: float,
        *predictor_values: float,
    ) -> Tuple[
        go.Figure,
        html.Div,
        float,
        Dict[float, str],
        float,
        Dict[float, str],
        str,
        Dict[str, Any],
        Dict[str, Any],
        str,
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        str,
        Dict[str, Any],
    ]:
        static_values, cyanide_visual, gangue = resolve_static_state(
            predictor_values=tuple(predictor_values),
            static_specs=static_specs,
        )
        chemistry = {
            "acid": float(static_values[ACID_SOLUBLE_COL]),
            "cyanide": float(static_values[CYANIDE_SOLUBLE_COL]),
            "residual": float(static_values[RESIDUAL_CPY_COL]),
            "acid_max": float(
                chemistry_state(static_values[ACID_SOLUBLE_COL], static_values[RESIDUAL_CPY_COL], static_specs)["acid_max"]
            ),
            "residual_max": float(
                chemistry_state(static_values[ACID_SOLUBLE_COL], static_values[RESIDUAL_CPY_COL], static_specs)["residual_max"]
            ),
        }

        pred = predict_ensemble_curves(
            members=members,
            static_values=static_values,
            weekly_catalyst_gt_week=float(weekly_catalyst_gt_week or 0.0),
            catalyst_addition_start_day=float(catalyst_addition_start_day or 0.0),
            confidence_interval_high=float(confidence_interval or 90.0),
            max_day=float(max_day or BASE.CONFIG.get("ensemble_plot_target_day", 2500.0)),
        )
        summary = html.Div(
            [
                html.Div(
                    f"Tau mean: {pred['tau_days_mean']:.1f} days | "
                    f"Temp mean: {pred['temp_days_mean']:.1f} days | "
                    f"Kappa mean: {pred['kappa_mean']:.4f} | "
                    f"Aging strength mean: {pred['aging_strength_mean']:.3f}"
                ),
                html.Div(
                    f"Catalyzed curve assumption: constant catalyst addition starting at day "
                    f"{pred['catalyst_addition_start_day']:.0f}, using the selected weekly rate "
                    "converted internally to cumulative kg/t."
                ),
                html.Div(f"Displayed ensemble band: {pred['band_label']}"),
                html.Div(
                    f"Derived chemistry: Acid {chemistry['acid']:.2f}% | "
                    f"Cyanide {chemistry['cyanide']:.2f}% | "
                    f"Residual {chemistry['residual']:.2f}%"
                ),
            ]
        )
        return (
            make_prediction_figure(pred, float(max_day)),
            summary,
            float(chemistry["acid_max"]),
            slider_marks(float(static_specs[ACID_SOLUBLE_COL]["min"]), float(chemistry["acid_max"])),
            float(chemistry["residual_max"]),
            slider_marks(float(static_specs[RESIDUAL_CPY_COL]["min"]), float(chemistry["residual_max"])),
            cyanide_visual["value_text"],
            cyanide_visual["fill_style"],
            cyanide_visual["thumb_style"],
            gangue["value_text"],
            gangue["value_style"],
            gangue["fill_style"],
            gangue["thumb_style"],
            gangue["note_text"],
            gangue["note_style"],
        )

    @app.callback(
        Output("download-predictions-excel", "data"),
        Input("download-predictions-excel-button", "n_clicks"),
        *callback_states,
        prevent_initial_call=True,
    )
    def download_predictions_excel(
        _n_clicks: int,
        max_day: float,
        weekly_catalyst_gt_week: float,
        confidence_interval: float,
        catalyst_addition_start_day: float,
        *predictor_values: float,
    ) -> Dict[str, Any]:
        static_values, _, _ = resolve_static_state(
            predictor_values=tuple(predictor_values),
            static_specs=static_specs,
        )
        resolved_max_day = float(max_day or BASE.CONFIG.get("ensemble_plot_target_day", 2500.0))
        pred = predict_ensemble_curves(
            members=members,
            static_values=static_values,
            weekly_catalyst_gt_week=float(weekly_catalyst_gt_week or 0.0),
            catalyst_addition_start_day=float(catalyst_addition_start_day or 0.0),
            confidence_interval_high=float(confidence_interval or 90.0),
            max_day=resolved_max_day,
        )
        excel_bytes = build_prediction_export_bytes(
            pred=pred,
            static_values=static_values,
            max_day=resolved_max_day,
        )
        return dcc.send_bytes(excel_bytes, "interactive_ensemble_prediction_7day.xlsx")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive Plotly/Dash app for NN_ExpEq_columns_only using the saved "
            "member checkpoints created after running NN_ExpEq_columns_only.py."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8056)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--project-root", default=None, help="Optional override for the NN_Pytorch_ExpEq_columns_only root.")
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    app = create_app(project_root)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

# %%
