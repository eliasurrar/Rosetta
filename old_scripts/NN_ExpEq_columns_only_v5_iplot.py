import argparse
import glob
import importlib.util
import io
import inspect
import json
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from dash import Dash, Input, Output, State, dcc, html


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_SCRIPT_PATH = os.path.join(THIS_DIR, "NN_ExpEq_columns_only_v5.py")
MINERALOGY_CLUSTERING_PATH = os.path.join(THIS_DIR, "rosetta_mineralogy_clustering.py")


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("nn_expeq_columns_only_v5_base", BASE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base model script: {BASE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()

ACID_SOLUBLE_COL = "acid_soluble_%"
CYANIDE_SOLUBLE_COL = "cyanide_soluble_%"
RESIDUAL_CPY_COL = "residual_cpy_%"
CU_INPUT_COL = "cu_%"
FE_INPUT_COL = "fe_%"
FE_HEAD_COL = "feed_head_fe_%"
CU_HEAD_COL = "feed_head_cu_%"
FE_CU_RATIO_COL = "fe:cu"
CU_FE_RATIO_COL = "cu:fe"
PRIMARY_SULFIDES_EQUIV_COL = "copper_primary_sulfides_equivalent"
SECONDARY_SULFIDES_EQUIV_COL = "copper_secondary_sulfides_equivalent"
SULFIDES_EQUIV_COL = "copper_sulfides_equivalent"
OXIDES_EQUIV_COL = "copper_oxides_equivalent"

RAW_CHEMISTRY_COLUMNS = [ACID_SOLUBLE_COL, RESIDUAL_CPY_COL]
RATIO_COLUMNS = [FE_CU_RATIO_COL, CU_FE_RATIO_COL]
EQUIVALENT_COLUMNS = [
    PRIMARY_SULFIDES_EQUIV_COL,
    SECONDARY_SULFIDES_EQUIV_COL,
    SULFIDES_EQUIV_COL,
    OXIDES_EQUIV_COL,
]
DERIVED_MODEL_COLUMNS = set(RATIO_COLUMNS + EQUIVALENT_COLUMNS)
TOP_CONTROL_STATIC_COLUMNS = [
    "column_height_m",
    "column_inner_diameter_m",
    "material_size_p80_in",
]

SPECIAL_COLUMN_LABELS = {
    ACID_SOLUBLE_COL: "Acid Soluble Cu (%)",
    CYANIDE_SOLUBLE_COL: "Cyanide Soluble Cu (%)",
    RESIDUAL_CPY_COL: "Residual Chalcopyrite (%)",
    FE_INPUT_COL: "Fe %",
    FE_HEAD_COL: "Fe %",
    CU_HEAD_COL: "Cu %",
    "column_volume_m3": "Column Volume (m3)",
    "apparent_bulk_density_t_m3": "Apparent Bulk Density (t/m3)",
    "material_size_to_column_diameter_ratio": "Material Size / Column Diameter Ratio",
}

READONLY_BLOCK_STYLE = {
    "marginBottom": "0",
    "padding": "2px 0 0 0",
    "minWidth": "0",
    "opacity": "0.92",
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

PNG_EXPORT_DPI = 300
PNG_EXPORT_WIDTH_IN = 12.0
PNG_EXPORT_HEIGHT_IN = 8.0
PNG_EXPORT_SCALE = 4
PNG_EXPORT_FINAL_WIDTH_PX = int(PNG_EXPORT_WIDTH_IN * PNG_EXPORT_DPI)
PNG_EXPORT_FINAL_HEIGHT_PX = int(PNG_EXPORT_HEIGHT_IN * PNG_EXPORT_DPI)
PNG_EXPORT_WIDTH_PX = int(PNG_EXPORT_FINAL_WIDTH_PX / PNG_EXPORT_SCALE)
PNG_EXPORT_HEIGHT_PX = int(PNG_EXPORT_FINAL_HEIGHT_PX / PNG_EXPORT_SCALE)
GEOMETRY_RATIO_MAX = 0.25
GEOMETRY_BULK_DENSITY_MIN_T_M3 = 0.70
GEOMETRY_BULK_DENSITY_MAX_T_M3 = 2.0
GEOMETRY_TOL = 1e-9
ORE_SOLID_DENSITY_T_M3 = 2.26
ROSIN_RAMMLER_SHAPE_N = 5.0
ROSIN_RAMMLER_BIN_COUNT = 24
ROSIN_RAMMLER_Q_MIN = 0.02
ROSIN_RAMMLER_Q_MAX = 0.98
ROSIN_RAMMLER_MAX_PACKING_FRACTION = 0.64
ROSIN_RAMMLER_WALL_ALPHA = 5.85
ROSIN_RAMMLER_WALL_EXPONENT = 2.75
CALIBRATED_PACKING_DENSITY_INTERCEPT_T_M3 = 0.9652606688354841
CALIBRATED_PACKING_DENSITY_PACKING_COEFF = 1.376443087819381
CALIBRATED_PACKING_DENSITY_LOG_HEIGHT_COEFF = -0.013423139214821887
CALIBRATED_PACKING_DENSITY_P80_COEFF_M = 4.814201741554867
GROUPED_GANGUE_PRIORITY_COLUMN = "grouped_gangue_silicates"
GROUPED_GANGUE_PRIORITY_SHARE = 0.80


def ordered_unique(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def component_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value).strip())
    token = re.sub(r"-{2,}", "-", token).strip("-")
    return token or "value"


def predictor_component_id(column: str) -> str:
    return f"predictor-{component_token(column)}"


def candidate_project_roots() -> List[str]:
    roots = []
    for attr in ["PROJECT_ROOT", "DEFAULT_PROJECT_ROOT", "LOCAL_PROJECT_ROOT"]:
        value = getattr(BASE, attr, None)
        if isinstance(value, str) and value not in roots:
            roots.append(value)
    sibling_root = os.path.join(THIS_DIR, "NN_Pytorch_ExpEq_columns_only_v5")
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
        "Run NN_ExpEq_columns_only_v5.py first so it writes the checkpoint paths listed in outputs/run_manifest.json.\n"
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
        "Run NN_ExpEq_columns_only_v5.py first and then use the checkpoint directory recorded in outputs/run_manifest.json.\n"
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


def safe_ratio(numerator: float, denominator: float) -> float:
    num = float(numerator) if np.isfinite(numerator) else 0.0
    den = float(denominator) if np.isfinite(denominator) else 0.0
    return float(num / max(den, 1e-6))


def slider_position_pct(value: float, min_value: float, max_value: float) -> float:
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
    input_specs: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    acid_spec = input_specs[ACID_SOLUBLE_COL]
    residual_spec = input_specs[RESIDUAL_CPY_COL]

    acid = float(
        np.clip(
            value_or_default(acid_value, float(acid_spec["default"])),
            float(acid_spec["min"]),
            float(acid_spec["max"]),
        )
    )
    residual = float(
        np.clip(
            value_or_default(residual_value, float(residual_spec["default"])),
            float(residual_spec["min"]),
            float(residual_spec["max"]),
        )
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
        "cyanide": cyanide,
        "residual": residual,
        "acid_max": acid_max,
        "residual_max": residual_max,
    }


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


def readonly_numeric_state(value: float, spec: Dict[str, float]) -> Dict[str, Any]:
    position_pct = slider_position_pct(
        float(value),
        float(spec["min"]),
        float(spec["max"]),
    )
    return {
        "value_text": f"{float(value):.2f}",
        "fill_style": readonly_fill_style(position_pct, READONLY_FILL_COLOR),
        "thumb_style": readonly_thumb_style(position_pct, READONLY_FILL_COLOR),
        "min_text": f"{float(spec['min']):.3g}",
        "max_text": f"{float(spec['max']):.3g}",
    }


def readonly_slider_block(
    label: str,
    value_text: str,
    fill_style: Dict[str, Any],
    thumb_style: Dict[str, Any],
    min_text: str,
    max_text: str,
    badge_text: str = "Auto",
    note_text: str = "",
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
                        html.Div(value_text, style=dict(READONLY_VALUE_STYLE)),
                        html.Span(badge_text, style=dict(READONLY_BADGE_STYLE)),
                    ],
                    style={"display": "flex", "alignItems": "center", "gap": "8px", "flexWrap": "wrap"},
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "gap": "8px",
                "alignItems": "center",
                "marginBottom": "6px",
            },
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
                html.Div(style=fill_style),
                html.Div(style=thumb_style),
            ],
            style={"position": "relative", "height": "18px", "padding": "0 7px", "cursor": "not-allowed"},
        ),
        html.Div(
            [html.Span(min_text), html.Span(max_text)],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "marginTop": "2px",
                "fontSize": "10px",
                "color": "#8a929c",
            },
        ),
    ]
    if note_text:
        children.append(
            html.Div(
                note_text,
                style=dict(READONLY_NOTE_STYLE if note_style is None else note_style),
            )
        )
    return html.Div(children, style=style)


def readonly_value_block(
    label: str,
    value_text: str,
    badge_text: str = "Auto",
    container_style: Dict[str, Any] | None = None,
) -> html.Div:
    style = dict(READONLY_BLOCK_STYLE)
    if container_style:
        style.update(container_style)
    return html.Div(
        [
            html.Div(label, style={"fontWeight": 600, "fontSize": "13px", "marginBottom": "4px"}),
            html.Div(
                [
                    html.Div(value_text, style=dict(READONLY_VALUE_STYLE)),
                    html.Span(badge_text, style=dict(READONLY_BADGE_STYLE)),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "8px", "flexWrap": "wrap"},
            ),
        ],
        style=style,
    )


def graph_export_config() -> Dict[str, Any]:
    return {
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "interactive_ensemble_prediction_v5_300dpi",
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


def display_label(column: str) -> str:
    if column in BASE.HEADERS_DICT_COLUMNS:
        return str(BASE.HEADERS_DICT_COLUMNS[column][0])
    if column in getattr(BASE, "INPUT_ONLY_HEADERS_DICT_COLUMNS", {}):
        return str(BASE.INPUT_ONLY_HEADERS_DICT_COLUMNS[column][0])
    if column in SPECIAL_COLUMN_LABELS:
        return SPECIAL_COLUMN_LABELS[column]
    text = column.replace("grouped_", "").replace("_", " ")
    text = text.replace(":", " / ")
    return text.title()


def format_ratio_one_to_x(ratio_value: float) -> str:
    ratio = float(ratio_value)
    if not np.isfinite(ratio) or ratio <= 0.0:
        return "n/a"
    inverse_ratio = 1.0 / ratio
    rounded_up = float(np.ceil(inverse_ratio * 10.0) / 10.0)
    return f"1:{int(rounded_up)}"


def source_candidates_for_column(column: str) -> List[str]:
    if column == FE_INPUT_COL:
        return [FE_HEAD_COL, FE_INPUT_COL]
    if column == CU_INPUT_COL:
        return [CU_INPUT_COL, CU_HEAD_COL]
    return [column]


def build_numeric_slider_specs(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    specs: Dict[str, Dict[str, float]] = {}
    for column in ordered_unique(columns):
        source_column = next((candidate for candidate in source_candidates_for_column(column) if candidate in df.columns), None)
        if source_column is None:
            specs[column] = {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.1}
            continue
        values = np.asarray([BASE.scalar_from_maybe_array(v) for v in df[source_column]], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            specs[column] = {"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.1}
            continue
        min_value = float(np.nanmin(values))
        max_value = float(np.nanmax(values))
        if max_value <= min_value:
            max_value = min_value + 1.0
        specs[column] = {
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


def build_irrigation_rate_spec(df: pd.DataFrame) -> Dict[str, float]:
    values_l_m2_h: List[float] = []
    for _, row in df.iterrows():
        t_raw = BASE.parse_listlike(row.get(BASE.TIME_COL_COLUMNS, np.nan))
        l_raw = BASE.parse_listlike(row.get(BASE.LIXIVIANT_CUM_COL, np.nan))
        feed_mass_kg = BASE.scalar_from_maybe_array(row.get(BASE.FEED_MASS_COL, np.nan))
        column_inner_diameter_m = BASE.scalar_from_maybe_array(row.get("column_inner_diameter_m", np.nan))
        t_clean, l_clean = BASE.prepare_cumulative_profile_with_time(t_raw, l_raw, force_zero=False)
        if t_clean.size < 2:
            continue
        rate_profile = BASE.convert_lixiviant_cum_to_irrigation_rate_l_m2_h(
            time_days=t_clean,
            cumulative_lixiviant_m3_t=l_clean,
            feed_mass_kg=feed_mass_kg,
            column_inner_diameter_m=column_inner_diameter_m,
        )
        positive = rate_profile[np.isfinite(rate_profile) & (rate_profile > 0.0)]
        if positive.size > 0:
            values_l_m2_h.append(float(np.nanmedian(positive)))

    if len(values_l_m2_h) == 0:
        return {"min": 0.0, "max": 30.0, "default": 10.0, "step": 0.5}
    arr = np.asarray(values_l_m2_h, dtype=float)
    min_value = float(np.nanmin(arr))
    max_value = float(np.nanmax(arr))
    if max_value <= min_value:
        max_value = min_value + 1.0
    min_value = max(0.0, min_value)
    return {
        "min": min_value,
        "max": max_value,
        "default": float(np.nanmedian(arr)),
        "step": nice_step(min_value, max_value),
    }


def extract_grouped_columns_from_mineralogy_script() -> List[str]:
    grouped_columns: List[str] = []
    if os.path.exists(MINERALOGY_CLUSTERING_PATH):
        with open(MINERALOGY_CLUSTERING_PATH, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                match = re.search(r"df_mineralogy_grouped\['(grouped_[^']+)'\]", line)
                if match:
                    grouped_columns.append(match.group(1))
    if len(grouped_columns) == 0:
        grouped_columns = [c for c in BASE.STATIC_PREDICTOR_COLUMNS if c.startswith("grouped_")]
    active_grouped = [c for c in BASE.STATIC_PREDICTOR_COLUMNS if c.startswith("grouped_")]
    return ordered_unique(grouped_columns + active_grouped)


ALL_GROUPED_COLUMNS = extract_grouped_columns_from_mineralogy_script()
GROUPED_MODEL_COLUMNS = [c for c in ALL_GROUPED_COLUMNS if c in BASE.STATIC_PREDICTOR_COLUMNS]
GROUPED_INACTIVE_COLUMNS = [c for c in ALL_GROUPED_COLUMNS if c not in BASE.STATIC_PREDICTOR_COLUMNS]
TOP_DISPLAY_COLUMNS = [c for c in TOP_CONTROL_STATIC_COLUMNS if c in getattr(BASE, "USER_INPUT_COLUMNS", BASE.STATIC_PREDICTOR_COLUMNS)]


def needs_fe_input() -> bool:
    return any(c in BASE.STATIC_PREDICTOR_COLUMNS for c in [FE_CU_RATIO_COL, CU_FE_RATIO_COL, FE_HEAD_COL])


def needs_cu_input() -> bool:
    helper_targets = [
        CU_INPUT_COL,
        CU_HEAD_COL,
        FE_CU_RATIO_COL,
        CU_FE_RATIO_COL,
        PRIMARY_SULFIDES_EQUIV_COL,
        SECONDARY_SULFIDES_EQUIV_COL,
        SULFIDES_EQUIV_COL,
        OXIDES_EQUIV_COL,
    ]
    return any(c in BASE.STATIC_PREDICTOR_COLUMNS for c in helper_targets)


def build_main_editable_columns() -> List[str]:
    prefix_columns: List[str] = []
    if needs_cu_input():
        prefix_columns.append(CU_INPUT_COL)
    if needs_fe_input():
        prefix_columns.append(FE_INPUT_COL)
    prefix_columns.extend(RAW_CHEMISTRY_COLUMNS)

    active_non_grouped = [
        c
        for c in BASE.STATIC_PREDICTOR_COLUMNS
        if c not in TOP_DISPLAY_COLUMNS
        and not c.startswith("grouped_")
        and c not in DERIVED_MODEL_COLUMNS
        and c not in {FE_HEAD_COL, CU_HEAD_COL}
    ]
    return ordered_unique(prefix_columns + active_non_grouped)


MAIN_EDITABLE_COLUMNS = build_main_editable_columns()
EDITABLE_INPUT_COLUMNS = ordered_unique(TOP_DISPLAY_COLUMNS + MAIN_EDITABLE_COLUMNS + GROUPED_MODEL_COLUMNS)


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
            "Run NN_ExpEq_columns_only_v5.py first to generate them."
        )

    checkpoint_loader = getattr(BASE, "load_torch_checkpoint", None)
    members: List[Dict[str, Any]] = []
    skipped_paths: List[str] = []
    for path in model_paths:
        if callable(checkpoint_loader):
            ckpt = checkpoint_loader(path, map_location=BASE.device)
        else:
            ckpt = torch.load(path, map_location=BASE.device)
        model_builder = getattr(BASE, "build_member_model_from_checkpoint", None)
        try:
            if callable(model_builder):
                model = model_builder(ckpt)
            else:
                model = BASE.PairCurveNet(**checkpoint_model_kwargs(ckpt)).to(BASE.device)
                model.load_state_dict(ckpt["state_dict"])
        except RuntimeError as exc:
            message = str(exc)
            if "Missing key(s) in state_dict" in message or "Unexpected key(s) in state_dict" in message:
                skipped_paths.append(path)
                continue
            raise RuntimeError(
                f"Checkpoint architecture mismatch for {path}. "
                "Re-run NN_ExpEq_columns_only_v5.py to regenerate the saved member models with the current model structure."
            ) from exc
        model.eval()
        members.append(
            {
                "path": path,
                "name": os.path.basename(path),
                "model": model,
                "checkpoint": ckpt,
                "cum_scale": float(ckpt["cum_scale"]),
                "lix_scale": float(ckpt.get("lix_scale", 1.0)),
                "irrigation_scale": float(ckpt.get("irrigation_scale", 1.0)),
            }
        )
    if len(members) == 0:
        skipped_text = "\n".join(skipped_paths) if skipped_paths else model_dir
        raise RuntimeError(
            "No compatible saved member checkpoints were found for the current model structure. "
            "Re-run NN_ExpEq_columns_only_v5.py to regenerate them.\n"
            f"Skipped:\n{skipped_text}"
        )
    return members, model_dir, skipped_paths


def impute_static_with_checkpoint(static_raw: np.ndarray, checkpoint: Dict[str, Any]) -> np.ndarray:
    x = np.asarray(static_raw, dtype=float).copy()
    imp = np.asarray(checkpoint["imputer_statistics"], dtype=float)
    missing = ~np.isfinite(x)
    x[missing] = imp[missing]
    return x


def scale_static_with_checkpoint(static_raw: np.ndarray, checkpoint: Dict[str, Any]) -> np.ndarray:
    x = impute_static_with_checkpoint(static_raw, checkpoint)
    mean = np.asarray(checkpoint["scaler_mean"], dtype=float)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=float)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return (x - mean) / scale


def predict_member_curves(
    member: Dict[str, Any],
    static_raw: np.ndarray,
    input_only_raw: np.ndarray,
    time_days: np.ndarray,
    catalyst_cum: np.ndarray,
    lixiviant_cum: np.ndarray,
    irrigation_rate_l_m2_h: np.ndarray,
    control_time_days: np.ndarray | None = None,
) -> Dict[str, Any]:
    model = member["model"]
    checkpoint = member["checkpoint"]
    cum_scale = max(float(member["cum_scale"]), 1e-6)
    lix_scale = max(float(member["lix_scale"]), 1e-6)
    irrigation_scale = max(float(member["irrigation_scale"]), 1e-6)
    static_imputed_raw = impute_static_with_checkpoint(static_raw, checkpoint)
    static_scaled = scale_static_with_checkpoint(static_raw, checkpoint)

    same_plot_grid = False
    if control_time_days is not None:
        same_plot_grid = (
            np.asarray(control_time_days, dtype=float).shape == np.asarray(time_days, dtype=float).shape
            and np.allclose(np.asarray(control_time_days, dtype=float), np.asarray(time_days, dtype=float))
        )

    with torch.no_grad():
        x = torch.tensor(static_scaled, dtype=torch.float32, device=BASE.device).unsqueeze(0)
        x_raw = torch.tensor(static_imputed_raw, dtype=torch.float32, device=BASE.device).unsqueeze(0)
        x_input_only = torch.tensor(np.asarray(input_only_raw, dtype=float), dtype=torch.float32, device=BASE.device).unsqueeze(0)
        t = torch.tensor(np.asarray(time_days, dtype=float), dtype=torch.float32, device=BASE.device)
        c = torch.tensor(np.asarray(catalyst_cum, dtype=float) / cum_scale, dtype=torch.float32, device=BASE.device)
        l = torch.tensor(np.asarray(lixiviant_cum, dtype=float) / lix_scale, dtype=torch.float32, device=BASE.device)
        irr = torch.tensor(
            np.asarray(irrigation_rate_l_m2_h, dtype=float) / irrigation_scale,
            dtype=torch.float32,
            device=BASE.device,
        )
        p_ctrl, p_cat, tau, temp, kappa, aging_strength, latent = model.predict_params(x, x_raw, x_input_only)
        pred_ctrl, pred_cat, states = model.curves_given_params(
            p_ctrl,
            p_cat,
            t,
            c,
            l,
            irr,
            tau,
            temp,
            kappa,
            aging_strength,
            latent_params=latent,
            return_states=True,
        )

        pred_ctrl_plot = None
        if control_time_days is not None:
            t_ctrl_plot = torch.tensor(np.asarray(control_time_days, dtype=float), dtype=torch.float32, device=BASE.device)
            c_ctrl_plot = torch.zeros_like(t_ctrl_plot)
            if same_plot_grid:
                l_ctrl_plot = l
                irr_ctrl_plot = irr
            else:
                l_ctrl_plot = torch.tensor(
                    np.interp(
                        np.asarray(control_time_days, dtype=float),
                        np.asarray(time_days, dtype=float),
                        np.asarray(lixiviant_cum, dtype=float),
                        left=float(np.asarray(lixiviant_cum, dtype=float)[0]),
                        right=float(np.asarray(lixiviant_cum, dtype=float)[-1]),
                    )
                    / lix_scale,
                    dtype=torch.float32,
                    device=BASE.device,
                )
                irr_ctrl_plot = torch.tensor(
                    np.interp(
                        np.asarray(control_time_days, dtype=float),
                        np.asarray(time_days, dtype=float),
                        np.asarray(irrigation_rate_l_m2_h, dtype=float),
                        left=float(np.asarray(irrigation_rate_l_m2_h, dtype=float)[0]),
                        right=float(np.asarray(irrigation_rate_l_m2_h, dtype=float)[-1]),
                    )
                    / irrigation_scale,
                    dtype=torch.float32,
                    device=BASE.device,
                )
            pred_ctrl_plot, _ = model.curves_given_params(
                p_ctrl,
                p_cat,
                t_ctrl_plot,
                c_ctrl_plot,
                l_ctrl_plot,
                irr_ctrl_plot,
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
        "tau_days": float(states["effective_tau_days"].squeeze().detach().cpu().item()),
        "temp_days": float(temp.squeeze().detach().cpu().item()),
        "kappa": float(kappa.squeeze().detach().cpu().item()),
        "aging_strength": float(aging_strength.squeeze().detach().cpu().item()),
    }
    if pred_ctrl_plot is not None:
        out["control_pred_plot"] = pred_ctrl_plot.detach().cpu().numpy()
    return out


def build_model_input_arrays(static_values: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    static_raw = np.asarray([float(static_values[col]) for col in BASE.STATIC_PREDICTOR_COLUMNS], dtype=float)
    input_only_raw = np.asarray(
        [float(static_values.get(col, np.nan)) for col in getattr(BASE, "INPUT_ONLY_COLUMNS", [])],
        dtype=float,
    )
    return static_raw, input_only_raw


def geometry_predictor_values(static_values: Dict[str, float]) -> Dict[str, float]:
    static_raw, input_only_raw = build_model_input_arrays(static_values)
    return BASE.derive_internal_geometry_predictors(static_raw, input_only_raw)


def feed_mass_bounds_from_geometry(
    ui_values: Dict[str, float],
    input_specs: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    feed_spec = input_specs.get(BASE.FEED_MASS_COL, {"min": 0.0, "max": 1.0, "default": 0.0})
    height_spec = input_specs["column_height_m"]
    diameter_spec = input_specs["column_inner_diameter_m"]
    height = float(
        np.clip(
            value_or_default(ui_values.get("column_height_m"), float(height_spec["default"])),
            float(height_spec["min"]),
            float(height_spec["max"]),
        )
    )
    diameter = float(
        np.clip(
            value_or_default(ui_values.get("column_inner_diameter_m"), float(diameter_spec["default"])),
            float(diameter_spec["min"]),
            float(diameter_spec["max"]),
        )
    )
    area_m2 = BASE.column_cross_section_area_m2(diameter)
    volume_m3 = (
        area_m2 * height
        if np.isfinite(area_m2) and area_m2 > 0.0 and np.isfinite(height) and height > 0.0
        else np.nan
    )
    if not np.isfinite(volume_m3) or volume_m3 <= 0.0:
        return {
            "min": float(feed_spec["min"]),
            "max": float(feed_spec["max"]),
            "default": float(feed_spec["default"]),
            "volume_m3": np.nan,
        }
    min_mass = GEOMETRY_BULK_DENSITY_MIN_T_M3 * volume_m3 * 1000.0
    max_mass = GEOMETRY_BULK_DENSITY_MAX_T_M3 * volume_m3 * 1000.0
    default_mass = np.clip(float(feed_spec["default"]), min_mass, max_mass)
    return {
        "min": float(max(0.0, min_mass)),
        "max": float(max(max_mass, min_mass + 1e-6)),
        "default": float(default_mass),
        "volume_m3": float(volume_m3),
    }


def rosin_rammler_diameter_m(
    passing_fraction: np.ndarray,
    p80_m: float,
    shape_n: float = ROSIN_RAMMLER_SHAPE_N,
) -> np.ndarray:
    q = np.clip(np.asarray(passing_fraction, dtype=float), 1e-6, 1.0 - 1e-6)
    n = float(max(shape_n, 1e-6))
    if not np.isfinite(p80_m) or p80_m <= 0.0:
        return np.full(q.shape, np.nan, dtype=float)
    characteristic_size_m = float(p80_m) / ((-np.log(0.2)) ** (1.0 / n))
    return characteristic_size_m * np.power(-np.log(1.0 - q), 1.0 / n)


def rosin_rammler_bins(
    p80_m: float,
    shape_n: float = ROSIN_RAMMLER_SHAPE_N,
) -> Tuple[np.ndarray, np.ndarray]:
    q_edges = np.linspace(ROSIN_RAMMLER_Q_MIN, ROSIN_RAMMLER_Q_MAX, ROSIN_RAMMLER_BIN_COUNT + 1, dtype=float)
    weights = np.diff(q_edges)
    weights = weights / float(np.sum(weights))
    q_mid = 0.5 * (q_edges[:-1] + q_edges[1:])
    diameters_m = rosin_rammler_diameter_m(q_mid, p80_m, shape_n=shape_n)
    return diameters_m, weights


def packing_fraction_from_sizes(
    column_inner_diameter_m: float,
    particle_diameters_m: np.ndarray,
    weights: np.ndarray,
) -> float:
    if (
        not np.isfinite(column_inner_diameter_m)
        or column_inner_diameter_m <= 0.0
        or particle_diameters_m.size == 0
        or weights.size == 0
    ):
        return np.nan
    diameter_ratios = float(column_inner_diameter_m) / np.clip(np.asarray(particle_diameters_m, dtype=float), 1e-9, None)
    wall_factor = np.clip(1.0 - ROSIN_RAMMLER_WALL_ALPHA / np.clip(diameter_ratios, 1e-9, None), 0.0, None)
    local_packing_fraction = ROSIN_RAMMLER_MAX_PACKING_FRACTION * np.power(wall_factor, ROSIN_RAMMLER_WALL_EXPONENT)
    return float(np.clip(np.sum(np.asarray(weights, dtype=float) * local_packing_fraction), 0.0, ROSIN_RAMMLER_MAX_PACKING_FRACTION))


def apparent_density_from_sizes(
    column_inner_diameter_m: float,
    column_height_m: float,
    p80_m: float,
    particle_diameters_m: np.ndarray,
    weights: np.ndarray,
) -> float:
    packing_fraction = packing_fraction_from_sizes(column_inner_diameter_m, particle_diameters_m, weights)
    if not np.isfinite(packing_fraction):
        return np.nan
    if not np.isfinite(column_height_m) or column_height_m <= 0.0 or not np.isfinite(p80_m) or p80_m <= 0.0:
        return np.nan
    apparent_density_t_m3 = (
        CALIBRATED_PACKING_DENSITY_INTERCEPT_T_M3
        + CALIBRATED_PACKING_DENSITY_PACKING_COEFF * packing_fraction
        + CALIBRATED_PACKING_DENSITY_LOG_HEIGHT_COEFF * np.log(max(float(column_height_m), 1e-9))
        + CALIBRATED_PACKING_DENSITY_P80_COEFF_M * float(p80_m)
    )
    return float(max(apparent_density_t_m3, 0.0))


def solve_diameter_for_apparent_density(
    particle_diameters_m: np.ndarray,
    weights: np.ndarray,
    diameter_lower_m: float,
    diameter_upper_m: float,
    column_height_m: float,
    p80_m: float,
    target_density_t_m3: float,
    iterations: int = 50,
) -> float:
    lower = float(diameter_lower_m)
    upper = float(diameter_upper_m)
    target = float(target_density_t_m3)
    density_lower = apparent_density_from_sizes(lower, column_height_m, p80_m, particle_diameters_m, weights)
    density_upper = apparent_density_from_sizes(upper, column_height_m, p80_m, particle_diameters_m, weights)
    if (
        not np.isfinite(density_lower)
        or not np.isfinite(density_upper)
        or target < density_lower - 1e-9
        or target > density_upper + 1e-9
    ):
        return np.nan
    lo = lower
    hi = upper
    for _ in range(max(8, int(iterations))):
        mid = 0.5 * (lo + hi)
        density_mid = apparent_density_from_sizes(mid, column_height_m, p80_m, particle_diameters_m, weights)
        if not np.isfinite(density_mid):
            return np.nan
        if density_mid < target:
            lo = mid
        else:
            hi = mid
    return float(hi)


def estimate_internal_column_loading(
    ui_values: Dict[str, float],
    input_specs: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    bounds = feed_mass_bounds_from_geometry(ui_values, input_specs)
    volume_m3 = float(bounds.get("volume_m3", np.nan))
    height_m = float(
        value_or_default(
            ui_values.get("column_height_m"),
            float(input_specs.get("column_height_m", {"default": np.nan})["default"]),
        )
    )
    diameter_m = float(
        value_or_default(
            ui_values.get("column_inner_diameter_m"),
            float(input_specs.get("column_inner_diameter_m", {"default": np.nan})["default"]),
        )
    )
    p80_in = float(
        value_or_default(
            ui_values.get("material_size_p80_in"),
            float(input_specs.get("material_size_p80_in", {"default": np.nan})["default"]),
        )
    )
    p80_m = float(p80_in) * 0.0254 if np.isfinite(p80_in) and p80_in > 0.0 else np.nan

    if (
        not np.isfinite(volume_m3)
        or volume_m3 <= 0.0
        or not np.isfinite(height_m)
        or height_m <= 0.0
        or not np.isfinite(diameter_m)
        or diameter_m <= 0.0
        or not np.isfinite(p80_m)
        or p80_m <= 0.0
    ):
        midpoint_density = 0.5 * (GEOMETRY_BULK_DENSITY_MIN_T_M3 + GEOMETRY_BULK_DENSITY_MAX_T_M3)
        return {
            "volume_m3": float(volume_m3),
            "p80_m": float(p80_m),
            "packing_score": np.nan,
            "packing_fraction": np.nan,
            "apparent_density_t_m3": float(midpoint_density),
            "estimated_particle_count": np.nan,
            "shape_n": float(ROSIN_RAMMLER_SHAPE_N),
            "used_fallback": True,
        }

    diameters_m, weights = rosin_rammler_bins(p80_m, shape_n=ROSIN_RAMMLER_SHAPE_N)
    packing_fraction = packing_fraction_from_sizes(diameter_m, diameters_m, weights)
    apparent_density_t_m3 = apparent_density_from_sizes(diameter_m, height_m, p80_m, diameters_m, weights)
    packing_fraction_range = ROSIN_RAMMLER_MAX_PACKING_FRACTION
    packing_score = (
        float(np.clip(packing_fraction / packing_fraction_range, 0.0, 1.0))
        if np.isfinite(packing_fraction) and packing_fraction_range > 0.0
        else np.nan
    )
    sphere_volumes_m3 = (np.pi / 6.0) * np.power(np.clip(diameters_m, 1e-9, None), 3.0)
    effective_solid_fraction = (
        float(np.clip(apparent_density_t_m3 / ORE_SOLID_DENSITY_T_M3, 0.0, 1.0))
        if np.isfinite(apparent_density_t_m3)
        else np.nan
    )
    solid_volume_m3 = effective_solid_fraction * volume_m3
    estimated_particle_count = float(solid_volume_m3 * np.sum(weights / sphere_volumes_m3))
    return {
        "volume_m3": float(volume_m3),
        "p80_m": float(p80_m),
        "packing_score": float(packing_score),
        "packing_fraction": float(packing_fraction),
        "apparent_density_t_m3": float(apparent_density_t_m3),
        "estimated_particle_count": float(estimated_particle_count),
        "shape_n": float(ROSIN_RAMMLER_SHAPE_N),
        "used_fallback": False,
    }


def apply_geometry_input_limits(
    ui_values: Dict[str, float],
    input_specs: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    out = dict(ui_values)
    height_spec = input_specs["column_height_m"]
    diameter_spec = input_specs["column_inner_diameter_m"]
    original_height = float(
        np.clip(
            value_or_default(out.get("column_height_m"), float(height_spec["default"])),
            float(height_spec["min"]),
            float(height_spec["max"]),
        )
    )
    original_diameter = float(
        np.clip(
            value_or_default(out.get("column_inner_diameter_m"), float(diameter_spec["default"])),
            float(diameter_spec["min"]),
            float(diameter_spec["max"]),
        )
    )
    height = float(original_height)
    diameter = float(original_diameter)
    material_size_p80_in = float(
        value_or_default(
            out.get("material_size_p80_in"),
            float(input_specs.get("material_size_p80_in", {"default": np.nan})["default"]),
        )
    )

    height_min_spec = float(height_spec["min"])
    height_max_spec = float(height_spec["max"])
    diameter_min_spec = float(diameter_spec["min"])
    diameter_max_spec = float(diameter_spec["max"])
    height = float(np.clip(height, height_min_spec, height_max_spec))
    diameter = float(np.clip(diameter, diameter_min_spec, diameter_max_spec))

    required_diameter = np.nan
    ratio_adjusted = False
    if np.isfinite(material_size_p80_in) and material_size_p80_in > 0.0:
        required_diameter = float(material_size_p80_in) * 0.0254 / GEOMETRY_RATIO_MAX
        min_ratio_diameter = max(diameter_min_spec, min(required_diameter, diameter_max_spec))
        ratio_adjusted = diameter < min_ratio_diameter - 1e-9
        diameter = float(np.clip(diameter, min_ratio_diameter, diameter_max_spec))

    out["column_height_m"] = float(height)
    out["column_inner_diameter_m"] = float(diameter)

    density_adjusted = False
    density_ok = True
    apparent_density_estimate_t_m3 = np.nan
    if np.isfinite(material_size_p80_in) and material_size_p80_in > 0.0:
        p80_m = float(material_size_p80_in) * 0.0254
        particle_diameters_m, particle_weights = rosin_rammler_bins(p80_m, shape_n=ROSIN_RAMMLER_SHAPE_N)
        apparent_density_estimate_t_m3 = apparent_density_from_sizes(
            diameter,
            height,
            p80_m,
            particle_diameters_m,
            particle_weights,
        )
        density_lower_bound_d = max(diameter_min_spec, float(required_diameter) if np.isfinite(required_diameter) else diameter_min_spec)

        if np.isfinite(apparent_density_estimate_t_m3) and apparent_density_estimate_t_m3 > GEOMETRY_BULK_DENSITY_MAX_T_M3 + GEOMETRY_TOL:
            density_at_lower = apparent_density_from_sizes(
                density_lower_bound_d,
                height,
                p80_m,
                particle_diameters_m,
                particle_weights,
            )
            if np.isfinite(density_at_lower) and density_at_lower <= GEOMETRY_BULK_DENSITY_MAX_T_M3 + GEOMETRY_TOL:
                solved_diameter = solve_diameter_for_apparent_density(
                    particle_diameters_m=particle_diameters_m,
                    weights=particle_weights,
                    diameter_lower_m=density_lower_bound_d,
                    diameter_upper_m=diameter,
                    column_height_m=height,
                    p80_m=p80_m,
                    target_density_t_m3=GEOMETRY_BULK_DENSITY_MAX_T_M3,
                )
                if np.isfinite(solved_diameter):
                    diameter = float(np.clip(solved_diameter, density_lower_bound_d, diameter))
                    density_adjusted = abs(diameter - float(out["column_inner_diameter_m"])) > 1e-9
                    out["column_inner_diameter_m"] = float(diameter)
                    apparent_density_estimate_t_m3 = apparent_density_from_sizes(
                        diameter,
                        height,
                        p80_m,
                        particle_diameters_m,
                        particle_weights,
                    )
            else:
                density_ok = False
        elif np.isfinite(apparent_density_estimate_t_m3) and apparent_density_estimate_t_m3 < GEOMETRY_BULK_DENSITY_MIN_T_M3 - GEOMETRY_TOL:
            density_at_upper = apparent_density_from_sizes(
                diameter_max_spec,
                height,
                p80_m,
                particle_diameters_m,
                particle_weights,
            )
            if np.isfinite(density_at_upper) and density_at_upper >= GEOMETRY_BULK_DENSITY_MIN_T_M3 - GEOMETRY_TOL:
                solved_diameter = solve_diameter_for_apparent_density(
                    particle_diameters_m=particle_diameters_m,
                    weights=particle_weights,
                    diameter_lower_m=diameter,
                    diameter_upper_m=diameter_max_spec,
                    column_height_m=height,
                    p80_m=p80_m,
                    target_density_t_m3=GEOMETRY_BULK_DENSITY_MIN_T_M3,
                )
                if np.isfinite(solved_diameter):
                    diameter = float(np.clip(solved_diameter, diameter, diameter_max_spec))
                    density_adjusted = abs(diameter - float(out["column_inner_diameter_m"])) > 1e-9
                    out["column_inner_diameter_m"] = float(diameter)
                    apparent_density_estimate_t_m3 = apparent_density_from_sizes(
                        diameter,
                        height,
                        p80_m,
                        particle_diameters_m,
                        particle_weights,
                    )
            else:
                density_ok = False

    density_in_band = (
        np.isfinite(apparent_density_estimate_t_m3)
        and GEOMETRY_BULK_DENSITY_MIN_T_M3 - GEOMETRY_TOL
        <= apparent_density_estimate_t_m3
        <= GEOMETRY_BULK_DENSITY_MAX_T_M3 + GEOMETRY_TOL
    )

    geometry_values = BASE.derive_internal_geometry_predictors(
        np.asarray(
            [
                float(
                    value_or_default(
                        out.get(col),
                        float(input_specs.get(col, {"default": np.nan})["default"]),
                    )
                )
                for col in BASE.STATIC_PREDICTOR_COLUMNS
            ],
            dtype=float,
        ),
        np.asarray([float(out.get(col, np.nan)) for col in getattr(BASE, "INPUT_ONLY_COLUMNS", [])], dtype=float),
    )
    ratio = float(geometry_values.get("material_size_to_column_diameter_ratio", np.nan))
    ratio_limit_text = format_ratio_one_to_x(GEOMETRY_RATIO_MAX)
    ratio_ok = (not np.isfinite(ratio)) or ratio <= GEOMETRY_RATIO_MAX + GEOMETRY_TOL
    height_adjusted = abs(height - original_height) > 1e-9
    diameter_adjusted = abs(float(out["column_inner_diameter_m"]) - original_diameter) > 1e-9
    adjusted = height_adjusted or diameter_adjusted
    if not ratio_ok:
        note = (
            "Geometry guardrail hit the current diameter slider bounds before the ratio target "
            f"{ratio_limit_text} could be fully met. "
            "Adjust material size or widen the geometry range."
        )
    elif not density_ok or not density_in_band:
        note = (
            "Geometry guardrail hit the current diameter slider bounds before the Rosin-Rammler implied apparent density "
            f"could be kept inside [{GEOMETRY_BULK_DENSITY_MIN_T_M3:.3f}, {GEOMETRY_BULK_DENSITY_MAX_T_M3:.2f}] t/m3."
        )
    elif ratio_adjusted and density_adjusted:
        note = (
            "Geometry guardrail applied: column diameter was adjusted so material-size / column-diameter ratio stays "
            f"<= {ratio_limit_text} "
            # f"and the Rosin-Rammler implied apparent density stays inside "
            # f"[{GEOMETRY_BULK_DENSITY_MIN_T_M3:.3f}, {GEOMETRY_BULK_DENSITY_MAX_T_M3:.2f}] t/m3."
        )
    elif ratio_adjusted:
        note = (
            "Geometry guardrail applied: column diameter was adjusted so "
            f"material-size / column-diameter ratio stays <= {ratio_limit_text}."
        )
    elif density_adjusted:
        note = (
            "Geometry guardrail applied: column diameter was adjusted so the Rosin-Rammler implied apparent density stays "
            f"inside [{GEOMETRY_BULK_DENSITY_MIN_T_M3:.3f}, {GEOMETRY_BULK_DENSITY_MAX_T_M3:.2f}] t/m3."
        )
    else:
        note = ""
    return out, {
        "adjusted": bool(adjusted),
        "height_adjusted": bool(height_adjusted),
        "diameter_adjusted": bool(diameter_adjusted),
        "ratio_adjusted": bool(ratio_adjusted),
        "density_adjusted": bool(density_adjusted),
        "ratio_ok": bool(ratio_ok),
        "density_ok": bool(density_ok and density_in_band),
        "apparent_density_estimate_t_m3": (
            float(apparent_density_estimate_t_m3) if np.isfinite(apparent_density_estimate_t_m3) else np.nan
        ),
        "note": note,
    }


def resolve_internal_feed_mass(
    ui_values: Dict[str, float],
    input_specs: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    out = dict(ui_values)
    if BASE.FEED_MASS_COL not in input_specs:
        return out, {
            "resolved": False,
            "min_mass_kg": np.nan,
            "max_mass_kg": np.nan,
            "feed_mass_kg": np.nan,
            "reference_density_t_m3": np.nan,
            "estimated_density_t_m3": np.nan,
            "packing_score": np.nan,
            "packing_fraction": np.nan,
            "estimated_particle_count": np.nan,
            "note": "",
        }
    bounds = feed_mass_bounds_from_geometry(out, input_specs)
    min_mass = float(bounds["min"])
    max_mass = float(bounds["max"])
    volume_m3 = float(bounds.get("volume_m3", np.nan))
    loading_estimate = estimate_internal_column_loading(out, input_specs)
    reference_density_t_m3 = float(loading_estimate.get("apparent_density_t_m3", np.nan))
    if np.isfinite(volume_m3) and volume_m3 > 0.0 and np.isfinite(reference_density_t_m3):
        estimated_mass = reference_density_t_m3 * volume_m3 * 1000.0
    else:
        estimated_mass = float(input_specs[BASE.FEED_MASS_COL]["default"])
    estimated_mass = float(max(0.0, estimated_mass))
    out[BASE.FEED_MASS_COL] = float(estimated_mass)
    estimated_density_t_m3 = (
        (estimated_mass / 1000.0) / volume_m3
        if np.isfinite(volume_m3) and volume_m3 > 0.0 and np.isfinite(estimated_mass) and estimated_mass > 0.0
        else np.nan
    )
    if np.isfinite(volume_m3) and volume_m3 > 0.0:
        p80_in = float(
            value_or_default(
                out.get("material_size_p80_in"),
                float(input_specs.get("material_size_p80_in", {"default": np.nan})["default"]),
            )
        )
        if loading_estimate.get("used_fallback", False):
            note = (
                "Feed mass is hidden from the UI and falls back to the midpoint apparent-density estimate until "
                "column volume and material size are both valid. "
                f"\nThe current hidden model input is feed_mass_kg={int(estimated_mass)}kg."
            )
        else:
            note = (
                "Feed mass is hidden from the UI and auto-estimated from column volume plus a calibrated "
                f"Rosin-Rammler sphere-loading approximation using P80={p80_in:.1f} in and shape n={loading_estimate['shape_n']:.1f}. "
                f"\nThe resulting hidden model input is feed_mass_kg={int(estimated_mass)}kg."
            )
    else:
        note = (
            "Feed mass is hidden from the UI and falls back to the internal default estimate until "
            f"a valid column height and diameter define the column volume. \nThe current hidden model input is feed_mass_kg={int(estimated_mass)}kg."
        )
    return out, {
        "resolved": True,
        "min_mass_kg": float(min_mass),
        "max_mass_kg": float(max_mass),
        "volume_m3": float(volume_m3),
        "feed_mass_kg": float(estimated_mass),
        "reference_density_t_m3": float(reference_density_t_m3),
        "estimated_density_t_m3": float(estimated_density_t_m3),
        "packing_score": float(loading_estimate.get("packing_score", np.nan)),
        "packing_fraction": float(loading_estimate.get("packing_fraction", np.nan)),
        "estimated_particle_count": float(loading_estimate.get("estimated_particle_count", np.nan)),
        "note": note,
    }


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


def resolve_ui_state(
    predictor_values: Tuple[float, ...],
    input_specs: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    ui_values: Dict[str, float] = {}
    for column, raw_value in zip(EDITABLE_INPUT_COLUMNS, predictor_values):
        spec = input_specs[column]
        ui_values[column] = float(
            np.clip(
                value_or_default(raw_value, float(spec["default"])),
                float(spec["min"]),
                float(spec["max"]),
            )
        )
    chemistry = chemistry_state(
        ui_values.get(ACID_SOLUBLE_COL),
        ui_values.get(RESIDUAL_CPY_COL),
        input_specs,
    )
    ui_values[ACID_SOLUBLE_COL] = float(chemistry["acid"])
    ui_values[RESIDUAL_CPY_COL] = float(chemistry["residual"])
    ui_values[CYANIDE_SOLUBLE_COL] = float(chemistry["cyanide"])
    return ui_values


def _distribute_with_bounds(
    start_values: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
    target_sum: float,
) -> np.ndarray:
    values = np.clip(np.asarray(start_values, dtype=float), min_values, max_values)
    target = float(np.clip(float(target_sum), float(np.sum(min_values)), float(np.sum(max_values))))
    current = float(np.sum(values))
    if abs(target - current) <= 1e-9:
        return values

    if target > current:
        capacity = np.clip(max_values - values, 0.0, None)
        total_capacity = float(np.sum(capacity))
        if total_capacity > 1e-12:
            values = values + (target - current) * capacity / total_capacity
    else:
        capacity = np.clip(values - min_values, 0.0, None)
        total_capacity = float(np.sum(capacity))
        if total_capacity > 1e-12:
            values = values - (current - target) * capacity / total_capacity

    residual = target - float(np.sum(values))
    if abs(residual) > 1e-8:
        if residual > 0.0:
            capacity = np.clip(max_values - values, 0.0, None)
        else:
            capacity = np.clip(values - min_values, 0.0, None)
        if float(np.max(capacity)) > 1e-12:
            idx = int(np.argmax(capacity))
            values[idx] = values[idx] + residual
            values = np.clip(values, min_values, max_values)
    return values


def _allocate_weighted_delta(
    capacity: np.ndarray,
    amount: float,
    priority_idx: int | None = None,
    priority_share: float = GROUPED_GANGUE_PRIORITY_SHARE,
) -> np.ndarray:
    cap = np.clip(np.asarray(capacity, dtype=float), 0.0, None)
    alloc = np.zeros_like(cap)
    remaining = float(max(0.0, amount))
    if remaining <= 1e-12 or cap.size == 0:
        return alloc

    while remaining > 1e-8:
        active = cap > 1e-12
        if not np.any(active):
            break

        weights = np.zeros_like(cap)
        if priority_idx is not None and 0 <= priority_idx < cap.size and active[priority_idx]:
            weights[priority_idx] = float(np.clip(priority_share, 0.0, 1.0))

        other_idx = np.where(active & (np.arange(cap.size) != int(priority_idx) if priority_idx is not None else active))[0]
        if other_idx.size > 0:
            other_capacity = cap[other_idx]
            other_weight_total = max(0.0, 1.0 - float(np.sum(weights)))
            if float(np.sum(other_capacity)) > 1e-12 and other_weight_total > 1e-12:
                weights[other_idx] = other_weight_total * other_capacity / float(np.sum(other_capacity))

        if float(np.sum(weights[active])) <= 1e-12:
            weights[active] = cap[active] / float(np.sum(cap[active]))
        else:
            weights = weights / float(np.sum(weights[active]))

        proposed = remaining * weights
        taken = np.minimum(proposed, cap)
        taken_sum = float(np.sum(taken))
        if taken_sum <= 1e-12:
            idx = int(np.where(active)[0][np.argmax(cap[active])])
            taken[idx] = min(remaining, float(cap[idx]))
            taken_sum = float(np.sum(taken))
            if taken_sum <= 1e-12:
                break

        alloc = alloc + taken
        cap = cap - taken
        remaining -= taken_sum
    return alloc


def _distribute_with_priority(
    start_values: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
    target_sum: float,
    priority_idx: int | None = None,
    priority_share: float = GROUPED_GANGUE_PRIORITY_SHARE,
) -> np.ndarray:
    values = np.clip(np.asarray(start_values, dtype=float), min_values, max_values)
    target = float(np.clip(float(target_sum), float(np.sum(min_values)), float(np.sum(max_values))))
    current = float(np.sum(values))
    if abs(target - current) <= 1e-9:
        return values
    if priority_idx is None or priority_idx < 0 or priority_idx >= values.size:
        return _distribute_with_bounds(values, min_values, max_values, target)

    if target > current:
        capacity = np.clip(max_values - values, 0.0, None)
        values = values + _allocate_weighted_delta(capacity, target - current, priority_idx, priority_share)
    else:
        capacity = np.clip(values - min_values, 0.0, None)
        values = values - _allocate_weighted_delta(capacity, current - target, priority_idx, priority_share)

    residual = target - float(np.sum(values))
    if abs(residual) > 1e-8:
        values = _distribute_with_bounds(values, min_values, max_values, target)
    return np.clip(values, min_values, max_values)


def resolve_grouped_balance(
    ui_values: Dict[str, float],
    input_specs: Dict[str, Dict[str, float]],
    grouped_model_columns: List[str],
    grouped_inactive_columns: List[str],
    target_total: float = 100.0,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    active_grouped_values = {column: float(ui_values.get(column, input_specs[column]["default"])) for column in grouped_model_columns}
    active_sum = float(sum(active_grouped_values.values()))

    if len(grouped_inactive_columns) == 0:
        total_sum = active_sum
        feasible = abs(total_sum - float(target_total)) <= 1e-6
        return active_grouped_values, {}, {
            "active_sum": active_sum,
            "inactive_sum": 0.0,
            "total_sum": total_sum,
            "target_total": float(target_total),
            "feasible": feasible,
            "lower_sum": 0.0,
            "upper_sum": 0.0,
            "inactive_target": float(target_total) - active_sum,
            "show_warning": not feasible,
            "note": (
                ""
                if feasible
                else f"Grouped mineralogy totals {total_sum:.2f}% and must be adjusted manually to reach {float(target_total):.2f}%."
            ),
        }

    min_values = np.asarray([float(input_specs[column]["min"]) for column in grouped_inactive_columns], dtype=float)
    max_values = np.asarray([float(input_specs[column]["max"]) for column in grouped_inactive_columns], dtype=float)
    default_values = np.asarray([float(input_specs[column]["default"]) for column in grouped_inactive_columns], dtype=float)
    lower_sum = float(np.sum(min_values))
    upper_sum = float(np.sum(max_values))
    inactive_target = float(target_total) - active_sum
    feasible = lower_sum - 1e-6 <= inactive_target <= upper_sum + 1e-6
    gangue_priority_idx = (
        grouped_inactive_columns.index(GROUPED_GANGUE_PRIORITY_COLUMN)
        if GROUPED_GANGUE_PRIORITY_COLUMN in grouped_inactive_columns
        else None
    )
    balanced_values = _distribute_with_priority(
        start_values=default_values,
        min_values=min_values,
        max_values=max_values,
        target_sum=inactive_target,
        priority_idx=gangue_priority_idx,
        priority_share=GROUPED_GANGUE_PRIORITY_SHARE,
    )
    inactive_grouped_values = {
        column: float(value) for column, value in zip(grouped_inactive_columns, balanced_values)
    }
    inactive_sum = float(np.sum(balanced_values))
    total_sum = active_sum + inactive_sum
    if feasible:
        balance_note = ""
    else:
        balance_note = (
            f"Auto-balanced grouped mineralogy hit train min/max bounds and totals {total_sum:.2f}% "
            f"instead of {float(target_total):.2f}%."
        )
    all_grouped_values = dict(active_grouped_values)
    all_grouped_values.update(inactive_grouped_values)
    return all_grouped_values, inactive_grouped_values, {
        "active_sum": active_sum,
        "inactive_sum": inactive_sum,
        "total_sum": total_sum,
        "target_total": float(target_total),
        "feasible": feasible,
        "lower_sum": lower_sum,
        "upper_sum": upper_sum,
        "inactive_target": inactive_target,
        "show_warning": not feasible,
        "note": balance_note,
    }


def resolve_model_static_values(
    ui_values: Dict[str, float],
    input_specs: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, Any], Dict[str, float]]:
    resolved_chemistry = chemistry_state(
        ui_values.get(ACID_SOLUBLE_COL),
        ui_values.get(RESIDUAL_CPY_COL),
        input_specs,
    )
    ui_values = dict(ui_values)
    ui_values, _feed_mass_state = resolve_internal_feed_mass(ui_values, input_specs)
    ui_values[ACID_SOLUBLE_COL] = float(resolved_chemistry["acid"])
    ui_values[RESIDUAL_CPY_COL] = float(resolved_chemistry["residual"])
    ui_values[CYANIDE_SOLUBLE_COL] = float(resolved_chemistry["cyanide"])
    grouped_values, inactive_grouped_values, grouped_balance_state = resolve_grouped_balance(
        ui_values=ui_values,
        input_specs=input_specs,
        grouped_model_columns=[c for c in GROUPED_MODEL_COLUMNS if c in input_specs],
        grouped_inactive_columns=[c for c in GROUPED_INACTIVE_COLUMNS if c in input_specs],
        target_total=100.0,
    )
    cu_value = float(
        ui_values.get(
            CU_INPUT_COL,
            input_specs.get(CU_INPUT_COL, {"default": 0.0})["default"],
        )
    )
    fe_value = float(
        ui_values.get(
            FE_INPUT_COL,
            input_specs.get(FE_INPUT_COL, {"default": 0.0})["default"],
        )
    )
    acid_value = float(
        ui_values.get(
            ACID_SOLUBLE_COL,
            input_specs.get(ACID_SOLUBLE_COL, {"default": 0.0})["default"],
        )
    )
    cyanide_value = float(
        ui_values.get(
            CYANIDE_SOLUBLE_COL,
            input_specs.get(CYANIDE_SOLUBLE_COL, {"default": 0.0})["default"],
        )
    )
    residual_value = float(
        ui_values.get(
            RESIDUAL_CPY_COL,
            input_specs.get(RESIDUAL_CPY_COL, {"default": 0.0})["default"],
        )
    )

    derived_values = {
        FE_CU_RATIO_COL: safe_ratio(fe_value, cu_value),
        CU_FE_RATIO_COL: safe_ratio(cu_value, fe_value),
        PRIMARY_SULFIDES_EQUIV_COL: cu_value * residual_value / 100.0,
        SECONDARY_SULFIDES_EQUIV_COL: cu_value * cyanide_value / 100.0,
        SULFIDES_EQUIV_COL: cu_value * (residual_value + cyanide_value) / 100.0,
        OXIDES_EQUIV_COL: cu_value * acid_value / 100.0,
    }

    static_values: Dict[str, float] = {}
    for column in BASE.STATIC_PREDICTOR_COLUMNS:
        if column in ui_values:
            static_values[column] = float(ui_values[column])
        elif column in grouped_values:
            static_values[column] = float(grouped_values[column])
        elif column in derived_values:
            static_values[column] = float(derived_values[column])
        elif column == FE_HEAD_COL:
            static_values[column] = fe_value
        elif column == CU_HEAD_COL:
            static_values[column] = cu_value
        else:
            static_values[column] = float(input_specs.get(column, {"default": 0.0})["default"])
    for column in getattr(BASE, "INPUT_ONLY_COLUMNS", []):
        if column in ui_values:
            static_values[column] = float(ui_values[column])
        else:
            static_values[column] = float(input_specs.get(column, {"default": 0.0})["default"])
    return static_values, derived_values, inactive_grouped_values, grouped_balance_state, resolved_chemistry


def predict_ensemble_curves(
    members: List[Dict[str, Any]],
    static_values: Dict[str, float],
    weekly_catalyst_gt_week: float,
    catalyst_addition_start_day: float,
    irrigation_rate_l_m2_h: float,
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
    static_raw, input_only_raw = build_model_input_arrays(static_values)
    input_only_idx = {name: idx for idx, name in enumerate(BASE.INPUT_ONLY_COLUMNS)}
    feed_mass_kg = (
        float(input_only_raw[input_only_idx[BASE.FEED_MASS_COL]])
        if BASE.FEED_MASS_COL in input_only_idx
        else np.nan
    )
    column_inner_diameter_m = float(static_values.get("column_inner_diameter_m", np.nan))
    lixiviant_cum = BASE.build_cumulative_lixiviant_from_irrigation_rate(
        time_days=time_days,
        irrigation_rate_l_m2_h=float(max(0.0, irrigation_rate_l_m2_h)),
        feed_mass_kg=feed_mass_kg,
        column_inner_diameter_m=column_inner_diameter_m,
    )
    irrigation_profile = BASE.convert_lixiviant_cum_to_irrigation_rate_l_m2_h(
        time_days=time_days,
        cumulative_lixiviant_m3_t=lixiviant_cum,
        feed_mass_kg=feed_mass_kg,
        column_inner_diameter_m=column_inner_diameter_m,
    )

    member_preds = []
    for member in members:
        member_preds.append(
            predict_member_curves(
                member=member,
                static_raw=static_raw,
                input_only_raw=input_only_raw,
                time_days=time_days,
                catalyst_cum=catalyst_cum,
                lixiviant_cum=lixiviant_cum,
                irrigation_rate_l_m2_h=irrigation_profile,
                control_time_days=time_days,
            )
        )

    ctrl_stack = np.vstack([p["control_pred"] for p in member_preds])
    cat_stack = np.vstack([p["catalyzed_pred"] for p in member_preds])
    ctrl_p10_raw = np.percentile(ctrl_stack, ci_state["pi_low"], axis=0)
    ctrl_p90_raw = np.percentile(ctrl_stack, ci_state["pi_high"], axis=0)
    cat_p10_raw = np.percentile(cat_stack, ci_state["pi_low"], axis=0)
    cat_p90_raw = np.percentile(cat_stack, ci_state["pi_high"], axis=0)
    interval_smoothing_days = float(BASE.CONFIG.get("ensemble_interval_smoothing_days", 0.0))
    smooth_bounds_fn = getattr(BASE, "smooth_predictive_interval_bounds", None)
    if callable(smooth_bounds_fn):
        ctrl_p10, ctrl_p90 = smooth_bounds_fn(
            time_days=time_days,
            mean_curve=np.mean(ctrl_stack, axis=0),
            low_curve=ctrl_p10_raw,
            high_curve=ctrl_p90_raw,
            smoothing_days=interval_smoothing_days,
        )
        cat_p10, cat_p90 = smooth_bounds_fn(
            time_days=time_days,
            mean_curve=np.mean(cat_stack, axis=0),
            low_curve=cat_p10_raw,
            high_curve=cat_p90_raw,
            smoothing_days=interval_smoothing_days,
        )
    else:
        ctrl_p10, ctrl_p90 = ctrl_p10_raw, ctrl_p90_raw
        cat_p10, cat_p90 = cat_p10_raw, cat_p90_raw
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
        "cumulative_lixiviant_m3_t": lixiviant_cum,
        "irrigation_rate_l_m2_h": irrigation_profile,
        "irrigation_rate_input_l_m2_h": float(max(0.0, irrigation_rate_l_m2_h)),
        "control_pred_mean": np.mean(ctrl_stack, axis=0),
        "control_pred_p10": ctrl_p10,
        "control_pred_p90": ctrl_p90,
        "catalyzed_pred_mean": np.mean(cat_stack, axis=0),
        "catalyzed_pred_p10": cat_p10,
        "catalyzed_pred_p90": cat_p90,
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
        "<br>Uplift = %{customdata[2]:.1f}%"
        "<br>Uplift Normalized = %{customdata[3]}"
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

    weekly_text = (
        f"Catalyst used: {pred['weekly_catalyst_gt_week']:.2f} g/t/week"
        f"<br>Catalyst addition start day: {pred['catalyst_addition_start_day']:.0f}"
        f"<br>Irrigation rate: {pred['irrigation_rate_input_l_m2_h']:.2f} L/h/m2"
        f"<br>Confidence band: {band_label}"
        f"<br>Assumption: constant catalyst addition from selected start day"
        f"<br>Assumption: constant irrigation over the plotted horizon"
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
        title="Interactive Ensemble Prediction (v5)",
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


def build_prediction_export_days(max_day: float, export_step_days: float = 7.0) -> np.ndarray:
    export_step_days = float(max(export_step_days, 1.0))
    max_day = float(np.clip(max_day, 1.0, 2500.0))
    export_days = np.arange(0.0, max_day + export_step_days, export_step_days, dtype=float)
    export_days = export_days[export_days <= max_day + 1e-9]
    if export_days.size == 0 or abs(export_days[-1] - max_day) > 1e-9:
        export_days = np.append(export_days, max_day)
    return np.unique(export_days.astype(float))


def build_export_dataframes(
    pred: Dict[str, Any],
    ui_values: Dict[str, float],
    static_values: Dict[str, float],
    derived_values: Dict[str, float],
    geometry_values: Dict[str, float],
    inactive_grouped_values: Dict[str, float],
    grouped_balance_state: Dict[str, Any],
    resolved_chemistry: Dict[str, float],
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
    lixiviant_cumulative = np.interp(
        export_days,
        time_days,
        np.asarray(pred["cumulative_lixiviant_m3_t"], dtype=float),
    )
    irrigation_rate = np.interp(
        export_days,
        time_days,
        np.asarray(pred["irrigation_rate_l_m2_h"], dtype=float),
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
            "label": "Catalyst used (g/t/week)",
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
            "parameter": "irrigation_rate_l_m2_h",
            "label": "Irrigation rate (L/h/m2)",
            "value": float(pred["irrigation_rate_input_l_m2_h"]),
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

    for column in EDITABLE_INPUT_COLUMNS:
        inputs_rows.append(
            {
                "section": "ui_inputs",
                "parameter": column,
                "label": display_label(column),
                "value": float(ui_values[column]),
            }
        )

    chemistry_export_map = {
        ACID_SOLUBLE_COL: "acid",
        CYANIDE_SOLUBLE_COL: "cyanide",
        RESIDUAL_CPY_COL: "residual",
    }
    for column, chemistry_key in chemistry_export_map.items():
        inputs_rows.append(
            {
                "section": "resolved_chemistry",
                "parameter": column,
                "label": display_label(column),
                "value": float(resolved_chemistry[chemistry_key]),
            }
        )

    derived_export_columns = list(EQUIVALENT_COLUMNS)
    if needs_fe_input():
        derived_export_columns = [FE_CU_RATIO_COL, CU_FE_RATIO_COL, *derived_export_columns]
    for column in derived_export_columns:
        inputs_rows.append(
            {
                "section": "derived_internal_values",
                "parameter": column,
                "label": display_label(column),
                "value": float(derived_values[column]),
            }
        )

    for column in BASE.STATIC_PREDICTOR_COLUMNS:
        inputs_rows.append(
            {
                "section": "model_inputs",
                "parameter": column,
                "label": display_label(column),
                "value": float(static_values[column]),
            }
        )
    for column in getattr(BASE, "INPUT_ONLY_COLUMNS", []):
        inputs_rows.append(
            {
                "section": "input_only_support_values",
                "parameter": column,
                "label": display_label(column),
                "value": float(static_values[column]),
            }
        )
    for column in ["column_volume_m3", "apparent_bulk_density_t_m3", "material_size_to_column_diameter_ratio"]:
        inputs_rows.append(
            {
                "section": "derived_geometry_values",
                "parameter": column,
                "label": display_label(column),
                "value": float(geometry_values.get(column, np.nan)),
            }
        )

    inputs_rows.append(
        {
            "section": "grouped_balance",
            "parameter": "grouped_total_pct",
            "label": "Grouped Total (%)",
            "value": float(grouped_balance_state["total_sum"]),
        }
    )

    for column, value in inactive_grouped_values.items():
        inputs_rows.append(
            {
                "section": "inactive_grouped_balanced_values",
                "parameter": column,
                "label": display_label(column),
                "value": float(value),
            }
        )

    inputs_df = pd.DataFrame(inputs_rows)
    predictions_df = pd.DataFrame(
        {
            "leach_duration_days": export_days,
            "cumulative_catalyst_addition_kg_t": catalyst_cumulative,
            "cumulative_lixiviant_m3_t": lixiviant_cumulative,
            "irrigation_rate_l_m2_h": irrigation_rate,
            "control_pred_mean_cu_recovery_pct": control_mean,
            "catalyzed_pred_mean_cu_recovery_pct": catalyzed_mean,
            "delta_cu_recovery_pct": delta,
            "delta_normalized_pct": delta_normalized_pct,
        }
    )
    return inputs_df, predictions_df


def build_prediction_export_bytes(
    pred: Dict[str, Any],
    ui_values: Dict[str, float],
    static_values: Dict[str, float],
    derived_values: Dict[str, float],
    geometry_values: Dict[str, float],
    inactive_grouped_values: Dict[str, float],
    grouped_balance_state: Dict[str, Any],
    resolved_chemistry: Dict[str, float],
    max_day: float,
) -> bytes:
    inputs_df, predictions_df = build_export_dataframes(
        pred=pred,
        ui_values=ui_values,
        static_values=static_values,
        derived_values=derived_values,
        geometry_values=geometry_values,
        inactive_grouped_values=inactive_grouped_values,
        grouped_balance_state=grouped_balance_state,
        resolved_chemistry=resolved_chemistry,
        max_day=max_day,
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine=resolve_excel_writer_engine()) as writer:
        inputs_df.to_excel(writer, sheet_name="inputs", index=False)
        predictions_df.to_excel(writer, sheet_name="predictions_7d", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def build_prediction_summary(
    pred: Dict[str, Any],
    ui_values: Dict[str, float],
    derived_values: Dict[str, float],
    geometry_values: Dict[str, float],
    geometry_guardrail_state: Dict[str, Any],
    feed_mass_state: Dict[str, Any],
    grouped_balance_state: Dict[str, Any],
    resolved_chemistry: Dict[str, float],
) -> html.Div:
    apparent_density = float(geometry_values.get("apparent_bulk_density_t_m3", np.nan))
    apparent_density_text = f"{apparent_density:.2f} t/m3" if np.isfinite(apparent_density) else "n/a"
    apparent_density_status = (
        (
            f"(inside expected {GEOMETRY_BULK_DENSITY_MIN_T_M3:.3f}-{GEOMETRY_BULK_DENSITY_MAX_T_M3:.2f} t/m3)"
            if GEOMETRY_BULK_DENSITY_MIN_T_M3 - GEOMETRY_TOL
            <= apparent_density
            <= GEOMETRY_BULK_DENSITY_MAX_T_M3 + GEOMETRY_TOL
            else f"(outside expected {GEOMETRY_BULK_DENSITY_MIN_T_M3:.3f}-{GEOMETRY_BULK_DENSITY_MAX_T_M3:.2f} t/m3)"
        )
        if np.isfinite(apparent_density)
        else f"(expected {GEOMETRY_BULK_DENSITY_MIN_T_M3:.3f}-{GEOMETRY_BULK_DENSITY_MAX_T_M3:.2f} t/m3)"
    )
    size_ratio_value = float(geometry_values.get("material_size_to_column_diameter_ratio", np.nan))
    size_ratio_text = format_ratio_one_to_x(size_ratio_value)
    size_ratio_limit_text = format_ratio_one_to_x(GEOMETRY_RATIO_MAX)
    size_ratio_status = (
        (
            f"(inside expected >= {size_ratio_limit_text})"
            if size_ratio_value <= GEOMETRY_RATIO_MAX + GEOMETRY_TOL
            else f"(outside expected >= {size_ratio_limit_text})"
        )
        if np.isfinite(size_ratio_value)
        else f"(expected >= {size_ratio_limit_text})"
    )
    summary_lines: List[Any] = [
        html.Div(
            f"Tau mean: {int(pred['tau_days_mean'])} days | "
            f"Temp mean: {int(pred['temp_days_mean'])} days | "
            f"Kappa mean: {pred['kappa_mean']:.2f} | "
            f"Aging strength mean: {pred['aging_strength_mean']:.2f}"
        ),
        html.Div(
            "Parameter meanings: Tau is the midpoint day where catalyzed uplift starts to turn on, "
            "after accounting for the first catalyst-addition day and now also allowing taller columns "
            "to delay that uplift through a learned height contribution; "
            "Temp is the width or softness of that turn-on; Kappa is the catalyst-response rate "
            "showing how fast added catalyst translates into uplift; Aging strength is the decay "
            "strength showing how fast older catalyst additions lose effectiveness over time."
        ),
        html.Div(
            f"Catalyzed curve assumption: constant catalyst addition starting at day "
            f"{pred['catalyst_addition_start_day']:.0f}, using the selected weekly rate "
            "converted internally to cumulative kg/t."
        ),
        html.Div(
            f"Irrigation assumption: constant {pred['irrigation_rate_input_l_m2_h']:.2f} L/h/m2 "
            "converted internally to cumulative lixiviant (m3/t) using feed mass and column diameter."
        ),
        html.Div(
            f"Internal geometry values: Column volume {geometry_values.get('column_volume_m3', np.nan):.2f} m3 | "
            f"Apparent bulk density {apparent_density_text} {apparent_density_status} | "
            f"Size/diameter ratio {size_ratio_text} {size_ratio_status}"
        ),
        html.Div(
            "Only apparent bulk density and material-size to column-diameter ratio are used as geometry predictors; "
            "column height and diameter stay user inputs only."
        ),
        html.Div(str(feed_mass_state.get("note", ""))),
        html.Div(f"Displayed ensemble band: {pred['band_label']}"),
        html.Div(
            f"Resolved chemistry: Acid {resolved_chemistry['acid']:.2f}% | "
            f"Cyanide {resolved_chemistry['cyanide']:.2f}% | "
            f"Residual {resolved_chemistry['residual']:.2f}%"
        ),
        html.Div(
            f"Internal copper equivalents: "
            f"Primary {derived_values[PRIMARY_SULFIDES_EQUIV_COL]:.3f} | "
            f"Secondary {derived_values[SECONDARY_SULFIDES_EQUIV_COL]:.3f} | "
            f"Oxides {derived_values[OXIDES_EQUIV_COL]:.3f}"
        ),
    ]
    if SULFIDES_EQUIV_COL in BASE.STATIC_PREDICTOR_COLUMNS:
        summary_lines.append(
            html.Div(f"Internal total sulfides equivalent: {derived_values[SULFIDES_EQUIV_COL]:.3f}")
        )
    if needs_fe_input():
        ratio_fragments = [f"Fe {ui_values.get(FE_INPUT_COL, 0.0):.2f}%"]
        if FE_CU_RATIO_COL in BASE.STATIC_PREDICTOR_COLUMNS:
            ratio_fragments.append(f"Fe/Cu {derived_values[FE_CU_RATIO_COL]:.3f}")
        if CU_FE_RATIO_COL in BASE.STATIC_PREDICTOR_COLUMNS:
            ratio_fragments.append(f"Cu/Fe {derived_values[CU_FE_RATIO_COL]:.3f}")
        summary_lines.append(html.Div("Internal ratio inputs: " + " | ".join(ratio_fragments)))
    grouped_balance_note = str(grouped_balance_state.get("note", "")).strip()
    if grouped_balance_note:
        summary_lines.append(html.Div(grouped_balance_note))
    geometry_guardrail_note = str(geometry_guardrail_state.get("note", "")).strip()
    if geometry_guardrail_note:
        summary_lines.append(html.Div(geometry_guardrail_note))
    return html.Div(summary_lines)


def build_cyanide_control(chemistry: Dict[str, float]) -> html.Div:
    return readonly_value_block(
        label=display_label(CYANIDE_SOLUBLE_COL),
        value_text=f"{float(chemistry['cyanide']):.2f}%",
        badge_text="Auto",
    )


def build_assumptions_section(content: html.Div) -> html.Div:
    return html.Div(
        [
            section_title("Assumptions And Calculations", margin_top="0"),
            html.Div(
                "Internal conversions, hidden model inputs, geometry guardrails, and resolved chemistry are listed below.",
                style={"fontSize": "12px", "color": "#5a626c", "marginBottom": "12px"},
            ),
            content,
        ],
        style={
            "padding": "24px 28px 32px 28px",
            "color": "#50555c",
            "fontSize": "14px",
            "lineHeight": "1.7",
            "borderTop": "1px solid #d9d3c5",
            "background": "#fffdf7",
        },
    )


def control_grid(children: List[html.Div]) -> html.Div:
    return html.Div(
        children,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))",
            "gap": "12px 14px",
            "alignItems": "start",
        },
    )


def section_title(text: str, margin_top: str = "18px") -> html.Div:
    return html.Div(
        text,
        style={"fontSize": "15px", "fontWeight": 700, "marginTop": margin_top, "marginBottom": "10px"},
    )


def create_app(project_root: str) -> Dash:
    df = load_training_dataframe()
    available_grouped = [c for c in ALL_GROUPED_COLUMNS if c in df.columns or c in BASE.STATIC_PREDICTOR_COLUMNS]
    grouped_model_columns = [c for c in available_grouped if c in BASE.STATIC_PREDICTOR_COLUMNS]

    spec_columns = ordered_unique(
        EDITABLE_INPUT_COLUMNS
        + available_grouped
        + list(getattr(BASE, "INPUT_ONLY_COLUMNS", []))
        + list(BASE.STATIC_PREDICTOR_COLUMNS)
        + [FE_INPUT_COL]
    )
    input_specs = build_numeric_slider_specs(df, spec_columns)
    weekly_spec = build_weekly_catalyst_spec(df)
    irrigation_spec = build_irrigation_rate_spec(df)
    members, _model_dir, _skipped_paths = load_saved_member_models(project_root)
    helper_inputs_note = (
        "Acid soluble and residual chalcopyrite stay user-editable, while cyanide is auto-balanced so the three sum to 100%. "
        "Fe % also stays user-editable. The app converts these inputs internally into the active model features before inference."
        if needs_fe_input()
        else "Acid soluble and residual chalcopyrite stay user-editable, while cyanide is auto-balanced so the three sum to 100%. "
        "The app converts these inputs internally into the active model features before inference."
    )

    default_ui_values = {column: float(input_specs[column]["default"]) for column in EDITABLE_INPUT_COLUMNS}
    default_ui_values, default_geometry_guardrail_state = apply_geometry_input_limits(default_ui_values, input_specs)
    default_ui_values, default_feed_mass_state = resolve_internal_feed_mass(default_ui_values, input_specs)
    default_static_values, default_derived_values, _default_inactive_grouped_values, default_grouped_balance_state, default_chemistry = resolve_model_static_values(default_ui_values, input_specs)
    default_geometry_values = geometry_predictor_values(default_static_values)
    default_pred = predict_ensemble_curves(
        members=members,
        static_values=default_static_values,
        weekly_catalyst_gt_week=float(weekly_spec["default"]),
        catalyst_addition_start_day=0.0,
        irrigation_rate_l_m2_h=float(irrigation_spec["default"]),
        confidence_interval_high=90.0,
        max_day=float(BASE.CONFIG.get("ensemble_plot_target_day", 2500.0)),
    )

    top_controls = [
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
            label="Catalyst used (g/t/week)",
            component_id="weekly-catalyst-gt-week",
            spec=weekly_spec,
        ),
        slider_block(
            label="Confidence interval (%)",
            component_id="confidence-interval",
            spec={"min": 60.0, "max": 95.0, "default": 90.0, "step": 1.0},
        ),
        slider_block(
            label="Irrigation rate (L/h/m2)",
            component_id="irrigation-rate-l-m2-h",
            spec=irrigation_spec,
        ),
        slider_block(
            label="Catalyst addition start day",
            component_id="catalyst-addition-start-day",
            spec={"min": 0.0, "max": 700.0, "default": 200.0, "step": 1.0},
        ),
    ]
    for column in TOP_DISPLAY_COLUMNS:
        top_controls.append(
            slider_block(
                label=display_label(column),
                component_id=predictor_component_id(column),
                spec={**dict(input_specs[column]), "default": float(default_ui_values[column])},
            )
        )

    main_controls = [
    ]
    for column in MAIN_EDITABLE_COLUMNS:
        slider_spec = {**dict(input_specs[column]), "default": float(default_ui_values[column])}
        if column == ACID_SOLUBLE_COL:
            slider_spec["max"] = float(default_chemistry["acid_max"])
        elif column == RESIDUAL_CPY_COL:
            slider_spec["max"] = float(default_chemistry["residual_max"])
        main_controls.append(
            slider_block(
                label=display_label(column),
                component_id=predictor_component_id(column),
                spec=slider_spec,
            )
        )
        if column == ACID_SOLUBLE_COL:
            main_controls.append(
                html.Div(
                    id="chemistry-cyanide-control",
                    children=build_cyanide_control(default_chemistry),
                )
            )
    grouped_model_controls = [
        slider_block(
            label=display_label(column),
            component_id=predictor_component_id(column),
            spec={**dict(input_specs[column]), "default": float(default_ui_values[column])},
        )
        for column in grouped_model_columns
    ]
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1("Rosetta NeuralNetwork Bi-Exponential Equation - Interactive Plot", style={"marginBottom": "6px"}),
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
                            control_grid(top_controls),
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
                                        "Exports current user inputs, derived internal values, model inputs, and predictions every 7 days.",
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
                            section_title("User-Facing Ore And Chemistry Inputs", margin_top="0"),
                            html.Div(
                                helper_inputs_note,
                                style={"fontSize": "12px", "color": "#5a626c", "marginBottom": "10px"},
                            ),
                            control_grid(main_controls),
                            section_title("Modals Predictors In Model"),
                            html.Div(
                                "Grouped mineralogy sliders below are active model inputs.",
                                style={"fontSize": "12px", "color": "#5a626c", "marginBottom": "10px"},
                            ),
                            control_grid(grouped_model_controls) if grouped_model_controls else html.Div(
                                "No grouped predictors are active in the current model.",
                                style={"fontSize": "12px", "color": "#5a626c"},
                            ),
                            html.Div(
                                default_grouped_balance_state["note"],
                                id="grouped-balance-note",
                                style={"fontSize": "12px", "color": "#5a626c", "marginTop": "10px"},
                            ),
                        ],
                        style={
                            "width": "34%",
                            "minWidth": "360px",
                            "padding": "22px 24px",
                            "background": "#f7f4ee",
                            "borderRight": "1px solid #d9d3c5",
                            "maxHeight": "calc(100vh - 88px)",
                            "overflowY": "auto",
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
                        ],
                        style={"width": "66%", "background": "#fffdf7"},
                    ),
                ],
                style={
                    "display": "flex",
                    "minHeight": "calc(100vh - 88px)",
                    "background": "#fffdf7",
                    "color": "#1c232b",
                },
            ),
            html.Div(
                id="prediction-summary",
                children=build_assumptions_section(
                    build_prediction_summary(
                        pred=default_pred,
                        ui_values=default_ui_values,
                        derived_values=default_derived_values,
                        geometry_values=default_geometry_values,
                        geometry_guardrail_state=default_geometry_guardrail_state,
                        feed_mass_state=default_feed_mass_state,
                        grouped_balance_state=default_grouped_balance_state,
                        resolved_chemistry=default_chemistry,
                    )
                ),
            ),
        ],
        style={"fontFamily": "Helvetica, Arial, sans-serif", "background": "#fffdf7"},
    )

    callback_inputs = [
        Input("max-day", "value"),
        Input("weekly-catalyst-gt-week", "value"),
        Input("confidence-interval", "value"),
        Input("irrigation-rate-l-m2-h", "value"),
        Input("catalyst-addition-start-day", "value"),
        *[Input(predictor_component_id(column), "value") for column in EDITABLE_INPUT_COLUMNS],
    ]
    callback_states = [
        State("max-day", "value"),
        State("weekly-catalyst-gt-week", "value"),
        State("confidence-interval", "value"),
        State("irrigation-rate-l-m2-h", "value"),
        State("catalyst-addition-start-day", "value"),
        *[State(predictor_component_id(column), "value") for column in EDITABLE_INPUT_COLUMNS],
    ]

    @app.callback(
        Output("prediction-graph", "figure"),
        Output("prediction-summary", "children"),
        Output(predictor_component_id("column_height_m"), "value"),
        Output(predictor_component_id("column_inner_diameter_m"), "value"),
        Output(predictor_component_id(ACID_SOLUBLE_COL), "value"),
        Output(predictor_component_id(ACID_SOLUBLE_COL), "max"),
        Output(predictor_component_id(ACID_SOLUBLE_COL), "marks"),
        Output(predictor_component_id(RESIDUAL_CPY_COL), "value"),
        Output(predictor_component_id(RESIDUAL_CPY_COL), "max"),
        Output(predictor_component_id(RESIDUAL_CPY_COL), "marks"),
        Output("chemistry-cyanide-control", "children"),
        Output("grouped-balance-note", "children"),
        callback_inputs,
    )
    def update_plot(
        max_day: float,
        weekly_catalyst_gt_week: float,
        confidence_interval: float,
        irrigation_rate_l_m2_h: float,
        catalyst_addition_start_day: float,
        *predictor_values: float,
    ) -> Tuple[Any, ...]:
        ui_values = resolve_ui_state(tuple(predictor_values), input_specs)
        ui_values, geometry_guardrail_state = apply_geometry_input_limits(ui_values, input_specs)
        ui_values, feed_mass_state = resolve_internal_feed_mass(ui_values, input_specs)
        static_values, derived_values, _inactive_grouped_values, grouped_balance_state, resolved_chemistry = resolve_model_static_values(ui_values, input_specs)
        geometry_values = geometry_predictor_values(static_values)
        resolved_max_day = float(max_day or BASE.CONFIG.get("ensemble_plot_target_day", 2500.0))
        pred = predict_ensemble_curves(
            members=members,
            static_values=static_values,
            weekly_catalyst_gt_week=float(weekly_catalyst_gt_week or 0.0),
            irrigation_rate_l_m2_h=float(irrigation_rate_l_m2_h or 0.0),
            catalyst_addition_start_day=float(catalyst_addition_start_day or 0.0),
            confidence_interval_high=float(confidence_interval or 90.0),
            max_day=resolved_max_day,
        )
        return (
            make_prediction_figure(pred, resolved_max_day),
            build_assumptions_section(
                build_prediction_summary(
                    pred=pred,
                    ui_values=ui_values,
                    derived_values=derived_values,
                    geometry_values=geometry_values,
                    geometry_guardrail_state=geometry_guardrail_state,
                    feed_mass_state=feed_mass_state,
                    grouped_balance_state=grouped_balance_state,
                    resolved_chemistry=resolved_chemistry,
                )
            ),
            float(ui_values["column_height_m"]),
            float(ui_values["column_inner_diameter_m"]),
            float(resolved_chemistry["acid"]),
            float(resolved_chemistry["acid_max"]),
            slider_marks(float(input_specs[ACID_SOLUBLE_COL]["min"]), float(resolved_chemistry["acid_max"])),
            float(resolved_chemistry["residual"]),
            float(resolved_chemistry["residual_max"]),
            slider_marks(float(input_specs[RESIDUAL_CPY_COL]["min"]), float(resolved_chemistry["residual_max"])),
            build_cyanide_control(resolved_chemistry),
            grouped_balance_state["note"],
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
        irrigation_rate_l_m2_h: float,
        catalyst_addition_start_day: float,
        *predictor_values: float,
    ) -> Dict[str, Any]:
        ui_values = resolve_ui_state(tuple(predictor_values), input_specs)
        ui_values, _geometry_guardrail_state = apply_geometry_input_limits(ui_values, input_specs)
        ui_values, _feed_mass_state = resolve_internal_feed_mass(ui_values, input_specs)
        static_values, derived_values, inactive_grouped_values, grouped_balance_state, resolved_chemistry = resolve_model_static_values(ui_values, input_specs)
        geometry_values = geometry_predictor_values(static_values)
        resolved_max_day = float(max_day or BASE.CONFIG.get("ensemble_plot_target_day", 2500.0))
        pred = predict_ensemble_curves(
            members=members,
            static_values=static_values,
            weekly_catalyst_gt_week=float(weekly_catalyst_gt_week or 0.0),
            irrigation_rate_l_m2_h=float(irrigation_rate_l_m2_h or 0.0),
            catalyst_addition_start_day=float(catalyst_addition_start_day or 0.0),
            confidence_interval_high=float(confidence_interval or 90.0),
            max_day=resolved_max_day,
        )
        excel_bytes = build_prediction_export_bytes(
            pred=pred,
            ui_values=ui_values,
            static_values=static_values,
            derived_values=derived_values,
            geometry_values=geometry_values,
            inactive_grouped_values=inactive_grouped_values,
            grouped_balance_state=grouped_balance_state,
            resolved_chemistry=resolved_chemistry,
            max_day=resolved_max_day,
        )
        return dcc.send_bytes(excel_bytes, "interactive_ensemble_prediction_v5_7day.xlsx")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive Plotly/Dash app for NN_ExpEq_columns_only_v5 using the saved "
            "member checkpoints created after running NN_ExpEq_columns_only_v5.py."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8057)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional override for the NN_Pytorch_ExpEq_columns_only_v5 root.",
    )
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    app = create_app(project_root)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
