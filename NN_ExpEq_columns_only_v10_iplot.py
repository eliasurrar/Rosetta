import argparse
import base64
import glob
import importlib.util
import io
import inspect
import json
import math
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update

# IMPORTANT: REMOVED CU_% FROM THE MODEL INPUT 
# BUT IT WILL BE NECESSARY HERE TO THE USER TO INPUT TO CALCULATE EQUIVALETS.

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_SCRIPT_PATH = os.path.join(THIS_DIR, "NN_ExpEq_columns_only_v10.py")
MINERALOGY_CLUSTERING_PATH = os.path.join(THIS_DIR, "rosetta_mineralogy_clustering.py")
APP_ASSETS_DIR = os.path.join(THIS_DIR, "assets")
HEADER_LOGO_ASSET = "rosetta_logo_v1.png"
GENERATED_FAVICON_ASSET = "favicon.ico"
APP_HEADER_HEIGHT_CSS = "112px"
APP_FONT_FAMILY = "'Aptos', 'Avenir Next', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
MOBILE_LAYOUT_BREAKPOINT_PX = 980
NARROW_LAYOUT_BREAKPOINT_PX = 640

APP_COLOR_BG = "#06121d"
APP_COLOR_BG_ALT = "#0a1f31"
APP_COLOR_HEADER = "#081726"
APP_COLOR_BG_LIGHT = "#eff5fa"
APP_COLOR_BG_LIGHT_ALT = "#e7eff6"
APP_COLOR_BG_LIGHT_PANEL = "#ffffff"
APP_COLOR_PANEL = "rgba(9, 27, 42, 0.96)"
APP_COLOR_PANEL_ALT = "rgba(12, 34, 52, 0.96)"
APP_COLOR_BORDER = "rgba(98, 159, 184, 0.22)"
APP_COLOR_TEXT = "#eef8ff"
APP_COLOR_TEXT_SECONDARY = "#9cb8c9"
APP_COLOR_TEXT_MUTED = "#6f8ea1"
APP_COLOR_TEXT_LIGHT = "#102638"
APP_COLOR_ACCENT = "#2fd5d0"
APP_COLOR_ACCENT_STRONG = "#14b8c6"
APP_COLOR_ACCENT_SOFT = "rgba(47, 213, 208, 0.14)"
APP_COLOR_INPUT_BG = "#091827"
APP_COLOR_INPUT_BORDER = "rgba(102, 159, 184, 0.26)"
APP_COLOR_WARNING = "#ff8f70"
APP_PANEL_SHADOW = "0 24px 60px rgba(2, 9, 18, 0.42)"

CONTROL_CURVE_COLOR = "#1f77b4"
CATALYZED_CURVE_COLOR = "#ff7f0e"

PREDICTION_FIGURE_THEMES: Dict[str, Dict[str, Any]] = {
    "light": {
        "template": "plotly_white",
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font_color": "#1f2933",
        "grid_color": "rgba(15, 23, 42, 0.12)",
        "axis_line_color": "rgba(15, 23, 42, 0.18)",
        "hover_bgcolor": "rgba(255, 255, 255, 0.88)",
        "hover_bordercolor": "rgba(60, 60, 60, 0.25)",
        "annotation_bgcolor": "rgba(255, 255, 255, 0.92)",
        "annotation_bordercolor": "#666666",
        "vline_color": "#7a7a7a",
        "band_alpha": 0.18,
        "glow_inner_alpha": 0.16,
        "glow_outer_alpha": 0.08,
    },
    "dark": {
        "template": "plotly",
        "paper_bgcolor": "#081625",
        "plot_bgcolor": "#102436",
        "font_color": APP_COLOR_TEXT,
        "grid_color": "rgba(152, 191, 210, 0.14)",
        "axis_line_color": "rgba(152, 191, 210, 0.22)",
        "hover_bgcolor": "rgba(8, 22, 35, 0.96)",
        "hover_bordercolor": "rgba(80, 156, 188, 0.32)",
        "annotation_bgcolor": "rgba(8, 20, 34, 0.92)",
        "annotation_bordercolor": "rgba(88, 160, 191, 0.28)",
        "vline_color": "rgba(181, 203, 214, 0.55)",
        "band_alpha": 0.24,
        "glow_inner_alpha": 0.24,
        "glow_outer_alpha": 0.12,
    },
}


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("nn_expeq_columns_only_v10_base", BASE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base model script: {BASE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def resolve_asset_path(asset_ref: str) -> str:
    return str(asset_ref) if os.path.isabs(str(asset_ref)) else os.path.join(APP_ASSETS_DIR, str(asset_ref))


def asset_public_name(asset_ref: str) -> str:
    return os.path.basename(resolve_asset_path(asset_ref))


@lru_cache(maxsize=None)
def inline_asset_data_url(filename: str, mime_type: str) -> str | None:
    asset_path = resolve_asset_path(filename)
    if not os.path.exists(asset_path):
        return None
    with open(asset_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def asset_mime_type(filename: str) -> str:
    ext = os.path.splitext(str(filename).lower())[1]
    if ext == ".png":
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".svg":
        return "image/svg+xml"
    if ext == ".webp":
        return "image/webp"
    if ext == ".ico":
        return "image/x-icon"
    return "application/octet-stream"


def png_dimensions(png_bytes: bytes) -> Tuple[int, int]:
    if len(png_bytes) < 24 or png_bytes[:8] != b"\x89PNG\r\n\x1a\n" or png_bytes[12:16] != b"IHDR":
        raise ValueError("Invalid PNG asset; could not read favicon dimensions.")
    width = int.from_bytes(png_bytes[16:20], byteorder="big", signed=False)
    height = int.from_bytes(png_bytes[20:24], byteorder="big", signed=False)
    return width, height


def build_ico_from_png_bytes(png_bytes: bytes) -> bytes:
    width, height = png_dimensions(png_bytes)
    width_byte = 0 if width >= 256 else max(1, width)
    height_byte = 0 if height >= 256 else max(1, height)
    image_offset = 6 + 16
    icon_dir = (0).to_bytes(2, "little") + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
    icon_entry = (
        bytes([width_byte, height_byte, 0, 0])
        + (1).to_bytes(2, "little")
        + (32).to_bytes(2, "little")
        + len(png_bytes).to_bytes(4, "little")
        + image_offset.to_bytes(4, "little")
    )
    return icon_dir + icon_entry + png_bytes


def ensure_generated_favicon_asset(source_filename: str, generated_filename: str = GENERATED_FAVICON_ASSET) -> str | None:
    source_path = resolve_asset_path(source_filename)
    if not os.path.exists(source_path):
        return None

    generated_path = resolve_asset_path(generated_filename)
    source_ext = os.path.splitext(source_path.lower())[1]

    if source_ext == ".ico":
        with open(source_path, "rb") as f:
            source_bytes = f.read()
    elif source_ext == ".png":
        with open(source_path, "rb") as f:
            source_bytes = build_ico_from_png_bytes(f.read())
    else:
        raise RuntimeError(
            "HEADER_LOGO_ASSET must be a PNG or ICO file to auto-generate favicon.ico without extra image dependencies."
        )

    existing_bytes = b""
    if os.path.exists(generated_path):
        with open(generated_path, "rb") as f:
            existing_bytes = f.read()
    if source_bytes != existing_bytes:
        with open(generated_path, "wb") as f:
            f.write(source_bytes)
        inline_asset_data_url.cache_clear()
    return generated_path

GROUPED_DERIVED_SOURCE_COLUMNS: Dict[str, str] = {
    "grouped_primary_copper_sulfides": "copper_primary_sulfides_equivalent",
    "grouped_secondary_copper_sulfides": "copper_secondary_sulfides_equivalent",
    "grouped_mixed_copper_ores": "copper_oxides_equivalent",
    "grouped_copper_oxides": "copper_oxides_equivalent",
}
GROUPED_DERIVED_FALLBACK_REGRESSIONS: Dict[str, Tuple[float, float]] = {
    "grouped_primary_copper_sulfides": (2.8918, 0.0965),
    "grouped_secondary_copper_sulfides": (1.2245, -0.0136),
    "grouped_mixed_copper_ores": (0.5380, -0.0015),
    "grouped_copper_oxides": (0.1632, -0.0013),
}


def infer_derived_mineralogy_modals(
    primary_copper_sulf: float,
    secondary_copper_sulf: float,
    copper_oxides: float,
    regressions: Dict[str, Tuple[float, float]] | None = None,
) -> Tuple[float, float, float, float]:
    # Infer grouped modal mineralogy from the equivalent chemistry values using
    # CSV-fitted linear regressions, with hard-coded fallbacks if the fit cannot be built.
    source_values = {
        PRIMARY_SULFIDES_EQUIV_COL: float(primary_copper_sulf),
        SECONDARY_SULFIDES_EQUIV_COL: float(secondary_copper_sulf),
        OXIDES_EQUIV_COL: float(copper_oxides),
    }
    regressions = regressions or load_grouped_modal_regressions()
    inferred_values: Dict[str, float] = {}
    for grouped_column, source_column in GROUPED_DERIVED_SOURCE_COLUMNS.items():
        slope, intercept = regressions.get(
            grouped_column,
            GROUPED_DERIVED_FALLBACK_REGRESSIONS[grouped_column],
        )
        inferred_values[grouped_column] = max(
            0.0,
            float(slope) * source_values[source_column] + float(intercept),
        )

    return (
        inferred_values["grouped_primary_copper_sulfides"],
        inferred_values["grouped_secondary_copper_sulfides"],
        inferred_values["grouped_mixed_copper_ores"],
        inferred_values["grouped_copper_oxides"],
    )


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
    "color": "var(--app-text)",
}

READONLY_NOTE_STYLE = {
    "marginTop": "4px",
    "fontSize": "10px",
    "color": "var(--app-text-secondary)",
    "lineHeight": "1.25",
}

READONLY_BADGE_STYLE = {
    "fontSize": "10px",
    "fontWeight": 700,
    "color": "var(--app-accent)",
    "background": "var(--app-accent-soft)",
    "border": "1px solid var(--app-border)",
    "borderRadius": "999px",
    "padding": "2px 7px",
    "letterSpacing": "0.02em",
    "textTransform": "uppercase",
}

READONLY_TRACK_COLOR = "var(--readonly-track)"
READONLY_FILL_COLOR = "var(--readonly-fill)"

CONTROL_NUMBER_INPUT_STYLE = {
    "width": "88px",
    "padding": "6px 8px",
    "border": "1px solid var(--app-input-border)",
    "borderRadius": "8px",
    "background": "var(--app-input-bg)",
    "color": "var(--app-text)",
    "fontSize": "13px",
    "boxSizing": "border-box",
}

CONTROL_STEP_BUTTON_STYLE = {
    "width": "18px",
    "height": "13px",
    "padding": "0",
    "border": "1px solid var(--app-input-border)",
    "borderRadius": "4px",
    "background": "var(--app-panel-alt-bg)",
    "color": "var(--app-text)",
    "fontSize": "10px",
    "fontWeight": 700,
    "cursor": "pointer",
    "lineHeight": "1",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
}

PNG_EXPORT_DPI = 300
PNG_EXPORT_WIDTH_IN = 12.0
PNG_EXPORT_HEIGHT_IN = 8.0
PNG_EXPORT_SCALE = 4
PNG_EXPORT_FINAL_WIDTH_PX = int(PNG_EXPORT_WIDTH_IN * PNG_EXPORT_DPI)
PNG_EXPORT_FINAL_HEIGHT_PX = int(PNG_EXPORT_HEIGHT_IN * PNG_EXPORT_DPI)
PNG_EXPORT_WIDTH_PX = int(PNG_EXPORT_FINAL_WIDTH_PX / PNG_EXPORT_SCALE)
PNG_EXPORT_HEIGHT_PX = int(PNG_EXPORT_FINAL_HEIGHT_PX / PNG_EXPORT_SCALE)
INTERACTIVE_PLOT_STEP_DAYS = 50.0
MAX_CONTROL_DECIMALS = 3
GEOMETRY_RATIO_MAX = 0.25 # 1:4
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
INPUT_ARROW_STEPS: Dict[str, float] = {
    "max-day": 100.0,
    "weekly-catalyst-gt-week": 0.5,
    "confidence-interval": 5.0,
    "irrigation-rate-l-m2-h": 0.5,
    "catalyst-addition-start-day": 50.0,
    "column_height_m": 1.0,
    "column_inner_diameter_m": 0.1,
    "material_size_p80_in": 0.5,
    CU_INPUT_COL: 0.1,
    FE_INPUT_COL: 0.5,
    ACID_SOLUBLE_COL: 1.0,
    RESIDUAL_CPY_COL: 5.0,
    "grouped_acid_generating_sulfides": 1.0,
    "grouped_carbonates": 0.1,
    "grouped_fe_oxides": 0.25,
    "grouped_accessory_minerals": 0.5, 
}
GROUPED_INPUT_CAPTIONS: Dict[str, str] = {
    "grouped_fe_oxides": "hematite, magnetite, limonite, etc.",
    "grouped_carbonates": "calcite, dolomite, siderite, etc.",
    "grouped_acid_generating_sulfides": "pyrite, molybdenite.",
    "grouped_gangue_sulfides": "sphaletire, galena, other sulfides.",
    "grouped_accessory_minerals": "apatite, zircon, barite, rutile, ilmenite, monzonite, etc.",
}
STRICTLY_POSITIVE_MIN_COLUMNS = {
    CU_INPUT_COL,
    FE_INPUT_COL,
    "column_inner_diameter_m",
}


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


def slider_input_component_id(component_id: str) -> str:
    return f"{component_id}-input"


def decrement_button_component_id(component_id: str) -> str:
    return f"{component_id}-decrement"


def increment_button_component_id(component_id: str) -> str:
    return f"{component_id}-increment"


def candidate_project_roots() -> List[str]:
    roots = []
    for attr in ["PROJECT_ROOT", "DEFAULT_PROJECT_ROOT", "LOCAL_PROJECT_ROOT"]:
        value = getattr(BASE, attr, None)
        if isinstance(value, str) and value not in roots:
            roots.append(value)
    sibling_root = os.path.join(THIS_DIR, "NN_Pytorch_ExpEq_columns_only_v10")
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
        "Run NN_ExpEq_columns_only_v10.py first so it writes the checkpoint paths listed in outputs/run_manifest.json.\n"
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
        "Run NN_ExpEq_columns_only_v10.py first and then use the checkpoint directory recorded in outputs/run_manifest.json.\n"
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


def control_step_decimals(step: float, max_decimals: int = MAX_CONTROL_DECIMALS) -> int:
    if not np.isfinite(step) or step <= 0.0:
        return 0
    for digits in range(max_decimals + 1):
        if abs(float(step) - round(float(step), digits)) <= 10.0 ** (-(max_decimals + 2)):
            return digits
    return max_decimals


def round_control_value(value: float, digits: int = MAX_CONTROL_DECIMALS) -> float:
    if not np.isfinite(value):
        return float(value)
    bounded_digits = int(np.clip(digits, 0, MAX_CONTROL_DECIMALS))
    return float(round(float(value), bounded_digits))


def round_control_value_for_spec(value: float, spec: Dict[str, float]) -> float:
    rounded = round_control_value(value, MAX_CONTROL_DECIMALS)
    step = float(spec.get("step", np.nan))
    if not np.isfinite(step) or step <= 0.0:
        return rounded
    step_digits = int(spec.get("decimals", control_step_decimals(step)))
    nearest_step_value = round(float(value) / step) * step
    if abs(float(value) - nearest_step_value) <= max(GEOMETRY_TOL, abs(step) * 1e-3):
        return round_control_value(nearest_step_value, step_digits)
    return rounded


def expand_numeric_bounds(min_value: float, max_value: float, step_value: float, digits: int) -> Tuple[float, float]:
    if not np.isfinite(step_value) or step_value <= 0.0:
        return round_control_value(min_value, digits), round_control_value(max_value, digits)
    lower = math.floor((float(min_value) / step_value) + GEOMETRY_TOL) * step_value
    upper = math.ceil((float(max_value) / step_value) - GEOMETRY_TOL) * step_value
    return round_control_value(lower, digits), round_control_value(upper, digits)


def finalize_numeric_spec(
    spec: Dict[str, float],
    step_value: float | None = None,
    force_min_zero: bool = False,
    force_positive_min: bool = False,
) -> Dict[str, float]:
    out = dict(spec)
    min_value = 0.0 if force_min_zero else float(out.get("min", 0.0))
    max_value = float(out.get("max", min_value))
    default_value = float(out.get("default", min_value))
    resolved_step = float(step_value if step_value is not None else out.get("step", nice_step(min_value, max_value)))
    if not np.isfinite(resolved_step) or resolved_step <= 0.0:
        resolved_step = 1.0
    digits = control_step_decimals(resolved_step)
    resolved_step = round_control_value(resolved_step, digits)
    rounded_min, rounded_max = expand_numeric_bounds(min_value, max_value, resolved_step, digits)
    if force_positive_min and min_value > 0.0 and rounded_min <= 0.0:
        positive_floor = round_control_value(resolved_step if resolved_step > 0.0 else 10.0 ** (-digits), digits)
        rounded_min = positive_floor if positive_floor > 0.0 else round_control_value(10.0 ** (-digits), digits)
    if rounded_max <= rounded_min:
        min_increment = resolved_step if resolved_step > 0.0 else (10.0 ** (-digits) if digits > 0 else 1.0)
        rounded_max = round_control_value(rounded_min + min_increment, digits)
    out["min"] = float(rounded_min)
    out["max"] = float(rounded_max)
    out["step"] = float(resolved_step)
    out["decimals"] = float(digits)
    out["default"] = quantize_default_for_step(default_value, resolved_step, rounded_min, rounded_max)
    return out


def format_spec_value(value: float, spec: Dict[str, float]) -> str:
    digits = int(spec.get("decimals", control_step_decimals(float(spec.get("step", 0.0)))))
    rounded = round_control_value(value, digits)
    return f"{rounded:.{digits}f}" if digits > 0 else f"{rounded:.0f}"


@lru_cache(maxsize=1)
def load_training_dataframe() -> pd.DataFrame:
    return pd.read_csv(BASE.DATA_PATH)


def fit_grouped_modal_regression(
    df: pd.DataFrame,
    source_column: str,
    target_column: str,
    fallback: Tuple[float, float],
) -> Tuple[float, float]:
    if source_column not in df.columns or target_column not in df.columns:
        return fallback
    x = pd.to_numeric(df[source_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[target_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 2:
        return fallback
    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return fallback
    return float(slope), float(intercept)


@lru_cache(maxsize=1)
def load_grouped_modal_regressions() -> Dict[str, Tuple[float, float]]:
    df = load_training_dataframe()
    return {
        grouped_column: fit_grouped_modal_regression(
            df=df,
            source_column=source_column,
            target_column=grouped_column,
            fallback=GROUPED_DERIVED_FALLBACK_REGRESSIONS[grouped_column],
        )
        for grouped_column, source_column in GROUPED_DERIVED_SOURCE_COLUMNS.items()
    }


def value_or_default(value: float | None, default: float) -> float:
    return float(default if value is None or not np.isfinite(value) else value)


def safe_ratio(numerator: float, denominator: float) -> float:
    num = float(numerator) if np.isfinite(numerator) else 0.0
    den = float(denominator) if np.isfinite(denominator) else 0.0
    return float(num / max(den, 1e-6))


def hex_to_rgba(color: str, alpha: float) -> str:
    value = str(color).strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        return color
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    return f"rgba({red}, {green}, {blue}, {alpha_clamped:.3f})"


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
        "border": "2px solid #d7f7f5",
        "borderRadius": "50%",
        "boxShadow": "0 0 0 1px rgba(4, 13, 22, 0.36)",
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


def normalize_control_value(value: float | None, spec: Dict[str, float], fallback: float | None = None) -> float:
    base_value = fallback if value is None else value
    resolved = round_control_value_for_spec(value_or_default(base_value, float(spec["default"])), spec)
    min_value = float(spec["min"])
    max_value = float(spec["max"])
    if resolved < min_value:
        return round_control_value(min_value, MAX_CONTROL_DECIMALS)
    if resolved > max_value:
        return round_control_value(max_value, MAX_CONTROL_DECIMALS)
    return round_control_value(float(np.clip(resolved, min_value, max_value)), MAX_CONTROL_DECIMALS)


def quantize_default_for_step(value: float, step: float, min_value: float, max_value: float) -> float:
    if not np.isfinite(step) or step <= 0.0:
        return round_control_value(float(np.clip(value, min_value, max_value)), MAX_CONTROL_DECIMALS)
    step = float(step)
    digits = control_step_decimals(step)
    clipped = float(np.clip(value, min_value, max_value))
    lower_bound = math.ceil((float(min_value) - GEOMETRY_TOL) / step) * step
    upper_bound = math.floor((float(max_value) + GEOMETRY_TOL) / step) * step
    if lower_bound > upper_bound:
        return round_control_value(float(clipped), digits)
    quantized = math.floor((clipped + GEOMETRY_TOL) / step) * step
    quantized = max(lower_bound, min(quantized, upper_bound))
    return round_control_value(float(np.clip(quantized, min_value, max_value)), digits)


def apply_input_arrow_step(spec: Dict[str, float], step_value: float) -> Dict[str, float]:
    out = dict(spec)
    digits = control_step_decimals(float(step_value))
    out["step"] = round_control_value(float(step_value), digits)
    out["decimals"] = float(digits)
    out["default"] = quantize_default_for_step(
        float(out["default"]),
        float(out["step"]),
        float(out["min"]),
        float(out["max"]),
    )
    return out


def diameter_ratio_guardrail_min(
    material_size_p80_in: float | None,
    input_specs: Dict[str, Dict[str, float]],
) -> float:
    diameter_spec = input_specs["column_inner_diameter_m"]
    diameter_min_spec = float(diameter_spec["min"])
    diameter_max_spec = float(diameter_spec["max"])
    p80_in = value_or_default(material_size_p80_in, np.nan)
    if not np.isfinite(p80_in) or p80_in <= 0.0:
        return diameter_min_spec
    required_diameter = float(p80_in) * 0.0254 / GEOMETRY_RATIO_MAX
    return round_control_value(float(max(diameter_min_spec, min(required_diameter, diameter_max_spec))), MAX_CONTROL_DECIMALS)


def blocking_constraint_message(
    geometry_guardrail_state: Dict[str, Any],
    grouped_balance_state: Dict[str, Any],
) -> str:
    messages: List[str] = []
    if not bool(geometry_guardrail_state.get("ratio_ok", True)) or not bool(geometry_guardrail_state.get("density_ok", True)):
        note_text = str(geometry_guardrail_state.get("note", "")).strip()
        messages.append(note_text or "Geometry constraints are not satisfied.")
    if bool(grouped_balance_state.get("show_warning", False)):
        note_text = str(grouped_balance_state.get("note", "")).strip()
        messages.append(note_text or "Grouped mineralogy constraints are not satisfied.")
    return " ".join(messages)


def current_control_bounds(
    component_id: str,
    resolved_control_values: Dict[str, float],
    control_specs_by_id: Dict[str, Dict[str, float]],
    input_specs: Dict[str, Dict[str, float]],
) -> Tuple[float, float]:
    spec = control_specs_by_id[component_id]
    min_value = float(spec["min"])
    max_value = float(spec["max"])

    acid_component_id = predictor_component_id(ACID_SOLUBLE_COL)
    residual_component_id = predictor_component_id(RESIDUAL_CPY_COL)
    diameter_component_id = predictor_component_id("column_inner_diameter_m")
    p80_component_id = predictor_component_id("material_size_p80_in")

    if component_id == acid_component_id:
        acid_spec = input_specs[ACID_SOLUBLE_COL]
        residual_value = float(resolved_control_values.get(residual_component_id, float(acid_spec["min"])))
        min_value = float(acid_spec["min"])
        max_value = max(min_value, min(float(acid_spec["max"]), 100.0 - residual_value))
    elif component_id == residual_component_id:
        residual_spec = input_specs[RESIDUAL_CPY_COL]
        acid_value = float(resolved_control_values.get(acid_component_id, float(residual_spec["min"])))
        min_value = float(residual_spec["min"])
        max_value = max(min_value, min(float(residual_spec["max"]), 100.0 - acid_value))
    elif component_id == diameter_component_id:
        p80_value = float(resolved_control_values.get(p80_component_id, float(input_specs["material_size_p80_in"]["default"])))
        min_value = diameter_ratio_guardrail_min(p80_value, input_specs)

    return round_control_value(float(min_value), MAX_CONTROL_DECIMALS), round_control_value(float(max_value), MAX_CONTROL_DECIMALS)


def slider_block(
    label: str,
    component_id: str,
    spec: Dict[str, float],
    caption_text: str = "",
    container_style: Dict[str, Any] | None = None,
) -> html.Div:
    style = {"marginBottom": "0", "minWidth": "0"}
    if container_style:
        style.update(container_style)
    caption = str(caption_text).strip()
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                label,
                                style={
                                    "fontWeight": 600,
                                    "fontSize": "13px",
                                    "lineHeight": "1.2",
                                    "minWidth": "0",
                                    "color": "var(--app-text)",
                                },
                            ),
                            html.Div(
                                caption,
                                style={
                                    "fontSize": "11px",
                                    "lineHeight": "1.35",
                                    "minWidth": "0",
                                    "color": "var(--app-text-secondary)",
                                    "display": "block" if caption else "none",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "4px",
                            "minWidth": "0",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id=slider_input_component_id(component_id),
                                type="number",
                                min=float(spec["min"]),
                                max=float(spec["max"]),
                                step="any",
                                value=normalize_control_value(float(spec["default"]), spec),
                                debounce=True,
                                inputMode="decimal",
                                className="control-number-input",
                                disabled=False,
                                style=dict(CONTROL_NUMBER_INPUT_STYLE),
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "+",
                                        id=increment_button_component_id(component_id),
                                        n_clicks=0,
                                        disabled=False,
                                        style=dict(CONTROL_STEP_BUTTON_STYLE),
                                    ),
                                    html.Button(
                                        "-",
                                        id=decrement_button_component_id(component_id),
                                        n_clicks=0,
                                        disabled=False,
                                        style=dict(CONTROL_STEP_BUTTON_STYLE),
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "height": "30px",
                                    "justifyContent": "space-between",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "6px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "flex-start",
                    "gap": "6px",
                    "marginBottom": "0",
                },
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
        "value_text": format_spec_value(float(value), spec),
        "fill_style": readonly_fill_style(position_pct, READONLY_FILL_COLOR),
        "thumb_style": readonly_thumb_style(position_pct, READONLY_FILL_COLOR),
        "min_text": format_spec_value(float(spec["min"]), spec),
        "max_text": format_spec_value(float(spec["max"]), spec),
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
                "color": "var(--app-text-muted)",
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
            html.Div(
                label,
                style={"fontWeight": 600, "fontSize": "13px", "marginBottom": "4px", "color": "var(--app-text)"},
            ),
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
        "modeBarButtonsToRemove": ["toImage"],
        "doubleClick": "reset",
        "responsive": True,
    }


def resolve_excel_writer_engine() -> str:
    for engine in ("openpyxl", "xlsxwriter"):
        if importlib.util.find_spec(engine) is not None:
            return engine
    raise RuntimeError("Excel export requires either openpyxl or xlsxwriter to be installed.")


def excel_column_display_width(df: pd.DataFrame, column_name: str, min_width: int = 10, max_width: int = 60) -> float:
    header_width = len(str(column_name))
    if column_name not in df.columns:
        return float(np.clip(header_width + 2, min_width, max_width))
    series = df[column_name]
    value_width = 0
    for value in series.tolist():
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            text = ""
        else:
            text = str(value)
        value_width = max(value_width, len(text))
    return float(np.clip(max(header_width, value_width) + 2, min_width, max_width))


def autosize_excel_sheet(writer: Any, sheet_name: str, df: pd.DataFrame) -> None:
    worksheet = writer.sheets.get(sheet_name)
    if worksheet is None or df.empty:
        return

    if str(getattr(writer, "engine", "")).lower() == "xlsxwriter":
        for idx, column_name in enumerate(df.columns):
            worksheet.set_column(idx, idx, excel_column_display_width(df, column_name))
        return

    if str(getattr(writer, "engine", "")).lower() == "openpyxl":
        from openpyxl.utils import get_column_letter

        for idx, column_name in enumerate(df.columns, start=1):
            worksheet.column_dimensions[get_column_letter(idx)].width = excel_column_display_width(df, column_name)


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
            specs[column] = finalize_numeric_spec({"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.1})
            continue
        values = np.asarray([BASE.scalar_from_maybe_array(v) for v in df[source_column]], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            specs[column] = finalize_numeric_spec({"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.1})
            continue
        min_value = float(np.nanmin(values))
        max_value = float(np.nanmax(values))
        if max_value <= min_value:
            max_value = min_value + 1.0
        specs[column] = finalize_numeric_spec(
            {
                "min": min_value,
                "max": max_value,
                "default": float(np.nanmedian(values)),
                "step": nice_step(min_value, max_value),
            },
            step_value=INPUT_ARROW_STEPS.get(column),
            force_positive_min=column in STRICTLY_POSITIVE_MIN_COLUMNS,
        )
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
        return finalize_numeric_spec(
            {"min": 0.0, "max": 50.0, "default": 5.0, "step": 0.5},
            step_value=INPUT_ARROW_STEPS["weekly-catalyst-gt-week"],
            force_min_zero=True,
        )
    arr = np.asarray(values_gt_week, dtype=float)
    min_value = float(np.nanmin(arr))
    max_value = float(np.nanmax(arr))
    if max_value <= min_value:
        max_value = min_value + 1.0
    return finalize_numeric_spec(
        {
            "min": max(0.0, min_value),
            "max": max_value,
            "default": float(np.nanmedian(arr)),
            "step": nice_step(max(0.0, min_value), max_value),
        },
        step_value=INPUT_ARROW_STEPS["weekly-catalyst-gt-week"],
        force_min_zero=True,
    )


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
        return finalize_numeric_spec(
            {"min": 0.0, "max": 30.0, "default": 10.0, "step": 0.5},
            step_value=INPUT_ARROW_STEPS["irrigation-rate-l-m2-h"],
            force_positive_min=True,
        )
    arr = np.asarray(values_l_m2_h, dtype=float)
    min_value = float(np.nanmin(arr))
    max_value = float(np.nanmax(arr))
    if max_value <= min_value:
        max_value = min_value + 1.0
    min_value = max(0.0, min_value)
    return finalize_numeric_spec(
        {
            "min": min_value,
            "max": max_value,
            "default": float(np.nanmedian(arr)),
            "step": nice_step(min_value, max_value),
        },
        step_value=INPUT_ARROW_STEPS["irrigation-rate-l-m2-h"],
        force_positive_min=True,
    )


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
            "Run NN_ExpEq_columns_only_v10.py first to generate them."
        )

    checkpoint_loader = getattr(BASE, "load_torch_checkpoint", None)
    base_logic_version = getattr(BASE, "MODEL_LOGIC_VERSION", None)
    members: List[Dict[str, Any]] = []
    skipped_paths: List[str] = []
    for path in model_paths:
        if callable(checkpoint_loader):
            ckpt = checkpoint_loader(path, map_location=BASE.device)
        else:
            ckpt = torch.load(path, map_location=BASE.device)
        checkpoint_logic_version = ckpt.get("model_logic_version")
        if base_logic_version and checkpoint_logic_version != base_logic_version:
            skipped_paths.append(
                f"{path} [logic_version={checkpoint_logic_version!r}, expected={base_logic_version!r}]"
            )
            continue
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
                "Re-run NN_ExpEq_columns_only_v10.py to regenerate the saved member models with the current model structure."
            ) from exc
        model.eval()
        scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=float)
        scaler_scale = np.where(np.abs(scaler_scale) < 1e-12, 1.0, scaler_scale)
        members.append(
            {
                "path": path,
                "name": os.path.basename(path),
                "model": model,
                "checkpoint": ckpt,
                "cum_scale": float(ckpt["cum_scale"]),
                "lix_scale": float(ckpt.get("lix_scale", 1.0)),
                "irrigation_scale": float(ckpt.get("irrigation_scale", 1.0)),
                "imputer_statistics": np.asarray(ckpt["imputer_statistics"], dtype=float),
                "scaler_mean": np.asarray(ckpt["scaler_mean"], dtype=float),
                "scaler_scale": scaler_scale,
            }
        )
    if len(members) == 0:
        skipped_text = "\n".join(skipped_paths) if skipped_paths else model_dir
        raise RuntimeError(
            "No compatible saved member checkpoints were found for the current model structure. "
            "Re-run NN_ExpEq_columns_only_v10.py to regenerate them. "
            f"Expected logic version: {base_logic_version!r}.\n"
            f"Skipped:\n{skipped_text}"
        )
    return members, model_dir, skipped_paths


@lru_cache(maxsize=4)
def get_cached_saved_member_models(project_root: str) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    return load_saved_member_models(project_root)


def normalize_cache_float(value: float | int, digits: int = 8) -> float:
    return round(float(value), digits)


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
    cum_scale = max(float(member["cum_scale"]), 1e-6)
    lix_scale = max(float(member["lix_scale"]), 1e-6)
    irrigation_scale = max(float(member["irrigation_scale"]), 1e-6)
    static_imputed_raw = np.asarray(static_raw, dtype=float).copy()
    imputer_statistics = np.asarray(member.get("imputer_statistics"), dtype=float)
    missing = ~np.isfinite(static_imputed_raw)
    static_imputed_raw[missing] = imputer_statistics[missing]
    scaler_mean = np.asarray(member.get("scaler_mean"), dtype=float)
    scaler_scale = np.asarray(member.get("scaler_scale"), dtype=float)
    static_scaled = (static_imputed_raw - scaler_mean) / scaler_scale

    same_plot_grid = False
    if control_time_days is not None:
        same_plot_grid = (
            np.asarray(control_time_days, dtype=float).shape == np.asarray(time_days, dtype=float).shape
            and np.allclose(np.asarray(control_time_days, dtype=float), np.asarray(time_days, dtype=float))
        )

    with torch.inference_mode():
        x = torch.as_tensor(static_scaled, dtype=torch.float32, device=BASE.device).unsqueeze(0)
        x_raw = torch.as_tensor(static_imputed_raw, dtype=torch.float32, device=BASE.device).unsqueeze(0)
        x_input_only = torch.as_tensor(np.asarray(input_only_raw, dtype=float), dtype=torch.float32, device=BASE.device).unsqueeze(0)
        t = torch.as_tensor(np.asarray(time_days, dtype=float), dtype=torch.float32, device=BASE.device)
        c = torch.as_tensor(np.asarray(catalyst_cum, dtype=float) / cum_scale, dtype=torch.float32, device=BASE.device)
        l = torch.as_tensor(np.asarray(lixiviant_cum, dtype=float) / lix_scale, dtype=torch.float32, device=BASE.device)
        irr = torch.as_tensor(
            np.asarray(irrigation_rate_l_m2_h, dtype=float) / irrigation_scale,
            dtype=torch.float32,
            device=BASE.device,
        )
        p_ctrl, p_cat, tau, temp, kappa, aging_strength, latent = model.predict_params(
            x,
            x_raw,
            x_input_only,
            catalyst_t_days=t,
            catalyst_cum_norm=c,
        )
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
        "tau_offset_days": float(tau.squeeze().detach().cpu().item()),
        "effective_tau_days": float(states["effective_tau_days"].squeeze().detach().cpu().item()),
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

    out["column_height_m"] = round_control_value(float(height), MAX_CONTROL_DECIMALS)
    out["column_inner_diameter_m"] = round_control_value(float(diameter), MAX_CONTROL_DECIMALS)

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
                    out["column_inner_diameter_m"] = round_control_value(float(diameter), MAX_CONTROL_DECIMALS)
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
                    out["column_inner_diameter_m"] = round_control_value(float(diameter), MAX_CONTROL_DECIMALS)
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
                f"\nThe current hidden model input is feed_mass_kg={int(estimated_mass)} kg."
            )
        else:
            note = (
                "Feed mass is hidden from the UI and auto-estimated from column volume plus a calibrated "
                f"Rosin-Rammler sphere-loading approximation using P80={p80_in:.1f} in and shape n={loading_estimate['shape_n']:.1f}. "
                f"\nThe resulting hidden model input is feed_mass_kg={int(estimated_mass)} kg."
            )
    else:
        note = (
            "Feed mass is hidden from the UI and falls back to the internal default estimate until "
            f"a valid column height and diameter define the column volume. \nThe current hidden model input is feed_mass_kg={int(estimated_mass)} kg."
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
    step_days = float(INTERACTIVE_PLOT_STEP_DAYS)
    max_day = float(np.clip(max_day, 100.0, 2500.0))
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


def _balance_grouped_with_gangue_priority(
    grouped_columns: List[str],
    start_values: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
    target_sum: float,
) -> np.ndarray:
    values = np.clip(np.asarray(start_values, dtype=float), min_values, max_values)
    target = float(np.clip(float(target_sum), float(np.sum(min_values)), float(np.sum(max_values))))
    if len(grouped_columns) == 0:
        return values

    gangue_priority_idx = (
        grouped_columns.index(GROUPED_GANGUE_PRIORITY_COLUMN)
        if GROUPED_GANGUE_PRIORITY_COLUMN in grouped_columns
        else None
    )
    if gangue_priority_idx is not None:
        non_gangue_sum = float(np.sum(values) - values[gangue_priority_idx])
        desired_gangue = target - non_gangue_sum
        values[gangue_priority_idx] = float(
            np.clip(desired_gangue, min_values[gangue_priority_idx], max_values[gangue_priority_idx])
        )

    residual = target - float(np.sum(values))
    if abs(residual) > 1e-8:
        values = _distribute_with_priority(
            start_values=values,
            min_values=min_values,
            max_values=max_values,
            target_sum=target,
            priority_idx=gangue_priority_idx,
            priority_share=GROUPED_GANGUE_PRIORITY_SHARE,
        )
    return np.clip(values, min_values, max_values)


def resolve_grouped_balance(
    ui_values: Dict[str, float],
    input_specs: Dict[str, Dict[str, float]],
    grouped_model_columns: List[str],
    grouped_inactive_columns: List[str],
    target_total: float = 100.0,
    fixed_inactive_values: Dict[str, float] | None = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    active_grouped_values = {
        column: float(ui_values.get(column, input_specs[column]["default"]))
        for column in grouped_model_columns
    }
    active_sum = float(sum(active_grouped_values.values()))
    fixed_inactive_values = dict(fixed_inactive_values or {})

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

    inactive_target = float(target_total) - active_sum
    clipped_fixed_inactive_values: Dict[str, float] = {}
    for column in grouped_inactive_columns:
        if column not in fixed_inactive_values:
            continue
        spec = input_specs[column]
        clipped_fixed_inactive_values[column] = float(
            np.clip(float(fixed_inactive_values[column]), float(spec["min"]), float(spec["max"]))
        )

    flexible_inactive_columns = [
        column for column in grouped_inactive_columns if column not in clipped_fixed_inactive_values
    ]
    fixed_inactive_sum = float(sum(clipped_fixed_inactive_values.values()))
    flexible_target = inactive_target - fixed_inactive_sum

    lower_sum = fixed_inactive_sum
    upper_sum = fixed_inactive_sum
    inactive_grouped_values = dict(clipped_fixed_inactive_values)

    if len(flexible_inactive_columns) > 0:
        min_values = np.asarray([float(input_specs[column]["min"]) for column in flexible_inactive_columns], dtype=float)
        max_values = np.asarray([float(input_specs[column]["max"]) for column in flexible_inactive_columns], dtype=float)
        default_values = np.asarray([float(input_specs[column]["default"]) for column in flexible_inactive_columns], dtype=float)
        flexible_lower_sum = float(np.sum(min_values))
        flexible_upper_sum = float(np.sum(max_values))
        lower_sum += flexible_lower_sum
        upper_sum += flexible_upper_sum
        balanced_values = _balance_grouped_with_gangue_priority(
            grouped_columns=flexible_inactive_columns,
            start_values=default_values,
            min_values=min_values,
            max_values=max_values,
            target_sum=flexible_target,
        )
        inactive_grouped_values.update(
            {column: float(value) for column, value in zip(flexible_inactive_columns, balanced_values)}
        )

    inactive_sum = float(sum(inactive_grouped_values.values()))
    total_sum = active_sum + inactive_sum
    feasible = lower_sum - 1e-6 <= inactive_target <= upper_sum + 1e-6
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

    grouped_model_columns = [c for c in GROUPED_MODEL_COLUMNS if c in input_specs]
    grouped_inactive_columns = [c for c in GROUPED_INACTIVE_COLUMNS if c in input_specs]
    grouped_regressions = load_grouped_modal_regressions()
    inferred_primary, inferred_secondary, inferred_mixed, inferred_oxides = infer_derived_mineralogy_modals(
        primary_copper_sulf=float(derived_values[PRIMARY_SULFIDES_EQUIV_COL]),
        secondary_copper_sulf=float(derived_values[SECONDARY_SULFIDES_EQUIV_COL]),
        copper_oxides=float(derived_values[OXIDES_EQUIV_COL]),
        regressions=grouped_regressions,
    )
    inferred_grouped_values = {
        "grouped_primary_copper_sulfides": inferred_primary,
        "grouped_secondary_copper_sulfides": inferred_secondary,
        "grouped_mixed_copper_ores": inferred_mixed,
        "grouped_copper_oxides": inferred_oxides,
    }
    fixed_inactive_grouped_values: Dict[str, float] = {}
    for column, inferred_value in inferred_grouped_values.items():
        if column not in input_specs:
            continue
        clipped_value = float(
            np.clip(
                float(inferred_value),
                float(input_specs[column]["min"]),
                float(input_specs[column]["max"]),
            )
        )
        if column in grouped_model_columns:
            ui_values[column] = clipped_value
        elif column in grouped_inactive_columns:
            fixed_inactive_grouped_values[column] = clipped_value

    grouped_values, inactive_grouped_values, grouped_balance_state = resolve_grouped_balance(
        ui_values=ui_values,
        input_specs=input_specs,
        grouped_model_columns=grouped_model_columns,
        grouped_inactive_columns=grouped_inactive_columns,
        target_total=100.0,
        fixed_inactive_values=fixed_inactive_grouped_values,
    )
    
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
    ctrl_cap, cat_cap = BASE.compute_sample_leach_caps(
        cu_oxides_equiv=float(static_values.get(OXIDES_EQUIV_COL, np.nan)),
        cu_secondary_equiv=float(static_values.get(SECONDARY_SULFIDES_EQUIV_COL, np.nan)),
        cu_primary_equiv=float(static_values.get(PRIMARY_SULFIDES_EQUIV_COL, np.nan)),
        material_size_p80_in=float(static_values.get("material_size_p80_in", np.nan)),
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
        "control_cap": float(ctrl_cap),
        "catalyzed_cap": float(cat_cap),
        "control_pred_mean": np.mean(ctrl_stack, axis=0),
        "control_pred_p10": ctrl_p10,
        "control_pred_p90": ctrl_p90,
        "catalyzed_pred_mean": np.mean(cat_stack, axis=0),
        "catalyzed_pred_p10": cat_p10,
        "catalyzed_pred_p90": cat_p90,
        "tau_offset_days_mean": float(np.mean([p["tau_offset_days"] for p in member_preds])),
        "effective_tau_days_mean": float(np.mean([p["effective_tau_days"] for p in member_preds])),
        "temp_days_mean": float(np.mean([p["temp_days"] for p in member_preds])),
        "kappa_mean": float(np.mean([p["kappa"] for p in member_preds])),
        "aging_strength_mean": float(np.mean([p.get("aging_strength", np.nan) for p in member_preds])),
        "n_members": int(len(member_preds)),
    }


def make_prediction_figure(pred: Dict[str, Any], max_day: float, theme: str = "dark") -> go.Figure:
    theme_key = theme if theme in PREDICTION_FIGURE_THEMES else "dark"
    theme_settings = PREDICTION_FIGURE_THEMES[theme_key]
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
        f"<br><span style='color:{CONTROL_CURVE_COLOR}'>Control Cu Rec = %{{customdata[0]:.1f}}</span>"
        f"<br><span style='color:{CATALYZED_CURVE_COLOR}'>Catalyzed Cu Rec = %{{customdata[1]:.1f}}</span>"
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
            fillcolor=hex_to_rgba(CONTROL_CURVE_COLOR, float(theme_settings["band_alpha"])),
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
            fillcolor=hex_to_rgba(CATALYZED_CURVE_COLOR, float(theme_settings["band_alpha"])),
            name=f"Catalyzed {band_label}",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_cat,
            y=catalyzed_mean[cat_mask],
            mode="lines",
            line={"color": hex_to_rgba(CATALYZED_CURVE_COLOR, float(theme_settings["glow_outer_alpha"])), "width": 11},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_cat,
            y=catalyzed_mean[cat_mask],
            mode="lines",
            line={"color": hex_to_rgba(CATALYZED_CURVE_COLOR, float(theme_settings["glow_inner_alpha"])), "width": 7},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_cat,
            y=catalyzed_mean[cat_mask],
            mode="lines",
            line={"color": CATALYZED_CURVE_COLOR, "width": 3},
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
            line={"color": hex_to_rgba(CONTROL_CURVE_COLOR, float(theme_settings["glow_outer_alpha"])), "width": 11},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=control_mean,
            mode="lines",
            line={"color": hex_to_rgba(CONTROL_CURVE_COLOR, float(theme_settings["glow_inner_alpha"])), "width": 7},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=control_mean,
            mode="lines",
            line={"color": CONTROL_CURVE_COLOR, "width": 3},
            name="Control Ensemble Mean",
            customdata=hover_customdata,
            hovertemplate=hover_template,
        )
    )

    full_annotation_text = (
        f"Catalyst used: {pred['weekly_catalyst_gt_week']:.2f} g/t/w"
        f"<br>Catalyst addition start day: {pred['catalyst_addition_start_day']:.0f}"
        f"<br>Irrigation rate: {pred['irrigation_rate_input_l_m2_h']:.2f} L/h/m2"
        f"<br>Confidence band: {band_label}"
        f"<br>Assumption: constant catalyst addition from selected start day"
        f"<br>Assumption: constant irrigation over the plotted horizon"
        f"<br>Members loaded: {pred['n_members']}"
    )
    compact_annotation_text = (
        f"Confidence band: {band_label}"
        f"<br>Catalyst addition start day: {pred['catalyst_addition_start_day']:.0f}"
    )
    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        text=full_annotation_text,
        showarrow=False,
        align="left",
        bgcolor=str(theme_settings["annotation_bgcolor"]),
        bordercolor=str(theme_settings["annotation_bordercolor"]),
        borderwidth=1,
        font={"size": 12, "color": str(theme_settings["font_color"])},
    )
    fig.update_layout(
        template=str(theme_settings["template"]),
        title={
            "text": "Interactive Ensemble Prediction (v10)",
            "font": {"size": 24, "color": str(theme_settings["font_color"])},
        },
        autosize=True,
        paper_bgcolor=str(theme_settings["paper_bgcolor"]),
        plot_bgcolor=str(theme_settings["plot_bgcolor"]),
        font={"family": APP_FONT_FAMILY, "color": str(theme_settings["font_color"])},
        margin={"l": 62, "r": 24, "t": 66, "b": 56},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": str(theme_settings["font_color"])},
        },
        hoverlabel={
            "bgcolor": str(theme_settings["hover_bgcolor"]),
            "bordercolor": str(theme_settings["hover_bordercolor"]),
            "font": {"color": str(theme_settings["font_color"])},
        },
        meta={
            "png_export_filename": "interactive_ensemble_prediction_v10_300dpi",
            "png_export_annotation_full": full_annotation_text,
            "png_export_annotation_compact": compact_annotation_text,
            "png_export_theme": theme_key,
        },
    )
    fig.update_xaxes(
        title_text="Leach duration (days)",
        range=[0, float(max_day)],
        autorange=False,
        showgrid=True,
        gridcolor=str(theme_settings["grid_color"]),
        linecolor=str(theme_settings["axis_line_color"]),
        tickfont={"color": str(theme_settings["font_color"])},
        title_font={"color": str(theme_settings["font_color"])},
        zeroline=False,
        automargin=True,
    )
    fig.update_yaxes(
        title_text="Cu Recovery %",
        range=[0, 80],
        autorange=False,
        showgrid=True,
        gridcolor=str(theme_settings["grid_color"]),
        linecolor=str(theme_settings["axis_line_color"]),
        tickfont={"color": str(theme_settings["font_color"])},
        title_font={"color": str(theme_settings["font_color"])},
        zeroline=False,
        automargin=True,
    )
    if 0.0 < float(pred["catalyst_addition_start_day"]) < float(max_day):
        fig.add_vline(
            x=float(pred["catalyst_addition_start_day"]),
            line_width=1,
            line_dash="dash",
            line_color=str(theme_settings["vline_color"]),
        )
    return fig


def viewport_bucket(viewport_state: Dict[str, Any] | None) -> str:
    if isinstance(viewport_state, dict):
        bucket = str(viewport_state.get("bucket", "")).strip().lower()
        if bucket in {"desktop", "mobile", "narrow"}:
            return bucket
        width = viewport_state.get("width")
        if isinstance(width, (int, float)):
            width_value = float(width)
            if width_value <= NARROW_LAYOUT_BREAKPOINT_PX:
                return "narrow"
            if width_value <= MOBILE_LAYOUT_BREAKPOINT_PX:
                return "mobile"
    return "desktop"


def make_display_figure(
    figure_source: Dict[str, Any] | go.Figure,
    viewport_state: Dict[str, Any] | None = None,
) -> go.Figure:
    figure = go.Figure(figure_source)
    bucket = viewport_bucket(viewport_state)
    is_mobile = bucket in {"mobile", "narrow"}
    is_narrow = bucket == "narrow"

    figure.layout.width = None
    figure.layout.height = None
    figure.update_layout(autosize=True)
    figure.update_xaxes(automargin=True, autorange=False)
    figure.update_yaxes(automargin=True, autorange=False)

    meta = figure.layout.meta if isinstance(figure.layout.meta, dict) else {}
    if figure.layout.annotations:
        annotation = figure.layout.annotations[0]
        annotation_font = annotation.font.to_plotly_json() if annotation.font else {}
        if is_mobile:
            compact_text = meta.get("png_export_annotation_compact")
            if compact_text:
                annotation.text = compact_text
            annotation_font["size"] = 10 if is_narrow else 11
            annotation.x = 0.02
            annotation.y = 0.985
        else:
            full_text = meta.get("png_export_annotation_full")
            if full_text:
                annotation.text = full_text
            annotation_font["size"] = 12
            annotation.x = 0.01
            annotation.y = 0.99
        annotation.font = annotation_font

    if is_mobile:
        figure.update_layout(
            title_text="Ensemble Prediction" if is_narrow else "Interactive Ensemble Prediction",
            title_font_size=16 if is_narrow else 18,
            margin={
                "l": 48 if is_narrow else 54,
                "r": 14 if is_narrow else 16,
                "t": 56 if is_narrow else 60,
                "b": 50 if is_narrow else 54,
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.01,
                "xanchor": "left",
                "x": 0.0,
                "bgcolor": "rgba(0,0,0,0)",
            },
            legend_font_size=10 if is_narrow else 11,
            hoverlabel_font_size=10 if is_narrow else 11,
        )
        figure.update_xaxes(
            title_font_size=12,
            tickfont_size=10,
            nticks=5 if is_narrow else 6,
        )
        figure.update_yaxes(
            title_font_size=12,
            tickfont_size=10,
            nticks=6 if is_narrow else 7,
        )
        for trace in figure.data:
            if not isinstance(trace, go.Scatter):
                continue
            if str(getattr(trace, "fill", "") or "") == "tonexty":
                trace.showlegend = False
            if trace.name == "Catalyzed Ensemble Mean":
                trace.name = "Catalyzed"
            elif trace.name == "Control Ensemble Mean":
                trace.name = "Control"
            line_width = getattr(getattr(trace, "line", None), "width", None)
            if line_width is None:
                continue
            width_value = float(line_width)
            if width_value >= 11.0 - 1e-9:
                trace.line.width = 8 if is_narrow else 9
            elif width_value >= 7.0 - 1e-9:
                trace.line.width = 5 if is_narrow else 6
            elif width_value >= 3.0 - 1e-9:
                trace.line.width = 2.5 if is_narrow else 2.75
    else:
        figure.update_layout(
            title_text="Interactive Ensemble Prediction (v10)",
            title_font_size=24,
            margin={"l": 62, "r": 24, "t": 66, "b": 56},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1.0,
                "bgcolor": "rgba(0,0,0,0)",
            },
            legend_font_size=12,
            hoverlabel_font_size=12,
        )
        figure.update_xaxes(title_font_size=14, tickfont_size=12)
        figure.update_yaxes(title_font_size=14, tickfont_size=12)

    return figure


def build_prediction_export_figure_payload(pred: Dict[str, Any], max_day: float) -> Dict[str, Any]:
    light_figure = make_prediction_figure(pred, max_day, theme="light")
    dark_figure = make_prediction_figure(pred, max_day, theme="dark")
    return {
        "light": json.loads(light_figure.to_json()),
        "dark": json.loads(dark_figure.to_json()),
    }


def build_prediction_export_days(max_day: float, export_step_days: float = 7.0) -> np.ndarray:
    export_step_days = float(max(export_step_days, 1.0))
    max_day = float(np.clip(max_day, 100.0, 2500.0))
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
        {
            "section": "model_metadata",
            "parameter": "model_logic_version",
            "label": "Model logic version",
            "value": str(getattr(BASE, "MODEL_LOGIC_VERSION", "unknown")),
        },
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
        {
            "section": "response_latents",
            "parameter": "tau_offset_days_mean",
            "label": "Tau offset mean (days)",
            "value": float(pred["tau_offset_days_mean"]),
        },
        {
            "section": "response_latents",
            "parameter": "effective_tau_days_mean",
            "label": "Effective tau mean (days)",
            "value": float(pred["effective_tau_days_mean"]),
        },
        {
            "section": "response_latents",
            "parameter": "temp_days_mean",
            "label": "Temp mean (days)",
            "value": float(pred["temp_days_mean"]),
        },
        {
            "section": "response_latents",
            "parameter": "kappa_mean",
            "label": "Kappa mean",
            "value": float(pred["kappa_mean"]),
        },
        {
            "section": "response_latents",
            "parameter": "aging_strength_mean",
            "label": "Aging strength mean",
            "value": float(pred["aging_strength_mean"]),
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
        autosize_excel_sheet(writer, "inputs", inputs_df)
        autosize_excel_sheet(writer, "predictions_7d", predictions_df)
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
            f"(inside expected {GEOMETRY_BULK_DENSITY_MIN_T_M3:.1f}-{GEOMETRY_BULK_DENSITY_MAX_T_M3:.1f} t/m3)"
            if GEOMETRY_BULK_DENSITY_MIN_T_M3 - GEOMETRY_TOL
            <= apparent_density
            <= GEOMETRY_BULK_DENSITY_MAX_T_M3 + GEOMETRY_TOL
            else f"(outside expected {GEOMETRY_BULK_DENSITY_MIN_T_M3:.1f}-{GEOMETRY_BULK_DENSITY_MAX_T_M3:.1f} t/m3)"
        )
        if np.isfinite(apparent_density)
        else f"(expected {GEOMETRY_BULK_DENSITY_MIN_T_M3:.1f}-{GEOMETRY_BULK_DENSITY_MAX_T_M3:.1f} t/m3)"
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
            f"Tau offset mean: {int(pred['tau_offset_days_mean'])} days | "
            f"Effective tau mean: {int(pred['effective_tau_days_mean'])} days | "
            f"Temp mean: {int(pred['temp_days_mean'])} days | "
            f"Kappa mean: {pred['kappa_mean']:.2f} | "
            f"Aging strength mean: {pred['aging_strength_mean']:.2f}"
        ),
        html.Div(
            "Parameter meanings: Tau offset is the learned catalyst-response delay from the model, "
            "and effective tau is that delay after adding the selected catalyst start day. "
            "Tau offset, Temp, and Kappa are learned from both ore characteristics and the active "
            "catalyst inventory implied by the plotted catalyst schedule. "
            "Temp is the width or softness of uplift turn-on; Kappa is the catalyst-response rate "
            "showing how fast active catalyst translates into uplift; Aging strength is the decay "
            "strength showing how fast older catalyst additions lose effectiveness over time."
        ),
        html.Div(
            f"Catalyzed curve assumption: constant catalyst addition starting at day "
            f"{pred['catalyst_addition_start_day']:.0f}, using the selected weekly rate "
            "converted internally to cumulative kg/t. Changing that schedule changes the active "
            "catalyst inventory seen by the model."
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
            f"Resolved chemistry: Acid {resolved_chemistry['acid']:.1f}% | "
            f"Cyanide {resolved_chemistry['cyanide']:.1f}% | "
            f"Residual {resolved_chemistry['residual']:.1f}%"
        ),
        html.Div(
            f"Internal copper equivalents: "
            f"Primary {derived_values[PRIMARY_SULFIDES_EQUIV_COL]:.3f} | "
            f"Secondary {derived_values[SECONDARY_SULFIDES_EQUIV_COL]:.3f} | "
            f"Oxides {derived_values[OXIDES_EQUIV_COL]:.3f}"
        ),
        html.Div(
            f"Derived leach caps used at inference: "
            f"Control {pred['control_cap']:.1f}% | "
            f"Catalyzed {pred['catalyzed_cap']:.1f}%"
        ),
    ]
    if SULFIDES_EQUIV_COL in BASE.STATIC_PREDICTOR_COLUMNS:
        summary_lines.append(
            html.Div(f"Internal total sulfides equivalent: {derived_values[SULFIDES_EQUIV_COL]:.3f}")
        )
    if needs_fe_input():
        ratio_fragments = [f"Fe {ui_values.get(FE_INPUT_COL, 0.0):.2f}%"]
        if FE_CU_RATIO_COL in BASE.STATIC_PREDICTOR_COLUMNS:
            ratio_fragments.append(f"Fe/Cu {derived_values[FE_CU_RATIO_COL]:.1f}")
        if CU_FE_RATIO_COL in BASE.STATIC_PREDICTOR_COLUMNS:
            ratio_fragments.append(f"Cu/Fe {derived_values[CU_FE_RATIO_COL]:.1f}")
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


def build_geometry_guardrail_summary(
    geometry_guardrail_state: Dict[str, Any],
) -> html.Div:
    ratio_ok = bool(geometry_guardrail_state.get("ratio_ok", True))
    density_ok = bool(geometry_guardrail_state.get("density_ok", True))
    note_text = str(geometry_guardrail_state.get("note", "")).strip()
    if not note_text:
        return html.Div()
    return html.Div(
        html.Div(
            note_text,
            style={
                "fontSize": "12px",
                "color": "var(--app-warning)" if (not ratio_ok or not density_ok) else "var(--app-text-secondary)",
                "lineHeight": "1.5",
            },
        ),
        style={
            "marginTop": "10px",
            "padding": "12px 14px",
            "border": "1px solid var(--app-border)",
            "borderRadius": "10px",
            "background": "var(--app-panel-alt-bg)",
            "boxShadow": "inset 0 1px 0 rgba(255, 255, 255, 0.03)",
        },
    )


def build_assumptions_section(content: html.Div) -> html.Div:
    return html.Div(
        [
            section_title("Assumptions And Calculations", margin_top="0"),
            html.Div(
                "Internal conversions, hidden model inputs, geometry guardrails, and resolved chemistry are listed below.",
                style={"fontSize": "12px", "color": "var(--app-text-secondary)", "marginBottom": "12px"},
            ),
            content,
        ],
        className="assumptions-section",
        style={
            "padding": "24px 28px 32px 28px",
            "color": "var(--app-text-secondary)",
            "fontSize": "14px",
            "lineHeight": "1.7",
            "borderTop": "1px solid var(--app-border)",
            "background": "var(--app-assumptions-bg)",
        },
    )


def build_loading_prediction_summary() -> html.Div:
    return build_assumptions_section(
        html.Div(
            [
                html.Div(
                    "Loading default prediction...",
                    style={"color": "var(--app-text)", "fontWeight": 700, "marginBottom": "6px"},
                ),
                html.Div(
                    "The app starts first and then loads the saved ensemble members in the background for the initial prediction.",
                ),
            ]
        )
    )


def build_placeholder_figure(theme: str = "dark", message: str = "Loading default prediction...") -> go.Figure:
    theme_key = theme if theme in PREDICTION_FIGURE_THEMES else "dark"
    theme_settings = PREDICTION_FIGURE_THEMES[theme_key]
    figure = go.Figure()
    figure.update_layout(
        template=theme_settings["template"],
        autosize=True,
        paper_bgcolor=theme_settings["paper_bgcolor"],
        plot_bgcolor=theme_settings["plot_bgcolor"],
        font={"family": APP_FONT_FAMILY, "color": theme_settings["font_color"]},
        margin={"l": 40, "r": 28, "t": 40, "b": 40},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 18, "color": theme_settings["font_color"]},
            }
        ],
        showlegend=False,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def build_placeholder_export_figure_payload() -> Dict[str, Any]:
    return {
        "light": build_placeholder_figure(theme="light"),
        "dark": build_placeholder_figure(theme="dark"),
    }


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
        style={
            "fontSize": "15px",
            "fontWeight": 700,
            "marginTop": margin_top,
            "marginBottom": "10px",
            "color": "var(--app-text)",
            "letterSpacing": "0.01em",
        },
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
    default_ui_values[ACID_SOLUBLE_COL] = float(default_chemistry["acid"])
    default_ui_values[RESIDUAL_CPY_COL] = float(default_chemistry["residual"])
    default_ui_values[CYANIDE_SOLUBLE_COL] = float(default_chemistry["cyanide"])

    max_day_spec = finalize_numeric_spec(
        {
        "min": 100.0,
        "max": float(BASE.CONFIG.get("ensemble_plot_target_day", 2500.0)),
        "default": float(BASE.CONFIG.get("ensemble_plot_target_day", 2500.0)),
        "step": 100.0,
        },
        step_value=INPUT_ARROW_STEPS["max-day"],
    )
    confidence_interval_spec = finalize_numeric_spec(
        {"min": 60.0, "max": 95.0, "default": 80.0, "step": 5.0},
        step_value=INPUT_ARROW_STEPS["confidence-interval"],
    )
    catalyst_start_day_spec = finalize_numeric_spec(
        {"min": 0.0, "max": 700.0, "default": 200.0, "step": 10.0},
        step_value=INPUT_ARROW_STEPS["catalyst-addition-start-day"],
    )
    control_specs_by_id: Dict[str, Dict[str, float]] = {
        "max-day": max_day_spec,
        "weekly-catalyst-gt-week": dict(weekly_spec),
        "confidence-interval": confidence_interval_spec,
        "irrigation-rate-l-m2-h": dict(irrigation_spec),
        "catalyst-addition-start-day": catalyst_start_day_spec,
    }
    for column in EDITABLE_INPUT_COLUMNS:
        slider_spec = {**dict(input_specs[column]), "default": float(default_ui_values[column])}
        if column == ACID_SOLUBLE_COL:
            slider_spec["max"] = float(default_chemistry["acid_max"])
        elif column == RESIDUAL_CPY_COL:
            slider_spec["max"] = float(default_chemistry["residual_max"])
        elif column == "column_inner_diameter_m":
            slider_spec["min"] = diameter_ratio_guardrail_min(default_ui_values.get("material_size_p80_in"), input_specs)
        if column in INPUT_ARROW_STEPS:
            slider_spec = apply_input_arrow_step(slider_spec, INPUT_ARROW_STEPS[column])
        control_specs_by_id[predictor_component_id(column)] = slider_spec

    default_ui_values = {
        column: float(control_specs_by_id[predictor_component_id(column)]["default"])
        for column in EDITABLE_INPUT_COLUMNS
    }
    default_ui_values = resolve_ui_state(
        tuple(default_ui_values[column] for column in EDITABLE_INPUT_COLUMNS),
        input_specs,
    )
    default_ui_values, default_geometry_guardrail_state = apply_geometry_input_limits(default_ui_values, input_specs)
    default_ui_values, default_feed_mass_state = resolve_internal_feed_mass(default_ui_values, input_specs)
    default_static_values, default_derived_values, _default_inactive_grouped_values, default_grouped_balance_state, default_chemistry = resolve_model_static_values(default_ui_values, input_specs)
    default_geometry_values = geometry_predictor_values(default_static_values)
    default_ui_values[ACID_SOLUBLE_COL] = float(default_chemistry["acid"])
    default_ui_values[RESIDUAL_CPY_COL] = float(default_chemistry["residual"])
    default_ui_values[CYANIDE_SOLUBLE_COL] = float(default_chemistry["cyanide"])

    top_controls = [
        slider_block(
            label="Max day",
            component_id="max-day",
            spec=dict(control_specs_by_id["max-day"]),
        ),
        slider_block(
            label="Catalyst used (g/t/week)",
            component_id="weekly-catalyst-gt-week",
            spec=dict(control_specs_by_id["weekly-catalyst-gt-week"]),
        ),
        slider_block(
            label="Confidence interval (%)",
            component_id="confidence-interval",
            spec=dict(control_specs_by_id["confidence-interval"]),
        ),
        slider_block(
            label="Irrigation rate (L/h/m2)",
            component_id="irrigation-rate-l-m2-h",
            spec=dict(control_specs_by_id["irrigation-rate-l-m2-h"]),
        ),
        slider_block(
            label="Catalyst addition start day",
            component_id="catalyst-addition-start-day",
            spec=dict(control_specs_by_id["catalyst-addition-start-day"]),
        ),
    ]
    for column in TOP_DISPLAY_COLUMNS:
        top_controls.append(
            slider_block(
                label=display_label(column),
                component_id=predictor_component_id(column),
                spec=dict(control_specs_by_id[predictor_component_id(column)]),
            )
        )

    main_controls = [
    ]
    for column in MAIN_EDITABLE_COLUMNS:
        main_controls.append(
            slider_block(
                label=display_label(column),
                component_id=predictor_component_id(column),
                spec=dict(control_specs_by_id[predictor_component_id(column)]),
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
            spec=dict(control_specs_by_id[predictor_component_id(column)]),
            caption_text=GROUPED_INPUT_CAPTIONS.get(column, ""),
        )
        for column in grouped_model_columns
    ]
    default_applied_controls = {
        "max_day": normalize_control_value(float(max_day_spec["default"]), max_day_spec),
        "weekly_catalyst_gt_week": normalize_control_value(float(weekly_spec["default"]), weekly_spec),
        "confidence_interval": normalize_control_value(float(confidence_interval_spec["default"]), confidence_interval_spec),
        "irrigation_rate_l_m2_h": normalize_control_value(float(irrigation_spec["default"]), irrigation_spec),
        "catalyst_addition_start_day": normalize_control_value(float(catalyst_start_day_spec["default"]), catalyst_start_day_spec),
        "predictor_values": {column: float(default_ui_values[column]) for column in EDITABLE_INPUT_COLUMNS},
    }
    default_max_day = float(max_day_spec["default"])
    default_display_figure = make_display_figure(build_placeholder_figure(theme="light"))
    default_export_figure_payload = build_placeholder_export_figure_payload()
    default_prediction_summary = build_loading_prediction_summary()

    @lru_cache(maxsize=32)
    def get_cached_prediction_bundle(
        max_day_value: float,
        weekly_catalyst_gt_week_value: float,
        confidence_interval_value: float,
        irrigation_rate_l_m2_h_value: float,
        catalyst_addition_start_day_value: float,
        predictor_values_value: Tuple[float, ...],
    ) -> Dict[str, Any]:
        members, _model_dir, _skipped_paths = get_cached_saved_member_models(project_root)
        predictor_values = tuple(float(v) for v in predictor_values_value)
        ui_values = resolve_ui_state(predictor_values, input_specs)
        ui_values, geometry_guardrail_state = apply_geometry_input_limits(ui_values, input_specs)
        ui_values, feed_mass_state = resolve_internal_feed_mass(ui_values, input_specs)
        static_values, derived_values, inactive_grouped_values, grouped_balance_state, resolved_chemistry = resolve_model_static_values(ui_values, input_specs)
        geometry_values = geometry_predictor_values(static_values)
        resolved_max_day = float(max_day_value or BASE.CONFIG.get("ensemble_plot_target_day", 2500.0))
        pred = predict_ensemble_curves(
            members=members,
            static_values=static_values,
            weekly_catalyst_gt_week=float(weekly_catalyst_gt_week_value or 0.0),
            irrigation_rate_l_m2_h=float(irrigation_rate_l_m2_h_value or 0.0),
            catalyst_addition_start_day=float(catalyst_addition_start_day_value or 0.0),
            confidence_interval_high=float(confidence_interval_value or 90.0),
            max_day=resolved_max_day,
        )
        return {
            "summary": build_assumptions_section(
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
            "export_figure_payload": build_prediction_export_figure_payload(pred, resolved_max_day),
            "loading_token": f"updated-{resolved_max_day:.1f}-{pred['n_members']}",
            "resolved_max_day": resolved_max_day,
            "pred": pred,
            "ui_values": ui_values,
            "static_values": static_values,
            "derived_values": derived_values,
            "geometry_values": geometry_values,
            "inactive_grouped_values": inactive_grouped_values,
            "grouped_balance_state": grouped_balance_state,
            "resolved_chemistry": resolved_chemistry,
        }

    app = Dash(
        __name__,
        title="Rosetta | Jetti Resources",
        assets_folder=APP_ASSETS_DIR,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
            {"name": "theme-color", "content": APP_COLOR_BG_LIGHT},
        ],
    )
    header_logo_mime_type = asset_mime_type(HEADER_LOGO_ASSET)
    generated_favicon_asset = ensure_generated_favicon_asset(HEADER_LOGO_ASSET)
    favicon_href = (
        inline_asset_data_url(generated_favicon_asset, "image/x-icon")
        if generated_favicon_asset is not None
        else None
    )
    header_logo_data_url = inline_asset_data_url(HEADER_LOGO_ASSET, header_logo_mime_type)
    apple_touch_icon_href = header_logo_data_url
    header_logo_src = header_logo_data_url or app.get_asset_url(asset_public_name(HEADER_LOGO_ASSET))
    favicon_tags = "{%favicon%}" if not favicon_href else (
        f'<link rel="icon" type="image/x-icon" href="{favicon_href}">\n'
        f'            <link rel="shortcut icon" type="image/x-icon" href="{favicon_href}">'
    )
    apple_touch_icon_tag = ""
    if apple_touch_icon_href:
        apple_touch_icon_tag = (
            f'\n            <link rel="apple-touch-icon" href="{apple_touch_icon_href}">'
        )
    app.index_string = f"""
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {favicon_tags}{apple_touch_icon_tag}
            {{%css%}}
            <script>
                (function() {{
                    if (typeof window === "undefined" || typeof document === "undefined") {{
                        return;
                    }}
                    const setThemeColor = (themeName) => {{
                        const meta = document.querySelector('meta[name="theme-color"]');
                        if (meta) {{
                            meta.setAttribute(
                                "content",
                                themeName === "dark" ? "{APP_COLOR_BG}" : "{APP_COLOR_BG_LIGHT}"
                            );
                        }}
                    }};
                    const applySystemTheme = (matchesDark) => {{
                        const themeName = matchesDark ? "dark" : "light";
                        document.documentElement.setAttribute("data-system-theme", themeName);
                        setThemeColor(themeName);
                    }};
                    const syncThemeColorFromApp = () => {{
                        const appRoot = document.getElementById("app-root");
                        if (appRoot) {{
                            if (appRoot.classList.contains("theme-dark")) {{
                                setThemeColor("dark");
                                return;
                            }}
                            if (appRoot.classList.contains("theme-light")) {{
                                setThemeColor("light");
                                return;
                            }}
                        }}
                        const systemTheme = document.documentElement.getAttribute("data-system-theme");
                        setThemeColor(systemTheme === "dark" ? "dark" : "light");
                    }};
                    try {{
                        const media = window.matchMedia("(prefers-color-scheme: dark)");
                        applySystemTheme(!!media.matches);
                        if (typeof media.addEventListener === "function") {{
                            media.addEventListener("change", function(event) {{
                                applySystemTheme(!!event.matches);
                            }});
                        }} else if (typeof media.addListener === "function") {{
                            media.addListener(function(event) {{
                                applySystemTheme(!!event.matches);
                            }});
                        }}
                    }} catch (error) {{
                        document.documentElement.setAttribute("data-system-theme", "light");
                        setThemeColor("light");
                    }}
                    try {{
                        const observer = new MutationObserver(syncThemeColorFromApp);
                        observer.observe(document.documentElement, {{
                            attributes: true,
                            attributeFilter: ["data-system-theme"],
                            subtree: false,
                        }});
                        const observeAppRoot = () => {{
                            const appRoot = document.getElementById("app-root");
                            if (!appRoot) {{
                                window.requestAnimationFrame(observeAppRoot);
                                return;
                            }}
                            observer.observe(appRoot, {{
                                attributes: true,
                                attributeFilter: ["class"],
                            }});
                            syncThemeColorFromApp();
                        }};
                        observeAppRoot();
                    }} catch (error) {{
                        syncThemeColorFromApp();
                    }}
                }})();
            </script>
            <style>
                * {{
                    box-sizing: border-box;
                }}
                :root {{
                    color-scheme: light;
                }}
                body {{
                    margin: 0;
                    background: {APP_COLOR_BG_LIGHT};
                    color: {APP_COLOR_TEXT_LIGHT};
                }}
                .app-root {{
                    --app-bg-gradient: linear-gradient(135deg, rgba(239, 245, 250, 0.98) 0%, rgba(231, 239, 246, 0.98) 52%, rgba(223, 233, 241, 0.98) 100%);
                    --app-header-bg: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 246, 250, 0.98) 100%);
                    --app-panel-bg: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(245, 249, 252, 0.98) 100%);
                    --app-panel-alt-bg: rgba(240, 246, 250, 0.98);
                    --app-graph-panel-bg: linear-gradient(145deg, rgba(240, 246, 250, 0.98) 0%, rgba(231, 239, 246, 0.98) 100%);
                    --app-assumptions-bg: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(242, 247, 251, 0.98) 100%);
                    --app-border: rgba(92, 126, 145, 0.22);
                    --app-text: #102638;
                    --app-text-secondary: #4d6577;
                    --app-text-muted: #6f8492;
                    --app-accent: #14b8c6;
                    --app-accent-soft: rgba(20, 184, 198, 0.12);
                    --app-accent-contrast: #082131;
                    --app-input-bg: #ffffff;
                    --app-input-border: rgba(92, 126, 145, 0.28);
                    --app-warning: #c2410c;
                    --readonly-track: rgba(111, 142, 161, 0.18);
                    --readonly-fill: #14b8c6;
                    --app-panel-shadow: 0 18px 42px rgba(118, 136, 150, 0.18);
                    --modebar-bg: rgba(255, 255, 255, 0.84);
                    --modebar-icon: #51606f;
                    background: var(--app-bg-gradient);
                    color: var(--app-text);
                    min-height: 100vh;
                }}
                .app-shell {{
                    min-width: 0;
                }}
                .plot-panel {{
                    min-width: 0;
                    overflow-x: hidden;
                }}
                .prediction-graph,
                .prediction-graph > div,
                .prediction-graph .js-plotly-plot,
                .prediction-graph .plot-container,
                .prediction-graph .svg-container {{
                    width: 100% !important;
                    min-width: 0 !important;
                    max-width: 100% !important;
                }}
                .prediction-graph .main-svg {{
                    max-width: 100% !important;
                }}
                .theme-dark,
                [data-system-theme="dark"] .app-root:not(.theme-light):not(.theme-dark) {{
                    color-scheme: dark;
                    --app-bg-gradient: linear-gradient(135deg, rgba(6, 18, 29, 0.98) 0%, rgba(10, 31, 49, 0.98) 52%, rgba(7, 19, 31, 0.98) 100%);
                    --app-header-bg: linear-gradient(135deg, rgba(8, 23, 38, 0.98) 0%, rgba(10, 31, 49, 0.98) 100%);
                    --app-panel-bg: linear-gradient(180deg, rgba(9, 27, 42, 0.98) 0%, rgba(8, 22, 35, 0.98) 100%);
                    --app-panel-alt-bg: rgba(12, 34, 52, 0.96);
                    --app-graph-panel-bg: linear-gradient(145deg, rgba(6, 18, 29, 0.98) 0%, rgba(10, 31, 49, 0.98) 100%);
                    --app-assumptions-bg: linear-gradient(180deg, rgba(9, 27, 42, 0.98) 0%, rgba(7, 20, 32, 0.98) 100%);
                    --app-border: rgba(98, 159, 184, 0.22);
                    --app-text: #eef8ff;
                    --app-text-secondary: #9cb8c9;
                    --app-text-muted: #6f8ea1;
                    --app-accent: #2fd5d0;
                    --app-accent-soft: rgba(47, 213, 208, 0.14);
                    --app-accent-contrast: #06121d;
                    --app-input-bg: #091827;
                    --app-input-border: rgba(102, 159, 184, 0.26);
                    --app-warning: #ff8f70;
                    --readonly-track: rgba(152, 191, 210, 0.18);
                    --readonly-fill: #2fd5d0;
                    --app-panel-shadow: 0 24px 60px rgba(2, 9, 18, 0.42);
                    --modebar-bg: rgba(8, 20, 34, 0.72);
                    --modebar-icon: #9ec1d3;
                }}
                .theme-light,
                [data-system-theme="light"] .app-root:not(.theme-light):not(.theme-dark) {{
                    color-scheme: light;
                    --app-bg-gradient: linear-gradient(135deg, rgba(239, 245, 250, 0.98) 0%, rgba(231, 239, 246, 0.98) 52%, rgba(223, 233, 241, 0.98) 100%);
                    --app-header-bg: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 246, 250, 0.98) 100%);
                    --app-panel-bg: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(245, 249, 252, 0.98) 100%);
                    --app-panel-alt-bg: rgba(240, 246, 250, 0.98);
                    --app-graph-panel-bg: linear-gradient(145deg, rgba(240, 246, 250, 0.98) 0%, rgba(231, 239, 246, 0.98) 100%);
                    --app-assumptions-bg: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(242, 247, 251, 0.98) 100%);
                    --app-border: rgba(92, 126, 145, 0.22);
                    --app-text: #102638;
                    --app-text-secondary: #4d6577;
                    --app-text-muted: #6f8492;
                    --app-accent: #14b8c6;
                    --app-accent-soft: rgba(20, 184, 198, 0.12);
                    --app-accent-contrast: #082131;
                    --app-input-bg: #ffffff;
                    --app-input-border: rgba(92, 126, 145, 0.28);
                    --app-warning: #c2410c;
                    --readonly-track: rgba(111, 142, 161, 0.18);
                    --readonly-fill: #14b8c6;
                    --app-panel-shadow: 0 18px 42px rgba(118, 136, 150, 0.18);
                    --modebar-bg: rgba(255, 255, 255, 0.84);
                    --modebar-icon: #51606f;
                }}
                .control-number-input::-webkit-outer-spin-button,
                .control-number-input::-webkit-inner-spin-button {{
                    -webkit-appearance: none;
                    margin: 0;
                }}
                .control-number-input {{
                    -moz-appearance: textfield;
                    appearance: textfield;
                }}
                .control-number-input:focus {{
                    border-color: var(--app-accent) !important;
                    box-shadow: 0 0 0 1px rgba(47, 213, 208, 0.18) !important;
                    outline: none !important;
                }}
                .control-number-input:invalid,
                .control-number-input:out-of-range,
                .control-number-input:focus:invalid,
                .control-number-input:focus:out-of-range {{
                    border-color: var(--app-input-border) !important;
                    box-shadow: none !important;
                    outline: none !important;
                }}
                .js-plotly-plot .plotly .modebar {{
                    background: var(--modebar-bg) !important;
                    border-radius: 999px;
                    padding: 4px;
                }}
                .js-plotly-plot .plotly .modebar-btn svg path {{
                    fill: var(--modebar-icon) !important;
                }}
                .js-plotly-plot .plotly .modebar-btn:hover svg path,
                .js-plotly-plot .plotly .modebar-btn.active svg path {{
                    fill: var(--app-accent) !important;
                }}
                .theme-toggle-button {{
                    width: 100%;
                    padding: 10px 14px;
                    border: 1px solid var(--app-border);
                    border-radius: 999px;
                    background: var(--app-panel-alt-bg);
                    color: var(--app-text);
                    font-weight: 700;
                    cursor: pointer;
                }}
                .theme-toggle-button:hover {{
                    border-color: var(--app-accent);
                }}
                button:disabled,
                .control-number-input:disabled {{
                    opacity: 0.55;
                    cursor: not-allowed !important;
                    box-shadow: none !important;
                }}
                .prediction-loading-overlay {{
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 100%;
                    height: 100%;
                    min-height: 360px;
                    background: rgba(8, 20, 34, 0.34);
                    backdrop-filter: blur(2px);
                }}
                .theme-light .prediction-loading-overlay {{
                    background: rgba(240, 246, 250, 0.55);
                }}
                .prediction-loading-card {{
                    padding: 18px 22px;
                    border: 1px solid var(--app-border);
                    border-radius: 14px;
                    background: var(--app-panel-alt-bg);
                    color: var(--app-text);
                    box-shadow: var(--app-panel-shadow);
                    text-align: center;
                }}
                .prediction-loading-title {{
                    font-size: 18px;
                    font-weight: 700;
                    letter-spacing: 0.02em;
                }}
                .prediction-loading-note {{
                    margin-top: 6px;
                    font-size: 12px;
                    color: var(--app-text-secondary);
                }}
                ::-webkit-scrollbar {{
                    width: 10px;
                    height: 10px;
                }}
                ::-webkit-scrollbar-track {{
                    background: rgba(8, 20, 32, 0.85);
                }}
                ::-webkit-scrollbar-thumb {{
                    background: rgba(111, 142, 161, 0.52);
                    border-radius: 999px;
                }}
                @media (max-width: 980px) {{
                    .app-header {{
                        padding: 14px 18px 12px 18px !important;
                        gap: 12px !important;
                        flex-wrap: wrap !important;
                    }}
                    .app-header img {{
                        height: 56px !important;
                    }}
                    .app-shell {{
                        display: block !important;
                        min-height: auto !important;
                    }}
                    .controls-panel {{
                        width: 100% !important;
                        min-width: 0 !important;
                        max-height: none !important;
                        overflow-y: visible !important;
                        border-right: none !important;
                        border-bottom: 1px solid var(--app-border) !important;
                        padding: 18px !important;
                    }}
                    .plot-panel {{
                        width: 100% !important;
                        min-width: 0 !important;
                        padding: 14px 12px 0 12px !important;
                        overflow-x: hidden !important;
                    }}
                    .prediction-graph {{
                        height: clamp(320px, 52vh, 520px) !important;
                        min-height: 320px !important;
                    }}
                    .prediction-loading-overlay {{
                        min-height: 300px !important;
                    }}
                    .assumptions-section {{
                        padding: 20px 18px 28px 18px !important;
                    }}
                }}
                @media (max-width: 640px) {{
                    .prediction-graph {{
                        height: clamp(300px, 48vh, 460px) !important;
                        min-height: 300px !important;
                    }}
                    .prediction-loading-overlay {{
                        min-height: 280px !important;
                    }}
                    .js-plotly-plot .plotly .modebar {{
                        transform: scale(0.88);
                        transform-origin: top right;
                    }}
                }}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    """
    app.layout = html.Div(
        [
            dcc.Store(id="applied-controls-store", data=default_applied_controls),
            dcc.Store(id="theme-store", data=None),
            dcc.Store(id="viewport-store", data=None),
            dcc.Store(id="prediction-export-figures", data=default_export_figure_payload),
            dcc.Interval(id="viewport-poll", interval=1200, n_intervals=0),
            dcc.Store(
                id="png-export-options",
                data={
                    "filename": "interactive_ensemble_prediction_v10_300dpi",
                    "width": PNG_EXPORT_WIDTH_PX,
                    "height": PNG_EXPORT_HEIGHT_PX,
                    "scale": PNG_EXPORT_SCALE,
                    "compact_suffix": "confidence_band_only",
                    "light_suffix": "white_bg",
                    "dark_suffix": "dark_bg",
                },
            ),
            html.Div(
                [
                    html.Img(
                        src=header_logo_src,
                        alt="Rosetta logo",
                        style={
                            "height": "72px",
                            "width": "auto",
                            "display": "block",
                            "flexShrink": 0,
                        },
                    ),
                    html.Div(
                        [
                            html.H1(
                                "Rosetta NeuralNetwork - Interactive Plot",
                                style={
                                    "margin": "0",
                                    "color": "var(--app-text)",
                                    "fontWeight": 500,
                                    "letterSpacing": "0.01em",
                                },
                            ),
                        ],
                        style={"minWidth": "0", "flex": "1 1 auto"},
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Toggle Theme",
                                id="theme-toggle-button",
                                n_clicks=0,
                                disabled=False,
                                className="theme-toggle-button",
                            ),
                        ],
                        style={"marginLeft": "auto", "width": "220px", "maxWidth": "100%"},
                    ),
                ],
                className="app-header",
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "18px",
                    "padding": "16px 28px 12px 28px",
                    "borderBottom": "1px solid var(--app-border)",
                    "background": "var(--app-header-bg)",
                    "boxShadow": "0 12px 40px rgba(1, 8, 14, 0.25)",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "Controls",
                                style={
                                    "fontSize": "20px",
                                    "fontWeight": 700,
                                    "marginBottom": "18px",
                                    "color": "var(--app-text)",
                                    "letterSpacing": "0.02em",
                                },
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Apply Changes",
                                        id="apply-controls-button",
                                        n_clicks=0,
                                        disabled=False,
                                        style={
                                            "padding": "10px 16px",
                                            "border": "1px solid var(--app-accent)",
                                            "background": "var(--app-accent)",
                                            "color": "var(--app-accent-contrast)",
                                            "borderRadius": "6px",
                                            "fontWeight": 700,
                                            "cursor": "pointer",
                                            "boxShadow": "0 10px 24px rgba(20, 184, 198, 0.25)",
                                        },
                                    ),
                                    html.Div(
                                        id="apply-warning-message",
                                        style={"display": "none"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "6px",
                                    "marginBottom": "12px",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Button(
                                                "Download Plot as PNG",
                                                id="download-prediction-pngs-button",
                                                n_clicks=0,
                                                disabled=False,
                                                style={
                                                    "width": "100%",
                                                    "padding": "10px 14px",
                                                    "border": "1px solid var(--app-border)",
                                                    "background": "var(--app-panel-alt-bg)",
                                                    "color": "var(--app-text)",
                                                    "borderRadius": "6px",
                                                    "fontWeight": 700,
                                                    "cursor": "pointer",
                                                },
                                            ),
                                            html.Div(
                                                "Downloads four PNGs: white and dark theme versions, each with full annotations and confidence-band-only variants.",
                                                style={"fontSize": "12px", "color": "var(--app-text-secondary)"},
                                            ),
                                        ],
                                        style={"display": "flex", "flexDirection": "column", "gap": "6px", "minWidth": "0"},
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Download Excel",
                                                id="download-predictions-excel-button",
                                                n_clicks=0,
                                                disabled=False,
                                                style={
                                                    "width": "100%",
                                                    "padding": "10px 14px",
                                                    "border": "1px solid var(--app-border)",
                                                    "background": "var(--app-panel-alt-bg)",
                                                    "color": "var(--app-text)",
                                                    "borderRadius": "6px",
                                                    "fontWeight": 700,
                                                    "cursor": "pointer",
                                                },
                                            ),
                                            html.Div(
                                                "Exports current user inputs, derived internal values, model inputs, and predictions every 7 days.",
                                                style={"fontSize": "12px", "color": "var(--app-text-secondary)"},
                                            ),
                                        ],
                                        style={"display": "flex", "flexDirection": "column", "gap": "6px", "minWidth": "0"},
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                                    "gap": "12px",
                                    "marginBottom": "14px",
                                },
                            ),
                            dcc.Download(id="download-predictions-excel"),
                            html.Div(id="png-export-trigger", style={"display": "none"}),
                            html.Div(id="plot-resize-trigger", style={"display": "none"}),
                            control_grid(top_controls),
                            html.Div(
                                id="geometry-guardrail-summary",
                                children=build_geometry_guardrail_summary(
                                    default_geometry_guardrail_state,
                                ),
                            ),
                            html.Hr(style={"border": "none", "borderTop": "1px solid var(--app-border)", "margin": "18px 0"}),
                            section_title("User-Facing Ore And Chemistry Inputs", margin_top="0"),
                            html.Div(
                                helper_inputs_note,
                                style={"fontSize": "12px", "color": "var(--app-text-secondary)", "marginBottom": "10px"},
                            ),
                            control_grid(main_controls),
                            section_title("Modals Predictors In Model"),
                            html.Div(
                                "Grouped mineralogy inputs below are active model inputs.",
                                style={"fontSize": "12px", "color": "var(--app-text-secondary)", "marginBottom": "10px"},
                            ),
                            control_grid(grouped_model_controls) if grouped_model_controls else html.Div(
                                "No grouped predictors are active in the current model.",
                                style={"fontSize": "12px", "color": "var(--app-text-secondary)"},
                            ),
                            html.Div(
                                default_grouped_balance_state["note"],
                                id="grouped-balance-note",
                                style={"fontSize": "12px", "color": "var(--app-text-secondary)", "marginTop": "10px"},
                            ),
                        ],
                        className="controls-panel",
                        style={
                            "width": "34%",
                            "minWidth": "360px",
                            "padding": "22px 24px",
                            "background": "var(--app-panel-bg)",
                            "borderRight": "1px solid var(--app-border)",
                            "maxHeight": f"calc(100vh - {APP_HEADER_HEIGHT_CSS})",
                            "overflowY": "auto",
                            "boxShadow": "var(--app-panel-shadow)",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Loading(
                                id="prediction-graph-loading",
                                type="default",
                                color="var(--app-accent)",
                                delay_show=150,
                                target_components={"prediction-loading-target": "children"},
                                overlay_style={
                                    "visibility": "visible",
                                    "backgroundColor": "transparent",
                                },
                                custom_spinner=html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div("Loading...", className="prediction-loading-title"),
                                                html.Div(
                                                    "Please wait while the prediction plot is updating.",
                                                    className="prediction-loading-note",
                                                ),
                                            ],
                                            className="prediction-loading-card",
                                        ),
                                    ],
                                    className="prediction-loading-overlay",
                                ),
                                children=[
                                    dcc.Graph(
                                        id="prediction-graph",
                                        figure=default_display_figure,
                                        config=graph_export_config(),
                                        className="prediction-graph",
                                        style={"height": "min(78vh, 820px)", "width": "100%"},
                                    ),
                                    html.Div(
                                        id="prediction-loading-target",
                                        children="ready",
                                        style={"display": "none"},
                                    ),
                                ],
                            ),
                        ],
                        className="plot-panel",
                        style={
                            "width": "66%",
                            "minWidth": "0",
                            "padding": "18px 20px 0 20px",
                            "background": "var(--app-graph-panel-bg)",
                        },
                    ),
                ],
                className="app-shell",
                style={
                    "display": "flex",
                    "minHeight": f"calc(100vh - {APP_HEADER_HEIGHT_CSS})",
                    "background": "var(--app-bg-gradient)",
                    "color": "var(--app-text)",
                },
            ),
            html.Div(
                id="prediction-summary",
                children=default_prediction_summary,
            ),
        ],
        id="app-root",
        className="app-root",
        style={
            "fontFamily": APP_FONT_FAMILY,
            "minHeight": "100vh",
        },
    )
    app.clientside_callback(
        """
        function(nIntervals, currentViewport) {
            const width = window.innerWidth || 1280;
            const bucket = width <= 640 ? "narrow" : (width <= 980 ? "mobile" : "desktop");
            if (currentViewport && currentViewport.bucket === bucket) {
                return window.dash_clientside.no_update;
            }
            return {width: width, bucket: bucket};
        }
        """,
        Output("viewport-store", "data"),
        Input("viewport-poll", "n_intervals"),
        State("viewport-store", "data"),
    )
    app.clientside_callback(
        """
        function(appRootId, nClicks, currentTheme) {
            const getSystemTheme = () => {
                try {
                    if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
                        return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
                    }
                } catch (error) {}
                return "light";
            };

            const triggered = (dash_clientside.callback_context && dash_clientside.callback_context.triggered) || [];
            const triggeredProp = triggered.length ? triggered[0].prop_id : "";

            if (triggeredProp === "theme-toggle-button.n_clicks") {
                const baseTheme = (currentTheme === "light" || currentTheme === "dark") ? currentTheme : getSystemTheme();
                return baseTheme === "dark" ? "light" : "dark";
            }

            if (currentTheme === "light" || currentTheme === "dark") {
                return window.dash_clientside.no_update;
            }

            return getSystemTheme();
        }
        """,
        Output("theme-store", "data"),
        Input("app-root", "id"),
        Input("theme-toggle-button", "n_clicks"),
        State("theme-store", "data"),
    )
    app.clientside_callback(
        """
        function(nClicks, exportFigures, exportOptions) {
            if (!nClicks || !exportFigures || !window.Plotly) {
                return window.dash_clientside.no_update;
            }

            const cloneObj = (obj) => {
                if (typeof structuredClone === "function") {
                    return structuredClone(obj);
                }
                return JSON.parse(JSON.stringify(obj));
            };

            const renderToDataUrl = async (fig, options) => {
                const tempDiv = document.createElement("div");
                tempDiv.style.position = "fixed";
                tempDiv.style.left = "-10000px";
                tempDiv.style.top = "0";
                tempDiv.style.width = `${options.width}px`;
                tempDiv.style.height = `${options.height}px`;
                tempDiv.style.pointerEvents = "none";
                document.body.appendChild(tempDiv);
                try {
                    await window.Plotly.newPlot(
                        tempDiv,
                        fig.data || [],
                        fig.layout || {},
                        {displayModeBar: false, responsive: false, staticPlot: true}
                    );
                    return await window.Plotly.toImage(tempDiv, {
                        format: "png",
                        width: options.width,
                        height: options.height,
                        scale: options.scale
                    });
                } finally {
                    try {
                        window.Plotly.purge(tempDiv);
                    } catch (e) {}
                    tempDiv.remove();
                }
            };

            const downloadBlob = (blob, filename) => {
                const url = URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = url;
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                link.remove();
                setTimeout(() => URL.revokeObjectURL(url), 1000);
            };

            const downloadDataUrl = (dataUrl, filename) => {
                const link = document.createElement("a");
                link.href = dataUrl;
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                link.remove();
            };

            const buildFigureVariant = (figure, annotationText) => {
                const nextFigure = cloneObj(figure);
                if (
                    annotationText &&
                    nextFigure.layout &&
                    Array.isArray(nextFigure.layout.annotations) &&
                    nextFigure.layout.annotations.length > 0
                ) {
                    nextFigure.layout.annotations[0].text = annotationText;
                }
                return nextFigure;
            };

            const lightFigure = exportFigures.light || null;
            const darkFigure = exportFigures.dark || null;
            if (!lightFigure || !darkFigure) {
                return window.dash_clientside.no_update;
            }

            const lightMeta = (lightFigure.layout && lightFigure.layout.meta) || {};
            const darkMeta = (darkFigure.layout && darkFigure.layout.meta) || {};
            const baseFilename = (exportOptions && exportOptions.filename) || lightMeta.png_export_filename || darkMeta.png_export_filename || "interactive_ensemble_prediction_v10_300dpi";
            const compactSuffix = (exportOptions && exportOptions.compact_suffix) || "confidence_band_only";
            const lightSuffix = (exportOptions && exportOptions.light_suffix) || "white_bg";
            const darkSuffix = (exportOptions && exportOptions.dark_suffix) || "dark_bg";
            const options = exportOptions || {width: 1200, height: 800, scale: 1};
            (async () => {
                const downloads = [
                    {
                        figure: buildFigureVariant(lightFigure, lightMeta.png_export_annotation_full || null),
                        filename: `${baseFilename}_${lightSuffix}`,
                    },
                    {
                        figure: buildFigureVariant(lightFigure, lightMeta.png_export_annotation_compact || null),
                        filename: `${baseFilename}_${lightSuffix}_${compactSuffix}`,
                    },
                    {
                        figure: buildFigureVariant(darkFigure, darkMeta.png_export_annotation_full || null),
                        filename: `${baseFilename}_${darkSuffix}`,
                    },
                    {
                        figure: buildFigureVariant(darkFigure, darkMeta.png_export_annotation_compact || null),
                        filename: `${baseFilename}_${darkSuffix}_${compactSuffix}`,
                    },
                ];

                const rendered = [];
                for (const item of downloads) {
                    rendered.push({
                        filename: `${item.filename}.png`,
                        dataUrl: await renderToDataUrl(item.figure, options),
                    });
                }

                if (window.JSZip) {
                    const zip = new window.JSZip();
                    for (const item of rendered) {
                        const base64Data = String(item.dataUrl).split(",", 2)[1] || "";
                        zip.file(item.filename, base64Data, {base64: true});
                    }
                    const zipBlob = await zip.generateAsync({type: "blob", compression: "DEFLATE"});
                    downloadBlob(zipBlob, `${baseFilename}.zip`);
                } else {
                    for (const item of rendered) {
                        downloadDataUrl(item.dataUrl, item.filename);
                        await new Promise((resolve) => setTimeout(resolve, 150));
                    }
                }
            })();

            return window.dash_clientside.no_update;
        }
        """,
        Output("png-export-trigger", "children"),
        Input("download-prediction-pngs-button", "n_clicks"),
        State("prediction-export-figures", "data"),
        State("png-export-options", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        """
        function(loadingToken, themeName, viewportState) {
            if (typeof window === "undefined") {
                return window.dash_clientside.no_update;
            }
            const triggerResize = () => {
                window.dispatchEvent(new Event("resize"));
            };
            if (typeof window.requestAnimationFrame === "function") {
                window.requestAnimationFrame(triggerResize);
            } else {
                setTimeout(triggerResize, 0);
            }
            setTimeout(triggerResize, 140);
            return window.dash_clientside.no_update;
        }
        """,
        Output("plot-resize-trigger", "children"),
        Input("prediction-loading-target", "children"),
        Input("theme-store", "data"),
        Input("viewport-store", "data"),
    )

    control_component_ids = [
        "max-day",
        "weekly-catalyst-gt-week",
        "confidence-interval",
        "irrigation-rate-l-m2-h",
        "catalyst-addition-start-day",
        *[predictor_component_id(column) for column in EDITABLE_INPUT_COLUMNS],
    ]
    control_input_ids = [slider_input_component_id(component_id) for component_id in control_component_ids]
    control_increment_ids = [increment_button_component_id(component_id) for component_id in control_component_ids]
    control_decrement_ids = [decrement_button_component_id(component_id) for component_id in control_component_ids]
    increment_button_to_component = dict(zip(control_increment_ids, control_component_ids))
    decrement_button_to_component = dict(zip(control_decrement_ids, control_component_ids))

    @app.callback(
        *[Output(input_id, "value") for input_id in control_input_ids],
        Output("chemistry-cyanide-control", "children"),
        Output("grouped-balance-note", "children"),
        Output("geometry-guardrail-summary", "children"),
        *[Input(input_id, "value") for input_id in control_input_ids],
        *[Input(button_id, "n_clicks") for button_id in control_increment_ids],
        *[Input(button_id, "n_clicks") for button_id in control_decrement_ids],
        prevent_initial_call=True,
    )
    def sync_input_values(*callback_values: float) -> Tuple[Any, ...]:
        raw_values = callback_values[: len(control_component_ids)]
        resolved_control_values: Dict[str, float] = {}
        for component_id, raw_value in zip(control_component_ids, raw_values):
            resolved_control_values[component_id] = normalize_control_value(
                raw_value,
                control_specs_by_id[component_id],
            )

        triggered_component_id = ""
        for triggered_prop_id in [item["prop_id"].split(".")[0] for item in callback_context.triggered]:
            if triggered_prop_id in increment_button_to_component:
                triggered_component_id = increment_button_to_component[triggered_prop_id]
                min_value, max_value = current_control_bounds(
                    triggered_component_id,
                    resolved_control_values,
                    control_specs_by_id,
                    input_specs,
                )
                step_value = float(control_specs_by_id[triggered_component_id].get("step", 0.0))
                resolved_control_values[triggered_component_id] = round_control_value(
                    float(np.clip(resolved_control_values[triggered_component_id] + step_value, min_value, max_value)),
                    MAX_CONTROL_DECIMALS,
                )
                break
            if triggered_prop_id in decrement_button_to_component:
                triggered_component_id = decrement_button_to_component[triggered_prop_id]
                min_value, max_value = current_control_bounds(
                    triggered_component_id,
                    resolved_control_values,
                    control_specs_by_id,
                    input_specs,
                )
                step_value = float(control_specs_by_id[triggered_component_id].get("step", 0.0))
                resolved_control_values[triggered_component_id] = round_control_value(
                    float(np.clip(resolved_control_values[triggered_component_id] - step_value, min_value, max_value)),
                    MAX_CONTROL_DECIMALS,
                )
                break

        normalized_top_values = {
            "max-day": resolved_control_values["max-day"],
            "weekly-catalyst-gt-week": resolved_control_values["weekly-catalyst-gt-week"],
            "confidence-interval": resolved_control_values["confidence-interval"],
            "irrigation-rate-l-m2-h": resolved_control_values["irrigation-rate-l-m2-h"],
            "catalyst-addition-start-day": resolved_control_values["catalyst-addition-start-day"],
        }
        predictor_values = tuple(float(resolved_control_values[predictor_component_id(column)]) for column in EDITABLE_INPUT_COLUMNS)
        ui_values = resolve_ui_state(predictor_values, input_specs)
        ui_values, geometry_guardrail_state = apply_geometry_input_limits(ui_values, input_specs)
        ui_values, _feed_mass_state = resolve_internal_feed_mass(ui_values, input_specs)
        _static_values, _derived_values, _inactive_grouped_values, grouped_balance_state, resolved_chemistry = resolve_model_static_values(ui_values, input_specs)
        ui_values.update(_inactive_grouped_values)

        final_component_values: Dict[str, float] = dict(normalized_top_values)
        for column in EDITABLE_INPUT_COLUMNS:
            final_component_values[predictor_component_id(column)] = float(ui_values[column])

        outputs: List[Any] = []
        outputs.extend(final_component_values[component_id] for component_id in control_component_ids)
        outputs.extend(
            [
                build_cyanide_control(resolved_chemistry),
                grouped_balance_state["note"],
                build_geometry_guardrail_summary(
                    geometry_guardrail_state,
                ),
            ]
        )
        return tuple(outputs)

    callback_states = [
        State(slider_input_component_id("max-day"), "value"),
        State(slider_input_component_id("weekly-catalyst-gt-week"), "value"),
        State(slider_input_component_id("confidence-interval"), "value"),
        State(slider_input_component_id("irrigation-rate-l-m2-h"), "value"),
        State(slider_input_component_id("catalyst-addition-start-day"), "value"),
        *[State(slider_input_component_id(predictor_component_id(column)), "value") for column in EDITABLE_INPUT_COLUMNS],
    ]

    @app.callback(
        Output("applied-controls-store", "data"),
        Output("apply-warning-message", "children"),
        Output("apply-warning-message", "style"),
        Input("apply-controls-button", "n_clicks"),
        *callback_states,
        prevent_initial_call=True,
    )
    def apply_current_controls(
        _n_clicks: int,
        max_day: float,
        weekly_catalyst_gt_week: float,
        confidence_interval: float,
        irrigation_rate_l_m2_h: float,
        catalyst_addition_start_day: float,
        *predictor_values: float,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        ui_values = resolve_ui_state(tuple(predictor_values), input_specs)
        ui_values, geometry_guardrail_state = apply_geometry_input_limits(ui_values, input_specs)
        ui_values, _feed_mass_state = resolve_internal_feed_mass(ui_values, input_specs)
        _static_values, _derived_values, _inactive_grouped_values, grouped_balance_state, resolved_chemistry = resolve_model_static_values(ui_values, input_specs)
        blocking_message = blocking_constraint_message(geometry_guardrail_state, grouped_balance_state)
        if blocking_message:
            return (
                no_update,
                f"Apply blocked: {blocking_message}",
                {
                    "display": "block",
                    "fontSize": "12px",
                    "color": "var(--app-warning)",
                    "lineHeight": "1.5",
                    "padding": "10px 12px",
                    "border": "1px solid color-mix(in srgb, var(--app-warning) 34%, transparent)",
                    "borderRadius": "10px",
                    "background": "color-mix(in srgb, var(--app-warning) 8%, transparent)",
                },
            )
        ui_values[ACID_SOLUBLE_COL] = float(resolved_chemistry["acid"])
        ui_values[RESIDUAL_CPY_COL] = float(resolved_chemistry["residual"])
        ui_values[CYANIDE_SOLUBLE_COL] = float(resolved_chemistry["cyanide"])
        return (
            {
                "max_day": normalize_control_value(max_day, max_day_spec),
                "weekly_catalyst_gt_week": normalize_control_value(weekly_catalyst_gt_week, weekly_spec),
                "confidence_interval": normalize_control_value(confidence_interval, confidence_interval_spec),
                "irrigation_rate_l_m2_h": normalize_control_value(irrigation_rate_l_m2_h, irrigation_spec),
                "catalyst_addition_start_day": normalize_control_value(catalyst_addition_start_day, catalyst_start_day_spec),
                "predictor_values": {column: float(ui_values[column]) for column in EDITABLE_INPUT_COLUMNS},
            },
            "",
            {"display": "none"},
        )

    @app.callback(
        Output("app-root", "className"),
        Output("theme-toggle-button", "children"),
        Input("theme-store", "data"),
    )
    def sync_theme_shell(theme_name: str | None) -> Tuple[str, str]:
        if str(theme_name or "").lower() not in {"light", "dark"}:
            return "app-root theme-light", "Switch to Dark Theme"
        theme_key = "light" if str(theme_name or "dark").lower() == "light" else "dark"
        button_label = "Switch to Dark Theme" if theme_key == "light" else "Switch to Light Theme"
        return f"app-root theme-{theme_key}", button_label

    @app.callback(
        Output("prediction-graph", "figure"),
        Input("prediction-export-figures", "data"),
        Input("theme-store", "data"),
        Input("viewport-store", "data"),
    )
    def update_display_theme(
        export_figures: Dict[str, Any] | None,
        theme_name: str | None,
        viewport_state: Dict[str, Any] | None,
    ) -> go.Figure:
        theme_key = "light" if str(theme_name or "light").lower() != "dark" else "dark"
        payload = export_figures or default_export_figure_payload
        figure = payload.get(theme_key) or payload.get("light") or default_export_figure_payload["light"]
        return make_display_figure(figure, viewport_state)

    update_running_outputs = [
        (Output("apply-controls-button", "disabled"), True, False),
        (Output("apply-controls-button", "children"), "Updating...", "Apply Changes"),
        (Output("theme-toggle-button", "disabled"), True, False),
        (Output("download-prediction-pngs-button", "disabled"), True, False),
        (Output("download-predictions-excel-button", "disabled"), True, False),
        *[(Output(input_id, "disabled"), True, False) for input_id in control_input_ids],
        *[(Output(button_id, "disabled"), True, False) for button_id in control_increment_ids],
        *[(Output(button_id, "disabled"), True, False) for button_id in control_decrement_ids],
    ]

    @app.callback(
        Output("prediction-summary", "children"),
        Output("prediction-export-figures", "data"),
        Output("prediction-loading-target", "children"),
        Input("applied-controls-store", "data"),
        running=update_running_outputs,
    )
    def update_plot(
        applied_controls: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        applied_controls = applied_controls or default_applied_controls
        stored_predictor_values = applied_controls.get("predictor_values", {})
        predictor_values = tuple(
            normalize_cache_float(stored_predictor_values.get(column, default_ui_values[column]))
            for column in EDITABLE_INPUT_COLUMNS
        )
        bundle = get_cached_prediction_bundle(
            normalize_cache_float(applied_controls.get("max_day", max_day_spec["default"])),
            normalize_cache_float(applied_controls.get("weekly_catalyst_gt_week", weekly_spec["default"])),
            normalize_cache_float(applied_controls.get("confidence_interval", confidence_interval_spec["default"])),
            normalize_cache_float(applied_controls.get("irrigation_rate_l_m2_h", irrigation_spec["default"])),
            normalize_cache_float(applied_controls.get("catalyst_addition_start_day", catalyst_start_day_spec["default"])),
            predictor_values,
        )
        return (
            bundle["summary"],
            bundle["export_figure_payload"],
            bundle["loading_token"],
        )

    @app.callback(
        Output("download-predictions-excel", "data"),
        Input("download-predictions-excel-button", "n_clicks"),
        State("applied-controls-store", "data"),
        prevent_initial_call=True,
    )
    def download_predictions_excel(
        _n_clicks: int,
        applied_controls: Dict[str, Any],
    ) -> Dict[str, Any]:
        applied_controls = applied_controls or default_applied_controls
        stored_predictor_values = applied_controls.get("predictor_values", {})
        predictor_values = tuple(
            normalize_cache_float(stored_predictor_values.get(column, default_ui_values[column]))
            for column in EDITABLE_INPUT_COLUMNS
        )
        bundle = get_cached_prediction_bundle(
            normalize_cache_float(applied_controls.get("max_day", max_day_spec["default"])),
            normalize_cache_float(applied_controls.get("weekly_catalyst_gt_week", weekly_spec["default"])),
            normalize_cache_float(applied_controls.get("confidence_interval", confidence_interval_spec["default"])),
            normalize_cache_float(applied_controls.get("irrigation_rate_l_m2_h", irrigation_spec["default"])),
            normalize_cache_float(applied_controls.get("catalyst_addition_start_day", catalyst_start_day_spec["default"])),
            predictor_values,
        )
        excel_bytes = build_prediction_export_bytes(
            pred=bundle["pred"],
            ui_values=bundle["ui_values"],
            static_values=bundle["static_values"],
            derived_values=bundle["derived_values"],
            geometry_values=bundle["geometry_values"],
            inactive_grouped_values=bundle["inactive_grouped_values"],
            grouped_balance_state=bundle["grouped_balance_state"],
            resolved_chemistry=bundle["resolved_chemistry"],
            max_day=float(bundle["resolved_max_day"]),
        )
        return dcc.send_bytes(excel_bytes, "interactive_ensemble_prediction_v10_7day.xlsx")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive Plotly/Dash app for NN_ExpEq_columns_only_v10 using the saved "
            "member checkpoints created after running NN_ExpEq_columns_only_v10.py."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8057)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional override for the NN_Pytorch_ExpEq_columns_only_v10 root.",
    )
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    app = create_app(project_root)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
