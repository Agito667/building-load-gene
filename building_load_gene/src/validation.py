"""输入校验模块。"""
from typing import List, Tuple
import pandas as pd
import numpy as np

from .models import BuildingParams


def validate_params(params: BuildingParams) -> Tuple[List[str], List[str]]:
    """校验建筑参数，返回 (errors, warnings)。"""
    errors: List[str] = []
    warnings: List[str] = []

    # 体积
    if params.volume < 0:
        errors.append("建筑体积不能为负")
    if params.volume == 0:
        errors.append("建筑体积不能为 0")

    # 层高
    if params.floor_height <= 0:
        errors.append("层高不能为 0 或负数")

    # 面积
    area_fields = [
        ("roof_area", "屋面面积"),
        ("wall_area_east", "东向外墙面积"),
        ("wall_area_west", "西向外墙面积"),
        ("wall_area_south", "南向外墙面积"),
        ("wall_area_north", "北向外墙面积"),
        ("window_area_east", "东向外窗面积"),
        ("window_area_west", "西向外窗面积"),
        ("window_area_south", "南向外窗面积"),
        ("window_area_north", "北向外窗面积"),
        ("door_area", "外门面积"),
    ]
    for field_name, label in area_fields:
        val = getattr(params, field_name)
        if val < 0:
            errors.append(f"{label}不能为负")

    # 传热系数
    u_fields = [
        ("u_wall", "外墙传热系数"),
        ("u_roof", "屋面传热系数"),
        ("u_window", "外窗传热系数"),
        ("u_door", "外门传热系数"),
    ]
    for field_name, label in u_fields:
        val = getattr(params, field_name)
        if val < 0:
            errors.append(f"{label}不能为负")

    # SHGC
    if params.shgc < 0 or params.shgc > 1.2:
        warnings.append(f"外窗太阳得热系数 SHGC={params.shgc} 不在合理范围 [0, 1.2]")

    # 遮阳修正系数
    if params.shade_factor < 0:
        errors.append("遮阳修正系数不能为负")

    # 太阳吸收率
    if params.solar_absorptance < 0 or params.solar_absorptance > 1.0:
        warnings.append(f"外表面太阳吸收率={params.solar_absorptance} 不在合理范围 [0, 1]")

    # 换气次数
    if params.air_change_rate < 0:
        errors.append("换气次数不能为负")

    # 空气参数
    if params.air_density <= 0:
        errors.append("空气密度不能为负或零")
    if params.air_specific_heat <= 0:
        errors.append("空气定压比热不能为负或零")
    if params.atmospheric_pressure <= 0:
        errors.append("大气压力不能为负或零")

    # 设定温度
    if params.cooling_setpoint < 10 or params.cooling_setpoint > 40:
        warnings.append(f"空调设定温度={params.cooling_setpoint}℃ 不在常见范围 [10, 40]")
    if params.heating_setpoint < 5 or params.heating_setpoint > 30:
        warnings.append(f"供暖设定温度={params.heating_setpoint}℃ 不在常见范围 [5, 30]")

    # 相对湿度
    if params.indoor_cooling_rh < 0 or params.indoor_cooling_rh > 100:
        errors.append("室内冷房相对湿度必须在 0~100 之间")

    # 时间参数
    if params.occupancy_start_hour < 0 or params.occupancy_start_hour > 23:
        errors.append("占用开始时间必须在 0~23 之间")
    if params.occupancy_end_hour < 0 or params.occupancy_end_hour > 24:
        errors.append("占用结束时间必须在 0~24 之间")

    # 手动建筑面积
    if params.floor_area_mode == "manual":
        if params.floor_area_manual is None or params.floor_area_manual <= 0:
            errors.append("手动建筑面积必须为正数")

    # 时间步长
    if params.time_step_minutes < 10 or params.time_step_minutes > 120:
        errors.append("时间步长必须在 10~120 分钟之间")
    if params.time_step_minutes % 10 != 0:
        errors.append("时间步长必须为 10 的整数倍")

    return errors, warnings


REQUIRED_COLUMNS = [
    "date", "time", "outdoor_temp", "relative_humidity",
    "ghi", "solar_north", "solar_east", "solar_south", "solar_west",
]

# 中文列名到英文列名的映射
COLUMN_NAME_MAP = {
    "日期": "date",
    "时间": "time",
    "室外干球温度": "outdoor_temp",
    "室外温度": "outdoor_temp",
    "相对湿度": "relative_humidity",
    "总水平辐射": "ghi",
    "水平面总辐射": "ghi",
    "北向辐射": "solar_north",
    "东向辐射": "solar_east",
    "南向辐射": "solar_south",
    "西向辐射": "solar_west",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将中文列名映射为英文列名。"""
    rename_map = {}
    for col in df.columns:
        col_stripped = str(col).strip()
        if col_stripped in COLUMN_NAME_MAP:
            rename_map[col] = COLUMN_NAME_MAP[col_stripped]
        else:
            rename_map[col] = col_stripped.lower().strip()
    df = df.rename(columns=rename_map)
    return df


def validate_weather_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str], pd.DataFrame]:
    """校验逐时气象数据。

    返回: (是否有效, 错误列表, 规范化后的 DataFrame)
    """
    errors: List[str] = []

    if df is None or df.empty:
        return False, ["数据为空，请上传有效文件或输入数据"], df

    # 规范化列名
    df = normalize_columns(df.copy())

    # 检查必需列
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return False, [f"缺少必需列: {', '.join(missing_cols)}"], df

    # 检查行数
    if len(df) == 0:
        return False, ["数据文件中没有数据行"], df

    # 日期解析
    try:
        df["date"] = pd.to_datetime(df["date"].astype(str)).dt.strftime("%Y-%m-%d")
    except Exception:
        errors.append("日期字段无法解析，请检查格式（应为 yyyy-mm-dd）")

    # 时间解析
    try:
        df["time"] = df["time"].astype(str).str.strip()
        # 尝试解析小时
        time_parsed = pd.to_datetime(df["time"], format="%H:%M", errors="coerce")
        if time_parsed.isna().all():
            time_parsed = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")
        if time_parsed.isna().all():
            errors.append("时间字段无法解析，请检查格式（应为 HH:MM）")
        else:
            df["hour"] = time_parsed.dt.hour
    except Exception:
        errors.append("时间字段无法解析，请检查格式（应为 HH:MM）")

    # 解析月份
    if "date" in df.columns:
        try:
            df["month"] = pd.to_datetime(df["date"]).dt.month
        except Exception:
            pass

    # 数值列校验
    numeric_checks = {
        "outdoor_temp": "室外干球温度",
        "relative_humidity": "相对湿度",
        "ghi": "水平面总辐射",
        "solar_north": "北向辐射",
        "solar_east": "东向辐射",
        "solar_south": "南向辐射",
        "solar_west": "西向辐射",
    }

    for col, label in numeric_checks.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            nan_rows = df[df[col].isna()].index.tolist()
            if nan_rows and len(nan_rows) <= 5:
                errors.append(f"{label}列存在非数值值，行号: {[r+2 for r in nan_rows]}")
            elif nan_rows:
                errors.append(f"{label}列存在 {len(nan_rows)} 个非数值值")

    # 相对湿度范围
    if "relative_humidity" in df.columns:
        rh = df["relative_humidity"].dropna()
        if (rh < 0).any() or (rh > 100).any():
            errors.append("相对湿度必须在 0~100 之间")

    # 辐射非负
    for col in ["ghi", "solar_north", "solar_east", "solar_south", "solar_west"]:
        if col in df.columns:
            vals = df[col].dropna()
            if (vals < 0).any():
                errors.append(f"{col} 列存在负值，太阳辐射不能为负")

    if errors:
        return False, errors, df

    return True, [], df
