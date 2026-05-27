"""CTM 特征温度法核心计算模块。"""
import numpy as np
import pandas as pd
from typing import Dict

from .models import BuildingParams


def calc_h_values(p: BuildingParams) -> Dict[str, float]:
    """计算各分项传热反应系数 H，单位 W/K。"""
    h_wall = p.u_wall * (p.wall_area_east + p.wall_area_west + p.wall_area_south + p.wall_area_north)
    h_roof = p.u_roof * p.roof_area
    h_window = p.u_window * (p.window_area_east + p.window_area_west + p.window_area_south + p.window_area_north)
    h_door = p.u_door * p.door_area
    h_air = p.air_density * p.air_specific_heat * p.air_change_rate * p.volume / 3600.0
    h_total = h_wall + h_roof + h_window + h_door + h_air
    return {
        "h_wall_w_per_k": h_wall,
        "h_roof_w_per_k": h_roof,
        "h_window_w_per_k": h_window,
        "h_door_w_per_k": h_door,
        "h_air_w_per_k": h_air,
        "h_total_w_per_k": h_total,
    }


def calc_satellite_air_temp(outdoor_temp: float, solar_irr: float, absorptance: float, h_out: float) -> float:
    """计算太阳空气温度。"""
    return outdoor_temp + absorptance * solar_irr / h_out


def calc_humidity_ratio(temp_c: float, rh_pct: float, pressure_kpa: float) -> float:
    """计算含湿量 (kg/kg干空气)。

    使用 Antoine 型公式计算饱和水汽压。
    """
    # 饱和水汽压 kPa
    p_ws = 0.61078 * np.exp(17.2694 * temp_c / (temp_c + 237.3))
    # 水蒸气分压力 kPa
    p_w = rh_pct / 100.0 * p_ws
    # 含湿量
    w = 0.62198 * p_w / (pressure_kpa - p_w)
    return w


def calculate_hourly(p: BuildingParams, weather_df: pd.DataFrame) -> pd.DataFrame:
    """对逐时气象数据计算 CTM 负荷。

    参数:
        p: 建筑参数
        weather_df: 包含 date, time, hour, month, outdoor_temp, relative_humidity,
                     ghi, solar_north, solar_east, solar_south, solar_west 的 DataFrame

    返回:
        包含所有逐时计算结果的 DataFrame
    """
    df = weather_df.copy()
    n = len(df)

    # 传热反应系数（常数）
    h_vals = calc_h_values(p)
    for k, v in h_vals.items():
        df[k] = v

    h_total = h_vals["h_total_w_per_k"]

    # 太阳空气温度
    df["t_sa_east_deg_c"] = df["outdoor_temp"] + p.solar_absorptance * df["solar_east"] / p.outdoor_heat_transfer_coeff
    df["t_sa_west_deg_c"] = df["outdoor_temp"] + p.solar_absorptance * df["solar_west"] / p.outdoor_heat_transfer_coeff
    df["t_sa_south_deg_c"] = df["outdoor_temp"] + p.solar_absorptance * df["solar_south"] / p.outdoor_heat_transfer_coeff
    df["t_sa_north_deg_c"] = df["outdoor_temp"] + p.solar_absorptance * df["solar_north"] / p.outdoor_heat_transfer_coeff
    df["t_sa_roof_deg_c"] = df["outdoor_temp"] + p.solar_absorptance * df["ghi"] / p.outdoor_heat_transfer_coeff

    # 玻璃太阳得热 Q_glass (W)
    df["q_glass_w"] = (
        p.shgc * p.shade_factor * (
            p.window_area_east * df["solar_east"]
            + p.window_area_west * df["solar_west"]
            + p.window_area_south * df["solar_south"]
            + p.window_area_north * df["solar_north"]
        )
    )

    # 占用系数
    if "hour" not in df.columns:
        df["hour"] = 12  # fallback
    df["occupancy_factor"] = np.where(
        (df["hour"] >= p.occupancy_start_hour) & (df["hour"] < p.occupancy_end_hour), 1.0, 0.0
    )

    # 人员、照明、设备显热 (W)
    floor_area = p.floor_area
    df["q_internal_sensible_w"] = (
        df["occupancy_factor"] * floor_area * (
            p.lighting_power_density
            + p.equipment_power_density
            + p.occupant_density * p.person_sensible_heat
        )
    )

    # ---- 分项特征温度 ----
    df["t_inf_wall_deg_c"] = p.u_wall * (
        p.wall_area_east * df["t_sa_east_deg_c"]
        + p.wall_area_west * df["t_sa_west_deg_c"]
        + p.wall_area_south * df["t_sa_south_deg_c"]
        + p.wall_area_north * df["t_sa_north_deg_c"]
    ) / h_total

    df["t_inf_roof_deg_c"] = (
        p.u_roof * p.roof_area * df["t_sa_roof_deg_c"] / h_total
    )

    window_area_total = p.window_area_total
    df["t_inf_window_deg_c"] = (
        p.u_window * window_area_total * df["outdoor_temp"] / h_total
    )

    df["t_inf_door_deg_c"] = (
        p.u_door * p.door_area * df["outdoor_temp"] / h_total
    )

    h_air = h_vals["h_air_w_per_k"]
    df["t_inf_air_deg_c"] = h_air * df["outdoor_temp"] / h_total

    df["t_inf_glass_deg_c"] = df["q_glass_w"] / h_total

    df["t_inf_internal_deg_c"] = df["q_internal_sensible_w"] / h_total

    # ---- 关机总特征温度 ----
    df["t_inf_shut_deg_c"] = (
        df["t_inf_wall_deg_c"]
        + df["t_inf_roof_deg_c"]
        + df["t_inf_window_deg_c"]
        + df["t_inf_door_deg_c"]
        + df["t_inf_air_deg_c"]
        + df["t_inf_glass_deg_c"]
        + df["t_inf_internal_deg_c"]
    )

    # 校验项：直接从原始量计算
    df["t_inf_shut_check_deg_c"] = (
        p.u_wall * (
            p.wall_area_east * df["t_sa_east_deg_c"]
            + p.wall_area_west * df["t_sa_west_deg_c"]
            + p.wall_area_south * df["t_sa_south_deg_c"]
            + p.wall_area_north * df["t_sa_north_deg_c"]
        )
        + p.u_roof * p.roof_area * df["t_sa_roof_deg_c"]
        + p.u_window * window_area_total * df["outdoor_temp"]
        + p.u_door * p.door_area * df["outdoor_temp"]
        + h_air * df["outdoor_temp"]
        + df["q_glass_w"]
        + df["q_internal_sensible_w"]
    ) / h_total

    # ---- 式 6-26 冷负荷（显热）----
    df["delta_cooling_deg_c"] = np.maximum(0, df["t_inf_shut_deg_c"] - p.cooling_setpoint)
    df["cooling_load_sensible_kw"] = np.maximum(0, (df["t_inf_shut_deg_c"] - p.cooling_setpoint) * h_total / 1000.0)

    # ---- 式 6-26 供暖负荷 ----
    df["delta_heating_deg_c"] = np.maximum(0, p.heating_setpoint - df["t_inf_shut_deg_c"])
    df["heating_load_kw"] = np.maximum(0, (p.heating_setpoint - df["t_inf_shut_deg_c"]) * h_total / 1000.0)

    # ---- 潜热附加项 ----
    # 室外含湿量
    df["outdoor_humidity_ratio"] = df.apply(
        lambda row: calc_humidity_ratio(row["outdoor_temp"], row["relative_humidity"], p.atmospheric_pressure),
        axis=1
    )
    # 室内含湿量
    df["indoor_humidity_ratio"] = calc_humidity_ratio(p.cooling_setpoint, p.indoor_cooling_rh, p.atmospheric_pressure)

    # 通风潜热冷负荷 (kW)
    q_lat_air = (
        p.air_density * p.air_change_rate * p.volume
        * p.latent_heat_vaporization
        * np.maximum(0, df["outdoor_humidity_ratio"] - df["indoor_humidity_ratio"])
        / (3600.0 * 1000.0)
    )
    df["ventilation_latent_load_kw"] = np.where(
        df["cooling_load_sensible_kw"] > 0, q_lat_air, 0.0
    )

    # 人员潜热冷负荷 (kW)
    q_lat_people = df["occupancy_factor"] * floor_area * p.occupant_density * p.person_latent_heat / 1000.0
    df["people_latent_load_kw"] = np.where(
        df["cooling_load_sensible_kw"] > 0, q_lat_people, 0.0
    )

    # 空调总冷负荷
    df["cooling_load_total_kw"] = (
        df["cooling_load_sensible_kw"]
        + df["ventilation_latent_load_kw"]
        + df["people_latent_load_kw"]
    )

    # 能耗（根据时间步长计算）
    time_step_h = p.time_step_minutes / 60.0
    df["cooling_energy_sensible_kwh"] = df["cooling_load_sensible_kw"] * time_step_h
    df["heating_energy_kwh"] = df["heating_load_kw"] * time_step_h
    df["cooling_energy_total_kwh"] = df["cooling_load_total_kw"] * time_step_h

    return df


def calc_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """计算日汇总。"""
    if "date" not in df.columns or len(df) == 0:
        return pd.DataFrame()

    daily = df.groupby("date").agg(
        _cool_sens=("cooling_energy_sensible_kwh", "sum"),
        _heat=("heating_energy_kwh", "sum"),
        _cool_total=("cooling_energy_total_kwh", "sum"),
        _cool_sens_peak=("cooling_load_sensible_kw", "max"),
        _heat_peak=("heating_load_kw", "max"),
        _cool_total_peak=("cooling_load_total_kw", "max"),
        _t_inf_mean=("t_inf_shut_deg_c", "mean"),
        _t_inf_max=("t_inf_shut_deg_c", "max"),
        _t_inf_min=("t_inf_shut_deg_c", "min"),
    ).reset_index()

    daily.columns = [
        "日期", "日冷负荷显热_式6-26_kWh", "日供暖负荷_式6-26_kWh",
        "日空调总冷负荷_含潜热_kWh", "日冷负荷显热峰值_kW", "日供暖负荷峰值_kW",
        "日空调总冷负荷峰值_kW", "日平均关机总特征温度_℃", "日最高关机总特征温度_℃",
        "日最低关机总特征温度_℃"
    ]
    return daily


def calc_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """计算月汇总。"""
    if "month" not in df.columns or len(df) == 0:
        return pd.DataFrame()

    monthly = df.groupby("month").agg(
        _cool_sens=("cooling_energy_sensible_kwh", "sum"),
        _heat=("heating_energy_kwh", "sum"),
        _cool_total=("cooling_energy_total_kwh", "sum"),
        _cool_sens_peak=("cooling_load_sensible_kw", "max"),
        _heat_peak=("heating_load_kw", "max"),
        _cool_total_peak=("cooling_load_total_kw", "max"),
        _t_inf_mean=("t_inf_shut_deg_c", "mean"),
        _t_inf_max=("t_inf_shut_deg_c", "max"),
        _t_inf_min=("t_inf_shut_deg_c", "min"),
    ).reset_index()

    monthly.columns = [
        "月份", "月冷负荷显热_式6-26_kWh", "月供暖负荷_式6-26_kWh",
        "月空调总冷负荷_含潜热_kWh", "月冷负荷显热峰值_kW", "月供暖负荷峰值_kW",
        "月空调总冷负荷峰值_kW", "月平均关机总特征温度_℃", "月最高关机总特征温度_℃",
        "月最低关机总特征温度_℃"
    ]
    return monthly


def get_peak_hours(df: pd.DataFrame, col: str, top_n: int = 20) -> pd.DataFrame:
    """获取指定负荷列的峰值前 N 小时。"""
    if col not in df.columns or len(df) == 0:
        return pd.DataFrame()

    display_cols = [
        "date", "time", "month", "hour", "outdoor_temp", "relative_humidity",
        "t_inf_shut_deg_c", "cooling_load_sensible_kw", "heating_load_kw",
        "cooling_load_total_kw"
    ]
    available = [c for c in display_cols if c in df.columns]
    sorted_df = df.nlargest(min(top_n, len(df)), col)[available].reset_index(drop=True)
    sorted_df.index = sorted_df.index + 1
    sorted_df.index.name = "排名"
    return sorted_df
