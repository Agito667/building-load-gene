"""CTM 特征温度法核心计算模块。"""
import numpy as np
import pandas as pd
from typing import Dict, List

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


def generate_formulas(p: BuildingParams) -> List[dict]:
    """生成计算公式说明列表，每项包含 title 和 lines，供界面展示。

    返回:
        列表，每项为 {"title": str, "lines": [str, ...]}
    """
    h = calc_h_values(p)
    floor_area = p.floor_area
    window_total = p.window_area_total
    wall_total = p.wall_area_total
    time_step_h = p.time_step_minutes / 60.0

    sections = []

    # 1. 建筑面积
    if p.floor_area_mode == "manual":
        sections.append({
            "title": "建筑面积",
            "lines": [
                f"建筑面积 A = 手动输入 = {floor_area:.1f} m²",
            ]
        })
    else:
        sections.append({
            "title": "建筑面积",
            "lines": [
                f"建筑面积 A = 体积 V / 层高 h = {p.volume:.1f} / {p.floor_height:.1f} = {floor_area:.1f} m²",
            ]
        })

    # 2. 传热反应系数 H
    sections.append({
        "title": "各分项传热反应系数 H（W/K）",
        "lines": [
            f"H_外墙 = U_外墙 × A_外墙总面积 = {p.u_wall:.4f} × ({p.wall_area_east:.1f}+{p.wall_area_west:.1f}+{p.wall_area_south:.1f}+{p.wall_area_north:.1f}) = {p.u_wall:.4f} × {wall_total:.1f} = {h['h_wall_w_per_k']:.2f} W/K",
            f"H_屋面 = U_屋面 × A_屋面 = {p.u_roof:.4f} × {p.roof_area:.1f} = {h['h_roof_w_per_k']:.2f} W/K",
            f"H_外窗 = U_外窗 × A_外窗总面积 = {p.u_window:.4f} × ({p.window_area_east:.1f}+{p.window_area_west:.1f}+{p.window_area_south:.1f}+{p.window_area_north:.1f}) = {p.u_window:.4f} × {window_total:.1f} = {h['h_window_w_per_k']:.2f} W/K",
            f"H_外门 = U_外门 × A_外门 = {p.u_door:.4f} × {p.door_area:.1f} = {h['h_door_w_per_k']:.2f} W/K",
            f"H_通风 = ρ × c × n × V / 3600 = {p.air_density:.2f} × {p.air_specific_heat:.0f} × {p.air_change_rate:.2f} × {p.volume:.1f} / 3600 = {h['h_air_w_per_k']:.2f} W/K",
            f"H_总 = H_外墙+H_屋面+H_外窗+H_外门+H_通风 = {h['h_wall_w_per_k']:.2f}+{h['h_roof_w_per_k']:.2f}+{h['h_window_w_per_k']:.2f}+{h['h_door_w_per_k']:.2f}+{h['h_air_w_per_k']:.2f} = {h['h_total_w_per_k']:.2f} W/K",
        ]
    })

    # 3. 太阳空气温度
    sections.append({
        "title": "太阳空气温度 T_sa（℃）",
        "lines": [
            f"T_sa = T_室外 + α × I / h_室外",
            f"其中：外表面太阳吸收率 α = {p.solar_absorptance:.2f}，室外综合换热系数 h_室外 = {p.outdoor_heat_transfer_coeff:.1f} W/(m²·K)",
            f"各朝向分别代入对应太阳辐射 I（东/西/南/北/屋面水平面）",
        ]
    })

    # 4. 玻璃太阳得热
    sections.append({
        "title": "玻璃太阳得热 Q_glass（W）",
        "lines": [
            f"Q_glass = SHGC × K_遮 × (A_东窗×I_东 + A_西窗×I_西 + A_南窗×I_南 + A_北窗×I_北)",
            f"其中：SHGC = {p.shgc:.4f}，K_遮 = {p.shade_factor:.4f}",
            f"A_东窗 = {p.window_area_east:.1f} m²，A_西窗 = {p.window_area_west:.1f} m²，A_南窗 = {p.window_area_south:.1f} m²，A_北窗 = {p.window_area_north:.1f} m²",
        ]
    })

    # 5. 内部负荷
    sections.append({
        "title": "内部显热负荷 Q_internal（W）",
        "lines": [
            f"Q_internal = f_占用 × A × (q_照明 + q_设备 + d_人员 × Q_人显)",
            f"其中：建筑面积 A = {floor_area:.1f} m²",
            f"q_照明 = {p.lighting_power_density:.1f} W/m²，q_设备 = {p.equipment_power_density:.1f} W/m²",
            f"d_人员 = {p.occupant_density:.4f} 人/m²，Q_人显 = {p.person_sensible_heat:.1f} W/人",
            f"f_占用：{p.occupancy_start_hour}:00~{p.occupancy_end_hour}:00 为 1，其余为 0",
        ]
    })

    # 6. 含湿量
    sections.append({
        "title": "含湿量计算（Antoine 型公式）",
        "lines": [
            f"饱和水汽压：P_ws = 0.61078 × exp(17.2694 × T / (T + 237.3))  （kPa）",
            f"水蒸气分压力：P_w = RH/100 × P_ws  （kPa）",
            f"含湿量：w = 0.62198 × P_w / (P_大气 - P_w)  （kg/kg干空气）",
            f"其中：大气压力 P_大气 = {p.atmospheric_pressure:.3f} kPa",
        ]
    })

    # 7. 分项特征温度
    sections.append({
        "title": "各分项特征温度 t_inf（℃）",
        "lines": [
            f"t_inf_外墙 = U_外墙 × Σ(A_i × T_sa_i) / H_总",
            f"t_inf_屋面 = U_屋面 × A_屋面 × T_sa_屋面 / H_总",
            f"t_inf_外窗 = U_外窗 × A_外窗总面积 × T_室外 / H_总",
            f"t_inf_外门 = U_外门 × A_外门 × T_室外 / H_总",
            f"t_inf_通风 = H_通风 × T_室外 / H_总",
            f"t_inf_玻璃得热 = Q_glass / H_总",
            f"t_inf_内部负荷 = Q_internal / H_总",
            f"其中 H_总 = {h['h_total_w_per_k']:.2f} W/K",
        ]
    })

    # 8. 关机总特征温度
    sections.append({
        "title": "关机总特征温度 t∞_shut（℃）",
        "lines": [
            f"t∞_shut = t_inf_外墙 + t_inf_屋面 + t_inf_外窗 + t_inf_外门 + t_inf_通风 + t_inf_玻璃得热 + t_inf_内部负荷",
        ]
    })

    # 9. 式 6-26 冷负荷与供暖负荷
    sections.append({
        "title": "式 6-26 冷负荷与供暖负荷",
        "lines": [
            f"冷负荷显热（kW）= max(0, t∞_shut - T_冷设定) × H_总 / 1000",
            f"其中：空调设定温度 T_冷设定 = {p.cooling_setpoint:.1f} ℃",
            f"",
            f"供暖负荷（kW）= max(0, T_暖设定 - t∞_shut) × H_总 / 1000",
            f"其中：供暖设定温度 T_暖设定 = {p.heating_setpoint:.1f} ℃",
        ]
    })

    # 10. 潜热负荷
    sections.append({
        "title": "潜热附加冷负荷",
        "lines": [
            f"通风潜热（kW）= ρ × n × V × r × max(0, w_室外 - w_室内) / (3600 × 1000)",
            f"其中：ρ = {p.air_density:.2f} kg/m³，n = {p.air_change_rate:.2f} h⁻¹，V = {p.volume:.1f} m³",
            f"r（汽化潜热）= {p.latent_heat_vaporization:.0f} J/kg",
            f"w_室内：按 T_冷设定={p.cooling_setpoint:.1f}℃、RH={p.indoor_cooling_rh:.0f}% 计算",
            f"",
            f"人员潜热（kW）= f_占用 × A × d_人员 × Q_人潜 / 1000",
            f"其中：Q_人潜 = {p.person_latent_heat:.1f} W/人",
            f"",
            f"以上潜热仅在冷负荷显热 > 0 时累加",
        ]
    })

    # 11. 能耗
    sections.append({
        "title": "能耗计算",
        "lines": [
            f"能耗（kWh）= 负荷（kW）× 时间步长（h）",
            f"时间步长 = {p.time_step_minutes} 分钟 = {time_step_h:.2f} h",
        ]
    })

    return sections
