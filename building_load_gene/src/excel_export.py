"""Excel 导出模块。"""
import io
import pandas as pd
import numpy as np
from typing import Optional

from .models import BuildingParams, PARAM_GROUPS
from .calculator import calc_h_values


def export_excel(
    params: BuildingParams,
    raw_df: pd.DataFrame,
    result_df: pd.DataFrame,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    peak_cooling: pd.DataFrame,
    peak_heating: pd.DataFrame,
    peak_total: pd.DataFrame,
    figs: dict = None,
) -> bytes:
    """导出完整 Excel 文件，返回 bytes。"""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # ---- 01_计算说明 ----
        desc = pd.DataFrame({
            "说明": [
                "本程序基于建筑能耗基因理论和 CTM 特征温度法。",
                "",
                "计算流程：",
                "  1. 输入建筑参数和逐时气象数据",
                "  2. 计算各分项传热反应系数 H",
                "  3. 计算太阳空气温度",
                "  4. 计算各分项特征温度",
                "  5. 合成关机总特征温度 t∞_shut",
                "  6. 代入式 6-26 计算逐时冷负荷和供暖负荷",
                "  7. 额外计算潜热附加项（通风潜热、人员潜热）",
                "  8. 输出逐时结果、日汇总、月汇总、峰值小时、图表",
                "",
                "重要说明：",
                "  - 冷负荷显热（式6-26）与空调总冷负荷（含潜热）需要区分。",
                "  - 潜热附加项不属于式 6-26 显热负荷。",
                "  - 本 Excel 为数值版，不含逐单元格公式。",
            ]
        })
        desc.to_excel(writer, sheet_name="01_计算说明", index=False)

        # ---- 02_输入参数 ----
        param_rows = []
        for group_name, fields in PARAM_GROUPS.items():
            param_rows.append({"参数组": group_name, "参数名": "", "数值": "", "单位": "", "说明": ""})
            for field_name, label, unit in fields:
                val = getattr(params, field_name, "")
                param_rows.append({
                    "参数组": "",
                    "参数名": field_name,
                    "数值": val,
                    "单位": unit,
                    "说明": label,
                })
        # 额外：建筑面积
        param_rows.append({"参数组": "", "参数名": "floor_area", "数值": params.floor_area, "单位": "m2", "说明": "建筑面积（自动计算）"})

        param_df = pd.DataFrame(param_rows)
        param_df.to_excel(writer, sheet_name="02_输入参数", index=False)

        # ---- 03_原始输入数据 ----
        if raw_df is not None and len(raw_df) > 0:
            raw_df.to_excel(writer, sheet_name="03_原始输入数据", index=False)
        else:
            pd.DataFrame({"说明": ["无原始输入数据"]}).to_excel(writer, sheet_name="03_原始输入数据", index=False)

        # ---- 04_逐时CTM中间计算 ----
        if len(result_df) > 0:
            mid_cols = [
                "date", "time", "month", "hour",
                "outdoor_temp", "relative_humidity", "ghi",
                "solar_north", "solar_east", "solar_south", "solar_west",
                "occupancy_factor",
                "t_sa_east_deg_c", "t_sa_west_deg_c", "t_sa_south_deg_c",
                "t_sa_north_deg_c", "t_sa_roof_deg_c",
                "h_wall_w_per_k", "h_roof_w_per_k", "h_window_w_per_k",
                "h_door_w_per_k", "h_air_w_per_k", "h_total_w_per_k",
                "q_glass_w", "q_internal_sensible_w",
            ]
            available = [c for c in mid_cols if c in result_df.columns]
            result_df[available].to_excel(writer, sheet_name="04_逐时CTM中间计算", index=False)

        # ---- 05_分项特征温度 ----
        if len(result_df) > 0:
            t_cols = [
                "date", "time", "month", "hour",
                "t_inf_wall_deg_c", "t_inf_roof_deg_c", "t_inf_window_deg_c",
                "t_inf_door_deg_c", "t_inf_air_deg_c", "t_inf_glass_deg_c",
                "t_inf_internal_deg_c", "t_inf_shut_deg_c",
            ]
            available = [c for c in t_cols if c in result_df.columns]
            result_df[available].to_excel(writer, sheet_name="05_分项特征温度", index=False)

        # ---- 06_逐时冷热负荷结果 ----
        if len(result_df) > 0:
            load_cols = [
                "date", "time", "month", "hour",
                "t_inf_shut_deg_c",
                "delta_cooling_deg_c", "cooling_load_sensible_kw",
                "delta_heating_deg_c", "heating_load_kw",
                "outdoor_humidity_ratio", "indoor_humidity_ratio",
                "ventilation_latent_load_kw", "people_latent_load_kw",
                "cooling_load_total_kw",
                "cooling_energy_sensible_kwh", "heating_energy_kwh",
                "cooling_energy_total_kwh",
            ]
            available = [c for c in load_cols if c in result_df.columns]
            result_df[available].to_excel(writer, sheet_name="06_逐时冷热负荷结果", index=False)

        # ---- 07_日汇总 ----
        if len(daily) > 0:
            daily.to_excel(writer, sheet_name="07_日汇总", index=False)
        else:
            pd.DataFrame({"说明": ["数据不足，无法生成日汇总"]}).to_excel(writer, sheet_name="07_日汇总", index=False)

        # ---- 08_月汇总 ----
        if len(monthly) > 0:
            monthly.to_excel(writer, sheet_name="08_月汇总", index=False)
        else:
            pd.DataFrame({"说明": ["数据不足，无法生成月汇总"]}).to_excel(writer, sheet_name="08_月汇总", index=False)

        # ---- 09_峰值小时核查 ----
        frames = []
        if len(peak_cooling) > 0:
            frames.append(pd.DataFrame({"": ["=== 冷负荷显热峰值前20小时 ==="]}))
            frames.append(peak_cooling.reset_index())
        if len(peak_heating) > 0:
            frames.append(pd.DataFrame({"": ["=== 供暖负荷峰值前20小时 ==="]}))
            frames.append(peak_heating.reset_index())
        if len(peak_total) > 0:
            frames.append(pd.DataFrame({"": ["=== 空调总冷负荷峰值前20小时 ==="]}))
            frames.append(peak_total.reset_index())
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_excel(writer, sheet_name="09_峰值小时核查", index=False)
        else:
            pd.DataFrame({"说明": ["无峰值数据"]}).to_excel(writer, sheet_name="09_峰值小时核查", index=False)

        # ---- 10_图表 ----
        # 写入图表说明（图表在 openpyxl 中需要额外处理）
        chart_info = pd.DataFrame({
            "图表名称": list(figs.keys()) if figs else ["无图表数据"],
            "说明": ["请在 Streamlit 网页端查看交互式图表"] * (len(figs) if figs else 1),
        })
        chart_info.to_excel(writer, sheet_name="10_图表", index=False)

        # 尝试插入 plotly 图表图片（需要 kaleido，可选）
        if figs:
            try:
                from openpyxl.drawing.image import Image as XlImage
                ws = writer.sheets["10_图表"]
                row_idx = 3
                for name, fig in figs.items():
                    try:
                        img_bytes = fig.to_image(format="png", width=1000, height=500, scale=1)
                        img = XlImage(io.BytesIO(img_bytes))
                        img.anchor = f"A{row_idx}"
                        ws.add_image(img)
                        row_idx += 22
                    except Exception:
                        ws.cell(row=row_idx, column=1, value=f"{name}（图片导出需要 kaleido 库）")
                        row_idx += 1
            except ImportError:
                pass

        # ---- 11_公式说明与取值来源 ----
        formula_info = pd.DataFrame({
            "符号/变量": [
                "H_wall", "H_roof", "H_window", "H_door", "H_air", "H_total",
                "T_sa", "Q_glass", "f_occ",
                "Q_internal_sensible",
                "t_inf_wall", "t_inf_roof", "t_inf_window", "t_inf_door",
                "t_inf_air", "t_inf_glass", "t_inf_internal",
                "t_inf_shut",
                "式6-26 冷负荷", "式6-26 供暖负荷",
                "Q_lat_air", "Q_lat_people",
                "p_ws", "p_w", "w",
            ],
            "公式": [
                "U_wall × (A_E + A_W + A_S + A_N)",
                "U_roof × A_roof",
                "U_window × (A_win_E + A_win_W + A_win_S + A_win_N)",
                "U_door × A_door",
                "ρ × cp × n × V / 3600",
                "H_wall + H_roof + H_window + H_door + H_air",
                "T_out + α × I / h_o",
                "SHGC × shade × Σ(A_win × I_dir)",
                "占用时段内=1，否则=0",
                "f_occ × floor_area × (照明+设备+人员显热)",
                "U_wall × Σ(A × T_sa) / H_total",
                "U_roof × A_roof × T_sa_roof / H_total",
                "U_window × A_window_total × T_out / H_total",
                "U_door × A_door × T_out / H_total",
                "H_air × T_out / H_total",
                "Q_glass / H_total",
                "Q_internal / H_total",
                "Σt_inf（七项分项特征温度之和）",
                "max(0, (t_inf_shut - T_cool_set) × H_total / 1000)  [kW]",
                "max(0, (T_heat_set - t_inf_shut) × H_total / 1000)  [kW]",
                "ρ×n×V×h_fg×max(0,w_out-w_in)/(3600×1000)  [kW]",
                "f_occ×floor_area×occupant_density×person_latent_heat/1000  [kW]",
                "0.61078 × exp(17.2694×T/(T+237.3))  [kPa]",
                "RH/100 × p_ws  [kPa]",
                "0.62198 × p_w / (P - p_w)  [kg/kg]",
            ],
            "单位": [
                "W/K", "W/K", "W/K", "W/K", "W/K", "W/K",
                "℃", "W", "-",
                "W",
                "℃", "℃", "℃", "℃", "℃", "℃", "℃",
                "℃",
                "kW", "kW",
                "kW", "kW",
                "kPa", "kPa", "kg/kg",
            ],
            "取值来源": [
                "外墙传热系数×外墙面积",
                "屋面传热系数×屋面面积",
                "外窗传热系数×外窗面积",
                "外门传热系数×外门面积",
                "空气密度×比热×换气次数×体积/3600",
                "以上五项之和",
                "室外温度+太阳吸收率×辐射/室外换热系数",
                "SHGC×遮阳系数×各朝向外窗面积×辐射之和",
                "运行时间判断",
                "占用系数×建筑面积×(照明+设备+人员显热)",
                "各朝向外墙加权太阳空气温度/总H",
                "屋面加权太阳空气温度/总H",
                "外窗温差传热/总H",
                "外门温差传热/总H",
                "通风温差传热/总H",
                "玻璃太阳得热/总H",
                "内部得热/总H",
                "七项分项特征温度之和",
                "《建筑节能原理》式6-26",
                "《建筑节能原理》式6-26",
                "通风潜热公式",
                "人员潜热公式",
                "Antoine型饱和水汽压公式",
                "相对湿度×饱和水汽压",
                "ASHRAE 含湿量公式",
            ],
        })
        formula_info.to_excel(writer, sheet_name="11_公式说明与取值来源", index=False)

    output.seek(0)
    return output.getvalue()
