"""建筑能耗基因理论 CTM 特征温度法 —— 逐时冷热负荷计算程序。"""
import sys
import os

# 确保 src 在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import json

from src.models import BuildingParams, PARAM_GROUPS
from src.validation import validate_params, validate_weather_dataframe
from src.calculator import (
    calc_h_values, calculate_hourly,
    calc_daily_summary, calc_monthly_summary,
    get_peak_hours,
)
from src.io_utils import (
    read_csv, read_excel, get_excel_sheet_names,
    make_single_hour_df, generate_template_csv,
    export_params_json, import_params_json, df_to_csv_bytes,
)
from src.plotting import plot_all
from src.excel_export import export_excel


def main():
    st.set_page_config(
        page_title="建筑能耗 CTM 特征温度法计算程序",
        page_icon="🏢",
        layout="wide",
    )
    st.title("建筑能耗基因理论 —— CTM 特征温度法逐时冷热负荷计算")

    # 初始化 session state
    if "params" not in st.session_state:
        st.session_state.params = BuildingParams()
    if "result_df" not in st.session_state:
        st.session_state.result_df = None
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None

    # ---- Tabs ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "项目说明", "建筑参数", "逐时数据输入", "计算结果", "图表与导出"
    ])

    # ==================== Tab 1: 项目说明 ====================
    with tab1:
        st.markdown("""
## 项目简介

本程序基于**建筑能耗基因理论**，采用 **CTM 特征温度法**（Characteristic Temperature Method）计算建筑逐时空调冷负荷和供暖热负荷。

## 理论依据

- 《建筑节能原理》—— 建筑能耗基因理论
- 式 6-26 冷负荷与供暖负荷计算公式
- CTM 特征温度法：通过计算各分项特征温度，合成关机总特征温度 t∞_shut，再代入式 6-26 得到逐时负荷

## 计算流程

1. **输入建筑参数**和逐时气象/太阳辐射数据
2. **计算各分项传热反应系数** H（外墙、屋面、外窗、外门、通风）
3. **计算太阳空气温度** T_sa（各朝向墙体和屋面）
4. **计算各分项特征温度** t_inf（七项：外墙、屋面、外窗、外门、通风、玻璃太阳得热、内部负荷）
5. **合成关机总特征温度** t∞_shut = Σt_inf
6. **代入式 6-26** 计算逐时冷负荷和供暖负荷
7. **额外计算潜热附加项**（通风潜热、人员潜热），与显热负荷分开
8. **输出**逐时结果、日汇总、月汇总、峰值小时、图表和 Excel 文件

## 重要说明

- **冷负荷显热（式 6-26）** 与 **空调总冷负荷（含潜热）** 需要区分
- 潜热附加项不属于式 6-26 显热负荷
- 默认时间步长为 1 小时，逐时 kW 求和即为 kWh

## 运行方式

```
cd building_load_gene
pip install -r requirements.txt
streamlit run app.py
```
        """)

    # ==================== Tab 2: 建筑参数 ====================
    with tab2:
        st.header("建筑参数设置")

        params: BuildingParams = st.session_state.params

        # 导入/导出按钮
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("恢复默认参数"):
                st.session_state.params = BuildingParams()
                st.rerun()
        with col_btn2:
            uploaded_json = st.file_uploader("导入参数 JSON", type=["json"], key="import_json")
            if uploaded_json is not None:
                try:
                    json_str = uploaded_json.read().decode("utf-8")
                    st.session_state.params = import_params_json(json_str)
                    st.success("参数导入成功！")
                    st.rerun()
                except Exception as e:
                    st.error(f"参数导入失败: {e}")
        with col_btn3:
            json_bytes = export_params_json(params).encode("utf-8")
            st.download_button(
                "导出参数 JSON",
                data=json_bytes,
                file_name="building_params.json",
                mime="application/json",
            )

        st.divider()

        # 分组显示参数
        new_values = {}
        for group_name, fields in PARAM_GROUPS.items():
            with st.expander(group_name, expanded=True):
                cols = st.columns(3)
                for i, (field_name, label, unit) in enumerate(fields):
                    val = getattr(params, field_name)
                    with cols[i % 3]:
                        if isinstance(val, int):
                            new_val = st.number_input(
                                f"{label} ({unit})", value=val, step=1, key=f"param_{field_name}"
                            )
                        else:
                            new_val = st.number_input(
                                f"{label} ({unit})", value=float(val), format="%.4f", key=f"param_{field_name}"
                            )
                        new_values[field_name] = new_val

        # 建筑面积计算方式
        with st.expander("建筑面积计算", expanded=True):
            floor_mode = st.radio(
                "建筑面积计算方式",
                ["自动估算 (体积/层高)", "手动输入"],
                index=0 if params.floor_area_mode == "auto" else 1,
            )
            if floor_mode == "自动估算 (体积/层高)":
                new_values["floor_area_mode"] = "auto"
                new_values["floor_area_manual"] = None
                auto_area = new_values.get("volume", params.volume) / new_values.get("floor_height", params.floor_height)
                st.info(f"自动计算建筑面积: {auto_area:.1f} m²")
            else:
                new_values["floor_area_mode"] = "manual"
                manual_area = st.number_input(
                    "手动输入建筑面积 (m²)",
                    value=float(params.floor_area_manual or params.floor_area),
                    format="%.1f",
                )
                new_values["floor_area_manual"] = manual_area

        # 更新参数
        for k, v in new_values.items():
            setattr(params, k, v)
        st.session_state.params = params

        # 参数校验
        st.divider()
        st.subheader("参数校验结果")
        errors, warnings = validate_params(params)
        if errors:
            for e in errors:
                st.error(e)
        if warnings:
            for w in warnings:
                st.warning(w)
        if not errors and not warnings:
            st.success("所有参数校验通过")

        # 显示关键计算值
        st.divider()
        st.subheader("关键参数汇总")
        h_vals = calc_h_values(params)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("建筑面积", f"{params.floor_area:.1f} m²")
            st.metric("外墙总面积", f"{params.wall_area_total:.1f} m²")
        with col2:
            st.metric("外窗总面积", f"{params.window_area_total:.1f} m²")
            st.metric("H_total", f"{h_vals['h_total_w_per_k']:.1f} W/K")
        with col3:
            st.metric("通风 H_air", f"{h_vals['h_air_w_per_k']:.1f} W/K")
            st.metric("外墙 H_wall", f"{h_vals['h_wall_w_per_k']:.1f} W/K")

    # ==================== Tab 3: 逐时数据输入 ====================
    with tab3:
        st.header("逐时数据输入")

        input_method = st.radio(
            "选择数据输入方式",
            ["上传 Excel", "上传 CSV", "手动输入单小时数据"],
            horizontal=True,
        )

        if input_method == "上传 Excel":
            uploaded_file = st.file_uploader("上传 Excel 文件", type=["xlsx", "xls"], key="excel_upload")
            if uploaded_file is not None:
                try:
                    sheet_names = get_excel_sheet_names(uploaded_file)
                    if len(sheet_names) > 1:
                        selected_sheet = st.selectbox("选择工作表", sheet_names)
                    else:
                        selected_sheet = sheet_names[0]
                    raw_df = read_excel(uploaded_file, sheet_name=selected_sheet)
                    st.session_state.raw_df = raw_df
                    st.success(f"成功读取工作表 '{selected_sheet}'，共 {len(raw_df)} 行")
                    st.dataframe(raw_df.head(10))
                    st.caption(f"识别到的列: {', '.join(raw_df.columns.tolist())}")
                except Exception as e:
                    st.error(f"读取 Excel 失败: {e}")

        elif input_method == "上传 CSV":
            uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"], key="csv_upload")
            if uploaded_file is not None:
                try:
                    raw_df = read_csv(uploaded_file)
                    st.session_state.raw_df = raw_df
                    st.success(f"成功读取 CSV，共 {len(raw_df)} 行")
                    st.dataframe(raw_df.head(10))
                    st.caption(f"识别到的列: {', '.join(raw_df.columns.tolist())}")
                except Exception as e:
                    st.error(f"读取 CSV 失败: {e}")

        elif input_method == "手动输入单小时数据":
            st.subheader("手动输入单小时气象数据")
            col1, col2 = st.columns(2)
            with col1:
                date_str = st.text_input("日期 (yyyy-mm-dd)", value="2005-08-22")
                time_str = st.text_input("时间 (HH:MM)", value="16:00")
                outdoor_temp = st.number_input("室外干球温度 (℃)", value=35.2, format="%.1f")
                relative_humidity = st.number_input("室外相对湿度 (%)", value=58.0, format="%.1f")
            with col2:
                ghi = st.number_input("水平面总辐射 (W/m²)", value=620.0, format="%.1f")
                solar_north = st.number_input("北向垂直面辐射 (W/m²)", value=80.0, format="%.1f")
                solar_east = st.number_input("东向垂直面辐射 (W/m²)", value=240.0, format="%.1f")
                solar_south = st.number_input("南向垂直面辐射 (W/m²)", value=360.0, format="%.1f")
                solar_west = st.number_input("西向垂直面辐射 (W/m²)", value=520.0, format="%.1f")

            if st.button("计算单小时负荷"):
                raw_df = make_single_hour_df(
                    date_str, time_str, outdoor_temp, relative_humidity,
                    ghi, solar_north, solar_east, solar_south, solar_west
                )
                st.session_state.raw_df = raw_df
                st.success("单小时数据已生成")

        # 数据校验与计算按钮
        if st.session_state.raw_df is not None:
            raw_df = st.session_state.raw_df
            st.divider()
            valid, errors, norm_df = validate_weather_dataframe(raw_df)

            if not valid:
                for e in errors:
                    st.error(e)
            else:
                st.success(f"数据校验通过，共 {len(norm_df)} 行有效数据")
                if st.button("开始计算", type="primary"):
                    with st.spinner("正在计算..."):
                        result_df = calculate_hourly(params, norm_df)
                        st.session_state.result_df = result_df
                    st.success(f"计算完成！共 {len(result_df)} 小时结果")

    # ==================== Tab 4: 计算结果 ====================
    with tab4:
        st.header("计算结果")

        result_df = st.session_state.result_df

        if result_df is None or len(result_df) == 0:
            st.info("请先在「逐时数据输入」标签页上传数据并点击「开始计算」")
        else:
            # 关键指标卡片
            st.subheader("关键指标")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                peak_cool = result_df["cooling_load_sensible_kw"].max()
                st.metric("冷负荷显热峰值", f"{peak_cool:.2f} kW")
            with col2:
                peak_heat = result_df["heating_load_kw"].max()
                st.metric("供暖负荷峰值", f"{peak_heat:.2f} kW")
            with col3:
                peak_total = result_df["cooling_load_total_kw"].max()
                st.metric("空调总冷负荷峰值", f"{peak_total:.2f} kW")
            with col4:
                sum_cool = result_df["cooling_energy_sensible_kwh"].sum()
                st.metric("冷负荷显热累计", f"{sum_cool:.1f} kWh")
            with col5:
                sum_heat = result_df["heating_energy_kwh"].sum()
                st.metric("供暖负荷累计", f"{sum_heat:.1f} kWh")
            with col6:
                sum_total = result_df["cooling_energy_total_kwh"].sum()
                st.metric("空调总冷负荷累计", f"{sum_total:.1f} kWh")

            # 逐时结果表
            st.divider()
            st.subheader("逐时结果表")
            display_cols = [
                "date", "time", "month", "hour",
                "outdoor_temp", "relative_humidity",
                "t_inf_shut_deg_c",
                "cooling_load_sensible_kw", "heating_load_kw",
                "cooling_load_total_kw",
            ]
            available = [c for c in display_cols if c in result_df.columns]
            st.dataframe(result_df[available], use_container_width=True)

            # 分项特征温度表
            st.divider()
            st.subheader("分项特征温度表")
            t_cols = [
                "date", "time",
                "t_inf_wall_deg_c", "t_inf_roof_deg_c", "t_inf_window_deg_c",
                "t_inf_door_deg_c", "t_inf_air_deg_c", "t_inf_glass_deg_c",
                "t_inf_internal_deg_c", "t_inf_shut_deg_c",
            ]
            available_t = [c for c in t_cols if c in result_df.columns]
            st.dataframe(result_df[available_t], use_container_width=True)

            # 日汇总
            daily = calc_daily_summary(result_df)
            if len(daily) > 0:
                st.divider()
                st.subheader("日汇总")
                st.dataframe(daily, use_container_width=True)

            # 月汇总
            monthly = calc_monthly_summary(result_df)
            if len(monthly) > 0:
                st.divider()
                st.subheader("月汇总")
                st.dataframe(monthly, use_container_width=True)

            # 峰值小时
            st.divider()
            st.subheader("峰值小时")

            peak_cool_df = get_peak_hours(result_df, "cooling_load_sensible_kw", 20)
            if len(peak_cool_df) > 0:
                st.markdown("**冷负荷显热前 20 小时**")
                st.dataframe(peak_cool_df, use_container_width=True)

            peak_heat_df = get_peak_hours(result_df, "heating_load_kw", 20)
            if len(peak_heat_df) > 0:
                st.markdown("**供暖负荷前 20 小时**")
                st.dataframe(peak_heat_df, use_container_width=True)

            peak_total_df = get_peak_hours(result_df, "cooling_load_total_kw", 20)
            if len(peak_total_df) > 0:
                st.markdown("**空调总冷负荷前 20 小时**")
                st.dataframe(peak_total_df, use_container_width=True)

    # ==================== Tab 5: 图表与导出 ====================
    with tab5:
        st.header("图表与导出")

        result_df = st.session_state.result_df

        if result_df is None or len(result_df) == 0:
            st.info("请先在「逐时数据输入」标签页上传数据并点击「开始计算」")
        else:
            # 计算汇总
            daily = calc_daily_summary(result_df)
            monthly = calc_monthly_summary(result_df)
            peak_cooling = get_peak_hours(result_df, "cooling_load_sensible_kw", 20)
            peak_heating = get_peak_hours(result_df, "heating_load_kw", 20)
            peak_total = get_peak_hours(result_df, "cooling_load_total_kw", 20)

            # 生成图表
            figs = plot_all(result_df, daily, monthly, peak_cooling, peak_heating)

            st.subheader("图表")
            for name, fig in figs.items():
                st.plotly_chart(fig, use_container_width=True)

            # 下载区
            st.divider()
            st.subheader("下载")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Excel
                try:
                    excel_bytes = export_excel(
                        params, st.session_state.raw_df, result_df,
                        daily, monthly,
                        peak_cooling, peak_heating, peak_total,
                        figs=figs,
                    )
                    st.download_button(
                        "下载完整 Excel",
                        data=excel_bytes,
                        file_name="CTM建筑负荷计算结果.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as e:
                    st.error(f"Excel 导出失败: {e}")

                # 逐时 CSV
                csv_bytes = df_to_csv_bytes(result_df)
                st.download_button(
                    "下载逐时结果 CSV",
                    data=csv_bytes,
                    file_name="逐时计算结果.csv",
                    mime="text/csv",
                )

            with col2:
                # 日汇总 CSV
                if len(daily) > 0:
                    daily_csv = df_to_csv_bytes(daily)
                    st.download_button(
                        "下载日汇总 CSV",
                        data=daily_csv,
                        file_name="日汇总.csv",
                        mime="text/csv",
                    )

                # 月汇总 CSV
                if len(monthly) > 0:
                    monthly_csv = df_to_csv_bytes(monthly)
                    st.download_button(
                        "下载月汇总 CSV",
                        data=monthly_csv,
                        file_name="月汇总.csv",
                        mime="text/csv",
                    )

            with col3:
                # 参数 JSON
                json_bytes = export_params_json(params).encode("utf-8")
                st.download_button(
                    "下载参数 JSON",
                    data=json_bytes,
                    file_name="building_params.json",
                    mime="application/json",
                )

                # 模板 CSV
                template_bytes = generate_template_csv()
                st.download_button(
                    "下载输入模板 CSV",
                    data=template_bytes,
                    file_name="输入模板.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
