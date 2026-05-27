"""图表生成模块，所有图表标题和标签使用中文。"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def _make_fig(title: str, x_label: str, y_label: str) -> go.Figure:
    """创建带中文标签的空白图表。"""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        font=dict(size=12),
    )
    return fig


def plot_t_inf_shut_hourly(df: pd.DataFrame) -> go.Figure:
    """逐时关机总特征温度曲线。"""
    x = list(range(len(df)))
    fig = _make_fig("逐时关机总特征温度", "小时序号", "温度 (℃)")
    fig.add_trace(go.Scatter(
        x=x, y=df["t_inf_shut_deg_c"].values,
        mode="lines", name="关机总特征温度"
    ))
    # 加设定温度参考线
    if "cooling_setpoint" in df.columns:
        fig.add_hline(y=df["cooling_setpoint"].iloc[0], line_dash="dash",
                      line_color="red", annotation_text="空调设定温度")
    if "heating_setpoint" in df.columns:
        fig.add_hline(y=df["heating_setpoint"].iloc[0], line_dash="dash",
                      line_color="blue", annotation_text="供暖设定温度")
    return fig


def plot_cooling_load_hourly(df: pd.DataFrame) -> go.Figure:
    """逐时冷负荷曲线。"""
    x = list(range(len(df)))
    fig = _make_fig("逐时冷负荷（显热与总冷负荷）", "小时序号", "负荷 (kW)")
    fig.add_trace(go.Scatter(x=x, y=df["cooling_load_sensible_kw"].values,
                             mode="lines", name="冷负荷_显热_式6-26"))
    fig.add_trace(go.Scatter(x=x, y=df["cooling_load_total_kw"].values,
                             mode="lines", name="空调总冷负荷_含潜热"))
    return fig


def plot_heating_load_hourly(df: pd.DataFrame) -> go.Figure:
    """逐时供暖负荷曲线。"""
    x = list(range(len(df)))
    fig = _make_fig("逐时供暖负荷", "小时序号", "负荷 (kW)")
    fig.add_trace(go.Scatter(x=x, y=df["heating_load_kw"].values,
                             mode="lines", name="供暖负荷_式6-26"))
    return fig


def plot_daily_energy(daily: pd.DataFrame) -> go.Figure:
    """逐日冷热耗量图。"""
    fig = _make_fig("逐日冷热耗量", "日期", "能耗 (kWh)")
    fig.add_trace(go.Bar(x=daily.iloc[:, 0], y=daily["日冷负荷显热_式6-26_kWh"], name="日冷负荷显热"))
    fig.add_trace(go.Bar(x=daily.iloc[:, 0], y=daily["日供暖负荷_式6-26_kWh"], name="日供暖负荷"))
    fig.update_layout(barmode="group")
    return fig


def plot_monthly_energy(monthly: pd.DataFrame) -> go.Figure:
    """逐月冷热耗量图。"""
    fig = _make_fig("逐月冷热耗量", "月份", "能耗 (kWh)")
    fig.add_trace(go.Bar(x=monthly["月份"].astype(str), y=monthly["月冷负荷显热_式6-26_kWh"], name="月冷负荷显热"))
    fig.add_trace(go.Bar(x=monthly["月份"].astype(str), y=monthly["月供暖负荷_式6-26_kWh"], name="月供暖负荷"))
    fig.update_layout(barmode="group")
    return fig


def plot_monthly_peak(monthly: pd.DataFrame) -> go.Figure:
    """逐月峰值负荷图。"""
    fig = _make_fig("逐月峰值负荷", "月份", "峰值负荷 (kW)")
    fig.add_trace(go.Bar(x=monthly["月份"].astype(str), y=monthly["月冷负荷显热峰值_kW"], name="冷负荷显热峰值"))
    fig.add_trace(go.Bar(x=monthly["月份"].astype(str), y=monthly["月供暖负荷峰值_kW"], name="供暖负荷峰值"))
    fig.update_layout(barmode="group")
    return fig


def plot_component_t_inf(df: pd.DataFrame) -> go.Figure:
    """分项特征温度图（堆叠面积图）。"""
    x = list(range(len(df)))
    fig = _make_fig("分项特征温度", "小时序号", "温度 (℃)")
    components = [
        ("t_inf_wall_deg_c", "外墙"),
        ("t_inf_roof_deg_c", "屋面"),
        ("t_inf_window_deg_c", "外窗温差传热"),
        ("t_inf_door_deg_c", "外门"),
        ("t_inf_air_deg_c", "通风渗透"),
        ("t_inf_glass_deg_c", "玻璃太阳得热"),
        ("t_inf_internal_deg_c", "内部负荷"),
    ]
    for col, name in components:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[col].values, mode="lines", name=name, stackgroup="one"))
    return fig


def plot_top20_cooling(peak_df: pd.DataFrame) -> go.Figure:
    """冷负荷峰值前 20 小时柱状图。"""
    fig = _make_fig("冷负荷峰值前20小时", "排名", "冷负荷 (kW)")
    labels = [f"{r.get('date','')}\n{r.get('time','')}" for _, r in peak_df.iterrows()]
    fig.add_trace(go.Bar(x=list(range(1, len(peak_df)+1)),
                         y=peak_df["cooling_load_sensible_kw"].values,
                         name="冷负荷显热_式6-26",
                         text=labels, textposition="outside"))
    return fig


def plot_top20_heating(peak_df: pd.DataFrame) -> go.Figure:
    """供暖负荷峰值前 20 小时柱状图。"""
    fig = _make_fig("供暖负荷峰值前20小时", "排名", "供暖负荷 (kW)")
    labels = [f"{r.get('date','')}\n{r.get('time','')}" for _, r in peak_df.iterrows()]
    fig.add_trace(go.Bar(x=list(range(1, len(peak_df)+1)),
                         y=peak_df["heating_load_kw"].values,
                         name="供暖负荷_式6-26",
                         text=labels, textposition="outside"))
    return fig


def plot_all(df: pd.DataFrame, daily: pd.DataFrame, monthly: pd.DataFrame,
             peak_cooling: pd.DataFrame, peak_heating: pd.DataFrame) -> dict:
    """生成所有图表，返回字典。"""
    figs = {}
    if len(df) > 0:
        figs["逐时关机总特征温度"] = plot_t_inf_shut_hourly(df)
        figs["逐时冷负荷"] = plot_cooling_load_hourly(df)
        figs["逐时供暖负荷"] = plot_heating_load_hourly(df)
        figs["分项特征温度"] = plot_component_t_inf(df)
    if len(daily) > 0:
        figs["逐日冷热耗量"] = plot_daily_energy(daily)
    if len(monthly) > 0:
        figs["逐月冷热耗量"] = plot_monthly_energy(monthly)
        figs["逐月峰值负荷"] = plot_monthly_peak(monthly)
    if len(peak_cooling) > 0:
        figs["冷负荷峰值前20小时"] = plot_top20_cooling(peak_cooling)
    if len(peak_heating) > 0:
        figs["供暖负荷峰值前20小时"] = plot_top20_heating(peak_heating)
    return figs
