"""CTM 特征温度法计算程序单元测试。"""
import sys
import os
import io
import pytest
import pandas as pd
import numpy as np

# 确保 src 可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import BuildingParams
from src.calculator import calc_h_values, calculate_hourly, calc_daily_summary, calc_monthly_summary, get_peak_hours
from src.validation import validate_params, validate_weather_dataframe
from src.io_utils import make_single_hour_df, df_to_csv_bytes
from src.excel_export import export_excel


@pytest.fixture
def default_params():
    return BuildingParams()


@pytest.fixture
def single_hour_df():
    return make_single_hour_df("2005-08-22", "16:00", 35.2, 58, 620, 80, 240, 360, 520)


@pytest.fixture
def multi_hour_df():
    return pd.DataFrame([
        {"date": "2005-08-22", "time": "16:00", "outdoor_temp": 35.2, "relative_humidity": 58,
         "ghi": 620, "solar_north": 80, "solar_east": 240, "solar_south": 360, "solar_west": 520},
        {"date": "2005-01-21", "time": "07:00", "outdoor_temp": 1.5, "relative_humidity": 75,
         "ghi": 0, "solar_north": 0, "solar_east": 0, "solar_south": 0, "solar_west": 0},
        {"date": "2005-04-15", "time": "12:00", "outdoor_temp": 24.0, "relative_humidity": 65,
         "ghi": 700, "solar_north": 120, "solar_east": 300, "solar_south": 420, "solar_west": 300},
    ])


class TestHValues:
    def test_h_wall_positive(self, default_params):
        h = calc_h_values(default_params)
        assert h["h_wall_w_per_k"] > 0

    def test_h_roof_positive(self, default_params):
        h = calc_h_values(default_params)
        assert h["h_roof_w_per_k"] > 0

    def test_h_air_positive(self, default_params):
        h = calc_h_values(default_params)
        assert h["h_air_w_per_k"] > 0

    def test_h_total_positive(self, default_params):
        h = calc_h_values(default_params)
        assert h["h_total_w_per_k"] > 0

    def test_h_total_sum(self, default_params):
        h = calc_h_values(default_params)
        expected = h["h_wall_w_per_k"] + h["h_roof_w_per_k"] + h["h_window_w_per_k"] + h["h_door_w_per_k"] + h["h_air_w_per_k"]
        assert abs(h["h_total_w_per_k"] - expected) < 1e-6


class TestCalculation:
    def test_single_hour_calculation(self, default_params, single_hour_df):
        _, _, norm_df = validate_weather_dataframe(single_hour_df)
        result = calculate_hourly(default_params, norm_df)
        assert len(result) == 1
        assert "t_inf_shut_deg_c" in result.columns
        assert "cooling_load_sensible_kw" in result.columns
        assert "heating_load_kw" in result.columns

    def test_multi_hour_calculation(self, default_params, multi_hour_df):
        _, _, norm_df = validate_weather_dataframe(multi_hour_df)
        result = calculate_hourly(default_params, norm_df)
        assert len(result) == 3

    def test_cooling_load_non_negative(self, default_params, multi_hour_df):
        _, _, norm_df = validate_weather_dataframe(multi_hour_df)
        result = calculate_hourly(default_params, norm_df)
        assert (result["cooling_load_sensible_kw"] >= 0).all()

    def test_heating_load_non_negative(self, default_params, multi_hour_df):
        _, _, norm_df = validate_weather_dataframe(multi_hour_df)
        result = calculate_hourly(default_params, norm_df)
        assert (result["heating_load_kw"] >= 0).all()

    def test_latent_load_non_negative(self, default_params, multi_hour_df):
        _, _, norm_df = validate_weather_dataframe(multi_hour_df)
        result = calculate_hourly(default_params, norm_df)
        assert (result["ventilation_latent_load_kw"] >= 0).all()
        assert (result["people_latent_load_kw"] >= 0).all()

    def test_t_inf_components_sum(self, default_params, multi_hour_df):
        """分项特征温度之和等于总特征温度。"""
        _, _, norm_df = validate_weather_dataframe(multi_hour_df)
        result = calculate_hourly(default_params, norm_df)
        component_sum = (
            result["t_inf_wall_deg_c"]
            + result["t_inf_roof_deg_c"]
            + result["t_inf_window_deg_c"]
            + result["t_inf_door_deg_c"]
            + result["t_inf_air_deg_c"]
            + result["t_inf_glass_deg_c"]
            + result["t_inf_internal_deg_c"]
        )
        np.testing.assert_allclose(
            component_sum.values,
            result["t_inf_shut_deg_c"].values,
            rtol=1e-10,
        )

    def test_humidity_ratio_reasonable(self, default_params, single_hour_df):
        """含湿量应在合理范围内。"""
        _, _, norm_df = validate_weather_dataframe(single_hour_df)
        result = calculate_hourly(default_params, norm_df)
        w_out = result["outdoor_humidity_ratio"].iloc[0]
        w_in = result["indoor_humidity_ratio"].iloc[0]
        # 含湿量应在 0~0.05 kg/kg 范围内
        assert 0 <= w_out <= 0.05
        assert 0 <= w_in <= 0.05


class TestValidation:
    def test_missing_columns_validation(self):
        """缺列时应返回错误。"""
        df = pd.DataFrame({"date": ["2005-08-22"], "time": ["16:00"]})
        valid, errors, _ = validate_weather_dataframe(df)
        assert not valid
        assert any("缺少" in e for e in errors)

    def test_valid_params(self, default_params):
        errors, warnings = validate_params(default_params)
        assert len(errors) == 0


class TestExport:
    def test_export_excel(self, default_params, single_hour_df):
        """Excel 能成功导出且包含规定工作表。"""
        _, _, norm_df = validate_weather_dataframe(single_hour_df)
        result = calculate_hourly(default_params, norm_df)
        daily = calc_daily_summary(result)
        monthly = calc_monthly_summary(result)
        peak_cooling = get_peak_hours(result, "cooling_load_sensible_kw", 20)
        peak_heating = get_peak_hours(result, "heating_load_kw", 20)
        peak_total = get_peak_hours(result, "cooling_load_total_kw", 20)

        excel_bytes = export_excel(
            default_params, single_hour_df, result,
            daily, monthly,
            peak_cooling, peak_heating, peak_total,
        )
        assert len(excel_bytes) > 0

        # 检查工作表
        xls = pd.ExcelFile(io.BytesIO(excel_bytes))
        expected_sheets = [
            "01_计算说明", "02_输入参数", "03_原始输入数据",
            "04_逐时CTM中间计算", "05_分项特征温度", "06_逐时冷热负荷结果",
            "07_日汇总", "08_月汇总", "09_峰值小时核查",
            "10_图表", "11_公式说明与取值来源",
        ]
        for sheet in expected_sheets:
            assert sheet in xls.sheet_names, f"缺少工作表: {sheet}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
