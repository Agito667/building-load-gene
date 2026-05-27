"""输入输出工具模块。"""
import io
import json
import pandas as pd
from typing import Optional, Tuple

from .models import BuildingParams
from .validation import normalize_columns, REQUIRED_COLUMNS


def read_csv(file) -> pd.DataFrame:
    """读取上传的 CSV 文件。"""
    return pd.read_csv(file)


def read_excel(file, sheet_name=None) -> pd.DataFrame:
    """读取上传的 Excel 文件。"""
    return pd.read_excel(file, sheet_name=sheet_name)


def get_excel_sheet_names(file) -> list:
    """获取 Excel 文件的工作表名列表。"""
    xls = pd.ExcelFile(file)
    return xls.sheet_names


def make_single_hour_df(
    date_str: str,
    time_str: str,
    outdoor_temp: float,
    relative_humidity: float,
    ghi: float,
    solar_north: float,
    solar_east: float,
    solar_south: float,
    solar_west: float,
) -> pd.DataFrame:
    """从手动输入创建单行 DataFrame。"""
    return pd.DataFrame([{
        "date": date_str,
        "time": time_str,
        "outdoor_temp": outdoor_temp,
        "relative_humidity": relative_humidity,
        "ghi": ghi,
        "solar_north": solar_north,
        "solar_east": solar_east,
        "solar_south": solar_south,
        "solar_west": solar_west,
    }])


def generate_template_csv() -> bytes:
    """生成输入模板 CSV。"""
    template = pd.DataFrame(columns=REQUIRED_COLUMNS)
    # 加一行示例
    template.loc[0] = ["2005-08-22", "16:00", 35.2, 58, 620, 80, 240, 360, 520]
    return template.to_csv(index=False).encode("utf-8-sig")


def export_params_json(params: BuildingParams) -> str:
    """导出参数 JSON。"""
    return params.to_json()


def import_params_json(json_str: str) -> BuildingParams:
    """从 JSON 字符串导入参数。"""
    return BuildingParams.from_json(json_str)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """DataFrame 转 CSV bytes。"""
    return df.to_csv(index=False).encode("utf-8-sig")
