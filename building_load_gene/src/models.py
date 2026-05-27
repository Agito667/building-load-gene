"""建筑参数数据模型。"""
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class BuildingParams:
    """建筑参数。"""

    # 建筑几何参数
    volume: float = 27799.2                    # 建筑体积，m3
    floor_height: float = 3.9                  # 层高，m
    roof_area: float = 712.8                   # 屋面面积，m2

    # 各朝向外墙面积
    wall_area_east: float = 516.0              # 东向外墙净面积，m2
    wall_area_west: float = 516.0              # 西向外墙净面积，m2
    wall_area_south: float = 992.4             # 南向外墙净面积，m2
    wall_area_north: float = 992.4             # 北向外墙净面积，m2

    # 各朝向外窗面积
    window_area_east: float = 180.0            # 东向外窗面积，m2
    window_area_west: float = 180.0            # 西向外窗面积，m2
    window_area_south: float = 540.0           # 南向外窗面积，m2
    window_area_north: float = 500.0           # 北向外窗面积，m2

    # 外门
    door_area: float = 32.0                    # 外门面积，m2

    # 围护结构热工参数
    u_wall: float = 0.9                        # 外墙传热系数，W/(m2·K)
    u_roof: float = 0.7                        # 屋面传热系数，W/(m2·K)
    u_window: float = 2.5                      # 外窗传热系数，W/(m2·K)
    u_door: float = 2.3                        # 外门传热系数，W/(m2·K)

    shgc: float = 0.45                         # 外窗太阳得热系数
    shade_factor: float = 1.0                  # 遮阳修正系数
    solar_absorptance: float = 0.60            # 外表面太阳吸收率
    outdoor_heat_transfer_coeff: float = 19.0  # 室外综合换热系数，W/(m2·K)

    # 室内设计参数
    cooling_setpoint: float = 26.0             # 空调室内设定温度，℃
    heating_setpoint: float = 18.0             # 供暖室内设定温度，℃

    # 通风与空气参数
    air_change_rate: float = 0.5               # 换气次数，h-1
    air_density: float = 1.2                   # 空气密度，kg/m3
    air_specific_heat: float = 1006.0          # 空气定压比热，J/(kg·K)
    latent_heat_vaporization: float = 2501000.0  # 水汽化潜热，J/kg
    atmospheric_pressure: float = 101.325      # 大气压力，kPa
    indoor_cooling_rh: float = 60.0            # 室内冷房相对湿度，%

    # 内部负荷参数
    lighting_power_density: float = 9.0        # 照明功率密度，W/m2
    equipment_power_density: float = 12.0      # 设备功率密度，W/m2
    occupant_density: float = 0.10             # 人员密度，人/m2
    person_sensible_heat: float = 75.0         # 人员显热，W/人
    person_latent_heat: float = 55.0           # 人员潜热，W/人

    # 时间参数
    occupancy_start_hour: int = 8              # 占用开始时间
    occupancy_end_hour: int = 18               # 占用结束时间
    time_step_minutes: int = 60                # 计算时间步长，分钟（10~120，步进10）

    # 建筑面积计算方式
    floor_area_mode: str = "auto"              # "auto" 或 "manual"
    floor_area_manual: Optional[float] = None  # 手动输入的建筑面积

    @property
    def floor_area(self) -> float:
        """建筑面积，m2。"""
        if self.floor_area_mode == "manual" and self.floor_area_manual is not None:
            return self.floor_area_manual
        return self.volume / self.floor_height

    @property
    def wall_area_total(self) -> float:
        """外墙总面积，m2。"""
        return self.wall_area_east + self.wall_area_west + self.wall_area_south + self.wall_area_north

    @property
    def window_area_total(self) -> float:
        """外窗总面积，m2。"""
        return self.window_area_east + self.window_area_west + self.window_area_south + self.window_area_north

    def to_dict(self) -> dict:
        """转为字典。"""
        d = asdict(self)
        d["floor_area"] = self.floor_area
        d["wall_area_total"] = self.wall_area_total
        d["window_area_total"] = self.window_area_total
        return d

    def to_json(self, indent: int = 2) -> str:
        """转为 JSON 字符串。"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "BuildingParams":
        """从字典创建参数对象，忽略未知字段。"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> "BuildingParams":
        """从 JSON 字符串创建参数对象。"""
        return cls.from_dict(json.loads(json_str))


# 参数分组定义，用于界面显示
PARAM_GROUPS = {
    "建筑几何参数": [
        ("volume", "建筑体积", "m3"),
        ("floor_height", "层高", "m"),
        ("roof_area", "屋面面积", "m2"),
        ("wall_area_east", "东向外墙面积", "m2"),
        ("wall_area_west", "西向外墙面积", "m2"),
        ("wall_area_south", "南向外墙面积", "m2"),
        ("wall_area_north", "北向外墙面积", "m2"),
        ("window_area_east", "东向外窗面积", "m2"),
        ("window_area_west", "西向外窗面积", "m2"),
        ("window_area_south", "南向外窗面积", "m2"),
        ("window_area_north", "北向外窗面积", "m2"),
        ("door_area", "外门面积", "m2"),
    ],
    "围护结构热工参数": [
        ("u_wall", "外墙传热系数", "W/(m2·K)"),
        ("u_roof", "屋面传热系数", "W/(m2·K)"),
        ("u_window", "外窗传热系数", "W/(m2·K)"),
        ("u_door", "外门传热系数", "W/(m2·K)"),
        ("shgc", "外窗太阳得热系数", "-"),
        ("shade_factor", "遮阳修正系数", "-"),
        ("solar_absorptance", "外表面太阳吸收率", "-"),
        ("outdoor_heat_transfer_coeff", "室外综合换热系数", "W/(m2·K)"),
    ],
    "室内设计参数": [
        ("cooling_setpoint", "空调设定温度", "℃"),
        ("heating_setpoint", "供暖设定温度", "℃"),
        ("indoor_cooling_rh", "室内冷房相对湿度", "%"),
    ],
    "通风与空气参数": [
        ("air_change_rate", "换气次数", "h-1"),
        ("air_density", "空气密度", "kg/m3"),
        ("air_specific_heat", "空气定压比热", "J/(kg·K)"),
        ("latent_heat_vaporization", "水汽化潜热", "J/kg"),
        ("atmospheric_pressure", "大气压力", "kPa"),
    ],
    "内部负荷参数": [
        ("lighting_power_density", "照明功率密度", "W/m2"),
        ("equipment_power_density", "设备功率密度", "W/m2"),
        ("occupant_density", "人员密度", "人/m2"),
        ("person_sensible_heat", "人员显热", "W/人"),
        ("person_latent_heat", "人员潜热", "W/人"),
    ],
    "时间参数": [
        ("occupancy_start_hour", "占用开始时间", "小时"),
        ("occupancy_end_hour", "占用结束时间", "小时"),
        ("time_step_minutes", "时间步长", "分钟"),
    ],
}
