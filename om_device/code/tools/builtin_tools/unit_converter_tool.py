"""
单位换算工具
"""

# 换算表：(from_unit, to_unit) -> multiplier
CONVERSIONS = {
    # 长度
    ("km", "m"): 1000, ("m", "km"): 0.001,
    ("m", "cm"): 100, ("cm", "m"): 0.01,
    ("km", "mile"): 0.621371, ("mile", "km"): 1.60934,
    ("m", "ft"): 3.28084, ("ft", "m"): 0.3048,
    ("cm", "inch"): 0.393701, ("inch", "cm"): 2.54,
    # 重量
    ("kg", "g"): 1000, ("g", "kg"): 0.001,
    ("kg", "lb"): 2.20462, ("lb", "kg"): 0.453592,
    ("g", "oz"): 0.035274, ("oz", "g"): 28.3495,
    # 温度（特殊处理）
    # 体积
    ("l", "ml"): 1000, ("ml", "l"): 0.001,
    ("l", "gallon"): 0.264172, ("gallon", "l"): 3.78541,
    # 面积
    ("km2", "m2"): 1e6, ("m2", "km2"): 1e-6,
    ("m2", "ft2"): 10.7639, ("ft2", "m2"): 0.092903,
    # 速度
    ("km/h", "m/s"): 0.277778, ("m/s", "km/h"): 3.6,
    ("km/h", "mph"): 0.621371, ("mph", "km/h"): 1.60934,
}


def execute(value: float, from_unit: str, to_unit: str) -> dict:
    """
    单位换算

    Args:
        value: 数值
        from_unit: 源单位
        to_unit: 目标单位

    Returns:
        换算结果字典
    """
    from_u = from_unit.lower().strip()
    to_u = to_unit.lower().strip()

    # 温度特殊处理
    if from_u in ("celsius", "c") and to_u in ("fahrenheit", "f"):
        result = value * 9 / 5 + 32
        return {"value": value, "from_unit": from_unit, "to_unit": to_unit, "result": round(result, 2)}
    if from_u in ("fahrenheit", "f") and to_u in ("celsius", "c"):
        result = (value - 32) * 5 / 9
        return {"value": value, "from_unit": from_unit, "to_unit": to_unit, "result": round(result, 2)}

    # 查表换算
    key = (from_u, to_u)
    if key in CONVERSIONS:
        result = value * CONVERSIONS[key]
        return {"value": value, "from_unit": from_unit, "to_unit": to_unit, "result": round(result, 4)}

    # 同单位
    if from_u == to_u:
        return {"value": value, "from_unit": from_unit, "to_unit": to_unit, "result": value}

    return {"value": value, "from_unit": from_unit, "to_unit": to_unit,
            "error": "不支持从 %s 到 %s 的换算" % (from_unit, to_unit)}


TOOL_CONFIG = {
    'name': 'unit_convert',
    'module_path': 'tools.builtin_tools.unit_converter_tool',
    'memory_size': 10,
    'description': '单位换算，支持长度、重量、温度、体积、面积、速度等常见单位',
    'parameters': {
        'value': {'type': 'number', 'required': True, 'description': '要换算的数值'},
        'from_unit': {'type': 'string', 'required': True,
                      'description': '源单位，如：km, m, kg, lb, celsius, l, km/h'},
        'to_unit': {'type': 'string', 'required': True,
                    'description': '目标单位，如：mile, ft, g, oz, fahrenheit, gallon, mph'},
    }
}
