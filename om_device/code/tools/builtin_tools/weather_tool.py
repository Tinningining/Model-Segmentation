"""
天气查询工具 - 调用 wttr.in API（无需 API Key）
"""
import json

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


def execute(city: str, unit: str = "celsius") -> dict:
    """
    查询天气信息（调用 wttr.in API）

    Args:
        city: 城市名称
        unit: 温度单位 (celsius/fahrenheit)

    Returns:
        天气信息字典
    """
    if not HAS_URLLIB:
        return _fallback(city, unit)

    try:
        # wttr.in 支持中文城市名，返回 JSON
        url = "https://wttr.in/%s?format=j1" % urllib.request.quote(city)
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        current = data.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C", "N/A")
        temp_f = current.get("temp_F", "N/A")
        humidity = current.get("humidity", "N/A")
        wind_speed = current.get("windspeedKmph", "N/A")
        # 天气描述（中文）
        desc_list = current.get("lang_zh", [])
        if desc_list:
            condition = desc_list[0].get("value", "")
        else:
            condition = current.get("weatherDesc", [{}])[0].get("value", "N/A")

        temperature = temp_f if unit == "fahrenheit" else temp_c

        return {
            "city": city,
            "temperature": temperature,
            "unit": unit,
            "condition": condition,
            "humidity": "%s%%" % humidity,
            "wind_speed": "%s km/h" % wind_speed,
            "source": "wttr.in",
        }
    except Exception as e:
        # API 调用失败时使用模拟数据
        print("[weather_tool] API failed (%s), using fallback" % str(e))
        return _fallback(city, unit)


def _fallback(city: str, unit: str) -> dict:
    """模拟数据兜底"""
    fallback_data = {
        "北京": {"temp_c": 15, "temp_f": 59, "condition": "晴", "humidity": "45%", "wind": "12 km/h"},
        "上海": {"temp_c": 18, "temp_f": 64, "condition": "多云", "humidity": "60%", "wind": "15 km/h"},
        "广州": {"temp_c": 25, "temp_f": 77, "condition": "阴", "humidity": "75%", "wind": "8 km/h"},
        "深圳": {"temp_c": 26, "temp_f": 79, "condition": "晴", "humidity": "70%", "wind": "10 km/h"},
    }
    default = {"temp_c": 20, "temp_f": 68, "condition": "晴", "humidity": "50%", "wind": "10 km/h"}
    w = fallback_data.get(city, default)
    temperature = w["temp_f"] if unit == "fahrenheit" else w["temp_c"]
    return {
        "city": city, "temperature": temperature, "unit": unit,
        "condition": w["condition"], "humidity": w["humidity"],
        "wind_speed": w["wind"], "source": "fallback",
    }


TOOL_CONFIG = {
    'name': 'get_weather',
    'module_path': 'tools.builtin_tools.weather_tool',
    'memory_size': 30,
    'description': '获取指定城市的实时天气信息',
    'parameters': {
        'city': {'type': 'string', 'required': True, 'description': '城市名称，如：北京、上海、Tokyo、London'},
        'unit': {'type': 'string', 'required': False, 'default': 'celsius',
                 'description': '温度单位', 'enum': ['celsius', 'fahrenheit']},
    }
}
