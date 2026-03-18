"""
时间日期查询工具
"""
from datetime import datetime, timedelta


# 时区偏移（模拟）
TIMEZONE_OFFSETS = {
    "北京": 8, "上海": 8, "东京": 9, "纽约": -5, "伦敦": 0,
    "巴黎": 1, "悉尼": 11, "洛杉矶": -8, "莫斯科": 3, "迪拜": 4,
    "新加坡": 8, "首尔": 9, "柏林": 1, "芝加哥": -6, "香港": 8,
}


def execute(city: str = "北京", format: str = "full") -> dict:
    """
    查询指定城市的当前时间

    Args:
        city: 城市名称，默认北京
        format: 输出格式 full=完整 / time=仅时间 / date=仅日期

    Returns:
        时间信息字典
    """
    offset = TIMEZONE_OFFSETS.get(city, 8)
    utc_now = datetime.utcnow()
    local_time = utc_now + timedelta(hours=offset)

    result = {
        "city": city,
        "timezone": "UTC%+d" % offset,
    }

    if format == "time":
        result["time"] = local_time.strftime("%H:%M:%S")
    elif format == "date":
        result["date"] = local_time.strftime("%Y-%m-%d")
        result["weekday"] = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][local_time.weekday()]
    else:
        result["datetime"] = local_time.strftime("%Y-%m-%d %H:%M:%S")
        result["date"] = local_time.strftime("%Y-%m-%d")
        result["time"] = local_time.strftime("%H:%M:%S")
        result["weekday"] = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][local_time.weekday()]

    return result


TOOL_CONFIG = {
    'name': 'get_time',
    'module_path': 'tools.builtin_tools.time_tool',
    'memory_size': 10,
    'description': '查询指定城市的当前时间和日期',
    'parameters': {
        'city': {'type': 'string', 'required': False, 'default': '北京',
                 'description': '城市名称，如：北京、东京、纽约、伦敦'},
        'format': {'type': 'string', 'required': False, 'default': 'full',
                   'description': '输出格式', 'enum': ['full', 'time', 'date']},
    }
}
