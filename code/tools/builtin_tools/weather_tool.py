"""
天气查询工具
"""


def execute(city: str, date: str = "today") -> dict:
    """
    查询天气信息
    
    Args:
        city: 城市名称
        date: 日期（默认今天）
    
    Returns:
        天气信息字典
    """
    # 模拟天气API调用
    weather_data = {
        "北京": {"temperature": 15, "condition": "晴", "wind": "微风"},
        "上海": {"temperature": 20, "condition": "多云", "wind": "东风"},
        "广州": {"temperature": 25, "condition": "阴", "wind": "南风"},
    }
    
    result = weather_data.get(city, {"temperature": 18, "condition": "未知", "wind": "无风"})
    result["city"] = city
    result["date"] = date
    
    return result


TOOL_CONFIG = {
    'name': 'get_weather',
    'module_path': 'tools.builtin_tools.weather_tool',
    'memory_size': 30,
    'description': '查询指定城市的天气信息',
    'parameters': {
        'city': {'type': 'string', 'required': True, 'description': '城市名称'},
        'date': {'type': 'string', 'required': False, 'default': 'today', 'description': '日期'}
    }
}
