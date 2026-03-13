"""
工具执行器 - 实际调用真实的API或函数
"""

import requests
import math
from datetime import datetime
import pytz


def execute_get_weather(city: str, unit: str = "celsius") -> dict:
    """
    获取天气信息 - 使用模拟数据（稳定可靠）
    
    注意：这是模拟实现，用于测试。实际应用中应替换为真实API。
    可选API：OpenWeatherMap, WeatherAPI.com, Visual Crossing等
    """
    # 模拟不同城市的天气数据
    weather_data = {
        "北京": {"temp_c": 15, "temp_f": 59, "condition": "晴", "humidity": "45%", "wind": "12 km/h"},
        "上海": {"temp_c": 18, "temp_f": 64, "condition": "多云", "humidity": "60%", "wind": "15 km/h"},
        "广州": {"temp_c": 25, "temp_f": 77, "condition": "阴", "humidity": "75%", "wind": "8 km/h"},
        "深圳": {"temp_c": 26, "temp_f": 79, "condition": "晴", "humidity": "70%", "wind": "10 km/h"},
        "杭州": {"temp_c": 20, "temp_f": 68, "condition": "小雨", "humidity": "80%", "wind": "18 km/h"},
        "成都": {"temp_c": 22, "temp_f": 72, "condition": "多云", "humidity": "65%", "wind": "5 km/h"},
        "beijing": {"temp_c": 15, "temp_f": 59, "condition": "Sunny", "humidity": "45%", "wind": "12 km/h"},
        "shanghai": {"temp_c": 18, "temp_f": 64, "condition": "Cloudy", "humidity": "60%", "wind": "15 km/h"},
    }
    
    # 默认天气（如果城市不在列表中）
    default_weather = {"temp_c": 20, "temp_f": 68, "condition": "晴", "humidity": "50%", "wind": "10 km/h"}
    
    # 获取天气数据
    weather = weather_data.get(city, weather_data.get(city.lower(), default_weather))
    
    temperature = weather["temp_f"] if unit == "fahrenheit" else weather["temp_c"]
    
    return {
        "city": city,
        "temperature": temperature,
        "unit": unit,
        "condition": weather["condition"],
        "humidity": weather["humidity"],
        "wind_speed": weather["wind"],
        "note": "模拟数据，仅用于测试"
    }


def execute_search_web(query: str, num_results: int = 5) -> dict:
    """
    网络搜索 - 使用 DuckDuckGo 即时答案API
    """
    try:
        # 使用 DuckDuckGo 的即时答案API
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            results = []
            
            # 获取摘要
            if data.get('AbstractText'):
                results.append({
                    "title": data.get('Heading', query),
                    "snippet": data['AbstractText'],
                    "url": data.get('AbstractURL', '')
                })
            
            # 获取相关主题
            for topic in data.get('RelatedTopics', [])[:num_results-1]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        "title": topic.get('Text', '')[:50],
                        "snippet": topic.get('Text', ''),
                        "url": topic.get('FirstURL', '')
                    })
            
            return {
                "query": query,
                "num_results": len(results),
                "results": results[:num_results]
            }
        else:
            return {"error": "搜索API调用失败"}
    except Exception as e:
        return {"error": f"搜索失败: {str(e)}"}


def execute_calculate(expression: str) -> dict:
    """
    数学计算 - 使用Python的math模块
    """
    try:
        # 安全的数学表达式求值
        # 只允许数字、运算符和math函数
        allowed_names = {
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'pow': math.pow,
            'pi': math.pi,
            'e': math.e
        }
        
        # 使用eval但限制命名空间
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        
        return {
            "expression": expression,
            "result": result
        }
    except Exception as e:
        return {"error": f"计算失败: {str(e)}"}


def execute_get_current_time(timezone: str = "Asia/Shanghai") -> dict:
    """
    获取当前时间
    """
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        
        return {
            "timezone": timezone,
            "datetime": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "date": current_time.strftime("%Y-%m-%d"),
            "time": current_time.strftime("%H:%M:%S"),
            "weekday": current_time.strftime("%A"),
            "timestamp": int(current_time.timestamp())
        }
    except Exception as e:
        return {"error": f"时间查询失败: {str(e)}"}


# 工具映射表
TOOL_FUNCTIONS = {
    "get_weather": execute_get_weather,
    "search_web": execute_search_web,
    "calculate": execute_calculate,
    "get_current_time": execute_get_current_time
}


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """
    执行工具调用
    
    Args:
        tool_name: 工具名称
        arguments: 工具参数
    
    Returns:
        工具执行结果
    """
    if tool_name not in TOOL_FUNCTIONS:
        return {"error": f"未知工具: {tool_name}"}
    
    try:
        func = TOOL_FUNCTIONS[tool_name]
        result = func(**arguments)
        return result
    except Exception as e:
        return {"error": f"工具执行失败: {str(e)}"}
