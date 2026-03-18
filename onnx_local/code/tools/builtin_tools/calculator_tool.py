"""
计算器工具
"""
import math


def execute(expression: str) -> dict:
    """
    执行数学计算

    Args:
        expression: 数学表达式，如 2+2, sqrt(16), sin(3.14)

    Returns:
        计算结果字典
    """
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
        'e': math.e,
        'abs': abs,
        'round': round,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {
            "expression": expression,
            "result": result,
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": "计算失败: %s" % str(e),
        }


# 内部工具配置（用于 ToolManager 注册）
TOOL_CONFIG = {
    'name': 'calculator',
    'module_path': 'tools.builtin_tools.calculator_tool',
    'memory_size': 10,
    'description': '执行数学计算',
    'parameters': {
        'expression': {'type': 'string', 'required': True,
                       'description': '数学表达式，如：2+2, sqrt(16), sin(3.14)'},
    }
}
