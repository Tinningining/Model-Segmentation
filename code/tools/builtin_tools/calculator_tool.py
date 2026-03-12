"""
计算器工具
"""


def execute(expression: str) -> float:
    """
    执行数学计算
    
    Args:
        expression: 数学表达式
    
    Returns:
        计算结果
    """
    try:
        # 安全的eval（仅支持数学运算）
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return float(result)
    except Exception as e:
        raise ValueError(f"计算错误: {e}")


TOOL_CONFIG = {
    'name': 'calculator',
    'module_path': 'tools.builtin_tools.calculator_tool',
    'memory_size': 10,
    'description': '执行数学计算',
    'parameters': {
        'expression': {'type': 'string', 'required': True, 'description': '数学表达式'}
    }
}
