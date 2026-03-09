"""
MCP 工具实现层
注意：
- 这里不包含任何模型逻辑
- 所有工具都是纯函数
"""

def search(query: str) -> str:
    # 示例：实际可替换为搜索引擎 / 向量库
    return f"【搜索结果】法国的首都是巴黎。（query={query}）"


def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"【计算结果】{expression} = {result}"
    except Exception as e:
        return f"【计算错误】{str(e)}"


TOOLS = {
    "search": {
        "description": "search(query: string) - 搜索事实性信息",
        "func": search
    },
    "calculator": {
        "description": "calculator(expression: string) - 计算数学表达式",
        "func": calculator
    }
}
