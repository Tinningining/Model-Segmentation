from mcp_tools.base import MCPTool


class CalculatorTool(MCPTool):
    name = "calculator"
    description = "calculator(expression: string) - 计算数学表达式"
    schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string"}
        },
        "required": ["expression"]
    }

    def _load(self):
        pass

    def _run(self, expression: str):
        return eval(expression, {"__builtins__": {}})

    def _unload(self):
        pass