from typing import Dict, Type
from mcp_tools.base import MCPTool
from mcp_tools.impl.search import SearchTool
from mcp_tools.impl.calculator import CalculatorTool


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Type[MCPTool]] = {}

    def register(self, tool_cls: Type[MCPTool]):
        self._tools[tool_cls.name] = tool_cls

    def get(self, name: str) -> Type[MCPTool]:
        return self._tools.get(name)

    def list_tools(self):
        return list(self._tools.keys())


def default_registry():
    reg = ToolRegistry()
    reg.register(SearchTool)
    reg.register(CalculatorTool)
    return reg
