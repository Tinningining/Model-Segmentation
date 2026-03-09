"""
定义 MCP 工具的 JSON Schema
用于：
- Prompt 约束
- 输出校验
"""

TOOL_CALL_SCHEMA = """
工具调用必须严格符合以下 JSON Schema：

{
  "tool": string,          // 工具名称
  "arguments": object      // 参数对象
}

示例：
{
  "tool": "search",
  "arguments": {
    "query": "法国的首都是哪里"
  }
}
"""
