import json
import re

class ToolCall:
    def __init__(self, tool: str, arguments: dict):
        self.tool = tool
        self.arguments = arguments


def parse_tool_call(text: str):
    """
    从模型输出中解析 MCP 工具调用

    返回：
    - ToolCall 实例
    - 或 None（表示不是工具调用）
    """
    try:
        # 提取第一个 JSON 对象
        match = re.search(r"\{[\s\S]*?\}", text)
        if not match:
            return None

        obj = json.loads(match.group())

        if "tool" not in obj or "arguments" not in obj:
            return None

        if not isinstance(obj["arguments"], dict):
            return None

        return ToolCall(obj["tool"], obj["arguments"])

    except Exception:
        return None
