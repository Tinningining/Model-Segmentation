"""
MCP Host
负责：
- 构造上下文
- 调用模型
- 解析工具调用
- 执行工具
- 回填结果
"""

from mcp.prompt import build_system_prompt
from mcp.parser import parse_tool_call
from mcp.tools import TOOLS


class MCPHost:
    def __init__(self, llm):
        """
        llm: 你的 ONNX LLM 封装
        需要提供：
            llm.generate(prompt: str, kv_cache=None) -> (text, kv_cache)
        """
        self.llm = llm
        self.tools = TOOLS

    def _format_tool_result(self, tool_name, arguments, result):
        return (
            "\n【工具调用】\n"
            f"{tool_name}({arguments})\n"
            "【工具返回】\n"
            f"{result}\n"
        )

    def run(self, user_question: str, max_steps: int = 5) -> str:
        system_prompt = build_system_prompt(self.tools)
        context = (
            system_prompt +
            "\n【用户问题】\n" +
            user_question +
            "\n"
        )

        kv_cache = None

        for step in range(max_steps):
            output, kv_cache = self.llm.generate(context, kv_cache)

            tool_call = parse_tool_call(output)
            if tool_call is None:
                # 模型给出了最终答案
                return output.strip()

            tool_name = tool_call.tool
            arguments = tool_call.arguments

            if tool_name not in self.tools:
                context += f"\n【错误】未知工具：{tool_name}\n"
                continue

            try:
                tool_func = self.tools[tool_name]["func"]
                result = tool_func(**arguments)
            except Exception as e:
                result = f"工具执行失败：{str(e)}"

            context += self._format_tool_result(
                tool_name, arguments, result
            )

        return "【错误】推理未在限定步数内完成"
