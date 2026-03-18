"""
工具调用协调器 - 单机版本（无网络依赖）
"""
import re
import json
import uuid
from typing import List, Dict, Any, Optional, Callable

THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"
THINK_RE = re.compile(THINK_OPEN + r'.*?' + THINK_CLOSE, re.DOTALL)


class ToolCoordinator:
    """工具调用协调器 - 支持分布式工具执行"""

    def __init__(
        self,
        tool_manager,
        scheduler,
        local_device_id: int = 0,
        remote_call_handler: Optional[Callable] = None
    ):
        self.tool_manager = tool_manager
        self.scheduler = scheduler
        self.local_device_id = local_device_id
        self.remote_call_handler = remote_call_handler

    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """从文本中解析工具调用（JSON 格式）"""
        tool_calls = []

        # 移除 think 块
        text_clean = THINK_RE.sub('', text)
        idx = text_clean.find(THINK_OPEN)
        if idx != -1:
            text_clean = text_clean[:idx]

        # 策略1: 从 ```json ... ``` 代码块提取
        for m in re.finditer(r'```(?:json)?\s*(\{.*?\})\s*```', text_clean, re.DOTALL):
            call = self._try_parse(m.group(1))
            if call:
                tool_calls.append(call)

        if tool_calls:
            return tool_calls

        # 策略2: 直接提取 JSON 对象
        for m in re.finditer(r'\{[^{}]*(?:"tool_name"|"name")[^{}]*\}', text_clean):
            call = self._try_parse(m.group(0))
            if call:
                tool_calls.append(call)

        if tool_calls:
            return tool_calls

        # 策略3: 嵌套 JSON（arguments 内含 {}）
        start = text_clean.find("{")
        end = text_clean.rfind("}") + 1
        if start != -1 and end > start:
            call = self._try_parse(text_clean[start:end])
            if call:
                tool_calls.append(call)

        return tool_calls

    def _try_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """尝试解析单个 JSON 工具调用"""
        text = text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        tool_name = data.get("tool_name") or data.get("name") or ""
        arguments = data.get("arguments") or data.get("parameters") or {}

        if not tool_name:
            return None

        return {"name": tool_name, "arguments": arguments}

    def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行工具调用（支持本地和远程）"""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call['name']
            arguments = tool_call['arguments']

            tool_info = self.tool_manager.get_tool_info(tool_name)
            if not tool_info:
                results.append({
                    'success': False,
                    'error': "Tool '%s' not found" % tool_name,
                    'tool_name': tool_name,
                    'device_id': None
                })
                continue

            device_id = self.scheduler.schedule(
                tool_name,
                tool_size=tool_info['memory_size']
            )

            print("[ToolCoordinator] Executing '%s' on Device %d" % (tool_name, device_id))

            if device_id == self.local_device_id:
                result = self._execute_local(tool_name, device_id, arguments)
            else:
                result = self._execute_remote(tool_name, device_id, arguments)

            results.append(result)

            if result['success']:
                print("[ToolCoordinator] Success: %s" % result['result'])
            else:
                print("[ToolCoordinator] Error: %s" % result['error'])

        return results

    def _execute_local(self, tool_name, device_id, arguments):
        """在本地设备执行工具"""
        print("[ToolCoordinator] Local execution on Device %d" % device_id)
        return self.tool_manager.execute_tool(tool_name, device_id, arguments)

    def _execute_remote(self, tool_name, device_id, arguments):
        """在远程设备执行工具（单机版本不支持）"""
        # 单机版本：所有工具都在本地执行
        return {
            'success': False,
            'error': "Remote execution not supported in local mode",
            'tool_name': tool_name,
            'device_id': device_id
        }

    def format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """格式化工具结果为 JSON 文本（便于注入 prompt）"""
        formatted = []
        for result in results:
            if result['success']:
                formatted.append(json.dumps({
                    "tool_name": result['tool_name'],
                    "result": result['result']
                }, ensure_ascii=False))
            else:
                formatted.append(json.dumps({
                    "tool_name": result['tool_name'],
                    "error": result['error']
                }, ensure_ascii=False))
        return "\n".join(formatted)
