"""
流式 tool_call 解析器
支持 Qwen 模型输出的 JSON 格式工具调用
"""
import json
import re
from typing import List, Dict, Any, Optional


THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"
THINK_PATTERN = re.compile(THINK_OPEN + r'.*?' + THINK_CLOSE, re.DOTALL)


class StreamingToolCallParser:
    """
    流式 tool_call 解析器
    支持边生成边解析，检测 JSON 格式的 tool_call：
      {"tool_name": "xxx", "arguments": {...}}
    """

    def __init__(self):
        self.buffer = ""
        self.completed_calls: List[Dict[str, Any]] = []
        self._parsed_spans: list = []

    def feed(self, new_text: str) -> List[Dict[str, Any]]:
        if not new_text:
            return []
        self.buffer += new_text
        return self._extract_complete_calls()

    def _strip_think(self, text: str) -> str:
        """移除 think 块"""
        cleaned = THINK_PATTERN.sub('', text)
        idx = cleaned.find(THINK_OPEN)
        if idx != -1:
            cleaned = cleaned[:idx]
        return cleaned

    def _find_json_objects(self, text: str) -> List[tuple]:
        """
        用括号计数法找到文本中所有完整的顶层 JSON 对象。
        返回 [(start, end), ...] 列表。
        """
        results = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                depth = 0
                in_string = False
                escape_next = False
                start = i
                for j in range(i, len(text)):
                    ch = text[j]
                    if escape_next:
                        escape_next = False
                        continue
                    if ch == '\\' and in_string:
                        escape_next = True
                        continue
                    if ch == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            results.append((start, j + 1))
                            i = j + 1
                            break
                else:
                    # 未闭合，停止搜索
                    break
            else:
                i += 1
        return results

    def _extract_complete_calls(self) -> List[Dict[str, Any]]:
        new_calls = []
        text = self._strip_think(self.buffer)

        # 用括号计数法找到所有完整的 JSON 对象
        json_spans = self._find_json_objects(text)

        for start, end in json_spans:
            span = (start, end)
            if span in self._parsed_spans:
                continue

            json_str = text[start:end]
            call = self._try_parse_tool_json(json_str)
            if call:
                self._parsed_spans.append(span)
                call["id"] = "call_%d" % len(self.completed_calls)
                new_calls.append(call)
                self.completed_calls.append(call)

        return new_calls

    def _try_parse_tool_json(self, text: str) -> Optional[Dict[str, Any]]:
        """尝试将文本解析为工具调用 JSON"""
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

        return {
            "name": tool_name,
            "arguments": arguments,
        }

    def has_pending_json(self) -> bool:
        """检查缓冲区是否有未闭合的 JSON"""
        text = self._strip_think(self.buffer)
        depth = 0
        in_string = False
        escape_next = False
        for ch in text:
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
        return depth > 0

    def get_all_calls(self) -> List[Dict[str, Any]]:
        return self.completed_calls.copy()

    def reset(self):
        self.buffer = ""
        self.completed_calls = []
        self._parsed_spans = []
