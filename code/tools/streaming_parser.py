"""
流式 tool_call 解析器
"""
import json
import re
from typing import List, Dict, Any


class StreamingToolCallParser:
    """
    流式 tool_call 解析器
    支持边生成边解析，检测完整的 tool_call
    """
    
    def __init__(self):
        self.buffer = ""
        self.completed_calls: List[Dict[str, Any]] = []
        self.last_check_pos = 0
    
    def feed(self, new_text: str) -> List[Dict[str, Any]]:
        """
        喂入新生成的文本片段
        返回新检测到的完整 tool_call 列表
        """
        if not new_text:
            return []
        self.buffer += new_text
        return self._extract_complete_calls()
    
    def _extract_complete_calls(self) -> List[Dict[str, Any]]:
        """从缓冲区提取完整的 tool_call"""
        new_calls = []
        
        search_text = self.buffer[self.last_check_pos:]
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.finditer(pattern, search_text, re.DOTALL)
        
        for match in matches:
            tool_call_text = match.group(1)
            call_start_pos = self.last_check_pos + match.start()
            
            tool_call = self._parse_tool_call(tool_call_text)
            if tool_call:
                tool_call["id"] = f"call_{len(self.completed_calls)}"
                new_calls.append(tool_call)
                self.completed_calls.append(tool_call)
                
                # 更新检查位置到当前匹配结束位置
                self.last_check_pos = call_start_pos + match.end()
        
        return new_calls
    
    def _parse_tool_call(self, text: str) -> Dict[str, Any]:
        """解析单个 tool_call"""
        try:
            name_match = re.search(r"<name>(.*?)</name>", text)
            args_match = re.search(r"<arguments>(.*?)</arguments>", text, re.DOTALL)
            
            if not name_match or not args_match:
                return None
            
            tool_name = name_match.group(1).strip()
            arguments_str = args_match.group(1).strip()
            arguments = json.loads(arguments_str)
            
            return {
                "name": tool_name,
                "arguments": arguments
            }
        except Exception as e:
            print(f"[StreamingParser] Failed to parse tool_call: {e}")
            return None
    
    def get_all_calls(self) -> List[Dict[str, Any]]:
        """获取所有已完成的 tool_call"""
        return self.completed_calls.copy()
    
    def reset(self):
        """重置解析器"""
        self.buffer = ""
        self.completed_calls = []
        self.last_check_pos = 0
