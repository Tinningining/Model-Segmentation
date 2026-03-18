"""
工具结果缓冲区
"""
import threading
import time
from typing import Dict, Any, List, Optional


class ToolResultBuffer:
    """
    工具结果缓冲区（线程安全）
    """
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.pending_ids = set()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def add_pending(self, tool_call_id: str):
        """添加待执行的工具"""
        with self.lock:
            self.pending_ids.add(tool_call_id)
            self.results[tool_call_id] = None
            print(f"[ResultBuffer] Added pending tool: {tool_call_id}, total pending: {len(self.pending_ids)}")
    
    def add_result(self, tool_call_id: str, result: Dict[str, Any]):
        """添加工具结果"""
        with self.lock:
            if tool_call_id in self.pending_ids:
                self.results[tool_call_id] = result
                self.pending_ids.remove(tool_call_id)
                print(f"[ResultBuffer] Tool {tool_call_id} completed, remaining: {len(self.pending_ids)}")
                self.condition.notify_all()
    
    def is_all_complete(self) -> bool:
        """检查是否所有工具都完成"""
        with self.lock:
            return len(self.pending_ids) == 0
    
    def get_pending_count(self) -> int:
        """获取待完成工具数量"""
        with self.lock:
            return len(self.pending_ids)
    
    def wait_all_complete(self, timeout: Optional[float] = None) -> bool:
        """
        等待所有工具完成
        timeout: 超时时间（秒），None表示无限等待
        """
        with self.condition:
            if timeout is None:
                while not self.is_all_complete():
                    self.condition.wait()
                return True
            end_time = time.time() + timeout
            while not self.is_all_complete():
                remaining = end_time - time.time()
                if remaining <= 0:
                    return False
                self.condition.wait(timeout=remaining)
            return True
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """获取所有结果（按添加顺序）"""
        with self.lock:
            return [
                self.results[call_id]
                for call_id in self.results.keys()
                if self.results[call_id] is not None
            ]
    
    def reset(self):
        """重置缓冲区"""
        with self.lock:
            self.results.clear()
            self.pending_ids.clear()
