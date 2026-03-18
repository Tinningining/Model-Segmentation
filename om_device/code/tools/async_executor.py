"""
异步工具执行器
"""
import concurrent.futures
from typing import Dict, Any, Callable, Optional

from .result_buffer import ToolResultBuffer


class AsyncToolExecutor:
    """
    异步工具执行器
    支持并行执行多个工具，不阻塞主线程
    """
    
    def __init__(self, tool_coordinator, max_workers: int = 4):
        self.tool_coordinator = tool_coordinator
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ToolExecutor"
        )
        self.result_buffer = ToolResultBuffer()
        self.futures = {}
    
    def execute_async(
        self,
        tool_call: Dict[str, Any],
        callback: Optional[Callable] = None
    ) -> concurrent.futures.Future:
        """
        异步执行工具
        
        tool_call: {'id', 'name', 'arguments'}
        """
        tool_call_id = tool_call["id"]
        
        self.result_buffer.add_pending(tool_call_id)
        
        future = self.executor.submit(self._execute_tool_call, tool_call)
        self.futures[tool_call_id] = future
        
        future.add_done_callback(
            lambda f: self._on_complete(tool_call_id, f, callback)
        )
        
        return future
    
    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个工具调用（同步）"""
        results = self.tool_coordinator.execute_tools([tool_call])
        return results[0] if results else {
            "success": False,
            "error": "Empty tool execution result",
            "tool_name": tool_call.get("name"),
            "device_id": None
        }
    
    def _on_complete(
        self,
        tool_call_id: str,
        future: concurrent.futures.Future,
        callback: Optional[Callable]
    ):
        """工具执行完成回调"""
        try:
            result = future.result()
            self.result_buffer.add_result(tool_call_id, result)
            
            if callback:
                callback(tool_call_id, result)
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "tool_name": tool_call_id,
                "device_id": -1
            }
            self.result_buffer.add_result(tool_call_id, error_result)
            print(f"[AsyncExecutor] Tool {tool_call_id} failed: {e}")
    
    def wait_all_complete(self, timeout: Optional[float] = None):
        """等待所有工具完成"""
        success = self.result_buffer.wait_all_complete(timeout)
        if not success:
            raise TimeoutError(f"Tool execution timeout after {timeout}s")
        return self.result_buffer.get_all_results()
    
    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
