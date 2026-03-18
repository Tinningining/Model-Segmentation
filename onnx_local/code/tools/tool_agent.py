"""
Tool Execution Agent - 单机版本（无网络依赖）
Runs locally to execute tools
"""
import threading
import uuid
from typing import Dict, Any, Optional
from queue import Queue, Empty

from tools.tool_manager import ToolManager


class ToolAgent:
    """Tool execution agent that runs on each node"""
    
    def __init__(self, device_id: int, tool_manager: ToolManager, node_name: str = "ToolAgent"):
        self.device_id = device_id
        self.tool_manager = tool_manager
        self.node_name = f"{node_name}-Device{device_id}"
        
        # Request queue for async execution
        self.request_queue = Queue()
        self.result_cache = {}
        
        # Worker thread
        self.worker_thread = None
        self.running = False
    
    def start(self):
        """Start the agent worker thread"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print(f"[{self.node_name}] Tool agent started")
    
    def stop(self):
        """Stop the agent"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        print(f"[{self.node_name}] Tool agent stopped")
    
    def _worker_loop(self):
        """Worker thread that processes tool execution requests"""
        while self.running:
            try:
                # Get request from queue with timeout
                request = self.request_queue.get(timeout=0.1)
                
                request_id = request.get('request_id')
                tool_name = request.get('tool_name')
                arguments = request.get('arguments', {})
                
                print(f"[{self.node_name}] Processing tool '{tool_name}' (request_id: {request_id})")
                
                # Execute tool
                result = self.tool_manager.execute_tool(
                    tool_name,
                    self.device_id,
                    arguments
                )
                
                # Cache result
                self.result_cache[request_id] = result
                
                if result['success']:
                    print(f"[{self.node_name}] ✓ Tool '{tool_name}' executed successfully")
                else:
                    print(f"[{self.node_name}] ✗ Tool '{tool_name}' failed: {result.get('error')}")
                
            except Empty:
                continue
            except Exception as e:
                print(f"[{self.node_name}] Worker error: {e}")
    
    def handle_tool_call(self, tool_call: dict) -> dict:
        """
        Handle tool call synchronously (单机版本简化接口)
        Returns tool result dict
        """
        request_id = tool_call.get('request_id', str(uuid.uuid4()))
        tool_name = tool_call.get('tool_name') or tool_call.get('name')
        arguments = tool_call.get('arguments', {})
        
        print(f"[{self.node_name}] Handling tool call '{tool_name}' (request_id: {request_id})")
        
        # Execute tool directly (synchronous)
        result = self.tool_manager.execute_tool(
            tool_name,
            self.device_id,
            arguments
        )
        
        # Return result dict
        result['request_id'] = request_id
        return result
    
    def submit_request(self, tool_name: str, arguments: dict) -> str:
        """
        Submit tool execution request (async)
        Returns request_id
        """
        request_id = str(uuid.uuid4())
        
        request = {
            'request_id': request_id,
            'tool_name': tool_name,
            'arguments': arguments
        }
        
        self.request_queue.put(request)
        return request_id
    
    def get_result(self, request_id: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """
        Get tool execution result (blocking)
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in self.result_cache:
                result = self.result_cache.pop(request_id)
                return result
            time.sleep(0.01)
        
        return None
    
    def list_loaded_tools(self) -> list:
        """List tools loaded on this device"""
        if self.device_id in self.tool_manager.loaded_tools:
            return list(self.tool_manager.loaded_tools[self.device_id].keys())
        return []
