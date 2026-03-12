"""
Tool Execution Agent
Runs on each node to execute tools locally
"""
import threading
import uuid
from typing import Dict, Any, Optional
from queue import Queue, Empty

from tools.tool_manager import ToolManager
from network import DistributedMessage


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
    
    def handle_tool_call(self, msg: DistributedMessage) -> DistributedMessage:
        """
        Handle tool call message synchronously
        Returns tool result message
        """
        request_id = msg.data.get('request_id')
        tool_name = msg.data.get('tool_name')
        arguments = msg.data.get('arguments', {})
        
        print(f"[{self.node_name}] Handling tool call '{tool_name}' (request_id: {request_id})")
        
        # Execute tool directly (synchronous)
        result = self.tool_manager.execute_tool(
            tool_name,
            self.device_id,
            arguments
        )
        
        # Create result message
        return DistributedMessage.create_tool_result_msg(
            request_id=request_id,
            success=result['success'],
            result=result.get('result'),
            error=result.get('error')
        )
    
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
