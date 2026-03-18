"""
工具管理器 - 管理多设备上的工具
"""
import time
import importlib
import threading
from typing import Dict, Any, Optional


class ToolManager:
    """工具管理器 - 管理多设备上的工具（线程安全）"""
    
    def __init__(self, devices: list, device_memory_limit: int = 500):
        self.devices = devices
        self.device_memory_limit = device_memory_limit
        
        # 工具注册表
        self.tool_registry = {}
        
        # 每个设备已加载的工具
        self.loaded_tools = {i: {} for i in devices}
        
        # 每个设备的内存使用
        self.device_memory = {i: 0 for i in devices}
        
        # 工具使用频率统计（用于改进 LRU 策略）
        self.tool_usage_count = {i: {} for i in devices}
        
        # 线程锁（保护并发访问）
        self._lock = threading.RLock()
        self._device_locks = {i: threading.Lock() for i in devices}
    
    def register_tool(self, tool_name: str, tool_config: Dict[str, Any]):
        """注册工具（线程安全）"""
        with self._lock:
            self.tool_registry[tool_name] = {
                'name': tool_name,
                'module_path': tool_config['module_path'],
                'memory_size': tool_config.get('memory_size', 50),
                'description': tool_config.get('description', ''),
                'parameters': tool_config.get('parameters', {}),
                'handler': None
            }
            print(f"[ToolManager] ✓ Tool '{tool_name}' registered")
    
    def load_tool(self, tool_name: str, device_id: int) -> bool:
        """在指定设备上加载工具（线程安全）"""
        with self._device_locks[device_id]:
            if tool_name not in self.tool_registry:
                print(f"[ToolManager] ✗ Tool '{tool_name}' not registered")
                return False
            
            # 检查是否已加载
            if tool_name in self.loaded_tools[device_id]:
                # 更新访问时间和使用计数
                self.loaded_tools[device_id][tool_name]['load_time'] = time.time()
                self.tool_usage_count[device_id][tool_name] = \
                    self.tool_usage_count[device_id].get(tool_name, 0) + 1
                return True
            
            tool_info = self.tool_registry[tool_name]
            required_memory = tool_info['memory_size']
            
            # 检查内存
            if self.device_memory[device_id] + required_memory > self.device_memory_limit:
                # 需要卸载一些工具（改进的 LRU 策略）
                self._evict_tools(device_id, required_memory)
            
            # 加载工具
            try:
                module = importlib.import_module(tool_info['module_path'])
                handler = getattr(module, 'execute')
                
                self.loaded_tools[device_id][tool_name] = {
                    'handler': handler,
                    'memory_size': required_memory,
                    'load_time': time.time()
                }
                
                self.device_memory[device_id] += required_memory
                self.tool_usage_count[device_id][tool_name] = 1
                
                print(f"[ToolManager] ✓ Tool '{tool_name}' loaded on Device {device_id}")
                return True
                
            except Exception as e:
                print(f"[ToolManager] ✗ Failed to load tool '{tool_name}': {e}")
                return False
    
    def execute_tool(
        self,
        tool_name: str,
        device_id: int,
        arguments: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """执行工具（线程安全，支持超时）"""
        # 确保工具已加载
        if tool_name not in self.loaded_tools[device_id]:
            success = self.load_tool(tool_name, device_id)
            if not success:
                return {
                    'success': False,
                    'error': f"Failed to load tool '{tool_name}'",
                    'tool_name': tool_name,
                    'device_id': device_id
                }
        
        with self._device_locks[device_id]:
            handler = self.loaded_tools[device_id][tool_name]['handler']
            # 更新访问时间和使用计数
            self.loaded_tools[device_id][tool_name]['load_time'] = time.time()
            self.tool_usage_count[device_id][tool_name] = \
                self.tool_usage_count[device_id].get(tool_name, 0) + 1
        
        # 使用线程执行工具（支持超时）
        result_container = {}
        
        def _execute():
            try:
                result_container['result'] = handler(**arguments)
                result_container['success'] = True
            except Exception as e:
                result_container['error'] = str(e)
                result_container['success'] = False
        
        thread = threading.Thread(target=_execute)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            return {
                'success': False,
                'error': f"Tool execution timeout after {timeout}s",
                'tool_name': tool_name,
                'device_id': device_id
            }
        
        if result_container.get('success'):
            return {
                'success': True,
                'result': result_container['result'],
                'tool_name': tool_name,
                'device_id': device_id
            }
        else:
            return {
                'success': False,
                'error': result_container.get('error', 'Unknown error'),
                'tool_name': tool_name,
                'device_id': device_id
            }
    
    def _evict_tools(self, device_id: int, required_memory: int):
        """驱逐工具以释放内存（改进的 LRU 策略，考虑使用频率）"""
        tools = list(self.loaded_tools[device_id].items())
        
        # 计算每个工具的优先级分数（越低越容易被驱逐）
        # 分数 = 使用次数 / (当前时间 - 最后访问时间 + 1)
        current_time = time.time()
        tool_scores = []
        for tool_name, tool_info in tools:
            usage_count = self.tool_usage_count[device_id].get(tool_name, 1)
            time_since_access = current_time - tool_info['load_time'] + 1
            score = usage_count / time_since_access
            tool_scores.append((tool_name, tool_info, score))
        
        # 按分数排序（分数低的优先驱逐）
        tool_scores.sort(key=lambda x: x[2])
        
        freed_memory = 0
        for tool_name, tool_info, score in tool_scores:
            if freed_memory >= required_memory:
                break
            
            del self.loaded_tools[device_id][tool_name]
            self.device_memory[device_id] -= tool_info['memory_size']
            freed_memory += tool_info['memory_size']
            
            # 清理使用计数
            if tool_name in self.tool_usage_count[device_id]:
                del self.tool_usage_count[device_id][tool_name]
            
            print(f"[ToolManager] Evicted '{tool_name}' from Device {device_id} (score={score:.3f})")
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具信息（线程安全）"""
        with self._lock:
            return self.tool_registry.get(tool_name)
    
    def list_tools(self) -> list:
        """列出所有已注册的工具（线程安全）"""
        with self._lock:
            return list(self.tool_registry.keys())
