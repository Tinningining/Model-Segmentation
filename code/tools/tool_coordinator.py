"""
工具调用协调器
"""
import re
import json
import uuid
from typing import List, Dict, Any, Optional, Callable

from network import DistributedMessage


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
        """从文本中解析工具调用"""
        tool_calls = []
        
        # 匹配 <tool_call>...</tool_call>
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                # 提取name和arguments
                name_match = re.search(r'<name>(.*?)</name>', match)
                args_match = re.search(r'<arguments>(.*?)</arguments>', match, re.DOTALL)
                
                if name_match and args_match:
                    tool_name = name_match.group(1).strip()
                    arguments_str = args_match.group(1).strip()
                    
                    # 解析JSON参数
                    try:
                        arguments = json.loads(arguments_str)
                    except json.JSONDecodeError:
                        # 如果不是有效的JSON，尝试简单解析
                        print(f"[ToolCoordinator] Warning: Invalid JSON arguments for {tool_name}, using raw string")
                        arguments = {'raw': arguments_str}
                    
                    tool_calls.append({
                        'name': tool_name,
                        'arguments': arguments
                    })
            except Exception as e:
                print(f"[ToolCoordinator] Failed to parse tool call: {e}")
        
        return tool_calls
    
    def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行工具调用（支持本地和远程）"""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            arguments = tool_call['arguments']
            
            # 获取工具信息
            tool_info = self.tool_manager.get_tool_info(tool_name)
            if not tool_info:
                results.append({
                    'success': False,
                    'error': f"Tool '{tool_name}' not found",
                    'tool_name': tool_name,
                    'device_id': None
                })
                continue
            
            # 1. 调度到设备
            device_id = self.scheduler.schedule(
                tool_name,
                tool_size=tool_info['memory_size']
            )
            
            print(f"[ToolCoordinator] Executing '{tool_name}' on Device {device_id}")
            
            # 2. 判断是本地还是远程执行
            if device_id == self.local_device_id:
                # 本地执行
                result = self._execute_local(tool_name, device_id, arguments)
            else:
                # 远程执行
                result = self._execute_remote(tool_name, device_id, arguments)
            
            results.append(result)
            
            if result['success']:
                print(f"[ToolCoordinator] ✓ Success: {result['result']}")
            else:
                print(f"[ToolCoordinator] ✗ Error: {result['error']}")
        
        return results
    
    def _execute_local(
        self,
        tool_name: str,
        device_id: int,
        arguments: dict
    ) -> Dict[str, Any]:
        """在本地设备执行工具"""
        print(f"[ToolCoordinator] Local execution on Device {device_id}")
        return self.tool_manager.execute_tool(tool_name, device_id, arguments)
    
    def _execute_remote(
        self,
        tool_name: str,
        device_id: int,
        arguments: dict
    ) -> Dict[str, Any]:
        """在远程设备执行工具"""
        if self.remote_call_handler is None:
            return {
                'success': False,
                'error': f"Remote execution not supported (no handler configured)",
                'tool_name': tool_name,
                'device_id': device_id
            }
        
        print(f"[ToolCoordinator] Remote execution on Device {device_id}")
        
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        # 创建工具调用消息
        tool_call_msg = DistributedMessage.create_tool_call_msg(
            tool_name=tool_name,
            arguments=arguments,
            request_id=request_id,
            target_device_id=device_id
        )
        
        # 调用远程处理器
        try:
            result_msg = self.remote_call_handler(device_id, tool_call_msg)
            
            if result_msg and result_msg.msg_type == DistributedMessage.MSG_TOOL_RESULT:
                response_id = result_msg.data.get('request_id')
                if response_id != request_id:
                    return {
                        'success': False,
                        'error': f"Request ID mismatch (expected {request_id}, got {response_id})",
                        'tool_name': tool_name,
                        'device_id': device_id
                    }
                
                return {
                    'success': result_msg.data.get('success', False),
                    'result': result_msg.data.get('result'),
                    'error': result_msg.data.get('error'),
                    'tool_name': tool_name,
                    'device_id': device_id
                }
            
            return {
                'success': False,
                'error': 'Invalid response from remote device',
                'tool_name': tool_name,
                'device_id': device_id
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Remote execution failed: {e}",
                'tool_name': tool_name,
                'device_id': device_id
            }
    
    def format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """格式化工具结果为文本"""
        formatted = []
        for result in results:
            if result['success']:
                formatted.append(f"{result['tool_name']}: {result['result']}")
            else:
                formatted.append(f"{result['tool_name']}: Error - {result['error']}")
        return "\n".join(formatted)
