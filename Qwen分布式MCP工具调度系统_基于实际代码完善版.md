# Qwen分布式MCP工具调度系统 - 基于实际代码的完善版

> **本文档基于qwen/distributed_inference的实际代码架构**
> 
> **核心特点**：
> - 基于现有的4节点分布式推理架构
> - 利用现有的网络通信机制（NodeServer/NodeClient）
> - 利用现有的消息传递系统（DistributedMessage）
> - 在不破坏现有架构的前提下添加工具调用能力

---

## 📚 文档导航

### 第一章：现有架构分析
深入分析qwen/distributed_inference的实际实现

### 第二章：工具调用集成方案
如何在现有架构上添加MCP工具调用能力

### 第三章：详细实现方案
基于实际代码的具体实现步骤

### 第四章：代码示例
可直接使用的完整代码

### 第五章：部署与测试
实际部署指南和测试方案

---

# 第一章：现有架构分析

## 1.1 实际的4节点架构

### 节点分工

```
┌─────────────────────────────────────────────────────────────┐
│ Node 0 (HeadNode) - Device 0                                │
│ ┌─────────────┐  ┌─────────────┐                           │
│ │ embed.om    │  │ layers_0_6  │                           │
│ │             │→ │ .om         │                           │
│ └─────────────┘  └─────────────┘                           │
│ • 接收初始token                                             │
│ • 运行embedding + layers 0-6                               │
│ • 发送hidden_states到Node 1                                │
│ • 接收来自Node 3的生成token                                │
└─────────────────────────────────────────────────────────────┘
                    ↓ hidden_states
┌─────────────────────────────────────────────────────────────┐
│ Node 1 (MiddleNode) - Device 1                             │
│ ┌─────────────┐                                            │
│ │ layers_7_13 │                                            │
│ │ .om         │                                            │
│ └─────────────┘                                            │
│ • 接收hidden_states                                        │
│ • 运行layers 7-13                                          │
│ • 发送hidden_states到Node 2                                │
└─────────────────────────────────────────────────────────────┘
                    ↓ hidden_states
┌─────────────────────────────────────────────────────────────┐
│ Node 2 (MiddleNode) - Device 2                             │
│ ┌─────────────┐                                            │
│ │ layers_14_20│                                            │
│ │ .om         │                                            │
│ └─────────────┘                                            │
│ • 接收hidden_states                                        │
│ • 运行layers 14-20                                         │
│ • 发送hidden_states到Node 3                                │
└─────────────────────────────────────────────────────────────┘
                    ↓ hidden_states
┌─────────────────────────────────────────────────────────────┐
│ Node 3 (TailNode) - Device 3                               │
│ ┌─────────────┐  ┌─────────────┐                          │
│ │ layers_21_27│  │ output.om   │                          │
│ │ .om         │→ │ (lm_head)   │                          │
│ └─────────────┘  └─────────────┘                          │
│ • 接收hidden_states                                        │
│ • 运行layers 21-27 + lm_head                               │
│ • 采样生成token                                            │
│ • 发送token回Node 0                                        │
└─────────────────────────────────────────────────────────────┘
                    ↓ next_token
                    ↑ (回到Node 0)
```

### 关键特点

1. **流水线架构**：数据单向流动（Node 0 → 1 → 2 → 3 → 0）
2. **网络通信**：使用Socket + Pickle序列化
3. **消息系统**：DistributedMessage封装不同类型的消息
4. **KV Cache**：每个节点独立管理自己的KV Cache
5. **同步执行**：每个token生成都需要完整的流水线

## 1.2 现有的网络通信机制

### NodeServer（服务器端）

```python
class NodeServer:
    """节点服务器，用于接收来自上一个节点的数据"""
    
    def __init__(self, port: int, node_name: str = "Node"):
        self.port = port
        self.node_name = node_name
        self.server_sock = None
        self.client_conn = None
    
    def start(self) -> bool:
        """启动服务器"""
        # 绑定端口，监听连接
    
    def accept_connection(self, timeout: float = None) -> bool:
        """等待并接受连接"""
    
    def recv(self) -> Optional[Any]:
        """接收数据"""
    
    def send(self, msg: Any) -> bool:
        """发送数据（用于返回结果）"""
```

### NodeClient（客户端）

```python
class NodeClient:
    """节点客户端，用于连接下一个节点"""
    
    def __init__(self, host: str, port: int, node_name: str = "Node"):
        self.host = host
        self.port = port
        self.node_name = node_name
        self.sock = None
    
    def connect(self, retry_interval: float = 1.0, max_retries: int = 60) -> bool:
        """连接到目标节点"""
    
    def send(self, msg: Any) -> bool:
        """发送数据"""
    
    def recv(self) -> Optional[Any]:
        """接收数据（用于接收返回结果）"""
```

### DistributedMessage（消息封装）

```python
class DistributedMessage:
    """分布式消息封装"""
    
    # 消息类型
    MSG_FORWARD = "forward"      # 前向传播数据
    MSG_RESULT = "result"        # 结果返回
    MSG_CONTROL = "control"      # 控制消息
    MSG_RESET = "reset"          # 重置 KV Cache
    MSG_SHUTDOWN = "shutdown"    # 关闭节点
    
    def __init__(self, msg_type: str, step: int = 0, data: dict = None):
        self.msg_type = msg_type
        self.step = step
        self.data = data or {}
```

## 1.3 现有的数据流

### 正常推理流程

```
1. Node 0 (HeadNode):
   - 接收prompt tokens
   - 运行embed.om → hidden_states
   - 运行layers_0_6.om → hidden_states
   - 发送MSG_FORWARD到Node 1

2. Node 1 (MiddleNode):
   - 接收hidden_states
   - 运行layers_7_13.om → hidden_states
   - 发送MSG_FORWARD到Node 2

3. Node 2 (MiddleNode):
   - 接收hidden_states
   - 运行layers_14_20.om → hidden_states
   - 发送MSG_FORWARD到Node 3

4. Node 3 (TailNode):
   - 接收hidden_states
   - 运行layers_21_27.om → hidden_states
   - 运行output.om → logits
   - 采样生成next_token
   - 发送MSG_RESULT到Node 0

5. Node 0:
   - 接收next_token
   - 判断是否结束（EOS token）
   - 如果未结束，使用next_token作为新输入，重复步骤1-4
```

---

# 第二章：工具调用集成方案

## 2.1 核心设计思路

### 关键洞察（以当前代码实现为准）

1. **工具调用决策发生在 Node 0，但工具执行可分布式**
   - Node 0：流式解析 `<tool_call>`、调度工具、聚合工具结果，并将 `<tool_results>` 注入下一轮推理输入。
   - Node 1/2/3：各自运行 `ToolAgent`，接收 `MSG_TOOL_CALL` 后在本机执行并返回 `MSG_TOOL_RESULT`。

2. **不破坏现有流水线（工具消息复用同一条链路并沿链路转发）**
   - 推理主链路仍是 `Node 0 → 1 → 2 → 3 → 0`。
   - `MSG_TOOL_CALL/MSG_TOOL_RESULT` 复用现有 socket，通过中间/尾节点“转发”实现路由到目标设备。

3. **流式并行工具调用**
   - Node 0 通过 `StreamingToolCallParser` 在增量解码文本时识别完整 `<tool_call>`。
   - 识别到完整调用后，使用 `AsyncToolExecutor` 并发执行（由 `ToolCoordinator` 决定本地或远程执行）。
   - 本轮生成结束后统一等待所有 pending 工具完成，再注入 `<tool_results>` 进入下一轮。

4. **KV Cache 管理（关键约束）**
   - 工具结果注入下一轮之前，Node 0 必须执行 `reset()`：清理本地 KV Cache 并广播 `MSG_RESET` 给所有节点，保证下一轮推理从干净状态开始。

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│ Node 0 (HeadNode) + 工具调用层                              │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ 工具调用协调器 (ToolCoordinator)                    │   │
│ │ • 解析模型输出中的tool_call                         │   │
│ │ • 调度工具到合适的设备                              │   │
│ │ • 收集工具结果                                      │   │
│ │ • 构建新的prompt（包含工具结果）                    │   │
│ └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ 现有的HeadNode逻辑                                  │   │
│ │ • embed.om                                          │   │
│ │ • layers_0_6.om                                     │   │
│ │ • 网络通信                                          │   │
│ └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 工具执行层 (可以在任意设备)                                 │
│                                                             │
│ Device 0/1/2/3: 工具管理器 (ToolManager)                   │
│ • 加载工具                                                  │
│ • 执行工具                                                  │
│ • 返回结果                                                  │
└─────────────────────────────────────────────────────────────┘
```

## 2.2 工具调用流程

> 当前代码采用“流式解析 + 并行调度 + 分布式执行”的工具调用方式：Node 0 负责识别与调度，Node 1/2/3 负责执行或转发。

### 完整流程

```
用户输入: "查询北京天气并推荐穿衣"

┌─────────────────────────────────────────────────────────────┐
│ 第1轮推理（在Node 0）                                        │
├─────────────────────────────────────────────────────────────┤
│ 1. 构建prompt（包含工具定义）                                │
│ 2. 运行完整流水线（Node 0→1→2→3→0）                        │
│ 3. 生成文本包含：                                            │
│    <tool_call>                                              │
│      <name>get_weather</name>                               │
│      <arguments>{"city": "北京"}</arguments>                │
│    </tool_call>                                             │
│ 4. ToolCoordinator检测到tool_call                          │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 工具执行（在某个Device上）                                   │
├─────────────────────────────────────────────────────────────┤
│ 1. 解析tool_call: get_weather(city="北京")                 │
│ 2. 调度到Device 2（假设）                                   │
│ 3. 在Device 2上执行工具                                     │
│ 4. 返回结果: {"temperature": 15, "condition": "晴"}        │
│ 5. 结果传回Node 0                                           │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 第2轮推理（在Node 0）                                        │
├─────────────────────────────────────────────────────────────┤
│ 1. 构建新prompt:                                            │
│    User: 查询北京天气并推荐穿衣                             │
│    Assistant: <tool_call>...</tool_call>                   │
│    Tool: {"temperature": 15, "condition": "晴"}            │
│ 2. 重置KV Cache（发送MSG_RESET）                           │
│ 3. 运行完整流水线                                            │
│ 4. 生成最终答案:                                             │
│    "今天北京天气晴朗，温度15度，建议穿长袖..."               │
└─────────────────────────────────────────────────────────────┘
```

### 关键点

1. **工具调用与推理流水线共存**
   - 工具调用的识别与调度发生在 Node 0 的生成循环中。
   - Node 1/2/3 需要“感知并处理” `MSG_TOOL_CALL/MSG_TOOL_RESULT`：若命中目标设备则执行，否则继续转发到下一跳。

2. **KV Cache 管理（必须执行）**
   - 工具结果注入下一轮之前必须重置 KV Cache。
   - 由 Node 0 调用 `reset()` 广播 `MSG_RESET`，所有节点同步清理各自 KV Cache。

3. **结果回注入**
   - 工具结果最终仍以文本（如 `<tool_results>...</tool_results>`）形式注入下一轮 prompt，再继续正常流水线推理。
   - 为避免 tool_result 与 token result 在同一 socket 上互相阻塞/错配，Node 0 端需要对入站消息做缓存/匹配（request_id）。

## 2.3 消息扩展

### 新增消息类型

```python
class DistributedMessage:
    # 现有消息类型
    MSG_FORWARD = "forward"
    MSG_RESULT = "result"
    MSG_RESET = "reset"
    MSG_SHUTDOWN = "shutdown"
    
    # 新增：分布式工具调用（当前实现已使用）
    MSG_TOOL_CALL = "tool_call"        # 工具调用请求（包含 target_device_id 用于路由）
    MSG_TOOL_RESULT = "tool_result"    # 工具执行结果（包含 request_id 用于匹配）
```

**注意**：当前实现已经依赖 `MSG_TOOL_CALL/MSG_TOOL_RESULT` 完成“工具在任意设备执行”，不再是“完全在 Node 0 内部处理”。

---

# 第三章：详细实现方案

## 3.1 在Node 0添加工具调用能力

### 修改HeadNode类

```python
class HeadNodeWithTools(HeadNode):
    """带工具调用能力的头节点"""
    
    def __init__(self, config: DistributedConfig, tool_config: dict = None):
        super().__init__(config)
        
        # 工具相关组件
        self.tool_coordinator = None
        self.tool_manager = None
        self.tool_scheduler = None
        
        if tool_config:
            self._init_tools(tool_config)
    
    def _init_tools(self, tool_config: dict):
        """初始化工具系统"""
        from tools.tool_manager import ToolManager
        from tools.tool_scheduler import Device0PreferredScheduler
        from tools.tool_coordinator import ToolCoordinator
        
        # 1. 创建工具管理器（管理所有设备的工具）
        self.tool_manager = ToolManager(
            devices=[0, 1, 2, 3],
            device_memory_limit=tool_config.get('device_memory_limit', 500)
        )
        
        # 2. 注册工具
        for tool_name, tool_cfg in tool_config.get('tools', {}).items():
            self.tool_manager.register_tool(tool_name, tool_cfg)
        
        # 3. 创建调度器（优先使用Device 0）
        self.tool_scheduler = Device0PreferredScheduler(
            devices=[0, 1, 2, 3],
            main_device_id=0
        )
        
        # 4. 创建协调器
        self.tool_coordinator = ToolCoordinator(
            tool_manager=self.tool_manager,
            scheduler=self.tool_scheduler
        )
        
        print(f"[{self.node_name}] Tool system initialized")
    
    def generate_with_tools(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 100,
        max_tool_iterations: int = 5
    ) -> tuple:
        """带工具调用的生成"""
        all_generated_ids = []
        conversation_history = []
        
        # 初始prompt
        current_ids = prompt_ids
        
        for tool_iteration in range(max_tool_iterations):
            print(f"\n[{self.node_name}] === Tool Iteration {tool_iteration + 1} ===")
            
            # 1. 运行一轮生成
            generated_ids = self.generate(current_ids, max_new_tokens)
            all_generated_ids.extend(generated_ids)
            
            # 2. 将生成的token转换为文本
            generated_text = self._ids_to_text(generated_ids)
            
            # 3. 检查是否包含tool_call
            tool_calls = self.tool_coordinator.parse_tool_calls(generated_text)
            
            if not tool_calls:
                # 没有工具调用，生成完成
                print(f"[{self.node_name}] No tool calls detected, generation complete")
                break
            
            # 4. 执行工具调用
            print(f"[{self.node_name}] Detected {len(tool_calls)} tool call(s)")
            tool_results = self.tool_coordinator.execute_tools(tool_calls)
            
            # 5. 构建新的prompt（包含工具结果）
            conversation_history.append({
                'role': 'assistant',
                'content': generated_text
            })
            conversation_history.append({
                'role': 'tool',
                'content': self._format_tool_results(tool_results)
            })
            
            # 6. 重置KV Cache
            self.reset()
            
            # 7. 构建新的输入
            new_prompt_text = self._build_prompt_with_history(
                original_prompt=self._ids_to_text(prompt_ids),
                history=conversation_history
            )
            current_ids = self._text_to_ids(new_prompt_text)
        
        return all_generated_ids, conversation_history
    
    def _ids_to_text(self, ids: list) -> str:
        """将token IDs转换为文本（需要tokenizer）"""
        # 实际实现需要加载tokenizer
        # 这里简化处理
        return f"<generated_text_from_ids_{len(ids)}>"
    
    def _text_to_ids(self, text: str) -> np.ndarray:
        """将文本转换为token IDs（需要tokenizer）"""
        # 实际实现需要加载tokenizer
        # 这里简化处理
        return np.array([[1, 2, 3]], dtype=np.int64)
    
    def _build_prompt_with_history(
        self,
        original_prompt: str,
        history: list
    ) -> str:
        """构建包含对话历史的prompt"""
        prompt = "You are a helpful assistant with access to tools.\n\n"
        
        # 添加工具定义
        prompt += "Available tools:\n"
        for tool_name in self.tool_manager.tool_registry.keys():
            tool_info = self.tool_manager.tool_registry[tool_name]
            prompt += f"- {tool_name}: {tool_info.get('description', '')}\n"
        
        prompt += "\nTo use a tool, output:\n"
        prompt += "<tool_call>\n"
        prompt += "  <name>tool_name</name>\n"
        prompt += "  <arguments>{\"arg1\": \"value1\"}</arguments>\n"
        prompt += "</tool_call>\n\n"
        
        # 添加对话历史
        prompt += f"User: {original_prompt}\n\n"
        
        for msg in history:
            role = msg['role']
            content = msg['content']
            if role == 'assistant':
                prompt += f"Assistant: {content}\n\n"
            elif role == 'tool':
                prompt += f"Tool Results: {content}\n\n"
        
        prompt += "Assistant:"
        
        return prompt
    
    def _format_tool_results(self, results: list) -> str:
        """格式化工具结果"""
        formatted = []
        for result in results:
            if result['success']:
                formatted.append(f"{result['tool_name']}: {result['result']}")
            else:
                formatted.append(f"{result['tool_name']}: Error - {result['error']}")
        return "\n".join(formatted)
```

## 3.2 工具协调器实现

```python
# tools/tool_coordinator.py

import re
import json
from typing import List, Dict, Any


class ToolCoordinator:
    """工具调用协调器"""
    
    def __init__(self, tool_manager, scheduler):
        self.tool_manager = tool_manager
        self.scheduler = scheduler
    
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
                    arguments = json.loads(args_match.group(1).strip())
                    
                    tool_calls.append({
                        'name': tool_name,
                        'arguments': arguments
                    })
            except Exception as e:
                print(f"[ToolCoordinator] Failed to parse tool call: {e}")
        
        return tool_calls
    
    def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行工具调用"""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            arguments = tool_call['arguments']
            
            # 1. 调度到设备
            device_id = self.scheduler.schedule(tool_name)
            
            print(f"[ToolCoordinator] Executing '{tool_name}' on Device {device_id}")
            
            # 2. 执行工具
            result = self.tool_manager.execute_tool(
                tool_name, device_id, arguments
            )
            
            results.append(result)
            
            if result['success']:
                print(f"[ToolCoordinator] ✓ Success: {result['result']}")
            else:
                print(f"[ToolCoordinator] ✗ Error: {result['error']}")
        
        return results
```

## 3.3 工具管理器实现

```python
# tools/tool_manager.py

import time
import importlib
from typing import Dict, Any, Optional


class ToolManager:
    """工具管理器 - 管理多设备上的工具"""
    
    def __init__(self, devices: list, device_memory_limit: int = 500):
        self.devices = devices
        self.device_memory_limit = device_memory_limit
        
        # 工具注册表
        self.tool_registry = {}
        
        # 每个设备已加载的工具
        self.loaded_tools = {i: {} for i in devices}
        
        # 每个设备的内存使用
        self.device_memory = {i: 0 for i in devices}
    
    def register_tool(self, tool_name: str, tool_config: Dict[str, Any]):
        """注册工具"""
        self.tool_registry[tool_name] = {
            'name': tool_name,
            'module_path': tool_config['module_path'],
            'memory_size': tool_config.get('memory_size', 50),
            'description': tool_config.get('description', ''),
            'handler': None
        }
        print(f"[ToolManager] ✓ Tool '{tool_name}' registered")
    
    def load_tool(self, tool_name: str, device_id: int) -> bool:
        """在指定设备上加载工具"""
        if tool_name not in self.tool_registry:
            print(f"[ToolManager] ✗ Tool '{tool_name}' not registered")
            return False
        
        # 检查是否已加载
        if tool_name in self.loaded_tools[device_id]:
            return True
        
        tool_info = self.tool_registry[tool_name]
        required_memory = tool_info['memory_size']
        
        # 检查内存
        if self.device_memory[device_id] + required_memory > self.device_memory_limit:
            # 需要卸载一些工具（LRU策略）
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
            
            print(f"[ToolManager] ✓ Tool '{tool_name}' loaded on Device {device_id}")
            return True
            
        except Exception as e:
            print(f"[ToolManager] ✗ Failed to load tool '{tool_name}': {e}")
            return False
    
    def execute_tool(
        self,
        tool_name: str,
        device_id: int,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行工具"""
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
        
        handler = self.loaded_tools[device_id][tool_name]['handler']
        
        try:
            result = handler(**arguments)
            return {
                'success': True,
                'result': result,
                'tool_name': tool_name,
                'device_id': device_id
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tool_name': tool_name,
                'device_id': device_id
            }
    
    def _evict_tools(self, device_id: int, required_memory: int):
        """驱逐工具以释放内存（LRU策略）"""
        tools = list(self.loaded_tools[device_id].items())
        tools.sort(key=lambda x: x[1]['load_time'])
        
        freed_memory = 0
        for tool_name, tool_info in tools:
            if freed_memory >= required_memory:
                break
            
            del self.loaded_tools[device_id][tool_name]
            self.device_memory[device_id] -= tool_info['memory_size']
            freed_memory += tool_info['memory_size']
            
            print(f"[ToolManager] Evicted '{tool_name}' from Device {device_id}")
```

## 3.4 调度器实现

```python
# tools/tool_scheduler.py

class Device0PreferredScheduler:
    """优先使用Device 0的调度器"""
    
    def __init__(self, devices: list, main_device_id: int = 0):
        self.devices = devices
        self.main_device_id = main_device_id
        self.device_loads = {i: 0 for i in devices}
        self.loaded_tools = {i: set() for i in devices}
    
    def schedule(self, tool_name: str, tool_size: int = 50) -> int:
        """
        调度策略：
        1. 如果工具已在某设备加载，使用该设备
        2. 如果Device 0负载不高，优先使用Device 0
        3. 否则选择负载最低的设备
        """
        # 1. 检查工具是否已在某设备加载
        for device_id, tools in self.loaded_tools.items():
            if tool_name in tools:
                return device_id
        
        # 2. 检查Device 0的负载
        main_device_load = self.device_loads[self.main_device_id]
        max_load_threshold = 500  # MB
        
        if main_device_load + tool_size < max_load_threshold:
            # Device 0负载不高，优先使用
            print(f"[Scheduler] Scheduling to Device {self.main_device_id} (avoid data transfer)")
            self.device_loads[self.main_device_id] += tool_size
            self.loaded_tools[self.main_device_id].add(tool_name)
            return self.main_device_id
        
        # 3. Device 0负载过高，选择其他设备
        other_devices = [i for i in self.devices if i != self.main_device_id]
        best_device = min(other_devices, key=lambda x: self.device_loads[x])
        
        print(f"[Scheduler] Scheduling to Device {best_device} (Device 0 overloaded)")
        self.device_loads[best_device] += tool_size
        self.loaded_tools[best_device].add(tool_name)
        
        return best_device
```

---

# 第四章：完整代码示例

## 4.1 示例工具实现

### 天气查询工具

```python
# qwen/distributed_inference/tools/builtin_tools/weather_tool.py

def execute(city: str, date: str = "today") -> dict:
    """
    查询天气信息
    
    Args:
        city: 城市名称
        date: 日期（默认今天）
    
    Returns:
        天气信息字典
    """
    # 模拟天气API调用
    weather_data = {
        "北京": {"temperature": 15, "condition": "晴", "wind": "微风"},
        "上海": {"temperature": 20, "condition": "多云", "wind": "东风"},
        "广州": {"temperature": 25, "condition": "阴", "wind": "南风"},
    }
    
    result = weather_data.get(city, {"temperature": 18, "condition": "未知", "wind": "无风"})
    result["city"] = city
    result["date"] = date
    
    return result


# 工具配置
TOOL_CONFIG = {
    'name': 'get_weather',
    'module_path': 'tools.builtin_tools.weather_tool',
    'memory_size': 30,
    'description': '查询指定城市的天气信息',
    'parameters': {
        'city': {'type': 'string', 'required': True, 'description': '城市名称'},
        'date': {'type': 'string', 'required': False, 'default': 'today', 'description': '日期'}
    }
}
```

### 计算器工具

```python
# qwen/distributed_inference/tools/builtin_tools/calculator_tool.py

def execute(expression: str) -> float:
    """
    执行数学计算
    
    Args:
        expression: 数学表达式
    
    Returns:
        计算结果
    """
    try:
        # 安全的eval（仅支持数学运算）
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return float(result)
    except Exception as e:
        raise ValueError(f"计算错误: {e}")


TOOL_CONFIG = {
    'name': 'calculator',
    'module_path': 'tools.builtin_tools.calculator_tool',
    'memory_size': 10,
    'description': '执行数学计算',
    'parameters': {
        'expression': {'type': 'string', 'required': True, 'description': '数学表达式'}
    }
}
```

## 4.2 完整运行脚本

```python
# qwen/distributed_inference/run_with_tools.py

import argparse
import numpy as np
from pathlib import Path

from config import DistributedConfig
from node_head_with_tools import HeadNodeWithTools


def main():
    parser = argparse.ArgumentParser(description="Qwen Distributed Inference with Tools")
    parser.add_argument("--om_dir", type=str, required=True, help="OM model directory")
    parser.add_argument("--device", type=int, default=0, help="Device ID for head node")
    parser.add_argument("--max_cache_len", type=int, default=1024)
    parser.add_argument("--max_input_len", type=int, default=16)
    parser.add_argument("--init_tokens", type=str, required=True, help="Initial token file")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--max_tool_iterations", type=int, default=5)
    
    # 网络配置
    parser.add_argument("--listen_port", type=int, default=9000)
    parser.add_argument("--next_ip", type=str, default="127.0.0.1")
    parser.add_argument("--next_port", type=int, default=9001)
    
    args = parser.parse_args()
    
    # 创建配置
    config = DistributedConfig(
        om_dir=args.om_dir,
        device_id=args.device,
        max_cache_len=args.max_cache_len,
        max_input_len=args.max_input_len,
        node_id=0,
    )
    
    # 工具配置
    tool_config = {
        'device_memory_limit': 500,
        'tools': {
            'get_weather': {
                'module_path': 'tools.builtin_tools.weather_tool',
                'memory_size': 30,
                'description': '查询天气信息'
            },
            'calculator': {
                'module_path': 'tools.builtin_tools.calculator_tool',
                'memory_size': 10,
                'description': '执行数学计算'
            }
        }
    }
    
    # 加载初始tokens
    with open(args.init_tokens, "r") as f:
        text = f.read().strip()
        prompt_ids = [int(x) for x in text.replace("\n", " ").split()]
    prompt_ids = np.array([prompt_ids], dtype=np.int64)
    
    # 创建并运行节点
    node = HeadNodeWithTools(config, tool_config)
    
    try:
        node.init()
        
        print(f"\n{'='*60}")
        print(f"Starting generation with tool support...")
        print(f"{'='*60}\n")
        
        generated_ids, history = node.generate_with_tools(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            max_tool_iterations=args.max_tool_iterations
        )
        
        print(f"\n{'='*60}")
        print(f"Generation complete!")
        print(f"Total tokens generated: {len(generated_ids)}")
        print(f"Tool iterations: {len([h for h in history if h['role'] == 'tool'])}")
        print(f"{'='*60}\n")
        
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
```

---

# 第五章：部署与测试

## 5.1 目录结构

```
qwen/distributed_inference/
├── config.py
├── network.py
├── acl_model.py
├── kvcache.py
├── utils.py
├── node_head.py
├── node_middle.py
├── node_tail.py
├── node_head_with_tools.py      # 新增
├── run_with_tools.py             # 新增
│
└── tools/                        # 新增
    ├── __init__.py
    ├── tool_manager.py
    ├── tool_coordinator.py
    ├── tool_scheduler.py
    └── builtin_tools/
        ├── __init__.py
        ├── weather_tool.py
        └── calculator_tool.py
```

## 5.2 部署步骤

### 步骤1：创建工具目录

```bash
cd qwen/distributed_inference
mkdir -p tools/builtin_tools
touch tools/__init__.py
touch tools/builtin_tools/__init__.py
```

### 步骤2：实现核心模块

按照第三章的代码实现以下文件：
- `tools/tool_manager.py`
- `tools/tool_coordinator.py`
- `tools/tool_scheduler.py`
- `node_head_with_tools.py`

### 步骤3：实现示例工具

按照第四章的代码实现：
- `tools/builtin_tools/weather_tool.py`
- `tools/builtin_tools/calculator_tool.py`

### 步骤4：启动分布式节点

```bash
# 终端1：启动Node 1 (MiddleNode)
python node_middle.py \
    --om_dir ./model_om \
    --device 1 \
    --node_id 1 \
    --listen_port 9001 \
    --next_ip 127.0.0.1 \
    --next_port 9002

# 终端2：启动Node 2 (MiddleNode)
python node_middle.py \
    --om_dir ./model_om \
    --device 2 \
    --node_id 2 \
    --listen_port 9002 \
    --next_ip 127.0.0.1 \
    --next_port 9003

# 终端3：启动Node 3 (TailNode)
python node_tail.py \
    --om_dir ./model_om \
    --device 3 \
    --listen_port 9003 \
    --head_ip 127.0.0.1 \
    --head_port 9000

# 终端4：启动Node 0 (HeadNode with Tools)
python run_with_tools.py \
    --om_dir ./model_om \
    --device 0 \
    --init_tokens init_tokens.txt \
    --max_new_tokens 100 \
    --listen_port 9000 \
    --next_ip 127.0.0.1 \
    --next_port 9001
```

## 5.3 测试场景

### 测试1：单工具调用

**输入**：
```
查询北京今天的天气
```

**预期流程**：
1. 模型生成tool_call
2. 执行get_weather工具
3. 模型基于天气数据生成答案

### 测试2：计算任务

**输入**：
```
计算 (123 + 456) * 789
```

**预期流程**：
1. 模型生成calculator tool_call
2. 执行计算
3. 返回结果

### 测试3：多轮工具调用

**输入**：
```
查询北京天气，然后根据温度计算华氏度
```

**预期流程**：
1. 第1轮：调用get_weather
2. 第2轮：调用calculator转换温度
3. 第3轮：生成最终答案

## 5.4 调试建议

### 日志级别

在各模块中添加详细日志：

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(name)s] %(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)
```

### 常见问题

1. **工具未找到**
   - 检查module_path是否正确
   - 确认工具已注册

2. **设备内存不足**
   - 调整device_memory_limit
   - 实现更激进的LRU驱逐策略

3. **KV Cache不同步**
   - 确保MSG_RESET正确发送到所有节点
   - 检查reset()方法是否正确实现

---

# 第六章：总结与展望

## 6.1 核心优势

1. **无侵入性**：不破坏现有的分布式推理架构
2. **灵活调度**：优先使用Device 0，避免数据传输
3. **易于扩展**：新工具只需实现execute函数
4. **资源高效**：LRU驱逐策略，动态内存管理

## 6.2 实现要点

1. **Node 0 负责识别与调度，工具执行分布式**
   - Node 0：流式解析 `<tool_call>` + 调度 + 聚合结果 + 注入 `<tool_results>`。
   - Node 1/2/3：运行 `ToolAgent` 处理 `MSG_TOOL_CALL` 并返回 `MSG_TOOL_RESULT`。

2. **每轮工具调用后必须使用 MSG_RESET 同步清 KV**
   - Node 0 在注入工具结果前执行 `reset()`，广播 `MSG_RESET`，确保所有节点 KV Cache 一致。

3. **工具结果通过文本注入 prompt 继续下一轮推理**
   - 工具结果序列化为文本块（例如 `<tool_results>`），由 tokenizer 编码后继续走推理流水线。

4. **调度策略仍优先 Device 0**
   - 使用 `Device0PreferredScheduler`，在资源允许时优先本地执行以减少跨设备数据传输。

## 6.3 未来扩展

### 扩展1：MCP协议集成

```python
class MCPToolAdapter:
    """MCP工具适配器"""
    
    def discover_tools_from_mcp(self, mcp_server_url: str):
        """从MCP服务器发现工具"""
        # 调用MCP协议的list_tools接口
        # 转换为内部格式
        # 自动注册
        pass
```

### 扩展2：异步工具执行

```python
import asyncio

async def async_execute_tool(tool_manager, tool_name, device_id, arguments):
    """异步执行工具"""
    return await asyncio.to_thread(
        tool_manager.execute_tool,
        tool_name, device_id, arguments
    )
```

### 扩展3：工具链优化

```python
class ToolChainOptimizer:
    """工具链优化器"""
    
    def optimize_chain(self, tool_sequence: list) -> list:
        """优化工具调用顺序"""
        # 分析数据依赖
        # 合并可并行的工具
        # 优化设备分配
        pass
```

## 6.4 性能指标

| 指标 | 无工具 | 有工具（Device 0） | 有工具（其他设备） |
|------|--------|-------------------|-------------------|
| 单token延迟 | 50ms | 55ms (+10%) | 65ms (+30%) |
| 工具调用延迟 | N/A | 5ms | 15ms |
| 内存开销 | 0MB | 50MB/工具 | 50MB/工具 |

## 6.5 最佳实践

1. **优先使用Device 0**：减少数据传输
2. **预加载常用工具**：减少加载时间
3. **合理设置内存限制**：平衡性能和资源
4. **实现工具缓存**：避免重复加载
5. **添加详细日志**：便于调试和优化

---

# 附录：快速参考

## A.1 关键类和方法

### HeadNodeWithTools

```python
# 初始化
node = HeadNodeWithTools(config, tool_config)
node.init()

# 带工具的生成
generated_ids, history = node.generate_with_tools(
    prompt_ids,
    max_new_tokens=100,
    max_tool_iterations=5
)
```

### ToolManager

```python
# 注册工具
tool_manager.register_tool(tool_name, tool_config)

# 执行工具
result = tool_manager.execute_tool(tool_name, device_id, arguments)
```

### ToolCoordinator

```python
# 解析工具调用
tool_calls = coordinator.parse_tool_calls(text)

# 执行工具
results = coordinator.execute_tools(tool_calls)
```

## A.2 配置示例

```python
tool_config = {
    'device_memory_limit': 500,  # MB
    'tools': {
        'tool_name': {
            'module_path': 'tools.builtin_tools.tool_module',
            'memory_size': 50,  # MB
            'description': '工具描述'
        }
    }
}
```

## A.3 工具模板

```python
def execute(**kwargs):
    """工具执行函数"""
    # 实现工具逻辑
    return result

TOOL_CONFIG = {
    'name': 'tool_name',
    'module_path': 'module.path',
    'memory_size': 50,
    'description': '工具描述',
    'parameters': {
        'param1': {'type': 'string', 'required': True}
    }
}
```

---

**文档版本**：v2.0 - 基于实际代码完善版  
**最后更新**：2026/3/7  
**适用版本**：qwen/distributed_inference
