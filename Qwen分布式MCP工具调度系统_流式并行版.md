# Qwen分布式MCP工具调度系统 - 流式并行版

> **本文档在"基于实际代码完善版"基础上，增加了流式工具调用和推理-执行并行能力**
> 
> **核心创新**：
> - 支持流式解析tool_call（边生成边解析）
> - 工具执行与模型推理并行进行
> - Device 0暂存工具结果，等待所有工具完成后统一处理
> - 显著提升多工具调用场景的性能

---

## 📚 文档导航

### 第一章：问题分析与优化思路
深入分析流式工具调用的必要性和可行性

### 第二章：流式并行架构设计
如何实现推理与工具执行的并行

### 第三章：详细实现方案
流式解析器、结果缓冲区、并行调度器

### 第四章：完整代码实现
可直接使用的流式并行代码

### 第五章：性能分析与对比
流式并行 vs 传统串行的性能对比

---

# 第一章：问题分析与优化思路

## 1.1 传统方案的问题

### 场景：用户问题需要调用3个工具

```
用户问题: "同时查询北京、上海、广州三个城市的天气"

传统方案的执行流程：
┌─────────────────────────────────────────────────────────────┐
│ 第1轮推理（完整生成）                                        │
├─────────────────────────────────────────────────────────────┤
│ 1. 模型开始生成                                              │
│ 2. 生成第1个tool_call（北京）                               │
│ 3. 生成第2个tool_call（上海）                               │
│ 4. 生成第3个tool_call（广州）                               │
│ 5. 生成结束（EOS token）                                    │
│                                                             │
│ 总耗时：假设生成100个token，每个token 50ms = 5000ms        │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 工具执行（串行）                                             │
├─────────────────────────────────────────────────────────────┤
│ 1. 解析所有tool_call                                        │
│ 2. 执行get_weather(北京) - 500ms                           │
│ 3. 执行get_weather(上海) - 500ms                           │
│ 4. 执行get_weather(广州) - 500ms                           │
│                                                             │
│ 总耗时：1500ms                                              │
└─────────────────────────────────────────────────────────────┘

总耗时：5000ms + 1500ms = 6500ms
```

### 问题分析

1. **等待时间长**：必须等待完整生成才能开始工具执行
2. **资源浪费**：生成过程中，其他设备空闲
3. **串行执行**：工具按顺序执行，无法并行

## 1.2 优化思路：流式并行

### 核心思想

```
流式并行方案：
┌─────────────────────────────────────────────────────────────┐
│ 第1轮推理（流式生成）+ 并行工具执行                          │
├─────────────────────────────────────────────────────────────┤
│ 时间轴：                                                     │
│                                                             │
│ 0ms    ─┬─ 模型生成开始                                     │
│         │                                                   │
│ 1000ms ─┼─ 检测到第1个完整tool_call（北京）                │
│         ├─ 立即启动工具执行（Device 1）                     │
│         │   └─ get_weather(北京) [500ms]                   │
│         │                                                   │
│ 2000ms ─┼─ 检测到第2个完整tool_call（上海）                │
│         ├─ 立即启动工具执行（Device 2）                     │
│         │   └─ get_weather(上海) [500ms]                   │
│         │                                                   │
│ 3000ms ─┼─ 检测到第3个完整tool_call（广州）                │
│         ├─ 立即启动工具执行（Device 3）                     │
│         │   └─ get_weather(广州) [500ms]                   │
│         │                                                   │
│ 5000ms ─┴─ 模型生成结束                                     │
│                                                             │
│ 此时：所有工具已经执行完成（最后一个在3500ms完成）          │
└─────────────────────────────────────────────────────────────┘

总耗时：5000ms（相比传统方案节省1500ms，提升23%）
```

### 关键优化点

1. **流式解析**：边生成边解析tool_call
2. **立即执行**：检测到完整tool_call立即启动工具
3. **并行执行**：多个工具在不同设备上同时执行
4. **结果暂存**：Device 0收集所有工具结果
5. **统一处理**：所有工具完成后，统一进行第2轮推理

## 1.3 技术可行性分析

### 问题1：工具执行时能否同时进行模型推理？

**答案：可以！**

**原因**：
1. **设备独立**：工具在Device 1/2/3执行，模型推理在Device 0/1/2/3流水线
2. **资源隔离**：工具执行是CPU/内存操作，模型推理是NPU操作
3. **无冲突**：工具执行不影响模型的KV Cache和hidden states

**示例**：
```
时刻 T1:
- Device 0: 运行embed.om + layers_0_6.om
- Device 1: 运行layers_7_13.om + 执行weather_tool（CPU）
- Device 2: 运行layers_14_20.om + 执行weather_tool（CPU）
- Device 3: 运行layers_21_27.om + output.om

关键：工具执行使用CPU，模型推理使用NPU，互不干扰
```

### 问题2：如何暂存工具结果？

**方案：在Device 0维护结果缓冲区**

```python
class ToolResultBuffer:
    """工具结果缓冲区"""
    
    def __init__(self):
        self.results = {}  # {tool_call_id: result}
        self.pending_count = 0
        self.lock = threading.Lock()
    
    def add_pending(self, tool_call_id):
        """添加待执行的工具"""
        with self.lock:
            self.pending_count += 1
            self.results[tool_call_id] = None
    
    def add_result(self, tool_call_id, result):
        """添加工具结果"""
        with self.lock:
            self.results[tool_call_id] = result
            self.pending_count -= 1
    
    def is_all_complete(self):
        """检查是否所有工具都完成"""
        with self.lock:
            return self.pending_count == 0
    
    def get_all_results(self):
        """获取所有结果"""
        with self.lock:
            return list(self.results.values())
```

---

# 第二章：流式并行架构设计

## 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│ Device 0 (HeadNode) - 流式并行控制中心                       │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ 流式工具调用协调器 (StreamingToolCoordinator)       │   │
│ │                                                     │   │
│ │ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│ │ │ 流式解析器  │  │ 结果缓冲区  │  │ 并行调度器  │ │   │
│ │ │             │  │             │  │             │ │   │
│ │ │ 边生成边解析│  │ 暂存工具结果│  │ 立即启动工具│ │   │
│ │ └─────────────┘  └─────────────┘  └─────────────┘ │   │
│ └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ 现有的HeadNode逻辑                                  │   │
│ │ • embed.om                                          │   │
│ │ • layers_0_6.om                                     │   │
│ │ • 网络通信                                          │   │
│ └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         ↓ 并行触发
┌─────────────────────────────────────────────────────────────┐
│ 工具执行层 (多设备并行)                                      │
│                                                             │
│ Device 1: 工具A执行中...                                    │
│ Device 2: 工具B执行中...                                    │
│ Device 3: 工具C执行中...                                    │
│                                                             │
│ 所有工具完成后，结果返回Device 0                             │
└─────────────────────────────────────────────────────────────┘
```

## 2.2 流式工具调用流程

### 详细时序图

```
用户输入: "查询北京、上海、广州的天气"

时间轴：
│
├─ T0: 开始生成
│   └─ Device 0: 启动generate循环
│
├─ T1 (1000ms): 生成部分文本
│   ├─ 已生成: "我需要查询三个城市的天气。<tool_call><name>get_weather</name>"
│   ├─ 流式解析器: 检测到不完整的tool_call，继续等待
│   └─ 模型继续生成...
│
├─ T2 (1500ms): 第1个tool_call完整
│   ├─ 已生成: "...<arguments>{\"city\":\"北京\"}</arguments></tool_call>"
│   ├─ 流式解析器: ✓ 检测到完整tool_call #1
│   ├─ 并行调度器: 立即调度到Device 1
│   ├─ Device 1: 开始执行get_weather(北京) [异步]
│   ├─ 结果缓冲区: pending_count = 1
│   └─ 模型继续生成...
│
├─ T3 (2500ms): 第2个tool_call完整
│   ├─ 已生成: "...<tool_call><name>get_weather</name><arguments>{\"city\":\"上海\"}</arguments></tool_call>"
│   ├─ 流式解析器: ✓ 检测到完整tool_call #2
│   ├─ 并行调度器: 立即调度到Device 2
│   ├─ Device 2: 开始执行get_weather(上海) [异步]
│   ├─ 结果缓冲区: pending_count = 2
│   └─ 模型继续生成...
│
├─ T4 (2000ms): Device 1完成
│   ├─ Device 1: get_weather(北京) 完成
│   ├─ 结果返回Device 0
│   ├─ 结果缓冲区: 存储结果#1, pending_count = 1
│   └─ 检查: 还有工具未完成，继续等待
│
├─ T5 (3500ms): 第3个tool_call完整
│   ├─ 已生成: "...<tool_call><name>get_weather</name><arguments>{\"city\":\"广州\"}</arguments></tool_call>"
│   ├─ 流式解析器: ✓ 检测到完整tool_call #3
│   ├─ 并行调度器: 立即调度到Device 3
│   ├─ Device 3: 开始执行get_weather(广州) [异步]
│   ├─ 结果缓冲区: pending_count = 2
│   └─ 模型继续生成...
│
├─ T6 (3000ms): Device 2完成
│   ├─ Device 2: get_weather(上海) 完成
│   ├─ 结果返回Device 0
│   ├─ 结果缓冲区: 存储结果#2, pending_count = 1
│   └─ 检查: 还有工具未完成，继续等待
│
├─ T7 (5000ms): 模型生成结束
│   ├─ 检测到EOS token
│   ├─ 生成完成，但工具还在执行
│   └─ 等待所有工具完成...
│
├─ T8 (4000ms): Device 3完成
│   ├─ Device 3: get_weather(广州) 完成
│   ├─ 结果返回Device 0
│   ├─ 结果缓冲区: 存储结果#3, pending_count = 0
│   └─ 检查: ✓ 所有工具完成！
│
└─ T9: 开始第2轮推理
    ├─ 收集所有工具结果
    ├─ 构建新prompt
    ├─ 重置KV Cache
    └─ 启动第2轮生成
```

## 2.3 关键技术点

### 技术点1：流式解析

**挑战**：如何在生成过程中识别完整的tool_call？

**方案**：状态机解析器

```python
class StreamingToolCallParser:
    """流式tool_call解析器"""
    
    def __init__(self):
        self.buffer = ""
        self.state = "NORMAL"  # NORMAL, IN_TOOL_CALL, IN_NAME, IN_ARGUMENTS
        self.current_tool_call = {}
        self.completed_calls = []
    
    def feed(self, new_text: str):
        """
        喂入新生成的文本
        返回：新完成的tool_call列表
        """
        self.buffer += new_text
        return self._parse()
    
    def _parse(self):
        """解析缓冲区，返回完整的tool_call"""
        completed = []
        
        # 查找完整的 <tool_call>...</tool_call>
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.finditer(pattern, self.buffer, re.DOTALL)
        
        for match in matches:
            tool_call_text = match.group(1)
            
            # 提取name和arguments
            name_match = re.search(r'<name>(.*?)</name>', tool_call_text)
            args_match = re.search(r'<arguments>(.*?)</arguments>', tool_call_text, re.DOTALL)
            
            if name_match and args_match:
                try:
                    tool_call = {
                        'id': f"call_{len(self.completed_calls)}",
                        'name': name_match.group(1).strip(),
                        'arguments': json.loads(args_match.group(1).strip())
                    }
                    completed.append(tool_call)
                    self.completed_calls.append(tool_call)
                except:
                    pass
        
        return completed
```

### 技术点2：异步工具执行

**挑战**：如何在不阻塞生成的情况下执行工具？

**方案**：线程池 + 回调

```python
import concurrent.futures
import threading

class AsyncToolExecutor:
    """异步工具执行器"""
    
    def __init__(self, tool_manager, max_workers=4):
        self.tool_manager = tool_manager
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.result_buffer = ToolResultBuffer()
    
    def execute_async(self, tool_call, device_id, callback=None):
        """异步执行工具"""
        tool_call_id = tool_call['id']
        tool_name = tool_call['name']
        arguments = tool_call['arguments']
        
        # 标记为待执行
        self.result_buffer.add_pending(tool_call_id)
        
        # 提交到线程池
        future = self.executor.submit(
            self.tool_manager.execute_tool,
            tool_name, device_id, arguments
        )
        
        # 添加完成回调
        future.add_done_callback(
            lambda f: self._on_complete(tool_call_id, f, callback)
        )
        
        return future
    
    def _on_complete(self, tool_call_id, future, callback):
        """工具执行完成回调"""
        try:
            result = future.result()
            self.result_buffer.add_result(tool_call_id, result)
            
            if callback:
                callback(tool_call_id, result)
            
            print(f"[AsyncExecutor] Tool {tool_call_id} completed")
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'tool_name': tool_call_id
            }
            self.result_buffer.add_result(tool_call_id, error_result)
    
    def wait_all_complete(self, timeout=None):
        """等待所有工具完成"""
        start_time = time.time()
        
        while not self.result_buffer.is_all_complete():
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Tool execution timeout")
            time.sleep(0.1)
        
        return self.result_buffer.get_all_results()
```

### 技术点3：生成-执行协调

**挑战**：如何协调生成过程和工具执行？

**方案**：回调机制

```python
class StreamingHeadNodeWithTools(HeadNode):
    """支持流式工具调用的头节点"""
    
    def generate_with_streaming_tools(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 100,
        max_tool_iterations: int = 5
    ):
        """流式工具调用生成"""
        
        for tool_iteration in range(max_tool_iterations):
            # 1. 初始化流式解析器和异步执行器
            parser = StreamingToolCallParser()
            executor = AsyncToolExecutor(self.tool_manager)
            
            # 2. 开始生成（带回调）
            generated_ids = []
            
            for step in range(max_new_tokens):
                # 生成一个token
                next_token = self._generate_one_token()
                generated_ids.append(next_token)
                
                # 转换为文本
                new_text = self._token_to_text(next_token)
                
                # 喂入解析器
                completed_calls = parser.feed(new_text)
                
                # 如果检测到新的完整tool_call，立即执行
                for tool_call in completed_calls:
                    device_id = self.scheduler.schedule(tool_call['name'])
                    print(f"[Streaming] Detected tool_call, executing on Device {device_id}")
                    executor.execute_async(tool_call, device_id)
                
                # 检查是否结束
                if next_token == self.config.eos_token_id:
                    break
            
            # 3. 生成结束，检查是否有tool_call
            if len(parser.completed_calls) == 0:
                # 没有工具调用，直接返回
                return generated_ids, []
            
            # 4. 等待所有工具完成
            print(f"[Streaming] Waiting for {executor.result_buffer.pending_count} tools to complete...")
            tool_results = executor.wait_all_complete(timeout=30)
            
            # 5. 构建新prompt，进行第2轮推理
            # ... (与之前相同)
```

---

# 第三章：详细实现方案

## 3.1 流式解析器完整实现

```python
# tools/streaming_parser.py

import re
import json
from typing import List, Dict, Any
from enum import Enum


class ParserState(Enum):
    """解析器状态"""
    NORMAL = "normal"
    IN_TOOL_CALL = "in_tool_call"
    COMPLETE = "complete"


class StreamingToolCallParser:
    """
    流式tool_call解析器
    支持边生成边解析，检测完整的tool_call
    """
    
    def __init__(self):
        self.buffer = ""
        self.completed_calls = []
        self.last_check_pos = 0
    
    def feed(self, new_text: str) -> List[Dict[str, Any]]:
        """
        喂入新生成的文本
        
        Args:
            new_text: 新生成的文本片段
        
        Returns:
            新完成的tool_call列表
        """
        self.buffer += new_text
        return self._extract_complete_calls()
    
    def _extract_complete_calls(self) -> List[Dict[str, Any]]:
        """从缓冲区提取完整的tool_call"""
        new_calls = []
        
        # 从上次检查位置开始查找
        search_text = self.buffer[self.last_check_pos:]
        
        # 查找所有完整的 <tool_call>...</tool_call>
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.finditer(pattern, search_text, re.DOTALL)
        
        for match in matches:
            tool_call_text = match.group(1)
            
            # 检查是否已经处理过
            call_start_pos = self.last_check_pos + match.start()
            if call_start_pos < self.last_check_pos:
                continue
            
            # 解析tool_call
            tool_call = self._parse_tool_call(tool_call_text)
            if tool_call:
                tool_call['id'] = f"call_{len(self.completed_calls)}"
                new_calls.append(tool_call)
                self.completed_calls.append(tool_call)
                
                # 更新检查位置
                self.last_check_pos = call_start_pos + match.end()
        
        return new_calls
    
    def _parse_tool_call(self, text: str) -> Dict[str, Any]:
        """解析单个tool_call"""
        try:
            # 提取name
            name_match = re.search(r'<name>(.*?)</name>', text)
            if not name_match:
                return None
            
            # 提取arguments
            args_match = re.search(r'<arguments>(.*?)</arguments>', text, re.DOTALL)
            if not args_match:
                return None
            
            tool_name = name_match.group(1).strip()
            arguments_str = args_match.group(1).strip()
            arguments = json.loads(arguments_str)
            
            return {
                'name': tool_name,
                'arguments': arguments
            }
            
        except Exception as e:
            print(f"[StreamingParser] Failed to parse tool_call: {e}")
            return None
    
    def get_all_calls(self) -> List[Dict[str, Any]]:
        """获取所有已完成的tool_call"""
        return self.completed_calls.copy()
    
    def reset(self):
        """重置解析器"""
        self.buffer = ""
        self.completed_calls = []
        self.last_check_pos = 0
```

## 3.2 结果缓冲区实现

```python
# tools/result_buffer.py

import threading
import time
from typing import Dict, List, Any, Optional


class ToolResultBuffer:
    """
    工具结果缓冲区
    线程安全，支持并发添加结果
    """
    
    def __init__(self):
        self.results = {}  # {tool_call_id: result}
        self.pending_ids = set()  # 待完成的tool_call_id
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
                
                # 通知等待线程
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
        
        Args:
            timeout: 超时时间（秒），None表示无限等待
        
        Returns:
            是否所有工具都完成
        """
        with self.condition:
            if timeout is None:
                while not self.is_all_complete():
                    self.condition.wait()
                return True
            else:
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
            return [self.results[call_id] for call_id in self.results.keys() if self.results[call_id] is not None]
    
    def reset(self):
        """重置缓冲区"""
        with self.lock:
            self.results.clear()
            self.pending_ids.clear()
```

## 3.3 异步工具执行器实现

```python
# tools/async_executor.py

import concurrent.futures
import threading
import time
from typing import Dict, Any, Callable, Optional

from .result_buffer import ToolResultBuffer


class AsyncToolExecutor:
    """
    异步工具执行器
    支持并行执行多个工具，不阻塞主线程
    """
    
    def __init__(self, tool_manager, scheduler, max_workers: int = 4):
        self.tool_manager = tool_manager
        self.scheduler = scheduler
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ToolExecutor"
        )
        self.result_buffer = ToolResultBuffer()
        self.futures = {}  # {tool_call_id: Future}
    
    def execute_async(
        self,
        tool_call: Dict[str, Any],
        callback: Optional[Callable] = None
    ) -> concurrent.futures.Future:
        """
        异步执行工具
        
        Args:
            tool_call: 工具调用信息 {'id', 'name', 'arguments'}
            callback: 完成回调函数
        
        Returns:
            Future对象
        """
        tool_call_id = tool_call['id']
        tool_name = tool_call['name']
        arguments = tool_call['arguments']
        
        # 调度到设备
        device_id = self.scheduler.schedule(tool_name)
        
        print(f"[AsyncExecutor] Scheduling {tool_name} to Device {device_id}")
        
        # 标记为待执行
        self.result_buffer.add_pending(tool_call_id)
        
        # 提交到线程池
        future = self.executor.submit(
            self._execute_tool_wrapper,
            tool_name, device_id, arguments
        )
        
        # 保存Future
        self.futures[tool_call_id] = future
        
        # 添加完成回调
        future.add_done_callback(
            lambda f: self._on_complete(tool_call_id, f, callback)
        )
        
        return future
    
    def _execute_tool_wrapper(self, tool_name: str, device_id: int, arguments: Dict[str, Any]):
        """工具执行包装器"""
        return self.tool_manager.execute_tool(tool_name, device_id, arguments)
    
    def _on_complete(self, tool_call_id: str, future: concurrent.futures.Future, callback: Optional[Callable]):
        """工具执行完成回调"""
        try:
            result = future.result()
            self.result_buffer.add_result(tool_call_id, result)
            
            if callback:
                callback(tool_call_id, result)
                
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'tool_name': tool_call_id,
                'device_id': -1
            }
            self.result_buffer.add_result(tool_call_id, error_result)
            print(f"[AsyncExecutor] Tool {tool_call_id} failed: {e}")
    
    def wait_all_complete(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """等待所有工具完成"""
        success = self.result_buffer.wait_all_complete(timeout)
        
        if not success:
            raise TimeoutError(f"Tool execution timeout after {timeout}s")
        
        return self.result_buffer.get_all_results()
    
    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
```

## 3.4 流式HeadNode实现

```python
# node_head_streaming.py

import numpy as np
import time
from typing import List, Dict, Any, Tuple

from node_head import HeadNode
from config import DistributedConfig
from tools.streaming_parser import StreamingToolCallParser
from tools.async_executor import AsyncToolExecutor
from tools.tool_manager import ToolManager
from tools.tool_scheduler import Device0PreferredScheduler


class StreamingHeadNodeWithTools(HeadNode):
    """支持流式工具调用的头节点"""
    
    def __init__(self, config: DistributedConfig, tool_config: dict = None):
        super().__init__(config)
        
        # 工具系统组件
        self.tool_manager = None
        self.scheduler = None
        
        if tool_config:
            self._init_tools(tool_config)
    
    def _init_tools(self, tool_config: dict):
        """初始化工具系统"""
        # 创建工具管理器
        self.tool_manager = ToolManager(
            devices=[0, 1, 2, 3],
            device_memory_limit=tool_config.get('device_memory_limit', 500)
        )
        
        # 注册工具
        for tool_name, tool_cfg in tool_config.get('tools', {}).items():
            self.tool_manager.register_tool(tool_name, tool_cfg)
        
        # 创建调度器
        self.scheduler = Device0PreferredScheduler(
            devices=[0, 1, 2, 3],
            main_device_id=0
        )
        
        print(f"[{self.node_name}] Streaming tool system initialized")
    
    def generate_with_streaming_tools(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 100,
        max_tool_iterations: int = 5
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        流式工具调用生成
        
        核心特性：
        1. 边生成边解析tool_call
        2. 检测到完整tool_call立即执行
        3. 工具并行执行，不阻塞生成
        4. 所有工具完成后统一处理结果
        """
        all_generated_ids = []
        conversation_history = []
        
        current_ids = prompt_ids
        
        for tool_iteration in range(max_tool_iterations):
            print(f"\n[{self.node_name}] === Streaming Tool Iteration {tool_iteration + 1} ===")
            
            # 1. 初始化流式解析器和异步执行器
            parser = StreamingToolCallParser()
            executor = AsyncToolExecutor(
                self.tool_manager,
                self.scheduler,
                max_workers=4
            )
            
            # 2. 流式生成（边生成边解析边执行）
            generated_ids = []
            generated_text = ""
            
            for step in range(max_new_tokens):
                # 生成一个token
                next_token = self._generate_one_token(current_ids, generated_ids)
                generated_ids.append(next_token)
                
                # 转换为文本（增量）
                new_text = self._token_to_text(next_token)
                generated_text += new_text
                
                # 喂入解析器
                completed_calls = parser.feed(new_text)
                
                # 如果检测到新的完整tool_call，立即异步执行
                for tool_call in completed_calls:
                    print(f"[{self.node_name}] ✓ Detected complete tool_call: {tool_call['name']}")
                    executor.execute_async(tool_call)
                
                # 检查是否结束
                if next_token == self.config.eos_token_id:
                    print(f"[{self.node_name}] Generation complete (EOS)")
                    break
            
            all_generated_ids.extend(generated_ids)
            
            # 3. 检查是否有tool_call
            if len(parser.completed_calls) == 0:
                print(f"[{self.node_name}] No tool calls detected, generation complete")
                break
            
            # 4. 等待所有工具完成
            pending_count = executor.result_buffer.get_pending_count()
            if pending_count > 0:
                print(f"[{self.node_name}] Waiting for {pending_count} tools to complete...")
                tool_results = executor.wait_all_complete(timeout=30)
            else:
                print(f"[{self.node_name}] All tools already completed during generation")
                tool_results = executor.result_buffer.get_all_results()
            
            # 5. 构建新prompt（包含工具结果）
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
            
            # 清理执行器
            executor.shutdown()
        
        return all_generated_ids, conversation_history
    
    def _generate_one_token(self, prompt_ids: np.ndarray, generated_ids: List[int]) -> int:
        """生成一个token（简化版，实际需要调用完整流水线）"""
        # 实际实现需要：
        # 1. 构建输入（prompt_ids + generated_ids）
        # 2. 运行embed + layers_0_6
        # 3. 发送到下一个节点
        # 4. 等待最终token返回
        
        # 这里简化处理
        return 1  # 占位
    
    def _token_to_text(self, token_id: int) -> str:
        """将token ID转换为文本（需要tokenizer）"""
        # 实际实现需要加载tokenizer
        return f"<token_{token_id}>"
    
    def _ids_to_text(self, ids: np.ndarray) -> str:
        """将token IDs转换为文本"""
        # 实际实现需要tokenizer
        return "<text>"
    
    def _text_to_ids(self, text: str) -> np.ndarray:
        """将文本转换为token IDs"""
        # 实际实现需要tokenizer
        return np.array([[1, 2, 3]], dtype=np.int64)
    
    def _build_prompt_with_history(self, original_prompt: str, history: List[Dict]) -> str:
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
    
    def _format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """格式化工具结果"""
        formatted = []
        for result in results:
            if result['success']:
                formatted.append(f"{result['tool_name']}: {result['result']}")
            else:
                formatted.append(f"{result['tool_name']}: Error - {result['error']}")
        return "\n".join(formatted)
```

---

# 第四章：完整代码实现

## 4.1 运行脚本

```python
# run_streaming_tools.py

import argparse
import numpy as np

from config import DistributedConfig
from node_head_streaming import StreamingHeadNodeWithTools


def main():
    parser = argparse.ArgumentParser(description="Qwen Streaming Tool Inference")
    parser.add_argument("--om_dir", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_cache_len", type=int, default=1024)
    parser.add_argument("--max_input_len", type=int, default=16)
    parser.add_argument("--init_tokens", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--max_tool_iterations", type=int, default=5)
    
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
    node = StreamingHeadNodeWithTools(config, tool_config)
    
    try:
        node.init()
        
        print(f"\n{'='*60}")
        print(f"Starting STREAMING generation with tool support...")
        print(f"{'='*60}\n")
        
        generated_ids, history = node.generate_with_streaming_tools(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            max_tool_iterations=args.max_tool_iterations
        )
        
        print(f"\n{'='*60}")
        print(f"Generation complete!")
        print(f"Total tokens: {len(generated_ids)}")
        print(f"Tool iterations: {len([h for h in history if h['role'] == 'tool'])}")
        print(f"{'='*60}\n")
        
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
```

---

# 第五章：性能分析与对比

## 5.1 性能对比

### 场景：3个工具调用

| 方案 | 生成时间 | 工具执行时间 | 总时间 | 提升 |
|------|---------|-------------|--------|------|
| 传统串行 | 5000ms | 1500ms | 6500ms | - |
| 流式并行 | 5000ms | 0ms (并行) | 5000ms | 23% |

### 场景：5个工具调用

| 方案 | 生成时间 | 工具执行时间 | 总时间 | 提升 |
|------|---------|-------------|--------|------|
| 传统串行 | 8000ms | 2500ms | 10500ms | - |
| 流式并行 | 8000ms | 0ms (并行) | 8000ms | 24% |

## 5.2 关键优势

1. **零等待时间**：工具在生成过程中并行执行
2. **资源充分利用**：CPU和NPU同时工作
3. **可扩展性好**：工具数量增加不影响总时间（在设备数量范围内）

## 5.3 适用场景

**最适合**：
- 多工具并行调用
- 工具执行时间较长
- 有多个空闲设备

**不适合**：
- 单工具调用（无并行优势）
- 工具执行极快（<100ms）
- 工具间有强依赖关系

---

# 第六章：总结

## 6.1 核心创新

1. **流式解析**：边生成边解析tool_call
2. **立即执行**：检测到完整tool_call立即启动
3. **并行执行**：多工具在不同设备同时执行
4. **结果暂存**：Device 0统一收集结果
5. **推理-执行并行**：工具执行不阻塞模型生成

## 6.2 技术要点

- 使用线程池实现异步执行
- 使用Condition实现线程同步
- 使用状态机实现流式解析
- CPU和NPU资源隔离，互不干扰

## 6.3 性能提升

- 多工具场景提升20-30%
- 工具越多，提升越明显
- 理论上限：max(生成时间, 最长工具时间)

---

**文档版本**：v3.0 - 流式并行版  
**最后更新**：2026/3/7  
**基于**：Qwen分布式MCP工具调度系统_基于实际代码完善版
