# Qwen 4 节点分布式推理框架（支持 MCP 工具调用）

基于华为昇腾 ACL 的 Qwen 模型分布式推理框架，支持 **4 节点**流水线并行推理和 **MCP 工具调用**。

## 目录

- [架构概述](#架构概述)
- [核心特性](#核心特性)
- [与 2 节点版本的对比](#与-2-节点版本的对比)
- [模型参数](#模型参数)
- [文件结构](#文件结构)
- [工具调用系统](#工具调用系统)
- [香橙派昇腾分布式部署指南](#香橙派昇腾分布式部署指南)
- [详细配置](#详细配置)
- [API 使用](#api-使用)
- [故障排除](#故障排除)

---

## 架构概述

### 4 节点流水线架构

本框架将 Qwen 模型（28 层 Transformer）切分到 **4 个设备**上进行流水线并行推理。相比 2 节点版本，4 节点版本的**模型常驻内存**，无需频繁加载/卸载，推理速度更快：

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    4 节点分布式推理数据流                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│   │    Node 0       │    │    Node 1       │    │    Node 2       │    │    Node 3       │         │
│   │   (头节点)       │    │  (中间节点1)     │    │  (中间节点2)     │    │   (尾节点)       │         │
│   │                 │    │                 │    │                 │    │                 │         │
│   │  embed.om       │    │                 │    │                 │    │                 │         │
│   │  layers_0_6.om  │───▶│ layers_7_13.om  │───▶│ layers_14_20.om │───▶│ layers_21_27.om │         │
│   │                 │    │                 │    │                 │    │  output.om      │         │
│   │                 │    │                 │    │                 │    │                 │         │
│   │  [主节点]        │    │                 │    │                 │    │                 │         │
│   │  7 层 KV Cache  │    │  7 层 KV Cache  │    │  7 层 KV Cache  │    │  7 层 KV Cache  │         │
│   │                 │    │                 │    │                 │    │                 │         │
│   │  [工具调度中心]  │    │  [工具执行节点]  │    │  [工具执行节点]  │    │  [工具执行节点]  │         │
│   │  • 解析tool_call│    │  • ToolAgent    │    │  • ToolAgent    │    │  • ToolAgent    │         │
│   │  • 调度工具      │    │  • 执行/转发     │    │  • 执行/转发     │    │  • 执行/转发     │         │
│   │  • 聚合结果      │    │                 │    │                 │    │                 │         │
│   └────────▲────────┘    └─────────────────┘    └─────────────────┘    └────────┬────────┘         │
│            │                                                                    │                  │
│            │    hidden_states ──────────────────────────────────────────────▶   │                  │
│            │◀───────────────────── next_token ◀─────────────────────────────────┘                  │
│            │                                                                                       │
│            │    MSG_TOOL_CALL ──────────────────────────────────────────────▶                      │
│            │◀──────────────────── MSG_TOOL_RESULT ◀─────────────────────────────                  │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘

数据传输说明：
  • hidden_states: Node 0 → Node 1 → Node 2 → Node 3 (形状: [1, 16, 2048])
  • next_token:    Node 3 → Node 0 (单个 token ID)
  • MSG_TOOL_CALL: Node 0 → Node 1/2/3 (工具调用请求，沿流水线转发到目标设备)
  • MSG_TOOL_RESULT: Node 1/2/3 → Node 0 (工具执行结果，沿流水线返回)
  • KV Cache:      保存在每个设备本地，不需要网络传输
```

### 模型切分详情

| 节点 | 模型文件 | 内容 | 层数 |
|------|---------|------|------|
| **Node 0** | embed.om | Embedding 层 | - |
| **Node 0** | layers_0_6.om | Transformer 层 0-6 | 7 层 |
| **Node 1** | layers_7_13.om | Transformer 层 7-13 | 7 层 |
| **Node 2** | layers_14_20.om | Transformer 层 14-20 | 7 层 |
| **Node 3** | layers_21_27.om | Transformer 层 21-27 | 7 层 |
| **Node 3** | output.om | LM Head (输出层) | - |

### 节点职责

| 节点 | 角色 | 加载的模型 | 主要职责 |
|------|------|-----------|----------|
| **Node 0** | 头节点 + 工具调度中心 | embed.om + layers_0_6.om | 接收输入 → embedding → block → 发送 hidden_states → 接收 token<br>**工具调度**：解析 tool_call、调度工具、聚合结果、注入下一轮推理 |
| **Node 1** | 中间节点1 + 工具执行节点 | layers_7_13.om | 接收 hidden_states → block → 发送 hidden_states<br>**工具执行**：接收 MSG_TOOL_CALL，执行或转发 |
| **Node 2** | 中间节点2 + 工具执行节点 | layers_14_20.om | 接收 hidden_states → block → 发送 hidden_states<br>**工具执行**：接收 MSG_TOOL_CALL，执行或转发 |
| **Node 3** | 尾节点 + 工具执行节点 | layers_21_27.om + output.om | 接收 hidden_states → block → lm_head → 采样 → 发送 token<br>**工具执行**：接收 MSG_TOOL_CALL，执行或转发 |

### 内存优化：模型常驻内存

4 节点版本的核心优势是**模型常驻内存**：

```
4 节点版本（模型常驻内存）：
  • 每个节点只加载 1-2 个模型
  • 模型在初始化时加载，推理期间保持常驻
  • 无需频繁加载/卸载，推理速度更快

2 节点版本（按需加载）：
  • 每个节点需要加载 3 个模型
  • 由于内存限制，每次推理都需要加载/卸载模型
  • 推理速度较慢
```

---

## 核心特性

### 1. 分布式流水线推理

- **4 节点流水线**：模型切分到 4 个设备，流水线并行执行
- **模型常驻内存**：无需频繁加载/卸载，推理速度快
- **KV Cache 管理**：每个节点独立管理 7 层 KV Cache
- **网络通信**：基于 TCP Socket + Pickle 序列化

### 2. MCP 工具调用系统

- **流式解析**：在生成过程中实时解析 `<tool_call>` 标签
- **异步执行**：检测到完整工具调用后立即异步执行，不阻塞生成
- **分布式调度**：工具可在任意设备上执行，实现负载均衡
- **智能调度**：优先使用 Device 0（避免数据传输），负载过高时分配到其他设备
- **结果聚合**：生成结束后等待所有工具完成，聚合结果并注入下一轮推理

### 3. 工具调用流程

```
┌─────────────────────────────────────────────────────────────┐
│ 第1轮推理：生成 tool_call                                    │
├─────────────────────────────────────────────────────────────┤
│ User: 查询北京天气并推荐穿衣                                 │
│                                                             │
│ Node 0 → 1 → 2 → 3 (流水线推理)                            │
│                                                             │
│ Assistant: 我需要查询天气信息                                │
│ <tool_call>                                                 │
│   <name>get_weather</name>                                  │
│   <arguments>{"city": "北京"}</arguments>                   │
│ </tool_call>                                                │
│                                                             │
│ [StreamingToolCallParser 检测到完整 tool_call]              │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 工具执行：分布式调度                                         │
├─────────────────────────────────────────────────────────────┤
│ 1. ToolCoordinator 调度工具到 Device 2                      │
│ 2. Node 0 发送 MSG_TOOL_CALL 到流水线                       │
│ 3. Node 1 转发 → Node 2 执行 → Node 3 转发                 │
│ 4. 结果通过 MSG_TOOL_RESULT 返回 Node 0                    │
│                                                             │
│ Result: {"temperature": 15, "condition": "晴"}             │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 第2轮推理：注入 tool_results                                 │
├─────────────────────────────────────────────────────────────┤
│ 1. Node 0 执行 reset() 清理 KV Cache                        │
│ 2. 广播 MSG_RESET 到所有节点                                │
│ 3. 构建新 prompt:                                           │
│    User: 查询北京天气并推荐穿衣                             │
│    Assistant: <tool_call>...</tool_call>                   │
│    <tool_results>                                           │
│      {"temperature": 15, "condition": "晴"}                │
│    </tool_results>                                          │
│                                                             │
│ 4. Node 0 → 1 → 2 → 3 (流水线推理)                         │
│                                                             │
│ Assistant: 今天北京天气晴朗，温度15度，建议穿长袖...         │
└─────────────────────────────────────────────────────────────┘
```

---

## 与 2 节点版本的对比

| 特性 | 2 节点版本 | 4 节点版本 |
|------|-----------|-----------|
| 设备数量 | 2 台 | 4 台 |
| 每节点层数 | 14 层 | 7 层 |
| 每节点模型数 | 3 个 | 1-2 个 |
| 每节点 KV Cache | 14 层 | 7 层 |
| 网络通信次数 | 2 次/step | 4 次/step |
| 模型加载方式 | 按需加载/卸载 | 常驻内存 |
| 内存需求/节点 | 较高 | 较低 |
| 推理速度 | 较慢 | 较快 |
| 工具调用支持 | ❌ | ✅ |
| 适用场景 | 设备有限 | 追求速度 + 工具调用 |

---

## 模型参数

### 模型配置（与 ATC 转换命令对应）

```bash
# embed.om 输入
atc --model="embed.onnx" --input_shape="input_ids:1,16"

# layers_X_Y.om 输入
atc --model="layers_X_Y.onnx" --input_shape="hidden_states:1,16,2048;attention_mask:1,1,16,1040;position_ids:1,16;past_key:7,1,8,1024,128;past_value:7,1,8,1024,128"

# output.om 输入
atc --model="output.onnx" --input_shape="hidden_states:1,16,2048"
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_size` | 2048 | 隐藏层维度 |
| `num_attention_heads` | 16 | 注意力头数 |
| `num_key_value_heads` | 8 | KV 头数 (GQA) |
| `head_dim` | 128 | 每个头的维度 |
| `num_hidden_layers` | 28 | Transformer 总层数 |
| `vocab_size` | 151936 | 词表大小 |
| `max_input_len` | 16 | 单次输入的最大 token 数 |
| `max_cache_len` | 1024 | KV Cache 最大缓存长度 |

### 采样参数（默认值）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | 1.0 | 采样温度 |
| `top_k` | 0 | Top-K 采样（0 表示禁用） |
| `top_p` | 1.0 | Top-P 采样 |
| `greedy` | True | 贪婪采样（直接取 argmax） |

---

## 文件结构

```
code/
├── config.py              # 4 节点配置类定义
├── network.py             # TCP 网络通信模块
├── kvcache.py             # KV Cache 管理
├── acl_model.py           # ACL 模型封装（常驻内存版本）
├── utils.py               # 工具函数
│
├── node_head.py           # 头节点实现 (Node 0) - 支持工具调用
├── node_middle.py         # 中间节点实现 (Node 1, 2) - 支持工具执行
├── node_tail.py           # 尾节点实现 (Node 3) - 支持工具执行
│
├── tools/                 # 工具调用系统
│   ├── __init__.py
│   ├── tool_manager.py           # 工具管理器（注册、加载、执行）
│   ├── tool_coordinator.py       # 工具协调器（调度、聚合）
│   ├── tool_scheduler.py         # 工具调度器（设备选择策略）
│   ├── tool_agent.py             # 工具代理（本地执行）
│   ├── streaming_parser.py       # 流式 tool_call 解析器
│   ├── async_executor.py         # 异步工具执行器
│   ├── result_buffer.py          # 工具结果缓冲区
│   └── builtin_tools/            # 内置工具
│       ├── __init__.py
│       ├── weather_tool.py       # 天气查询工具
│       └── calculator_tool.py    # 计算器工具
│
└── README.md              # 本文档
```

---

## 工具调用系统

### 架构概述

本框架集成了分布式工具调用系统，支持在推理过程中调用外部工具（如天气查询、计算器等）。工具可以在任意设备上执行，实现负载均衡。

```
┌─────────────────────────────────────────────────────────────┐
│ Node 0 (HeadNode) - 工具调度中心                            │
│ • StreamingToolCallParser: 流式解析 <tool_call>            │
│ • ToolCoordinator: 调度工具到合适设备                       │
│ • AsyncToolExecutor: 并行执行多个工具                       │
│ • 聚合工具结果并注入下一轮推理                              │
└─────────────────────────────────────────────────────────────┘
                         ↓ MSG_TOOL_CALL
┌─────────────────────────────────────────────────────────────┐
│ Node 1/2/3 - 工具执行节点                                   │
│ • ToolAgent: 本地工具执行代理                               │
│ • 接收 MSG_TOOL_CALL，执行工具，返回 MSG_TOOL_RESULT        │
│ • 支持工具转发（若非目标设备则继续转发）                    │
└─────────────────────────────────────────────────────────────┘
```

### 工具调用流程详解

1. **流式解析**：Node 0 在生成过程中实时解析 `<tool_call>` 标签
   - 使用 `StreamingToolCallParser` 增量解析生成的文本
   - 检测到完整的 `<tool_call>...</tool_call>` 后立即触发

2. **异步执行**：检测到完整工具调用后立即异步执行，不阻塞生成
   - 使用 `AsyncToolExecutor` 并发执行多个工具（max_workers=4）
   - 生成过程继续进行，工具在后台执行

3. **设备调度**：优先使用 Device 0（避免数据传输），负载过高时分配到其他设备
   - `Device0PreferredScheduler`：优先本地执行
   - Device 0 负载 >500MB 时自动分配到其他设备
   - 支持 LRU 缓存驱逐策略

4. **分布式执行**：工具调用请求沿流水线转发到目标设备
   - Node 0 发送 `MSG_TOOL_CALL`（包含 target_device_id）
   - 中间节点检查 target_device_id，匹配则执行，否则转发
   - 执行结果通过 `MSG_TOOL_RESULT` 返回 Node 0

5. **结果聚合**：生成结束后等待所有工具完成，聚合结果
   - `AsyncToolExecutor.wait_all_complete()` 等待所有 pending 工具
   - `ToolCoordinator.format_tool_results()` 格式化结果

6. **注入推理**：将工具结果格式化为 `<tool_results>` 注入下一轮推理
   - 构建新 prompt：原始输入 + assistant 输出 + tool_results
   - 使用 tokenizer 编码为 token IDs
   - 继续流水线推理

7. **KV Cache 重置**：每次工具调用后必须执行全链路 reset
   - Node 0 调用 `reset()` 清理本地 KV Cache
   - 广播 `MSG_RESET` 到所有节点
   - 确保下一轮推理从干净状态开始

### 内置工具

| 工具名 | 功能 | 参数 | 内存占用 |
|--------|------|------|----------|
| `get_weather` | 查询天气信息 | city: 城市名, date: 日期 | 30MB |
| `calculator` | 执行数学计算 | expression: 数学表达式 | 10MB |

### 扩展自定义工具

#### 步骤 1：创建工具模块

```python
# code/tools/builtin_tools/my_tool.py

def execute(param1: str, param2: int = 10) -> dict:
    """
    工具执行函数
    
    Args:
        param1: 参数1描述
        param2: 参数2描述（可选）
    
    Returns:
        执行结果字典
    """
    # 实现工具逻辑
    result = {"status": "success", "data": f"Processed {param1}"}
    return result


TOOL_CONFIG = {
    'name': 'my_tool',
    'module_path': 'tools.builtin_tools.my_tool',
    'memory_size': 50,  # MB
    'description': '工具功能描述',
    'parameters': {
        'param1': {
            'type': 'string',
            'required': True,
            'description': '参数1描述'
        },
        'param2': {
            'type': 'integer',
            'required': False,
            'default': 10,
            'description': '参数2描述'
        }
    }
}
```

#### 步骤 2：注册工具

在 `node_head.py` 和 `node_middle.py` 的 `_init_tools()` 方法中添加：

```python
from tools.builtin_tools import my_tool

# 注册工具
self.tool_manager.register_tool('my_tool', my_tool.TOOL_CONFIG)
```

#### 步骤 3：使用工具

在 prompt 中使用标准格式调用：

```
请使用 my_tool 处理数据

<tool_call>
  <name>my_tool</name>
  <arguments>{"param1": "test_data", "param2": 20}</arguments>
</tool_call>
```

### 工具调度策略

**Device0PreferredScheduler**（默认）：
- 优先使用 Device 0（头节点），避免跨设备数据传输
- Device 0 负载过高时（>500MB）自动分配到其他设备
- 支持 LRU 缓存驱逐策略

**自定义调度器**：

```python
from tools.tool_scheduler import ToolScheduler

class CustomScheduler(ToolScheduler):
    def schedule(self, tool_name: str, tool_size: int = 50) -> int:
        # 实现自定义调度逻辑
        # 返回目标 device_id (0-3)
        return 0
```

### 工具系统配置

```python
# 在 node_head.py 中配置
tool_config = {
    'device_memory_limit': 500,  # 每个设备的内存限制（MB）
    'max_workers': 4,            # 异步执行器的最大工作线程数
    'tool_timeout': 30.0,        # 工具执行超时时间（秒）
}
```

### 消息类型扩展

```python
class DistributedMessage:
    # 现有消息类型
    MSG_FORWARD = "forward"      # 前向传播数据
    MSG_RESULT = "result"        # 结果返回
    MSG_RESET = "reset"          # 重置 KV Cache
    MSG_SHUTDOWN = "shutdown"    # 关闭节点
    
    # 工具调用消息类型
    MSG_TOOL_CALL = "tool_call"        # 工具调用请求
    MSG_TOOL_RESULT = "tool_result"    # 工具执行结果
```

---

## 香橙派昇腾分布式部署指南

本节详细介绍如何在 4 台香橙派昇腾开发板上部署分布式推理框架（支持工具调用）。

### 硬件准备

| 设备 | 数量 | 角色 | 需要的模型文件 |
|------|------|------|---------------|
| 香橙派昇腾 | **4 台** | 推理节点 | 见下表 |
| 开发机（电脑） | 1 台 | 远程控制 | 无（通过 SSH 控制香橙派） |
| 交换机 | 1 台 | 网络连接 | - |

**各香橙派需要的模型文件：**

| 设备 | IP 地址 | 角色 | 需要的模型文件 |
|------|---------|------|---------------|
| 香橙派 1 | 192.168.137.100 | Node 0 (头节点 + 工具调度) | embed.om, layers_0_6.om |
| 香橙派 2 | 192.168.137.101 | Node 1 (中间节点1 + 工具执行) | layers_7_13.om |
| 香橙派 3 | 192.168.137.102 | Node 2 (中间节点2 + 工具执行) | layers_14_20.om |
| 香橙派 4 | 192.168.137.103 | Node 3 (尾节点 + 工具执行) | layers_21_27.om, output.om |

### 网络连接说明

本方案使用**交换机直连**方式，所有设备（开发机 + 香橙派）通过网线连接到同一台交换机，无需路由器。

**网络连接方式：**

```
                              ┌─────────────────────────────────────────┐
                              │                 交换机                   │
                              └─────────────────────────────────────────┘
                                                │
        ┌─────────────────┬─────────────────────┼─────────────────┬─────────────────┐
        │                 │                     │                 │                 │
        ▼                 ▼                     ▼                 ▼                 ▼
   ┌────────┐        ┌────────┐           ┌────────┐        ┌────────┐        ┌────────┐
   │ 开发机  │        │香橙派1 │           │香橙派2 │        │香橙派3 │        │香橙派4 │
   │ (PC)   │        │ Node 0 │           │ Node 1 │        │ Node 2 │        │ Node 3 │
   │ (有线) │        │ (有线) │           │ (有线) │        │ (有线) │        │ (有线) │
   │192.168.│        │192.168.│           │192.168.│        │192.168.│        │192.168.│
   │137.99  │        │137.100 │           │137.101 │        │137.102 │        │137.103 │
   └────────┘        └────────┘           └────────┘        └────────┘        └────────┘
```

- **开发机**：通过网线连接到交换机，需要配置静态 IP（192.168.137.99）
- **香橙派**：通过网线连接到交换机，配置静态 IP
- **关键要求**：所有设备必须在同一网段（192.168.137.x），能够互相 ping 通
- **无需网关**：由于没有路由器，不需要配置网关

### 网络拓扑与数据流

```
                              ┌─────────────────────────────────────────┐
                              │                 交换机                   │
                              └─────────────────────────────────────────┘
                                                │
        ┌─────────────────┬─────────────────────┼─────────────────┬─────────────────┐
        │                 │                     │                 │                 │
   ┌────┴────┐       ┌────┴────┐          ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
   │ 开发机   │       │ Node 0  │          │ Node 1  │       │ Node 2  │       │ Node 3  │
   │ (PC)    │       │ 头节点   │          │中间节点1│       │中间节点2│       │ 尾节点   │
   │192.168. │       │192.168. │          │192.168. │       │192.168. │       │192.168. │
   │137.99   │       │137.100  │          │137.101  │       │137.102  │       │137.103  │
   │         │       │端口:9000│          │端口:9001│       │端口:9002│       │端口:9003│
   └─────────┘       └────┬────┘          └────┬────┘       └────┬────┘       └────┬────┘
                          │                    │                 │                 │
                          │    hidden_states   │  hidden_states  │  hidden_states  │
                          └───────────────────▶└────────────────▶└────────────────▶│
                          │                                                        │
                          │◀───────────────────── next_token ◀─────────────────────┘
                          │                                                        │
                          │    MSG_TOOL_CALL ──────────────────────────────────▶   │
                          │◀──────────────────── MSG_TOOL_RESULT ◀─────────────────┘
```

### 步骤 1：配置网络

（详细步骤请参考原文档的网络配置部分）

### 步骤 2：准备环境

在**每台香橙派**上执行以下操作：

```bash
# 安装 Python 依赖
pip3 install numpy

# 确认 ACL 环境已配置
python3 -c "import acl; print('ACL OK')"
```

### 步骤 3：分发代码和模型文件

```bash
# 在开发机上执行
# 复制代码到所有节点
for ip in 192.168.137.100 192.168.137.101 192.168.137.102 192.168.137.103; do
    scp -r code/ orangepi@$ip:~/qwen_distributed/
done

# 分发模型文件（按节点需求）
scp embed.om layers_0_6.om orangepi@192.168.137.100:~/qwen_distributed/models/
scp layers_7_13.om orangepi@192.168.137.101:~/qwen_distributed/models/
scp layers_14_20.om orangepi@192.168.137.102:~/qwen_distributed/models/
scp layers_21_27.om output.om orangepi@192.168.137.103:~/qwen_distributed/models/
```

### 步骤 4：启动分布式推理（支持工具调用）

**重要**：必须按照以下顺序启动节点！

```
启动顺序：Node 3 → Node 2 → Node 1 → Node 0
         (尾节点)  (中间2)  (中间1)  (头节点)
```

#### 终端 1：启动 Node 3（尾节点）

```bash
ssh orangepi@192.168.137.103
cd ~/qwen_distributed/code
python3 node_tail.py \
    --om_dir ~/qwen_distributed/models \
    --device 0 \
    --listen_port 9003 \
    --head_ip 192.168.137.100 \
    --head_port 9000
```

#### 终端 2：启动 Node 2（中间节点2）

```bash
ssh orangepi@192.168.137.102
cd ~/qwen_distributed/code
python3 node_middle.py \
    --node_id 2 \
    --om_dir ~/qwen_distributed/models \
    --device 0 \
    --listen_port 9002 \
    --next_ip 192.168.137.103 \
    --next_port 9003
```

#### 终端 3：启动 Node 1（中间节点1）

```bash
ssh orangepi@192.168.137.101
cd ~/qwen_distributed/code
python3 node_middle.py \
    --node_id 1 \
    --om_dir ~/qwen_distributed/models \
    --device 0 \
    --listen_port 9001 \
    --next_ip 192.168.137.102 \
    --next_port 9002
```

#### 终端 4：启动 Node 0（头节点 + 工具调度）

```bash
ssh orangepi@192.168.137.100
cd ~/qwen_distributed/code

# 创建输入文件
echo "查询北京天气并推荐穿衣" > input.txt

# 启动头节点
python3 node_head.py \
    --om_dir ~/qwen_distributed/models \
    --tokenizer_dir ~/qwen_distributed/tokenizer \
    --device 0 \
    --input_file input.txt \
    --max_new_tokens 100 \
    --listen_port 9000 \
    --next_ip 192.168.137.101 \
    --next_port 9001
```

### 步骤 5：观察工具调用过程

启动后，你将看到类似以下的输出：

```
[Node0-Head] === Tool Iteration 1/5 ===
[Node0-Head] Step 0: generated token 12345
[Node0-Head] Step 0: generated text '我需要'
...
[Node0-Head] Streaming detected tool_call: get_weather (id=call_001)
[Node0-Head] Sending tool call to Device 2
[Node2-Middle] Executing tool locally on Device 2
[Node2-Middle] Tool result forwarded to next node
[Node0-Head] Waiting for 1 streaming tool(s) to complete...
[Node0-Head] Aggregated tool results:
{"temperature": 15, "condition": "晴"}
[Node0-Head] Continuing with tool results after EOS
[Node0-Head] === Tool Iteration 2/5 ===
...
```

---

## 详细配置

### config.py 配置类

```python
from config import DistributedConfig4Nodes

config = DistributedConfig4Nodes(
    om_dir="/path/to/models",
    device_id=0,
    max_cache_len=1024,
    max_input_len=16,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    greedy=True,
    node_id=0,  # 0=头节点, 1=中间1, 2=中间2, 3=尾节点
)
```

### 命令行参数说明

#### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--om_dir` | OM 模型目录 | 必填 |
| `--device` | NPU 设备 ID | 0 |
| `--max_cache_len` | KV Cache 最大长度 | 1024 |
| `--max_input_len` | 单次输入最大长度 | 16 |

#### 头节点 (Node 0) 特有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_file` | 输入文本文件路径 | 必填 |
| `--max_new_tokens` | 最大生成 token 数 | 100 |
| `--temperature` | 采样温度 | 1.0 |
| `--top_k` | Top-K 采样 | 0 |
| `--top_p` | Top-P 采样 | 1.0 |
| `--greedy` | 贪婪采样 | True |
| `--listen_port` | 监听端口 | 9000 |
| `--next_ip` | 下一节点 IP | 192.168.137.101 |
| `--next_port` | 下一节点端口 | 9001 |
| `--tokenizer_dir` | tokenizer 目录 | 必填 |

---

## API 使用

### 头节点 API（支持工具调用）

```python
import numpy as np
from config import DistributedConfig4Nodes
from node_head import HeadNodeWithTools

config = DistributedConfig4Nodes(
    om_dir="/path/to/models",
    device_id=0,
    greedy=True,
    node_id=0,
)

node = HeadNodeWithTools(config)
node.init()

prompt_ids = np.array([[151644, 8948, 198]], dtype=np.int64)
generated_ids = node.generate(prompt_ids, max_new_tokens=100)

node.shutdown()
```

---

## 故障排除

### 常见问题

#### 1. 工具调用未触发

**症状**：生成文本包含 `<tool_call>` 但未执行

**解决方案**：
- 检查 `StreamingToolCallParser` 是否正确解析
- 确认工具已在所有节点注册
- 查看 Node 0 日志中的 "Streaming detected tool_call" 消息

#### 2. 工具执行超时

**症状**：`Waiting for tool(s) to complete...` 后长时间无响应

**解决方案**：
- 检查目标设备节点是否正常运行
- 确认 `MSG_TOOL_CALL` 和 `MSG_TOOL_RESULT` 消息正常传递
- 增加 `tool_timeout` 参数（默认 30 秒）

#### 3. KV Cache 不同步

**症状**：工具调用后生成结果异常

**解决方案**：
- 确保 `reset()` 方法在工具结果注入前调用
- 检查 `MSG_RESET` 是否广播到所有节点
- 查看各节点日志确认 KV Cache 已重置

#### 4. 设备内存不足

**症状**：工具加载失败或节点崩溃

**解决方案**：
- 调整 `device_memory_limit` 参数（默认 500MB）
- 减少同时加载的工具数量
- 使用更激进的 LRU 驱逐策略

---

## 性能优化建议

1. **优先本地执行**：工具调度器默认优先使用 Device 0，减少网络传输
2. **异步并行**：使用 `AsyncToolExecutor` 并发执行多个工具
3. **流式解析**：边生成边解析 tool_call，无需等待完整生成
4. **智能调度**：根据设备负载动态分配工具执行位置
5. **结果缓存**：可扩展实现工具结果缓存，避免重复执行

---

## 依赖

- Python 3.8+
- NumPy
- transformers（用于 tokenizer）
- 华为昇腾 ACL SDK（pyACL）
- 香橙派昇腾开发板 × 4

---

## 许可证

MIT License