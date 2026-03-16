# Qwen 4 节点分布式推理框架（支持 MCP 工具调用）

基于华为昇腾 ACL 的 Qwen 模型分布式推理框架，支持 **4 节点**流水线并行推理和 **MCP 工具调用**。

## 目录

- [架构概述](#架构概述)
- [核心特性](#核心特性)
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

本框架将 Qwen 模型（28 层 Transformer）切分到 **4 个设备**上进行流水线并行推理。采用 **Prefill + Decode 双模型**架构：

- Prefill 阶段：一次处理完整输入序列（seq_len=prefill_len，无 past KV）
- Decode 阶段：逐 token 生成（seq_len=1，带 past KV）
- 模型按需加载/卸载，同一时间每个节点只保留一组模型

### 模型切分详情

每个节点有 prefill 和 decode 两组模型（存放在不同目录），按需加载/卸载。

| 节点 | 模型文件 | 内容 | 层数 |
|------|---------|------|------|
| **Node 0** | embed.om + layers_0_6.om | Embedding + Transformer 层 0-6 | 7 层 |
| **Node 1** | layers_7_13.om | Transformer 层 7-13 | 7 层 |
| **Node 2** | layers_14_20.om | Transformer 层 14-20 | 7 层 |
| **Node 3** | layers_21_27.om + output.om | Transformer 层 21-27 + LM Head | 7 层 |

### Prefill + Decode 双模型 & 按需加载

```
Prefill 模型（prefill_om_dir/）：
  • seq_len = prefill_len（如 512）
  • 输入：hidden_states + attention_mask + position_ids（无 past KV）
  • attention_mask 形状：[1, 1, prefill_len, prefill_len]（纯 causal）

Decode 模型（decode_om_dir/）：
  • seq_len = 1
  • 输入：hidden_states + attention_mask + position_ids + past_key + past_value
  • attention_mask 形状：[1, 1, 1, max_cache_len+1]

按需加载策略（_ensure_mode）：
  • 启动时不加载任何模型（_loaded_mode = None）
  • 第一次 forward（prefill）→ 加载 prefill 模型 → 执行
  • 第二次 forward（decode）→ 卸载 prefill → 加载 decode 模型 → 执行
  • 后续 decode 步骤不再切换
  • reset 后下一次 prefill 再切换回来
```

---

## 核心特性

### 1. 分布式流水线推理

- **4 节点流水线**：模型切分到 4 个设备，流水线并行执行
- **Prefill + Decode 双模型**：Prefill 支持长序列输入，Decode 逐 token 生成
- **按需加载/卸载**：同一时间只保留一组模型，节省设备内存
- **KV Cache 管理**：每个节点独立管理 7 层 KV Cache
- **网络通信**：基于 TCP Socket + Pickle 序列化

### 2. MCP 工具调用系统

- **流式解析**：在生成过程中实时解析 tool_call
- **异步执行**：检测到完整工具调用后立即异步执行，不阻塞生成
- **分布式调度**：工具可在任意设备上执行，实现负载均衡
- **结果聚合**：生成结束后等待所有工具完成，聚合结果并注入下一轮推理

---

## 模型参数

### ATC 转换命令（Prefill + Decode）

每个节点有独立的转换脚本（`convert_to_om_node0.sh` ~ `convert_to_om_node3.sh`），分别在对应设备上执行。

```bash
# Prefill embed（seq_len=512，无 past KV）
atc --model="prefill/embed.onnx" --input_shape="input_ids:1,512"

# Prefill block（seq_len=512，无 past KV）
atc --model="prefill/layers_X_Y.onnx" \
    --input_shape="hidden_states:1,512,2048;attention_mask:1,1,512,512;position_ids:1,512"

# Prefill output
atc --model="prefill/output.onnx" --input_shape="hidden_states:1,512,2048"

# Decode embed（seq_len=1）
atc --model="decode/embed.onnx" --input_shape="input_ids:1,1"

# Decode block（seq_len=1，带 past KV）
atc --model="decode/layers_X_Y.onnx" \
    --input_shape="hidden_states:1,1,2048;attention_mask:1,1,1,1025;position_ids:1,1;past_key:7,1,8,1024,128;past_value:7,1,8,1024,128"

# Decode output
atc --model="decode/output.onnx" --input_shape="hidden_states:1,1,2048"
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_size` | 2048 | 隐藏层维度 |
| `num_attention_heads` | 16 | 注意力头数 |
| `num_key_value_heads` | 8 | KV 头数 (GQA) |
| `head_dim` | 128 | 每个头的维度 |
| `num_hidden_layers` | 28 | Transformer 总层数 |
| `vocab_size` | 151936 | 词表大小 |
| `prefill_len` | 512 | Prefill 阶段输入序列长度 |
| `max_cache_len` | 1024 | KV Cache 最大缓存长度 |

---

## 文件结构

```
code/
├── config.py              # 4 节点配置类（支持 prefill/decode 双模型目录）
├── network.py             # TCP 网络通信模块
├── kvcache.py             # KV Cache 管理
├── acl_model.py           # ACL 模型封装
├── utils.py               # 工具函数（含 build_prefill_attention_mask）
│
├── node_head.py           # 头节点实现 (Node 0) - 支持工具调用
├── node_middle.py         # 中间节点实现 (Node 1, 2) - 支持工具执行
├── node_tail.py           # 尾节点实现 (Node 3) - 支持工具执行
│
├── tools/                 # 工具调用系统
│   ├── __init__.py
│   ├── tool_manager.py
│   ├── tool_coordinator.py
│   ├── tool_scheduler.py
│   ├── tool_agent.py
│   ├── streaming_parser.py
│   ├── async_executor.py
│   ├── result_buffer.py
│   └── builtin_tools/
│       ├── __init__.py
│       ├── weather_tool.py
│       ├── calculator_tool.py
│       ├── time_tool.py
│       ├── unit_converter_tool.py
│       └── translate_tool.py
│
├── README.md              # 本文档
│
convert_to_om_node0.sh     # Node 0 OM 转换脚本（embed + layers_0_6）
convert_to_om_node1.sh     # Node 1 OM 转换脚本（layers_7_13）
convert_to_om_node2.sh     # Node 2 OM 转换脚本（layers_14_20）
convert_to_om_node3.sh     # Node 3 OM 转换脚本（layers_21_27 + output）
```

---

## 香橙派昇腾分布式部署指南

### 硬件准备

| 设备 | IP 地址 | 角色 | 需要的模型文件（prefill/ + decode/ 各一份） |
|------|---------|------|---------------------------------------------|
| 香橙派 1 | 192.168.137.100 | Node 0 (头节点) | embed.om, layers_0_6.om |
| 香橙派 2 | 192.168.137.101 | Node 1 (中间节点1) | layers_7_13.om |
| 香橙派 3 | 192.168.137.102 | Node 2 (中间节点2) | layers_14_20.om |
| 香橙派 4 | 192.168.137.103 | Node 3 (尾节点) | layers_21_27.om, output.om |

### 步骤 1：在各设备上转换 OM 模型

将 ONNX 模型和对应的转换脚本拷贝到各设备，分别执行：

```bash
# Node 0 设备上
bash convert_to_om_node0.sh

# Node 1 设备上
bash convert_to_om_node1.sh

# Node 2 设备上
bash convert_to_om_node2.sh

# Node 3 设备上
bash convert_to_om_node3.sh
```

转换完成后，每个设备的 `model_om/` 目录下会有 `prefill/` 和 `decode/` 两个子目录。

### 步骤 2：分发代码

```bash
for ip in 192.168.137.100 192.168.137.101 192.168.137.102 192.168.137.103; do
    scp -r code/ orangepi@$ip:~/qwen_distributed2/
done
```

### 步骤 3：启动分布式推理

**启动顺序：Node 3 → Node 2 → Node 1 → Node 0**

#### 终端 1：启动 Node 3（尾节点）

```bash
ssh orangepi@192.168.137.103
cd ~/qwen_distributed2/code
python3 node_tail.py \
    --prefill_om_dir ~/qwen_distributed2/model_om/prefill \
    --decode_om_dir ~/qwen_distributed2/model_om/decode \
    --device 0 \
    --prefill_len 512 \
    --max_cache_len 1024 \
    --listen_port 9003 \
    --head_ip 192.168.137.100 \
    --head_port 9000
```

#### 终端 2：启动 Node 2（中间节点2）

```bash
ssh orangepi@192.168.137.102
cd ~/qwen_distributed2/code
python3 node_middle.py \
    --node_id 2 \
    --prefill_om_dir ~/qwen_distributed2/model_om/prefill \
    --decode_om_dir ~/qwen_distributed2/model_om/decode \
    --device 0 \
    --prefill_len 512 \
    --max_cache_len 1024 \
    --listen_port 9002 \
    --next_ip 192.168.137.103 \
    --next_port 9003
```

#### 终端 3：启动 Node 1（中间节点1）

```bash
ssh orangepi@192.168.137.101
cd ~/qwen_distributed2/code
python3 node_middle.py \
    --node_id 1 \
    --prefill_om_dir ~/qwen_distributed2/model_om/prefill \
    --decode_om_dir ~/qwen_distributed2/model_om/decode \
    --device 0 \
    --prefill_len 512 \
    --max_cache_len 1024 \
    --listen_port 9001 \
    --next_ip 192.168.137.102 \
    --next_port 9002
```

#### 终端 4：启动 Node 0（头节点）

```bash
ssh orangepi@192.168.137.100
cd ~/qwen_distributed2/code
echo "查询北京天气并推荐穿衣" > input.txt

python3 node_head.py \
    --prefill_om_dir ~/qwen_distributed2/model_om/prefill \
    --decode_om_dir ~/qwen_distributed2/model_om/decode \
    --tokenizer_dir ~/qwen_distributed2/tokenizer \
    --device 0 \
    --prefill_len 512 \
    --max_cache_len 1024 \
    --input_file input.txt \
    --max_new_tokens 100 \
    --greedy \
    --listen_port 9000 \
    --next_ip 192.168.137.101 \
    --next_port 9001
```

---

## 详细配置

### config.py 配置类

```python
from config import DistributedConfig4Nodes

config = DistributedConfig4Nodes(
    prefill_om_dir="/path/to/models/prefill",
    decode_om_dir="/path/to/models/decode",
    device_id=0,
    max_cache_len=1024,
    prefill_len=512,
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
| `--prefill_om_dir` | Prefill OM 模型目录 | 必填 |
| `--decode_om_dir` | Decode OM 模型目录 | 必填 |
| `--device` | NPU 设备 ID | 0 |
| `--max_cache_len` | KV Cache 最大长度 | 1024 |
| `--prefill_len` | Prefill 阶段输入序列长度 | 512 |

#### 头节点 (Node 0) 特有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_file` | 输入文本文件路径 | 必填 |
| `--tokenizer_dir` | tokenizer 目录 | 必填 |
| `--max_new_tokens` | 最大生成 token 数 | 100 |
| `--temperature` | 采样温度 | 1.0 |
| `--top_k` | Top-K 采样 | 0 |
| `--top_p` | Top-P 采样 | 1.0 |
| `--greedy` | 贪婪采样 | True |
| `--listen_port` | 监听端口 | 9000 |
| `--next_ip` | 下一节点 IP | 192.168.137.101 |
| `--next_port` | 下一节点端口 | 9001 |

---

## API 使用

```python
import numpy as np
from config import DistributedConfig4Nodes
from node_head import HeadNodeWithTools

config = DistributedConfig4Nodes(
    prefill_om_dir="/path/to/models/prefill",
    decode_om_dir="/path/to/models/decode",
    device_id=0,
    prefill_len=512,
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

### 1. 模型切换延迟

**症状**：Prefill → Decode 切换时有明显延迟

**说明**：这是按需加载的正常行为。每次 prefill→decode 切换需要卸载旧模型并加载新模型。后续 decode 步骤不再切换。

### 2. 工具调用未触发

检查 `StreamingToolCallParser` 是否正确解析，确认工具已在所有节点注册。

### 3. KV Cache 不同步

确保 `reset()` 在工具结果注入前调用，检查 `MSG_RESET` 是否广播到所有节点。

### 4. 设备内存不足

按需加载策略已将内存占用减半（同一时间只保留一组模型）。如仍不足，可减小 `prefill_len` 或 `max_cache_len`。

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
