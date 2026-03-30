# ONNX Local Execution - 完整文档

单机运行 ONNX 模型的实现，支持多轮对话、工具调用和并行/串行执行计划。

## 目录

- [概述](#概述)
- [文件结构](#文件结构)
- [快速开始](#快速开始)
- [多轮对话系统](#多轮对话系统)
- [执行计划模式](#执行计划模式)
- [详细设计文档](#详细设计文档)
- [性能考虑](#性能考虑)

---

## 概述

### 特点

- 使用 ONNX Runtime 执行模型（CPU）
- 单机运行，无需网络通信
- 支持工具调用（继承自 code 文件夹）
- 支持 prefill/decode 两阶段推理
- KV cache 管理和快照机制
- **支持多轮对话**：通过 prefill 模型统一处理历史记忆
- **执行计划模式**：支持并行/串行工具调用

### 与其他实现的对比

| 特性 | file_pipeline | code | onnx_local |
|------|--------------|------|------------|
| 模型类型 | ONNX | OM | ONNX |
| 执行设备 | CPU | NPU | CPU |
| 分布式 | 否 | 是 | 否 |
| 工具调用 | 否 | 是 | 是 |
| 网络通信 | 否 | 是 | 否 |
| 多轮对话 | 否 | 是 | 是 |
| 并行工具调用 | 否 | 否 | 是 |

---

## 文件结构

```
onnx_local/
├── convert_to_onnx.py              # ONNX 模型导出脚本
├── qwen3_custom_modules.py         # 自定义 Qwen3 模块
├── config.json                     # 配置文件
├── code/
│   ├── README.md                   # 本文档
│   ├── run_local.py                # 单轮推理脚本
│   ├── run_local_multiturn.py      # 多轮对话脚本
│   ├── run_local_multiturn_plan.py # 执行计划模式脚本
│   ├── onnx_model.py               # ONNX 模型封装
│   ├── kvcache.py                  # KV cache 管理
│   ├── config.py                   # 配置
│   ├── utils.py                    # 工具函数
│   ├── user_prompt.txt             # 用户提示示例
│   └── tools/                      # 工具系统
│       ├── __init__.py
│       ├── tool_manager.py
│       ├── tool_coordinator.py
│       ├── tool_scheduler.py
│       ├── tool_agent.py
│       ├── streaming_parser.py
│       ├── async_executor.py
│       ├── execution_plan_parser.py    # 执行计划解析器
│       ├── dependency_resolver.py      # 依赖关系解析器
│       ├── plan_executor.py            # 执行计划调度器
│       └── builtin_tools/
│           ├── __init__.py
│           ├── calculator_tool.py
│           ├── weather_tool.py
│           ├── time_tool.py
│           ├── translate_tool.py
│           └── unit_converter_tool.py
└── onnx_models/                    # ONNX 模型目录
    ├── system/                     # System 模型（可选）
    ├── prefill/                    # Prefill 模型
    └── decode/                     # Decode 模型
```

---

## 快速开始

### 1. 模型转换（ONNX 导出）

首先需要将 PyTorch 模型转换为 ONNX 格式：

```bash
python convert_to_onnx.py --model_path D:\qwen_split\qwen3_1.7b --onnx_dir ./onnx_models --system_len 1024 --prefill_len 512 --max_cache_len 1024
```

**参数说明**：
- `--model_path`：PyTorch 模型路径
- `--onnx_dir`：ONNX 模型输出目录
- `--system_len`：System 阶段最大输入长度（默认 1024，推荐用于固定 system prompt）
- `--prefill_len`：Prefill 阶段最大输入长度（默认 512）
- `--max_cache_len`：KV cache 最大长度（默认 1024）

这会生成三组 ONNX 模型：
- `onnx_models/system/`：处理固定 system prompt（最大 1024 tokens，无 past KV，可选）
- `onnx_models/prefill/`：处理用户输入和历史记忆（最大 512 tokens，带 past KV）
- `onnx_models/decode/`：逐 token 生成（带 past KV）

### 2. 单轮推理

```bash
python run_local.py --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --input_file ./user_prompt.txt --max_new_tokens 100
```

### 3. 多轮对话

#### 不使用 System 模型（传统方式，已弃用）

```bash
# 注意：此方式已不推荐使用，因为每次对话都需要重新计算 system prompt
python run_local_multiturn.py --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --interactive
```

#### 使用 System 模型（推荐）

```bash
# 交互模式
python run_local_multiturn.py --system_onnx_dir ../onnx_models/system --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --system_kv_dir ./system_kv_cache --system_len 1024 --prefill_len 512 --interactive

# 批量问题模式
python run_local_multiturn.py --system_onnx_dir ../onnx_models/system --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --system_kv_dir ./system_kv_cache --system_len 1024 --prefill_len 512 --questions "北京天气怎么样？" "那上海呢？" "温度差多少？"
```

**System 模型优势**：
- 固定的 system prompt（工具描述）只计算一次
- 生成的 KV cache 可以复用，节省计算
- Prefill 的 512 tokens 完全用于用户输入和历史记忆
- **KV cache 文件缓存**：首次运行生成缓存文件，后续运行直接加载，无需重新计算

### 4. 执行计划模式（支持并行/串行工具调用）

#### 不使用 System 模型（传统方式）

```bash
python run_local_multiturn_plan.py --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --enable_plan_mode --interactive
```

#### 使用 System 模型（推荐）

```bash
python run_local_multiturn_plan.py --system_onnx_dir ../onnx_models/system --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --system_len 1024 --prefill_len 512 --enable_plan_mode --interactive
```

**System 模型优势**：
- 固定的 system prompt（工具描述）只计算一次
- 生成的 KV cache 可以复用，节省计算
- Prefill 的 512 tokens 完全用于用户输入和历史记忆

---

## 多轮对话系统

### 核心设计理念

#### 1. Prefill 模型统一处理

系统统一使用 prefill 模型处理所有输入：
- 工具描述
- 历史对话记忆
- 用户问题
- 工具结果提示

**优势**：
- 避免了 system 模型无 past KV 的限制
- 灵活的 past KV 机制支持历史记忆追加
- 简洁统一的架构设计

#### 2. KV Cache 管理策略

系统维护多个 KV cache 快照：

- **system_only_kv_snapshot**: 仅含工具描述的 KV（永不变）
- **base_kv_snapshot**: 第一轮 system + 所有历史记忆的 KV
- **round2_system_only_kv_snapshot**: 仅含第二轮 system prompt 的 KV（永不变）
- **round2_base_kv_snapshot**: 第二轮 system + 所有历史记忆的 KV
- **question_kv_snapshot**: 当前问题第一轮推理后的 KV

#### 3. 两轮推理流程

每个问题的处理分为两轮推理：

**第一轮推理**（用户问题 → 工具调用）：
1. 恢复到 base_kv_snapshot（包含工具描述 + 历史记忆）
2. Prefill + Decode: 用户问题 → 工具调用
3. 保存 question_kv_snapshot

**第二轮推理**（工具结果 → 最终回答）：
4. 恢复到 round2_base_kv_snapshot（包含工具结果提示 + 历史记忆）
5. Prefill + Decode: 工具结果 → 最终回答
6. 存储到对话记忆
7. 增量追加最新历史到 base_kv_snapshot 和 round2_base_kv_snapshot

### 对话记忆格式

```python
{
    'question': '用户问题',
    'tool_calls': [工具调用列表],
    'tool_results': [工具结果列表],
    'final_answer': '助手最终回答'
}
```

记忆文本格式化为：

```
<|im_start|>user
北京天气怎么样？<|im_end|>
<|im_start|>assistant
根据查询结果，北京今天是晴天，温度25°C，湿度60%。<|im_end|>
```

### 完整示例：三轮对话

#### 第一次提问："北京天气怎么样？"

**第一轮推理**：
1. 初始化：prefill 处理工具描述 → base_kv_snapshot（80 tokens）
2. 用户问题：`<|im_start|>user\n北京天气怎么样？\n\n【输出格式提示】<|im_end|>\n<|im_start|>assistant\n`
3. 模型输出：`{"tool_name": "get_weather", "arguments": {"city": "Beijing"}}`
4. 工具执行：返回 "北京今天晴天，温度25°C"

**第二轮推理**：
5. 初始化：prefill 处理工具结果提示 → round2_base_kv_snapshot（30 tokens）
6. 工具结果：`<|im_start|>user\n【工具结果】\n北京天气怎么样？\n\n【输出格式提示】<|im_end|>\n<|im_start|>assistant\n`
7. 模型输出：`根据查询结果，北京今天是晴天，温度25°C，湿度60%。`
8. 存储记忆并更新 base_kv_snapshot

#### 第二次提问："那上海呢？"

**第一轮推理**：
1. 恢复到 base_kv_snapshot（80 tokens 工具描述 + 50 tokens 历史记忆 = 130 tokens）
2. 用户问题：`<|im_start|>user\n那上海呢？\n\n【输出格式提示】<|im_end|>\n<|im_start|>assistant\n`
3. 模型理解"那上海呢"指上海天气（基于历史记忆）
4. 模型输出：`{"tool_name": "get_weather", "arguments": {"city": "Shanghai"}}`
5. 工具执行：返回 "上海今天多云，温度22°C"

**第二轮推理**：
6. 恢复到 round2_base_kv_snapshot（30 tokens 提示 + 50 tokens 历史记忆 = 80 tokens）
7. 工具结果：`<|im_start|>user\n【工具结果】\n那上海呢？\n\n【输出格式提示】<|im_end|>\n<|im_start|>assistant\n`
8. 模型输出：`上海今天是多云天气，温度22°C。相比北京，上海温度稍低一些。`
9. 存储记忆并更新 base_kv_snapshot

#### 第三次提问："两个城市温度差多少？"

**第一轮推理**：
1. 恢复到 base_kv_snapshot（80 + 50 + 70 = 200 tokens）
2. 用户问题：`<|im_start|>user\n两个城市温度差多少？\n\n【输出格式提示】<|im_end|>\n<|im_start|>assistant\n`
3. 模型理解"两个城市"指北京和上海，且已知温度（基于历史记忆）
4. 模型输出：`{"tool_name": "calculator", "arguments": {"expression": "25 - 22"}}`
5. 工具执行：返回 "3"

**第二轮推理**：
6. 恢复到 round2_base_kv_snapshot（30 + 50 + 70 = 150 tokens）
7. 工具结果：`<|im_start|>user\n【工具结果】\n两个城市温度差多少？\n\n【输出格式提示】<|im_end|>\n<|im_start|>assistant\n`
8. 模型输出：`根据之前查询的结果，北京温度是25°C，上海温度是22°C，温度差是3°C。`

### 关键特性

1. **上下文理解**：模型能理解代词和省略（"那上海呢"、"两个城市"）
2. **避免重复查询**：第三次直接用已知温度计算，无需重新查询
3. **KV Cache 复用**：每次新问题都基于包含历史记忆的 base_kv_snapshot
4. **增量追加**：历史记忆增量追加到 KV cache，无需重新计算

### System 模型架构（可选）

System 模型用于处理固定的 system prompt（工具描述等），生成的 KV cache 可以复用，避免每次对话都重新计算。

#### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     多轮对话流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. System 模型（只运行一次）                                 │
│     ├─ 输入：固定的 system prompt（工具描述等）              │
│     ├─ 最大长度：1024 tokens                                 │
│     └─ 输出：System KV cache（保存复用）                     │
│                                                               │
│  2. Prefill 模型（每轮对话）                                  │
│     ├─ 输入：用户问题 + 历史记忆                             │
│     ├─ 最大长度：512 tokens                                  │
│     ├─ Past KV：System KV cache                              │
│     └─ 输出：工具调用计划或直接回答                          │
│                                                               │
│  3. Decode 模型（逐 token 生成）                             │
│     ├─ 输入：单个 token                                      │
│     ├─ Past KV：System KV + Prefill KV                       │
│     └─ 输出：下一个 token                                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

#### 工作流程

**初始化阶段（只运行一次）**：

1. 加载 System 模型
2. 处理第一轮 System Prompt（工具描述）→ 生成 System KV cache
3. 处理第二轮 System Prompt（工具结果提示）→ 生成 Round2 System KV cache

**对话阶段（每次对话）**：

1. **第一轮推理**：恢复 System KV → Prefill 处理用户问题 → 生成工具调用计划
2. **执行工具**：解析执行计划 → 并行/串行执行工具
3. **第二轮推理**：恢复 Round2 System KV → Prefill 处理工具结果 → 生成最终回答

#### 性能对比

| 方案 | System Prompt 计算 | Prefill 可用 tokens | 首次延迟 | 后续延迟 |
|------|-------------------|---------------------|----------|----------|
| 不使用 System 模型 | 每次对话 | ~300 (512 - 工具描述) | 低 | 高 |
| 使用 System 模型 | 只运行一次 | 512 (完整) | 稍高 | 低 |

#### KV Cache 文件缓存

为了进一步优化性能，系统支持将 System 模型生成的 KV cache 保存到文件，后续运行时直接加载。

**工作流程**：

1. **首次运行**：
   - 加载 System 模型
   - 处理 system prompt，生成 KV cache
   - 保存到 `system_kv_cache/` 目录
   - 继续正常运行

2. **后续运行**：
   - 检查缓存目录
   - 直接从文件加载 KV cache
   - 无需加载 System 模型
   - 显著加速启动

**缓存文件**：

- 普通模式：
  - `system_kv_cache/system_only_kv.npz`: 第一轮 system prompt 的 KV
  - `system_kv_cache/round2_system_only_kv.npz`: 第二轮 system prompt 的 KV

- Plan 模式：
  - `system_kv_cache/system_only_kv_plan.npz`: Plan mode 第一轮 system prompt 的 KV
  - `system_kv_cache/round2_system_only_kv_plan.npz`: Plan mode 第二轮 system prompt 的 KV

**优势**：

- **启动加速**：后续运行无需加载和运行 System 模型，启动时间从数秒降至毫秒级
- **内存节省**：不需要保持 System 模型在内存中
- **磁盘占用小**：使用 `np.savez_compressed` 压缩，单个文件通常只有几 MB
- **自动管理**：首次运行自动生成，后续自动加载，无需手动干预
- **独立缓存**：普通模式和 Plan 模式使用不同的缓存文件，互不干扰

**使用示例**：

```bash
# 指定缓存目录（默认为 ./system_kv_cache）
python run_local_multiturn.py --system_onnx_dir ../onnx_models/system --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir /path/to/tokenizer --system_kv_dir ./my_cache_dir --system_len 1024 --prefill_len 512 --interactive
```

**注意事项**：

- 如果修改了工具配置或 system prompt，需要删除缓存文件重新生成
- 缓存文件与模型配置（system_len、工具列表等）绑定，配置变化后需重新生成
- 可以通过删除 `system_kv_cache/` 目录来清除所有缓存

---

## 执行计划模式

### 概述

执行计划模式（Plan Mode）是对现有工具调用系统的增强，支持在单次对话中并行和串行调用多个工具。

### 核心特性

#### 1. 三种执行模式

- **并行模式（Parallel）**：工具间无依赖，可同时执行
- **串行模式（Sequential）**：工具间有依赖，必须按顺序执行
- **混合模式（Mixed）**：部分并行，部分串行

#### 2. 依赖关系管理

- 使用 `depends_on` 声明步骤间的依赖关系
- 使用 `$stepN_result` 或自定义引用名引用前序结果
- 使用 `.字段名` 访问结果的特定字段
- 自动进行拓扑排序和循环依赖检测

#### 3. 智能调度

- 自动识别可并行执行的步骤
- 组内并行，组间串行
- 支持参数引用自动替换

### 使用方法

```bash
python run_local_multiturn_plan.py --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --enable_plan_mode --interactive
```

### 示例场景

#### 示例 1：并行查询（无依赖）

**用户问题**："北京和上海的天气如何？"

**模型输出**：
```json
{
  "execution_plan": {
    "mode": "parallel",
    "steps": [
      {
        "step_id": 1,
        "parallel_calls": [
          {"tool_name": "get_weather", "arguments": {"city": "北京"}},
          {"tool_name": "get_weather", "arguments": {"city": "上海"}}
        ]
      }
    ]
  }
}
```

**执行流程**：
1. 两个 `get_weather` 调用并行执行
2. 执行时间 ≈ 单次调用时间（而非 2 倍）

#### 示例 2：串行调用（有依赖）

**用户问题**："北京现在多少华氏度？"

**模型输出**：
```json
{
  "execution_plan": {
    "mode": "sequential",
    "steps": [
      {
        "step_id": 1,
        "call": {
          "tool_name": "get_weather",
          "arguments": {"city": "北京"}
        },
        "output_ref": "$step1_result"
      },
      {
        "step_id": 2,
        "call": {
          "tool_name": "unit_convert",
          "arguments": {
            "value": "$step1_result.temperature",
            "from_unit": "celsius",
            "to_unit": "fahrenheit"
          }
        },
        "depends_on": [1]
      }
    ]
  }
}
```

**执行流程**：
1. 执行 step 1：获取北京天气（摄氏温度）
2. 缓存结果：`$step1_result = {"temperature": 15, ...}`
3. 执行 step 2：将 `$step1_result.temperature` (15) 转换为华氏度
4. 返回最终结果

#### 示例 3：混合模式（部分并行，部分串行）

**用户问题**："北京和上海的温差是多少？"

**模型输出**：
```json
{
  "execution_plan": {
    "mode": "mixed",
    "steps": [
      {
        "step_id": 1,
        "parallel_calls": [
          {
            "tool_name": "get_weather",
            "arguments": {"city": "北京"},
            "output_ref": "$beijing"
          },
          {
            "tool_name": "get_weather",
            "arguments": {"city": "上海"},
            "output_ref": "$shanghai"
          }
        ]
      },
      {
        "step_id": 2,
        "call": {
          "tool_name": "calculator",
          "arguments": {
            "expression": "$beijing.temperature - $shanghai.temperature"
          }
        },
        "depends_on": [1]
      }
    ]
  }
}
```

**执行流程**：
1. **Group 1**（并行）：同时获取北京和上海天气
   - 缓存：`$beijing = {"temperature": 15, ...}`
   - 缓存：`$shanghai = {"temperature": 18, ...}`
2. **Group 2**（串行）：计算温差
   - 解析引用：`$beijing.temperature` → 15，`$shanghai.temperature` → 18
   - 执行：`calculator("15 - 18")` → -3
3. 返回最终结果

### 架构说明

#### 核心模块

```
onnx_local/code/tools/
├── execution_plan_parser.py   # 执行计划解析器
├── dependency_resolver.py     # 依赖关系解析器
└── plan_executor.py           # 执行计划调度器
```

#### 数据流

```
用户输入
    ↓
Round 1: 生成执行计划
    ↓
ExecutionPlanParser.parse()
    ↓
DependencyResolver.topological_sort()
    ↓
PlanExecutor.execute()
    ├─ Group 1 (并行)
    ├─ Group 2 (并行)
    └─ Group 3 (串行)
    ↓
Round 2: 生成最终回答
    ↓
最终输出
```

### 与传统模式的对比

| 特性 | 传统模式 | 执行计划模式 |
|------|----------|--------------|
| 工具调用方式 | 逐个调用 | 批量调用 |
| 并行支持 | 有限 | 完全支持 |
| 依赖管理 | 手动 | 自动 |
| 参数引用 | 不支持 | 支持 |
| 执行效率 | 较低 | 较高 |
| 复杂场景 | 需多轮 | 单轮完成 |

### 兼容性

- 完全向后兼容传统工具调用模式
- 如果模型未输出执行计划格式，自动回退到传统模式
- 可通过 `--enable_plan_mode=False` 禁用执行计划模式

---

## 详细设计文档

### System Prompt 设计

#### 传统模式

```
你是AI助手，可用工具：
- get_weather(city*:string) — Get weather information
- calculator(expression*:string) — Perform calculations
...

调用工具时输出JSON：{"tool_name":"名称","arguments":{"参数":"值"}}
每行一个JSON，可同时调用多个工具。
```

#### 执行计划模式

```
你是AI助手，可用工具：
- get_weather(city*:string) — Get weather information
- calculator(expression*:string) — Perform calculations
...

【工具调用规则】
1. 分析用户问题，判断需要哪些工具
2. 判断工具间是否有依赖关系：
   - 无依赖：可并行执行
   - 有依赖：必须串行执行，后续工具的参数引用前序工具的结果
3. 输出执行计划（JSON格式）

【输出格式】
并行模式：{"execution_plan":{"mode":"parallel","steps":[...]}}
串行模式：{"execution_plan":{"mode":"sequential","steps":[...]}}
混合模式：{"execution_plan":{"mode":"mixed","steps":[...]}}
```

### 性能优化

#### 并行执行

- 使用 `ThreadPoolExecutor` 实现真正的并行
- 默认最大 4 个工作线程
- 可通过 `PlanExecutor(max_workers=N)` 调整

#### 缓存机制

- 结果缓存避免重复计算
- 支持 `$stepN_result` 和自定义引用名
- 自动管理缓存生命周期

#### 拓扑排序

- O(V+E) 时间复杂度
- 自动检测循环依赖
- 最大化并行度

### 错误处理

#### 格式错误

- JSON 解析失败 → 回退到传统模式
- 缺少必需字段 → 跳过该步骤
- 工具名称无效 → 返回错误信息

#### 依赖错误

- 循环依赖 → 拒绝执行，返回错误
- 引用不存在 → 使用 None 或保留原值
- 字段不存在 → 返回 None

#### 执行错误

- 工具执行失败 → 记录错误，继续执行其他步骤
- 超时 → 60 秒超时，返回部分结果
- 线程异常 → 捕获并记录，不影响其他线程

---

## 性能考虑

### 优点

1. **真正的多轮对话**：模型能够理解上下文
2. **统一的处理流程**：所有输入都用 prefill 模型处理
3. **灵活的历史记忆**：支持任意长度的历史记忆追加
4. **并行工具调用**：显著提升复杂场景的执行效率
5. **智能依赖管理**：自动识别并行机会

### 注意事项

#### 1. KV Cache 容量

- 每次追加记忆会增加 past_len
- 需要监控 `past_len` 避免超过 `max_cache_len`
- 建议实现记忆压缩或选择性遗忘

#### 2. Prefill 模型调用次数

- 每次新问题: 1次（处理历史记忆和工具描述）
- 如果有工具调用: 再1次（处理工具结果）
- 总计: 每个问题最多2次 prefill 调用

#### 3. 记忆管理

可以实现记忆压缩：

```python
def _compress_memory(self):
    """压缩对话记忆，只保留最近 N 条"""
    if len(self.conversation_memory) > MAX_MEMORY_ITEMS:
        self.conversation_memory = self.conversation_memory[-MAX_MEMORY_ITEMS:]
```

#### 4. 模型能力依赖

- 执行计划模式需要模型能够理解并输出正确的格式
- 建议使用支持 function calling 的模型
- 可以通过 prompt engineering 提升模型的输出准确性

### 调试技巧

#### 启用详细日志

代码中已包含详细的日志输出：
- `[MultiTurn]` 前缀：多轮对话相关
- `[PlanExecutor]` 前缀：执行计划相关
- `[DependencyResolver]` 前缀：依赖解析相关
- `[ExecutionPlanParser]` 前缀：计划解析相关

#### 检查 KV Cache 状态

```python
print(f"past_len: {runner.past_len}")
print(f"base_kv current_len: {runner.base_kv_snapshot['current_len']}")
```

#### 验证执行计划

在第一轮推理后，检查模型输出是否包含正确的 `execution_plan` 格式。

---

## 扩展方向

### 1. 记忆持久化

```python
def save_conversation_history(self, filepath):
    """保存对话历史到文件"""
    with open(filepath, 'w') as f:
        json.dump(self.conversation_memory, f)

def load_conversation_history(self, filepath):
    """从文件加载对话历史"""
    with open(filepath, 'r') as f:
        self.conversation_memory = json.load(f)
```

### 2. 多轮工具调用

支持 Round 2 继续调用工具，实现更复杂的交互。

### 3. 条件执行

根据前序结果决定是否执行后续步骤。

### 4. 执行计划可视化

展示依赖图和执行进度，帮助调试和理解。

### 5. 更复杂的引用

支持嵌套引用、数组索引等高级特性。

---

## 总结

本系统通过巧妙的 KV cache 管理和 prefill 模型的灵活使用，实现了强大的多轮对话能力。执行计划模式进一步增强了工具调用的效率，支持并行和串行执行。这种设计在保持单机执行简洁性的同时，提供了接近分布式系统的功能，适合需要多轮交互和复杂工具调用的应用场景。

## 许可证

与主项目保持一致。
