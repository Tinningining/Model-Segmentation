# ONNX Local Execution

单机运行 ONNX 模型的实现，借鉴 file_pipeline 的 ONNX 执行方式和 code 文件夹的工具调用能力。

## 特点

- 使用 ONNX Runtime 执行模型（CPU）
- 单机运行，无需网络通信
- 支持工具调用（继承自 code 文件夹）
- 支持 system/prefill/decode 三阶段推理
- KV cache 管理
- **支持多轮对话**：每轮推理都通过 system 模型处理历史记忆
- **System 模型复用**：同一个 system 模型可处理工具描述、历史记忆、工具结果提示等

## 文件结构

```
onnx_local/
├── convert_to_onnx.py              # ONNX 模型导出脚本
├── qwen3_custom_modules.py         # 自定义 Qwen3 模块
├── config.json                     # 配置文件
├── code/
│   ├── README.md                   # 说明文档
│   ├── run_local.py                # 单轮推理脚本
│   ├── run_local_multiturn.py      # 多轮对话脚本
│   ├── onnx_model.py               # ONNX 模型封装
│   ├── kvcache.py                  # KV cache 管理
│   ├── config.py                   # 配置
│   ├── utils.py                    # 工具函数
│   ├── MULTITURN_DESIGN.md         # 多轮对话设计文档 V1
│   ├── MULTITURN_DESIGN_V2.md      # 多轮对话设计文档 V2（推荐）
│   ├── user_prompt.txt             # 用户提示示例
│   └── tools/                      # 工具系统（从 code 复制）
│       ├── __init__.py
│       ├── tool_manager.py
│       ├── tool_coordinator.py
│       ├── tool_agent.py
│       ├── streaming_parser.py
│       ├── async_executor.py
│       └── builtin_tools/
│           ├── __init__.py
│           ├── calculator_tool.py
│           ├── weather_tool.py
│           ├── time_tool.py
│           ├── translate_tool.py
│           └── unit_converter_tool.py
└── file_pipeline/                  # 参考实现（ONNX 管道）
    ├── run_pipeline_auto.py
    ├── stage_*.py
    └── ...
```

## 使用方法

### 模型转换（ONNX 导出）

首先需要将 PyTorch 模型转换为 ONNX 格式。在 `onnx_local/code/` 目录下运行：

```bash
python ../convert_to_onnx.py --model_path D:\qwen_split\qwen3_1.7b --onnx_dir ../onnx_models --system_len 256 --prefill_len 512 --max_cache_len 1024
```

**参数说明**：
- `--model_path`：PyTorch 模型路径
- `--onnx_dir`：ONNX 模型输出目录
- `--system_len`：System 阶段最大输入长度（默认 256）
- `--prefill_len`：Prefill 阶段最大输入长度（默认 512）
- `--max_cache_len`：KV cache 最大长度（默认 1024）

这会生成三组 ONNX 模型：
- `onnx_models/system/`：处理 system prompt（无 past KV）
  - `embed.onnx`、`layers_0_6.onnx`、`layers_7_13.onnx`、`layers_14_20.onnx`、`layers_21_27.onnx`、`output.onnx`、`config.json`
- `onnx_models/prefill/`：处理用户输入（带 past KV）
  - 同上结构
- `onnx_models/decode/`：逐 token 生成（带 past KV）
  - 同上结构

**相关文件**：
- `convert_to_onnx.py`：导出脚本
- `qwen3_custom_modules.py`：自定义 Qwen3 模块定义
- `config.json`：模型配置

### 基本用法（不使用 system KV cache）

```bash
python run_local.py --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --input_file ./user_prompt.txt --max_new_tokens 100
```

### 使用 system KV cache（推荐，可复用）

首次运行时需要提供 system 模型来生成 KV cache：

```bash
python run_local.py --system_onnx_dir ../onnx_models/system --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --system_kv_dir ../system_kv_cache --input_file ./user_prompt.txt --max_new_tokens 100
```

后续运行可以复用缓存的 system KV，无需再提供 system 模型：

```bash
python run_local.py --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --system_kv_dir ../system_kv_cache --input_file ./user_prompt.txt --max_new_tokens 100
```

## System KV Cache 说明

- **作用**：缓存 system prompt（工具描述等）的 KV，避免每次推理都重新计算
- **首次运行**：需要 system 模型生成 KV cache 并保存到磁盘
- **后续运行**：直接加载缓存的 KV，节省计算时间
- **可选性**：如果不使用 system KV cache，会从 prefill 阶段开始（past_len=0）

## 多轮对话支持

### 运行多轮对话

使用 `run_local_multiturn.py` 支持多轮对话和工具调用：

```bash
# 交互模式
python run_local_multiturn.py --system_onnx_dir ../onnx_models/system --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --interactive

# 批量问题模式
python run_local_multiturn.py --system_onnx_dir ../onnx_models/system --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --questions "北京天气怎么样？" "那上海呢？" "温度差多少？"
```

### 多轮对话流程

每个问题的处理分为两轮推理：

**第一轮推理**：
1. 如果有历史记忆，通过 system 模型处理历史对话（用户问题 + 助手回答）
2. 恢复到基础 KV 状态
3. Prefill + Decode：用户问题 → 工具调用

**第二轮推理**（如果有工具调用）：
1. Reset KV cache
2. System 模型处理：
   - 生成第二轮的初始 system KV（工具结果提示）
   - 如果有历史记忆，追加到 system KV
3. Prefill + Decode：工具结果 → 最终回答

### System 模型复用

**关键特性**：同一个 system 模型可以处理多种不同的输入：

1. **初始化阶段**：处理工具描述
   ```
   你是AI助手，可用工具：
   - get_weather(city*:string) — Get weather information for a city
   ...
   ```

2. **第一轮推理前**：处理历史对话（用户问题 + 助手回答）
   ```
   ## 历史对话 1
   用户: 北京天气怎么样？
   助手: 根据查询结果，北京今天是晴天...
   ```

3. **第二轮推理前**：处理工具结果提示
   ```
   你是一个AI助手。用户问了一个问题，你调用了工具获取了信息。
   以下是历史对话信息，可以帮助你更好地理解上下文：
   ```

**优势**：
- 无需导出多个 system 模型
- 灵活处理不同的 system prompt
- 节省模型存储空间

### 详细设计文档

参考 `MULTITURN_DESIGN_V2.md` 了解：
- 完整的三轮对话示例
- 每个阶段的实际 token 文本
- KV cache 的累积增长
- 模型的上下文理解能力

## 与其他实现的对比

| 特性 | file_pipeline | code | onnx_local |
|------|--------------|------|------------|
| 模型类型 | ONNX | OM | ONNX |
| 执行设备 | CPU | NPU | CPU |
| 分布式 | 否 | 是 | 否 |
| 工具调用 | 否 | 是 | 是 |
| 网络通信 | 否 | 是 | 否 |
| 多轮对话 | 否 | 是 | 是 |
| System 模型复用 | 否 | 否 | 是 |
