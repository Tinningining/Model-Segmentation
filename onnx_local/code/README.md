# ONNX Local Execution

单机运行 ONNX 模型的实现，借鉴 file_pipeline 的 ONNX 执行方式和 code 文件夹的工具调用能力。

## 特点

- 使用 ONNX Runtime 执行模型（CPU）
- 单机运行，无需网络通信
- 支持工具调用（继承自 code 文件夹）
- 支持 prefill/decode 两阶段推理（统一使用 prefill 模型处理所有输入）
- KV cache 管理
- **支持多轮对话**：每轮推理都通过 prefill 模型处理历史记忆
- **灵活的历史记忆处理**：prefill 模型可处理工具描述、历史记忆、工具结果提示等

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

使用 `run_local_multiturn.py` 支持多轮对话和工具调用（仅需 prefill 和 decode 模型）：

```bash
# 交互模式
python run_local_multiturn.py --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --interactive

# 批量问题模式
python run_local_multiturn.py --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --questions "北京天气怎么样？" "那上海呢？" "温度差多少？"
```

### 多轮对话流程

每个问题的处理分为两轮推理（使用 prefill 模型处理所有输入）：

**第一次提问**：
1. 初始化：prefill 模型处理工具描述 → 生成 base_kv_snapshot
2. 恢复到 base KV
3. 第一轮推理：Prefill + Decode → 用户问题 → 工具调用
4. 保存 question_kv_snapshot

**第二次及之后提问**：
1. 如果有历史记忆：prefill 模型处理（工具描述 + 历史记忆）→ 更新 base_kv_snapshot
2. 恢复到 base KV
3. 第一轮推理：Prefill + Decode → 用户问题 → 工具调用
4. 第二轮推理：prefill 模型处理（工具结果提示 + 历史记忆）→ Prefill + Decode → 最终回答
5. 存储到对话记忆

### Prefill 模型统一处理

**架构特点**：使用 prefill 模型处理所有输入（工具描述、历史记忆、用户问题等）：

1. **初始化阶段**：prefill 模型处理工具描述
   ```
   你是AI助手，可用工具：
   - get_weather(city*:string) — Get weather information for a city
   ...
   ```

2. **多轮对话中**：prefill 模型处理历史记忆
   ```
   工具描述...
   
   ## 历史对话 1
   用户: 北京天气怎么样？
   助手: 根据查询结果，北京今天是晴天...
   ```

3. **工具结果处理**：prefill 模型处理工具结果提示 + 历史记忆
   ```
   你是一个AI助手。用户问了一个问题，你调用了工具获取了信息。
   以下是历史对话信息...
   ```

**优势**：
- 统一使用 prefill 模型，避免 system 模型无 past KV 的限制
- 灵活的 past KV 机制支持历史记忆追加
- 简洁的架构设计

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
| Prefill 模型统一处理 | 否 | 否 | 是 |
