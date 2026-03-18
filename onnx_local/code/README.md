# ONNX Local Execution

单机运行 ONNX 模型的实现，借鉴 file_pipeline 的 ONNX 执行方式和 code 文件夹的工具调用能力。

## 特点

- 使用 ONNX Runtime 执行模型（CPU）
- 单机运行，无需网络通信
- 支持工具调用（继承自 code 文件夹）
- 支持 system/prefill/decode 三阶段推理
- KV cache 管理

## 文件结构

```
onnx_local/
├── README.md                 # 说明文档
├── run_local.py             # 主运行脚本
├── onnx_model.py            # ONNX 模型封装
├── kvcache.py               # KV cache 管理
├── config.py                # 配置
├── utils.py                 # 工具函数
└── tools/                   # 工具系统（从 code 复制）
    ├── __init__.py
    ├── tool_manager.py
    ├── tool_coordinator.py
    ├── tool_agent.py
    ├── streaming_parser.py
    ├── async_executor.py
    └── builtin_tools/
        ├── __init__.py
        ├── calculator_tool.py
        ├── weather_tool.py
        ├── time_tool.py
        ├── translate_tool.py
        └── unit_converter_tool.py
```

## 使用方法

### 基本用法（不使用 system KV cache）

```bash
python run_local.py --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --input_file user_prompt.txt --max_new_tokens 100
```

### 使用 system KV cache（推荐，可复用）

首次运行时需要提供 system 模型来生成 KV cache：

```bash
python run_local.py --system_onnx_dir ../onnx_models/system --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --system_kv_dir ./system_kv_cache --input_file user_prompt.txt --max_new_tokens 100
```

后续运行可以复用缓存的 system KV，无需再提供 system 模型：

```bash
python run_local.py --prefill_onnx_dir ../onnx_models/prefill --decode_onnx_dir ../onnx_models/decode --tokenizer_dir D:\qwen_split\qwen3_1.7b --system_kv_dir ./system_kv_cache --input_file user_prompt.txt --max_new_tokens 100
```

## System KV Cache 说明

- **作用**：缓存 system prompt（工具描述等）的 KV，避免每次推理都重新计算
- **首次运行**：需要 system 模型生成 KV cache 并保存到磁盘
- **后续运行**：直接加载缓存的 KV，节省计算时间
- **可选性**：如果不使用 system KV cache，会从 prefill 阶段开始（past_len=0）

## 与其他实现的对比

| 特性 | file_pipeline | code | onnx_local |
|------|--------------|------|------------|
| 模型类型 | ONNX | OM | ONNX |
| 执行设备 | CPU | NPU | CPU |
| 分布式 | 否 | 是 | 否 |
| 工具调用 | 否 | 是 | 是 |
| 网络通信 | 否 | 是 | 否 |
