# 多轮对话 ONNX 执行设计文档 V2

## 概述

`run_local_multiturn.py` 实现了支持多轮对话和工具调用的单机 ONNX 模型执行系统。**关键特性**：
- 第一轮推理前：通过 system 模型处理历史对话（用户问题 + 助手回答）
- 第二轮推理前：也通过 system 模型处理历史对话
- 记忆格式简化为：用户问题 + 助手最终回答

## 核心设计理念

### 1. KV Cache 管理策略

系统维护两个关键的 KV cache 快照：

- **base_kv_snapshot**: 当前对话状态的基础 KV
  - 初始时包含工具描述的 system prompt
  - 每次新问题开始前，如果有历史记忆，会通过 system 模型追加记忆信息
  
- **question_kv_snapshot**: 当前问题第一轮推理后的 KV
  - 保存用户问题输入后、工具调用前的状态
  - 用于作为下一个问题的 base_kv_snapshot

### 2. 多轮对话流程

每个问题的处理分为以下步骤：

```
第一轮推理：
1. 如果有历史记忆 → 通过 system 模型处理记忆 → 更新 base_kv_snapshot
2. 恢复到 base_kv_snapshot
3. Prefill + Decode: 用户问题 → 工具调用
4. 保存 question_kv_snapshot

第二轮推理（如果有工具调用）：
5. Reset KV cache
6. System 模型处理：
   a. 生成第二轮的初始 system KV（工具结果提示）
   b. 如果有历史记忆，追加到 system KV
7. Prefill + Decode: 工具结果 → 最终回答
8. 存储到对话记忆（包含最终回答）
9. 将 question_kv_snapshot 设为新的 base_kv_snapshot
```

### 3. 记忆格式

对话记忆以结构化格式存储：

```python
{
    'question': '用户问题',
    'tool_calls': [工具调用列表],
    'tool_results': [工具结果列表],
    'final_answer': '助手最终回答'
}
```

记忆文本格式化为（简化版）：

```
## 历史对话 1
用户: 北京天气怎么样？
助手: 根据查询结果，北京今天是晴天，温度25°C，湿度60%。

## 历史对话 2
用户: 那上海呢？
助手: 上海今天是多云天气，温度22°C，湿度70%。
```

## 完整示例：Token 文本详解

### 示例场景

用户连续提问三个问题：
1. "北京天气怎么样？"
2. "那上海呢？"
3. "两个城市温度差多少？"

---

### 第一次提问

#### 第一轮推理

**System 阶段（初始化）**

输入文本（使用 `build_tool_system_prompt`）:
```
你是AI助手，可用工具：
- get_weather(city*:string) — Get weather information for a city
- calculator(expression*:string) — Perform mathematical calculations
- get_time() — Get current time
- unit_convert(value*:number, from_unit*:string, to_unit*:string) — Convert units
- translate(text*:string, target_lang*:string) — Translate text

调用工具时输出JSON：{"tool_name":"名称","arguments":{"参数":"值"}}
可一次调用多个，每行一个JSON。不需要工具则直接回答。
/no_think
```

生成的 KV Cache: base_kv_snapshot（past_len ≈ 80 tokens）

**Prefill 阶段（用户问题）**

输入文本（基于 base_kv_snapshot）:
```
<|im_start|>user
北京天气怎么样？<|im_end|>
<|im_start|>assistant
```

模型输出:
```json
{"tool_name": "get_weather", "arguments": {"city": "Beijing"}}
```

生成的 KV Cache: question_kv_snapshot（past_len ≈ 95 tokens）

**工具执行**: get_weather(Beijing) → "北京今天晴天，温度25°C，湿度60%"

#### 第二轮推理

**System 阶段（第二轮初始化）**

输入文本:
```
你是一个AI助手。用户问了一个问题，你调用了工具获取了信息。以下是历史对话信息：
```

生成的 KV Cache: round2_base_snapshot（past_len ≈ 30 tokens）

**注意**: 第一次提问时没有历史记忆，所以不追加历史对话

**Prefill 阶段（工具结果）**

输入文本（使用 `build_tool_result_prompt` 和 `build_chat_prompt`）:
```
<|im_start|>system
你是一个AI助手。用户问了一个问题，你调用了工具获取了信息。

用户问题：北京天气怎么样？

工具调用1：get_weather
  参数：{"city": "Beijing"}
  返回：{"success": true, "result": "北京今天晴天，温度25°C，湿度60%", "tool_name": "get_weather"}

请基于工具返回的真实数据，用自然语言回答用户的问题。回答要具体、准确、友好。
/no_think<|im_end|>
<|im_start|>user
北京天气怎么样？<|im_end|>
<|im_start|>assistant
```

模型输出:
```
根据查询结果，北京今天是晴天，温度25°C，湿度60%。天气不错，适合外出活动。
```

**对话记忆存储**:
```python
{
    'question': '北京天气怎么样？',
    'tool_calls': [...],
    'tool_results': [...],
    'final_answer': '根据查询结果，北京今天是晴天，温度25°C，湿度60%。天气不错，适合外出活动。'
}
```

**更新 base_kv_snapshot**: question_kv_snapshot → base_kv_snapshot（past_len ≈ 95 tokens）

---

### 第二次提问

#### 第一轮推理

**System 阶段（追加历史记忆）**

输入文本（基于上一次的 base_kv_snapshot，追加历史）:
```
<|im_start|>system
## 历史对话 1
用户: 北京天气怎么样？
助手: 根据查询结果，北京今天是晴天，温度25°C，湿度60%。天气不错，适合外出活动。

<|im_end|>
```

生成的 KV Cache: 新的 base_kv_snapshot（past_len ≈ 130 tokens）
- 包含：工具描述（80 tokens）+ 第一次对话记忆（50 tokens）

**Prefill 阶段（用户问题）**

输入文本（基于新的 base_kv_snapshot）:
```
<|im_start|>user
那上海呢？<|im_end|>
<|im_start|>assistant
```

模型输出（模型理解"那上海呢"指的是上海天气）:
```json
{"tool_name": "get_weather", "arguments": {"city": "Shanghai"}}
```

生成的 KV Cache: question_kv_snapshot（past_len ≈ 142 tokens）

**工具执行**: get_weather(Shanghai) → "上海今天多云，温度22°C，湿度70%"

#### 第二轮推理

**System 阶段（第二轮初始化 + 追加历史）**

输入文本（初始化）:
```
你是一个AI助手。用户问了一个问题，你调用了工具获取了信息。以下是历史对话信息：
```

然后追加历史记忆:
```
<|im_start|>system
## 历史对话 1
用户: 北京天气怎么样？
助手: 根据查询结果，北京今天是晴天，温度25°C，湿度60%。天气不错，适合外出活动。

<|im_end|>
```

生成的 KV Cache: round2_base_snapshot（past_len ≈ 80 tokens）

**Prefill 阶段（工具结果）**

输入文本:
```
<|im_start|>system
你是一个AI助手。用户问了一个问题，你调用了工具获取了信息。

用户问题：那上海呢？

工具调用1：get_weather
  参数：{"city": "Shanghai"}
  返回：{"success": true, "result": "上海今天多云，温度22°C，湿度70%", "tool_name": "get_weather"}

请基于工具返回的真实数据，用自然语言回答用户的问题。回答要具体、准确、友好。
/no_think<|im_end|>
<|im_start|>user
那上海呢？<|im_end|>
<|im_start|>assistant
```

模型输出:
```
上海今天是多云天气，温度22°C，湿度70%。相比北京，上海温度稍低一些。
```

**对话记忆存储**:
```python
{
    'question': '那上海呢？',
    'tool_calls': [...],
    'tool_results': [...],
    'final_answer': '上海今天是多云天气，温度22°C，湿度70%。相比北京，上海温度稍低一些。'
}
```

**更新 base_kv_snapshot**: question_kv_snapshot → base_kv_snapshot（past_len ≈ 142 tokens）

---

### 第三次提问

#### 第一轮推理

**System 阶段（追加历史记忆）**

输入文本（基于上一次的 base_kv_snapshot，追加历史）:
```
<|im_start|>system
## 历史对话 1
用户: 北京天气怎么样？
助手: 根据查询结果，北京今天是晴天，温度25°C，湿度60%。天气不错，适合外出活动。

## 历史对话 2
用户: 那上海呢？
助手: 上海今天是多云天气，温度22°C，湿度70%。相比北京，上海温度稍低一些。

<|im_end|>
```

生成的 KV Cache: 新的 base_kv_snapshot（past_len ≈ 200 tokens）
- 包含：工具描述（80 tokens）+ 第一次对话记忆（50 tokens）+ 第二次对话记忆（70 tokens）

**Prefill 阶段（用户问题）**

输入文本（基于新的 base_kv_snapshot）:
```
<|im_start|>user
两个城市温度差多少？<|im_end|>
<|im_start|>assistant
```

模型输出（模型理解"两个城市"指北京和上海，且已知温度）:
```json
{"tool_name": "calculator", "arguments": {"expression": "25 - 22"}}
```

生成的 KV Cache: question_kv_snapshot（past_len ≈ 215 tokens）

**工具执行**: calculator(25-22) → "3"

#### 第二轮推理

**System 阶段（第二轮初始化 + 追加历史）**

输入文本（初始化）:
```
你是一个AI助手。用户问了一个问题，你调用了工具获取了信息。以下是历史对话信息：
```

然后追加历史记忆:
```
<|im_start|>system
## 历史对话 1
用户: 北京天气怎么样？
助手: 根据查询结果，北京今天是晴天，温度25°C，湿度60%。天气不错，适合外出活动。

## 历史对话 2
用户: 那上海呢？
助手: 上海今天是多云天气，温度22°C，湿度70%。相比北京，上海温度稍低一些。

<|im_end|>
```

生成的 KV Cache: round2_base_snapshot（past_len ≈ 150 tokens）

**Prefill 阶段（工具结果）**

输入文本:
```
<|im_start|>system
你是一个AI助手。用户问了一个问题，你调用了工具获取了信息。

用户问题：两个城市温度差多少？

工具调用1：calculator
  参数：{"expression": "25 - 22"}
  返回：{"success": true, "result": "3", "tool_name": "calculator"}

请基于工具返回的真实数据，用自然语言回答用户的问题。回答要具体、准确、友好。
/no_think<|im_end|>
<|im_start|>user
两个城市温度差多少？<|im_end|>
<|im_start|>assistant
```

模型输出:
```
根据之前查询的结果，北京温度是25°C，上海温度是22°C，两个城市的温度差是3°C。北京比上海高3度。
```

## 关键观察

### 1. System KV 的累积增长

**第一轮推理前的 base_kv**:
- 第1次: 80 tokens（工具描述）
- 第2次: 130 tokens（工具描述 + 记忆1）
- 第3次: 200 tokens（工具描述 + 记忆1 + 记忆2）

**第二轮推理前的 round2_base_kv**:
- 第1次: 30 tokens（工具结果提示）
- 第2次: 80 tokens（工具结果提示 + 记忆1）
- 第3次: 150 tokens（工具结果提示 + 记忆1 + 记忆2）

### 2. 模型的上下文理解

- 第2次提问："那上海呢？" → 模型理解是问上海天气
- 第3次提问："两个城市温度差多少？" → 模型知道是北京和上海，且已知温度

### 3. 两次 System 调用的作用

**第一轮推理前的 System**:
- 提供工具描述
- 提供历史对话上下文
- 让模型理解当前问题的背景

**第二轮推理前的 System**:
- 提供工具结果处理的指导
- 提供历史对话上下文
- 让模型基于历史和工具结果生成最终回答

### 4. 记忆格式的简化

相比之前的版本，新版本的记忆格式更简洁：
- **之前**: 包含工具调用详情和工具结果详情
- **现在**: 只包含用户问题和助手最终回答

这样做的好处：
- 减少 token 消耗
- 更符合自然对话的记忆方式
- 模型更容易理解和利用历史信息

## 使用示例

### 交互模式

```bash
python run_local_multiturn.py \
    --system_onnx_dir ../onnx_models/system \
    --prefill_onnx_dir ../onnx_models/prefill \
    --decode_onnx_dir ../onnx_models/decode \
    --tokenizer_dir D:\qwen_split\qwen3_1.7b \
    --interactive
```

### 批量问题模式

```bash
python run_local_multiturn.py \
    --system_onnx_dir ../onnx_models/system \
    --prefill_onnx_dir ../onnx_models/prefill \
    --decode_onnx_dir ../onnx_models/decode \
    --tokenizer_dir D:\qwen_split\qwen3_1.7b \
    --questions "北京天气怎么样？" "那上海呢？" "温度差多少？"
```

## 性能考虑

### 优点

1. **真正的多轮对话**: 模型能够理解上下文
2. **双重历史注入**: 第一轮和第二轮都能利用历史信息
3. **简化的记忆格式**: 减少 token 消耗

### 注意事项

1. **System 模型调用次数增加**:
   - 每次新问题: 1次（第一轮前）
   - 如果有工具调用: 再1次（第二轮前）
   - 总计: 每个问题最多2次 system 调用

2. **KV Cache 容量**:
   - 需要监控 past_len 避免超过 max_cache_len
   - 建议实现记忆压缩或选择性遗忘

3. **记忆管理**:
   - 可以限制记忆条目数量
   - 可以压缩或总结历史对话

## 总结

V2 版本通过在第一轮和第二轮推理前都使用 system 模型处理历史记忆，实现了更强大的多轮对话能力。记忆格式的简化（用户问题 + 助手回答）使得系统更高效，同时保持了良好的上下文理解能力。
