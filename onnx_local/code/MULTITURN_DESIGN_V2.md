# 多轮对话 ONNX 执行设计文档 V2（Prefill 模型统一处理）

## 概述

`run_local_multiturn.py` 实现了支持多轮对话和工具调用的单机 ONNX 模型执行系统。**关键特性**：
- 统一使用 prefill 模型处理所有输入（工具描述、历史记忆、用户问题等）
- 避免了 system 模型无 past KV 的限制
- 灵活的 past KV 机制支持历史记忆追加
- 输出格式提示和 `/no_think` 放在最后，提高输出准确性

## 核心设计理念

### 1. KV Cache 管理策略

系统维护两个关键的 KV cache 快照：

- **base_kv_snapshot**: 当前对话状态的基础 KV
  - 初始时包含工具描述的 prefill 处理结果
  - 每次新问题开始前，如果有历史记忆，会通过 prefill 模型追加记忆信息
  
- **question_kv_snapshot**: 当前问题第一轮推理后的 KV
  - 保存用户问题输入后、工具调用前的状态
  - 用于作为下一个问题的 base_kv_snapshot

### 2. 多轮对话流程

每个问题的处理分为以下步骤：

```
第一轮推理：
1. 如果有历史记忆 → 通过 prefill 模型处理（工具描述 + 历史记忆）→ 更新 base_kv_snapshot
2. 恢复到 base_kv_snapshot
3. Prefill + Decode: 用户问题 + 输出格式提示 + /no_think → 工具调用
4. 保存 question_kv_snapshot

第二轮推理（如果有工具调用）：
5. Reset KV cache
6. Prefill 模型处理：
   a. 生成第二轮的初始 KV（工具结果提示 + 历史记忆）
7. Prefill + Decode: 工具结果 + 输出格式提示 + /no_think → 最终回答
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

记忆文本格式化为：

```
## 历史对话 1
用户: 北京天气怎么样？
助手: 根据查询结果，北京今天是晴天，温度25°C，湿度60%。

## 历史对话 2
用户: 那上海呢？
助手: 上海今天是多云天气，温度22°C，湿度70%。
```

## 完整示例：Prompt 样式详解

### 示例场景

用户连续提问三个问题：
1. "北京天气怎么样？"
2. "那上海呢？"
3. "两个城市温度差多少？"

---

## 第一次提问

### 第一轮推理

#### 初始化阶段（Prefill 模型处理工具描述）

**输入文本**（使用 `build_tool_system_prompt`）:
```
你是AI助手，可用工具：
- get_weather(city*:string) — Get weather information for a city
- calculator(expression*:string) — Perform mathematical calculations
- get_time() — Get current time
- unit_convert(value*:number, from_unit*:string, to_unit*:string) — Convert units
- translate(text*:string, target_lang*:string) — Translate text

注意：上面提供了历史对话记录。
- 如果用户问题可以直接从历史对话中找到答案，请直接回答，无需调用工具。
- 如果需要新信息或计算，请调用相应工具。
```

**处理方式**: Prefill 模型处理（无 past KV，初始化）
**生成的 KV Cache**: base_kv_snapshot（past_len ≈ 80 tokens）

#### 用户问题阶段（Prefill + Decode）

**输入文本**（基于 base_kv_snapshot）:
```
<|im_start|>user
北京天气怎么样？

调用工具时输出JSON：{"tool_name":"名称","arguments":{"参数":"值"}}
可一次调用多个，每行一个JSON。不需要工具则直接回答。
/no_think<|im_end|>
<|im_start|>assistant
```

**处理方式**: Prefill 模型处理用户问题（带 past KV），然后 Decode 逐 token 生成

**模型输出**:
```json
{"tool_name": "get_weather", "arguments": {"city": "Beijing"}}
```

**生成的 KV Cache**: question_kv_snapshot（past_len ≈ 95 tokens）

**工具执行**: get_weather(Beijing) → "北京今天晴天，温度25°C，湿度60%"

### 第二轮推理

#### 工具结果处理阶段（Prefill 模型处理）

**输入文本**（使用 `build_round2_system_prompt` 和 `build_tool_result_prompt`）:
```
你是一个AI助手。用户问了一个问题，你调用了工具获取了信息。

用户问题：北京天气怎么样？

工具调用1：get_weather
  参数：{"city": "Beijing"}
  返回：{"success": true, "result": "北京今天晴天，温度25°C，湿度60%", "tool_name": "get_weather"}

请基于工具返回的真实数据，用自然语言回答用户的问题。回答要具体、准确、友好。
```

**处理方式**: Prefill 模型处理（无 past KV，重新初始化）
**生成的 KV Cache**: round2_base_snapshot（past_len ≈ 30 tokens）

#### 最终回答阶段（Prefill + Decode）

**输入文本**（基于 round2_base_snapshot）:
```
<|im_start|>user
北京天气怎么样？

如果还需要调用工具，输出JSON：{"tool_name":"名称","arguments":{"参数":"值"}}
否则直接用自然语言回答用户的问题。
/no_think<|im_end|>
<|im_start|>assistant
```

**处理方式**: Prefill 模型处理用户问题（带 past KV），然后 Decode 逐 token 生成

**模型输出**:
```
根据查询结果，北京今天是晴天，温度25°C，湿度60%。天气不错，适合外出活动。
```

**对话记忆存储**:
```python
{
    'question': '北京天气怎么样？',
    'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'Beijing'}, 'id': '...'}],
    'tool_results': [{'success': True, 'result': '北京今天晴天，温度25°C，湿度60%', 'tool_name': 'get_weather'}],
    'final_answer': '根据查询结果，北京今天是晴天，温度25°C，湿度60%。天气不错，适合外出活动。'
}
```

**更新 base_kv_snapshot**: question_kv_snapshot → base_kv_snapshot（past_len ≈ 95 tokens）

---

## 第二次提问

### 第一轮推理

#### 历史记忆处理阶段（Prefill 模型处理）

**输入文本**（工具描述 + 历史记忆）:
```
你是AI助手，可用工具：
- get_weather(city*:string) — Get weather information for a city
- calculator(expression*:string) — Perform mathematical calculations
- get_time() — Get current time
- unit_convert(value*:number, from_unit*:string, to_unit*:string) — Convert units
- translate(text*:string, target_lang*:string) — Translate text

注意：上面提供了历史对话记录。
- 如果用户问题可以直接从历史对话中找到答案，请直接回答，无需调用工具。
- 如果需要新信息或计算，请调用相应工具。

## 历史对话 1
用户: 北京天气怎么样？
助手: 根据查询结果，北京今天是晴天，温度25°C，湿度60%。天气不错，适合外出活动。
```

**处理方式**: Prefill 模型处理（无 past KV，初始化）
**生成的 KV Cache**: 新的 base_kv_snapshot（past_len ≈ 130 tokens）
- 包含：工具描述（80 tokens）+ 第一次对话记忆（50 tokens）

#### 用户问题阶段（Prefill + Decode）

**输入文本**（基于新的 base_kv_snapshot）:
```
<|im_start|>user
那上海呢？

调用工具时输出JSON：{"tool_name":"名称","arguments":{"参数":"值"}}
可一次调用多个，每行一个JSON。不需要工具则直接回答。
/no_think<|im_end|>
<|im_start|>assistant
```

**处理方式**: Prefill 模型处理用户问题（带 past KV），然后 Decode 逐 token 生成

**模型输出**（模型理解"那上海呢"指的是上海天气）:
```json
{"tool_name": "get_weather", "arguments": {"city": "Shanghai"}}
```

**生成的 KV Cache**: question_kv_snapshot（past_len ≈ 142 tokens）

**工具执行**: get_weather(Shanghai) → "上海今天多云，温度22°C，湿度70%"

### 第二轮推理

#### 工具结果处理阶段（Prefill 模型处理）

**输入文本**（工具结果提示 + 历史记忆）:
```
你是一个AI助手。用户问了一个问题，你调用了工具获取了信息。以下是历史对话信息，可以帮助你更好地理解上下文：

## 历史对话 1
用户: 北京天气怎么样？
助手: 根据查询结果，北京今天是晴天，温度25°C，湿度60%。天气不错，适合外出活动。

用户问题：那上海呢？

工具调用1：get_weather
  参数：{"city": "Shanghai"}
  返回：{"success": true, "result": "上海今天多云，温度22°C，湿度70%", "tool_name": "get_weather"}

请基于工具返回的真实数据，用自然语言回答用户的问题。回答要具体、准确、友好。
```

**处理方式**: Prefill 模型处理（无 past KV，重新初始化）
**生成的 KV Cache**: round2_base_snapshot（past_len ≈ 80 tokens）

#### 最终回答阶段（Prefill + Decode）

**输入文本**（基于 round2_base_snapshot）:
```
<|im_start|>user
那上海呢？

如果还需要调用工具，输出JSON：{"tool_name":"名称","arguments":{"参数":"值"}}
否则直接用自然语言回答用户的问题。
/no_think<|im_end|>
<|im_start|>assistant
```

**处理方式**: Prefill 模型处理用户问题（带 past KV），然后 Decode 逐 token 生成

**模型输出**:
```
上海今天是多云天气，温度22°C，湿度70%。相比北京，上海温度稍低一些。
```

**对话记忆存储**:
```python
{
    'question': '那上海呢？',
    'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'Shanghai'}, 'id': '...'}],
    'tool_results': [{'success': True, 'result': '上海今天多云，温度22°C，湿度70%', 'tool_name': 'get_weather'}],
    'final_answer': '上海今天是多云天气，温度22°C，湿度70%。相比北京，上海温度稍低一些。'
}
```

**更新 base_kv_snapshot**: question_kv_snapshot → base_kv_snapshot（past_len ≈ 142 tokens）

---

## 第三次提问

### 第一轮推理

#### 历史记忆处理阶段（Prefill 模型处理）

**输入文本**（工具描述 + 两次历史记忆）:
```
你是AI助手，可用工具：
- get_weather(city*:string) — Get weather information for a city
- calculator(expression*:string) — Perform mathematical calculations
- get_time() — Get current time
- unit_convert(value*:number, from_unit*:string, to_unit*:string) — Convert units
- translate(text*:string, target_lang*:string) — Translate text

注意：上面提供了历史对话记录。
- 如果用户问题可以直接从历史对话中找到答案，请直接回答，无需调用工具。
- 如果需要新信息或计算，请调用相应工具。

## 历史对话 1
用户: 北京天气怎么样？
助手: 根据查询结果，北京今天是晴天，温度25°C，湿度60%。天气不错，适合外出活动。

## 历史对话 2
用户: 那上海呢？
助手: 上海今天是多云天气，温度22°C，湿度70%。相比北京，上海温度稍低一些。
```

**处理方式**: Prefill 模型处理（无 past KV，初始化）
**生成的 KV Cache**: 新的 base_kv_snapshot（past_len ≈ 200 tokens）
- 包含：工具描述（80 tokens）+ 第一次对话记忆（50 tokens）+ 第二次对话记忆（70 tokens）

#### 用户问题阶段（Prefill + Decode）

**输入文本**（基于新的 base_kv_snapshot）:
```
<|im_start|>user
两个城市温度差多少？

调用工具时输出JSON：{"tool_name":"名称","arguments":{"参数":"值"}}
可一次调用多个，每行一个JSON。不需要工具则直接回答。
/no_think<|im_end|>
<|im_start|>assistant
```

**处理方式**: Prefill 模型处理用户问题（带 past KV），然后 Decode 逐 token 生成

**模型输出**（模型理解"两个城市"指北京和上海，且已知温度）:
```json
{"tool_name": "calculator", "arguments": {"expression": "25 - 22"}}
```

**生成的 KV Cache**: question_kv_snapshot（past_len ≈ 215 tokens）

**工具执行**: calculator(25-22) → "3"

### 第二轮推理

#### 工具结果处理阶段（Prefill 模型处理）

**输入文本**（工具结果提示 + 两次历史记忆）:
```
你是一个AI助手。用户问了一个问题，你调用了工具获取了信息。以下是历史对话信息，可以帮助你更好地理解上下文：

## 历史对话 1
用户: 北京天气怎么样？
助手: 根据查询结果，北京今天是晴天，温度25°C，湿度60%。天气不错，适合外出活动。

## 历史对话 2
用户: 那上海呢？
助手: 上海今天是多云天气，温度22°C，湿度70%。相比北京，上海温度稍低一些。

用户问题：两个城市温度差多少？

工具调用1：calculator
  参数：{"expression": "25 - 22"}
  返回：{"success": true, "result": "3", "tool_name": "calculator"}

请基于工具返回的真实数据，用自然语言回答用户的问题。回答要具体、准确、友好。
```

**处理方式**: Prefill 模型处理（无 past KV，重新初始化）
**生成的 KV Cache**: round2_base_snapshot（past_len ≈ 150 tokens）

#### 最终回答阶段（Prefill + Decode）

**输入文本**（基于 round2_base_snapshot）:
```
<|im_start|>user
两个城市温度差多少？

如果还需要调用工具，输出JSON：{"tool_name":"名称","arguments":{"参数":"值"}}
否则直接用自然语言回答用户的问题。
/no_think<|im_end|>
<|im_start|>assistant
```

**处理方式**: Prefill 模型处理用户问题（带 past KV），然后 Decode 逐 token 生成

**模型输出**:
```
根据之前查询的结果，北京温度是25°C，上海温度是22°C，两个城市的温度差是3°C。北京比上海高3度。
```

---

## 关键观察

### 1. Prefill 模型处理的累积增长

**第一轮推理前的 base_kv**:
- 第1次: 80 tokens（工具描述）
- 第2次: 130 tokens（工具描述 + 记忆1）
- 第3次: 200 tokens（工具描述 + 记忆1 + 记忆2）

**第二轮推理前的 round2_base_kv**:
- 第1次: 30 tokens（工具结果提示）
- 第2次: 80 tokens（工具结果提示 + 记忆1）
- 第3次: 150 tokens（工具结果提示 + 记忆1 + 记忆2）

### 2. 模型的上下文理解

- 第2次提问："那上海呢？" → 模型理解是问上海天气（基于历史记忆）
- 第3次提问："两个城市温度差多少？" → 模型知道是北京和上海，且已知温度（基于历史记忆）

### 3. 输出格式提示的位置

**关键改进**：输出格式提示和 `/no_think` 放在最后面
- 第一轮推理：在用户问题后添加
- 第二轮推理：在用户问题后添加
- 这样做的好处：
  - 格式提示始终在最后，更有利于 AI 按照格式输出
  - `/no_think` 在最后，避免模型产生思考过程
  - 提高了输出的准确性和一致性

### 4. Prefill 模型统一处理的优势

相比 system 模型：
- **避免了 past KV 限制**: Prefill 模型总是支持 past KV，更灵活
- **统一的处理流程**: 所有输入都用 prefill 模型处理，逻辑更清晰
- **更好的上下文利用**: 历史记忆可以直接追加到 KV cache 中

## 使用示例

### 交互模式

```bash
python run_local_multiturn.py \
    --prefill_onnx_dir ../onnx_models/prefill \
    --decode_onnx_dir ../onnx_models/decode \
    --tokenizer_dir D:\qwen_split\qwen3_1.7b \
    --interactive
```

### 批量问题模式

```bash
python run_local_multiturn.py \
    --prefill_onnx_dir ../onnx_models/prefill \
    --decode_onnx_dir ../onnx_models/decode \
    --tokenizer_dir D:\qwen_split\qwen3_1.7b \
    --questions "北京天气怎么样？" "那上海呢？" "温度差多少？"
```

## 性能考虑

### 优点

1. **真正的多轮对话**: 模型能够理解上下文
2. **统一的处理流程**: 所有输入都用 prefill 模型处理
3. **灵活的历史记忆**: 支持任意长度的历史记忆追加
4. **简化的记忆格式**: 减少 token 消耗

### 注意事项

1. **Prefill 模型调用次数**:
   - 每次新问题: 1次（处理历史记忆和工具描述）
   - 如果有工具调用: 再1次（处理工具结果）
   - 总计: 每个问题最多2次 prefill 调用

2. **KV Cache 容量**:
   - 需要监控 past_len 避免超过 max_cache_len
   - 建议实现记忆压缩或选择性遗忘

3. **记忆管理**:
   - 可以限制记忆条目数量
   - 可以压缩或总结历史对话

## 总结

V2 版本通过统一使用 prefill 模型处理所有输入（工具描述、历史记忆、工具结果提示），实现了更强大的多轮对话能力。将输出格式提示和 `/no_think` 放在最后面，提高了模型的输出准确性。记忆格式的简化（用户问题 + 助手回答）使得系统更高效，同时保持了良好的上下文理解能力。
