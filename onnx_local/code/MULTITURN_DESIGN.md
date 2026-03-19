# 多轮对话 ONNX 执行设计文档

## 概述

`run_local_multiturn.py` 实现了支持多轮对话和工具调用的单机 ONNX 模型执行系统。每次提问都会调用 system 模型处理历史记忆，实现真正的多轮对话能力。

## 核心设计理念

### 1. KV Cache 管理策略

系统维护两个关键的 KV cache 快照：

- **base_kv_snapshot**: 当前对话状态的基础 KV
  - 初始时包含工具描述的 system prompt
  - 每次新问题开始前，如果有历史记忆，会通过 system 模型追加记忆信息
  
- **question_kv_snapshot**: 当前问题第一轮推理后的 KV
  - 保存用户问题输入后、工具调用前的状态
  - 用于第二轮推理时恢复到问题开始的状态

### 2. 多轮对话流程

每个问题的处理分为以下步骤：

```
1. 如果有历史记忆 → 通过 system 模型处理记忆 → 更新 base_kv_snapshot
2. 恢复到 base_kv_snapshot
3. 第一轮推理：用户问题 → 工具调用
4. 保存 question_kv_snapshot
5. 如果有工具调用：
   a. 恢复到 question_kv_snapshot
   b. 第二轮推理：工具结果 → 最终回答
6. 存储到对话记忆
7. 将 question_kv_snapshot 设为新的 base_kv_snapshot
```

### 3. 记忆格式

对话记忆以结构化格式存储：

```python
{
    'question': '用户问题',
    'tool_calls': [工具调用列表],
    'tool_results': [工具结果列表]
}
```

记忆文本格式化为：

```
## Previous Question 1
User: 北京天气怎么样？
Tool Calls:
  - get_weather: {'city': 'Beijing'}
Tool Results:
  - get_weather: 晴天，25°C

## Previous Question 2
...
```

## 与原版本的对比

### 原版本 (run_local.py)

- **单次对话**: 只能问一次问题
- **KV 管理**: 
  - 初始化时生成 system KV（工具描述）
  - 第一轮推理后 reset 到 system KV
  - 第二轮推理使用工具结果
- **限制**: 无法保留历史对话信息

### 多轮版本 (run_local_multiturn.py)

- **多轮对话**: 支持连续提问
- **KV 管理**:
  - 每次新问题前，通过 system 模型处理历史记忆
  - 第一轮推理后保存 question KV
  - 第二轮推理恢复到 question KV
  - 问题结束后，question KV 成为下次的 base KV
- **优势**: 模型能够"记住"之前的对话和工具调用结果

## 关键实现细节

### System 模型的两种使用模式

1. **初始模式** (`has_past_kv=False`):
   - 用于生成初始 system KV（工具描述）
   - 使用 `build_system_attention_mask`
   - 无 past KV 输入

2. **追加模式** (`has_past_kv=True`):
   - 用于追加历史记忆到现有 KV
   - 使用 `build_prefill_with_past_attention_mask`
   - 需要 past KV 输入

### KV Cache 快照机制

```python
# 保存快照
snapshot = self.kv_cache.save_snapshot()
# snapshot = {
#     'past_key': np.array(...),
#     'past_value': np.array(...),
#     'current_len': int
# }

# 恢复快照
self.kv_cache.restore_snapshot(snapshot)
self.past_len = snapshot['current_len']
```

### 模式切换逻辑

```python
# Prefill: 有 past KV 且输入多个 token
mode = "prefill" if self.past_len > 0 and q_len > 1 else "decode"
```

## 完整示例：Token 文本详解

为了更好地理解系统的工作原理，下面展示一个完整的多轮对话示例，包括每次 system 模型处理的完整 token 文本。

### 示例场景

用户连续提问三个问题：
1. "北京天气怎么样？"
2. "那上海呢？"
3. "两个城市温度差多少？"

### 第一次提问

#### System 阶段（初始化）

**输入文本**（通过 system 模型生成 base_kv_snapshot，使用 `build_tool_system_prompt`）:
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

**注意**: 这是紧凑格式的工具描述，相比标准 OpenAI 格式节省了大量 tokens。

**生成的 KV Cache**: base_kv_snapshot（past_len ≈ 200 tokens）

#### Prefill 阶段（用户问题）

**输入文本**（基于 base_kv_snapshot）:
```
<|im_start|>user
北京天气怎么样？<|im_end|>
<|im_start|>assistant
```

**模型输出**（第一轮推理）:
```
<tool_call>
{"name": "get_weather", "arguments": {"city": "Beijing"}}
</tool_call>
```

**生成的 KV Cache**: question_kv_snapshot（past_len ≈ 220 tokens）

#### 工具执行

工具调用结果：
```json
{
    "tool_name": "get_weather",
    "result": "北京今天晴天，温度25°C，湿度60%"
}
```

#### Prefill 阶段（工具结果，恢复到 question_kv_snapshot）

**输入文本**（使用 `build_tool_result_prompt` 和 `build_chat_prompt`）:
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

**模型输出**（第二轮推理）:
```
根据查询结果，北京今天是晴天，温度25°C，湿度60%。天气不错，适合外出活动。
```

**对话记忆存储**:
```python
{
    'question': '北京天气怎么样？',
    'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'Beijing'}, 'id': 'call_1'}],
    'tool_results': [{'tool_name': 'get_weather', 'result': '北京今天晴天，温度25°C，湿度60%'}]
}
```

**更新 base_kv_snapshot**: question_kv_snapshot → base_kv_snapshot（past_len ≈ 220 tokens）

---

### 第二次提问

#### System 阶段（追加历史记忆）

**输入文本**（基于上一次的 base_kv_snapshot，通过 system 模型追加）:
```
<|im_start|>system
## Previous Question 1
User: 北京天气怎么样？
Tool Calls:
  - get_weather: {'city': 'Beijing'}
Tool Results:
  - get_weather: 北京今天晴天，温度25°C，湿度60%

<|im_end|>
```

**生成的 KV Cache**: 新的 base_kv_snapshot（past_len ≈ 260 tokens）
- 包含：工具描述（200 tokens）+ 第一次对话记忆（60 tokens）

#### Prefill 阶段（用户问题）

**输入文本**（基于新的 base_kv_snapshot）:
```
<|im_start|>user
那上海呢？<|im_end|>
<|im_start|>assistant
```

**模型输出**（第一轮推理，模型理解"那上海呢"指的是上海天气）:
```
<tool_call>
{"name": "get_weather", "arguments": {"city": "Shanghai"}}
</tool_call>
```

**生成的 KV Cache**: question_kv_snapshot（past_len ≈ 275 tokens）

#### 工具执行

工具调用结果：
```json
{
    "tool_name": "get_weather",
    "result": "上海今天多云，温度22°C，湿度70%"
}
```

#### Prefill 阶段（工具结果，恢复到 question_kv_snapshot）

**输入文本**（使用 `build_tool_result_prompt` 和 `build_chat_prompt`）:
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

**模型输出**（第二轮推理）:
```
上海今天是多云天气，温度22°C，湿度70%。相比北京，上海温度稍低一些。
```

**对话记忆存储**:
```python
{
    'question': '那上海呢？',
    'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'Shanghai'}, 'id': 'call_2'}],
    'tool_results': [{'tool_name': 'get_weather', 'result': '上海今天多云，温度22°C，湿度70%'}]
}
```

**更新 base_kv_snapshot**: question_kv_snapshot → base_kv_snapshot（past_len ≈ 275 tokens）

---

### 第三次提问

#### System 阶段（追加历史记忆）

**输入文本**（基于上一次的 base_kv_snapshot，通过 system 模型追加）:
```
<|im_start|>system
## Previous Question 1
User: 北京天气怎么样？
Tool Calls:
  - get_weather: {'city': 'Beijing'}
Tool Results:
  - get_weather: 北京今天晴天，温度25°C，湿度60%

## Previous Question 2
User: 那上海呢？
Tool Calls:
  - get_weather: {'city': 'Shanghai'}
Tool Results:
  - get_weather: 上海今天多云，温度22°C，湿度70%

<|im_end|>
```

**生成的 KV Cache**: 新的 base_kv_snapshot（past_len ≈ 330 tokens）
- 包含：工具描述（200 tokens）+ 第一次对话记忆（60 tokens）+ 第二次对话记忆（70 tokens）

#### Prefill 阶段（用户问题）

**输入文本**（基于新的 base_kv_snapshot）:
```
<|im_start|>user
两个城市温度差多少？<|im_end|>
<|im_start|>assistant
```

**模型输出**（第一轮推理，模型理解"两个城市"指北京和上海，且已知温度）:
```
<tool_call>
{"name": "calculator", "arguments": {"expression": "25 - 22"}}
</tool_call>
```

**生成的 KV Cache**: question_kv_snapshot（past_len ≈ 350 tokens）

#### 工具执行

工具调用结果：
```json
{
    "tool_name": "calculator",
    "result": "3"
}
```

#### Prefill 阶段（工具结果，恢复到 question_kv_snapshot）

**输入文本**（使用 `build_tool_result_prompt` 和 `build_chat_prompt`）:
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

**模型输出**（第二轮推理）:
```
根据之前查询的结果，北京温度是25°C，上海温度是22°C，两个城市的温度差是3°C。北京比上海高3度。
```

### 关键观察

1. **System KV 的累积增长**:
   - 第1次: 200 tokens（工具描述）
   - 第2次: 260 tokens（工具描述 + 记忆1）
   - 第3次: 330 tokens（工具描述 + 记忆1 + 记忆2）

2. **模型的上下文理解**:
   - 第2次提问："那上海呢？" → 模型理解是问上海天气
   - 第3次提问："两个城市温度差多少？" → 模型知道是北京和上海，且已知温度

3. **KV Cache 的复用**:
   - 每次新问题都基于包含历史记忆的 base_kv_snapshot
   - 第一轮推理后保存 question_kv_snapshot
   - 第二轮推理恢复到 question_kv_snapshot

4. **记忆的作用**:
   - 使模型能够理解代词和省略（"那上海呢"、"两个城市"）
   - 避免重复查询已知信息（第3次直接用已知温度计算）

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

对话示例：
```
You: 北京天气怎么样？
Assistant: [调用 get_weather 工具] 北京今天晴天，温度25°C

You: 那上海呢？
Assistant: [模型记得之前问过天气] 上海今天多云，温度22°C

You: 两个城市温度差多少？
Assistant: [模型记得两次天气查询结果] 北京比上海高3°C
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
2. **灵活的记忆管理**: 可以选择性保留重要信息
3. **工具调用历史**: 避免重复调用相同工具

### 注意事项

1. **KV Cache 容量**: 
   - 每次追加记忆会增加 past_len
   - 需要监控 `past_len` 避免超过 `max_cache_len`
   - 可以实现记忆压缩或选择性遗忘

2. **System 模型调用开销**:
   - 每次新问题都需要调用 system 模型
   - 记忆文本越长，system 阶段耗时越长
   - 建议限制记忆条目数量或文本长度

3. **模型切换**:
   - System/Prefill/Decode 模型按需加载
   - 切换时会卸载当前模型，节省内存

## 扩展方向

### 1. 记忆压缩

```python
def _compress_memory(self):
    """压缩对话记忆，只保留最近 N 条或最重要的信息"""
    if len(self.conversation_memory) > MAX_MEMORY_ITEMS:
        # 保留最近的 N 条
        self.conversation_memory = self.conversation_memory[-MAX_MEMORY_ITEMS:]
```

### 2. 选择性记忆

```python
def _should_remember(self, tool_calls, tool_results):
    """判断是否需要记住这次对话"""
    # 例如：只记住有工具调用的对话
    return len(tool_calls) > 0
```

### 3. 记忆持久化

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

## 总结

多轮对话版本通过巧妙的 KV cache 管理和 system 模型的灵活使用，实现了真正的多轮对话能力。每次新问题都会通过 system 模型处理历史记忆，使模型能够理解上下文并做出更智能的回答。

这种设计在保持单机执行简洁性的同时，提供了强大的对话能力，适合需要多轮交互的应用场景。
