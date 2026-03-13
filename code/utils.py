"""
工具函数模块
"""
from typing import Tuple

import numpy as np

NEG_INF = -1e9


def load_tokenizer(tokenizer_dir: str):
    """
    使用 transformers.AutoTokenizer 加载 tokenizer。

    Args:
        tokenizer_dir: tokenizer 所在目录

    Returns:
        tokenizer 实例

    Raises:
        RuntimeError: 加载失败时抛出
    """
    if not tokenizer_dir:
        raise RuntimeError("Tokenizer directory is empty")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for tokenizer loading, please run: pip install transformers sentencepiece"
        ) from exc

    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_dir,
            trust_remote_code=True,
            use_fast=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer from: {tokenizer_dir}"
        ) from exc


def encode_text(tokenizer, text: str) -> list:
    """
    将输入文本编码为 token id 列表（基于 transformers.AutoTokenizer）。
    """
    if tokenizer is None:
        raise RuntimeError("Tokenizer is not loaded")

    if text is None:
        return []

    return tokenizer.encode(text, add_special_tokens=True)


def decode_token_ids(tokenizer, token_ids, skip_special_tokens: bool = True) -> str:
    """
    将 token id 列表解码为文本（基于 transformers.AutoTokenizer）。
    """
    if tokenizer is None or token_ids is None:
        return ""

    if isinstance(token_ids, np.ndarray):
        token_ids = token_ids.tolist()

    # tokenizers.Tokenizer 的 decode 方法参数名是 skip_special_tokens
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def decode_incremental_text(
    tokenizer,
    token_ids,
    previous_text: str = "",
    skip_special_tokens: bool = True,
) -> Tuple[str, str]:
    """
    增量解码文本，返回 (新增文本, 当前完整文本)。
    """
    full_text = decode_token_ids(
        tokenizer,
        token_ids,
        skip_special_tokens=skip_special_tokens,
    )

    if not previous_text:
        return full_text, full_text

    if full_text.startswith(previous_text):
        return full_text[len(previous_text):], full_text

    return full_text, full_text


def build_attention_mask(
    past_len: int,
    q_len: int,
    max_cache_len: int,
    max_input_len: int,
) -> np.ndarray:
    """
    构建静态 attention mask

    Args:
        past_len: 已有的 KV cache 长度
        q_len: 当前输入的序列长度
        max_cache_len: 最大缓存长度
        max_input_len: 最大输入长度

    Returns:
        attention_mask: shape (1, 1, max_input_len, max_cache_len + max_input_len)
    """
    if q_len > max_input_len:
        raise ValueError(f"q_len {q_len} exceeds max_input_len {max_input_len}")
    if past_len > max_cache_len:
        raise ValueError(f"past_len {past_len} exceeds max_cache_len {max_cache_len}")

    total = max_cache_len + max_input_len
    mask = np.full((max_input_len, total), NEG_INF, dtype=np.float32)

    if q_len == 0:
        return mask.reshape(1, 1, max_input_len, total)

    # 可以看到过去的 KV
    if past_len > 0:
        mask[:q_len, :past_len] = 0.0

    # 因果 mask：每个位置只能看到自己和之前的位置
    for row in range(q_len):
        cols_end = max_cache_len + row + 1
        mask[row, max_cache_len:cols_end] = 0.0

    return mask.reshape(1, 1, max_input_len, total)


def build_position_ids(
    past_len: int,
    q_len: int,
    max_input_len: int,
) -> np.ndarray:
    """
    构建静态 position ids

    Args:
        past_len: 已有的 KV cache 长度
        q_len: 当前输入的序列长度
        max_input_len: 最大输入长度

    Returns:
        position_ids: shape (1, max_input_len)
    """
    if q_len > max_input_len:
        raise ValueError(f"q_len {q_len} exceeds max_input_len {max_input_len}")

    pos = np.zeros((1, max_input_len), dtype=np.int64)

    if q_len > 0:
        pos[0, :q_len] = np.arange(past_len, past_len + q_len, dtype=np.int64)
        # 填充剩余位置
        if q_len < max_input_len:
            pos[0, q_len:] = pos[0, q_len - 1]

    return pos


def sample_logits(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
) -> int:
    """
    从 logits 中采样下一个 token

    Args:
        logits: shape (vocab_size,)
        temperature: 温度参数
        top_k: top-k 采样
        top_p: top-p (nucleus) 采样
        greedy: 是否贪婪采样

    Returns:
        next_token_id: 下一个 token 的 ID
    """
    logits = logits.astype(np.float64)

    # 贪婪采样
    if greedy or (top_k == 0 and top_p >= 1.0 and abs(temperature - 1.0) < 1e-6):
        return int(np.argmax(logits))

    # 应用温度
    logits = logits / max(temperature, 1e-5)

    # 计算概率
    probs = logits - np.max(logits)
    probs = np.exp(probs)
    probs = probs / np.sum(probs)

    working = probs.copy()

    # Top-K 采样
    if top_k > 0:
        idx = np.argpartition(-working, top_k - 1)[:top_k]
        mask = np.zeros_like(working)
        mask[idx] = working[idx]
        working = mask

    # Top-P 采样
    if top_p < 1.0:
        sorted_idx = np.argsort(-working)
        sorted_probs = working[sorted_idx]
        cumulative = np.cumsum(sorted_probs)
        cutoff = np.argmax(cumulative >= top_p)
        keep = sorted_idx[: cutoff + 1]
        mask = np.zeros_like(working)
        mask[keep] = working[keep]
        working = mask

    # 归一化
    working = working / np.sum(working)

    return int(np.random.choice(len(working), p=working))


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
) -> int:
    """
    从 logits 中采样下一个 token（支持 2D logits）

    Args:
        logits: shape (1, vocab_size) 或 (vocab_size,)
        temperature: 温度参数
        top_k: top-k 采样
        top_p: top-p (nucleus) 采样
        greedy: 是否贪婪采样

    Returns:
        next_token_id: 下一个 token 的 ID
    """
    # 如果是 2D，取最后一个位置
    if logits.ndim == 2:
        logits = logits[0]

    return sample_logits(logits, temperature, top_k, top_p, greedy)


# 别名
sample_next_token = sample_token


def reshape_hidden_output(
    raw_output: np.ndarray,
    batch_size: int,
    max_input_len: int,
    hidden_size: int,
) -> np.ndarray:
    """
    重塑 hidden states 输出
    """
    return raw_output.view(np.float32).reshape(batch_size, max_input_len, hidden_size)


def reshape_kv_output(
    raw_output: np.ndarray,
    num_layers: int,
    batch_size: int,
    num_kv_heads: int,
    q_len: int,
    head_dim: int,
) -> np.ndarray:
    """
    重塑 KV cache 输出
    """
    target_shape = (num_layers, batch_size, num_kv_heads, q_len, head_dim)
    num_elements = np.prod(target_shape)

    flat = raw_output.view(np.float32).reshape(-1)
    return flat[:num_elements].reshape(target_shape)


def reshape_logits_output(
    raw_output: np.ndarray,
    batch_size: int,
    max_input_len: int,
    vocab_size: int,
) -> np.ndarray:
    """
    重塑 logits 输出
    """
    return raw_output.view(np.float32).reshape(batch_size, max_input_len, vocab_size)


# ── Qwen Chat Prompt 构建 ──────────────────────────────────────────

def build_chat_prompt(system: str, user: str) -> str:
    """
    构建 Qwen 聊天格式的 prompt（im_start/im_end 模板）。

    Args:
        system: system prompt（含工具描述等）
        user: 用户消息

    Returns:
        完整的 prompt 字符串，以 assistant 开头等待模型续写
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_tool_system_prompt(tools: list) -> str:
    """
    根据工具列表构建 system prompt（紧凑格式，节省 token）。

    Args:
        tools: OpenAI function calling 格式的工具列表

    Returns:
        system prompt 字符串
    """
    # 紧凑格式：每个工具一行
    lines = []
    for t in tools:
        func = t.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {}).get("properties", {})
        required = func.get("parameters", {}).get("required", [])
        # 构建参数签名：name*(必填):type 或 name:type=default
        param_parts = []
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "str")
            star = "*" if pname in required else ""
            default = pinfo.get("default")
            if default is not None and not star:
                param_parts.append(f'{pname}:{ptype}="{default}"')
            else:
                param_parts.append(f"{pname}{star}:{ptype}")
        sig = ", ".join(param_parts)
        lines.append(f"- {name}({sig}) — {desc}")

    tool_list = "\n".join(lines)
    return (
        f"你是AI助手，可用工具：\n{tool_list}\n\n"
        f'调用工具时输出JSON：{{"tool_name":"名称","arguments":{{"参数":"值"}}}}\n'
        f"可一次调用多个，每行一个JSON。不需要工具则直接回答。\n"
        f"/no_think"
    )


def build_tool_result_prompt(user_message: str,
                              tool_calls: list,
                              tool_results: list,
                              tools: list = None) -> str:
    """
    构建工具结果注入后的 system prompt（第二轮推理）。

    Args:
        user_message: 原始用户问题
        tool_calls: 工具调用列表 [{"name": ..., "arguments": ...}, ...]
        tool_results: 工具结果列表 [{"success": ..., "result": ..., "tool_name": ...}, ...]
        tools: OpenAI 格式工具列表（可选，传入则允许模型继续调用工具）

    Returns:
        system prompt 字符串
    """
    import json

    # 构建工具调用与结果的描述
    call_descriptions = []
    for i, (call, result) in enumerate(zip(tool_calls, tool_results)):
        desc = (
            f"工具调用{i+1}：{call.get('name', '')}\n"
            f"  参数：{json.dumps(call.get('arguments', {}), ensure_ascii=False)}\n"
            f"  返回：{json.dumps(result, ensure_ascii=False)}"
        )
        call_descriptions.append(desc)

    calls_text = "\n\n".join(call_descriptions)

    prompt = (
        f"你是一个AI助手。用户问了一个问题，你调用了工具获取了信息。\n\n"
        f"用户问题：{user_message}\n\n"
        f"{calls_text}\n\n"
    )

    if tools:
        tool_desc = json.dumps(tools, ensure_ascii=False, indent=2)
        prompt += (
            f"你还可以使用以下工具：\n{tool_desc}\n\n"
            f"如果还需要调用更多工具，请输出JSON格式的工具调用。\n"
            f"如果信息已经足够，请直接用自然语言回答用户的问题。回答要具体、准确、友好。\n"
        )
    else:
        prompt += f"请基于工具返回的真实数据，用自然语言回答用户的问题。回答要具体、准确、友好。\n"

    prompt += "/no_think"
    return prompt


def build_tools_openai_schema(tool_configs: dict) -> list:
    """
    将内部工具配置转换为 OpenAI function calling 格式。

    Args:
        tool_configs: {tool_name: TOOL_CONFIG, ...}

    Returns:
        OpenAI 格式的工具列表
    """
    tools = []
    for name, config in tool_configs.items():
        params = config.get('parameters', {})
        # 转换为 JSON Schema 格式
        properties = {}
        required = []
        for param_name, param_info in params.items():
            prop = {"type": param_info.get("type", "string")}
            if "description" in param_info:
                prop["description"] = param_info["description"]
            if "enum" in param_info:
                prop["enum"] = param_info["enum"]
            if "default" in param_info:
                prop["default"] = param_info["default"]
            properties[param_name] = prop
            if param_info.get("required", False):
                required.append(param_name)

        tools.append({
            "type": "function",
            "function": {
                "name": config.get("name", name),
                "description": config.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        })
    return tools


def pad_input_ids(
    input_ids: np.ndarray,
    max_input_len: int,
    pad_id: int = 0,
) -> np.ndarray:
    """
    填充 input_ids 到固定长度

    Args:
        input_ids: shape (1, q_len)
        max_input_len: 目标长度
        pad_id: 填充 ID

    Returns:
        padded: shape (1, max_input_len)
    """
    q_len = input_ids.shape[1]
    if q_len >= max_input_len:
        return input_ids[:, :max_input_len]

    padded = np.full((1, max_input_len), pad_id, dtype=np.float32)
    padded[:, :q_len] = input_ids.astype(np.float32)
    return padded
