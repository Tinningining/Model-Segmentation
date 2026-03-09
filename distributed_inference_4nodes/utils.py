"""
工具函数模块
"""
import numpy as np
from typing import Tuple

NEG_INF = -1e9


def build_attention_mask(
    past_len: int,
    q_len: int,
    max_cache_len: int,
    max_input_len: int
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
    max_input_len: int
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
    greedy: bool = False
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
        keep = sorted_idx[:cutoff + 1]
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
    greedy: bool = False
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
    hidden_size: int
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
    head_dim: int
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
    vocab_size: int
) -> np.ndarray:
    """
    重塑 logits 输出
    """
    return raw_output.view(np.float32).reshape(batch_size, max_input_len, vocab_size)


def pad_input_ids(
    input_ids: np.ndarray,
    max_input_len: int,
    pad_id: int = 0
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
