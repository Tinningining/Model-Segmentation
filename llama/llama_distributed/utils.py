"""
工具函数模块
"""
import numpy as np


def build_attention_mask(
    past_len: int,
    q_len: int,
    max_cache_len: int,
    max_input_len: int
) -> np.ndarray:
    """
    构建注意力掩码
    
    Args:
        past_len: 过去的序列长度
        q_len: 查询长度
        max_cache_len: 最大缓存长度
        max_input_len: 最大输入长度
        
    Returns:
        attention_mask [1, max_input_len, max_cache_len]
    """
    # 创建掩码矩阵
    mask = np.zeros((1, max_input_len, max_cache_len), dtype=np.float32)
    
    # 填充有效区域
    total_len = past_len + q_len
    for i in range(q_len):
        current_pos = past_len + i
        # 当前位置可以看到之前的所有位置
        mask[0, i, :current_pos + 1] = 1.0
    
    return mask


def build_position_ids(
    past_len: int,
    q_len: int,
    max_input_len: int
) -> np.ndarray:
    """
    构建位置 ID
    
    Args:
        past_len: 过去的序列长度
        q_len: 查询长度
        max_input_len: 最大输入长度
        
    Returns:
        position_ids [1, max_input_len]
    """
    pos_ids = np.zeros((1, max_input_len), dtype=np.int64)
    
    # 填充位置 ID
    for i in range(q_len):
        pos_ids[0, i] = past_len + i
    
    return pos_ids


def pad_input_ids(
    input_ids: np.ndarray,
    max_len: int,
    pad_id: int = 0
) -> np.ndarray:
    """
    填充输入 ID 到固定长度
    
    Args:
        input_ids: 输入 ID [1, seq_len]
        max_len: 最大长度
        pad_id: 填充 ID
        
    Returns:
        padded_ids [1, max_len]
    """
    batch_size, seq_len = input_ids.shape
    
    if seq_len >= max_len:
        return input_ids[:, :max_len]
    
    # 创建填充后的数组
    padded = np.full((batch_size, max_len), pad_id, dtype=input_ids.dtype)
    padded[:, :seq_len] = input_ids
    
    return padded


def reshape_hidden_output(
    output: np.ndarray,
    batch_size: int,
    max_input_len: int,
    hidden_size: int
) -> np.ndarray:
    """
    重塑隐藏状态输出
    
    Args:
        output: 模型输出
        batch_size: 批大小
        max_input_len: 最大输入长度
        hidden_size: 隐藏层大小
        
    Returns:
        reshaped [batch_size, max_input_len, hidden_size]
    """
    # 将输出重塑为正确的形状
    flat = output.view(np.float32).reshape(-1)
    target_size = batch_size * max_input_len * hidden_size
    
    if flat.shape[0] >= target_size:
        return flat[:target_size].reshape(batch_size, max_input_len, hidden_size)
    else:
        # 如果输出不够，填充零
        padded = np.zeros(target_size, dtype=np.float32)
        padded[:flat.shape[0]] = flat
        return padded.reshape(batch_size, max_input_len, hidden_size)


def reshape_kv_output(
    output: np.ndarray,
    num_layers: int,
    batch_size: int,
    num_kv_heads: int,
    max_input_len: int,
    head_dim: int
) -> np.ndarray:
    """
    重塑 KV 输出
    
    Args:
        output: 模型输出
        num_layers: 层数
        batch_size: 批大小
        num_kv_heads: KV 头数
        max_input_len: 最大输入长度
        head_dim: 头维度
        
    Returns:
        reshaped [num_layers, batch_size, num_kv_heads, max_input_len, head_dim]
    """
    target_shape = (num_layers, batch_size, num_kv_heads, max_input_len, head_dim)
    num_elements = np.prod(target_shape)
    
    flat = output.view(np.float32).reshape(-1)
    
    if flat.shape[0] >= num_elements:
        return flat[:num_elements].reshape(target_shape)
    else:
        # 如果输出不够，填充零
        padded = np.zeros(num_elements, dtype=np.float32)
        padded[:flat.shape[0]] = flat
        return padded.reshape(target_shape)


def sample_logits(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = True
) -> int:
    """
    从 logits 中采样下一个 token
    
    Args:
        logits: logits [vocab_size]
        temperature: 温度
        top_k: top-k 采样
        top_p: top-p 采样
        greedy: 是否贪婪采样
        
    Returns:
        next_token
    """
    if greedy or temperature == 0:
        return int(np.argmax(logits))
    
    # 应用温度
    logits = logits.astype(np.float32) / temperature
    
    # 计算概率
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    # Top-k 采样
    if top_k > 0:
        top_k_indices = np.argsort(probs)[-top_k:]
        top_k_probs = probs[top_k_indices]
        top_k_probs = top_k_probs / np.sum(top_k_probs)
        return int(np.random.choice(top_k_indices, p=top_k_probs))
    
    # Top-p 采样
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)
        
        # 找到累积概率超过 top_p 的位置
        cutoff_index = np.searchsorted(cumsum_probs, top_p)
        
        # 选择前 cutoff_index 个 token
        top_p_indices = sorted_indices[:cutoff_index + 1]
        top_p_probs = probs[top_p_indices]
        top_p_probs = top_p_probs / np.sum(top_p_probs)
        
        return int(np.random.choice(top_p_indices, p=top_p_probs))
    
    # 普通采样
    return int(np.random.choice(len(probs), p=probs))
