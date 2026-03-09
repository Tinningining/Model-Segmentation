"""
KV Cache 管理模块
"""
import numpy as np
from typing import Tuple


class KVCache:
    """KV Cache 管理类"""
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_cache_len: int,
        dtype=np.float16
    ):
        """
        初始化 KV Cache
        
        Args:
            num_layers: 层数
            num_kv_heads: KV 头数
            head_dim: 头维度
            max_cache_len: 最大缓存长度
            dtype: 数据类型
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_cache_len = max_cache_len
        self.dtype = dtype
        
        # 初始化缓存
        self.reset()
    
    def reset(self):
        """重置缓存"""
        shape = (
            self.num_layers,
            1,  # batch_size
            self.num_kv_heads,
            self.max_cache_len,
            self.head_dim
        )
        self.past_key = np.zeros(shape, dtype=self.dtype)
        self.past_value = np.zeros(shape, dtype=self.dtype)
        self.current_len = 0
    
    def get_cache(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前缓存
        
        Returns:
            (past_key, past_value)
        """
        return self.past_key, self.past_value
    
    def update(
        self,
        present_key: np.ndarray,
        present_value: np.ndarray,
        q_len: int
    ):
        """
        更新缓存
        
        Args:
            present_key: 新的 key
            present_value: 新的 value
            q_len: 查询长度
        """
        # 更新缓存
        self.past_key = present_key.astype(self.dtype)
        self.past_value = present_value.astype(self.dtype)
        self.current_len += q_len
    
    def rollback(self, seq_len: int):
        """
        回滚缓存
        
        Args:
            seq_len: 回滚的序列长度
        """
        self.current_len = max(0, self.current_len - seq_len)


def create_kvcache(
    cache_type: str,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_len: int
) -> KVCache:
    """
    创建 KV Cache
    
    Args:
        cache_type: 缓存类型（目前只支持 "basic"）
        num_layers: 层数
        num_kv_heads: KV 头数
        head_dim: 头维度
        max_cache_len: 最大缓存长度
        
    Returns:
        KVCache 实例
    """
    if cache_type == "basic":
        return KVCache(num_layers, num_kv_heads, head_dim, max_cache_len)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
