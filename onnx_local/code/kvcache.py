"""
KV Cache 管理模块 - 单机 ONNX 执行
"""
import numpy as np


class KVCache:
    """KV Cache 管理器"""
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_cache_len: int,
        dtype=np.float16
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_cache_len = max_cache_len
        self.dtype = dtype
        
        # 初始化 KV cache
        self.past_key = np.zeros(
            (num_layers, 1, num_kv_heads, max_cache_len, head_dim),
            dtype=dtype
        )
        self.past_value = np.zeros(
            (num_layers, 1, num_kv_heads, max_cache_len, head_dim),
            dtype=dtype
        )
        
        self.current_len = 0
    
    def reset(self):
        """重置 KV cache"""
        self.past_key.fill(0)
        self.past_value.fill(0)
        self.current_len = 0
    
    def update(self, new_key: np.ndarray, new_value: np.ndarray, q_len: int):
        """
        更新 KV cache
        
        Args:
            new_key: 新的 key，shape = (num_layers, 1, num_kv_heads, q_len, head_dim)
            new_value: 新的 value，shape = (num_layers, 1, num_kv_heads, q_len, head_dim)
            q_len: 新增的序列长度
        """
        if self.current_len + q_len > self.max_cache_len:
            raise ValueError(
                f"KV cache overflow: current_len={self.current_len}, "
                f"q_len={q_len}, max_cache_len={self.max_cache_len}"
            )
        
        # 将新的 KV 写入 cache
        start = self.current_len
        end = start + q_len
        self.past_key[:, :, :, start:end, :] = new_key.astype(self.dtype)
        self.past_value[:, :, :, start:end, :] = new_value.astype(self.dtype)
        
        self.current_len += q_len
    
    def get_cache(self):
        """获取当前的 KV cache"""
        return self.past_key, self.past_value
    
    def get_current_len(self):
        """获取当前 cache 长度"""
        return self.current_len
    
    def save_snapshot(self):
        """保存当前 KV cache 快照"""
        return {
            'past_key': self.past_key.copy(),
            'past_value': self.past_value.copy(),
            'current_len': self.current_len
        }
    
    def restore_snapshot(self, snapshot: dict):
        """恢复 KV cache 快照"""
        self.past_key[:] = snapshot['past_key']
        self.past_value[:] = snapshot['past_value']
        self.current_len = snapshot['current_len']
