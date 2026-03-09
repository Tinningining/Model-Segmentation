"""
KV Cache 管理模块
用于管理分布式推理中的 KV 缓存
"""
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class KVCache:
    """KV Cache 管理器"""
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_cache_len: int,
        dtype: np.dtype = np.float16
    ):
        """
        初始化 KV Cache
        
        Args:
            num_layers: 当前节点的层数
            num_kv_heads: KV 头数
            head_dim: 每个头的维度
            max_cache_len: 最大缓存长度
            dtype: 数据类型
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_cache_len = max_cache_len
        self.dtype = dtype
        
        # 当前缓存长度
        self.current_len = 0
        
        # 初始化缓存
        # Shape: (num_layers, batch_size=1, num_kv_heads, max_cache_len, head_dim)
        self.past_key = None
        self.past_value = None
        
        self._init_cache()
    
    def _init_cache(self):
        """初始化空缓存"""
        shape = (self.num_layers, 1, self.num_kv_heads, self.max_cache_len, self.head_dim)
        self.past_key = np.zeros(shape, dtype=self.dtype)
        self.past_value = np.zeros(shape, dtype=self.dtype)
        self.current_len = 0
    
    def reset(self):
        """重置缓存"""
        self._init_cache()
    
    def get_cache(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前缓存"""
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
            present_key: 新的 key，shape: (num_layers, batch, num_kv_heads, q_len, head_dim)
            present_value: 新的 value，shape 同上
            q_len: 当前输入的序列长度
        """
        if self.current_len + q_len > self.max_cache_len:
            raise ValueError(
                f"KV Cache overflow: current_len={self.current_len}, "
                f"q_len={q_len}, max_cache_len={self.max_cache_len}"
            )
        
        start = self.current_len
        end = start + q_len
        
        # 更新缓存
        self.past_key[:, :, :, start:end, :] = present_key[:, :, :, :q_len, :]
        self.past_value[:, :, :, start:end, :] = present_value[:, :, :, :q_len, :]
        
        self.current_len = end
    
    def get_current_len(self) -> int:
        """获取当前缓存长度"""
        return self.current_len
    
    def save(self, key_path: str, value_path: str):
        """保存缓存到文件"""
        np.save(key_path, self.past_key)
        np.save(value_path, self.past_value)
    
    def load(self, key_path: str, value_path: str):
        """从文件加载缓存"""
        if Path(key_path).exists() and Path(value_path).exists():
            self.past_key = np.load(key_path)
            self.past_value = np.load(value_path)


class SlidingWindowKVCache(KVCache):
    """滑动窗口 KV Cache"""
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_cache_len: int,
        head_len: int = 32,
        dtype: np.dtype = np.float16
    ):
        """
        初始化滑动窗口 KV Cache
        
        Args:
            head_len: 保留的头部 token 数量
        """
        super().__init__(num_layers, num_kv_heads, head_dim, max_cache_len, dtype)
        self.head_len = head_len
        self.write_pos = 0  # 写入位置
        self.is_full = False  # 是否已满
    
    def update(
        self,
        present_key: np.ndarray,
        present_value: np.ndarray,
        q_len: int
    ):
        """
        更新缓存（滑动窗口方式）
        """
        for i in range(q_len):
            # 获取单个 token 的 KV
            key_token = present_key[:, :, :, i:i+1, :]
            value_token = present_value[:, :, :, i:i+1, :]
            
            if self.write_pos >= self.max_cache_len:
                # 缓存已满，开始覆盖（保留 head_len）
                self.write_pos = self.head_len
                self.is_full = True
            
            self.past_key[:, :, :, self.write_pos:self.write_pos+1, :] = key_token
            self.past_value[:, :, :, self.write_pos:self.write_pos+1, :] = value_token
            
            self.write_pos += 1
        
        self.current_len = min(self.write_pos, self.max_cache_len)
    
    def reset(self):
        """重置缓存"""
        super().reset()
        self.write_pos = 0
        self.is_full = False


def create_kvcache(
    cache_type: str,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_len: int,
    **kwargs
) -> KVCache:
    """
    创建 KV Cache 实例
    
    Args:
        cache_type: 缓存类型 ("basic" 或 "sliding-window")
    """
    if cache_type == "sliding-window":
        return SlidingWindowKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_cache_len=max_cache_len,
            head_len=kwargs.get("head_len", 32)
        )
    else:
        return KVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_cache_len=max_cache_len
        )
