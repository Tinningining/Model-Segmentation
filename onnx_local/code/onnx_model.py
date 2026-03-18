"""
ONNX 模型封装 - 单机执行
"""
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Tuple, Optional


class ONNXModelRunner:
    """ONNX 模型运行器"""
    
    def __init__(self, model_paths: List[str]):
        """
        初始化 ONNX 模型
        
        Args:
            model_paths: ONNX 模型路径列表
                [embed.onnx, layers_0_6.onnx, layers_7_13.onnx, 
                 layers_14_20.onnx, layers_21_27.onnx, lm_head.onnx]
        """
        self.model_paths = model_paths
        self.sessions = {}
        self._load_models()
    
    def _load_models(self):
        """加载所有 ONNX 模型"""
        model_names = ['embed', 'block0', 'block1', 'block2', 'block3', 'lm_head']
        
        for name, path in zip(model_names, self.model_paths):
            if not Path(path).exists():
                raise FileNotFoundError(f"Model not found: {path}")
            
            print(f"Loading {name} from {path}")
            session = ort.InferenceSession(
                path,
                providers=['CPUExecutionProvider']
            )
            self.sessions[name] = session
    
    def run_embed(self, input_ids: np.ndarray) -> np.ndarray:
        """
        运行 embedding 层
        
        Args:
            input_ids: shape (1, max_input_len), dtype=float32
        
        Returns:
            hidden_states: shape (1, max_input_len, hidden_size)
        """
        outputs = self.sessions['embed'].run(None, {'input_ids': input_ids})
        return outputs[0].astype(np.float32)
    
    def run_block(
        self,
        block_idx: int,
        hidden_states: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        past_key: Optional[np.ndarray] = None,
        past_value: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        运行 transformer block
        
        Args:
            block_idx: block 索引 (0-3)
            hidden_states: shape (1, max_input_len, hidden_size)
            attention_mask: shape (1, 1, max_input_len, max_cache_len + max_input_len)
            position_ids: shape (1, max_input_len)
            past_key: shape (num_layers, 1, num_kv_heads, max_cache_len, head_dim) or None
            past_value: shape (num_layers, 1, num_kv_heads, max_cache_len, head_dim) or None
        
        Returns:
            hidden_states: shape (1, max_input_len, hidden_size)
            present_key: shape (num_layers, 1, num_kv_heads, q_len, head_dim)
            present_value: shape (num_layers, 1, num_kv_heads, q_len, head_dim)
        """
        session = self.sessions[f'block{block_idx}']
        
        feeds = {
            'hidden_states': hidden_states.astype(np.float32),
            'attention_mask': attention_mask.astype(np.float32),
            'position_ids': position_ids.astype(np.int64),
        }
        
        # 如果有 past KV，添加到输入
        if past_key is not None and past_value is not None:
            feeds['past_key'] = past_key.astype(np.float32)
            feeds['past_value'] = past_value.astype(np.float32)
        
        outputs = session.run(None, feeds)
        
        hidden_out = outputs[0].astype(np.float32)
        present_key = outputs[1].astype(np.float32)
        present_value = outputs[2].astype(np.float32)
        
        return hidden_out, present_key, present_value
    
    def run_lm_head(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        运行 LM head
        
        Args:
            hidden_states: shape (1, max_input_len, hidden_size)
        
        Returns:
            logits: shape (1, max_input_len, vocab_size)
        """
        outputs = self.sessions['lm_head'].run(
            None,
            {'hidden_states': hidden_states.astype(np.float32)}
        )
        return outputs[0].astype(np.float32)
    
    def __del__(self):
        """清理资源"""
        self.sessions.clear()
