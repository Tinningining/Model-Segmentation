"""
配置模块 - 单机 ONNX 执行
"""
from pathlib import Path
import json


class LocalConfig:
    """单机 ONNX 执行配置"""
    
    def __init__(
        self,
        system_onnx_dir: str = "",
        prefill_onnx_dir: str = "",
        decode_onnx_dir: str = "",
        tokenizer_dir: str = "",
        system_kv_dir: str = "./system_kv_cache",
        system_len: int = 256,
        prefill_len: int = 512,
        max_cache_len: int = 1024,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        greedy: bool = True,
    ):
        self.system_onnx_dir = system_onnx_dir
        self.prefill_onnx_dir = prefill_onnx_dir
        self.decode_onnx_dir = decode_onnx_dir
        self.tokenizer_dir = tokenizer_dir
        self.system_kv_dir = system_kv_dir
        
        self.system_len = system_len
        self.prefill_len = prefill_len
        self.max_cache_len = max_cache_len
        
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.greedy = greedy
        
        # 模型配置（从 config.json 读取）
        self._load_model_config()
    
    def _load_model_config(self):
        """从 ONNX 目录加载模型配置"""
        # 优先从 prefill 目录读取配置
        config_path = None
        for onnx_dir in [self.prefill_onnx_dir, self.decode_onnx_dir, self.system_onnx_dir]:
            if onnx_dir:
                p = Path(onnx_dir) / "config.json"
                if p.exists():
                    config_path = p
                    break
        
        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            self.hidden_size = config.get("hidden_size", 1536)
            self.num_attention_heads = config.get("num_attention_heads", 12)
            self.num_key_value_heads = config.get("num_key_value_heads", 2)
            self.head_dim = self.hidden_size // self.num_attention_heads
            self.num_hidden_layers = config.get("num_hidden_layers", 28)
            self.vocab_size = config.get("vocab_size", 151936)
            self.eos_token_id = config.get("eos_token_id", 151643)
            self.max_position_embeddings = config.get("max_position_embeddings", 32768)
        else:
            # 默认值（Qwen3-1.7B）
            self.hidden_size = 1536
            self.num_attention_heads = 12
            self.num_key_value_heads = 2
            self.head_dim = 128
            self.num_hidden_layers = 28
            self.vocab_size = 151936
            self.eos_token_id = 151643
            self.max_position_embeddings = 32768
    
    def get_system_model_paths(self):
        """获取 system 模型路径"""
        if not self.system_onnx_dir:
            return []
        base = Path(self.system_onnx_dir)
        return [
            str(base / "embed.onnx"),
            str(base / "layers_0_6.onnx"),
            str(base / "layers_7_13.onnx"),
            str(base / "layers_14_20.onnx"),
            str(base / "layers_21_27.onnx"),
            str(base / "output.onnx"),
        ]
    
    def get_prefill_model_paths(self):
        """获取 prefill 模型路径"""
        if not self.prefill_onnx_dir:
            return []
        base = Path(self.prefill_onnx_dir)
        return [
            str(base / "embed.onnx"),
            str(base / "layers_0_6.onnx"),
            str(base / "layers_7_13.onnx"),
            str(base / "layers_14_20.onnx"),
            str(base / "layers_21_27.onnx"),
            str(base / "output.onnx"),
        ]
    
    def get_decode_model_paths(self):
        """获取 decode 模型路径"""
        if not self.decode_onnx_dir:
            return []
        base = Path(self.decode_onnx_dir)
        return [
            str(base / "embed.onnx"),
            str(base / "layers_0_6.onnx"),
            str(base / "layers_7_13.onnx"),
            str(base / "layers_14_20.onnx"),
            str(base / "layers_21_27.onnx"),
            str(base / "output.onnx"),
        ]
