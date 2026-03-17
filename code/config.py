"""
4 节点分布式推理配置模块
支持 Qwen 模型的 4 节点分布式推理（Prefill + Decode 双模型）
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os
import json


@dataclass
class DistributedConfig4Nodes:
    """4 节点分布式推理配置"""
    # 模型配置 - 三模型目录
    system_om_dir: str = ""   # System OM 模型目录（预计算固定 system prompt）
    prefill_om_dir: str = ""  # Prefill OM 模型目录（带 past KV）
    decode_om_dir: str = ""   # Decode OM 模型目录
    om_dir: str = ""          # 兼容旧接口，若设置则 prefill/decode 都用此目录
    tokenizer_dir: str = ""   # tokenizer 目录

    # System KV cache 配置
    system_kv_dir: str = ""   # System KV cache 独立存储目录
    system_len: int = 256     # System 阶段的 max_input_len

    # 模型参数（从 config.json 加载）
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    max_position_embeddings: int = 40960

    # KV Cache 配置
    max_cache_len: int = 1024
    prefill_len: int = 512     # Prefill 阶段的 max_input_len
    max_input_len: int = 512   # 兼容旧接口，decode 阶段固定为 1

    # 采样配置
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    greedy: bool = True
    max_gen_len: int = 100

    # 设备配置
    device_id: int = 0

    # 网络配置
    node_id: int = 0
    total_nodes: int = 4

    # 网络地址配置
    node_addresses: Dict[int, Dict[str, any]] = field(default_factory=lambda: {
        0: {"ip": "192.168.137.100", "port": 9000},
        1: {"ip": "192.168.137.101", "port": 9001},
        2: {"ip": "192.168.137.102", "port": 9002},
        3: {"ip": "192.168.137.103", "port": 9003},
    })

    # 每个节点的模型文件（prefill 和 decode 共用相同文件名，目录不同）
    node_models: Dict[int, List[str]] = field(default_factory=lambda: {
        0: ["embed.om", "layers_0_6.om"],
        1: ["layers_7_13.om"],
        2: ["layers_14_20.om"],
        3: ["layers_21_27.om", "output.om"],
    })

    node_layers: Dict[int, int] = field(default_factory=lambda: {
        0: 7,
        1: 7,
        2: 7,
        3: 7,
    })

    # EOS token ID
    eos_token_id: int = 151645
    bos_token_id: int = 151643

    def __post_init__(self):
        """初始化后处理"""
        # 兼容旧接口：如果只设了 om_dir，所有阶段都用它
        if self.om_dir and not self.system_om_dir:
            self.system_om_dir = self.om_dir
        if self.om_dir and not self.prefill_om_dir:
            self.prefill_om_dir = self.om_dir
        if self.om_dir and not self.decode_om_dir:
            self.decode_om_dir = self.om_dir

        self._validate_config()

        # 从 prefill 目录加载 config.json
        for d in [self.prefill_om_dir, self.decode_om_dir]:
            if d and os.path.isdir(d):
                config_path = os.path.join(d, "config.json")
                if os.path.exists(config_path):
                    self._load_model_config(config_path)
                    break

    def _validate_config(self):
        if not 0 <= self.node_id < self.total_nodes:
            raise ValueError(f"Invalid node_id: {self.node_id}")
        if self.device_id < 0:
            raise ValueError(f"Invalid device_id: {self.device_id}")
        if self.max_cache_len <= 0:
            raise ValueError(f"Invalid max_cache_len: {self.max_cache_len}")
        if self.prefill_len <= 0:
            raise ValueError(f"Invalid prefill_len: {self.prefill_len}")
        if self.temperature <= 0:
            raise ValueError(f"Invalid temperature: {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"Invalid top_k: {self.top_k}")
        if not 0 < self.top_p <= 1.0:
            raise ValueError(f"Invalid top_p: {self.top_p}")
        if self.max_gen_len <= 0:
            raise ValueError(f"Invalid max_gen_len: {self.max_gen_len}")
        if self.hidden_size <= 0:
            raise ValueError(f"Invalid hidden_size: {self.hidden_size}")
        if self.num_hidden_layers != 28:
            raise ValueError(f"Invalid num_hidden_layers: {self.num_hidden_layers}")
        if self.num_attention_heads <= 0:
            raise ValueError(f"Invalid num_attention_heads: {self.num_attention_heads}")
        if self.num_key_value_heads <= 0 or self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(f"Invalid num_key_value_heads: {self.num_key_value_heads}")
        if self.vocab_size <= 0:
            raise ValueError(f"Invalid vocab_size: {self.vocab_size}")
        if len(self.node_layers) != self.total_nodes:
            raise ValueError(f"node_layers must have {self.total_nodes} entries")
        if sum(self.node_layers.values()) != self.num_hidden_layers:
            raise ValueError(f"Sum of node_layers must equal num_hidden_layers")

    def _load_model_config(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.hidden_size = cfg.get("hidden_size", self.hidden_size)
        self.num_hidden_layers = cfg.get("num_hidden_layers", self.num_hidden_layers)
        self.num_attention_heads = cfg.get("num_attention_heads", self.num_attention_heads)
        self.num_key_value_heads = cfg.get("num_key_value_heads", self.num_key_value_heads)
        self.head_dim = cfg.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.vocab_size = cfg.get("vocab_size", self.vocab_size)
        self.max_position_embeddings = cfg.get("max_position_embeddings", self.max_position_embeddings)
        self.eos_token_id = cfg.get("eos_token_id", self.eos_token_id)
        self.bos_token_id = cfg.get("bos_token_id", self.bos_token_id)

    def get_system_model_paths(self, node_id: int = None) -> List[str]:
        """获取 system 模型路径"""
        if node_id is None:
            node_id = self.node_id
        model_files = self.node_models.get(node_id, [])
        return [os.path.join(self.system_om_dir, f) for f in model_files]

    def get_prefill_model_paths(self, node_id: int = None) -> List[str]:
        """获取 prefill 模型路径"""
        if node_id is None:
            node_id = self.node_id
        model_files = self.node_models.get(node_id, [])
        return [os.path.join(self.prefill_om_dir, f) for f in model_files]

    def get_decode_model_paths(self, node_id: int = None) -> List[str]:
        """获取 decode 模型路径"""
        if node_id is None:
            node_id = self.node_id
        model_files = self.node_models.get(node_id, [])
        return [os.path.join(self.decode_om_dir, f) for f in model_files]

    def get_model_paths(self, node_id: int = None) -> List[str]:
        """兼容旧接口，返回 decode 模型路径"""
        return self.get_decode_model_paths(node_id)

    def get_next_node_address(self) -> Optional[Dict[str, any]]:
        next_id = self.node_id + 1
        if next_id >= self.total_nodes:
            return None
        return self.node_addresses.get(next_id)

    def get_prev_node_address(self) -> Optional[Dict[str, any]]:
        prev_id = self.node_id - 1
        if prev_id < 0:
            return None
        return self.node_addresses.get(prev_id)

    def get_head_node_address(self) -> Dict[str, any]:
        return self.node_addresses.get(0)

    def get_listen_port(self) -> int:
        return self.node_addresses.get(self.node_id, {}).get("port", 9000 + self.node_id)

    def get_num_layers(self) -> int:
        return self.node_layers.get(self.node_id, 7)

    def is_head_node(self) -> bool:
        return self.node_id == 0

    def is_tail_node(self) -> bool:
        return self.node_id == self.total_nodes - 1

    def is_middle_node(self) -> bool:
        return not self.is_head_node() and not self.is_tail_node()
