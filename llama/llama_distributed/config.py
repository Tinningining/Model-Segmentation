"""
Llama 分布式推理配置
基于 Qwen 架构重构
"""
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class DistributedConfig:
    """分布式推理配置"""
    
    # 模型路径
    om_dir: str
    
    # 设备配置
    device_id: int = 0
    
    # 模型参数
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 64
    vocab_size: int = 32000
    
    # KV Cache 配置
    max_cache_len: int = 1024
    max_input_len: int = 16
    
    # 生成参数
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    greedy: bool = True
    eos_token_id: int = 2
    
    # 节点配置
    node_id: int = 0
    num_nodes: int = 4
    
    # 网络配置
    node_addresses: List[Dict[str, any]] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.node_addresses is None:
            # 默认网络配置（本地测试）
            self.node_addresses = [
                {"id": 0, "ip": "127.0.0.1", "port": 8000},  # Head
                {"id": 1, "ip": "127.0.0.1", "port": 8001},  # Middle 1
                {"id": 2, "ip": "127.0.0.1", "port": 8002},  # Middle 2
                {"id": 3, "ip": "127.0.0.1", "port": 8003},  # Tail
            ]
    
    def get_model_paths(self) -> List[str]:
        """获取当前节点的模型路径"""
        import os
        
        # Llama 4节点切分：
        # Node 0: M0 (embed + layers 0-4)
        # Node 1: M1 (layers 5-10)
        # Node 2: M2 (layers 11-16)
        # Node 3: M3 (layers 17-21 + lm_head)
        
        model_files = {
            0: ["llama_m0_embed_layers_0_4.om"],
            1: ["llama_m1_layers_5_10.om"],
            2: ["llama_m2_layers_11_16.om"],
            3: ["llama_m3_layers_17_21_lmhead.om"],
        }
        
        files = model_files.get(self.node_id, [])
        return [os.path.join(self.om_dir, f) for f in files]
    
    def get_num_layers(self) -> int:
        """获取当前节点的层数"""
        # Llama 层数分配
        layer_counts = {
            0: 5,   # layers 0-4
            1: 6,   # layers 5-10
            2: 6,   # layers 11-16
            3: 5,   # layers 17-21
        }
        return layer_counts.get(self.node_id, 0)
    
    def get_listen_port(self) -> int:
        """获取当前节点的监听端口"""
        return self.node_addresses[self.node_id]["port"]
    
    def get_next_node_address(self) -> Optional[Dict[str, any]]:
        """获取下一个节点的地址"""
        next_id = self.node_id + 1
        if next_id < self.num_nodes:
            return self.node_addresses[next_id]
        return None
    
    def get_head_node_address(self) -> Dict[str, any]:
        """获取头节点地址（用于尾节点返回结果）"""
        return self.node_addresses[0]
