"""
Node 1: 尾节点 - 2 节点版本
负责 layers_14_20.om + layers_21_27.om + output.om，是流水线的终点
按顺序加载模型以节省 NPU 内存
"""
import argparse
import numpy as np
from pathlib import Path

from config import DistributedConfig2Nodes
from network import NodeServer, NodeClient, DistributedMessage
from acl_model import ACLModelLazy
from kvcache import create_kvcache
from utils import (
    build_attention_mask, build_position_ids,
    reshape_hidden_output, sample_token
)


class TailNode2Nodes:
    """尾节点：按顺序加载 layers_14_20 + layers_21_27 + output"""
    
    def __init__(self, config: DistributedConfig2Nodes):
        self.config = config
        self.node_name = "Node1-Tail"
        
        # 网络组件
        self.server = None
        self.client = None
        
        # 延迟加载的模型
        self.block_model_0 = None  # layers_14_20
        self.block_model_1 = None  # layers_21_27
        self.output_model = None   # output
        
        # KV Cache
        self.kv_cache_0 = None
        self.kv_cache_1 = None
        
        # 状态
        self.past_len = 0
        self.step = 0
    
    def init(self):
        """初始化节点"""
        print(f"[{self.node_name}] Initializing...")
        
        # 1. 创建延迟加载的模型（不立即加载）
        model_paths = self.config.get_model_paths()
        
        print(f"[{self.node_name}] Creating lazy models...")
        self.block_model_0 = ACLModelLazy(model_paths[0], self.config.device_id)
        self.block_model_1 = ACLModelLazy(model_paths[1], self.config.device_id)
        self.output_model = ACLModelLazy(model_paths[2], self.config.device_id)
        
        # 2. 初始化 KV Cache
        layers_per_block = 7
        
        self.kv_cache_0 = create_kvcache(
            cache_type="basic",
            num_layers=layers_per_block,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_cache_len=self.config.max_cache_len
        )
        
        self.kv_cache_1 = create_kvcache(
            cache_type="basic",
            num_layers=layers_per_block,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_cache_len=self.config.max_cache_len
        )
        print(f"[{self.node_name}] KV Cache initialized")
        
        # 3. 启动服务器
        listen_port = self.config.get_listen_port()
        self.server = NodeServer(listen_port, self.node_name)
        self.server.start()
        
        # 4. 连接到头节点
        head_addr = self.config.node_addresses[0]
        self.client = NodeClient(head_addr["ip"], head_addr["port"], self.node_name)
        self.client.connect()
        
        # 5. 等待上一个节点连接
        print(f"[{self.node_name}] Waiting for previous node connection...")
        self.server.accept_connection()
        
        print(f"[{self.node_name}] Initialization complete!")
    
    def run_block(self, model: ACLModelLazy, kv_cache, hidden: np.ndarray,
                  attention_mask: np.ndarray, position_ids: np.ndarray, q_len: int) -> np.ndarray:
        """运行单个 transformer block"""
        past_key, past_value = kv_cache.get_cache()
        
        inputs = [
            hidden.astype(np.float32),
            attention_mask.astype(np.float32),
            position_ids.astype(np.int64),
            past_key.astype(np.float32),
            past_value.astype(np.float32),
        ]
        
        outputs = model.execute(inputs)
        
        # 解析输出
        hidden_out = reshape_hidden_output(
            outputs[0], batch_size=1,
            max_input_len=self.config.max_input_len,
            hidden_size=self.config.hidden_size
        )
        
        # 更新 KV Cache
        layers_per_block = 7
        target_shape = (layers_per_block, 1, self.config.num_key_value_heads,
                        self.config.max_input_len, self.config.head_dim)
        num_elements = np.prod(target_shape)
        
        present_key = outputs[1].view(np.float32).reshape(-1)[:num_elements].reshape(target_shape)
        present_value = outputs[2].view(np.float32).reshape(-1)[:num_elements].reshape(target_shape)
        
        kv_cache.update(present_key.astype(np.float16), present_value.astype(np.float16), q_len)
        
        return hidden_out
    
    def run_output(self, hidden: np.ndarray, q_len: int) -> np.ndarray:
        """运行 output 模型"""
        outputs = self.output_model.execute([hidden.astype(np.float32)])
        
        raw_logits = outputs[0].view(np.float32)
        
        if raw_logits.size == self.config.vocab_size:
            logits = raw_logits.reshape(1, -1)
        elif raw_logits.size == self.config.max_input_len * self.config.vocab_size:
            logits = raw_logits.reshape(1, self.config.max_input_len, -1)
            logits = logits[:, q_len-1, :]
        else:
            logits = raw_logits.reshape(1, -1)
        
        return logits
    
    def process_forward(self, hidden: np.ndarray, attention_mask: np.ndarray,
                        position_ids: np.ndarray, q_len: int) -> int:
        """处理一次前向传播，按顺序加载/卸载模型"""
        
        # 1. 加载并运行 block 0
        self.block_model_0.load()
        hidden = self.run_block(self.block_model_0, self.kv_cache_0, hidden,
                                attention_mask, position_ids, q_len)
        self.block_model_0.unload()
        
        # 2. 加载并运行 block 1
        self.block_model_1.load()
        hidden = self.run_block(self.block_model_1, self.kv_cache_1, hidden,
                                attention_mask, position_ids, q_len)
        self.block_model_1.unload()
        
        # 3. 加载并运行 output
        self.output_model.load()
        logits = self.run_output(hidden, q_len)
        self.output_model.unload()
        
        # 4. 采样生成 token
        next_token = sample_token(
            logits,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            greedy=self.config.greedy
        )
        
        return next_token
    
    def process_loop(self):
        """处理循环"""
        print(f"[{self.node_name}] Starting processing loop...")
        
        while True:
            data = self.server.recv()
            if data is None:
                print(f"[{self.node_name}] Connection closed")
                break
            
            msg = DistributedMessage.from_dict(data)
            
            if msg.msg_type == DistributedMessage.MSG_FORWARD:
                hidden = msg.data["hidden"]
                attention_mask = msg.data["attention_mask"]
                position_ids = msg.data["position_ids"]
                meta = msg.data.get("meta", {})
                
                q_len = meta.get("q_len", 1)
                self.past_len = meta.get("past_len", 0)
                
                # 按顺序处理
                next_token = self.process_forward(hidden, attention_mask, position_ids, q_len)
                
                # 发送结果
                result_msg = DistributedMessage.create_result_msg(step=msg.step, next_token=next_token)
                self.client.send(result_msg.to_dict())
                
                self.step = msg.step
                print(f"[{self.node_name}] Step {self.step}: generated token {next_token}")
            
            elif msg.msg_type == DistributedMessage.MSG_RESET:
                self.reset()
                print(f"[{self.node_name}] Reset")
            
            elif msg.msg_type == DistributedMessage.MSG_SHUTDOWN:
                print(f"[{self.node_name}] Received shutdown signal")
                break
    
    def reset(self):
        """重置状态"""
        self.past_len = 0
        self.step = 0
        self.kv_cache_0.reset()
        self.kv_cache_1.reset()
    
    def shutdown(self):
        """关闭节点"""
        if self.block_model_0:
            self.block_model_0.unload()
        if self.block_model_1:
            self.block_model_1.unload()
        if self.output_model:
            self.output_model.unload()
        if self.server:
            self.server.close()
        if self.client:
            self.client.close()


def main():
    parser = argparse.ArgumentParser(description="Node 1: Tail Node - 2 Nodes Version")
    parser.add_argument("--om_dir", type=str, required=True, help="OM model directory")
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    parser.add_argument("--max_cache_len", type=int, default=1024)
    parser.add_argument("--max_input_len", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--listen_port", type=int, default=9001)
    parser.add_argument("--head_ip", type=str, default="127.0.0.1")
    parser.add_argument("--head_port", type=int, default=9000)
    
    args = parser.parse_args()
    
    config = DistributedConfig2Nodes(
        om_dir=args.om_dir,
        device_id=args.device,
        max_cache_len=args.max_cache_len,
        max_input_len=args.max_input_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
        node_id=1,
    )
    
    config.node_addresses[1]["port"] = args.listen_port
    config.node_addresses[0]["ip"] = args.head_ip
    config.node_addresses[0]["port"] = args.head_port
    
    node = TailNode2Nodes(config)
    
    try:
        node.init()
        node.process_loop()
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
