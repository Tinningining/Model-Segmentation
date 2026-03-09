"""
Node 3: 尾节点
负责 M3 (layers 17-21 + lm_head)，是流水线的终点
生成 logits 并采样 token，返回给头节点
"""
import argparse
import numpy as np

from config import DistributedConfig
from network import NodeServer, NodeClient, DistributedMessage
from acl_model import ACLModel
from kvcache import create_kvcache
from utils import reshape_hidden_output, reshape_kv_output, sample_logits


class TailNode:
    """尾节点：M3 (layers 17-21 + lm_head)"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.node_name = "Node3-Tail"
        
        # 网络组件
        self.server = None  # 接收来自上一个节点的数据
        self.client = None  # 发送 token 回头节点
        
        # ACL 模型
        self.model = None
        
        # KV Cache
        self.kv_cache = None
    
    def init(self):
        """初始化节点"""
        print(f"[{self.node_name}] Initializing...")
        
        # 1. 加载模型
        model_paths = self.config.get_model_paths()
        print(f"[{self.node_name}] Loading model: {model_paths[0]}")
        self.model = ACLModel(model_paths[0], self.config.device_id)
        self.model.init()
        
        # 2. 初始化 KV Cache
        self.kv_cache = create_kvcache(
            cache_type="basic",
            num_layers=self.config.get_num_layers(),
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_cache_len=self.config.max_cache_len
        )
        print(f"[{self.node_name}] KV Cache initialized: {self.config.get_num_layers()} layers")
        
        # 3. 启动服务器
        listen_port = self.config.get_listen_port()
        self.server = NodeServer(listen_port, self.node_name)
        self.server.start()
        
        # 4. 连接到头节点（返回结果）
        head_addr = self.config.get_head_node_address()
        self.client = NodeClient(
            head_addr["ip"],
            head_addr["port"],
            self.node_name
        )
        self.client.connect()
        
        # 5. 等待上一个节点连接
        print(f"[{self.node_name}] Waiting for previous node connection...")
        self.server.accept_connection()
        
        print(f"[{self.node_name}] Initialization complete!")
    
    def run_model(
        self,
        hidden: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        past_key_values_0: np.ndarray,
        q_len: int
    ) -> tuple:
        """
        运行模型
        
        Returns:
            (logits, next_token)
        """
        # 获取 KV Cache
        past_key, past_value = self.kv_cache.get_cache()
        
        # 准备输入
        inputs = [
            hidden.astype(np.float32),
            input_ids.astype(np.int64),
            attention_mask.astype(np.float32),
            position_ids.astype(np.int64),
            past_key_values_0.astype(np.float32),  # 用于 RoPE
        ]
        
        # 添加每一层的 KV Cache
        for i in range(self.config.get_num_layers()):
            inputs.append(past_key[i:i+1].astype(np.float32))
            inputs.append(past_value[i:i+1].astype(np.float32))
        
        # 执行推理
        outputs = self.model.execute(inputs)
        
        # 解析输出
        # outputs[0]: logits [1, max_input_len, vocab_size]
        # outputs[1:]: present_key_values (key0, value0, key1, value1, ...)
        logits_raw = outputs[0].view(np.float32).reshape(-1)
        
        # 重塑 logits
        logits_size = 1 * self.config.max_input_len * self.config.vocab_size
        if logits_raw.shape[0] >= logits_size:
            logits = logits_raw[:logits_size].reshape(
                1, self.config.max_input_len, self.config.vocab_size
            )
        else:
            logits = np.zeros((1, self.config.max_input_len, self.config.vocab_size), dtype=np.float32)
            logits.reshape(-1)[:logits_raw.shape[0]] = logits_raw
        
        # 获取最后一个有效位置的 logits
        # q_len 是当前查询长度，最后一个 token 在 q_len-1 位置
        last_logits = logits[0, q_len - 1, :]
        
        # 采样下一个 token
        next_token = sample_logits(
            last_logits,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            greedy=self.config.greedy
        )
        
        # 重组 KV Cache
        present_keys = []
        present_values = []
        for i in range(self.config.get_num_layers()):
            key_idx = 1 + i * 2
            value_idx = 1 + i * 2 + 1
            present_keys.append(outputs[key_idx])
            present_values.append(outputs[value_idx])
        
        # 堆叠成正确的形状
        target_shape = (
            self.config.get_num_layers(),
            1,
            self.config.num_key_value_heads,
            self.config.max_input_len,
            self.config.head_dim
        )
        
        present_key = np.zeros(target_shape, dtype=np.float32)
        present_value = np.zeros(target_shape, dtype=np.float32)
        
        for i in range(self.config.get_num_layers()):
            key_data = present_keys[i].view(np.float32).reshape(-1)
            value_data = present_values[i].view(np.float32).reshape(-1)
            
            layer_size = np.prod(target_shape[1:])
            present_key[i] = key_data[:layer_size].reshape(target_shape[1:])
            present_value[i] = value_data[:layer_size].reshape(target_shape[1:])
        
        # 更新 KV Cache
        self.kv_cache.update(
            present_key.astype(np.float16),
            present_value.astype(np.float16),
            q_len
        )
        
        return logits, next_token
    
    def process_loop(self):
        """处理循环"""
        print(f"[{self.node_name}] Starting process loop...")
        
        while True:
            # 接收消息
            data = self.server.recv()
            if data is None:
                print(f"[{self.node_name}] Connection closed")
                break
            
            msg = DistributedMessage.from_dict(data)
            
            if msg.msg_type == DistributedMessage.MSG_FORWARD:
                # 提取数据
                hidden = msg.data["hidden"]
                input_ids = msg.data["input_ids"]
                attention_mask = msg.data["attention_mask"]
                position_ids = msg.data["position_ids"]
                past_key_values_0 = msg.data["past_key_values_0"]
                meta = msg.data.get("meta", {})
                
                q_len = meta.get("q_len", 1)
                
                # 运行模型
                logits, next_token = self.run_model(
                    hidden, input_ids, attention_mask,
                    position_ids, past_key_values_0, q_len
                )
                
                # 返回结果给头节点
                result_msg = DistributedMessage.create_result_msg(
                    step=msg.step,
                    logits=logits,
                    next_token=next_token
                )
                self.client.send(result_msg.to_dict())
                
                print(f"[{self.node_name}] Processed step {msg.step}, next_token={next_token}")
            
            elif msg.msg_type == DistributedMessage.MSG_RESET:
                print(f"[{self.node_name}] Resetting KV Cache")
                self.kv_cache.reset()
                # 不需要转发，直接确认
                self.client.send(msg.to_dict())
            
            elif msg.msg_type == DistributedMessage.MSG_SHUTDOWN:
                print(f"[{self.node_name}] Received shutdown signal")
                # 确认关闭
                self.client.send(msg.to_dict())
                break
    
    def shutdown(self):
        """关闭节点"""
        if self.model:
            self.model.finalize()
        if self.server:
            self.server.close()
        if self.client:
            self.client.close()


def main():
    parser = argparse.ArgumentParser(description="Node 3: Tail Node (M3: layers 17-21 + lm_head)")
    parser.add_argument("--om_dir", type=str, required=True, help="OM model directory")
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    parser.add_argument("--max_cache_len", type=int, default=1024)
    parser.add_argument("--max_input_len", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", default=True)
    
    # 网络配置
    parser.add_argument("--listen_port", type=int, default=8003)
    parser.add_argument("--head_ip", type=str, default="127.0.0.1")
    parser.add_argument("--head_port", type=int, default=8000)
    
    args = parser.parse_args()
    
    # 创建配置
    config = DistributedConfig(
        om_dir=args.om_dir,
        device_id=args.device,
        max_cache_len=args.max_cache_len,
        max_input_len=args.max_input_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
        node_id=3,
    )
    
    # 更新网络配置
    config.node_addresses[3]["port"] = args.listen_port
    config.node_addresses[0]["ip"] = args.head_ip
    config.node_addresses[0]["port"] = args.head_port
    
    # 创建并运行节点
    node = TailNode(config)
    
    try:
        node.init()
        node.process_loop()
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
