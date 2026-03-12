"""
Node 3: 尾节点 - 4 节点版本
负责 layers_21_27.om + output.om，是流水线的终点
模型常驻内存，无需频繁加载/卸载
"""
import argparse
import numpy as np
from pathlib import Path

from config import DistributedConfig4Nodes
from network import NodeServer, NodeClient, DistributedMessage
from acl_model import ACLModel
from kvcache import create_kvcache
from utils import reshape_hidden_output, sample_token
from tools import ToolManager, ToolAgent
from tools.builtin_tools import weather_tool, calculator_tool


class TailNode4Nodes:
    """尾节点：layers_21_27 + output（模型常驻内存）"""
    
    def __init__(self, config: DistributedConfig4Nodes):
        self.config = config
        self.node_name = "Node3-Tail"
        
        # 网络组件
        self.server = None  # 接收上一个节点的数据
        self.client = None  # 发送结果回头节点
        
        # 模型（常驻内存）
        self.block_model = None
        self.output_model = None
        
        # KV Cache
        self.kv_cache = None
        
        # 工具系统
        self.tool_manager = None
        self.tool_agent = None
        
        # 状态
        self.past_len = 0
        self.step = 0
    
    def init(self):
        """初始化节点"""
        print(f"[{self.node_name}] Initializing...")
        
        # 1. 加载模型（常驻内存）
        model_paths = self.config.get_model_paths()
        
        print(f"[{self.node_name}] Loading models...")
        self.block_model = ACLModel(model_paths[0], self.config.device_id)
        self.block_model.init()
        
        self.output_model = ACLModel(model_paths[1], self.config.device_id)
        self.output_model.init()
        
        # 2. 初始化 KV Cache
        num_layers = self.config.get_num_layers()  # 7 层
        
        self.kv_cache = create_kvcache(
            cache_type="basic",
            num_layers=num_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_cache_len=self.config.max_cache_len
        )
        print(f"[{self.node_name}] KV Cache initialized for {num_layers} layers")
        
        # 3. 初始化工具系统
        self._init_tools()
        
        # 4. 启动服务器（接收上一个节点的数据）
        listen_port = self.config.get_listen_port()
        self.server = NodeServer(listen_port, self.node_name)
        self.server.start()
        
        # 5. 连接到头节点（返回结果）
        head_addr = self.config.get_head_node_address()
        self.client = NodeClient(head_addr["ip"], head_addr["port"], self.node_name)
        self.client.connect()
        
        # 6. 等待上一个节点连接
        print(f"[{self.node_name}] Waiting for previous node connection...")
        self.server.accept_connection()
        
        print(f"[{self.node_name}] Initialization complete!")
    
    def run_block(self, hidden: np.ndarray, attention_mask: np.ndarray,
                  position_ids: np.ndarray, q_len: int) -> np.ndarray:
        """运行 transformer block"""
        past_key, past_value = self.kv_cache.get_cache()
        
        inputs = [
            hidden.astype(np.float32),
            attention_mask.astype(np.float32),
            position_ids.astype(np.int64),
            past_key.astype(np.float32),
            past_value.astype(np.float32),
        ]
        
        outputs = self.block_model.execute(inputs)
        
        hidden_out = reshape_hidden_output(
            outputs[0], batch_size=1,
            max_input_len=self.config.max_input_len,
            hidden_size=self.config.hidden_size
        )
        
        # 更新 KV Cache
        num_layers = self.config.get_num_layers()
        target_shape = (num_layers, 1, self.config.num_key_value_heads,
                        self.config.max_input_len, self.config.head_dim)
        num_elements = np.prod(target_shape)
        
        present_key = outputs[1].view(np.float32).reshape(-1)[:num_elements].reshape(target_shape)
        present_value = outputs[2].view(np.float32).reshape(-1)[:num_elements].reshape(target_shape)
        
        self.kv_cache.update(present_key.astype(np.float16), present_value.astype(np.float16), q_len)
        
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
                
                # 运行 block
                hidden = self.run_block(hidden, attention_mask, position_ids, q_len)
                
                # 运行 output
                logits = self.run_output(hidden, q_len)
                
                # 采样生成 token
                next_token = sample_token(
                    logits,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    greedy=self.config.greedy
                )
                
                # 发送结果回头节点
                result_msg = DistributedMessage.create_result_msg(step=msg.step, next_token=next_token)
                self.client.send(result_msg.to_dict())
                
                self.step = msg.step
                print(f"[{self.node_name}] Step {self.step}: generated token {next_token}")
            
            elif msg.msg_type == DistributedMessage.MSG_TOOL_CALL:
                # 处理工具调用请求
                target_device = msg.data.get('target_device_id')
                
                if target_device == self.config.device_id:
                    # 本地执行
                    print(f"[{self.node_name}] Executing tool locally on Device {self.config.device_id}")
                    result_msg = self.tool_agent.handle_tool_call(msg)
                    # 发送结果回头节点
                    self.client.send(result_msg.to_dict())
                    print(f"[{self.node_name}] Tool result sent to head node")
                else:
                    # 不应该到达这里，因为这是最后一个节点
                    print(f"[{self.node_name}] Warning: Received tool call for device {target_device}, but I'm the tail node")
            
            elif msg.msg_type == DistributedMessage.MSG_TOOL_RESULT:
                # 将工具结果回传给头节点
                self.client.send(msg.to_dict())
                print(f"[{self.node_name}] Forwarded tool result to head node")
            
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
        self.kv_cache.reset()
    
    def _init_tools(self):
        """初始化工具系统"""
        print(f"[{self.node_name}] Initializing tool system...")
        
        # 创建工具管理器
        devices = [0, 1, 2, 3]
        self.tool_manager = ToolManager(devices, device_memory_limit=500)
        
        # 注册内置工具
        self.tool_manager.register_tool('get_weather', weather_tool.TOOL_CONFIG)
        self.tool_manager.register_tool('calculator', calculator_tool.TOOL_CONFIG)
        
        # 创建本地工具代理
        self.tool_agent = ToolAgent(
            device_id=self.config.device_id,
            tool_manager=self.tool_manager,
            node_name=self.node_name
        )
        self.tool_agent.start()
        
        print(f"[{self.node_name}] Tool agent started on Device {self.config.device_id}")
    
    def shutdown(self):
        """关闭节点"""
        if self.tool_agent:
            self.tool_agent.stop()
        if self.block_model:
            self.block_model.finalize()
        if self.output_model:
            self.output_model.finalize()
        if self.server:
            self.server.close()
        if self.client:
            self.client.close()


def main():
    parser = argparse.ArgumentParser(description="Node 3: Tail Node - 4 Nodes Version")
    parser.add_argument("--om_dir", type=str, required=True, help="OM model directory")
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    parser.add_argument("--max_cache_len", type=int, default=1024)
    parser.add_argument("--max_input_len", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--listen_port", type=int, default=9003)
    parser.add_argument("--head_ip", type=str, default="192.168.137.100")
    parser.add_argument("--head_port", type=int, default=9000)
    
    args = parser.parse_args()
    
    config = DistributedConfig4Nodes(
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
    
    config.node_addresses[3]["port"] = args.listen_port
    config.node_addresses[0]["ip"] = args.head_ip
    config.node_addresses[0]["port"] = args.head_port
    
    node = TailNode4Nodes(config)
    
    try:
        node.init()
        node.process_loop()
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
