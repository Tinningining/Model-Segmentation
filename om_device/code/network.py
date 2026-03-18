"""
网络通信模块
实现节点间的数据传输
"""
import socket
import struct
import pickle
import time
import numpy as np
from typing import List, Optional, Any, Tuple


def send_msg(sock: socket.socket, msg: Any) -> bool:
    """
    发送消息（支持 numpy 数组和其他可序列化对象）
    使用 pickle 序列化，先发送长度再发送数据
    """
    try:
        data = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
        # 发送 4 字节长度头
        sock.sendall(struct.pack('>I', len(data)))
        # 发送数据
        sock.sendall(data)
        return True
    except (socket.error, OSError) as e:
        print(f"[Network] Send socket error: {e}")
        return False
    except (pickle.PicklingError, TypeError) as e:
        print(f"[Network] Send serialization error: {e}")
        return False
    except Exception as e:
        print(f"[Network] Send unexpected error: {e}")
        return False


def recv_msg(sock: socket.socket, timeout: float = None) -> Optional[Any]:
    """
    接收消息
    
    Args:
        sock: socket 对象
        timeout: 接收超时时间（秒），None 表示无限等待
    """
    if timeout:
        sock.settimeout(timeout)
    
    try:
        # 读取 4 字节长度头
        raw_msglen = _recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        
        # 验证消息长度合法性（防止恶意数据）
        if msglen > 100 * 1024 * 1024:  # 100MB 限制
            print(f"[Network] Recv error: message too large ({msglen} bytes)")
            return None
        
        # 读取数据
        data = _recvall(sock, msglen)
        if not data:
            return None
        return pickle.loads(data)
    except socket.timeout:
        print(f"[Network] Recv timeout after {timeout}s")
        return None
    except (socket.error, OSError) as e:
        print(f"[Network] Recv socket error: {e}")
        return None
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"[Network] Recv deserialization error: {e}")
        return None
    except Exception as e:
        print(f"[Network] Recv unexpected error: {e}")
        return None
    finally:
        if timeout:
            sock.settimeout(None)  # 恢复阻塞模式


def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    """
    接收指定长度的数据
    """
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


class NodeServer:
    """节点服务器，用于接收来自上一个节点的数据"""
    
    def __init__(self, port: int, node_name: str = "Node"):
        self.port = port
        self.node_name = node_name
        self.server_sock = None
        self.client_conn = None
        self.client_addr = None
    
    def start(self) -> bool:
        """启动服务器"""
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind(('0.0.0.0', self.port))
            self.server_sock.listen(1)
            print(f"[{self.node_name}] Server listening on port {self.port}")
            return True
        except Exception as e:
            print(f"[{self.node_name}] Server start failed: {e}")
            return False
    
    def accept_connection(self, timeout: float = None) -> bool:
        """等待并接受连接"""
        try:
            if timeout:
                self.server_sock.settimeout(timeout)
            self.client_conn, self.client_addr = self.server_sock.accept()
            print(f"[{self.node_name}] Accepted connection from {self.client_addr}")
            return True
        except socket.timeout:
            print(f"[{self.node_name}] Accept timeout")
            return False
        except Exception as e:
            print(f"[{self.node_name}] Accept failed: {e}")
            return False
    
    def recv(self) -> Optional[Any]:
        """接收数据"""
        if self.client_conn is None:
            return None
        return recv_msg(self.client_conn)
    
    def send(self, msg: Any) -> bool:
        """发送数据（用于返回结果）"""
        if self.client_conn is None:
            return False
        return send_msg(self.client_conn, msg)
    
    def close(self):
        """关闭服务器"""
        if self.client_conn:
            self.client_conn.close()
        if self.server_sock:
            self.server_sock.close()


class NodeClient:
    """节点客户端，用于连接下一个节点"""
    
    def __init__(self, host: str, port: int, node_name: str = "Node"):
        self.host = host
        self.port = port
        self.node_name = node_name
        self.sock = None
    
    def connect(self, retry_interval: float = 1.0, max_retries: int = 600) -> bool:
        """连接到目标节点"""
        for i in range(max_retries):
            # 每次重试创建新的 socket（避免资源泄露）
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
            
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            try:
                self.sock.connect((self.host, self.port))
                print(f"[{self.node_name}] Connected to {self.host}:{self.port}")
                return True
            except ConnectionRefusedError:
                if i % 10 == 0:
                    print(f"[{self.node_name}] Waiting for {self.host}:{self.port}...")
                time.sleep(retry_interval)
            except (socket.error, OSError) as e:
                print(f"[{self.node_name}] Connect socket error: {e}")
                time.sleep(retry_interval)
            except Exception as e:
                print(f"[{self.node_name}] Connect unexpected error: {e}")
                time.sleep(retry_interval)
        
        # 连接失败，清理 socket
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        
        print(f"[{self.node_name}] Failed to connect after {max_retries} retries")
        return False
    
    def send(self, msg: Any) -> bool:
        """发送数据"""
        if self.sock is None:
            return False
        return send_msg(self.sock, msg)
    
    def recv(self) -> Optional[Any]:
        """接收数据（用于接收返回结果）"""
        if self.sock is None:
            return None
        return recv_msg(self.sock)
    
    def close(self):
        """关闭连接"""
        if self.sock:
            self.sock.close()


class DistributedMessage:
    """分布式消息封装"""
    
    # 消息类型
    MSG_FORWARD = "forward"      # 前向传播数据
    MSG_RESULT = "result"        # 结果返回
    MSG_CONTROL = "control"      # 控制消息
    MSG_RESET = "reset"          # 重置 KV Cache
    MSG_SHUTDOWN = "shutdown"    # 关闭节点
    MSG_TOOL_CALL = "tool_call"  # 工具调用请求
    MSG_TOOL_RESULT = "tool_result"  # 工具执行结果
    
    def __init__(
        self,
        msg_type: str,
        step: int = 0,
        data: dict = None
    ):
        self.msg_type = msg_type
        self.step = step
        self.data = data or {}
    
    def to_dict(self) -> dict:
        return {
            "msg_type": self.msg_type,
            "step": self.step,
            "data": self.data
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'DistributedMessage':
        return DistributedMessage(
            msg_type=d.get("msg_type", ""),
            step=d.get("step", 0),
            data=d.get("data", {})
        )
    
    @staticmethod
    def create_forward_msg(
        step: int,
        hidden: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        meta: dict
    ) -> 'DistributedMessage':
        """创建前向传播消息"""
        return DistributedMessage(
            msg_type=DistributedMessage.MSG_FORWARD,
            step=step,
            data={
                "hidden": hidden,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "meta": meta
            }
        )
    
    @staticmethod
    def create_result_msg(step: int, next_token: int, logits: np.ndarray = None) -> 'DistributedMessage':
        """创建结果消息"""
        data = {"next_token": next_token}
        if logits is not None:
            data["logits"] = logits
        return DistributedMessage(
            msg_type=DistributedMessage.MSG_RESULT,
            step=step,
            data=data
        )
    
    @staticmethod
    def create_reset_msg() -> 'DistributedMessage':
        """创建重置消息"""
        return DistributedMessage(
            msg_type=DistributedMessage.MSG_RESET,
            step=0,
            data={}
        )
    
    @staticmethod
    def create_shutdown_msg() -> 'DistributedMessage':
        """创建关闭消息"""
        return DistributedMessage(
            msg_type=DistributedMessage.MSG_SHUTDOWN,
            step=0,
            data={}
        )
    
    @staticmethod
    def create_tool_call_msg(
        tool_name: str,
        arguments: dict,
        request_id: str,
        target_device_id: int
    ) -> 'DistributedMessage':
        """创建工具调用请求消息"""
        return DistributedMessage(
            msg_type=DistributedMessage.MSG_TOOL_CALL,
            step=0,
            data={
                "tool_name": tool_name,
                "arguments": arguments,
                "request_id": request_id,
                "target_device_id": target_device_id
            }
        )
    
    @staticmethod
    def create_tool_result_msg(
        request_id: str,
        success: bool,
        result: Any = None,
        error: str = None
    ) -> 'DistributedMessage':
        """创建工具执行结果消息"""
        return DistributedMessage(
            msg_type=DistributedMessage.MSG_TOOL_RESULT,
            step=0,
            data={
                "request_id": request_id,
                "success": success,
                "result": result,
                "error": error
            }
        )
