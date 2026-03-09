# Llama 分布式推理框架

基于 Qwen 架构重构的 Llama 分布式推理框架，采用消息驱动的节点通信机制。

## 架构设计

### 节点划分

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Node 0     │────▶│   Node 1     │────▶│   Node 2     │────▶│   Node 3     │
│   (Head)     │     │  (Middle 1)  │     │  (Middle 2)  │     │   (Tail)     │
│              │     │              │     │              │     │              │
│ M0: embed +  │     │ M1: layers   │     │ M2: layers   │     │ M3: layers   │
│ layers 0-4   │     │ 5-10         │     │ 11-16        │     │ 17-21 +      │
│              │     │              │     │              │     │ lm_head      │
└──────┬───────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
       │                                                               │
       │                    next_token                                 │
       └───────────────────────◀───────────────────────────────────────┘
```

### 核心特性

- **消息驱动架构**：使用结构化消息（DistributedMessage）进行节点间通信
- **独立 KV Cache**：每个节点管理自己的 KV Cache，无需网络传输
- **双向通信**：结果直接返回头节点，无需中间转发
- **自动重连**：连接失败自动重试，提高可靠性
- **控制消息**：支持 reset、shutdown 等控制操作

## 文件结构

```
llama_distributed/
├── config.py           # 配置管理
├── network.py          # 网络通信模块
├── acl_model.py        # ACL 模型封装
├── kvcache.py          # KV Cache 管理
├── utils.py            # 工具函数
├── node_head.py        # 头节点（Node 0）
├── node_middle.py      # 中间节点（Node 1/2）
├── node_tail.py        # 尾节点（Node 3）
├── run_distributed.sh  # 部署脚本
└── README.md           # 本文档
```

## 快速开始

### 1. 准备模型文件

确保已经将 Llama 模型切分为 4 个 OM 文件：

```
om_models/
├── llama_m0_embed_layers_0_4.om
├── llama_m1_layers_5_10.om
├── llama_m2_layers_11_16.om
└── llama_m3_layers_17_21_lmhead.om
```

### 2. 准备初始 tokens

创建 `init_tokens.txt` 文件，包含初始 token IDs（空格分隔）：

```
1 2 3 4 5
```

### 3. 启动节点

**重要：必须按照从尾到头的顺序启动节点（Node 3 → Node 2 → Node 1 → Node 0）**

#### 单机部署（本地测试）

```bash
# 终端 1: 启动 Node 3 (尾节点)
python node_tail.py \
    --om_dir ./om_models \
    --device 0 \
    --listen_port 8003 \
    --head_ip 127.0.0.1 \
    --head_port 8000

# 终端 2: 启动 Node 2 (中间节点)
python node_middle.py \
    --node_id 2 \
    --om_dir ./om_models \
    --device 0 \
    --listen_port 8002 \
    --next_ip 127.0.0.1 \
    --next_port 8003

# 终端 3: 启动 Node 1 (中间节点)
python node_middle.py \
    --node_id 1 \
    --om_dir ./om_models \
    --device 0 \
    --listen_port 8001 \
    --next_ip 127.0.0.1 \
    --next_port 8002

# 终端 4: 启动 Node 0 (头节点)
python node_head.py \
    --om_dir ./om_models \
    --device 0 \
    --init_tokens init_tokens.txt \
    --max_new_tokens 100 \
    --listen_port 8000 \
    --next_ip 127.0.0.1 \
    --next_port 8001
```

#### 多机部署

假设有 4 台机器：
- Machine A (192.168.1.10): Node 0
- Machine B (192.168.1.11): Node 1
- Machine C (192.168.1.12): Node 2
- Machine D (192.168.1.13): Node 3

```bash
# Machine D: 启动 Node 3
python node_tail.py \
    --om_dir ./om_models \
    --device 0 \
    --listen_port 9003 \
    --head_ip 192.168.1.10 \
    --head_port 9000

# Machine C: 启动 Node 2
python node_middle.py \
    --node_id 2 \
    --om_dir ./om_models \
    --device 0 \
    --listen_port 9002 \
    --next_ip 192.168.1.13 \
    --next_port 9003

# Machine B: 启动 Node 1
python node_middle.py \
    --node_id 1 \
    --om_dir ./om_models \
    --device 0 \
    --listen_port 9001 \
    --next_ip 192.168.1.12 \
    --next_port 9002

# Machine A: 启动 Node 0
python node_head.py \
    --om_dir ./om_models \
    --device 0 \
    --init_tokens init_tokens.txt \
    --max_new_tokens 100 \
    --listen_port 9000 \
    --next_ip 192.168.1.11 \
    --next_port 9001
```

### 4. 使用部署脚本

也可以使用提供的部署脚本（单机测试）：

```bash
chmod +x run_distributed.sh
./run_distributed.sh
```

## 命令行参数

### 通用参数

- `--om_dir`: OM 模型目录路径
- `--device`: 设备 ID（默认：0）
- `--max_cache_len`: 最大缓存长度（默认：1024）
- `--max_input_len`: 最大输入长度（默认：16）

### 头节点（node_head.py）

- `--init_tokens`: 初始 token 文件路径
- `--max_new_tokens`: 最大生成 token 数（默认：100）
- `--temperature`: 采样温度（默认：1.0）
- `--top_k`: Top-K 采样（默认：0，不使用）
- `--top_p`: Top-P 采样（默认：1.0）
- `--greedy`: 贪婪采样（默认：True）
- `--listen_port`: 监听端口（默认：8000）
- `--next_ip`: 下一个节点 IP（默认：127.0.0.1）
- `--next_port`: 下一个节点端口（默认：8001）

### 中间节点（node_middle.py）

- `--node_id`: 节点 ID（1 或 2）
- `--listen_port`: 监听端口
- `--next_ip`: 下一个节点 IP
- `--next_port`: 下一个节点端口

### 尾节点（node_tail.py）

- `--temperature`: 采样温度（默认：1.0）
- `--top_k`: Top-K 采样（默认：0）
- `--top_p`: Top-P 采样（默认：1.0）
- `--greedy`: 贪婪采样（默认：True）
- `--listen_port`: 监听端口（默认：8003）
- `--head_ip`: 头节点 IP（默认：127.0.0.1）
- `--head_port`: 头节点端口（默认：8000）

## 消息类型

框架支持以下消息类型：

1. **MSG_FORWARD**: 前向传播数据
   - 包含：hidden_states, input_ids, attention_mask, position_ids, past_key_values_0, meta

2. **MSG_RESULT**: 结果返回
   - 包含：logits, next_token

3. **MSG_RESET**: 重置 KV Cache
   - 用于开始新的生成任务

4. **MSG_SHUTDOWN**: 关闭节点
   - 优雅地关闭所有节点

## 性能优化建议

1. **网络优化**
   - 使用高速网络（10Gbps+）
   - 减少网络跳数
   - 使用专用网络接口

2. **模型优化**
   - 使用量化模型（INT8/INT4）
   - 优化模型切分点
   - 调整 max_input_len 和 max_cache_len

3. **系统优化**
   - 绑定 CPU 核心
   - 调整网络缓冲区大小
   - 使用 NUMA 亲和性

## 故障排查

### 连接失败

**问题**：节点无法连接到下一个节点

**解决方案**：
1. 检查启动顺序（必须从尾到头）
2. 检查防火墙设置
3. 检查 IP 和端口配置
4. 查看节点日志

### 性能问题

**问题**：生成速度慢

**解决方案**：
1. 检查网络延迟（ping 测试）
2. 检查模型加载是否成功
3. 调整 max_input_len（减小可提高速度）
4. 使用量化模型

### 内存问题

**问题**：内存不足

**解决方案**：
1. 减小 max_cache_len
2. 减小 max_input_len
3. 使用更小的模型
4. 分配更多设备内存

## 与原 Llama 框架的对比

| 特性 | 原框架 | 新框架 |
|------|--------|--------|
| 节点抽象 | 无统一抽象 | 明确的角色划分 |
| 通信机制 | 简单序列化 | 结构化消息 |
| KV Cache | 传递依赖 | 独立管理 |
| 部署复杂度 | 高（多种脚本） | 中（统一接口） |
| 可扩展性 | 差 | 好 |
| 维护性 | 差 | 好 |
| 性能 | 一般 | 优 |

## 开发指南

### 添加新节点

1. 继承 `MiddleNode` 或创建新的节点类
2. 实现 `run_model` 方法
3. 更新 `config.py` 中的节点配置
4. 添加启动脚本

### 自定义消息类型

1. 在 `network.py` 中添加新的消息类型常量
2. 在 `DistributedMessage` 中添加创建方法
3. 在节点的 `process_loop` 中处理新消息

### 调试技巧

1. 使用 `print` 语句跟踪消息流
2. 检查 `msg.step` 确保消息顺序
3. 验证数据形状和类型
4. 使用小的 `max_new_tokens` 测试

## 许可证

本项目基于原 Llama 推理框架重构，遵循相同的许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请联系项目维护者。
