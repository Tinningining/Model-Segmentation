# Qwen 2 节点分布式推理框架

基于华为昇腾 ACL 的 Qwen 模型分布式推理框架，支持 **2 节点**流水线并行推理。

## 目录

- [架构概述](#架构概述)
- [模型参数](#模型参数)
- [文件结构](#文件结构)
- [香橙派昇腾分布式部署指南](#香橙派昇腾分布式部署指南)
- [单机测试](#单机测试)
- [详细配置](#详细配置)
- [API 使用](#api-使用)
- [故障排除](#故障排除)

---

## 架构概述

### 2 节点流水线架构

本框架将 Qwen 模型（28 层 Transformer）切分到 **2 个设备**上进行流水线并行推理：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           2 节点分布式推理数据流                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────────────────────┐    ┌──────────────────────────────┐  │
│   │         Node 0 (设备1)            │    │         Node 1 (设备2)        │  │
│   │                                  │    │                              │  │
│   │  embed.om                        │    │                              │  │
│   │  layers_0_6.om   (层 0-6)        │    │  layers_14_20.om (层 14-20)  │  │
│   │  layers_7_13.om  (层 7-13)       │───▶│  layers_21_27.om (层 21-27)  │  │
│   │                                  │    │  output.om                   │  │
│   │  [头节点/主节点]                  │    │  [尾节点]                     │  │
│   │  14 层 KV Cache                  │    │  14 层 KV Cache              │  │
│   └────────────▲─────────────────────┘    └──────────────┬───────────────┘  │
│                │                                         │                  │
│                └─────────── next_token ◀─────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

数据传输说明：
  • hidden_states: Node 0 → Node 1 (形状: [1, 16, 2048])
  • next_token:    Node 1 → Node 0 (单个 token ID)
  • KV Cache:      保存在每个设备本地，不需要网络传输
```

### 与 4 节点版本的对比

| 特性 | 4 节点版本 | 2 节点版本 |
|------|-----------|-----------|
| 设备数量 | 4 台 | 2 台 |
| 每节点层数 | 7 层 | 14 层 |
| 每节点 KV Cache | 7 层 | 14 层 |
| 网络通信次数 | 4 次/step | 2 次/step |
| 内存需求/节点 | 较低 | 较高 |

### 模型切分详情

| 节点 | 模型文件 | 内容 | 层数 |
|------|---------|------|------|
| **Node 0** | embed.om | Embedding 层 | - |
| **Node 0** | layers_0_6.om | Transformer 层 0-6 | 7 层 |
| **Node 0** | layers_7_13.om | Transformer 层 7-13 | 7 层 |
| **Node 1** | layers_14_20.om | Transformer 层 14-20 | 7 层 |
| **Node 1** | layers_21_27.om | Transformer 层 21-27 | 7 层 |
| **Node 1** | output.om | LM Head (输出层) | - |

### 节点职责

| 节点 | 角色 | 加载的模型 | 主要职责 |
|------|------|-----------|----------|
| **Node 0** | 头节点 | embed.om + layers_0_6.om + layers_7_13.om | 接收输入 → embedding → block0 → block1 → 发送 hidden_states → 接收 token |
| **Node 1** | 尾节点 | layers_14_20.om + layers_21_27.om + output.om | 接收 hidden_states → block0 → block1 → lm_head → 采样 → 发送 token |

### 内存优化：按顺序加载模型

由于香橙派昇腾的 NPU 内存有限（约 15GB），无法同时加载 3 个大模型（每个约 1.4GB）。本框架采用**按顺序加载模型**的策略：

```
每次推理步骤的模型加载流程：

Node 0 (头节点):
  1. 加载 embed.om → 执行 → 卸载
  2. 加载 layers_0_6.om → 执行 → 卸载
  3. 加载 layers_7_13.om → 执行 → 卸载
  4. 发送 hidden_states 到 Node 1

Node 1 (尾节点):
  1. 加载 layers_14_20.om → 执行 → 卸载
  2. 加载 layers_21_27.om → 执行 → 卸载
  3. 加载 output.om → 执行 → 卸载
  4. 发送 token 到 Node 0
```

**优点**：解决 NPU 内存不足问题
**缺点**：每次推理都需要加载/卸载模型，速度较慢

---

## 模型参数

### 模型配置（与 ATC 转换命令对应）

```bash
# embed.om 输入
atc --model="embed.onnx" --input_shape="input_ids:1,16"

# layers_X_Y.om 输入
atc --model="layers_X_Y.onnx" --input_shape="hidden_states:1,16,2048;attention_mask:1,1,16,1040;position_ids:1,16;past_key:7,1,8,1024,128;past_value:7,1,8,1024,128"

# output.om 输入
atc --model="output.onnx" --input_shape="hidden_states:1,16,2048"
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_size` | 2048 | 隐藏层维度 |
| `num_attention_heads` | 16 | 注意力头数 |
| `num_key_value_heads` | 8 | KV 头数 (GQA) |
| `head_dim` | 128 | 每个头的维度 |
| `num_hidden_layers` | 28 | Transformer 总层数 |
| `vocab_size` | 151936 | 词表大小 |
| `max_input_len` | 16 | 单次输入的最大 token 数 |
| `max_cache_len` | 1024 | KV Cache 最大缓存长度 |

### 采样参数（默认值）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | 1.0 | 采样温度 |
| `top_k` | 0 | Top-K 采样（0 表示禁用） |
| `top_p` | 1.0 | Top-P 采样 |
| `greedy` | True | 贪婪采样（直接取 argmax） |

---

## 文件结构

```
distributed_inference_2nodes/
├── config.py              # 2 节点配置类定义
├── network.py             # TCP 网络通信模块
├── kvcache.py             # KV Cache 管理
├── acl_model.py           # ACL 模型封装
├── utils.py               # 工具函数
│
├── node_head.py           # 头节点实现 (Node 0)
├── node_tail.py           # 尾节点实现 (Node 1)
│
└── README.md              # 本文档
```

---

## 香橙派昇腾分布式部署指南

本节详细介绍如何在 2 台香橙派昇腾开发板上部署分布式推理框架。

### 硬件准备

| 设备 | 数量 | 角色 | 需要的模型文件 |
|------|------|------|---------------|
| 香橙派昇腾 | **2 台** | 推理节点 | 见下表 |
| 开发机（电脑） | 1 台 | 远程控制 | 无（通过 SSH 控制香橙派） |
| 交换机 | 1 台 | 网络连接 | - |

**各香橙派需要的模型文件：**

| 设备 | IP 地址 | 角色 | 需要的模型文件 |
|------|---------|------|---------------|
| 香橙派 1 | 192.168.137.102 | Node 0 (头节点) | embed.om, layers_0_6.om, layers_7_13.om |
| 香橙派 2 | 192.168.137.100 | Node 1 (尾节点) | layers_14_20.om, layers_21_27.om, output.om |

### 网络连接说明

本方案使用**交换机直连**方式，所有设备（开发机 + 香橙派）通过网线连接到同一台交换机，无需路由器。

**网络连接方式：**

```
                    ┌─────────────────────────────────────────┐
                    │                 交换机                   │
                    └─────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────┐
          │                           │                       │
          ▼                           ▼                       ▼
     ┌────────┐                 ┌────────┐              ┌────────┐
     │ 开发机  │                 │香橙派1 │              │香橙派2 │
     │ (PC)   │                 │        │              │        │
     │ (有线) │                 │ (有线) │              │ (有线) │
     │192.168.137.101│           │192.168.137.102│       │192.168.137.100│
     └────────┘                 └────────┘              └────────┘
```

- **开发机**：通过网线连接到交换机，需要配置静态 IP（192.168.137.101）
- **香橙派**：通过网线连接到交换机，配置静态 IP
- **关键要求**：所有设备必须在同一网段（192.168.137.x），能够互相 ping 通
- **无需网关**：由于没有路由器，不需要配置网关

### 网络拓扑

```
                    ┌─────────────────────────────────────────┐
                    │                 交换机                   │
                    └─────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────┐
          │                           │                       │
     ┌────┴────┐                ┌─────┴─────┐           ┌─────┴─────┐
     │ 开发机   │                │ 香橙派 1   │           │ 香橙派 2   │
     │ (PC)    │                │ Node 0    │           │ Node 1    │
     │192.168.137.101│           │192.168.137.102│        │192.168.137.100│
     │          │                │ 端口:9000  │           │ 端口:9001  │
     └─────────┘                └───────────┘           └───────────┘
```

### 步骤 1：配置网络

配置香橙派有**两种方式**，你可以根据实际情况选择：

#### 方式 A：通过显示器和键盘直接配置（推荐初次配置）

如果你有显示器和键盘，可以直接连接到每台香橙派进行配置：

1. 将显示器（HDMI）和键盘连接到香橙派
2. 开机后直接在香橙派上操作
3. 配置完成后拔掉显示器和键盘，连接下一台

#### 方式 B：通过开发机 SSH 远程配置

如果香橙派已经有网络连接（例如通过 DHCP 获取了 IP），可以从开发机 SSH 连接进行配置：

```bash
# 首先找到香橙派的 IP（可以在路由器管理页面查看，或使用 nmap 扫描）
nmap -sn 192.168.137.0/24

# 然后 SSH 连接
ssh orangepi@<香橙派IP>
```

> **提示**：初次配置时，建议使用**方式 A**（显示器+键盘），因为此时香橙派可能还没有固定 IP，不方便 SSH 连接。配置好静态 IP 后，后续操作就可以全部通过 SSH 完成了。

#### 1.1 为每台香橙派设置静态 IP

提供两种配置方式，根据实际情况选择：

##### 方式一：使用 ip 命令（临时配置，重启后失效）

```bash
# 首先查看网络接口名称
ip link show
# 常见接口名：eth0, enp0s3, end0 等，以下以 eth0 为例

# === 香橙派 1 (Node 0 - 192.168.137.102) ===
# 清除 eth0 的 IP
sudo ip addr flush dev eth0

# 设置新 IP
sudo ip addr add 192.168.137.102/24 dev eth0

# 启用接口
sudo ip link set eth0 up

# 检查设置
ip addr show eth0

# === 香橙派 2 (Node 1 - 192.168.137.100) ===
# 清除 eth0 的 IP
sudo ip addr flush dev eth0

# 设置新 IP
sudo ip addr add 192.168.137.100/24 dev eth0

# 启用接口
sudo ip link set eth0 up

# 检查设置
ip addr show eth0
```

> **注意**：`ip` 命令设置的 IP 是临时的，系统重启后会丢失。适合临时测试使用。

##### 方式二：使用 nmcli 命令（持久化配置，重启后保留）

```bash
# 香橙派 1 (Node 0 - 192.168.137.102)
sudo nmcli con mod "Wired connection 1" ipv4.addresses 192.168.137.102/24
sudo nmcli con mod "Wired connection 1" ipv4.method manual
sudo nmcli con mod "Wired connection 1" ipv4.gateway ""
sudo nmcli con up "Wired connection 1"

# 香橙派 2 (Node 1 - 192.168.137.100)
sudo nmcli con mod "Wired connection 1" ipv4.addresses 192.168.137.100/24
sudo nmcli con mod "Wired connection 1" ipv4.method manual
sudo nmcli con mod "Wired connection 1" ipv4.gateway ""
sudo nmcli con up "Wired connection 1"

# 开发机 (PC) - Windows 系统需在网络设置中手动配置静态 IP: 192.168.137.101
# 或使用命令行（以管理员身份运行）：
# netsh interface ip set address "以太网" static 192.168.137.101 255.255.255.0
```

> **注意**：
> - 连接名称为 "Wired connection 1"，可用 `nmcli con show` 查看实际名称
> - 由于使用交换机直连无路由器，网关设置为空
> - 开发机也需要配置同网段的静态 IP 才能 SSH 连接到香橙派

#### 1.2 验证网络连通性

在香橙派 1 上测试与香橙派 2 的连接：

```bash
# 从 Node 0 (192.168.137.102) ping Node 1 (192.168.137.100)
ping 192.168.137.100

# 从 Node 1 (192.168.137.100) ping Node 0 (192.168.137.102)
ping 192.168.137.102
```

#### 1.3 开放防火墙端口

在每台香橙派上开放对应端口：

```bash
# 香橙派 1 (Node 0 - 192.168.137.102)
sudo ufw allow 9000/tcp

# 香橙派 2 (Node 1 - 192.168.137.100)
sudo ufw allow 9001/tcp

# 或者直接关闭防火墙（测试环境）
sudo ufw disable
```

### 步骤 2：准备环境

在**每台香橙派**上执行以下操作：

#### 2.1 安装依赖

```bash
# 安装 Python 依赖
pip3 install numpy

# 确认 ACL 环境已配置
# 香橙派昇腾通常已预装 CANN 和 pyACL
python3 -c "import acl; print('ACL OK')"
```

#### 2.2 创建工作目录

```bash
# 在每台香橙派上创建相同的目录结构
mkdir -p ~/qwen_distributed/models
mkdir -p ~/qwen_distributed/code
```

#### 2.3 复制代码文件

将 `distributed_inference_2nodes/` 目录下的所有 Python 文件复制到每台香橙派：

```bash
# 在开发机上执行（假设香橙派用户名为 orangepi）
scp *.py orangepi@192.168.137.102:~/qwen_distributed/code/
scp *.py orangepi@192.168.137.100:~/qwen_distributed/code/
```

### 步骤 3：分发模型文件

将对应的 .om 模型文件复制到各香橙派：

```bash
# 香橙派 1 (Node 0 - 192.168.137.102) - 需要 embed + layers_0_6 + layers_7_13
scp embed.om layers_0_6.om layers_7_13.om orangepi@192.168.137.102:~/qwen_distributed/models/

# 香橙派 2 (Node 1 - 192.168.137.100) - 需要 layers_14_20 + layers_21_27 + output
scp layers_14_20.om layers_21_27.om output.om orangepi@192.168.137.100:~/qwen_distributed/models/

# 可选：复制 config.json
scp config.json orangepi@192.168.137.102:~/qwen_distributed/models/
scp config.json orangepi@192.168.137.100:~/qwen_distributed/models/
```

### 步骤 4：准备输入文件

在香橙派 1 (Node 0) 上创建输入 token 文件：

```bash
# SSH 到香橙派 1 (Node 0)
ssh orangepi@192.168.137.102

# 创建输入 token 文件
cd ~/qwen_distributed/code
echo "151644 8948 198 2610 525 264 10950 17847 13" > init_tokens.txt
```

> **说明**：这些 token ID 对应 Qwen 的 tokenizer 编码结果。你可以使用 transformers 库预先编码你的 prompt。

### 步骤 5：启动分布式推理

**重要**：必须先启动 Node 1（尾节点），再启动 Node 0（头节点）！

> **操作说明**：你需要在你的**开发机（电脑）**上打开 **2 个终端窗口**，分别通过 SSH 连接到 2 台香橙派。每个终端窗口控制一台香橙派。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           开发机（你的电脑）                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────┐    ┌─────────────────────────────┐       │
│   │        终端窗口 1            │    │        终端窗口 2            │       │
│   │                             │    │                             │       │
│   │  SSH 连接到香橙派 1          │    │  SSH 连接到香橙派 2          │       │
│   │  (Node 0 - 头节点)          │    │  (Node 1 - 尾节点)          │       │
│   │  192.168.137.102            │    │  192.168.137.100            │       │
│   │                             │    │                             │       │
│   │  第二启动 ②                 │    │  首先启动 ①                 │       │
│   └─────────────────────────────┘    └─────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.1 【终端窗口 2】启动 Node 1（香橙派 2 - 尾节点）- 首先启动！

在你的**开发机**上打开一个新的终端窗口，执行：

```bash
# 从开发机 SSH 连接到香橙派 2 (Node 1 - 尾节点)
ssh orangepi@192.168.137.100
```

连接成功后，在香橙派 2 上执行：

```bash
# 进入代码目录
cd ~/qwen_distributed/code

# 启动尾节点
python3 node_tail.py \
    --om_dir ~/qwen_distributed/models \
    --device 0 \
    --max_cache_len 1024 \
    --max_input_len 16 \
    --listen_port 9001 \
    --head_ip 192.168.137.102 \
    --head_port 9000
```

等待看到以下输出后，**保持此终端窗口运行**，然后继续下一步：
```
[Node1-Tail] Initializing...
[Node1-Tail] Creating lazy models...
[Node1-Tail] KV Cache initialized
[Node1-Tail] Waiting for previous node connection...
```

> **注意**：由于采用按顺序加载模型的策略，初始化时不会立即加载模型，而是在推理时按需加载。

#### 5.2 【终端窗口 1】启动 Node 0（香橙派 1 - 头节点）- 第二启动！

在你的**开发机**上打开**另一个新的终端窗口**，执行：

```bash
# 从开发机 SSH 连接到香橙派 1 (Node 0 - 头节点)
ssh orangepi@192.168.137.102
```

连接成功后，在香橙派 1 上执行：

```bash
# 进入代码目录
cd ~/qwen_distributed/code

# 启动头节点
python3 node_head.py \
    --om_dir ~/qwen_distributed/models \
    --device 0 \
    --max_cache_len 1024 \
    --max_input_len 16 \
    --init_tokens init_tokens.txt \
    --max_new_tokens 100 \
    --listen_port 9000 \
    --next_ip 192.168.137.100 \
    --next_port 9001
```

启动后，你应该看到推理开始运行：
```
[Node0-Head] Initializing...
[Node0-Head] Creating lazy models...
[Node0-Head] KV Cache initialized
[Node0-Head] Waiting for tail node connection...
[Node0-Head] Initialization complete!
[ACLModelLazy] Loaded: .../embed.om
[ACLModelLazy] Unloaded: .../embed.om
[ACLModelLazy] Loaded: .../layers_0_6.om
[ACLModelLazy] Unloaded: .../layers_0_6.om
...
[Node0-Head] Step 0: generated token 12345
[Node0-Head] Step 1: generated token 67890
...
```

> **注意**：每个推理步骤都会看到模型的加载和卸载日志，这是正常的内存优化行为。

> **提示**：此时你可以观察 2 个终端窗口，每个节点都会打印自己的处理日志。

### 步骤 6：查看结果

推理完成后，Node 0 会输出：
```
Generated 100 tokens in 45.23s
Speed: 2.21 tokens/s
Generated IDs: [12345, 67890, ...]
```

### 快速启动脚本

为了方便部署，可以在每台香橙派上创建启动脚本：

#### 香橙派 1 (Node 0 - 192.168.137.102) - start_node0.sh

```bash
#!/bin/bash
cd ~/qwen_distributed/code
python3 node_head.py \
    --om_dir ~/qwen_distributed/models \
    --device 0 \
    --max_cache_len 1024 \
    --max_input_len 16 \
    --init_tokens init_tokens.txt \
    --max_new_tokens 100 \
    --listen_port 9000 \
    --next_ip 192.168.137.100 \
    --next_port 9001
```

#### 香橙派 2 (Node 1 - 192.168.137.100) - start_node1.sh

```bash
#!/bin/bash
cd ~/qwen_distributed/code
python3 node_tail.py \
    --om_dir ~/qwen_distributed/models \
    --device 0 \
    --max_cache_len 1024 \
    --max_input_len 16 \
    --listen_port 9001 \
    --head_ip 192.168.137.102 \
    --head_port 9000
```

### 一键部署脚本（在开发机上运行）

创建 `deploy_2nodes.sh`：

```bash
#!/bin/bash

# 配置
NODES=("192.168.137.102" "192.168.137.100")
USER="orangepi"
REMOTE_DIR="~/qwen_distributed"

echo "=== 步骤 1: 创建远程目录 ==="
for ip in "${NODES[@]}"; do
    echo "Creating directories on $ip..."
    ssh $USER@$ip "mkdir -p $REMOTE_DIR/models $REMOTE_DIR/code"
done

echo "=== 步骤 2: 复制代码文件 ==="
for ip in "${NODES[@]}"; do
    echo "Copying code to $ip..."
    scp *.py $USER@$ip:$REMOTE_DIR/code/
done

echo "=== 步骤 3: 分发模型文件 ==="
echo "Copying models to Node 0 (192.168.137.102)..."
scp embed.om layers_0_6.om layers_7_13.om config.json $USER@192.168.137.102:$REMOTE_DIR/models/

echo "Copying models to Node 1 (192.168.137.100)..."
scp layers_14_20.om layers_21_27.om output.om config.json $USER@192.168.137.100:$REMOTE_DIR/models/

echo "=== 部署完成 ==="
echo "请按以下顺序启动节点："
echo "1. SSH 到 192.168.137.100 运行 start_node1.sh (尾节点)"
echo "2. SSH 到 192.168.137.102 运行 start_node0.sh (头节点)"
```

---

## 单机测试

在部署到多台设备之前，建议先在单台香橙派上进行测试（如果内存足够）：

```bash
# 将所有模型文件复制到一台香橙派
scp *.om config.json orangepi@192.168.137.102:~/qwen_distributed/models/

# SSH 到香橙派
ssh orangepi@192.168.137.102

# 运行单机测试（需要足够内存）
cd ~/qwen_distributed/code
python3 run_single_machine.py \
    --om_dir ~/qwen_distributed/models \
    --device 0 \
    --init_tokens init_tokens.txt \
    --max_new_tokens 50
```

---

## 详细配置

### config.py 配置类

```python
from config import DistributedConfig2Nodes

config = DistributedConfig2Nodes(
    om_dir="/path/to/models",
    device_id=0,
    max_cache_len=1024,
    max_input_len=16,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    greedy=True,
    node_id=0,
)
```

---

## API 使用

### 头节点 API

```python
import numpy as np
from config import DistributedConfig2Nodes
from node_head import HeadNode

config = DistributedConfig2Nodes(
    om_dir="/path/to/models",
    device_id=0,
    greedy=True,
)

node = HeadNode(config)
node.init()

prompt_ids = np.array([[151644, 8948, 198, 2610, 525]], dtype=np.int64)
generated_ids = node.generate(prompt_ids, max_new_tokens=100)

node.finalize()
```

---

## 故障排除

### 常见问题

#### 1. 连接超时

**症状**：节点启动后卡在 "Waiting for connection..."

**解决方案**：
- 检查启动顺序：必须先启动 Node 1，再启动 Node 0
- 检查网络连通性：`ping 目标IP`
- 检查防火墙：`sudo ufw status`
- 检查端口是否被占用：`netstat -tlnp | grep 900`

#### 2. 模型加载失败

**症状**：`ACL model load failed`

**解决方案**：
- 检查 .om 文件是否存在：`ls ~/qwen_distributed/models/`
- 检查 NPU 设备：`npu-smi info`
- 检查 ACL 环境：`python3 -c "import acl; print('OK')"`

#### 3. 内存不足 / 进程被 Killed

**症状**：`Out of memory`、程序崩溃或显示 `Killed`

**原因**：NPU 内存不足以同时加载多个模型

**解决方案**：
- 本框架已采用按顺序加载模型的策略，每次只加载一个模型
- 如果仍然出现问题，尝试：
  - 减少 `max_cache_len`（如从 1024 改为 512）
  - 重启香橙派清理 NPU 内存：`sudo reboot`
  - 检查是否有其他程序占用 NPU 内存：`npu-smi info`

#### 4. 推理速度慢

**症状**：每个 token 生成时间较长

**原因**：按顺序加载模型策略需要频繁加载/卸载模型

**说明**：这是内存优化的代价。如果有更大内存的设备，可以修改代码同时加载所有模型以提高速度。

---

## 依赖

- Python 3.8+
- NumPy
- 华为昇腾 ACL SDK（pyACL）
- 香橙派昇腾开发板

---

## 许可证

MIT License
