# Qwen 4 节点分布式推理框架

基于华为昇腾 ACL 的 Qwen 模型分布式推理框架，支持 **4 节点**流水线并行推理。

## 目录

- [架构概述](#架构概述)
- [与 2 节点版本的对比](#与-2-节点版本的对比)
- [模型参数](#模型参数)
- [文件结构](#文件结构)
- [香橙派昇腾分布式部署指南](#香橙派昇腾分布式部署指南)
- [详细配置](#详细配置)
- [API 使用](#api-使用)
- [故障排除](#故障排除)

---

## 架构概述

### 4 节点流水线架构

本框架将 Qwen 模型（28 层 Transformer）切分到 **4 个设备**上进行流水线并行推理。相比 2 节点版本，4 节点版本的**模型常驻内存**，无需频繁加载/卸载，推理速度更快：

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    4 节点分布式推理数据流                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│   │    Node 0       │    │    Node 1       │    │    Node 2       │    │    Node 3       │         │
│   │   (头节点)       │    │  (中间节点1)     │    │  (中间节点2)     │    │   (尾节点)       │         │
│   │                 │    │                 │    │                 │    │                 │         │
│   │  embed.om       │    │                 │    │                 │    │                 │         │
│   │  layers_0_6.om  │───▶│ layers_7_13.om  │───▶│ layers_14_20.om │───▶│ layers_21_27.om │         │
│   │                 │    │                 │    │                 │    │  output.om      │         │
│   │                 │    │                 │    │                 │    │                 │         │
│   │  [主节点]        │    │                 │    │                 │    │                 │         │
│   │  7 层 KV Cache  │    │  7 层 KV Cache  │    │  7 层 KV Cache  │    │  7 层 KV Cache  │         │
│   └────────▲────────┘    └─────────────────┘    └─────────────────┘    └────────┬────────┘         │
│            │                                                                    │                  │
│            └─────────────────────── next_token ◀────────────────────────────────┘                  │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘

数据传输说明：
  • hidden_states: Node 0 → Node 1 → Node 2 → Node 3 (形状: [1, 16, 2048])
  • next_token:    Node 3 → Node 0 (单个 token ID)
  • KV Cache:      保存在每个设备本地，不需要网络传输
```

### 模型切分详情

| 节点 | 模型文件 | 内容 | 层数 |
|------|---------|------|------|
| **Node 0** | embed.om | Embedding 层 | - |
| **Node 0** | layers_0_6.om | Transformer 层 0-6 | 7 层 |
| **Node 1** | layers_7_13.om | Transformer 层 7-13 | 7 层 |
| **Node 2** | layers_14_20.om | Transformer 层 14-20 | 7 层 |
| **Node 3** | layers_21_27.om | Transformer 层 21-27 | 7 层 |
| **Node 3** | output.om | LM Head (输出层) | - |

### 节点职责

| 节点 | 角色 | 加载的模型 | 主要职责 |
|------|------|-----------|----------|
| **Node 0** | 头节点 | embed.om + layers_0_6.om | 接收输入 → embedding → block → 发送 hidden_states → 接收 token |
| **Node 1** | 中间节点1 | layers_7_13.om | 接收 hidden_states → block → 发送 hidden_states |
| **Node 2** | 中间节点2 | layers_14_20.om | 接收 hidden_states → block → 发送 hidden_states |
| **Node 3** | 尾节点 | layers_21_27.om + output.om | 接收 hidden_states → block → lm_head → 采样 → 发送 token |

### 内存优化：模型常驻内存

4 节点版本的核心优势是**模型常驻内存**：

```
4 节点版本（模型常驻内存）：
  • 每个节点只加载 1-2 个模型
  • 模型在初始化时加载，推理期间保持常驻
  • 无需频繁加载/卸载，推理速度更快

2 节点版本（按需加载）：
  • 每个节点需要加载 3 个模型
  • 由于内存限制，每次推理都需要加载/卸载模型
  • 推理速度较慢
```

---

## 与 2 节点版本的对比

| 特性 | 2 节点版本 | 4 节点版本 |
|------|-----------|-----------|
| 设备数量 | 2 台 | 4 台 |
| 每节点层数 | 14 层 | 7 层 |
| 每节点模型数 | 3 个 | 1-2 个 |
| 每节点 KV Cache | 14 层 | 7 层 |
| 网络通信次数 | 2 次/step | 4 次/step |
| 模型加载方式 | 按需加载/卸载 | 常驻内存 |
| 内存需求/节点 | 较高 | 较低 |
| 推理速度 | 较慢 | 较快 |
| 适用场景 | 设备有限 | 追求速度 |

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
distributed_inference_4nodes/
├── config.py              # 4 节点配置类定义
├── network.py             # TCP 网络通信模块
├── kvcache.py             # KV Cache 管理
├── acl_model.py           # ACL 模型封装（常驻内存版本）
├── utils.py               # 工具函数
│
├── node_head.py           # 头节点实现 (Node 0)
├── node_middle.py         # 中间节点实现 (Node 1, 2)
├── node_tail.py           # 尾节点实现 (Node 3)
│
└── README.md              # 本文档
```

---

## 香橙派昇腾分布式部署指南

本节详细介绍如何在 4 台香橙派昇腾开发板上部署分布式推理框架。

### 硬件准备

| 设备 | 数量 | 角色 | 需要的模型文件 |
|------|------|------|---------------|
| 香橙派昇腾 | **4 台** | 推理节点 | 见下表 |
| 开发机（电脑） | 1 台 | 远程控制 | 无（通过 SSH 控制香橙派） |
| 交换机 | 1 台 | 网络连接 | - |

**各香橙派需要的模型文件：**

| 设备 | IP 地址 | 角色 | 需要的模型文件 |
|------|---------|------|---------------|
| 香橙派 1 | 192.168.137.100 | Node 0 (头节点) | embed.om, layers_0_6.om |
| 香橙派 2 | 192.168.137.101 | Node 1 (中间节点1) | layers_7_13.om |
| 香橙派 3 | 192.168.137.102 | Node 2 (中间节点2) | layers_14_20.om |
| 香橙派 4 | 192.168.137.103 | Node 3 (尾节点) | layers_21_27.om, output.om |

### 网络连接说明

本方案使用**交换机直连**方式，所有设备（开发机 + 香橙派）通过网线连接到同一台交换机，无需路由器。

**网络连接方式：**

```
                              ┌─────────────────────────────────────────┐
                              │                 交换机                   │
                              └─────────────────────────────────────────┘
                                                │
        ┌─────────────────┬─────────────────────┼─────────────────┬─────────────────┐
        │                 │                     │                 │                 │
        ▼                 ▼                     ▼                 ▼                 ▼
   ┌────────┐        ┌────────┐           ┌────────┐        ┌────────┐        ┌────────┐
   │ 开发机  │        │香橙派1 │           │香橙派2 │        │香橙派3 │        │香橙派4 │
   │ (PC)   │        │ Node 0 │           │ Node 1 │        │ Node 2 │        │ Node 3 │
   │ (有线) │        │ (有线) │           │ (有线) │        │ (有线) │        │ (有线) │
   │192.168.│        │192.168.│           │192.168.│        │192.168.│        │192.168.│
   │137.99  │        │137.100 │           │137.101 │        │137.102 │        │137.103 │
   └────────┘        └────────┘           └────────┘        └────────┘        └────────┘
```

- **开发机**：通过网线连接到交换机，需要配置静态 IP（192.168.137.99）
- **香橙派**：通过网线连接到交换机，配置静态 IP
- **关键要求**：所有设备必须在同一网段（192.168.137.x），能够互相 ping 通
- **无需网关**：由于没有路由器，不需要配置网关

### 网络拓扑与数据流

```
                              ┌─────────────────────────────────────────┐
                              │                 交换机                   │
                              └─────────────────────────────────────────┘
                                                │
        ┌─────────────────┬─────────────────────┼─────────────────┬─────────────────┐
        │                 │                     │                 │                 │
   ┌────┴────┐       ┌────┴────┐          ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
   │ 开发机   │       │ Node 0  │          │ Node 1  │       │ Node 2  │       │ Node 3  │
   │ (PC)    │       │ 头节点   │          │中间节点1│       │中间节点2│       │ 尾节点   │
   │192.168. │       │192.168. │          │192.168. │       │192.168. │       │192.168. │
   │137.99   │       │137.100  │          │137.101  │       │137.102  │       │137.103  │
   │         │       │端口:9000│          │端口:9001│       │端口:9002│       │端口:9003│
   └─────────┘       └────┬────┘          └────┬────┘       └────┬────┘       └────┬────┘
                          │                    │                 │                 │
                          │    hidden_states   │  hidden_states  │  hidden_states  │
                          └───────────────────▶└────────────────▶└────────────────▶│
                          │                                                        │
                          │◀───────────────────── next_token ◀─────────────────────┘
                          │
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

# === 香橙派 1 (Node 0 - 192.168.137.100) ===
# 清除 eth0 的 IP
sudo ip addr flush dev eth0

# 设置新 IP
sudo ip addr add 192.168.137.100/24 dev eth0

# 启用接口
sudo ip link set eth0 up

# 检查设置
ip addr show eth0

# === 香橙派 2 (Node 1 - 192.168.137.101) ===
sudo ip addr flush dev eth0
sudo ip addr add 192.168.137.101/24 dev eth0
sudo ip link set eth0 up
ip addr show eth0

# === 香橙派 3 (Node 2 - 192.168.137.102) ===
sudo ip addr flush dev eth0
sudo ip addr add 192.168.137.102/24 dev eth0
sudo ip link set eth0 up
ip addr show eth0

# === 香橙派 4 (Node 3 - 192.168.137.103) ===
sudo ip addr flush dev eth0
sudo ip addr add 192.168.137.103/24 dev eth0
sudo ip link set eth0 up
ip addr show eth0
```

> **注意**：`ip` 命令设置的 IP 是临时的，系统重启后会丢失。适合临时测试使用。

##### 方式二：使用 nmcli 命令（持久化配置，重启后保留）

```bash
# 香橙派 1 (Node 0 - 192.168.137.100)
sudo nmcli con mod "Wired connection 1" ipv4.addresses 192.168.137.100/24
sudo nmcli con mod "Wired connection 1" ipv4.method manual
sudo nmcli con mod "Wired connection 1" ipv4.gateway ""
sudo nmcli con up "Wired connection 1"

# 香橙派 2 (Node 1 - 192.168.137.101)
sudo nmcli con mod "Wired connection 1" ipv4.addresses 192.168.137.101/24
sudo nmcli con mod "Wired connection 1" ipv4.method manual
sudo nmcli con mod "Wired connection 1" ipv4.gateway ""
sudo nmcli con up "Wired connection 1"

# 香橙派 3 (Node 2 - 192.168.137.102)
sudo nmcli con mod "Wired connection 1" ipv4.addresses 192.168.137.102/24
sudo nmcli con mod "Wired connection 1" ipv4.method manual
sudo nmcli con mod "Wired connection 1" ipv4.gateway ""
sudo nmcli con up "Wired connection 1"

# 香橙派 4 (Node 3 - 192.168.137.103)
sudo nmcli con mod "Wired connection 1" ipv4.addresses 192.168.137.103/24
sudo nmcli con mod "Wired connection 1" ipv4.method manual
sudo nmcli con mod "Wired connection 1" ipv4.gateway ""
sudo nmcli con up "Wired connection 1"

# 开发机 (PC) - Windows 系统需在网络设置中手动配置静态 IP: 192.168.137.99
# 或使用命令行（以管理员身份运行）：
# netsh interface ip set address "以太网" static 192.168.137.99 255.255.255.0
```

> **注意**：
> - 连接名称为 "Wired connection 1"，可用 `nmcli con show` 查看实际名称
> - 由于使用交换机直连无路由器，网关设置为空
> - 开发机也需要配置同网段的静态 IP 才能 SSH 连接到香橙派

#### 1.2 验证网络连通性

在每台香橙派上测试与其他节点的连接：

```bash
# 从 Node 0 (192.168.137.100) 测试
ping 192.168.137.101  # Node 1
ping 192.168.137.102  # Node 2
ping 192.168.137.103  # Node 3

# 从 Node 1 (192.168.137.101) 测试
ping 192.168.137.100  # Node 0
ping 192.168.137.102  # Node 2
ping 192.168.137.103  # Node 3

# 从 Node 2 (192.168.137.102) 测试
ping 192.168.137.100  # Node 0
ping 192.168.137.101  # Node 1
ping 192.168.137.103  # Node 3

# 从 Node 3 (192.168.137.103) 测试
ping 192.168.137.100  # Node 0
ping 192.168.137.101  # Node 1
ping 192.168.137.102  # Node 2
```

#### 1.3 开放防火墙端口

在每台香橙派上开放对应端口：

```bash
# 香橙派 1 (Node 0 - 192.168.137.100)
sudo ufw allow 9000/tcp

# 香橙派 2 (Node 1 - 192.168.137.101)
sudo ufw allow 9001/tcp

# 香橙派 3 (Node 2 - 192.168.137.102)
sudo ufw allow 9002/tcp

# 香橙派 4 (Node 3 - 192.168.137.103)
sudo ufw allow 9003/tcp

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

将 `distributed_inference_4nodes/` 目录下的所有 Python 文件复制到每台香橙派：

```bash
# 在开发机上执行（假设香橙派用户名为 orangepi）
scp *.py orangepi@192.168.137.100:~/qwen_distributed/code/
scp *.py orangepi@192.168.137.101:~/qwen_distributed/code/
scp *.py orangepi@192.168.137.102:~/qwen_distributed/code/
scp *.py orangepi@192.168.137.103:~/qwen_distributed/code/
```

### 步骤 3：分发模型文件

将对应的 .om 模型文件复制到各香橙派：

```bash
# 香橙派 1 (Node 0 - 192.168.137.100) - 需要 embed + layers_0_6
scp embed.om layers_0_6.om config.json orangepi@192.168.137.100:~/qwen_distributed/models/

# 香橙派 2 (Node 1 - 192.168.137.101) - 需要 layers_7_13
scp layers_7_13.om config.json orangepi@192.168.137.101:~/qwen_distributed/models/

# 香橙派 3 (Node 2 - 192.168.137.102) - 需要 layers_14_20
scp layers_14_20.om config.json orangepi@192.168.137.102:~/qwen_distributed/models/

# 香橙派 4 (Node 3 - 192.168.137.103) - 需要 layers_21_27 + output
scp layers_21_27.om output.om config.json orangepi@192.168.137.103:~/qwen_distributed/models/
```

### 步骤 4：准备输入文件

在香橙派 1 (Node 0) 上创建输入 token 文件：

```bash
# SSH 到香橙派 1 (Node 0)
ssh orangepi@192.168.137.100

# 创建输入 token 文件
cd ~/qwen_distributed/code
echo "151644 8948 198 2610 525 264 10950 17847 13" > init_tokens.txt
```

> **说明**：这些 token ID 对应 Qwen 的 tokenizer 编码结果。你可以使用 transformers 库预先编码你的 prompt。

### 步骤 5：启动分布式推理

**重要**：必须按照以下顺序启动节点！

```
启动顺序：Node 3 → Node 2 → Node 1 → Node 0
         (尾节点)  (中间2)  (中间1)  (头节点)
```

> **操作说明**：你需要在你的**开发机（电脑）**上打开 **4 个终端窗口**，分别通过 SSH 连接到 4 台香橙派。每个终端窗口控制一台香橙派。

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      开发机（你的电脑）                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│   ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐       │
│   │    终端窗口 1      │  │    终端窗口 2      │  │    终端窗口 3      │  │    终端窗口 4      │       │
│   │                   │  │                   │  │                   │  │                   │       │
│   │  SSH 连接到       │  │  SSH 连接到       │  │  SSH 连接到       │  │  SSH 连接到       │       │
│   │  香橙派 1         │  │  香橙派 2         │  │  香橙派 3         │  │  香橙派 4         │       │
│   │  (Node 0-头节点)  │  │  (Node 1-中间1)   │  │  (Node 2-中间2)   │  │  (Node 3-尾节点)  │       │
│   │  192.168.137.100  │  │  192.168.137.101  │  │  192.168.137.102  │  │  192.168.137.103  │       │
│   │                   │  │                   │  │                   │  │                   │       │
│   │  第四启动 ④       │  │  第三启动 ③       │  │  第二启动 ②       │  │  首先启动 ①       │       │
│   └───────────────────┘  └───────────────────┘  └───────────────────┘  └───────────────────┘       │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 5.1 【终端窗口 4】启动 Node 3（香橙派 4 - 尾节点）- 首先启动！

在你的**开发机**上打开一个新的终端窗口，执行：

```bash
# 从开发机 SSH 连接到香橙派 4 (Node 3 - 尾节点)
ssh orangepi@192.168.137.103
```

连接成功后，在香橙派 4 上执行：

```bash
# 进入代码目录
cd ~/qwen_distributed/code

# 启动尾节点
python3 node_tail.py \
    --om_dir ~/qwen_distributed/models \
    --device 0 \
    --max_cache_len 1024 \
    --max_input_len 16 \
    --listen_port 9003 \
    --head_ip 192.168.137.100 \
    --head_port 9000 \
    --greedy
```

等待看到以下输出后，**保持此终端窗口运行**，然后继续下一步：
```
[Node3-Tail] Initializing...
[Node3-Tail] Loading models...
[ACLModel] Loaded: .../layers_21_27.om
[ACLModel] Loaded: .../output.om
[Node3-Tail] KV Cache initialized for 7 layers
[Node3-Tail] Waiting for previous node connection...
```

> **注意**：4 节点版本的模型在初始化时就加载完成，并常驻内存。

#### 5.2 【终端窗口 3】启动 Node 2（香橙派 3 - 中间节点2）- 第二启动！

在你的**开发机**上打开**另一个新的终端窗口**，执行：

```bash
# 从开发机 SSH 连接到香橙派 3 (Node 2 - 中间节点2)
ssh orangepi@192.168.137.102
```

连接成功后，在香橙派 3 上执行：

```bash
# 进入代码目录
cd ~/qwen_distributed/code

# 启动中间节点2
python3 node_middle.py \
    --node_id 2 \
    --om_dir ~/qwen_distributed/models \
    --device 0 \
    --max_cache_len 1024 \
    --max_input_len 16 \
    --listen_port 9002 \
    --next_ip 192.168.137.103 \
    --next_port 9003
```

等待看到以下输出后，**保持此终端窗口运行**，然后继续下一步：
```
[Node2-Middle] Initializing...
[Node2-Middle] Loading model: .../layers_14_20.om
[ACLModel] Loaded: .../layers_14_20.om
[Node2-Middle] KV Cache initialized for 7 layers
[Node2-Middle] Waiting for previous node connection...
```

#### 5.3 【终端窗口 2】启动 Node 1（香橙派 2 - 中间节点1）- 第三启动！

在你的**开发机**上打开**另一个新的终端窗口**，执行：

```bash
# 从开发机 SSH 连接到香橙派 2 (Node 1 - 中间节点1)
ssh orangepi@192.168.137.101
```

连接成功后，在香橙派 2 上执行：

```bash
# 进入代码目录
cd ~/qwen_distributed/code

# 启动中间节点1
python3 node_middle.py \
    --node_id 1 \
    --om_dir ~/qwen_distributed/models \
    --device 0 \
    --max_cache_len 1024 \
    --max_input_len 16 \
    --listen_port 9001 \
    --next_ip 192.168.137.102 \
    --next_port 9002
```

等待看到以下输出后，**保持此终端窗口运行**，然后继续下一步：
```
[Node1-Middle] Initializing...
[Node1-Middle] Loading model: .../layers_7_13.om
[ACLModel] Loaded: .../layers_7_13.om
[Node1-Middle] KV Cache initialized for 7 layers
[Node1-Middle] Waiting for previous node connection...
```

#### 5.4 【终端窗口 1】启动 Node 0（香橙派 1 - 头节点）- 最后启动！

在你的**开发机**上打开**另一个新的终端窗口**，执行：

```bash
# 从开发机 SSH 连接到香橙派 1 (Node 0 - 头节点)
ssh orangepi@192.168.137.100
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
    --next_ip 192.168.137.101 \
    --next_port 9001 \
    --greedy
```

启动后，你应该看到推理开始运行：
```
[Node0-Head] Initializing...
[Node0-Head] Loading models...
[ACLModel] Loaded: .../embed.om
[ACLModel] Loaded: .../layers_0_6.om
[Node0-Head] KV Cache initialized for 7 layers
[Node0-Head] Waiting for tail node connection...
[Node0-Head] Initialization complete!
[Node0-Head] Step 0: generated token 12345
[Node0-Head] Step 1: generated token 67890
...
```

> **提示**：此时你可以观察 4 个终端窗口，每个节点都会打印自己的处理日志。

### 步骤 6：查看结果

推理完成后，Node 0 会输出：
```
Generated 100 tokens in 15.23s
Speed: 6.57 tokens/s
Generated IDs: [12345, 67890, ...]
```

> **注意**：4 节点版本由于模型常驻内存，推理速度比 2 节点版本快很多。

---

## 快速启动脚本

为了方便部署，可以在每台香橙派上创建启动脚本：

### 香橙派 1 (Node 0 - 192.168.137.100) - start_node0.sh

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
    --next_ip 192.168.137.101 \
    --next_port 9001 \
    --greedy
```

### 香橙派 2 (Node 1 - 192.168.137.101) - start_node1.sh

```bash
#!/bin/bash
cd ~/qwen_distributed/code
python3 node_middle.py \
    --node_id 1 \
    --om_dir ~/qwen_distributed/models \
    --device 0 \
    --max_cache_len 1024 \
    --max_input_len 16 \
    --listen_port 9001 \
    --next_ip 192.168.137.102 \
    --next_port 9002
```

### 香橙派 3 (Node 2 - 192.168.137.102) - start_node2.sh

```bash
#!/bin/bash
cd ~/qwen_distributed/code
python3 node_middle.py \
    --node_id 2 \
    --om_dir ~/qwen_distributed/models \
    --device 0 \
    --max_cache_len 1024 \
    --max_input_len 16 \
    --listen_port 9002 \
    --next_ip 192.168.137.103 \
    --next_port 9003
```

### 香橙派 4 (Node 3 - 192.168.137.103) - start_node3.sh

```bash
#!/bin/bash
cd ~/qwen_distributed/code
python3 node_tail.py \
    --om_dir ~/qwen_distributed/models \
    --device 0 \
    --max_cache_len 1024 \
    --max_input_len 16 \
    --listen_port 9003 \
    --head_ip 192.168.137.100 \
    --head_port 9000 \
    --greedy
```

### 创建并设置脚本权限

在每台香橙派上：

```bash
# 创建脚本（以 Node 0 为例）
cat > ~/qwen_distributed/code/start_node0.sh << 'EOF'
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
    --next_ip 192.168.137.101 \
    --next_port 9001 \
    --greedy
EOF

# 设置执行权限
chmod +x ~/qwen_distributed/code/start_node0.sh
```

---

## 一键部署脚本（在开发机上运行）

创建 `deploy_4nodes.sh`：

```bash
#!/bin/bash

# 配置
NODES=("192.168.137.100" "192.168.137.101" "192.168.137.102" "192.168.137.103")
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
echo "Copying models to Node 0 (192.168.137.100)..."
scp embed.om layers_0_6.om config.json $USER@192.168.137.100:$REMOTE_DIR/models/

echo "Copying models to Node 1 (192.168.137.101)..."
scp layers_7_13.om config.json $USER@192.168.137.101:$REMOTE_DIR/models/

echo "Copying models to Node 2 (192.168.137.102)..."
scp layers_14_20.om config.json $USER@192.168.137.102:$REMOTE_DIR/models/

echo "Copying models to Node 3 (192.168.137.103)..."
scp layers_21_27.om output.om config.json $USER@192.168.137.103:$REMOTE_DIR/models/

echo "=== 步骤 4: 创建启动脚本 ==="

# Node 0 启动脚本
ssh $USER@192.168.137.100 "cat > $REMOTE_DIR/code/start.sh << 'EOF'
#!/bin/bash
cd ~/qwen_distributed/code
python3 node_head.py \\
    --om_dir ~/qwen_distributed/models \\
    --device 0 \\
    --max_cache_len 1024 \\
    --max_input_len 16 \\
    --init_tokens init_tokens.txt \\
    --max_new_tokens 100 \\
    --listen_port 9000 \\
    --next_ip 192.168.137.101 \\
    --next_port 9001 \\
    --greedy
EOF
chmod +x $REMOTE_DIR/code/start.sh"

# Node 1 启动脚本
ssh $USER@192.168.137.101 "cat > $REMOTE_DIR/code/start.sh << 'EOF'
#!/bin/bash
cd ~/qwen_distributed/code
python3 node_middle.py \\
    --node_id 1 \\
    --om_dir ~/qwen_distributed/models \\
    --device 0 \\
    --max_cache_len 1024 \\
    --max_input_len 16 \\
    --listen_port 9001 \\
    --next_ip 192.168.137.102 \\
    --next_port 9002
EOF
chmod +x $REMOTE_DIR/code/start.sh"

# Node 2 启动脚本
ssh $USER@192.168.137.102 "cat > $REMOTE_DIR/code/start.sh << 'EOF'
#!/bin/bash
cd ~/qwen_distributed/code
python3 node_middle.py \\
    --node_id 2 \\
    --om_dir ~/qwen_distributed/models \\
    --device 0 \\
    --max_cache_len 1024 \\
    --max_input_len 16 \\
    --listen_port 9002 \\
    --next_ip 192.168.137.103 \\
    --next_port 9003
EOF
chmod +x $REMOTE_DIR/code/start.sh"

# Node 3 启动脚本
ssh $USER@192.168.137.103 "cat > $REMOTE_DIR/code/start.sh << 'EOF'
#!/bin/bash
cd ~/qwen_distributed/code
python3 node_tail.py \\
    --om_dir ~/qwen_distributed/models \\
    --device 0 \\
    --max_cache_len 1024 \\
    --max_input_len 16 \\
    --listen_port 9003 \\
    --head_ip 192.168.137.100 \\
    --head_port 9000 \\
    --greedy
EOF
chmod +x $REMOTE_DIR/code/start.sh"

echo "=== 部署完成 ==="
echo ""
echo "请按以下顺序启动节点："
echo "1. SSH 到 192.168.137.103 运行 ./start.sh (Node 3 - 尾节点)"
echo "2. SSH 到 192.168.137.102 运行 ./start.sh (Node 2 - 中间节点2)"
echo "3. SSH 到 192.168.137.101 运行 ./start.sh (Node 1 - 中间节点1)"
echo "4. SSH 到 192.168.137.100 运行 ./start.sh (Node 0 - 头节点)"
```

---

## 详细配置

### config.py 配置类

```python
from config import DistributedConfig4Nodes

config = DistributedConfig4Nodes(
    om_dir="/path/to/models",
    device_id=0,
    max_cache_len=1024,
    max_input_len=16,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    greedy=True,
    node_id=0,  # 0=头节点, 1=中间1, 2=中间2, 3=尾节点
)
```

### 命令行参数说明

#### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--om_dir` | OM 模型目录 | 必填 |
| `--device` | NPU 设备 ID | 0 |
| `--max_cache_len` | KV Cache 最大长度 | 1024 |
| `--max_input_len` | 单次输入最大长度 | 16 |

#### 头节点 (Node 0) 特有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--init_tokens` | 初始 token 文件 | 必填 |
| `--max_new_tokens` | 最大生成 token 数 | 100 |
| `--temperature` | 采样温度 | 1.0 |
| `--top_k` | Top-K 采样 | 0 |
| `--top_p` | Top-P 采样 | 1.0 |
| `--greedy` | 贪婪采样 | True |
| `--listen_port` | 监听端口 | 9000 |
| `--next_ip` | 下一节点 IP | 192.168.137.101 |
| `--next_port` | 下一节点端口 | 9001 |

#### 中间节点 (Node 1, 2) 特有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--node_id` | 节点 ID (1 或 2) | 必填 |
| `--listen_port` | 监听端口 | 9000 + node_id |
| `--next_ip` | 下一节点 IP | 自动设置 |
| `--next_port` | 下一节点端口 | 9000 + node_id + 1 |

#### 尾节点 (Node 3) 特有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--listen_port` | 监听端口 | 9003 |
| `--head_ip` | 头节点 IP | 192.168.137.100 |
| `--head_port` | 头节点端口 | 9000 |

---

## API 使用

### 头节点 API

```python
import numpy as np
from config import DistributedConfig4Nodes
from node_head import HeadNode4Nodes

config = DistributedConfig4Nodes(
    om_dir="/path/to/models",
    device_id=0,
    greedy=True,
    node_id=0,
)

node = HeadNode4Nodes(config)
node.init()

prompt_ids = np.array([[151644, 8948, 198, 2610, 525]], dtype=np.int64)
generated_ids = node.generate(prompt_ids, max_new_tokens=100)

node.shutdown()
```

### 中间节点 API

```python
from config import DistributedConfig4Nodes
from node_middle import MiddleNode4Nodes

config = DistributedConfig4Nodes(
    om_dir="/path/to/models",
    device_id=0,
    node_id=1,  # 或 2
)

node = MiddleNode4Nodes(config)
node.init()
node.process_loop()  # 阻塞，等待处理请求
node.shutdown()
```

### 尾节点 API

```python
from config import DistributedConfig4Nodes
from node_tail import TailNode4Nodes

config = DistributedConfig4Nodes(
    om_dir="/path/to/models",
    device_id=0,
    greedy=True,
    node_id=3,
)

node = TailNode4Nodes(config)
node.init()
node.process_loop()  # 阻塞，等待处理请求
node.shutdown()
```

---

## 故障排除

### 常见问题

#### 1. 连接超时

**症状**：节点启动后卡在 "Waiting for connection..."

**解决方案**：
- 检查启动顺序：必须按 Node 3 → Node 2 → Node 1 → Node 0 的顺序启动
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

**原因**：NPU 内存不足

**解决方案**：
- 4 节点版本每个节点只加载 1-2 个模型，内存需求较低
- 如果仍然出现问题，尝试：
  - 减少 `max_cache_len`（如从 1024 改为 512）
  - 重启香橙派清理 NPU 内存：`sudo reboot`
  - 检查是否有其他程序占用 NPU 内存：`npu-smi info`

#### 4. 网络传输错误

**症状**：`Connection reset by peer` 或 `Broken pipe`

**解决方案**：
- 检查网络稳定性
- 确保所有节点都在运行
- 检查交换机连接是否正常

#### 5. 推理结果异常

**症状**：生成的 token 不正确或重复

**解决方案**：
- 检查所有节点的 `max_cache_len` 和 `max_input_len` 参数是否一致
- 确保模型文件正确分发到各节点
- 尝试使用 `--greedy` 参数进行贪婪采样

---

## 性能优化建议

1. **使用千兆网络**：确保开发板之间使用千兆以太网连接，减少网络延迟
2. **减少日志输出**：在生产环境中减少 print 语句，可以提高性能
3. **调整 KV Cache 大小**：根据实际需求调整 `max_cache_len`，过大会占用更多内存
4. **使用 greedy 采样**：贪婪采样比随机采样更快
5. **预热模型**：首次推理可能较慢，后续推理会更快

---

## 依赖

- Python 3.8+
- NumPy
- 华为昇腾 ACL SDK（pyACL）
- 香橙派昇腾开发板 × 4

---

## 许可证

MIT License
