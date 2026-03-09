#!/bin/bash

# Llama 分布式推理部署脚本（单机测试）
# 按照从尾到头的顺序启动节点

set -e

# 配置参数
OM_DIR="./om_models"
DEVICE=0
MAX_CACHE_LEN=1024
MAX_INPUT_LEN=16
MAX_NEW_TOKENS=100
INIT_TOKENS="init_tokens.txt"

# 网络配置
NODE0_PORT=8000
NODE1_PORT=8001
NODE2_PORT=8002
NODE3_PORT=8003
HOST="127.0.0.1"

# 日志目录
LOG_DIR="./logs"
mkdir -p $LOG_DIR

echo "=========================================="
echo "Llama 分布式推理部署"
echo "=========================================="
echo "模型目录: $OM_DIR"
echo "设备 ID: $DEVICE"
echo "最大缓存长度: $MAX_CACHE_LEN"
echo "最大输入长度: $MAX_INPUT_LEN"
echo "最大生成 tokens: $MAX_NEW_TOKENS"
echo "=========================================="

# 检查模型文件
echo "检查模型文件..."
if [ ! -d "$OM_DIR" ]; then
    echo "错误: 模型目录不存在: $OM_DIR"
    exit 1
fi

required_models=(
    "llama_m0_embed_layers_0_4.om"
    "llama_m1_layers_5_10.om"
    "llama_m2_layers_11_16.om"
    "llama_m3_layers_17_21_lmhead.om"
)

for model in "${required_models[@]}"; do
    if [ ! -f "$OM_DIR/$model" ]; then
        echo "错误: 模型文件不存在: $OM_DIR/$model"
        exit 1
    fi
done

echo "✓ 所有模型文件存在"

# 检查初始 tokens 文件
if [ ! -f "$INIT_TOKENS" ]; then
    echo "警告: 初始 tokens 文件不存在: $INIT_TOKENS"
    echo "创建示例文件..."
    echo "1 2 3 4 5" > $INIT_TOKENS
    echo "✓ 已创建示例文件"
fi

# 清理旧进程
echo ""
echo "清理旧进程..."
pkill -f "node_tail.py" || true
pkill -f "node_middle.py" || true
pkill -f "node_head.py" || true
sleep 2

# 启动 Node 3 (尾节点)
echo ""
echo "启动 Node 3 (尾节点)..."
python node_tail.py \
    --om_dir $OM_DIR \
    --device $DEVICE \
    --max_cache_len $MAX_CACHE_LEN \
    --max_input_len $MAX_INPUT_LEN \
    --listen_port $NODE3_PORT \
    --head_ip $HOST \
    --head_port $NODE0_PORT \
    --greedy \
    > $LOG_DIR/node3.log 2>&1 &

NODE3_PID=$!
echo "✓ Node 3 已启动 (PID: $NODE3_PID)"
sleep 3

# 启动 Node 2 (中间节点)
echo ""
echo "启动 Node 2 (中间节点)..."
python node_middle.py \
    --node_id 2 \
    --om_dir $OM_DIR \
    --device $DEVICE \
    --max_cache_len $MAX_CACHE_LEN \
    --max_input_len $MAX_INPUT_LEN \
    --listen_port $NODE2_PORT \
    --next_ip $HOST \
    --next_port $NODE3_PORT \
    > $LOG_DIR/node2.log 2>&1 &

NODE2_PID=$!
echo "✓ Node 2 已启动 (PID: $NODE2_PID)"
sleep 3

# 启动 Node 1 (中间节点)
echo ""
echo "启动 Node 1 (中间节点)..."
python node_middle.py \
    --node_id 1 \
    --om_dir $OM_DIR \
    --device $DEVICE \
    --max_cache_len $MAX_CACHE_LEN \
    --max_input_len $MAX_INPUT_LEN \
    --listen_port $NODE1_PORT \
    --next_ip $HOST \
    --next_port $NODE2_PORT \
    > $LOG_DIR/node1.log 2>&1 &

NODE1_PID=$!
echo "✓ Node 1 已启动 (PID: $NODE1_PID)"
sleep 3

# 启动 Node 0 (头节点)
echo ""
echo "启动 Node 0 (头节点)..."
echo "=========================================="
python node_head.py \
    --om_dir $OM_DIR \
    --device $DEVICE \
    --max_cache_len $MAX_CACHE_LEN \
    --max_input_len $MAX_INPUT_LEN \
    --init_tokens $INIT_TOKENS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --listen_port $NODE0_PORT \
    --next_ip $HOST \
    --next_port $NODE1_PORT \
    --greedy

# 等待头节点完成
wait

echo ""
echo "=========================================="
echo "推理完成"
echo "=========================================="

# 清理后台进程
echo "清理后台进程..."
kill $NODE1_PID $NODE2_PID $NODE3_PID 2>/dev/null || true

echo "✓ 所有节点已关闭"
echo ""
echo "日志文件位于: $LOG_DIR/"
echo "  - node1.log"
echo "  - node2.log"
echo "  - node3.log"
