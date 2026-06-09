#!/bin/bash
#
# DoLa + SGLang + KTransformers 一键运行脚本
#
# 用法:
#   ./run_server.sh [options]
#
# 示例:
#   ./run_server.sh --model qwen3-8b --mode baseline
#   ./run_server.sh --model qwen3-8b --mode dola --dola-layers low
#   ./run_server.sh --model qwen3.5-122b --mode dola --dola-layers low --gpus 0,1
#
# 选项:
#   --model          模型类型: qwen3-8b, qwen3.5-122b, qwen3.5-397b (默认: qwen3-8b)
#   --mode           运行模式: baseline, dola (默认: baseline)
#   --dola-layers    DoLa 层配置: low, mid, high, 或自定义如 "0,2,4,6,8" (默认: low)
#   --dola-rt        DoLa relative_top 值: 0.0 用于 MC2 评测, 0.1 用于开放生成 (默认: 0.0)
#   --gpus           GPU 设备 ID (默认: 0)
#   --port           服务端口 (默认: 30000)
#   --model-path     模型路径 (可选，不指定则使用默认路径)
#   --help           显示帮助信息
#

set -e

# ==================== 默认配置 ====================
DEFAULT_MODEL="qwen3-8b"
DEFAULT_MODE="baseline"
DEFAULT_DOLA_LAYERS="low"
DEFAULT_DOLA_RT="0.0"
DEFAULT_GPUS="0"
DEFAULT_PORT="30000"

# 模型默认路径
QWEN3_8B_PATH="/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/default"
QWEN35_122B_PATH="/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/default"
QWEN35_397B_PATH="/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/default"

# DoLa 层配置预设
# Qwen3-8B: 36 层
DOLA_LOW_QWEN3="0,2,4,6,8,10,12,14,16"
DOLA_MID_QWEN3="12,14,16,18,20,22,24,26,28"
DOLA_HIGH_QWEN3="18,20,22,24,26,28,30,32,34"

# Qwen3.5-122B: 122 层
DOLA_LOW_QWEN35_122B="0,2,4,6,8,10,12,14,16,18,20"
DOLA_MID_QWEN35_122B="40,44,48,52,56,60,64,68,72,76,80"
DOLA_HIGH_QWEN35_122B="102,104,106,108,110,112,114,116,118,120"

# Qwen3.5-397B: 397 层
DOLA_LOW_QWEN35_397B="0,2,4,6,8,10,12,14,16,18,20"
DOLA_MID_QWEN35_397B="180,185,190,195,200,205,210,215,220"
DOLA_HIGH_QWEN35_397B="370,375,380,385,390,395"

# ==================== 帮助信息 ====================
show_help() {
    cat << EOF
DoLa + SGLang + KTransformers 一键运行脚本

用法:
  $0 [options]

选项:
  --model          模型类型: qwen3-8b, qwen3.5-122b, qwen3.5-397b (默认: $DEFAULT_MODEL)
  --mode           运行模式: baseline, dola (默认: $DEFAULT_MODE)
  --dola-layers    DoLa 层配置: low, mid, high, 或自定义如 "0,2,4,6,8" (默认: $DEFAULT_DOLA_LAYERS)
  --dola-rt        DoLa relative_top 值: 0.0 用于 MC2 评测, 0.1 用于开放生成 (默认: $DEFAULT_DOLA_RT)
  --gpus           GPU 设备 ID (默认: $DEFAULT_GPUS)
  --port           服务端口 (默认: $DEFAULT_PORT)
  --model-path     模型路径 (可选，不指定则使用默认路径)
  --help           显示帮助信息

示例:
  # Qwen3-8B Baseline 服务 (单卡)
  $0 --model qwen3-8b --mode baseline

  # Qwen3-8B DoLa 服务 (low 层配置)
  $0 --model qwen3-8b --mode dola --dola-layers low

  # Qwen3.5-122B DoLa 服务 (双卡, 开放生成)
  $0 --model qwen3.5-122b --mode dola --dola-layers low --dola-rt 0.1 --gpus 0,1

  # Qwen3.5-397B 纯 CPU DoLa 服务
  $0 --model qwen3.5-397b --mode dola --dola-layers low --gpus 0

  # 自定义模型路径
  $0 --model qwen3-8b --mode dola --model-path /path/to/model

DoLa 层配置说明:
  - low:  使用模型前半层（适合 Qwen 系列）
  - mid:  使用模型中间层
  - high: 使用模型后半层（适合 LLaMA 系列）
  - 自定义: 直接传入层索引，如 "0,2,4,6,8,10"

EOF
    exit 0
}

# ==================== 参数解析 ====================
MODEL="$DEFAULT_MODEL"
MODE="$DEFAULT_MODE"
DOLA_LAYERS_CONFIG="$DEFAULT_DOLA_LAYERS"
DOLA_RT="$DEFAULT_DOLA_RT"
GPUS="$DEFAULT_GPUS"
PORT="$DEFAULT_PORT"
MODEL_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --dola-layers)
            DOLA_LAYERS_CONFIG="$2"
            shift 2
            ;;
        --dola-rt)
            DOLA_RT="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# ==================== 参数校验 ====================
if [[ "$MODE" != "baseline" && "$MODE" != "dola" ]]; then
    echo "错误: --mode 必须是 baseline 或 dola"
    exit 1
fi

if [[ "$MODEL" != "qwen3-8b" && "$MODEL" != "qwen3.5-122b" && "$MODEL" != "qwen3.5-397b" ]]; then
    echo "错误: --model 必须是 qwen3-8b, qwen3.5-122b 或 qwen3.5-397b"
    exit 1
fi

# ==================== 设置模型路径 ====================
if [[ -z "$MODEL_PATH" ]]; then
    case $MODEL in
        qwen3-8b)
            MODEL_PATH="$QWEN3_8B_PATH"
            ;;
        qwen3.5-122b)
            MODEL_PATH="$QWEN35_122B_PATH"
            ;;
        qwen3.5-397b)
            MODEL_PATH="$QWEN35_397B_PATH"
            ;;
    esac
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "警告: 模型路径不存在: $MODEL_PATH"
    echo "请使用 --model-path 指定正确的模型路径"
fi

# ==================== 设置 DoLa 层配置 ====================
get_dola_layers() {
    local model=$1
    local config=$2

    # 如果是自定义配置（包含逗号或数字），直接返回
    if [[ "$config" =~ ^[0-9,]+$ ]]; then
        echo "$config"
        return
    fi

    case $model in
        qwen3-8b)
            case $config in
                low)  echo "$DOLA_LOW_QWEN3" ;;
                mid)  echo "$DOLA_MID_QWEN3" ;;
                high) echo "$DOLA_HIGH_QWEN3" ;;
                *)    echo "$DOLA_LOW_QWEN3" ;;
            esac
            ;;
        qwen3.5-122b)
            case $config in
                low)  echo "$DOLA_LOW_QWEN35_122B" ;;
                mid)  echo "$DOLA_MID_QWEN35_122B" ;;
                high) echo "$DOLA_HIGH_QWEN35_122B" ;;
                *)    echo "$DOLA_LOW_QWEN35_122B" ;;
            esac
            ;;
        qwen3.5-397b)
            case $config in
                low)  echo "$DOLA_LOW_QWEN35_397B" ;;
                mid)  echo "$DOLA_MID_QWEN35_397B" ;;
                high) echo "$DOLA_HIGH_QWEN35_397B" ;;
                *)    echo "$DOLA_LOW_QWEN35_397B" ;;
            esac
            ;;
    esac
}

# ==================== 环境准备 ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 激活 conda 环境
echo "=========================================="
echo "DoLa + SGLang + KTransformers 服务启动"
echo "=========================================="
echo ""
echo "配置信息:"
echo "  模型:       $MODEL"
echo "  模式:       $MODE"
echo "  GPU(s):     $GPUS"
echo "  端口:       $PORT"
echo "  模型路径:   $MODEL_PATH"

# 初始化 conda
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate serve

# 检查 DoLa patch 状态
PYTHON_DIR=$(python -c "import sglang; import os; print(os.path.dirname(sglang.__file__))")
if grep -q "_dola_contrast_logits" "$PYTHON_DIR/srt/layers/logits_processor.py" 2>/dev/null; then
    echo "  DoLa Patch: 已应用 ✓"
else
    echo "  DoLa Patch: 未应用，正在应用..."
    cd "$SCRIPT_DIR/sglang_patch"
    python apply_patch.py --apply
    cd "$SCRIPT_DIR"
fi

echo ""

# ==================== 构建启动命令 ====================
if [[ "$MODE" == "dola" ]]; then
    DOLA_LAYERS=$(get_dola_layers "$MODEL" "$DOLA_LAYERS_CONFIG")
    echo "DoLa 配置:"
    echo "  候选层:     $DOLA_LAYERS"
    echo "  RelativeTop: $DOLA_RT"
    export DOLA_LAYERS="$DOLA_LAYERS"
    export DOLA_RELATIVE_TOP="$DOLA_RT"
fi

# 计算使用的 GPU 数量
GPU_COUNT=$(echo "$GPUS" | tr ',' '\n' | wc -l)

echo ""
echo "启动命令:"
echo "-------------------------------------------"

case $MODEL in
    qwen3-8b)
        # Qwen3-8B 单卡配置
        CMD="CUDA_VISIBLE_DEVICES=$GPUS python -m sglang.launch_server \\
  --model-path $MODEL_PATH \\
  --tensor-parallel-size 1 \\
  --host 0.0.0.0 \\
  --port $PORT \\
  --trust-remote-code \\
  --attention-backend flashinfer \\
  --mem-fraction-static 0.90 \\
  --max-running-requests 4 \\
  --max-total-tokens 16384 \\
  --disable-cuda-graph"
        ;;

    qwen3.5-122b)
        # Qwen3.5-122B KTransformers 双卡配置
        CMD="CUDA_VISIBLE_DEVICES=$GPUS SGLANG_DISABLE_CUDNN_CHECK=1 \\
  python -m sglang.launch_server \\
  --model-path $MODEL_PATH \\
  --tensor-parallel-size $GPU_COUNT \\
  --host 0.0.0.0 \\
  --port $PORT \\
  --trust-remote-code \\
  --kt-weight-path $MODEL_PATH \\
  --kt-method BF16 \\
  --kt-cpuinfer 64 \\
  --kt-threadpool-count 2 \\
  --kt-numa-nodes 0 1 \\
  --kt-num-gpu-experts 1 \\
  --kt-gpu-prefill-token-threshold 4096 \\
  --kt-enable-dynamic-expert-update \\
  --attention-backend flashinfer \\
  --mem-fraction-static 0.88 \\
  --chunked-prefill-size 4096 \\
  --max-running-requests 2 \\
  --max-total-tokens 16384 \\
  --watchdog-timeout 6000 \\
  --enable-mixed-chunk \\
  --enable-p2p-check \\
  --disable-cuda-graph"
        ;;

    qwen3.5-397b)
        # Qwen3.5-397B 纯 CPU 推理配置
        CMD="CUDA_VISIBLE_DEVICES=$GPUS SGLANG_DISABLE_CUDNN_CHECK=1 \\
  python -m sglang.launch_server \\
  --model-path $MODEL_PATH \\
  --tensor-parallel-size 1 \\
  --host 0.0.0.0 \\
  --port $PORT \\
  --trust-remote-code \\
  --kt-weight-path $MODEL_PATH \\
  --kt-method FP8 \\
  --kt-cpuinfer 128 \\
  --kt-threadpool-count 4 \\
  --kt-numa-nodes 0 1 \\
  --kt-num-gpu-experts 0 \\
  --kt-gpu-prefill-token-threshold 512 \\
  --kt-enable-dynamic-expert-update \\
  --attention-backend flashinfer \\
  --mem-fraction-static 0.95 \\
  --chunked-prefill-size 512 \\
  --max-running-requests 1 \\
  --max-total-tokens 8192 \\
  --watchdog-timeout 12000 \\
  --enable-mixed-chunk \\
  --disable-cuda-graph"
        ;;
esac

echo "$CMD"
echo "-------------------------------------------"
echo ""

# 执行启动命令
eval $CMD
