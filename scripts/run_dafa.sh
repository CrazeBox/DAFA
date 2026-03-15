#!/bin/bash
# ==============================================================================
# DAFA联邦学习实验启动脚本
# ==============================================================================
# 使用方法:
#   ./run_dafa.sh                    # 交互模式
#   ./run_dafa.sh --experiment all   # 运行所有实验
#   ./run_dafa.sh --daemon           # 后台守护进程模式
# ==============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
LOG_DIR="$PROJECT_DIR/logs"
RESULTS_DIR="$PROJECT_DIR/results"

# 默认参数
EXPERIMENT="all"
METHOD="dafa"
DATASET="cifar10"
NUM_ROUNDS=100
DEVICE="cuda"
DAEMON_MODE=false

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --experiment)
                EXPERIMENT="$2"
                shift 2
                ;;
            --method)
                METHOD="$2"
                shift 2
                ;;
            --dataset)
                DATASET="$2"
                shift 2
                ;;
            --num_rounds)
                NUM_ROUNDS="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --daemon)
                DAEMON_MODE=true
                shift
                ;;
            --help)
                echo "使用方法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --experiment TYPE    实验类型: all, baseline, sensitivity, ablation, dsnr"
                echo "  --method METHOD      聚合方法: dafa, fedavg, fedprox, etc."
                echo "  --dataset DATASET    数据集: cifar10, femnist, shakespeare"
                echo "  --num_rounds N       训练轮数"
                echo "  --device DEVICE      设备: cuda, cpu"
                echo "  --daemon             后台守护进程模式"
                echo "  --help               显示帮助信息"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done
}

# 检查环境
check_environment() {
    log_info "检查运行环境..."
    
    # 检查虚拟环境
    if [[ ! -d "$VENV_DIR" ]]; then
        log_error "虚拟环境不存在: $VENV_DIR"
        log_info "请先运行: ./scripts/setup_ubuntu.sh"
        exit 1
    fi
    
    # 激活虚拟环境
    source "$VENV_DIR/bin/activate"
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        log_error "Python未安装"
        exit 1
    fi
    
    # 检查PyTorch
    python -c "import torch" 2>/dev/null || {
        log_error "PyTorch未安装"
        exit 1
    }
    
    # 检查CUDA
    if [[ "$DEVICE" == "cuda" ]]; then
        if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            log_warning "CUDA不可用，切换到CPU模式"
            DEVICE="cpu"
        fi
    fi
    
    # 创建必要目录
    mkdir -p "$LOG_DIR"
    mkdir -p "$RESULTS_DIR"
    
    log_success "环境检查通过"
}

# 检查GPU状态
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU状态:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    fi
}

# 运行单个实验
run_single_experiment() {
    log_info "运行实验: method=$METHOD, dataset=$DATASET, rounds=$NUM_ROUNDS"
    
    python "$PROJECT_DIR/scripts/run_experiment.py" \
        --method "$METHOD" \
        --dataset "$DATASET" \
        --num_rounds "$NUM_ROUNDS" \
        --device "$DEVICE" \
        --output_dir "$RESULTS_DIR"
}

# 运行批量实验
run_batch_experiments() {
    log_info "运行批量实验: experiment=$EXPERIMENT"
    
    python "$PROJECT_DIR/scripts/run_all_experiments.py" \
        --experiment "$EXPERIMENT" \
        --output_dir "$RESULTS_DIR"
}

# 守护进程模式
run_daemon() {
    log_info "启动守护进程模式..."
    
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    LOG_FILE="$LOG_DIR/dafa_${EXPERIMENT}_${TIMESTAMP}.log"
    PID_FILE="$PROJECT_DIR/dafa.pid"
    
    # 检查是否已有进程运行
    if [[ -f "$PID_FILE" ]]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            log_error "已有实验进程运行 (PID: $OLD_PID)"
            exit 1
        else
            rm -f "$PID_FILE"
        fi
    fi
    
    # 启动后台进程
    nohup python "$PROJECT_DIR/scripts/run_all_experiments.py" \
        --experiment "$EXPERIMENT" \
        --output_dir "$RESULTS_DIR" \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo $PID > "$PID_FILE"
    
    log_success "实验已在后台启动"
    log_info "PID: $PID"
    log_info "日志文件: $LOG_FILE"
    log_info ""
    log_info "查看日志: tail -f $LOG_FILE"
    log_info "停止实验: kill $PID"
}

# 显示进度
show_progress() {
    LATEST_RESULT=$(find "$RESULTS_DIR" -name "results.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [[ -n "$LATEST_RESULT" ]]; then
        log_info "最新结果: $LATEST_RESULT"
        python -c "
import json
with open('$LATEST_RESULT') as f:
    data = json.load(f)
print(f\"最佳准确率: {data.get('best_accuracy', 'N/A')}\")
print(f\"收敛轮次: {data.get('convergence_round', 'N/A')}\")
" 2>/dev/null || true
    fi
}

# 主函数
main() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}DAFA联邦学习实验启动器${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    parse_args "$@"
    check_environment
    check_gpu
    
    if [[ "$DAEMON_MODE" == true ]]; then
        run_daemon
    elif [[ "$EXPERIMENT" == "single" ]]; then
        run_single_experiment
    else
        run_batch_experiments
    fi
    
    show_progress
    
    log_success "完成"
}

# 运行主函数
main "$@"
