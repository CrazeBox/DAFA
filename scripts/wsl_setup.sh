#!/bin/bash
# ==============================================================================
# DAFA项目 - WSL GPU环境快速配置脚本
# ==============================================================================
# 专为WSL环境设计，自动检测并配置GPU环境
# 使用方法: chmod +x wsl_setup.sh && ./wsl_setup.sh
# ==============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

# 检测WSL环境
check_wsl() {
    log_step "检测WSL环境..."
    
    if grep -qi microsoft /proc/version 2>/dev/null; then
        log_success "运行在WSL环境中"
        IS_WSL=true
    else
        log_error "此脚本仅适用于WSL环境"
        log_info "如果是原生Linux，请使用 install_gpu_env.sh"
        exit 1
    fi
}

# 检测GPU
check_gpu() {
    log_step "检查GPU..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi不可用"
        log_info "请确保Windows已安装NVIDIA驱动"
        log_info "下载地址: https://www.nvidia.com/Download/index.aspx"
        exit 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        log_error "GPU不可用"
        exit 1
    fi
    
    log_success "检测到GPU:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    log_info "支持CUDA版本: $CUDA_VERSION"
}

# 安装系统依赖
install_dependencies() {
    log_step "安装系统依赖..."
    
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv build-essential
    
    log_success "系统依赖安装完成"
}

# 创建虚拟环境
create_venv() {
    log_step "创建Python虚拟环境..."
    
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    VENV_DIR="$PROJECT_DIR/venv"
    
    if [[ -d "$VENV_DIR" ]]; then
        log_warning "虚拟环境已存在"
        read -p "是否重新创建? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
            python3 -m venv "$VENV_DIR"
        fi
    else
        python3 -m venv "$VENV_DIR"
    fi
    
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel
    
    log_success "虚拟环境创建完成"
}

# 安装PyTorch
install_pytorch() {
    log_step "安装PyTorch GPU版本..."
    
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    
    log_info "检测到CUDA $CUDA_VERSION"
    
    case $CUDA_MAJOR in
        11)
            log_info "安装PyTorch (CUDA 11.8)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            ;;
        12|13)
            log_info "安装PyTorch (CUDA 12.1)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            ;;
        *)
            log_warning "未知CUDA版本，尝试CUDA 12.1..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            ;;
    esac
    
    log_success "PyTorch安装完成"
}

# 安装项目依赖
install_project_deps() {
    log_step "安装项目依赖..."
    
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    
    if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
        pip install -r "$PROJECT_DIR/requirements.txt"
        log_success "项目依赖安装完成"
    else
        log_warning "未找到requirements.txt"
    fi
}

# 验证安装
verify() {
    log_step "验证GPU环境..."
    
    python3 << 'EOF'
import torch
import sys

print("=" * 50)
print("GPU环境验证")
print("=" * 50)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    显存: {props.total_memory / 1024**3:.1f} GB")
    
    print("\n测试GPU计算...")
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print(f"矩阵乘法测试: 成功 (结果形状: {z.shape})")
    
    print("\n✓ GPU环境配置成功!")
    sys.exit(0)
else:
    print("\n✗ GPU不可用")
    sys.exit(1)
EOF
}

# 显示后续步骤
show_next_steps() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}WSL GPU环境配置完成!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "后续步骤:"
    echo ""
    echo "1. 激活虚拟环境:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. 验证GPU可用:"
    echo "   python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')\""
    echo ""
    echo "3. 运行快速测试:"
    echo "   python scripts/run_experiment.py --method dafa --dataset cifar10 --num_rounds 10 --device cuda"
    echo ""
    echo "4. 运行完整实验:"
    echo "   python scripts/run_all_experiments.py --experiment all"
    echo ""
}

# 主函数
main() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}DAFA项目 - WSL GPU环境配置${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    check_wsl
    check_gpu
    install_dependencies
    create_venv
    install_pytorch
    install_project_deps
    verify
    show_next_steps
}

main "$@"
