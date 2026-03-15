#!/bin/bash
# ==============================================================================
# DAFA项目 - Ubuntu GPU环境一键安装脚本
# ==============================================================================
# 支持系统: Ubuntu 20.04 LTS / 22.04 LTS
# 目标: 安装NVIDIA驱动 + CUDA Toolkit + PyTorch GPU版本
# 使用方法: chmod +x install_gpu_env.sh && ./install_gpu_env.sh
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

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# 检测系统版本
detect_system() {
    log_step "检测系统信息..."
    
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        log_info "系统: $NAME $VERSION"
        UBUNTU_VERSION=$(echo $VERSION_ID | cut -d. -f1)
        UBUNTU_CODENAME=$UBUNTU_CODENAME
        
        case $UBUNTU_VERSION in
            20) UBUNTU_CODENAME="ubuntu2004" ;;
            22) UBUNTU_CODENAME="ubuntu2204" ;;
            *) log_error "不支持的Ubuntu版本: $UBUNTU_VERSION"; exit 1 ;;
        esac
        
        log_success "Ubuntu版本代码: $UBUNTU_CODENAME"
    else
        log_error "无法检测系统版本"
        exit 1
    fi
}

# 检测GPU
detect_gpu() {
    log_step "检测GPU硬件..."
    
    GPU_DETECTED=false
    
    # 方法1: 使用nvidia-smi检测（适用于WSL和原生Linux）
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        GPU_DETECTED=true
        log_success "检测到NVIDIA GPU (通过nvidia-smi):"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    # 方法2: 使用lspci检测（原生Linux）
    elif command -v lspci &> /dev/null && lspci | grep -i nvidia > /dev/null; then
        GPU_DETECTED=true
        log_success "检测到NVIDIA GPU (通过lspci):"
        lspci | grep -i nvidia | head -3
    fi
    
    if [[ "$GPU_DETECTED" == false ]]; then
        log_error "未检测到NVIDIA GPU"
        log_info "如果你在WSL中运行，请确保:"
        log_info "  1. Windows已安装NVIDIA驱动"
        log_info "  2. WSL版本为WSL2 (运行: wsl -l -v 检查)"
        exit 1
    fi
}

# 检测WSL环境
detect_wsl() {
    if grep -qi microsoft /proc/version 2>/dev/null; then
        IS_WSL=true
        log_info "检测到WSL环境"
    else
        IS_WSL=false
    fi
}

# 检查现有驱动
check_existing_driver() {
    log_step "检查现有NVIDIA驱动..."
    
    # WSL环境特殊处理
    if [[ "$IS_WSL" == true ]]; then
        log_info "运行在WSL环境中，使用Windows NVIDIA驱动"
        
        if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
            log_success "WSL GPU驱动正常:"
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
            
            CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
            log_info "支持CUDA版本: $CUDA_VERSION"
            log_info "WSL不需要安装驱动，跳过驱动安装"
            SKIP_DRIVER=true
        else
            log_error "WSL中nvidia-smi不可用"
            log_info "请确保Windows已安装最新NVIDIA驱动"
            log_info "下载地址: https://www.nvidia.com/Download/index.aspx"
            exit 1
        fi
        return
    fi
    
    # 原生Linux环境
    if command -v nvidia-smi &> /dev/null; then
        log_info "已安装NVIDIA驱动:"
        nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
        
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        
        log_info "驱动版本: $DRIVER_VERSION"
        log_info "支持CUDA版本: $CUDA_VERSION"
        
        read -p "是否重新安装驱动? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "跳过驱动安装"
            SKIP_DRIVER=true
        fi
    else
        log_warning "未检测到NVIDIA驱动，将进行安装"
        SKIP_DRIVER=false
    fi
}

# 安装NVIDIA驱动
install_nvidia_driver() {
    if [[ "$SKIP_DRIVER" == true ]]; then
        return
    fi
    
    log_step "安装NVIDIA驱动..."
    
    log_info "添加NVIDIA驱动PPA..."
    sudo add-apt-repository -y ppa:graphics-drivers/ppa
    sudo apt-get update
    
    log_info "查看推荐驱动版本..."
    ubuntu-drivers devices
    
    log_info "安装NVIDIA驱动 (nvidia-driver-535)..."
    sudo apt-get install -y nvidia-driver-535
    
    log_success "NVIDIA驱动安装完成"
    log_warning "需要重启系统才能生效"
}

# 安装CUDA Toolkit
install_cuda() {
    log_step "安装CUDA Toolkit..."
    
    # WSL环境可以选择跳过CUDA Toolkit安装
    if [[ "$IS_WSL" == true ]]; then
        log_info "检测到WSL环境"
        log_info "WSL可以直接使用Windows的CUDA驱动"
        log_info "你只需要安装PyTorch GPU版本即可"
        
        read -p "是否跳过CUDA Toolkit安装? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            log_info "跳过CUDA Toolkit安装"
            return
        fi
    fi
    
    CUDA_VERSION="${CUDA_VERSION:-12-1}"
    
    log_info "目标CUDA版本: $CUDA_VERSION"
    
    log_info "下载CUDA仓库密钥..."
    cd /tmp
    
    if [[ "$UBUNTU_CODENAME" == "ubuntu2004" ]]; then
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    else
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    fi
    
    log_info "安装CUDA仓库..."
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    
    log_info "安装CUDA Toolkit $CUDA_VERSION..."
    sudo apt-get install -y cuda-toolkit-$CUDA_VERSION
    
    log_success "CUDA Toolkit安装完成"
}

# 配置环境变量
configure_environment() {
    log_step "配置环境变量..."
    
    CUDA_PATH="/usr/local/cuda"
    
    if [[ ! -L "$CUDA_PATH" ]]; then
        log_info "创建CUDA符号链接..."
        sudo ln -sf /usr/local/cuda-12.1 /usr/local/cuda
    fi
    
    if ! grep -q "CUDA环境变量" ~/.bashrc; then
        log_info "添加环境变量到 ~/.bashrc..."
        
        cat >> ~/.bashrc << 'EOF'

# ==============================================================================
# CUDA环境变量
# ==============================================================================
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# PyTorch CUDA配置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
EOF
        
        log_success "环境变量已添加"
    else
        log_info "环境变量已存在"
    fi
    
    source ~/.bashrc
}

# 创建虚拟环境
create_venv() {
    log_step "创建Python虚拟环境..."
    
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    VENV_DIR="$PROJECT_DIR/venv"
    
    if [[ -d "$VENV_DIR" ]]; then
        log_warning "虚拟环境已存在: $VENV_DIR"
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

# 安装PyTorch GPU版本
install_pytorch() {
    log_step "安装PyTorch GPU版本..."
    
    source venv/bin/activate 2>/dev/null || true
    
    CUDA_VERSION_NUM=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -c2- | cut -d. -f1,2 || echo "12.1")
    
    log_info "检测到CUDA版本: $CUDA_VERSION_NUM"
    
    case $CUDA_VERSION_NUM in
        11.8|11.*)
            log_info "安装PyTorch (CUDA 11.8)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            ;;
        12.1|12.*)
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
install_dependencies() {
    log_step "安装项目依赖..."
    
    source venv/bin/activate 2>/dev/null || true
    
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    
    if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
        pip install -r "$PROJECT_DIR/requirements.txt"
        log_success "项目依赖安装完成"
    else
        log_warning "未找到requirements.txt"
    fi
}

# 验证安装
verify_installation() {
    log_step "验证安装..."
    
    echo ""
    echo "========================================"
    echo "GPU环境验证"
    echo "========================================"
    
    # 验证NVIDIA驱动
    echo ""
    echo "[1/4] NVIDIA驱动状态:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        echo "✓ NVIDIA驱动正常"
    else
        echo "✗ NVIDIA驱动未安装"
    fi
    
    # 验证CUDA
    echo ""
    echo "[2/4] CUDA状态:"
    if command -v nvcc &> /dev/null; then
        nvcc --version | grep "release"
        echo "✓ CUDA Toolkit正常"
    else
        echo "✗ CUDA Toolkit未安装"
    fi
    
    # 验证PyTorch
    echo ""
    echo "[3/4] PyTorch GPU状态:"
    source venv/bin/activate 2>/dev/null || true
    python3 << 'PYEOF'
import torch
import sys

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
    print("✓ PyTorch GPU加速可用")
    sys.exit(0)
else:
    print("✗ PyTorch GPU加速不可用")
    sys.exit(1)
PYEOF

    # 验证环境变量
    echo ""
    echo "[4/4] 环境变量:"
    echo "PATH: $(echo $PATH | tr ':' '\n' | grep cuda)"
    echo "CUDA_HOME: $CUDA_HOME"
    
    echo ""
    echo "========================================"
}

# 显示后续步骤
show_next_steps() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}GPU环境安装完成!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    if [[ "$SKIP_DRIVER" != true ]]; then
        echo -e "${YELLOW}重要: 需要重启系统才能使NVIDIA驱动生效${NC}"
        echo ""
    fi
    
    echo "后续步骤:"
    echo ""
    echo "1. 重启系统 (如果安装了新驱动):"
    echo "   sudo reboot"
    echo ""
    echo "2. 激活虚拟环境:"
    echo "   cd $(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    echo "   source venv/bin/activate"
    echo ""
    echo "3. 验证GPU可用:"
    echo "   python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')\""
    echo ""
    echo "4. 运行快速测试:"
    echo "   python scripts/run_experiment.py --method dafa --dataset cifar10 --num_rounds 10 --device cuda"
    echo ""
    echo "5. 运行完整实验:"
    echo "   python scripts/run_all_experiments.py --experiment all"
    echo ""
}

# 主函数
main() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}DAFA项目 - Ubuntu GPU环境安装${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    detect_system
    detect_wsl
    detect_gpu
    check_existing_driver
    install_nvidia_driver
    install_cuda
    configure_environment
    create_venv
    install_pytorch
    install_dependencies
    verify_installation
    show_next_steps
}

# 运行主函数
main "$@"
