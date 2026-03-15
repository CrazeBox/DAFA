#!/bin/bash
# ==============================================================================
# DAFA联邦学习框架 - Ubuntu系统环境配置脚本
# ==============================================================================
# 支持系统: Ubuntu 20.04 LTS / 22.04 LTS
# 使用方法: chmod +x setup_ubuntu.sh && ./setup_ubuntu.sh
# ==============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为root用户
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_warning "此脚本需要sudo权限，部分操作可能需要输入密码"
    fi
}

# 检测Ubuntu版本
detect_ubuntu_version() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        log_info "检测到系统: $NAME $VERSION"
        UBUNTU_VERSION=$(echo $VERSION_ID | cut -d. -f1)
    else
        log_error "无法检测系统版本，请确保运行在Ubuntu系统上"
        exit 1
    fi
}

# 安装系统依赖
install_system_dependencies() {
    log_info "更新软件包列表..."
    sudo apt-get update

    log_info "安装基础依赖..."
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        vim \
        htop \
        tmux \
        tree \
        zip \
        unzip \
        software-properties-common

    log_success "基础依赖安装完成"
}

# 安装Python环境
install_python() {
    log_info "检查Python版本..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        log_info "已安装Python版本: $PYTHON_VERSION"
    else
        log_info "安装Python 3..."
        sudo apt-get install -y python3 python3-dev python3-pip python3-venv
    fi

    # 确保pip是最新版本
    log_info "升级pip..."
    python3 -m pip install --upgrade pip setuptools wheel

    log_success "Python环境配置完成"
}

# 安装CUDA支持 (可选)
install_cuda() {
    log_info "检查NVIDIA GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        log_success "检测到NVIDIA GPU:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        
        # 检查CUDA
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            log_success "CUDA版本: $CUDA_VERSION"
        else
            log_warning "未检测到CUDA，如需GPU加速请手动安装CUDA Toolkit"
            log_info "CUDA安装指南: https://developer.nvidia.com/cuda-downloads"
        fi
    else
        log_warning "未检测到NVIDIA GPU，将使用CPU模式运行"
    fi
}

# 创建虚拟环境
create_virtualenv() {
    log_info "创建Python虚拟环境..."
    
    VENV_DIR="${PROJECT_DIR:-$(pwd)}/venv"
    
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
    
    log_success "虚拟环境创建完成: $VENV_DIR"
}

# 安装Python依赖
install_python_dependencies() {
    log_info "激活虚拟环境并安装依赖..."
    
    source "${PROJECT_DIR:-$(pwd)}/venv/bin/activate"
    
    # 检测CUDA版本并安装对应PyTorch
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        
        if [[ -n "$CUDA_VERSION" ]]; then
            log_info "检测到CUDA $CUDA_VERSION，安装对应PyTorch版本..."
            
            # 根据CUDA版本选择PyTorch
            if [[ "$CUDA_VERSION" == "11."* ]]; then
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            elif [[ "$CUDA_VERSION" == "12."* ]]; then
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            else
                log_warning "未知CUDA版本，安装CPU版本PyTorch"
                pip install torch torchvision torchaudio
            fi
        else
            pip install torch torchvision torchaudio
        fi
    else
        log_info "安装CPU版本PyTorch..."
        pip install torch torchvision torchaudio
    fi
    
    # 安装其他依赖
    if [[ -f "requirements.txt" ]]; then
        log_info "安装项目依赖..."
        pip install -r requirements.txt
    fi
    
    log_success "Python依赖安装完成"
}

# 配置环境变量
configure_environment() {
    log_info "配置环境变量..."
    
    PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
    ENV_FILE="$PROJECT_DIR/.env"
    
    cat > "$ENV_FILE" << EOF
# DAFA联邦学习框架环境配置
# 生成时间: $(date)

# 项目根目录
PROJECT_ROOT=$PROJECT_DIR

# 数据目录
DATA_DIR=$PROJECT_DIR/data
RESULTS_DIR=$PROJECT_DIR/results
CHECKPOINT_DIR=$PROJECT_DIR/checkpoints
LOG_DIR=$PROJECT_DIR/logs

# Python配置
PYTHONPATH=$PROJECT_DIR/src:\$PYTHONPATH
PYTHONUNBUFFERED=1

# CUDA配置 (如有GPU)
EOF

    # 添加CUDA配置
    if command -v nvidia-smi &> /dev/null; then
        cat >> "$ENV_FILE" << EOF
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
EOF
    fi

    # 添加到.bashrc
    if ! grep -q "source $PROJECT_DIR/.env" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# DAFA项目环境变量" >> ~/.bashrc
        echo "source $PROJECT_DIR/.env" >> ~/.bashrc
        log_info "已将环境变量添加到 ~/.bashrc"
    fi
    
    source "$ENV_FILE"
    log_success "环境变量配置完成"
}

# 创建必要目录
create_directories() {
    log_info "创建项目目录结构..."
    
    PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
    
    mkdir -p "$PROJECT_DIR/data/cifar10"
    mkdir -p "$PROJECT_DIR/data/femnist"
    mkdir -p "$PROJECT_DIR/data/shakespeare"
    mkdir -p "$PROJECT_DIR/results/baseline"
    mkdir -p "$PROJECT_DIR/results/sensitivity"
    mkdir -p "$PROJECT_DIR/results/ablation"
    mkdir -p "$PROJECT_DIR/results/dsnr"
    mkdir -p "$PROJECT_DIR/checkpoints"
    mkdir -p "$PROJECT_DIR/logs"
    
    log_success "目录结构创建完成"
}

# 设置文件权限
set_permissions() {
    log_info "设置文件权限..."
    
    PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
    
    # 设置脚本执行权限
    find "$PROJECT_DIR/scripts" -name "*.py" -exec chmod +x {} \;
    find "$PROJECT_DIR/scripts" -name "*.sh" -exec chmod +x {} \;
    
    # 设置数据目录权限
    chmod -R 755 "$PROJECT_DIR/data"
    chmod -R 755 "$PROJECT_DIR/results"
    chmod -R 755 "$PROJECT_DIR/checkpoints"
    chmod -R 755 "$PROJECT_DIR/logs"
    
    log_success "文件权限设置完成"
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    source "${PROJECT_DIR:-$(pwd)}/venv/bin/activate"
    
    python3 << 'EOF'
import sys
print("=" * 50)
print("DAFA联邦学习框架 - 环境验证")
print("=" * 50)

errors = []

# 检查Python版本
print(f"Python版本: {sys.version}")
if sys.version_info < (3, 8):
    errors.append("Python版本过低，需要3.8+")

# 检查核心依赖
dependencies = [
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("numpy", "NumPy"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
    ("matplotlib", "matplotlib"),
]

for module, name in dependencies:
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "未知")
        print(f"✓ {name}: {version}")
    except ImportError:
        errors.append(f"缺少依赖: {name}")
        print(f"✗ {name}: 未安装")

# 检查CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA: 可用")
        print(f"  - GPU数量: {torch.cuda.device_count()}")
        print(f"  - GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA版本: {torch.version.cuda}")
    else:
        print("⚠ CUDA: 不可用 (将使用CPU模式)")
except Exception as e:
    print(f"✗ CUDA检查失败: {e}")

print("=" * 50)
if errors:
    print("验证失败，请检查以下问题:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("环境验证通过!")
    sys.exit(0)
EOF

    if [[ $? -eq 0 ]]; then
        log_success "安装验证通过!"
    else
        log_error "安装验证失败，请检查错误信息"
        exit 1
    fi
}

# 显示后续步骤
show_next_steps() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}DAFA联邦学习框架安装完成!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "后续步骤:"
    echo ""
    echo "1. 激活虚拟环境:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. 运行测试:"
    echo "   python scripts/run_experiment.py --method dafa --dataset cifar10 --num_rounds 10"
    echo ""
    echo "3. 运行完整实验:"
    echo "   python scripts/run_all_experiments.py --experiment all"
    echo ""
    echo "4. 后台运行 (推荐使用tmux):"
    echo "   tmux new -s dafa"
    echo "   python scripts/run_all_experiments.py --experiment all"
    echo "   # 按 Ctrl+B 然后按 D 分离会话"
    echo "   # 重新连接: tmux attach -t dafa"
    echo ""
    echo "5. 查看GPU状态:"
    echo "   watch -n 1 nvidia-smi"
    echo ""
    echo "详细文档请参阅: docs/EXPERIMENT_GUIDE.md"
    echo ""
}

# 主函数
main() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}DAFA联邦学习框架 - Ubuntu环境配置${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    check_root
    detect_ubuntu_version
    
    install_system_dependencies
    install_python
    install_cuda
    create_virtualenv
    install_python_dependencies
    configure_environment
    create_directories
    set_permissions
    verify_installation
    show_next_steps
}

# 运行主函数
main "$@"
