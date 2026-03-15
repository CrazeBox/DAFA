#!/bin/bash
# ==============================================================================
# GPU环境快速验证脚本
# ==============================================================================
# 使用方法: ./verify_gpu.sh
# ==============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GPU环境验证${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

ERRORS=0

# 1. 检查NVIDIA驱动
echo -e "${YELLOW}[1/5] 检查NVIDIA驱动...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA驱动已安装${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu --format=csv,noheader
else
    echo -e "${RED}✗ NVIDIA驱动未安装${NC}"
    echo "  解决: sudo apt install nvidia-driver-535"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 2. 检查CUDA Toolkit
echo -e "${YELLOW}[2/5] 检查CUDA Toolkit...${NC}"
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✓ CUDA Toolkit已安装${NC}"
    nvcc --version | grep "release"
else
    echo -e "${RED}✗ CUDA Toolkit未安装${NC}"
    echo "  解决: 运行 ./scripts/install_gpu_env.sh"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 3. 检查环境变量
echo -e "${YELLOW}[3/5] 检查环境变量...${NC}"
if [[ -n "$CUDA_HOME" ]]; then
    echo -e "${GREEN}✓ CUDA_HOME: $CUDA_HOME${NC}"
else
    echo -e "${YELLOW}⚠ CUDA_HOME未设置${NC}"
    echo "  解决: export CUDA_HOME=/usr/local/cuda"
fi

if [[ "$PATH" == *"cuda"* ]]; then
    echo -e "${GREEN}✓ CUDA已添加到PATH${NC}"
else
    echo -e "${YELLOW}⚠ CUDA未添加到PATH${NC}"
    echo "  解决: export PATH=/usr/local/cuda/bin:\$PATH"
fi
echo ""

# 4. 检查虚拟环境
echo -e "${YELLOW}[4/5] 检查Python虚拟环境...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

if [[ -d "$VENV_DIR" ]]; then
    echo -e "${GREEN}✓ 虚拟环境存在: $VENV_DIR${NC}"
    
    # 激活虚拟环境
    source "$VENV_DIR/bin/activate"
    echo -e "${GREEN}✓ 虚拟环境已激活${NC}"
else
    echo -e "${RED}✗ 虚拟环境不存在${NC}"
    echo "  解决: python3 -m venv venv && source venv/bin/activate"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 5. 检查PyTorch GPU
echo -e "${YELLOW}[5/5] 检查PyTorch GPU支持...${NC}"
python3 << 'EOF'
import sys
import torch

print(f"PyTorch版本: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        mem = props.total_memory / 1024**3
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")
    
    # 测试GPU计算
    print("\n测试GPU计算...")
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print(f"矩阵乘法测试: 成功 (结果形状: {z.shape})")
    
    print("\n✓ PyTorch GPU加速可用!")
    sys.exit(0)
else:
    print("\n✗ PyTorch GPU加速不可用")
    print("可能原因:")
    print("  1. NVIDIA驱动未安装")
    print("  2. CUDA Toolkit未安装")
    print("  3. PyTorch是CPU版本")
    print("\n解决方案:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)
EOF

if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}GPU环境验证通过!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "现在可以运行实验:"
    echo "  python scripts/run_experiment.py --method dafa --dataset cifar10 --num_rounds 10 --device cuda"
else
    ERRORS=$((ERRORS + 1))
fi

# 总结
echo ""
if [[ $ERRORS -gt 0 ]]; then
    echo -e "${RED}发现 $ERRORS 个问题，请根据上述提示解决${NC}"
    exit 1
else
    echo -e "${GREEN}所有检查通过!${NC}"
    exit 0
fi
