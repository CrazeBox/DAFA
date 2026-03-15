#!/bin/bash
# ==============================================================================
# DAFA项目 - GPU快速测试脚本
# ==============================================================================
# 使用方法: ./quick_gpu_test.sh
# 功能: 快速验证GPU加速是否正常工作
# ==============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DAFA GPU快速测试${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 激活虚拟环境
if [[ -d "$PROJECT_DIR/venv" ]]; then
    source "$PROJECT_DIR/venv/bin/activate"
    echo -e "${GREEN}✓ 虚拟环境已激活${NC}"
else
    echo -e "${RED}✗ 虚拟环境不存在，请先运行 install_gpu_env.sh${NC}"
    exit 1
fi

# 显示GPU信息
echo ""
echo -e "${YELLOW}GPU信息:${NC}"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu --format=csv
echo ""

# 运行快速测试
echo -e "${YELLOW}开始快速测试 (10轮训练)...${NC}"
echo ""

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$PROJECT_DIR/results/gpu_test_$TIMESTAMP"

python "$PROJECT_DIR/scripts/run_experiment.py" \
    --method dafa \
    --dataset cifar10 \
    --num_rounds 10 \
    --num_clients 10 \
    --clients_per_round 2 \
    --local_epochs 1 \
    --batch_size 32 \
    --device cuda \
    --output_dir "$OUTPUT_DIR" \
    --seed 42

# 显示结果
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}测试结果${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [[ -f "$OUTPUT_DIR"/*/results.json ]]; then
    RESULT_FILE=$(find "$OUTPUT_DIR" -name "results.json" | head -1)
    
    python3 << EOF
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)

print(f"最佳准确率: {data['best_accuracy']:.4f}")
print(f"最终轮次: {data['final_round']}")
print(f"总耗时: {data['total_time']:.2f} 秒")

if data.get('convergence_round'):
    print(f"收敛轮次: {data['convergence_round']}")

if data.get('dsnr_summary'):
    print(f"DSNR均值: {data['dsnr_summary']['mean']:.4f}")
EOF
    
    echo ""
    echo -e "${GREEN}✓ GPU测试完成!${NC}"
    echo ""
    echo "结果保存在: $OUTPUT_DIR"
else
    echo -e "${RED}✗ 测试失败，请检查日志${NC}"
fi
