# DAFA联邦学习实验框架操作手册

## 目录

1. [环境准备](#1-环境准备)
2. [依赖安装](#2-依赖安装)
3. [配置参数说明](#3-配置参数说明)
4. [执行流程](#4-执行流程)
5. [预期输出结果](#5-预期输出结果)
6. [常见问题排查](#6-常见问题排查)
7. [断点续跑功能](#7-断点续跑功能)

---

## 1. 环境准备

### 1.1 系统要求

- **操作系统**: Linux (推荐 Ubuntu 18.04+) 或 WSL (Windows Subsystem for Linux)
- **Python**: 3.8 或更高版本
- **CUDA**: 10.2 或更高版本 (如使用GPU)
- **内存**: 建议 16GB 以上
- **存储**: 建议 50GB 以上可用空间

### 1.2 WSL环境配置 (Windows用户)

```bash
# 1. 安装WSL (如未安装)
wsl --install -d Ubuntu-20.04

# 2. 启动WSL
wsl

# 3. 更新系统包
sudo apt update && sudo apt upgrade -y

# 4. 安装基础工具
sudo apt install -y build-essential curl git

# 5. 安装Python 3.8+
sudo apt install -y python3 python3-pip python3-venv
```

### 1.3 创建虚拟环境

```bash
# 进入项目目录
cd /path/to/FedJD/DAFA

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/WSL
# 或
.\venv\Scripts\activate   # Windows PowerShell

# 验证Python版本
python --version  # 应显示 3.8+
```

---

## 2. 依赖安装

### 2.1 安装核心依赖

```bash
# 升级pip
pip install --upgrade pip

# 安装PyTorch (根据CUDA版本选择)
# CPU版本
pip install torch torchvision torchaudio

# CUDA 11.8版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2.2 安装项目依赖

```bash
# 安装所有依赖
pip install -r requirements.txt
```

### 2.3 验证安装

```bash
# 运行验证脚本
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 预期输出:
# PyTorch: 2.x.x
# CUDA available: True/False
```

---

## 3. 配置参数说明

### 3.1 实验配置文件

配置文件位于 `configs/` 目录下，使用YAML格式。

#### 数据集配置示例 (`configs/datasets/cifar10.yaml`)

```yaml
dataset:
  name: cifar10
  root: data/cifar10
  num_clients: 100
  alpha: 0.5        # Dirichlet浓度参数，越小数据越非独立同分布
  download: true

training:
  batch_size: 64
  num_workers: 4
```

#### 方法配置示例 (`configs/methods/dafa.yaml`)

```yaml
method:
  name: dafa
  mu: 0.5           # DAFA对齐阈值
  use_dsnr_weighting: true
  use_dataset_size_weighting: true
  normalize_updates: true
```

### 3.2 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--method` | str | fedavg | 聚合方法名称 |
| `--dataset` | str | cifar10 | 数据集名称 |
| `--model` | str | resnet18 | 模型架构 |
| `--num_rounds` | int | 100 | 训练轮数 |
| `--num_clients` | int | 100 | 客户端总数 |
| `--clients_per_round` | int | 10 | 每轮参与客户端数 |
| `--local_epochs` | int | 5 | 本地训练轮数 |
| `--local_lr` | float | 0.01 | 本地学习率 |
| `--batch_size` | int | 64 | 批次大小 |
| `--alpha` | float | 0.5 | Dirichlet参数 |
| `--seed` | int | 42 | 随机种子 |
| `--device` | str | cuda | 计算设备 |
| `--mu` | float | 0.01 | DAFA阈值/FedProx系数 |
| `--output_dir` | str | results | 输出目录 |
| `--resume` | str | None | 恢复检查点路径 |

---

## 4. 执行流程

### 4.1 阶段一：公平调参实验

#### 4.1.1 统一阶段一调参入口

```bash
python scripts/run_phase1_tuning.py \
    --dataset cifar10 \
    --alpha 0.1 \
    --num_rounds 50 \
    --num_repeats 3 \
    --output_dir results/phase1_tuning
```

#### 4.1.2 一次运行五阶段实验

```bash
python scripts/run_five_stages.py \
    --stages all \
    --device cuda \
    --num_rounds 100 \
    --output_dir results/five_stages
```

### 4.2 阶段二：核心性能对比

```bash
# 运行所有方法的对比实验
for method in fedavg fedprox scaffold fednova fedavgm fedadam dafa; do
    python scripts/run_experiment.py \
        --method $method \
        --dataset cifar10 \
        --num_rounds 200 \
        --alpha 0.5 \
        --output_dir results/phase2_comparison
done

# 分析对比结果
python scripts/run_analysis.py \
    --compare results/phase2_comparison/* \
    --output_dir analysis/phase2 \
    --plot
```

```bash
# 提取最优运行并生成汇总表
python scripts/extract_best_runs.py \
    --results_root results \
    --output_dir results/summary

# 生成论文风格图表
python scripts/plot_results.py \
    --best_runs results/summary/best_runs.json \
    --output_dir results/summary/plots \
    --format pdf
```

### 4.3 阶段三：机制分析

```bash
# DSNR分析实验
python scripts/run_experiment.py \
    --method dafa \
    --dataset cifar10 \
    --num_rounds 100 \
    --output_dir results/phase3_dsnr_analysis

# 分析DSNR变化
python scripts/run_analysis.py \
    --results_dir results/phase3_dsnr_analysis \
    --output_dir analysis/phase3 \
    --plot
```

### 4.4 阶段四：消融实验

```bash
# 无数据集大小加权
python scripts/run_experiment.py \
    --method dafa \
    --dataset cifar10 \
    --num_rounds 100 \
    --output_dir results/phase4_ablation/no_size_weight

# 不同阈值μ
for mu in 0.0 0.3 0.5 0.7 1.0; do
    python scripts/run_experiment.py \
        --method dafa \
        --dataset cifar10 \
        --mu $mu \
        --num_rounds 100 \
        --output_dir results/phase4_ablation/mu_${mu}
done
```

### 4.5 阶段五：理论验证

```bash
# 收集对齐分数与漂移数据
python scripts/run_experiment.py \
    --method dafa \
    --dataset cifar10 \
    --num_rounds 100 \
    --verbose \
    --output_dir results/phase5_theory
```

### 4.6 一键流水线（推荐）

```bash
bash scripts/study_pipeline.sh results/study_pipeline cuda 42,123,456 100
```

---

## 5. 预期输出结果

### 5.1 实验输出目录结构

```
results/
└── default/
    └── cifar10_dafa_seed42_20240101_120000/
        ├── config.json           # 实验配置
        ├── metadata.json         # 运行元数据
        ├── results.json          # 最终结果
        ├── experiment.log        # 运行日志
        └── checkpoints/
            ├── checkpoint_round_10.pt
            ├── checkpoint_round_20.pt
            └── best_model.pt
```

流水线目录：

```
results/study_pipeline/
├── phase1/
├── five_stages/
│   ├── five_stages_summary.json
│   ├── pipeline_status.json
│   └── stage*/**/failure.json   # 失败样本归档
└── summary/
    ├── all_runs.json
    ├── best_runs.json
    ├── best_runs.csv
    └── plots/*.pdf
```

### 5.2 结果文件格式

`results.json` 示例：

```json
{
    "best_accuracy": 0.8523,
    "final_round": 100,
    "total_time": 3600.5,
    "history": [
        {
            "round": 1,
            "accuracy": 0.4521,
            "loss": 1.5234,
            "round_time": 35.2
        },
        ...
    ],
    "config": {
        "method": "dafa",
        "dataset": "cifar10",
        "num_rounds": 100,
        ...
    }
}
```

### 5.3 预期性能指标

| 方法 | CIFAR-10 (α=0.5) | FEMNIST | Shakespeare |
|------|------------------|---------|-------------|
| FedAvg | ~75% | ~78% | ~45% |
| FedProx | ~76% | ~79% | ~46% |
| SCAFFOLD | ~78% | ~80% | ~47% |
| FedNova | ~77% | ~79% | ~46% |
| FedAvgM | ~77% | ~80% | ~47% |
| FedAdam | ~76% | ~79% | ~46% |
| DAFA | **~80%** | **~82%** | **~49%** |

---

## 6. 常见问题排查

### 6.1 CUDA内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减小批次大小
python scripts/run_experiment.py --batch_size 32 ...

# 或使用CPU
python scripts/run_experiment.py --device cpu ...
```

### 6.2 数据集下载失败

**问题**: 数据集下载超时或失败

**解决方案**:
```bash
# 手动下载数据集
mkdir -p data/cifar10
cd data/cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz

# 或使用镜像源
export TORCH_HOME=/path/to/local/torch
```

### 6.3 依赖版本冲突

**问题**: 包版本不兼容

**解决方案**:
```bash
# 清理并重新安装
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install -r requirements.txt
```

### 6.4 多进程数据加载错误

**问题**: `RuntimeError: DataLoader worker exited unexpectedly`

**解决方案**:
```bash
# 减少worker数量
python scripts/run_experiment.py --num_workers 0 ...

# 或在代码中设置
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn')
```

### 6.5 检查点加载失败

**问题**: 加载检查点时参数不匹配

**解决方案**:
```bash
# 检查检查点内容
python -c "import torch; ckpt = torch.load('checkpoints/checkpoint.pt', map_location='cpu'); print(ckpt.keys())"

# 使用strict=False加载
# 在代码中修改: model.load_state_dict(ckpt['model_state_dict'], strict=False)
```

---

## 7. 断点续跑功能

### 7.1 自动保存机制

程序会在以下情况自动保存检查点：
- 每隔 `save_every` 轮（默认10轮）
- 达到最佳准确率时
- 程序被中断时（Ctrl+C）

### 7.2 手动恢复训练

```bash
# 从指定检查点恢复
python scripts/run_experiment.py \
    --method dafa \
    --dataset cifar10 \
    --resume results/default/cifar10_dafa_seed42_xxx/checkpoints/checkpoint_round_50.pt \
    --num_rounds 100
```

### 7.3 检查点管理

```bash
# 列出所有检查点
ls -la results/experiment_name/checkpoints/

# 查看检查点信息
python -c "
import torch
ckpt = torch.load('checkpoint.pt', map_location='cpu')
print(f'Round: {ckpt[\"round\"]}')
print(f'Best Accuracy: {ckpt[\"best_accuracy\"]}')
"
```

### 7.4 断点续跑命令

```bash
# 完整的断点续跑示例
python scripts/run_experiment.py \
    --config configs/base_config.yaml \
    --method dafa \
    --dataset cifar10 \
    --resume results/default/cifar10_dafa_seed42_20240101/checkpoints/checkpoint_round_50.pt \
    --output_dir results/cifar10_dafa_resumed
```

---

## 附录A: 快速开始脚本

创建 `quick_start.sh`:

```bash
#!/bin/bash

# 激活虚拟环境
source venv/bin/activate

# 运行快速测试
python scripts/run_experiment.py \
    --method dafa \
    --dataset cifar10 \
    --num_rounds 10 \
    --num_clients 10 \
    --clients_per_round 5 \
    --output_dir results/quick_test

echo "Quick test completed. Check results in results/quick_test/"
```

运行：
```bash
chmod +x quick_start.sh
./quick_start.sh
```

---

## 附录B: 联系与支持

如遇到问题，请检查：
1. 本操作手册的常见问题部分
2. 项目日志文件 (`experiment.log`)
3. GitHub Issues (如有)

---

*文档版本: v1.0*
*最后更新: 2024年*
