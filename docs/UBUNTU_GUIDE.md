# DAFA联邦学习框架 - Ubuntu部署指南

## 目录
1. [系统要求](#1-系统要求)
2. [快速部署](#2-快速部署)
3. [手动安装](#3-手动安装)
4. [GPU配置](#4-gpu配置)
5. [运行实验](#5-运行实验)
6. [后台运行](#6-后台运行)
7. [常见问题](#7-常见问题)

---

## 1. 系统要求

### 1.1 支持的Ubuntu版本

| 版本 | 状态 | 备注 |
|------|------|------|
| Ubuntu 20.04 LTS | ✅ 完全支持 | 推荐 |
| Ubuntu 22.04 LTS | ✅ 完全支持 | 推荐 |
| Ubuntu 18.04 LTS | ⚠️ 部分支持 | 需要额外配置 |

### 1.2 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| CPU | 4核心 | 8核心以上 |
| 内存 | 16 GB | 32 GB |
| GPU | GTX 1060 6GB | RTX 3080/3090 |
| 存储 | 50 GB SSD | 200 GB NVMe SSD |

### 1.3 软件依赖

```bash
# 检查系统版本
lsb_release -a

# 检查内核版本
uname -r
```

---

## 2. 快速部署

### 2.1 一键安装

```bash
# 克隆项目 (如果还没有)
git clone https://github.com/yourname/DAFA.git
cd DAFA

# 赋予执行权限
chmod +x scripts/setup_ubuntu.sh

# 运行安装脚本
./scripts/setup_ubuntu.sh
```

### 2.2 安装脚本功能

- 自动检测系统版本
- 安装系统依赖 (build-essential, cmake等)
- 安装Python 3和pip
- 创建虚拟环境
- 安装PyTorch (自动检测CUDA版本)
- 安装项目依赖
- 配置环境变量
- 创建目录结构
- 设置文件权限
- 验证安装

---

## 3. 手动安装

### 3.1 安装系统依赖

```bash
# 更新软件包列表
sudo apt-get update

# 安装基础工具
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

# 安装Python
sudo apt-get install -y python3 python3-dev python3-pip python3-venv
```

### 3.2 创建虚拟环境

```bash
# 进入项目目录
cd /path/to/DAFA

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip
pip install --upgrade pip setuptools wheel
```

### 3.3 安装PyTorch

```bash
# 检测CUDA版本
nvidia-smi | grep "CUDA Version"

# 根据CUDA版本安装PyTorch
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU版本 (无GPU)
pip install torch torchvision torchaudio
```

### 3.4 安装项目依赖

```bash
# 安装其他依赖
pip install -r requirements.txt
```

### 3.5 验证安装

```bash
# 运行验证脚本
python3 << 'EOF'
import torch
import torchvision
import numpy as np
import yaml

print("=" * 50)
print("DAFA联邦学习框架 - 环境验证")
print("=" * 50)

print(f"Python版本: {torch.__version__}")
print(f"PyTorch版本: {torch.__version__}")
print(f"TorchVision版本: {torchvision.__version__}")
print(f"NumPy版本: {np.__version__}")

if torch.cuda.is_available():
    print(f"CUDA: 可用")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA: 不可用 (CPU模式)")

print("=" * 50)
EOF
```

---

## 4. GPU配置

### 4.1 NVIDIA驱动安装

```bash
# 检查是否已安装驱动
nvidia-smi

# 如果未安装，添加NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update

# 查看推荐驱动
ubuntu-drivers devices

# 安装推荐驱动 (示例)
sudo apt-get install -y nvidia-driver-535

# 重启系统
sudo reboot
```

### 4.2 CUDA Toolkit安装

```bash
# 下载CUDA Toolkit (访问 NVIDIA官网获取最新版本)
# https://developer.nvidia.com/cuda-downloads

# 示例: CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install -y cuda-11-8

# 配置环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证CUDA安装
nvcc --version
```

### 4.3 cuDNN安装

```bash
# 下载cuDNN (需要NVIDIA账号)
# https://developer.nvidia.com/cudnn

# 解压并复制文件
tar -xzvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

---

## 5. 运行实验

### 5.1 激活环境

```bash
# 进入项目目录
cd /path/to/DAFA

# 激活虚拟环境
source venv/bin/activate
```

### 5.2 单次实验

```bash
# 使用启动脚本
./scripts/run_dafa.sh --method dafa --dataset cifar10 --num_rounds 100

# 或直接使用Python
python scripts/run_experiment.py \
    --method dafa \
    --dataset cifar10 \
    --num_rounds 100 \
    --device cuda
```

### 5.3 批量实验

```bash
# 运行所有实验
./scripts/run_dafa.sh --experiment all

# 运行特定类型实验
./scripts/run_dafa.sh --experiment baseline
./scripts/run_dafa.sh --experiment sensitivity
./scripts/run_dafa.sh --experiment ablation
```

---

## 6. 后台运行

### 6.1 使用tmux (推荐)

```bash
# 启动tmux会话
./scripts/tmux_session.sh start

# 连接到会话
./scripts/tmux_session.sh attach

# 分离会话: Ctrl+B 然后按 D

# 查看状态
./scripts/tmux_session.sh status

# 停止会话
./scripts/tmux_session.sh stop
```

### 6.2 使用nohup

```bash
# 后台运行
nohup python scripts/run_all_experiments.py --experiment all > experiment.log 2>&1 &

# 查看进程
ps aux | grep python

# 查看日志
tail -f experiment.log

# 停止进程
kill <PID>
```

### 6.3 使用systemd服务

```bash
# 复制服务文件
sudo cp scripts/dafa-experiment.service /etc/systemd/system/

# 编辑服务文件，修改路径和用户名
sudo nano /etc/systemd/system/dafa-experiment.service

# 重载systemd
sudo systemctl daemon-reload

# 启用服务 (开机自启)
sudo systemctl enable dafa-experiment

# 启动服务
sudo systemctl start dafa-experiment

# 查看状态
sudo systemctl status dafa-experiment

# 查看日志
sudo journalctl -u dafa-experiment -f

# 停止服务
sudo systemctl stop dafa-experiment
```

---

## 7. 常见问题

### 7.1 CUDA内存不足

```bash
# 错误: RuntimeError: CUDA out of memory

# 解决方案1: 减小批大小
python scripts/run_experiment.py --batch_size 32 ...

# 解决方案2: 减少客户端数量
python scripts/run_experiment.py --clients_per_round 5 ...

# 解决方案3: 使用CPU
python scripts/run_experiment.py --device cpu ...

# 解决方案4: 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### 7.2 权限问题

```bash
# 错误: Permission denied

# 解决方案: 修复文件权限
chmod +x scripts/*.sh
chmod +x scripts/*.py

# 修复目录权限
chmod -R 755 data results checkpoints logs
```

### 7.3 Python版本问题

```bash
# 错误: Python版本不兼容

# 检查Python版本
python3 --version

# 如果版本过低，安装新版本
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

# 使用指定版本创建虚拟环境
python3.10 -m venv venv
```

### 7.4 依赖冲突

```bash
# 错误: 依赖版本冲突

# 解决方案: 重新创建虚拟环境
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 7.5 数据集下载失败

```bash
# 错误: ConnectionError / Download failed

# 解决方案1: 手动下载
# CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# 解压到 data/cifar10/

# 解决方案2: 使用代理
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

### 7.6 进程意外终止

```bash
# 查看系统日志
dmesg | tail -50

# 检查内存使用
free -h

# 检查磁盘空间
df -h

# 从检查点恢复
python scripts/run_experiment.py --resume checkpoints/round_XX.pt ...
```

---

## 附录A: 快速命令参考

```bash
# 环境管理
source venv/bin/activate          # 激活环境
deactivate                         # 退出环境

# 实验运行
python scripts/run_experiment.py --method dafa --dataset cifar10 --num_rounds 100
python scripts/run_all_experiments.py --experiment all

# 后台运行
./scripts/run_dafa.sh --daemon --experiment all
./scripts/tmux_session.sh start

# GPU监控
watch -n 1 nvidia-smi              # 实时GPU状态
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1

# 日志查看
tail -f logs/experiment.log        # 实时日志
grep -i error logs/*.log           # 搜索错误

# 进程管理
ps aux | grep python               # 查看Python进程
kill <PID>                         # 终止进程
pkill -f run_experiment            # 终止所有实验进程
```

---

## 附录B: 目录结构

```
DAFA/
├── configs/                    # 配置文件
│   ├── datasets/              # 数据集配置
│   └── methods/               # 方法配置
├── data/                       # 数据目录
│   ├── cifar10/
│   ├── femnist/
│   └── shakespeare/
├── results/                    # 实验结果
├── checkpoints/                # 模型检查点
├── logs/                       # 日志文件
├── scripts/                    # 脚本文件
│   ├── setup_ubuntu.sh        # Ubuntu安装脚本
│   ├── run_dafa.sh            # 实验启动脚本
│   ├── tmux_session.sh        # tmux会话管理
│   ├── run_experiment.py      # 单实验运行
│   └── run_all_experiments.py # 批量实验
├── src/                        # 源代码
│   ├── core/                  # 核心模块
│   ├── methods/               # 聚合方法
│   ├── models/                # 模型定义
│   ├── data/                  # 数据处理
│   └── utils/                 # 工具函数
├── docs/                       # 文档
├── requirements.txt            # Python依赖
└── .env                        # 环境变量
```

---

*文档版本: 1.0*
*最后更新: 2024年*
