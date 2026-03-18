# DAFA联邦学习实验运行手册

## 目录
1. [硬件设备清单](#1-硬件设备清单)
2. [软件环境配置](#2-软件环境配置)
3. [操作步骤](#3-操作步骤)
4. [关键参数设置说明](#4-关键参数设置说明)
5. [安全注意事项](#5-安全注意事项)
6. [数据记录方法](#6-数据记录方法)
7. [异常情况处理](#7-异常情况处理)
8. [外部环境运行指南](#8-外部环境运行指南)

---

## 1. 硬件设备清单

### 1.1 最低配置要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **CPU** | 4核心 | 8核心以上 (Intel i7/i9 或 AMD Ryzen 7/9) |
| **GPU** | NVIDIA GTX 1060 6GB | NVIDIA RTX 3080/3090 或 A100 |
| **内存** | 16 GB | 32 GB 或更高 |
| **存储** | 50 GB SSD | 200 GB NVMe SSD |
| **网络** | 稳定互联网连接 | 千兆以太网 |

### 1.2 实验规模与硬件对应

| 实验类型 | GPU显存需求 | 预计运行时间 |
|----------|-------------|--------------|
| 单数据集单方法 | 4-6 GB | 2-4 小时 |
| 基线对比实验 (全部) | 6-8 GB | 24-48 小时 |
| 超参数敏感性分析 | 6-8 GB | 12-24 小时 |
| 完整实验套件 | 8+ GB | 72-120 小时 |

### 1.3 GPU兼容性检查

```powershell
# 检查CUDA是否可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## 2. 软件环境配置

### 2.1 操作系统要求

- **推荐**: Ubuntu 20.04/22.04 LTS 或 Windows 10/11
- **备选**: macOS 12+ (需使用MPS或CPU)

### 2.2 Python环境

```powershell
# 推荐Python版本: 3.9 - 3.11
python --version

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate
```

### 2.3 依赖安装

```powershell
# 升级pip
python -m pip install --upgrade pip

# 安装PyTorch (根据CUDA版本选择)
# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU版本 (无GPU):
pip install torch torchvision torchaudio

# 安装其他依赖
pip install -r requirements.txt
```

### 2.4 完整依赖列表

```
# 核心依赖
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scipy>=1.7.0

# 数据处理
Pillow>=9.0.0

# 配置管理
pyyaml>=6.0
omegaconf>=2.3.0

# 可视化
matplotlib>=3.5.0
seaborn>=0.12.0

# 日志与进度
tqdm>=4.64.0
colorlog>=6.6.0
```

### 2.5 环境验证

```powershell
# 运行环境检查脚本
python -c "
import torch
import torchvision
import numpy as np
import yaml

print('=== 环境检查 ===')
print(f'Python: OK')
print(f'PyTorch: {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')
print(f'NumPy: {np.__version__}')
print(f'PyYAML: {yaml.__version__}')
print(f'CUDA: {\"可用\" if torch.cuda.is_available() else \"不可用\"}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print('=== 检查完成 ===')
"
```

---

## 3. 操作步骤

### 3.1 准备阶段

#### 步骤1: 获取代码
```powershell
# 进入项目目录
cd "e:\AI Project\FedJD\DAFA"

# 确认文件完整性
ls scripts/
ls src/
ls configs/
```

#### 步骤2: 准备数据集

```powershell
# CIFAR-10 (自动下载)
python -c "
from src.data.cifar10 import get_cifar10_loaders
print('正在下载/验证CIFAR-10数据集...')
loaders, val_loader, test_loader, data_manager = get_cifar10_loaders(root='data/cifar10', num_clients=10, alpha=0.5, batch_size=64, seed=42)
print('CIFAR-10准备完成')
"

# FEMNIST (需手动下载LEAF数据集)
# 参考: https://github.com/TalwalkarLab/LEAF

# Shakespeare (需手动下载)
# 参考: https://github.com/TalwalkarLab/LEAF
```

#### 步骤3: 创建输出目录

```powershell
# 创建必要的目录
mkdir -p results checkpoints logs 2>$null
mkdir -p results/baseline results/sensitivity results/ablation results/dsnr 2>$null
```

### 3.2 运行实验

#### 单个实验运行

```powershell
# 基本用法
python scripts/run_experiment.py --method dafa --dataset cifar10 --num_rounds 100

# 完整参数示例
python scripts/run_experiment.py `
    --method dafa `
    --dataset cifar10 `
    --model resnet18 `
    --num_rounds 100 `
    --num_clients 100 `
    --clients_per_round 10 `
    --local_epochs 5 `
    --local_lr 0.01 `
    --alpha 0.5 `
    --gamma 1.0 `
    --beta 0.9 `
    --mu 0.01 `
    --device cuda `
    --seed 42 `
    --output_dir results `
    --track_dsnr true `
    --track_variance true `
    --track_convergence true
```

#### 批量实验运行

```powershell
# 运行所有基线对比实验
python scripts/run_all_experiments.py --experiment baseline

# 运行超参数敏感性分析
python scripts/run_all_experiments.py --experiment sensitivity

# 运行消融实验
python scripts/run_all_experiments.py --experiment ablation

# 运行DSNR分析
python scripts/run_all_experiments.py --experiment dsnr

# 运行全部实验
python scripts/run_all_experiments.py --experiment all
```

### 3.3 实验执行顺序建议

```
推荐执行顺序:
1. 环境验证 (5分钟)
2. 数据集准备 (10-30分钟)
3. 单次测试运行 (15分钟)
4. 基线对比实验 (24-48小时)
5. 超参数敏感性分析 (12-24小时)
6. 消融实验 (12-24小时)
7. DSNR分析 (6-12小时)
8. 结果汇总与分析
```

---

## 4. 关键参数设置说明

### 4.1 聚合方法参数

| 方法 | 关键参数 | 默认值 | 说明 |
|------|----------|--------|------|
| **FedAvg** | - | - | 基线方法，无特殊参数 |
| **FedProx** | mu | 0.01 | 近端项系数 |
| **SCAFFOLD** | server_lr | 1.0 | 服务器学习率 |
| **FedNova** | - | - | 自适应归一化 |
| **FedAvgM** | server_momentum | 0.9 | 服务器动量 |
| **FedAdam** | server_lr | 0.01 | 服务器学习率 |
| **DAFA** | gamma, beta, mu | 1.0, 0.9, 0.01 | 温度/动量/阈值 |
| **Dir-Weight** | gamma, mu | 1.0, 0.01 | 温度/阈值 |

### 4.2 DAFA核心参数详解

```yaml
# gamma (γ): Softmax温度参数
# - 控制对齐度对权重的影响强度
# - 较大值(5.0): 更强的方向偏好，可能忽略数据量
# - 较小值(0.5): 更接近FedAvg，平滑权重分布
# - 推荐范围: 0.5 - 5.0

# beta (β): 动量系数
# - 控制代理方向的时序平滑程度
# - 较大值(0.99): 高度平滑，稳定但响应慢
# - 较小值(0.5): 快速响应，但可能不稳定
# - 推荐范围: 0.7 - 0.95

# mu (μ): 范数阈值
# - 用于优雅降级，过滤异常小的更新
# - 较大值(0.1): 更严格的过滤
# - 较小值(0.001): 更宽松的过滤
# - 推荐范围: 0.001 - 0.1
```

### 4.3 Non-IID程度参数

```yaml
# alpha: Dirichlet浓度参数
# - 控制数据分布的非独立同分布程度
# - alpha = 0.1: 高度Non-IID (极端异构)
# - alpha = 0.5: 中等Non-IID
# - alpha = 1.0: 轻度Non-IID
# - alpha → ∞: IID分布
```

### 4.4 训练参数推荐

| 数据集 | 模型 | 学习率 | 批大小 | 本地轮数 |
|--------|------|--------|--------|----------|
| CIFAR-10 | ResNet-18 | 0.01 | 64 | 5 |
| FEMNIST | CNN | 0.01 | 64 | 5 |
| Shakespeare | LSTM | 0.8 | 64 | 5 |

---

## 5. 安全注意事项

### 5.1 数据安全

```powershell
# 敏感数据不要提交到版本控制
# 检查.gitignore是否包含:
# data/
# results/
# checkpoints/
# *.log
# *.json
```

### 5.2 资源管理

```powershell
# 监控GPU使用情况
nvidia-smi -l 1  # 每秒刷新

# 设置GPU内存限制 (可选)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache(); print('GPU缓存已清理')"
```

### 5.3 进程管理

```powershell
# 后台运行实验 (Linux/Mac)
nohup python scripts/run_all_experiments.py --experiment all > experiment.log 2>&1 &

# 后台运行实验 (Windows PowerShell)
Start-Process -NoNewWindow python -ArgumentList "scripts/run_all_experiments.py --experiment all" -RedirectStandardOutput "experiment.log" -RedirectStandardError "error.log"

# 查看运行中的Python进程
tasklist | findstr python  # Windows
ps aux | grep python       # Linux/Mac
```

### 5.4 检查点保护

```powershell
# 定期备份检查点
# 程序自动每10轮保存一次检查点
# 手动备份:
Copy-Item -Path "checkpoints/*" -Destination "backup/checkpoints_$(Get-Date -Format 'yyyyMMdd_HHmmss')/" -Recurse
```

---

## 6. 数据记录方法

### 6.1 自动记录内容

每次实验自动生成以下文件:

```
results/{run_group}/{run_name}/
├── config.json          # 实验配置
├── metadata.json        # 运行元数据
├── experiment.log       # 运行日志
├── results.json         # 完整结果
└── checkpoints/         # 模型检查点
    ├── checkpoint_round_10.pt
    ├── checkpoint_round_20.pt
    └── best_model.pt
```

### 6.2 结果文件结构

```json
{
  "best_accuracy": 0.8523,
  "final_round": 100,
  "total_time": 12345.67,
  "convergence_round": 45,
  "dsnr_summary": {
    "mean": 2.345,
    "min": 0.123,
    "max": 5.678
  },
  "variance_summary": {
    "mean": 0.0012,
    "min": 0.0001,
    "max": 0.0056
  },
  "history": [
    {
      "round": 10,
      "accuracy": 0.6523,
      "loss": 0.8934,
      "dsnr": 1.234,
      "update_variance": 0.0012,
      "alignment_mean": 0.8567,
      "round_time": 123.45
    }
  ]
}
```

### 6.3 数据导出与分析

```powershell
# 导出为CSV格式
python -c "
import json
import pandas as pd

with open('results/experiment/results.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['history'])
df.to_csv('results/history.csv', index=False)
print('历史数据已导出到 history.csv')
"

# 生成实验报告
python scripts/run_analysis.py --results_dir results/
```

---

## 7. 异常情况处理

### 7.1 常见错误及解决方案

#### 错误1: CUDA内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案**:
```powershell
# 减小批大小
--batch_size 32

# 减少客户端数量
--clients_per_round 5

# 使用CPU (较慢)
--device cpu

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
```

#### 错误2: 数据集下载失败

```
ConnectionError: Failed to download dataset
```

**解决方案**:
```powershell
# 手动下载CIFAR-10
# 访问: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# 解压到 data/cifar10/

# 使用代理 (如需要)
set HTTP_PROXY=http://proxy:port
set HTTPS_PROXY=http://proxy:port
```

#### 错误3: 检查点加载失败

```
RuntimeError: Error loading checkpoint
```

**解决方案**:
```powershell
# 验证检查点文件
python -c "
import torch
try:
    ckpt = torch.load('checkpoints/best_model.pt')
    print('检查点加载成功')
    print(f'轮次: {ckpt.get(\"round\", \"未知\")}')
except Exception as e:
    print(f'检查点损坏: {e}')
"

# 从最近的检查点恢复
python scripts/run_experiment.py --resume checkpoints/round_90.pt ...
```

#### 错误4: 进程意外终止

**解决方案**:
```powershell
# 查看最后的日志
Get-Content experiment.log -Tail 50

# 从检查点恢复
python scripts/run_experiment.py --resume checkpoints/round_XX.pt --method dafa --dataset cifar10 ...
```

### 7.2 错误日志分析

```powershell
# 提取错误信息
Select-String -Path experiment.log -Pattern "Error|Exception|Failed" | Select-Object -Last 20

# 监控日志实时输出
Get-Content experiment.log -Wait
```

---

## 8. 外部环境运行指南

### 8.1 与实验室环境的差异调整

| 方面 | 实验室环境 | 外部环境 | 调整建议 |
|------|-----------|----------|----------|
| **GPU性能** | 高端GPU (A100) | 消费级GPU (RTX) | 减小batch_size, 减少clients_per_round |
| **网络** | 高速内网 | 家庭网络 | 数据集预下载, 减少网络依赖 |
| **电力** | UPS保障 | 可能不稳定 | 增加保存频率, 使用检查点恢复 |
| **监控** | 专业运维 | 自行监控 | 设置自动通知, 定期检查 |

### 8.2 远程监控方案

#### 方案A: 日志文件监控

```powershell
# 实时监控日志
Get-Content results\experiment.log -Wait

# 定期检查进度
while ($true) {
    $lastRound = (Select-String -Path results\experiment.log -Pattern "Round \d+" | Select-Object -Last 1)
    Write-Host "$(Get-Date): $lastRound"
    Start-Sleep -Seconds 300  # 每5分钟检查
}
```

#### 方案B: 邮件通知

```python
# 创建通知脚本 notify.py
import smtplib
from email.mime.text import MIMEText
import json
import sys

def send_notification(subject, body, to_email):
    # 配置SMTP服务器 (需要自行配置)
    smtp_server = "smtp.example.com"
    smtp_port = 587
    smtp_user = "your_email@example.com"
    smtp_pass = "your_password"
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = to_email
    
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)

if __name__ == "__main__":
    event = sys.argv[1]  # "start", "complete", "error"
    results_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if event == "complete" and results_file:
        with open(results_file) as f:
            data = json.load(f)
        body = f"实验完成!\n最佳准确率: {data['best_accuracy']:.4f}\n收敛轮次: {data.get('convergence_round', 'N/A')}"
        send_notification("DAFA实验完成", body, "your_email@example.com")
```

#### 方案C: Telegram Bot通知

```python
# telegram_notify.py
import requests
import json
import sys

BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=data)

if __name__ == "__main__":
    event = sys.argv[1]
    if event == "progress":
        round_num = sys.argv[2]
        accuracy = sys.argv[3]
        send_telegram(f"🔄 Round {round_num}: Accuracy = {accuracy}")
    elif event == "complete":
        results_file = sys.argv[2]
        with open(results_file) as f:
            data = json.load(f)
        send_telegram(f"✅ 实验完成!\n最佳准确率: {data['best_accuracy']:.4f}")
```

### 8.3 数据传输方案

#### 方案A: 云存储同步

```powershell
# 使用rclone同步到云存储
# 安装: https://rclone.org/

# 配置远程存储
rclone config

# 同步结果到云端
rclone sync results/ remote:DAFA_results/ --progress

# 定期自动同步
while ($true) {
    rclone sync results/ remote:DAFA_results/ --progress
    Start-Sleep -Seconds 3600  # 每小时同步
}
```

#### 方案B: SCP/SFTP传输

```powershell
# 使用WinSCP或命令行传输
# Linux/Mac:
scp -r results/ user@server:/path/to/backup/

# Windows (使用WinSCP脚本):
winscp.com /script=upload_script.txt

# upload_script.txt内容:
# open sftp://user:password@server
# put -delete results\* /remote/path/
# exit
```

#### 方案C: Git版本控制

```powershell
# 初始化结果仓库
cd results
git init
git remote add origin https://github.com/yourname/dafa-results.git

# 定期提交
git add .
git commit -m "Results update $(Get-Date -Format 'yyyyMMdd_HHmmss')"
git push origin main
```

### 8.4 断点续传策略

```powershell
# 自动恢复脚本 auto_resume.ps1
$max_retries = 5
$retry_count = 0

while ($retry_count -lt $max_retries) {
    python scripts/run_experiment.py --method dafa --dataset cifar10 --num_rounds 100
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "实验成功完成"
        break
    }
    
    $retry_count++
    Write-Host "实验中断，尝试恢复 ($retry_count/$max_retries)..."
    
    # 找到最新检查点
    $latest_checkpoint = Get-ChildItem -Path checkpoints -Filter "*.pt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if ($latest_checkpoint) {
        python scripts/run_experiment.py --resume $latest_checkpoint.FullName --method dafa --dataset cifar10 --num_rounds 100
    }
    
    Start-Sleep -Seconds 60
}
```

### 8.5 资源监控脚本

```powershell
# monitor.ps1 - 资源监控脚本
while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    # GPU状态
    $gpu_info = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    
    # CPU和内存
    $cpu = Get-WmiObject Win32_Processor | Measure-Object -Property LoadPercentage -Average
    $mem = Get-WmiObject Win32_OperatingSystem
    $mem_used = [math]::Round(($mem.TotalVisibleMemorySize - $mem.FreePhysicalMemory) / 1MB, 2)
    $mem_total = [math]::Round($mem.TotalVisibleMemorySize / 1MB, 2)
    
    # 磁盘
    $disk = Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='C:'"
    $disk_free = [math]::Round($disk.FreeSpace / 1GB, 2)
    
    Write-Host "[$timestamp] GPU: $gpu_info | CPU: $($cpu.Average)% | RAM: ${mem_used}/${mem_total}GB | Disk Free: ${disk_free}GB"
    
    # 警告阈值
    if ($disk_free -lt 10) {
        Write-Host "警告: 磁盘空间不足!" -ForegroundColor Red
    }
    
    Start-Sleep -Seconds 300
}
```

---

## 附录A: 快速启动命令

```powershell
# 完整实验流程 (复制粘贴执行)

# 1. 环境检查
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 2. 单次测试
python scripts/run_experiment.py --method dafa --dataset cifar10 --num_rounds 10 --device cuda

# 3. 正式实验
python scripts/run_all_experiments.py --experiment all --output_dir results

# 4. 查看结果
Get-ChildItem results -Recurse -Filter "results.json" | ForEach-Object { Write-Host $_.FullName; python -c "import json; print(json.load(open('$($_.FullName.Replace('\', '/'))'))['best_accuracy'])" }
```

---

## 附录B: 联系与支持

如遇到问题，请检查:
1. 本手册相关章节
2. `EXPERIMENT_DESIGN.md` 实验设计文档
3. `NeurlPS_Paper4.pdf` 理论基础
4. GitHub Issues (如有)

---

*文档版本: 1.0*
*最后更新: 2024年*
