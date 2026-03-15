# DAFA 复现实验设计文档

> 文档版本: v1.2  
> 更新日期: 2026-03-13  
> 目的: 系统性复现DAFA论文实验，回应评审意见

---
## 优先级声明 (给执行 Agent 的提示)
- **P0 (必须优先完成，放入正文)**: 阶段一(调参), 阶段二(Table 1主表), 阶段三(Fig 1收敛图, Fig 2 DSNR图), 实验5A(散点图，极大提升理论好感度)。
- **P1 (强烈建议完成，用于应对Reviewer)**: 实验4A(p_i消融), 实验4B(μ消融), 实验4C(γ消融), 实验4D(β消融)。
- **P2 (算力充裕时执行，放入Appendix)**: 实验5B(对抗客户端), 实验5C(大规模), 实验5D(异质性极端测试)。
---

---
## 一、实验总览

本实验分为五个阶段，旨在系统性验证DAFA方法的有效性并回应评审意见。

| 阶段 | 目标 | 回应评审意见 |
|------|------|--------------|
| 阶段一 | 标准化实验设置与公平调参 | Fair Tuning |
| 阶段二 | 核心性能对比 | Missing Baselines (Q4) |
| 阶段三 | 机制分析与去中心化DSNR | DSNR有效性 (Q1, Q8) |
| 阶段四 | 理论验证与消融实验 | 理论边界验证 |
| 阶段五 | 扩展实验与鲁棒性测试 | 鲁棒性与可扩展性 |

---

## 二、项目结构

```
DAFA/
├── configs/                     # 配置文件目录
│   ├── base_config.yaml         # 基础配置
│   ├── datasets/                # 数据集配置
│   │   ├── cifar10.yaml
│   │   ├── femnist.yaml
│   │   └── shakespeare.yaml
│   └── methods/                 # 各方法配置
│       ├── fedavg.yaml
│       ├── fedprox.yaml
│       ├── scaffold.yaml
│       ├── fednova.yaml
│       ├── fedavgm.yaml
│       ├── fedadam.yaml
│       ├── dir_weight.yaml
│       └── dafa.yaml
├── data/                        # 数据目录
│   ├── cifar10/
│   ├── femnist/
│   └── shakespeare/
├── src/                         # 源代码目录
│   ├── __init__.py
│   ├── data/                    # 数据处理模块
│   │   ├── __init__.py
│   │   ├── partition.py         # 数据分区 (Dirichlet)
│   │   ├── cifar10.py
│   │   ├── femnist.py
│   │   └── shakespeare.py
│   ├── models/                  # 模型定义
│   │   ├── __init__.py
│   │   ├── resnet.py            # ResNet-18
│   │   ├── cnn.py               # 2-layer CNN
│   │   └── lstm.py              # LSTM
│   ├── methods/                 # 聚合方法实现
│   │   ├── __init__.py
│   │   ├── base.py              # 基类 BaseAggregator
│   │   ├── fedavg.py
│   │   ├── fedprox.py
│   │   ├── scaffold.py
│   │   ├── fednova.py
│   │   ├── fedavgm.py
│   │   ├── fedadam.py
│   │   ├── dir_weight.py        # DAFA消融: 无动量版本
│   │   └── dafa.py
│   ├── analysis/                # 分析模块
│   │   ├── __init__.py
│   │   ├── dsnr.py              # DSNR计算
│   │   ├── variance.py          # 方差追踪
│   │   └── correlation.py       # 相关性分析
│   ├── utils/                   # 工具函数
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── seed.py
│   └── trainer.py               # 训练器 FederatedTrainer
├── scripts/                     # 运行脚本
│   ├── run_tuning.py            # 调参脚本
│   ├── run_experiment.py        # 单次实验
│   ├── run_grid_search.py       # 网格搜索
│   └── run_analysis.py          # 分析脚本
├── results/                     # 结果目录
│   ├── tuning/                  # 调参结果
│   ├── experiments/             # 实验结果
│   ├── analysis/                # 分析结果
│   │   ├── dsnr_curves/
│   │   └── correlation/
│   └── logs/                    # 日志
├── requirements.txt
└── README.md
```

---

## 三、数据集与模型配置

### 3.1 数据集配置详情

| 数据集 | 模型 | 分区方式 | 客户端数 | 本地Epoch | 通信轮数 | Batch Size |
|--------|------|----------|----------|-----------|----------|------------|
| CIFAR-10 | ResNet-18 | Dirichlet α=0.5 | 100 | 5 | 200 | 64 |
| CIFAR-10 | ResNet-18 | Dirichlet α=0.1 | 100 | 5 | 200 | 64 |
| FEMNIST | 2-layer CNN | Natural Partition | 采样100 | 10 | 200 | 64 |
| Shakespeare | LSTM | Natural Partition | 采样100 | 2 | 200 | 64 |

### 3.2 全局设置

- 客户端采样比例: 10% (每轮)
- 本地优化器: SGD with momentum=0.9
- 学习率调度: 固定（无衰减）
- 随机种子: [42, 123, 456, 789, 1024] (阶段二使用3-5个)

### 3.3 数据预处理

**CIFAR-10**:
```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])
```

**FEMNIST**:
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

**Shakespeare**:
- 字符级建模
- 序列长度: 80
- 词表大小: 80

---

## 四、阶段一：标准化实验设置与公平调参

### 4.1 目标

为所有基线方法找到最佳超参数，确保DAFA的胜利是令人信服的。

### 4.2 调参方法列表

| 序号 | 方法 | 调参参数 | 组合数 | 说明 |
|------|------|----------|--------|------|
| 1 | FedAvg | local_lr ∈ {0.1, 0.01, 0.001} | 3 | 基线 |
| 2 | FedProx | local_lr × μ ∈ {0.001, 0.01, 0.1} | 6 | 先用FedAvg最佳lr |
| 3 | SCAFFOLD | local_lr × global_lr ∈ {0.5, 1.0, 2.0} | 9 | 控制变量 |
| 4 | FedNova | local_lr ∈ {0.1, 0.01, 0.001} | 3 | 归一化聚合 |
| 5 | FedAvgM | local_lr × server_momentum ∈ {0.5, 0.7, 0.9, 0.99} | 12 | 服务器动量 |
| 6 | FedAdam | local_lr × server_lr ∈ {0.01, 0.1, 0.3} | 9 | 自适应优化 |
| 7 | Dir-Weight | local_lr × γ ∈ {0.5, 1.0, 2.0} | 9 | DAFA消融(β=0) |
| 8 | DAFA | local_lr × use_pi_weighting | 6 | 固定γ=1.0, β=0.9, μ=0.01 |

### 4.3 调参流程

```
Step 1: FedAvg调参
├── 对每个数据集 × 每个α设置
├── Grid Search: local_lr ∈ {0.1, 0.01, 0.001}
├── 评估指标: 最终测试准确率
└── 输出: 最佳local_lr

Step 2: FedProx调参 (基于FedAvg最佳lr)
├── 使用FedAvg最佳local_lr
├── Grid Search: μ ∈ {0.001, 0.01, 0.1}
└── 输出: 最佳μ

Step 3: SCAFFOLD调参
├── Grid Search: local_lr × global_lr (3×3=9组)
└── 输出: 最佳(local_lr, global_lr)

Step 4: FedNova调参
├── Grid Search: local_lr ∈ {0.1, 0.01, 0.001}
└── 输出: 最佳local_lr

Step 5: FedAvgM调参
├── Grid Search: local_lr × server_momentum (3×4=12组)
└── 输出: 最佳(local_lr, server_momentum)

Step 6: FedAdam调参
├── Grid Search: local_lr × server_lr (3×3=9组)
└── 输出: 最佳(local_lr, server_lr)

Step 7: Dir-Weight调参
├── Grid Search: local_lr × γ (3×3=9组)
├── 固定: β=0 (无动量代理)
└── 输出: 最佳(local_lr, γ)

Step 8: DAFA调参
├── Grid Search: local_lr ∈ {0.1, 0.01, 0.001}
├── 默认开启 p_i 数据量加权，固定 γ=1.0, β=0.9, μ=0.01
└── 输出: 最佳 local_lr
```

### 4.4 实验矩阵

| 数据集 | α设置 | 方法数 | 调参组合数 |
|--------|-------|--------|------------|
| CIFAR-10 | 0.5 | 8 | 57 |
| CIFAR-10 | 0.1 | 8 | 57 |
| FEMNIST | - | 8 | 57 |
| Shakespeare | - | 8 | 57 |
| **总计** | - | - | **228次** |

### 4.5 评估指标

- **主指标**: 最终测试准确率 (Final Test Accuracy)
- **辅助指标**:
  - 收敛速度 (达到90%最终精度的轮数)
  - 训练曲线稳定性 (最后50轮方差)

---

## 五、阶段二：核心性能对比

### 5.1 目标

生成论文中最核心的 **Table 1**，证明DAFA的改进来自方向对齐而非单纯动量。

### 5.2 对比方法

| 序号 | 方法 | 说明 |
|------|------|------|
| 1 | FedAvg | 基线 |
| 2 | FedProx | 近端项正则化 |
| 3 | SCAFFOLD | 控制变量 |
| 4 | FedNova | 归一化聚合 |
| 5 | **FedAvgM** | 服务器动量 (重点对比) |
| 6 | **FedAdam** | 自适应优化 (重点对比) |
| 7 | Dir-Weight | DAFA消融: 无动量版本 |
| 8 | **DAFA (Ours)** | 本文方法 |

### 5.3 关键对比组

| 对比组 | 目的 |
|--------|------|
| DAFA vs FedAvgM | 证明提升来自方向对齐，而非单纯动量 |
| DAFA vs Dir-Weight | 证明动量代理的价值 |
| DAFA vs FedAdam | 证明方向对齐优于自适应优化 |
| Dir-Weight vs FedAvg | 证明纯方向加权的价值 |

### 5.4 实验设置

- 使用阶段一得到的最佳超参数
- 每个方法跑 3-5 个随机种子
- 记录 Mean ± Std

### 5.5 评估指标

| 数据集 | 主指标 | 目标阈值（收敛速度） |
|--------|--------|----------------------|
| CIFAR-10 | Test Accuracy (%) | 达到80%准确率的轮数 |
| FEMNIST | Test Accuracy (%) | 达到80%准确率的轮数 |
| Shakespeare | Test Perplexity | 达到Perplexity=5的轮数 |

### 5.6 输出格式

**Table 1**: 各方法在三个数据集上的性能对比

```
| Method     | CIFAR-10 (α=0.5) | CIFAR-10 (α=0.1) | FEMNIST | Shakespeare |
|------------|------------------|------------------|---------|-------------|
| FedAvg     | 78.5 ± 0.3       | 72.1 ± 0.5       | ...     | ...         |
| FedProx    | 79.2 ± 0.4       | 73.0 ± 0.4       | ...     | ...         |
| SCAFFOLD   | 80.1 ± 0.3       | 75.2 ± 0.4       | ...     | ...         |
| FedNova    | 79.8 ± 0.3       | 74.5 ± 0.4       | ...     | ...         |
| FedAvgM    | 81.2 ± 0.2       | 77.8 ± 0.3       | ...     | ...         |
| FedAdam    | 80.8 ± 0.3       | 76.5 ± 0.4       | ...     | ...         |
| Dir-Weight | 80.5 ± 0.3       | 76.0 ± 0.4       | ...     | ...         |
| DAFA       | 82.1 ± 0.2       | 79.5 ± 0.3       | ...     | ...         |
```

---

## 六、阶段三：机制分析与去中心化DSNR

### 6.1 目标

生成 **Figure 1 (收敛曲线)** 和 **Figure 2 (DSNR与方差)**，验证DSNR代理指标有效性。

### 6.2 实验3A：DSNR与方差演化

**数据集**: CIFAR-10 α=0.1

**对比方法**: FedAvg, FedAvgM, SCAFFOLD, DAFA

**记录指标**:
- 中心化经验DSNR (每轮)
- 客户端更新方差 Var(Δi) (每轮)
- 测试准确率 (每轮)

**预期结果**:
- DAFA的方差显著低于FedAvgM
- 证明即使都有动量，DAFA也能进一步抑制噪声

### 6.3 实验3B：去中心化DSNR代理

**问题**: 中心化验证集不符合隐私要求

**解决方案**: 提出去中心化DSNR代理

**公式定义**:

中心化DSNR:
$$DSNR^t = \frac{\langle \Delta_{agg}^t, v_{true} \rangle^2}{\mathbb{E}[\|\Delta_{agg}^t - v_{true}\|^2]}$$

去中心化DSNR代理:
$$DSNR_{decentralized}^t = \frac{\langle \Delta_{agg}^t, m_t \rangle^2}{\frac{1}{|\mathcal{S}_t|}\sum_{i \in \mathcal{S}_t} \|\Delta_i^t - \Delta_{agg}^t\|^2}$$

其中:
- $\Delta_{agg}^t$: 第t轮聚合更新
- $m_t$: 服务器动量向量
- $\mathcal{S}_t$: 第t轮参与的客户端集合

**执行动作**:
1. 计算DAFA训练过程中的 $DSNR_{decentralized}^t$
2. 计算真实中心化DSNR
3. 绘制两者折线图
4. 计算Pearson相关系数

**预期结果**:
- 去中心化DSNR与中心化DSNR在趋势上高度一致
- Pearson相关系数 > 0.8

**论文Discussion**:
> "该指标可作为隐私合规的超参数(γ)调节依据"

### 6.4 输出

- **Figure 1**: 收敛曲线 (4条线: FedAvg, FedAvgM, SCAFFOLD, DAFA)
- **Figure 2**: DSNR与方差演化曲线
- **Figure 3**: 去中心化DSNR vs 中心化DSNR相关性图

---

## 七、阶段四：理论验证与消融实验

### 7.1 目标

验证理论边界，探索超参数敏感性，回应评审意见Q5和Q7。

### 7.2 实验4A：p_i数据量加权消融 (回应Q5)

**评审意见**: 要求对比有无数据集大小加权(p_i)的效果

**数据集**: FEMNIST (客户端数据量极度不平衡，最有说服力)

**对比组**:
| 变体 | 加权公式 | 说明 |
|------|----------|------|
| DAFA-u (Uniform) | $w_i \propto \exp(\gamma s_i)$ | 旧版，无数据量加权 |
| DAFA-p (Proportional) | $w_i \propto p_i \exp(\gamma s_i)$ | 新版，加入数据量加权 |

**实验设置**:
- 固定: γ=1.0, β=0.9
- 运行3个随机种子
- 记录: 最终准确率、长尾类别准确率

**预期结果**:
- DAFA-p 的最终准确率比 DAFA-u 高 1%~2%
- DAFA-p 在长尾类别上表现更好

**输出**: Table A1 (p_i消融结果)

### 7.3 实验4B：μ阈值截断消融 (回应Q7)

**评审意见**: 验证阈值μ的实际作用

**数据集**: CIFAR-10 (α=0.1)

**参数设置**:
| μ值 | 说明 |
|-----|------|
| 0.0 | 无截断 |
| 0.01 | 轻度截断 |
| 0.05 | 中度截断 |
| 0.1 | 强截断 |

**μ的作用**: 对更新范数过小的客户端进行得分清零（Graceful Degradation），防止纯噪声被 softmax 放大：
$$s_i = 0 \text{ if } \|\Delta_i\| < \mu$$
（此时该客户端的权重将自然退化为标准的 FedAvg 数据量加权 $w_i \propto p_i$）

**实验设置**:
- 固定: γ=1.0, β=0.9, local_lr=最佳值
- 运行3个随机种子
- 记录: 收敛曲线、末期抖动程度

**预期结果**:
- μ=0.0 时，训练末期（150轮后）准确率出现明显抖动或下降
- μ=0.01 时，末期收敛曲线平滑稳定
- μ过大时，有效客户端减少，性能下降

**输出**: Figure 4 (μ消融收敛曲线对比)

### 7.4 实验4C：γ温度参数消融

**数据集**: CIFAR-10 (α=0.1), FEMNIST

**参数设置**:
| γ值 | 说明 |
|-----|------|
| 0.1 | 接近均匀加权 |
| 0.5 | 轻度对齐 |
| 1.0 | 默认值 |
| 2.0 | 强对齐 |
| 5.0 | 过度聚焦 |

**实验设置**:
- 固定: β=0.9, local_lr=最佳值
- 运行3个随机种子

**预期结果**:
- γ过小: 退化为均匀加权，无法利用方向信息
- γ=1.0: 最佳平衡
- γ过大: 过度聚焦单一方向，忽略其他有用更新

**输出**: Table 2 (γ消融结果)

### 7.5 实验4D：β动量参数消融

**数据集**: CIFAR-10 (α=0.1), FEMNIST

**参数设置**:
| β值 | 说明 |
|-----|------|
| 0.0 | 无动量，退化为Dir-Weight |
| 0.5 | 轻度平滑 |
| 0.7 | 中度平滑 |
| 0.9 | 默认值 |
| 0.99 | 强平滑 |

**实验设置**:
- 固定: γ=1.0, local_lr=最佳值
- 运行3个随机种子

**预期结果**:
- β=0: 退化为Dir-Weight，代理方向不稳定
- β=0.9: 最佳平衡
- β过大: 历史信息过多，响应迟钝

**输出**: Table 3 (β消融结果)

### 7.6 消融实验汇总矩阵

| 实验 | 数据集 | 参数 | 组合数 | 种子数 | 总次数 |
|------|--------|------|--------|--------|--------|
| 4A | FEMNIST | 2变体 | 2 | 3 | 6 |
| 4B | CIFAR-10 | 4个μ值 | 4 | 3 | 12 |
| 4C | CIFAR-10, FEMNIST | 5个γ值 | 10 | 3 | 30 |
| 4D | CIFAR-10, FEMNIST | 5个β值 | 10 | 3 | 30 |
| **总计** | - | - | - | - | **78次** |

### 7.7 理论边界验证

验证Theorem 2中的聚合偏差界:
$$\|\Delta_{agg} - \Delta^*\| \leq O(\gamma \cdot \text{heterogeneity})$$

**方法**: 记录不同γ值下的聚合偏差与异质性指标，验证线性关系

---

## 八、阶段五：扩展实验与鲁棒性测试

### 8.1 目标

验证方法的鲁棒性与可扩展性，提供理论假设的实证支持（回应Q3）。

### 8.2 实验5A：方向得分与漂移相关性分析 (回应Q3)

**评审意见**: 定理3声称 $\Gamma_{DAFA} < \Gamma_{FedAvg}$，缺乏"对齐度越高、漂移越小"的严格实证依据

**数据集**: CIFAR-10 (α=0.1)

**采样轮次**: 第 50, 100, 150 轮

**Agent记录任务**:
1. 每一轮开始时，在包含所有数据的中央验证集上执行一次全批量梯度下降，得到真实的全局更新方向 $\Delta_*^t$。
2. 对被选中的每个Client $i$，计算其局部更新 $\Delta_i^t$ 和 代理方向得分 $s_i = \langle \frac{\Delta_i^t}{\|\Delta_i^t\|}, v_*^t \rangle$。
3. 计算该客户端的确定性漂移大小: $\|d_i\| = \|\Delta_i^t - \Delta_*^t\|$。

**可视化输出**:
- 横轴: 方向得分 $s_i \in [-1, 1]$
- 纵轴: 漂移大小 $\|d_i\|$
- 图表类型: 散点图 (Scatter Plot)
- 统计指标: Pearson相关系数

**预期结果**:
- 散点图呈现**左高右低**的负相关趋势
- 方向越偏（$s_i$接近-1），漂移$\|d_i\|$越大
- Pearson相关系数 < -0.5

**论文论证逻辑**:
> 因为DAFA赋予低$s_i$的客户端极小的权重$w_i$，所以它严格压制了大的$\|d_i\|$，从而证明 $\Gamma_{DAFA} < \Gamma_{FedAvg}$

**输出**: Figure 5 (漂移-对齐度散点图)

### 8.3 实验5B：对抗客户端鲁棒性

**数据集**: CIFAR-10 (α=0.1)

**设置**:
- 引入10%恶意客户端
- 恶意行为: 随机梯度 / 反向梯度
- 对比: FedAvg, FedAvgM, DAFA

**预期结果**:
- DAFA对恶意客户端有更强的鲁棒性
- 方向对齐机制能自动降低恶意更新的权重

### 8.4 实验5C：大规模客户端

**数据集**: CIFAR-10

**设置**:
- 客户端数: 100, 500, 1000
- 采样比例: 10%
- 对比: FedAvg, DAFA

**预期结果**:
- DAFA在大规模场景下保持稳定优势
- 计算开销增长可控

### 8.5 实验5D：不同异质性程度

**数据集**: CIFAR-10

**设置**:
- α ∈ {0.05, 0.1, 0.3, 0.5, 1.0}
- 对比: FedAvg, FedAvgM, DAFA

**预期结果**:
- α越小（异质性越强），DAFA优势越明显
- α=1.0时（接近IID），各方法差距缩小

### 8.6 阶段五实验汇总

| 实验 | 数据集 | 设置 | 种子数 | 总次数 |
|------|--------|------|--------|--------|
| 5A | CIFAR-10 | 3轮采样 | 3 | 9 |
| 5B | CIFAR-10 | 3方法×恶意比例 | 3 | 9 |
| 5C | CIFAR-10 | 3规模×2方法 | 3 | 18 |
| 5D | CIFAR-10 | 5α值×3方法 | 3 | 45 |
| **总计** | - | - | - | **81次** |

---

## 九、代码接口设计

### 9.1 核心类接口

```python
# src/methods/base.py
class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(self, global_model: nn.Module, 
                  client_updates: List[Dict], 
                  **kwargs) -> nn.Module:
        pass

# src/trainer.py
class FederatedTrainer:
    def __init__(self, config: Config, aggregator: BaseAggregator):
        pass
    
    def train(self) -> Dict[str, Any]:
        pass
    
    def evaluate(self) -> float:
        pass

# src/analysis/dsnr.py
class DSNRTracker:
    def compute_centralized_dsnr(self, agg_update, v_true) -> float:
        pass
    
    def compute_decentralized_dsnr(self, agg_update, momentum, 
                                    client_updates) -> float:
        pass
```

### 9.2 运行脚本接口

```python
# scripts/run_tuning.py
def run_grid_search(
    dataset: str,
    method: str,
    param_grid: Dict[str, List],
    num_seeds: int = 3
) -> pd.DataFrame:
    pass

# scripts/run_analysis.py
def analyze_dsnr_correlation(
    centralized_dsnr: List[float],
    decentralized_dsnr: List[float]
) -> Tuple[float, plt.Figure]:
    pass
```

---

## 十、实验时间估算

| 阶段 | 实验次数 | 单次时长 | 总时长 |
|------|----------|----------|--------|
| 阶段一 | 228 | 3-8小时 | ~1000小时 |
| 阶段二 | 96-160 | 3-8小时 | ~500小时 |
| 阶段三 | 4 | 3-8小时 | ~20小时 |
| 阶段四 | 78 | 3-8小时 | ~400小时 |
| 阶段五 | 81 | 3-8小时 | ~400小时 |
| **总计** | **~550** | - | **~2320小时** |

**建议**: 使用多GPU并行加速

---

## 十一、交付物清单

### 11.1 论文图表输出

| 序号 | 输出物 | 类型 | 来源阶段 | 说明 |
|------|--------|------|----------|------|
| 1 | **Table 1** | 表格 | 阶段二 | 8个方法的准确率/困惑度对比（含FedAvgM, FedAdam） |
| 2 | **Figure 1** | 图表 | 阶段二 | 收敛曲线对比（DAFA, FedAvg, FedAvgM, SCAFFOLD） |
| 3 | **Figure 2** | 图表 | 阶段三 | 中心化DSNR与更新方差的时间折线图 |
| 4 | **Table 2** | 表格 | 阶段四 | γ温度参数消融结果 |
| 5 | **Table 3** | 表格 | 阶段四 | β动量参数消融结果 |
| 6 | **Table A1** | 表格 | 阶段四 | DAFA-p vs DAFA-u对比（FEMNIST，回应Q5） |
| 7 | **Figure 4** | 图表 | 阶段四 | μ阈值消融收敛曲线（回应Q7） |
| 8 | **Figure 5** | 图表 | 阶段五 | 漂移-对齐度散点图（回应Q3） |
| 9 | **Figure 3** | 图表 | 阶段三 | 去中心化DSNR vs 中心化DSNR相关性 |

### 11.2 论文文本输出

| 序号 | 输出物 | 类型 | 来源阶段 | 说明 |
|------|--------|------|----------|------|
| 1 | **Main Results描述** | 文本 | 阶段二 | Table 1的详细分析与讨论 |
| 2 | **DSNR机制描述** | 文本 | 阶段三 | DSNR演化与方差抑制机制 |
| 3 | **去中心化DSNR讨论** | 文本 | 阶段三 | 证明DSNR_decentralized趋势与准确率吻合（回应Q8） |
| 4 | **消融实验分析** | 文本 | 阶段四 | γ, β, p_i, μ消融的详细分析 |
| 5 | **理论实证支持** | 文本 | 阶段五 | Γ_DAFF < Γ_FedAvg的实证论证 |

### 11.3 评审意见回应映射

| 评审问题 | 回应方式 | 输出物 |
|----------|----------|--------|
| Q1: DSNR界限推导 | 阶段三验证DSNR有效性 | Figure 2, Figure 3 |
| Q3: 异质性常数证明 | 阶段五漂移-对齐度分析 | Figure 5 |
| Q4: 缺少基线 | 阶段二新增FedAvgM, FedAdam | Table 1 |
| Q5: 数据量加权 | 阶段四p_i消融 | Table A1 |
| Q7: μ阈值作用 | 阶段四μ消融 | Figure 4 |
| Q8: 去中心化DSNR | 阶段三去中心化DSNR分析 | Figure 3 + Discussion文本 |

---

## 十二、参考文献

1. DAFA原论文: Directionally Aligned Federated Aggregation
2. FedAvg: McMahan et al., 2017
3. FedProx: Li et al., 2020
4. SCAFFOLD: Karimireddy et al., 2020
5. FedNova: Wang et al., 2020
6. FedAvgM: Hsu et al., 2019
7. FedAdam: Reddi et al., 2020

---

## 十三、更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-03-13 | 初始版本，完成阶段一至三设计 |
| v1.1 | 2026-03-13 | 补充阶段四详细设计：实验4A(p_i消融)、4B(μ阈值)、4C(γ消融)、4D(β消融)；更新实验矩阵 |
| v1.2 | 2026-03-13 | 补充阶段五详细设计：实验5A(漂移-对齐度分析)、5B(对抗鲁棒性)、5C(大规模)、5D(异质性)；新增交付物清单；完成评审意见回应映射 |
