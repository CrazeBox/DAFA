# DAFA 复现指南

本指南对应当前仓库实现，目标是复现 `NeurlPS_Paper4.pdf` 中的核心实验，并与 `EXPERIMENT_DESIGN.md` 保持一致。

## 1. 论文默认实验设置

| 数据集 | 模型 | 总客户端/采样客户端 | 本地轮数 K | Batch Size | 通信轮数 T | 主指标 |
| --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 | ResNet-18 | 100 / 10 | 5 | 32 | 200 | Accuracy |
| FEMNIST | TwoLayerCNN | 自然分区 / 200 | 10 | 32 | 200 | Accuracy |
| Shakespeare | 2-layer LSTM, hidden=256, embedding=200 | 自然分区 / 100 | 2 | 32 | 200 | Perplexity |

当前 `scripts/run_experiment.py` 已内置这些默认值。只要指定 `--dataset`，其余关键参数会自动补齐；如果你手动传参，手动值优先。

## 2. 环境与数据

### 2.1 环境

推荐 Linux / WSL。项目自带环境脚本：

```bash
bash scripts/setup_env.sh --profile basic
```

### 2.2 数据准备

- `CIFAR-10` 支持自动下载。
- `FEMNIST` 和 `Shakespeare` 需要准备 `data/<dataset>/{train,test}/all_data.json`。
- 当前默认不会静默回退到 synthetic data；若只想做调试，可显式传 `--allow_synthetic_data true`。

## 3. 单次实验

### 3.1 CIFAR-10

```bash
python scripts/run_experiment.py \
  --method dafa \
  --dataset cifar10 \
  --alpha 0.1 \
  --device cuda
```

### 3.2 FEMNIST

```bash
python scripts/run_experiment.py \
  --method dafa \
  --dataset femnist \
  --device cuda
```

### 3.3 Shakespeare

```bash
python scripts/run_experiment.py \
  --method dafa \
  --dataset shakespeare \
  --device cuda
```

说明：
- `Shakespeare` 的主指标是 `perplexity`，结果文件中会同时保存 `accuracy` 和 `perplexity`。
- `dsnr` 现在优先记录论文定义下、基于小验证子集估计的 centralized DSNR；若验证集不可用，则退回经验 SNR 代理。

## 4. 五阶段实验

### Stage 1: 调参

```bash
python scripts/run_five_stages.py --stages 1 --device cuda --num_rounds 100 --seeds 42,123,456
```

输出：
- `results/five_stages/stage1_tuning/best_hyperparams_generated.yaml`
- `results/five_stages/stage1_tuning/stage1_summary.json`

### Stage 2: 主表

```bash
python scripts/run_five_stages.py --stages 2 --device cuda
```

输出：
- `results/five_stages/stage2_comparison/stage2_summary.json`

说明：
- `CIFAR-10/FEMNIST` 汇总 `accuracy`
- `Shakespeare` 汇总 `perplexity`

### Stage 3: 机制分析

```bash
python scripts/run_five_stages.py --stages 3 --device cuda
```

输出：
- `results/five_stages/stage3_mechanism/stage3_summary.json`

### Stage 4: 消融

```bash
python scripts/run_five_stages.py --stages 4 --device cuda
```

输出：
- `results/five_stages/stage4_ablation/stage4_summary.json`

### Stage 5: 扩展实验

```bash
python scripts/run_five_stages.py --stages 5 --device cuda
```

输出：
- `results/five_stages/stage5_extension/stage5_summary.json`

## 5. 结果目录

单次实验默认保存在：

```text
results/<run_group>/<dataset>_<method>_seed<seed>_<timestamp>/
```

关键文件：
- `config.json`: 实际运行参数
- `metadata.json`: 运行状态与摘要
- `results.json`: 训练历史和汇总指标
- `experiment.log`: 日志
- `checkpoints/`: 检查点

## 6. 测试

### 6.1 单元测试

```bash
pytest tests/test_all.py -v
```

### 6.2 环境检查脚本

`tests/test_ubuntu_environment.py` 是脚本，不是 pytest 用例：

```bash
python tests/test_ubuntu_environment.py
```

## 7. 已知边界

- `FEMNIST/Shakespeare` 没有真实数据时，synthetic fallback 会污染实验结论。
- `run_analysis.py` 对 `Shakespeare` 已可读取 `perplexity`，但更复杂的论文图表仍建议直接读取 `results.json` 二次分析。
- 评审关心的 fairness 指标 `bottom_10_accuracy` 主要适用于分类数据集；在 `Shakespeare` 上仅作为次要参考，不作为主表指标。
