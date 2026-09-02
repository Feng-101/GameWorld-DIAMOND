# 训练结果保留与 Hugging Face 发布

## 1. 服务器上要找的准确 run

```bash
RUN=/home/scc/pb23611950/GameWorld_DIAMOND/diamond/results/atari_v9_level5_20260721_163502
```

本地下载中缺少整个 `diamond/results`，因此最终模型目前**不在 handoff 文件夹中**。
至少需要从服务器补回下列文件。

## 2. Hugging Face 模型仓库的最低必要文件

最关键 checkpoint 是：

```text
$RUN/checkpoints/agent_versions/agent_epoch_01000.pt
```

这是完整 `Agent.state_dict`，同时包含：

- diffusion denoiser/world model；
- reward/end model；
- LSTM actor-critic policy/value network。

它不是只有 game actor，因此既可跑真实环境 agent，也可检查 world model。

此前从服务器取回并完成五关诊断的该文件应为 54,309,858 bytes，SHA-256：

```text
cd45e24de20b9e7c1b52a52af08479c0a946d2187800c2d31ca7a6ed1bfd6244
```

重新下载后必须核对；若不一致，先确认服务器文件和本地 `level_v9` 文件是否来自
同一 run，不要直接覆盖已验证版本。

同时上传：

```text
$RUN/config/trainer.yaml
$RUN/validation_metrics.jsonl
$RUN/validation_latest.json
$RUN/checkpoints/info_for_import_script.json   # 如果存在
```

另外从本工程附上：

```text
PROJECT_HANDOFF.md
docs/INFERENCE_VIDEO.md
runtime_logs/atari_v9_level5/atari_v9_level5_20260721_163502.launcher.log
```

完整 console log 约 100 MB，可放 Hugging Face dataset/release assets；模型仓库只放
压缩日志摘要也可以，但最终 100 局指标必须在 model card 中记录。

建议 Hugging Face 布局：

```text
gameworld-diamond-level5-f4/
├── agent_epoch_01000.pt
├── trainer.yaml
├── validation_metrics.jsonl
├── validation_latest.json
├── SHA256SUMS
├── README.md
└── logs/
    └── final_console.log.gz
```

生成校验值：

```bash
sha256sum agent_epoch_01000.pt trainer.yaml validation_metrics.jsonl \
  validation_latest.json > SHA256SUMS
```

## 3. 建议额外保留的版本

`checkpointing.num_to_keep=4`，正式 run 正常应保留最后四个周期版本，通常为
Epoch 700、800、900、1000。若存储允许，建议全部归档用于学习曲线/退化分析；公开
模型最低只需 Epoch 1000。

## 4. 续训归档与公开推理模型不是一回事

若未来可能续训，必须另行完整备份：

```text
$RUN/checkpoints/state.pt
$RUN/dataset/
$RUN/config/
$RUN/src/
$RUN/scripts/
$RUN/checkpoints/agent_versions/
```

`state.pt` 含 Trainer、模型、优化器、LR scheduler、epoch/RNG 等状态；`dataset/`
包含 replay 数据。二者通常才是 run 最大的部分。它们不必上传到公开模型仓库，但
应放实验室对象存储或私有 Hugging Face dataset，否则无法精确续训。

## 5. 推荐下载命令

在本地或目标服务器执行：

```bash
mkdir -p final_release
rsync -avP user@server:$RUN/checkpoints/agent_versions/agent_epoch_01000.pt final_release/
rsync -avP user@server:$RUN/config/trainer.yaml final_release/
rsync -avP user@server:$RUN/validation_metrics.jsonl final_release/
rsync -avP user@server:$RUN/validation_latest.json final_release/
```

注意：上例中的 `$RUN` 必须在发起 rsync 的 shell 中展开，或改成完整远程路径。

完整续训备份可使用：

```bash
rsync -avP --partial user@server:/home/scc/pb23611950/GameWorld_DIAMOND/diamond/results/atari_v9_level5_20260721_163502/ \
  atari_v9_level5_20260721_163502/
```

## 6. 已确认的最终结果

从完整 launcher log 提取：

| 指标 | 最终 100 局 Level 5 |
| --- | ---: |
| progress mean | 0.974583 |
| progress std | 0.066625 |
| success rate | 0.65 |
| native return mean | 10848.65 |
| native return std | 846.51 |
| task steps mean | 1905.01 |
| lives lost mean | 3.42 |

本地固定 seed=4242、policy seed=20260720 的 Epoch 1000 五关诊断：

| Level | progress | return | lives lost | terminal |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.000 | 2550 | 1 | level cleared |
| 2 | 0.791 | 4260 | 5 | game over |
| 3 | 0.175 | 1255 | 5 | game over |
| 4 | 0.566 | 2645 | 5 | game over |
| 5 | 1.000 | 11160 | 1 | level cleared |

五关表仅是一条固定随机策略轨迹/关，不应代替多 seed 统计；最终 100 局 Level 5
结果是正式汇报的主要指标。
