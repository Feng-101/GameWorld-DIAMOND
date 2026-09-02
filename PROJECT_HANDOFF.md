# GameWorld × DIAMOND 项目交接总览

## 1. 项目目标与最终状态

本项目将 DIAMOND（diffusion world model + imagined RL）迁移到浏览器版
GameWorld Breakout，并训练一个单关 specialist agent。

最终有效实验不是 GameWorld 原始 0.25 秒宏步版本，而是独立的
`GameWorld_ATARI_ENV`：保留 GameWorld Breakout 的画面、球速、关卡、计分和
碰撞物理，只把训练所需的关键时序与 episode 逻辑改为确定性 emulator 风格。

最终正式 run：

```text
diamond/results/atari_v9_level5_20260721_163502
```

关键配置：Level 5、4 个动作、每动作固定推进 4 个 60 Hz 游戏帧、5 命、无
人为步数上限、horizon 15、100k 真实环境交互步、总计 1000 epoch、
`torch.compile` 开启。

最终训练日志位于：

```text
runtime_logs/atari_v9_level5/atari_v9_level5_20260721_163502.launcher.log
```

日志记录的最终 100 局 Level 5 验证结果为：

- mean progress：0.97458
- progress std：0.06662
- success rate：0.65
- mean native return：10848.65
- mean lives lost：3.42

固定 seed 的本地诊断还显示 Epoch 1000 可清关 Level 1 和 Level 5；Level 2、
4 有一定泛化，Level 3 明显较弱。该诊断不是 GameWorld 官方评估。

## 2. 接手人首先应阅读什么

1. 本文件：项目结论、目录入口和应避免的旧代码。
2. [`docs/SETUP_AND_TRAINING.md`](docs/SETUP_AND_TRAINING.md)：从零安装、
   检查、启动、监控、停止和恢复训练。
3. [`docs/RESULTS_AND_HUGGINGFACE.md`](docs/RESULTS_AND_HUGGINGFACE.md)：
   从服务器保留哪些结果，以及 Hugging Face 上传清单。
4. [`docs/INFERENCE_VIDEO.md`](docs/INFERENCE_VIDEO.md)：使用最终 checkpoint
   在真实固定帧环境中推理并生成带动作标注的 60 Hz MP4。
5. [`docs/LEGACY_CONTENT.md`](docs/LEGACY_CONTENT.md)：旧环境、旧配置和旧脚本，
   防止误用。
6 [`docs/HANDOFF_CHECKLIST.md`](docs/HANDOFF_CHECKLIST.md)：移交双方逐项确认。

## 3. 顶层目录说明

| 目录 | 状态 | 用途 |
| --- | --- | --- |
| `diamond/` | 当前核心代码 | 原 DIAMOND 仓库及 GameWorld adapter、训练配置、测试和推理脚本。原始上游 commit 为 `5bcd159`，大量项目修改尚未提交到 git。交接时必须整体保留。 |
| `GameWorld_ATARI_ENV/` | **最终推荐** | 独立、确定性的固定 4 帧 Breakout 训练/验证环境。正式 Level 5 成果来自这里。 |
| `GameWorld/` | 官方评估副本 | 保留 GameWorld 原项目和 `05_breakout`。只用于理解/运行原评估，不要把训练改动写入这里。 |
| `GameWorld_TRAIN_ENV/` | **旧版保留** | 早期 0.2 秒按键 + 0.05 秒空转、截图时游戏仍运行的宏步训练环境。用于历史对照，不再推荐训练。 |
| `runtime_logs/` | 证据/结果摘要 | 最终正式训练完整 console log、浏览器日志和 PID 记录。PID 文件只是旧服务器进程号，不可复用。 |
| `.playwright-browsers/` | 机器依赖缓存 | 服务器下载的 Playwright Chromium。它与操作系统/架构相关，不应上传到模型仓库；新机器最好重新安装。 |
| `tmp/` | 临时文件 | 非正式产物，可检查后清理。 |

## 4. 最终训练数据流

```text
GameWorld_ATARI_ENV (train browser, tcp://127.0.0.1:5661)
  -> 真实 64x64 observation / score delta / life / terminal events
  -> DIAMOND train replay dataset
  -> diffusion denoiser + reward/end model
  -> WorldModelEnv 中 horizon=15 的 imagined rollout
  -> LSTM actor-critic 更新
  -> 新策略继续采集真实数据

GameWorld_ATARI_ENV (validation browser, tcp://127.0.0.1:5662)
  -> 与训练 browser 完全独立，周期性跑完整 5 命游戏
  -> validation_metrics.jsonl / validation_latest.json / W&B offline log
```

训练和验证必须使用两个独立浏览器服务。验证 reset 不得影响训练中的物理游戏。

## 5. 最终环境契约

- 动作：`0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT`。
- 每次非终止动作同步推进 4 次 `runner.update(1/60)` 与 draw，即 0.066667 秒
  游戏时间。
- 浏览器保持暂停；截图、ZeroMQ、模型推理、磁盘 I/O 和真实墙钟耗时均不推进
  游戏时间。
- observation 为最后一个已执行帧；不做 Atari 两帧 max-pooling，因为黑球会被
  浅色背景抹掉。
- reset 为 5 命并执行 1--30 个隐藏 NOOP；球等待 FIRE 后立即发射，无 2 秒倒计时。
- 前 4 次丢命继续同一物理棋盘；第 5 命耗尽为 `game_over`。
- 清空砖块为独立的 `level_cleared` 成功终止，不自动进入下一关。
- 训练和测试均没有步数上限。
- 训练 collector 在中间丢命时重置 policy recurrent state，但不 reset 物理棋盘；
  validation 保持完整五命 episode。

## 6. DIAMOND 中的主要改动

- `src/envs/gameworld_atari_breakout.py`：4 动作、固定帧 RPC 环境 adapter。
- `src/integrations/gameworld/atari_rpc_client.py`：版本化 ZeroMQ client。
- `src/integrations/gameworld/preprocess.py`：从 1280×720 截取游戏区域并缩放到
  64×64。
- `src/envs/factory.py`：GameWorld 环境构造与配置验证。
- `src/coroutines/collector.py`：浏览器任务边界、生命边界与 GameWorld 指标。
- `src/trainer.py`：固定实验 seed、隔离 validation RNG、持久化验证指标、每 100
  epoch 保存版本 checkpoint，并强制保存最终 checkpoint。
- `src/models/rew_end_model.py`：适配 GameWorld reward/end 标签。
- `src/validation_logging.py`：写入 `validation_metrics.jsonl` 和
  `validation_latest.json`。
- `config/env/gameworld_atari_breakout.yaml`：最终单关环境。
- `config/experiment/gameworld_atari_breakout_formal.yaml`：正式 100k Atari 风格
  训练预算。
- `scripts/run_gameworld_atari_breakout.sh`：最终非 Slurm 训练入口。
- `scripts/preflight_gameworld_atari_training.py`：训练前的环境/模型/预算检查。
- `src/evaluate_gameworld_atari_agent_60hz.py`：最终真实环境推理与 60 Hz 视频。

## 7. 版本控制注意事项

`diamond/.git` 仍指向上游 DIAMOND，但当前 GameWorld 工作大多是未提交文件和工作区
修改。不要执行 `git reset --hard`、`git clean -fd` 或直接重新 clone 覆盖，否则会
删除本项目的核心实现。交接后的第一项版本管理工作应是：审核当前 diff，将完整
工程提交到新的私有仓库，并为最终代码打 tag。

`GameWorld/` 自身有 git 历史，其中 `games/benchmark/05_breakout/` 是未跟踪加入的
游戏目录，也不能被 `git clean` 删除。

## 8. 成果边界

该 checkpoint 证明 DIAMOND 能在我们构造的确定性 GameWorld Breakout Level 5
训练环境中学习出强 specialist。它不等于已完成 GameWorld 官方评估接入：官方环境
的墙钟截图/动作节奏与本训练环境不同。所有报告必须分别标注：

- `training-time deterministic real-environment validation`；或
- `official GameWorld evaluation`。

目前保存的最终指标属于前者。
