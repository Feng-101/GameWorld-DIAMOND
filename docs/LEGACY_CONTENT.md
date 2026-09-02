# 旧版内容与已知陷阱

工程保留旧实现用于审计和比较，但接手人必须区分以下两代环境。

## 1. 当前推荐：`GameWorld_ATARI_ENV`

固定 4×60 Hz frame step、截图时暂停、4 动作 FIRE 语义、5 命、无 step cap、单关
specialist。最终 `atari_v9_level5_20260721_163502` 来自该环境。

应使用：

- `config/env/gameworld_atari_breakout.yaml`
- `config/experiment/gameworld_atari_breakout_formal.yaml`
- `scripts/run_gameworld_atari_breakout.sh`
- `scripts/preflight_gameworld_atari_training.py`
- `src/evaluate_gameworld_atari_agent_60hz.py`

## 2. 旧版：`GameWorld_TRAIN_ENV`

旧环境试图模拟 GameWorld 原生评估 cadence：动作 0.2 秒、空转 0.05 秒、游戏仍运行
时截图，然后暂停。实测浏览器截图的墙钟开销会继续推进游戏，导致 observation 的
实际游戏时间间隔可能达到约 0.6 秒量级，远大于名义 0.25 秒。该时序对 Breakout
控制不利，早期 mixed-level 和旧 Level 5 训练结果不作为最终成果。

以下配置和 Python 实现仍属于这条旧路线，仅用于审计历史结果：

- `config/env/gameworld_breakout.yaml`
- `config/env/gameworld_breakout_level5.yaml`
- `config/experiment/gameworld_breakout_formal.yaml`
- `config/experiment/gameworld_breakout_level5_atari.yaml`
- `config/experiment/gameworld_breakout_horizon_probe.yaml`
- `config/experiment/gameworld_breakout_smoke.yaml`
- `src/evaluate_gameworld_real_agent.py`
- `src/view_gameworld_real_steps.py`
- `src/visualize_gameworld_world_model.py`（默认连接旧环境）

旧路线的训练 launcher、Slurm、预检、smoke test、边界诊断和 viewer wrapper 已从
`diamond/scripts/` 删除，避免接手人误启动失败协议。旧环境源码与配置仍保留，以便
解释实验演进和失败原因；除非明确做历史审计，否则不要执行上述旧 Python 入口。

## 3. 原始 `GameWorld/`

该目录是官方评估工程副本，包含手动加入的 `games/benchmark/05_breakout/`。训练
环境改动不应写回这里。官方评估和最终 deterministic training validation 是两个
不同协议，报告时不能混称。

## 4. 旧 PID、端口和缓存

- `runtime_logs/**/*.pid` 是旧服务器 PID，只作为记录。
- 5661/5662、8201/8202 是推荐默认端口，不是硬编码要求；端口占用时可以更换。
- `.torchinductor/` 与 `.playwright-browsers/` 是机器相关缓存，不是模型权重。
- `wandb/` 为 offline 日志；`wandb sync` 才会上传到云端。

## 5. 当前 git 状态

`diamond` 的 GameWorld 适配多数尚未进入上游 git commit。执行清理命令会破坏工程。
交接后应先创建新分支并提交当前全部改动，再进行重构。不要以为重新 clone 上游
DIAMOND 后复制少量 config 就能复现当前结果。

## 6. 新增视频诊断命令

交接整理时在固定 4 帧 browser protocol 中增加了 `step_record` 诊断命令，并添加
`src/evaluate_gameworld_atari_agent_60hz.py`。普通训练使用的 `step` 消息、4 帧推进和
observation 完全不变；只有明确调用 `step_record` 时才逐帧截取 canvas，用于生成
带动作标注的 60 Hz 视频。
