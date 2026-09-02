# GameWorld Breakout integration

交接后的默认且唯一正式训练入口是 deterministic Atari-style 路线：

- browser：`GameWorld_ATARI_ENV`
- environment config：`config/env/gameworld_atari_breakout.yaml`
- experiment config：`config/experiment/gameworld_atari_breakout_formal.yaml`
- preflight：`scripts/preflight_gameworld_atari_training.py`
- launcher：`scripts/run_gameworld_atari_breakout.sh`
- real-environment video：`src/evaluate_gameworld_atari_agent_60hz.py`

该协议使用四动作 `NOOP/FIRE/RIGHT/LEFT`、每动作固定推进 4 个 60 Hz 游戏帧、
5 条命、无人工 step cap。截图和模型推理不推进游戏时间。每次丢命会形成训练用的
recurrent boundary，但只有第五次丢命 game-over 或清关才物理 reset 浏览器任务。

旧 `GameWorld_TRAIN_ENV` 的 0.2 秒动作保持、0.05 秒空转和运行中截图路线仍保留在
源码与旧配置中，只用于理解历史实验。它的训练、预检和 wrapper 脚本已从
`scripts/` 删除，不能作为新实验入口。

完整说明：

- `../../../../PROJECT_HANDOFF.md`
- `../../../../docs/SETUP_AND_TRAINING.md`
- `../../../../docs/LEGACY_CONTENT.md`
- `../../../../docs/INFERENCE_VIDEO.md`

