# `diamond/scripts` 交接说明

本目录已经移除会启动旧 0.25 秒宏步训练协议的脚本。正式训练只使用
`GameWorld_ATARI_ENV` 和下列 Atari-v9 入口。

| 文件 | 用途 |
|---|---|
| `preflight_gameworld_atari_training.py` | 正式训练前核验两个浏览器服务、协议和 Hydra 配置 |
| `run_gameworld_atari_breakout.sh` | Level 1-5 单关 specialist 正式训练入口，默认 Level 5 |
| `smoke_test_gameworld_atari_env.py` | 对固定 4 帧环境做短程真实 RPC 测试 |
| `resume.sh` | 从已有 Hydra run 目录恢复完整训练 |
| `run_gameworld_env_tests.py` | 从任意工作目录运行 GameWorld 相关单元测试 |
| `stage_gameworld_checkpoints.py` | 将版本 checkpoint 复制到便携目录并生成校验信息 |
| `watch_gameworld_validation.py` | 从旧式 console log 提取简洁 validation 记录 |
| `export_gameworld_preprocess_preview.py` | 离线检查截图裁剪和 64×64 policy observation |
| `measure_atari_breakout_speed.py` | 比较 ALE Atari Breakout 球速的研究诊断工具 |
| `summarize_gameworld_horizon_pair.py` | 只读解析早期 H=10/H=15 历史日志，不会启动训练 |
| `import_run.py` | DIAMOND 上游提供的远程 run 导入工具 |

不要使用 `config/env/gameworld_breakout*.yaml` 或
`config/experiment/gameworld_breakout*.yaml` 启动新训练；这些配置仅为历史审计保留。
