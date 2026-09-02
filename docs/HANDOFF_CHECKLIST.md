# 交接检查清单

## 原负责人在服务器释放前

- [ ] 下载 `agent_epoch_01000.pt`，核对 54,309,858 bytes 与文档 SHA-256。
- [ ] 下载最终 run 的 `config/trainer.yaml`、`validation_metrics.jsonl`、
      `validation_latest.json`。
- [ ] 决定是否需要续训；若需要，完整备份 `state.pt` 和 `dataset/`。
- [ ] 保留 run 内复制的 `src/`、`scripts/`，作为精确运行代码快照。
- [ ] 将 handoff 工程和 checkpoint 放入实验室持久存储，不只留在临时计算节点。
- [ ] 按 `MODEL_CARD_TEMPLATE.md` 建立 Hugging Face model card 和校验文件。

## 接手人首次运行

- [ ] 阅读 `PROJECT_HANDOFF.md` 和 `LEGACY_CONTENT.md`。
- [ ] 创建 Python 3.12 `gameworld` 与 Python 3.10 `gw-diamond` 两个环境。
- [ ] 安装/定位 Playwright Chromium，设置 `PLAYWRIGHT_BROWSERS_PATH`。
- [ ] 运行 browser protocol unit tests 与 live smoke test。
- [ ] 启动两个独立 browser service，运行 Atari training preflight。
- [ ] 用 `--safety-max-steps 5` 生成短 MP4，确认 checkpoint、GPU、视频 codec。
- [ ] 核对 Level 5 完整推理可自然得到 `level_cleared` 或 `game_over`。
- [ ] 把当前未提交的 DIAMOND 工作区提交到新的私有仓库并打版本 tag。

## 不能做

- [ ] 不运行 `git reset --hard` 或 `git clean -fd`。
- [ ] 不把 `GameWorld_TRAIN_ENV` 的旧脚本当作最终复现入口。
- [ ] 不把 training-time validation 写成官方 GameWorld evaluation。
- [ ] 不用单独的 `agent_epoch_01000.pt` 声称可以精确 resume training。
- [ ] 不在同一 GPU 上并发两个正式 DIAMOND world-model 训练。
