# 环境安装与正式训练

以下命令以 Linux 服务器为主。推荐使用一张 A100/A800 80 GB；RTX 4090 也可，但
训练耗时和可用 batch/compile 行为需重新验证。工程已在 Ubuntu 24.04 + Apptainer、
A100 80 GB 上完成正式训练。

## 1. 目录与两个 Conda 环境

```bash
export ROOT=/path/to/GameWorld_DIAMOND
cd "$ROOT"
```

浏览器与 DIAMOND 使用两个环境，避免 Python 3.12 Playwright 依赖和 Python 3.10
PyTorch 训练依赖互相污染。

### 浏览器环境

```bash
conda create -n gameworld python=3.12 -y
conda activate gameworld
python -m pip install -r "$ROOT/GameWorld_ATARI_ENV/requirements.txt"

export PLAYWRIGHT_BROWSERS_PATH="$ROOT/.playwright-browsers"
python -m playwright install chromium
```

Playwright 安装到用户目录，不需要 sudo。`.playwright-browsers` 来自另一台 Linux
机器时可能可用，但跨 OS（例如 Linux 到 Windows）不可用，必须重新安装。

### DIAMOND 环境

```bash
conda create -n gw-diamond python=3.10 -y
conda activate gw-diamond
python -m pip install -r "$ROOT/diamond/requirements.txt"
```

若启用 `training.compile_wm=true`，TorchInductor/Triton 还需要可用的 C/C++ 编译器。
没有系统 gcc 权限时可安装到 Conda：

```bash
conda install -c conda-forge gcc_linux-64=12 gxx_linux-64=12 -y
```

检查 GPU：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Apptainer 中应确保工程、Conda 和 GPU 已绑定，并以 `apptainer exec --nv ...` 进入。
容器外的网页/SSH 断开不应影响后台进程，但必须用 `nohup`、`tmux`、`screen` 或
Slurm job 管理训练，不能依赖 VS Code Web 终端前台会话。

## 2. 环境回归测试

纯 Python 测试：

```bash
conda activate gameworld
cd "$ROOT/GameWorld_ATARI_ENV"
python -m unittest tools.diamond_bridge.test_breakout_bridge -v
```

Chromium live smoke test：

```bash
export PLAYWRIGHT_BROWSERS_PATH="$ROOT/.playwright-browsers"
python -m tools.diamond_bridge.smoke_test \
  --game-port 8291 --level 5 --seed 4242 --noop-max 30 --steps 20
```

期望：非终止 step 执行 4 帧、游戏时间约推进 66.667 ms、截图期间推进 0 ms，FIRE
能立即发球，返回帧包含黑球。

终止边界测试：

```bash
python -m tools.diamond_bridge.verify_terminal_boundaries \
  --game-port 8292 --level 5 --seed 4242
```

## 3. 启动训练和验证浏览器

终端 1：

```bash
conda activate gameworld
export ROOT=/path/to/GameWorld_DIAMOND
export PLAYWRIGHT_BROWSERS_PATH="$ROOT/.playwright-browsers"
cd "$ROOT/GameWorld_ATARI_ENV"
python -m tools.diamond_bridge.browser_env_server \
  --bind tcp://127.0.0.1:5661 --game-port 8201 --noop-max 30
```

终端 2：

```bash
conda activate gameworld
export ROOT=/path/to/GameWorld_DIAMOND
export PLAYWRIGHT_BROWSERS_PATH="$ROOT/.playwright-browsers"
cd "$ROOT/GameWorld_ATARI_ENV"
python -m tools.diamond_bridge.browser_env_server \
  --bind tcp://127.0.0.1:5662 --game-port 8202 --noop-max 30
```

长任务推荐把两条命令放入 `nohup`/tmux/Slurm，并分别记录日志和 PID。端口冲突时
可换四个端口，但训练命令的 `TRAIN_ENDPOINT`、`TEST_ENDPOINT` 必须同步更新。

## 4. 正式训练前检查

```bash
conda activate gw-diamond
cd "$ROOT/diamond"
python scripts/smoke_test_gameworld_atari_env.py \
  --endpoint tcp://127.0.0.1:5661 --level 5
python scripts/preflight_gameworld_atari_training.py \
  --level 5 \
  --train-endpoint tcp://127.0.0.1:5661 \
  --test-endpoint tcp://127.0.0.1:5662
```

Preflight 必须报告 `ok: true`、4 actions、frame skip 4、5 lives、无 step cap、
horizon 15、100k real steps、batch size 32、CUDA 可用。

## 5. 启动最终同配置训练

```bash
conda activate gw-diamond
cd "$ROOT/diamond"
export TRAIN_ENDPOINT=tcp://127.0.0.1:5661
export TEST_ENDPOINT=tcp://127.0.0.1:5662

RUN_NAME="atari_f4_level5_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/run_gameworld_atari_breakout.sh \
  "$RUN_NAME" 5 0 true \
  > "$ROOT/runtime_logs/${RUN_NAME}.launcher.log" 2>&1 &
echo $! > "$ROOT/runtime_logs/${RUN_NAME}.launcher.pid"
```

脚本参数依次为 run name、level、CUDA device、是否 compile world model。此交接
版本固定 4 帧/动作，正是最终 `atari_v9` run 的配置。Level 可改为 1--5，但一次
正式实验只训练一个 level。

不要在同一 GPU 上同时跑两个 DIAMOND 正式训练。显存虽然足够，但两份 world-model
训练会竞争相同 GPU 计算资源，实测总耗时可能比串行更差。

## 6. 监控

```bash
tail -f "$ROOT/runtime_logs/${RUN_NAME}.launcher.log"
cat "$ROOT/diamond/results/$RUN_NAME/validation_latest.json"
tail -n 5 "$ROOT/diamond/results/$RUN_NAME/validation_metrics.jsonl"
nvidia-smi
```

正式配置每 10 epoch 跑 4 局固定 seed validation；训练结束再跑 100 局 final
validation。`agent_versions` 每 100 epoch 保存一次，最多保留 4 个周期版本，并始终
强制保存最终 epoch。

## 7. 安全停止与恢复

先用 `ps -eo pid,ppid,etime,args | grep '[p]ython .*src/main.py'` 找到真正的 Python
trainer PID，向它发送 `kill -INT <TRAINER_PID>`。等待它退出后再结束两个 browser
service。已经完成的最近 epoch 会保留在 `checkpoints/state.pt`；被中断的当前 epoch
可能需要重做。不要 `kill -9`，除非进程完全无响应。

恢复前重新启动同样的 train/test browser，然后：

```bash
conda activate gw-diamond
cd "$ROOT/diamond/results/$RUN_NAME"
python src/main.py \
  common.resume=True \
  hydra.output_subdir=null \
  hydra.run.dir=. \
  env.train.endpoint=tcp://127.0.0.1:5661 \
  env.test.endpoint=tcp://127.0.0.1:5662 \
  common.devices=0
```

恢复要求 run 目录内仍有 `checkpoints/state.pt`、`dataset/`、`config/trainer.yaml`、
复制进去的 `src/` 与 `scripts/`。只有 `agent_epoch_01000.pt` 不能恢复优化器、replay
dataset 或 epoch 计数，只能用于推理/初始化。

## 8. Slurm

旧 0.25 秒环境的 Slurm 脚本已经删除，防止误用于最终 f4 方案。若迁移到 Slurm，
应在同一 allocation 中启动两个 `GameWorld_ATARI_ENV` 服务，再运行本节第 5 步的
正式入口。参见 `LEGACY_CONTENT.md` 的警告。
