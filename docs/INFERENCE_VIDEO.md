# 最终 checkpoint 的真实环境推理视频

该流程复现训练过程中的 real-environment test policy，而不是 GameWorld 官方评估，
也不是让 world model 自回归生成画面。

特性：

- actor 每一步只接收真实环境的 64×64 observation；
- 4 个 60 Hz 游戏帧/动作；
- 每个真实 60 Hz canvas frame 都写入视频；
- 视频保留 agent 相关区域的原生像素，不放大 64×64 输入；
- 底部显示动作、4 个动作概率、value、生命、分数和 progress；
- 5 命、无步数上限，直到清关或 game over；
- 默认 `mp4v`，Windows 常见播放器可直接播放。

实现代码：

```text
GameWorld_ATARI_ENV/tools/diamond_bridge/browser_env_server.py  # step_record
diamond/src/integrations/gameworld/atari_rpc_client.py          # recording client
diamond/src/evaluate_gameworld_atari_agent_60hz.py              # policy + MP4
```

## 1. 从 Hugging Face 下载最终模型

最终 Epoch 1000 checkpoint 已公开在：

<https://huggingface.co/FCZ7/gameworld-diamond-breakout-level5>

安装当前 Hugging Face CLI：

```bash
python -m pip install -U huggingface_hub
```

该仓库为 public，下载不要求登录。Linux：

```bash
export ROOT=/path/to/GameWorld_DIAMOND
mkdir -p "$ROOT/checkpoints/level5_f4"
hf download FCZ7/gameworld-diamond-breakout-level5 \
  agent_epoch_01000.pt \
  --local-dir "$ROOT/checkpoints/level5_f4"
export CHECKPOINT="$ROOT/checkpoints/level5_f4/agent_epoch_01000.pt"
sha256sum "$CHECKPOINT"
```

Windows PowerShell：

```powershell
$env:ROOT = "D:\path\to\GameWorld_DIAMOND"
New-Item -ItemType Directory -Force `
  "$env:ROOT\checkpoints\level5_f4" | Out-Null
hf download FCZ7/gameworld-diamond-breakout-level5 `
  agent_epoch_01000.pt `
  --local-dir "$env:ROOT\checkpoints\level5_f4"
$CHECKPOINT = "$env:ROOT\checkpoints\level5_f4\agent_epoch_01000.pt"
Get-FileHash -Algorithm SHA256 -LiteralPath $CHECKPOINT
```

正确文件大小为 54,309,858 bytes，SHA-256 必须是：

```text
cd45e24de20b9e7c1b52a52af08479c0a946d2187800c2d31ca7a6ed1bfd6244
```

若校验不一致，不要继续推理。checkpoint 的服务器来源和组成见
`RESULTS_AND_HUGGINGFACE.md`。

## 2. 启动专用浏览器服务

不要复用正在训练的 5661/5662。Linux：

```bash
conda activate gameworld
export ROOT=/path/to/GameWorld_DIAMOND
export PLAYWRIGHT_BROWSERS_PATH="$ROOT/.playwright-browsers"
cd "$ROOT/GameWorld_ATARI_ENV"
python -m tools.diamond_bridge.browser_env_server \
  --bind tcp://127.0.0.1:5675 --game-port 8375 --noop-max 30
```

Windows PowerShell：

```powershell
conda activate gameworld
$env:ROOT = "D:\path\to\GameWorld_DIAMOND"
cd "$env:ROOT\GameWorld_ATARI_ENV"
python -m tools.diamond_bridge.browser_env_server `
  --bind tcp://127.0.0.1:5675 --game-port 8375 --noop-max 30
```

Windows 的 PyZMQ asyncio 若提示 Proactor/tornado 错误，安装：

```powershell
python -m pip install "tornado>=6.1,<7"
```

## 3. 运行一关

另开终端：

```bash
conda activate gw-diamond
cd "$ROOT/diamond"
python src/evaluate_gameworld_atari_agent_60hz.py \
  --checkpoint "$CHECKPOINT" \
  --endpoint tcp://127.0.0.1:5675 \
  --level 5 \
  --game-seed 4242 \
  --policy-seed 20260720 \
  --device auto \
  --video-codec mp4v \
  --output-dir "$ROOT/inference_videos/epoch1000_level5"
```

输出：

```text
agent_real_60hz.mp4
preview_action_overlay.png
initial_real_observation_1280x720.png
trace.json
```

`trace.json` 记录 checkpoint SHA-256、每步动作概率、reward、progress、生命事件和
终止原因。

## 4. Level 1--5

Linux：

```bash
for level in 1 2 3 4 5; do
  python src/evaluate_gameworld_atari_agent_60hz.py \
    --checkpoint "$CHECKPOINT" \
    --endpoint tcp://127.0.0.1:5675 \
    --level "$level" --game-seed 4242 --policy-seed 20260720 \
    --video-codec mp4v \
    --output-dir "$ROOT/inference_videos/epoch1000_level${level}"
done
```

PowerShell：

```powershell
1..5 | ForEach-Object {
  python src/evaluate_gameworld_atari_agent_60hz.py `
    --checkpoint $CHECKPOINT `
    --endpoint tcp://127.0.0.1:5675 `
    --level $_ --game-seed 4242 --policy-seed 20260720 `
    --video-codec mp4v `
    --output-dir "$env:ROOT\inference_videos\epoch1000_level$_"
}
```

策略默认进行 categorical sample，与训练时 test collector 的 epsilon=0 口径一致。
添加 `--deterministic` 会改为 argmax，只适合额外诊断，不能和默认随机评估混为一谈。

## 5. 快速试跑

完整游戏可能很长。先验证依赖、视频编码和 RPC：

```bash
python src/evaluate_gameworld_atari_agent_60hz.py \
  --checkpoint "$CHECKPOINT" \
  --level 5 --safety-max-steps 20 --video-codec mp4v
```

正式结果必须使用 `--safety-max-steps 0`（默认），让环境自然终止。
