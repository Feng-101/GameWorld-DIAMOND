"""Web UI for inspecting discrete real GameWorld Breakout observations.

This viewer deliberately has no policy and no world model.  One key press
sends one discrete action to the browser RPC service.  The page keeps showing
the previous 64x64 agent observation until the approximately 0.25-second
GameWorld macro-step has completed, then replaces it with the returned frame.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import threading
import time
from typing import Any

from PIL import Image

from integrations.gameworld import (
    EXPECTED_EVALUATION_TIMING,
    BreakoutRPCClient,
    RPCObservation,
    preprocess_gameworld_frame,
)


ACTION_NAMES = ("wait", "left", "right")


def _rgb_png(rgb) -> bytes:
    output = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG")
    return output.getvalue()


class RealStepSession:
    """Own the single real browser session used by the step-by-step UI."""

    def __init__(
        self,
        client: BreakoutRPCClient,
        *,
        level: int,
        game_seed: int,
        initial_lives: int,
        observation_size: int = 64,
        check_timing: bool = True,
    ) -> None:
        if level not in range(1, 6):
            raise ValueError("level must be between 1 and 5")
        if initial_lives not in range(1, 6):
            raise ValueError("initial_lives must be between 1 and 5")
        if observation_size < 1:
            raise ValueError("observation_size must be positive")

        self.client = client
        self.observation_size = observation_size
        self.lock = threading.RLock()
        self.level = level
        self.game_seed = game_seed
        self.initial_lives = initial_lives
        self.metadata: dict[str, Any] = {}
        self.frame_png = b""
        self.last_action: int | None = None
        self.last_rpc_elapsed_s = 0.0
        self.terminal = False
        self.closed = False

        health = self.client.health()
        if check_timing and health.get("timing") != EXPECTED_EVALUATION_TIMING:
            raise RuntimeError(
                "Browser service timing does not match the evaluation cadence: "
                f"received={health.get('timing')!r}, "
                f"expected={EXPECTED_EVALUATION_TIMING!r}"
            )
        if health.get("viewport") not in (None, [1280, 720]):
            raise RuntimeError(
                "Browser viewport does not match the fixed observation crop: "
                f"{health.get('viewport')!r}"
            )
        self.health = health
        self.reset(level=level, game_seed=game_seed, initial_lives=initial_lives)

    def _accept_observation(
        self,
        observation: RPCObservation,
        *,
        elapsed_s: float,
        action: int | None,
    ) -> None:
        rgb = preprocess_gameworld_frame(
            observation.png,
            size=self.observation_size,
        )
        self.frame_png = _rgb_png(rgb)
        self.metadata = observation.metadata
        self.last_action = action
        self.last_rpc_elapsed_s = elapsed_s

        events = observation.metadata.get("transition_events")
        self.terminal = bool(
            isinstance(events, dict)
            and (
                events.get("task_success")
                or events.get("task_time_limit")
                or events.get("terminal_failure")
            )
        )

    def reset(
        self,
        *,
        level: int | None = None,
        game_seed: int | None = None,
        initial_lives: int | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            next_level = self.level if level is None else level
            next_seed = self.game_seed if game_seed is None else game_seed
            next_lives = self.initial_lives if initial_lives is None else initial_lives
            if next_level not in range(1, 6):
                raise ValueError("level must be between 1 and 5")
            if next_lives not in range(1, 6):
                raise ValueError("initial_lives must be between 1 and 5")
            if isinstance(next_seed, bool) or not isinstance(next_seed, int):
                raise ValueError("game_seed must be an integer")

            started = time.perf_counter()
            observation = self.client.reset(
                level=next_level,
                seed=next_seed,
                initial_lives=next_lives,
            )
            elapsed = time.perf_counter() - started
            self.level = next_level
            self.game_seed = next_seed
            self.initial_lives = next_lives
            self._accept_observation(observation, elapsed_s=elapsed, action=None)
            self.terminal = False
            return self.state()

    def step(self, action: int) -> dict[str, Any]:
        with self.lock:
            if action not in range(3):
                raise ValueError("action must be 0=wait, 1=left or 2=right")
            if self.terminal:
                raise RuntimeError("The real task ended; reset before taking another action")

            started = time.perf_counter()
            observation = self.client.step(action)
            elapsed = time.perf_counter() - started
            self._accept_observation(
                observation,
                elapsed_s=elapsed,
                action=action,
            )
            return self.state()

    def state(self) -> dict[str, Any]:
        with self.lock:
            state = self.metadata.get("state")
            game_state = state.get("game_state") if isinstance(state, dict) else None
            metrics = state.get("metrics") if isinstance(state, dict) else None
            timing = self.metadata.get("timing", self.health.get("timing", {}))
            return {
                "level": self.level,
                "game_seed": self.game_seed,
                "initial_lives": self.initial_lives,
                "observation_shape": [
                    self.observation_size,
                    self.observation_size,
                    3,
                ],
                "step_count": self.metadata.get("step_count", 0),
                "last_action": self.last_action,
                "last_action_name": (
                    ACTION_NAMES[self.last_action]
                    if self.last_action is not None
                    else None
                ),
                "score": game_state.get("score") if isinstance(game_state, dict) else None,
                "completion_progress": (
                    game_state.get("completion_progress")
                    if isinstance(game_state, dict)
                    else None
                ),
                "lives": metrics.get("lives") if isinstance(metrics, dict) else None,
                "transition_events": self.metadata.get("transition_events"),
                "terminal": self.terminal,
                "last_rpc_elapsed_s": self.last_rpc_elapsed_s,
                "timing": timing,
            }

    def get_frame_png(self) -> bytes:
        with self.lock:
            return self.frame_png

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.client.close()


VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>GameWorld Breakout — discrete agent view</title>
  <style>
    :root { color-scheme: dark; }
    body { margin:0; background:#101216; color:#eef2f7; font:16px system-ui,sans-serif; }
    main { max-width:850px; margin:24px auto; padding:0 18px; }
    .frame-shell { width:min(78vw,640px); aspect-ratio:1; background:#000; border:1px solid #56606d; }
    #frame { display:block; width:100%; height:100%; image-rendering:pixelated; }
    .row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:14px 0; }
    button,select,input { font:inherit; padding:8px 11px; }
    button { min-width:105px; }
    pre { padding:12px; background:#191d23; border:1px solid #343b45; overflow:auto; }
    .hint { color:#afd4ff; }
    #notice { color:#ffd37a; min-height:1.4em; }
  </style>
</head>
<body><main>
  <h2>GameWorld Breakout — discrete 64×64 agent observation</h2>
  <p class="hint">The image stays frozen during a macro-step. It is replaced only after the new observation returns.</p>
  <div class="frame-shell"><img id="frame" alt="current agent observation"></div>
  <div class="row">
    <button onclick="takeStep(1)">← / A: left</button>
    <button onclick="takeStep(0)">Space: wait</button>
    <button onclick="takeStep(2)">→ / D: right</button>
  </div>
  <div class="row">
    <label>Level
      <select id="level">
        <option value="1">1</option><option value="2">2</option>
        <option value="3">3</option><option value="4">4</option>
        <option value="5">5</option>
      </select>
    </label>
    <label>Seed <input id="seed" type="number" step="1"></label>
    <label>Lives
      <select id="lives">
        <option value="1">1</option><option value="2">2</option>
        <option value="3">3</option><option value="4">4</option>
        <option value="5">5</option>
      </select>
    </label>
    <button onclick="resetTask()">Reset / switch level</button>
  </div>
  <div id="notice"></div>
  <pre id="state"></pre>
</main>
<script>
let state = null;
let busy = false;
const frame = document.getElementById("frame");
const stateBox = document.getElementById("state");
const notice = document.getElementById("notice");

async function request(path, payload) {
  const options = payload === undefined ? {} : {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(payload)
  };
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || response.statusText);
  return result;
}
async function refresh() {
  state = await request("/api/state");
  frame.src = "/frame.png?t=" + Date.now();
  stateBox.textContent = JSON.stringify(state, null, 2);
  document.getElementById("level").value = String(state.level);
  document.getElementById("seed").value = String(state.game_seed);
  document.getElementById("lives").value = String(state.initial_lives);
  notice.textContent = state.terminal ? "Task ended. Reset before the next action." : "";
}
async function takeStep(action) {
  if (busy || (state && state.terminal)) return;
  busy = true;
  notice.textContent = "Executing one 0.25 s macro-step; the displayed frame is intentionally frozen…";
  try {
    await request("/api/step", {action:action});
    await refresh();
  } catch (error) {
    notice.textContent = error.message;
  } finally {
    busy = false;
  }
}
async function resetTask() {
  if (busy) return;
  busy = true;
  notice.textContent = "Resetting the selected level…";
  try {
    await request("/api/reset", {
      level:Number(document.getElementById("level").value),
      game_seed:Number(document.getElementById("seed").value),
      initial_lives:Number(document.getElementById("lives").value)
    });
    await refresh();
  } catch (error) {
    notice.textContent = error.message;
  } finally {
    busy = false;
  }
}
document.addEventListener("keydown", event => {
  if (event.repeat || event.target.matches("input,select")) return;
  if (event.key === "ArrowLeft" || event.key.toLowerCase() === "a") takeStep(1);
  else if (event.key === "ArrowRight" || event.key.toLowerCase() === "d") takeStep(2);
  else if (event.key === " " || event.key.toLowerCase() === "s") {
    event.preventDefault();
    takeStep(0);
  } else if (event.key.toLowerCase() === "r") resetTask();
});
refresh();
</script></body></html>"""


class RealStepRequestHandler(BaseHTTPRequestHandler):
    session: RealStepSession

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[real-viewer] {self.address_string()} - {fmt % args}", flush=True)

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(
        self,
        payload: Any,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self._send_bytes(
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("Request JSON must be an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path == "/" or self.path.startswith("/index.html"):
                self._send_bytes(VIEWER_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path.startswith("/frame.png"):
                self._send_bytes(self.session.get_frame_png(), "image/png")
            elif self.path.startswith("/api/state"):
                self._send_json(self.session.state())
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/api/step":
                result = self.session.step(int(payload.get("action")))
            elif self.path == "/api/reset":
                result = self.session.reset(
                    level=int(payload.get("level")),
                    game_seed=int(payload.get("game_seed")),
                    initial_lives=int(payload.get("initial_lives")),
                )
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(result)
        except (TypeError, ValueError, RuntimeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5581")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--game-seed", type=int, default=4242)
    parser.add_argument("--initial-lives", type=int, default=3)
    parser.add_argument("--observation-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = BreakoutRPCClient(args.endpoint)
    session: RealStepSession | None = None
    server: ThreadingHTTPServer | None = None
    try:
        session = RealStepSession(
            client,
            level=args.level,
            game_seed=args.game_seed,
            initial_lives=args.initial_lives,
            observation_size=args.observation_size,
        )
        handler = type(
            "BoundRealStepRequestHandler",
            (RealStepRequestHandler,),
            {"session": session},
        )
        server = ThreadingHTTPServer((args.host, args.port), handler)
        print(f"Discrete real GameWorld viewer: http://{args.host}:{args.port}", flush=True)
        print("Press Ctrl+C to stop. The browser RPC service is left running.", flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("Viewer interrupted.", flush=True)
    finally:
        if server is not None:
            server.server_close()
        if session is not None:
            session.close()
        else:
            client.close()


if __name__ == "__main__":
    main()
