# Deterministic Atari-style bridge

Protocol version 9 exposes one paused browser as an emulator-like GameWorld
Breakout environment.

Requests contain one JSON frame:

- `{"cmd":"health"}`
- `{"cmd":"reset","level":5,"seed":42,"initial_lives":5}`
- `{"cmd":"step","action":0}`
- `{"cmd":"close"}`

Reset and step return JSON metadata followed by a normalized 1280×720 PNG.
Actions are:

- `0 = NOOP`
- `1 = FIRE`
- `2 = RIGHT`
- `3 = LEFT`

For a normal step, the selected key is held while exactly four calls to
`runner.update(1 / 60)` and `runner.draw()` execute. The returned PNG is the
last executed frame, captured while the virtual clock is stopped. Pixelwise
max pooling is intentionally disabled because it can erase GameWorld's moving
black ball against the light court. Terminal game-over or level-clear may end
the frame repeat early. Reset and every non-final life loss leave the ball
attached with no automatic countdown; `FIRE` invokes GameWorld's existing
immediate launch path.

`transition_events` includes raw non-negative score delta, bricks destroyed,
life loss, game-over, and level-clear. The latter two are mutually exclusive.
The Python DIAMOND adapter decides whether an intermediate life loss is a
logical recurrent-state boundary; the browser never resets it.

`health` advertises `max_steps: null`, five initial lives, zero-life game-over,
the four ALE action meanings, reset `noop_max=30`, and the exact timing
contract. DIAMOND rejects a
service whose contract differs.
