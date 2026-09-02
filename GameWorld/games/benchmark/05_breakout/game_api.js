(function () {
  const GAME_ID = "05_breakout";
  const DEFAULT_SEED = 42;

  const capabilities = {
    supports_seed: true,
    supports_level_select: true,
    supports_difficulty: false,
    supports_inplace_reset: true,
    supports_reload_reset: true,
    supports_pause_detection: false,
    supports_menu_detection: true,
    provides_actionable_flag: true
  };

  const session = {
    seed: DEFAULT_SEED,
    requestedLevel: null,
    requestedDifficulty: null,
    episodeStartMs: Date.now(),
    episodeCount: 0
  };

  const runtimeState = {
    lastState: null,
    lastLevel: null,
    lastBricksTotal: null,
    lastResetMethod: null,
    terminalLatch: {
      isTerminal: false,
      outcome: null,
      reason: null,
      ts: 0
    },
    terminalDisplay: {
      level: null,
      completionProgress: null,
      bricksRemaining: null,
      bricksTotal: null
    }
  };

  function getGame() {
    return typeof window !== "undefined" ? window.__breakoutGame || null : null;
  }

  function finiteOrNull(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  function intOrNull(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return null;
    return Math.trunc(value);
  }

  function clamp01(value) {
    if (!Number.isFinite(value)) return null;
    if (value <= 0) return 0;
    if (value >= 1) return 1;
    return value;
  }

  function normalizeLevel(value) {
    const numeric = intOrNull(value);
    if (numeric === null || numeric < 1) return null;
    const totalLevels =
      typeof window !== "undefined" &&
      window.Breakout &&
      Array.isArray(window.Breakout.Levels)
        ? window.Breakout.Levels.length
        : null;
    if (typeof totalLevels === "number" && totalLevels > 0) {
      return Math.max(1, Math.min(totalLevels, numeric));
    }
    return numeric;
  }

  function normalizeOptions(options) {
    const opts = options || {};
    return {
      seed: intOrNull(opts.seed),
      level:
        opts.level === undefined || opts.level === null
          ? null
          : normalizeLevel(opts.level),
      difficulty:
        opts.difficulty === undefined || opts.difficulty === null
          ? null
          : String(opts.difficulty)
    };
  }

  function getCurrentSeed() {
    if (typeof window !== "undefined" && typeof window.__getDeterministicSeed === "function") {
      try {
        return normalizeSeed(window.__getDeterministicSeed());
      } catch (error) {
        return session.seed;
      }
    }
    return session.seed;
  }

  function normalizeSeed(seed) {
    const numeric = Number(seed);
    if (!Number.isFinite(numeric)) {
      return DEFAULT_SEED;
    }
    return (numeric >>> 0);
  }

  function applySeed(seed) {
    const normalized = normalizeSeed(seed);
    if (typeof window !== "undefined" && typeof window.__setDeterministicSeed === "function") {
      try {
        return normalizeSeed(window.__setDeterministicSeed(normalized));
      } catch (error) {
        return normalized;
      }
    }
    return normalized;
  }

  function beginEpisode(options, appliedSeed, appliedLevel) {
    const accepted = normalizeOptions(options);
    const notes = [];
    const actualSeed = appliedSeed === null ? DEFAULT_SEED : normalizeSeed(appliedSeed);

    if (accepted.difficulty !== null) notes.push("difficulty_not_supported");

    session.seed = actualSeed;
    session.requestedLevel = accepted.level;
    session.requestedDifficulty = accepted.difficulty;
    session.episodeStartMs = Date.now();
    session.episodeCount += 1;

    runtimeState.lastState = null;
    runtimeState.lastLevel = null;
    runtimeState.terminalLatch.isTerminal = false;
    runtimeState.terminalLatch.outcome = null;
    runtimeState.terminalLatch.reason = null;
    runtimeState.terminalLatch.ts = 0;
    runtimeState.terminalDisplay.level = null;
    runtimeState.terminalDisplay.completionProgress = null;
    runtimeState.terminalDisplay.bricksRemaining = null;
    runtimeState.terminalDisplay.bricksTotal = null;

    return {
      accepted: accepted,
      applied: {
        seed: actualSeed,
        level: appliedLevel,
        difficulty: null
      },
      notes: notes
    };
  }

  function latchTerminal(outcome, reason, displayLevel, displayProgress) {
    runtimeState.terminalLatch.isTerminal = true;
    runtimeState.terminalLatch.outcome = outcome;
    runtimeState.terminalLatch.reason = reason;
    runtimeState.terminalLatch.ts = Date.now();
    runtimeState.terminalDisplay.level = displayLevel === undefined ? null : displayLevel;
    runtimeState.terminalDisplay.completionProgress =
      displayProgress === undefined ? null : displayProgress;
    runtimeState.terminalDisplay.bricksRemaining = null;
    runtimeState.terminalDisplay.bricksTotal = null;
  }

  function getLatchedTerminal(now) {
    if (!runtimeState.terminalLatch.isTerminal) return null;
    if ((now - runtimeState.terminalLatch.ts) > 2500) {
      runtimeState.terminalLatch.isTerminal = false;
      runtimeState.terminalLatch.outcome = null;
      runtimeState.terminalLatch.reason = null;
      runtimeState.terminalLatch.ts = 0;
      runtimeState.terminalDisplay.level = null;
      runtimeState.terminalDisplay.completionProgress = null;
      runtimeState.terminalDisplay.bricksRemaining = null;
      runtimeState.terminalDisplay.bricksTotal = null;
      return null;
    }
    return {
      isTerminal: true,
      outcome: runtimeState.terminalLatch.outcome,
      reason: runtimeState.terminalLatch.reason
    };
  }

  function countBricks(game) {
    if (!game || !game.court || !Array.isArray(game.court.bricks)) {
      return { remaining: null, total: null };
    }
    let remaining = 0;
    const total = game.court.bricks.length;
    for (let index = 0; index < total; index += 1) {
      if (!game.court.bricks[index].hit) remaining += 1;
    }
    return {
      remaining: finiteOrNull(remaining),
      total: finiteOrNull(total)
    };
  }

  function getPaddleState(paddle) {
    if (!paddle) return null;
    const dleft = finiteOrNull(paddle.dleft);
    const dright = finiteOrNull(paddle.dright);
    if (dleft !== null && dleft > 0) return "moving_left";
    if (dright !== null && dright > 0) return "moving_right";
    return "idle";
  }

  function getPaddleSnapshot(game) {
    if (!game || !game.paddle) return null;
    return {
      x: finiteOrNull(game.paddle.x),
      y: finiteOrNull(game.paddle.y),
      w: finiteOrNull(game.paddle.w),
      h: finiteOrNull(game.paddle.h),
      state: getPaddleState(game.paddle)
    };
  }

  function getBallSnapshot(game) {
    if (!game || !game.ball) return null;
    const dx = finiteOrNull(game.ball.dx);
    const dy = finiteOrNull(game.ball.dy);
    return {
      type: "ball",
      x: finiteOrNull(game.ball.x),
      y: finiteOrNull(game.ball.y),
      vx: dx,
      vy: dy,
      state: game.ball.moving ? "moving" : "attached",
      props: {
        countdown: intOrNull(game.ball.countdown),
        radius: finiteOrNull(game.ball.radius)
      }
    };
  }

  function getEnvironmentSnapshot(game, ball) {
    if (!game || !game.court) return null;
    return {
      court_left: finiteOrNull(game.court.left),
      court_top: finiteOrNull(game.court.top),
      court_width: finiteOrNull(game.court.width),
      court_height: finiteOrNull(game.court.height),
      launch_pending: !!(ball && ball.state === "attached"),
      launch_countdown: ball && ball.props ? ball.props.countdown : null
    };
  }

  function computeCompletionProgress(bricks) {
    if (
      bricks.total === null ||
      bricks.remaining === null ||
      bricks.total <= 0
    ) {
      return null;
    }
    return clamp01(1 - (bricks.remaining / bricks.total));
  }

  function buildUnavailableState(now) {
    return {
      schemaVersion: "2.0",
      gameId: GAME_ID,
      seed: session.seed,
      timestampMs: now,
      gameTimeMs: finiteOrNull(now - session.episodeStartMs),
      status: "loading",
      is_actionable: false,
      terminal: {
        isTerminal: false,
        outcome: null,
        reason: null
      },
      game_state: {
        score: null,
        level: null,
        player: null,
        environment: null,
        completion_progress: null
      },
      metrics: {
        primary_score: null,
        lives: null,
        bricks_remaining: null,
        bricks_total: null
      },
      debug: {
        ready: false,
        last_reset_method: runtimeState.lastResetMethod,
        episode_count: session.episodeCount
      }
    };
  }

  function runInplaceReset(game, appliedLevel) {
    if (!game) return false;

    if (appliedLevel !== null && typeof game.setLevel === "function") {
      game.setLevel(appliedLevel - 1);
    } else if (typeof game.resetLevel === "function") {
      game.resetLevel();
    }

    if (game.current === "menu" && typeof game.play === "function") {
      game.play();
      return true;
    }

    if (game.score && typeof game.score.reset === "function") {
      game.score.reset();
    }
    if (game.paddle && typeof game.paddle.reset === "function") {
      game.paddle.reset();
    }
    if (game.ball && typeof game.ball.reset === "function") {
      game.ball.reset({ launch: true });
    }
    return true;
  }

  function prepareEpisode(options) {
    const accepted = normalizeOptions(options);
    const appliedSeed = applySeed(accepted.seed === null ? getCurrentSeed() : accepted.seed);
    const game = getGame();
    const appliedLevel =
      accepted.level !== null
        ? accepted.level
        : (game && typeof game.level === "number" ? game.level + 1 : null);
    const episode = beginEpisode(accepted, appliedSeed, appliedLevel);
    return {
      episode: episode,
      game: game,
      appliedLevel: appliedLevel
    };
  }

  window.gameAPI = {
    version: "2.0",
    capabilities: capabilities,

    init: async function init(config) {
      const prepared = prepareEpisode(config || null);
      const notes = prepared.episode.notes.slice();

      try {
        if (runInplaceReset(prepared.game, prepared.appliedLevel)) {
          runtimeState.lastResetMethod = "inplace";
          return {
            ok: true,
            accepted: prepared.episode.accepted,
            applied: prepared.episode.applied,
            notes: notes
          };
        }
      } catch (error) {
        notes.push("init_inplace_reset_failed");
      }

      runtimeState.lastResetMethod = "reload";
      return {
        ok: true,
        accepted: prepared.episode.accepted,
        applied: prepared.episode.applied,
        notes: notes.concat(["state_prepared_without_immediate_reset"])
      };
    },

    getState: function getState() {
      const now = Date.now();
      const game = getGame();
      if (!game) {
        return buildUnavailableState(now);
      }

      const bricks = countBricks(game);
      const state = typeof game.current === "string" ? game.current : null;
      const level = finiteOrNull((game.level || 0) + 1);
      const score = game.score ? finiteOrNull(game.score.score) : null;
      const lives = game.score ? finiteOrNull(game.score.lives) : null;
      const player = getPaddleSnapshot(game);
      const ball = getBallSnapshot(game);
      const environment = getEnvironmentSnapshot(game, ball);
      let completionProgress = computeCompletionProgress(bricks);

      let terminal = null;
      if (lives !== null && lives <= 0 && state !== "game") {
        terminal = { isTerminal: true, outcome: "fail", reason: "no_lives_left" };
        latchTerminal(terminal.outcome, terminal.reason, level, completionProgress);
      } else if (
        runtimeState.lastState === "game" &&
        state === "game" &&
        runtimeState.lastLevel !== null &&
        level !== null &&
        level !== runtimeState.lastLevel
      ) {
        terminal = { isTerminal: true, outcome: "success", reason: "level_cleared" };
        latchTerminal(terminal.outcome, terminal.reason, runtimeState.lastLevel, 1);
        runtimeState.terminalDisplay.bricksRemaining = 0;
        runtimeState.terminalDisplay.bricksTotal = runtimeState.lastBricksTotal;
      } else if (runtimeState.lastState === "game" && state === "menu") {
        terminal = { isTerminal: true, outcome: "quit", reason: "returned_to_menu" };
        latchTerminal(terminal.outcome, terminal.reason, runtimeState.lastLevel, completionProgress);
      } else {
        terminal = getLatchedTerminal(now);
      }

      if (!terminal) {
        terminal = {
          isTerminal: false,
          outcome: null,
          reason: null
        };
      }

      if (terminal.isTerminal && terminal.outcome === "success") {
        completionProgress = 1;
      }

      const reportedLevel =
        terminal.isTerminal && runtimeState.terminalDisplay.level !== null
          ? runtimeState.terminalDisplay.level
          : level;
      const reportedBricksRemaining =
        terminal.isTerminal && runtimeState.terminalDisplay.bricksRemaining !== null
          ? runtimeState.terminalDisplay.bricksRemaining
          : bricks.remaining;
      const reportedBricksTotal =
        terminal.isTerminal && runtimeState.terminalDisplay.bricksTotal !== null
          ? runtimeState.terminalDisplay.bricksTotal
          : bricks.total;

      let status = "loading";
      if (terminal.isTerminal) {
        status = "terminal";
      } else if (state === "game") {
        status = "playing";
      } else if (state === "menu") {
        status = "menu";
      }

      runtimeState.lastState = state;
      runtimeState.lastLevel = level;
      runtimeState.lastBricksTotal = bricks.total;

      const gameState = {
        score: score,
        level: reportedLevel,
        player: player,
        environment: environment,
        completion_progress: completionProgress
      };

      if (ball) {
        gameState.entities = [ball];
      }

      return {
        schemaVersion: "2.0",
        gameId: GAME_ID,
        seed: session.seed,
        timestampMs: now,
        gameTimeMs: finiteOrNull(now - session.episodeStartMs),
        status: status,
        is_actionable: status === "playing",
        terminal: terminal,
        game_state: gameState,
        metrics: {
          primary_score: completionProgress,
          lives: lives,
          bricks_remaining: reportedBricksRemaining,
          bricks_total: reportedBricksTotal
        },
        debug: {
          state: state,
          last_reset_method: runtimeState.lastResetMethod,
          requested_level: session.requestedLevel,
          requested_difficulty: session.requestedDifficulty,
          ball_moving: !!(ball && ball.state === "moving")
        }
      };
    },

    reset: async function reset(options) {
      const prepared = prepareEpisode(options || null);
      const notes = prepared.episode.notes.slice();

      try {
        if (runInplaceReset(prepared.game, prepared.appliedLevel)) {
          runtimeState.lastResetMethod = "inplace";
          return {
            ok: true,
            method: "inplace",
            accepted: prepared.episode.accepted,
            applied: prepared.episode.applied,
            notes: notes
          };
        }
      } catch (error) {
        notes.push("inplace_reset_failed");
      }

      runtimeState.lastResetMethod = "reload";
      if (typeof window !== "undefined" && window.location && typeof window.location.reload === "function") {
        window.location.reload();
        return {
          ok: true,
          method: "reload",
          accepted: prepared.episode.accepted,
          applied: prepared.episode.applied,
          notes: notes.concat(["falling_back_to_page_reload"])
        };
      }

      runtimeState.lastResetMethod = "unsupported";
      return {
        ok: false,
        method: "unsupported",
        accepted: prepared.episode.accepted,
        applied: prepared.episode.applied,
        notes: notes.concat(["no_reset_method_available"])
      };
    }
  };
})();
