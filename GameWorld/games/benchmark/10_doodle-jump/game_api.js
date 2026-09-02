(function () {
  'use strict';

  var GAME_ID = '10_doodle-jump';
  var DEFAULT_SEED = 42;
  var MAX_PLATFORM_ENTITIES = 12;

  var PLATFORM_KIND = {
    1: 'normal',
    2: 'moving',
    3: 'breakable',
    4: 'vanishing'
  };

  var capabilities = {
    supports_seed: false,
    supports_level_select: false,
    supports_difficulty: false,
    supports_inplace_reset: true,
    supports_reload_reset: true,
    supports_pause_detection: false,
    supports_menu_detection: true,
    provides_actionable_flag: true
  };

  var session = {
    seed: DEFAULT_SEED,
    requestedLevel: null,
    requestedDifficulty: null,
    episodeStartMs: Date.now(),
    episodeCount: 0
  };

  var runtime = {
    attempts: 0,
    gameplayStartMs: null,
    lastResetMethod: null
  };

  function finiteNumber(value) {
    return (typeof value === 'number' && Number.isFinite(value)) ? value : null;
  }

  function finiteInt(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return null;
    return Math.trunc(value);
  }

  function normalizeSeed(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return null;
    return value >>> 0;
  }

  function normalizeOptions(options) {
    var opts = options || {};
    return {
      seed: normalizeSeed(opts.seed),
      level:
        opts.level === undefined || opts.level === null
          ? null
          : (typeof opts.level === 'number' && Number.isFinite(opts.level))
            ? Math.trunc(opts.level)
            : String(opts.level),
      difficulty:
        opts.difficulty === undefined || opts.difficulty === null
          ? null
          : String(opts.difficulty)
    };
  }

  function beginEpisode(options) {
    var accepted = normalizeOptions(options);
    var notes = [];

    if (accepted.seed !== null) notes.push('seed_not_supported');
    if (accepted.level !== null) notes.push('level_not_supported');
    if (accepted.difficulty !== null) notes.push('difficulty_not_supported');

    session.seed = DEFAULT_SEED;
    session.requestedLevel = accepted.level;
    session.requestedDifficulty = accepted.difficulty;
    session.episodeStartMs = Date.now();
    session.episodeCount += 1;

    runtime.attempts += 1;
    runtime.gameplayStartMs = null;

    return {
      accepted: accepted,
      applied: {
        seed: session.seed,
        level: null,
        difficulty: null
      },
      notes: notes
    };
  }

  function getMenuVisible(id) {
    if (typeof document === 'undefined') return null;
    var el = document.getElementById(id);
    if (!el) return null;
    var style = window.getComputedStyle ? window.getComputedStyle(el) : null;
    if (!style) return null;
    return (
      style.visibility !== 'hidden' &&
      style.display !== 'none' &&
      style.pointerEvents !== 'none' &&
      style.zIndex !== '-1'
    );
  }

  function getPlayer() {
    return (typeof window !== 'undefined' && window.player) ? window.player : null;
  }

  function getPlatforms() {
    return (typeof window !== 'undefined' && Array.isArray(window.platforms)) ? window.platforms : null;
  }

  function getSpring() {
    return (typeof window !== 'undefined' && window.Spring) ? window.Spring : null;
  }

  function getBase() {
    return (typeof window !== 'undefined' && window.base) ? window.base : null;
  }

  function platformKind(type) {
    return PLATFORM_KIND[type] || 'unknown';
  }

  function isVisibleVertical(y, height) {
    if (y === null) return false;
    var h = height === null ? 0 : height;
    return y >= -h && y <= 552 + h;
  }

  function collectVisiblePlatforms(platforms) {
    if (!platforms) return [];
    var visible = [];

    for (var i = 0; i < platforms.length; i += 1) {
      var platform = platforms[i];
      if (!platform) continue;
      var y = finiteNumber(platform.y);
      if (!isVisibleVertical(y, finiteNumber(platform.height))) continue;
      visible.push(platform);
    }

    visible.sort(function (left, right) {
      var leftY = finiteNumber(left && left.y);
      var rightY = finiteNumber(right && right.y);
      if (leftY === null && rightY === null) return 0;
      if (leftY === null) return 1;
      if (rightY === null) return -1;
      return leftY - rightY;
    });

    return visible;
  }

  function buildPlayerState(player) {
    if (!player) return null;
    var deadRaw = (player.isDead === true || player.isDead === 'lol');
    return {
      x: finiteNumber(player.x),
      y: finiteNumber(player.y),
      vx: finiteNumber(player.vx),
      vy: finiteNumber(player.vy),
      dir: (typeof player.dir === 'string') ? player.dir : null,
      is_dead: deadRaw
    };
  }

  function buildEnvironment(visiblePlatforms, spring, base) {
    var counts = {
      normal: 0,
      moving: 0,
      breakable: 0,
      vanishing: 0
    };

    for (var i = 0; i < visiblePlatforms.length; i += 1) {
      var kind = platformKind(finiteInt(visiblePlatforms[i].type));
      if (Object.prototype.hasOwnProperty.call(counts, kind)) {
        counts[kind] += 1;
      }
    }

    var springX = spring ? finiteNumber(spring.x) : null;
    var springY = spring ? finiteNumber(spring.y) : null;
    var springHeight = spring ? finiteNumber(spring.height) : null;
    var springVisible = springY !== null && isVisibleVertical(springY, springHeight);

    return {
      viewport: {
        width: 422,
        height: 552,
        wraps_horizontally: true
      },
      visible_platform_count: finiteInt(visiblePlatforms.length),
      visible_platform_type_counts: counts,
      base_y: base ? finiteNumber(base.y) : null,
      spring: {
        visible: springVisible,
        x: springX,
        y: springY,
        state: spring ? finiteInt(spring.state) : null
      }
    };
  }

  function serializeEntities(visiblePlatforms, spring) {
    var entities = [];

    for (var i = 0; i < visiblePlatforms.length && entities.length < MAX_PLATFORM_ENTITIES; i += 1) {
      var platform = visiblePlatforms[i];
      if (!platform) continue;
      entities.push({
        type: 'platform',
        x: finiteNumber(platform.x),
        y: finiteNumber(platform.y),
        vx: finiteNumber(platform.vx),
        props: {
          platform_kind: platformKind(finiteInt(platform.type)),
          state: finiteInt(platform.state),
          flag: finiteInt(platform.flag)
        }
      });
    }

    if (spring) {
      var springY = finiteNumber(spring.y);
      if (isVisibleVertical(springY, finiteNumber(spring.height))) {
        entities.push({
          type: 'spring',
          x: finiteNumber(spring.x),
          y: springY,
          props: {
            state: finiteInt(spring.state)
          }
        });
      }
    }

    return entities.length ? entities : null;
  }

  function getScore() {
    if (typeof window === 'undefined') return null;
    return finiteInt(window.score);
  }

  function getStatus(player, gameOverVisible, menuVisible) {
    if (!player) return 'loading';
    if (gameOverVisible === true || player.isDead === true || player.isDead === 'lol') return 'terminal';
    if (menuVisible === true) return 'menu';
    return 'playing';
  }

  function getTerminal(status) {
    if (status === 'terminal') {
      return {
        isTerminal: true,
        outcome: 'fail',
        reason: 'fell_off_screen'
      };
    }
    return {
      isTerminal: false,
      outcome: null,
      reason: null
    };
  }

  function getGameTimeMs(now, status) {
    if (status === 'playing' && runtime.gameplayStartMs === null) {
      runtime.gameplayStartMs = now;
    }
    if (runtime.gameplayStartMs === null) return null;
    return finiteNumber(now - runtime.gameplayStartMs);
  }

  function buildLoadingState(now) {
    return {
      schemaVersion: '2.0',
      gameId: GAME_ID,
      seed: session.seed,
      timestampMs: now,
      gameTimeMs: null,
      status: 'loading',
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
        distance: null,
        platforms_visible: null,
        attempts: finiteInt(runtime.attempts)
      },
      debug: {
        menu_visible: getMenuVisible('mainMenu'),
        game_over_visible: getMenuVisible('gameOverMenu'),
        first_run: (typeof window !== 'undefined' && typeof window.firstRun === 'boolean') ? window.firstRun : null,
        current_seed: session.seed,
        last_reset_method: runtime.lastResetMethod
      }
    };
  }

  // Keep low-level reset details here instead of exposing extra snapshot fields.
  function performReset() {
    if (typeof window !== 'undefined' && typeof window.__resetRandom === 'function') {
      try {
        window.__resetRandom();
      } catch (error) {}
    }

    if (typeof window !== 'undefined' && typeof window.reset === 'function') {
      try {
        window.reset();
        return 'inplace';
      } catch (error) {}
    }

    if (typeof window !== 'undefined' && typeof window.init === 'function') {
      try {
        window.init();
        return 'soft_restart';
      } catch (error) {}
    }

    if (typeof window !== 'undefined' && window.location && typeof window.location.reload === 'function') {
      window.location.reload();
      return 'reload';
    }

    return 'unsupported';
  }

  window.gameAPI = {
    version: '2.0',
    capabilities: capabilities,

    init: async function init(config) {
      var episode = beginEpisode(config);
      runtime.lastResetMethod = null;
      return {
        ok: true,
        accepted: episode.accepted,
        applied: episode.applied,
        notes: episode.notes.length ? episode.notes : []
      };
    },

    reset: async function reset(options) {
      var episode = beginEpisode(options || null);
      var method = performReset();
      runtime.lastResetMethod = method;
      return {
        ok: method !== 'unsupported',
        method: method,
        accepted: episode.accepted,
        applied: episode.applied,
        notes: episode.notes.length ? episode.notes : []
      };
    },

    getState: function getState() {
      var now = Date.now();
      var player = getPlayer();
      var menuVisible = getMenuVisible('mainMenu');
      var gameOverVisible = getMenuVisible('gameOverMenu');
      var status = getStatus(player, gameOverVisible, menuVisible);

      if (!player && status === 'loading') {
        return buildLoadingState(now);
      }

      var platforms = getPlatforms();
      var spring = getSpring();
      var base = getBase();
      var visiblePlatforms = collectVisiblePlatforms(platforms);
      var terminal = getTerminal(status);
      var score = getScore();
      var playerState = buildPlayerState(player);
      var environment = buildEnvironment(visiblePlatforms, spring, base);
      var entities = serializeEntities(visiblePlatforms, spring);
      var gameState = {
        score: score,
        level: null,
        player: playerState,
        environment: environment,
        completion_progress: null
      };

      if (entities !== null) {
        gameState.entities = entities;
      }

      return {
        schemaVersion: '2.0',
        gameId: GAME_ID,
        seed: session.seed,
        timestampMs: now,
        gameTimeMs: getGameTimeMs(now, status),
        status: status,
        is_actionable: status === 'playing',
        terminal: terminal,
        game_state: gameState,
        metrics: {
          primary_score: score,
          distance: score,
          platforms_visible: finiteInt(visiblePlatforms.length),
          attempts: finiteInt(runtime.attempts)
        },
        debug: {
          menu_visible: menuVisible,
          game_over_visible: gameOverVisible,
          first_run: (typeof window !== 'undefined' && typeof window.firstRun === 'boolean') ? window.firstRun : null,
          current_seed: session.seed,
          last_reset_method: runtime.lastResetMethod
        }
      };
    }
  };
})();
