"""Tests for SnakebotEnv observation encoding.

Verifies that channel 2 (others) correctly distinguishes:
  - ally body  = +1.0
  - ally head  = +0.5  (distinct from ally body)
  - enemy body = -1.0
  - enemy head = -0.5  (distinct from enemy body)
"""
from __future__ import annotations

from collections import deque

import numpy as np

from snakebot_env.core.grid import Grid
from snakebot_env.core.game import GameState
from snakebot_env.core.snakebot import Snakebot
from snakebot_env.env import SnakebotEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env_with_two_players() -> tuple[SnakebotEnv, Snakebot, Snakebot]:
    """Minimal env: one p0 bot and one p1 bot at known positions."""
    grid = Grid(width=10, height=10)

    # p0 bot: head=(1,1), tail toward y+
    p0_bot = Snakebot(id=0, owner=0)
    p0_bot.body = deque([(1, 1), (1, 2), (1, 3)])

    # p1 (enemy) bot: head=(5,5), tail toward y+
    p1_bot = Snakebot(id=1, owner=1)
    p1_bot.body = deque([(5, 5), (5, 6), (5, 7)])

    game = GameState(grid=grid, snakebots=[p0_bot, p1_bot])
    env = SnakebotEnv()
    env._game = game
    env._bot_by_agent = {"p0_b0": p0_bot, "p1_b0": p1_bot}
    env.agents = ["p0_b0", "p1_b0"]
    return env, p0_bot, p1_bot


def _make_env_with_ally() -> tuple[SnakebotEnv, Snakebot, Snakebot, Snakebot]:
    """Minimal env: two p0 bots and one p1 bot; p0_b1 observes p0_b0 as ally."""
    grid = Grid(width=10, height=10)

    p0_bot0 = Snakebot(id=0, owner=0)
    p0_bot0.body = deque([(1, 1), (1, 2), (1, 3)])

    p0_bot1 = Snakebot(id=1, owner=0)
    p0_bot1.body = deque([(3, 3), (3, 4), (3, 5)])

    p1_bot = Snakebot(id=2, owner=1)
    p1_bot.body = deque([(8, 8)])

    game = GameState(grid=grid, snakebots=[p0_bot0, p0_bot1, p1_bot])
    env = SnakebotEnv()
    env._game = game
    env._bot_by_agent = {
        "p0_b0": p0_bot0,
        "p0_b1": p0_bot1,
        "p1_b0": p1_bot,
    }
    env.agents = ["p0_b0", "p0_b1", "p1_b0"]
    return env, p0_bot0, p0_bot1, p1_bot


# ---------------------------------------------------------------------------
# Enemy encoding (from p0_b0's perspective, p1_b0 is the enemy)
# ---------------------------------------------------------------------------

def test_enemy_head_differs_from_enemy_body_in_channel2():
    """Enemy head must have a distinct value from enemy body in channel 2."""
    env, _, p1_bot = _make_env_with_two_players()
    obs = env._get_obs("p0_b0")

    # p1 head=(5,5) → obs[2, 5, 5]; p1 body[1]=(5,6) → obs[2, 6, 5]
    head_val = obs[2, 5, 5]
    body_val = obs[2, 6, 5]

    assert head_val != body_val, (
        f"Enemy head value ({head_val}) must differ from enemy body value ({body_val})"
    )


def test_enemy_head_is_negative_in_channel2():
    """Enemy head must be negative (indicating it belongs to the enemy team)."""
    env, _, _ = _make_env_with_two_players()
    obs = env._get_obs("p0_b0")

    head_val = obs[2, 5, 5]
    assert head_val < 0, f"Enemy head should be negative, got {head_val}"


def test_enemy_body_is_negative_one_in_channel2():
    """Enemy body cells (non-head) must remain -1.0."""
    env, _, _ = _make_env_with_two_players()
    obs = env._get_obs("p0_b0")

    # body[1] and body[2] of p1 are non-head cells
    assert obs[2, 6, 5] == -1.0, f"Expected -1.0, got {obs[2, 6, 5]}"
    assert obs[2, 7, 5] == -1.0, f"Expected -1.0, got {obs[2, 7, 5]}"


def test_enemy_head_value_is_minus_half_in_channel2():
    """Enemy head must be encoded as -0.5 in channel 2."""
    env, _, _ = _make_env_with_two_players()
    obs = env._get_obs("p0_b0")

    assert obs[2, 5, 5] == -0.5, f"Expected -0.5, got {obs[2, 5, 5]}"


# ---------------------------------------------------------------------------
# Ally encoding (from p0_b1's perspective, p0_b0 is the ally)
# ---------------------------------------------------------------------------

def test_ally_head_differs_from_ally_body_in_channel2():
    """Ally head must have a distinct value from ally body in channel 2."""
    env, p0_bot0, _, _ = _make_env_with_ally()
    obs = env._get_obs("p0_b1")

    # p0_b0 head=(1,1) → obs[2, 1, 1]; body[1]=(1,2) → obs[2, 2, 1]
    head_val = obs[2, 1, 1]
    body_val = obs[2, 2, 1]

    assert head_val != body_val, (
        f"Ally head value ({head_val}) must differ from ally body value ({body_val})"
    )


def test_ally_head_is_positive_in_channel2():
    """Ally head must be positive (indicating it belongs to the ally team)."""
    env, _, _, _ = _make_env_with_ally()
    obs = env._get_obs("p0_b1")

    head_val = obs[2, 1, 1]
    assert head_val > 0, f"Ally head should be positive, got {head_val}"


def test_ally_body_is_positive_one_in_channel2():
    """Ally body cells (non-head) must remain +1.0."""
    env, _, _, _ = _make_env_with_ally()
    obs = env._get_obs("p0_b1")

    assert obs[2, 2, 1] == 1.0, f"Expected 1.0, got {obs[2, 2, 1]}"
    assert obs[2, 3, 1] == 1.0, f"Expected 1.0, got {obs[2, 3, 1]}"


def test_ally_head_value_is_half_in_channel2():
    """Ally head must be encoded as +0.5 in channel 2."""
    env, _, _, _ = _make_env_with_ally()
    obs = env._get_obs("p0_b1")

    assert obs[2, 1, 1] == 0.5, f"Expected 0.5, got {obs[2, 1, 1]}"
