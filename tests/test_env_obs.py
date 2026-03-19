"""Tests for SnakebotEnv observation encoding.

Channel 1 (self):
  1.0  = body cell
  0.5  = tail cell (last body segment, vacates next turn unless apple eaten)
  -1.0 = head cell

Channel 2 (others):
  ally:  body=1.0, tail=0.75, head=0.5
  enemy: body=-1.0, tail=-0.75, head=-0.5

Grid centering:
  The grid is centered in the fixed (MAX_HEIGHT, MAX_WIDTH) observation array.
  row_off = (MAX_HEIGHT - grid.height) // 2
  col_off = (MAX_WIDTH  - grid.width)  // 2
  All coordinate assertions account for this offset.
"""
from __future__ import annotations

from collections import deque

from snakebot_env.core.grid import Grid
from snakebot_env.core.game import GameState
from snakebot_env.core.snakebot import Snakebot
from snakebot_env.env import SnakebotEnv, MAX_HEIGHT, MAX_WIDTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _offsets(grid: Grid) -> tuple[int, int]:
    """Return (row_off, col_off) applied by _get_obs for this grid."""
    return (MAX_HEIGHT - grid.height) // 2, (MAX_WIDTH - grid.width) // 2


def _make_env_with_two_players() -> tuple[SnakebotEnv, Snakebot, Snakebot, Grid]:
    """Minimal env: one p0 bot and one p1 bot at known positions."""
    grid = Grid(width=10, height=10)

    # p0 bot: head=(1,1), body=(1,2), tail=(1,3)
    p0_bot = Snakebot(id=0, owner=0)
    p0_bot.body = deque([(1, 1), (1, 2), (1, 3)])

    # p1 (enemy) bot: head=(5,5), body=(5,6), tail=(5,7)
    p1_bot = Snakebot(id=1, owner=1)
    p1_bot.body = deque([(5, 5), (5, 6), (5, 7)])

    game = GameState(grid=grid, snakebots=[p0_bot, p1_bot])
    env = SnakebotEnv()
    env._game = game
    env._bot_by_agent = {"p0_b0": p0_bot, "p1_b0": p1_bot}
    env.agents = ["p0_b0", "p1_b0"]
    return env, p0_bot, p1_bot, grid


def _make_env_with_ally() -> tuple[SnakebotEnv, Snakebot, Snakebot, Snakebot, Grid]:
    """Minimal env: two p0 bots and one p1 bot; p0_b1 observes p0_b0 as ally."""
    grid = Grid(width=10, height=10)

    # p0_b0: head=(1,1), body=(1,2), tail=(1,3)
    p0_bot0 = Snakebot(id=0, owner=0)
    p0_bot0.body = deque([(1, 1), (1, 2), (1, 3)])

    # p0_b1: head=(3,3), body=(3,4), tail=(3,5)
    p0_bot1 = Snakebot(id=1, owner=0)
    p0_bot1.body = deque([(3, 3), (3, 4), (3, 5)])

    # p1 bot: head=(8,8) only — length 1, so head == tail
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
    return env, p0_bot0, p0_bot1, p1_bot, grid


# ---------------------------------------------------------------------------
# Centering
# ---------------------------------------------------------------------------

def test_centering_offset_is_symmetric():
    """Grid content must be placed with equal padding on opposite sides."""
    grid = Grid(width=10, height=10)
    row_off, col_off = _offsets(grid)
    assert row_off == (MAX_HEIGHT - grid.height) // 2
    assert col_off == (MAX_WIDTH - grid.width) // 2


def test_wall_is_placed_at_centered_coordinates():
    grid = Grid(width=10, height=10)
    grid.walls.add((0, 0))
    bot = Snakebot(id=0, owner=0)
    bot.body = deque([(5, 5)])
    game = GameState(grid=grid, snakebots=[bot])
    env = SnakebotEnv()
    env._game = game
    env._bot_by_agent = {"p0_b0": bot}
    env.agents = ["p0_b0"]
    obs = env._get_obs("p0_b0")
    row_off, col_off = _offsets(grid)
    assert obs[0, row_off, col_off] == 1.0, (
        f"Wall at (0,0) should appear at obs[0,{row_off},{col_off}]"
    )
    # Top-left corner of obs must be empty (not the wall)
    assert obs[0, 0, 0] == 0.0, "Top-left obs cell should be zero-padded"


# ---------------------------------------------------------------------------
# Channel 1 — self encoding
# ---------------------------------------------------------------------------

def test_self_head_is_minus_one_in_channel1():
    env, p0_bot, _, grid = _make_env_with_two_players()
    obs = env._get_obs("p0_b0")
    row_off, col_off = _offsets(grid)
    # head=(1,1) → obs[1, 1+row_off, 1+col_off]
    assert obs[1, 1 + row_off, 1 + col_off] == -1.0


def test_self_body_mid_is_one_in_channel1():
    env, _, _, grid = _make_env_with_two_players()
    obs = env._get_obs("p0_b0")
    row_off, col_off = _offsets(grid)
    # body[1]=(1,2)
    assert obs[1, 2 + row_off, 1 + col_off] == 1.0


def test_self_tail_is_half_in_channel1():
    env, _, _, grid = _make_env_with_two_players()
    obs = env._get_obs("p0_b0")
    row_off, col_off = _offsets(grid)
    # body[2]=(1,3) — tail
    assert obs[1, 3 + row_off, 1 + col_off] == 0.5


def test_self_tail_differs_from_body_and_head_in_channel1():
    env, _, _, grid = _make_env_with_two_players()
    obs = env._get_obs("p0_b0")
    row_off, col_off = _offsets(grid)
    head_val = obs[1, 1 + row_off, 1 + col_off]
    body_val = obs[1, 2 + row_off, 1 + col_off]
    tail_val = obs[1, 3 + row_off, 1 + col_off]
    assert len({head_val, body_val, tail_val}) == 3, (
        f"head={head_val}, body={body_val}, tail={tail_val} must all differ"
    )


def test_self_length1_head_equals_tail_is_minus_one_in_channel1():
    """When the snake has length 1, head == tail; head value (-1.0) wins."""
    grid = Grid(width=10, height=10)
    bot = Snakebot(id=0, owner=0)
    bot.body = deque([(4, 4)])
    game = GameState(grid=grid, snakebots=[bot])
    env = SnakebotEnv()
    env._game = game
    env._bot_by_agent = {"p0_b0": bot}
    env.agents = ["p0_b0"]
    obs = env._get_obs("p0_b0")
    row_off, col_off = _offsets(grid)
    assert obs[1, 4 + row_off, 4 + col_off] == -1.0


# ---------------------------------------------------------------------------
# Channel 2 — enemy encoding (from p0_b0's perspective, p1_b0 is enemy)
# ---------------------------------------------------------------------------

def test_enemy_head_differs_from_enemy_body_in_channel2():
    env, _, _, grid = _make_env_with_two_players()
    obs = env._get_obs("p0_b0")
    row_off, col_off = _offsets(grid)
    head_val = obs[2, 5 + row_off, 5 + col_off]
    body_val = obs[2, 6 + row_off, 5 + col_off]
    assert head_val != body_val, (
        f"Enemy head ({head_val}) must differ from enemy body ({body_val})"
    )


def test_enemy_head_is_negative_in_channel2():
    env, _, _, grid = _make_env_with_two_players()
    obs = env._get_obs("p0_b0")
    row_off, col_off = _offsets(grid)
    assert obs[2, 5 + row_off, 5 + col_off] < 0


def test_enemy_head_value_is_minus_half_in_channel2():
    env, _, _, grid = _make_env_with_two_players()
    obs = env._get_obs("p0_b0")
    row_off, col_off = _offsets(grid)
    assert obs[2, 5 + row_off, 5 + col_off] == -0.5


def test_enemy_mid_body_is_negative_one_in_channel2():
    env, _, _, grid = _make_env_with_two_players()
    obs = env._get_obs("p0_b0")
    row_off, col_off = _offsets(grid)
    # body[1]=(5,6) — mid body, not tail
    assert obs[2, 6 + row_off, 5 + col_off] == -1.0


def test_enemy_tail_is_minus_three_quarters_in_channel2():
    env, _, _, grid = _make_env_with_two_players()
    obs = env._get_obs("p0_b0")
    row_off, col_off = _offsets(grid)
    # body[2]=(5,7) — tail
    assert obs[2, 7 + row_off, 5 + col_off] == -0.75


def test_enemy_tail_differs_from_body_and_head_in_channel2():
    env, _, _, grid = _make_env_with_two_players()
    obs = env._get_obs("p0_b0")
    row_off, col_off = _offsets(grid)
    head_val = obs[2, 5 + row_off, 5 + col_off]
    body_val = obs[2, 6 + row_off, 5 + col_off]
    tail_val = obs[2, 7 + row_off, 5 + col_off]
    assert len({head_val, body_val, tail_val}) == 3, (
        f"enemy head={head_val}, body={body_val}, tail={tail_val} must all differ"
    )


# ---------------------------------------------------------------------------
# Channel 2 — ally encoding (from p0_b1's perspective, p0_b0 is ally)
# ---------------------------------------------------------------------------

def test_ally_head_differs_from_ally_body_in_channel2():
    env, _, _, _, grid = _make_env_with_ally()
    obs = env._get_obs("p0_b1")
    row_off, col_off = _offsets(grid)
    head_val = obs[2, 1 + row_off, 1 + col_off]
    body_val = obs[2, 2 + row_off, 1 + col_off]
    assert head_val != body_val, (
        f"Ally head ({head_val}) must differ from ally body ({body_val})"
    )


def test_ally_head_is_positive_in_channel2():
    env, _, _, _, grid = _make_env_with_ally()
    obs = env._get_obs("p0_b1")
    row_off, col_off = _offsets(grid)
    assert obs[2, 1 + row_off, 1 + col_off] > 0


def test_ally_head_value_is_half_in_channel2():
    env, _, _, _, grid = _make_env_with_ally()
    obs = env._get_obs("p0_b1")
    row_off, col_off = _offsets(grid)
    assert obs[2, 1 + row_off, 1 + col_off] == 0.5


def test_ally_mid_body_is_positive_one_in_channel2():
    env, _, _, _, grid = _make_env_with_ally()
    obs = env._get_obs("p0_b1")
    row_off, col_off = _offsets(grid)
    # p0_b0 body[1]=(1,2) — mid body, not tail
    assert obs[2, 2 + row_off, 1 + col_off] == 1.0


def test_ally_tail_is_three_quarters_in_channel2():
    env, _, _, _, grid = _make_env_with_ally()
    obs = env._get_obs("p0_b1")
    row_off, col_off = _offsets(grid)
    # p0_b0 body[2]=(1,3) — tail
    assert obs[2, 3 + row_off, 1 + col_off] == 0.75


def test_ally_tail_differs_from_body_and_head_in_channel2():
    env, _, _, _, grid = _make_env_with_ally()
    obs = env._get_obs("p0_b1")
    row_off, col_off = _offsets(grid)
    head_val = obs[2, 1 + row_off, 1 + col_off]
    body_val = obs[2, 2 + row_off, 1 + col_off]
    tail_val = obs[2, 3 + row_off, 1 + col_off]
    assert len({head_val, body_val, tail_val}) == 3, (
        f"ally head={head_val}, body={body_val}, tail={tail_val} must all differ"
    )
