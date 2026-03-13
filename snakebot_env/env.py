"""
PettingZoo ParallelEnv implementation for the Snakebot game.

Each snakebot is a separate agent. Agent IDs follow the format "p{player}_b{bot}".

Observation per agent: numpy array of shape (3, height, width), dtype float32
  Channel 0 (map):    0.0=empty, 1.0=wall, -1.0=apple
  Channel 1 (self):   1.0=own body cell, 0.0=elsewhere
  Channel 2 (others): 1.0=ally body, -1.0=enemy body, 0.0=elsewhere

Action per agent: Discrete(4) → 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

Reward:
  +1.0   when this agent eats an apple
  -1.0   when beheaded (loses 1 body part)
  -N     when killed outright (loses N body parts)
  0.0    otherwise
"""
from __future__ import annotations

import random as stdlib_random
from collections import deque
from typing import Any, Optional

import numpy as np
import gymnasium
from gymnasium import spaces
from pettingzoo import ParallelEnv

from snakebot_env.core.grid import UP, DOWN, LEFT, RIGHT
from snakebot_env.core.game import GameState
from snakebot_env.core.snakebot import Snakebot
from snakebot_env.generation.grid_maker import GridMaker

# Action index → direction tuple
ACTION_TO_DIR = {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}

MAX_WIDTH = 45   # from constraints: width ≤ 45
MAX_HEIGHT = 25  # height ≤ 24, +1 buffer


class SnakebotEnv(ParallelEnv):
    """Cooperative-competitive multi-agent snakebot environment."""

    metadata = {
        "name": "snakebot_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_players: int = 2,
        bots_per_player: int = 2,
        league_level: int = 4,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        apple_density: Optional[float] = None,
    ):
        super().__init__()
        assert num_players == 2, "Currently only 2-player mode is supported."
        self.num_players = num_players
        self.bots_per_player = bots_per_player
        self.league_level = league_level
        self.render_mode = render_mode
        self._seed = seed
        self._apple_density = apple_density

        # Stable list of all possible agents across an episode
        self.possible_agents = [
            f"p{p}_b{b}"
            for p in range(num_players)
            for b in range(bots_per_player)
        ]

        # Set on reset()
        self.agents: list[str] = []
        self._game: Optional[GameState] = None
        self._bot_by_agent: dict[str, Snakebot] = {}
        self._renderer = None

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(
            low=-1.0, high=1.0,
            shape=(3, MAX_HEIGHT, MAX_WIDTH),
            dtype=np.float32,
        )

    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Discrete(4)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        rng = stdlib_random.Random(seed if seed is not None else self._seed)

        # Generate grid
        gm_kwargs = {"rng": rng, "league_level": self.league_level}
        if self._apple_density is not None:
            gm_kwargs["apple_density"] = self._apple_density
        grid = GridMaker(**gm_kwargs).make()

        # Assign spawn locations to bots
        spawn_islands = grid.detect_spawn_islands()

        # Create snakebots
        bots: list[Snakebot] = []
        bot_id = 0
        # Distribute spawn islands across bots_per_player (per player)
        for player in range(self.num_players):
            bot_count = 0
            for island in spawn_islands:
                if bot_count >= self.bots_per_player:
                    break
                bot = Snakebot(id=bot_id, owner=player)
                for coord in island:
                    # Player 1 spawns on the mirrored side
                    c = grid.opposite(*coord) if player == 1 else coord
                    bot.body.append(c)
                bots.append(bot)
                bot_id += 1
                bot_count += 1

        self._game = GameState(grid=grid, snakebots=bots, turn=0)

        # Build agent → bot mapping
        self._bot_by_agent = {}
        bot_idx = 0
        for agent in self.possible_agents:
            p, b = _parse_agent_id(agent)
            player_bots = [bot for bot in bots if bot.owner == p]
            if b < len(player_bots):
                self._bot_by_agent[agent] = player_bots[b]

        # Active agents: only those with a valid, alive bot
        self.agents = [
            a for a in self.possible_agents
            if a in self._bot_by_agent and self._bot_by_agent[a].alive
        ]

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}

        self._init_renderer()
        return observations, infos

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        assert self._game is not None, "Call reset() before step()."

        # Translate agent actions to bot directions
        bot_actions: dict[int, tuple[int, int]] = {}
        for agent, action_idx in actions.items():
            if agent in self._bot_by_agent:
                bot = self._bot_by_agent[agent]
                if bot.alive:
                    bot_actions[bot.id] = ACTION_TO_DIR[int(action_idx)]

        try:
            step_result = self._game.step(bot_actions)
        except IndexError:
            # A snake body grew outside the array bounds (can happen during
            # training when apple density is high). Treat as an immediate
            # endgame: ignore the step and signal termination for all agents.
            rewards = {a: 0.0 for a in self.agents}
            terminated = {a: True for a in self.agents}
            truncated = {a: False for a in self.agents}
            infos = {a: {} for a in self.agents}
            self.agents = []
            return {}, rewards, terminated, truncated, infos

        # Compute per-apple reward for eaters (+1 per apple eaten)
        # We track apples before step to know who ate what
        # (step_result doesn't report eaters; we use reward delta instead)
        # Reward = penalty (from beheading/death) + apple reward (0 here —
        # apple collection is implicit via body growth, penalty captures loss)
        rewards: dict[str, float] = {}
        terminated: dict[str, bool] = {}
        truncated: dict[str, bool] = {}

        for agent in self.agents:
            bot = self._bot_by_agent[agent]
            r = step_result.rewards.get(bot.id, 0.0)
            rewards[agent] = r
            terminated[agent] = step_result.terminated.get(bot.id, False)
            truncated[agent] = self._game.is_game_over() and not terminated[agent]

        # Remove dead agents
        self.agents = [
            a for a in self.agents
            if not terminated.get(a, False) and not truncated.get(a, False)
        ]

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {"turn": self._game.turn} for a in self.agents}

        return observations, rewards, terminated, truncated, infos

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self, agent: str) -> np.ndarray:
        obs = np.zeros((3, MAX_HEIGHT, MAX_WIDTH), dtype=np.float32)
        grid = self._game.grid

        # Channel 0: map
        for wx, wy in grid.walls:
            if 0 <= wy < MAX_HEIGHT and 0 <= wx < MAX_WIDTH:
                obs[0, wy, wx] = 1.0
        for ax, ay in grid.apples:
            if 0 <= ay < MAX_HEIGHT and 0 <= ax < MAX_WIDTH:
                obs[0, ay, ax] = -1.0

        bot = self._bot_by_agent.get(agent)
        if bot is None:
            return obs

        my_player = bot.owner
        my_bot_id = bot.id

        # Channel 1: self body
        for bx, by in bot.body:
            if 0 <= by < MAX_HEIGHT and 0 <= bx < MAX_WIDTH:
                obs[1, by, bx] = 1.0

        # Channel 2: others (ally=+1, enemy=-1)
        for other_bot in self._game.snakebots:
            if other_bot.id == my_bot_id or not other_bot.alive:
                continue
            value = 1.0 if other_bot.owner == my_player else -1.0
            for bx, by in other_bot.body:
                if 0 <= by < MAX_HEIGHT and 0 <= bx < MAX_WIDTH:
                    obs[2, by, bx] = value

        return obs

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _init_renderer(self) -> None:
        """Create or reinitialize the renderer after reset()."""
        if self.render_mode is None:
            return
        from snakebot_env.renderer import Renderer
        if self._renderer is None:
            self._renderer = Renderer(self._game, mode=self.render_mode)
        else:
            self._renderer.game = self._game

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._init_renderer()
        return self._renderer.render(self.render_mode)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _parse_agent_id(agent: str) -> tuple[int, int]:
    """Parse 'p{player}_b{bot}' → (player_idx, bot_idx)."""
    parts = agent.split("_")
    p = int(parts[0][1:])
    b = int(parts[1][1:])
    return p, b
