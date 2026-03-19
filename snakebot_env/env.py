"""
PettingZoo ParallelEnv implementation for the Snakebot game.

Each snakebot is a separate agent. Agent IDs follow the format "p{player}_b{bot}".

Observation per agent: numpy array of shape (3, height, width), dtype float32
  Channel 0 (map):    0.0=empty, 1.0=wall, -1.0=apple
  Channel 1 (self):   1.0=own body cell, 0.5=own tail cell, -1.0=own head cell, 0.0=elsewhere
  Channel 2 (others): 1.0=ally body, 0.75=ally tail, 0.5=ally head,
                      -1.0=enemy body, -0.75=enemy tail, -0.5=enemy head, 0.0=elsewhere

  The tail cell (last body segment) is encoded with a distinct value because it
  will vacate next turn (unless an apple is eaten), making it safe to move into.
  Direction of travel is inferrable from the head (-1.0) relative to the
  adjacent body cell (1.0) behind it.

Action per agent: Discrete(4) → 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

Reward:
  +1.0   when this agent eats an apple
  -1.0   when beheaded (loses 1 body part)
  -N     when killed outright (loses N body parts)
  0.0    otherwise

  Optional cooperative/competitive reward shaping (all default to 0.0):
  reward_win        ±bonus to surviving agents of the winning/losing team at game end
  reward_kill_credit  per-kill bonus distributed to alive allies when an enemy bot dies
  reward_team_share   each bot receives this fraction of its teammates' net rewards
"""
from __future__ import annotations

import random as stdlib_random
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
NUM_OBS_CHANNELS = 3


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
        max_steps: Optional[int] = None,
        reward_win: float = 0.0,
        reward_kill_credit: float = 0.0,
        reward_team_share: float = 0.0,
    ):
        """
        Args:
            num_players: Number of players (currently only 2 supported).
            bots_per_player: Number of snakebots per player.
            league_level: Difficulty level (1=bronze … 4=legend).
            render_mode: Optional render mode ("human" or "rgb_array").
            seed: Optional RNG seed for reproducibility.
            apple_density: Override default apple spawn density.
            max_steps: Override default maximum turns per episode.
            reward_win: Terminal bonus added to surviving agents of the winning
                team (and subtracted from the losing team) when the game ends.
                0.0 disables this signal.
            reward_kill_credit: Bonus given to each alive ally when an enemy
                bot is killed this step.  0.0 disables.
            reward_team_share: Fraction of teammates' net step rewards added
                to each bot's reward.  0.0 disables; 1.0 = full sharing.
        """
        super().__init__()
        assert num_players == 2, "Currently only 2-player mode is supported."
        self.num_players = num_players
        self.bots_per_player = bots_per_player
        self.league_level = league_level
        self.render_mode = render_mode
        self._seed = seed
        self._apple_density = apple_density
        self._max_steps = max_steps
        self.reward_win = reward_win
        self.reward_kill_credit = reward_kill_credit
        self.reward_team_share = reward_team_share

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
            shape=(NUM_OBS_CHANNELS, MAX_HEIGHT, MAX_WIDTH),
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

        gs_kwargs: dict = {"grid": grid, "snakebots": bots, "turn": 0}
        if self._max_steps is not None:
            gs_kwargs["max_turns"] = self._max_steps
        self._game = GameState(**gs_kwargs)

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

        # Compute per-agent base rewards from the game step
        rewards: dict[str, float] = {}
        terminated: dict[str, bool] = {}
        truncated: dict[str, bool] = {}

        game_over = self._game.is_game_over()

        for agent in self.agents:
            bot = self._bot_by_agent[agent]
            r = step_result.rewards.get(bot.id, 0.0)
            rewards[agent] = r
            terminated[agent] = step_result.terminated.get(bot.id, False)
            truncated[agent] = game_over and not terminated[agent]

        # --- Optional reward shaping ---

        if self.reward_kill_credit != 0.0:
            # Count how many bots died per player this step
            kills: dict[int, int] = {0: 0, 1: 0}
            for agent in terminated:
                if terminated[agent]:
                    p = self._bot_by_agent[agent].owner
                    kills[p] += 1
            # Credit surviving agents of the opposing team
            for agent in rewards:
                if not terminated.get(agent, False):
                    p = self._bot_by_agent[agent].owner
                    enemy_kills = kills.get(1 - p, 0)
                    if enemy_kills > 0:
                        rewards[agent] += self.reward_kill_credit * enemy_kills

        if self.reward_team_share != 0.0:
            # Sum each player's total reward across all their bots this step
            team_net: dict[int, float] = {0: 0.0, 1: 0.0}
            for agent in rewards:
                p = self._bot_by_agent[agent].owner
                team_net[p] += rewards[agent]
            # Each bot gains a fraction of its teammates' combined reward
            for agent in rewards:
                p = self._bot_by_agent[agent].owner
                others_net = team_net[p] - rewards[agent]
                if others_net != 0.0:
                    rewards[agent] += self.reward_team_share * others_net

        if game_over and self.reward_win != 0.0:
            scores = self._game.scores()
            s0, s1 = scores[0], scores[1]
            if s0 != s1:
                winning_player = 0 if s0 > s1 else 1
                for agent in rewards:
                    if not terminated.get(agent, False):
                        p = self._bot_by_agent[agent].owner
                        if p == winning_player:
                            rewards[agent] += self.reward_win
                        else:
                            rewards[agent] -= self.reward_win

        # Remove dead agents
        self.agents = [
            a for a in self.agents
            if not terminated.get(a, False) and not truncated.get(a, False)
        ]

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {
            a: {"turn": self._game.turn, "max_turns": self._game.max_turns}
            for a in self.agents
        }

        return observations, rewards, terminated, truncated, infos

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self, agent: str) -> np.ndarray:
        obs = np.zeros((NUM_OBS_CHANNELS, MAX_HEIGHT, MAX_WIDTH), dtype=np.float32)
        grid = self._game.grid

        bot = self._bot_by_agent.get(agent)
        if bot is None:
            return obs

        # Egocentric: center the observation window on the agent's head so that
        # the head always appears at (MAX_HEIGHT // 2, MAX_WIDTH // 2).  All
        # other entities (walls, apples, other snakes) are placed at positions
        # relative to the head; cells outside the buffer bounds are clipped.
        hx, hy = bot.head
        row_off = MAX_HEIGHT // 2 - hy
        col_off = MAX_WIDTH // 2 - hx

        def _row(y: int) -> int:
            return y + row_off

        def _col(x: int) -> int:
            return x + col_off

        def _in_bounds(x: int, y: int) -> bool:
            r, c = _row(y), _col(x)
            return 0 <= r < MAX_HEIGHT and 0 <= c < MAX_WIDTH

        # Channel 0: map
        for wx, wy in grid.walls:
            if _in_bounds(wx, wy):
                obs[0, _row(wy), _col(wx)] = 1.0
        for ax, ay in grid.apples:
            if _in_bounds(ax, ay):
                obs[0, _row(ay), _col(ax)] = -1.0

        my_player = bot.owner
        my_bot_id = bot.id
        tail_idx = len(bot.body) - 1

        # Channel 1: self body
        #   1.0 = body, 0.5 = tail (vacates next turn unless apple eaten), -1.0 = head
        for i, (bx, by) in enumerate(bot.body):
            if _in_bounds(bx, by):
                obs[1, _row(by), _col(bx)] = 0.5 if i == tail_idx else 1.0
        hx, hy = bot.head
        if _in_bounds(hx, hy):
            obs[1, _row(hy), _col(hx)] = -1.0

        # Channel 2: others
        #   ally:  body=1.0, tail=0.75, head=0.5
        #   enemy: body=-1.0, tail=-0.75, head=-0.5
        for other_bot in self._game.snakebots:
            if other_bot.id == my_bot_id or not other_bot.alive:
                continue
            is_ally = other_bot.owner == my_player
            body_val = 1.0 if is_ally else -1.0
            tail_val = 0.75 if is_ally else -0.75
            head_val = 0.5 if is_ally else -0.5
            other_tail_idx = len(other_bot.body) - 1
            for i, (bx, by) in enumerate(other_bot.body):
                if _in_bounds(bx, by):
                    obs[2, _row(by), _col(bx)] = tail_val if i == other_tail_idx else body_val
            hx, hy = other_bot.head
            if _in_bounds(hx, hy):
                obs[2, _row(hy), _col(hx)] = head_val

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
