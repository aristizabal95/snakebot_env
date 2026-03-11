"""
Core game logic for the Snakebot environment.

Faithfully ports the Java Game.java turn sequence:
  1. do_moves()       – move each live bot in its direction
  2. do_eats()        – collect apples at head positions
  3. do_beheadings()  – resolve wall/body collisions
  4. do_falls()       – apply gravity (individual + inter-coiled groups)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from snakebot_env.core.grid import Grid, ADJACENCY_4, OPPOSITE
from snakebot_env.core.snakebot import Snakebot, Coord

MAX_TURNS = 200


@dataclass
class StepResult:
    """Per-bot results from one game step."""
    rewards: dict[int, float] = field(default_factory=dict)   # bot_id → reward
    terminated: dict[int, bool] = field(default_factory=dict)  # bot_id → dead this step


@dataclass
class GameState:
    grid: Grid
    snakebots: list[Snakebot]  # all bots from both players
    turn: int = 0

    # --- Helpers -------------------------------------------------------

    def live_bots(self) -> list[Snakebot]:
        return [b for b in self.snakebots if b.alive]

    def get_bot(self, bot_id: int) -> Optional[Snakebot]:
        for b in self.snakebots:
            if b.id == bot_id:
                return b
        return None

    def _solid_below(self, cx: int, cy: int, ignore: set[Coord]) -> bool:
        """Is there something solid directly below (cx, cy)?"""
        bx, by = cx, cy + 1
        if (bx, by) in ignore:
            return False
        if self.grid.is_wall(bx, by):
            return True
        for bot in self.live_bots():
            if (bx, by) in bot.body:
                return True
        if (bx, by) in self.grid.apples:
            return True
        return False

    def _touching(self, a: Snakebot, b: Snakebot) -> bool:
        """True if any cell of a is manhattan-1 from any cell of b."""
        for ax, ay in a.body:
            for bx, by in b.body:
                if abs(ax - bx) + abs(ay - by) == 1:
                    return True
        return False

    def _get_touch_groups(self) -> list[list[Snakebot]]:
        """Find groups of mutually-touching live bots (connected components)."""
        live = self.live_bots()
        visited: set[int] = set()
        groups: list[list[Snakebot]] = []
        for bot in live:
            if bot.id in visited:
                continue
            group: list[Snakebot] = []
            queue: deque[Snakebot] = deque([bot])
            visited.add(bot.id)
            while queue:
                cur = queue.popleft()
                group.append(cur)
                for other in live:
                    if other.id in visited:
                        continue
                    if self._touching(cur, other):
                        visited.add(other.id)
                        queue.append(other)
            if len(group) > 1:
                groups.append(group)
        return groups

    # --- Turn phases ---------------------------------------------------

    def _do_moves(self, actions: dict[int, Coord]) -> None:
        """Move each live bot one step. actions: {bot_id: direction}."""
        for bot in self.live_bots():
            # Apply direction command (reversal rejected inside set_direction)
            if bot.id in actions:
                bot.set_direction(actions[bot.id])

            new_head = (bot.head[0] + bot.direction[0],
                        bot.head[1] + bot.direction[1])

            will_eat = new_head in self.grid.apples
            if not will_eat:
                bot.body.pop()  # remove tail
            bot.body.appendleft(new_head)

    def _do_eats(self) -> set[int]:
        """Remove apples eaten this step. Returns set of bot_ids that ate."""
        eaten_apples: set[Coord] = set()
        eaters: set[int] = set()
        for bot in self.live_bots():
            if bot.head in self.grid.apples:
                eaten_apples.add(bot.head)
                eaters.add(bot.id)
        self.grid.apples -= eaten_apples
        return eaters

    def _do_beheadings(self) -> dict[int, float]:
        """
        Resolve head-into-wall or head-into-body collisions.
        Returns {bot_id: penalty} for any bot that was beheaded or killed.
        """
        all_bodies: dict[int, set[Coord]] = {
            b.id: set(b.body) for b in self.live_bots()
        }
        to_behead: list[Snakebot] = []

        for bot in self.live_bots():
            hx, hy = bot.head
            in_wall = self.grid.is_wall(hx, hy)

            in_body = False
            for other in self.live_bots():
                if other.id == bot.id:
                    # Own body: head into non-head own cell
                    if bot.head in set(list(bot.body)[1:]):
                        in_body = True
                        break
                else:
                    if bot.head in all_bodies[other.id]:
                        in_body = True
                        break

            if in_wall or in_body:
                to_behead.append(bot)

        penalties: dict[int, float] = {}
        for bot in to_behead:
            length = len(bot.body)
            if length <= 3:
                # Kill entirely
                penalties[bot.id] = -float(length)
                bot.alive = False
            else:
                # Behead: remove head only
                bot.body.popleft()
                penalties[bot.id] = -1.0

        return penalties

    def _do_falls(self) -> None:
        """Apply gravity repeatedly until nothing moves."""
        something_fell = True
        while something_fell:
            something_fell = False

            # Individual bot falls
            live = self.live_bots()
            for bot in live:
                ignore = set(bot.body)
                can_fall = all(not self._solid_below(cx, cy, ignore)
                               for cx, cy in bot.body)
                if can_fall:
                    bot.body = deque((x, y + 1) for x, y in bot.body)
                    something_fell = True
                    # Kill if entirely out of bounds
                    if all(y >= self.grid.height for _, y in bot.body):
                        bot.alive = False

            # Inter-coiled group falls
            something_fell |= self._do_intercoiled_falls()

    def _do_intercoiled_falls(self) -> bool:
        """
        Groups of touching live bots fall together as a unit if no part of
        the combined body has solid below it. Returns True if anything moved.
        """
        fell_any = True
        moved = False
        while fell_any:
            fell_any = False
            for group in self._get_touch_groups():
                meta_body: set[Coord] = set()
                for bot in group:
                    meta_body |= set(bot.body)

                can_fall = all(not self._solid_below(cx, cy, meta_body)
                               for cx, cy in meta_body)
                if can_fall:
                    for bot in group:
                        bot.body = deque((x, y + 1) for x, y in bot.body)
                        if bot.head[1] >= self.grid.height:
                            bot.alive = False
                    fell_any = True
                    moved = True
        return moved

    # --- Public API ----------------------------------------------------

    def step(self, actions: dict[int, Coord]) -> StepResult:
        """
        Run one full game turn.

        actions: {bot_id: direction_tuple}
        Returns StepResult with per-bot rewards and termination flags.
        """
        self.turn += 1
        result = StepResult()

        # Track which bots are alive at the start (to detect falls→death later)
        alive_before = {b.id for b in self.snakebots if b.alive}

        self._do_moves(actions)
        eaters = self._do_eats()
        penalties = self._do_beheadings()
        self._do_falls()

        # Collect rewards
        for bot in self.snakebots:
            reward = 0.0
            # Apple eating reward
            if bot.id in eaters:
                reward += 1.0
            # Collision / death penalty
            if bot.id in penalties:
                reward += penalties[bot.id]
            # Fell out of bounds (alive before, dead now, not beheaded)
            if bot.id in alive_before and not bot.alive and bot.id not in penalties:
                reward -= float(len(bot.body))  # whole body lost
            result.rewards[bot.id] = reward
            result.terminated[bot.id] = bot.id in alive_before and not bot.alive

        return result

    def is_game_over(self) -> bool:
        no_apples = len(self.grid.apples) == 0
        p0_dead = all(not b.alive for b in self.snakebots if b.owner == 0)
        p1_dead = all(not b.alive for b in self.snakebots if b.owner == 1)
        return no_apples or p0_dead or p1_dead or self.turn >= MAX_TURNS

    def scores(self) -> dict[int, int]:
        """Total body size of all live bots per player."""
        result: dict[int, int] = {0: 0, 1: 0}
        for bot in self.snakebots:
            if bot.alive:
                result[bot.owner] += len(bot.body)
        return result
