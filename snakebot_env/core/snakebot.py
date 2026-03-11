from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from snakebot_env.core.grid import (
    UP, DOWN, LEFT, RIGHT,
    OPPOSITE,
)

Coord = tuple[int, int]


@dataclass
class Snakebot:
    id: int
    owner: int  # player index (0 or 1)
    body: deque[Coord] = field(default_factory=deque)
    direction: Coord = UP  # current intended direction
    alive: bool = True

    @property
    def head(self) -> Coord:
        return self.body[0]

    @property
    def facing(self) -> Coord:
        """Direction from body[1] → body[0], i.e., the movement direction."""
        if len(self.body) < 2:
            return UP
        hx, hy = self.body[0]
        nx, ny = self.body[1]
        return (hx - nx, hy - ny)

    def set_direction(self, new_dir: Coord) -> None:
        """Set direction, rejecting reversal."""
        if new_dir != OPPOSITE.get(self.facing):
            self.direction = new_dir

    def agent_id(self, bot_idx: int) -> str:
        return f"p{self.owner}_b{bot_idx}"
