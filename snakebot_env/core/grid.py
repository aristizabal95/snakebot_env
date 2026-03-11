from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# Direction constants: (dx, dy)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

DIRECTIONS = (UP, DOWN, LEFT, RIGHT)
ADJACENCY_4 = DIRECTIONS
ADJACENCY_8 = (
    UP, DOWN, LEFT, RIGHT,
    (-1, -1), (1, 1), (1, -1), (-1, 1),
)

OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


@dataclass
class Grid:
    width: int
    height: int
    walls: set[tuple[int, int]] = field(default_factory=set)
    apples: set[tuple[int, int]] = field(default_factory=set)
    spawns: list[tuple[int, int]] = field(default_factory=list)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, x: int, y: int) -> bool:
        return (x, y) in self.walls

    def is_empty(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and (x, y) not in self.walls

    def neighbours(
        self, x: int, y: int, adjacency: tuple = ADJACENCY_4
    ) -> list[tuple[int, int]]:
        result = []
        for dx, dy in adjacency:
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                result.append((nx, ny))
        return result

    def opposite(self, x: int, y: int) -> tuple[int, int]:
        """Mirror x-axis (left/right symmetry). Used by grid generation."""
        return (self.width - x - 1, y)

    def flood_fill_empty(
        self, start: tuple[int, int]
    ) -> Optional[set[tuple[int, int]]]:
        """BFS flood-fill through non-wall cells from start."""
        sx, sy = start
        if not self.in_bounds(sx, sy) or self.is_wall(sx, sy):
            return None
        visited: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque([start])
        visited.add(start)
        while queue:
            cx, cy = queue.popleft()
            for nx, ny in self.neighbours(cx, cy):
                if (nx, ny) not in visited and not self.is_wall(nx, ny):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return visited

    def detect_air_pockets(self) -> list[set[tuple[int, int]]]:
        """Find all disconnected empty regions."""
        islands: list[set[tuple[int, int]]] = []
        computed: set[tuple[int, int]] = set()
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in computed or self.is_wall(x, y):
                    computed.add((x, y))
                    continue
                island: set[tuple[int, int]] = set()
                queue: deque[tuple[int, int]] = deque([(x, y)])
                computed.add((x, y))
                while queue:
                    cx, cy = queue.popleft()
                    island.add((cx, cy))
                    for nx, ny in self.neighbours(cx, cy):
                        if (nx, ny) not in computed and not self.is_wall(nx, ny):
                            computed.add((nx, ny))
                            queue.append((nx, ny))
                islands.append(island)
        return islands

    def detect_spawn_islands(self) -> list[list[tuple[int, int]]]:
        """Group adjacent spawn cells into islands, return sorted lists."""
        spawn_set = set(self.spawns)
        islands: list[list[tuple[int, int]]] = []
        computed: set[tuple[int, int]] = set()
        for sp in self.spawns:
            if sp in computed:
                continue
            island: set[tuple[int, int]] = set()
            queue: deque[tuple[int, int]] = deque([sp])
            computed.add(sp)
            while queue:
                cx, cy = queue.popleft()
                island.add((cx, cy))
                for nx, ny in self.neighbours(cx, cy):
                    if (nx, ny) not in computed and (nx, ny) in spawn_set:
                        computed.add((nx, ny))
                        queue.append((nx, ny))
            islands.append(sorted(island))
        return islands

    def detect_lowest_island(self) -> list[tuple[int, int]]:
        """Flood-fill walls from bottom-left corner."""
        start = (0, self.height - 1)
        if not self.is_wall(*start):
            return []
        visited: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque([start])
        visited.add(start)
        while queue:
            cx, cy = queue.popleft()
            for nx, ny in self.neighbours(cx, cy):
                if (nx, ny) not in visited and self.is_wall(nx, ny):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return list(visited)
