"""
Procedural level generator ported from GridMaker.java.

Generates a symmetric, gravity-aware grid with walls, apples, and spawn locations.
"""
from __future__ import annotations

import random as stdlib_random
from collections import deque
from typing import Optional

from snakebot_env.core.grid import Grid, ADJACENCY_8

MIN_HEIGHT = 10
MAX_HEIGHT = 24
ASPECT_RATIO = 1.8
SPAWN_HEIGHT = 3
DESIRED_SPAWNS = 4


class GridMaker:
    def __init__(self, rng: Optional[stdlib_random.Random] = None, league_level: int = 4):
        """
        league_level: 1=bronze(easy), 2=silver, 3=gold, 4=legend(hard)
        Higher league → smaller grids (more wall density near top).
        """
        self.rng = rng or stdlib_random.Random()
        self.league_level = league_level
        self.grid: Grid = None  # type: ignore

    def make(self) -> Grid:
        # --- Grid size ------------------------------------------------
        if self.league_level == 1:
            skew = 2.0
        elif self.league_level == 2:
            skew = 1.0
        elif self.league_level == 3:
            skew = 0.8
        else:
            skew = 0.3

        rand = self.rng.random()
        height = MIN_HEIGHT + round((rand ** skew) * (MAX_HEIGHT - MIN_HEIGHT))
        width = round(height * ASPECT_RATIO)
        if width % 2 != 0:
            width += 1

        self.grid = Grid(width, height)

        # --- Base wall generation -------------------------------------
        b = 5 + self.rng.random() * 10

        # Bottom row: all walls
        for x in range(width):
            self.grid.walls.add((x, height - 1))

        # Rows from bottom to top with decreasing wall probability
        for y in range(height - 2, -1, -1):
            y_norm = (height - 1 - y) / (height - 1)
            block_chance = 1 / (y_norm + 0.1) / b
            for x in range(width):
                if self.rng.random() < block_chance:
                    self.grid.walls.add((x, y))

        # --- X-axis symmetry (left → right mirror) --------------------
        mirrored_walls: set[tuple[int, int]] = set()
        for x, y in list(self.grid.walls):
            ox, oy = self.grid.opposite(x, y)
            mirrored_walls.add((ox, oy))
        self.grid.walls |= mirrored_walls

        # --- Fill small air pockets (< 10 cells) with walls -----------
        for island in self.grid.detect_air_pockets():
            if len(island) < 10:
                self.grid.walls |= island
                for x, y in island:
                    self.grid.walls.add(self.grid.opposite(x, y))

        # --- Open heavily-enclosed cells ------------------------------
        changed = True
        while changed:
            changed = False
            for x in range(width):
                for y in range(height):
                    if self.grid.is_wall(x, y):
                        continue
                    wall_neighs = [
                        (nx, ny) for nx, ny in self.grid.neighbours(x, y)
                        if self.grid.is_wall(nx, ny)
                    ]
                    if len(wall_neighs) >= 3:
                        # Destroy one wall neighbour at or above current cell
                        destroyable = [(nx, ny) for nx, ny in wall_neighs if ny <= y]
                        if destroyable:
                            self.rng.shuffle(destroyable)
                            dx, dy = destroyable[0]
                            self.grid.walls.discard((dx, dy))
                            self.grid.walls.discard(self.grid.opposite(dx, dy))
                            changed = True

        # --- Sink the lowest wall island downward ---------------------
        island = self.grid.detect_lowest_island()
        lower_by = 0
        can_lower = True
        while can_lower:
            for x in range(width):
                candidate = (x, height - 1 - (lower_by + 1))
                if candidate not in island:
                    can_lower = False
                    break
            if can_lower:
                lower_by += 1

        if lower_by >= 2:
            lower_by = self.rng.randint(2, lower_by)

        # Remove old island, place sunk version
        for x, y in island:
            self.grid.walls.discard((x, y))
            self.grid.walls.discard(self.grid.opposite(x, y))
        for x, y in island:
            new_y = y + lower_by
            if self.grid.in_bounds(x, new_y):
                self.grid.walls.add((x, new_y))
                self.grid.walls.add(self.grid.opposite(x, new_y))

        # --- Spawn apples randomly (left half + mirror) ---------------
        for y in range(height):
            for x in range(width // 2):
                if not self.grid.is_wall(x, y) and self.rng.random() < 0.025:
                    self.grid.apples.add((x, y))
                    self.grid.apples.add(self.grid.opposite(x, y))

        # --- Convert lone walls (no wall-8-neighbours) to apples ------
        for x in range(width):
            for y in range(height):
                if not self.grid.is_wall(x, y):
                    continue
                wall_8_count = sum(
                    1 for nx, ny in self.grid.neighbours(x, y, ADJACENCY_8)
                    if self.grid.is_wall(nx, ny)
                )
                if wall_8_count == 0:
                    self.grid.walls.discard((x, y))
                    self.grid.walls.discard(self.grid.opposite(x, y))
                    self.grid.apples.add((x, y))
                    self.grid.apples.add(self.grid.opposite(x, y))

        # --- Find spawn locations -------------------------------------
        potential_spawns: list[tuple[int, int]] = []
        for x in range(width // 2):  # left half only (will be mirrored)
            for y in range(height):
                if not self.grid.is_wall(x, y):
                    continue
                frees = self._free_above(x, y, SPAWN_HEIGHT)
                if len(frees) >= SPAWN_HEIGHT:
                    potential_spawns.append((x, y))

        self.rng.shuffle(potential_spawns)

        desired = DESIRED_SPAWNS
        if height <= 15:
            desired -= 1
        if height <= 10:
            desired -= 1

        spawn_set: set[tuple[int, int]] = set()
        while desired > 0 and potential_spawns:
            sx, sy = potential_spawns.pop(0)
            spawn_loc = self._free_above(sx, sy, SPAWN_HEIGHT)

            # Check too close to center line or other spawns
            too_close = False
            for cx, cy in spawn_loc:
                if cx == width // 2 - 1 or cx == width // 2:
                    too_close = True
                    break
                for nx, ny in self.grid.neighbours(cx, cy, ADJACENCY_8):
                    opp = self.grid.opposite(nx, ny)
                    if (nx, ny) in spawn_set or opp in spawn_set:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                continue

            for c in spawn_loc:
                spawn_set.add(c)
                self.grid.apples.discard(c)
                self.grid.apples.discard(self.grid.opposite(*c))
            desired -= 1

        self.grid.spawns = list(spawn_set)
        # Mirror spawns to right side
        mirrored_spawns = [self.grid.opposite(x, y) for x, y in spawn_set]
        self.grid.spawns.extend(mirrored_spawns)

        self._validate()
        return self.grid

    def _free_above(self, x: int, y: int, by: int) -> list[tuple[int, int]]:
        """Return up to `by` free cells directly above (x, y)."""
        result = []
        for i in range(1, by + 1):
            ay = y - i
            if self.grid.in_bounds(x, ay) and not self.grid.is_wall(x, ay):
                result.append((x, ay))
            else:
                break
        return result

    def _validate(self) -> None:
        for c in self.grid.apples:
            assert not self.grid.is_wall(*c), f"Apple on wall at {c}"
        assert len(self.grid.apples) == len(set(self.grid.apples)), "Duplicate apples"
