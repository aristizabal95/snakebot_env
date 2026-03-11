"""
Pygame renderer for the Snakebot environment.

Supports render_mode="human" (display window) and "rgb_array" (returns numpy array).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from snakebot_env.core.game import GameState

# Colors (RGB)
COLOR_BG = (30, 30, 40)
COLOR_WALL = (100, 100, 115)
COLOR_APPLE = (220, 60, 60)
PLAYER_BODY = [(60, 120, 220), (220, 140, 40)]   # p0=blue, p1=orange
PLAYER_HEAD = [(120, 180, 255), (255, 200, 80)]   # brighter heads
COLOR_TEXT = (200, 200, 210)

CELL_SIZE = 20   # pixels per tile
HUD_HEIGHT = 40  # pixels for HUD strip at top


class Renderer:
    def __init__(self, game: GameState, mode: str = "human"):
        import pygame
        self.game = game
        self._mode = mode
        if not pygame.get_init():
            pygame.init()
        grid = game.grid
        win_w = grid.width * CELL_SIZE
        win_h = grid.height * CELL_SIZE + HUD_HEIGHT
        self._surf = pygame.Surface((win_w, win_h))
        if mode == "human":
            self._display = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption("Snakebot")
        else:
            self._display = None
        self._font = pygame.font.SysFont("monospace", 14)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        import pygame

        grid = self.game.grid
        win_w = grid.width * CELL_SIZE
        win_h = grid.height * CELL_SIZE + HUD_HEIGHT

        # Resize surface/display if grid changed between episodes
        if self._surf.get_size() != (win_w, win_h):
            self._surf = pygame.Surface((win_w, win_h))
            if mode == "human":
                self._display = pygame.display.set_mode((win_w, win_h))

        surf = self._surf
        surf.fill(COLOR_BG)

        # --- Grid cells -----------------------------------------------
        for wx, wy in grid.walls:
            rect = (wx * CELL_SIZE, HUD_HEIGHT + wy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surf, COLOR_WALL, rect)
            # subtle inner shadow
            pygame.draw.rect(surf, (70, 70, 85), rect, 1)

        for ax, ay in grid.apples:
            cx = ax * CELL_SIZE + CELL_SIZE // 2
            cy = HUD_HEIGHT + ay * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(surf, COLOR_APPLE, (cx, cy), CELL_SIZE // 2 - 2)

        # --- Snakebots ------------------------------------------------
        for bot in self.game.snakebots:
            if not bot.alive:
                continue
            body_color = PLAYER_BODY[bot.owner]
            head_color = PLAYER_HEAD[bot.owner]
            for i, (bx, by) in enumerate(bot.body):
                color = head_color if i == 0 else body_color
                rect = (
                    bx * CELL_SIZE + 1,
                    HUD_HEIGHT + by * CELL_SIZE + 1,
                    CELL_SIZE - 2,
                    CELL_SIZE - 2,
                )
                pygame.draw.rect(surf, color, rect, border_radius=3)

        # --- HUD -------------------------------------------------------
        scores = self.game.scores()
        turn_txt = self._font.render(
            f"Turn {self.game.turn} / {200}", True, COLOR_TEXT
        )
        p0_txt = self._font.render(f"P0: {scores[0]}", True, PLAYER_HEAD[0])
        p1_txt = self._font.render(f"P1: {scores[1]}", True, PLAYER_HEAD[1])
        surf.blit(turn_txt, (5, 10))
        surf.blit(p0_txt, (win_w // 2 - 60, 10))
        surf.blit(p1_txt, (win_w // 2 + 20, 10))

        if mode == "human" and self._display is not None:
            self._display.blit(surf, (0, 0))
            pygame.display.flip()
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.array3d(surf)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        import pygame
        if pygame.get_init():
            pygame.quit()
        self._surf = None
        self._display = None
