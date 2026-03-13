# Snakebot Environment

A multi-agent reinforcement learning environment for the Winter Challenge 2026 (Exotec theme) on CodinGame. Two players each control multiple snakebots on a 2D grid with gravity mechanics. Snakebots grow by eating power sources (apples), and the player with the highest total body size at the end wins.

Built on [PettingZoo](https://pettingzoo.farama.org/) (ParallelEnv API) with [pygame](https://www.pygame.org/) rendering.

## Game Overview

- **Players:** 2 players, each controlling 2 snakebots (default)
- **Objective:** Accumulate the highest total body size by eating apples
- **Mechanics:**
  - Snakebots move on a 2D grid subject to gravity
  - Eating an apple grows the bot by 1 body part
  - Head-on collisions with walls or bodies result in beheading (losing body parts)
  - Game ends when apples are exhausted, all bots of a player die, or 200 turns elapse
- **Leagues:** 1 (Bronze/easy) to 4 (Legend/hard) — affects grid size and complexity

## Installation

### Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install with uv

```bash
git clone <repository-url>
cd snakebot_env
uv sync
```

### Install with pip

```bash
git clone <repository-url>
cd snakebot_env
pip install -e .
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `gymnasium` | >=1.2.3 | Base RL environment interface |
| `pettingzoo` | >=1.25.0 | Multi-agent ParallelEnv API |
| `numpy` | >=2.4.3 | Observation arrays |
| `pygame` | >=2.6.1 | Rendering |

## Quick Start

### Run the Interactive Demo

```bash
uv run python main.py
```

This launches a pygame window with two random agents playing at 4 FPS. Turn-by-turn rewards and final scores are printed to the console.

### Basic Usage

```python
from snakebot_env import SnakebotEnv

env = SnakebotEnv(
    num_players=2,
    bots_per_player=2,
    league_level=3,       # 1=easy (bronze), 4=hard (legend)
    render_mode="human",  # "human", "rgb_array", or None
    seed=42
)

observations, infos = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminated, truncated, infos = env.step(actions)
    env.render()

scores = env._game.scores()
print(f"Final scores: {scores}")  # {0: <p0_size>, 1: <p1_size>}
env.close()
```

## Environment API

### Constructor

```python
SnakebotEnv(
    num_players: int = 2,
    bots_per_player: int = 2,
    league_level: int = 4,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None
)
```

### Agents

Agent IDs follow the pattern `"p{player}_b{bot}"`:

```python
env.possible_agents  # ["p0_b0", "p0_b1", "p1_b0", "p1_b1"]
env.agents           # Active agents (shrinks as bots die)
```

### Observation Space

Each agent receives a `Box(-1.0, 1.0, shape=(3, H, W))` array with 3 channels:

| Channel | Contents |
|---------|----------|
| 0 — Map | `1.0` = wall, `-1.0` = apple, `0.0` = empty |
| 1 — Self | `1.0` = own body, `-1.0` = own head, `0.0` = elsewhere |
| 2 — Others | `1.0` = ally body, `-1.0` = enemy body, `0.0` = elsewhere |

Grid size varies by league level (up to 25×45 at legend level).

### Action Space

`Discrete(4)` — one action per agent per step:

| Value | Direction |
|-------|-----------|
| 0 | UP |
| 1 | DOWN |
| 2 | LEFT |
| 3 | RIGHT |

180° reversals are silently ignored (the bot continues in its current direction).

### Reward Structure

| Event | Reward |
|-------|--------|
| Eat an apple | `+1.0` |
| Beheaded (lose 1 body part) | `-1.0` |
| Killed outright (lose N parts) | `-N` |
| Otherwise | `0.0` |

## Project Structure

```
snakebot_env/
├── pyproject.toml          # Project metadata and dependencies
├── main.py                 # Interactive demo script
└── snakebot_env/
    ├── __init__.py         # Exports SnakebotEnv
    ├── env.py              # PettingZoo ParallelEnv implementation
    ├── renderer.py         # Pygame renderer (human / rgb_array)
    ├── core/
    │   ├── game.py         # GameState, turn logic (moves, eats, collisions, gravity)
    │   ├── grid.py         # Grid data structure and helpers
    │   └── snakebot.py     # Snakebot agent definition
    └── generation/
        └── grid_maker.py   # Procedural level generator
```

## Development

### Running Tests

```bash
uv run pytest
```

### Code Style

```bash
uv run ruff check .
uv run mypy snakebot_env/
```

## License

MIT License
