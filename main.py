"""
Interactive demo: run the Snakebot env with random agents and pygame renderer.

Usage:
    uv run python main.py
"""
import time

from snakebot_env import SnakebotEnv

FPS = 4  # steps per second


def main() -> None:
    import pygame

    env = SnakebotEnv(
        num_players=2,
        bots_per_player=2,
        league_level=3,
        render_mode="human",
        seed=None,
    )

    running = True
    episode = 0

    while running:
        episode += 1
        obs, _ = env.reset()
        print(f"\n=== Episode {episode} | Grid {env._game.grid.width}x{env._game.grid.height} "
              f"| Apples: {len(env._game.grid.apples)} | Agents: {env.agents}")

        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
                    break

            if not running:
                break
            if not env.agents:
                done = True
                break

            actions = {a: env.action_space(a).sample() for a in env.agents}
            _, rewards, term, trunc, _ = env.step(actions)
            nonzero = {k: v for k, v in rewards.items() if v != 0}
            if nonzero:
                print(f"  turn={env._game.turn} rewards={nonzero}")

            env.render()
            time.sleep(1 / FPS)

        scores = env._game.scores()
        print(f"Episode {episode} ended at turn {env._game.turn} | Scores: {scores}")
        time.sleep(1.0)

    env.close()


if __name__ == "__main__":
    main()
