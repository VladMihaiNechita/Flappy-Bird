import argparse

import numpy as np

from compact_planner import CompactPlanner
from game_env import make_env


def evaluate(episodes: int, score_limit: int) -> None:
    env = make_env(score_limit=score_limit)
    agent = CompactPlanner()
    scores = []

    for seed in range(2_000, 2_000 + episodes):
        observation, _ = env.reset(seed=seed)
        agent.reset()

        while True:
            action = agent.predict(observation)
            observation, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                scores.append(info["score"])
                break

    env.close()
    scores = np.asarray(scores)
    print(
        f"Planner over {episodes} episodes: "
        f"mean={scores.mean():.2f}, median={np.median(scores):.1f}, "
        f"min={scores.min()}, max={scores.max()}"
    )


def play(score_limit: int) -> None:
    env = make_env(render_mode="human", score_limit=score_limit)
    agent = CompactPlanner()
    observation, _ = env.reset()
    agent.reset()

    while True:
        action = agent.predict(observation)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
    print(f"Planner score: {info['score']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model-based planner agent.")
    parser.add_argument("mode", choices=("play", "evaluate"), nargs="?", default="play")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--score-limit", type=int, default=1_100)
    args = parser.parse_args()

    if args.mode == "evaluate":
        evaluate(args.episodes, args.score_limit)
    else:
        play(args.score_limit)
