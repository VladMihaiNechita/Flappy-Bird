import argparse
from pathlib import Path

import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from game_env import make_env as make_base_env


MODEL_PATH = Path(__file__).parent / "models" / "compact_ppo"
SCORE_LIMIT = 100
BIRD_CENTER_OFFSET = 12 / 512
GAP_TARGET_OFFSET = 25 / 512
NEXT_PIPE_X_THRESHOLD = 5 / 288


class GapReward(gymnasium.Wrapper):
    """Give the learner a small hint toward the center of the upcoming gap."""

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        self.previous_score = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_score = info["score"]
        return observation, info

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)

        if terminated:
            reward = -1.0
        elif info["score"] > self.previous_score:
            reward = 1.0
        else:
            pipes = observation[:9].reshape(3, 3)
            upcoming = pipes[pipes[:, 0] > NEXT_PIPE_X_THRESHOLD]
            next_pipe = upcoming[np.argmin(upcoming[:, 0])]
            gap_center = (next_pipe[1] + next_pipe[2]) / 2 + GAP_TARGET_OFFSET
            bird_center = observation[9] + BIRD_CENTER_OFFSET
            reward = 0.1 - abs(bird_center - gap_center)

        self.previous_score = info["score"]
        return observation, reward, terminated, truncated, info


def make_env(render_mode=None, training: bool = False, score_limit: int = SCORE_LIMIT):
    env = make_base_env(render_mode=render_mode, score_limit=score_limit)
    return GapReward(env) if training else env


def train(total_steps: int) -> None:
    env = make_vec_env(
        lambda: make_env(training=True),
        n_envs=8,
        seed=0,
    )
    model_file = MODEL_PATH.with_suffix(".zip")

    if model_file.exists():
        print(f"Continuing training from {model_file}")
        model = PPO.load(MODEL_PATH, env=env, device="cuda")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [128, 128]},
            device="cuda",
            seed=0,
            verbose=1,
        )

    model.learn(
        total_timesteps=total_steps,
        reset_num_timesteps=False,
        log_interval=10,
    )
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    env.close()
    print(f"Saved model to {MODEL_PATH}.zip")


def load_model() -> PPO:
    model_file = MODEL_PATH.with_suffix(".zip")
    if not model_file.exists():
        raise FileNotFoundError("No trained model found. Run: python main.py train")
    return PPO.load(MODEL_PATH, device="cpu")


def evaluate(episodes: int = 100) -> None:
    model = load_model()
    env = make_env()
    scores = []

    for seed in range(1_000, 1_000 + episodes):
        observation, _ = env.reset(seed=seed)

        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(int(action))

            if terminated or truncated:
                scores.append(info["score"])
                break

    env.close()
    scores = np.asarray(scores)
    print(
        f"Evaluation over {episodes} episodes: "
        f"mean={scores.mean():.2f}, median={np.median(scores):.1f}, "
        f"min={scores.min()}, max={scores.max()}, "
        f"reached {SCORE_LIMIT}={np.mean(scores == SCORE_LIMIT):.0%}"
    )


def play() -> None:
    model = load_model()
    env = make_env(render_mode="human")
    observation, _ = env.reset()

    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(int(action))

        if terminated or truncated:
            break

    env.close()
    print(f"Score: {info['score']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run the compact Flappy Bird agent.")
    parser.add_argument(
        "mode",
        choices=("train", "evaluate", "play"),
        nargs="?",
        default="play",
    )
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "train":
        train(args.steps)
        evaluate(args.episodes)
    elif args.mode == "evaluate":
        evaluate(args.episodes)
    else:
        play()
