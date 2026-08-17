import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from compact_planner import CompactPlanner
from game_env import make_env


MODEL_PATH = Path(__file__).parent / "models" / "compact_ppo_improved"
PRETRAINED_PATH = Path(__file__).parent / "models" / "compact_ppo_pretrained"
TRAIN_SCORE_LIMIT = 200


def build_model(env) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        policy_kwargs={"net_arch": [256, 256]},
        device="cuda",
        seed=1,
        verbose=1,
    )


def collect_demonstrations(episodes: int, score_limit: int):
    env = make_env(score_limit=score_limit)
    expert = CompactPlanner()
    observations = []
    actions = []
    scores = []

    for seed in range(4_000, 4_000 + episodes):
        observation, _ = env.reset(seed=seed)
        expert.reset()

        while True:
            action = expert.predict(observation)
            observations.append(observation.copy())
            actions.append(action)
            observation, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                scores.append(info["score"])
                break

    env.close()
    print(
        f"Collected {len(actions):,} expert decisions; "
        f"expert mean={np.mean(scores):.2f}, max={np.max(scores)}"
    )
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.int64)


def pretrain_policy(model: PPO, observations, actions, epochs: int = 20) -> None:
    device = model.device
    observation_tensor = torch.as_tensor(observations, device=device)
    action_tensor = torch.as_tensor(actions, device=device)
    counts = np.bincount(actions, minlength=2)
    class_weights = len(actions) / (2 * np.maximum(counts, 1))
    class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-4)
    batch_size = 1_024

    for epoch in range(epochs):
        permutation = torch.randperm(len(actions), device=device)
        total_loss = 0.0
        correct = 0

        for start in range(0, len(actions), batch_size):
            indices = permutation[start : start + batch_size]
            distribution = model.policy.get_distribution(observation_tensor[indices])
            logits = distribution.distribution.logits
            loss = functional.cross_entropy(
                logits,
                action_tensor[indices],
                weight=class_weights,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(indices)
            correct += (logits.argmax(dim=1) == action_tensor[indices]).sum().item()

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"Imitation epoch {epoch + 1:2d}: "
                f"loss={total_loss / len(actions):.4f}, "
                f"accuracy={correct / len(actions):.2%}"
            )


def load_model(device: str = "cpu") -> PPO:
    if not MODEL_PATH.with_suffix(".zip").exists():
        raise FileNotFoundError("No improved model found. Run: python ppo_agent.py train")
    return PPO.load(MODEL_PATH, device=device)


def evaluate(episodes: int, score_limit: int) -> list[int]:
    model = load_model()
    env = make_env(score_limit=score_limit)
    scores = []

    for seed in range(6_000, 6_000 + episodes):
        observation, _ = env.reset(seed=seed)

        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(int(action))

            if terminated or truncated:
                scores.append(info["score"])
                break

    env.close()
    score_array = np.asarray(scores)
    print(
        f"Improved PPO over {episodes} episodes: "
        f"mean={score_array.mean():.2f}, median={np.median(score_array):.1f}, "
        f"min={score_array.min()}, max={score_array.max()}"
    )
    return scores


def train(steps: int, expert_episodes: int, expert_score_limit: int) -> None:
    training_env = make_vec_env(
        lambda: make_env(score_limit=TRAIN_SCORE_LIMIT),
        n_envs=8,
        seed=1,
    )

    if MODEL_PATH.with_suffix(".zip").exists():
        print(f"Continuing from {MODEL_PATH}.zip")
        model = PPO.load(MODEL_PATH, env=training_env, device="cuda")
    else:
        model = build_model(training_env)
        observations, actions = collect_demonstrations(
            expert_episodes,
            expert_score_limit,
        )
        pretrain_policy(model, observations, actions)
        PRETRAINED_PATH.parent.mkdir(exist_ok=True)
        model.save(PRETRAINED_PATH)
        print(f"Saved imitation checkpoint to {PRETRAINED_PATH}.zip")

    model.learn(
        total_timesteps=steps,
        reset_num_timesteps=False,
        log_interval=10,
    )
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    training_env.close()
    print(f"Saved improved PPO model to {MODEL_PATH}.zip")


def play(score_limit: int) -> None:
    model = load_model()
    env = make_env(render_mode="human", score_limit=score_limit)
    observation, _ = env.reset()

    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            break

    env.close()
    print(f"Improved PPO score: {info['score']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run the improved PPO agent.")
    parser.add_argument("mode", choices=("train", "evaluate", "play"), nargs="?", default="play")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--score-limit", type=int, default=200)
    parser.add_argument("--expert-episodes", type=int, default=10)
    parser.add_argument("--expert-score-limit", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "train":
        train(args.steps, args.expert_episodes, args.expert_score_limit)
        evaluate(args.episodes, args.score_limit)
    elif args.mode == "evaluate":
        evaluate(args.episodes, args.score_limit)
    else:
        play(args.score_limit)
