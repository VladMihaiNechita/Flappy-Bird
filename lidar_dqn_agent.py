import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from lidar_ablation import (
    AblationDoubleDQN,
    LidarFeatureExtractor,
    ScoreCheckpointCallback,
    VALIDATION_SEEDS,
    evaluate_scores,
    make_experiment_env,
)


MODEL_PATH = Path(__file__).parent / "models" / "lidar_dqn_improved"
OBSERVATION_MODE = "delta"


def make_lidar_env(
    training: bool = False,
    render_mode=None,
    score_limit: int = 200,
):
    return make_experiment_env(
        observation_mode=OBSERVATION_MODE,
        training=training,
        shaping=False,
        render_mode=render_mode,
        score_limit=score_limit,
    )


def build_model(
    env,
    flap_probability: float,
    seed: int,
    n_envs: int,
) -> AblationDoubleDQN:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AblationDoubleDQN(
        "MlpPolicy",
        env,
        random_flap_probability=flap_probability,
        structured_exploration=True,
        cooldown_steps=2,
        learning_rate=2e-4,
        buffer_size=50_000,
        learning_starts=5_000,
        batch_size=128,
        gamma=0.99,
        train_freq=max(16 // n_envs, 1),
        gradient_steps=1,
        n_steps=3,
        target_update_interval=2_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.5,
        policy_kwargs={
            "features_extractor_class": LidarFeatureExtractor,
            "net_arch": [256],
        },
        device=device,
        seed=seed,
        verbose=1,
    )


def train(
    steps: int,
    flap_probability: float,
    seed: int,
    n_envs: int,
    score_limit: int,
    fresh: bool,
) -> None:
    env = make_vec_env(
        lambda: make_lidar_env(training=True, score_limit=score_limit),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    model_file = MODEL_PATH.with_suffix(".zip")
    candidate_path = MODEL_PATH.parent / "lidar_dqn_candidate"
    candidate_file = candidate_path.with_suffix(".zip")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # A previous interrupted run must never be mistaken for this run's result.
    if candidate_file.exists():
        candidate_file.unlink()

    if model_file.exists() and not fresh:
        print(f"Continuing from {model_file}")
        model = AblationDoubleDQN.load(MODEL_PATH, env=env, device=device)
        model.random_flap_probability = flap_probability
    else:
        print(
            "Training improved LIDAR DQN from scratch with random triggers: "
            f"idle={1 - flap_probability:.0%}, flap={flap_probability:.0%}"
        )
        model = build_model(env, flap_probability, seed, n_envs)

    checkpoint = ScoreCheckpointCallback(
        model_path=candidate_path,
        observation_mode=OBSERVATION_MODE,
        interval=50_000,
        score_limit=score_limit,
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint,
        reset_num_timesteps=fresh,
        log_interval=200,
    )
    env.close()
    if not candidate_file.exists():
        model.save(candidate_path)

    candidate = AblationDoubleDQN.load(candidate_path, device="cpu")
    candidate_result = evaluate_scores(
        candidate,
        OBSERVATION_MODE,
        VALIDATION_SEEDS,
        score_limit,
    )
    candidate_key = (
        candidate_result["median"],
        candidate_result["mean"],
        candidate_result["min"],
    )

    promote = not model_file.exists()
    if model_file.exists():
        current = AblationDoubleDQN.load(MODEL_PATH, device="cpu")
        current_result = evaluate_scores(
            current,
            OBSERVATION_MODE,
            VALIDATION_SEEDS,
            score_limit,
        )
        current_key = (
            current_result["median"],
            current_result["mean"],
            current_result["min"],
        )
        promote = candidate_key > current_key
        print(
            "Validation comparison: "
            f"candidate median={candidate_result['median']:.1f}, "
            f"mean={candidate_result['mean']:.2f}; "
            f"current median={current_result['median']:.1f}, "
            f"mean={current_result['mean']:.2f}"
        )

    if promote:
        candidate.save(MODEL_PATH)
        print(f"Candidate promoted to {model_file}")
    else:
        print(f"Candidate rejected; retained {model_file}")
    candidate_file.unlink()


def load_model(device: str = "cpu") -> AblationDoubleDQN:
    if not MODEL_PATH.with_suffix(".zip").exists():
        raise FileNotFoundError(
            "No improved LIDAR model found. Run: python lidar_dqn_agent.py train"
        )
    return AblationDoubleDQN.load(MODEL_PATH, device=device)


def evaluate(episodes: int, score_limit: int) -> dict:
    model = load_model()
    env = make_lidar_env(score_limit=score_limit)
    scores = []

    for seed in range(10_000, 10_000 + episodes):
        observation, _ = env.reset(seed=seed)
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                scores.append(info["score"])
                break

    env.close()
    score_array = np.asarray(scores)
    result = {
        "mean": float(score_array.mean()),
        "median": float(np.median(score_array)),
        "min": int(score_array.min()),
        "max": int(score_array.max()),
    }
    print(
        f"Improved LIDAR DQN over {episodes} episodes: "
        f"mean={result['mean']:.2f}, median={result['median']:.1f}, "
        f"min={result['min']}, max={result['max']}"
    )
    return result


def play(score_limit: int) -> None:
    model = load_model()
    env = make_lidar_env(render_mode="human", score_limit=score_limit)
    observation, _ = env.reset()

    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            break

    env.close()
    print(f"Improved LIDAR DQN score: {info['score']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved scratch LIDAR DQN.")
    parser.add_argument(
        "mode",
        choices=("train", "evaluate", "play"),
        nargs="?",
        default="play",
    )
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--score-limit", type=int, default=200)
    parser.add_argument("--flap-probability", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if args.flap_probability <= 0 or args.flap_probability >= 0.5:
        parser.error("Exploration flap probability must be above 0 and below 0.5.")

    if args.mode == "train":
        train(
            args.steps,
            args.flap_probability,
            args.seed,
            args.n_envs,
            args.score_limit,
            args.fresh,
        )
        evaluate(args.episodes, args.score_limit)
    elif args.mode == "evaluate":
        evaluate(args.episodes, args.score_limit)
    else:
        play(args.score_limit)
