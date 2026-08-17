import argparse
from pathlib import Path

import gymnasium
import numpy as np
import torch
import torch.nn.functional as functional
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from game_env import make_env


MODEL_DIRECTORY = Path(__file__).parent / "models"
BEST_MODEL_PATH = MODEL_DIRECTORY / "dqn_scratch_best"
BIRD_X = 0.2
BIRD_CENTER_OFFSET = 12 / 512
GAP_TARGET_OFFSET = 25 / 512
NEXT_PIPE_THRESHOLD = 5 / 288


class RelativeCompactObservation(gymnasium.ObservationWrapper):
    """Keep 12 values, but put the upcoming pipe first and use relative distances."""

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        bird_center = float(observation[9]) + BIRD_CENTER_OFFSET
        pipes = []

        for pipe in observation[:9].reshape(3, 3):
            x, top, bottom = map(float, pipe)
            hidden = np.isclose(top, 0.0) and np.isclose(bottom, 1.0)
            if hidden:
                priority = 2
            elif x > NEXT_PIPE_THRESHOLD:
                priority = 0
            else:
                priority = 1
            pipes.append((priority, x, top, bottom, hidden))

        pipes.sort(key=lambda pipe: (pipe[0], pipe[1]))
        relative_pipes = []

        for _, x, top, bottom, hidden in pipes:
            if hidden:
                relative_pipes.extend((0.8, -1.0, 1.0))
            else:
                relative_pipes.extend(
                    (
                        x - BIRD_X,
                        top - bird_center,
                        bottom - bird_center,
                    )
                )

        result = np.asarray(
            relative_pipes + observation[9:12].tolist(),
            dtype=np.float32,
        )
        return np.clip(result, -1.0, 1.0)


class CompactTrainingReward(gymnasium.Wrapper):
    """Dense training feedback derived only from the relative 12-value state."""

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
            reward = -5.0
        elif info["score"] > self.previous_score:
            reward = 5.0
        else:
            gap_error = (observation[1] + observation[2]) / 2 + GAP_TARGET_OFFSET
            reward = 0.1 - abs(gap_error)

        self.previous_score = info["score"]
        return observation, reward, terminated, truncated, info


def make_dqn_env(training: bool = False, render_mode=None, score_limit: int = 200):
    env = make_env(render_mode=render_mode, score_limit=score_limit)
    env = RelativeCompactObservation(env)
    return CompactTrainingReward(env) if training else env


class BiasedDoubleDQN(DQN):
    """Double DQN with configurable biased epsilon-random actions."""

    def __init__(self, *args, random_flap_probability: float = 0.2, **kwargs):
        self.random_flap_probability = random_flap_probability
        super().__init__(*args, **kwargs)

    def _biased_actions(self, count: int) -> np.ndarray:
        return np.random.binomial(
            1,
            self.random_flap_probability,
            size=count,
        ).astype(np.int64)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            action = self._biased_actions(n_envs)
            return action, action
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic=False,
    ):
        if not deterministic and np.random.random() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                count = observation.shape[0]
                action = self._biased_actions(count)
            else:
                action = np.asarray(self._biased_actions(1)[0])
            return action, state

        return self.policy.predict(
            observation,
            state,
            episode_start,
            deterministic,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with torch.no_grad():
                next_actions = self.q_net(replay_data.next_observations).argmax(
                    dim=1,
                    keepdim=True,
                )
                next_q_values = self.q_net_target(replay_data.next_observations).gather(
                    dim=1,
                    index=next_actions,
                )
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations).gather(
                dim=1,
                index=replay_data.actions.long(),
            )
            loss = functional.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


def probability_name(probability: float) -> str:
    return f"dqn_scratch_flap_{round(probability * 100):02d}"


def model_path(probability: float) -> Path:
    return MODEL_DIRECTORY / probability_name(probability)


def build_model(env, flap_probability: float, seed: int) -> BiasedDoubleDQN:
    return BiasedDoubleDQN(
        "MlpPolicy",
        env,
        random_flap_probability=flap_probability,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.5,
        policy_kwargs={"net_arch": [256, 256]},
        device="cuda",
        seed=seed,
        verbose=1,
    )


def train(
    flap_probability: float,
    steps: int,
    seed: int,
    fresh: bool = False,
) -> Path:
    env = make_vec_env(
        lambda: make_dqn_env(training=True),
        n_envs=1,
        seed=seed,
    )
    path = model_path(flap_probability)

    if path.with_suffix(".zip").exists() and not fresh:
        print(f"Continuing {path}.zip")
        model = BiasedDoubleDQN.load(path, env=env, device="cuda")
        model.random_flap_probability = flap_probability
    else:
        print(
            f"Training from scratch with random actions: "
            f"idle={1 - flap_probability:.0%}, flap={flap_probability:.0%}"
        )
        model = build_model(env, flap_probability, seed)

    model.learn(
        total_timesteps=steps,
        reset_num_timesteps=not (path.with_suffix(".zip").exists() and not fresh),
        log_interval=100,
    )
    MODEL_DIRECTORY.mkdir(exist_ok=True)
    model.save(path)
    env.close()
    print(f"Saved {path}.zip")
    return path


def evaluate_path(path: Path, episodes: int, score_limit: int) -> dict:
    model = BiasedDoubleDQN.load(path, device="cpu")
    env = make_dqn_env(score_limit=score_limit)
    scores = []

    for seed in range(8_000, 8_000 + episodes):
        observation, _ = env.reset(seed=seed)

        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                scores.append(info["score"])
                break

    env.close()
    scores = np.asarray(scores)
    result = {
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "min": int(scores.min()),
        "max": int(scores.max()),
    }
    print(
        f"{path.stem}: mean={result['mean']:.2f}, "
        f"median={result['median']:.1f}, min={result['min']}, max={result['max']}"
    )
    return result


def run_experiment(probabilities, steps: int, episodes: int, score_limit: int) -> None:
    results = []

    for index, probability in enumerate(probabilities):
        path = train(probability, steps, seed=100 + index, fresh=True)
        result = evaluate_path(path, episodes, score_limit)
        results.append((result["mean"], probability, path))

    _, best_probability, best_path = max(results)
    best_model = BiasedDoubleDQN.load(best_path, device="cpu")
    best_model.save(BEST_MODEL_PATH)
    print(
        f"Best exploration mix: idle={1 - best_probability:.0%}, "
        f"flap={best_probability:.0%}. Saved {BEST_MODEL_PATH}.zip"
    )


def play(score_limit: int) -> None:
    if not BEST_MODEL_PATH.with_suffix(".zip").exists():
        raise FileNotFoundError("No best DQN model found. Run the experiment first.")

    model = BiasedDoubleDQN.load(BEST_MODEL_PATH, device="cpu")
    env = make_dqn_env(render_mode="human", score_limit=score_limit)
    observation, _ = env.reset()

    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            break

    env.close()
    print(f"Scratch DQN score: {info['score']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Biased-exploration Double DQN agent.")
    parser.add_argument(
        "mode",
        choices=("experiment", "train", "evaluate", "play"),
        nargs="?",
        default="play",
    )
    parser.add_argument("--flap-probability", type=float, default=0.2)
    parser.add_argument("--flap-probabilities", nargs="+", type=float, default=(0.1, 0.2, 0.3))
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--score-limit", type=int, default=200)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    probabilities = list(args.flap_probabilities)
    if any(probability <= 0 or probability >= 0.5 for probability in probabilities):
        parser.error("Exploration flap probabilities must be above 0 and below 0.5.")
    if args.flap_probability <= 0 or args.flap_probability >= 0.5:
        parser.error("Exploration flap probability must be above 0 and below 0.5.")

    if args.mode == "experiment":
        run_experiment(probabilities, args.steps, args.episodes, args.score_limit)
    elif args.mode == "train":
        train(args.flap_probability, args.steps, seed=100, fresh=args.fresh)
    elif args.mode == "evaluate":
        evaluate_path(BEST_MODEL_PATH, args.episodes, args.score_limit)
    else:
        play(args.score_limit)
