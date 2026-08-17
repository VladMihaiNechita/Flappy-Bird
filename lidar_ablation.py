import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import gymnasium
import numpy as np
import torch
import torch.nn.functional as functional
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

from dqn_agent import BiasedDoubleDQN
from game_env import make_lidar_env as make_base_lidar_env


MODEL_DIRECTORY = Path(__file__).parent / "models" / "lidar_ablation"
VALIDATION_SEEDS = tuple(range(12_000, 12_010))
SCREENING_SEEDS = tuple(range(13_000, 13_020))


class LidarObservation(gymnasium.ObservationWrapper):
    """Create raw stacks, proximity/delta channels, or proximity sequences."""

    def __init__(self, env: gymnasium.Env, mode: str) -> None:
        super().__init__(env)
        self.mode = mode
        self.stack_size = 8 if mode == "sequence" else 4
        self.frames = deque(maxlen=self.stack_size)
        low = -1.0 if mode == "delta" else 0.0
        self.observation_space = gymnasium.spaces.Box(
            low=low,
            high=1.0,
            shape=(self.stack_size if mode == "sequence" else 4, 180),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(observation.copy())
        return self._transform(), info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.frames.append(observation.copy())
        return self._transform()

    def _transform(self) -> np.ndarray:
        frames = np.asarray(self.frames, dtype=np.float32)
        if self.mode == "raw":
            return frames

        proximity = 1.0 - frames
        if self.mode == "sequence":
            return proximity

        return np.stack(
            (
                proximity[-1],
                proximity[-1] - proximity[-2],
                proximity[-2] - proximity[-3],
                proximity[-3] - proximity[-4],
            )
        ).astype(np.float32)


class LidarReward(gymnasium.Wrapper):
    """Use only LIDAR-derived feedback, optionally with potential shaping."""

    def __init__(self, env: gymnasium.Env, shaping: bool) -> None:
        super().__init__(env)
        self.shaping = shaping
        self.previous_score = 0
        self.previous_potential = 0.0

    @staticmethod
    def potential(observation: np.ndarray) -> float:
        forward_cone = observation[45:135]
        return float(np.quantile(forward_cone, 0.20))

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_score = info["score"]
        self.previous_potential = self.potential(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            reward = -5.0
        elif info["score"] > self.previous_score:
            reward = 5.0

        if self.shaping:
            next_potential = 0.0 if terminated else self.potential(observation)
            reward += 2.0 * (0.99 * next_potential - self.previous_potential)
            self.previous_potential = next_potential

        self.previous_score = info["score"]
        return observation, reward, terminated, truncated, info


def make_experiment_env(
    observation_mode: str,
    training: bool = False,
    shaping: bool = False,
    render_mode=None,
    score_limit: int = 100,
):
    env = make_base_lidar_env(render_mode=render_mode, score_limit=score_limit)
    if training:
        env = LidarReward(env, shaping=shaping)
    return LidarObservation(env, mode=observation_mode)


class LidarFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Box) -> None:
        super().__init__(observation_space, features_dim=256)
        channels = observation_space.shape[0]
        self.convolution = torch.nn.Sequential(
            torch.nn.Conv1d(channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            flattened_size = self.convolution(sample).shape[1]
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(flattened_size, 256),
            torch.nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolution(observations))


class RecurrentLidarFeatureExtractor(BaseFeaturesExtractor):
    """Encode each scan spatially, then integrate the eight scans with a GRU."""

    def __init__(self, observation_space: gymnasium.spaces.Box) -> None:
        super().__init__(observation_space, features_dim=256)
        self.spatial = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, 1, observation_space.shape[1])
            spatial_size = self.spatial(sample).shape[1]
        self.scan_projection = torch.nn.Sequential(
            torch.nn.Linear(spatial_size, 128),
            torch.nn.ReLU(),
        )
        self.recurrent = torch.nn.GRU(128, 256, batch_first=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch, sequence, rays = observations.shape
        scans = observations.reshape(batch * sequence, 1, rays)
        features = self.scan_projection(self.spatial(scans))
        features = features.reshape(batch, sequence, -1)
        _, hidden = self.recurrent(features)
        return hidden[-1]


class DuelingQNetwork(QNetwork):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.q_net = torch.nn.Identity()
        action_count = int(self.action_space.n)
        self.value_stream = torch.nn.Sequential(
            *create_mlp(
                self.features_dim,
                1,
                self.net_arch,
                self.activation_fn,
            )
        )
        self.advantage_stream = torch.nn.Sequential(
            *create_mlp(
                self.features_dim,
                action_count,
                self.net_arch,
                self.activation_fn,
            )
        )

    def forward(self, obs: PyTorchObs) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(self) -> DuelingQNetwork:
        net_args = self._update_features_extractor(
            self.net_args,
            features_extractor=None,
        )
        return DuelingQNetwork(**net_args).to(self.device)


class PrioritizedReplaySamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    weights: torch.Tensor
    flat_indices: np.ndarray


class SumTree:
    def __init__(self, capacity: int) -> None:
        self.leaf_count = 1
        while self.leaf_count < capacity:
            self.leaf_count *= 2
        self.values = np.zeros(2 * self.leaf_count, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.values[1])

    def update(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for index, priority in zip(indices, priorities):
            position = self.leaf_count + int(index)
            difference = float(priority) - self.values[position]
            while position >= 1:
                self.values[position] += difference
                position //= 2

    def sample(self, count: int) -> np.ndarray:
        segments = self.total / count
        masses = (np.arange(count) + np.random.random(count)) * segments
        positions = np.ones(count, dtype=np.int64)
        while positions[0] < self.leaf_count:
            left = positions * 2
            go_right = masses >= self.values[left]
            masses = masses - self.values[left] * go_right
            positions = left + go_right
        return positions - self.leaf_count


class PrioritizedNStepReplayBuffer(NStepReplayBuffer):
    def __init__(
        self,
        *args,
        alpha: float = 0.6,
        beta: float = 0.4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.maximum_priority = 1.0
        self.transition_capacity = self.buffer_size * self.n_envs
        self.tree = SumTree(self.transition_capacity)

    def add(self, *args, **kwargs) -> None:
        super().add(*args, **kwargs)
        written_position = (self.pos - 1) % self.buffer_size
        flat_indices = written_position * self.n_envs + np.arange(self.n_envs)
        priorities = np.full(self.n_envs, self.maximum_priority**self.alpha)
        self.tree.update(flat_indices, priorities)

    def sample(
        self,
        batch_size: int,
        env: VecNormalize | None = None,
    ) -> PrioritizedReplaySamples:
        flat_indices = self.tree.sample(batch_size)
        batch_inds = flat_indices // self.n_envs
        env_indices = flat_indices % self.n_envs
        probabilities = self.tree.values[
            self.tree.leaf_count + flat_indices
        ] / self.tree.total
        available = self.buffer_size * self.n_envs if self.full else self.pos * self.n_envs
        weights = (available * probabilities) ** (-self.beta)
        weights /= weights.max()
        return self._get_prioritized_samples(
            batch_inds,
            env_indices,
            weights.astype(np.float32),
            flat_indices,
            env,
        )

    def _get_prioritized_samples(
        self,
        batch_inds: np.ndarray,
        env_indices: np.ndarray,
        weights: np.ndarray,
        flat_indices: np.ndarray,
        env: VecNormalize | None,
    ) -> PrioritizedReplaySamples:
        last_valid_index = self.pos - 1
        original_timeouts = self.timeouts[last_valid_index].copy()
        self.timeouts[last_valid_index] = np.logical_or(
            original_timeouts,
            np.logical_not(self.dones[last_valid_index]),
        )
        steps = np.arange(self.n_steps).reshape(1, -1)
        indices = (batch_inds[:, None] + steps) % self.buffer_size
        rewards = self._normalize_reward(
            self.rewards[indices, env_indices[:, None]],
            env,
        )
        dones = self.dones[indices, env_indices[:, None]]
        timeouts = self.timeouts[indices, env_indices[:, None]]
        ended = np.logical_or(dones, timeouts)
        end_index = ended.argmax(axis=1)
        has_ended = ended.any(axis=1)
        end_index = np.where(has_ended, end_index, self.n_steps - 1)
        mask = np.arange(self.n_steps).reshape(1, -1) <= end_index[:, None]
        discounts = self.gamma ** mask.sum(axis=1, keepdims=True).astype(np.float32)
        step_discounts = self.gamma ** np.arange(self.n_steps, dtype=np.float32)
        returns = (rewards * step_discounts.reshape(1, -1) * mask).sum(
            axis=1,
            keepdims=True,
        )
        last_indices = (batch_inds + end_index) % self.buffer_size
        next_observations = self._normalize_obs(
            self.next_observations[last_indices, env_indices],
            env,
        )
        final_dones = (
            self.dones[last_indices, env_indices, None]
            * (1.0 - self.timeouts[last_indices, env_indices, None])
        ).astype(np.float32)
        self.timeouts[last_valid_index] = original_timeouts
        observations = self._normalize_obs(
            self.observations[batch_inds, env_indices],
            env,
        )
        actions = self.actions[batch_inds, env_indices]
        return PrioritizedReplaySamples(
            observations=self.to_torch(observations),
            actions=self.to_torch(actions),
            next_observations=self.to_torch(next_observations),
            dones=self.to_torch(final_dones),
            rewards=self.to_torch(returns),
            discounts=self.to_torch(discounts),
            weights=self.to_torch(weights.reshape(-1, 1)),
            flat_indices=flat_indices,
        )

    def update_priorities(
        self,
        flat_indices: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        priorities = np.maximum(priorities, 1e-6)
        self.maximum_priority = max(self.maximum_priority, float(priorities.max()))
        self.tree.update(flat_indices, priorities**self.alpha)


class AblationDoubleDQN(BiasedDoubleDQN):
    def __init__(
        self,
        *args,
        structured_exploration: bool = False,
        cooldown_steps: int = 2,
        **kwargs,
    ) -> None:
        self.structured_exploration = structured_exploration
        self.cooldown_steps = cooldown_steps
        self.exploration_cooldowns = np.zeros(1, dtype=np.int64)
        super().__init__(*args, **kwargs)

    def _biased_actions(self, count: int) -> np.ndarray:
        if not self.structured_exploration:
            return super()._biased_actions(count)
        if len(self.exploration_cooldowns) != count:
            self.exploration_cooldowns = np.zeros(count, dtype=np.int64)
        actions = np.zeros(count, dtype=np.int64)
        available = self.exploration_cooldowns == 0
        flap = np.random.random(count) < self.random_flap_probability
        actions[available & flap] = 1
        self.exploration_cooldowns = np.maximum(
            self.exploration_cooldowns - 1,
            0,
        )
        self.exploration_cooldowns[actions == 1] = self.cooldown_steps
        return actions

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        prioritized = hasattr(self.replay_buffer, "update_priorities")
        if prioritized:
            self.replay_buffer.beta = 1.0 - 0.6 * self._current_progress_remaining

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
                next_values = self.q_net_target(
                    replay_data.next_observations
                ).gather(1, next_actions)
                target = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_values
                )
            current = self.q_net(replay_data.observations).gather(
                1,
                replay_data.actions.long(),
            )
            element_losses = functional.smooth_l1_loss(
                current,
                target,
                reduction="none",
            )
            if prioritized:
                loss = (element_losses * replay_data.weights).mean()
                priorities = (target - current).detach().abs().cpu().numpy().ravel()
                self.replay_buffer.update_priorities(
                    replay_data.flat_indices,
                    priorities,
                )
            else:
                loss = element_losses.mean()
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
            )
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


def evaluate_scores(
    model: AblationDoubleDQN,
    observation_mode: str,
    seeds,
    score_limit: int,
) -> dict:
    env = make_experiment_env(
        observation_mode=observation_mode,
        score_limit=score_limit,
    )
    scores = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                scores.append(info["score"])
                break
    env.close()
    score_array = np.asarray(scores)
    return {
        "mean": float(score_array.mean()),
        "median": float(np.median(score_array)),
        "min": int(score_array.min()),
        "max": int(score_array.max()),
        "scores": scores,
    }


def _evaluation_worker(arguments) -> list[int]:
    model_path, observation_mode, seeds, score_limit = arguments
    torch.set_num_threads(1)
    model = AblationDoubleDQN.load(model_path, device="cpu")
    return evaluate_scores(
        model,
        observation_mode,
        seeds,
        score_limit,
    )["scores"]


def parallel_evaluate(
    model_path: Path,
    observation_mode: str,
    seeds,
    score_limit: int,
    workers: int,
) -> dict:
    seed_groups = [group.tolist() for group in np.array_split(seeds, workers) if len(group)]
    tasks = [
        (model_path, observation_mode, group, score_limit)
        for group in seed_groups
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        groups = list(executor.map(_evaluation_worker, tasks))
    scores = [score for group in groups for score in group]
    score_array = np.asarray(scores)
    result = {
        "mean": float(score_array.mean()),
        "median": float(np.median(score_array)),
        "min": int(score_array.min()),
        "max": int(score_array.max()),
        "scores": scores,
    }
    print(
        f"TEST mean={result['mean']:.2f} median={result['median']:.1f} "
        f"min={result['min']} max={result['max']} episodes={len(scores)}"
    )
    return result


class ScoreCheckpointCallback(BaseCallback):
    def __init__(
        self,
        model_path: Path,
        observation_mode: str,
        interval: int,
        score_limit: int,
    ) -> None:
        super().__init__(verbose=0)
        self.model_path = model_path
        self.observation_mode = observation_mode
        self.interval = interval
        self.score_limit = score_limit
        self.next_evaluation = interval
        self.best_key = (-np.inf, -np.inf, -np.inf)

    def _on_training_start(self) -> None:
        # When training is resumed, schedule the first evaluation relative to
        # the restored step counter instead of immediately evaluating once for
        # every interval that has already elapsed.
        self.next_evaluation = self.num_timesteps + self.interval

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_evaluation:
            return True
        result = evaluate_scores(
            self.model,
            self.observation_mode,
            VALIDATION_SEEDS,
            self.score_limit,
        )
        key = (result["median"], result["mean"], result["min"])
        print(
            f"CHECKPOINT steps={self.num_timesteps} "
            f"mean={result['mean']:.2f} median={result['median']:.1f} "
            f"min={result['min']} max={result['max']}"
        )
        if key > self.best_key:
            self.best_key = key
            self.model.save(self.model_path)
            print(f"CHECKPOINT saved {self.model_path}.zip")
        self.next_evaluation += self.interval
        return True


def build_model(args, env) -> AblationDoubleDQN:
    feature_extractor = (
        RecurrentLidarFeatureExtractor
        if args.recurrent
        else LidarFeatureExtractor
    )
    policy = DuelingDQNPolicy if args.dueling else "MlpPolicy"
    replay_buffer_class = None
    replay_buffer_kwargs = None
    n_steps = args.n_steps
    if args.prioritized:
        replay_buffer_class = PrioritizedNStepReplayBuffer
        replay_buffer_kwargs = {
            "n_steps": n_steps,
            "gamma": 0.99,
            "alpha": 0.6,
            "beta": 0.4,
        }
        n_steps = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AblationDoubleDQN(
        policy,
        env,
        random_flap_probability=args.flap_probability,
        structured_exploration=args.structured,
        cooldown_steps=2,
        learning_rate=2e-4,
        buffer_size=50_000,
        learning_starts=5_000,
        batch_size=128,
        gamma=0.99,
        train_freq=max(16 // args.n_envs, 1),
        gradient_steps=1,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        n_steps=n_steps,
        target_update_interval=2_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.5,
        policy_kwargs={
            "features_extractor_class": feature_extractor,
            "net_arch": [256],
        },
        device=device,
        seed=args.seed,
        verbose=0,
    )


def run(args) -> dict:
    observation_mode = "sequence" if args.recurrent else args.observation
    env = make_vec_env(
        lambda: make_experiment_env(
            observation_mode=observation_mode,
            training=True,
            shaping=args.shaping,
            score_limit=args.score_limit,
        ),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIRECTORY / f"{args.name}_best"
    model = build_model(args, env)
    callback = ScoreCheckpointCallback(
        model_path=model_path,
        observation_mode=observation_mode,
        interval=args.eval_interval,
        score_limit=args.score_limit,
    )
    print(
        f"START name={args.name} observation={observation_mode} "
        f"dueling={args.dueling} prioritized={args.prioritized} "
        f"n_steps={args.n_steps} structured={args.structured} "
        f"flap={args.flap_probability:.2f} shaping={args.shaping} "
        f"recurrent={args.recurrent} steps={args.steps}"
    )
    model.learn(total_timesteps=args.steps, callback=callback, log_interval=200)
    env.close()
    if not model_path.with_suffix(".zip").exists():
        model.save(model_path)
    best_model = AblationDoubleDQN.load(model_path, device="cpu")
    result = evaluate_scores(
        best_model,
        observation_mode,
        SCREENING_SEEDS,
        args.score_limit,
    )
    print(
        f"RESULT name={args.name} mean={result['mean']:.2f} "
        f"median={result['median']:.1f} min={result['min']} "
        f"max={result['max']} scores={result['scores']}"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental LIDAR DQN tests.")
    parser.add_argument("--name")
    parser.add_argument("--evaluate-model", type=Path)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--test-seed-start", type=int, default=14_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--eval-interval", type=int, default=30_000)
    parser.add_argument("--score-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--observation", choices=("raw", "delta"), default="raw")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--prioritized", action="store_true")
    parser.add_argument("--n-steps", type=int, default=1)
    parser.add_argument("--structured", action="store_true")
    parser.add_argument("--flap-probability", type=float, default=0.2)
    parser.add_argument("--shaping", action="store_true")
    parser.add_argument("--recurrent", action="store_true")
    arguments = parser.parse_args()
    if arguments.evaluate_model:
        parallel_evaluate(
            arguments.evaluate_model,
            "sequence" if arguments.recurrent else arguments.observation,
            range(
                arguments.test_seed_start,
                arguments.test_seed_start + arguments.test_episodes,
            ),
            arguments.score_limit,
            arguments.workers,
        )
    elif arguments.name:
        run(arguments)
    else:
        parser.error("Use --name to train or --evaluate-model to test.")
