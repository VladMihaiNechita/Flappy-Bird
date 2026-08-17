import flappy_bird_gymnasium
import gymnasium
import numpy as np


class SafeObservation(gymnasium.ObservationWrapper):
    """Clip numerical noise and return neural-network-friendly float32 values."""

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        low = env.observation_space.low.astype(np.float32)
        high = env.observation_space.high.astype(np.float32)
        self.observation_space = gymnasium.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.clip(
            observation,
            self.observation_space.low,
            self.observation_space.high,
        ).astype(np.float32)


def make_env(render_mode=None, score_limit: int = 100):
    env = gymnasium.make(
        "FlappyBird-v0",
        render_mode=render_mode,
        use_lidar=False,
        score_limit=score_limit,
        disable_env_checker=True,
    )
    return SafeObservation(env)


def make_lidar_env(render_mode=None, score_limit: int = 100):
    env = gymnasium.make(
        "FlappyBird-v0",
        render_mode=render_mode,
        use_lidar=True,
        score_limit=score_limit,
        disable_env_checker=True,
    )
    return SafeObservation(env)
