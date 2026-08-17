from collections import deque
from heapq import nsmallest

import numpy as np


SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
BIRD_X = int(SCREEN_WIDTH * 0.2)
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
PIPE_WIDTH = 52
PIPE_SPEED = 4
GROUND_LIMIT = int(SCREEN_HEIGHT * 0.79 - BIRD_HEIGHT - 1)
MAX_FALL_SPEED = 10
FLAP_SPEED = -9
SAFETY_MARGIN = 4


class CompactPlanner:
    """Plan safe flap sequences directly from the compact 12-value state."""

    def __init__(self, beam_width: int = 200) -> None:
        self.beam_width = beam_width
        self.actions = deque()
        self.visible_pipe_count = 0

    def reset(self) -> None:
        self.actions.clear()
        self.visible_pipe_count = 0

    def predict(self, observation: np.ndarray) -> int:
        pipes, visible_count = self._read_pipes(observation)

        if not self.actions or visible_count > self.visible_pipe_count:
            self.actions = deque(self._plan(observation, pipes))

        self.visible_pipe_count = visible_count
        return self.actions.popleft()

    @staticmethod
    def _read_pipes(observation: np.ndarray):
        pipes = []
        visible_count = 0

        for x, top, bottom in observation[:9].reshape(3, 3):
            hidden = np.isclose(top, 0.0) and np.isclose(bottom, 1.0)
            if hidden:
                continue

            visible_count += 1
            pipe = (
                float(x * SCREEN_WIDTH),
                float(top * SCREEN_HEIGHT),
                float(bottom * SCREEN_HEIGHT),
            )
            if pipe[0] + PIPE_WIDTH > BIRD_X:
                pipes.append(pipe)

        return sorted(pipes), visible_count

    def _plan(self, observation: np.ndarray, pipes) -> list[int]:
        bird_y = int(round(float(observation[9]) * SCREEN_HEIGHT))
        bird_velocity = int(round(float(observation[10]) * MAX_FALL_SPEED))
        last_pipe_x = max(pipe[0] for pipe in pipes)
        horizon = int(np.ceil((last_pipe_x + PIPE_WIDTH - BIRD_X) / PIPE_SPEED)) + 1

        layers = [{(bird_y, bird_velocity): (0.0, None, None)}]

        for step in range(1, horizon + 1):
            next_layer = {}

            for state, (cost, _, _) in layers[-1].items():
                y, velocity = state

                for action in (0, 1):
                    if action == 1:
                        next_velocity = FLAP_SPEED
                    else:
                        next_velocity = min(velocity + 1, MAX_FALL_SPEED)

                    next_y = y + next_velocity
                    future_pipes = [
                        (x - PIPE_SPEED * step, top, bottom)
                        for x, top, bottom in pipes
                    ]

                    if not self._is_safe(next_y, future_pipes):
                        continue

                    next_cost = cost + self._step_cost(
                        next_y,
                        next_velocity,
                        action,
                        future_pipes,
                    )
                    next_state = (next_y, next_velocity)
                    previous = next_layer.get(next_state)

                    if previous is None or next_cost < previous[0]:
                        next_layer[next_state] = (next_cost, state, action)

            if not next_layer:
                return [self._fallback_action(observation)]

            if len(next_layer) > self.beam_width:
                next_layer = dict(
                    nsmallest(
                        self.beam_width,
                        next_layer.items(),
                        key=lambda item: item[1][0],
                    )
                )

            layers.append(next_layer)

        state = min(layers[-1], key=lambda item: layers[-1][item][0])
        actions = []

        for step in range(horizon, 0, -1):
            _, previous_state, action = layers[step][state]
            actions.append(action)
            state = previous_state

        actions.reverse()
        return actions

    @staticmethod
    def _is_safe(bird_y: int, pipes) -> bool:
        if bird_y < 0 or bird_y > GROUND_LIMIT:
            return False

        for pipe_x, gap_top, gap_bottom in pipes:
            overlaps = pipe_x < BIRD_X + BIRD_WIDTH and pipe_x + PIPE_WIDTH > BIRD_X
            if overlaps and not (
                bird_y >= gap_top + SAFETY_MARGIN
                and bird_y + BIRD_HEIGHT <= gap_bottom - SAFETY_MARGIN
            ):
                return False

        return True

    @staticmethod
    def _step_cost(bird_y: int, velocity: int, action: int, pipes) -> float:
        upcoming = [pipe for pipe in pipes if pipe[0] + PIPE_WIDTH > BIRD_X]
        if not upcoming:
            return 0.01 * abs(velocity) + 0.01 * action

        current = min(upcoming)
        gap_center = (current[1] + current[2] - BIRD_HEIGHT) / 2

        later = [pipe for pipe in upcoming if pipe is not current]
        if later:
            next_pipe = min(later)
            next_center = (next_pipe[1] + next_pipe[2] - BIRD_HEIGHT) / 2
            low = current[1] + SAFETY_MARGIN
            high = current[2] - BIRD_HEIGHT - SAFETY_MARGIN
            gap_center = float(np.clip(next_center, low, high))

        distance_to_pipe = max(0.0, current[0] - (BIRD_X + BIRD_WIDTH))
        position_weight = 1.0 / (1.0 + distance_to_pipe / 20.0)

        return (
            position_weight * (bird_y - gap_center) ** 2
            + 0.05 * abs(velocity)
            + 0.01 * action
        )

    @staticmethod
    def _fallback_action(observation: np.ndarray) -> int:
        pipes, _ = CompactPlanner._read_pipes(observation)
        if not pipes:
            return 0

        _, top, bottom = pipes[0]
        target = (top + bottom - BIRD_HEIGHT) / 2 + 25
        bird_y = float(observation[9]) * SCREEN_HEIGHT
        velocity = float(observation[10]) * MAX_FALL_SPEED
        return int(bird_y > target and velocity >= 1)
