"""Linear masked True Online Sarsa(lambda) building blocks.

The learner consumes the public Gym observation and action mask only. Its
feature encoder uses deterministic sparse tile coding so tests can inspect the
exact active feature indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np

from .agent import Agent, Transition
from .rules import ACTION_COUNT, OBSERVATION_SIZE, PELLET_COUNT, SURGE_ORB_COUNT


FEATURE_CAPACITY: Final = 1 << 18
TILE_BINS: Final = 8
TILE_TILINGS: Final = 8
TRACE_PRUNE_THRESHOLD: Final = 1e-12

_PLAYER_POSITION = slice(0, 2)
_PLAYER_VELOCITY = slice(2, 4)
_PLAYER_SURGE = 4
_ENEMY_POSITION = slice(5, 7)
_ENEMY_HEADING = slice(7, 15)
_ENEMY_CLOCK = 15
_BARS = slice(16, 24)
_PELLETS = slice(24, 120)
_ORBS = slice(120, 129)
_TIME = 129

# Disjoint hash salts. Rank offsets stay inside each block so pellet rank 1
# cannot collide with an orb or bar that shares the same relative tile.
GROUP_PLAYER_POSITION: Final = 1
GROUP_PLAYER_VELOCITY: Final = 2
GROUP_ENEMY_RELATIVE: Final = 3
GROUP_ENEMY_CLOCK: Final = 4
GROUP_PELLET_RANK: Final = 10
GROUP_ORB_RANK: Final = 20
GROUP_BAR: Final = 30
GROUP_SURGE_TIME: Final = 40
GROUP_ENEMY_HEADING: Final = 50
GROUP_ACTION_MASK: Final = 60
NEAREST_PELLET_COUNT: Final = 4
OCCUPANCY_SALT: Final = -1

FEATURE_GROUP_SALTS: Final = (
    GROUP_PLAYER_POSITION,
    GROUP_PLAYER_VELOCITY,
    GROUP_ENEMY_RELATIVE,
    GROUP_ENEMY_CLOCK,
    *range(GROUP_PELLET_RANK, GROUP_PELLET_RANK + NEAREST_PELLET_COUNT),
    *range(GROUP_ORB_RANK, GROUP_ORB_RANK + SURGE_ORB_COUNT),
    GROUP_BAR,
    GROUP_BAR + 1,
    GROUP_SURGE_TIME,
    GROUP_ENEMY_HEADING,
    GROUP_ACTION_MASK,
)


@dataclass(frozen=True)
class SparseFeatures:
    """Sorted feature indices and their values for one public state."""

    indices: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.indices.ndim != 1 or self.values.ndim != 1:
            raise ValueError("Sparse feature indices and values must be one-dimensional.")
        if self.indices.shape != self.values.shape:
            raise ValueError("Sparse feature indices and values must have matching shapes.")


class FeatureEncoder:
    """Encode the documented public state into hashed sparse tile features."""

    def __init__(
        self,
        capacity: int = FEATURE_CAPACITY,
        bins: int = TILE_BINS,
        tilings: int = TILE_TILINGS,
    ) -> None:
        if capacity < 1:
            raise ValueError("Feature capacity must be positive.")
        if bins < 1 or tilings < 1:
            raise ValueError("Tile bins and tilings must be positive.")
        self.capacity = capacity
        self.bins = bins
        self.tilings = tilings

    def encode(self, state: Mapping[str, np.ndarray]) -> SparseFeatures:
        """Tile-code the Gym observation and action mask into binary features.

        Each named group is hashed independently (Sutton tile coding): `tilings`
        overlapping grids of `bins` per dimension, then FNV-1a into `capacity`
        buckets. Colliding hashes collapse to one binary index.
        """
        observation = np.asarray(state["observation"], dtype=np.float64)
        action_mask = np.asarray(state["action_mask"], dtype=np.int8)
        if observation.shape != (OBSERVATION_SIZE,):
            raise ValueError(
                f"Expected observation shape ({OBSERVATION_SIZE},), got {observation.shape}."
            )
        if action_mask.shape != (ACTION_COUNT,):
            raise ValueError(
                f"Expected action mask shape ({ACTION_COUNT},), got {action_mask.shape}."
            )
        if not np.all((action_mask == 0) | (action_mask == 1)):
            raise ValueError("Action mask values must be binary.")
        if not np.any(action_mask):
            raise ValueError("Action mask must contain at least one valid action.")

        features: set[int] = set()
        player = observation[_PLAYER_POSITION]
        self._add_tiles(features, GROUP_PLAYER_POSITION, player, -2.0, 2.0)
        self._add_tiles(
            features, GROUP_PLAYER_VELOCITY, observation[_PLAYER_VELOCITY], -1.0, 1.0
        )
        self._add_tiles(
            features, GROUP_ENEMY_RELATIVE, observation[_ENEMY_POSITION] - player, -2.0, 2.0
        )
        self._add_tiles(features, GROUP_ENEMY_CLOCK, [observation[_ENEMY_CLOCK]], 0.0, 1.0)
        self._add_tiles(
            features, GROUP_SURGE_TIME, [observation[_PLAYER_SURGE], observation[_TIME]], 0.0, 1.0
        )
        self._add_active_direction_features(
            features, GROUP_ENEMY_HEADING, observation[_ENEMY_HEADING]
        )
        self._add_active_direction_features(features, GROUP_ACTION_MASK, action_mask)
        self._add_nearest_collectibles(
            features,
            observation[_PELLETS],
            PELLET_COUNT,
            NEAREST_PELLET_COUNT,
            player,
            GROUP_PELLET_RANK,
        )
        self._add_nearest_collectibles(
            features,
            observation[_ORBS],
            SURGE_ORB_COUNT,
            SURGE_ORB_COUNT,
            player,
            GROUP_ORB_RANK,
        )
        self._add_bar_features(features, observation[_BARS], player)

        indices = np.asarray(sorted(features), dtype=np.int32)
        return SparseFeatures(indices=indices, values=np.ones(indices.shape, dtype=np.float64))

    def _add_tiles(
        self,
        features: set[int],
        group: int,
        values: np.ndarray | list[float],
        lower: float,
        upper: float,
    ) -> None:
        """Activate one overlapping tiling per offset `t / tilings` (Sutton).

        Bin indices stay in `0 … bins-1`. The clipped upper endpoint shares the
        last bin instead of opening a bins-th index.
        """
        clipped = np.clip(np.asarray(values, dtype=np.float64), lower, upper)
        normalized = (clipped - lower) / (upper - lower)
        for tiling in range(self.tilings):
            offset = tiling / self.tilings
            coordinates = tuple(
                int(bin_index)
                for bin_index in np.clip(
                    np.floor(normalized * self.bins + offset).astype(int),
                    0,
                    self.bins - 1,
                )
            )
            features.add(self._hash(group, tiling, *coordinates))

    def _add_active_direction_features(
        self, features: set[int], group: int, values: np.ndarray
    ) -> None:
        for direction, value in enumerate(values):
            if value > 0:
                features.add(self._hash(group, direction))

    def _add_nearest_collectibles(
        self,
        features: set[int],
        encoded: np.ndarray,
        count: int,
        nearest_count: int,
        player: np.ndarray,
        group: int,
    ) -> None:
        collectibles = encoded.reshape(count, 3)
        active = collectibles[collectibles[:, 2] > 0.5, :2]
        ordered = sorted(
            active, key=lambda point: (float(np.dot(point - player, point - player)), *point)
        )
        for rank in range(nearest_count):
            group_id = group + rank
            occupied = rank < len(ordered)
            features.add(self._hash(group_id, OCCUPANCY_SALT, int(occupied)))
            if occupied:
                self._add_tiles(features, group_id, ordered[rank] - player, -2.0, 2.0)

    def _add_bar_features(self, features: set[int], encoded: np.ndarray, player: np.ndarray) -> None:
        bars = encoded.reshape(2, 4)
        canonical = []
        for bar in bars:
            start = bar[:2]
            end = bar[2:]
            if tuple(end) < tuple(start):
                start, end = end, start
            canonical.append((start, end))
        for index, (start, end) in enumerate(
            sorted(canonical, key=lambda bar: (*bar[0], *bar[1]))
        ):
            self._add_tiles(
                features,
                GROUP_BAR + index,
                _closest_point(start, end, player) - player,
                -2.0,
                2.0,
            )

    def _hash(self, *components: int) -> int:
        """Hash integer components with 64-bit FNV-1a and reduce to capacity."""
        value = 0xCBF29CE484222325
        for component in components:
            encoded = int(component).to_bytes(8, "little", signed=True)
            for byte in encoded:
                value ^= byte
                value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return value % self.capacity


def _closest_point(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Return the closest point on one normalized bar segment."""
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared == 0.0:
        return start
    fraction = np.clip(np.dot(point - start, segment) / length_squared, 0.0, 1.0)
    return start + fraction * segment


@dataclass(frozen=True)
class SarsaConfig:
    """Validation-tunable settings for one linear True Online Sarsa run."""

    alpha: float = 0.1
    gamma: float = 0.995
    lambda_: float = 0.90
    epsilon: float = 0.20

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("Alpha must be positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("Gamma must lie in [0, 1].")
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError("Lambda must lie in [0, 1].")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("Epsilon must lie in [0, 1].")


class LinearSarsaAgent(Agent):
    """Sparse-trace linear True Online Sarsa(lambda).

    Action values are `Q(s, a) = w_a · phi(s)` with a hashed tile-coded `phi`.
    Eligibility traces live in a sparse dict keyed by `(action, feature_index)`.
    """

    name = "linear-sarsa"

    def __init__(
        self,
        encoder: FeatureEncoder,
        config: SarsaConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)
        self.encoder = encoder
        self.config = SarsaConfig() if config is None else config
        self.weights = np.zeros((ACTION_COUNT, encoder.capacity), dtype=np.float64)
        self.traces: dict[tuple[int, int], float] = {}
        self.q_old = 0.0

    def reset_episode(self) -> None:
        """Clear eligibility traces and the true-online correction term `q_old`."""
        self.traces.clear()
        self.q_old = 0.0

    def snapshot(self) -> dict:
        return {
            "trace_count": len(self.traces),
            "nonzero_weights": int(np.count_nonzero(self.weights)),
        }

    def q_values(self, state: dict[str, np.ndarray]) -> np.ndarray:
        """Return unmasked linear Q(s, ·) from the Gym observation."""
        return self.q_from_features(self.encoder.encode(state))

    def q_from_features(self, features: SparseFeatures) -> np.ndarray:
        """Return `w[:, phi] @ phi_values` in Direction index order."""
        return self.weights[:, features.indices] @ features.values

    def q_value(self, features: SparseFeatures, action: int) -> float:
        """Return `Q(s, a) = w_a · phi(s)` for one action index."""
        _validate_action(action)
        return float(np.dot(self.weights[action, features.indices], features.values))

    def update(self, transition: Transition) -> float:
        """True Online Sarsa(lambda) on one Gym transition.

        Encodes `state` / `next_state`, then applies `update_features`.
        """
        features = self.encoder.encode(transition.state)
        next_features = (
            None if transition.terminal else self.encoder.encode(transition.next_state)
        )
        return self.update_features(
            features,
            transition.action,
            transition.reward,
            next_features,
            transition.next_action,
            transition.terminal,
        )

    def update_features(
        self,
        features: SparseFeatures,
        action: int,
        reward: float,
        next_features: SparseFeatures | None = None,
        next_action: int | None = None,
        terminal: bool = False,
    ) -> float:
        """True Online Sarsa(lambda) (van Seijen & Sutton) on sparse features:

        q      = w · x(s, a)
        q_next = 0 if terminal else w · x(s', a')
        delta  = r + gamma * q_next - q
        e      = gamma * lambda * e + (1 - alpha * gamma * lambda * (e · x)) * x
        w      = w + alpha * (delta + q - q_old) * e - alpha * (q - q_old) * x
        q_old  = q_next

        `alpha` here is the used step `config.alpha / ||x||^2`, so the effective
        step `α ‖x‖²` equals the configured target. `e · x` uses the trace from
        the previous step, before decay.
        """
        _validate_action(action)
        q = self.q_value(features, action)
        delta, q_next = self._td_error(q, reward, next_features, next_action, terminal)
        step_size = self._step_size(features)
        self._accumulate_true_online_traces(features, action, step_size)
        self._apply_weight_update(features, action, delta, q, step_size)
        self.q_old = q_next
        return delta

    def _td_error(
        self,
        q: float,
        reward: float,
        next_features: SparseFeatures | None,
        next_action: int | None,
        terminal: bool,
    ) -> tuple[float, float]:
        """Return `(delta, q_next)` with `delta = r + gamma * q_next - q`."""
        if terminal:
            q_next = 0.0
        else:
            if next_features is None or next_action is None:
                raise ValueError("Nonterminal Sarsa updates require next features and action.")
            q_next = self.q_value(next_features, next_action)
        return float(reward) + self.config.gamma * q_next - q, q_next

    def _step_size(self, features: SparseFeatures) -> float:
        """Return `config.alpha / ‖x‖²` so the effective step is `config.alpha`."""
        norm_squared = float(np.dot(features.values, features.values))
        return self.config.alpha / max(norm_squared, 1e-12)

    def _accumulate_true_online_traces(
        self, features: SparseFeatures, action: int, step_size: float
    ) -> None:
        """Replace traces: `e <- γλ e + (1 - αγλ (e_{t-1} · x)) x`."""
        trace_dot_x = sum(
            self.traces.get((action, int(index)), 0.0) * value
            for index, value in zip(features.indices, features.values)
        )
        self._decay_traces()
        correction = 1.0 - step_size * self.config.gamma * self.config.lambda_ * trace_dot_x
        for index, value in zip(features.indices, features.values):
            key = (action, int(index))
            self.traces[key] = self.traces.get(key, 0.0) + correction * float(value)

    def _apply_weight_update(
        self,
        features: SparseFeatures,
        action: int,
        delta: float,
        q: float,
        step_size: float,
    ) -> None:
        """`w <- w + α (δ + q - q_old) e - α (q - q_old) x`."""
        trace_scale = step_size * (delta + q - self.q_old)
        for (trace_action, index), trace_value in self.traces.items():
            self.weights[trace_action, index] += trace_scale * trace_value
        current_scale = step_size * (q - self.q_old)
        for index, value in zip(features.indices, features.values):
            self.weights[action, int(index)] -= current_scale * float(value)

    def _decay_traces(self) -> None:
        scale = self.config.gamma * self.config.lambda_
        if scale == 0.0:
            self.traces.clear()
            return
        for key, value in list(self.traces.items()):
            decayed = value * scale
            if abs(decayed) < TRACE_PRUNE_THRESHOLD:
                del self.traces[key]
            else:
                self.traces[key] = decayed


def _validate_action(action: int) -> None:
    if not 0 <= action < ACTION_COUNT:
        raise ValueError(f"Invalid action index: {action}.")
