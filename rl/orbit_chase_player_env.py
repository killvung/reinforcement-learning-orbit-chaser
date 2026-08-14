"""Gymnasium environment for the Orbit Chase player agent.

This is a deterministic, headless port of the TypeScript game contract.  One
Gym step holds a player action for 100 ms (ten 10 ms physics ticks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTIONS = (
    "up",
    "up-right",
    "right",
    "down-right",
    "down",
    "down-left",
    "left",
    "up-left",
)
VECTORS = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
DT, DECISION_TICKS, ENEMY_INTERVAL = 0.01, 10, 0.28


def mulberry32(seed: int):
    value = seed & 0xFFFFFFFF

    def random() -> float:
        nonlocal value
        value = (value + 0x6D2B79F5) & 0xFFFFFFFF
        result = value
        result = ((result ^ (result >> 15)) * (result | 1)) & 0xFFFFFFFF
        result ^= (
            result + (((result ^ (result >> 7)) * (result | 61)) & 0xFFFFFFFF)
        ) & 0xFFFFFFFF
        return ((result ^ (result >> 14)) & 0xFFFFFFFF) / 4294967296

    return random


@dataclass
class Actor:
    x: float
    y: float
    radius: float
    speed: float


class OrbitChasePlayerEnv(gym.Env[dict[str, np.ndarray], int]):
    """Player-clearance MDP; the red enemy is a fixed greedy controller."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, (130,), np.float32),
                "action_mask": spaces.MultiBinary(8),
            }
        )
        self.center = (400.0, 322.0)
        self.radius, self.core_radius = 242.0, 36.0
        self.reset(seed=0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        value = int(self.np_random.integers(0, 2**32)) if seed is None else seed
        random = mulberry32(value)
        angle = random() * math.tau
        unit, perpendicular = (math.cos(angle), math.sin(angle)), (
            -math.sin(angle),
            math.cos(angle),
        )
        self.bars = []
        for sign in (-1, 1):
            offset, length = 92 * sign, 112 + random() * 38
            self.bars.append(
                (
                    (
                        400 + unit[0] * offset - perpendicular[0] * length / 2,
                        322 + unit[1] * offset - perpendicular[1] * length / 2,
                    ),
                    (
                        400 + unit[0] * offset + perpendicular[0] * length / 2,
                        322 + unit[1] * offset + perpendicular[1] * length / 2,
                    ),
                )
            )
        self.player, self.enemy = Actor(235, 407, 12, 175), Actor(565, 237, 13, 110)
        self.velocity, self.enemy_action, self.enemy_remaining = (0.0, 0.0), 6, 0.0
        self.time_remaining, self.surge_remaining = 60.0, 0.0
        spawn_random = mulberry32(value ^ 0xA5A5A5A5)
        self.pellets, self.orbs = [], []
        while len(self.pellets) < 32:
            point = self._spawn(spawn_random, 52, 158)
            if (
                not self._blocked(*point, 8)
                and self._distance(point, (235, 407)) > 32
                and self._distance(point, (565, 237)) > 32
            ):
                self.pellets.append([*point, True])
        while len(self.orbs) < 3:
            point = self._spawn(spawn_random, 65, 125)
            if not self._blocked(*point, 13) and all(
                self._distance(point, pellet) > 34 for pellet in self.pellets
            ):
                self.orbs.append([*point, True])
        return self._observation(), {"seed": value}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        reward, collected, orb = -0.01, 0, 0
        for _ in range(DECISION_TICKS):
            surge = self.surge_remaining > 0
            self._move(self.player, action, 1.32 if surge else 1)
            vx, vy = VECTORS[action]
            norm = math.hypot(vx, vy)
            self.velocity = (
                vx / norm * self.player.speed * (1.32 if surge else 1),
                vy / norm * self.player.speed * (1.32 if surge else 1),
            )
            if self.enemy_remaining <= 0:
                self.enemy_action = self._greedy_enemy()
                self.enemy_remaining = ENEMY_INTERVAL
            self._move(self.enemy, self.enemy_action, 0.78 if surge else 1)
            self.enemy_remaining -= DT
            self.surge_remaining = max(0, self.surge_remaining - DT)
            self.time_remaining -= DT
            for slot in self.pellets:
                if (
                    slot[2]
                    and self._distance((self.player.x, self.player.y), slot) <= 18
                ):
                    slot[2] = False
                    collected += 1
            for slot in self.orbs:
                if (
                    slot[2]
                    and self._distance((self.player.x, self.player.y), slot) <= 22
                ):
                    slot[2] = False
                    orb += 1
                    self.surge_remaining = 4
            captured = (
                self._distance(
                    (self.player.x, self.player.y), (self.enemy.x, self.enemy.y)
                )
                < 25
            )
            cleared, timeout = (
                not any(slot[2] for slot in self.pellets),
                self.time_remaining <= 0,
            )
            if captured or cleared or timeout:
                break
        reward += collected * 3 + orb * 10
        if cleared:
            reward += 100
        elif captured:
            reward -= 100
        elif timeout:
            reward -= 30
        return (
            self._observation(),
            reward,
            captured or cleared,
            timeout and not (captured or cleared),
            {"captured": captured, "cleared": cleared, "pellets_collected": collected},
        )

    def _observation(self):
        norm = lambda x, y: ((x - 400) / 242, (y - 322) / 242)
        values = [
            *norm(self.player.x, self.player.y),
            self.velocity[0] / 175,
            self.velocity[1] / 175,
            self.surge_remaining / 4,
            *norm(self.enemy.x, self.enemy.y),
            *[float(i == self.enemy_action) for i in range(8)],
            max(0, self.enemy_remaining) / ENEMY_INTERVAL,
        ]
        for start, end in self.bars:
            values.extend((*norm(*start), *norm(*end)))
        for x, y, active in self.pellets + self.orbs:
            values.extend((*norm(x, y), float(active)))
        values.append(max(0, self.time_remaining) / 60)
        return {
            "observation": np.asarray(values, np.float32),
            "action_mask": np.asarray(self._mask(), np.int8),
        }

    def _mask(self):
        return [
            self._can_travel(
                self.player, action, 0.1, 1.32 if self.surge_remaining > 0 else 1
            )
            for action in range(8)
        ]

    def _greedy_enemy(self):
        candidates = [
            a for a in range(8) if self._can_travel(self.enemy, a, ENEMY_INTERVAL, 1)
        ]
        return min(
            candidates or [self.enemy_action],
            key=lambda a: self._distance(
                (self.enemy.x + VECTORS[a][0] * 42, self.enemy.y + VECTORS[a][1] * 42),
                (self.player.x, self.player.y),
            ),
        )

    def _move(self, actor, action, multiplier):
        if not self._can_travel(actor, action, DT, multiplier):
            return
        x, y = VECTORS[action]
        length = math.hypot(x, y)
        actor.x += x / length * actor.speed * multiplier * DT
        actor.y += y / length * actor.speed * multiplier * DT

    def _can_travel(self, actor, action, duration, multiplier):
        x, y = VECTORS[action]
        length, distance = math.hypot(x, y), actor.speed * multiplier * duration
        for index in range(1, max(1, math.ceil(distance / 4)) + 1):
            if self._blocked(
                actor.x
                + x / length * distance * index / max(1, math.ceil(distance / 4)),
                actor.y
                + y / length * distance * index / max(1, math.ceil(distance / 4)),
                actor.radius,
            ):
                return False
        return True

    def _blocked(self, x, y, body):
        if (
            self._distance((x, y), self.center) > self.radius - body
            or self._distance((x, y), self.center) < self.core_radius + body
        ):
            return True
        return any(
            self._segment_distance((x, y), start, end) < 9 + body
            for start, end in self.bars
        )

    @staticmethod
    def _distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _segment_distance(point, start, end):
        dx, dy = end[0] - start[0], end[1] - start[1]
        denom = dx * dx + dy * dy
        t = (
            0
            if denom == 0
            else max(
                0,
                min(
                    1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denom
                ),
            )
        )
        return math.hypot(
            point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy)
        )

    @staticmethod
    def _spawn(random, low, spread):
        angle, distance = random() * math.tau, low + random() * spread
        return 400 + math.cos(angle) * distance, 322 + math.sin(angle) * distance
