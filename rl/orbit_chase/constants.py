"""Shared, versioned Orbit Chase rules."""

from enum import IntEnum
from typing import Final


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class Direction(IntEnum):
    UP = 0
    UP_RIGHT = 1
    RIGHT = 2
    DOWN_RIGHT = 3
    DOWN = 4
    DOWN_LEFT = 5
    LEFT = 6
    UP_LEFT = 7


DIRECTION_VECTORS: Final[dict[Direction, tuple[int, int]]] = {
    Direction.UP: (0, -1),
    Direction.UP_RIGHT: (1, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN_RIGHT: (1, 1),
    Direction.DOWN: (0, 1),
    Direction.DOWN_LEFT: (-1, 1),
    Direction.LEFT: (-1, 0),
    Direction.UP_LEFT: (-1, -1),
}


# ---------------------------------------------------------------------------
# Arena geometry
# ---------------------------------------------------------------------------

ARENA_CENTER: Final = (400.0, 322.0)
ARENA_RADIUS: Final = 242.0
CORE_RADIUS: Final = 36.0
BAR_WIDTH: Final = 18.0
BAR_OFFSET: Final = 92.0
BAR_MIN_LENGTH: Final = 112.0
BAR_LENGTH_VARIATION: Final = 38.0


# ---------------------------------------------------------------------------
# Actor movement and collision
# ---------------------------------------------------------------------------

PLAYER_SPEED: Final = 175.0
PLAYER_RADIUS: Final = 12.0
PLAYER_START: Final = (235.0, 407.0)
ENEMY_SPEED: Final = 110.0
ENEMY_RADIUS: Final = 13.0
ENEMY_START: Final = (565.0, 237.0)
SURGE_DURATION_SECONDS: Final = 4.0
PLAYER_SURGE_SPEED_MULTIPLIER: Final = 1.32
ENEMY_SURGE_SPEED_MULTIPLIER: Final = 0.78


# ---------------------------------------------------------------------------
# Simulation timing
# ---------------------------------------------------------------------------

PHYSICS_DT_SECONDS: Final = 0.01
PLAYER_DECISION_SECONDS: Final = 0.10
ENEMY_DECISION_SECONDS: Final = 0.28
ROUND_DURATION_SECONDS: Final = 60.0
PELLET_COLLECTION_RADIUS: Final = 18.0
ORB_COLLECTION_RADIUS: Final = 22.0


# ---------------------------------------------------------------------------
# Environment interface
# ---------------------------------------------------------------------------

PELLET_COUNT: Final = 32
SURGE_ORB_COUNT: Final = 3
COLLECTIBLE_SEED_SALT: Final = 0xA5A5A5A5
PELLET_SPAWN_MIN_RADIUS: Final = 52.0
PELLET_SPAWN_RADIUS_VARIATION: Final = 158.0
ORB_SPAWN_MIN_RADIUS: Final = 65.0
ORB_SPAWN_RADIUS_VARIATION: Final = 125.0
PELLET_SPAWN_BODY_RADIUS: Final = 8.0
ORB_SPAWN_BODY_RADIUS: Final = 13.0
ACTOR_SPAWN_CLEARANCE: Final = 32.0
ORB_TO_PELLET_CLEARANCE: Final = 34.0
OBSERVATION_SIZE: Final = 130
ACTION_COUNT: Final = len(Direction)
