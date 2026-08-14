import math

from orbit_chase.constants import PHYSICS_DT_SECONDS, PLAYER_SPEED, Direction
from orbit_chase.simulation import GameSimulation


def test_reset_reproduces_the_same_static_episode_state():
    first = GameSimulation(73)
    second = GameSimulation(73)

    assert first.arena == second.arena
    assert first.player == second.player
    assert first.enemy == second.enemy


def test_player_moves_one_fixed_tick_and_updates_velocity():
    simulation = GameSimulation(73)
    initial_x = simulation.player.x

    simulation.move_player(Direction.RIGHT)

    assert simulation.player.x == initial_x + PLAYER_SPEED * PHYSICS_DT_SECONDS
    assert simulation.player_velocity == (PLAYER_SPEED, 0.0)


def test_player_movement_is_normalized_for_diagonals():
    simulation = GameSimulation(73)
    initial = (simulation.player.x, simulation.player.y)

    simulation.move_player(Direction.UP_RIGHT)

    moved = math.hypot(
        simulation.player.x - initial[0], simulation.player.y - initial[1]
    )
    assert math.isclose(moved, PLAYER_SPEED * PHYSICS_DT_SECONDS)
