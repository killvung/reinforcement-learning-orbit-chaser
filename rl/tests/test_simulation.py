import math

from orbit_chase.arena import Arena
from orbit_chase.rules import (
    ARENA_CENTER,
    ARENA_RADIUS,
    CORE_RADIUS,
    PHYSICS_DT_SECONDS,
    PLAYER_SPEED,
    SURGE_DURATION_SECONDS,
    Direction,
)
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


def test_enemy_greedy_action_reduces_distance_to_player():
    simulation = GameSimulation(73)
    before = math.dist(
        (simulation.enemy.x, simulation.enemy.y),
        (simulation.player.x, simulation.player.y),
    )

    simulation.step(Direction.RIGHT)

    after = math.dist(
        (simulation.enemy.x, simulation.enemy.y),
        (simulation.player.x, simulation.player.y),
    )
    assert after < before


def test_enemy_action_is_held_until_its_decision_clock_expires():
    simulation = GameSimulation(73)
    decisions = 0

    def choose_action():
        nonlocal decisions
        decisions += 1
        return Direction.LEFT

    simulation._choose_enemy_action = choose_action
    simulation.step(Direction.RIGHT)
    assert decisions == 1

    while simulation.enemy_remaining > 0:
        simulation.step(Direction.RIGHT)
        assert decisions == 1

    simulation.step(Direction.RIGHT)
    assert decisions == 2


def test_player_surge_slows_enemy_movement():
    normal = GameSimulation(73)
    surge = GameSimulation(73)
    for simulation in (normal, surge):
        simulation.enemy.x, simulation.enemy.y = 400.0, 550.0
        simulation.enemy_action = Direction.UP
        simulation.enemy_remaining = 1.0

    surge.surge_remaining = 2.0
    normal.step(Direction.RIGHT)
    surge.step(Direction.RIGHT)

    assert surge.enemy.y > normal.enemy.y


def test_pellet_collection_deactivates_slot_and_reports_event():
    simulation = GameSimulation(73)
    simulation.pellet_active = [False] * len(simulation.pellet_active)
    simulation.pellet_active[0] = True
    simulation.player.x, simulation.player.y = simulation.arena.pellet_slots[0].position

    events = simulation.step(Direction.UP)

    assert events.pellets_collected == 1
    assert simulation.pellet_active[0] is False
    assert simulation.step(Direction.UP).pellets_collected == 0


def test_orb_collection_starts_surge_and_deactivates_slot():
    simulation = GameSimulation(73)
    simulation.orb_active = [False] * len(simulation.orb_active)
    simulation.orb_active[0] = True
    simulation.player.x, simulation.player.y = simulation.arena.orb_slots[0].position

    events = simulation.step(Direction.UP)

    assert events.orbs_collected == 1
    assert simulation.orb_active[0] is False
    assert simulation.surge_remaining == SURGE_DURATION_SECONDS


def test_final_surge_tick_expires_before_player_movement():
    simulation = GameSimulation(73)
    simulation.surge_remaining = PHYSICS_DT_SECONDS
    initial_x = simulation.player.x

    simulation.step(Direction.RIGHT)

    assert simulation.surge_remaining == 0.0
    assert simulation.player.x == initial_x + PLAYER_SPEED * PHYSICS_DT_SECONDS


def test_overlapping_actors_report_capture():
    simulation = GameSimulation(73)
    simulation.player.x, simulation.player.y = simulation.enemy.x, simulation.enemy.y

    assert simulation.step(Direction.UP).captured is True


def test_collecting_final_pellet_reports_clear():
    simulation = GameSimulation(73)
    simulation.pellet_active = [False] * len(simulation.pellet_active)

    assert simulation.step(Direction.UP).cleared is True


def test_timer_crossing_zero_reports_timeout():
    simulation = GameSimulation(73)
    simulation.time_remaining = PHYSICS_DT_SECONDS / 2

    assert simulation.step(Direction.UP).timed_out is True


def test_movement_into_outer_boundary_keeps_player_in_place():
    simulation = GameSimulation(73)
    simulation.player.x = ARENA_CENTER[0] + ARENA_RADIUS - simulation.player.radius - 0.1
    simulation.player.y = ARENA_CENTER[1]
    initial = (simulation.player.x, simulation.player.y)

    simulation.move_player(Direction.RIGHT)

    assert (simulation.player.x, simulation.player.y) == initial


def test_movement_into_core_keeps_player_in_place():
    simulation = GameSimulation(73)
    simulation.player.x = ARENA_CENTER[0] + CORE_RADIUS + simulation.player.radius + 0.1
    simulation.player.y = ARENA_CENTER[1]
    initial = (simulation.player.x, simulation.player.y)

    simulation.move_player(Direction.LEFT)

    assert (simulation.player.x, simulation.player.y) == initial


def test_movement_into_bar_keeps_player_in_place():
    simulation = GameSimulation(73)
    simulation.arena = Arena(
        bars=(((300.0, 100.0), (500.0, 100.0)),) * 2,
        pellet_slots=simulation.arena.pellet_slots,
        orb_slots=simulation.arena.orb_slots,
    )
    simulation.player.x, simulation.player.y = 400.0, 121.1
    initial = (simulation.player.x, simulation.player.y)

    simulation.move_player(Direction.UP)

    assert (simulation.player.x, simulation.player.y) == initial


def test_valid_actions_mask_blocks_boundary_core_and_bar_paths():
    simulation = GameSimulation(73)
    simulation.player.x = ARENA_CENTER[0] + ARENA_RADIUS - simulation.player.radius - 0.1
    simulation.player.y = ARENA_CENTER[1]
    assert simulation.valid_actions()[Direction.RIGHT] is False

    simulation.player.x = ARENA_CENTER[0] + CORE_RADIUS + simulation.player.radius + 0.1
    assert simulation.valid_actions()[Direction.LEFT] is False

    simulation.arena = Arena(
        bars=(((300.0, 100.0), (500.0, 100.0)),) * 2,
        pellet_slots=simulation.arena.pellet_slots,
        orb_slots=simulation.arena.orb_slots,
    )
    simulation.player.x, simulation.player.y = 400.0, 121.1
    assert simulation.valid_actions()[Direction.UP] is False


def test_surge_speed_can_turn_a_safe_path_into_a_masked_path():
    simulation = GameSimulation(73)
    simulation.player.x = ARENA_CENTER[0] + ARENA_RADIUS - simulation.player.radius - 20.0
    simulation.player.y = ARENA_CENTER[1]

    assert simulation.valid_actions()[Direction.RIGHT] is True
    simulation.surge_remaining = 1.0
    assert simulation.valid_actions()[Direction.RIGHT] is False
