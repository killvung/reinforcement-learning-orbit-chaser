from orbit_chase.geometry import is_blocked, point_segment_distance


def test_segment_distance_projects_onto_segment():
    assert point_segment_distance((5, 3), (0, 0), (10, 0)) == 3


def test_arena_boundary_blocks_player_body():
    assert is_blocked((631, 322), 12, [])
    assert not is_blocked((620, 322), 12, [])


def test_bar_collision_expands_bar_by_actor_radius():
    bar = ((300, 100), (500, 100))
    player_radius = 12

    # Bar half-width (9) + player radius (12) gives a 21 px collision limit.
    assert is_blocked((400, 120.9), player_radius, [bar])
    assert not is_blocked((400, 121.1), player_radius, [bar])
