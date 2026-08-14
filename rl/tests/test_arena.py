from orbit_chase.arena import make_arena


def test_seed_reproduces_static_arena_and_collectible_slots():
    assert make_arena(73) == make_arena(73)


def test_different_seeds_change_arena_geometry():
    assert make_arena(73).bars != make_arena(74).bars
