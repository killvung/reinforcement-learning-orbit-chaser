# Reinforcement Learning

This directory contains the headless implementation for the yellow player
agent. The red enemy stays a fixed deterministic controller.

## Environment contract

The Python environment matches the player-agent rules in
`src/game/simulation.ts`:

- fixed 10 ms physics ticks and seeded arena/pickup generation;
- a 100 ms player decision interval with eight movement actions;
- static circular boundary, core, and two generated bars;
- 32 pellet slots, three Surge Orb slots, capture, clear, and timeout rules;
- player Surge speed multiplier and the greedy enemy's 0.28 s decision clock.

Its 130-element observation is:

| Features | Count |
| --- | ---: |
| Player position, velocity, Surge fraction | 5 |
| Enemy position, heading one-hot, decision-clock fraction | 11 |
| Two bar endpoints | 8 |
| 32 pellet positions and active flags | 96 |
| 3 orb positions and active flags | 9 |
| Time fraction | 1 |
| **Total** | **130** |

### Player observation vector (130 values)

All coordinates use the same arena-relative normalization:
`(coordinate - arenaCenter) / arenaRadius`. Velocities are divided by the
player's base speed. Each reset keeps collectible-slot positions fixed; only
the associated active flag changes after collection.

| Indices | Values | Count |
| --- | --- | ---: |
| 0–1 | Player normalized `x`, `y` | 2 |
| 2–3 | Player velocity `x`, `y` | 2 |
| 4 | Player Surge time remaining / 4 seconds | 1 |
| 5–6 | Enemy normalized `x`, `y` | 2 |
| 7–14 | Enemy heading one-hot: `up`, `up-right`, `right`, `down-right`, `down`, `down-left`, `left`, `up-left` | 8 |
| 15 | Remaining fraction of the enemy's 0.28-second decision interval | 1 |
| 16–19 | First bar: normalized `from.x`, `from.y`, `to.x`, `to.y` | 4 |
| 20–23 | Second bar: normalized `from.x`, `from.y`, `to.x`, `to.y` | 4 |
| 24–119 | Pellets 0–31, each: normalized `x`, `y`, `active` (1 or 0) | 96 |
| 120–128 | Surge Orbs 0–2, each: normalized `x`, `y`, `active` (1 or 0) | 9 |
| 129 | Remaining round time / 60 seconds | 1 |

The eight-element action mask is returned alongside, not inside, this vector.
It uses the same direction order as the enemy-heading one-hot vector. A value
of `true` means the player can complete that direction's next 100 ms swept
path without hitting the boundary, core, or a bar. Fixed-tick collision checks
remain authoritative if a caller selects an invalid action.

## Non-learning baselines

`orbit_chase.policies` contains three fixed policies that use the Gym state and
its action mask only:

- `RandomValidPolicy` samples one valid action from a seed-local NumPy RNG.
- `PelletSeekingPolicy` projects each valid 100 ms move, then prefers the one
  nearest to an active pellet while retaining distance from the enemy.
- `EnemyEvadePolicy` projects each valid 100 ms move and chooses the one that
  ends farthest from the enemy.

Run the held-out suite with:

```bash
PYTHONPATH=rl .venv/bin/python rl/evaluate_baselines.py --episodes 100
```

The command evaluates final-test seeds `10000` through `10099` by default and
reports clear, capture, and timeout rates; average pellets; return; and time
to clear. Time to clear is elapsed simulation time at the clearing tick.

Use seeds `0` through `7999` for training and `8000` through `8999` for
validation. Select hyperparameters and model families on validation only. Run
the final-test range after selection.

### Reference run

Reference values for seeds `10000` through `10099`:

| Policy | Clear rate | Capture rate | Timeout rate | Mean pellets | Mean return | Mean clear time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random valid | 0.00 | 1.00 | 0.00 | 2.11 | -91.6897 | n/a |
| Pellet seeking | 0.10 | 0.68 | 0.22 | 13.62 | -22.3917 | 8.901 s |
| Enemy evade | 0.00 | 0.02 | 0.98 | 1.41 | -32.9302 | n/a |

## Next learning algorithm

Sarsa(lambda) is the first learning baseline, not PPO. It will use linear
action values with sparse tile-coded features, an epsilon-greedy policy over
valid actions, and true-online eligibility traces. This makes reward,
feature, action-mask, and trace behavior inspectable before using a neural
network.

[`sarsa.md`](sarsa.md) defines the linear feature map, true-online update, and
validation gate. [`neural-sarsa.md`](neural-sarsa.md) defines the separate
online Expected-Sarsa fallback. The linear agent comes first, and the neural
agent does not reuse true-online traces.

Each run records clear, capture, and timeout rates; pellets collected; return;
and time to clear. The training, validation, and final-test ranges stay
disjoint.
