# Reinforcement Learning

This directory contains the headless implementation for the yellow **player**
agent. The red enemy remains the fixed deterministic greedy controller in the
browser simulation.

## Environment contract

The Python environment ports `src/game/simulation.ts` exactly:

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
path without hitting the boundary, core, or a bar.


Coordinates are arena-centered and divided by the arena radius. The matching
eight-element action mask uses swept collision checks over the next decision
interval. Collision handling remains authoritative even if an invalid action
is selected.

## First algorithm: True Online Sarsa(lambda)

Sarsa(lambda) is the first learning baseline, not PPO. It will use linear
action values with sparse tile-coded features, an epsilon-greedy policy over
valid actions, and true-online eligibility traces. This makes reward,
feature, action-mask, and trace behavior inspectable before using a neural
network.

Each run records clear, capture, and timeout rates; pellets collected; return;
and time to clear. Training and held-out seed ranges must remain disjoint.