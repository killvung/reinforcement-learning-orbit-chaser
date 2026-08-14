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

## Planned layout

```text
rl/
  orbit_chase_player_env.py  # Headless environment
  tile_coder.py              # Sparse tile features
  train_sarsa.py             # True Online Sarsa(lambda)
  evaluate.py                # Shared held-out evaluation
  tests/                     # Python environment/parity tests
  models/                    # Ignored checkpoints
  results/                   # Ignored experiment data
```

PPO is explicitly deferred until this environment, heuristic baseline, and
Sarsa(lambda) evaluation are reproducible.
