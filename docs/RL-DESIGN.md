# Orbit Chase — Player MDP Design

## Decision

Train one yellow player agent to clear pellets against the fixed greedy red
enemy. This is a fully observable single-agent MDP.

## State and actions

The player acts every 100 ms from eight normalized movement directions. Physics
advances in 10 ms ticks. The state is the 130-feature vector defined in
[`rl/README.md`](../rl/README.md): normalized actor state, enemy direction and
decision-clock fraction, exact bar endpoints, fixed collectible slots with
active flags, and time remaining.

The outer arena and central core are fixed constants. They are represented by
the simulator and do not consume observation dimensions. Endpoint geometry is
used instead of a raster image because it is exact, compact, and can be
identically encoded in TypeScript and Python.

An eight-action mask identifies directions that would collide during the next
player decision interval. The mask aids exploration; swept collision remains
the final authority.

## Rewards

| Event | Reward |
| --- | ---: |
| Clear all pellets | +100 |
| Pellet collected | +3 |
| Surge Orb collected | +10 |
| Captured | -100 |
| Timeout | -30 |
| Per decision | -0.01 |

Terminal outcomes dominate collection rewards. When capture and clear occur on
the same tick, capture takes precedence in the Gym reward.

## Evaluation sequence

1. Maintain deterministic TypeScript/Python parity fixtures for every rule
   transition.
2. Measure random-valid and pellet-seeking players on held-out seeds.
3. Train True Online Sarsa(lambda) with linear tile coding and inspect values,
   traces, action masks, and reward components.
4. Select models on held-out clear rate, with pellet count as a secondary
   metric.
5. Consider masked PPO only after it can be fairly compared with Sarsa(lambda).
