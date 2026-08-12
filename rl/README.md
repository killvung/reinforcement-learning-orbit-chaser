# Reinforcement Learning

This directory will contain the headless training environment, agents,
evaluators, checkpoints, and experiment outputs for the Orbit Chase **player
agent**. The red enemy remains the deterministic greedy controller from the
game.

## Training objective

Train the yellow player to clear all 32 pellets within 60 seconds while
avoiding capture. The primary held-out metric is **arena clear rate**. A policy
that only survives until timeout is not considered successful.

## Environment contract

The Python environment must reproduce `src/game/simulation.ts`:

- Circular arena, central core, and two seeded rotating obstacle bars.
- Player start, enemy start, 32 seeded pellets, and three seeded power orbs.
- Player speed 175 px/s; enemy speed 110 px/s.
- A four-second player surge after an orb: player 1.32x speed, enemy 0.78x.
- Player collection radii, collision radii, 60-second timer, and swept-path
  collision handling.
- Enemy decisions every 0.28 seconds using the fixed greedy pursuit rule.
- Deterministic reset for a supplied seed.

The player takes one of eight movement directions per decision. The
environment should mask directions whose short swept path immediately hits a
wall; collision handling remains authoritative during movement.

## Observation

Use a normalized fixed-size numerical observation including:

- Player position and velocity.
- Enemy position and current heading.
- Surge state and remaining round time.
- Both obstacle-bar endpoints.
- Every pellet and orb position plus active flag.

The encoder and its feature order are versioned. The browser export must use
the identical encoder before an agent is deployed.

## Reward

Initial reward schedule:

| Event | Reward |
| --- | ---: |
| Clear all pellets | +100 |
| Pellet collected | +3 |
| Power orb collected | +10 |
| Captured | -100 |
| Timeout | -30 |
| Per decision | -0.01 |
| Progress toward nearest active pellet | Small bounded shaping term |

Do not add a positive timeout reward. Review reward breakdowns alongside
episode outcomes to make sure the agent is learning to clear, not hide.

## Algorithms

### 1. True Online Sarsa(lambda): first learning baseline

Use True Online Sarsa(lambda) with **linear action-value approximation and
tile coding**. A tabular lookup is not viable because positions, obstacle
angles, and generated collectibles produce a continuous, procedural state
space.

This baseline should be quick to train and inspect. Expected behavior is basic
pellet progress and local evasive habits. It may struggle with obstacle detours
and long-horizon strategy, so it is a validation baseline rather than the
expected final controller.

### 2. Masked discrete-action PPO: candidate final agent

Use an actor--critic with eight masked discrete actions, parallel headless
environments, and PyTorch. PPO should be trained only after Sarsa(lambda) has
validated the environment and reward. It must outperform Sarsa(lambda) on the
same held-out seeds before selection.

## Evaluation protocol

- Reserve a fixed held-out seed range; never train on it.
- Run random, hand-authored, Sarsa(lambda), and PPO agents on exactly the same
  held-out seeds.
- Report clear, capture, and timeout rates; mean pellets collected; mean return;
  and median clear time for successful episodes.
- Save a CSV row per checkpoint and select by clear rate, using capture rate as
  a tie-breaker.
- Add fixed-seed Python/browser parity tests before exporting a policy.

## Planned layout

```text
rl/
  orbit_chase_player_env.py  # Gymnasium-style player environment
  tile_coder.py              # Sparse tile features for Sarsa(lambda)
  train_sarsa.py             # True Online Sarsa(lambda) baseline
  train_ppo.py               # PPO player training
  evaluate.py                # Common held-out evaluator
  export_player.py           # Browser actor export
  tests/                     # Environment and parity tests
  models/                    # Checkpoints (ignored by Git)
  results/                   # CSV/JSON/log artifacts (ignored by Git)
```

