# Orbit Chase — Project Checkpoint

## Current state

- The Phaser game is playable with a fixed greedy enemy.
- Gameplay rules advance on a deterministic 10 ms simulation tick.
- The player-agent observation contains 130 fixed-order features plus an
  eight-action swept-path mask.
- TypeScript and Python share that contract through 15 parity fixtures.
- Three held-out non-learning player policies live in
  `rl/orbit_chase/policies.py`: random-valid, pellet-seeking, and enemy-evade.
  Reference numbers for seeds `10000` through `10099` are in `rl/README.md`.
- The browser no longer attempts to load the obsolete enemy PPO export.
- This repository has no trained player models yet.

## Next implementation work

1. Implement True Online Sarsa(lambda) with linear tile coding.
2. Train and evaluate Sarsa(lambda) on held-out seeds before any PPO work.
