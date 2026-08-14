# Orbit Chase — Project Checkpoint

## Current state

- The Phaser game is playable with a fixed greedy enemy.
- Gameplay rules now advance on a deterministic 10 ms simulation tick.
- The player-agent observation contains 130 fixed-order features plus an
  eight-action swept-path mask.
- The browser no longer attempts to load the obsolete enemy PPO export.

## Next implementation work

1. Add TypeScript regression tests for deterministic game behavior.
2. Port the same contract to a Python headless environment.
3. Benchmark random and pellet-seeking player policies.
4. Implement True Online Sarsa(lambda) with tile coding.
5. Train/evaluate Sarsa(lambda) on held-out seeds before any PPO work.

There are currently no verified trained player models, Python trainers, or
benchmark results in this repository.
