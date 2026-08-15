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
- Linear True Online Sarsa(lambda) is trained. Checkpoint
  `rl/models/linear-sarsa-20260815-030554-ep8000.npz` clears about 90% of
  10,000 greedy eval seeds `10000–19999` (the 100-seed slice `10000–10099`
  was 96/100).
- The browser loads the matching sparse JSON player artifact from
  `public/models/linear-sarsa-20260815-030554-ep8000.json` and can switch
  between Human and Trained Player. Decisions stay on the 100 ms Gym cadence.

## Next implementation work

1. Showcase the trained player in the Phaser game.
2. Consider further training or a neural fallback only if linear Sarsa no
   longer meets the held-out clear-rate bar.
