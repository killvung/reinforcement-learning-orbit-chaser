# Orbit Chase — Project Checkpoint

## Current game and MDP

- Phaser game: continuous circular arena, core obstacle, two generated bars,
  pellets, Surge Orbs, 60-second rounds, and eight directional movement.
- The RL environment is a fully observable MDP with 150 features and eight
  state-masked actions.
- State includes full geometry, player/enemy state, collectibles, timer,
  player-controller mode, and deterministic safe-route/clearance features.
- Reward uses capture/clear/timeout outcomes, safe-path progress, a step cost,
  and a smooth obstacle-clearance cost. History-based stuckness is not used as
  reward because it would break the Markov property.

## Training and evaluation

- Algorithms: True Online Sarsa(lambda), Double DQN, and PPO.
- Benchmarks use disjoint evaluation seed ranges and report capture, timeout,
  return, capture time, barrier-risk exposure, and clearance recovery.
- Navigation cache precomputes static walkability/neighbors per arena and
  reuses Dijkstra fields while the player remains in the same route-grid cell.
- Parallel Sarsa training uses eight worker processes with independent trace
  state per arena and one shared policy weight matrix.

## Model artifacts

- PPO: `rl/train_ppo.py` saves `.pt`; `rl/export_ppo.py` exports actor JSON.
- Sarsa: `rl/train_sarsa.py` trains, saves `.npz`, and exports JSON.
- Sarsa export rejects non-finite weights. An earlier checkpoint diverged to
  NaN and must not be used.
- Phaser automatically loads `public/models/sarsa-route-aware.json`; otherwise
  it falls back to the greedy controller.

## Current limitations

- The included browser Sarsa JSON is only a 2,000-step smoke model. Retrain
  for 300k+ steps before judging gameplay.
- Browser inference uses the `pellet` player-policy proxy for a human player.
  Better real-player behavior requires human-like trajectory training.
- The browser simulation and Python training environment are aligned at the
  feature/controller boundary but remain separate implementations; exact
  simulator parity is future work.
- The coarse route graph is a useful learned feature, not an exact continuous
  planner. Its cached grid edges should eventually use swept collision checks.

## Next recommended action

Train and export a real Sarsa model, then run the held-out benchmark:

```bash
.venv/bin/python rl/train_sarsa.py --steps 300000 --seed 0 --environments 8 --output rl/models/sarsa-route-aware.npz
.venv/bin/python rl/train_sarsa.py --export --output rl/models/sarsa-route-aware.npz --export-output public/models/sarsa-route-aware.json
.venv/bin/python rl/benchmark_sarsa.py --steps 300000 --runs 3 --episodes 300 --environments 8 --output rl/results/sarsa-parallel.csv
```
