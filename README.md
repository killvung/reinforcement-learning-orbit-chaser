# Orbit Chase

Orbit Chase is a Phaser 3 top-down pursuit game. The yellow player wins by
collecting all 32 pellets in 60 seconds; the red enemy wins by touching the
player. A seeded circular arena contains a central core and two **static**
procedural obstacle bars. Three Surge Orbs give the player four seconds of
speed advantage.

The player moves at 175 px/s (1.32x during Surge). The fixed red enemy moves
at 110 px/s (0.78x during player Surge) and chooses a collision-safe greedy
pursuit direction every 0.28 seconds.

## Player-agent objective

The project trains a single RL **player** agent against that fixed greedy
enemy. This is a single-agent MDP: the enemy is deterministic game dynamics,
not a second learning agent.

The primary held-out metric is arena clear rate. A policy that survives until
timeout without clearing pellets is not successful.

## Deterministic simulation contract

`src/game/simulation.ts` owns gameplay rules; Phaser only presents them. It
uses a fixed 10 ms physics tick, so a seed and action sequence produce the
same episode regardless of browser render rate.

The headless training implementation must match this contract:

- `reset(seed)` regenerates arena geometry and all collectible slots.
- `stepFixed(playerInput)` advances exactly one physics tick.
- `step(elapsed, playerInput)` is the browser adapter and preserves leftover
  time between render frames.
- `observe()` returns a versioned 130-feature player observation and an
  eight-action collision mask for the next 100 ms decision interval.

Observation features are normalized arena-relative actor state, enemy heading
and decision-clock fraction, both bar endpoints, 32 pellet slots, three orb
slots, and remaining time. Collected slots keep their position and switch only
their active flag, keeping the feature order fixed.

## RL progression

1. Implement and verify the headless Python environment against fixed seeds.
2. Benchmark random and hand-authored pellet-seeking players.
3. Train **True Online Sarsa(lambda)** with linear tile-coded features. This is
   the first learning agent because its values and eligibility traces are easy
   to inspect scientifically.
4. Evaluate Sarsa(lambda) on held-out seeds before considering PPO.
5. Add masked PPO only if it beats Sarsa(lambda) and the pellet heuristic on
   held-out clear rate and pellets collected.

The initial reward schedule is: clear `+100`, pellet `+3`, Surge Orb `+10`,
capture `-100`, timeout `-30`, and a small per-decision cost/progress shaping
term. Shaping remains small enough that clearing dominates surviving.

## Development

```bash
npm install
npm run dev
npm run build
```

`npm run build` type-checks TypeScript and creates the production browser
bundle. Training scripts and reproducible evaluation commands will live in
`rl/`.
