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

The browser and Python implementations share these player-agent rules:

- seeded resets regenerate arena geometry and collectible slots;
- fixed-step calls advance one 10 ms physics tick;
- the browser adapter preserves render-time remainders;
- observations contain 130 player features and an eight-action mask for the
  next 100 ms decision interval.

The parity suite runs 15 shared fixtures. They cover seeded movement, turns,
the arena boundary, core and bar collisions, Orbs and Surge, pickups, capture,
clear, timeout, and simultaneous terminal events.

## RL progression

1. Run the fixed random-valid and pellet-seeking baselines on held-out seeds.
2. Train **True Online Sarsa(lambda)** with linear tile-coded features. This is
   the first learning agent because its values and eligibility traces are easy
   to inspect scientifically.
3. Evaluate Sarsa(lambda) on held-out seeds before considering PPO.
4. Add masked PPO only if it beats Sarsa(lambda) and the pellet heuristic on
   held-out clear rate and pellets collected.

The reward schedule is: clear `+100`, pellet `+3`, Surge Orb `+10`, capture
`-100`, timeout `-30`, and `-0.01` per decision. Capture takes precedence
when a capture and clear occur on the same tick.

## Development

```bash
npm install
npm run dev
npm run build
```

`npm run build` type-checks TypeScript and creates the production browser
bundle.

Run the held-out non-learning baselines with:

```bash
PYTHONPATH=rl .venv/bin/python rl/evaluate_baselines.py --episodes 100
```
