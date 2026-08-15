# Linear Sarsa handoff (2026-08-15)

Read this before continuing player-agent training. Design lives in
[`sarsa.md`](sarsa.md). Do not use final-test seeds `10000–10099` for
tuning. Do not start neural Expected Sarsa until linear fails the
validation gate on **greedy clear rate**.

## Goal

Learn a policy that **clears**: all 32 pellets collected, not captured,
before the 60 s timeout. That is pellet collection **and** enemy evasion,
not either one.

Validation gate ([`sarsa.md`](sarsa.md)): promote only when greedy
clear-rate improvement over pellet-seeking (0.10 clear, 13.62 pellets,
return −22.3917 on seeds `10000–10099`) has a positive bootstrap CI.
Pellets and return are secondary.

## Seed protocol

| Split | Seeds | Use |
| --- | --- | --- |
| Training | `0–7999` | Weight updates |
| Validation | `8000–8999` | Greedy eval, hyperparameter choice |
| Final test | `10000–10099` | One comparison after selection |

Agent seed for these runs: `73`.

## Audit fixes (do not reuse old checkpoints)

The λ=0.90 abort and λ=0.50 800-episode diagnostic used a step size, greedy
tie-break, epsilon schedule, and action mask that the 2026-08-15 audit
rejected. Those weights are not a prefix of the next run. Retrain from
scratch.

Fixes now in the code:

- Effective α is `0.1 / ‖x‖²` (about 120–128 active binary features), not
  `0.1 / 8`.
- Training randomizes among tied greedy actions; evaluation still uses the
  lowest valid index.
- Epsilon decays 0.20 → 0.02 over a fixed 8000-episode horizon, so an
  800-episode diagnostic is a true prefix (ε ≈ 0.182 at episode 800).
- The action mask dry-runs ten 10 ms ticks, including Surge expiry and Orb
  pickups.

Pellet-rank encoding (four nearest) and γ=0.995 are unchanged.

## What already ran (obsolete)

### λ = 0.90, aborted (~1600 / 8000)

- Command used `--episodes 8000`, so ε was still ~0.16 at episode 1600.
- 0 clears, mean pellets stuck ~3.1, capture ~0.99, cumulative |δ| rose
  7.77 → 9.21 after episode 200.
- Halted. Weights were not saved (checkpoints only existed at end then).
- Log: `rl/models/linear-sarsa-train.jsonl` (partial).

### λ = 0.50, 800-episode diagnostic (obsolete)

`--episodes 800` decayed ε from 0.20 to 0.02 across 800 episodes. That
schedule bug is fixed; do not compare new diagnostics to this run.

Training (exploring): 1 clear, 790 captures, 9 timeouts, mean pellets
6.485, return −71.114, |δ| 8.78 → 9.12. Greedy eval on seeds `8000–8049`:
0 clears, 12.3 pellets. Linear Sarsa found pellet-seeking, not survival.
Judge the next run on **greedy validation clears**.

## Next experiment

Train λ = 0.5 for 8000 episodes on seeds `0–7999`, from scratch, with the
new α default and the fixed ε horizon. Checkpoint every 100 (default). An
optional `--episodes 800` diagnostic now shares the 8000-episode epsilon
schedule. Do not treat pellets ~3 in the first ~3000 episodes as failure.

```bash
PYTHONPATH=rl .venv/bin/python rl/train_sarsa.py --episodes 8000 --seed-start 0 \
  --agent-seed 73 --lambda 0.5 --log rl/models/linear-sarsa-lambda05-8000.jsonl
```

Then greedy-eval the latest `.npz` on validation, not final-test:

```bash
PYTHONPATH=rl .venv/bin/python rl/evaluate_baselines.py \
  --checkpoint rl/models/linear-sarsa-TIMESTAMP.npz \
  --start-seed 8000 --episodes 100
```

Watch `|δ|`. Do not change the reward (timeout is not success; that
would become enemy-evade). Do not jump to neural unless this 8000-episode
greedy eval still sits near 12 pellets and ~0 clears.

## Code already on `main`

- `rl/orbit_chase/checkpoint.py` — timestamped `.npz` save/load
- `train_sarsa.py` — `--lambda`, `--alpha` (effective step, default 0.1),
  `--epsilon-horizon-episodes` (default 8000), `--checkpoint-every` (default 100)
- Rolling stderr is timestamped and aligned; `clears=N` is on the **final**
  stdout line, not the `clear 0.00` rate
- `evaluate_baselines.py --checkpoint` skips heuristics unless `--policy` is set
- `npm run test:python` runs `rl/tests`
- `rl/models/` is gitignored
