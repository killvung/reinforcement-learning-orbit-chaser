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

## What already ran

### λ = 0.90, aborted (~1600 / 8000)

- Command used `--episodes 8000`, so ε was still ~0.16 at episode 1600.
- 0 clears, mean pellets stuck ~3.1, capture ~0.99, cumulative |δ| rose
  7.77 → 9.21 after episode 200.
- Halted. Weights were not saved (checkpoints only existed at end then).
- Log: `rl/models/linear-sarsa-train.jsonl` (partial).

### λ = 0.50, 800-episode diagnostic (done)

**This run is not a prefix of an 8000-episode run.** `--episodes 800`
decayed ε from 0.20 to 0.02 across 800 episodes. On the 8000-episode
schedule, episode 800 is still ε ≈ 0.18. Pellet learning in this
diagnostic started after ε dropped below ~0.13 (would be ~episode 4000
of 8000).

```
PYTHONPATH=rl .venv/bin/python rl/train_sarsa.py --episodes 800 --seed-start 0 \
  --agent-seed 73 --lambda 0.5 --log rl/models/linear-sarsa-lambda05-train.jsonl
```

Training (exploring): 1 clear, 790 captures, 9 timeouts, mean pellets
6.485, return −71.114, |δ| 8.78 → 9.12. Last 100 episodes: 11.65 pellets.
The one clear is **seed 774** (32 pellets, return 222.16). Rolling stderr
printed `clear 0.00` because 1/800 rounds to two decimals.

Checkpoint: `rl/models/linear-sarsa-20260814-234835.npz`
(plus `…-ep0100.npz` … `…-ep0800.npz`).

Greedy eval, validation seeds `8000–8049`:

```
PYTHONPATH=rl .venv/bin/python rl/evaluate_baselines.py \
  --checkpoint rl/models/linear-sarsa-20260814-234835.npz \
  --start-seed 8000 --episodes 50
```

Result (`rl/models/eval-linear-sarsa-20260814-235347.json`): 0 clears,
12.3 pellets, return −42.61, capture 0.96, timeout 0.04.

Interpretation: linear Sarsa found pellet-seeking, not survival. That is
the dense +3 pellet reward, not a failed algorithm. Pellet-seeking
itself is 0.68 capture / 0.10 clear. Keep linear; judge the next run on
**greedy validation clears**.

## Next experiment

Train λ = 0.5 for 8000 episodes on seeds `0–7999` (ε 0.20 → 0.02 over
8000, not 800). There is no resume-from-checkpoint yet, so this is from
scratch. Checkpoint every 100 (default). Do not treat pellets ~3 in the
first ~3000 episodes as failure; that prefix should look like the λ=0.90
stall until ε falls.

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
- `train_sarsa.py` — `--lambda`, `--alpha`, `--checkpoint-every` (default 100)
- Rolling stderr is timestamped and aligned; `clears=N` is on the **final**
  stdout line, not the `clear 0.00` rate
- `evaluate_baselines.py --checkpoint` skips heuristics unless `--policy` is set
- `rl/models/` is gitignored

Commits: `6c762fa` checkpoints; `ee65871` mid-run saves and λ CLI.
