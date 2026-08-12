# Orbit Chase

Orbit Chase is a Phaser 3 top-down pursuit game. The yellow player starts in a
circular arena containing 32 pellets and three power orbs. The player wins by
collecting every pellet before the 60-second timer expires. The red enemy wins
by touching the player. Two rotating obstacle bars and the arena boundary block
both actors.

The player moves at 175 px/s (1.32x while surge is active); the enemy moves at
110 px/s (0.78x while surge is active). Power orbs grant the player four
seconds of surge. The built-in enemy is a deterministic greedy pursuer: every
0.28 seconds it chooses the collision-safe direction that most reduces its
straight-line distance to the player.

## Train the player agent

The goal is a single player PPO agent that **clears pellets against the fixed
greedy enemy**. With the enemy policy fixed, the game is a standard
single-agent MDP rather than a multi-agent learning problem.

### MDP

| Component | Definition |
| --- | --- |
| Agent | Yellow player |
| Actions | Eight movement directions; held for a short fixed decision interval |
| Environment | Procedural arena seed, pellets, orbs, surge state, timer, and deterministic greedy red enemy |
| Observation | Normalized player/enemy positions and velocities, surge, enemy heading, bar endpoints, pellet/orb locations and active flags, and remaining time |
| Terminal states | Player clears all pellets (win), enemy captures player (loss), or 60 seconds elapse (loss) |
| Primary metric | Held-out arena clear rate |

### Reward design

The reward must make clearing the arena more valuable than merely surviving:

| Event | Initial reward |
| --- | ---: |
| Clear all pellets | +100 |
| Pellet collected | +3 |
| Power orb collected | +10 |
| Capture | -100 |
| Timeout | -30 |
| Each decision | -0.01 |
| Movement toward nearest active pellet | Small, bounded shaping reward |

Capture avoidance is represented by the terminal penalty. We will not give a
positive timeout reward: otherwise the player learns to hide rather than win.
The shaping term is only an aid to exploration and must remain much smaller
than the win/loss rewards.

## Technical plan

### Stack

- **Game:** TypeScript, Phaser 3, Vite (existing browser implementation).
- **Training:** Python, Gymnasium-style environment, NumPy, and PyTorch.
- **Algorithm:** masked discrete-action PPO actor--critic. PyTorch keeps the
  model and browser-export format under project control without adding a large
  RL framework dependency.
- **Evaluation and artifacts:** JSON/CSV metrics, PyTorch checkpoints, and a
  dependency-free JSON actor export for browser inference.

### Stages

1. **Rebuild a faithful headless environment.** Port the current game rules to
   Python with seeded procedural arenas, swept-path collision checks, player
   collection/surge, and exactly the same greedy enemy decision rule. Add unit
   tests for geometry, collection, terminal conditions, and deterministic
   resets.
2. **Establish non-learning baselines.** Measure a random player and a simple
   hand-authored pellet-seeking player over held-out seeds. These are the floor
   for learning agents and validate the evaluation harness.
3. **Train True Online Sarsa(lambda).** Use linear action-value approximation
   with tile coding as the first learning baseline. It should learn local
   pellet-seeking, immediate danger avoidance, and orb collection; it is
   unlikely to be the final policy because the procedural, continuous arena
   needs nonlinear long-horizon strategy. Its value is fast iteration and
   transparent debugging of rewards, masks, terminal states, and parity.
4. **Train player PPO.** Use parallel seeded environments, action masking for
   directions that immediately hit walls, periodic checkpoints, verbose
   progress, and a CSV log. Keep training and held-out seeds disjoint.
5. **Evaluate for the actual objective.** Report clear rate first, then capture
   rate, timeout rate, average pellets collected, average return, and median
   time to clear. Evaluate every checkpoint on the same held-out seed suite and
   retain the best clear-rate checkpoint rather than blindly exporting the
   final one. PPO must beat the Sarsa(lambda) baseline on held-out clear rate
   and pellets collected.
6. **Validate browser parity.** Export the selected player actor, run it in the
   existing TypeScript simulation while retaining the built-in greedy enemy,
   and compare a fixed set of seeds with the Python evaluator. Instrument any
   mismatch before further training.

## Acceptance criteria

The first usable player agent must beat the hand-authored pellet baseline on a
held-out seed suite, clear a meaningful fraction of arenas, and preserve the
same outcome on fixed browser-versus-Python parity seeds. We will not claim
success from survival/timeouts alone.
