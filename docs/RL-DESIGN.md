# Orbit Chase — Fully Observable MDP Design

## Decision

Orbit Chase uses a **fully observable Markov Decision Process (MDP)**. The player can see the arena and enemy; correspondingly, the RL enemy sees the complete arena, player, objectives, and timer at every decision point. It does not need tactile-only sensing, a hidden-map belief state, or a recurrent network to remember unseen barriers.

The game is still visually rendered in Phaser, but authoritative rules stay in an engine-independent simulator for repeatable headless training.

## Game rules

- The player and enemy move in continuous two-dimensional space inside a circular arena.
- The arena has a circular outer boundary, one central circular obstacle, and two static bars.
- At reset, a seed regenerates the angle and lengths of the two bars; the arena remains fixed during an episode.
- Both actors have eight discrete desired movement actions: `up`, `up-right`, `right`, `down-right`, `down`, `down-left`, `left`, `up-left`.
- Diagonal motion is normalized so it is not faster than cardinal motion.
- Barriers, the central obstacle, and the arena boundary block movement.
- The player collects pellets. Surge Orbs are player-facing pickups that provide score and a temporary speed advantage.
- A round ends on enemy capture, pellet clear, or expiry of the 60-second timer.

## Formal MDP

`M = (S, A, P, R, gamma, rho0)`

### State: `S`

The state must contain all information needed to predict the next state and reward after an enemy action. At an enemy decision point, the exact state consists of:

| State group | Contents |
| --- | --- |
| Arena geometry | Outer radius, central-obstacle radius, and both bar endpoints/widths. Constants may be omitted once fixed. |
| Player | Continuous x/y position, current velocity or held direction, speed, Surge time remaining, and the active scripted player-controller identity. |
| Enemy | Continuous x/y position, current movement direction, speed, and time until its next decision if decisions are not aligned exactly to the fixed step. |
| Objectives | Every pellet and Surge Orb position plus its active/collected flag. |
| Clock | Normalized time remaining. |

The simulator has an exact object state. The learning model needs a fixed-size numerical encoding. With a maximum of 32 pellets and 3 Surge Orbs, one exact vector representation is:

| Features | Count |
| --- | ---: |
| Player x/y, velocity x/y, Surge fraction | 5 |
| Enemy x/y, heading one-hot | 10 |
| Two bars: each endpoint x/y pair, normalized | 8 |
| 32 pellets: x/y plus active flag | 96 |
| 3 Surge Orbs: x/y plus active flag | 9 |
| Time fraction and bias | 2 |
| Player-controller identity: random / evade / pellet | 3 |
| Current safe-path distance; 8 candidate safe-path distances; 8 candidate clearances | 17 |
| **Total** | **150** |

Inactive pellets and orbs retain their fixed spawn position but have an active flag of `0`. The player-controller one-hot is required because different controller rules can produce different next player moves from otherwise identical world positions. This preserves a fixed feature order and keeps the state Markov. A spatial occupancy-map input is an alternative once layouts become more varied.

The earlier tactile-feeler and short-history proposal is intentionally retired for this MDP version.

### Actions: `A`

`A = {up, up-right, right, down-right, down, down-left, left, up-left}`

The enemy selects one desired direction every fixed decision interval, initially 0.28 seconds. It moves continuously in that normalized direction between decisions.

The environment applies an eight-action collision mask. Directions that would immediately collide during the next decision interval are unavailable. Unlike the POMDP version, this mask is not information leakage: the enemy already has the full arena geometry.

### Transition: `P`

Use a fixed physics timestep, for example 1/30 second. A transition consists of:

`state + enemy action + player action -> movement/collision -> collection/Surge checks -> terminal check -> next state`

The arena seed and player policy create variation between episodes. The active player policy is part of the state, so it is not a hidden transition mode. Given a seed, fixed timestep, actions, and deterministic player policy, the simulator should be reproducible.

### Reward: `R`

Enemy reward is separate from player-visible score:

| Event | Reward |
| --- | ---: |
| Capture player | +100 |
| Player clears all pellets | -100 |
| Player survives until timeout | -50 |
| Every enemy decision step | -0.02 |
| Useful reduction in safe-path distance to player | +0.05, small shaping term |
| Enemy finishes a move within 24 px of an obstacle | smooth penalty from 0 to -0.20 |

Capture must dominate shaping rewards. Safe-path distance is calculated from a coarse collision-aware grid. The obstacle-clearance penalty is based only on the current transition (`s, a, s'`), not a hidden action/position history, so it does not break the Markov property. Later, failed-round penalties may scale with player pellet progress and time remaining.

### Discount and reset distribution

- Initial discount: `gamma = 0.99`.
- `rho0`: sample an arena seed, validate fair non-colliding spawns and reachable pellets, then start at full time.
- Episodes end on capture, pellet clear, or time limit.

## Player behavior during training

The player is part of the environment's transition dynamics. The current environment supports three visible controller modes: `random`, `evade`, and `pellet`. Train/evaluate on held-out map seeds and all three modes. Future cautious/wall-hugging, scripted, and recorded-human modes must each be added to the state encoding before use; otherwise they would become hidden transition modes. An enemy trained against only one player style will overfit.

## Navigation curriculum

Training starts with the central obstacle only, progresses to short bars, then uses normal and long-bar hard layouts. The final stage deterministically oversamples hard layouts one episode out of three. Evaluation always uses held-out `full` layouts so the curriculum does not make scores easier.

## RL algorithm families

### Tabular methods

Tabular dynamic programming, Monte Carlo control, Sarsa, and Q-learning are useful conceptual references, but the continuous positions and generated geometry make a literal state table infeasible.

### Linear value-based approximation

| Algorithm | Learns | Concrete game example | Pros | Cons |
| --- | --- | --- | --- | --- |
| Semi-gradient Sarsa | `Q(s,a)` | Learns that `up-right` is valuable when it leads around the upper end of a bar toward the player. | Simple, on-policy, transparent. | One-step capture credit moves backward slowly. |
| n-step Sarsa | `Q(s,a)` with n-step targets | A capture three decisions after a turn teaches that turn sooner. | Faster delayed credit. | Extra `n` tuning. |
| Sarsa(lambda) / True Online Sarsa(lambda) | `Q(s,a)` plus eligibility traces | A capture rewards the recent sequence: avoid bar, circle core, then intercept player. | Strong interpretable baseline; traces suit delayed capture rewards. | Linear features may not represent complex geometry/player interactions well. |
| Expected Sarsa | Expected action value under current policy | Smooths learning when multiple safe diagonals are plausible. | Lower target variance. | Usually a modest improvement only. |

### Off-policy value-based approximation

| Algorithm | Learns | Concrete game example | Pros | Cons |
| --- | --- | --- | --- |
| Q-learning | Greedy `Q(s,a)` from exploratory data | Learns the best chase turn even while training behavior occasionally explores another direction. | Reuses exploratory experience. | Function approximation plus bootstrapping and off-policy targets can be unstable. |
| Double Q-learning / Double DQN | Two Q estimates | Prevents an overoptimistic `up-right` value caused by rare lucky captures. | Reduces value overestimation. | More machinery; still needs state encoder and replay design. |
| DQN | Neural `Q(s,a)` plus replay | Learns nonlinear interaction: player near a pellet cluster + bar location + remaining time. | Natural 8-action output; replay can be sample-efficient. | Target/replay/hyperparameter complexity; player-policy changes can stale replay data. |
| Dueling / prioritized / distributional DQN variants | Improved DQN representations or replay | Distinguishes a generally strong arena state from the best immediate direction. | Can improve a mature DQN experiment. | Premature complexity for first implementation. |

Gradient-TD, emphatic-TD, and related algorithms are valuable off-policy **prediction** methods, but are not the first choice for this chase controller because they do not directly supply a practical eight-action control policy.

### Policy-gradient and actor-critic methods

| Algorithm | Learns | Concrete game example | Pros | Cons |
| --- | --- | --- | --- |
| REINFORCE | Direct policy `pi(a|s)` | Increases probability of a successful diagonal interception after a high-return episode. | Very clear direct-policy idea. | High variance: a 60-second result gives noisy credit to early turns. |
| REINFORCE with baseline | Policy plus value baseline | Credits a turn only when capture was better than predicted for that state. | Lower variance. | Usually superseded by actor-critic. |
| Actor-critic | Actor policy plus critic `V(s)` | Actor favors the safe path around a bar; critic evaluates whether this state predicts capture. | Direct stochastic policy and useful critic. | Sensitive to actor/critic learning-rate tuning. |
| Actor-critic with eligibility traces | Actor/critic plus traces | A capture strengthens several earlier intercept choices. | Better delayed-credit assignment. | More moving parts. |
| A2C / A3C | Parallel actor-critic | Multiple generated arena seeds train simultaneously. | Efficient headless simulation use. | Usually replaced by PPO in a new project. |
| PPO | Clipped actor-critic policy optimization | Actor outputs eight direction probabilities, then the collision mask removes unsafe directions. | Stable practical method; natural for continuous-state, discrete-action game. | On-policy, so it needs fresh simulation data. |
| TRPO | Trust-region actor-critic | Limits how sharply the chase policy changes after a batch. | Strong theoretical stability. | More complicated than PPO for little benefit here. |

### Model-based and planning methods

| Algorithm | Concrete game example | Pros | Cons |
| --- | --- | --- | --- |
| Dyna-Q | Learn from real chase steps, then simulate extra steps around the known bars. | Uses the known MDP structure efficiently. | Needs a player-dynamics model. |
| Model predictive control | Simulate each of eight directions over the next few seconds and choose the best predicted interception route. | Strong planning baseline with full map. | Requires modeling/predicting player behavior; less a pure learned chase policy. |
| World models / Dreamer-style methods | Learn a latent simulator for generated arenas and player motion. | Powerful if the world later becomes much more complex. | Unnecessary complexity for current static two-bar arena. |

### Multi-agent and imitation extensions

- **Self-play or adversarial training:** train player and enemy policies against each other. Good for robustness, but can become unstable if both change too quickly.
- **Imitation plus RL:** initialize the enemy from A* or expert demonstrations, then refine its behavior using rewards. Useful if a high-quality pathfinding enemy already exists.

## Recommendation

### First benchmark: True Online Sarsa(lambda)

Use the full MDP state, eight actions, action masking, fixed timestep, and a linear/tile-coded value approximation. It is the best initial benchmark because every learned `Q(s,a)` can be inspected. If it turns around the correct end of a bar and captures faster over training, the environment and reward design are behaving sensibly.

### Final RL controller: feed-forward PPO

Use a feed-forward actor-critic with a small MLP, not an LSTM/GRU:

`full state vector -> MLP actor -> 8 masked action probabilities`

`full state vector -> MLP critic -> V(state)`

PPO is the best overall fit for the current MDP because it handles continuous state features, eight discrete actions, stochastic player opponents, and procedural training arenas without the recurrent-memory complexity required by a POMDP. It should train on many parallel headless environments.

If the goal is merely the strongest possible enemy rather than an RL experiment, a full-map A* or short-horizon planner should remain the performance reference and will likely be simpler.

## Evaluation

Train and evaluate all agents with identical rewards, arena seeds, player-policy mixtures, action masks, and environment-step budgets. At regular checkpoints, freeze the policy and evaluate without exploration over multiple training seeds and held-out maps.

Track capture rate, median capture time, pellet-clear rate, timeout-survival rate, mean episode return, learning-curve area, invalid-action/loop rate, training-seed variance, and held-out generalization gap.

## Next implementation phase

1. Refactor `GameSimulation` into deterministic `reset(seed)`, `observe()`, `validActions()`, and `step(playerAction, enemyAction)` APIs.
2. Add arena validation and simulator tests.
3. Implement a random and full-map greedy/A* baseline.
4. Implement True Online Sarsa(lambda) as the first RL benchmark.
5. Add feed-forward PPO only after the evaluation harness is trustworthy.
