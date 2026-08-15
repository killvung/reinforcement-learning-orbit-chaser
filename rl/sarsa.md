# Linear True Online Sarsa(lambda)

## Decision

Use one linear, masked True Online Sarsa(lambda) agent before considering a
neural action-value function. The implementation keeps each feature, action
value, weight, and eligibility trace inspectable.

The agent acts once per 100 ms Gym decision. The fixed red enemy, collision
rules, and action mask remain environment dynamics.

`LinearSarsaAgent` implements the shared `Agent` contract in
`orbit_chase.agent`: Gym `q_values(state)`, masked epsilon-greedy
`select_action`, and `update(Transition)`. A later neural Expected-Sarsa
agent uses the same surface. It ignores `Transition.next_action` and queues
n-step returns inside `update`. Held-out evaluation uses `choose` (greedy).

## Seed protocol

| Split | Seeds | Purpose |
| --- | --- | --- |
| Training | `0`–`7999` | Weight updates and exploration |
| Validation | `8000`–`8999` | Hyperparameter selection and model choice |
| Final test | `10000`–`10099` | One final comparison with fixed baselines |

Do not select tile widths, epsilon schedules, or a neural fallback from final
test results. Pellet-seeking sets the initial final-test reference: 0.10 clear
rate and 13.62 mean pellets over 100 seeds.

## Policy contract

The learner receives the public Gym state:

```python
state["observation"]  # float32, shape (130,)
state["action_mask"]  # int8, shape (8,)
```

It chooses only actions with a true mask entry. Training uses masked
epsilon-greedy selection. Evaluation uses masked greedy selection with the
same tie break: the lowest action index.

At a terminal transition, the bootstrap value is zero. The agent does not
sample a next action after capture, clear, or timeout.

## Feature map

Do not tile the raw 130-value vector as one joint space. Raw pellet and Orb
slots have procedural order, and high-dimensional joint tilings waste hash
capacity.

Build a deterministic `FeatureEncoder` from the observation alone. It must
never inspect `GameSimulation`, arena objects, or unencoded pickup positions.
Version one uses these state groups:

| Group | Source | Representation |
| --- | --- | --- |
| Player position | indices 0–1 | one 2-D tile group |
| Player velocity | indices 2–3 | one 2-D tile group |
| Enemy relative position | indices 0–1 and 5–6 | one 2-D tile group |
| Enemy heading and clock | indices 7–15 | eight direct binary heading features and one 1-D tile group for the clock |
| Nearest pellets | active entries from 24–119 | occupancy bit plus relative `(dx, dy)` tiles for the four nearest; absent bit only when a rank is empty |
| Orbs | active entries from 120–128 | occupancy bit plus relative `(dx, dy)` tiles for each of the three ranks; absent bit when empty |
| Bars | endpoints 16–23 | one relative closest-point `(dx, dy)` pair per bar, with bars sorted lexicographically |
| Surge and time | indices 4 and 129 | one 2-D tile group |
| Action mask | public mask | eight direct binary features |

Each tile group uses eight offset tilings with eight bins per active dimension.
Clip normalized positions and relative offsets to `[-2, 2]`; clip normalized
velocities to `[-1, 1]`; clip fractions to `[0, 1]`. Tile bin indices stay in
`0` through `bins-1`; the clipped upper endpoint uses the last bin. Give every feature group
and tiling a fixed integer salt. Pellet ranks use `10–13`, Orb ranks `20–22`,
bars `30–31`, surge and time `40`, heading `50`, and the action mask `60`.
Hash into `2^18` feature indices with one documented hash implementation and
seed. Deduplicate collisions into binary features, then return sorted indices
so tests can compare exact outputs.

The nearest-pellet and Orb groups keep rank in the feature-group salt. An
occupied rank hashes a present bit plus relative `(dx, dy)` tiles. An empty
rank hashes only an absent bit, so it does not look like an object on the
player.

## Linear action values and traces

Let `phi(s)` be the sparse feature indices from the encoder. Build an
action-value feature vector with eight blocks:

```text
x(s, a) = [0, ..., phi(s), ..., 0]
```

Only block `a` contains active features. With `d = 2^18` state features, the
weight vector and eligibility trace each have `8 × d` entries. Store them as
sparse traces; dense weights remain practical at this size.

Use the true-online update at each decision transition:

```text
q      = w · x(s, a)
q_next = 0 if terminal else w · x(s', a')
delta  = r + gamma * q_next - q
e      = gamma * lambda * e + (1 - alpha * gamma * lambda * (e · x)) * x
w      = w + alpha * (delta + q - q_old) * e - alpha * (q - q_old) * x
q_old  = q_next
```

Initialize `q_old` to zero after reset. Clear traces after each episode. The
implementation must unit-test the terminal bootstrap, sparse trace update,
and one-step equivalence with Sarsa(0) when `lambda = 0`.

## Initial configuration

Treat these values as validation-tunable settings, not environment rules:

| Setting | Initial value |
| --- | ---: |
| Gamma per 100 ms decision | 0.995 |
| Lambda | 0.90 |
| Alpha | `0.1 / 8` |
| Epsilon | linearly decay from 0.20 to 0.02 |
| Hash capacity | `2^18` state features |
| Tile tilings and bins | 8 and 8 |

Record the full configuration, training seed range, random seed, validation
metrics, and checkpoint after each run. Compare candidates on paired
validation seeds. Promote a candidate only when its clear-rate improvement
over pellet-seeking has a positive bootstrap confidence interval; use pellets
and return as secondary checks.

## Neural fallback

Consider a neural action-value function only after the linear agent fails the
validation gate. [`neural-sarsa.md`](neural-sarsa.md) defines its typed-set
encoder, online Expected Sarsa update, and validation protocol. It does not
reuse true-online traces.
