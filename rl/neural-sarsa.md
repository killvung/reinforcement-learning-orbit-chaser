# Neural Sarsa fallback

## Decision gate

Use this design only after the selected linear Sarsa(lambda) configuration
fails the validation gate in [`sarsa.md`](sarsa.md). Tune both agents on seeds
`8000` through `8999`. Run final-test seeds `10000` through `10099` once after
choosing the model family.

The neural agent uses masked, online 5-step Expected Sarsa. It uses no replay
buffer and no true-online eligibility traces. This keeps the update on-policy
while the fallback stays small enough to inspect.

Pin PyTorch in `requirements.txt` before implementation. Do not rely on a
locally installed copy.

## Public policy state

The network receives only the Gym observation and action mask. It cannot read
`GameSimulation`, an arena object, or unencoded pickup positions.

The action mask has two roles:

- concatenate its eight binary values to the global input because local
  mobility affects value;
- set invalid Q-values to negative infinity during action selection and
  Expected-Sarsa target calculation.

The network returns unmasked Q-values. The policy applies the mask outside the
forward pass.

## Set-based encoder

Pellet, Orb, and bar slot order has no gameplay meaning. Encode each as a set.
Keep object types separate because pellets drive the clear condition and Orbs
start Surge.

| Input | Item features | Encoder and pool |
| --- | --- | --- |
| Pellets | normalized `(x, y)`, player-relative `(dx, dy)`, active | shared 5→64→64 MLP; masked max and active mean |
| Orbs | normalized `(x, y)`, player-relative `(dx, dy)`, active | separate shared 5→32→32 MLP; masked max and active mean |
| Bars | canonicalized endpoint `(x1, y1, x2, y2)` | shared 4→32→32 MLP; max and mean over two bars |

Canonicalize each bar by ordering its endpoints lexicographically, then sort
the two bars lexicographically. Sort active pellets and active Orbs by distance
to the player, then coordinates, before their shared encoders. The sort is not
required for pooling, but it makes test snapshots and debugging stable.

For masked max, replace inactive embeddings with negative infinity. Return a
zero vector when no active item exists. For active mean, divide by
`max(active_count, 1)`. Concatenate the count feature so an empty set differs
from a set whose pooled embeddings sum to zero.

## Global encoder and Q head

The global vector contains these public values:

```text
player: position, velocity, Surge                  5
enemy: position, heading, decision-clock fraction 11
time fraction                                      1
action mask                                        8
pellet and Orb active counts                       2
```

Pass the 27 values through a 27→64→64 MLP. Concatenate its output with the
pellet, Orb, and bar pooled embeddings. Feed the resulting 320 values through
a 320→128→8 MLP with a linear final layer. The eight outputs follow `Direction`
order and may take any real value.

Apply Huber loss to the selected valid action only. Clip global gradient norm
to 10. Initialize AdamW with learning rate `3e-4` and weight decay `1e-5`.
Treat these settings as validation-tunable values.

## Online 5-step Expected Sarsa update

At each decision, sample `a_t` from the masked epsilon-greedy policy based on
`Q_theta(s_t)`. Epsilon decays from 0.20 to 0.02 during training. The lowest
valid action index breaks greedy ties. For a nonterminal bootstrap state,
construct the same masked epsilon-greedy distribution `pi_theta(a | s)` and
compute:

```text
V_theta(s) = sum_a pi_theta(a | s) Q_theta(s, a)
G_t       = r_t + gamma r_(t+1) + ... + gamma^(n-1) r_(t+n-1)
            + gamma^n V_theta(s_(t+n))
```

Use `n = 5` and `gamma = 0.995` per 100 ms decision. Cut the return at a
terminal state and omit its bootstrap term. Detach `G_t` before the Huber loss.

Keep the last five transitions in an episode-local deque. Once the deque has
five rewards, update the oldest state-action pair. Flush the remaining entries
at capture, clear, and timeout. The agent never samples a next action after a
terminal transition.

This is an online control method. Adding replay changes the state-action
distribution and requires a separate off-policy design. If replay becomes
necessary, choose and document Double DQN, Retrace, or another explicit method
before writing the buffer.

## Required tests

Before training, test these properties:

1. Permuting pellet, Orb, or bar order leaves Q-values unchanged.
2. An all-inactive pellet or Orb set produces finite Q-values and no NaNs.
3. Masked epsilon-greedy samples no invalid action and uses uniform exploration
   over valid actions.
4. Terminal capture, clear, and timeout targets omit the bootstrap term.
5. A hand-computed five-step sequence matches the queued update target.
6. The forward pass produces shape `(batch, 8)` and preserves `Direction`
   order.

Log seed ranges, network configuration, optimizer settings, epsilon schedule,
validation metrics, and checkpoints. Compare the chosen neural run with all
three fixed-policy rows and the linear Sarsa agent on the final test only.
