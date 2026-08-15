# Player Model Browser Bridge

## Goal

Run the trained linear Sarsa player policy in the Phaser game without changing
the policy state, feature map, action order, or 100 ms decision cadence used
during Python evaluation.

The selected checkpoint is
`rl/models/linear-sarsa-20260815-030554-ep8000.npz`. It stores a linear
action-value function with eight action rows and a 262,144-bucket hashed tile
encoder. Greedy evaluation recorded about a 90% clear rate on 10,000 seeds
`10000–19999`. The documented 100-seed final-test slice `10000–10099` was 96/100.

## Existing contract

Python creates a 130-value observation and an eight-value action mask in
`rl/orbit_chase/observation.py` and `rl/orbit_chase/environment.py`.
`src/game/simulation.ts` already creates the same feature vector in
`GameSimulation.observe()`.

The action index contract must remain fixed:

| Index | Direction |
| ---: | --- |
| 0 | `up` |
| 1 | `up-right` |
| 2 | `right` |
| 3 | `down-right` |
| 4 | `down` |
| 5 | `down-left` |
| 6 | `left` |
| 7 | `up-left` |

The action mask remains outside the linear model. Browser inference must score
all eight actions, then choose the highest value among valid actions. Equal
values choose the lowest valid action index, matching Python evaluation.

## Artifact boundary

Browsers should not parse NumPy `.npz` files at runtime. Add a Python export
command that validates the checkpoint with `load_linear_sarsa()` and writes a
browser artifact into `public/models/`.

Use the initial JSON format below. Sparse storage retains the nonzero values
from the checkpoint and avoids placing a dense 16 MiB float64 matrix in the
asset.

```json
{
  "format": "orbit-chase-linear-sarsa-browser-v1",
  "observation_size": 130,
  "action_order": ["up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left"],
  "encoder": { "capacity": 262144, "bins": 8, "tilings": 8 },
  "weights": [
    { "indices": [17, 429], "values": [0.125, -0.5] }
  ]
}
```

The exporter must reject unsupported checkpoint formats, unexpected weight
shapes, non-finite weights, and metadata that conflicts with the game
constants. It must sort each action's indices in ascending order.

## TypeScript implementation

Add `src/game/SarsaPlayerController.ts` with two construction paths:

- `fromArtifact(artifact)` validates an already parsed artifact. Tests use
  this path.
- `load(url)` fetches JSON and delegates to `fromArtifact()`.

The controller receives `GameSimulation`, calls `simulation.observe()`, ports
the Python `FeatureEncoder.encode()` logic, calculates the eight Q-values, and
returns a `Direction`.

The encoder port must preserve these details from `rl/orbit_chase/sarsa.py`:

- tile groups, occupancy features, nearest-collectible ranking, bar
  canonicalization, clipping bounds, and tiling offsets;
- 64-bit signed-little-endian FNV-1a component hashing; TypeScript must use
  `BigInt`, then reduce the unsigned hash by the configured capacity;
- binary feature values and sparse row dot products;
- masked greedy selection with lowest-index tie breaking.

Do not use JavaScript `number` arithmetic for the FNV accumulator. Its 53-bit
integer precision changes feature indices.

## Player decision lifecycle

Add a player-controller interface beside the existing enemy-controller
interface in `src/game/simulation.ts`. Store the chosen player direction and a
player decision timer in `GameSimulation`.

At the start of each 100 ms player interval, the simulation should request one
direction from the active player controller and hold it for ten 10 ms physics
ticks. `PlayScene` should continue to supply human input when no player
controller is active. A Human / Trained Player control can switch modes after
the artifact loads.

The simulation owns this timer so the model sees the same velocity, enemy
clock, Surge state, and collectible state as the Gym environment. Render-frame
timing must not trigger extra model decisions.

## Cleanup

Delete `src/game/PpoEnemyController.ts`. It is unused and targets a different
133-feature enemy PPO artifact. Remove its stale `rl/export_ppo.py` reference.

Update `docs/CHECKPOINT.md`: it currently says that no trained player models
exist, despite the saved linear-Sarsa checkpoints and evaluation artifacts.

## Verification

The bridge ships only with the following tests.

1. **NPZ export test, Python**
   - Create a deterministic `LinearSarsaAgent`, save its `.npz`, export it,
     and check the schema, encoder metadata, sparse indices, and values.
   - Reject a malformed checkpoint and non-finite weights.

2. **Feature and Q-value parity test, Python to TypeScript**
   - Python creates a deterministic checkpoint and fixed game states.
   - Python writes each state, mask, expected active feature indices, Q-values,
     and greedy action to a temporary fixture.
   - The TypeScript test loads the exported artifact through `fromArtifact()`
     and compares its feature indices, Q-values, and action against the fixture.
   - Q-values use a strict floating-point tolerance; indices and actions must
     match exactly.

3. **Mask and tie-break test, TypeScript**
   - Verify the controller cannot return a masked action.
   - Verify equal valid Q-values choose the lowest valid action index.

4. **Decision cadence test, TypeScript**
   - Advance a simulation through multiple fixed ticks and count controller
     calls.
   - Verify one call per 100 ms interval and one held direction across its ten
     physics ticks.

Run all checks with:

```bash
npm run test
npm run test:python
npm run test:parity
```

## Delivery order

1. Add the exporter and its Python tests.
2. Add the artifact fixture generator and cross-runtime parity test.
3. Add the TypeScript encoder and controller unit tests.
4. Add the player-controller lifecycle to `GameSimulation` and test its
   cadence.
5. Wire the controller into `PlayScene`, place the chosen exported model in
   `public/models/`, and test it in the browser.
6. Remove the PPO controller and update stale documentation.

Keep the exported artifact versioned with the checkpoint used to produce it.
Changing the feature map, action order, or checkpoint schema requires a new
artifact format version and a new parity fixture.
