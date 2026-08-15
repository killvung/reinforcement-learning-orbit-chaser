import {
  Direction,
  GameSimulation,
  PLAYER_OBSERVATION_SIZE,
  PlayerController,
  PlayerObservation,
  actionOrder,
} from './simulation.js';

export const BROWSER_LINEAR_SARSA_FORMAT = 'orbit-chase-linear-sarsa-browser-v1';

const GROUP_PLAYER_POSITION = 1;
const GROUP_PLAYER_VELOCITY = 2;
const GROUP_ENEMY_RELATIVE = 3;
const GROUP_ENEMY_CLOCK = 4;
const GROUP_PELLET_RANK = 10;
const GROUP_ORB_RANK = 20;
const GROUP_BAR = 30;
const GROUP_SURGE_TIME = 40;
const GROUP_ENEMY_HEADING = 50;
const GROUP_ACTION_MASK = 60;
const NEAREST_PELLET_COUNT = 4;
const PELLET_COUNT = 32;
const SURGE_ORB_COUNT = 3;
const OCCUPANCY_SALT = -1;
const FNV_OFFSET = 0xCBF29CE484222325n;
const FNV_PRIME = 0x100000001B3n;
const UINT64_MASK = 0xFFFFFFFFFFFFFFFFn;

type EncoderConfig = { capacity: number; bins: number; tilings: number };
type WeightRow = { indices: number[]; values: number[] };

export type BrowserSarsaArtifact = {
  format: string;
  source_format?: string;
  checkpoint?: string;
  observation_size: number;
  action_order: string[];
  encoder: EncoderConfig;
  weights: WeightRow[];
};

/** Sparse hashed tile encoder matching `orbit_chase.sarsa.FeatureEncoder`. */
export class FeatureEncoder {
  constructor(
    readonly capacity: number,
    readonly bins: number,
    readonly tilings: number,
  ) {}

  encode(observation: readonly number[], actionMask: readonly boolean[]): number[] {
    if (observation.length !== PLAYER_OBSERVATION_SIZE) {
      throw new Error(`Expected observation length ${PLAYER_OBSERVATION_SIZE}, got ${observation.length}.`);
    }
    if (actionMask.length !== actionOrder.length) {
      throw new Error(`Expected action mask length ${actionOrder.length}, got ${actionMask.length}.`);
    }
    if (actionMask.some((value) => value !== true && value !== false)) {
      throw new Error('Action mask values must be binary.');
    }
    if (!actionMask.some(Boolean)) {
      throw new Error('Action mask must contain at least one valid action.');
    }

    const features = new Set<number>();
    const player = [observation[0], observation[1]];
    this.addTiles(features, GROUP_PLAYER_POSITION, player, -2, 2);
    this.addTiles(features, GROUP_PLAYER_VELOCITY, [observation[2], observation[3]], -1, 1);
    this.addTiles(features, GROUP_ENEMY_RELATIVE, [observation[5] - player[0], observation[6] - player[1]], -2, 2);
    this.addTiles(features, GROUP_ENEMY_CLOCK, [observation[15]], 0, 1);
    this.addTiles(features, GROUP_SURGE_TIME, [observation[4], observation[129]], 0, 1);
    this.addActiveDirectionFeatures(features, GROUP_ENEMY_HEADING, observation.slice(7, 15));
    this.addActiveDirectionFeatures(features, GROUP_ACTION_MASK, actionMask.map((valid) => valid ? 1 : 0));
    this.addNearestCollectibles(features, observation.slice(24, 120), PELLET_COUNT, NEAREST_PELLET_COUNT, player, GROUP_PELLET_RANK);
    this.addNearestCollectibles(features, observation.slice(120, 129), SURGE_ORB_COUNT, SURGE_ORB_COUNT, player, GROUP_ORB_RANK);
    this.addBarFeatures(features, observation.slice(16, 24), player);
    return [...features].sort((left, right) => left - right);
  }

  private addTiles(features: Set<number>, group: number, values: readonly number[], lower: number, upper: number): void {
    const normalized = values.map((value) => (clip(value, lower, upper) - lower) / (upper - lower));
    for (let tiling = 0; tiling < this.tilings; tiling += 1) {
      const offset = tiling / this.tilings;
      const coordinates = normalized.map((value) => clip(Math.floor(value * this.bins + offset), 0, this.bins - 1));
      features.add(this.hash(group, tiling, ...coordinates));
    }
  }

  private addActiveDirectionFeatures(features: Set<number>, group: number, values: readonly number[]): void {
    values.forEach((value, direction) => {
      if (value > 0) features.add(this.hash(group, direction));
    });
  }

  private addNearestCollectibles(
    features: Set<number>,
    encoded: readonly number[],
    count: number,
    nearestCount: number,
    player: readonly number[],
    group: number,
  ): void {
    const active: number[][] = [];
    for (let index = 0; index < count; index += 1) {
      const offset = index * 3;
      if (encoded[offset + 2] > 0.5) active.push([encoded[offset], encoded[offset + 1]]);
    }
    active.sort((left, right) => comparePoints(left, right, player));
    for (let rank = 0; rank < nearestCount; rank += 1) {
      const groupId = group + rank;
      const occupied = rank < active.length;
      features.add(this.hash(groupId, OCCUPANCY_SALT, occupied ? 1 : 0));
      if (occupied) {
        this.addTiles(features, groupId, [active[rank][0] - player[0], active[rank][1] - player[1]], -2, 2);
      }
    }
  }

  private addBarFeatures(features: Set<number>, encoded: readonly number[], player: readonly number[]): void {
    const bars: Array<[number[], number[]]> = [];
    for (let index = 0; index < 2; index += 1) {
      const offset = index * 4;
      let start = [encoded[offset], encoded[offset + 1]];
      let end = [encoded[offset + 2], encoded[offset + 3]];
      if (lexLess(end, start)) [start, end] = [end, start];
      bars.push([start, end]);
    }
    bars.sort((left, right) => compareBars(left, right));
    bars.forEach(([start, end], index) => {
      const closest = closestPoint(start, end, player);
      this.addTiles(features, GROUP_BAR + index, [closest[0] - player[0], closest[1] - player[1]], -2, 2);
    });
  }

  private hash(...components: number[]): number {
    let value = FNV_OFFSET;
    const bytes = new Uint8Array(8);
    const view = new DataView(bytes.buffer);
    for (const component of components) {
      view.setBigInt64(0, BigInt(component), true);
      for (let index = 0; index < 8; index += 1) {
        value ^= BigInt(bytes[index]);
        value = (value * FNV_PRIME) & UINT64_MASK;
      }
    }
    return Number(value % BigInt(this.capacity));
  }
}

/** Browser inference for a frozen linear Sarsa player checkpoint. */
export class SarsaPlayerController implements PlayerController {
  readonly name = 'linear-sarsa';
  private readonly lookups: ReadonlyArray<ReadonlyMap<number, number>>;

  private constructor(
    private readonly encoder: FeatureEncoder,
    lookups: ReadonlyArray<ReadonlyMap<number, number>>,
  ) {
    this.lookups = lookups;
  }

  static fromArtifact(artifact: unknown): SarsaPlayerController {
    const document = validateArtifact(artifact);
    const encoder = new FeatureEncoder(document.encoder.capacity, document.encoder.bins, document.encoder.tilings);
    const lookups = document.weights.map((row) => {
      const lookup = new Map<number, number>();
      row.indices.forEach((index, offset) => lookup.set(index, row.values[offset]));
      return lookup;
    });
    return new SarsaPlayerController(encoder, lookups);
  }

  static async load(url: string): Promise<SarsaPlayerController> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load linear Sarsa artifact (${response.status}).`);
    return SarsaPlayerController.fromArtifact(await response.json());
  }

  encode(observation: PlayerObservation): number[] {
    return this.encoder.encode(observation.features, observation.actionMask);
  }

  qValues(indices: readonly number[]): number[] {
    return this.lookups.map((lookup) => {
      let total = 0;
      for (const index of indices) total += lookup.get(index) ?? 0;
      return total;
    });
  }

  choose(observation: PlayerObservation): Direction {
    const indices = this.encode(observation);
    return actionOrder[selectGreedyAction(this.qValues(indices), observation.actionMask)];
  }

  selectAction(simulation: GameSimulation): Direction {
    return this.choose(simulation.observe());
  }
}

/** Masked greedy action: highest valid Q, lowest index on exact ties. */
export function selectGreedyAction(qValues: readonly number[], actionMask: readonly boolean[]): number {
  if (qValues.length !== actionOrder.length || actionMask.length !== actionOrder.length) {
    throw new Error(`Expected ${actionOrder.length} Q-values and mask entries.`);
  }
  let bestIndex = -1;
  let bestValue = -Infinity;
  for (let index = 0; index < actionOrder.length; index += 1) {
    if (!actionMask[index]) continue;
    if (bestIndex < 0 || qValues[index] > bestValue) {
      bestIndex = index;
      bestValue = qValues[index];
    }
  }
  if (bestIndex < 0) throw new Error('Action mask must contain at least one valid action.');
  return bestIndex;
}

function validateArtifact(artifact: unknown): BrowserSarsaArtifact {
  if (!isRecord(artifact)) throw new Error('Linear Sarsa artifact must be an object.');
  if (artifact.format !== BROWSER_LINEAR_SARSA_FORMAT) {
    throw new Error(`Unsupported browser artifact format: ${String(artifact.format)}.`);
  }
  if (artifact.observation_size !== PLAYER_OBSERVATION_SIZE) {
    throw new Error(`Browser artifact observation_size must be ${PLAYER_OBSERVATION_SIZE}.`);
  }
  if (!Array.isArray(artifact.action_order) || artifact.action_order.join(',') !== actionOrder.join(',')) {
    throw new Error('Browser artifact action_order does not match the game contract.');
  }
  const encoder = artifact.encoder;
  if (!isRecord(encoder) || !isPositiveInt(encoder.capacity) || !isPositiveInt(encoder.bins) || !isPositiveInt(encoder.tilings)) {
    throw new Error('Browser artifact encoder metadata is invalid.');
  }
  const capacity = encoder.capacity;
  const bins = encoder.bins;
  const tilings = encoder.tilings;
  if (!Array.isArray(artifact.weights) || artifact.weights.length !== actionOrder.length) {
    throw new Error(`Browser artifact must contain ${actionOrder.length} weight rows.`);
  }
  const rows = artifact.weights.map((row, action) => validateWeightRow(row, capacity, action));
  return {
    format: BROWSER_LINEAR_SARSA_FORMAT,
    source_format: typeof artifact.source_format === 'string' ? artifact.source_format : undefined,
    checkpoint: typeof artifact.checkpoint === 'string' ? artifact.checkpoint : undefined,
    observation_size: PLAYER_OBSERVATION_SIZE,
    action_order: [...actionOrder],
    encoder: { capacity, bins, tilings },
    weights: rows,
  };
}

function validateWeightRow(row: unknown, capacity: number, action: number): WeightRow {
  if (!isRecord(row) || !Array.isArray(row.indices) || !Array.isArray(row.values)) {
    throw new Error(`Weight row ${action} must contain indices and values arrays.`);
  }
  const rawIndices = row.indices;
  const rawValues = row.values;
  if (rawIndices.length !== rawValues.length) {
    throw new Error(`Weight row ${action} indices and values must have matching lengths.`);
  }
  const indices: number[] = [];
  const values: number[] = [];
  rawIndices.forEach((index, offset) => {
    if (typeof index !== 'number' || !Number.isInteger(index) || index < 0 || index >= capacity) {
      throw new Error(`Weight row ${action} has an index outside [0, ${capacity}).`);
    }
    if (offset > 0 && !(index > indices[offset - 1])) {
      throw new Error(`Weight row ${action} indices must be strictly increasing.`);
    }
    const value = rawValues[offset];
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new Error(`Weight row ${action} values must be finite numbers.`);
    }
    indices.push(index);
    values.push(value);
  });
  return { indices, values };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isPositiveInt(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0;
}

function clip(value: number, lower: number, upper: number): number {
  return Math.min(upper, Math.max(lower, value));
}

function squaredDistance(point: readonly number[], player: readonly number[]): number {
  const dx = point[0] - player[0];
  const dy = point[1] - player[1];
  return dx * dx + dy * dy;
}

function comparePoints(left: readonly number[], right: readonly number[], player: readonly number[]): number {
  const distance = squaredDistance(left, player) - squaredDistance(right, player);
  if (distance !== 0) return distance;
  if (left[0] !== right[0]) return left[0] < right[0] ? -1 : 1;
  if (left[1] !== right[1]) return left[1] < right[1] ? -1 : 1;
  return 0;
}

function lexLess(left: readonly number[], right: readonly number[]): boolean {
  if (left[0] !== right[0]) return left[0] < right[0];
  return left[1] < right[1];
}

function compareBars(left: [number[], number[]], right: [number[], number[]]): number {
  const keysLeft = [...left[0], ...left[1]];
  const keysRight = [...right[0], ...right[1]];
  for (let index = 0; index < keysLeft.length; index += 1) {
    if (keysLeft[index] !== keysRight[index]) return keysLeft[index] < keysRight[index] ? -1 : 1;
  }
  return 0;
}

function closestPoint(start: readonly number[], end: readonly number[], point: readonly number[]): number[] {
  const sx = end[0] - start[0];
  const sy = end[1] - start[1];
  const lengthSquared = sx * sx + sy * sy;
  if (lengthSquared === 0) return [start[0], start[1]];
  const fraction = clip(((point[0] - start[0]) * sx + (point[1] - start[1]) * sy) / lengthSquared, 0, 1);
  return [start[0] + fraction * sx, start[1] + fraction * sy];
}
