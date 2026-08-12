import { isBlocked, Point } from './arena';
import { Direction, directions, EnemyController, EnemyDebugState, GameSimulation } from './simulation';

const actionOrder: Direction[] = ['up', 'up-right', 'right', 'down-right', 'down', 'down-left', 'left', 'up-left'];

type Layer = { weight: number[][]; bias: number[] };
type BrowserActor = {
  format: 'orbit-chase-ppo-actor-v1';
  observation_size: number;
  action_order: Direction[];
  player_policy_proxy: 'pellet';
  layers: [Layer, Layer, Layer];
};

function matVec(layer: Layer, input: number[], activation: (value: number) => number): number[] {
  return layer.weight.map((row, output) => activation(row.reduce((sum, value, index) => sum + value * input[index], layer.bias[output])));
}

/**
 * Browser inference for the actor exported by rl/export_ppo.py. Its feature
 * layout intentionally matches OrbitChaseEnemyEnv._observe exactly.
 */
export class PpoEnemyController implements EnemyController {
  readonly name = 'PPO';
  private ppoAction = 'n/a';
  private activeAction = 'n/a';
  private validActionCount = 0;

  private constructor(private readonly actor: BrowserActor) {}

  static async load(url: string): Promise<PpoEnemyController> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load PPO actor (${response.status}).`);
    const actor = await response.json() as BrowserActor;
    if (actor.format !== 'orbit-chase-ppo-actor-v1' || actor.observation_size !== 133 || actor.layers.length !== 3 || actor.action_order.join(',') !== actionOrder.join(',')) {
      throw new Error('PPO actor has an unsupported format. Re-export it with rl/export_ppo.py.');
    }
    return new PpoEnemyController(actor);
  }

  selectAction(simulation: GameSimulation): Direction {
    const observation = this.observe(simulation);
    const hidden = matVec(this.actor.layers[0], observation, Math.tanh);
    const secondHidden = matVec(this.actor.layers[1], hidden, Math.tanh);
    const logits = matVec(this.actor.layers[2], secondHidden, (value) => value);
    const candidates = actionOrder
      .map((direction, index) => ({ direction, logit: logits[index] }))
      // Test the whole path, not just its endpoint. An endpoint-only test can
      // incorrectly allow crossing a narrow bar and leave the enemy stuck.
      .filter(({ direction }) => this.canTravel(simulation, direction, 34))
      .sort((a, b) => b.logit - a.logit);
    const action = candidates[0]?.direction ?? simulation.enemyDirection;
    this.ppoAction = candidates[0]?.direction ?? 'none';
    this.activeAction = action;
    this.validActionCount = candidates.length;
    return action;
  }

  getDebugState(): EnemyDebugState {
    return {
      'PPO proposal': this.ppoAction,
      'Applied action': this.activeAction,
      'Valid actions': this.validActionCount,
    };
  }

  private canTravel(simulation: GameSimulation, direction: Direction, distance: number): boolean {
    const vector = directions[direction];
    const length = Math.hypot(vector.x, vector.y);
    const steps = Math.ceil(distance / 4);
    let point = { x: simulation.enemy.x, y: simulation.enemy.y };
    for (let step = 0; step < steps; step += 1) {
      point = { x: point.x + vector.x / length * distance / steps, y: point.y + vector.y / length * distance / steps };
      if (isBlocked(simulation.arena, point, simulation.enemy.radius)) return false;
    }
    return true;
  }

  private observe(simulation: GameSimulation): number[] {
    const { arena, player, enemy } = simulation;
    const normalized = (point: Point): [number, number] => [(point.x - arena.center.x) / arena.radius, (point.y - arena.center.y) / arena.radius];
    const features: number[] = [
      ...normalized(player),
      simulation.playerVelocity.x / player.speed, simulation.playerVelocity.y / player.speed,
      simulation.surgeRemaining / 4,
      ...normalized(enemy),
      ...actionOrder.map((direction) => direction === simulation.enemyDirection ? 1 : 0),
    ];
    for (const bar of arena.bars) features.push(...normalized(bar.from), ...normalized(bar.to));
    for (const slot of simulation.pelletSlots) features.push(...normalized(slot.point), slot.active ? 1 : 0);
    for (const slot of simulation.orbSlots) features.push(...normalized(slot.point), slot.active ? 1 : 0);
    // The model was trained with random/evade/pellet scripted players. A human
    // is closest to the pellet-seeking proxy used for browser deployment.
    features.push(Math.max(0, simulation.timeRemaining) / 60, 1, 0, 0, 1);
    if (features.length !== 133) throw new Error(`PPO observation mismatch: expected 133 features, got ${features.length}.`);
    return features;
  }
}
