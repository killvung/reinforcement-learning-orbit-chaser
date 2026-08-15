import { Arena, Point, isBlocked, makeArena, seededRandom } from './arena.js';

export type Direction = 'up' | 'up-right' | 'right' | 'down-right' | 'down' | 'down-left' | 'left' | 'up-left';
export const directions: Record<Direction, Point> = {
  up: { x: 0, y: -1 }, 'up-right': { x: 1, y: -1 }, right: { x: 1, y: 0 }, 'down-right': { x: 1, y: 1 },
  down: { x: 0, y: 1 }, 'down-left': { x: -1, y: 1 }, left: { x: -1, y: 0 }, 'up-left': { x: -1, y: -1 },
};

export const actionOrder = Object.keys(directions) as Direction[];
export const FIXED_TIMESTEP = 0.01;
export const PLAYER_DECISION_INTERVAL = 0.1;
export const ENEMY_DECISION_INTERVAL = 0.28;
export const PLAYER_OBSERVATION_SIZE = 130;

export type Actor = Point & { radius: number; speed: number };
export type CollectibleSlot = { point: Point; active: boolean };
export type StepResult = { captured: boolean; cleared: boolean; timedOut: boolean; pelletsCollected: Point[]; powerCollected: Point[] };
export type EnemyDebugState = Record<string, string | number | boolean>;
export interface EnemyController { readonly name: string; selectAction(simulation: GameSimulation): Direction; getDebugState?(): EnemyDebugState; }
export interface PlayerController { readonly name: string; selectAction(simulation: GameSimulation): Direction; }

/** Exact, fixed-order input supplied to the player policy. */
export type PlayerObservation = { features: number[]; actionMask: boolean[] };

const emptyResult = (): StepResult => ({ captured: false, cleared: false, timedOut: false, pelletsCollected: [], powerCollected: [] });

/**
 * Deterministic game rules shared by the browser UI and future headless agents.
 * `step()` accepts browser elapsed time but advances only in fixed 10 ms ticks;
 * training code should call `stepFixed()` directly.
 */
export class GameSimulation {
  arena: Arena;
  player: Actor;
  enemy: Actor;
  pellets: Point[] = [];
  powerOrbs: Point[] = [];
  pelletSlots: CollectibleSlot[] = [];
  orbSlots: CollectibleSlot[] = [];
  playerVelocity: Point = { x: 0, y: 0 };
  timeRemaining = 60;
  score = 0;
  multiplier = 1;
  comboRemaining = 0;
  surgeRemaining = 0;
  enemyDirection: Direction = 'left';
  playerDirection: Direction = 'up';
  private enemyDecisionRemaining = 0;
  private playerDecisionTicksRemaining = 0;
  private elapsedAccumulator = 0;
  private enemyController: EnemyController | null = null;
  private playerController: PlayerController | null = null;

  constructor(seed = Date.now()) {
    this.arena = makeArena(seed);
    this.player = { x: 0, y: 0, radius: 12, speed: 175 };
    this.enemy = { x: 0, y: 0, radius: 13, speed: 110 };
    this.reset(seed);
  }

  /** Reset every procedural element from `seed`, making episode starts reproducible. */
  reset(seed = Date.now()): void {
    this.arena = makeArena(seed);
    const { center } = this.arena;
    this.player = { x: center.x - 165, y: center.y + 85, radius: 12, speed: 175 };
    this.enemy = { x: center.x + 165, y: center.y - 85, radius: 13, speed: 110 };
    this.enemyDirection = 'left';
    this.playerDirection = 'up';
    this.enemyDecisionRemaining = 0;
    this.playerDecisionTicksRemaining = 0;
    this.elapsedAccumulator = 0;
    this.timeRemaining = 60;
    this.pellets = []; this.powerOrbs = []; this.pelletSlots = []; this.orbSlots = [];
    this.playerVelocity = { x: 0, y: 0 };
    this.score = 0; this.multiplier = 1; this.comboRemaining = 0; this.surgeRemaining = 0;
    const random = seededRandom(seed ^ 0xa5a5a5a5);
    while (this.pellets.length < 32) {
      const angle = random() * Math.PI * 2;
      const distance = 52 + random() * 158;
      const candidate = { x: center.x + Math.cos(angle) * distance, y: center.y + Math.sin(angle) * distance };
      if (!isBlocked(this.arena, candidate, 8) && Math.hypot(candidate.x - this.player.x, candidate.y - this.player.y) > 32 && Math.hypot(candidate.x - this.enemy.x, candidate.y - this.enemy.y) > 32) {
        this.pellets.push(candidate); this.pelletSlots.push({ point: candidate, active: true });
      }
    }
    while (this.powerOrbs.length < 3) {
      const angle = random() * Math.PI * 2;
      const distance = 65 + random() * 125;
      const candidate = { x: center.x + Math.cos(angle) * distance, y: center.y + Math.sin(angle) * distance };
      if (!isBlocked(this.arena, candidate, 13) && this.pellets.every((pellet) => Math.hypot(pellet.x - candidate.x, pellet.y - candidate.y) > 34)) {
        this.powerOrbs.push(candidate); this.orbSlots.push({ point: candidate, active: true });
      }
    }
  }

  setEnemyController(controller: EnemyController | null): void { this.enemyController = controller; }
  setPlayerController(controller: PlayerController | null): void {
    this.playerController = controller;
    this.playerDecisionTicksRemaining = 0;
  }
  get enemyControllerName(): string { return this.enemyController?.name ?? 'Greedy'; }
  get playerControllerName(): string { return this.playerController?.name ?? 'Human'; }
  get enemyDebugState(): EnemyDebugState { return this.enemyController?.getDebugState?.() ?? { Navigation: 'Built-in greedy pursuit' }; }
  get enemyDecisionFraction(): number { return Math.max(0, this.enemyDecisionRemaining) / ENEMY_DECISION_INTERVAL; }

  /** Advance browser time in fixed ticks, preserving sub-tick remainder between render frames. */
  step(dt: number, playerInput: Point): StepResult {
    this.elapsedAccumulator += Math.min(Math.max(0, dt), 0.25);
    const result = emptyResult();
    while (this.elapsedAccumulator + 1e-9 >= FIXED_TIMESTEP && !result.captured && !result.cleared && !result.timedOut) {
      this.elapsedAccumulator -= FIXED_TIMESTEP;
      this.mergeResult(result, this.stepFixed(playerInput));
    }
    return result;
  }

  /** Advance precisely one physics tick. This is the headless-training contract. */
  stepFixed(playerInput: Point): StepResult {
    const dt = FIXED_TIMESTEP;
    this.comboRemaining = Math.max(0, this.comboRemaining - dt);
    if (this.comboRemaining === 0) this.multiplier = 1;
    this.surgeRemaining = Math.max(0, this.surgeRemaining - dt);
    const playerMultiplier = this.surgeRemaining > 0 ? 1.32 : 1;
    let movement = playerInput;
    if (this.playerController !== null) {
      if (this.playerDecisionTicksRemaining <= 0) {
        this.playerDirection = this.playerController.selectAction(this);
        this.playerDecisionTicksRemaining = Math.round(PLAYER_DECISION_INTERVAL / FIXED_TIMESTEP);
      }
      movement = directions[this.playerDirection];
    }
    this.move(this.player, movement, dt, playerMultiplier);
    this.playerVelocity = this.velocityFor(movement, this.player.speed * playerMultiplier);
    if (this.playerController !== null) this.playerDecisionTicksRemaining -= 1;

    if (this.enemyDecisionRemaining <= 0) {
      this.enemyDirection = this.enemyController?.selectAction(this) ?? this.chooseBaselineAction();
      this.enemyDecisionRemaining = ENEMY_DECISION_INTERVAL;
    }
    this.move(this.enemy, directions[this.enemyDirection], dt, this.surgeRemaining > 0 ? 0.78 : 1);
    this.enemyDecisionRemaining -= dt;

    const pelletsCollected = this.pelletSlots.filter((slot) => slot.active && Math.hypot(slot.point.x - this.player.x, slot.point.y - this.player.y) <= 18).map((slot) => slot.point);
    if (pelletsCollected.length) {
      this.score += 10 * this.multiplier * pelletsCollected.length;
      this.multiplier = Math.min(5, this.multiplier + pelletsCollected.length);
      this.comboRemaining = 2.4;
      for (const slot of this.pelletSlots) if (pelletsCollected.includes(slot.point)) slot.active = false;
      this.pellets = this.pelletSlots.filter((slot) => slot.active).map((slot) => slot.point);
    }
    const powerCollected = this.orbSlots.filter((slot) => slot.active && Math.hypot(slot.point.x - this.player.x, slot.point.y - this.player.y) <= 22).map((slot) => slot.point);
    if (powerCollected.length) {
      this.score += 75 * this.multiplier * powerCollected.length;
      this.surgeRemaining = 4;
      for (const slot of this.orbSlots) if (powerCollected.includes(slot.point)) slot.active = false;
      this.powerOrbs = this.orbSlots.filter((slot) => slot.active).map((slot) => slot.point);
    }
    this.timeRemaining -= dt;
    return { captured: Math.hypot(this.player.x - this.enemy.x, this.player.y - this.enemy.y) < this.player.radius + this.enemy.radius, cleared: this.pellets.length === 0, timedOut: this.timeRemaining <= 0, pelletsCollected, powerCollected };
  }

  /** Vector observation and swept-path action mask for a player policy. */
  observe(): PlayerObservation {
    const normalized = (point: Point): [number, number] => [(point.x - this.arena.center.x) / this.arena.radius, (point.y - this.arena.center.y) / this.arena.radius];
    const features: number[] = [
      ...normalized(this.player), this.playerVelocity.x / this.player.speed, this.playerVelocity.y / this.player.speed, this.surgeRemaining / 4,
      ...normalized(this.enemy), ...actionOrder.map((direction) => direction === this.enemyDirection ? 1 : 0), this.enemyDecisionFraction,
    ];
    for (const bar of this.arena.bars) features.push(...normalized(bar.from), ...normalized(bar.to));
    for (const slot of this.pelletSlots) features.push(...normalized(slot.point), slot.active ? 1 : 0);
    for (const slot of this.orbSlots) features.push(...normalized(slot.point), slot.active ? 1 : 0);
    features.push(Math.max(0, this.timeRemaining) / 60);
    if (features.length !== PLAYER_OBSERVATION_SIZE) throw new Error(`Player observation mismatch: expected ${PLAYER_OBSERVATION_SIZE}, got ${features.length}.`);
    return { features, actionMask: this.validActions() };
  }

  /** Actions safe for the player's next 100 ms decision interval. */
  validActions(): boolean[] {
    return actionOrder.map((direction) => this.canCompletePlayerDecision(directions[direction]));
  }

  /** Dry-run ten ticks of player motion with mid-interval Surge expiry and Orb pickups. */
  private canCompletePlayerDecision(input: Point): boolean {
    const player: Actor = { x: this.player.x, y: this.player.y, radius: this.player.radius, speed: this.player.speed };
    let surge = this.surgeRemaining;
    const orbs = this.orbSlots.map((slot) => slot.active);
    const ticks = Math.round(PLAYER_DECISION_INTERVAL / FIXED_TIMESTEP);
    for (let tick = 0; tick < ticks; tick += 1) {
      surge = Math.max(0, surge - FIXED_TIMESTEP);
      const multiplier = surge > 0 ? 1.32 : 1;
      if (!this.canTravel(player, input, FIXED_TIMESTEP, multiplier)) return false;
      const length = Math.hypot(input.x, input.y);
      if (length > 0) {
        const travel = player.speed * multiplier * FIXED_TIMESTEP;
        player.x += input.x / length * travel;
        player.y += input.y / length * travel;
      }
      for (let index = 0; index < this.orbSlots.length; index += 1) {
        if (orbs[index] && Math.hypot(this.orbSlots[index].point.x - player.x, this.orbSlots[index].point.y - player.y) <= 22) {
          orbs[index] = false;
          surge = 4;
        }
      }
    }
    return true;
  }

  private mergeResult(target: StepResult, next: StepResult): void {
    target.captured ||= next.captured; target.cleared ||= next.cleared; target.timedOut ||= next.timedOut;
    target.pelletsCollected.push(...next.pelletsCollected); target.powerCollected.push(...next.powerCollected);
  }

  private canTravel(actor: Actor, input: Point, duration: number, speedMultiplier = 1): boolean {
    const length = Math.hypot(input.x, input.y);
    if (length === 0) return true;
    const distance = actor.speed * speedMultiplier * duration;
    const steps = Math.max(1, Math.ceil(distance / 4));
    for (let step = 1; step <= steps; step += 1) {
      const candidate = { x: actor.x + input.x / length * distance * step / steps, y: actor.y + input.y / length * distance * step / steps };
      if (isBlocked(this.arena, candidate, actor.radius)) return false;
    }
    return true;
  }

  private move(actor: Actor, input: Point, dt: number, speedMultiplier = 1): void {
    const length = Math.hypot(input.x, input.y);
    if (length === 0) return;
    const speed = actor.speed * speedMultiplier;
    const steps = Math.max(1, Math.ceil(speed * dt / 4));
    const dx = input.x / length * speed * dt / steps;
    const dy = input.y / length * speed * dt / steps;
    for (let index = 0; index < steps; index += 1) {
      const candidate = { x: actor.x + dx, y: actor.y + dy };
      if (isBlocked(this.arena, candidate, actor.radius)) return;
      actor.x = candidate.x; actor.y = candidate.y;
    }
  }

  private velocityFor(input: Point, speed: number): Point {
    const length = Math.hypot(input.x, input.y);
    return length === 0 ? { x: 0, y: 0 } : { x: input.x / length * speed, y: input.y / length * speed };
  }

  private chooseBaselineAction(): Direction {
    const choices = actionOrder.filter((direction) => this.canTravel(this.enemy, directions[direction], ENEMY_DECISION_INTERVAL));
    if (choices.length === 0) return this.enemyDirection;
    choices.sort((a, b) => {
      const av = directions[a]; const bv = directions[b];
      return Math.hypot(this.player.x - (this.enemy.x + av.x * 42), this.player.y - (this.enemy.y + av.y * 42)) - Math.hypot(this.player.x - (this.enemy.x + bv.x * 42), this.player.y - (this.enemy.y + bv.y * 42));
    });
    return choices[0];
  }
}
