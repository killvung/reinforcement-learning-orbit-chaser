import { Arena, Point, isBlocked, makeArena, seededRandom } from './arena';

export type Direction = 'up' | 'up-right' | 'right' | 'down-right' | 'down' | 'down-left' | 'left' | 'up-left';
export const directions: Record<Direction, Point> = {
  up: { x: 0, y: -1 },
  'up-right': { x: 1, y: -1 },
  right: { x: 1, y: 0 },
  'down-right': { x: 1, y: 1 },
  down: { x: 0, y: 1 },
  'down-left': { x: -1, y: 1 },
  left: { x: -1, y: 0 },
  'up-left': { x: -1, y: -1 },
};
export type Actor = Point & { radius: number; speed: number };
export type StepResult = { captured: boolean; cleared: boolean; timedOut: boolean; pelletsCollected: Point[]; powerCollected: Point[] };
export type CollectibleSlot = { point: Point; active: boolean };
export type EnemyDebugState = Record<string, string | number | boolean>;
export interface EnemyController { readonly name: string; selectAction(simulation: GameSimulation): Direction; getDebugState?(): EnemyDebugState; }

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
  private decisionTime = 0;
  private enemyController: EnemyController | null = null;

  constructor(seed = Date.now()) { this.arena = makeArena(seed); this.player = { x: 0, y: 0, radius: 12, speed: 175 }; this.enemy = { x: 0, y: 0, radius: 13, speed: 110 }; this.reset(seed); }

  reset(seed = Date.now()): void {
    this.arena = makeArena(seed);
    const { center } = this.arena;
    this.player = { x: center.x - 165, y: center.y + 85, radius: 12, speed: 175 };
    this.enemy = { x: center.x + 165, y: center.y - 85, radius: 13, speed: 110 };
    this.enemyDirection = 'left'; this.decisionTime = 0; this.timeRemaining = 60; this.pellets = []; this.powerOrbs = []; this.pelletSlots = []; this.orbSlots = []; this.playerVelocity = { x: 0, y: 0 };
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
  get enemyControllerName(): string { return this.enemyController?.name ?? 'Greedy'; }
  get enemyDebugState(): EnemyDebugState { return this.enemyController?.getDebugState?.() ?? { Navigation: 'Built-in greedy pursuit' }; }

  step(dt: number, playerInput: Point): StepResult {
    const cappedDt = Math.min(dt, 0.04);
    this.comboRemaining = Math.max(0, this.comboRemaining - cappedDt);
    if (this.comboRemaining === 0) this.multiplier = 1;
    this.surgeRemaining = Math.max(0, this.surgeRemaining - cappedDt);
    this.move(this.player, playerInput, cappedDt, this.surgeRemaining > 0 ? 1.32 : 1);
    this.playerVelocity = this.velocityFor(playerInput, this.player.speed * (this.surgeRemaining > 0 ? 1.32 : 1));
    this.decisionTime -= cappedDt;
    if (this.decisionTime <= 0) { this.enemyDirection = this.enemyController?.selectAction(this) ?? this.chooseBaselineAction(); this.decisionTime = 0.28; }
    this.move(this.enemy, directions[this.enemyDirection], cappedDt, this.surgeRemaining > 0 ? 0.78 : 1);
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
    this.timeRemaining -= cappedDt;
    return { captured: Math.hypot(this.player.x - this.enemy.x, this.player.y - this.enemy.y) < this.player.radius + this.enemy.radius, cleared: this.pellets.length === 0, timedOut: this.timeRemaining <= 0, pelletsCollected, powerCollected };
  }

  private move(actor: Actor, input: Point, dt: number, speedMultiplier = 1): void {
    const length = Math.hypot(input.x, input.y);
    if (length === 0) return;
    const speed = actor.speed * speedMultiplier;
    const steps = Math.max(1, Math.ceil(speed * dt / 4));
    const dx = input.x / length * speed * dt / steps;
    const dy = input.y / length * speed * dt / steps;
    for (let i = 0; i < steps; i += 1) {
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
    const choices = (Object.keys(directions) as Direction[]).filter((direction) => {
      const vector = directions[direction];
      return !isBlocked(this.arena, { x: this.enemy.x + vector.x * 42, y: this.enemy.y + vector.y * 42 }, this.enemy.radius);
    });
    if (choices.length === 0) return this.enemyDirection;
    choices.sort((a, b) => {
      const av = directions[a]; const bv = directions[b];
      return Math.hypot(this.player.x - (this.enemy.x + av.x * 42), this.player.y - (this.enemy.y + av.y * 42)) - Math.hypot(this.player.x - (this.enemy.x + bv.x * 42), this.player.y - (this.enemy.y + bv.y * 42));
    });
    return choices[0];
  }
}
