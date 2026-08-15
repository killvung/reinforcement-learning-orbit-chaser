import Phaser from 'phaser';
import { GameSimulation, directions } from './simulation.js';
import { Point } from './arena.js';
import { ArcadeAudio } from './audio.js';
import { updateDebugPanel } from '../debugPanel.js';
import { SarsaPlayerController } from './SarsaPlayerController.js';

const TRAINED_PLAYER_URL = '/models/linear-sarsa-20260815-030554-ep8000.json';

const palette = { background: 0x071426, arena: 0x0b2941, outline: 0x84dcff, wall: 0x2b86bb, wallEdge: 0x82dcff, pellet: 0xf5dc83, player: 0xffda45, enemy: 0xfa637d };
type TrailDot = Point & { radius: number; alpha: number; color: number };
type Spark = Point & { vx: number; vy: number; life: number; color: number };

export class PlayScene extends Phaser.Scene {
  private simulation = new GameSimulation();
  private arenaGraphics!: Phaser.GameObjects.Graphics;
  private actorGraphics!: Phaser.GameObjects.Graphics;
  private hud!: Phaser.GameObjects.Text;
  private message!: Phaser.GameObjects.Text;
  private status!: Phaser.GameObjects.Text;
  private keys!: Record<'up' | 'down' | 'left' | 'right' | 'w' | 'a' | 's' | 'd', Phaser.Input.Keyboard.Key>;
  private touchDirection: Point = { x: 0, y: 0 };
  private finished = false;
  private playerTrail: TrailDot[] = [];
  private enemyTrail: TrailDot[] = [];
  private sparks: Spark[] = [];
  private audio = new ArcadeAudio();
  private nextDangerSound = 0;
  private trainedPlayer: SarsaPlayerController | null = null;
  private trainedEnabled = false;

  constructor() { super('play'); }

  create(): void {
    this.arenaGraphics = this.add.graphics();
    this.actorGraphics = this.add.graphics();
    this.hud = this.add.text(28, 24, '', { fontFamily: 'system-ui', fontSize: '18px', color: '#e8f5ff' });
    this.add.text(28, 52, 'Collect every pellet. Avoid the red enemy.', { fontFamily: 'system-ui', fontSize: '14px', color: '#a6c8da' });
    this.status = this.add.text(400, 52, '', { fontFamily: 'system-ui', fontSize: '14px', color: '#ffd971' }).setOrigin(0.5, 0);
    this.message = this.add.text(400, 318, '', { fontFamily: 'system-ui', fontSize: '34px', fontStyle: 'bold', color: '#ffffff', align: 'center' }).setOrigin(0.5).setDepth(3);
    this.keys = this.input.keyboard!.addKeys({ up: 'UP', down: 'DOWN', left: 'LEFT', right: 'RIGHT', w: 'W', a: 'A', s: 'S', d: 'D' }) as Record<'up' | 'down' | 'left' | 'right' | 'w' | 'a' | 's' | 'd', Phaser.Input.Keyboard.Key>;
    this.input.keyboard!.on('keydown', () => this.audio.unlock());
    this.input.keyboard!.addKey('R').on('down', () => this.restart());
    this.createTouchPad();
    this.bindPlayerToggle();
    this.restart();
    void this.loadTrainedPlayer();
  }

  update(_: number, delta: number): void {
    if (this.finished) return;
    const input = this.currentInput();
    const result = this.simulation.step(delta / 1000, input);
    for (const pellet of result.pelletsCollected) { this.burst(pellet, palette.pellet, 7); this.audio.pellet(); }
    for (const orb of result.powerCollected) { this.burst(orb, 0x91efff, 18); this.audio.surge(); }
    this.playerTrail.unshift({ ...this.simulation.player, radius: 9, alpha: 0.42, color: palette.player });
    this.enemyTrail.unshift({ ...this.simulation.enemy, radius: 9, alpha: 0.3, color: palette.enemy });
    this.playerTrail = this.playerTrail.slice(0, 12).map((dot) => ({ ...dot, alpha: dot.alpha * 0.84 }));
    this.enemyTrail = this.enemyTrail.slice(0, 10).map((dot) => ({ ...dot, alpha: dot.alpha * 0.82 }));
    this.sparks = this.sparks.map((spark) => ({ ...spark, x: spark.x + spark.vx * delta / 1000, y: spark.y + spark.vy * delta / 1000, life: spark.life - delta / 1000 })).filter((spark) => spark.life > 0);
    const danger = Math.hypot(this.simulation.player.x - this.simulation.enemy.x, this.simulation.player.y - this.simulation.enemy.y) < 105;
    if (danger && this.time.now >= this.nextDangerSound) { this.audio.danger(); this.nextDangerSound = this.time.now + 650; }
    if (result.captured) { this.audio.captured(); this.endRound('Captured!'); }
    else if (result.cleared) { this.audio.cleared(); this.endRound('Arena cleared!'); }
    else if (result.timedOut) { this.audio.timeout(); this.endRound('Time up!'); }
    this.render();
  }

  private restart(): void {
    this.audio.unlock(); this.finished = false; this.touchDirection = { x: 0, y: 0 }; this.playerTrail = []; this.enemyTrail = []; this.sparks = []; this.nextDangerSound = 0;
    this.simulation.reset(Math.floor(Math.random() * 0xffffffff));
    this.message.setText(''); this.render();
  }

  private endRound(text: string): void {
    this.finished = true;
    this.message.setText(`${text}\nScore ${this.simulation.score}\n\nPress R or Restart`);
  }

  private currentInput(): Point {
    let x = this.touchDirection.x;
    let y = this.touchDirection.y;
    if (this.keys.left.isDown || this.keys.a.isDown) x -= 1;
    if (this.keys.right.isDown || this.keys.d.isDown) x += 1;
    if (this.keys.up.isDown || this.keys.w.isDown) y -= 1;
    if (this.keys.down.isDown || this.keys.s.isDown) y += 1;
    return { x, y };
  }

  private render(): void {
    const { arena, pellets, powerOrbs, player, enemy, timeRemaining, score, multiplier, comboRemaining, surgeRemaining } = this.simulation;
    const g = this.arenaGraphics.clear();
    g.fillStyle(palette.arena).fillCircle(arena.center.x, arena.center.y, arena.radius);
    g.lineStyle(4, palette.outline).strokeCircle(arena.center.x, arena.center.y, arena.radius);
    g.lineStyle(1, palette.outline, 0.15).strokeCircle(arena.center.x, arena.center.y, 155);
    for (const bar of arena.bars) {
      g.lineStyle(bar.width, palette.wall).lineBetween(bar.from.x, bar.from.y, bar.to.x, bar.to.y);
      g.lineStyle(2, palette.wallEdge).lineBetween(bar.from.x, bar.from.y, bar.to.x, bar.to.y);
    }
    g.fillStyle(palette.wall).fillCircle(arena.center.x, arena.center.y, arena.coreRadius);
    g.lineStyle(2, palette.wallEdge).strokeCircle(arena.center.x, arena.center.y, arena.coreRadius);
    g.fillStyle(palette.pellet); for (const pellet of pellets) g.fillCircle(pellet.x, pellet.y, 4);
    const pulse = 9 + Math.sin(this.time.now / 120) * 2;
    for (const orb of powerOrbs) {
      g.fillStyle(0x72dfff, 0.16).fillCircle(orb.x, orb.y, pulse + 6);
      g.fillStyle(0x91efff).fillCircle(orb.x, orb.y, pulse);
      g.lineStyle(2, 0xffffff, 0.8).strokeCircle(orb.x, orb.y, pulse);
    }
    const actors = this.actorGraphics.clear();
    for (const dot of this.enemyTrail) actors.fillStyle(dot.color, dot.alpha).fillCircle(dot.x, dot.y, dot.radius);
    for (const dot of this.playerTrail) actors.fillStyle(dot.color, dot.alpha).fillCircle(dot.x, dot.y, dot.radius);
    for (const spark of this.sparks) actors.fillStyle(spark.color, Math.min(1, spark.life * 2)).fillCircle(spark.x, spark.y, 3);
    if (surgeRemaining > 0) actors.fillStyle(0x91efff, 0.22).fillCircle(player.x, player.y, player.radius + 13 + Math.sin(this.time.now / 90) * 3);
    actors.fillStyle(palette.player).fillCircle(player.x, player.y, player.radius);
    actors.lineStyle(2, 0xfff0a9).strokeCircle(player.x, player.y, player.radius);
    actors.fillStyle(palette.enemy).fillCircle(enemy.x, enemy.y, enemy.radius);
    actors.lineStyle(2, 0xffb2be).strokeCircle(enemy.x, enemy.y, enemy.radius);
    this.hud.setText(`Score ${score.toString().padStart(4, '0')}   •   Pellets ${pellets.length}   •   Time ${Math.max(0, Math.ceil(timeRemaining))}   •   Player ${this.trainedEnabled ? 'Trained' : 'Human'}   •   Enemy ${this.simulation.enemyControllerName}`);
    if (surgeRemaining > 0) this.status.setText(`SURGE ×${multiplier} · ${surgeRemaining.toFixed(1)}s`);
    else if (comboRemaining > 0 && multiplier > 1) this.status.setText(`STREAK ×${multiplier} · ${comboRemaining.toFixed(1)}s`);
    else if (Math.hypot(player.x - enemy.x, player.y - enemy.y) < 105) this.status.setText('DANGER ZONE');
    else this.status.setText('');
    updateDebugPanel(this.simulation);
  }

  private createTouchPad(): void {
    const positions: Record<keyof typeof directions, Point> = {
      'up-left': { x: 638, y: 492 }, up: { x: 688, y: 492 }, 'up-right': { x: 738, y: 492 },
      left: { x: 638, y: 542 }, right: { x: 738, y: 542 },
      'down-left': { x: 638, y: 592 }, down: { x: 688, y: 592 }, 'down-right': { x: 738, y: 592 },
    };
    for (const [direction, position] of Object.entries(positions) as [keyof typeof directions, Point][]) {
      const button = this.add.circle(position.x, position.y, 25, 0x173a55, 0.95).setStrokeStyle(1, 0x6baed0).setInteractive({ useHandCursor: true });
      this.add.text(position.x, position.y - 2, { up: '↑', 'up-right': '↗', right: '→', 'down-right': '↘', down: '↓', 'down-left': '↙', left: '←', 'up-left': '↖' }[direction], { fontFamily: 'system-ui', fontSize: '21px', color: '#e8f5ff' }).setOrigin(0.5);
      button.on('pointerdown', () => { this.audio.unlock(); this.touchDirection = directions[direction]; });
      button.on('pointerup', () => { this.touchDirection = { x: 0, y: 0 }; });
      button.on('pointerout', () => { this.touchDirection = { x: 0, y: 0 }; });
    }
    const restart = this.add.text(52, 584, 'Restart', { fontFamily: 'system-ui', fontSize: '17px', color: '#e8f5ff', backgroundColor: '#173a55', padding: { x: 12, y: 7 } }).setInteractive({ useHandCursor: true });
    restart.on('pointerdown', () => this.restart());
    const sound = this.add.text(165, 584, 'Sound: On', { fontFamily: 'system-ui', fontSize: '17px', color: '#e8f5ff', backgroundColor: '#173a55', padding: { x: 12, y: 7 } }).setInteractive({ useHandCursor: true });
    sound.on('pointerdown', () => { this.audio.unlock(); this.audio.enabled = !this.audio.enabled; sound.setText(`Sound: ${this.audio.enabled ? 'On' : 'Off'}`); });
  }

  private bindPlayerToggle(): void {
    document.getElementById('player-toggle')?.addEventListener('click', () => this.toggleTrainedPlayer());
  }

  private async loadTrainedPlayer(): Promise<void> {
    try {
      this.trainedPlayer = await SarsaPlayerController.load(TRAINED_PLAYER_URL);
      const toggle = document.getElementById('player-toggle');
      if (toggle instanceof HTMLButtonElement) toggle.disabled = false;
    } catch (error) {
      console.error('Could not load the trained player.', error);
    }
  }

  private toggleTrainedPlayer(): void {
    if (this.trainedPlayer === null) return;
    this.trainedEnabled = !this.trainedEnabled;
    this.simulation.setPlayerController(this.trainedEnabled ? this.trainedPlayer : null);
    const toggle = document.getElementById('player-toggle');
    if (toggle) toggle.textContent = this.trainedEnabled ? 'Player: Trained' : 'Player: Human';
  }

  private burst(point: Point, color: number, amount: number): void {
    for (let index = 0; index < amount; index += 1) {
      const angle = Phaser.Math.FloatBetween(0, Math.PI * 2);
      const speed = Phaser.Math.Between(45, 150);
      this.sparks.push({ x: point.x, y: point.y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed, life: Phaser.Math.FloatBetween(0.35, 0.8), color });
    }
  }
}
