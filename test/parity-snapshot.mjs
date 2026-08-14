import { readFileSync } from 'node:fs';
import { GameSimulation, directions } from '../.test-build/game/simulation.js';

const fixtureUrl = new URL('../rl/parity/fixtures/movement.json', import.meta.url);
const fixture = JSON.parse(readFileSync(fixtureUrl));
const simulation = new GameSimulation(fixture.seed);
let events = { captured: false, cleared: false, timedOut: false };
const snapshots = [];

function snapshot(tick) {
  return {
    tick,
    player: { x: simulation.player.x, y: simulation.player.y },
    enemy: { x: simulation.enemy.x, y: simulation.enemy.y },
    enemy_direction: simulation.enemyDirection,
    enemy_decision_fraction: simulation.enemyDecisionFraction,
    surge_remaining: simulation.surgeRemaining,
    time_remaining: simulation.timeRemaining,
    bars: simulation.arena.bars.map((bar) => ({
      from: { x: bar.from.x, y: bar.from.y },
      to: { x: bar.to.x, y: bar.to.y },
      width: bar.width,
    })),
    pellet_slots: simulation.pelletSlots.map((slot) => ({ x: slot.point.x, y: slot.point.y })),
    orb_slots: simulation.orbSlots.map((slot) => ({ x: slot.point.x, y: slot.point.y })),
    pellet_active: simulation.pelletSlots.map((slot) => slot.active),
    orb_active: simulation.orbSlots.map((slot) => slot.active),
    events: {
      captured: events.captured,
      cleared: events.cleared,
      timed_out: events.timedOut,
    },
  };
}

for (let tick = 0; tick <= fixture.actions.length; tick += 1) {
  if (fixture.snapshot_ticks.includes(tick)) snapshots.push(snapshot(tick));
  if (tick < fixture.actions.length) {
    events = simulation.stepFixed(directions[fixture.actions[tick]]);
  }
}

process.stdout.write(JSON.stringify({ seed: fixture.seed, snapshots }) + '\n');
