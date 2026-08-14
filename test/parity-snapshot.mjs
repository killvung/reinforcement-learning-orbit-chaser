import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { GameSimulation, directions } from '../.test-build/game/simulation.js';

const defaultFixtureUrl = new URL('../rl/parity/fixtures/movement.json', import.meta.url);
const PARITY_BAR_WIDTH = 18;
const EMPTY_EVENTS = { captured: false, cleared: false, timedOut: false, pelletsCollected: [], powerCollected: [] };
const fixturePath = process.argv[2] ? resolve(process.argv[2]) : defaultFixtureUrl;
const fixture = JSON.parse(readFileSync(fixturePath));
const simulation = new GameSimulation(fixture.seed);
let events = EMPTY_EVENTS;
const snapshots = [];

/** Expand readable direction/tick segments into one fixed-tick input per action. */
function fixtureActions({ actions, segments }) {
  if (segments !== undefined) {
    if (!Array.isArray(segments)) throw new Error('Fixture segments must be an array.');
    return segments.flatMap(({ direction, ticks }) => {
      if (!(direction in directions)) throw new Error(`Unknown fixture direction: ${direction}`);
      if (!Number.isInteger(ticks) || ticks < 1) throw new Error(`Fixture ticks must be a positive integer: ${ticks}`);
      return Array(ticks).fill(direction);
    });
  }
  if (!Array.isArray(actions)) throw new Error('Fixture must contain actions or segments.');
  return actions;
}

const actions = fixtureActions(fixture);

/** Apply a minimal, explicit state override for an event-focused fixture. */
function applySetup(setup = {}) {
  const point = ([x, y]) => ({ x, y });
  if (setup.player) Object.assign(simulation.player, point(setup.player));
  if (setup.enemy) Object.assign(simulation.enemy, point(setup.enemy));
  if (setup.time_remaining !== undefined) simulation.timeRemaining = setup.time_remaining;
  if (setup.surge_remaining !== undefined) simulation.surgeRemaining = setup.surge_remaining;
  if (setup.bars) {
    simulation.arena.bars = setup.bars.map(([from, to]) => ({ from: point(from), to: point(to), width: PARITY_BAR_WIDTH }));
  }
  simulation.pelletSlots = setupSlots(setup.pellet_slots, setup.pellet_active, simulation.pelletSlots, point);
  simulation.orbSlots = setupSlots(setup.orb_slots, setup.orb_active, simulation.orbSlots, point);
  simulation.pellets = simulation.pelletSlots.filter((slot) => slot.active).map((slot) => slot.point);
  simulation.powerOrbs = simulation.orbSlots.filter((slot) => slot.active).map((slot) => slot.point);
}

/** Replace optional fixture slots and validate any corresponding active flags. */
function setupSlots(positions, active, existingSlots, point) {
  const slots = positions ? positions.map((position) => ({ point: point(position), active: true })) : existingSlots;
  if (active === undefined) return slots;
  if (!Array.isArray(active) || active.length !== slots.length) throw new Error('Fixture active flags must match their slot count.');
  return slots.map((slot, index) => ({ ...slot, active: active[index] }));
}

applySetup(fixture.setup);

function snapshot(tick) {
  return {
    tick,
    player: { x: simulation.player.x, y: simulation.player.y },
    player_velocity: { x: simulation.playerVelocity.x, y: simulation.playerVelocity.y },
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
      pellets_collected: events.pelletsCollected.length,
      orbs_collected: events.powerCollected.length,
      captured: events.captured,
      cleared: events.cleared,
      timed_out: events.timedOut,
    },
  };
}

for (let tick = 0; tick <= actions.length; tick += 1) {
  if (fixture.snapshot_ticks.includes(tick)) snapshots.push(snapshot(tick));
  if (tick < actions.length) {
    events = simulation.stepFixed(directions[actions[tick]]);
  }
}

process.stdout.write(JSON.stringify({ seed: fixture.seed, snapshots }) + '\n');
