import assert from 'node:assert/strict';
import test from 'node:test';
import { GameSimulation, PLAYER_OBSERVATION_SIZE } from '../.test-build/game/simulation.js';

const snapshot = (simulation) => ({
  player: { ...simulation.player },
  enemy: { ...simulation.enemy },
  pellets: simulation.pelletSlots.map(({ point, active }) => ({ ...point, active })),
  orbs: simulation.orbSlots.map(({ point, active }) => ({ ...point, active })),
  timeRemaining: simulation.timeRemaining,
  enemyDirection: simulation.enemyDirection,
});

test('reset reproduces a complete seeded episode start', () => {
  const first = new GameSimulation(73);
  const second = new GameSimulation(73);
  assert.deepEqual(snapshot(first), snapshot(second));
});

test('browser elapsed-time adapter equals repeated fixed physics ticks', () => {
  const viaElapsed = new GameSimulation(101);
  const viaFixed = new GameSimulation(101);
  viaElapsed.step(0.1, { x: 1, y: 0 });
  for (let tick = 0; tick < 10; tick += 1) viaFixed.stepFixed({ x: 1, y: 0 });
  assert.deepEqual(snapshot(viaElapsed), snapshot(viaFixed));
});

test('player observation has stable shape and detects an immediately blocked direction', () => {
  const simulation = new GameSimulation(211);
  const observation = simulation.observe();
  assert.equal(observation.features.length, PLAYER_OBSERVATION_SIZE);
  assert.equal(observation.actionMask.length, 8);
  simulation.player.x = simulation.arena.center.x + simulation.arena.radius - simulation.player.radius - 1;
  simulation.player.y = simulation.arena.center.y;
  const nearWall = simulation.observe().actionMask;
  assert.equal(nearWall[2], false, 'right is blocked by the arena boundary');
  assert.equal(nearWall.some(Boolean), true, 'the mask retains at least one safe direction');
});

test('action mask uses remaining Surge duration instead of a constant 100 ms', () => {
  const simulation = new GameSimulation(211);
  simulation.player.x = simulation.arena.center.x + simulation.arena.radius - simulation.player.radius - 20;
  simulation.player.y = simulation.arena.center.y;
  simulation.surgeRemaining = 0.02;
  assert.equal(simulation.observe().actionMask[2], true, 'right remains safe after Surge expires mid-interval');
  simulation.surgeRemaining = 1;
  assert.equal(simulation.observe().actionMask[2], false, 'right is blocked for a full 100 ms of Surge');
});
