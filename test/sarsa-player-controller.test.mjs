import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import {
  SarsaPlayerController,
  selectGreedyAction,
} from '../.test-build/game/SarsaPlayerController.js';
import { actionOrder } from '../.test-build/game/simulation.js';

const fixturePath = join(dirname(fileURLToPath(import.meta.url)), 'fixtures', 'linear-sarsa-parity.json');
const fixture = JSON.parse(readFileSync(fixturePath, 'utf8'));
const Q_TOLERANCE = 1e-9;

test('fromArtifact matches Python feature indices, Q-values, and greedy actions', () => {
  const controller = SarsaPlayerController.fromArtifact(fixture.artifact);
  assert.equal(fixture.cases.length > 0, true);
  for (const testCase of fixture.cases) {
    const observation = {
      features: testCase.observation,
      actionMask: testCase.action_mask.map(Boolean),
    };
    const indices = controller.encode(observation);
    const qValues = controller.qValues(indices);
    assert.deepEqual(indices, testCase.indices, testCase.name);
    assert.equal(controller.choose(observation), actionOrder[testCase.action], testCase.name);
    assert.equal(qValues.length, testCase.q_values.length, testCase.name);
    qValues.forEach((value, index) => {
      assert.ok(
        Math.abs(value - testCase.q_values[index]) < Q_TOLERANCE,
        `${testCase.name} Q[${index}]: ${value} vs ${testCase.q_values[index]}`,
      );
    });
  }
});

test('masked greedy selection never returns a masked action', () => {
  const qValues = [9, 8, 7, 6, 5, 4, 3, 2];
  const actionMask = [false, false, false, false, false, true, true, false];
  assert.equal(selectGreedyAction(qValues, actionMask), 5);
});

test('equal valid Q-values choose the lowest valid action index', () => {
  const qValues = [0, 0, 4, 4, 0, 0, 0, 0];
  const actionMask = [false, false, true, true, false, false, false, false];
  assert.equal(selectGreedyAction(qValues, actionMask), 2);
});
