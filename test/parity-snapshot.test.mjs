import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import test from 'node:test';

const root = new URL('..', import.meta.url);
const fixture = new URL('../rl/parity/fixtures/movement-diagonal.json', import.meta.url);
const longFixture = new URL('../rl/parity/fixtures/movement-long.json', import.meta.url);

test('parity snapshot executables accept compact movement segments', () => {
  const typescript = execFileSync('node', ['test/parity-snapshot.mjs', fixture.pathname], {
    cwd: root,
    encoding: 'utf8',
  });
  const python = execFileSync('.venv/bin/python', ['rl/parity/python_snapshot.py', fixture.pathname], {
    cwd: root,
    encoding: 'utf8',
    env: { ...process.env, PYTHONPATH: 'rl' },
  });

  const expectedTicks = [0, 1, 10, 20, 30];
  assert.deepEqual(JSON.parse(typescript).snapshots.map(({ tick }) => tick), expectedTicks);
  assert.deepEqual(JSON.parse(python).snapshots.map(({ tick }) => tick), expectedTicks);
});

test('long movement fixture snapshots turns and enemy decision rollovers', () => {
  const output = execFileSync('node', ['test/parity-snapshot.mjs', longFixture.pathname], {
    cwd: root,
    encoding: 'utf8',
  });
  const ticks = JSON.parse(output).snapshots.map(({ tick }) => tick);

  assert.deepEqual(ticks, [0, 1, 27, 28, 29, 39, 40, 41, 55, 56, 57, 79, 80, 81, 83, 84, 85, 111, 112, 113, 119, 120, 121, 139, 140, 141, 167, 168, 169, 195, 196, 197, 199, 200, 201, 223, 224, 225, 239, 240, 241, 251, 252, 253, 279, 280, 281, 319, 320]);
});

test('Python long-fixture snapshots preserve their initial collectible flags', () => {
  const output = execFileSync('.venv/bin/python', ['rl/parity/python_snapshot.py', longFixture.pathname], {
    cwd: root,
    encoding: 'utf8',
    env: { ...process.env, PYTHONPATH: 'rl' },
  });
  const initial = JSON.parse(output).snapshots[0];

  assert.ok(initial.pellet_active.every(Boolean));
  assert.ok(initial.orb_active.every(Boolean));
});
