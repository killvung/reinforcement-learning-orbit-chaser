import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import test from 'node:test';

const root = new URL('..', import.meta.url);
const fixture = (name) => new URL(`../rl/parity/fixtures/${name}.json`, import.meta.url);

function snapshot(name) {
  const output = execFileSync('node', ['test/parity-snapshot.mjs', fixture(name).pathname], {
    cwd: root,
    encoding: 'utf8',
  });
  return JSON.parse(output).snapshots;
}

test('collision fixtures keep the player blocked by the core and a bar', () => {
  for (const name of ['collision-core', 'collision-bar']) {
    const [before, after] = snapshot(name);
    assert.deepEqual(after.player, before.player, name);
  }
});

test('event fixtures expose terminal flags and pickup counts on the transition tick', () => {
  assert.equal(snapshot('event-capture')[1].events.captured, true);

  const clear = snapshot('event-clear')[1];
  assert.equal(clear.events.cleared, true);
  assert.equal(clear.events.pellets_collected, 1);

  assert.equal(snapshot('event-timeout')[1].events.timed_out, true);

  const precedence = snapshot('event-precedence')[1].events;
  assert.deepEqual(precedence, {
    pellets_collected: 1,
    orbs_collected: 0,
    captured: true,
    cleared: true,
    timed_out: true,
  });
});

test('Surge fixtures capture pickup, multiplier, expiry, and multi-pickup behavior', () => {
  const orb = snapshot('event-orb-surge');
  assert.equal(orb[1].events.orbs_collected, 1);
  assert.equal(orb[1].surge_remaining, 4);
  assert.equal(orb[2].player_velocity.y, -231);

  const expiry = snapshot('event-surge-expiry')[1];
  assert.equal(expiry.surge_remaining, 0);
  assert.equal(expiry.player_velocity.y, -175);

  const multi = snapshot('event-multiple-pickups')[1].events;
  assert.equal(multi.pellets_collected, 2);
  assert.equal(multi.orbs_collected, 2);
});
