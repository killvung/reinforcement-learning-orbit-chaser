import { GameSimulation } from './game/simulation';

const panel = document.querySelector<HTMLElement>('#debug-panel');
const content = document.querySelector<HTMLPreElement>('#debug-content');
const toggle = document.querySelector<HTMLButtonElement>('#debug-toggle');
let lastRender = 0;

if (toggle && panel) {
  toggle.addEventListener('click', () => {
    panel.hidden = !panel.hidden;
    toggle.textContent = panel.hidden ? 'Show debug' : 'Hide debug';
    toggle.setAttribute('aria-expanded', String(!panel.hidden));
    lastRender = 0;
  });
}

const point = (x: number, y: number): string => `(${x.toFixed(1)}, ${y.toFixed(1)})`;

export function updateDebugPanel(simulation: GameSimulation): void {
  // Keep a fresh snapshot while hidden, so opening the panel after a capture
  // immediately shows the last live state instead of a placeholder.
  if (!panel || !content || performance.now() - lastRender < 120) return;
  lastRender = performance.now();
  const debug = simulation.enemyDebugState;
  const distance = Math.hypot(simulation.player.x - simulation.enemy.x, simulation.player.y - simulation.enemy.y);
  const fields = [
    ['Controller', simulation.enemyControllerName],
    ['Time remaining', `${Math.max(0, simulation.timeRemaining).toFixed(2)} s`],
    ['Direct distance', `${distance.toFixed(1)} px`],
    ['Player position', point(simulation.player.x, simulation.player.y)],
    ['Player velocity', point(simulation.playerVelocity.x, simulation.playerVelocity.y)],
    ['Enemy position', point(simulation.enemy.x, simulation.enemy.y)],
    ['Enemy direction', simulation.enemyDirection],
    ['Pellets / orbs', `${simulation.pellets.length} / ${simulation.powerOrbs.length}`],
    ['Surge remaining', `${simulation.surgeRemaining.toFixed(2)} s`],
    ...Object.entries(debug).map(([label, value]) => [label, String(value)]),
  ];
  content.textContent = fields.map(([label, value]) => `${label.padEnd(18)} ${value}`).join('\n');
}
