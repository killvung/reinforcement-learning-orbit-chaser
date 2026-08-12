export type Point = { x: number; y: number };
export type Segment = { from: Point; to: Point; width: number };

export type Arena = {
  center: Point;
  radius: number;
  coreRadius: number;
  bars: Segment[];
};

export function seededRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let result = value;
    result = Math.imul(result ^ (result >>> 15), result | 1);
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeArena(seed: number): Arena {
  const random = seededRandom(seed);
  const center = { x: 400, y: 322 };
  const angle = random() * Math.PI * 2;
  const unit = { x: Math.cos(angle), y: Math.sin(angle) };
  const perpendicular = { x: -unit.y, y: unit.x };
  const makeBar = (sign: number): Segment => {
    const offset = 92 * sign;
    const length = 112 + random() * 38;
    return {
      from: {
        x: center.x + unit.x * offset - perpendicular.x * length / 2,
        y: center.y + unit.y * offset - perpendicular.y * length / 2,
      },
      to: {
        x: center.x + unit.x * offset + perpendicular.x * length / 2,
        y: center.y + unit.y * offset + perpendicular.y * length / 2,
      },
      width: 18,
    };
  };
  return { center, radius: 242, coreRadius: 36, bars: [makeBar(-1), makeBar(1)] };
}

export function pointSegmentDistance(point: Point, segment: Segment): number {
  const dx = segment.to.x - segment.from.x;
  const dy = segment.to.y - segment.from.y;
  const denominator = dx * dx + dy * dy;
  const t = denominator === 0 ? 0 : Math.max(0, Math.min(1, ((point.x - segment.from.x) * dx + (point.y - segment.from.y) * dy) / denominator));
  return Math.hypot(point.x - (segment.from.x + t * dx), point.y - (segment.from.y + t * dy));
}

export function isBlocked(arena: Arena, point: Point, bodyRadius: number): boolean {
  if (Math.hypot(point.x - arena.center.x, point.y - arena.center.y) > arena.radius - bodyRadius) return true;
  if (Math.hypot(point.x - arena.center.x, point.y - arena.center.y) < arena.coreRadius + bodyRadius) return true;
  return arena.bars.some((bar) => pointSegmentDistance(point, bar) < bar.width / 2 + bodyRadius);
}
