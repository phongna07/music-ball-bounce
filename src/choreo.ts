// Data contract with pipeline/choreograph.py, plus the pure functions of time
// that every visual is derived from. Nothing here keeps state between frames.

export type Segment = { t0: number; x: number; y: number; vx: number; vy: number };

export type Plate = {
  t: number;
  x: number;
  y: number;
  nx: number;
  ny: number;
  len: number;
  hx: number;
  hy: number;
  color: string;
  strength: number;
  speed: number;
  /** Which ball lands on this plate (version 2+). */
  ball?: number;
  /** Snapped video frame index for impact (when frame snapping is enabled). */
  frame?: number;
};

/** One ball: its flight segments, and for each of its hits the index into `Choreo.plates`. */
export type BallPath = { segments: Segment[]; plates: number[] };

export type Choreo = {
  version: number;
  duration: number;
  tempo: number;
  gravity: number;
  ballRadius: number;
  plateThickness: number;
  /** One entry per ball (one or two). Older files only have `segments`; see normalizeChoreo. */
  balls: BallPath[];
  /** Every ball's plates, sorted by hit time. */
  plates: Plate[];
  camera: { fps: number; x: number[]; y: number[]; dist?: number };
  kicks: { t: number; strength: number }[];
  beats: number[];
  energy: { fps: number; rms: number[]; low: number[]; high: number[] };
};

export const BALL_Z = 0.62;
export const PLATE_DEPTH = 0.95;
/** Tolerance (seconds) so exact frame-boundary hits never slip to the next frame from float rounding. */
export const HIT_EPS = 1e-4;

/** Index of the last item whose time is <= t, or -1. */
export const lastIndexAtOrBefore = <T>(items: T[], t: number, time: (item: T) => number): number => {
  let lo = 0;
  let hi = items.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (time(items[mid]) <= t) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
};

/** Accepts version 1 files (a single top-level `segments` list) as one ball. */
export const normalizeChoreo = (raw: Choreo & { segments?: Segment[] }): Choreo => {
  if (raw.balls && raw.balls.length > 0) return raw;
  return { ...raw, balls: [{ segments: raw.segments ?? [], plates: raw.plates.map((_, i) => i) }] };
};

export const ballAt = (c: Choreo, t: number, ball = 0) => {
  const path = c.balls[ball];
  const first = path.segments[0];
  const i = Math.max(0, lastIndexAtOrBefore(path.segments, t + HIT_EPS, (s) => s.t0));
  const s = path.segments[i];
  const tau = Math.max(0, Math.max(t, first.t0) - s.t0);
  return {
    x: s.x + s.vx * tau,
    y: s.y + s.vy * tau - 0.5 * c.gravity * tau * tau,
    vx: s.vx,
    vy: s.vy - c.gravity * tau,
    // segment i starts at this ball's hit i-1; mapped to an index into c.plates (-1 before the first hit)
    lastPlate: i >= 1 ? path.plates[i - 1] : -1,
  };
};

export const sampleCurve = (values: number[], fps: number, t: number): number => {
  if (values.length === 0) return 0;
  const f = Math.min(Math.max(t * fps, 0), values.length - 1);
  const i = Math.floor(f);
  const j = Math.min(i + 1, values.length - 1);
  return values[i] + (values[j] - values[i]) * (f - i);
};

/** Decaying pulse from recent kicks, roughly 0..1. */
export const kickPulse = (c: Choreo, t: number): number => {
  let i = lastIndexAtOrBefore(c.kicks, t, (k) => k.t);
  let sum = 0;
  while (i >= 0 && t - c.kicks[i].t < 0.6) {
    sum += c.kicks[i].strength * Math.exp(-(t - c.kicks[i].t) * 9);
    i--;
  }
  return Math.min(sum, 1.2);
};

/** Small deterministic PRNG so sparks and textures are identical on every render. */
export const mulberry32 = (seed: number) => {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let x = Math.imul(a ^ (a >>> 15), 1 | a);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
};
