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
  /** Musical importance score 0..1 (version 3+ / accent detection). */
  importance?: number;
  /** Selected accent strength (>0 triggers screen splash/shake). */
  accent?: number;
};

export type ScreenEffectMode = "both" | "splash" | "shake" | "off";
export type ScreenEffectConfig = { mode: ScreenEffectMode; strength: number };
export type ScreenEffectState = {
  shakeX: number;
  shakeY: number;
  shakeZoom: number;
  shakeRoll: number;
  splashIntensity: number;
  splashProgress: number;
  splashSeed: number;
  splashColor: string;
  splashWorldX: number;
  splashWorldY: number;
  bloomBoost: number;
  chroma: number;
  lightIntensity: number;
  lightX: number;
  lightY: number;
  lightColor: string;
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
  screenEffect?: ScreenEffectConfig;
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

const clamp = (v: number, lo: number, hi: number): number => Math.min(Math.max(v, lo), hi);

const percentile = (sorted: number[], p: number): number => {
  if (sorted.length === 0) return 0;
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
};

const robustNorm = (values: number[]): number[] => {
  if (values.length <= 1) return values.map(() => 0);
  const sorted = [...values].sort((a, b) => a - b);
  let lo = sorted[0];
  let hi = sorted[sorted.length - 1];
  if (hi - lo < 1e-9) return values.map(() => 0);
  const pLo = percentile(sorted, 15);
  const pHi = percentile(sorted, 95);
  if (pHi - pLo >= 1e-9) {
    lo = pLo;
    hi = pHi;
  }
  const denom = Math.max(hi - lo, 1e-9);
  return values.map((v) => clamp((v - lo) / denom, 0, 1));
};

const annotateLegacyAccents = (c: Choreo): Plate[] => {
  const plates = c.plates;
  const n = plates.length;
  if (n === 0) return plates;

  const rms = c.energy?.rms ?? [];
  const low = c.energy?.low ?? [];
  const efps = c.energy?.fps || 60;
  const preW = Math.max(2, Math.round(0.22 * efps));
  const postW = Math.max(2, Math.round(0.2 * efps));

  const rmsSorted = [...rms].sort((a, b) => a - b);
  const lowSorted = [...low].sort((a, b) => a - b);
  const medRms = rmsSorted.length ? percentile(rmsSorted, 50) : 0;
  const medLow = lowSorted.length ? percentile(lowSorted, 50) : 0;

  const iois: number[] = [];
  for (let i = 1; i < n; i++) iois.push(plates[i].t - plates[i - 1].t);
  iois.sort((a, b) => a - b);
  const medIoi = iois.length ? Math.max(0.15, percentile(iois, 50)) : 0.5;

  const beats = c.beats ?? [];
  const downbeats: number[] = [];
  for (let i = 0; i < beats.length; i += 4) downbeats.push(beats[i]);
  const kicks = c.kicks ?? [];

  const strRaw: number[] = [];
  const spdRaw: number[] = [];
  const rmsRaw: number[] = [];
  const lowRaw: number[] = [];
  const agoRaw: number[] = [];
  const metRaw: number[] = [];

  for (let i = 0; i < n; i++) {
    const p = plates[i];
    const t = p.t;
    const f = Math.round(t * efps);

    strRaw.push(p.strength);
    spdRaw.push(p.speed);

    if (rms.length) {
      let rPre = 0;
      if (f > 1) {
        const a = Math.max(0, f - preW);
        const b = Math.max(1, f - 1);
        for (let j = a; j < b; j++) rPre += rms[j];
        rPre /= Math.max(1, b - a);
      }
      let rPost = 0;
      for (let j = Math.max(0, f); j < Math.min(rms.length, f + postW); j++) {
        if (rms[j] > rPost) rPost = rms[j];
      }
      const baseR = Math.max(rPre, 0.25 * medRms);
      rmsRaw.push(Math.max(0, rPost - baseR) + 0.35 * rPost);
    } else {
      rmsRaw.push(0);
    }

    if (low.length) {
      let lPre = 0;
      if (f > 1) {
        const a = Math.max(0, f - preW);
        const b = Math.max(1, f - 1);
        for (let j = a; j < b; j++) lPre += low[j];
        lPre /= Math.max(1, b - a);
      }
      let lPost = 0;
      for (let j = Math.max(0, f); j < Math.min(low.length, f + postW); j++) {
        if (low[j] > lPost) lPost = low[j];
      }
      const baseL = Math.max(lPre, 0.25 * medLow);
      lowRaw.push(Math.max(0, lPost - baseL) + 0.25 * lPost);
    } else {
      lowRaw.push(0);
    }

    const gapPrev = i > 0 ? plates[i].t - plates[i - 1].t : medIoi;
    const gapNext = i + 1 < n ? plates[i + 1].t - plates[i].t : Math.max(0, c.duration - plates[i].t);
    agoRaw.push(0.65 * Math.min(2.5, gapNext / medIoi) + 0.35 * Math.min(2.0, gapPrev / medIoi));

    let m = 0;
    if (downbeats.some((d) => Math.abs(d - t) <= 0.095)) m += 0.65;
    let bestKick = 0;
    for (const k of kicks) {
      if (Math.abs(k.t - t) <= 0.075 && k.strength > bestKick) bestKick = k.strength;
    }
    m += 0.35 * bestKick;
    metRaw.push(Math.min(1, m));
  }

  const nStr = robustNorm(strRaw);
  const nSpd = robustNorm(spdRaw);
  const nRms = robustNorm(rmsRaw);
  const nLow = robustNorm(lowRaw);
  const nAgo = robustNorm(agoRaw);

  const imps = plates.map((_, i) =>
    clamp(0.25 * nStr[i] + 0.15 * nSpd[i] + 0.24 * nRms[i] + 0.14 * nLow[i] + 0.1 * nAgo[i] + 0.12 * metRaw[i], 0, 1),
  );

  const tempo = c.tempo || 100;
  const beatSec = 60 / clamp(tempo, 50, 180);
  const cooldown = clamp(2.5 * beatSec, 1.5, 2.4);
  const targetMax = Math.max(2, Math.round(c.duration / 3.0));
  const sortedImps = [...imps].sort((a, b) => a - b);
  const thresh = Math.max(0.52, percentile(sortedImps, 78));

  const order = imps.map((_, i) => i).sort((a, b) => imps[b] - imps[a] || plates[a].t - plates[b].t);
  const selected: number[] = [];
  for (const idx of order) {
    if (imps[idx] < thresh || selected.length >= targetMax) break;
    const t = plates[idx].t;
    if (selected.every((s) => Math.abs(t - plates[s].t) >= cooldown)) {
      selected.push(idx);
    }
  }

  const accents = new Array<number>(n).fill(0);
  if (selected.length > 0) {
    const hiImp = Math.max(...selected.map((s) => imps[s]));
    const span = Math.max(hiImp - thresh, 0.12);
    for (const idx of selected) {
      const rel = clamp((imps[idx] - thresh) / span, 0, 1);
      accents[idx] = Number((0.45 + 0.55 * (0.65 * rel + 0.35 * imps[idx])).toFixed(3));
    }
  }

  return plates.map((p, i) => ({
    ...p,
    importance: Number(imps[i].toFixed(3)),
    accent: accents[i],
  }));
};

/** Accepts version 1 files (a single top-level `segments` list) as one ball, and backfills accents on legacy files. */
export const normalizeChoreo = (raw: Choreo & { segments?: Segment[] }): Choreo => {
  const base: Choreo =
    raw.balls && raw.balls.length > 0
      ? raw
      : { ...raw, balls: [{ segments: raw.segments ?? [], plates: raw.plates.map((_, i) => i) }] };
  const hasAccentMeta = base.plates.some((p) => p.accent !== undefined || p.importance !== undefined);
  return {
    ...base,
    screenEffect: base.screenEffect ?? { mode: "both", strength: 1 },
    plates: hasAccentMeta || base.plates.length === 0 ? base.plates : annotateLegacyAccents(base),
  };
};

const VALID_MODES = new Set<ScreenEffectMode>(["both", "splash", "shake", "off"]);

export const resolveScreenEffect = (
  c: Choreo,
  modeOverride?: ScreenEffectMode,
  strengthOverride?: number,
): ScreenEffectConfig => {
  const rawMode = modeOverride ?? c.screenEffect?.mode ?? "both";
  const mode: ScreenEffectMode = VALID_MODES.has(rawMode) ? rawMode : "both";
  const rawStrength = strengthOverride ?? c.screenEffect?.strength ?? 1;
  const strength = Number.isFinite(rawStrength) ? clamp(rawStrength, 0, 2.5) : 1;
  return { mode, strength };
};

const ACCENT_WINDOW = 0.85;

/** Pure function of time driving light splash & camera shake on important note hits. */
export const accentEffectsAt = (c: Choreo, t: number, config: ScreenEffectConfig): ScreenEffectState => {
  const zero: ScreenEffectState = {
    shakeX: 0,
    shakeY: 0,
    shakeZoom: 0,
    shakeRoll: 0,
    splashIntensity: 0,
    splashProgress: 0,
    splashSeed: 0,
    splashColor: "#ffffff",
    splashWorldX: 0,
    splashWorldY: 0,
    bloomBoost: 0,
    chroma: 0,
    lightIntensity: 0,
    lightX: 0,
    lightY: 0,
    lightColor: "#ffffff",
  };
  if (config.mode === "off" || config.strength <= 0 || c.plates.length === 0) return zero;

  const doShake = config.mode === "both" || config.mode === "shake";
  const doSplash = config.mode === "both" || config.mode === "splash";
  let i = lastIndexAtOrBefore(c.plates, t + HIT_EPS, (p) => p.t);
  let bestGlow = 0;

  while (i >= 0 && t - c.plates[i].t < ACCENT_WINDOW) {
    const p = c.plates[i];
    const acc = p.accent ?? 0;
    if (acc > 0) {
      const tau = Math.max(0, t - p.t);
      if (doShake && tau < 0.55) {
        const shakeScale = (config.mode === "shake" ? 1.15 : 1.0) * config.strength * acc;
        const rand = mulberry32(i * 4099 + 17);
        const phaseX = rand() * Math.PI * 2;
        const phaseY = rand() * Math.PI * 2;
        const rollSign = rand() < 0.5 ? -1 : 1;
        const recoilEnv = Math.exp(-18 * tau);
        const kick = 0.22 * recoilEnv * shakeScale;
        const env = Math.exp(-7.5 * tau);
        const amp = 0.42 * shakeScale;
        const oscX =
          0.65 * Math.sin(2 * Math.PI * 13.5 * tau + phaseX) +
          0.35 * Math.sin(2 * Math.PI * 21.3 * tau + phaseX * 1.7);
        const oscY =
          0.65 * Math.sin(2 * Math.PI * 15.8 * tau + phaseY) +
          0.35 * Math.sin(2 * Math.PI * 23.7 * tau + phaseY * 1.3);
        zero.shakeX += -p.nx * kick + amp * env * oscX;
        zero.shakeY += -p.ny * kick + amp * env * oscY;
        zero.shakeRoll += rollSign * 0.022 * shakeScale * Math.exp(-8.5 * tau) * Math.cos(2 * Math.PI * 11.2 * tau);
        zero.shakeZoom += 1.05 * shakeScale * Math.exp(-7.0 * tau) * (0.45 + 0.55 * (1 - Math.exp(-40 * tau)));
        zero.chroma += 0.35 * shakeScale * Math.exp(-16 * tau);
      }
      if (doSplash && tau < 0.85) {
        const splashScale = (config.mode === "splash" ? 1.15 : 1.0) * config.strength * acc;
        const glow = splashScale * Math.exp(-5.8 * tau);
        zero.splashIntensity += glow;
        zero.bloomBoost += 0.75 * splashScale * Math.exp(-7.0 * tau);
        zero.chroma += 0.45 * splashScale * Math.exp(-13 * tau);
        if (glow > bestGlow) {
          bestGlow = glow;
          zero.splashProgress = Math.min(1, tau / 0.62);
          zero.splashColor = p.color;
          zero.splashWorldX = p.hx;
          zero.splashWorldY = p.hy;
          zero.splashSeed = (i * 2.39996) % (Math.PI * 2);
          zero.lightIntensity = 32 * splashScale * Math.exp(-6.5 * tau);
          zero.lightX = p.hx;
          zero.lightY = p.hy;
          zero.lightColor = p.color;
        }
      }
    }
    i--;
  }

  zero.shakeX = clamp(zero.shakeX, -0.95, 0.95);
  zero.shakeY = clamp(zero.shakeY, -0.95, 0.95);
  zero.shakeRoll = clamp(zero.shakeRoll, -0.05, 0.05);
  zero.shakeZoom = clamp(zero.shakeZoom, 0, 1.8);
  zero.splashIntensity = clamp(zero.splashIntensity, 0, 1.8);
  zero.bloomBoost = clamp(zero.bloomBoost, 0, 1.4);
  zero.chroma = clamp(zero.chroma, 0, 1.2);
  return zero;
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
