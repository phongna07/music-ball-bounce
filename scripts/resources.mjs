/**
 * Sizes a CPU render to this machine's free CPU and RAM.
 *
 * Rendering is done in software: SwiftShader rasterises WebGL behind ANGLE, inside
 * Chrome's GPU process. A browser has exactly one GPU process, shared by all its tabs,
 * and it tops out at a handful of cores. So parallelism comes from running several
 * browsers ("workers"), each with a couple of tabs to keep its GPU process busy.
 *
 * RAM per worker = BROWSER (Chrome, its GPU process, the frame encoder)
 *                + TABS x TAB (one copy of the scene, shadow maps and half-float
 *                  post-processing buffers each).
 * The worker count is the lower of what the cores and what the RAM free right now
 * allow, so the render never pushes the machine into swap or the OOM killer.
 *
 * Environment overrides (all optional):
 *   MBB_WORKERS             browsers to run in parallel (skips CPU/RAM detection)
 *   MBB_TABS                tabs per browser
 *   MBB_CORES_PER_WORKER    cores one browser can keep busy
 *   MBB_BROWSER_MEM_MB      RAM per browser at full 1080x1920, excluding tabs
 *   MBB_TAB_MEM_MB          RAM per tab at full 1080x1920
 *   MBB_RESERVE_MEM_MB      RAM left free for the OS and everything else
 */

import fs from "node:fs";
import os from "node:os";

const MB = 1024 * 1024;

// Measured on this project at 1080x1920 with --gl=swangle: ~950 MB per one-tab browser
// including its encoder, ~1.2 fps per browser, no gain past 4 cores each (see README).
export const DEFAULTS = {
  tabs: 1,
  coresPerWorker: 4,
  browserMemMb: 800,
  tabMemMb: 500,
  reserveMemMb: 4096,
};

export const envNumber = (name, fallback) => {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) throw new Error(`${name} must be a positive number, got "${raw}"`);
  return n;
};

const readFile = (path) => {
  try {
    return fs.readFileSync(path, "utf8").trim();
  } catch {
    return null;
  }
};

/** Cores this process may use: CPU affinity, then any cgroup (container) CPU quota. */
export const detectCpus = () => {
  let cpus = typeof os.availableParallelism === "function" ? os.availableParallelism() : os.cpus().length;
  const v2 = readFile("/sys/fs/cgroup/cpu.max"); // "max 100000" or "<quota> <period>"
  if (v2) {
    const [quota, period] = v2.split(/\s+/);
    if (quota !== "max" && Number(period) > 0) cpus = Math.min(cpus, Number(quota) / Number(period));
  } else {
    const quota = Number(readFile("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"));
    const period = Number(readFile("/sys/fs/cgroup/cpu/cpu.cfs_period_us"));
    if (quota > 0 && period > 0) cpus = Math.min(cpus, quota / period);
  }
  return Math.max(1, Math.floor(cpus));
};

/**
 * RAM usable right now without swapping, in MB: the kernel's MemAvailable (free plus
 * reclaimable cache), capped by any cgroup memory limit. It is what is free at render
 * time, not the total, so programs already running keep their memory.
 */
export const detectAvailableMemMb = () => {
  let available = os.freemem();
  const m = readFile("/proc/meminfo")?.match(/^MemAvailable:\s+(\d+) kB/m);
  if (m) available = Number(m[1]) * 1024;

  const cgroups = [
    ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"], // v2
    ["/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"], // v1
  ];
  for (const [limitPath, usagePath] of cgroups) {
    const limit = Number(readFile(limitPath)); // NaN for "max"
    const usage = Number(readFile(usagePath));
    // v1 reports "no limit" as a huge number, so ignore limits beyond physical RAM
    if (limit > 0 && limit < os.totalmem() && Number.isFinite(usage)) {
      available = Math.min(available, limit - usage);
      break;
    }
  }
  return Math.max(0, Math.floor(available / MB));
};

/**
 * @param {{
 *   scale?: number,
 *   frames?: number,
 *   config?: {
 *     workers?: number,
 *     tabs?: number,
 *     coresPerWorker?: number,
 *     browserMemMb?: number,
 *     tabMemMb?: number,
 *     reserveMemMb?: number,
 *   },
 * }} opts  output scale, frame count, and optional pre-resolved config overrides
 */
export const planRender = ({ scale = 1, frames = Infinity, config = {} } = {}) => {
  const cpus = detectCpus();
  const availableMemMb = detectAvailableMemMb();

  const tabs = Math.floor(config.tabs ?? envNumber("MBB_TABS", DEFAULTS.tabs));
  const coresPerWorker = config.coresPerWorker ?? envNumber("MBB_CORES_PER_WORKER", DEFAULTS.coresPerWorker);
  const reserveMemMb = config.reserveMemMb ?? envNumber("MBB_RESERVE_MEM_MB", DEFAULTS.reserveMemMb);
  // Buffers shrink with pixel count; Chrome's own footprint does not.
  const pixels = Math.max(0.35, scale * scale);
  const browserMemMb = (config.browserMemMb ?? envNumber("MBB_BROWSER_MEM_MB", DEFAULTS.browserMemMb)) * Math.max(0.6, pixels);
  const tabMemMb = (config.tabMemMb ?? envNumber("MBB_TAB_MEM_MB", DEFAULTS.tabMemMb)) * pixels;
  const workerMemMb = Math.ceil(browserMemMb + tabs * tabMemMb);

  const byCpu = Math.max(1, Math.floor(cpus / coresPerWorker));
  const byMem = Math.max(1, Math.floor((availableMemMb - reserveMemMb) / workerMemMb));
  const byFrames = Math.max(1, Math.ceil(frames / MIN_CHUNK));

  const forced =
    config.workers !== undefined
      ? Math.floor(config.workers)
      : process.env.MBB_WORKERS
        ? Math.floor(envNumber("MBB_WORKERS", 1))
        : null;
  const workers = forced ?? Math.min(byCpu, byMem, byFrames);
  const limitedBy = forced
    ? config.workers !== undefined
      ? "config"
      : "MBB_WORKERS"
    : workers === byMem && byMem < byCpu
      ? "RAM"
      : workers === byCpu
        ? "CPU"
        : "length";

  // Each chunk's x264 encoder gets its share of the cores (it would otherwise take them all).
  const encoderThreads = Math.max(1, Math.min(4, Math.floor(cpus / workers)));

  return { workers, tabs, encoderThreads, cpus, availableMemMb, workerMemMb, reserveMemMb, byCpu, byMem, limitedBy };
};

/** Frames per chunk never drop below this, so browser start-up stays a small share. */
export const MIN_CHUNK = 120;
