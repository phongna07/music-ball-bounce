/**
 * Machine-aware render configuration loader (shared by scripts/render.mjs and remotion.config.ts).
 *
 * Precedence (highest to lowest):
 *   1. CLI flags            (--renderer, --gl, --concurrency, --workers, --tabs)
 *   2. Shell environment    (MBB_RENDERER, MBB_GL, MBB_CONCURRENCY, MBB_WORKERS, ...)
 *   3. Local .env file      (gitignored repo-root .env for per-machine overrides)
 *   4. render.config.json   (optional hostname -> mode/settings map, plus "default")
 *   5. Built-in defaults    (renderer: "gpu", gl: "angle" for gpu / "swangle" for cpu)
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const findRepoRoot = (startDir = process.cwd()) => {
  let dir = path.resolve(startDir);
  while (true) {
    if (fs.existsSync(path.join(dir, "package.json"))) return dir;
    const parent = path.dirname(dir);
    if (parent === dir) return path.resolve(startDir);
    dir = parent;
  }
};

export const VALID_RENDERERS = new Set(["gpu", "cpu"]);

/** Parse a .env file into a key->value map without mutating process.env. */
export const parseDotEnv = (text) => {
  const out = {};
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const m = line.match(/^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
    if (!m) continue;
    const [, key, rawVal] = m;
    let val = rawVal.trim();
    if (
      (val.startsWith('"') && val.endsWith('"') && val.length >= 2) ||
      (val.startsWith("'") && val.endsWith("'") && val.length >= 2)
    ) {
      val = val.slice(1, -1);
    } else {
      const hash = val.indexOf(" #");
      if (hash !== -1) val = val.slice(0, hash).trim();
    }
    out[key] = val;
  }
  return out;
};

const readDotEnv = (rootDir) => {
  const envPath = path.join(rootDir, ".env");
  if (!fs.existsSync(envPath)) return {};
  return parseDotEnv(fs.readFileSync(envPath, "utf8"));
};

const readMachineConfigFile = (rootDir) => {
  const cfgPath = path.join(rootDir, "render.config.json");
  if (!fs.existsSync(cfgPath)) return null;
  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(cfgPath, "utf8"));
  } catch (err) {
    throw new Error(`Invalid JSON in ${cfgPath}: ${err.message}`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${cfgPath} must contain a JSON object`);
  }
  return parsed;
};

const normalizeProfile = (entry, label) => {
  if (entry === undefined || entry === null) return {};
  if (typeof entry === "string") return { renderer: entry };
  if (typeof entry === "object" && !Array.isArray(entry)) return { ...entry };
  throw new Error(`Invalid profile for ${label} in render.config.json (expected "gpu", "cpu", or an object)`);
};

const matchesHostPattern = (pattern, host) => {
  const p = pattern.toLowerCase();
  const h = host.toLowerCase();
  if (p === h) return true;
  if (!p.includes("*")) return false;
  const escaped = p.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*");
  return new RegExp(`^${escaped}$`).test(h);
};

const findMachineProfile = (machines, host) => {
  if (!machines || typeof machines !== "object" || Array.isArray(machines)) return null;
  const shortHost = host.split(".")[0];
  const entries = Object.entries(machines);

  // Exact match on full hostname first, then short hostname, then wildcard patterns.
  for (const candidate of [host, shortHost]) {
    for (const [key, val] of entries) {
      if (!key.includes("*") && key.toLowerCase() === candidate.toLowerCase()) {
        return { key, profile: normalizeProfile(val, `machines.${key}`) };
      }
    }
  }
  for (const [key, val] of entries) {
    if (key.includes("*") && (matchesHostPattern(key, host) || matchesHostPattern(key, shortHost))) {
      return { key, profile: normalizeProfile(val, `machines.${key}`) };
    }
  }
  return null;
};

const parsePositiveNumber = (raw, label) => {
  if (raw === undefined || raw === null || raw === "") return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) throw new Error(`${label} must be a positive number, got "${raw}"`);
  return n;
};

const pickEnv = (env, dotenv, name) => {
  if (env[name] !== undefined && env[name] !== "") return { value: env[name], source: `env:${name}` };
  if (dotenv[name] !== undefined && dotenv[name] !== "") return { value: dotenv[name], source: `.env:${name}` };
  return { value: undefined, source: null };
};

/**
 * Resolve the effective render configuration for the current machine.
 *
 * @param {{
 *   cli?: { renderer?: string, gl?: string, concurrency?: string|number, workers?: string|number, tabs?: string|number },
 *   env?: Record<string, string|undefined>,
 *   rootDir?: string,
 *   hostname?: string,
 * }} [opts]
 */
export const resolveRenderConfig = ({ cli = {}, env = process.env, rootDir = findRepoRoot(), hostname } = {}) => {
  const dotenv = readDotEnv(rootDir);
  const fileCfg = readMachineConfigFile(rootDir);

  const machineEnv = pickEnv(env, dotenv, "MBB_MACHINE");
  const machine = (hostname || machineEnv.value || os.hostname()).trim();

  const fileDefault = fileCfg ? normalizeProfile(fileCfg.default, "default") : {};
  const matchedMachine = fileCfg ? findMachineProfile(fileCfg.machines, machine) : null;
  const fileProfile = { ...fileDefault, ...(matchedMachine?.profile ?? {}) };

  const resolveString = (cliVal, cliFlag, envName, fileKey) => {
    if (cliVal !== undefined && cliVal !== "") return { value: String(cliVal).trim(), source: cliFlag };
    const fromEnv = pickEnv(env, dotenv, envName);
    if (fromEnv.value !== undefined) return { value: String(fromEnv.value).trim(), source: fromEnv.source };
    if (matchedMachine?.profile?.[fileKey] !== undefined && matchedMachine.profile[fileKey] !== "") {
      return {
        value: String(matchedMachine.profile[fileKey]).trim(),
        source: `render.config.json (machines.${matchedMachine.key})`,
      };
    }
    if (fileDefault[fileKey] !== undefined && fileDefault[fileKey] !== "") {
      return { value: String(fileDefault[fileKey]).trim(), source: "render.config.json (default)" };
    }
    return { value: undefined, source: null };
  };

  const resolveNumber = (cliVal, cliFlag, envName, fileKey) => {
    if (cliVal !== undefined && cliVal !== "") return parsePositiveNumber(cliVal, cliFlag);
    const fromEnv = pickEnv(env, dotenv, envName);
    if (fromEnv.value !== undefined) return parsePositiveNumber(fromEnv.value, envName);
    if (fileProfile[fileKey] !== undefined && fileProfile[fileKey] !== "") {
      return parsePositiveNumber(fileProfile[fileKey], `render.config.json:${fileKey}`);
    }
    return undefined;
  };

  const rendererPick = resolveString(cli.renderer, "--renderer", "MBB_RENDERER", "renderer");
  const renderer = (rendererPick.value ?? "gpu").toLowerCase();
  const rendererSource = rendererPick.source ?? "default";
  if (!VALID_RENDERERS.has(renderer)) {
    throw new Error(`Invalid renderer "${renderer}" from ${rendererSource} (expected "gpu" or "cpu")`);
  }

  const defaultGl = renderer === "cpu" ? "swangle" : "angle";
  const fileRenderer = (fileProfile.renderer ?? "gpu").toLowerCase();
  const glFromCliOrEnv =
    cli.gl !== undefined && cli.gl !== ""
      ? { value: String(cli.gl).trim(), source: "--gl" }
      : pickEnv(env, dotenv, "MBB_GL");
  const glPick =
    glFromCliOrEnv.value !== undefined
      ? glFromCliOrEnv
      : renderer === fileRenderer
        ? resolveString(undefined, "", "MBB_GL", "gl")
        : { value: undefined, source: null };
  const gl = glPick.value ?? defaultGl;
  const glSource = glPick.source ?? `default (${renderer})`;

  return {
    machine,
    matchedMachine: matchedMachine?.key ?? null,
    renderer,
    rendererSource,
    gl,
    glSource,
    concurrency: resolveNumber(cli.concurrency, "--concurrency", "MBB_CONCURRENCY", "concurrency"),
    workers: resolveNumber(cli.workers, "--workers", "MBB_WORKERS", "workers"),
    tabs: resolveNumber(cli.tabs, "--tabs", "MBB_TABS", "tabs"),
    coresPerWorker: resolveNumber(undefined, "", "MBB_CORES_PER_WORKER", "coresPerWorker"),
    browserMemMb: resolveNumber(undefined, "", "MBB_BROWSER_MEM_MB", "browserMemMb"),
    tabMemMb: resolveNumber(undefined, "", "MBB_TAB_MEM_MB", "tabMemMb"),
    reserveMemMb: resolveNumber(undefined, "", "MBB_RESERVE_MEM_MB", "reserveMemMb"),
  };
};
