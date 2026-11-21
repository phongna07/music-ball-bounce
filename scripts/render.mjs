#!/usr/bin/env node
/**
 * Flexible Remotion renderer supporting both GPU (default Remotion CLI) and
 * parallel multi-browser CPU rendering (SwiftShader).
 *
 *   node scripts/render.mjs <output.mp4> [--renderer=gpu|cpu] [--gl=angle|swangle|...]
 *        [--props=<json or file>] [--scale=1] [--frames=0-299]
 *        [--concurrency=N] [--workers=N] [--tabs=N] [--composition=Bounce]
 *   node scripts/render.mjs --show-config [--scale=1] [--renderer=gpu|cpu]
 *
 * Renderer selection (highest precedence first):
 *   --renderer flag > MBB_RENDERER env > repo .env > render.config.json > default ("gpu")
 *
 * In "gpu" mode, delegates to Remotion's standard single-browser CLI renderer.
 * In "cpu" mode, bundles the project once, starts several browsers sized to free
 * CPU and RAM (scripts/resources.mjs), renders chunks in parallel, and joins them
 * with the soundtrack via ffmpeg.
 */

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { bundle } from "@remotion/bundler";
import { openBrowser, renderMedia, selectComposition } from "@remotion/renderer";
import { resolveRenderConfig } from "./config.mjs";
import { MIN_CHUNK, planRender } from "./resources.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const CRF = 22;
const TIMEOUT_MS = 120_000;
const VALUE_FLAGS = new Set([
  "renderer",
  "gl",
  "props",
  "scale",
  "frames",
  "concurrency",
  "workers",
  "tabs",
  "composition",
]);

// ---- arguments --------------------------------------------------------------
const flags = {};
const positional = [];
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  const arg = argv[i];
  const m = arg.match(/^--([^=]+)(?:=(.*))?$/);
  if (m) {
    const [, name, val] = m;
    if (val !== undefined) {
      flags[name] = val;
    } else if (VALUE_FLAGS.has(name) && i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
      flags[name] = argv[++i];
    } else {
      flags[name] = "true";
    }
  } else {
    positional.push(arg);
  }
}

const scale = Number(flags.scale ?? 1);
if (!Number.isFinite(scale) || scale <= 0) {
  console.error(`--scale must be a positive number, got "${flags.scale}"`);
  process.exit(2);
}

const config = resolveRenderConfig({
  rootDir: ROOT,
  cli: {
    renderer: flags.renderer,
    gl: flags.gl,
    concurrency: flags.concurrency,
    workers: flags.workers,
    tabs: flags.tabs,
  },
});

if (flags["show-config"]) {
  const summary = {
    ...config,
    ...(config.renderer === "cpu" ? { cpuPlan: planRender({ scale, config }) } : {}),
  };
  console.log(JSON.stringify(summary, null, 2));
  process.exit(0);
}

if (positional.length !== 1) {
  console.error(
    "Usage: node scripts/render.mjs <output.mp4> [--renderer=gpu|cpu] [--gl=...] [--props=...] " +
      "[--scale=1] [--frames=a-b] [--concurrency=N] [--workers=N] [--tabs=N]\n" +
      "       node scripts/render.mjs --show-config",
  );
  process.exit(2);
}

const output = path.resolve(positional[0]);
const compositionId = flags.composition ?? "Bounce";

// ---- GPU mode (default Remotion single-browser CLI) -------------------------
if (config.renderer === "gpu") {
  if (flags.workers || flags.tabs) {
    console.warn("[render] note: --workers/--tabs only apply to cpu mode; use --concurrency in gpu mode");
  }
  const require = createRequire(import.meta.url);
  const remotionCli = path.join(path.dirname(require.resolve("@remotion/cli/package.json")), "remotion-cli.js");
  const cliArgs = [remotionCli, "render", compositionId, output, `--gl=${config.gl}`, `--scale=${scale}`];
  if (flags.props) {
    const propsArg = fs.existsSync(flags.props) ? path.resolve(flags.props) : flags.props;
    cliArgs.push(`--props=${propsArg}`);
  }
  if (flags.frames) cliArgs.push(`--frames=${flags.frames}`);
  if (config.concurrency !== undefined) cliArgs.push(`--concurrency=${Math.floor(config.concurrency)}`);

  fs.mkdirSync(path.dirname(output), { recursive: true });
  console.log(
    `[render] mode=gpu, gl=${config.gl} (from ${config.rendererSource}), machine=${config.machine}` +
      (config.concurrency !== undefined ? `, concurrency=${Math.floor(config.concurrency)}` : ""),
  );
  const res = spawnSync(process.execPath, cliArgs, {
    cwd: ROOT,
    stdio: "inherit",
    env: {
      ...process.env,
      MBB_RENDERER: "gpu",
      MBB_GL: config.gl,
    },
  });
  if (res.error) {
    console.error(res.error);
    process.exit(1);
  }
  process.exit(res.status ?? 1);
}

// ---- CPU mode (multi-browser SwiftShader parallel render) -------------------
if (flags.concurrency) {
  console.warn("[render] note: --concurrency only applies to gpu mode; use --workers/--tabs in cpu mode");
}

let inputProps = {};
if (flags.props) {
  inputProps = JSON.parse(fs.existsSync(flags.props) ? fs.readFileSync(flags.props, "utf8") : flags.props);
}

const ffmpeg = (args) => {
  const r = spawnSync("ffmpeg", ["-v", "error", "-y", ...args], { stdio: "inherit" });
  if (r.error) throw new Error(`ffmpeg not found on PATH (${r.error.message})`);
  if (r.status !== 0) throw new Error(`ffmpeg exited with ${r.status}`);
};

const fmt = (s) => `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, "0")}s`;

// Every browser registers its own exit handlers; more than 10 is expected here.
process.setMaxListeners(0);
const t0 = Date.now();
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "mbb-render-"));
const browsers = [];
let exitCode = 0;
try {
  console.log(`[render] mode=cpu, gl=${config.gl} (from ${config.rendererSource}), machine=${config.machine}`);
  console.log("[render] bundling");
  const serveUrl = await bundle({ entryPoint: path.join(ROOT, "src", "index.ts"), rspack: true, symlinkPublicDir: true });

  const chromiumOptions = { gl: config.gl };
  browsers.push(await openBrowser("chrome", { chromiumOptions, logLevel: "error" }));
  const composition = await selectComposition({
    serveUrl,
    id: compositionId,
    inputProps,
    puppeteerInstance: browsers[0],
    chromiumOptions,
    timeoutInMilliseconds: TIMEOUT_MS,
    logLevel: "error",
  });

  let [first, last] = [0, composition.durationInFrames - 1];
  if (flags.frames) {
    const m = flags.frames.match(/^(\d+)(?:-(\d+))?$/);
    if (!m) throw new Error(`--frames must look like 0-299, got "${flags.frames}"`);
    first = Number(m[1]);
    last = Math.min(last, Number(m[2] ?? m[1]));
  }
  const total = last - first + 1;

  const plan = planRender({ scale, frames: total, config });
  console.log(
    `[render] ${total} frames, gl=${config.gl}: ${plan.workers} browsers x ${plan.tabs} tabs ` +
      `(${plan.cpus} cores allow ${plan.byCpu}; ${plan.availableMemMb} MB free RAM at ` +
      `~${plan.workerMemMb} MB each allows ${plan.byMem}; limited by ${plan.limitedBy})`,
  );

  // Several chunks per browser, so a browser that finishes early takes more work.
  const chunkSize = Math.max(MIN_CHUNK, Math.ceil(total / (plan.workers * 3)));
  const chunks = [];
  for (let a = first; a <= last; a += chunkSize) {
    chunks.push({ range: [a, Math.min(last, a + chunkSize - 1)], file: path.join(tmp, `chunk-${chunks.length}.mp4`), done: 0 });
  }

  while (browsers.length < Math.min(plan.workers, chunks.length)) {
    browsers.push(await openBrowser("chrome", { chromiumOptions, logLevel: "error" }));
  }

  const common = { composition, serveUrl, inputProps, chromiumOptions, timeoutInMilliseconds: TIMEOUT_MS, logLevel: "error", overwrite: true };

  // The soundtrack, rendered once for the whole range; it needs no WebGL frames.
  const audioFile = path.join(tmp, "audio.wav");
  const audioJob = renderMedia({ ...common, codec: "wav", frameRange: [first, last], outputLocation: audioFile });

  let next = 0;
  const progress = setInterval(() => {
    const done = chunks.reduce((n, c) => n + c.done, 0);
    const secs = (Date.now() - t0) / 1000;
    console.log(`[render] ${done}/${total} frames (${Math.round((100 * done) / total)}%), ${fmt(secs)} elapsed`);
  }, 10_000);

  const worker = async (browser) => {
    while (next < chunks.length) {
      const chunk = chunks[next++];
      await renderMedia({
        ...common,
        puppeteerInstance: browser,
        codec: "h264",
        crf: CRF,
        imageFormat: "jpeg",
        muted: true,
        scale,
        concurrency: plan.tabs,
        frameRange: chunk.range,
        outputLocation: chunk.file,
        onProgress: ({ renderedFrames }) => (chunk.done = renderedFrames),
        // x264 otherwise starts ~1.5 threads per core in every encoder; with one
        // encoder per browser that oversubscribes the CPU and each one buffers ~1 GB.
        ffmpegOverride: ({ args }) => {
          const out = args.length - 1;
          return [...args.slice(0, out), "-threads", String(plan.encoderThreads), args[out]];
        },
      });
      chunk.done = chunk.range[1] - chunk.range[0] + 1;
    }
  };
  try {
    await Promise.all([...browsers.map(worker), audioJob]);
  } finally {
    clearInterval(progress);
  }

  console.log("[render] joining chunks");
  const list = path.join(tmp, "chunks.txt");
  fs.writeFileSync(list, chunks.map((c) => `file '${c.file}'`).join("\n"));
  fs.mkdirSync(path.dirname(output), { recursive: true });
  ffmpeg([
    "-f", "concat", "-safe", "0", "-i", list,
    "-i", audioFile,
    "-map", "0:v", "-map", "1:a",
    "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
    "-movflags", "+faststart",
    output,
  ]);
  console.log(`[render] done in ${fmt((Date.now() - t0) / 1000)}: ${output}`);
} catch (err) {
  console.error(err);
  exitCode = 1;
} finally {
  await Promise.allSettled(browsers.map((b) => b.close({ silent: true })));
  fs.rmSync(tmp, { recursive: true, force: true });
}
process.exit(exitCode);
