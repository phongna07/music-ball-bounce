/**
 * Note: When using the Node.JS APIs, the config file
 * doesn't apply. Instead, pass options directly to the APIs.
 *
 * All configuration options: https://remotion.dev/docs/config
 */

import { Config } from "@remotion/cli/config";
import { resolveRenderConfig } from "./scripts/config.mjs";

const renderConfig = resolveRenderConfig();

Config.setRspack(true);
Config.setVideoImageFormat("jpeg");
Config.setOverwriteOutput(true);
Config.setDefaultPremountInSeconds(2);
Config.setCrf(22);

// Resolves from MBB_RENDERER / MBB_GL env, .env, or render.config.json:
// "gpu" defaults to "angle"; "cpu" defaults to "swangle" (SwiftShader).
Config.setChromiumOpenGlRenderer(
  renderConfig.gl as Parameters<typeof Config.setChromiumOpenGlRenderer>[0],
);
if (
  renderConfig.renderer === "cpu" ||
  renderConfig.gl === "swangle" ||
  renderConfig.gl === "swiftshader"
) {
  // Software rendering is slower per frame, especially when CPU is shared.
  Config.setDelayRenderTimeoutInMilliseconds(120_000);
}
