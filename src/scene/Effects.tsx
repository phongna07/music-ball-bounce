import React, { useLayoutEffect, useMemo } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import {
  BlendFunction,
  BloomEffect,
  ChromaticAberrationEffect,
  Effect,
  EffectComposer,
  EffectPass,
  RenderPass,
  ToneMappingEffect,
  ToneMappingMode,
  VignetteEffect,
} from "postprocessing";
import { Color, HalfFloatType, Uniform, Vector2, Vector3 } from "three";
import { BALL_Z, ScreenEffectState } from "../choreo";

const splashShader = /* glsl */ `
uniform float intensity;
uniform float progress;
uniform float raySeed;
uniform vec2 origin;
uniform vec3 splashColor;

void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  if (intensity <= 0.001) {
    outputColor = inputColor;
    return;
  }
  float aspectRatio = resolution.x / max(resolution.y, 1.0);
  vec2 da = vec2((uv.x - origin.x) * aspectRatio, uv.y - origin.y);
  float dist = length(da);

  // 1. Tight radiant core burst right at the impact point (fast spatial falloff)
  float core = exp(-dist * (10.5 - 2.5 * progress)) * (0.55 - 0.25 * progress);
  vec3 hotCore = mix(splashColor, vec3(1.0), 0.35);

  // 2. Sleek thin feathered neon wavefront ring expanding outward
  float waveRadius = 0.04 + 1.05 * (1.0 - pow(1.0 - progress, 2.8));
  float waveWidth = 0.022 + 0.028 * progress;
  float waveBand = exp(-pow((dist - waveRadius) / waveWidth, 2.0));
  float waveFade = (1.0 - progress) * smoothstep(0.0, 0.05, progress);

  // 3. Sharp, jewel-like starburst flare rays radiating locally from impact
  float angle = atan(da.y, da.x);
  float rays = pow(max(0.0, 0.5 + 0.5 * cos(angle * 7.0 + raySeed)), 7.0) * 0.7
             + pow(max(0.0, 0.5 + 0.5 * cos(angle * 11.0 - raySeed * 1.7)), 8.0) * 0.3;
  float rayEnv = exp(-dist * 4.8) * smoothstep(0.01, 0.09, dist) * (1.0 - 0.55 * progress);

  // 4. Crisp saturated neon screen-border rim flash (hugs outer edges, leaves center dark)
  vec2 edge = abs(uv - 0.5) * 2.0;
  float rim = pow(clamp(max(edge.x, edge.y), 0.0, 1.0), 4.5);
  float rimEnv = 0.16 * rim * (1.0 - 0.55 * progress);

  vec3 splash = hotCore * core
              + splashColor * (waveBand * waveFade * 0.42 + rays * rayEnv * 0.65 + rimEnv);
  outputColor = vec4(inputColor.rgb + splash * intensity, inputColor.a);
}`;

class ScreenSplashEffect extends Effect {
  constructor() {
    super("ScreenSplashEffect", splashShader, {
      blendFunction: BlendFunction.NORMAL,
      uniforms: new Map<string, Uniform>([
        ["intensity", new Uniform(0)],
        ["progress", new Uniform(0)],
        ["raySeed", new Uniform(0)],
        ["origin", new Uniform(new Vector2(0.5, 0.5))],
        ["splashColor", new Uniform(new Color("#ffffff"))],
      ]),
    });
  }
}

const grainShader = /* glsl */ `
uniform float seed;
uniform float amount;
float hash(vec2 p) {
  vec3 q = fract(vec3(p.xyx) * 0.1031 + seed * 0.137);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}
void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  float n = hash(floor(uv * resolution)) - 0.5;
  outputColor = vec4(inputColor.rgb + n * amount, inputColor.a);
}`;

/** Film grain seeded by the frame number, so every render is identical. */
class GrainEffect extends Effect {
  constructor() {
    super("GrainEffect", grainShader, {
      blendFunction: BlendFunction.NORMAL,
      uniforms: new Map<string, Uniform>([
        ["seed", new Uniform(0)],
        ["amount", new Uniform(0.006)],
      ]),
    });
  }
}

/**
 * Chromatic aberration, bloom, accent screen splash, tone mapping, vignette and grain.
 *
 * The composer is built synchronously and drawn from the one render hook, so the
 * very first frame a browser tab renders is already complete. (The React wrapper
 * from @react-three/postprocessing needs extra React passes before it draws,
 * which leaves that first frame black in a Remotion render.)
 */
export const Effects: React.FC<{ frame: number; pulse: number; accentFx: ScreenEffectState }> = ({
  frame,
  pulse,
  accentFx,
}) => {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const camera = useThree((s) => s.camera);

  const fx = useMemo(() => {
    const composer = new EffectComposer(gl, { frameBufferType: HalfFloatType, multisampling: 4 });
    const chroma = new ChromaticAberrationEffect({
      offset: new Vector2(0, 0),
      radialModulation: true,
      modulationOffset: 0.2,
    });
    const bloom = new BloomEffect({
      mipmapBlur: true,
      luminanceThreshold: 1.0,
      luminanceSmoothing: 0.3,
      intensity: 1,
      radius: 0.8,
    });
    const splash = new ScreenSplashEffect();
    const vignette = new VignetteEffect({ offset: 0.3, darkness: 0.45 });
    const grain = new GrainEffect();
    composer.addPass(new RenderPass(scene, camera));
    composer.addPass(
      new EffectPass(
        camera,
        chroma,
        bloom,
        splash,
        new ToneMappingEffect({ mode: ToneMappingMode.ACES_FILMIC }),
        vignette,
        grain,
      ),
    );
    return {
      composer,
      chroma,
      bloom,
      splash,
      vignette,
      grain,
      splashWorld: new Vector3(),
      proj: new Vector3(),
      size: new Vector2(-1, -1),
      ratio: -1,
    };
  }, [gl, scene, camera]);

  useLayoutEffect(() => () => fx.composer.dispose(), [fx]);

  fx.bloom.intensity = 0.8 + 0.6 * pulse + accentFx.bloomBoost;
  fx.vignette.uniforms.get("darkness")!.value = Math.max(0.36, 0.45 - 0.09 * accentFx.splashIntensity);
  fx.chroma.offset.set(0.0045 * accentFx.chroma, 0.0022 * accentFx.chroma);
  fx.splash.uniforms.get("intensity")!.value = accentFx.splashIntensity;
  fx.splash.uniforms.get("progress")!.value = accentFx.splashProgress;
  fx.splash.uniforms.get("raySeed")!.value = accentFx.splashSeed;
  (fx.splash.uniforms.get("splashColor")!.value as Color).set(accentFx.splashColor);
  fx.splashWorld.set(accentFx.splashWorldX, accentFx.splashWorldY, BALL_Z);
  fx.grain.uniforms.get("seed")!.value = frame % 1000;

  const current = useMemo(() => new Vector2(), []);
  useFrame(() => {
    gl.getSize(current);
    const ratio = gl.getPixelRatio();
    if (!current.equals(fx.size) || ratio !== fx.ratio) {
      fx.composer.setSize(current.x, current.y);
      fx.size.copy(current);
      fx.ratio = ratio;
    }
    if ((fx.splash.uniforms.get("intensity")!.value as number) > 0.001) {
      fx.proj.copy(fx.splashWorld).project(camera);
      const u = Math.min(Math.max(fx.proj.x * 0.5 + 0.5, -0.2), 1.2);
      const v = Math.min(Math.max(fx.proj.y * 0.5 + 0.5, -0.2), 1.2);
      (fx.splash.uniforms.get("origin")!.value as Vector2).set(u, v);
    }
    fx.composer.render(0);
  }, 1);

  return null;
};
