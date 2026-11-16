import React, { useLayoutEffect, useMemo } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import {
  BlendFunction,
  BloomEffect,
  Effect,
  EffectComposer,
  EffectPass,
  RenderPass,
  ToneMappingEffect,
  ToneMappingMode,
  VignetteEffect,
} from "postprocessing";
import { HalfFloatType, Uniform, Vector2 } from "three";

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
 * Bloom, tone mapping, vignette and grain.
 *
 * The composer is built synchronously and drawn from the one render hook, so the
 * very first frame a browser tab renders is already complete. (The React wrapper
 * from @react-three/postprocessing needs extra React passes before it draws,
 * which leaves that first frame black in a Remotion render.)
 */
export const Effects: React.FC<{ frame: number; pulse: number }> = ({ frame, pulse }) => {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const camera = useThree((s) => s.camera);

  const fx = useMemo(() => {
    const composer = new EffectComposer(gl, { frameBufferType: HalfFloatType, multisampling: 4 });
    const bloom = new BloomEffect({
      mipmapBlur: true,
      luminanceThreshold: 1.0,
      luminanceSmoothing: 0.3,
      intensity: 1,
      radius: 0.8,
    });
    const grain = new GrainEffect();
    composer.addPass(new RenderPass(scene, camera));
    composer.addPass(
      new EffectPass(
        camera,
        bloom,
        new ToneMappingEffect({ mode: ToneMappingMode.ACES_FILMIC }),
        new VignetteEffect({ offset: 0.3, darkness: 0.45 }),
        grain,
      ),
    );
    return { composer, bloom, grain, size: new Vector2(-1, -1), ratio: -1 };
  }, [gl, scene, camera]);

  useLayoutEffect(() => () => fx.composer.dispose(), [fx]);

  fx.bloom.intensity = 0.8 + 0.6 * pulse;
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
    fx.composer.render(0);
  }, 1);

  return null;
};
