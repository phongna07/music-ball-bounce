import React, { useMemo } from "react";
import * as THREE from "three";
import { mulberry32 } from "../choreo";

const TEX_WORLD = 12; // world units covered by one texture repeat
const TEX_PX = 1024;
const WALL_SIZE = 96;

type Rect = { x: number; y: number; w: number; h: number; shade: number };

/** Irregular stone panels that tile seamlessly in both directions. */
const layoutPanels = (): Rect[] => {
  const rand = mulberry32(7);
  const rects: Rect[] = [];
  const rows = [0.22, 0.3, 0.2, 0.28];
  let y = 0;
  for (const rh of rows) {
    const h = rh * TEX_PX;
    // each row starts at its own offset so seams never line up across rows
    const shift = rand() * TEX_PX;
    const start = rects.length;
    let x = 0;
    while (x < TEX_PX) {
      let w = (0.22 + rand() * 0.2) * TEX_PX;
      if (TEX_PX - (x + w) < 0.16 * TEX_PX) w = TEX_PX - x;
      rects.push({ x, y, w, h, shade: rand() });
      x += w;
    }
    // shift the row and duplicate it one period to the left, so it still wraps
    const row = rects.splice(start);
    for (const r of row) {
      rects.push({ ...r, x: r.x + shift }, { ...r, x: r.x + shift - TEX_PX });
    }
    y += h;
  }
  return rects;
};

const makeTextures = () => {
  const rects = layoutPanels();
  const rand = mulberry32(21);
  const canvas = (draw: (ctx: CanvasRenderingContext2D) => void, srgb: boolean) => {
    const el = document.createElement("canvas");
    el.width = TEX_PX;
    el.height = TEX_PX;
    draw(el.getContext("2d")!);
    const tex = new THREE.CanvasTexture(el);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(WALL_SIZE / TEX_WORLD, WALL_SIZE / TEX_WORLD);
    tex.anisotropy = 8;
    if (srgb) tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  };
  const speckle = (ctx: CanvasRenderingContext2D, count: number, alpha: number) => {
    for (let i = 0; i < count; i++) {
      const v = Math.floor(rand() * 255);
      ctx.fillStyle = `rgba(${v},${v},${v},${alpha})`;
      const s = 1 + rand() * 3;
      ctx.fillRect(rand() * TEX_PX, rand() * TEX_PX, s, s);
    }
  };

  const map = canvas((ctx) => {
    for (const r of rects) {
      const v = 46 + Math.floor(r.shade * 14);
      ctx.fillStyle = `rgb(${v},${v},${v + 3})`;
      ctx.fillRect(r.x, r.y, r.w, r.h);
    }
    speckle(ctx, 9000, 0.035);
    ctx.strokeStyle = "rgb(9,9,11)";
    ctx.lineWidth = 5;
    for (const r of rects) ctx.strokeRect(r.x, r.y, r.w, r.h);
  }, true);

  const bump = canvas((ctx) => {
    ctx.fillStyle = "rgb(200,200,200)";
    ctx.fillRect(0, 0, TEX_PX, TEX_PX);
    speckle(ctx, 14000, 0.06);
    ctx.strokeStyle = "rgb(0,0,0)";
    ctx.lineWidth = 7;
    for (const r of rects) ctx.strokeRect(r.x, r.y, r.w, r.h);
  }, false);

  const rough = canvas((ctx) => {
    for (const r of rects) {
      const v = 120 + Math.floor(r.shade * 70);
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(r.x, r.y, r.w, r.h);
    }
    speckle(ctx, 16000, 0.12);
  }, false);

  return { map, bump, rough };
};

export const Wall: React.FC<{ camY: number }> = ({ camY }) => {
  const tex = useMemo(makeTextures, []);
  // The plane follows the camera in whole texture periods, so the pattern stays put.
  const y = Math.round(camY / TEX_WORLD) * TEX_WORLD;
  return (
    <mesh position={[0, y, 0]} receiveShadow>
      <planeGeometry args={[WALL_SIZE, WALL_SIZE]} />
      <meshStandardMaterial
        map={tex.map}
        bumpMap={tex.bump}
        bumpScale={1.6}
        roughnessMap={tex.rough}
        roughness={1}
        metalness={0.15}
      />
    </mesh>
  );
};
