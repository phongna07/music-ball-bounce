import React, { useLayoutEffect, useMemo } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import {
  BALL_Z,
  Choreo,
  PLATE_DEPTH,
  Plate,
  ballAt,
  kickPulse,
  lastIndexAtOrBefore,
  mulberry32,
  sampleCurve,
} from "../choreo";
import { Effects } from "./Effects";
import { Wall } from "./Wall";

const VIEW_RANGE = 15; // plates further than this from the camera centre are skipped
const LIGHT_POOL = 18;
const SPARKS_PER_HIT = 14;
const SPARK_LIFE = 0.75;
const TRAIL_SAMPLES = 36;
const TRAIL_SECONDS = 0.38;

const WHITE = new THREE.Color("#ffffff");
const UNLIT = new THREE.Color("#0c0c0f");

// ---------------------------------------------------------------- camera
const CameraRig: React.FC<{ x: number; y: number; pulse: number; aspect: number }> = ({ x, y, pulse, aspect }) => {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  useLayoutEffect(() => {
    camera.fov = 38;
    camera.aspect = aspect;
    camera.near = 1;
    camera.far = 120;
    // Looking slightly up and across the wall shows the depth of the plates.
    camera.position.set(x + 2.6, y + 5.5, 25.5 - 0.7 * pulse);
    camera.up.set(0, 1, 0);
    camera.lookAt(x, y - 2.2, 0);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
  }, [camera, x, y, pulse, aspect]);
  return null;
};

// ---------------------------------------------------------------- plates
const PlateMesh: React.FC<{ plate: Plate; t: number; thickness: number }> = ({ plate, t, thickness }) => {
  const tau = t - plate.t;
  const lit = tau >= 0;
  const color = useMemo(() => new THREE.Color(plate.color), [plate.color]);
  const angle = Math.atan2(plate.ny, plate.nx) - Math.PI / 2;
  const hitForce = Math.min(1, plate.speed / 18);
  // damped spring recoil along the normal, plus a little twist
  const recoil = lit ? -0.2 * hitForce * Math.exp(-7 * tau) * Math.sin(30 * tau) : 0;
  const twist = lit ? 0.06 * hitForce * Math.exp(-6 * tau) * Math.sin(24 * tau) : 0;
  const glow = lit ? 0.8 + 4.5 * Math.exp(-6.5 * tau) : 0;
  const z = 0.16 + PLATE_DEPTH / 2;
  return (
    <group position={[plate.x + plate.nx * recoil, plate.y + plate.ny * recoil, 0]} rotation={[0, 0, angle + twist]}>
      <mesh position={[0, 0, z]} castShadow receiveShadow>
        <boxGeometry args={[plate.len, thickness, PLATE_DEPTH]} />
        <meshStandardMaterial
          color={lit ? color : UNLIT}
          emissive={color}
          emissiveIntensity={glow}
          roughness={lit ? 0.45 : 0.18}
          metalness={lit ? 0 : 0.6}
        />
      </mesh>
      {[-0.32, 0.32].map((side) => (
        <mesh key={side} position={[plate.len * side, -thickness * 0.2, 0.09]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.045, 0.045, 0.18, 10]} />
          <meshStandardMaterial color="#8d9096" roughness={0.3} metalness={0.9} />
        </mesh>
      ))}
    </group>
  );
};

// ---------------------------------------------------------------- impact extras
const Sparks: React.FC<{ plate: Plate; index: number; t: number; gravity: number }> = ({ plate, index, t, gravity }) => {
  const tau = t - plate.t;
  const sparks = useMemo(() => {
    const rand = mulberry32(index * 7919 + 13);
    const base = Math.atan2(plate.ny, plate.nx);
    return new Array(SPARKS_PER_HIT).fill(0).map(() => {
      const a = base + (rand() - 0.5) * 2.4;
      const speed = (2.5 + rand() * 7) * (0.5 + plate.strength);
      return {
        vx: Math.cos(a) * speed,
        vy: Math.sin(a) * speed,
        vz: (rand() - 0.35) * 3,
        size: 0.03 + rand() * 0.05,
        life: SPARK_LIFE * (0.5 + rand() * 0.5),
      };
    });
  }, [index, plate]);
  const color = useMemo(() => new THREE.Color(plate.color).lerp(WHITE, 0.3).multiplyScalar(2.6), [plate.color]);
  return (
    <>
      {sparks.map((s, k) => {
        if (tau >= s.life) return null;
        const drag = (1 - Math.exp(-2.2 * tau)) / 2.2;
        const fade = 1 - tau / s.life;
        return (
          <mesh
            key={k}
            position={[
              plate.hx + s.vx * drag,
              plate.hy + s.vy * drag - 0.25 * gravity * tau * tau,
              Math.max(0.05, BALL_Z + s.vz * drag),
            ]}
            scale={s.size * fade}
          >
            <icosahedronGeometry args={[1, 0]} />
            <meshBasicMaterial color={color} toneMapped={false} />
          </mesh>
        );
      })}
    </>
  );
};

const Shockwave: React.FC<{ plate: Plate; t: number }> = ({ plate, t }) => {
  const tau = t - plate.t;
  const life = 0.45;
  if (tau < 0 || tau > life) return null;
  const k = tau / life;
  const radius = 0.3 + 1.9 * (1 - (1 - k) ** 3) * (0.6 + plate.strength * 0.6);
  return (
    <mesh position={[plate.hx, plate.hy, 0.02]} scale={radius}>
      <ringGeometry args={[0.965, 1, 64]} />
      <meshBasicMaterial
        color={plate.color}
        transparent
        opacity={(1 - k) ** 2 * 0.45}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        toneMapped={false}
      />
    </mesh>
  );
};

// ---------------------------------------------------------------- ball
const Trail: React.FC<{ choreo: Choreo; t: number }> = ({ choreo, t }) => {
  const geometry = useMemo(() => {
    const positions: number[] = [];
    const colors: number[] = [];
    const indices: number[] = [];
    const tint = new THREE.Color();
    for (let i = 0; i < TRAIL_SAMPLES; i++) {
      const k = i / (TRAIL_SAMPLES - 1);
      const b = ballAt(choreo, t - k * TRAIL_SECONDS);
      const speed = Math.hypot(b.vx, b.vy) || 1;
      const px = -b.vy / speed;
      const py = b.vx / speed;
      const half = choreo.ballRadius * 0.8 * (1 - k) ** 1.3;
      positions.push(b.x + px * half, b.y + py * half, BALL_Z, b.x - px * half, b.y - py * half, BALL_Z);
      tint.set(b.lastPlate >= 0 ? choreo.plates[b.lastPlate].color : "#ffffff");
      const fade = (1 - k) ** 2 * 1.6;
      colors.push(tint.r * fade, tint.g * fade, tint.b * fade, tint.r * fade, tint.g * fade, tint.b * fade);
      if (i > 0) {
        const a = (i - 1) * 2;
        indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    g.setIndex(indices);
    return g;
  }, [choreo, t]);
  useLayoutEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <mesh geometry={geometry} frustumCulled={false}>
      <meshBasicMaterial
        vertexColors
        transparent
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        side={THREE.DoubleSide}
        toneMapped={false}
      />
    </mesh>
  );
};

const Ball: React.FC<{ choreo: Choreo; t: number }> = ({ choreo, t }) => {
  const b = ballAt(choreo, t);
  const last = b.lastPlate >= 0 ? choreo.plates[b.lastPlate] : null;
  const sinceHit = last ? t - last.t : 10;
  const tint = useMemo(() => new THREE.Color(), []);
  tint.set(last ? last.color : "#ffffff").lerp(WHITE, 0.45);

  // squash against the plate right after a hit, stretch along the velocity otherwise
  let angle: number;
  let along: number;
  if (last && sinceHit < 0.09) {
    angle = Math.atan2(last.ny, last.nx);
    along = 1 - 0.32 * (1 - sinceHit / 0.09) * Math.min(1, last.speed / 14);
  } else {
    angle = Math.atan2(b.vy, b.vx);
    along = 1 + Math.min(0.22, Math.hypot(b.vx, b.vy) * 0.007);
  }
  const across = 1 / Math.sqrt(along);
  const ring = useMemo(() => new THREE.Color(), []);
  ring.set(last ? last.color : "#3df0e0").multiplyScalar(2.2);
  const flash = 0.7 + 1.5 * Math.exp(-sinceHit * 9);

  return (
    <>
      <Trail choreo={choreo} t={t} />
      <group position={[b.x, b.y, BALL_Z]} rotation={[0, 0, angle]} scale={[along, across, across]}>
        <mesh castShadow>
          <sphereGeometry args={[choreo.ballRadius, 48, 32]} />
          <meshStandardMaterial color="#ffffff" emissive={tint} emissiveIntensity={flash} roughness={0.25} />
        </mesh>
        <mesh>
          <torusGeometry args={[choreo.ballRadius * 1.08, 0.04, 12, 48]} />
          <meshBasicMaterial color={ring} />
        </mesh>
      </group>
      <pointLight position={[b.x, b.y, BALL_Z + 0.5]} color={tint} intensity={5 + 14 * Math.exp(-sinceHit * 8)} distance={9} decay={2} />
    </>
  );
};

// ---------------------------------------------------------------- lights
const KeyLight: React.FC<{ x: number; y: number }> = ({ x, y }) => {
  const light = useMemo(() => {
    const l = new THREE.DirectionalLight("#e6ebff", 3.2);
    l.castShadow = true;
    l.shadow.mapSize.set(2048, 2048);
    l.shadow.camera.left = -14;
    l.shadow.camera.right = 14;
    l.shadow.camera.top = 18;
    l.shadow.camera.bottom = -18;
    l.shadow.camera.near = 1;
    l.shadow.camera.far = 60;
    l.shadow.bias = -0.0004;
    l.shadow.normalBias = 0.03;
    l.shadow.radius = 7;
    l.shadow.blurSamples = 16;
    return l;
  }, []);
  // snap to whole units so the shadow map does not shimmer as the camera glides
  const sx = Math.round(x);
  const sy = Math.round(y);
  light.position.set(sx - 5, sy + 13, 22);
  light.target.position.set(sx, sy, 0);
  return (
    <>
      <primitive object={light} />
      <primitive object={light.target} />
    </>
  );
};

const PlateLights: React.FC<{ plates: Plate[]; t: number }> = ({ plates, t }) => {
  // Fixed pool: the number of lights never changes, so shaders never recompile.
  const slots = new Array<Plate | null>(LIGHT_POOL).fill(null).map((_, i) => plates[i] ?? null);
  return (
    <>
      {slots.map((p, i) => {
        if (!p) return <pointLight key={i} intensity={0} distance={8} decay={2} />;
        const tau = t - p.t;
        return (
          <pointLight
            key={i}
            position={[p.x - p.nx * 0.55, p.y - p.ny * 0.55, 0.55]}
            color={p.color}
            intensity={8 + 30 * Math.exp(-5 * tau)}
            distance={8}
            decay={2}
          />
        );
      })}
    </>
  );
};

// ---------------------------------------------------------------- scene
export const Scene: React.FC<{ choreo: Choreo; t: number; frame: number; aspect: number }> = ({ choreo, t, frame, aspect }) => {
  const camX = sampleCurve(choreo.camera.x, choreo.camera.fps, t);
  const camY = sampleCurve(choreo.camera.y, choreo.camera.fps, t);
  const pulse = kickPulse(choreo, t);
  const level = sampleCurve(choreo.energy.rms, choreo.energy.fps, t);

  const visible = useMemo(() => {
    const out: { plate: Plate; index: number }[] = [];
    choreo.plates.forEach((plate, index) => {
      if (Math.abs(plate.y - camY) < VIEW_RANGE) out.push({ plate, index });
    });
    return out;
  }, [choreo, camY]);

  // newest hits first, so the freshest plates always own a light
  const litNewestFirst = useMemo(() => {
    const newest = lastIndexAtOrBefore(choreo.plates, t, (p) => p.t);
    const out: Plate[] = [];
    for (let i = newest; i >= 0 && out.length < LIGHT_POOL; i--) {
      if (Math.abs(choreo.plates[i].y - camY) < VIEW_RANGE + 4) out.push(choreo.plates[i]);
    }
    return out;
  }, [choreo, t, camY]);

  return (
    <>
      <CameraRig x={camX} y={camY} pulse={pulse} aspect={aspect} />
      <color attach="background" args={["#050506"]} />
      <ambientLight intensity={1.1 + 0.5 * level} color="#b4bcda" />
      <KeyLight x={camX} y={camY} />
      <PlateLights plates={litNewestFirst} t={t} />
      <Wall camY={camY} />
      {visible.map(({ plate, index }) => (
        <PlateMesh key={index} plate={plate} t={t} thickness={choreo.plateThickness} />
      ))}
      {visible.map(({ plate, index }) =>
        t >= plate.t && t - plate.t < SPARK_LIFE ? (
          <React.Fragment key={index}>
            <Sparks plate={plate} index={index} t={t} gravity={choreo.gravity} />
            <Shockwave plate={plate} t={t} />
          </React.Fragment>
        ) : null,
      )}
      <Ball choreo={choreo} t={t} />
      <Effects frame={frame} pulse={pulse} />
    </>
  );
};
