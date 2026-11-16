import React from "react";
import { Audio } from "@remotion/media";
import { ThreeCanvas } from "@remotion/three";
import { AbsoluteFill, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import * as THREE from "three";
import type { Choreo } from "./choreo";
import { Scene } from "./scene/Scene";

export type BounceProps = {
  /** Folder name under public/songs, or "latest" for the most recent pipeline run. */
  song: string;
  /** Filled in by calculateMetadata. */
  choreo: Choreo | null;
};

export const Bounce: React.FC<BounceProps> = ({ song, choreo }) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  if (!choreo) {
    return <AbsoluteFill style={{ backgroundColor: "#050506" }} />;
  }
  return (
    <AbsoluteFill style={{ backgroundColor: "#050506" }}>
      <ThreeCanvas
        width={width}
        height={height}
        flat
        shadows={{ type: THREE.VSMShadowMap }}
        gl={{ antialias: false, powerPreference: "high-performance" }}
      >
        <Scene choreo={choreo} t={frame / fps} frame={frame} aspect={width / height} />
      </ThreeCanvas>
      <Audio src={staticFile(`songs/${song}/audio.wav`)} />
    </AbsoluteFill>
  );
};
