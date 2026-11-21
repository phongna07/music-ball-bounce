import React from "react";
import { CalculateMetadataFunction, Composition, staticFile } from "remotion";
import { Bounce, BounceProps } from "./Bounce";
import { Choreo, normalizeChoreo } from "./choreo";

const FPS = 60;

const calculateMetadata: CalculateMetadataFunction<BounceProps> = async ({ props, abortSignal }) => {
  let song = props.song;
  if (song === "latest") {
    const res = await fetch(staticFile("songs/latest.json"), { signal: abortSignal });
    if (!res.ok) throw new Error("No song found. Run pipeline/mbb.py on a music file first.");
    song = (await res.json()).song;
  }
  const res = await fetch(staticFile(`songs/${song}/choreo.json`), { signal: abortSignal });
  if (!res.ok) throw new Error(`No choreography for "${song}". Run pipeline/mbb.py on it first.`);
  const choreo: Choreo = normalizeChoreo(await res.json());
  return {
    durationInFrames: Math.max(1, Math.floor(choreo.duration * FPS)),
    props: { ...props, song, choreo },
    defaultOutName: song,
  };
};

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Bounce"
      component={Bounce}
      width={1080}
      height={1920}
      fps={FPS}
      durationInFrames={FPS * 10}
      defaultProps={{ song: "latest", choreo: null }}
      calculateMetadata={calculateMetadata}
    />
  );
};
