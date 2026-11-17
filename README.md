# Music Ball Bounce

Turns a music file into a 1080×1920, 60 fps video of a ball bouncing down a wall of
plates, landing on every note. Everything runs locally.

## Make a video

```bash
uv run --project pipeline python pipeline/mbb.py path/to/song.mp3
```

The video is written to `out/<song>.mp4`. Useful options:

| Option | Effect |
|---|---|
| `--seed 2` | A different plate layout for the same song |
| `--palette neon\|sunset\|ice` | Plate colours |
| `--min-gap 0.17` | Shortest time between bounces, in seconds. Raise it for fewer bounces |
| `--offset -0.023` | Shift every bounce earlier or later, in seconds |
| `--no-render` | Stop after choreography, then preview with `npm run dev` |
| `--frames 0-299` | Render only part of the video |
| `--scale 0.5` | Render at half size for a quick draft |

## Preview and tweak the look

```bash
npm run dev
```

Remotion Studio opens the most recently processed song with audio and a scrubber.

## How it works

1. **Decode** (`ffmpeg`): any input becomes `public/songs/<song>/audio.wav`.
2. **Analyse** (`pipeline/analyze.py`): onsets, pitch, beats, kicks, sections and
   energy curves, cached in `work/<song>/analysis.json`.
3. **Choreograph** (`pipeline/choreograph.py`): picks the notes to bounce on, then
   solves where each plate goes. The ball is never simulated: each arc is a
   closed-form parabola and each plate is tilted so the reflection produces the
   next arc. A fine-step check then confirms no arc touches a plate it should not.
   Output: `public/songs/<song>/choreo.json`.
4. **Render** (`src/`): a Three.js scene in Remotion where every frame is a pure
   function of time, so frames can render in parallel and stay in sync.

The look lives in `src/scene/Scene.tsx` (plates, ball, lights, camera),
`src/scene/Wall.tsx` and `src/scene/Effects.tsx` (bloom, tone mapping, grain).

Remotion is free for individuals and teams of up to three; larger companies need a
licence (https://www.remotion.dev/docs/license).
