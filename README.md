# Music Ball Bounce

Turns a music file or YouTube URL into a 1080×1920, 60 fps video of a ball bouncing down a wall of
plates, landing on every note. Everything runs locally.

## Make a video

```bash
uv run --project pipeline python pipeline/mbb.py path/to/song.mp3
# or directly from a YouTube URL (downloads and caches the .mp3 in work/youtube/):
uv run --project pipeline python pipeline/mbb.py "https://youtu.be/VIDEO_ID"
# or automatically find a popular slow piano track and make a video:
uv run --project pipeline python pipeline/mbb.py --discover
```

## Auto-discover slow piano tracks

Instead of hunting YouTube manually, `pipeline/discover.py` systematically searches for popular
music videos (high views — current or classic hits), finds their piano versions, downloads a
fast low-bandwidth preview, and verifies both **piano acoustics** (spectral purity, harmonicity,
low harsh high-frequency energy) and **slow rhythm pace** (onset density, median note spacing,
and single-ball `duo_gain` matching `choreograph.py`).

```bash
# Find 5 popular songs with verified slow piano covers (saved to work/discover/catalog.json)
uv run --project pipeline python pipeline/discover.py --count 5

# Search a specific era, genre, or mood
uv run --project pipeline python pipeline/discover.py --query "2010s pop ballads" --count 5

# Find & verify the best slow piano version for specific song(s) or a YouTube music video URL
uv run --project pipeline python pipeline/discover.py --song "Adele - Someone Like You"
uv run --project pipeline python pipeline/discover.py --song "https://www.youtube.com/watch?v=RgKAFK5djSk"

# Search single-track piano covers directly (instead of popular-original -> piano cover)
uv run --project pipeline python pipeline/discover.py --mode direct-piano --min-piano-views 500000

# View previously discovered qualifying tracks without hitting YouTube
uv run --project pipeline python pipeline/discover.py --list --unused-only

# Discover, download MP3s, or render directly into videos
uv run --project pipeline python pipeline/discover.py --count 3 --download
uv run --project pipeline python pipeline/discover.py --count 1 --render
```

Key discovery options:

| Option | Effect |
|---|---|
| `--mode popular-to-piano\|direct-piano\|both` | Strategy: find high-view originals then match piano covers (default), search piano tracks directly, or both |
| `--count 5` | Target number of verified tracks to collect |
| `--query "..."` | Custom YouTube search query (repeatable; omit to use the built-in curated hits pool) |
| `--song "..."` | Specific song title or YouTube URL to find a slow piano version for (repeatable) |
| `--min-original-views 50000000` | Minimum views on the original hit music video |
| `--min-piano-views 20000` | Minimum views on the piano cover video |
| `--max-onsets-per-sec 2.0` | Maximum full-track average note density (lower = slower pace) |
| `--max-p90-onsets-per-sec 2.5` | Maximum 90th-percentile 10s-window density (catches busy choruses) |
| `--max-sensitive-onsets-per-sec 2.5` | Sensitive onset cap that catches pedal-blurred fast arpeggios |
| `--min-median-ioi 0.40` | Minimum median gap between notes, in seconds |
| `--max-duo-gain 1.20` | Keeps clean single-ball pacing (below the 1.3 two-ball threshold) |
| `--min-piano-score 0.65` | Minimum acoustic piano purity score (0–1) |
| `--list` / `--unused-only` | List saved tracks from `work/discover/catalog.json` |
| `--download` / `--render` | Download full MP3s to `work/youtube/` or run `mbb.py` on each winner |
| `--json` / `--export-urls file.txt` | Machine-readable output for batch scripts |

Evaluated tracks and originals are saved in `work/discover/catalog.json`, so repeated runs
automatically skip already-checked songs and keep exploring fresh hits. Passing `--discover`
to `pipeline/mbb.py` picks the highest-scoring unused track from the catalog (or runs a fresh
search if none is left) and marks it used once processed.

The video is written to `out/<song>.mp4`. By default:
- Every YouTube track is transcribed locally to MIDI (`Transkun v2 ONNX`) and synthesized with **FluidSynth + Salamander Grand Piano V3** (`work/youtube/<id>/synth/`) instead of redistributing the original YouTube audio.
- Tracks longer than 60 seconds are automatically trimmed to the strongest **35–60s musical highlight clip** (chorus/climax with an immediate opening hook and smooth fade-in/fade-out; pass `--full` to disable).

Useful options:

| Option | Effect |
|---|---|
| `--full` | Render the full song instead of auto-selecting a 35–60s highlight clip |
| `--clip-min 35` / `--clip-max 60` | Highlight clip length bounds in seconds (target `--clip-target`, default `45`) |
| `--clip-start 95.0` | Override highlight selection with a manual clip start time in seconds |
| `--synthesize auto\|on\|off` | Local MIDI→piano synthesis (`auto` = all YouTube inputs, `on` = local files too, `off` = raw audio) |
| `--soundfont path/to/piano.sf2` | Use a custom SoundFont instead of the auto-downloaded Salamander Grand Piano V3 |
| `--no-reverb` / `--resynthesize` | Disable FluidSynth reverb or force re-transcription/re-synthesis |
| `--seed 2` | A different plate layout for the same song |
| `--palette neon\|sunset\|ice` | Plate colours |
| `--min-gap 0.17` | Shortest time between bounces, in seconds. Raise it for fewer bounces |
| `--balls auto\|1\|2` | Number of balls. `auto` (default) switches to two balls side by side when the rhythm is too fast for one |
| `--offset -0.023` | Shift every bounce earlier or later, in seconds |
| `--redownload` | Re-download the `.mp3` even if this YouTube URL is already cached |
| `--no-render` | Stop after choreography, then preview with `npm run dev` |
| `--frames 0-299` | Render only part of the video |
| `--scale 0.5` | Render at half size for a quick draft |
| `--renderer gpu\|cpu` | Override render mode (`gpu` = default Remotion CLI, `cpu` = multi-browser SwiftShader) |
| `--gl angle\|swangle\|...` | Override the Chromium OpenGL backend |
| `--concurrency 4` | Parallel browser tabs for `gpu` mode |
| `--workers 8` | Parallel browsers for `cpu` mode (default: sized to free CPU and RAM) |

## GPU vs. CPU rendering

By default, rendering uses **`gpu`** mode (standard single-browser `remotion render` with
hardware ANGLE WebGL). On headless or GPU-less machines, switch to **`cpu`** mode, which
rasterises WebGL in software via SwiftShader (`--gl=swangle`) across multiple parallel
Chrome browsers (`scripts/render.mjs`).

You can configure which machine uses `gpu` vs. `cpu` in any of these ways (highest
precedence first):

1. **Per-run CLI flag**: `--renderer gpu` or `--renderer cpu` (on `mbb.py` or `npm run render`).
2. **Environment variable**: `MBB_RENDERER=cpu` (or `gpu`), plus optional `MBB_GL`,
   `MBB_CONCURRENCY`, `MBB_WORKERS`, `MBB_TABS`, etc.
3. **Per-machine `.env` file** (gitignored — see `.env.example`):
   ```bash
   echo "MBB_RENDERER=cpu" > .env
   ```
4. **Shared hostname map in `render.config.json`**: map each machine's hostname (exact,
   short name, or `*` wildcard) to `"gpu"`, `"cpu"`, or a settings object:
   ```json
   {
     "default": "gpu",
     "machines": {
       "cpu-server-*": "cpu",
       "my-big-box": { "renderer": "cpu", "workers": 12, "reserveMemMb": 4096 },
       "my-workstation": { "renderer": "gpu", "gl": "angle", "concurrency": 4 }
     }
   }
   ```

Inspect what a machine will use without rendering:

```bash
npm run render -- --show-config
```

`mbb.py`, `npm run render`, `npx remotion render`, and `npm run dev` all read the same
configuration (`scripts/config.mjs`). `scripts/render.mjs` can also be run on its own:

```bash
npm run render -- out/song.mp4 --props='{"song":"song"}' [--renderer=gpu|cpu] [--frames=0-299] [--scale=0.5]
```

### How multi-browser CPU rendering works

A Chrome browser draws all its tabs through one GPU process, which keeps only about four
cores busy under SwiftShader, so `cpu` mode runs several browsers at once, each on its own
chunk of frames, then joins the chunks and adds the soundtrack with `ffmpeg`.

The number of browsers is chosen at start-up from what the machine has free right then:

- **CPU**: usable cores (CPU affinity and any container quota) / 4 cores per browser.
- **RAM**: (MemAvailable, capped by any container limit, minus 4 GB kept free) / ~1.3 GB
  per browser at full size, less when `--scale` is below 1.

The lower of the two wins, and the first line of the render log says which. On a 48-core,
48 GB machine that is 12 browsers at about 6.6 frames per second (one browser alone does
about 1.2) with a peak of about 12 GB. Each browser's x264 encoder is held to its share
of the cores; left alone, every encoder starts threads for all of them and buffers about
1 GB.

Every figure can be overridden via environment variables, `.env`, or `render.config.json`:
`MBB_WORKERS`, `MBB_TABS`, `MBB_CORES_PER_WORKER`, `MBB_BROWSER_MEM_MB`, `MBB_TAB_MEM_MB`,
`MBB_RESERVE_MEM_MB`, and `MBB_GL`.

## Preview and tweak the look

```bash
npm run dev
```

Remotion Studio opens the most recently processed song with audio and a scrubber.

## How it works

0. **Download** (`pipeline/youtube.py`, when given a YouTube URL): fetches audio with `yt-dlp` and extracts an `.mp3` cached under `work/youtube/<video-id>/`.
1. **Synthesize piano** (`pipeline/synthesize.py`, default for YouTube tracks): transcribes the audio locally to expressive MIDI (notes, velocities, sustain & soft pedals) via `Transkun v2 ONNX` (`work/models/transkun/`) and renders our own mastered studio piano track with `FluidSynth` + Salamander Grand Piano V3 (`work/soundfonts/`), cached under `work/youtube/<video-id>/synth/`.
2. **Decode & Highlight Clip** (`ffmpeg` + `pipeline/highlight.py`): prepares 44.1 kHz stereo audio and, by default (`--clip-min 35 --clip-max 60`), scores sections/downbeats/onsets to extract the strongest 35–60s highlight window with raised-cosine fades into `public/songs/<song>/audio.wav` (pass `--full` to keep the full song).
3. **Analyse** (`pipeline/analyze.py`): onsets, pitch, beats, kicks, sections and
   energy curves, cached in `work/<song>/analysis.json`.
4. **Choreograph** (`pipeline/choreograph.py`): picks the notes to bounce on, then
   solves where each plate goes. The ball is never simulated: each arc is a
   closed-form parabola and each plate is tilted so the reflection produces the
   next arc. A fine-step check then confirms no arc touches a plate it should not.
   Output: `public/songs/<song>/choreo.json`.

   **Two balls.** The notes are also picked at half the minimum gap and dealt out to
   two balls, each still keeping the full gap between its own bounces. A note goes to
   the ball that has waited longer, with low pitches leaning left and high pitches
   leaning right. If two balls would catch at least 30% more of the music (weighted
   by onset strength) than one, `auto` uses two. Each ball is solved in its own lane
   (nothing crosses the centre line) and follows a shared descent line, so the two
   stay level on screen. Verification also checks each ball against the other
   ball's plates and checks that the balls never touch.
5. **Render** (`src/`): a Three.js scene in Remotion where every frame is a pure
   function of time, so frames can render in parallel (in separate browsers) and stay in sync.

The look lives in `src/scene/Scene.tsx` (plates, ball, lights, camera),
`src/scene/Wall.tsx` and `src/scene/Effects.tsx` (bloom, tone mapping, grain).

Remotion is free for individuals and teams of up to three; larger companies need a
licence (https://www.remotion.dev/docs/license).
