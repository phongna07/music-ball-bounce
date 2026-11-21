"""Music Ball Bounce: one command from a music file to a finished video.

    uv run --project pipeline python pipeline/mbb.py path/to/song.mp3

Stages: decode -> analyse (cached) -> choreograph + verify -> render.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="music or video file (anything ffmpeg can decode)")
    ap.add_argument("--seed", type=int, default=1, help="different seeds give different layouts")
    ap.add_argument("--palette", default="neon", choices=["neon", "sunset", "ice"])
    ap.add_argument("--min-gap", type=float, default=0.17, help="shortest time between bounces, seconds")
    ap.add_argument("--max-gap", type=float, default=1.3, help="longest silence before a beat is used instead")
    # -0.023 cancels the lag of the coarse onset frames in analyze.py (measured with check_sync.py)
    ap.add_argument("--offset", type=float, default=-0.023, help="shift every bounce by this many seconds")
    ap.add_argument("--balls", default="auto", choices=["auto", "1", "2"],
                    help="number of balls; auto uses two side by side when the rhythm is too fast for one")
    ap.add_argument("--no-render", action="store_true", help="stop after choreography (use Studio to preview)")
    ap.add_argument("--frames", help="render only this frame range, e.g. 0-299")
    ap.add_argument("--scale", type=float, default=1.0, help="output scale, 0.5 renders 540x960")
    ap.add_argument("--renderer", choices=["gpu", "cpu"], help="render mode: gpu (default Remotion) or cpu (multi-browser SwiftShader)")
    ap.add_argument("--gl", help="OpenGL backend override (e.g. angle, swangle, egl, vulkan)")
    ap.add_argument("--concurrency", help="parallel browser tabs for GPU render")
    ap.add_argument("--workers", help="parallel browsers for CPU render (default: sized to free CPU and RAM)")
    ap.add_argument("--tabs", help="tabs per browser for CPU render (default 1)")
    ap.add_argument("--out", type=Path, help="output mp4 path")
    args = ap.parse_args()

    src = args.input.resolve()
    if not src.is_file():
        sys.exit(f"Input not found: {src}")
    slug = re.sub(r"[^a-z0-9]+", "-", src.stem.lower()).strip("-") or "song"
    song_dir = ROOT / "public" / "songs" / slug
    work_dir = ROOT / "work" / slug
    song_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    digest = sha1(src)

    # 1. decode ------------------------------------------------------------
    wav = song_dir / "audio.wav"
    stamp = work_dir / "source.sha1"
    fresh = wav.exists() and stamp.exists() and stamp.read_text() == digest
    if not fresh:
        print("[1/4] decoding audio")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", "44100", str(wav)],
            check=True,
        )
        stamp.write_text(digest)
    else:
        print("[1/4] audio already decoded")

    # 2. analyse -----------------------------------------------------------
    analysis_path = work_dir / "analysis.json"
    if fresh and analysis_path.exists():
        print("[2/4] analysis cached")
        analysis = json.loads(analysis_path.read_text())
    else:
        print("[2/4] analysing audio")
        t0 = time.time()
        from analyze import analyze

        analysis = analyze(str(wav))
        analysis_path.write_text(json.dumps(analysis))
        print(f"      {len(analysis['notes'])} onsets, {analysis['tempo']} bpm, {time.time() - t0:.1f}s")

    # 3. choreograph -------------------------------------------------------
    print("[3/4] solving choreography")
    t0 = time.time()
    from choreograph import choreograph

    choreo = choreograph(analysis, seed=args.seed, palette=args.palette, min_gap=args.min_gap,
                         max_gap=args.max_gap, offset=args.offset, balls=args.balls)
    (song_dir / "choreo.json").write_text(json.dumps(choreo, separators=(",", ":")))
    r = choreo["report"]
    why = "auto" if args.balls == "auto" else "forced"
    print(f"      {r['balls']} ball(s) ({why}; two balls would catch {r['duo_gain']:.2f}x the notes of one)")
    print(f"      {len(choreo['plates'])} bounces verified, {r['dropped_notes']} dropped, {time.time() - t0:.1f}s")
    (ROOT / "public" / "songs" / "latest.json").write_text(json.dumps({"song": slug}))

    if args.no_render:
        print(f"Done. Preview with: npm run dev   (song: {slug})")
        return

    # 4. render ------------------------------------------------------------
    out = (args.out or ROOT / "out" / f"{slug}.mp4").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    props = work_dir / "props.json"
    props.write_text(json.dumps({"song": slug}))
    cmd = ["node", "scripts/render.mjs", str(out), f"--props={props}", f"--scale={args.scale}"]
    if args.frames:
        cmd.append(f"--frames={args.frames}")
    if args.renderer:
        cmd.append(f"--renderer={args.renderer}")
    if args.gl:
        cmd.append(f"--gl={args.gl}")
    if args.concurrency:
        cmd.append(f"--concurrency={args.concurrency}")
    if args.workers:
        cmd.append(f"--workers={args.workers}")
    if args.tabs:
        cmd.append(f"--tabs={args.tabs}")
    print("[4/4] rendering")
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"Done in {time.time() - t0:.0f}s: {out}")


if __name__ == "__main__":
    main()
