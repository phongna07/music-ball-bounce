"""Music Ball Bounce: one command from a music file or YouTube URL to a finished video.

    uv run --project pipeline python pipeline/mbb.py path/to/song.mp3
    uv run --project pipeline python pipeline/mbb.py "https://youtu.be/VIDEO_ID"
    uv run --project pipeline python pipeline/mbb.py --discover

Stages:
    [discover ->] [youtube mp3 ->] [piano MIDI synth ->] prepare / 35-60s highlight ->
    analyse (cached) -> choreograph + verify -> render.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

from youtube import download_youtube_mp3, is_http_url, is_youtube_url, slugify

ROOT = Path(__file__).resolve().parent.parent


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_mmss(sec: float) -> str:
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def _is_standard_wav(path: Path) -> bool:
    if path.suffix.lower() != ".wav":
        return False
    try:
        info = sf.info(str(path))
        return info.samplerate == 44100 and info.channels == 2 and info.subtype == "PCM_16"
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="music/video file (anything ffmpeg can decode) or YouTube URL")
    ap.add_argument("--discover", action="store_true",
                    help="automatically pick (or search for) a popular slow piano track from YouTube")
    ap.add_argument("--discover-query", action="append", dest="discover_queries",
                    help="custom YouTube search query for --discover (repeatable)")
    ap.add_argument("--discover-song", action="append", dest="discover_songs",
                    help="specific song title or YouTube URL to find a slow piano version for")
    ap.add_argument("--discover-mode", choices=["popular-to-piano", "direct-piano", "both"],
                    default="popular-to-piano", help="discovery strategy for --discover")
    ap.add_argument("--discover-fresh", action="store_true",
                    help="run a fresh YouTube search instead of picking an unused track from the catalog")
    ap.add_argument("--synthesize", choices=["auto", "on", "off"], default="auto",
                    help="transcribe to MIDI and synthesize local piano audio (default: auto = all YouTube inputs)")
    ap.add_argument("--soundfont", type=Path,
                    help="SF2/SF3 soundfont path for piano synthesis (default: Salamander Grand Piano)")
    ap.add_argument("--resynthesize", action="store_true",
                    help="re-run MIDI transcription and piano synthesis even if cached")
    ap.add_argument("--no-reverb", action="store_true",
                    help="disable FluidSynth reverb during piano synthesis")
    ap.add_argument("--full", action="store_true",
                    help="render the full track instead of auto-selecting a 35-60s highlight clip")
    ap.add_argument("--clip-min", type=float, default=35.0,
                    help="minimum highlight clip length in seconds (default: 35)")
    ap.add_argument("--clip-max", type=float, default=60.0,
                    help="maximum highlight clip length in seconds (default: 60)")
    ap.add_argument("--clip-target", type=float, default=45.0,
                    help="target highlight clip length in seconds (default: 45)")
    ap.add_argument("--clip-start", type=float, default=None,
                    help="manual clip start time in seconds (uses --clip-target as length)")
    ap.add_argument("--seed", type=int, default=1, help="different seeds give different layouts")
    ap.add_argument("--palette", default="neon", choices=["neon", "sunset", "ice"])
    ap.add_argument("--min-gap", type=float, default=0.17, help="shortest time between bounces, seconds")
    ap.add_argument("--max-gap", type=float, default=1.8, help="longest silence before a beat is used instead")
    ap.add_argument("--offset", type=float, default=0.0, help="optional manual timing shift for every bounce, seconds")
    ap.add_argument("--no-frame-snap", action="store_true",
                    help="disable snapping bounce times to exact 60 fps frame boundaries")
    ap.add_argument("--balls", default="auto", choices=["auto", "1", "2"],
                    help="number of balls; auto uses two side by side when the rhythm is too fast for one")
    ap.add_argument("--redownload", action="store_true", help="re-download MP3 even if this YouTube URL is cached")
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

    if args.clip_min <= 0 or args.clip_max < args.clip_min:
        sys.exit(f"Invalid clip bounds: --clip-min={args.clip_min}, --clip-max={args.clip_max}")
    if args.clip_target <= 0:
        sys.exit(f"Invalid --clip-target={args.clip_target}")

    discovered_id: str | None = None
    catalog = None
    if args.discover or args.discover_queries or args.discover_songs:
        if args.input:
            sys.exit("Pass either an input path/URL or --discover, not both.")
        from discover import DEFAULT_CATALOG_PATH, DiscoveryCatalog, DiscoveryCriteria, discover_tracks

        catalog = DiscoveryCatalog(DEFAULT_CATALOG_PATH)
        chosen = None
        if not args.discover_fresh and not args.discover_queries and not args.discover_songs:
            unused = catalog.list_tracks(unused_only=True)
            if unused:
                chosen = unused[0]
                print(f"[discover] using catalog track: {chosen.title} (score {chosen.score:.1f})")
        if chosen is None:
            print("[discover] searching YouTube for a popular slow piano track...")
            found = discover_tracks(
                count=1,
                mode=args.discover_mode,
                queries=args.discover_queries,
                songs=args.discover_songs,
                criteria=DiscoveryCriteria(min_gap=args.min_gap),
                catalog=catalog,
            )
            if not found:
                sys.exit("No qualifying slow piano track found. Try relaxing thresholds in pipeline/discover.py.")
            chosen = found[0]
            print(f"[discover] selected: {chosen.title} (score {chosen.score:.1f}) -> {chosen.url}")
        args.input = chosen.url
        discovered_id = chosen.video_id
    elif not args.input:
        ap.error("provide an input audio/video file, a YouTube URL, or --discover")

    is_yt = is_youtube_url(args.input)
    should_synthesize = (args.synthesize == "on") or (args.synthesize == "auto" and is_yt)
    should_clip = not args.full

    steps = 4 + (1 if is_yt else 0) + (1 if should_synthesize else 0)
    step = 1
    yt = None

    if is_yt:
        print(f"[{step}/{steps}] fetching YouTube mp3")
        t0 = time.time()
        try:
            yt = download_youtube_mp3(args.input, ROOT / "work" / "youtube", redownload=args.redownload)
        except (ValueError, RuntimeError) as exc:
            sys.exit(str(exc))
        src = yt.mp3_path
        slug = yt.slug
        rel = src.relative_to(ROOT)
        if yt.cached:
            print(f"      cached: {yt.title} ({rel})")
        else:
            print(f"      downloaded: {yt.title} ({rel}), {time.time() - t0:.1f}s")
        step += 1
    elif is_http_url(args.input):
        sys.exit(f"Unsupported URL (only YouTube URLs are supported): {args.input}")
    else:
        src = Path(args.input).expanduser().resolve()
        if not src.is_file():
            sys.exit(f"Input not found: {src}")
        slug = slugify(src.stem)

    song_dir = ROOT / "public" / "songs" / slug
    work_dir = ROOT / "work" / slug
    song_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # synthesize -----------------------------------------------------------
    synth = None
    active_src = src
    attack_map = None
    sf_path: Path | None = None
    if should_synthesize:
        print(f"[{step}/{steps}] synthesizing local piano audio")
        from synthesize import measure_soundfont_attack_map, synthesize_piano_version

        synth_dir = (ROOT / "work" / "youtube" / yt.video_id / "synth") if yt is not None else (work_dir / "synth")
        synth = synthesize_piano_version(
            src,
            synth_dir,
            soundfont=args.soundfont,
            resynthesize=args.resynthesize,
            reverb=not args.no_reverb,
        )
        active_src = synth.wav_path
        sf_path = synth.soundfont_path
        if sf_path is not None and sf_path.is_file():
            attack_map = measure_soundfont_attack_map(sf_path)
        rel_wav = synth.wav_path.relative_to(ROOT) if synth.wav_path.is_relative_to(ROOT) else synth.wav_path
        state = "cached" if synth.cached else "synthesized"
        print(f"      {state}: {synth.note_count} notes, {synth.pedal_count} pedals ({synth.soundfont}) -> {rel_wav}")
        step += 1

    # decode / highlight clip ----------------------------------------------
    from analyze import ANALYSIS_VERSION, analyze
    from highlight import select_highlight, slice_audio_clip

    full_digest = sha1(active_src)
    if _is_standard_wav(active_src):
        full_audio_path = active_src
    else:
        full_wav = work_dir / "full.wav"
        full_stamp = work_dir / "full.sha1"
        full_fresh = full_wav.exists() and full_stamp.exists() and full_stamp.read_text() == full_digest
        if not full_fresh:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(active_src), "-vn", "-ac", "2", "-ar", "44100", str(full_wav)],
                check=True,
            )
            full_stamp.write_text(full_digest)
        full_audio_path = full_wav

    full_duration = float(sf.info(str(full_audio_path)).duration)
    if not should_clip:
        clip_spec: dict[str, object] = {"mode": "full"}
    else:
        clip_spec = {
            "mode": "highlight",
            "min": args.clip_min,
            "max": args.clip_max,
            "target": args.clip_target,
            "start": args.clip_start,
        }

    wav = song_dir / "audio.wav"
    clip_meta_path = work_dir / "clip.json"
    old_clip: dict[str, object] = {}
    clip_cached = False
    if wav.exists() and clip_meta_path.is_file():
        try:
            old_clip = json.loads(clip_meta_path.read_text(encoding="utf-8"))
            if old_clip.get("full_sha1") == full_digest and old_clip.get("clip_spec") == clip_spec:
                clip_cached = True
        except (OSError, json.JSONDecodeError):
            old_clip = {}

    print(f"[{step}/{steps}] preparing audio ({'highlight clip' if should_clip else 'full track'})")
    if clip_cached:
        if old_clip.get("clipped"):
            print(
                f"      cached highlight: {_fmt_mmss(float(old_clip['start']))} -> {_fmt_mmss(float(old_clip['end']))} "
                f"({float(old_clip['duration']):.1f}s / {float(old_clip['full_duration']):.1f}s, "
                f"{old_clip.get('reason', '')})"
            )
        else:
            print(f"      audio cached ({float(old_clip.get('duration', full_duration)):.1f}s)")
    else:
        if should_clip and (args.clip_start is not None or full_duration > args.clip_max):
            full_analysis_path = work_dir / "full_analysis.json"
            full_analysis_stamp = work_dir / "full_analysis.sha1"
            full_analysis = None
            if (
                full_analysis_path.is_file()
                and full_analysis_stamp.is_file()
                and full_analysis_stamp.read_text() == full_digest
            ):
                try:
                    cached_full = json.loads(full_analysis_path.read_text(encoding="utf-8"))
                    if cached_full.get("version") == ANALYSIS_VERSION:
                        full_analysis = cached_full
                except (OSError, json.JSONDecodeError):
                    full_analysis = None
            if full_analysis is None:
                t0 = time.time()
                print(f"      analysing full track ({full_duration:.1f}s) for highlight selection...")
                full_analysis = analyze(
                    str(full_audio_path),
                    midi_path=synth.midi_path if synth is not None else None,
                    clip_start=0.0,
                    attack_map=attack_map,
                )
                full_analysis_path.write_text(json.dumps(full_analysis), encoding="utf-8")
                full_analysis_stamp.write_text(full_digest, encoding="utf-8")
                print(f"      full analysis ready in {time.time() - t0:.1f}s")

            win = select_highlight(
                full_analysis,
                min_sec=args.clip_min,
                max_sec=args.clip_max,
                target_sec=args.clip_target,
                clip_start=args.clip_start,
            )
            if win.clipped:
                clip_dur = slice_audio_clip(full_audio_path, wav, win.start, win.end)
                print(
                    f"      highlight: {_fmt_mmss(win.start)} -> {_fmt_mmss(win.end)} "
                    f"({clip_dur:.1f}s / {full_duration:.1f}s, score {win.score:.1f}: {win.reason})"
                )
            else:
                shutil.copyfile(full_audio_path, wav)
                clip_dur = full_duration
                print(f"      track ({full_duration:.1f}s) already within {args.clip_max:.0f}s max; keeping full audio")

            clip_payload = {
                "full_sha1": full_digest,
                "clip_spec": clip_spec,
                "start": win.start,
                "end": win.end,
                "duration": clip_dur,
                "full_duration": round(full_duration, 3),
                "score": win.score,
                "reason": win.reason,
                "clipped": win.clipped,
                "synthesized": should_synthesize,
                "midi_path": str(synth.midi_path) if synth is not None else None,
                "soundfont_path": str(sf_path) if sf_path is not None else None,
            }
        else:
            shutil.copyfile(full_audio_path, wav)
            why = "full mode requested" if not should_clip else f"already <= {args.clip_max:.0f}s"
            print(f"      keeping full audio ({full_duration:.1f}s; {why})")
            clip_payload = {
                "full_sha1": full_digest,
                "clip_spec": clip_spec,
                "start": 0.0,
                "end": round(full_duration, 3),
                "duration": round(full_duration, 3),
                "full_duration": round(full_duration, 3),
                "score": 100.0,
                "reason": why,
                "clipped": False,
                "synthesized": should_synthesize,
                "midi_path": str(synth.midi_path) if synth is not None else None,
                "soundfont_path": str(sf_path) if sf_path is not None else None,
            }
        clip_meta_path.write_text(json.dumps(clip_payload, indent=2), encoding="utf-8")
    clip_meta = old_clip if clip_cached else clip_payload
    step += 1

    # analyse --------------------------------------------------------------
    active_midi: Path | None = None
    if should_synthesize:
        midi_str = clip_meta.get("midi_path")
        if isinstance(midi_str, str) and Path(midi_str).is_file():
            active_midi = Path(midi_str)
        elif synth is not None and synth.midi_path.is_file():
            active_midi = synth.midi_path
    clip_start = float(clip_meta.get("start", 0.0)) if clip_meta.get("clipped") else 0.0

    clip_digest = sha1(wav)
    midi_digest = sha1(active_midi) if active_midi is not None else "none"
    sf_tag = (
        f"{sf_path.name}:{sf_path.stat().st_size}:{attack_map.get('median_delay', 0.0)}"
        if sf_path is not None and attack_map is not None
        else "none"
    )
    analysis_stamp = f"v{ANALYSIS_VERSION}:{clip_digest}:{midi_digest}:{sf_tag}:{clip_start:.6f}"
    stamp = work_dir / "source.sha1"
    analysis_path = work_dir / "analysis.json"
    analysis = None
    if stamp.is_file() and stamp.read_text() == analysis_stamp and analysis_path.is_file():
        try:
            cached_analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            if cached_analysis.get("version") == ANALYSIS_VERSION:
                analysis = cached_analysis
        except (OSError, json.JSONDecodeError):
            analysis = None
    if analysis is not None:
        src_tag = "midi+sf2" if analysis.get("note_source") == "midi" else "audio"
        print(f"[{step}/{steps}] analysis cached ({src_tag})")
    else:
        print(f"[{step}/{steps}] analysing audio")
        t0 = time.time()
        analysis = analyze(
            str(wav),
            midi_path=active_midi,
            clip_start=clip_start,
            attack_map=attack_map,
        )
        analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
        stamp.write_text(analysis_stamp, encoding="utf-8")
        src_tag = "midi+sf2" if analysis.get("note_source") == "midi" else "audio"
        print(f"      {len(analysis['notes'])} onsets ({src_tag}), {analysis['tempo']} bpm, {time.time() - t0:.1f}s")
    step += 1

    # choreograph ----------------------------------------------------------
    print(f"[{step}/{steps}] solving choreography")
    t0 = time.time()
    from choreograph import VIDEO_FPS, choreograph

    snap_fps = None if args.no_frame_snap else VIDEO_FPS
    choreo = choreograph(analysis, seed=args.seed, palette=args.palette, min_gap=args.min_gap,
                         max_gap=args.max_gap, offset=args.offset, balls=args.balls, snap_fps=snap_fps)
    (song_dir / "choreo.json").write_text(json.dumps(choreo, separators=(",", ":")))
    r = choreo["report"]
    why = "auto" if args.balls == "auto" else "forced"
    print(f"      {r['balls']} ball(s) ({why}; two balls would catch {r['duo_gain']:.2f}x the notes of one)")
    print(f"      {len(choreo['plates'])} bounces verified, {r['dropped_notes']} dropped, {time.time() - t0:.1f}s")
    (ROOT / "public" / "songs" / "latest.json").write_text(json.dumps({"song": slug}))
    step += 1

    if args.no_render:
        if discovered_id and catalog is not None:
            catalog.mark_used(discovered_id)
        print(f"Done. Preview with: npm run dev   (song: {slug})")
        return

    # render ---------------------------------------------------------------
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
    print(f"[{step}/{steps}] rendering")
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    if discovered_id and catalog is not None:
        catalog.mark_used(discovered_id)
    print(f"Done in {time.time() - t0:.0f}s: {out}")


if __name__ == "__main__":
    main()
