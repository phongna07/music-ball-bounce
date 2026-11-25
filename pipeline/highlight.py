"""Automatic 35-60s musical highlight selection and smooth audio slicing.

Picks the strongest chorus/climax window from an audio analysis dict (avoiding
slow intros, dead-air gaps, and mid-note cuts) and slices the matching WAV
with raised-cosine fades so short-form videos open with an immediate hook.

Usage:
    uv run --project pipeline python pipeline/highlight.py path/to/song.mp3
    uv run --project pipeline python pipeline/highlight.py path/to/song.mp3 --min 35 --max 60 -o clip.wav
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class HighlightWindow:
    start: float
    end: float
    duration: float
    full_duration: float
    score: float
    reason: str
    energy_mean: float
    energy_peak: float
    hook_score: float
    note_rate: float
    clipped: bool


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _ramp_up(x: float, bad: float, good: float) -> float:
    if x >= good:
        return 1.0
    if x <= bad:
        return 0.0
    return (x - bad) / (good - bad)


def _ramp_down(x: float, good: float, bad: float) -> float:
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return (bad - x) / (bad - good)


def _band_score(x: float, lo_bad: float, lo_good: float, hi_good: float, hi_bad: float) -> float:
    if x < lo_good:
        return _ramp_up(x, lo_bad, lo_good)
    if x > hi_good:
        return _ramp_down(x, hi_good, hi_bad)
    return 1.0


def _build_rms_curve(analysis: dict[str, Any], duration: float) -> tuple[np.ndarray, float]:
    energy = analysis.get("energy") or {}
    rms = np.asarray(energy.get("rms") or [], dtype=np.float32)
    fps = float(energy.get("fps") or 60.0)
    if rms.size >= 2 and fps > 0:
        return rms, fps

    # Fallback curve built from sections when energy curve is absent (e.g., synthetic tests).
    fps = 60.0
    n = max(2, int(round(duration * fps)))
    curve = np.zeros(n, dtype=np.float32)
    sections = analysis.get("sections") or []
    for s in sections:
        i0 = max(0, int(round(float(s.get("t0", 0.0)) * fps)))
        i1 = min(n, int(round(float(s.get("t1", duration)) * fps)))
        if i1 > i0:
            curve[i0:i1] = float(s.get("energy", 0.5))
    if not sections:
        curve[:] = 0.5
    return curve, fps


def _thin_points(points: list[float], min_spacing: float) -> list[float]:
    out: list[float] = []
    for p in sorted(points):
        if not out or p - out[-1] >= min_spacing:
            out.append(float(p))
    return out


def _snap_start(t_phrase: float, note_times: np.ndarray, duration: float) -> float | None:
    """Snap a candidate phrase start cleanly before the next strong note for choreograph.py."""
    if note_times.size == 0:
        return max(0.0, round(t_phrase, 3))
    idx = int(np.searchsorted(note_times, t_phrase - 0.15, side="left"))
    if idx >= note_times.size:
        return None
    t_first = float(note_times[idx])
    if t_first >= duration - 5.0:
        return None

    target_a = max(0.0, t_first - 0.55)
    if idx > 0:
        t_prev = float(note_times[idx - 1])
        if target_a <= t_prev + 0.05 < t_first:
            gap = t_first - t_prev
            if gap >= 0.36:
                a = max(t_prev + 0.06, min(t_first - 0.35, 0.5 * (t_prev + t_first)))
            elif gap >= 0.14:
                a = 0.5 * (t_prev + t_first)
            else:
                a = max(0.0, t_first - 0.45)
        else:
            a = target_a
    else:
        a = target_a

    return max(0.0, round(a, 3))


def _snap_end(
    a: float,
    b_raw: float,
    note_times: np.ndarray,
    *,
    min_sec: float,
    max_sec: float,
    duration: float,
) -> float:
    """Adjust window end to leave clean decay room after the final note within bounds."""
    lo = a + min_sec
    hi = min(duration, a + max_sec)
    b = max(lo, min(hi, b_raw))
    if note_times.size == 0:
        return round(b, 3)

    idx = int(np.searchsorted(note_times, b - 0.08, side="right")) - 1
    if idx >= 0:
        t_last = float(note_times[idx])
        if t_last > a + 1.0 and (b - t_last) < 0.45:
            extended = min(hi, t_last + 0.65)
            if extended - t_last >= 0.30:
                b = max(lo, extended)
            elif t_last - 0.08 >= lo:
                b = t_last - 0.08
    return round(max(lo, min(hi, b)), 3)


def select_highlight(
    analysis: dict[str, Any],
    *,
    min_sec: float = 35.0,
    max_sec: float = 60.0,
    target_sec: float = 45.0,
    clip_start: float | None = None,
) -> HighlightWindow:
    """Select the strongest 35-60s highlight window from a full-track analysis dict."""
    if min_sec <= 0 or max_sec < min_sec:
        raise ValueError(f"Invalid clip bounds: min_sec={min_sec}, max_sec={max_sec}")

    target_sec = max(min_sec, min(max_sec, target_sec))
    duration = float(analysis["duration"])
    notes = [n for n in (analysis.get("notes") or []) if float(n.get("strength", 0.0)) >= 0.10]
    note_times = np.array(sorted(float(n["t"]) for n in notes), dtype=np.float64)
    note_weights = np.array(
        [float(n["strength"]) * (0.7 + 0.6 * float(n.get("melodic", 0.5))) for n in notes],
        dtype=np.float64,
    )
    rms, fps = _build_rms_curve(analysis, duration)
    song_p90 = max(float(np.percentile(rms, 90)) if rms.size else 1.0, 1e-6)
    song_p95 = max(float(np.percentile(rms, 95)) if rms.size else 1.0, 1e-6)

    def _window_stats(a: float, b: float) -> tuple[float, float, float, float]:
        i0 = max(0, int(round(a * fps)))
        i1 = min(rms.size, max(i0 + 1, int(round(b * fps))))
        seg = rms[i0:i1]
        e_mean = float(np.mean(seg)) if seg.size else 0.0
        e_peak = float(np.max(seg)) if seg.size else 0.0
        n_in = int(np.sum((note_times >= a) & (note_times <= b)))
        rate = n_in / max(b - a, 1e-6)
        return e_mean, e_peak, rate, float(np.percentile(seg, 85)) if seg.size else 0.0

    # Manual override path
    if clip_start is not None:
        a = max(0.0, min(float(clip_start), max(0.0, duration - 1.0)))
        b = min(duration, a + target_sec)
        e_mean, e_peak, rate, _ = _window_stats(a, b)
        return HighlightWindow(
            start=round(a, 3),
            end=round(b, 3),
            duration=round(b - a, 3),
            full_duration=round(duration, 3),
            score=100.0,
            reason=f"manual clip window ({a:.1f}s-{b:.1f}s)",
            energy_mean=round(e_mean / song_p95, 3),
            energy_peak=round(e_peak / song_p95, 3),
            hook_score=1.0,
            note_rate=round(rate, 2),
            clipped=(a > 0.01 or b < duration - 0.01),
        )

    # Short-track passthrough
    if duration <= max_sec:
        e_mean, e_peak, rate, _ = _window_stats(0.0, duration)
        return HighlightWindow(
            start=0.0,
            end=round(duration, 3),
            duration=round(duration, 3),
            full_duration=round(duration, 3),
            score=100.0,
            reason="track already within highlight duration",
            energy_mean=round(e_mean / song_p95, 3),
            energy_peak=round(e_peak / song_p95, 3),
            hook_score=1.0,
            note_rate=round(rate, 2),
            clipped=False,
        )

    sections = analysis.get("sections") or []
    section_starts = [float(s["t0"]) for s in sections if "t0" in s]
    section_ends = [float(s["t1"]) for s in sections if "t1" in s]
    downbeats = [float(t) for t in (analysis.get("downbeats") or [])]

    phrase_pts: list[float] = [0.0] + section_starts + _thin_points(downbeats, 1.5)
    if len(phrase_pts) < 8:
        phrase_pts.extend(float(t) for t in np.arange(0.0, max(1.0, duration - min_sec), 2.0))

    snapped_starts: list[float] = []
    for tp in phrase_pts:
        if tp > duration - min_sec + 0.5:
            continue
        snapped = _snap_start(tp, note_times, duration)
        if snapped is None or snapped + min_sec > duration + 1e-6:
            continue
        if not any(abs(snapped - existing) < 0.75 for existing in snapped_starts):
            snapped_starts.append(snapped)

    if not snapped_starts:
        snapped_starts = [0.0]

    db_arr = np.array(sorted(downbeats), dtype=np.float64)
    best: HighlightWindow | None = None
    desired_offsets = sorted({min_sec, 40.0, target_sec, 50.0, 55.0, max_sec})

    for a in snapped_starts:
        lo = a + min_sec
        hi = min(duration, a + max_sec)
        if lo > duration + 1e-6:
            continue

        raw_ends: list[float] = [min(duration, a + target_sec)]
        for se in section_ends:
            if lo - 0.5 <= se <= hi + 0.5:
                raw_ends.append(max(lo, min(hi, se)))
        if db_arr.size > 0:
            for off in desired_offsets:
                desired = min(hi, a + off)
                if desired < lo - 0.2:
                    continue
                j = int(np.argmin(np.abs(db_arr - desired)))
                cand_db = float(db_arr[j])
                if lo <= cand_db <= hi:
                    raw_ends.append(cand_db)
        else:
            for off in desired_offsets:
                raw_ends.append(max(lo, min(hi, a + off)))

        unique_ends: list[float] = []
        for b_raw in raw_ends:
            b = _snap_end(a, b_raw, note_times, min_sec=min_sec, max_sec=max_sec, duration=duration)
            if not any(abs(b - e) < 0.75 for e in unique_ends):
                unique_ends.append(b)

        for b in unique_ends:
            win_dur = b - a
            if win_dur < min_sec - 0.05 or win_dur > max_sec + 0.05:
                continue

            i0 = max(0, int(round(a * fps)))
            i1 = min(rms.size, max(i0 + 1, int(round(b * fps))))
            seg = rms[i0:i1]
            e_mean = float(np.mean(seg)) if seg.size else 0.0
            e_p85 = float(np.percentile(seg, 85)) if seg.size else e_mean
            e_peak = float(np.max(seg)) if seg.size else e_mean

            # 1. Energy / climax strength (0..1)
            s_energy = _clamp01(0.60 * (e_mean / song_p95) + 0.40 * (e_p85 / song_p95))

            # 2. Immediate opening hook in the first 4.5s (after 0.30s lead-in)
            hook_mask = (note_times >= a + 0.25) & (note_times <= min(b, a + 4.8))
            hook_notes = int(np.sum(hook_mask))
            early_notes = int(np.sum((note_times >= a + 0.25) & (note_times <= min(b, a + 3.0))))
            h0 = max(0, int(round((a + 0.30) * fps)))
            h1 = min(rms.size, max(h0 + 1, int(round(min(b, a + 4.8) * fps))))
            hook_rms = float(np.mean(rms[h0:h1])) if h1 > h0 else 0.0
            s_hook_notes = _ramp_up(hook_notes, bad=1.0, good=4.0)
            s_hook_energy = _clamp01(hook_rms / max(0.75 * e_mean, 0.25 * song_p90, 1e-6))
            s_hook = 0.55 * s_hook_notes + 0.45 * s_hook_energy
            if early_notes < 2:
                s_hook *= 0.35

            # 3. Continuity / dead-air avoidance
            inner_mask = (note_times >= a + 0.25) & (note_times <= b - 0.3)
            inner_notes = note_times[inner_mask]
            if inner_notes.size >= 2:
                gaps = np.diff(np.concatenate([[a + 0.25], inner_notes, [b - 0.3]]))
                max_gap = float(np.max(gaps))
            else:
                max_gap = win_dur

            # Downsample RMS to ~1s chunks to detect sustained quiet gaps
            step_1s = max(1, int(round(fps)))
            n_chunks = max(1, seg.size // step_1s)
            if seg.size >= step_1s:
                chunk_means = np.mean(seg[: n_chunks * step_1s].reshape(n_chunks, step_1s), axis=1)
                quiet_frac = float(np.mean(chunk_means < 0.08 * song_p90))
            else:
                quiet_frac = 0.0
            s_gap = _ramp_down(max_gap, good=1.8, bad=3.5)
            s_quiet = _ramp_down(quiet_frac, good=0.03, bad=0.18)
            s_continuity = 0.65 * s_gap + 0.35 * s_quiet

            # 4. Bounce playability (steady note rate & melodic quality)
            win_mask = (note_times >= a + 0.25) & (note_times <= b - 0.2)
            n_notes = int(np.sum(win_mask))
            note_rate = n_notes / max(win_dur, 1e-6)
            s_rate = _band_score(note_rate, lo_bad=0.5, lo_good=1.1, hi_good=2.3, hi_bad=3.8)
            mean_weight = float(np.mean(note_weights[win_mask])) if n_notes > 0 and note_weights.size else 0.3
            s_weight = _ramp_up(mean_weight, bad=0.15, good=0.65)
            s_play = 0.70 * s_rate + 0.30 * s_weight

            # 5. Musical shape, phrase boundary alignment & target length comfort
            align_start = 1.0 if any(abs(a - st) <= 1.5 for st in section_starts) else 0.0
            align_end = 1.0 if any(abs(b - se) <= 2.0 for se in section_ends) else 0.0
            s_align = 0.5 * align_start + 0.5 * align_end

            mid = seg.size // 2
            first_half = float(np.mean(seg[:mid])) if mid > 0 else e_mean
            second_half = float(np.mean(seg[mid:])) if mid < seg.size else e_mean
            s_arc = _ramp_up(second_half / max(first_half, 1e-6), bad=0.55, good=0.95)
            s_len = math.exp(-0.5 * ((win_dur - target_sec) / 10.0) ** 2)
            s_shape = 0.35 * s_align + 0.35 * s_arc + 0.30 * s_len

            total = 100.0 * (
                0.34 * s_energy
                + 0.22 * s_hook
                + 0.20 * s_continuity
                + 0.14 * s_play
                + 0.10 * s_shape
            )

            if best is None or total > best.score:
                e_norm = round(e_mean / song_p95, 2)
                best = HighlightWindow(
                    start=round(a, 3),
                    end=round(b, 3),
                    duration=round(win_dur, 3),
                    full_duration=round(duration, 3),
                    score=round(total, 1),
                    reason=(
                        f"highlight energy {e_norm:.2f}, hook {s_hook:.2f}, "
                        f"{note_rate:.1f} notes/s, max gap {max_gap:.1f}s"
                    ),
                    energy_mean=round(e_mean / song_p95, 3),
                    energy_peak=round(e_peak / song_p95, 3),
                    hook_score=round(s_hook, 3),
                    note_rate=round(note_rate, 2),
                    clipped=True,
                )

    assert best is not None
    return best


def slice_audio_clip(
    src_wav: Path,
    out_wav: Path,
    start: float,
    end: float,
    *,
    fade_in: float = 0.12,
    fade_out: float = 2.0,
) -> float:
    """Slice [start, end] seconds from src_wav with raised-cosine boundary fades."""
    src_wav = Path(src_wav)
    out_wav = Path(out_wav)
    info = sf.info(str(src_wav))

    if info.samplerate != 44100 or info.channels != 2:
        with tempfile.TemporaryDirectory(prefix="mbb-norm-") as tmp:
            norm_wav = Path(tmp) / "norm.wav"
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-y",
                    "-i", str(src_wav),
                    "-vn", "-ac", "2", "-ar", "44100",
                    "-c:a", "pcm_s16le",
                    str(norm_wav),
                ],
                check=True,
            )
            audio, sr = sf.read(str(norm_wav), dtype="float32", always_2d=True)
    else:
        audio, sr = sf.read(str(src_wav), dtype="float32", always_2d=True)

    n_total = audio.shape[0]
    full_dur = n_total / sr
    i0 = max(0, min(n_total - 1, int(round(start * sr))))
    i1 = max(i0 + 1, min(n_total, int(round(end * sr))))
    clip = np.array(audio[i0:i1], dtype=np.float32, copy=True)
    clip_dur = clip.shape[0] / sr

    if start > 0.05 and fade_in > 0:
        fi_dur = min(fade_in, 0.25 * clip_dur)
        n_fi = min(clip.shape[0], int(round(fi_dur * sr)))
        if n_fi > 1:
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n_fi, dtype=np.float32)))
            clip[:n_fi] *= ramp[:, None]

    if end < full_dur - 0.05 and fade_out > 0:
        fo_dur = min(fade_out, 0.25 * clip_dur)
        n_fo = min(clip.shape[0], int(round(fo_dur * sr)))
        if n_fo > 1:
            ramp = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0.0, 1.0, n_fo, dtype=np.float32)))
            clip[-n_fo:] *= ramp[:, None]

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_wav.with_name(f".{out_wav.name}.tmp.{os.getpid()}.wav")
    try:
        sf.write(str(tmp), clip, sr, subtype="PCM_16")
        os.replace(tmp, out_wav)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return round(clip_dur, 4)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="audio/video file or pre-computed analysis.json")
    ap.add_argument("--min", type=float, default=35.0, dest="min_sec", help="minimum highlight duration (s)")
    ap.add_argument("--max", type=float, default=60.0, dest="max_sec", help="maximum highlight duration (s)")
    ap.add_argument("--target", type=float, default=45.0, dest="target_sec", help="target highlight duration (s)")
    ap.add_argument("--start", type=float, default=None, dest="clip_start", help="manual clip start time (s)")
    ap.add_argument("-o", "--out", type=Path, help="optional output WAV path to slice the highlight clip into")
    ap.add_argument("--json", action="store_true", dest="as_json", help="print result as JSON")
    args = ap.parse_args(argv)

    src = args.input.expanduser().resolve()
    if not src.is_file():
        sys.exit(f"Input not found: {src}")

    if src.suffix.lower() == ".json":
        analysis = json.loads(src.read_text(encoding="utf-8"))
        if args.out is not None:
            sys.exit("Pass an audio file (not analysis.json) when using -o/--out.")
    else:
        from analyze import analyze

        with tempfile.TemporaryDirectory(prefix="mbb-hl-") as tmp:
            wav = Path(tmp) / "full.wav"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", "44100", str(wav)],
                check=True,
            )
            analysis = analyze(str(wav))
            win = select_highlight(
                analysis,
                min_sec=args.min_sec,
                max_sec=args.max_sec,
                target_sec=args.target_sec,
                clip_start=args.clip_start,
            )
            if args.out is not None:
                slice_audio_clip(wav, args.out.expanduser().resolve(), win.start, win.end)
            if args.as_json:
                print(json.dumps(win.__dict__, indent=2))
            else:
                print(
                    f"{win.start:.2f}s -> {win.end:.2f}s ({win.duration:.2f}s / {win.full_duration:.2f}s) "
                    f"| score {win.score:.1f} | {win.reason}"
                )
                if args.out is not None:
                    print(f"Wrote clip: {args.out.expanduser().resolve()}")
            return

    win = select_highlight(
        analysis,
        min_sec=args.min_sec,
        max_sec=args.max_sec,
        target_sec=args.target_sec,
        clip_start=args.clip_start,
    )
    if args.as_json:
        print(json.dumps(win.__dict__, indent=2))
    else:
        print(
            f"{win.start:.2f}s -> {win.end:.2f}s ({win.duration:.2f}s / {win.full_duration:.2f}s) "
            f"| score {win.score:.1f} | {win.reason}"
        )


if __name__ == "__main__":
    main()
