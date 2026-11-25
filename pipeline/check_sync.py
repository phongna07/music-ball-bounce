"""Measure audio-visual sync of a rendered video.

    uv run --project pipeline python pipeline/check_sync.py out/song.mp4 public/songs/song/choreo.json

Checks:
  1. MP4 mux alignment: cross-correlates extracted MP4 audio vs source audio.wav.
  2. Frame quantization: verifies bounce times match 60 fps frame boundaries.
  3. Acoustic transient sync: localizes MP4 audio attacks with 44.1 kHz fine
     spectral flux (non-circular, independent of coarse onset_detect) and
     measures both choreo plate time and actual visual contact frame.
  4. MIDI + SoundFont ground-truth sync (when clip.json / --midi is present):
     compares rendered visual contact frames against exact MIDI note_on +
     measured SoundFont sample attack latency, plus note coverage and phantoms.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import correlate

from analyze import FINE_SR, extract_midi_notes, refine_onset_times_from_mono
from synthesize import DEFAULT_SOUNDFONT_PATH, measure_soundfont_attack_map

ROOT = Path(__file__).resolve().parent.parent
HIT_EPS = 1e-4


def _load_wav_mono(path: Path, target_sr: int = FINE_SR) -> np.ndarray:
    info = sf.info(str(path))
    if info.samplerate == target_sr:
        y, _ = sf.read(str(path), dtype="float32", always_2d=True)
        return y.mean(axis=1)
    y, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return np.asarray(y, dtype=np.float32)


def _detect_fine_audio_transients(mono: np.ndarray, sr: int = FINE_SR) -> np.ndarray:
    hop_c = 128
    S = np.abs(librosa.stft(mono, n_fft=1024, hop_length=hop_c))
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=80, fmin=27.5, fmax=10000.0),
        ref=np.max,
    )
    env = librosa.onset.onset_strength(S=mel, sr=sr, hop_length=hop_c)
    frames = librosa.onset.onset_detect(
        onset_envelope=env,
        sr=sr,
        hop_length=hop_c,
        backtrack=False,
        delta=0.05,
        wait=int(0.05 * sr / hop_c),
    )
    coarse = librosa.frames_to_time(frames, sr=sr, hop_length=hop_c)
    return refine_onset_times_from_mono(mono, coarse, sr=sr)


def _format_stats(
    label: str,
    bounces: np.ndarray,
    refs: np.ndarray,
    *,
    fps: int,
    max_match_ms: float = 60.0,
) -> None:
    if bounces.size == 0 or refs.size == 0:
        print(f"  {label}: no comparable events")
        return
    idx = np.abs(refs[None, :] - bounces[:, None]).argmin(axis=1)
    err_ms = (bounces - refs[idx]) * 1000.0  # positive: bounce after reference
    matched = err_ms[np.abs(err_ms) < max_match_ms]
    unmatched = int(bounces.size - matched.size)
    frame_ms = 1000.0 / fps
    half_frame_ms = 0.5 * frame_ms
    if matched.size == 0:
        print(f"  {label}: 0/{bounces.size} matched within {max_match_ms:.0f} ms")
        return
    abs_err = np.abs(matched)
    print(
        f"  {label}: matched {matched.size}/{bounces.size} (unmatched={unmatched}) | "
        f"median={np.median(matched):+.2f} ms, mean|err|={abs_err.mean():.2f} ms, "
        f"p90|err|={np.percentile(abs_err, 90):.2f} ms, worst={abs_err.max():.2f} ms | "
        f"<=0.5 frame ({half_frame_ms:.1f}ms): {np.mean(abs_err <= half_frame_ms) * 100:.0f}%, "
        f"<=1 frame ({frame_ms:.1f}ms): {np.mean(abs_err <= frame_ms) * 100:.0f}%"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path, help="rendered .mp4 video")
    ap.add_argument("choreo", type=Path, help="choreo.json used for rendering")
    ap.add_argument("--wav", type=Path, help="source audio.wav (auto-detected beside choreo.json)")
    ap.add_argument("--clip-json", type=Path, help="work/<song>/clip.json (auto-detected)")
    ap.add_argument("--midi", type=Path, help="ground-truth MIDI file")
    ap.add_argument("--soundfont", type=Path, help="SF2/SF3 soundfont for attack calibration")
    ap.add_argument("--fps", type=int, default=60, help="video frame rate (default: 60)")
    args = ap.parse_args(argv)

    video = args.video.expanduser().resolve()
    choreo_path = args.choreo.expanduser().resolve()
    choreo = json.loads(choreo_path.read_text(encoding="utf-8"))

    wav_path = args.wav.expanduser().resolve() if args.wav else (choreo_path.parent / "audio.wav")
    if not wav_path.is_file():
        wav_path = None

    slug = choreo_path.parent.name
    clip_json_path = args.clip_json.expanduser().resolve() if args.clip_json else (ROOT / "work" / slug / "clip.json")
    clip_meta: dict = {}
    if clip_json_path.is_file():
        try:
            clip_meta = json.loads(clip_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            clip_meta = {}

    midi_path: Path | None = args.midi.expanduser().resolve() if args.midi else None
    if midi_path is None and clip_meta.get("synthesized") and isinstance(clip_meta.get("midi_path"), str):
        cand = Path(clip_meta["midi_path"])
        if cand.is_file():
            midi_path = cand

    sf_path: Path | None = args.soundfont.expanduser().resolve() if args.soundfont else None
    if sf_path is None and isinstance(clip_meta.get("soundfont_path"), str):
        cand_sf = Path(clip_meta["soundfont_path"])
        if cand_sf.is_file():
            sf_path = cand_sf
    if sf_path is None and DEFAULT_SOUNDFONT_PATH.is_file():
        sf_path = DEFAULT_SOUNDFONT_PATH

    with tempfile.TemporaryDirectory() as tmp:
        ext_wav = Path(tmp) / "track.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", str(FINE_SR), "-c:a", "pcm_f32le", str(ext_wav)],
            check=True,
        )
        mp4_mono = _load_wav_mono(ext_wav, FINE_SR)

    mp4_dur = len(mp4_mono) / FINE_SR
    print(f"Video audio duration: {mp4_dur:.3f}s ({len(choreo['plates'])} total plates in choreo.json)")

    # 1. MP4 mux sync vs audio.wav
    if wav_path is not None:
        wav_mono = _load_wav_mono(wav_path, FINE_SR)
        n = min(len(mp4_mono), len(wav_mono), FINE_SR * 10)
        max_lag = int(0.5 * FINE_SR)
        if n > max_lag:
            a = wav_mono[:n] - wav_mono[:n].mean()
            b = mp4_mono[:n] - mp4_mono[:n].mean()
            corr = correlate(b, a, mode="full")
            mid = n - 1
            win = corr[mid - max_lag : mid + max_lag + 1]
            best = int(np.argmax(win))
            lag_samples = best - max_lag
            norm = float(np.linalg.norm(a) * np.linalg.norm(b))
            peak_corr = float(win[best] / max(norm, 1e-9))
            print(
                f"Mux sync (MP4 audio vs audio.wav): corr={peak_corr:.4f}, "
                f"lag={lag_samples:+d} samples ({lag_samples / FINE_SR * 1000.0:+.2f} ms)"
            )

    # Filter plates to the rendered audio range (supports partial --frames checks)
    hits = np.array([float(p["t"]) for p in choreo["plates"] if float(p["t"]) <= mp4_dur - 0.05], dtype=np.float64)
    lit_t = np.ceil((hits - HIT_EPS) * args.fps) / args.fps
    snapped = np.abs(hits - np.round(hits * args.fps) / args.fps) <= HIT_EPS
    if hits.size:
        print(
            f"Frame alignment ({args.fps} fps): {int(snapped.sum())}/{hits.size} "
            f"({snapped.mean() * 100:.0f}%) snapped to exact frame boundaries"
        )

    # 2. Acoustic transient sync on MP4 audio
    transients = _detect_fine_audio_transients(mp4_mono, FINE_SR)
    print("Acoustic transient sync (MP4 audio, 44.1 kHz fine flux):")
    _format_stats("choreo plate.t  ", hits, transients, fps=args.fps)
    _format_stats("visual contact  ", lit_t, transients, fps=args.fps)

    # 3. MIDI + SoundFont ground-truth sync
    if midi_path is not None and midi_path.is_file():
        clip_start = float(clip_meta.get("start", 0.0)) if clip_meta.get("clipped") else 0.0
        attack_map = measure_soundfont_attack_map(sf_path) if sf_path is not None and sf_path.is_file() else None
        gt_notes = extract_midi_notes(
            midi_path,
            duration=mp4_dur,
            clip_start=clip_start,
            attack_map=attack_map,
        )
        gt_all = np.array([float(n["t"]) for n in gt_notes], dtype=np.float64)
        print(f"MIDI+SF2 ground truth ({midi_path.name}, clip_start={clip_start:.3f}s):")
        _format_stats("choreo plate.t  ", hits, gt_all, fps=args.fps)
        _format_stats("visual contact  ", lit_t, gt_all, fps=args.fps)
        cov_end = min(mp4_dur, float(choreo.get("duration", mp4_dur))) - 0.20
        gt_cov = gt_all[(gt_all >= 0.25) & (gt_all <= cov_end)]
        if gt_cov.size and hits.size:
            nearest_hit = np.min(np.abs(hits[None, :] - gt_cov[:, None]), axis=1) * 1000.0
            covered = int(np.sum(nearest_hit < 60.0))
            print(f"  MIDI note coverage (<60 ms): {covered}/{gt_cov.size} ({covered / gt_cov.size * 100:.0f}%)")


if __name__ == "__main__":
    main()
