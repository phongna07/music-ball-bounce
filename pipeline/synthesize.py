"""Local piano transcription (Transkun v2 ONNX) and acoustic synthesis (FluidSynth + SF2).

Turns any audio file (e.g., a YouTube piano cover download) into an expressive
MIDI transcription (notes, velocities, sustain & soft pedals) and renders a
clean, mastered studio-piano WAV locally using FluidSynth and the Salamander
Grand Piano SoundFont.

Usage:
    uv run --project pipeline python pipeline/synthesize.py path/to/song.mp3
    uv run --project pipeline python pipeline/synthesize.py path/to/song.mp3 -o out.wav --midi out.mid
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any
import urllib.request

import mido
import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = ROOT / "work" / "models" / "transkun"
DEFAULT_SOUNDFONT_PATH = ROOT / "work" / "soundfonts" / "salamander-grand-piano.sf2"

HF_TRANSKUN_BASE = "https://huggingface.co/TuesdayCrowd/transkun-onnx/resolve/main"
SALAMANDER_TAR_URL = (
    "https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/"
    "SalamanderGrandPiano-SF2-V3+20200602.tar.xz"
)
TRANSKUN_ASSETS = (
    "transkun.onnx",
    "transkun-heads.onnx",
    "freq2mels.f32",
    "windows.f32",
    "symbols.i32",
    "params.json",
)
SYNTH_CACHE_VERSION = 1


@dataclass
class MidiEvent:
    start: float
    end: float
    pitch: int
    velocity: int
    has_onset: bool = True
    has_offset: bool = True


@dataclass(frozen=True)
class SynthesisResult:
    wav_path: Path
    midi_path: Path
    note_count: int
    pedal_count: int
    duration: float
    soundfont: str
    cached: bool


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path, *, label: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
    req = urllib.request.Request(url, headers={"User-Agent": "MusicBallBounce/1.0"})
    name = label or dest.name
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            total = resp.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None
            done = 0
            last_pct = -10
            while True:
                chunk = resp.read(1 << 18)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total_bytes:
                    pct = int(100 * done / total_bytes)
                    if pct >= last_pct + 20:
                        last_pct = pct
                        print(
                            f"      downloading {name}: {done / 1e6:.1f}/{total_bytes / 1e6:.1f} MB ({pct}%)",
                            file=sys.stderr,
                        )
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _resolve_model_dir(model_dir: Path | None = None) -> Path:
    if model_dir is not None:
        return Path(model_dir).expanduser().resolve()
    env_dir = os.environ.get("MBB_MODELS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return DEFAULT_MODELS_DIR


def ensure_transkun_assets(model_dir: Path | None = None) -> Path:
    """Download Transkun v2 ONNX weights and frozen front-end arrays if missing."""
    dest_dir = _resolve_model_dir(model_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for filename in TRANSKUN_ASSETS:
        target = dest_dir / filename
        if target.is_file() and target.stat().st_size > 0:
            continue
        url = f"{HF_TRANSKUN_BASE}/{filename}"
        print(f"      fetching model asset: {filename}", file=sys.stderr)
        _download_file(url, target, label=filename)
    return dest_dir


def ensure_soundfont(soundfont_path: Path | None = None) -> Path:
    """Resolve a SoundFont path, auto-downloading Salamander Grand Piano V3 SF2 by default."""
    if soundfont_path is not None:
        resolved = Path(soundfont_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"SoundFont not found: {resolved}")
        return resolved

    env_sf = os.environ.get("MBB_SOUNDFONT")
    if env_sf:
        resolved = Path(env_sf).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"MBB_SOUNDFONT file not found: {resolved}")
        return resolved

    target = DEFAULT_SOUNDFONT_PATH
    if target.is_file() and target.stat().st_size > 0:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print("      downloading Salamander Grand Piano V3 SoundFont (~310 MB, one-time)...", file=sys.stderr)
    fd, tmp_tar_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-sf2-", suffix=".tar.xz")
    os.close(fd)
    tmp_tar = Path(tmp_tar_name)
    tmp_sf2 = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        _download_file(SALAMANDER_TAR_URL, tmp_tar, label="SalamanderGrandPiano-SF2-V3.tar.xz")
        print("      extracting SoundFont archive...", file=sys.stderr)
        with tarfile.open(tmp_tar, mode="r:xz") as tf:
            sf2_member = next(
                (m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".sf2")),
                None,
            )
            if sf2_member is None:
                raise RuntimeError("No .sf2 file found inside Salamander Grand Piano archive.")
            src_stream = tf.extractfile(sf2_member)
            if src_stream is None:
                raise RuntimeError(f"Could not extract {sf2_member.name} from archive.")
            with src_stream, tmp_sf2.open("wb") as out:
                shutil.copyfileobj(src_stream, out, length=1 << 20)
        os.replace(tmp_sf2, target)
    except Exception:
        tmp_sf2.unlink(missing_ok=True)
        raise
    finally:
        tmp_tar.unlink(missing_ok=True)
    return target


# ---------------------------------------------------------------------------
# Transkun v2 ONNX transcription engine
# ---------------------------------------------------------------------------
def make_frames(x: np.ndarray, hop_size: int = 1024, window_size: int = 4096) -> np.ndarray:
    """Frame 1D mono audio identically to transkun.Util.makeFrame(leftPaddingHalfFrame=True)."""
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    n = arr.shape[0]
    n_frame = math.ceil(n / hop_size) + 1
    l_pad = window_size // 2
    r_pad = (n_frame - 1) * hop_size + window_size // 2 - n
    xp = np.pad(arr, (l_pad, r_pad), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(xp, window_size)[::hop_size]
    return np.ascontiguousarray(frames, dtype=np.float32)


def extract_features(
    audio_slice: np.ndarray,
    windows: np.ndarray,
    freq2mels: np.ndarray,
    *,
    window_size: int = 4096,
    hop_size: int = 1024,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute [1, T, nMels, nWindows] log-mel features matching Transkun exactly."""
    frames = make_frames(audio_slice, hop_size=hop_size, window_size=window_size)
    mean = float(frames.mean())
    std = float(frames.std(ddof=1)) if frames.size > 1 else 0.0
    frames_n = (frames - mean) / (std + 1e-8)

    windowed = frames_n[:, None, :] * windows[None, :, :]
    spec = np.fft.rfft(windowed, axis=-1, norm="ortho")
    power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)

    mel = np.matmul(power, freq2mels)  # [T, nWindows, nMels]
    mel = np.transpose(mel, (0, 2, 1))  # [T, nMels, nWindows]
    log_eps = math.log(eps)
    mel = (np.log(mel + eps) - log_eps) / (-log_eps)
    return np.ascontiguousarray(mel[None, :, :, :], dtype=np.float32)


def viterbi_backward(
    score: np.ndarray,
    forced_start_pos: list[int] | None = None,
) -> list[list[tuple[int, int]]]:
    """NumPy port of transkun.CRF.NeuralSemiCRFInterval.viterbiBackward (zero noiseScore)."""
    if score.ndim != 3 or score.shape[0] != score.shape[1]:
        raise ValueError(f"Expected score shape [T, T, nSym], got {score.shape}")
    T = score.shape[0]
    n_batch = score.shape[2]
    if T == 0:
        return [[] for _ in range(n_batch)]

    score_t = np.ascontiguousarray(score.transpose(1, 0, 2))  # [begin, end, nSym]
    q = np.zeros((T, n_batch), dtype=np.float32)
    q[T - 1] = np.maximum(score[T - 1, T - 1, :], 0.0)
    ptr = np.empty((max(0, T - 1), n_batch), dtype=np.int32)

    for b in range(T - 2, -1, -1):
        sub_score = score_t[b, b + 1 :, :]  # [T - 1 - b, nSym]
        skip = q[b + 1 : b + 2, :]  # [1, nSym]
        intervals = q[b + 1 :, :] + sub_score  # [T - 1 - b, nSym]
        tmp = np.concatenate([skip, intervals], axis=0)
        selection = np.argmax(tmp, axis=0)
        cur_v = np.take_along_axis(tmp, selection[None, :], axis=0)[0]
        ptr[b] = selection.astype(np.int32) - 1
        diag_pos = np.maximum(score[b, b, :], 0.0)
        q[b] = cur_v + diag_pos

    diag_inc = np.diagonal(score, axis1=0, axis2=1) > 0.0  # [nSym, T]
    starts = forced_start_pos if forced_start_pos is not None else [0] * n_batch

    result: list[list[tuple[int, int]]] = []
    for idx in range(n_batch):
        j = int(starts[idx])
        cur: list[tuple[int, int]] = []
        cur_diag = diag_inc[idx]
        while j < T - 1:
            if cur_diag[j]:
                cur.append((j, j))
            sel = int(ptr[j, idx])
            if sel < 0:
                j += 1
            else:
                end_idx = sel + j + 1
                cur.append((j, end_idx))
                j = end_idx
        if score[T - 1, T - 1, idx] > 0.0:
            cur.append((T - 1, T - 1))
        result.append(cur)
    return result


def continuous_bernoulli_mean(logits: np.ndarray) -> np.ndarray:
    """Compute ContinuousBernoulli(logits).mean stably for arbitrary logit magnitudes."""
    x = np.clip(logits.astype(np.float64), -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-x))
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    near_half = np.abs(p - 0.5) < 1e-4
    safe_p = np.where(near_half, 0.25, p)
    exact = safe_p / (2.0 * safe_p - 1.0) + 1.0 / (2.0 * np.arctanh(1.0 - 2.0 * safe_p))
    taylor = 0.5 + (p - 0.5) / 3.0
    return np.where(near_half, taylor, exact).astype(np.float32)


def resolve_overlapping(events: list[MidiEvent]) -> list[MidiEvent]:
    """Clamp overlapping same-pitch intervals and drop zero-length notes."""
    ordered = sorted(events, key=lambda e: (e.start, e.end, e.pitch))
    out: list[MidiEvent] = []
    last_by_pitch: dict[int, int] = {}
    for ev in ordered:
        prev_idx = last_by_pitch.get(ev.pitch)
        if prev_idx is not None and out[prev_idx].end > ev.start:
            out[prev_idx].end = ev.start
        last_by_pitch[ev.pitch] = len(out)
        out.append(ev)
    out.sort(key=lambda e: (e.start, e.end, e.pitch))
    return [e for e in out if e.start < e.end]


def _load_mono_audio(audio_path: Path, target_sr: int) -> np.ndarray:
    """Decode any audio/video file to mono float32 at target_sr via ffmpeg + soundfile."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required on PATH to decode audio for synthesis")
    with tempfile.TemporaryDirectory(prefix="mbb-decode-") as tmp:
        wav_tmp = Path(tmp) / "mono.wav"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(audio_path),
                "-vn", "-ac", "1", "-ar", str(target_sr),
                "-c:a", "pcm_f32le",
                str(wav_tmp),
            ],
            check=True,
        )
        y, sr = sf.read(str(wav_tmp), dtype="float32", always_2d=False)
        if sr != target_sr:
            raise RuntimeError(f"Expected {target_sr} Hz WAV, got {sr} Hz")
        return np.asarray(y, dtype=np.float32).reshape(-1)


def _create_ort_session(onnx_path: Path, intra_threads: int | None = None) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    cpus = os.cpu_count() or 4
    opts.intra_op_num_threads = intra_threads or min(cpus, 16)
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def transcribe_audio_to_events(
    audio_path: Path,
    model_dir: Path | None = None,
    *,
    intra_threads: int | None = None,
) -> tuple[list[MidiEvent], float]:
    """Transcribe piano audio to note and pedal MidiEvents using Transkun v2 ONNX."""
    mdir = ensure_transkun_assets(model_dir)
    params = json.loads((mdir / "params.json").read_text(encoding="utf-8"))
    fs = int(params["fs"])
    window_size = int(params["windowSize"])
    hop_size = int(params["hopSize"])
    n_mels = int(params["nMels"])
    n_wins = int(params["nWindows"])
    eps = float(params["eps"])
    seg_sec = float(params["segmentSizeSeconds"])
    hop_sec = float(params["segmentHopSeconds"])
    n_sym = int(params["nSymbols"])

    freq2mels = np.fromfile(mdir / "freq2mels.f32", dtype="<f4").reshape(window_size // 2 + 1, n_mels)
    windows = np.fromfile(mdir / "windows.f32", dtype="<f4").reshape(n_wins, window_size)
    symbols = np.fromfile(mdir / "symbols.i32", dtype="<i4").reshape(n_sym).tolist()

    main_sess = _create_ort_session(mdir / "transkun.onnx", intra_threads=intra_threads)
    heads_sess = _create_ort_session(mdir / "transkun-heads.onnx", intra_threads=intra_threads)

    audio = _load_mono_audio(audio_path, fs)
    source_duration = len(audio) / fs
    if source_duration < 0.1:
        return [], source_duration

    pad_time_begin = seg_sec - hop_sec
    pad_samples = math.ceil(pad_time_begin * fs)
    padded = np.pad(audio, (pad_samples, pad_samples), mode="constant")
    n_sample = padded.shape[0]

    step_size = math.ceil(hop_sec * fs / hop_size) * hop_size
    segment_size = math.ceil(seg_sec * fs)
    step_frames = step_size // hop_size
    last_frame_idx = round(segment_size / hop_size)
    start_frame_idx = math.floor(pad_time_begin * fs / hop_size)
    start_pos = [start_frame_idx] * n_sym
    frame_dur = hop_size / fs

    events_by_type: dict[int, list[MidiEvent]] = defaultdict(list)

    for i in range(0, n_sample, step_size):
        j = min(i + segment_size, n_sample)
        begin_time = i / fs - pad_time_begin
        if begin_time >= source_duration + 1.0:
            break

        cur_slice = padded[i:j]
        if cur_slice.shape[0] < segment_size:
            cur_slice = np.pad(cur_slice, (0, segment_size - cur_slice.shape[0]), mode="constant")

        features = extract_features(
            cur_slice,
            windows,
            freq2mels,
            window_size=window_size,
            hop_size=hop_size,
            eps=eps,
        )
        S, ctx = main_sess.run(["S", "ctx"], {"featuresBatch": features})
        paths = viterbi_backward(S, forced_start_pos=start_pos)

        n_intervals = sum(len(p) for p in paths)
        if n_intervals == 0:
            start_pos = [0] * n_sym
            continue

        sym_indices: list[int] = []
        a_indices: list[int] = []
        b_indices: list[int] = []
        for sym_idx, p_list in enumerate(paths):
            for a_fr, b_fr in p_list:
                sym_indices.append(sym_idx)
                a_indices.append(a_fr)
                b_indices.append(b_fr)

        ctx_a = ctx[sym_indices, a_indices]
        ctx_b = ctx[sym_indices, b_indices]
        attr = np.ascontiguousarray(np.concatenate([ctx_a, ctx_b, ctx_a * ctx_b], axis=-1), dtype=np.float32)
        vel_logits, of_raw = heads_sess.run(["velLogits", "ofRaw"], {"attr": attr})

        velocities = np.argmax(vel_logits, axis=-1)
        of_val = np.clip((continuous_bernoulli_mean(of_raw[:, :2]) - 0.5) / 0.99, -0.5, 0.5)
        of_pres = of_raw[:, 2:] > 0.0

        seg_events: list[MidiEvent] = []
        last_p: list[int] = []
        idx = 0
        for sym_idx, event_type in enumerate(symbols):
            last_end = 0.0
            cur_last_p = 0
            for a_fr, b_fr in paths[sym_idx]:
                vel = int(velocities[idx])
                s_local = (a_fr + float(of_val[idx, 0])) * frame_dur
                e_local = (b_fr + float(of_val[idx, 1])) * frame_dur
                has_onset = (a_fr > 0) or bool(of_pres[idx, 0])
                has_offset = (b_fr < last_frame_idx) or bool(of_pres[idx, 1])

                s_local = max(s_local, last_end)
                e_local = max(e_local, s_local + 1e-8)
                last_end = e_local

                s_abs = max(0.0, s_local + begin_time)
                e_abs = max(s_abs, e_local + begin_time)
                seg_events.append(
                    MidiEvent(
                        start=s_abs,
                        end=e_abs,
                        pitch=int(event_type),
                        velocity=vel,
                        has_onset=has_onset,
                        has_offset=has_offset,
                    )
                )
                if has_offset:
                    cur_last_p = b_fr
                idx += 1
            last_p.append(cur_last_p)

        start_pos = [max(k - step_frames, 0) for k in last_p]
        seg_events.sort(key=lambda e: (e.start, e.end, e.pitch))

        for ev in seg_events:
            by_pitch = events_by_type[ev.pitch]
            if by_pitch:
                prev = by_pitch[-1]
                if ev.start < prev.end:
                    if ev.has_onset:
                        by_pitch[-1] = ev
                    else:
                        prev.has_offset = ev.has_offset
                        prev.end = max(ev.end, prev.end)
                    continue
            if ev.has_onset:
                by_pitch.append(ev)

    for by_pitch in events_by_type.values():
        if by_pitch:
            by_pitch[-1].has_offset = True

    events_all = [
        ev
        for by_pitch in events_by_type.values()
        for ev in by_pitch
        if ev.has_offset and ev.start < source_duration + 0.5
    ]
    for ev in events_all:
        ev.end = min(ev.end, source_duration + 0.8)

    return resolve_overlapping(events_all), source_duration


def write_events_to_midi(events: list[MidiEvent], midi_path: Path, *, resolution: int = 960) -> None:
    """Write transcribed piano note and pedal events to a standard Type-1 MIDI file."""
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    mid = mido.MidiFile(type=1, ticks_per_beat=resolution)

    tempo_us = 500_000  # 120 BPM -> resolution * 2 ticks per second
    ticks_per_sec = resolution * (1_000_000.0 / tempo_us)

    meta_track = mido.MidiTrack()
    meta_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    meta_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta_track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta_track)

    # Sort priority on identical ticks: CC off (0) -> note_off (1) -> CC on (2) -> note_on (3)
    raw_msgs: list[tuple[int, int, int, mido.Message]] = []
    for ev in events:
        t_on = max(0, int(round(ev.start * ticks_per_sec)))
        t_off = max(t_on + 1, int(round(ev.end * ticks_per_sec)))
        if ev.pitch > 0:
            pitch = int(np.clip(ev.pitch, 21, 108))
            vel = int(np.clip(ev.velocity, 1, 127))
            raw_msgs.append((t_on, 3, pitch, mido.Message("note_on", channel=0, note=pitch, velocity=vel, time=0)))
            raw_msgs.append((t_off, 1, pitch, mido.Message("note_off", channel=0, note=pitch, velocity=0, time=0)))
        else:
            cc = int(-ev.pitch)
            val = int(np.clip(ev.velocity, 0, 127))
            # FluidSynth engages CC64 sustain only for values >= 64
            if cc == 64:
                val = max(64, val)
            raw_msgs.append((t_on, 2, cc, mido.Message("control_change", channel=0, control=cc, value=val, time=0)))
            raw_msgs.append((t_off, 0, cc, mido.Message("control_change", channel=0, control=cc, value=0, time=0)))

    raw_msgs.sort(key=lambda item: (item[0], item[1], item[2]))

    piano_track = mido.MidiTrack()
    piano_track.append(mido.Message("program_change", channel=0, program=0, time=0))
    prev_tick = 0
    for tick, _prio, _code, msg in raw_msgs:
        delta = max(0, tick - prev_tick)
        prev_tick = tick
        piano_track.append(msg.copy(time=delta))
    piano_track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(piano_track)

    tmp = midi_path.with_name(f".{midi_path.name}.tmp.{os.getpid()}")
    try:
        mid.save(str(tmp))
        os.replace(tmp, midi_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# FluidSynth rendering & transparent mastering
# ---------------------------------------------------------------------------
def render_midi_to_wav(
    midi_path: Path,
    wav_path: Path,
    soundfont_path: Path,
    *,
    sample_rate: int = 44100,
    gain: float = 0.8,
    reverb: bool = True,
    reverb_room: float = 0.55,
    reverb_damp: float = 0.40,
    reverb_width: float = 0.80,
    reverb_level: float = 0.35,
) -> None:
    """Render a MIDI file to raw stereo WAV using FluidSynth."""
    if shutil.which("fluidsynth") is None:
        raise RuntimeError(
            "fluidsynth is required on PATH to synthesize piano audio "
            "(install with: apt-get install -y fluidsynth  or  brew install fluid-synth)"
        )
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = wav_path.with_name(f".{wav_path.name}.tmp.{os.getpid()}.wav")
    cmd = [
        "fluidsynth",
        "-n",
        "-i",
        "-F", str(tmp),
        "-r", str(sample_rate),
        "-T", "wav",
        "-g", str(gain),
        "-o", "synth.polyphony=256",
        "-o", f"synth.reverb.active={'1' if reverb else '0'}",
        "-o", "synth.chorus.active=0",
    ]
    if reverb:
        cmd += [
            "-o", f"synth.reverb.room-size={reverb_room}",
            "-o", f"synth.reverb.damp={reverb_damp}",
            "-o", f"synth.reverb.width={reverb_width}",
            "-o", f"synth.reverb.level={reverb_level}",
        ]
    cmd += [str(soundfont_path), str(midi_path)]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            detail = (res.stderr or res.stdout or "").strip()
            raise RuntimeError(f"fluidsynth failed (exit {res.returncode}): {detail}")
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError("fluidsynth exited 0 but produced an empty WAV file")
        os.replace(tmp, wav_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def master_piano_wav(
    raw_wav: Path,
    out_wav: Path,
    *,
    target_rms_db: float = -16.0,
    target_peak_db: float = -1.0,
    max_gain_db: float = 18.0,
) -> dict[str, float]:
    """Normalize loudness and apply soft-knee peak protection while preserving t=0 alignment."""
    audio, sr = sf.read(str(raw_wav), dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)

    # Trim only excessive trailing digital silence past the reverb decay (never trim leading silence).
    abs_frame = np.max(np.abs(audio), axis=1)
    silence_thresh = 10.0 ** (-70.0 / 20.0)
    non_silent = np.flatnonzero(abs_frame > silence_thresh)
    if non_silent.size > 0:
        end_idx = min(audio.shape[0], int(non_silent[-1]) + int(0.5 * sr))
        audio = audio[:end_idx]

    # Measure active RMS across frames above -45 dBFS so pauses don't skew loudness.
    hop = max(1, int(0.05 * sr))
    n_blocks = max(1, audio.shape[0] // hop)
    trimmed = audio[: n_blocks * hop].reshape(n_blocks, hop, audio.shape[1])
    block_rms = np.sqrt(np.mean(trimmed * trimmed, axis=(1, 2)))
    active_thresh = 10.0 ** (-45.0 / 20.0)
    active = block_rms[block_rms > active_thresh]
    if active.size > 0:
        rms = float(np.sqrt(np.mean(active * active)))
    else:
        rms = float(np.sqrt(np.mean(audio * audio)))

    peak_before = float(np.max(np.abs(audio))) if audio.size else 0.0
    target_rms_lin = 10.0 ** (target_rms_db / 20.0)
    target_peak_lin = 10.0 ** (target_peak_db / 20.0)
    max_gain_lin = 10.0 ** (max_gain_db / 20.0)

    if rms > 1e-8:
        gain = min(target_rms_lin / rms, max_gain_lin)
    elif peak_before > 1e-8:
        gain = min(target_peak_lin / peak_before, max_gain_lin)
    else:
        gain = 1.0

    scaled = audio * gain
    peak_scaled = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if peak_scaled > target_peak_lin:
        overshoot_db = 20.0 * math.log10(peak_scaled / target_peak_lin)
        if overshoot_db < 2.0:
            scaled = scaled * (target_peak_lin / peak_scaled)
        else:
            knee = 0.75 * target_peak_lin
            head = target_peak_lin - knee
            mag = np.abs(scaled)
            over = mag > knee
            limited = np.where(over, knee + head * np.tanh((mag - knee) / max(head, 1e-6)), mag)
            scaled = np.sign(scaled) * limited

    scaled = np.clip(scaled, -target_peak_lin, target_peak_lin).astype(np.float32)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_wav.with_name(f".{out_wav.name}.tmp.{os.getpid()}.wav")
    try:
        sf.write(str(tmp), scaled, sr, subtype="PCM_16")
        os.replace(tmp, out_wav)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    final_rms = float(np.sqrt(np.mean(scaled * scaled))) if scaled.size else 0.0
    final_peak = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    return {
        "duration": round(scaled.shape[0] / sr, 4),
        "gain_db": round(20.0 * math.log10(max(gain, 1e-9)), 2),
        "rms_db": round(20.0 * math.log10(max(final_rms, 1e-9)), 2),
        "peak_db": round(20.0 * math.log10(max(final_peak, 1e-9)), 2),
    }


# ---------------------------------------------------------------------------
# High-level cached synthesizer
# ---------------------------------------------------------------------------
def synthesize_piano_version(
    source_audio: Path,
    cache_dir: Path,
    *,
    soundfont: Path | None = None,
    model_dir: Path | None = None,
    resynthesize: bool = False,
    reverb: bool = True,
    gain: float = 0.8,
) -> SynthesisResult:
    """Transcribe source_audio to MIDI and synthesize a mastered studio piano WAV."""
    source_audio = Path(source_audio).expanduser().resolve()
    if not source_audio.is_file():
        raise FileNotFoundError(f"Source audio not found: {source_audio}")

    sf_path = ensure_soundfont(soundfont)
    sf_stat = sf_path.stat()
    src_sha1 = _sha1_file(source_audio)

    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    midi_path = cache_dir / "transcribed.mid"
    wav_path = cache_dir / "synthesized.wav"
    meta_path = cache_dir / "synth_meta.json"

    old_meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old_meta = {}

    same_midi = (
        not resynthesize
        and old_meta.get("synth_version") == SYNTH_CACHE_VERSION
        and old_meta.get("source_sha1") == src_sha1
        and midi_path.is_file()
        and midi_path.stat().st_size > 0
    )
    same_render = (
        same_midi
        and old_meta.get("soundfont_name") == sf_path.name
        and old_meta.get("soundfont_size") == sf_stat.st_size
        and old_meta.get("reverb") == bool(reverb)
        and old_meta.get("gain") == float(gain)
        and wav_path.is_file()
        and wav_path.stat().st_size > 0
    )

    if same_render:
        return SynthesisResult(
            wav_path=wav_path,
            midi_path=midi_path,
            note_count=int(old_meta.get("note_count", 0)),
            pedal_count=int(old_meta.get("pedal_count", 0)),
            duration=float(old_meta.get("duration") or sf.info(str(wav_path)).duration),
            soundfont=sf_path.name,
            cached=True,
        )

    if same_midi:
        note_count = int(old_meta.get("note_count", 0))
        pedal_count = int(old_meta.get("pedal_count", 0))
    else:
        t0 = time.time()
        print("      transcribing piano audio to MIDI (Transkun v2 ONNX)...", file=sys.stderr)
        events, _src_dur = transcribe_audio_to_events(source_audio, model_dir=model_dir)
        note_count = sum(1 for e in events if e.pitch > 0)
        pedal_count = sum(1 for e in events if e.pitch < 0)
        write_events_to_midi(events, midi_path)
        print(
            f"      transcribed {note_count} notes, {pedal_count} pedal events in {time.time() - t0:.1f}s",
            file=sys.stderr,
        )

    t1 = time.time()
    print(f"      synthesizing with FluidSynth ({sf_path.name})...", file=sys.stderr)
    with tempfile.TemporaryDirectory(dir=cache_dir, prefix=".tmp-synth-") as tmp:
        raw_wav = Path(tmp) / "raw.wav"
        render_midi_to_wav(midi_path, raw_wav, sf_path, gain=gain, reverb=reverb)
        master_stats = master_piano_wav(raw_wav, wav_path)

    dur = float(master_stats["duration"])
    print(
        f"      synthesized {dur:.1f}s WAV (RMS {master_stats['rms_db']:.1f} dBFS, "
        f"peak {master_stats['peak_db']:.1f} dBFS) in {time.time() - t1:.1f}s",
        file=sys.stderr,
    )

    meta = {
        "synth_version": SYNTH_CACHE_VERSION,
        "source_sha1": src_sha1,
        "source_name": source_audio.name,
        "soundfont_name": sf_path.name,
        "soundfont_size": sf_stat.st_size,
        "reverb": bool(reverb),
        "gain": float(gain),
        "note_count": note_count,
        "pedal_count": pedal_count,
        "duration": dur,
        "master": master_stats,
    }
    meta_tmp = meta_path.with_name(f".{meta_path.name}.tmp.{os.getpid()}")
    meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(meta_tmp, meta_path)

    return SynthesisResult(
        wav_path=wav_path,
        midi_path=midi_path,
        note_count=note_count,
        pedal_count=pedal_count,
        duration=dur,
        soundfont=sf_path.name,
        cached=False,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="input audio/video file to transcribe & synthesize")
    ap.add_argument("-o", "--out", type=Path, help="output WAV path")
    ap.add_argument("--midi", type=Path, help="optional output MIDI path")
    ap.add_argument("--soundfont", type=Path, help="SF2/SF3 soundfont path")
    ap.add_argument("--model-dir", type=Path, help="directory for Transkun ONNX assets")
    ap.add_argument("--gain", type=float, default=0.8, help="FluidSynth output gain (default: 0.8)")
    ap.add_argument("--no-reverb", action="store_true", help="disable FluidSynth reverb")
    ap.add_argument("--resynthesize", action="store_true", help="ignore cache and re-transcribe/re-render")
    args = ap.parse_args(argv)

    src = args.input.expanduser().resolve()
    if not src.is_file():
        sys.exit(f"Input file not found: {src}")

    cache_dir = ROOT / "work" / "synth_cli" / _sha1_file(src)[:12]
    res = synthesize_piano_version(
        src,
        cache_dir,
        soundfont=args.soundfont,
        model_dir=args.model_dir,
        resynthesize=args.resynthesize,
        reverb=not args.no_reverb,
        gain=args.gain,
    )

    if args.out:
        out_wav = args.out.expanduser().resolve()
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(res.wav_path, out_wav)
    else:
        out_wav = res.wav_path

    if args.midi:
        out_mid = args.midi.expanduser().resolve()
        out_mid.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(res.midi_path, out_mid)
    else:
        out_mid = res.midi_path

    state = "cached" if res.cached else "synthesized"
    print(f"[{state}] {res.duration:.2f}s | {res.note_count} notes, {res.pedal_count} pedals | {res.soundfont}")
    print(f"  WAV:  {out_wav}")
    print(f"  MIDI: {out_mid}")


if __name__ == "__main__":
    main()
