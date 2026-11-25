"""Stage 1: audio analysis. WAV in, analysis dict out.

Everything is derived from one STFT: onsets (with a harmonic/percussive split),
a rough pitch per onset, beats and downbeats, kicks, sections and 60 Hz energy
curves for the ambient effects.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mido
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

SR = 22050
HOP = 128  # 5.8 ms per frame
N_FFT = 1024
ENERGY_FPS = 60

ANALYSIS_VERSION = 3
FINE_SR = 44100
FINE_N_FFT = 512  # 11.6 ms window for sharp onset localization
FINE_HOP = 64  # 1.45 ms hop


def _norm(x: np.ndarray, pct: float = 95.0) -> np.ndarray:
    ref = np.percentile(x, pct) if x.size else 1.0
    return x / ref if ref > 1e-9 else np.zeros_like(x)


def _times(frames) -> np.ndarray:
    return librosa.frames_to_time(np.asarray(frames), sr=SR, hop_length=HOP)


def compute_fine_flux(mono: np.ndarray, sr: int = FINE_SR) -> np.ndarray:
    """Compute a 44.1 kHz, 1.45 ms-hop log-compressed spectral flux curve."""
    arr = np.asarray(mono, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    S_f = np.abs(librosa.stft(arr, n_fft=FINE_N_FFT, hop_length=FINE_HOP, center=True))
    s_ref = float(np.percentile(S_f, 99.5)) if S_f.size else 1.0
    if s_ref <= 1e-9:
        s_ref = 1.0
    C_f = np.log1p(50.0 * (S_f / s_ref))
    flux = np.zeros(C_f.shape[1], dtype=np.float32)
    if C_f.shape[1] > 1:
        flux[1:] = np.maximum(0.0, C_f[:, 1:] - C_f[:, :-1]).mean(axis=0)
    return flux


def _load_fine_mono(wav_path: str | Path) -> np.ndarray:
    path = Path(wav_path)
    info = sf.info(str(path))
    if info.samplerate == FINE_SR:
        audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
        return audio.mean(axis=1)
    y, _ = librosa.load(str(path), sr=FINE_SR, mono=True)
    return np.asarray(y, dtype=np.float32)


def refine_onset_times_from_mono(
    mono: np.ndarray,
    coarse_times: np.ndarray,
    *,
    sr: int = FINE_SR,
    search_back: float = 0.055,
    search_fwd: float = 0.025,
) -> np.ndarray:
    """Refine coarse onset timestamps onto fine spectral-flux transient peaks."""
    coarse = np.asarray(coarse_times, dtype=np.float64)
    if coarse.size == 0:
        return coarse.copy()
    flux = compute_fine_flux(mono, sr=sr)
    if flux.size < 2:
        return coarse.copy()

    dur = len(mono) / sr
    out = np.empty_like(coarse)
    for i, t_c in enumerate(coarse):
        lo_t = t_c - search_back
        if i > 0:
            lo_t = max(lo_t, 0.5 * (coarse[i - 1] + t_c))
        lo_t = max(0.0, lo_t)

        hi_t = t_c + search_fwd
        if i + 1 < len(coarse):
            hi_t = min(hi_t, 0.5 * (t_c + coarse[i + 1]))
        hi_t = min(dur, hi_t)

        k_lo = max(1, int(math.ceil(lo_t * sr / FINE_HOP)))
        k_hi = min(len(flux) - 1, int(math.floor(hi_t * sr / FINE_HOP)))
        if k_hi < k_lo:
            k_mid = min(len(flux) - 1, max(1, int(round(t_c * sr / FINE_HOP))))
            k_lo = k_hi = k_mid

        k_pk = k_lo + int(np.argmax(flux[k_lo : k_hi + 1]))
        if float(flux[k_pk]) <= 1e-9:
            out[i] = t_c
            continue

        if 0 < k_pk < len(flux) - 1:
            y0, y1, y2 = float(flux[k_pk - 1]), float(flux[k_pk]), float(flux[k_pk + 1])
            denom = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            f_ex = k_pk + float(np.clip(delta, -0.5, 0.5))
        else:
            f_ex = float(k_pk)
        out[i] = float(np.clip(f_ex * FINE_HOP / sr, 0.0, dur))
    return out


def refine_onset_times(
    wav_path: str | Path,
    coarse_times: np.ndarray,
    *,
    search_back: float = 0.055,
    search_fwd: float = 0.025,
) -> np.ndarray:
    """Load WAV at 44.1 kHz and refine coarse onset timestamps onto transient peaks."""
    coarse = np.asarray(coarse_times, dtype=np.float64)
    if coarse.size == 0:
        return coarse.copy()
    mono = _load_fine_mono(wav_path)
    return refine_onset_times_from_mono(
        mono, coarse, sr=FINE_SR, search_back=search_back, search_fwd=search_fwd
    )


def extract_midi_notes(
    midi_path: str | Path,
    *,
    duration: float,
    clip_start: float = 0.0,
    attack_map: dict[str, Any] | None = None,
    chord_window: float = 0.035,
    sample_rate: int = 44100,
) -> list[dict]:
    """Extract onset clusters from MIDI, shifted by clip start and SoundFont attack delay."""
    if attack_map is not None:
        from synthesize import lookup_attack_delay
    else:
        lookup_attack_delay = None

    actual_start = round(clip_start * sample_rate) / sample_rate if clip_start > 0 else 0.0
    mid = mido.MidiFile(str(midi_path))
    t_cur = 0.0
    events: list[tuple[float, int, int]] = []
    for msg in mid:
        t_cur += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            delay = lookup_attack_delay(attack_map, msg.note, msg.velocity) if lookup_attack_delay else 0.0
            t_ac = (t_cur - actual_start) + delay
            if 0.0 <= t_ac <= duration:
                events.append((t_ac, int(msg.note), int(msg.velocity)))

    events.sort(key=lambda e: (e[0], e[1]))
    clusters: list[tuple[float, list[tuple[float, int, int]]]] = []
    for t_ac, pitch, vel in events:
        if not clusters or t_ac - clusters[-1][0] > chord_window:
            clusters.append((t_ac, [(t_ac, pitch, vel)]))
        else:
            clusters[-1][1].append((t_ac, pitch, vel))

    notes: list[dict] = []
    for _t0, group in clusters:
        w_sum = sum(v * v for _t, _p, v in group)
        t_cl = sum(t * (v * v) for t, _p, v in group) / max(w_sum, 1)
        max_vel = max(v for _t, _p, v in group)
        top_pitch = max(p for _t, p, v in group if v >= 0.7 * max_vel)
        pitches = [p for _t, p, _v in group]
        notes.append(
            {
                "t": round(float(t_cl), 6),
                "strength": round(float(np.clip(max_vel / 100.0, 0.2, 1.0)), 3),
                "melodic": 0.95,
                "pitch": int(top_pitch),
                "chord_size": len(group),
                "pitch_span": int(max(pitches) - min(pitches)),
                "vel_energy": round(float(math.sqrt(w_sum)) / 100.0, 3),
            }
        )
    notes.sort(key=lambda n: n["t"])
    return notes


def analyze(
    wav_path: str,
    *,
    midi_path: str | Path | None = None,
    clip_start: float = 0.0,
    attack_map: dict[str, Any] | None = None,
) -> dict:
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    duration = len(y) / SR

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    H, P = librosa.decompose.hpss(S)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

    def mel_db(mag):
        return librosa.power_to_db(
            librosa.feature.melspectrogram(S=mag**2, sr=SR, n_mels=64), ref=np.max
        )

    mel_full, mel_h, mel_p = mel_db(S), mel_db(H), mel_db(P)

    def env(m):
        return librosa.onset.onset_strength(S=m, sr=SR, hop_length=HOP)

    env_full, env_h, env_p = env(mel_full), env(mel_h), env(mel_p)

    # --- onsets -----------------------------------------------------------
    if midi_path is not None:
        notes = extract_midi_notes(
            midi_path,
            duration=duration,
            clip_start=clip_start,
            attack_map=attack_map,
        )
        note_source = "midi"
    else:
        wait = int(0.06 * SR / HOP)
        frames = librosa.onset.onset_detect(
            onset_envelope=env_full, sr=SR, hop_length=HOP, backtrack=False, delta=0.07, wait=wait
        )
        coarse_t = _times(frames)
        refined_t = refine_onset_times(wav_path, coarse_t)
        nf, nh, np_ = _norm(env_full), _norm(env_h), _norm(env_p)
        env_ref = float(np.percentile(env_full, 99.5)) if env_full.size else 1.0
        lo, hi = np.searchsorted(freqs, 130.0), np.searchsorted(freqs, 2000.0)
        notes = []
        for f, t_ref in zip(frames, refined_t):
            seg = H[lo:hi, f + 3 : f + 12]
            pitch = None
            if seg.size and seg.max() > 1e-6:
                hz = freqs[lo + int(np.argmax(seg.mean(axis=1)))]
                pitch = int(round(librosa.hz_to_midi(hz)))
            h, p = float(nh[f]), float(np_[f])
            notes.append(
                {
                    "t": round(float(t_ref), 6),
                    "strength": round(float(np.clip(nf[f], 0, 1)), 3),
                    "melodic": round(h / (h + p + 1e-9), 3),
                    "pitch": pitch,
                    "onset_raw": round(float(env_full[f] / env_ref) if env_ref > 1e-9 else 0.0, 3),
                }
            )
        notes.sort(key=lambda n: n["t"])
        note_source = "audio"

    # --- beats and downbeats ---------------------------------------------
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=env_full, sr=SR, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])
    low_band = S[freqs < 150.0].mean(axis=0)
    phase = 0
    if len(beat_frames) >= 8:
        phase = int(np.argmax([low_band[beat_frames[p::4]].mean() for p in range(4)]))
    beats = [round(float(t), 4) for t in _times(beat_frames)]
    downbeats = beats[phase::4]

    # --- kicks ------------------------------------------------------------
    env_kick = librosa.onset.onset_strength(S=mel_p[:6], sr=SR, hop_length=HOP)
    kick_frames = librosa.onset.onset_detect(
        onset_envelope=env_kick, sr=SR, hop_length=HOP, backtrack=False, delta=0.12,
        wait=int(0.12 * SR / HOP),
    )
    nk = _norm(env_kick)
    kicks = [
        {"t": round(float(_times(f)), 4), "strength": round(float(np.clip(nk[f], 0, 1)), 3)}
        for f in kick_frames
        if nk[f] > 0.25
    ]

    # --- energy curves ----------------------------------------------------
    frame_t = _times(np.arange(S.shape[1]))
    out_t = np.arange(0, duration, 1.0 / ENERGY_FPS)

    def curve(x):
        x = gaussian_filter1d(x.astype(float), 3)
        return np.round(np.clip(np.interp(out_t, frame_t, _norm(x, 98)), 0, 1.5), 3).tolist()

    rms = np.sqrt((S**2).mean(axis=0))
    energy = {
        "fps": ENERGY_FPS,
        "rms": curve(rms),
        "low": curve(low_band),
        "high": curve(S[freqs > 4000.0].mean(axis=0)),
    }

    # --- sections ---------------------------------------------------------
    step = max(1, int(0.5 * SR / HOP))
    feat = librosa.feature.mfcc(S=mel_full[:, ::step], n_mfcc=13)
    k = int(np.clip(round(duration / 25.0), 1, 12))
    k = min(k, feat.shape[1])
    bounds = librosa.segment.agglomerative(feat, k) if k > 1 else np.array([0])
    edges = [float(_times(b * step)) for b in bounds] + [duration]
    edges[0] = 0.0
    sections = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (frame_t >= a) & (frame_t < b)
        sections.append({"t0": round(a, 3), "t1": round(b, 3), "energy": float(rms[m].mean()) if m.any() else 0.0})
    top = max((s["energy"] for s in sections), default=1.0) or 1.0
    for s in sections:
        s["energy"] = round(s["energy"] / top, 3)

    return {
        "version": ANALYSIS_VERSION,
        "note_source": note_source,
        "duration": round(duration, 4),
        "tempo": round(tempo, 2),
        "beats": beats,
        "downbeats": downbeats,
        "notes": notes,
        "kicks": kicks,
        "sections": sections,
        "energy": energy,
    }
