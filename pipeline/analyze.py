"""Stage 1: audio analysis. WAV in, analysis dict out.

Everything is derived from one STFT: onsets (with a harmonic/percussive split),
a rough pitch per onset, beats and downbeats, kicks, sections and 60 Hz energy
curves for the ambient effects.
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d

SR = 22050
HOP = 128  # 5.8 ms per frame
N_FFT = 1024
ENERGY_FPS = 60


def _norm(x: np.ndarray, pct: float = 95.0) -> np.ndarray:
    ref = np.percentile(x, pct) if x.size else 1.0
    return x / ref if ref > 1e-9 else np.zeros_like(x)


def _times(frames) -> np.ndarray:
    return librosa.frames_to_time(np.asarray(frames), sr=SR, hop_length=HOP)


def analyze(wav_path: str) -> dict:
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
    wait = int(0.06 * SR / HOP)
    frames = librosa.onset.onset_detect(
        onset_envelope=env_full, sr=SR, hop_length=HOP, backtrack=False, delta=0.07, wait=wait
    )
    nf, nh, np_ = _norm(env_full), _norm(env_h), _norm(env_p)
    lo, hi = np.searchsorted(freqs, 130.0), np.searchsorted(freqs, 2000.0)
    notes = []
    for f in frames:
        seg = H[lo:hi, f + 3 : f + 12]
        pitch = None
        if seg.size and seg.max() > 1e-6:
            hz = freqs[lo + int(np.argmax(seg.mean(axis=1)))]
            pitch = int(round(librosa.hz_to_midi(hz)))
        h, p = float(nh[f]), float(np_[f])
        notes.append(
            {
                "t": round(float(_times(f)), 4),
                "strength": round(float(np.clip(nf[f], 0, 1)), 3),
                "melodic": round(h / (h + p + 1e-9), 3),
                "pitch": pitch,
            }
        )

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
        "duration": round(duration, 4),
        "tempo": round(tempo, 2),
        "beats": beats,
        "downbeats": downbeats,
        "notes": notes,
        "kicks": kicks,
        "sections": sections,
        "energy": energy,
    }
