"""Automated discovery of popular, slow-paced, piano-style music on YouTube.

Finds high-view songs (either by discovering popular original music videos and
matching them to piano covers, or by searching piano covers directly), screens
metadata, downloads a short audio preview, and verifies both piano acoustics and
slow rhythm pace using the same onset/choreography rules as the MBB pipeline.

Examples:
    # Find 5 popular songs with verified slow piano covers
    uv run --project pipeline python pipeline/discover.py --count 5

    # Search around a specific era or theme
    uv run --project pipeline python pipeline/discover.py --query "2010s pop ballads" --count 5

    # Find & vet the best slow piano version of specific song(s)
    uv run --project pipeline python pipeline/discover.py --song "Adele - Someone Like You"

    # List previously discovered qualifying tracks from the local catalog
    uv run --project pipeline python pipeline/discover.py --list
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from typing import Any, Callable

import librosa
import numpy as np

from analyze import HOP, N_FFT, SR
from choreograph import split_notes
from youtube import download_youtube_mp3, extract_video_id, is_youtube_url, slugify

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = ROOT / "work" / "discover" / "catalog.json"

# Built-in seed pools covering widely known hits across eras and genres.
DEFAULT_POPULAR_QUERIES: tuple[str, ...] = (
    "most viewed pop ballads official music video",
    "2010s iconic pop songs official music video",
    "2000s hit love songs official music video",
    "famous sad pop songs official music video",
    "classic pop love songs official music video",
    "2020s popular emotional songs official music video",
    "iconic movie theme songs official video",
    "classic rock ballads official music video",
    "most popular r&b ballads official music video",
    "famous disney and movie ballads official video",
    "most viewed acoustic pop songs official music video",
    "famous anime soundtrack ost official video",
)

DEFAULT_DIRECT_PIANO_QUERIES: tuple[str, ...] = (
    "slow piano cover popular song",
    "easy slow piano ballad cover",
    "soft minimalist piano cover pop song",
    "piano lullaby popular song",
    "slow emotional solo piano cover",
    "movie soundtrack slow solo piano",
)

# Keywords indicating piano-style audio.
_PIANO_POSITIVE_RE = re.compile(
    r"\b(piano|pianist|grand\s+piano|solo\s+piano|acoustic\s+piano|synthesia|piano\s+cover|piano\s+version|piano\s+instrumental|piano\s+arrangement)\b",
    re.IGNORECASE,
)

# Positive ranking signal for genuinely slow/sparse arrangements.
_SLOW_PIANO_HINT_RE = re.compile(
    r"\b(slow|slowed|easy|simple|beginner|lullaby|soft|gentle|minimalist|calm|peaceful|relaxing|acoustic|quiet)\b",
    re.IGNORECASE,
)

# Keywords indicating compilations, tutorials, vocals, loops, or non-track content.
_COMPILATION_OR_NOISE_RE = re.compile(
    r"("
    r"\b(compilation|playlist|medley|mashup|full\s+album|nonstop|non-stop|megamix)\b|"
    r"\b(top\s+\d+|best\s+of|\d+\s+most\s+popular|\d+\s+best|\d+\s+songs|\d+\s+piano)\b|"
    r"\b(\d+\s*hours?|\d+\s*hrs?|1\s*hr|10\s*hours?|loop|extended\s+version)\b|"
    r"\b(reaction|reacts|interview|behind\s+the\s+scenes|documentary|podcast|review|ranking|tier\s+list)\b|"
    r"\b(live\s+at|live\s+in|live\s+performance|concert|festival|tour\s+vlog)\b|"
    r"#shorts\b|\bshorts\b"
    r")",
    re.IGNORECASE,
)

_PIANO_NEGATIVE_RE = re.compile(
    r"("
    r"\b(tutorial|lesson|how\s+to\s+play|easy\s+tutorial|slow\s+tutorial|learn\s+to\s+play|explained|chord\s+progression)\b|"
    r"\b(karaoke|lyrics|lyric\s+video|with\s+vocals?|piano\s*&\s*vocal|piano\s+and\s+vocal|vocal\s+cover|singing|sung\s+by|duet\s+with\s+vocal)\b|"
    r"\b(nightcore|sped\s+up|speed\s+up|8d\s+audio|bass\s+boosted|remix|edm|dubstep|trap\s+remix|hardstyle|metal\s+cover|rock\s+cover|guitar\s+cover|violin\s+cover|cello\s+cover|sax\s+cover|orchestra\s+cover)\b|"
    r"\b(sleep\s+music|study\s+music|relaxing\s+music\s+for|rain\s+sounds|fireplace)\b"
    r")",
    re.IGNORECASE,
)

_ORIGINAL_NEGATIVE_RE = re.compile(
    r"\b(piano\s+cover|piano\s+version|karaoke|instrumental|tutorial|cover\s+by|remix|sped\s+up|slowed|nightcore|lyrics|lyric\s+video)\b",
    re.IGNORECASE,
)

_BRACKET_NOISE_RE = re.compile(
    r"[\(\[\{][^)\]\}]*?"
    r"(official|video|audio|lyric|lyrics|hd|4k|hq|remaster|remastered|visualizer|mv|m/v|explicit|clean|version|edit|radio|live|performance|music\s+video|full\s+song|with\s+lyrics)"
    r"[^)\]\}]*[\)\]\}]",
    re.IGNORECASE,
)

_TRAILING_NOISE_RE = re.compile(
    r"\s*[\|\-–—:]\s*(official\s+(music\s+)?video|official\s+audio|lyric\s+video|lyrics|4k|hd|hq|mv|m/v|vevo).*$",
    re.IGNORECASE,
)

_FEAT_RE = re.compile(r"\b(ft\.?|feat\.?|featuring)\b.*$", re.IGNORECASE)

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "by", "with",
    "from", "at", "is", "it", "ft", "feat", "featuring", "official", "music",
    "video", "audio", "song", "songs", "piano", "cover", "version", "solo",
    "instrumental", "arrangement", "acoustic", "hd", "4k", "hq", "mv", "slow",
}


@dataclass(frozen=True)
class VideoCandidate:
    video_id: str
    url: str
    title: str
    channel: str = ""
    duration: float | None = None
    view_count: int | None = None
    like_count: int | None = None
    upload_date: str | None = None
    is_live: bool = False
    source_query: str = ""


@dataclass(frozen=True)
class DiscoveryCriteria:
    min_original_views: int = 50_000_000
    min_piano_views: int = 20_000
    min_duration: float = 90.0
    max_duration: float = 420.0
    min_onsets_per_sec: float = 0.5
    max_onsets_per_sec: float = 2.0
    max_p90_onsets_per_sec: float = 2.5
    max_peak_onsets_per_sec: float = 4.0
    min_median_ioi: float = 0.40
    fast_ioi_threshold: float = 0.25
    max_fast_frac: float = 0.20
    max_sensitive_onsets_per_sec: float = 2.5
    max_sensitive_p90_onsets_per_sec: float = 3.2
    min_sensitive_median_ioi: float = 0.32
    max_duo_gain: float = 1.20
    min_piano_score: float = 0.65
    min_gap: float = 0.17


@dataclass(frozen=True)
class AudioAssessment:
    sample_duration: float
    windows_analyzed: int
    total_onsets: int
    onsets_per_sec: float
    p90_onsets_per_sec: float
    peak_onsets_per_sec: float
    picked_per_sec: float
    median_ioi: float
    p25_ioi: float
    fast_frac: float
    sensitive_onsets_per_sec: float
    sensitive_p90_onsets_per_sec: float
    sensitive_median_ioi: float
    sensitive_fast_frac: float
    duo_gain: float
    tempo: float
    harmonic_ratio: float
    spectral_flatness: float
    spectral_centroid: float
    spectral_rolloff: float
    high_freq_ratio: float
    piano_band_ratio: float
    rms_db: float
    piano_score: float
    pace_score: float


@dataclass
class EvaluatedTrack:
    video_id: str
    url: str
    title: str
    channel: str
    duration: float | None
    view_count: int | None
    mode: str
    song_key: str
    source_query: str
    accepted: bool
    score: float
    rejection_reasons: list[str]
    evaluated_at: str
    original: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluatedTrack:
        return cls(
            video_id=str(data["video_id"]),
            url=str(data["url"]),
            title=str(data["title"]),
            channel=str(data.get("channel") or ""),
            duration=data.get("duration"),
            view_count=data.get("view_count"),
            mode=str(data.get("mode") or "direct-piano"),
            song_key=str(data.get("song_key") or ""),
            source_query=str(data.get("source_query") or ""),
            accepted=bool(data.get("accepted")),
            score=float(data.get("score") or 0.0),
            rejection_reasons=list(data.get("rejection_reasons") or []),
            evaluated_at=str(data.get("evaluated_at") or ""),
            original=data.get("original"),
            metrics=data.get("metrics"),
            used=bool(data.get("used", False)),
        )


# ---------------------------------------------------------------------------
# Text normalization & metadata screening
# ---------------------------------------------------------------------------
def _ascii_tokens(text: str) -> list[str]:
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return [t for t in re.findall(r"[a-z0-9]+", norm.lower()) if t]


def significant_tokens(text: str) -> set[str]:
    return {t for t in _ascii_tokens(text) if len(t) >= 2 and t not in _STOP_WORDS}


def clean_song_title(title: str) -> str:
    """Strip YouTube video noise to get a clean '<artist> <song>' search query."""
    cleaned = title.strip()
    cleaned = _BRACKET_NOISE_RE.sub(" ", cleaned)
    cleaned = _TRAILING_NOISE_RE.sub("", cleaned)
    # Remove remaining empty brackets/parens
    cleaned = re.sub(r"[\(\[\{]\s*[\)\]\}]", " ", cleaned)
    # Remove trailing featured artist clauses inside main title for cleaner piano search
    # Keep track title before 'ft.' / 'feat.' when there is already a clear artist - title
    if " - " in cleaned:
        artist, rest = cleaned.split(" - ", 1)
        artist = _FEAT_RE.sub("", artist).strip(" -–—|,")
        rest = _FEAT_RE.sub("", rest).strip(" -–—|,")
        cleaned = f"{artist} {rest}".strip()
    else:
        cleaned = _FEAT_RE.sub("", cleaned)
    cleaned = re.sub(r"[|•·]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—:,")
    return cleaned


def song_identity_key(text: str) -> str:
    """Canonical key for deduplicating different covers of the same underlying song."""
    cleaned = clean_song_title(text)
    toks = sorted(significant_tokens(cleaned))
    if not toks:
        return slugify(text, fallback="unknown")
    return "-".join(toks[:6])


def titles_match_song(target_song: str, candidate_title: str) -> bool:
    """Check whether a piano candidate title plausibly matches the target song."""
    target_toks = significant_tokens(target_song)
    cand_toks = significant_tokens(candidate_title)
    if not target_toks or not cand_toks:
        return False
    overlap = target_toks & cand_toks
    if len(target_toks) == 1:
        return len(overlap) == 1
    if len(target_toks) <= 3:
        return len(overlap) >= 2 or len(overlap) / len(target_toks) >= 0.6
    # Require at least 2 matching significant tokens and >= 40% coverage
    return len(overlap) >= 2 and (len(overlap) / len(target_toks)) >= 0.4


def screen_original_candidate(
    cand: VideoCandidate,
    criteria: DiscoveryCriteria,
    *,
    require_views: bool = True,
) -> list[str]:
    """Return rejection reasons for a candidate popular original music video."""
    reasons: list[str] = []
    if cand.is_live:
        reasons.append("live stream")
    if cand.duration is not None:
        if cand.duration < criteria.min_duration:
            reasons.append(f"too short ({cand.duration:.0f}s < {criteria.min_duration:.0f}s)")
        elif cand.duration > max(criteria.max_duration, 480.0):
            reasons.append(f"too long ({cand.duration:.0f}s)")
    if _COMPILATION_OR_NOISE_RE.search(cand.title):
        reasons.append("compilation/non-song title")
    if _ORIGINAL_NEGATIVE_RE.search(cand.title):
        reasons.append("not an original track")
    if not clean_song_title(cand.title):
        reasons.append("unparseable title")
    if require_views and cand.view_count is not None and cand.view_count < criteria.min_original_views:
        reasons.append(f"low original views ({cand.view_count:,} < {criteria.min_original_views:,})")
    return reasons


def screen_piano_candidate(
    cand: VideoCandidate,
    criteria: DiscoveryCriteria,
    *,
    target_song: str | None = None,
    require_views: bool = True,
) -> list[str]:
    """Return rejection reasons for a candidate piano cover/track based on metadata."""
    reasons: list[str] = []
    if cand.is_live:
        reasons.append("live stream")
    if cand.duration is not None:
        if cand.duration < criteria.min_duration:
            reasons.append(f"too short ({cand.duration:.0f}s < {criteria.min_duration:.0f}s)")
        elif cand.duration > criteria.max_duration:
            reasons.append(f"too long ({cand.duration:.0f}s > {criteria.max_duration:.0f}s)")
    if _COMPILATION_OR_NOISE_RE.search(cand.title):
        reasons.append("compilation/loop/non-song title")
    if _PIANO_NEGATIVE_RE.search(cand.title):
        reasons.append("excluded style (tutorial/vocal/remix/etc)")
    combined_meta = f"{cand.title} {cand.channel}"
    if not _PIANO_POSITIVE_RE.search(combined_meta):
        reasons.append("no piano signal in title/channel")
    if target_song and not titles_match_song(target_song, cand.title):
        reasons.append("title does not match target song")
    if require_views and cand.view_count is not None and cand.view_count < criteria.min_piano_views:
        reasons.append(f"low piano views ({cand.view_count:,} < {criteria.min_piano_views:,})")
    return reasons


# ---------------------------------------------------------------------------
# Acoustic piano & pace analysis
# ---------------------------------------------------------------------------
def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _ramp_down(x: float, good: float, bad: float) -> float:
    """1.0 when x <= good, 0.0 when x >= bad, linear in between."""
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return (bad - x) / (bad - good)


def _ramp_up(x: float, bad: float, good: float) -> float:
    """0.0 when x <= bad, 1.0 when x >= good, linear in between."""
    if x >= good:
        return 1.0
    if x <= bad:
        return 0.0
    return (x - bad) / (good - bad)


def _band_score(x: float, lo_bad: float, lo_good: float, hi_good: float, hi_bad: float) -> float:
    """Trapezoidal 0..1 score peaking between lo_good and hi_good."""
    if x < lo_good:
        return _ramp_up(x, lo_bad, lo_good)
    if x > hi_good:
        return _ramp_down(x, hi_good, hi_bad)
    return 1.0


def _norm(x: np.ndarray, pct: float = 95.0) -> np.ndarray:
    ref = float(np.percentile(x, pct)) if x.size else 1.0
    return x / ref if ref > 1e-9 else np.zeros_like(x)


def _window_pace_stats(
    times: np.ndarray,
    duration: float,
    *,
    window_sec: float = 10.0,
    hop_sec: float = 2.5,
    fast_ioi_threshold: float = 0.25,
) -> tuple[float, float, float, float, float, float]:
    """Compute (avg_rate, p90_rate, peak_rate, median_ioi, p25_ioi, fast_frac) over a full track."""
    if len(times) < 2:
        rate = len(times) / max(duration, 1e-6)
        return rate, rate, rate, duration, duration, 0.0
    avg_rate = len(times) / max(duration, 1e-6)
    iois = np.diff(times)
    median_ioi = float(np.median(iois))
    p25_ioi = float(np.percentile(iois, 25))
    fast_frac = float(np.mean(iois < fast_ioi_threshold))
    if duration <= window_sec:
        return avg_rate, avg_rate, avg_rate, median_ioi, p25_ioi, fast_frac
    starts = np.arange(0.0, max(hop_sec, duration - window_sec + 1e-6), hop_sec)
    rates = [
        float(np.sum((times >= s) & (times < s + window_sec)) / min(window_sec, duration - s))
        for s in starts
    ]
    return avg_rate, float(np.percentile(rates, 90)), float(max(rates)), median_ioi, p25_ioi, fast_frac


def _analyze_acoustics_slice(y: np.ndarray, sr: int, min_gap: float) -> dict[str, Any]:
    """Run HPSS + spectral piano-purity & choreography duo_gain check on one slice."""
    duration = len(y) / sr
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    H, P = librosa.decompose.hpss(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    S2 = S**2
    total_energy = float(np.sum(S2)) + 1e-12
    h_energy = float(np.sum(H**2))
    p_energy = float(np.sum(P**2))
    harmonic_ratio = h_energy / max(h_energy + p_energy, 1e-12)

    flatness = float(np.median(librosa.feature.spectral_flatness(S=S)))
    centroid = float(np.median(librosa.feature.spectral_centroid(S=S, sr=sr)))
    rolloff = float(np.median(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)))
    high_freq_ratio = float(np.sum(S2[freqs > 4000.0]) / total_energy)
    piano_band_ratio = float(np.sum(S2[(freqs >= 100.0) & (freqs <= 3500.0)]) / total_energy)

    def mel_db(mag: np.ndarray) -> np.ndarray:
        return librosa.power_to_db(
            librosa.feature.melspectrogram(S=mag**2, sr=sr, n_mels=64), ref=np.max
        )

    mel_full, mel_h, mel_p = mel_db(S), mel_db(H), mel_db(P)
    env_full = librosa.onset.onset_strength(S=mel_full, sr=sr, hop_length=HOP)
    env_h = librosa.onset.onset_strength(S=mel_h, sr=sr, hop_length=HOP)
    env_p = librosa.onset.onset_strength(S=mel_p, sr=sr, hop_length=HOP)

    wait = int(0.06 * sr / HOP)
    frames = librosa.onset.onset_detect(
        onset_envelope=env_full, sr=sr, hop_length=HOP, backtrack=False, delta=0.07, wait=wait
    )
    nf, nh, np_ = _norm(env_full), _norm(env_h), _norm(env_p)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)

    lo, hi = np.searchsorted(freqs, 130.0), np.searchsorted(freqs, 2000.0)
    notes: list[dict[str, Any]] = []
    for f, t in zip(frames, times):
        seg = H[lo:hi, f + 3 : f + 12]
        pitch = None
        if seg.size and seg.max() > 1e-6:
            hz = freqs[lo + int(np.argmax(seg.mean(axis=1)))]
            pitch = int(round(librosa.hz_to_midi(hz)))
        h_val, p_val = float(nh[f]), float(np_[f])
        notes.append(
            {
                "t": round(float(t), 4),
                "strength": round(float(np.clip(nf[f], 0, 1)), 3),
                "melodic": round(h_val / (h_val + p_val + 1e-9), 3),
                "pitch": pitch,
            }
        )

    mini_analysis = {"duration": duration, "notes": notes}
    _, duo_gain = split_notes(mini_analysis, min_gap)
    cands = [n for n in notes if n["strength"] >= 0.1 and 0.25 <= n["t"] <= duration - 0.2]
    cands.sort(key=lambda n: -(n["strength"] * (0.7 + 0.6 * n["melodic"])))
    picked: list[dict[str, Any]] = []
    for n in cands:
        if all(abs(n["t"] - c["t"]) >= min_gap for c in picked):
            picked.append(n)

    return {
        "picked_per_sec": len(picked) / max(duration, 1e-6),
        "duo_gain": float(duo_gain),
        "harmonic_ratio": harmonic_ratio,
        "spectral_flatness": flatness,
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "high_freq_ratio": high_freq_ratio,
        "piano_band_ratio": piano_band_ratio,
    }


def compute_piano_score(
    *,
    harmonic_ratio: float,
    spectral_flatness: float,
    spectral_rolloff: float,
    high_freq_ratio: float,
    piano_band_ratio: float,
) -> float:
    """Estimate acoustic piano purity on a 0..1 scale.

    Calibrated against solo piano covers vs full-production pop/vocal/drum mixes:
    solo piano exhibits very low spectral flatness (< 5e-4), high harmonic ratio
    (> 0.90), concentrated energy in 100-3500 Hz (> 0.78), warm rolloff (< 2300 Hz),
    and minimal >4 kHz cymbal/sibilance energy (< 0.005).
    """
    s_flat = _ramp_down(spectral_flatness, good=0.0004, bad=0.0030)
    s_band = _ramp_up(piano_band_ratio, bad=0.45, good=0.78)
    s_high = _ramp_down(high_freq_ratio, good=0.004, bad=0.022)
    s_roll = _ramp_down(spectral_rolloff, good=2300.0, bad=3800.0)
    s_harm = _ramp_up(harmonic_ratio, bad=0.68, good=0.90)
    score = 0.28 * s_flat + 0.25 * s_band + 0.20 * s_high + 0.15 * s_roll + 0.12 * s_harm
    return round(_clamp01(score), 4)


def compute_pace_score(
    *,
    onsets_per_sec: float,
    p90_onsets_per_sec: float,
    median_ioi: float,
    fast_frac: float,
    sensitive_onsets_per_sec: float,
    sensitive_p90_onsets_per_sec: float,
    sensitive_median_ioi: float,
    duo_gain: float,
) -> float:
    """Score how comfortable and readable the rhythm pace is for Music Ball Bounce (0..1).

    Sweet spot: ~0.9-1.7 onsets/sec (<= ~2.1 sensitive onsets/sec), median note
    spacing 0.45-0.95s, low <0.25s burstiness, and clean single-ball tracking.
    """
    s_density = _band_score(onsets_per_sec, lo_bad=0.35, lo_good=0.85, hi_good=1.70, hi_bad=2.60)
    s_p90 = _ramp_down(p90_onsets_per_sec, good=2.0, bad=3.1)
    s_ioi = _band_score(median_ioi, lo_bad=0.30, lo_good=0.45, hi_good=0.95, hi_bad=1.80)
    s_fast = _ramp_down(fast_frac, good=0.06, bad=0.28)
    s_sen_avg = _ramp_down(sensitive_onsets_per_sec, good=1.8, bad=3.0)
    s_sen_p90 = _ramp_down(sensitive_p90_onsets_per_sec, good=2.3, bad=3.7)
    s_sen_ioi = _ramp_up(sensitive_median_ioi, bad=0.24, good=0.44)
    s_duo = _ramp_down(duo_gain, good=1.05, bad=1.28)
    score = (
        0.18 * s_density
        + 0.14 * s_p90
        + 0.16 * s_ioi
        + 0.10 * s_fast
        + 0.14 * s_sen_avg
        + 0.12 * s_sen_p90
        + 0.08 * s_sen_ioi
        + 0.08 * s_duo
    )
    return round(_clamp01(score), 4)


def assess_audio_file(
    audio_path: Path | str,
    *,
    min_gap: float = 0.17,
    fast_ioi_threshold: float = 0.25,
    slice_seconds: float = 15.0,
) -> AudioAssessment:
    """Assess an audio file's full-track pace (standard + sensitive) and piano acoustics.

    Pace is measured across the ENTIRE trimmed track using sliding 10-second windows
    so busy final choruses and pedal-blurred arpeggios are caught. Piano acoustic
    purity and choreography duo_gain are measured across 3 representative slices
    (at ~20%, ~50%, and ~75% of the track).
    """
    path = Path(audio_path)
    y_raw, sr = librosa.load(str(path), sr=SR, mono=True)
    if len(y_raw) / sr < 3.0:
        raise ValueError(f"Audio clip too short to assess ({len(y_raw) / sr:.1f}s)")

    # Trim leading/trailing silence so long fade-outs don't artificially lower density
    y_trimmed, _ = librosa.effects.trim(y_raw, top_db=35)
    y = y_trimmed if len(y_trimmed) / sr >= 8.0 else y_raw
    total_dur = len(y) / sr
    rms = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
    rms_db = float(20.0 * math.log10(max(rms, 1e-9)))

    # 1. Full-track pace scan (fast: STFT + mel onset envelope, no full-track HPSS)
    S_full = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    mel_full = librosa.power_to_db(
        librosa.feature.melspectrogram(S=S_full**2, sr=sr, n_mels=64), ref=np.max
    )
    env_full = librosa.onset.onset_strength(S=mel_full, sr=sr, hop_length=HOP)
    nf = _norm(env_full)
    wait = int(0.06 * sr / HOP)

    # Standard onsets (matches pipeline/analyze.py & choreograph._pick)
    f_std = librosa.onset.onset_detect(
        onset_envelope=env_full, sr=sr, hop_length=HOP, backtrack=False, delta=0.07, wait=wait
    )
    t_std = librosa.frames_to_time(f_std[nf[f_std] >= 0.10], sr=sr, hop_length=HOP)
    onsets_per_sec, p90_onsets, peak_onsets, median_ioi, p25_ioi, fast_frac = _window_pace_stats(
        t_std, total_dur, fast_ioi_threshold=fast_ioi_threshold
    )

    # Sensitive onsets (catches pedal-blurred 8th/16th-note arpeggios & soft fills)
    f_sen = librosa.onset.onset_detect(
        onset_envelope=env_full, sr=sr, hop_length=HOP, backtrack=False, delta=0.04, wait=wait
    )
    t_sen = librosa.frames_to_time(f_sen[nf[f_sen] >= 0.07], sr=sr, hop_length=HOP)
    sen_avg, sen_p90, _sen_peak, sen_med_ioi, _sen_p25, sen_fast_frac = _window_pace_stats(
        t_sen, total_dur, fast_ioi_threshold=fast_ioi_threshold
    )

    tempo_val, _ = librosa.beat.beat_track(onset_envelope=env_full, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo_val)[0])

    # 2. Piano acoustics + choreography duo_gain across 3 slices (verse, mid, climax)
    slices: list[np.ndarray] = []
    if total_dur <= slice_seconds * 3.2:
        slices.append(y)
    else:
        win_len = int(slice_seconds * sr)
        for frac in (0.20, 0.50, 0.75):
            start = int(max(0, min(len(y) - win_len, total_dur * frac * sr)))
            slices.append(y[start : start + win_len])

    slice_results = [_analyze_acoustics_slice(s, sr, min_gap) for s in slices]
    picked_per_sec = float(np.mean([r["picked_per_sec"] for r in slice_results]))
    duo_gain = max(r["duo_gain"] for r in slice_results)
    harmonic_ratio = float(np.mean([r["harmonic_ratio"] for r in slice_results]))
    spectral_flatness = float(np.mean([r["spectral_flatness"] for r in slice_results]))
    spectral_centroid = float(np.mean([r["spectral_centroid"] for r in slice_results]))
    spectral_rolloff = float(np.mean([r["spectral_rolloff"] for r in slice_results]))
    high_freq_ratio = float(np.mean([r["high_freq_ratio"] for r in slice_results]))
    piano_band_ratio = float(np.mean([r["piano_band_ratio"] for r in slice_results]))

    piano_score = compute_piano_score(
        harmonic_ratio=harmonic_ratio,
        spectral_flatness=spectral_flatness,
        spectral_rolloff=spectral_rolloff,
        high_freq_ratio=high_freq_ratio,
        piano_band_ratio=piano_band_ratio,
    )
    pace_score = compute_pace_score(
        onsets_per_sec=onsets_per_sec,
        p90_onsets_per_sec=p90_onsets,
        median_ioi=median_ioi,
        fast_frac=fast_frac,
        sensitive_onsets_per_sec=sen_avg,
        sensitive_p90_onsets_per_sec=sen_p90,
        sensitive_median_ioi=sen_med_ioi,
        duo_gain=duo_gain,
    )

    return AudioAssessment(
        sample_duration=round(total_dur, 2),
        windows_analyzed=len(slice_results),
        total_onsets=int(len(t_std)),
        onsets_per_sec=round(onsets_per_sec, 3),
        p90_onsets_per_sec=round(p90_onsets, 3),
        peak_onsets_per_sec=round(peak_onsets, 3),
        picked_per_sec=round(picked_per_sec, 3),
        median_ioi=round(median_ioi, 4),
        p25_ioi=round(p25_ioi, 4),
        fast_frac=round(fast_frac, 4),
        sensitive_onsets_per_sec=round(sen_avg, 3),
        sensitive_p90_onsets_per_sec=round(sen_p90, 3),
        sensitive_median_ioi=round(sen_med_ioi, 4),
        sensitive_fast_frac=round(sen_fast_frac, 4),
        duo_gain=round(duo_gain, 3),
        tempo=round(tempo, 2),
        harmonic_ratio=round(harmonic_ratio, 4),
        spectral_flatness=round(spectral_flatness, 6),
        spectral_centroid=round(spectral_centroid, 1),
        spectral_rolloff=round(spectral_rolloff, 1),
        high_freq_ratio=round(high_freq_ratio, 5),
        piano_band_ratio=round(piano_band_ratio, 4),
        rms_db=round(rms_db, 2),
        piano_score=piano_score,
        pace_score=pace_score,
    )


def evaluate_audio_criteria(assessment: AudioAssessment, criteria: DiscoveryCriteria) -> list[str]:
    """Return all failed criteria reasons for an analyzed audio sample."""
    reasons: list[str] = []
    if assessment.rms_db < -42.0 or assessment.total_onsets < 6:
        reasons.append("audio too quiet or sparse")
    if assessment.onsets_per_sec < criteria.min_onsets_per_sec:
        reasons.append(
            f"too sparse ({assessment.onsets_per_sec:.2f}/s < {criteria.min_onsets_per_sec:.2f}/s)"
        )
    if assessment.onsets_per_sec > criteria.max_onsets_per_sec:
        reasons.append(
            f"pace too fast ({assessment.onsets_per_sec:.2f}/s > {criteria.max_onsets_per_sec:.2f}/s)"
        )
    if assessment.p90_onsets_per_sec > criteria.max_p90_onsets_per_sec:
        reasons.append(
            f"busy sections too fast (p90 {assessment.p90_onsets_per_sec:.2f}/s > {criteria.max_p90_onsets_per_sec:.2f}/s)"
        )
    if assessment.peak_onsets_per_sec > criteria.max_peak_onsets_per_sec:
        reasons.append(
            f"peak section too fast ({assessment.peak_onsets_per_sec:.2f}/s > {criteria.max_peak_onsets_per_sec:.2f}/s)"
        )
    if assessment.median_ioi < criteria.min_median_ioi:
        reasons.append(
            f"notes too close together (median IOI {assessment.median_ioi:.3f}s < {criteria.min_median_ioi:.3f}s)"
        )
    if assessment.fast_frac > criteria.max_fast_frac:
        reasons.append(
            f"too many rapid notes ({assessment.fast_frac * 100:.0f}% < {criteria.fast_ioi_threshold:.2f}s > {criteria.max_fast_frac * 100:.0f}%)"
        )
    if assessment.sensitive_onsets_per_sec > criteria.max_sensitive_onsets_per_sec:
        reasons.append(
            f"dense arpeggios/notes ({assessment.sensitive_onsets_per_sec:.2f}/s > {criteria.max_sensitive_onsets_per_sec:.2f}/s)"
        )
    if assessment.sensitive_p90_onsets_per_sec > criteria.max_sensitive_p90_onsets_per_sec:
        reasons.append(
            f"dense chorus/climax (sensitive p90 {assessment.sensitive_p90_onsets_per_sec:.2f}/s > {criteria.max_sensitive_p90_onsets_per_sec:.2f}/s)"
        )
    if assessment.sensitive_median_ioi < criteria.min_sensitive_median_ioi:
        reasons.append(
            f"rapid inner notes (sensitive median IOI {assessment.sensitive_median_ioi:.3f}s < {criteria.min_sensitive_median_ioi:.3f}s)"
        )
    if assessment.duo_gain > criteria.max_duo_gain:
        reasons.append(
            f"requires 2 balls (duo_gain {assessment.duo_gain:.2f} > {criteria.max_duo_gain:.2f})"
        )
    if assessment.piano_score < criteria.min_piano_score:
        reasons.append(
            f"low piano purity ({assessment.piano_score:.2f} < {criteria.min_piano_score:.2f})"
        )
    return reasons


def compute_overall_score(
    assessment: AudioAssessment,
    *,
    piano_views: int | None,
    original_views: int | None = None,
) -> float:
    """Compute a 0..100 ranking score combining pace comfort, piano purity, and popularity."""
    # Popularity score on log10 scale
    if original_views and original_views > 0:
        # 10M -> 0.55, 100M -> 0.78, 1B+ -> 1.0
        orig_pop = _ramp_up(math.log10(original_views), bad=5.5, good=9.0)
        cover_pop = _ramp_up(math.log10(max(piano_views or 1, 1)), bad=3.5, good=7.0)
        pop_score = 0.70 * orig_pop + 0.30 * cover_pop
    else:
        # Direct piano mode: 50K -> 0.30, 1M -> 0.65, 20M+ -> 1.0
        pop_score = _ramp_up(math.log10(max(piano_views or 1, 1)), bad=3.5, good=7.3)

    total = 100.0 * (0.40 * assessment.pace_score + 0.35 * assessment.piano_score + 0.25 * pop_score)
    return round(max(0.0, min(100.0, total)), 1)


# ---------------------------------------------------------------------------
# YouTube search, metadata enrichment & preview download
# ---------------------------------------------------------------------------
def _base_ydl_opts() -> dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ignoreerrors": True,
        "js_runtimes": {"deno": {}, "node": {}},
    }


def _entry_to_candidate(entry: dict[str, Any], source_query: str = "") -> VideoCandidate | None:
    vid = entry.get("id")
    if not vid or not isinstance(vid, str):
        return None
    url = entry.get("webpage_url") or entry.get("url") or f"https://www.youtube.com/watch?v={vid}"
    if not url.startswith("http"):
        url = f"https://www.youtube.com/watch?v={vid}"
    duration = entry.get("duration")
    views = entry.get("view_count")
    likes = entry.get("like_count")
    return VideoCandidate(
        video_id=vid,
        url=url,
        title=str(entry.get("title") or vid),
        channel=str(entry.get("channel") or entry.get("uploader") or ""),
        duration=float(duration) if duration is not None else None,
        view_count=int(views) if views is not None else None,
        like_count=int(likes) if likes is not None else None,
        upload_date=str(entry.get("upload_date")) if entry.get("upload_date") else None,
        is_live=bool(entry.get("is_live") or entry.get("live_status") == "is_live"),
        source_query=source_query,
    )


class YouTubeClient:
    """Thin wrapper around yt-dlp for searching, fetching metadata, and audio previews."""

    def search(self, query: str, limit: int = 12) -> list[VideoCandidate]:
        import yt_dlp

        opts = {**_base_ydl_opts(), "extract_flat": True, "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
        if not info:
            return []
        out: list[VideoCandidate] = []
        for entry in info.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            cand = _entry_to_candidate(entry, source_query=query)
            if cand is not None:
                out.append(cand)
        return out

    def fetch_metadata(self, url_or_id: str, source_query: str = "") -> VideoCandidate | None:
        import yt_dlp

        url = url_or_id if url_or_id.startswith("http") else f"https://www.youtube.com/watch?v={url_or_id}"
        opts = {**_base_ydl_opts(), "skip_download": True, "noplaylist": True}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception:
            return None
        if not isinstance(info, dict):
            return None
        return _entry_to_candidate(info, source_query=source_query)

    def enrich_candidates(
        self, candidates: list[VideoCandidate], max_workers: int = 4
    ) -> list[VideoCandidate]:
        """Fill in missing view_count / duration / live status in parallel."""
        needs = [i for i, c in enumerate(candidates) if c.view_count is None or c.duration is None]
        if not needs:
            return list(candidates)
        enriched = list(candidates)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(needs))) as pool:
            futures = {
                pool.submit(self.fetch_metadata, candidates[i].url, candidates[i].source_query): i
                for i in needs
            }
            for fut, idx in futures.items():
                meta = fut.result()
                if meta is not None:
                    enriched[idx] = meta
        return enriched

    def download_preview_audio(self, url: str, dest_dir: Path) -> Path:
        """Download lowest-bandwidth audio track for fast local acoustic analysis."""
        import yt_dlp

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required on PATH for audio analysis")

        def _reject_live(info: dict, *, incomplete: bool) -> str | None:
            if info.get("is_live"):
                return "Live streams are not supported"
            return None

        opts = {
            **_base_ydl_opts(),
            "format": "worstaudio/bestaudio/best",
            "noplaylist": True,
            "outtmpl": str(dest_dir / "preview.%(ext)s"),
            "match_filter": _reject_live,
            "ignoreerrors": False,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)

        files = sorted(dest_dir.glob("preview.*"))
        if not files:
            raise RuntimeError("yt-dlp did not produce an audio preview file")
        src = files[0]
        wav = dest_dir / "preview.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), str(wav)],
            check=True,
            timeout=45,
        )
        return wav


# ---------------------------------------------------------------------------
# Persistent catalog
# ---------------------------------------------------------------------------
class DiscoveryCatalog:
    """JSON catalog tracking all evaluated tracks and originals across runs."""

    def __init__(self, path: Path = DEFAULT_CATALOG_PATH) -> None:
        self.path = path
        self.tracks: dict[str, EvaluatedTrack] = {}
        self.seen_originals: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for vid, item in (raw.get("tracks") or {}).items():
            try:
                self.tracks[vid] = EvaluatedTrack.from_dict(item)
            except (KeyError, TypeError, ValueError):
                continue
        self.seen_originals = dict(raw.get("seen_originals") or {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": _now_iso(),
            "tracks": {vid: t.to_dict() for vid, t in self.tracks.items()},
            "seen_originals": self.seen_originals,
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def add_track(self, track: EvaluatedTrack) -> None:
        existing = self.tracks.get(track.video_id)
        if existing and existing.used:
            track.used = True
        self.tracks[track.video_id] = track

    def record_original(self, cand: VideoCandidate, status: str, matched_video_id: str | None = None) -> None:
        self.seen_originals[cand.video_id] = {
            "video_id": cand.video_id,
            "title": cand.title,
            "url": cand.url,
            "view_count": cand.view_count,
            "status": status,
            "matched_video_id": matched_video_id,
            "checked_at": _now_iso(),
        }

    def mark_used(self, video_id: str) -> None:
        if video_id in self.tracks:
            self.tracks[video_id].used = True
            self.save()

    def accepted_song_keys(self) -> set[str]:
        return {t.song_key for t in self.tracks.values() if t.accepted and t.song_key}

    def list_tracks(
        self,
        *,
        include_rejected: bool = False,
        unused_only: bool = False,
    ) -> list[EvaluatedTrack]:
        items = list(self.tracks.values())
        if not include_rejected:
            items = [t for t in items if t.accepted]
        if unused_only:
            items = [t for t in items if not t.used]
        items.sort(key=lambda t: (-t.score, -(t.view_count or 0)))
        return items


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def format_views(n: int | None) -> str:
    if n is None:
        return "n/a"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Discovery orchestrator
# ---------------------------------------------------------------------------
def _evaluate_piano_candidate(
    cand: VideoCandidate,
    *,
    mode: str,
    criteria: DiscoveryCriteria,
    client: YouTubeClient,
    original: VideoCandidate | None = None,
    song_key_override: str | None = None,
    assessor: Callable[[Path], AudioAssessment] | None = None,
) -> EvaluatedTrack:
    orig_dict = (
        {
            "video_id": original.video_id,
            "title": original.title,
            "url": original.url,
            "view_count": original.view_count,
            "channel": original.channel,
        }
        if original is not None
        else None
    )
    skey = song_key_override or song_identity_key(original.title if original else cand.title)

    with tempfile.TemporaryDirectory(prefix="mbb-discover-") as tmp:
        wav_path = client.download_preview_audio(cand.url, Path(tmp))
        assessment = (
            assessor(wav_path)
            if assessor is not None
            else assess_audio_file(
                wav_path,
                min_gap=criteria.min_gap,
                fast_ioi_threshold=criteria.fast_ioi_threshold,
            )
        )

    reasons = evaluate_audio_criteria(assessment, criteria)
    score = compute_overall_score(
        assessment,
        piano_views=cand.view_count,
        original_views=original.view_count if original else None,
    )
    return EvaluatedTrack(
        video_id=cand.video_id,
        url=cand.url,
        title=cand.title,
        channel=cand.channel,
        duration=cand.duration,
        view_count=cand.view_count,
        mode=mode,
        song_key=skey,
        source_query=cand.source_query,
        accepted=len(reasons) == 0,
        score=score,
        rejection_reasons=reasons,
        evaluated_at=_now_iso(),
        original=orig_dict,
        metrics=asdict(assessment),
    )


def discover_tracks(
    *,
    count: int = 5,
    mode: str = "popular-to-piano",
    queries: list[str] | None = None,
    songs: list[str] | None = None,
    criteria: DiscoveryCriteria | None = None,
    catalog: DiscoveryCatalog | None = None,
    client: YouTubeClient | None = None,
    max_checks: int | None = None,
    search_limit: int = 12,
    piano_search_limit: int = 6,
    recheck: bool = False,
    allow_duplicate_songs: bool = False,
    seed: int | None = None,
    log: Callable[[str], None] | None = print,
    assessor: Callable[[Path], AudioAssessment] | None = None,
) -> list[EvaluatedTrack]:
    """Systematically search, filter, and acoustically verify slow piano tracks."""
    criteria = criteria or DiscoveryCriteria()
    catalog = catalog or DiscoveryCatalog()
    client = client or YouTubeClient()
    emit = log or (lambda _msg: None)
    budget = max_checks if max_checks is not None else max(count * 8, 25)

    accepted_now: list[EvaluatedTrack] = []
    seen_song_keys = set() if allow_duplicate_songs else set(catalog.accepted_song_keys())
    checks_done = 0

    def _can_continue() -> bool:
        return len(accepted_now) < count and checks_done < budget

    def _piano_candidate_rank(c: VideoCandidate) -> tuple[int, int]:
        # Prefer explicitly slow/easy/lullaby/soft arrangements before dense virtuoso covers
        has_slow_hint = 1 if _SLOW_PIANO_HINT_RE.search(f"{c.title} {c.channel}") else 0
        return (-has_slow_hint, -(c.view_count or 0))

    def _find_piano_for_song(
        song_label: str,
        *,
        original: VideoCandidate | None,
        run_mode: str,
        per_song_checks: int = 4,
    ) -> EvaluatedTrack | None:
        nonlocal checks_done
        cleaned = clean_song_title(song_label)
        if not cleaned:
            return None
        skey = song_identity_key(cleaned)
        if not allow_duplicate_songs and skey in seen_song_keys:
            return None

        search_queries = [
            f"{cleaned} slow piano",
            f"{cleaned} easy piano",
            f"{cleaned} piano lullaby",
            f"{cleaned} solo piano",
        ]
        raw_cands: list[VideoCandidate] = []
        seen_ids: set[str] = set()
        for sq in search_queries:
            for c in client.search(sq, limit=piano_search_limit):
                if c.video_id in seen_ids:
                    continue
                if original and c.video_id == original.video_id:
                    continue
                seen_ids.add(c.video_id)
                raw_cands.append(c)
            if len(raw_cands) >= piano_search_limit * 2:
                break

        # Fast title + duration pre-screen before metadata enrichment
        pre_screened = [
            c
            for c in raw_cands
            if not screen_piano_candidate(c, criteria, target_song=cleaned, require_views=False)
            and (recheck or c.video_id not in catalog.tracks)
        ]
        if not pre_screened:
            return None

        # Prioritize slow/easy/lullaby titles before enriching so we spend network calls wisely
        pre_screened.sort(key=lambda c: 0 if _SLOW_PIANO_HINT_RE.search(f"{c.title} {c.channel}") else 1)
        enriched = client.enrich_candidates(pre_screened[: max(per_song_checks * 2, 8)])
        passing: list[VideoCandidate] = []
        for c in enriched:
            if not screen_piano_candidate(c, criteria, target_song=cleaned, require_views=True):
                passing.append(c)

        passing.sort(key=_piano_candidate_rank)

        tried_for_song = 0
        for cand in passing:
            if not _can_continue() or tried_for_song >= per_song_checks:
                break
            tried_for_song += 1
            checks_done += 1
            emit(
                f"  [{checks_done}/{budget}] checking audio: {cand.title} "
                f"({format_views(cand.view_count)} views)"
            )
            try:
                track = _evaluate_piano_candidate(
                    cand,
                    mode=run_mode,
                    criteria=criteria,
                    client=client,
                    original=original,
                    song_key_override=skey,
                    assessor=assessor,
                )
            except Exception as exc:
                emit(f"      skipped (audio error: {exc})")
                continue

            catalog.add_track(track)
            catalog.save()
            m = track.metrics or {}
            if track.accepted:
                seen_song_keys.add(skey)
                accepted_now.append(track)
                emit(
                    f"      ACCEPTED (score {track.score:.1f}): "
                    f"{m.get('onsets_per_sec', 0):.2f}/s (p90 {m.get('p90_onsets_per_sec', 0):.2f}/s, "
                    f"sen {m.get('sensitive_onsets_per_sec', 0):.2f}/s), "
                    f"med IOI {m.get('median_ioi', 0):.2f}s, "
                    f"duo {m.get('duo_gain', 0):.2f}x, "
                    f"piano {m.get('piano_score', 0):.2f}"
                )
                return track
            else:
                emit(f"      rejected: {'; '.join(track.rejection_reasons)}")
        return None

    # ------------------------------------------------------------------
    # Mode 0: Explicit --song targets (URL or "<artist> - <song>")
    # ------------------------------------------------------------------
    if songs:
        for item in songs:
            if not _can_continue():
                break
            original: VideoCandidate | None = None
            song_label = item.strip()
            if is_youtube_url(song_label):
                emit(f"Resolving seed YouTube URL: {song_label}")
                meta = client.fetch_metadata(song_label, source_query="seed-url")
                if meta is None:
                    emit(f"  could not fetch metadata for {song_label}")
                    continue
                original = meta
                song_label = meta.title
                emit(f"Target song: {clean_song_title(song_label)} ({format_views(meta.view_count)} views)")
            else:
                emit(f"Target song: {song_label}")
                # Best-effort lookup of original song for view count metadata
                orig_hits = client.search(f"{song_label} official music video", limit=3)
                orig_pre = [
                    c for c in orig_hits if not screen_original_candidate(c, criteria, require_views=False)
                ]
                if orig_pre:
                    orig_enr = client.enrich_candidates(orig_pre[:2])
                    orig_enr.sort(key=lambda c: -(c.view_count or 0))
                    original = orig_enr[0]
            _find_piano_for_song(song_label, original=original, run_mode="song-to-piano", per_song_checks=4)
        accepted_now.sort(key=lambda t: -t.score)
        return accepted_now

    # ------------------------------------------------------------------
    # Mode 1: popular-to-piano (discover high-view originals -> piano covers)
    # ------------------------------------------------------------------
    if mode in ("popular-to-piano", "both") and _can_continue():
        pop_queries = list(queries) if queries else list(DEFAULT_POPULAR_QUERIES)
        if not queries:
            rng = random.Random(seed)
            rng.shuffle(pop_queries)

        for q in pop_queries:
            if not _can_continue():
                break
            emit(f"Searching popular songs: {q!r}")
            raw_originals = client.search(q, limit=search_limit)
            pre_originals = [
                c
                for c in raw_originals
                if not screen_original_candidate(c, criteria, require_views=False)
                and (recheck or c.video_id not in catalog.seen_originals)
            ]
            if not pre_originals:
                continue
            enriched_originals = client.enrich_candidates(pre_originals)
            popular_originals = [
                c
                for c in enriched_originals
                if not screen_original_candidate(c, criteria, require_views=True)
            ]
            popular_originals.sort(key=lambda c: -(c.view_count or 0))
            emit(
                f"  found {len(popular_originals)} popular track(s) "
                f"(>= {format_views(criteria.min_original_views)} views)"
            )

            for orig in popular_originals:
                if not _can_continue():
                    break
                cleaned = clean_song_title(orig.title)
                skey = song_identity_key(cleaned)
                if not allow_duplicate_songs and skey in seen_song_keys:
                    continue
                emit(f"Finding piano version for: {cleaned} ({format_views(orig.view_count)} views)")
                matched = _find_piano_for_song(
                    orig.title, original=orig, run_mode="popular-to-piano", per_song_checks=3
                )
                catalog.record_original(
                    orig,
                    status="matched" if matched else "no_qualifying_piano",
                    matched_video_id=matched.video_id if matched else None,
                )
                catalog.save()

    # ------------------------------------------------------------------
    # Mode 2: direct-piano (search high-view single-track piano covers directly)
    # ------------------------------------------------------------------
    if mode in ("direct-piano", "both") and _can_continue():
        piano_queries = list(queries) if queries else list(DEFAULT_DIRECT_PIANO_QUERIES)
        if not queries:
            rng = random.Random(seed)
            rng.shuffle(piano_queries)

        for q in piano_queries:
            if not _can_continue():
                break
            emit(f"Searching piano tracks directly: {q!r}")
            raw_cands = client.search(q, limit=search_limit)
            pre = [
                c
                for c in raw_cands
                if not screen_piano_candidate(c, criteria, require_views=False)
                and (recheck or c.video_id not in catalog.tracks)
            ]
            if not pre:
                continue
            enriched = client.enrich_candidates(pre)
            passing = [c for c in enriched if not screen_piano_candidate(c, criteria, require_views=True)]
            passing.sort(key=lambda c: -(c.view_count or 0))
            emit(f"  {len(passing)} candidate(s) passed metadata & view filters")

            for cand in passing:
                if not _can_continue():
                    break
                skey = song_identity_key(cand.title)
                if not allow_duplicate_songs and skey in seen_song_keys:
                    continue
                checks_done += 1
                emit(
                    f"  [{checks_done}/{budget}] checking audio: {cand.title} "
                    f"({format_views(cand.view_count)} views)"
                )
                try:
                    track = _evaluate_piano_candidate(
                        cand,
                        mode="direct-piano",
                        criteria=criteria,
                        client=client,
                        song_key_override=skey,
                        assessor=assessor,
                    )
                except Exception as exc:
                    emit(f"      skipped (audio error: {exc})")
                    continue

                catalog.add_track(track)
                catalog.save()
                m = track.metrics or {}
                if track.accepted:
                    seen_song_keys.add(skey)
                    accepted_now.append(track)
                    emit(
                        f"      ACCEPTED (score {track.score:.1f}): "
                        f"{m.get('onsets_per_sec', 0):.2f} onsets/s, "
                        f"med IOI {m.get('median_ioi', 0):.2f}s, "
                        f"duo {m.get('duo_gain', 0):.2f}x, "
                        f"piano {m.get('piano_score', 0):.2f}"
                    )
                else:
                    emit(f"      rejected: {'; '.join(track.rejection_reasons)}")

    accepted_now.sort(key=lambda t: -t.score)
    return accepted_now


# ---------------------------------------------------------------------------
# Presentation & CLI
# ---------------------------------------------------------------------------
def format_results_table(tracks: list[EvaluatedTrack]) -> str:
    if not tracks:
        return "No qualifying tracks found."
    lines = [
        f"Found {len(tracks)} qualifying slow piano track(s):",
        "",
    ]
    for idx, t in enumerate(tracks, start=1):
        m = t.metrics or {}
        orig = t.original
        orig_str = (
            f" | orig: {clean_song_title(orig['title'])} ({format_views(orig.get('view_count'))})"
            if orig
            else ""
        )
        dur_str = f"{int(t.duration // 60)}:{int(t.duration % 60):02d}" if t.duration else "?:??"
        used_tag = " [used]" if t.used else ""
        status_tag = "" if t.accepted else f" [REJECTED: {'; '.join(t.rejection_reasons)}]"
        lines.append(f"{idx:>2}. [{t.score:5.1f}] {t.title}{used_tag}{status_tag}")
        lines.append(
            f"    {t.url}  ({dur_str}, piano views: {format_views(t.view_count)}{orig_str})"
        )
        if m:
            sen_str = (
                f", sen {m['sensitive_onsets_per_sec']:.2f}/s (p90 {m.get('sensitive_p90_onsets_per_sec', 0):.2f}/s)"
                if "sensitive_onsets_per_sec" in m
                else ""
            )
            p90_str = (
                f"p90 {m['p90_onsets_per_sec']:.2f}/s"
                if "p90_onsets_per_sec" in m
                else f"peak {m.get('peak_onsets_per_sec', 0):.2f}/s"
            )
            lines.append(
                f"    pace: {m.get('onsets_per_sec', 0):.2f}/s ({p90_str}{sen_str}), "
                f"med IOI {m.get('median_ioi', 0):.2f}s, "
                f"fast {m.get('fast_frac', 0) * 100:.0f}%, "
                f"duo {m.get('duo_gain', 0):.2f}x | "
                f"piano purity: {m.get('piano_score', 0):.2f}"
            )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--count", type=int, default=5, help="number of qualifying tracks to find (default: 5)")
    ap.add_argument(
        "--mode",
        choices=["popular-to-piano", "direct-piano", "both"],
        default="popular-to-piano",
        help="discovery strategy (default: popular-to-piano)",
    )
    ap.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="custom YouTube search query (repeatable; defaults to curated popular hits)",
    )
    ap.add_argument(
        "--song",
        action="append",
        dest="songs",
        help="specific song title or YouTube URL to find a slow piano version for (repeatable)",
    )
    ap.add_argument(
        "--min-original-views",
        type=int,
        default=50_000_000,
        help="minimum views for original popular song (default: 50000000)",
    )
    ap.add_argument(
        "--min-piano-views",
        type=int,
        default=20_000,
        help="minimum views for piano cover video (default: 20000)",
    )
    ap.add_argument("--min-duration", type=float, default=90.0, help="minimum track length in seconds (default: 90)")
    ap.add_argument("--max-duration", type=float, default=420.0, help="maximum track length in seconds (default: 420)")
    ap.add_argument(
        "--max-onsets-per-sec",
        type=float,
        default=2.0,
        help="maximum full-track average onsets/second (default: 2.0)",
    )
    ap.add_argument(
        "--max-p90-onsets-per-sec",
        type=float,
        default=2.5,
        help="maximum 90th-percentile 10s-window onsets/second across the full song (default: 2.5)",
    )
    ap.add_argument(
        "--max-peak-onsets-per-sec",
        type=float,
        default=4.0,
        help="maximum single 10s-window peak onsets/second (default: 4.0)",
    )
    ap.add_argument(
        "--min-onsets-per-sec",
        type=float,
        default=0.5,
        help="minimum average onsets/second to avoid near-silence (default: 0.5)",
    )
    ap.add_argument(
        "--min-median-ioi",
        type=float,
        default=0.40,
        help="minimum median interval between notes in seconds (default: 0.40)",
    )
    ap.add_argument(
        "--max-fast-frac",
        type=float,
        default=0.20,
        help="maximum fraction of note intervals shorter than 0.25s (default: 0.20)",
    )
    ap.add_argument(
        "--max-sensitive-onsets-per-sec",
        type=float,
        default=2.5,
        help="maximum sensitive-detector onsets/sec to catch pedal-blurred arpeggios (default: 2.5)",
    )
    ap.add_argument(
        "--max-sensitive-p90-onsets-per-sec",
        type=float,
        default=3.2,
        help="maximum 90th-percentile sensitive onsets/sec in busy sections (default: 3.2)",
    )
    ap.add_argument(
        "--min-sensitive-median-ioi",
        type=float,
        default=0.32,
        help="minimum sensitive-detector median interval between notes in seconds (default: 0.32)",
    )
    ap.add_argument(
        "--max-duo-gain",
        type=float,
        default=1.20,
        help="maximum two-ball gain (below 1.3 keeps clean 1-ball pacing; default: 1.20)",
    )
    ap.add_argument(
        "--min-piano-score",
        type=float,
        default=0.65,
        help="minimum acoustic piano purity score 0..1 (default: 0.65)",
    )
    ap.add_argument("--min-gap", type=float, default=0.17, help="choreography min-gap in seconds (default: 0.17)")
    ap.add_argument("--max-checks", type=int, help="maximum audio previews to download & inspect")
    ap.add_argument("--search-limit", type=int, default=12, help="YouTube results fetched per search query")
    ap.add_argument("--seed", type=int, help="random seed for query order")
    ap.add_argument("--recheck", action="store_true", help="re-evaluate videos already in the catalog")
    ap.add_argument(
        "--allow-duplicate-songs",
        action="store_true",
        help="allow multiple piano covers of the same underlying song",
    )
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH, help="path to discovery catalog JSON")
    ap.add_argument("--list", action="store_true", help="list tracks already saved in the catalog (no network)")
    ap.add_argument("--include-rejected", action="store_true", help="with --list, also show rejected tracks")
    ap.add_argument("--unused-only", action="store_true", help="with --list, only show tracks not yet rendered")
    ap.add_argument("--download", action="store_true", help="download full MP3s for accepted tracks into work/youtube/")
    ap.add_argument("--render", action="store_true", help="run pipeline/mbb.py on each accepted track")
    ap.add_argument("--no-render", action="store_true", help="when combined with --render, pass --no-render to mbb.py")
    ap.add_argument("--full", action="store_true", help="when combined with --render, pass --full to mbb.py")
    ap.add_argument("--synthesize", choices=["auto", "on", "off"], help="when combined with --render, pass --synthesize mode to mbb.py")
    ap.add_argument("--soundfont", type=Path, help="when combined with --render, pass --soundfont path to mbb.py")
    ap.add_argument("--clip-min", type=float, help="when combined with --render, pass --clip-min to mbb.py")
    ap.add_argument("--clip-max", type=float, help="when combined with --render, pass --clip-max to mbb.py")
    ap.add_argument("--export-urls", type=Path, help="write accepted YouTube URLs (one per line) to this file")
    ap.add_argument("--json", action="store_true", dest="as_json", help="output JSON instead of a formatted table")
    ap.add_argument("--quiet", action="store_true", help="suppress progress logs")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    catalog = DiscoveryCatalog(args.catalog)

    if args.list:
        tracks = catalog.list_tracks(
            include_rejected=args.include_rejected,
            unused_only=args.unused_only,
        )
        if args.count and len(tracks) > args.count:
            tracks = tracks[: args.count]
    else:
        criteria = DiscoveryCriteria(
            min_original_views=args.min_original_views,
            min_piano_views=args.min_piano_views,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            min_onsets_per_sec=args.min_onsets_per_sec,
            max_onsets_per_sec=args.max_onsets_per_sec,
            max_p90_onsets_per_sec=args.max_p90_onsets_per_sec,
            max_peak_onsets_per_sec=args.max_peak_onsets_per_sec,
            min_median_ioi=args.min_median_ioi,
            max_fast_frac=args.max_fast_frac,
            max_sensitive_onsets_per_sec=args.max_sensitive_onsets_per_sec,
            max_sensitive_p90_onsets_per_sec=args.max_sensitive_p90_onsets_per_sec,
            min_sensitive_median_ioi=args.min_sensitive_median_ioi,
            max_duo_gain=args.max_duo_gain,
            min_piano_score=args.min_piano_score,
            min_gap=args.min_gap,
        )
        t0 = time.time()
        emit = (lambda _m: None) if (args.quiet or args.as_json) else (lambda m: print(m, file=sys.stderr))
        tracks = discover_tracks(
            count=args.count,
            mode=args.mode,
            queries=args.queries,
            songs=args.songs,
            criteria=criteria,
            catalog=catalog,
            max_checks=args.max_checks,
            search_limit=args.search_limit,
            recheck=args.recheck,
            allow_duplicate_songs=args.allow_duplicate_songs,
            seed=args.seed,
            log=emit,
        )
        emit(f"Discovery finished in {time.time() - t0:.1f}s ({len(tracks)} accepted). Catalog: {catalog.path}")

    if args.export_urls:
        args.export_urls.parent.mkdir(parents=True, exist_ok=True)
        args.export_urls.write_text(
            "\n".join(t.url for t in tracks if t.accepted) + ("\n" if tracks else ""),
            encoding="utf-8",
        )

    if args.download:
        for t in tracks:
            if not t.accepted:
                continue
            dl = download_youtube_mp3(t.url, ROOT / "work" / "youtube")
            if not args.as_json and not args.quiet:
                state = "cached" if dl.cached else "downloaded"
                print(f"[{state}] {dl.title} -> {dl.mp3_path}", file=sys.stderr)

    if args.as_json:
        print(json.dumps([t.to_dict() for t in tracks], indent=2))
    else:
        print(format_results_table(tracks))
        accepted = [t for t in tracks if t.accepted]
        if accepted and not args.render:
            print()
            print("Next step — generate a 3D video from any track above:")
            print(f'  uv run --project pipeline python pipeline/mbb.py "{accepted[0].url}"')

    if args.render:
        for t in tracks:
            if not t.accepted:
                continue
            cmd = [sys.executable, str(ROOT / "pipeline" / "mbb.py"), t.url]
            if args.no_render:
                cmd.append("--no-render")
            if args.full:
                cmd.append("--full")
            if args.synthesize:
                cmd.extend(["--synthesize", args.synthesize])
            if args.soundfont:
                cmd.extend(["--soundfont", str(args.soundfont)])
            if args.clip_min is not None:
                cmd.extend(["--clip-min", str(args.clip_min)])
            if args.clip_max is not None:
                cmd.extend(["--clip-max", str(args.clip_max)])
            subprocess.run(cmd, cwd=ROOT, check=True)
            catalog.mark_used(t.video_id)


if __name__ == "__main__":
    main()
