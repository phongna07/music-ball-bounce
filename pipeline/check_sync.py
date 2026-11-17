"""Measure audio-visual sync of a rendered video.

    uv run --project pipeline python pipeline/check_sync.py out/song.mp4 public/songs/song/choreo.json

Extracts the audio track from the finished mp4 (so encoder delay and muxing are
included), detects onsets in it, and reports how far each bounce lands from the
nearest onset. Beat-grid fill-ins are not tied to an onset, so the summary uses
the bounces that have one within 60 ms.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import librosa

SR = 22050
HOP = 64


def main() -> None:
    video, choreo_path = Path(sys.argv[1]), Path(sys.argv[2])
    choreo = json.loads(choreo_path.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "track.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1",
                        "-ar", str(SR), str(wav)], check=True)
        y, _ = librosa.load(str(wav), sr=SR)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, units="time",
                                        backtrack=False, delta=0.05)
    hits = np.array([p["t"] for p in choreo["plates"]])
    nearest = onsets[np.abs(onsets[None, :] - hits[:, None]).argmin(axis=1)]
    err = (hits - nearest) * 1000.0  # positive: bounce after the onset
    matched = err[np.abs(err) < 60]
    frame_ms = 1000.0 / 60
    print(f"bounces: {len(hits)}, matched to an onset within 60 ms: {len(matched)}")
    if len(matched):
        print(f"median offset {np.median(matched):+.1f} ms, mean |error| {np.abs(matched).mean():.1f} ms, "
              f"worst {np.abs(matched).max():.1f} ms (one frame = {frame_ms:.1f} ms)")
        print(f"within one frame: {np.mean(np.abs(matched) <= frame_ms) * 100:.0f}%")


if __name__ == "__main__":
    main()
