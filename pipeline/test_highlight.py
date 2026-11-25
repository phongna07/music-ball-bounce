"""Unit tests for pipeline/highlight.py."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from highlight import select_highlight, slice_audio_clip


def _make_synthetic_analysis() -> dict:
    """Build a 240s analysis dict with:
    0-10s silent intro, 10-60s soft verse, 60-105s strong chorus,
    105-150s soft bridge, 150-175s climax followed by 10s dead air (175-185s),
    185-240s soft outro.
    """
    duration = 240.0
    fps = 60
    n_frames = int(duration * fps)
    rms = np.zeros(n_frames, dtype=np.float32)
    notes: list[dict] = []

    def add_region(t0: float, t1: float, level: float, note_step: float, strength: float) -> None:
        i0, i1 = int(t0 * fps), int(t1 * fps)
        rms[i0:i1] = level
        if note_step > 0:
            t = t0 + 0.2
            while t < t1 - 0.2:
                notes.append({"t": round(t, 4), "strength": strength, "melodic": 0.85, "pitch": 64})
                t += note_step

    add_region(0.0, 10.0, 0.01, 0.0, 0.0)
    add_region(10.0, 60.0, 0.20, 1.0, 0.45)
    add_region(60.0, 105.0, 0.92, 0.5, 0.85)
    add_region(105.0, 150.0, 0.25, 0.9, 0.50)
    add_region(150.0, 175.0, 1.00, 0.5, 0.90)
    add_region(175.0, 185.0, 0.01, 0.0, 0.0)  # 10s dead-air trap
    add_region(185.0, 240.0, 0.18, 1.1, 0.40)

    downbeats = [round(t, 3) for t in np.arange(0.0, duration, 2.0)]
    sections = [
        {"t0": 0.0, "t1": 10.0, "energy": 0.01},
        {"t0": 10.0, "t1": 60.0, "energy": 0.20},
        {"t0": 60.0, "t1": 105.0, "energy": 0.92},
        {"t0": 105.0, "t1": 150.0, "energy": 0.25},
        {"t0": 150.0, "t1": 175.0, "energy": 1.00},
        {"t0": 175.0, "t1": 185.0, "energy": 0.01},
        {"t0": 185.0, "t1": 240.0, "energy": 0.18},
    ]
    return {
        "duration": duration,
        "tempo": 120.0,
        "beats": downbeats,
        "downbeats": downbeats,
        "notes": notes,
        "kicks": [],
        "sections": sections,
        "energy": {"fps": fps, "rms": rms.tolist(), "low": rms.tolist(), "high": rms.tolist()},
    }


class TestHighlightSelection(unittest.TestCase):
    def test_short_track_passthrough(self) -> None:
        analysis = {
            "duration": 28.5,
            "notes": [{"t": 1.0, "strength": 0.8, "melodic": 0.9, "pitch": 60}],
            "downbeats": [0.5, 2.5],
            "sections": [{"t0": 0.0, "t1": 28.5, "energy": 0.8}],
            "energy": {"fps": 60, "rms": [0.5] * (28 * 60)},
        }
        win = select_highlight(analysis, min_sec=35.0, max_sec=60.0, target_sec=45.0)
        self.assertFalse(win.clipped)
        self.assertEqual(win.start, 0.0)
        self.assertAlmostEqual(win.end, 28.5, places=2)
        self.assertAlmostEqual(win.duration, 28.5, places=2)

    def test_selects_chorus_avoids_intro_and_dead_air(self) -> None:
        analysis = _make_synthetic_analysis()
        win = select_highlight(analysis, min_sec=35.0, max_sec=60.0, target_sec=45.0)

        self.assertTrue(win.clipped)
        self.assertGreaterEqual(win.duration, 35.0)
        self.assertLessEqual(win.duration, 60.0)
        # Should pick the continuous high-energy 60-105s chorus rather than 0-10s intro or 175-185s dead air
        self.assertGreaterEqual(win.start, 55.0)
        self.assertLessEqual(win.end, 110.0)
        self.assertFalse(win.start < 175.0 and win.end > 180.0)

        # First strong note inside the clip should have >= 0.25s lead-in for choreograph._pick
        notes_in = [n["t"] for n in analysis["notes"] if n["t"] >= win.start and n["t"] <= win.end]
        self.assertTrue(notes_in)
        self.assertGreaterEqual(notes_in[0] - win.start, 0.25)

    def test_manual_clip_start_override(self) -> None:
        analysis = _make_synthetic_analysis()
        win = select_highlight(analysis, min_sec=35.0, max_sec=60.0, target_sec=42.0, clip_start=112.0)
        self.assertTrue(win.clipped)
        self.assertAlmostEqual(win.start, 112.0)
        self.assertAlmostEqual(win.end, 154.0)
        self.assertAlmostEqual(win.duration, 42.0)

    def test_slice_audio_clip_duration_format_and_fades(self) -> None:
        sr = 44100
        dur = 90.0
        t = np.arange(int(dur * sr), dtype=np.float32) / sr
        sig = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
        stereo = np.stack([sig, sig], axis=1)

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "full.wav"
            out = Path(tmp) / "clip.wav"
            sf.write(str(src), stereo, sr, subtype="PCM_16")

            clip_dur = slice_audio_clip(src, out, 20.0, 65.0, fade_in=0.12, fade_out=2.0)
            self.assertAlmostEqual(clip_dur, 45.0, places=3)

            info = sf.info(str(out))
            self.assertEqual(info.samplerate, 44100)
            self.assertEqual(info.channels, 2)
            self.assertEqual(info.subtype, "PCM_16")

            y, _ = sf.read(str(out), dtype="float32", always_2d=True)
            self.assertAlmostEqual(y.shape[0] / sr, 45.0, places=3)
            # Raised-cosine fade-in & fade-out keep boundaries near zero while middle is full level
            self.assertLess(float(np.max(np.abs(y[:20]))), 0.01)
            self.assertLess(float(np.max(np.abs(y[-20:]))), 0.01)
            self.assertGreater(float(np.max(np.abs(y[20 * sr : 21 * sr]))), 0.45)


if __name__ == "__main__":
    unittest.main()
