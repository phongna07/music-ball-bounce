"""Unit tests for pipeline/synthesize.py."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mido
import numpy as np
import soundfile as sf

from synthesize import (
    MidiEvent,
    continuous_bernoulli_mean,
    extract_features,
    make_frames,
    master_piano_wav,
    resolve_overlapping,
    synthesize_piano_version,
    viterbi_backward,
    write_events_to_midi,
)


class TestTranskunMath(unittest.TestCase):
    def test_make_frames_and_extract_features(self) -> None:
        sr = 44100
        x = np.sin(2.0 * np.pi * 440.0 * np.arange(sr, dtype=np.float32) / sr)
        frames = make_frames(x, hop_size=1024, window_size=4096)
        self.assertEqual(frames.ndim, 2)
        self.assertEqual(frames.shape[1], 4096)
        self.assertEqual(frames.shape[0], int(np.ceil(sr / 1024)) + 1)

        windows = np.ones((6, 4096), dtype=np.float32) * 0.5
        freq2mels = np.ones((2049, 229), dtype=np.float32) * 0.01
        feat = extract_features(x, windows, freq2mels)
        self.assertEqual(feat.shape, (1, frames.shape[0], 229, 6))
        self.assertEqual(feat.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(feat)))

    def test_continuous_bernoulli_mean_stability(self) -> None:
        logits = np.array([-20.0, -1.0, -1e-6, 0.0, 1e-6, 1.0, 20.0], dtype=np.float32)
        means = continuous_bernoulli_mean(logits)
        self.assertTrue(np.all(np.isfinite(means)))
        self.assertTrue(np.all((means > 0.0) & (means < 1.0)))
        self.assertAlmostEqual(float(means[3]), 0.5, places=5)
        self.assertTrue(np.all(np.diff(means) >= 0.0))

    def test_viterbi_backward_synthetic_and_forced(self) -> None:
        T, n_sym = 6, 90
        score = np.full((T, T, n_sym), -5.0, dtype=np.float32)
        score[3, 1, 5] = 4.0
        score[5, 5, 5] = 2.0
        score[4, 0, 10] = 6.0
        score[2, 2, 20] = 3.0

        intervals = viterbi_backward(score)
        self.assertEqual(intervals[5], [(1, 3), (5, 5)])
        self.assertEqual(intervals[10], [(0, 4)])
        self.assertEqual(intervals[20], [(2, 2)])
        self.assertEqual(intervals[0], [])

        forced = [0] * n_sym
        forced[5] = 4
        intervals_forced = viterbi_backward(score, forced_start_pos=forced)
        self.assertEqual(intervals_forced[5], [(5, 5)])
        self.assertEqual(intervals_forced[10], [(0, 4)])

    def test_resolve_overlapping(self) -> None:
        events = [
            MidiEvent(start=1.0, end=2.5, pitch=60, velocity=80),
            MidiEvent(start=2.0, end=3.0, pitch=60, velocity=70),
            MidiEvent(start=2.0, end=3.0, pitch=64, velocity=75),
            MidiEvent(start=3.0, end=3.0, pitch=67, velocity=60),
        ]
        resolved = resolve_overlapping(events)
        p60 = [e for e in resolved if e.pitch == 60]
        self.assertEqual(len(p60), 2)
        self.assertAlmostEqual(p60[0].end, 2.0)
        self.assertEqual(len([e for e in resolved if e.pitch == 64]), 1)
        self.assertEqual(len([e for e in resolved if e.pitch == 67]), 0)


class TestMidiAndMastering(unittest.TestCase):
    def test_write_events_to_midi_and_ordering(self) -> None:
        events = [
            MidiEvent(start=0.5, end=1.0, pitch=60, velocity=85),
            MidiEvent(start=1.0, end=1.5, pitch=60, velocity=90),
            MidiEvent(start=0.5, end=1.4, pitch=-64, velocity=40),
            MidiEvent(start=0.6, end=1.2, pitch=-67, velocity=55),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            midi_path = Path(tmp) / "out.mid"
            write_events_to_midi(events, midi_path)
            mid = mido.MidiFile(str(midi_path))
            self.assertEqual(len(mid.tracks), 2)

            msgs = [m for m in mid.tracks[1] if not m.is_meta]
            note_ons = [m for m in msgs if m.type == "note_on" and m.velocity > 0]
            note_offs = [m for m in msgs if m.type == "note_off"]
            cc64 = [m for m in msgs if m.type == "control_change" and m.control == 64]
            cc67 = [m for m in msgs if m.type == "control_change" and m.control == 67]
            self.assertEqual(len(note_ons), 2)
            self.assertEqual(len(note_offs), 2)
            self.assertEqual([m.value for m in cc64], [64, 0])
            self.assertEqual([m.value for m in cc67], [55, 0])

            # At t=1.0s (tick 1920), note_off for first note 60 must precede second note_on
            abs_tick = 0
            tick_1920_types = []
            for m in msgs:
                abs_tick += m.time
                if abs_tick == 1920 and m.type in ("note_on", "note_off"):
                    tick_1920_types.append(m.type)
            self.assertEqual(tick_1920_types, ["note_off", "note_on"])

    def test_master_piano_wav_preserves_lead_and_limits_peak(self) -> None:
        sr = 44100
        t = np.arange(int(3.0 * sr), dtype=np.float32) / sr
        tone = np.zeros_like(t)
        # 0.5s leading silence (must be preserved), 1.5s quiet tone, 1.0s trailing silence
        mask = (t >= 0.5) & (t < 2.0)
        tone[mask] = 0.05 * np.sin(2.0 * np.pi * 440.0 * t[mask])
        stereo = np.stack([tone, tone], axis=1)

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.wav"
            out = Path(tmp) / "mastered.wav"
            sf.write(str(raw), stereo, sr, subtype="PCM_16")
            stats = master_piano_wav(raw, out)
            y, out_sr = sf.read(str(out), dtype="float32", always_2d=True)
            self.assertEqual(out_sr, sr)
            self.assertEqual(y.shape[1], 2)
            # Leading 0.45s silence remains silent -> t=0 alignment preserved
            self.assertLess(float(np.max(np.abs(y[: int(0.45 * sr)]))), 1e-4)
            # Trailing silence trimmed reasonably (~2.5s instead of 3.0s)
            self.assertLess(y.shape[0] / sr, 2.7)
            self.assertGreater(y.shape[0] / sr, 2.0)
            # Peak protected <= -0.95 dBFS
            self.assertLessEqual(stats["peak_db"], -0.95)
            self.assertGreater(stats["rms_db"], -22.0)

    def test_synthesize_piano_version_caching_and_soundfont_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "input.mp3"
            src.write_bytes(b"fake-audio-bytes")
            sf2 = root / "piano.sf2"
            sf2.write_bytes(b"fake-soundfont")
            cache_dir = root / "cache"

            calls = {"transcribe": 0, "render": 0}

            def fake_transcribe(audio_path: Path, model_dir=None):
                calls["transcribe"] += 1
                return [
                    MidiEvent(start=0.5, end=1.0, pitch=60, velocity=80),
                    MidiEvent(start=0.5, end=1.2, pitch=-64, velocity=100),
                ], 2.0

            def fake_render(midi_path: Path, wav_path: Path, soundfont_path: Path, **kwargs):
                calls["render"] += 1
                sr = 44100
                t = np.arange(sr * 2, dtype=np.float32) / sr
                sig = 0.2 * np.sin(2.0 * np.pi * 440.0 * t)
                sf.write(str(wav_path), np.stack([sig, sig], axis=1), sr, subtype="PCM_16")

            with (
                patch("synthesize.transcribe_audio_to_events", side_effect=fake_transcribe),
                patch("synthesize.render_midi_to_wav", side_effect=fake_render),
            ):
                res1 = synthesize_piano_version(src, cache_dir, soundfont=sf2)
                self.assertFalse(res1.cached)
                self.assertEqual(res1.note_count, 1)
                self.assertEqual(res1.pedal_count, 1)
                self.assertEqual(calls, {"transcribe": 1, "render": 1})

                # Second call with identical params -> full cache hit
                res2 = synthesize_piano_version(src, cache_dir, soundfont=sf2)
                self.assertTrue(res2.cached)
                self.assertEqual(calls, {"transcribe": 1, "render": 1})

                # Changing reverb reuses transcribed.mid and only re-runs FluidSynth
                res3 = synthesize_piano_version(src, cache_dir, soundfont=sf2, reverb=False)
                self.assertFalse(res3.cached)
                self.assertEqual(calls, {"transcribe": 1, "render": 2})

                # resynthesize=True re-runs both transcription and rendering
                res4 = synthesize_piano_version(src, cache_dir, soundfont=sf2, reverb=False, resynthesize=True)
                self.assertFalse(res4.cached)
                self.assertEqual(calls, {"transcribe": 2, "render": 3})


if __name__ == "__main__":
    unittest.main()
