"""Unit tests for precision audio-visual synchronization."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import mido
import numpy as np
import soundfile as sf

from analyze import extract_midi_notes, refine_onset_times
from choreograph import VIDEO_FPS, choreograph
from synthesize import lookup_attack_delay


class TestSyncAccuracy(unittest.TestCase):
    def test_lookup_attack_delay(self) -> None:
        self.assertEqual(lookup_attack_delay(None, 60, 80), 0.0)
        attack_map = {
            "pitches": [36, 60, 84],
            "velocities": [45, 80, 115],
            "median_delay": 0.016,
            "delays": {
                "36:45": 0.021,
                "60:80": 0.0165,
                "84:115": 0.012,
            },
        }
        self.assertAlmostEqual(lookup_attack_delay(attack_map, 60, 80), 0.0165)
        self.assertAlmostEqual(lookup_attack_delay(attack_map, 61, 82), 0.0165)
        self.assertAlmostEqual(lookup_attack_delay(attack_map, 83, 110), 0.012)
        # Missing grid cell falls back to median_delay
        self.assertAlmostEqual(lookup_attack_delay(attack_map, 36, 115), 0.016)

    def test_extract_midi_notes_clip_alignment_and_clustering(self) -> None:
        attack_map = {
            "pitches": [60, 72],
            "velocities": [50, 100],
            "median_delay": 0.015,
            "delays": {
                "60:50": 0.020,
                "60:100": 0.015,
                "72:50": 0.020,
                "72:100": 0.012,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            mid_path = Path(tmp) / "test.mid"
            mid = mido.MidiFile(type=1, ticks_per_beat=960)
            mt = mido.MidiTrack()
            mt.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
            mt.append(mido.MetaMessage("end_of_track", time=0))
            mid.tracks.append(mt)

            pt = mido.MidiTrack()
            pt.append(mido.Message("program_change", channel=0, program=0, time=0))
            tps = 1920.0
            # Notes at t=2.0 (before clip_start), t=10.500 (chord of 60@50 + 72@100), t=11.000, t=20.0 (after clip)
            raw = [
                (int(round(2.0 * tps)), mido.Message("note_on", channel=0, note=60, velocity=80, time=0)),
                (int(round(2.4 * tps)), mido.Message("note_off", channel=0, note=60, velocity=0, time=0)),
                (int(round(10.500 * tps)), mido.Message("note_on", channel=0, note=60, velocity=50, time=0)),
                (int(round(10.510 * tps)), mido.Message("note_on", channel=0, note=72, velocity=100, time=0)),
                (int(round(10.900 * tps)), mido.Message("note_off", channel=0, note=60, velocity=0, time=0)),
                (int(round(10.900 * tps)), mido.Message("note_off", channel=0, note=72, velocity=0, time=0)),
                (int(round(11.000 * tps)), mido.Message("note_on", channel=0, note=60, velocity=100, time=0)),
                (int(round(11.400 * tps)), mido.Message("note_off", channel=0, note=60, velocity=0, time=0)),
                (int(round(20.000 * tps)), mido.Message("note_on", channel=0, note=60, velocity=100, time=0)),
                (int(round(20.400 * tps)), mido.Message("note_off", channel=0, note=60, velocity=0, time=0)),
            ]
            raw.sort(key=lambda item: item[0])
            prev = 0
            for tick, msg in raw:
                pt.append(msg.copy(time=tick - prev))
                prev = tick
            pt.append(mido.MetaMessage("end_of_track", time=0))
            mid.tracks.append(pt)
            mid.save(str(mid_path))

            clip_start = 10.00001  # snaps to sample round(10.00001*44100)/44100 = 10.0
            notes = extract_midi_notes(
                mid_path,
                duration=5.0,
                clip_start=clip_start,
                attack_map=attack_map,
            )
            self.assertEqual(len(notes), 2)
            # First cluster: (0.500+0.020)=0.520 @ w=2500 and (0.510+0.012)=0.522 @ w=10000 -> 0.5216
            expected_t0 = (0.520 * (50**2) + 0.522 * (100**2)) / (50**2 + 100**2)
            self.assertAlmostEqual(notes[0]["t"], expected_t0, delta=1e-3)
            self.assertEqual(notes[0]["pitch"], 72)
            self.assertAlmostEqual(notes[1]["t"], 1.015, delta=1e-3)

    def test_refine_onset_times_sub_3ms_accuracy(self) -> None:
        sr = 44100
        dur = 4.0
        n = int(dur * sr)
        t_axis = np.arange(n, dtype=np.float32) / sr
        y = np.zeros(n, dtype=np.float32)

        true_onsets = np.array([0.500, 0.900, 0.985, 1.500, 2.250, 3.100], dtype=np.float64)
        for t0 in true_onsets:
            i0 = int(round(t0 * sr))
            seg_t = np.arange(0, int(0.25 * sr), dtype=np.float32) / sr
            # Crisp harmonic burst with realistic 3 ms attack ramp and exponential decay
            env = np.minimum(1.0, seg_t / 0.003) * np.exp(-12.0 * seg_t)
            sig = (
                0.5 * np.sin(2.0 * np.pi * 440.0 * seg_t)
                + 0.3 * np.sin(2.0 * np.pi * 880.0 * seg_t)
                + 0.2 * np.sin(2.0 * np.pi * 1760.0 * seg_t)
            ) * env
            y[i0 : i0 + len(sig)] += sig

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "pulses.wav"
            sf.write(str(wav_path), y, sr, subtype="PCM_16")
            # Intentionally offset coarse guesses (+22 ms lag + jitter)
            coarse = true_onsets + np.array([0.024, 0.018, 0.027, 0.021, 0.019, 0.025])
            refined = refine_onset_times(wav_path, coarse)
            err_ms = (refined - true_onsets) * 1000.0
            self.assertLess(float(np.max(np.abs(err_ms))), 3.5)
            self.assertLess(abs(float(np.median(err_ms))), 2.5)
            # Adjacent 85 ms notes must remain distinct and ordered
            self.assertGreater(float(refined[2] - refined[1]), 0.060)

    def test_choreograph_frame_snapping_and_max_gap(self) -> None:
        # Note gap of 1.5s should NOT get a phantom beat fill under default max_gap=1.8,
        # while a 3.5s silence SHOULD get a beat fill.
        raw_times = [0.503, 1.007, 2.502, 3.011, 6.504, 7.009, 7.501, 8.006]
        analysis = {
            "version": 2,
            "note_source": "midi",
            "duration": 9.0,
            "tempo": 120.0,
            "beats": [round(0.5 * k, 3) for k in range(18)],
            "downbeats": [0.0, 2.0, 4.0, 6.0, 8.0],
            "notes": [
                {"t": t, "strength": 0.85, "melodic": 0.95, "pitch": 60 + k}
                for k, t in enumerate(raw_times)
            ],
            "kicks": [],
            "sections": [{"t0": 0.0, "t1": 9.0, "energy": 0.8}],
            "energy": {"fps": 60, "rms": [0.5] * (9 * 60), "low": [0.3] * (9 * 60), "high": [0.2] * (9 * 60)},
        }
        choreo = choreograph(analysis, seed=1, balls=1)
        plates = choreo["plates"]
        # 8 real notes + 2 beat-grid fills across the 3.5s gap (6.504 - 3.011 = 3.493s -> 2 fills)
        self.assertEqual(len(plates), 10)
        self.assertEqual(choreo["report"]["offset"], 0.0)
        self.assertEqual(choreo["report"]["snap_fps"], VIDEO_FPS)

        for p, seg in zip(plates, choreo["balls"][0]["segments"][1:]):
            expected = round(p["frame"] / VIDEO_FPS, 6)
            self.assertAlmostEqual(p["t"], expected, places=6)
            self.assertAlmostEqual(seg["t0"], expected, places=6)
            # Every real note stays within half a frame (8.34 ms)
            if p["strength"] > 0.4:
                nearest = min(raw_times, key=lambda rt: abs(rt - p["t"]))
                self.assertLessEqual(abs(p["t"] - nearest), 0.5 / VIDEO_FPS + 1e-6)


if __name__ == "__main__":
    unittest.main()
