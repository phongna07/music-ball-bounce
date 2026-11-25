"""Unit tests for important-note accent detection and screen-effect options."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import mido

from analyze import extract_midi_notes
from choreograph import SCREEN_EFFECT_MODES, VIDEO_FPS, choreograph


def _make_sample_analysis(*, with_midi_meta: bool = True, duration: float = 18.0) -> dict:
    efps = 60
    n_frames = int(duration * efps)
    rms = [0.22] * n_frames
    low = [0.15] * n_frames

    # Musical climax hits at t = 4.0s and t = 12.0s with acoustic + bass surges
    for peak_t, r_peak, l_peak in ((4.0, 0.95, 0.85), (12.0, 1.0, 0.92)):
        i0 = int(round(peak_t * efps))
        for k in range(18):
            idx = i0 + k
            if idx < n_frames:
                decay = 1.0 - k / 18.0
                rms[idx] = max(rms[idx], 0.22 + (r_peak - 0.22) * decay)
                low[idx] = max(low[idx], 0.15 + (l_peak - 0.15) * decay)

    notes = []
    t = 0.5
    while t <= duration - 0.5:
        is_peak = abs(t - 4.0) < 0.05 or abs(t - 12.0) < 0.05
        # Leave a 2.5s silence around t = 8.0s..10.5s so beat-fill kicks in
        if 7.7 < t < 10.3:
            t += 0.5
            continue
        item = {
            "t": round(t, 4),
            "strength": 0.96 if is_peak else 0.55,
            "melodic": 0.95,
            "pitch": 72 if is_peak else 60 + (int(round(t * 2)) % 5),
        }
        if with_midi_meta:
            item["chord_size"] = 5 if is_peak else 1
            item["pitch_span"] = 36 if is_peak else 0
            item["vel_energy"] = 1.9 if is_peak else 0.55
        else:
            item["onset_raw"] = 1.25 if is_peak else 0.45
        notes.append(item)
        t += 0.5

    return {
        "version": 3,
        "note_source": "midi" if with_midi_meta else "audio",
        "duration": duration,
        "tempo": 120.0,
        "beats": [round(0.5 * k, 3) for k in range(int(duration * 2))],
        "downbeats": [round(2.0 * k, 3) for k in range(int(duration / 2))],
        "notes": notes,
        "kicks": [{"t": 4.0, "strength": 0.95}, {"t": 12.0, "strength": 0.98}],
        "sections": [
            {"t0": 0.0, "t1": 11.9, "energy": 0.65},
            {"t0": 11.9, "t1": duration, "energy": 1.0},
        ],
        "energy": {"fps": efps, "rms": rms, "low": low, "high": [0.1] * n_frames},
    }


class TestScreenEffects(unittest.TestCase):
    def test_extract_midi_notes_records_chord_and_velocity_energy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mid_path = Path(tmp) / "chords.mid"
            mid = mido.MidiFile(type=1, ticks_per_beat=480)
            mt = mido.MidiTrack()
            mt.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))  # 960 ticks/sec
            mt.append(mido.MetaMessage("end_of_track", time=0))
            mid.tracks.append(mt)

            pt = mido.MidiTrack()
            tps = 960.0
            evs = [
                # Single soft note at t=0.5s
                (int(round(0.5 * tps)), mido.Message("note_on", channel=0, note=60, velocity=60, time=0)),
                (int(round(0.8 * tps)), mido.Message("note_off", channel=0, note=60, velocity=0, time=0)),
                # 3-note chord at t=1.5s spanning 24 semitones
                (int(round(1.500 * tps)), mido.Message("note_on", channel=0, note=48, velocity=90, time=0)),
                (int(round(1.510 * tps)), mido.Message("note_on", channel=0, note=60, velocity=100, time=0)),
                (int(round(1.515 * tps)), mido.Message("note_on", channel=0, note=72, velocity=110, time=0)),
                (int(round(1.900 * tps)), mido.Message("note_off", channel=0, note=48, velocity=0, time=0)),
                (int(round(1.900 * tps)), mido.Message("note_off", channel=0, note=60, velocity=0, time=0)),
                (int(round(1.900 * tps)), mido.Message("note_off", channel=0, note=72, velocity=0, time=0)),
            ]
            evs.sort(key=lambda x: x[0])
            prev = 0
            for tick, msg in evs:
                pt.append(msg.copy(time=tick - prev))
                prev = tick
            pt.append(mido.MetaMessage("end_of_track", time=0))
            mid.tracks.append(pt)
            mid.save(str(mid_path))

            notes = extract_midi_notes(mid_path, duration=3.0)
            self.assertEqual(len(notes), 2)
            self.assertEqual(notes[0]["chord_size"], 1)
            self.assertEqual(notes[0]["pitch_span"], 0)
            self.assertAlmostEqual(notes[0]["vel_energy"], 0.6, places=3)
            self.assertEqual(notes[1]["chord_size"], 3)
            self.assertEqual(notes[1]["pitch_span"], 24)
            expected_vel_e = ((90**2 + 100**2 + 110**2) ** 0.5) / 100.0
            self.assertAlmostEqual(notes[1]["vel_energy"], expected_vel_e, places=2)

    def test_choreograph_accent_detection_cooldown_and_fill_exclusion(self) -> None:
        analysis = _make_sample_analysis(with_midi_meta=True)
        choreo = choreograph(analysis, seed=1, balls=1, screen_effect="both", effect_strength=1.2)
        plates = choreo["plates"]
        self.assertEqual(choreo["screenEffect"], {"mode": "both", "strength": 1.2})
        self.assertEqual(choreo["report"]["screen_effect"], "both")
        self.assertGreaterEqual(choreo["report"]["accent_notes"], 2)

        accents = [p for p in plates if p["accent"] > 0]
        self.assertEqual(len(accents), choreo["report"]["accent_notes"])

        # Cooldown spacing respected (tempo=120 -> beat=0.5s -> cooldown=1.5s)
        for a, b in zip(accents[:-1], accents[1:]):
            self.assertGreaterEqual(b["t"] - a["t"], 1.5 - 1e-6)

        # The two climax hits (t=4.0s and t=12.0s) must be selected as top accents
        accent_times = [p["t"] for p in accents]
        self.assertTrue(any(abs(t - 4.0) <= 1.0 / VIDEO_FPS for t in accent_times))
        self.assertTrue(any(abs(t - 12.0) <= 1.0 / VIDEO_FPS for t in accent_times))

        # Beat-grid fill notes placed in the 8.0s..10.0s gap must never be accented
        fill_plates = [p for p in plates if 8.0 <= p["t"] <= 10.0]
        self.assertTrue(len(fill_plates) >= 1)
        for p in fill_plates:
            self.assertEqual(p["importance"], 0.0)
            self.assertEqual(p["accent"], 0.0)

    def test_audio_only_and_two_ball_accents(self) -> None:
        audio_analysis = _make_sample_analysis(with_midi_meta=False)
        choreo_audio = choreograph(audio_analysis, seed=1, balls=1, screen_effect="splash")
        self.assertEqual(choreo_audio["screenEffect"], {"mode": "splash", "strength": 1.0})
        self.assertGreaterEqual(choreo_audio["report"]["accent_notes"], 2)

        # Force two-ball mode and confirm accents align across sorted plates
        choreo_duo = choreograph(_make_sample_analysis(with_midi_meta=True), seed=2, balls=2, screen_effect="shake")
        self.assertEqual(choreo_duo["report"]["balls"], 2)
        self.assertEqual(choreo_duo["screenEffect"]["mode"], "shake")
        self.assertGreaterEqual(choreo_duo["report"]["accent_notes"], 1)
        for p in choreo_duo["plates"]:
            self.assertIn("importance", p)
            self.assertIn("accent", p)
            self.assertGreaterEqual(p["accent"], 0.0)
            self.assertLessEqual(p["accent"], 1.0)

    def test_screen_effect_validation(self) -> None:
        analysis = _make_sample_analysis()
        for mode in SCREEN_EFFECT_MODES:
            res = choreograph(analysis, seed=1, balls=1, screen_effect=mode)
            self.assertEqual(res["screenEffect"]["mode"], mode)
        with self.assertRaises(SystemExit):
            choreograph(analysis, seed=1, balls=1, screen_effect="invalid")
        with self.assertRaises(SystemExit):
            choreograph(analysis, seed=1, balls=1, effect_strength=-0.5)


if __name__ == "__main__":
    unittest.main()
