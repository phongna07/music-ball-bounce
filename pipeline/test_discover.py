"""Unit tests for pipeline/discover.py."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from discover import (
    AudioAssessment,
    DiscoveryCatalog,
    DiscoveryCriteria,
    EvaluatedTrack,
    VideoCandidate,
    assess_audio_file,
    clean_song_title,
    compute_overall_score,
    compute_piano_score,
    discover_tracks,
    evaluate_audio_criteria,
    format_views,
    screen_original_candidate,
    screen_piano_candidate,
    song_identity_key,
    titles_match_song,
)


def _synth_piano_wav(path: Path, *, duration: float = 16.0, interval: float = 0.5, sr: int = 22050) -> None:
    """Synthesize a clean harmonic piano-like sequence (exponentially decaying harmonics)."""
    n = int(duration * sr)
    y = np.zeros(n, dtype=np.float32)
    freqs = [261.63, 293.66, 329.63, 392.00, 440.00, 329.63, 293.66, 261.63]
    t_note = 0.3
    idx = 0
    note_len = int(min(1.2, interval * 1.8) * sr)
    t_local = np.arange(note_len, dtype=np.float32) / sr
    fade_out = np.minimum((note_len - 1 - np.arange(note_len, dtype=np.float32)) / (0.04 * sr), 1.0)
    env = np.minimum(t_local / 0.006, 1.0) * np.exp(-t_local * 5.0) * np.clip(fade_out, 0.0, 1.0)
    while t_note < duration - 0.3:
        f0 = freqs[idx % len(freqs)]
        tone = (
            1.00 * np.sin(2 * np.pi * f0 * t_local)
            + 0.40 * np.sin(2 * np.pi * 2 * f0 * t_local)
            + 0.15 * np.sin(2 * np.pi * 3 * f0 * t_local)
        ) * env
        start = int(t_note * sr)
        end = min(n, start + note_len)
        y[start:end] += 0.35 * tone[: end - start]
        t_note += interval
        idx += 1
    peak = float(np.max(np.abs(y))) or 1.0
    sf.write(str(path), y / peak * 0.8, sr)


def _synth_noisy_pop_wav(path: Path, *, duration: float = 16.0, sr: int = 22050) -> None:
    """Synthesize a harsh, noisy, broadband percussive track (non-piano)."""
    rng = np.random.default_rng(42)
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr
    # Broadband white noise hi-hats + strong sub-bass + high-frequency buzz
    y = 0.25 * rng.normal(0.0, 1.0, size=n).astype(np.float32)
    y += 0.45 * np.sin(2 * np.pi * 55.0 * t)
    y += 0.25 * np.sin(2 * np.pi * 6200.0 * t)
    # Rapid snare-like bursts every 0.12s
    for hit in np.arange(0.2, duration - 0.2, 0.12):
        s = int(hit * sr)
        e = min(n, s + int(0.04 * sr))
        y[s:e] += 0.5 * rng.normal(0.0, 1.0, size=e - s).astype(np.float32)
    peak = float(np.max(np.abs(y))) or 1.0
    sf.write(str(path), y / peak * 0.8, sr)


class _FakeClient:
    """Offline mock of YouTubeClient for testing discover_tracks end to end."""

    def __init__(
        self,
        search_map: dict[str, list[VideoCandidate]],
        meta_map: dict[str, VideoCandidate],
        wav_map: dict[str, Path],
    ) -> None:
        self.search_map = search_map
        self.meta_map = meta_map
        self.wav_map = wav_map
        self.downloaded: list[str] = []

    def search(self, query: str, limit: int = 12) -> list[VideoCandidate]:
        return list(self.search_map.get(query, []))[:limit]

    def fetch_metadata(self, url_or_id: str, source_query: str = "") -> VideoCandidate | None:
        for key, val in self.meta_map.items():
            if key in url_or_id:
                return val
        return None

    def enrich_candidates(self, candidates: list[VideoCandidate], max_workers: int = 4) -> list[VideoCandidate]:
        return [self.meta_map.get(c.video_id, c) for c in candidates]

    def download_preview_audio(self, url: str, dest_dir: Path) -> Path:
        self.downloaded.append(url)
        for vid, wav in self.wav_map.items():
            if vid in url:
                out = dest_dir / "preview.wav"
                out.write_bytes(wav.read_bytes())
                return out
        raise RuntimeError(f"No mock wav for {url}")


class TestTitleAndMetadataScreening(unittest.TestCase):
    def test_clean_song_title(self) -> None:
        self.assertEqual(
            clean_song_title("Wiz Khalifa - See You Again ft. Charlie Puth [Official Video]"),
            "Wiz Khalifa See You Again",
        )
        self.assertEqual(
            clean_song_title("Adele - Someone Like You (Official Music Video) [4K Remaster]"),
            "Adele Someone Like You",
        )
        self.assertEqual(
            clean_song_title("John Legend - All of Me | Official Video"),
            "John Legend All of Me",
        )

    def test_titles_match_song(self) -> None:
        self.assertTrue(
            titles_match_song(
                "Adele Someone Like You",
                "Adele - Someone Like You (Piano Cover by Riyandi Kusuma)",
            )
        )
        self.assertFalse(
            titles_match_song(
                "Adele Someone Like You",
                "Yiruma - River Flows in You (Solo Piano)",
            )
        )

    def test_screen_original_candidate(self) -> None:
        criteria = DiscoveryCriteria(min_original_views=50_000_000)
        good = VideoCandidate(
            video_id="RgKAFK5djSk",
            url="https://www.youtube.com/watch?v=RgKAFK5djSk",
            title="Wiz Khalifa - See You Again ft. Charlie Puth [Official Video]",
            duration=237.0,
            view_count=6_400_000_000,
        )
        self.assertEqual(screen_original_candidate(good, criteria), [])

        compilation = VideoCandidate(
            video_id="70R19IuoyLQ",
            url="https://www.youtube.com/watch?v=70R19IuoyLQ",
            title="THE 100 MOST POPULAR SONGS OF ALL TIME",
            duration=660.0,
            view_count=90_000_000,
        )
        self.assertTrue(screen_original_candidate(compilation, criteria))

        low_views = VideoCandidate(
            video_id="abc12345678",
            url="https://www.youtube.com/watch?v=abc12345678",
            title="Unknown Artist - Demo Song (Official Video)",
            duration=210.0,
            view_count=12_000,
        )
        self.assertTrue(screen_original_candidate(low_views, criteria))

    def test_screen_piano_candidate(self) -> None:
        criteria = DiscoveryCriteria(min_piano_views=50_000)
        good = VideoCandidate(
            video_id="DGjKDQc8LMM",
            url="https://www.youtube.com/watch?v=DGjKDQc8LMM",
            title="Wiz Khalifa - See You Again ft. Charlie Puth (Piano Cover)",
            channel="Piano Nest",
            duration=256.0,
            view_count=1_200_000,
        )
        self.assertEqual(
            screen_piano_candidate(good, criteria, target_song="Wiz Khalifa See You Again"),
            [],
        )

        # Reject vocal cover, tutorial, 1-hour loop, and mismatched song
        rejected_titles = [
            "Wiz Khalifa - See You Again | Piano & Vocal Cover",
            "Wiz Khalifa - See You Again (Easy Piano Tutorial)",
            "Wiz Khalifa - See You Again Piano Cover 1 Hour Loop",
            "Best Popular Piano Covers of 2024 - 100 Songs Compilation",
            "Ed Sheeran - Perfect (Piano Cover)",
        ]
        for title in rejected_titles:
            with self.subTest(title=title):
                cand = VideoCandidate(
                    video_id="test1234567",
                    url="https://www.youtube.com/watch?v=test1234567",
                    title=title,
                    duration=240.0,
                    view_count=500_000,
                )
                self.assertTrue(
                    screen_piano_candidate(cand, criteria, target_song="Wiz Khalifa See You Again")
                )

    def test_format_views(self) -> None:
        self.assertEqual(format_views(6_400_000_000), "6.40B")
        self.assertEqual(format_views(12_500_000), "12.5M")
        self.assertEqual(format_views(85_000), "85K")
        self.assertEqual(format_views(None), "n/a")


class TestAudioAssessmentAndScoring(unittest.TestCase):
    def test_slow_piano_vs_fast_piano_vs_noisy_pop(self) -> None:
        criteria = DiscoveryCriteria()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            slow_wav = tmp_dir / "slow_piano.wav"
            fast_wav = tmp_dir / "fast_piano.wav"
            pop_wav = tmp_dir / "noisy_pop.wav"

            _synth_piano_wav(slow_wav, duration=14.0, interval=0.50)  # ~2 onsets/s
            _synth_piano_wav(fast_wav, duration=14.0, interval=0.14)  # ~7 onsets/s
            _synth_noisy_pop_wav(pop_wav, duration=14.0)

            slow_a = assess_audio_file(slow_wav)
            fast_a = assess_audio_file(fast_wav)
            pop_a = assess_audio_file(pop_wav)

            # Slow clean piano passes all criteria
            self.assertEqual(evaluate_audio_criteria(slow_a, criteria), [])
            self.assertGreaterEqual(slow_a.piano_score, 0.80)
            self.assertLessEqual(slow_a.onsets_per_sec, 3.0)

            # Fast piano fails pace checks but still reads as piano
            fast_reasons = evaluate_audio_criteria(fast_a, criteria)
            self.assertTrue(fast_reasons)
            self.assertGreater(fast_a.onsets_per_sec, criteria.max_onsets_per_sec)
            self.assertGreaterEqual(fast_a.piano_score, 0.70)

            # Noisy broadband pop fails piano purity check
            pop_reasons = evaluate_audio_criteria(pop_a, criteria)
            self.assertTrue(any("piano purity" in r for r in pop_reasons))
            self.assertLess(pop_a.piano_score, criteria.min_piano_score)

    def test_overall_score_rewards_higher_popularity(self) -> None:
        a = AudioAssessment(
            sample_duration=50.0,
            windows_analyzed=2,
            total_onsets=110,
            onsets_per_sec=2.2,
            peak_onsets_per_sec=2.6,
            picked_per_sec=2.1,
            median_ioi=0.40,
            p25_ioi=0.28,
            fast_frac=0.03,
            duo_gain=1.03,
            tempo=80.0,
            harmonic_ratio=0.96,
            spectral_flatness=0.00003,
            spectral_centroid=850.0,
            spectral_rolloff=1400.0,
            high_freq_ratio=0.0005,
            piano_band_ratio=0.90,
            rms_db=-18.0,
            piano_score=0.98,
            pace_score=0.96,
        )
        hi = compute_overall_score(a, piano_views=5_000_000, original_views=3_000_000_000)
        lo = compute_overall_score(a, piano_views=60_000, original_views=10_000_000)
        self.assertGreater(hi, lo)
        self.assertGreater(hi, 90.0)

    def test_piano_score_calibration(self) -> None:
        piano = compute_piano_score(
            harmonic_ratio=0.96,
            spectral_flatness=0.00002,
            spectral_rolloff=1600.0,
            high_freq_ratio=0.0004,
            piano_band_ratio=0.85,
        )
        pop = compute_piano_score(
            harmonic_ratio=0.93,
            spectral_flatness=0.0031,
            spectral_rolloff=3424.0,
            high_freq_ratio=0.0185,
            piano_band_ratio=0.44,
        )
        self.assertGreater(piano, 0.95)
        self.assertLess(pop, 0.35)


class TestCatalogAndEndToEndDiscovery(unittest.TestCase):
    def test_catalog_persistence_and_mark_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cat_path = Path(tmp) / "catalog.json"
            cat = DiscoveryCatalog(cat_path)
            t1 = EvaluatedTrack(
                video_id="vidA1234567",
                url="https://www.youtube.com/watch?v=vidA1234567",
                title="Song A (Piano Cover)",
                channel="Pianist",
                duration=220.0,
                view_count=500_000,
                mode="popular-to-piano",
                song_key="song-a",
                source_query="q",
                accepted=True,
                score=91.5,
                rejection_reasons=[],
                evaluated_at="2026-01-01T00:00:00+00:00",
            )
            t2 = EvaluatedTrack(
                video_id="vidB1234567",
                url="https://www.youtube.com/watch?v=vidB1234567",
                title="Song B (Fast Piano)",
                channel="Pianist",
                duration=200.0,
                view_count=300_000,
                mode="popular-to-piano",
                song_key="song-b",
                source_query="q",
                accepted=False,
                score=40.0,
                rejection_reasons=["pace too fast"],
                evaluated_at="2026-01-01T00:00:00+00:00",
            )
            cat.add_track(t1)
            cat.add_track(t2)
            cat.save()

            reloaded = DiscoveryCatalog(cat_path)
            self.assertEqual([t.video_id for t in reloaded.list_tracks()], ["vidA1234567"])
            self.assertEqual(len(reloaded.list_tracks(include_rejected=True)), 2)

            reloaded.mark_used("vidA1234567")
            reloaded2 = DiscoveryCatalog(cat_path)
            self.assertEqual(reloaded2.list_tracks(unused_only=True), [])

    def test_popular_to_piano_discovery_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            slow_wav = tmp_dir / "slow.wav"
            fast_wav = tmp_dir / "fast.wav"
            _synth_piano_wav(slow_wav, duration=12.0, interval=0.50)
            _synth_piano_wav(fast_wav, duration=12.0, interval=0.14)

            orig = VideoCandidate(
                video_id="ORIG0000001",
                url="https://www.youtube.com/watch?v=ORIG0000001",
                title="Adele - Someone Like You (Official Music Video)",
                channel="Adele",
                duration=285.0,
                view_count=None,  # flat search missing views -> triggers enrichment
            )
            orig_enriched = VideoCandidate(
                video_id="ORIG0000001",
                url="https://www.youtube.com/watch?v=ORIG0000001",
                title="Adele - Someone Like You (Official Music Video)",
                channel="Adele",
                duration=285.0,
                view_count=2_200_000_000,
            )
            fast_cover = VideoCandidate(
                video_id="PIANOFAST01",
                url="https://www.youtube.com/watch?v=PIANOFAST01",
                title="Adele - Someone Like You (Piano Cover Fast)",
                channel="Fast Keys",
                duration=240.0,
                view_count=2_000_000,
            )
            slow_cover = VideoCandidate(
                video_id="PIANOSLOW01",
                url="https://www.youtube.com/watch?v=PIANOSLOW01",
                title="Adele - Someone Like You (Solo Piano Cover)",
                channel="Gentle Piano",
                duration=290.0,
                view_count=900_000,
            )

            client = _FakeClient(
                search_map={
                    "2010s ballads": [orig],
                    "Adele Someone Like You piano cover": [fast_cover, slow_cover],
                },
                meta_map={
                    "ORIG0000001": orig_enriched,
                    "PIANOFAST01": fast_cover,
                    "PIANOSLOW01": slow_cover,
                },
                wav_map={
                    "PIANOFAST01": fast_wav,
                    "PIANOSLOW01": slow_wav,
                },
            )
            cat = DiscoveryCatalog(tmp_dir / "catalog.json")
            found = discover_tracks(
                count=2,
                mode="popular-to-piano",
                queries=["2010s ballads"],
                catalog=cat,
                client=client,
                log=None,
            )

            # Fast cover was checked first (more views) and rejected; slow cover accepted!
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].video_id, "PIANOSLOW01")
            self.assertTrue(found[0].accepted)
            self.assertEqual(found[0].original["view_count"], 2_200_000_000)
            self.assertIn("PIANOFAST01", cat.tracks)
            self.assertFalse(cat.tracks["PIANOFAST01"].accepted)

            # Second run without --recheck skips already-seen original & videos
            found_again = discover_tracks(
                count=2,
                mode="popular-to-piano",
                queries=["2010s ballads"],
                catalog=cat,
                client=client,
                log=None,
            )
            self.assertEqual(found_again, [])

    def test_song_identity_key_dedup(self) -> None:
        k1 = song_identity_key("Adele - Someone Like You (Official Music Video)")
        k2 = song_identity_key("Someone Like You - Adele [Piano Cover]")
        self.assertEqual(k1, k2)


if __name__ == "__main__":
    unittest.main()
