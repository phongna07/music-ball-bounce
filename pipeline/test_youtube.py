"""Unit tests for pipeline/youtube.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from youtube import download_youtube_mp3, extract_video_id, is_http_url, is_youtube_url, slugify


class TestYouTubeHelpers(unittest.TestCase):
    def test_is_youtube_url(self) -> None:
        valid = [
            "https://youtu.be/DGjKDQc8LMM",
            "https://youtu.be/DGjKDQc8LMM?si=abc123&t=42",
            "https://www.youtube.com/watch?v=DGjKDQc8LMM",
            "https://youtube.com/watch?v=DGjKDQc8LMM&list=PL123",
            "https://m.youtube.com/watch?v=DGjKDQc8LMM",
            "https://music.youtube.com/watch?v=DGjKDQc8LMM",
            "https://www.youtube.com/shorts/DGjKDQc8LMM",
            "https://www.youtube-nocookie.com/embed/DGjKDQc8LMM",
            "http://youtu.be/DGjKDQc8LMM",
        ]
        for url in valid:
            with self.subTest(url=url):
                self.assertTrue(is_youtube_url(url))

        invalid = [
            "path/to/song.mp3",
            "/abs/path/youtube.com.mp3",
            "https://example.com/watch?v=DGjKDQc8LMM",
            "https://notyoutube.com/watch?v=DGjKDQc8LMM",
            "ftp://youtu.be/DGjKDQc8LMM",
        ]
        for val in invalid:
            with self.subTest(val=val):
                self.assertFalse(is_youtube_url(val))

    def test_is_http_url(self) -> None:
        self.assertTrue(is_http_url("https://example.com/song.mp3"))
        self.assertTrue(is_http_url("http://youtu.be/DGjKDQc8LMM"))
        self.assertFalse(is_http_url("songs/track.mp3"))
        self.assertFalse(is_http_url("/tmp/track.mp3"))

    def test_extract_video_id(self) -> None:
        cases = {
            "https://youtu.be/DGjKDQc8LMM": "DGjKDQc8LMM",
            "https://youtu.be/DGjKDQc8LMM?t=12": "DGjKDQc8LMM",
            "https://www.youtube.com/watch?v=DGjKDQc8LMM&feature=share": "DGjKDQc8LMM",
            "https://music.youtube.com/watch?v=DGjKDQc8LMM": "DGjKDQc8LMM",
            "https://www.youtube.com/shorts/DGjKDQc8LMM": "DGjKDQc8LMM",
            "https://www.youtube.com/embed/DGjKDQc8LMM": "DGjKDQc8LMM",
            "https://www.youtube.com/live/DGjKDQc8LMM": "DGjKDQc8LMM",
            "https://www.youtube.com/playlist?list=PL12345": None,
            "https://youtu.be/too-short": None,
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(extract_video_id(url), expected)

    def test_slugify(self) -> None:
        self.assertEqual(
            slugify("Wiz Khalifa - See You Again ft. Charlie Puth (Piano Cover)"),
            "wiz-khalifa-see-you-again-ft-charlie-puth-piano-cover",
        )
        self.assertEqual(slugify("Café del Mar"), "cafe-del-mar")
        self.assertEqual(slugify("🎵🎶", fallback="youtube-dgjkdqc8lmm"), "youtube-dgjkdqc8lmm")
        long_title = "a-very-long-title-" + "-".join(["word"] * 25)
        self.assertLessEqual(len(slugify(long_title, max_len=60)), 60)

    def test_cache_hit_skips_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            vid_dir = cache_dir / "DGjKDQc8LMM"
            vid_dir.mkdir(parents=True)
            mp3 = vid_dir / "see-you-again.mp3"
            mp3.write_bytes(b"ID3fake-mp3-data")
            (vid_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "id": "DGjKDQc8LMM",
                        "title": "See You Again",
                        "slug": "see-you-again",
                        "mp3": "see-you-again.mp3",
                    }
                ),
                encoding="utf-8",
            )
            res = download_youtube_mp3("https://youtu.be/DGjKDQc8LMM?t=5", cache_dir)
            self.assertTrue(res.cached)
            self.assertEqual(res.mp3_path, mp3)
            self.assertEqual(res.slug, "see-you-again")
            self.assertEqual(res.title, "See You Again")

    def test_rejects_playlist_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                download_youtube_mp3("https://www.youtube.com/playlist?list=PL12345", Path(tmp))


if __name__ == "__main__":
    unittest.main()
