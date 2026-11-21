"""Download audio from a YouTube URL as an MP3 file for the MBB pipeline."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_YOUTUBE_HOSTS = {
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
}
_YOUTUBE_SUBDOMAIN_PREFIXES = ("www.", "m.", "music.")
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


@dataclass(frozen=True)
class YouTubeDownload:
    mp3_path: Path
    video_id: str
    title: str
    slug: str
    cached: bool


def _normalize_host(hostname: str | None) -> str:
    host = (hostname or "").lower().rstrip(".")
    for prefix in _YOUTUBE_SUBDOMAIN_PREFIXES:
        if host.startswith(prefix):
            host = host[len(prefix) :]
            break
    return host


def is_http_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_youtube_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    return _normalize_host(parsed.hostname) in _YOUTUBE_HOSTS


def extract_video_id(url: str) -> str | None:
    parsed = urlparse(url.strip())
    host = _normalize_host(parsed.hostname)
    parts = [p for p in parsed.path.split("/") if p]

    candidate: str | None = None
    if host == "youtu.be":
        candidate = parts[0] if parts else None
    elif host in {"youtube.com", "youtube-nocookie.com"}:
        if parts and parts[0] == "watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        elif len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
            candidate = parts[1]

    if candidate and _VIDEO_ID_RE.match(candidate):
        return candidate
    return None


def slugify(text: str, *, fallback: str = "song", max_len: int = 80) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    if len(slug) > max_len:
        cut = slug[:max_len]
        if "-" in cut and cut.rfind("-") >= max_len // 2:
            cut = cut[: cut.rfind("-")]
        slug = cut.rstrip("-")
    return slug or fallback


def _load_cached(cache_dir: Path, video_id: str) -> YouTubeDownload | None:
    video_dir = cache_dir / video_id
    meta_path = video_dir / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mp3_name = meta.get("mp3")
        if not mp3_name:
            return None
        mp3_path = video_dir / mp3_name
        if not mp3_path.is_file() or mp3_path.stat().st_size == 0:
            return None
        return YouTubeDownload(
            mp3_path=mp3_path,
            video_id=str(meta.get("id") or video_id),
            title=str(meta.get("title") or video_id),
            slug=str(meta.get("slug") or slugify(mp3_path.stem, fallback=f"youtube-{video_id.lower()}")),
            cached=True,
        )
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def download_youtube_mp3(url: str, cache_dir: Path, *, redownload: bool = False) -> YouTubeDownload:
    """Download a YouTube video's audio track as an MP3 file, caching by video ID."""
    url = url.strip()
    if not is_youtube_url(url):
        raise ValueError(f"Not a YouTube URL: {url}")
    parsed = urlparse(url)
    if _normalize_host(parsed.hostname) in {"youtube.com", "youtube-nocookie.com"} and parsed.path.rstrip("/") == "/playlist":
        raise ValueError("YouTube playlist URLs are not supported; pass a single video URL instead.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required on PATH to extract MP3 audio from YouTube")

    cache_dir.mkdir(parents=True, exist_ok=True)
    if not redownload:
        known_id = extract_video_id(url)
        if known_id:
            cached = _load_cached(cache_dir, known_id)
            if cached is not None:
                return cached

    import yt_dlp

    def _reject_live(info: dict, *, incomplete: bool) -> str | None:
        if info.get("is_live"):
            return "Live streams are not supported"
        return None

    with tempfile.TemporaryDirectory(dir=cache_dir, prefix=".tmp-") as tmp:
        tmp_dir = Path(tmp)
        ydl_opts = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "outtmpl": str(tmp_dir / "%(id)s.%(ext)s"),
            "js_runtimes": {"deno": {}, "node": {}},
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "0",
                }
            ],
            "match_filter": _reject_live,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            raise RuntimeError(f"Failed to download YouTube audio: {exc}") from exc

        if not info:
            raise RuntimeError("YouTube download was skipped (live streams are not supported).")

        video_id = str(info.get("id") or extract_video_id(url) or "video")
        title = str(info.get("title") or video_id)
        slug = slugify(title, fallback=f"youtube-{video_id.lower()}")

        tmp_mp3 = tmp_dir / f"{video_id}.mp3"
        if not tmp_mp3.is_file():
            mp3s = sorted(tmp_dir.glob("*.mp3"))
            if not mp3s:
                raise RuntimeError("yt-dlp finished without producing an MP3 file.")
            tmp_mp3 = mp3s[0]

        video_dir = cache_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        final_mp3 = video_dir / f"{slug}.mp3"
        for old_mp3 in video_dir.glob("*.mp3"):
            if old_mp3 != final_mp3:
                old_mp3.unlink(missing_ok=True)
        os.replace(tmp_mp3, final_mp3)

        meta = {
            "id": video_id,
            "title": title,
            "slug": slug,
            "mp3": final_mp3.name,
            "url": url,
        }
        meta_tmp = video_dir / "meta.json.tmp"
        meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(meta_tmp, video_dir / "meta.json")

        return YouTubeDownload(
            mp3_path=final_mp3,
            video_id=video_id,
            title=title,
            slug=slug,
            cached=False,
        )
