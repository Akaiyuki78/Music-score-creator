from __future__ import annotations
import base64
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .models import SongScore, SongSection, SongLine
from .pinyin_engine import tokenize_chordpro_line
from .theory import recommend_guitar_capo

YOUTUBE_URL_REGEX = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[a-zA-Z0-9_\-]{6,16}"
)


def is_youtube_url(url_or_path: str) -> bool:
    """Checks whether a string is a valid YouTube URL."""
    if not url_or_path or not isinstance(url_or_path, str):
        return False
    return bool(YOUTUBE_URL_REGEX.match(url_or_path.strip()))


def download_youtube_audio(
    youtube_url: str,
    output_dir: Optional[Union[str, Path]] = None,
    preferred_format: str = "mp3",
) -> Tuple[Path, Dict[str, str]]:
    """
    Downloads the audio track of a YouTube video using `yt-dlp` and extracts title & uploader metadata.
    Returns (audio_file_path, metadata_dict).
    """
    import yt_dlp
    import certifi

    out_folder = Path(output_dir or tempfile.mkdtemp(prefix="yt_audio_"))
    out_folder.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(out_folder / "%(title).60s-%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "noplaylist": True,
        "extract_flat": False,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "zh-Hans", "zh-Hant", "zh-TW", "zh"],
        "subtitlesformat": "vtt",
        "ca_certs": certifi.where(),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": preferred_format,
                "preferredquality": "192",
            }
        ],
    }

    # Clean URL if it has radio/playlist params
    clean_url = re.sub(r"&list=[^&]+", "", youtube_url.strip())
    clean_url = re.sub(r"&start_radio=[^&]+", "", clean_url)
    clean_url = re.sub(r"&index=[^&]+", "", clean_url)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(clean_url, download=True)
        if "entries" in info and info["entries"]:
            info = info["entries"][0]
        title = info.get("title", "YouTube Song")
        uploader = info.get("uploader", "")
        # Find the downloaded file
        expected_path = Path(ydl.prepare_filename(info)).with_suffix(f".{preferred_format}")
        if not expected_path.exists():
            # Search output directory in case of character sanitization
            matches = list(out_folder.glob(f"*.{preferred_format}"))
            if matches:
                expected_path = matches[0]

    # Try extracting lyrics from downloaded subtitles (.vtt)
    subtitles_lyrics = ""
    vtt_candidates = list(out_folder.glob("*.vtt"))
    if vtt_candidates:
        # Prefer English if title is mostly English, or Chinese if CJK
        has_cjk_title = any("\u4e00" <= char <= "\u9fff" for char in title)
        chosen_vtt = None
        for vtt in vtt_candidates:
            if has_cjk_title and ("zh" in vtt.name or "Hans" in vtt.name or "Hant" in vtt.name):
                chosen_vtt = vtt
                break
            elif not has_cjk_title and ("en" in vtt.name):
                chosen_vtt = vtt
                break
        if not chosen_vtt:
            chosen_vtt = vtt_candidates[0]

        try:
            vtt_text = chosen_vtt.read_text(encoding="utf-8", errors="ignore")
            clean_lines: List[str] = []
            for raw_line in vtt_text.splitlines():
                raw_line = raw_line.strip()
                if (
                    not raw_line
                    or raw_line.startswith("WEBVTT")
                    or "-->" in raw_line
                    or raw_line.startswith("Kind:")
                    or raw_line.startswith("Language:")
                    or raw_line.startswith("NOTE")
                ):
                    continue
                cleaned = re.sub(r"<[^>]+>", "", raw_line).strip()
                # Strip musical note characters (♪, ♫)
                cleaned = re.sub(r"[\u266a\u266b\u266c\u266d\u266f♪♫]", "", cleaned).strip()
                # Skip pure non-sung sound descriptions like (music) or [Applause]
                if not cleaned or re.match(r"^[\[\(].+[\]\)]$", cleaned):
                    continue
                if not clean_lines or clean_lines[-1].lower() != cleaned.lower():
                    clean_lines.append(cleaned)
            if clean_lines:
                subtitles_lyrics = "\n".join(clean_lines)
        except Exception:
            pass

    # Check if title has Chinese characters vs extracted subtitles
    has_cjk_title = any("\u4e00" <= char <= "\u9fff" for char in title)
    has_cjk_lyrics = any("\u4e00" <= char <= "\u9fff" for char in subtitles_lyrics)

    # If YouTube had no captions OR if YouTube only provided an English translation track for a Chinese song
    # (very common on BinMusic / Mayday / JVR videos with burned-in video text and English CC for foreign viewers),
    # fetch the authentic Chinese lyrics from the free open music lyrics database (lrclib)!
    synced_lyrics_list: List[Tuple[float, str]] = []
    if not subtitles_lyrics or (has_cjk_title and not has_cjk_lyrics):
        plain_l, synced_l = fetch_lyrics_online(title, uploader)
        if plain_l:
            subtitles_lyrics = plain_l
            synced_lyrics_list = synced_l

    return expected_path, {
        "title": title,
        "artist": uploader,
        "id": info.get("id", ""),
        "duration": info.get("duration", 0),
        "subtitles_lyrics": subtitles_lyrics,
        "synced_lyrics": synced_lyrics_list,
    }


def parse_lrc_timestamps(lrc_text: str) -> List[Tuple[float, str]]:
    """Parses standard LRC synchronized lines like [01:23.45] lyric text into (seconds, text)."""
    results: List[Tuple[float, str]] = []
    for line in lrc_text.splitlines():
        m = re.match(r"\[(\d{2}):(\d{2}(?:\.\d{1,3})?)\](.*)", line.strip())
        if m:
            m_min, m_sec, txt = m.groups()
            try:
                t = float(m_min) * 60.0 + float(m_sec)
                clean_t = re.sub(r"[\u266a\u266b\u266c\u266d\u266f♪♫]", "", txt).strip()
                if clean_t and not re.match(r"^[\[\(].+[\]\)]$", clean_t):
                    results.append((t, clean_t))
            except ValueError:
                pass
    return results


def fetch_lyrics_online(raw_title: str, uploader: str = "") -> Tuple[str, List[Tuple[float, str]]]:
    """
    Auto-fetches song lyrics from the free, open-source lrclib database
    when a YouTube video lacks closed captions. Supports Mandopop, worship songs, and English pop.
    Returns (plain_lyrics_text, list_of_synced_timestamp_tuples).
    """
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    queries: List[str] = []
    # 1. Look for song title inside brackets like 【晴天】 or [Sunny Day]
    m = re.search(r"[【\[「『(]([^】\]」』)]+)[】\]」』)]", raw_title)
    if m:
        inner = m.group(1).strip()
        cjk = "".join(re.findall(r"[\u4e00-\u9fff]+", inner))
        if cjk:
            queries.append(f"{cjk} 周杰伦" if ("周杰" in raw_title or "Jay" in raw_title) else cjk)
        queries.append(inner)

    # 2. Clean general YouTube metadata noise
    clean = re.sub(
        r"(Official\s*(Music\s*)?Video|Official\s*MV|MV|HD|4K|1080p|Audio|Lyrics|720p)",
        "",
        raw_title,
        flags=re.I,
    )
    clean = clean.replace("【", " ").replace("】", " ").replace("[", " ").replace("]", " ").strip()
    clean = re.sub(r"\s+", " ", clean)
    queries.append(clean)

    for q in queries:
        try:
            resp = requests.get(
                "https://lrclib.net/api/search",
                params={"q": q},
                timeout=5,
                verify=False,
            )
            if resp.status_code == 200:
                items = resp.json()
                if items and isinstance(items, list):
                    synced_raw = items[0].get("syncedLyrics") or ""
                    plain_raw = items[0].get("plainLyrics") or ""
                    synced_tuples = parse_lrc_timestamps(synced_raw) if synced_raw else []

                    plain = plain_raw or synced_raw
                    # Strip LRC timestamps [01:23.45]
                    plain = re.sub(r"\[\d{2}:\d{2}(?:\.\d{2,3})?\]", "", plain).strip()
                    if plain:
                        return plain, synced_tuples
        except Exception:
            continue
    return "", []


def analyze_youtube_with_gemini(
    youtube_url: str,
    lyrics_hint: Optional[str] = None,
    title_hint: Optional[str] = None,
    artist_hint: Optional[str] = None,
    tone_style: str = "marks",
    api_key: Optional[str] = None,
) -> SongScore:
    """
    Directly feeds a YouTube video URL into Gemini 3.8 Flash via the Google GenAI Interactions API,
    generating accurate, syllable-aligned ChordPro with chords placed directly over the singing.
    """
    effective_api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not effective_api_key:
        raise ValueError(
            "GEMINI_API_KEY is required to process YouTube videos directly with AI. "
            "Please provide an API key in the UI or set GEMINI_API_KEY in your environment."
        )

    from google import genai
    from pydantic import BaseModel, Field

    class GeminiSection(BaseModel):
        name: str = Field(description="Section name, e.g. Intro, Verse 1, Pre-Chorus, Chorus, Bridge, Outro")
        section_type: str = Field(description="verse, chorus, bridge, or intro")
        chordpro_lines: List[str] = Field(
            description=(
                "Lines in ChordPro format. Crucial: Insert [Chord] tags (e.g. [Eb], [Bb/D], [Cm7], [Abmaj7]) "
                "IMMEDIATELY before the exact sung syllable or Chinese character where the chord changes. "
                "Example: '[Eb]久未放[Bb/D]晴的天[Cm7]空  依旧[Cm7/Bb]留着你的笑[Abmaj7]容'"
            )
        )

    class GeminiYouTubeAnalysis(BaseModel):
        title: str = Field(description="Song title")
        artist: str = Field(description="Artist, singer, or worship team")
        key: str = Field(description="Concert Key (e.g. Eb, G, C, F, Bb, Am)")
        tempo_bpm: int = Field(description="Estimated Tempo in BPM")
        time_signature: str = Field(description="e.g. 4/4 or 3/4 or 6/8")
        sections: List[GeminiSection]

    client = genai.Client(api_key=effective_api_key)

    prompt = (
        "You are an expert guitar accompanist, church worship music director, and Mandopop music transcriber.\n"
        "Listen to this YouTube music performance carefully and transcribe the chords and lyrics in ChordPro format.\n\n"
        "Rules:\n"
        "1. Identify the exact Concert Key (e.g. C, G, D, Eb, F, Bb, Am) and Tempo (BPM).\n"
        "2. Transcribe every lyric line in Chinese characters (or English if English song).\n"
        "3. Chord Placement: Place chords in brackets like [C], [G/B], [Am7], [Fmaj7] at the PRECISE syllable "
        "where the guitar/piano chord changes on the beat. Do NOT clump chords at the start of the line.\n"
        "4. Include Intro, Verse, Pre-Chorus, Chorus, Bridge, and Outro sections.\n"
    )

    if title_hint:
        prompt += f"\nSong Title Hint: {title_hint}"
    if artist_hint:
        prompt += f"\nArtist Hint: {artist_hint}"
    if lyrics_hint and lyrics_hint.strip():
        prompt += f"\n\nAlign against these reference lyrics if applicable:\n{lyrics_hint.strip()}\n"

    # Download clean audio track first to ensure full reliability
    tmp_out = Path(tempfile.mkdtemp(prefix="yt_gemini_"))
    audio_path, meta = download_youtube_audio(youtube_url, output_dir=tmp_out)

    from google.genai import types

    uploaded_audio = client.files.upload(file=str(audio_path))

    # Generate Content with Gemini Multimodal Audio
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[uploaded_audio, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiYouTubeAnalysis,
        ),
    )

    data = GeminiYouTubeAnalysis.model_validate_json(response.text)

    score = SongScore(
        title=title_hint or meta.get("title") or data.title or "YouTube Song",
        artist=artist_hint or meta.get("artist") or data.artist or "",
        key=data.key or "C",
        tempo_bpm=data.tempo_bpm or 72,
        time_signature=data.time_signature or "4/4",
    )

    for sec in data.sections:
        sec_obj = SongSection(name=sec.name, section_type=sec.section_type)
        for line_str in sec.chordpro_lines:
            tokens = tokenize_chordpro_line(line_str, tone_style=tone_style)
            if tokens:
                sec_obj.lines.append(SongLine(tokens=tokens))
        if sec_obj.lines:
            score.sections.append(sec_obj)

    # Recommend capo
    capo_fret, _ = recommend_guitar_capo(score.get_unique_chords(), score.key)
    score.capo = capo_fret
    return score
