from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from score_pipeline.audio_analyzer import analyze_audio_to_score
from score_pipeline.chordpro_io import (
    song_score_to_chordpro,
    parse_chordpro_to_song_score,
)
from score_pipeline.musicxml_io import (
    song_score_to_musicxml,
    musicxml_to_song_score,
)
from score_pipeline.pdf_renderer import render_score_to_pdf, render_songbook_to_pdf
from score_pipeline.sample_generator import (
    DEMO_CHORDPRO_Preset,
    GE_QIAN_CHORDPRO_PRESET,
    generate_demo_wav_and_lyrics,
    generate_ge_qian_wav_and_score,
)
from score_pipeline.theory import (
    transpose_chord,
    transpose_key,
    chord_to_nashville,
    use_flats_for_key,
    recommend_guitar_capo,
    get_guitar_voicing,
)

app = FastAPI(
    title="Music Score Creator (WAV -> MusicXML -> ChordPro -> PDF)",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"
WORKSPACE_OUT_DIR = Path(__file__).parent.parent / "output"
WORKSPACE_OUT_DIR.mkdir(parents=True, exist_ok=True)


class RenderPreviewRequest(BaseModel):
    chordpro_text: str
    transpose_semitones: int = 0
    capo: Optional[int] = None
    use_capo_shapes: bool = True
    use_nashville: bool = False
    tone_style: str = "marks"
    show_pinyin: bool = True


class LoadChordTabRequest(BaseModel):
    tab_url: str
    tone_style: str = "marks"


def _serialize_score_for_ui(
    chordpro_text: str,
    transpose_semitones: int = 0,
    capo_override: Optional[int] = None,
    use_capo_shapes: bool = True,
    use_nashville: bool = False,
    tone_style: str = "marks",
) -> Dict[str, Any]:
    score = parse_chordpro_to_song_score(chordpro_text, tone_style=tone_style)
    if capo_override is not None:
        score.capo = capo_override

    concert_key = transpose_key(score.key, transpose_semitones)
    effective_capo = score.capo if use_capo_shapes else 0
    play_key = transpose_key(concert_key, -effective_capo)
    effective_semi = transpose_semitones - effective_capo
    prefer_flats = use_flats_for_key(play_key)

    rec_capo, rec_play_key = recommend_guitar_capo(score.get_unique_chords(), concert_key)

    def _transform_chord(ch: Optional[str]) -> Optional[str]:
        if not ch:
            return None
        transposed = transpose_chord(ch, effective_semi, prefer_flats=prefer_flats)
        if use_nashville:
            return chord_to_nashville(transposed, play_key)
        return transposed

    sections_out: List[Dict[str, Any]] = []
    unique_display_chords: List[str] = []

    for sec in score.sections:
        lines_out = []
        for ln in sec.lines:
            if ln.is_comment:
                lines_out.append({"is_comment": True, "comment_text": ln.comment_text, "tokens": []})
                continue
            toks_out = []
            for t in ln.tokens:
                disp_c = _transform_chord(t.chord)
                if disp_c and disp_c not in unique_display_chords and not use_nashville:
                    unique_display_chords.append(disp_c)
                toks_out.append(
                    {
                        "text": t.text,
                        "pinyin": t.pinyin,
                        "chord": disp_c,
                        "raw_chord": t.chord,
                        "is_cjk": t.is_cjk,
                    }
                )
            lines_out.append({"is_comment": False, "comment_text": "", "tokens": toks_out})
        sections_out.append(
            {
                "name": sec.name,
                "section_type": sec.section_type,
                "lines": lines_out,
            }
        )

    chord_diagrams = []
    for ch_sym in unique_display_chords:
        chord_diagrams.append(
            {
                "symbol": ch_sym,
                "frets": get_guitar_voicing(ch_sym),
            }
        )

    return {
        "title": score.title,
        "subtitle": score.subtitle,
        "artist": score.artist,
        "concert_key": concert_key,
        "play_key": play_key,
        "capo": score.capo,
        "recommended_capo": rec_capo,
        "recommended_play_key": rec_play_key,
        "tempo_bpm": score.tempo_bpm,
        "time_signature": score.time_signature,
        "sections": sections_out,
        "chord_diagrams": chord_diagrams,
    }


@app.get("/", response_class=HTMLResponse)
async def index_page() -> HTMLResponse:
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


@app.get("/api/demo")
async def get_demo_preset(preset: str = "ge_qian", tone_style: str = "marks") -> Dict[str, Any]:
    from score_pipeline.sample_generator import (
        GE_QIAN_CHORDPRO_PRESET,
        PENG_YOU_CHORDPRO_PRESET,
        SHOUT_TO_THE_LORD_CHORDPRO_PRESET,
        DEMO_CHORDPRO_Preset,
    )
    if preset == "shout":
        cho_text = SHOUT_TO_THE_LORD_CHORDPRO_PRESET
    elif preset == "worship" or preset == "peng_you":
        cho_text = PENG_YOU_CHORDPRO_PRESET
    elif preset == "still":
        cho_text = DEMO_CHORDPRO_Preset
    else:
        cho_text = GE_QIAN_CHORDPRO_PRESET

    preview = _serialize_score_for_ui(cho_text, tone_style=tone_style)
    return {
        "chordpro_text": cho_text,
        "preview": preview,
    }


@app.post("/api/preview")
async def render_preview(req: RenderPreviewRequest) -> Dict[str, Any]:
    preview = _serialize_score_for_ui(
        chordpro_text=req.chordpro_text,
        transpose_semitones=req.transpose_semitones,
        capo_override=req.capo,
        use_capo_shapes=req.use_capo_shapes,
        use_nashville=req.use_nashville,
        tone_style=req.tone_style,
    )
    return {"preview": preview}


@app.post("/api/analyze-audio")
async def analyze_uploaded_audio(
    audio_file: Optional[UploadFile] = File(None),
    youtube_url: str = Form(""),
    gemini_api_key: str = Form(""),
    use_demo_wav: bool = Form(False),
    lyrics_text: str = Form(""),
    title: str = Form(""),
    artist: str = Form(""),
    key_override: str = Form(""),
    tempo_override: Optional[int] = Form(None),
    separate_stems: bool = Form(False),
    auto_transcribe: bool = Form(False),
    use_gemini: bool = Form(False),
    tone_style: str = Form("marks"),
) -> Dict[str, Any]:
    from score_pipeline.youtube_transcriber import (
        is_youtube_url,
        analyze_youtube_with_gemini,
        download_youtube_audio,
    )

    clean_yt = youtube_url.strip()
    score = None

    try:
        # Option 2: YouTube URL workflow
        if clean_yt and is_youtube_url(clean_yt):
            eff_key = gemini_api_key.strip() or os.environ.get("GEMINI_API_KEY", "").strip()
            # Only use Gemini if user explicitly checked the Gemini box
            if use_gemini and eff_key:
                try:
                    score = analyze_youtube_with_gemini(
                        youtube_url=clean_yt,
                        lyrics_hint=lyrics_text if lyrics_text.strip() else None,
                        title_hint=title if title.strip() else None,
                        artist_hint=artist if artist.strip() else None,
                        tone_style=tone_style,
                        api_key=eff_key,
                    )
                except Exception as e:
                    print(f"[Notice] Direct YouTube Gemini error: {e}. Downloading audio fallback.")

            if score is None:
                tmp_dir = Path(tempfile.mkdtemp(prefix="yt_dl_"))
                dl_path, meta = download_youtube_audio(clean_yt, output_dir=tmp_dir)
                eff_lyrics = lyrics_text.strip() or meta.get("subtitles_lyrics", "").strip()
                synced_lrc = meta.get("synced_lyrics", [])
                score = analyze_audio_to_score(
                    audio_path=dl_path,
                    lyrics_text=eff_lyrics if eff_lyrics else None,
                    synced_lyrics=synced_lrc if synced_lrc else None,
                    title=title or meta.get("title", "YouTube Song"),
                    artist=artist or meta.get("artist", ""),
                    key_override=key_override.strip() if key_override.strip() else None,
                    tempo_override=tempo_override,
                    tone_style=tone_style,
                )

        if score is None:
            tmp_dir = Path(tempfile.mkdtemp(prefix="score_audio_"))
            if use_demo_wav or audio_file is None:
                wav_path, demo_lyrics = generate_demo_wav_and_lyrics(tmp_dir / "demo_worship.wav")
                if not lyrics_text.strip():
                    lyrics_text = demo_lyrics
                if not title.strip():
                    title = "安静在祢面前 (Still Before You)"
                if not artist.strip():
                    artist = "Mandopop & Worship Guitar Demo"
            else:
                suffix = Path(audio_file.filename or "recording.wav").suffix or ".wav"
                wav_path = tmp_dir / f"uploaded{suffix}"
                content = await audio_file.read()
                wav_path.write_bytes(content)
                if not title.strip() and audio_file.filename:
                    title = Path(audio_file.filename).stem.replace("_", " ").replace("-", " ").title()

            score = analyze_audio_to_score(
                audio_path=wav_path,
                lyrics_text=lyrics_text if lyrics_text.strip() else None,
                title=title or "Untitled Song",
                artist=artist,
                key_override=key_override.strip() if key_override.strip() else None,
                tempo_override=tempo_override if (tempo_override and tempo_override > 0) else None,
                separate_stems=separate_stems,
                auto_transcribe_lyrics=auto_transcribe,
                tone_style=tone_style,
                use_gemini=use_gemini or bool(gemini_api_key.strip()),
            )

        # Save MusicXML & ChordPro artifacts in workspace output folder
        safe_stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in score.title)[:40] or "song"
        xml_path = WORKSPACE_OUT_DIR / f"{safe_stem}.musicxml"
        cho_path = WORKSPACE_OUT_DIR / f"{safe_stem}.cho"
        musicxml_str = song_score_to_musicxml(score, output_path=xml_path)
        chordpro_str = song_score_to_chordpro(score, output_path=cho_path)

        preview = _serialize_score_for_ui(chordpro_str, capo_override=score.capo, tone_style=tone_style)
        return {
            "chordpro_text": chordpro_str,
            "musicxml_text": musicxml_str,
            "preview": preview,
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Audio processing error: {exc}")


@app.get("/api/scrape-youtube")
async def scrape_youtube_info(url: str) -> Dict[str, Any]:
    """
    Lightweight, fast YouTube scraper:
    Scrapes video title, artist/uploader, and online lyrics/subtitles without downloading audio.
    Also queries verified community chords for the song title so the user gets accurate chords in seconds!
    """
    import re
    import certifi
    import yt_dlp
    from score_pipeline.youtube_transcriber import fetch_lyrics_online
    from score_pipeline.community_chords import search_community_chords

    clean_url = url.strip()
    if not clean_url:
        raise HTTPException(status_code=400, detail="Missing YouTube URL")

    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "nocheckcertificate": True,
            "ca_certs": certifi.where(),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(clean_url, download=False)
            if "entries" in info and info["entries"]:
                info = info["entries"][0]
            title = info.get("title", "YouTube Song")
            uploader = info.get("uploader", "")

        # Clean query for search
        clean_q = re.sub(r"(Official\s*(Music\s*)?Video|Official\s*MV|MV|HD|4K|1080p|Audio|Lyrics|720p)", "", title, flags=re.I)
        clean_q = re.sub(r"[【】\[\]「」『』()|]", " ", clean_q)
        clean_q = re.sub(r"\s+", " ", clean_q).strip()

        # Fetch lyrics from lrclib
        plain_lrc, _ = fetch_lyrics_online(title, uploader)

        # Search community chord sheets
        chord_versions = search_community_chords(clean_q)

        return {
            "title": title,
            "artist": uploader,
            "search_query": clean_q,
            "lyrics": plain_lrc,
            "chord_versions": chord_versions,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to scrape YouTube link: {exc}")


@app.get("/api/chords/search")
async def search_chords(q: str = "") -> Dict[str, Any]:
    from score_pipeline.community_chords import search_community_chords
    if not q.strip():
        return {"query": "", "results": []}
    results = search_community_chords(q)
    return {"query": q, "results": results}


@app.post("/api/chords/load")
async def load_chord_tab(req: LoadChordTabRequest) -> Dict[str, Any]:
    from score_pipeline.community_chords import fetch_community_tab_as_chordpro
    tab_data = fetch_community_tab_as_chordpro(req.tab_url)
    if not tab_data:
        raise HTTPException(status_code=404, detail="Failed to fetch chord tab.")

    cho_text = tab_data["chordpro_text"]
    capo_val = tab_data["capo"] if tab_data["capo"] > 0 else None
    preview = _serialize_score_for_ui(
        chordpro_text=cho_text,
        capo_override=capo_val,
        tone_style=req.tone_style,
    )
    score = parse_chordpro_to_song_score(cho_text, tone_style=req.tone_style)
    if capo_val is not None:
        score.capo = capo_val
    xml_str = song_score_to_musicxml(score)

    return {
        "title": tab_data["title"],
        "artist": tab_data["artist"],
        "key": tab_data["key"],
        "capo": tab_data["capo"],
        "chordpro_text": cho_text,
        "musicxml_text": xml_str,
        "preview": preview,
    }


@app.post("/api/import-musicxml")
async def import_musicxml_file(
    musicxml_file: UploadFile = File(...),
    tone_style: str = Form("marks"),
) -> Dict[str, Any]:
    raw_bytes = await musicxml_file.read()
    try:
        xml_str = raw_bytes.decode("utf-8")
        score = musicxml_to_song_score(xml_str, tone_style=tone_style)
        chordpro_str = song_score_to_chordpro(score)
        preview = _serialize_score_for_ui(chordpro_str, capo_override=score.capo, tone_style=tone_style)
        return {
            "chordpro_text": chordpro_str,
            "preview": preview,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid MusicXML file: {exc}")


@app.post("/api/export/pdf")
async def export_pdf(req: RenderPreviewRequest) -> FileResponse:
    score = parse_chordpro_to_song_score(req.chordpro_text, tone_style=req.tone_style)
    if req.capo is not None:
        score.capo = req.capo

    tmp_pdf = WORKSPACE_OUT_DIR / "exported_score.pdf"
    render_score_to_pdf(
        score=score,
        output_path=tmp_pdf,
        transpose_semitones=req.transpose_semitones,
        use_capo_shapes=req.use_capo_shapes,
        use_nashville=req.use_nashville,
        show_pinyin=req.show_pinyin,
        show_chord_diagrams=True,
    )
    filename = f"{score.title or 'guitar_score'}.pdf"
    return FileResponse(
        path=str(tmp_pdf),
        media_type="application/pdf",
        filename=filename,
    )


class ExportSongbookRequest(BaseModel):
    title: str = "Sunday Worship Setlist"
    songs: List[RenderPreviewRequest]


@app.post("/api/export/songbook-pdf")
async def export_songbook_pdf(req: ExportSongbookRequest) -> FileResponse:
    if not req.songs:
        raise HTTPException(status_code=400, detail="No songs in setlist to export")

    scores_with_settings = []
    for s_req in req.songs:
        score = parse_chordpro_to_song_score(s_req.chordpro_text, tone_style=s_req.tone_style)
        if s_req.capo is not None:
            score.capo = s_req.capo
        scores_with_settings.append(
            (
                score,
                s_req.transpose_semitones,
                s_req.use_capo_shapes,
                s_req.use_nashville,
                s_req.show_pinyin,
                True,  # diagrams
            )
        )

    tmp_pdf = WORKSPACE_OUT_DIR / "worship_setlist_songbook.pdf"
    render_songbook_to_pdf(
        scores_with_settings=scores_with_settings,
        output_path=tmp_pdf,
        setlist_title=req.title,
    )
    return FileResponse(
        path=str(tmp_pdf),
        media_type="application/pdf",
        filename=f"{req.title.replace(' ', '_')}.pdf",
    )


@app.post("/api/export/musicxml")
async def export_musicxml(req: RenderPreviewRequest) -> Response:
    score = parse_chordpro_to_song_score(req.chordpro_text, tone_style=req.tone_style)
    if req.capo is not None:
        score.capo = req.capo
    xml_str = song_score_to_musicxml(score)
    return Response(
        content=xml_str.encode("utf-8"),
        media_type="application/vnd.recordare.musicxml+xml",
        headers={"Content-Disposition": 'attachment; filename="score.musicxml"'},
    )


@app.post("/api/export/chordpro")
async def export_chordpro(req: RenderPreviewRequest) -> Response:
    score = parse_chordpro_to_song_score(req.chordpro_text, tone_style=req.tone_style)
    if req.capo is not None:
        score.capo = req.capo
    cho_str = song_score_to_chordpro(
        score=score,
        include_pinyin_comments=True,
        transpose_semitones=req.transpose_semitones,
        use_capo_shapes=req.use_capo_shapes,
        use_nashville=req.use_nashville,
    )
    return Response(
        content=cho_str.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="score.cho"'},
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
