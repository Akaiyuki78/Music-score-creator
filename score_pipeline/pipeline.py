from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .models import SongScore
from .audio_analyzer import analyze_audio_to_score
from .musicxml_io import song_score_to_musicxml, musicxml_to_song_score
from .chordpro_io import song_score_to_chordpro, parse_chordpro_to_song_score
from .pdf_renderer import render_score_to_pdf
from .youtube_transcriber import (
    is_youtube_url,
    download_youtube_audio,
    analyze_youtube_with_gemini,
)


@dataclass
class PipelineOutput:
    """Holds all generated artifacts from the WAV -> MusicXML -> ChordPro -> PDF pipeline."""
    score: SongScore
    musicxml_path: Path
    chordpro_path: Path
    pdf_path: Path
    musicxml_content: str
    chordpro_content: str


def run_full_pipeline(
    audio_path: Union[str, Path],
    output_dir: Union[str, Path],
    lyrics_text: Optional[str] = None,
    title: Optional[str] = None,
    artist: str = "",
    key_override: Optional[str] = None,
    tempo_override: Optional[int] = None,
    capo_override: Optional[int] = None,
    transpose_semitones: int = 0,
    use_capo_shapes: bool = True,
    use_nashville: bool = False,
    tone_style: str = "marks",
    show_pinyin: bool = True,
    show_chord_diagrams: bool = True,
    separate_stems: bool = False,
    auto_transcribe_lyrics: bool = False,
    language: str = "zh",
    use_gemini: bool = False,
    gemini_api_key: Optional[str] = None,
) -> PipelineOutput:
    """
    Executes the complete 4-stage pipeline:
      1. WAV / Audio / YouTube URL -> `SongScore` (Chords, Key, Beat Grid, Melody, Lyrics + Hanyu Pinyin)
      2. `SongScore` -> MusicXML 4.0 (`.musicxml`)
      3. MusicXML / `SongScore` -> ChordPro (`.cho`)
      4. ChordPro / `SongScore` -> Guitarist & Church Musician PDF (`.pdf`)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_str = str(audio_path).strip()
    score_from_audio: Optional[SongScore] = None

    # Handle direct YouTube URL input
    if is_youtube_url(audio_str):
        # Try direct Gemini Multimodal YouTube analysis if Gemini enabled or API key available
        if use_gemini or gemini_api_key or os.environ.get("GEMINI_API_KEY"):
            try:
                score_from_audio = analyze_youtube_with_gemini(
                    youtube_url=audio_str,
                    lyrics_hint=lyrics_text,
                    title_hint=title,
                    artist_hint=artist,
                    tone_style=tone_style,
                    api_key=gemini_api_key,
                )
            except Exception as e:
                print(f"[Notice] Direct YouTube Gemini analysis notice: {e}. Falling back to yt-dlp audio download.")

        # Fallback: Download audio locally with yt-dlp and run local DSP pipeline
        if score_from_audio is None:
            downloaded_audio, meta = download_youtube_audio(audio_str, output_dir=output_dir)
            audio_path = downloaded_audio
            title = title or meta.get("title", "YouTube Song")
            artist = artist or meta.get("artist", "")
            if not lyrics_text and meta.get("subtitles_lyrics"):
                lyrics_text = meta.get("subtitles_lyrics")
            synced_lyrics = meta.get("synced_lyrics")
        else:
            synced_lyrics = None
    else:
        synced_lyrics = None

    if score_from_audio is None:
        audio_path = Path(audio_path)
        stem_name = audio_path.stem
        song_title = title or stem_name.replace("_", " ").replace("-", " ").title()

        # Stage 1: WAV -> SongScore
        score_from_audio = analyze_audio_to_score(
            audio_path=audio_path,
            lyrics_text=lyrics_text,
            synced_lyrics=synced_lyrics,
            title=song_title,
            artist=artist,
            key_override=key_override,
            tempo_override=tempo_override,
            capo_override=capo_override,
            separate_stems=separate_stems,
            auto_transcribe_lyrics=auto_transcribe_lyrics,
            language=language,
            tone_style=tone_style,
            use_gemini=use_gemini,
        )

    stem_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in score_from_audio.title)[:45] or "song"
    if key_override:
        score_from_audio.key = key_override
    if capo_override is not None:
        score_from_audio.capo = capo_override

    # Stage 2: SongScore -> MusicXML (.musicxml)
    musicxml_path = output_dir / f"{stem_name}.musicxml"
    musicxml_content = song_score_to_musicxml(score_from_audio, output_path=musicxml_path)

    # Stage 3: MusicXML -> ChordPro (.cho)
    score_for_chordpro = score_from_audio
    chordpro_path = output_dir / f"{stem_name}.cho"
    chordpro_content = song_score_to_chordpro(
        score_for_chordpro,
        output_path=chordpro_path,
        include_pinyin_comments=True,
        transpose_semitones=transpose_semitones,
        use_capo_shapes=False,
        use_nashville=False,
    )

    # Stage 4: ChordPro -> PDF (.pdf)
    pdf_path = output_dir / f"{stem_name}.pdf"
    render_score_to_pdf(
        score=score_for_chordpro,
        output_path=pdf_path,
        transpose_semitones=transpose_semitones,
        use_capo_shapes=use_capo_shapes,
        use_nashville=use_nashville,
        show_pinyin=show_pinyin,
        show_chord_diagrams=show_chord_diagrams,
    )

    return PipelineOutput(
        score=score_for_chordpro,
        musicxml_path=musicxml_path,
        chordpro_path=chordpro_path,
        pdf_path=pdf_path,
        musicxml_content=musicxml_content,
        chordpro_content=chordpro_content,
    )

    return PipelineOutput(
        score=score_for_chordpro,
        musicxml_path=musicxml_path,
        chordpro_path=chordpro_path,
        pdf_path=pdf_path,
        musicxml_content=musicxml_content,
        chordpro_content=chordpro_content,
    )
