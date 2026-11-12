#!/usr/bin/env python3
"""
CLI Entrypoint for the Music Score Creator Pipeline:
  WAV / Recording -> MusicXML -> ChordPro (with Hanyu Pinyin) -> Guitarist PDF
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from score_pipeline.pipeline import run_full_pipeline
from score_pipeline.audio_analyzer import analyze_audio_to_score
from score_pipeline.musicxml_io import song_score_to_musicxml, musicxml_to_song_score
from score_pipeline.chordpro_io import song_score_to_chordpro, parse_chordpro_to_song_score
from score_pipeline.pdf_renderer import render_score_to_pdf
from score_pipeline.sample_generator import (
    generate_demo_wav_and_lyrics,
    generate_ge_qian_wav_and_score,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="music-score-creator",
        description="Pipeline: WAV / Audio Recording -> MusicXML -> ChordPro (with Hanyu Pinyin) -> Guitarist PDF",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. `run` - Full pipeline: WAV -> MusicXML -> ChordPro -> PDF
    p_run = subparsers.add_parser(
        "run",
        help="Run the complete pipeline: WAV/MP3 -> MusicXML -> ChordPro -> PDF",
    )
    p_run.add_argument("audio", type=str, help="Path to input .wav/.mp3 file OR a YouTube video URL")
    p_run.add_argument("-l", "--lyrics", type=Path, help="Optional path to plain text or ChordPro lyrics file")
    p_run.add_argument("-o", "--outdir", type=Path, default=Path("output"), help="Output directory (default: ./output)")
    p_run.add_argument("--api-key", type=str, default=None, help="Gemini API key (overrides GEMINI_API_KEY env)")
    p_run.add_argument("--title", type=str, default=None, help="Song title")
    p_run.add_argument("--artist", type=str, default="", help="Artist / Composer / Worship team")
    p_run.add_argument("--key", type=str, default=None, help="Override Concert Key (e.g., G, C, F#, Eb)")
    p_run.add_argument("--tempo", type=int, default=None, help="Override Tempo BPM")
    p_run.add_argument("--capo", type=int, default=None, help="Override Guitar Capo fret (0-11)")
    p_run.add_argument("--transpose", type=int, default=0, help="Transpose semitones (e.g., +2 or -1)")
    p_run.add_argument("--nashville", action="store_true", help="Render chords using Nashville Number System (1, 4, 5, 6m)")
    p_run.add_argument("--no-capo-shapes", action="store_true", help="Show Concert Key chords instead of Capo play shapes")
    p_run.add_argument("--tone-style", choices=["marks", "numbers", "none"], default="marks", help="Hanyu Pinyin tone format")
    p_run.add_argument("--no-pinyin", action="store_true", help="Hide Hanyu Pinyin line in PDF")
    p_run.add_argument("--no-diagrams", action="store_true", help="Hide guitar chord diagrams in PDF")
    p_run.add_argument("--separate-stems", action="store_true", help="Run Demucs stem separation if installed")
    p_run.add_argument("--auto-transcribe", action="store_true", help="Auto-transcribe lyrics from audio with Faster-Whisper")
    p_run.add_argument("--gemini", action="store_true", help="Use Gemini 3.8 Flash API if GEMINI_API_KEY is set")

    # 2. `wav2xml` - Stage 1 & 2: WAV -> MusicXML
    p_w2x = subparsers.add_parser("wav2xml", help="Convert WAV/MP3 recording to MusicXML (.musicxml)")
    p_w2x.add_argument("audio", type=Path, help="Input audio file")
    p_w2x.add_argument("-o", "--output", type=Path, required=True, help="Output .musicxml path")
    p_w2x.add_argument("-l", "--lyrics", type=Path, help="Optional lyrics text file")
    p_w2x.add_argument("--title", type=str, default="Untitled Song", help="Song title")
    p_w2x.add_argument("--artist", type=str, default="", help="Artist")

    # 3. `xml2cho` - Stage 3: MusicXML -> ChordPro (with Pinyin)
    p_x2c = subparsers.add_parser("xml2cho", help="Convert MusicXML (.musicxml) to ChordPro (.cho)")
    p_x2c.add_argument("musicxml", type=Path, help="Input .musicxml file")
    p_x2c.add_argument("-o", "--output", type=Path, required=True, help="Output .cho file")
    p_x2c.add_argument("--tone-style", choices=["marks", "numbers", "none"], default="marks")

    # 4. `cho2pdf` - Stage 4: ChordPro -> Guitarist PDF (with Hanyu Pinyin & Fretboard Diagrams)
    p_c2p = subparsers.add_parser("cho2pdf", help="Render ChordPro (.cho) to Guitarist PDF with Hanyu Pinyin")
    p_c2p.add_argument("chordpro", type=Path, help="Input .cho file")
    p_c2p.add_argument("-o", "--output", type=Path, required=True, help="Output .pdf file")
    p_c2p.add_argument("--transpose", type=int, default=0, help="Transpose semitones")
    p_c2p.add_argument("--capo", type=int, default=None, help="Set Capo fret")
    p_c2p.add_argument("--nashville", action="store_true", help="Use Nashville Number System")
    p_c2p.add_argument("--tone-style", choices=["marks", "numbers", "none"], default="marks")
    p_c2p.add_argument("--no-pinyin", action="store_true", help="Disable Hanyu Pinyin row")
    p_c2p.add_argument("--no-diagrams", action="store_true", help="Disable Guitar Fretboard diagrams")

    # 5. `demo` - Generate sample WAV + Chinese Worship/Mandopop song and run full pipeline
    p_demo = subparsers.add_parser(
        "demo",
        help="Generate a sample acoustic guitar + vocal WAV and run the full pipeline into ./examples",
    )
    p_demo.add_argument("-o", "--outdir", type=Path, default=Path("examples"), help="Directory for demo outputs")
    p_demo.add_argument(
        "--song",
        choices=["ge-qian", "worship"],
        default="ge-qian",
        help="Which demo song to synthesize and run through the pipeline (default: ge-qian by Jay Chou)",
    )

    # 6. `web` - Launch interactive Web UI
    p_web = subparsers.add_parser("web", help="Launch the local interactive Web UI")
    p_web.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    p_web.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")

    args = parser.parse_args()

    if args.command == "run":
        lyrics_str = args.lyrics.read_text(encoding="utf-8") if args.lyrics else None
        res = run_full_pipeline(
            audio_path=args.audio,
            output_dir=args.outdir,
            lyrics_text=lyrics_str,
            title=args.title,
            artist=args.artist,
            key_override=args.key,
            tempo_override=args.tempo,
            capo_override=args.capo,
            transpose_semitones=args.transpose,
            use_capo_shapes=not args.no_capo_shapes,
            use_nashville=args.nashville,
            tone_style=args.tone_style,
            show_pinyin=not args.no_pinyin,
            show_chord_diagrams=not args.no_diagrams,
            separate_stems=args.separate_stems,
            auto_transcribe_lyrics=args.auto_transcribe,
            use_gemini=args.gemini or bool(args.api_key),
            gemini_api_key=args.api_key,
        )
        print(f"[OK] Detected Key: {res.score.key} | Suggested Capo: {res.score.capo} | Tempo: {res.score.tempo_bpm} BPM")
        print(f"[OK] MusicXML : {res.musicxml_path}")
        print(f"[OK] ChordPro : {res.chordpro_path}")
        print(f"[OK] PDF Chart: {res.pdf_path}")
        return 0

    if args.command == "wav2xml":
        lyrics_str = args.lyrics.read_text(encoding="utf-8") if args.lyrics else None
        score = analyze_audio_to_score(args.audio, lyrics_text=lyrics_str, title=args.title, artist=args.artist)
        song_score_to_musicxml(score, output_path=args.output)
        print(f"[OK] Saved MusicXML to {args.output}")
        return 0

    if args.command == "xml2cho":
        score = musicxml_to_song_score(args.musicxml, tone_style=args.tone_style)
        song_score_to_chordpro(score, output_path=args.output, include_pinyin_comments=True)
        print(f"[OK] Saved ChordPro to {args.output}")
        return 0

    if args.command == "cho2pdf":
        score = parse_chordpro_to_song_score(args.chordpro, tone_style=args.tone_style)
        if args.capo is not None:
            score.capo = args.capo
        render_score_to_pdf(
            score,
            output_path=args.output,
            transpose_semitones=args.transpose,
            use_capo_shapes=True,
            use_nashville=args.nashville,
            show_pinyin=not args.no_pinyin,
            show_chord_diagrams=not args.no_diagrams,
        )
        print(f"[OK] Saved PDF chart to {args.output}")
        return 0

    if args.command == "demo":
        outdir: Path = args.outdir
        outdir.mkdir(parents=True, exist_ok=True)
        if args.song == "ge-qian":
            wav_path, cho_preset = generate_ge_qian_wav_and_score(outdir / "jay_chou_ge_qian.wav")
            lyrics_file = outdir / "jay_chou_ge_qian_lyrics.cho"
            lyrics_file.write_text(cho_preset, encoding="utf-8")
            res = run_full_pipeline(
                audio_path=wav_path,
                output_dir=outdir,
                lyrics_text=cho_preset,
                title="搁浅 (Gē Qiǎn / Step Aside)",
                artist="周杰伦 (Jay Chou) | 词: 宋健彰 | 曲: 周杰伦",
                key_override="Eb",
                tempo_override=64,
                capo_override=3,
            )
        else:
            wav_path, lyrics_text = generate_demo_wav_and_lyrics(outdir / "still_before_you_demo.wav")
            lyrics_file = outdir / "still_before_you_lyrics.txt"
            lyrics_file.write_text(lyrics_text, encoding="utf-8")
            res = run_full_pipeline(
                audio_path=wav_path,
                output_dir=outdir,
                lyrics_text=lyrics_text,
                title="安静在祢面前 (Still Before You)",
                artist="Mandopop & Worship Guitar Demo",
            )
        print("=== Demo Pipeline Completed Successfully ===")
        print(f"1. Demo Audio WAV : {wav_path}")
        print(f"2. Demo Lyrics/CHO: {lyrics_file}")
        print(f"3. MusicXML Score : {res.musicxml_path}")
        print(f"4. ChordPro File  : {res.chordpro_path}")
        print(f"5. Guitar PDF     : {res.pdf_path}")
        return 0

    if args.command == "web":
        import uvicorn
        print(f"Starting Music Score Creator Web UI at http://{args.host}:{args.port} ...")
        uvicorn.run("web.app:app", host=args.host, port=args.port, reload=False)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
