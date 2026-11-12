# Music Score Creator Pipeline (WAV → MusicXML → ChordPro + Hanyu Pinyin → PDF)

A complete local pipeline and interactive Web UI built for guitarists, church worship musicians, and Mandopop / Chinese music enthusiasts.

## Pipeline Architecture

1. **Stage 1 — Audio Analysis (`score_pipeline/audio_analyzer.py`)**
   - Loads `.wav`, `.mp3`, `.m4a`, `.flac` recordings (solo acoustic guitar + vocal or full band).
   - Separates harmonic and vocal components (via `librosa` HPSS or `demucs` if installed).
   - Tracks tempo (BPM), beats, and 4/4 measure boundaries.
   - Detects Concert Key (Krumhansl-Schmuckler profiles) and beat-synchronized Guitar Chords (Chroma CQT + diatonic priors).
   - Extracts vocal/lead melody notes (`librosa.pyin`) for MusicXML lead sheet generation.
   - Aligns pasted Chinese/English lyrics or auto-transcribes lyrics locally using `faster-whisper` (with optional `gemini-3.8-flash` multimodal fallback when `GEMINI_API_KEY` is set).

2. **Stage 2 — MusicXML 4.0 Lead Sheet (`score_pipeline/musicxml_io.py`)**
   - Exports and imports standard `.musicxml` files compatible with MuseScore, Finale, Sibelius, Dorico, and Guitar Pro.
   - Embeds `<harmony>` chord symbols, `<note>` melody pitches, and dual `<lyric>` lines (Line 1: Hanyu Pinyin, Line 2: Chinese Hanzi).

3. **Stage 3 — Hanyu Pinyin & ChordPro Engine (`score_pipeline/pinyin_engine.py`, `score_pipeline/chordpro_io.py`, `score_pipeline/theory.py`)**
   - Uses `jieba` word segmentation + `pypinyin` with custom Mandopop & Chinese Worship polyphone overrides (`音乐` yīn yuè vs `喜乐` xǐ lè, `同行` tóng xíng, `长存` cháng cún vs `生长` shēng zhǎng, `祢` nǐ, `哈利路亚` hā lì lù yà).
   - Supports guitar transposition, automatic **Capo Fret recommendation** for open guitar shapes, and **Nashville Number System** (`1 4 5 6m`) for worship bands.

4. **Stage 4 — Guitarist & Worship PDF Renderer (`score_pipeline/pdf_renderer.py`)**
   - Renders print-ready A4 PDFs with CJK + Pinyin tone-mark font support (`Microsoft YaHei`).
   - Draws **6-string Guitar Fretboard Diagrams** at the top of the chart.
   - Renders each lyric line as a synchronized **3-tier stack**:
     1. **Guitar Chord** (`[G]`, `[D/F#]`, `[Em7]`, `[Cadd9]`)
     2. **Hanyu Pinyin** (`ān jìng xià lái...`)
     3. **Chinese Characters (Hanzi) / English Lyrics** (`安静下来...`)

---

## Quick Start

### 1. Launch the Interactive Web UI
```powershell
.\.venv\Scripts\python.exe cli.py web
```
Then open **http://127.0.0.1:8000** in your browser.

### 2. Run the Built-in Demo Pipeline (Generates Demo WAV + MusicXML + ChordPro + PDF)
```powershell
.\.venv\Scripts\python.exe cli.py demo -o examples
```

### 3. Run the Full Pipeline on a YouTube URL or Local WAV/MP3
```powershell
# From a YouTube URL directly (using Gemini 3.8 Flash for exact syllable chord placement):
.\.venv\Scripts\python.exe cli.py run "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --title "搁浅" --artist "周杰伦" --api-key YOUR_GEMINI_KEY -o output

# Or from a local WAV/MP3 recording:
.\.venv\Scripts\python.exe cli.py run path\to\song.wav --lyrics path\to\lyrics.txt --title "安静" --artist "Jay Chou" --capo 2 -o output
```

### 4. Run Individual Pipeline Stages
```powershell
# Step A: WAV -> MusicXML
.\.venv\Scripts\python.exe cli.py wav2xml examples\still_before_you_demo.wav -l examples\still_before_you_lyrics.txt -o output\song.musicxml

# Step B: MusicXML -> ChordPro (with Hanyu Pinyin)
.\.venv\Scripts\python.exe cli.py xml2cho output\song.musicxml -o output\song.cho

# Step C: ChordPro -> Guitarist PDF (with Fretboard Diagrams & Pinyin)
.\.venv\Scripts\python.exe cli.py cho2pdf output\song.cho --capo 2 -o output\song_capo2.pdf
```
