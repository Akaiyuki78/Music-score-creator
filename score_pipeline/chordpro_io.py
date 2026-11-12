from __future__ import annotations
import re
from pathlib import Path
from typing import List, Optional, Union

from .models import SongScore, SongSection, SongLine, LyricToken
from .pinyin_engine import tokenize_chordpro_line
from .theory import transpose_chord, transpose_key, chord_to_nashville, use_flats_for_key

DIRECTIVE_RE = re.compile(r"^\s*\{\s*([a-zA-Z0-9_]+)\s*(?::\s*(.*?))?\s*\}\s*$")
BRACKET_SECTION_RE = re.compile(
    r"^\s*\[(Intro|Verse\s*\d*|Chorus\s*\d*|Pre-Chorus|Bridge|Outro|Interlude|Instrumental|Solo|Tag|前奏|主歌\s*\d*|副歌\s*\d*|桥段|间奏|尾奏).*?\]\s*$",
    re.IGNORECASE,
)


def song_score_to_chordpro(
    score: SongScore,
    output_path: Optional[Union[str, Path]] = None,
    include_pinyin_comments: bool = False,
    transpose_semitones: int = 0,
    use_capo_shapes: bool = False,
    use_nashville: bool = False,
) -> str:
    """
    Formats a `SongScore` into a standard ChordPro (.cho) document.
    Supports transposing chords, converting to Capo play shapes, or Nashville numbers.
    """
    lines: List[str] = []

    effective_semi = transpose_semitones - (score.capo if use_capo_shapes else 0)
    concert_key = transpose_key(score.key, transpose_semitones)
    play_key = transpose_key(concert_key, -(score.capo if use_capo_shapes else 0))
    prefer_flats = use_flats_for_key(play_key)

    lines.append(f"{{title: {score.title}}}")
    if score.subtitle:
        lines.append(f"{{subtitle: {score.subtitle}}}")
    if score.artist:
        lines.append(f"{{artist: {score.artist}}}")
    lines.append(f"{{key: {concert_key}}}")
    if score.capo > 0:
        lines.append(f"{{capo: {score.capo}}}")
        if use_capo_shapes:
            lines.append(f"{{comment: Concert Key: {concert_key} | Play Shapes: {play_key} (Capo {score.capo})}}")
    lines.append(f"{{tempo: {score.tempo_bpm}}}")
    lines.append(f"{{time: {score.time_signature}}}")
    lines.append("")

    def _transform_chord(ch: Optional[str]) -> Optional[str]:
        if not ch:
            return None
        transposed = transpose_chord(ch, effective_semi, prefer_flats=prefer_flats)
        if use_nashville:
            return chord_to_nashville(transposed, play_key)
        return transposed

    for sec in score.sections:
        sec_type = sec.section_type.lower()
        if sec_type == "chorus":
            lines.append(f"{{start_of_chorus: {sec.name}}}")
        elif sec_type == "bridge":
            lines.append(f"{{start_of_bridge: {sec.name}}}")
        else:
            lines.append(f"{{start_of_verse: {sec.name}}}")

        for s_line in sec.lines:
            if s_line.is_comment:
                lines.append(f"{{comment: {s_line.comment_text}}}")
                continue

            if include_pinyin_comments:
                py_parts = [t.pinyin for t in s_line.tokens if t.pinyin]
                if py_parts:
                    lines.append(f"# Pinyin: {' '.join(py_parts)}")

            buf: List[str] = []
            for idx, tok in enumerate(s_line.tokens):
                c_str = _transform_chord(tok.chord)
                if c_str:
                    buf.append(f"[{c_str}]")
                if tok.text:
                    buf.append(tok.text)
                elif c_str and idx < len(s_line.tokens) - 1:
                    # Chord-only token (e.g. Intro/Interlude bar) - add spacing
                    buf.append("  ")
            lines.append("".join(buf).rstrip())

        if sec_type == "chorus":
            lines.append("{end_of_chorus}")
        elif sec_type == "bridge":
            lines.append("{end_of_bridge}")
        else:
            lines.append("{end_of_verse}")
        lines.append("")

    chordpro_text = "\n".join(lines).strip() + "\n"
    if output_path:
        Path(output_path).write_text(chordpro_text, encoding="utf-8")
    return chordpro_text


def parse_chordpro_to_song_score(
    chordpro_source: Union[str, Path],
    tone_style: str = "marks",
) -> SongScore:
    """
    Parses a ChordPro (.cho) file or text string into a `SongScore` object,
    automatically generating character-aligned Hanyu Pinyin for all Chinese lyrics.
    """
    if isinstance(chordpro_source, Path) or (
        isinstance(chordpro_source, str)
        and "\n" not in chordpro_source
        and Path(chordpro_source).exists()
    ):
        raw_text = Path(chordpro_source).read_text(encoding="utf-8")
    else:
        raw_text = str(chordpro_source)

    score = SongScore()
    current_section: Optional[SongSection] = None

    def _ensure_section(default_name: str = "Verse", default_type: str = "verse") -> SongSection:
        nonlocal current_section
        if current_section is None:
            current_section = SongSection(name=default_name, section_type=default_type)
            score.sections.append(current_section)
        return current_section

    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            # Skip raw comment lines (like "# Pinyin: ...")
            continue

        # Check ChordPro {directive: value}
        m_dir = DIRECTIVE_RE.match(stripped)
        if m_dir:
            d_name = m_dir.group(1).lower()
            d_val = (m_dir.group(2) or "").strip()

            if d_name in ("title", "t"):
                score.title = d_val or score.title
            elif d_name in ("subtitle", "st", "su"):
                score.subtitle = d_val
            elif d_name in ("artist", "a", "composer"):
                score.artist = d_val
            elif d_name == "key":
                score.key = d_val or "C"
            elif d_name == "capo":
                try:
                    score.capo = int(d_val)
                except ValueError:
                    pass
            elif d_name == "tempo":
                try:
                    score.tempo_bpm = int(float(d_val))
                except ValueError:
                    pass
            elif d_name == "time":
                score.time_signature = d_val or "4/4"
            elif d_name in ("start_of_chorus", "soc"):
                current_section = SongSection(name=d_val or "Chorus", section_type="chorus")
                score.sections.append(current_section)
            elif d_name in ("start_of_verse", "sov"):
                current_section = SongSection(name=d_val or "Verse", section_type="verse")
                score.sections.append(current_section)
            elif d_name in ("start_of_bridge", "sob"):
                current_section = SongSection(name=d_val or "Bridge", section_type="bridge")
                score.sections.append(current_section)
            elif d_name in ("end_of_chorus", "eoc", "end_of_verse", "eov", "end_of_bridge", "eob"):
                current_section = None
            elif d_name in ("comment", "c", "comment_italic", "ci"):
                # If the comment is actually a section label (e.g., {comment: Verse 1})
                low_val = d_val.lower()
                if any(
                    k in low_val
                    for k in ("intro", "verse", "chorus", "bridge", "outro", "interlude", "前奏", "主歌", "副歌", "桥段", "间奏", "尾奏")
                ) and len(d_val) < 28:
                    sec_type = "chorus" if ("chorus" in low_val or "副歌" in low_val) else (
                        "bridge" if ("bridge" in low_val or "桥段" in low_val) else "verse"
                    )
                    current_section = SongSection(name=d_val, section_type=sec_type)
                    score.sections.append(current_section)
                else:
                    sec = _ensure_section()
                    sec.lines.append(SongLine(is_comment=True, comment_text=d_val))
            continue

        # Check bracketed section header like [Verse 1] or [副歌]
        if BRACKET_SECTION_RE.match(stripped) and stripped.count("[") == 1:
            sec_label = stripped.strip("[] ")
            low_label = sec_label.lower()
            sec_type = "chorus" if ("chorus" in low_label or "副歌" in low_label) else (
                "bridge" if ("bridge" in low_label or "桥段" in low_label) else "verse"
            )
            current_section = SongSection(name=sec_label, section_type=sec_type)
            score.sections.append(current_section)
            continue

        # Normal lyric/chord line
        sec = _ensure_section()
        tokens = tokenize_chordpro_line(raw_line, tone_style=tone_style)
        if tokens:
            sec.lines.append(SongLine(tokens=tokens))

    return score
