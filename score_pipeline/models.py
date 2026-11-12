from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class NoteEvent:
    """Represents a musical note or rest in a measure (for MusicXML lead sheet)."""
    pitch_midi: Optional[int] = None  # None = rest
    step: str = "C"                   # C, D, E, F, G, A, B
    alter: int = 0                    # -1 = flat, 0 = natural, 1 = sharp
    octave: int = 4
    duration_beats: float = 1.0       # in quarter-note beats
    note_type: str = "quarter"        # whole, half, quarter, eighth, 16th
    lyric_hanzi: Optional[str] = None
    lyric_pinyin: Optional[str] = None
    chord_symbol: Optional[str] = None


@dataclass
class ChordEvent:
    """Represents a detected or annotated chord at a specific time/beat."""
    time_sec: float
    beat_index: int
    measure_index: int
    beat_in_measure: float
    symbol: str                       # e.g., "C", "G/B", "Am7", "Fmaj7"
    confidence: float = 1.0


@dataclass
class MeasureData:
    """Represents a single bar/measure in MusicXML."""
    number: int
    start_time_sec: float
    end_time_sec: float
    chords: List[ChordEvent] = field(default_factory=list)
    notes: List[NoteEvent] = field(default_factory=list)
    section_label: Optional[str] = None


@dataclass
class LyricToken:
    """
    A single aligned unit in a lyric line.
    For Chinese: typically 1 Hanzi character + its Pinyin syllable + optional chord above it.
    For English: a word or syllable + optional chord above it.
    For instrumental/chord-only tokens: text="", pinyin="", chord="G".
    """
    text: str = ""
    pinyin: str = ""
    chord: Optional[str] = None
    is_cjk: bool = False
    time_sec: Optional[float] = None


@dataclass
class SongLine:
    """A line of music containing aligned tokens (Chords + Pinyin + Lyrics) or comments."""
    tokens: List[LyricToken] = field(default_factory=list)
    is_comment: bool = False
    comment_text: str = ""
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None


@dataclass
class SongSection:
    """A section of a song (e.g., Intro, Verse 1, Chorus, Bridge, Outro)."""
    name: str = "Verse"
    section_type: str = "verse"       # verse, chorus, bridge, intro, outro, tab
    lines: List[SongLine] = field(default_factory=list)


@dataclass
class SongScore:
    """Complete unified representation across WAV analysis, MusicXML, ChordPro, and PDF."""
    title: str = "Untitled Song"
    subtitle: str = ""
    artist: str = ""
    key: str = "C"                    # Concert key, e.g., "C", "G", "F#", "Eb", "Am"
    capo: int = 0                     # Guitar capo fret (0-11)
    tempo_bpm: int = 120
    time_signature: str = "4/4"
    sections: List[SongSection] = field(default_factory=list)
    measures: List[MeasureData] = field(default_factory=list)
    detected_chords: List[ChordEvent] = field(default_factory=list)
    duration_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_unique_chords(self) -> List[str]:
        """Returns unique chord symbols in order of appearance."""
        seen = set()
        result = []
        for section in self.sections:
            for line in section.lines:
                for token in line.tokens:
                    if token.chord:
                        c = token.chord.strip()
                        if c and c not in seen:
                            seen.add(c)
                            result.append(c)
        if not result:
            for m in self.measures:
                for c in m.chords:
                    if c.symbol and c.symbol not in seen:
                        seen.add(c.symbol)
                        result.append(c.symbol)
        return result
