from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

# Pitch class constants
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

NOTE_TO_SEMITONE: Dict[str, int] = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

# Keys that conventionally use flats
FLAT_KEYS = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm"}

CHORD_PARSE_RE = re.compile(r"^([A-G][#b]?)(.*?)(?:/([A-G][#b]?))?$")

# Guitar-friendly open chord scores for Auto-Capo recommendation
GUITAR_FRIENDLY_CHORDS: Dict[str, float] = {
    "G": 3.0, "C": 3.0, "D": 3.0, "Em": 3.0, "Am": 3.0,
    "A": 2.5, "E": 2.5, "Dm": 2.5, "Fmaj7": 2.5, "Cadd9": 3.0,
    "G/B": 2.8, "D/F#": 2.8, "Am7": 3.0, "Em7": 3.0, "Dm7": 2.5,
    "G7": 2.5, "C7": 2.5, "D7": 2.8, "E7": 2.8, "A7": 2.8, "B7": 2.2,
    "Dsus4": 2.8, "Asus4": 2.8, "Asus2": 2.8, "Esus4": 2.5,
    "F": 1.2, "Bm": 1.4, "Bm7": 1.8, "F#m": 1.2, "C#m": 1.2,
}


def parse_chord(chord_symbol: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    Splits a chord symbol into (root, quality, bass).
    Example: 'F#m7/C#' -> ('F#', 'm7', 'C#')
    """
    if not chord_symbol:
        return None
    s = chord_symbol.strip()
    if s in ("N.C.", "NC", "-", "x", "%"):
        return None
    m = CHORD_PARSE_RE.match(s)
    if not m:
        return None
    return m.group(1), m.group(2) or "", m.group(3)


def use_flats_for_key(key_str: str) -> bool:
    """Determines whether a key signature should spell accidentals with flats."""
    k = key_str.strip()
    if k in FLAT_KEYS:
        return True
    if len(k) >= 2 and k[1] == "b":
        return True
    return False


def transpose_note(note: str, semitones: int, prefer_flats: bool = False) -> str:
    """Transposes a single note name by `semitones`."""
    if note not in NOTE_TO_SEMITONE:
        return note
    idx = (NOTE_TO_SEMITONE[note] + semitones) % 12
    return FLAT_NAMES[idx] if prefer_flats else SHARP_NAMES[idx]


def transpose_chord(chord_symbol: str, semitones: int, prefer_flats: Optional[bool] = None) -> str:
    """Transposes a chord symbol (or multiple chords separated by spaces, slashes, or dashes) by `semitones`."""
    if not chord_symbol:
        return ""
    if semitones % 12 == 0 and prefer_flats is None:
        return chord_symbol

    # If chord_symbol contains multiple chords separated by delimiter (e.g. "C G7" or "C - G7" or "C | G7" or "C, G7")
    delims = re.split(r"(\s*[-|,]\s*|\s+)", chord_symbol.strip())
    if len(delims) > 1:
        out_parts = []
        for part in delims:
            # Check if this subpart looks like a chord
            p_strip = part.strip()
            if p_strip and parse_chord(p_strip):
                out_parts.append(transpose_chord(p_strip, semitones, prefer_flats))
            else:
                out_parts.append(part)
        return "".join(out_parts)

    parsed = parse_chord(chord_symbol)
    if not parsed:
        return chord_symbol
    root, quality, bass = parsed
    if prefer_flats is None:
        prefer_flats = "b" in root or (bass is not None and "b" in bass)
    new_root = transpose_note(root, semitones, prefer_flats)
    new_bass = transpose_note(bass, semitones, prefer_flats) if bass else None
    if new_bass:
        return f"{new_root}{quality}/{new_bass}"
    return f"{new_root}{quality}"


def transpose_key(key_str: str, semitones: int) -> str:
    """Transposes a key string like 'F#' or 'Eb' or 'G#m' by semitones."""
    k = key_str.strip()
    is_minor = k.endswith("m")
    root = k[:-1] if is_minor else k
    if root not in NOTE_TO_SEMITONE:
        return key_str
    idx = (NOTE_TO_SEMITONE[root] + semitones) % 12
    # Choose cleaner enharmonic key spelling
    flat_preferred_indices = {1, 3, 5, 8, 10}  # Db, Eb, F, Ab, Bb
    if is_minor:
        flat_preferred_indices = {2, 5, 7, 10}  # Dm, Fm, Gm, Bbm
    name = FLAT_NAMES[idx] if idx in flat_preferred_indices else SHARP_NAMES[idx]
    return f"{name}m" if is_minor else name


def recommend_guitar_capo(chords: List[str], concert_key: str = "C") -> Tuple[int, str]:
    """
    Evaluates capo frets 0..7 and returns (best_capo_fret, play_key_shapes)
    that maximizes open guitar-friendly chords.
    """
    if not chords:
        return 0, concert_key

    best_fret = 0
    best_score = -999.0
    best_play_key = concert_key

    for fret in range(8):
        play_key = transpose_key(concert_key, -fret)
        prefer_flats = use_flats_for_key(play_key)
        score = 0.0
        for c in chords:
            shaped = transpose_chord(c, -fret, prefer_flats=prefer_flats)
            parsed = parse_chord(shaped)
            if not parsed:
                continue
            root, qual, _ = parsed
            base_chord = f"{root}{'m' if qual.startswith('m') and not qual.startswith('maj') else ''}"
            if shaped in GUITAR_FRIENDLY_CHORDS:
                score += GUITAR_FRIENDLY_CHORDS[shaped]
            elif base_chord in GUITAR_FRIENDLY_CHORDS:
                score += GUITAR_FRIENDLY_CHORDS[base_chord] * 0.85
            elif "#" in root or "b" in root:
                score -= 1.5
            else:
                score += 1.0
        # Slight preference for lower capo frets if tied
        score -= fret * 0.08
        if score > best_score:
            best_score = score
            best_fret = fret
            best_play_key = play_key

    return best_fret, best_play_key


def chord_to_nashville(chord_symbol: str, key_str: str = "C") -> str:
    """
    Converts a chord symbol into the Nashville Number System relative to the major tonic of `key_str`.
    Example in key of G: 'G' -> '1', 'C' -> '4', 'D/F#' -> '5/7', 'Em7' -> '6m7'
    """
    parsed = parse_chord(chord_symbol)
    if not parsed:
        return chord_symbol
    root, quality, bass = parsed

    k = key_str.strip()
    is_minor_key = k.endswith("m")
    key_root = k[:-1] if is_minor_key else k
    if key_root not in NOTE_TO_SEMITONE:
        return chord_symbol

    # Relative major tonic if minor key
    tonic_semi = (NOTE_TO_SEMITONE[key_root] + (3 if is_minor_key else 0)) % 12

    interval_map = {
        0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4",
        6: "#4", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7"
    }

    root_deg = interval_map[(NOTE_TO_SEMITONE[root] - tonic_semi) % 12]
    if bass and bass in NOTE_TO_SEMITONE:
        bass_deg = interval_map[(NOTE_TO_SEMITONE[bass] - tonic_semi) % 12]
        return f"{root_deg}{quality}/{bass_deg}"
    return f"{root_deg}{quality}"


# Standard Guitar Chord Fretboard Voicings (low E to high e: -1 = muted 'x', 0 = open 'o', >0 = fret)
GUITAR_CHORD_VOICINGS: Dict[str, List[int]] = {
    "C":     [-1, 3, 2, 0, 1, 0],
    "Cmaj7": [-1, 3, 2, 0, 0, 0],
    "Cadd9": [-1, 3, 2, 0, 3, 0],
    "C7":    [-1, 3, 2, 3, 1, 0],
    "Cm":    [-1, 3, 5, 5, 4, 3],
    "D":     [-1, -1, 0, 2, 3, 2],
    "D/F#":  [2, 0, 0, 2, 3, 2],
    "Dm":    [-1, -1, 0, 2, 3, 1],
    "Dm7":   [-1, -1, 0, 2, 1, 1],
    "Dmaj7": [-1, -1, 0, 2, 2, 2],
    "D7":    [-1, -1, 0, 2, 1, 2],
    "Dsus4": [-1, -1, 0, 2, 3, 3],
    "Dsus2": [-1, -1, 0, 2, 3, 0],
    "E":     [0, 2, 2, 1, 0, 0],
    "Em":    [0, 2, 2, 0, 0, 0],
    "Em7":   [0, 2, 2, 0, 3, 0],
    "E7":    [0, 2, 0, 1, 0, 0],
    "Esus4": [0, 2, 2, 2, 0, 0],
    "F":     [1, 3, 3, 2, 1, 1],
    "Fmaj7": [-1, -1, 3, 2, 1, 0],
    "Fm":    [1, 3, 3, 1, 1, 1],
    "F#m":   [2, 4, 4, 2, 2, 2],
    "F#m7":  [2, 4, 2, 2, 2, 2],
    "G":     [3, 2, 0, 0, 3, 3],
    "G/B":   [-1, 2, 0, 0, 3, 3],
    "G7":    [3, 2, 0, 0, 0, 1],
    "Gm":    [3, 5, 5, 3, 3, 3],
    "G#m":   [4, 6, 6, 4, 4, 4],
    "A":     [-1, 0, 2, 2, 2, 0],
    "A/C#":  [-1, 4, 2, 2, 2, 0],
    "Am":    [-1, 0, 2, 2, 1, 0],
    "Am/G":  [3, 0, 2, 2, 1, 0],
    "Am7":   [-1, 0, 2, 0, 1, 0],
    "Am7/G": [3, 0, 2, 0, 1, 0],
    "Gsus4": [3, 3, 0, 0, 1, 3],
    "Bm7/A": [-1, 0, 2, 0, 2, 2],
    "Amaj7": [-1, 0, 2, 1, 2, 0],
    "A7":    [-1, 0, 2, 0, 2, 0],
    "Asus2": [-1, 0, 2, 2, 0, 0],
    "Asus4": [-1, 0, 2, 2, 3, 0],
    "Bb":    [-1, 1, 3, 3, 3, 1],
    "Bbm":   [-1, 1, 3, 3, 2, 1],
    "B":     [-1, 2, 4, 4, 4, 2],
    "Bm":    [-1, 2, 4, 4, 3, 2],
    "Bm7":   [-1, 2, 0, 2, 0, 2],
    "B7":    [-1, 2, 1, 2, 0, 2],
    "C#m":   [-1, 4, 6, 6, 5, 4],
    "C#m7":  [-1, 4, 6, 4, 5, 4],
    "Eb":    [-1, 6, 8, 8, 8, 6],
    "Ab":    [4, 6, 6, 5, 4, 4],
}


def get_guitar_voicing(chord_symbol: str) -> List[int]:
    """Returns a 6-string guitar fret array [E, A, D, G, B, e] for `chord_symbol`."""
    if chord_symbol in GUITAR_CHORD_VOICINGS:
        return GUITAR_CHORD_VOICINGS[chord_symbol]
    parsed = parse_chord(chord_symbol)
    if not parsed:
        return [-1, 0, 0, 0, 0, 0]
    root, qual, _ = parsed
    # Try root + quality without slash bass
    simple = f"{root}{qual}"
    if simple in GUITAR_CHORD_VOICINGS:
        return GUITAR_CHORD_VOICINGS[simple]
    # Try major/minor triad fallback
    is_min = qual.startswith("m") and not qual.startswith("maj")
    triad = f"{root}{'m' if is_min else ''}"
    if triad in GUITAR_CHORD_VOICINGS:
        return GUITAR_CHORD_VOICINGS[triad]
    # Fallback: barre chord on E or A string
    semi = NOTE_TO_SEMITONE.get(root, 0)
    if semi <= 4:
        f = semi + 0
        return [f, f + 2, f + 2, f + (1 if not is_min else 0), f, f]
    f = semi - 9 if semi >= 9 else semi + 3
    return [-1, f, f + 2, f + 2, f + (2 if not is_min else 1), f]
