from __future__ import annotations
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .models import (
    SongScore,
    SongSection,
    SongLine,
    LyricToken,
    MeasureData,
    NoteEvent,
    ChordEvent,
)
from .theory import parse_chord, NOTE_TO_SEMITONE, SHARP_NAMES
from .pinyin_engine import tokenize_chordpro_line, is_cjk_char, char_list_to_pinyin

# Key signature fifths mapping
KEY_TO_FIFTHS = {
    "C": 0, "Am": 0,
    "G": 1, "Em": 1,
    "D": 2, "Bm": 2,
    "A": 3, "F#m": 3,
    "E": 4, "C#m": 4,
    "B": 5, "G#m": 5,
    "F#": 6, "D#m": 6,
    "Gb": -6, "Ebm": -6,
    "Db": -5, "Bbm": -5,
    "Ab": -4, "Fm": -4,
    "Eb": -3, "Cm": -3,
    "Bb": -2, "Gm": -2,
    "F": -1, "Dm": -1,
}

FIFTHS_TO_KEY = {
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#",
    -1: "F", -2: "Bb", -3: "Eb", -4: "Ab", -5: "Db", -6: "Gb",
}

QUALITY_TO_MUSICXML_KIND = {
    "": ("major", ""),
    "m": ("minor", "m"),
    "min": ("minor", "m"),
    "7": ("dominant", "7"),
    "maj7": ("major-seventh", "maj7"),
    "M7": ("major-seventh", "maj7"),
    "m7": ("minor-seventh", "m7"),
    "min7": ("minor-seventh", "m7"),
    "dim": ("diminished", "dim"),
    "dim7": ("diminished-seventh", "dim7"),
    "m7b5": ("half-diminished", "m7b5"),
    "aug": ("augmented", "aug"),
    "+": ("augmented", "+"),
    "sus4": ("suspended-fourth", "sus4"),
    "sus2": ("suspended-second", "sus2"),
    "add9": ("major", "add9"),
    "m9": ("minor-ninth", "m9"),
    "9": ("dominant-ninth", "9"),
    "6": ("major-sixth", "6"),
    "m6": ("minor-sixth", "m6"),
}

MUSICXML_KIND_TO_QUALITY = {v[0]: k for k, v in QUALITY_TO_MUSICXML_KIND.items() if k not in ("min", "M7", "min7", "+")}


def midi_to_step_alter_octave(midi_num: int) -> Tuple[str, int, int]:
    """Converts a MIDI pitch number (e.g. 60 = C4) into (step, alter, octave)."""
    octave = (midi_num // 12) - 1
    semi = midi_num % 12
    name = SHARP_NAMES[semi]
    if len(name) == 2:
        return name[0], 1, octave
    return name, 0, octave


def _note_str_to_step_alter(note_str: str) -> Tuple[str, int]:
    step = note_str[0].upper()
    alter = 0
    if len(note_str) > 1:
        if note_str[1] == "#":
            alter = 1
        elif note_str[1] == "b":
            alter = -1
    return step, alter


def _step_alter_to_note_str(step: str, alter: int) -> str:
    if alter == 1:
        return f"{step}#"
    if alter == -1:
        return f"{step}b"
    return step


def _ensure_measures_populated(score: SongScore) -> List[MeasureData]:
    """
    Returns populated measures for MusicXML export:
      - If `score.measures` already contains authentic audio-extracted measures
        (with chords, notes, timing), uses them directly as the primary ground truth.
      - Otherwise, constructs clean 4/4 measures from `score.sections` (e.g. from imported ChordPro).
    """
    if score.measures and any(m.chords or m.notes for m in score.measures):
        return score.measures

    # Collect all non-comment lyric lines with their section labels
    line_entries: List[Tuple[Optional[str], List[LyricToken]]] = []
    for sec in score.sections:
        first_in_sec = True
        for line in sec.lines:
            if line.is_comment:
                continue
            meaningful = [t for t in line.tokens if t.text.strip() or t.chord]
            if meaningful:
                line_entries.append((sec.name if first_in_sec else None, meaningful))
                first_in_sec = False

    if not line_entries and score.measures:
        return score.measures

    measures: List[MeasureData] = []
    measure_num = 1
    sec_per_measure = (60.0 / max(score.tempo_bpm, 40)) * 4.0
    src_measures = score.measures or []

    for sec_label, tokens in line_entries:
        # Split each lyric line across 2 measures (first half of tokens in bar 1, second half in bar 2)
        mid = max(1, (len(tokens) + 1) // 2)
        bar_token_groups = [tokens[:mid], tokens[mid:]] if len(tokens) > 4 else [tokens]

        for g_idx, grp in enumerate(bar_token_groups):
            if not grp:
                continue
            src_m = src_measures[(measure_num - 1) % len(src_measures)] if src_measures else None
            m_start = (measure_num - 1) * sec_per_measure
            m = MeasureData(
                number=measure_num,
                start_time_sec=m_start,
                end_time_sec=m_start + sec_per_measure,
                section_label=sec_label if g_idx == 0 else None,
            )

            beat_dur = 4.0 / len(grp)
            if beat_dur >= 3.5:
                note_type = "whole"
            elif beat_dur >= 1.75:
                note_type = "half"
            elif beat_dur >= 0.85:
                note_type = "quarter"
            else:
                note_type = "eighth"

            for idx, tok in enumerate(grp):
                beat_pos = idx * beat_dur
                chord_sym = tok.chord
                if not chord_sym and idx == 0 and src_m and src_m.chords:
                    chord_sym = src_m.chords[0].symbol

                if chord_sym:
                    m.chords.append(
                        ChordEvent(
                            time_sec=m_start + beat_pos * (60.0 / max(score.tempo_bpm, 40)),
                            beat_index=(measure_num - 1) * 4 + int(beat_pos),
                            measure_index=measure_num - 1,
                            beat_in_measure=beat_pos,
                            symbol=chord_sym,
                        )
                    )

                # Choose pitch from source audio measure if available, else chord root
                if src_m and src_m.notes:
                    src_note = src_m.notes[min(int(beat_pos), len(src_m.notes) - 1)]
                    midi_pitch = src_note.pitch_midi or 60
                    step, alter, octave = src_note.step, src_note.alter, src_note.octave
                else:
                    active_chord = chord_sym or (m.chords[-1].symbol if m.chords else score.key)
                    parsed = parse_chord(active_chord)
                    root_name = parsed[0] if parsed else "C"
                    midi_pitch = 60 + NOTE_TO_SEMITONE.get(root_name, 0)
                    step, alter, octave = midi_to_step_alter_octave(midi_pitch)

                m.notes.append(
                    NoteEvent(
                        pitch_midi=midi_pitch,
                        step=step,
                        alter=alter,
                        octave=octave,
                        duration_beats=beat_dur,
                        note_type=note_type,
                        lyric_hanzi=tok.text.strip() if tok.text.strip() else None,
                        lyric_pinyin=tok.pinyin.strip() if tok.pinyin else None,
                        chord_symbol=chord_sym,
                    )
                )

            measures.append(m)
            measure_num += 1

    return measures


def song_score_to_musicxml(score: SongScore, output_path: Optional[Union[str, Path]] = None) -> str:
    """
    Exports a `SongScore` to a standard MusicXML 4.0 string (and writes to `output_path` if given).
    Includes:
      - Key, Time Signature, Tempo, Capo annotation
      - Harmony elements (<harmony>) for guitar chords
      - Melody notes (<note>)
      - Dual lyric lines (<lyric number="1"> for Hanyu Pinyin, <lyric number="2"> for Chinese Hanzi / Lyrics)
    """
    measures = _ensure_measures_populated(score)

    root = ET.Element("score-partwise", version="4.0")

    # Work title & subtitle
    work_el = ET.SubElement(root, "work")
    ET.SubElement(work_el, "work-title").text = score.title
    if score.subtitle or score.capo:
        sub_parts = []
        if score.subtitle:
            sub_parts.append(score.subtitle)
        if score.capo > 0:
            sub_parts.append(f"Capo {score.capo}")
        ET.SubElement(work_el, "work-number").text = " | ".join(sub_parts)

    # Identification
    ident_el = ET.SubElement(root, "identification")
    if score.artist:
        creator_el = ET.SubElement(ident_el, "creator", type="composer")
        creator_el.text = score.artist
    encoding_el = ET.SubElement(ident_el, "encoding")
    ET.SubElement(encoding_el, "software").text = "Music-Score-Creator Pipeline (WAV -> MusicXML -> ChordPro -> PDF)"

    # Part list
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Vocal / Guitar Lead Sheet"

    part_el = ET.SubElement(root, "part", id="P1")
    divisions = 4  # 4 divisions per quarter note (supports 16th notes)

    num_beats, beat_type = 4, 4
    if "/" in score.time_signature:
        try:
            parts = score.time_signature.split("/")
            num_beats, beat_type = int(parts[0]), int(parts[1])
        except ValueError:
            num_beats, beat_type = 4, 4

    for m_idx, m in enumerate(measures):
        m_el = ET.SubElement(part_el, "measure", number=str(m.number))

        if m_idx == 0:
            attr_el = ET.SubElement(m_el, "attributes")
            ET.SubElement(attr_el, "divisions").text = str(divisions)

            key_el = ET.SubElement(attr_el, "key")
            fifths = KEY_TO_FIFTHS.get(score.key, 0)
            ET.SubElement(key_el, "fifths").text = str(fifths)
            ET.SubElement(key_el, "mode").text = "minor" if score.key.endswith("m") else "major"

            time_el = ET.SubElement(attr_el, "time")
            ET.SubElement(time_el, "beats").text = str(num_beats)
            ET.SubElement(time_el, "beat-type").text = str(beat_type)

            clef_el = ET.SubElement(attr_el, "clef")
            ET.SubElement(clef_el, "sign").text = "G"
            ET.SubElement(clef_el, "line").text = "2"

            # Tempo direction
            dir_el = ET.SubElement(m_el, "direction", placement="above")
            dir_type = ET.SubElement(dir_el, "direction-type")
            metro_el = ET.SubElement(dir_type, "metronome")
            ET.SubElement(metro_el, "beat-unit").text = "quarter"
            ET.SubElement(metro_el, "per-minute").text = str(score.tempo_bpm)
            ET.SubElement(dir_el, "sound", tempo=str(score.tempo_bpm))

        # Section rehearsal mark if present
        if m.section_label:
            dir_el = ET.SubElement(m_el, "direction", placement="above")
            dir_type = ET.SubElement(dir_el, "direction-type")
            reh_el = ET.SubElement(dir_type, "rehearsal")
            reh_el.text = m.section_label

        # Map chords by beat position so we emit <harmony> right before the corresponding <note>
        chords_by_note_idx = {}
        notes_list = m.notes if m.notes else [
            NoteEvent(pitch_midi=None, duration_beats=float(num_beats), note_type="whole")
        ]
        for ch_ev in m.chords:
            # Find closest note index in this measure
            target_idx = min(int(ch_ev.beat_in_measure), max(0, len(notes_list) - 1))
            chords_by_note_idx.setdefault(target_idx, []).append(ch_ev)

        for n_idx, n in enumerate(notes_list):
            # Emit any harmony elements at this note position
            for ch_ev in chords_by_note_idx.get(n_idx, []):
                parsed = parse_chord(ch_ev.symbol)
                if parsed:
                    root_str, qual_str, bass_str = parsed
                    harm_el = ET.SubElement(m_el, "harmony")
                    root_node = ET.SubElement(harm_el, "root")
                    r_step, r_alter = _note_str_to_step_alter(root_str)
                    ET.SubElement(root_node, "root-step").text = r_step
                    if r_alter != 0:
                        ET.SubElement(root_node, "root-alter").text = str(r_alter)

                    kind_val, kind_text = QUALITY_TO_MUSICXML_KIND.get(qual_str, ("other", qual_str))
                    kind_el = ET.SubElement(harm_el, "kind", text=kind_text)
                    kind_el.text = kind_val

                    if bass_str:
                        bass_node = ET.SubElement(harm_el, "bass")
                        b_step, b_alter = _note_str_to_step_alter(bass_str)
                        ET.SubElement(bass_node, "bass-step").text = b_step
                        if b_alter != 0:
                            ET.SubElement(bass_node, "bass-alter").text = str(b_alter)

            note_el = ET.SubElement(m_el, "note")
            if n.pitch_midi is None:
                ET.SubElement(note_el, "rest")
            else:
                pitch_el = ET.SubElement(note_el, "pitch")
                ET.SubElement(pitch_el, "step").text = n.step
                if n.alter != 0:
                    ET.SubElement(pitch_el, "alter").text = str(n.alter)
                ET.SubElement(pitch_el, "octave").text = str(n.octave)

            dur_int = max(1, int(round(n.duration_beats * divisions)))
            ET.SubElement(note_el, "duration").text = str(dur_int)
            ET.SubElement(note_el, "type").text = n.note_type

            # Attach lyrics: Line 1 = Pinyin (if present), Line 2 = Hanzi / original text
            if n.lyric_pinyin:
                lyr1 = ET.SubElement(note_el, "lyric", number="1")
                ET.SubElement(lyr1, "syllabic").text = "single"
                ET.SubElement(lyr1, "text").text = n.lyric_pinyin
                if n.lyric_hanzi:
                    lyr2 = ET.SubElement(note_el, "lyric", number="2")
                    ET.SubElement(lyr2, "syllabic").text = "single"
                    ET.SubElement(lyr2, "text").text = n.lyric_hanzi
            elif n.lyric_hanzi:
                lyr1 = ET.SubElement(note_el, "lyric", number="1")
                ET.SubElement(lyr1, "syllabic").text = "single"
                ET.SubElement(lyr1, "text").text = n.lyric_hanzi

    raw_xml = ET.tostring(root, encoding="utf-8")
    parsed_dom = minidom.parseString(raw_xml)
    pretty_xml = parsed_dom.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

    if output_path:
        Path(output_path).write_text(pretty_xml, encoding="utf-8")

    return pretty_xml


def musicxml_to_song_score(xml_source: Union[str, Path], tone_style: str = "marks") -> SongScore:
    """
    Parses a MusicXML file or XML string and converts it into a `SongScore`
    ready for ChordPro and PDF export (with full Hanyu Pinyin enrichment).
    """
    if isinstance(xml_source, Path) or (
        isinstance(xml_source, str) and not xml_source.lstrip().startswith("<") and Path(xml_source).exists()
    ):
        xml_text = Path(xml_source).read_text(encoding="utf-8")
    else:
        xml_text = str(xml_source)

    root = ET.fromstring(xml_text)
    score = SongScore()

    # Extract title & artist
    wt = root.find(".//work/work-title")
    if wt is not None and wt.text:
        score.title = wt.text.strip()
    else:
        mt = root.find(".//movement-title")
        if mt is not None and mt.text:
            score.title = mt.text.strip()

    wn = root.find(".//work/work-number")
    if wn is not None and wn.text:
        score.subtitle = wn.text.strip()
        if "Capo " in score.subtitle:
            try:
                capo_part = score.subtitle.split("Capo ")[1].split()[0]
                score.capo = int(capo_part)
            except (IndexError, ValueError):
                pass

    creator = root.find(".//identification/creator[@type='composer']")
    if creator is not None and creator.text:
        score.artist = creator.text.strip()

    # Key, Time, Tempo
    fifths_el = root.find(".//attributes/key/fifths")
    mode_el = root.find(".//attributes/key/mode")
    if fifths_el is not None and fifths_el.text:
        try:
            fifths = int(fifths_el.text.strip())
            base_key = FIFTHS_TO_KEY.get(fifths, "C")
            if mode_el is not None and mode_el.text and mode_el.text.strip() == "minor":
                # Convert major fifths to relative minor
                from .theory import transpose_key
                score.key = transpose_key(base_key, -3) + "m"
            else:
                score.key = base_key
        except ValueError:
            pass

    beats_el = root.find(".//attributes/time/beats")
    beat_type_el = root.find(".//attributes/time/beat-type")
    if beats_el is not None and beat_type_el is not None and beats_el.text and beat_type_el.text:
        score.time_signature = f"{beats_el.text.strip()}/{beat_type_el.text.strip()}"

    sound_el = root.find(".//direction/sound[@tempo]")
    if sound_el is not None and sound_el.get("tempo"):
        try:
            score.tempo_bpm = int(float(sound_el.get("tempo", "120")))
        except ValueError:
            pass

    # Iterate through measures in first part
    part = root.find(".//part")
    if part is None:
        return score

    divisions = 4
    current_section = SongSection(name="Verse 1", section_type="verse")
    score.sections.append(current_section)
    current_line_tokens: List[LyricToken] = []
    measures_in_current_line = 0

    for m_idx, m_el in enumerate(part.findall("measure")):
        m_num = m_idx + 1
        div_el = m_el.find("attributes/divisions")
        if div_el is not None and div_el.text:
            try:
                divisions = max(1, int(div_el.text.strip()))
            except ValueError:
                pass

        # Check rehearsal mark for section boundaries
        reh_el = m_el.find(".//direction-type/rehearsal")
        if reh_el is not None and reh_el.text:
            sec_title = reh_el.text.strip()
            if current_line_tokens:
                current_section.lines.append(SongLine(tokens=current_line_tokens))
                current_line_tokens = []
                measures_in_current_line = 0
            if current_section.lines:
                sec_type = "chorus" if "chorus" in sec_title.lower() or "副歌" in sec_title else "verse"
                current_section = SongSection(name=sec_title, section_type=sec_type)
                score.sections.append(current_section)
            else:
                current_section.name = sec_title

        m_data = MeasureData(number=m_num, start_time_sec=0.0, end_time_sec=0.0, section_label=reh_el.text.strip() if (reh_el is not None and reh_el.text) else None)
        pending_chord: Optional[str] = None
        beat_cursor = 0.0

        for child in m_el:
            if child.tag == "harmony":
                r_step_el = child.find("root/root-step")
                r_alter_el = child.find("root/root-alter")
                kind_el = child.find("kind")
                b_step_el = child.find("bass/bass-step")
                b_alter_el = child.find("bass/bass-alter")
                if r_step_el is not None and r_step_el.text:
                    r_alter = int(r_alter_el.text.strip()) if (r_alter_el is not None and r_alter_el.text) else 0
                    root_str = _step_alter_to_note_str(r_step_el.text.strip(), r_alter)
                    qual_str = ""
                    if kind_el is not None:
                        if kind_el.get("text") is not None:
                            qual_str = kind_el.get("text", "")
                        elif kind_el.text:
                            qual_str = MUSICXML_KIND_TO_QUALITY.get(kind_el.text.strip(), "")
                    bass_str = ""
                    if b_step_el is not None and b_step_el.text:
                        b_alter = int(b_alter_el.text.strip()) if (b_alter_el is not None and b_alter_el.text) else 0
                        bass_str = "/" + _step_alter_to_note_str(b_step_el.text.strip(), b_alter)
                    chord_sym = f"{root_str}{qual_str}{bass_str}"
                    pending_chord = chord_sym
                    m_data.chords.append(
                        ChordEvent(
                            time_sec=0.0,
                            beat_index=int(beat_cursor),
                            measure_index=m_idx,
                            beat_in_measure=beat_cursor,
                            symbol=chord_sym,
                        )
                    )
            elif child.tag == "note":
                dur_el = child.find("duration")
                dur_beats = float(dur_el.text.strip()) / divisions if (dur_el is not None and dur_el.text) else 1.0
                type_el = child.find("type")
                note_type = type_el.text.strip() if (type_el is not None and type_el.text) else "quarter"

                is_rest = child.find("rest") is not None
                step, alter, octave, midi_num = "C", 0, 4, None
                if not is_rest:
                    st_el = child.find("pitch/step")
                    al_el = child.find("pitch/alter")
                    oc_el = child.find("pitch/octave")
                    if st_el is not None and st_el.text:
                        step = st_el.text.strip()
                    if al_el is not None and al_el.text:
                        alter = int(al_el.text.strip())
                    if oc_el is not None and oc_el.text:
                        octave = int(oc_el.text.strip())
                    note_name = _step_alter_to_note_str(step, alter)
                    midi_num = (octave + 1) * 12 + NOTE_TO_SEMITONE.get(note_name, 0)

                # Extract lyrics
                lyrics_els = child.findall("lyric")
                lyr_texts = []
                for l_el in lyrics_els:
                    t_el = l_el.find("text")
                    if t_el is not None and t_el.text:
                        lyr_texts.append(t_el.text.strip())

                hanzi_text: Optional[str] = None
                pinyin_text: Optional[str] = None
                if len(lyr_texts) >= 2:
                    pinyin_text = lyr_texts[0]
                    hanzi_text = lyr_texts[1]
                elif len(lyr_texts) == 1:
                    hanzi_text = lyr_texts[0]
                    if any(is_cjk_char(c) for c in hanzi_text):
                        py_res = char_list_to_pinyin(hanzi_text, tone_style=tone_style)
                        pinyin_text = " ".join(p for p in py_res if p)

                m_data.notes.append(
                    NoteEvent(
                        pitch_midi=midi_num,
                        step=step,
                        alter=alter,
                        octave=octave,
                        duration_beats=dur_beats,
                        note_type=note_type,
                        lyric_hanzi=hanzi_text,
                        lyric_pinyin=pinyin_text,
                        chord_symbol=pending_chord,
                    )
                )

                if hanzi_text or pending_chord:
                    has_cjk = bool(hanzi_text and any(is_cjk_char(c) for c in hanzi_text))
                    current_line_tokens.append(
                        LyricToken(
                            text=hanzi_text or "",
                            pinyin=pinyin_text or "",
                            chord=pending_chord,
                            is_cjk=has_cjk,
                        )
                    )
                    if not has_cjk and hanzi_text:
                        current_line_tokens.append(LyricToken(text=" ", pinyin="", chord=None, is_cjk=False))
                    pending_chord = None

                beat_cursor += dur_beats

        if pending_chord:
            current_line_tokens.append(LyricToken(text="", pinyin="", chord=pending_chord, is_cjk=False))
            pending_chord = None

        score.measures.append(m_data)
        measures_in_current_line += 1
        # Wrap line every 2 measures (or 4 measures if instrumental) for clean readability
        if measures_in_current_line >= 2 and current_line_tokens:
            # Re-run context-aware pinyin across the whole line in case multi-char words spanned notes
            raw_chordpro_line = "".join(
                (f"[{t.chord}]" if t.chord else "") + t.text for t in current_line_tokens
            )
            retokenized = tokenize_chordpro_line(raw_chordpro_line, tone_style=tone_style)
            current_section.lines.append(SongLine(tokens=retokenized))
            current_line_tokens = []
            measures_in_current_line = 0

    if current_line_tokens:
        raw_chordpro_line = "".join(
            (f"[{t.chord}]" if t.chord else "") + t.text for t in current_line_tokens
        )
        retokenized = tokenize_chordpro_line(raw_chordpro_line, tone_style=tone_style)
        current_section.lines.append(SongLine(tokens=retokenized))

    return score
