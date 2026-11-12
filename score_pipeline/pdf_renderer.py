from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from .models import SongScore, LyricToken
from .theory import (
    transpose_chord,
    transpose_key,
    chord_to_nashville,
    use_flats_for_key,
    get_guitar_voicing,
)

_REGISTERED_FONT_REGULAR = "Helvetica"
_REGISTERED_FONT_BOLD = "Helvetica-Bold"
_FONTS_INITIALIZED = False


def _ensure_cjk_fonts() -> Tuple[str, str]:
    """
    Registers a system TrueType/OpenType font that supports both Simplified/Traditional
    Chinese characters (Hanzi) AND Hanyu Pinyin tone marks (ā á ǎ à ē é ě è ...).
    On Windows, Microsoft YaHei (`msyh.ttc` / `msyhbd.ttc`) or SimHei/SimSun are built-in.
    """
    global _REGISTERED_FONT_REGULAR, _REGISTERED_FONT_BOLD, _FONTS_INITIALIZED
    if _FONTS_INITIALIZED:
        return _REGISTERED_FONT_REGULAR, _REGISTERED_FONT_BOLD

    win_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates_reg = [
        ("MSYaHei", win_fonts / "msyh.ttc", 0),
        ("SimHei", win_fonts / "simhei.ttf", None),
        ("SimSun", win_fonts / "simsun.ttc", 0),
        ("ArialUnicode", win_fonts / "ARIALUNI.TTF", None),
    ]
    candidates_bold = [
        ("MSYaHei-Bold", win_fonts / "msyhbd.ttc", 0),
        ("MSYaHei", win_fonts / "msyh.ttc", 0),
    ]

    for font_name, font_path, sub_idx in candidates_reg:
        if font_path.exists():
            try:
                if sub_idx is not None:
                    pdfmetrics.registerFont(TTFont(font_name, str(font_path), subfontIndex=sub_idx))
                else:
                    pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                _REGISTERED_FONT_REGULAR = font_name
                break
            except Exception:
                continue

    for font_name, font_path, sub_idx in candidates_bold:
        if font_path.exists():
            try:
                if font_name != _REGISTERED_FONT_REGULAR:
                    if sub_idx is not None:
                        pdfmetrics.registerFont(TTFont(font_name, str(font_path), subfontIndex=sub_idx))
                    else:
                        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                _REGISTERED_FONT_BOLD = font_name
                break
            except Exception:
                continue
    else:
        _REGISTERED_FONT_BOLD = _REGISTERED_FONT_REGULAR

    _FONTS_INITIALIZED = True
    return _REGISTERED_FONT_REGULAR, _REGISTERED_FONT_BOLD


def _draw_guitar_chord_diagram(
    c: canvas.Canvas,
    x: float,
    y_top: float,
    chord_name: str,
    font_reg: str,
    font_bold: str,
    box_w: float = 46.0,
    box_h: float = 52.0,
) -> None:
    """
    Draws a clean 6-string, 4-fret guitar chord diagram at (x, y_top).
    """
    voicing = get_guitar_voicing(chord_name)
    active_frets = [f for f in voicing if f > 0]
    min_fret = min(active_frets) if active_frets else 1
    max_fret = max(active_frets) if active_frets else 1
    base_fret = 1 if max_fret <= 4 else min_fret

    # Chord label centered above diagram
    c.setFont(font_bold, 9.5)
    c.setFillColor(colors.HexColor("#1E3A8A"))
    c.drawCentredString(x + box_w / 2.0, y_top - 9, chord_name)

    grid_x = x + 7
    grid_w = box_w - 14
    grid_h = box_h - 22
    grid_top = y_top - 18
    grid_bottom = grid_top - grid_h

    string_spacing = grid_w / 5.0
    fret_spacing = grid_h / 4.0

    # Draw nut (thick line if base_fret == 1)
    c.setStrokeColor(colors.HexColor("#334155"))
    if base_fret == 1:
        c.setLineWidth(2.2)
        c.line(grid_x, grid_top, grid_x + grid_w, grid_top)
    else:
        c.setLineWidth(0.8)
        c.line(grid_x, grid_top, grid_x + grid_w, grid_top)
        c.setFont(font_reg, 6.5)
        c.setFillColor(colors.HexColor("#475569"))
        c.drawRightString(grid_x - 2, grid_top - fret_spacing * 0.75, f"{base_fret}fr")

    # Draw frets & strings
    c.setLineWidth(0.6)
    for f_idx in range(1, 5):
        fy = grid_top - f_idx * fret_spacing
        c.line(grid_x, fy, grid_x + grid_w, fy)

    for s_idx in range(6):
        sx = grid_x + s_idx * string_spacing
        c.line(sx, grid_top, sx, grid_bottom)

    # Draw open ('o'), muted ('x'), and fretted finger dots
    for s_idx, fret_val in enumerate(voicing):
        sx = grid_x + s_idx * string_spacing
        if fret_val == -1:
            c.setFont(font_reg, 6.5)
            c.setFillColor(colors.HexColor("#64748B"))
            c.drawCentredString(sx, grid_top + 2.5, "×")
        elif fret_val == 0:
            c.setStrokeColor(colors.HexColor("#475569"))
            c.setLineWidth(0.7)
            c.circle(sx, grid_top + 4.5, 2.0, stroke=1, fill=0)
        else:
            rel_fret = fret_val - base_fret + 1
            if 1 <= rel_fret <= 4:
                dot_y = grid_top - (rel_fret - 0.5) * fret_spacing
                c.setFillColor(colors.HexColor("#1D4ED8"))
                c.circle(sx, dot_y, 2.7, stroke=0, fill=1)


def render_score_to_pdf(
    score: SongScore,
    output_path: Union[str, Path],
    transpose_semitones: int = 0,
    use_capo_shapes: bool = True,
    use_nashville: bool = False,
    show_pinyin: bool = True,
    show_chord_diagrams: bool = True,
) -> Path:
    """
    Renders a `SongScore` into a high-readability Guitarist & Church Worship PDF.
    Features:
      - Header banner with Concert Key, Capo Play Key, Tempo, Time Signature
      - Guitar Fretboard Chord Diagrams for all chords in the song
      - 3-tier vertical alignment per lyric line:
          Row 1: Guitar Chord (e.g. [G], [D/F#], [Em7])
          Row 2: Hanyu Pinyin with tone marks (e.g. ān jìng)
          Row 3: Chinese Characters (Hanzi) / English lyrics
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_reg, font_bold = _ensure_cjk_fonts()

    page_w, page_h = A4
    margin_x = 38.0
    margin_top = 32.0
    margin_bottom = 28.0
    usable_w = page_w - 2 * margin_x

    c = canvas.Canvas(str(output_path), pagesize=A4)
    c.setTitle(f"{score.title} - Guitar & Pinyin Score")

    effective_semi = transpose_semitones - (score.capo if use_capo_shapes else 0)
    concert_key = transpose_key(score.key, transpose_semitones)
    play_key = transpose_key(concert_key, -(score.capo if use_capo_shapes else 0))
    prefer_flats = use_flats_for_key(play_key)

    def _transform_chord(ch: Optional[str]) -> Optional[str]:
        if not ch:
            return None
        transposed = transpose_chord(ch, effective_semi, prefer_flats=prefer_flats)
        if use_nashville:
            return chord_to_nashville(transposed, play_key)
        return transposed

    page_num = 1

    def _draw_footer() -> None:
        c.setFont(font_reg, 8.5)
        c.setFillColor(colors.HexColor("#64748B"))
        footer_left = f"{score.title} — Concert Key: {concert_key}"
        if score.capo > 0 and use_capo_shapes:
            footer_left += f" (Capo {score.capo} -> Play {play_key} shapes)"
        c.drawString(margin_x, 20, footer_left)
        c.drawRightString(page_w - margin_x, 20, f"Page {page_num}")

    def _new_page() -> float:
        nonlocal page_num
        _draw_footer()
        c.showPage()
        page_num += 1
        return page_h - margin_top

    y = page_h - margin_top

    # --- Header Block ---
    c.setFont(font_bold, 20)
    c.setFillColor(colors.HexColor("#0F172A"))
    c.drawString(margin_x, y, score.title)

    if score.artist:
        c.setFont(font_reg, 11)
        c.setFillColor(colors.HexColor("#475569"))
        c.drawRightString(page_w - margin_x, y + 2, score.artist)
    y -= 18

    if score.subtitle:
        c.setFont(font_reg, 10)
        c.setFillColor(colors.HexColor("#64748B"))
        c.drawString(margin_x, y, score.subtitle)
        y -= 14

    # Metadata Pill Bar (Key, Capo, Tempo, Time Signature)
    y -= 4
    c.setFillColor(colors.HexColor("#F1F5F9"))
    c.roundRect(margin_x, y - 16, usable_w, 22, 4, stroke=0, fill=1)

    c.setFont(font_bold, 9.5)
    c.setFillColor(colors.HexColor("#1E293B"))
    meta_items = [f"Concert Key: {concert_key}"]
    if score.capo > 0:
        if use_capo_shapes:
            meta_items.append(f"Capo Fret: {score.capo} (Play: {play_key})")
        else:
            meta_items.append(f"Suggested Capo: {score.capo}")
    elif use_capo_shapes:
        meta_items.append(f"Play Key: {play_key} (No Capo)")
    if use_nashville:
        meta_items.append("Notation: Nashville Numbers")
    meta_items.append(f"Tempo: {score.tempo_bpm} BPM")
    meta_items.append(f"Time: {score.time_signature}")

    c.drawString(margin_x + 10, y - 9, "   •   ".join(meta_items))
    y -= 28

    # --- Guitar Fretboard Diagrams Row ---
    if show_chord_diagrams and not use_nashville:
        raw_unique = score.get_unique_chords()
        transformed_unique: List[str] = []
        for ch in raw_unique:
            tc = _transform_chord(ch)
            if tc and tc not in transformed_unique:
                transformed_unique.append(tc)

        if transformed_unique:
            diag_w = 50.0
            diag_h = 56.0
            per_row = int(usable_w // diag_w)
            for idx, ch_sym in enumerate(transformed_unique[: per_row * 2]):
                col = idx % per_row
                if idx > 0 and col == 0:
                    y -= diag_h + 6
                dx = margin_x + col * diag_w
                _draw_guitar_chord_diagram(c, dx, y, ch_sym, font_reg, font_bold)
            y -= diag_h + 10
            c.setStrokeColor(colors.HexColor("#E2E8F0"))
            c.setLineWidth(0.8)
            c.line(margin_x, y, page_w - margin_x, y)
            y -= 14

    # --- Sections & 3-Tier Interlinear Lines ---
    chord_font_size = 10.5
    pinyin_font_size = 8.5
    lyric_font_size = 12.5

    for sec in score.sections:
        # Check space for section header + at least one 3-tier line
        if y < margin_bottom + 65:
            y = _new_page()

        # Section Title Pill / Header
        is_chorus = sec.section_type.lower() == "chorus" or "chorus" in sec.name.lower() or "副歌" in sec.name
        c.setFont(font_bold, 10.5)
        if is_chorus:
            c.setFillColor(colors.HexColor("#1D4ED8"))
        else:
            c.setFillColor(colors.HexColor("#334155"))
        c.drawString(margin_x, y, f"[{sec.name}]")
        y -= 13

        section_top_y = y + 4

        for s_line in sec.lines:
            if s_line.is_comment:
                if y < margin_bottom + 20:
                    y = _new_page()
                c.setFont(font_reg, 9.0)
                c.setFillColor(colors.HexColor("#64748B"))
                c.drawString(margin_x + (10 if is_chorus else 0), y, s_line.comment_text)
                y -= 14
                continue

            if not s_line.tokens:
                continue

            # Determine whether this line has any Pinyin or Chords
            has_any_chord = any(t.chord for t in s_line.tokens)
            has_any_pinyin = show_pinyin and any(t.pinyin for t in s_line.tokens)
            has_any_text = any(t.text.strip() for t in s_line.tokens)

            # Calculate x-widths of each token cell so Chord, Pinyin, and Hanzi never collide!
            token_cells: List[Tuple[LyricToken, Optional[str], float]] = []
            for tok in s_line.tokens:
                disp_chord = _transform_chord(tok.chord)
                w_chord = (
                    pdfmetrics.stringWidth(disp_chord, font_bold, chord_font_size) + 5.0
                    if disp_chord
                    else 0.0
                )
                w_pinyin = (
                    pdfmetrics.stringWidth(tok.pinyin, font_reg, pinyin_font_size) + 3.0
                    if (show_pinyin and tok.pinyin)
                    else 0.0
                )
                if tok.text == " ":
                    w_text = 6.0
                else:
                    w_text = (
                        pdfmetrics.stringWidth(tok.text, font_reg, lyric_font_size)
                        + (3.0 if tok.is_cjk else 2.0)
                    )
                cell_w = max(w_chord, w_pinyin, w_text)
                token_cells.append((tok, disp_chord, cell_w))

            # Wrap cells into visual rows that fit within `usable_w`
            indent_x = 10.0 if is_chorus else 0.0
            max_row_w = usable_w - indent_x
            rows: List[List[Tuple[LyricToken, Optional[str], float]]] = []
            curr_row: List[Tuple[LyricToken, Optional[str], float]] = []
            curr_w = 0.0

            for cell in token_cells:
                if curr_row and (curr_w + cell[2] > max_row_w):
                    rows.append(curr_row)
                    curr_row = [cell]
                    curr_w = cell[2]
                else:
                    curr_row.append(cell)
                    curr_w += cell[2]
            if curr_row:
                rows.append(curr_row)

            # Render each wrapped visual row
            for r_cells in rows:
                row_height = (
                    (13.0 if has_any_chord else 0.0)
                    + (13.0 if has_any_pinyin else 0.0)
                    + (15.5 if has_any_text else 3.5)
                    + 2.5
                )
                if y - row_height < margin_bottom:
                    y = _new_page()
                    section_top_y = y

                cursor_x = margin_x + indent_x
                y_chord = y
                y_pinyin = y_chord - (12.5 if has_any_chord else 0.0)
                y_lyric = y_pinyin - (13.5 if has_any_pinyin else (12.5 if has_any_chord else 0.0))

                for tok, disp_chord, cell_w in r_cells:
                    # 1. Draw Chord
                    if disp_chord:
                        c.setFont(font_bold, chord_font_size)
                        c.setFillColor(colors.HexColor("#1D4ED8"))
                        c.drawString(cursor_x, y_chord, disp_chord)

                    # 2. Draw Hanyu Pinyin (centered over CJK character, or left-aligned)
                    if show_pinyin and tok.pinyin:
                        c.setFont(font_reg, pinyin_font_size)
                        c.setFillColor(colors.HexColor("#64748B"))
                        if tok.is_cjk:
                            c.drawCentredString(cursor_x + cell_w / 2.0 - 1.5, y_pinyin, tok.pinyin)
                        else:
                            c.drawString(cursor_x, y_pinyin, tok.pinyin)

                    # 3. Draw Lyric (Hanzi or English)
                    if tok.text and tok.text != " ":
                        c.setFont(font_reg, lyric_font_size)
                        c.setFillColor(colors.HexColor("#0F172A"))
                        if tok.is_cjk:
                            c.drawCentredString(cursor_x + cell_w / 2.0 - 1.5, y_lyric, tok.text)
                        else:
                            c.drawString(cursor_x, y_lyric, tok.text)

                    cursor_x += cell_w

                y -= row_height

        # Chorus accent bar on the left
        if is_chorus and section_top_y > y + 6:
            c.setStrokeColor(colors.HexColor("#3B82F6"))
            c.setLineWidth(2.5)
            c.line(margin_x + 2, section_top_y, margin_x + 2, y + 8)

        y -= 6

    _draw_footer()
    c.save()
    return output_path
