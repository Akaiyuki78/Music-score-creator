from __future__ import annotations
import html
import json
import re
from typing import List, Dict, Any, Optional, Tuple

import requests
import urllib3

from .theory import transpose_key, use_flats_for_key, transpose_chord

urllib3.disable_warnings()

UG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def search_community_chords(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """
    Searches the community chord database for verified guitar chord charts.
    Returns a list of versions sorted by rating and popularity.
    """
    clean_q = query.strip()
    # Strip URL formatting or noise if user pasted a link
    if "youtube.com" in clean_q or "youtu.be" in clean_q:
        return []

    # Clean punctuation brackets like 【晴天】 -> 晴天
    clean_q = re.sub(r"[【\[「『(]([^】\]」』)]+)[】\]」』)]", r" \1 ", clean_q)
    clean_q = re.sub(
        r"(Official\s*(Music\s*)?Video|Official\s*MV|MV|HD|4K|1080p|Audio|Lyrics|720p)",
        "",
        clean_q,
        flags=re.I,
    )
    clean_q = re.sub(r"\s+", " ", clean_q).strip()

    url = f"https://www.ultimate-guitar.com/search.php?search_type=title&value={requests.utils.quote(clean_q)}"
    try:
        r = requests.get(url, headers=UG_HEADERS, verify=False, timeout=8)
        if r.status_code != 200:
            return []
        m = re.search(r'class="js-store"\s+data-content="([^"]+)"', r.text)
        if not m:
            return []
        raw_json = html.unescape(m.group(1))
        data = json.loads(raw_json)
        results = data.get("store", {}).get("page", {}).get("data", {}).get("results", [])
        chord_results = [x for x in results if x.get("type") == "Chords"]

        # Sort by rating * log(votes) or votes
        def _score(item):
            votes = item.get("votes", 0) or 0
            rating = item.get("rating", 0.0) or 0.0
            return (rating * 100) + (min(votes, 1000) * 0.5)

        chord_results.sort(key=_score, reverse=True)

        items_out = []
        for cr in chord_results[:limit]:
            ver = cr.get("version", 1)
            votes = cr.get("votes", 0) or 0
            rating = float(cr.get("rating", 0.0) or 0.0)
            key_name = cr.get("tonality_name", "") or ""
            items_out.append({
                "id": cr.get("id"),
                "song_name": cr.get("song_name", clean_q),
                "artist_name": cr.get("artist_name", ""),
                "version": ver,
                "version_desc": f"Version {ver}" + (f" (Key of {key_name})" if key_name else ""),
                "rating": round(rating, 2),
                "votes": votes,
                "key": key_name,
                "tab_url": cr.get("tab_url", ""),
            })
        return items_out
    except Exception as e:
        print(f"[Notice] Community chord search error: {e}")
        return []


def fetch_community_tab_as_chordpro(tab_url: str) -> Optional[Dict[str, Any]]:
    """
    Fetches the verified chord tab at `tab_url` and parses it into clean ChordPro format.
    """
    try:
        r = requests.get(tab_url, headers=UG_HEADERS, verify=False, timeout=10)
        if r.status_code != 200:
            return None
        m = re.search(r'class="js-store"\s+data-content="([^"]+)"', r.text)
        if not m:
            return None
        raw_json = html.unescape(m.group(1))
        data = json.loads(raw_json)
        page_data = data.get("store", {}).get("page", {}).get("data", {})
        tab_view = page_data.get("tab_view", {})
        wiki_tab = tab_view.get("wiki_tab", {})
        content = wiki_tab.get("content", "")
        meta = page_data.get("tab", {})

        song_name = meta.get("song_name", "Untitled Song")
        artist_name = meta.get("artist_name", "")
        key = tab_view.get("meta", {}).get("tonality", "") or meta.get("tonality_name", "") or "C"
        capo = int(tab_view.get("meta", {}).get("capo", 0) or meta.get("capo", 0) or 0)

        chordpro_text = ug_content_to_chordpro(
            raw_text=content,
            title=song_name,
            artist=artist_name,
            key=key,
            capo=capo,
        )

        return {
            "title": song_name,
            "artist": artist_name,
            "key": key,
            "capo": capo,
            "chordpro_text": chordpro_text,
            "source_url": tab_url,
        }
    except Exception as e:
        print(f"[Notice] Failed to fetch tab {tab_url}: {e}")
        return None


def ug_content_to_chordpro(
    raw_text: str,
    title: str = "",
    artist: str = "",
    key: str = "C",
    capo: int = 0,
) -> str:
    """
    Converts Ultimate Guitar tab text with [ch]Chord[/ch] tags into standard ChordPro format.
    """
    lines = raw_text.splitlines()
    cho_lines = []
    if title:
        cho_lines.append(f"{{title: {title}}}")
    if artist:
        cho_lines.append(f"{{artist: {artist}}}")
    if key:
        cho_lines.append(f"{{key: {key}}}")
    if capo > 0:
        cho_lines.append(f"{{capo: {capo}}}")
    cho_lines.append("{time: 4/4}\n")

    i = 0
    in_section = False
    current_sec_type = "verse"

    while i < len(lines):
        line = lines[i]
        clean_l = line.strip()

        # Section headers like [Verse 1], [Chorus], [Bridge], [Intro]
        m_sec = re.match(r"^\[(verse\s*\d*|chorus\s*\d*|bridge\s*\d*|intro|outro|pre-chorus)\]", clean_l, re.I)
        if m_sec:
            sec_name = m_sec.group(1).title()
            if in_section:
                cho_lines.append(f"{{end_of_{current_sec_type}}}\n")
            current_sec_type = "chorus" if "chorus" in sec_name.lower() else (
                "bridge" if "bridge" in sec_name.lower() else "verse"
            )
            cho_lines.append(f"{{start_of_{current_sec_type}: {sec_name}}}")
            in_section = True
            i += 1
            continue

        line_no_tab = line.replace("[tab]", "").replace("[/tab]", "")

        # Check if line contains [ch] tags
        if "[ch]" in line_no_tab:
            chord_line = line_no_tab
            # Check if next line is a lyric line
            if (i + 1) < len(lines) and "[ch]" not in lines[i + 1]:
                lyric_line = lines[i + 1].replace("[tab]", "").replace("[/tab]", "")
                merged = _merge_chords_above_lyrics(chord_line, lyric_line)
                cho_lines.append(merged)
                i += 2
                continue
            else:
                # Instrumental chord line
                clean_chords = re.sub(r"\[ch\]([^\[]+)\[/ch\]", r"[\1]", chord_line).strip()
                cho_lines.append(clean_chords)
                i += 1
                continue

        # Plain text / comment line
        cleaned_plain = re.sub(r"\[/?[a-z]+\]", "", line_no_tab).strip()
        if cleaned_plain:
            cho_lines.append(cleaned_plain)
        elif cho_lines and cho_lines[-1] != "":
            cho_lines.append("")
        i += 1

    if in_section:
        cho_lines.append(f"{{end_of_{current_sec_type}}}")

    return "\n".join(cho_lines)


def _merge_chords_above_lyrics(chord_line: str, lyric_line: str) -> str:
    """Inserts chords [Chord] into the lyric line at the exact character column."""
    chords_with_cols = []
    clean_chord_line = ""
    idx = 0
    while idx < len(chord_line):
        if chord_line[idx:].startswith("[ch]"):
            end_ch = chord_line.find("[/ch]", idx)
            if end_ch != -1:
                c_name = chord_line[idx + 4 : end_ch].strip()
                chords_with_cols.append((len(clean_chord_line), c_name))
                idx = end_ch + 5
                continue
        clean_chord_line += chord_line[idx]
        idx += 1

    if not chords_with_cols:
        return lyric_line.strip()

    max_col = max(c[0] for c in chords_with_cols)
    if len(lyric_line) <= max_col:
        lyric_line = lyric_line.ljust(max_col + 10)

    res = list(lyric_line)
    for col, c_name in sorted(chords_with_cols, key=lambda x: x[0], reverse=True):
        insert_idx = min(col, len(res))
        res.insert(insert_idx, f"[{c_name}]")

    merged = "".join(res).rstrip()
    # Normalize spaces after chord brackets
    merged = re.sub(r"\[([^\]]+)\]\s+([A-Za-z\u4e00-\u9fff])", r"[\1]\2", merged)
    return merged
