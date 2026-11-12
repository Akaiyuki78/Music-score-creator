from __future__ import annotations
import re
from typing import List, Tuple, Optional
from .models import LyricToken

try:
    import jieba
    from pypinyin import pinyin, Style, load_phrases_dict
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False


# Musical, Mandopop & Chinese Worship polyphone overrides (多音字修正)
MUSIC_PHRASES_DICT = {
    "音乐": [["yīn"], ["yuè"]],
    "器乐": [["qì"], ["yuè"]],
    "声乐": [["shēng"], ["yuè"]],
    "配乐": [["pèi"], ["yuè"]],
    "乐团": [["yuè"], ["tuán"]],
    "乐队": [["yuè"], ["duì"]],
    "乐器": [["yuè"], ["qì"]],
    "乐章": [["yuè"], ["zhāng"]],
    "乐曲": [["yuè"], ["qǔ"]],
    "交响乐": [["jiāo"], ["xiǎng"], ["yuè"]],
    "快乐": [["kuài"], ["lè"]],
    "喜乐": [["xǐ"], ["lè"]],
    "欢乐": [["huān"], ["lè"]],
    "安乐": [["ān"], ["lè"]],
    "常乐": [["cháng"], ["lè"]],
    "弹琴": [["tán"], ["qín"]],
    "弹唱": [["tán"], ["chàng"]],
    "吉他弹": [["jí"], ["tā"], ["tán"]],
    "子弹": [["zǐ"], ["dàn"]],
    "同行": [["tóng"], ["xíng"]],
    "行事": [["xíng"], ["shì"]],
    "行神迹": [["xíng"], ["shén"], ["jì"]],
    "行在": [["xíng"], ["zài"]],
    "长存": [["cháng"], ["cún"]],
    "长久": [["cháng"], ["jiǔ"]],
    "长远": [["cháng"], ["yuǎn"]],
    "生长": [["shēng"], ["zhǎng"]],
    "成长": [["chéng"], ["zhǎng"]],
    "长大": [["zhǎng"], ["dà"]],
    "归还": [["guī"], ["huán"]],
    "偿还": [["cháng"], ["huán"]],
    "还在": [["hái"], ["zài"]],
    "还有": [["hái"], ["yǒu"]],
    "还要": [["hái"], ["yào"]],
    "还是": [["hái"], ["shì"]],
    "重来": [["chóng"], ["lái"]],
    "重新": [["chóng"], ["xīn"]],
    "重复": [["chóng"], ["fù"]],
    "宝贵": [["bǎo"], ["guì"]],
    "哈利路亚": [["hā"], ["lì"], ["lù"], ["yà"]],
    "阿们": [["ā"], ["men"]],
    "阿门": [["ā"], ["mén"]],
    "看守": [["kān"], ["shǒu"]],
    "看见": [["kàn"], ["jiàn"]],
    "看顾": [["kàn"], ["gù"]],
    "称颂": [["chēng"], ["sòng"]],
    "称谢": [["chēng"], ["xiè"]],
    "曲调": [["qǔ"], ["diào"]],
    "变调": [["biàn"], ["diào"]],
    "转调": [["zhuǎn"], ["diào"]],
    "搁浅": [["gē"], ["qiǎn"]],
    "天空": [["tiān"], ["kōng"]],
    "空白": [["kòng"], ["bái"]],
    "留着": [["liú"], ["zhe"]],
    "拉着": [["lā"], ["zhe"]],
    "读着": [["dú"], ["zhe"]],
    "看着": [["kàn"], ["zhe"]],
    "掩埋": [["yǎn"], ["mái"]],
    "给的": [["gěi"], ["de"]],
    "给你": [["gěi"], ["nǐ"]],
    "不了": [["bù"], ["liǎo"]],
    "原谅不了": [["yuán"], ["liàng"], ["bù"], ["liǎo"]],
    "完了": [["wán"], ["le"]],
    "读完了": [["dú"], ["wán"], ["le"]],
    "当作": [["dàng"], ["zuò"]],
    "只能": [["zhǐ"], ["néng"]],
    "风筝": [["fēng"], ["zheng"]],
    "一旁": [["yì"], ["páng"]],
    "朋友": [["péng"], ["you"]],
    "一生": [["yì"], ["shēng"]],
    "一起": [["yì"], ["qǐ"]],
    "日子": [["rì"], ["zi"]],
    "不再有": [["bù"], ["zài"], ["yǒu"]],
    "一句话": [["yí"], ["jù"], ["huà"]],
    "一辈子": [["yí"], ["bèi"], ["zi"]],
    "一生情": [["yì"], ["shēng"], ["qíng"]],
    "一杯酒": [["yì"], ["bēi"], ["jiǔ"]],
    "不曾孤单": [["bù"], ["céng"], ["gū"], ["dān"]],
    "真朋友": [["zhēn"], ["péng"], ["you"]],
    "你懂我": [["nǐ"], ["dǒng"], ["wǒ"]],
    "还有伤": [["hái"], ["yǒu"], ["shāng"]],
    "还有痛": [["hái"], ["yǒu"], ["tòng"]],
    "还要走": [["hái"], ["yào"], ["zǒu"]],
    "还有我": [["hái"], ["yǒu"], ["wǒ"]],
}

# Single-character overrides for worship pronouns & special chars
SINGLE_CHAR_OVERRIDES = {
    "祢": "nǐ",
    "祂": "tā",
}

_DICT_INITIALIZED = False


def _ensure_dict_initialized() -> None:
    global _DICT_INITIALIZED
    if HAS_PYPINYIN and not _DICT_INITIALIZED:
        load_phrases_dict(MUSIC_PHRASES_DICT)
        for phrase in MUSIC_PHRASES_DICT:
            jieba.add_word(phrase)
        _DICT_INITIALIZED = True


def is_cjk_char(ch: str) -> bool:
    """Returns True if the character is a CJK Unified Ideograph."""
    if not ch:
        return False
    cp = ord(ch[0])
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0xF900 <= cp <= 0xFAFF
    )


def contains_cjk(text: str) -> bool:
    """Returns True if any character in text is CJK."""
    return any(is_cjk_char(ch) for ch in text)


def char_list_to_pinyin(chars: str, tone_style: str = "marks") -> List[str]:
    """
    Converts a string of characters into a list of Pinyin syllables of the exact same length
    as `chars`, using jieba word segmentation for accurate context-dependent polyphones.
    """
    if not chars:
        return []
    if not HAS_PYPINYIN:
        return ["" for _ in chars]

    _ensure_dict_initialized()

    if tone_style == "numbers":
        style = Style.TONE3
    elif tone_style == "none":
        style = Style.NORMAL
    else:
        style = Style.TONE

    # Segment with jieba first to preserve word boundaries for pypinyin
    words = list(jieba.cut(chars))
    result: List[str] = []
    for word in words:
        if len(word) == 1 and word in SINGLE_CHAR_OVERRIDES and tone_style == "marks":
            result.append(SINGLE_CHAR_OVERRIDES[word])
            continue
        py_list = pinyin(word, style=style, heteronym=False, errors="ignore")
        # If pypinyin returns a syllable per CJK char, match 1-to-1
        if len(py_list) == len(word):
            for ch, py_item in zip(word, py_list):
                if ch in SINGLE_CHAR_OVERRIDES and tone_style == "marks":
                    result.append(SINGLE_CHAR_OVERRIDES[ch])
                elif is_cjk_char(ch):
                    result.append(py_item[0] if py_item else "")
                else:
                    result.append("")
        else:
            # Fallback character-by-character if non-CJK mixed inside word
            for ch in word:
                if ch in SINGLE_CHAR_OVERRIDES and tone_style == "marks":
                    result.append(SINGLE_CHAR_OVERRIDES[ch])
                elif is_cjk_char(ch):
                    ch_py = pinyin(ch, style=style, heteronym=False)
                    result.append(ch_py[0][0] if ch_py and ch_py[0] else "")
                else:
                    result.append("")
    return result


CHORD_TAG_RE = re.compile(r"\[([^\]]+)\]")


def tokenize_chordpro_line(line_text: str, tone_style: str = "marks") -> List[LyricToken]:
    """
    Parses a single ChordPro line (e.g., "[C]安静 [G/B]下来的 [Am7]时刻")
    into a list of LyricTokens with accurate Hanyu Pinyin for every Chinese character
    and word-level grouping for English lyrics.
    """
    # Strip trailing newline
    line_text = line_text.rstrip("\r\n")
    if not line_text.strip():
        return []

    # Step 1: Parse into raw segments of (optional_chord, text_chunk)
    parts = CHORD_TAG_RE.split(line_text)
    # parts[0] is text before first chord, then alternating (chord, text_after_chord)
    segments: List[Tuple[Optional[str], str]] = []
    if parts[0]:
        segments.append((None, parts[0]))
    for i in range(1, len(parts), 2):
        chord_str = parts[i].strip()
        following_text = parts[i + 1] if (i + 1) < len(parts) else ""
        segments.append((chord_str, following_text))

    # Step 2: Extract all pure text to compute context-aware Pinyin across the whole line!
    # Even if a [Chord] sits in the middle of a 2-character Chinese phrase (e.g. "安[G]静"),
    # computing Pinyin on the full concatenated text ("安静") keeps polyphone context intact!
    full_plain_text = "".join(seg[1] for seg in segments)
    full_pinyin = char_list_to_pinyin(full_plain_text, tone_style=tone_style)

    tokens: List[LyricToken] = []
    char_cursor = 0

    for chord_str, text_chunk in segments:
        chunk_len = len(text_chunk)
        chunk_pinyins = full_pinyin[char_cursor : char_cursor + chunk_len]
        char_cursor += chunk_len

        if not text_chunk:
            # Chord with no text immediately following (e.g. instrumental bar or trailing chord)
            if chord_str:
                tokens.append(LyricToken(text="", pinyin="", chord=chord_str, is_cjk=False))
            continue

        # Iterate through characters in text_chunk; CJK characters become individual tokens
        # (attaching any trailing punctuation like ，。！？ to the character or as a slim token),
        # while contiguous Latin/English characters group into words.
        pending_chord = chord_str
        i = 0
        while i < chunk_len:
            ch = text_chunk[i]
            py = chunk_pinyins[i] if i < len(chunk_pinyins) else ""

            if is_cjk_char(ch):
                tokens.append(
                    LyricToken(
                        text=ch,
                        pinyin=py,
                        chord=pending_chord,
                        is_cjk=True,
                    )
                )
                pending_chord = None
                i += 1
            elif ch.isspace():
                # Preserve visual spacing
                space_run = ch
                i += 1
                while i < chunk_len and text_chunk[i].isspace():
                    space_run += text_chunk[i]
                    i += 1
                if pending_chord:
                    tokens.append(LyricToken(text=" ", pinyin="", chord=pending_chord, is_cjk=False))
                    pending_chord = None
                else:
                    tokens.append(LyricToken(text=" ", pinyin="", chord=None, is_cjk=False))
            else:
                # English word or punctuation run
                word_run = ch
                i += 1
                while i < chunk_len and not is_cjk_char(text_chunk[i]) and not text_chunk[i].isspace():
                    word_run += text_chunk[i]
                    i += 1
                tokens.append(
                    LyricToken(
                        text=word_run,
                        pinyin="",
                        chord=pending_chord,
                        is_cjk=False,
                    )
                )
                pending_chord = None

    return tokens
