from __future__ import annotations
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import soundfile as sf


DEMO_MANDOPOP_WORSHIP_LYRICS = """[Verse 1]
安静下来在祢的面前
放下所有的疲惫与思念
微风轻轻吹过我的心田
听见祢温柔慈爱的诺言

[Chorus]
哈利路亚我心向祢歌颂
祢的恩典陪伴我每一分钟
就算黑夜漫长风雨多汹涌
有祢同行心中就充满喜乐与彩虹

[Bridge]
我不再害怕勇敢向前走
因為祢的爱永远长存不放手
"""

DEMO_CHORDPRO_Preset = """{title: 安静在祢面前 (Still Before You)}
{subtitle: Mandopop & Worship Acoustic Arrangement}
{artist: Praise & Acoustic Guitar Demo}
{key: G}
{capo: 0}
{tempo: 96}
{time: 4/4}

{start_of_verse: Intro}
[G]   [D/F#]   [Em7]   [Cadd9]
{end_of_verse}

{start_of_verse: Verse 1 (主歌)}
[G]安静下来[D/F#]在祢的面[Em7]前
放下[Cadd9]所有的疲[G/B]惫与[Am7]思念[D]
[G]微风轻轻[D/F#]吹过我的[Em7]心田
听见[Cadd9]祢温柔慈[D]爱的诺[G]言
{end_of_verse}

{start_of_chorus: Chorus (副歌)}
[Cadd9]哈利路亚[D]我心向祢[G]歌[D/F#]颂[Em7]
祢的[Am7]恩典陪伴[D]我每一分[G]钟[G7]
就算[Cadd9]黑夜漫长[D]风雨多汹[Bm7]涌[Em7]
有祢[Am7]同行心中就充[D]满喜乐与彩[G]虹
{end_of_chorus}

{start_of_bridge: Bridge (桥段)}
[Em7]我不再害[Bm7]怕勇敢[Cadd9]向前[G]走
因为[Am7]祢的爱永[C]远长存不[D]放手[Dsus4][D]
{end_of_bridge}
"""


def generate_demo_wav_and_lyrics(
    wav_path: Union[str, Path],
    bpm: int = 96,
) -> Tuple[Path, str]:
    """
    Synthesizes a clean 16-bar acoustic guitar chord + lead melody WAV file in the Key of G
    (Progression: G -> D -> Em -> C -> G -> D -> C -> G, repeated twice)
    using Karplus-Strong / harmonic synthesis so the audio analyzer detects real beats,
    key (G major), chords, and melody notes.
    """
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    sr = 22050
    sec_per_beat = 60.0 / bpm
    sec_per_bar = sec_per_beat * 4.0

    # 8-bar progression (G, D, Em, C, G, D, C, G) with MIDI pitches for chord + melody
    progression = [
        # (Chord frequencies in MIDI, 4 quarter-note melody MIDI pitches)
        ([43, 47, 50, 55, 59, 67], [67, 69, 71, 67]),  # G
        ([50, 57, 62, 66],         [66, 67, 69, 66]),  # D
        ([40, 47, 52, 55, 59, 64], [64, 67, 71, 69]),  # Em
        ([48, 52, 55, 60, 64],     [67, 64, 62, 60]),  # C
        ([43, 47, 50, 55, 59, 67], [67, 71, 74, 71]),  # G
        ([50, 57, 62, 66],         [69, 66, 62, 66]),  # D
        ([48, 52, 55, 60, 64],     [64, 67, 69, 66]),  # C
        ([43, 47, 50, 55, 59, 67], [67, 67, 67, 67]),  # G
    ] * 2  # 16 bars total

    total_bars = len(progression)
    total_sec = total_bars * sec_per_bar
    total_samples = int(total_sec * sr)
    audio = np.zeros(total_samples, dtype=np.float32)

    def midi_to_hz(m: int) -> float:
        return 440.0 * (2.0 ** ((m - 69.0) / 12.0))

    for bar_idx, (chord_midis, mel_midis) in enumerate(progression):
        bar_start_t = bar_idx * sec_per_bar

        # Strum acoustic guitar chord on beat 0 and beat 2
        for strum_beat in (0, 2):
            t0 = bar_start_t + strum_beat * sec_per_beat
            dur = sec_per_beat * 1.95
            idx0 = int(t0 * sr)
            idx1 = min(total_samples, idx0 + int(dur * sr))
            if idx1 <= idx0:
                continue
            t_vec = np.linspace(0, (idx1 - idx0) / sr, idx1 - idx0, endpoint=False)
            env = np.exp(-2.2 * t_vec) * (1.0 - np.exp(-60.0 * t_vec))

            chord_sig = np.zeros_like(t_vec)
            for s_idx, m_note in enumerate(chord_midis):
                freq = midi_to_hz(m_note)
                # Slight strum delay across strings + warm harmonics
                strum_offset = s_idx * 0.012
                t_s = np.maximum(0.0, t_vec - strum_offset)
                tone = (
                    0.65 * np.sin(2 * np.pi * freq * t_s)
                    + 0.25 * np.sin(2 * np.pi * (freq * 2) * t_s)
                    + 0.10 * np.sin(2 * np.pi * (freq * 3) * t_s)
                )
                chord_sig += tone / len(chord_midis)

            audio[idx0:idx1] += (chord_sig * env * 0.55).astype(np.float32)

        # Add gentle percussive beat click/shaker and lead melody on each quarter beat
        for beat_i, mel_m in enumerate(mel_midis):
            bt0 = bar_start_t + beat_i * sec_per_beat
            b_idx0 = int(bt0 * sr)
            b_idx1 = min(total_samples, b_idx0 + int(sec_per_beat * 0.9 * sr))
            if b_idx1 <= b_idx0:
                continue
            bt_vec = np.linspace(0, (b_idx1 - b_idx0) / sr, b_idx1 - b_idx0, endpoint=False)

            # Lead vocal/flute-like melody note
            m_freq = midi_to_hz(mel_m)
            m_env = np.exp(-1.8 * bt_vec) * (1.0 - np.exp(-35.0 * bt_vec))
            mel_sig = (
                0.75 * np.sin(2 * np.pi * m_freq * bt_vec)
                + 0.20 * np.sin(2 * np.pi * (m_freq * 2) * bt_vec)
            )
            audio[b_idx0:b_idx1] += (mel_sig * m_env * 0.30).astype(np.float32)

            # Soft percussive transient on every beat for accurate beat tracking
            click_len = min(len(bt_vec), int(0.03 * sr))
            if click_len > 0:
                c_t = bt_vec[:click_len]
                c_env = np.exp(-120.0 * c_t)
                rng = np.random.default_rng(seed=bar_idx * 10 + beat_i)
                click_sig = rng.standard_normal(click_len) * c_env * (0.18 if beat_i in (0, 2) else 0.10)
                audio[b_idx0 : b_idx0 + click_len] += click_sig.astype(np.float32)

    # Normalize safely
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio / peak * 0.88

    sf.write(str(wav_path), audio, sr)
    return wav_path, DEMO_MANDOPOP_WORSHIP_LYRICS


GE_QIAN_CHORDPRO_PRESET = """{title: 搁浅 (Gē Qiǎn / Step Aside)}
{subtitle: 专辑《七里香》 (2004) | 原调 Eb (变调夹 Capo 3 弹 C 调指法)}
{artist: 周杰伦 (Jay Chou) | 词: 宋健彰 | 曲: 周杰伦}
{key: Eb}
{capo: 3}
{tempo: 64}
{time: 4/4}

{start_of_verse: Intro (前奏)}
[Eb]   [Bb/D]   [Cm7]   [Cm7/Bb]   [Abmaj7]   [Gm7]   [Fm7]   [Bb]
{end_of_verse}

{start_of_verse: Verse 1 (主歌)}
[Eb]久未放[Bb/D]晴的天[Cm7]空  依旧[Cm7/Bb]留着你的笑[Abmaj7]容
[Gm7]哭过[Cm7]却无法掩[Fm7]埋歉[Bb]疚
[Eb]风筝在[Bb/D]阴天搁[Cm7]浅  想念[Cm7/Bb]还在等待救[Abmaj7]援
[Gm7]我拉着[Cm7]线复习你[Fm7]给的温[Bb]柔
{end_of_verse}

{start_of_verse: Pre-Chorus (导歌)}
[Gm7]曝晒在一[C7]旁的寂寞  [Fm7]笑我给不起[Bb]承诺
[Gm7]怎么会怎[C7]么会你竟[Fm7]原谅着[Bb]我[Bbsus4][Bb]
{end_of_verse}

{start_of_chorus: Chorus 1 (副歌)}
[Eb]我只能[Bb/D]永远读着对[Cm7]白  读着[Cm7/Bb]我给你的伤[Abmaj7]害
[Gm7]我原谅[Cm7]不了我  [Fm7]就请你当作[Bb]我已不在
[Eb]我睁开[Bb/D]双眼看着空[Cm7]白  忘记[Cm7/Bb]你对我的期[Abmaj7]待
[Gm7]读完了依[Cm7]赖  我[Fm7]很快就[Bb]离[Eb]开
{end_of_chorus}

{start_of_chorus: Final Chorus (尾段升调副歌 +2 Key: F / Capo 3 Play D)}
{comment: 升全音转 F 调 (Capo 3 夹三品弹 D 调指法，或移至 Capo 5 弹 C 调指法)}
[F]我只能[C/E]永远读着对[Dm7]白  读着[Dm7/C]我给你的伤[Bbmaj7]害
[Am7]我原谅[Dm7]不了我  [Gm7]就请你当作[C]我已不在
[F]我睁开[C/E]双眼看着空[Dm7]白  忘记[Dm7/C]你对我的期[Bbmaj7]待
[Am7]读完了依[Dm7]赖  我[Gm7]很快就[C]离[F]开
{end_of_chorus}
"""

PENG_YOU_CHORDPRO_PRESET = """{title: 朋友 (Friends)}
{subtitle: 经典民谣吉他弹唱谱 | 原调 G (免变调夹 / No Capo)}
{artist: 周华健 (Wakin Chau) | 词: 刘思铭 | 曲: 刘志宏}
{key: G}
{capo: 0}
{tempo: 78}
{time: 4/4}

{start_of_verse: Intro (前奏)}
[G]   [D/F#]   [Em7]   [Bm7]   [C]   [G/B]   [Am7]   [D]
{end_of_verse}

{start_of_verse: Verse 1 (主歌 1)}
[G]这些年 [D/F#]一个人 [Em7]风也过 [Bm7]雨也走
[C]有过泪 [G/B]有过错 [Am7]还记得坚[D]持甚么
[G]真爱过 [D/F#]才会懂 [Em7]会寂寞 [Bm7]会回首
[C]终有梦 [G/B]终有你 [Am7]在[D]心[G]头
{end_of_verse}

{start_of_chorus: Chorus (副歌)}
[G]朋友 一生[D/F#]一起走 [Em7]那些日子 [Bm7]不再有
[C]一句话 [G/B]一辈子 [Am7]一生情 [D]一杯酒
[G]朋友 不曾[D/F#]孤单过 [Em7]一声朋友 [Bm7]你会懂
[C]还有伤 [G/B]还有痛 [Am7]还要走 [D]还有[G]我
{end_of_chorus}
"""

SHOUT_TO_THE_LORD_CHORDPRO_PRESET = """{title: Shout to the Lord}
{subtitle: Darlene Zschech / Hillsong Worship | Original Key A (Capo 2 Play G)}
{artist: Hillsong Worship}
{key: A}
{capo: 2}
{tempo: 76}
{time: 4/4}

{start_of_verse: Intro}
[G]   [D]   [Em]   [D]   [C]
{end_of_verse}

{start_of_verse: Verse 1}
[G]My Jesus, [D]My Savior
[Em]Lord, there is [D]none like [C]You
[C]All of my [G]days, [C]I want to [G]praise
The [Em]wonders of Your [F]mighty [C/E]love [D]
[G]My comfort, [D]My shelter
[Em]Tower of [D]refuge and [C]strength
[C]Let every [G]breath, [C]all that I [G]am
[Em]Never cease to [F]worship [C/E]You [D]
{end_of_verse}

{start_of_chorus: Chorus}
[G]Shout to the [Em]Lord, all the [C]earth let us [D]sing
[G]Power and [Em]majesty, [C]praise to the [D]King
[Em]Mountains bow down and the [C]seas will roar
At the [D]sound [Em]of Your [D]name
[G]I sing for [Em]joy at the [C]work of Your [D]hands
[G]Forever I'll [Em]love You, [C]forever I'll [D]stand
[Em]Nothing compares to the [C]promise I [D]have in [G]You
{end_of_chorus}
"""


def generate_ge_qian_wav_and_score(
    wav_path: Union[str, Path],
    bpm: int = 64,
) -> Tuple[Path, str]:
    """
    Synthesizes an acoustic guitar + vocal melody WAV recording of Jay Chou's 《搁浅》 (Gē Qiǎn)
    in Concert Key Eb Major (64 BPM) following the Canon-like descending bass progression:
      Eb -> Bb/D -> Cm7 -> Cm7/Bb -> Abmaj7 -> Gm7 -> Fm7 -> Bb
    and returns (wav_path, GE_QIAN_CHORDPRO_Preset).
    """
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    sr = 22050
    sec_per_beat = 60.0 / bpm
    sec_per_bar = sec_per_beat * 4.0

    # Each bar has two half-bar chords: ((chord1_midis,chord2_midis), [4 quarter-note vocal melody midis])
    # Concert Key Eb Major (Eb4=63, F4=65, G4=67, Ab4=68, Bb4=70, C5=72, D5=74, Eb5=75)
    bars = [
        # Bar 1: Eb -> Bb/D ("久未放晴的天空")
        (([51, 55, 58, 63, 67], [50, 58, 62, 65, 70]), [67, 68, 70, 63]),
        # Bar 2: Cm7 -> Cm7/Bb ("依旧留着你的笑容")
        (([48, 55, 60, 63, 67], [46, 55, 60, 63, 67]), [63, 65, 67, 68]),
        # Bar 3: Abmaj7 -> Gm7 ("哭过却无法")
        (([44, 51, 56, 60, 63], [43, 50, 55, 58, 62]), [65, 63, 60, 62]),
        # Bar 4: Fm7 -> Bb ("掩埋歉疚")
        (([41, 48, 53, 56, 60], [46, 53, 58, 62, 65]), [60, 58, 63, 62]),
        # Bar 5: Chorus Eb -> Bb/D ("我只能永远读着对白")
        (([51, 55, 58, 63, 70], [50, 58, 62, 65, 70]), [70, 74, 75, 74]),
        # Bar 6: Cm7 -> Cm7/Bb ("读着我给你的伤害")
        (([48, 55, 60, 63, 72], [46, 55, 60, 63, 70]), [72, 70, 68, 67]),
        # Bar 7: Abmaj7 -> Gm7 -> Cm7 ("我原谅不了我")
        (([44, 51, 56, 60, 68], [43, 50, 55, 58, 67]), [68, 67, 65, 63]),
        # Bar 8: Fm7 -> Bb -> Eb ("就请你当作我已不在")
        (([41, 48, 53, 56, 65], [46, 53, 58, 62, 65]), [65, 67, 65, 63]),
    ]

    total_bars = len(bars)
    total_sec = total_bars * sec_per_bar
    total_samples = int(total_sec * sr)
    audio = np.zeros(total_samples, dtype=np.float32)

    def midi_to_hz(m: int) -> float:
        return 440.0 * (2.0 ** ((m - 69.0) / 12.0))

    for bar_idx, ((ch1_midis, ch2_midis), mel_midis) in enumerate(bars):
        bar_start_t = bar_idx * sec_per_bar

        for strum_beat, chord_midis in ((0, ch1_midis), (2, ch2_midis)):
            t0 = bar_start_t + strum_beat * sec_per_beat
            dur = sec_per_beat * 1.95
            idx0 = int(t0 * sr)
            idx1 = min(total_samples, idx0 + int(dur * sr))
            if idx1 <= idx0:
                continue
            t_vec = np.linspace(0, (idx1 - idx0) / sr, idx1 - idx0, endpoint=False)
            env = np.exp(-1.9 * t_vec) * (1.0 - np.exp(-55.0 * t_vec))

            chord_sig = np.zeros_like(t_vec)
            for s_idx, m_note in enumerate(chord_midis):
                freq = midi_to_hz(m_note)
                strum_offset = s_idx * 0.015
                t_s = np.maximum(0.0, t_vec - strum_offset)
                tone = (
                    0.68 * np.sin(2 * np.pi * freq * t_s)
                    + 0.22 * np.sin(2 * np.pi * (freq * 2) * t_s)
                    + 0.10 * np.sin(2 * np.pi * (freq * 3) * t_s)
                )
                chord_sig += tone / len(chord_midis)

            audio[idx0:idx1] += (chord_sig * env * 0.58).astype(np.float32)

        for beat_i, mel_m in enumerate(mel_midis):
            bt0 = bar_start_t + beat_i * sec_per_beat
            b_idx0 = int(bt0 * sr)
            b_idx1 = min(total_samples, b_idx0 + int(sec_per_beat * 0.92 * sr))
            if b_idx1 <= b_idx0:
                continue
            bt_vec = np.linspace(0, (b_idx1 - b_idx0) / sr, b_idx1 - b_idx0, endpoint=False)

            m_freq = midi_to_hz(mel_m)
            m_env = np.exp(-1.5 * bt_vec) * (1.0 - np.exp(-30.0 * bt_vec))
            mel_sig = (
                0.78 * np.sin(2 * np.pi * m_freq * bt_vec)
                + 0.18 * np.sin(2 * np.pi * (m_freq * 2) * bt_vec)
            )
            audio[b_idx0:b_idx1] += (mel_sig * m_env * 0.28).astype(np.float32)

    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio / peak * 0.88

    sf.write(str(wav_path), audio, sr)
    return wav_path, GE_QIAN_CHORDPRO_PRESET
