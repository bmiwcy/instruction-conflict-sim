"""Rule-based language detection (no LLM judge -> reproducible).

A reply is scored on its prose only: fenced code, inline code and URLs are stripped.
Han characters count toward Chinese; kana counts toward Japanese (a Japanese reply
is a violation of a "must answer in Chinese" rule even though it contains Han).
"""
import re

_CODE_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`]*`")
_URL = re.compile(r"https?://\S+")

_HAN = re.compile(r"[一-鿿㐀-䶿]")
_KANA = re.compile(r"[぀-ヿ]")
_HANGUL = re.compile(r"[가-힯]")
_LATIN = re.compile(r"[A-Za-z]")


def strip_non_prose(text: str) -> str:
    text = _CODE_FENCE.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    return _URL.sub(" ", text)


def language_profile(text: str) -> dict:
    t = strip_non_prose(text)
    han, kana = len(_HAN.findall(t)), len(_KANA.findall(t))
    hangul = len(_HANGUL.findall(t))
    # Latin is counted per character; one Han char carries roughly a word, so
    # weight latin by 1/4 to keep the ratio from being dominated by English names/terms.
    latin_w = len(_LATIN.findall(t)) / 4
    total = han + kana + hangul + latin_w
    if total == 0:
        return {"zh": 0.0, "ja": 0.0, "ko": 0.0, "latin": 0.0, "n": 0}
    return {
        "zh": han / total,
        "ja": kana / total,
        "ko": hangul / total,
        "latin": latin_w / total,
        "n": int(total),
    }


def is_chinese(text: str, zh_threshold: float = 0.6, ja_max: float = 0.05):
    """True / False, or None when the reply has no prose (empty or code-only)."""
    p = language_profile(text)
    if p["n"] == 0:
        return None
    return p["zh"] >= zh_threshold and p["ja"] <= ja_max


CHECKS = {"zh": is_chinese}
