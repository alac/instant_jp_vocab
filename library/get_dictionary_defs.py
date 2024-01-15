from fugashi import Tagger
import json
from typing import NamedTuple, Optional


_sentence_parser = None  # type: Optional[Tagger]
_meaning_dict = {}   # type: dict[str, str]
USE_BASE_WORDS = False


def _initialize():
    global _sentence_parser, _meaning_dict
    if _sentence_parser:
        return

    _sentence_parser = Tagger('-Owakati')
    with open("jitendex.json", "r", encoding="utf-8") as f:
        _meaning_dict = json.load(f)


class WordDefinition(NamedTuple):
    word: str
    reading: str
    hiragana_reading: str
    meanings: list[str]


def get_definitions_for_sentence(sentence: str) -> list[WordDefinition]:
    """
    Take a sentence and return definitions for each word.
    :param sentence:
    :return:
    """
    _initialize()
    _sentence_parser.parse(sentence)

    readings = []
    for word in _sentence_parser(sentence):
        # skip particles (助詞) and aux verbs (助動詞)
        if word.feature.pos1 in ["助詞", "助動詞"]:
            continue
        # skip punctuation
        if word.feature.pronBase in ["*"]:
            continue
        if USE_BASE_WORDS:
            base_word = word.feature.lemma
            base_word_reading = word.feature.pronBase
        else:
            base_word = str(word)
            base_word_reading = word.feature.pron
        meanings = _meaning_dict.get(base_word, {}).get("meanings", [])
        readings.append(WordDefinition(base_word, base_word_reading, hiragana_reading(base_word_reading), meanings))
    return readings


def hiragana_reading(katakana_reading: str) -> str:
    katakana_to_hiragana = {
        "ア": "あ",
        "イ": "い",
        "ウ": "う",
        "エ": "え",
        "オ": "お",
        "カ": "か",
        "キ": "き",
        "ク": "く",
        "ケ": "け",
        "コ": "こ",
        "サ": "さ",
        "シ": "し",
        "ス": "す",
        "セ": "せ",
        "ソ": "そ",
        "タ": "た",
        "チ": "ち",
        "ツ": "つ",
        "テ": "て",
        "ト": "と",
        "ナ": "な",
        "ニ": "に",
        "ヌ": "ぬ",
        "ネ": "ね",
        "ノ": "の",
        "ハ": "は",
        "ヒ": "ひ",
        "フ": "ふ",
        "ヘ": "へ",
        "ホ": "ほ",
        "マ": "ま",
        "ミ": "み",
        "ム": "む",
        "メ": "め",
        "モ": "も",
        "ヤ": "や",
        "ユ": "ゆ",
        "ヨ": "よ",
        "ラ": "ら",
        "リ": "り",
        "ル": "る",
        "レ": "れ",
        "ロ": "ろ",
        "ワ": "わ",
        "ヲ": "を",
        "ン": "ん",
        "ヂ": "じ",
        "ヅ": "づ",
        "ッ": "っ",
        "ヰ": "ゐ",
        "ヱ": "ゑ"
    }
    result = [katakana_to_hiragana.get(c, c) for c in katakana_reading]
    return "".join(result)


if __name__ == "__main__":
    text = "麩菓子は、麩を主材料とした日本の菓子。"
    for reading in get_definitions_for_sentence(text):
        print(reading)
