"""
mdict_to_json unpacks the jitendex.mdx file into a json file.

Assumes the source file is in the calling folder and writes the json to the same place.
The output json is formatted:
    reading -> {reading, hiragana_reading, meanings[]}
"""
from readmdict import MDX
from bs4 import BeautifulSoup
import json


def parse(xml: str) -> dict:
    soup = BeautifulSoup(xml, 'html.parser')
    results = {
        "reading": "",
        "hiragana_reading": "",
        "meanings": [],
    }
    kanji_with_furigana = soup.find("span", class_="kanji-form-furigana")

    kanji_readings = []
    hiragana_readings = []

    # kanji_with_furigana is something like "ABCD<ruby>K<rt>R</rt><ruby>E" (and so on)
    # where ABCDE are katakana, K is kanji and R is the katakana reading for K
    for child in kanji_with_furigana.children:
        if isinstance(child, str):
            kanji_readings.append(child)
            hiragana_readings.append(child)
        elif child.name == "ruby":
            for ruby_child in child:
                if isinstance(ruby_child, str):
                    kanji = ruby_child
                    kanji_readings.append(kanji)
                elif ruby_child.name == "rt":
                    hiragana = ruby_child.text
                    hiragana_readings.append(hiragana)

    results["reading"] = "".join(kanji_readings)
    results["hiragana_reading"] = "".join(hiragana_readings)

    try:
        explanation = soup.find("ol", class_="sense-list").find("li", class_="sense").find("div", class_="extra-info").find(
            "fieldset", class_="info-gloss").find("span", class_="info-gloss-content").text
        results["meanings"].append(explanation)
    except:
        pass

    try:
        gloss_tags = soup.find("ol", class_="sense-list").find("li", class_="sense").find_all(
            "li", class_="gloss")
        for tag in gloss_tags:
            results["meanings"].append(tag.text)
    except:
        pass

    return results


if __name__ == "__main__":
    in_filename = r"jitendex.mdx"
    out_filename = r"jitendex.json"

    items = [*MDX(in_filename).items()]
    out_json = {}
    for _, val in items:
        parsed_dict = parse(val.decode())
        out_json[parsed_dict["reading"]] = parsed_dict
    with open(out_filename, "w", encoding="utf-8") as f:
        json.dump(out_json, f)