import json
import os.path
import pyperclip
import time

from library.ai_requests import run_ai_request
from library.get_dictionary_defs import get_definitions_for_sentence
from library.token_count import get_token_count
from library.settings_manager import settings


class ANSIColors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def monitor_clipboard(source: str):
    previous_content = ""
    history = []
    history_length = settings.get_setting('vocab_list.ai_translation_history_length')

    cache_file = os.path.join("translation_history", f"{source}.json")
    if os.path.isfile(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            history = json.load(f)

    while True:
        current_clipboard = pyperclip.paste()
        if current_clipboard != previous_content:
            if should_generate_vocabulary_list(sentence=current_clipboard):
                for task in settings.get_setting('vocab_list.processing_order'):
                    if task == "defs":
                        print(get_definitions_string(current_clipboard))
                    if task == "ai_defs":
                        run_vocabulary_list(current_clipboard,
                                            settings.get_setting('vocab_list.ai_definitions_temp'),
                                            use_dictionary=False)
                    if task == "ai_def_rag":
                        run_vocabulary_list(current_clipboard,
                                            settings.get_setting('vocab_list.ai_definitions_augmented_temp'),
                                            use_dictionary=True)
                    if task == "ai_translation":
                        print(ANSIColors.GREEN, end="")
                        translate_with_context(history, current_clipboard)
                        print(ANSIColors.END, end="")
                print("\n\n")
                if current_clipboard not in history:
                    history.append(current_clipboard)
                history = history[-history_length:]
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2)
        previous_content = current_clipboard
        time.sleep(1.0)


def run_vocabulary_list(sentence, temp, use_dictionary=True):
    definitions = ""
    if use_dictionary:
        definitions = get_definitions_string(sentence)

    prompt = """In this task, you are a Japanese teacher. These are notes that you're prepping for your students.

Sentence: 璃燈「……さっきのは言動や行動は全部、この件を解決させる為のものだったんだよな？」
Vocabulary:
    璃燈 (りとう): Rito
    さっきの (さっき の): the just before
    言動や行動 (げんどう や こうどう): words and actions
    全部 (ぜんぶ): all
    この件 (この けん): this matter
    解決させる為のもの (かいけつさせる ため の も の): for the sake of resolving
    だったんだよな？ (だったんだよな): wasn't it?
Idioms:
    N/A

""" + definitions + """

Sentence: 佳伽「でも、そうやってあんまりストイック過ぎると、本番前に心身共に参っちゃうかもしれないよ？」
Vocabulary:
    佳伽 (よしか): Yoshika
    でも (でも): but
    そうやって (そう やって): in that way
    あんまり (あんまり): too much
    ストイック過ぎると (ストイック すぎる と): if you're too strict
    本番前に (ほんばん まえ に): before the main event
    心身共に (しんしん ともに): both physically and mentally
    参っちゃうかもしれないよ？ (まっちゃう かもしれない よ？): you might end up feeling unwell, you know?
Idioms:
    N/A


Sentence: 璃燈「でもな。ちょっと、やり過ぎじゃねぇかな。あたしの気持ちを随分、かき乱してくれたよな？」
Vocabulary:
    璃燈 (りとう): Ritou
    でもな (でもな): but
    ちょっと (ちょっと): a little
    やり過ぎじゃねぇかな (やりすぎ じゃねぇかな): don't you think it's a bit too much?
    あたしの (あたしの): my
    気持ちを (きもちを): feelings
    随分 (ずいぶん): quite a bit
    かき乱してくれた (かきみだしてくれた): you stirred up
    よな？ (よな？): right?
Idioms:
    N/A


Sentence: 結灯「……あの……差し出がましいかもしれませんが……」
Vocabulary:
    結灯 (ゆうひ): Yuuhi
    あの (あの): um
    差し出がましいかもしれませんが (さしでがましいですが): it may be presumptuous, but
Idioms:
    差し出がましいかもしれませんが (sashi dega mashikamo shiremasen ga): it may be presumptuous, but.
        It is used to introduce a suggestion or an opinion that may be considered rude or unnecessary by the listener.
        差し出が (さしでが): to offer, to present
        あげます (あげます): to give


Sentence: """ + sentence.strip() + """
Vocabulary: """
    print("prompt length:", get_token_count(prompt))
    print("Sentence: """ + sentence.strip())
    run_ai_request(prompt, ["Sentence:", "\n\n"], print_prompt=False, temperature=temp,
                   ban_eos_token=False, max_response=500)


def get_definitions_string(sentence: str):
    text = ""
    seen = []
    for definition in get_definitions_for_sentence(sentence):
        if definition.meanings:
            new = f"- {definition.word} ({definition.hiragana_reading})\n"
            if new in seen:
                continue
            text += new
            seen.append(new)

    return f"Definitions:\n{text}"


def should_generate_vocabulary_list(sentence):
    if 5 > len(sentence) or 100 < len(sentence):
        print("Failed length check.")
        return False
    if "\n" in sentence:
        print("Found newline.")
        return False
    jp_grammar_parts = ["。", "」", "「", "は" "に", "が", "か？", "…？", "―"]
    if [p for p in jp_grammar_parts if p in sentence]:
        return True
    print("No sentence parts detected.")
    return False


def translate_with_context(context, sentence, temp=.7):
    prompt = "In this task, use the context and the previous lines to translate the japanese sentence to english\n"
    prompt += settings.get_setting('vocab_list.ai_translation_context')

    if context:
        prompt += ">CONTEXT_START\n"
        for line in context:
            prompt += f"{line}\n"
        prompt += ">CONTEXT_END\n"
    prompt += f">SENTENCE_START\n{sentence}\n>SENTENCE_END\n"
    prompt += f">ENGLISH_START\n"

    print("Translation: ")
    result = run_ai_request(prompt, [">ENGLISH_END", ">END_ENGLISH", ">SENTENCE_END"],
                            print_prompt=False, temperature=temp, ban_eos_token=False, max_response=100)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source",
                        help="The name associated with each 'translation history'. Providing a unique name for each"
                        " allows for tracking each translation history separately when switching sources.",
                        type=str)
    args = parser.parse_args()

    source_settings_path = os.path.join("settings", f"{args.source}.toml")
    if os.path.isfile(source_settings_path):
        settings.override_settings(source_settings_path)

    monitor_clipboard(args.source)
