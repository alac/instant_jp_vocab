import json
import os.path
import pyperclip
import time
from queue import SimpleQueue
from typing import Optional
from threading import Lock

from library.ai_requests import run_ai_request_stream
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


class UIUpdateCommand:
    def __init__(self, update_type: str, sentence: str, token: str):
        self.update_type = update_type
        self.sentence = sentence
        self.token = token


REQUEST_INTERRUPT_FLAG = False
REQUEST_INTERRUPT_LOCK = Lock()


def request_interrupt_atomic_swap(new_value: bool) -> bool:
    global REQUEST_INTERRUPT_FLAG
    with REQUEST_INTERRUPT_LOCK:
        old_value = REQUEST_INTERRUPT_FLAG
        REQUEST_INTERRUPT_FLAG = new_value
    return old_value


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
                    # if task == "defs":
                    #     print(get_definitions_string(current_clipboard))
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
                        translate_with_context(history, current_clipboard)
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


def run_vocabulary_list(sentence: str, temp: float, use_dictionary: bool = True,
                        update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None):
    request_interrupt_atomic_swap(False)

    definitions = ""
    if use_dictionary:
        definitions = get_definitions_string(sentence)

    prompt = """<|system|>Enter RP mode. Pretend to be a Japanese teacher whose persona follows:
As a Japanese teacher, you're working on helping your students learn how to parse sentences, breaking them down into words and pointing out idioms. For each word or idiom you define, you include the reading in parenthesis and the definition after a "-" character.

You shall reply to the user while staying in character, and generate accurate responses.
    

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

Sentence: 璃燈「カレシなら、カノジョをその気にさせた責任取れよ」
Vocabulary:
   璃燈 (りとう): Rito
   カレシ (かれし): boyfriend
   カノジョ (かのじょ): girlfriend
   その気にさせた (そのきにさせた): to make someone fall in love
   責任 (せきにん): responsibilities
   取れよ (とれよ): should take
Idioms:
   その気にさせる (sono ki ni saseru): to make someone fall in love
       It is a common phrase in romantic manga and anime.
       その (その): that
       気 (き): feeling
       に (に): at
       させる (させる): to make


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
    last_tokens = []
    for tok in run_ai_request_stream(prompt, ["Sentence:", "\n\n"], print_prompt=False,
                                     temperature=temp, ban_eos_token=False, max_response=500):
        if request_interrupt_atomic_swap(False):
            break
        if update_queue is not None:
            update_queue.put(UIUpdateCommand("define", sentence, tok))
        last_tokens.append(tok)
        last_tokens = last_tokens[-10:]
        if len(last_tokens) == 10 and len(set(last_tokens)) <= 3:
            break


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
    jp_grammar_parts = ["。", "」", "「", "は" "に", "が", "？", "―", "…", "！", "』", "『"]
    if [p for p in jp_grammar_parts if p in sentence]:
        return True
    print("No sentence parts detected.")
    return False


def translate_with_context(context, sentence, temp=.7,
                           update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None):
    request_interrupt_atomic_swap(False)
    prompt = ("<|system|>Enter RP mode. Pretend to be a Japanese translator whose persona follows:"
              " You are a Japanese teacher, working on study material for your students. You take into account "
              " information about the characters, the previous lines from stories and provide an accurate translation "
              " for the sentence given. You shall reply to the user while staying in character, and generate accurate"
              " responses.\n")

    prompt += settings.get_setting('vocab_list.ai_translation_context')

    if context:
        prompt += ">CONTEXT_START\n"
        for line in context:
            prompt += f"{line}\n"
        prompt += ">CONTEXT_END\n"
    prompt += f">SENTENCE_START\n{sentence}\n>SENTENCE_END\n"
    prompt += f">ENGLISH_START\n"

    print("Translation: ")
    last_tokens = []
    for tok in run_ai_request_stream(prompt,
                              [">ENGLISH_END", ">END_ENGLISH", ">SENTENCE_END", "\n\n\n", ">\n>\n>"],
                              print_prompt=False, temperature=temp, ban_eos_token=False, max_response=100):
        if request_interrupt_atomic_swap(False):
            break
        if update_queue is not None:
            update_queue.put(UIUpdateCommand("translate", sentence, tok))
        last_tokens.append(tok)
        last_tokens = last_tokens[-10:]
        if len(last_tokens) == 10 and len(set(last_tokens)) <= 3:
            break


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
