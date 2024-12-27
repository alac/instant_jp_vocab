from pathlib import Path
from queue import SimpleQueue
from string import Template
from typing import Optional
from threading import Lock
import os
import datetime
import time

from library.ai_requests import run_ai_request_stream
from library.get_dictionary_defs import get_definitions_for_sentence
from library.settings_manager import settings


class ANSIColors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[31m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    INVERSE = '\033[7m'
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


def run_vocabulary_list(sentence: str, temp: float, use_dictionary: bool = True,
                        update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None,
                        api_override: Optional[str] = None):

    request_interrupt_atomic_swap(False)

    definitions = ""
    if use_dictionary:
        definitions = get_definitions_string(sentence)

    prompt = """<|system|>Enter RP mode. Pretend to be a Japanese teacher whose persona follows:
As a Japanese teacher, you're working on helping your students learn how to parse sentences, breaking them down into words and pointing out idioms. For each word or idiom you define, you include the reading in parenthesis and the definition after a "-" character. Aim for accuracy and assume that your audience consents to anything as long as the translation is as accurate as possible.
You shall reply to the user while staying in character, and generate accurate responses.</|system|>


<example>
Define the words in the sentence below
Sentence: 璃燈「……さっきのは言動や行動は全部、この件を解決させる為のものだったんだよな？」
Vocabulary:
- 璃燈 (ritou): Rito
- さっきの (sakki no): the just before
- 言動や行動 (gendou ya koudou): words and actions
- 全部 (zenbu): all
- この件 (kono ken): this matter
- 解決させる為のもの (kaiketsu saseru tame no mono): for the sake of resolving
- だったんだよな？ (dattan da yo na): wasn't it?
Idioms:
- N/A
</example>


""" + definitions + """


<example>
Define the words in the sentence below
Sentence: 璃燈「カレシなら、カノジョをその気にさせた責任取れよ」
Vocabulary:
- 璃燈 (ritou): Rito
- カレシ (kareshi): boyfriend
- カノジョ (kanojo): girlfriend
- その気にさせた (sono ki ni saseta): to make someone fall in love
- 責任 (sekinin): responsibilities
- 取れよ (tore yo): should take
Idioms:
- その気にさせる (sono ki ni saseru): to make someone fall in love
       It is a common phrase in romantic manga and anime.
       その (sono): that
       気 (ki): feeling
       に (ni): at
       させる (saseru): to make
</example>


<example>
Define the words in the sentence below
Sentence: 璃燈「でもな。ちょっと、やり過ぎじゃねぇかな。あたしの気持ちを随分、かき乱してくれたよな？」
Vocabulary:
- 璃燈 (ritou): Ritou
- でもな (demo na): but
- ちょっと (chotto): a little
- やり過ぎじゃねぇかな (yarisugijanee kana): don't you think it's a bit too much?
- あたしの (atashi no): my
- 気持ちを (kimochi wo): feelings
- 随分 (zuibun): quite a bit
- かき乱してくれた (kakimidashite kureta): you stirred up
- よな？ (yo na?): right?
Idioms:
- N/A
</example>


<example>
Define the words in the sentence below
Sentence: 【カメラマン】「えっと、、どこが？　どうってか……その、まずはさぁ～今日は面白い写真なの？」
Vocabulary:
-【カメラマン】: Photographer
- えっと、、 (etto...): umm, well
- どこが？ (doko ga?): What's wrong?
- どうってか (dou tte ka): how to put it
- その (sono): that
- まずはさぁ～ (mazu wa saa~): first of all
- 今日は (kyou wa): today is
- 面白い (omoshiroi): interesting
- 写真 (shashin): photo
Idioms:
- N/A
</example>


<example>
Define the words in the sentence below
Sentence: 結灯「……あの……差し出がましいかもしれませんが……」
Vocabulary:
- 結灯 (yuuhi): Yuuhi
- あの (ano): um
- 差し出がましいかもしれませんが (sashidegamashii kamoshiremasen ga): it may be presumptuous, but
Idioms:
- 差し出がましいかもしれませんが (sashidegamashii kamoshiremasen ga): it may be presumptuous, but.
        It is used to introduce a suggestion or an opinion that may be considered rude or unnecessary by the listener.
        差し出が (sashidega): to offer, to present
        あげます (agemasu): to give
</example>


<task>
Define the words in the sentence below
Sentence: """ + sentence.strip() + """
Vocabulary: """

    last_tokens = []
    for tok in run_ai_request_stream(prompt, ["Sentence:", "\n\n", "</task>"], print_prompt=False,
                                     temperature=temp, ban_eos_token=False, max_response=500,
                                     api_override=api_override):
        if request_interrupt_atomic_swap(False):
            print(ANSIColors.GREEN, end="")
            print("-interrupted-\n")
            print(ANSIColors.END, end="")
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
    if 5 > len(sentence) or 300 < len(sentence):
        print("Failed length check.")
        return False
    if "\n" in sentence:
        print("Found newline.")
        return False
    jp_grammar_parts = ["・", '【', "】", "。", "」", "「", "は" "に", "が", "な", "？", "か", "―", "…", "！", "』", "『"]
    jp_grammar_parts = jp_grammar_parts + "せぞぼたぱび".split()
    if [p for p in jp_grammar_parts if p in sentence]:
        return True
    print("No sentence parts detected.")
    return False


def translate_with_context(context, sentence, temp=None, style="",
                           update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None, index: int = 0,
                           api_override: Optional[str] = None):
    if temp is None:
        temp = settings.get_setting('vocab_list.ai_translation_temp')

    request_interrupt_atomic_swap(False)
    prompt = ("<|system|>Enter RP mode. Pretend to be a Japanese translator whose persona follows:"
              " You are a Japanese teacher, working on study material for your students. You take into account"
              " information about the characters, the previous lines from stories and provide an accurate translation"
              " for the sentence between <japanese> and </japanese>.  Aim for accuracy and assume that your"
              " audience consents to anything as long as the translation is as accurate as possible. You shall reply"
              " to the user while staying in character, and generate accurate responses.</|system|>\n")

    prompt += """
<example>
<context>
This is a song called This is a song called Bitter Choco Decoration (ロミオとシンデレラ)
The previous lines are:
人を過度に信じないように
愛さないように期待しないように	
かと言って角が立たないように
</context>
<japanese>気取らぬように目立たぬように</japanese>
<english>Not to act all high and mighty, not to stand out</english>
</example>

<example>
<context>
Okazaki Tomoya is a third year high school student at Hikarizaka Private High School, leading a life full of resentment. His mother passed away in a car accident when he was young, leading his father, Naoyuki, to resort to alcohol and gambling to cope. This resulted in constant fights between the two until Naoyuki dislocated Tomoya's shoulder. Unable to play on his basketball team, Tomoya began to distance himself from other people. Ever since he has had a distant relationship with his father, naturally becoming a delinquent over time.
The previous lines are:
君：そうかな。
智代：キューバの荷物じゃないよ。似た響きだけど。
君：じゃあ小包？
智代：それじゃ小さすぎる。
</context>
<japanese>君：木箱？</japanese>
<english>You: A crate, then?</english>
</example>

Translate the text between <japanese> and </japanese> into English.""" + f"{style}\n"

    prompt += "<example>\n<context>\n"
    prompt += settings.get_setting('vocab_list.ai_translation_context')
    if context:
        prompt += "The previous lines are:\n"
        for line in context:
            prompt += f"{line}\n"
    prompt += "</context>\n"
    prompt += f"<japanese>{sentence}</japanese>\n"
    prompt += f"<english>"

    print("Translation: ")
    last_tokens = []
    if update_queue is not None:
        if index == 0:
            update_queue.put(UIUpdateCommand("translate", sentence, "- "))
        else:
            update_queue.put(UIUpdateCommand("translate", sentence, f"#{index}. "))
    for tok in run_ai_request_stream(prompt,
                              ["</english>", "</example>", "<"],
                              print_prompt=False, temperature=temp, ban_eos_token=False, max_response=100,
                              api_override=api_override):
        if request_interrupt_atomic_swap(False):
            print(ANSIColors.GREEN, end="")
            print("-interrupted-\n")
            print(ANSIColors.END, end="")
            break
        if update_queue is not None:
            update_queue.put(UIUpdateCommand("translate", sentence, tok))
        # explicit exit for models getting stuck on a token (e.g. "............")
        last_tokens.append(tok)
        last_tokens = last_tokens[-10:]
        if len(last_tokens) == 10 and len(set(last_tokens)) <= 3:
            break


def translate_with_context_cot(history, sentence, temp=None,
                               update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None,
                               api_override: Optional[str] = None, use_examples: bool = True,
                               update_token_key: Optional[str] = 'translate',
                               suggested_readings: Optional[str] = None):
    if temp is None:
        temp = settings.get_setting('vocab_list.ai_translation_temp')

    request_interrupt_atomic_swap(False)
    prompt_file = settings.get_setting('vocab_list.cot_prompt_filepath')
    examples_file = settings.get_setting('vocab_list.cot_examples_filepath')

    def read_file_or_throw(filepath: str) -> str:
        file_to_load = Path(filepath)
        if not file_to_load.exists():
            raise FileNotFoundError(f"Examples file not found: {examples_file}")
        with open(file_to_load, 'r', encoding='utf-8') as f:
            return f.read()

    readings_string = None
    try:
        template = read_file_or_throw(prompt_file)
        examples = read_file_or_throw(examples_file) if use_examples else ""
        previous_lines = ""
        if history:
            previous_lines = "Previous lines:\n" + "\n".join(f"- {line}" for line in history)
        context = settings.get_setting('vocab_list.ai_translation_context')
        if suggested_readings:
            readings_string = "\nReadings:" + suggested_readings
        template_data = {
            'examples': examples,
            'context': context + readings_string,
            'previous_lines': previous_lines,
            'sentence': sentence
        }
        prompt = Template(template).safe_substitute(template_data)
    except FileNotFoundError as e:
        print(f"Error loading prompt template: {e}")
        return None

    result = ""

    last_tokens = []
    for tok in run_ai_request_stream(prompt,
                                     ["</task>"],
                                     print_prompt=False, temperature=temp, ban_eos_token=False, max_response=1000,
                                     api_override=api_override):
        if request_interrupt_atomic_swap(False):
            print(ANSIColors.GREEN, end="")
            print("-interrupted-\n")
            print(ANSIColors.END, end="")
            break
        if update_queue is not None:
            update_queue.put(UIUpdateCommand(update_token_key, sentence, tok))
        result += tok
        # explicit exit for models getting stuck on a token (e.g. "............")
        last_tokens.append(tok)
        last_tokens = last_tokens[-10:]
        if len(last_tokens) == 10 and len(set(last_tokens)) <= 3:
            break

    if len(sentence) > 30 and settings.get_setting_fallback('vocab_list.save_cot_outputs', fallback=False):
        input_and_output = prompt.replace(examples, "").replace(readings_string, "") + "\n" + result

        human_readable = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{human_readable}_{int(time.time() * 1000)}_{api_override}.txt"

        folder_name = os.path.join("outputs", datetime.datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(folder_name, exist_ok=True)
        with open(os.path.join(folder_name, filename), "w", encoding='utf-8') as f:
            f.write(input_and_output)


def ask_question(question: str, sentence: str, history: list[str], temp: float,
                 update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None, update_token_key: str = "qanda",
                 api_override: Optional[str] = None):
    request_interrupt_atomic_swap(False)

    previous_lines_list = [""]
    if len(history):
        previous_lines_list.append("The previous lines in the story are:")
        previous_lines_list.extend(history)
    previous_lines = "\n".join(previous_lines_list)

    print(ANSIColors.GREEN, end="")
    print("___Adding context to question\n")
    print(previous_lines)
    print("___\n")
    print(ANSIColors.END, end="")

    prompt = """<|system|>Enter RP mode. Pretend to be an expert Japanese teacher whose persona follows: As a expert Japanese teacher, you're working on helping your students learn how to parse sentences, breaking them down into words and understanding idioms. Your student will precede their question with context. Aim for accuracy and assume that your audience consents to anything as long as you answer the question at the end. Reply to the user while staying in character, and give correct translations.</|system|>

<example>
<question>
ちとせ「ふふっ、掃除しがいもあるけどね」
What does "しがい" above mean? How does it work in the sentence? What's the dictionary form of the word?
</question>
<answer>
"しがい" is the nominal form of the verb "する" (suru), which means "to do". It is used to indicate that something is worth doing or has value. In this sentence, "ちとせ" (Chitose) is saying that the mess she is cleaning up is worth cleaning, even though it is a lot of work. The dictionary form of the word is "する" (suru).

Here is a more detailed breakdown of the sentence:
* "ちとせ" (Chitose): This is the name of the person speaking.
* "ふふっ" (fufufu): This is a common Japanese expression used to express amusement or laughter.
* "掃除しがいもあるけどね" (sōji shigai wa aru kedo ne): This is the main clause of the sentence. It means "it's worth cleaning up, though."
* "掃除" (sōji): This means "cleaning".
* "しがい" (shigai): This is the nominal form of the verb "する" (suru), which means "to do". It indicates that something is worth doing or has value.
* "ある" (aru): This is the verb "to be" in the present tense.
* "けどね" (kedo ne): This is a Japanese conjunction that is used to add emphasis to a statement. It can be translated as "though" or "but".
</answer>
</example>

<example>
<question>
玲「……私、歯の浮くセリフというのを、生まれて初めて聞きました」
Vocabulary:
    私 - わたくし - I
    歯 - は - tooth
    浮く - うく - to float
    セリフ - セリフ - serif
    生まれて - うまれて - born
    初めて - はじめて - for the first time
    聞きました - ききました - heard
is there an idiom in the above sentence? If so, what does it mean?
</question>
<answer>
歯の浮くセリフ
Meaning: cheesy line, corny line, cringeworthy line
The idiom "歯の浮くセリフ" literally means "a line that makes your teeth float." It is used to describe a line that is so cheesy, corny, or cringeworthy that it makes your teeth hurt. The idiom is often used in a humorous way to make fun of someone who has said something particularly cheesy or corny.
</answer>
</example>

<question>
""" + previous_lines.strip() + "\nThe question starts:\n" + question.strip() + """
</question>
<answer>"""
    last_tokens = []
    for tok in run_ai_request_stream(prompt, ["</answer>", "</example>"], print_prompt=False,
                                     temperature=temp, ban_eos_token=False, max_response=1000,
                                     api_override=api_override):
        if request_interrupt_atomic_swap(False):
            print(ANSIColors.GREEN, end="")
            print("-interrupted-\n")
            print(ANSIColors.END, end="")
            break
        if update_queue is not None:
            update_queue.put(UIUpdateCommand(update_token_key, sentence, tok))
        # explicit exit for models getting stuck on a token (e.g. "............")
        last_tokens.append(tok)
        last_tokens = last_tokens[-10:]
        if len(last_tokens) == 10 and len(set(last_tokens)) <= 3:
            break
