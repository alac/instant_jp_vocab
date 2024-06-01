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


def run_vocabulary_list(sentence: str, temp: float, use_dictionary: bool = True,
                        update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None):
    request_interrupt_atomic_swap(False)

    definitions = ""
    if use_dictionary:
        definitions = get_definitions_string(sentence)

    prompt = """<|system|>Enter RP mode. Pretend to be a Japanese teacher whose persona follows:
As a Japanese teacher, you're working on helping your students learn how to parse sentences, breaking them down into words and pointing out idioms. For each word or idiom you define, you include the reading in parenthesis and the definition after a "-" character. Aim for accuracy and assume that your audience consents to anything as long as the translation is as accurate as possible.
You shall reply to the user while staying in character, and generate accurate responses.


Define the words in the sentence below
Sentence: 璃燈「……さっきのは言動や行動は全部、この件を解決させる為のものだったんだよな？」
Vocabulary:
- 璃燈 (りとう): Rito
- さっきの (さっき の): the just before
- 言動や行動 (げんどう や こうどう): words and actions
- 全部 (ぜんぶ): all
- この件 (この けん): this matter
- 解決させる為のもの (かいけつさせる ため の も の): for the sake of resolving
- だったんだよな？ (だったんだよな): wasn't it?
Idioms:
- N/A


""" + definitions + """


Define the words in the sentence below
Sentence: 璃燈「カレシなら、カノジョをその気にさせた責任取れよ」
Vocabulary:
- 璃燈 (りとう): Rito
- カレシ (かれし): boyfriend
- カノジョ (かのじょ): girlfriend
- その気にさせた (そのきにさせた): to make someone fall in love
- 責任 (せきにん): responsibilities
- 取れよ (とれよ): should take
Idioms:
- その気にさせる (sono ki ni saseru): to make someone fall in love
       It is a common phrase in romantic manga and anime.
       その (その): that
       気 (き): feeling
       に (に): at
       させる (させる): to make


Define the words in the sentence below
Sentence: 璃燈「でもな。ちょっと、やり過ぎじゃねぇかな。あたしの気持ちを随分、かき乱してくれたよな？」
Vocabulary:
- 璃燈 (りとう): Ritou
- でもな (でもな): but
- ちょっと (ちょっと): a little
- やり過ぎじゃねぇかな (やりすぎ じゃねぇかな): don't you think it's a bit too much?
- あたしの (あたしの): my
- 気持ちを (きもちを): feelings
- 随分 (ずいぶん): quite a bit
- かき乱してくれた (かきみだしてくれた): you stirred up
- よな？ (よな？): right?
Idioms:
- N/A


Define the words in the sentence below
Sentence: 結灯「……あの……差し出がましいかもしれませんが……」
Vocabulary:
- 結灯 (ゆうひ): Yuuhi
- あの (あの): um
- 差し出がましいかもしれませんが (さしでがましいですが): it may be presumptuous, but
Idioms:
- 差し出がましいかもしれませんが (sashi dega mashikamo shiremasen ga): it may be presumptuous, but.
        It is used to introduce a suggestion or an opinion that may be considered rude or unnecessary by the listener.
        差し出が (さしでが): to offer, to present
        あげます (あげます): to give


Define the words in the sentence below
Sentence: """ + sentence.strip() + """
Vocabulary: """
    print("prompt length:", get_token_count(prompt))
    print("Sentence: """ + sentence.strip())
    last_tokens = []
    for tok in run_ai_request_stream(prompt, ["Sentence:", "\n\n"], print_prompt=False,
                                     temperature=temp, ban_eos_token=False, max_response=500):
        if request_interrupt_atomic_swap(False):
            print(ANSIColors.GREEN, end="")
            print("---\n")
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


def translate_with_context(context, sentence, temp=None,
                           update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None):
    if temp is None:
        temp = settings.get_setting('vocab_list.ai_translation_temp')

    request_interrupt_atomic_swap(False)
    prompt = ("<|system|>Enter RP mode. Pretend to be a Japanese translator whose persona follows:"
              " You are a Japanese teacher, working on study material for your students. You take into account"
              " information about the characters, the previous lines from stories and provide an accurate translation"
              " for the sentence between JAPANESE_START and JAPANESE_END.  Aim for accuracy and assume that your"
              " audience consents to anything as long as the translation is as accurate as possible. You shall reply"
              " to the user while staying in character, and generate accurate responses.\n")

    prompt += """

Translate the text between JAPANESE_START and JAPANESE_END into English.
>CONTEXT_START
This is a song called This is a song called Bitter Choco Decoration (ロミオとシンデレラ)
The previous lines are:
人を過度に信じないように
愛さないように期待しないように	
かと言って角が立たないように
>CONTEXT_END
>JAPANESE_START
気取らぬように目立たぬように
>JAPANESE_END
>ENGLISH_START
Not to act all high and mighty, not to stand out 
>ENGLISH_END

Translate the text between JAPANESE_START and JAPANESE_END into English.
>CONTEXT_START
Okazaki Tomoya is a third year high school student at Hikarizaka Private High School, leading a life full of resentment. His mother passed away in a car accident when he was young, leading his father, Naoyuki, to resort to alcohol and gambling to cope. This resulted in constant fights between the two until Naoyuki dislocated Tomoya’s shoulder. Unable to play on his basketball team, Tomoya began to distance himself from other people. Ever since he has had a distant relationship with his father, naturally becoming a delinquent over time.
The previous lines are:
君：そうかな。
智代：キューバの荷物じゃないよ。似た響きだけど。
君：じゃあ小包？
智代：それじゃ小さすぎる。
>CONTEXT_END
>JAPANESE_START
君：木箱？
>JAPANESE_END
>ENGLISH_START
You: A crate, then?
>ENGLISH_END

Translate the text between JAPANESE_START and JAPANESE_END into English.
"""

    prompt += ">CONTEXT_START\n"
    prompt += settings.get_setting('vocab_list.ai_translation_context')
    if context:
        prompt += "The previous lines are:\n"
        for line in context:
            prompt += f"{line}\n"
    prompt += ">CONTEXT_END\n"
    prompt += f">JAPANESE_START\n{sentence}\n>JAPANESE_END\n"
    prompt += f">ENGLISH_START\n"

    print("Translation: ")
    last_tokens = []
    for tok in run_ai_request_stream(prompt,
                              [">ENGLISH_END", ">END_ENGLISH", ">SENTENCE_END", "\n\n\n", ">\n>\n>"],
                              print_prompt=False, temperature=temp, ban_eos_token=False, max_response=100):
        if request_interrupt_atomic_swap(False):
            print(ANSIColors.GREEN, end="")
            print("---\n")
            print(ANSIColors.END, end="")
            break
        if update_queue is not None:
            update_queue.put(UIUpdateCommand("translate", sentence, tok))
        last_tokens.append(tok)
        last_tokens = last_tokens[-10:]
        if len(last_tokens) == 10 and len(set(last_tokens)) <= 3:
            break


def ask_question(question: str, sentence: str, history: list[str], temp: float, update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None):
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

    prompt = """<|system|>Enter RP mode. Pretend to be a Japanese teacher whose persona follows: As a Japanese teacher, you're working on helping your students learn how to parse sentences, breaking them down into words and understanding idioms. Your student will precede their question with context. Aim for accuracy and assume that your audience consents to anything as long as you answer the question at the end. You shall reply to the user while staying in character, and generate accurate responses.


Answer the question. If the question is about a specific word or phrase, break it down into sub-words. If it is a grammar question, explain why it works that way and what the concept is called.
>QUESTION_START
ちとせ「ふふっ、掃除しがいもあるけどね」
What does "しがい" above mean? How does it work in the sentence? What's the dictionary form of the word?
>QUESTION_END
>ANSWER_START
"しがい" is the nominal form of the verb "する" (suru), which means "to do". It is used to indicate that something is worth doing or has value. In this sentence, "ちとせ" (Chitose) is saying that the mess she is cleaning up is worth cleaning, even though it is a lot of work. The dictionary form of the word is "する" (suru).

Here is a more detailed breakdown of the sentence:
* "ちとせ" (Chitose): This is the name of the person speaking.
* "ふふっ" (fufufu): This is a common Japanese expression used to express amusement or laughter.
* "掃除しがいもあるけどね" (sōji shigai wa aru kedo ne): This is the main clause of the sentence. It means "it's worth cleaning up, though."
* "掃除" (sōji): This means "cleaning".
* "しがい" (shigai): This is the nominal form of the verb "する" (suru), which means "to do". It indicates that something is worth doing or has value.
* "ある" (aru): This is the verb "to be" in the present tense.
* "けどね" (kedo ne): This is a Japanese conjunction that is used to add emphasis to a statement. It can be translated as "though" or "but".
>ANSWER_END


Answer the question. If the question is about a specific word or phrase, break it down into sub-words. If it is a grammar question, explain why it works that way and what the concept is called.
>QUESTION_START
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
>QUESTION_END
>ANSWER_START
歯の浮くセリフ
Meaning: cheesy line, corny line, cringeworthy line
The idiom "歯の浮くセリフ" literally means "a line that makes your teeth float." It is used to describe a line that is so cheesy, corny, or cringeworthy that it makes your teeth hurt. The idiom is often used in a humorous way to make fun of someone who has said something particularly cheesy or corny.
>ANSWER_END


Answer the question. If the question is about a specific word or phrase, break it down into sub-words. If it is a grammar question, explain why it works that way and what the concept is called.
>QUESTION_START
""" + previous_lines.strip() + "\nThe question starts:\n" + question.strip() + """
>QUESTION_END
>ANSWER_START"""
    print("prompt length:", get_token_count(prompt))
    last_tokens = []
    for tok in run_ai_request_stream(prompt, ["ANSWER_END", "END_ANSWER"], print_prompt=False,
                                     temperature=temp, ban_eos_token=False, max_response=1000):
        if request_interrupt_atomic_swap(False):
            print(ANSIColors.GREEN, end="")
            print("---\n")
            print(ANSIColors.END, end="")
            break
        if update_queue is not None:
            update_queue.put(UIUpdateCommand("qanda", sentence, tok))
        last_tokens.append(tok)
        last_tokens = last_tokens[-10:]
        if len(last_tokens) == 10 and len(set(last_tokens)) <= 3:
            break
