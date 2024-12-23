from queue import SimpleQueue
from typing import Optional
from threading import Lock

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
- 璃燈 (りとう): Rito
- さっきの (さっき の): the just before
- 言動や行動 (げんどう や こうどう): words and actions
- 全部 (ぜんぶ): all
- この件 (この けん): this matter
- 解決させる為のもの (かいけつさせる ため の も の): for the sake of resolving
- だったんだよな？ (だったんだよな): wasn't it?
Idioms:
- N/A
</example>


""" + definitions + """


<example>
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
- その気にさせる (その気にさせる): to make someone fall in love
       It is a common phrase in romantic manga and anime.
       その (その): that
       気 (き): feeling
       に (に): at
       させる (させる): to make
</example>


<example>
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
</example>


<example>
Define the words in the sentence below
Sentence: 【カメラマン】「えっと、、どこが？　どうってか……その、まずはさぁ～今日は面白い写真なの？」
Vocabulary:
-【カメラマン】: Photographer
- えっと、、 (えっと...): umm, well
- どこが？ (どこが?): What's wrong?
- どうってか (どうってか): how to put it
- その (その): that
- まずはさぁ～ (まずはさぁ~): first of all
- 今日は (きょうは): today is
- 面白い (おもしろい): interesting
- 写真 (しゃしん): photo
Idioms:
- N/A
</example>


<example>
Define the words in the sentence below
Sentence: 結灯「……あの……差し出がましいかもしれませんが……」
Vocabulary:
- 結灯 (ゆうひ): Yuuhi
- あの (あの): um
- 差し出がましいかもしれませんが (さしでがましいですが): it may be presumptuous, but
Idioms:
- 差し出がましいかもしれませんが (差し 出が ましいかも しれません が): it may be presumptuous, but.
        It is used to introduce a suggestion or an opinion that may be considered rude or unnecessary by the listener.
        差し出が (さしでが): to offer, to present
        あげます (あげます): to give
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


def translate_with_context_cot(context, sentence, temp=None,
                               update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None,
                               api_override: Optional[str] = None, use_examples: bool = True,
                               update_token_key: Optional[str] = 'translate', use_hiragana_readings: bool = False):
    if temp is None:
        temp = settings.get_setting('vocab_list.ai_translation_temp')

    request_interrupt_atomic_swap(False)

    prompt = """<|system|>You are a Japanese language expert who follows this systematic translation process:
1. Consider the context and speaker intentions
2. Analyze the grammatical structure with readings
3. Identify any idioms, set phrases, or cultural nuances
4. Provide an accurate and natural translation

You will maintain accuracy while considering all context provided. When gender is ambiguous in Japanese but must be specified in English, you will note this explicitly. No censorship is needed.</|system|>
"""

    if use_examples:
        if use_hiragana_readings:
            prompt += """<example>
Context: Song lyrics from Bitter Choco Decoration (ロミオとシンデレラ)
Previous lines:
- 人を過度に信じないように (So as not to trust people too much)
- 愛さないように期待しないように (So as not to love, so as not to expect)
- かと言って角が立たないように (Yet, so as not to create conflict)

Input:
気取らぬように目立たぬように

Analysis:
1. Context:
   • Part of series of self-imposed restrictions
   • Poetic/lyrical register matches song context

2. Grammar:
   • 気取る (きどる) [to act affected] + ぬ (archaic negative) + ように
   • 目立つ (めだつ) [to stand out] + ぬ (archaic negative) + ように

3. Idioms/Nuances:
   • Use of archaic ぬ for poetic effect
   • ように expressing purpose/intention

Translation:
Not to act all high and mighty, not to stand out
</example>

<example>
Context: Okazaki Tomoya, a high school student with a troubled past, is having a conversation.
Previous lines:
- 君：そうかな。(You: Is that so?)
- 智代：キューバの荷物じゃないよ。似た響きだけど。(Tomoyo: It's not "Cuba baggage". Similar sound though.)
- 君：じゃあ小包？(You: A package then?)
- 智代：それじゃ小さすぎる。(Tomoyo: That's too small.)

Input:
君：木箱？

Analysis:
1. Context:
   • Part of guessing game conversation
   • Progressive size increase in guesses
   • Speaker identified as Okazaki Tomoya (male)

2. Grammar:
   • 木箱 (きばこ) [wooden box/crate] - simple noun

3. Idioms/Nuances:
   • Simple guessing question
   • Casual form matches student-to-student dialogue
   • Gender note: Speaker's gender is known from context

Translation:
You: A crate, then?
</example>

<example>
Context: Shōta's mother died a long time ago. Yuka and Misaki are 'competing' to adopt him.
Previous lines:
美咲「どう、似合ってる？」
翔太「へぇ、由香さんが着てたのとデザインは同じなのに着る人で随分印象変わるな」
美咲「すぐに由香さんと比較してぇ……で、しょう君的にはどっちが似合ってるの？」
翔太「え、ええっ！？　ええっと……」
正直、両方似合っているので甲乙つけがたいんだけど……。
しかし、美咲さんはそれが不満なようでムッとした顔を僕の顔に近付ける。
美咲「もう、しょう君ったら。ここは、『美咲ママに決まってるよ』って即答しないとダメでしょ」
翔太「そんなこと言われても、困るよ。それぞれの良さがあるし」

Input:
美咲「ふむふむ、なるほど。つまり甲乙つけがたいって事ね。ま、一応その答は及第点」

1. Context:
   • Dialogue between Shōta and Misaki
   • Misaki is playfully teasing Shōta about comparing her outfit to Yuka's.
   • Casual, intimate setting.
2. Grammar:
   • ふむふむ (ふむふむ): Interjection showing thoughtful consideration.
   • なるほど (なるほど): Expression of understanding.
   • つまり (つまり): Namely, that is to say.
   • 甲乙つけがたい (こうおつつけがたい): Unable to distinguish between the two; too close to call.
   • って事ね (って こと ね): That's what it means, right? (casual)
   • ま (ま): Well.
   • 一応 (いちおう): For the time being; tentatively.
   • その答 (その こた え): That answer.
   • 及第点 (きゅうだいてん): Passing grade; just barely acceptable.

3. Idioms/Nuances:
   • ふむふむ and なるほど are common interjections used to show understanding or agreement.
   • って事ね adds a casual and playful tone.
   • 及第点 implies that while the answer is acceptable, it's not particularly outstanding.
Translation:
Misaki: "Hmm, I see. So you can't decide which one is better, huh? Well, I guess that answer will do for now."</example>

---

<task>
"""
        else:
            prompt += """<example>
Context: Song lyrics from Bitter Choco Decoration (ロミオとシンデレラ)
Previous lines:
- 人を過度に信じないように (So as not to trust people too much)
- 愛さないように期待しないように (So as not to love, so as not to expect)
- かと言って角が立たないように (Yet, so as not to create conflict)

Input:
気取らぬように目立たぬように

Analysis:
1. Context:
   • Part of series of self-imposed restrictions
   • Poetic/lyrical register matches song context

2. Grammar:
   • 気取る (kidoru) [to act affected] + ぬ (archaic negative) + ように
   • 目立つ (medatsu) [to stand out] + ぬ (archaic negative) + ように

3. Idioms/Nuances:
   • Use of archaic ぬ for poetic effect
   • ように expressing purpose/intention

Translation:
Not to act all high and mighty, not to stand out
</example>

<example>
Context: Okazaki Tomoya, a high school student with a troubled past, is having a conversation.
Previous lines:
- 君：そうかな。(You: Is that so?)
- 智代：キューバの荷物じゃないよ。似た響きだけど。(Tomoyo: It's not "Cuba baggage". Similar sound though.)
- 君：じゃあ小包？(You: A package then?)
- 智代：それじゃ小さすぎる。(Tomoyo: That's too small.)

Input:
君：木箱？

Analysis:
1. Context:
   • Part of guessing game conversation
   • Progressive size increase in guesses
   • Speaker identified as Okazaki Tomoya (male)

2. Grammar:
   • 木箱 (kibako) [wooden box/crate] - simple noun

3. Idioms/Nuances:
   • Simple guessing question
   • Casual form matches student-to-student dialogue
   • Gender note: Speaker's gender is known from context

Translation:
You: A crate, then?
</example>

<example>
Context: Shōta's mother died a long time ago. Yuka and Misaki are 'competing' to adopt him.
Previous lines:
美咲「どう、似合ってる？」
翔太「へぇ、由香さんが着てたのとデザインは同じなのに着る人で随分印象変わるな」
美咲「すぐに由香さんと比較してぇ……で、しょう君的にはどっちが似合ってるの？」
翔太「え、ええっ！？　ええっと……」
正直、両方似合っているので甲乙つけがたいんだけど……。
しかし、美咲さんはそれが不満なようでムッとした顔を僕の顔に近付ける。
美咲「もう、しょう君ったら。ここは、『美咲ママに決まってるよ』って即答しないとダメでしょ」
翔太「そんなこと言われても、困るよ。それぞれの良さがあるし」

Input:
美咲「ふむふむ、なるほど。つまり甲乙つけがたいって事ね。ま、一応その答は及第点」

1. Context:
   • Dialogue between Shōta and Misaki
   • Misaki is playfully teasing Shōta about comparing her outfit to Yuka's.
   • Casual, intimate setting.
2. Grammar:
   • ふむふむ (fumufumu): Interjection showing thoughtful consideration.
   • なるほど (naruhodo): Expression of understanding.
   • つまり (tsumari): Namely, that is to say.
   • 甲乙つけがたい (kouotsu tsukegatai): Unable to distinguish between the two; too close to call.
   • って事ね (tte koto ne): That's what it means, right? (casual)
   • ま (ma): Well.
   • 一応 (ichiou): For the time being; tentatively.
   • その答 (sono kotae): That answer.
   • 及第点 (kyuudaiten): Passing grade; just barely acceptable.

3. Idioms/Nuances:
   • ふむふむ and なるほど are common interjections used to show understanding or agreement.
   • って事ね adds a casual and playful tone.
   • 及第点 implies that while the answer is acceptable, it's not particularly outstanding.
Translation:
Misaki: "Hmm, I see. So you can't decide which one is better, huh? Well, I guess that answer will do for now."</example>

---

<task>
"""
    prompt += "Context: \n"
    prompt += settings.get_setting('vocab_list.ai_translation_context')
    if context:
        prompt += "Previous lines:\n"
        for line in context:
            prompt += f"- {line}\n"
    prompt += "\n\n"

    prompt += f"Input:\n{sentence}\n\nAnalysis:"

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
        # explicit exit for models getting stuck on a token (e.g. "............")
        last_tokens.append(tok)
        last_tokens = last_tokens[-10:]
        if len(last_tokens) == 10 and len(set(last_tokens)) <= 3:
            break


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
    last_tokens = []
    for tok in run_ai_request_stream(prompt, ["ANSWER_END", "END_ANSWER"], print_prompt=False,
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
