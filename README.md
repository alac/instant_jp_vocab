# instant_jp_vocab
Monitor the clipboard and generate vocabulary lists for Japanese sentences.
Useful with a text hooking utility like Textractor.

## Setup
1. Install with `pip install -r requirements.txt` and `python -m unidic download`.
2. Edit `settings.toml` to enable what you want.
3. For AI behaviors, install Oobabooga's Text-Generation-WebUI and enable the API. *Use an AI that has decent Japanese performance and sanity check responses.*
4. Start the app by entering `python -m jp_vocab_monitor_ui [story_name]`.
(`story_name` is used to track a history of previously seen sentences, providing coherence to the translations.)

## Configuration
You can also configure the program by creating a `user.toml` in the root directory. Then, settings will be loaded from `settings.toml` first, with any overlapping values overridden by `user.toml`.

Per story configuration is also possible by adding a `story.toml` in the `settings` folder.

## Suggested Models
I've seen decent translation quality with the following local models:
[Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
[Mixtral-8x7B-instruct-cosmopedia-japanese20k](https://huggingface.co/aixsatoshi/Mixtral-8x7B-instruct-cosmopedia-japanese20k)

If you're going to _ask_ the AI questions about Japanese, I'd recommend using Google's Gemini Pro via API (Gemini 1.5's accuracy is great; and the free-tier rate limiting should be fine for reading).

## Credits
For definitions and katakana readings in non-ML mode, we use a modified [Jitendex](https://github.com/stephenmk/Jitendex), which is under the [ShareAlike license](https://creativecommons.org/licenses/by-sa/4.0/).

For breaking sentences into words, we use [fugashi](https://github.com/polm/fugashi).