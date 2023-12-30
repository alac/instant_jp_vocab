# instant_jp_vocab
Monitor the clipboard and generate vocabulary lists for Japanese sentences.
Useful with a text hooking utility like Textractor.

## Setup
1. Install with `pip install -r requirements.txt` and `python -m unidic download`.
2. Edit `settings.toml` to enable what you want.
3. For AI behaviors, install Oobabooga's Text-Generation-WebUI and enable the API. *Use an AI that has decent Japanese performance and sanity check responses.*

## Credits
For definitions and katakana readings in non-ML mode, we use a modified [Jitendex](https://github.com/stephenmk/Jitendex), which is under the [ShareAlike license](https://creativecommons.org/licenses/by-sa/4.0/).

For breaking sentences into words, we use [fugashi](https://github.com/polm/fugashi).