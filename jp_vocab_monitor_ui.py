import argparse
import threading
from queue import SimpleQueue, Empty
import os.path
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import json
import pyperclip

from library.settings_manager import settings
from jp_vocab_clipboard_monitor import (should_generate_vocabulary_list, UIUpdateCommand, run_vocabulary_list,
                                        translate_with_context, request_interrupt_atomic_swap, ANSIColors, ask_question)


class MonitorCommand:
    def __init__(self, command_type: str, sentence: str, history: list[str], prompt: str = None):
        self.command_type = command_type
        self.sentence = sentence
        self.history = history
        self.prompt = prompt


class JpVocabUI:
    def __init__(self, source: str):
        self.text_output_scrolledtext = None
        self.get_definitions_button = None
        self.retry_translation_button = None
        self.ask_question_button = None
        self.retry_button = None
        self.toggle_monitor_button = None

        self.source = source  # the name of the config

        self.command_queue = SimpleQueue()
        self.ui_update_queue = SimpleQueue()
        self.last_command = None

        # ui data
        self.ui_sentence = ""
        self.ui_monitor_is_enabled = True
        self.ui_translation = ""
        self.ui_definitions = ""
        self.ui_question = ""
        self.ui_response = ""
        self.last_textfield_value = ""

        # monitor data
        self.history = []
        self.history_length = -1
        self.previous_clipboard = ""

        # synchronization
        self.locked_sentence = ""
        self.sentence_lock = threading.Lock()

        # initialize
        cache_file = os.path.join("translation_history", f"{self.source}.json")
        if os.path.isfile(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
        self.history_length = settings.get_setting('vocab_list.ai_translation_history_length')

    def start(self):
        self.start_processing_thread()
        self.start_ui()

    def start_processing_thread(self):
        thread = threading.Thread(target=self.processing_thread, args=(self.command_queue,))
        thread.daemon = True
        thread.start()

    def processing_thread(self, queue: SimpleQueue[MonitorCommand]):
        while True:
            command = queue.get(block=True)  # type: MonitorCommand
            try:
                with self.sentence_lock:
                    latest_sentence = self.locked_sentence
                    if command.sentence != latest_sentence:
                        continue
                    self.last_command = command
                if command.command_type == "translate":
                    translate_with_context(command.history, command.sentence, update_queue=self.ui_update_queue)
                    self.ui_update_queue.put(UIUpdateCommand("translate", command.sentence, "\n"))
                if command.command_type == "define":
                    temp = settings.get_setting('vocab_list.ai_definitions_augmented_temp')
                    run_vocabulary_list(command.sentence, temp=temp, use_dictionary=True,
                                        update_queue=self.ui_update_queue)
                if command.command_type == "qanda":
                    temp = settings.get_setting('vocab_list.ai_definitions_augmented_temp')
                    ask_question(command.prompt, command.sentence, temp=temp, update_queue=self.ui_update_queue)
            except Empty:
                pass

    def start_ui(self):
        # Create the root window
        root = tk.Tk()

        root.geometry("{}x{}+0+0".format(655, 500))
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)

        self.toggle_monitor_button = tk.Button(
            root, text="toggle_monitor", command=self.toggle_monitor
        )
        self.toggle_monitor_button.grid(row=0, column=0)
        self.retry_translation_button = tk.Button(
            root, text="get_translation", command=self.trigger_translation
        )
        self.retry_translation_button.grid(row=0, column=1)
        self.get_definitions_button = tk.Button(
            root, text="get_definitions", command=self.get_definitions
        )
        self.get_definitions_button.grid(row=0, column=2)
        self.ask_question_button = tk.Button(
            root, text="ask_question", command=self.ask_question
        )
        self.ask_question_button.grid(row=0, column=3)
        self.retry_button = tk.Button(
            root, text="retry", command=self.retry
        )
        self.retry_button.grid(row=0, column=4)

        self.text_output_scrolledtext = ScrolledText(root)
        self.text_output_scrolledtext.grid(row=1, column=0, columnspan=5, sticky="nsew")

        # Run the Tkinter event loop
        root.after(200, lambda: self.update_status(root))
        root.mainloop()

    def toggle_monitor(self):
        self.ui_monitor_is_enabled = not self.ui_monitor_is_enabled

    def trigger_translation(self):
        self.ui_translation = ""
        self.command_queue.put(MonitorCommand("translate", self.ui_sentence, self.history[:]))
        self.command_queue.put(MonitorCommand("translate", self.ui_sentence, self.history[:]))
        self.command_queue.put(MonitorCommand("translate", self.ui_sentence, self.history[:]))

    def get_definitions(self):
        self.ui_definitions = ""
        self.command_queue.put(MonitorCommand("define", self.ui_sentence, []))

    def ask_question(self):
        self.ui_question = self.text_output_scrolledtext.get("1.0", tk.END)
        self.ui_response = ""
        self.command_queue.put(MonitorCommand("qanda", self.ui_sentence, [], self.ui_question))

    def retry(self):
        with self.sentence_lock:
            if self.last_command:
                if self.last_command.command_type == "translate":
                    self.ui_translation = ""
                if self.last_command.command_type == "define":
                    self.ui_definitions = ""
                if self.last_command.command_type == "qanda":
                    self.ui_response = ""
                self.command_queue.put(self.last_command)

    def update_status(self, root: tk.Tk):
        self.check_clipboard()

        try:
            while True:
                self.consume_update()
        except Empty:
            pass

        self.update_ui()
        root.after(1, lambda: self.update_status(root))

    def check_clipboard(self):
        if not self.ui_monitor_is_enabled:
            return

        current_clipboard = pyperclip.paste()
        if current_clipboard != self.previous_clipboard:
            if should_generate_vocabulary_list(sentence=current_clipboard):
                if not any([(current_clipboard in previous or previous in current_clipboard) for previous in
                            self.history]):
                    self.history.append(current_clipboard)

                request_interrupt_atomic_swap(True)

                self.ui_sentence = current_clipboard
                self.ui_translation = ""
                self.ui_definitions = ""
                self.ui_question = ""
                self.ui_response = ""
                self.last_textfield_value = None
                with self.sentence_lock:
                    self.locked_sentence = current_clipboard

                print(ANSIColors.BOLD, end="")
                print("New sentence: ", current_clipboard)
                print(ANSIColors.END, end="")

                self.trigger_translation()

                self.history = self.history[-self.history_length:]
                cache_file = os.path.join("translation_history", f"{self.source}.json")
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.history, f, indent=2)
            else:
                print(ANSIColors.BOLD, end="")
                print("Skipping sentence: ", current_clipboard.encode('utf-8', 'ignore').decode('utf-8'))
                print(ANSIColors.END, end="")

        self.previous_clipboard = current_clipboard

    def consume_update(self):
        # throws Empty if no elements
        update_command = self.ui_update_queue.get(False)  # type: UIUpdateCommand
        if update_command.sentence == self.ui_sentence:
            if update_command.update_type == "translate":
                self.ui_translation += update_command.token
            if update_command.update_type == "define":
                self.ui_definitions += update_command.token
            if update_command.update_type == "qanda":
                self.ui_response += update_command.token

    def update_ui(self):
        if not self.ui_monitor_is_enabled:
            textfield_value = "Monitoring is disabled!"
        else:
            if len(self.ui_question) > 0:
                textfield_value = f"{self.ui_question}\n{self.ui_response}"
            else:
                textfield_value = f"{self.ui_sentence.strip()}\n\n{self.ui_translation.strip()}\n{self.ui_definitions}"
        if self.last_textfield_value is None or self.last_textfield_value != textfield_value:
            self.text_output_scrolledtext.delete("1.0", tk.END)  # Clear current contents.
            self.text_output_scrolledtext.insert(tk.INSERT, textfield_value)
            self.last_textfield_value = textfield_value


if __name__ == '__main__':
    source = None

    if not source:
        parser = argparse.ArgumentParser()
        parser.add_argument("source",
                            help="The name associated with each 'translation history'. Providing a unique name for each"
                            " allows for tracking each translation history separately when switching sources.",
                            type=str)
        args = parser.parse_args()
        source = args.source

    source_settings_path = os.path.join("settings", f"{source}.toml")
    if os.path.isfile(source_settings_path):
        settings.override_settings(source_settings_path)

    monitor_ui = JpVocabUI(source)
    monitor_ui.start()
