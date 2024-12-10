import argparse
import threading
from queue import SimpleQueue, Empty
import os.path
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import json
import pyperclip
from typing import Optional
import re
import time

from library.settings_manager import settings
from jp_vocab_clipboard_monitor import (should_generate_vocabulary_list, UIUpdateCommand, run_vocabulary_list,
                                        translate_with_context, request_interrupt_atomic_swap, ANSIColors, ask_question)

CLIPBOARD_CHECK_LATENCY_MS = 250


class MonitorCommand:
    def __init__(self, command_type: str, sentence: str, history: list[str], prompt: str = None,
                 temp: Optional[float] = None, style: str = None, index: int = 0):
        self.command_type = command_type
        self.sentence = sentence
        self.history = history
        self.prompt = prompt
        self.temp = temp
        self.style = style
        self.index = index


class HistoryState:
    def __init__(self, sentence, translation, translation_validation, definitions, question, response,
                 history):
        self.ui_sentence = sentence
        self.ui_translation = translation
        self.ui_translation_validation = translation_validation
        self.ui_definitions = definitions
        self.ui_question = question
        self.ui_response = response
        self.history = history


class JpVocabUI:
    def __init__(self, source: str):
        self.tk_root = None
        self.text_output_scrolledtext = None
        self.get_definitions_button = None
        self.retry_translation_button = None
        self.ask_question_button = None
        self.retry_button = None
        self.stop_button = None
        self.toggle_monitor_button = None
        self.switch_view_button = None

        self.source = source  # the name of the config

        self.command_queue = SimpleQueue()
        self.ui_update_queue = SimpleQueue()
        self.last_command = None

        self.last_clipboard_ts = 0

        # ui data state
        self.ui_sentence = ""
        self.ui_translation = ""
        self.ui_translation_validation = ""
        self.ui_definitions = ""
        self.ui_question = ""
        self.ui_response = ""
        # transient ui state
        self.last_textfield_value = ""
        self.ui_monitor_is_enabled = True
        self.show_qanda = False

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

        self.history_states = []  # type: list[HistoryState]
        self.history_states_index = -1

    def start_ui(self):
        root = tk.Tk()
        self.tk_root = root

        root.geometry("{}x{}+0+0".format(655, 500))
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Create menu bar frame
        menu_bar = tk.Frame(root)
        menu_bar.grid(row=0, column=0, columnspan=6, sticky="ew")
        menu_bar.grid_columnconfigure(1, weight=1)  # Make middle section expandable

        # Left side navigation buttons
        nav_frame = tk.Frame(menu_bar)
        nav_frame.grid(row=0, column=0, sticky="w")

        prev_button = tk.Button(
            nav_frame,
            text="⬅️",
            command=self.go_to_previous,
            font=('TkDefaultFont', 12)
        )
        prev_button.pack(side=tk.LEFT, padx=2)

        next_button = tk.Button(
            nav_frame,
            text="➡️",
            command=self.go_to_next,
            font=('TkDefaultFont', 12)
        )
        next_button.pack(side=tk.LEFT, padx=2)

        # Middle buttons
        buttons_frame = tk.Frame(menu_bar)
        buttons_frame.grid(row=0, column=1, sticky="ew")

        self.toggle_monitor_button = tk.Button(
            buttons_frame, text="toggle_monitor", command=self.toggle_monitor
        )
        self.toggle_monitor_button.pack(side=tk.LEFT)

        self.retry_translation_button = tk.Button(
            buttons_frame, text="get_translation", command=self.trigger_translation
        )
        self.retry_translation_button.pack(side=tk.LEFT)

        self.get_definitions_button = tk.Button(
            buttons_frame, text="get_definitions", command=self.get_definitions
        )
        self.get_definitions_button.pack(side=tk.LEFT)

        self.ask_question_button = tk.Button(
            buttons_frame, text="ask_question", command=self.ask_question
        )
        self.ask_question_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(
            buttons_frame, text="stop", command=self.stop
        )
        self.stop_button.pack(side=tk.LEFT)

        self.retry_button = tk.Button(
            buttons_frame, text="retry", command=self.retry
        )
        self.retry_button.pack(side=tk.LEFT)

        self.switch_view_button = tk.Button(
            buttons_frame, text="switch_view", command=self.switch_view
        )
        self.switch_view_button.pack(side=tk.LEFT)

        # Right side history button
        history_button = tk.Button(
            menu_bar,
            text="📋",
            command=self.show_history,
            font=('TkDefaultFont', 12)
        )
        history_button.grid(row=0, column=2, sticky="e", padx=2)

        self.text_output_scrolledtext = ScrolledText(root, wrap="word")
        self.text_output_scrolledtext.grid(row=1, column=0, columnspan=6, sticky="nsew")

        # Run the Tkinter event loop
        root.after(200, lambda: self.update_status(root))
        root.bind("<Shift-Return>", lambda e: self.ask_question())
        root.mainloop()

    # button handlers

    def go_to_previous(self):
        if self.history_states_index <= 0:
            return
        self.stop()
        self.save_history_state()
        self.history_states_index -= 1
        self.load_history_state_at_index(self.history_states_index)
        # ui will be updated on the next update_ui tick

    def go_to_next(self):
        if self.history_states_index < 0:
            return
        if (self.history_states_index + 1) > (len(self.history_states) - 1):
            return
        self.stop()
        self.save_history_state()
        self.history_states_index += 1
        self.load_history_state_at_index(self.history_states_index)
        # ui will be updated on the next update_ui tick

    def save_history_state(self):
        self.history_states[self.history_states_index] = (
            HistoryState(self.ui_sentence, self.ui_translation, self.ui_translation_validation,
                         self.ui_definitions, self.ui_question, self.ui_response, self.history[:]))

    def load_history_state_at_index(self, index):
        history_state = self.history_states[index]  # type: HistoryState
        self.ui_sentence = history_state.ui_sentence
        self.ui_translation = history_state.ui_translation
        self.ui_translation_validation = history_state.ui_translation_validation
        self.ui_definitions = history_state.ui_definitions
        self.ui_question = history_state.ui_question
        self.ui_response = history_state.ui_response
        self.history = history_state.history

    def toggle_monitor(self):
        self.ui_monitor_is_enabled = not self.ui_monitor_is_enabled

    def trigger_translation(self):
        request_interrupt_atomic_swap(True)
        self.ui_translation = ""
        self.ui_translation_validation = ""
        self.show_qanda = False
        self.command_queue.put(MonitorCommand(
            "translate",
            self.ui_sentence,
            self.history[:],
            temp=0,
            index=1))
        self.command_queue.put(MonitorCommand(
            "translate",
            self.ui_sentence,
            self.history[:],
            temp=0,
            style="Aim for a literal translation.",
            index=2),)
        self.command_queue.put(MonitorCommand(
            "translate",
            self.ui_sentence,
            self.history[:],
            temp=0,
            style="Aim for a natural translation.",
            index=3))
        if settings.get_setting_fallback('vocab_list.enable_ai_translation_validation', False):
            self.command_queue.put(MonitorCommand("translation_validation",
                                                  self.ui_sentence,
                                                  self.history[:],
                                                  "",
                                                  temp=0))

    def get_definitions(self):
        request_interrupt_atomic_swap(True)
        self.ui_definitions = ""
        self.show_qanda = False
        self.command_queue.put(MonitorCommand("define", self.ui_sentence, [], temp=0))

    def ask_question(self):
        request_interrupt_atomic_swap(True)
        self.ui_question = self.text_output_scrolledtext.get("1.0", tk.END)
        self.ui_response = ""
        self.show_qanda = True
        self.command_queue.put(MonitorCommand("qanda", self.ui_sentence, self.history[:], self.ui_question, temp=0))

    def retry(self):
        with self.sentence_lock:
            if self.last_command:
                if self.last_command.command_type == "translate":
                    self.ui_translation = ""
                    self.ui_translation_validation = ""
                if self.last_command.command_type == "define":
                    self.ui_definitions = ""
                    self.last_command.temp = settings.get_setting('vocab_list.ai_definitions_temp')
                if self.last_command.command_type == "qanda":
                    self.ui_response = ""
                    self.last_command.temp = settings.get_setting('vocab_list.ai_qanda_temp')
                self.show_qanda = self.last_command.command_type == "qanda"
                self.command_queue.put(self.last_command)

    def stop(self):
        request_interrupt_atomic_swap(True)

    def switch_view(self):
        self.show_qanda = not self.show_qanda

    def show_history(self):
        # Create popup window
        history_window = tk.Toplevel(self.tk_root)
        history_window.title("Translation History")
        history_window.geometry("500x400")
        history_window.grid_rowconfigure(0, weight=1)
        history_window.grid_columnconfigure(0, weight=1)

        # Create text area
        text_area = ScrolledText(history_window, wrap="word")
        text_area.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # Populate text area with history
        text_area.insert("1.0", "\n".join(self.history))

        # Create button frame
        button_frame = tk.Frame(history_window)
        button_frame.grid(row=1, column=0, columnspan=2, pady=5)

        def save_history():
            # Get content and split into lines
            content = text_area.get("1.0", tk.END).strip()
            new_history = [line.strip() for line in content.split("\n") if line.strip()]

            # Update history
            self.history = new_history

            # Save to file
            cache_file = os.path.join("translation_history", f"{self.source}.json")
            os.makedirs("translation_history", exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)

            history_window.destroy()

        def cancel():
            history_window.destroy()

        # Create buttons
        save_button = tk.Button(
            button_frame,
            text="Save",
            command=save_history
        )
        save_button.pack(side=tk.LEFT, padx=5)

        cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=cancel
        )
        cancel_button.pack(side=tk.LEFT, padx=5)

        # Make the window modal
        history_window.transient(self.tk_root)
        history_window.grab_set()
        self.tk_root.wait_window(history_window)

    # threading etc

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
                    if command.command_type != "translation_validation":
                        self.last_command = command

                if command.command_type == "translate":
                    translate_with_context(command.history,
                                           command.sentence,
                                           update_queue=self.ui_update_queue,
                                           temp=command.temp,
                                           index=command.index)
                    self.ui_update_queue.put(UIUpdateCommand("translate", command.sentence, "\n"))
                if command.command_type == "translation_validation":
                    prompt = (f"{self.ui_sentence}\n\n{self.ui_translation}\n\n"
                              f"Which translation is most accurate? Or are they equivalent?")
                    command.prompt = prompt
                    ask_question(command.prompt, command.sentence, command.history, temp=command.temp,
                                 update_queue=self.ui_update_queue, update_token_key="translation_validation")
                if command.command_type == "define":
                    add_readings = settings.get_setting('vocab_list.ai_definitions_add_readings')
                    run_vocabulary_list(command.sentence, temp=command.temp, use_dictionary=add_readings,
                                        update_queue=self.ui_update_queue)
                if command.command_type == "qanda":
                    ask_question(command.prompt, command.sentence, command.history, temp=command.temp,
                                 update_queue=self.ui_update_queue)
            except Empty:
                pass

    def update_status(self, root: tk.Tk):
        current_time_ms = time.time()
        if (current_time_ms - self.last_clipboard_ts)*1000 > CLIPBOARD_CHECK_LATENCY_MS:
            try:
                self.check_clipboard()
                self.last_clipboard_ts = current_time_ms
            except pyperclip.PyperclipWindowsException as e:
                print(ANSIColors.RED, end="")
                print("EXCEPTION!")
                print(e)
                print(ANSIColors.END, end="")

        try:
            while True:
                self.consume_update()
        except Empty:
            pass

        self.update_ui()
        root.after(50, lambda: self.update_status(root))

    def check_clipboard(self):
        if not self.ui_monitor_is_enabled:
            return

        current_clipboard = pyperclip.paste()
        current_clipboard = undo_repetition(current_clipboard)
        if current_clipboard != self.previous_clipboard:
            japanese_detected = should_generate_vocabulary_list(sentence=current_clipboard)
            is_editing_textfield = (current_clipboard in self.last_textfield_value
                                    and self.tk_root.focus_get() == self.text_output_scrolledtext)
            if japanese_detected and not is_editing_textfield:
                if not any([(current_clipboard in previous or previous in current_clipboard) for previous in
                            self.history]):
                    self.history.append(current_clipboard)
                next_sentence = current_clipboard
                request_interrupt_atomic_swap(True)

                if self.history_states:
                    # if the current sentence was the most recent sentence, update its history state before we move on
                    if self.ui_sentence == self.history_states[len(self.history_states) - 1].ui_sentence:
                        history_state = HistoryState(self.ui_sentence, self.ui_translation, self.ui_translation_validation,
                                                     self.ui_definitions, self.ui_question, self.ui_response,
                                                     self.history[:])
                        self.history_states[len(self.history_states) - 1] = history_state

                    # since we could be _anywhere_ in history, snap to the latest history
                    self.history = self.history_states[len(self.history_states) - 1].history

                # a sentence can be split across lines for _dramatic_ purpose, so un-split them if possible
                connectors = [["「", "」",], ["『", "』"]]
                if self.ui_sentence and self.ui_sentence == self.previous_clipboard:
                    for left, right in connectors:
                        if (left in self.previous_clipboard and right not in self.previous_clipboard
                                and right in current_clipboard):
                            (self.previous_clipboard in self.history) and self.history.remove(self.previous_clipboard)
                            (current_clipboard in self.history) and self.history.remove(current_clipboard)
                            next_sentence = self.previous_clipboard + current_clipboard
                            self.history.append(next_sentence)

                self.ui_sentence = next_sentence
                self.ui_translation = ""
                self.ui_definitions = ""
                self.ui_question = ""
                self.ui_response = ""
                self.last_textfield_value = None
                with self.sentence_lock:
                    self.locked_sentence = next_sentence

                print(ANSIColors.BOLD, end="")
                print("New sentence: ", next_sentence)
                print(ANSIColors.END, end="")

                self.trigger_translation()

                self.history = self.history[-self.history_length:]

                # each time we add a new sentence, we add a placeholder for it to HistoryStates
                # we'll overwrite it when the next sentence comes in, OR when we got back/forward
                self.history_states.append(
                    HistoryState(self.ui_sentence, self.ui_translation, self.ui_translation_validation,
                                 self.ui_definitions, self.ui_question, self.ui_response, self.history[:])
                )
                self.history_states_index = len(self.history_states) - 1

                cache_file = os.path.join("translation_history", f"{self.source}.json")
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.history, f, indent=2)
            else:
                print(ANSIColors.BOLD, end="")
                if not japanese_detected:
                    print("Japanese not detected: ", current_clipboard.encode('utf-8', 'ignore').decode('utf-8'))
                if is_editing_textfield:
                    print("Skipping textfield edit.")
                print(ANSIColors.END, end="")

        self.previous_clipboard = current_clipboard

    def consume_update(self):
        # throws Empty if no elements
        update_command = self.ui_update_queue.get(False)  # type: UIUpdateCommand
        if update_command.sentence == self.ui_sentence:
            if update_command.update_type == "translate":
                self.ui_translation += update_command.token
            if update_command.update_type == "translation_validation":
                self.ui_translation_validation += update_command.token
            if update_command.update_type == "define":
                self.ui_definitions += update_command.token
            if update_command.update_type == "qanda":
                self.ui_response += update_command.token

    def update_ui(self):
        if not self.ui_monitor_is_enabled:
            textfield_value = "Monitoring is disabled!"
        else:
            if self.show_qanda:
                textfield_value = f"{self.ui_question.strip()}\n{self.ui_response}"
            else:
                clean_translation = self.ui_translation.strip().replace("\n\n", "\n")
                textfield_value = (f"{self.ui_sentence.strip()}\n\n{clean_translation}\n\n{self.ui_definitions}"
                                   f"\n\n{self.ui_translation_validation}")
        if self.last_textfield_value is None or self.last_textfield_value != textfield_value:
            self.text_output_scrolledtext.delete("1.0", tk.END)  # Clear current contents.
            self.text_output_scrolledtext.insert(tk.INSERT, textfield_value)
            self.last_textfield_value = textfield_value


def undo_repetition(input_string):
    # Function to find the repetition count and return the base character
    def process_group(match):
        group = match.group(0)
        count = len(group)
        char = group[0]

        # Special case for Japanese characters that might be repeated in pairs
        if len(char.encode('utf-8')) > 1:
            if count % 2 == 0:
                return char * 2
            else:
                return char
        else:
            return char

    # Use regex to find consecutive identical characters, including pairs
    pattern = r'(.)(\1+|\1)'

    # Apply the process_group function to each match
    result = re.sub(pattern, process_group, input_string)

    return result


if __name__ == '__main__':
    source_tag = None

    if not source_tag:
        parser = argparse.ArgumentParser()
        parser.add_argument("source",
                            help="The name associated with each 'translation history'. Providing a unique name for each"
                            " allows for tracking each translation history separately when switching sources.",
                            type=str)
        args = parser.parse_args()
        source_tag = args.source

    source_settings_path = os.path.join("settings", f"{source_tag}.toml")
    if os.path.isfile(source_settings_path):
        settings.override_settings(source_settings_path)

    monitor_ui = JpVocabUI(source_tag)
    monitor_ui.start()
