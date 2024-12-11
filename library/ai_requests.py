import requests
import json
import os
from typing import Optional
import sseclient
import google.generativeai as google_genai

from library.settings_manager import settings, ROOT_FOLDER
from library.token_count import get_token_count


class EmptyResponseException(ValueError):
    pass


def run_ai_request(prompt: str, custom_stopping_strings: Optional[list[str]] = None, temperature: float = .1,
                   clean_blank_lines: bool = True, max_response: int = 1536, ban_eos_token: bool = True,
                   print_prompt=True):
    result = ""
    for tok in run_ai_request_stream(prompt, custom_stopping_strings, temperature, max_response,
                                               ban_eos_token, print_prompt):
        result += tok
    if clean_blank_lines:
        result = "\n".join([l for l in "".join(result).splitlines() if len(l.strip()) > 0])
    if result.endswith("</s>"):
        result = result[:-len("</s>")]
    return result


def run_ai_request_stream(prompt: str, custom_stopping_strings: Optional[list[str]] = None, temperature: float = .1,
                          max_response: int = 1536, ban_eos_token: bool = True, print_prompt=True,
                          api_override: Optional[str] = None):
    api_choice = settings.get_setting('ai_settings.api')
    if api_override:
        api_choice = api_override
    match api_choice:
        case "Oogabooga":
            for tok in run_ai_request_ooba(prompt, custom_stopping_strings, temperature, max_response, ban_eos_token,
                                           print_prompt):
                yield tok
        case "Gemini":
            yield run_ai_request_gemini_pro(prompt, custom_stopping_strings, temperature, max_response)
        case _:
            raise ValueError(f"{api_choice} is unsupported for the setting ai_settings.api")


def run_ai_request_ooba(prompt: str, custom_stopping_strings: Optional[list[str]] = None, temperature: float = .1,
                        max_response: int = 1536, ban_eos_token: bool = True, print_prompt=True):
    request_url = settings.get_setting('oobabooga_api.request_url')
    max_context = settings.get_setting('oobabooga_api.context_length')
    if not custom_stopping_strings:
        custom_stopping_strings = []
    prompt_length = get_token_count(prompt)
    if prompt_length + max_response > max_context:
        raise ValueError(f"run_ai_request: the prompt ({prompt_length}) and response length ({max_response}) are "
                         f"longer than max context! ({max_context})")

    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        'temperature': temperature,
        "max_tokens": max_response,
        'truncation_length': max_context - max_response,
        'stop': custom_stopping_strings,
        'ban_eos_token': ban_eos_token,
        "stream": True,
    }
    preset = settings.get_setting('oobabooga_api.preset_name')
    if preset.lower() not in ['', 'none']:
        data['preset'] = preset
    else:
        extra_settings = {
            'min_p': 0.05,
            'top_k': 0,
            'repetition_penalty': 1.05,
            'repetition_penalty_range': 1024,
            'typical_p': 1,
            'tfs': 1,
            'top_a': 0,
            'epsilon_cutoff': 0,
            'eta_cutoff': 0,
            'guidance_scale': 1,
            'negative_prompt': '',
            'penalty_alpha': 0,
            'mirostat_mode': 0,
            'mirostat_tau': 5,
            'mirostat_eta': 0.1,
            'temperature_last': False,
            'do_sample': True,
            'seed': -1,
            'encoder_repetition_penalty': 1,
            'no_repeat_ngram_size': 0,
            'min_length': 0,
            'num_beams': 1,
            'length_penalty': 1,
            'early_stopping': False,
            'add_bos_token': False,
            'skip_special_tokens': True,
            'top_p': 0.98,
        }
        data.update(extra_settings)

    stream_response = requests.post(request_url, headers=headers, json=data, verify=False, stream=True)
    client = sseclient.SSEClient(stream_response)

    result = ""
    if print_prompt:
        print(data['prompt'], end='')
    with open(os.path.join(ROOT_FOLDER, "response.txt"), "w", encoding='utf-8') as f:
        for event in client.events():
            payload = json.loads(event.data)
            new_text = payload['choices'][0]['text']
            print(new_text, end='')
            f.write(new_text)
            result += new_text
            yield new_text
    print()


def run_ai_request_gemini_pro(prompt: str, custom_stopping_strings: Optional[list[str]] = None, temperature: float = .1,
                              max_response: int = 1536):
    prompt = ("Respond directly with only the requested information. "
              "Do not add any conversational elements, greetings, or explanations. "
              "You are a translation engine. " + prompt)
    google_genai.configure(api_key=settings.get_setting('gemini_pro_api.api_key'))
    model = google_genai.GenerativeModel(settings.get_setting('gemini_pro_api.api_model'),
                                         # Lower safety settings since we're processing public YouTube content
                                         safety_settings={
                                              "harassment": "block_none",
                                              "hate_speech": "block_none",
                                              "sexually_explicit": "block_none",
                                              "dangerous": "block_none",
                                         },
                                         # Add generation config
                                         generation_config={
                                              "temperature": temperature,
                                              "stop_sequences": custom_stopping_strings,
                                              "max_output_tokens": max_response,
                                         })
    response = model.generate_content(prompt)
    return response.text
