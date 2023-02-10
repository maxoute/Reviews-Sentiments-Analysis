import re
import logging
import urllib.request
from urlextract import URLExtract
from typing import Dict, Sequence

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
)

MODEL_LIST = {
    'emotion': {
        "default": "j-hartmann/emotion-english-distilroberta-base"
    },
    'sentiment': {
        "default": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "multilingual": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    },
    'topic': {
        "default": "cardiffnlp/tweet-topic-21-multi",
    }   
}


def check_network():
    try:
        urllib.request.urlopen('http://google.com')
        return False
    except Exception:
        return True


def get_preprocessor(processor_type: str = None):
    url_ex = URLExtract()

    if processor_type is None:
        def preprocess(text):
            text = re.sub(r"@[A-Z,0-9]+", "@user", text)
            urls = url_ex.find_urls(text)
            for url in urls:
                try:
                    text = text.replace(url, "http")
                except re.error:
                    logging.warning(f're.error:\t - {text}\n\t - {url}')
            return text
    elif processor_type == 'tweet_topic':
        def preprocess(text):
            urls = url_ex.find_urls(text)
            for url in urls:
                text = text.replace(url, "{{URL}}")
            text = re.sub(r"\b(\s*)(@[\S]+)\b", r'\1{\2@}', text)
            return text
    else:
        raise ValueError(f"unknown type: {processor_type}")

    return preprocess


def get_label2id(dataset, label_name='label'):
    label_info = dataset[list(dataset.keys())[0]].features[label_name]
    while True:
        if type(label_info) is Sequence:
            label_info = label_info.feature
        else:
            assert type(label_info) is ClassLabel, f"Error at retrieving label information {label_info}"
            break
    return {name: n for n, name in enumerate(label_info.names)}

def load_model(model, task='sequence_classification', use_auth_token=False, return_dict=False, config_argument=None, model_argument=None, tokenizer_argument=None, model_only=False):
    no_network = check_network()
    model_argument = {} if model_argument is None else model_argument
    model_argument.update({"use_auth_token": use_auth_token})
    tokenizer_argument = {} if tokenizer_argument is None else tokenizer_argument   

    if not no_network:
        found = False
        for task, models in MODEL_LIST.items():
            for model_name, model_id in models.items():
                if model == model_id:   
                    print(f"Model {model} is for task {task} and model_name {model_name}")
                found = True
                break
            if found:
                break
        if not found:
            print(f"Model {model} not found in MODEL_LIST")
            
        #if model not in MODEL_LIST:
            #raise ValueError("model not supported")
        model_class = MODEL_LIST[task][model_name]
        if use_auth_token and not has_auth_token(model):
            raise ValueError("You don't have authorization to use this model")
        try:
            if task == 'sequence_classification':
                config = model_class.config_class.from_pretrained(model, **config_argument)
                tokenizer = tokenizer_class.from_pretrained(model, **tokenizer_argument)
                model = model_class.from_pretrained(model, config=config, **model_argument)
            elif task == 'question_answering':
                config = model_class.config_class.from_pretrained(model, **config_argument)
                tokenizer = tokenizer_class.from_pretrained(model, **tokenizer_argument)
                model = model_class.from_pretrained(model, config=config, **model_argument)
            else:
                raise ValueError("task not supported")
        except Exception as e:
            raise e
    else:
        raise Exception("No network connection")

    if model_only:
        return model

    if return_dict:
        return {
            "model": model,
            "tokenizer": tokenizer
        }

    return model, tokenizer
