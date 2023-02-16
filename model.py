import logging
import urllib.request
from typing import Dict
from urlextract import URLExtract # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoConfig,\
    AutoModelForMaskedLM
import re
import logging
from typing import List, Dict
import torch
from palettable.colorbrewer.qualitative import Pastel1_7
import spacy
nlp = spacy.load("en_core_web_lg")  

def get_preprocessor(processor_type: str = None):
    url_ex = URLExtract()

    if processor_type is None:
        def preprocess(text):
            text = re.sub(r"@[A-Z,0-9]+", "@user", text)
            urls = url_ex.find_urls(text)
            for _url in urls:
                try:
                    text = text.replace(_url, "http")
                except re.error:
                    logging.warning(f're.error:\t - {text}\n\t - {_url}')
            return text

    elif processor_type == 'tweet_topic':

        def preprocess(tweet):
            urls = url_ex.find_urls(tweet)
            for url in urls:
                tweet = tweet.replace(url, "{{URL}}")
            tweet = re.sub(r"\b(\s*)(@[\S]+)\b", r'\1{\2@}', tweet)
            return tweet
    else:
        raise ValueError(f"unknown type: {processor_type}")

    return preprocess


# def get_label2id(dataset: DatasetDict, label_name: str = 'label'):
#     label_info = dataset[list(dataset.keys())[0]].features[label_name]
#     while True:
#         if type(label_info) is Sequence:
#             label_info = label_info.feature
#         else:
#             assert type(label_info) is ClassLabel, f"Error at retrieving label information {label_info}"
#             break
#     return {k: n for n, k in enumerate(label_info.names)}


def load_model(model: str,
               task: str = 'sequence_classification',
               use_auth_token: bool = False,
               return_dict: bool = False,
               config_argument: Dict = None,
               model_argument: Dict = None,
               tokenizer_argument: Dict = None,
               model_only: bool = False):
    try:
        urllib.request.urlopen('http://google.com')
        no_network = False
    except Exception:
        no_network = True
    model_argument = {} if model_argument is None else model_argument
    model_argument.update({"use_auth_token": use_auth_token, "local_files_only": no_network})

    if return_dict or model_only:
        if task == 'sequence_classification':
            model = AutoModelForSequenceClassification.from_pretrained(model, return_dict=return_dict, **model_argument)
        elif task == 'token_classification':
            model = AutoModelForTokenClassification.from_pretrained(model, return_dict=return_dict, **model_argument)
        elif task == 'masked_language_model':
            model = AutoModelForMaskedLM.from_pretrained(model, return_dict=return_dict, **model_argument)
        else:
            raise ValueError(f'unknown task: {task}')
        return model
    config_argument = {} if config_argument is None else config_argument
    config_argument.update({"use_auth_token": use_auth_token, "local_files_only": no_network})
    config = AutoConfig.from_pretrained(model, **config_argument)

    tokenizer_argument = {} if tokenizer_argument is None else tokenizer_argument
    tokenizer_argument.update({"use_auth_token": use_auth_token, "local_files_only": no_network})
    tokenizer = AutoTokenizer.from_pretrained(model, **tokenizer_argument)

    model_argument.update({"config": config})
    if task == 'sequence_classification':
        model = AutoModelForSequenceClassification.from_pretrained(model, **model_argument)
    elif task == 'token_classification':
        model = AutoModelForTokenClassification.from_pretrained(model, **model_argument)
    elif task == 'masked_language_model':
        model = AutoModelForMaskedLM.from_pretrained(model, **model_argument)
    else:
        raise ValueError(f'unknown task: {task}')
    return config, tokenizer, model

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
# Modeling

class Classifier:
    def __init__(self,
                 model_name: str = None,
                 max_length: int = None,
                 multi_label: bool = False,
                 use_auth_token: bool = False,
                 loaded_model_config_tokenizer: Dict = None):
        if loaded_model_config_tokenizer is not None:
            assert all(i in loaded_model_config_tokenizer.keys() for i in ['model', 'config', 'tokenizer'])
            self.config = loaded_model_config_tokenizer['config']
            self.tokenizer = loaded_model_config_tokenizer['tokenizer']
            self.model = loaded_model_config_tokenizer['model']
        else:
            assert model_name is not None, "model_name is required"
            logging.debug(f'loading {model_name}')
            self.config, self.tokenizer, self.model = load_model(
                model_name, task='sequence_classification', use_auth_token=use_auth_token)
        self.max_length = max_length
        self.multi_label = multi_label
        self.id_to_label = {str(v): k for k, v in self.config.label2id.items()}
        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = torch.cuda.device_count() > 1
        if self.parallel:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        logging.debug(f'{torch.cuda.device_count()} GPUs are in use')

        self.model.eval()
        self.preprocess = get_preprocessor()

    def predict(self,
                #hypothesis_template: str or List,
                text: str or List,
                batch_size: int = None,
                return_probability: bool = True,
                skip_preprocess: bool = False):
        single_input_flag = type(text) is str
        text = [text] if single_input_flag else text
        #hypothesis_template = [hypothesis_template] if single_input_flag else hypothesis_template

        if not skip_preprocess:
            text = [self.preprocess(t) for t in text]
        assert all(type(t) is str for t in text), text
        batch_size = len(text) if batch_size is None else batch_size
        _index = list(range(0, len(text), batch_size)) + [len(text) + 1]
        probs = []
        with torch.no_grad():
            for i in range(len(_index) - 1):
                encoded_input = self.tokenizer.batch_encode_plus(
                    text[_index[i]: _index[i+1]],
                    max_length=self.max_length,
                    return_tensors='pt',
                    padding=True,
                    truncation=True)
                output = self.model(**{k: v.to(self.device) for k, v in encoded_input.items()})
                if self.multi_label:
                    probs += torch.sigmoid(output.logits).cpu().tolist()
                else:
                    probs += torch.softmax(output.logits, -1).cpu().tolist()

        if return_probability:
            if self.multi_label:
                out = [{
                    'label': [self.id_to_label[str(n)] for n, p in enumerate(_pr) if p > 0.5],
                    'probability': {self.id_to_label[str(n)]: p for n, p in enumerate(_pr)}
                } for _pr in probs]
            else:
                out = [{
                    'label': self.id_to_label[str(p.index(max(p)))],
                    'probability': {self.id_to_label[str(n)]: _p for n, _p in enumerate(p)}
                } for p in probs]
        else:
            if self.multi_label:
                out = [{'label': [self.id_to_label[str(n)] for n, p in enumerate(_pr) if p > 0.5]} for _pr in probs]
            else:
                out = [{'label': self.id_to_label[str(p.index(max(p)))]} for p in probs]
        if single_input_flag:
            return out[0]
        return out


class Sentiment(Classifier):
    def __init__(self,
                 model_name: str = None,
                 max_length: int = None,
                 multilingual: bool = False,
                 use_auth_token: bool = False):
        if model_name is None:
            model_name = MODEL_LIST['sentiment']['multilingual' if multilingual else 'default']
        super().__init__(model_name, max_length=max_length, use_auth_token=use_auth_token)
        self.sentiment = self.predict
        
class Emotion(Classifier):

    def __init__(self,
                 model_name: str = None,
                 max_length: int = None,
                 use_auth_token: bool = False):
        if model_name is None:
            model_name = MODEL_LIST['emotion']['default']
        super().__init__(model_name, max_length=max_length, use_auth_token=use_auth_token)
        self.emotion = self.predict
        
        
class Topic_extract(Classifier):
    def __init__(self,
                 model_name: str = None,
                 max_length: int = None,
                 use_auth_token: bool = False):
        if model_name is None:
            model_name = MODEL_LIST['topic']['default']
        super().__init__(model_name, max_length=max_length, use_auth_token=use_auth_token)
        self.Topic_extract = self.predict