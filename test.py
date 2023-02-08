import model 
from model import *
import torch


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
            self.config, self.tokenizer, self.model = model.load_model(
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
            model_name = model.MODEL_LIST['sentiment']['multilingual' if multilingual else 'default']
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


model_sentiment=Sentiment()
reviews_test="at first I love it,now i hate it"

print(Sentiment.predict(reviews_test))