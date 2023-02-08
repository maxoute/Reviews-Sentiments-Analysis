import logging
import urllib.request
from typing import Dict
from urlextract import URLExtract # type: ignore

from datasets.features import Sequence, ClassLabel
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoConfig,\
    AutoModelForMaskedLM

import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordcloud')
nltk.download('vader_lexicon')
nltk.download('wordnet')
import sys
#print(sys.executable)
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
import wordcloud
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import wordnet

import re
import string
import csv
import logging
import os
from typing import List, Dict

import torch
import plotly.express as px
import matplotlib.pyplot as plt

from palettable.colorbrewer.qualitative import Pastel1_7

import numpy as np
from PIL import Image
import spacy

nlp = spacy.load("en_core_web_lg")  


import networkx as nx
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from pyvis.network import Network
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


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


def get_label2id(dataset: DatasetDict, label_name: str = 'label'):
    label_info = dataset[list(dataset.keys())[0]].features[label_name]
    while True:
        if type(label_info) is Sequence:
            label_info = label_info.feature
        else:
            assert type(label_info) is ClassLabel, f"Error at retrieving label information {label_info}"
            break
    return {k: n for n, k in enumerate(label_info.names)}


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

#DEFAULT_CACHE_DIR = f"{os.path.expanduser('~')}/.cache/reviewsnlp/classification"
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


#model_topic=Topic_extract()
model_emotion=Emotion()
model_sentiment=Sentiment()
reviews_test="at first I love it,now i hate it"

model_sentiment.predict(reviews_test)

data_reviews = pd.read_csv("dataset-test/reviews.csv")

sequences = data_reviews['review'].to_list()

data_reviews.dropna(subset=["review"], inplace=True)
stop_words = set(stopwords.words("english")) #+ list(string.punctuation)
reviews = data_reviews['review']

for i, review in enumerate(reviews):
    words = word_tokenize(review)
    words = [word for word in words if word.lower() not in stop_words]
    data_reviews.at[i, "review_ST"] = " ".join(words)


reviews = data_reviews['review_ST']

def clean_text(reviews):
    reviews = str(reviews).lower()
    reviews = re.sub('\[.*?\]', '', reviews)
    reviews = re.sub('https?://\S+|www\.\S+', '', reviews)
    reviews = re.sub('<.*?>+', '', reviews)
    reviews = reviews.replace(string.punctuation, '')
    reviews = reviews.replace('\n', '')
    reviews = reviews.replace(',', '')
    reviews = reviews.replace('.', '')
    reviews = reviews.replace('!', '')
    reviews = reviews.replace('\'', '')
    reviews = re.sub('\w*\d\w*', '', reviews)
    return reviews


data_reviews['review_ST'] = data_reviews['review_ST'].apply(clean_text)


reviews = data_reviews['review_ST']

data_reviews["label_sentiment"] = ""
data_reviews["negative_probability_sentiment"] = ""
data_reviews["neutral_probability_sentiment"] = ""
data_reviews["positive_probability_sentiment"] = ""

for i, review in enumerate(reviews):
    prediction = model_sentiment.predict(review)
    data_reviews.at[i, "label_sentiment"] = prediction["label"]
    data_reviews.at[i, "negative_probability_sentiment"] = prediction["probability"]["negative"]
    data_reviews.at[i, "neutral_probability_sentiment"] = prediction["probability"]["neutral"]
    data_reviews.at[i, "positive_probability_sentiment"] = prediction["probability"]["positive"]


reviews = data_reviews['review']

data_reviews["label_emotion"] = ""
data_reviews["joy_probability_emotion"] = ""
data_reviews["surprise_probability_emotion"] = ""
data_reviews["sadness_probability_emotion"] = ""
data_reviews["neutral_probability_emotion"] = ""
data_reviews["fear_probability_emotion"] = ""

for i, review in enumerate(reviews):
    prediction = model_emotion.predict(review)
    data_reviews.at[i, "label_emotion"] = prediction["label"]
    data_reviews.at[i, "joy_probability_emotion"] = prediction["probability"]["joy"]
    data_reviews.at[i, "surprise_probability_emotion"] = prediction["probability"]["surprise"]
    data_reviews.at[i, "anger_probability_emotion"] = prediction["probability"]["anger"]
    data_reviews.at[i, "sadness_probability_emotion"] = prediction["probability"]["sadness"]
    data_reviews.at[i, "neutral_probability_emotion"] = prediction["probability"]["neutral"]
    data_reviews.at[i, "disgust_probability_emotion"] = prediction["probability"]["disgust"]
    data_reviews.at[i, "fear_probability_emotion"] = prediction["probability"]["fear"]





data_reviews_positive=data_reviews[data_reviews['label_sentiment']=='positive'].dropna(subset=["review_ST"])
data_reviews_neutral = data_reviews[data_reviews['label_sentiment']=='neutral'].dropna(subset=["review_ST"])
data_reviews_negative = data_reviews[data_reviews['label_sentiment']=='negative'].dropna(subset=["review_ST"])

data_reviews_positive

#  **COMMON WORDS ALL**

data_reviews['temp_list'] = data_reviews['review_ST'].apply(lambda x:str(x).split())
top = Counter([item for sublist in data_reviews['temp_list'] for item in sublist])
data_common_words = pd.DataFrame(top.most_common(20))
data_common_words.columns = ['Common_words','count']
data_common_words.style.background_gradient(cmap='Blues')

#data_common_words.head()


fig = px.treemap(data_common_words, path=['Common_words'], values='count',title='Tree of Most Common Words ALL')
#fig.show()


# COMMON WORDS POSTIVE



data_reviews_positive['temp_list'] = data_reviews_positive['review_ST'].apply(lambda x:str(x).split())
top = Counter([item for sublist in data_reviews_positive['temp_list'] for item in sublist])
data_common_words_positive = pd.DataFrame(top.most_common(20))
data_common_words_positive.columns = ['Common_words','count']
data_common_words_positive.style.background_gradient(cmap='summer')

data_common_words.head()

fig = px.treemap(data_common_words_positive, path=['Common_words'], values='count',title='Tree of Most Common Words postive')
#fig.show()

# COMMON WORDS NEUTRAL

data_reviews_neutral['temp_list'] = data_reviews_neutral['review_ST'].dropna().apply(lambda x:str(x).split())
top = Counter([item for sublist in data_reviews_neutral['temp_list'] for item in sublist])
data_common_words_neutral = pd.DataFrame(top.most_common(20))
data_common_words_neutral.columns = ['Common_words','count']
data_common_words_neutral.style.background_gradient(cmap='summer')

fig = px.treemap(data_common_words_neutral, path=['Common_words'], values='count',title='Tree of Most Common Words NEUTRAL')
#fig.show()


# %% [markdown]
# ValueError: 'Yellows' is not a valid value for name; supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'

# %% [markdown]
# 

# %% [markdown]
# COMMON WORDS NEGATIVE
# 

data_reviews_negative['temp_list'] = data_reviews_negative['review_ST'].dropna().apply(lambda x:str(x).split())
top = Counter([item for sublist in data_reviews_negative['temp_list'] for item in sublist])
data_common_words_negative = pd.DataFrame(top.most_common(20))
data_common_words_negative.columns = ['Common_words','count']
data_common_words_negative.style.background_gradient(cmap='Reds_r')

# %% [markdown]
# 

fig = px.treemap(data_common_words_negative, path=['Common_words'], values='count',title='Tree of Most Common Words NEGATIVE')
#fig.show()


data_reviews['review_ST'] = data_reviews['review_ST'].str.split()
raw_text = [word for word_list in data_reviews['review_ST'] for word in word_list]

label_sentiment=data_reviews['label_sentiment']


def words_unique(label_sentiment,numwords,raw_words):
    '''
    Input:
        segment - Segment category (ex. 'Neutral');
        numwords - how many specific words do you want to see in the final result; 
        raw_words - list  for item in train_data[train_data.segments == segments]['temp_list1']:
    Output: 
        dataframe giving information about the name of the specific ingredient and how many times it occurs in the chosen cuisine (in descending order based on their counts)..
    '''
    allother = []
    for item in data_reviews[data_reviews.label_sentiment != label_sentiment]['review_ST']:
        for word in item:
            allother .append(word)
    allother  = list(set(allother ))
    
    specificnonly = [x for x in raw_text if x not in allother]
    
    mycounter = Counter()
    
    for item in data_reviews[data_reviews.label_sentiment == label_sentiment]['review_ST']:
        for word in item:
            mycounter[word] += 1
    keep = list(specificnonly)
    
    for word in list(mycounter):
        if word not in keep:
            del mycounter[word]
    
    Unique_words_dict = {word: mycounter[word] for word in set(mycounter)}
    Unique_words = pd.DataFrame(mycounter.most_common(numwords), columns=['words', 'count'])
    Unique_words = Unique_words.sort_values(by='count', ascending=False)

    return Unique_words

Unique_Positive= words_unique('positive', 20, raw_text)
print("The top 20 unique words in Positive reviews are:")
Unique_Positive.style.background_gradient(cmap='Greens')


plt.figure(figsize=(16,10))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.pie(Unique_Positive['count'], labels=Unique_Positive.words, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Unique Positive Words')
#plt.show()


Unique_Neutral= words_unique('neutral', 20, raw_text)
print("The top 20 unique words in neutral reviews are:")
Unique_Neutral.style.background_gradient(cmap='summer')


plt.figure(figsize=(16,10))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.pie(Unique_Neutral['count'], labels=Unique_Neutral.words, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Unique Neutral Words')
#plt.show()

Unique_Negative= words_unique('negative', 20, raw_text)
print("The top 20 unique words in negative reviews are:")
Unique_Negative.style.background_gradient(cmap='Reds')

plt.figure(figsize=(16,10))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.pie(Unique_Negative['count'], labels=Unique_Negative.words, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Unique Negative Words')
#plt.show()

#  **DATA TEXT   MINING**

def plot_wordcloud(text, mask=None, max_words=300, max_font_size=100, figure_size=(24.0,16.0), color = 'white',
                   title = None, title_size=40, image_color=False):
    
    stopwords = set(STOPWORDS)
    more_stopwords = {'u', "im"}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color=color,
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=400, 
                    height=200,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  

#d = '/kaggle/input/'

data_reviews_positive=data_reviews[data_reviews['label_sentiment']=='positive'].dropna(subset=["review_ST"])
text = ' '.join([' '.join(review) for review in data_reviews_positive['review_ST']])
heart_mask = np.array(Image.open(os.path.join('mask/comment.png')))
plot_wordcloud(text, mask=heart_mask, color='white', max_font_size=100, title_size=30, title="WordCloud of Positive Reviews")
plt.savefig(os.path.join('positive.png'))

data_reviews_neutral = data_reviews[data_reviews['label_sentiment']=='neutral'].dropna(subset=["review_ST"])
text = ' '.join([' '.join(review) for review in data_reviews_neutral['review_ST']])
heart_mask = np.array(Image.open(os.path.join('mask/comment.png')))
plot_wordcloud(text, mask=heart_mask, color='white', max_font_size=100, title_size=30, title="WordCloud of Neutral Reviews")
plt.savefig(os.path.join('neutral.png'))


data_reviews_negative = data_reviews[data_reviews['label_sentiment']=='negative'].dropna(subset=["review_ST"])
text = ' '.join([' '.join(review) for review in data_reviews_negative['review_ST']])
heart_mask = np.array(Image.open(os.path.join('mask/comment.png')))
plot_wordcloud(text, mask=heart_mask, color='white', max_font_size=100, title_size=30, title="WordCloud of Negative Reviews")
plt.savefig(os.path.join('Negative.png'))


# Create a sentiment intensity analyzer object
sia = SentimentIntensityAnalyzer()

# Define a list of words
words = ['happy', 'sad', 'angry', 'excited']

# Map each word to its sentiment score
word_sentiments = {}
for word in words:
    sentiment_score = sia.polarity_scores(word)['compound']
    word_sentiments[word] = sentiment_score

print(word_sentiments)


# **WORD GRAPH**

# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.5.tar.gz


#nlp = en_core_web_lg.load()


data_reviews['review_ST'] = data_reviews['review_ST'].apply(lambda x: ' '.join(x))
article = ' '.join(data_reviews['review_ST'])
doc = nlp(article)

print(doc)

text_list = []
head_list = []

for token in doc:
    if token.is_alpha:
        if not token.is_stop:
            text_list.append(token.lemma_)
            head_list.append(token.head.text.lower())

df = pd.DataFrame(list(zip(text_list, head_list)), 
               columns =['text', 'head'])

combos = df.groupby(['text','head']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)

combos.to_csv('results/test.csv', index=False)


class TextNetwork:
    def __init__(self):
        self.g = nx.Graph()

    def create_network(self, data_reviews, column_name):
        for index, row in data_reviews.iterrows():
            text = row[column_name]
            tokens = word_tokenize(text)
            # Add edges between each pair of tokens
            for i in range(len(tokens)):
                for j in range(i+1, len(tokens)):
                    self.g.add_edge(tokens[i], tokens[j])
    def draw_network(self):
        net = Network(notebook=True, cdn_resources='in_line')
        net.width = "100%"
        net.height = "800px"
        net.from_nx(self.g)
        return net.show("graph.html")

# Create an instance of the class
tn = TextNetwork()
tn.create_network(data_reviews, 'review_ST')
tn.draw_network()

class Synonyms:
    def __init__(self, data_reviews):
        self.data_reviews = data_reviews
    
    def tokenize_text(self, column_name):
        for index, row in self.data_reviews.iterrows():
            text = row[column_name]
            tokens = word_tokenize(text)
            
            for word in tokens:
                synonyms = []
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonyms.append(lemma.name())
                print(f"Synonyms of {word}: {synonyms}")
                #print(synonyms)
syn = Synonyms(data_reviews)
synonyms = syn.tokenize_text('review_ST')


# %% [code]


# %% [code]
