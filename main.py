
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
from wordcloud import ImageColorGenerator, WordCloud, STOPWORDS
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
from model import Emotion, Sentiment, Topic_extract
from data_cleaning import clean_text
from dataset_annotation import Dataset_anotation


model_topic=Topic_extract()
model_emotion=Emotion()
model_sentiment=Sentiment()

Dataset_anotator = Dataset_anotation()

Dataset_anotator.read_csv("dataset-test/reviews.csv")
Dataset_anotator.preprocessing_data()
Dataset_anotator.annotation_sentiment()
Dataset_anotator.write_dataframe_csv("dataset-test/





# reviews_test="at first I love it,now i hate it"

# model_sentiment.predict(reviews_test)

# print('sentiment test', model_sentiment.sentiment("at first I love it,now i hate it"))



# data_reviews = pd.read_csv("dataset-test/reviews.csv")


#  **COMMON WORDS ALL**

data_reviews['temp_list'] = data_reviews['review_ST'].apply(lambda x:str(x).split())
top = Counter([item for sublist in data_reviews['temp_list'] for item in sublist])
data_common_words = pd.DataFrame(top.most_common(20))
data_common_words.columns = ['Common_words','count']
data_common_words.style.background_gradient(cmap='Blues')
data_common_words.to_csv('results/Result_final_Common_Word.csv', index=False)

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

# COMMON WORDS NEGATIVE

data_reviews_negative['temp_list'] = data_reviews_negative['review_ST'].dropna().apply(lambda x:str(x).split())
top = Counter([item for sublist in data_reviews_negative['temp_list'] for item in sublist])
data_common_words_negative = pd.DataFrame(top.most_common(20))
data_common_words_negative.columns = ['Common_words','count']
data_common_words_negative.style.background_gradient(cmap='Reds_r')


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
data_reviews['review_ST'] = data_reviews['review_ST'].apply(lambda x: ' '.join(x))
article = ' '.join(data_reviews['review_ST'])
doc = nlp(article)

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
