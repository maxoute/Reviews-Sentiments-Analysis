
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from dataset_annotation import Dataset_anotation
from common_word_extraction import Common_word_extractor
from text_network import TextNetwork
from synonyms_extraction import Synonyms_extarction

dataset_anotator = Dataset_anotation()


dataset_anotator.read_csv("dataset-test/reviews.csv")
dataset_anotator.preprocessing_data()
dataset_anotator.annotation_sentiment()


positive_reviews = dataset_anotator.filter_sentiment("positive")
neutral_reviews = dataset_anotator.filter_sentiment("neutral")
negative_reviews = dataset_anotator.filter_sentiment("negative")



print('positive_reviews : ',positive_reviews)
print('neutral_reviews : ',neutral_reviews)
print('negative_reviews : ',negative_reviews)


dataset_anotator.write_dataframe_csv(positive_reviews, "dataset-test/positive_reviews.csv")
dataset_anotator.write_dataframe_csv(neutral_reviews, "dataset-test/neutral_reviews.csv")
dataset_anotator.write_dataframe_csv(negative_reviews, "dataset-test/negative_reviews.csv")

data_reviews = dataset_anotator.data_reviews
print('data_reviews.columns ' , data_reviews.columns)

common_words_extract = Common_word_extractor(20)

a = common_words_extract.common_words_counter(data_reviews)


data_reviews['review_ST'] = data_reviews['review_ST'].str.split()
raw_text = [word for word_list in data_reviews['review_ST'] for word in word_list]

label_sentiment=data_reviews['label_sentiment']

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



# Create an instance of the class
tn = TextNetwork()
tn.create_network(data_reviews, 'review_ST')
tn.draw_network()



syn = Synonyms(data_reviews)
synonyms = syn.tokenize_text('review_ST')
