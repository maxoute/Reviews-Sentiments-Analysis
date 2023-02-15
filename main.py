
import pandas as pd
from dataset_annotation import Dataset_anotation
from common_word_extraction import Common_word_extractor
from text_network import TextNetwork
from synonyms_extraction import Synonyms_extarction
from sentiment_intensity import Sentiment_Intensity_Analyzer
import spacy
nlp = spacy.load("en_core_web_lg")  

# creation of the object that annotate the dataset : add columns that specifies if the reviews is positive negative ...
dataset_anotator = Dataset_anotation()

# creation of the object that generate a text netwok ( visualization)
text_network = TextNetwork()
# tn.create_network(data_reviews, 'review_ST')
# tn.draw_network()

# create the object that return synonyms of a certain word
synonym_extractor = Synonyms_extarction()
# synonyms = syn.tokenize_text('review_ST')

# create the object that return the most commons words, specify how many words you want
common_words_extract = Common_word_extractor(20)

# create the object that return the intensity of the sentiment
sia = Sentiment_Intensity_Analyzer()



dataset_anotator.annotation_sentiment("dataset-test/reviews.csv")


positive_reviews = dataset_anotator.filter_sentiment("positive")
neutral_reviews = dataset_anotator.filter_sentiment("neutral")
negative_reviews = dataset_anotator.filter_sentiment("negative")

# print('positive_reviews : ',positive_reviews)
# print('neutral_reviews : ',neutral_reviews)
# print('negative_reviews : ',negative_reviews)

# dataset_anotator.write_dataframe_csv(positive_reviews, "dataset-test/positive_reviews.csv")
# dataset_anotator.write_dataframe_csv(neutral_reviews, "dataset-test/neutral_reviews.csv")
# dataset_anotator.write_dataframe_csv(negative_reviews, "dataset-test/negative_reviews.csv")

data_reviews = dataset_anotator.data_reviews

common_words = common_words_extract.common_words_counter(data_reviews)




# **WORD GRAPH**
data_reviews['review_ST'] = data_reviews['review_ST'].apply(lambda x: ' '.join(x))
print("data_reviews['review_ST'] ", data_reviews['review_ST'] )

article = ' '.join(data_reviews['review_ST'])
doc = nlp(article)

print('--- doc ---', doc)

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

print('--- combo ---', combos)

combos.to_csv('results/test.csv', index=False)



