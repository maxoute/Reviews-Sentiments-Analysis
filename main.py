
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

print('common_words',common_words)

test, text = synonym_extractor.find_synonyms_dataset(data_reviews)



print('text :\n', text)
test = synonym_extractor.return_synonym_one_word(str('computer'))


print(test)

print('data_reviews.columns', data_reviews.columns)