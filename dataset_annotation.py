import pandas as pd 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordcloud')
nltk.download('vader_lexicon')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from data_cleaning import clean_text
from model import Emotion, Sentiment, Topic_extract

class Dataset_anotation():
    def __init__(self):
        self.model_sentiment = Sentiment()
        self.model_emotion = Emotion()
        self.topic_extract = Topic_extract()

    def read_csv(self, csv_path):
        self.data_reviews = pd.read_csv(csv_path)
    
    def preprocessing_data(self):
        
        
        list_stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                  "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
                  "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom"
                  "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
                  "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
                  "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
                  "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
                  "there", "when", "where", "why", "how", "all", "any", "both", "each", "more", "most",
                  "other", "some", "such", "nor", "only", "own", "same", "so", "too",
                  "very", "s", "t", "can", "will", "just", "don", "should", "now", "'s","'re", "'m" , "'ve"]

        self.data_reviews.dropna(subset=["review"], inplace=True)        
        self.reviews = self.data_reviews['review']

        for i, review in enumerate(self.reviews):
            words = word_tokenize(review)
            words = [word for word in words if word.lower() not in list_stopwords]
            self.data_reviews.at[i, "review_ST"] = " ".join(words)

        self.data_reviews['review_ST'] = self.data_reviews['review_ST'].apply(clean_text)
        self.reviews_clean = self.data_reviews['review_ST']
        
        
    def annotation_sentiment(self, csv_path):
        
        self.read_csv(csv_path)
        self.preprocessing_data()
        
        
        """ we are adding 4 more columns to the original dataset, 
        label sentiment, the proba of positive, negative and neutral then we are filling thooses columns"""
        
        self.data_reviews["label_sentiment"] = ""
        self.data_reviews["negative_probability_sentiment"] = ""
        self.data_reviews["neutral_probability_sentiment"] = ""
        self.data_reviews["positive_probability_sentiment"] = ""

        for i, review in enumerate(self.reviews_clean):
            prediction = self.model_sentiment.predict(review)
            self.data_reviews.at[i, "label_sentiment"] = prediction["label"]
            self.data_reviews.at[i, "negative_probability_sentiment"] = prediction["probability"]["negative"]
            self.data_reviews.at[i, "neutral_probability_sentiment"] = prediction["probability"]["neutral"]
            self.data_reviews.at[i, "positive_probability_sentiment"] = prediction["probability"]["positive"]
            
    def annotation_emotion(self, csv_path):
        
        self.read_csv(csv_path)
        self.preprocessing_data()


        self.data_reviews["label_emotion"] = ""
        self.data_reviews["joy_probability_emotion"] = ""
        self.data_reviews["surprise_probability_emotion"] = ""
        self.data_reviews["sadness_probability_emotion"] = ""
        self.data_reviews["neutral_probability_emotion"] = ""
        self.data_reviews["fear_probability_emotion"] = ""

        for i, review in enumerate(self.reviews_clean):
            prediction = self.model_emotion.predict(review)
            self.data_reviews.at[i, "label_emotion"] = prediction["label"]
            self.data_reviews.at[i, "joy_probability_emotion"] = prediction["probability"]["joy"]
            self.data_reviews.at[i, "surprise_probability_emotion"] = prediction["probability"]["surprise"]
            self.data_reviews.at[i, "anger_probability_emotion"] = prediction["probability"]["anger"]
            self.data_reviews.at[i, "sadness_probability_emotion"] = prediction["probability"]["sadness"]
            self.data_reviews.at[i, "neutral_probability_emotion"] = prediction["probability"]["neutral"]
            self.data_reviews.at[i, "disgust_probability_emotion"] = prediction["probability"]["disgust"]
            self.data_reviews.at[i, "fear_probability_emotion"] = prediction["probability"]["fear"]

    def write_dataframe_csv(self, dataframe, path):
        
        dataframe.to_csv(path, index=False)
    
    def filter_sentiment(self, sentiment):

        data_reviews_filtered=self.data_reviews[self.data_reviews['label_sentiment']==sentiment].dropna(subset=["review_ST"])
        return data_reviews_filtered
