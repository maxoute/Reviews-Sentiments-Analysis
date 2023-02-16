from nltk.sentiment import SentimentIntensityAnalyzer

class Sentiment_Intensity_Analyzer():
    
    def __init__(self):
        self.SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
        
    def analyse_sentiment(self, words):
        # Map each word to its sentiment score
        word_sentiments = {}
        for word in words:
            sentiment_score = self.SentimentIntensityAnalyzer .polarity_scores(word)['compound']
            word_sentiments[word] = sentiment_score
            
            return word_sentiments

# print('--- word_sentiments --- ',word_sentiments)
        
    
# # Define a list of words
# words = ['happy', 'sad', 'angry', 'excited', 'carot', 'fuck the system']

