from transformers import pipeline
from model import Sentiment_model, Emotion_model, Topic_model

sentiment_model = Sentiment_model()

# classifier = pipeline("text-classification",
#                       model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
# print(classifier("I love this!"))


# classifier = pipeline("text-classification",
#                       model="cardiffnlp/twitter-roberta-base-sentiment-latest", return_all_scores=True)
# print(classifier("I love this!"))


# classifier = pipeline("text-classification",
#                       model="cardiffnlp/tweet-topic-21-multi", return_all_scores=True)
# print(classifier("I love this!"))

# # classifier = pipeline("text-classification",
# #                       model="KnutJaegersberg/topic-classification-IPTC-subject-labels", return_all_scores=True)
# # print(classifier("I love this!"))


test,pred = sentiment_model.predict("i love when you hug me")
print(test, '\n', pred)
