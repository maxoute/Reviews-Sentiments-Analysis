
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
model='cardiffnlp/twitter-roberta-base-sentiment-latest'

for julie in MODEL_LIST:
    for maxens in julie:
        print(maxens)
        if maxens==model:
            print(julie)
            print(MODEL_LIST[julie])