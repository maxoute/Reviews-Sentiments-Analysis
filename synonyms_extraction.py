
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

class Synonyms_extarction:
    def __init__(self):
        pass
        
    
    def tokenize_text(self, column_name, data_reviews):
        self.data_reviews = data_reviews
        
        for index, row in self.data_reviews.iterrows():
            text = row[column_name]
            tokens = word_tokenize(text)
            
            for word in tokens:
                synonyms = []
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonyms.append(lemma.name())
                print(f"Synonyms of {word}: {synonyms}")
        return synonyms
