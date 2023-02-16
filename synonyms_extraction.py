
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

class Synonyms_extarction:
    def __init__(self):
        pass
        
    def find_synonyms_dataset(self,  data_reviews, column_name = 'review_ST'):
        self.data_reviews = data_reviews
        self.column_name = column_name
        synonyms = []
        for index, row in self.data_reviews.iterrows():
            text = row[self.column_name]
            self.tokens = word_tokenize(text)

            for word in self.tokens:
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonyms.append(lemma.name())
        return synonyms, text
        
    def return_synonym_one_word(self, word):
        token = word_tokenize(word)
        synonyms = []
        for word in token:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
        return synonyms


