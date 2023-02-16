

import pandas as pd 
from collections import Counter




class Common_word_extractor():
    def __init__(self, num_words = 10) -> None:
        self.num_words = num_words
        pass

    def common_words_counter(self, data_reviews):

        data_reviews['temp_list'] = data_reviews['review_ST'].apply(lambda x:str(x).split())
        top = Counter([item for sublist in data_reviews['temp_list'] for item in sublist])
        data_common_words = pd.DataFrame(top.most_common(self.num_words))


        data_common_words.columns = ['Common_words','count']
        data_common_words.style.background_gradient(cmap='Blues')
        # data_common_words.to_csv('results/Result_final_Common_Word.csv', index=False)

        return data_common_words

        
# print('data_common_words', data_common_words)
