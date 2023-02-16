
import re
import string


def clean_text(reviews):
    reviews = str(reviews).lower()
    reviews = re.sub('\[.*?\]', '', reviews)
    reviews = re.sub('https?://\S+|www\.\S+', '', reviews)
    reviews = re.sub('<.*?>+', '', reviews)
    reviews = reviews.replace(string.punctuation, '')
    reviews = reviews.replace('\n', '')
    reviews = reviews.replace(',', '')
    reviews = reviews.replace('.', '')
    reviews = reviews.replace('!', '')
    reviews = reviews.replace('\'', '')
    reviews = re.sub('\w*\d\w*', '', reviews)
    return reviews