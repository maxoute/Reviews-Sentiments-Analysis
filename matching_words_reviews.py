import csv
import pandas as pd


csv_path = 'dataset-test/reviews.csv'
data_reviews = pd.read_csv(csv_path)
reviews = data_reviews['review']


print(reviews)

# # read the CSV file and store the reviews in a list
# reviews = []
# with open('your_file.csv') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     for row in csv_reader:
#         reviews.append(row['review'])

# loop through each review and check for 2-word intersections

# creation d'un dict : {[word1, word2, word 3 ] : nb occurence , [word1, word2, word 3 ] : nb occurence }

# on enregistre les commons words pour chaque review, si les common words en questions existent deja, alors on augmente de 1 le nb occurence 
# la comparaison des commons words doit se faire idenpendament de leur ordre dans la liste

common_words_list =  []

for i, review in enumerate(reviews):
    nb_matching_reviews = 0
    words_i = set(review.split())
    for j, other_review in enumerate(reviews[i+1:], start=i+1):
        words_j = set(other_review.split())
        common_words = words_i.intersection(words_j)
        if len(common_words) == 3:


            nb_matching_reviews = nb_matching_reviews+1

            for common_word in common_words_list:
                if common_word['word_list'] == common_words:
                    common_word['occurence'] +=1
                else : 
                    temp_dict = {'word_list': common_words, 'occurence':1}


            print(f'Review {i+1}: {review}\nReview {j+1}: {other_review}')
            print(f'Commons words{common_words}\n\n')

    print(f'number of matching reviews: {nb_matching_reviews}\nFor the review:\n{review}\n')