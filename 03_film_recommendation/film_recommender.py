# -*- coding: utf-8 -*-
'''
It is program that recommends movies    
Authors:
Maciej Rybacki
Łukasz Ćwikliński

- zbuduj silnik rekomendacji filmów lub seriali
- w systemie podaj użytkownika wejściowego
- zaproponuj 5 filmów interesujących dla wybranego użytkownika, których nie oglądał
- zaproponouj 5 film, których nie należy oglądać;

'''


import argparse
import json
import numpy as np

from compute_scores import euclidean_score

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find the movie recommendations for the given user')
    parser.add_argument('--user', dest='user', required=True,
            help='Input user')
    return parser
 
def get_recommendations(dataset, input_user):
    # Get movie recommendations for the input user
    overall_scores = {}
    similarity_scores = {}

    # iterate over dataset different than user we'd like to do recommendations
    for user in [x for x in dataset if x != input_user]:
        similarity_score = euclidean_score(dataset, input_user, user)

        if similarity_score <= 0:
            continue
        
        # filter out movies already seen by user
        filtered_list = [
            movie for movie in dataset[user] 
            if movie not in dataset[input_user] or 
            dataset[input_user][movie] == 0
        ]

        # get scores
        for movie in filtered_list: 
            overall_scores.update({movie: dataset[user][movie] * similarity_score})
            similarity_scores.update({movie: similarity_score})

    if len(overall_scores) == 0:
        return ['No recommendations possible']

    # Generate movie ranks by normalization 
    movie_scores = np.array([[score/similarity_scores[item], item] 
            for item, score in overall_scores.items()])

    # Sort in decreasing order 
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]

    # Extract the movie recommendations
    movie_recommendations = [movie for _, movie in movie_scores]

    return movie_recommendations
 
if __name__=='__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    ratings_file = 'my_ratings.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    print("\nResult for " + user + ":")
    movies = get_recommendations(data, user)

    print("recommended top 5 movies\n")
    for movie in movies[:5]:
        print(movie)

    print("\nunrecommended top 5 movies\n")
    for movie in movies[-5:]:
        print(movie)
