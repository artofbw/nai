# -*- coding: utf-8 -*-
'''
It is program that recommends movies
Authors:
Maciej Rybacki
Łukasz Ćwikliński
 
- Build movies recommendation engine
- in the system provide the user
- recommend 5 movies or series, excluding those, which user has seen
- unrecommend 5 movies or series, excluding those, which user has seen
 
'''
 
import argparse
import json
import numpy as np

 
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find recommended movies for the user')
    parser.add_argument('--user', dest='user', required=True, help='Name of the user')
 
    return parser
 
def recommendations(dataset, input_user):
    # Get recommended movies for the given user as argument
    affinity_scores = {}
    global_scores = {}
 
    # iterate over json excluding the user's recommendations
    for user in [x for x in dataset if x != input_user]:
        affinity_score = euclidean_score(dataset, input_user, user)
 
        if affinity_score <= 0:
            continue
        
        # removing films already seen by the given user as argument
        filtered_list = [
            movie for movie in dataset[user] 
            if movie not in dataset[input_user] or 
            dataset[input_user][movie] == 0
        ]
 
        # updating scores
        for movie in filtered_list: 
            global_scores.update({movie: dataset[user][movie] * affinity_score})
            affinity_scores.update({movie: affinity_score})
 
    if len(global_scores) == 0:
        return ['No recommendations possible']
 
    # Creating list of movies with scores
    movie_scores = np.array([[score/affinity_scores[item], item] 
            for item, score in global_scores.items()])
 
    # Sorting movies by scores
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]
 
    # Extract the movie recommendations
    movie_recommendations = [movie for _, movie in movie_scores]
 
    return movie_recommendations
 
# Compute the Euclidean distance score between user1 and user2
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')
 
    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')
 
    # Movies rated by both user1 and user2
    common_movies = {}
 
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1
 
    if len(common_movies) == 0:
        return 0
 
    squared_diff = []
 
    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))
 
    return 1 / (1 + np.sqrt(np.sum(squared_diff)))
 
if __name__=='__main__':
    # Declaring user argument
    args = build_arg_parser().parse_args()
    user = args.user
 
    # Declaring name of the json file
    ratings_file = 'my_ratings.json'
 
    # Opening file with read only mode
    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())
 
    # Writing summary for the given user as argument
    print("\nResult for " + user + ":")
    movies = recommendations(data, user)
 
    # Writing 5 recommended movies for the given user as argument
    print("recommended top 5 movies\n")
    for movie in movies[:5]:
        print(movie)
 
    # Writing 5 unrecommended movies for the given user as argument
    print("\nunrecommended top 5 movies\n")
    for movie in movies[-5:]:
        print(movie)
