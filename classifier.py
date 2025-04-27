import math
import pandas as pd
import numpy as np
import heapq
import matplotlib.pyplot as plt
import seaborn as sns

def extract_and_divide_data_task1(file, features_str):
    data_frame = pd.read_csv(file, sep='\t')# Load file (tab-separated)

    train_matrix = data_frame[data_frame['Type'] == 'Train'][features_str]
    test_matrix = data_frame[data_frame['Type'] == 'Test'][features_str]

    train_matrix = train_matrix.to_numpy()
    test_matrix = test_matrix.to_numpy()
    return train_matrix, test_matrix

#can also use np.linalg.norm
# def find_distance_in_future_space(futures_tests, futures_train):
#     diff = futures_tests - futures_train
#     dot = np.dot(diff, diff)
#     return math.sqrt(dot)

def task1(features = None):
    K = 15
    categories = 10 

    features_str = features if features is not None else ['GenreID', 'spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']
    file = 'Classification music\GenreClassData_30s.txt'
    train_matrix, test_matrix = extract_and_divide_data_task1(file, features_str)

    n_test_samples = test_matrix.shape[0]
    #for every future: -distance & genre:
    heap_k_smallest = [[] for _ in range(n_test_samples)] #will become 198 * k * 2 for the 198 tests // OBS: min-heap--> smallest value at index 0

    correct_results = [0] *categories

    for i, test_i in enumerate(test_matrix):             # does not know class

        for j, train_j in enumerate(train_matrix):        # know class
            futures_tests = test_i[1:]
            futures_train = train_j[1:]
            
            distance = np.linalg.norm(futures_tests - futures_train)
            
            if len(heap_k_smallest[i]) < K:
                heapq.heappush(heap_k_smallest[i], (-distance, train_j[0]))
            else:
                largest_negative_distance = heap_k_smallest[i][0][0]
                if -distance > largest_negative_distance:
                    heapq.heappushpop(heap_k_smallest[i], (-distance, train_j[0]))# pop the smallest value [0]
        
        category_vote = [0] * categories
        for tupple in heap_k_smallest[i]:
            category_vote[ int(tupple[1])] += 1

        category = np.argmax(category_vote) # TODO: hva hvis flere er like
        if category == test_i[0] :
            correct_results[category] += 1
        

    print(correct_results)

    n_test_of_each_categories = n_test_samples/10
    preformance = [x / n_test_of_each_categories for x in correct_results]
    print(preformance)
    return preformance

def extract_and_divide_data_task2(file, features):
    data_frame = pd.read_csv(file, sep='\t')

    # Extract the data for each genre
    genre_0 = data_frame[data_frame['GenreID'] == 0][features]
    genre_1 = data_frame[data_frame['GenreID'] == 1][features]
    genre_2 = data_frame[data_frame['GenreID'] == 2][features]
    genre_9 = data_frame[data_frame['GenreID'] == 9][features]
    
    # Assign a Genre column to each DataFrame
    genre_0['Genre'] = 0
    genre_1['Genre'] = 1
    genre_2['Genre'] = 2
    genre_9['Genre'] = 9
    
    # Concatenate all the genre DataFrames into one DataFrame
    genre_combined = pd.concat([genre_0, genre_1, genre_2, genre_9])
    
    return genre_combined

def task2(): 
    features_str = ['GenreID', 'spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']
    file = 'Classification music/GenreClassData_30s.txt'  # Make sure the path is correct
    features_distribution = extract_and_divide_data_task2(file, features_str)

    for feature in features_str[1:]:  # Skip GenreID as it's not a feature to be plotted
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Genre', y=feature, data=features_distribution)
        plt.title(f'Distribution of {feature} by Genre')
        plt.show()

task2()

#TODO: choose other parameters:
def task3():
    #features = ['GenreID', 'rmse_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']
    features = ['GenreID', 'spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']
    preformance = task1(features)

def task4():
    return 0