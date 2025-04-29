import math
import pandas as pd
import numpy as np
import heapq
import matplotlib.pyplot as plt
import seaborn as sns
import os

#we use np.linalg.norm instead
# def find_distance_in_future_space(futures_tests, futures_train):
#     diff = futures_tests - futures_train
#     dot = np.dot(diff, diff)
#     return math.sqrt(dot)

def standardize_matrix(reference_matrix, matrix_to_standardize):
    mean_feature_values = reference_matrix.mean(axis=0)
    std_values = reference_matrix.std(axis=0)

    standardized_reference = (reference_matrix - mean_feature_values) / std_values
    standardized_other = (matrix_to_standardize - mean_feature_values) / std_values

    return standardized_reference, standardized_other

def extract_and_divide_data_task1(file, features_str):
    data_frame = pd.read_csv(file, sep='\t')  # Load file (tab-separated)

    train_data = data_frame[data_frame['Type'] == 'Train']
    test_data = data_frame[data_frame['Type'] == 'Test']

    train_labels = train_data['GenreID'].to_numpy()
    test_labels = test_data['GenreID'].to_numpy()

    feature_cols = [f for f in features_str if f != 'GenreID']

    train_matrix = train_data[feature_cols].to_numpy()
    test_matrix = test_data[feature_cols].to_numpy()
    train_std, test_std = standardize_matrix(train_matrix, test_matrix)

    return train_std, train_labels, test_std, test_labels

#TODO: check if we can inmpove the agmax desition
def KNN_classifier(features=None):
    K = 5
    categories = 10 

    features_str = features if features is not None else ['GenreID', 'spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']
    #file = 'Classification music\GenreClassData_30s.txt'
    #file = os.path.join('Classification music', 'GenreClassData_30s.txt')
    file = 'C:/Users/sarao/OneDrive - NTNU/Documents/KYB/kyb6/ESTIMERING/project/Classification music/GenreClassData_30s.txt'
    train_matrix, train_labels, test_matrix, test_labels = extract_and_divide_data_task1(file, features_str)

    n_test_samples = test_matrix.shape[0]
    n_train_samples = train_matrix.shape[0]

    heap_k_smallest = [[] for _ in range(n_test_samples)] 
    correct_results = [0] * categories

    for i, test_i in enumerate(test_matrix):  # For each test sample

        for j, train_j in enumerate(train_matrix):  # For each train sample
            #distance = np.linalg.norm(test_i - train_j)
            distance = np.sum((test_i - train_j)**2) 

            if len(heap_k_smallest[i]) < K:
                heapq.heappush(heap_k_smallest[i], (-distance, train_labels[j]))
            else:
                largest_negative_distance = heap_k_smallest[i][0][0]
                if -distance > largest_negative_distance:
                    heapq.heappushpop(heap_k_smallest[i], (-distance, train_labels[j]))
        
        category_vote = [0] * categories
        for tupple in heap_k_smallest[i]:
            category_vote[int(tupple[1])] += 1

        category = np.argmax(category_vote)
        #-----------------------------------------------------
        # category_vote = [0] * categories
        # category_distances = [0.0] * categories

        # for distance_neg, genre_id in heap_k_smallest[i]:
        #     genre_idx = int(genre_id)
        #     category_vote[genre_idx] += 1
        #     category_distances[genre_idx] += -distance_neg  # positive distance

        # max_votes = max(category_vote)
        # tied_classes = [idx for idx, votes in enumerate(category_vote) if votes == max_votes]

        # if len(tied_classes) == 1:
        #     category = tied_classes[0]
        # else:
        #     min_total_distance = float('inf')
        #     best_class = None
        #     for cls in tied_classes:
        #         if category_distances[cls] < min_total_distance:
        #             min_total_distance = category_distances[cls]
        #             best_class = cls
        #     category = best_class
        #-----------------------------------------------------

        if category == test_labels[i]:
            correct_results[category] += 1
        

    correct_results_prosentage = sum(correct_results) / n_test_samples

    n_test_of_each_category = n_test_samples / categories
    performance = [x / n_test_of_each_category for x in correct_results]

    return K, correct_results_prosentage, correct_results, performance

def task1():
    K, correct_results_prosentage, correct_results, performance = KNN_classifier()
    print("K = ", K)
    print("Preformance =", correct_results_prosentage)
    print("Preformance per class =", performance)

def extract_and_divide_data_task2(file, features):
    data_frame = pd.read_csv(file, sep='\t')

    genre_0 = data_frame[data_frame['GenreID'] == 0][features]
    genre_1 = data_frame[data_frame['GenreID'] == 1][features]
    genre_2 = data_frame[data_frame['GenreID'] == 2][features]
    genre_9 = data_frame[data_frame['GenreID'] == 9][features]
    
    genre_0['Genre'] = 0
    genre_1['Genre'] = 1
    genre_2['Genre'] = 2
    genre_9['Genre'] = 9
    
    genre_combined = pd.concat([genre_0, genre_1, genre_2, genre_9])
    genre_combined['Genre'] = genre_combined['Genre'].astype(str)  # <-- Force Genre to string

    return genre_combined

def task2(): 
    features_str = ['GenreID', 'spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']
    file = 'Classification music/GenreClassData_30s.txt'  # Make sure the path is correct
    features_distribution = extract_and_divide_data_task2(file, features_str)

    for feature in features_str[1:]:  # Skip GenreID as it's not a feature to be plotted
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=features_distribution, x=feature, hue='Genre', fill=True, common_norm=False, alpha=0.5)
        plt.title(f'Distribution (Bell Curve) of {feature} by Genre')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend(title='Genre')
        plt.show()

#TODO: choose other parameters: OBS: ikke likt for ulike K-verdier
def task3():
    #features = ['GenreID', 'rmse_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']
    features = ['GenreID', 'spectral_rolloff_mean', 'mfcc_1_mean', 'mfcc_3_mean', 'tempo']
    K, correct_results_prosentage, correct_results, performance = KNN_classifier(features)
    print("K = ", K)
    print("Preformance =", correct_results_prosentage)
    print("Preformance per class =", performance)

task1()
