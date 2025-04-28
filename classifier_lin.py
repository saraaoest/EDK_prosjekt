import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def linear_classifier(features = None):
    categories = 10

    #TODO: se på alle features
    features_str = features if features is not None else ['GenreID', 'spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']
    file = 'Classification music\GenreClassData_30s.txt'
    train_matrix, train_labels, test_matrix, test_labels = extract_and_divide_data_task1(file, features_str)

    n_test_samples = test_matrix.shape[0]
    n_train_samples = train_matrix.shape[0]


    
    G_i = [0] * categories

    for i in range(categories):
        a = 0

    return 0

def task4():
    linear_classifier()