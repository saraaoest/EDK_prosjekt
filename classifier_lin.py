import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

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
    feature_cols = [f for f in features_str if f != 'GenreID']

    train_labels = train_data['GenreID'].to_numpy()
    test_labels = test_data['GenreID'].to_numpy()
    train_matrix = train_data[feature_cols].to_numpy()
    test_matrix = test_data[feature_cols].to_numpy()

    train_std, test_std = standardize_matrix(train_matrix, test_matrix)
    return train_std, train_labels, test_std, test_labels

def linear_classifier(features=None):
    categories = 10

    features_str = features if features is not None else ['GenreID', 'spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']
    file = 'Classification music\\GenreClassData_30s.txt'
    train_matrix, train_labels, test_matrix, test_labels = extract_and_divide_data_task1(file, features_str)

    n_test_samples = test_matrix.shape[0]
    n_train_samples = train_matrix.shape[0]

    X_train = np.hstack((train_matrix, np.ones((n_train_samples, 1))))
    X_test = np.hstack((test_matrix, np.ones((n_test_samples, 1))))
    T_train_categories = np.zeros((n_train_samples, categories))
    for idx, label in enumerate(train_labels):
        T_train_categories[idx, label] = 1

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def cost_MSE(W_flat):
        W = W_flat.reshape(X_train.shape[1], categories)
        G = X_train @ W
        preds = sigmoid(G)  # Bruk sigmoid!
        return 0.5 * np.mean((preds - T_train_categories) ** 2)

    W0 = np.random.randn(X_train.shape[1] * categories)

    result = minimize(cost_MSE, W0, method='L-BFGS-B')
    W_opt = result.x.reshape(X_train.shape[1], categories)

    # Test
    G_test = X_test @ W_opt
    preds_test = sigmoid(G_test)

    predicted_labels = np.argmax(preds_test, axis=1)

    accuracy = np.mean(predicted_labels == test_labels)
    print(f"Test accuracy: {accuracy:.2f}")

    return accuracy

def task4():
    linear_classifier()

task4()