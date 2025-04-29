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

    columns = ['GenreID',
                'zero_cross_rate_mean', 'zero_cross_rate_std',
                'rmse_mean', 'rmse_var',
                'spectral_centroid_mean', 'spectral_centroid_var',
                'spectral_bandwidth_mean', 'spectral_bandwidth_var',
                'spectral_rolloff_mean', 'spectral_rolloff_var',
                'spectral_contrast_mean', 'spectral_contrast_var',
                'spectral_flatness_mean', 'spectral_flatness_var',
                'chroma_stft_1_mean', 'chroma_stft_2_mean', 'chroma_stft_3_mean',
                'chroma_stft_4_mean', 'chroma_stft_5_mean', 'chroma_stft_6_mean',
                'chroma_stft_7_mean', 'chroma_stft_8_mean', 'chroma_stft_9_mean',
                'chroma_stft_10_mean', 'chroma_stft_11_mean', 'chroma_stft_12_mean',
                'chroma_stft_1_std', 'chroma_stft_2_std', 'chroma_stft_3_std',
                'chroma_stft_4_std', 'chroma_stft_5_std', 'chroma_stft_6_std',
                'chroma_stft_7_std', 'chroma_stft_8_std', 'chroma_stft_9_std',
                'chroma_stft_10_std', 'chroma_stft_11_std', 'chroma_stft_12_std',
                'tempo',
                'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean',
                'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_7_mean', 'mfcc_8_mean',
                'mfcc_9_mean', 'mfcc_10_mean', 'mfcc_11_mean', 'mfcc_12_mean',
                'mfcc_1_std', 'mfcc_2_std', 'mfcc_3_std', 'mfcc_4_std',
                'mfcc_5_std', 'mfcc_6_std', 'mfcc_7_std', 'mfcc_8_std',
                'mfcc_9_std', 'mfcc_10_std', 'mfcc_11_std', 'mfcc_12_std'
            ]

    features_str = features if features is not None else columns
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
    
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def cost_cross_entropy(W_flat):
        W = W_flat.reshape(X_train.shape[1], categories)
        logits = X_train @ W
        probs = softmax(logits)
        return -np.mean(np.sum(T_train_categories * np.log(probs + 1e-12), axis=1))

    def cost_MSE(W_flat):
        W = W_flat.reshape(X_train.shape[1], categories)
        G = X_train @ W
        preds = sigmoid(G)  
        return 0.5 * np.mean((preds - T_train_categories) ** 2)

    W0 = np.random.randn(X_train.shape[1] * categories)

    result = minimize(cost_cross_entropy, W0, method='L-BFGS-B')# Bruk sigmoid! --> heller bruke softmax?
    W_opt = result.x.reshape(X_train.shape[1], categories)

    # Test
    G_test = X_test @ W_opt
    preds_test = softmax(G_test) # Bruk sigmoid! --> heller bruke softmax?
    predicted_labels = np.argmax(preds_test, axis=1)
    accuracy = np.mean(predicted_labels == test_labels)

    return accuracy

def task4():
    accuracy = linear_classifier()
    print(f"Test accuracy: {accuracy:.2f}")

task4()