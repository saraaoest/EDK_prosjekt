import numpy as np
from numpy.linalg import inv

def normalize_features(X_train, X_test):
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Handle constant features
    
    X_train_norm = (X_train - min_vals) / range_vals
    X_test_norm = (X_test - min_vals) / range_vals
    
    return X_train_norm, X_test_norm

def standardize_features(X_train, X_test):
    mean_vals = X_train.mean(axis=0)
    std_vals = X_train.std(axis=0)
    
    std_vals[std_vals == 0] = 1  # Avoid division by zero for constant features
    
    X_train_std = (X_train - mean_vals) / std_vals
    X_test_std = (X_test - mean_vals) / std_vals
    
    return X_train_std, X_test_std

def euclidian_dist_sqr(x1, x2, inv_cov):
    return np.sum((x1 - x2) ** 2)

def mahanalios_dist_sqr(x1, x2, inv_cov):
    diff = x1 - x2
    return diff.dot(inv_cov).dot(diff)

def find_k_closest(datapoint, training_X, training_Y, k, dist):
    cov = np.cov(training_X, rowvar=False)
    cov_inv = inv(cov)

    dist_arr = np.empty(len(training_Y))
    for i in range(len(training_X)):
        dist_arr[i] = dist(datapoint, training_X[i], cov_inv)

    nearest_indices = np.argsort(dist_arr)[:k]
    
    return training_Y[nearest_indices]

def get_most_common_label(labels):
    counts = np.bincount(labels)
    return np.argmax(counts)

def predict(datapoint, training_X, training_Y, dist, k):
    k_nearest_labels = find_k_closest(datapoint, training_X, training_Y, k, dist)
    return get_most_common_label(k_nearest_labels)

def knn_classifier(X_test_norm, X_train_norm, y_train, dist):
    predictions = []
    for test_point in X_test_norm:
        pred = predict(test_point, X_train_norm, y_train, dist, k=5)
        predictions.append(pred)

    return np.array(predictions)