import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from numpy.linalg import inv
import seaborn as sns
import matplotlib.pyplot as plt

def extract_data(file_path, features):

    df = pd.read_csv(file_path, sep='\t')

    if 'GenreID' not in features:
        features.append('GenreID')
    
    df_filtered = df[features]

    train_data = df[df['Type'] == 'Train']
    test_data = df[df['Type'] == 'Test']
    
    X_train = train_data[features[:-1]].values
    y_train = train_data['GenreID'].values
    
    X_test = test_data[features[:-1]].values
    y_test = test_data['GenreID'].values
    
    return X_train, y_train, X_test, y_test

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

def task_1(dist, scaling = False, avg = False, num_k_avg = 10):
    features = ['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']

    X_train, y_train, X_test, y_test = extract_data('Classification music/GenreClassData_30s.txt', features)

    if scaling is not False:
        X_train_norm, X_test_norm = scaling(X_train, X_test)
    else:
        X_train_norm, X_test_norm = X_train, X_test

    if not avg:
        predictions = []
        for test_point in X_test_norm:
            pred = predict(test_point, X_train_norm, y_train, dist, k=5)
            predictions.append(pred)

        predictions = np.array(predictions)

    # Create confusion matrix
        cm = confusion_matrix(y_test, predictions)
        genre_names = ['Pop', 'Metal', 'Disco', 'Blues', 'Reggae', 'Classical', 'Rock', 'Hiphop', 'Country', 'Jazz']
        # Plot confusion matrix using seaborn
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genre_names, yticklabels=genre_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label', labelpad=20, fontweight= 'bold')
        plt.xlabel('Predicted Label', labelpad=20, fontweight= 'bold')
        plt.show()

        # Calculate accuracy
        accuracy = np.sum(predictions == y_test) / len(y_test)
        print(f"Overall accuracy: {accuracy:.2%} Number of correct: ", np.sum(predictions ==  y_test))
    else :
        res = []
        for k in range(1, num_k_avg):
            predictions = []
            for test_point in X_test_norm:
                pred = predict(test_point, X_train_norm, y_train, dist, k)
                predictions.append(pred)

            predictions = np.array(predictions)
            accuracy = np.sum(predictions == y_test) / len(y_test)
            res.append(accuracy)
        print(np.mean(res))

        x = np.linspace(1, num_k_avg-1, len(res))
        plt.plot(x, res, label='Accuracy')

        # Find the highest accuracy point
        best_idx = np.argmax(res)
        best_x = x[best_idx]
        best_y = res[best_idx]

        # Highlight it
        plt.scatter(best_x, best_y, color='red', zorder=5, label=f'Best: k={best_x:.0f}, acc={best_y:.2f}')

        plt.title('Accuracy vs k-value')
        plt.ylabel('Accuracy')
        plt.xlabel('k')
        plt.xticks(np.arange(1, num_k_avg, 2))  # Show all odd numbers
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()

# Use: either euclidian distance or mahalaroian distance. 
# potential arg: avg = True, and num_k_avg. Takes an average over the k first k values and plots it. from 1 to k.
# def task_1(dist, scaling = False, avg = False, num_k_avg = 10):
task_1(mahanalios_dist_sqr)