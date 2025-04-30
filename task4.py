import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
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

def extract_data_all(file_path):
    df = pd.read_csv(file_path)
    
    # Remove TrackID, Genre and Type columns, keep all features
    feature_columns = df.columns.tolist()
    feature_columns.remove('Track ID')
    feature_columns.remove('Genre')
    feature_columns.remove('Type')
    feature_columns.remove('GenreID')
    
    # Split into training and test sets
    train_data = df[df['Type'] == 'Train']
    test_data = df[df['Type'] == 'Test']
    
    # Extract features and labels
    X_train = train_data[feature_columns].values
    y_train = train_data['GenreID'].values
    
    X_test = test_data[feature_columns].values
    y_test = test_data['GenreID'].values
    
    return X_train, y_train, X_test, y_test

def linear_classifier(cost_function, features=None):
    categories = 10

    exclude_columns = ['Track ID', 'File', 'Genre', 'Type', 'GenreID']
    
    file = 'Classification music/GenreClassData_30s.txt'
    #file = 'Classification music/aggregated_5s_tracks.csv'
    data_frame = pd.read_csv(file, sep='\t')
    
    features_str = [col for col in data_frame.columns if col not in exclude_columns]

    train_matrix, train_labels, test_matrix, test_labels = extract_and_divide_data_task1(file, features_str)
    #train_matrix, train_labels, test_matrix, test_labels = extract_data_all(file)
    #train_matrix, test_matrix = standardize_matrix(train_matrix, test_matrix)


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

    print("Starting test for cost function ", cost_function)
    acc = []
    for i in range(0, 10):
        # Optimization
        W0 = np.random.randn(X_train.shape[1] * categories)*0.01
        if cost_function == 'MSE':
            result = minimize(cost_MSE, W0, method='L-BFGS-B')
        else: 
            result = minimize(cost_cross_entropy, W0, method='L-BFGS-B')

        W_opt = result.x.reshape(X_train.shape[1], categories)

        # Testing
        G_test = X_test @ W_opt

        predicted_labels = np.argmax(G_test, axis=1)

        accuracy = np.mean(predicted_labels == test_labels)
        acc.append(accuracy)
        print(f"Single accuracy: {accuracy:.2%}")

    cm = confusion_matrix(test_labels, predicted_labels)

    genre_names = ['Pop', 'Metal', 'Disco', 'Blues', 'Reggae', 'Classical', 'Rock', 'Hiphop', 'Country', 'Jazz']
    # Plot confusion matrix using seaborn
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genre_names, yticklabels=genre_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label', labelpad=20, fontweight= 'bold')
    plt.xlabel('Predicted Label', labelpad=20, fontweight= 'bold')
    plt.show()

    mean_acc = np.mean(acc)
    return mean_acc

def task4():
    accuracy = linear_classifier('MSE')
    print(f"Total test accuracy: {accuracy:.2%}")

    accuracy = linear_classifier('CE')
    print(f"Total test accuracy: {accuracy:.2%}")

task4()