import numpy as np
from scipy.optimize import minimize

def linear_classifier(cost_function, features=None):
    categories = 10

    exclude_columns = ['Track ID', 'File', 'Genre', 'Type', 'GenreID']
    
    file = 'Classification music/GenreClassData_30s.txt'
    data_frame = pd.read_csv(file, sep='\t')
    
    features_str = [col for col in data_frame.columns if col not in exclude_columns]

    train_matrix, train_labels, test_matrix, test_labels = extract_and_divide_data(file, features_str)

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

    mean_acc = np.mean(acc)
    return mean_acc