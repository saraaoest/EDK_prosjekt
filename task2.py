
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import stats

def extract_data(file_path, features):
    df = pd.read_csv(file_path, sep='\t')

    if 'GenreID' not in features:
        features = features + ['GenreID']  # Changed from append to concatenation
    if 'Genre' not in features:
        features = features + ['Genre']    # Changed from append to concatenation
    
    df_filtered = df[features + ['Type']]
    
    train_data = df_filtered[df_filtered['Type'] == 'Train']
    test_data = df_filtered[df_filtered['Type'] == 'Test']
    
    feature_cols = [f for f in features if f not in ['GenreID', 'Genre']]  # Get only the feature columns
    
    X_train = train_data[feature_cols].values
    y_train = train_data['GenreID'].values
    genre_names_train = train_data['Genre'].values

    X_test = test_data[feature_cols].values
    y_test = test_data['GenreID'].values
    genre_names_test = test_data['Genre'].values

    return X_train, y_train, genre_names_train, X_test, y_test, genre_names_test

def standardize_features(X_train, X_test):
    mean_vals = X_train.mean(axis=0)
    std_vals = X_train.std(axis=0)
    
    std_vals[std_vals == 0] = 1  # Avoid division by zero for constant features
    
    X_train_std = (X_train - mean_vals) / std_vals
    X_test_std = (X_test - mean_vals) / std_vals
    
    return X_train_std, X_test_std

features = ['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']

X_train, y_train, genre_names_train, X_test, y_test, genre_names_test = extract_data('Classification music/GenreClassData_30s.txt', features)

allowed_genres = ['pop', 'disco', 'metal', 'classical']
mask = np.isin(genre_names_train, allowed_genres)

X_train = X_train[mask]
y_train = y_train[mask]
genre_names_train = genre_names_train[mask]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

# Colors and transparency for each genre
colors = ['blue', 'green','red', 'gray']
alpha = 0.7

# Plot distributions for each feature
for idx, feature in enumerate(features):
    for genre, color in zip(allowed_genres, colors):
        genre_mask = genre_names_train == genre
        feature_data = X_train[genre_mask][:, idx]
        
        kernel = stats.gaussian_kde(feature_data)
        x_range = np.linspace(min(feature_data), max(feature_data), 200)
        
        axes[idx].plot(x_range, kernel(x_range), color=color, label=genre, alpha=alpha)
        axes[idx].fill_between(x_range, kernel(x_range), color=color, alpha=alpha/2)
        axes[idx].set_title(feature)
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Density')
        axes[idx].legend()

plt.tight_layout()
plt.show()

X_train_norm, X_test_norm = standardize_features(X_train, X_test)

genre_to_color = {genre: idx for idx, genre in enumerate(allowed_genres)}
colors = [genre_to_color[genre] for genre in genre_names_train]

plt.figure(figsize=(15, 6))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_norm)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='Set1')
plt.title('PCA projection of selected genres')
plt.legend(handles=scatter.legend_elements()[0], labels=allowed_genres, title='Genres')
plt.show()