import pandas as pd
import numpy as np
import seaborn as sns
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt

df = pd.read_csv('Classification music/GenreClassData_30s.txt', sep='\t')

mandatory_features = ['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']

excluded_cols = ['Track ID', 'File', 'GenreID', 'Genre', 'Type']
df_features_only = df.drop(columns=excluded_cols)

numeric_features = df_features_only.select_dtypes(include=[np.number]).columns.tolist()

train_df = df[df['Type'] == 'Train']
test_df = df[df['Type'] == 'Test']

X_train_full = train_df[numeric_features]
y_train_full = train_df['GenreID']

X_test_full = test_df[numeric_features]
y_test_full = test_df['GenreID']

scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_full_scaled = scaler.transform(X_test_full)

X_train_full_scaled = pd.DataFrame(X_train_full_scaled, columns=numeric_features)
X_test_full_scaled = pd.DataFrame(X_test_full_scaled, columns=numeric_features)

# Store all combinations and their accuracy
results = []

for mandatory_combo in combinations(mandatory_features, 3):
    omitted_feature = list(set(mandatory_features) - set(mandatory_combo))[0]
    
    remaining_features = [f for f in numeric_features if f not in mandatory_combo]
    for extra_feature in remaining_features:
        selected_features = list(mandatory_combo) + [extra_feature]
        
        X_train = X_train_full_scaled[selected_features]
        X_test = X_test_full_scaled[selected_features]
        
        knn = KNeighborsClassifier(n_neighbors=5) # Use optimized classifier to speed up calculation
        knn.fit(X_train, y_train_full)
        y_pred = knn.predict(X_test)
        
        acc = accuracy_score(y_test_full, y_pred)
        
        results.append({
            'accuracy': acc,
            'omitted_feature': omitted_feature,
            'extra_feature': extra_feature,
            'full_feature_set': selected_features
        })

# Now analyze results
results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

accuracies = [entry['accuracy'] for entry in results]

plt.figure(figsize=(8, 5))
sns.histplot(accuracies, bins=20, kde=True, color='mediumseagreen')
benchmark = 0.399
plt.axvline(benchmark, color='red', linestyle='--', linewidth=2, label='Benchmark (39.90%)')

# Add labels and title
plt.title('Accuracy Distribution Across Feature Combinations')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

stopm = 0
for i, result in enumerate(results):
    stop = i
    if result['accuracy'] < benchmark: # Benchmark from task 1
        break 
top_n = stop
top_results = results[:top_n]

# Analyze omissions
omitted_counter = Counter(r['omitted_feature'] for r in top_results)
extra_counter = Counter(r['extra_feature'] for r in top_results)

# Plot omitted mandatory features
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(omitted_counter.keys(), omitted_counter.values(), color='tomato')
plt.title("Most omitted mandatory features")
plt.xlabel("Omitted Feature", fontweight= 'bold')
plt.ylabel("Count")
plt.xticks(rotation=45)

# Plot most included extra features
plt.subplot(1, 2, 2)
top_extra_features = extra_counter.most_common(10)
features, counts = zip(*top_extra_features)

feature_groups = {
    'Temporal': ['zero_cross_rate_mean', 'zero_cross_rate_std', 'rmse_mean', 'rmse_var'],
    'Spectral': ['spectral_centroid_mean', 'spectral_centroid_var',
                 'spectral_bandwidth_mean', 'spectral_bandwidth_var',
                 'spectral_rolloff_mean', 'spectral_rolloff_var',
                 'spectral_contrast_mean', 'spectral_contrast_var',
                 'spectral_flatness_mean', 'spectral_flatness_var'],
    'Chroma Mean': [f'chroma_stft_{i}_mean' for i in range(1, 13)],
    'Chroma Std': [f'chroma_stft_{i}_std' for i in range(1, 13)],
    'MFCC Mean': [f'mfcc_{i}_mean' for i in range(1, 13)],
    'MFCC Std': [f'mfcc_{i}_std' for i in range(1, 13)],
    'Tempo': ['tempo']
}

# Count features by group
group_counts = {group: 0 for group in feature_groups}
for feature, count in extra_counter.items():
    for group_name, group_features in feature_groups.items():
        if feature in group_features:
            group_counts[group_name] += count
            break

# Plot grouped results
plt.bar(group_counts.keys(), group_counts.values(), color='mediumseagreen')
plt.title("Most selected by feature group")
plt.xlabel("Added feature", fontweight='bold')
plt.ylabel("Total Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()