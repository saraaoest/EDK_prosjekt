#mylines = []
#with open("Classification music\GenreClassData_30s.txt", "rt") as myfile:
#    for track in myfile: #each line represents a track
#        mylines.append(track)

#contents = myfile.read()
#myfile.close()  needed only when myfile = open(...)
#print(contents) 

def extract_data():
    import pandas as pd

    # Load the data (tab-separated with header)
    df = pd.read_csv('Classification music\GenreClassData_30s.txt', delimiter='\t') 

    # Extract the relevant features
    features = df[['Type', 'GenreID', 'spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']]

    # Convert to matrix form (NumPy array)
    matrix = features.to_numpy()
    return matrix

#import numpy as np
#np.set_printoptions(precision=16, suppress=False)


def extract_and_devide_data():
    import pandas as pd

    # Load file (tab-separated)
    df = pd.read_csv('Classification music\GenreClassData_30s.txt', sep='\t')

    selected_columns = ['GenreID', 'spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']

    train_matrix = df[df['Type'] == 'Train'][selected_columns]
    test_matrix = df[df['Type'] == 'Test'][selected_columns]

    train_matrix = train_matrix.to_numpy()
    test_matrix = test_matrix.to_numpy()

    return train_matrix, test_matrix

traing_matrix, test_matrix = extract_and_devide_data()
#shape = matrix.shape
print(traing_matrix)
shape = traing_matrix.shape
print(shape)