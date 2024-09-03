import numpy as np
from sklearn.decomposition import PCA

def reduce_features(data, p):
    """
    Perform PCA dimensionality reduction

    Args:
        data (list): The data table of the form m*n,
            m - features' columns and a target column
            n - training examples
        p - the limit of the drop in accuracy, if compared to the model without features' reduction

    Returns:
        v - number of features after reduction
        X_reduced - reduced design matrix
        y - target values
    """

    # Separate features and labels
    X = data[:][:-1]
    y = data[:][-1]

    # Normalize the data
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Perform PCA for feature reduction
    pca = PCA()
    pca.fit(X_normalized)

    # Determine the number of components to keep
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    v = np.argmax(cumulative_variance >= (1 - p))

    # Limit v to satisfy v <= 2/3 * m
    max_v = int(2 / 3 * np.shape(X)[1])
    v = min(v, max_v)

    # Apply PCA transformation
    pca = PCA(n_components=v)
    X_reduced = pca.fit_transform(X_normalized)

    return v, X_reduced, y

# read data from the provided dataset
dataframe=[]
with open("C:/Users/Nosac/Desktop/ML_sandbox/static/data/00_559",mode='r') as data:
    for line in data.readlines():
        dataframe.append(list(map(float,line.strip().split())))

# read number of examples n, features m, accuracy drop limit k
n,m,k = list(map(float, dataframe[0]))
df = dataframe[1:][:]

# apple pca dim reduction function on the df
v,X_reduced, y = (reduce_features(df,k))

"""

OPTIONAL: write the reduced feature matrix into the csv file

with(open('C:/Users/Nosac/Desktop/ML_sandbox/00_reduced_features',mode='w',newline=''))as f:
    f.write('\n')
    features_reduced.to_csv(f, sep=' ', index=False, header=False)
    
for row in features_reduced.itertuples(index=False, name=None):
    print(*row)
    
"""