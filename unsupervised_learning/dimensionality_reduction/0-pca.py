#!/usr/bin/env python3
'''task 1'''
import numpy as np

def pca(X, var=0.95):
    ''' performs PCA on dataset'''
    centeredata = X

    # Step 2: Compute SVD
    _, S, Vt = np.linalg.svd(centeredata, full_matrices=False)

    # Step 3: Calculate explained variance
    explained_variance_ratio = (S ** 2) / np.sum(S ** 2)

    # Step 4: Determine the number of components to keep
    cumulative_variance = np.cumsum(explained_variance_ratio)
    num_components_to_keep = np.argmax(cumulative_variance >= var) + 1

    # Step 5: Select the top principal components
    top_singular_vectors = Vt[:num_components_to_keep]

    # Step 6: Create the weights matrix W
    W = top_singular_vectors.T  # Transpose to get (d, nd)


    return W


'''def cov(x):
    covariance helper func
    n, d = x.shape
    mean = np.mean(x, axis=0)  # Calculate the mean along each dimension

    # Subtract the mean from each data point
    centered_data = x - mean

    # Compute the covariance matrix
    cov_matrix = np.dot(centered_data.T, centered_data) / (n - 1)

    return cov_matrix'''
