import numpy as np

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)


def compute_laplacian_weighted(adjacency_matrix):
    # Degree matrix: Diagonal matrix with the sum of weights of edges for each node
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    
    # Laplacian matrix: L = D - A
    laplacian_matrix = degree_matrix - adjacency_matrix
    
    return laplacian_matrix
