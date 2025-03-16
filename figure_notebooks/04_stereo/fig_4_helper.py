
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from sklearn.manifold import SpectralEmbedding
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances

from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.neighbors import NearestNeighbors
import networkx as nx


catname = np.array(['AS', 'S1', 'S2', 'BS', 'JT', 'HAT', 'RT', 'SAT',
       'OB', 'LLC', 'SLC'])

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)


def normalize_matrix(matrix):
    """
    Normalizes a matrix so that all values are between 0 and 1.
    :param matrix: Input 2D numpy array (matrix).
    :return: Normalized matrix with values between 0 and 1.
    """
    matrix_min = matrix.min()
    matrix_max = matrix.max()

    # Apply min-max normalization
    normalized_matrix = (matrix - matrix_min) / (matrix_max - matrix_min)
    
    return normalized_matrix


def geodesic_distances(X,k,mymetric = 'euclidean'):
    
    knn = NearestNeighbors(n_neighbors=k, metric=mymetric)
    knn.fit(X)
    graph = knn.kneighbors_graph(X, mode='connectivity')

    # Step 5: Convert the k-NN graph to a NetworkX graph and compute shortest paths
    # G = nx.from_scipy_sparse_matrix(graph)
    G = nx.from_scipy_sparse_array(graph)
    # G = nx.to_scipy_sparse_array(graph)

    #geodesic_distance = nx.shortest_path_length(G, source=0, target=1, weight='weight')
    geodesic_distance = dict(nx.shortest_path_length(G, weight='weight'))
    
    geodesic_distance_matrix = np.zeros((X.shape[0], X.shape[0]))

    # Compute the geodesic distance matrix (all-pairs shortest path)
    for i in range(X.shape[0]):
        for j, length in geodesic_distance[i].items():
            geodesic_distance_matrix[i, j] = length

    
    return geodesic_distance_matrix


def reshape_feature_array(feature_vector_array):
    """
    Reshape the feature vector array into a specified shape and extract sub-arrays.

    Parameters:
    - feature_vector_array: numpy.ndarray, the array to be reshaped.

    Returns:
    - reshaped_array: numpy.ndarray, the reshaped array.
    - peaks_a_array, peaks_i_array, valleys_a_array, valleys_i_array: separate sub-arrays.
    """
    max_n = int(feature_vector_array.shape[1] / 4)

    # Reshape the array
    reshaped_array = feature_vector_array.reshape(feature_vector_array.shape[0], 4, max_n)

    # Extract sub-arrays
    peaks_a_array = reshaped_array[:, 0, :]
    peaks_i_array = reshaped_array[:, 1, :]
    valleys_a_array = reshaped_array[:, 2, :]
    valleys_i_array = reshaped_array[:, 3, :]

    print(f"Reshaped array shape: {reshaped_array.shape}")

    return reshaped_array, peaks_a_array, peaks_i_array, valleys_a_array, valleys_i_array


def sort_ipsi_contra_arrays(array1, array2, ipsi_indicator):
    # Determine ipsilateral and contralateral
    """
    ipsi_fin = 0, left fin is ipsi, 
    ipsi_fin = 1, right fin is ipsi_fin,
    """
    ipsi_fin = np.full(array1.shape[0], np.nan)
    contra_fin = np.full(array2.shape[0], np.nan)

    for i, dir_value in enumerate(ipsi_indicator):
        if dir_value == 0:  # Left 
            ipsi_fin[i] = array1[i]
            contra_fin[i] = array2[i]
        elif dir_value == 1:  # Right
            ipsi_fin[i] = array2[i]
            contra_fin[i] = array1[i]
    return ipsi_fin, contra_fin

def get_ipsi_contra_col(df, col1, col2, ipsi_indicator_col):
    """
    Function to separate ipsilateral and contralateral fin durations.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    col1 (str): The name of the first duration column (e.g., 'l_fin_duration').
    col2 (str): The name of the second duration column (e.g., 'r_fin_duration').
    ipsi_indicator_col (str): The name of the column indicating which fin is ipsi (0 for col1, 1 for col2).

    Returns:
    np.ndarray: Array of ipsilateral fin durations.
    np.ndarray: Array of contralateral fin durations.
    """
    # Determine ipsilateral and contralateral durations
    ipsi_fin_duration = np.where(df[ipsi_indicator_col] == 0, df[col1], df[col2])
    contra_fin_duration = np.where(df[ipsi_indicator_col] == 0, df[col2], df[col1])
    
    return ipsi_fin_duration, contra_fin_duration