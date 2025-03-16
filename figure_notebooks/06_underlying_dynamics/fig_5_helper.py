import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error


catname = np.array(['AS', 'S1', 'S2', 'BS', 'JT', 'HAT', 'RT', 'SAT',
        'LLC'])
bout_cols =  ['#82cfff','#4589ff','#0000c8','#fcaf6d','#ffb3b8','#08bdba','#24a148','#9b82f3','#ee5396','#e3bc13','#fa4d56']

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

def flatten(xss):
    return [x for xs in xss for x in xs]

def shuffle_along_axis_coherent(data, axis=-1):
    """ Shuffle data along a specified axis coherently across multiple datasets. """
    # Get the shape of the data
    data_shape = data.shape
    
    # Generate a permutation index for shuffling along the specified axis
    # Assume axis is the last dimension (e.g., frequency components)
    indices = np.arange(data_shape[axis])
    np.random.shuffle(indices)  # Shuffle the index array
    
    # Use numpy's advanced indexing to shuffle both fins in a coherent manner
    shuffled_data = np.take_along_axis(data, np.expand_dims(indices, axis=(0, 1)), axis=axis)
    return shuffled_data

def shuffle_phase_coherent(fft_data):
    magnitude = np.abs(fft_data)
    phase = np.angle(fft_data)
    
    # Generate a permutation based on the first fin
    shuffled_indices = np.random.permutation(len(phase[0]))
    
    # Apply the same permutation to phase of each fin
    shuffled_phase = np.array([phase[fin][shuffled_indices] for fin in range(fft_data.shape[0])])
    
    # Reconstruct the FFT data with the shuffled phase
    shuffled_fft = magnitude * np.exp(1j * shuffled_phase)
    
    return shuffled_fft
    
def shuffle_magnitude_coherent(fft_data):
    magnitude = np.abs(fft_data)
    phase = np.angle(fft_data)
    
    # Generate a permutation based on the first fin
    shuffled_indices = np.random.permutation(len(magnitude[0]))
    
    # Apply the same permutation to magnitude of each fin
    shuffled_magnitude = np.array([magnitude[fin][shuffled_indices] for fin in range(fft_data.shape[0])])
    
    # Reconstruct the FFT data with the shuffled magnitude
    shuffled_fft = shuffled_magnitude * np.exp(1j * phase)
    
    return shuffled_fft

def compute_r2(latents, fins):
    # Flatten the latents and fins arrays
    X = latents.reshape(latents.shape[0], -1)

    #Separate Real and Imaginary Parts: You can concatenate the real and imaginary parts of the complex numbers into a real-valued array.
    y = np.hstack((fins.real.reshape(fins.shape[0], -1), 
                fins.imag.reshape(fins.shape[0], -1)))
    # y = fins.reshape(fins.shape[0], -1)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the Ridge regression model with cross-validation
    alphas = [1e-4, 1e-3, 0.01, 0.1, 1.0]
    ridge_cv = RidgeCV(alphas=alphas)

    # Fit the model on the training data
    ridge_cv.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = ridge_cv.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    # Fit the model on the entire dataset
    ridge_cv.fit(X, y)
    # Predict the entire dataset
    y_pred_full = ridge_cv.predict(X)
    # Compute R² scores for each feature's prediction
    r_vals = np.array([r2_score(y_true, y_pred) for y_true, y_pred in zip(y, y_pred_full)])
    return r_vals

def compute_statistics(bout_cat, swim_cat, r_vals, catname):
    means, errs, seq1, seq2 = [], [], [], []
    for a_cat in range(len(bout_cat)):
        if a_cat == 10:
            ids_ = np.where(swim_cat == a_cat)[0][:-1]
        else:
            ids_ = np.where(swim_cat == a_cat)[0]
        means.append(np.mean(r_vals[ids_]))
        seq1.append(r_vals[ids_][r_vals[ids_]>0.0])
        seq2.append(np.clip(r_vals[ids_], 0, None))
        errs.append(np.std(r_vals[ids_]))
    
    means = np.array(means)
    errs = np.array(errs)
    
    return means, errs, seq1, seq2

