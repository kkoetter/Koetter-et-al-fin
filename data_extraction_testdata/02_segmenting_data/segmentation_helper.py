import numpy as np
import pandas as pd

import numpy as np
from scipy.signal import savgol_filter

def smooth_trace(trace, wnd=5, poly=2):
    return savgol_filter(trace, wnd, poly)  # window size 5, polynomial order 2


def reindex_clusters(clusters):
    # Create a new array for reindexed values
    reindexed_clusters = np.copy(clusters)
    
    # Subtract 2 from elements greater than 2
    reindexed_clusters[clusters > 2] -= 2
    
    return reindexed_clusters
    
def get_motor_pos(df_dlc):
    motor_x = df_dlc.motor.x.values.astype('float')
    motor_y = df_dlc.motor.y.values.astype('float')
    motor_z = df_dlc.motor.z.values.astype('float')
    return motor_x, motor_y, motor_z


def get_eye_data(data):
    left_eye_angle = data['eye_angles'][0]
    right_eye_angle = data['eye_angles'][1]
    eye_rot = data['rotation']
    eye_vergence = data['vergence']
    return left_eye_angle, right_eye_angle, eye_rot, eye_vergence

def create_buffer_lists():
    cluster_n_vector =[]
    body_angles_delta= []
    eye_angles_vector= []
    fin_angles_vector= []
    tail_vectors =[]
    tailsums = []
    body_angles =[]
    eye_vergence_vect =[]
    eye_rot_vect =[]
    
    bout_times = []
    motor_values = []
    mb_outlier =[]
    mb_proba = []
    dlc_filter =[]
    edge_filter =[]
    
    return cluster_n_vector,body_angles_delta,eye_angles_vector,fin_angles_vector,tail_vectors,tailsums ,body_angles,eye_vergence_vect,eye_rot_vect,bout_times ,motor_values ,mb_outlier,mb_proba ,dlc_filter,edge_filter


def nanzscore(array, axis=0):
    return (array - np.nanmean(array, axis=axis))/np.nanstd(array, axis=axis)

def reduce_to_pi(ar):
    """Reduce angles to the -pi to pi range"""
    return np.mod(ar + np.pi, np.pi * 2) - np.pi

def compute_tailsum(tail_angle):
    pre_tailsum= np.zeros((tail_angle.shape[0], tail_angle.shape[1]))
    for segment in range(tail_angle.shape[1]):
        pre_tailsum[:,segment]= (tail_angle[:, segment] - tail_angle[:,0])

    tailsum= np.sum(pre_tailsum, axis=1)/pre_tailsum.shape[1]
    tailsum = reduce_to_pi(tailsum)
    return tailsum

def moving_average(x, w):
    return np.hstack(([0,0], np.convolve(x, np.ones(w), 'valid') / w))


labels_cat = ['approach_swim',
 'slow1',
 'slow2',
 'burst_swim',
 'J_turn',
 'high_angle_turn',
 'routine_turn',
 'spot_avoidance_turn',
 'O_bend',
 'long_latency_C_start',
 'C_start']


color =  ['#82cfff',
  '#4589ff',
  '#0000c8',
  '#fcaf6d',
  '#ffb3b8',
  '#08bdba',
  '#24a148',
  '#9b82f3',
  '#ee5396',
  '#e3bc13',
  '#fa4d56']