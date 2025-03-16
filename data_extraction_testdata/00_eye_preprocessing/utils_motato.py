import os
import json

# Data Wrangling
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import tables
import flammkuchen as fl
from datetime import datetime

# Computation
from scipy.interpolate import interp1d

# Image
import cv2 

def nanzscore(array, axis=0):
    return (array - np.nanmean(array, axis=axis))/np.nanstd(array, axis=axis)


def compute_angle_between_vect_tail(v1, v2):
    dot = np.einsum('ijk,ijk->ij',[v1,v1,v2],[v2,v1,v2])
    cos_= dot[0,:]
    sin_= np.cross(v1,v2)
    angle_= np.arctan2(sin_,cos_)
    return angle_


def compute_angle_between_vect(u,v):
    u = u/np.linalg.norm(u)
    v = v/np.linalg.norm(v)
    return np.arctan2(u[0]*v[1]-u[1]*v[0],u[0]*v[0]+u[1]*v[1])


def midpoint(x1, y1, x2, y2):
    return ((x1 + x2)/2, (y1 + y2)/2)


def exptrapolate_segments(tail_x, tail_y, N_seg):
    T = tail_x.shape[1]
    tail_x_10 = np.zeros((N_seg+1,T))
    tail_y_10 = np.zeros((N_seg+1,T))

    for i in range(T):
        points = np.array([tail_x[:,i],tail_y[:,i]]).T  # a (nbre_points x nbre_dim) array

        alpha = np.linspace(0, 1, 11)
        # Linear length along the line:
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        interpolator =  interp1d(distance, points, kind='cubic', axis=0)
        curve = interpolator(alpha)
        tail_x_10[:,i] = curve[:,0]
        tail_y_10[:,i] = curve[:,1]
    return tail_x_10, tail_y_10
    
    
def tail_angles(tail_x_10, tail_y_10, body_x, body_y, N_seg):
    # Tail angle computations
    vect_segment = np.concatenate((np.diff(tail_x_10,axis=0)[:,:,np.newaxis],np.diff(tail_y_10,axis=0)[:,:,np.newaxis]),axis=2)
    vect_segment = np.swapaxes(vect_segment,0,2)

    start_vect = np.vstack((tail_x_10[0,:]-body_x,tail_y_10[0,:]-body_y)).T
    body_vect = -np.vstack((tail_x_10[0,:]-body_x,tail_y_10[0,:]-body_y)).T
    body_angle = np.arctan2(body_vect[:,1],body_vect[:,0])
    body_angle = np.unwrap(body_angle)

    relative_angle = np.zeros((vect_segment.shape[1],N_seg))
    start_vect = np.vstack((tail_x_10[0,:]-body_x,tail_y_10[0,:]-body_y)).T

    for i in range(N_seg):
        relative_angle[:,i] = compute_angle_between_vect_tail(start_vect,vect_segment[:,:,i].T)#,vect_segment[:,:,i+1].T)
        start_vect = np.copy(vect_segment[:,:,i].T)

    tail_angle=np.cumsum(relative_angle,1)

    return tail_angle, body_angle


def eye_preprocess(df):
    # Eye angle computations

    right_eye_posterior_x =  df['right_eye_posterior'].values[:, 0].astype('float')
    right_eye_posterior_y =  df['right_eye_posterior'].values[:, 1].astype('float')
    right_eye_anterior_x =  df['right_eye_anterior'].values[:, 0].astype('float')
    right_eye_anterior_y =  df['right_eye_anterior'].values[:, 1].astype('float')

    left_eye_posterior_x =  df['left_eye_posterior'].values[:, 0].astype('float')
    left_eye_posterior_y =  df['left_eye_posterior'].values[:, 1].astype('float')
    left_eye_anterior_x =   df['left_eye_anterior'].values[:, 0].astype('float')
    left_eye_anterior_y =   df['left_eye_anterior'].values[:, 1].astype('float')

    left_mid_eye_y = (left_eye_anterior_y+left_eye_posterior_y)/2
    left_mid_eye_x = (left_eye_anterior_x+left_eye_posterior_x)/2

    right_mid_eye_x = (right_eye_anterior_x+right_eye_posterior_x)/2
    right_mid_eye_y = (right_eye_anterior_y+right_eye_posterior_y)/2

    a = left_eye_posterior_x-left_eye_anterior_x
    b = left_eye_posterior_y-left_eye_anterior_y
    left_eye_vect = np.array([-b,a])

    a = right_eye_posterior_x-right_eye_anterior_x 
    b = right_eye_posterior_y-right_eye_anterior_y
    right_eye_vect = np.array([b,-a])

    # Vergence: Angle between the two vectors
    vergence = compute_angle_between_vect(right_eye_vect,left_eye_vect)

    # Rotation: Add Angle of two eyes with respect to center to eye axis:
    center_to_left_vect = np.vstack((left_mid_eye_x-right_mid_eye_x,left_mid_eye_y-right_mid_eye_y)) 
    center_to_right_vect = np.vstack((right_mid_eye_x-left_mid_eye_x,right_mid_eye_y-left_mid_eye_y)) 

    left_eye_angle =  compute_angle_between_vect(left_eye_vect,center_to_left_vect)

    right_eye_angle =  compute_angle_between_vect(center_to_right_vect,right_eye_vect)

    rotation_eye = right_eye_angle-left_eye_angle

    mid_headx, mid_heady = midpoint(left_mid_eye_x,left_mid_eye_y, right_mid_eye_x, right_mid_eye_y) #xy left, xy right

    return left_eye_vect, right_eye_vect, left_eye_angle, right_eye_angle, vergence, rotation_eye, mid_headx, mid_heady 


def fin_preprocess(df, body_angle, mid_headx, mid_heady, tail_x_10, tail_y_10):
    ##Fin angle computatright
    #Fin angle computatright
    right_fin_tip_x =  df['right_fin_tip'].values[:, 0].astype('float')
    right_fin_tip_y =  df['right_fin_tip'].values[:, 1].astype('float')
    right_fin_base_x =  df['right_fin_base'].values[:, 0].astype('float')
    right_fin_base_y =  df['right_fin_base'].values[:, 1].astype('float')

    left_fin_tip_x =  df['left_fin_tip'].values[:, 0].astype('float')
    left_fin_tip_y =  df['left_fin_tip'].values[:, 1].astype('float')
    left_fin_base_x =   df['left_fin_base'].values[:, 0].astype('float')
    left_fin_base_y =   df['left_fin_base'].values[:, 1].astype('float')

    # lets make all the vectors
    a = left_fin_base_x-left_fin_tip_x
    b = left_fin_base_y-left_fin_tip_y
    left_fin_vect = np.array([b,-a])

    a = right_fin_base_x-right_fin_tip_x 
    b = right_fin_base_y-right_fin_tip_y
    right_fin_vect = np.array([-b,a])

#     mid_headx, mid_heady = midpoint(left_mid_eye_x,left_mid_eye_y, right_mid_eye_x, right_mid_eye_y) #xy left, xy right
    body_vect = np.vstack((mid_headx -tail_x_10[0,:] , mid_heady - tail_y_10[0,:])) 

    ## Compute angles between vectors
    left_fin_angle =  compute_angle_between_vect(left_fin_vect, body_vect)
    right_fin_angle =  compute_angle_between_vect(right_fin_vect, body_vect)

    #nan movement artifacts
    left_fin_angle = left_fin_angle - left_fin_angle[0]
    right_fin_angle = right_fin_angle - right_fin_angle[0]
    left_fin_angle[abs(np.diff(left_fin_angle, prepend=[0])) >= 2] = 0 #np.nan #np.pi
    right_fin_angle[abs(np.diff(right_fin_angle, prepend=[0])) >= 2] = 0 #np.nan #np.pi
    
    # left_fin_angle[abs(left_fin_angle) >= np.pi] = 0 #np.nan #np.pi
    # right_fin_angle[abs(right_fin_angle)  >= np.pi] = 0 #np.nan #np.pi
    print ('corr fins')

    return left_fin_vect, right_fin_vect, left_fin_angle, right_fin_angle


def resample_behavior(df, motor):
    beh =df.copy()
    resampled_behavior = pd.DataFrame(motor, index=motor.index, columns=beh.columns) #create empty df to fill

    for i in beh.columns:
        interpolated_data = np.interp(motor.t, beh.t, beh[i])
        resampled_behavior[i] = interpolated_data

    return resampled_behavior


def match_times(meta, vid_times):
    #rework both in the same data format
    t_start = meta['general']['t_protocol_start'] 
    t_start = t_start.replace('T', ' ')
    t_end = meta['general']['t_protocol_end']
    t_end = t_end.replace('T', ' ')
    
    prot_start = datetime.strptime(t_start, "%Y-%m-%d %H:%M:%S.%f")
    prot_end = datetime.strptime(t_end, "%Y-%m-%d %H:%M:%S.%f")
    prot_start, prot_end

    vid_start = vid_times.t[0].to_pydatetime()
    vid_end =   vid_times.t[vid_times.shape[0]-1].to_pydatetime()

    #they start quite differently but end quite the same
    delta_start = prot_start.microsecond - vid_start.microsecond
    delta_end = prot_end.microsecond - vid_end.microsecond

    # Motor log is in s and here we convert ms into s
    mstos_start = delta_start/1000000
    mstos_end = delta_end/1000000
    
    return mstos_start

def reduce_to_pi(ar):
    """Reduce angles to the -pi to pi range"""
    return np.mod(ar + np.pi, np.pi * 2) - np.pi



def reconstruct(df_orig, px_mm=150, wnd=50, thres =0.028):
    df = df_orig.copy()
    center_y = 512 #from camera
    center_x = 640 #from camera
    ##filter dataframe to remove motor artifacts          
    filtered_values = np.where((abs(df['x'].diff())<=thres) & (abs(df['y'].diff())<= thres))
    df = df.drop([*filtered_values[0]])
    #Smooth out motor trajectory
    mot_x = df.x.rolling(3).mean()
    mot_y = df.y.rolling(3).mean()
    
    #smooth out fish trajectory and combine
    rec_x = mot_x + (df['body']["x"].rolling(wnd).mean() - center_x)/px_mm 
    rec_y = mot_y + (df['body']["y"].rolling(wnd).mean() - center_y)/px_mm 

    return df, rec_x, rec_y


