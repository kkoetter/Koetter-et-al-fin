### preprocess helper functions
import numpy as np
import pandas as pd

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def midpoint(x1, y1, x2, y2):
    return ((x1 + x2)/2, (y1 + y2)/2)

def resample(df, df2):
    #df gets resampled to df2
    df1 =df.copy()
    resampled_df = pd.DataFrame(np.zeros((df2.shape[0], df1.shape[1])), columns=df1.columns)

    for i in df1.columns:
        interpolated_data = np.interp(df2.t, df1.t, df1[i])
        resampled_df[i] = interpolated_data

    return resampled_df


def mid_head(df):
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

    mid_headx, mid_heady = midpoint(left_mid_eye_x,left_mid_eye_y, right_mid_eye_x, right_mid_eye_y) #xy left, xy right

    return mid_headx, mid_heady

def compute_tailsum(tail_angle):
    pre_tailsum= np.zeros((tail_angle.shape[0], tail_angle.shape[1]))
    for segment in range(tail_angle.shape[1]):
        pre_tailsum[:,segment]= (tail_angle[:, segment] - tail_angle[:,0])

    tailsum= np.sum(pre_tailsum, axis=1)/pre_tailsum.shape[1]
    tailsum = reduce_to_pi(tailsum)
    return tailsum

def clean_tail(tail_x, tail_y, rolling=True,  wnd=5, thresh=100):
    new_tail_x = tail_x[:,:].copy()
    new_tail_y = tail_y[:,:].copy()

    #find regions with peaks and filter with median
    for segment in range(new_tail_x.shape[0]):
        test = np.argwhere([abs(np.diff(tail_x[segment,:],prepend=[0]))>=thresh])[:,1]
        for i in test[1:]:
            if i <= wnd:
                pass
            else:
                new_tail_x[segment, i] = np.median(tail_x[segment,i-wnd: i+wnd])

    for segment in range(new_tail_y.shape[0]):
        test = np.argwhere([abs(np.diff(tail_y[segment,:],prepend=[0]))>=thresh])[:,1]
        for i in test[1:]:
            if i <= wnd:
                pass
            else:
                new_tail_y[segment, i] = np.median(tail_y[segment,i-wnd: i+wnd])

    #add rolling window
    if rolling:
        testx = np.mean(rolling_window(new_tail_x[4,:], wnd),1)
        testy = np.mean(rolling_window(new_tail_y[4,:], wnd),1)
        new_tail_x[4,:] = np.concatenate((new_tail_x[4,:wnd-1], testx))
        new_tail_y[4,:]= np.concatenate((new_tail_y[4,:wnd-1], testy))

        testx = np.mean(rolling_window(new_tail_x[3,:], wnd),1)
        testy = np.mean(rolling_window(new_tail_y[3,:], wnd),1)
        new_tail_x[3,:] = np.concatenate((new_tail_x[3,:wnd-1], testx))
        new_tail_y[3,:]= np.concatenate((new_tail_y[3,:wnd-1], testy))
        
    return new_tail_x, new_tail_y


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


def reduce_to_pi(ar):
    """Reduce angles to the -pi to pi range"""
    return np.mod(ar + np.pi, np.pi * 2) - np.pi
