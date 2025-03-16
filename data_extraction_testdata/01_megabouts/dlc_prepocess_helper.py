import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def are_points_in_circle(points, a, b, r):
    # calculate the distance from each point to the center of the circle
    distances = (points[:, 0] - a)**2 + (points[:, 1] - b)**2

    # check if the distances are less than or equal to the square of the radius
    return distances <= r**2

def resample(df, df2):
    #df gets resampled to df2
    df1 =df.copy()
    resampled_df = pd.DataFrame(np.zeros((df2.shape[0], df1.shape[1])), columns=df1.columns)

    for i in df1.columns:
        interpolated_data = np.interp(df2.t, df1.t, df1[i])
        resampled_df[i] = interpolated_data

    return resampled_df

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

def midpoint(x1, y1, x2, y2):
    return ((x1 + x2)/2, (y1 + y2)/2)

def mid_head(right_eye_posterior_x, right_eye_posterior_y,
             right_eye_anterior_x, right_eye_anterior_y,
             left_eye_posterior_x, left_eye_posterior_y,
             left_eye_anterior_x, left_eye_anterior_y
):
    left_mid_eye_y = (left_eye_anterior_y+left_eye_posterior_y)/2
    left_mid_eye_x = (left_eye_anterior_x+left_eye_posterior_x)/2

    right_mid_eye_x = (right_eye_anterior_x+right_eye_posterior_x)/2
    right_mid_eye_y = (right_eye_anterior_y+right_eye_posterior_y)/2

    mid_headx, mid_heady = midpoint(left_mid_eye_x,left_mid_eye_y, right_mid_eye_x, right_mid_eye_y) #xy left, xy right

    return mid_headx, mid_heady, left_mid_eye_x, left_mid_eye_y, right_mid_eye_x, right_mid_eye_y

def get_eye_points(eye_coords):
    left_eye_points = np.asarray(eye_coords)[:,0]
    right_eye_points = np.asarray(eye_coords)[:,1]
    
    l_anterior = np.asarray(left_eye_points[:,0])
    l_posterior = np.asarray(left_eye_points[:,1])
    
    left_eye_anterior_x = []
    left_eye_anterior_y = []
    left_eye_posterior_x = []
    left_eye_posterior_y = []
    for i in range(l_anterior.shape[0]):
        left_eye_anterior_x.append(l_anterior[i][0])
        left_eye_anterior_y.append(l_anterior[i][1])
        left_eye_posterior_x.append(l_posterior[i][0])
        left_eye_posterior_y.append(l_posterior[i][1])
        
    r_anterior = np.asarray(right_eye_points[:,0])
    r_posterior = np.asarray(right_eye_points[:,1])
    
    right_eye_anterior_x = []
    right_eye_anterior_y = []
    right_eye_posterior_x = []
    right_eye_posterior_y = []
    for i in range(r_anterior.shape[0]):
        right_eye_anterior_x.append(r_anterior[i][0])
        right_eye_anterior_y.append(r_anterior[i][1])
        right_eye_posterior_x.append(r_posterior[i][0])
        right_eye_posterior_y.append(r_posterior[i][1])

    return  left_eye_anterior_x, left_eye_anterior_y, left_eye_posterior_x, left_eye_posterior_y, right_eye_anterior_x, right_eye_anterior_y, right_eye_posterior_x, right_eye_posterior_y,

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

def compute_body_angle(head_x, head_y, body_x, body_y):
    """
    Computes the angle between two points in 2D space.
    
    Parameters:
    head_x, head_y: Coordinates of the first point (head).
    body_x, body_y: Coordinates of the second point (body).
    
    Returns:
    angles_radians: The angle in radians.
    angles_degrees: The angle in degrees.
    """
    # Calculate the differences in the x and y coordinates
    delta_x = body_x - head_x
    delta_y = body_y - head_y

    # Calculate the angle using numpy's arctan2
    angles_radians = np.arctan2(delta_y, delta_x)

    # Convert the angle from radians to degrees
    angles_degrees = np.degrees(angles_radians)

    return angles_radians, angles_degrees


### compute fin and body angles
def fin_preprocess(df, mid_headx, mid_heady, body_x, body_y):
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

    body_vect = np.vstack((mid_headx -body_x , mid_heady - body_y)) 

    ## Compute angles between vectors
    left_fin_angle =  compute_angle_between_vect(left_fin_vect, body_vect)
    right_fin_angle =  compute_angle_between_vect(right_fin_vect, body_vect)

    #nan movement artifacts
    left_fin_angle = left_fin_angle - left_fin_angle[0]
    right_fin_angle = right_fin_angle - right_fin_angle[0]
    left_fin_angle[abs(np.diff(left_fin_angle, prepend=[0])) >= 2] = 0 #np.nan #np.pi
    right_fin_angle[abs(np.diff(right_fin_angle, prepend=[0])) >= 2] = 0 #np.nan #np.pi
    
    return left_fin_vect, right_fin_vect, left_fin_angle, right_fin_angle




    