## import functions
import cv2 as cv
from multiprocessing import Process, Queue
import numpy as np
from pathlib import Path
import pandas as pd
from eye_extraction_helpers import preprocess_img
from eye_extraction_helpers import mid_head_parallel, angles_parallel, compute_angles_parallel
import h5py


def get_df_values(df, ind):
    # Extract angles
    body_x = df.iloc[ind].body.values[0].astype('float')
    body_y = df.iloc[ind].body.values[1].astype('float')
    # computations
    mid_headx, mid_heady, left_mid_eye_x, left_mid_eye_y, right_mid_eye_x, right_mid_eye_y = mid_head_parallel(
        df.iloc[ind])
    body_vect = np.vstack((mid_headx - body_x, mid_heady - body_y))

    return body_x, body_y, mid_headx, mid_heady, left_mid_eye_x, left_mid_eye_y, right_mid_eye_x, right_mid_eye_y, body_vect


# Will be processed in parallel.
def process_eyes(frame, df, ind):
    # Do some Python operation to show its not OpenCV dependent.
    body_x, body_y, mid_headx, mid_heady, left_mid_eye_x, left_mid_eye_y, right_mid_eye_x, right_mid_eye_y, body_vect = get_df_values(
        df, ind)

    mask_left = preprocess_img(frame, left_mid_eye_y, left_mid_eye_x, tol_val=10)
    mask_right = preprocess_img(frame, right_mid_eye_y, right_mid_eye_x, tol_val=10)

    left_eye_vect, right_eye_vect, l_eye_points, r_eye_points, [l_x0, l_y0], [r_x0, r_y0] = angles_parallel(mask_right,
                                                                                                            mask_left,
                                                                                                            body_x,
                                                                                                            body_y)
    left_eye_angle, right_eye_angle, rotation_eye, vergence = compute_angles_parallel(left_eye_vect, right_eye_vect,
                                                                                      body_vect)

    res = [left_eye_angle[0], right_eye_angle[0], vergence, rotation_eye[0]]
    coords = [[l_eye_points, r_eye_points]]

    return res, ind, coords


def eye_process_frame(queue, filename, num_frames, ind, df):
    # Get the frames from the file.
    cap = cv.VideoCapture(filename)
    cap.set(cv.CAP_PROP_POS_FRAMES, ind)
    results = []
    inds_ = []
    coords_ =[]
    for i in range(num_frames):
        ret, frame = cap.read()
        res, ind_, coords = process_eyes(frame, df, ind+i)
        results.append(res)
        inds_.append(ind_)
        coords_.append(coords)
    # Return index to keep track of the indices as it may shuffle.
    queue.put([results, inds_, coords_])

def multiprocess_run(filename, df):
    cap = cv.VideoCapture(filename)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    n_frames = frame_count

    queue = Queue()
    threadn = cv.getNumberOfCPUs()
    frames_per_thread = int(n_frames / threadn)

    processes = []
    for i in range(threadn):
        process = Process(target=eye_process_frame,
                          args=(queue, filename, frames_per_thread, i * frames_per_thread, df))
        processes.append(process)
        process.start()

    results_ = []
    inds_ = []
    coords_ = []
    for i in range(threadn):
        result = queue.get(block=True)
        results_.append(result[0])
        inds_.append(result[1])
        coords_.append(result[2])

    for process in processes:
        process.join()

    cap.release()

    return results_, inds_, coords_

def flatten(l):
    return [item for sublist in l for item in sublist]


if __name__ == "__main__":
    fish =0
    fps = 200

    master_path = Path(r"\\portulab.synology.me\data\Kata\Data\230307_visstim_2D")
    out_path = Path(r"\\portulab.synology.me\data\Kata\Processed_Data\230307_visstim_2D_")

    fish_paths = list(master_path.glob('*f[0-9]*'))
    fish_path = fish_paths[fish]
    fish_id = fish_paths[fish].name
    filename = str(list(fish_path.glob('*video*'))[0])
    print ("Working on {}".format(filename))

    filename_dlc = list(fish_path.glob('*316000.h5*'))[0]
    df = pd.read_hdf(filename_dlc, header=[1, 2], index_col=0)
    df = df['DLC_resnet50_dlc_2Dec12shuffle1_316000']
    print(f'{df.shape[0] / (fps * 60)} minutes at {fps} fps')
    print('working on {} frames'.format(df.shape[0]))
    target_n = df.shape[0]
    print("About {} min for {} frames".format((target_n * 0.02) / 60, target_n))

    # run the processing loop
    results_, inds_, coords_ = multiprocess_run(filename, df)

    # sort data accroding to array
    inds = flatten(inds_)
    res = flatten(results_)
    coords = flatten(coords_)

    inds = np.asarray(inds)
    res = np.asarray(res)
    coords = np.asarray(coords)

    sorted_res = [x for _, x in sorted(zip(inds, res))]
    sorted_coords = [x for _, x in sorted(zip(inds, coords))]
    sorted_inds = np.sort(inds)

    #save data
    hf = h5py.File(out_path/'{}_eye_angles.h5'.format(fish_id), 'w')
    hf.create_dataset('eye_angles', data= np.asarray(sorted_res)[:,:2], compression="gzip", compression_opts=9)
    hf.close()

    hf = h5py.File(out_path/'{}_eye_rot.h5'.format(fish_id), 'w')
    hf.create_dataset('eye_rot', data= np.asarray(sorted_res)[:,2], compression="gzip", compression_opts=9)
    hf.close()

    hf = h5py.File(out_path/'{}_eye_verg.h5'.format(fish_id), 'w')
    hf.create_dataset('eye_verg', data= np.asarray(sorted_res)[:,3], compression="gzip", compression_opts=9)
    hf.close()

    hf = h5py.File(out_path/'{}_eye_coords.h5'.format(fish_id), 'w')
    hf.create_dataset('eye_coords', data= np.asarray(sorted_coords), compression="gzip", compression_opts=9)
    hf.close()

    print ('done')