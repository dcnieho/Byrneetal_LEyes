# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import sys
import os
import pathlib
import pandas as pd


sys.path.insert(1, os.path.join(sys.path[0], '..', 'common'))
import cv


IMAGE_SIZE = (180,180)
data_freq = 500 # 500 or 1000

CR_model    = f'high_res_CR_{data_freq}Hz'
pupil_model = f'high_res_pupil_{data_freq}Hz'
video_base_folder = pathlib.Path(f'../datasets/high_res_{data_freq}Hz')
subjects    = ["ss01","ss02","ss03","ss04"]

def process(video_file, result_folder, CR_model, pupil_model):
    from tensorflow import keras

    param_file = pathlib.Path(video_file).parent/'cam1_params.tsv'
    try:
        df_params = pd.read_csv(param_file, sep='\t')
    except:
        print('Error: Run 0_threshold_estimation first!')
        sys.exit()

    #%% Detection parameters (in relatation to iris diameter and intensity
    # threshold of the pupil, cr, and iris)
    iris_area = int(np.pi * (df_params['iris_diameter'].iloc[0] / 2) **2 )
    pupil_area_min = int(np.pi * (df_params['iris_diameter'].iloc[0] / 16) **2 )
    cr_area_min = 2
    cr_area_max = pupil_area_min
    cr_size_limits = [cr_area_min, cr_area_max]
    batch_size=100

    pupil_size_limits = [pupil_area_min, iris_area] # min, max in pixels
    pupil_intensity_threshold = int(df_params['pup_threshold'].iloc[0])
    cr_intensity_threshold = int(df_params['cr_threshold'].iloc[0])
    pupil_cr_distance_max = int(df_params['iris_diameter'].iloc[0] / 4)

    CR_mod = keras.models.load_model(f'..\\trained_model\\{CR_model}.h5', compile=False)
    pupil_mod = keras.models.load_model(f'..\\trained_model\\{pupil_model}.h5', compile=False)

    out_data_file= pathlib.Path(result_folder) / (video_file.stem + '.tsv')


    cap = cv2.VideoCapture(str(video_file))
    total_nr_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(str(video_file), total_nr_of_frames)


    done= False
    t0 = time.time()
    data_out = []
    while not done:
        to_process = {}
        while len(to_process)<batch_size:
            ret, frame = cap.read()
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if frame is None:
                print(frame_number, total_nr_of_frames)
                done = True
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = img[int(df_params['y_ul'].iloc[0]):int(df_params['y_lr'].iloc[0]),
                      int(df_params['x_ul'].iloc[0]):int(df_params['x_lr'].iloc[0])]

            # Detect pupil - thresholding
            pupil_features = cv.detect_pupil(img,
                                             pupil_intensity_threshold,
                                             pupil_size_limits)

            if np.isnan(pupil_features['area']):
                print(f'{int(frame_number)}: pupil not found ({video_file.name})')
                pupil_patch = None
                pupil_patch_off = None
            else:
                # get pupil cutout
                pupil_patch, pupil_patch_off = cv.img_cutout(img,pupil_features['cog'], IMAGE_SIZE, filler=255/2)

                # add mask to pupil cutout
                pupil_pos_local = [int(x-y) for x,y in zip(pupil_features['ellipse'][0],pupil_patch_off)]
                radii = [x/2*1.4 for x in pupil_features['ellipse'][1]]
                angle = pupil_features['ellipse'][2]
                mask  = cv.make_mask(pupil_patch,pupil_pos_local,radii,angle,0,255)
                pupil_patch = (mask/2 + np.multiply(1-mask/255,pupil_patch))/255

            # Detect CR - thresholding
            cr_features = cv.detect_cr(img, cr_intensity_threshold,
                                       cr_size_limits,
                                       pupil_cr_distance_max, pupil_features['cog'],
                                       no_cr = 1)

            if cr_features:
                # get CR cutout
                cr_patch, cr_patch_off = cv.img_cutout(img,cr_features[0][0:2], IMAGE_SIZE, filler=0)

                # add mask to CR cutout
                cr_pos_local = [int(x-y) for x,y in zip(cr_features[0][0:2],cr_patch_off)]
                radius = 32
                mask   = cv.make_mask(cr_patch,cr_pos_local,radius,val=255,bg_val=0)
                cr_patch = np.multiply(mask/255,cr_patch)/255
            else:
                cr_patch = None
                cr_patch_off = None

            to_process[frame_number] = (pupil_features, pupil_patch, pupil_patch_off,
                                           cr_features,    cr_patch,    cr_patch_off)

        print(f'{video_file.name}: {int(frame_number)}/{total_nr_of_frames}')

        # run through LEyes dual CNN
        pupil_frames = [(fr,to_process[fr][1]) for fr in to_process if to_process[fr][1] is not None]
        pupil_pos_LEyes = {}
        if pupil_frames:
            measured_positions_pupil = pupil_mod.predict(np.array([fr[1] for fr in pupil_frames]), verbose=0)*IMAGE_SIZE
            for i,fr in enumerate(pupil_frames):
                fr_num = fr[0]
                pupil_pos_LEyes[fr_num] = measured_positions_pupil[i,:]+to_process[fr_num][2]+[df_params['x_ul'].iloc[0], df_params['y_ul'].iloc[0]]

        CR_frames = [(fr,to_process[fr][4]) for fr in to_process if to_process[fr][4] is not None]
        cr_pos_LEyes = {}
        if CR_frames:
            measured_positions_CR = CR_mod.predict(np.array([fr[1] for fr in CR_frames]), verbose=0)*IMAGE_SIZE
            for i,fr in enumerate(CR_frames):
                fr_num = fr[0]
                cr_pos_LEyes[fr_num] = measured_positions_CR[i,:]+to_process[fr_num][5]+[df_params['x_ul'].iloc[0], df_params['y_ul'].iloc[0]]

        # store in output
        for frame_number in to_process:
            pupil_features, _, _, cr_features, _, _ = to_process[frame_number]

            pupil = [pupil_features['cog'][0]+df_params['x_ul'].iloc[0],
                     pupil_features['cog'][1]+df_params['y_ul'].iloc[0],
                     pupil_features['area'],
                     pupil_features['ellipse'][0][0]+df_params['x_ul'].iloc[0],
                     pupil_features['ellipse'][0][1]+df_params['y_ul'].iloc[0],
                     pupil_features['ellipse'][1][0],
                     pupil_features['ellipse'][1][1],
                     pupil_features['ellipse'][2],
                     pupil_features['ellipse'][3]]

            if not cr_features:
                cr_features.append([np.nan, np.nan, np.nan])
            cr_features[0][0] += df_params['x_ul'].iloc[0]
            cr_features[0][1] += df_params['y_ul'].iloc[0]

            pupil_LEyes = [np.nan,np.nan]
            if frame_number in pupil_pos_LEyes:
                pupil_LEyes = [*pupil_pos_LEyes[frame_number].flatten()]
            CR_LEyes = [np.nan,np.nan]
            if frame_number in cr_pos_LEyes:
                CR_LEyes = [*cr_pos_LEyes[frame_number].flatten()]

            temp_list = [frame_number] + \
                        pupil + \
                        pupil_LEyes + \
                        [x for b in cr_features for x in b] + \
                        CR_LEyes

            data_out.append(temp_list)

    # pp.close()
    print('fps: {}'.format(total_nr_of_frames / (time.time() - t0)))
    print(len(data_out))

    # Save data to dataframe
    col_lbls = [
        'frame_no',
        'pup_x_thresh','pup_y_thresh', 'pup_area',
        'pup_x_ellipse', 'pup_y_ellipse',
        'Ma', 'ma',
        'angle', 'area_ellipse',
        'pup_x_LEyes','pup_y_LEyes',
        'cr_x_thresh','cr_y_thresh',
        'cr_area','cr_x_LEyes','cr_y_LEyes'
    ]
    df = pd.DataFrame(data_out, columns = col_lbls)


    # Save files if not in debug mode
    df.to_csv(out_data_file, sep='\t', float_format='%.8f', na_rep='nan',index=False)

    # When everything done, release the capture
    cap.release()

    return video_file



if __name__ == "__main__":
    import pebble

    with pebble.ProcessPool(max_workers=2, max_tasks=1) as pool:
        for v in subjects:
            video_folder = video_base_folder / v
            results_folder = f"results\\{video_base_folder.name}\\{video_folder.name}"

            if not pathlib.Path(results_folder).is_dir():
                pathlib.Path(results_folder).mkdir(parents=True)

            video_files = list(pathlib.Path(video_folder).glob('*.mp4'))
            res_folds   = [results_folder]*len(video_files)
            CR_models   = [CR_model]*len(video_files)
            pupil_models= [pupil_model]*len(video_files)
            for result in pool.map(process, video_files, res_folds, CR_models, pupil_models).result():
                print(f'done with {result}')
