# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pathlib
import sys
import os


sys.path.insert(1, os.path.join(sys.path[0], '..', 'common'))
import cal

plt.close('all')

# Parameters
do_plot = True

data_freq = 500 # 500 or 1000

video_base_folder = pathlib.Path(f'../datasets/high_res_{data_freq}Hz')
subjects = ["ss01","ss02","ss03","ss04"]

# Info required to convert to deg
scr_res             = [1920, 1080]
scr_size            = [531, 299]
scr_viewDist        = 790
scr_FOV             = 2*np.arctan(np.array(scr_size)/2./scr_viewDist)


deg2PixFunc         = lambda x: np.mean(np.tan(x/2)/np.tan(scr_FOV/2)*scr_res)
pix2DegFunc         = lambda x: np.mean(2*np.arctan(x*np.tan(scr_FOV/2)/scr_res))*180/np.pi

for v in subjects:
    results_folder = str(pathlib.Path('results') / video_base_folder.name / v)
    video_folder = str(video_base_folder / v)
    # Files that contain pupil and cr locations extracted from eye images
    data_files = pathlib.Path(results_folder).glob('cam1_R*.tsv')
    data_files = list(data_files)
    trial_nrs  = [int(x.name[6:9]) for x in data_files]
    df_raw = pd.concat((pd.read_csv(rec, delimiter='\t').assign(point_number=i) for i,rec in zip(trial_nrs,data_files)), ignore_index=True)

    # measurement trials
    all_trials         = df_raw.point_number.unique()
    calibration_trials = list(range(1, 10)) # Calibration trials 1-9
    measurement_trials = [x for x in all_trials if (x not in calibration_trials)]

    # Calibration grid used
    tar_df = pd.read_csv(pathlib.Path(video_folder)/'targetPos_1stFrame_pixels.txt', delimiter='\t',header=None, names=['tar_x', 'tar_y'])
    tar_df['tar_x_deg'] = tar_df.apply(lambda row: pix2DegFunc(row['tar_x']-scr_res[0]/2), axis=1)
    tar_df['tar_y_deg'] = tar_df.apply(lambda row: pix2DegFunc(row['tar_y']-scr_res[1]/2), axis=1)

    xy_target = tar_df.iloc[0:9,2:4].to_numpy()


    # %%
    # Plot feature data for the calibration points
    if do_plot:
        plt.figure()
        df_temp = df_raw[[x in calibration_trials for x in df_raw.point_number]]
        plt.plot(df_temp.pup_x_thresh - df_temp.cr_x_thresh, df_temp.pup_y_thresh - df_temp.cr_y_thresh, '*')
        plt.plot(df_temp.pup_x_LEyes  - df_temp.cr_x_LEyes , df_temp.pup_y_LEyes  - df_temp.cr_y_LEyes , '*')

        # Plot calibration targets
        plt.plot(xy_target[:, 0], xy_target[:, 1], '*')

        plt.legend(['P-CR thresh (pixels)', 'P-CR LEyes (pixels)', 'cal targets (deg)'])
        plt.show()


    # calibrate pupil-CR for the three CR features
    cal_df = [[],[]]
    for i,feat in enumerate(['thresh','LEyes']):
        crx  = 'cr_x_'+feat
        cry  = 'cr_y_'+feat
        pupx = 'pup_x_'+feat
        pupy = 'pup_y_'+feat

        xy_raw = []

        for p in calibration_trials:
            df_point = df_raw[df_raw.point_number == p]

            pcr_x = df_point[pupx] - df_point[crx]
            pcr_y = df_point[pupy] - df_point[cry]
            xy_raw.append([np.nanmedian(pcr_x), np.nanmedian(pcr_y)])


        #%% Find mapping function
        eye_data   = np.array(xy_raw)
        cal_targets= xy_target[np.array(calibration_trials)-1,:]

        gamma_x = cal.biqubic_calibration_with_cross_term(eye_data[:, 0], eye_data[:, 1],
                                                                       cal_targets[:, 0])
        gamma_y = cal.biqubic_calibration_with_cross_term(eye_data[:, 0], eye_data[:, 1],
                                                                       cal_targets[:, 1])


        #%% Apply mapping function
        for p in all_trials:
            # Read json file with timestamps to find timestamps and to decide
            # which frames to be included
            pid_str = f'{p:03d}'
            json_path = pathlib.Path(video_folder) / ('cam1_R' + pid_str + '_info+.json')

            with open(json_path) as f:
                data = json.load(f)

            ts = []
            to_analyze = []
            for d in data:
                ts.append(d['systemTimeStamp'] * 1000)  # to ms
                to_analyze.append(d['toAnalyze'])

            # Remove some frames (before and after the actual recording started)
            df_ts = pd.DataFrame(ts)
            df_ts = df_ts[to_analyze]
            df_ts.reset_index(inplace=True)
            df_ts.drop('index', inplace=True, axis=1)

            df_ts.columns = ['time']
            df_ts['frame_no'] = np.array(df_ts.index) + 1

            df_temp = df_raw[df_raw.point_number == p]
            df_temp = df_temp[to_analyze]
            df_temp.reset_index(inplace=True)
            df_temp.drop('index', inplace=True, axis=1)
            df_temp['frame_no'] = np.array(df_temp.index) + 1

            if len(df_temp) == 0:
                continue

            # compute p-cr, cr, and pupil data
            cr_x = df_temp[crx]
            cr_y = df_temp[cry]
            pupil_x = df_temp[pupx]
            pupil_y = df_temp[pupy]
            pcr_x = pupil_x - cr_x
            pcr_y = pupil_y - cr_y

            frame_number = np.array(df_temp.frame_no).astype('int')

            # Estimate gaze from pcr
            xy = np.vstack((pcr_x, pcr_y)).T

            g_x = cal.biqubic_estimation_with_cross_term(xy[:,0], xy[:,1], gamma_x)
            g_y = cal.biqubic_estimation_with_cross_term(xy[:,0], xy[:,1], gamma_y)

            # Make dataframe and save to file
            D = np.c_[np.repeat(p, len(g_x)), frame_number, np.vstack((g_x, g_y)).T]

            df_calibrated = pd.DataFrame(data=D, columns=['trial', 'frame_no', 'x_'+feat, 'y_'+feat])

            df_cal = pd.merge(df_ts, df_calibrated, on="frame_no")

            cal_df[i].append(df_cal)


    # store calibrated data to file
    df_temp = [pd.concat(x) for x in cal_df]
    df_out = df_temp[0]
    for i in range(1,len(df_temp)):
        df_out = pd.merge(df_out,df_temp[i].drop('time',axis=1), on=['trial','frame_no'])
    df_out.to_csv(pathlib.Path(results_folder)/'calibrated_data.tsv', sep='\t', float_format='%.8f', na_rep='nan',index=False)
