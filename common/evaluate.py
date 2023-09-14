# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import patches
import cv2
from scipy import ndimage

import dataset_loaders

def plot_unet_prediction(x, p, t, n, dataset_path):
    fig, axs = plt.subplots(1, 5)

    axs[0].imshow(x.squeeze(), cmap='gray')
    axs[1].imshow(p["pmap"])
    axs[2].imshow(p["mask"])
    if isinstance(t,np.ndarray):
        axs[3].imshow(t)
    elif t is not None:
        axs[3].imshow(t.detach().cpu().squeeze())

    axs[0].set_title("Model Input")
    axs[1].set_title("Prob Map")
    axs[2].set_title("Mask")
    axs[3].set_title("Target")

    if not np.isnan(p["centroid"][0]):
        if 'oripupil' in p:
            mypup = p['oripupil']
        else:
            mypup = p
        x0, y0 = mypup["centroid"]
        orientation = mypup["orientation"]
        x1 = x0 + math.cos(math.radians(orientation)) * mypup["axis_major_radius"]
        y1 = y0 + math.sin(math.radians(orientation)) * mypup["axis_major_radius"]
        x2 = x0 - math.sin(math.radians(orientation)) * mypup["axis_minor_radius"]
        y2 = y0 + math.cos(math.radians(orientation)) * mypup["axis_minor_radius"]
        axs[2].plot((x0, x1), (y0, y1), '-r', linewidth=2)
        axs[2].plot((x0, x2), (y0, y2), '-b', linewidth=2)
        axs[2].plot(x0, y0, '.g', markersize=5)
        el=patches.Ellipse(xy=mypup["centroid"],width=mypup["axis_major_radius"]*2,height=mypup["axis_minor_radius"]*2,angle=mypup["orientation"],facecolor='none',edgecolor='red')
        axs[2].add_patch(el)

    im = cv2.imread(str(dataset_path/'images'/n), cv2.IMREAD_GRAYSCALE)
    axs[4].imshow(im, cmap='gray')
    if not np.isnan(p["centroid"][0]):
        x0, y0 = p["centroid"]
        orientation = p["orientation"]
        x1 = x0 + math.cos(math.radians(orientation)) * p["axis_major_radius"]
        y1 = y0 + math.sin(math.radians(orientation)) * p["axis_major_radius"]
        x2 = x0 - math.sin(math.radians(orientation)) * p["axis_minor_radius"]
        y2 = y0 + math.cos(math.radians(orientation)) * p["axis_minor_radius"]
        axs[4].plot((x0, x1), (y0, y1), '-r', linewidth=2)
        axs[4].plot((x0, x2), (y0, y2), '-b', linewidth=2)
        axs[4].plot(x0, y0, '.g', markersize=5)
        el=patches.Ellipse(xy=p["centroid"],width=p["axis_major_radius"]*2,height=p["axis_minor_radius"]*2,angle=p["orientation"],facecolor='none',edgecolor='red')
        axs[4].add_patch(el)

    for a in axs.ravel():
        a.axis("off")



def get_pure_pupil(file, pure_df):
    if 'subject' in pure_df.columns:
        fname = file.split('/')[-1].split('_')[-1]
        subject = file.split('/')[0]
        sel = (pure_df['File_name'] == fname) & (pure_df['subject'] == subject)
    else:
        sel = (pure_df['File_name'] == file)

    x           = pure_df[sel]['pup_x'].tolist()[0]
    y           = pure_df[sel]['pup_y'].tolist()[0]
    confidence  = pure_df[sel]["confidence"].tolist()[0]

    return x, y, confidence


def get_ground_truth(gt, gt_is_mask, sfac):
    gt   =   gt.squeeze().detach().numpy()
    sfac = sfac.squeeze().detach().numpy()
    if gt_is_mask:
        # NB: only a pupil in this mask
        gt_has_pupil = np.any(gt.squeeze())
        if not gt_has_pupil:
            gt = np.array([np.nan, np.nan])
        else:
            # input masks are downscaled if any scaling was applied
            gt = np.array(ndimage.center_of_mass(gt)[::-1])*sfac  # (x, y)
        gt = np.reshape(gt, (1,2))
    else:
        gt = gt.squeeze()
        gt_has_pupil = not np.isnan(gt[0,0])

    return gt, gt_has_pupil


def do_hourglass_prediction(model, device, file, im, pure_df, confidence_threshold, crop_size, sfac):
    assert im.shape[0] == 1, "Make sure only one image per batch!"
    im   =   im.squeeze().detach().numpy()
    sfac = sfac.squeeze().detach().numpy()

    pure_x, pure_y, confidence = get_pure_pupil(file, pure_df)

    # if confidence >= threshold -> use pure pupil estimate to center crop else use naive
    # else, get center crop, detect pupil, and recrop centered on detected pupil
    if confidence < confidence_threshold:
        X, _, ox, oy = dataset_loaders.process_image(im, None, None, crop_size, crop_mode=1, device=device)
        pred_p, _, _ = model.predict(X, ox, oy, sfac)
        if not np.isnan(pred_p[0,0]):
            X, _, ox, oy = dataset_loaders.process_image(im, None, None, crop_size, crop_mode=2, crop_center=(pred_p[0,0]/sfac, pred_p[0,1]/sfac), device=device)
    else:
        X, _, ox, oy = dataset_loaders.process_image(im, None, None, crop_size, crop_mode=2, crop_center=(int(pure_x/sfac), int(pure_y/sfac)), device=device)

    pred_p, pred_m, pred_maps = model.predict(X, ox, oy, sfac)

    return pred_p, pred_m, confidence, pred_maps, [ox,oy]

def plot_hourglass_prediction(im, offset, gt_pos, pred_p, pred_m, pmaps, sfac):
    im   =   im.squeeze().detach().numpy()
    sfac = sfac.squeeze().detach().numpy()
    nmaps = len(pmaps)

    # prep for drawing: gt_pos and prediction are in original image size,
    # but image and offsets are downscaled by sfac. Correct gt_pos and prediction
    gt_pos /= sfac
    pred_p /= sfac
    # find best two CRs
    first, second = np.argsort(pred_m[1:])[::-1][:2]

    fig, axs = plt.subplots(1, nmaps+2)
    for a in axs.ravel():
        a.axis("off")

    axs[0].imshow(im, cmap='gray')
    axs[0].scatter(*gt_pos[0,:], marker='+', c='red')
    rect = patches.Rectangle(offset, pmaps.shape[1], pmaps.shape[2], linewidth=1, edgecolor='r', facecolor='none')
    axs[0].add_patch(rect)
    if len(gt_pos)>1:
        axs[0].scatter(gt_pos[1:,0], gt_pos[1:,1], marker='+', c='green')

    axs[-1].imshow(im, cmap='gray')
    axs[-1].scatter(*pred_p[0,:], marker='+', c='red')
    axs[-1].scatter(pred_p[1:,0], pred_p[1:,1], marker='+', c='green')
    axs[-1].scatter(*pred_p[first+1,:], marker='o', facecolors='none', edgecolors='green')
    axs[-1].scatter(*pred_p[second+1,:], marker='o', facecolors='none', edgecolors='green')

    for i, p in enumerate(pmaps):
        axs[i+1].imshow(p, cmap='seismic')
        axs[i+1].text(0.5, -0.1, round(pred_m[i], 3), transform=axs[i + 1].transAxes,
                                  horizontalalignment='center', verticalalignment='center', fontsize=20)
        
def report_hourglass(df, isChugh, nCR, confidence_threshold):
    # average error for all pupil
    sel = df.has_pupil & ~np.isnan(df.pred_px_p)
    print(f"Pure + LEyes + Threshold ({confidence_threshold}):\n"
          f"Mean pixel error excluding blinks {df[sel]['error_p'].mean()}"
          )

    # select two best CRs, by peak value
    CRs = np.arange(1,nCR+1)
    CR_values = [f'pred_v_cr{i}' for i in CRs]
    bestCRs = CRs[np.argsort(df[CR_values], axis=1)[:, :-2-1:-1]]
    df['CR_first']  = bestCRs[:,0]
    df['CR_second'] = bestCRs[:,1]
    CR_values.extend(['CR_first','CR_second'])

    if isChugh:
        # CR analyses
        df['valid_LEyes'] = df.apply(lambda x: not np.isnan(x['pred_px_p']) and np.all(x[[f"pred_v_cr{x['CR_first']}",f"pred_v_cr{x['CR_second']}"]]>=1), axis=1)
        # additionally apply Chugh's criterion
        CR_errors = [f'error_cr{i}' for i in range(1,nCR+1)]
        df['valid_Chugh'] = df[CR_errors].apply(lambda x: np.sum(x<=5)>=2, raw=True, axis=1)
        df['valid'] = df['valid_LEyes'] & df['valid_Chugh']

        # get valid rate (called matching accuracy by Chugh)
        print(f"Overall matching accuracy {df['valid'].mean()}")

        # get error for all CRs and two selected CRs
        if isChugh:
            df['error_cr_matched'] = np.nan
            df['error_cr_best'] = np.nan

            df.loc[df.valid,'error_cr_matched'] = df.loc[df.valid].apply(lambda x: x[[c for c in CR_errors if x[c]<=5]].mean(), axis=1)
            print(f"Average error across matched CRs {df['error_cr_matched'].mean()}")

            best_2_matched = df.valid & df.apply(lambda x: np.all(x[[f"error_cr{x['CR_first']}",f"error_cr{x['CR_second']}"]]<=5), axis=1)
            df.loc[best_2_matched,'error_cr_best'] = df.loc[best_2_matched].apply(lambda x: x[[f"error_cr{x['CR_first']}",f"error_cr{x['CR_second']}"]].mean(), axis=1)
            print(f"Average error across best two CRs {df['error_cr_best'].mean()}")

    return df