import sys
import os
import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


sys.path.insert(1, os.path.join(sys.path[0], '..', 'common'))
import dataset_loaders
import evaluate
import models
from image_sizes import HOURGLASS_IMAGE_SIZE as IMAGE_SIZE


dataset = 'EDS2020'           # Chugh or EDS2020
do_debug = False
scale_fac = 2               # should be 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}')

isChugh = dataset=='Chugh'
if isChugh:
    model_path = pathlib.Path(r'../trained_model/Chugh.pt')
    dataset_path = pathlib.Path(r"../datasets/Chugh")
else:
    model_path = pathlib.Path(r'../trained_model/EDS2020.pt')
    dataset_path = pathlib.Path(r"../datasets/openEDS2020_partialSeg")




def evaluate_pure_Leyes_wThreshold(model, pure_df, dataloader, gt_is_mask, crop_size, isChugh, confidence_threshold=0.65):
    filenames = []
    target_px = []
    target_py = []
    has_pupil = []
    pred_px = []
    pred_py = []
    pred_v  = []
    confs = []

    for files, im, gt, sfac in tqdm(dataloader, unit=' images'):
        gt_pos, gt_has_pupil = evaluate.get_ground_truth(gt, gt_is_mask, sfac)
        pred_p, pred_m, confidence, _, _ = evaluate.do_hourglass_prediction(model, device, files[0], im, pure_df, confidence_threshold, crop_size, sfac)
        filenames.append(files[0])
        target_px.append(gt_pos[:,0])
        target_py.append(gt_pos[:,1])
        pred_px.append(pred_p[:,0])
        pred_py.append(pred_p[:,1])
        pred_v .append(pred_m)
        confs.append(confidence)
        has_pupil.append(gt_has_pupil)

    df = pd.DataFrame(data=
    {
        "filename": filenames,
        "has_pupil": has_pupil,
        "pure_confidence": confs,
    })
    for i in range(len(pred_px[0])):
        if i==0:
            label = 'p'
        else:
            label = f'cr{i}'
        if len(target_px[0])>i:
            df[f'target_px_{label}'] = [x[i] for x in target_px]
            df[f'target_py_{label}'] = [x[i] for x in target_py]
        df[f'pred_px_{label}']   = [x[i] for x in pred_px]
        df[f'pred_py_{label}']   = [x[i] for x in pred_py]
        df[f'pred_v_{label}']    = [x[i] for x in pred_v]
        if len(target_px[0])>i:
            df[f'error_{label}'] = np.sqrt(
                (df[f'pred_px_{label}'] - df[f'target_px_{label}']) ** 2 + (df[f'pred_py_{label}'] - df[f'target_py_{label}']) ** 2)

    return evaluate.report_hourglass(df, isChugh, len(pred_v[0])-1, confidence_threshold)


if __name__ == '__main__':
    model = models.load_virnet(model_path)
    model.eval()

    if isChugh:
        fnames, images, labels, sfacs = dataset_loaders.get_Chugh_set(dataset_path, sfac=scale_fac, up_to_n_files=8 if do_debug else None)
        pure_df = pd.read_csv("pure_Chugh.tsv", sep='\t')
        real_test_loader = DataLoader(list(zip(fnames, images, labels, sfacs)), batch_size=1)
        gt_is_mask = False
        confidence_threshold = 0.7
    else:
        fnames, images, masks, sfacs = dataset_loaders.get_eds2020_set(dataset_path, sfac=scale_fac, up_to_n_files=8 if do_debug else None)
        pure_df = pd.read_csv("pure_EDS2020.tsv", sep='\t')
        real_test_loader = DataLoader(list(zip(fnames, images, masks, sfacs)), batch_size=1)
        gt_is_mask = True
        confidence_threshold = 0.9

    if do_debug:
        for files, im, gt, sfac in tqdm(real_test_loader, unit=' images'):
            gt_pos, gt_has_pupil = evaluate.get_ground_truth(gt, gt_is_mask, sfac)
            pred_p, pred_m, confidence, pmaps, off = evaluate.do_hourglass_prediction(model, device, files[0], im, pure_df, confidence_threshold, (IMAGE_SIZE,IMAGE_SIZE), sfac)
            evaluate.plot_hourglass_prediction(im, off, gt_pos, pred_p, pred_m, pmaps, sfac)
            plt.show()
    else:
        for ct in [0., 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]: # 0. means always use PuRe
            df = evaluate_pure_Leyes_wThreshold(model, pure_df, real_test_loader, gt_is_mask, (IMAGE_SIZE,IMAGE_SIZE), isChugh, confidence_threshold=ct)
            if ct==confidence_threshold:
                out_data_file= pathlib.Path('results') / f'{dataset}.tsv'
                df.to_csv(out_data_file, sep='\t', float_format='%.8f', na_rep='nan',index=False)