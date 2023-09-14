# -*- coding: utf-8 -*-
import sys
import os
import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt


sys.path.insert(1, os.path.join(sys.path[0], '..', 'common'))
import dataset_loaders
import evaluate
from models import PupilNet
from image_sizes import EDS2019_IMAGE_SIZE as IMAGE_SIZE


model = torch.load('../trained_model/EDS2019.pt')
dataset_path = pathlib.Path(r"../datasets/openEDS2019_segmentation")
set_folders = ['train','validation','test']
batch_size = 8
do_debug = False
scale_fac = 2       # should be 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}')


def predict_and_plot(loader, model, dataset_path, sfac):
    na, X, y, o = next(iter(loader))

    pred = model.predict(X.to(device), sfac=sfac, offsets=o)

    for x, p, yy, n in zip(X, pred, y, na):
        evaluate.plot_unet_prediction(x, p, yy, n, dataset_path)
    plt.show()


if __name__ == '__main__':
    all_dfs = []
    for v in set_folders:
        names, imgs, labels, offsets = dataset_loaders.get_eds2019_set(dataset_path/v, IMAGE_SIZE, up_to_n_files=batch_size if do_debug else None, sfac=scale_fac)
        real_test_loader = DataLoader(list(zip(names, imgs, labels, offsets)), batch_size=batch_size)

        if do_debug:
            predict_and_plot(real_test_loader, model, dataset_path/v, scale_fac)
            plt.show()

        else:
            data = []
            for n, X, y, o in real_test_loader:
                pred = model.predict(X.to(device), sfac=scale_fac, offsets=o)
                for na, x, p, yy in zip(n,X,pred,y):
                    # adjust cutout and repredict if needed
                    do_plot = False
                    if p['too_close_edge']:
                        # pupil too close to the edge, recrop centered on detected pupil and repredict
                        im   = cv2.imread(str(dataset_path/v/'images'/na), cv2.IMREAD_GRAYSCALE)
                        x, yy, ox, oy = dataset_loaders.process_image(im, None, 3, IMAGE_SIZE, crop_mode=2, crop_center=p['centroid'], sfac=scale_fac)
                        im  = torch.FloatTensor(np.expand_dims(np.asarray([x]),1))
                        off = torch.FloatTensor(np.expand_dims(np.asarray((ox,oy)),1)).unsqueeze(0)
                        p   = model.predict(im.to(device), sfac=scale_fac, offsets=off)[0]

                dat = [na,v]
                dat += list(p['centroid'])
                dat += [p['axis_major_radius'],p['axis_minor_radius'],p['orientation']]
                data.append(dat)

            col_lbls = ['File_name','set',
                    'pup_x_unet', 'pup_y_unet',
                    'Ma_unet', 'ma_unet',
                    'angle_unet'
                ]
            all_dfs.append(pd.DataFrame(data, columns=col_lbls))

    if not do_debug:
        out_data_file= pathlib.Path('results') / 'EDS2019.tsv'
        pd.concat(all_dfs).to_csv(out_data_file, sep='\t', float_format='%.8f', na_rep='nan',index=False)