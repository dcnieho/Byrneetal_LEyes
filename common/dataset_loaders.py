# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import cv2
import pickle
from tqdm import tqdm

def crop_around_pos(image, w, h, px, py):
    x = max(px - w / 2, 0)
    y = max(py - h / 2, 0)

    if y + h >= image.shape[0]:
        y -= (y + h - image.shape[0])
    if x + w >= image.shape[1]:
        x -= (x + w - image.shape[1])

    crop_img = image[int(y):int(y + h), int(x):int(x + w)]

    cy, cx = crop_img.shape
    if cy != h or cx != w:
        raise RuntimeError(f"Cropped image does not have the expected size ({w}x{h}), but ({cx}x{cy})")

    return crop_img, x, y


def process_image(im, mask, feature_value, crop_size, crop_mode, crop_center=None, sfac=1, device=None):
    # sfac of 2 means image downscaled by a factor 2
    # crop size format (x,y)
    if mask is not None:
        if not mask.dtype=='bool':
            mask = mask == feature_value
        is_feature_present = np.any(mask)
        assert np.all(im.shape==mask.shape), "Mask not same size as image"
    else:
        is_feature_present = True

    if crop_size is None:
        crop_size = (int(im.shape[1]/sfac), int(im.shape[0]/sfac))

    if not isinstance(crop_size, tuple):
        if isinstance(crop_size, list):
            crop_size = (*crop_size,)
        else:
            crop_size = (crop_size, crop_size)
    if crop_center:
        crop_center = [int(x) for x in crop_center]

    # determine crop position
    if crop_mode==1 or not is_feature_present:
        # use image center
        y, x = [i/2 for i in im.shape]
    elif crop_mode==2 and crop_center:
        # user specified center
        x, y = crop_center

    if crop_mode>0:
        im, ox, oy = crop_around_pos(im, crop_size[0]*sfac, crop_size[1]*sfac, x, y)
        if mask is not None:
            mask, _, _ = crop_around_pos(mask, crop_size[0]*sfac, crop_size[1]*sfac, x, y)
    else:
        ox, oy = 0, 0

    if sfac!=1:
        im   = cv2.resize(im,crop_size,interpolation=cv2.INTER_LANCZOS4)
        if mask is not None:
            mask = cv2.resize(mask.astype('uint8'),crop_size,interpolation=cv2.INTER_NEAREST)==1

    if device is not None:
        im = torch.FloatTensor(im).to(device)

    return im, mask, ox, oy

def get_eds2019_set(dataset_path, crop_size, up_to_n_files=None, sfac=1):
    pupil_lbl = 3
    imgfs = list((dataset_path/'images').glob("*.png"))

    imgs    = []
    labels  = []
    offsets = []
    names   = []
    if up_to_n_files is not None:
        imgfs = imgfs[:up_to_n_files]
    for f in tqdm(imgfs, unit=' images'):
        im   = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        mask = np.load(dataset_path/'labels'/(f.stem +'.npy'))

        # get scaled crop around center of image
        im, mask, ox, oy = process_image(im, mask, pupil_lbl, crop_mode=1, crop_size=crop_size, sfac=sfac)

        imgs.append(torch.tensor(im, dtype=torch.float32).unsqueeze(0)/255)
        labels.append(torch.tensor(mask, dtype=float).unsqueeze(0))
        offsets.append(torch.tensor(np.asarray((ox,oy)), dtype=torch.float32).unsqueeze(0))
        names.append(f.name)

    return names, imgs, labels, offsets

def get_Chugh_set(dataset_path, sfac, only_test_set=False, up_to_n_files=None):
    assert sfac in [1,2], "scale factor must be 1 or 2"
    labels_path = dataset_path / "label.txt"
    imgfs = list(dataset_path.glob("*.jpg"))

    labeldf = pd.read_json(labels_path)

    # if only_test_set==True, evaluate only on his test set (up_to_n_files must be None for this to work, or part of test set may be missed)
    chugh_test_set = [2, 6, 15]

    fnames  = []
    images  = []
    labels  = []
    sfacs   = []
    if up_to_n_files is not None:
        imgfs = imgfs[:up_to_n_files]
    for f in tqdm(imgfs, unit=' images'):
        subj = int(f.name.split('_')[0][1:])
        if only_test_set and subj not in chugh_test_set:
            continue

        px = labeldf[f.name]['PupilCenter']['PupilX']
        py = labeldf[f.name]['PupilCenter']['PupilY']
        cxs = labeldf[f.name]['CornealReflectionLocations']['CornealX']
        cys = labeldf[f.name]['CornealReflectionLocations']['CornealY']
        l = np.array([
            [px, *cxs],
            [py, *cys]
        ]).astype(np.float64).T
        l[l==-1] = np.nan   # better value to indicate missing

        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)

        if im.shape[:2]==(480, 640) and sfac==2:
            # downscale images that are full resolution
            im = cv2.resize(im, (320,240))
            scale = 2
        else:
            # some images already are half resolution, so no need to downscale
            scale = 1

        fnames.append(f.name)
        images.append(im.astype(np.float32)/255)
        labels.append(l)
        sfacs .append(scale)

    return fnames, images, labels, sfacs

def get_eds2020_set(dataset_path, sfac, up_to_n_files=None):
    pupil_lbl = 3
    fnames  = []
    images  = []
    masks   = []
    sfacs   = []

    # load train and validate sets
    with open(dataset_path / "labels.txt") as f:
        lines = f.readlines()
    if up_to_n_files and len(lines)>up_to_n_files:
        lines = lines[0:up_to_n_files]

    for l in tqdm(lines, unit=' images'):
        l = l.strip()
        subj = l.split("/")[0]
        id = l.split("/")[-1].split("_")[-1][:-4]
        f  = dataset_path / subj / f"{id}.png"
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        mask = np.load(dataset_path / subj / f"label_{id}.npy")

        im, mask, _, _ = process_image(im, mask, pupil_lbl, None, crop_mode=0, sfac=sfac)

        fnames.append(f"{subj}/{f.name}")
        images.append(im.astype(np.float32))
        masks.append(mask)
        sfacs.append(sfac)


    # load test set
    if up_to_n_files and len(fnames)==up_to_n_files:
        return fnames, images, masks, sfacs
    with open(dataset_path / "test_sampleName_GT.pkl", "rb") as f:
        test_set = pickle.load(f)
        if up_to_n_files and len(fnames)<up_to_n_files:
            n_needed = up_to_n_files-len(fnames)
            test_set = {k:test_set[k] for k in list(test_set.keys())[0:n_needed]}

    for k, mask in tqdm(test_set.items(), unit=' images'):
        subj = k.split("/")[0]
        id = k.split("/")[-1].split("_")[-1][:-4]
        f  = dataset_path / subj / f"{id}.png"
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)

        im, mask, _, _ = process_image(im, mask, pupil_lbl, None, crop_mode=0, sfac=sfac)

        fnames.append(f"{subj}/{f.name}")
        images.append(im.astype(np.float32))
        masks.append(mask)
        sfacs.append(sfac)

    return fnames, images, masks, sfacs