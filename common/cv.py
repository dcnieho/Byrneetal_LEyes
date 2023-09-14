# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math


kernel_pup  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15, 15))
kernel_pup2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7))
kernel_cr   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))


def detect_pupil_from_thresholded(thresholded, size_limits=None, symmetry_tresh=0.5, fill_thresh=0.2, kernel=kernel_pup, window_name=None):
    # Compute center location of image
    im_height, im_width = thresholded.shape
    center_x, center_y = im_width/2, im_height/2

    # Close  holes in the pupil, e.g., created by the CR
    blobs = cv2.morphologyEx(thresholded,cv2.MORPH_OPEN,kernel)
    blobs = cv2.morphologyEx(blobs,cv2.MORPH_CLOSE,kernel)

    # Visualized blobs if windown name given
    if window_name:
        cv2.imshow(window_name, blobs)

    # Find countours of the detected blobs
    blob_contours, hierarchy  = cv2.findContours(blobs,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    # Find pupil but checking one blob at the time. Pupils are round, so
    # add checks for 'roundness' criteria
    # If serveral blobs are found, select the one
    # closest to the center
    '''
    For a blob to be a pupil candidate
    1. blob must have the right area
    2. must be circular
    '''

    pupil_detected = False
    old_distance_image_center = np.inf
    for i, cnt in enumerate(blob_contours):

        # Take convex hull of countour points to alleviate holes
        cnt = cv2.convexHull(cnt)

        # Only contours with enouth points are of interest
        if len(cnt) < 10:
            continue

        # Compute area and bounding rect around blob
        temp_area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        x, y, width, height = rect
        radius = 0.25 * (width + height)
        r1,r2 = width,height
        if r1>r2:
            r1,r2 = r2,r1

        # Check area and roundness criteria
        area_condition = True if size_limits is None else (size_limits[0] <= temp_area <= size_limits[1])
        symmetry_condition = (abs(1 - float(r1)/float(r2)) <= symmetry_tresh)
        fill_condition = (abs(1 - (temp_area / (np.pi * radius**2))) <= fill_thresh)

        # If these criteria are fulfilled, a pupil is probably detected
        if area_condition and symmetry_condition and fill_condition:
            # Compute moments of blob
            moments = cv2.moments(cnt)

            # Compute blob center of gravity from the moments
            cx, cy = moments['m10']/moments['m00'], \
                     moments['m01']/moments['m00']

            # Compute distance blob - image center
            distance_image_center = np.sqrt((cx - center_x)**2 +
                                            (cy - center_y)**2)
            # Check if the current blob-center is closer
            # to the image center than the previous one
            if distance_image_center < old_distance_image_center:
                pupil_detected = True

                # Store pupil variables
                contour_points = cnt
                area = temp_area

                cx_best = cx
                cy_best = cy

                # Fit an ellipse to the contour, and compute its area
                ellipse = cv2.fitEllipse(contour_points.squeeze().astype('int'))
                (x_ellipse, y_ellipse), (MA, ma), angle = ellipse
                area_ellipse = np.pi / 4.0 * MA * ma

                old_distance_image_center = distance_image_center

    # If no potential pupil is found, due to e.g., blinks,
    # return nans
    if not pupil_detected:
        cx_best = np.nan
        cy_best = np.nan
        area = np.nan
        contour_points = np.nan
        x_ellipse = np.nan
        y_ellipse = np.nan
        MA = np.nan
        ma = np.nan
        angle = np.nan
        area_ellipse = np.nan

    pupil_features = {'cog':(cx_best, cy_best), 'area':area, 'contour_points': contour_points,
                      'ellipse' : ((x_ellipse, y_ellipse), (MA, ma), angle,
                                    area_ellipse)}
    return pupil_features
def detect_pupil(img, intensity_threshold, size_limits, window_name=None):
    ''' Identifies pupil blob
    Args:
        img - grayscale eye image
        intensity_threshold - threshold used to find pupil area
        size_limite - [min_pupil_size, max_pupil_size]
        window_name - plots detected blobs is window
                        name is given

    Returns:
        (cx, cy) - center of gravity binary pupil blob
        area -  area of pupil blob
        countour_points - contour points of pupil
        ellipse - parameters of ellipse fit to pupil blob
            (x_centre,y_centre),(minor_axis,major_axis),angle, area

    '''
    # Threshold image to get binary image
    ret,thresh = cv2.threshold(img, intensity_threshold, 255, cv2.THRESH_BINARY)

    return detect_pupil_from_thresholded(thresh, size_limits=size_limits, window_name=window_name)

#%%
def detect_cr(img, intensity_threshold, size_limits,
              pupil_cr_distance_max, pup_center, no_cr=2, cr_img_size = (20,20),
              window_name=None):
    ''' Identifies cr blob (must be located below pupil center)
    Args:
        img - grayscale eye image
        intensity_threshold - threshold used to find cr area(s)
        size_limite - [min_cr_size, max_cr_size]
        pupil_cr_distance_max - maximum allowed distance between
                                pupil and CR
        no_cr - number of cr's to be found

    Returns:
        cr - cr featuers

    '''
    if np.isnan(pup_center[0]):
        return []

    # Threshold image to get binary image
    ret,thresh1 = cv2.threshold(img, intensity_threshold,
                               255,cv2.THRESH_BINARY)

    # Close  holes in the cr, if any
    blobs = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel_cr)
    blobs = cv2.morphologyEx(blobs,cv2.MORPH_CLOSE,kernel_cr)

    if window_name:
        cv2.imshow(window_name, blobs)

    # Find countours of the detected blobs
    blob_contours, hierarchy = cv2.findContours(blobs,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    cr = []
    for i, cnt in enumerate(blob_contours):
        # Only contours with enouth points are of interest
        if len(cnt) < 4:
            continue

        # Compute area and bounding rect around blob
        temp_area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        x, y, width, height = rect
        radius = 0.25 * (width + height)
        r1,r2 = width,height
        if r1>r2:
            r1,r2 = r2,r1

        # Check area and roundness criteria
        area_condition = (size_limits[0] <= temp_area <= size_limits[1])
        symmetry_condition = (abs(1 - float(r1)/float(r2)) <= 0.7)
        fill_condition = (abs(1 - (temp_area / (np.pi * radius**2))) <= 0.7)

        # If these criteria are fulfilled, a pupil is probably detected
        if area_condition and symmetry_condition and fill_condition:
            # Compute moments of blob
            moments = cv2.moments(cnt)

            # Compute blob center of gravity from the moments
            # Coordinate system (0, 0) upper left
            cx, cy = moments['m10']/moments['m00'], \
                     moments['m01']/moments['m00']

            # Check distance between pupil and cr
            d = np.sqrt((cx - pup_center[0])**2 + (cy - pup_center[1])**2)
            if d > pupil_cr_distance_max:
                # print('cr too far away from pupil')
                continue

            if cy < (pup_center[1] - 0):
                # print('cr1 above pupil center')
                continue

            cr.append([cx, cy, temp_area])

    # if more crs than expected are found, then take the closest to the center
    if len(cr) > no_cr:
        dist = []
        for c in cr:
            dist.append(np.sqrt((c[0] - pup_center[0])**2 + \
                                (c[1] - pup_center[1])**2))

        # sort and select the closest distances
        idx = np.argsort(dist)
        cr = [cr[i] for i in idx[:no_cr]]

    # If the correct number of cr's are detected,
    # distinguish between them using x-position, i.e.,
    # give them an identity, cr1, cr2, cr2, etc.
    if len(cr) == no_cr:
        x_pos = []
        for c in cr:
            x_pos.append(c[0])

        # sort
        idx = np.argsort(x_pos)
        cr = [cr[i] for i in idx]

    return cr


def img_cutout(img,pos,cutout_sz,mode=2,filler=0):
    # Cut out image patch around pupil center location
    cx,cy = pos
    half_width  = cutout_sz[0]/2
    half_height = cutout_sz[1]/2
    im_height, im_width = img.shape

    # mode: either move cutout to fit on image (mode 1), or
    # replace parts of cutout beyond image with filler
    padding = [0,0,0,0] # left, top, right, bottom

    # get cutout pos
    x_range = [int(cx - half_width ), int(cx + half_width )]
    y_range = [int(cy - half_height), int(cy + half_height)]
    # make sure we don't run off the image
    if x_range[0]<0:
        x_range[0] = 0
    sdiff = x_range[1]-x_range[0]
    if x_range[0]==0 and sdiff<cutout_sz[0]:
        if mode==1:
            x_range[1] += (cutout_sz[0]-sdiff)
        else:
            padding[0] = cutout_sz[0]-sdiff

    if y_range[0]<0:
        y_range[0] = 0
    sdiff = y_range[1]-y_range[0]
    if y_range[0]==0 and sdiff<cutout_sz[1]:
        if mode==1:
            y_range[1] += (cutout_sz[1]-sdiff)
        else:
            padding[1] = cutout_sz[1]-sdiff

    if x_range[1]>im_width:
        x_range[1] = im_width
    sdiff = x_range[1]-x_range[0]
    if x_range[1]==im_width and sdiff<cutout_sz[0]:
        if mode==1:
            x_range[0] -= (cutout_sz[0]-sdiff)
        else:
            padding[2] = cutout_sz[0]-sdiff

    if y_range[1]>im_height:
        y_range[1] = im_height
    sdiff = y_range[1]-y_range[0]
    if y_range[1]==im_height and sdiff<cutout_sz[1]:
        if mode==1:
            y_range[0] -= (cutout_sz[1]-sdiff)
        else:
            padding[3] = cutout_sz[1]-sdiff

    cutout  = img[y_range[0] : y_range[1],
                  x_range[0] : x_range[1]]
    off     = [x_range[0], y_range[0]]
    if any([p!=0 for p in padding]):
        if padding[0]:
            pad = np.zeros((cutout.shape[0],padding[0]),img.dtype)
            if filler!=0:
                pad[:,:] = filler
            cutout = np.hstack((pad, cutout))
            off[0] -= padding[0]
        if padding[1]:
            pad = np.zeros((padding[1],cutout.shape[1]),img.dtype)
            if filler!=0:
                pad[:,:] = filler
            cutout = np.vstack((pad, cutout))
            off[1] -= padding[1]
        if padding[2]:
            pad = np.zeros((cutout.shape[0],padding[2]),img.dtype)
            if filler!=0:
                pad[:,:] = filler
            cutout = np.hstack((cutout, pad))
        if padding[3]:
            pad = np.zeros((padding[3],cutout.shape[1]),img.dtype)
            if filler!=0:
                pad[:,:] = filler
            cutout = np.vstack((cutout, pad))
    return cutout, off

def make_mask(ref_img,center,radii,angle=0,val=255,bg_val=0):
    subPixelFac = 8
    mask = np.zeros_like(ref_img)
    if bg_val!=0:
        mask[:,:] = bg_val

    center = [int(np.round(x*subPixelFac)) for x in center]
    if not isinstance(radii,list):
        radii = [radii]
    radii  = [int(np.round(x*subPixelFac)) for x in radii]
    if len(radii)==1:
        mask = cv2.circle (mask, center, radii[0]            , val, -1, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))
    else:
        mask = cv2.ellipse(mask, center, radii, angle, 360, 0, val, -1, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))
    return mask