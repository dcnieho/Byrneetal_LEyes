import numpy as np
import copy
import sys

import torch
import torch.nn as nn
from einops.layers.torch import Reduce
import segmentation_models_pytorch as smp

import cv
import ellipse

class PupilNet(nn.Module):
    def __init__(self):
        super(PupilNet, self).__init__()
        self.backbone = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        )

    def forward(self, inputs):
        return self.backbone(inputs)

    def predict(self, inputs, sfac=None, offsets=None):
        with torch.no_grad():
            pred = self.forward(inputs)
        pred = nn.Sigmoid()(pred).detach().cpu().squeeze()
        if pred.ndim==2:
            pred = pred.unsqueeze(0)
        mask = pred > 0.99

        pupils = []
        for i, m in enumerate(mask):
            pup = cv.detect_pupil_from_thresholded(m.numpy().astype('uint8')*255, symmetry_tresh=0.3, kernel=cv.kernel_pup2)
            pupil = {'centroid': (pup['ellipse'][0]),
                    'axis_major_radius': pup['ellipse'][1][0]/2,
                    'axis_minor_radius': pup['ellipse'][1][1]/2,
                    'orientation': pup['ellipse'][2]
                    }
            max_rad = max(pupil['axis_major_radius'],pupil['axis_minor_radius'])
            pupil['too_close_edge'] = pupil['centroid'][0] < max_rad or pupil['centroid'][0]>m.shape[1] or pupil['centroid'][1] < max_rad or pupil['centroid'][1]>m.shape[0]

            if (not np.isnan(pupil['centroid'][0])) and (sfac is not None):
                el = ellipse.my_ellipse((*(pupil['centroid']),pupil['axis_major_radius'],pupil['axis_minor_radius'], pupil['orientation']/180*np.pi))
                tform = ellipse.scale_2d(sfac,sfac)
                nelpar = el.transform(tform)[0][:-1]
                pupil['oripupil'] = copy.deepcopy(pupil)
                pupil['centroid'] = (nelpar[0],nelpar[1])
                pupil['axis_major_radius'] = nelpar[2]
                pupil['axis_minor_radius'] = nelpar[3]
                pupil['orientation'] = nelpar[4]/np.pi*180

            if (not np.isnan(pupil['centroid'][0])) and (offsets[i] is not None):
                if sfac is None:
                    pupil['oripupil'] = copy.deepcopy(pupil)
                pupil['centroid'] = tuple(x+y for x,y in zip(pupil['centroid'],offsets[i].numpy().flatten()))

            pupil["mask"] = m
            pupil["pmap"] = pred[i]
            pupils.append(pupil)

        return pupils


class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x


class residual_block(nn.Module):
    def __init__(self, in_c, h_c):
        super().__init__()

        """ Convolutional layer """
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv2d(in_c, h_c, kernel_size=1, padding="same")

        self.b2 = batchnorm_relu(h_c)
        self.c2 = nn.Conv2d(h_c, h_c, kernel_size=3, padding="same")

        self.b3 = batchnorm_relu(h_c)
        self.c3 = nn.Conv2d(h_c, in_c, kernel_size=1, padding="same")

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)

        x = self.b2(x)
        x = self.c2(x)

        x = self.b3(x)
        x = self.c3(x)

        skip = x + inputs
        return skip


class encoder_block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.r = residual_block(in_c, int(in_c // 2))
        self.downsample = nn.MaxPool2d(2)

    def forward(self, inputs):
        skip = self.r(inputs)
        x = self.downsample(skip)
        return x, skip


class decoder_block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = residual_block(in_c, int(in_c // 2))

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = x + skip
        x = self.r(x)
        return x

class Hourglass(nn.Module):
    """
    Hourglass model made of residual blocks based on Niu et al. 2021

    - the hourglass halves the feature map after each layer but keeps the depth the same (64)
    - the residual blocks consist of three convolutions. c(64, 32, 1x1) -> c(32, 32 3x3) -> c(32, 64, 1x1) -> add input to output
    """

    def __init__(self, conv_dims=64, base_dims=64):
        super().__init__()

        """ Input """
        self.c0 = nn.Conv2d(1, conv_dims, kernel_size=1)
        self.br0 = batchnorm_relu(conv_dims)

        """ Encoder  """
        self.e1 = encoder_block(conv_dims)
        self.e2 = encoder_block(conv_dims)
        self.e3 = encoder_block(conv_dims)
        self.e4 = encoder_block(conv_dims)
        self.e5 = encoder_block(conv_dims)
        self.e6 = encoder_block(conv_dims)

        """ Bridge """
        self.b1 = residual_block(base_dims, int(base_dims // 2))
        self.b2 = residual_block(base_dims, int(base_dims // 2))
        self.b3 = residual_block(base_dims, int(base_dims // 2))

        """ Decoder """
        self.d1 = decoder_block(conv_dims)
        self.d2 = decoder_block(conv_dims)
        self.d3 = decoder_block(conv_dims)
        self.d4 = decoder_block(conv_dims)
        self.d5 = decoder_block(conv_dims)
        self.d6 = decoder_block(conv_dims)

    def forward(self, inputs):
        """ Input """
        inp = self.c0(inputs)
        inp = self.br0(inp)

        """ Encoder"""
        x, skip1 = self.e1(inp)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)
        x, skip6 = self.e6(x)

        """ Bridge """
        b = self.b1(x)
        b = self.b2(b)
        b = self.b3(b)

        """ Decoder """
        d = self.d1(b, skip6)
        d = self.d2(d, skip5)
        d = self.d3(d, skip4)
        d = self.d4(d, skip3)
        d = self.d5(d, skip2)
        d = self.d6(d, skip1)

        return d

class Virnet(nn.Module):
    def __init__(self, device, num_cr, conv_dims=64, base_dims=64, CR2pup=True):
        super().__init__()

        self.device = device
        self.backbone = Hourglass(conv_dims, base_dims).to(self.device)
        self.num_cr = num_cr

        """ Output """
        # CR head
        self.ccr = nn.Conv2d(conv_dims, self.num_cr, kernel_size=1)
        self.bcr = nn.BatchNorm2d(self.num_cr)

        # Attention
        self.max_pooling_layer = Reduce('b c h w -> b 1 h w', 'max')
        self.avg_pooling_layer = Reduce('b c h w -> b 1 h w', 'mean')
        self.ca = nn.Conv2d(2 if CR2pup else 1
                            , 1
                            , kernel_size=7
                            , padding="same")

        # Pupil head
        self.cp = nn.Conv2d(conv_dims, 1, kernel_size=1)
        self.bp = nn.BatchNorm2d(1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.CR2pup = CR2pup  # Attention from CR fed to pupil? If false, attention is fed from pupil to CR

    def forward(self, inputs):
        x = self.backbone(inputs)

        if self.CR2pup:
            cr = self.ccr(x)
            cr = self.bcr(cr)

            # Attention
            max = self.max_pooling_layer(self.relu(cr))
            avg = self.avg_pooling_layer(self.relu(cr))

            attention = torch.cat([max, avg], 1)
            attention = self.ca(attention)
            attention = self.sigmoid(attention)

            # Pupil head
            cp = torch.multiply(attention, x)
            cp = self.cp(cp)
            cp = self.bp(cp)
        else:
            cp = self.cp(x)
            cp = self.bp(cp)

            attention = self.ca(cp)
            attention = self.sigmoid(attention)

            cr = torch.multiply(attention, x)
            cr = self.ccr(cr)
            cr = self.bcr(cr)

        return cp, cr

    def predict(self, X, ox, oy, sfac):
        # pupil and CR predictions
        with torch.no_grad():
            if X.dim() == 2:
                X = X.unsqueeze(0)
            if X.dim() == 3:
                X = X.unsqueeze(0)
            pred_p, pred_cr = self.forward(X)

            pred = torch.cat([pred_p, pred_cr], 1)
            pred = torch.nn.ReLU()(pred)
            pred = pred.detach().cpu().numpy().squeeze()

            pupil_m, pupil_p = get_peak(pred[0])
            pupil_p = (pupil_p+[ox, oy])*sfac

            crs_m = []
            crs_p = []
            for c in pred[1:]:
                cr_m, cr_p = get_peak(c)
                crs_m.append(cr_m)
                crs_p.append((cr_p+[ox, oy])*sfac)

        out_p = np.vstack((pupil_p,crs_p))
        out_m = np.array([pupil_m]+crs_m)
        return out_p, out_m, pred

def load_virnet(path):
    # ensure models can be found when loading
    # Chugh model was defined in main during training
    import __main__
    setattr(__main__, "Virnet", Virnet)
    setattr(__main__, "Hourglass", Hourglass)
    setattr(__main__, "decoder_block", decoder_block)
    setattr(__main__, "encoder_block", encoder_block)
    setattr(__main__, "residual_block", residual_block)
    setattr(__main__, "batchnorm_relu", batchnorm_relu)
    # EDS2020 model was defined in the 'model' module when training
    sys.modules['model'] = sys.modules['models']

    # load actual model
    m = torch.load(path)

    # clean up
    del sys.modules['model']
    delattr(__main__, "Virnet")
    delattr(__main__, "Hourglass")
    delattr(__main__, "decoder_block")
    delattr(__main__, "encoder_block")
    delattr(__main__, "residual_block")
    delattr(__main__, "batchnorm_relu")
    return m


def get_peak(pm):
    m = np.max(pm)
    if m == 0:
        peak = np.array([np.nan, np.nan])
    else:
        peak = (pm == np.max(pm)).nonzero()
        peak = np.array([peak[1][0],peak[0][0]], dtype=np.float32)  # (x,y)
    return m, peak