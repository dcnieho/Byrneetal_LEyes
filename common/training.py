import numpy as np

import tensorflow.keras.backend as K
import torch
import torch.nn as nn
from torchvision.ops import focal_loss
from torchmetrics import Dice


def scaled_pixel_error(T, P):
    return K.mean(K.sqrt(K.sum(K.square(T - P), axis=-1)))
def pixel_error(T, P, IMAGE_SIZE):
    return K.mean(K.sqrt(K.sum(K.square(T - P), axis=-1))) * IMAGE_SIZE

# Function that retrieves the location of a feature's center
# and divides it by image size to get values between 0 and 1
def get_scaled_feature_location(image, IMAGE_SIZE, feature='position'):
    return image.get_property(feature) / IMAGE_SIZE

def freeze_layers(model, layer_indices):
    """
    Freezes the layers with the given indices in a Keras model.

    :param model: The Keras model.
    :param layer_indices: A list of layer indices or a slice object to freeze.
    """
    if isinstance(layer_indices, slice):
        start, stop, step = layer_indices.indices(len(model.layers))
        layer_indices = range(start, stop, step)
    for i, layer in enumerate(model.layers):
        if i in layer_indices:
            layer.trainable = False
            print(f"Layer '{layer.name}' frozen.")
        else:
            layer.trainable = True
            print(f"Layer '{layer.name}' unfrozen.")



def train_step(device, dataloader, model, loss_fn, optimizer, scheduler=None):
    train_loss = 0.0
    model.train()

    for X, y in dataloader:
        pred = model(X.to(device).unsqueeze(1))
        if isinstance(pred,tuple):
            loss = loss_fn(torch.cat(pred, 1).cpu(), y)
        else:
            loss = loss_fn(pred.cpu(), y.unsqueeze(1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    print()

    if scheduler:
        scheduler.step()
    return train_loss / len(dataloader)

def evaluation_step(device, dataloader, model, loss_fn):
    eval_loss = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device).unsqueeze(1))
            if isinstance(pred,tuple):
                loss = loss_fn(torch.cat(pred, 1).cpu(), y.squeeze().float())
            else:
                loss = loss_fn(pred.cpu(), y.unsqueeze(1).float())

            eval_loss += loss.item()
        print()
        return eval_loss / len(dataloader)



class DiceLoss(nn.Module):
    """Wrapper for dice """
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.dice = Dice()

    def forward(self, pred, target):
        return 1 - self.dice(pred, target.int())

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="none"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        return focal_loss.sigmoid_focal_loss(pred, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)

class BCElogitsWithDiceFocalLoss(nn.Module):
    def __init__(self, weight, alpha=0.25, gamma=2, reduction="none"):
        super(BCElogitsWithDiceFocalLoss, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha, gamma, reduction)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target):
        return self.bceloss(pred, target) + self.dice(self.sigmoid(pred), target) + self.focal(pred, target)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.path:
            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model, self.path)
        self.val_loss_min = val_loss