# -*- coding: utf-8 -*-
import random
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import deeptrack as dt

import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader


sys.path.insert(1, os.path.join(sys.path[0], '..', 'common'))
import simulations
import training
import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")

seed=42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


"""## 1. Stimulus"""
IMAGE_SIZE, segment_pipeline, image_pipeline = simulations.get_EDS2019_pipeline()

for p in range(5):
    output_image,mask = segment_pipeline.update()()

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(output_image, cmap='gray', vmin=0, vmax=1)
    axs[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    plt.show()

"""## 2. Set up training"""
batch_size = 8
num_epochs = 100
lr = 1e-4
scheduler_gamma = 0.9
earlystop_tolerance = 30
earlystop_mindelta = 0.

model = models.PupilNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

validation_set_size = 400
validation_set, validation_labels = list(zip(*[segment_pipeline.update()() for _ in range(validation_set_size)]))
validation_set    = [torch.tensor(v, dtype=torch.float) for v in validation_set]
validation_labels = [torch.tensor(np.array(v)) for v in validation_labels]
validation_loader = DataLoader(list(zip(validation_set, validation_labels)), batch_size=batch_size, shuffle=True)

generator = dt.generators.PyTorchContinuousGenerator(
    segment_pipeline,
    batch_size=batch_size,
    min_data_size=1e3,
    max_data_size=1e3,
    max_epochs_per_sample=1
)

weight, alpha, gamma = 100, 0.25, 20
criterion = training.BCElogitsWithDiceFocalLoss(weight, alpha=alpha, gamma=gamma, reduction="mean")
earlystopper = training.EarlyStopping(patience=earlystop_tolerance, verbose=False, delta=earlystop_mindelta, path='EDS2019.pt')
scheduler = ExponentialLR(optimizer, gamma=scheduler_gamma)

"""## 3. Train model on simulation"""
with generator:
    for epoch in range(num_epochs):
        print(f"\n| Epoch {epoch+1} / {num_epochs} lr: {scheduler.get_last_lr() if scheduler else lr} =======================")

        epoch_train_loss = training.train_step(device, generator, model, criterion, optimizer, scheduler)
        epoch_val_loss = training.evaluation_step(device, validation_loader, model, criterion)

        print(f"|   Epoch Train Loss: {epoch_train_loss}, Epoch Validation Loss: {epoch_val_loss}")

        earlystopper(epoch_val_loss, model) # best model is stored in this step
        if earlystopper.early_stop:
            print("Early stopping")
            break

        generator.on_epoch_end()