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

dataset = 'EDS2020'   # Chugh or EDS2020

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
isChugh = dataset=='Chugh'
IMAGE_SIZE, N_CRs, segment_pipeline, _, image_pipeline = simulations.get_hourglass_pipeline(dataset=dataset, stage=1)

n = 10
fig, axs = plt.subplots(n, N_CRs+2)
for r in range(n):
    output_image,masks = segment_pipeline.update()()
    for i in range(masks.shape[0]):
        axs[r, i+1].imshow(masks[i], cmap='gray', vmin=0, vmax=1)
    axs[r, 0].imshow(output_image, cmap='gray', vmin=0, vmax=1)
plt.show()

"""## 2. Set up training"""
batch_size = 8
num_epochs = 30 if isChugh else 500
lr = 1e-4
scheduler_gamma = 0.9
earlystop_tolerance = 30
earlystop_mindelta = 0.01 if isChugh else 0.

uniform_dim = 64*4
model = models.Virnet(device=device, num_cr=N_CRs, conv_dims=uniform_dim, base_dims=uniform_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

validation_set_size = 400 if isChugh else 300
validation_set, validation_labels = list(zip(*[segment_pipeline.update()() for _ in range(validation_set_size)]))
validation_set = [torch.tensor(v, dtype=torch.float) for v in validation_set]
validation_labels = [torch.tensor(np.array(v)) for v in validation_labels]
validation_loader = DataLoader(list(zip(validation_set, validation_labels)), batch_size=batch_size, shuffle=True)

generator = dt.generators.PyTorchContinuousGenerator(
    segment_pipeline,
    batch_size=batch_size,
    min_data_size=2e4 if isChugh else 1e3,
    max_data_size=3e4 if isChugh else 5e3,
    max_epochs_per_sample=1
)

if isChugh:
    weight, alpha, gamma = 1, 1, 2
else:
    weight, alpha, gamma = 100, 0.25, 20
criterion = training.BCElogitsWithDiceFocalLoss(weight, alpha=alpha, gamma=gamma, reduction="mean")
earlystopper = training.EarlyStopping(patience=earlystop_tolerance, verbose=False, delta=earlystop_mindelta, path=f'{dataset}_stage_1.pt')
scheduler = ExponentialLR(optimizer, gamma=scheduler_gamma)

"""## 3. Train model on simulation"""
with generator:
    for epoch in range(num_epochs):
        print(f"\n| Epoch {epoch+1} / {num_epochs} lr: {scheduler.get_last_lr() if scheduler else lr} =======================")

        epoch_train_loss = training.train_step(device, generator, model, criterion, optimizer, scheduler)
        epoch_val_loss = training.evaluation_step(device, validation_loader, model, criterion)

        print(f"|   Epoch Train Loss: {epoch_train_loss}, Epoch Validation Loss: {epoch_val_loss}")

        earlystopper(epoch_val_loss, model)  # best model is also cached in this step
        if earlystopper.early_stop:
            print("Early stopping")
            break

        generator.on_epoch_end()