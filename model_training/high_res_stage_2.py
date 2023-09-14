# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import deeptrack as dt
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from tensorflow.python.client import device_lib
device_lib.list_local_devices()


sys.path.insert(1, os.path.join(sys.path[0], '..', 'common'))
import simulations
import training

feature     = 'CR'  # CR or pupil
dataset_freq= 500   # 500 or 1000


"""## 1. Stimulus"""
if feature=='CR':
    IMAGE_SIZE, data_pipeline, image_pipeline = simulations.get_CR_pipeline(stage=2, freq=dataset_freq)
elif feature=='pupil':
    IMAGE_SIZE, data_pipeline, image_pipeline = simulations.get_pupil_pipeline(stage=2, freq=dataset_freq)

# test data pipeline
for p in range(5):
    output_image,position = data_pipeline.update()()
    print(f'position: {position}')

    plt.imshow(output_image, cmap='gray', vmin=0, vmax=1)
    plt.scatter(position[0], position[1], c="r", s=100, linewidths=4, marker="x")
    plt.show()


"""## 2. Network training, Transfer training"""
model = keras.models.load_model(f'high_res_{feature}_{dataset_freq}Hz_stage_1.h5', compile=False)
model.summary()

if feature=='CR':
    label_name = 'position'
    monitor = 'val_pixel_error'
    patience = 120 if dataset_freq==500 else 20
    early_stop_mode = "min"
    optimizer = tf.keras.optimizers.Adam
    decay_steps = 1000
    loss = "mse"
    validation_set_size = 300
    max_epochs_per_sample = 1
    epochs = 700
elif feature=='pupil':
    label_name = 'pupil_position'
    monitor = 'val_loss'
    patience = 5 if dataset_freq==500 else 10
    early_stop_mode = 'auto'
    optimizer = tf.keras.optimizers.experimental.AdamW if dataset_freq==500 else tf.keras.optimizers.Adam
    decay_steps = 10000
    loss = "mae"
    validation_set_size = 600
    max_epochs_per_sample = 2
    epochs = 40 if dataset_freq==500 else 100

if feature=='CR' and dataset_freq==500:
    training.freeze_layers(model, slice(None, 6))
else:
    training.freeze_layers(model, slice(None, 3))

lr = 1e-6
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    lr,
    decay_steps=decay_steps,
    decay_rate=0.95,
    staircase=True
)

opt = optimizer(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
)

model.compile(optimizer=opt, loss=loss, metrics=[lambda T,P: training.pixel_error(T, P, IMAGE_SIZE), training.scaled_pixel_error])

# set up early-stopping:
EARLY_STOPPING = tf.keras.callbacks.EarlyStopping(
    monitor=monitor,
    min_delta = 0.0001,
    verbose=1,
    patience=patience,
    mode=early_stop_mode,
    restore_best_weights=True
)

validation_set    = [image_pipeline.update()() for _ in range(validation_set_size)]
validation_labels = [training.get_scaled_feature_location(image, IMAGE_SIZE, label_name) for image in validation_set]

generator = dt.generators.ContinuousGenerator(
    image_pipeline,
    lambda im: training.get_scaled_feature_location(im, IMAGE_SIZE, label_name),
    min_data_size=int(1e3),
    max_data_size=int(2e4),
    batch_size=4,
    max_epochs_per_sample=max_epochs_per_sample
)

with generator:
    h = model.fit(
        generator,
        validation_data = (
            np.array(validation_set),
            np.array(validation_labels)
        ),
        epochs = epochs,
            callbacks = EARLY_STOPPING,
            shuffle = True
    )

plt.plot(h.history["loss"], 'g')
plt.plot(h.history["val_loss"], 'r')
plt.legend(["loss", "val_loss"])
plt.yscale('log')
plt.show()

validation_prediction = model.predict(np.array(validation_set))
labels = np.array(validation_labels)
for col in range(validation_prediction.shape[-1]):
    label_col = labels[:, col]
    prediction_col = validation_prediction[:, col]
    plt.scatter(label_col, prediction_col, alpha=0.1)

    plt.plot([np.min(label_col), np.max(label_col)],
             [np.min(label_col), np.max(label_col)], c='k')
    plt.show()

# Model save
model.save(f'high_res_{feature}_{dataset_freq}Hz_stage_2.h5')