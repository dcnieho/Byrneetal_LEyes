# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import deeptrack as dt
import tensorflow as tf

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
    IMAGE_SIZE, data_pipeline, image_pipeline = simulations.get_CR_pipeline(stage=1, freq=dataset_freq)
elif feature=='pupil':
    IMAGE_SIZE, data_pipeline, image_pipeline = simulations.get_pupil_pipeline(stage=1, freq=dataset_freq)

# test data pipeline
for p in range(5):
    output_image,position = data_pipeline.update()()
    print(f'position: {position}')

    plt.imshow(output_image, cmap='gray', vmin=0, vmax=1)
    plt.scatter(position[0], position[1], c="r", s=100, linewidths=4, marker="x")
    plt.show()


"""## 2. Network training"""
if feature=='CR':
    label_name = 'position'
    optimizer = tf.keras.optimizers.Adam
    decay_steps = 5000
    decay_rate = 0.6
    conv_layers_dimensions = (64, 64, 128, 128, 256, 256, 512)
    dense_layers_dimensions = (64,32)
    loss = "mse"
    dropout = ()
    output_kernel_size = 3
    patience = 40
    validation_set_size = 300
    batch_size = 4
    max_epochs_per_sample = 1
    epochs = 700
elif feature=='pupil':
    label_name = 'pupil_position'
    optimizer = tf.keras.optimizers.experimental.AdamW if dataset_freq==500 else tf.keras.optimizers.Adam
    decay_steps = 10000
    decay_rate = 0.95
    conv_layers_dimensions = (64, 64, 128, 128, 256, 256, 512) if dataset_freq==500 else (128, 128, 256, 256, 512, 512, 768)
    dense_layers_dimensions = (64,64)
    loss = "mae"
    dropout = ([0.1]) if dataset_freq==500 else ([0.2])
    output_kernel_size = 1 if dataset_freq==500 else 2
    patience = 20
    validation_set_size = 600
    batch_size = 16 if dataset_freq==500 else 8
    max_epochs_per_sample = 2
    epochs = 500

lr = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    lr,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
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

model = dt.models.Convolutional(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
    conv_layers_dimensions=conv_layers_dimensions,
    dense_layers_dimensions=dense_layers_dimensions,
    steps_per_pooling=2,
    number_of_outputs=2,
    loss=loss,
    metrics=[lambda T,P: training.pixel_error(T, P, IMAGE_SIZE), training.scaled_pixel_error],
    optimizer= opt,
    dropout=dropout,
    flatten_method='global_max',
    dense_block=dt.layers.DenseBlock(activation="relu"),
    pooling_block=dt.layers.PoolingBlock(padding="valid"),
    output_kernel_size=output_kernel_size
)
model.summary()

# set up early-stopping:
EARLY_STOPPING = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=patience,
    mode='auto',
    restore_best_weights=True
)

validation_set    = [image_pipeline.update()() for _ in range(validation_set_size)]
validation_labels = [training.get_scaled_feature_location(image, IMAGE_SIZE, label_name) for image in validation_set]

generator = dt.generators.ContinuousGenerator(
    image_pipeline,
    lambda im: training.get_scaled_feature_location(im, IMAGE_SIZE, label_name),
    min_data_size=int(1e3),
    max_data_size=int(2e4),
    batch_size=batch_size,
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
model.save(f'high_res_{feature}_{dataset_freq}Hz_stage_1.h5')