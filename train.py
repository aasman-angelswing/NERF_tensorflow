# importing the necessary libraries
#%%
from utils import config
from utils.nerf_trainer import NeRF
from utils.nerf import get_nerf_model, render_rgb_depth
from utils.data import *
from utils.config import create_dir
from utils.train_monitor import get_train_monitor
from visual import inference, create_gif
from rendering import render_videos

from tensorflow import keras

import tensorflow as tf
import numpy as np
#%%
# setting seed for reproducibility
tf.random.set_seed(42)

create_dir()

# get the train validation and test data
print("[INFO] grabbing the data from json files...")
jsonTrainData = read_json(config.TRAIN_JSON)
jsonValData = read_json(config.VAL_JSON)
jsonTestData = read_json(config.TEST_JSON)
focalLength = 22
# print the focal length of the camera
print(f"[INFO] focal length of the camera: {focalLength}...")

# get the train, validation, and test image paths and camera2world
# matrices
print("[INFO] grabbing the image paths and camera2world matrices...")
trainImagePaths, trainC2Ws = get_image_c2w(jsonData=jsonTrainData,
                                           datasetPath=config.DATASET_PATH)
train_images = GetImages(trainImagePaths)

valImagePaths, valC2Ws = get_image_c2w(jsonData=jsonValData,
                                       datasetPath=config.DATASET_PATH)
val_images = GetImages(valImagePaths)
testImagePaths, testC2Ws = get_image_c2w(jsonData=jsonTestData,
                                         datasetPath=config.DATASET_PATH)
test_images = GetImages(testImagePaths)
# instantiate a object of our class used to load images from disk
valC2Ws = np.array(valC2Ws)
valC2Ws = tf.cast(valC2Ws, tf.float32)

# get the train, validation, and test image dataset
print("[INFO] building the image dataset pipeline...")
trainImageDs = tf.data.Dataset.from_tensor_slices(train_images)
valImageDs = tf.data.Dataset.from_tensor_slices(val_images)
testImageDs = tf.data.Dataset.from_tensor_slices(test_images)
train_pose_ds = tf.data.Dataset.from_tensor_slices(trainC2Ws)
val_pose_ds = tf.data.Dataset.from_tensor_slices(valC2Ws)
test_pose_ds = tf.data.Dataset.from_tensor_slices(testC2Ws)


# get the train validation and test rays dataset
print("[INFO] building the rays dataset pipeline...")
trainRayDs = train_pose_ds.map(map_fn, num_parallel_calls=config.AUTO)
valRayDs = val_pose_ds.map(map_fn, num_parallel_calls=config.AUTO)
testRayDs = test_pose_ds.map(map_fn, num_parallel_calls=config.AUTO)


# zip the images and rays dataset together
trainDs = tf.data.Dataset.zip((trainImageDs, trainRayDs))
valDs = tf.data.Dataset.zip((valImageDs, valRayDs,))
testDs = tf.data.Dataset.zip((testImageDs, testRayDs,))
# build data input pipeline for train, val, and test datasets
trainDs = (
    trainDs
    .shuffle(config.BATCH_SIZE,)
    .batch(config.BATCH_SIZE, drop_remainder=True, num_parallel_calls=config.AUTO).
    repeat(2)
    .prefetch(config.AUTO)
)
valDs = (
    valDs
    .shuffle(config.BATCH_SIZE)
    .batch(config.BATCH_SIZE, drop_remainder=True, num_parallel_calls=config.AUTO)
    .repeat(2)
    .prefetch(config.AUTO)
)
testDs = (
    testDs
    .batch(config.BATCH_SIZE)
    .prefetch(config.AUTO)
)

'''

trainMonitorCallback = get_train_monitor(
    testDs, render_rgb_depth=render_rgb_depth, OUTPUT_IMAGE_PATH=config.OUTPUT_IMAGE_PATH)

num_pos = config.IMAGE_HEIGHT * config.IMAGE_WIDTH * config.NUM_SAMPLES
nerf_model = get_nerf_model(num_layers=8, num_pos=num_pos)

model = NeRF(nerf_model)
model.compile(
    optimizer=tf.keras.optimizers.Adam(), loss_fn=tf.keras.losses.MeanSquaredError()
)
'''

'''
model.fit(
    trainDs,
    validation_data=valDs,
    # batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS,
    callbacks=[trainMonitorCallback],
    # steps_per_epoch=config.STEPS_PER_EPOCH,
)


model.nerf_model.save('C:/Users/lihsu/OneDrive/Desktop/NERF_tensorflow/output',save_format='tf')
'''
#%%
model = tf.keras.models.load_model(config.MODEL_PATH ,compile = False)

#%%
inference(nerf_model=model, render_rgb_depth=render_rgb_depth,
          testDs=testDs, OUTPUT_INFERENCE_PATH=config.OUTPUT_INFERENCE_PATH)

# %%
render_videos(nerf_model=model)
# %%
