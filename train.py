# USAGE
# python train.py
# setting seed for reproducibility
#%%
import os
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from nerfutils import config
from nerfutils.nerf_trainer import NeRF
from nerfutils.nerf import get_nerf_model
from nerfutils.encoder import encode_position
from nerfutils.data import get_rays
from nerfutils.data import GetImages
from nerfutils.data import get_image_c2w
from nerfutils.data import read_json
from nerfutils.data import map_fn
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from nerfutils.nerf import render_rgb_depth
import glob
import imageio
from tqdm import tqdm
import numpy as np
tf.random.set_seed(42)
from PIL import Image
import glob
# import the necessary packages

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
trainDs = tf.data.Dataset.zip((trainImageDs,trainRayDs))
valDs = tf.data.Dataset.zip(( valImageDs,valRayDs,))
testDs = tf.data.Dataset.zip((testImageDs,testRayDs,))
# build data input pipeline for train, val, and test datasets
trainDs = (
    trainDs
    .shuffle(config.BATCH_SIZE)
    .batch(config.BATCH_SIZE)
    .prefetch(config.AUTO)
)
valDs = (
    valDs
    .shuffle(config.BATCH_SIZE)
    .batch(config.BATCH_SIZE)
    .prefetch(config.AUTO)
)
testDs = (
    testDs
    .batch(config.BATCH_SIZE)
    .prefetch(config.AUTO)
)

test_imgs, test_rays = next(iter(trainDs))
test_rays_flat, test_t_vals = test_rays

loss_list = []

class TrainMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs["loss"]
        loss_list.append(loss)
        test_recons_images, depth_maps = render_rgb_depth(
            model=self.model.nerf_model,
            rays_flat=test_rays_flat,
            t_vals=test_t_vals,
            rand=True,
            train=False,
        )

        # Plot the rgb, depth and the loss plot.
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        ax[0].imshow(keras.preprocessing.image.array_to_img(test_recons_images[0]))
        ax[0].set_title(f"Predicted Image: {epoch:03d}")

        ax[1].imshow(keras.preprocessing.image.array_to_img(depth_maps[0, ..., None]))
        ax[1].set_title(f"Depth Map: {epoch:03d}")

        ax[2].plot(loss_list)
        ax[2].set_xticks(np.arange(0, config.EPOCHS + 1, 5.0))
        ax[2].set_title(f"Loss Plot: {epoch:03d}")

        fig.savefig(f"images/{epoch:03d}.png")
        plt.show()
        plt.close()
        
        
num_pos = config.IMAGE_HEIGHT * config.IMAGE_WIDTH * config.NUM_SAMPLES
nerf_model = get_nerf_model(num_layers=8, num_pos=num_pos)

model = NeRF(nerf_model)
model.compile(
    optimizer=keras.optimizers.Adam(), loss_fn=keras.losses.MeanSquaredError()
)

# Create a directory to save the images during training.
if not os.path.exists("images"):
    os.makedirs("images")

model.fit(
    trainDs,
    validation_data=valDs,
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS,
    callbacks=[TrainMonitor()],
    steps_per_epoch=config.STEPS_PER_EPOCH,
)


def create_gif(path_to_images, name_gif):
    filenames = glob.glob(path_to_images)
    filenames = sorted(filenames)
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
    kargs = {"duration": 0.25}
    imageio.mimsave(name_gif, images, "GIF", **kargs)


create_gif("images/*.png", "training.gif")


