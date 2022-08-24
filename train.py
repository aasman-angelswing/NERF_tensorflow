
import tensorflow as tf
import numpy as np
import time

# setting seed for reproducibility
tf.random.set_seed(42)


# importing the necessary libraries
from utils import config
from utils.nerf_trainer import NeRF
from utils.nerf import get_nerf_model
from utils.data import *


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



json_train_data = read_json(config.TRAIN_JSON)


train_image_paths, train_camera_to_world = get_image_c2w(jsonData=json_train_data,
                                                         datasetPath=config.DATASET_PATH)
train_images = GetImages(train_image_paths)



train_image_datasets = tf.data.Dataset.from_tensor_slices(train_images)

train_pose_dataset = tf.data.Dataset.from_tensor_slices(train_camera_to_world)



train_rays_dataset = train_pose_dataset.map(map_fn, num_parallel_calls=config.AUTO)

train_dataset = tf.data.Dataset.zip((train_image_datasets, train_rays_dataset))

train_dataset = (
    train_dataset
    .shuffle(config.BATCH_SIZE,)
    .batch(config.BATCH_SIZE, drop_remainder=True, num_parallel_calls=config.AUTO).
    repeat(2)
    .prefetch(config.AUTO)
)



num_pos = config.IMAGE_HEIGHT * config.IMAGE_WIDTH * config.NUM_SAMPLES

nerf_model = get_nerf_model(num_layers=8, num_pos=num_pos)

model = NeRF(nerf_model)

model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss_fn=tf.keras.losses.MeanSquaredError())
    
    


start1 = time.time()
loop = 5
for i in range(loop):
    start = time.time()

    model.fit(train_dataset,epochs=config.EPOCHS,)
    end = time.time()
    print(f"Total time for {i} epoch is {(end - start)/60} minutes")
    model.nerf_model.save(config.MODEL_PATH ,save_format='tf')
end1 = time.time()
print(f"Total time for {loop} epoch is {(end1 - start1)/60} minutes")

