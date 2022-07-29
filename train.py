# import the necessary packages

import os
import tensorflow as tf

from utils import config
from utils.train_monitor import get_train_monitor
from utils.nerf_trainer import Nerf_Trainer
from utils.nerf import get_model
from utils.encoder import pos_embedding
from utils.utility import render_image_depth, sample_pdf
from utils.data import GetRays
from utils.data import GetImages
from utils.data import get_image_c2w
from utils.data import read_json

tf.random.set_seed(42)

# get the train validation and test data
print("[INFO] grabbing the data from json files...")
jsonTrainData = read_json(config.TRAIN_JSON)
jsonValData = read_json(config.VAL_JSON)
jsonTestData = read_json(config.TEST_JSON)

#! todo : focal length using imagewidth and fov
focalLength = 22

# get the train, validation, and test image paths and camera2world
# matrices
trainImgPaths, trainC2W = get_image_c2w(jsonData=jsonTrainData,
                                           datasetPath=config.DATASET_PATH)
valImgPaths, valC2W = get_image_c2w(jsonData=jsonValData,
                                       datasetPath=config.DATASET_PATH)
testImgPaths, testC2W = get_image_c2w(jsonData=jsonTestData,
                                         datasetPath=config.DATASET_PATH)

# instantiate a object of our class used to load images from disk
getImages = GetImages(imageHeight=config.IMAGE_HEIGHT,
                      imageWidth=config.IMAGE_WIDTH)
# get the train, validation, and test image dataset

trainImageDs = (
    tf.data.Dataset.from_tensor_slices(trainImgPaths)
    .map(getImages, num_parallel_calls=config.AUTO)
)

valImageDs = (
    tf.data.Dataset.from_tensor_slices(valImgPaths)
    .map(getImages, num_parallel_calls=config.AUTO)
)

testImageDs = (
    tf.data.Dataset.from_tensor_slices(testImgPaths)
    .map(getImages, num_parallel_calls=config.AUTO)
)

# instantiate the GetRays object
getRays = GetRays(focalLength=focalLength, imageWidth=config.IMAGE_WIDTH,
                  imageHeight=config.IMAGE_HEIGHT, near=config.NEAR, far=config.FAR,
                  nC=config.N_C)
# get the train validation and test rays dataset
trainRayDs = (
    tf.data.Dataset.from_tensor_slices(trainC2W)
    .map(getRays, num_parallel_calls=config.AUTO)
)
valRayDs = (
    tf.data.Dataset.from_tensor_slices(valC2W)
    .map(getRays, num_parallel_calls=config.AUTO)
)
testRayDs = (
    tf.data.Dataset.from_tensor_slices(testC2W)
    .map(getRays, num_parallel_calls=config.AUTO)
)


# zip the images and rays dataset together
trainDs = tf.data.Dataset.zip((trainRayDs, trainImageDs))
valDs = tf.data.Dataset.zip((valRayDs, valImageDs))
testDs = tf.data.Dataset.zip((testRayDs, testImageDs))
# build data input pipeline for train, val, and test datasets
trainDs = (
    trainDs
    .shuffle(config.BATCH_SIZE)
    .batch(config.BATCH_SIZE)
    .repeat()
    .prefetch(config.AUTO)
)
valDs = (
    valDs
    .shuffle(config.BATCH_SIZE)
    .batch(config.BATCH_SIZE)
    .repeat()
    .prefetch(config.AUTO)
)
testDs = (
    testDs
    .batch(config.BATCH_SIZE)
    .prefetch(config.AUTO)
)

# instantiate the coarse model
coarseModel = get_model(lxyz=config.L_XYZ, lDir=config.L_DIR,
                        batchSize=config.BATCH_SIZE, denseUnits=config.DENSE_UNITS,
                        skipLayer=config.SKIP_LAYER)
# instantiate the fine model
fineModel = get_model(lxyz=config.L_XYZ, lDir=config.L_DIR,
                      batchSize=config.BATCH_SIZE, denseUnits=config.DENSE_UNITS,
                      skipLayer=config.SKIP_LAYER)
# instantiate the nerf trainer model
nerfTrainerModel = Nerf_Trainer(coarseModel=coarseModel, fineModel=fineModel,
                                lxyz=config.L_XYZ, lDir=config.L_DIR, encoderFn=pos_embedding,
                                renderImageDepth=render_image_depth, samplePdf=sample_pdf,
                                nF=config.N_F)
# compile the nerf trainer model with Adam optimizer and MSE loss
nerfTrainerModel.compile(optimizerCoarse=tf.keras.optimizers.Adam(), optimizerFine=tf.keras.optimizers.Adam(),
                         lossFn=tf.keras.losses.MeanSquaredError())

# check if the output image directory already exists, if it doesn't,
# then create it
if not os.path.exists(config.IMAGE_PATH):
    os.makedirs(config.IMAGE_PATH)
# get the train monitor callback
trainMonitorCallback = get_train_monitor(testDs=testDs,
                                         encoderFn=pos_embedding, lxyz=config.L_XYZ, lDir=config.L_DIR,
                                         imagePath=config.IMAGE_PATH)
# train the NeRF model
nerfTrainerModel.fit(trainDs, steps_per_epoch=config.STEPS_PER_EPOCH,
                     validation_data=valDs, validation_steps=config.VALIDATION_STEPS,
                     epochs=config.EPOCHS, callbacks=[trainMonitorCallback]
                     )
# save the coarse and fine model
nerfTrainerModel.coarseModel.save(config.COARSE_PATH)
nerfTrainerModel.fineModel.save(config.FINE_PATH)
