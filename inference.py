# import the necessary packages
import tensorflow as tf
import numpy as np
import imageio
from tqdm import tqdm
import os

from utils import config
from utils.utility import pose_spherical
from utils.data import GetRays
from utils.data import read_json
from utils.encoder import pos_embedding
from utils.utility import render_image_depth
from utils.utility import sample_pdf


# create a camera2world matrix list to store the novel view
# camera2world matrices
c2wList = []
# iterate over theta and generate novel view camera2world matrices
for theta in np.linspace(0.0, 360.0, config.SAMPLE_THETA_POINTS,
                         endpoint=False):
    # generate camera2world matrix
    c2w = pose_spherical(theta, -30.0, 4.0)

    # append the new camera2world matrix into the collection
    c2wList.append(c2w)
# get the train validation and test data
print("[INFO] grabbing the data from json files...")
jsonTrainData = read_json(config.TRAIN_JSON)
focalLength = 22
# instantiate the GetRays object
getRays = GetRays(focalLength=focalLength, imageWidth=config.IMAGE_WIDTH,
                  imageHeight=config.IMAGE_HEIGHT, near=config.NEAR, far=config.FAR,
                  nC=config.N_C)
# create a dataset from the novel view camera2world matrices
ds = (
    tf.data.Dataset.from_tensor_slices(c2wList)
    .map(getRays)
    .batch(config.BATCH_SIZE)
)
# load the coarse and the fine model
coarseModel = tf.keras.models.tf.keras.models.load_model(config.COARSE_PATH, compile=False)
fineModel = tf.keras.models.tf.keras.models.load_model(config.FINE_PATH, compile=False)

# create a list to hold all the novel view from the nerf model
print("[INFO] grabbing the novel views...")
frameList = []
for element in tqdm(ds):
    (raysOriCoarse, raysDirCoarse, tValsCoarse) = element
    # generate the coarse rays
    raysCoarse = (raysOriCoarse[..., None, :] +
                  (raysDirCoarse[..., None, :] * tValsCoarse[..., None]))
    # positional encode the rays and dirs
    raysCoarse = pos_embedding(raysCoarse, config.L_XYZ)
    dirCoarseShape = tf.shape(raysCoarse[..., :3])
    dirsCoarse = tf.broadcast_to(raysDirCoarse[..., None, :],
                                 shape=dirCoarseShape)
    dirsCoarse = pos_embedding(dirsCoarse, config.L_DIR)
    # compute the predictions from the coarse model
    (rgbCoarse, sigmaCoarse) = coarseModel.predict(
        [raysCoarse, dirsCoarse])

    # render the image from the predictions
    renderCoarse = render_image_depth(rgb=rgbCoarse,
                                      sigma=sigmaCoarse, tVals=tValsCoarse)
    (_, _, weightsCoarse) = renderCoarse
    # compute the middle values of t vals
    tValsCoarseMid = (0.5 *
                      (tValsCoarse[..., 1:] + tValsCoarse[..., :-1]))
    # apply hierarchical sampling and get the t vals for the fine
    # model 
    tValsFine = sample_pdf(bins=tValsCoarseMid,
                           weights=weightsCoarse, N_importance=config.N_F)
    tValsFine = tf.sort(
        tf.concat([tValsCoarse, tValsFine], axis=-1), axis=-1)
    # build the fine rays and positional encode it
    raysFine = (raysOriCoarse[..., None, :] +
                (raysDirCoarse[..., None, :] * tValsFine[..., None]))
    raysFine = pos_embedding(raysFine, config.L_XYZ)

    # build the fine directions and positional encode it
    dirsFineShape = tf.shape(raysFine[..., :3])
    dirsFine = tf.broadcast_to(raysDirCoarse[..., None, :],
                               shape=dirsFineShape)
    dirsFine = pos_embedding(dirsFine, config.L_DIR)
    # compute the predictions from the fine model
    (rgbFine, sigmaFine) = fineModel.predict([raysFine, dirsFine])

    # render the image from the predictions
    renderFine = render_image_depth(rgb=rgbFine, sigma=sigmaFine,
                                    tVals=tValsFine)
    (imageFine, _, _) = renderFine
    # insert the rendered fine image to the collection
    frameList.append(imageFine.numpy()[0])

 # check if the output video directory exists, if it does not, then
# create it
if not os.path.exists(config.VIDEO_PATH):
    os.makedirs(config.VIDEO_PATH)
# build the video from the frames and save it to disk
print("[INFO] creating the video from the frames...")
imageio.mimwrite(config.OUTPUT_VIDEO_PATH+'.gif', frameList, fps=config.FPS)
