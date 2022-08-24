# importing the necessary libraries
from utils import config
from utils.data import *
from rendering import render_videos
import tensorflow as tf


model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)

render_videos(nerf_model=model)
