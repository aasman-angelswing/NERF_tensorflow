import tensorflow as tf
import os

SAMPLE_THETA_POINTS = 10
FOCAL_LENGTH = 50

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
AUTO = tf.data.AUTOTUNE

L_XYZ = 4
L_DIR = 4

main_dir = "/home/ec2-user/SageMaker/NERF_tensorflow"

OUTPUT_IMAGE_PATH = main_dir + "/output/images"
OUTPUT_INFERENCE_PATH = main_dir + "/output/inferences"
OUTPUT_VIDEO_PATH = "/output/videos"
MODEL_PATH = main_dir + "/output"

FPS = 30
QUALITY = 7

TRAIN_JSON = main_dir + "/dataset/transforms_train.json"
VAL_JSON = main_dir + "/dataset/transforms_val.json"
TEST_JSON = main_dir + "/dataset/transforms_test.json"
DATASET_PATH = main_dir + "/dataset"

NEAR = 2.0
FAR = 6.0

BATCH_SIZE = 2
NUM_SAMPLES = 16
POS_ENCODE_DIMS = 16
DENSE_UNITS = 64

# STEPS_PER_EPOCH = 20
# VALIDATION_STEPS = 20
EPOCHS = 10


def create_dir():
    if not os.path.exists(OUTPUT_INFERENCE_PATH):
        os.makedirs(OUTPUT_INFERENCE_PATH)
    if not os.path.exists(OUTPUT_IMAGE_PATH):
        os.makedirs(OUTPUT_IMAGE_PATH)
    if not os.path.exists(OUTPUT_VIDEO_PATH):
        os.makedirs(OUTPUT_VIDEO_PATH)
