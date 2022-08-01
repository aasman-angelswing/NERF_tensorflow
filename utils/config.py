import tensorflow as tf


SAMPLE_THETA_POINTS = 10

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
AUTO = tf.data.AUTOTUNE

L_XYZ = 16
L_DIR = 16
N_F = 8
N_C = 8

main_dir = "/home/aasman/angelswing/NERF_tensorflow"

IMAGE_PATH = main_dir + "/output/image"
VIDEO_PATH = main_dir + "/output/image"
OUTPUT_VIDEO_PATH = main_dir + "/output/image"
FPS = 30
QUALITY = 7

TRAIN_JSON = main_dir + "/dataset/transforms_train.json"
VAL_JSON = main_dir + "/dataset/transforms_val.json"
TEST_JSON = main_dir + "/dataset/transforms_test.json"
DATASET_PATH = main_dir + "/dataset"

NEAR = 2
FAR = 6

SKIP_LAYER = 4
DENSE_UNITS = 64
TRAIN_EXAMPLE = 100
TRAIN_EXAMPLE = 100
VALID_EXAMPLE = 100


BATCH_SIZE = 2

STEPS_PER_EPOCH =  TRAIN_EXAMPLE // BATCH_SIZE
VALIDATION_STEPS = VALID_EXAMPLE // BATCH_SIZE
EPOCHS = 10

COARSE_PATH = main_dir + "/output"
FINE_PATH = main_dir + "/output"
