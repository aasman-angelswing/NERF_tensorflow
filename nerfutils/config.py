import tensorflow as tf

SAMPLE_THETA_POINTS = 12

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800
AUTO = tf.data.AUTOTUNE

L_XYZ = 4
L_DIR = 4
N_F = 32
N_C = 32

main_dir = "/content/NERF_tensorflow"

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
SKIP_LAYER = 5
DENSE_UNITS = 256

BATCH_SIZE = 5
NUM_SAMPLES = 32
POS_ENCODE_DIMS = 16

STEPS_PER_EPOCH = 80
VALIDATION_STEPS = 2
EPOCHS = 10

COARSE_PATH = main_dir + "/output"
FINE_PATH = main_dir + "/output"
