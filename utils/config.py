import tensorflow as tf

SAMPLE_THETA_POINTS = 10

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
AUTO = tf.data.AUTOTUNE

L_XYZ = 4
L_DIR = 4
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

BATCH_SIZE = 2
NUM_SAMPLES = 32
POS_ENCODE_DIMS = 16

STEPS_PER_EPOCH =  50
VALIDATION_STEPS = 2
EPOCHS = 10

COARSE_PATH = main_dir + "/output"
FINE_PATH = main_dir + "/output"
