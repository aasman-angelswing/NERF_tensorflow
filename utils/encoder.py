# import the necessary packages
import tensorflow as tf
from utils import config


def encode_position(x):
    """Encodes the position into its corresponding Fourier feature.

    Args:
        x: The input coordinate.

    Returns:
        Fourier features tensors of the position.
    """
    with tf.device(config.DEVICE):

        positions = [x]
        for i in range(config.POS_ENCODE_DIMS):
            positions.append(tf.sin(2.0**i * x))
            positions.append(tf.cos(2.0**i * x))

    return tf.concat(positions, axis=-1)
