# import the necessary packages
import tensorflow as tf


def pos_embedding(p, L):
    # build the list of positional encodings
    posVector = [p]
    # iterate over the number of dimensions in time
    for i in range(L):
        # insert sine and cosine of the product of current dimension
        # and the position vector
        posVector.append(tf.sin((2.0 ** i) * p))
        posVector.append(tf.cos((2.0 ** i) * p))

    # concatenate the positional encodings into a positional vector
    posVector = tf.concat(posVector, axis=-1)
    # return the positional encoding vector
    return posVector
