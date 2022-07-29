# import the necessary packages
import tensorflow as tf


def get_model(lxyz, lDir, batchSize, denseUnits, skipLayer):
    # build input layer for rays
    rayInput = tf.keras.Input(shape=(None, None, None, 2 * 3 * lxyz + 3),
                     batch_size=batchSize)

    # build input layer for direction of the rays
    dirInput = tf.keras.Input(shape=(None, None, None, 2 * 3 * lDir + 3),
                     batch_size=batchSize)

    # creating an input for the MLP
    x = rayInput
    for i in range(8):
        # build a dense layer
        x = tf.keras.layers.Dense(units=denseUnits, activation="relu")(x)
        # check if we have to include residual connection
        if i % skipLayer == 0 and i > 0:
            # inject the residual connection
            x = tf.keras.layers.concatenate([x, rayInput], axis=-1)

    # get the sigma value
    sigma = tf.keras.layers.Dense(units=1, activation="relu")(x)
    # create the feature vector
    feature = tf.keras.layers.Dense(units=denseUnits)(x)
    # concatenate the feature vector with the direction input and put
    # it through a dense layer
    feature = tf.keras.layers.concatenate([feature, dirInput], axis=-1)
    x = tf.keras.layers.Dense(units=denseUnits//2, activation="relu")(feature)
    # get the rgb value
    rgb = tf.keras.layers.Dense(units=3, activation="sigmoid")(x)
    # create the nerf model
    nerfModel = tf.keras.Model(inputs=[rayInput, dirInput],
                      outputs=[rgb, sigma])

    # return the nerf model
    return nerfModel
