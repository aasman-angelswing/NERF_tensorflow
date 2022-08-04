# import the necessary packages
import tensorflow as tf
<<<<<<< HEAD


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
=======
from utils import config


def get_nerf_model(num_layers, num_pos):
    inputs = tf.keras.Input(
        shape=(num_pos, 2 * 3 * config.POS_ENCODE_DIMS + 3))
    x = inputs
    for i in range(num_layers):
        x = tf.keras.layers.Dense(units=64, activation="relu")(x)
        if i % 4 == 0 and i > 0:
            # Inject residual connection.
            x = tf.keras.layers.concatenate([x, inputs], axis=-1)
    outputs = tf.keras.layers.Dense(units=4)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def render_rgb_depth(model, rays_flat, t_vals, rand=True, train=True):

    # Get the predictions from the nerf model and reshape it.
    if train:
        predictions = model(rays_flat)
    else:
        predictions = model.predict(rays_flat)
    predictions = tf.reshape(predictions, shape=(
        config.BATCH_SIZE, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.NUM_SAMPLES, 4))

    # Slice the predictions into rgb and sigma.
    rgb = tf.sigmoid(predictions[..., :-1])
    sigma_a = tf.nn.relu(predictions[..., -1])

    # Get the distance of adjacent intervals.
    delta = t_vals[..., 1:] - t_vals[..., :-1]
    # delta shape = (num_samples)
    if rand:
        delta = tf.concat(
            [delta, tf.broadcast_to([1e10], shape=(config.BATCH_SIZE, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1))], axis=-1
        )
        alpha = 1.0 - tf.exp(-sigma_a * delta)
    else:
        delta = tf.concat(
            [delta, tf.broadcast_to([1e10], shape=(config.BATCH_SIZE, 1))], axis=-1
        )
        alpha = 1.0 - tf.exp(-sigma_a * delta[:, None, None, :])

    # Get transmittance.
    exp_term = 1.0 - alpha
    epsilon = 1e-10
    transmittance = tf.math.cumprod(
        exp_term + epsilon, axis=-1, exclusive=True)
    weights = alpha * transmittance
    rgb = tf.reduce_sum(weights[..., None] * rgb, axis=-2)

    if rand:
        depth_map = tf.reduce_sum(weights * t_vals, axis=-1)
    else:
        depth_map = tf.reduce_sum(weights * t_vals[:, None, None], axis=-1)
    return (rgb, depth_map)
>>>>>>> production
