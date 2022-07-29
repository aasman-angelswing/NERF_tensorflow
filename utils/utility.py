import tensorflow as tf
from tensorflow import searchsorted
from utils import config
import numpy as np


def render_image_depth(rgb, sigma, tVals):
    # squeeze the last dimension of sigma
    sigma = sigma[..., 0]
    # calculate the delta between adjacent tVals
    delta = tVals[..., 1:] - tVals[..., :-1]
    deltaShape = [config.BATCH_SIZE,
                  config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1]
    delta = tf.concat(
        [delta, tf.broadcast_to([1e10], shape=deltaShape)], axis=-1)
    # calculate alpha from sigma and delta values
    alpha = 1.0 - tf.exp(-sigma * delta)
    # calculate the exponential term for easier calculations
    expTerm = 1.0 - alpha
    epsilon = 1e-10
    # calculate the transmittance and weights of the ray points
    transmittance = tf.math.cumprod(expTerm + epsilon, axis=-1,
                                    exclusive=True)
    weights = alpha * transmittance
    # build the image and depth map from the points of the rays
    image = tf.reduce_sum(weights[..., None] * rgb, axis=-2)
    depth = tf.reduce_sum(weights * tVals, axis=-1)
    # return rgb, depth map and weights
    return (image, depth, weights)


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):

    print(weights.shape)
    N_rays, N_samples_ = 2*200*200, 16
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    # (N_rays, N_samples_)
    pdf = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
    # (N_rays, N_samples), cumulative distribution function
    cdf = tf.cumsum(pdf, -1)
    # (N_rays, N_samples_+1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)
    # padded to 0~1 inclusive
    bins = tf.concat([tf.zeros_like(bins[..., :2]), bins], -1)

    if det:
        u = tf.linspace(0, 1, N_importance)
        u = u.expand(N_rays, N_importance)
    else:
        uShape = [config.BATCH_SIZE, config.IMAGE_HEIGHT,
                  config.IMAGE_WIDTH, N_importance]
        u = tf.random.uniform(uShape)

    inds = searchsorted(cdf, u, side='right')
    below = tf.minimum(inds-1, 0)
    above = tf.maximum(inds, N_samples_)

    inds_sampled = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_sampled, axis=-1,
                      batch_dims=len(inds_sampled.shape)-2)
    bins_g = tf.gather(bins, inds_sampled, axis=-1,
                       batch_dims=len(inds_sampled.shape)-2)
    denom = cdf_g[..., 1]-cdf_g[..., 0]
    # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u-cdf_g[..., 0]) / \
        denom * (bins_g[..., 1]-bins_g[..., 0])
    return samples


def get_translation_t(t):
    """Get the translation matrix for movement in t."""
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def get_rotation_phi(phi):
    """Get the rotation matrix for movement in phi."""
    matrix = [
        [1, 0, 0, 0],
        [0, tf.cos(phi), -tf.sin(phi), 0],
        [0, tf.sin(phi), tf.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def get_rotation_theta(theta):
    """Get the rotation matrix for movement in theta."""
    matrix = [
        [tf.cos(theta), 0, -tf.sin(theta), 0],
        [0, 1, 0, 0],
        [tf.sin(theta), 0, tf.cos(theta), 0],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def pose_spherical(theta, phi, t):
    """
    Get the camera to world matrix for the corresponding theta, phi
    and t.
    """
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0],
                   [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w
