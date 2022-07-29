# import the necessary packages
import tensorflow as tf
from tensorflow.io import read_file
from tensorflow.image import decode_jpeg
from tensorflow.image import convert_image_dtype
from tensorflow.image import resize
from tensorflow import reshape
import json
from nerfutils.encoder import encode_position
from nerfutils import config


def read_json(jsonPath):
    # open the json file
    with open(jsonPath, "r") as fp:
        # read the json data
        data = json.load(fp)

    # return the data
    return data


def get_image_c2w(jsonData, datasetPath):
    # define a list to store the image paths
    imagePaths = []

    # define a list to store the camera2world matrices
    c2ws = []
    # iterate over each frame of the data
    for frame in jsonData["frames"]:
        # grab the image file name
        imagePath = frame["file_path"]
        imagePath = imagePath.replace(".", datasetPath)
        imagePaths.append(f"{imagePath}.png")
        # grab the camera2world matrix
        c2ws.append(frame["transform_matrix"])

    # return the image file names and the camera2world matrices
    return (imagePaths, c2ws)




def GetImages(imagePath):
    images = []
    for imagePath in imagePath
        image = read_file(imagePath)
            # decode the image string
        image = decode_jpeg(image, 3)
            # convert the image dtype from uint8 to float32
        image = convert_image_dtype(image, dtype=tf.float32)
            # resize the image to the height and width in config
        image = resize(image, (200, 200))
        image = reshape(image, (200, 200, 3))
        images.append(image)
            # return the image
    return images


def get_rays(height, width, focal, pose):
  
    # Build a meshgrid for the rays.
    i, j = tf.meshgrid(
        tf.range(width, dtype=tf.float32),
        tf.range(height, dtype=tf.float32),
        indexing="xy",
    )

    # Normalize the x axis coordinates.
    transformed_i = (i - width * 0.5) / focal

    # Normalize the y axis coordinates.
    transformed_j = (j - height * 0.5) / focal

    # Create the direction unit vectors.
    directions = tf.stack([transformed_i, -transformed_j, -tf.ones_like(i)], axis=-1)

    # Get the camera matrix.
    camera_matrix = pose[:3, :3]
    print(camera_matrix.shape)
    height_width_focal = pose[:3, -1]

    # Get origins and directions for the rays.
    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = tf.reduce_sum(camera_dirs, axis=-1)
    ray_origins = tf.broadcast_to(height_width_focal, tf.shape(ray_directions))

    # Return the origins and directions.
    return (ray_origins, ray_directions)

def render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=False):
  
    # Compute 3D query points.
    # Equation: r(t) = o+td -> Building the "t" here.
    t_vals = tf.linspace(near, far, num_samples)
    if rand:
        # Inject uniform noise into sample space to make the sampling
        # continuous.
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = tf.random.uniform(shape=shape) * (far - near) / num_samples
        t_vals = t_vals + noise

    # Equation: r(t) = o + td -> Building the "r" here.
    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., None]
    )
    rays_flat = tf.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat)
    return (rays_flat, t_vals)

def map_fn(pose):
    
    (ray_origins, ray_directions) = get_rays(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH, focal=22, pose=pose)
    (rays_flat, t_vals) = render_flat_rays(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        near=2.0,
        far=6.0,
        num_samples=config.NUM_SAMPLES,
        rand=True,
    )
    return (rays_flat, t_vals)