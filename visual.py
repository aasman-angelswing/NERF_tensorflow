import matplotlib.pyplot as plt
import tensorflow as tf
import imageio
import glob
from tqdm import tqdm


def create_gif(path_to_images, name_gif):
    filenames = glob.glob(path_to_images)
    filenames = sorted(filenames)
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
    kargs = {"duration": 0.25}
    imageio.mimsave(name_gif, images, "GIF", **kargs)


def inference(nerf_model, render_rgb_depth, testDs, OUTPUT_INFERENCE_PATH):

    # Get the trained NeRF model and infer.
    test_imgs, test_rays = next(iter(testDs))
    test_rays_flat, test_t_vals = test_rays
    test_recons_images, depth_maps = render_rgb_depth(
        model=nerf_model,
        rays_flat=test_rays_flat,
        t_vals=test_t_vals,
        rand=True,
        train=False,
    )

    # Create subplots.
    fig, axes = plt.subplot(nrows=5, ncols=3, figsize=(10, 20))
    namecount = 0
    for ax, ori_img, recons_img, depth_map in zip(
        axes, test_imgs, test_recons_images, depth_maps
    ):
        ax[0].imshow(tf.keras.preprocessing.image.array_to_img(ori_img))
        ax[0].set_title("Original")

        ax[1].imshow(tf.keras.preprocessing.image.array_to_img(recons_img))
        ax[1].set_title("Reconstructed")

        ax[2].imshow(
            tf.keras.preprocessing.image.array_to_img(depth_map[..., None]), cmap="inferno"
        )
        ax[2].set_title("Depth Map")
        fig.savefig(f"{OUTPUT_INFERENCE_PATH}/{namecount}.png")
        namecount += namecount
