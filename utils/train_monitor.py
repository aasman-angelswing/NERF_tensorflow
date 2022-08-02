import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from utils import config



def get_train_monitor(trainDs, render_rgb_depth, OUTPUT_IMAGE_PATH):
    
    _, test_rays = next(iter(trainDs))
    test_rays_flat, test_t_vals = test_rays

    loss_list = []
    
    class TrainMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss = logs["loss"]
            loss_list.append(loss)
            test_recons_images, depth_maps = render_rgb_depth(
                model=self.model.nerf_model,
                rays_flat=test_rays_flat,
                t_vals=test_t_vals,
                rand=True,
                train=False,
            )

            # Plot the rgb, depth and the loss plot.
            fig, ax = plt.subplot(nrows=1, ncols=3, figsize=(20, 5))
            ax[0].imshow(tf.keras.preprocessing.image.array_to_img(test_recons_images[0]))
            ax[0].set_title(f"Predicted Image: {epoch:03d}")

            ax[1].imshow(tf.keras.preprocessing.image.array_to_img(depth_maps[0, ..., None]))
            ax[1].set_title(f"Depth Map: {epoch:03d}")

            ax[2].plot(loss_list)
            ax[2].set_xticks(np.arange(0, config.EPOCHS + 1, 5.0))
            ax[2].set_title(f"Loss Plot: {epoch:03d}")

            fig.savefig(f"{OUTPUT_IMAGE_PATH}/{epoch:03d}.png")
         

    trainMonitor = TrainMonitor()
    # return the train monitor
    return trainMonitor

