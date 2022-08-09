import numpy as np

poses_bounds = np.load("/home/aasman/angelswing/NERF_tensorflow/dataset/room/poses_bounds.npy")
poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N_images, 3, 5)

aasman = 10
