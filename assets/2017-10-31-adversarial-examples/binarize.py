from scipy.ndimage import imread
import numpy as np
import matplotlib.pyplot as plt

im = imread('panda.png')
mask = np.mean(im, axis=2) > 255. / 2

plt.imshow(mask == False, cmap="Greys")
plt.savefig('binary_panda.png')
