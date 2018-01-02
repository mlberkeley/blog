import os
from scipy.misc import imresize, imsave
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import numpy as np

def zip_im():
    for i in range(10):
        num = imread('targeted/' + str(i) + '.png')
        scores = imread('targeted/' + str(i) + '_scores.png')
        num = imresize(num, size=(300, 300), interp='nearest')
        scores = scores[:,:,:-1]

        cat = np.concatenate((num,scores), 1)       

        imsave('targeted/combined_' + str(i) + '.png', cat)

zip_im()
