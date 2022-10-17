from cv2 import waitKey
import numpy as np
import skimage
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os
import scipy.misc as sm
import cv2

def rgb2gray(rgb):
    """After the experiment in 'EdgeDetection.ipynb', 
    I choose weighted average grayscale process

    Args:
        rgb (array): colorful image

    Returns:
        array: gray image
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.3 * r + 0.59 * g + 0.11 * b

    return gray

def load_data(dir_name = './img'):    
    """load img

    Args:
        dir_name (str, optional): _description_. Defaults to 'img'.

    Returns:
        list : a list of imgs
    """
    imgs = []
    for filename in os.listdir(dir_name):
        if os.path.isfile(dir_name + '/' + filename):
            img = mpimg.imread(dir_name + '/' + filename)
            img = rgb2gray(img)
            imgs.append(img)
    return imgs


def visualize(imgs):
    for i, img in enumerate(imgs):
        cv2.imshow("1",imgs[i].astype("uint8"))
        #cv2.imwrite(str("img"+si),imgs[i])
        cv2.waitKey(0)

def visualize_save(imgs):
    for i, img in enumerate(imgs):
        cv2.imshow("1",imgs[i].astype("uint8"))
        cv2.imwrite("img"+str(i),imgs[i])
        cv2.waitKey(0)

    