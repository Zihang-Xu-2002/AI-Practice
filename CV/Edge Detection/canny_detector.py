from scipy import ndimage
from scipy.ndimage.filters import convolve

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os
import scipy.misc as sm
class CannyDetector:
    def __init__(self, imgs, sigma=1, kernel_size = 3, weak=50, strong=255, lowthresholdrate=0.07, highthresholdrate=0.5):
        self.imgs = imgs
        self.img_result = []
        self.img_smoothed = None
        self.img_gd_x = None
        self.img_gd_y = None
        self.img_gd = None
        self.gaussian_kernel = None
        self.gaussian_kernel_x = None
        self.gaussian_kernel_y = None
        self.img_NMS = None
        self.img_DT = None
        self.sigma = None
        self.kernel_size = None
        self.weak = weak
        self.strong = strong
        self.lowthresholdrate = lowthresholdrate
        self.highthresholdrate = highthresholdrate
        return
    def grayscale_process(self,src,way=1):
        """_summary_

        Args:
            src (matplotlib.pyplot): _description_
            way (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        row,col,channel = src.shape
        #平均法：把一个像素位置的3个通道的RGB值进行平均
        img_gray = np.zeros((row,col))
        if way == 1:
            for r in range(row):
                for l in range(col):
                    img_gray[r,l]=(1/3)*src[r,l,0]+(1/3)*src[r,l,1]+(1/3)*src[r,l,2]
        #最大最小平均法    
        if way == 2:
            for r in range(row):
                for l in range(col):
                    img_gray[r,l]=(1/2)*max(src[r,l,0],src[r,l,1],src[r,l,2])+(1/2)*max(src[r,l,0],src[r,l,1],src[r,l,2])
        #加权法
        if way == 3:
            for r in range(row):
                for l in range(col):
                    img_gray[r,l]=(0.11)*src[r,l,0]+(0.59)*src[r,l,1]+(0.3)*src[r,l,2]
        dst = img_gray.astype("uint8")
        return dst

    def gaussian(self,size,sigma=1):
        """_summary_

        Args:
            size (_type_): _description_
            sigma (int, optional): _description_. Defaults to 1.
        """
