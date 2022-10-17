from scipy import ndimage
from scipy.ndimage.filters import convolve

from scipy import misc
import numpy as np
import utils

class CannyDetector:
    def __init__(self, imgs, sigma=1, kernel_size=5, weak=75, strong=255, lowthresholdrate=0.05, highthresholdrate=0.15):
        self.imgs = imgs
        self.imgs_final = []
        self.img_smoothed = None
        self.img_gradient = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak = weak
        self.strong = strong
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthresholdrate
        self.highThreshold = highthresholdrate
        return 
    
    def gaussian_kernel(self, size, sigma=1):
        """_summary_

        Args:
            size (int): kernel size
            sigma (int, optional): the value of sigma. Defaults to 1.

        Returns:
            ndarray: gaussian kernel and the first derivative kernel in two direction
        """
        half_size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        gk =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

        gaussian_first_deriv_x = np.zeros_like(gk)
        gaussian_first_deriv_y = np.zeros_like(gk)
        half_kernel_size = int(size / 2)
        #print("half:",half_kernel_size)
        for i in range(size):
            #print(i)
            x = - half_size + i
            y = - half_size + i
            #print(x)
            factor_x = - x/ (sigma**2)
            #print("factor_x",factor_x)
            factor_y = - y/ (sigma**2)
            #print("factor_x",x)
            #print("gk[",i,"]",gk[i])
            gaussian_first_deriv_x[i] = gk[i] * factor_x
        gaussian_first_deriv_y[i] = gk[i] * factor_y
        gaussian_first_deriv_y=gaussian_first_deriv_y.T
        return gk,gaussian_first_deriv_x,gaussian_first_deriv_y
    
    def gk_derivative(self,img,gd_x,gd_y):
        """calculate the gradient amplitude and direction

        Args:
            img (ndarray): gray image
            gd_x (ndarray): derivative kernel
            gd_y (ndarray): derivative kernel

        Returns:
            ndarray: gradient amplitude and direction
        """
        img_gd_x = ndimage.filters.convolve(img, gd_x)
        img_gd_y = ndimage.filters.convolve(img, gd_y)

        G = np.hypot(img_gd_x, img_gd_y)
        G = G / G.max() * 255
        theta = np.arctan2(img_gd_y, img_gd_x)
        return (G, theta)

    
    def sobel_filters(self, img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        img_gd_x = ndimage.filters.convolve(img, Kx)
        img_gd_y = ndimage.filters.convolve(img, Ky)

        G = np.hypot(img_gd_x, img_gd_y)
        G = G / G.max() * 255
        theta = np.arctan2(img_gd_y, img_gd_x)
        return (G, theta)
    

    def non_max_suppression(self, img, arc_tan):
        """make the edge thinner

        Args:
            img (ndarray): gradient image
            arc_tan (ndarray): the radian system angle

        Returns:
            ndarray: NMS image
        """
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = arc_tan * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0


                except IndexError as e:
                    pass

        return Z

    def threshold(self, img):
        """get the strong edge and weak edge

        Args:
            img (ndarray): NMS image

        Returns:
            ndarray: get 
        """

        highThreshold = img.max() * self.highThreshold
        lowThreshold = highThreshold * self.lowThreshold

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak)
        strong = np.int32(self.strong)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        res[zeros_i,zeros_j]=0

        return (res)

    def hysteresis(self, img):
        """connect edges

        Args:
            img (ndarray): image processed by double threshold

        Returns:
            ndarray: the result
        """

        M, N = img.shape
        weak = self.weak
        strong = self.strong

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img
    
    def detect(self):
        """the procedure of canny detector

        Returns:
            ndarray: final image
        """
        
        gk,gk_x,gk_y = self.gaussian_kernel(self.kernel_size, self.sigma)
        for i, img in enumerate(self.imgs):    
            self.img_smoothed = convolve(img,gk)
            #self.img_gradient, self.thetaMat = self.gk_derivative(self.img_smoothed,gk_x,gk_y)
            #sobel works better than gaussian
            self.img_gradient, self.thetaMat = self.sobel_filters(self.img_smoothed)
            self.nonMaxImg = self.non_max_suppression(self.img_gradient, self.thetaMat)
            self.thresholdImg = self.threshold(self.nonMaxImg)
            img_final = self.hysteresis(self.thresholdImg)
            self.imgs_final.append(img_final)

        return self.imgs_final

