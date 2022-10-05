import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#这里引入cv2仅仅是用Mat存储图片
from cv2 import waitKey



path = './img/hill.jpg'
src=cv2.imread(path)
cv2.imshow('raw image',src)
cv2.waitKey(0)
# image_src = Image.fromarray(cv2.cvtColor(src,cv2.COLOR_BGR2RGB))
# display(image_src)
print(type(src))
print(src.shape)

def grayscale_process(src,way=1):
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



dst = grayscale_process(src,3)
cv2.imshow("grayscale",dst) 
cv2.waitKey(0)

def gaussian_kernel(size, sigma=1):
    half_size = int(size) // 2
    x, y = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    gk =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

    gaussian_first_deriv_x = np.zeros_like(gk)
    gaussian_first_deriv_y = np.zeros_like(gk)
    #assert(size % 2 == 1) # I assume you are using an odd kernel_size
    #print("size",size)
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
        print("factor_x",x)
        print("gk[",i,"]",gk[i])
        gaussian_first_deriv_x[i] = gk[i] * factor_x
        gaussian_first_deriv_y[i] = gk[i] * factor_y
    gaussian_first_deriv_y=gaussian_first_deriv_y.T
    return gk,gaussian_first_deriv_x,gaussian_first_deriv_y

gk,gd_x,gd_y=gaussian_kernel(5)
print("----------------------")
print("-----gk---------------")
print(gk)
print("-----gd_x-------------")
print(gd_x)
print("-----gd_y-------------")
print(gd_y)

def convolve(img,fil,mode = 'same'):                #分别提取三个通道

    if mode == 'fill':
        h = fil.shape[0] // 2
        w = fil.shape[1] // 2
        img = np.pad(img, ((h, h), (w, w),(0, 0)), 'constant')
    # conv_b = _convolve(img[:,:,0],fil)              #然后去进行卷积操作
    # conv_g = _convolve(img[:,:,1],fil)
    # conv_r = _convolve(img[:,:,2],fil)
    conv = _convolve(img,fil)
 
    #dstack = np.dstack([conv_b,conv_g,conv_r])      #将卷积后的三个通道合并
    return conv                                   #返回卷积后的结果
def _convolve(img,fil):         
     
    fil_heigh = fil.shape[0]                        #获取卷积核(滤波)的高度
    fil_width = fil.shape[1]                        #获取卷积核(滤波)的宽度
   
    conv_heigh = img.shape[0] - fil.shape[0] + 1    #确定卷积结果的大小
    conv_width = img.shape[1] - fil.shape[1] + 1

    conv = np.zeros((conv_heigh,conv_width),dtype = 'uint8')
     
    for i in range(conv_heigh):
        for j in range(conv_width):                 #逐点相乘并求和得到每一个点
            conv[i][j] = wise_element_sum(img[i:i + fil_heigh,j:j + fil_width ],fil)
    return conv   
def wise_element_sum(img,fil):
    res = (img * fil).sum() 
    if(res < 0):
        res = 0
    elif res > 255:
        res  = 255
    return res

img_gd_x = convolve(dst,50*gd_x,'same')
img_gd_y = convolve(dst,50*gd_y,'same')
G = np.hypot(img_gd_x, img_gd_y)
G = G / G.max() * 255
theta = np.arctan2(img_gd_y, img_gd_x)

print(img_gd_x)
print(img_gd_y)
print(img_gd_x.shape)
cv2.imshow('img_gd_x',img_gd_x)
cv2.imshow('img_gd_y',img_gd_y)
#cv2.imshow('G',G)
cv2.waitKey(0)

print(type(G))
print(G)
mixed = G.astype("uint8")
cv2.imshow("window",mixed)
cv2.waitKey(0)

def non_max_suppression(img, arc_angle):
    m,n = img.shape
    Z = np.zeros((m,n))
    angle = arc_angle* 180. / np.pi #因为之前用arctan算出来的是弧度制，后面要分类讨论，用角度比较方便
    angle[angle < 0] += 180

    for i in range(1,m-1):
        for j in range(1,n-1):
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
    Z=Z.astype("uint8")
    
    return Z

img_NMS = non_max_suppression(mixed,theta)
cv2.imshow("img_NMS",img_NMS)
cv2.waitKey(0)
print(type(img_NMS))

def double_threshold(img,lowthresholdrate=0.05,highthresholdrate=0.11):
    HT=highthresholdrate*img.max()
    LT=lowthresholdrate*img.max()

    m,n = img.shape
    result_DT=np.zeros((m,n))
    img_low=25
    img_high=255

    strong_i,strong_j = np.where(img>=HT)
    weak_i,weak_j = np.where((img>=LT)&(img<=HT))
    zeros_i,zeros_j = np.where(img<LT)

    result_DT[strong_i,strong_j] = img_high
    result_DT[weak_i,weak_j] = img_low
    
    return result_DT
def hysteresis(img, weak=25,strong=255):
    M, N = img.shape  
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
    
img_DT = double_threshold(img_NMS)
img_final = hysteresis(img_DT)
print(type(img_final))
img_canny = img_final.astype("uint8")
plt.imshow(img_canny)
plt.show()
cv2.imshow('canny',img_canny)
cv2.waitKey(0)