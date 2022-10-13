from turtle import width
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#这里引入cv2仅仅是用Mat存储图片
from cv2 import waitKey






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

def sobel():
    sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]
    return sobel_x,sobel_y

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

    fil_heigh = 3
    fil_width = 3      
     
    # fil_heigh = fil.shape[0]                        #获取卷积核(滤波)的高度
    # fil_width = fil.shape[1]                        #获取卷积核(滤波)的宽度
   
    conv_heigh = img.shape[0] - fil_heigh + 1    #确定卷积结果的大小
    conv_width = img.shape[1] - fil_width + 1

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



#@gd_x
#@gd_y
#
def calculate_value_and_arctan(gd_x,gd_y):
    G = np.hypot(gd_x, gd_y)
    G = G / G.max() * 255
    theta = np.arctan2(gd_y, gd_x)
    return G,theta

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
                
                                #angle 45
                if (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]

                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]
                #angle 0
                elif (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]

                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]


                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    Z=Z.astype("uint8")
    
    return Z

#hill : lowthresholdrate=0.06,highthresholdrate=0.6
def double_threshold(img,lowthresholdrate=0.07,highthresholdrate=0.15):
    HT=highthresholdrate*img.max()
    LT=lowthresholdrate*img.max()

    HT = 50
    LT = 10



    print("HT",HT)
    print("LT",LT)

    m,n = img.shape
    result_DT=np.zeros((m,n))
    img_low=50
    img_high=255

    strong_i,strong_j = np.where(img>=HT)
    weak_i,weak_j = np.where((img>=LT)&(img<=HT))
    zeros_i,zeros_j = np.where(img<LT)

    result_DT[strong_i,strong_j] = img_high
    result_DT[weak_i,weak_j] = img_low
    result_DT[zeros_i,zeros_j]=0
    
    return result_DT
def hysteresis(img, weak=50,strong=255):
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
    
def Canny_Detector():
    #path = './img/hill.jpg'
    path = './img/yw.jpg'
    kernelsize = 3
    
    src=cv2.imread(path)
    
    cv2.imwrite("./result/raw_yw.jpg",src)
    # cv2.imshow('raw image',src)
    # cv2.waitKey(0)
    f = open("./log.txt",'w')
    print(type(src))
    f.write(path + '\n')
    f.write("Begin the log\n")
    f.write("The shape of raw image:"+str(src.shape)+'\n')

############ OpenCV自带Canny检测 ##############################
#     t_lower = 50  # Lower Threshold
#     t_upper = 150  # Upper threshold
  
# # Applying the Canny Edge filter
#     edge = cv2.Canny(src, t_lower, t_upper)
#     cv2.imwrite("./result/hill_opencv_canny.jpg",edge)
#     cv2.imshow('original', src)
#     cv2.imshow('edge', edge)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
############ OpenCV自带Canny检测 ##############################

    gray_src = grayscale_process(src,3)
    cv2.imwrite("./result/yw_gray_3.jpg",gray_src)
    gk,gd_x,gd_y=gaussian_kernel(kernelsize)
    f.write("---------image gradient-------------\n")
    f.write("---------gaussian kernel-------------\n")
    f.write(str(gk)+'\n')
    f.write("---------G_x-------------\n")
    f.write(str(gd_x)+'\n')
    f.write("---------G_y-------------\n")
    f.write(str(gd_y)+'\n')
    f.write("---------image gradient-------------\n")
    sobel_x,sobel_y = sobel()
    img_smooth = convolve(gray_src,gk,'same')
    # img_gd_x = convolve(img_smooth,50*gd_x,'same')
    # img_gd_y = convolve(img_smooth,50*gd_y,'same')
    img_gd_x = convolve(img_smooth,sobel_x,'same')
    img_gd_y = convolve(img_smooth,sobel_y,'same')
    #cv2.imwrite("./result/hill_gd_x.jpg",img_gd_x)
    #cv2.imwrite("./result/hill_gd_y.jpg",img_gd_y)
    cv2.imwrite("./result/yw_gd_x_sobel.jpg",img_gd_x)
    cv2.imwrite("./result/yw_gd_y_sobel.jpg",img_gd_y)

    G, theta = calculate_value_and_arctan(img_gd_x,img_gd_y)
    # f.write("---------Gradient value-------------\n")
    # f.write(str(G)+'\n')
    # f.write("---------Gradient angle-------------\n")
    # f.write(str(theta)+'\n')

    gradient_img = G.astype("uint8")
    #cv2.imwrite("./result/hill_gd_mix.jpg",gradient_img)
    cv2.imwrite("./result/yw_gd_mix_sobel.jpg",gradient_img)

    #below is NMS
    print("begin NMS")
    NMS_img = non_max_suppression(gradient_img,theta)
    #cv2.imwrite("./result/hill_NMS.jpg",NMS_img)
    cv2.imwrite("./result/yw_NMS_sobel.jpg",NMS_img)

    #below is double threshold
    DT_img = double_threshold(NMS_img)
    #cv2.imwrite("./result/hill_DT.jpg",DT_img)
    cv2.imwrite("./result/yw_DT_sobel.jpg",DT_img)
    final_img = hysteresis(DT_img)
    #cv2.imwrite("./result/hill_final.jpg",final_img)
    cv2.imwrite("./result/yw_final_sobel.jpg",final_img)





if __name__=="__main__":
    Canny_Detector()

