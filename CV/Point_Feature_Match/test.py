import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

MIN_MATCH_COUNT = 10

img_sample = cv.cvtColor(cv.imread("./img/dataset/9.png",cv.IMREAD_COLOR),cv.COLOR_BGR2GRAY)

img_q = cv.cvtColor(cv.imread("./img/query/3.png",cv.IMREAD_COLOR),cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img_q, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img_sample, None)

outImage_1 = cv.drawKeypoints(img_q, keypoints_1,None)
outImage_2 = cv.drawKeypoints(img_sample, keypoints_2,None)

print(len(keypoints_1))
print(len(keypoints_2))

#cv.imwrite('image.jpg', outImage_1)
#cv.waitKey(0)


# BFMatcher 
def BFMatcher(descript_1,descript_2):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descript_1,descript_2,k=2)
    return matches

# FLANNMatcher
def FLANNMatcher(descript_1,descript_2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descript_1,descript_2,k=2)
    return matches

start = time.time()
matches = BFMatcher(descriptors_1,descriptors_2)
#matches = FLANNMatcher(descriptors_1,descriptors_2)
end = time.time()
print("time cost: ",end-start)

# ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)
# cv.drawMatchesKnn expects list of lists as matches.
print("match pairs: ", len(good))
# img4 = cv.drawMatchesKnn(img_q,keypoints_1,img_sample,keypoints_2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img4),plt.show()

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ keypoints_1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints_2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img_q.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    #box_color = (0,0,255)
    sample_draw = cv.merge((img_sample.copy(),img_sample.copy(),img_sample.copy()))
    img_sample_detected = cv.polylines(sample_draw,[np.int32(dst)],True,(255,0,0),5, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
query_draw = cv.merge((img_q.copy(),img_q.copy(),img_q.copy()))
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(query_draw,keypoints_1,img_sample_detected,keypoints_2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()


