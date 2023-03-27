import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
from os import walk
from os.path import join

MIN_MATCH_COUNT = 10

img_query = cv.cvtColor(cv.imread("./img/query/1.png",cv.IMREAD_COLOR),cv.COLOR_BGR2GRAY)

folder = 'D:\\learn\\Computer\\HRI\\HW01\\img\\dataset'
descriptors = []
for (dirpath, dirnames, filenames) in walk(folder):
    for f in filenames:
        if f.endswith("npy"):
            descriptors.append(f)
    print(descriptors)


sift = cv.SIFT_create()
keypoints_q, descriptors_q = sift.detectAndCompute(img_query, None)

bf = cv.BFMatcher()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)

candidate = {}
for d in descriptors:
    matches = flann.knnMatch(descriptors_q, np.load(join(folder, d)), k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    print("dataset img is %s . matching value is %d" % (d, len(good)))
    candidate[d] = len(good)

max_match_value = None
candidate_path = None
for im_path, match_value in candidate.items():
    if max_match_value == None or match_value > max_match_value:
        max_match_value = match_value
        candidate_path = im_path

#print(candidate_path.replace(".npy",""))

print("most similar image is pic%s" % candidate_path.replace(".npy",""))

most_similar_img = join(folder,candidate_path.replace(".npy",".png"))
print(most_similar_img)

img_dataset = cv.cvtColor(cv.imread(most_similar_img,cv.IMREAD_COLOR),cv.COLOR_BGR2GRAY)

keypoints_d, descriptors_d = sift.detectAndCompute(img_dataset,None)
match_points = flann.knnMatch(descriptors_q,descriptors_d,k=2)
good_match = []
for m,n in match_points:
    if m.distance < 0.75*n.distance:
        good_match.append(m)

if len(good_match)>MIN_MATCH_COUNT:
    src_pts = np.float32([ keypoints_q[m.queryIdx].pt for m in good_match ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints_d[m.trainIdx].pt for m in good_match ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img_query.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    #box_color = (0,0,255)
    sample_draw = cv.merge((img_dataset.copy(),img_dataset.copy(),img_dataset.copy()))
    img_sample_detected = cv.polylines(sample_draw,[np.int32(dst)],True,(255,0,0),5, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
query_draw = cv.merge((img_query.copy(),img_query.copy(),img_query.copy()))
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(query_draw,keypoints_q,img_sample_detected,keypoints_d,good_match,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()