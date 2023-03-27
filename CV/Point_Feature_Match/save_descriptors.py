import cv2
import numpy as np
from os import walk
from os.path import join

def create_descriptors(folder):
    files = []
    for (dirpath, dirnames, filenames) in walk(folder):
        files.extend(filenames)
    for f in files:
        if '.png' in f:
            save_descriptor(folder, f, cv2.SIFT_create())

def save_descriptor(folder, image_path, feature_detector):
    # judge npy
    if image_path.endswith("npy"):
        return
    img = cv2.cvtColor(cv2.imread(join(folder,image_path), cv2.IMREAD_COLOR),cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = feature_detector.detectAndCompute(img, None)
    # save into .npy
    descriptor_file = image_path.replace("png", "npy")
    np.save(join(folder, descriptor_file), descriptors)

if __name__=='__main__':
    path = 'D:\\learn\\Computer\\HRI\\HW01\\img\\dataset'
    create_descriptors(path)
