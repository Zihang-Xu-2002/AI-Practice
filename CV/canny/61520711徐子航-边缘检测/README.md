# README

The './img' fold contains images to be tested

The './result' fold contains final edge images

Initially, I do the experiment in './EdgeDetection_gaussian.ipynb', where I get the gradient with the first order derivative gaussian kernel.

Later, I pack the functions used in './EdgeDetection_gaussian.ipynb' like NMS, double threshold and hysteresis into './canny.py', some tools are put into utils.py like grayscale process and visualize. This make it easier for me to use this canny detector. Besides, I tried sobel kernel in './canny.py' and test it in './EdgeDetection_sobel.ipynb'

As for the experiment analysis and procedure, these contents are in './61520711徐子航-边缘检测报告.pdf'.