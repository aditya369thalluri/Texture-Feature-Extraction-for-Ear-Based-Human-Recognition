import numpy as np
import cv2,Tkinter
import scipy
from scipy.stats.stats import pearsonr
import os
import glob
from matplotlib import pyplot as plt
from numpy import *

def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])

	# return the chi-squared distance
	return d

def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx,idy]
    except IndexError:
        return default
imgd = cv2.imread('001_zoom_ear.jpg', 0)
transformed_imgd = cv2.imread('001_zoom_ear.jpg', 0)
lbp2=list()
for x in range(0, len(imgd)):
    for y in range(0, len(imgd[0])):
        center        = imgd[x,y]
        top_left      = get_pixel_else_0(imgd, x-1, y-1)
        top_up        = get_pixel_else_0(imgd, x, y-1)
        top_right     = get_pixel_else_0(imgd, x+1, y-1)
        right         = get_pixel_else_0(imgd, x+1, y )
        left          = get_pixel_else_0(imgd, x-1, y )
        bottom_left   = get_pixel_else_0(imgd, x-1, y+1)
        bottom_right  = get_pixel_else_0(imgd, x+1, y+1)
        bottom_down   = get_pixel_else_0(imgd, x,   y+1 )

        values = thresholded(center, [top_left, top_up, top_right, right, bottom_right,
                                      bottom_down, bottom_left, left])

        weights = [1, 2, 4, 8, 16, 32, 64, 128]
        res = 0
        for a in range(0, len(values)):
            res += weights[a] * values[a]

        transformed_imgd.itemset((x,y), res)
    lbp2.append(res)
path='C:/Users/premchand/Desktop/AMI Ear DB/subset-1'
i=1
for infile in glob.glob(os.path.join(path,'*.jpg')):
    print "IMAGE",i 
    imgb = cv2.imread(infile,0)
    transformed_imgb = cv2.imread(infile, 0)
    lbp1=list()
    for x in range(0, len(imgb)):
        for y in range(0, len(imgb[0])):
            center        = imgb[x,y]
            top_left      = get_pixel_else_0(imgb, x-1, y-1)
            top_up        = get_pixel_else_0(imgb, x, y-1)
            top_right     = get_pixel_else_0(imgb, x+1, y-1)
            right         = get_pixel_else_0(imgb, x+1, y )
            left          = get_pixel_else_0(imgb, x-1, y )
            bottom_left   = get_pixel_else_0(imgb, x-1, y+1)
            bottom_right  = get_pixel_else_0(imgb, x+1, y+1)
            bottom_down   = get_pixel_else_0(imgb, x,   y+1 )

            values = thresholded(center, [top_left, top_up, top_right, right, bottom_right,
                                      bottom_down, bottom_left, left])

            weights = [1, 2, 4, 8, 16, 32, 64, 128]
            res = 0
            for a in range(0, len(values)):
                res += weights[a] * values[a]

            transformed_imgb.itemset((x,y), res)
        lbp1.append(res)

    pc,ps=pearsonr(lbp1,lbp2)
    if pc==1.0 and ps==0.0:
        
        print "matched!!!"
        print "pearson coffecient=",pc,"pearson significant=",ps
        hist1 = cv2.calcHist([imgb],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([imgd],[0],None,[256],[0,256])
        distance=chi2_distance(hist1,hist2)
        print distance
        
        cv2.imshow('back ear image', imgb)
        cv2.imshow('thresholded image of back ear', transformed_imgb)
        cv2.imshow('down ear image', imgd)
        cv2.imshow('thresholded image of down ear', transformed_imgd)
        hist,bins = np.histogram(imgb.flatten(),256,[0,256])
        plt.hist(transformed_imgb.flatten(),256,[0,256], color = 'r')
        hist,bins = np.histogram(imgd.flatten(),256,[0,256])
        plt.hist(transformed_imgd.flatten(),256,[0,256], color = 'g')
        plt.xlim([0,256])
        plt.legend(('back ear','down ear'), loc = 'upper left')
        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()
    else:
        print "not matched"
        print "pearson coffecient=",pc,"pearson significant=",ps
        i=i+1
print "no matching"
exit()
