from skimage.measure import compare_ssim
import imutils
import cv2
import os
import numpy as np

def load_images(imageA,imageB):
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    return imageA, imageB, grayA, grayB


def compute_SSIM(img1, img2, full=True):
    # compute the Structural Similarity Index (SSIM) between the two images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(img1, img2, full=full)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    return diff

def compute_thresh_and_cnts(diff, thresh_delta=70):
    # threshold the difference image, followed by finding contours to obtain the regions of the two input images that differ
    # use thresh_delta to make the differ region more accurate
    ret = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[0]
    print(ret)
    ret = ret-thresh_delta if ret>thresh_delta else 0
    thresh = cv2.threshold(diff, ret, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    return thresh, cnts

def draw_bounding_box(img1,img2, cnts, max_region=True):
    # loop over the contours
    max_size = 0
    coord = list()
    for c in cnts:
        # compute the bounding box of the contour and then draw the bounding box on both input images to represent where the two images differ
        (x, y, w, h) = cv2.boundingRect(c)
        if max_region:
            size = w * h
            if size > max_size:
                max_size = size
                coord = [x, y, w, h]
        else:
            img1 =255 * img1
            img1 = np.ascontiguousarray(img1, dtype=np.uint8)
            img2 =255 * img2
            img2 = np.ascontiguousarray(img2, dtype=np.uint8)
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if max_region: 
        img1 =255 * img1
        img1 = np.ascontiguousarray(img1, dtype=np.uint8)
        img2 =255 * img2
        img2 = np.ascontiguousarray(img2, dtype=np.uint8)
        #img2 =np.array(img2, dtype=np.unit8)
        cv2.rectangle(img1, (coord[0], coord[1]), (coord[0] + coord[2], coord[1] + coord[3]), (0, 0, 255), 2)
        cv2.rectangle(img2, (coord[0], coord[1]), (coord[0] + coord[2], coord[1] + coord[3]), (0, 0, 255), 2)
        print(coord[0] + coord[2]/2, coord[1] + coord[3]/2)

    return img1, img2