import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3

import cv2
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import numpy as np
import glob

PERCENTAGE = 0.35  # indicate how an image should be resized into


# this function downscalles every image in the list of images given. This
# images are downsized by the percentage amount

def downscale(img, percent):
    width = int(img.shape[1] * percent)
    height = int(img.shape[0] * percent)
    dim = (width, height)

    resized = cv2.resize(img, dim)

    return resized


# This function resizes all the images inthe given list by a PERCENTAGE
def resizeAll(images):
    retImages = []
    for img in images:
        retImages.append(downscale(img, percent=PERCENTAGE))
    return retImages


# the function will read all the images into a list from the simutosot file
def readImg():
    images = np.array([cv2.imread(file) for file in glob.glob("simurosot/*.jpg")])
    return images


# Perform the closing morphological operations in alist of images
# and return the new images in alist

def getClosing(images):
    res = []
    for img in images:
        kernel = np.ones((2, 2), np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        res.append(closing)
    return res


# Perform the opeining morphological operation on the list of images passed into
# function

def getOpening(images):
    res = []
    for img in images:
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        res.append(opening)
    return res


# Draw a boundary rectangle
def boundingRectangle(x, y, w, h, img):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# draw a bounding rectangle for a list of images that are passed int o the function

def boundingRectangleAllImages(contourValues, img):
    for contour, img1 in zip(contourValues, img):
        x, y, w, h = contour
        boundingRectangle(x, y, w, h, img1)


# Purpose: Get the mask of the image given the lower and upper number and return the masked images
# return : The function should return the list of masked images in the file
def getMask(images, lower, upper):
    mask_images = []
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask_images.append(mask)
    return mask_images


# Purpose : Get the contour value of an image and draw a rectangle box on the contour area given a contour threshold
# Retun : The function should return the x,y,w,h values if the object
def getContours(img, img_copy):
    # we find the contours of an image
    # recicev  the outer corners(by request for all contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # loop through the contours
    x = 0
    y = 0
    w = 0
    h = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)

        # give minimum threshold so that we dont detetct any noise (area of 500)
        if area > 500:
            peri = cv2.arcLength(cnt, True)  # the contour parameter
            print("Area is {}".format(area))
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)  # should give the coner points of shapes

            # using a rectanglar box
            x, y, w, h = cv2.boundingRect(approx)

        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return x, y, w, h


# This function get the contour value of the list of images passed into the function
# retun: The functiuon should return the list of the conotur value of all theimagbes in the list
def getAllContour(images, originalImage):
    # get the contour images of all the images in the list pf images
    contourImages = []
    for img, imgCopy in zip(images, originalImage):
        contourImages.append(getContours(img, imgCopy))
    return contourImages


# The function should add the mask of all the images in the list

def addMask(mask1, mask2):
    valList = []
    for ma1, ma2 in zip(mask1, mask2):
        val = ma1 + ma2
        valList.append(val)
    return valList


while True:
    frame = resizeAll(readImg())

    red = np.array([121, 161, 0, 179, 255, 255])
    white = np.array([116, 0, 0, 179, 255, 175])
    # this gives us the result bafter from nthe track pads movement
    # define the lower bound and the upper bound
    lower_white = np.array(white[0:3])
    upper_white = np.array(white[3:6])

    lower = np.array(red[0:3])
    upper = np.array(red[3:6])

    mask = getMask(frame, lower, upper)

    mask2 = getMask(frame, lower_white, upper_white)
    opening_list = getOpening(mask2)
    cllosing_list = getClosing(opening_list)

    img_copy = frame.copy()
    finalImage = addMask(mask, opening_list)

    getAllContour(finalImage, img_copy)

    smImage = np.hstack(img_copy[0:(len(img_copy) // 2)])
    sm2 = np.hstack(img_copy[-(len(img_copy) // 2):])
    ver = np.vstack((sm2, smImage))
    cv2.imshow("<Mask>", ver)

    cv2.waitKey(1)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break;

# cv2.destroyWindow()
