# coding: UTF-8

import cv2
import glob
import os
import shutil
import numpy as np
import math
import copy

def giveblackarea(img_input):
    img = copy.copy(img_input)
    #http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.medianBlur(img,5)
    img_processed = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    whole_area = img_processed.size
    whitePixels = cv2.countNonZero(img_processed)
    blackPixels = whole_area - whitePixels
    # print(f'blackPixelsは{blackPixels}')
    return blackPixels


def determine_alphabeta(img_input):
   
    #determine alpha and beta in the negaposi image
    img_template = copy.copy(img_input)
    img_copy = copy.copy(img_input)
    height, width, c = img_template.shape
    alpha_best = 0.0
    beta_best = 0.0
    blackarea_minimum = 416 * 416 #the size of img
    for alpha_pre in range(-5, 30):
        alpha = float(alpha_pre / 1000)
        for beta_pre in range(-40, 40):
            beta = float(beta_pre / 10000)
            staffmiddle = int(0.5*416 + alpha*416)
            heightInterval =  int(height/(8 +2*(8*1.2)) + beta*416)
            img_copy = copy.copy(img_template)
            for i in range(0,3):
                #draw an upper and a lower line:blue
                upper_line_position = int(staffmiddle - 2*heightInterval*i)
                if upper_line_position >= 0:
                    img_copy = cv2.line(img_copy,(0,upper_line_position),(416,upper_line_position),(255,0,0),2)
                lower_line_position = int(staffmiddle + 2*heightInterval*i)
                if upper_line_position <= 416:
                    img_copy = cv2.line(img_copy,(0,lower_line_position),(416,lower_line_position),(255,0,0),2)
            #draw a middle line:red and title in the image
            img_copy = cv2.line(img_copy,(0,staffmiddle),(416,staffmiddle),(0,0,255),2)
            blackarea = giveblackarea(img_copy)
            if blackarea < blackarea_minimum:
                alpha_best = alpha
                beta_best = beta
                blackarea_minimum = blackarea
    return alpha_best, beta_best


