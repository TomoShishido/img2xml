# coding: UTF-8

import cv2
import glob
import os
import shutil
import numpy as np
import math
import copy

MIN_X_WIDTH = 300
MIN_LINE_LENGTH = 100

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def leveloriginalimg(FILE_PATH):
    img = cv2.imread(FILE_PATH)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)  
    d_delta = math.degrees(math.atan((y2-y1)/(x2-x1)))
    print(d_delta)
    image = rotate_image(img,d_delta)
    
    files_temp = glob.glob(FILE_PATH) #"./tmp/*":beforehand prepare images and Yolov5 anotation files in ./tmp/subdirectory
    #To skip .txt files
    FILE_DIR_PATH = ''
    FILE_BASENAME = ''
    FILE_BASENAME_WITHOUTEXT = ''
    for file_temp in files_temp:
        if file_temp.endswith('jpg') or file_temp.endswith('png'):
            FILE_DIR_PATH = os.path.dirname(file_temp)
            FILE_BASENAME = os.path.basename(file_temp)
            FILE_BASENAME_WITHOUTEXT = os.path.splitext(FILE_BASENAME)[0]
            FILE_EXT = os.path.splitext(FILE_BASENAME)[1].lower()
            LEVELED_FILE_PATH = FILE_DIR_PATH + '/leveled_' + FILE_BASENAME_WITHOUTEXT + FILE_EXT
            cv2.imwrite(LEVELED_FILE_PATH, image)
    return LEVELED_FILE_PATH

def leveleachmeasure(FILE_PATH):
    img = cv2.imread(FILE_PATH)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180, MIN_LINE_LENGTH)
    lines_coord = []
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            if x2 - x1 < MIN_X_WIDTH:
                continue

            # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
            lines_coord.append((x1, y1, x2, y2))
        
    # plt.imshow(img)
    # f_out = 'out_' + filename
    # cv2.imwrite(f_out, img)

    (x1, y1, x2, y2) = lines_coord[0]
    d_delta = math.degrees(math.atan((y2-y1)/(x2-x1)))

    out_image = rotate_image(img, d_delta) 
    cv2.imwrite(FILE_PATH, out_image)

    return d_delta

