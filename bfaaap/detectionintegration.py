"""
From the sheet music img file path, perform the first detection for staff recognition in sheet music
"""
import cv2
import glob
import os
import subprocess
import shutil
import numpy as np
import math
from enum import Enum, auto
import copy
from itertools import chain
import time
import subprocess


#時間計測start
start = time.time()
print(start)

#work at the yolov5 directory
os.chdir('/Users/tomoimacpro/TSh_project/bfaaap/yolov5/')

FILE_PATH ='/Users/tomoimacpro/TSh_project/bfaaap/musicdata/emisan/sarabandePhotoInclinedのコピー.jpg'

#First, level the original sheet music image
from leveloriginalimg.leveloriginalimg import leveloriginalimg

FILE_PATH = leveloriginalimg(FILE_PATH)

#extract staves with measures
SAVE_DIRECTORY_PATH = os.path.dirname(FILE_PATH) + '/staff'

proc = subprocess.Popen(['python','detect.py', '--weights', '/Users/tomoimacpro/TSh_project/bfaaap/yolov5/weightsstock/last_0.95_staff4_20201230.pt', '--SAVE_PATH', SAVE_DIRECTORY_PATH ,'--img', '416', '--conf', '0.75', '--source', FILE_PATH, '--save-txt'])
proc.wait()


#perform inference on sheet music

#After the measure-recognizing model is applied to a piece of sheet music

#copy and move the relevant files under a dirctory ./musicdata/AAA/(staff/labels)
#sheet music (.jpg) provided in FILE_PATH
# FILE_PATH = '/Users/tomoimacpro/TSh_project/bfaaap/musicdata/patheticMov2/PatheticMov2_1.jpg'

files_temp = glob.glob(FILE_PATH) #"./tmp/*":beforehand prepare images and Yolov5 anotation files in ./tmp/subdirectory
    #To skip .txt files
for file_temp in files_temp:
    if file_temp.endswith('jpg') or file_temp.endswith('png'):
        img = cv2.imread(file_temp)
        dirname = os.path.dirname(file_temp)
        basename = os.path.basename(file_temp)
        cv2.imwrite(dirname + '/staff/labels/' + basename, img)



from alignmeasures.align_measures import generate_measures_in_eachstave_aslist
#sheet music provided in FILE_PATH
staves_with_measures_in_sheetmusic = generate_measures_in_eachstave_aslist(FILE_PATH)
print(f'staves_with_measures_in_sheetmusicのstave数は{len(staves_with_measures_in_sheetmusic)}')
for i, each_staff in enumerate(staves_with_measures_in_sheetmusic):
    print(f'{i}番目の小節(measure)の数は{len(each_staff)}個です。')
    # print(f'{i}番目の小節(measure)は{each_staff}個です。')

#input wheter staves are paired
areStavesPaired = True
# isInputFinished = False
# while not isInputFinished:
#     user_answer = input("Are staves paird? ").lower().strip()
#     if user_answer == "yes":
#         areStavesPaired = True
#         isInputFinished = True
#     elif user_answer == "no":
#         areStavesPaired = False
#         isInputFinished = True
#     else:
#         print("Error: Answer must be yes or no") 
   

#excise and enlarge each measure img at 412 x 412 pixels in each staff and stored in musicdata/AAA/measure/staff1/ or staff2/
from enlargemeasures.enlargeeachmeasure import produceResizedMeasuresFromAlignedStaves
#in the case of wide staff extraction, set staff_magnification = 1.2
staff_magnification = 1.2

produceResizedMeasuresFromAlignedStaves(img_FILE_PATH=FILE_PATH, aligned_staves=staves_with_measures_in_sheetmusic, isPaired=areStavesPaired, upper_margin=staff_magnification, lower_margin=staff_magnification)

#level again each measure one by one
from leveloriginalimg.leveloriginalimg import leveleachmeasure

MEASURES_STAFF1_PATH = os.path.dirname(FILE_PATH) + '/measure/staff1/*'
files_temp = glob.glob(MEASURES_STAFF1_PATH)
for file_temp in files_temp:
    if file_temp.endswith('jpg') or file_temp.endswith('png'):
        FILE_DIR_PATH = os.path.dirname(file_temp)
        FILE_BASENAME = os.path.basename(file_temp)
        THIS_PATH = FILE_DIR_PATH + '/' + FILE_BASENAME
        result0 = leveleachmeasure(THIS_PATH)

MEASURES_STAFF2_PATH = os.path.dirname(FILE_PATH) + '/measure/staff2/*'
files_temp = glob.glob(MEASURES_STAFF2_PATH)
for file_temp in files_temp:
    if file_temp.endswith('jpg') or file_temp.endswith('png'):
        FILE_DIR_PATH = os.path.dirname(file_temp)
        FILE_BASENAME = os.path.basename(file_temp)
        THIS_PATH = FILE_DIR_PATH + '/' + FILE_BASENAME
        leveleachmeasure(THIS_PATH)


"""
apply individual models to the measures selected for staff 1 or 2
"""
#work at the yolov5 directory
os.chdir('/Users/tomoimacpro/TSh_project/bfaaap/yolov5/')

#processes in parallel
processes = []


# FILE_PATH ='/Users/tomoimacpro/TSh_project/bfaaap/musicdata/testfordetection1/PatheticMov2_1.jpg'
#image source directory path for either staff1 or staff2
"""
#For staff1
"""
SOURCE_PATH = os.path.dirname(FILE_PATH) + '/measure/staff1/*'

#body
SAVE_DIRECTORY_PATH = os.path.dirname(FILE_PATH) + '/staff1/body'
proc = subprocess.Popen(['python','detect.py', '--weights', '/Users/tomoimacpro/TSh_project/bfaaap/yolov5/weightsstock/last_0.94_body4_20210208.pt', '--SAVE_PATH', SAVE_DIRECTORY_PATH ,'--img', '416', '--conf', '0.60', '--source', SOURCE_PATH, '--save-txt'])
# proc.wait()
processes.append((0, proc))
#armbeam
SAVE_DIRECTORY_PATH = os.path.dirname(FILE_PATH) + '/staff1/armbeam'
proc = subprocess.Popen(['python','detect.py', '--weights', '/Users/tomoimacpro/TSh_project/bfaaap/yolov5/weightsstock/last_0.99_armbeam2_20210214.pt', '--SAVE_PATH', SAVE_DIRECTORY_PATH ,'--img', '416', '--conf', '0.60', '--source', SOURCE_PATH, '--save-txt'])
# proc.wait()
processes.append((1, proc))
#accidental
SAVE_DIRECTORY_PATH = os.path.dirname(FILE_PATH) + '/staff1/accidental'
proc = subprocess.Popen(['python','detect.py', '--weights', '/Users/tomoimacpro/TSh_project/bfaaap/yolov5/weightsstock/last_0.99_Accidental2_20210209.pt', '--SAVE_PATH', SAVE_DIRECTORY_PATH ,'--img', '416', '--conf', '0.60', '--source', SOURCE_PATH, '--save-txt'])
# proc.wait()
processes.append((2, proc))
#rest
SAVE_DIRECTORY_PATH = os.path.dirname(FILE_PATH) + '/staff1/rest'
proc = subprocess.Popen(['python','detect.py', '--weights', '/Users/tomoimacpro/TSh_project/bfaaap/yolov5/weightsstock/last_0.99_rest1_20210107.pt', '--SAVE_PATH', SAVE_DIRECTORY_PATH ,'--img', '416', '--conf', '0.60', '--source', SOURCE_PATH, '--save-txt'])
# proc.wait()
processes.append((3, proc))
#clef
SAVE_DIRECTORY_PATH = os.path.dirname(FILE_PATH) + '/staff1/clef'
proc = subprocess.Popen(['python','detect.py', '--weights', '/Users/tomoimacpro/TSh_project/bfaaap/yolov5/weightsstock/last_0.99_Clef3_20210129.pt', '--SAVE_PATH', SAVE_DIRECTORY_PATH ,'--img', '416', '--conf', '0.60', '--source', SOURCE_PATH, '--save-txt'])
# proc.wait()
processes.append((4, proc))
"""
#For staff2
"""
SOURCE_PATH = os.path.dirname(FILE_PATH) + '/measure/staff2/*'

#body
SAVE_DIRECTORY_PATH = os.path.dirname(FILE_PATH) + '/staff2/body'
proc = subprocess.Popen(['python','detect.py', '--weights', '/Users/tomoimacpro/TSh_project/bfaaap/yolov5/weightsstock/last_0.94_body4_20210208.pt', '--SAVE_PATH', SAVE_DIRECTORY_PATH ,'--img', '416', '--conf', '0.60', '--source', SOURCE_PATH, '--save-txt'])
# proc.wait()
processes.append((5, proc))
#armbeam
SAVE_DIRECTORY_PATH = os.path.dirname(FILE_PATH) + '/staff2/armbeam'
proc = subprocess.Popen(['python','detect.py', '--weights', '/Users/tomoimacpro/TSh_project/bfaaap/yolov5/weightsstock/last_0.99_armbeam2_20210214.pt', '--SAVE_PATH', SAVE_DIRECTORY_PATH ,'--img', '416', '--conf', '0.60', '--source', SOURCE_PATH, '--save-txt'])
# proc.wait()
processes.append((6, proc))
#accidental
SAVE_DIRECTORY_PATH = os.path.dirname(FILE_PATH) + '/staff2/accidental'
proc = subprocess.Popen(['python','detect.py', '--weights', '/Users/tomoimacpro/TSh_project/bfaaap/yolov5/weightsstock/last_0.99_Accidental2_20210209.pt', '--SAVE_PATH', SAVE_DIRECTORY_PATH ,'--img', '416', '--conf', '0.60', '--source', SOURCE_PATH, '--save-txt'])
# proc.wait()
processes.append((7, proc))
#rest
SAVE_DIRECTORY_PATH = os.path.dirname(FILE_PATH) + '/staff2/rest'
proc = subprocess.Popen(['python','detect.py', '--weights', '/Users/tomoimacpro/TSh_project/bfaaap/yolov5/weightsstock/last_0.99_rest1_20210107.pt', '--SAVE_PATH', SAVE_DIRECTORY_PATH ,'--img', '416', '--conf', '0.60', '--source', SOURCE_PATH, '--save-txt'])
# proc.wait()
processes.append((8, proc))
#clef
SAVE_DIRECTORY_PATH = os.path.dirname(FILE_PATH) + '/staff2/clef'
proc = subprocess.Popen(['python','detect.py', '--weights', '/Users/tomoimacpro/TSh_project/bfaaap/yolov5/weightsstock/last_0.99_Clef3_20210129.pt', '--SAVE_PATH', SAVE_DIRECTORY_PATH ,'--img', '416', '--conf', '0.60', '--source', SOURCE_PATH, '--save-txt'])
# proc.wait()
processes.append((9, proc))

for i, p in processes:
    print(f'waiting process {i} to finish')
    p.wait()

#時間計測end
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
