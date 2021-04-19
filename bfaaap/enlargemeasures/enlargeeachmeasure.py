# coding: UTF-8
""" Enlarge Each Measure
    To enlarge each measure obtained (defined) in Yolo labeling
"""
import cv2
import glob
import os
import shutil
import copy


def produce_enlargedmeasures(img, txtlines, upper_margin, lower_margin):#upper margin: upper magnification; lower margin: lower magnification
    # to return each resized measure image
    return_images = []
    # image_height, image_width
    img_height, img_width = img.shape[:2]
    
    # To obtain the center position (x, y), width, and height of each target using each textline
    for textline in txtlines:
        target_info = textline.split() #target_info =[label, x, y, w, h]
        # set a processed area roi(left(x1), top(y1), right(x2), bottom(y2))
        left = int((float(target_info[1]) - float(target_info[3]) / 2)*img_width)
        top = int((float(target_info[2]) - float(target_info[4]) / 2)*img_height)
        right =int((float(target_info[1]) + float(target_info[3]) / 2)*img_width)
        bottom = int((float(target_info[2]) + float(target_info[4]) / 2)*img_height)
        width = right - left
        height = bottom - top
        print('widthは{}、heightは{}'.format(width, height))
        #add upper and lower margins to select roi (region of interest)
        mod_top = top - int(upper_margin * height)
        mod_bottom = bottom + int(lower_margin * height)
        if mod_top < 0:
            mod_top = 0
        if mod_bottom > img_height:
            mod_bottom = img_height
        roi = (left, mod_top, right, mod_bottom)
        # To select the roi in the img
        # [top:bottom, left:right] 
        s_roi = img[roi[1]: roi[3], roi[0]: roi[2]]
        #resize s_roi (img) to 416 x 416 (Yolov5 input size)
        dim = (416, 416)#dim = (width, height)
        resized_measure_img = cv2.resize(s_roi, dim, interpolation = cv2.INTER_AREA)
        #save each resized measure image in return_images = [resized_measure_image]
        return_images.append(resized_measure_img)
    return return_images




def enlarge_eachmeasure_in_eachstaff(FILE_PATH, pairedStaff=True, upper_margin=1.0, lower_margin=1.0):
    files_parent = glob.glob(FILE_PATH) #provide PATH of an original sheet music image at the base location
    for file_parent in files_parent:
       # dirname/staff/labels/, basename + ext(.txt)
        dirname = os.path.dirname(file_parent) + '/staff/labels/'
        namewithoutext = os.path.splitext(os.path.basename(file_parent))[0]
        files_DIR = dirname + '/' + '*' 

        files = glob.glob(files_DIR) #"./tmp/*":beforehand prepare images and Yolov5 anotation files in ./tmp/subdirectory
        #To skip .txt files
        for file in files:
            if file.endswith('txt'):
                print('There is a text file: {}'.format(file))
            else:
                #To prepare each cv2.img and textlines from the corresponding .txt file
                img = cv2.imread(file)
                # To get the corresponding .txt file
                # dirname, basename + ext(.txt)
                dirname = os.path.dirname(file)
                namewithoutext = os.path.splitext(os.path.basename(file))[0]
                image_ext = os.path.splitext(os.path.basename(file))[1]
                txtfile = dirname + '/' + namewithoutext + '.txt'
                # To obtain textlines
                txtlines = []
                with open(txtfile) as f:
                    txtlines = f.readlines()
                
                #Please provide here with your desired augmentation   
                measure_images = produce_enlargedmeasures(img, txtlines, upper_margin=upper_margin, lower_margin=lower_margin)
                print(f'measure_imagesの個数は{len(measure_images)}です。')
                #save each resized measure as a file
                for i, resized_measure_image in enumerate(measure_images):
                    cv2.imwrite(dirname + '/../../measure/' + namewithoutext + '_measure#' + '{:0=3}'.format(i) + image_ext, resized_measure_image)

# enlarge_eachmeasure_in_eachfile(upper_margin=1.0, lower_margin=1.0)


def giveResizedMeasureImage(measureOfInterest, img_input, img_width, img_height, upper_margin, lower_margin):
    eachmeasure = copy.copy(measureOfInterest)
    img = copy.copy(img_input)
    # set a processed area roi(left(x1), top(y1), right(x2), bottom(y2))
    left = int(eachmeasure['left']*img_width)
    top = int(eachmeasure['top']*img_height)
    right =int(eachmeasure['right']*img_width)
    bottom = int(eachmeasure['bottom']*img_height)
    width = right - left
    height = bottom - top
    print('eachmeasureのwidthは{}、heightは{}'.format(width, height))
    #add upper and lower margins to select roi (region of interest)
    mod_top = top - int(upper_margin * height)
    mod_bottom = bottom + int(lower_margin * height)
    if mod_top < 0:
        mod_top = 0
    if mod_bottom > img_height:
        mod_bottom = img_height
    roi = (left, mod_top, right, mod_bottom)
    # To select the roi in the img
    # [top:bottom, left:right] 
    s_roi = img[roi[1]: roi[3], roi[0]: roi[2]]
    #resize s_roi (img) to 416 x 416 (Yolov5 input size)
    dim = (416, 416)#dim = (width, height)
    resized_measure_img = cv2.resize(s_roi, dim, interpolation = cv2.INTER_AREA)
    return resized_measure_img





def produceResizedMeasuresFromAlignedStaves(img_FILE_PATH, aligned_staves, isPaired=True, upper_margin=1.0, lower_margin=1.0):#upper margin: upper magnification; lower margin: lower magnification
    
    # to return each resized measure image
    return_resizedimages_for_staff1 = []
    return_resizedimages_for_staff2 = []

    measures_in_staff1 = []
    measures_in_staff2 = []

    #assign each measure to staff1 or staff2; staff1 as default
    if isPaired and len(aligned_staves) > 1:
        for i, eachstaff in enumerate(aligned_staves):
            if i % 2 == 0:
                for eachmeasure in eachstaff:
                    measures_in_staff1.append(eachmeasure)
            else:
                for eachmeasure in eachstaff:
                    measures_in_staff2.append(eachmeasure)
    else:
        for eachstaff in aligned_staves:
            for eachmeasure in eachstaff:
                    measures_in_staff1.append(eachmeasure)
    # print(f'measures_in_staff2の個数は{len(measures_in_staff2)}')
    #resize each measure image in the image file
    files = glob.glob(img_FILE_PATH) #as an original sheet music image
    for file in files:
        if file.endswith('jpg') or file.endswith('png'):               
            # dirname, basename including an extention
            dirname = os.path.dirname(file)
            image_ext = os.path.splitext(os.path.basename(file))[1]
            # image_height, image_width
            img = cv2.imread(file)
            img_height, img_width = img.shape[:2]
            #resize each image for staff1 or, if any, staff2
            if len(measures_in_staff1) > 0 and isPaired:
                for eachmeasure in measures_in_staff1:
                    resized_measure_img = giveResizedMeasureImage(eachmeasure, img, img_width, img_height, upper_margin, lower_margin)
                    #save each resized measure image in return_images = [resized_measure_image]
                    return_resizedimages_for_staff1.append(resized_measure_img)
            if len(measures_in_staff2) > 0 and isPaired:
                for eachmeasure in measures_in_staff2:
                    resized_measure_img = giveResizedMeasureImage(eachmeasure, img, img_width, img_height, upper_margin, lower_margin)
                    #save each resized measure image in return_images = [resized_measure_image]
                    return_resizedimages_for_staff2.append(resized_measure_img)
            if len(measures_in_staff1) > 0 and not isPaired:
                    for eachmeasure in measures_in_staff1:
                        resized_measure_img = giveResizedMeasureImage(eachmeasure, img, img_width, img_height, upper_margin, lower_margin)
                        #save each resized measure image in return_images = [resized_measure_image]
                        return_resizedimages_for_staff1.append(resized_measure_img)
            #save the resized images under a directory ./measure/staff1/ or ./measure/staff2/
            new_dir_path_staff1 = dirname+ '/measure/staff1'
            new_dir_path_staff2 = dirname+ '/measure/staff2'
            os.makedirs(new_dir_path_staff1, exist_ok=True)
            os.makedirs(new_dir_path_staff2, exist_ok=True)
            if len(return_resizedimages_for_staff1) > 0:
                #save each resized measure as a file under a directory /measure/staff1/
                for i, resized_measure_image in enumerate(return_resizedimages_for_staff1):
                    cv2.imwrite(dirname + '/measure/staff1/' +  'measure#' + '{:0=3}'.format(i) + image_ext, resized_measure_image)
            if len(return_resizedimages_for_staff2) > 0:
                #save each resized measure as a file
                for i, resized_measure_image in enumerate(return_resizedimages_for_staff2):
                    # print(f'{i}個目のreturn_resizedimages_for_staff2')
                    cv2.imwrite(dirname + '/measure/staff2/' +  'measure#' + '{:0=3}'.format(i) + image_ext, resized_measure_image)





