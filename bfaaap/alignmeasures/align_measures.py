# coding: UTF-8

import glob
import os
import copy

#To obtain measures in a list [measures]
def collect_measures(txtlines): #sorted in the Y direction
    # To identify and collect each measure
    measures = [] # store measures as a list
    for textline in txtlines:
        measure_info = textline.split() #target_info =[label, x, y, w, h]
        # To obtain information on each measure
        label = measure_info[0]
        center_x = float(measure_info[1])
        center_y = float(measure_info[2])
        width = float(measure_info[3])
        height = float(measure_info[4])
        left = center_x - (width / 2)
        top = center_y - (height / 2)
        right = center_x + (width / 2)
        bottom = center_y + (height / 2)
        # To store the information as a dictionary
        each_measure = {"label":label, "center_x":center_x, "center_y":center_y, "width":width, "height":height, "left":left, "top":top, "right":right, "bottom":bottom}
        measures.append(each_measure)
    #To sort the measures in the Y direction
    measures_sorted = sorted(measures, key=lambda x:x['center_y'])
    return measures_sorted


def grouping_measures(measures): # to group the measures in the X direction
    measures_copy1 = copy.copy(measures)
    measures_copy2 = copy.copy(measures)
    
    staves = []#initialize staves
    for measure in measures_copy1:
        if (measure['label'] == '0') or (measure['label'] == '1'):
            # print('measureのlabel{}がありました'.format(measure['label']))
            #collect measures in a stave, the center_y position of which is between the top and the bottom of the measure (X0 or X1) of interest
            stave = [] # initialize a stave as an element of staves
            for candidatemeasure in measures_copy2:
               if  (measure['center_y'] - 0.02) <= candidatemeasure['center_y'] and candidatemeasure['center_y'] <= (measure['center_y'] + 0.02):
                    # print('candidateがありました。')
                    stave.append(candidatemeasure)
            #sort the resulting measures in the stave in the X-direction
            # print('得られたstaveは{}です。'.format(stave))
            stave_sorted = sorted(stave, key=lambda x:x['center_x'])
            staves.append(stave_sorted)
    return staves

def deleteOverlaps(staves_input):
    staves = copy.copy(staves_input)
    staves_tmp = copy.copy(staves_input)
    print('deleteOverlaps(staves_input)を通過しました。')
    # to remove overlapping x0 and x1
    for i, stave in enumerate(staves_tmp):
        if i == 0:
            continue
        else:
            if (staves_tmp[i - 1][0]['center_y'] - 0.01) < stave[0]['center_y'] and stave[0]['center_y'] < (staves_tmp[i - 1][0]['center_y'] + 0.01):
                print('there is an overlapping x0 or x1, so remove it')
                staves.remove(stave)
    # to remove one of overlapping y0 items
    for i, stave in enumerate(staves_tmp):
        if len(stave) > 1:
            for j, eachmeasure in enumerate(stave):
                if j == 0:
                    continue
                else:
                    if abs(staves_tmp[i][j-1]['left'] - staves_tmp[i][j]['left']) < 0.02:
                        del staves[i][j]

    return staves



def generate_measures_in_eachstave_aslist(FILE_PATH):
    # collect .txt files (inferred Yolov5 text filed (.txt)) and extract txtlines from each file
    
    files_parent = glob.glob(FILE_PATH) #provide PATH of an original sheet music image at the base location
    for file_parent in files_parent:
       # dirname/staff/labels/, basename + ext(.txt)
        dirname = os.path.dirname(file_parent) + '/staff/labels/'
        namewithoutext = os.path.splitext(os.path.basename(file_parent))[0]
        txtfile_PATH = dirname + '/' + namewithoutext + '.txt' 

        files_txt = glob.glob(txtfile_PATH)
        for txtfile in files_txt:
            # To obtain textlines
            txtlines = []
            with open(txtfile) as f:
                txtlines = f.readlines()
            measures = collect_measures(txtlines=txtlines)
            # print(measures)
            staves = grouping_measures(measures)
            # print('得られたstavesを表示します。')
            # print(staves)
            # print('stavesの個数は：{}個です。'.format(len(staves)))
            adjusted_staves = deleteOverlaps(staves)
            count_staves = 0
            print(f'adjusted_stavesのstaffの個数は{len(adjusted_staves)}')
            for i, stave in enumerate(adjusted_staves):
                print(f'{i}番目のstaff中のmeasureの個数は{len(stave)}')
                center_y = stave[0]['center_y']
                print(f'{i}番目のstaffのcenter_yは：{center_y}')
                
    return adjusted_staves
