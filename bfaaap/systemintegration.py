# coding: UTF-8

import cv2
import glob
import os
import shutil
import numpy as np
import math
from enum import Enum, auto
import copy
from itertools import chain

#perform inference on sheet music

#After the measure-recognizing model is applied to a piece of sheet music

#copy and move the relevant files under a dirctory ./musicdata/AAA/(staff/labels)
#sheet music (.jpg) provided in FILE_PATH
FILE_PATH = '/Users/tomoimacpro/TSh_project/bfaaap/musicdata/emisan/leveled_sarabandePhotoInclinedのコピー.jpg'


files_temp = glob.glob(FILE_PATH) #"./tmp/*":beforehand prepare images and Yolov5 anotation files in ./tmp/subdirectory
#To skip .txt files
FILE_DIR_PATH = ''
FILE_BASENAME = ''
FILE_BASENAME_WITHOUTEXT = ''
for file_temp in files_temp:
    if file_temp.endswith('jpg') or file_temp.endswith('png'):
        img = cv2.imread(file_temp)
        FILE_DIR_PATH = os.path.dirname(file_temp)
        FILE_BASENAME = os.path.basename(file_temp)
        FILE_BASENAME_WITHOUTEXT = os.path.splitext(FILE_BASENAME)[0]
        cv2.imwrite(FILE_DIR_PATH + '/staff/labels/' + FILE_BASENAME, img)



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
   
"""
#excise and enlarge each measure img at 412 x 412 pixels in each staff and stored in musicdata/AAA/measure/staff1/ or staff2/
from enlargemeasures.enlargeeachmeasure import produceResizedMeasuresFromAlignedStaves

staff_magnification = 1.2

produceResizedMeasuresFromAlignedStaves(img_FILE_PATH=FILE_PATH, aligned_staves=staves_with_measures_in_sheetmusic, isPaired=areStavesPaired, upper_margin=staff_magnification, lower_margin=staff_magnification)

#perform inference about each image in staff1 or staff2 by using models (e.g., body, armbeam, accidental, rest)
"""

"""generate ms sequence in each staff"""
# print(f'staves_with_measures_in_sheetmusicは\n{staves_with_measures_in_sheetmusic}')
from makeyolomusicdict.generatedictforxml import give_all_ms_in_eachmeasure_for_staff1or2

all_ms_in_eachmeasure_staff1, all_ms_in_eachmeasure_staff2 = give_all_ms_in_eachmeasure_for_staff1or2(isPaired=areStavesPaired, aligned_staves_input=staves_with_measures_in_sheetmusic, img_FILE_PATH=FILE_PATH)
print(f'all_ms_in_eachmeasure_staff1の要素数は{len(all_ms_in_eachmeasure_staff1)}')
aaa1 = all_ms_in_eachmeasure_staff1['measure#001']
print(f'aaa1は{aaa1}')
print(f'all_ms_in_eachmeasure_staff2の要素数は{len(all_ms_in_eachmeasure_staff2)}')
bbb1 = all_ms_in_eachmeasure_staff2['measure#001']
print(f'bbb1は{bbb1}')
# print(f'all_ms_in_eachmeasure_staff1は\n{all_ms_in_eachmeasure_staff1}')


#input base data for sheet music:
tempo = 60 #public data
beats = 3 #public data
beat_type = 2 #public data
preset_measure_duration = 1024 * beats / beat_type #public data
fifths = -1 #public data
staff = 1

isWideStaff = False
# #provide determined alpha and beta
# alpha = 0.024
# beta = 0.0


from makeyolomusicdict.generatedictforxml import setCurrentAccidentalTable, generateMSsequenceForStaff1or2, Clef

#classはimportすること
current_clef = Clef.G
current_accidental_table_template ={'A':'', 'B':'', 'C':'', 'D':'', 'E':'', 'F':'', 'G':''}
current_accidental_table = setCurrentAccidentalTable(current_accidental_table_template, fifths)
ms_sequenceOfInterest_staff1 = generateMSsequenceForStaff1or2(all_ms_in_eachmeasure_input=all_ms_in_eachmeasure_staff1, current_accidental_table_input=current_accidental_table, staff=1, current_clef_input=current_clef, preset_measure_duration=preset_measure_duration, FILE_PATH=FILE_PATH, isWideStaff=isWideStaff)

#for staff2: check current_clef
current_clef = Clef.F
ms_sequenceOfInterest_staff2 = generateMSsequenceForStaff1or2(all_ms_in_eachmeasure_input=all_ms_in_eachmeasure_staff2, current_accidental_table_input=current_accidental_table, staff=2, current_clef_input=current_clef, preset_measure_duration=preset_measure_duration, FILE_PATH=FILE_PATH, isWideStaff=isWideStaff)

print(f'staves_with_measures_in_sheetmusicのstave数は{len(staves_with_measures_in_sheetmusic)}')
for i, each_staff in enumerate(staves_with_measures_in_sheetmusic):
    print(f'{i}番目の小節(measure)の数は{len(each_staff)}個です。')

print(f'all_ms_in_eachmeasure_staff1の要素数は{len(all_ms_in_eachmeasure_staff1)}')
print(f'all_ms_in_eachmeasure_staff2の要素数は{len(all_ms_in_eachmeasure_staff2)}')

print(f'ms_sequenceOfInterest_staff1の要素数は{len(ms_sequenceOfInterest_staff1)}')
print(f'ms_sequenceOfInterest_staff2の要素数は{len(ms_sequenceOfInterest_staff2)}')

# print(f'ms_sequenceOfInterest_staff1の[6:8]は\n{ms_sequenceOfInterest_staff1[6:8]}')

"""
generate a dictionary for ET

"""
from makeyolomusicdict.generatedictforxml import generateDictForET_singlestaff, generateDictForET_twostaves

#for staff1
current_staff1_clef = Clef.G
dictionary_for_ET_staff1 = generateDictForET_singlestaff(ms_sequenceOfInterest_staff_input=ms_sequenceOfInterest_staff1, tempo=tempo, beats=beats, beat_type=beat_type, fifths=fifths, clef=current_staff1_clef)
part_content1 = dictionary_for_ET_staff1['part']
print(f'dictionary_for_ET_staff1[0]の要素の数は\n{len(part_content1)}')
#for staff2
current_staff2_clef = Clef.F
dictionary_for_ET_staff2 = generateDictForET_singlestaff(ms_sequenceOfInterest_staff_input=ms_sequenceOfInterest_staff2, tempo=tempo, beats=beats, beat_type=beat_type, fifths=fifths, clef=current_staff2_clef)
part_content2 = dictionary_for_ET_staff2['part']
print(f'dictionary_for_ET_staff2[0]の要素の数は\n{len(part_content2)}')
# print(f'part_content1とpart_contentは同じですか{part_content1 == part_content2}')

print(f'current_staff1_clef:{current_staff1_clef}\ncurrent_staff2_clef:{current_staff2_clef}')
#for both staff1 and staff2
dictionary_for_ET_staves1and2 = generateDictForET_twostaves(ms_sequence_staff1_input=ms_sequenceOfInterest_staff1, ms_sequence_staff2_input=ms_sequenceOfInterest_staff2, tempo=tempo, beats=beats, beat_type=beat_type, fifths=fifths, clef_staff1_input=current_staff1_clef, clef_staff2_input=current_staff2_clef)
part_content1and2 = dictionary_for_ET_staves1and2['part']
print(f'dictionary_for_ET_staves1and2[0]の要素の数は\n{len(part_content1and2)}')

#generate XML
from yoloToxml.yoloToxml import musicData2XML
import xml.etree.ElementTree as ET
from xml.dom import minidom

"""
#for staff1
"""
part_et = ET.Element('part')
part_et.attrib = {'id':'P1'}
part_et_1 = musicData2XML(part_et, dictionary_for_ET_staff1)

xmlstr_1 = minidom.parseString(ET.tostring(part_et_1)).toprettyxml(indent="   ")

#1行目の<?xml version="1.0"　?>を除く：　　結合するため
xmlstr_1 = xmlstr_1[23:]
# print(xmlstr_1)
# print(f'XML化した結果データは{xmlstr_1}')                

#template.xmlファイルを読み込んで得られたpart_etのXMLデータと結合して全体XMLを作る
wholeXML_staff1_text = ""
with open("/Users/tomoimacpro/TSh_project/bfaaap/yoloToxml/template.xml", 'r') as f:
    template_text = f.read()
    wholeXML_staff1_text = template_text +'\n' + xmlstr_1 +'\n</score-partwise>'

#save the resulting xml in ./xml/ directory
FILE_DIR_PATH
new_dir_path = FILE_DIR_PATH + '/xml'
os.makedirs(new_dir_path, exist_ok=True)
new_xml_filepath = new_dir_path + '/' + FILE_BASENAME_WITHOUTEXT + '_staff1.xml'
with open(new_xml_filepath, 'w') as f:
    f.write(wholeXML_staff1_text)

"""
#for staff2
"""
part_et = ET.Element('part')
part_et.attrib = {'id':'P1'}
part_et_2 = musicData2XML(part_et, dictionary_for_ET_staff2)

xmlstr_2 = minidom.parseString(ET.tostring(part_et_2)).toprettyxml(indent="   ")

#1行目の<?xml version="1.0"　?>を除く：　　結合するため
xmlstr_2 = xmlstr_2[23:]
# print(xmlstr_1)
# print(f'XML化した結果データは{xmlstr_1}')                

#template.xmlファイルを読み込んで得られたpart_etのXMLデータと結合して全体XMLを作る
wholeXML_staff2_text = ""
with open("/Users/tomoimacpro/TSh_project/bfaaap/yoloToxml/template.xml", 'r') as f:
    template_text = f.read()
    wholeXML_staff2_text = template_text +'\n' + xmlstr_2 +'\n</score-partwise>'

#save the resulting xml in ./xml/ directory
FILE_DIR_PATH
new_dir_path = FILE_DIR_PATH + '/xml'
os.makedirs(new_dir_path, exist_ok=True)
new_xml_filepath = new_dir_path + '/' + FILE_BASENAME_WITHOUTEXT + '_staff2.xml'
with open(new_xml_filepath, 'w') as f:
    f.write(wholeXML_staff2_text)


"""
#for both staff1 and staff2
"""
part_et = ET.Element('part')
part_et.attrib = {'id':'P1'}
part_et_1and2 = musicData2XML(part_et, dictionary_for_ET_staves1and2)

xmlstr_1and2 = minidom.parseString(ET.tostring(part_et_1and2)).toprettyxml(indent="   ")

#1行目の<?xml version="1.0"　?>を除く：　　結合するため
xmlstr_1and2 = xmlstr_1and2[23:]
# print(xmlstr_1)
# print(f'XML化した結果データは{xmlstr_1}')                

#template.xmlファイルを読み込んで得られたpart_etのXMLデータと結合して全体XMLを作る
wholeXML_staves1and2_text = ""
with open("/Users/tomoimacpro/TSh_project/bfaaap/yoloToxml/template.xml", 'r') as f:
    template_text = f.read()
    wholeXML_staves1and2_text = template_text +'\n' + xmlstr_1and2 +'\n</score-partwise>'

#save the resulting xml in ./xml/ directory
FILE_DIR_PATH
new_dir_path = FILE_DIR_PATH + '/xml'
os.makedirs(new_dir_path, exist_ok=True)
new_xml_filepath = new_dir_path + '/' + FILE_BASENAME_WITHOUTEXT + '_staves1and2.xml'
with open(new_xml_filepath, 'w') as f:
    f.write(wholeXML_staves1and2_text)


