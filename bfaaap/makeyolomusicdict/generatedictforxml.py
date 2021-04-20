# coding: UTF-8
""" Yolov5 to musicXML generation
    To generate an musicXML file from musical symbols obtained by multiple Yolov5 models
"""

import cv2
import glob
import os
import shutil
import numpy as np
import math
from enum import Enum, auto
import copy
from itertools import chain

# Scale definition
"""
G-clef, F-clef (8va, 8vb)
"""

class Clef(Enum):
    G = auto()
    F = auto()
    G8va = auto()
    F8vb = auto()
    none = auto()

    def __eq__(self, other):
        if self.name == other.name:
            return True
        return False


"""#label#
body: bd0 (black), bd1 (black dot), bd2 (white), bd3 (white dot), bd4 (large white), bd5 (large white dot)
armbeam: am0 (upper stick), am1 (lower stick), am2 (upper flag), am3 (lower flag), bm0 (upper single beam), bm1 (lower single beam), bm2 (upper double beam), bm3 (lower double beam)
accidental: ac0 (#), ac1 (b), ac2 (natural) 
rest: re0 (whole), re1 (half), re2 (quarter), re3 (eighth), re4 (16th)
clef: cf0 (clef G), cf1 (clef F), cf2 (clef 8 (8va or 8vb), cf3 (clef 8va_stop), cf4( 8vb_stop):attention! changed
"""
class Category(Enum):
    bd0 = auto()
    bd1 = auto()
    bd2 = auto()
    bd3 = auto()
    bd4 = auto()
    bd5 = auto()
    am0 = auto()
    am1 = auto()
    am2 = auto()
    am3 = auto()
    bm0 = auto()
    bm1 = auto()
    bm2 = auto()
    bm3 = auto()
    ac0 = auto()
    ac1 = auto()
    ac2 = auto()
    re0 = auto()
    re1 = auto()
    re2 = auto()
    re3 = auto()
    re4 = auto()
    cf0 = auto()
    cf1 = auto()
    cf2 = auto()
    cf3 = auto()
    cf4 = auto()
    none = auto()

    def __eq__(self, other):
        if self.name == other.name:
            return True
        return False

class MeasureType(Enum):
    x0 = auto()
    x1 = auto()
    y0 = auto()
    none = auto()

    def __eq__(self, other):
        if self.name == other.name:
            return True
        return False


# Musical symbol anotation
musicalSymbol_template = {'measuretype':MeasureType.none, 'category':Category.none, 'x':0., 'y':0., 'w':0., 'h':0., 'alpha':0., 'beta':0.,
    'step':'', 'alter':'', 'octave':'', 'duration':0, 'voice':1, 'type':'', 'dot':False, 'stem':'', 'staff':1,
    'beam_number':0, 'beam_content':'', 'chord':False, 'rest':False, 'clef':Clef.none, 'fifths':0, 'octave_shift':'', 'clefchange':False}


#set the clef positions
def set_staffmiddle_heightInterval_for_notePosition(alpha, beta):
    staffmiddle = 0.5 + alpha
    heightInterval =  1.0/(8 +2*(8*1.2)) + beta
    notePositionInClef_G = {'D3':(staffmiddle + heightInterval*12.0), 'E3':(staffmiddle + heightInterval*11.0), 'F3':(staffmiddle + heightInterval*10.0), 'G3':(staffmiddle + heightInterval*9.0), 'A3':(staffmiddle + heightInterval*8.0), 'B3':(staffmiddle + heightInterval*7.0),
    'C4':(staffmiddle + heightInterval*6.0), 'D4':(staffmiddle + heightInterval*5.0), 'E4':(staffmiddle + heightInterval*4.0), 'F4':(staffmiddle + heightInterval*3.0), 'G4':(staffmiddle + heightInterval*2.0), 'A4':(staffmiddle + heightInterval*1.0), 'B4':(staffmiddle),
    'C5':(staffmiddle - heightInterval*1.0), 'D5':(staffmiddle - heightInterval*2.0), 'E5':(staffmiddle - heightInterval*3.0), 'F5':(staffmiddle - heightInterval*4.0), 'G5':(staffmiddle - heightInterval*5.0), 'A5':(staffmiddle - heightInterval*6.0), 'B5':(staffmiddle - heightInterval*7.0),
    'C6':(staffmiddle - heightInterval*8.0), 'D6':(staffmiddle - heightInterval*9.0), 'E6':(staffmiddle - heightInterval*10.0),'F6':(staffmiddle - heightInterval*11.0), 'G6':(staffmiddle - heightInterval*12.0),
    }
    notePositionInClef_F = {'F1':(staffmiddle + heightInterval*12.0), 'G1':(staffmiddle + heightInterval*11.0), 'A1':(staffmiddle + heightInterval*10.0), 'B1':(staffmiddle + heightInterval*9.0), 'C2':(staffmiddle + heightInterval*8.0), 'D2':(staffmiddle + heightInterval*7.0),
    'E2':(staffmiddle + heightInterval*6.0), 'F2':(staffmiddle + heightInterval*5.0), 'G2':(staffmiddle + heightInterval*4.0), 'A2':(staffmiddle + heightInterval*3.0), 'B2':(staffmiddle + heightInterval*2.0), 'C3':(staffmiddle + heightInterval*1.0), 'D3':(staffmiddle),
    'E3':(staffmiddle - heightInterval*1.0), 'F3':(staffmiddle - heightInterval*2.0), 'G3':(staffmiddle - heightInterval*3.0), 'A3':(staffmiddle - heightInterval*4.0), 'B3':(staffmiddle - heightInterval*5.0), 'C4':(staffmiddle - heightInterval*6.0), 'D4':(staffmiddle - heightInterval*7.0),
    'E4':(staffmiddle - heightInterval*8.0), 'F4':(staffmiddle - heightInterval*9.0), 'G4':(staffmiddle - heightInterval*10.0),'A4':(staffmiddle - heightInterval*11.0), 'B4':(staffmiddle - heightInterval*12.0),
    }
    notePositionInClef_G8va = {'D4':(staffmiddle + heightInterval*12.0), 'E4':(staffmiddle + heightInterval*11.0), 'F4':(staffmiddle + heightInterval*10.0), 'G4':(staffmiddle + heightInterval*9.0), 'A4':(staffmiddle + heightInterval*8.0), 'B4':(staffmiddle + heightInterval*7.0),
    'C5':(staffmiddle + heightInterval*6.0), 'D5':(staffmiddle + heightInterval*5.0), 'E5':(staffmiddle + heightInterval*4.0), 'F5':(staffmiddle + heightInterval*3.0), 'G5':(staffmiddle + heightInterval*2.0), 'A5':(staffmiddle + heightInterval*1.0), 'B5':(staffmiddle),
    'C6':(staffmiddle - heightInterval*1.0), 'D6':(staffmiddle - heightInterval*2.0), 'E6':(staffmiddle - heightInterval*3.0), 'F6':(staffmiddle - heightInterval*4.0), 'G6':(staffmiddle - heightInterval*5.0), 'A6':(staffmiddle - heightInterval*6.0), 'B6':(staffmiddle - heightInterval*7.0),
    'C7':(staffmiddle - heightInterval*8.0), 'D7':(staffmiddle - heightInterval*9.0), 'E7':(staffmiddle - heightInterval*10.0),'F7':(staffmiddle - heightInterval*11.0), 'G7':(staffmiddle - heightInterval*12.0),
    }
    notePositionInClef_F8vb = {'F0':(staffmiddle + heightInterval*12.0), 'G0':(staffmiddle + heightInterval*11.0), 'A0':(staffmiddle + heightInterval*10.0), 'B0':(staffmiddle + heightInterval*9.0), 'C1':(staffmiddle + heightInterval*8.0), 'D1':(staffmiddle + heightInterval*7.0),
    'E1':(staffmiddle + heightInterval*6.0), 'F1':(staffmiddle + heightInterval*5.0), 'G1':(staffmiddle + heightInterval*4.0), 'A1':(staffmiddle + heightInterval*3.0), 'B1':(staffmiddle + heightInterval*2.0), 'C2':(staffmiddle + heightInterval*1.0), 'D2':(staffmiddle),
    'E2':(staffmiddle - heightInterval*1.0), 'F2':(staffmiddle - heightInterval*2.0), 'G2':(staffmiddle - heightInterval*3.0), 'A2':(staffmiddle - heightInterval*4.0), 'B2':(staffmiddle - heightInterval*5.0), 'C3':(staffmiddle - heightInterval*6.0), 'D3':(staffmiddle - heightInterval*7.0),
    'E3':(staffmiddle - heightInterval*8.0), 'F3':(staffmiddle - heightInterval*9.0), 'G3':(staffmiddle - heightInterval*10.0),'A3':(staffmiddle - heightInterval*11.0), 'B3':(staffmiddle - heightInterval*12.0),
    }
    return staffmiddle, heightInterval, notePositionInClef_G, notePositionInClef_F, notePositionInClef_G8va, notePositionInClef_F8vb

# To determine the positional relationship between musical symbols

def isHorizontallyOverlapping(ms1, ms2):
    #ms1 (musical symbol 1 (ms1) is horizontally overlapping with ms2?)
    #check and compare the vertical positions of ms1 and ms2
    if (ms1["y"] < ms2["y"]):
        if (ms1["y"] + ms1["h"]*0.5) >= (ms2["y"] - ms2["h"]*0.5):
            return True
        else:
            return False
    else:
        if (ms1["y"] - ms1["h"]*0.5) <= (ms2["y"] + ms2["h"]*0.5):
            return True
        else:
            return False 

def isVerticallyOverlapping(ms1, ms2):
    #ms1 (musical symbol 1 (ms1) is vertically overlapping with ms2?)
    #check and compare the horizontal positions of ms1 and ms2
    if (ms1["x"] < ms2["x"]):
        if (ms1["x"] + ms1["w"]*0.5) >= (ms2["x"] - ms2["w"]*0.5):
            return True
        else:
            return False
    else:
        if (ms1["x"] - ms1["w"]*0.5) <= (ms2["x"] + ms2["w"]*0.5):
            return True
        else:
            return False

def isContacting(ms1, ms2):
     #Is ms1 (musical symbol 1 (ms1) in contact with ms2?)
     #first check whether they are horizontally overlapping
     #second check whether they are vertically overlapping
    if isHorizontallyOverlapping(ms1, ms2) and isVerticallyOverlapping(ms1, ms2):
        return True
    else:
        return False 
#musicalSymbol (ms) arrangement
ms_all = []# ms_all = [ms1, ms2, ms3, ms4, ms5, ms6, ms7, ms8, ms9]

#horizontal sorting (x direction: increasing (ascending) ->)
def ms_horizontalsorting(ms_group):
    #ms_group should contain multiple mucicalSymbols to be sorted in the x direction
    return sorted(ms_group, key=lambda x: x['x'])

#vertical sorting (y direction: decreasing (descending))
def ms_verticallysorting(ms_group):
    #ms_group should contain multiple mucicalSymbols to be sorted in the x direction
    return sorted(ms_group, key=lambda x: x['y'], reverse=True)

#collect ms items in contact with ms1: (ms: musical symboles)
def collectContactingMusicalSymbols(ms_all, msOfInterest):
    ms_group = []
    for each_ms in ms_all:
        if isContacting(each_ms, msOfInterest):
            ms_group.append(each_ms)
    return ms_group

#collect vertically overlapping ms items in an ms group
def collectVerticallyOverlappingDescendingMusicalSymbols(ms_all, msOfInterest):
    verticallyOverlappingDescendingMSGroup = []
    for eachms in ms_all:
        if isVerticallyOverlapping(msOfInterest, eachms):
            verticallyOverlappingDescendingMSGroup.append(eachms)
    verticallyOverlappingDescendingMSGroup = ms_verticallysorting(verticallyOverlappingDescendingMSGroup)
    return verticallyOverlappingDescendingMSGroup



"""
To determine an accidental
prepare a accidental table for reference

"""
current_accidental_table_template ={'A':'', 'B':'', 'C':'', 'D':'', 'E':'', 'F':'', 'G':''}
fifths = 0# should be preset.
def setCurrentAccidentalTable(current_accidental_table, fifths):
    if fifths == 0:
        current_accidental_table = copy.copy(current_accidental_table_template)
    elif fifths == -1:
        current_accidental_table = copy.copy(current_accidental_table_template)
        current_accidental_table['B'] = 'b'
    elif fifths == -2:
        current_accidental_table = copy.copy(current_accidental_table_template)
        current_accidental_table['B'] = 'b'
        current_accidental_table['E'] = 'b'
    elif fifths == -3:
        current_accidental_table = copy.copy(current_accidental_table_template)
        current_accidental_table['B'] = 'b'
        current_accidental_table['E'] = 'b'
        current_accidental_table['A'] = 'b'
    elif fifths == -4:
        current_accidental_table = copy.copy(current_accidental_table_template)
        current_accidental_table['B'] = 'b'
        current_accidental_table['E'] = 'b'
        current_accidental_table['A'] = 'b'
        current_accidental_table['D'] = 'b'
    elif fifths == -5:
        current_accidental_table = copy.copy(current_accidental_table_template)
        current_accidental_table['B'] = 'b'
        current_accidental_table['E'] = 'b'
        current_accidental_table['A'] = 'b'
        current_accidental_table['D'] = 'b'
        current_accidental_table['G'] = 'b'
    elif fifths == -6:
        current_accidental_table = copy.copy(current_accidental_table_template)
        current_accidental_table['B'] = 'b'
        current_accidental_table['E'] = 'b'
        current_accidental_table['A'] = 'b'
        current_accidental_table['D'] = 'b'
        current_accidental_table['G'] = 'b'
        current_accidental_table['C'] = 'b'
    elif fifths == -7:
        current_accidental_table = copy.copy(current_accidental_table_template)
        current_accidental_table['B'] = 'b'
        current_accidental_table['E'] = 'b'
        current_accidental_table['A'] = 'b'
        current_accidental_table['D'] = 'b'
        current_accidental_table['G'] = 'b'
        current_accidental_table['C'] = 'b'
        current_accidental_table['F'] = 'b'
    elif fifths == 1:
        current_accidental_table = copy.copy(current_accidental_table_template)
        current_accidental_table['F'] = '#'
    elif fifths == 2:
        current_accidental_table = copy.copy(current_accidental_table_template)
        current_accidental_table['F'] = '#'
        current_accidental_table['C'] = '#'
    elif fifths == 3:
        current_accidental_table = copy.copy(current_accidental_table_template)
        current_accidental_table['F'] = '#'
        current_accidental_table['C'] = '#'
        current_accidental_table['G'] = '#'
    elif fifths == 4:
        current_accidental_table = copy.copy(current_accidental_table_template)
        current_accidental_table['F'] = '#'
        current_accidental_table['C'] = '#'
        current_accidental_table['G'] = '#'
        current_accidental_table['D'] = '#'
    elif fifths == 5:
        current_accidental_table = copy.copy(current_accidental_table_template)
        current_accidental_table['F'] = '#'
        current_accidental_table['C'] = '#'
        current_accidental_table['G'] = '#'
        current_accidental_table['D'] = '#'
        current_accidental_table['A'] = '#'
    elif fifths == 6:
        current_accidental_table = copy.copy(current_accidental_table_template)
        current_accidental_table['F'] = '#'
        current_accidental_table['C'] = '#'
        current_accidental_table['G'] = '#'
        current_accidental_table['D'] = '#'
        current_accidental_table['A'] = '#'
        current_accidental_table['E'] = '#'
    elif fifths == 7:
        current_accidental_table = copy.copy(current_accidental_table_template)
        current_accidental_table['F'] = '#'
        current_accidental_table['C'] = '#'
        current_accidental_table['G'] = '#'
        current_accidental_table['D'] = '#'
        current_accidental_table['A'] = '#'
        current_accidental_table['E'] = '#'
        current_accidental_table['B'] = '#'
    else:
        current_accidental_table = copy.copy(current_accidental_table_template)
    return current_accidental_table

def changeAccidentalTable(current_accidental_table_input, ms1, clef_temp, alpha, beta):# called if the musicalSymbol is an ac class (e.g., ac0(#), ac1(b), ac2(natural))
    #ms1 (musicalSymbol) should have an accidental label (please check before the call)
    #extract each combination of note and middle height and compare the height to that of ms1
    #change a current_accidental_table
    current_accidental_table = copy.copy(current_accidental_table_input)
    # print(f'changeAccidentalTable関数にinputしたcurrent_accidental_table_inputは{current_accidental_table_input}\n関数内のcurrent_accidental_tableは{current_accidental_table}')
    #set clef_temp to ms1
    ms1['clef'] = clef_temp
    
    #set clef position with alpha and beta
    staffmiddle, heightInterval, notePositionInClef_G, notePositionInClef_F, notePositionInClef_G8va, notePositionInClef_F8vb = set_staffmiddle_heightInterval_for_notePosition(alpha, beta)
    if ms1['clef'] == Clef.G:
        # print(f'clefは{Clef.G}')
        noteFlag = False
        for note, middleheight in notePositionInClef_G.items():
            # the case depends on the category
            if ms1["category"] == Category.ac0:
                if ((middleheight - heightInterval*0.5) < ms1["y"]) and (ms1["y"] <= (middleheight + heightInterval*0.5)):
                    noteFlag = True
                    pitchClass = note[0]
                    #add the accidental to ms1
                    current_accidental_table[pitchClass] = '#'
            elif ms1["category"] == Category.ac1:
                #judge the bottom(ms1["y"] + ms1["h"]*0.5) - heightInterval as the b position
                # print(f'middleheight - heightInterval*0.5  is {middleheight - heightInterval*0.5}')
                # print(f'(ms1["y"] + ms1["h"]*0.5 - heightInterval) is {(ms1["y"] + ms1["h"]*0.5 - heightInterval)}')
                if ((middleheight - heightInterval*0.5) < (ms1["y"] + ms1["h"]*0.5 - heightInterval - 0.01)) and ((ms1["y"] + ms1["h"]*0.5 - heightInterval -0.01) <= (middleheight + heightInterval*0.5)):
                    noteFlag = True
                    pitchClass = note[0]
                    print(f'pitchClass changed is {pitchClass}')
                    #add the accidental to ms1
                    current_accidental_table[pitchClass] = 'b'
            elif ms1["category"] == Category.ac2:
                 if ((middleheight - heightInterval*0.5) < ms1["y"]) and (ms1["y"] <= (middleheight + heightInterval*0.5)):
                    noteFlag = True
                    pitchClass = note[0]
                    #add the accidental to ms1
                    current_accidental_table[pitchClass] = ''
            else:
                print('there is no accidental')
        return current_accidental_table
        if noteFlag == False:
            return current_accidental_table
    elif ms1['clef'] == Clef.G8va:
        noteFlag = False
        for note, middleheight in notePositionInClef_G8va.items():
            # the case depends on the category
            if ms1["category"] == Category.ac0:
                if ((middleheight - heightInterval*0.5) < ms1["y"]) and (ms1["y"] <= (middleheight + heightInterval*0.5)):
                    noteFlag = True
                    pitchClass = note[0]
                    #add the accidental to ms1
                    current_accidental_table[pitchClass] = '#'
            elif ms1["category"] == Category.ac1:
                #judge the bottom(ms1["y"] + ms1["h"]*0.5) - heightInterval as the b position
                # print(f'middleheight - heightInterval*0.5  is {middleheight - heightInterval*0.5}')
                # print(f'(ms1["y"] + ms1["h"]*0.5 - heightInterval) is {(ms1["y"] + ms1["h"]*0.5 - heightInterval)}')
                if ((middleheight - heightInterval*0.5) < (ms1["y"] + ms1["h"]*0.5 - heightInterval - 0.01)) and ((ms1["y"] + ms1["h"]*0.5 - heightInterval - 0.01) <= (middleheight + heightInterval*0.5)):
                    pitchClass = note[0]
                    print(f'変更されるpitchClassは{pitchClass}')
                    #add the accidental to ms1
                    current_accidental_table[pitchClass] = 'b'
            elif ms1["category"] == Category.ac2:
                 if ((middleheight - heightInterval*0.5) < ms1["y"]) and (ms1["y"] <= (middleheight + heightInterval*0.5)):
                    noteFlag = True
                    pitchClass = note[0]
                    #add the accidental to ms1
                    current_accidental_table[pitchClass] = ''
            else:
                print('there is no accidental')
        return current_accidental_table
        if noteFlag == False:
            return current_accidental_table
    elif ms1['clef'] == Clef.F:
        noteFlag = False
        for note, middleheight in notePositionInClef_F.items():
            # the case depends on the category
            if ms1["category"] == Category.ac0:
                if ((middleheight - heightInterval*0.5) < ms1["y"]) and (ms1["y"] <= (middleheight + heightInterval*0.5)):
                    noteFlag = True
                    pitchClass = note[0]
                    #add the accidental to ms1
                    current_accidental_table[pitchClass] = '#'
            elif ms1["category"] == Category.ac1:
                #judge the bottom(ms1["y"] + ms1["h"]*0.5) - heightInterval as the b position
                # print(f'middleheight - heightInterval*0.5  is {middleheight - heightInterval*0.5}')
                # print(f'(ms1["y"] + ms1["h"]*0.5 - heightInterval) is {(ms1["y"] + ms1["h"]*0.5 - heightInterval)}')
                if ((middleheight - heightInterval*0.5) < (ms1["y"] + ms1["h"]*0.5 - heightInterval - 0.01)) and ((ms1["y"] + ms1["h"]*0.5 - heightInterval - 0.01) <= (middleheight + heightInterval*0.5)):
                    noteFlag = True
                    pitchClass = note[0]
                    print(f'pitchClass changed is {pitchClass}')
                    #add the accidental to ms1
                    current_accidental_table[pitchClass] = 'b'
            elif ms1["category"] == Category.ac2:
                 if ((middleheight - heightInterval*0.5) < ms1["y"]) and (ms1["y"] <= (middleheight + heightInterval*0.5)):
                    noteFlag = True
                    pitchClass = note[0]
                    #add the accidental to ms1
                    current_accidental_table[pitchClass] = ''
            else:
                print('there is no accidental')
        return current_accidental_table
        if noteFlag == False:
            return current_accidental_table
    elif ms1['clef'] == Clef.F8vb:
        noteFlag = False
        for note, middleheight in notePositionInClef_F8vb.items():
            # the case depends on the category
            if ms1["category"] == Category.ac0:
                if ((middleheight - heightInterval*0.5) < ms1["y"]) and (ms1["y"] <= (middleheight + heightInterval*0.5)):
                    noteFlag = True
                    pitchClass = note[0]
                    #add the accidental to ms1
                    current_accidental_table[pitchClass] = '#'
            elif ms1["category"] == Category.ac1:
                #judge the bottom(ms1["y"] + ms1["h"]*0.5) - heightInterval as the b position
                # print(f'middleheight - heightInterval*0.5  is {middleheight - heightInterval*0.5}')
                # print(f'(ms1["y"] + ms1["h"]*0.5 - heightInterval) is {(ms1["y"] + ms1["h"]*0.5 - heightInterval)}')
                if ((middleheight - heightInterval*0.5) < (ms1["y"] + ms1["h"]*0.5 - heightInterval - 0.01)) and ((ms1["y"] + ms1["h"]*0.5 - heightInterval - 0.01) <= (middleheight + heightInterval*0.5)):
                    noteFlag = True
                    pitchClass = note[0]
                    print(f'pitchClass changed is {pitchClass}')
                    #add the accidental to ms1
                    current_accidental_table[pitchClass] = 'b'
            elif ms1["category"] == Category.ac2:
                 if ((middleheight - heightInterval*0.5) < ms1["y"]) and (ms1["y"] <= (middleheight + heightInterval*0.5)):
                    noteFlag = True
                    pitchClass = note[0]
                    #add the accidental to ms1
                    current_accidental_table[pitchClass] = ''
            else:
                print('there is no accidental')
        return current_accidental_table
        if noteFlag == False:
            return current_accidental_table
    else:
        return current_accidental_table




"""#label#
body: bd0 (black), bd1 (black dot), bd2 (white), bd3 (white dot), bd4 (large white), bd5 (large white dot)
armbeam: am0 (upper stem), am1 (lower stem), am2 (upper flag), am3 (lower flag), bm0 (upper single beam), bm1 (lower single beam), bm2 (upper double beam), bm3 (lower double beam)
accidental: ac0 (#), ac1 (b), ac2 (natural) 
rest: re0 (whole), re1 (half), re2 (quarter), re3 (eighth), re4 (16th)
clef: cf0 (clef G), cf1 (clef F), cf2 (clef 8 (8va or 8bv)), cf3 (clef 8va_stop), cf4(8vb_stop): attention! changed
"""
"""
Information about duraion (4/4)
type:     name        duration
bd4:     whole         1024
bd5:     whole dot     1536
bd2:     half           512
bd3:     half dot       768

bd0:
  am0
  am1    quarter         256

  am2
  am3
  bm0
  bm1    eighth          128

  bm2
  bm3     16th            64

bd1 (+ dot):
  am0
  am1    quarter         384

  am2
  am3
  bm0
  bm1    eighth          192

  bm2
  bm3     16th            96

  re0    whole           1024
  re1    half             512
  re2    quarter           256
  re3    eighth           128
  re4    16th              64

"""


#identify note (step, octave, alter) of ms of interest
def identifyNoteFromMS(msOfInterest, current_accidental_table_input, clef_input, alpha, beta):
    #set clef position with alpha and beta
    staffmiddle, heightInterval, notePositionInClef_G, notePositionInClef_F, notePositionInClef_G8va, notePositionInClef_F8vb = set_staffmiddle_heightInterval_for_notePosition(alpha, beta)

    #extract each combination of note and middle height and compare the height to that of msOfInterest
    #then assign the msOfInterest to the corresponding note
    
    note_candidate = ''
    if clef_input == Clef.G:
        for note, middleheight in notePositionInClef_G.items():
            if ((middleheight - heightInterval*0.5) < msOfInterest["y"]) and (msOfInterest["y"] <= (middleheight + heightInterval*0.5)):
                note_candidate = note
            else:
                pass
                # print(f'compare：middleheight - heightInterval*0.5：{middleheight - heightInterval*0.5},msOfInterest["y"]:{msOfInterest["y"]},middleheight + heightInterval*0.5:{middleheight + heightInterval*0.5}')
    elif clef_input == Clef.F:
        for note, middleheight in notePositionInClef_F.items():
            if ((middleheight - heightInterval*0.5) < msOfInterest["y"]) and (msOfInterest["y"] <= (middleheight + heightInterval*0.5)):
                note_candidate = note
            else:
                pass
                # print(f'compare：middleheight - heightInterval*0.5：{middleheight - heightInterval*0.5},msOfInterest["y"]:{msOfInterest["y"]},middleheight + heightInterval*0.5:{middleheight + heightInterval*0.5}')

    elif clef_input == Clef.G8va:
        for note, middleheight in notePositionInClef_G8va.items():
            if ((middleheight - heightInterval*0.5) < msOfInterest["y"]) and (msOfInterest["y"] <= (middleheight + heightInterval*0.5)):
                note_candidate = note
    elif clef_input == Clef.F8vb:
        for note, middleheight in notePositionInClef_F8vb.items():
            if ((middleheight - heightInterval*0.5) < msOfInterest["y"]) and (msOfInterest["y"] <= (middleheight + heightInterval*0.5)):
                note_candidate = note
    else:
        print('no clef is found in NoteIdentifycation')
    #identify the octave, step
    if note_candidate == '':
        print('note_candidate is empty, so the original msOfInterest is returned')
        return msOfInterest
    else:
        step_str = note_candidate[0]
        octave_str = note_candidate[1]
        msOfInterest["step"] = step_str
        msOfInterest["octave"] = octave_str
        #identify the alter
        for step_in_table in current_accidental_table_input:
            if step_str == step_in_table:
                if current_accidental_table_input[step_str] == '#':
                    msOfInterest["alter"] = '1'
                elif current_accidental_table_input[step_str] == 'b':
                    msOfInterest["alter"] = '-1'
                else:
                    msOfInterest["alter"] = ''
        return msOfInterest

#Are topMS and bottomMS different armbeam ms items?
def areTopMSandBottomMSDifferentArmBeams(topMS_input, bottomMS_input):#return True or False
    if topMS_input == bottomMS_input:
        return False
    elif isArmOrBeam(topMS_input) and isArmOrBeam(bottomMS_input):
        return True
    else:
        return False

#check which class the msOfInterest belongs to?
def isRest(msOfInterest):
    if msOfInterest['category'] == Category.re0 or msOfInterest['category'] == Category.re1 or msOfInterest['category'] == Category.re2 or msOfInterest['category'] == Category.re3 or msOfInterest['category'] == Category.re4:
        return True
    else:
        return False
def isAccidental(msOfInterest):
    if msOfInterest['category'] == Category.ac0 or msOfInterest['category'] == Category.ac1 or msOfInterest['category'] == Category.ac2:
        return True
    else:
        return False
def isClef(msOfInterest):
    if msOfInterest['category'] == Category.cf0 or msOfInterest['category'] == Category.cf1 or msOfInterest['category'] == Category.cf2 or msOfInterest['category'] == Category.cf3 or msOfInterest['category'] == Category.cf4:
        return True
    else:
        return False
def isArm(msOfInterest):
    if msOfInterest['category'] == Category.am0 or msOfInterest['category'] == Category.am1 or msOfInterest['category'] == Category.am2 or msOfInterest['category'] == Category.am3:
        return True
    else:
        return False
def isBeam(msOfInterest):
    if msOfInterest['category'] == Category.bm0 or msOfInterest['category'] == Category.bm1 or msOfInterest['category'] == Category.bm2 or msOfInterest['category'] == Category.bm3:
        return True
    else:
        return False
def isArmOrBeam(msOfInterest):
    if isArm(msOfInterest) or isBeam(msOfInterest):
        return True
    else:
        return False

def isStickedBody(msOfInterest):
    if msOfInterest['category'] == Category.bd0 or msOfInterest['category'] == Category.bd1 or msOfInterest['category'] == Category.bd2 or msOfInterest['category'] == Category.bd3:
        return True
    else:
        return False
def isUnstickedBody(msOfInterest):
    if msOfInterest['category'] == Category.bd4 or msOfInterest['category'] == Category.bd5:
        return True
    else:
        return False
def isBody(msOfInterest):
    if isStickedBody(msOfInterest) or isUnstickedBody(msOfInterest):
        return True
    else:
        return False

def closerToLower(msOfInterest, ms_lower, ms_upper):
    if abs(msOfInterest['y'] - ms_upper['y']) > abs(msOfInterest['y'] - ms_lower['y']):
        return True
    else:
        False

def giveWhichBeamTypeStr(msOfInterest, ms_beam):
    #in the case of an upper beam
    if ms_beam['category'] == Category.bm0 or ms_beam['category'] == Category.bm2:
        if (msOfInterest['x'] - (msOfInterest['w'])) < (ms_beam['x'] - ms_beam['w']*0.5) and (ms_beam['x'] - ms_beam['w']*0.5) < (msOfInterest['x'] + (msOfInterest['w']*0.45)):
            return 'begin'
        elif (msOfInterest['x'] - (msOfInterest['w']*0.2)) > (ms_beam['x'] - ms_beam['w']*0.5) and (ms_beam['x'] + ms_beam['w']*0.5) > (msOfInterest['x'] + (msOfInterest['w']*0.9)):
            return 'continue'
        else:
            return 'end'
        # elif msOfInterest['x'] < (ms_beam['x'] + ms_beam['w']*0.5) and (ms_beam['x'] + ms_beam['w']*0.5) < (msOfInterest['x'] + (msOfInterest['w']*1.0)):
        #     return 'end'
        # else:
        #     return 'continue'
    elif ms_beam['category'] == Category.bm1 or ms_beam['category'] == Category.bm3:
        #in the case of an lower beam
        if (msOfInterest['x'] - (msOfInterest['w']*0.58)) < (ms_beam['x'] - ms_beam['w']*0.5) and (ms_beam['x'] - ms_beam['w']*0.5) < msOfInterest['x']:
            return 'begin'
        elif (msOfInterest['x'] - (msOfInterest['w']*0.5)) > (ms_beam['x'] - ms_beam['w']*0.5) and (ms_beam['x'] + ms_beam['w']*0.5) > (msOfInterest['x'] + (msOfInterest['w']*1.0)):
            return 'continue'
        else:
            return 'end'
        # elif (msOfInterest['x'] - (msOfInterest['w']*0.6)) < (ms_beam['x'] + ms_beam['w']*0.5) and (ms_beam['x'] + ms_beam['w']*0.5) < (msOfInterest['x'] + (msOfInterest['w']*0.2)):
        #     return 'end'
        # else:
        #     return 'continue'
    else:
        pass
    

def giveWhichVoiceIfAnyRest(ms_rest):
    if ms_rest['category'] == Category.re2 or ms_rest['category'] == Category.re3 or ms_rest['category'] == Category.re4:
        # if y >= 0.5: staff1, if y< 0.5: staff2
        if ms_rest['y'] >= 0.45:
            return 1#voice1
        else:
            return 2#voice2
    else:
        return 1


#assign a specific voice to a rest ms item
def assignVoice1ToBottomRest(vodMS_input):
    vodMS = copy.copy(vodMS_input)
    #assign the rest to voice1 and the rest to voice2
    if vodMS[0]["category"] == Category.re0:
        vodMS[0]['rest'] = True
        vodMS[0]['duration'] = 1024
        vodMS[0]['type'] = 'whole'
        vodMS[0]['voice'] = 1
        print('rest:whole in combination with body ms item(s)')
    elif vodMS[0]["category"] == Category.re1:
        vodMS[0]['rest'] = True
        vodMS[0]['duration'] = 512
        vodMS[0]['type'] = 'half'
        vodMS[0]['voice'] = 1
        print('rest:half in combination with body ms item(s)')
    elif vodMS[0]["category"] == Category.re2:
        vodMS[0]['rest'] = True
        vodMS[0]['duration'] = 256
        vodMS[0]['type'] = 'quarter'
        vodMS[0]['voice'] = 1
        print('rest:quarter in combination with body ms item(s)')
    elif vodMS[0]["category"] == Category.re3:
        vodMS[0]['rest'] = True
        vodMS[0]['duration'] = 128
        vodMS[0]['type'] = 'eighth'
        vodMS[0]['voice'] = 1
        print('rest:eighth in combination with body ms item(s)')
    elif vodMS[0]["category"] == Category.re4:
        vodMS[0]['rest'] = True
        vodMS[0]['duration'] = 64
        vodMS[0]['type'] = '16th'
        vodMS[0]['voice'] = 1
        print('rest:16th in combination with body ms item(s)')
    else:
        print('There is no bottom rest ms item in combination with body ms item(s)')
    return vodMS

def assignVoice2ToTopRest(vodMS_input):
    vodMS = copy.copy(vodMS_input)
    #assign the rest to voice1 and the rest to voice2
    if vodMS[-1]["category"] == Category.re0:
        vodMS[-1]['rest'] = True
        vodMS[-1]['duration'] = 1024
        vodMS[-1]['type'] = 'whole'
        vodMS[-1]['voice'] = 2
        print('rest:whole in combination with body ms item(s)')
    elif vodMS[-1]["category"] == Category.re1:
        vodMS[-1]['rest'] = True
        vodMS[-1]['duration'] = 512
        vodMS[-1]['type'] = 'half'
        vodMS[-1]['voice'] = 2
        print('rest:half in combination with body ms item(s)')
    elif vodMS[-1]["category"] == Category.re2:
        vodMS[-1]['rest'] = True
        vodMS[-1]['duration'] = 256
        vodMS[-1]['type'] = 'quarter'
        vodMS[-1]['voice'] = 2
        print('rest:quarter in combination with body ms item(s)')
    elif vodMS[-1]["category"] == Category.re3:
        vodMS[-1]['rest'] = True
        vodMS[-1]['duration'] = 128
        vodMS[-1]['type'] = 'eighth'
        vodMS[-1]['voice'] = 2
        print('rest:eighth in combination with body ms item(s)')
    elif vodMS[-1]["category"] == Category.re4:
        vodMS[-1]['rest'] = True
        vodMS[-1]['duration'] = 64
        vodMS[-1]['type'] = '16th'
        vodMS[-1]['voice'] = 2
        print('rest:16th in combination with body ms item(s)')
    else:
        print('There is no top rest ms item in combination with body ms item(s)')
    return vodMS

#exclude beam ms item(s) from vodMS for preparation of ms_body_rest_items_excluded
def excludeBeamFromvodMS(vodMS_input):
    vodMS_template = copy.copy(vodMS_input)
    vodMS = copy.copy(vodMS_input)
    for each_item in vodMS_template:
        if isBeam(each_item):
            vodMS.remove(each_item)
    return vodMS

#check whether vodMS contains any body item
def doesvodMSContainBody(vodMS_input):
    vodMS = copy.copy(vodMS_input)
    bodyFlag = False
    for each_item in vodMS:
        if isBody(each_item):
            bodyFlag = True
    if bodyFlag:
        return True
    else:
        return False
#check whether vodMS contains any rest item
def doesvodMSContainRest(vodMS_input):
    vodMS = copy.copy(vodMS_input)
    restFlag = False
    for each_item in vodMS:
        if isRest(each_item):
            restFlag = True
    if restFlag:
        return True
    else:
        return False
#check whether vodMS contains any clef item
def doesvodMSContainClef(vodMS_input):
    vodMS = copy.copy(vodMS_input)
    clefFlag = False
    for each_item in vodMS:
        if isClef(each_item):
            restFlag = True
    if clefFlag:
        return True
    else:
        return False

#annotate a rest item with a specific voice
def annotateRestWithVoice(rest_ms, voice):
    restMS = copy.copy(rest_ms)
    if restMS["category"] == Category.re0:
        restMS['rest'] = True
        restMS['duration'] = 1024
        restMS['type'] = 'whole'
        restMS['voice'] = voice       
    elif restMS["category"] == Category.re1:
        restMS['rest'] = True
        restMS['duration'] = 512
        restMS['type'] = 'half'
        restMS['voice'] = voice       
    elif restMS["category"] == Category.re2:
        restMS['rest'] = True
        restMS['duration'] = 256
        restMS['type'] = 'quarter'
        restMS['voice'] = voice
    elif restMS["category"] == Category.re3:
        restMS['rest'] = True
        restMS['duration'] = 128
        restMS['type'] = 'eighth'
        restMS['voice'] = voice
    elif restMS["category"] == Category.re4:
        restMS['rest'] = True
        restMS['duration'] = 64
        restMS['type'] = '16th'
        restMS['voice'] = voice
    else:
        print('')
    return restMS

"""
#revise and simplify the annotation system
"""
def annotateLowerBodies(vodMS_input, accidental_table_input, clef_temp_input, alpha, beta, voice):
    vodMS = copy.copy(vodMS_input)
    accidental_table = copy.copy(accidental_table_input)
    clef_temp = copy.copy(clef_temp_input)

    for eachms in vodMS:
        if isArm(vodMS[0]):
            if eachms['category'] == Category.bd0:
                if vodMS[0]['category'] == Category.am1:
                    eachms['voice'] = voice
                    eachms['type'] = 'quarter'
                    eachms['stem'] = 'down'
                    eachms['duration'] = 256
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
                elif vodMS[0]['category'] == Category.am3:
                    eachms['voice'] = voice
                    eachms['type'] = 'eighth'
                    eachms['stem'] = 'down'
                    eachms['duration'] = 128
                    eachms['beam_number'] = 1
                    eachms['beam_content'] = 'begin'
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
            elif eachms['category'] == Category.bd1:
                if vodMS[0]['category'] == Category.am1:
                    eachms['voice'] = voice
                    eachms['type'] = 'quarter'
                    eachms['dot'] = True
                    eachms['stem'] = 'down'
                    eachms['duration'] = 384
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
                elif vodMS[0]['category'] == Category.am3:
                    eachms['voice'] = voice
                    eachms['type'] = 'eighth'
                    eachms['dot'] = True
                    eachms['stem'] = 'down'
                    eachms['duration'] = 192
                    eachms['beam_number'] = 1
                    eachms['beam_content'] = 'begin'
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
            elif eachms['category'] == Category.bd2:
                if vodMS[0]['category'] == Category.am1:
                    eachms['voice'] = voice
                    eachms['type'] = 'half'
                    # vodMS[1]['dot'] = True
                    eachms['stem'] = 'down'
                    eachms['duration'] = 512
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
            elif eachms['category'] == Category.bd3:
                if vodMS[0]['category'] == Category.am1:
                    eachms['voice'] = voice
                    eachms['type'] = 'half'
                    eachms['dot'] = True
                    eachms['stem'] = 'down'
                    eachms['duration'] = 768
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
            else:
                pass
                # print('the bottom arm has an unidentified body or else at the i-th ms place from the bottom.')
        elif isBeam(vodMS[0]):
            if eachms['category'] == Category.bd0:
                if vodMS[0]['category'] == Category.bm1:
                    eachms['voice'] = voice
                    eachms['type'] = 'eighth'
                    eachms['stem'] = 'down'
                    eachms['duration'] = 128
                    eachms['beam_number'] = 1
                    eachms['beam_content'] = giveWhichBeamTypeStr(eachms, vodMS[0])
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
                elif vodMS[0]['category'] == Category.bm3:
                    eachms['voice'] = voice
                    eachms['type'] = '16th'
                    eachms['stem'] = 'down'
                    eachms['duration'] = 64
                    eachms['beam_number'] = 1
                    eachms['beam_content'] = giveWhichBeamTypeStr(eachms, vodMS[0])
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
            elif eachms['category'] == Category.bd1:
                if vodMS[0]['category'] == Category.bm1:
                    eachms['voice'] = voice
                    eachms['type'] = 'eighth'
                    eachms['dot'] = True
                    eachms['stem'] = 'down'
                    eachms['duration'] = 192
                    eachms['beam_number'] = 1
                    eachms['beam_content'] = giveWhichBeamTypeStr(eachms, vodMS[0])
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
                elif vodMS[0]['category'] == Category.bm3:
                    eachms['voice'] = voice
                    eachms['type'] = '16th'
                    eachms['dot'] = True
                    eachms['stem'] = 'down'
                    eachms['duration'] = 96
                    eachms['beam_number'] = 1
                    eachms['beam_content'] = giveWhichBeamTypeStr(eachms, vodMS[0])
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
            else:
                pass
                # print('the bottom beam has an unidentified body or else at the i-th ms place from the bottom.')
        #in the case of unsticked bodies
        if isUnstickedBody(eachms):
            if eachms['category'] == Category.bd4:
                eachms['voice'] = voice
                eachms['type'] = 'whole'
                eachms['stem'] = 'up'
                eachms['duration'] = 1024
                eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
            elif eachms['category'] == Category.bd5:
                eachms['voice'] = voice
                eachms['type'] = 'whole'
                eachms['dot'] = True
                eachms['stem'] = 'up'
                eachms['duration'] = 1536                
                eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
            else:
                pass
    return vodMS

def annotateUpperBodies(vodMS_input, accidental_table_input, clef_temp_input, alpha, beta, voice):
    vodMS = copy.copy(vodMS_input)
    accidental_table = copy.copy(accidental_table_input)
    clef_temp = copy.copy(clef_temp_input)
    
    for eachms in vodMS:
        if isArm(vodMS[-1]):
            if eachms['category'] == Category.bd0:
                if vodMS[-1]['category'] == Category.am0:
                    eachms['voice'] = voice
                    eachms['type'] = 'quarter'
                    eachms['stem'] = 'up'
                    eachms['duration'] = 256
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
                elif vodMS[-1]['category'] == Category.am2:
                    eachms['voice'] = voice
                    eachms['type'] = 'eighth'
                    eachms['stem'] = 'up'
                    eachms['duration'] = 128
                    eachms['beam_number'] = 1
                    eachms['beam_content'] = 'begin'
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
            elif eachms['category'] == Category.bd1:
                if vodMS[-1]['category'] == Category.am0:
                    eachms['voice'] = voice
                    eachms['type'] = 'quarter'
                    eachms['dot'] = True
                    eachms['stem'] = 'up'
                    eachms['duration'] = 384
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
                elif vodMS[-1]['category'] == Category.am2:
                    eachms['voice'] = voice
                    eachms['type'] = 'eighth'
                    eachms['dot'] = True
                    eachms['stem'] = 'up'
                    eachms['duration'] = 192
                    eachms['beam_number'] = 1
                    eachms['beam_content'] = 'begin'
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
            elif eachms['category'] == Category.bd2:
                if vodMS[-1]['category'] == Category.am0:
                    eachms['voice'] = voice
                    eachms['type'] = 'half'
                    # vodMS[1]['dot'] = True
                    eachms['stem'] = 'up'
                    eachms['duration'] = 512
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
            elif eachms['category'] == Category.bd3:
                if vodMS[-1]['category'] == Category.am0:
                    eachms['voice'] = voice
                    eachms['type'] = 'half'
                    eachms['dot'] = True
                    eachms['stem'] = 'up'
                    eachms['duration'] = 768
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
            else:
                pass
                # print('the bottom arm has an unidentified body or else at the i-th ms place from the bottom.')
        elif isBeam(vodMS[-1]):
            if eachms['category'] == Category.bd0:
                if vodMS[-1]['category'] == Category.bm0:
                    eachms['voice'] = voice
                    eachms['type'] = 'eighth'
                    eachms['stem'] = 'up'
                    eachms['duration'] = 128
                    eachms['beam_number'] = 1
                    eachms['beam_content'] = giveWhichBeamTypeStr(eachms, vodMS[-1])
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
                elif vodMS[-1]['category'] == Category.bm2:
                    eachms['voice'] = voice
                    eachms['type'] = '16th'
                    eachms['stem'] = 'up'
                    eachms['duration'] = 64
                    eachms['beam_number'] = 1
                    eachms['beam_content'] = giveWhichBeamTypeStr(eachms, vodMS[-1])
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
            elif eachms['category'] == Category.bd1:
                if vodMS[-1]['category'] == Category.bm0:
                    eachms['voice'] = voice
                    eachms['type'] = 'eighth'
                    eachms['dot'] = True
                    eachms['stem'] = 'up'
                    eachms['duration'] = 192
                    eachms['beam_number'] = 1
                    eachms['beam_content'] = giveWhichBeamTypeStr(eachms, vodMS[-1])
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                    
                elif vodMS[-1]['category'] == Category.bm2:
                    eachms['voice'] = voice
                    eachms['type'] = '16th'
                    eachms['dot'] = True
                    eachms['stem'] = 'up'
                    eachms['duration'] = 96
                    eachms['beam_number'] = 1
                    eachms['beam_content'] = giveWhichBeamTypeStr(eachms, vodMS[-1])
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
            else:
                pass
                # print('the bottom beam has an unidentified body or else at the i-th ms place from the bottom.')
        #in the case of unsticked bodies
        if isUnstickedBody(eachms):
            if eachms['category'] == Category.bd4:
                eachms['voice'] = voice
                eachms['type'] = 'whole'
                eachms['stem'] = 'up'
                eachms['duration'] = 1024
                eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
            elif eachms['category'] == Category.bd5:
                eachms['voice'] = voice
                eachms['type'] = 'whole'
                eachms['dot'] = True
                eachms['stem'] = 'up'
                eachms['duration'] = 1536                
                eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
            else:
                pass
    return vodMS

def annotateUnannotatedStickedBodies(vodMS_input, accidental_table_input, clef_temp_input, alpha, beta, voice):
    vodMS = copy.copy(vodMS_input)
    accidental_table = copy.copy(accidental_table_input)
    clef_temp = copy.copy(clef_temp_input)
    
    for eachms in vodMS:
        if eachms['y'] < 0.5:
            if eachms['category'] == Category.bd0:
                eachms['voice'] = voice
                eachms['type'] = 'quarter'
                eachms['stem'] = 'up'
                eachms['duration'] = 256
                eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
            elif eachms['category'] == Category.bd1:
                eachms['voice'] = voice
                eachms['type'] = 'quarter'
                eachms['dot'] = True
                eachms['stem'] = 'up'
                eachms['duration'] = 384
                eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
            elif eachms['category'] == Category.bd2:
                eachms['voice'] = voice
                eachms['type'] = 'half'
                eachms['stem'] = 'up'
                eachms['duration'] = 512
                eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
            elif eachms['category'] == Category.bd3:
                eachms['voice'] = voice
                eachms['type'] = 'half'
                eachms['dot'] = True
                eachms['stem'] = 'up'
                eachms['duration'] = 768
                eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
            else:
                pass # in the case of unsticked body item (e.g., bd4, bd5)
        else:
            if eachms['category'] == Category.bd0:
                eachms['voice'] = voice
                eachms['type'] = 'quarter'
                eachms['stem'] = 'down'
                eachms['duration'] = 256
                eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
            elif eachms['category'] == Category.bd1:
                eachms['voice'] = voice
                eachms['type'] = 'quarter'
                eachms['dot'] = True
                eachms['stem'] = 'down'
                eachms['duration'] = 384
                eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
            elif eachms['category'] == Category.bd2:
                eachms['voice'] = voice
                eachms['type'] = 'half'
                eachms['stem'] = 'down'
                eachms['duration'] = 512
                eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
            elif eachms['category'] == Category.bd3:
                eachms['voice'] = voice
                eachms['type'] = 'half'
                eachms['dot'] = True
                eachms['stem'] = 'down'
                eachms['duration'] = 768
                eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
            else:
                pass # in the case of unsticked body item (e.g., bd4, bd5)
    return vodMS

def annotateUnannotatedUnstickedBodies(vodMS_input, accidental_table_input, clef_temp_input, alpha, beta, voice):
    vodMS = copy.copy(vodMS_input)
    accidental_table = copy.copy(accidental_table_input)
    clef_temp = copy.copy(clef_temp_input)
    for eachms in vodMS:
        if isUnstickedBody(eachms):
                if eachms['category'] == Category.bd4:
                    eachms['voice'] = voice
                    eachms['type'] = 'whole'
                    eachms['stem'] = 'up'
                    eachms['duration'] = 1024
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                elif eachms['category'] == Category.bd5:
                    eachms['voice'] = voice
                    eachms['type'] = 'whole'
                    eachms['dot'] = True
                    eachms['stem'] = 'up'
                    eachms['duration'] = 1536                
                    eachms = identifyNoteFromMS(eachms, accidental_table, clef_temp, alpha, beta)
                else:
                    pass
    return vodMS
    


#calculate a duration for each of voice1 or voice2 in vodMS
def calculateDurationForEachVoiceInvodMS(vodMS_input):
    vodMS = copy.copy(vodMS_input)
    durationlist_temp_voice1 = []
    durationlist_temp_voice2 = []
    for eachMS in vodMS:
        if eachMS['voice'] == 1:
            durationlist_temp_voice1.append(eachMS['duration'])
        elif eachMS['voice'] == 2:
            durationlist_temp_voice2.append(eachMS['duration'])
    # set the maximum duration in each duration list as a duration for each voice in the vodMS
    if len(durationlist_temp_voice1) >0:
        duration_max_voice1 = max(durationlist_temp_voice1)
    else:
        duration_max_voice1 = 0
    if len(durationlist_temp_voice2) >0:
        duration_max_voice2 = max(durationlist_temp_voice2)
    else:
        duration_max_voice2 = 0       
    return duration_max_voice1, duration_max_voice2

def calculateDurationForEachVoiceInall_vodMS_rest_items(all_vodMS_rest_items_input):
    all_vodMS_rest_items = copy.copy(all_vodMS_rest_items_input)
    #initialize duration lists
    durationlist_voice1 = []
    durationlist_voice2 = []
    for each_vodMS in all_vodMS_rest_items:
        maxduration_voice1, maxduration_voice2 = calculateDurationForEachVoiceInvodMS(each_vodMS)
        durationlist_voice1.append(maxduration_voice1)
        durationlist_voice2.append(maxduration_voice2)
    totalduration_voice1 = sum(durationlist_voice1)
    totalduration_voice2 = sum(durationlist_voice2)
    # print(f'preset_measure_duration：{preset_measure_duration}、totalduration_voice1は{totalduration_voice1}、totalduration_voice2は{totalduration_voice2}です')
    return totalduration_voice1, totalduration_voice2


#check, change, and review voices in all_vodMS_rest_items
def checkChangeReviewVoices(all_vodMS_rest_items_input, preset_measure_duration):
    all_vodMS_rest_items = copy.copy(all_vodMS_rest_items_input)
    
    totalduration_voice1, totalduration_voice2 = calculateDurationForEachVoiceInall_vodMS_rest_items(all_vodMS_rest_items)
    #if totalduration_voice1 exceeds the preset_measure_duration, adjust the voice
    if totalduration_voice1 > (preset_measure_duration) or totalduration_voice2 > (preset_measure_duration):
        print(f'preset_measure_duration：{preset_measure_duration}; the totalduration_voice1 before the first adjustment is {totalduration_voice1}; the totalduration_voice2は is {totalduration_voice2}')
        #first revision: specifically, if the stem of body is 'down', the voice is 1; if the stem of body is 'up', the voice is 2
        for each_vodMS in all_vodMS_rest_items:
            for eachms in each_vodMS:
                if eachms['stem'] == 'down':
                    eachms['voice'] = 1
                elif eachms['stem'] == 'up':
                    eachms['voice'] = 2
        totalduration_voice1, totalduration_voice2 = calculateDurationForEachVoiceInall_vodMS_rest_items(all_vodMS_rest_items)
        print(f'preset_measure_duration：{preset_measure_duration}; the totalduration_voice1 after the first adjustment is {totalduration_voice1}; the totalduration_voice2 is {totalduration_voice2}')
    else:
        print('duration adjustment is not carried out')

    totalduration_voice1, totalduration_voice2 = calculateDurationForEachVoiceInall_vodMS_rest_items(all_vodMS_rest_items)
    #if totalduration_voice1 exceeds the preset_measure_duration, adjust the voice
    if totalduration_voice1 > (preset_measure_duration) or totalduration_voice2 > (preset_measure_duration):
        print(f'preset_measure_duration：{preset_measure_duration}; the totalduration_voice1 before the second adjustment (unsticked bodies in cases 4 and 5) is {totalduration_voice1}; the totalduration_voice2 is {totalduration_voice2}')
        #second revision: specifically, voices of unsticked body items in the cases 4 and 5 are assigned accordingly
        for each_vodMS in all_vodMS_rest_items:
            if isUnstickedBody(each_vodMS[-1]) and isArmOrBeam(each_vodMS[0]):#case (4) with unsticked body(ies)
                for eachMS in each_vodMS:
                    if isUnstickedBody(eachMS):
                        eachMS['voice'] = 2
            elif isUnstickedBody(each_vodMS[0]) and isArmOrBeam(each_vodMS[-1]):#case (5) with unsticked body(ies)
                for eachMS in each_vodMS:
                    if isStickedBody(eachMS):
                        eachMS['voice'] = 2
            if isBody(each_vodMS[-1]) and isBody(each_vodMS[0]):
                for eachMS in each_vodMS:
                    if isUnstickedBody(eachMS):
                        eachMS['voice'] = 2
        totalduration_voice1, totalduration_voice2 = calculateDurationForEachVoiceInall_vodMS_rest_items(all_vodMS_rest_items)
        print(f'preset_measure_duration：{preset_measure_duration}; the totalduration_voice1 after the second adjustment (unsticked bodies in cases 4 and 5) is {totalduration_voice1}; the totalduration_voice2 is {totalduration_voice2}')

    return all_vodMS_rest_items

def giveChordAccordingly(vodMS):
    #do not copy vodMS because the ['chord'] of the vodMS ms itmes should be changed accordingly
    vodMS_for_mssorting = []
    ms_group = []
    for eachms in vodMS:
        #only body ms should be added and checked
        if isBody(eachms):
            vodMS_for_mssorting.append(eachms)
    ms_bodygroup = ms_horizontalsorting(vodMS_for_mssorting)
    #give a chord:True or False accordingly: those except for the first appearing body or rest should be assigned as ['chord'] = True
    for i, eachms in enumerate(ms_bodygroup):
        if i == 0:
            eachms['chord'] = False
        else:
            eachms['chord'] = True
    

#annotate each ms item in the horizontallysortedMSlist as an input
def annotateEachMS(horizontallysortedMSlist_input, current_accidental_table_input, staff_input, clef_input, preset_measure_duration, alpha, beta):
    horizontallysortedMSlist = copy.copy(horizontallysortedMSlist_input)
    accidental_table = copy.copy(current_accidental_table_input)
    staff_temp = copy.copy(staff_input)
    clef_temp =  copy.copy(clef_input)
    # #set noteposition adjusted by alpha and beta
    # notePositionInClef_G, notePositionInClef_F, notePositionInClef_G8va, notePositionInClef_F8vb = set_staffmiddle_heightInterval_for_notePosition(alpha, beta)

    # print(f'annotateEachMS関数内のhorizontallysortedMSlistは{horizontallysortedMSlist}')
    # print(f'annotateEachMS関数にinputしたcurrent_accidental_table_inputは{current_accidental_table_input}\n関数内のaccidental_tableは{accidental_table}')
    #Analysis(x axis: ascending (0 to X); y axis: descending (Y to 0))
    #voice flag:set if there are possible voices in a measure
    areVoices = False
    #clef flag:set if the clef is changed
    # clefchangeFlag = False
    #check how long the all ms items (excluding chord portions), voice1, or voice 2 are
    durationlist_voice1 = []
    durationlist_voice2 = []
    #for intermediate rest items
    prebody_intermediate_rest = []
    postbody_intermediate_rest = []
    #analyzed body and associated arm items and top and bottom rest ms items are excluded to avoid repeated analysis
    #beam ms item should be reused, so that ms_body_rest_items_excluded should not contain any beam item
    ms_body_arm_rest_items_excluded = []#e.g., [[ms_rest0],vodMS_0,vodMS_1,[ms_restt1]];vodMS as a list of ms items; each ms is a dictionary
    #all vodMS and [rest] are stored for voice adjustmet later
    all_vodMS_rest_items = []#e.g., [[ms_rest0],vodMS_0,vodMS_1,[ms_cf2], vodMS_2, [ms_restt1]];

    #output ms items sequence
    output_ms_sequecne = []

    #analyze each ms item in the order of clef, accidental, rest, body (with beam)
    for eachms in horizontallysortedMSlist:
        # print(f'解析するカテゴリは{eachms["category"]}')
        #check clef or octave-shift changes
        if isClef(eachms):
            # if eachms['y'] < 0.3 or eachms['y'] > 0.7:
            #     pass
            #     # ms_body_arm_rest_items_excluded.append(eachms)
            if eachms['clefchange'] == True:
                clef_temp = eachms['clef']
                all_vodMS_rest_items.append([eachms])
                # print('x0,x1でのclefchangeの追加設定を通りました。')
            else:
                if eachms["category"] == Category.cf0:
                    if eachms['y'] < 0.3 or eachms['y'] > 0.7:
                        pass
                    elif clef_temp == (Clef.F or Clef.F8vb):
                        eachms['clef'] = Clef.G
                        clef_temp = Clef.G
                        eachms['clefchange'] = True
                        all_vodMS_rest_items.append([eachms])
                        # clefchangeFlag = True
                elif eachms["category"] == Category.cf1:
                    if eachms['y'] < 0.3 or eachms['y'] > 0.7:
                        pass
                    elif clef_temp == (Clef.G or Clef.G8va):
                        eachms['clef'] = Clef.F
                        clef_temp = Clef.F
                        eachms['clefchange'] = True
                        all_vodMS_rest_items.append([eachms])
                        # clefchangeFlag = True
            if eachms["category"] == Category.cf2:
                #temporarily disabled
                ms_body_arm_rest_items_excluded.append(eachms)
                # if eachms['y'] < 0.2:
                #     eachms['octave_shift'] = 'down'
                #     eachms['clef'] = Clef.G8va
                #     clef_temp = Clef.G8va
                #     all_vodMS_rest_items.append([eachms])
                #     print('octave_shift:downです。')
                # elif eachms['y'] > 0.8:
                #     eachms['octave_shift'] = 'up'
                #     eachms['clef'] = Clef.F8vb
                #     clef_temp = Clef.F8vb
                #     all_vodMS_rest_items.append([eachms])
                #     print('octave_shift:upです。')
            elif eachms["category"] == Category.cf3:
                #temporarily disabled
                ms_body_arm_rest_items_excluded.append(eachms)
                # if eachms['y'] < 0.2:
                #     if clef_temp == Clef.G8va:
                #         eachms['octave_shift'] = 'stop'
                #         eachms['clef'] = Clef.G
                #         clef_temp = Clef.G
                #         all_vodMS_rest_items.append([eachms])  
                #         print('octave_shift:stopです。')
            elif eachms["category"] == Category.cf4:
                #temporarily disabled
                ms_body_arm_rest_items_excluded.append(eachms)
                # if eachms['y'] > 0.8:
                #     if clef_temp == Clef.F8vb:
                #         eachms['octave_shift'] = 'stop'
                #         eachms['clef'] = Clef.F
                #         clef_temp = Clef.F
                #         all_vodMS_rest_items.append([eachms])
                #         print('octave_shift:stopです。')
        #check accidental changes
        elif isAccidental(eachms):
            accidental_table = changeAccidentalTable(accidental_table, eachms, clef_temp, alpha, beta)
            
        
        #annotate bodies in combination with arms (stems) and beams
        #if any, give different voices and the areVoices flag is set to review the entire measure again
        elif isBody(eachms):
            #set the present clef:clef_temp
            eachms["clef"] == clef_temp
            #empty pre- and post-body rest list
            prebody_intermediate_rest = []
            postbody_intermediate_rest = []
            #to avoid repeated analysis; first flatten a list of lists(containing dictionary ms items)
            chain_object = chain.from_iterable(ms_body_arm_rest_items_excluded)
            flattened_list = list(chain_object)
            if not eachms in flattened_list:
                #from experience, the width and the x (center) of each body (bd0, bd2, bd4) should be a little narrower
                if eachms['category'] == Category.bd0 or eachms['category'] == Category.bd2 or eachms['category'] == Category.bd4:
                    # eachms['x'] = eachms['x'] - (eachms['w']*0.2)
                    eachms['w'] *= 0.75

                #if eachms is bd1, bd3, or bd5 (dot body), shift the x position and change w (0.8x) to avoid false verticall overlapping ms items.
                if eachms['category'] == Category.bd1 or eachms['category'] == Category.bd3 or eachms['category'] == Category.bd5:
                    eachms['x'] = eachms['x'] - (eachms['w']*0.1)
                    eachms['w'] *= 0.8

                #collect vertically overlapping ms items
                vodMS = collectVerticallyOverlappingDescendingMusicalSymbols(horizontallysortedMSlist, eachms)
                # print(f'vodMS(VerticallyOverlappingDescendingMusicalSymbols) is {vodMS}')
                #first change the clef_temp and remove a clef if the included clef is before eachms or just remove it if after eachms
                #if 8va, 8vb, or 8vstop　is included, change vodMS['clef'] and clef_temp accordingly and it should be removed from vodMS
                vodMS_copy = copy.copy(vodMS)
               
                for item in vodMS_copy:
                    if item['x'] < eachms['x']:
                        if item['category'] == Category.cf0:
                            clef_temp = Clef.G
                            vodMS.remove(item)                            
                        elif item['category'] == Category.cf1:
                            clef_temp = Clef.F
                            vodMS.remove(item)
                    else:
                        if item['category'] == Category.cf0:
                            vodMS.remove(item)                            
                        elif item['category'] == Category.cf1:
                            vodMS.remove(item)
                        #temporalily added 
                        elif item['category'] == Category.cf2 or item['category'] == Category.cf3 or item['category'] == Category.cf4:
                            vodMS.remove(item)
                """temporalily disabled
                # for item in vodMS_copy:
                #     if item['category'] == Category.cf2 and clef_temp == Clef.G:
                #         clef_temp = Clef.G8va
                #         vodMS.remove(item)
                        
                #     elif item['category'] == Category.cf2 and clef_temp == Clef.F:
                #         clef_temp = Clef.F8vb
                #         vodMS.remove(item)
                        
                #     elif item['category'] == Category.cf3 and clef_temp == Clef.G8va:
                #         clef_temp = Clef.G
                #         vodMS.remove(item)
                        
                #     elif item['category'] == Category.cf4 and clef_temp == Clef.F8vb:
                #         clef_temp = Clef.F
                #         vodMS.remove(item)
                #     else:
                #         pass
                #         # print('other than the combination of 8vb or vbstop or the clef-temp is wrong') 
                """
                #remove arm/beam items  (am0, am2, bm0, bm2) at vodMS[0] from a lower staff2 and arm/beam items (am1, am3, bm1, bm 3) at vodMS[-1] from an upper staff1 
                if vodMS_copy[0]['category'] == Category.am0 or vodMS_copy[0]['category'] == Category.am2 or vodMS_copy[0]['category'] == Category.bm0 or vodMS_copy[0]['category'] == Category.bm2:
                    vodMS.pop(0)
                if vodMS_copy[-1]['category'] == Category.am1 or vodMS_copy[-1]['category'] == Category.am3 or vodMS_copy[-1]['category'] == Category.bm1 or vodMS_copy[-1]['category'] == Category.bm3:
                    vodMS.pop()

                for every_ms_in_vodMS in vodMS:
                    every_ms_in_vodMS['clef'] = clef_temp
                #remove ms_clef and ms_accidental
                #and remove any intermediate rest
                for item in vodMS_copy:
                    if isAccidental(item):
                        vodMS.remove(item)
                    elif isRest(item):
                        if item == vodMS[0]:
                            print('there is the bottom rest ms item in vodMS[].')
                        elif item == vodMS[-1]:
                            print('there is the top rest ms item in vodMS[].')
                        else:
                            #intermediate rest is added to either prebody or postbody list and ms_body_arm_rest_items_excluded for exclusion from the subsequent analysis
                            if item['x'] < eachms['x']:
                                prebody_intermediate_rest.append(item)
                            else:
                                postbody_intermediate_rest.append(item)
                            ms_body_arm_rest_items_excluded.append(item)
                            vodMS.remove(item)
                            print('there is an intermediate rest ms item in vodMS[].')
                    else:
                        print('')
                        # print('there is other than clef, accidental, or rest ms items')

                #for bd1, bd3, or bd5, if the x distance therebetween is  >0.05, exclude it
                for analyteMS in vodMS:
                    if analyteMS['category'] == Category.bd1 or analyteMS['category'] == Category.bd3 or analyteMS['category'] == Category.bd5:
                        if abs(analyteMS['x'] - eachms['x']) > 0.05:
                            vodMS.remove(analyteMS)
                # print(f'除去後のvodMSは{vodMS}')

                #give a chord such as the first horizontally aligned body or rest item is given no chord and the remaining body or rest items are ginve 'chord' = True.
                giveChordAccordingly(vodMS)

                #start analysis in the decending order
                #first, determine the number of ms items in the vodMS
                ms_item_count = len(vodMS)
                if ms_item_count == 0:
                    continue

                #the case (1) where both the top and the boddom ms items are arm and/or beam items
                if ms_item_count >= 4 and isArmOrBeam(vodMS[0]) and isArmOrBeam(vodMS[-1]):
                    areVoices = True
                    # #About the bottom vodMS[0]
                    # vodMS = annotateFirstlowerStickedBody(vodMS, accidental_table=accidental_table, clef_temp=clef_temp, voice=1)
                    # #About the top vodMS[-1]
                    # vodMS = annotateFirstUpperStickedBody(vodMS, accidental_table=accidental_table, clef_temp=clef_temp, voice=2)
                    if ms_item_count == 4:
                        vodMS[:2] = annotateLowerBodies(vodMS[:2], accidental_table, clef_temp, alpha, beta, voice=1)
                        vodMS[2:] = annotateUpperBodies(vodMS[2:], accidental_table, clef_temp, alpha, beta, voice=2)
                    #About intermediate ms items (at most 3 body ms items)
                    elif ms_item_count == 5:
                        if isStickedBody(vodMS[2]) and closerToLower(vodMS[2], ms_lower=vodMS[1], ms_upper=vodMS[3]):
                            vodMS[:3] = annotateLowerBodies(vodMS[:3], accidental_table, clef_temp, alpha, beta, voice=1)
                            vodMS[3:] = annotateUpperBodies(vodMS[3:], accidental_table, clef_temp, alpha, beta, voice=2)
                        else:
                            vodMS[:2] = annotateLowerBodies(vodMS[:2], accidental_table, clef_temp, alpha, beta, voice=1)
                            vodMS[2:] = annotateUpperBodies(vodMS[2:], accidental_table, clef_temp, alpha, beta, voice=2)

                    elif ms_item_count == 6:
                        if isStickedBody(vodMS[2]) and closerToLower(vodMS[2], ms_lower=vodMS[1], ms_upper=vodMS[-2]):#'1' indicates the botoom body;'-2' indicates the top body
                            if isStickedBody(vodMS[3]) and closerToLower(vodMS[3], ms_lower=vodMS[2], ms_upper=vodMS[-2]):
                                vodMS[:4] = annotateLowerBodies(vodMS[:4], accidental_table, clef_temp, alpha, beta, voice=1)
                                vodMS[4:] = annotateUpperBodies(vodMS[4:], accidental_table, clef_temp, alpha, beta, voice=2)
                            else:
                                vodMS[:3] = annotateLowerBodies(vodMS[:3], accidental_table, clef_temp, alpha, beta, voice=1)
                                vodMS[3:] = annotateUpperBodies(vodMS[3:], accidental_table, clef_temp, alpha, beta, voice=2)
                        else:
                            vodMS[:2] = annotateLowerBodies(vodMS[:2], accidental_table, clef_temp, alpha, beta, voice=1)
                            vodMS[2:] = annotateUpperBodies(vodMS[2:], accidental_table, clef_temp, alpha, beta, voice=2)
                    elif ms_item_count == 7:
                        if isStickedBody(vodMS[2]) and closerToLower(vodMS[2], ms_lower=vodMS[1], ms_upper=vodMS[-2]):#'1' indicates the botoom body;'-2' indicates the top body
                            if isStickedBody(vodMS[3]) and closerToLower(vodMS[3], ms_lower=vodMS[2], ms_upper=vodMS[-2]):
                                if isStickedBody(vodMS[4]) and closerToLower(vodMS[4], ms_lower=vodMS[3], ms_upper=vodMS[-2]):
                                    #the case where all intermediate ms body items belong to the lower arm or beam (voice1)
                                    vodMS[:5] = annotateLowerBodies(vodMS[:5], accidental_table, clef_temp, alpha, beta, voice=1)
                                    vodMS[5:] = annotateUpperBodies(vodMS[5:], accidental_table, clef_temp, alpha, beta, voice=2)
                                else:
                                    vodMS[:4] = annotateLowerBodies(vodMS[:4], accidental_table, clef_temp, alpha, beta, voice=1)
                                    vodMS[4:] = annotateUpperBodies(vodMS[4:], accidental_table, clef_temp, alpha, beta, voice=2)
                            else:
                                vodMS[:3] = annotateLowerBodies(vodMS[:3], accidental_table, clef_temp, alpha, beta, voice=1)
                                vodMS[3:] = annotateUpperBodies(vodMS[3:], accidental_table, clef_temp, alpha, beta, voice=2)
                        else:
                            vodMS[:2] = annotateLowerBodies(vodMS[:2], accidental_table, clef_temp, alpha, beta, voice=1)
                            vodMS[2:] = annotateUpperBodies(vodMS[2:], accidental_table, clef_temp, alpha, beta, voice=2)

                    #    print(f'The ms_item_count is {ms_item_count} and there should be an ms item other than the body ms item.')
                    #add vodMS and pre/post intermediate rest items to all_vodMS_rest_items and add analyzed vodMS (witout beam) to ms_body_arm_rest_items_excluded list for exclusion 
                    if len(prebody_intermediate_rest) > 0:
                        all_vodMS_rest_items.append(prebody_intermediate_rest)
                    all_vodMS_rest_items.append(vodMS)
                    if len(postbody_intermediate_rest) > 0:
                        all_vodMS_rest_items.append(postbody_intermediate_rest)
                    vodMS = excludeBeamFromvodMS(vodMS)
                    ms_body_arm_rest_items_excluded.append(vodMS)

                #the case (2) where the bottom ms is a rest
                elif ms_item_count >= 2 and isRest(vodMS[0]):
                    areVoices = True
                    #assign voice1 to the bottom rest ms item
                    vodMS = assignVoice1ToBottomRest(vodMS)
                    #annotate other body components in vodMS and assign voice2 to the compoents
                    vodMS = annotateUpperBodies(vodMS, accidental_table, clef_temp, alpha, beta, voice=2)
                    #add vodMS and pre/post intermediate rest items to all_vodMS_rest_items and add analyzed vodMS (witout beam) to ms_body_arm_rest_items_excluded list for exclusion 
                    if len(prebody_intermediate_rest) > 0:
                        all_vodMS_rest_items.append(prebody_intermediate_rest)
                    all_vodMS_rest_items.append(vodMS)
                    if len(postbody_intermediate_rest) > 0:
                        all_vodMS_rest_items.append(postbody_intermediate_rest)
                    vodMS = excludeBeamFromvodMS(vodMS)
                    ms_body_arm_rest_items_excluded.append(vodMS)
                #the case (3) where the top ms is a rest
                elif ms_item_count >= 2 and isRest(vodMS[-1]):
                    areVoices = True
                    #assign voice2 to the top rest ms item
                    vodMS = assignVoice2ToTopRest(vodMS)
                    #annotate other body components in vodMS and assign voice1 to the compoents
                    vodMS = annotateLowerBodies(vodMS, accidental_table, clef_temp, alpha, beta, voice=1)
                    #add vodMS and pre/post intermediate rest items to all_vodMS_rest_items and add analyzed vodMS (witout beam) to ms_body_arm_rest_items_excluded list for exclusion 
                    if len(prebody_intermediate_rest) > 0:
                        all_vodMS_rest_items.append(prebody_intermediate_rest)
                    all_vodMS_rest_items.append(vodMS)
                    if len(postbody_intermediate_rest) > 0:
                        all_vodMS_rest_items.append(postbody_intermediate_rest)
                    vodMS = excludeBeamFromvodMS(vodMS)
                    ms_body_arm_rest_items_excluded.append(vodMS)
                #the case (4) where the bottom ms is an arm or beam ms and the top ms is a sticked or unsticked body
                elif ms_item_count >= 2 and isArmOrBeam(vodMS[0]) and isBody(vodMS[-1]):
                    #annotate each ms component and assign voice1 to all
                    vodMS = annotateLowerBodies(vodMS, accidental_table, clef_temp, alpha, beta, voice=1)
                    #add vodMS and pre/post intermediate rest items to all_vodMS_rest_items and add analyzed vodMS (witout beam) to ms_body_arm_rest_items_excluded list for exclusion 
                    if len(prebody_intermediate_rest) > 0:
                        all_vodMS_rest_items.append(prebody_intermediate_rest)
                    all_vodMS_rest_items.append(vodMS)
                    if len(postbody_intermediate_rest) > 0:
                        all_vodMS_rest_items.append(postbody_intermediate_rest)
                    vodMS = excludeBeamFromvodMS(vodMS)
                    ms_body_arm_rest_items_excluded.append(vodMS)
                #the case (5) where the top ms is an arm or beam ms and the bottom ms is a sticked or unsticked body
                elif ms_item_count >= 2 and isArmOrBeam(vodMS[-1]) and isBody(vodMS[0]):
                    #annotate each ms component and assign voice1 to all
                    vodMS = annotateUpperBodies(vodMS, accidental_table, clef_temp, alpha, beta, voice=1)
                    #add vodMS and pre/post intermediate rest items to all_vodMS_rest_items and add analyzed vodMS (witout beam) to ms_body_arm_rest_items_excluded list for exclusion 
                    if len(prebody_intermediate_rest) > 0:
                        all_vodMS_rest_items.append(prebody_intermediate_rest)
                    all_vodMS_rest_items.append(vodMS)
                    if len(postbody_intermediate_rest) > 0:
                        all_vodMS_rest_items.append(postbody_intermediate_rest)
                    vodMS = excludeBeamFromvodMS(vodMS)
                    ms_body_arm_rest_items_excluded.append(vodMS)
                    #the case where both the top and bottom ms items are sticked/unsticked body ms items (including the only one sticked/unsticked body ms item)
                #the case (6) where all ms items are body items (i.e., bd4 (whole) or bd5 (whole dot))
                elif isBody(vodMS[0]) and isBody(vodMS[-1]):
                    vodMS = annotateLowerBodies(vodMS, accidental_table, clef_temp, alpha, beta, voice=1)
                    #in the case of having any unannotated body (e.g., bd0, bd1, bd2, or bd3 without any upper or lower arm or beam)
                    vodMS = annotateUnannotatedStickedBodies(vodMS, accidental_table, clef_temp, alpha, beta, voice=1)
                    #add vodMS and pre/post intermediate rest items to all_vodMS_rest_items and add analyzed vodMS (witout beam) to ms_body_arm_rest_items_excluded list for exclusion 
                    if len(prebody_intermediate_rest) > 0:
                        all_vodMS_rest_items.append(prebody_intermediate_rest)
                    all_vodMS_rest_items.append(vodMS)
                    if len(postbody_intermediate_rest) > 0:
                        all_vodMS_rest_items.append(postbody_intermediate_rest)
                    vodMS = excludeBeamFromvodMS(vodMS)
                    ms_body_arm_rest_items_excluded.append(vodMS)

                else:
                    #in the case of having any unannotated body (e.g., bd0, bd1, bd2, or bd3 without any upper or lower arm or beam)
                    vodMS = annotateUnannotatedStickedBodies(vodMS, accidental_table, clef_temp, alpha, beta, voice=1)
                    vodMS = annotateUnannotatedUnstickedBodies(vodMS, accidental_table, clef_temp, alpha, beta, voice=1)
                    print('there may be another possible vodMS[] combination, so annotate bodies anyway')
                    #add vodMS and pre/post intermediate rest items to all_vodMS_rest_items and add analyzed vodMS (witout beam) to ms_body_arm_rest_items_excluded list for exclusion 
                    if len(prebody_intermediate_rest) > 0:
                        all_vodMS_rest_items.append(prebody_intermediate_rest)
                    all_vodMS_rest_items.append(vodMS)
                    if len(postbody_intermediate_rest) > 0:
                        all_vodMS_rest_items.append(postbody_intermediate_rest)
                    vodMS = excludeBeamFromvodMS(vodMS)
                    ms_body_arm_rest_items_excluded.append(vodMS)
    
        #check any rest item independent from body items; i.e., any single intermediate rest item                
        elif isRest(eachms):
            #collect vertically overlapping ms items
            vodMS = collectVerticallyOverlappingDescendingMusicalSymbols(horizontallysortedMSlist, eachms)
            #if vodMS contains any body item, this rest item should be analyzed in the body section
            if doesvodMSContainBody(vodMS):
                print(f'this rest item{eachms["category"]} will be analyazed in the body section.')
            else:
                eachms = annotateRestWithVoice(eachms, voice=1)
                all_vodMS_rest_items.append([eachms])# append eachms as a list [eachms]
                ms_body_arm_rest_items_excluded.append([eachms])

        else:
            pass
            # print(f'unaanalyzed category')
    
    #start voice adjustment in all_vodMS_rest_items.append([eachms])
    
    #the case where there are voices (see above cases: both ends are arm/beam and/or rest items)
    if areVoices:
        #the initial duration for each voice when areVoices are true
        
        totalduration_voice1, totalduration_voice2 = calculateDurationForEachVoiceInall_vodMS_rest_items(all_vodMS_rest_items)
        
        print(f'preset_measure_duration：{preset_measure_duration}; totalduration_voice1 is {totalduration_voice1}; totalduration_voice2 is {totalduration_voice2}')
        #rest items at a Y position less than 0.45 should be assighed to voice2
        # ms items where one end is an upper arm/beam and the other end is a body should be assigned to voice2
        vodMSitems_count = len(all_vodMS_rest_items)
        for i in range(0, vodMSitems_count):
            if doesvodMSContainRest(all_vodMS_rest_items[i]):
                for j in range(0,len(all_vodMS_rest_items[i])):
                    if all_vodMS_rest_items[i][j]['y'] < 0.45:
                        all_vodMS_rest_items[i][j] = annotateRestWithVoice(all_vodMS_rest_items[i][j],voice=2)
            elif isBody(all_vodMS_rest_items[i][0]) and isArmOrBeam(all_vodMS_rest_items[i][-1]):
                
                for j in range(0,len(all_vodMS_rest_items[i])):
                    all_vodMS_rest_items[i][j]['voice'] = 2

        all_vodMS_rest_items = checkChangeReviewVoices(all_vodMS_rest_items, preset_measure_duration)
    else:
        #if areVoices is False, check and change voices and the total duration of vodMS and review the voices
        all_vodMS_rest_items = checkChangeReviewVoices(all_vodMS_rest_items, preset_measure_duration)

    #return the body_rest_items_sequence, accidental_tabel, clef_temp, and clefchangeFlag
    #prepare the body_rest_items_sequence by removing arm/beam items from all_vodMS_rest_items, by aligning chord items, and by flattening the resulting list
    template_all_vodMS_rest_items = copy.copy(all_vodMS_rest_items)
    for each_vodMS in template_all_vodMS_rest_items:
        for eachMS in each_vodMS:
            if isArmOrBeam(eachMS):
                each_vodMS.remove(eachMS)
    for each_vodMS in template_all_vodMS_rest_items:
        if doesvodMSContainBody(each_vodMS):
            voice1_group = []
            voice2_group = []
            rest_in_group1 = []
            rest_in_group2 = []
            aligned_voice1_group = []
            aligned_voice2_group = []
            for eachMS in each_vodMS:
                if eachMS['voice'] == 1:
                    voice1_group.append(eachMS)
                elif eachMS['voice'] == 2:
                    voice2_group.append(eachMS)
            #align the eachMS in each voice in the order of 'non-chord' to 'chord' items
            if len(voice1_group) >0:
                for ingroup_item in voice1_group:
                    if isRest(ingroup_item):
                       rest_in_group1.append(ingroup_item) 
                    elif ingroup_item['chord'] == True:
                        aligned_voice1_group.append(ingroup_item)
                    elif ingroup_item['chord'] == False:
                        aligned_voice1_group.insert(0, ingroup_item)
            if len(voice2_group) >0:
                for ingroup_item in voice2_group:
                    if isRest(ingroup_item):
                       rest_in_group2.append(ingroup_item)
                    elif ingroup_item['chord'] == True:
                        aligned_voice2_group.append(ingroup_item)
                    elif ingroup_item['chord'] == False:
                        aligned_voice2_group.insert(0, ingroup_item)
            each_vodMS = rest_in_group1 + aligned_voice1_group + rest_in_group2 + aligned_voice2_group
    body_rest_items_sequence = list(chain.from_iterable(template_all_vodMS_rest_items))
    return body_rest_items_sequence, accidental_table, clef_temp # clefchangeFlag


#for systemintegration
# input data: isPaired(areStavesPaired), aligned_staves(staves_with_measures_in_sheetmusic), FILE_PATH (as an original sheet music image file path)

def collectAllmsInEachmeasureForStaff1or2(PATH_dirname, type_of_eachmeasure_input, staff):
    type_in_eachmeasure = copy.copy(type_of_eachmeasure_input)
    #serially add each ms in each category from each measure
    #initialize and prepare a dictionary
    
    empty_ms = musicalSymbol_template
    all_ms_in_eachmeasure = {'measure_dummy':[empty_ms]}
    # eachmeasure has an empty_ms as default
    for eachmeasure in type_in_eachmeasure:
        all_ms_in_eachmeasure[eachmeasure] = [empty_ms]
    
    #Category:body
    body_PATH = PATH_dirname + '/staff' + str(staff) + '/body/labels/*'
    print(f'body_PATHは{body_PATH}')
    files = glob.glob(body_PATH)
    
    for file in files:
        if not file.endswith('txt'):
            print('The file is not a text file: {}'.format(file))
        else:
            dirname = os.path.dirname(file)
            namewithoutext = os.path.splitext(os.path.basename(file))[0]
            nameOfMeasure = namewithoutext[0:11]
            txtfile = dirname + '/' + namewithoutext + '.txt'
            # To obtain textlines
            txtlines = []
            with open(txtfile) as f:
                txtlines = f.readlines()
                for textline in txtlines:
                    target_info = textline.split() #target_info =[label, x, y, w, h]
                    ms_temp = copy.copy(musicalSymbol_template)
                    ms_temp['measuretype'] = type_in_eachmeasure[nameOfMeasure]
                    #assign clef
                    if ms_temp['measuretype'] == MeasureType.x0:
                        ms_temp['clef'] = Clef.G
                    elif ms_temp['measuretype'] == MeasureType.x1:
                        ms_temp['clef'] = Clef.F
                    else:
                        ms_temp['clef'] = Clef.none
                    ms_temp['x'] = float(target_info[1])
                    ms_temp['y'] = float(target_info[2])
                    ms_temp['w'] = float(target_info[3])
                    ms_temp['h'] = float(target_info[4])
                    #depending on each category
                    if target_info[0] == '0':
                        ms_temp['category'] = Category.bd0
                    elif target_info[0] == '1':
                        ms_temp['category'] = Category.bd1
                    elif target_info[0] == '2':
                        ms_temp['category'] = Category.bd2
                    elif target_info[0] == '3':
                        ms_temp['category'] = Category.bd3
                    elif target_info[0] == '4':
                        ms_temp['category'] = Category.bd4
                    elif target_info[0] == '5':
                        ms_temp['category'] = Category.bd5
                    else:
                        ms_temp['category'] = Category.none

                    # add ms to a previous [ms] in all_ms_in_eachmeasure['nameOfMeasure']
                    previous_mslist = all_ms_in_eachmeasure[nameOfMeasure]
                    previous_mslist.append(ms_temp)
                    all_ms_in_eachmeasure[nameOfMeasure] = previous_mslist

    #Category:armbeam
    armbeam_PATH = PATH_dirname + '/staff' + str(staff) + '/armbeam/labels/*'
    files = glob.glob(armbeam_PATH)
    
    for file in files:
        if not file.endswith('txt'):
            print('The file is not a text file: {}'.format(file))
        else:
            dirname = os.path.dirname(file)
            namewithoutext = os.path.splitext(os.path.basename(file))[0]
            nameOfMeasure = namewithoutext[0:11]
            txtfile = dirname + '/' + namewithoutext + '.txt'
            # To obtain textlines
            txtlines = []
            with open(txtfile) as f:
                txtlines = f.readlines()
                for textline in txtlines:
                    target_info = textline.split() #target_info =[label, x, y, w, h]
                    ms_temp = copy.copy(musicalSymbol_template)
                    ms_temp['measuretype'] = type_in_eachmeasure[nameOfMeasure]
                    #assign clef
                    if ms_temp['measuretype'] == MeasureType.x0:
                        ms_temp['clef'] = Clef.G
                    elif ms_temp['measuretype'] == MeasureType.x1:
                        ms_temp['clef'] = Clef.F
                    else:
                        ms_temp['clef'] = Clef.none
                    ms_temp['x'] = float(target_info[1])
                    ms_temp['y'] = float(target_info[2])
                    ms_temp['w'] = float(target_info[3])
                    ms_temp['h'] = float(target_info[4])
                    #depending on each category
                    if target_info[0] == '0':
                        ms_temp['category'] = Category.am0
                    elif target_info[0] == '1':
                        ms_temp['category'] = Category.am1
                    elif target_info[0] == '2':
                        ms_temp['category'] = Category.am2
                    elif target_info[0] == '3':
                        ms_temp['category'] = Category.am3
                    elif target_info[0] == '4':
                        ms_temp['category'] = Category.bm0
                    elif target_info[0] == '5':
                        ms_temp['category'] = Category.bm1
                    elif target_info[0] == '6':
                        ms_temp['category'] = Category.bm2
                    elif target_info[0] == '7':
                        ms_temp['category'] = Category.bm3
                    else:
                        ms_temp['category'] = Category.none

                    # add ms to a previous [ms] in all_ms_in_eachmeasure['nameOfMeasure']
                    previous_mslist = all_ms_in_eachmeasure[nameOfMeasure]
                    previous_mslist.append(ms_temp)
                    all_ms_in_eachmeasure[nameOfMeasure] = previous_mslist

    #Category:rest
    rest_PATH = PATH_dirname + '/staff' + str(staff) + '/rest/labels/*'
    files = glob.glob(rest_PATH)
    
    for file in files:
        if not file.endswith('txt'):
            print('The file is not a text file: {}'.format(file))
        else:
            dirname = os.path.dirname(file)
            namewithoutext = os.path.splitext(os.path.basename(file))[0]
            nameOfMeasure = namewithoutext[0:11]
            txtfile = dirname + '/' + namewithoutext + '.txt'
            # To obtain textlines
            txtlines = []
            with open(txtfile) as f:
                txtlines = f.readlines()
                for textline in txtlines:
                    target_info = textline.split() #target_info =[label, x, y, w, h]
                    ms_temp = copy.copy(musicalSymbol_template)
                    ms_temp['measuretype'] = type_in_eachmeasure[nameOfMeasure]
                    #assign clef
                    if ms_temp['measuretype'] == MeasureType.x0:
                        ms_temp['clef'] = Clef.G
                    elif ms_temp['measuretype'] == MeasureType.x1:
                        ms_temp['clef'] = Clef.F
                    else:
                        ms_temp['clef'] = Clef.none
                    ms_temp['x'] = float(target_info[1])
                    ms_temp['y'] = float(target_info[2])
                    ms_temp['w'] = float(target_info[3])
                    ms_temp['h'] = float(target_info[4])
                    #depending on each category
                    if target_info[0] == '0':
                        ms_temp['category'] = Category.re0
                    elif target_info[0] == '1':
                        ms_temp['category'] = Category.re1
                    elif target_info[0] == '2':
                        ms_temp['category'] = Category.re2
                    elif target_info[0] == '3':
                        ms_temp['category'] = Category.re3
                    elif target_info[0] == '4':
                        ms_temp['category'] = Category.re4
                    else:
                        ms_temp['category'] = Category.none

                    # add ms to a previous [ms] in all_ms_in_eachmeasure['nameOfMeasure']
                    previous_mslist = all_ms_in_eachmeasure[nameOfMeasure]
                    previous_mslist.append(ms_temp)
                    all_ms_in_eachmeasure[nameOfMeasure] = previous_mslist
    #Category:accidental
    accidental_PATH = PATH_dirname + '/staff' + str(staff) + '/accidental/labels/*'
    files = glob.glob(accidental_PATH)
    
    for file in files:
        if not file.endswith('txt'):
            print('The file is not a text file: {}'.format(file))
        else:
            dirname = os.path.dirname(file)
            namewithoutext = os.path.splitext(os.path.basename(file))[0]
            nameOfMeasure = namewithoutext[0:11]
            txtfile = dirname + '/' + namewithoutext + '.txt'
            # To obtain textlines
            txtlines = []
            with open(txtfile) as f:
                txtlines = f.readlines()
                for textline in txtlines:
                    target_info = textline.split() #target_info =[label, x, y, w, h]
                    ms_temp = copy.copy(musicalSymbol_template)
                    ms_temp['measuretype'] = type_in_eachmeasure[nameOfMeasure]
                    #assign clef
                    if ms_temp['measuretype'] == MeasureType.x0:
                        ms_temp['clef'] = Clef.G
                    elif ms_temp['measuretype'] == MeasureType.x1:
                        ms_temp['clef'] = Clef.F
                    else:
                        ms_temp['clef'] = Clef.none
                    ms_temp['x'] = float(target_info[1])
                    ms_temp['y'] = float(target_info[2])
                    ms_temp['w'] = float(target_info[3])
                    ms_temp['h'] = float(target_info[4])
                    #depending on each category
                    if target_info[0] == '0':
                        ms_temp['category'] = Category.ac0
                    elif target_info[0] == '1':
                        ms_temp['category'] = Category.ac1
                    elif target_info[0] == '2':
                        ms_temp['category'] = Category.ac2
                    else:
                        ms_temp['category'] = Category.none

                    # add ms to a previous [ms] in all_ms_in_eachmeasure['nameOfMeasure']
                    previous_mslist = all_ms_in_eachmeasure[nameOfMeasure]
                    previous_mslist.append(ms_temp)
                    all_ms_in_eachmeasure[nameOfMeasure] = previous_mslist

    #Category:clef
    accidental_PATH = PATH_dirname + '/staff' + str(staff) + '/clef/labels/*'
    files = glob.glob(accidental_PATH)
    
    for file in files:
        if not file.endswith('txt'):
            print('The file is not a text file: {}'.format(file))
        else:
            dirname = os.path.dirname(file)
            namewithoutext = os.path.splitext(os.path.basename(file))[0]
            nameOfMeasure = namewithoutext[0:11]
            txtfile = dirname + '/' + namewithoutext + '.txt'
            # To obtain textlines
            txtlines = []
            with open(txtfile) as f:
                txtlines = f.readlines()
                for textline in txtlines:
                    target_info = textline.split() #target_info =[label, x, y, w, h]
                    ms_temp = copy.copy(musicalSymbol_template)
                    ms_temp['measuretype'] = type_in_eachmeasure[nameOfMeasure]
                    #assign clef
                    if ms_temp['measuretype'] == MeasureType.x0:
                        ms_temp['clef'] = Clef.G
                    elif ms_temp['measuretype'] == MeasureType.x1:
                        ms_temp['clef'] = Clef.F
                    else:
                        ms_temp['clef'] = Clef.none
                    ms_temp['x'] = float(target_info[1])
                    ms_temp['y'] = float(target_info[2])
                    ms_temp['w'] = float(target_info[3])
                    ms_temp['h'] = float(target_info[4])
                    #depending on each category
                    if target_info[0] == '0':
                        ms_temp['category'] = Category.cf0
                    elif target_info[0] == '1':
                        ms_temp['category'] = Category.cf1
                    elif target_info[0] == '2':
                        ms_temp['category'] = Category.cf2
                    elif target_info[0] == '3':
                        ms_temp['category'] = Category.cf3
                    elif target_info[0] == '4':
                        ms_temp['category'] = Category.cf4
                    else:
                        ms_temp['category'] = Category.none

                    # add ms to a previous [ms] in all_ms_in_eachmeasure['nameOfMeasure']
                    previous_mslist = all_ms_in_eachmeasure[nameOfMeasure]
                    previous_mslist.append(ms_temp)
                    all_ms_in_eachmeasure[nameOfMeasure] = previous_mslist

    
    return all_ms_in_eachmeasure



#to collect ms items from all categories in each measure 
#type_in_eachmeasure = {'measure#000':MeasureType.x0, 'measure#001':MeasureType.y0, 'measure#000':MeasureType.y0,'measure#000':MeasureType.x1,}
# all_ms_in_eachmeasure = {'measure#000':[ms0, ms1, ms2, ms3, ms4], 'measure#001':[ms0, ms1, ms2, ms3, ms4] }
def give_all_ms_in_eachmeasure_for_staff1or2(isPaired, aligned_staves_input, img_FILE_PATH):
    aligned_staves = copy.copy(aligned_staves_input)
    # #set noteposition adjusted by alpha and beta
    # set_staffmiddle_heightInterval_for_notePosition(alpha, beta)
    #return results
    all_ms_in_eachmeasure_staff1 = {}
    all_ms_in_eachmeasure_staff2 = {}
    
    # to create a type_of_eachmeasure dictionary
    type_of_eachmeasure_staff1 = {}
    type_of_eachmeasure_staff2 = {}

    measures_in_staff1 = []
    measures_in_staff2 = []

    #first divide the measures into a staff1 group or staff2 group
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
    print(f'the number of itmes in measures_in_staff1 is {len(measures_in_staff1)}')
    print(f'the number of items in measures_in_staff2 is {len(measures_in_staff2)}')
    print(f'are measures_in_staff1 and measures_in_staff２ identical?: {measures_in_staff1 == measures_in_staff2}')
    
    if len(measures_in_staff1) > 0:
        for i, eachmeasure in enumerate(measures_in_staff1):
            name_temp = 'measure#' + '{:0=3}'.format(i)
            # print(f'{i}番目のname_tempは{name_temp}')
            if eachmeasure['label'] == '0':
                # print(f'label==0です。')
                type_of_eachmeasure_staff1[name_temp] = MeasureType.x0
            elif eachmeasure['label'] == '1':
                type_of_eachmeasure_staff1[name_temp] = MeasureType.x1
            elif eachmeasure['label'] == '2':
                type_of_eachmeasure_staff1[name_temp] = MeasureType.y0
    if len(measures_in_staff2) > 0:
        for i, eachmeasure in enumerate(measures_in_staff2):
            name_temp = 'measure#' + '{:0=3}'.format(i)
            if eachmeasure['label'] == '0':
                type_of_eachmeasure_staff2[name_temp] = MeasureType.x0
            elif eachmeasure['label'] == '1':
                type_of_eachmeasure_staff2[name_temp] = MeasureType.x1
            elif eachmeasure['label'] == '2':
                type_of_eachmeasure_staff2[name_temp] = MeasureType.y0
    print(f'the number of items in type_of_eachmeasure_staff1 is {len(type_of_eachmeasure_staff1)}')
    print(f'type_of_eachmeasure_staff1 is {type_of_eachmeasure_staff1}')
    print(f'the number of items in type_of_eachmeasure_staff2 is {len(type_of_eachmeasure_staff2)}')
    print(f'type_of_eachmeasure_staff2 is {type_of_eachmeasure_staff2}')
    #to get the PATH directory name
    files = glob.glob(img_FILE_PATH) #as an original sheet music image
    
    for file in files:
        if file.endswith('jpg') or file.endswith('png'):               
            # dirname, basename including an extention
            PATH_dirname = os.path.dirname(file)
            if len(type_of_eachmeasure_staff1) > 0:
                all_ms_in_eachmeasure_staff1 = collectAllmsInEachmeasureForStaff1or2(PATH_dirname, type_of_eachmeasure_staff1, staff=1)
            if len(type_of_eachmeasure_staff2) > 0:
                all_ms_in_eachmeasure_staff2 = collectAllmsInEachmeasureForStaff1or2(PATH_dirname, type_of_eachmeasure_staff2, staff=2)
    print(f'are all_ms_in_eachmeasure_staff1 and all_ms_in_eachmeasure_staff2 identical?:{all_ms_in_eachmeasure_staff1 == all_ms_in_eachmeasure_staff2}')
    return all_ms_in_eachmeasure_staff1, all_ms_in_eachmeasure_staff2

"""
#a bit faster but less accurate
def giveblackarea(img_input):
    img = copy.copy(img_input)   
    #http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    ret, img_processed = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    whole_area = img_processed.size
    whitePixels = cv2.countNonZero(img_processed)
    blackPixels = whole_area - whitePixels
    # print(f'blackPixelsは{blackPixels}')
    return blackPixels
"""
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
    img_template = copy.copy(img_input)
    img_copy = copy.copy(img_input)
    height, width, c = img_template.shape
    alpha_best = 0.0
    beta_best = 0.0
    blackarea_minimum = 416 * 416 #the size of img
    for alpha_pre in range(-40, 40):
        alpha = float(alpha_pre / 1000)
        for beta_pre in range(-5, 5):
            beta = float(beta_pre / 1000)
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




def generateMSsequenceForStaff1or2(all_ms_in_eachmeasure_input, current_accidental_table_input, staff, current_clef_input, preset_measure_duration, FILE_PATH, isWideStaff=False):
    all_ms_in_eachmeasure = copy.copy(all_ms_in_eachmeasure_input)
    current_accidental_table_original = copy.copy(current_accidental_table_input)
    
    current_clef = copy.copy(current_clef_input)
    #obtain PATH for each measure# image
    files_temp = glob.glob(FILE_PATH) #"./tmp/*":beforehand prepare images and Yolov5 anotation files in ./tmp/subdirectory
    
    FILE_DIR_PATH = ''
    FILE_EXT = ''
    for file_temp in files_temp:
        if file_temp.endswith('jpg') or file_temp.endswith('png'):
            FILE_DIR_PATH = os.path.dirname(file_temp)
            FILE_BASENAME = os.path.basename(file_temp)
            FILE_BASENAME_WITHOUTEXT = os.path.splitext(FILE_BASENAME)[0]
            FILE_EXT = os.path.splitext(FILE_BASENAME)[1]
            
    MEASURE_DIR = FILE_DIR_PATH + '/measure/staff' + str(staff) +'/'
    #return results as
    ms_sequenceOfInterest = []
    # count excluding a dummy item
    measure_count = len(all_ms_in_eachmeasure) -1
    for i in range(0, measure_count):
        nameOfMeasure = 'measure#' + '{:0=3}'.format(i)
        if nameOfMeasure in all_ms_in_eachmeasure:
            #determine alpha, beta
            MEASURE_PATH = MEASURE_DIR + nameOfMeasure + FILE_EXT
            img_measure = cv2.imread(MEASURE_PATH)
            alpha, beta = determine_alphabeta(img_measure)
            print(f'{nameOfMeasure}: alpha is {alpha} and beta is {beta}')
            #sort ms items horizontally
            hsMSlist = ms_horizontalsorting(all_ms_in_eachmeasure[nameOfMeasure])#hs means horizontally sorted one
            
            #isWideStaff
            if not isWideStaff:
                hsMSlist_copy = copy.copy(hsMSlist)
                for eachms in hsMSlist_copy:
                    if eachms['y'] < 0.15 or eachms['y'] > 0.85:
                        hsMSlist.remove(eachms)

            current_accidental_table = copy.copy(current_accidental_table_original)
            current_clef_input = copy.copy(current_clef)
            
            print(f'the number of items in hsMSlist is {len(hsMSlist)}')
            print(f'hsMSlist[0]["measuretype"] is {hsMSlist[0]["measuretype"]}; the category is {hsMSlist[0]["category"]}')
            if hsMSlist[1]['measuretype'] == MeasureType.x0 and (hsMSlist[1]['category'] == Category.cf0):
                current_clef_input = hsMSlist[1]['clef']
                hsMSlist[1]['clefchange'] = True
                # print('generateMSsequenceForStaff1or2のclefchangeを通過しました。')
            elif hsMSlist[1]['measuretype'] == MeasureType.x1 and (hsMSlist[1]['category'] == Category.cf1):
                current_clef_input = hsMSlist[0]['clef']
                hsMSlist[1]['clefchange'] = True
                # print('generateMSsequenceForStaff1or2のclefchangeを通過しました。')
            body_rest_items_sequence_result, current_accidental_table, current_clef = annotateEachMS(hsMSlist, current_accidental_table, staff, current_clef_input, preset_measure_duration, alpha, beta)
            ms_sequenceOfInterest.append(body_rest_items_sequence_result)
    return ms_sequenceOfInterest

"""
generate a dictionary for ET
source: ms_sequenceOfInterest_staff1 and/or ms_sequenceOfInterest_staff2
"""
def giveBodyNoteDict(eachnote_ms, note_count, staff):
    nameOfNote = 'note' + str(note_count)
    note_unit = {}
    note_unit[nameOfNote] = {}


    if isBody(eachnote_ms):
        # print(f'eachnote{note_count}は{eachnote_ms}')
        if eachnote_ms['chord'] == True:
            note_unit[nameOfNote].update({'chord':''})
    
        if eachnote_ms['dot'] == True:
            if eachnote_ms['alter'] == '':
                note_unit[nameOfNote].update({'pitch': {
                        'step':eachnote_ms['step'],
                        
                        'octave':str(eachnote_ms['octave'])},
                    'duration':str(eachnote_ms['duration']),
                    'voice':str(eachnote_ms['voice']),
                    'type':eachnote_ms['type'],
                    'dot':'',
                    'stem':eachnote_ms['stem'],
                    'staff':str(staff)})
            else:
                note_unit[nameOfNote].update({'pitch': {
                    'step':eachnote_ms['step'],
                    'alter':str(eachnote_ms['alter']),
                    'octave':str(eachnote_ms['octave'])},
                'duration':str(eachnote_ms['duration']),
                'voice':str(eachnote_ms['voice']),
                'type':eachnote_ms['type'],
                'dot':'',
                'stem':eachnote_ms['stem'],
                'staff':str(staff)})
        else:
            #dot == False
            if eachnote_ms['alter'] == '':
                note_unit[nameOfNote].update({'pitch': {
                        'step':eachnote_ms['step'],
                        
                        'octave':str(eachnote_ms['octave'])},
                    'duration':str(eachnote_ms['duration']),
                    'voice':str(eachnote_ms['voice']),
                    'type':eachnote_ms['type'],
                    'stem':eachnote_ms['stem'],
                    'staff':str(staff)})
            else:
                note_unit[nameOfNote].update({'pitch': {
                    'step':eachnote_ms['step'],
                    'alter':str(eachnote_ms['alter']),
                    'octave':str(eachnote_ms['octave'])},
                'duration':str(eachnote_ms['duration']),
                'voice':str(eachnote_ms['voice']),
                'type':eachnote_ms['type'],
                'stem':eachnote_ms['stem'],
                'staff':str(staff)})
        #beam
        if eachnote_ms['type'] == 'eighth' and eachnote_ms['beam_content'] == '':
            #in the case of arm
            note_unit[nameOfNote].update({'beam':{
                    'number':'1',
                    'content':'begin'
                }})
        elif eachnote_ms['beam_content'] == 'begin' or eachnote_ms['beam_content'] == 'continue' or eachnote_ms['beam_content'] == 'end':
            note_unit[nameOfNote].update({'beam':{
                    'number':str(eachnote_ms['beam_number']),
                    'content':eachnote_ms['beam_content']}})
    if eachnote_ms['step'] == '': 
        return {}
    else: 
        return {nameOfNote:note_unit[nameOfNote]}


def giveRestNoteDict(eachnote_ms, note_count, staff):
    nameOfNote = 'note' + str(note_count)
    note_unit = {}
    note_unit[nameOfNote] = {}
    if isRest(eachnote_ms):
        # print(f'eachnote{note_count}は{eachnote_ms}')
        if eachnote_ms['chord'] == True:
            note_unit[nameOfNote].update({'chord':''})
        
        note_unit[nameOfNote].update({'rest':'',
                'duration':str(eachnote_ms['duration']),
                'voice':str(eachnote_ms['voice']),
                'type':eachnote_ms['type'],
                'staff':str(staff)})
    return {nameOfNote:note_unit[nameOfNote]}    

def giveDirectionDict(eachnote_ms, direction_count, staff):
    nameOfDirection = 'direction' + str(direction_count)
    direction_unit = {}
    direction_unit[nameOfDirection] = {}
    if isClef(eachnote_ms):
        # print(f'eachnote{note_count}は{eachnote_ms}')
        if eachnote_ms['octave_shift'] == 'down' or eachnote_ms['octave_shift'] == 'up' or eachnote_ms['octave_shift'] == 'stop':
            direction_unit[nameOfDirection].update({'direction-type':{
                    'octave-shift':{
                        'type':eachnote_ms['octave_shift'],
                    }                    
                }})
        
    return {nameOfDirection:direction_unit[nameOfDirection]}

def giveClefDict(eachnote_ms, staff):
    eachnote_ms = copy.copy(eachnote_ms)
    staff = copy.copy(staff)
    nameOfAttributes_added = 'attributes_added'
    attributes_added_unit = {}
    attributes_added_unit[nameOfAttributes_added] = {}
    print('giveClefDictを通りました。')

    if eachnote_ms['clefchange'] == True:
        
        if eachnote_ms['clef'] == Clef.F or eachnote_ms['clef'] == Clef.F8vb:
            # sign_str = 'F'
            attributes_added_unit[nameOfAttributes_added].update({'clef':{
                    'attrib': {
                        'number': str(staff)
                    },
                    'sign':'F',
                    'line':'4'
                }})
        elif eachnote_ms['clef'] == Clef.G or eachnote_ms['clef'] == Clef.G8va:
            # sign_str = 'G'
            attributes_added_unit[nameOfAttributes_added].update({'clef':{
                    'attrib': {
                        'number': str(staff)
                    },
                    'sign':'G',
                    'line':'2'
                }})
        # print(f'eachnote{note_count}は{eachnote_ms}')
        
        
    return {nameOfAttributes_added:attributes_added_unit[nameOfAttributes_added]}


def generateDictForET_singlestaff(ms_sequenceOfInterest_staff_input, tempo, beats, beat_type, fifths, clef):
    ms_sequenceOfInterest_staff = copy.copy(ms_sequenceOfInterest_staff_input)
    music_data_template = {'part':{}}
    measure_group_dict = {}
    #set clef string (G or F)
    clef_str = ''
    line_str = '2'
    if clef == Clef.G or clef == Clef.G8va:
        clef_str = 'G'
        line_str = '2'
    elif clef == Clef.F or clef == Clef.F8vb:
        clef_str = 'F'
        line_str = '4'
    #in the case of only staff 1
    if len(ms_sequenceOfInterest_staff) > 0:
        for i, eachmeasure in enumerate(ms_sequenceOfInterest_staff):
            nameOfMeasure = 'measure' + str(i+1)
            component_unit = {}
            note_count = 1
            direction_count = 1
            if i == 0:
                #when i == 0, provides basic sheet music information
                component_unit[nameOfMeasure] = {'attrib': {'number':str(i+1),'width':'360'},'attributes': {'divisions':'256','key':{'fifths':str(fifths),'mode':'major'},'time':{'beats':str(beats),'beat-type':str(beat_type)},'staves':'1','clef':{'sign':clef_str,'line':line_str }} }#'direction0':{'sound':{'tempo':str(tempo)}}
                for j, eachnote_ms in enumerate(eachmeasure):
                    if isBody(eachnote_ms):
                        component_unit[nameOfMeasure].update(giveBodyNoteDict(eachnote_ms, note_count=note_count, staff=1))
                        note_count += 1
                    elif isRest(eachnote_ms):
                        component_unit[nameOfMeasure].update(giveRestNoteDict(eachnote_ms, note_count=note_count, staff=1))
                        note_count += 1
                    elif eachnote_ms['octave_shift'] != '':
                        component_unit[nameOfMeasure].update(giveDirectionDict(eachnote_ms, direction_count=direction_count, staff=1))
                    elif eachnote_ms['clefchange'] == True:
                        component_unit[nameOfMeasure].update(giveClefDict(eachnote_ms, staff=1))

                music_data_template['part'].update({nameOfMeasure:component_unit[nameOfMeasure]})
            else:
                component_unit[nameOfMeasure] = {'attrib': {'number':str(i+1),'width':'360'}}
                for j, eachnote_ms in enumerate(eachmeasure):
                        if isBody(eachnote_ms):
                            component_unit[nameOfMeasure].update(giveBodyNoteDict(eachnote_ms, note_count=note_count, staff=1))
                            note_count += 1
                        elif isRest(eachnote_ms):
                            component_unit[nameOfMeasure].update(giveRestNoteDict(eachnote_ms, note_count=note_count, staff=1))
                            note_count += 1
                        elif eachnote_ms['octave_shift'] == 'down' or eachnote_ms['octave_shift'] == 'up' or eachnote_ms['octave_shift'] == 'stop':
                            component_unit[nameOfMeasure].update(giveDirectionDict(eachnote_ms, direction_count=direction_count, staff=1))
                        elif eachnote_ms['clefchange'] == True:
                            component_unit[nameOfMeasure].update(giveClefDict(eachnote_ms, staff=1))
                music_data_template['part'].update({nameOfMeasure:component_unit[nameOfMeasure]})
        
        return music_data_template    



