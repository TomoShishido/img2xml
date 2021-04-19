# # coding: UTF-8
""" Yolov5 to musicXML generation
    To generate an musicXML file from musical symbols obtained by multiple Yolov5 models
"""

#To generate an element tree from Yolov5 data
import xml.etree.ElementTree as ET
import copy

"""
music_data = {
    'part': {
        'measure': {
            'attrib': {
                'number': "1",
                'width': "360"
            },
            'direction': '10',
            'attributes': '10',
            'note': {
                'pitch': {
                    'step': '10',
                    'alter': '10',
                    'octave': 'a'
                },
                'duration': '10',
                'voice': '20',
                'type': '30',
            }
        }
    }
}
"""

music_data = {
    'part': {
        'measure1': {
            'attrib': {
                'number':'1',
                'width':'360'
            },
            'attributes': {
                'divisions':'256',
                'key':{
                    'fifths':'2',
                    'mode':'major'
                },
                'time':{
                    'beats':'4',
                    'beat-type':'4'
                },
                'staves':'1',
                'clef':{
                    'attrib': {
                        'number': '1'
                    },
                    'sign':'G',
                    'line':'2'
                }
            },
            'direction0':{
                'sound':{
                    'tempo':'60'
                }
            },
            'note1': {
                'pitch': {
                    'step':'F',
                    'alter':'1',
                    'octave':'5'
                },
                'duration':'1024',
                'voice':'1',
                'type':'whole',
                'stem':'up',
                'staff':'1'
            }
        },
        'measure2': {
            'attrib': {
                'number':'2',
                'width':'360'
            },
            'note1': {
                'pitch': {
                    'step':'G',
                    'alter':'-1',
                    'octave':'5'
                },
                'duration':'128',
                'voice':'1',
                'type':'eighth',
                'stem':'down',
                'staff':'1',
                'beam':{
                    'number':'1',
                    'content':'begin'
                }
            },
            'direction0':{
                'direction-type':{
                    'octave-shift':{
                        'type':'down'
                    }                    
                }
            },
            'note2': {
                'pitch': {
                    'step':'F',
                    'octave':'6'
                },
                'duration':'128',
                'voice':'1',
                'type':'eighth',
                'stem':'down',
                'staff':'1',
                'beam':{
                    'number':'1',
                    'content':'end'
                }
            },
            'note3': {
                'pitch': {
                    'step':'C',
                    'alter':'1',
                    'octave':'6'
                },
                'duration':'128',
                'voice':'1',
                'type':'eighth',
                'stem':'down',
                'staff':'1',
                'beam':{
                    'number':'1',
                    'content':'end'
                }
            },
            'direction1':{
                'direction-type':{
                    'octave-shift':{
                        'type':'stop',
                    }                    
                }
            },
        },
        'measure3':{
            'attrib': {
                'number':'3',
                'width':'480'
            },
            'note1':{
                'rest':'',
                'duration':'128',
                'voice':'1',
                'type':'eighth',
                'staff':'1'
            },
            'note2':{
                'rest':'',
                'duration':'384',
                'voice':'1',
                'type':'quarter',
                'dot':'',
                'staff':'1'
            }
        },
        'measure4':{
            'attrib': {
                'number':'4',
                'width':'360'
            },
            'note1': {
                'pitch': {
                    'step':'G',
                    
                    'octave':'5'
                },
                'duration':'128',
                'voice':'1',
                'type':'eighth',
                'stem':'down',
                'staff':'1',
                
            },
            'note2': {
                'chord':'',
                'pitch': {
                    'step':'F',
                    
                    'octave':'5'
                },
                'duration':'128',
                'voice':'1',
                'type':'eighth',
                'stem':'down',
                'staff':'1',

            },
            'note3': {
                'chord':'',
                'pitch': {
                    'step':'C',
                    
                    'octave':'5'
                },
                'duration':'128',
                'voice':'1',
                'type':'eighth',
                'stem':'down',
                'staff':'1',

            }
        }
    }
}

part_et = ET.Element('part')
part_et.attrib = {'id':'P1'}

def musicData2XML(part_et, music_data):
    for part, measures_value in music_data.items():
        for measure, measure_value in measures_value.items():
            measure_et = ET.SubElement(part_et, 'measure')
            for measure_key, measure_value in measure_value.items():
                # print(measure_key, measure_value)
                if not isinstance(measure_value, dict): # key value pair
                    elem_et = ET.SubElement(measure_et, measure_key)
                    elem_et.text = measure_value
                elif measure_key == 'attrib':
                    measure_et.attrib = measure_value
                
                elif measure_key == 'attributes' or measure_key == 'attributes1':
                    attributes_et = ET.SubElement(measure_et, 'attributes')
                    for attributes_key, attributes_value in measure_value.items():
                        if not isinstance(attributes_value, dict):
                            subattributes_elem_et = ET.SubElement(attributes_et, attributes_key)
                            subattributes_elem_et.text = attributes_value
                        else:
                            subattributes_dicelem_et = ET.SubElement(attributes_et, attributes_key)
                            for subattributes_dicelem_key, subattributes_dicelem_value in attributes_value.items():
                                if subattributes_dicelem_key == 'attrib':
                                    subattributes_dicelem_et.attrib = subattributes_dicelem_value
                                    continue
                                sub_subattributes_dicelem_et = ET.SubElement(subattributes_dicelem_et, subattributes_dicelem_key)
                                sub_subattributes_dicelem_et.text = subattributes_dicelem_value
                elif measure_key == 'attributes_added':
                    attributes_added_et = ET.SubElement(measure_et, 'attributes')
                    #check again later whether this is OK
                    for attributes_key, attributes_value in measure_value.items():
                        if not isinstance(attributes_value, dict):
                            subattributes_elem_et = ET.SubElement(attributes_added_et, attributes_key)
                            subattributes_elem_et.text = attributes_value
                        else:
                            subattributes_dicelem_et = ET.SubElement(attributes_added_et, attributes_key)
                            for subattributes_dicelem_key, subattributes_dicelem_value in attributes_value.items():
                                if subattributes_dicelem_key == 'attrib':
                                    subattributes_dicelem_et.attrib = subattributes_dicelem_value
                                    continue
                                sub_subattributes_dicelem_et = ET.SubElement(subattributes_dicelem_et, subattributes_dicelem_key)
                                sub_subattributes_dicelem_et.text = subattributes_dicelem_value


                elif measure_key == 'direction0' or measure_key == 'direction1' or measure_key == 'direction2':
                    direction_et = ET.SubElement(measure_et, 'direction')
                    for direction_key, direction_value in measure_value.items():
                        # if direction_key == 'sound':
                        #     sound_et = ET.SubElement(direction_et, direction_key)
                        #     for sound_key, sound_value in direction_value.items():
                        #         sound_et.attrib = {'tempo':sound_value}
                        if direction_key == 'direction-type':
                            direction_type_et = ET.SubElement(direction_et, 'direction-type')
                            for direction_type_key, direction_type_value in direction_value.items():
                                if direction_type_key == 'octave-shift':
                                    octave_shift_et = ET.SubElement(direction_type_et, 'octave-shift')
                                    octave_shift_et.attrib = direction_type_value
                                    # for octave_shift_key, octave_shift_value in direction_type_value.items():
                                    #     print(f'octave_shift_valueは{octave_shift_value}')
                                    #     octave_shift_et.attrib = direction_type_value


                else:  # dict and key=note
                    note_et = ET.SubElement(measure_et, 'note')
                    for note_key, note_value in measure_value.items():
                        if not isinstance(note_value, dict):
                            subelem_et = ET.SubElement(note_et, note_key)
                            subelem_et.text = note_value
                        elif note_key == 'pitch':
                            pitch_et = ET.SubElement(note_et, note_key)
                            for pitch_key, pitch_value in note_value.items():
                                subsubelem_et = ET.SubElement(pitch_et, pitch_key)
                                subsubelem_et.text = pitch_value
                        elif note_key == 'beam':
                            beam_et = ET.SubElement(note_et, note_key)
                            for beam_key, beam_value in note_value.items():
                                if beam_key == 'number':
                                    beam_et.attrib = {'number':beam_value}
                                if beam_key == 'content':
                                    beam_et.text = beam_value
                        else:
                            print(f'Another note item  is {note_key}.')
    return part_et










