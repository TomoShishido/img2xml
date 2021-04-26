# img2Mxml (sheetmusic2musicxml) (GNU licence)
To produce a musicxml file from a sheet music image
(You must comply with and are also resibonsible for copyright laws, etc. for use of sheet music images in img2Mxml)

1. img2Mxml (sheetmusic2musicxml) can convert a sheet music image (.jpg or .png) for a piano piece (i.e., with (right hand) staff 1 and (left hand) staff 2) into a musicxml file (staff1 or staff 2).
2. Use img2xml.ipynb on GoogleColab or use detectionintegration.py and systemintegration.py (you need to pip install requirements_for_localenv.txt and change FILE_PATH and parameters (e.g., fifths, beats, beat_type, staff)) in your local environment.
3. An Web img2Mxml (sheetmusic2musicxml) application is available at https://ui.saaipf.com/app/upload
4. Use, for instance, Logic Pro (https://www.apple.com/jp/logic-pro/), MuseScore (https://musescore.org/), Sibelius First (https://my.avid.com/get/sibelius-first), or xml2sound (soon available at https://ui.saaipf.com/app2/upload) to produce a sound from the resulting xml file.

5. To further train YOLOv5 models (stored at yolov5/weightsstock/) for inference, check each model with specific labels set forth in lines 429 to 474 of makeyolomusicdict/generatedictforxml.py.
6. Each measure training and test data can be created by using enlargemeasures/enlargeeachmeasure.py, labelImg (https://github.com/tzutalin/labelImg), and roboflow (https://roboflow.com).

7. Actual inference results of two sheet music images are provided in musicdata/menuetBach/ or musicdata/sarabandePhotoInclined/ for your reference.

8. img2Mxml is part of bFaaaP (barrier-Free assist as a Pedal) project (https://bfaaap.com) or (https://www.barks.jp/keywords/mf2020_bfaaap.html)

9. You may contact Tomo at info.shishido.and.associates@gmail.com

 

