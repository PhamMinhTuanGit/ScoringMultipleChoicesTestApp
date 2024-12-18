# test_bubble_detection.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'multichoice_cv')))

import multichoice_cv.image_processing as image_processing

input_image_path = 'IMG_1581_iter_0.jpg'
output_image_path = 'test_image_processing.jpg'
output_txt_path = 'test_image_processing.txt'
#file_path = 'IMG_1581_iter_0.txt'

centers = image_processing.getNails(input_image_path,output_image_path,output_txt_path)
print("The centers nails is", centers)
