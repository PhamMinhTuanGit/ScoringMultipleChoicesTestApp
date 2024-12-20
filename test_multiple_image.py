from fixed_coordinates import fixed_circle
from visualize import draw_boxes_from_yolo
import os

folder_path = r"Image\test2\val"
output_file = "test.txt"

with open('error.txt', 'w') as file:
    pass

# Lấy danh sách file .jpg
files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

for file in files:
    input_image_path = os.path.join(folder_path, file)
    #print(input_image_path)
    fixed_circle(input_image_path, output_file)
    #draw_boxes_from_yolo(input_image_path, output_file, 'Image/annotated_image.jpg')