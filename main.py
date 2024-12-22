from fixed_coordinates import fixed_circle
from visualize import draw_boxes_from_yolo

input_image_path = r"D:\THO\Bach_Khoa\AI Challenge\Data\testset1\testset1\images\IMG_3960_iter_128.jpg"
output_file = "test.txt"

print(input_image_path)
fixed_circle(input_image_path, output_file)
draw_boxes_from_yolo(input_image_path, output_file, 'Image/annotated_image.jpg')