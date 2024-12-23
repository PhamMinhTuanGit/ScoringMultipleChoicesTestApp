from fixed_coordinates import fixed_circle
from visualize import draw_boxes_from_txt, draw_boxes_from_list

input_image_path = r"Image\IMG_3958_iter_219.jpg"
output_file = "test.txt"

print(input_image_path)
matrix_coordinate = fixed_circle(input_image_path, output_file)
draw_boxes_from_txt(input_image_path, output_file, 'Image/annotated_image2.jpg')
draw_boxes_from_list(input_image_path, matrix_coordinate[2][1], "Image/Draw_region.jpg")