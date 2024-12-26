import cv2
from detect_black_square import detect_black_square_centers
from preprocess import find_angle, rotate
from generate_points import add_points_by_region, add_points_by_region_custom, add_points_by_region_three_1, add_points_by_region_two, add_points_by_region_three_2
from utils import convert_to_yolo_format, save_to_txt, create_matrix

def fixed_circle(input_image_path, output_file):
    image = cv2.imread(input_image_path)
    height, width = image.shape[:2]
    image_size = (width, height)

    final_coordinate = []
    matrix_coordinate = create_matrix()
    centers = detect_black_square_centers(input_image_path)

    # idx_list = [1, 7, 30, 31]

    # # 4 góc tọa độ cụ thể (YOLO normalized format)
    # points_of_interest = [
    #     centers[idx_list[0]-1],  # Point 1
    #     centers[idx_list[1]-1],  # Point 2
    #     centers[idx_list[2]-1],  # Point 3
    #     centers[idx_list[3]-1],  # Point 4
    # ]

    # angle = find_angle(points_of_interest)
    # image = rotate(image, angle, 1)

    # height, width = image.shape[:2]
    # centers = detect_black_square_centers(image)

    # Add the coordinate of each area
    final_coordinate, matrix_coordinate = add_points_by_region(image, [29, 28, 23, 24], (0.175, 0.105, 0.0127, 0.011), (10, 6), (0.105, 0.077), centers, final_coordinate, image_size, matrix_coordinate, 0, 0, width, height) # SBD
    final_coordinate, matrix_coordinate = add_points_by_region_two(image, [28, 29, 24, 23], (0.256, 0.105, 0.0127, 0.011), (10, 3), (0.164, 0.077), centers, final_coordinate, image_size, matrix_coordinate, 0, 1, width, height) #MDT

    final_coordinate, matrix_coordinate = add_points_by_region(image, [27, 26, 20, 21], (0.233, 0.2, 0.016, 0.0105), (10, 4), (0.195, 0.063), centers, final_coordinate, image_size, matrix_coordinate, 1, 0, width, height) # Phan1_1
    final_coordinate, matrix_coordinate = add_points_by_region(image, [26, 25, 19, 20], (0.233, 0.2, 0.016, 0.0105), (10, 4), (0.195, 0.063), centers, final_coordinate, image_size, matrix_coordinate, 1, 1, width, height) # Phan1_2
    final_coordinate, matrix_coordinate = add_points_by_region_three_1(image, [25, 22, 18, 19], (0.233, 0.2, 0.016, 0.0105), (10, 4), (0.195, 0.063), centers, final_coordinate, image_size, matrix_coordinate, 1, 2, width, height) # Phan1_3
    final_coordinate, matrix_coordinate = add_points_by_region_three_2(image, [25, 22, 17, 18], (0.233, 0.2, 0.016, 0.0105), (10, 4), (0.195, 0.063), centers, final_coordinate, image_size, matrix_coordinate, 1, 3, width, height) # Phan1_4

    final_coordinate, matrix_coordinate = add_points_by_region(image, [21, 20, 14, 16], (0.235, 0.475, 0.016, 0.0105), (4, 4), (0.195, 0.105), centers, final_coordinate, image_size, matrix_coordinate, 2, 0, width, height) # Phan2_1
    final_coordinate, matrix_coordinate = add_points_by_region(image, [20, 19, 12, 14], (0.235, 0.475, 0.016, 0.0105), (4, 4), (0.195, 0.105), centers, final_coordinate, image_size, matrix_coordinate, 2, 1, width, height) # Phan2_2
    final_coordinate, matrix_coordinate = add_points_by_region(image, [19, 18, 10, 12], (0.235, 0.475, 0.016, 0.0105), (4, 4), (0.195, 0.105), centers, final_coordinate, image_size, matrix_coordinate, 2, 2, width, height) # Phan2_3
    final_coordinate, matrix_coordinate = add_points_by_region(image, [18, 17, 8, 10], (0.235, 0.475, 0.016, 0.0105), (4, 4), (0.195, 0.105), centers, final_coordinate, image_size, matrix_coordinate, 2, 3, width, height) # Phan2_4

    final_coordinate, matrix_coordinate = add_points_by_region_custom(image, [16, 15, 6, 7], (0.344, 0.37, 0.0158, 0.012), (10, 4), (0.132, 0.046), centers, final_coordinate, image_size, matrix_coordinate, 3, 0, width, height) # Phan3_1
    final_coordinate, matrix_coordinate = add_points_by_region_custom(image, [15, 13, 5, 6], (0.344, 0.37, 0.0158, 0.012), (10, 4), (0.132, 0.046), centers, final_coordinate, image_size, matrix_coordinate, 3, 1, width, height) # Phan3_2
    final_coordinate, matrix_coordinate = add_points_by_region_custom(image, [13, 12, 4, 5], (0.344, 0.37, 0.0158, 0.012), (10, 4), (0.132, 0.046), centers, final_coordinate, image_size, matrix_coordinate, 3, 2, width, height) # Phan3_3
    final_coordinate, matrix_coordinate = add_points_by_region_custom(image, [12, 11, 3, 4], (0.35, 0.38, 0.0158, 0.012), (10, 4), (0.132, 0.044), centers, final_coordinate, image_size, matrix_coordinate, 3, 3, width, height) # Phan3_4
    final_coordinate, matrix_coordinate = add_points_by_region_custom(image, [11, 9, 2, 3], (0.35, 0.38, 0.0158, 0.012), (10, 4), (0.132, 0.044), centers, final_coordinate, image_size, matrix_coordinate, 3, 4, width, height) # Phan3_5
    final_coordinate, matrix_coordinate = add_points_by_region_custom(image, [9, 8, 1, 2], (0.35, 0.38, 0.0158, 0.012), (10, 4), (0.132, 0.044), centers, final_coordinate, image_size, matrix_coordinate, 3, 5, width, height) # Phan3_6


    yolo_bboxes = convert_to_yolo_format(final_coordinate, width, height)
    # Thêm class 0 vào đầu mỗi tuple
    yolo_coordinates_with_class = [(0, *coords) for coords in yolo_bboxes]

    save_to_txt(yolo_coordinates_with_class, output_file)

    return matrix_coordinate
