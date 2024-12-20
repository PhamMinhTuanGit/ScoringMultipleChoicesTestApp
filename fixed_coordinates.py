import cv2
from detect_black_square import detect_black_square_centers
from preprocess import find_angle, rotate
from generate_points import add_points_by_region, add_points_by_region_custom, add_points_by_region_three, add_points_by_region_two
from utils import convert_to_yolo_format, save_to_txt

def fixed_circle(input_image_path, output_file):
    image = cv2.imread(input_image_path)
    height, width = image.shape[:2]

    final_coordinate = []
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
    final_coordinate = add_points_by_region(image, [29, 28, 23, 24], (0.185, 0.11, 0.09, 0.05), (10, 6), (0.025, 0.037), centers, final_coordinate) # SBD
    final_coordinate = add_points_by_region_two(image, [28, 23], (0.28, 0.11, 0.125, 0.05), (10, 3), (0.063, 0.037), centers, final_coordinate) #MDT

    final_coordinate = add_points_by_region(image, [27, 26, 20, 21], (0.245, 0.21, 0.07, 0.05), (10, 4), (0.135, 0.0227), centers, final_coordinate) # Phan1_1
    final_coordinate = add_points_by_region(image, [25, 26, 20, 19], (0.245, 0.21, 0.07, 0.05), (10, 4), (0.135, 0.0227), centers, final_coordinate) # Phan1_2
    final_coordinate = add_points_by_region_three(image, [25, 18, 19], (0.245, 0.21, 0.07, 0.05), (10, 4), (0.135, 0.0227), centers, final_coordinate) # Phan1_3
    final_coordinate = add_points_by_region_three(image, [22, 18, 17], (0.245, 0.21, 0.07, 0.05), (10, 4), (0.135, 0.0227), centers, final_coordinate) # Phan1_4

    final_coordinate = add_points_by_region(image, [21, 20, 14, 16], (0.245, 0.47, 0.07, 0.085), (4, 4), (0.1365, 0.032), centers, final_coordinate) # Phan2_1
    final_coordinate = add_points_by_region(image, [19, 20, 14, 12], (0.245, 0.47, 0.07, 0.085), (4, 4), (0.1365, 0.032), centers, final_coordinate) # Phan2_2
    final_coordinate = add_points_by_region(image, [19, 18, 10, 12], (0.245, 0.47, 0.07, 0.085), (4, 4), (0.1365, 0.032), centers, final_coordinate) # Phan2_3
    final_coordinate = add_points_by_region(image, [17, 18, 10, 8], (0.245, 0.47, 0.07, 0.085), (4, 4), (0.1365, 0.032), centers, final_coordinate) # Phan2_4

    final_coordinate = add_points_by_region_custom(image, [16, 15, 7, 6], (0.36, 0.37, 0.1, 0.04), (10, 4), (0.044, 0.017), centers, final_coordinate) # Phan3_1
    final_coordinate = add_points_by_region_custom(image, [13, 15, 5, 6], (0.36, 0.37, 0.1, 0.04), (10, 4), (0.044, 0.017), centers, final_coordinate) # Phan3_2
    final_coordinate = add_points_by_region_custom(image, [13, 12, 5, 4], (0.36, 0.37, 0.1, 0.04), (10, 4), (0.044, 0.017), centers, final_coordinate) # Phan3_3
    final_coordinate = add_points_by_region_custom(image, [11, 12, 3, 4], (0.36, 0.37, 0.1, 0.04), (10, 4), (0.044, 0.017), centers, final_coordinate) # Phan3_4
    final_coordinate = add_points_by_region_custom(image, [11, 9, 3, 2], (0.36, 0.37, 0.1, 0.04), (10, 4), (0.044, 0.017), centers, final_coordinate) # Phan3_5
    final_coordinate = add_points_by_region_custom(image, [8, 9, 1, 2], (0.36, 0.37, 0.1, 0.04), (10, 4), (0.044, 0.017), centers, final_coordinate) # Phan3_6


    yolo_bboxes = convert_to_yolo_format(final_coordinate, width, height)
    # Thêm class 0 vào đầu mỗi tuple
    yolo_coordinates_with_class = [(0, *coords) for coords in yolo_bboxes]

    save_to_txt(yolo_coordinates_with_class, output_file)
