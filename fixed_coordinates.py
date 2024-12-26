import cv2
from detect_black_square import detect_black_square_centers, detect_black_squares
from preprocess import find_angle, rotate
from generate_points import add_points_by_region, add_points_by_region_custom, add_points_by_region_three_1, add_points_by_region_two, add_points_by_region_three_2
from utils import convert_to_yolo_format, save_to_txt, create_matrix, yolo_to_pixel, add_missing_corners
from draw_black_square import draw_rectangles
from warp_perspective import warp_perspective_for_detect
from visualize import draw_boxes_from_list
def fixed_circle(input_image_path, output_file):
    image = cv2.imread(input_image_path)
    cv2.imwrite("Image/raw_image.jpg", image)
    height, width = image.shape[:2]
    image_size = (width, height)
    print("Shape: ", image_size)

    final_coordinate = []
    matrix_coordinate = create_matrix()

    corner_squares = detect_black_squares(image, 35, 35)
    #print(len(corner_squares))

    pixel_points = yolo_to_pixel(corner_squares, image_size)
    if len(corner_squares) <= 3:
        pixel_points = add_missing_corners(pixel_points)

    sorted_by_y = sorted(pixel_points, key=lambda t: t[1]) # Sắp xếp danh sách ban đầu theo giá trị y tăng dần
    group1 = sorted_by_y[:2]  # 2 phần tử có y nhỏ nhất
    group2 = sorted_by_y[2:]  # Các phần tử còn lại

    group1_sorted = sorted(group1, key=lambda t: t[0])  # x tăng dần
    group2_sorted = sorted(group2, key=lambda t: t[0], reverse=True)  # x giảm dần

    pixel_points = group1_sorted + group2_sorted

    #print("4 corners old: ", pixel_points)
    pixel_points = [
        (x - 60, y - 60) if i == 0 else 
        (x + 60, y - 60) if i == 1 else 
        (x + 60, y + 60) if i == 2 else 
        (x - 60, y + 60)
        for i, (x, y) in enumerate(pixel_points)
    ]
    #print("4 corners new: ", pixel_points)
    # draw_rectangles(input_image_path, corner_squares, 20, 20, "Image/Four_corners.jpg")

    #centers = detect_black_square_centers(input_image_path)
    center_pixel = warp_perspective_for_detect(image, pixel_points, image_size)
    center_yolo = []
    for bbox in center_pixel:
        center_x, center_y = bbox
        # Normalize the values
        center_x_normalized = center_x / width
        center_y_normalized = center_y / height
        # Add to the YOLO formatted list
        center_yolo.append((center_x_normalized, center_y_normalized))
    centers = center_yolo

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
    final_coordinate, matrix_coordinate = add_points_by_region(image, [29, 28, 23, 24], (0.175, 0.1055, 0.0127, 0.011), (10, 6), (0.105, 0.076), centers, final_coordinate, image_size, matrix_coordinate, 0, 0, width, height) # SBD
    final_coordinate, matrix_coordinate = add_points_by_region_two(image, [28, 29, 24, 23], (0.256, 0.1055, 0.0127, 0.011), (10, 3), (0.164, 0.076), centers, final_coordinate, image_size, matrix_coordinate, 0, 1, width, height) #MDT

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

def arrange_elements(elements, number_of_elements, axis):
    """
    Arrange elements into rows based on number_of_elements and specified axis.

    Args:
        elements (list of tuples): List of tuples (x, y, width, height) to arrange.
        number_of_elements (int): Number of elements per row.
        axis (str): Axis to arrange by, 'x' or 'y'.

    Returns:
        list of list: A list of rows, where each row is a list of tuples.
    """
    if axis not in ('x', 'y'):
        raise ValueError("Axis must be either 'x' or 'y'.")

    # Group elements into rows of number_of_elements each
    rows = [elements[i:i + number_of_elements] for i in range(0, len(elements), number_of_elements)]

    # If axis is 'y', compute the transpose of the rows
    if axis == 'y':
        rows = list(map(list, zip(*rows)))

    return rows

def split_and_group_elements(elements):
    """
    Split a list of tuples into sections based on blocks of 4 elements.

    Args:
        elements (list of tuples): List of tuples (x, y, w, h).

    Returns:
        tuple: (list of section_2_1, list of section_2_2)
    """
    section_2_1 = []
    section_2_2 = []

    for i in range(0, len(elements), 4):
        block = elements[i:i + 4]
        if len(block) == 4:
            section_2_1.append(block[:2])
            section_2_2.append(block[2:])

    return section_2_1, section_2_2

def write_result(section_name, class_name,i, dot,file_name="result_update.txt"):
    """
    Write formatted section data to a text file.

    Args:
        section_name (int): Section number.
        class_name (int): Class number.
        i (int): Index of the row in the section.
        dot (tuple): (x, y, w, h) of the dot.
        file_name (str, optional): Name of the output file. Defaults to "result_update.txt".

    Writes:
        Data in the format specific to the section.
    """
    with open(file_name, 'a') as file:
        (x, y, w, h) = dot
        print("section_name is",section_name)
        print("dot is: ",(x, y, w, h))
        if section_name == 0 and class_name == 0:
            file.write(f" SBD{i+1} {x},{y},{w},{h}")
        elif section_name == 0 and class_name == 1:
            file.write(f" MDT{i+1} {x},{y},{w},{h}")
        elif section_name == 1 and class_name == 0:
            file.write(f" 1.{i+1} {x},{y},{w},{h}")
        elif section_name == 1 and class_name == 1:
            file.write(f" 1.{i+11} {x},{y},{w},{h}")
        elif section_name == 1 and class_name == 2:
            file.write(f" 1.{i+21} {x},{y},{w},{h}")
        elif section_name == 1 and class_name == 3:
            file.write(f" 1.{i+31} {x},{y},{w},{h}")
        elif section_name == 2 and class_name == 0:
            file.write(f" 2.1.{chr(ord('a') + i)} {x},{y},{w},{h}")
        elif section_name == 2 and class_name == 1:
            file.write(f" 2.2.{chr(ord('a') + i)} {x},{y},{w},{h}")
        elif section_name == 2 and class_name == 2:
            file.write(f" 2.3.{chr(ord('a') + i)} {x},{y},{w},{h}")
        elif section_name == 2 and class_name == 3:
            file.write(f" 2.4.{chr(ord('a') + i)} {x},{y},{w},{h}")
        elif section_name == 2 and class_name == 4:
            file.write(f" 2.5.{chr(ord('a') + i)} {x},{y},{w},{h}")
        elif section_name == 2 and class_name == 5:
            file.write(f" 2.6.{chr(ord('a') + i)} {x},{y},{w},{h}")
        elif section_name == 2 and class_name == 6:
            file.write(f" 2.7.{chr(ord('a') + i)} {x},{y},{w},{h}")
        elif section_name == 2 and class_name == 7:
            file.write(f" 2.8.{chr(ord('a') + i)} {x},{y},{w},{h}")
        else:
            file.write(f" {x},{y},{w},{h}")
def cluster_section(matrix_coordinate):
    clustered_matrix = [[None for _ in range(8)] for _ in range(4)]

    # Assign values to the clustered_matrix based on your logic
    clustered_matrix[0][0] = arrange_elements(matrix_coordinate[0][0], 6, 'y')
    clustered_matrix[0][1] = arrange_elements(matrix_coordinate[0][1], 6, 'y')
    clustered_matrix[1][0] = arrange_elements(matrix_coordinate[1][0], 4, 'x')
    clustered_matrix[1][1] = arrange_elements(matrix_coordinate[1][1], 4, 'x')
    clustered_matrix[1][2] = arrange_elements(matrix_coordinate[1][2], 4, 'x')
    clustered_matrix[1][3] = arrange_elements(matrix_coordinate[1][3], 4, 'x')
    clustered_matrix[2][0], clustered_matrix[2][1] = split_and_group_elements(matrix_coordinate[2][0])
    clustered_matrix[2][2], clustered_matrix[2][3] = split_and_group_elements(matrix_coordinate[2][1])
    clustered_matrix[2][4], clustered_matrix[2][5] = split_and_group_elements(matrix_coordinate[2][2])
    clustered_matrix[2][6], clustered_matrix[2][7] = split_and_group_elements(matrix_coordinate[2][3])
    clustered_matrix[3][0] = matrix_coordinate[3][0]
    clustered_matrix[3][1] = matrix_coordinate[3][1]
    clustered_matrix[3][2] = matrix_coordinate[3][2]
    clustered_matrix[3][3] = matrix_coordinate[3][3]
    return clustered_matrix

if __name__ == "__main__":
    input_image_path = "IMG_1581_iter_1.jpg"
    output_file = "fix_circle.txt"
    matrix_coordinate = fixed_circle(input_image_path, output_file)
    clustered_matrix = cluster_section(matrix_coordinate)
    print(clustered_matrix[0][0][0])
    
    # sau moi lan phan loai se goi ham nay
    write_result(2,1,0,clustered_matrix[2][1][0][0])
