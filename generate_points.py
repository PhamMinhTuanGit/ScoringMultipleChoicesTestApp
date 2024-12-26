from utils import generate_grid_coordinates, cut_image_from_points, yolo_to_pixel, convert_to_yolo_format
from detect_black_square import detect_black_squares
from draw_black_square import draw_rectangles
import cv2
from warp_perspective import warp_perspective_with_reversed_boxes

def insert_element(matrix, row, col, element):
    """
    Chèn phần tử mới vào ma trận ô tròn.

    Args:
        matrix (list): Ma trận ban đầu.
        row (int): Chỉ số hàng (bắt đầu từ 0).
        col (int): Chỉ số cột (bắt đầu từ 0).
        element (list): Phần tử cần chèn (list 5 phần tử).

    Returns:
        list: Ma trận sau khi đã chèn phần tử.
    """
    # Chèn phần tử vào vị trí chỉ định
    matrix[row][col] = element
    return matrix

def add_points_by_region(image, idx_list, start_coord, grid_size, cell_spacing, centers, final_coordinate, image_size, matrix_coordinate, row, col, width, height):
    points_of_interest = [
        centers[idx_list[0]-1],  # Point 1
        centers[idx_list[1]-1],  # Point 2
        centers[idx_list[2]-1],  # Point 3
        centers[idx_list[3]-1],  # Point 4
    ]

    x_min, y_min, x_max, y_max = cut_image_from_points(image, points_of_interest)

    yolo_boxes = generate_grid_coordinates(start_coord, grid_size, cell_spacing)

    pixel_points = yolo_to_pixel(points_of_interest, image_size)
    # Perform warping and reverse transformation
    warped_image, transformed_boxes_original = warp_perspective_with_reversed_boxes(
        image, pixel_points, yolo_boxes, image_size
    )

    region_lst_temp = []
    for box in transformed_boxes_original:
        final_coordinate.append(box)
        region_lst_temp.append(box)
    
    matrix_coordinate = insert_element(matrix_coordinate, row, col, convert_to_yolo_format(region_lst_temp, width, height))
    
    return final_coordinate, matrix_coordinate

    # size_x = x_max - x_min
    # size_y = y_max - y_min
    # region_lst_temp = []

    # for box in yolo_boxes:
    #   final_coordinate.append((x_min + size_x*box[0], y_min + size_y*box[1], int(box[2]*(size_x)), box[3]*size_y))
    #   region_lst_temp.append((x_min + size_x*box[0], y_min + size_y*box[1], int(box[2]*(size_x)), box[3]*size_y))

    # matrix_coordinate = insert_element(matrix_coordinate, row, col, convert_to_yolo_format(region_lst_temp, width, height))
    
    # return final_coordinate, matrix_coordinate

def add_points_by_region_two(image, idx_list, start_coord, grid_size, cell_spacing, centers, final_coordinate, image_size, matrix_coordinate, row, col, width, height):
    points_of_interest = [
        centers[idx_list[0]-1],  # Point 1
        (centers[idx_list[0]-1][0]+(abs(centers[idx_list[0]-1][0]-centers[idx_list[1]-1][0])*2/3), centers[idx_list[0]-1][1]+(abs(centers[idx_list[0]-1][1]-centers[idx_list[1]-1][1])*2/3)),  # Point 3
        (centers[idx_list[3]-1][0]+(abs(centers[idx_list[3]-1][0]-centers[idx_list[2]-1][0])*2/3), centers[idx_list[3]-1][1]+(abs(centers[idx_list[3]-1][1]-centers[idx_list[2]-1][1])*2/3)),   # Point 4
        centers[idx_list[3]-1],  # Point 4
    ]

    x_min, y_min, x_max, y_max = cut_image_from_points(image, points_of_interest)

    yolo_boxes = generate_grid_coordinates(start_coord, grid_size, cell_spacing)

    pixel_points = yolo_to_pixel(points_of_interest, image_size)
    # Perform warping and reverse transformation
    warped_image, transformed_boxes_original = warp_perspective_with_reversed_boxes(
        image, pixel_points, yolo_boxes, image_size
    )

    region_lst_temp = []
    for box in transformed_boxes_original:
        final_coordinate.append(box)
        region_lst_temp.append(box)
    
    matrix_coordinate = insert_element(matrix_coordinate, row, col, convert_to_yolo_format(region_lst_temp, width, height))
    
    return final_coordinate, matrix_coordinate

    # size_x = x_max - x_min
    # size_y = y_max - y_min
    # region_lst_temp = []

    # for box in yolo_boxes:
    #   final_coordinate.append((x_min + size_x*box[0], y_min + size_y*box[1], int(box[2]*(size_x)), box[3]*size_y))
    #   region_lst_temp.append((x_min + size_x*box[0], y_min + size_y*box[1], int(box[2]*(size_x)), box[3]*size_y))

    # matrix_coordinate = insert_element(matrix_coordinate, row, col, convert_to_yolo_format(region_lst_temp, width, height))
    
    # return final_coordinate, matrix_coordinate

def add_points_by_region_three_1(image, idx_list, start_coord, grid_size, cell_spacing, centers, final_coordinate, image_size, matrix_coordinate, row, col, width, height):
    points_of_interest = [
        centers[idx_list[0]-1],  # Point 1
        ((centers[idx_list[1]-1][0]+centers[idx_list[0]-1][0])/2, (centers[idx_list[1]-1][1]+centers[idx_list[0]-1][1])/2),   # Point 2
        centers[idx_list[2]-1],  # Point 3
        centers[idx_list[3]-1],  # Point 4
    ]

    x_min, y_min, x_max, y_max = cut_image_from_points(image, points_of_interest)

    yolo_boxes = generate_grid_coordinates(start_coord, grid_size, cell_spacing)

    pixel_points = yolo_to_pixel(points_of_interest, image_size)
    # Perform warping and reverse transformation
    warped_image, transformed_boxes_original = warp_perspective_with_reversed_boxes(
        image, pixel_points, yolo_boxes, image_size
    )

    region_lst_temp = []
    for box in transformed_boxes_original:
        final_coordinate.append(box)
        region_lst_temp.append(box)
    
    matrix_coordinate = insert_element(matrix_coordinate, row, col, convert_to_yolo_format(region_lst_temp, width, height))
    
    return final_coordinate, matrix_coordinate

    # size_x = x_max - x_min
    # size_y = y_max - y_min
    # region_lst_temp = []

    # for box in yolo_boxes:
    #   final_coordinate.append((x_min + size_x*box[0], y_min + size_y*box[1], int(box[2]*(size_x)), box[3]*size_y))
    #   region_lst_temp.append((x_min + size_x*box[0], y_min + size_y*box[1], int(box[2]*(size_x)), box[3]*size_y))

    # matrix_coordinate = insert_element(matrix_coordinate, row, col, convert_to_yolo_format(region_lst_temp, width, height))
    
    # return final_coordinate, matrix_coordinate

def add_points_by_region_three_2(image, idx_list, start_coord, grid_size, cell_spacing, centers, final_coordinate, image_size, matrix_coordinate, row, col, width, height):
    points_of_interest = [
        ((centers[idx_list[1]-1][0]+centers[idx_list[0]-1][0])/2, (centers[idx_list[1]-1][1]+centers[idx_list[0]-1][1])/2),   # Point 1
        centers[idx_list[1]-1],  # Point 2
        centers[idx_list[2]-1],  # Point 3
        centers[idx_list[3]-1],  # Point 4
    ]

    x_min, y_min, x_max, y_max = cut_image_from_points(image, points_of_interest)

    yolo_boxes = generate_grid_coordinates(start_coord, grid_size, cell_spacing)

    pixel_points = yolo_to_pixel(points_of_interest, image_size)
    # Perform warping and reverse transformation
    warped_image, transformed_boxes_original = warp_perspective_with_reversed_boxes(
        image, pixel_points, yolo_boxes, image_size
    )

    region_lst_temp = []
    for box in transformed_boxes_original:
        final_coordinate.append(box)
        region_lst_temp.append(box)
    
    matrix_coordinate = insert_element(matrix_coordinate, row, col, convert_to_yolo_format(region_lst_temp, width, height))
    
    return final_coordinate, matrix_coordinate

def add_points_by_region_custom(image, idx_list, start_coord, grid_size, cell_spacing, centers, final_coordinate, image_size, matrix_coordinate, row, col, width, height):
    points_of_interest = [
        centers[idx_list[0]-1],  # Point 1
        centers[idx_list[1]-1],  # Point 2
        centers[idx_list[2]-1],  # Point 3
        centers[idx_list[3]-1],  # Point 4
    ]

    x_min, y_min, x_max, y_max = cut_image_from_points(image, points_of_interest)

    yolo_boxes = generate_grid_coordinates(start_coord, grid_size, cell_spacing)

    yolo_boxes.insert(0, (0.6462, 0.3153, 0.0158, 0.012)) # Comma mark 2
    yolo_boxes.insert(0, (0.4862, 0.3153, 0.0158, 0.012)) # Comma mark 1
    yolo_boxes.insert(0, (0.351, 0.261, 0.0158, 0.012)) # Minus mark
    
    pixel_points = yolo_to_pixel(points_of_interest, image_size)
    # Perform warping and reverse transformation
    warped_image, transformed_boxes_original = warp_perspective_with_reversed_boxes(
        image, pixel_points, yolo_boxes, image_size
    )

    region_lst_temp = []
    for box in transformed_boxes_original:
        final_coordinate.append(box)
        region_lst_temp.append(box)

    # size_x = x_max - x_min
    # size_y = y_max - y_min
    # for box in yolo_boxes:
    #   final_coordinate.append((x_min + size_x*box[0], y_min + size_y*box[1], int(box[2]*(size_x)), box[3]*size_y))
    
    matrix_coordinate = insert_element(matrix_coordinate, row, col, convert_to_yolo_format(region_lst_temp, width, height))
    
    return final_coordinate, matrix_coordinate