import cv2
from draw_black_square import draw_rectangles
import numpy as np

ground_truth_lst = [
    (0.9157, 0.9289), (0.7707, 0.9248), (0.6284, 0.9216),
    (0.4878, 0.9191), (0.3481, 0.9175), (0.2071, 0.9162),
    (0.0661, 0.9153), (0.9153, 0.6709), (0.7725, 0.6687),
    (0.7016, 0.6677), (0.631, 0.6668), (0.4909, 0.6652),
    (0.3508, 0.6639), (0.2812, 0.6636), (0.2106, 0.663),
    (0.0687, 0.6617), (0.9171, 0.5446), (0.7033, 0.5421),
    (0.4922, 0.5405), (0.282, 0.5386), (0.0696, 0.5363),
    (0.9206, 0.3431), (0.831, 0.3269), (0.6865, 0.3253),
    (0.4936, 0.3405), (0.282, 0.3386), (0.0718, 0.3377),
    (0.8337, 0.1308), (0.6887, 0.1301), (0.9255, 0.0746),
    (0.0723, 0.0727)
]

def match_coordinates(ground_truth_lst, predict_lst, k=0.01):
    """
    Compare coordinates in ground_truth_lst with predict_lst and generate result_lst.
    
    Args:
        ground_truth_lst (list of tuples): List of ground truth coordinates (x, y).
        predict_lst (list of tuples): List of predicted coordinates (x, y).
        k (int): Maximum allowable deviation between coordinates.
    
    Returns:
        list: result_lst containing coordinates from predict_lst or ground_truth_lst.
    """
    result_lst = []  # Initialize the result list
    
    for gt_x, gt_y in ground_truth_lst:  # Iterate through ground truth coordinates
        matched = False  # Flag to check if a match is found
        
        for pred_x, pred_y in predict_lst:  # Iterate through predicted coordinates
            # Check if the predicted coordinate is within the allowable deviation
            if abs(gt_x - pred_x) <= k and abs(gt_y - pred_y) <= k:
                result_lst.append((pred_x, pred_y))  # Add the predicted coordinate to result_lst
                matched = True  # Mark as matched
                break  # Exit the inner loop
        
        if not matched:  # If no match is found
            result_lst.append((gt_x, gt_y))  # Add the ground truth coordinate to result_lst
    
    return result_lst  # Return the final result list

def getDotsbetween(centers, dot1, dot2, axis=1):
    """
    Finds all dots with coordinates between the values of two given dots along a specified axis.

    :param centers: List of all points in the format [(x, y), ...]
    :param dot1: The first reference dot as (x, y)
    :param dot2: The second reference dot as (x, y)
    :param axis: The axis to filter by ('x' or 'y')
    :return: List of dots with coordinates between dot1 and dot2 on the specified axis
    """
    if axis not in (0, 1):
        raise ValueError("Invalid axis. Choose 'x' or 'y'.")
    
    # Extract the values for the chosen axis
    coord1, coord2 = dot1[axis], dot2[axis]

    # Determine the range (min and max values on the chosen axis)
    coord_min, coord_max = min(coord1, coord2)-0.001, max(coord1, coord2)+0.001

    # Filter points with coordinates within the range on the chosen axis
    dots_in_range = [
        point for point in centers if coord_min <= point[axis] <= coord_max
    ]

    return dots_in_range

def flattenMatrix(matrix):
    """
    Flattens a matrix of tuples (x, y) into a single list of tuples.

    :param matrix: A 2D list (matrix) where each element is a tuple (x, y)
    :return: A flattened list of tuples
    """
    return [item for row in matrix for item in row]

def getGridmatrix(centers):
    """
    Finds the 5 points with the highest x values, sorts them by y values,
    and draws lines sequentially between consecutive points.

    Appends leftover points from centers (not in dots_matrix) to dots_matrix,
    with specific conditions for inclusion based on y-values.

    :param centers: List of center coordinates in YOLOv8 format [(x, y), ...]
    :return: Dots matrix containing points between drawn lines and leftover points
    """
    if len(centers) < 2:
        print("Not enough points to draw lines.")
        return []

    # Find the 5 points with the highest and lowest x values
    top_5_x_max = sorted(centers, key=lambda p: p[0], reverse=True)[:5]
    top_5_x_min = sorted(centers, key=lambda p: p[0])[:5]

    # Sort the points by their y values
    top_5_x_max_sorted_by_y = sorted(top_5_x_max, key=lambda p: p[1])
    top_5_x_min_sorted_by_y = sorted(top_5_x_min, key=lambda p: p[1])

    # Initialize the matrix to store dots found between pairs
    dots_matrix = []

    # Draw lines sequentially between the sorted points
    for i in range(len(top_5_x_max_sorted_by_y) - 1):
        point1 = top_5_x_max_sorted_by_y[i]
        point2 = top_5_x_max_sorted_by_y[i + 1]

    for i in range(len(top_5_x_min_sorted_by_y) - 1):
        point1 = top_5_x_min_sorted_by_y[i]
        point2 = top_5_x_min_sorted_by_y[i + 1]

    for i in range(len(top_5_x_min_sorted_by_y)):
        point1 = top_5_x_min_sorted_by_y[i]
        point2 = top_5_x_max_sorted_by_y[i]
        # Find dots between these two points
        # Sort from lowest x to highest
        dots_between = sorted(getDotsbetween(centers, point1, point2, 1), key=lambda p: p[0])
        # Add to the matrix
        dots_matrix.append(dots_between)

    # Complement the lost dot by vector trick
    dots_buffer = (0, 0)
    dots_matrix[1].append(dots_buffer)
    dots_matrix[1][4] = dots_matrix[1][3]

    dots_matrix[1][3] = (
        dots_matrix[1][4][0] - dots_matrix[2][4][0] + dots_matrix[2][3][0],  # x-coordinate
        dots_matrix[1][4][1] - dots_matrix[2][4][1] + dots_matrix[2][3][1]   # y-coordinate
    )

    # Get leftover points not in dots_matrix
    used_points = {tuple(dot) for row in dots_matrix for dot in row}
    leftover_points = [point for point in centers if tuple(point) not in used_points]

    # Sort leftover points by y-values
    leftover_sorted_by_y = sorted(leftover_points, key=lambda p: p[1])

    # Get the smallest and second smallest y-value points
    smallest_and_second_smallest_y = leftover_sorted_by_y[:2] if len(leftover_sorted_by_y) >= 2 else []
    #print("The smallest_and second smallest",smallest_and_second_smallest_y)

    # Find 2 nearest points to any point with y < dots_matrix[1][0]
    reference_y = dots_matrix[1][0][1]
    smaller_y_points = [point for point in leftover_points if point[1] < reference_y]
    nearest_points = sorted(smaller_y_points, key=lambda p: abs(p[1] - reference_y))[:2]
    #print("nearest point is",nearest_points)

    # Append the specific leftover points to the matrix
    dots_matrix.append(smallest_and_second_smallest_y)
    dots_matrix[5].append(nearest_points[0])
    dots_matrix[5].append(nearest_points[1])
    #print("The dotmatrix5 is ",dots_matrix[5])

    return dots_matrix


def detect_black_square_centers(image_path):
    """
    Detects black squares in a given image, saves the output, and displays numbered IDs.

    :param image: Input image
    :return: List of centers of the black squares in YOLOv8 format
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold để phát hiện vùng đen
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("Image/thresh.jpg", thresh)

    # Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    image_height, image_width = image.shape[:2]

    idx = 0  # Khởi tạo bộ đếm
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Kiểm tra contour có 4 đỉnh
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Kiểm tra tỉ lệ và kích thước khung hình
            if 0.8 < aspect_ratio < 1.2 and w > 20 and h > 20:
                roi = thresh[y:y+h, x:x+w]
                black_pixels = cv2.countNonZero(roi)
                total_pixels = w * h
                fill_ratio = black_pixels / total_pixels

                # Kiểm tra mật độ màu đen
                if fill_ratio > 0.9:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])

                        yolov8_x = center_x / image_width
                        yolov8_y = center_y / image_height

                        centers.append((yolov8_x, yolov8_y))
    print(len(centers))

    output_path = "Image/Draw_square.jpg"
    draw_rectangles(image_path, centers, 40, 40, output_path)

    gridmatrix=getGridmatrix(centers)
    flaten_matrix=flattenMatrix(gridmatrix)
    #print("Flatten: ", flaten_matrix)

    # Sắp xếp các điểm theo tọa độ y giảm dần, nếu y bằng nhau thì theo x giảm dần
    centers_sorted = sorted(flaten_matrix, key=lambda x: (x[1], x[0]), reverse=True)

    # Tạo các nhóm điểm theo yêu cầu:
    group_1 = centers_sorted[:7]   # 7 điểm có y lớn nhất
    group_2 = centers_sorted[7:16] # 9 điểm có y lớn tiếp theo
    group_3 = centers_sorted[16:21] # 5 điểm có y lớn tiếp theo
    group_4 = centers_sorted[21:28] # 7 điểm có y lớn tiếp theo
    group_5 = centers_sorted[28:30] # 2 điểm có y lớn tiếp theo
    group_6 = centers_sorted[30:]   # 2 điểm còn lại

    # Đảm bảo các nhóm đều đã được sắp xếp theo x giảm dần
    group_1_sorted = sorted(group_1, key=lambda x: x[0], reverse=True)
    group_2_sorted = sorted(group_2, key=lambda x: x[0], reverse=True)
    group_3_sorted = sorted(group_3, key=lambda x: x[0], reverse=True)
    group_4_sorted = sorted(group_4, key=lambda x: x[0], reverse=True)
    group_5_sorted = sorted(group_5, key=lambda x: x[0], reverse=True)
    group_6_sorted = sorted(group_6, key=lambda x: x[0], reverse=True)

    del group_4_sorted[2] # Remove virtual point

    # Kết hợp tất cả các nhóm lại theo thứ tự đã yêu cầu
    final_sorted_centers = group_1_sorted + group_2_sorted + group_3_sorted + group_4_sorted + group_5_sorted + group_6_sorted
    #print("Centers of black squares (YOLOv8 format):", final_sorted_centers)

    #final_centers = match_coordinates(ground_truth_lst, final_sorted_centers, 0.01)
    final_centers = final_sorted_centers

    output_path = "Image/Draw_square.jpg"
    draw_rectangles(image_path, final_centers, 40, 40, output_path)

    #print(len(final_sorted_centers))
    if len(final_centers) != 31:
        #print("##################################    ERROR     #############################")
        print(len(final_sorted_centers))
        with open('error.txt', 'a') as file:
            file.write(image_path + '\n')  # Ghi thêm một dòng mới

    return final_centers
