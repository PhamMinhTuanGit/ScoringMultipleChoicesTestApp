# bubble_detection.py
from sklearn.cluster import DBSCAN
import numpy as np

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
    coord_min, coord_max = min(coord1, coord2), max(coord1, coord2)

    # Filter points with coordinates within the range on the chosen axis
    dots_in_range = [
        point for point in centers if coord_min <= point[axis] <= coord_max
    ]

    return dots_in_range

def getGridmatrix(centers):
    """
    Finds the 5 points with the highest x values, sorts them by y values,
    and draws lines sequentially between consecutive points.

    Appends leftover points from centers (not in dots_matrix) to dots_matrix,
    sorted by 2 highest and 2 lowest y-values.

    :param centers: List of center coordinates in YOLOv8 format [(x, y), ...]
    :param image: The image to draw on
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
        #cv2.line(image, point1, point2, (255, 0, 0), 2)  # Blue line

    for i in range(len(top_5_x_min_sorted_by_y) - 1):
        point1 = top_5_x_min_sorted_by_y[i]
        point2 = top_5_x_min_sorted_by_y[i + 1]
        #cv2.line(image, point1, point2, (255, 0, 0), 2)  # Blue line

    for i in range(len(top_5_x_min_sorted_by_y)):
        point1 = top_5_x_min_sorted_by_y[i]
        point2 = top_5_x_max_sorted_by_y[i]
        # Find dots between these two points
        # Sort from lowest x to hightest
        dots_between = sorted(getDotsbetween(centers, point1, point2,1), key=lambda p:p[0])
        # Add to the matrix
        dots_matrix.append(dots_between)
        #cv2.line(image, point1, point2, (0, 0, 255), 2)  # Red line

    # Complement the lost dot by vector trick 
    dots_buffer=(0,0)
    dots_matrix[1].append(dots_buffer)   
    dots_matrix[1][4]=dots_matrix[1][3]
    dots_matrix[1][3] = (
    dots_matrix[1][4][0] - dots_matrix[2][4][0] + dots_matrix[2][3][0],  # x-coordinate
    dots_matrix[1][4][1] - dots_matrix[2][4][1] + dots_matrix[2][3][1]   # y-coordinate
    )
    #print("new dot is",dots_matrix[1][3]," New vector is: ",dots_matrix[1])

    # Get leftover points not in dots_matrix
    used_points = {tuple(dot) for row in dots_matrix for dot in row}
    leftover_points = [point for point in centers if tuple(point) not in used_points]

    # Sort leftover points by y-values
    leftover_sorted_by_x = sorted(leftover_points, key=lambda p: p[0])
    print("\n The left over is: ", leftover_sorted_by_x )
    dots_matrix.append(leftover_sorted_by_x)  # 2 highest y-values
    
    # Append 2 highest and 2 lowest y-value points to the dots_matrix
    # if len(leftover_sorted_by_x) >= 2:
    #     dots_matrix.append(leftover_sorted_by_x[-2:])  # 2 highest y-values
    # if len(leftover_sorted_by_x) >= 4:
    #     dots_matrix.append(leftover_sorted_by_x[:2])  # 2 lowest y-values

    #cv2.imwrite('final_output.jpg', image)

    return dots_matrix

def getExtractsections(matrix_dots):
    # Initialize 'sections' as a 4x6 2D list
    sections = [[None for _ in range(6)] for _ in range(4)]  # Creates a 4x6 list filled with None

    # Assign values to the sections
    sections[0][0] = matrix_dots[5]
    sections[0][1] = (matrix_dots[0][1], matrix_dots[1][4], matrix_dots[5][2], matrix_dots[5][3],)
    sections[1][0] = (matrix_dots[1][0], matrix_dots[1][1], matrix_dots[2][0], matrix_dots[2][1])
    sections[1][1] = (matrix_dots[1][1], matrix_dots[1][2], matrix_dots[2][1], matrix_dots[2][2])
    sections[1][2] = (matrix_dots[1][2], matrix_dots[1][3], matrix_dots[2][2], matrix_dots[2][3])
    sections[1][3] = (matrix_dots[1][3], matrix_dots[1][4], matrix_dots[2][3], matrix_dots[2][4])
    sections[2][0] = (matrix_dots[2][0], matrix_dots[2][1], matrix_dots[3][0], matrix_dots[3][2])
    sections[2][1] = (matrix_dots[2][1], matrix_dots[2][2], matrix_dots[3][2], matrix_dots[3][4])
    sections[2][2] = (matrix_dots[2][2], matrix_dots[2][3], matrix_dots[3][4], matrix_dots[3][6])
    sections[2][3] = (matrix_dots[2][3], matrix_dots[2][4], matrix_dots[3][6], matrix_dots[3][8])
    sections[3][0] = (matrix_dots[3][0], matrix_dots[4][0], matrix_dots[3][1], matrix_dots[4][1])
    sections[3][1] = (matrix_dots[3][1], matrix_dots[4][1], matrix_dots[3][3], matrix_dots[4][2])
    sections[3][2] = (matrix_dots[3][3], matrix_dots[4][2], matrix_dots[3][4], matrix_dots[4][3])
    sections[3][3] = (matrix_dots[3][4], matrix_dots[4][3], matrix_dots[3][5], matrix_dots[4][4])
    sections[3][4] = (matrix_dots[3][5], matrix_dots[4][4], matrix_dots[3][7], matrix_dots[4][5])
    sections[3][5] = (matrix_dots[3][7], matrix_dots[4][5], matrix_dots[3][8], matrix_dots[4][6])

    return sections

