# bubble_detection.py
from sklearn.cluster import DBSCAN
import numpy as np
from bubble_classify import sort_to_convex_quadrilateral
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
    coord_min, coord_max = min(coord1, coord2)-0.002, max(coord1, coord2)+0.002

    # Filter points with coordinates within the range on the chosen axis
    dots_in_range = [
        point for point in centers if coord_min <= point[axis] <= coord_max
    ]

    return dots_in_range
def linearForwardDot(dotA,dotB,k=0.56):
    """
    Linear forward a dot with the start from A

    :param dotA: the first dot a tupple of (x,y)
    :param dotB: the second dot
    :param k: define the scale
    :return a tupple (x,y)
    """
    newdot = (
    k*(dotB[0] - dotA[0]) + dotA[0],  # x-coordinate
    k*(dotB[1] - dotA[1]) + dotA[1]   # y-coordinate
    )
    return newdot

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
    print("The smallest_and second smallest",smallest_and_second_smallest_y)
    # Find 2 nearest points to any point with y < dots_matrix[1][0]
    reference_y = dots_matrix[1][0][1]
    smaller_y_points = [point for point in leftover_points if point[1] < reference_y]
    nearest_points = sorted(smaller_y_points, key=lambda p: abs(p[1] - reference_y))[:2]
    print("nearest point is",nearest_points)
    # Append the specific leftover points to the matrix
    dots_matrix.append(smallest_and_second_smallest_y)
    dots_matrix[5].append(nearest_points[0])
    dots_matrix[5].append(nearest_points[1])
    print("The dotmatrix5 is ",dots_matrix[5])
    return dots_matrix

def getExtractsections(matrix_dots):
    # Initialize 'sections' as a 4x6 2D list
    sections = [[None for _ in range(8)] for _ in range(4)]  # Creates a 4x6 list filled with None
    # Assign values to the sections
    sections[0][0] = sort_to_convex_quadrilateral(matrix_dots[5])
    sections[0][1] = sort_to_convex_quadrilateral((matrix_dots[0][1], matrix_dots[1][4], sections[0][0][1], sections[0][0][2]))
    sections[1][0] = sort_to_convex_quadrilateral((matrix_dots[1][0], matrix_dots[1][1], matrix_dots[2][0], matrix_dots[2][1]))
    sections[1][1] = sort_to_convex_quadrilateral((matrix_dots[1][1], matrix_dots[1][2], matrix_dots[2][1], matrix_dots[2][2]))
    sections[1][2] = sort_to_convex_quadrilateral((matrix_dots[1][2], matrix_dots[1][3], matrix_dots[2][2], matrix_dots[2][3]))
    sections[1][3] = sort_to_convex_quadrilateral((matrix_dots[1][3], matrix_dots[1][4], matrix_dots[2][3], matrix_dots[2][4]))
    sections[2][0] = sort_to_convex_quadrilateral((matrix_dots[2][0], linearForwardDot(matrix_dots[2][0],matrix_dots[2][1]), 
                                                   matrix_dots[3][0],linearForwardDot(matrix_dots[3][0],matrix_dots[3][1])))
    sections[2][1] = sort_to_convex_quadrilateral((matrix_dots[2][1], linearForwardDot(matrix_dots[2][0],matrix_dots[2][1]), 
                                                   matrix_dots[3][2],linearForwardDot(matrix_dots[3][0],matrix_dots[3][2])))
    
    sections[2][2] = sort_to_convex_quadrilateral((matrix_dots[2][1], linearForwardDot(matrix_dots[2][1],matrix_dots[2][2]), 
                                                   matrix_dots[3][2],linearForwardDot(matrix_dots[3][2],matrix_dots[3][4])))
    sections[2][3] = sort_to_convex_quadrilateral((matrix_dots[2][2], linearForwardDot(matrix_dots[2][1],matrix_dots[2][2]), 
                                                   matrix_dots[3][4],linearForwardDot(matrix_dots[3][2],matrix_dots[3][4])))
    
    sections[2][4] = sort_to_convex_quadrilateral((matrix_dots[2][2], linearForwardDot(matrix_dots[2][2],matrix_dots[2][3]), 
                                                   matrix_dots[3][4],linearForwardDot(matrix_dots[3][4],matrix_dots[3][6])))
    sections[2][5] = sort_to_convex_quadrilateral((matrix_dots[2][3], linearForwardDot(matrix_dots[2][2],matrix_dots[2][3]), 
                                                   matrix_dots[3][6],linearForwardDot(matrix_dots[3][4],matrix_dots[3][6])))
    
    sections[2][6] = sort_to_convex_quadrilateral((matrix_dots[2][3], linearForwardDot(matrix_dots[2][3],matrix_dots[2][4]), 
                                                   matrix_dots[3][6],linearForwardDot(matrix_dots[3][6],matrix_dots[3][8])))
    sections[2][7] = sort_to_convex_quadrilateral((matrix_dots[2][4], linearForwardDot(matrix_dots[2][3],matrix_dots[2][4]), 
                                                   matrix_dots[3][8],linearForwardDot(matrix_dots[3][6],matrix_dots[3][8])))
    sections[3][0] = sort_to_convex_quadrilateral((matrix_dots[3][0], matrix_dots[4][0], matrix_dots[3][1], matrix_dots[4][1]))
    sections[3][1] = sort_to_convex_quadrilateral((matrix_dots[3][1], matrix_dots[4][1], matrix_dots[3][3], matrix_dots[4][2]))
    sections[3][2] = sort_to_convex_quadrilateral((matrix_dots[3][3], matrix_dots[4][2], matrix_dots[3][4], matrix_dots[4][3]))
    sections[3][3] = sort_to_convex_quadrilateral((matrix_dots[3][4], matrix_dots[4][3], matrix_dots[3][5], matrix_dots[4][4]))
    sections[3][4] = sort_to_convex_quadrilateral((matrix_dots[3][5], matrix_dots[4][4], matrix_dots[3][7], matrix_dots[4][5]))
    sections[3][5] = sort_to_convex_quadrilateral((matrix_dots[3][7], matrix_dots[4][5], matrix_dots[3][8], matrix_dots[4][6]))

    return sections

if __name__ == "__main__":
    input_image_path = 'IMG_1581_iter_0.jpg'
    input_data = 'IMG_1581_iter_0.txt'
    result_txt_path = 'results_test_2.txt'

