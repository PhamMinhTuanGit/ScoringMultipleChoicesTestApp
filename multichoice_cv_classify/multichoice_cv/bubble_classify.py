# quadrilateral.py
from shapely.geometry import Point, Polygon
import math
import numpy as np
from sklearn.cluster import DBSCAN


def sort_to_convex_quadrilateral(dots):
    """
    Sorts a tuple of 4 dots into the order of a convex quadrilateral.

    :param dots: A tuple or list of 4 tuples, each representing (x, y) coordinates of a point
    :return: A list of 4 tuples sorted into a convex quadrilateral order
    """
    if len(dots) != 4:
        raise ValueError("Input must contain exactly 4 dots.")

    # Calculate the centroid of the points
    centroid_x = sum(dot[0] for dot in dots) / 4
    centroid_y = sum(dot[1] for dot in dots) / 4

    # Function to calculate the polar angle of a point relative to the centroid
    def polar_angle(dot):
        x, y = dot
        return math.atan2(y - centroid_y, x - centroid_x)

    # Sort the dots by their polar angle relative to the centroid
    sorted_dots = sorted(dots, key=polar_angle)
    return sorted_dots

def dots_in_quadrilateral(corners, dots):
    """
    Finds all the dots inside the quadrilateral defined by four corners.

    :param corners: A tuple of 4 tuples, each representing (x, y) coordinates of the quadrilateral's corners
    :param dots: A list of tuples representing (flag, x, y, width, height) of the dots
    :return: A list of tuples (dots in full format) that are inside the quadrilateral
    """
    # Ensure the corners are sorted correctly (in order of the quadrilateral's path)
    corners = sort_to_convex_quadrilateral(corners)
    if len(corners) != 4:
        raise ValueError("Exactly 4 corners are required to define a quadrilateral.")
    
    # Create a Polygon object using the corners
    polygon = Polygon(corners)
    
    # Filter dots that are inside the polygon (only checking x and y values)
    inside_dots = [dot for dot in dots if polygon.contains(Point(dot[1], dot[2]))]
    
    return inside_dots

def classify_batches(dots, axis=1, eps=0.001):
    """
    Classify dots into batches based on their axis-values (x or y) using DBSCAN.

    :param dots: List of tuples representing (flag, x, y, width, height) of the dots
    :param axis: 1 for y (default), 0 for x
    :param eps: Maximum gap between points in a batch (adjust based on data skew)
    :return: List of batches, each containing full-format dots, sorted by the value of the first element in each batch
    """
    # Extract the specified axis values (x or y)
    values = np.array([dot[axis + 1] for dot in dots]).reshape(-1, 1)
    if axis==0: eps = 0.005
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=1).fit(values)

    # Group dots by their cluster labels
    batches = {}
    for label, dot in zip(clustering.labels_, dots):
        if label not in batches:
            batches[label] = []
        batches[label].append(dot)

    # Convert batches to a list of clusters
    if axis == 1:
        clustered_batches = [
            sorted(batches[label], key=lambda dot: dot[1])  # Sort dots in each batch by x-value (dot[1])
            for label in sorted(batches.keys())
        ]
    else:
        clustered_batches = [
            sorted(batches[label], key=lambda dot: dot[2])  # Sort dots in each batch by y-value (dot[1])
            for label in sorted(batches.keys())
        ]
    # Sort the batches by the axis value of the first element in each batch
    clustered_batches.sort(key=lambda batch: batch[0][axis + 1])  # Sort by x or y value of the first dot

    return clustered_batches

def process_batches_to_file(clustered_batches, input_string, output_file):
    """
    Processes each batch in clustered_batches to find tuples with a flag of 0, 
    removes the flag, and appends a formatted string to a text file.

    :param clustered_batches: List of batches, where each batch contains full-format dots (flag, x, y, width, height)
    :param input_string: The input string to prepend to the formatted output
    :param output_file: Path to the text file to write the output
    :return: None
    """
    with open(output_file, "a") as file:  # Open the file in append mode
        for batch_num, batch in enumerate(clustered_batches):
            for dot in batch:
                if dot[0] == 0:  # Check if the flag is 0
                    # Remove the flag (dot[1:] gives x, y, width, height)
                    x, y, width, height = dot[1:]
                    
                    # Write the formatted string to the file
                    file.write(f"{input_string}{batch_num+1} {x},{y},{width},{height} ")
    file.close()
