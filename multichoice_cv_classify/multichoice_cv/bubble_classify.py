# quadrilateral.py
from shapely.geometry import Point, Polygon
from visualization import visualize_batches
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
    :param dots: A list of tuples representing (x, y) coordinates of the dots to check
    :return: A list of tuples (dots) that are inside the quadrilateral
    """
    corners=sort_to_convex_quadrilateral(corners)
    # Ensure the corners are sorted correctly (in order of the quadrilateral's path)
    if len(corners) != 4:
        raise ValueError("Exactly 4 corners are required to define a quadrilateral.")
    
    # Create a Polygon object using the corners
    polygon = Polygon(corners)
    
    # Filter dots that are inside the polygon
    inside_dots = [dot for dot in dots if polygon.contains(Point(dot))]
    
    return inside_dots

def classify_batches(dots, eps=0.001):
    """
    Classify dots into batches based on their y-values using DBSCAN.

    :param dots: List of (x, y) tuples representing the dots
    :param eps: Maximum gap between points in a batch (adjust based on data skew)
    :return: List of batches, each containing dots
    """
    # Extract y-values
    y_values = np.array([dot[1] for dot in dots]).reshape(-1, 1)

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=1).fit(y_values)

    # Group dots by their cluster labels
    batches = {}
    for label, dot in zip(clustering.labels_, dots):
        if label not in batches:
            batches[label] = []
        batches[label].append(dot)

    # Convert batches to a sorted list of clusters
    clustered_batches = [batches[label] for label in sorted(batches.keys())]

    # Visualize the clusters using the new function
    visualize_batches(batches)

    return clustered_batches
