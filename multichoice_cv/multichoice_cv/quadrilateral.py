# quadrilateral.py
from shapely.geometry import Point, Polygon

def bubbles_inside_quadrilateral(bubbles, quadrilateral):
    """
    Check if a point is inside a quadrilateral.
    :param point: Tuple (x, y).
    :param quadrilateral: Tuple of 4 tuples, each representing (x, y) coordinates of the corners.
    :return: True if the point is inside the quadrilateral, False otherwise.
    """
    polygon = Polygon(quadrilateral)
    inside_bubbles = [bubble for bubble in bubbles if polygon.contains(Point(bubble))]
    return inside_bubbles
