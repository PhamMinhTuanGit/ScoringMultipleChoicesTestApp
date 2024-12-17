class Point:
    """
    Represents a 2D point with x and y coordinates.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other):
        """
        Calculates the distance to another Point.
        """
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"


class Line:
    """
    Represents a line defined by two points.
    """
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def length(self):
        """
        Calculates the length of the line.
        """
        return self.point1.distance_to(self.point2)

    def slope(self):
        """
        Calculates the slope of the line.
        """
        if self.point2.x == self.point1.x:
            return None  # Undefined slope (vertical line)
        return (self.point2.y - self.point1.y) / (self.point2.x - self.point1.x)

    def __repr__(self):
        return f"Line({self.point1}, {self.point2})"


class GeometryUtils:
    """
    Utility class for general geometric calculations.
    """
    @staticmethod
    def is_point_in_range(point, y_min, y_max):
        """
        Checks if a point's y-coordinate is within a specified range.
        """
        return y_min <= point.y <= y_max

    @staticmethod
    def sort_points_by_y(points):
        """
        Sorts a list of points based on their y-coordinate.
        """
        return sorted(points, key=lambda p: p.y)
