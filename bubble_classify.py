from shapely.geometry import Point, Polygon
import math
import numpy as np
from sklearn.cluster import DBSCAN
from utilities import append_to_file
import torch

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

class BubbleClassifier:
    def __init__(self, grid_section, dots):
        self.grid_section = grid_section
        self.dots = dots
        self.bubble_coordinates_tensor = None

    def dots_in_quadrilateral(self, corners):
        """
        Finds all the dots inside the quadrilateral defined by four corners.

        :param corners: A tuple of 4 tuples, each representing (x, y) coordinates of the quadrilateral's corners
        :return: A list of tuples (dots in full format) that are inside the quadrilateral
        """
        # Ensure the corners are sorted correctly (in order of the quadrilateral's path)
        corners = sort_to_convex_quadrilateral(corners)
        if len(corners) != 4:
            raise ValueError("Exactly 4 corners are required to define a quadrilateral.")

        # Create a Polygon object using the corners
        polygon = Polygon(corners)

        # Filter dots that are inside the polygon (only checking x and y values)
        inside_dots = [dot for dot in self.dots if polygon.contains(Point(dot[1], dot[2]))]

        return inside_dots

    def classify_batches(self, dots, axis=1, eps=0.002):
        """
        Classify dots into batches based on their axis-values (x or y) using DBSCAN.

        :param dots: List of tuples representing (flag, x, y, width, height) of the dots
        :param axis: 1 for y (default), 0 for x
        :param eps: Maximum gap between points in a batch (adjust based on data skew)
        :return: List of batches, each containing full-format dots, sorted by the value of the first element in each batch
        """
        # Extract the specified axis values (x or y)
        values = np.array([dot[axis + 1] for dot in dots]).reshape(-1, 1)
        if axis == 0:
            eps = 0.005

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=1).fit(values)

        # Group dots by their cluster labels
        batches = {}
        for label, dot in zip(clustering.labels_, dots):
            if label not in batches:
                batches[label] = []
            batches[label].append(dot)

        # Convert batches to a list of clusters
        clustered_batches = [
            sorted(batches[label], key=lambda dot: dot[1 if axis == 1 else 2])
            for label in sorted(batches.keys())
        ]
        return sorted(clustered_batches, key=lambda batch: batch[0][axis + 1])

    def process_batches_to_file(self, clustered_batches, input_string, output_file, gap_string="none"):
        """
        Processes each batch in clustered_batches and writes results to a file.

        :param clustered_batches: List of batches, where each batch contains full-format dots.
        :param input_string: The input string to prepend to the formatted output.
        :param output_file: Path to the text file to write the output.
        :param gap_string: Determines the suffix after batch_num:
                        - "none" means no suffix and writes the input string once per batch.
                        - 'a' means suffixes will be a, b, c, etc., and writes input string for each dot.
        :return: None
        """
        try:
            output_content = ""  # Initialize an empty string to hold the content
            if gap_string == "none":
                # Append the input string and batch_num once
                output_content += f"{input_string} "
        
            for batch_num, batch in enumerate(clustered_batches):
                for dot_index, dot in enumerate(batch):
                    if dot[0] == 0:  # Check if the flag is 0
                        x, y, width, height = dot[1:]

                        # Append each dot's coordinates and size
                        if gap_string == "none":
                            output_content += f"{x},{y},{width},{height} "
                        if gap_string == "a":
                            output_content += f"{input_string}.{chr(ord('a') + batch_num)} {x},{y},{width},{height} "                            
                        if input_string =="1." or input_string=="SBD" or input_string=="MDT":
                            # Append input string and suffix for each dot
                            output_content += f"{input_string}{batch_num+gap_string+1} {x},{y},{width},{height} "

            # Write all the constructed content to the file
            with open(output_file, "a") as file:
                file.write(output_content)

        except Exception as e:
            print(f"Error occurred while writing to file: {e}")

    def process_grid_section(self, section_index, axis, eps, input_string, gap_string, output_file):
        """
        Process a specific grid section: find dots, classify them, and save the results to a file.

        :param section_index: Tuple indicating the grid section (e.g., (3, 0))
        :param axis: Axis for classification (0 for x, 1 for y)
        :param eps: Maximum gap between points in a batch
        :param input_string: Input string to prepend to formatted output
        :param output_file: Path to the text file to write the output
        :return: None
        """
        corners = self.grid_section[section_index[0]][section_index[1]]
        dots_inside = self.dots_in_quadrilateral(corners)
        clustered_batches = self.classify_batches(dots_inside, axis=axis, eps=eps)
        self.process_batches_to_file(clustered_batches, input_string, output_file , gap_string)
       


    
# Example Usage
# grid_section = [
#     [
#         [(0, 0), (0, 100), (100, 100), (100, 0)],
#         [(100, 0), (100, 100), (200, 100), (200, 0)],
#     ]
# ]
# dots = [
#     (0, 10, 10, 5, 5),
#     (0, 20, 20, 5, 5),
#     (0, 150, 50, 5, 5),
#     (1, 110, 70, 5, 5),
# ]

# classifier = BubbleClassifier(grid_section, dots)
# classifier.process_grid_section(
#     section_index=(0, 0),
#     axis=1,
#     eps=0.002,
#     input_string="SBD",
#     gap_string='1'
#     output_file="test_results.txt"
# )

# classifier.process_grid_section(
#     section_index=(0, 1),
#     axis=1,
#     eps=0.002,
#     input_string="SBD",
#     gap_string='1'
#     output_file="test_results.txt"
# )
