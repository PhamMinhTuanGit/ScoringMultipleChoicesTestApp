# bubble_detection.py
from multichoice_cv.quadrilateral import bubbles_inside_quadrilateral
from sklearn.cluster import DBSCAN
import numpy as np

class BubbleDetector:
    def __init__(self, quadrilaterals):
        """
        Initialize with a list of quadrilaterals for classification.
        :param quadrilaterals: List of tuples representing the 4 corners of each quadrilateral.
        """
        self.quadrilaterals = quadrilaterals

    def detect_filled_bubbles(self, bubbles):
        """
        Detect filled bubbles and classify them based on their position in quadrilaterals.
        :param bubbles: List of tuples {flag},(x,y),(width,height).
        :return: List of tuples {class},(x,y),(width,height) for filled bubbles.
        """
        filled_bubbles = []
        for flag, (x, y), (width, height) in bubbles:
            for idx, quad in enumerate(self.quadrilaterals):
                if is_point_inside_quadrilateral((x, y), quad):
                    filled_bubbles.append((idx, (x, y), (width, height)))
                    break
        return filled_bubbles
    