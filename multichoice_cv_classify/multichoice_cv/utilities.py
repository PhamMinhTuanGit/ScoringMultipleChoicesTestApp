# utilities.py
import random
import math 

def generate_random_colors(n):
    """
    Generate `n` distinct random colors.

    :param n: Number of colors to generate
    :return: List of random colors
    """
    return [tuple(random.random() for _ in range(3)) for _ in range(n)]

def extract_bubbles(file_path):
    """
    Reads the file and extracts all columns from each row, returning them as a list of tuples.

    :param file_path: Path to the .txt file
    :return: List of tuples containing (flag, x, y, width, height) from all rows
    """
    with open(file_path, 'r') as file:
        # Process all lines in the file
        bubbles = [
            (float(line.split()[0]),  # flag
             float(line.split()[1]),  # x
             float(line.split()[2]),  # y
             float(line.split()[3]),  # width
             float(line.split()[4]))  # height
            for line in file
        ]
        
    return bubbles
