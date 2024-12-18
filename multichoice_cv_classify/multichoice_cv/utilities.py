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
    Reads the file and extracts the second and third columns from all rows,
    returning them as a list of tuples.

    :param file_path: Path to the .txt file
    :return: List of tuples containing (column2, column3) from all rows
    """
    with open(file_path, 'r') as file:
        # Process all lines in the file
        bubbles = [
            (float(line.split()[1]), float(line.split()[2]))
            for line in file
        ]
        
    return bubbles