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

def append_to_file(file_path, string_to_append):
    """
    Appends a string to a text file.

    :param file_path: Path to the text file.
    :param string_to_append: The string to append to the file.
    :return: None
    """
    try:
        with open(file_path, "a") as file:  # Open the file in append mode
            file.write(string_to_append +" ")  # Append the string with a newline
    except Exception as e:
        print(f"Error occurred while appending to the file: {e}")
