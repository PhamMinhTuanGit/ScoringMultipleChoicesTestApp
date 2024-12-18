# utilities.py
import random

def generate_random_colors(n):
    """
    Generate `n` distinct random colors.

    :param n: Number of colors to generate
    :return: List of random colors
    """
    return [tuple(random.random() for _ in range(3)) for _ in range(n)]