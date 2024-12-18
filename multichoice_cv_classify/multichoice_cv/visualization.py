# visualization.py
import cv2
import matplotlib.pyplot as plt
from utilities import generate_random_colors
import random


def drawDots(image_path, dots, output_path="draw_dots.jpg", dot_radius=10,random_color = 0):
    """
    Draws large dots on an image based on YOLOv8 format coordinates with random colors.

    :param image_path: Path to the input image
    :param dots: A list of (x, y) coordinates scaled between 0 and 1
    :param output_path: Path to save the output image with dots drawn
    :param dot_radius: Radius of the dots (default is 10)
    :return: None (saves the output image to `output_path`)
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    height, width, _ = image.shape

    if random_color == 1:
        # Generate a random color in BGR format
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    else:
        color = (0,0,255)
    # Draw each dot on the image with a random color
    for dot in dots:
        # Convert normalized coordinates to pixel values
        x = int(dot[0] * width)
        y = int(dot[1] * height)

        # Draw a circle (dot) on the image
        cv2.circle(image, (x, y), dot_radius, color, thickness=-1)  # -1 fills the circle

    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Output image saved at {output_path}")

def drawClustereddots(image_path, clustered_batches, output_path="draw_batches.jpg", dot_radius=10):
    """
    Draws large dots for batches on an image, assigning a unique color for each batch.

    :param image_path: Path to the input image
    :param clustered_batches: List of batches, where each batch contains full-format dots
    :param output_path: Path to save the output image with dots drawn
    :param dot_radius: Radius of the dots (default is 10)
    :return: None (saves the output image to `output_path`)
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    height, width, _ = image.shape

    # Assign a unique random color to each batch
    for batch in clustered_batches:
        batch_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for dot in batch:
            # Extract normalized x and y coordinates
            x = int(dot[1] * width)  # dot[1] is x
            y = int(dot[2] * height)  # dot[2] is y

            # Draw a circle (dot) on the image
            cv2.circle(image, (x, y), dot_radius, batch_color, thickness=-1)  # -1 fills the circle

    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Output image saved at {output_path}")


def visualize_batches(batches):
    """
    Visualizes clusters of dots using Matplotlib.

    :param batches: A dictionary where keys are cluster labels and values are lists of (x, y) tuples
    """
    plt.figure(figsize=(8, 6))
    colors = generate_random_colors(len(batches))

    for label, dots in batches.items():
        x, y = zip(*dots)  # Separate x and y coordinates
        plt.scatter(x, y, color=colors[label], label=f"Batch {label}", s=100)  # Use larger dots for visibility

    plt.title("Dot Batches Based on Y-Values")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()
