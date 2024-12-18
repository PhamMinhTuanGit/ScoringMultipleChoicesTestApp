# visualization.py
import cv2
import matplotlib.pyplot as plt
from utilities import generate_random_colors

def draw_dots_on_image(image_path, dots, output_path="draw_dots.jpg", dot_color=(0, 0, 255), dot_radius=10):
    """
    Draws large dots on an image based on YOLOv8 format coordinates.

    :param image_path: Path to the input image
    :param dots: A tuple of (x, y) coordinates scaled between 0 and 1
    :param output_path: Path to save the output image with dots drawn
    :param dot_color: Color of the dots in BGR format (default is red)
    :param dot_radius: Radius of the dots (default is 10)
    :return: None (saves the output image to `output_path`)
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    height, width, _ = image.shape
    
    # Draw each dot on the image
    for dot in dots:
        # Convert normalized coordinates to pixel values
        x = int(dot[0] * width)
        y = int(dot[1] * height)

        # Draw a circle (dot) on the image
        cv2.circle(image, (x, y), dot_radius, dot_color, thickness=-1)  # -1 fills the circle

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
