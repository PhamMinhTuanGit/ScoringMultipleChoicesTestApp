import cv2
import numpy as np
import math
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import random
# Input and output paths
input_image_path = 'IMG_1581_iter_0.jpg'
output_image_path = 'output1.jpg'
output_txt_path = 'output_centers.txt'
file_path = 'IMG_1581_iter_0.txt'  # Replace with the actual file path

def detect_black_square_centers(image_path, output_image_path, output_txt_path):
    """
    Detects black squares in a given image and saves the output to a file.
    
    This function takes an image path, an output image path, and an output text path as input. 
    It reads the image, processes it to find the centers of black squares, and saves the results 
    in both an annotated image and a text file in YOLOv8 format.
    
    The coordinates of each detected square's center are also written beside the corresponding box 
    in the annotated image.
    
    :param image_path: Path to the input image
    :param output_image_path: Path to the annotated output image
    :param output_txt_path: Path to the output text file
    :return: List of centers of the black squares in YOLOv8 format
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    image_height, image_width = image.shape[:2]

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon is a quadrilateral
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Ensure it's a square and of a reasonable size
            if 0.9 < aspect_ratio < 1.1 and w > 20 and h > 20:
                roi = thresh[y:y+h, x:x+w]
                black_pixels = cv2.countNonZero(roi)
                total_pixels = w * h
                fill_ratio = black_pixels / total_pixels

                # Check if the square is mostly filled
                if fill_ratio > 0.9:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])

                        # Convert to YOLOv8 format
                        yolov8_x = center_x / image_width
                        yolov8_y = center_y / image_height

                        centers.append((yolov8_x, yolov8_y))

                        # Annotate the image
                        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Write coordinates beside the black box (Black text color)
                        coordinate_text = f"({yolov8_x:.4f}, {yolov8_y:.4f})"
                        cv2.putText(image, coordinate_text, (center_x + 10, center_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Save the annotated image
    cv2.imwrite(output_image_path, image)

    # Save the centers to a text file
    with open(output_txt_path, 'w') as file:
        for center in centers:
            file.write(f"{center[0]:.6f}, {center[1]:.6f}\n")

    print(f"Processed image saved to: {output_image_path}")
    print(f"Centers of black squares saved to: {output_txt_path}")
    return centers

def find_dots_between_y(points, dot1, dot2):
    """
    Finds all dots with y-coordinates between the y-values of two given dots.

    :param points: List of all points in the format [(x, y), ...]
    :param dot1: The first reference dot as (x, y)
    :param dot2: The second reference dot as (x, y)
    :return: List of dots with y-coordinates between dot1 and dot2
    """
    # Extract y-values from input dots
    y1, y2 = dot1[1], dot2[1]

    # Determine the range (min and max y)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Filter points with y-values within the range
    dots_in_range = [point for point in points if y_min <= point[1] <= y_max]

    return dots_in_range

def draw_lines_and_store_dots_matrix(centers, image):
    """
    Finds the 5 points with the highest x values, sorts them by y values,
    and draws lines sequentially between consecutive points.

    Appends leftover points from centers (not in dots_matrix) to dots_matrix,
    sorted by 2 highest and 2 lowest y-values.

    :param centers: List of center coordinates in YOLOv8 format [(x, y), ...]
    :param image: The image to draw on
    :return: Dots matrix containing points between drawn lines and leftover points
    """
    if len(centers) < 2:
        print("Not enough points to draw lines.")
        return []

    # Find the 5 points with the highest and lowest x values
    top_5_x_max = sorted(centers, key=lambda p: p[0], reverse=True)[:5]
    top_5_x_min = sorted(centers, key=lambda p: p[0])[:5]

    # Sort the points by their y values
    top_5_x_max_sorted_by_y = sorted(top_5_x_max, key=lambda p: p[1])
    top_5_x_min_sorted_by_y = sorted(top_5_x_min, key=lambda p: p[1])

    # Initialize the matrix to store dots found between pairs
    dots_matrix = []

    # Draw lines sequentially between the sorted points
    for i in range(len(top_5_x_max_sorted_by_y) - 1):
        point1 = top_5_x_max_sorted_by_y[i]
        point2 = top_5_x_max_sorted_by_y[i + 1]
        #cv2.line(image, point1, point2, (255, 0, 0), 2)  # Blue line

    for i in range(len(top_5_x_min_sorted_by_y) - 1):
        point1 = top_5_x_min_sorted_by_y[i]
        point2 = top_5_x_min_sorted_by_y[i + 1]
        #cv2.line(image, point1, point2, (255, 0, 0), 2)  # Blue line

    for i in range(len(top_5_x_min_sorted_by_y)):
        point1 = top_5_x_min_sorted_by_y[i]
        point2 = top_5_x_max_sorted_by_y[i]
        # Find dots between these two points
        # Sort from lowest x to hightest
        dots_between = sorted(find_dots_between_y(centers, point1, point2), key=lambda p:p[0])
        # Add to the matrix
        dots_matrix.append(dots_between)
        #cv2.line(image, point1, point2, (0, 0, 255), 2)  # Red line

    # Complement the lost dot by vector trick 
    dots_buffer=(0,0)
    dots_matrix[1].append(dots_buffer)   
    dots_matrix[1][4]=dots_matrix[1][3]
    dots_matrix[1][3] = (
    dots_matrix[1][4][0] - dots_matrix[2][4][0] + dots_matrix[2][3][0],  # x-coordinate
    dots_matrix[1][4][1] - dots_matrix[2][4][1] + dots_matrix[2][3][1]   # y-coordinate
    )
    #print("new dot is",dots_matrix[1][3]," New vector is: ",dots_matrix[1])

    # Get leftover points not in dots_matrix
    used_points = {tuple(dot) for row in dots_matrix for dot in row}
    leftover_points = [point for point in centers if tuple(point) not in used_points]

    # Sort leftover points by y-values
    leftover_sorted_by_x = sorted(leftover_points, key=lambda p: p[0])
    dots_matrix.append(leftover_sorted_by_x)  # 2 highest y-values

    # Append 2 highest and 2 lowest y-value points to the dots_matrix
    # if len(leftover_sorted_by_x) >= 2:
    #     dots_matrix.append(leftover_sorted_by_x[-2:])  # 2 highest y-values
    # if len(leftover_sorted_by_x) >= 4:
    #     dots_matrix.append(leftover_sorted_by_x[:2])  # 2 lowest y-values

    print("Lines drawn and dots matrix updated.")
    #cv2.imwrite('final_output.jpg', image)
    return dots_matrix

def dots_in_quadrilateral(corners, dots):
    """
    Finds all the dots inside the quadrilateral defined by four corners.
    
    :param corners: A tuple of 4 tuples, each representing (x, y) coordinates of the quadrilateral's corners
    :param dots: A list of tuples representing (x, y) coordinates of the dots to check
    :return: A list of tuples (dots) that are inside the quadrilateral
    """
    from shapely.geometry import Point, Polygon

    # Ensure the corners are sorted correctly (in order of the quadrilateral's path)
    if len(corners) != 4:
        raise ValueError("Exactly 4 corners are required to define a quadrilateral.")
    
    # Create a Polygon object using the corners
    polygon = Polygon(corners)
    
    # Filter dots that are inside the polygon
    inside_dots = [dot for dot in dots if polygon.contains(Point(dot))]
    
    return inside_dots

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
def draw_dots_on_image(image_path, dots, output_path="output_image.jpg", dot_color=(0, 0, 255), dot_radius=10):
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

def classify_and_visualize_batches(dots, eps=5):
    """
    Classify dots into batches based on their y-values using DBSCAN and visualize the result.

    :param dots: List of (x, y) tuples representing the dots
    :param eps: Maximum gap between points in a batch (adjust based on data skew)
    :return: List of batches, each containing dots
    """
    # Extract y-values
    y_values = np.array([dot[1] for dot in dots]).reshape(-1, 1)

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=1).fit(y_values)

    # Group dots by their cluster labels
    batches = {}
    for label, dot in zip(clustering.labels_, dots):
        if label not in batches:
            batches[label] = []
        batches[label].append(dot)

    # Convert batches to a sorted list of clusters
    clustered_batches = [batches[label] for label in sorted(batches.keys())]

    # Visualize clusters
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

    return clustered_batches

def generate_random_colors(n):
    """
    Generate `n` distinct random colors.

    :param n: Number of colors to generate
    :return: List of random colors
    """
    return [tuple(random.random() for _ in range(3)) for _ in range(n)]

# Run the function
centers = detect_black_square_centers(input_image_path, output_image_path, output_txt_path)
#print("The dots is: ",centers)
image = cv2.imread(output_image_path)
bubbles = extract_bubbles(file_path)
print("\n Do dai cua bubbles",len(bubbles))
#print(bubbles)
dots_matrix=draw_lines_and_store_dots_matrix(centers,image)
class_corner=sort_to_convex_quadrilateral(dots_matrix[5])
print("\n 4 goc cua SBD la: ",class_corner)
sbd = dots_in_quadrilateral(class_corner,bubbles)
print("\n SBD bao gom ",len(sbd)," diem gom nhung diem sau: ",sbd)
classified_bubble = classify_and_visualize_batches(sbd,0.001)
print("\n the classified bubble is: ", classified_bubble)
draw_dots_on_image(input_image_path, sbd, output_path="output_with_dots.jpg", dot_radius=15)
