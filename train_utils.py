import torch  # Import PyTorch for tensors
import cv2
from shapely.geometry import Point, Polygon
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import image_processing
import grid_info
import visualization
from utilities import extract_bubbles, generate_random_colors, append_to_file
from bubble_classify import BubbleClassifier
from torchvision import transforms
from fixed_coordinates import fixed_circle
def read_file_to_tensors(file_path):
    """
    Reads a text file line by line, extracts labels and coordinates, and converts them into separate tensors.

    Args:
        file_path (str): Path to the text file.

    Returns:
        torch.Tensor: A tensor containing all labels (first number of each line).
        torch.Tensor: A tensor containing coordinates (remaining numbers of each line).
    """
    labels = []  # List to store the first numbers (labels)
    coords = []  # List to store the remaining numbers (coordinates)

    # Open the file and process it line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Split line into components by spaces
            split_line = line.strip().split()
            # Check if line has enough values
            if len(split_line) > 1:
                # Extract the first value as label and convert it to float
                labels.append(float(split_line[0]))
                # Extract the remaining values as coordinates
                coords.append([float(x) for x in split_line[1:]])

    # Convert the lists into PyTorch tensors
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)

    return label_tensor, coords_tensor

def find_yolov8_square(image, yolo_coords, output_path=None):
    """
    Finds and draws a square based on YOLOv8 coordinates on an image.

    Args:
        image_path (str): Path to the input image.
        yolo_coords (tuple): YOLOv8 coordinates (x_center, y_center, width, height).
        output_path (str): Path to save the output image. Optional.

    Returns:
        tuple: Top-left and bottom-right coordinates of the bounding box (x1, y1, x2, y2).
    """
    # Load the image
    h, w = image.shape[:2]  # Get image dimensions (height and width)

    # Extract YOLOv8 normalized coordinates
    x_center_norm, y_center_norm, width_norm, height_norm = yolo_coords

    # Convert normalized coordinates to pixel values
    x_center = int(x_center_norm * w)
    y_center = int(y_center_norm * h)
    width = int(width_norm * w)
    height = int(height_norm * h)

    # Calculate top-left and bottom-right coordinates of the bounding box
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return (x1, y1, x2, y2)

def get_box(input, coords):
    """
    Extracts a sub-region from an input image or array based on given coordinates.

    Args:
        input: The input image or array from which to extract the sub-region.
        coords (tuple): A tuple containing the coordinates (x1, y1, x2, y2) that define 
                        the top-left and bottom-right corners of the sub-region.

    Returns:
        The extracted sub-region from the input, determined by the specified coordinates.
    """

    x1, y1, x2, y2 = coords
    output = input[y1:y2, x1:x2]
    return output
import math
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
def getBubblesInClass(input_image_path, input_data, section_index, axis = 0, eps = 0.002):
        # centers = image_processing.getNails(input_image_path)
        # gridmatrix = grid_info.getGridmatrix(centers)
        # gridsection = grid_info.getExtractsections(gridmatrix)
        # dots = extract_bubbles(input_data)
        # corners = gridsection[section_index[0]][section_index[1]]
        # corners = sort_to_convex_quadrilateral(corners)
        # 
        # if len(corners) != 4:
        #         raise ValueError("Exactly 4 corners are required to define a quadrilateral.")

        #         # Create a Polygon object using the corners
        # polygon = Polygon(corners)

                # Filter dots that are inside the polygon (only checking x and y values)
        matrix_coordinate = fixed_circle(input_image_path, output_file = "test.txt")
        inside_dots = matrix_coordinate[section_index[0]][section_index[1]]
        dots = inside_dots
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
      

def Process(input_image_path,input_data, model, output_txt_path):
    device = torch.device("mps")
    image = cv2.imread(input_image_path)

    sections = [
            {"name": "SBD", "section_index": (0, 0), "axis": 0, "eps": 0.002, "input_string": "SBD", "gap_string": 0},
            {"name": "MDT", "section_index": (0, 1), "axis": 0, "eps": 0.002, "input_string": "MDT", "gap_string": 0},
            {"name": "phan1_1", "section_index": (1, 0), "axis": 1, "eps": 0.002, "input_string": "1.", "gap_string": 0},
            {"name": "phan1_2", "section_index": (1, 1), "axis": 1, "eps": 0.002, "input_string": "1.", "gap_string": 10},
            {"name": "phan1_3", "section_index": (1, 2), "axis": 1, "eps": 0.002, "input_string": "1.", "gap_string": 20},
            {"name": "phan1_4", "section_index": (1, 3), "axis": 1, "eps": 0.002, "input_string": "1.", "gap_string": 30},
            {"name": "phan2_1", "section_index": (2, 0), "axis": 1, "eps": 0.002, "input_string": "2.1", "gap_string": "a"},
            {"name": "phan2_2", "section_index": (2, 1), "axis": 1, "eps": 0.002, "input_string": "2.2", "gap_string": "a"},
            {"name": "phan2_3", "section_index": (2, 2), "axis": 1, "eps": 0.002, "input_string": "2.3", "gap_string": "a"},
            {"name": "phan2_4", "section_index": (2, 3), "axis": 1, "eps": 0.002, "input_string": "2.4", "gap_string": "a"},
            {"name": "phan2_5", "section_index": (2, 4), "axis": 1, "eps": 0.002, "input_string": "2.5", "gap_string": "a"},
            {"name": "phan2_6", "section_index": (2, 5), "axis": 1, "eps": 0.002, "input_string": "2.6", "gap_string": "a"},
            {"name": "phan2_7", "section_index": (2, 6), "axis": 1, "eps": 0.002, "input_string": "2.7", "gap_string": "a"},
            {"name": "phan2_8", "section_index": (2, 7), "axis": 1, "eps": 0.002, "input_string": "2.8", "gap_string": "a"},
            {"name": "phan3_1", "section_index": (3, 0), "axis": 0, "eps": 0.002, "input_string": "3.1", "gap_string": "none"},
            {"name": "phan3_2", "section_index": (3, 1), "axis": 0, "eps": 0.002, "input_string": "3.2", "gap_string": "none"},
            {"name": "phan3_3", "section_index": (3, 2), "axis": 0, "eps": 0.002, "input_string": "3.3", "gap_string": "none"},
            {"name": "phan3_4", "section_index": (3, 3), "axis": 0, "eps": 0.002, "input_string": "3.4", "gap_string": "none"},
            {"name": "phan3_5", "section_index": (3, 4), "axis": 0, "eps": 0.002, "input_string": "3.5", "gap_string": "none"},
            {"name": "phan3_6", "section_index": (3, 5), "axis": 0, "eps": 0.002, "input_string": "3.6", "gap_string": "none"}
        ]
    with open(output_txt_path, 'w') as f:
        for section in sections:
            Bubbles = getBubblesInClass(input_image_path, input_data, section["section_index"], axis=section["axis"], eps=section["eps"])
            for Class in Bubbles:
                max_confidence = -float('inf')
                best_coords = None   
                for label in Class:
                    _, x1, y1, x2, y2 = label
                    coord = (x1, y1, x2, y2)
                    x1, y1, x2, y2 = find_yolov8_square(image, coord)
                    input = get_box(image, (x1, y1, x2, y2))
                    
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((32, 32))
                    ])
                    model.eval()
                    input = transform(input).to(device)
                    output = model(input.unsqueeze(0))
                    confidence_score = output[0][0].item()  # Access the first element, which is the confidence score

                    # Cập nhật bounding box có confidence cao nhất trong class
                    if confidence_score > max_confidence:
                        max_confidence = confidence_score
                        best_coords = (x1, y1, x2, y2)

                # Sau khi đã tìm được bounding box có confidence cao nhất trong class
                if best_coords:
                    x1, y1, x2, y2 = best_coords
                    # Ghi tọa độ vào file txt
                    f.write(f"{x1},{y1},{x2},{y2}\n")  # Ghi một dòng tọa độ