import torch  # Import PyTorch for tensors
import cv2
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
