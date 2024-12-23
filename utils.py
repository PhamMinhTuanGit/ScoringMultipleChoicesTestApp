import cv2
import numpy as np

def generate_grid_coordinates(start_coord, grid_size, cell_spacing):
    """
    Generate a grid of bounding boxes based on the starting YOLO coordinate.

    :param start_coord: Tuple (center_x, center_y, width, height) of the top-left box (YOLO format)
    :param grid_size: Tuple (rows, cols) indicating the size of the grid (e.g., (6, 10))
    :param cell_spacing: Tuple (x_spacing, y_spacing) indicating spacing between boxes (YOLO format)
    :return: List of bounding boxes in YOLO format [(center_x, center_y, width, height), ...]
    """
    center_x, center_y, width, height = start_coord
    rows, cols = grid_size
    x_spacing, y_spacing = cell_spacing

    # Create grid coordinates
    grid_coords = []
    for row in range(rows):
        for col in range(cols):
            new_center_x = center_x + col * (width + x_spacing)
            new_center_y = center_y + row * (height + y_spacing)
            grid_coords.append((new_center_x, new_center_y, width, height))

    return grid_coords

def cut_image_from_points(image, points):
    """
    Cuts a region of the image using 4 specific points without resizing and displays it.

    :param image: Input image
    :param points: List of 4 specific points [(x, y), ...] in normalized coordinates (YOLO format)
    :param output_path: Path to save the cropped image
    """
    # Lấy kích thước ảnh
    image_height, image_width = image.shape[:2]

    # Chuyển đổi tọa độ từ normalized (YOLO format) sang pixel
    pixel_points = np.array([
        [int(point[0] * image_width), int(point[1] * image_height)]
        for point in points
    ], dtype=np.int32)

    # Tính bounding box (tọa độ tối thiểu và tối đa)
    x_min, y_min = np.min(pixel_points, axis=0)
    x_max, y_max = np.max(pixel_points, axis=0)

    return x_min, y_min, x_max, y_max

def convert_to_yolo_format(bboxes, img_width, img_height):
    """
    Convert bounding boxes to YOLO format.

    Parameters:
        bboxes (list of tuples): List of bounding boxes in (center_x, center_y, width, height) format (in pixels).
        img_width (int): Width of the image (in pixels).
        img_height (int): Height of the image (in pixels).
    Returns:
        list of lists: List of bounding boxes in YOLO format.
    """
    yolo_bboxes = []
    for bbox in bboxes:
        center_x, center_y, width, height = bbox
        # Normalize the values
        center_x_normalized = center_x / img_width
        center_y_normalized = center_y / img_height
        width_normalized = width / img_width
        height_normalized = height / img_height
        # Add to the YOLO formatted list
        yolo_bboxes.append((center_x_normalized, center_y_normalized, width_normalized, height_normalized))
    return yolo_bboxes

def yolo_to_pixel(points, img_size):
    """
    Convert YOLOv8 normalized coordinates to pixel coordinates.
    
    :param points: List of YOLOv8 points [(center_x, center_y, width, height), ...]
    :param img_size: Tuple (image_width, image_height)
    :return: List of points in pixel coordinates [(x1, y1), ...]
    """
    img_w, img_h = img_size
    return [(int(x * img_w), int(y * img_h)) for x, y in points]

def save_to_txt(yolo_coordinates_with_class, output_file):
    # Write to txt
    with open(output_file, "w") as file:
        for coords in yolo_coordinates_with_class:
            line = " ".join(map(str, coords))
            file.write(line + "\n")

    print(f"Dữ liệu đã được ghi vào file {output_file}.")