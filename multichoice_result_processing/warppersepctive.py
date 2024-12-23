import cv2
import numpy as np
import image_processing
import grid_info
import visualization

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def yolo_to_pixel(points, img_size):
    """
    Convert YOLOv8 normalized coordinates to pixel coordinates.
    
    :param points: List of YOLOv8 points [(center_x, center_y, width, height), ...]
    :param img_size: Tuple (image_width, image_height)
    :return: List of points in pixel coordinates [(x1, y1), ...]
    """
    img_w, img_h = img_size
    return [(int(x * img_w), int(y * img_h)) for x, y in points]

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

    grid_coords = []
    for row in range(rows):
        for col in range(cols):
            new_center_x = center_x + col * (width + x_spacing)
            new_center_y = center_y + row * (height + y_spacing)
            grid_coords.append((new_center_x, new_center_y, width, height))

    return grid_coords

# ------------------------------------------------------------------------------
# Perspective Transformation
# ------------------------------------------------------------------------------

def warp_perspective_with_reversed_boxes(image, points_of_interest, yolo_boxes, image_size):
    """
    Warp perspective and calculate transformed boxes on the original image.
    
    :param image: The original image.
    :param points_of_interest: List of 4 irregular points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    :param yolo_boxes: Target grid points in YOLO format [(center_x, center_y, width, height), ...].
    :param image_size: Tuple (width, height) of the image to determine scaling.
    :return: Transformed boxes matching the original image.
    """
    src_pts = np.array(points_of_interest, dtype="float32")

    img_w, img_h = image_size
    dst_pts = np.array([
        [0, 0],  # Top-left corner
        [img_w - 1, 0],  # Top-right corner
        [img_w - 1, img_h - 1],  # Bottom-right corner
        [0, img_h - 1]  # Bottom-left corner
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)  # Inverse matrix

    warped_image = cv2.warpPerspective(image, M, (img_w, img_h))

    # Transform YOLO boxes to pixel coordinates
    transformed_boxes_warped = []
    for box in yolo_boxes:
        box_center_x, box_center_y, box_width, box_height = box
        pixel_x = int(box_center_x * img_w)
        pixel_y = int(box_center_y * img_h)
        pixel_w = int(box_width * img_w)
        pixel_h = int(box_height * img_h)

        transformed_boxes_warped.append((pixel_x, pixel_y, pixel_w, pixel_h))

    # Reverse-transform the box centers to the original image
    transformed_boxes_original = []
    for pixel_x, pixel_y, pixel_w, pixel_h in transformed_boxes_warped:
        point = np.array([[[pixel_x, pixel_y]]], dtype="float32")
        transformed_point = cv2.perspectiveTransform(point, M_inv)

        transformed_boxes_original.append((
            transformed_point[0][0][0],  # Transformed center x
            transformed_point[0][0][1],  # Transformed center y
            pixel_w, pixel_h
        ))

    return warped_image, transformed_boxes_original

# ------------------------------------------------------------------------------
# Main Process - Workflow
# ------------------------------------------------------------------------------

def main():
    # Load the image
    image_path = 'D:\\4_2_AREAS\\testset1\\testset1\\images\\IMG_3976_iter_0.jpg'
    image = cv2.imread(image_path)
    img_h, img_w, _ = image.shape
    image_size = (img_w, img_h)

    # Get the centers of interest from the image (processing step)
    centers = image_processing.getNails(image_path)
    
    # Get grid matrix and section grid info
    grid_matrix = grid_info.getGridmatrix(centers)
    section_grid = grid_info.getExtractsections(grid_matrix)
    
    # Flatten the grid matrix for visualization
    flat_matrix = grid_info.flattenMatrix(grid_matrix)
    visualization.drawDots(image_path, flat_matrix)

    # Example: Select points of interest for warping
    points_of_interest = yolo_to_pixel(section_grid[3][0], image_size)

    # Generate YOLO grid boxes
    yolo_boxes = generate_grid_coordinates((0.344, 0.262, 0.0158, 0.012), (12, 4), (0.132, 0.046))

    # Perform warping and reverse transformation
    warped_image, transformed_boxes_original = warp_perspective_with_reversed_boxes(
        image, points_of_interest, yolo_boxes, image_size
    )

    # Visualize the transformed points and boxes on the image
    visualize_results(image, points_of_interest, transformed_boxes_original)

    # Save or display the image
    cv2.imwrite("original_with_points_and_boxes.jpg", image)

def visualize_results(image, points_of_interest, transformed_boxes_original):
    """
    Visualize the results: Draw points of interest and transformed boxes on the image.

    :param image: The image to draw on.
    :param points_of_interest: List of points to be visualized.
    :param transformed_boxes_original: List of transformed bounding boxes.
    """
    # Draw points of interest (red dots)
    for i, (x, y) in enumerate(points_of_interest):
        cv2.circle(image, (int(x), int(y)), radius=20, color=(0, 0, 255), thickness=-1)  # Red dots
        cv2.putText(image, f"P{i+1}", (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Draw transformed boxes (green rectangles)
    for x, y, w, h in transformed_boxes_original:
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green boxes

# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
