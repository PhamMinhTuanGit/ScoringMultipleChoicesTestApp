import cv2
import numpy as np
from detect_black_square import detect_black_square_centers

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


def warp_perspective_for_detect(image, points_of_interest, image_size):
    """
    Warp perspective and calculate transformed boxes on the original image.
    
    :param image: The original image.
    :param points_of_interest: List of 4 irregular points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    :param yolo_boxes: Target grid points in YOLO format [(center_x, center_y, width, height), ...].
    :param image_size: Tuple (width, height) of the image to determine scaling.
    :return: Transformed boxes matching the original image.
    """
    image_copy = image.copy()
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

    warped_image = cv2.warpPerspective(image_copy, M, (img_w, img_h))
    warp_tmp_path = r"Image/warp_tmp.jpg"
    cv2.imwrite(warp_tmp_path,warped_image)

    centers = detect_black_square_centers(warp_tmp_path)

    # Transform YOLO boxes to pixel coordinates
    transformed_boxes_warped = []
    for box in centers:
        box_center_x, box_center_y = box
        pixel_x = int(box_center_x * img_w)
        pixel_y = int(box_center_y * img_h)

        transformed_boxes_warped.append((pixel_x, pixel_y))

    # Reverse-transform the box centers to the original image
    transformed_boxes_original = []
    for pixel_x, pixel_y in transformed_boxes_warped:
        point = np.array([[[pixel_x, pixel_y]]], dtype="float32")
        transformed_point = cv2.perspectiveTransform(point, M_inv)

        transformed_boxes_original.append((
            transformed_point[0][0][0],  # Transformed center x
            transformed_point[0][0][1]  # Transformed center y
        ))

    return transformed_boxes_original