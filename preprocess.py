import cv2
import math

def find_top_left_and_top_right(points):
    """
    Xác định điểm top-left và top-right từ danh sách tọa độ YOLO.

    :param points: Danh sách tọa độ [(x1, y1), (x2, y2), ...] theo định dạng YOLO.
    :return: Tọa độ top-left và top-right.
    """
    # Sắp xếp các điểm theo giá trị y trước, nếu bằng thì theo x
    sorted_points = sorted(points, key=lambda p: (p[1], p[0]))

    # Top-left là điểm có x nhỏ nhất và y nhỏ nhất
    top_left = sorted_points[0]

    # Top-right là điểm có x lớn nhất và y nhỏ nhất
    top_right_candidates = [p for p in sorted_points if abs(p[1] - top_left[1]) < 0.01]  # Cùng hàng với top-left
    top_right = max(top_right_candidates, key=lambda p: p[0])  # Điểm có x lớn nhất

    return top_left, top_right

def find_angle(points):
    """
    Tính góc giữa đường nối top-left và top-right với trục x.

    :param points: Danh sách tọa độ [(x1, y1), (x2, y2), ...] theo định dạng YOLO.
    :return: Góc tính bằng độ.
    """
    # Xác định top-left và top-right
    top_left, top_right = find_top_left_and_top_right(points)

    # Tính góc dựa trên tọa độ
    delta_x = top_right[0] - top_left[0]
    delta_y = top_right[1] - top_left[1]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle

def rotate(image, angle, scale):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Tạo ma trận xoay
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Áp dụng ma trận xoay
    rotated_image = cv2.warpAffine(image, M, (w, h))

    return rotated_image