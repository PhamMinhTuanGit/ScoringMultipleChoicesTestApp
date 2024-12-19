import cv2

def detect_black_square_centers(image):
    """
    Detects black squares in a given image, saves the output, and displays numbered IDs.

    :param image: Input image
    :return: List of centers of the black squares in YOLOv8 format
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold để phát hiện vùng đen
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    image_height, image_width = image.shape[:2]

    idx = 0  # Khởi tạo bộ đếm
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Kiểm tra contour có 4 đỉnh
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Kiểm tra tỉ lệ và kích thước khung hình
            if 0.9 < aspect_ratio < 1.1 and w > 20 and h > 20:
                roi = thresh[y:y+h, x:x+w]
                black_pixels = cv2.countNonZero(roi)
                total_pixels = w * h
                fill_ratio = black_pixels / total_pixels

                # Kiểm tra mật độ màu đen
                if fill_ratio > 0.9:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])

                        yolov8_x = center_x / image_width
                        yolov8_y = center_y / image_height

                        centers.append((yolov8_x, yolov8_y))

    # Sắp xếp các điểm theo tọa độ y giảm dần, nếu y bằng nhau thì theo x giảm dần
    centers_sorted = sorted(centers, key=lambda point: (-point[1], -point[0]))

    # Tạo các nhóm điểm theo yêu cầu:
    group_1 = centers_sorted[:7]   # 7 điểm có y lớn nhất
    group_2 = centers_sorted[7:16] # 9 điểm có y lớn tiếp theo
    group_3 = centers_sorted[16:21] # 5 điểm có y lớn tiếp theo
    group_4 = centers_sorted[21:25] # 4 điểm có y lớn tiếp theo
    group_5 = centers_sorted[25:27] # 2 điểm có y lớn tiếp theo
    group_6 = centers_sorted[27:29] # 2 điểm có y lớn tiếp theo
    group_7 = centers_sorted[29:]   # 2 điểm còn lại

    # Đảm bảo các nhóm đều đã được sắp xếp theo x giảm dần
    group_1_sorted = sorted(group_1, key=lambda point: -point[0])
    group_2_sorted = sorted(group_2, key=lambda point: -point[0])
    group_3_sorted = sorted(group_3, key=lambda point: -point[0])
    group_4_sorted = sorted(group_4, key=lambda point: -point[0])
    group_5_sorted = sorted(group_5, key=lambda point: -point[0])
    group_6_sorted = sorted(group_6, key=lambda point: -point[0])
    group_7_sorted = sorted(group_7, key=lambda point: -point[0])

    # Kết hợp tất cả các nhóm lại theo thứ tự đã yêu cầu
    final_sorted_centers = group_1_sorted + group_2_sorted + group_3_sorted + group_4_sorted + group_5_sorted + group_6_sorted + group_7_sorted
    print("Centers of black squares (YOLOv8 format):", final_sorted_centers)
    return final_sorted_centers
