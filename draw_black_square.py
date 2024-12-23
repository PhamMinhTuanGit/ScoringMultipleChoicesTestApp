import cv2

def draw_rectangles(image_path, centers, rect_width, rect_height, output_path, color=(0, 255, 0)):
    """
    Vẽ các hình chữ nhật lên ảnh dựa trên tọa độ trung tâm YOLO.

    Args:
        image_path (str): Đường dẫn tới ảnh.
        centers (list): Danh sách tọa độ trung tâm theo YOLO [(center_x, center_y), ...].
        rect_width (int): Chiều rộng cố định của hình chữ nhật (pixels).
        rect_height (int): Chiều cao cố định của hình chữ nhật (pixels).
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print("Không thể tải ảnh. Vui lòng kiểm tra đường dẫn!")
        return
    
    height, width, _ = image.shape  # Kích thước ảnh

    # Duyệt qua danh sách các tọa độ
    for idx, (center_x, center_y) in enumerate(centers, start=1):
        # Chuyển từ tỷ lệ YOLO sang pixel
        cx = int(center_x * width)
        cy = int(center_y * height)

        # Tính tọa độ góc trái trên của hình chữ nhật
        x = int(cx - rect_width / 2)
        y = int(cy - rect_height / 2)

        # Vẽ hình chữ nhật
        cv2.rectangle(image, (x, y), (x + rect_width, y + rect_height), color, 2)

        # Đánh số thứ tự tại trung tâm hình chữ nhật
        cv2.putText(image, str(idx), (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    cv2.imwrite(output_path, image)
