import cv2
import numpy as np
import matplotlib.pyplot as plt

def yolo_boxes_color_check(image_path, txt_path, threshold=220):
    """
    Xác định mỗi vùng từ file tọa độ YOLO là đen hay trắng và vẽ minh họa trên ảnh.
    
    Args:
        image_path (str): Đường dẫn đến ảnh đầu vào.
        txt_path (str): Đường dẫn đến file tọa độ YOLO.
        threshold (int): Ngưỡng để phân biệt đen và trắng (0-255).
        
    Returns:
        None. Hiển thị ảnh minh họa với matplotlib.
    """
    # Đọc ảnh và chuyển sang grayscale
    image = cv2.imread(image_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray_img.shape

    # Đọc file tọa độ YOLO
    with open(txt_path, "r") as f:
        boxes = f.readlines()
    
    # Tạo ảnh để vẽ
    output_img = image.copy()

    for box in boxes:
        # Mỗi dòng YOLO: class x_center y_center width height
        _, x_center, y_center, box_width, box_height = map(float, box.split())
        x_center, y_center = int(x_center * w), int(y_center * h)
        box_width, box_height = int(box_width * w), int(box_height * h)
        
        # Xác định tọa độ góc trái trên và phải dưới
        x1, y1 = x_center - box_width // 2, y_center - box_height // 2
        x2, y2 = x_center + box_width // 2, y_center + box_height // 2
        
        # Trích xuất vùng và kiểm tra trung bình độ sáng
        region = gray_img[y1:y2, x1:x2]
        mean_intensity = np.mean(region)
        
        # Xác định màu vẽ
        if mean_intensity < threshold:
            color = (255, 0, 0)  # Red for white region
        else:
            color = (0, 128, 255)  # Blue for black region
        
        # Vẽ hình chữ nhật lên ảnh
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)

    # Hiển thị kết quả
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Regions classified as Black or White")
    plt.show()


yolo_boxes_color_check('Image\Copy of IMG_1584_iter_36.jpg', "test.txt")
