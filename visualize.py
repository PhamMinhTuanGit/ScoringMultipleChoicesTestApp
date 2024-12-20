import cv2
import matplotlib.pyplot as plt

def draw_boxes_from_yolo(image_path, txt_path, output_path):
    """
    Vẽ các hình chữ nhật từ file txt định dạng YOLO lên ảnh và hiển thị bằng Matplotlib.

    :param image_path: Đường dẫn ảnh gốc
    :param txt_path: Đường dẫn file txt chứa tọa độ YOLO
    :param output_path: Đường dẫn để lưu ảnh đã vẽ
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # Đọc file txt
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Tách các giá trị trong mỗi dòng
        class_id, x_center, y_center, width, height = map(float, line.split())

        # Chuyển đổi tọa độ từ YOLO format sang pixel
        x_center = int(x_center * image_width)
        y_center = int(y_center * image_height)
        width = int(width * image_width)
        height = int(height * image_height)

        # Tính tọa độ góc trái trên và góc phải dưới
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Vẽ hình chữ nhật lên ảnh
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Vẽ nhãn (class_id)
        cv2.putText(image, f"Class {int(class_id)}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Lưu ảnh đã vẽ
    cv2.imwrite(output_path, image)
    print(f"Processed image saved to: {output_path}")

    # Chuyển ảnh từ BGR sang RGB để hiển thị bằng Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Hiển thị ảnh bằng Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.title("Image with Bounding Boxes")
    plt.axis("off")
    plt.show()