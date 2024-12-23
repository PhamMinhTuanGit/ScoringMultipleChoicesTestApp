import cv2
import matplotlib.pyplot as plt

def draw_boxes_from_txt(image_path, txt_path, output_path):
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
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image_rgb)
    # plt.title("Image with Bounding Boxes")
    # plt.axis("off")
    # plt.show()

def draw_boxes_from_list(image_path, coordinate_lst, output_path):
    """
    This function draws bounding boxes on an image based on YOLO coordinates.

    Parameters:
    - image_path (str): The path to the input image.
    - coordinate_lst (list of tuples): A list containing tuples with YOLO format coordinates (x_center, y_center, width, height).
    - output_path (str): The path where the output image with bounding boxes will be saved.

    The function reads the input image, draws bounding boxes, and saves the resulting image at output_path.
    """
    # Read the input image
    image = cv2.imread(image_path)
    
    # Loop through each coordinate in the coordinate_lst
    for coord in coordinate_lst:
        # Extract the YOLO coordinates (x_center, y_center, width, height)
        x_center, y_center, width, height = coord
        
        # Convert YOLO coordinates to pixel coordinates
        h, w, _ = image.shape  # Get the height and width of the image
        x1 = int((x_center - width / 2) * w)  # Calculate the top-left corner's x-coordinate
        y1 = int((y_center - height / 2) * h)  # Calculate the top-left corner's y-coordinate
        x2 = int((x_center + width / 2) * w)  # Calculate the bottom-right corner's x-coordinate
        y2 = int((y_center + height / 2) * h)  # Calculate the bottom-right corner's y-coordinate
        
        # Draw the bounding box on the image (green color with thickness of 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Save the image with the drawn bounding boxes
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")