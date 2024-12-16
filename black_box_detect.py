import cv2
import numpy as np

def detect_black_square_centers(image_path, output_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    image_height, image_width = image.shape[:2]

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            if 0.9 < aspect_ratio < 1.1 and w > 20 and h > 20:
                roi = thresh[y:y+h, x:x+w]
                black_pixels = cv2.countNonZero(roi)
                total_pixels = w * h
                fill_ratio = black_pixels / total_pixels

                if fill_ratio > 0.9:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])

                        yolov8_x = center_x / image_width
                        yolov8_y = center_y / image_height

                        centers.append((yolov8_x, yolov8_y))

                        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_path, image)
    print(f"Processed image saved to: {output_path}")
    print("Centers of black squares (YOLOv8 format):", centers)
    return centers

input_image_path = '/Users/phamminhtuan/Downloads/Trainning_SET/Images/IMG_1581_iter_0.jpg' 
output_image_path = '/Users/phamminhtuan/Desktop/output1.jpg' 

centers = detect_black_square_centers(input_image_path, output_image_path)
