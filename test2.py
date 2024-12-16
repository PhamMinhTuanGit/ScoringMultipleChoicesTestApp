import cv2
import numpy as np

def process_omr_sheet(input_image_path, output_image_path):
    """
    Process an OMR sheet image and save the result.
    
    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to the output image.
    """
    image = cv2.imread(input_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if 20 < w < 50 and 20 < h < 50 and 0.8 < aspect_ratio < 1.2:
            detected_boxes.append((x, y, w, h))
    # Sort the boxes based on their position in the image
    detected_boxes = sorted(detected_boxes, key=lambda b: (b[1], b[0]))
    for box in detected_boxes:
        x, y, w, h = box
        # Extract the region of interest (ROI) from the thresholded image
        roi = thresh[y:y+h, x:x+w]
        # Calculate the ratio of the number of non-zero pixels to the total size of the box
        filled_ratio = cv2.countNonZero(roi) / (w * h)
        # Check if the box is filled
        if filled_ratio > 0.5:
            # Draw a green rectangle around the filled box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Save the result image
    cv2.imwrite(output_image_path, image)
    print(f"Processed image saved to: {output_image_path}")
input_image_path = '/Users/phamminhtuan/Downloads/Trainning_SET/Images/IMG_1581_iter_0.jpg' 
output_image_path = '/Users/phamminhtuan/Desktop/output.jpg'  
process_omr_sheet(input_image_path, output_image_path)
