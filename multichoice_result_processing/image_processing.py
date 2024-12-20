import cv2

def getNails(image_path):
    """
    Detects black squares in a given image and saves the output to a file.
    
    This function takes an image path, an output image path, and an output text path as input. 
    It reads the image, processes it to find the centers of black squares, and saves the results 
    in both an annotated image and a text file in YOLOv8 format.
    
    The coordinates of each detected square's center are also written beside the corresponding box 
    in the annotated image.
    
    :param image_path: Path to the input image
    :param output_image_path: Path to the annotated output image
    :param output_txt_path: Path to the output text file
    :return: List of centers of the black squares in YOLOv8 format
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    image_height, image_width = image.shape[:2]

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon is a quadrilateral
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Ensure it's a square and of a reasonable size
            if 0.9 < aspect_ratio < 1.1 and w > 20 and h > 20:
                roi = thresh[y:y+h, x:x+w]
                black_pixels = cv2.countNonZero(roi)
                total_pixels = w * h
                fill_ratio = black_pixels / total_pixels

                # Check if the square is mostly filled
                if fill_ratio > 0.9:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])

                        # Convert to YOLOv8 format
                        yolov8_x = center_x / image_width
                        yolov8_y = center_y / image_height

                        centers.append((yolov8_x, yolov8_y))

                        # Annotate the image
                        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Write coordinates beside the black box (Black text color)
                        coordinate_text = f"({yolov8_x:.4f}, {yolov8_y:.4f})"
                        cv2.putText(image, coordinate_text, (center_x + 10, center_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Save the annotated image
    #cv2.imwrite(output_image_path, image)

    # Save the centers to a text file
    # with open(output_txt_path, 'w') as file:
    #     for center in centers:
    #         file.write(f"{center[0]:.6f}, {center[1]:.6f}\n")

    #print(f"Processed image saved to: {output_image_path}")
    #print(f"Centers of black squares saved to: {output_txt_path}")
    return centers