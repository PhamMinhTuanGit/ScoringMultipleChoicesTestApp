import torch
from model import Model
import cv2
from fixed_coordinates import *
import os
from train_utils import *
from result_processsing import *

device = torch.device("mps")
sections = Section()
model = Model()
# Your existing loop where you get the output from the model
folder_path = "/Users/phamminhtuan/Desktop/Trainning_SET/Images"
coord_saver = "test.txt"
result_txt_path = 'results_test_cnn.txt'
for filename in os.listdir(folder_path):
    with open("temp.txt", 'w') as file:
        pass
    with open("test.txt", 'w') as file:
        pass
    if filename.startswith("IMG_1581"):
        input_path = os.path.join(folder_path, filename)
        fixed_circle(input_path, coord_saver)
        image = cv2.imread(os.path.join(folder_path, filename))
        h, w = image.shape[:2]
        labels, coords = read_file_to_tensors(coord_saver)
        for i, coord in enumerate(coords):
            xy_coord = find_yolov8_square(image, coord)
            x1, y1, x2, y2 = xy_coord
            input = get_box(image, xy_coord)
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
             ])
            input = transform(input).to(device)
            output = model(input.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            if(predicted == 0):
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                with open("temp.txt", 'a') as f:     
                    f.write(f"0 {x1/w} {y1/h} {x2/w} {y2/h}\n")  # Ghi một dòng tọa độ
        getSubmitResult(input_path, "temp.txt", result_txt_path)
        cv2.imwrite("/Users/phamminhtuan/Desktop/AIChallenge/output1.jpg", image)         

            
