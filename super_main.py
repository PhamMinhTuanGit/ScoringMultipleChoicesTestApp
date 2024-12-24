import torch
from model import Model
import cv2
from fixed_coordinates import *
import os
from train_utils import *
from result_processsing import *
with open("temp.txt", 'w') as file:
        pass
with open("test.txt", 'w') as file:
        pass
with open("results_test_cnn.txt", 'w') as file:
        pass
device = torch.device("mps")
sections = Section()
model = Model()
# Your existing loop where you get the output from the model
folder_path = "/Users/phamminhtuan/Downloads/testset1/images"
coord_saver = "test.txt"
result_txt_path = 'results_test_fix_bug.txt'
folder_path = "/Users/phamminhtuan/Downloads/testset1/images"
# for prefix in prefixes:
#     common_prefix_files = get_files_with_prefix(folder_path, prefix=prefix)
for filename in os.listdir(folder_path):
     if filename.startswith("IMG_3967"):
        input_path = os.path.join(folder_path, filename)
        fixed_circle(input_path, coord_saver)
        labels, coords = read_file_to_tensors(coord_saver)
        image = cv2.imread(input_path)
        h, w = image.shape[:2]
        with open("temp.txt", 'w') as file:
                pass
        with open("test.txt", 'w') as file:
                pass
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
                        x1,y1,x2,y2 = coord
                        with open("temp.txt", 'a') as f:     
                            f.write(f"0 {x1} {y1} {x2} {y2}\n")  # Ghi một dòng tọa độ
        getSubmitResult(input_path, "temp.txt", result_txt_path)         

                    
