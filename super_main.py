import torch
from model import Model
import cv2
from fixed_coordinates import *
import os
from train_utils import *
from result_processsing import *
from bubble_classify import *
with open("temp.txt", 'w') as file:
        pass
with open("test.txt", 'w') as file:
        pass
with open("results_test_cnn.txt", 'w') as file:
        pass
device = torch.device("mps")
sections = Section()
model = Model().to(device)
# Your existing loop where you get the output from the model
sections = [
            {"name": "SBD", "section_index": (0, 0), "axis": 0, "eps": 0.002, "input_string": "SBD", "gap_string": 0},
            {"name": "MDT", "section_index": (0, 1), "axis": 0, "eps": 0.002, "input_string": "MDT", "gap_string": 0},
            {"name": "phan1_1", "section_index": (1, 0), "axis": 1, "eps": 0.002, "input_string": "1.", "gap_string": 0},
            {"name": "phan1_2", "section_index": (1, 1), "axis": 1, "eps": 0.002, "input_string": "1.", "gap_string": 10},
            {"name": "phan1_3", "section_index": (1, 2), "axis": 1, "eps": 0.002, "input_string": "1.", "gap_string": 20},
            {"name": "phan1_4", "section_index": (1, 3), "axis": 1, "eps": 0.002, "input_string": "1.", "gap_string": 30},
            {"name": "phan2_1", "section_index": (2, 0), "axis": 1, "eps": 0.002, "input_string": "2.1", "gap_string": "a"},
            {"name": "phan2_2", "section_index": (2, 1), "axis": 1, "eps": 0.002, "input_string": "2.2", "gap_string": "a"},
            {"name": "phan2_3", "section_index": (2, 2), "axis": 1, "eps": 0.002, "input_string": "2.3", "gap_string": "a"},
            {"name": "phan2_4", "section_index": (2, 3), "axis": 1, "eps": 0.002, "input_string": "2.4", "gap_string": "a"},
            {"name": "phan2_5", "section_index": (2, 4), "axis": 1, "eps": 0.002, "input_string": "2.5", "gap_string": "a"},
            {"name": "phan2_6", "section_index": (2, 5), "axis": 1, "eps": 0.002, "input_string": "2.6", "gap_string": "a"},
            {"name": "phan2_7", "section_index": (2, 6), "axis": 1, "eps": 0.002, "input_string": "2.7", "gap_string": "a"},
            {"name": "phan2_8", "section_index": (2, 7), "axis": 1, "eps": 0.002, "input_string": "2.8", "gap_string": "a"},
            {"name": "phan3_1", "section_index": (3, 0), "axis": 0, "eps": 0.002, "input_string": "3.1", "gap_string": "none"},
            {"name": "phan3_2", "section_index": (3, 1), "axis": 0, "eps": 0.002, "input_string": "3.2", "gap_string": "none"},
            {"name": "phan3_3", "section_index": (3, 2), "axis": 0, "eps": 0.002, "input_string": "3.3", "gap_string": "none"},
            {"name": "phan3_4", "section_index": (3, 3), "axis": 0, "eps": 0.002, "input_string": "3.4", "gap_string": "none"},
            {"name": "phan3_5", "section_index": (3, 4), "axis": 0, "eps": 0.002, "input_string": "3.5", "gap_string": "none"},
            {"name": "phan3_6", "section_index": (3, 5), "axis": 0, "eps": 0.002, "input_string": "3.6", "gap_string": "none"}
        ]
coord_saver = "test.txt"
result_txt_path = 'results_test_cnn.txt'
folder_path = "/Users/phamminhtuan/Downloads/images 2"
# for prefix in prefixes:
#     common_prefix_files = get_files_with_prefix(folder_path, prefix=prefix)
for filename in os.listdir(folder_path):
        print("Processing file:", filename)
        picked_coord = []
        with open("temp.txt", 'w') as file:
            pass
        with open("test.txt", 'w') as file:
            pass
        input_path = os.path.join(folder_path, filename)
        fixed_circle(input_path, coord_saver)
        labels, coords = read_file_to_tensors(coord_saver)
        image = cv2.imread(input_path)
        h, w = image.shape[:2]
        # image = cv2.imread(input_path)
        # h, w = image.shape[:2]
        # input_data = extract_bubbles(coord_saver)
        # # Step 1: Detect nail centers in the image
        # centers = detect_black_square.detect_black_square_centers(input_path)
        # print("Centers:", centers)
        # # Step 2: Extract grid information from nail centers
        # gridmatrix = grid_info.getGridmatrix(centers)
        # print("Grid matrix:", gridmatrix)
        # gridsection = grid_info.getExtractsections(gridmatrix)
        # #print("Grid sections:", gridsection)
        # # Step 3: Extract bubble data from input file
        # dots = input_data
        # bubble_classifier = BubbleClassifier(gridsection, dots)


        # for section in sections:
        #         corners = gridsection[section["section_index"][0]][section["section_index"][1]]
        #         dots_inside = bubble_classifier.dots_in_quadrilateral(corners)
        #         clustered = bubble_classifier.classify_batches(dots_inside, axis=section["axis"], eps=section["eps"])
        #     # Chuyển coords từ tensor sang list
        #         for i in range(len(clustered)):
        #             best_val = -99999
        #             largest_prob_dot = None
        #             for j in range(len(clustered[i])):
        #                     coord = clustered[i][j]
        #                     _, x1,y1,x2,y2 = coord
        #                     xy_coord = find_yolov8_square(image, (x1,y1,x2,y2))
                            
                        
        #                     input = get_box(image, xy_coord)
        #                     transform = transforms.Compose([
        #                         transforms.ToTensor(),
        #                         transforms.Resize((32, 32))
        #                     ])
        #                     input = transform(input).to(device)
        #                     output = model(input.unsqueeze(0))
        #                     # _, predicted = torch.max(output, 1)
        #                     temp = output[0][0].item()
        #                     if temp > best_val:
        #                         # Thêm tuple (0, x1, y1, x2, y2) vào danh sách
        #                         best_val = temp
        #                         largest_prob_dot = (x1, y1, x2, y2)
                                
        #             if largest_prob_dot is not None:
                            
        #                     x1,y1,x2,y2 = largest_prob_dot
        #                     picked_coord.append((0,x1,y1,x2,y2))
        # getSubmitResult(input_path, picked_coord, result_txt_path)


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
                        picked_coord.append((0,x1/w,y1/h,x2/w,y2/h))
                        cv2.rectangle(image, (x1,y1), (x2,y2),(0,255,0),2)
        getSubmitResult(input_path, picked_coord, result_txt_path)
        cv2.imwrite("avg_coords.jpg", image)