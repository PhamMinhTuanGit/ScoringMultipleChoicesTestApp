import os
import torch
import torch.nn as nn
import cv2
import utils
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.optim as optim
image_folder = "/Users/phamminhtuan/Desktop/Trainning_SET/Images"
label_folder = "/Users/phamminhtuan/Downloads/Trainning_SET/Labels"
class MyDataset(torch.utils.data.Dataset):
    """
    Initializes the dataset by reading the image and label files, and processes them into tensors.

    Args:
        image_path (str): Path to the input image.
        label_path (str): Path to the label file (YOLO format).

    The method reads the image from the specified path and the label data, then processes each labeled 
    bounding box to crop and transform the image into a tensor. The processed tensors and corresponding 
    labels are stored in `self.inputs` and `self.targets` respectively.
    """
    def __init__(self, image_path, label_path):
        super().__init__()
        
        self.image_path = image_path
        self.label_path = label_path
        self.inputs, self.targets = [], []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ])
        image = cv2.imread(self.image_path)
        labels, coords = utils.read_file_to_tensors(self.label_path)
        for i, coord in enumerate(coords):
            square = utils.find_yolov8_square(image, coord)
            cropped_image = utils.get_box(image, square)
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_tensor = self.transform(cropped_image)
            self.inputs.append(cropped_tensor)
            self.targets.append(labels[i])
    def __getitem__(self, idx):
        image = self.inputs[idx]
        label = self.targets[idx]
        return image, label
    def __len__(self):
        return len(self.inputs)
train_img = []
train_label = []
for filename in os.listdir(image_folder):
    if filename.endswith('iter_0.jpg'):  # Lọc các file hình ảnh
        image_path = os.path.join(image_folder, filename)
        label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + ".txt")  # Giả định file nhãn cùng tên với file ảnh
        temp = MyDataset(image_path,label_path)
        for i in range(len(temp)):
            train_img.append(temp[i][0])
            train_label.append(temp[i][1])

