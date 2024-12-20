from fixed_coordinates import fixed_circle
import utils
import torchvision 
import cv2
from torchvision import transforms
import torch.nn as nn
import torch
from train_utils import *

class EfficientCNN(nn.Module):
    def __init__(self):
        """
        Initializes the EfficientCNN model. This model is an efficient convolutional neural network for the 
        classification task. It consists of a feature extraction part and a classification part. The feature 
        extraction part is a convolutional neural network which consists of 3 convolutional layers with 
        maxpooling, where the number of channels are 32, 64, 128 respectively. The classification part is a 
        fully connected neural network which consists of 2 fully connected layers with dropout rate 0.5.

        Args:
            None
        """
        super(EfficientCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
device = torch.device("mps")
model = EfficientCNN()
model.load_state_dict(torch.load('/Users/phamminhtuan/Desktop/AIChallenge/cnn.pth', weights_only=True))
model = model.to(device)
image_path = "/Users/phamminhtuan/Desktop/Trainning_SET/Images/IMG_1581_iter_0.jpg"
label_path = "/Users/phamminhtuan/Desktop/Trainning_SET/Labels/IMG_1581_iter_0.txt"


def handle_predicted_zero(file_path, label, x1, y1, x2, y2):
    """
    Ghi thông tin vào file khi predicted == 0.
    
    Args:
        file_path (str): Đường dẫn tới file cần ghi.
        label (int): Nhãn dự đoán.
        x1, y1, x2, y2 (int): Tọa độ của hình chữ nhật.
    """
    with open(file_path, 'a') as f:  # Mở file ở chế độ append
        f.write(f"{label} {x1} {y1} {x2} {y2}\n")

picked_output_path = "/Users/phamminhtuan/Desktop/AIChallenge/picked_output.txt"
image = cv2.imread(image_path)
labels, coords = read_file_to_tensors(label_path)
for i, coord in enumerate(coords):
    xy_coord = find_yolov8_square(image, coord)
    x1, y1, x2, y2 = xy_coord
    input = get_box(image, xy_coord)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))
    ])
    input = transform(input)
    input = input.to(device)
    output = model(input.unsqueeze(0))
    _, predicted = torch.max(output.data, 1)

    if predicted == 0:
        # Vẽ hình chữ nhật lên ảnh
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Ghi thông tin vào file picked_output.txt
        a1, b1, a2, b2 = coord
        handle_predicted_zero(picked_output_path, int(predicted.item()), a1, b1, a2, b2)

cv2.imwrite("/Users/phamminhtuan/Desktop/AIChallenge/avg_coords.jpg", image)

# for i, coord in enumerate(coords):
#     xy_coord = find_yolov8_square(image, coord)
#     x1, y1, x2, y2 = xy_coord
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# cv2.imwrite("/Users/phamminhtuan/Desktop/AIChallenge/avg_coords.jpg", image)