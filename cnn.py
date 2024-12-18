import torch
import torch.nn as nn
import cv2
import utils
import torchvision.transforms as transforms

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path):
        """
        Args:
            image_path (str): Path to the input image.
            label_path (str): Path to the label file (YOLO format).
        """
        super().__init__()
        self.image_path = image_path
        self.label_path = label_path
        self.inputs, self.targets = [], []
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Resize((28, 28))  # Resize image to 28x28
        ])
        image = cv2.imread(self.image_path)

        # Read label and coordinates
        labels, coords = utils.read_file_to_tensors(self.label_path)

        # Process each coordinate (bounding box)
        for i, coord in enumerate(coords):
            square = utils.find_yolov8_square(image, coord)  # Get bounding box
            cropped_image = utils.get_box(image, square)  # Crop the image
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Apply transformation
            cropped_tensor = self.transform(cropped_image)  # Convert to tensor & resize
            
            self.inputs.append(cropped_tensor)
            self.targets.append(labels[i])  # Append corresponding label

        # Convert inputs and targets to tensors
        
       
    def __getitem__(self, idx):
        image = self.inputs[idx]
        label = self.targets[idx]
        return image, label

    def __len__(self):
        return len(self.inputs)
image_path = '/Users/phamminhtuan/Desktop/AIChallenge/IMG_1581_iter_0.jpg'
label_path = '/Users/phamminhtuan/Desktop/AIChallenge/IMG_1581_iter_0.txt'
train_dataset = MyDataset(image_path, label_path)
