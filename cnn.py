import torch
import torch.nn as nn
import cv2
import utils
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.optim as optim

#========================================================================================================
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
image_path = '/Users/phamminhtuan/Desktop/AIChallenge/IMG_1581_iter_0.jpg'
label_path = '/Users/phamminhtuan/Desktop/AIChallenge/IMG_1581_iter_0.txt'
train_dataset = MyDataset(image_path, label_path)
val_dataset = MyDataset(image_path, label_path)
#Model
#========================================================================================================
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

#========================================================================================================
model = EfficientCNN()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
def train_model(train_loader, val_loader, epochs=50, batch_size=64):
    """
    Train the EfficientCNN model on the given training data.

    Args:
        train_loader (DataLoader): A DataLoader for the training data.
        val_loader (DataLoader): A DataLoader for the validation data.
        epochs (int, optional): The number of epochs to train. Defaults to 50.
        batch_size (int, optional): The batch size to use for training. Defaults to 64.

    Returns:
        nn.Module: The trained EfficientCNN model.
    """
    model = EfficientCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    best_val_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            labels = labels.long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                labels = labels.long()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Accuracy: {val_accuracy:.2f}%')
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), '/Users/phamminhtuan/Desktop/AIChallenge/cnn.pth')
    
    return model


train_model(train_loader, val_loader, epochs=50, batch_size=64)
