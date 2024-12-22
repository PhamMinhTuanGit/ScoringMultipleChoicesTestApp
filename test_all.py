import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from utilities import extract_bubbles, append_to_file
from image_processing import getNails
from grid_info import getGridmatrix, getExtractsections
from fixed_coordinates import fixed_circle
img_path = "IMG_1581_iter_0.jpg"
coord_saver = "test.txt"
fixed_circle(img_path, coord_saver)
















class EfficientCNN(nn.Module):
    def __init__(self):
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


class BubbleClassifier:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ])

    def classify_bubble(self, bubble_image):
        input_tensor = self.transform(bubble_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            return probs[0, 0].item()  # P(label=0)

    def process_grid_section(self, grid_section, dots):
        """
        Classifies bubbles within a grid section.

        :param grid_section: Coordinates of the grid section.
        :param dots: List of bubble locations in the grid section.
        :return: List of tuples (x, y, P(label=0)).
        """
        results = []
        for dot in dots:
            x, y, bubble_img = dot["x"], dot["y"], dot["image"]
            prob_label_0 = self.classify_bubble(bubble_img)
            results.append((x, y, prob_label_0))
        return results


def getSubmitResult(input_image_path, input_data, result_txt_path):
    """
    Processes an input image and data to extract grid and bubble information and writes results to a text file.

    :param input_image_path: Path to the input image file.
    :param input_data: Path to the input data file containing bubble information.
    :param result_txt_path: Path to the output results text file.
    """
    try:
        # Load the pre-trained model
        device = torch.device("mps") if torch.has_mps else torch.device("cpu")
        model = EfficientCNN()
        model.load_state_dict(torch.load('/Users/phamminhtuan/Desktop/AIChallenge/cnn.pth'))
        model.to(device)
        model.eval()

        # Initialize the BubbleClassifier
        bubble_classifier = BubbleClassifier(model, device)

        # Predefine configurations for sections
        sections = [
            {"name": "SBD", "section_index": (0, 0)},
            {"name": "MDT", "section_index": (0, 1)},
            {"name": "phan1_1", "section_index": (1, 0)},
            {"name": "phan1_2", "section_index": (1, 1)},
            # Add more sections as needed
        ]

        # Step 1: Detect nail centers in the image
        centers = getNails(input_image_path)

        # Step 2: Extract grid information from nail centers
        gridmatrix = getGridmatrix(centers)
        gridsection = getExtractsections(gridmatrix)

        # Step 3: Extract bubble data from input file
        dots = extract_bubbles(input_data)

        # Step 4: Classify bubbles and write results
        append_to_file(result_txt_path, input_image_path)

        for section in sections:
            print(f"Processing {section['name']}...")

            # Get the grid section coordinates
            grid_section = gridsection[section["section_index"][0]][section["section_index"][1]]

            # Filter dots for this section
            section_dots = [dot for dot in dots if dot_in_section(dot, grid_section)]

            # Classify bubbles in the section
            bubble_probs = bubble_classifier.process_grid_section(grid_section, section_dots)

            # Find the bubble with the highest P(label=0)
            if bubble_probs:
                highest_prob_bubble = max(bubble_probs, key=lambda x: x[2])  # x[2] is P(label=0)
                x, y, _ = highest_prob_bubble
                append_to_file(result_txt_path, f"{section['name']}: ({x}, {y})\n")

    except Exception as e:
        print(f"Error occurred during processing: {e}")


def dot_in_section(dot, grid_section):
    """
    Checks if a dot is within the given grid section.

    :param dot: Dictionary with dot coordinates (x, y).
    :param grid_section: Tuple of grid section coordinates (x1, y1, x2, y2).
    :return: Boolean, True if dot is in section, False otherwise.
    """
    x, y = dot["x"], dot["y"]
    x1, y1, x2, y2 = grid_section
    return x1 <= x <= x2 and y1 <= y <= y2


# Example usage
if __name__ == "__main__":
    input_image_path = '/Users/phamminhtuan/Desktop/Trainning_SET/Images/IMG_1581_iter_0.jpg'
    input_data = '/Users/phamminhtuan/Desktop/AIChallenge/picked_output.txt'
    result_txt_path = 'results_test_cnn.txt'
    getSubmitResult(input_image_path, input_data, result_txt_path)
