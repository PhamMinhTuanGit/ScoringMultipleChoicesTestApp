import numpy as np
import pandas as pd
import image_processing
import grid_info
import visualization
from utilities import extract_bubbles, generate_random_colors, append_to_file
from bubble_classify import BubbleClassifier
import os
import detect_black_square
def getSubmitResult(input_image_path, input_data, result_txt_path):
    """
    Processes an input image and data to extract grid and bubble information and writes results to a text file.

    :param input_image_path: Path to the input image file.
    :param input_data: Tuple with the format like the txt file [() () () () ()]
    :param result_txt_path: Path to the output results text file.
    """
    
    try:
        # Predefine configurations for sections
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
            {"name": "phan3_1", "section_index": (3, 0), "axis": 1, "eps": 0.002, "input_string": "3.1", "gap_string": "none"},
            {"name": "phan3_2", "section_index": (3, 1), "axis": 1, "eps": 0.002, "input_string": "3.2", "gap_string": "none"},
            {"name": "phan3_3", "section_index": (3, 2), "axis": 1, "eps": 0.002, "input_string": "3.3", "gap_string": "none"},
            {"name": "phan3_4", "section_index": (3, 3), "axis": 1, "eps": 0.002, "input_string": "3.4", "gap_string": "none"},
            {"name": "phan3_5", "section_index": (3, 4), "axis": 1, "eps": 0.002, "input_string": "3.5", "gap_string": "none"},
            {"name": "phan3_6", "section_index": (3, 5), "axis": 1, "eps": 0.002, "input_string": "3.6", "gap_string": "none"}
        ]

        # Step 1: Detect nail centers in the image
        centers = detect_black_square.detect_black_square_centers(input_image_path)
        #print("Centers:", centers)
        filename = os.path.basename(input_image_path)
        # Step 2: Extract grid information from nail centers
        gridmatrix = grid_info.getGridmatrix(centers)
        #print("Grid matrix:", gridmatrix)
        gridsection = grid_info.getExtractsections(gridmatrix)
        #print("Grid sections:", gridsection)
        # Step 3: Extract bubble data from input file
        dots = input_data
        #print("Dots:", dots)
        # Step 4: Classify bubbles and write results
        append_to_file(result_txt_path, filename+' ')
        bubble_classifier = BubbleClassifier(gridsection, dots)
        for section in sections:
            bubble_classifier.process_grid_section(
                section_index=section["section_index"],
                axis=section["axis"],
                eps=section["eps"],
                input_string=section["input_string"],
                gap_string=section["gap_string"],
                output_file=result_txt_path
            )
            print(f"Processed {section['name']}...")
        append_to_file(result_txt_path,"\n")

    except Exception as e:
        print(f"Error occurred during processing: {e}")

# Example usage
if __name__ == "__main__":
    input_image_path = "IMG_1581_iter_1.jpg"
    input_data = extract_bubbles('IMG_1581_iter_1.txt')
    result_txt_path = 'results_test_template.txt'
    
    getSubmitResult(input_image_path, input_data, result_txt_path)
