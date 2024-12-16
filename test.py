import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class OMRGrader:
    def __init__(self, answer_key_path, template_path):
        """
        Initialize the OMR grader with answer key and template
        
        :param answer_key_path: Path to the CSV file containing the answer key
        :param template_path: Path to the exam template image
        """
        self.answer_key = self.load_answer_key(answer_key_path)
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        # Set up the neural network for grayscale threshold adjustment
        self.nn_model = self._build_nn_model(self.template.shape)
        self.nn_model.compile(optimizer='adam', loss='mean_squared_error')
        
    def load_answer_key(self, answer_key_path):
        """
        Load the answer key from a CSV file
        
        :param answer_key_path: Path to the CSV file
        :return: Dictionary of correct answers {question_number: correct_option}
        """
        answer_key = pd.read_csv(answer_key_path)
        return {row['question']: row['answer'] for _, row in answer_key.iterrows()}
        
    def _build_nn_model(self, image_shape):
        """
        Build a simple neural network model to adjust the grayscale threshold
        
        :param image_shape: Shape of the exam template image
        :return: Compiled Keras model
        """
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(image_shape[0] * image_shape[1],)))
        model.add(Dense(1, activation='sigmoid'))
        return model
        
    def preprocess_image(self, image_path):
        """
        Preprocess the scanned exam sheet
        
        :param image_path: Path to the scanned exam image
        :return: Preprocessed binary image
        """
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Adjust the grayscale threshold using the neural network
        img_height, img_width = img.shape
        img = img.reshape(1, img_height * img_width)
        img = img / 255.0
        threshold = self.nn_model.predict(img)[0][0]
        binary = cv2.threshold(img.reshape(img_height, img_width), threshold * 255, 255, cv2.THRESH_BINARY_INV)[1]
        
        return binary
    
    def detect_marks(self, binary_image, num_questions, options_per_question):
        """
        Detect marked bubbles in the exam sheet
        
        :param binary_image: Preprocessed binary image
        :param num_questions: Total number of questions
        :param options_per_question: Number of options for each question
        :return: Dictionary of student's answers
        """
        student_answers = {}
        
        for q in range(num_questions):
            for opt in range(options_per_question):
                # Define bubble region (you'd need precise coordinates)
                x, y, w, h = self.get_bubble_coordinates(q, opt)
                
                # Extract bubble region
                bubble_region = binary_image[y:y+h, x:x+w]
                
                # Count non-zero pixels to determine if marked
                marked_pixels = cv2.countNonZero(bubble_region)
                total_pixels = w * h
                
                # Mark is considered if more than 40% of pixels are non-zero
                if marked_pixels / total_pixels > 0.4:
                    student_answers[q+1] = opt + 1
                    break
        
        return student_answers
    
    def grade_exam(self, student_answers):
        """
        Compare student answers with answer key and calculate score
        
        :param student_answers: Dictionary of student's answers
        :return: Exam result details
        """
        total_questions = len(self.answer_key)
        correct_answers = 0
        
        for q, correct_ans in self.answer_key.items():
            if student_answers.get(q) == correct_ans:
                correct_answers += 1
        
        score_percentage = (correct_answers / total_questions) * 100
        
        return {
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'score_percentage': score_percentage,
            'missed_questions': [q for q in self.answer_key if student_answers.get(q) != self.answer_key[q]]
        }
    
    def get_bubble_coordinates(self, question, option):
        """
        Placeholder method to return bubble coordinates
        In a real implementation, this would be precisely mapped
        
        :param question: Question number
        :param option: Option number
        :return: Coordinates (x, y, width, height)
        """
        # This is a dummy implementation
        # You'd replace with actual coordinate mapping
        base_x, base_y = 100, 200  # Starting point
        bubble_width, bubble_height = 20, 20
        horizontal_spacing = 30
        vertical_spacing = 40
        
        x = base_x + (option * horizontal_spacing)
        y = base_y + (question * vertical_spacing)
        
        return (x, y, bubble_width, bubble_height)

# Example usage
grader = OMRGrader('answer_key.csv', 'exam_template.jpg')
processed_image = grader.preprocess_image('student_exam.jpg')
student_answers = grader.detect_marks(processed_image, num_questions=3, options_per_question=4)
result = grader.grade_exam(student_answers)

print(result)