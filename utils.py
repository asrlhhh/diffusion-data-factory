import random
import json
import cv2
import dlib
from deepface import DeepFace
from PIL import Image
import numpy as np

def display_annotations(image, annotations):
    # Define colors for positive and negative labels
    pos_color = (0, 255, 0)   # Green
    neg_color = (0, 0, 255)   # Red

    # Define font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Increase font_scale for larger text
    font_thickness = 1
    
    margin = 10

    # Create a copy of the image to draw the text on
    overlay = image.copy()

    # Iterate through annotations and display them on the image
    for idx, (label, value) in enumerate(annotations.items()):
        color = pos_color if value == 1 else neg_color
        text = f"{label}: {value}"
        y = 20 * (idx + 1)  # Vertical spacing for each annotation (adjust according to the new font_scale)

        # Calculate the width of the text string
        (text_width, _), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        #print(text_width)
        # Calculate the x-coordinate for the right-aligned text
        x = image.shape[1] - text_width - margin

        cv2.putText(overlay, text, (x, y), font, font_scale, color, font_thickness)

    # Blend the original image and the overlay with text
    alpha = 0.8  # Transparency level (0: fully transparent, 1: fully opaque)
    output_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return output_image

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    # Convert the PIL Image object to a NumPy array
    image_np = np.array(image)

    # Convert RGB to BGR (OpenCV uses BGR format)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    return image_np

def save_image(image_np: np.ndarray, output_path: str) -> None:
    # Save the NumPy array as an image file using OpenCV
    cv2.imwrite(output_path, image_np)

def has_single_high_quality_face(img, min_confidence=0.9):
    """
    Check if the input image has one and only one high-quality face.
    
    Args:
        image_path (str): Path to the input image file.
        min_confidence (float): Minimum confidence score to consider a face as high-quality.
        
    Returns:
        bool: True if the image contains one and only one high-quality face, False otherwise.
    """

    # Detect faces using DeepFace
    face_detections = DeepFace.extract_faces(img, detector_backend='retinaface', enforce_detection=False)
    
    # Filter high-quality faces based on confidence
    high_quality_faces = [face for face in face_detections if face['confidence'] >= min_confidence]
    if len(high_quality_faces) != 1:
        return False

    # Use Dlib's shape predictor for facial landmarks
    predictor_path = "./shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()
    dlib_rects = detector(img, 1)
    if len(dlib_rects) != 1:
        return False
    dlib_rect = dlib_rects[0]

    # Get facial landmarks
    shape = predictor(img, dlib_rect)
    padding_ratio = 10
    
    # Calculate the bounding box including hair
    min_x = max(0, dlib_rect.left() - int(padding_ratio * dlib_rect.width()))
    min_y = max(0, dlib_rect.top() - int(2 * padding_ratio * dlib_rect.height()))
    max_x = min(img.shape[1], dlib_rect.right() + int(padding_ratio * dlib_rect.width()))
    max_y = min(img.shape[0], dlib_rect.bottom() + int(padding_ratio * dlib_rect.height()))

    # Check if the entire bounding box is within the image dimensions
    if min_x >= 0 and min_y >= 0 and max_x <= img.shape[1] and max_y <= img.shape[0]:
        return True
    else:
        return False
    
def attach_prompt(attr,direction_lst):

    attached_prompt = ''

    for single_attr, single_direction in zip(attr,direction_lst):
        if single_direction == True:
            attached_prompt = attached_prompt + "with " + single_attr + ", "
        else:
            attached_prompt = attached_prompt +  "without " + single_attr + ", "

    return attached_prompt


def read_attributes_json(json_f = 'attributes.json'):
    """
    Reads the 'attributes.json' file and returns a list of all the attribute values.

    Returns:
        list: A list of all the attribute values in the 'attributes.json' file.
    """
    with open(json_f, 'r') as f:
        attributes = json.load(f)
    #attribute_values = list(attributes.values())
    return attributes

def read_setups_json(json_f = 'setups.json'):
    """
    Reads the 'attributes.json' file and returns a list of all the attribute values.

    Returns:
        list: A list of all the attribute values in the 'attributes.json' file.
    """
    with open(json_f, 'r') as f:
        setups = json.load(f)
    #attribute_values = list(attributes.values())
    return setups

def merge_prompts_setups(attributes_prompts,attributes_setups):
    
    new_dict = {}
    attributes = list(attributes_setups.keys())
    for single_attribute in attributes:
        new_dict[single_attribute] = {}
        new_dict[single_attribute]['prompt'] = attributes_prompts[single_attribute]
        new_dict[single_attribute]['warmups'] = attributes_setups[single_attribute]['warmups']
        new_dict[single_attribute]['cooldowns'] = attributes_setups[single_attribute]['cooldowns']
        
    return new_dict
        

def generate_random_numbers(num_numbers, start=0, end=10000):
    """
    Generates a list of random integers between start and end (inclusive).

    Args:
        num_numbers (int): The number of random numbers to generate.
        start (int, optional): The lower bound of the range. Defaults to 0.
        end (int, optional): The upper bound of the range. Defaults to 10000.

    Returns:
        list: A list of random integers.
    """
    random_numbers = [random.randint(start, end) for _ in range(num_numbers)]
    return random_numbers

def generate_random_bool_list(x):
    """
    Generates a list of `x` random boolean values.

    Args:
        x (int): The number of boolean values to generate.

    Returns:
        list: A list of random boolean values.
    """
    bool_list = [random.choice([True, False]) for _ in range(x)]
    return bool_list


def generate_random_int_list(x, y, z):
    """
    Generates a list of `x` random integers between `y` and `z` (inclusive).

    Args:
        x (int): The number of integers to generate.
        y (int): The lower bound of the range.
        z (int): The upper bound of the range.

    Returns:
        list: A list of random integers between `y` and `z`.
    """
    int_list = [random.randint(y, z) for _ in range(x)]
    return int_list


def generate_random_float_list(x, y, z):
    """
    Generates a list of `x` random floats between `y` and `z` (inclusive).

    Args:
        x (int): The number of floats to generate.
        y (float): The lower bound of the range.
        z (float): The upper bound of the range.

    Returns:
        list: A list of random floats between `y` and `z`.
    """
    float_list = [random.uniform(y, z) for _ in range(x)]
    return float_list

def generate_random_sample_2d_list(x, z, k):
    """
    Generates a 2D list `y` from list `x`, where `y` has `z` number of elements and each element of `y` is a random
    sample of `k` elements from `x`.

    Args:
        x (list): The original list from which to sample.
        y (list): The list to populate with random samples of `x`.
        z (int): The number of elements to create in `y`.
        k (int): The number of elements to include in each random sample.

    Returns:
        list: A 2D list `y` containing `z` random samples of `k` elements from `x`.
    """
    y = []
    for _ in range(z):
        sample = random.sample(x, k)
        y.append(sample)
    return y