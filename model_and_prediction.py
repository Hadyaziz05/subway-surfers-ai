import numpy as np
import cv2
import tensorflow as tf
import pyautogui

########## This file is  to collect your own dataset while playing the game ##########


IMG_SIZE = 256  # The size of the image that the model expects
CATEGORIES = ['down', 'right' , 'left', 'up']

# These thresholds to prevent the character from moving unnecessary movments (an alternative aproach is to make a fifth class called none where the character should not move)
THRESHOLD = 0.55
THRESHOLDU = 0.87
THRESHOLDD = 0.43
THRESHOLDL = 0.7
# Load the trained model
model = tf.keras.models.load_model('subway_surfers_ai_model.h5')

# Preprocess the screenshot before feeding it to the model
def preprocess_image(image):
    image = np.array(image)  # Convert the image to a numpy array
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize to the same size as the training images
    image = image.reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # Ensure it has 3 color channels (RGB)
    image = image / 255.0  # Normalize pixel values (0 to 1)
    return image

# Predict the action (up, down, left, or right) based on the screenshot
def predict_action(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    max_confidence = np.max(prediction)  # Get the highest confidence score
    predicted_label = np.argmax(prediction)  # Get the label with the highest score
    print("score", max_confidence)
    if  CATEGORIES[predicted_label] == 'right' and max_confidence < THRESHOLD:
        return 'nothing'  # If confidence is below threshold, no movement
    elif  CATEGORIES[predicted_label] == 'left' and max_confidence < THRESHOLDL:
        return 'nothing'  # If confidence is below threshold, no movement
    elif  CATEGORIES[predicted_label] == 'up' and max_confidence < THRESHOLDU:
         return 'nothing'  # If confidence is below threshold, no movement
    elif  CATEGORIES[predicted_label] == 'down' and max_confidence < THRESHOLDD:
        return 'nothing'  # If confidence is below threshold, no movement
    else:
        return CATEGORIES[predicted_label]

# Function to simulate key press based on prediction
def press_key(action):
    if action == 'up':
        pyautogui.press('up')
    elif action == 'down':
        pyautogui.press('down')
    elif action == 'left':
        pyautogui.press('left')
    elif action == 'right':
        pyautogui.press('right')
    print(f"Action predicted: {action} - Key pressed!")