import os
import time
import keyboard  # To detect key presses
from PIL import ImageGrab  # To capture screenshots
import uuid  # To generate unique names for screenshots

########## This file is  to collect your own dataset while playing the game ##########

# Paths to save the screenshots for each move
DATASET_PATH = 'final_moves_final'
CATEGORIES = ['up', 'down', 'left', 'right']

# Create directories for each move if they don't exist
for category in CATEGORIES:
    path = os.path.join(DATASET_PATH, category)
    if not os.path.exists(path):
        os.makedirs(path)

# Function to take a screenshot and return it (without saving yet)
def capture_screenshot():
    screenshot = ImageGrab.grab()
    return screenshot

# Function to save the screenshot to a specific category folder
def save_screenshot(screenshot, action):
    # Generate a unique filename using UUID
    filename = os.path.join(DATASET_PATH, action, f"{uuid.uuid4()}.png")
    
    # Save the screenshot in the corresponding folder
    screenshot.save(filename)
    print(f"Screenshot saved for action '{action}' at {filename}")

# Dictionary to map key presses to folder names
key_to_action = {
    'up': 'up',
    'down': 'down',
    'left': 'left',
    'right': 'right',
    # 'n': 'none'
    # All screenshots will be saved in 'nothing' unless another key is pressed
}

# Main loop to capture and save screenshots
print("Press 'up', 'down', 'left', 'right' to save the last screenshot.")
print("Press 'q' to quit.")

last_screenshot = None  # Store the last screenshot

while True:
    # Capture a new screenshot continuously
    last_screenshot = capture_screenshot()
    
    # Check if any of the mapped keys are pressed and save the screenshot to the corresponding folder
    key_pressed = None
    for key, action in key_to_action.items():
        if keyboard.is_pressed(key):
            key_pressed = action
            save_screenshot(last_screenshot, action)  # Save the screenshot in the appropriate folder
            break
    
    # If no key is pressed, save the screenshot to the "nothing" folder
    # if key_pressed is None:
    #     save_screenshot(last_screenshot, 'nothing')
    
    # Break the loop if 'q' is pressed (to quit the program)
    if keyboard.is_pressed('q'):
        print("Exiting...")
        break
    
    time.sleep(0.2)  # Slight delay to control screenshot frequency
