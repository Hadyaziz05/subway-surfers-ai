import time
import pygetwindow as gw
import pyautogui
from PIL import ImageGrab
from model_and_prediction import predict_action, press_key

########## This file is for playing the game with the model ##########

# Function to focus on the BlueStacks window
def focus_bluestacks():
    windows = gw.getWindowsWithTitle("BlueStacks")
    if windows:
        bluestacks_window = windows[0]
        bluestacks_window.restore()  # Restores the window if minimized
        bluestacks_window.activate()  # Bring the window to the foreground
        pyautogui.click(bluestacks_window.left + 50, bluestacks_window.top + 400)  # Click inside the window
    else:
        print("BlueStacks window not found.")

# Function to capture the game screen from BlueStacks
def capture_game_screen():
    screenshot = ImageGrab.grab()  # Capture the screen
    return screenshot

# Main loop to continuously monitor the game, take screenshots, and simulate key presses
def monitor_and_play_game():
    focus_bluestacks()  # Focus on the game window
    time.sleep(2)  # Give some time to ensure the window is active

    print("Starting game automation...")

    while True:
        screenshot = capture_game_screen()  # Take a screenshot
        action = predict_action(screenshot)  # Predict the action using the trained model
        press_key(action)  # Press the appropriate key based on the predicted action

        time.sleep(0.5)  # Add a small delay to avoid flooding the game with too many inputs

if __name__ == "__main__":
    monitor_and_play_game()  # Start the game automation
