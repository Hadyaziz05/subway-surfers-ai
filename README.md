﻿# subway-surfers-ai
This repository contains scripts for automating gameplay in Subway Surfers using a machine learning model. The model captures the game screen, predicts actions, and simulates key presses to automate the gameplay.

Table of Contents
play.py - Real-time automation of Subway Surfers
model_and_prediction.py - Model prediction and keypress simulation
model.py - Dataset collection tool
model_training.py - Model training using MobileNetV2
Improvements & Debugging Suggestions
play.py
This script focuses on the real-time automation of playing Subway Surfers by predicting actions and simulating key presses.

Window Focus (focus_bluestacks): Ensures that the BlueStacks emulator (where the game is running) is the active window and ready for automation.
Screenshot Capture (capture_game_screen): Takes a screenshot of the entire screen.
Action Prediction & Automation (monitor_and_play_game): Continuously captures the game screen, predicts the appropriate action using a trained model, and simulates key presses accordingly.
model_and_prediction.py
This script handles loading the trained model, processing images, predicting the action, and simulating key presses.

Image Preprocessing (preprocess_image): Screenshots are resized and normalized for input into the model.
Action Prediction (predict_action): The model predicts the direction the player should move (up, down, left, right). Thresholds prevent unnecessary movements.
Key Press Simulation (press_key): Depending on the predicted action, the corresponding arrow key is pressed using pyautogui.
model.py
This is a dataset collection tool that allows you to collect training data (screenshots) for each action (up, down, left, right).

Screenshot Capture & Saving: Continuously captures screenshots and saves them in folders corresponding to the key pressed. The folders are named based on the direction of movement (e.g., up, down, left, right).
Key Mapping: Screenshots are saved based on the key pressed (e.g., pressing the 'up' key saves the image in the up folder).
model_training.py
This script defines the model training pipeline using MobileNetV2 as a base. It performs data augmentation and fine-tunes the model.

Dataset Loading & Preprocessing (load_dataset): Loads images from the dataset, preprocesses them, and splits them into training and test sets.
Model Definition & Training: The model is built on MobileNetV2 with custom layers on top. After initial training, fine-tuning is done by unfreezing the top layers for further optimization.
