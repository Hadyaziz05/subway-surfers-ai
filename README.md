# Subway Surfers AI

This repository contains scripts for automating gameplay in *Subway Surfers* using a machine learning model. The model captures the game screen, predicts actions, and simulates key presses to automate the gameplay.

## Table of Contents
- [play.py](#playpy) - Real-time automation of Subway Surfers
- [model_and_prediction.py](#model_and_predictionpy) - Model prediction and keypress simulation
- [model.py](#modelpy) - Dataset collection tool
- [model_training.py](#model_trainingpy) - Model training using MobileNetV2
- [Improvements & Debugging Suggestions](#improvements--debugging-suggestions)

---

## play.py

This script focuses on the real-time automation of playing Subway Surfers by predicting actions and simulating key presses.

### Functions:
- *Window Focus (focus_bluestacks)*:  
  Ensures that the BlueStacks emulator (where the game is running) is the active window and ready for automation.
  
- *Screenshot Capture (capture_game_screen)*:  
  Takes a screenshot of the entire screen.
  
- *Action Prediction & Automation (monitor_and_play_game)*:  
  Continuously captures the game screen, predicts the appropriate action using a trained model, and simulates key presses accordingly.

---

## model_and_prediction.py

This script handles loading the trained model, processing images, predicting actions, and simulating key presses.

### Functions:
- *Image Preprocessing (preprocess_image)*:  
  Screenshots are resized and normalized for input into the model.
  
- *Action Prediction (predict_action)*:  
  The model predicts the direction the player should move (up, down, left, right). Thresholds are implemented to prevent unnecessary movements.
  
- *Key Press Simulation (press_key)*:  
  Depending on the predicted action, the corresponding arrow key is pressed using pyautogui.

---

## model.py

This script is a dataset collection tool that allows you to gather training data (screenshots) for each action (up, down, left, right).

### Features:
- *Screenshot Capture & Saving*:  
  Continuously captures screenshots and saves them in folders corresponding to the key pressed. The folders are named based on the direction of movement (e.g., up, down, left, right).
  
- *Key Mapping*:  
  Screenshots are saved based on the key pressed (e.g., pressing the 'up' key saves the image in the up folder).

---

## model_training.py

This script defines the model training pipeline using *MobileNetV2* as a base. It performs data augmentation and fine-tunes the model.

### Key Steps:
- *Dataset Loading & Preprocessing (load_dataset)*:  
  Loads images from the dataset, preprocesses them, and splits them into training and test sets.
  
- *Model Definition & Training*:  
  The model is built on *MobileNetV2* with custom layers added on top. After the initial training, fine-tuning is done by unfreezing the top layers for further optimization.

---

## Improvements & Debugging Suggestions

Here are some areas for improvement and suggestions for debugging common issues:

- *Window Focus*:  
  Ensure that the correct window is in focus before capturing screenshots.
  
- *Screenshot Quality*:  
  Verify that the resolution and preprocessing of screenshots match the model's input requirements.
  
- *Prediction Threshold*:  
  Adjust thresholds for action prediction to avoid erratic or unnecessary movements.
  
- *Model Accuracy*:  
  Fine-tune the model further by experimenting with different layers and optimization techniques to improve performance.
