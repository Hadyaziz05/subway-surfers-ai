# subway-surfers-ai
1. play.py
This script focuses on the real-time automation of playing the game by predicting actions and simulating key presses.

Window Focus (focus_bluestacks): Ensures that the BlueStacks emulator (where the game is running) is the active window and ready for automation.
Screenshot Capture (capture_game_screen): Takes a screenshot of the entire screen.
Action Prediction & Automation (monitor_and_play_game): This continuously captures the game screen, predicts the appropriate action using a trained model, and simulates key presses accordingly.
2. model_and_prediction.py
This script handles loading the trained model, processing images, predicting the action, and simulating key presses.

Image Preprocessing (preprocess_image): Screenshots are resized and normalized for input into the model.
Action Prediction (predict_action): The model predicts which direction the player should move (up, down, left, right), and thresholds prevent unnecessary movements.
Key Press Simulation (press_key): Depending on the predicted action, the corresponding arrow key is pressed using pyautogui.
3. model.py
This is a dataset collection tool that allows you to collect training data (screenshots) for each action (up, down, left, right).

Screenshot Capture & Saving: Continuously captures screenshots and saves them in folders corresponding to the key pressed. The folders are named based on the direction of movement.
Key Mapping: Screenshots are saved based on the key pressed (e.g., 'up' saves the image in the 'up' folder).
4. model_training.py
This script defines the model training pipeline using MobileNetV2 as a base, performs data augmentation, and fine-tunes the model.

Dataset Loading & Preprocessing (load_dataset): Loads images from the dataset, preprocesses them, and splits them into training and test sets.
Model Definition & Training: The model is built on MobileNetV2 and further trained with custom layers on top. After initial training, fine-tuning is done by unfreezing the top layers for further optimization.
Improvements & Debugging Suggestions:
Model Thresholding: You have different thresholds for actions, which is good for eliminating low-confidence predictions, but you might want to implement additional fine-tuning to balance responsiveness and accuracy.

Performance Optimization:

The use of ImageGrab might slow down your process. Consider using more efficient libraries or focusing the capture area on the game window only.
Augmented Data: The data augmentation strategy you’ve implemented should help with creating a more robust model. If your dataset is small, you may want to try additional augmentation techniques (such as contrast changes or different lighting conditions).

Key Simulation Tuning: Ensure the timing of pyautogui.press is well-suited to the game mechanics. Games often require certain key press timings for smooth controls.

Testing Fine-tuning: After fine-tuning the model, test extensively to ensure the additional training benefits performance in real-world scenarios.