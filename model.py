import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator

########## This file is the CNN model I used mobilenet_v2 ##########

# Define to_categorical function from TensorFlow
to_categorical = tf.keras.utils.to_categorical

# Paths to the folders with the images for each move
DATASET_PATH = 'final_moves_final'
# You can change the order based on your dataset
CATEGORIES = ['down', 'right' , 'left', 'up']
IMG_SIZE = 256  # Size to resize images to


# Function to load and preprocess the dataset
def load_dataset():
    data = []
    labels = []
    
    for category in CATEGORIES:
        path = os.path.join(DATASET_PATH, category)
        class_num = CATEGORIES.index(category)  # Assign a numeric label for each category
        
        for img_name in os.listdir(path):
            try:
                # Load image in grayscale and resize to (IMG_SIZE, IMG_SIZE)
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                # Convert to 3-channel image by duplicating the grayscale channel
                img = np.stack((img,)*3, axis=-1)
                
                # Append the image data and its corresponding label
                data.append(img)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    
    # Convert data and labels to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    return data, labels

# Load dataset
data, labels = load_dataset()

# Normalize and preprocess pixel values for MobileNetV2 (-1 to 1)
data = tf.keras.applications.mobilenet_v2.preprocess_input(data)

# One-hot encode the labels
labels = to_categorical(labels, num_classes=len(CATEGORIES))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Display an example image from the dataset
img = data[92]
img = img.reshape(IMG_SIZE, IMG_SIZE, 3)
print("Label[92]", labels[92])
plt.imshow(img, cmap='gray')
plt.title(f"Example Image - Index {92}")
plt.axis('off')  # Hide axes
plt.show()

# Data Augmentation: Adding more aggressive transformations
datagen = ImageDataGenerator(
    rotation_range=30,       # Rotate images randomly by 30 degrees
    width_shift_range=0.3,   # Shift images horizontally by 30% of total width
    height_shift_range=0.3,  # Shift images vertically by 30% of total height
    shear_range=0.2,         # Shear angle (slanting)
    zoom_range=0.3,          # Random zoom
    horizontal_flip=True,    # Random horizontal flips
    fill_mode='nearest'      # Fill in missing pixels after transformation
)

# Fit the datagen to the training data
datagen.fit(X_train)

# Load MobileNetV2 with pretrained weights, excluding the top layers
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False, 
                                               weights='imagenet')
base_model.trainable = False  # Freeze base model layers

# Add custom layers on top
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(CATEGORIES), activation='softmax') #softmax since it is a multi-class classification return 1 in the correct class
])

# Compile the model with Adam optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=16),
                    validation_data=(X_test, y_test),
                    epochs=20)

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Save the trained model
model.save('subway_surfers_ai_model.h5')

# Optional: Fine-tuning (Unfreeze some layers)
base_model.trainable = True

# Unfreeze the top 20 layers of the MobileNetV2 model
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Re-compile the model for fine-tuning with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training for fine-tuning
fine_tune_history = model.fit(datagen.flow(X_train, y_train, batch_size=16),
                              validation_data=(X_test, y_test),
                              epochs=10)

# Plot fine-tuning accuracy
plt.plot(fine_tune_history.history['accuracy'], label='Train Accuracy (Fine-tune)')
plt.plot(fine_tune_history.history['val_accuracy'], label='Validation Accuracy (Fine-tune)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the fine-tuned model on the test set
fine_tune_test_loss, fine_tune_test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Fine-tuned test accuracy: {fine_tune_test_acc}")

# Save the fine-tuned model
model.save('subway_surfers_ai_model_finetuned.h5')