#Step 1: Imports and Data Loading
#First, we import the necessary libraries and load the image paths.

# Import necessary libraries
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import os
import cv2
import numpy as np

# Define the path to your dataset
# Replace this string with the actual path to your 'brain_tumor_dataset' folder
path = "./Desktop/DataFlair/brain_tumor_dataset"

# Get a list of all image paths in that directory
image_paths = list(paths.list_images(path))

#Step 2: Image Preprocessing
#We loop through every image path, read the image, resize it to 224x224 (required for VGG16), and extract the label from the folder name.

# Initialize lists to store the images and their labels
images = []
labels = []

# Loop over every image path found
for image_path in image_paths:
    # Extract the label (folder name) from the path. 
    # splits the path by separator and takes the second to last item (folder name 'yes' or 'no')
    label = image_path.split(os.path.sep)[-2]
    
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Resize the image to 224x224 pixels to match VGG16 input size
    image = cv2.resize(image, (224, 224))
    
    # Add the processed image to the list
    images.append(image)
    
    # Add the extracted label to the list
    labels.append(label)

#Step 3: Data Normalization and Encoding
#Machine learning models need numerical data. We normalize pixel values to the range 0-1 and convert string labels ("yes"/"no") into binary vectors (One-Hot Encoding).

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

# Convert the image list to a NumPy array and normalize pixel values (0 to 1)
images = np.array(images) / 255.0

# Convert the labels list to a NumPy array
labels = np.array(labels)

# Initialize the LabelBinarizer to convert string labels to integers
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)

# Convert the binary integers to categorical vectors (one-hot encoding)
# e.g., 'yes' becomes [0, 1] and 'no' becomes [1, 0]
labels = to_categorical(labels)

#Step 4: Splitting the Data
#We split the data into training (90%) and testing (10%) sets.

from sklearn.model_selection import train_test_split

# Split the data
# train_X, test_X: Image data
# train_Y, test_Y: Labels
# test_size=0.10 means 10% is used for testing
# stratify=labels ensures the ratio of 'yes'/'no' is consistent in both splits
(train_X, test_X, train_Y, test_Y) = train_test_split(images, labels, test_size=0.10, random_state=42, stratify=labels)

#Step 5: Data Augmentation
#Since the dataset is small, we use augmentation (rotation, zooming, etc.) to artificially increase the variety of training data.

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize the training data generator with augmentation rules
train_generator = ImageDataGenerator(
    fill_mode='nearest',  # How to fill missing pixels after rotation
    rotation_range=15     # Rotate images randomly up to 15 degrees
)

#Step 6: Building the Model (VGG16 Transfer Learning)
#We load the pre-trained VGG16 model (without its top classification layer) and add our own layers for binary classification.

from tensorflow.keras.layers import Input, Dense, AveragePooling2D, Dropout, Flatten
from tensorflow.keras.applications import VGG16

# Load the VGG16 network, ensuring the head FC layer sets are left off (include_top=False)
base_model = VGG16(weights='imagenet', input_tensor=Input(shape=(224, 224, 3)), include_top=False)

# Construct the head of the model that will be placed on top of the the base model
base_input = base_model.input
base_output = base_model.output

# Add a pooling layer to reduce dimensions
base_output = AveragePooling2D(pool_size=(4, 4))(base_output)

# Flatten the output to feed into dense layers
base_output = Flatten(name="flatten")(base_output)

# Add a dense hidden layer with 64 neurons and ReLU activation
base_output = Dense(64, activation="relu")(base_output)

# Add dropout to prevent overfitting (50% dropout rate)
base_output = Dropout(0.5)(base_output)

# Add the final output layer with 2 neurons (for 2 classes) and softmax activation
base_output = Dense(2, activation="softmax")(base_output)

#Step 7: Freezing Layers and Compilation
#We freeze the VGG16 layers so we don't destroy the patterns it has already learned (ImageNet data). We only train the new layers we added.

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Freeze the layers of the base model so they are not updated during training
for layer in base_model.layers:
    layer.trainable = False

# Combine base input and new output into the final model
model = Model(inputs=base_input, outputs=base_output)

# Compile the model
# Optimizer: Adam with learning rate 0.001
# Loss: Binary Crossentropy (standard for two-class classification)
model.compile(optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'], loss='binary_crossentropy')

# Display the model architecture
model.summary()

#Step 8: Training the Model
#We define hyperparameters and start the training process.

# Define hyperparameters
batch_size = 8
train_steps = len(train_X) // batch_size
validation_steps = len(test_X) // batch_size
epochs = 10

# Train the model
history = model.fit(
    train_generator.flow(train_X, train_Y, batch_size=batch_size),
    steps_per_epoch=train_steps,
    validation_data=(test_X, test_Y),
    validation_steps=validation_steps,
    epochs=epochs
)

#Step 9: Evaluation
#Finally, we predict on the test set and print the classification report and confusion matrix.

from sklearn.metrics import classification_report, confusion_matrix

# Make predictions on the testing set
predictions = model.predict(test_X, batch_size=batch_size)

# Find the index of the label with corresponding largest predicted probability
predictions = np.argmax(predictions, axis=1)
actuals = np.argmax(test_Y, axis=1)

# Print the classification report
print(classification_report(actuals, predictions, target_names=label_binarizer.classes_))

# Compute and print the confusion matrix
cm = confusion_matrix(actuals, predictions)
print(cm)
