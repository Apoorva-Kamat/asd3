import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_data(data_dir):
    label_map = {'autistic': 1, 'non_autistic': 0}  # Define the label mapping
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (128, 128))
                images.append(image)
                labels.append(label_map[label])  # Use the label mapping
    images = np.array(images)
    labels = np.array(labels)
    images = images.reshape(-1, 128, 128, 1).astype('float32') / 255.0
    labels = to_categorical(labels, num_classes=2)
    return images, labels

# Define the path to the dataset
data_dir = 'consolidated'  # Update this with the actual path

images, labels = load_data(data_dir)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the neural network model
nn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Assuming binary classification
])

nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the neural network model
nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained neural network model
nn_model.save('autism_detection_model.h5')
print("Neural Network model saved successfully!")

# Train and save Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train.reshape(X_train.shape[0], -1), np.argmax(y_train, axis=1))
joblib.dump(dt_model, 'decision_tree_model.pkl')
print("Decision Tree model saved successfully!")

# Train and save Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train.reshape(X_train.shape[0], -1), np.argmax(y_train, axis=1))
joblib.dump(lr_model, 'logistic_regression_model.pkl')
print("Logistic Regression model saved successfully!")
