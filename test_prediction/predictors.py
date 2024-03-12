import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("models\\upgraded_DS.h5")
# model = load_model("models\\fruits.h5")
model = load_model("models\\fruits_transfer_learning.h5")

# Function to preprocess the input image
def preprocess_image(image):
    # Resize the image to match the input size expected by the model
    resized_image = cv2.resize(image, (224, 224))  # Resize to (224, 224)
    # Normalize pixel values to the range [0, 1]
    normalized_image = resized_image / 255.0
    # Expand dimensions to match the expected input shape of the model
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

# Function to make predictions on the image using the loaded model
def make_predictions(image):
    # Preprocess the input image
    preprocessed_image = preprocess_image(image)
    # Make predictions using the loaded model
    predictions = model.predict(preprocessed_image)
    return predictions

# Load the image
# image = cv2.imread("PerfectApple.15.jpg")
image = cv2.imread("DamageApple.13.jpg")

# Make predictions on the resized image
predictions = make_predictions(image)
print(predictions)

# Interpret the predictions
if predictions[0][0] >= 0.50:
    print("The fruit is fresh.")
else:
    print("The fruit is rotten.")
