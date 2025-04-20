import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("model/model.keras")  # Make sure path is correct

# Print model summary to confirm it's loaded
model.summary()

# Create a dummy input image (same shape as training images)
dummy_image = np.random.rand(1, 224, 224, 3)  # 1 sample, 224x224 RGB

# Make prediction
prediction = model.predict(dummy_image)

print("Prediction on dummy image:", prediction)
