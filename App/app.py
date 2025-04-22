import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "../Model/model.keras")
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Preprocessing function
def preprocess_image(image):
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized) / 255.0
    image_input = np.expand_dims(image_array, axis=0)
    return image_input, image_array

# Prediction function
def predict_oscc(image):
    if image is None:
        return None, "Please upload an image."

    image_input, image_array = preprocess_image(image)
    prediction = model.predict(image_input)[0][0]
    label = "Effected (OSCC Detected)" if prediction > 0.5 else "Not Effected (Healthy)"
    display_preprocessed = (image_array * 255).astype(np.uint8)
    display_preprocessed = Image.fromarray(display_preprocessed)
    return display_preprocessed, label

# Resolve full path of image
image_path = os.path.join(os.path.dirname(__file__), "architecture.png")

# Gradio Blocks UI
with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("## Oral Cancer Detection (OSCC)")

    with gr.Row():
        home_btn = gr.Button("Home")
        prediction_btn = gr.Button("Prediction")
        model_btn = gr.Button("Model")

    home_page = gr.Column(visible=True)
    prediction_page = gr.Column(visible=False)
    model_page = gr.Column(visible=False)

    with home_page:
        gr.Markdown("### Welcome to the OSCC Detection App")
        gr.Markdown(
            "Click on **Prediction** to upload an image and check for signs of Oral Squamous Cell Carcinoma, or **Model** to learn more about the underlying model."
        )
        gr.Image(value=image_path, label="Model Architecture Diagram")

    with prediction_page:
        image_input = gr.Image(type="pil", label="", show_label=False)
        submit_btn = gr.Button("Submit")
        preprocessed_output = gr.Image(label="Preprocessed Image (224x224)")
        prediction_output = gr.Textbox(label="Prediction Result")

    with model_page:
        gr.Markdown("### Model Details")
        gr.Markdown("""
- **Architecture**: Convolutional Neural Network with X convolutional layers and Y dense layers.
- **Input Size**: 224×224 RGB images.
- **Output**: Binary classification (OSCC detected vs. Healthy).
- **Framework**: TensorFlow Keras.
- **Training Data**: Description of dataset used.
- **Performance**: Accuracy: 0.XX, Precision: 0.XX, Recall: 0.XX.
""")

    def show_home():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    def show_prediction():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

    def show_model():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    home_btn.click(fn=show_home, outputs=[home_page, prediction_page, model_page])
    prediction_btn.click(fn=show_prediction, outputs=[home_page, prediction_page, model_page])
    model_btn.click(fn=show_model, outputs=[home_page, prediction_page, model_page])

    submit_btn.click(
        fn=predict_oscc,
        inputs=[image_input],
        outputs=[preprocessed_output, prediction_output]
    )

if __name__ == "__main__":
    demo.launch(pwa=True)

    
