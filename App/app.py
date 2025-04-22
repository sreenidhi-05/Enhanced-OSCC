import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

model_path = os.path.join(os.path.dirname(__file__), "model.keras")
if not os.path.exists(model_path):
    raise RuntimeError(f"Model file not found at {model_path}")

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
    
def predict_and_show(image):
    preprocessed, result = predict_oscc(image)
    return gr.update(value=preprocessed, visible=True), gr.update(value=result, visible=True)

# Image paths
home_image_path = os.path.join(os.path.dirname(__file__), "architecture.png")
model_image_path = os.path.join(os.path.dirname(__file__), "model_architecture.png")

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

    # Home Page
    with home_page:
        gr.Markdown("### Welcome to the OSCC Detection App")
        gr.Markdown(
            "Click on **Prediction** to upload an image and check for signs of Oral Squamous Cell Carcinoma, or **Model** to learn more about the underlying model."
        )
        gr.Image(value=home_image_path, label="Model Architecture Diagram",width=500,height=300)
        gr.Markdown(
        """### Model Architecture Description
        The architecture diagram above illustrates the structure of the model used for OSCC detection. 
        It highlights the various layers, including convolutional and fully connected layers, that work together to classify images of oral lesions into 'Normal' and 'OSCC' categories.
        To know more about the Model visit *Model* tab"""
        )

    # Prediction Page
    with prediction_page:
        image_input = gr.Image(type="pil", label="", show_label=False)
        submit_btn = gr.Button("Submit")
        preprocessed_output = gr.Image(label="Preprocessed Image (224x224)",visible=False)
        prediction_output = gr.Textbox(label="Prediction Result",visible=False)

    # Model Page
    with model_page:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Architecture")
                gr.Image(value=model_image_path, label="Architecture")

            with gr.Column(scale=1):
                gr.Markdown("### Model Description & Metrics")
                gr.Markdown("""
**Input Size**: 224×224×3 (RGB Images)  

**Architecture**:
- 6 convolutional blocks:
  - Each with 2 Conv2D layers (last block has 1)
  - BatchNormalization + ReLU after each layer
  - MaxPooling2D after each block
- Filters: **32 → 64 → 128 → 256 → 512 → 1024**
- Total Conv2D layers: **11**
- Final conv output: **7×7×1024**, flattened to **9216**

**Fully Connected Layers**:
- Dense: **512 → 256 → 128 → 1**
- Dropout after first two dense layers

**Output**: Binary classification — OSCC Detected / Healthy  
**Trainable Parameters**: ~14 Million  

**Performance**:
- **Accuracy**: 0.89  
- **Precision**: 0.91  
- **Recall**: 0.91  
""")

    # Navigation functions
    def show_home():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    def show_prediction():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

    def show_model():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    # Button logic
    home_btn.click(fn=show_home, outputs=[home_page, prediction_page, model_page])
    prediction_btn.click(fn=show_prediction, outputs=[home_page, prediction_page, model_page])
    model_btn.click(fn=show_model, outputs=[home_page, prediction_page, model_page])

    # Submit logic
    submit_btn.click(
        fn=predict_and_show,
        inputs=[image_input],
        outputs=[preprocessed_output, prediction_output]
    )

# Launch app
if __name__ == "__main__":
    demo.launch(pwa=True)
