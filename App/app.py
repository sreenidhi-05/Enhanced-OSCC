import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.keras")
if not os.path.exists(model_path):
    raise RuntimeError(f"Model file not found at {model_path}")
model = tf.keras.models.load_model(model_path)

# Preprocessing function
def preprocess_image(image):
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized) / 255.0
    image_input = np.expand_dims(image_array, axis=0)
    return image_input, image_array

# Prediction and UI update function
def predict_and_show(image):
    try:
        if image is None:
            return (
                gr.update(value=None, visible=False),
                gr.update(value="Please upload an image.", visible=True),
                gr.update(value='<div id="blur-overlay" style="display:none;"></div>')
            )

        image_input, image_array = preprocess_image(image)
        prediction = model.predict(image_input)[0][0]
        label = "Effected (OSCC Detected)" if prediction > 0.5 else "Not Effected (Healthy)"

        display_preprocessed = (image_array * 255).astype(np.uint8)
        display_preprocessed = Image.fromarray(display_preprocessed)

        return (
            gr.update(value=display_preprocessed, visible=True),
            gr.update(value=label, visible=True),
            gr.update(value='<div id="blur-overlay" style="display:none;"></div>')
        )
    except Exception as e:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=f"Prediction failed: {str(e)}", visible=True),
            gr.update(value='<div id="blur-overlay" style="display:none;"></div>')
        )

# Reset outputs when a new image is selected
def reset_outputs():
    return (
        gr.update(value=None, visible=False),
        gr.update(value="", visible=False),
        gr.update(value='<div id="blur-overlay" style="display:none;"></div>')
    )

# Show blur overlay
def show_blur():
    return gr.update(value='<div id="blur-overlay" style="display:flex;">Please wait...</div>')

# Static image paths
home_image_path = os.path.join(os.path.dirname(__file__), "architecture.png")
model_image_path = os.path.join(os.path.dirname(__file__), "model_architecture.png")

# Gradio UI
with gr.Blocks(css="""
#blur-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(3px);
    z-index: 999;
    display: none;
    font-size: 1.5rem;
    color: black;
}
""") as demo:

    gr.Markdown("## Oral Cancer Detection (OSCC)")

    with gr.Tabs():
        with gr.Tab("Home"):
            gr.Markdown("### Welcome to the OSCC Detection")
            gr.Markdown('Visit *Prediction* Tab to upload an image')
            gr.Image(value=home_image_path, label="Model Architecture Diagram", width=500, height=300)
            gr.Markdown(
                """### Model Architecture Description
This diagram shows the process of building and using a deep learning model to detect Oral Squamous Cell Carcinoma (OSCC) from images. 
Images are split into train, validation, and test sets. The model is trained, evaluated, and deployed via tools like Gradio. To know the more about model visit *Model* Tab.
""" 
            )
            gr.Markdown("Visit the [GitHub Repository](https://github.com/sreenidhi-05/Enhanced-OSCC) for more details.")
            gr.Markdown("To know more details view our [documentation](https://www.overleaf.com/read/gqbzgwbchjrd#072f5b)")

        with gr.Tab("Prediction"):
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Submit")
            preprocessed_output = gr.Image(value=None, label="Preprocessed Image (224x224)", visible=False)
            prediction_output = gr.Textbox(value="", label="Prediction Result", visible=False)
            blur_overlay = gr.HTML('<div id="blur-overlay"></div>')

            # Show blur overlay first
            submit_btn.click(fn=show_blur, inputs=[], outputs=[blur_overlay])

            # Run prediction (with queue so it runs after blur is shown)
            submit_btn.click(
                fn=predict_and_show,
                inputs=[image_input],
                outputs=[preprocessed_output, prediction_output, blur_overlay],
                queue=True
            )

            # Reset when image changes
            image_input.change(
                fn=reset_outputs,
                inputs=[],
                outputs=[preprocessed_output, prediction_output, blur_overlay]
            )

        with gr.Tab("Model"):
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
                - **Accuracy**: 0.91 
                - **Precision**: 0.91  
                - **Recall**: 0.91  
                """)
        with gr.Tab("Team"):
            gr.Markdown("### 👥 Meet the Team")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    <div style="border: 1px solid #f9f9f9; border-radius: 15px; padding: 15px; text-align: center;">
                        <h3>P. Sai Sreenidhi</h3>
                        <p><strong>Role:</strong> Model Developer</p>
                        <p style="font-style: italic; color: #FFFAF0;">"Crafting a robust deep learning model for early OSCC detection has been both challenging and incredibly rewarding."</p>
                    </div>
                    """)

                with gr.Column():
                    gr.Markdown("""
                    <div style="border: 1px solid #f9f9f9; border-radius: 15px; padding: 15px; text-align: center;">
                        <h3>K. Gayathri Chinmayee</h3>
                        <p><strong>Role:</strong> UI Developer</p>
                        <p style="font-style: italic; color: #FFFAF0;">"Designing a user-friendly and intuitive interface was my top priority. I loved making something so impactful look beautiful."</p>
                    </div>
                    """)

                with gr.Column():
                    gr.Markdown("""
                    <div style="border: 1px solid #f9f9f9; border-radius: 15px; padding: 15px; text-align: center;">
                        <h3>Ayesha Tabassum</h3>
                        <p><strong>Role:</strong> Deployer</p>
                        <p style="font-style: italic; color: #FFFAF0;">"Ensuring that this model works seamlessly in deployment has been a great learning experience — bridging tech with real-world use."</p>
                    </div>
                    """)


# Run the app
if __name__ == "__main__":
    demo.launch(pwa=True)
