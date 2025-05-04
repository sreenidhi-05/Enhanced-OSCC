# A Deep Convolutional Neural Network for Oral Squamous Cell Carcinoma from Histopathological Images
This project presents a deep learning-based approach for the early detection of Oral Squamous Cell Carcinoma (OSCC) from histopathological images. Leveraging Convolutional Neural Networks (CNNs) and transfer learning models (VGG16, VGG19), the system aims to provide a fast, accurate, and real-time diagnostic tool.

## Project Highlights
- Real-time OSCC detection from histopathological images
- A CNN model is designed and compared with VGG16 and VGG19
- Publicly accessible tool designed for early cancer screening
  
## Models Used
- Proposed CNN model: A CNN model with deep layers is designed.
- VGG16 and VGG19: Transfer learning architecture is used, but weights are trained from scratch.
  
## Dataset
- Source: [Kaggle-OSCC Histopathology dataset](https://www.kaggle.com/datasets/ashenafifasilkebede/dataset/data)
- Total images 1224 was collected and then extended to 5192 for training purposes
  
## Performance metrics
- Confusion matrix , loss curves and auc curves are shown in Results folder

## How to Run
- Clone the respository
- Install required packages:'pip install -r requirements.txt'
- Run the notebook: In jupyter notebook and save the model in models folder
- Launch Gradio app:'python app.py'
- Check the .yml file in GitHub Workflow and deploy it to your Hugging Face Space
  
## 🧪 Experimental Results
| Model            | Accuracy  |
|------------------|-----------|
| Proposed CNN     | 91%       |
| VGG16            | 90%       |
| VGG19            | 88%       |

## Visit our real-time oral cancer detection tool on [Hugging Face Space](https://huggingface.co/spaces/Sreenidhi31/Enhanced-OSCC?logs=container)



