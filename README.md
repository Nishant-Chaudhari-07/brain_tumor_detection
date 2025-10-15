# 🧠 Brain Tumor Detection — Deep Learning with Streamlit

🎯 **Live Demo:** [Brain Tumor Detection App](https://braintumordetection-aaef3bkcrtznxfsvzvngpt.streamlit.app/)  
An interactive web application that classifies brain MRI scans as **Tumor** or **No Tumor** using a deep learning model built with TensorFlow and deployed via Streamlit.


## 🚀 Project Overview
This project leverages **Convolutional Neural Networks (CNNs)** and **Transfer Learning** to automate brain tumor detection from MRI scans.  
The model was trained on a dataset of labeled MRI brain images and fine-tuned using **VGG16** (pretrained on ImageNet).  
It demonstrates how AI and deep learning can assist medical professionals in identifying tumors quickly and accurately.

**Objectives:**
- Build a deep learning model capable of accurately detecting brain tumors from MRI images.  
- Provide interpretable visualizations using Grad-CAM to show which parts of the brain influenced predictions.  
- Deploy the model as a live, user-friendly web app using Streamlit.


## ⚙️ Key Features
✅ **Transfer Learning with VGG16** — uses pretrained CNN layers to improve accuracy with limited data.  
✅ **Interactive Web App** — drag-and-drop MRI images to get instant predictions.  
✅ **Explainability with Grad-CAM** — visualize which regions in the scan influenced the model’s output.  
✅ **Training Diagnostics** — displays training/validation accuracy, loss curves, and confusion matrix.  
✅ **Google Drive Model Loading** — large model files automatically download via `gdown`.  
✅ **Clean Architecture** — modular code separated into `src/` for prediction logic and `app.py` for deployment.


## 📂 File Structure
```bash

brain_tumor_detection/
├── app.py # Streamlit app (main entry point)
├── requirements.txt # Dependencies list
├── README.md # Project documentation
├── .streamlit/
│ └── config.toml # Streamlit UI theme
├── src/
│ ├── init.py
│ ├── predict.py # Model loading and inference functions
│ └── gradcam.py # Grad-CAM visualization utility
├── models/
│ ├── brain_tumor_classifier.keras # Trained model (auto-downloaded)
│ ├── class_names.json # Label mapping
│ ├── training_curves.png # Training/validation accuracy & loss curves
│ ├── confusion_matrix.png # Confusion matrix visualization
│ └── metrics.json (optional) # Validation accuracy, loss, AUC summary
└── data/
└── .gitkeep

```

## 📊 Model Performance

| Metric | Value |
|:--|:--|
| **Validation Accuracy** | ~92% |
| **Validation Loss** | ~0.28 |
| **AUC (Area Under Curve)** | ~0.95 |
| **Architecture** | VGG16 + Custom Dense Layers |
| **Optimizer** | Adam (lr = 1e-3) |
| **Loss Function** | Categorical Cross-Entropy |
| **Input Size** | 224 × 224 × 3 |

### 📈 Training Curves  
The training and validation curves show smooth convergence — validation accuracy stabilizes around 92%, indicating good generalization.  

### 🧩 Confusion Matrix  
The confusion matrix demonstrates strong true positive and true negative performance, confirming the model’s ability to distinguish tumor vs. non-tumor classes.


## 💡 Use Cases
- 🏥 **Medical Imaging Support:** Assist radiologists by flagging potential tumor regions for review.  
- 🧪 **Research & Education:** Showcase deep learning applications in medical image analysis.  
- ⚙️ **AI Diagnostics:** Integrate into healthcare systems for automated MRI triage or screening.  
- 🌐 **Portfolio Demonstration:** An end-to-end ML project—training, evaluation, and web deployment.


## 📘 About the Model
- **Base Model:** VGG16 pretrained on ImageNet for feature extraction.  
- **Custom Layers:**  
  `AveragePooling2D → Flatten → Dense(64, relu) → Dropout(0.5) → Dense(2, softmax)`  
- **Training:** 10 epochs, batch size 8, Adam optimizer (lr=1e-3).  
- **Dataset:** MRI brain images labeled *Tumor* / *No Tumor*.  
- **Explainability:** Grad-CAM highlights regions most influential in the model’s prediction.

