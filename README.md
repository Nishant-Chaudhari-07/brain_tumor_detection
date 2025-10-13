# 🧠 Brain Tumor Detection — Streamlit App

Upload a brain MRI image and get a model-generated prediction. Built for easy deployment to Streamlit Cloud.

**Last updated:** 2025-10-13 19:26:40 UTC

## Features
- Drag-and-drop image upload
- Works with **binary** or **multi-class** Keras models
- Probability bar chart
- Optional Grad‑CAM heatmap

## Structure
```
.
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── src/
│   ├── predict.py
│   └── gradcam.py
├── models/
│   ├── brain_tumor_classifier.keras   # <-- add your model
│   └── class_names.json               # <-- add your label list
└── data/
    └── .gitkeep
```