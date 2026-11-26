# Image Classification Web App (PyTorch & Streamlit)

**Course:** BSD3513 – Introduction to Artificial Intelligence  
**Institution:** Universiti Malaysia Pahang Al-Sultan Abdullah (UMPSA)  
**Topic:** Chapter 5 – Computer Vision (Lab Report 5)  

## Live Demo
- **Deployed Streamlit App:** `[Insert Your Streamlit Community Cloud URL Here]`
- **Author:** `[Your Full Name]` (`[Your Student ID]`)

---

## Overview
This repository contains a CPU-based Computer Vision web application built using **PyTorch** and **Streamlit**. The application leverages a pre-trained **ResNet-18** convolutional neural network (trained on the ImageNet dataset) to classify user-uploaded images into 1,000 object categories and visualizes the Top-5 prediction probabilities.

### Key Features
- **CPU-Optimized Inference:** Explicitly configured to run efficiently on CPU environments (`torch.device("cpu")`) with gradient computation disabled (`torch.no_grad()`).
- **Standardized Preprocessing:** Automatically applies the official `ResNet18_Weights.DEFAULT` transformations (resizing, center cropping to $224 \times 224$, tensor conversion, and ImageNet normalization).
- **Interactive Visualization:** Displays the uploaded image, Top-5 predicted classes, a detailed probability table, and a Streamlit bar chart.

---

## Repository Structure
```text
├── StudentID_Lab5.py      # Main Streamlit web application
├── requirements.txt       # Python dependencies for deployment
├── screenshots/           # Folder for test result screenshots
└── README.md              # Project documentation
```

---

## Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run StudentID_Lab5.py
   ```

---

## Processing Pipeline
1. **Input Acquisition:** Accepts `.jpg`, `.jpeg`, or `.png` files via Streamlit's file uploader and converts them to RGB format using PIL.
2. **Preprocessing:** Resizes and crops the image to $224 \times 224$ pixels, converts it to a PyTorch tensor, and normalizes pixel values using ImageNet mean and standard deviation.
3. **Model Inference:** Adds a batch dimension (`[1, 3, 224, 224]`) and passes the tensor through `ResNet-18` in evaluation mode (`model.eval()`).
4. **Post-Processing & Output:** Applies the Softmax function to convert raw logits into probabilities, extracts the Top-5 classes via `torch.topk()`, and renders both a DataFrame and a bar chart.

---

## Sample Screenshots
> *Replace the placeholders below with actual screenshots from your running application to complete your documentation.*

| Test Case 1 | Test Case 2 |
| :---: | :---: |
| ![Test 1](screenshots/test1.png) | ![Test 2](screenshots/test2.png) |