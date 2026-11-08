# Neural Network Activation Functions Visualizer 🧠


*(https://sd24040lab4.streamlit.app/)*

## 📌 Project Overview

This repository contains a Python-based web application developed for **BSD3513: Introduction to Artificial Intelligence (Lab 4)**. The application is built using Streamlit and `scikit-learn` to interactively visualize how different neural network activation functions (ReLU, Sigmoid, and Tanh) handle data classification.

By allowing users to generate both linear and non-linear datasets (such as Moons and Circles), the application plots decision boundaries from a Multi-Layer Perceptron (MLP) and provides the mathematical formulas to effectively explain the complexity of neural networks.

## ✨ Features

* **Interactive Sidebar:** Dynamically adjust data generation (samples, noise) and model parameters (epochs, random seed).

* **Dataset Generation:** Toggle between linearly separable data and complex non-linear structures (Moons, Circles).

* **Activation Functions:** Choose between three distinct models:

  1. **Rectified Linear Unit (ReLU):** $f(x) = \max(0, x)$

  2. **Sigmoid:** $f(x) = \frac{1}{1 + e^{-x}}$

  3. **Hyperbolic Tangent (Tanh):** $f(x) = \tanh(x)$

* **Real-time Visualization:** Displays the calculated decision boundary alongside the mathematical activation curve.

## 🚀 Live Demo

You can access the live deployed application here: **\[https://sd24040lab4.streamlit.app/]**

## 📸 Screenshots

**1. ReLU Network**
![ReLU Output](Screenshot%20Output/ReLU%20network.png)

**2. Sigmoid Network**
![Sigmoid Output](Screenshot%20Output/Sigmoid%20Network.png)

**3. Tanh Network**
![Tanh Output](Screenshot%20Output/Tanh%20Network.png)

## 🛠️ Installation & Local Setup

To run this application locally on your machine, follow these instructions:

1. **Clone the repository:**

   ```
   git clone [https://github.com/Kwchiam/Intro-AI-LAB-REPORT.git](https://github.com/Kwchiam/Intro-AI-LAB-REPORT.git)
   cd "Intro-AI-LAB-REPORT/Neural Networks LAB 4"
   
   ```

2. **Install the required modules:**
   Ensure you have Python installed, then run:

   ```
   pip install -r requirements.txt
   
   ```

3. **Run the Streamlit application:**

   ```
   streamlit run SD24040_Lab4.py
   
   ```

## 🗂️ Project Structure / Modules

* `SD24040_Lab4.py`: The main Streamlit application script containing the UI, data generation, and MLP model training logic.

* `requirements.txt`: Contains all necessary Python dependencies (`streamlit`, `numpy`, `matplotlib`, `scikit-learn`) for deployment.

## 🏷️ Tags

`Artificial Intelligence`, `Neural Networks`, `Machine Learning`, `Streamlit`, `Python`, `Scikit-Learn`, `Data Visualization`
