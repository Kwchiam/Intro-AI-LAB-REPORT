# Genetic Algorithm Search Visualizer 🧬

*(https://sd24040ga.streamlit.app/)*

## 📌 Project Overview

This repository contains a Python-based web application developed for **BSD3513: Introduction to Artificial Intelligence (Lab 2: Search Algorithms)**. The application is built using Streamlit to interactively visualize the evolutionary process of a Genetic Algorithm (GA) designed to optimize an 80-bit binary pattern.

The algorithm simulates natural selection—utilizing evaluation, selection, crossover, and mutation—to evolve a population of 300 individuals over 50 generations. The specific goal of this implementation is to reach a target fitness where the chromosome contains exactly 50 ones out of its 80-bit length.

## ✨ Features

- **Interactive GA Control Options:** Dynamically adjust the algorithm's behavior via the sidebar:
  - **Operator Selection:** Choose selection methods (e.g., Tournament with adjustable size $k$).
  - **Crossover Method:** Configurable crossover techniques (e.g., Single-Point).
  - **Hyperparameters:** Fine-tune the Crossover Rate ($P_c$) and Mutation Rate ($P_m$ per bit).
  - **Execution Settings:** Options to bias the initial population (90% zeros to emphasize the learning curve), enable early stopping upon reaching max fitness, and set random seeds for reproducibility.

- **Evolutionary Tracking:** Monitors and displays the best individual found, the highest fitness score (target: 80/80), and the exact bitstring composition (Ones vs. Zeros).

- **Real-time Convergence Visualization:** Generates a dynamic line chart comparing the **Best Fitness** (elite individual) against the **Average Fitness** of the population across generations, complete with a visual marker for when the target maximum fitness is reached.

## 🚀 Live Demo

You can access the live deployed application here: **[https://sd24040ga.streamlit.app/]**

## 📸 Screenshots

**1. Setup**
![Setup](Screenshot%20Output/Setup.png)

**2. Convergence Graph**
![Graph](Screenshot%20Output/Graph.png)

## 🛠️ Installation & Local Setup

To run this application locally on your machine, follow these instructions:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Kwchiam/Linear%20Search.git
   cd "Genetic Search LAB 2"
   ```

2. **Install the required modules:**
   Ensure you have Python installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**

   ```bash
   streamlit run SD24040_Lab2.py
   ```

## 🗂️ Project Structure / Modules

- `SD24040_Lab2.py`: The main Streamlit application script containing the UI layout, plotting functions, and the core Genetic Algorithm logic (Evaluation, Selection, Crossover, and Mutation).

- `requirements.txt`: Contains all necessary Python dependencies (e.g., `streamlit`, `numpy`, `matplotlib`) for deployment.

## 🏷️ Tags

`Artificial Intelligence`, `Genetic Algorithms`, `Search Algorithms`, `Evolutionary Computing`, `Streamlit`, `Python`, `Data Visualization`
