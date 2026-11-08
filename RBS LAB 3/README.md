# Scholarship Advisory Rule-Based System 🎓

*(https://sd24040lab3.streamlit.app/)*

## 📌 Project Overview

This repository contains a Python-based web application developed for **BSD3513: Introduction to Artificial Intelligence (Lab 3)**. The application is built using Streamlit to serve as a transparent decision support tool for a university's scholarship committee. 

Instead of manual screening, this Knowledge Representation System utilizes a **Rule-Based Engine** to evaluate applicant facts (such as CGPA, family income, and disciplinary records) against a set of customizable criteria. It helps the committee automatically determine the eligibility and specific type of scholarship to be awarded based on priority logic.

## ✨ Features

- **Interactive Applicant Profile:** A sidebar to dynamically input applicant facts, including CGPA, Monthly Family Income, Co-curricular Score, Community Service Hours, Current Semester, and Disciplinary Actions.
- **Dynamic Rule Configuration:** A live JSON editor allowing the academic and financial aid committees to review, adjust, and add new rules instantly without changing the core application code.
- **Priority-Based Evaluation Engine:** Evaluates conditions logically using Python's operator mapping. The engine sorts and executes rules based on their assigned `priority` weight (highest to lowest).
- **Transparent Decision Output:** Clearly displays the final decision (e.g., `AWARD_FULL`, `AWARD_PARTIAL`, `REVIEW`, `REJECT`), the triggering rule name, and the specific reasoning behind the decision.

## 🚀 Live Demo

You can access the live deployed application here: **\[https://sd24040lab3.streamlit.app/]**

## 📸 Screenshots

**1. Applicant Fact Input & Rule Editor**
![Input Interface](Screenshot%20Output/Full_Award1.png) 

**2. Decision Engine Output (Successful Award)**
![Award Output](Screenshot%20Output/Partial_Award1.png) 

**3. Decision Engine Output (Rejection)**
![Reject Output](Screenshot%20Output/Reject_Decision.png) 

## 🛠️ Installation & Local Setup

To run this application locally on your machine, follow these instructions:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Kwchiam/Intro-AI-LAB-REPORT.git
   cd "Intro-AI-LAB-REPORT/RBS LAB 3"
   ```

2. **Install the required modules:**
   Ensure you have Python installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**

   ```bash
   python -m streamlit run SD24040_Lab3.py
   ```
  
## 🗂️ Project Structure / Modules

- `SD24040_Lab3.py`: The main Streamlit application script containing the UI, JSON parsing, and the priority-sorted rule evaluation engine.
- `requirements.txt`: Contains all necessary Python dependencies (`streamlit`) for deployment.

## 🏷️ Tags

`Artificial Intelligence`, `Knowledge Representation`, `Rule-Based System`, `Expert System`, `Streamlit`, `Python`, `Decision Support`
