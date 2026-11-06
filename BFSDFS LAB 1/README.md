# 🔍 Interactive BFS & DFS Graph Visualizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app/)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An interactive, web-based educational tool designed to visualize **Breadth-First Search (BFS)** and **Depth-First Search (DFS)** graph traversal algorithms step-by-step. Built for **BSD3513 Artificial Intelligence**.

---

## 🚀 Live Demo

Access the deployed application here: **[Live Streamlit App](https://your-app-name.streamlit.app/)**

![App Demo](screenshots/app_demo.png)

---

## ✨ Features

- **Custom Graph Input:** Enter custom directed graph edges or isolated nodes via simple text syntax.
- **Dual Traversal Comparison:** Run BFS and DFS side-by-side on identical graph structures.
- **Step-by-Step Step Trace:** Slide through or auto-play search expansion frames with live node highlighting.
- **Frontier & State Tracking:** Real-time visibility into queue/stack contents, visited nodes, depth, and parent relationships.

---

## 📖 Instructions & Usage

### 1. Defining the Graph
In the sidebar **Graph Editor**, input directed edges line by line:
- `A B` — Creates a directed edge from node **A** to node **B**.
- `C` — Creates an isolated node **C**.
- `# comment` — Lines starting with `#` or empty lines are ignored.

### 2. Running Search
1. Select a **Starting Node** and an optional **Goal Node**.
2. Select your algorithm preference (**BFS**, **DFS**, or **Both**).
3. Use the **Step Inspector slider** or click **▶ Auto-Play** to view node expansion frame-by-frame.

---

## 🛠️ Modules Breakdown

| Module | File | Purpose |
| :--- | :--- | :--- |
| **Parser & Graph** | `modules/graph.py` | Handles text parsing, edge validation, and graph normalization. |
| **Search Engine** | `modules/search.py` | Implements BFS/DFS search algorithms and builds step traces. |
| **Visualizer** | `modules/visualization.py` | Generates NetworkX plots with custom node color states. |
| **App Entrypoint** | `app.py` | Streamlit UI layout, state handling, and widget orchestration. |

---

## 💻 Local Setup & Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/bfs-dfs-visualizer.git](https://github.com/your-username/bfs-dfs-visualizer.git)
cd bfs-dfs-visualizer

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
