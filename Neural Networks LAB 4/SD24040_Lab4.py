import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap

# Page Configuration
st.set_page_config(page_title="Neural Network Activations", layout="wide")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Select Application")
    # This radio button acts as the navigation between the 3 "separate" apps
    app_mode = st.radio("Activation Function:", ["1. ReLU", "2. Sigmoid", "3. Tanh"])
    
    st.header("Dataset Generation")
    ds = st.selectbox("Select toy dataset", ["Linear", "Moons (Non-linear)", "Circles (Non-linear)"])
    n_samples = st.slider("Samples", 100, 1000, 300, 50)
    noise = st.slider("Noise", 0.00, 0.40, 0.20, 0.01)
    max_iter = st.slider("Max Epochs", 100, 3000, 800, 100)
    random_state = st.number_input("Random seed", 0, 9999, 42)

# --- Data Generation Function ---
@st.cache_data
def load_data(dataset_name, n, noise_level, rs):
    if dataset_name == "Linear":
        X, y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_informative=2,
                                   n_clusters_per_class=1, class_sep=2.0, random_state=rs)
        if noise_level > 0:
            rng = np.random.RandomState(rs)
            X += noise_level * rng.randn(*X.shape)
        return X, y
    elif dataset_name == "Moons (Non-linear)":
        return make_moons(n_samples=n, noise=noise_level, random_state=rs)
    else:
        return make_circles(n_samples=n, noise=noise_level, factor=0.5, random_state=rs)

X, y = load_data(ds, n_samples, noise, random_state)

# --- Configuration based on Selected App ---
if app_mode == "1. ReLU":
    st.title("1. Rectified Linear Unit (ReLU) Network")
    mlp_activation = "relu"
    math_latex = r"f(x) = \max(0, x)"
    description = "ReLU introduces non-linearity by mapping all negative values to zero, which allows the network to learn complex patterns without vanishing gradients."
    curve_y = lambda x: np.maximum(0, x)
    line_color = "blue"

elif app_mode == "2. Sigmoid":
    st.title("2. Sigmoid Network")
    mlp_activation = "logistic"
    math_latex = r"f(x) = \frac{1}{1 + e^{-x}}"
    description = "The Sigmoid function maps input values to a probability-like range between 0 and 1, providing smooth, non-linear decision boundaries."
    curve_y = lambda x: 1 / (1 + np.exp(-x))
    line_color = "green"

else:
    st.title("3. Hyperbolic Tangent (Tanh) Network")
    mlp_activation = "tanh"
    math_latex = r"f(x) = \tanh(x)"
    description = "The Tanh function is mathematically similar to the sigmoid function but maps the input to values between -1 and 1, ensuring the outputs are zero-centered."
    curve_y = lambda x: np.tanh(x)
    line_color = "red"

# --- Model Training ---
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(hidden_layer_sizes=(16, 16),
                          activation=mlp_activation, 
                          learning_rate_init=0.01,
                          max_iter=max_iter,
                          random_state=random_state))
])

clf.fit(X, y)
score = clf.score(X, y)

# --- Visualization Section ---
st.subheader("Mathematical Function")
st.latex(math_latex)
st.write(description)

# Create two perfectly aligned columns
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Decision Boundary (Accuracy: {score:.3f})")
    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
        np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    cmap_light = ListedColormap(["#FFEEEE", "#EEFFEE"])
    cmap_bold = ["#CC0000", "#007700"]
    
    # Standardized figsize for alignment
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.7)
    ax.scatter(X[:,0], X[:,1], c=y, s=25, edgecolor="k", cmap=ListedColormap(cmap_bold))
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    st.pyplot(fig)

with col2:
    st.subheader("Activation Curve")
    xgrid = np.linspace(-6, 6, 400)
    
    # Standardized figsize for alignment
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(xgrid, curve_y(xgrid), label=app_mode[3:], color=line_color, linewidth=2)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.legend()
    st.pyplot(fig2)