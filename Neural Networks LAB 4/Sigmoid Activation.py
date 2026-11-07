import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="Sigmoid Neural Network", layout="wide")

st.title("2. Sigmoid Network")
st.write("Visualizing how the Sigmoid activation function handles linear and non-linear data classification.")

# Sidebar controls for data generation
with st.sidebar:
    st.header("Dataset Generation")
    ds = st.selectbox("Select toy dataset", ["Linear", "Moons (Non-linear)", "Circles (Non-linear)"])
    n_samples = st.slider("Samples", 100, 1000, 300, 50)
    noise = st.slider("Noise", 0.00, 0.40, 0.20, 0.01)
    max_iter = st.slider("Max Epochs", 100, 3000, 800, 100)
    random_state = st.number_input("Random seed", 0, 9999, 42)

def make_linear(n, noise, rs):
    X, y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, class_sep=2.0, random_state=rs)
    if noise > 0:
        rng = np.random.RandomState(rs)
        X += noise * rng.randn(*X.shape)
    return X, y

# Generate Data
if ds == "Linear":
    X, y = make_linear(n_samples, noise, random_state)
elif ds == "Moons (Non-linear)":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
else:
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)

# Model Training (Strictly Sigmoid/Logistic)
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(hidden_layer_sizes=(16, 16),
                          activation="logistic", 
                          learning_rate_init=0.01,
                          max_iter=max_iter,
                          random_state=random_state))
])

clf.fit(X, y)
score = clf.score(X, y)

# Visualization
st.subheader("Mathematical Function")
st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
st.write("The Sigmoid function maps input values to a probability-like range between 0 and 1, providing smooth, non-linear decision boundaries.")

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
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.7)
    ax.scatter(X[:,0], X[:,1], c=y, s=25, edgecolor="k", cmap=ListedColormap(cmap_bold))
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    st.pyplot(fig)

with col2:
    st.subheader("Activation Curve")
    xgrid = np.linspace(-6, 6, 400)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(xgrid, 1 / (1 + np.exp(-xgrid)), label="Sigmoid", color="green", linewidth=2)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.legend()
    st.pyplot(fig2)