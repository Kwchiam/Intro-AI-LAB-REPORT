import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import pandas as pd

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import pandas as pd

# -----------------------------
# Step 1: Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Image Classification with PyTorch & Streamlit",
    page_icon="🖼️",
    layout="centered"
)

st.title("CPU-Based Image Classification Web App")
st.write("Using **PyTorch ResNet-18 (Pre-trained on ImageNet)** + Streamlit")

# -----------------------------
# Step 3: Configure CPU-Only Settings
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# Step 4 & 5: Load Pre-trained Model, Weights & Transforms
# -----------------------------
@st.cache_resource
def load_model_and_transforms():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.to(device)
    model.eval()  # Set model to evaluation mode
    preprocess = weights.transforms()
    labels = weights.meta["categories"]
    return model, preprocess, labels

model, preprocess, labels = load_model_and_transforms()

# -----------------------------
# Step 6: File Uploader UI
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG, JPEG, or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -----------------------------
    # Step 7: Preprocess & Inference (No Gradient)
    # -----------------------------
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(input_batch)

        # -----------------------------
        # Step 8: Softmax & Top-5 Predictions
        # -----------------------------
        probabilities = F.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Prepare results DataFrame
    results_df = pd.DataFrame({
        "Class": [labels[idx] for idx in top5_catid],
        "Probability": [float(p) for p in top5_prob]
    })

    st.subheader("Top-5 Predictions")
    for _, row in results_df.iterrows():
        st.write(f"**{row['Class']}** — {row['Probability']:.4f} ({row['Probability'] * 100:.2f}%)")

    # Display table
    st.subheader("Predictions Table")
    st.dataframe(results_df, use_container_width=True)

    # -----------------------------
    # Step 9: Bar Chart Visualization
    # -----------------------------
    st.subheader("Prediction Probabilities Bar Chart")
    chart_df = results_df.set_index("Class")
    st.bar_chart(chart_df)

else:
    st.info("Please upload an image to start classification.")