import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.fft import fft
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import joblib
import os

# Streamlit page config
st.set_page_config(page_title="NVH Vibration Analysis", page_icon="🔊", layout="wide")

# Title
st.title("🔊 NVH Vibration Analysis Web App")
st.markdown("This app performs vibration signal analysis using machine learning to classify normal vs fault vibrations.")

# Sidebar for controls
st.sidebar.header("Controls")
run_analysis = st.sidebar.button("Run Analysis", help="Click to train the model and display results")

# Main content
if run_analysis or True:  # Always run for demo, or condition on button
    with st.spinner("Generating data and training model..."):
        # Set seed
        np.random.seed(42)
        
        def generate_signal(label, size=300):
            t = np.linspace(0, 1, size)
            if label == 0:
                # Normal vibration
                signal = np.sin(2*np.pi*5*t) + np.random.normal(0, 0.2, size)
            else:
                # Fault vibration
                signal = (
                    np.sin(2*np.pi*5*t) +
                    np.sin(2*np.pi*18*t) +
                    np.random.normal(0, 1.0, size)
                )
            return signal
        
        # Generate dataset
        X, y = [], []
        for _ in range(2000):
            X.append(generate_signal(0))
            y.append(0)
            X.append(generate_signal(1))
            y.append(1)
        X = np.array(X)
        y = np.array(y)
        
        def extract_features(signal):
            fft_vals = np.abs(fft(signal))[:len(signal)//2]
            return [
                np.mean(signal),
                np.std(signal),
                np.max(signal),
                np.min(signal),
                np.sqrt(np.mean(signal**2)),   # RMS
                skew(signal),
                kurtosis(signal),
                np.mean(fft_vals),
                np.max(fft_vals),
                np.sum(fft_vals**2),
                np.argmax(fft_vals)
            ]
        
        X_feat = np.array([extract_features(s) for s in X])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_feat, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Save model
        os.makedirs('model', exist_ok=True)
        joblib.dump(model, 'model/model.pkl')
        
        st.success("Model trained and saved successfully!")
    
    # Display metrics
    st.header("Model Performance")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Precision", f"{precision:.2%}")
    with col3:
        st.metric("Recall", f"{recall:.2%}")
    with col4:
        st.metric("F1 Score", f"{f1:.2%}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # ROC Curve
    st.subheader("ROC Curve")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()
    st.pyplot(fig2)

# Display output images
st.header("Output Images")
outputs_dir = "outputs"
if os.path.exists(outputs_dir):
    images = [f for f in os.listdir(outputs_dir) if f.endswith('.png')]
    if images:
        cols = st.columns(3)  # 3 columns grid
        for i, img in enumerate(images):
            with cols[i % 3]:
                st.image(os.path.join(outputs_dir, img), caption=img, use_column_width=True)
    else:
        st.write("No images found in outputs folder.")
else:
    st.write("Outputs folder not found.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Ready for deployment on Streamlit Cloud")