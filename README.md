# 🔊 NVH Vibration Analysis & Fault Detection System (AI + Streamlit)

An end-to-end Machine Learning + Signal Processing + Web Deployment project that detects Normal vs Faulty vibration patterns using advanced feature engineering, FFT analysis, and real-time visualization.

This project simulates a real-world NVH (Noise, Vibration, Harshness) system widely used in automotive engineering, predictive maintenance, and industrial IoT monitoring.

---

## 🚀 Key Features

✔ Real-time model training inside Streamlit  
✔ Synthetic vibration signal generation (Normal vs Fault)  
✔ Time-domain + Frequency-domain feature extraction  
✔ FFT-based spectral analysis  
✔ Machine Learning classification (Random Forest)  
✔ Performance metrics dashboard (Accuracy, Precision, Recall, F1)  
✔ Confusion matrix visualization  
✔ ROC curve analysis with AUC score  
✔ Auto model saving using joblib  
✔ Output image visualization panel  
✔ Fully interactive Streamlit web UI  

---

## 🧠 Problem Statement

Machines generate vibration signals that indicate their health condition.

This system classifies vibration signals into:

✔ Normal Vibration  
⚠ Faulty Vibration  

Using AI-based pattern recognition on extracted signal features.

---

## ⚙️ Tech Stack

Python, Streamlit, NumPy, SciPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Joblib

---

## 📊 System Pipeline

Signal Generation → Feature Extraction → Model Training → Prediction → Evaluation → Visualization → Deployment

---

## 📌 Feature Engineering

### Time Domain Features
- Mean  
- Standard Deviation  
- Maximum / Minimum values  
- RMS (Root Mean Square)  
- Skewness  
- Kurtosis  

### Frequency Domain Features (FFT)
- Mean FFT amplitude  
- Maximum FFT amplitude  
- Spectral energy  
- Dominant frequency index  

---

## 🤖 Machine Learning Model

- Algorithm: Random Forest Classifier  
- Estimators: 200  
- Max Depth: 12  
- Random State: 42  
- Trained on synthetic vibration dataset  

---

## 📈 Model Performance Metrics

The Streamlit dashboard displays:

✔ Accuracy Score  
✔ Precision Score  
✔ Recall Score  
✔ F1 Score  

---

## 📊 Visualizations

✔ Confusion Matrix (Actual vs Predicted)  
✔ ROC Curve (AUC evaluation)  
✔ Feature-based signal analysis (outputs folder images)  

---

## 🌐 Streamlit Web App Features

✔ One-click model training  
✔ Real-time evaluation metrics  
✔ Interactive visualization dashboard  
✔ Sidebar control panel  
✔ Image output viewer  
✔ Clean UI for deployment  

---

## 📁 Project Structure

NVH_Project/
│
├── app.py                 # Streamlit web app
├── model/
│     └── model.pkl       # Trained ML model
├── outputs/              # Visualization images
├── requirements.txt      # Dependencies
└── README.md             # Documentation

---

## 🚀 How to Run Locally

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run Streamlit app
streamlit run app.py

### 3. Open in browser
https://vehicle-vibration-analysis-nvh-fault-detection-system-vvrjefz5.streamlit.app/

---

## 🌍 Deployment Options

✔ Streamlit Cloud  
✔ Render  
✔ HuggingFace Spaces  
✔ Local VS Code / Jupyter  

---

## 🎯 Real-World Applications

- Automotive NVH analysis  
- Predictive maintenance systems  
- Industrial machine fault detection  
- IoT-based vibration monitoring  
- Smart manufacturing systems  

---

## 💡 Project Highlights

✔ End-to-end ML pipeline  
✔ Real-time training + prediction system  
✔ Industrial-grade vibration simulation  
✔ Full evaluation metrics dashboard  
✔ Deployment-ready architecture  
✔ Professional Streamlit UI  

---

## 🔮 Future Enhancements

- Deep Learning (1D CNN for vibration signals)  
- Real sensor (IoT/Arduino) integration  
- Multi-class fault detection  
- Cloud-based monitoring dashboard  
- Live streaming vibration analysis  

---

## 👩‍💻 Author

Vaidehi Sharma  
B.E CSE (AI & ML)  
Machine Learning | Data Science | AI Systems Enthusiast  

---

## ⭐ Support

If you like this project, please give it a ⭐ on GitHub and share it.

---

## 🚀 “Turning vibration signals into intelligent predictions using AI”
